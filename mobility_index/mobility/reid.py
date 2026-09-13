"""Abdomen-only re-identification with the 5 cross-validation ``proposed`` models.

Architecture + weights + preprocessing all come from the unified re-ID benchmark
(``reid_benchmark/``): ``models.build_model(METHODS["proposed"])`` builds a
``torchvision.mobilenet_v2`` whose ``classifier[1]`` is replaced by a single
``nn.Linear(1280, 1024)`` (a 1024-D embedding), and the input pipeline is the
exact ``reid_benchmark/data.py`` ``BeetleDataset`` transform (BGR ``/255`` ->
resize 128 -> ImageNet normalisation, **cosine** distance).

Weights: ``reid_benchmark/runs/thorax/proposed/split_{1..5}/model_best_epoch.pt``.

Identification = cosine k-NN of the query embedding against a gallery built from
the ``isolated`` abdomen crops of every individual (80 across 4 series).

Ensemble decision (:meth:`ReIDEnsemble.decide`): an identity is *accepted* only
when at least ``min_agree`` of the 5 models predict the same ``(series, tag)``,
each with cosine score ``>= conf_threshold``.  Every individual model inference
is logged regardless of the verdict.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

_GALLERY_GLOB = "*/*/isolated/*thorax*.png"
_RB_CACHE: dict = {}


def _load_reid_benchmark(bench_dir) -> dict:
    """Import ``reid_benchmark`` (models + config + data) in isolation.

    ``reid_benchmark`` ships top-level modules named ``config`` / ``data`` /
    ``models`` that clash with this package's own ``config``; we import them
    with those names temporarily removed from ``sys.modules`` and restore
    everything afterwards.
    """
    key = str(bench_dir)
    if key in _RB_CACHE:
        return _RB_CACHE[key]

    import importlib

    bench_dir = str(Path(bench_dir).resolve())
    if not (Path(bench_dir) / "models.py").exists():
        raise FileNotFoundError(
            f"reid_benchmark not found at {bench_dir} "
            f"(set TENEBRIO_REID_BENCHMARK_ROOT or config.REID_BENCHMARK_ROOT)"
        )
    names = ("data", "config", "models", "losses", "engine", "utils")
    saved_mods = {n: sys.modules.pop(n, None) for n in names}
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, bench_dir)
        data_mod = importlib.import_module("data")
        config_mod = importlib.import_module("config")
        models_mod = importlib.import_module("models")
        out = {
            "METHODS": config_mod.METHODS,
            "build_model": models_mod.build_model,
            "load_checkpoint": models_mod.load_checkpoint,
            "build_transform": data_mod.build_transform,
        }
    finally:
        sys.path[:] = saved_path
        for n in names:
            sys.modules.pop(n, None)
        for n, m in saved_mods.items():
            if m is not None:
                sys.modules[n] = m
    _RB_CACHE[key] = out
    return out


def label_to_series_tag(label: str) -> tuple[str, str]:
    """``"seria_2_A1.g_w"`` -> ``("seria_2", "g_w")``."""
    series = "_".join(label.split("_")[:2])
    folder = label[len(series) + 1 :]
    tag = folder.split(".", 1)[1] if "." in folder else folder
    return series, tag


class ReIDModel:
    """One cross-validation split of the ``proposed`` re-ID model."""

    def __init__(self, weights_path, config, split: int):
        self.cfg = config
        self.split = split
        self.device = torch.device(config.device)

        rb = _load_reid_benchmark(config.reid_benchmark_dir)
        method_cfg = replace(rb["METHODS"][config.reid_method], pretrained=False)
        model = rb["build_model"](method_cfg, device="cpu", num_classes=None)
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        rb["load_checkpoint"](model, state)
        self.model = model.to(self.device).eval()
        self.embedding_dim = int(getattr(model, "embedding_dim", method_cfg.embedding_dim))

        # exact BeetleDataset transform (ToTensor + ConvertImageDtype + Normalize)
        self._transform = rb["build_transform"](config.reid_norm_mean, config.reid_norm_std)
        self.size = config.reid_input_size

        self.gallery_emb: np.ndarray | None = None
        self.gallery_labels: np.ndarray | None = None

    # -- embedding ----------------------------------------------------- #
    def _to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Identical to ``reid_benchmark/data.py`` ``BeetleDataset.__getitem__``."""
        image = crop_bgr.astype("float") / 255.0        # float64, BGR, like training
        image = cv2.resize(image, (self.size, self.size))
        return self._transform(image)                   # -> (3, 128, 128) float32

    @torch.no_grad()
    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([self._to_tensor(c) for c in crops_bgr]).to(self.device, torch.float32)
        emb = self.model(x).cpu().numpy().astype(np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop_bgr])[0]

    # -- gallery ----------------------------------------------------- #
    def build_gallery(self, gallery_dir, cache_path: Path | None, batch_size=256, progress=True):
        if cache_path and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self.gallery_emb, self.gallery_labels = data["emb"], data["labels"]
            return
        paths = sorted(Path(gallery_dir).glob(_GALLERY_GLOB))
        if not paths:
            raise FileNotFoundError(f"no gallery crops under {gallery_dir}")
        labels = [f"{p.parents[2].name}_{p.parents[1].name}" for p in paths]
        embs = []
        it = range(0, len(paths), batch_size)
        if progress:
            from tqdm import tqdm

            it = tqdm(it, desc=f"gallery split {self.split}")
        for start in it:
            chunk = paths[start : start + batch_size]
            crops = [cv2.imread(str(p)) for p in chunk]
            embs.append(self.embed_batch(crops))
        self.gallery_emb = np.concatenate(embs, axis=0)
        self.gallery_labels = np.array(labels)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, emb=self.gallery_emb, labels=self.gallery_labels)

    # -- identification ------------------------------------------- #
    def identify(self, crop_bgr: np.ndarray, top_k: int = 1, allowed_series: str | None = None):
        return self.identify_from_embedding(self.embed(crop_bgr), top_k, allowed_series)

    def identify_from_embedding(self, emb: np.ndarray, top_k: int = 1, allowed_series: str | None = None):
        gal_emb, gal_lab = self.gallery_emb, self.gallery_labels
        if allowed_series is not None:
            m = np.array([lab.startswith(allowed_series + "_") for lab in gal_lab])
            gal_emb, gal_lab = gal_emb[m], gal_lab[m]
        sims = gal_emb @ emb
        order = np.argsort(sims)[::-1][:top_k]
        top_labels = gal_lab[order]
        vals, counts = np.unique(top_labels, return_counts=True)
        best = vals[np.argmax(counts)] if len(vals) < top_k else top_labels[0]
        score = float(sims[order[0]])
        series, tag = label_to_series_tag(str(best))
        return {"label": str(best), "series": series, "tag": tag, "score": score}


class ReIDEnsemble:
    def __init__(self, config):
        self.cfg = config
        self.models: dict[int, ReIDModel] = {}

    def load(self, splits=None, build_gallery=True, progress=True):
        splits = splits or self.cfg.reid_splits
        for s in splits:
            m = ReIDModel(self.cfg.reid_split_weights(s), self.cfg, s)
            if build_gallery:
                m.build_gallery(
                    self.cfg.reid_gallery_dir,
                    self.cfg.cache_dir / "gallery" / f"proposed_split_{s}.npz",
                    progress=progress,
                )
            self.models[s] = m
        return self

    def identify_all(self, crop_bgr: np.ndarray, series: str) -> dict[int, dict]:
        scope = None if self.cfg.reid_candidate_scope == "global" else series
        return {
            s: m.identify(crop_bgr, self.cfg.reid_top_k, scope) for s, m in self.models.items()
        }

    # -- ensemble decision ---------------------------------------- #
    @staticmethod
    def decide(per_model_preds: dict[int, dict], *, min_agree: int = 3,
               conf_threshold: float = 0.6) -> dict:
        """Majority vote over the per-split predictions.

        Accepted iff ``>= min_agree`` splits agree on the same ``(series, tag)``,
        each with ``score >= conf_threshold``.  Returns the verdict plus a
        per-model breakdown (``confident`` / ``voted_for_winner`` flags) so every
        inference that entered the vote can be logged.
        """
        rows, votes = [], Counter()
        for split in sorted(per_model_preds):
            p = per_model_preds[split]
            confident = float(p["score"]) >= conf_threshold
            rows.append(
                {
                    "model_split": split,
                    "predicted_label": p["label"],
                    "predicted_series": p["series"],
                    "predicted_tag": p["tag"],
                    "cos_score": round(float(p["score"]), 4),
                    "confident": bool(confident),
                }
            )
            if confident:
                votes[(p["series"], p["tag"])] += 1

        accepted = False
        win_series = win_tag = win_label = None
        n_agree = 0
        if votes:
            (ws, wt), n_agree = votes.most_common(1)[0]
            if n_agree >= min_agree:
                accepted = True
                win_series, win_tag = ws, wt
                labs = [
                    r["predicted_label"]
                    for r in rows
                    if r["confident"] and r["predicted_series"] == ws and r["predicted_tag"] == wt
                ]
                win_label = Counter(labs).most_common(1)[0][0]

        for r in rows:
            r["voted_for_winner"] = bool(
                accepted
                and r["confident"]
                and r["predicted_series"] == win_series
                and r["predicted_tag"] == win_tag
            )

        return {
            "accepted": accepted,
            "series": win_series,
            "tag": win_tag,
            "label": win_label,
            "n_agree": int(n_agree),
            "n_confident": int(sum(r["confident"] for r in rows)),
            "min_agree": int(min_agree),
            "conf_threshold": float(conf_threshold),
            "per_model": rows,
        }

    def identify_and_decide(self, crop_bgr: np.ndarray, series: str) -> dict:
        preds = self.identify_all(crop_bgr, series)
        return self.decide(
            preds,
            min_agree=self.cfg.reid_ensemble_min_agree,
            conf_threshold=self.cfg.reid_ensemble_conf_threshold,
        )
