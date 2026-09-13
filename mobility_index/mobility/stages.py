"""Stage 1 (marker ground truth) and stage 2 (ensemble model re-ID) runners."""
from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

import pandas as pd

from .aggregation import MobilityAggregator
from .exclude import filter_inventory, load_exclusions
from .frame_pipeline import FramePipeline
from .inventory import InventoryWriter
from .markers import MarkerReader
from .occlusion import OcclusionClassifier
from .reid import ReIDEnsemble
from .segmentation import HeadThoraxSegmenter


def list_frames(cfg, series: str) -> list[tuple[int, Path]]:
    folder = cfg.frames_dir(series)
    frames = sorted(
        (int(p.stem), p) for p in folder.glob("*.jpg") if p.stem.isdigit()
    )
    frames = frames[:: cfg.frame_stride]
    if cfg.max_frames is not None:
        frames = frames[: cfg.max_frames]
    return frames


def _make_occlusion(cfg):
    """Occlusion classifier is OPTIONAL - built only when ``cfg.use_occlusion``."""
    if not getattr(cfg, "use_occlusion", False):
        return None
    return OcclusionClassifier(cfg.occlusion_weights, cfg)


def _write_frames_processed(cfg, stage_dir, series) -> None:
    """Record how many frames each series contributed (denominator for the
    stage-3 detection-fraction filter)."""
    rows = [{"series": s, "n_frames": len(list_frames(cfg, s))} for s in series]
    stage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(stage_dir / "frames_processed.csv", index=False, sep=";")


def _apply_exclusions(cfg, df: pd.DataFrame, stage: str, stage_dir):
    """Drop hand-picked bad detections (``cfg.exclude_dir``) before aggregation."""
    excl = load_exclusions(
        cfg.exclude_dir, getattr(cfg, "exclude_default_stage", "stage1")
    ).get(stage, set())
    kept, dropped = filter_inventory(df, excl)
    if len(dropped):
        stage_dir.mkdir(parents=True, exist_ok=True)
        dropped.to_csv(stage_dir / "excluded_rows.csv", index=False, sep=";")
        print(f"[{stage}] excluded {len(dropped)} hand-picked detections "
              f"({dropped[['series', 'frame_idx', 'beetle_idx']].drop_duplicates().shape[0]} unique)")
    return kept


class Stage1Runner:
    """Deterministic re-identification via head markers -> ground-truth mobility."""

    def __init__(self, cfg, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.segmenter = HeadThoraxSegmenter(cfg.seg_weights, cfg)
        self.occlusion = _make_occlusion(cfg)
        self.markers = MarkerReader(cfg.marker_model, cfg.marker_scaler)
        self.pipeline = FramePipeline(cfg, self.segmenter, self.occlusion)

    def run(self, series=None) -> dict:
        cfg = self.cfg
        series = series or cfg.series
        inv = InventoryWriter(cfg.stage1_dir / "inventory_positions.csv")

        _write_frames_processed(cfg, cfg.stage1_dir, series)
        for s in series:
            frames = list_frames(cfg, s)
            for frame_idx, path in tqdm(frames, desc=f"stage1 {s}"):
                img = cv2.imread(str(path))
                beetles = self.pipeline.process_frame(s, frame_idx, img, use_cache=self.use_cache)
                for b in beetles:
                    tag = self.markers.read_tag(
                        b.head_crop_bgr, b.head_midpoint, b.abdomen_midpoint
                    )
                    inv.add(
                        series=s,
                        frame_idx=frame_idx,
                        beetle_idx=b.beetle_idx,
                        tag=tag,
                        x=int(b.abdomen_midpoint[1]),
                        y=int(b.abdomen_midpoint[0]),
                        rotation_angle=round(b.rotation_angle, 2),
                        occlusion_score=round(b.occlusion_score, 4) if b.occlusion_score is not None else "",
                        merge_metric=round(b.merge_metric, 4),
                    )
        inv.save()

        df = _apply_exclusions(cfg, inv.to_frame(), "stage1", cfg.stage1_dir)
        mob = MobilityAggregator(cfg).aggregate(
            df, identity_col="tag",
            track_out=cfg.stage1_dir / "track_positions.csv",
        )
        mob_path = cfg.stage1_dir / "mobility_gt.csv"
        mob.to_csv(mob_path, index=False, sep=";")
        return {"inventory": inv.path, "mobility": mob_path, "n_positions": len(inv)}


class Stage2Runner:
    """Re-identification from the abdomen image with the 5 ``proposed`` CV models,
    combined by a majority-vote ENSEMBLE."""

    def __init__(self, cfg, build_gallery=True, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.segmenter = HeadThoraxSegmenter(cfg.seg_weights, cfg)
        self.occlusion = _make_occlusion(cfg)
        self.pipeline = FramePipeline(cfg, self.segmenter, self.occlusion)
        self.ensemble = ReIDEnsemble(cfg).load(build_gallery=build_gallery)

    def run(self, series=None) -> dict:
        cfg = self.cfg
        series = series or cfg.series

        # (a) one row per (frame, beetle): the ensemble verdict + abdomen centre
        inv = InventoryWriter(cfg.stage2_dir / "inventory_positions.csv")
        # (b) one row per (frame, beetle, model_split): every inference that
        #     entered the ensemble vote (audit trail)
        infer = InventoryWriter(cfg.stage2_dir / "reid_inferences.csv")

        _write_frames_processed(cfg, cfg.stage2_dir, series)
        for s in series:
            frames = list_frames(cfg, s)
            for frame_idx, path in tqdm(frames, desc=f"stage2 {s}"):
                img = cv2.imread(str(path))
                beetles = self.pipeline.process_frame(s, frame_idx, img, use_cache=self.use_cache)
                for b in beetles:
                    preds = self.ensemble.identify_all(b.abdomen_crop_norm_bgr, s)
                    dec = self.ensemble.decide(
                        preds,
                        min_agree=cfg.reid_ensemble_min_agree,
                        conf_threshold=cfg.reid_ensemble_conf_threshold,
                    )
                    for r in dec["per_model"]:
                        infer.add(series=s, frame_idx=frame_idx, beetle_idx=b.beetle_idx, **r)
                    inv.add(
                        series=s,
                        frame_idx=frame_idx,
                        beetle_idx=b.beetle_idx,
                        accepted=int(dec["accepted"]),
                        predicted_series=dec["series"] or "",
                        predicted_tag=dec["tag"] or "",
                        predicted_label=dec["label"] or "",
                        n_agree=dec["n_agree"],
                        n_confident=dec["n_confident"],
                        x=int(b.abdomen_midpoint[1]),
                        y=int(b.abdomen_midpoint[0]),
                        occlusion_score=round(b.occlusion_score, 4) if b.occlusion_score is not None else "",
                    )
        inv.save()
        infer.save()

        # mobility uses the ENSEMBLE identity: keep only accepted detections
        df = inv.to_frame()
        df = df[df["accepted"] == 1]
        df = _apply_exclusions(cfg, df, "stage2", cfg.stage2_dir)
        mob = MobilityAggregator(cfg).aggregate(
            df, identity_col="predicted_tag", series_col="predicted_series",
            track_out=cfg.stage2_dir / "track_positions.csv",
        )
        mob.insert(0, "model_split", "ensemble")
        mob_path = cfg.stage2_dir / "mobility_pred_all_splits.csv"
        mob.to_csv(mob_path, index=False, sep=";")
        mob.to_csv(cfg.stage2_dir / "mobility_pred_ensemble.csv", index=False, sep=";")

        n_beetle_frames = len(inv)
        n_accepted = int(df.shape[0])
        return {
            "inventory": inv.path,
            "inferences": infer.path,
            "mobility": mob_path,
            "n_positions": n_beetle_frames,
            "n_accepted": n_accepted,
            "acceptance_rate": round(n_accepted / n_beetle_frames, 4) if n_beetle_frames else 0.0,
        }
