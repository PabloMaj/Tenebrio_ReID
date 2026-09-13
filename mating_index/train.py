"""Train the mating-pattern detector - **YOLOv8n-seg**, one *holdout* model per
evaluation series (seria_2 / 3 / 4 / 5).

Independence (user requirement): the model used to compute the mating index for
series ``X`` must have seen **no** ``X`` sample in training or validation, so for
``holdout_X`` the series ``X`` appears only as the ``test`` split.  Every other
series - real *and* synthetic - is routed to train / val by
``mating.splits.SPLITS`` (an extension of
``mating_pattern_detection/create_datasets_for_splits.py`` to seria_5).

Real vs synthetic: the **training** set is balanced 1:1 over *all* images -
after every real training image is placed, an equal number of synthetic images
is sampled (with replacement only if the pool is too small).  Validation keeps
whatever its val series provide (real + synthetic, no forced ratio; disable the
synthetic half with ``--no-val-synthetic``).  The test split is **real only**.

Live metrics: Ultralytics prints per-epoch precision / recall / mAP and writes
``results.csv`` + PR/F1/confusion curves into
``models/holdout_<series>/``; after training the detector is evaluated on the
held-out series (``model.val(split="test")``) into ``test_metrics.json``.

    python mating_index/train.py --holdout-series all
    python mating_index/train.py --holdout-series seria_2 --epochs 200 --batch 8
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mating.splits import EVAL_SERIES, SYNTHETIC_SERIES, check_independence, split_for  # noqa: E402

PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PKG_ROOT.parent / "datasets" / "mating_pattern_detection"


# --------------------------------------------------------------------------- #
# dataset assembly                                                             #
# --------------------------------------------------------------------------- #
def _real_sources(data_root: Path, series: str) -> list[tuple[Path, str]]:
    """``(jpg_path, kind)`` for a real series; kind in {with_mating, without_mating}."""
    out = []
    for kind in ("with_mating", "without_mating"):
        d = data_root / "ROIs" / series / kind
        if d.is_dir():
            out += [(p, kind) for p in sorted(d.glob("*.jpg"))]
    return out


def _synth_sources(data_root: Path, series: str) -> list[Path]:
    d = data_root / "synthetic_data" / series
    return sorted(d.glob("*.jpg")) if d.is_dir() else []


def _copy_pair(jpg: Path, label_src: Path | None, dst_img: Path, dst_lbl: Path) -> None:
    shutil.copy(jpg, dst_img)
    if label_src and label_src.exists():
        shutil.copy(label_src, dst_lbl)
    else:                                   # background image -> empty label
        dst_lbl.write_text("")


def build_dataset(holdout: str, data_root: Path, out_root: Path, *,
                  seed: int, val_synthetic: bool) -> dict:
    check_independence(holdout)
    sp = split_for(holdout)
    ds = out_root / f"holdout_{holdout}"
    if ds.exists():
        shutil.rmtree(ds)
    for sub in ("train", "val", "test"):
        for f in ("images", "labels"):
            (ds / sub / f).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    inv_rows: list[dict] = []
    n_real_train = 0

    # ---- real images (all subsets) + synthetic for train/val ------------- #
    for sub in ("train", "val", "test"):
        for series in sp[sub]:
            reals = _real_sources(data_root, series)
            for jpg, kind in tqdm(reals, desc=f"{holdout} {sub}/{series} real", leave=False):
                stem = f"{series}__{jpg.stem}"
                lbl = jpg.with_suffix(".txt") if kind == "with_mating" else None
                _copy_pair(jpg, lbl, ds / sub / "images" / f"{stem}.jpg",
                           ds / sub / "labels" / f"{stem}.txt")
                inv_rows.append({"holdout": holdout, "subset": sub, "series": series,
                                 "source": "real", "kind": kind, "file": f"{stem}.jpg"})
                if sub == "train":
                    n_real_train += 1

            # train synthetic is added in the 1:1 balancing pass below;
            # val synthetic (by series routing) is copied here
            if series in SYNTHETIC_SERIES and sub == "val" and val_synthetic:
                for jpg in tqdm(_synth_sources(data_root, series),
                                desc=f"{holdout} val/{series} synth", leave=False):
                    stem = f"{series}__{jpg.stem}"
                    _copy_pair(jpg, jpg.with_suffix(".txt"),
                               ds / sub / "images" / f"{stem}.jpg",
                               ds / sub / "labels" / f"{stem}.txt")
                    inv_rows.append({"holdout": holdout, "subset": sub, "series": series,
                                     "source": "synthetic", "kind": "with_mating",
                                     "file": f"{stem}.jpg"})

    # ---- 1:1 real:synthetic in TRAIN ----------------------------------- #
    pool = [(series, p) for series in sp["train"] if series in SYNTHETIC_SERIES
            for p in _synth_sources(data_root, series)]
    rng.shuffle(pool)
    if not pool:
        picks = []
    elif len(pool) >= n_real_train:
        picks = pool[:n_real_train]
    else:
        picks = [pool[i % len(pool)] for i in range(n_real_train)]   # oversample
    seen: dict = {}
    for series, jpg in tqdm(picks, desc=f"{holdout} train synth 1:1", leave=False):
        seen[jpg] = seen.get(jpg, 0) + 1
        suffix = "" if seen[jpg] == 1 else f"__r{seen[jpg]}"
        stem = f"{series}__{jpg.stem}{suffix}"
        _copy_pair(jpg, jpg.with_suffix(".txt"),
                   ds / "train" / "images" / f"{stem}.jpg",
                   ds / "train" / "labels" / f"{stem}.txt")
        inv_rows.append({"holdout": holdout, "subset": "train", "series": series,
                         "source": "synthetic", "kind": "with_mating", "file": f"{stem}.jpg"})

    # ---- data.yaml + inventory ---------------------------------------- #
    (ds / "data.yaml").write_text(
        f"path: {ds.as_posix()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n\n"
        'names: ["mating"]\n'
    )
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(ds / "inventory.csv", index=False, sep=";")
    summary = (
        inv.groupby(["subset", "source"]).size().unstack(fill_value=0)
        if not inv.empty else pd.DataFrame()
    )
    summary.to_csv(ds / "inventory_summary.csv", sep=";")
    print(f"\n[{holdout}] dataset @ {ds}")
    print(f"  train series: {sp['train']}")
    print(f"  val series:   {sp['val']}")
    print(f"  test series:  {sp['test']}")
    print(summary.to_string() if not summary.empty else "  (empty)")
    return {"dataset": ds, "data_yaml": ds / "data.yaml", "split": sp}


# --------------------------------------------------------------------------- #
# training                                                                     #
# --------------------------------------------------------------------------- #
def train_one(holdout: str, data_yaml: Path, out_models: Path, args) -> None:
    from ultralytics import YOLO

    model = YOLO(f"{args.model}.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        workers=args.workers,
        project=str(out_models),
        name=f"holdout_{holdout}",
        exist_ok=True,
        plots=True,
        # geometry-preserving (orientation of the pair carries the male/female cue)
        mosaic=0.0, mixup=0.0, degrees=0.0, translate=0.0, scale=0.0,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
    )

    run_dir = out_models / f"holdout_{holdout}"
    best = run_dir / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device,
        split="test", save_json=True, project=str(out_models),
        name=f"holdout_{holdout}_test", exist_ok=True,
    )
    md = {k: float(v) for k, v in getattr(metrics, "results_dict", {}).items()}
    p, r = md.get("metrics/precision(B)", 0.0), md.get("metrics/recall(B)", 0.0)
    md["metrics/F1(B)"] = 2 * p * r / (p + r) if (p + r) else 0.0
    (run_dir / "test_metrics.json").write_text(json.dumps(md, indent=2))
    print(f"[{holdout}] test (held-out series) metrics:")
    for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/F1(B)",
              "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if k in md:
            print(f"    {k:28s} {md[k]:.4f}")

    res_csv = run_dir / "results.csv"
    if res_csv.exists():
        last = pd.read_csv(res_csv).iloc[-1]
        print(f"[{holdout}] last training epoch: "
              + ", ".join(f"{c.strip()}={last[c]:.4f}" for c in last.index
                          if "metrics/" in c and pd.notna(last[c])))


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-series", nargs="+", default=["all"],
                    help=f"one or more of {EVAL_SERIES}, or 'all' (default: all)")
    ap.add_argument("--model", default="yolov8n-seg", help="base model (default yolov8n-seg)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-val-synthetic", action="store_true",
                    help="keep the validation set real-only")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="mating_pattern_detection/ (ROIs/, synthetic_data/)")
    ap.add_argument("--datasets-dir", type=Path, default=PKG_ROOT / "datasets")
    ap.add_argument("--models-dir", type=Path, default=PKG_ROOT / "models")
    ap.add_argument("--build-only", action="store_true", help="assemble datasets, skip training")
    args = ap.parse_args(argv)

    try:
        import torch

        if str(args.device) not in ("cpu",) and not torch.cuda.is_available():
            print("[warn] CUDA not available - falling back to --device cpu")
            args.device = "cpu"
    except Exception:
        pass

    holdouts = EVAL_SERIES if "all" in args.holdout_series else list(args.holdout_series)
    unknown = [h for h in holdouts if h not in EVAL_SERIES]
    if unknown:
        ap.error(f"unknown holdout series {unknown}; expected a subset of {EVAL_SERIES}")
    for h in holdouts:
        info = build_dataset(h, args.data_root, args.datasets_dir,
                             seed=args.seed, val_synthetic=not args.no_val_synthetic)
        if not args.build_only:
            train_one(h, info["data_yaml"], args.models_dir, args)


if __name__ == "__main__":
    main()
