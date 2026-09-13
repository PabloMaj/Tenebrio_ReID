"""Dump the crops that drove identity + mobility, for visual auditing.

Runs AFTER stages 1 and 2 (reads their CSVs + the per-frame segmentation cache;
no models needed).  Layout under ``outputs/crops/`` mirrors the CSVs - every
filename carries ``<frame>`` and ``b<beetle_idx>`` so it maps 1:1 to a row of
``stage{1,2}/inventory_positions.csv`` (``series`` is the parent folder), and the
folder it sits in is the *label that was assigned*:

    crops/
      stage1_heads_by_tag/  <series>/<tag>/<frame>_b<beetle>.png
          every head crop fed to the marker reader, filed by the tag it got
          (unreadable -> "_unknown")
      stage2_abdomens_by_id/ <series>/<pred_tag>/<frame>_b<beetle>_n<n_agree>.png
          every rotation-normalised abdomen crop fed to the 5-model ensemble,
          filed by the ensemble ID (not accepted -> "_rejected")
      used_for_mobility/
          stage1/<series>/<tag>/<frame>_b<beetle>.png          kept for the GT track
          stage1/_outlier/<series>/<tag>/...                    dropped by the outlier gate
          stage2/<series>/<pred_tag>/<frame>_b<beetle>.png      kept for the ensemble track
          stage2/_outlier/<series>/<pred_tag>/...
      tag_conflicts/ <series>/gt.<tag1>__pred.<tag2>/<frame>_b<beetle>_{head,abdomen}.png
          beetle-frames where the marker tag (stage 1) disagrees with the
          ensemble tag (stage 2) - the prime suspects for a mis-read

Examples:
    python scripts/dump_crops.py
    python scripts/dump_crops.py --sections tag_conflicts --series seria_2
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import pandas as pd

from _common import config_from_args
from mobility.frame_pipeline import frame_cache_path

_SECTIONS = ["stage1_heads_by_tag", "stage2_abdomens_by_id", "used_for_mobility", "tag_conflicts"]


def _tagdir(tag) -> str:
    tag = "" if tag is None else str(tag)
    return {"": "_empty", "nan": "_empty", "unknown": "_unknown"}.get(tag, tag)


def _save(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _read_track(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep=";")
    return {(r.series, int(r.frame_idx), int(r.beetle_idx)): r.status for r in df.itertuples()}


def dump_crops(cfg, sections=None, series=None) -> Path:
    """Write ``<output_dir>/crops/`` from the stage-1/2 CSVs + the frame cache."""
    sections = list(sections) if sections else list(_SECTIONS)
    series_filter = set(series) if series else None
    crops_root = cfg.output_dir / "crops"

    s1 = pd.read_csv(cfg.stage1_dir / "inventory_positions.csv", sep=";")
    s2 = pd.read_csv(cfg.stage2_dir / "inventory_positions.csv", sep=";")
    s1_trk = _read_track(cfg.stage1_dir / "track_positions.csv")
    s2_trk = _read_track(cfg.stage2_dir / "track_positions.csv")

    s1_tag = {(r.series, int(r.frame_idx), int(r.beetle_idx)): r.tag for r in s1.itertuples()}
    s2_id = {
        (r.series, int(r.frame_idx), int(r.beetle_idx)):
        (int(r.accepted), r.predicted_tag, int(r.n_agree))
        for r in s2.itertuples()
    }

    frames = sorted(
        {(s, f) for s, f, _ in list(s1_tag) + list(s2_id)}
        | {(s, f) for (s, f, _b) in list(s1_trk) + list(s2_trk)}
    )
    n_written = 0
    for ser, frame in frames:
        if series_filter and ser not in series_filter:
            continue
        cache = frame_cache_path(cfg, ser, frame)
        if not cache.exists():
            continue
        with open(cache, "rb") as fh:
            beetles = {b.beetle_idx: b for b in pickle.load(fh)}

        for bidx, b in beetles.items():
            k = (ser, frame, bidx)
            # self-identifying name -> a crop stays traceable to (series, frame,
            # beetle) even after it is copied into a flat "exclude/" folder
            fname = f"{ser}_{frame:05d}_b{bidx}.png"

            if "stage1_heads_by_tag" in sections and k in s1_tag:
                _save(b.head_crop_bgr,
                      crops_root / "stage1_heads_by_tag" / ser / _tagdir(s1_tag[k]) / fname)
                n_written += 1

            if "stage2_abdomens_by_id" in sections and k in s2_id:
                acc, ptag, nag = s2_id[k]
                sub = _tagdir(ptag) if acc else "_rejected"
                suffix = f"_n{nag}" if acc else ""
                _save(b.abdomen_crop_norm_bgr,
                      crops_root / "stage2_abdomens_by_id" / ser / sub
                      / f"{ser}_{frame:05d}_b{bidx}{suffix}.png")
                n_written += 1

            if "used_for_mobility" in sections:
                if k in s1_trk and k in s1_tag:
                    st = s1_trk[k]
                    d = crops_root / "used_for_mobility" / "stage1"
                    d = d / ("_outlier" if st == "outlier" else "") / ser / _tagdir(s1_tag[k])
                    _save(b.head_crop_bgr, d / fname)
                    n_written += 1
                if k in s2_trk and k in s2_id:
                    st = s2_trk[k]
                    _, ptag, _ = s2_id[k]
                    d = crops_root / "used_for_mobility" / "stage2"
                    d = d / ("_outlier" if st == "outlier" else "") / ser / _tagdir(ptag)
                    _save(b.abdomen_crop_norm_bgr, d / fname)
                    n_written += 1

            if "tag_conflicts" in sections and k in s1_tag and k in s2_id:
                acc, ptag, _ = s2_id[k]
                gt_tag = str(s1_tag[k])
                if acc and gt_tag not in ("unknown", "nan", "") and str(ptag) != gt_tag:
                    d = (crops_root / "tag_conflicts" / ser
                         / f"gt.{_tagdir(gt_tag)}__pred.{_tagdir(ptag)}")
                    _save(b.head_crop_bgr, d / f"{ser}_{frame:05d}_b{bidx}_head.png")
                    _save(b.abdomen_crop_norm_bgr, d / f"{ser}_{frame:05d}_b{bidx}_abdomen.png")
                    n_written += 2

    print(f"[dump_crops] {n_written} images -> {crops_root}")
    for sec in sections:
        d = crops_root / sec
        if d.exists():
            print(f"  {sec}: {sum(1 for _ in d.rglob('*.png'))} png")
    return crops_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--sections", nargs="+", choices=_SECTIONS, default=_SECTIONS)
    ap.add_argument("--occlusion", action="store_true",
                    help="read the occlusion-filtered frame cache (must match the stage run)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    dump_crops(cfg, sections=args.sections, series=args.series)


if __name__ == "__main__":
    main()
