"""Manual sample exclusion.

Drop hand-picked ``(series, frame, beetle)`` detections from the mobility
computation - e.g. crops where the marker tag was clearly mis-read.

Workflow: ``dump_crops.py`` writes crops whose name ends ``<frame>_b<beetle>.png``
and whose path contains ``seria_N``.  Copy the bad ones into ``outputs/exclude/``
(any sub-folders you like - a ``<series>/.../<tag>/`` layout is fine).

Which stage a file affects:
  * path segment ``stage2`` / ``pred`` / ``ensemble`` / ``reid``  -> stage 2 only
  * path segment ``stage1`` / ``gt``                              -> stage 1 only
  * path segment ``both`` / ``all``                               -> both
  * otherwise -> ``default_stage`` (``config.exclude_default_stage``, default
    ``"stage1"`` - a mis-read *marker tag* only corrupts the stage-1 ground truth)

A ``*.csv`` in that folder (``series;frame_idx;beetle_idx[;stage]``) works too.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SERIES_RE = re.compile(r"(seria_\d+)")
_FB_RE = re.compile(r"(\d{1,6})_b(\d{1,4})")


def _parse_path(p: Path):
    """``.../seria_2/.../seria_2_00045_b3_head.png`` -> ("seria_2", 45, 3) | None."""
    text = str(p)
    ser = _SERIES_RE.search(text)
    fb = _FB_RE.search(p.name)
    if not ser or not fb:
        return None
    return ser.group(1), int(fb.group(1)), int(fb.group(2))


_STAGE1_SEG = {"stage1", "gt", "ground_truth"}
_STAGE2_SEG = {"stage2", "pred", "prediction", "ensemble", "reid"}
_BOTH_SEG = {"both", "all"}


def _stage_of(p: Path, default: str = "stage1") -> str:
    parts = {seg.lower() for seg in p.parts}
    if parts & _BOTH_SEG:
        return "both"
    hit1, hit2 = bool(parts & _STAGE1_SEG), bool(parts & _STAGE2_SEG)
    if hit1 and not hit2:
        return "stage1"
    if hit2 and not hit1:
        return "stage2"
    if hit1 and hit2:
        return "both"
    return default


def load_exclusions(exclude_dir, default_stage: str = "stage1") -> dict:
    """Return ``{"stage1": {(series, frame, beetle), ...}, "stage2": {...}}``."""
    out = {"stage1": set(), "stage2": set()}
    d = Path(exclude_dir)
    if not d.exists():
        return out

    def _add(key, stage):
        if stage in ("stage1", "both"):
            out["stage1"].add(key)
        if stage in ("stage2", "both"):
            out["stage2"].add(key)

    unparsed = 0
    for img in list(d.rglob("*.png")) + list(d.rglob("*.jpg")):
        parsed = _parse_path(img)
        if parsed:
            _add(parsed, _stage_of(img, default_stage))
        else:
            unparsed += 1
    if unparsed:
        print(f"[exclude] warning: {unparsed} image(s) under {d} had no parseable "
              f"seria_N / <frame>_b<beetle> - ignored")

    for csv in d.rglob("*.csv"):
        df = pd.read_csv(csv, sep=None, engine="python")
        cols = {c.lower(): c for c in df.columns}
        if not {"series", "frame_idx", "beetle_idx"} <= set(cols):
            continue
        for r in df.itertuples(index=False):
            row = r._asdict()
            key = (str(row[cols["series"]]), int(row[cols["frame_idx"]]), int(row[cols["beetle_idx"]]))
            stage = str(row[cols["stage"]]).lower() if "stage" in cols else default_stage
            _add(key, stage if stage in ("stage1", "stage2", "both") else default_stage)

    return out


def filter_inventory(df: pd.DataFrame, excluded: set,
                     series_col="series", frame_col="frame_idx", beetle_col="beetle_idx"):
    """Return ``(kept_df, dropped_df)`` splitting off the excluded detections."""
    if not excluded or df.empty or beetle_col not in df.columns:
        return df, df.iloc[0:0]
    keys = list(
        zip(df[series_col].astype(str), df[frame_col].astype(int), df[beetle_col].astype(int))
    )
    mask = pd.Series([k in excluded for k in keys], index=df.index)
    return df[~mask], df[mask]
