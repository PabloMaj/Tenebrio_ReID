"""Stage-3+ post-processing filter.

Keep only individuals that were actually observed often enough for their mobility
value to mean something: a beetle is accepted iff it contributed a kept position
on at least ``min_detection_frac`` of the frames processed for its series
(default 25 % - e.g. >= 50 of 200 frames).

Applied identically to the stage-1 ground truth and the stage-2 ensemble
prediction, so stage 3 compares only beetles that both methods saw enough of.
"""
from __future__ import annotations

import pandas as pd


def frames_per_series(cfg, series_list) -> dict:
    """Number of frames processed per series.

    ``stage{1,2}/frames_processed.csv`` (written by the runners) -> per-frame
    cache-file count -> raw ``*.jpg`` count, whichever is available first.
    """
    out: dict[str, int] = {}
    for csv in (cfg.stage1_dir / "frames_processed.csv", cfg.stage2_dir / "frames_processed.csv"):
        if csv.exists():
            for r in pd.read_csv(csv, sep=";").itertuples():
                out.setdefault(str(r.series), int(r.n_frames))

    for s in map(str, series_list):
        if s in out:
            continue
        cdir = cfg.cache_dir / "frames" / s
        pkls = list(cdir.glob("*.pkl")) if cdir.exists() else []
        if pkls:
            out[s] = len(pkls)
        else:
            fd = cfg.frames_dir(s)
            out[s] = len(list(fd.glob("*.jpg"))) if fd.exists() else 0
    return out


def add_detection_fraction(mob: pd.DataFrame, n_frames: dict,
                           count_col: str = "n_positions_used") -> pd.DataFrame:
    m = mob.copy()
    m["n_frames_series"] = m["series"].map(lambda s: n_frames.get(str(s), 0))
    m["detection_frac"] = [
        round(c / n, 4) if n else 0.0
        for c, n in zip(m[count_col], m["n_frames_series"])
    ]
    return m


def apply_detection_filter(mob: pd.DataFrame, n_frames: dict, min_frac: float,
                           count_col: str = "n_positions_used"):
    """Return ``(kept, dropped)`` DataFrames with ``n_frames_series`` /
    ``detection_frac`` columns added.  ``min_frac <= 0`` keeps everything."""
    m = add_detection_fraction(mob, n_frames, count_col)
    if not min_frac or min_frac <= 0:
        return m.reset_index(drop=True), m.iloc[0:0]
    keep = m["detection_frac"] >= float(min_frac)
    return m[keep].reset_index(drop=True), m[~keep].reset_index(drop=True)
