"""Turn a per-frame position inventory into a mobility index per individual.

Mobility = (sum of euclidean displacements of the abdomen centre between
consecutive frames) / (recording time).

* Recording time uses the *observed* span: ``seconds_per_frame * (last_seen -
  first_seen)``.  If the beetle disappears on the last frames, the span (and
  therefore the denominator) shrinks accordingly.
* Outlier gate: the allowed abdomen displacement scales with the frame gap.
  A detection ``g = frame_idx - last_kept_frame_idx`` frames after the last kept
  position may move at most ``max_step_px * g`` px; a larger jump is treated as a
  re-identification error - that frame is dropped and the path stays anchored on
  the previous good position.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .geometry import euclidean


@dataclass
class MobilityResult:
    key: str
    series: str
    identity: str            # tag (stage 1) or predicted tag (stage 2)
    n_positions_raw: int
    n_positions_used: int
    n_outliers: int
    first_idx: int
    last_idx: int
    path_len_px_raw: float    # path length on the picked (un-smoothed) positions
    path_len_px: float        # path length on the smoothed trajectory (-> mobility)
    time_s: float
    mobility_px_per_frame: float
    mobility_px_per_s: float
    mobility_mm_per_s: float


def smooth_trajectory(positions, method: str = "savgol", window: int = 5,
                      polyorder: int = 2):
    """Smooth a sequence of ``(row, col)`` points to suppress per-frame detection
    jitter before measuring path length.

    * ``savgol`` (default) - Savitzky-Golay: fits a local polynomial, so a
      straight or gently curved run is left essentially unchanged while noise is
      removed (no endpoint shrinkage).
    * ``moving_average`` - centred box filter (simpler, slight endpoint pull-in).
    * ``none`` - identity.

    The window is shrunk (kept odd) for short tracks; tracks shorter than the
    minimum usable window are returned unchanged.
    """
    n = len(positions)
    arr = np.asarray(positions, dtype=float)
    if method == "none" or window is None or window <= 1 or n < 3:
        return [tuple(p) for p in arr]

    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return [tuple(p) for p in arr]

    if method == "savgol":
        from scipy.signal import savgol_filter

        po = min(polyorder, w - 1)
        sm = savgol_filter(arr, w, po, axis=0, mode="interp")
    elif method == "moving_average":
        k = w // 2
        sm = np.empty_like(arr)
        for i in range(n):
            sm[i] = arr[max(0, i - k):i + k + 1].mean(axis=0)
    else:
        raise ValueError(f"unknown smoothing method {method!r}")
    return [tuple(p) for p in sm]


def _pick_positions(frame_to_points: dict[int, list[tuple[float, float]]], max_step_px: float):
    """Greedy one-position-per-frame selection with a frame-gap-scaled outlier gate.

    The allowed jump from the last kept position is ``max_step_px * gap`` where
    ``gap`` is the number of frame intervals since that position (1 for
    consecutive frames, larger when frames in between were dropped or the run
    used ``frame_stride``).
    """
    order = sorted(frame_to_points)
    accepted: list[tuple[int, tuple]] = []
    dropped: list[tuple[int, tuple]] = []
    n_outliers = 0
    for f in order:
        cands = frame_to_points[f]                      # each candidate is (row, col, *extra)
        if not accepted:
            accepted.append((f, cands[0]))
            continue
        prev_f, prev_pos = accepted[-1]
        best = min(cands, key=lambda p: euclidean(p, prev_pos))
        gap = max(int(f) - int(prev_f), 1)
        if euclidean(best, prev_pos) > max_step_px * gap:
            n_outliers += 1
            dropped.append((f, best))
            continue
        accepted.append((f, best))
    return accepted, n_outliers, dropped


def compute_mobility_for_track(
    records: list[tuple],
    *,
    key: str,
    series: str,
    identity: str,
    seconds_per_frame: float,
    k_mm_per_px: float,
    max_step_px: float,
    min_positions: int,
    smooth_method: str = "savgol",
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
    return_track: bool = False,
):
    """``records`` = list of ``(frame_idx, row, col[, beetle_idx])`` for one individual.

    With ``return_track`` also returns a list of
    ``(frame_idx, beetle_idx, "kept" | "outlier")`` describing which detections
    fed the mobility number (kept) and which were rejected by the outlier gate.
    """
    frame_to_points: dict[int, list[tuple]] = {}
    for rec in records:
        f, r, c = rec[0], rec[1], rec[2]
        bidx = rec[3] if len(rec) > 3 else -1
        frame_to_points.setdefault(int(f), []).append((float(r), float(c), int(bidx)))

    accepted, n_outliers, dropped = _pick_positions(frame_to_points, max_step_px)
    if len(accepted) < min_positions:
        return (None, []) if return_track else None

    first_idx, last_idx = accepted[0][0], accepted[-1][0]
    span_frames = last_idx - first_idx
    if span_frames <= 0:
        return (None, []) if return_track else None

    raw_pts = [(p[0], p[1]) for _, p in accepted]
    pts = smooth_trajectory(raw_pts, smooth_method, smooth_window, smooth_polyorder)

    path_len_raw = sum(euclidean(raw_pts[i], raw_pts[i - 1]) for i in range(1, len(raw_pts)))
    path_len = sum(euclidean(pts[i], pts[i - 1]) for i in range(1, len(pts)))
    time_s = seconds_per_frame * span_frames
    mob_px_frame = path_len / span_frames
    mob_px_s = path_len / time_s
    res = MobilityResult(
        key=key,
        series=series,
        identity=identity,
        n_positions_raw=len(records),
        n_positions_used=len(accepted),
        n_outliers=n_outliers,
        first_idx=first_idx,
        last_idx=last_idx,
        path_len_px_raw=round(path_len_raw, 3),
        path_len_px=round(path_len, 3),
        time_s=round(time_s, 3),
        mobility_px_per_frame=round(mob_px_frame, 5),
        mobility_px_per_s=round(mob_px_s, 5),
        mobility_mm_per_s=round(mob_px_s * k_mm_per_px, 6),
    )
    if not return_track:
        return res
    track = [(int(f), int(p[2]), "kept") for f, p in accepted]
    track += [(int(f), int(p[2]), "outlier") for f, p in dropped]
    return res, sorted(track)


class MobilityAggregator:
    def __init__(self, config):
        self.cfg = config

    def aggregate(
        self,
        inventory: pd.DataFrame,
        *,
        identity_col: str,
        series_col: str = "series",
        frame_col: str = "frame_idx",
        row_col: str = "y",
        col_col: str = "x",
        beetle_col: str = "beetle_idx",
        drop_values: tuple = ("unknown", "", None),
        track_out=None,
    ) -> pd.DataFrame:
        """Aggregate a position inventory into one mobility row per individual.

        ``identity_col`` holds the tag (stage 1) or predicted tag (stage 2).
        The grouping key is ``<series>::<identity>``.  When ``track_out`` is a
        path, a ``key;series;identity;frame_idx;beetle_idx;status`` CSV of which
        detections fed each mobility value (``kept`` / ``outlier``) is written.
        """
        cols = [
            "key", "series", "identity", "n_positions_raw", "n_positions_used",
            "n_outliers", "first_idx", "last_idx", "path_len_px_raw", "path_len_px",
            "time_s", "mobility_px_per_frame", "mobility_px_per_s", "mobility_mm_per_s",
        ]
        df = inventory.copy()
        if identity_col not in df.columns or df.empty:
            return pd.DataFrame(columns=cols)
        df = df[~df[identity_col].isin(drop_values)]
        df = df.dropna(subset=[identity_col, frame_col, row_col, col_col])
        has_bidx = beetle_col in df.columns

        results = []
        track_rows = []
        for (series, identity), grp in df.groupby([series_col, identity_col]):
            if has_bidx:
                recs = list(zip(grp[frame_col], grp[row_col], grp[col_col], grp[beetle_col]))
            else:
                recs = list(zip(grp[frame_col], grp[row_col], grp[col_col]))
            out = compute_mobility_for_track(
                recs,
                key=f"{series}::{identity}",
                series=str(series),
                identity=str(identity),
                seconds_per_frame=self.cfg.seconds_per_frame,
                k_mm_per_px=self.cfg.k_mm_per_px,
                max_step_px=self.cfg.max_step_px,
                min_positions=self.cfg.min_positions_for_mobility,
                smooth_method=getattr(self.cfg, "traj_smooth", "savgol"),
                smooth_window=getattr(self.cfg, "traj_smooth_window", 5),
                smooth_polyorder=getattr(self.cfg, "traj_smooth_polyorder", 2),
                return_track=track_out is not None,
            )
            res, track = out if track_out is not None else (out, [])
            if res is not None:
                results.append(asdict(res))
                for fr, bidx, status in track:
                    track_rows.append(
                        {"key": f"{series}::{identity}", "series": str(series),
                         "identity": str(identity), "frame_idx": fr,
                         "beetle_idx": bidx, "status": status}
                    )

        if track_out is not None:
            from pathlib import Path

            Path(track_out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                track_rows, columns=["key", "series", "identity", "frame_idx", "beetle_idx", "status"]
            ).to_csv(track_out, index=False, sep=";")
        if not results:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(results).sort_values("key").reset_index(drop=True)
