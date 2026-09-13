"""Stage 4 (example trajectories) and stage 5 (mobility histogram)."""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .aggregation import _pick_positions, smooth_trajectory
from .frame_pipeline import frame_cache_path
from .postfilter import apply_detection_filter, frames_per_series


def _detection_filter(cfg, mob):
    if mob.empty or not float(getattr(cfg, "min_detection_frac", 0.0) or 0.0):
        return mob
    nf = frames_per_series(cfg, mob["series"].unique())
    kept, _ = apply_detection_filter(
        mob, nf, cfg.min_detection_frac,
        getattr(cfg, "detection_count_col", "n_positions_used"),
    )
    return kept


_PATH_COLORS_BGR = [(60, 60, 220), (40, 160, 240), (80, 200, 80)]  # red / orange / green
_LEVELS = ["low mobility", "medium mobility", "high mobility"]


def _load_cached_beetle(cfg, series: str, frame_idx: int, beetle_idx: int):
    p = frame_cache_path(cfg, series, frame_idx)
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        for b in pickle.load(fh):
            if b.beetle_idx == beetle_idx:
                return b
    return None


class TrajectoryVisualizer:
    def __init__(self, config):
        self.cfg = config

    # -- stage 4 ------------------------------------------------------- #
    def render_example_paths(self, split: int | None = None) -> Path:
        cfg = self.cfg
        inv = pd.read_csv(cfg.stage2_dir / "inventory_positions.csv", sep=";")
        inv = inv[inv["accepted"] == 1]
        mob = pd.read_csv(cfg.stage2_dir / "mobility_pred_ensemble.csv", sep=";")
        mob = _detection_filter(cfg, mob)
        mob = mob.sort_values("mobility_mm_per_s").reset_index(drop=True)
        if len(mob) < 3:
            picks = list(mob["key"])
        else:
            picks = [mob.iloc[0]["key"], mob.iloc[len(mob) // 2]["key"], mob.iloc[-1]["key"]]

        panels = []
        for level, key in zip(_LEVELS, picks):
            row = mob[mob["key"] == key].iloc[0]
            panels.append(self._render_one_path(inv, key, row, level))

        h = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
        canvas = np.hstack(panels)
        out = cfg.stage4_dir / "example_paths_ensemble.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)
        return out

    def _render_one_path(self, inv: pd.DataFrame, key: str, mob_row, level: str) -> np.ndarray:
        cfg = self.cfg
        sub = inv[inv["predicted_series"] + "::" + inv["predicted_tag"] == key]
        sub = sub.sort_values("frame_idx")
        frame_to_points: dict[int, list] = {}
        for f, r, c in zip(sub["frame_idx"], sub["y"], sub["x"]):
            frame_to_points.setdefault(int(f), []).append((float(r), float(c)))
        accepted, _, _ = _pick_positions(frame_to_points, cfg.max_step_px)
        frames = [p[0] for p in accepted]
        smoothed = smooth_trajectory(
            [p[1] for p in accepted],
            getattr(cfg, "traj_smooth", "savgol"),
            getattr(cfg, "traj_smooth_window", 5),
            getattr(cfg, "traj_smooth_polyorder", 2),
        )
        pts = np.array([[r, c] for r, c in smoothed])  # row, col (smoothed path drawn)
        if len(pts) < 2:
            return np.zeros((200, 300, 3), np.uint8)

        margin = cfg.stage4_margin_px
        r0, c0 = pts[:, 0].min() - margin, pts[:, 1].min() - margin
        r1, c1 = pts[:, 0].max() + margin, pts[:, 1].max() + margin
        canvas = np.zeros((int(r1 - r0), int(c1 - c0), 3), np.uint8)

        # paste segmented abdomen at a few well-identified positions
        # (ensemble agreement is the per-detection confidence proxy)
        conf_col = "n_agree" if "n_agree" in sub.columns else None
        scores = sub.set_index("frame_idx")[conf_col].to_dict() if conf_col else {}
        idx_order = sorted(
            range(len(accepted)), key=lambda i: scores.get(frames[i], 0), reverse=True
        )
        n_paste = min(cfg.stage4_n_paste, len(accepted))
        chosen = set(np.linspace(0, len(accepted) - 1, n_paste).round().astype(int))
        chosen |= set(idx_order[:2])
        series = sub.iloc[0]["series"]
        for i in sorted(chosen):
            # the detection at this frame closest to the accepted position
            same = sub[sub["frame_idx"] == frames[i]]
            j = ((same["y"] - pts[i, 0]) ** 2 + (same["x"] - pts[i, 1]) ** 2).idxmin()
            b = _load_cached_beetle(cfg, series, int(frames[i]),
                                    int(same.loc[j, "beetle_idx"]))
            if b is None:
                continue
            crop = b.abdomen_crop_bgr
            rr = int(pts[i, 0] - r0 - crop.shape[0] / 2)
            cc = int(pts[i, 1] - c0 - crop.shape[1] / 2)
            self._alpha_paste(canvas, crop, rr, cc)

        # polyline through abdomen centres
        color = _PATH_COLORS_BGR[_LEVELS.index(level)]
        poly = np.array([[int(c - c0), int(r - r0)] for r, c in pts], np.int32)
        cv2.polylines(canvas, [poly], False, color, 2, cv2.LINE_AA)
        for x, y in poly:
            cv2.circle(canvas, (x, y), 3, color, -1)

        label = f"{level}  |  {key}  |  {mob_row['mobility_mm_per_s']:.3f} mm/s"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _alpha_paste(canvas: np.ndarray, crop: np.ndarray, r: int, c: int) -> None:
        H, W = canvas.shape[:2]
        h, w = crop.shape[:2]
        r0, c0 = max(r, 0), max(c, 0)
        r1, c1 = min(r + h, H), min(c + w, W)
        if r1 <= r0 or c1 <= c0:
            return
        sub_crop = crop[r0 - r : r1 - r, c0 - c : c1 - c]
        mask = sub_crop.sum(axis=2) > 0
        canvas[r0:r1, c0:c1][mask] = sub_crop[mask]

    # -- stage 5 ------------------------------------------------------- #
    def render_histogram(self) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cfg = self.cfg
        gt = _detection_filter(cfg, pd.read_csv(cfg.stage1_dir / "mobility_gt.csv", sep=";"))
        pred = _detection_filter(cfg, pd.read_csv(cfg.stage2_dir / "mobility_pred_all_splits.csv", sep=";"))
        pred_mean = pred.groupby("key")["mobility_mm_per_s"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.histogram_bin_edges(
            np.concatenate([gt["mobility_mm_per_s"].values, pred_mean.values]), bins=20
        )
        ax.hist(gt["mobility_mm_per_s"], bins=bins, alpha=0.6, label=f"stage 1 GT (n={len(gt)})")
        ax.hist(pred_mean.values, bins=bins, alpha=0.6, label=f"stage 2 re-ID mean (n={len(pred_mean)})")
        ax.set_xlabel("mobility [mm/s]")
        ax.set_ylabel("number of beetles")
        ax.set_title("Distribution of beetle mobility")
        ax.legend()
        fig.tight_layout()
        out = cfg.stage5_dir / "mobility_histogram.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)

        # per-split variant
        by_split = cfg.stage5_dir / "by_split"
        by_split.mkdir(parents=True, exist_ok=True)
        for split, g in pred.groupby("model_split"):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(g["mobility_mm_per_s"], bins=20, alpha=0.8)
            ax.set_title(f"stage 2 mobility - {split}")
            ax.set_xlabel("mobility [mm/s]")
            fig.tight_layout()
            fig.savefig(by_split / f"split_{split}.png", dpi=130)
            plt.close(fig)
        return out
