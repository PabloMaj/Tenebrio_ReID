"""Per-stage JPG overlays + the mating-index histogram.

Every stage drops a ``vis/`` folder next to its CSV so the whole run can be
audited by eye.  ``cfg.vis_max_frames`` caps how many overlays are written per
series (``None`` = all).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MATING_BGR = (60, 60, 220)      # red
_MALE_BGR = (80, 200, 80)        # green
_HEAD_BGR = (240, 180, 40)       # blue


def _mask_outline(img, inst, color, thick=3):
    m = inst.full_mask(img.shape[:2])
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thick)


def _should_draw(cfg, drawn: int) -> bool:
    return cfg.vis_max_frames is None or drawn < cfg.vis_max_frames


def draw_stage1(cfg, series, frame_idx, img_bgr, matings, drawn: int) -> bool:
    if not _should_draw(cfg, drawn):
        return False
    vis = img_bgr.copy()
    for j, m in enumerate(matings, 1):
        _mask_outline(vis, m, _MATING_BGR)
        r0, c0, r1, c1 = m.box
        cv2.rectangle(vis, (c0, r0), (c1, r1), _MATING_BGR, 2)
        cv2.putText(vis, f"mating {j} ({m.meta.get('conf', 0):.2f})", (c0, max(r0 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _MATING_BGR, 2)
    out = cfg.stage_dir(1) / "vis" / series
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{frame_idx:05d}.jpg"), vis)
    return True


def draw_stage2(cfg, series, frame_idx, img_bgr, items, drawn: int) -> bool:
    """``items``: list of dict(mating, body, det_id)."""
    if not _should_draw(cfg, drawn):
        return False
    vis = img_bgr.copy()
    for it in items:
        _mask_outline(vis, it["mating"], _MATING_BGR, 2)
        body = it["body"]
        if body is None:
            continue
        cy, cx = body["abdomen_centroid"]
        cv2.circle(vis, (int(cx), int(cy)), 26, _MALE_BGR, 4)
        if body.get("head_centroid"):
            hy, hx = body["head_centroid"]
            cv2.circle(vis, (int(hx), int(hy)), 16, _HEAD_BGR, 3)
            cv2.line(vis, (int(hx), int(hy)), (int(cx), int(cy)), _HEAD_BGR, 2)
        cv2.putText(vis, f'{it["det_id"]}  tag={body["marker_tag"]}',
                    (int(cx) - 40, int(cy) - 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, _MALE_BGR, 2)
    out = cfg.stage_dir(2) / "vis" / series
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{frame_idx:05d}.jpg"), vis)
    return True


def mating_index_histogram(cfg, index_df, groups_df, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if not index_df.empty:
        counts = index_df["n_confirmed_matings"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_xlabel("confirmed matings per individual")
    ax.set_ylabel("# individuals")
    ax.set_title("Mating index distribution")

    ax = axes[1]
    if not groups_df.empty:
        conf = groups_df[groups_df["confirmed"] == 1]["n_frames"]
        rej = groups_df[groups_df["confirmed"] == 0]["n_frames"]
        bins = range(1, int(groups_df["n_frames"].max()) + 2)
        ax.hist([rej, conf], bins=bins, stacked=True,
                label=["rejected group", "confirmed group"], color=["#BBBBBB", "#55A868"])
        ax.axvline(cfg.min_group_frames - 0.5, ls="--", c="k", lw=1)
        ax.legend()
    ax.set_xlabel("group span [frames]")
    ax.set_ylabel("# groups")
    ax.set_title("Cross-frame groups")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
