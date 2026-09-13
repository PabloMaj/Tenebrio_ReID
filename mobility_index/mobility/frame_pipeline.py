"""Shared per-frame core for stage 1 and stage 2.

Steps (1)-(3) from the task description:
    (1) instance segmentation of head + abdomen
    (2) classify and reject occluded abdomens
    (3) merge heads with the surviving abdomens

The result (a list of :class:`MergedBeetle` with small cached crops) is pickled
per ``(series, frame)`` so stage 2 can re-run the 5 re-ID models without
re-segmenting.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .parts import MergedBeetle, merge_heads_and_abdomens
from .geometry import mask_bbox


def frame_cache_path(cfg, series: str, frame_idx: int) -> Path:
    """Per-frame beetle-list cache path.

    The explicit ``_occ`` / ``_noocc`` suffix keeps the occlusion-filtered and
    the unfiltered caches apart (and invalidates any pre-rework unsuffixed cache).
    """
    suffix = "_occ" if getattr(cfg, "use_occlusion", False) else "_noocc"
    return cfg.cache_dir / "frames" / series / f"{frame_idx:05d}{suffix}.pkl"


class FramePipeline:
    def __init__(self, config, segmenter, occlusion):
        self.cfg = config
        self.segmenter = segmenter
        self.occlusion = occlusion

    def _cache_path(self, series: str, frame_idx: int) -> Path:
        return frame_cache_path(self.cfg, series, frame_idx)

    def process_frame(
        self, series: str, frame_idx: int, img_bgr: np.ndarray, use_cache: bool = True
    ) -> list[MergedBeetle]:
        cache = self._cache_path(series, frame_idx)
        if use_cache and cache.exists():
            with open(cache, "rb") as fh:
                return pickle.load(fh)

        heads, abdomens = self.segmenter.segment(img_bgr)

        # (2) occlusion filtering on abdomen crops
        kept = []
        if abdomens and self.occlusion is not None:
            crops = []
            for ab in abdomens:
                r0, c0, r1, c1 = mask_bbox(ab.binary_mask)
                m3 = np.dstack([ab.binary_mask] * 3).astype("uint8")
                crops.append((img_bgr.astype("uint8") * m3)[r0 : r1 + 1, c0 : c1 + 1])
            for ab, sc in zip(abdomens, self.occlusion.score_batch(crops)):
                ab.occ_score = float(sc)
                if sc <= self.cfg.occlusion_threshold:
                    kept.append(ab)
        else:
            kept = abdomens

        # (3) merge
        beetles = merge_heads_and_abdomens(
            heads,
            kept,
            metric_threshold=self.cfg.merge_metric_threshold,
            img_bgr=img_bgr,
            series=series,
            frame_idx=frame_idx,
        )

        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as fh:
                pickle.dump(beetles, fh, protocol=4)
        return beetles
