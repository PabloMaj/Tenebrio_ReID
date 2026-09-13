"""Per-frame core, shared by every stage and cached compactly.

For one ``(series, frame)`` we store:
    * ``matings``  - list[:class:`~mating.geometry.MaskInstance`] (holdout detector)
    * ``heads``    - list[MaskInstance]  (mobility head/thorax segmenter, class "head")
    * ``abdomens`` - list[MaskInstance]  (class "thorax" = the dorsal abdomen)

so stages 2-4 never re-run YOLO.  Each :class:`MaskInstance` keeps only its
bounding box + the mask cropped to it, so a frame pickle stays small.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from mating.geometry import from_full_mask


def frame_cache_path(cfg, series: str, frame_idx: int) -> Path:
    return cfg.cache_dir / "frames" / series / f"{frame_idx:05d}.pkl"


class MatingFramePipeline:
    def __init__(self, cfg, detector, segmenter):
        self.cfg = cfg
        self.detector = detector
        self.segmenter = segmenter

    def process_frame(self, series, frame_idx, img_bgr, use_cache=True) -> dict:
        cache = frame_cache_path(self.cfg, series, frame_idx)
        if use_cache and cache.exists():
            with open(cache, "rb") as fh:
                return pickle.load(fh)

        matings = self.detector.detect(img_bgr, series)
        heads, abdomens = self.segmenter.segment(img_bgr)     # mobility Part lists

        def _pack(parts, cls):
            out = []
            for p in parts:
                inst = from_full_mask(
                    p.binary_mask, conf=float(p.conf), midpoint=list(p.midpoint),
                    cls=cls, series=series,
                )
                if inst is not None:
                    out.append(inst)
            return out

        result = {
            "series": series,
            "frame_idx": int(frame_idx),
            "matings": matings,
            "heads": _pack(heads, "head"),
            "abdomens": _pack(abdomens, "thorax"),
        }
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as fh:
                pickle.dump(result, fh, protocol=4)
        return result


def load_frame_cache(cfg, series, frame_idx):
    p = frame_cache_path(cfg, series, frame_idx)
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def det_id(series: str, frame_idx: int, j: int) -> str:
    """Stable id for the ``j``-th (1-based) mating detection of a frame."""
    return f"{series}_{int(frame_idx):05d}_{j}"
