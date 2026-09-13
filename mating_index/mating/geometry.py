"""Compact mask instances (bounding box + cropped mask) and overlap metrics.

A full 3000x4096 boolean mask is ~12 MB; caching a dozen per frame is not
viable.  :class:`MaskInstance` keeps only the tight bounding box and the mask
cropped to it, and every set operation is evaluated in the overlapping box
region only.

Coordinate convention: boxes are ``(row_min, col_min, row_max, col_max)``,
inclusive on both ends; points are ``[row, col]`` (matching ``mobility_index``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MaskInstance:
    box: tuple                              # (r0, c0, r1, c1) inclusive, full-frame
    mask: np.ndarray = field(repr=False)    # uint8 {0,1}, shape (r1-r0+1, c1-c0+1)
    area: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.area:
            self.area = int(np.count_nonzero(self.mask))

    @property
    def centroid(self) -> tuple:
        r0, c0, _, _ = self.box
        ys, xs = np.nonzero(self.mask)
        if len(ys) == 0:
            return (float(r0), float(c0))
        return (float(r0 + ys.mean()), float(c0 + xs.mean()))

    def full_mask(self, shape) -> np.ndarray:
        r0, c0, r1, c1 = self.box
        out = np.zeros(shape, dtype=np.uint8)
        out[r0:r1 + 1, c0:c1 + 1] = self.mask
        return out


def from_full_mask(full_mask: np.ndarray, **meta) -> "MaskInstance | None":
    ys, xs = np.nonzero(full_mask)
    if len(ys) == 0:
        return None
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    crop = (full_mask[r0:r1 + 1, c0:c1 + 1] > 0).astype(np.uint8)
    return MaskInstance((r0, c0, r1, c1), crop, int(crop.sum()), dict(meta))


def from_box_mask(box: tuple, crop_mask: np.ndarray, **meta) -> MaskInstance:
    return MaskInstance(tuple(int(v) for v in box),
                        (crop_mask > 0).astype(np.uint8), 0, dict(meta))


def _box_overlap(a: tuple, b: tuple):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    r0, c0 = max(ar0, br0), max(ac0, bc0)
    r1, c1 = min(ar1, br1), min(ac1, bc1)
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def intersection_area(a: MaskInstance, b: MaskInstance) -> int:
    ov = _box_overlap(a.box, b.box)
    if ov is None:
        return 0
    r0, c0, r1, c1 = ov
    sa = a.mask[r0 - a.box[0]:r1 - a.box[0] + 1, c0 - a.box[1]:c1 - a.box[1] + 1]
    sb = b.mask[r0 - b.box[0]:r1 - b.box[0] + 1, c0 - b.box[1]:c1 - b.box[1] + 1]
    return int(np.logical_and(sa, sb).sum())


def mask_iou(a: MaskInstance, b: MaskInstance) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def frac_inside(a: MaskInstance, b: MaskInstance) -> float:
    """Fraction of ``a``'s pixels that lie inside ``b`` (0..1)."""
    if a.area == 0:
        return 0.0
    return intersection_area(a, b) / a.area


def bbox_iou(a: tuple, b: tuple) -> float:
    ov = _box_overlap(a, b)
    if ov is None:
        return 0.0
    r0, c0, r1, c1 = ov
    inter = (r1 - r0 + 1) * (c1 - c0 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (area_a + area_b - inter)
