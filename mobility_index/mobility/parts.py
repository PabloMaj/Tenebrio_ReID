"""Data containers for detected body parts and merged beetles, plus the
head <-> abdomen matching logic.

``thorax`` is the segmentation model's name for the big dorsal part (elytra /
abdomen).  The experiment calls this the *odwłok* ("abdomen"); we keep the
model's label ``thorax`` internally and expose ``abdomen_*`` on the public
:class:`MergedBeetle`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import beetle_rotation_angle, mask_bbox, rotation_normalise


@dataclass
class Part:
    """A single head / thorax / down instance in full-frame coordinates."""

    class_name: str
    binary_mask: np.ndarray          # full-frame {0,1}
    box: tuple                       # (row_min, col_min, row_max, col_max)
    midpoint: list                   # [row, col]
    conf: float = 0.0
    area: int = 0
    occ_score: Optional[float] = None


@dataclass
class MergedBeetle:
    """A head paired with a non-occluded abdomen, with small cached crops."""

    series: str
    frame_idx: int
    beetle_idx: int
    abdomen_midpoint: tuple          # (row, col) in the full frame
    head_midpoint: tuple
    rotation_angle: float
    bbox: tuple                      # union bbox (row_min, col_min, row_max, col_max)
    head_crop_bgr: np.ndarray = field(repr=False)
    abdomen_crop_bgr: np.ndarray = field(repr=False)
    abdomen_crop_norm_bgr: np.ndarray = field(repr=False)
    occlusion_score: Optional[float] = None
    merge_metric: float = 0.0


def _segment_coverage(mask_a: np.ndarray, mask_b: np.ndarray, p1, p2) -> float:
    """Fraction of pixels on the segment ``p1``->``p2`` covered by either mask.

    Faithful re-implementation of the metric inside
    ``FrameImg.merge_parts`` but vectorised.
    """
    r0, c0 = p1
    r1, c1 = p2
    n = max(abs(r1 - r0), abs(c1 - c0), 1)
    rr = np.linspace(r0, r1, n + 1).round().astype(int)
    cc = np.linspace(c0, c1, n + 1).round().astype(int)
    rr = np.clip(rr, 0, mask_a.shape[0] - 1)
    cc = np.clip(cc, 0, mask_a.shape[1] - 1)
    covered = (mask_a[rr, cc] > 0) | (mask_b[rr, cc] > 0)
    return float(covered.mean())


def _masked_crop(img_bgr: np.ndarray, mask: np.ndarray, box) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask3 = np.dstack([mask, mask, mask]).astype("uint8")
    return (img_bgr.astype("uint8") * mask3)[r0 : r1 + 1, c0 : c1 + 1, :]


def merge_heads_and_abdomens(
    heads: list[Part],
    abdomens: list[Part],
    *,
    metric_threshold: float,
    img_bgr: np.ndarray,
    series: str,
    frame_idx: int,
) -> list[MergedBeetle]:
    """Pair every head with its best-matching abdomen (>= ``metric_threshold``)."""
    beetles: list[MergedBeetle] = []
    for head in heads:
        if not abdomens:
            break
        metrics = [
            _segment_coverage(head.binary_mask, ab.binary_mask, head.midpoint, ab.midpoint)
            for ab in abdomens
        ]
        best = int(np.argmax(metrics))
        if metrics[best] < metric_threshold:
            continue
        abdomen = abdomens[best]

        hr0, hc0, hr1, hc1 = mask_bbox(head.binary_mask)
        ar0, ac0, ar1, ac1 = mask_bbox(abdomen.binary_mask)
        union = (min(hr0, ar0), min(hc0, ac0), max(hr1, ar1), max(hc1, ac1))

        angle = beetle_rotation_angle(head.midpoint, abdomen.midpoint)
        abdomen_crop = _masked_crop(img_bgr, abdomen.binary_mask, (ar0, ac0, ar1, ac1))
        beetles.append(
            MergedBeetle(
                series=series,
                frame_idx=frame_idx,
                beetle_idx=len(beetles) + 1,
                abdomen_midpoint=tuple(abdomen.midpoint),
                head_midpoint=tuple(head.midpoint),
                rotation_angle=angle,
                bbox=union,
                head_crop_bgr=_masked_crop(img_bgr, head.binary_mask, (hr0, hc0, hr1, hc1)),
                abdomen_crop_bgr=abdomen_crop,
                abdomen_crop_norm_bgr=rotation_normalise(abdomen_crop, angle),
                occlusion_score=abdomen.occ_score,
                merge_metric=metrics[best],
            )
        )
    return beetles
