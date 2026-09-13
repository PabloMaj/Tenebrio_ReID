"""Step (2) + (5): from a mating mask, pick the **male** abdomen and read the
head-marker tag of that same beetle.

The male mounts the female, so among the abdomen masks that lie mostly inside
the mating mask the one with the **largest area** is the male (the female's
abdomen is largely hidden underneath).  All selection is done on masks, never
on bounding boxes.
"""
from __future__ import annotations

import numpy as np

from mating._mobility import (
    Part,
    UNKNOWN_TAG,
    mask_bbox,
    merge_heads_and_abdomens,
)
from mating.geometry import MaskInstance, frac_inside


def select_male_abdomen(mating: MaskInstance, abdomens, min_frac: float):
    """Return ``(male_MaskInstance, frac_inside)`` or ``(None, 0.0)``.

    A candidate must have ``>= min_frac`` of its mask inside the mating mask;
    among candidates the largest-area mask wins.
    """
    cands = [(frac_inside(a, mating), a) for a in abdomens]
    cands = [(f, a) for f, a in cands if f >= min_frac]
    if not cands:
        return None, 0.0
    frac, best = max(cands, key=lambda t: t[1].area)
    return best, frac


def _to_part(inst: MaskInstance, cls_name: str, shape) -> Part:
    fm = inst.full_mask(shape)
    mid = inst.meta.get("midpoint") or list(inst.centroid)
    return Part(cls_name, fm, mask_bbox(fm), [int(mid[0]), int(mid[1])],
                float(inst.meta.get("conf", 0.0)), int(inst.area))


def resolve_body(mating: MaskInstance, heads, abdomens, img_bgr, cfg, markers):
    """Resolve the male beetle of one mating mask.

    Returns a dict (abdomen crop for re-ID, marker tag, geometry) or ``None``
    when no abdomen is >= ``cfg.min_frac_in_mating`` inside the mating mask.
    """
    male, frac = select_male_abdomen(mating, abdomens, cfg.min_frac_in_mating)
    if male is None:
        return None

    shape = img_bgr.shape[:2]
    ab_part = _to_part(male, "thorax", shape)
    head_parts = [_to_part(h, "head", shape) for h in heads]

    merged = merge_heads_and_abdomens(
        head_parts, [ab_part],
        metric_threshold=cfg.merge_metric_threshold,
        img_bgr=img_bgr, series=str(mating.meta.get("series", "")), frame_idx=0,
    )
    if merged:
        b = max(merged, key=lambda m: m.merge_metric)
        tag = markers.read_tag(b.head_crop_bgr, b.head_midpoint, b.abdomen_midpoint)
        return {
            "abdomen_crop_bgr": b.abdomen_crop_norm_bgr,
            "abdomen_centroid": tuple(float(v) for v in b.abdomen_midpoint),
            "head_centroid": tuple(float(v) for v in b.head_midpoint),
            "marker_tag": tag,
            "rotation_angle": round(float(b.rotation_angle), 2),
            "merge_metric": round(float(b.merge_metric), 4),
            "frac_in_mating": round(float(frac), 4),
            "abdomen_area": int(male.area),
            "has_head": True,
        }

    # no head merged - re-ID on the un-rotated masked abdomen crop
    r0, c0, r1, c1 = ab_part.box
    m3 = np.dstack([ab_part.binary_mask] * 3).astype("uint8")
    crop = (img_bgr.astype("uint8") * m3)[r0:r1 + 1, c0:c1 + 1]
    return {
        "abdomen_crop_bgr": crop,
        "abdomen_centroid": tuple(float(v) for v in ab_part.midpoint),
        "head_centroid": None,
        "marker_tag": UNKNOWN_TAG,
        "rotation_angle": 0.0,
        "merge_metric": 0.0,
        "frac_in_mating": round(float(frac), 4),
        "abdomen_area": int(male.area),
        "has_head": False,
    }
