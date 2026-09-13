"""Beetle mobility-index experiment package.

Stages:
    1. marker-based deterministic re-ID  -> ground-truth mobility
    2. abdomen-image model re-ID (5 CV splits) -> predicted mobility
    3. evaluation of stage 2 vs stage 1
    4. example trajectory visualisation
    5. mobility distribution histogram

Only the light-weight, dependency-free helpers are re-exported here.  The model
wrappers and stage runners (``segmentation``, ``occlusion``, ``reid``,
``stages``, ``evaluation``, ``visualization``) pull in torch / ultralytics /
matplotlib and must be imported from their own modules.
"""
from .aggregation import MobilityAggregator, MobilityResult, compute_mobility_for_track
from .geometry import beetle_rotation_angle, euclidean, mask_bbox, mask_centroid
from .inventory import InventoryWriter, read_inventory
from .parts import MergedBeetle, Part, merge_heads_and_abdomens

__all__ = [
    "MobilityAggregator",
    "MobilityResult",
    "compute_mobility_for_track",
    "beetle_rotation_angle",
    "euclidean",
    "mask_bbox",
    "mask_centroid",
    "InventoryWriter",
    "read_inventory",
    "MergedBeetle",
    "Part",
    "merge_heads_and_abdomens",
]
