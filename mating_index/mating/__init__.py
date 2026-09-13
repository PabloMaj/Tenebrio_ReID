"""Beetle mating-index experiment package.

A per-individual **mating index** = the number of *confirmed* mating patterns in
which that beetle took part as the male (the animal mounted on top of the
female).

Pipeline (raw ``free/`` frames of seria_2/3/4/5):

    1. detect the mating pattern as an instance mask (holdout ``yolov8n-seg``)
    2. inside the mating mask, pick the largest abdomen mask that lies >= 75 %
       inside it  -> the male
    3. re-identify that abdomen with the 5 ``proposed`` CV models + ensemble
       (min_agree = 3, cos >= 0.6), exactly as ``mobility_index``
    4. also read the head-marker tag of the same beetle  -> real ID
    5. link mating masks across neighbouring frames (index distance <= 3) by
       mask IoU; a group is *confirmed* when it spans >= 5 frames and the modal
       beetle ID is identical on >= 75 % of the grouped detections
    6. count confirmed matings per individual

Only dependency-free helpers are re-exported here; the model wrappers and stage
runners live in their own modules.
"""
from .geometry import MaskInstance, from_full_mask, frac_inside, mask_iou
from .splits import ALL_SERIES, SYNTHETIC_SERIES, SPLITS, split_for

__all__ = [
    "MaskInstance",
    "from_full_mask",
    "frac_inside",
    "mask_iou",
    "ALL_SERIES",
    "SYNTHETIC_SERIES",
    "SPLITS",
    "split_for",
]
