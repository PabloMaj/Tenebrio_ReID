"""Bridge to the ``mobility_index`` package - the single source of truth for the
head/thorax segmenter, the HSV/SVM marker reader and the 5-model ``proposed``
re-ID ensemble.

``mobility_index`` ships a top-level ``config.py``; we import only the *package*
``mobility.*`` (which never imports that top-level module) and keep
``mobility_index`` on ``sys.path`` so ``mobility.reid`` can, in turn, locate
``reid_benchmark``.
"""
from __future__ import annotations

import sys

from mating.config import MOBILITY_ROOT

if not (MOBILITY_ROOT / "mobility" / "__init__.py").exists():
    raise FileNotFoundError(
        f"mobility_index not found at {MOBILITY_ROOT} - set TENEBRIO_MOBILITY_ROOT"
    )
if str(MOBILITY_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILITY_ROOT))

from mobility.geometry import (  # noqa: E402
    beetle_rotation_angle,
    euclidean,
    mask_bbox,
    rotation_normalise,
)
from mobility.inventory import InventoryWriter, read_inventory  # noqa: E402
from mobility.markers import UNKNOWN_TAG, MarkerReader  # noqa: E402
from mobility.parts import MergedBeetle, Part, merge_heads_and_abdomens  # noqa: E402
from mobility.reid import ReIDEnsemble, label_to_series_tag  # noqa: E402
from mobility.segmentation import HeadThoraxSegmenter  # noqa: E402

__all__ = [
    "HeadThoraxSegmenter",
    "MarkerReader",
    "UNKNOWN_TAG",
    "ReIDEnsemble",
    "label_to_series_tag",
    "Part",
    "MergedBeetle",
    "merge_heads_and_abdomens",
    "mask_bbox",
    "beetle_rotation_angle",
    "rotation_normalise",
    "euclidean",
    "InventoryWriter",
    "read_inventory",
    "mobility_config",
]


def mobility_config(mcfg):
    """Build a ``MobilityConfig`` from a :class:`~mating.config.MatingConfig`.

    Only the values the reused classes read are forwarded; ``MobilityConfig``
    derives ``seg_weights`` / ``marker_model`` / ``reid_models_dir`` /
    ``reid_gallery_dir`` from its own module constants (relative to
    ``mobility_index/``), which is exactly what we want.
    """
    from config import MobilityConfig  # mobility_index/config.py

    return MobilityConfig(
        experiment_root=mcfg.experiment_root,
        series=tuple(mcfg.series),
        stage_type=mcfg.stage_type,
        reid_splits=tuple(mcfg.reid_splits),
        reid_method=mcfg.reid_method,
        reid_candidate_scope=mcfg.reid_candidate_scope,
        reid_ensemble_min_agree=mcfg.reid_ensemble_min_agree,
        reid_ensemble_conf_threshold=mcfg.reid_ensemble_conf_threshold,
        reid_top_k=mcfg.reid_top_k,
        merge_metric_threshold=mcfg.merge_metric_threshold,
        use_occlusion=False,
        device=("cuda" if str(mcfg.device) in ("0", "cuda") else "cpu"),
        output_dir=mcfg.output_dir,
        cache_dir=mcfg.cache_dir,
        seed=mcfg.seed,
    )
