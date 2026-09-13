"""Central configuration for the beetle mating-index experiment.

Mirrors ``mobility_index/config.py``.  The head/thorax segmentation model, the
HSV/SVM marker reader, the 5 ``proposed`` re-ID splits, the re-ID gallery and
the physical data layout are all **shared with the mobility experiment**:
:pyattr:`MatingConfig.mobility` returns a ``MobilityConfig`` (imported from
``mobility_index``) built from the same values, so the reused
``HeadThoraxSegmenter`` / ``MarkerReader`` / ``ReIDEnsemble`` classes receive
exactly what they expect.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parent          # mating_index/mating
PACKAGE_ROOT = THIS_DIR.parent                       # mating_index
REPO_ROOT = PACKAGE_ROOT.parent                      # tenebrio_reid_organized
DATASETS_ROOT = REPO_ROOT / "datasets"

MOBILITY_ROOT = Path(os.environ.get("TENEBRIO_MOBILITY_ROOT", REPO_ROOT / "mobility_index"))
#: raw frame series (``datasets/raw``); the packaged copy is a sample, override
#: with the full archive to reproduce every frame.
EXPERIMENT_ROOT = Path(
    os.environ.get("TENEBRIO_EXPERIMENT_ROOT", DATASETS_ROOT / "raw")
)
MATING_DATA_ROOT = DATASETS_ROOT / "mating_pattern_detection"


def _default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class MatingConfig:
    # ---- data ----------------------------------------------------------- #
    experiment_root: Path = EXPERIMENT_ROOT
    mating_data_root: Path = MATING_DATA_ROOT
    #: series the mating index is evaluated on (each has its own holdout detector)
    series: Sequence[str] = ("seria_2", "seria_3", "seria_4", "seria_5")
    stage_type: str = "free"

    # ---- mating detector (yolov8n-seg, one holdout model per eval series) - #
    mating_models_dir: Path = PACKAGE_ROOT / "models"
    mating_base_weights: str = "yolov8n-seg.pt"
    mating_conf_threshold: float = 0.25
    mating_iou_merge: float = 0.5          # de-dup the same pattern across overlapping tiles
    tile_size: int = 640                   # matches the 640 training crops
    tile_overlap: float = 0.5              # step = 320 px (as in extract_ROIs.py)

    # ---- head / thorax segmentation + markers + re-ID (reuse mobility) --- #
    reid_splits: Sequence[int] = (1, 2, 3, 4, 5)
    reid_method: str = "proposed"
    reid_candidate_scope: str = "series"   # the beetle is known to be from series X
    reid_ensemble_min_agree: int = 3
    reid_ensemble_conf_threshold: float = 0.6
    reid_top_k: int = 1

    # ---- male-abdomen selection (mask-based) --------------------------- #
    #: an abdomen mask needs >= this fraction of its pixels inside the mating
    #: mask to be a candidate; the largest-area candidate is the male (on top).
    min_frac_in_mating: float = 0.75
    merge_metric_threshold: float = 0.8    # head <-> abdomen join (mobility default)

    # ---- cross-frame confirmation ------------------------------------- #
    assoc_iou_threshold: float = 0.3       # link two mating masks between frames
    max_frame_gap: int = 3                 # "neighbouring" = frame-index distance <= this
    min_group_frames: int = 3             # a confirmed pattern spans >= this many frames
    id_consistency: float = 0.75          # modal beetle ID must cover >= this fraction ...
    id_consistency_base: str = "identified" # ... of "all" grouped detections or "identified" only
    #: absolute floor: >= this many ensemble-accepted detections must carry the modal ID
    min_identified: int = 3

    # ---- frame sampling --------------------------------------------- #
    frame_stride: int = 1
    max_frames: int | None = None

    # ---- runtime -------------------------------------------------- #
    device: str = field(default_factory=_default_device)
    output_dir: Path = PACKAGE_ROOT / "outputs"
    cache_dir: Path = PACKAGE_ROOT / "outputs" / "cache"
    seed: int = 42
    vis_max_frames: int | None = None      # None -> draw an overlay for every processed frame

    # ------------------------------------------------------------------ #
    def with_overrides(self, **kw) -> "MatingConfig":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})

    # -- derived paths ---------------------------------------------- #
    def stage_dir(self, n: int) -> Path:
        return self.output_dir / f"stage{n}"

    def frames_dir(self, series: str) -> Path:
        return self.experiment_root / series / self.stage_type

    def mating_weights(self, series: str) -> Path:
        return self.mating_models_dir / f"holdout_{series}" / "weights" / "best.pt"

    def roi_json(self, series: str) -> Path:
        s = series.replace("_unlabelled", "")
        return DATASETS_ROOT / "additional" / "mating_phenotyping_rois" / f"{s}_A_00001.json"

    @property
    def mobility(self):
        """A ``MobilityConfig`` (from ``mobility_index``) sharing every reused value."""
        from mating._mobility import mobility_config

        return mobility_config(self)


CONFIG = MatingConfig()
