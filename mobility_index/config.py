"""Central configuration for the beetle-mobility experiment.

Every path, threshold and tunable parameter lives here so the rest of the code
never hard-codes a location.  Import :data:`CONFIG` for the defaults or build a
custom :class:`MobilityConfig` (the CLI scripts do this from argparse).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Repository layout                                                            #
# --------------------------------------------------------------------------- #
# ``mobility_index`` sits inside ``tenebrio_reid_organized``, next to ``datasets/``.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DATASETS_ROOT = REPO_ROOT / "datasets"
MODELS_ROOT = REPO_ROOT / "models"
#: raw frame series (``datasets/raw/<series>/<free|isolated>/*.jpg``).
EXPERIMENT_ROOT = DATASETS_ROOT / "raw"
#: the unified re-ID benchmark package (architecture + trained `proposed` weights)
REID_BENCHMARK_ROOT = Path(
    os.environ.get("TENEBRIO_REID_BENCHMARK_ROOT", REPO_ROOT / "reid_benchmark")
)

# Allow an override when the raw frame archive lives elsewhere (the packaged
# ``datasets/raw`` is a sample; point this at the full archive to reproduce
# every frame).
EXPERIMENT_ROOT = Path(os.environ.get("TENEBRIO_EXPERIMENT_ROOT", EXPERIMENT_ROOT))


@dataclass(frozen=True)
class MobilityConfig:
    """All knobs for the five processing stages."""

    # ---- data -------------------------------------------------------------- #
    experiment_root: Path = EXPERIMENT_ROOT
    #: Series that were part of the re-identification training / validation.
    series: Sequence[str] = ("seria_2", "seria_3", "seria_4", "seria_5")
    #: Phenotyping stage in which the beetles could move freely.
    stage_type: str = "free"

    # ---- models ---------------------------------------------------------- #
    #: head/thorax YOLOv8n-seg checkpoint; bundled in ``models/``
    #: (retrain via ``segmentation_head_thorax/train_seg_model.py`` to replace it).
    seg_weights: Path = MODELS_ROOT / "head_thorax_seg/best.pt"
    #: optional occlusion classifier - NOT bundled (opt-in, off by default via
    #: ``use_occlusion``); train your own with ``occlusion_classification/train_classifier.py``
    #: and place it here, or override via ``with_overrides(occlusion_weights=...)``.
    occlusion_weights: Path = MODELS_ROOT / "occlusion_classifier/best.pt"
    #: 5 cross-validation `proposed` checkpoints, trained by ``reid_benchmark``
    #: (MobileNetV2, ``classifier[1] = Linear(1280, 1024)`` -> 1024-D embedding).
    reid_models_dir: Path = REID_BENCHMARK_ROOT / "runs/thorax/proposed"
    #: which ``reid_benchmark`` method definition to instantiate (its exact architecture).
    reid_method: str = "proposed"
    reid_gallery_dir: Path = (
        DATASETS_ROOT / "re_identification/re_identification_dataset_only_thorax"
    )
    #: HSV/SVM marker-colour reader; bundled in ``models/``
    #: (retrain via ``marker_recognition/train_model_to_sem_seg_markers.py``).
    marker_model: Path = MODELS_ROOT / "marker_svm/model_HSV_SVM_rbf.sav"
    marker_scaler: Path = MODELS_ROOT / "marker_svm/scaler_HSV_SVM_rbf.sav"
    reid_splits: Sequence[int] = (1, 2, 3, 4, 5)

    # ---- segmentation (values taken from phenotyping_pipeline.py) --------- #
    roi_size: int = 512
    roi_overlap: float = 0.25
    seg_conf_threshold: float = 0.7
    frame_rows: int = 3000
    frame_cols: int = 4096
    #: min / max mask area (pixels) accepted per class.
    size_thresholds: dict = field(
        default_factory=lambda: {
            "head": (1000, 2800),
            "thorax": (4000, 7000),
            "down": (5000, 9800),
        }
    )
    #: min head/thorax overlap ratio along the connecting segment to accept a pair.
    merge_metric_threshold: float = 0.8

    # ---- occlusion (OPTIONAL, off by default) --------------------------- #
    #: run the occlusion classifier and drop occluded abdomens before merge.
    use_occlusion: bool = False
    # tuned on free-frame crops: scores cluster ~0.45-0.9, true occlusion is rare
    occlusion_threshold: float = 0.7
    occlusion_norm_mean: tuple = (0.186, 0.219, 0.256)
    occlusion_norm_std: tuple = (0.175, 0.196, 0.199)

    # ---- re-identification --------------------------------------------- #
    reid_input_size: int = 128
    reid_norm_mean: tuple = (0.485, 0.456, 0.406)  # ImageNet (see training script)
    reid_norm_std: tuple = (0.229, 0.224, 0.225)
    reid_top_k: int = 1
    #: "global" -> match against all 80 individuals; "series" -> only same series.
    reid_candidate_scope: str = "global"
    #: ENSEMBLE decision (stage 2): accept an identity only if at least
    #: ``reid_ensemble_min_agree`` of the 5 models predict the same (series, tag)
    #: each with cosine score >= ``reid_ensemble_conf_threshold``.
    reid_ensemble_min_agree: int = 3
    reid_ensemble_conf_threshold: float = 0.6

    # ---- frame sampling ------------------------------------------------ #
    frame_stride: int = 1
    max_frames: int | None = None

    # ---- physical units ---------------------------------------------- #
    seconds_per_frame: float = 5.0      # time between two consecutive frames [s]
    k_mm_per_px: float = 0.0943         # spatial scale [mm / px]

    # ---- aggregation ------------------------------------------------- #
    #: allowed abdomen displacement between *consecutive frames* (px); a step
    #: above this is a re-ID outlier.  The budget scales with the frame gap, so a
    #: detection ``g`` frames after the last kept one may move up to
    #: ``max_step_px * g`` px.
    max_step_px: float = 750.0
    #: an individual needs at least this many kept positions to get a mobility value.
    min_positions_for_mobility: int = 3
    #: stage-3+ filter: drop an individual unless it contributed a kept position
    #: on >= this fraction of the frames processed for its series (0 disables).
    min_detection_frac: float = 0.25
    #: which count feeds the fraction: "n_positions_used" (kept) or "n_positions_raw".
    detection_count_col: str = "n_positions_used"
    #: trajectory smoothing before measuring path length (suppresses per-frame
    #: detection jitter).  "savgol" keeps straight/curved runs, removes noise;
    #: "moving_average"; "none".
    traj_smooth: str = "savgol"
    traj_smooth_window: int = 5
    traj_smooth_polyorder: int = 2

    # ---- visualisation (stage 4) ----------------------------------- #
    stage4_split: int = 1
    stage4_n_paste: int = 8
    stage4_margin_px: int = 120

    # ---- runtime --------------------------------------------------- #
    device: str = field(default_factory=_default_device)
    output_dir: Path = THIS_DIR / "outputs"
    cache_dir: Path = THIS_DIR / "outputs" / "cache"
    seed: int = 42

    # ------------------------------------------------------------------ #
    def with_overrides(self, **kwargs) -> "MobilityConfig":
        """Return a copy with ``kwargs`` replaced (dataclasses.replace wrapper)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    # -- derived paths -------------------------------------------------- #
    @property
    def stage1_dir(self) -> Path:
        return self.output_dir / "stage1"

    @property
    def stage2_dir(self) -> Path:
        return self.output_dir / "stage2"

    @property
    def stage3_dir(self) -> Path:
        return self.output_dir / "stage3"

    @property
    def stage4_dir(self) -> Path:
        return self.output_dir / "stage4"

    @property
    def stage5_dir(self) -> Path:
        return self.output_dir / "stage5"

    def frames_dir(self, series: str) -> Path:
        return self.experiment_root / series / self.stage_type

    def reid_split_weights(self, split: int) -> Path:
        return self.reid_models_dir / f"split_{split}" / "model_best_epoch.pt"

    @property
    def reid_benchmark_dir(self) -> Path:
        return REID_BENCHMARK_ROOT

    #: reproduce the published run by default: drop the manually-reviewed bad
    #: detections bundled at ``manual_exclusions/`` (a compact
    #: ``series;frame_idx;beetle_idx;stage`` CSV - see ``mobility/exclude.py``).
    #: Set False, or pass ``--exclude-dir`` at your own folder/CSV, to review from scratch.
    use_manual_exclusions: bool = True
    #: explicit override; ``None`` -> the bundled list (if ``use_manual_exclusions``)
    #: else ``<output_dir>/exclude``.
    exclude_dir_override: Path | None = None
    #: which stage a bare (unscoped) exclude crop hits - "stage1" | "stage2" | "both".
    #: default "stage1": a mis-read marker tag only corrupts the GT.
    exclude_default_stage: str = "stage1"

    @property
    def exclude_dir(self) -> Path:
        if self.exclude_dir_override:
            return self.exclude_dir_override
        if self.use_manual_exclusions:
            return THIS_DIR / "manual_exclusions"
        return self.output_dir / "exclude"


CONFIG = MobilityConfig()
