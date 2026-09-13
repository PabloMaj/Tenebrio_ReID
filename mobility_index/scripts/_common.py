"""Shared argparse / config plumbing for the stage scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, MobilityConfig  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--series", nargs="+", default=None,
                   help="subset of series, e.g. --series seria_2 seria_3 (default: all 4)")
    p.add_argument("--frame-stride", type=int, default=None,
                   help="process every Nth frame (default 1 = every frame)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap on number of (strided) frames per series")
    p.add_argument("--seconds-per-frame", type=float, default=None,
                   help="time between two consecutive frames [s] (default 5)")
    p.add_argument("--k-mm-per-px", type=float, default=None,
                   help="spatial scale [mm / px] (default 0.0943)")
    p.add_argument("--max-step-px", type=float, default=None,
                   help="allowed abdomen displacement between consecutive frames "
                        "[px] (default 750); scales with the frame gap")
    p.add_argument("--traj-smooth", choices=["savgol", "moving_average", "none"], default=None,
                   help="trajectory smoothing before path-length (default savgol)")
    p.add_argument("--traj-smooth-window", type=int, default=None,
                   help="smoothing window in frames (odd, default 5)")
    p.add_argument("--min-detection-frac", type=float, default=None,
                   help="stage 3+: drop a beetle seen on < this fraction of its "
                        "series' frames (default 0.25; 0 = keep all)")
    p.add_argument("--exclude-dir", type=Path, default=None,
                   help="folder of hand-picked bad crops / a CSV to drop from the "
                        "mobility computation (default: the bundled published-run "
                        "list at manual_exclusions/)")
    p.add_argument("--no-manual-exclusions", action="store_true",
                   help="disable the bundled manually-reviewed exclusion list "
                        "(on by default; reproduces the published run)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="ignore/refresh the per-frame segmentation cache")
    p.add_argument("--occlusion", action="store_true",
                   help="OPTIONAL: run the occlusion classifier and drop occluded "
                        "abdomens (off by default)")
    return p


def config_from_args(args) -> MobilityConfig:
    cfg = CONFIG.with_overrides(
        series=tuple(args.series) if getattr(args, "series", None) else None,
        frame_stride=getattr(args, "frame_stride", None),
        max_frames=getattr(args, "max_frames", None),
        seconds_per_frame=getattr(args, "seconds_per_frame", None),
        k_mm_per_px=getattr(args, "k_mm_per_px", None),
        max_step_px=getattr(args, "max_step_px", None),
        traj_smooth=getattr(args, "traj_smooth", None),
        traj_smooth_window=getattr(args, "traj_smooth_window", None),
        min_detection_frac=getattr(args, "min_detection_frac", None),
        exclude_dir_override=getattr(args, "exclude_dir", None),
        use_manual_exclusions=False if getattr(args, "no_manual_exclusions", False) else None,
        output_dir=getattr(args, "output_dir", None),
        reid_candidate_scope=getattr(args, "candidate_scope", None),
        reid_top_k=getattr(args, "top_k", None),
        stage4_split=getattr(args, "split", None),
        # None -> keep config default; True only when the flag is passed
        use_occlusion=True if getattr(args, "occlusion", False) else None,
        reid_ensemble_min_agree=getattr(args, "reid_min_agree", None),
        reid_ensemble_conf_threshold=getattr(args, "reid_conf_threshold", None),
    )
    if getattr(args, "output_dir", None):
        cfg = cfg.with_overrides(cache_dir=args.output_dir / "cache")
    return cfg
