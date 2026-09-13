"""Shared argparse / config plumbing for the mating-index stage scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mating.config import CONFIG, MatingConfig  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--series", nargs="+", default=None,
                   help="subset of eval series (default: seria_2 seria_3 seria_4 seria_5)")
    p.add_argument("--frame-stride", type=int, default=None,
                   help="process every Nth frame (default 1)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap on (strided) frames per series - smoke tests")
    p.add_argument("--min-frac-in-mating", type=float, default=None,
                   help="min fraction of an abdomen mask inside the mating mask (default 0.75)")
    p.add_argument("--iou-threshold", type=float, default=None,
                   help="mask IoU to link mating detections between frames (default 0.3)")
    p.add_argument("--max-frame-gap", type=int, default=None,
                   help="max frame-index distance for 'neighbouring' frames (default 3)")
    p.add_argument("--min-group-frames", type=int, default=None,
                   help="a confirmed pattern spans >= this many frames (default 5)")
    p.add_argument("--id-consistency", type=float, default=None,
                   help="modal beetle ID must cover >= this fraction of a group (default 0.75)")
    p.add_argument("--id-consistency-base", choices=["all", "identified"], default=None,
                   help="denominator for --id-consistency (default 'all')")
    p.add_argument("--min-identified", type=int, default=None,
                   help="absolute floor: >= this many ensemble-accepted detections must "
                        "carry the modal ID for a group to confirm (default 3)")
    p.add_argument("--reid-min-agree", type=int, default=None,
                   help="ensemble: min models that must agree (default 3)")
    p.add_argument("--reid-conf-threshold", type=float, default=None,
                   help="ensemble: min cosine score per vote (default 0.6)")
    p.add_argument("--candidate-scope", choices=["global", "series"], default=None,
                   help="re-ID gallery scope (default 'series')")
    p.add_argument("--vis-max-frames", type=int, default=None,
                   help="cap overlay JPGs written per series (default: all)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="ignore / refresh the per-frame detection+segmentation cache")
    return p


def config_from_args(args) -> MatingConfig:
    cfg = CONFIG.with_overrides(
        series=tuple(args.series) if getattr(args, "series", None) else None,
        frame_stride=getattr(args, "frame_stride", None),
        max_frames=getattr(args, "max_frames", None),
        min_frac_in_mating=getattr(args, "min_frac_in_mating", None),
        assoc_iou_threshold=getattr(args, "iou_threshold", None),
        max_frame_gap=getattr(args, "max_frame_gap", None),
        min_group_frames=getattr(args, "min_group_frames", None),
        id_consistency=getattr(args, "id_consistency", None),
        id_consistency_base=getattr(args, "id_consistency_base", None),
        min_identified=getattr(args, "min_identified", None),
        reid_ensemble_min_agree=getattr(args, "reid_min_agree", None),
        reid_ensemble_conf_threshold=getattr(args, "reid_conf_threshold", None),
        reid_candidate_scope=getattr(args, "candidate_scope", None),
        vis_max_frames=getattr(args, "vis_max_frames", None),
        output_dir=getattr(args, "output_dir", None),
    )
    if getattr(args, "output_dir", None):
        cfg = cfg.with_overrides(cache_dir=args.output_dir / "cache")
    return cfg
