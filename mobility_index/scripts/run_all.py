"""Run stages 1-5 end to end.

Example:
    python scripts/run_all.py --series seria_2 --max-frames 5
    python scripts/run_all.py                       # full 4-series run (slow, CPU)
"""
from _common import base_parser, config_from_args
from dump_crops import dump_crops
from mobility.evaluation import MobilityEvaluator
from mobility.stages import Stage1Runner, Stage2Runner
from mobility.visualization import TrajectoryVisualizer


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--candidate-scope", choices=["global", "series"], default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--reid-min-agree", type=int, default=None,
                        help="ensemble: min. models that must agree (default 3)")
    parser.add_argument("--reid-conf-threshold", type=float, default=None,
                        help="ensemble: min. cosine score per vote (default 0.6)")
    parser.add_argument("--split", type=int, default=None, help="(unused, kept for compat)")
    parser.add_argument("--dump-crops", action="store_true",
                        help="also write outputs/crops/ (head + abdomen crops for auditing)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    use_cache = not args.no_cache

    print("=== stage 1: marker ground truth ===")
    Stage1Runner(cfg, use_cache=use_cache).run()

    print("=== stage 2: model re-identification ===")
    Stage2Runner(cfg, use_cache=use_cache).run()

    print("=== stage 3: evaluation ===")
    MobilityEvaluator(cfg).evaluate(
        cfg.stage1_dir / "mobility_gt.csv",
        cfg.stage2_dir / "mobility_pred_all_splits.csv",
    )

    print("=== stage 4: example trajectories ===")
    TrajectoryVisualizer(cfg).render_example_paths()

    print("=== stage 5: histogram ===")
    TrajectoryVisualizer(cfg).render_histogram()

    if args.dump_crops:
        print("=== crop audit dump ===")
        dump_crops(cfg)

    print(f"\nAll outputs under {cfg.output_dir}")


if __name__ == "__main__":
    main()
