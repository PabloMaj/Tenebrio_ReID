"""Stage 3 - evaluate stage-2 mobility against stage-1 ground truth."""
from _common import base_parser, config_from_args
from mobility.evaluation import MobilityEvaluator


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    gt = cfg.stage1_dir / "mobility_gt.csv"
    pred = cfg.stage2_dir / "mobility_pred_all_splits.csv"
    for p in (gt, pred):
        if not p.exists():
            raise SystemExit(f"missing input: {p} (run stages 1 and 2 first)")
    res = MobilityEvaluator(cfg).evaluate(gt, pred)
    print(f"[stage3] n_gt={res['n_gt']}")
    print(f"[stage3] summary   -> {res['summary']}")
    print(f"[stage3] per-beetle -> {res['per_beetle']}")


if __name__ == "__main__":
    main()
