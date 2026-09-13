"""Stage 2 - abdomen-image re-identification with the 5 ``proposed`` CV models,
combined by a majority-vote ENSEMBLE.

An identity is accepted only if >= --reid-min-agree models agree on the same
(series, tag), each with cosine score >= --reid-conf-threshold.  Every single
model inference is still written to ``reid_inferences.csv``.

Example:
    python scripts/run_stage2.py --series seria_2 --max-frames 3
    python scripts/run_stage2.py --reid-min-agree 3 --reid-conf-threshold 0.6
    python scripts/run_stage2.py --occlusion          # opt-in occlusion filtering
"""
from _common import base_parser, config_from_args
from mobility.stages import Stage2Runner


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--candidate-scope", choices=["global", "series"], default=None,
                        help="match against all 80 individuals (default) or only the same series")
    parser.add_argument("--top-k", type=int, default=None, help="cosine k-NN k (default 1)")
    parser.add_argument("--reid-min-agree", type=int, default=None,
                        help="min. number of the 5 models that must agree (default 3)")
    parser.add_argument("--reid-conf-threshold", type=float, default=None,
                        help="min. cosine score for a model's vote to count (default 0.6)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    print(f"[stage2] series={list(cfg.series)} stride={cfg.frame_stride} "
          f"scope={cfg.reid_candidate_scope} occlusion={cfg.use_occlusion}")
    print(f"[stage2] ensemble: >= {cfg.reid_ensemble_min_agree} models agree, "
          f"score >= {cfg.reid_ensemble_conf_threshold}")
    res = Stage2Runner(cfg, use_cache=not args.no_cache).run()
    print(f"[stage2] {res['n_positions']} beetle-frames, "
          f"{res['n_accepted']} accepted by ensemble ({res['acceptance_rate']:.1%})")
    print(f"[stage2]   verdicts    -> {res['inventory']}")
    print(f"[stage2]   inferences  -> {res['inferences']}")
    print(f"[stage2]   mobility    -> {res['mobility']}")


if __name__ == "__main__":
    main()
