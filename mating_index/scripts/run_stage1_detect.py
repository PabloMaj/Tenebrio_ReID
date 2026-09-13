"""Stage 1 - detect the mating pattern as an instance mask on every raw frame.

    python scripts/run_stage1_detect.py --series seria_2 --max-frames 3
"""
from _common import base_parser, config_from_args

from mating.stages import Stage1Detect


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    out = Stage1Detect(cfg, use_cache=not args.no_cache).run(cfg.series)
    print(f"\nstage 1 -> {out}")


if __name__ == "__main__":
    main()
