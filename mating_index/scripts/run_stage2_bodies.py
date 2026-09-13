"""Stage 2 - pick the male abdomen (mask-based) inside each mating mask and read
its head-marker tag.

    python scripts/run_stage2_bodies.py --series seria_2 --max-frames 3
"""
from _common import base_parser, config_from_args

from mating.stages import Stage2Bodies


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    out = Stage2Bodies(cfg, use_cache=not args.no_cache).run(cfg.series)
    print(f"\nstage 2 -> {out}")


if __name__ == "__main__":
    main()
