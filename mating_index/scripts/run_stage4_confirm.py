"""Stage 4 - link mating masks across neighbouring frames (IoU) and apply the
confirmation rule (>= 5 frames, ID identical on >= 75 % of the group).

    python scripts/run_stage4_confirm.py --series seria_2
"""
from _common import base_parser, config_from_args

from mating.stages import Stage4Confirm


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    out = Stage4Confirm(cfg).run(cfg.series)
    print(f"\nstage 4 -> {out}")


if __name__ == "__main__":
    main()
