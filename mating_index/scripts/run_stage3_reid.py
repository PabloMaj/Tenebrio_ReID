"""Stage 3 - re-identify the male abdomen with the 5 ``proposed`` CV models +
ensemble (min_agree 3, cos >= 0.6), exactly as ``mobility_index``.

    python scripts/run_stage3_reid.py --series seria_2
"""
from _common import base_parser, config_from_args

from mating.stages import Stage3ReID


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    out = Stage3ReID(cfg).run(cfg.series)
    print(f"\nstage 3 -> {out}")


if __name__ == "__main__":
    main()
