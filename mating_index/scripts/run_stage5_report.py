"""Stage 5 - mating index per individual + distribution histogram + overlays.

    python scripts/run_stage5_report.py --series seria_2
"""
from _common import base_parser, config_from_args

from mating.stages import Stage5Report


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    out = Stage5Report(cfg).run(cfg.series)
    print(f"\nstage 5 -> {out}")


if __name__ == "__main__":
    main()
