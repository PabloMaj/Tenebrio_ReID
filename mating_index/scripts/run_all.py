"""Run stages 1-5 of the mating-index pipeline end to end.

    python scripts/run_all.py --series seria_2 --max-frames 3     # smoke test
    python scripts/run_all.py                                     # full 4-series run
"""
from _common import base_parser, config_from_args

from mating.stages import (
    Stage1Detect,
    Stage2Bodies,
    Stage3ReID,
    Stage4Confirm,
    Stage5Report,
)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    use_cache = not args.no_cache

    print("=== stage 1: mating-pattern detection ===")
    Stage1Detect(cfg, use_cache=use_cache).run(cfg.series)

    print("=== stage 2: male abdomen + head marker tag ===")
    Stage2Bodies(cfg, use_cache=use_cache).run(cfg.series)

    print("=== stage 3: ensemble re-identification ===")
    Stage3ReID(cfg).run(cfg.series)

    print("=== stage 4: cross-frame association + confirmation ===")
    Stage4Confirm(cfg).run(cfg.series)

    print("=== stage 5: mating index per individual ===")
    out = Stage5Report(cfg).run(cfg.series)

    print(f"\nall outputs under {cfg.output_dir}\nmating index -> {out}")


if __name__ == "__main__":
    main()
