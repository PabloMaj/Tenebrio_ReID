"""Stage 1 - marker-based ground-truth mobility.

Example:
    python scripts/run_stage1.py --series seria_2 --max-frames 3
    python scripts/run_stage1.py                       # all 4 series, every frame
"""
from _common import base_parser, config_from_args
from mobility.stages import Stage1Runner


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    print(f"[stage1] series={list(cfg.series)} stride={cfg.frame_stride} "
          f"max_frames={cfg.max_frames} output={cfg.output_dir}")
    res = Stage1Runner(cfg, use_cache=not args.no_cache).run()
    print(f"[stage1] done: {res['n_positions']} positions -> {res['inventory']}")
    print(f"[stage1] mobility -> {res['mobility']}")


if __name__ == "__main__":
    main()
