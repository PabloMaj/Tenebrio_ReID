"""Stage 5 - mobility distribution histogram."""
from _common import base_parser, config_from_args
from mobility.visualization import TrajectoryVisualizer


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    out = TrajectoryVisualizer(cfg).render_histogram()
    print(f"[stage5] -> {out}")


if __name__ == "__main__":
    main()
