"""Stage 4 - render example low / medium / high mobility trajectories."""
from _common import base_parser, config_from_args
from mobility.visualization import TrajectoryVisualizer


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--split", type=int, default=None,
                        help="(unused - stage 2 is now a single ensemble)")
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = TrajectoryVisualizer(cfg).render_example_paths()
    print(f"[stage4] -> {out}")


if __name__ == "__main__":
    main()
