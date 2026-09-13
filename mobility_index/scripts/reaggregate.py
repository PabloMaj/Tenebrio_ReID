"""Re-run stage-1/2 aggregation from the EXISTING inventories - no segmentation,
no models.

Use it to re-apply the current config to already-computed results: hand-picked
exclusions (``outputs/exclude/``), the outlier gate, trajectory smoothing,
physical units.  It rewrites ``mobility_gt.csv`` / ``mobility_pred_ensemble.csv``
/ ``mobility_pred_all_splits.csv`` / ``track_positions.csv``; then re-run
stages 3-5.

    python scripts/reaggregate.py                 # picks up outputs/exclude/
    python scripts/reaggregate.py --exclude-dir path/to/bad_crops --max-step-px 900
"""
import pandas as pd

from _common import base_parser, config_from_args
from mobility.aggregation import MobilityAggregator
from mobility.exclude import filter_inventory, load_exclusions


def _excl_log(dropped, path):
    if len(dropped):
        dropped.to_csv(path, index=False, sep=";")


def main() -> None:
    cfg = config_from_args(base_parser(__doc__).parse_args())
    excl = load_exclusions(cfg.exclude_dir, getattr(cfg, "exclude_default_stage", "stage1"))
    agg = MobilityAggregator(cfg)

    s1 = pd.read_csv(cfg.stage1_dir / "inventory_positions.csv", sep=";")
    s1k, s1d = filter_inventory(s1, excl["stage1"])
    _excl_log(s1d, cfg.stage1_dir / "excluded_rows.csv")
    mob1 = agg.aggregate(s1k, identity_col="tag",
                         track_out=cfg.stage1_dir / "track_positions.csv")
    mob1.to_csv(cfg.stage1_dir / "mobility_gt.csv", index=False, sep=";")

    s2 = pd.read_csv(cfg.stage2_dir / "inventory_positions.csv", sep=";")
    s2 = s2[s2["accepted"] == 1]
    s2k, s2d = filter_inventory(s2, excl["stage2"])
    _excl_log(s2d, cfg.stage2_dir / "excluded_rows.csv")
    mob2 = agg.aggregate(s2k, identity_col="predicted_tag", series_col="predicted_series",
                         track_out=cfg.stage2_dir / "track_positions.csv")
    mob2.insert(0, "model_split", "ensemble")
    mob2.to_csv(cfg.stage2_dir / "mobility_pred_all_splits.csv", index=False, sep=";")
    mob2.to_csv(cfg.stage2_dir / "mobility_pred_ensemble.csv", index=False, sep=";")

    print(f"[reaggregate] exclude_dir={cfg.exclude_dir}")
    print(f"[reaggregate] stage1: {len(mob1)} tracks  ({len(s1d)} detections excluded)")
    print(f"[reaggregate] stage2: {len(mob2)} tracks  ({len(s2d)} detections excluded)")
    print("[reaggregate] now: python scripts/run_stage3.py ; run_stage4.py ; run_stage5.py")


if __name__ == "__main__":
    main()
