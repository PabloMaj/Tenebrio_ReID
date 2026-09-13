"""Offline sweep of the stage-2 ensemble decision.

``reid_inferences.csv`` logs all 5 raw model predictions for every beetle-frame,
so the ensemble verdict can be recomputed for any ``min_agree`` /
``conf_threshold`` **without re-running segmentation or the re-ID models**.  For
each requested combination this recomputes the accepted identities, re-aggregates
mobility and scores it against the stage-1 ground truth.

Run stage 1 + stage 2 once (any ``--reid-min-agree`` - it does not affect the
logged inferences), then:

    python scripts/run_reid_sweep.py
    python scripts/run_reid_sweep.py --min-agree 3 4 5 --conf-threshold 0.5 0.6 0.7
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from _common import config_from_args
from mobility.aggregation import MobilityAggregator
from mobility.evaluation import _error_metrics
from mobility.reid import ReIDEnsemble


def _ensemble_mobility(inferences: pd.DataFrame, positions: pd.DataFrame, cfg,
                       min_agree: int, conf_threshold: float) -> pd.DataFrame:
    pos_ix = positions.set_index(["series", "frame_idx", "beetle_idx"])
    rows = []
    for (s, f, b), grp in inferences.groupby(["series", "frame_idx", "beetle_idx"]):
        preds = {
            int(r.model_split): {
                "label": r.predicted_label,
                "series": r.predicted_series,
                "tag": r.predicted_tag,
                "score": float(r.cos_score),
            }
            for r in grp.itertuples()
        }
        dec = ReIDEnsemble.decide(preds, min_agree=min_agree, conf_threshold=conf_threshold)
        if not dec["accepted"]:
            continue
        try:
            p = pos_ix.loc[(s, f, b)]
        except KeyError:
            continue
        rows.append(
            {
                "series": s,
                "frame_idx": f,
                "predicted_series": dec["series"],
                "predicted_tag": dec["tag"],
                "x": int(p["x"]),
                "y": int(p["y"]),
            }
        )
    df = pd.DataFrame(rows)
    return MobilityAggregator(cfg).aggregate(
        df, identity_col="predicted_tag", series_col="predicted_series"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=__import__("pathlib").Path, default=None)
    ap.add_argument("--min-agree", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--conf-threshold", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    # keep the physical-unit / smoothing / outlier knobs consistent with the run
    ap.add_argument("--seconds-per-frame", type=float, default=None)
    ap.add_argument("--k-mm-per-px", type=float, default=None)
    ap.add_argument("--max-step-px", type=float, default=None)
    ap.add_argument("--traj-smooth", choices=["savgol", "moving_average", "none"], default=None)
    ap.add_argument("--traj-smooth-window", type=int, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)

    inferences = pd.read_csv(cfg.stage2_dir / "reid_inferences.csv", sep=";")
    positions = pd.read_csv(cfg.stage2_dir / "inventory_positions.csv", sep=";")
    gt = pd.read_csv(cfg.stage1_dir / "mobility_gt.csv", sep=";")
    n_gt = gt["key"].nunique()
    n_bf = positions[["series", "frame_idx", "beetle_idx"]].drop_duplicates().shape[0]

    out_rows = []
    for ma, ct in itertools.product(sorted(args.min_agree), sorted(args.conf_threshold)):
        pred = _ensemble_mobility(inferences, positions, cfg, ma, ct)
        merged = gt.merge(pred, on="key", suffixes=("_gt", "_pred"), how="inner")
        m = _error_metrics(
            merged["mobility_mm_per_s_gt"].values, merged["mobility_mm_per_s_pred"].values
        )
        # acceptance rate over beetle-frames
        acc = 0
        for _, grp in inferences.groupby(["series", "frame_idx", "beetle_idx"]):
            preds = {
                int(r.model_split): {"label": r.predicted_label, "series": r.predicted_series,
                                     "tag": r.predicted_tag, "score": float(r.cos_score)}
                for r in grp.itertuples()
            }
            acc += ReIDEnsemble.decide(preds, min_agree=ma, conf_threshold=ct)["accepted"]
        out_rows.append(
            {
                "min_agree": ma,
                "conf_threshold": ct,
                "acceptance_%": round(100.0 * acc / n_bf, 1) if n_bf else 0.0,
                "n_identities": len(pred),
                "coverage_%": round(100.0 * m["n"] / n_gt, 1) if n_gt else 0.0,
                "RMSE_mm_s": round(m["RMSE"], 4),
                "MAE_mm_s": round(m["MAE"], 4),
                "pRMSE_%": round(m["pRMSE_%"], 2),
                "MAPE_%": round(m["MAPE_%"], 2),
            }
        )
        print(
            f"min_agree={ma} conf>={ct}: accept={out_rows[-1]['acceptance_%']}%  "
            f"cov={out_rows[-1]['coverage_%']}%  MAE={out_rows[-1]['MAE_mm_s']} mm/s  "
            f"pRMSE={out_rows[-1]['pRMSE_%']}%"
        )

    res = pd.DataFrame(out_rows)
    cfg.stage3_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.stage3_dir / "reid_sweep.csv"
    res.to_csv(path, index=False, sep=";")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
