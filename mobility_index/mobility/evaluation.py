"""Stage 3 - evaluate stage-2 mobility against the stage-1 ground truth.

The two stages describe the *same* individuals, keyed by ``<series>::<tag>``
(stage 1) and ``<predicted_series>::<predicted_tag>`` (stage 2).  We report
absolute and relative error metrics, both overall and per individual, per model
split and averaged over the 5 splits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .postfilter import apply_detection_filter, frames_per_series

_METRIC_COLS = {
    "mm_per_s": "mobility_mm_per_s",
    "px_per_frame": "mobility_px_per_frame",
}


def _error_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    if len(gt) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "pRMSE_%": np.nan, "MAPE_%": np.nan}
    err = pred - gt
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mean_gt = float(np.mean(gt))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(err) / gt) * 100)
    return {
        "n": len(gt),
        "RMSE": rmse,
        "MAE": mae,
        "pRMSE_%": rmse / mean_gt * 100 if mean_gt else np.nan,
        "MAPE_%": mape,
    }


class MobilityEvaluator:
    def __init__(self, config):
        self.cfg = config

    def evaluate(self, gt_csv: Path, pred_all_splits_csv: Path) -> dict:
        gt = pd.read_csv(gt_csv, sep=";")
        pred = pd.read_csv(pred_all_splits_csv, sep=";")
        out_dir = self.cfg.stage3_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # -- stage-3 detection-fraction filter -------------------------------- #
        min_frac = float(getattr(self.cfg, "min_detection_frac", 0.0) or 0.0)
        count_col = getattr(self.cfg, "detection_count_col", "n_positions_used")
        n_frames = frames_per_series(self.cfg, pd.concat([gt["series"], pred["series"]]).unique())
        n_gt_all = gt["key"].nunique()
        gt, gt_drop = apply_detection_filter(gt, n_frames, min_frac, count_col)
        pred, pred_drop = apply_detection_filter(pred, n_frames, min_frac, count_col)
        self._write_filter_log(gt, gt_drop, pred, pred_drop, n_frames, min_frac, out_dir)
        if min_frac:
            print(
                f"[stage3] detection filter (>= {min_frac:.0%}): "
                f"GT {len(gt)}/{n_gt_all} kept, pred {len(pred)} kept "
                f"({len(pred_drop)} dropped)"
            )

        n_gt = gt["key"].nunique()
        summary_rows = []
        per_split_matched = {}

        for split, grp in pred.groupby("model_split"):
            merged = gt.merge(
                grp, on="key", suffixes=("_gt", "_pred"), how="inner"
            )
            per_split_matched[split] = merged
            for unit, col in _METRIC_COLS.items():
                m = _error_metrics(merged[f"{col}_gt"].values, merged[f"{col}_pred"].values)
                m["model_split"] = split
                m["unit"] = unit
                m["coverage_%"] = 100.0 * len(merged) / n_gt
                summary_rows.append(m)

        summary = pd.DataFrame(summary_rows)

        # mean +/- std over the numeric CV splits (skipped for the single
        # "ensemble" pseudo-split produced by Stage2Runner)
        numeric_splits = [s for s in summary["model_split"].unique() if str(s).isdigit()]
        if len(numeric_splits) >= 2:
            agg_rows = []
            for unit in _METRIC_COLS:
                sub = summary[
                    (summary["unit"] == unit) & (summary["model_split"].isin(numeric_splits))
                ]
                for stat, fn in (("mean", np.mean), ("std", np.std)):
                    row = {"model_split": stat, "unit": unit}
                    for c in ["n", "RMSE", "MAE", "pRMSE_%", "MAPE_%", "coverage_%"]:
                        row[c] = float(fn(sub[c].values))
                    agg_rows.append(row)
            summary = pd.concat([summary, pd.DataFrame(agg_rows)], ignore_index=True)
        summary_path = out_dir / "summary_metrics.csv"
        summary.to_csv(summary_path, index=False, sep=";")

        per_beetle = self._per_beetle(gt, pred, out_dir)
        for name, fn in (("per_beetle_error.png", lambda: self._plot_per_beetle(per_beetle, out_dir)),
                         ("mobility_scatter.png", lambda: self._plot_scatter(gt, pred, out_dir))):
            try:
                fn()
            except OSError as e:                   # e.g. the PNG is open in a viewer
                print(f"[stage3] skipped {name} ({e})")

        return {
            "summary": summary_path,
            "per_beetle": out_dir / "per_beetle_metrics.csv",
            "n_gt": n_gt,
            "n_gt_before_filter": n_gt_all,
        }

    @staticmethod
    def _write_filter_log(gt, gt_drop, pred, pred_drop, n_frames, min_frac, out_dir: Path) -> None:
        def _rows(df, side, passed):
            for r in df.itertuples():
                yield {
                    "key": r.key, "series": r.series, "identity": r.identity, "side": side,
                    "n_frames_series": r.n_frames_series, "n_positions_used": r.n_positions_used,
                    "detection_frac": r.detection_frac, "passed": passed,
                }
        rows = list(_rows(gt, "gt", True)) + list(_rows(gt_drop, "gt", False))
        rows += list(_rows(pred, "pred", True)) + list(_rows(pred_drop, "pred", False))
        df = pd.DataFrame(rows).sort_values(["side", "detection_frac"], ascending=[True, False])
        df.attrs["min_detection_frac"] = min_frac
        df.to_csv(out_dir / "detection_filter.csv", index=False, sep=";")

    # ------------------------------------------------------------------ #
    def _per_beetle(self, gt: pd.DataFrame, pred: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
        rows = []
        pred_by_key = {k: g for k, g in pred.groupby("key")}
        for _, g in gt.iterrows():
            key = g["key"]
            pk = pred_by_key.get(key)
            row = {
                "key": key,
                "series": g["series"],
                "tag": g["identity"],
                "gt_mm_per_s": g["mobility_mm_per_s"],
                "gt_px_per_frame": g["mobility_px_per_frame"],
                "n_splits_matched": 0 if pk is None else len(pk),
            }
            if pk is not None:
                for unit, col in _METRIC_COLS.items():
                    vals = pk[col].values
                    gt_val = g[col]
                    pred_mean = float(np.mean(vals))
                    row[f"pred_{unit}_mean"] = pred_mean
                    row[f"pred_{unit}_std"] = float(np.std(vals))
                    row[f"abs_err_{unit}"] = abs(pred_mean - gt_val)
                    row[f"rel_err_{unit}"] = (
                        abs(pred_mean - gt_val) / gt_val if gt_val else np.nan
                    )
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("gt_mm_per_s", ascending=False)
        df.to_csv(out_dir / "per_beetle_metrics.csv", index=False, sep=";")
        return df

    def _plot_per_beetle(self, per_beetle: pd.DataFrame, out_dir: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if "pred_mm_per_s_mean" not in per_beetle.columns:
            return
        d = per_beetle.dropna(subset=["pred_mm_per_s_mean"]).copy()
        if d.empty:
            return
        d = d.sort_values("gt_mm_per_s", ascending=False)
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(max(8, len(d) * 0.28), 5))
        ax.bar(x - 0.2, d["gt_mm_per_s"], width=0.4, label="GT (stage 1)")
        ax.bar(
            x + 0.2,
            d["pred_mm_per_s_mean"],
            width=0.4,
            yerr=d["pred_mm_per_s_std"],
            capsize=2,
            label="pred (stage 2, mean±std)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(d["key"], rotation=90, fontsize=6)
        ax.set_ylabel("mobility [mm/s]")
        ax.set_title("Per-beetle mobility: ground truth vs re-ID")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "per_beetle_error.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _r2_pearson(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        """R^2 (1 - SS_res/SS_tot, the regression coefficient of determination)
        and Pearson's r between two paired vectors."""
        ss_res = float(np.sum((pred - gt) ** 2))
        ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
        if len(gt) < 2 or np.std(gt) == 0 or np.std(pred) == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(gt, pred)[0, 1])
        return r2, r

    def _plot_scatter(self, gt: pd.DataFrame, pred: pd.DataFrame, out_dir: Path) -> None:
        """Log-log scatter of ground-truth vs predicted mobility, one point per
        matched beetle, with R^2 and Pearson r (linear and log10) overlaid."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        col = _METRIC_COLS["mm_per_s"]
        pred_by_key = (
            pred.groupby("key")
            .agg(mob_pred=(col, "mean"), series=("series", "first"))
            .reset_index()
        )
        merged = gt[["key", "series", col]].rename(columns={col: "mob_gt"}).merge(
            pred_by_key[["key", "mob_pred"]], on="key", how="inner"
        )
        merged = merged[(merged["mob_gt"] > 0) & (merged["mob_pred"] > 0)]
        if len(merged) < 3:
            return

        g = merged["mob_gt"].to_numpy(dtype=float)
        p = merged["mob_pred"].to_numpy(dtype=float)
        r2_lin, r_lin = self._r2_pearson(g, p)
        r2_log, r_log = self._r2_pearson(np.log10(g), np.log10(p))

        # Okabe-Ito, fixed order -> assigned to the sorted series list
        okabe_ito = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                     "#56B4E9", "#D55E00", "#F0E442", "#000000"]
        series_order = sorted(merged["series"].unique())
        cmap = {s: okabe_ito[i % len(okabe_ito)] for i, s in enumerate(series_order)}

        lo = float(min(g.min(), p.min())) * 0.8
        hi = float(max(g.max(), p.max())) * 1.25

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#888888",
                zorder=1, label="1:1")
        for s in series_order:
            sub = merged[merged["series"] == s]
            ax.scatter(sub["mob_gt"], sub["mob_pred"], s=64, alpha=0.85,
                       color=cmap[s], edgecolors="white", linewidths=0.6,
                       zorder=3, label=s)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, which="both", color="#dddddd", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("ground-truth mobility [mm/s]")
        ax.set_ylabel("predicted mobility [mm/s]")
        ax.set_title(f"Mobility agreement (n = {len(merged)})")

        txt = (
            f"linear:  R² = {r2_lin:.3f}   r = {r_lin:.3f}\n"
            f"log10:  R² = {r2_log:.3f}   r = {r_log:.3f}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=10, color="#222222", family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "mobility_scatter.png", dpi=150)
        plt.close(fig)
