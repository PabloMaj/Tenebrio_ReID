"""Unified training + evaluation framework for the Tenebrio re-ID benchmark.

Runs every requested method over every requested split (1..5), using the exact
published training scheme (same pre-processing, same 5-fold split, same triplet
mining / evaluation protocol, same epoch budget, best epoch chosen on the
validation set).  Method ``proposed`` reproduces the published best model;
``convnext_tiny`` / ``resnet50_arcface`` / ``dinov2_vits14`` are the SOTA
options.

Examples
--------
    # full benchmark (4 methods x 5 splits)
    python train.py --dataset-root ".../re_identification_dataset_only_thorax"

    # just the proposed model, split 1
    python train.py --methods proposed --splits 1 --dataset-root ...

    # quick functional check (tiny, ~1 min on GPU)
    python train.py --methods proposed convnext_tiny resnet50_arcface dinov2_vits14 \
        --splits 1 --num-epochs 2 --eval-every 1 --limit-classes 8 --dataset-root ...
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ALL_METHODS, METHODS, ExperimentConfig
from data import BeetleDataset, build_transform, read_data
from engine import (
    METRIC_NAMES,
    aggregate,
    append_results_file,
    evaluate,
    make_accuracy_calculator,
    train_one_epoch,
)
from losses import build_loss, build_optimizer, build_scheduler
from models import build_model, checkpoint_state_dict, count_parameters
from utils import (
    env_info,
    make_loader_generator,
    set_seed,
    write_json,
    write_params_txt,
)

DEFAULT_DATASET_ROOT = str(
    Path(__file__).resolve().parent.parent
    / "datasets/re_identification/re_identification_dataset_only_thorax"
)


# --------------------------------------------------------------------------- #
def build_params_txt(method_cfg, exp: ExperimentConfig, split_id: int) -> dict:
    """Superset of the original ``params.txt`` keys + the extra knobs."""
    p = {
        "method": method_cfg.name,
        "model_name": method_cfg.backbone,
        "weights_source": method_cfg.weights_source,
        "number_of_fc_layers": 0 if method_cfg.loss in ("bot", "agw", "clip_reid") else 1,
        "embedding_dim": method_cfg.embedding_dim,
        "bnneck": method_cfg.bnneck,
        "feature_extract": method_cfg.feature_extract,
        "split_id": split_id,
        "part_for_analysis": exp.part_for_analysis,
        "batch_size": exp.batch_size,
        "num_epochs": exp.num_epochs,
        "eval_every": exp.eval_every,
        "use_oversampling": exp.use_oversampling,
        "best_metric": exp.best_metric,
        "loss_func": method_cfg.loss,
        "distance_type": method_cfg.distance,
        "seed": exp.seed,
        "input_adaptation": method_cfg.input_adaptation,
    }
    if method_cfg.loss == "triplet":
        p.update(
            {
                "mining_func": "TripletMarginMiner",
                "margin_value_loss": method_cfg.margin_loss,
                "margin_value_miner": method_cfg.margin_miner,
                "type_of_triplets": method_cfg.type_of_triplets,
                "reducer": "ThresholdReducer(low=0)",
            }
        )
    elif method_cfg.loss == "subcenter_arcface":
        p.update(
            {
                "mining_func": "none",
                "arcface_margin": method_cfg.arcface_margin,
                "arcface_scale": method_cfg.arcface_scale,
                "arcface_sub_centers": method_cfg.arcface_sub_centers,
                "loss_lr": method_cfg.loss_lr,
            }
        )
    elif method_cfg.loss in ("bot", "agw"):
        p.update(
            {
                "mining_func": "BatchHardMiner" if method_cfg.loss == "bot" else "WRT (weighted)",
                "id_loss": "CrossEntropy(label_smoothing=%s)" % method_cfg.id_label_smoothing,
                "triplet_loss": (
                    "TripletMargin(%s)" % method_cfg.triplet_margin_reid
                    if method_cfg.loss == "bot"
                    else "WeightedRegularizationTriplet"
                ),
                "last_stride1": method_cfg.last_stride1,
                "non_local": method_cfg.non_local,
                "pooling": method_cfg.pooling,
                "warmup_epochs": method_cfg.warmup_epochs,
            }
        )
    elif method_cfg.loss == "clip_reid":
        p.update(
            {
                "clip_arch": method_cfg.clip_arch,
                "clip_n_ctx": method_cfg.clip_n_ctx,
                "stage1_epochs": max(1, round(method_cfg.clip_stage1_frac * exp.num_epochs)),
                "stage1_loss": "image<->text SupCon (prompt learning)",
                "stage2_loss": "ID CE(label_smoothing) + batch-hard triplet + image->text CE",
                "clip_prompt_lr": method_cfg.clip_prompt_lr,
                "clip_image_lr": method_cfg.clip_image_lr,
                "clip_head_lr": method_cfg.clip_head_lr,
            }
        )
        p["optimizer"] = "adam (prompt/image/head param groups)"
        p["distance_type"] = method_cfg.distance
        return p
    opt_lr = f"{method_cfg.optimizer}_{method_cfg.lr}"
    if method_cfg.weight_decay:
        opt_lr += f"_wd{method_cfg.weight_decay}"
    p["optimizer"] = opt_lr
    return p


def run_one(method_name: str, split_id: int, exp: ExperimentConfig, device) -> dict:
    method_cfg = METHODS[method_name]
    run_dir = os.path.join(exp.output_dir, exp.part_for_analysis, method_name, f"split_{split_id}")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(exp.seed)

    # ---- data (identical pre-processing + split for every method) ----
    split = read_data(
        split_id=split_id,
        path_to_dataset=exp.dataset_root,
        part_for_analysis=exp.part_for_analysis,
        use_oversampling=exp.use_oversampling,
        limit_classes=exp.limit_classes,
    )
    transform = build_transform(exp.mean, exp.std)
    ds_train = BeetleDataset(split.image_train_paths, split.train_labels, transform, split.label_to_num)
    ds_val = BeetleDataset(split.image_val_paths, split.val_labels, transform, split.label_to_num)
    ds_test = BeetleDataset(split.image_test_paths, split.test_labels, transform, split.label_to_num)

    loader_train = DataLoader(
        ds_train,
        batch_size=exp.batch_size,
        shuffle=True,
        num_workers=exp.num_workers,
        generator=make_loader_generator(exp.seed),
        drop_last=False,
    )

    with open(os.path.join(run_dir, "label_to_num.pkl"), "wb") as f:
        pickle.dump(split.label_to_num, f)

    # ---- model / loss / optimiser ----
    model = build_model(method_cfg, device=device, num_classes=split.num_classes)
    loss_bundle = build_loss(
        method_cfg, num_classes=split.num_classes, device=device, num_epochs=exp.num_epochs
    )
    optimizer = build_optimizer(method_cfg, model)
    scheduler = build_scheduler(method_cfg, optimizer)
    acc_calc = make_accuracy_calculator()

    if method_cfg.loss == "clip_reid":
        # count in stage 2 (image-encoder fine-tuning) - stage 1 only trains prompts
        model.set_stage(2)
        param_counts = count_parameters(model)
        param_counts["note"] = "trainable = stage 2 (image encoder); stage 1 trains only the prompts"
        model.set_stage(1)
    else:
        param_counts = count_parameters(model)

    write_params_txt(os.path.join(run_dir, "params.txt"), build_params_txt(method_cfg, exp, split_id))

    train_inventory = {
        "method": method_cfg.to_dict(),
        "experiment": {
            "part_for_analysis": exp.part_for_analysis,
            "split_id": split_id,
            "num_epochs": exp.num_epochs,
            "eval_every": exp.eval_every,
            "batch_size": exp.batch_size,
            "use_oversampling": exp.use_oversampling,
            "best_metric": exp.best_metric,
            "seed": exp.seed,
            "preprocessing": {
                "reader": "cv2.imread (BGR, no channel swap)",
                "scale": "x / 255",
                "resize": "128x128",
                "normalize_mean": list(exp.mean),
                "normalize_std": list(exp.std),
            },
        },
        "dataset": split.inventory(),
        "model": {
            "backbone": method_cfg.backbone,
            "embedding_dim": model.embedding_dim,
            "parameters": param_counts,
        },
        "env": env_info(),
    }
    write_json(os.path.join(run_dir, "inventory_train.json"), train_inventory)

    # ---- training loop ----
    history = []
    best = {"epoch": None, "val_metric": -1.0, "val": None, "test": None}
    t_start = time.time()

    epoch_bar = tqdm(range(1, exp.num_epochs + 1), desc=f"{method_name} split{split_id}", unit="ep")
    first_eval_done = False
    for epoch in epoch_bar:
        stats = train_one_epoch(
            model, loss_bundle, optimizer, loader_train, device, epoch, exp.num_epochs
        )
        if scheduler is not None:
            stats["lr"] = round(scheduler.get_last_lr()[0], 8)
            scheduler.step()
        record = {"train": stats}

        if epoch % exp.eval_every == 0 or epoch == exp.num_epochs:
            val_labels, val_acc = evaluate(ds_train, ds_val, model, acc_calc)
            test_labels, test_acc = evaluate(ds_train, ds_test, model, acc_calc)

            append_results_file(
                os.path.join(run_dir, "results_eval_val.txt"),
                epoch, val_labels, val_acc, write_header=not first_eval_done,
            )
            append_results_file(
                os.path.join(run_dir, "results_eval_test.txt"),
                epoch, test_labels, test_acc, write_header=not first_eval_done,
            )
            first_eval_done = True

            val_agg = aggregate(val_acc)
            test_agg = aggregate(test_acc)
            record["val"] = val_agg
            record["test"] = test_agg
            epoch_bar.set_postfix(
                val=f"{val_agg[exp.best_metric]:.3f}", test_mAP=f"{test_agg['mean_average_precision']:.3f}"
            )

            if val_agg[exp.best_metric] > best["val_metric"]:
                best = {
                    "epoch": epoch,
                    "val_metric": val_agg[exp.best_metric],
                    "val": val_agg,
                    "test": test_agg,
                }
                torch.save(checkpoint_state_dict(model), os.path.join(run_dir, "model_best_epoch.pt"))

        history.append(record)

    torch.save(checkpoint_state_dict(model), os.path.join(run_dir, "model_last_epoch.pt"))
    wall = round(time.time() - t_start, 1)

    eval_inventory = {
        "method": method_name,
        "split_id": split_id,
        "wall_seconds": wall,
        "epochs_trained": exp.num_epochs,
        "best_epoch": best["epoch"],
        "best_metric_name": exp.best_metric,
        "best_val_metrics": best["val"],
        "selected_test_metrics": best["test"],
        "history": history,
    }
    write_json(os.path.join(run_dir, "inventory_eval.json"), eval_inventory)
    write_json(os.path.join(run_dir, "history.json"), {"history": history})

    return {
        "method": method_name,
        "split_id": split_id,
        "best_epoch": best["epoch"],
        "wall_seconds": wall,
        **{f"val_{k}": (best["val"] or {}).get(k) for k in METRIC_NAMES},
        **{f"test_{k}": (best["test"] or {}).get(k) for k in METRIC_NAMES},
    }


# --------------------------------------------------------------------------- #
def write_summary(rows: list, exp: ExperimentConfig):
    import csv

    out_base = os.path.join(exp.output_dir, exp.part_for_analysis)
    os.makedirs(out_base, exist_ok=True)
    csv_path = os.path.join(out_base, "benchmark_summary.csv")
    fieldnames = ["method", "split_id", "best_epoch", "wall_seconds"] + [
        f"{s}_{m}" for s in ("val", "test") for m in METRIC_NAMES
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # markdown table: mean +/- std over splits, per method (test set, best epoch)
    md = ["# Re-ID benchmark summary", "",
          f"- part: **{exp.part_for_analysis}**",
          f"- epochs: {exp.num_epochs}, eval every {exp.eval_every}, best epoch by val `{exp.best_metric}`",
          f"- splits: {sorted({r['split_id'] for r in rows})}", "",
          "Test-set metrics at the selected (best-val) epoch, mean ± std over splits:", "",
          "| method | " + " | ".join(METRIC_NAMES) + " |",
          "|" + "---|" * (len(METRIC_NAMES) + 1)]
    for method in sorted({r["method"] for r in rows}):
        mrows = [r for r in rows if r["method"] == method]
        cells = []
        for m in METRIC_NAMES:
            vals = np.array([r[f"test_{m}"] for r in mrows if r[f"test_{m}"] is not None], dtype=float)
            cells.append(f"{vals.mean():.3f} ± {vals.std():.3f}" if len(vals) else "n/a")
        md.append(f"| {method} | " + " | ".join(cells) + " |")
    md_path = os.path.join(out_base, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nsummary -> {csv_path}\n         -> {md_path}")
    print("\n".join(md[-(len({r['method'] for r in rows}) + 2):]))


def collect_existing_rows(exp: ExperimentConfig) -> list:
    """Rebuild summary rows from every ``inventory_eval.json`` already in the run tree."""
    import glob
    import json

    base = os.path.join(exp.output_dir, exp.part_for_analysis)
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "*", "split_*", "inventory_eval.json"))):
        with open(path, encoding="utf-8") as f:
            inv = json.load(f)
        val, test = inv.get("best_val_metrics") or {}, inv.get("selected_test_metrics") or {}
        rows.append(
            {
                "method": inv["method"],
                "split_id": inv["split_id"],
                "best_epoch": inv.get("best_epoch"),
                "wall_seconds": inv.get("wall_seconds"),
                **{f"val_{k}": val.get(k) for k in METRIC_NAMES},
                **{f"test_{k}": test.get(k) for k in METRIC_NAMES},
            }
        )
    return rows


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--part", default="thorax", help="body part token filtered in file names")
    ap.add_argument("--num-epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--oversampling", action="store_true", help="enable train-set class oversampling")
    ap.add_argument("--best-metric", default="mean_average_precision_at_r", choices=METRIC_NAMES)
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip training; rebuild benchmark_summary.{csv,md} from existing runs/")
    ap.add_argument("--limit-classes", type=int, default=None, help="use only the first N labels (smoke tests)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cuda", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    exp = ExperimentConfig(
        dataset_root=args.dataset_root,
        part_for_analysis=args.part,
        splits=tuple(args.splits),
        num_epochs=args.num_epochs,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        use_oversampling=args.oversampling,
        best_metric=args.best_metric,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        limit_classes=args.limit_classes,
        device=str(device),
    )

    if args.summarize_only:
        rows = collect_existing_rows(exp)
        if not rows:
            print(f"no runs found under {os.path.join(exp.output_dir, exp.part_for_analysis)}")
            return []
        print(f"rebuilding summary from {len(rows)} existing runs")
        write_summary(rows, exp)
        return rows

    print(f"device: {device}")
    print(f"methods: {args.methods}")
    print(f"splits : {args.splits}")
    for m in args.methods:
        if METHODS[m].download_note:
            print(f"  [{m}] {METHODS[m].download_note}")

    rows = []
    combos = [(m, s) for m in args.methods for s in args.splits]
    for method_name, split_id in tqdm(combos, desc="benchmark", unit="run"):
        rows.append(run_one(method_name, split_id, exp, device))

    # fold in any earlier runs so a partial invocation still yields a full summary
    seen = {(r["method"], r["split_id"]) for r in rows}
    rows += [r for r in collect_existing_rows(exp) if (r["method"], r["split_id"]) not in seen]
    write_summary(rows, exp)
    return rows


if __name__ == "__main__":
    main()
