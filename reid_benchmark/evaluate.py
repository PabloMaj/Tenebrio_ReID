"""Stand-alone evaluation of a trained run directory.

    python evaluate.py --run-dir runs/thorax/proposed/split_1 \
        --dataset-root ".../re_identification_dataset_only_thorax"

Re-loads ``model_best_epoch.pt``, rebuilds the exact method config, recomputes
the test-set (and val-set) metrics with the same protocol as training, and
writes ``inventory_reeval.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import os

import torch

from config import METHODS
from data import BeetleDataset, build_transform, read_data
from engine import METRIC_NAMES, aggregate, evaluate, make_accuracy_calculator
from models import build_model, load_checkpoint
from utils import read_json, write_json


def load_run(run_dir: str):
    params = {}
    with open(os.path.join(run_dir, "params.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                params[k] = v
    return params


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint", default="model_best_epoch.pt")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    params = load_run(args.run_dir)
    method_name = params["method"]
    method_cfg = METHODS[method_name]
    split_id = int(params["split_id"])
    part = params["part_for_analysis"]

    inv = read_json(os.path.join(args.run_dir, "inventory_train.json"))
    mean = inv["experiment"]["preprocessing"]["normalize_mean"]
    std = inv["experiment"]["preprocessing"]["normalize_std"]
    use_oversampling = str(params.get("use_oversampling", "False")) == "True"
    limit_classes = inv["dataset"].get("n_labels")
    # only pass limit if it looks like a smoke run (< full set)
    limit_classes = limit_classes if limit_classes and limit_classes < 60 else None

    split = read_data(split_id, args.dataset_root, part, use_oversampling, limit_classes=limit_classes)
    transform = build_transform(mean, std)
    ds_train = BeetleDataset(split.image_train_paths, split.train_labels, transform, split.label_to_num)
    ds_val = BeetleDataset(split.image_val_paths, split.val_labels, transform, split.label_to_num)
    ds_test = BeetleDataset(split.image_test_paths, split.test_labels, transform, split.label_to_num)

    model = build_model(method_cfg, device=device, num_classes=split.num_classes)
    state = torch.load(
        os.path.join(args.run_dir, args.checkpoint), map_location=device, weights_only=True
    )
    load_checkpoint(model, state)

    acc_calc = make_accuracy_calculator()
    _, val_acc = evaluate(ds_train, ds_val, model, acc_calc)
    _, test_acc = evaluate(ds_train, ds_test, model, acc_calc)
    val_agg, test_agg = aggregate(val_acc), aggregate(test_acc)

    print(f"\n{method_name}  split {split_id}  ({args.checkpoint})")
    print("metric".ljust(30), "val".rjust(8), "test".rjust(8))
    for m in METRIC_NAMES:
        print(m.ljust(30), f"{val_agg[m]:8.4f}", f"{test_agg[m]:8.4f}")

    write_json(
        os.path.join(args.run_dir, "inventory_reeval.json"),
        {"method": method_name, "split_id": split_id, "checkpoint": args.checkpoint,
         "val": val_agg, "test": test_agg},
    )


if __name__ == "__main__":
    main()
