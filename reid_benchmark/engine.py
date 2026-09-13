"""Training / evaluation engine shared by all benchmarked methods.

The evaluation protocol is identical to the published one:

* embeddings for the reference (train) and query (val or test) sets are computed
  with ``pytorch_metric_learning``'s tester,
* ``AccuracyCalculator(include=(), return_per_class=True)`` gives per-class
  AMI / NMI / mAP / mAP@R / MRR / P@1 / r-precision,
* results are appended every ``eval_every`` epochs to ``results_eval_val.txt`` /
  ``results_eval_test.txt`` in the same tab-separated layout as before,
* the "best epoch" is the one with the highest mean-over-classes value of
  ``best_metric`` on the validation set.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm

# Fixed column order -> output files stay comparable with the published results.
METRIC_NAMES = [
    "AMI",
    "NMI",
    "mean_average_precision",
    "mean_average_precision_at_r",
    "mean_reciprocal_rank",
    "precision_at_1",
    "r_precision",
]
_SCALAR_METRICS = {"AMI", "NMI"}


def make_accuracy_calculator() -> AccuracyCalculator:
    return AccuracyCalculator(
        include=(), avg_of_avgs=False, return_per_class=True, device=torch.device("cpu")
    )


def train_one_epoch(model, loss_bundle, optimizer, loader, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    n_batches = 0
    total_triplets = 0
    t0 = time.time()

    clip_stage = None
    if loss_bundle.mode == "clip_reid":
        clip_stage = 1 if epoch <= loss_bundle.stage1_epochs else 2
        model.set_stage(clip_stage)

    pbar = tqdm(
        loader,
        desc=f"  epoch {epoch:3d}/{num_epochs}"
        + (f" [clip stage {clip_stage}]" if clip_stage else ""),
        leave=False,
        unit="batch",
    )
    for data, labels in pbar:
        data = data.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        if loss_bundle.loss_optimizer is not None:
            loss_bundle.loss_optimizer.zero_grad()

        if loss_bundle.mode == "clip_reid":
            if clip_stage == 1:
                img_f = model.encode_image_norm(data)
                uniq = torch.unique(labels)
                txt_f = model.text_features(uniq, normed=True)
                loss = loss_bundle.loss_func.stage1(txt_f, img_f, uniq, labels)
            else:
                img_f, id_logits, i2t_logits = model.forward_stage2(data)
                loss = loss_bundle.loss_func.stage2(img_f, id_logits, i2t_logits, labels)
        elif loss_bundle.mode == "reid_combined":
            # dedicated re-ID: pre-BNNeck feature + ID logits -> combined loss
            feat_triplet, id_logits = model.forward_train(data)
            loss = loss_bundle.loss_func(feat_triplet, id_logits, labels)
        else:
            embeddings = model(data)
            if loss_bundle.miner is not None:
                indices_tuple = loss_bundle.miner(embeddings, labels)
                loss = loss_bundle.loss_func(embeddings, labels, indices_tuple)
                total_triplets += int(getattr(loss_bundle.miner, "num_triplets", 0))
            else:
                loss = loss_bundle.loss_func(embeddings, labels)

        loss.backward()
        optimizer.step()
        if loss_bundle.loss_optimizer is not None:
            loss_bundle.loss_optimizer.step()

        running_loss += float(loss.detach().cpu())
        n_batches += 1
        pbar.set_postfix(loss=f"{running_loss / max(n_batches, 1):.4f}")

    stats = {
        "epoch": epoch,
        "avg_loss": running_loss / max(n_batches, 1),
        "n_batches": n_batches,
        "mined_triplets": total_triplets,
        "seconds": round(time.time() - t0, 2),
    }
    if clip_stage is not None:
        stats["clip_stage"] = clip_stage
    return stats


def _get_all_embeddings(dataset, model):
    device = next(model.parameters()).device
    tester = testers.BaseTester(data_device=device)
    return tester.get_all_embeddings(dataset, model)


@torch.no_grad()
def evaluate(reference_set, query_set, model, accuracy_calculator):
    """Returns ``(per_class_labels, accuracies_dict)`` for one query set."""
    model.eval()
    ref_emb, ref_labels = _get_all_embeddings(reference_set, model)
    qry_emb, qry_labels = _get_all_embeddings(query_set, model)
    ref_labels = ref_labels.squeeze(1)
    qry_labels = qry_labels.squeeze(1)

    accuracies = accuracy_calculator.get_accuracy(
        query=qry_emb,
        query_labels=qry_labels,
        reference=ref_emb,
        reference_labels=ref_labels,
        ref_includes_query=False,
    )
    per_class_labels = np.unique(qry_labels.cpu().numpy())
    return per_class_labels, accuracies


def aggregate(accuracies: dict) -> dict:
    """Mean over classes for the per-class metrics (scalars kept as-is)."""
    out = {}
    for key in METRIC_NAMES:
        val = accuracies[key]
        if key in _SCALAR_METRICS:
            out[key] = float(np.round(val, 4))
        else:
            out[key] = float(np.round(np.mean(val), 4))
    return out


def append_results_file(path, epoch, per_class_labels, accuracies, write_header: bool):
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("epoch\tlabel\t" + "\t".join(METRIC_NAMES) + "\n")
        for i, label in enumerate(per_class_labels):
            row = [str(epoch), str(label)]
            for key in METRIC_NAMES:
                if key in _SCALAR_METRICS:
                    row.append(f"{np.round(accuracies[key], 4)}")
                else:
                    row.append(f"{np.round(accuracies[key][i], 4)}")
            f.write("\t".join(row) + "\n")
