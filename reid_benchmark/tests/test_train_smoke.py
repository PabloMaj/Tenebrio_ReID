"""End-to-end functional check: the framework must actually train and evaluate,
produce the published-style output files, a machine-readable inventory, a
loadable checkpoint, and a benchmark summary - for all four methods."""

import copy
import importlib.util
import json
import os

import pytest

import config
import evaluate as evaluate_mod
import train as train_mod
from engine import METRIC_NAMES

# clip_reid needs the OpenAI CLIP package, installed separately (see requirements.txt);
# skip it from the combined smoke run rather than failing the whole test when it's
# missing - it's an optional extra, not a core dependency.
_HAS_CLIP = importlib.util.find_spec("clip") is not None
METHODS_UNDER_TEST = [m for m in config.METHODS.keys() if m != "clip_reid" or _HAS_CLIP]


@pytest.fixture(autouse=True)
def _no_pretrained(monkeypatch):
    for name, cfg in config.METHODS.items():
        patched = copy.deepcopy(cfg)
        patched.pretrained = False
        monkeypatch.setitem(config.METHODS, name, patched)


def test_full_benchmark_smoke(synthetic_root, tmp_path):
    out = tmp_path / "runs"
    rows = train_mod.main(
        [
            "--methods", *METHODS_UNDER_TEST,
            "--splits", "1",
            "--dataset-root", synthetic_root,
            "--part", "thorax",
            "--num-epochs", "2",
            "--eval-every", "1",
            "--batch-size", "8",
            "--limit-classes", "6",
            "--output-dir", str(out),
            "--no-cuda",
        ]
    )
    assert len(rows) == len(METHODS_UNDER_TEST)

    for method in METHODS_UNDER_TEST:
        run_dir = out / "thorax" / method / "split_1"
        for fname in (
            "params.txt",
            "inventory_train.json",
            "inventory_eval.json",
            "results_eval_val.txt",
            "results_eval_test.txt",
            "label_to_num.pkl",
            "model_best_epoch.pt",
            "model_last_epoch.pt",
        ):
            assert (run_dir / fname).exists(), f"{method}: missing {fname}"

        inv = json.loads((run_dir / "inventory_eval.json").read_text())
        assert inv["best_epoch"] in (1, 2)
        assert set(METRIC_NAMES).issubset(inv["selected_test_metrics"])
        # metrics are real numbers in [0, 1-ish], i.e. the model was actually scored
        for m in METRIC_NAMES:
            v = inv["selected_test_metrics"][m]
            assert isinstance(v, float) and -1.0 <= v <= 1.0

        # results file has the exact published column layout
        header = (run_dir / "results_eval_test.txt").read_text().splitlines()[0].split("\t")
        assert header == ["epoch", "label"] + METRIC_NAMES

        train_inv = json.loads((run_dir / "inventory_train.json").read_text())
        assert train_inv["model"]["parameters"]["trainable"] > 0
        assert train_inv["experiment"]["preprocessing"]["resize"] == "128x128"

    assert (out / "thorax" / "benchmark_summary.csv").exists()
    assert (out / "thorax" / "benchmark_summary.md").exists()


def test_checkpoint_reloads_and_evaluates(synthetic_root, tmp_path):
    out = tmp_path / "runs"
    train_mod.main(
        [
            "--methods", "proposed",
            "--splits", "1",
            "--dataset-root", synthetic_root,
            "--part", "thorax",
            "--num-epochs", "1",
            "--eval-every", "1",
            "--batch-size", "8",
            "--limit-classes", "6",
            "--output-dir", str(out),
            "--no-cuda",
        ]
    )
    run_dir = out / "thorax" / "proposed" / "split_1"
    evaluate_mod.main(
        ["--run-dir", str(run_dir), "--dataset-root", synthetic_root, "--no-cuda"]
    )
    assert (run_dir / "inventory_reeval.json").exists()
    reeval = json.loads((run_dir / "inventory_reeval.json").read_text())
    assert set(METRIC_NAMES).issubset(reeval["test"])
