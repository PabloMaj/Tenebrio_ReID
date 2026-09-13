# Tenebrio Re-Identification - Unified Benchmark Framework

Trains and evaluates 8 re-identification embedding models for individual *Tenebrio
molitor* beetles under one identical protocol (same pre-processing, 5-fold split,
triplet/loss protocol, epoch budget, best-epoch selection on validation) so a proposed
model can be compared fairly against modern SOTA alternatives.

## Methods

| key | backbone | loss | year |
|-----|----------|------|------|
| `proposed` | MobileNetV2 | TripletMarginLoss | 2018 |
| `convnext_tiny` | ConvNeXt-Tiny | TripletMarginLoss | 2022 |
| `resnet50_arcface` | ResNet50 + BNNeck | SubCenterArcFaceLoss | 2019/20 |
| `dinov2_vits14` | DINOv2 ViT-S/14 | TripletMarginLoss | 2023 |
| `bot_reid` | ResNet50 (Bag of Tricks) | ID CE + batch-hard triplet | 2019 |
| `agw_reid` | ResNet50 + Non-local + GeM | ID CE + Weighted Reg. Triplet | 2021 |
| `convnextv2_reid` | ConvNeXt V2-Tiny + GeM | ID CE + Weighted Reg. Triplet | 2023 |
| `clip_reid` | CLIP ViT-B/16 | 2-stage prompt learning + triplet | 2023 |

All pretrained weights auto-download on first use (no manual step). `clip_reid`
additionally needs `pip install git+https://github.com/openai/CLIP.git` (see
`requirements.txt`) and is memory-heavy - use `--batch-size 64` on GPUs with < 16 GB.

Full method configs, exact recipes and deviations from the original papers are
documented in [`config.py`](config.py).

## Quick start

```bash
cd reid_benchmark

# functional check - tiny, verifies every method trains & scores
python train.py --splits 1 --num-epochs 2 --eval-every 1 --limit-classes 8 --output-dir runs_smoke

# reproduce the proposed model (all 5 splits, full schedule)
python train.py --methods proposed

# full benchmark: 8 methods × 5 splits
python train.py

# re-evaluate a saved checkpoint
python evaluate.py --run-dir runs/thorax/proposed/split_1 \
    --dataset-root "../datasets/re_identification/re_identification_dataset_only_thorax"
```

`--dataset-root` defaults to `../datasets/re_identification/re_identification_dataset_only_thorax`.

### Useful flags

| flag | default | meaning |
|------|---------|---------|
| `--methods` | all 8 | subset of methods to run |
| `--splits` | `1 2 3 4 5` | which folds |
| `--num-epochs` | `200` | training epochs |
| `--eval-every` | `10` | evaluation cadence (epochs) |
| `--batch-size` | `128` | - |
| `--part` | `thorax` | body-part token filtered in file names (`thorax` / `head` / `beetle`) |
| `--oversampling` | off | enable train-set class oversampling |
| `--best-metric` | `mean_average_precision_at_r` | validation metric for best-epoch selection |
| `--limit-classes` | - | use only the first *N* beetles (smoke tests) |
| `--no-cuda` | - | force CPU |

## Output layout

```
runs/<part>/
├── benchmark_summary.csv          # one row per (method, split): best epoch + val/test metrics
├── benchmark_summary.md           # test metrics, mean ± std over splits, per method
└── <method>/split_<k>/
    ├── params.txt                 # published-style key=value settings
    ├── inventory_train.json       # full training config, dataset stats, env info
    ├── inventory_eval.json        # per-epoch history, best epoch, selected test metrics
    ├── results_eval_{val,test}.txt
    ├── label_to_num.pkl
    ├── model_best_epoch.pt
    └── model_last_epoch.pt
```

## Tests

```bash
cd reid_benchmark
python -m pytest
```

29 tests (`test_data.py`, `test_models_losses.py`, `test_train_smoke.py`), run on CPU
with `pretrained=False`. `clip_reid` cases skip automatically if the CLIP package isn't
installed.
