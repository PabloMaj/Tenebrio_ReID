# Dynamic Phenotyping of Tenebrio molitor Beetles for Selective Breeding Using Re-Identification

A set of independent modules for automated analysis of mealworm beetles (*Tenebrio molitor*) from high-resolution photographs. Each module addresses a distinct computer vision task and can be run independently.

---

## Project Structure

```
tenebrio_reid_organized/
├── datasets/
│   ├── re_identification/
│   │   ├── re_identification_dataset/         # Cropped beetle parts by individual ID
│   │   └── re_identification_dataset_only_thorax/
│   ├── segmentation_head_thorax/
│   │   ├── ROIs/                              # 512×512 crops with JSON annotations
│   │   └── datasets/                          # YOLO-format (train/val/test)
│   ├── occlusion_classification/
│   │   ├── samples/                           # Raw labelled crops
│   │   └── dataset/                           # Split into train/val/test
│   ├── marker_recognition/
│   │   ├── marker_colors_samples/             # Per-colour pixel samples
│   │   └── example_samples/                   # JSON-annotated images
│   ├── mating_pattern_detection/               # Used by mating_index/ to train the detector
│   │   ├── ROIs/                              # 640×640 crops with mating annotations
│   │   ├── datasets/                          # YOLO-format
│   │   └── synthetic_data/                    # Generated mating pairs
│   ├── raw/                                   # Sample raw frame series (seria_2..5)
│   └── additional/
│       └── mating_phenotyping_rois/           # One hand-annotated ROI rectangle per series
├── models/                                    # Small pretrained checkpoints (tracked in git)
│   ├── marker_svm/                            # HSV/SVM colour-marker classifier
│   └── head_thorax_seg/                       # YOLOv8n-seg head/thorax checkpoint
├── re_identification/
│   └── *.py
├── segmentation_head_thorax/
│   ├── data.yaml
│   └── *.py
├── occlusion_classification/
│   └── *.py
├── marker_recognition/
│   └── *.py
├── mobility_index/                            # Per-beetle mobility index pipeline
├── mating_index/                              # Per-beetle mating index pipeline
├── reid_benchmark/                            # Re-ID methods benchmark (train/eval)
├── requirements.txt
└── README.md
```

---

## Input Data (Dataset)

### Dataset availability - hosted on Mendeley Data

The dataset itself (~7 GB) is **not included in this git repository** - it is published
separately on **Mendeley Data**: **`<Mendeley Data DOI / link - TODO>`**.

To run anything in this repo:

1. Download the dataset from the Mendeley Data record above.
2. Extract it so it lands at `datasets/` at the repo root, matching the layout below
   (i.e. `datasets/re_identification/`, `datasets/raw/`, `datasets/additional/`, etc.
   sit directly under `tenebrio_reid_organized/datasets/`).

---

## Installation

```bash
# Windows
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

This installs the CPU build of PyTorch — every script works out of the box on any
machine. For GPU acceleration, check `nvidia-smi` for your driver's CUDA version and
reinstall accordingly, e.g. `pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu121`.

---

## Tests

```bash
pytest tests/test_repo_paths.py   # from the repo root
cd mobility_index  && pytest      # 51 tests
cd mating_index    && pytest      # 31 tests
cd reid_benchmark  && pytest      # 29 tests, incl. an end-to-end training smoke test (~6 min)
```

---

## Modules

---

### 1. Re-identification (`re_identification/`)

**Goal:** Train a CNN to produce n-dimensional embeddings that distinguish individual beetles using metric learning.

#### Key Files

| File | Description |
|------|-------------|
| `train_and_evaluate_re_identification.py` | Train a custom CNN with metric learning |
| `train_and_evaluate_standard_embeddings.py` | Train with transfer learning (EfficientNet, ResNet) |
| `train_and_evaluate_standard_embeddings_with_splitting.py` | Cross-validation variant |

#### Running Training

```bash
cd re_identification

# Custom CNN with TripletLoss
python train_and_evaluate_re_identification.py

# Pre-trained backbone
python train_and_evaluate_standard_embeddings.py
```

#### Dataset Structure

```
datasets/re_identification/re_identification_dataset/
├── seria_2/
│   ├── A1.g_w/
│   │   ├── isolated/
│   │   │   ├── seria_2_isolated_00001_beetle_A1.g_w.png
│   │   │   ├── seria_2_isolated_00001_head_A1.g_w.png
│   │   │   └── seria_2_isolated_00001_thorax_A1.g_w.png
│   │   └── free/
│   └── B1.g_y/
│       └── ...
├── seria_3/ ... seria_5/
```

---

### 2. Head and Thorax Segmentation (`segmentation_head_thorax/`)

**Goal:** Instance segmentation - delineate the head, thorax using YOLOv8.

#### Key Files

| File | Description |
|------|-------------|
| `train_seg_model.py` | Train YOLOv8-seg (n/s/m/l/x) on the head/thorax dataset |
| `data.yaml` | YOLO dataset configuration |

#### Running Training

```bash
cd segmentation_head_thorax
python train_seg_model.py
```

The script trains the model and immediately evaluates it on the test split.

---

### 3. Occlusion Classification (`occlusion_classification/`)

**Goal:** Binary classification - determine whether a beetle is occluded by another beetle or an arena element (*occluded* / *non_occluded*).

#### Key Files

| File | Description |
|------|-------------|
| `train_classifier.py` | Train classifier |

#### Running Training

```bash
cd occlusion_classification
python train_classifier.py
```

#### Dataset Structure

```
datasets/occlusion_classification/
├── samples/
│   ├── non_occluded/   # Fully visible beetles
│   └── occluded/       # Partially hidden beetles
└── dataset/
    ├── train/
    ├── val/
    └── test/
```

---

### 4. Marker Recognition (`marker_recognition/`)

**Goal:** Pixel-level semantic segmentation - classify each pixel as a colour marker (red, blue, yellow, gold, white) or background. Uses classical ML models (SVM, LDA, LogReg) in HSV/RGB/Lab colour spaces.

#### Key Files

| File | Description |
|------|-------------|
| `train_model_to_sem_seg_markers.py` | Train SVM/LDA/LogReg on colour pixel samples |
| `train_model_to_sem_seg_markers_with_splits.py` | Cross-validation variant (StratifiedKFold) |

#### Running Training

```bash
cd marker_recognition

# Train the best-performing model (HSV + SVM rbf)
python train_model_to_sem_seg_markers.py

# Cross-validation variant
python train_model_to_sem_seg_markers_with_splits.py
```

#### Dataset Structure

```
datasets/marker_recognition/marker_colors_samples/
├── background/
├── red/
├── blue/
├── yellow/
├── gold/
└── white/
```

---

### 5. Re-ID Methods Benchmark (`reid_benchmark/`)

**Goal:** Train and evaluate 8 re-identification embedding models (the proposed
MobileNetV2 + triplet loss, plus SOTA alternatives) under one identical protocol, for a
fair comparison.

```bash
cd reid_benchmark
python train.py --methods proposed   # or omit --methods for the full 8-method benchmark
```

Dataset root defaults to `../datasets/re_identification/re_identification_dataset_only_thorax`.
See [`reid_benchmark/README.md`](reid_benchmark/README.md) for the full method table,
output layout and test suite.

---

### 6. Mobility Index (`mobility_index/`)

**Goal:** A 5-stage pipeline (segmentation → re-identification → mobility estimation →
visualisation → reporting) that computes a per-beetle mobility index from the raw
`free`-stage frame series.

```bash
cd mobility_index
python scripts/run_all.py --series seria_2
```

Needs the `proposed` re-ID weights trained by `reid_benchmark` first. Raw frames are
read from `../datasets/raw/` (a sample ships with the repo); the head/thorax segmenter
and marker-colour SVM are bundled in `../models/` (the occlusion classifier is not
bundled and is off by default). See [`mobility_index/README.md`](mobility_index/README.md).

---

### 7. Mating Index (`mating_index/`)

**Goal:** Reuses the `mobility_index` models (segmentation, marker reading, re-ID
ensemble) plus a dedicated holdout-trained mating-pattern detector to compute a
per-beetle mating index (confirmed matings where the beetle was the male).

```bash
cd mating_index
python train.py --holdout-series all   # once, to train the 4 holdout detectors
python scripts/run_all.py --series seria_2
```

See [`mating_index/README.md`](mating_index/README.md) for the full pipeline, the
independence rule behind the holdout split, and the confirmation criteria.

---