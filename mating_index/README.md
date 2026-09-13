# Beetle Mating Index (Tenebrio molitor)

A per-individual **mating index** = number of *confirmed* mating patterns in which a
beetle was the **male** (mounted on top), computed from the `free/` image series
seria_2-5. Reuses [`mobility_index/`](../mobility_index)'s head/thorax segmenter,
marker reader and re-ID ensemble; only the mating-pattern detector is trained here.

## Pipeline

| Stage | What | Key output |
|---|---|---|
| 1 - detect | tiled `yolov8n-seg` holdout model over the phenotyping ROI | `outputs/stage1/mating_detections.csv` |
| 2 - bodies | pick the male abdomen (largest mask ≥75% inside the mating mask), read its marker tag | `outputs/stage2/mating_bodies.csv` |
| 3 - reid | re-identify the male abdomen with the 5 `proposed` models + ensemble | `outputs/stage3/mating_bodies_id.csv` |
| 4 - confirm | group mating masks across frames (IoU union-find), apply the confirmation rule | `outputs/stage4/confirmed_matings.csv` |
| 5 - report | mating index per individual, histogram, overlays | `outputs/stage5/mating_index.csv` |

A group is **confirmed** when it spans ≥ `min_group_frames` frames and the modal
re-ID identity covers ≥ `id_consistency` of the *identified* detections (≥
`min_identified` of them).

## Quick start

```bash
# from the repo root: pip install -r requirements.txt
cd mating_index
python -m pytest                       # 31 tests, pure logic, no weights
```

### 1. Train the models (once — needs only the bundled dataset)

```bash
# reid_benchmark/: the proposed re-ID model (5 splits)
cd ../reid_benchmark && python train.py --methods proposed

# mating_index/: one holdout yolov8n-seg detector per evaluation series
cd ../mating_index && python train.py --holdout-series all --epochs 200 --batch 8
```

### 2. Compute the mating index

```bash
python scripts/run_all.py --series seria_2 seria_3 seria_4
```

Re-runs reuse `outputs/cache/`; pass `--no-cache` to refresh. Each stage also has its
own script (`scripts/run_stage{1..5}_*.py`) — re-running stage 4+5 only (after tuning
a confirmation threshold) needs no models.

## Layout

```
mating_index/
├── mating/          # config.py, pipeline stages, geometry, splits
├── scripts/         # run_stage{1..5}_*.py, run_all.py
├── train.py         # holdout yolov8n-seg training
└── tests/
```
