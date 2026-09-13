# Beetle Mobility Index (Tenebrio molitor)

Per-individual **mobility index** computed from the free-movement (`free/`) image
series seria_2–5: summed euclidean displacement of the abdomen centre between
consecutive frames, divided by recording time, smoothed (Savitzky-Golay) before
path length is measured.

```
mobility = Σ ||p_i - p_{i-1}||  /  (seconds_per_frame * (last_seen - first_seen))
```

## Pipeline

| Stage | What | Output |
|---|---|---|
| 1 | Segment head+abdomen, read marker tag → GT mobility | `outputs/stage1/mobility_gt.csv` |
| 2 | Same, but identity from a **5-model re-ID ensemble** (≥3 models agree, cos≥0.6) → predicted mobility | `outputs/stage2/mobility_pred_ensemble.csv`, `reid_inferences.csv` (every model's raw prediction) |
| 3 | Detection-fraction filter (≥25% of frames), match stage 2 to stage 1, RMSE/MAE/pRMSE/MAPE | `outputs/stage3/summary_metrics.csv`, `mobility_scatter.png` |
| 4 | Example trajectories (low/med/high mobility) | `outputs/stage4/example_paths_ensemble.png` |
| 5 | Mobility distribution histogram | `outputs/stage5/mobility_histogram.png` |

The 5 re-ID models are `reid_benchmark`'s `proposed` method, used exactly as trained
(same architecture, weights, preprocessing - imported directly from `reid_benchmark`).

## Quick start

```bash
# from the repo root: pip install -r requirements.txt
cd mobility_index
python -m pytest                                          # unit tests, no weights needed

# smoke test
python scripts/run_stage1.py --series seria_2 --max-frames 3
python scripts/run_stage2.py --series seria_2 --max-frames 3

# full run (all 4 series; use --frame-stride N for a faster approximate pass)
python scripts/run_all.py
python scripts/run_all.py --frame-stride 3
```

Needs the `proposed` re-ID weights (`reid_benchmark/train.py --methods proposed`)
and the bundled `models/` checkpoints (segmenter, marker SVM - see the top-level
README). Each stage also has its own script (`run_stage1.py` … `run_stage5.py`).

## Manual exclusion

**On by default**: detections manually reviewed as bad (mostly mis-read marker
tags) are dropped, read from the bundled `manual_exclusions/exclusions.csv` -
reproduces the published run out of the box. Disable with
`--no-manual-exclusions`, or point `--exclude-dir` at your own folder/CSV to
review from scratch (see `mobility/exclude.py`).

## Key parameters (`config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `seconds_per_frame`, `k_mm_per_px` | `5.0`, `0.0943` | time/frame [s], spatial scale [mm/px] |
| `max_step_px` | `750` | max abdomen displacement between frames (scales with frame gap) |
| `traj_smooth(_window)` | `savgol`, `5` | trajectory smoothing before path length |
| `min_detection_frac` | `0.25` | drop a beetle seen on less than this fraction of its series' frames |
| `reid_ensemble_min_agree` / `_conf_threshold` | `3` / `0.6` | ensemble acceptance rule |
| `reid_candidate_scope` | `global` | match against all 80 individuals, or same series only |
| `use_occlusion` | `False` | opt-in occlusion classifier (not bundled; train your own) |
| `use_manual_exclusions` | `True` | see Manual exclusion above |

Raw frames are read from `../datasets/raw/` (override with `TENEBRIO_EXPERIMENT_ROOT`
for the full archive). `reid_inferences.csv` logs every model's raw prediction, so
`scripts/run_reid_sweep.py` can re-score any `(min_agree, conf_threshold)` offline
without re-running the models.

## Layout

```
mobility_index/
├── config.py       # MobilityConfig - single source of paths/params
├── mobility/        # pipeline stages, geometry, re-ID, aggregation
├── scripts/         # run_stage{1..5}.py, run_all.py, dump_crops.py, reaggregate.py
└── tests/
```
