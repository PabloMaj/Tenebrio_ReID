import numpy as np
import pandas as pd
import pytest

from mobility.aggregation import (
    MobilityAggregator,
    compute_mobility_for_track,
    smooth_trajectory,
)

KW = dict(
    key="k", series="seria_2", identity="g_w",
    seconds_per_frame=1.0, k_mm_per_px=1.0, max_step_px=50.0, min_positions=3,
)


def test_straight_line_path_and_time():
    # 5 frames, +10 px in col each step -> path 40 px over 4 s
    recs = [(i, 0.0, 10.0 * i) for i in range(5)]
    res = compute_mobility_for_track(recs, **KW)
    assert res.path_len_px == pytest.approx(40.0)
    assert res.time_s == pytest.approx(4.0)
    assert res.mobility_px_per_s == pytest.approx(10.0)
    assert res.mobility_mm_per_s == pytest.approx(10.0)


def test_mm_scaling_and_5s_frames():
    recs = [(i, 0.0, 10.0 * i) for i in range(5)]  # 40 px over span 4
    res = compute_mobility_for_track(
        recs, key="k", series="s", identity="g", seconds_per_frame=5.0,
        k_mm_per_px=0.0943, max_step_px=750.0, min_positions=3,
    )
    assert res.time_s == pytest.approx(20.0)                     # 5 s * 4
    assert res.mobility_px_per_s == pytest.approx(2.0)           # 40 / 20
    assert res.mobility_mm_per_s == pytest.approx(2.0 * 0.0943)  # px/s * mm/px


def test_outlier_budget_scales_with_frame_gap():
    # last kept at frame 1 (x=5); frames 2..6 are dropped/absent; frame 7 lands
    # 700 px away.  With max_step_px=150, gap=6 -> budget 900 -> ACCEPTED.
    recs = [(0, 0.0, 0.0), (1, 0.0, 5.0), (7, 0.0, 705.0), (8, 0.0, 710.0)]
    res = compute_mobility_for_track(
        recs, key="k", series="s", identity="g", seconds_per_frame=1.0,
        k_mm_per_px=1.0, max_step_px=150.0, min_positions=3,
    )
    assert res.n_outliers == 0
    assert res.n_positions_used == 4
    # same jump one frame after the anchor (gap=1, budget 150) -> rejected
    res2 = compute_mobility_for_track(
        [(0, 0.0, 0.0), (1, 0.0, 5.0), (2, 0.0, 705.0), (3, 0.0, 10.0), (4, 0.0, 15.0)],
        key="k", series="s", identity="g", seconds_per_frame=1.0,
        k_mm_per_px=1.0, max_step_px=150.0, min_positions=3,
    )
    assert res2.n_outliers == 1 and res2.n_positions_used == 4  # frame 2 jump 700 > 150*1


def test_missing_data_at_end_shrinks_time():
    # seen on frames 0..4 then absent until frame 10
    recs = [(i, 0.0, 5.0 * i) for i in range(5)]
    res = compute_mobility_for_track(recs, **KW)
    assert res.last_idx == 4
    assert res.time_s == pytest.approx(4.0)  # not 10


def test_outlier_step_is_dropped():
    recs = [(0, 0.0, 0.0), (1, 0.0, 5.0), (2, 0.0, 500.0), (3, 0.0, 10.0), (4, 0.0, 15.0)]
    res = compute_mobility_for_track(recs, **KW)
    assert res.n_outliers == 1
    assert res.n_positions_used == 4
    assert res.path_len_px == pytest.approx(15.0)


def test_too_few_positions_returns_none():
    assert compute_mobility_for_track([(0, 0, 0), (1, 0, 1)], **KW) is None


def test_smoothing_preserves_straight_line():
    line = [(0.0, float(i)) for i in range(9)]        # perfectly linear
    sm = smooth_trajectory(line, "savgol", window=5, polyorder=2)
    assert np.allclose(np.asarray(sm), np.asarray(line), atol=1e-6)


def test_smoothing_removes_jitter_and_shortens_noisy_path():
    rng = np.random.default_rng(0)
    n = 40
    true = np.stack([np.zeros(n), np.linspace(0, 200, n)], axis=1)   # straight, 200 px
    noisy = [tuple(p) for p in true + rng.normal(0, 4.0, true.shape)]
    raw_len = sum(
        ((noisy[i][0] - noisy[i - 1][0]) ** 2 + (noisy[i][1] - noisy[i - 1][1]) ** 2) ** 0.5
        for i in range(1, n)
    )
    sm = smooth_trajectory(noisy, "savgol", window=7, polyorder=2)
    sm_len = sum(
        ((sm[i][0] - sm[i - 1][0]) ** 2 + (sm[i][1] - sm[i - 1][1]) ** 2) ** 0.5
        for i in range(1, n)
    )
    assert raw_len > 230           # jitter inflated the ~200 px path
    assert 195 < sm_len < raw_len  # smoothing pulls it back toward the truth


def test_track_reports_raw_and_smoothed_path():
    rng = np.random.default_rng(1)
    recs = [(i, float(rng.normal(0, 3)), 8.0 * i + float(rng.normal(0, 3))) for i in range(15)]
    res = compute_mobility_for_track(recs, **KW)
    assert res.path_len_px_raw >= res.path_len_px          # smoothing never lengthens jittery data
    assert res.mobility_px_per_s == pytest.approx(res.path_len_px / res.time_s, rel=1e-3)


def test_smoothing_none_is_identity():
    pts = [(1.0, 2.0), (3.0, 5.0), (0.0, 9.0), (7.0, 1.0)]
    assert smooth_trajectory(pts, "none", 5, 2) == pts


def test_aggregator_groups_and_drops_unknown(cfg):
    rows = []
    for i in range(4):
        rows.append(dict(series="seria_2", tag="g_w", frame_idx=i, x=10 * i, y=0))
        rows.append(dict(series="seria_2", tag="unknown", frame_idx=i, x=i, y=0))
    out = MobilityAggregator(cfg).aggregate(pd.DataFrame(rows), identity_col="tag")
    assert list(out["identity"]) == ["g_w"]
    assert out.iloc[0]["n_positions_used"] == 4
