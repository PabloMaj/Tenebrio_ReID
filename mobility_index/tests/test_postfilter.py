"""Stage-3 detection-fraction filter: keep a beetle only if it was seen on
>= min_detection_frac of the frames processed for its series."""
import pandas as pd

from mobility.postfilter import add_detection_fraction, apply_detection_filter


def _mob(**per_id):
    rows = []
    for ident, (series, n_used) in per_id.items():
        rows.append({"key": f"{series}::{ident}", "series": series, "identity": ident,
                     "n_positions_used": n_used, "n_positions_raw": n_used})
    return pd.DataFrame(rows)


NF = {"seria_2": 200, "seria_3": 40}


def test_fraction_is_n_used_over_series_frames():
    m = add_detection_fraction(_mob(a=("seria_2", 50), b=("seria_3", 20)), NF)
    fr = dict(zip(m["identity"], m["detection_frac"]))
    assert fr["a"] == 0.25            # 50 / 200
    assert fr["b"] == 0.5             # 20 / 40


def test_filter_25pct_threshold():
    m = _mob(keep=("seria_2", 60), edge=("seria_2", 50), drop=("seria_2", 49))
    kept, dropped = apply_detection_filter(m, NF, 0.25)
    assert set(kept["identity"]) == {"keep", "edge"}     # 50/200 = 0.25 passes (>=)
    assert set(dropped["identity"]) == {"drop"}


def test_zero_threshold_keeps_all():
    m = _mob(a=("seria_2", 1), b=("seria_3", 1))
    kept, dropped = apply_detection_filter(m, NF, 0.0)
    assert len(kept) == 2 and len(dropped) == 0
    assert "detection_frac" in kept.columns


def test_missing_series_count_gives_zero_fraction():
    m = _mob(x=("seria_9", 100))
    kept, dropped = apply_detection_filter(m, NF, 0.25)
    assert len(kept) == 0 and dropped.iloc[0]["detection_frac"] == 0.0


def test_count_col_switch():
    df = pd.DataFrame([{"key": "seria_2::a", "series": "seria_2", "identity": "a",
                        "n_positions_used": 40, "n_positions_raw": 60}])
    assert add_detection_fraction(df, NF, "n_positions_used")["detection_frac"][0] == 0.2
    assert add_detection_fraction(df, NF, "n_positions_raw")["detection_frac"][0] == 0.3
