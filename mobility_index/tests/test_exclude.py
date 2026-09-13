"""Manual sample exclusion: parse crop filenames / CSV, drop those detections."""
import pandas as pd

from mobility.exclude import _parse_path, filter_inventory, load_exclusions
from pathlib import Path


def test_parse_crop_filename():
    assert _parse_path(Path("x/seria_2/g_w/seria_2_00045_b3_head.png")) == ("seria_2", 45, 3)
    assert _parse_path(Path("seria_5_00007_b12.png")) == ("seria_5", 7, 12)
    assert _parse_path(Path("random.png")) is None


def test_load_from_folder_and_stage_scoping(tmp_path):
    # user's real layout: <series>/manually_checked/<tag>/<frame>_b<beetle>.png
    d = tmp_path / "seria_2" / "manually_checked" / "b_g"
    d.mkdir(parents=True)
    (d / "00027_b6.png").write_bytes(b"x")                                 # bare -> default stage1
    (tmp_path / "stage2").mkdir()
    (tmp_path / "stage2" / "seria_3_00020_b2_abdomen.png").write_bytes(b"x")  # -> stage2 only
    ex = load_exclusions(tmp_path)                                         # default_stage="stage1"
    assert ("seria_2", 27, 6) in ex["stage1"] and ("seria_2", 27, 6) not in ex["stage2"]
    assert ("seria_3", 20, 2) in ex["stage2"] and ("seria_3", 20, 2) not in ex["stage1"]


def test_default_stage_override(tmp_path):
    d = tmp_path / "seria_4" / "x"
    d.mkdir(parents=True)
    (d / "00099_b5.png").write_bytes(b"x")
    assert load_exclusions(tmp_path, "both")["stage1"] == {("seria_4", 99, 5)}
    assert load_exclusions(tmp_path, "both")["stage2"] == {("seria_4", 99, 5)}


def test_load_from_csv(tmp_path):
    pd.DataFrame({"series": ["seria_4"], "frame_idx": [99], "beetle_idx": [5], "stage": ["stage2"]}) \
        .to_csv(tmp_path / "bad.csv", index=False)
    ex = load_exclusions(tmp_path)
    assert ex["stage2"] == {("seria_4", 99, 5)} and ex["stage1"] == set()


def test_filter_inventory():
    df = pd.DataFrame({
        "series": ["seria_2"] * 3, "frame_idx": [1, 2, 3], "beetle_idx": [1, 1, 2],
        "tag": ["g_w"] * 3, "x": [0, 10, 20], "y": [0, 0, 0],
    })
    kept, dropped = filter_inventory(df, {("seria_2", 2, 1)})
    assert list(kept["frame_idx"]) == [1, 3]
    assert list(dropped["frame_idx"]) == [2]


def test_empty_exclusions_is_noop():
    df = pd.DataFrame({"series": ["s"], "frame_idx": [1], "beetle_idx": [1]})
    kept, dropped = filter_inventory(df, set())
    assert len(kept) == 1 and len(dropped) == 0
