import numpy as np

from mating.association import DSU, group_detections
from mating.geometry import from_full_mask


def _inst(r0, c0, r1, c1, shape=(60, 60)):
    m = np.zeros(shape, np.uint8)
    m[r0:r1, c0:c1] = 1
    return from_full_mask(m)


def test_dsu_basic():
    d = DSU()
    d.union("a", "b")
    d.union("b", "c")
    assert d.find("a") == d.find("c")
    assert d.find("a") != d.find("z")


def test_single_moving_pattern_is_one_group():
    dbf = {f: [(f"d{f}", _inst(0 + f, 0 + f, 20 + f, 20 + f))] for f in range(5)}
    groups = group_detections(dbf, iou_threshold=0.3, max_frame_gap=3)
    assert len(groups) == 1
    assert sorted(groups[0]) == [f"d{f}" for f in range(5)]


def test_gap_larger_than_max_breaks_the_group():
    dbf = {
        0: [("a0", _inst(0, 0, 20, 20))],
        1: [("a1", _inst(0, 0, 20, 20))],
        9: [("a9", _inst(0, 0, 20, 20))],   # 8 frames later -> not linked
    }
    groups = group_detections(dbf, iou_threshold=0.3, max_frame_gap=3)
    comps = sorted(sorted(v) for v in groups.values())
    assert comps == [["a0", "a1"], ["a9"]]


def test_two_patterns_same_frames_stay_separate():
    left = _inst(0, 0, 20, 20)
    right = _inst(0, 40, 20, 60)
    dbf = {f: [(f"L{f}", left), (f"R{f}", right)] for f in range(4)}
    groups = group_detections(dbf, iou_threshold=0.3, max_frame_gap=2)
    comps = sorted(sorted(v) for v in groups.values())
    assert comps == [[f"L{f}" for f in range(4)], [f"R{f}" for f in range(4)]]


def test_link_survives_one_missing_frame():
    a = _inst(0, 0, 20, 20)
    dbf = {0: [("a0", a)], 1: [("a1", a)], 3: [("a3", a)]}   # frame 2 missing, gap 2 <= 3
    groups = group_detections(dbf, iou_threshold=0.3, max_frame_gap=3)
    assert len(groups) == 1
