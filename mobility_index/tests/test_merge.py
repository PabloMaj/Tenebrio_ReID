import numpy as np

from mobility.parts import Part, merge_heads_and_abdomens


def _part(cls, mask):
    coords = np.argwhere(mask)
    mid = [int(coords[:, 0].mean()), int(coords[:, 1].mean())]
    r0, c0 = coords.min(0)
    r1, c1 = coords.max(0)
    return Part(cls, mask, (int(r0), int(c0), int(r1), int(c1)), mid, 0.9, int(mask.sum()))


def test_adjacent_head_and_abdomen_are_merged(rect_mask):
    head = _part("head", rect_mask(40, 40, 60, 70))
    abdomen = _part("thorax", rect_mask(60, 40, 110, 70))  # touches head, on the connecting line
    img = np.zeros((200, 200, 3), np.uint8)
    beetles = merge_heads_and_abdomens(
        [head], [abdomen], metric_threshold=0.8, img_bgr=img,
        series="seria_2", frame_idx=1,
    )
    assert len(beetles) == 1
    assert beetles[0].abdomen_midpoint == tuple(abdomen.midpoint)


def test_far_apart_parts_are_not_merged(rect_mask):
    head = _part("head", rect_mask(10, 10, 25, 30))
    abdomen = _part("thorax", rect_mask(150, 150, 190, 190))
    img = np.zeros((200, 200, 3), np.uint8)
    beetles = merge_heads_and_abdomens(
        [head], [abdomen], metric_threshold=0.8, img_bgr=img,
        series="seria_2", frame_idx=1,
    )
    assert beetles == []


def test_best_abdomen_is_selected(rect_mask):
    head = _part("head", rect_mask(40, 40, 60, 70))
    good = _part("thorax", rect_mask(60, 40, 110, 70))
    bad = _part("thorax", rect_mask(60, 120, 110, 160))
    img = np.zeros((200, 200, 3), np.uint8)
    beetles = merge_heads_and_abdomens(
        [head], [bad, good], metric_threshold=0.8, img_bgr=img,
        series="seria_2", frame_idx=1,
    )
    assert len(beetles) == 1
    assert beetles[0].abdomen_midpoint == tuple(good.midpoint)
