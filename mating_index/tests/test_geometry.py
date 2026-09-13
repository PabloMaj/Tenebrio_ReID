import numpy as np

from mating.geometry import bbox_iou, frac_inside, from_full_mask, mask_iou


def _inst(r0, c0, r1, c1, shape=(40, 40)):
    m = np.zeros(shape, np.uint8)
    m[r0:r1, c0:c1] = 1
    return from_full_mask(m)


def test_from_full_mask_tight_box_and_area():
    inst = _inst(5, 10, 15, 20)
    assert inst.box == (5, 10, 14, 19)          # inclusive
    assert inst.area == 100
    assert inst.mask.shape == (10, 10)


def test_from_full_mask_empty_is_none():
    assert from_full_mask(np.zeros((8, 8), np.uint8)) is None


def test_mask_iou_half_overlap():
    a = _inst(0, 0, 10, 10)          # 100 px
    b = _inst(0, 5, 10, 15)          # 100 px, 50 shared
    assert abs(mask_iou(a, b) - 50 / 150) < 1e-9


def test_mask_iou_disjoint_is_zero():
    assert mask_iou(_inst(0, 0, 5, 5), _inst(20, 20, 30, 30)) == 0.0


def test_frac_inside_is_relative_to_first_arg():
    small = _inst(2, 2, 6, 6)        # 16 px, fully inside big
    big = _inst(0, 0, 20, 20)
    assert frac_inside(small, big) == 1.0
    assert frac_inside(big, small) == 16 / 400


def test_frac_inside_partial():
    a = _inst(0, 0, 10, 10)          # 100 px
    b = _inst(5, 0, 15, 10)          # overlap rows 5..9 -> 50 px of a
    assert abs(frac_inside(a, b) - 0.5) < 1e-9


def test_bbox_iou():
    assert abs(bbox_iou((0, 0, 9, 9), (0, 0, 9, 9)) - 1.0) < 1e-9
    assert bbox_iou((0, 0, 9, 9), (100, 100, 109, 109)) == 0.0
