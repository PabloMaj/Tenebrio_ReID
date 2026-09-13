import numpy as np
import pytest

from mobility.geometry import (
    beetle_rotation_angle,
    euclidean,
    mask_bbox,
    mask_centroid,
)


def test_mask_bbox_and_centroid(rect_mask):
    m = rect_mask(10, 20, 40, 60)
    assert mask_bbox(m) == (10, 20, 39, 59)
    assert mask_centroid(m) == [24, 39]


def test_euclidean():
    assert euclidean((0, 0), (3, 4)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "head, abdomen, expected",
    [
        ([0, 0], [10, 0], 0.0),     # head at smaller row than abdomen
        ([10, 0], [0, 0], 180.0),   # head at larger row than abdomen
        ([0, 10], [0, 0], 90.0),    # same row, head to the right
        ([0, 0], [0, 10], -90.0),   # same row, head to the left
    ],
)
def test_beetle_rotation_angle(head, abdomen, expected):
    assert beetle_rotation_angle(head, abdomen) == pytest.approx(expected)


def test_rotation_angle_is_finite_for_diagonal():
    a = beetle_rotation_angle([5, 5], [0, 0])
    assert np.isfinite(a)
