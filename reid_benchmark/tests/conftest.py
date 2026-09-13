import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402


def _make_png(path, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """A minimal beetle re-ID dataset tree:  <serie>/<beetle>/{isolated,free}/*.png"""
    root = tmp_path_factory.mktemp("reid_ds")
    series = ["seria_2", "seria_3", "seria_4", "seria_5"]
    beetles = ["A1.g_w", "A2.y_b", "A3.b_g", "A4.b_r"]
    n_iso, n_free = 12, 5
    counter = 0
    for si, serie in enumerate(series):
        for bi, beetle in enumerate(beetles):
            for stage, n in (("isolated", n_iso), ("free", n_free)):
                d = root / serie / beetle / stage
                d.mkdir(parents=True, exist_ok=True)
                for k in range(n):
                    counter += 1
                    name = f"{k:05d}_{k % 7 + 1}_thorax_{beetle[-3:]}.png"
                    _make_png(str(d / name), seed=counter)
                    # also a non-thorax file that must be ignored
                    _make_png(str(d / f"{k:05d}_{k}_head_{beetle[-3:]}.png"), seed=counter + 9000)
    return str(root)
