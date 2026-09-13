import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG  # noqa: E402


@pytest.fixture
def cfg():
    return CONFIG.with_overrides(
        seconds_per_frame=1.0,
        k_mm_per_px=1.0,
        max_step_px=50.0,
        min_positions_for_mobility=3,
    )


@pytest.fixture
def rect_mask():
    """Factory: filled rectangle mask on a 200x200 canvas."""

    def _make(r0, c0, r1, c1, shape=(200, 200)):
        m = np.zeros(shape, dtype=np.uint8)
        m[r0:r1, c0:c1] = 1
        return m

    return _make
