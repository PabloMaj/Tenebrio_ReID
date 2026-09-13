"""Small geometry helpers.

Ported (with light clean-up) from
``re_identification/phenotyping_pipeline.py`` so this package has no
import-time dependency on that script.

Coordinate convention throughout the package: points are ``[row, col]`` (i.e.
``[y, x]`` in image terms), matching the output of
``FrameImg.convert_outcomes_to_frame_coords`` in the original pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate as ndi_rotate


def mask_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(row_min, col_min, row_max, col_max)`` of the non-zero region."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return int(row_min), int(col_min), int(row_max), int(col_max)


def mask_centroid(binary_mask: np.ndarray) -> list[int]:
    """Return the integer centroid ``[row, col]`` of a binary mask."""
    coords = np.argwhere(binary_mask != 0)
    row_mid, col_mid = coords.mean(axis=0)
    return [int(row_mid), int(col_mid)]


def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    return np.mean(rgb_image, axis=2)


def euclidean(p1, p2) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def angle_at_vertex(point1, vertex, point3) -> float:
    """Angle (degrees) of ``point1 - vertex - point3`` at ``vertex``."""
    v1 = np.asarray(point1, float) - np.asarray(vertex, float)
    v2 = np.asarray(point3, float) - np.asarray(vertex, float)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = min(max(cos, -1.0), 1.0)
    return float(np.degrees(np.arccos(cos)))


def beetle_rotation_angle(head_midpoint, abdomen_midpoint) -> float:
    """Rotation (degrees) that makes the head->abdomen axis vertical.

    Identical logic to ``Beetle.determine_rotation_angle`` in the original
    pipeline (points are ``[row, col]``).
    """
    hm, tm = head_midpoint, abdomen_midpoint
    if hm[0] == tm[0]:
        return 90.0 if hm[1] >= tm[1] else -90.0
    slope = (hm[1] - tm[1]) / (hm[0] - tm[0])
    angle = -np.degrees(np.arctan(slope))
    if hm[0] >= tm[0]:
        angle += 180.0
    return float(angle)


def rotation_normalise(rgb_masked: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a masked RGB crop by ``angle_degrees`` and tightly re-crop it.

    Mirrors the ``rotation_normalized`` branch of
    ``FrameImg.save_beetles_images`` – this is exactly how the re-ID gallery
    crops (``re_identification_dataset_only_thorax``) were produced.
    """
    rotated = ndi_rotate(rgb_masked, angle_degrees, reshape=True)
    gray = (rgb_to_gray(rotated) >= 1).astype("int")
    if gray.sum() == 0:
        return rotated
    r0, c0, r1, c1 = mask_bbox(gray)
    return rotated[r0:r1, c0:c1, :]
