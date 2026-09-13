"""Deterministic re-identification from the two colour markers on the head.

This is the *ground-truth* identity source for stage 1.  Faithful port of
``Beetle.determine_marker`` in ``phenotyping_pipeline.py`` with two changes:

* the per-pixel SVM call is vectorised (one ``predict`` for the whole head);
* clustering uses ``scipy.ndimage.label`` instead of ``skimage.measure.label``
  (avoids the extra dependency; identical result for our purpose).

The head/abdomen offset that the original code applied to cluster centroids
cancels out (it is added to *both* clusters and only their relative order is
used), so it is dropped here.
"""
from __future__ import annotations

import pickle

import cv2
import numpy as np
from scipy.ndimage import label as ndi_label

LABEL_TO_ID = {"background": 1, "red": 2, "blue": 3, "yellow": 4, "gold": 5, "white": 6}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
UNKNOWN_TAG = "unknown"


def _sector(head_mid, abdomen_mid) -> str:
    """Coarse orientation of the beetle (quadrants I-IV), points are [row, col]."""
    hr, hc = head_mid
    ar, ac = abdomen_mid
    if ar == hr:
        orientation = 90.0
    else:
        orientation = np.degrees(np.arctan((ac - hc) / (ar - hr)))
    if -45.0 <= orientation <= 45.0:
        return "I" if hr < ar else "II"
    return "III" if hc < ac else "IV"


def _order_tag(sector: str, lab1: int, lab2: int, c1, c2) -> str:
    """Colour-initial ordering rule from the original pipeline."""
    i1, i2 = ID_TO_LABEL[lab1][0], ID_TO_LABEL[lab2][0]
    if sector == "I":
        return f"{i1}_{i2}" if c1[1] < c2[1] else f"{i2}_{i1}"
    if sector == "II":
        return f"{i2}_{i1}" if c1[1] < c2[1] else f"{i1}_{i2}"
    if sector == "III":
        return f"{i2}_{i1}" if c1[0] < c2[0] else f"{i1}_{i2}"
    return f"{i1}_{i2}" if c1[0] < c2[0] else f"{i2}_{i1}"


class MarkerReader:
    def __init__(self, model_path, scaler_path):
        self.model = pickle.load(open(model_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))

    # -- pixel semantic segmentation ---------------------------------- #
    def _segment_markers(self, head_bgr_half: np.ndarray) -> np.ndarray:
        """Return an id map (``LABEL_TO_ID`` values, 0 outside the head)."""
        hsv = cv2.cvtColor(head_bgr_half, cv2.COLOR_BGR2HSV)
        fg = head_bgr_half.sum(axis=2) != 0
        id_map = np.zeros(hsv.shape[:2], dtype=np.int32)
        if fg.sum() == 0:
            return id_map
        feats = self.scaler.transform(hsv[fg].astype(np.float64))
        preds = self.model.predict(feats)
        id_map[fg] = [LABEL_TO_ID[p] for p in preds]
        return id_map

    # -- public ------------------------------------------------------ #
    def read_tag(self, head_crop_bgr: np.ndarray, head_midpoint, abdomen_midpoint) -> str:
        if head_crop_bgr.size == 0:
            return UNKNOWN_TAG
        h, w = head_crop_bgr.shape[:2]
        half = cv2.resize(head_crop_bgr, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
        id_map = self._segment_markers(half)

        # connected components *per colour* (skimage.measure.label with an int
        # image splits on value; scipy.ndimage.label does not, so we loop).
        clusters = []  # (size, id_label, centroid[row, col])
        for id_label in range(LABEL_TO_ID["background"] + 1, max(LABEL_TO_ID.values()) + 1):
            labelled, n = ndi_label(id_map == id_label)
            for cid in range(1, n + 1):
                m = labelled == cid
                coords = np.argwhere(m)
                clusters.append((int(m.sum()), id_label, coords.mean(axis=0)))

        if len(clusters) < 2:
            return UNKNOWN_TAG
        clusters.sort(key=lambda c: c[0], reverse=True)
        (_, lab1, c1), (_, lab2, c2) = clusters[0], clusters[1]
        return _order_tag(_sector(head_midpoint, abdomen_midpoint), lab1, lab2, c1, c2)
