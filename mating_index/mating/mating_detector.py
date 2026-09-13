"""Mating-pattern instance segmentation of a raw ``free/`` frame.

The frame's phenotyping ROI is cut into overlapping ``tile_size`` tiles (step =
``tile_size * (1 - tile_overlap)`` = 320 px, as in ``extract_ROIs.py``); every
tile is run through the **holdout** ``yolov8n-seg`` model for that series, the
masks are lifted back to full-frame coordinates and detections of the same
pattern coming from overlapping tiles are merged by mask IoU.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mating.geometry import MaskInstance, from_full_mask


def load_roi_box(cfg, series: str, frame_shape) -> tuple:
    """``(row_min, col_min, row_max, col_max)`` phenotyping ROI for a series."""
    rows, cols = frame_shape[:2]
    p = Path(cfg.roi_json(series))
    if series.startswith("seria_1") or not p.exists():
        return 0, 0, rows, cols
    data = json.load(open(p))
    pts = data["shapes"][0]["points"]              # labelme convention: [x, y]
    xs = sorted(int(round(pt[0])) for pt in pts)
    ys = sorted(int(round(pt[1])) for pt in pts)
    return (max(ys[0], 0), max(xs[0], 0), min(ys[-1], rows), min(xs[-1], cols))


class MatingDetector:
    def __init__(self, weights_path, cfg):
        from ultralytics import YOLO

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"mating detector weights missing:\n  {weights_path}\n"
                f"train it first:  python mating_index/train.py --holdout-series <series>"
            )
        self.cfg = cfg
        self.model = YOLO(str(weights_path))
        self.tile = int(cfg.tile_size)

    # -- low level ---------------------------------------------------- #
    def _predict_tile(self, tile_bgr):
        res = self.model(
            tile_bgr, verbose=False, conf=self.cfg.mating_conf_threshold,
            device=self.cfg.device,
        )[0]
        if res.masks is None or len(res.boxes) == 0:
            return []
        masks = res.masks.data.cpu().numpy()
        if masks.shape[1:] != (self.tile, self.tile):
            masks = np.stack([
                cv2.resize(m, (self.tile, self.tile), interpolation=cv2.INTER_NEAREST)
                for m in masks
            ])
        confs = res.boxes.conf.cpu().numpy()
        return [((m > 0.5).astype(np.uint8), float(c)) for m, c in zip(masks, confs)]

    # -- public ----------------------------------------------------- #
    def detect(self, img_bgr: np.ndarray, series: str) -> list[MaskInstance]:
        rows, cols = img_bgr.shape[:2]
        r0, c0, r1, c1 = load_roi_box(self.cfg, series, img_bgr.shape)
        step = max(int(self.tile * (1 - self.cfg.tile_overlap)), 1)
        raw: list[MaskInstance] = []
        for y in range(r0, max(r1 - 1, r0 + 1), step):
            for x in range(c0, max(c1 - 1, c0 + 1), step):
                ye, xe = min(y + self.tile, rows), min(x + self.tile, cols)
                tile = np.zeros((self.tile, self.tile, 3), dtype=img_bgr.dtype)
                tile[: ye - y, : xe - x] = img_bgr[y:ye, x:xe]
                for m, conf in self._predict_tile(tile):
                    full = np.zeros((rows, cols), dtype=np.uint8)
                    full[y:ye, x:xe] = m[: ye - y, : xe - x]
                    inst = from_full_mask(full, conf=conf, series=series)
                    if inst is not None and inst.area > 0:
                        raw.append(inst)
        return self._merge(raw, (rows, cols))

    def _merge(self, insts: list[MaskInstance], shape) -> list[MaskInstance]:
        """Union masks whose IoU >= ``mating_iou_merge`` (same pattern, two tiles)."""
        insts = sorted(insts, key=lambda i: i.meta.get("conf", 0.0), reverse=True)
        clusters: list[dict] = []
        for cand in insts:
            cf = cand.full_mask(shape)
            for cl in clusters:
                inter = int(np.logical_and(cf, cl["full"]).sum())
                if inter == 0:
                    continue
                union = int(np.logical_or(cf, cl["full"]).sum())
                if union and inter / union >= self.cfg.mating_iou_merge:
                    cl["full"] = np.logical_or(cf, cl["full"]).astype(np.uint8)
                    cl["conf"] = max(cl["conf"], cand.meta.get("conf", 0.0))
                    break
            else:
                clusters.append({"full": cf, "conf": float(cand.meta.get("conf", 0.0)),
                                 "series": cand.meta.get("series", "")})
        out = []
        for cl in clusters:
            inst = from_full_mask(cl["full"], conf=cl["conf"], series=cl["series"])
            if inst is not None:
                out.append(inst)
        out.sort(key=lambda i: (i.box[0], i.box[1]))
        return out
