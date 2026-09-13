"""Head / abdomen instance segmentation of a full 3000x4096 frame.

The frame is cut into overlapping 512x512 tiles, each tile is run through the
YOLOv8-seg model, detections near tile borders are dropped (they are recovered
from the neighbouring tile) and the surviving masks are lifted back into
full-frame coordinates.  This mirrors ``FrameImg`` / ``Tile`` / ``Model`` in
``phenotyping_pipeline.py``.
"""
from __future__ import annotations

import cv2
import numpy as np

from .parts import Part

_CLASS_NAMES = {0: "head", 1: "thorax", 2: "down"}


class HeadThoraxSegmenter:
    def __init__(self, weights_path, config):
        from ultralytics import YOLO

        self.cfg = config
        self.model = YOLO(str(weights_path))
        self.roi = config.roi_size

    # -- low level -------------------------------------------------------- #
    def _predict_tile(self, roi_bgr: np.ndarray):
        """Return ``(masks, boxes_xyxy, confs, classes, midpoints)`` for one tile.

        ``masks`` are ``(N, roi, roi)`` float in {0,1}; ``midpoints`` are
        ``[col_center, row_center]`` (tile coordinates), as in the original code.
        """
        result = self.model(roi_bgr, verbose=False)[0]
        if result.masks is None or len(result.boxes) == 0:
            return None
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != (self.roi, self.roi):
            masks = np.stack(
                [cv2.resize(m, (self.roi, self.roi), interpolation=cv2.INTER_NEAREST) for m in masks]
            )
        masks = (masks > 0.5).astype(np.uint8)
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        midpoints = np.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
        )
        return masks, boxes, confs, classes, midpoints

    def _keep_border_filter(self, midpoints, confs, x_start, y_start):
        """Replicates ``Tile.detect_objects`` border + confidence filtering."""
        roi, cols, rows = self.roi, self.cfg.frame_cols, self.cfg.frame_rows
        keep = []
        for (mx, my), conf in zip(midpoints, confs):
            xmin = (mx >= roi / 8) or (y_start == 0)
            xmax = (mx < roi * 7 / 8) or (y_start + roi >= cols)
            ymin = (my >= roi / 8) or (x_start == 0)
            ymax = (my < roi * 7 / 8) or (x_start + roi >= rows)
            keep.append(bool(xmin and xmax and ymin and ymax and conf >= self.cfg.seg_conf_threshold))
        return np.array(keep, dtype=bool)

    def _to_frame(self, mask, box_xyxy, x_start, y_start):
        rows, cols, roi = self.cfg.frame_rows, self.cfg.frame_cols, self.roi
        x_end, y_end = min(x_start + roi, rows), min(y_start + roi, cols)
        full = np.zeros((rows, cols), dtype=np.uint8)
        full[x_start:x_end, y_start:y_end] = mask[: x_end - x_start, : y_end - y_start]
        x1, y1, x2, y2 = box_xyxy  # tile coords, (col, row) order
        box = (int(y1 + x_start), int(x1 + y_start), int(y2 + x_start), int(x2 + y_start))
        return full, box

    # -- public --------------------------------------------------------- #
    def segment(self, img_bgr: np.ndarray) -> tuple[list[Part], list[Part]]:
        """Return ``(heads, abdomens)`` as full-frame :class:`Part` lists."""
        roi = self.roi
        step = int(roi * (1 - self.cfg.roi_overlap))
        rows, cols = img_bgr.shape[:2]
        parts: list[Part] = []

        for x_start in range(0, rows, step):
            for y_start in range(0, cols, step):
                x_end, y_end = min(x_start + roi, rows), min(y_start + roi, cols)
                tile = np.zeros((roi, roi, 3), dtype=img_bgr.dtype)
                tile[: x_end - x_start, : y_end - y_start] = img_bgr[x_start:x_end, y_start:y_end]

                out = self._predict_tile(tile)
                if out is None:
                    continue
                masks, boxes, confs, classes, midpoints = out
                keep = self._keep_border_filter(midpoints, confs, x_start, y_start)
                for i in np.where(keep)[0]:
                    cls_name = _CLASS_NAMES.get(int(classes[i]))
                    if cls_name is None:
                        continue
                    full_mask, box = self._to_frame(masks[i], boxes[i], x_start, y_start)
                    area = int(full_mask.sum())
                    lo, hi = self.cfg.size_thresholds[cls_name]
                    if not (lo <= area <= hi):
                        continue
                    coords = np.argwhere(full_mask)
                    midpoint = [int(coords[:, 0].mean()), int(coords[:, 1].mean())]
                    parts.append(
                        Part(cls_name, full_mask, box, midpoint, float(confs[i]), area)
                    )

        heads = [p for p in parts if p.class_name == "head"]
        abdomens = [p for p in parts if p.class_name == "thorax"]
        return heads, abdomens
