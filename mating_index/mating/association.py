"""Step (4a): link mating masks across *neighbouring* frames into groups.

Two mating detections are linked when they are on frames whose index distance is
``<= max_frame_gap`` and their masks overlap with ``IoU >= iou_threshold``.
Groups are the connected components of that graph (union-find), so a pattern
that is missed on one or two frames still stays a single group.  Several mating
patterns on the same frame are handled naturally - they land in different
components.
"""
from __future__ import annotations

from collections import defaultdict

from mating.geometry import mask_iou


class DSU:
    def __init__(self):
        self._p: dict = {}

    def find(self, x):
        self._p.setdefault(x, x)
        root = x
        while self._p[root] != root:
            root = self._p[root]
        while self._p[x] != root:
            self._p[x], x = root, self._p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def group_detections(dets_by_frame: dict, iou_threshold: float, max_frame_gap: int) -> dict:
    """``dets_by_frame``: ``{frame_idx: [(det_id, MaskInstance), ...]}``.

    Returns ``{group_id: [det_id, ...]}`` with ``group_id`` a small int, groups
    ordered by their earliest ``det_id``.
    """
    dsu = DSU()
    frames = sorted(dets_by_frame)
    for i, f in enumerate(frames):
        for did, _ in dets_by_frame[f]:
            dsu.find(did)
        for g in frames[i + 1:]:
            if g - f > max_frame_gap:
                break
            for did_a, ma in dets_by_frame[f]:
                for did_b, mb in dets_by_frame[g]:
                    if mask_iou(ma, mb) >= iou_threshold:
                        dsu.union(did_a, did_b)

    comps: dict = defaultdict(list)
    for f in frames:
        for did, _ in dets_by_frame[f]:
            comps[dsu.find(did)].append(did)
    ordered = sorted((sorted(v) for v in comps.values()), key=lambda v: v[0])
    return {i: v for i, v in enumerate(ordered)}
