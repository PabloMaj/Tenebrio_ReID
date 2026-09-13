"""Stage runners for the mating-index pipeline.

    stage 1  detect      mating masks on every raw frame              -> stage1/
    stage 2  bodies       male abdomen (mask-based) + head marker tag  -> stage2/
    stage 3  reid         5-model ``proposed`` ensemble on the male    -> stage3/
    stage 4  confirm       cross-frame IoU groups + confirmation rule   -> stage4/
    stage 5  report        mating index per individual + histogram      -> stage5/
"""
from __future__ import annotations

import cv2
import pandas as pd
from tqdm import tqdm

from mating import visualization as viz
from mating._mobility import HeadThoraxSegmenter, InventoryWriter, MarkerReader, ReIDEnsemble
from mating.aggregation import confirm_groups, mating_index
from mating.association import group_detections
from mating.bodies import resolve_body
from mating.frame_pipeline import MatingFramePipeline, det_id, load_frame_cache
from mating.mating_detector import MatingDetector


def list_frames(cfg, series: str):
    folder = cfg.frames_dir(series)
    frames = sorted((int(p.stem), p) for p in folder.glob("*.jpg") if p.stem.isdigit())
    frames = frames[:: cfg.frame_stride]
    if cfg.max_frames:
        frames = frames[: cfg.max_frames]
    return frames


class _LazyPipeline:
    """Build the YOLO models only on the first frame-cache miss."""

    def __init__(self, cfg, series):
        self.cfg, self.series = cfg, series
        self._pipe = None

    def get(self):
        if self._pipe is None:
            mob = self.cfg.mobility
            self._pipe = MatingFramePipeline(
                self.cfg,
                MatingDetector(self.cfg.mating_weights(self.series), self.cfg),
                HeadThoraxSegmenter(mob.seg_weights, mob),
            )
        return self._pipe


def _frame_records(cfg, series, use_cache, desc):
    """Yield ``(frame_idx, path, frame_dict)`` - segmenting only on cache miss."""
    lazy = _LazyPipeline(cfg, series)
    for fi, path in tqdm(list_frames(cfg, series), desc=desc):
        if use_cache:
            cached = load_frame_cache(cfg, series, fi)
            if cached is not None:
                yield fi, path, cached
                continue
        img = cv2.imread(str(path))
        yield fi, path, lazy.get().process_frame(series, fi, img, use_cache=use_cache)


# --------------------------------------------------------------------------- #
# stage 1                                                                      #
# --------------------------------------------------------------------------- #
class Stage1Detect:
    def __init__(self, cfg, use_cache=True):
        self.cfg, self.use_cache = cfg, use_cache

    def run(self, series=None):
        cfg = self.cfg
        inv = InventoryWriter(cfg.stage_dir(1) / "mating_detections.csv")
        for s in series or cfg.series:
            drawn = 0
            for fi, path, fr in _frame_records(cfg, s, self.use_cache, f"stage1 {s}"):
                for j, m in enumerate(fr["matings"], 1):
                    m.meta.setdefault("series", s)
                    r0, c0, r1, c1 = m.box
                    inv.add(
                        series=s, frame_idx=fi, det_id=det_id(s, fi, j),
                        bbox_r0=r0, bbox_c0=c0, bbox_r1=r1, bbox_c1=c1,
                        mask_area=m.area, conf=round(float(m.meta.get("conf", 0.0)), 4),
                        n_heads=len(fr["heads"]), n_abdomens=len(fr["abdomens"]),
                    )
                if fr["matings"]:
                    img = cv2.imread(str(path))
                    if viz.draw_stage1(cfg, s, fi, img, fr["matings"], drawn):
                        drawn += 1
        return inv.save()


# --------------------------------------------------------------------------- #
# stage 2                                                                      #
# --------------------------------------------------------------------------- #
class Stage2Bodies:
    def __init__(self, cfg, use_cache=True):
        self.cfg, self.use_cache = cfg, use_cache
        mob = cfg.mobility
        self.markers = MarkerReader(mob.marker_model, mob.marker_scaler)

    def run(self, series=None):
        cfg = self.cfg
        inv = InventoryWriter(cfg.stage_dir(2) / "mating_bodies.csv")
        for s in series or cfg.series:
            crop_dir = cfg.stage_dir(2) / "crops" / s
            crop_dir.mkdir(parents=True, exist_ok=True)
            drawn = 0
            for fi, path, fr in _frame_records(cfg, s, self.use_cache, f"stage2 {s}"):
                if not fr["matings"]:
                    continue
                img = cv2.imread(str(path))
                items = []
                for j, m in enumerate(fr["matings"], 1):
                    m.meta.setdefault("series", s)
                    did = det_id(s, fi, j)
                    body = resolve_body(m, fr["heads"], fr["abdomens"], img, cfg, self.markers)
                    items.append({"mating": m, "body": body, "det_id": did})
                    row = dict(series=s, frame_idx=fi, det_id=did, resolved=int(body is not None))
                    if body is not None:
                        crop = body.pop("abdomen_crop_bgr")
                        cp = crop_dir / f"{did}.png"
                        if crop.size:
                            cv2.imwrite(str(cp), crop)
                        row.update(
                            marker_tag=body["marker_tag"],
                            has_head=int(body["has_head"]),
                            frac_in_mating=body["frac_in_mating"],
                            abdomen_area=body["abdomen_area"],
                            abdomen_y=round(body["abdomen_centroid"][0], 1),
                            abdomen_x=round(body["abdomen_centroid"][1], 1),
                            rotation_angle=body["rotation_angle"],
                            merge_metric=body["merge_metric"],
                            crop=str(cp.relative_to(cfg.output_dir)),
                        )
                    inv.add(**row)
                if viz.draw_stage2(cfg, s, fi, img, items, drawn):
                    drawn += 1
        return inv.save()


# --------------------------------------------------------------------------- #
# stage 3                                                                      #
# --------------------------------------------------------------------------- #
class Stage3ReID:
    def __init__(self, cfg, build_gallery=True):
        self.cfg = cfg
        self.ensemble = ReIDEnsemble(cfg.mobility).load(build_gallery=build_gallery)

    def run(self, series=None):
        cfg = self.cfg
        bodies = pd.read_csv(cfg.stage_dir(2) / "mating_bodies.csv", sep=";")
        infer = InventoryWriter(cfg.stage_dir(3) / "reid_inferences.csv")
        verdict = InventoryWriter(cfg.stage_dir(3) / "mating_bodies_id.csv")
        sub = bodies[bodies["resolved"] == 1]
        if series:
            sub = sub[sub["series"].isin(series)]
        for r in tqdm(list(sub.itertuples()), desc="stage3 reid"):
            crop_path = cfg.output_dir / r.crop
            crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
            if crop is None or crop.size == 0:
                verdict.add(series=r.series, frame_idx=r.frame_idx, det_id=r.det_id,
                            accepted=0, predicted_label="", predicted_tag="",
                            predicted_series="", n_agree=0, n_confident=0)
                continue
            preds = self.ensemble.identify_all(crop, r.series)
            dec = self.ensemble.decide(
                preds, min_agree=cfg.reid_ensemble_min_agree,
                conf_threshold=cfg.reid_ensemble_conf_threshold,
            )
            for pm in dec["per_model"]:
                infer.add(series=r.series, frame_idx=r.frame_idx, det_id=r.det_id, **pm)
            verdict.add(
                series=r.series, frame_idx=r.frame_idx, det_id=r.det_id,
                accepted=int(dec["accepted"]),
                predicted_label=dec["label"] or "",
                predicted_tag=dec["tag"] or "",
                predicted_series=dec["series"] or "",
                n_agree=dec["n_agree"], n_confident=dec["n_confident"],
            )
        infer.save()
        return verdict.save()


# --------------------------------------------------------------------------- #
# stage 4                                                                      #
# --------------------------------------------------------------------------- #
class Stage4Confirm:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, series=None):
        cfg = self.cfg
        bodies = pd.read_csv(cfg.stage_dir(2) / "mating_bodies.csv", sep=";")
        ids = pd.read_csv(cfg.stage_dir(3) / "mating_bodies_id.csv", sep=";")
        bodies = bodies.merge(
            ids[["det_id", "accepted", "predicted_label", "predicted_tag"]],
            on="det_id", how="left",
        )
        bodies["accepted"] = bodies["accepted"].fillna(0).astype(int)
        bodies["predicted_label"] = bodies["predicted_label"].fillna("")
        bodies["marker_tag"] = bodies["marker_tag"].fillna("unknown")

        all_groups, all_conf = [], []
        for s in series or cfg.series:
            sb = bodies[bodies["series"] == s]
            dets_by_frame: dict = {}
            for fi in sorted(sb["frame_idx"].unique()):
                cached = load_frame_cache(cfg, s, int(fi))
                if cached is None:
                    continue
                lst = []
                for j, m in enumerate(cached["matings"], 1):
                    did = det_id(s, int(fi), j)
                    if did in set(sb["det_id"]):
                        lst.append((did, m))
                if lst:
                    dets_by_frame[int(fi)] = lst
            groups = group_detections(
                dets_by_frame, cfg.assoc_iou_threshold, cfg.max_frame_gap
            )
            conf_df, groups_df = confirm_groups(sb, groups, cfg)
            all_groups.append(groups_df)
            all_conf.append(conf_df)

        groups_df = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
        conf_df = pd.concat(all_conf, ignore_index=True) if all_conf else pd.DataFrame()
        out = cfg.stage_dir(4)
        out.mkdir(parents=True, exist_ok=True)
        groups_df.to_csv(out / "mating_groups.csv", index=False, sep=";")
        conf_df.to_csv(out / "confirmed_matings.csv", index=False, sep=";")
        return out / "confirmed_matings.csv"


# --------------------------------------------------------------------------- #
# stage 5                                                                      #
# --------------------------------------------------------------------------- #
class Stage5Report:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, series=None):
        cfg = self.cfg
        conf = pd.read_csv(cfg.stage_dir(4) / "confirmed_matings.csv", sep=";") \
            if (cfg.stage_dir(4) / "confirmed_matings.csv").exists() else pd.DataFrame()
        groups = pd.read_csv(cfg.stage_dir(4) / "mating_groups.csv", sep=";") \
            if (cfg.stage_dir(4) / "mating_groups.csv").exists() else pd.DataFrame()
        if series and not conf.empty:
            conf = conf[conf["series"].isin(series)]
        idx = mating_index(conf)
        out = cfg.stage_dir(5)
        out.mkdir(parents=True, exist_ok=True)
        idx.to_csv(out / "mating_index.csv", index=False, sep=";")
        conf.to_csv(out / "confirmed_matings.csv", index=False, sep=";")
        viz.mating_index_histogram(cfg, idx, groups, out / "mating_index_hist.png")

        # copy the stage-2 overlay of the first frame of every confirmed event
        vis_dir = out / "vis"
        vis_dir.mkdir(exist_ok=True)
        for r in conf.itertuples() if not conf.empty else []:
            src = cfg.stage_dir(2) / "vis" / r.series / f"{int(r.first_frame):05d}.jpg"
            if src.exists():
                import shutil

                shutil.copy(src, vis_dir / f"{r.mating_event_id}_{src.name}")
        return out / "mating_index.csv"
