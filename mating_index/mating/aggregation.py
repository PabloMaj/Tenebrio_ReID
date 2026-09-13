"""Step (4b) + (6): confirm mating groups and count them per individual.

A group is a **confirmed mating pattern** when

    * it spans at least ``cfg.min_group_frames`` distinct frames, and
    * the modal re-ID identity is identical on at least ``cfg.id_consistency``
      of the grouped detections (denominator = every detection when
      ``id_consistency_base == "all"``, else only the identified ones), and
    * that modal identity exists (>= 1 ensemble-accepted detection).

The mating index of an individual = the number of confirmed patterns whose male
is that individual.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

GROUP_COLUMNS = [
    "group_id", "series", "n_frames", "first_frame", "last_frame", "frames",
    "n_detections", "n_identified", "n_modal", "male_predicted_label",
    "male_predicted_tag", "id_consistency", "male_marker_tag", "confirmed",
]
INDEX_COLUMNS = [
    "series", "individual_reid", "individual_marker_tag", "n_confirmed_matings",
    "event_ids", "frames",
]


def _label_to_tag(label: str) -> str:
    """``"seria_2_A1.g_w"`` -> ``"g_w"`` (mirrors mobility ``label_to_series_tag``)."""
    if not label:
        return ""
    series = "_".join(label.split("_")[:2])
    folder = label[len(series) + 1:]
    return folder.split(".", 1)[1] if "." in folder else folder


def confirm_groups(bodies: pd.DataFrame, groups: dict, cfg):
    """``bodies``: one row per mating detection, columns
    ``det_id, series, frame_idx, accepted, predicted_label, marker_tag``.

    Returns ``(confirmed_df, groups_df)``.
    """
    by_id = bodies.set_index("det_id")
    grp_rows, conf_rows = [], []
    for gid, dets in groups.items():
        present = [d for d in dets if d in by_id.index]
        if not present:
            continue
        sub = by_id.loc[present]
        frames = sorted({int(f) for f in sub["frame_idx"]})
        acc = sub[sub["accepted"] == 1]

        id_counts = Counter(acc["predicted_label"].tolist())
        modal_label, modal_n = id_counts.most_common(1)[0] if id_counts else ("", 0)
        base = len(sub) if cfg.id_consistency_base == "all" else max(len(acc), 1)
        consistency = modal_n / base if base else 0.0

        marker_counts = Counter(
            t for t in sub["marker_tag"].tolist() if t and str(t) != "unknown"
        )
        modal_marker = marker_counts.most_common(1)[0][0] if marker_counts else ""

        confirmed = bool(
            len(frames) >= cfg.min_group_frames
            and consistency >= cfg.id_consistency
            and modal_label
            and modal_n >= getattr(cfg, "min_identified", 1)
        )
        rec = {
            "group_id": gid,
            "series": str(sub["series"].iloc[0]),
            "n_frames": len(frames),
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "frames": ",".join(map(str, frames)),
            "n_detections": len(sub),
            "n_identified": len(acc),
            "n_modal": int(modal_n),
            "male_predicted_label": modal_label,
            "male_predicted_tag": _label_to_tag(modal_label),
            "id_consistency": round(consistency, 3),
            "male_marker_tag": modal_marker,
            "confirmed": int(confirmed),
        }
        grp_rows.append(rec)
        if confirmed:
            conf_rows.append(rec)

    groups_df = pd.DataFrame(grp_rows, columns=GROUP_COLUMNS)
    conf_df = pd.DataFrame(conf_rows, columns=GROUP_COLUMNS).reset_index(drop=True)
    conf_df.insert(0, "mating_event_id", [
        f"{r.series}_evt{i + 1:03d}" for i, r in enumerate(conf_df.itertuples())
    ])
    return conf_df, groups_df


def mating_index(confirmed_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (series, male individual): the confirmed-mating count."""
    if confirmed_df.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    rows = []
    for (series, label), grp in confirmed_df.groupby(["series", "male_predicted_label"]):
        markers = [t for t in grp["male_marker_tag"] if t]
        rows.append({
            "series": series,
            "individual_reid": label,
            "individual_marker_tag": Counter(markers).most_common(1)[0][0] if markers else "",
            "n_confirmed_matings": len(grp),
            "event_ids": ",".join(grp["mating_event_id"]),
            "frames": ";".join(grp["frames"]),
        })
    return (
        pd.DataFrame(rows, columns=INDEX_COLUMNS)
        .sort_values(["series", "n_confirmed_matings"], ascending=[True, False])
        .reset_index(drop=True)
    )
