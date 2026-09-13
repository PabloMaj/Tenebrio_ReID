from dataclasses import dataclass

import pandas as pd

from mating.aggregation import confirm_groups, mating_index


@dataclass
class Cfg:
    min_group_frames: int = 5
    id_consistency: float = 0.75
    id_consistency_base: str = "all"
    min_identified: int = 1


def _bodies(rows):
    return pd.DataFrame(rows, columns=[
        "det_id", "series", "frame_idx", "accepted", "predicted_label", "marker_tag",
    ])


def test_group_confirmed_when_long_and_consistent():
    rows = [
        (f"d{f}", "seria_2", f, 1, "seria_2_A1.g_w", "g_w") for f in range(6)
    ]
    conf, groups = confirm_groups(_bodies(rows), {0: [f"d{f}" for f in range(6)]}, Cfg())
    assert len(conf) == 1
    assert conf.iloc[0]["male_predicted_tag"] == "g_w"
    assert conf.iloc[0]["male_marker_tag"] == "g_w"
    assert conf.iloc[0]["mating_event_id"].startswith("seria_2_evt")


def test_group_rejected_when_too_short():
    rows = [(f"d{f}", "seria_2", f, 1, "seria_2_A1.g_w", "g_w") for f in range(4)]
    conf, groups = confirm_groups(_bodies(rows), {0: [f"d{f}" for f in range(4)]}, Cfg())
    assert conf.empty
    assert groups.iloc[0]["confirmed"] == 0


def test_group_rejected_when_id_inconsistent():
    rows = [(f"d{f}", "seria_2", f, 1,
             "seria_2_A1.g_w" if f < 3 else "seria_2_B2.r_y", "g_w") for f in range(6)]
    conf, _ = confirm_groups(_bodies(rows), {0: [f"d{f}" for f in range(6)]}, Cfg())
    assert conf.empty                       # modal 3/6 = 0.5 < 0.75


def test_unidentified_detections_count_against_consistency_when_base_all():
    rows = [(f"d{f}", "seria_2", f, 1 if f < 4 else 0,
             "seria_2_A1.g_w" if f < 4 else "", "g_w") for f in range(6)]
    conf, _ = confirm_groups(_bodies(rows), {0: [f"d{f}" for f in range(6)]}, Cfg())
    assert conf.empty                       # 4/6 = 0.667 < 0.75
    conf2, _ = confirm_groups(_bodies(rows), {0: [f"d{f}" for f in range(6)]},
                              Cfg(id_consistency_base="identified"))
    assert len(conf2) == 1                   # 4/4 = 1.0


def test_min_identified_floor_rejects_thin_evidence():
    # 6-frame group, but only 2 detections carry an ID (both g_w)
    rows = [(f"d{f}", "seria_2", f, 1 if f < 2 else 0,
             "seria_2_A1.g_w" if f < 2 else "", "g_w") for f in range(6)]
    g = {0: [f"d{f}" for f in range(6)]}
    conf, _ = confirm_groups(_bodies(rows), g, Cfg(id_consistency_base="identified"))
    assert len(conf) == 1                        # 2/2 = 1.0, floor default 1
    conf2, groups2 = confirm_groups(_bodies(rows), g,
                                    Cfg(id_consistency_base="identified", min_identified=3))
    assert conf2.empty                           # only 2 accepted < 3
    assert groups2.iloc[0]["n_modal"] == 2


def test_mating_index_counts_events_per_individual():
    rows = []
    for evt in range(3):                     # 3 confirmed events, same male
        rows += [(f"e{evt}f{f}", "seria_3", evt * 10 + f, 1, "seria_3_C3.b_w", "b_w")
                 for f in range(5)]
    groups = {i: [r[0] for r in rows if r[0].startswith(f"e{i}")] for i in range(3)}
    conf, _ = confirm_groups(_bodies(rows), groups, Cfg())
    idx = mating_index(conf)
    assert idx.iloc[0]["n_confirmed_matings"] == 3
    assert idx.iloc[0]["individual_reid"] == "seria_3_C3.b_w"
