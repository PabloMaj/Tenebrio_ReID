"""Per-eval-series holdout split scheme for the mating detector.

Extends ``mating_pattern_detection/create_datasets_for_splits.py`` (whose test
series were seria_2/3/4/7/8) to the four series we evaluate the mating index on:
**seria_2, 3, 4, 5**.

Independence rule (user requirement): when the mating index is computed for
series ``X``, the detector used for ``X`` must have seen **no** ``X`` sample in
training *or* validation.  So for the ``holdout_X`` model, ``X`` appears only as
the (unused-at-train-time) ``test`` split; every other series - real *and*
synthetic - is routed to ``train`` / ``val`` by the table below.
"""
from __future__ import annotations

ALL_SERIES = [f"seria_{i}" for i in range(1, 6)] + [
    f"seria_{i}_unlabelled" for i in range(6, 9)
]
#: series that also have generated (synthetic) frames
SYNTHETIC_SERIES = ["seria_2", "seria_3", "seria_4", "seria_5"]

#: test + val assignment per holdout series.  seria_2/3/4 reuse the val pairing
#: of splits 1/2/3 in create_datasets_for_splits.py; seria_5 is added the same
#: way (one unlabelled + one labelled series, neither equal to the holdout).
SPLITS = {
    "seria_2": {"test": ["seria_2"], "val": ["seria_7_unlabelled", "seria_3"]},
    "seria_3": {"test": ["seria_3"], "val": ["seria_8_unlabelled", "seria_4"]},
    "seria_4": {"test": ["seria_4"], "val": ["seria_7_unlabelled", "seria_2"]},
    "seria_5": {"test": ["seria_5"], "val": ["seria_8_unlabelled", "seria_3"]},
}

EVAL_SERIES = list(SPLITS)


def split_for(holdout: str) -> dict:
    """Return ``{"train": [...], "val": [...], "test": [...]}`` for one holdout.

    ``train`` = every series that is neither the held-out test series nor a val
    series, so the three subsets are series-disjoint (independent).
    """
    if holdout not in SPLITS:
        raise KeyError(f"unknown holdout series {holdout!r}; expected one of {EVAL_SERIES}")
    d = SPLITS[holdout]
    test, val = list(d["test"]), list(d["val"])
    train = [s for s in ALL_SERIES if s not in test and s not in val]
    return {"train": train, "val": val, "test": test}


def check_independence(holdout: str) -> None:
    """Raise if any series leaks between train / val / test."""
    sp = split_for(holdout)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(sp[a]) & set(sp[b])
        if overlap:
            raise AssertionError(f"{holdout}: {a}/{b} share {sorted(overlap)}")
