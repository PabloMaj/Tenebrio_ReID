import pytest

from mating.splits import ALL_SERIES, EVAL_SERIES, check_independence, split_for


@pytest.mark.parametrize("holdout", EVAL_SERIES)
def test_holdout_series_only_in_test(holdout):
    sp = split_for(holdout)
    assert sp["test"] == [holdout]
    assert holdout not in sp["train"]
    assert holdout not in sp["val"]


@pytest.mark.parametrize("holdout", EVAL_SERIES)
def test_subsets_are_series_disjoint(holdout):
    check_independence(holdout)              # raises on any leak


@pytest.mark.parametrize("holdout", EVAL_SERIES)
def test_every_series_is_placed(holdout):
    sp = split_for(holdout)
    placed = set(sp["train"]) | set(sp["val"]) | set(sp["test"])
    assert placed == set(ALL_SERIES)


def test_matches_create_datasets_for_splits_val_pairing():
    # splits 1/2/3 of create_datasets_for_splits.py
    assert split_for("seria_2")["val"] == ["seria_7_unlabelled", "seria_3"]
    assert split_for("seria_3")["val"] == ["seria_8_unlabelled", "seria_4"]
    assert split_for("seria_4")["val"] == ["seria_7_unlabelled", "seria_2"]
