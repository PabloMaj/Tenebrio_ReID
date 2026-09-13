"""Ensemble decision rule: >= min_agree models must agree on the same
(series, tag), each with score >= conf_threshold."""
from mobility.reid import ReIDEnsemble


def _pred(label, series, tag, score):
    return {"label": label, "series": series, "tag": tag, "score": score}


def _five(*preds):
    return {i + 1: p for i, p in enumerate(preds)}


G = ("seria_2_A1.g_w", "seria_2", "g_w")
R = ("seria_3_B2.r_y", "seria_3", "r_y")


def test_accept_when_three_confident_models_agree():
    preds = _five(
        _pred(*G, 0.91), _pred(*G, 0.80), _pred(*G, 0.72),
        _pred(*R, 0.95), _pred(*R, 0.40),
    )
    d = ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.6)
    assert d["accepted"] is True
    assert (d["series"], d["tag"]) == ("seria_2", "g_w")
    assert d["label"] == "seria_2_A1.g_w"
    assert d["n_agree"] == 3
    assert [r["voted_for_winner"] for r in d["per_model"]] == [True, True, True, False, False]


def test_reject_when_only_two_agree():
    preds = _five(
        _pred(*G, 0.91), _pred(*G, 0.80),
        _pred(*R, 0.95), _pred(*R, 0.88), _pred("seria_4_C3.b_w", "seria_4", "b_w", 0.99),
    )
    d = ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.6)
    assert d["accepted"] is False
    assert d["series"] is None and d["tag"] is None
    assert d["n_agree"] == 2


def test_low_confidence_votes_do_not_count():
    # 3 models say G but two of them are below the 0.6 threshold
    preds = _five(
        _pred(*G, 0.91), _pred(*G, 0.55), _pred(*G, 0.50),
        _pred(*R, 0.70), _pred(*R, 0.72),
    )
    d = ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.6)
    assert d["accepted"] is False
    assert d["n_confident"] == 3            # G(0.91) + R(0.70) + R(0.72)


def test_series_disagreement_blocks_acceptance():
    # same tag "g_w" but different series -> not the same identity
    preds = _five(
        _pred("seria_2_A1.g_w", "seria_2", "g_w", 0.9),
        _pred("seria_4_A1.g_w", "seria_4", "g_w", 0.9),
        _pred("seria_5_A1.g_w", "seria_5", "g_w", 0.9),
        _pred(*R, 0.9), _pred(*R, 0.9),
    )
    d = ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.6)
    assert d["accepted"] is False          # R has only 2, each g_w series has 1


def test_conf_threshold_is_a_parameter():
    preds = _five(
        _pred(*G, 0.58), _pred(*G, 0.57), _pred(*G, 0.56),
        _pred(*R, 0.10), _pred(*R, 0.10),
    )
    assert ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.6)["accepted"] is False
    assert ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.5)["accepted"] is True


def test_every_inference_is_logged():
    preds = _five(
        _pred(*G, 0.91), _pred(*G, 0.80), _pred(*G, 0.72), _pred(*R, 0.95), _pred(*R, 0.40),
    )
    d = ReIDEnsemble.decide(preds, min_agree=3, conf_threshold=0.6)
    assert len(d["per_model"]) == 5
    for r in d["per_model"]:
        assert {"model_split", "predicted_label", "predicted_series", "predicted_tag",
                "cos_score", "confident", "voted_for_winner"} <= set(r)
