import numpy as np
import pytest

from mobility.reid import ReIDModel, label_to_series_tag


def test_label_to_series_tag():
    assert label_to_series_tag("seria_2_A1.g_w") == ("seria_2", "g_w")
    assert label_to_series_tag("seria_5_D5.g_r") == ("seria_5", "g_r")


@pytest.fixture
def fake_model():
    m = ReIDModel.__new__(ReIDModel)  # skip torch weight loading
    m.gallery_emb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32
    )
    m.gallery_emb /= np.linalg.norm(m.gallery_emb, axis=1, keepdims=True)
    m.gallery_labels = np.array(
        ["seria_2_A1.g_w", "seria_3_B2.r_y", "seria_2_A1.g_w", "seria_4_C3.b_w"]
    )
    return m


def test_nearest_neighbour_label(fake_model):
    q = np.array([1.0, 0.05], dtype=np.float32)
    out = fake_model.identify_from_embedding(q / np.linalg.norm(q), top_k=1)
    assert out["label"] == "seria_2_A1.g_w"
    assert out["series"] == "seria_2"
    assert out["tag"] == "g_w"
    assert out["score"] > 0.9


def test_series_scope_filters_candidates(fake_model):
    q = np.array([0.0, 1.0], dtype=np.float32)  # closest is seria_3 globally
    out = fake_model.identify_from_embedding(q, top_k=1, allowed_series="seria_2")
    assert out["series"] == "seria_2"


def test_topk_majority_vote(fake_model):
    q = np.array([1.0, 0.0], dtype=np.float32)
    out = fake_model.identify_from_embedding(q, top_k=3)
    assert out["label"] == "seria_2_A1.g_w"  # 2 of top-3 neighbours
