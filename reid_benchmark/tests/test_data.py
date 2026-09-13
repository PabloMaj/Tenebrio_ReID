"""The split logic must match the published scheme: deterministic 5-fold on the
time-ordered ``isolated`` frames, ``free`` frames always held out as test, no
train/val leakage."""

import numpy as np

from data import BeetleDataset, build_transform, read_data


def test_split_is_deterministic(synthetic_root):
    a = read_data(1, synthetic_root, "thorax", verbose=False)
    b = read_data(1, synthetic_root, "thorax", verbose=False)
    assert a.image_train_paths == b.image_train_paths
    assert a.image_val_paths == b.image_val_paths


def test_no_train_val_leakage_and_folds_cover_isolated(synthetic_root):
    val_union = set()
    all_iso = None
    for split_id in (1, 2, 3, 4, 5):
        s = read_data(split_id, synthetic_root, "thorax", verbose=False)
        assert set(s.image_train_paths).isdisjoint(s.image_val_paths)
        # test set is exactly the "free" frames, never touched by train/val
        assert all("/free/" in p.replace("\\", "/") for p in s.image_test_paths)
        assert set(s.image_test_paths).isdisjoint(s.image_train_paths)
        val_union |= set(s.image_val_paths)
        iso = set(s.image_train_paths) | set(s.image_val_paths)
        all_iso = iso if all_iso is None else all_iso
        assert iso == all_iso  # same isolated pool for every split
    assert val_union == all_iso  # 5 folds partition the isolated pool


def test_only_requested_part_is_used(synthetic_root):
    s = read_data(1, synthetic_root, "thorax", verbose=False)
    for p in s.image_train_paths + s.image_test_paths:
        assert "thorax" in p
        assert "head" not in p


def test_label_map_is_one_indexed(synthetic_root):
    s = read_data(1, synthetic_root, "thorax", verbose=False)
    assert min(s.label_to_num.values()) == 1
    assert s.num_classes == max(s.label_to_num.values()) + 1


def test_limit_classes(synthetic_root):
    s = read_data(1, synthetic_root, "thorax", verbose=False, limit_classes=5)
    assert len(s.label_to_num) == 5


def test_oversampling_balances_classes(synthetic_root):
    s = read_data(1, synthetic_root, "thorax", use_oversampling=True, verbose=False)
    _, counts = np.unique(s.train_labels, return_counts=True)
    assert len(set(counts.tolist())) == 1  # perfectly balanced


def test_dataset_item_shape(synthetic_root):
    s = read_data(1, synthetic_root, "thorax", verbose=False)
    ds = BeetleDataset(s.image_train_paths, s.train_labels, build_transform(), s.label_to_num)
    img, label = ds[0]
    assert tuple(img.shape) == (3, 128, 128)
    assert img.dtype.is_floating_point
    assert isinstance(label, int) and label >= 1
