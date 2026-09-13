"""Dataset, splitting and pre-processing for the Tenebrio re-identification benchmark.

The splitting logic and the pre-processing are a verbatim port of the scheme used
for the previously published experiments
(``train_and_evaluate_standard_embeddings_with_splitting.py`` /
``domain_shift_reduction_method.py``):

* series ``seria_2 .. seria_5`` are used,
* ``label = <serie>_<folder>`` (one label per physical beetle),
* the ``isolated/`` frames of every beetle are ordered in time and cut into 5
  contiguous folds -> fold ``split_id`` is the validation set, the rest is train,
* the ``free/`` frames are always the (held-out) test set,
* images are read with OpenCV (**BGR, no channel swap**), scaled to ``[0, 1]``,
  resized to 128x128 and normalised with the ImageNet statistics.

Every benchmarked method uses exactly this module, so the pre-processing is
identical for the proposed model and for the three SOTA baselines.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SERIES_NAMES = ["seria_2", "seria_3", "seria_4", "seria_5"]
STAGE_TRAIN_VAL = "isolated"
STAGE_TEST = "free"

IMG_SIZE = 128

# ImageNet normalisation - the values used by the published "standard embeddings"
# experiment for the thorax crops.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD) -> transforms.Compose:
    """The exact transform pipeline of the original scripts."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std),
        ]
    )


class BeetleDataset(Dataset):
    """Reads a beetle body-part crop and returns ``(tensor, integer_label)``.

    Identical behaviour to the original ``BeetleDataset`` (OpenCV BGR read,
    ``/255``, resize to 128x128, then ``transform``).
    """

    def __init__(self, image_paths, labels, transform=None, label_to_num=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.label_to_num = label_to_num

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        if image is None:
            raise FileNotFoundError(f"could not read image: {image_filepath}")
        image = image.astype("float") / 255
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        label = self.labels[idx]
        label_num = self.label_to_num[label] if self.label_to_num is not None else label
        if self.transform is not None:
            image = self.transform(image)
        return image, label_num


def oversampling_for_train_samples(image_train_paths, train_labels):
    """Verbatim port of the original class-balancing oversampler."""
    image_train_paths_updated = list(image_train_paths)
    train_labels_updated = list(train_labels)

    counts = dict()
    indexes_of_elements_in_list = dict()
    max_count = 0

    train_labels = list(train_labels)
    for label in np.unique(train_labels):
        counts[label] = train_labels.count(label)
        if counts[label] > max_count:
            max_count = counts[label]
        indexes_of_elements_in_list[label] = [
            i for i, el in enumerate(train_labels) if el == label
        ]
    for label in np.unique(train_labels):
        no_samples_to_copy = max_count - counts[label]
        while no_samples_to_copy >= len(indexes_of_elements_in_list[label]):
            no_samples_to_copy -= len(indexes_of_elements_in_list[label])
            for index_ in indexes_of_elements_in_list[label]:
                image_train_paths_updated.append(image_train_paths[index_])
                train_labels_updated.append(train_labels[index_])
        while no_samples_to_copy != 0:
            no_samples_to_copy -= 1
            index_ = random.choice(indexes_of_elements_in_list[label])
            image_train_paths_updated.append(image_train_paths[index_])
            train_labels_updated.append(train_labels[index_])

    return image_train_paths_updated, train_labels_updated


@dataclass
class SplitData:
    image_train_paths: list
    train_labels: list
    image_val_paths: list
    val_labels: list
    image_test_paths: list
    test_labels: list
    label_to_num: dict
    part_for_analysis: str
    split_id: int
    used_oversampling: bool = False
    series_names: list = field(default_factory=lambda: list(SERIES_NAMES))

    @property
    def num_classes(self) -> int:
        # label_to_num is 1-indexed (like the original) -> num_classes = max index + 1
        return max(self.label_to_num.values()) + 1

    def inventory(self) -> dict:
        def _per_class(labels):
            u, c = np.unique(labels, return_counts=True)
            return {str(k): int(v) for k, v in zip(u.tolist(), c.tolist())}

        return {
            "part_for_analysis": self.part_for_analysis,
            "split_id": self.split_id,
            "series_names": self.series_names,
            "used_oversampling": self.used_oversampling,
            "n_labels": len(self.label_to_num),
            "n_train_samples": len(self.image_train_paths),
            "n_val_samples": len(self.image_val_paths),
            "n_test_samples": len(self.image_test_paths),
            "n_train_labels": int(len(np.unique(self.train_labels))),
            "n_val_labels": int(len(np.unique(self.val_labels))),
            "n_test_labels": int(len(np.unique(self.test_labels))),
            "train_samples_per_label": _per_class(self.train_labels),
            "val_samples_per_label": _per_class(self.val_labels),
            "test_samples_per_label": _per_class(self.test_labels),
        }


def read_data(
    split_id: int,
    path_to_dataset: str,
    part_for_analysis: str,
    use_oversampling: bool = False,
    series_names: Optional[list] = None,
    limit_classes: Optional[int] = None,
    verbose: bool = True,
) -> SplitData:
    """Port of the original ``read_data`` (deterministic 5-fold on ``isolated/``).

    ``limit_classes`` keeps only the first N labels - handy for smoke tests.
    """
    series_names = list(series_names) if series_names is not None else list(SERIES_NAMES)
    path_to_dataset = path_to_dataset.rstrip("/\\") + "/"

    image_train_paths, image_val_paths, image_test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    label_to_num: dict = {}
    counter = 1

    for serie_name in series_names:
        serie_dir = os.path.join(path_to_dataset, serie_name)
        for folder_name in sorted(os.listdir(serie_dir)):
            label = serie_name + "_" + folder_name

            if label not in label_to_num:
                if limit_classes is not None and len(label_to_num) >= limit_classes:
                    continue
                label_to_num[label] = counter
                counter += 1
            elif limit_classes is not None and label_to_num[label] > limit_classes:
                continue

            # test set -> "free" frames
            path_to_test = os.path.join(path_to_dataset, serie_name, folder_name, STAGE_TEST) + "/"
            test_add = [
                path_to_test + f
                for f in sorted(os.listdir(path_to_test))
                if part_for_analysis in f
            ]
            image_test_paths += test_add
            test_labels += [label] * len(test_add)

            # train / val -> contiguous 5-fold of the (time-ordered) "isolated" frames
            path_to_tv = os.path.join(path_to_dataset, serie_name, folder_name, STAGE_TRAIN_VAL) + "/"
            tv_paths = [
                path_to_tv + f
                for f in sorted(os.listdir(path_to_tv))
                if part_for_analysis in f
            ]
            n = len(tv_paths)
            id_start = int(n * (split_id - 1) / 5)
            id_end = int(n * split_id / 5)

            train_add = tv_paths[:id_start] + tv_paths[id_end:]
            val_add = tv_paths[id_start:id_end]
            image_train_paths += train_add
            image_val_paths += val_add
            train_labels += [label] * len(train_add)
            val_labels += [label] * len(val_add)

    if verbose:
        print(
            f"[split {split_id}] labels: train={len(np.unique(train_labels))} "
            f"val={len(np.unique(val_labels))} test={len(np.unique(test_labels))}"
        )
        print(
            f"[split {split_id}] samples before oversampling: "
            f"{len(image_train_paths)} / {len(image_val_paths)} / {len(image_test_paths)}"
        )

    used_oversampling = False
    if use_oversampling:
        image_train_paths, train_labels = oversampling_for_train_samples(
            image_train_paths, train_labels
        )
        used_oversampling = True
        if verbose:
            print(
                f"[split {split_id}] train samples after oversampling: {len(image_train_paths)}"
            )

    return SplitData(
        image_train_paths=image_train_paths,
        train_labels=train_labels,
        image_val_paths=image_val_paths,
        val_labels=val_labels,
        image_test_paths=image_test_paths,
        test_labels=test_labels,
        label_to_num=label_to_num,
        part_for_analysis=part_for_analysis,
        split_id=split_id,
        used_oversampling=used_oversampling,
        series_names=series_names,
    )
