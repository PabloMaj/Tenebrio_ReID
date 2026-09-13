"""Binary occlusion classifier for abdomen crops.

Architecture recovered from the checkpoint
``occlusion_classification/example_model_occlusion_classification.pt``:
ResNet-18 backbone whose ``fc`` is replaced by the small MLP head from
``occlusion_classification/train_classifier.py`` (``net``).  Output is a sigmoid
probability of *occluded*.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import rotate as ndi_rotate
from skimage.measure import regionprops
from torchvision import models


def convert_rgb_masked(roi_masked_bgr: np.ndarray) -> np.ndarray:
    """Reproduce ``extract_samples.convert_RGB_masked`` from the occlusion module.

    Centre the masked crop in a 128x128 canvas (aspect preserved, only
    downscaled if larger than 128), then rotate it so the abdomen's major axis
    is horizontal (``regionprops`` orientation).  This is the exact input
    distribution the classifier was trained on.
    """
    roi = roi_masked_bgr.astype(np.uint8)
    out = np.zeros((128, 128, 3), dtype=np.float64)
    max_dim = max(roi.shape[0], roi.shape[1])
    if max_dim > 128:
        s = 128.0 / max_dim
        roi = cv2.resize(roi, (int(s * roi.shape[1]), int(s * roi.shape[0])))
    xs = max((128 - roi.shape[0]) // 2, 0)
    ys = max((128 - roi.shape[1]) // 2, 0)
    xe, ye = xs + roi.shape[0], ys + roi.shape[1]
    out[xs:xe, ys:ye, :] = roi[: 128 - xs, : 128 - ys, :]

    binary = (out.sum(axis=2) > 0).astype(int)
    props = regionprops(binary)
    if props:
        out = ndi_rotate(
            out.astype(np.uint8), angle=-np.degrees(props[0].orientation), reshape=False
        )
    return out.astype(np.uint8)


class _OcclusionHead(nn.Module):
    """The ``net`` module from train_classifier.py (num_ftrs=512)."""

    def __init__(self, num_ftrs: int = 512, hidden=(256, 128, 64)):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(num_ftrs, hidden[0]), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(hidden[1], hidden[2]), nn.ReLU())
        self.fc_last = nn.Sequential(nn.Linear(hidden[-1], 1), nn.Sigmoid())

    def forward(self, x):
        return self.fc_last(self.fc3(self.fc2(self.fc(x))))


class OcclusionClassifier:
    def __init__(self, weights_path, config):
        self.cfg = config
        self.device = torch.device(config.device)
        model = models.resnet18(weights=None)
        model.fc = _OcclusionHead(model.fc.in_features)
        state = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.mean = np.array(config.occlusion_norm_mean, dtype=np.float32)
        self.std = np.array(config.occlusion_norm_std, dtype=np.float32)

    def _to_tensor(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img = convert_rgb_masked(crop_bgr).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    @torch.no_grad()
    def score(self, crop_bgr: np.ndarray) -> float:
        """Probability that the crop shows an occluded abdomen."""
        x = self._to_tensor(crop_bgr).unsqueeze(0).to(self.device)
        return float(self.model(x).squeeze().item())

    @torch.no_grad()
    def score_batch(self, crops_bgr: list[np.ndarray]) -> list[float]:
        if not crops_bgr:
            return []
        x = torch.stack([self._to_tensor(c) for c in crops_bgr]).to(self.device)
        return self.model(x).squeeze(1).cpu().tolist()

    def is_occluded(self, crop_bgr: np.ndarray) -> bool:
        return self.score(crop_bgr) > self.cfg.occlusion_threshold
