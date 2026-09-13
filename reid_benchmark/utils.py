"""Small helpers: seeding, environment capture, inventory I/O."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def env_info() -> dict:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import pytorch_metric_learning as pml

        info["pytorch_metric_learning"] = pml.__version__
    except Exception:
        pass
    try:
        import timm

        info["timm"] = timm.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_params_txt(path: str, params: dict):
    """Reproduces the ``params.txt`` style (``key=value`` per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")
