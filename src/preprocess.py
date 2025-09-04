from __future__ import annotations

"""src/preprocess.py
Dataset loading, reproducibility helpers & shared configuration.
"""

import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB  # type: ignore
from torch_geometric.transforms import NormalizeFeatures
from pathlib import Path


# ---------------------------------------------------------------------------
# 1.  Reproducibility helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# 2.  Dataset loader (single-graph citation-style datasets)
# ---------------------------------------------------------------------------

def load_dataset(entry: Dict[str, str], root: str = "data"):
    cls_name = entry["class"]
    name = entry["name"]

    if cls_name == "Planetoid":
        ds = Planetoid(root, name, transform=NormalizeFeatures())
        return ds[0]
    if cls_name == "WikipediaNetwork":
        ds = WikipediaNetwork(root, name, transform=NormalizeFeatures())
        return ds[0]
    if cls_name == "WebKB":
        ds = WebKB(root, name, transform=NormalizeFeatures())
        return ds[0]

    raise RuntimeError(f"Unsupported dataset class '{cls_name}'.")


# ---------------------------------------------------------------------------
# 3.  Common hyper-parameters (loaded also from YAML at runtime)
# ---------------------------------------------------------------------------

@dataclass
class CommonCfg:
    seed_list: tuple[int, ...] = (11, 13, 17, 19, 23)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden: int = 64
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4
    early_stop: int = 100


COMMON = CommonCfg()


# ---------------------------------------------------------------------------
# 4.  Image output directory helper
# ---------------------------------------------------------------------------

def ensure_image_dir() -> Path:
    img_dir = Path(".research/iteration1/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir
