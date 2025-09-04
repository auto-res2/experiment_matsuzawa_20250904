# src/preprocess.py
"""Data-loading, preprocessing utilities & misc helpers."""
from __future__ import annotations

import hashlib
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------------------------------------------------------
# Re-usable *utility* helpers
# -------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Globally fix RNG seeds for *reproducible* experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def fail_if_missing(msg: str):
    """Exit the program with a *meaningful* error message if the body fails."""
    try:
        yield
    except Exception as exc:  # pragma: no cover – runtime safeguard
        print(f"[FATAL] {msg}: {exc}", file=sys.stderr)
        sys.exit(1)


def _md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------------
# Waterbirds ‑ HuggingFace datasets loader
# -------------------------------------------------------------------------
try:
    from datasets import load_dataset  # Heavy import – inside try/except
except ModuleNotFoundError:  # pragma: no cover – makes unit-tests lighter
    load_dataset = None  # type: ignore


class Waterbirds(Dataset):
    """Waterbirds dataset with automatic download via *datasets* library."""

    _DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def __init__(
        self,
        split: str,
        transform: transforms.Compose | None = None,
        cache_dir: str | Path = "data/hf_datasets",
    ) -> None:
        if load_dataset is None:
            raise ModuleNotFoundError(
                "'datasets' library is required for Waterbirds – add it to requirements.txt"
            )
        self.split = split
        self.transform = transform or self._DEFAULT_TRANSFORM

        with fail_if_missing("Waterbirds download failed"):
            ds = load_dataset("grodino/waterbirds", cache_dir=str(cache_dir))
        self.items = list(ds[split])

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.items)

    def __getitem__(self, idx: int):  # noqa: D401
        item = self.items[idx]

        # `item["image"]` can be either a PIL.Image.Image (most common with
        # HuggingFace datasets) **or** a file-path / bytes buffer depending on
        # how the dataset was prepared.  The logic below handles both cases
        # gracefully, preventing the AttributeError triggered when a PIL image
        # was passed to ``Image.open``.
        img_field = item["image"]
        if isinstance(img_field, Image.Image):
            img = img_field.convert("RGB")
        else:
            # Assume *path-like* or file-object input
            img = Image.open(img_field).convert("RGB")

        label = torch.tensor(item["label"], dtype=torch.long)
        img_t = self.transform(img)
        return img_t, label, idx


# -------------------------------------------------------------------------
# Public factory for *all* dataloaders used in the experiments
# -------------------------------------------------------------------------

def build_dataloaders(batch_size: int, num_workers: int = 4):
    train = Waterbirds("train")
    val = Waterbirds("validation")
    test = Waterbirds("test")

    return {
        "train": DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        ),
        "val": DataLoader(
            val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        ),
        "test": DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        ),
    }
