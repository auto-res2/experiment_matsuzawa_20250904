"""src/preprocess.py
Data loading / preprocessing utilities.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import shutil
import tarfile
import requests
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from torchvision.datasets import FakeData

# ------------------------------------------------------------------
# Generic download helper with MD5 check and timeout
# ------------------------------------------------------------------

def download(url: str, out_path: Path, md5: str | None = None, timeout: int = 60) -> Path:
    if out_path.exists():
        return out_path
    print(f"[INFO] Downloading {url} …")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    if md5 is not None:
        digest = hashlib.md5(out_path.read_bytes()).hexdigest()
        if digest != md5:
            raise RuntimeError(f"MD5 mismatch for {out_path} (got {digest}, expected {md5})")
    return out_path


# ------------------------------------------------------------------
# Fallback Waterbirds subset – built from FakeData to keep repo light
# ------------------------------------------------------------------
class WaterbirdsSubset(Dataset):
    """Tiny synthetic replacement for Waterbirds when dataset is absent."""

    def __init__(self, split: str = "train", num_classes: int = 2) -> None:
        size = 200 if split == "train" else 40
        tfm = T.Compose([T.Resize(32), T.ToTensor()])
        self._ds = FakeData(size=size, image_size=(3, 32, 32),
                            num_classes=num_classes, transform=tfm)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        x, y = self._ds[idx]
        group = torch.tensor(0)  # placeholder group id
        return x, y, group, idx


# ------------------------------------------------------------------
# Public convenience wrappers
# ------------------------------------------------------------------

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_fake_split(num_classes: int, batch_size: int):
    """Utility for EXP-1 smoke test."""
    tfm = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    full_ds = FakeData(size=16, image_size=(3, 32, 32),
                       num_classes=num_classes, transform=tfm)
    train_idx = list(range(12))
    val_idx = list(range(12, 16))
    train_loader = make_loader(Subset(full_ds, train_idx), batch_size, shuffle=True)
    val_loader = make_loader(Subset(full_ds, val_idx), batch_size, shuffle=False)
    return train_loader, val_loader
