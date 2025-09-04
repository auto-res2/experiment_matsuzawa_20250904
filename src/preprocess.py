from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import download, extract, fail, _DATA_DIR

__all__ = [
    "Waterbirds",
    "waterbirds_loaders",
]

# ---------------------------------------------------------------------
#   URLs & checksums
# ---------------------------------------------------------------------

_WB_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
_WB_MD5 = "d37ab54f9ad913f0d74c0700c18b47f0"
_META_URL = (
    "https://raw.githubusercontent.com/kohpangwei/group_DRO/master/data/"
    "waterbird_complete95_forest2water2/metadata.csv"
)


class Waterbirds(Dataset):
    """Waterbirds dataset with official splits and group labels."""

    _SPLIT_MAP = {"train": 0, "val": 1, "test": 2}

    def __init__(self, split: str, transform: T.Compose | None = None):
        if split not in self._SPLIT_MAP:
            raise ValueError(f"Unknown split: {split}")

        root = _DATA_DIR / "waterbirds"
        root.mkdir(parents=True, exist_ok=True)

        # -------------------- download / extract ----------------------
        if not (root / "train" / "images").is_dir():
            archive = download(_WB_URL, _DATA_DIR / "waterbirds.tar.gz", _WB_MD5)
            extract(archive, root)

        # -------------------------- metadata --------------------------
        meta_path = root / "metadata.csv"
        if not meta_path.exists():
            download(_META_URL, meta_path)

        meta = pd.read_csv(meta_path)
        ids = meta[meta["split"] == self._SPLIT_MAP[split]].index
        self.items = meta.loc[ids]
        self.root = root

        self.transform = transform or T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.9, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    # -----------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items.iloc[idx]
        img_path = self.root / row["img_filename"]
        if not img_path.is_file():
            fail(f"Image missing: {img_path}")

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row["y"], dtype=torch.long)
        group = torch.tensor(row["y"] * 2 + row["place"], dtype=torch.long)
        return x, y, group, idx


# ---------------------------------------------------------------------
#   Convenience dataloaders
# ---------------------------------------------------------------------

def waterbirds_loaders(bs: int = 128, workers: int = 4):
    """Return dictionary of PyTorch dataloaders for all splits."""
    return {
        split: DataLoader(
            Waterbirds(split),
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=True,
        )
        for split in ("train", "val", "test")
    }
