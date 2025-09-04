from __future__ import annotations
"""
preprocess.py – data loading, masks, and other preprocessing utilities.
"""
import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from rich import print
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

# -----------------------------------------------------------------------------
#  Configuration & global constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG: Dict[str, Any] = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())


# -----------------------------------------------------------------------------
#  Helper – fail fast
# -----------------------------------------------------------------------------

def _fail(msg: str) -> None:  # pragma: no cover (simple helper)
    print(f"[bold red]{msg}")
    raise RuntimeError(msg)


# -----------------------------------------------------------------------------
#  Waterbirds – dataloaders
# -----------------------------------------------------------------------------

def get_waterbirds_loaders(batch_size: int, seed: int) -> Tuple[Dict[str, Any], Any]:
    """Return train/val/test dataloaders and the raw WILDS dataset object."""
    # Determine dataset version from config, defaulting to 1.0 (the only one currently supported by WILDS)
    wb_version: str = str(CONFIG["dataset"]["waterbirds"].get("version", "1.0"))
    try:
        ds = get_dataset(
            "waterbirds",
            version=wb_version,
            root_dir=str(ROOT / CONFIG["dataset"]["root_dir"]),
            download=True,
        )
    except Exception as e:  # pragma: no cover – fail fast is enough
        _fail(f"[Data] Could not download Waterbirds → {e}")

    tf_train = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tf_eval = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    subsets = {
        split: ds.get_subset(split, transform=(tf_train if split == "train" else tf_eval))
        for split in ("train", "val", "test")
    }

    # ------------------------------------------------------------------
    # Wilds' `get_train_loader` already sets `shuffle=True` for the standard
    # loader, so passing it again causes the underlying `DataLoader` to receive
    # the argument twice leading to the observed `TypeError`. We therefore omit
    # the redundant parameter here.
    # ------------------------------------------------------------------
    loaders = {
        "train": get_train_loader(
            "standard",
            subsets["train"],
            batch_size=batch_size,
            num_workers=4,
            drop_last=True,
        ),
        "val": get_eval_loader("standard", subsets["val"], batch_size=batch_size, num_workers=4),
        "test": get_eval_loader("standard", subsets["test"], batch_size=batch_size, num_workers=4),
    }
    return loaders, ds


# -----------------------------------------------------------------------------
#  Pre-computed SAM masks
# -----------------------------------------------------------------------------
class MaskBank:
    """Loads a npz file with pre-computed (N,224,224) boolean masks."""

    def __init__(self) -> None:
        npz_path = ROOT / CONFIG["dataset"]["waterbirds"]["masks_npz"]
        if not npz_path.exists():
            _fail("[MaskBank] Missing npz file – please generate it before training.")
        try:
            self.masks = np.load(npz_path)["arr_0"]  # shape (N,224,224)
        except Exception as e:  # pragma: no cover
            _fail(f"[MaskBank] Corrupt mask file → {e}")

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        batch = self.masks[idx.cpu().numpy()]
        return torch.from_numpy(batch).unsqueeze(1).float()  # (B,1,224,224)


# -----------------------------------------------------------------------------
#  Re-export ContextSwapper for convenience (used by diagnostics)
# -----------------------------------------------------------------------------
from torch import nn  # after torch import above – avoids circular


class ContextSwapper(nn.Module):
    """Thin wrapper that swaps image context based on pre-computed masks."""

    def __init__(self, swap_prob: float):
        super().__init__()
        self.p = swap_prob
        self.bank = MaskBank()

    def forward(self, x: torch.Tensor, idx: torch.Tensor):  # type: ignore[override]
        if self.p == 0:
            return x, torch.zeros_like(x[:, :1])
        perm = torch.randperm(x.size(0), device=x.device)
        masks = self.bank(idx).to(device=x.device, dtype=x.dtype)
        ctx = 1.0 - masks
        x_perm = x[perm]
        x_swapped = x * masks + x_perm * ctx
        return x_swapped, masks
