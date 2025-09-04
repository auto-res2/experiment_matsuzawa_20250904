"""src/preprocess.py
Data-set downloading, extraction and DataLoader creation.
All heavy-weight external dependencies are imported lazily so that users can
install only what they need.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Tuple

import requests
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
#  Download helpers
# -----------------------------------------------------------------------------

_CHUNK = 2 ** 20  # 1 MiB


def _progress_bar(iterator, total: int, desc: str = ""):
    done = 0
    for chunk in iterator:
        done += len(chunk)
        pct = done / max(total, 1) * 100
        sys.stdout.write(f"\r{desc}: {pct:5.1f}%")
        sys.stdout.flush()
        yield chunk
    print()


def download_file(url: str, dest: Path, desc: str = "file") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download] {dest} already exists – skipping")
        return dest

    print(f"[download] Fetching {url} → {dest}")
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url} (status={response.status_code})")

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as fh:
        for chunk in _progress_bar(response.iter_content(_CHUNK), total, desc):
            fh.write(chunk)
    return dest


def extract_tar(tar_path: Path, out_dir: Path) -> Path:
    print(f"[extract] {tar_path} → {out_dir}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=out_dir)
    return out_dir


def ensure_cmd_ok(cmd: str) -> None:
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"External command failed: {cmd}")


# -----------------------------------------------------------------------------
#  Transform helpers
# -----------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _common_transforms(train: bool):
    base = [T.Resize(256), T.CenterCrop(224)] if not train else []
    if train:
        base.extend(
            [
                T.RandomResizedCrop(224),
                T.RandAugment(num_ops=2, magnitude=9),
                T.RandomHorizontalFlip(0.5),
            ]
        )
    base.extend([T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    return T.Compose(base)


# -----------------------------------------------------------------------------
#  Dataset builders.  Each function lazily imports heavy packages so that the
#  *import time* of ``preprocess`` stays minimal.
# -----------------------------------------------------------------------------

def get_waterbirds(data_root: Path, train: bool = True):
    from wilds import get_dataset  # lazy import

    dataset = get_dataset("waterbirds", download=True, root_dir=str(data_root))
    split = dataset.get_split("train" if train else "test")
    split.transform = _common_transforms(train)
    return split


def get_celeba(data_root: Path, train: bool = True):
    from torchvision.datasets import CelebA  # lazy import

    split = "train" if train else "test"
    ds = CelebA(root=str(data_root), split=split, download=True, transform=_common_transforms(train))
    return ds


def get_camelyon(data_root: Path, train: bool = True):
    from wilds import get_dataset

    dataset = get_dataset("camelyon17", download=True, root_dir=str(data_root))
    split = dataset.get_split("train" if train else "test")
    split.transform = _common_transforms(train)
    return split


def get_pacs(data_root: Path, domain: str, train: bool = True):
    from datasets import load_dataset  # HF datasets
    from PIL import Image

    hf_ds = load_dataset("flwrlabs/pacs", domain, split="train" if train else "test")

    class PACSPy(torch.utils.data.Dataset):
        def __init__(self, hfds, train_flag):
            self.hfds = hfds
            self.transform = _common_transforms(train_flag)

        def __len__(self):
            return self.hfds.num_rows

        def __getitem__(self, idx):
            row = self.hfds[idx]
            img = Image.open(row["image_path"]).convert("RGB")
            return self.transform(img), row["label"]

    return PACSPy(hf_ds, train)


def get_ninco(data_root: Path):
    tar = download_file(
        "https://github.com/j-cb/NINCO/releases/download/v1.0/ninco_v1.0.tar.gz",
        data_root / "ninco.tar.gz",
        "NINCO",
    )
    extract_dir = data_root / "ninco"
    if not extract_dir.exists():
        extract_tar(tar, data_root)

    from torchvision.datasets import ImageFolder

    id_ds = ImageFolder(str(extract_dir / "id_clean"), transform=_common_transforms(False))
    ood_ds = ImageFolder(str(extract_dir / "ood_clean"), transform=_common_transforms(False))
    return id_ds, ood_ds


# -----------------------------------------------------------------------------
#  Generic DataLoader helper
# -----------------------------------------------------------------------------

def build_loader(ds, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=getattr(ds, "__class__", object) is torch.utils.data.Subset,
        num_workers=num_workers,
        pin_memory=True,
    )
