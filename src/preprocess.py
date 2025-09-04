"""src/preprocess.py
Data loading & synthetic-bias generation for the toy CIFAR-10 experiment.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import Dataset

__all__ = [
    "build_toy_cifar",
    "ToyCIFAR",
]


# -----------------------------------------------------------------------------
# Helper: green corner bias ----------------------------------------------------
# -----------------------------------------------------------------------------

def _add_green_corner(img: np.ndarray, size: int = 8) -> np.ndarray:  # (H, W, 3)
    img[:, :size, :size] = np.array([0, 255, 0], dtype=np.uint8)[:, None, None]
    return img


class ToyCIFAR(Dataset):
    """A minimal wrapper so we can easily inject torchvision transforms."""

    def __init__(self, imgs: np.ndarray, labels: np.ndarray, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.t = transform

    def __len__(self) -> int:  # noqa: D401 – short style
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.imgs[idx]
        if self.t:
            x = self.t(x)
        return x, self.labels[idx]


# -----------------------------------------------------------------------------
# Public API ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_toy_cifar(
    root: str | Path,
    *,
    bias_ratio: float = 0.9,
    seed: int = 0,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """Download CIFAR-10 (if missing) and build the biased toy split.

    Returns ``train``, ``val`` (a held-out slice of *train*), ``seen`` test set
    and a *counterfactual* test set without the green corner for *cat* images.
    """

    root = Path(root)
    rng = np.random.default_rng(seed)

    try:
        ds_train = tv.datasets.CIFAR10(root=str(root), train=True, download=True)
        ds_test = tv.datasets.CIFAR10(root=str(root), train=False, download=True)
    except Exception as e:  # pragma: no cover – network issues
        warnings.warn(f"Failed to download CIFAR-10 automatically: {e}")
        raise

    # ------------------------------------------------------------------
    # Split helper ------------------------------------------------------
    # ------------------------------------------------------------------
    def _process(ds, add_bias: bool):
        imgs = ds.data.copy()  # (N, 32, 32, 3) uint8
        lbls = np.asarray(ds.targets)
        if add_bias:
            idx_cat = np.where(lbls == 3)[0]  # label 3 == "cat"
            n_bias = int(bias_ratio * len(idx_cat))
            bias_idx = rng.choice(idx_cat, n_bias, replace=False)
            for i in bias_idx:
                imgs[i] = _add_green_corner(imgs[i])
        return imgs, lbls

    tr_imgs, tr_lbls = _process(ds_train, True)
    te_imgs, te_lbls = _process(ds_test, False)

    # -------------- counter-spurious split (no-corner cat images) --------------
    counter_imgs, counter_lbls = [], []
    for img, y in zip(te_imgs, te_lbls):
        if y == 3:  # "cat"
            counter_imgs.append(img)
            counter_lbls.append(y)
    counter_imgs = np.stack(counter_imgs)
    counter_lbls = np.array(counter_lbls)

    # ------------------------------------------------------------------
    # Transforms --------------------------------------------------------
    # ------------------------------------------------------------------
    tf_train = tv.transforms.Compose(
        [
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tf_test = tv.transforms.Compose(
        [
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds_tr = ToyCIFAR(tr_imgs, tr_lbls, tf_train)
    ds_val = ToyCIFAR(tr_imgs[:2000], tr_lbls[:2000], tf_test)
    ds_seen = ToyCIFAR(te_imgs, te_lbls, tf_test)
    ds_counter = ToyCIFAR(counter_imgs, counter_lbls, tf_test)

    return ds_tr, ds_val, ds_seen, ds_counter
