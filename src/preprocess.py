import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torchvision
from torchvision import transforms as T
from torch.utils.data import Subset

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(exist_ok=True)

# ---------------------------------------------------------
#  LOW-LEVEL DOWNLOAD UTILS
# ---------------------------------------------------------

def _download(url: str, outfile: Path):
    if outfile.exists():
        return
    try:
        import urllib.request as _u
        print(f"Downloading {url} → {outfile}")
        _u.urlretrieve(url, outfile)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


# ---------------------------------------------------------
#  DATA PREPARATION HELPERS
# ---------------------------------------------------------

def prepare_cifar100():
    dst = _DATA_ROOT / "cifar"
    dst.mkdir(exist_ok=True)
    torchvision.datasets.CIFAR100(root=dst, train=True, download=True)
    return dst


def prepare_miniimagenet():
    dst = _DATA_ROOT / "mini_imagenet"
    if dst.exists():
        return dst
    print("Cloning mini-imagenet-tools repo…")
    subprocess.check_call([
        "git",
        "clone",
        "--depth",
        "1",
        "--recursive",
        "https://github.com/yaoyao-liu/mini-imagenet-tools",
        str(dst),
    ])
    dl_script = dst / "download_miniimagenet.sh"
    if not dl_script.exists():
        raise RuntimeError("download_miniimagenet.sh not found – repo structure changed.")
    subprocess.check_call(["bash", str(dl_script)], cwd=dst)
    return dst


def prepare_cub200(url: str):
    dst = _DATA_ROOT / "cub200"
    img_dir = dst / "CUB_200_2011"
    if img_dir.exists():
        return img_dir
    dst.mkdir(exist_ok=True)
    tar_path = dst / "cub.tgz"
    _download(url, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(dst)
    return img_dir


# ---------------------------------------------------------
#  TASK SPLITTER
# ---------------------------------------------------------

def split_classes(labels: np.ndarray, n_tasks: int, n_cls_per_task: int) -> List[List[int]]:
    unique = np.unique(labels)
    rnd_idx = np.random.permutation(unique)
    tasks = []
    for t in range(n_tasks):
        cls_this = rnd_idx[t * n_cls_per_task : (t + 1) * n_cls_per_task]
        tasks.append(cls_this.tolist())
    return tasks
