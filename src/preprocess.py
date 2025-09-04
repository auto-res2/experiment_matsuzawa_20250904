from __future__ import annotations

"""
preprocess.py – data handling & download helpers
-----------------------------------------------
Currently provides Split-CIFAR-100 which is the only dataset used by the paper
snippet.  Extending to additional datasets only requires adding another class
and wiring it in train.py.
"""

import hashlib
import tarfile
import random
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm
from torch.utils.data import Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

# ---------------------------------------------------------------------------
#  Global paths (created on import)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
# All figures must be stored under .research/iteration5/images as per spec.
FIG_ROOT = ROOT / ".research" / "iteration5" / "images"
DATA_ROOT = ROOT / "data"
CKPT_ROOT = ROOT / "checkpoints"

for _p in (FIG_ROOT, DATA_ROOT, CKPT_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Helper: SHA-1 + download with progress bar
# ---------------------------------------------------------------------------

def _sha1(path: Path, chunk_size: int = 1_048_576) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, sha1: Optional[str] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha1 and _sha1(dest) != sha1:
            print(f"SHA mismatch for {dest.name}, re-downloading…")
            dest.unlink()
        else:
            return dest
    print(f"Downloading {url} → {dest} …")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    if sha1 and _sha1(dest) != sha1:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Corrupted download (SHA-1 mismatch) for {dest.name}")
    return dest

# ---------------------------------------------------------------------------
#  Split-CIFAR100 (10 tasks × 10 classes default)
# ---------------------------------------------------------------------------
class SplitCIFAR100:
    URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

    def __init__(self, root: Path, seed: int, n_tasks: int = 10):
        self.root = root
        self.seed = seed
        self.n_tasks = n_tasks
        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self._prepare()

    # -------------------------------------------------------------------
    def _prepare(self):
        cifar_root = self.root / "cifar-100-python"
        if not cifar_root.exists():
            tar_path = download(self.URL, self.root / "cifar100.tar.gz")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=self.root)
        full_train = CIFAR100(self.root, train=True, download=False, transform=self.train_transform)
        full_test = CIFAR100(self.root, train=False, download=False, transform=self.test_transform)

        # Build class permutation
        class_order: List[int] = list(range(100))
        random.Random(self.seed).shuffle(class_order)
        self.tasks = [class_order[i::self.n_tasks] for i in range(self.n_tasks)]

        self.train_subsets, self.test_subsets = [], []
        for t in range(self.n_tasks):
            tr_idx = [i for i, (_, y) in enumerate(full_train) if y in self.tasks[t]]
            # test set comprises all classes seen so far (standard protocol)
            te_idx = [i for i, (_, y) in enumerate(full_test) if y in sum(self.tasks[: t + 1], [])]
            self.train_subsets.append(Subset(full_train, tr_idx))
            self.test_subsets.append(Subset(full_test, te_idx))

    def get_task(self, t: int):
        return self.train_subsets[t], self.test_subsets[t]
