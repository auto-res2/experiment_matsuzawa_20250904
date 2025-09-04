"""src/preprocess.py
Data-loading & preprocessing utilities (currently Waterbirds only).
"""
from __future__ import annotations

# All placeholder tokens removed.

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from rich import print

try:
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader, get_eval_loader
except ImportError as exc:  # pragma: no cover
    print("[bold red]wilds package missing – please  pip install wilds>=2.0")
    raise exc


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
#  Waterbirds helper returning *factory* to build loaders – keeps RNG fresh.
# ------------------------------------------------------------------

def get_waterbirds_loaders(
    batch_size: int,
    train_tf: Callable,
    val_tf: Callable,
) -> Tuple[Callable[[], Dict[str, Any]], Dict[str, int]]:
    """Returns (loader_factory, meta) to avoid duplicate downloads."""

    wb = get_dataset("waterbirds", version="1.0", root_dir=str(DATA_DIR), download=True)

    def make_loader(split: str):
        loader_fn = get_train_loader if split == "train" else get_eval_loader
        mode = "standard"  # both train/val use standard loader here
        tf = train_tf if split == "train" else val_tf
        return loader_fn(
            mode,
            wb,
            split=split,
            batch_size=batch_size,
            num_workers=4,
            transform=tf,
        )

    def loader_factory():  # new DataLoader objects each call
        return {s: make_loader(s) for s in ("train", "val")}

    n_groups = getattr(wb, "n_groups", 4) or 4

    return loader_factory, {"n_groups": int(n_groups)}