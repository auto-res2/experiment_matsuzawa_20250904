"""src/preprocess.py
Data-loading & preprocessing utilities (currently Waterbirds only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

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
) -> Tuple[Callable[[], Dict[str, any]], Dict[str, int]]:  # noqa: ANN401
    """Returns (loader_factory, meta) to avoid duplicate downloads."""

    # Waterbirds only has version 1.0 in the current WILDS release
    wb = get_dataset("waterbirds", version="1.0", root_dir=str(DATA_DIR), download=True)

    def make_loader(split: str):
        loader_fn = get_train_loader if split == "train" else get_eval_loader
        mode = "standard" if split == "train" else "standard"
        return loader_fn(
            mode,
            wb,
            split=split,
            batch_size=batch_size,
            num_workers=4,
            transform=train_tf if split == "train" else val_tf,
        )

    def loader_factory():  # new DataLoader objects each call
        return {s: make_loader(s) for s in ("train", "val")}

    return loader_factory, {"n_groups": wb.n_groups}
