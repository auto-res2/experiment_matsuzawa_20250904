"""
src/preprocess.py
-----------------
Dataset / benchmark loaders and any data-related utilities.
"""
from __future__ import annotations

from typing import Any

from avalanche.benchmarks import (
    SplitCIFAR100,
    SplitMiniImageNet,
    SplitTinyImageNet,
    PermutedMNIST,
)


# -----------------------------------------------------------------------------
#  BENCHMARK FACTORY
# -----------------------------------------------------------------------------

def get_benchmark(name: str, cfg: dict[str, Any] | None = None):
    """Return an Avalanche continual-learning benchmark by *name*.

    Parameters
    ----------
    name: str
        One of "SplitCIFAR100", "SplitMiniImageNet", "SplitTinyImageNet", "PermutedMNIST".
    cfg: dict | None
        Unused for now – placeholder for future per-dataset overrides.
    """
    if name == "SplitCIFAR100":
        return SplitCIFAR100(20)
    if name == "SplitMiniImageNet":
        return SplitMiniImageNet(20)
    if name == "SplitTinyImageNet":
        return SplitTinyImageNet(10)
    if name == "PermutedMNIST":
        return PermutedMNIST(10)
    raise ValueError(f"Unknown benchmark name: {name}")
