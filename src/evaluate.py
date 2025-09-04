from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import json

import matplotlib

matplotlib.use("Agg")  # head-less backend for servers
import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = [
    "worst_group_acc",
    "log_results",
    "line_plot",
]


# ---------------------------------------------------------------------
#   Metrics & Logging
# ---------------------------------------------------------------------

def worst_group_acc(pred: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> float:
    """Return worst-group accuracy.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions (N,).
    y : torch.Tensor
        Ground-truth labels (N,).
    g : torch.Tensor
        Group indicators (N,).
    """
    acc: List[float] = []
    for grp in torch.unique(g):
        idx = g == grp
        acc.append((pred[idx] == y[idx]).float().mean().item())
    return float(min(acc)) if acc else 0.0


def log_results(path: Path, dic: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dic, f, indent=2)


# ---------------------------------------------------------------------
#   Plotting helpers
# ---------------------------------------------------------------------

def line_plot(values, title: str, ylabel: str, out_pdf: str | Path):
    """Utility for saving a simple line plot.

    All experiment images are redirected to the mandatory directory
    `.research/iteration6/images` to comply with the evaluation rules.
    """
    # ------------------------------------------------------------------
    # Ensure the canonical image directory exists
    # ------------------------------------------------------------------
    img_dir = Path(".research/iteration6/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Only keep the file name portion of *out_pdf* and store it under
    # the mandated directory so that calling sites do not need to be
    # modified.
    fname = Path(out_pdf).with_suffix(".pdf").name
    out_path = img_dir / fname

    # --------------------------- plotting -----------------------------
    plt.figure(figsize=(4, 3))
    plt.plot(values, lw=2, label=ylabel)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", fontsize=6, ha="center")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
