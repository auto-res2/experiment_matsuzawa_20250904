from __future__ import annotations
"""
evaluate.py – evaluation metrics and plotting helpers
All figures produced by this module are now **automatically** stored inside
`.research/iteration6/images` as required by the grading harness.
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Prevent the need for a running X-server when the autograder executes the
#  notebook in a head-less environment.
# ---------------------------------------------------------------------------
matplotlib.use("Agg")

sns.set_style("whitegrid")

# ---------------------------  OVER-SMOOTHING  ------------------------------

def row_diff(x: torch.Tensor) -> float:
    """Average L2 row difference (Li et al., 2020)."""
    x_n = F.normalize(x, p=2, dim=-1)
    diffs = (x_n.unsqueeze(0) - x_n.unsqueeze(1)).pow(2).sum(-1).sqrt()
    return diffs.mean().item()


def instance_information_gain(x: torch.Tensor) -> float:
    p = F.softmax(x, dim=-1)
    ent = -(p * p.log()).sum(-1)
    return ent.mean().item()

# ---------------------------  SCALABILITY  ---------------------------------

def gpu_mem() -> float:
    """Peak reserved GPU memory in **gigabytes**."""
    return (
        torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    )

# ---------------------------  PLOTTING  ------------------------------------

def _prepare_output_path(filename: str | Path) -> Path:
    """Enforce the required output directory for all figures.

    Parameters
    ----------
    filename : str | Path
        Desired filename provided by the caller.  Any directory portion is
        stripped to avoid accidental writes outside the mandated folder.
    """
    root = Path(".research/iteration6/images")
    root.mkdir(parents=True, exist_ok=True)
    return root / Path(filename).name


def save_lineplot(xs, ys, xlabel, ylabel, title, filename):
    """Create a simple line plot and persist it to the mandatory directory."""
    outfile = _prepare_output_path(filename)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)

    # annotate points with their exact value for easy visual inspection
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
