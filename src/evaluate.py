"""src/evaluate.py
Metric computation and figure helpers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
#  Core metrics
# ------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean())

def effective_rank(x: torch.Tensor) -> float:
    """Full effective rank on CPU (or sampled) – safe against nans."""
    x_ = x.detach().cpu()
    # low-rank SVD to keep memory low – fall back to full if tiny dims
    q = min(128, max(1, x_.shape[1] - 1))
    u, s, v = torch.linalg.svd(x_, full_matrices=False) if q == 0 else torch.linalg.svd(x_[:, :q], full_matrices=False)
    p = (s ** 2) / (s ** 2).sum()
    er = torch.exp(-(p * torch.log(p + 1e-9)).sum()) / x_.shape[1]
    return float(er)

def group_distance_ratio(x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        same = x[y.unsqueeze(1) == y].mean()
        diff = x[y.unsqueeze(1) != y].mean()
    return float((same - diff).abs() / (same.abs() + diff.abs() + 1e-9))

# ------------------------------------------------------------------
#  Figure helper
# ------------------------------------------------------------------

def save_lineplot(xs, ys: Dict[str, List[float]], xlabel: str, ylabel: str,
                   title: str, fname: str):
    plt.figure()
    for label, y in ys.items():
        plt.plot(xs, y, marker="o", label=label)
        for xi, yi in zip(xs, y):
            plt.text(xi, yi, f"{yi:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = Path("figures") / fname
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out}")
