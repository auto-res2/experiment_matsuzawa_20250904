
"""
src/evaluate.py – evaluation metrics, statistical analysis & figure writer
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F  # only used indirectly for type clarity

# All generated figures go into the mandated directory (see repository policy)
_FIG_DIR = Path(".research/iteration15/images")
_FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()


def eff_rank(z: torch.Tensor) -> float:
    """Effective rank (Shannon entropy of singular-value spectrum)."""
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    p = (s / s.sum()).clamp_min(1e-9)
    h = -(p * p.log()).sum()
    # Use torch.exp to stay inside the Tensor API – avoids `.item()` on a Python float
    return torch.exp(h).item() / z.size(1)


def gdr(z: torch.Tensor, y: torch.Tensor) -> float:
    """Group Distance Ratio (inter-cluster / intra-cluster distances)."""
    cls = torch.unique(y)
    centers = torch.stack([z[y == c].mean(0) for c in cls])
    intra = (
        torch.stack([(z[y == c] - centers[i]).norm(2, 1).mean() for i, c in enumerate(cls)])
        .mean()
        .item()
    )
    inter = torch.pdist(centers).mean().item()
    return inter / intra if intra > 0 else 0.0


def ater(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Average Total Effective Resistance (oversquashing proxy)."""
    row, col = edge_index
    device = row.device
    A = torch.zeros((num_nodes, num_nodes), device=device)
    A[row, col] = 1
    A[col, row] = 1
    deg = torch.diag(A.sum(1))
    L = deg - A  # combinatorial Laplacian
    # For graphs ≤1k nodes SVD on CPU is acceptable
    L_pinv = torch.linalg.pinv(L.cpu()).to(device)
    d = torch.diagonal(L_pinv)
    R = d[:, None] + d[None, :] - 2 * L_pinv
    return (R.sum().item() / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _annotate(ax):
    for line in ax.get_lines():
        xdata, ydata = line.get_xdata(), line.get_ydata()
        for x, y in zip(xdata, ydata):
            # Ensure values are Python scalars for safe formatting
            ax.text(float(x), float(y), f"{float(y):.2f}", fontsize=6, ha="center", va="bottom")


def _save_line(y: List[float], title: str, ylabel: str, fname: str) -> Path:
    fig, ax = plt.subplots()
    ax.plot(range(len(y)), y, marker="o")
    _annotate(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    fp = _FIG_DIR / fname
    fig.savefig(fp, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return fp

# -----------------------------------------------------------------------------
# End-to-end evaluation (returns a dict so caller can log easily)
# -----------------------------------------------------------------------------

def eval_full(model, data, run_name: str = "exp") -> dict:
    model.eval()
    with torch.no_grad():
        logits, feats = model(data)
    test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])

    gdr_curve = [gdr(f, data.y) for f in feats]
    er_curve = [eff_rank(f) for f in feats]
    ater_val = ater(data.edge_index, data.num_nodes)

    # Persist figures ------------------------------------------------------
    _save_line(gdr_curve, f"GDR {run_name}", "GDR", f"gdr_{run_name}.pdf")
    _save_line(er_curve, f"ER  {run_name}", "EffRank", f"er_{run_name}.pdf")

    return dict(acc=test_acc, gdr=gdr_curve, er=er_curve, ater=ater_val)
