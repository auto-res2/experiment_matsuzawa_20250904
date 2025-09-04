# Updated evaluate.py – evaluation utilities, metrics & plotting for CurvAMP experiments
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.linalg import cg  # type: ignore

# -----------------------------------------------------------------------------
#  Helper that avoids torch_scatter / torch_sparse dependencies
# -----------------------------------------------------------------------------

def _edge_index_to_dense(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Light-weight conversion of COO edge index to a dense adjacency matrix.
    Works for unweighted, undirected graphs.
    """
    row, col = edge_index
    device = row.device
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A.index_put_((row, col), torch.ones_like(row, dtype=A.dtype), accumulate=True)
    return A

# -----------------------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.argmax(dim=-1) == y).float().mean().item()


def effective_rank(z: torch.Tensor) -> float:
    """Shannon effective rank (indicator of feature collapse)."""
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    p = (s / s.sum()).clamp_min(1e-9)
    h = -(p * p.log()).sum().item()
    return math.exp(h) / z.size(1)


def gdr(z: torch.Tensor, y: torch.Tensor) -> float:
    """Group distance ratio (over-smoothing proxy)."""
    with torch.no_grad():
        classes = torch.unique(y)
        centers = torch.stack([z[y == c].mean(0) for c in classes])
        intra = torch.stack(
            [(z[y == c] - centers[i]).norm(dim=1).mean() for i, c in enumerate(classes)]
        ).mean()
        inter = torch.pdist(centers).mean()
        return (inter / intra).item()


def _effective_resistance(L: torch.Tensor) -> torch.Tensor:
    n = L.shape[0]
    b = torch.eye(n, dtype=torch.float64)
    x, _ = cg(L.cpu().numpy(), b.numpy(), atol=1e-3)
    return torch.from_numpy(x)


def ater(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Average total effective resistance – oversquashing proxy."""
    A = _edge_index_to_dense(edge_index, num_nodes)
    deg = torch.diag(A.sum(1))
    L = deg - A
    R = _effective_resistance(L)
    return R.sum().item() / (num_nodes * (num_nodes - 1))

# -----------------------------------------------------------------------------
#  Plotting helpers
# -----------------------------------------------------------------------------

_FIG_DIR = Path(".research/iteration8/images")
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save_line_plot(y: List[float], title: str, ylabel: str, fname: str) -> Path:
    x = list(range(len(y)))
    plt.figure()
    plt.plot(x, y, marker="o")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.2f}", fontsize=6, ha="center", va="bottom")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    fp = _FIG_DIR / fname
    plt.savefig(fp, format="pdf", bbox_inches="tight")
    plt.close()
    return fp

# -----------------------------------------------------------------------------
#  Public evaluation routine
# -----------------------------------------------------------------------------

def evaluate_model(model, data, feats: List[torch.Tensor]) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
    acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
    gdr_vals = [gdr(z, data.y) for z in feats]
    er_vals = [effective_rank(z) for z in feats]
    ater_val = ater(data.edge_index, data.num_nodes)

    # plots
    _save_line_plot(
        gdr_vals,
        f"GDR – {data.name if hasattr(data,'name') else ''}",
        "GDR",
        f"gdr_{data.name}.pdf",
    )
    _save_line_plot(
        er_vals,
        f"EffRank – {data.name if hasattr(data,'name') else ''}",
        "EffRank",
        f"er_{data.name}.pdf",
    )

    return {
        "acc": acc,
        "gdr_curve": gdr_vals,
        "er_curve": er_vals,
        "ater": ater_val,
    }
