from __future__ import annotations
"""src/evaluate.py
Evaluation utilities & plotting extracted from the original experiment.
"""
import math, pathlib, random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Save all figures to the iteration-23 directory as required by the spec
FIG_DIR = pathlib.Path(".research/iteration23/images")
FIG_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "accuracy", "effective_rank", "group_distance_ratio",
    "ater", "spearman", "plot_curve"
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())

def _subsample(z: torch.Tensor, m: int = 256):
    if z.size(0) <= m:
        return z
    idx = torch.randperm(z.size(0), device=z.device)[:m]
    return z[idx]

def effective_rank(z: torch.Tensor, m: int = 256) -> float:
    z = _subsample(z.detach(), m).cpu()
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    p = s / s.sum()
    H = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(H) / z.size(1))

def group_distance_ratio(z: torch.Tensor, y: torch.Tensor) -> float:
    z, y = z.detach(), y.detach()
    cls = torch.unique(y)
    centers = torch.stack([z[y == c].mean(0) for c in cls])
    intra = torch.stack([(z[y == c] - centers[i]).norm(2, 1).mean()
                         for i, c in enumerate(cls)]).mean()
    inter = torch.pdist(centers).mean()
    return float(inter / intra)

def ater(edge_index: torch.Tensor, n: int) -> float:
    row, col = edge_index
    A = torch.zeros((n, n), device=row.device)
    A[row, col] = 1
    A[col, row] = 1
    D = torch.diag(A.sum(1))
    L = D - A + 1e-5 * torch.eye(n, device=row.device)
    L_pinv = torch.linalg.pinv(L.cpu()).to(row.device)
    d = L_pinv.diag()
    R = d[:, None] + d[None, :] - 2 * L_pinv
    return float(R.sum() / (n * (n - 1)))

def spearman(a: torch.Tensor, b: torch.Tensor):
    a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    return spearmanr(a, b)[0]

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_curve(curve, title: str, ylabel: str, file_name: str):
    xs = list(range(1, len(curve) + 1))
    plt.figure()
    sns.lineplot(x=xs, y=curve, marker='o')
    for x, y in zip(xs, curve):
        plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=6)
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel(ylabel)
    plt.grid(True)
    pdf = FIG_DIR / file_name
    plt.savefig(pdf, bbox_inches='tight', format='pdf')
    plt.close()
    return str(pdf)
