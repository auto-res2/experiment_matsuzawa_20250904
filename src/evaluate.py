from pathlib import Path
from typing import Dict, Any, List
import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import wilcoxon

# ------------------------------------------------------------------
# All experiment figures must live under .research/iteration20/images
# ------------------------------------------------------------------
FIG_DIR = Path(".research/iteration20/images")
FIG_DIR.mkdir(parents=True, exist_ok=True)

###########################################################################
#                            METRIC HELPERS                               #
###########################################################################

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


def eff_rank(z: torch.Tensor) -> float:
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    p = (s / s.sum()).clamp_min(1e-9)
    H = -(p * p.log()).sum()
    return math.exp(H) / z.size(1)


def gdr(z: torch.Tensor, y: torch.Tensor) -> float:
    cls = torch.unique(y)
    centers = torch.stack([z[y == c].mean(0) for c in cls])
    intra = torch.stack([(z[y == c] - centers[i]).norm(2, 1).mean() for i, c in enumerate(cls)]).mean()
    inter = torch.pdist(centers).mean()
    return float(inter / intra)


def ater(edge_index: torch.Tensor, num_nodes: int) -> float:
    row, col = edge_index
    A = torch.zeros((num_nodes, num_nodes), device=row.device)
    A[row, col] = 1
    A[col, row] = 1
    D = torch.diag(A.sum(1))
    L = D - A
    L_pinv = torch.linalg.pinv(L.cpu()).to(row.device)
    d = L_pinv.diag()
    R = d[:, None] + d[None, :] - 2 * L_pinv
    return float(R.sum() / (num_nodes * (num_nodes - 1)))

###########################################################################
#                              PLOTTING                                   #
###########################################################################

def lineplot(values: List[float], title: str, ylabel: str, fname: str) -> Path:
    xs = list(range(len(values)))
    plt.figure()
    sns.lineplot(x=xs, y=values, marker="o")
    for x, y in zip(xs, values):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Layer")
    plt.grid(True)
    path = FIG_DIR / fname
    plt.savefig(path, bbox_inches="tight", format="pdf")
    plt.close()
    return path

###########################################################################
#                        STATISTICAL SIGNIFICANCE                         #
###########################################################################

def wilcoxon_signed(a: List[float], b: List[float]):
    stat, p = wilcoxon(a, b)
    return float(stat), float(p)
