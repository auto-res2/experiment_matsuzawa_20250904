"""
evaluate.py – evaluation metrics and plotting helpers
"""
from __future__ import annotations
import torch, math
import torch.nn.functional as F

# ---------------------------  OVER-SMOOTHING  ---------------------------

def row_diff(x: torch.Tensor) -> float:
    """Average L2 row difference (Li et al., 2020)"""
    x_n = F.normalize(x, p=2, dim=-1)
    diffs = (x_n.unsqueeze(0) - x_n.unsqueeze(1)).pow(2).sum(-1).sqrt()
    return diffs.mean().item()


def instance_information_gain(x: torch.Tensor) -> float:
    p = F.softmax(x, dim=-1)
    ent = -(p * p.log()).sum(-1)
    return ent.mean().item()

# ---------------------------  SCALABILITY  ---------------------------

def gpu_mem() -> float:
    """Peak reserved GPU memory (GB)"""
    return torch.cuda.max_memory_reserved() / 1024 ** 3 if torch.cuda.is_available() else 0.0

# ---------------------------  PLOTTING  ---------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def save_lineplot(xs, ys, xlabel, ylabel, title, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
