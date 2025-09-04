"""src/evaluate.py
Metrics and plotting utilities.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

__all__ = ["accuracy", "row_diff", "save_loss_plot"]

# ----------------------------------------------------------------------------
#  Metrics
# ----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def row_diff(emb: torch.Tensor) -> float:
    emb_n = F.normalize(emb, p=2, dim=-1)
    diffs = (emb_n.unsqueeze(0) - emb_n.unsqueeze(1)).pow(2).sum(-1).sqrt()
    return diffs.mean().item()

# ----------------------------------------------------------------------------
#  Plotting helpers
# ----------------------------------------------------------------------------

def save_loss_plot(losses: Dict[str, List[float]], fig_path: Path):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for label, curve in losses.items():
        xs = list(range(1, len(curve) + 1))
        plt.plot(xs, curve, marker="o", label=label)
        for x_, y_ in zip(xs, curve):
            plt.annotate(f"{y_:.2f}", (x_, y_), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=6)
    plt.xlabel("Epoch"); plt.ylabel("Loss");
    plt.title("Training loss curve"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
