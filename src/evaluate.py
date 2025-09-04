"""src/evaluate.py
Evaluation and analysis helpers – accuracy, effective-rank, plotting …
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
#                               Metrics
# -----------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.argmax(dim=-1) == y).float().mean().item()


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon-entropy based effective rank (Roy & Vetterli, 2007)."""
    # torch.linalg.svd is differentiable and GPU-friendly.
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    p = (s / s.sum()).clamp_min(eps)
    h = -(p * p.log()).sum()
    return math.exp(h).item() / z.size(1)

# -----------------------------------------------------------------------------
#                                Evaluation
# -----------------------------------------------------------------------------

def evaluate(model: nn.Module, data):
    """Return accuracy + per-layer GDR / effective rank."""
    model.eval()
    with torch.no_grad():
        out, layer_feats = model(data)
        acc = accuracy(out[data.test_mask], data.y[data.test_mask])
        gdrs: List[float] = []
        efrs: List[float] = []
        y = data.y
        for h in layer_feats:
            efrs.append(effective_rank(h))
            # Group-Distance-Ratio – ultra-simple implementation
            d = torch.cdist(h, h, p=2)
            intra, inter = [], []
            for i in range(h.size(0)):
                same = y == y[i]
                intra.append(d[i][same].mean())
                inter.append(d[i][~same].mean())
            intra = torch.stack(intra).mean()
            inter = torch.stack(inter).mean()
            gdrs.append((inter / (intra + 1e-6)).item())
        return acc, gdrs, efrs

# -----------------------------------------------------------------------------
#                                Plotting
# -----------------------------------------------------------------------------

def plot_curve(y: List[float], title: str, ylabel: str, pdf_path: Path) -> None:
    xs = list(range(1, len(y) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, y, marker="o", label=title)
    for x, v in zip(xs, y):
        plt.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
    plt.close()
