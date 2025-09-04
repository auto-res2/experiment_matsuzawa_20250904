"""src/evaluate.py
Collection of metric & visualisation helpers used by ``main.py``.
"""
from __future__ import annotations
from pathlib import Path
import math, itertools

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

__all__ = [
    'accuracy',
    'effective_rank',
    'group_distance_ratio',
    'ater',
    'plot_curve',
]

# -----------------------------------------------------------------------------
# Simple helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 categorical accuracy in [0, 1]."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.numel())


def effective_rank(X: torch.Tensor) -> float:
    """Compute the *effective rank* of a matrix (Roy & Vetterli).  The measure
    equals ``exp(H)`` where *H* is the Shannon entropy of normalised singular
    values.
    """
    if X.ndim != 2:
        X = X.flatten(1)
    with torch.no_grad():
        # centre
        Xc = X - X.mean(0, keepdim=True)
        u, s, v = torch.linalg.svd(Xc, full_matrices=False)
        p = s / s.sum()
        # avoid log(0)
        p = p + 1e-12
        H = -(p * torch.log(p)).sum().item()
        return math.exp(H)


def group_distance_ratio(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Ratio of *inter*-class centre distances to *intra*-class spread.
    Simple proxy for how well separated the groups are in the logit space.
    """
    with torch.no_grad():
        unique = torch.unique(labels)
        centres = []
        intra_var = 0.0
        for c in unique:
            mask = labels == c
            feats = logits[mask]
            centre = feats.mean(0, keepdim=True)
            centres.append(centre)
            intra_var += ((feats - centre)**2).mean().item()
        centres = torch.cat(centres, 0)
        # pairwise distances between class centres
        dist = 0.0
        count = 0
        for i, j in itertools.combinations(range(len(unique)), 2):
            dist += F.pairwise_distance(centres[i:i+1], centres[j:j+1]).item()
            count += 1
        inter = dist / max(1, count)
        intra = intra_var / max(1, len(unique))
        return inter / (intra + 1e-8)


def ater(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Approximate *Average Triangle Edge Ratio* – ratio of edges that belong to
    at least one triangle.  Uses a naive O(E * deg) algorithm which is fine for
    < 10⁴ edges (the synthetic graphs are tiny).
    """
    u, v = edge_index  # expected to be 2×E
    adj = {i: set() for i in range(num_nodes)}
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].add(b)
    triangle_edges = 0
    total_edges = u.numel()
    for a, b in zip(u.tolist(), v.tolist()):
        if a == b:
            continue  # self-loop
        if any(c in adj[a] and c in adj[b] for c in adj[a]):
            triangle_edges += 1
    return triangle_edges / max(1, total_edges)

# -----------------------------------------------------------------------------
# Visualisation -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def plot_curve(values: list[float], title: str, ylabel: str, save_path: str | Path):
    """Store a simple line plot of *values* as PDF.*"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 3), dpi=150)
    plt.plot(values, lw=1.2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
