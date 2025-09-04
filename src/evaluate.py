from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 – after Agg backend set

# ------------------------------------------------------------------
#  Core metrics
# ------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Simple top-1 accuracy."""
    return float((logits.argmax(dim=-1) == y).float().mean())


def effective_rank(x: torch.Tensor) -> float:
    """Full effective rank on CPU (or sampled) – safe against nans."""
    x_ = x.detach().cpu()
    # Low-rank SVD to keep memory low – fall back to full if tiny dims
    q = min(128, max(1, x_.shape[1] - 1))
    u, s, v = (
        torch.linalg.svd(x_, full_matrices=False)
        if q == 0
        else torch.linalg.svd(x_[:, :q], full_matrices=False)
    )
    p = (s ** 2) / (s ** 2).sum()
    er = torch.exp(-(p * torch.log(p + 1e-9)).sum()) / x_.shape[1]
    return float(er)


def group_distance_ratio(x: torch.Tensor, y: torch.Tensor) -> float:
    """Average ratio of within-class to between-class feature distances.

    For a mini-batch of N nodes (N ≤ 2 048 in the current call-sites)
    we compute the pair-wise Euclidean distance matrix ∈ R^{N×N} and
    take the mean distance of node pairs that share the same label
    (within-class) versus pairs of different labels (between-class).

    To keep memory usage predictable we operate on CPU and detach all
    inputs from the computation graph – the metric is logging-only.
    """
    with torch.no_grad():
        # Move to CPU to avoid GPU sync & make memory accounting easier
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        n = x_cpu.size(0)

        # Pair-wise euclidean distances –  float32 (n≤2 048 ⇒ ≤16 MB)
        diff = x_cpu.unsqueeze(0) - x_cpu.unsqueeze(1)  # (N, N, D)
        dist = diff.pow(2).sum(-1).sqrt()  # (N, N)

        # Masks for same / different labels
        same_mask = y_cpu.unsqueeze(0) == y_cpu.unsqueeze(1)
        same_mask.fill_diagonal_(False)  # ignore trivial zero-distance
        diff_mask = ~same_mask

        same_mean = dist[same_mask].mean()
        diff_mean = dist[diff_mask].mean()

        ratio = (same_mean - diff_mean).abs() / (
            same_mean.abs() + diff_mean.abs() + 1e-9
        )
        return float(ratio)

# ------------------------------------------------------------------
#  Figure helper
# ------------------------------------------------------------------

# Enforce the updated output directory (see task instructions)
_SAVE_DIR = Path(".research/iteration5/images")


def save_lineplot(
    xs, ys: Dict[str, List[float]], xlabel: str, ylabel: str, title: str, fname: str
):
    """Persist a simple line-plot to the mandated output directory."""
    plt.figure()
    for label, y_vals in ys.items():
        plt.plot(xs, y_vals, marker="o", label=label)
        for xi, yi in zip(xs, y_vals):
            plt.text(xi, yi, f"{yi:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _SAVE_DIR / fname
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out_path}")
