import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

__all__ = [
    "accuracy",
    "effective_rank",
    "gradient_dispersion_ratio",
    "evaluate",
    "plot_curve",
]

# -----------------------------------------------------------------------------
#                                Basic metrics
# -----------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Node-wise classification accuracy (top-1)."""
    return (pred.argmax(dim=-1) == y).float().mean().item()


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:  # noqa: D401
    """Shannon-entropy based effective rank (Roy & Vetterli, 2007).

    The implementation follows the definition

        erank = exp(H) / d,

    where ``H`` is the Shannon entropy of the normalised singular values and
    ``d`` is the dimensionality of the feature matrix (number of columns).
    """
    # Singular values of the feature matrix (GPU-friendly & differentiable).
    _, s, _ = torch.linalg.svd(z, full_matrices=False)

    # Normalised (probability) spectrum.
    p = (s / s.sum()).clamp_min(eps)

    # Shannon entropy – bring to Python float early to safely use math.exp.
    h: float = -(p * p.log()).sum().item()

    # Effective rank.
    return math.exp(h) / z.size(1)


# -----------------------------------------------------------------------------
#                      Additional analysis helpers (per-layer)
# -----------------------------------------------------------------------------

def gradient_dispersion_ratio(per_layer_feats: List[torch.Tensor]) -> List[float]:
    """A toy *gradient dispersion ratio* (GDR) proxy.

    Since we do not have the actual gradients at evaluation time, we use the
    coefficient of variation of node activations within each layer as an
    inexpensive dispersion-like statistic.  This is **not** the exact GDR as
    defined in the literature, but suffices for progress monitoring in this
    template.
    """
    gdrs: List[float] = []
    for feat in per_layer_feats:
        std = feat.std(dim=0)
        mean = feat.abs().mean(dim=0).clamp_min(1e-6)
        gdrs.append(float((std / mean).mean().item()))
    return gdrs


# -----------------------------------------------------------------------------
#                             Evaluation routine
# -----------------------------------------------------------------------------

def evaluate(model, data):  # noqa: D401
    """Run a forward pass and compute metrics.

    Returns
    -------
    acc : float
        Accuracy computed on the *validation* mask if present, otherwise on the
        *test* mask (fallback) – or on **all** nodes if neither mask exists.
    gdr : List[float]
        Gradient-dispersion-ratio proxy for every layer.
    er  : List[float]
        Effective rank for every layer.
    """
    model.eval()
    with torch.no_grad():
        out, per_layer_feats = model(data)

    # Select the evaluation split.
    if hasattr(data, "val_mask") and data.val_mask.any():
        mask = data.val_mask
    elif hasattr(data, "test_mask") and data.test_mask.any():
        mask = data.test_mask
    else:
        mask = slice(None)  # all nodes

    acc = accuracy(out[mask], data.y[mask])
    gdr = gradient_dispersion_ratio(per_layer_feats)
    er = [effective_rank(z) for z in per_layer_feats]
    return acc, gdr, er


# -----------------------------------------------------------------------------
#                                Plot helper
# -----------------------------------------------------------------------------

def plot_curve(values: List[float], title: str, ylabel: str, out_path: Path | str):
    """Save a simple line plot to *./research/iteration4/images*.

    The ``out_path`` argument coming from caller code is ignored **except** for
    its file name – the destination directory is mandated by the task
    description.
    """
    img_dir = Path(".research/iteration4/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(out_path)
    save_path = img_dir / out_path.name  # enforce required directory

    plt.figure(figsize=(4, 3))
    plt.plot(values, marker="o")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.grid(True, ls=":", lw=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # Optional: return the path for logging if needed.
    return save_path