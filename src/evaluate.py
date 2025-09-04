import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .train import DEVICE  # reuse the same device constant

# ---------------------------------------------------------
#  FIGURE OUTPUT DIRECTORY – MUST MATCH SPECIFICATION
# ---------------------------------------------------------
_IMG_DIR = Path(".research/iteration2/images")
_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
#  EVALUATION
# ---------------------------------------------------------

def evaluate(model: torch.nn.Module, loader) -> float:
    """Return accuracy (%) on the provided loader."""
    model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for imgs, y in loader:
            imgs, y = imgs.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits, _ = model(imgs)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            n += y.numel()
    return 100.0 * correct / max(n, 1)


# ---------------------------------------------------------
#  PLOTTING
# ---------------------------------------------------------

def plot_curve(xs, ys, xlab: str, ylab: str, title: str, fname: str):
    """Generic line plot with value annotations."""
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}", (x, y))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    out_path = _IMG_DIR / fname
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"[Figure saved] {out_path}")
    plt.close()
