"""src/evaluate.py
Evaluation helpers (metrics & tiny visualisations).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

sns.set_theme(style="whitegrid")


def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Standard top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def save_bar(value: float, name: str, out_dir: Union[str, Path]) -> None:
    """Save a minimalist bar plot so that CI stores a visual artefact."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar([0], [value])
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Accuracy")
    ax.set_title(name)
    ax.text(0, min(value + 0.02, 0.98), f"{value * 100:.1f}%", ha="center")

    pdf = out_dir / f"accuracy_{name}.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    print("[figure] saved", pdf)
    plt.close(fig)
