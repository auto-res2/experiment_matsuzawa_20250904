"""src/evaluate.py
Model evaluation utilities + simple plotting helpers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
#  Core evaluation – returns accuracy / loss.  If ``detailed`` is True the
#  dictionary is augmented with additional diagnostics for later analysis.
# -----------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    detailed: bool = False,
) -> Dict[str, Any]:
    model.eval()
    correct = total = 0
    losses = []
    ce = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = ce(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            losses.append(loss.item() * y.size(0))

    acc = correct / max(total, 1)
    res: Dict[str, Any] = {
        "acc": acc,
        "n": total,
        "loss": sum(losses) / max(total, 1),
    }

    if detailed:
        res["all_preds"] = None  # place-holder – could store logits / preds here

    return res


# -----------------------------------------------------------------------------
#  Simple bar plot – saves to ``figures/accuracy_<tag>.pdf``.
# -----------------------------------------------------------------------------

def save_figures(res_dict: Dict[str, Any], tag: str) -> None:
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0], [res_dict["acc"]])
    ax.set_xticks([0])
    ax.set_xticklabels(["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(tag)

    # annotate
    ax.text(0, res_dict["acc"] + 0.01, f"{res_dict['acc'] * 100:.1f}%", ha="center")

    out_path = Path("figures") / f"accuracy_{tag}.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print("[evaluate] Saved figure:", out_path)
    plt.close(fig)
