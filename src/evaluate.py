"""src/evaluate.py
Evaluation, statistics, and plotting helpers.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def lineplot(values: Sequence[float], ylabel: str, fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 2))
    plt.plot(values, label=ylabel, lw=2)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.2f}", fontsize=6, ha="center")
    plt.xlabel("iter")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", format="pdf")
    plt.close()
