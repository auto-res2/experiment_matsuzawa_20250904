"""
src/evaluate.py
----------------
Evaluation utilities (plots, metrics post-processing, …).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # enforce headless backend
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


# -----------------------------------------------------------------------------
#  FIGURES
# -----------------------------------------------------------------------------

def accuracy_memory_curve(df: pd.DataFrame, outdir: str | Path) -> None:
    """Line plot: Average Accuracy (%) vs buffer memory budget (MB).

    The input dataframe must contain columns: `method`, `mem_mb`, `AA`.
    """
    if df.empty:
        print("[WARN] Empty dataframe passed to accuracy_memory_curve – skipping plot.")
        return

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, g in df.groupby("method"):
        g_sorted = g.sort_values("mem_mb")
        ax.plot(g_sorted["mem_mb"], g_sorted["AA"], marker="o", label=method)
        for _, r in g_sorted.iterrows():
            ax.annotate(f"{r['AA']:.1f}", (r["mem_mb"], r["AA"]))

    ax.set_xscale("log")
    ax.set_xlabel("Buffer size (MB)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Memory–Accuracy trade-off (Split CIFAR-100)")
    ax.legend()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / "accuracy_memory.pdf"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved figure: {fname}")
