from __future__ import annotations

"""
src/evaluate.py
----------------
Evaluation utilities (plots, metrics post-processing, …).

All experiment figures *must* be stored under the path dictated by the
assignment instructions:
    .research/iteration2/images
Regardless of the output directory requested by the caller we therefore
override the destination to comply with that requirement.
"""

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

# Centralised location where *all* figures will be saved.
_IMGS_DIR = Path(".research/iteration2/images")


def _prepare_dir() -> None:
    """Create the images directory the first time it is needed."""
    _IMGS_DIR.mkdir(parents=True, exist_ok=True)


def accuracy_memory_curve(df: pd.DataFrame, _ignored_outdir: str | Path | None = None) -> None:
    """Line plot: Average Accuracy (%) vs buffer memory budget (MB).

    The input dataframe must contain columns: `method`, `mem_mb`, `AA`.

    Notes
    -----
    The original implementation allowed the caller to pick the destination
    folder.  The updated version enforces the standardised location required
    by the automated evaluation harness while remaining *call-signature
    compatible* (the second argument is simply ignored).
    """
    if df.empty:
        print("[WARN] Empty dataframe passed to accuracy_memory_curve – skipping plot.")
        return

    _prepare_dir()

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

    fname = _IMGS_DIR / "accuracy_memory.pdf"
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved figure: {fname}")
