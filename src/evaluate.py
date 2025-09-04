# src/evaluate.py
"""Evaluation & plotting helpers."""
from __future__ import annotations

import numpy as np
import matplotlib

# Head-less backend is mandatory on many servers (no DISPLAY)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

__all__ = [
    "plot_curves",
]


_DEF_STYLE = dict(lw=2)

# -----------------------------------------------------------------------------
# All figures must be saved to this directory as per the build rules
# -----------------------------------------------------------------------------
_IMAGES_DIR = ".research/iteration4/images"  # ← updated to *iteration4*


def _ensure_images_dir() -> None:
    """Create the shared images directory if it does not yet exist."""
    import pathlib

    path = pathlib.Path(_IMAGES_DIR)
    path.mkdir(parents=True, exist_ok=True)


def plot_curves(
    train_loss: list[float],
    val_loss: list[float],
    out: str = "training_loss.pdf",
) -> None:  # noqa: E501
    """Plot *train* vs *val* loss curves and save them to *out*.

    The figure is stored directly in ``.research/iteration4/images`` as
    required by the build rules.  The directory is created on-the-fly if
    it does not already exist.
    """
    _ensure_images_dir()
    import pathlib

    out_path = pathlib.Path(_IMAGES_DIR) / out

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_loss, label="train", **_DEF_STYLE)
    plt.plot(epochs, val_loss, label="val", **_DEF_STYLE)

    for e, v in zip(epochs, val_loss):
        plt.text(e, v + 0.01, f"{v:.2f}", fontsize=6, ha="center")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig(out_path, bbox_inches="tight")
    print(f"[FIG] training_loss → {out_path}")
    plt.close()
