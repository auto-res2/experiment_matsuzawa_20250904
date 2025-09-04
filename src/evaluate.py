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


# All figures must be saved to this directory as per the build rules
_IMAGES_DIR = ".research/iteration2/images"


def plot_curves(train_loss: list[float], val_loss: list[float], out: str = "training_loss.pdf") -> None:  # noqa: E501
    """Plot *train* vs *val* loss curves and save them to *out*.

    The figure is stored directly in ``.research/iteration2/images`` if
    that directory exists, otherwise next to the current working
    directory.
    """
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_loss, label="train", **_DEF_STYLE)
    plt.plot(epochs, val_loss, label="val", **_DEF_STYLE)

    for e, v in zip(epochs, val_loss):
        plt.text(e, v + 0.01, f"{v:.2f}", fontsize=6, ha="center")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    import pathlib

    p = pathlib.Path(_IMAGES_DIR)
    if p.is_dir():  # pragma: no cover – optional convenience
        out = str(p / out)

    plt.savefig(out, bbox_inches="tight")
    print(f"[FIG] training_loss → {out}")
    plt.close()
