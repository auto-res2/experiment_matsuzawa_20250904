from __future__ import annotations
"""
evaluate.py – evaluation utilities, metrics, and diagnostics
"""
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
import torch
from matplotlib import pyplot as plt
from rich import print
from timm import create_model

# -----------------------------------------------------------------------------
# Configuration & paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# -----------------------------------------------------------------------------
#   Metrics
# -----------------------------------------------------------------------------

def worst_group_acc(pred: torch.Tensor, y: torch.Tensor, group_ids: torch.Tensor, n_groups: int = 4) -> float:
    """Compute worst-group accuracy (lower-bound) as commonly used in WILDS papers."""
    acc = [((pred[group_ids == g] == y[group_ids == g]).float().mean()).item() for g in range(n_groups)]
    return float(min(acc))


# -----------------------------------------------------------------------------
#   Plotting helpers
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# All images must be stored under this directory according to the task rules
_IMAGES_ROOT = Path(".research/iteration22/images")
_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)


def save_bar_plot(series, title: str, fname: Path | str, ylabel: str) -> None:  # type: ignore
    """Save a simple bar plot.

    Irrespective of the *fname* requested by the caller, the figure is saved to
    `.research/iteration22/images/<basename(fname)>` to comply with the
    evaluation framework requirements.
    """
    # Map requested filename to mandated directory while preserving basename
    fname = _IMAGES_ROOT / Path(fname).name
    _ensure_dir(Path(fname))

    plt.figure(figsize=(4, 3))
    bars = plt.bar(range(len(series)), series.values * 100, tick_label=series.index)
    for b in bars:
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# -----------------------------------------------------------------------------
# Diagnostics: invariance check (Δ logits after swapping)
# -----------------------------------------------------------------------------

def diagnostics() -> None:
    from src.preprocess import ContextSwapper  # avoid top-level heavy deps
    from wilds import get_dataset
    from wilds.common.data_loaders import get_eval_loader
    import torchvision.transforms as T

    ckpt_dir = ROOT / "models"
    erm_w = ckpt_dir / "wb_erm.pt"
    auto_w = ckpt_dir / "wb_autospuswap.pt"
    if not (erm_w.exists() and auto_w.exists()):
        print("diagnostics skipped – checkpoints missing")
        return

    erm = create_model(CONFIG["model"]["backbone"], pretrained=False, num_classes=CONFIG["model"]["num_classes"])
    auto = create_model(CONFIG["model"]["backbone"], pretrained=False, num_classes=CONFIG["model"]["num_classes"])
    erm.load_state_dict(torch.load(erm_w, map_location=DEVICE))
    auto.load_state_dict(torch.load(auto_w, map_location=DEVICE))
    erm, auto = erm.to(DEVICE, dtype=DTYPE), auto.to(DEVICE, dtype=DTYPE)
    erm.eval(); auto.eval()

    # ------------------------------------------------------------------
    # Use version from config (default 1.0) instead of hard-coding 2.0
    # ------------------------------------------------------------------
    wb_version: str = str(CONFIG["dataset"]["waterbirds"].get("version", "1.0"))

    ds = get_dataset("waterbirds", version=wb_version, root_dir=str(ROOT / "data"))
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    val = ds.get_subset("val", transform=tf)
    loader = get_eval_loader("standard", val, batch_size=32, num_workers=2)

    swapper = ContextSwapper(1.0)
    deltas: Dict[str, list[float]] = {"erm": [], "auto": []}
    with torch.no_grad():
        for b in loader:
            x = b["images"].to(DEVICE, dtype=DTYPE)
            idx = b["index"].to(DEVICE)
            x_cf, _ = swapper(x, idx)
            deltas["erm"].append((erm(x) - erm(x_cf)).pow(2).mean().sqrt().item())
            deltas["auto"].append((auto(x) - auto(x_cf)).pow(2).mean().sqrt().item())

    out = {
        "Δlogits_ERM": float(np.mean(deltas["erm"])),
        "Δlogits_AutoSpuSwap": float(np.mean(deltas["auto"])),
    }
    p = ROOT / "results" / "invariance_diag.json"
    _ensure_dir(p)
    p.write_text(json.dumps(out, indent=2))
    print("saved diagnostics to", p)
