from __future__ import annotations

"""src/evaluate.py
Evaluation, metrics, plotting utilities and diagnostic routines.
Refactored verbatim from the original single-file script.
The former circular import with src.train has been resolved by
removing the top-level dependency on that module.  Only the symbols that
are genuinely required at runtime (ContextSwapper) are imported lazily
inside the diagnostic routine.  DEVICE / DTYPE are now defined locally
so that this file is fully self-contained.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import torch
from matplotlib import pyplot as plt
from rich import print
from scipy.stats import ttest_rel
from timm import create_model

# ------------------------------------------------------------------
#  Hardware helpers – keep identical logic as in train.py but local
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ------------------------------------------------------------------
#  Paths – images are required to live under .research/iteration15/images
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / ".research" / "iteration15" / "images"
CKPT_DIR = ROOT / "models"
for d in (RESULTS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------------

def compute_wg_acc(
    pred: torch.Tensor, y: torch.Tensor, g: torch.Tensor, n_groups: int = 4
) -> float:
    """Worst-group accuracy used for Waterbirds."""
    accs: List[float] = []
    for gid in range(n_groups):
        idx = g == gid
        if idx.sum() == 0:
            continue
        accs.append(((pred[idx] == y[idx]).float().mean()).item())
    return min(accs) if accs else 0.0


def expected_calibration_error(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = logits.softmax(1).max(1)
    acc = pred.eq(y)
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        idx = (conf > bins[i]) & (conf <= bins[i + 1])
        if idx.sum() == 0:
            continue
        ece += (conf[idx].mean() - acc[idx].float().mean()).abs() * idx.float().mean()
    return ece.item()


@torch.no_grad()
def eval_waterbirds(model, loader, n_groups: int):  # noqa: ANN001
    """Return in-distribution and worst-group accuracy on Waterbirds."""
    model.eval()
    y_cat, g_cat, pred_cat, logits_cat = [], [], [], []
    for batch in loader:
        x = batch["images"].to(DEVICE, dtype=DTYPE)
        y = batch["y"].to(DEVICE)
        g = batch["metadata"][:, 0].to(DEVICE)
        logits = model(x)
        pred = logits.argmax(1)
        y_cat.append(y)
        g_cat.append(g)
        pred_cat.append(pred)
        logits_cat.append(logits)
    y_cat = torch.cat(y_cat)
    g_cat = torch.cat(g_cat)
    pred_cat = torch.cat(pred_cat)
    logits_cat = torch.cat(logits_cat)
    id_acc = (pred_cat == y_cat).float().mean().item()
    wg_acc = compute_wg_acc(pred_cat, y_cat, g_cat, n_groups)
    return id_acc, wg_acc


# ------------------------------------------------------------------
#  Plotting helper
# ------------------------------------------------------------------

def save_bar(series: pd.Series, fname: str, ylabel: str, multiply: float = 1.0) -> None:
    plt.figure(figsize=(4, 2.5))
    bars = plt.bar(
        range(len(series)), series.values * multiply, tick_label=series.index, color="steelblue"
    )
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig_path = FIG_DIR / fname
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure:", fig_path)


# ------------------------------------------------------------------
#  Diagnostics (EXP-3)
# ------------------------------------------------------------------

def run_diagnostics() -> None:  # noqa: D401
    """Post-training invariance sanity checks – seed0 checkpoints only."""

    # The ContextSwapper symbol is imported lazily to avoid circular import
    from .train import ContextSwapper  # pylint: disable=import-inside-function

    print("\n[bold cyan]Running invariance diagnostics …[/]")
    ckpt_erm = CKPT_DIR / "wb_erm.pt"
    ckpt_auto = CKPT_DIR / "wb_autospuswap.pt"
    if not (ckpt_erm.exists() and ckpt_auto.exists()):
        print("[red]Checkpoints missing – run experiments first.")
        return

    erm = create_model("resnetv2_50x1_bit.goog_in21k_ft_in1k", pretrained=False)
    auto = create_model("resnetv2_50x1_bit.goog_in21k_ft_in1k", pretrained=False)
    erm.load_state_dict(torch.load(ckpt_erm, map_location=DEVICE))
    auto.load_state_dict(torch.load(ckpt_auto, map_location=DEVICE))
    erm, auto = erm.to(DEVICE, dtype=DTYPE), auto.to(DEVICE, dtype=DTYPE)

    # minimal loader – no extra download  ------------------------------------
    from wilds import get_dataset  # local import avoids hard dependency at module load
    from wilds.common.data_loaders import get_eval_loader
    import torchvision.transforms as T

    wb = get_dataset("waterbirds", version="2.0", root_dir=str(ROOT / "data"), download=False)
    val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    val_loader = get_eval_loader(
        "standard", wb, split="val", batch_size=64, num_workers=2, transform=val_tf
    )

    swapper = ContextSwapper(1.0)
    deltas_erm, deltas_auto = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["images"].to(DEVICE, dtype=DTYPE)
            x_cf, _ = swapper(x, batch["y"])
            logits_e = erm(x)
            logits_a = auto(x)
            logits_e_cf = erm(x_cf)
            logits_a_cf = auto(x_cf)
            deltas_erm.append((logits_e - logits_e_cf).pow(2).mean(1).sqrt())
            deltas_auto.append((logits_a - logits_a_cf).pow(2).mean(1).sqrt())
    delta_erm = torch.cat(deltas_erm).mean().item()
    delta_auto = torch.cat(deltas_auto).mean().item()

    diag = {"Δlogits_ERM": delta_erm, "Δlogits_AutoSpuSwap": delta_auto}
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = RESULTS_DIR / "invariance_diag.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print("Diagnostics saved →", out_path)


# ------------------------------------------------------------------
#  Simple paired t-test helper (may be imported in notebooks)
# ------------------------------------------------------------------

def paired_ttest(csv_path: Path, metric: str, method_a: str, method_b: str):  # noqa: D401
    df = pd.read_csv(csv_path)
    a = df[df.method == method_a][metric]
    b = df[df.method == method_b][metric]
    t_stat, p_val = ttest_rel(a, b)
    print(f"Paired t-test {method_a} vs {method_b} on {metric}:  t={t_stat:.3f}  p={p_val:.4f}")