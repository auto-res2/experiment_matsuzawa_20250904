"""
train.py – model, losses, and training loop for the AutoSpuSwap experiments
The file contains only code that is strictly necessary for modelling and training.
All heavy lifting (data-loading, evaluation, plotting) is delegated to the other
modules so that this file can stay focussed on the learning logic.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from timm import create_model
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"
CONFIG: Dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    import random, numpy as np  # local import – keeps global namespace clean

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#  Model building blocks
# -----------------------------------------------------------------------------

def get_backbone() -> nn.Module:
    """Create a timm backbone and move it to the correct device/precision."""
    model = create_model(
        CONFIG["model"]["backbone"], pretrained=True, num_classes=CONFIG["model"]["num_classes"]
    )
    return model.to(device=DEVICE, dtype=DTYPE)


class FourierPerturb(nn.Module):
    """Light-weight spectrum perturbation used for style/Fourier robustness."""

    def __init__(self, phase_max: float = 0.2, amp_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        super().__init__()
        self.phase_max = phase_max
        self.amp_lo, self.amp_hi = amp_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            X = torch.fft.rfft2(x, norm="ortho")
            mag, phi = torch.abs(X), torch.angle(X)
            phi += (torch.rand_like(phi) * 2 - 1) * self.phase_max
            mag *= torch.rand_like(mag) * (self.amp_hi - self.amp_lo) + self.amp_lo
            out = torch.fft.irfft2(mag * torch.exp(1j * phi), s=x.shape[-2:], norm="ortho")
            return out.clamp_(0, 1)


class ContextSwapper(nn.Module):
    """Implements the context-swapping counterfactual generation step."""

    def __init__(self, swap_prob: float) -> None:
        super().__init__()
        from src.preprocess import MaskBank  # local to avoid circular import

        self.p = swap_prob
        self.bank = MaskBank()

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if self.p == 0:
            return x, torch.zeros_like(x[:, :1])
        # random permutation within the mini-batch
        perm = torch.randperm(x.size(0), device=x.device)
        masks = self.bank(idx).to(device=x.device, dtype=x.dtype)  # (B,1,224,224)
        ctx = 1.0 - masks
        x_perm = x[perm]
        x_swapped = x * masks + x_perm * ctx
        return x_swapped, masks


class AutoSpuSwapLoss(nn.Module):
    """Combined CE + consistency + Fourier invariance loss used for training."""

    def __init__(self) -> None:
        super().__init__()
        autospu = CONFIG["autospu"]
        self.swapper = ContextSwapper(autospu["swap_prob"])
        self.l_cons = autospu["lambda_consistency"]
        self.l_four = autospu["lambda_fourier"]
        self.fourier = FourierPerturb()
        self.mse = nn.MSELoss()

    def forward(  # type: ignore[override]
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = model(x)
        loss_ce = F.cross_entropy(logits, y)

        # generate counterfactuals
        x_cf, _ = self.swapper(x, idx)
        if self.l_four > 0:
            x_cf = self.fourier(x_cf)
        logits_cf = model(x_cf)
        loss_cons = self.mse(logits, logits_cf)

        total = loss_ce + self.l_cons * loss_cons
        log_dict = {"loss_ce": loss_ce.detach(), "loss_cons": loss_cons.detach()}
        return total, log_dict


# -----------------------------------------------------------------------------
#  Training loop – Waterbirds experiment
# -----------------------------------------------------------------------------

def _extract_group(meta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """group = 2 * y + env  (definition from WILDS-Waterbirds v2.0)."""
    env = meta[:, 1]
    return y * 2 + env


def run_waterbirds() -> None:
    """Full training loop over all seeds + methods (ERM / AutoSpuSwap / swap0)."""
    from src.preprocess import get_waterbirds_loaders  # local import avoids circular dep
    from src.evaluate import worst_group_acc, save_bar_plot

    results: List[Dict[str, Any]] = []
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in CONFIG["training"]["seeds"]:
        set_seed(seed)
        loaders, dataset = get_waterbirds_loaders(CONFIG["training"]["batch_size"], seed)

        shared_init: Dict[str, torch.Tensor] | None = None  # weight snapshot for fair init
        for method in ["ERM", "AutoSpuSwap", "swap0"]:
            model = get_backbone()
            if shared_init is None:
                shared_init = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                model.load_state_dict(shared_init, strict=True)

            optimiser = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG["training"]["lr"],
                weight_decay=CONFIG["training"]["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=CONFIG["training"]["epochs"]
            )
            criterion: nn.Module
            if method == "ERM":
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = AutoSpuSwapLoss().to(device=DEVICE)

            tic = time.time()
            for epoch in range(CONFIG["training"]["epochs"]):
                model.train()
                pbar = tqdm(loaders["train"], desc=f"seed{seed}-{method}-e{epoch+1}", leave=False)
                for batch in pbar:
                    optimiser.zero_grad(set_to_none=True)
                    x = batch["images"].to(device=DEVICE, dtype=DTYPE)
                    y = batch["y"].to(device=DEVICE)
                    idx = (
                        batch["index"]
                        if "index" in batch
                        else torch.arange(x.size(0), device=DEVICE)
                    )

                    if method == "ERM":
                        loss = criterion(model(x), y)
                    elif method == "swap0":
                        # turn off swapping but keep Fourier etc. intact
                        criterion.swapper.p = 0.0
                        loss, _ = criterion(model, x, y, idx)
                    else:
                        loss, _ = criterion(model, x, y, idx)

                    loss.backward()
                    optimiser.step()
                scheduler.step()

            # ---------------- Validation ----------------
            model.eval()
            preds, ys, gs = [], [], []
            with torch.no_grad():
                for batch in loaders["val"]:
                    x = batch["images"].to(device=DEVICE, dtype=DTYPE)
                    y = batch["y"].to(device=DEVICE)
                    logits = model(x)
                    preds.append(logits.argmax(1).cpu())
                    ys.append(y.cpu())
                    gs.append(_extract_group(batch["metadata"], y).cpu())

            pred = torch.cat(preds)
            y_all = torch.cat(ys)
            g_all = torch.cat(gs)
            id_acc = (pred == y_all).float().mean().item()
            wg_acc = worst_group_acc(pred, y_all, g_all)
            elapsed = time.time() - tic
            results.append(
                {
                    "seed": seed,
                    "method": method,
                    "id_acc": id_acc,
                    "wg_acc": wg_acc,
                    "time": elapsed,
                }
            )
            # first seed – save checkpoint for later diagnostics
            if seed == 0:
                ckpt_dir = ROOT / "models"
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                torch.save(model.state_dict(), ckpt_dir / f"wb_{method.lower()}.pt")

            print(
                f"seed{seed} {method}:  ID {id_acc:.3f}  WG {wg_acc:.3f}  time {elapsed:.1f}s"
            )

    # ---------- Persist and plot summary ----------
    import pandas as pd

    df = pd.DataFrame(results)
    csv_path = results_dir / "waterbirds_full.csv"
    df.to_csv(csv_path, index=False)
    print("[green]Saved", csv_path)

    save_bar_plot(
        series=df.groupby("method")["wg_acc"].mean(),
        title="Worst-Group Accuracy",
        fname=results_dir / "training_wg_acc.pdf",
        ylabel="WG-Acc (%)",
    )
