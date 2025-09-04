"""src/train.py
Model-training related classes and functions.
The code is **directly refactored** from the single-file implementation so
logic, default values and public APIs stay identical.
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from rich import print
from timm import create_model

from .evaluate import (
    compute_wg_acc,
    expected_calibration_error,
    eval_waterbirds,
    save_bar,
)
from .preprocess import get_waterbirds_loaders

# ────────────────────────────────────────────────────────────
#  Paths & configuration
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file {CONFIG_PATH} missing – please add one before running."    )
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG: Dict[str, Any] = yaml.safe_load(f)

DATA_DIR = ROOT / CFG["dataset"]["waterbirds"]["root_dir"]
CKPT_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
AMP_CONTEXT = (
    torch.autocast  # type: ignore[attr-defined]
    if DEVICE == "cuda"
    else torch.cpu.amp.autocast  # type: ignore[attr-defined]
)

# ImageNet statistics kept exactly as in the monolithic script
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

# ────────────────────────────────────────────────────────────
#  1.  Utilities
# ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Ensure full determinism across NumPy / Python / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────
#  2.  Augmentations / auxiliary modules
# ────────────────────────────────────────────────────────────


class FourierPerturb(nn.Module):
    """Fast amplitude/phase jitter in the Fourier domain."""

    def __init__(self, phase_max: float = 0.2, amp_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        super().__init__()
        self.phase_max = phase_max
        self.amp_lo, self.amp_hi = amp_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        with torch.no_grad():
            x_fft = torch.fft.rfft2(x, norm="ortho")
            mag, phase = torch.abs(x_fft), torch.angle(x_fft)
            phase += (torch.rand_like(phase) * 2 - 1) * self.phase_max
            mag *= torch.rand_like(mag) * (self.amp_hi - self.amp_lo) + self.amp_lo
            out = torch.fft.irfft2(mag * torch.exp(1j * phase), s=x.shape[-2:], norm="ortho")
            return out.clamp_(min=x.min(), max=x.max())


# ------------------------------------------------------------------
#  Dummy SAM-replacement mask generator (keeps code self-contained)
# ------------------------------------------------------------------


class DummyMaskBank:
    """Produces a left-half object / right-half context mask."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        m = torch.zeros_like(x[:, :1])
        m[:, :, :, : x.size(-1) // 2] = 1.0
        return m


try:
    import segment_anything  # noqa: F401, pylint: disable=unused-import

    HAVE_SAM = True
except ImportError:  # pragma: no cover – SAM is optional
    HAVE_SAM = False


class ContextSwapper(nn.Module):
    """Swaps background regions between two randomly permuted images."""

    def __init__(self, swap_prob: float = 1.0):
        super().__init__()
        self.swap_prob = swap_prob
        self.mask_bank = DummyMaskBank()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.swap_prob == 0.0:
            return x, torch.zeros_like(x[:, :1])
        perm = torch.randperm(x.size(0), device=x.device)
        x_perm = x[perm]
        mask = self.mask_bank(x)
        ctx_mask = 1.0 - mask
        x_cf = x * mask + x_perm * ctx_mask
        return x_cf, mask


class AutoSpuSwapModule(nn.Module):
    """Composite loss wrapper used in AutoSpuSwap training."""

    def __init__(self, swap_prob: float, lambda_consistency: float, lambda_fourier: float):
        super().__init__()
        self.swapper = ContextSwapper(swap_prob)
        self.lambda_cons = lambda_consistency
        self.lambda_fourier = lambda_fourier
        self.fourier = FourierPerturb()
        self.mse = nn.MSELoss()

    def forward(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits = model(x)
        loss_cls = F.cross_entropy(logits, y)
        x_cf, _ = self.swapper(x, y)
        if self.lambda_fourier > 0:
            x_cf = self.fourier(x_cf)
        logits_cf = model(x_cf)
        loss_cons = self.mse(logits, logits_cf)
        loss = loss_cls + self.lambda_cons * loss_cons
        return loss, {
            "loss_cls": loss_cls.detach(),
            "loss_cons": loss_cons.detach(),
            "Δlogits": (logits - logits_cf).pow(2).mean().sqrt().detach(),
        }


# ------------------------------------------------------------------
#  3.  Baseline losses
# ------------------------------------------------------------------

class IRMLoss(nn.Module):
    def __init__(self, penalty_weight: float = 1_000.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.dummy_w = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: D401
        scale = self.dummy_w
        loss = F.cross_entropy(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        penalty = grad.pow(2)
        return loss + self.penalty_weight * penalty


class RExLoss(nn.Module):
    def __init__(self, penalty_weight: float = 10.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:  # noqa: D401
        mean = torch.stack(losses).mean()
        var = torch.stack([(l - mean).pow(2) for l in losses]).mean()
        return mean + self.penalty_weight * var


class GroupDROLoss:
    """Implements exponentiated gradient update from Sagawa et al."""

    def __init__(self, n_groups: int, eta: float = 0.2):
        self.eta = eta
        self.q = torch.zeros(n_groups, device=DEVICE)

    def update(self, group_losses: torch.Tensor) -> torch.Tensor:  # noqa: D401
        self.q = self.q * torch.exp(self.eta * group_losses.detach())
        self.q = self.q / self.q.sum()
        return (self.q * group_losses).sum()


# ------------------------------------------------------------------
#  4.  Model factory – centralised to ensure identical init per seed
# ------------------------------------------------------------------

def build_model() -> nn.Module:
    model_cfg = CFG["model"]
    m = create_model(model_cfg["name"], pretrained=model_cfg.get("pretrained", True))
    # Waterbirds has exactly 2 classes
    m.reset_classifier(num_classes=2)
    return m.to(DEVICE, dtype=DTYPE)


# ------------------------------------------------------------------
#  5.  Waterbirds experiment (EXP-1)
# ------------------------------------------------------------------

def run_waterbirds() -> None:  # noqa: D401
    """Full 5-seed Waterbirds experiment with all baselines."""

    loaders_fn, dataset_meta = get_waterbirds_loaders(
        batch_size=CFG["training"]["batch_size"],
        train_tf=T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMNET_MEAN, IMNET_STD),
            ]
        ),
        val_tf=T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMNET_MEAN, IMNET_STD),
            ]
        ),
    )
    n_groups = dataset_meta["n_groups"]

    rows: List[Dict[str, Any]] = []

    for seed in CFG["training"]["seeds"]["waterbirds"]:
        set_seed(seed)
        loaders = loaders_fn()

        # identical starting weights per seed
        backbone_state = build_model().state_dict()

        for method in [
            "ERM",
            "IRM",
            "REx",
            "GroupDRO",
            "DFR",
            "AutoSpuSwap",
            "swap0",
        ]:
            model = build_model()
            model.load_state_dict(backbone_state)
            optimiser = torch.optim.AdamW(
                model.parameters(),
                lr=CFG["training"]["lr"],
                weight_decay=CFG["training"]["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=CFG["training"]["epochs"]
            )

            # helpers instantiation ------------------------------------------------------------
            autospu = None
            irm_loss = None
            rex_loss = None
            gdro_helper = None
            if method.startswith("AutoSpuSwap"):
                autospu = AutoSpuSwapModule(
                    swap_prob=1.0 if method == "AutoSpuSwap" else 0.0,
                    lambda_consistency=CFG["loss"]["autospu"]["lambda_consistency"],
                    lambda_fourier=CFG["loss"]["autospu"]["lambda_fourier"],
                )
            elif method == "IRM":
                irm_loss = IRMLoss()
            elif method == "REx":
                rex_loss = RExLoss()
            elif method == "GroupDRO":
                gdro_helper = GroupDROLoss(n_groups)

            # ----------------------- training loop -------------------------------------------
            for epoch in range(CFG["training"]["epochs"]):
                model.train()
                for batch in loaders["train"]:
                    x = batch["images"].to(DEVICE, dtype=DTYPE)
                    y = batch["y"].to(DEVICE)
                    g = batch["metadata"][:, 0].to(DEVICE)

                    optimiser.zero_grad(set_to_none=True)
                    with AMP_CONTEXT(device_type=DEVICE, dtype=DTYPE):  # type: ignore[misc]
                        if autospu is not None:
                            loss, _ = autospu(model, x, y)
                        else:
                            logits = model(x)
                            if irm_loss is not None:
                                loss = irm_loss(logits, y)
                            elif rex_loss is not None:
                                loss_env = [
                                    F.cross_entropy(logits[i :: 2], y[i :: 2]) for i in (0, 1)
                                ]
                                loss = rex_loss(loss_env)
                            elif gdro_helper is not None:
                                group_losses = torch.zeros(n_groups, device=DEVICE)
                                for gid in range(n_groups):
                                    idx = g == gid
                                    if idx.any():
                                        group_losses[gid] = F.cross_entropy(logits[idx], y[idx])
                                loss = gdro_helper.update(group_losses)
                            else:
                                loss = F.cross_entropy(logits, y)

                    loss.backward()
                    optimiser.step()
                scheduler.step()

                # quick val every 5 epochs
                if (epoch + 1) % 5 == 0:
                    id_acc, wg_acc = eval_waterbirds(model, loaders["val"], n_groups)
                    print(
                        f"seed{seed} [{method}] epoch{epoch+1:02d}  val-acc={id_acc:.2%}  WG={wg_acc:.2%}"
                    )

            # ----------------------- final validation ---------------------------------------
            logits_all, y_all, g_all = [], [], []
            model.eval()
            for batch in loaders["val"]:
                x = batch["images"].to(DEVICE, dtype=DTYPE)
                y = batch["y"].to(DEVICE)
                g = batch["metadata"][:, 0].to(DEVICE)
                with AMP_CONTEXT(device_type=DEVICE, dtype=DTYPE):  # type: ignore[misc]
                    logits = model(x)
                logits_all.append(logits.cpu())
                y_all.append(y.cpu())
                g_all.append(g.cpu())
            logits_all = torch.cat(logits_all)
            y_all = torch.cat(y_all)
            g_all = torch.cat(g_all)
            pred_all = logits_all.argmax(1)
            id_acc = (pred_all == y_all).float().mean().item()
            wg_acc = compute_wg_acc(pred_all, y_all, g_all, n_groups)
            ece = expected_calibration_error(logits_all, y_all)

            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "id_acc": id_acc,
                    "wg_acc": wg_acc,
                    "ece": ece,
                }
            )

            # save checkpoint for seed0 to save disk space
            if seed == 0:
                torch.save(model.state_dict(), CKPT_DIR / f"wb_{method.lower()}.pt")

    # ----------------------- aggregation & plots --------------------------------------------
    df = pd.DataFrame(rows)
    results_csv = RESULTS_DIR / "waterbirds_full.csv"
    df.to_csv(results_csv, index=False)
    print(f"[green]Saved per-seed CSV → {results_csv}")

    means = df.groupby("method")[["id_acc", "wg_acc"]].mean()
    save_bar(means["wg_acc"], "worst_group_accuracy.pdf", ylabel="WG-Acc (%)", multiply=100)

    print("\n[bold]Waterbirds summary:[/]")
    print(means)


# ------------------------------------------------------------------
#  Optional entry point (useful for pytest-style execution)
# ------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    tic = time.time()
    run_waterbirds()
    print(f"[bold green]Finished Waterbirds experiment in {time.time() - tic:.1f}s.")
