"""src/train.py
Training loop and utilities used by all experiments.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch

from .evaluate import evaluate, save_figures

# -----------------------------------------------------------------------------
# AMP is enabled by default on CUDA devices that support it.  Fallback to FP32
# on CPU or older GPUs.
# -----------------------------------------------------------------------------

def _prepare_amp(cfg: Dict[str, Any]) -> torch.cuda.amp.GradScaler:
    """Return a GradScaler that is enabled only when autocast should be used."""
    amp_requested = bool(cfg.get("amp", True))
    device_is_cuda = torch.cuda.is_available() and torch.cuda.current_device() >= 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp_requested and device_is_cuda)
    return scaler


# -----------------------------------------------------------------------------
#  Main training function – one model / one seed
# -----------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    loader_tr: torch.utils.data.DataLoader,
    loader_val: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    exp_name: str,
    method_name: str,
    seed: int,
) -> Dict[str, Any]:
    """Train ``model`` for a single run and return the evaluation dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network to train.  Will be **moved** onto ``device``.
    loader_tr / loader_val : DataLoader
        Training / validation loaders.
    cfg : Dict[str, Any]
        Hyper-parameters *including* ``lr, momentum, weight_decay, num_epochs``.
    device : torch.device
    exp_name / method_name / seed : str | int
        Identifiers used for checkpoint & figure file names.
    """

    model = model.to(device)
    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    scaler = _prepare_amp(cfg)

    best_val = 0.0
    ckpt_path = Path(f"ckpt_{exp_name}_{method_name}_seed{seed}.pt")

    for epoch in range(int(cfg["num_epochs"])):
        model.train()
        running_loss, seen = 0.0, 0

        for x, y in loader_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                if method_name.lower() == "cader" and hasattr(model, "loss_fn"):
                    loss, _ = model.loss_fn(logits, y)
                else:
                    loss = torch.nn.functional.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / max(seen, 1)
        val_res = evaluate(model, loader_val, device)
        val_acc = val_res["acc"]
        print(
            f"[{exp_name}] {method_name} – seed {seed}  epoch {epoch:03d}  "
            f"loss {train_loss:.3f}  val-acc {val_acc:.3f}"
        )

        # Save best checkpoint
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)

    # ---------------------------------------------------------------------
    # Finished training – load best model & return final metrics
    # ---------------------------------------------------------------------
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    final_res = evaluate(model, loader_val, device, detailed=True)

    # Persist figures
    save_figures(final_res, f"{exp_name}_{method_name}_seed{seed}")
    return final_res
