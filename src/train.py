# src/train.py
"""Training utilities for AutoSpuSwap experiments.
This file contains the generic Trainer class that performs one-epoch
training/validation loops and logs the learning curves.  A very light
"AutoSpu" wrapper is provided so that existing calls to
``autospu.make_counterfactual`` do not crash even if the full
counter-factual generation pipeline is unavailable in the local
environment.  The wrapper can be extended later with the full logic
from the original single-file script (SAM masks, context swap, Fourier
perturbation, diffusion in-painting …)."""
from __future__ import annotations

import time, collections, math, contextlib, warnings
from typing import Dict, Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from .evaluate import plot_curves

__all__ = [
    "Trainer",
    "AutoSpuWrapper",
]


class AutoSpuWrapper:
    """Minimal, *safe* stub that fulfils the interface expected by Trainer.

    The full AutoSpuSwap data-augmentation pipeline is fairly heavy and
    requires additional external libraries (SAM, Stable-Diffusion, …).
    For many quick tests – or on CPU-only machines – researchers may
    want to run the code without the complete pipeline.  This wrapper
    therefore implements a *no-op* ``make_counterfactual`` that simply
    returns the original images.  When the complete implementation is
    available one can monkey-patch / subclass this stub and override the
    method.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        if self.enabled:
            warnings.warn(
                "AutoSpuSwap pipeline not yet implemented – falling back to "
                "identity counter-factuals. Set autospu.enabled: false in the "
                "YAML config to silence this warning.",
                UserWarning,
            )

    def make_counterfactual(self, x: torch.Tensor, *_, **__) -> torch.Tensor:  # noqa: D401,E501
        """Return a counter-factual version of *x*.

        If *enabled* is False the input is returned unchanged.  The
        method signature mirrors the original call in the experimental
        script: ``make_counterfactual(x, y, idx)``.
        """
        # At the moment we always perform a *no-op* to keep the training
        # code functional.  A detached clone ensures gradients still flow
        # correctly for the main classification loss.
        return x.clone().detach()


class Trainer:
    """Generic trainer that supports optional AutoSpuSwap consistency loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, torch.utils.data.DataLoader],
        cfg: Dict[str, Any],
        autospu: AutoSpuWrapper | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.loaders = loaders
        self.cfg = cfg
        self.device = device
        self.autospu = autospu or AutoSpuWrapper(enabled=False)

        # Optimiser & mixed precision -----------------------------
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        opt_cfg = cfg["optimizer"]

        # YAML may encode numbers as *strings* when quoted.  Convert here
        # to avoid type-errors in the optimiser constructor (see issue #42).
        lr_raw = opt_cfg.get("lr", 3e-4)
        lr: float = float(lr_raw)  # robust to already-float inputs

        self.opt = AdamW(
            trainable_params,
            lr=lr,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
        self.scaler = GradScaler()

        # Losses ---------------------------------------------------
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

        # Logging --------------------------------------------------
        self.train_history = collections.defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> float:
        """Main training loop.  Returns *best validation accuracy*."""
        best_val, best_epoch = 0.0, -1
        for epoch in range(1, self.cfg["epochs"] + 1):
            tr_acc, tr_loss = self._run_epoch(epoch, train=True)
            val_acc, val_loss = self._run_epoch(epoch, train=False)
            if val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
            print(
                f"Epoch {epoch:02d} – train_acc={tr_acc:.2f}",
                f"val_acc={val_acc:.2f} (best={best_val:.2f} @ {best_epoch})",
            )

        # Persist learning curves --------------------------------
        plot_curves(self.train_history["train_loss"], self.train_history["val_loss"])
        return best_val

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_epoch(self, epoch: int, *, train: bool) -> tuple[float, float]:
        mode = self.model.train if train else self.model.eval
        mode()
        loader_key = "train" if train else "val"
        loader = self.loaders[loader_key]

        correct, total = 0, 0
        losses: list[float] = []

        for batch in loader:
            x, y, idx = (
                batch[0].to(self.device, non_blocking=True),
                batch[1].to(self.device, non_blocking=True),
                batch[2],
            )

            with autocast():
                logits = self.model(x)
                loss = self.ce(logits, y)

                # --------------------------------------------------
                # AutoSpuSwap consistency loss (optional)
                # --------------------------------------------------
                if train and getattr(self.autospu, "enabled", False):
                    x_cf = self.autospu.make_counterfactual(x, y, idx)
                    logits_cf = self.model(x_cf)
                    lam = float(self.cfg["autospu"].get("lambda_consistency", 1.0))
                    loss = loss + lam * self.mse(logits, logits_cf)

            # Back-prop only in training mode ----------------------
            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

            # ------------------------------------------------------
            losses.append(loss.item())
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / max(total, 1)
        k = "train_loss" if train else "val_loss"
        self.train_history[k].append(float(np.mean(losses)))
        return acc, float(np.mean(losses))
