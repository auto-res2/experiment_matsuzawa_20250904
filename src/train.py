from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any

import timm
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from .evaluate import worst_group_acc, line_plot
from .utils import set_seed

__all__ = [
    "create_model",
    "Trainer",
]


def create_model(backbone: str, num_classes: int) -> torch.nn.Module:
    """Factory for backbone models.

    Parameters
    ----------
    backbone : str
        Backbone name recognised by `timm`.
    num_classes : int
        Final classifier output dimension.
    """
    if backbone == "resnet18":
        model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    elif backbone == "resnet50_in21k":
        model = timm.create_model("resnetv2_50x1_bit.goog_in21k", pretrained=True)
        model.reset_classifier(num_classes)
    elif backbone.startswith("vit"):
        model = timm.create_model(backbone, pretrained=True)
        model.reset_classifier(num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


class Trainer:
    """Generic supervised training loop with optional AutoSpuSwap losses."""

    def __init__(
        self,
        model: torch.nn.Module,
        loaders: Dict[str, torch.utils.data.DataLoader],
        cfg: Dict[str, Any],
        autospu=None,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.loaders = loaders
        self.cfg = cfg
        self.device = device
        self.autospu = autospu

        self.opt = AdamW(
            self.model.parameters(),
            lr=cfg["optimizer"].get("lr", 1e-3),
            weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
        )
        self.scaler = GradScaler()
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.history = {"train": [], "val": []}

    # ------------------------------------------------------------------
    def _iterate(self, train: bool = True):
        """Run a single pass over *train* or *val* split."""
        split = "train" if train else "val"
        loader = self.loaders[split]
        self.model.train(mode=train)

        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            x, y, *_ = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast():
                logits = self.model(x)
                loss = self.ce(logits, y)

                # ---------- AutoSpuSwap consistency loss -------------
                if train and self.autospu is not None:
                    x_cf = self.autospu.make_counterfactual(x, y, None)
                    logits_cf = self.model(x_cf)
                    lam = self.cfg["autospu"]["lambda_consistency"]
                    loss = loss + lam * self.mse(logits, logits_cf)

            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def fit(self):
        best_wg = 0.0
        epochs = self.cfg.get("epochs", 1)
        for ep in range(1, epochs + 1):
            tl, ta = self._iterate(train=True)
            vl, va = self._iterate(train=False)
            self.history["train"].append(tl)
            self.history["val"].append(vl)

            # worst-group accuracy (validation)
            y_all, pred_all, g_all = [], [], []
            self.model.eval()
            with torch.no_grad():
                for x, y, g, _ in self.loaders["val"]:
                    logits = self.model(x.to(self.device))
                    pred = logits.argmax(1).cpu()
                    y_all.append(y)
                    pred_all.append(pred)
                    g_all.append(g)

            y_cat = torch.cat(y_all)
            pred_cat = torch.cat(pred_all)
            g_cat = torch.cat(g_all)
            wg = worst_group_acc(pred_cat, y_cat, g_cat)
            best_wg = max(best_wg, wg)

            print(
                f"Epoch {ep:02d}  trLoss {tl:.3f}  valAcc {va * 100:.2f}%  WG-Acc {wg * 100:.2f}%"
            )

        # save learning-curve PDF in project root
        line_plot(self.history["train"], "Training loss", "loss", "training_loss.pdf")
        return best_wg
