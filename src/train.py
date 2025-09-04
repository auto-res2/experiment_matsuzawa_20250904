"""src/train.py
Training utilities – wraps the whole optimisation loop that is reused by
main.py and potential future experiments.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .evaluate import accuracy, save_bar


# Directory where visual artefacts must be stored according to the task
_IMG_OUT_DIR = Path(".research/iteration8/images")


def run_one(
    model: torch.nn.Module,
    loader_tr: DataLoader,
    loader_val: DataLoader,
    cfg: Dict[str, float | int],
    device: torch.device,
    tag: str,
) -> float:
    """Train *model* on *loader_tr* and evaluate on *loader_val*.

    Parameters
    ----------
    model : torch.nn.Module
        The network that should be trained.
    loader_tr / loader_val : DataLoader
        Dataloaders for training and validation.
    cfg : Dict
        Contains the hyper-parameters ``lr`` and ``epochs``.
    device : torch.device
        CPU or GPU.
    tag : str
        A short string to identify artefacts on disk.
    """

    model.to(device)
    opt = torch.optim.SGD(
        model.parameters(), lr=float(cfg["lr"]), momentum=0.9, weight_decay=1e-4
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best = 0.0
    for ep in range(int(cfg["epochs"])):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                # If the model implements a specialised loss function, use it –
                # otherwise, fall back to plain cross-entropy.
                if hasattr(model, "loss_fn"):
                    loss, _parts = model.loss_fn(logits, y)  # type: ignore[attr-defined]
                else:
                    loss = torch.nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)

        val_acc = accuracy(model, loader_val, device)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), f"best_{tag}.pt")

        print(
            f"[epoch {ep:02d}] loss {loss_sum / max(n, 1):.3f}  "
            f"val {val_acc:.3f}  best {best:.3f}  time {time.time() - t0:.1f}s"
        )

    # ---- load best checkpoint & final metric --------------------------------
    model.load_state_dict(torch.load(f"best_{tag}.pt", map_location=device))
    best_acc = accuracy(model, loader_val, device)

    # ---- persist artefacts ---------------------------------------------------
    _IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_bar(best_acc, tag, _IMG_OUT_DIR)
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{tag}.json", "w") as fh:
        json.dump({"val_acc": best_acc}, fh)

    return best_acc
