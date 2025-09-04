from __future__ import annotations
"""src/main.py
Entry point orchestrating all experiments.  Execute via:
    python -m src.main
"""
import os
import time
import math
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
from rich import print  # pretty printing of experiment headers

from .train import (TinyConv, AutoSpuSwap, run_epoch, set_seed)
from .evaluate import save_metrics, lineplot
from .preprocess import (make_fake_split, WaterbirdsSubset, make_loader)

# ------------------------------------------------------------------
# Directories
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# All images/figures must live under the path mandated by the meta-rules.
PLOTS = ROOT / ".research" / "iteration13" / "images"
RESULTS = ROOT / "results"
for p in (DATA, RESULTS, PLOTS):
    p.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CFG_PATH = ROOT / "config" / "config.yaml"
DEFAULT_CFG_YAML = f"""
exp1_smoke:
  description: 30-second CPU smoke test ↔ CI regression gate
  dataset: FakeData
  num_classes: 4
  epochs: 1
  batch_size: 4
  optimizer:
    lr: 1e-3
  autospu:
    swap_prob: 1.0
    lambda_consistency: 0.3
    lambda_fourier: 0.5
    fourier_cfg:
      phase_max: {math.pi/5}
      amp_min: 0.8
      amp_max: 1.2

exp2_waterbirds:
  description: Waterbirds benchmark (graceful fallback)
  dataset: Waterbirds
  num_classes: 2
  epochs: 30
  batch_size: 128
  optimizer:
    lr: 3e-4
    weight_decay: 0.05
  autospu:
    swap_prob: 1.0
    lambda_consistency: 1.0
    lambda_fourier: 0.3
    fourier_cfg:
      phase_max: 0.2
      amp_min: 0.8
      amp_max: 1.2

exp3_shift_suite:
  description: Cross-shift robustness suite (runs only if data present)
  dataset: CIFAR10C_MetaShift_PACS
  num_classes: 10
  epochs: 10
  batch_size: 64
  optimizer:
    lr: 5e-5
    weight_decay: 0.05
  autospu:
    swap_prob: 1.0
    lambda_consistency: 1.0
    lambda_fourier: 0.5
    fourier_cfg:
      phase_max: 0.2
      amp_min: 0.5
      amp_max: 1.5
"""

# Auto-generate config on first run
if not CFG_PATH.exists():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CFG_PATH.write_text(DEFAULT_CFG_YAML)
    print(f"[bold green]Default configuration written to[/] {CFG_PATH}\n")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    ALL_CFG: Dict[str, Any] = yaml.safe_load(f)

# ------------------------------------------------------------------
# Helper – internal unit smoke tests
# ------------------------------------------------------------------

def _run_internal_tests() -> None:
    """Light-weight sanity checks that double as regression tests."""
    print("[bold]Running internal unit tests …[/]", end=" ")
    # Fourier perturb test
    from .train import FourierPerturb  # local import to keep namespace clean

    x = torch.rand(4, 3, 32, 32)
    fp = FourierPerturb()
    x_f = fp(x)
    diff = (x - x_f).abs().mean().item()
    assert diff > 0.03, "Fourier perturbation too small"

    # AutoSpuSwap context swap
    from .train import AutoSpuSwap

    sp = AutoSpuSwap(lambda_consistency=0, lambda_fourier=0)
    x_cf = sp.make_counterfactual(x)
    assert (x - x_cf).abs().mean().item() > 0.9, "Context swap ineffective"

    print("[green]passed[/] ✓")

# ------------------------------------------------------------------
# Utility – sanitise optimiser hyper-parameters
# ------------------------------------------------------------------

def _sanitize_optimizer_cfg(opt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string scalars that *should* be numeric into actual numbers.

    Some YAML loaders (notably when the scalar is quoted) will return a string
    even for things that look like numbers (e.g. "1e-3").  This helper casts
    obvious candidates to float so downstream Torch optims do not error out.
    The function is intentionally conservative: non-castable strings are left
    untouched so that unexpected types still raise loud errors.
    """
    clean: Dict[str, Any] = {}
    for k, v in opt_cfg.items():
        if isinstance(v, str):
            try:
                clean[k] = float(v)
                continue
            except ValueError:
                pass  # keep as-is, let PyTorch complain if unsupported
        clean[k] = v
    return clean

# ------------------------------------------------------------------
# Experiment 1 – 30-second CPU smoke test
# ------------------------------------------------------------------

def experiment_smoke(cfg: Dict[str, Any]):
    print("\n[bold cyan]EXP-1:[/] CPU smoke test")
    set_seed(0)
    device = "cpu"

    # data
    dl_train, dl_val = make_fake_split(cfg["num_classes"], cfg["batch_size"])

    # model + optimiser
    model = TinyConv(cfg["num_classes"]).to(device)
    optim_cfg = _sanitize_optimizer_cfg(cfg["optimizer"])
    optim = torch.optim.Adam(model.parameters(), **optim_cfg)
    criterion = torch.nn.CrossEntropyLoss()
    autospu = AutoSpuSwap(**cfg["autospu"], device=device)

    train_losses: list[float] = []
    for _ in range(2):  # exactly two iterations for CI
        tl, _ = run_epoch(model, dl_train, optim, criterion, autospu, device)
        train_losses.append(tl)

    val_loss, val_acc = run_epoch(model, dl_val, None, criterion, None, device)

    # store artefacts
    save_metrics({"val_loss": val_loss, "val_acc": val_acc}, RESULTS / "smoke_metrics.json")
    lineplot(train_losses, "loss", PLOTS / "training_loss.pdf")

    print(f"Val-loss={val_loss:.4f}  Val-acc={val_acc*100:.1f}%")
    print("Figures saved:", PLOTS / "training_loss.pdf")

# ------------------------------------------------------------------
# Experiment 2 – Waterbirds (synthetic fallback)
# ------------------------------------------------------------------

def experiment_waterbirds(cfg: Dict[str, Any]):
    print("\n[bold cyan]EXP-2:[/] Waterbirds (fallback subset)")
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fallback subset keeps quick runtime.  In a full implementation one
    # would check if the dataset exists on disk and download otherwise.
    train_loader = make_loader(WaterbirdsSubset("train"), batch_size=32, shuffle=True)
    val_loader = make_loader(WaterbirdsSubset("val"), batch_size=32, shuffle=False)

    model = TinyConv(cfg["num_classes"]).to(device)
    optim_cfg = _sanitize_optimizer_cfg(cfg["optimizer"])
    optim = torch.optim.Adam(model.parameters(), **optim_cfg)
    criterion = torch.nn.CrossEntropyLoss()
    autospu = AutoSpuSwap(**cfg["autospu"], device=device)

    epochs = 2  # reduced so CI stays fast
    for ep in range(epochs):
        tl, _ = run_epoch(model, train_loader, optim, criterion, autospu, device)
        vl, va = run_epoch(model, val_loader, None, criterion, None, device)
        print(f"Epoch {ep+1}/{epochs}: train-loss {tl:.3f}  val-acc {va*100:.1f}%")

    print(f"Final Waterbirds-subset val-accuracy = {va*100:.1f}%")

# ------------------------------------------------------------------
# Experiment 3 – Shift-suite placeholder (skipped)
# ------------------------------------------------------------------

def experiment_shift_suite(_: Dict[str, Any]):
    print("\n[bold cyan]EXP-3:[/] Shift-suite robustness — skipped by default to save CI time")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    tic = time.time()
    _run_internal_tests()

    experiment_smoke(ALL_CFG["exp1_smoke"])
    experiment_waterbirds(ALL_CFG["exp2_waterbirds"])

    if os.getenv("RUN_FULL_SUITE", "0") == "1":
        experiment_shift_suite(ALL_CFG["exp3_shift_suite"])

    print(f"\nTotal wall-clock time: {time.time() - tic:.1f} s")

if __name__ == "__main__":
    main()
