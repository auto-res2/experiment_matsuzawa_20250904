#!/usr/bin/env python
"""Entry-point that orchestrates all experiments.

Run with
    python -m src.main
"""
from __future__ import annotations

import yaml
from pathlib import Path

import torch

from .autospu import AutoSpuSwap
from .preprocess import waterbirds_loaders
from .train import Trainer, create_model
from .utils import set_seed

# ---------------------------------------------------------------------
#   Configuration
# ---------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

_DEFAULT_YAML = """
experiment_1:
  name: "EXP1 – Pipeline Integrity"
  backbone: resnet18
  epochs: 2
  batch_size: 64
  optimizer: {lr: 1e-3}
  autospu:
    lambda_consistency: 1.0
    fourier_cfg: {phase_max: 0.2, amp_min: 0.5, amp_max: 1.5}
    swap_prob: 1.0
experiment_2:
  name: "EXP2 – Waterbirds full"
  backbone: resnet50_in21k
  epochs: 30
  batch_size: 128
  optimizer: {lr: 3e-4, weight_decay: 0.05}
  autospu:
    lambda_consistency: 1.0
    fourier_cfg: {phase_max: 0.2, amp_min: 0.5, amp_max: 1.5}
    swap_prob: 1.0
"""

# Write default config if not present ---------------------------------
if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text(_DEFAULT_YAML)
    print(f"[INFO] Default config written to {CONFIG_PATH}")


# ---------------------------------------------------------------------
#   Single-experiment runner
# ---------------------------------------------------------------------

def _run_exp(exp_key: str, cfg):
    print(f"\n========== Running {exp_key}: {cfg['name']} ==========")
    set_seed(0)

    # ------------ data -------------
    loaders = waterbirds_loaders(bs=cfg["batch_size"])

    # ------------ model ------------
    model = create_model(cfg["backbone"], num_classes=2)

    # ------------ AutoSpuSwap ------
    autospu = AutoSpuSwap(cfg, device="cuda" if torch.cuda.is_available() else "cpu")

    # ------------ training loop ----
    trainer = Trainer(model, loaders, cfg, autospu)
    best_wg = trainer.fit()

    print(f"[RESULT] {exp_key} best WG-Acc = {best_wg * 100:.2f}%")


# ---------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in sorted(cfg.keys()):
        _run_exp(key, cfg[key])


if __name__ == "__main__":
    main()
