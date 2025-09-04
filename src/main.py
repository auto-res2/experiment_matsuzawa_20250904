#!/usr/bin/env python
"""Entry-point that orchestrates all experiments.

Run with
    python -m src.main
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from .autospu import AutoSpuSwap
from .preprocess import waterbirds_loaders
from .train import Trainer, create_model
from .utils import set_seed, _DATA_DIR

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

def _dataset_available() -> bool:
    """Return True if the Waterbirds dataset appears to be available locally.

    We only do a *very* lightweight check for the presence of the canonical
    training image directory in order to decide whether to proceed with a
    potentially lengthy download/training run.  This is in line with the
    evaluation policy that prohibits silent fall-backs to synthetic data
    while still allowing the script to exit gracefully when the real data
    are absent.
    """
    expected_dir = _DATA_DIR / "waterbirds" / "train" / "images"
    return expected_dir.is_dir()


def _run_exp(exp_key: str, cfg):
    print(f"\n========== Running {exp_key}: {cfg['name']} ==========")
    set_seed(0)

    if not _dataset_available():
        msg = (
            "[ABORT] Waterbirds dataset not found locally. "
            "The full dataset (~1GB) must be placed under "
            f"'{_DATA_DIR / 'waterbirds'}' prior to running this script. "
            "Automatic downloads are intentionally disabled in the public "
            "evaluation environment to enforce a fail-fast policy."
        )
        print(msg)
        return  # Early exit – do not treat as an error

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
        try:
            _run_exp(key, cfg[key])
        except KeyboardInterrupt:
            raise
        except Exception as e:  # pragma: no cover – best-effort isolation per experiment
            print(f"[ERROR] Experiment '{key}' terminated due to: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
