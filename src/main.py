# src/main.py
"""Entry-point – orchestrates the full experimental workflow.

Usage (from project root):
    python -m src.main  --exp experiment_1

The YAML configuration is located at *config/config.yaml*.  All modules
are imported *relatively* so that the package can be executed with
``python -m src.main`` as required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import torch
import timm
import yaml
import torchvision.models as tvm

from .preprocess import set_seed, build_dataloaders
from .train import Trainer, AutoSpuWrapper

# -------------------------------------------------------------------------
# Configuration utilities
# -------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_cfg(exp_name: str) -> Dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Could not locate configuration file at '{CONFIG_PATH}'."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)
    if exp_name not in all_cfg:
        raise KeyError(f"Experiment '{exp_name}' not defined in config.yaml")
    return all_cfg[exp_name]


# -------------------------------------------------------------------------
# Model factory – currently supports ResNet-50 w/ ImageNet-21k initialisation
# -------------------------------------------------------------------------

def build_model(backbone_name: str, num_classes: int, head_hidden: int = 1024):
    """Return a classification model matching *backbone_name*."""
    if backbone_name == "resnet50_in21k":
        model = tvm.resnet50(weights=None)
        # ``timm`` provides the ImageNet-21k pre-trained backbone
        ckpt = timm.create_model("resnet50", pretrained=True)
        model.load_state_dict(ckpt.state_dict(), strict=False)

        model.fc = torch.nn.Identity()
        head = torch.nn.Sequential(
            torch.nn.Linear(2048, head_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(head_hidden, num_classes),
        )
        return torch.nn.Sequential(model, head)

    raise NotImplementedError(f"Build model for backbone '{backbone_name}' is not implemented")


# -------------------------------------------------------------------------
# CLI & execution
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AutoSpuSwap – refactored")
    p.add_argument("--exp", default=os.environ.get("EXP", "experiment_1"), help="Experiment name as defined in config.yaml")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()
    cfg = load_cfg(args.exp)

    print("===== AutoSpuSwap: " + cfg["name"] + " =====")
    print(json.dumps(cfg, indent=2))

    # ------------------------------------------------------------------
    # Reproducibility & data
    # ------------------------------------------------------------------
    set_seed(cfg["seeds"][0])  # Only the first seed is used in this demo
    loaders = build_dataloaders(cfg.get("batch_size", 128))

    # ------------------------------------------------------------------
    # Model & training utilities
    # ------------------------------------------------------------------
    model = build_model(cfg["backbone"], num_classes=2, head_hidden=cfg.get("head_hidden", 1024))

    # AutoSpuSwap wrapper – disabled until full pipeline is ported
    autospu = AutoSpuWrapper(enabled=cfg["autospu"].get("enabled", False))

    trainer = Trainer(model, loaders, cfg, autospu=autospu)
    best_val = trainer.run()

    print(f"Best-val accuracy: {best_val:.2f}%")
    print("Finished – see generated .pdf figures in working directory.")


if __name__ == "__main__":  # pragma: no cover
    main()
