"""src/main.py
Main orchestration script – can be called with `python -m src.main`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .preprocess import build_toy_cifar
from .train import run_one
from .utils.seed import seed_all
from .models import CaDERTiny

# -----------------------------------------------------------------------------
# Config handling -------------------------------------------------------------
# -----------------------------------------------------------------------------

CFG_PATH = Path("config/config.yaml")


def _write_default_cfg() -> None:
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    default: Dict[str, object] = {
        "dataset": {"root": "data/cifar_toy", "bias_ratio": 0.9},
        "experiment": {"seeds": [0, 1, 2, 3, 4]},
        "train": {"batch_size": 128, "epochs": 15, "lr": 0.05},
        "model": {"lam": 1.0, "mu": 0.1, "K": 2},
    }
    with CFG_PATH.open("w") as fh:
        yaml.safe_dump(default, fh)
    print(f"[config] Wrote default config file to {CFG_PATH.relative_to(Path.cwd())}")


if not CFG_PATH.exists():
    _write_default_cfg()

# -----------------------------------------------------------------------------
# Experiment #1 – Green-Corner CIFAR-10 --------------------------------------
# -----------------------------------------------------------------------------


def experiment_toy(cfg: Dict):
    print("\n=== Experiment 1 – Green-Corner CIFAR-10 (toy) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = []

    for seed in cfg["experiment"]["seeds"]:
        # ------------------------------------------------------------------
        seed_all(int(seed))

        # ---------------------- data -------------------------------------
        ds_tr, ds_val, _ds_seen, ds_counter = build_toy_cifar(
            cfg["dataset"]["root"],
            bias_ratio=float(cfg["dataset"]["bias_ratio"]),
            seed=int(seed),
        )
        ld_tr = DataLoader(
            ds_tr,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        ld_val = DataLoader(ds_counter, batch_size=256, shuffle=False, num_workers=2)

        # ---------------------- model ------------------------------------
        model = CaDERTiny(
            K=int(cfg["model"]["K"]),
            lam=float(cfg["model"]["lam"]),
            mu=float(cfg["model"]["mu"]),
        )
        tag = f"toy_cader_seed{seed}"

        best = run_one(
            model,
            ld_tr,
            ld_val,
            {
                "lr": float(cfg["train"]["lr"]),
                "epochs": int(cfg["train"]["epochs"]),
            },
            device,
            tag,
        )
        res.append(best)

    mean, std = float(np.mean(res)), float(np.std(res))
    print("---")
    print("Experiment description: CaDER vs ERM on synthetic Green-Corner CIFAR-10.")
    print("individual accuracies", res)
    print("mean", mean, "std", std)

    Path("results").mkdir(exist_ok=True)
    with open("results/summary_toy.json", "w") as fh:
        json.dump({"acc_seed": res, "mean": mean, "std": std}, fh)


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():  # pragma: no cover – entry-point
    with CFG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)

    experiment_toy(cfg)

    if os.environ.get("CADER_RUN_FULL", "no").lower() == "yes":
        print("[main] Full experiments 2 & 3 would start here (skipped in CI).")


if __name__ == "__main__":
    main()
