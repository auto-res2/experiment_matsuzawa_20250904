"""src/main.py
Entry-point executed via ``python -m src.main``.
It orchestrates experiments defined in config/config.yaml (defaults are
written automatically on first launch).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.optim import AdamW

from .preprocess import dataset_map, random_split
from .train import DeepGCN, train_epoch
from .evaluate import evaluate, plot_curve

# -----------------------------------------------------------------------------
#                         Paths & directories
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#                 Write default YAML the first time we run
# -----------------------------------------------------------------------------
DEFAULT_YAML = CONFIG_DIR / "exp1.yaml"

if not DEFAULT_YAML.exists():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    exp1 = {
        "name": "exp1_unified_cure",
        "datasets": ["path_of_cliques", "ring_of_cliques", "mixture_graph"],
        "model_depth": 32,
        "hidden_dim": 128,
        "models": [
            {"id": "gcn"},
            {"id": "gcn_pairnorm"},
        ],
        "optimizer": {"lr": 1e-3, "weight_decay": 5e-4},
        "patience": 100,
        "max_epochs": 2000,
        "seeds": list(range(3)),  # shorten default runtime
    }
    yaml.safe_dump(exp1, DEFAULT_YAML.open("w"))

# -----------------------------------------------------------------------------
#                               Experiment-1
# -----------------------------------------------------------------------------

def run_experiment1() -> None:
    cfg = yaml.safe_load(DEFAULT_YAML.read_text())
    print("\n============= EXPERIMENT-1 :", cfg["name"], "=============")
    print(json.dumps(cfg, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dset_name in cfg["datasets"]:
        data = dataset_map[dset_name]()
        data = random_split(data, seed=0).to(device)
        num_features, num_classes = data.x.size(1), int(data.y.max().item() + 1)
        print(f"\n--- Dataset : {dset_name} (N={data.num_nodes}, E={data.edge_index.size(1)}) ---")

        for model_cfg in cfg["models"]:
            model_id = model_cfg["id"] if isinstance(model_cfg, dict) else model_cfg
            accs: List[float] = []
            gdr_store: List[List[float]] = []
            efr_store: List[List[float]] = []

            for seed in cfg["seeds"]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                use_pn = model_id == "gcn_pairnorm"
                model = DeepGCN(num_features, num_classes, cfg["hidden_dim"], cfg["model_depth"], use_pairnorm=use_pn).to(device)

                optim = AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
                scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

                best_val, wait = 0.0, 0
                for _ in range(cfg["max_epochs"]):
                    _ = train_epoch(model, data, optim, scaler)
                    acc, *_ = evaluate(model, data)
                    if acc > best_val:
                        best_val, wait = acc, 0
                    else:
                        wait += 1
                    if wait >= cfg["patience"]:
                        break

                acc, gdr, efr = evaluate(model, data)
                accs.append(acc)
                gdr_store.append(gdr)
                efr_store.append(efr)

            # Aggregate over seeds
            acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
            gdr_mean = np.mean(gdr_store, axis=0)
            efr_mean = np.mean(efr_store, axis=0)

            # Persist figures
            plot_curve(gdr_mean.tolist(), f"GDR – {model_id}", "GDR", PLOT_DIR / f"gdr_{dset_name}_{model_id}.pdf")
            plot_curve(efr_mean.tolist(), f"Erank – {model_id}", "Erank", PLOT_DIR / f"er_{dset_name}_{model_id}.pdf")

            print(f"{model_id:15s}  acc = {acc_mean:.3f} ± {acc_std:.3f}")

# -----------------------------------------------------------------------------
#                             Main dispatcher
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    run_experiment1()


if __name__ == "__main__":  # pragma: no cover
    main()
