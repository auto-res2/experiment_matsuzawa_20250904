"""
main.py – orchestrates training & evaluation for CurvAMP experiments
Execute with:  python -m src.main  (assuming src is a package or on PYTHONPATH)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import yaml

from .preprocess import load_dataset
from .train import train_model
from .evaluate import evaluate_model

# -----------------------------------------------------------------------------
#  Configuration handling
# -----------------------------------------------------------------------------
_CFG_PATH = Path("config/config.yaml")


def _write_default_cfg() -> None:
    if _CFG_PATH.exists():
        return
    _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": "curvamp_exp1",
        "datasets": ["Path-of-Cliques", "Ring-of-Cliques"],
        "depth": 32,
        "hidden": 128,
        "K": 3,
        "rewire_ratio": 0.02,
        "lambda_c": 0.1,
        "lr": 5e-4,
        "epochs": 400,
        "patience": 100,
        "seeds": [0, 1, 2],
    }
    yaml.safe_dump(cfg, _CFG_PATH.open("w"))


# -----------------------------------------------------------------------------
#  Main driver
# -----------------------------------------------------------------------------

def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    _write_default_cfg()
    cfg: Dict[str, Any] = yaml.safe_load(_CFG_PATH.read_text())
    print("\n===== CurvAMP EXPERIMENT –", cfg["name"], "=====")
    print(yaml.safe_dump(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]

    for dset_name in cfg["datasets"]:
        print(f"\n--- Dataset: {dset_name} ---")
        data = load_dataset(dset_name).to(device)
        data.name = dset_name  # for nice figure names

        accs: List[float] = []
        for seed in cfg["seeds"]:
            _set_seeds(seed)
            model, train_log = train_model(data, cfg, device)
            model.eval()
            with torch.no_grad():
                logits, feats = model(data)
            result = evaluate_model(model, data, feats)
            accs.append(result["acc"])

        print(
            f"Test accuracy mean±std: {np.mean(accs):.3f} ± {np.std(accs):.3f}\nFigures saved in .research/iteration6/images."
        )


if __name__ == "__main__":
    main()
