"""src/main.py
Entry-point of the CaDER project.  Run via ``python -m src.main``.
The script orchestrates the whole experimental workflow:
  1. Load configuration from *config/config.yaml*.
  2. Prepare datasets & loaders.
  3. Train selected methods over the requested random seeds.
  4. Save checkpoints and figures into the project directory.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml

from . import train as train_mod
from . import preprocess as pp
from .models import CaDER, resnet50

# ----------------------------------------------------------------------------
#  Reproducibility helpers
# ----------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # faster
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------
#  Configuration loader
# ----------------------------------------------------------------------------

CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _load_cfg() -> Dict[str, Any]:
    with open(CFG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ----------------------------------------------------------------------------
#  SINGLE EXPERIMENT (shortened to Waterbirds for demonstration)
# ----------------------------------------------------------------------------

def _run_experiment_1(cfg: Dict[str, Any]):
    print("========== Experiment-1:", cfg["experiment1"]["description"], "==========")

    device = torch.device(cfg["global"]["device"] if torch.cuda.is_available() else "cpu")
    bs = cfg["global"]["batch_size"]
    nw = cfg["global"]["num_workers"]
    data_root = Path(cfg["global"]["data_dir"])

    # Waterbirds only (for brevity)
    ds_tr = pp.get_waterbirds(data_root, train=True)
    ds_va = pp.get_waterbirds(data_root, train=False)
    ld_tr = pp.build_loader(ds_tr, bs, nw)
    ld_va = pp.build_loader(ds_va, bs, nw)

    # Backbone checkpoint (download if necessary)
    back_ckpt = data_root / "resnet50.pth"
    if not back_ckpt.exists():
        pp.download_file(cfg["models"]["resnet50"]["url"], back_ckpt, "resnet50")

    for method in cfg["experiment1"]["methods"]:
        for seed in cfg["global"]["seed_list"]:
            _set_seed(seed)

            if method.lower() == "cader":
                model = CaDER(cfg["experiment1"]["cader"], str(back_ckpt))
            else:
                model = resnet50(str(back_ckpt))

            # Merge global + hyper parameters for the trainer
            train_cfg = copy.deepcopy(cfg["global"])
            train_cfg.update(cfg["experiment1"]["hyper"])

            train_mod.train(
                model,
                ld_tr,
                ld_va,
                train_cfg,
                device,
                "exp1",
                method,
                seed,
            )


# ----------------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------------

def main():
    cfg = _load_cfg()

    # Save a copy of the loaded configuration (useful for versioning)
    with open("full_config_loaded.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)

    _run_experiment_1(cfg)
    # (Experiments 2 & 3 are analogous and can be added here.)


if __name__ == "__main__":
    main()
