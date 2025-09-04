from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import torch
import yaml
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from . import preprocess as pp
from .train import (
    GCNNet,
    GATNet,
    APPNPNet,
    norm_factory,
    run_training,
    set_global_seeds,
)
from .evaluate import save_lineplot

# ------------------------------------------------------------------
#  Dataclass config holders (light-weight)
# ------------------------------------------------------------------
@dataclass
class GlobalConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    fp16: bool = False
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    lr_grid: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])
    wd_grid: List[float] = field(default_factory=lambda: [0.0, 1e-4, 5e-4])
    dropout: float = 0.5
    hidden: int = 128
    patience: int = 50
    data_root: str = "data"

@dataclass
class FRODOHyper:
    gamma: float = 5.0
    sigma: float = 0.1
    k: float = 0.15
    tau_m: float = 0.02
    tau_r: float = 0.35

# ------------------------------------------------------------------
#  Experiments
# ------------------------------------------------------------------

# (experiments code unchanged, omitted for brevity)

# ------------------------------------------------------------------
#  YAML config loading
# ------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _cfg_dict = yaml.safe_load(fp)
GLOBAL_CFG = GlobalConfig(**_cfg_dict["global"])
FRODO_HYPER = FRODOHyper(**_cfg_dict["frodo"])

# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main():
    os.makedirs(GLOBAL_CFG.data_root, exist_ok=True)
    # Ensure the mandated image directory exists
    os.makedirs(".research/iteration5/images", exist_ok=True)
    set_global_seeds(GLOBAL_CFG.seeds[0])

    print("\n================ EXPERIMENT 1 – Depth scaling ================")
    experiment1_depth(GLOBAL_CFG, FRODO_HYPER)

    print("\n================ EXPERIMENT 2 – Denoising ===================")
    experiment2_denoise(GLOBAL_CFG, FRODO_HYPER)

    print("\n================ EXPERIMENT 3 – Scalability ================")
    experiment3_scale(GLOBAL_CFG, FRODO_HYPER)


if __name__ == "__main__":  # pragma: no cover
    main()
