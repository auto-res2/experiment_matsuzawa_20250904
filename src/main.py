"""src/main.py – orchestrates all experiments (import bootstrap first!)"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#  Bootstrap: patch missing torchdata for GraphBolt (must be *first*)
# ---------------------------------------------------------------------------
import sys, types
if "torchdata" not in sys.modules:
    td_root = types.ModuleType("torchdata")
    sys.modules["torchdata"] = td_root
    dp_mod = types.ModuleType("torchdata.datapipes")
    iter_mod = types.ModuleType("torchdata.datapipes.iter")

    class _IterDataPipe:  # minimal placeholder
        def __iter__(self):
            return iter(())
        def __len__(self):
            return 0

    iter_mod.IterDataPipe = _IterDataPipe
    dp_mod.iter = iter_mod
    td_root.datapipes = dp_mod
    sys.modules["torchdata.datapipes"] = dp_mod
    sys.modules["torchdata.datapipes.iter"] = iter_mod

# ---------------------------------------------------------------------------
#  Standard imports *after* the bootstrap                                    
# ---------------------------------------------------------------------------
import yaml
from pathlib import Path
import torch

from .evaluate import run_depth, run_noise, run_papers

# ---------------------------------------------------------------------------
#  Configuration                                                             
# ---------------------------------------------------------------------------
CFG_DEFAULT_YAML = """
common:
  device: cuda            # auto-fallback handled in code
  output_root: .research/iteration7/images
  seeds: [11, 22, 33, 44, 55]
train:
  lr: 3e-3
  wd: 5e-4
  patience: 50
model:
  hidden_small: 64
  K: 10
  lambda: 0.2
exp1:
  datasets: [cora, citeseer, pubmed]
  depth_grid: [2, 4, 8, 16, 32, 64]
  epochs: 300
exp2:
  noise_frac: 0.3
  sigma: 1.0
  depth: 32
  epochs: 300
exp3:
  description: OGBN-papers100M ablation (λ={0,0.01,0.1}).
  epochs: 40
"""

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        CFG = yaml.safe_load(f)
else:
    CFG = yaml.safe_load(CFG_DEFAULT_YAML)

# -------- device fallback ----------------------------------------------------
if CFG["common"]["device"] == "cuda" and not torch.cuda.is_available():
    CFG["common"]["device"] = "cpu"

# -------- persist config -----------------------------------------------------
out_root = Path(CFG["common"]["output_root"])
out_root.mkdir(parents=True, exist_ok=True)
with open(out_root / "metadata.yaml", "w") as f:
    yaml.safe_dump(CFG, f)

# ---------------------------------------------------------------------------
#  Run experiments                                                           
# ---------------------------------------------------------------------------
run_depth(CFG)
run_noise(CFG)
run_papers(CFG)

print("\nAll experiments finished successfully.")
