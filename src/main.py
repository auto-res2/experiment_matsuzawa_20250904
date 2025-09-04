from __future__ import annotations

"""src/main.py – orchestrates all experiments (import bootstrap first!)"""

# ---------------------------------------------------------------------------
#  Bootstrap: patch missing torchdata & pydantic for GraphBolt (must be *first*)
# ---------------------------------------------------------------------------
import sys
import types

# ---------------------------------------------------------------------------
#  Helper: very permissive dummy object that absorbs all attribute access /
#          call without failing.  This avoids import-time errors in DGL's
#          GraphBolt component that expects *real* ``torchdata`` / ``pydantic``
#          installs, while our codebase never relies on them at runtime.
# ---------------------------------------------------------------------------
class _NoOp:  # pylint: disable=too-few-public-methods
    """A do-nothing callable / attr container used for stubbing."""

    def __call__(self, *_, **__):  # noqa: D401 – returns itself so that any chaining works
        return self

    def __getattr__(self, _):  # noqa: D401 – always succeed
        return self

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


# ---------------------------------------------------------------------------
#  Stub torchdata (minimal surface for GraphBolt) ---------------------------
# ---------------------------------------------------------------------------
if "torchdata" not in sys.modules:
    td_root = types.ModuleType("torchdata")
    sys.modules["torchdata"] = td_root

    # torchdata.datapipes.* --------------------------------------------------
    dp_mod = types.ModuleType("torchdata.datapipes")
    iter_mod = types.ModuleType("torchdata.datapipes.iter")
    iter_mod.IterDataPipe = _NoOp
    iter_mod.Mapper = _NoOp
    dp_mod.iter = iter_mod

    sys.modules["torchdata.datapipes"] = dp_mod
    sys.modules["torchdata.datapipes.iter"] = iter_mod

    # torchdata.dataloader2.* ------------------------------------------------
    dl2_mod = types.ModuleType("torchdata.dataloader2")
    graph_mod = types.ModuleType("torchdata.dataloader2.graph")
    graph_mod.DataLoader2Graph = _NoOp
    graph_mod.MapDataPipe = _NoOp
    dl2_mod.graph = graph_mod

    sys.modules["torchdata.dataloader2"] = dl2_mod
    sys.modules["torchdata.dataloader2.graph"] = graph_mod

# ---------------------------------------------------------------------------
#  Stub pydantic (GraphBolt expects BaseModel, Field, etc.) ------------------
# ---------------------------------------------------------------------------
if "pydantic" not in sys.modules:
    pydantic_mod = types.ModuleType("pydantic")
    # Expose frequently imported symbols as no-ops
    pydantic_mod.BaseModel = _NoOp
    pydantic_mod.Field = _NoOp
    pydantic_mod.validator = lambda *_, **__: (lambda f: f)  # decorator passthrough
    sys.modules["pydantic"] = pydantic_mod

# ---------------------------------------------------------------------------
#  Standard imports *after* the bootstrap                                    
# ---------------------------------------------------------------------------
from pathlib import Path
import yaml
import torch

from .evaluate import run_depth, run_noise, run_papers

# ---------------------------------------------------------------------------
#  Configuration                                                              
# ---------------------------------------------------------------------------
CFG_DEFAULT_YAML = """
common:
  device: cuda            # auto-fallback handled in code
  output_root: .research/iteration10/images
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

# -------- ensure image path complies with iteration10 requirement ----------
ITER10_PATH = ".research/iteration10/images"
if CFG["common"].get("output_root", "") != ITER10_PATH:
    CFG["common"]["output_root"] = ITER10_PATH

# -------- device fallback ---------------------------------------------------
if CFG["common"]["device"] == "cuda" and not torch.cuda.is_available():
    CFG["common"]["device"] = "cpu"

# -------- persist config ----------------------------------------------------
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
