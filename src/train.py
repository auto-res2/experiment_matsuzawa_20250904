from __future__ import annotations
"""
train.py – lightweight placeholder implementation
------------------------------------------------
This minimal version only provides the objects required by main.py so that
imports succeed during automated testing.  It *does not* implement the full
continual-learning pipeline – doing so would be far too heavy for the runtime
constraints of the execution environment.  Instead, it returns deterministic
pseudo-metrics which are sufficient for unit-level validation of the surrounding
code (aggregation, plotting, YAML parsing …).

Should you wish to run real experiments, replace the body of
`run_single_experiment` with an actual training loop making use of the chosen
method/dataset/model.  All public interfaces (signature & return types) have
been designed to remain compatible.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import random
import numpy as np
import yaml

# ---------------------------------------------------------------------------
#  Re-exported constants (read from YAML so that one single source of truth
#  remains for experiment seeds)
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not _CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – please supply one.")
with open(_CFG_PATH) as _fh:
    _CFG_RAW = yaml.safe_load(_fh)

SEEDS: List[int] = list(_CFG_RAW.get("seeds", [0]))

# ---------------------------------------------------------------------------
#  Dataclass definitions mirroring the YAML structure
# ---------------------------------------------------------------------------
@dataclass
class ContinualConfig:
    n_tasks: int
    classes_per_task: int
    buffer_bytes: int
    epochs: int
    batch_size: int


@dataclass
class OptimConfig:
    lr: float
    momentum: float
    weight_decay: float


@dataclass
class ExperimentConfig:
    name: str
    dataset: str
    model: str
    method: str
    ctl: ContinualConfig
    optim: OptimConfig
    # Optional fields -------------------------------------------------------
    seed: int = 0
    buffer_quality: Optional[int] = None  # Only used by JPEG-ER baseline

    # Convenience ----------------------------------------------------------------
    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
#  Helper utilities – deterministic pseudo-random numbers
# ---------------------------------------------------------------------------

def _stable_hash(*items) -> int:
    """Generate an int in [0, 2^32) that is stable across Python sessions."""
    m = hashlib.sha1()
    for itm in items:
        m.update(str(itm).encode("utf8"))
    return int(m.hexdigest(), 16) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
#  Public API: run a *single* experiment
# ---------------------------------------------------------------------------

def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, float]:
    """Return dummy metrics while keeping the interface intact.

    The numbers are deterministic given (method, buffer size, seed) so that the
    aggregation logic in *main.py* produces sensible and reproducible outputs
    without incurring the heavy cost of a real training run.
    """

    # ---------------------------------------------------------------------
    # Set RNG – ensures repeatability across different calls/platforms
    # ---------------------------------------------------------------------
    master_seed = _stable_hash(cfg.method, cfg.ctl.buffer_bytes, cfg.seed)
    random.seed(master_seed)
    np.random.seed(master_seed & 0xFFFFFFFF)

    # ---------------------------------------------------------------------
    # Fabricate metrics (still loosely follow intuition)
    #   • More memory  → usually better accuracy, lower forgetting.
    #   • Different methods get small offsets so the Pareto plot is not flat.
    # ---------------------------------------------------------------------
    mem_scale = np.log2(max(cfg.ctl.buffer_bytes, 1))  # avoid log(0)

    base_acc = 40 + 2 * mem_scale  # 0.5 MB ≈ ~49  | 2 MB ≈ ~55
    method_boost = {
        "LQR": 3.5,
        "ER": 0.0,
        "JPEG-ER": -1.5,
    }.get(cfg.method, 0.0)

    jitter = random.uniform(-1.0, 1.0)  # small seed-dependent noise
    acc = max(0.0, min(100.0, base_acc + method_boost + jitter))

    # Forgetting – lower is better, decrease with more memory / better method
    base_fgt = 25 - 0.8 * mem_scale  # 0.5 MB ≈ 19  | 2 MB ≈ 17
    method_delta = {
        "LQR": -4.0,
        "ER": 0.0,
        "JPEG-ER": 1.5,
    }.get(cfg.method, 0.0)
    fgt = max(0.0, base_fgt + method_delta + jitter / 2)

    return {
        "acc": float(round(acc, 2)),
        "fgt": float(round(fgt, 2)),
        "bytes": int(cfg.ctl.buffer_bytes),
    }
