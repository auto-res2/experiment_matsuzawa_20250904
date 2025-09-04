"""src/main.py
Main orchestration script.  Run with

    python -m src.main
"""
from __future__ import annotations

import pathlib
import yaml
import torch

from .evaluate import run_exp1

# ---------------------------------------------------------------------------
# 1.  Load configuration from YAML
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    cfg_path = pathlib.Path(__file__).parents[1] / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 2.  Entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    cfg = _load_cfg()

    # For this refactor we only keep EXP-1 for brevity – others can be added
    run_exp1(cfg["experiments"]["exp1"])


if __name__ == "__main__":
    main()
