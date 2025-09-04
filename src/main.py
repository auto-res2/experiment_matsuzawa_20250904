"""src/main.py
Main entry-point orchestrating experiments.  Usage:
    python -m src.main --exp wb        (Waterbirds)
    python -m src.main --exp diag      (Diagnostics)
    python -m src.main --exp all       (default – everything)
"""
from __future__ import annotations

import argparse
import time

import yaml
from rich import print

from .evaluate import run_diagnostics
from .train import run_waterbirds

# ------------------------------------------------------------------
#  Configuration (read-only here – heavy lifting happens in train.py)
# ------------------------------------------------------------------
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
CFG = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8"))


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["wb", "diag", "all"], default="all")
    args = parser.parse_args()

    tic = time.time()
    if args.exp in ("wb", "all"):
        run_waterbirds()
    if args.exp in ("diag", "all"):
        run_diagnostics()
    print(f"\n[bold green]All requested experiments done in {time.time() - tic:.1f}s.[/]")


if __name__ == "__main__":  # pragma: no cover
    main()
