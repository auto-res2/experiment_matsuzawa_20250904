"""
main.py – project entry point
Use as:  python -m src.main  [--exp wb|diag|all]
"""
from __future__ import annotations
import argparse
import time
from rich import print

from src.train import run_waterbirds
from src.evaluate import diagnostics


def _parse_args() -> argparse.Namespace:  # pragma: no cover – cli helper
    ap = argparse.ArgumentParser(description="AutoSpuSwap experimental driver")
    ap.add_argument(
        "--exp",
        choices=["wb", "diag", "all"],
        default="all",
        help="Which experiment to run",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    tic = time.time()
    if args.exp in ("wb", "all"):
        run_waterbirds()
    if args.exp in ("diag", "all"):
        diagnostics()
    print(f"[bold green]✓ Done in {time.time() - tic:.1f}s")


if __name__ == "__main__":  # pragma: no cover – standard pattern
    main()
