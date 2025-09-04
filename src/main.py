"""
main.py – single entry-point.  Usage:  python -m src.main
"""
from __future__ import annotations

import yaml
from pathlib import Path

from .train import GlobalCfg, FrodoHyper
from .evaluate import experiment_depth

# -----------------------------------------------------------------------------
#  Load configuration YAML
# -----------------------------------------------------------------------------

def _load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    g_raw = raw.get("global", {})
    f_raw = raw.get("frodo", {})
    g_cfg = GlobalCfg(**g_raw)
    f_cfg = FrodoHyper(**f_raw)
    return g_cfg, f_cfg


def main():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    global_cfg, frodo_cfg = _load_cfg(cfg_path)

    # run experiments (expand here if multiple experiments)
    experiment_depth(global_cfg, frodo_cfg)


if __name__ == "__main__":
    main()
