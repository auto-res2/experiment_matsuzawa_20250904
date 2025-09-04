"""
main.py – project entry point.  Usage:  python -m src.main
"""
from __future__ import annotations
import argparse, pprint, textwrap
from pathlib import Path

import torch, yaml, pandas as pd

from .preprocess import load_dataset, compute_or_curvature
from .train import train_on_graph

# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
DEFAULT_CFG = {
    "description": "Depth scaling stress-test on medium graphs.",
    "datasets": ["Cora", "Citeseer", "Chameleon", "Squirrel", "ogbn-arxiv"],
    "depths": [16, 32, 64, 128],
    "variants": ["vanilla", "dropedge", "dgn", "cap"],
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "max_epochs": 500,
    "patience": 50,
    "log_every": 10,
    "seeds": list(range(10)),
    "act_params": {"beta": 0.01, "tau": 1.0},
    "cache_dir": "cache",
    "output_dir": "figures",
}

if not CFG_PATH.exists():
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump({"exp1": DEFAULT_CFG}, f)

with open(CFG_PATH) as fp:
    USER_CFG = yaml.safe_load(fp)["exp1"]

# -----------------------------------------------------------------------------
#  CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


# -----------------------------------------------------------------------------
#  Main execution
# -----------------------------------------------------------------------------

def main():
    cfg = USER_CFG
    pprint.pprint(cfg)

    cache_dir = Path(cfg["cache_dir"]);
    out_dir = Path(cfg["output_dir"])
    cache_dir.mkdir(exist_ok=True); out_dir.mkdir(exist_ok=True)

    all_results = []
    for ds_name in cfg["datasets"]:
        data = load_dataset(ds_name)
        kappa = compute_or_curvature(data, cache_dir / f"{ds_name}_kappa.pt")
        cfg_local = {**cfg, "edge_kappa": kappa, "output_dir": out_dir}
        all_results = train_on_graph(ds_name, data, cfg_local, all_results)

    # -------------------- summary & persistence --------------------
    df = pd.DataFrame(all_results)
    summary = df.groupby(["dataset", "variant", "depth"]).agg({"best_val_acc": "mean", "gpu_mem": "mean"})
    print("\n=====  Numerical Summary  =====")
    print(summary.round(4))

    summary.to_csv(out_dir / "exp1_summary.csv")


if __name__ == "__main__":
    main()
