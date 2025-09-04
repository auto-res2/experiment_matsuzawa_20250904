"""
main.py – orchestrates the full experimental grid
-------------------------------------------------
Reads *config/config.yaml* to build the list of experiments, then calls
run_single_experiment from train.py.  Aggregation, pretty printing and Pareto
plot are handled here as they are strictly evaluation-level tasks.
"""
from __future__ import annotations

import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path
from typing import List, Dict

from .train import (
    run_single_experiment,
    ExperimentConfig,
    ContinualConfig,
    OptimConfig,
    SEEDS,
)
from .preprocess import FIG_ROOT, CKPT_ROOT

# ---------------------------------------------------------------------------
#  Parse YAML configuration
# ---------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – please supply one.")
with open(CFG_PATH) as fh:
    CFG = yaml.safe_load(fh)

# ---------------------------------------------------------------------------
#  Build experiment list according to YAML
# ---------------------------------------------------------------------------
exp_cfgs: List[ExperimentConfig] = []
for name, cap in CFG["memory_budgets"].items():
    exp_cfgs.append(
        ExperimentConfig(
            name=f"LQR_{name}",
            dataset=CFG["dataset"],
            model="resnet18_vq",
            method="LQR",
            ctl=ContinualConfig(
                n_tasks=CFG["n_tasks"],
                classes_per_task=CFG["classes_per_task"],
                buffer_bytes=cap,
                epochs=CFG["epochs"],
                batch_size=CFG["batch_size"],
            ),
            optim=OptimConfig(
                lr=CFG["learning_rate"],
                momentum=CFG["momentum"],
                weight_decay=CFG["weight_decay"],
            ),
        )
    )
    exp_cfgs.append(
        ExperimentConfig(
            name=f"ER_raw_{name}",
            dataset=CFG["dataset"],
            model="resnet18_vq",
            method="ER",
            ctl=ContinualConfig(
                n_tasks=CFG["n_tasks"],
                classes_per_task=CFG["classes_per_task"],
                buffer_bytes=cap,
                epochs=CFG["epochs"],
                batch_size=CFG["batch_size"],
            ),
            optim=OptimConfig(
                lr=CFG["learning_rate"],
                momentum=CFG["momentum"],
                weight_decay=CFG["weight_decay"],
            ),
        )
    )
    exp_cfgs.append(
        ExperimentConfig(
            name=f"JPEG_ER_{name}",
            dataset=CFG["dataset"],
            model="resnet18_vq",
            method="JPEG-ER",
            buffer_quality=CFG["jpeg_quality"],
            ctl=ContinualConfig(
                n_tasks=CFG["n_tasks"],
                classes_per_task=CFG["classes_per_task"],
                buffer_bytes=cap,
                epochs=CFG["epochs"],
                batch_size=CFG["batch_size"],
            ),
            optim=OptimConfig(
                lr=CFG["learning_rate"],
                momentum=CFG["momentum"],
                weight_decay=CFG["weight_decay"],
            ),
        )
    )

# ---------------------------------------------------------------------------
#  Launch grid over seeds
# ---------------------------------------------------------------------------
all_results: Dict[str, List[Dict]] = {}
for cfg in exp_cfgs:
    res_per_seed: List[Dict] = []
    for seed in SEEDS:
        cfg.seed = seed
        # save config snapshot – aids reproducibility
        cfg_yaml = CKPT_ROOT / f"{cfg.name}_seed{seed}.yml"
        with open(cfg_yaml, "w") as fh:
            yaml.safe_dump(asdict(cfg), fh)
        res_per_seed.append(run_single_experiment(cfg))
    all_results[cfg.name] = res_per_seed

# ---------------------------------------------------------------------------
#  Aggregate & pretty print
# ---------------------------------------------------------------------------
print("\n================ Aggregate Results (mean ± std over seeds) ================")
print("Method                |  Acc   |  Forget  |  Mem(kB)")
print("--------------------------------------------------------------------------")
for name, runs in all_results.items():
    accs = [r["acc"] for r in runs]
    fgts = [r["fgt"] for r in runs]
    mem_kb = runs[0]["bytes"] / 1024.0
    print(f"{name:22s} {np.mean(accs):6.2f}±{np.std(accs):4.2f}   {np.mean(fgts):5.2f}±{np.std(fgts):4.2f}   {mem_kb:7.1f}")

# ---------------------------------------------------------------------------
#  Pareto plot Acc vs Memory
# ---------------------------------------------------------------------------
pareto_path = FIG_ROOT / "accuracy_vs_memory.pdf"
plt.figure(figsize=(6, 4))
for name, runs in all_results.items():
    mean_acc = np.mean([r["acc"] for r in runs])
    mem_kb = runs[0]["bytes"] / 1024.0
    plt.scatter(mem_kb, mean_acc, label=name, s=50)
    plt.text(mem_kb, mean_acc + 0.4, name, fontsize=6, rotation=45)
plt.xscale("log")
plt.xlabel("Replay memory (kB)")
plt.ylabel("Average Accuracy (%)")
plt.grid(True, which="both", ls="--")
plt.title("Memory-Accuracy Pareto (Split CIFAR-100)")
plt.savefig(pareto_path, bbox_inches="tight")
plt.close()
print(f"Saved figure  → {pareto_path.name}")
