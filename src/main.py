"""
src/main.py
-----------
Entry point:  python -m src.main
Orchestrates the complete Experiment-1 suite using the refactored modules.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

# local imports – note the leading dot for relative package paths
from .evaluate import accuracy_memory_curve
from .preprocess import get_benchmark
from .train import (
    CPQRBuffer,
    build_model,
    cuda_usage,
    make_replay_strategy,
    set_seed,
)

# -----------------------------------------------------------------------------
#  CONFIG HANDLING
# -----------------------------------------------------------------------------


def load_cfg() -> dict:
    """Load YAML experiment settings from config/config.yaml (create if absent)."""
    # path:  <project-root>/config/config.yaml
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            "Configuration file not found. Expected at: " + str(cfg_path)
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -----------------------------------------------------------------------------
#  MAIN EXPERIMENT LOOP
# -----------------------------------------------------------------------------

def run_experiment() -> None:
    cfg = load_cfg()

    out_dir = Path(cfg["workspace"]) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exp1_results.csv"
    all_rows: list[dict] = []

    # textual description first (as demanded by paper for log-parsing)
    print(
        """
===========================  EXPERIMENT 1  ===========================
Memory vs Accuracy / Forgetting – CPQR vs baselines
Datasets: Split CIFAR-100, Split Mini-ImageNet, Tiny-ImageNet, Permuted-MNIST
Budgets (MB): 0.25, 0.5, 1, 2     Seeds: 2023, 2024, 2025
Optimiser: SGD  lr=0.1  momentum=0.9  cosine decay per task  batch=128
Each vision task: 5 epochs   MNIST: 1 epoch
=======================================================================
"""
    )
    sys.stdout.flush()

    device = torch.device("cuda" if cfg.get("cuda", True) and torch.cuda.is_available() else "cpu")

    # -----------------------------  outer loops  -----------------------------
    for dataset_key, ds_cfg in cfg["datasets"].items():
        benchmark = get_benchmark(ds_cfg["name"], cfg)

        for budget_mb in cfg["buffer_budgets_mb"]:
            max_bytes = int(budget_mb * 1024 * 1024)

            for seed in cfg["seed"]:
                set_seed(seed)
                model = build_model(dataset_key).to(device)
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg["optim"]["lr"],
                    momentum=cfg["optim"]["momentum"],
                    weight_decay=cfg["optim"]["weight_decay"],
                )
                cpqr = CPQRBuffer(
                    dim=cfg["cpqr"]["dim"],
                    M=cfg["cpqr"]["M"],
                    codebook_size=cfg["cpqr"]["codebook_size"],
                    max_bytes=max_bytes,
                    device=device,
                    dtype=getattr(torch, cfg["cpqr"].get("dtype", "float16")),
                )
                logger = InteractiveLogger()
                eval_plugin = EvaluationPlugin(
                    accuracy_metrics(epoch=True, experience=True, stream=True),
                    forgetting_metrics(stream=True),
                    loggers=[logger],
                )
                cl_strategy = make_replay_strategy(
                    model,
                    optimizer,
                    nn.CrossEntropyLoss(),
                    cfg["optim"]["batch_size"],
                    device,
                    buffer=cpqr,
                    eval_plugin=eval_plugin,
                )

                # ----------------  training & evaluation  ----------------
                with cuda_usage():
                    t0 = time.time()
                    cl_strategy.train(benchmark.train_stream)
                    res = cl_strategy.eval(benchmark.test_stream)
                    wall = time.time() - t0

                aa = res["Top1_Accuracy_Stream/eval"] * 100
                ff = res["StreamForgetting/eval"] * 100
                stats = cpqr.stats()
                row = dict(
                    dataset=dataset_key,
                    method="CPQR",
                    B=budget_mb,
                    seed=seed,
                    AA=aa,
                    FF=ff,
                    mem_mb=stats.total_buffer / 1024 / 1024,
                    bytes_per_sample=stats.bytes_per_sample,
                    wallclock=wall,
                )
                print(json.dumps(row, indent=2))
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    # ---------------------------  figures  ---------------------------
    df = pd.read_csv(csv_path)
    accuracy_memory_curve(df[df.dataset == "split_cifar100"], out_dir)
    print("Figures saved.  Experiment 1 completed successfully.")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment()
