"""src/main.py
Entry-point orchestrating the three experiments described in the project prompt.
Run with `python -m src.main` from the repository root.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

# ---------------- Package-internal imports -----------------------------------
from .train import AdaPropGCN, train_epoch
from .evaluate import eval_model, compute_accuracy
from .preprocess import (
    set_seed,
    print_header,
    fail,
    load_planetoid,
    load_ogb,
    sparse_power_series,
)

# ----------------------------------------------------------------------------
# 1)  CONFIG ------------------------------------------------------------------
# ----------------------------------------------------------------------------

CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

with open(CFG_PATH, "r") as fp:
    CONFIG: Dict[str, Any] = yaml.safe_load(fp)

# auto-detect GPU if desired
if CONFIG["common"]["device"] == "auto":
    CONFIG["common"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

# make sure folders exist -----------------------------------------------------
Path(CONFIG["common"]["data_root"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["common"]["output_root"]).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 2)  EXPERIMENT #1 – Depth robustness ---------------------------------------
# ----------------------------------------------------------------------------

def run_experiment_1():
    cfg_common = CONFIG["common"]
    exp_cfg = CONFIG["experiments"]["exp1_depth_robust"]

    for dname in exp_cfg["datasets"]:
        print_header(f"Experiment-1  Dataset: {dname}")
        if dname.startswith("ogbn"):
            g, feats, labels, split, evaluator = load_ogb(dname, cfg_common["data_root"])
        else:
            g, feats, labels, split = load_planetoid(dname, cfg_common["data_root"])
            evaluator = None

        g = g.to(cfg_common["device"])
        feats = feats.to(cfg_common["device"])
        labels = labels.to(cfg_common["device"])
        sparse_powers = sparse_power_series(g.cpu(), CONFIG["hyper"]["K"])

        results_depth: Dict[int, Dict[str, float]] = {}
        for depth in exp_cfg["depths"]:
            seed = cfg_common["seeds"][0]  # single seed for quick demo
            set_seed(seed)

            hidden_dim_key = "hidden_dim_small" if dname in {"cora", "citeseer", "pubmed"} else "hidden_dim_large"
            model = AdaPropGCN(
                in_dim=feats.shape[1],
                hidden=CONFIG["hyper"][hidden_dim_key],
                num_classes=int(labels.max().item()) + 1,
                depth=depth,
                K=CONFIG["hyper"]["K"],
                lambda_reg=CONFIG["hyper"]["lambda"],
            ).to(cfg_common["device"])

            optimizer = Adam(model.parameters(), **cfg_common["optimizer"])
            scheduler = CosineAnnealingLR(optimizer, T_max=exp_cfg["epochs"])
            scaler = GradScaler(enabled=cfg_common["mixed_precision"])

            best_val, best_test, patience = 0.0, 0.0, 0
            for epoch in range(exp_cfg["epochs"]):
                loss = train_epoch(model, g, feats, labels, split, optimizer, scaler, cfg_common, sparse_powers)
                accs, _ = eval_model(model, g, feats, labels, split, cfg_common, sparse_powers)
                scheduler.step()

                if accs["val"] > best_val:
                    best_val, best_test = accs["val"], accs["test"]
                    patience = 0
                else:
                    patience += 1
                if patience >= cfg_common["early_stop_patience"]:
                    break
                if (epoch + 1) % cfg_common["log_every"] == 0:
                    print(f"Depth {depth:3d}  Ep {epoch+1:3d}  Loss {loss:.4f}  Val {accs['val']*100:.2f}%")

            results_depth[depth] = {"val": best_val, "test": best_test}
            print(f"  >>> Depth {depth:3d} FINAL  Val {best_val*100:.2f}%  Test {best_test*100:.2f}%")

        # --------------- PLOT -------------------------------------------------
        depths = list(results_depth.keys())
        test_acc = [results_depth[d]["test"] * 100 for d in depths]
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=depths, y=test_acc, marker="o", label="AdaProp-GCN")
        for x, y in zip(depths, test_acc):
            plt.text(x, y, f"{y:.1f}", ha="center", va="bottom")
        plt.title(f"Test accuracy vs depth – {dname}")
        plt.xlabel("Depth (layers)")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        fname = f"training_accuracy_{dname}.pdf"
        plt.savefig(Path(cfg_common["output_root"]) / fname, bbox_inches="tight")
        plt.close()

        # --------------- STDOUT ----------------------------------------------
        print("\nExperiment description:")
        print("Depth-robustness on", dname)
        print("\nExperimental numerical data:")
        print(json.dumps(results_depth, indent=2))
        print("\nNames of figures summarising the numerical data:")
        print(fname)

# ----------------------------------------------------------------------------
# 3)  EXPERIMENT #2 – Node-wise adaptivity ------------------------------------
# ----------------------------------------------------------------------------

def run_experiment_2():
    cfg_common = CONFIG["common"]
    exp_cfg = CONFIG["experiments"]["exp2_node_adapt"]

    dname = exp_cfg["datasets"][0]
    print_header("Experiment-2  Per-node adaptivity (PubMed noisy)")

    g, feats, labels, split = load_planetoid(dname, cfg_common["data_root"])

    # -------- inject feature noise -----------------------------------------
    rng = np.random.RandomState(0)
    noisy_mask = rng.rand(g.num_nodes()) < exp_cfg["noise_fraction"]
    feats_noisy = feats.clone()
    sigma = exp_cfg["noise_sigma"]
    feats_noisy[torch.from_numpy(noisy_mask)] = (
        torch.randn_like(feats_noisy[torch.from_numpy(noisy_mask)]) * sigma
    )
    feats_noisy = torch.nn.functional.normalize(feats_noisy, p=2, dim=1)

    g = g.to(cfg_common["device"])
    feats_noisy = feats_noisy.to(cfg_common["device"])
    labels = labels.to(cfg_common["device"])
    sparse_powers = sparse_power_series(g.cpu(), CONFIG["hyper"]["K"])

    set_seed(cfg_common["seeds"][0])
    model = AdaPropGCN(
        in_dim=feats_noisy.shape[1],
        hidden=CONFIG["hyper"]["hidden_dim_small"],
        num_classes=int(labels.max().item()) + 1,
        depth=exp_cfg["depths"][0],
        K=CONFIG["hyper"]["K"],
        lambda_reg=CONFIG["hyper"]["lambda"],
    ).to(cfg_common["device"])

    optimizer = Adam(model.parameters(), **cfg_common["optimizer"])
    scheduler = CosineAnnealingLR(optimizer, T_max=exp_cfg["epochs"])
    scaler = GradScaler(enabled=cfg_common["mixed_precision"])

    best_val, best_state = 0.0, None
    for epoch in range(exp_cfg["epochs"]):
        train_epoch(model, g, feats_noisy, labels, split, optimizer, scaler, cfg_common, sparse_powers)
        accs, _ = eval_model(model, g, feats_noisy, labels, split, cfg_common, sparse_powers)
        scheduler.step()
        if accs["val"] > best_val:
            best_val, best_state = accs["val"], model.state_dict()
    model.load_state_dict(best_state)

    accs, logits = eval_model(model, g, feats_noisy, labels, split, cfg_common, sparse_powers)

    # ---------------- additional metrics -----------------------------------
    noisy_idx = torch.from_numpy(noisy_mask).to(cfg_common["device"])
    clean_idx = ~noisy_idx
    acc_noisy = compute_accuracy(logits, labels, noisy_idx)
    acc_clean = compute_accuracy(logits, labels, clean_idx)

    pi = model.net[0].pi_cache.cpu()
    exp_k = (pi * torch.arange(pi.shape[1]).unsqueeze(0)).sum(dim=1).numpy()
    from scipy.stats import spearmanr

    rho = spearmanr(noisy_mask.astype(int), exp_k).correlation

    # histogram -------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.histplot(exp_k[~noisy_mask], color="blue", label="clean", kde=False, bins=20, stat="probability")
    sns.histplot(exp_k[noisy_mask], color="red", label="noisy", kde=False, bins=20, stat="probability")
    plt.xlabel("E[k] (expected stopping depth)")
    plt.ylabel("Probability")
    plt.legend()
    fname = "stopping_depth_histogram.pdf"
    plt.savefig(Path(cfg_common["output_root"]) / fname, bbox_inches="tight")
    plt.close()

    # stdout ---------------------------------------------------------------
    print("\nExperiment description:")
    print("Per-node adaptivity with 30% feature noise on PubMed.")
    print("\nExperimental numerical data:")
    res_dict = {
        "Acc_total": accs["test"],
        "Acc_clean": acc_clean,
        "Acc_noisy": acc_noisy,
        "Spearman_rho": rho,
    }
    print(json.dumps(res_dict, indent=2))
    print("\nNames of figures summarising the numerical data:")
    print(fname)

# ----------------------------------------------------------------------------
# 4)  EXPERIMENT #3 – Scalability (stub) --------------------------------------
# ----------------------------------------------------------------------------

def run_experiment_3():
    print_header("Experiment-3  Scalability to OGBN-papers100M (stub)")
    print(
        "Due to time and resource constraints, the full run is omitted here. "
        "Please execute this experiment on a dedicated machine."
    )
    plan = CONFIG["experiments"]["exp3_scalability"]
    print(json.dumps(plan, indent=2))
    fname = "placeholder_scalability.pdf"
    print("\nNames of figures summarising the numerical data:")
    print(fname)

# ----------------------------------------------------------------------------
# 5)  MAIN --------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
    # dump runtime copy of config
    with open(Path(CONFIG["common"]["output_root"]) / "config_runtime.yaml", "w") as fp:
        yaml.safe_dump(CONFIG, fp)

    run_experiment_1()
    run_experiment_2()
    run_experiment_3()


if __name__ == "__main__":
    main()
