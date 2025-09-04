"""src/evaluate.py – metrics, plotting & high-level experimental routines"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .train import AdaPropGCN, train_model
from .preprocess import load_data, cached_powers

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
#  Helper utils
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    return float((logits[idx].argmax(1) == y[idx]).float().mean().item()) * 100.0


def _save_lineplot(x, y, *, title, xlabel, ylabel, fname):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=x, y=y, marker="o")
    for _x, _y in zip(x, y):
        plt.text(_x, _y, f"{_y:.1f}", ha="center", va="bottom", fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    assert Path(fname).exists(), f"Figure {fname} not created!"

# ---------------------------------------------------------------------------
#  Experiment-1: depth robustness                                             
# ---------------------------------------------------------------------------

def run_depth(cfg: Dict[str, Any]):
    out_dir = Path(cfg["common"]["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["common"]["device"])

    results_all: Dict[str, Any] = {}
    for dataset in cfg["exp1"]["datasets"]:
        g, X, y, split, evaluator = load_data(dataset, out_dir)
        powers = cached_powers(g, cfg["model"]["K"], out_dir / "cache") if evaluator is None else cached_powers(g, cfg["model"]["K"], out_dir / "cache")
        g, X, y = g.to(device), X.to(device), y.to(device)
        powers = [p.to(device) for p in powers]

        res_depth = {}
        for depth in cfg["exp1"]["depth_grid"]:
            acc_seeds: List[float] = []
            for seed in cfg["common"]["seeds"]:
                set_seed(seed)
                model = AdaPropGCN(
                    in_dim=X.size(1),
                    hidden=cfg["model"]["hidden_small"],
                    num_classes=int(y.max().item()) + 1,
                    depth=depth,
                    K=cfg["model"]["K"],
                    lambda_=cfg["model"]["lambda"],
                )
                _, test_acc = train_model(
                    model,
                    g,
                    X,
                    y,
                    split,
                    powers,
                    cfg_train={
                        "lr": cfg["train"]["lr"],
                        "wd": cfg["train"]["wd"],
                        "patience": cfg["train"]["patience"],
                        "max_epochs": cfg["exp1"]["epochs"],
                    },
                    device=device,
                )
                acc_seeds.append(test_acc * 100)
            res_depth[depth] = {"mean": float(np.mean(acc_seeds)), "std": float(np.std(acc_seeds))}

        # figure ------------------------------------------------------
        depths, means = list(res_depth.keys()), [res_depth[d]["mean"] for d in res_depth]
        fname = out_dir / f"accuracy_vs_depth_{dataset}.pdf"
        _save_lineplot(depths, means, title=f"Accuracy vs depth – {dataset}", xlabel="Depth", ylabel="Acc (%)", fname=fname)

        results_all[dataset] = res_depth
        print("\nExperiment description: Depth-robustness on", dataset)
        print("Experimental numerical data:")
        print(json.dumps(res_depth, indent=2))
        print("Names of figures summarising the numerical data:")
        print(fname.name)
    return results_all

# ---------------------------------------------------------------------------
#  Experiment-2: feature noise & adaptivity                                    
# ---------------------------------------------------------------------------

def run_noise(cfg: Dict[str, Any]):
    out_dir = Path(cfg["common"]["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["common"]["device"])

    g, X, y, split, _ = load_data("pubmed", out_dir)
    powers = cached_powers(g, cfg["model"]["K"], out_dir / "cache")
    noisy_mask = np.random.RandomState(0).rand(g.num_nodes()) < cfg["exp2"]["noise_frac"]

    X_noisy = X.clone()
    X_noisy[torch.from_numpy(noisy_mask)] = (
        torch.randn_like(X_noisy[torch.from_numpy(noisy_mask)]) * cfg["exp2"]["sigma"]
    )
    X_noisy = torch.nn.functional.normalize(X_noisy, p=2, dim=1)

    g, X_noisy, y = g.to(device), X_noisy.to(device), y.to(device)
    powers = [p.to(device) for p in powers]

    acc_noisy_runs: List[float] = []
    last_model = None
    for seed in cfg["common"]["seeds"]:
        set_seed(seed)
        model = AdaPropGCN(
            in_dim=X_noisy.size(1),
            hidden=cfg["model"]["hidden_small"],
            num_classes=int(y.max().item()) + 1,
            depth=cfg["exp2"]["depth"],
            K=cfg["model"]["K"],
            lambda_=cfg["model"]["lambda"],
        )
        _, _ = train_model(
            model,
            g,
            X_noisy,
            y,
            split,
            powers,
            cfg_train={
                "lr": cfg["train"]["lr"],
                "wd": cfg["train"]["wd"],
                "patience": cfg["train"]["patience"],
                "max_epochs": cfg["exp2"]["epochs"],
            },
            device=device,
        )
        model.eval()
        with torch.no_grad():
            logits, pis, _ = model(g, X_noisy, powers)
        acc_noisy_runs.append(accuracy(logits, y, torch.from_numpy(noisy_mask).to(device)))
        last_model = (model, pis)  # store for histogram after loop

    acc_noisy_mean = float(np.mean(acc_noisy_runs))

    # ------ expected depth correlation --------------------------------
    model, pis = last_model  # type: ignore
    pi_concat = torch.stack(pis, dim=0).mean(0).cpu()  # (N,K+1)
    exp_k = (pi_concat * torch.arange(cfg["model"]["K"] + 1)).sum(1).numpy()
    rho = spearmanr(noisy_mask.astype(int), exp_k).correlation

    # ------ histogram --------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.histplot(exp_k[~noisy_mask], color="blue", label="clean", stat="probability", bins=20)
    sns.histplot(exp_k[noisy_mask], color="red", label="noisy", stat="probability", bins=20)
    plt.xlabel("E[k]")
    plt.ylabel("Probability")
    plt.legend()
    fname = out_dir / "Edepth_hist.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    res = {"Acc_noisy_mean": acc_noisy_mean, "Spearman_rho": float(rho)}
    print("\nExperiment description: Per-node adaptivity with noise (PubMed)")
    print("Experimental numerical data:")
    print(json.dumps(res, indent=2))
    print("Names of figures summarising the numerical data:")
    print(fname.name)
    return res

# ---------------------------------------------------------------------------
#  Experiment-3: OGBN-papers100M (plan only)                                   
# ---------------------------------------------------------------------------

def run_papers(cfg):
    print("\nExperiment description: Ablation & scalability on papers100M (plan only – heavy)")
    print(json.dumps(cfg["exp3"], indent=2))
    print("Names of figures summarising the numerical data:")
    print("training_accuracy_papers100M.pdf")
    return {}
