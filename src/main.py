"""src/main.py
Entry point that orchestrates all three experiments.
Run via  `python -m src.main`.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import torch
import yaml
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from . import preprocess as pp
from .train import (
    GCNNet, GATNet, APPNPNet, norm_factory,
    run_training, set_global_seeds,
)
from .evaluate import save_lineplot

# ------------------------------------------------------------------
#  Dataclass config holders (light-weight)
# ------------------------------------------------------------------
@dataclass
class GlobalConfig:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    fp16: bool = False
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    lr_grid: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])
    wd_grid: List[float] = field(default_factory=lambda: [0.0, 1e-4, 5e-4])
    dropout: float = 0.5
    hidden: int = 128
    patience: int = 50
    data_root: str = "data"

@dataclass
class FRODOHyper:
    gamma: float = 5.0
    sigma: float = 0.1
    k: float = 0.15
    tau_m: float = 0.02
    tau_r: float = 0.35

# ------------------------------------------------------------------
#  Experiments
# ------------------------------------------------------------------

def experiment1_depth(cfg: GlobalConfig, frodo_h: FRODOHyper):
    depths = [2, 16, 64, 128]
    datasets = ["Pubmed", "chameleon", "ogbn-arxiv"]
    variants = ["vanilla", "dropedge", "pairnorm", "contranorm", "ndls", "frodo"]
    summary: Dict[str, List[float]] = {v: [] for v in variants}

    for depth in depths:
        for dname in datasets:
            ds = pp.build_dataset(dname, cfg)
            data = ds[0]
            if dname == "ogbn-arxiv":
                split_idx = ds.get_idx_split()
            else:
                split_idx = {
                    "train": ds[0].train_mask.nonzero(as_tuple=True)[0],
                    "valid": ds[0].val_mask.nonzero(as_tuple=True)[0],
                    "test": ds[0].test_mask.nonzero(as_tuple=True)[0],
                }
            in_dim, out_dim = data.num_features, ds.num_classes
            for variant in variants:
                exp_name = f"Exp1-{dname}-d{depth}-{variant}"
                print(f"\n===== {exp_name} =====")
                nf = norm_factory(variant, cfg.hidden, frodo_h)
                model = GCNNet(in_dim, cfg.hidden, out_dim, depth, cfg.dropout, nf)
                _, metrics = run_training(model, data, split_idx, cfg, exp_name)
                if dname == "Pubmed":
                    summary[variant].append(metrics["test_acc"])
    # plot on Pubmed
    save_lineplot(depths, summary, "Depth (layers)", "Accuracy",
                  "Accuracy vs Depth – Pubmed", "accuracy_pubmed.pdf")

def experiment2_denoise(cfg: GlobalConfig, frodo_h: FRODOHyper):
    datasets = ["Cora", "Citeseer"]
    corruption = [0.0, 0.5, 1.0]
    variants = ["vanilla", "ndls", "pairnorm", "frodo"]
    for dname in datasets:
        acc_table: Dict[str, List[float]] = {v: [] for v in variants}
        ds = pp.build_dataset(dname, cfg)
        data = ds[0]
        split_idx = {
            "train": ds[0].train_mask.nonzero(as_tuple=True)[0],
            "valid": ds[0].val_mask.nonzero(as_tuple=True)[0],
            "test": ds[0].test_mask.nonzero(as_tuple=True)[0],
        }
        in_dim, out_dim = data.num_features, ds.num_classes
        for c in corruption:
            x_orig = data.x.clone()
            if c > 0:
                mask = torch.rand_like(x_orig.float()) > c
                data.x = x_orig * mask
            for variant in variants:
                nf = norm_factory(variant, cfg.hidden, frodo_h)
                model = GATNet(in_dim, cfg.hidden // 8, out_dim, 16, 8, cfg.dropout, nf)
                _, metrics = run_training(model, data, split_idx, cfg,
                                          f"Exp2-{dname}-corr{c}-{variant}")
                acc_table[variant].append(metrics["test_acc"])
            data.x = x_orig  # restore
        # plot per-dataset
        save_lineplot(corruption, {v: acc_table[v] for v in variants},
                      "Feature corruption", "Accuracy",
                      f"Accuracy vs corruption – {dname}", f"accuracy_{dname.lower()}.pdf")

def experiment3_scale(cfg: GlobalConfig, frodo_h: FRODOHyper):
    # ogbn-papers100M – may fail on low memory, be graceful
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name="ogbn-papers100M", root=os.path.join(cfg.data_root, "ogbn-papers100M"))
    except Exception as e:  # pragma: no cover – runtime env dependent
        print("Skipping ogbn-papers100M →", e)
        ds = None
    if ds is not None:
        data = ds[0]
        split_idx = ds.get_idx_split()
        nf = norm_factory("frodo", cfg.hidden * 2, frodo_h)
        model = APPNPNet(data.num_features, 256, ds.num_classes, 64, 10, 0.0, nf)
        run_training(model, data, split_idx, cfg, "Exp3A-papers100M-frodo")

    # synthetic CSBM heterophily sweep (small – always runs)
    import networkx as nx
    from networkx.generators.community import stochastic_block_model
    n = 100_000
    p_in = 0.01
    ratios = [5, 2, 1, 0.5]
    acc_vs_ratio: Dict[str, List[float]] = {v: [] for v in ["vanilla", "contranorm", "dropedge", "frodo"]}
    for r in ratios:
        p_out = p_in / r
        sizes = [n // 2, n - n // 2]
        probs = [[p_in, p_out], [p_out, p_in]]
        G = stochastic_block_model(sizes, probs, seed=0)
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_index = to_undirected(edge_index)
        x = torch.randn(n, cfg.hidden)
        y = torch.tensor([0] * sizes[0] + [1] * sizes[1])
        data = Data(x=x, edge_index=edge_index, y=y)
        idx = torch.randperm(n)
        split_idx = {"train": idx[: n // 2], "valid": idx[n // 2 : 3 * n // 4], "test": idx[3 * n // 4 :]}
        for variant in acc_vs_ratio.keys():
            nf = norm_factory(variant, cfg.hidden, frodo_h)
            model = GCNNet(cfg.hidden, cfg.hidden, 2, 32, cfg.dropout, nf)
            _, metrics = run_training(model, data, split_idx, cfg, f"Exp3B-CSBM-r{r}-{variant}")
            acc_vs_ratio[variant].append(metrics["test_acc"])
    save_lineplot(ratios, acc_vs_ratio, "p/q", "Accuracy", "Heterophily sweep – CSBM", "accuracy_csbm.pdf")

# ------------------------------------------------------------------
#  YAML config loading
# ------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _cfg_dict = yaml.safe_load(fp)
GLOBAL_CFG = GlobalConfig(**_cfg_dict["global"])
FRODO_HYPER = FRODOHyper(**_cfg_dict["frodo"])

# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main():
    os.makedirs(GLOBAL_CFG.data_root, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    set_global_seeds(GLOBAL_CFG.seeds[0])

    print("\n================ EXPERIMENT 1 – Depth scaling ================")
    experiment1_depth(GLOBAL_CFG, FRODO_HYPER)

    print("\n================ EXPERIMENT 2 – Denoising ===================")
    experiment2_denoise(GLOBAL_CFG, FRODO_HYPER)

    print("\n================ EXPERIMENT 3 – Scalability ================")
    experiment3_scale(GLOBAL_CFG, FRODO_HYPER)


if __name__ == "__main__":  # pragma: no cover
    main()
