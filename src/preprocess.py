"""
preprocess.py – data loading and curvature pre-computation
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.utils import add_self_loops

# datasets
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx
from graphricci.curvature import OllivierRicci  # pip install GraphRicciCurvature==0.6

# --------------------------------------------------

def load_dataset(name: str, data_dir: Path = Path("data")):
    name_l = name.lower()
    if name_l in {"cora", "citeseer"}:
        data = Planetoid(root=data_dir.as_posix(), name=name, split="public")[0]
        data.edge_index, _ = add_self_loops(data.edge_index)
    elif name_l in {"chameleon", "squirrel"}:
        data = WikipediaNetwork(root=data_dir.as_posix(), name=name, geom_gcn_preprocess=False)[0]
        data.edge_index, _ = add_self_loops(data.edge_index)
    elif name_l == "ogbn-arxiv":
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=data_dir.as_posix())
        data = ds[0]
        split_idx = ds.get_idx_split()
        for split in ("train", "valid", "test"):
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            setattr(data, f"{split[:3]}_mask", mask)  # train_mask etc.
        data.edge_index, _ = add_self_loops(data.edge_index)
    else:
        raise ValueError(f"Dataset {name} not supported.")
    return data


# --------------------------------------------------

def compute_or_curvature(data, cache_file: Path) -> torch.Tensor:
    """Compute (or load) Ollivier-Ricci curvature for each edge."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.is_file():
        return torch.load(cache_file)

    g = data.to_networkx().to_undirected()
    orc = OllivierRicci(g, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()

    kappa_vals = []
    rows, cols = data.edge_index
    for u, v in zip(rows.tolist(), cols.tolist()):
        kappa_vals.append(orc.G[u][v]["ricciCurvature"])
    kappa = torch.tensor(kappa_vals, dtype=torch.float32)
    torch.save(kappa, cache_file)
    return kappa
