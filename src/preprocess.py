"""
preprocess.py – data loading, preprocessing & split helpers.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np

# -----------------------------------------------------------------------------
#  Data loading utilities
# -----------------------------------------------------------------------------

def load_dataset(name: str, data_root: str):
    root = Path(data_root) / name
    root.mkdir(parents=True, exist_ok=True)
    try:
        if name in {"Cora", "Citeseer", "Pubmed"}:
            ds = Planetoid(root=str(root), name=name, split="public")
            data = ds[0]
        elif name.lower() == "chameleon":
            ds = WikipediaNetwork(root=str(root), name="chameleon")
            data = ds[0]
        elif name == "ogbn-arxiv":
            ds = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))
            data = ds[0]
            data.y = data.y.squeeze()
        else:
            raise RuntimeError(f"Unknown dataset {name}")
    except Exception as e:
        raise RuntimeError(f"Dataset loading failed for {name}: {e}") from e

    #  standardise dtypes / add self-loops
    data.x = data.x.to(torch.float32)
    data.edge_index, _ = add_self_loops(data.edge_index)
    return data

# -----------------------------------------------------------------------------
#  Split helper – 60/20/20 random partition (Planetoid style)
# -----------------------------------------------------------------------------

def random_split(n_nodes: int, seed: int) -> Dict[str, torch.Tensor]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n_nodes)
    rng.shuffle(idx)
    n_train = int(0.6 * n_nodes)
    n_val = int(0.2 * n_nodes)
    return {
        "train": torch.tensor(idx[:n_train]),
        "valid": torch.tensor(idx[n_train:n_train + n_val]),
        "test": torch.tensor(idx[n_train + n_val:]),
    }
