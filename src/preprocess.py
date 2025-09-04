# Updated preprocess.py – dataset loading & synthetic graph helpers
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import torch
from torch_geometric.utils import from_networkx

# -----------------------------------------------------------------------------
#  Synthetic graphs used for fast verification
# -----------------------------------------------------------------------------


def _ring_of_cliques() -> Any:
    g = nx.ring_of_cliques(30, 10)
    for v in g.nodes():
        g.nodes[v]["x"] = torch.nn.functional.one_hot(torch.tensor(v % 10), 10).float()
        g.nodes[v]["y"] = v // 10
    return from_networkx(g)


def _path_of_cliques() -> Any:
    g = nx.connected_caveman_graph(30, 10)
    for v in g.nodes():
        g.nodes[v]["x"] = torch.nn.functional.one_hot(torch.tensor(v % 10), 10).float()
        g.nodes[v]["y"] = v // 10
    return from_networkx(g)


def synthetic_graph(name: str):
    if name == "Path-of-Cliques":
        return _path_of_cliques()
    if name == "Ring-of-Cliques":
        return _ring_of_cliques()
    raise KeyError(f"Unknown synthetic graph '{name}'")


# -----------------------------------------------------------------------------
#  Public loader (real + synthetic)
# -----------------------------------------------------------------------------


def _add_masks(data, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 0):
    """Creates boolean train/val/test masks if they are absent."""
    if getattr(data, "train_mask", None) is not None:
        return data  # already present

    torch.manual_seed(seed)
    N = data.num_nodes
    idx = torch.randperm(N)
    n_train = int(train_ratio * N)
    n_val = int(val_ratio * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    data.train_mask = torch.zeros(N, dtype=torch.bool)
    data.val_mask = torch.zeros(N, dtype=torch.bool)
    data.test_mask = torch.zeros(N, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    return data


def load_dataset(name: str):
    """Loads either a real benchmark dataset (if requested) or one of the built-in
    synthetic graphs.  Heavy libraries such as torch_sparse / torch_scatter are
    intentionally avoided; hence, the real datasets are imported lazily only
    when necessary.
    """

    # --- real datasets -------------------------------------------------------
    if name in {"Cora", "CiteSeer", "PubMed"}:
        from torch_geometric.datasets import Planetoid  # lazy import

        data = Planetoid(root=f"data/{name}", name=name)[0]
        return _add_masks(data)

    if name in {"Texas", "Cornell", "Wisconsin"}:
        from torch_geometric.datasets import WebKB  # lazy import

        data = WebKB(root=f"data/{name}", name=name)[0]
        return _add_masks(data)

    if name in {"Peptides-func", "Peptides-struct", "PCQM-Contact"}:
        from torch_geometric.datasets import LRGB  # lazy import

        key = name.replace("-", "")
        data = LRGB(root="data/LRGB", name=key)[0]
        return _add_masks(data)

    # --- synthetic fallback --------------------------------------------------
    data = synthetic_graph(name)
    return _add_masks(data)
