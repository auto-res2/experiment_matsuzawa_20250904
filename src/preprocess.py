"""
src/preprocess.py – dataset loading & synthetic generators
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import LRGB, Planetoid, WebKB
from torch_geometric.utils import from_networkx

# -----------------------------------------------------------------------------
# Synthetic clique-based graphs (Path-of-Cliques, Ring-of-Cliques, Mixture)
# -----------------------------------------------------------------------------


def _gen_clique_graph(kind: str, n_cliques: int = 30, clique_size: int = 10) -> Data:
    if kind == "poc":
        g = nx.connected_caveman_graph(n_cliques, clique_size)  # path-of-cliques
    elif kind == "roc":
        g = nx.ring_of_cliques(n_cliques, clique_size)  # ring-of-cliques
    else:
        raise ValueError(f"Unknown kind {kind}")

    # Node features = slightly noised one-hot inside each clique
    for v in g.nodes:
        base = torch.zeros(clique_size)
        base[v % clique_size] = 1.0
        g.nodes[v]["x"] = (base + 0.01 * torch.randn_like(base)).float()
        g.nodes[v]["y"] = v // clique_size
    return from_networkx(g)


def _get_synthetic(name: str) -> Data:
    if name == "Path-of-Cliques":
        return _gen_clique_graph("poc")
    if name == "Ring-of-Cliques":
        return _gen_clique_graph("roc")
    if name == "Mixture":
        poc = _gen_clique_graph("poc")
        roc = _gen_clique_graph("roc")
        roc.edge_index += poc.num_nodes
        data = Data(
            x=torch.cat([poc.x, roc.x]),
            edge_index=torch.cat([poc.edge_index, roc.edge_index], dim=1),
            y=torch.cat([poc.y, roc.y]),
        )
        return data
    raise KeyError(name)


# -----------------------------------------------------------------------------
# Public loader – downloads real datasets on-the-fly if needed
# -----------------------------------------------------------------------------

def load_dataset(name: str, split_seed: int = 0):
    if name in {"Path-of-Cliques", "Ring-of-Cliques", "Mixture"}:
        data = _get_synthetic(name)
    elif name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(root="data/" + name, name=name)[0]
    elif name in {"Texas", "Cornell", "Wisconsin"}:
        data = WebKB(root="data/" + name, name=name)[0]
    elif name in {"Peptides-func", "Peptides-struct", "PCQM-Contact"}:
        key = name.replace("-", "")  # LRGB naming quirk
        data = LRGB(root="data/LRGB", name=key)[0]
    else:
        raise KeyError(f"Unknown dataset {name}")

    # Split masks -----------------------------------------------------------
    torch.manual_seed(split_seed)
    if getattr(data, "train_mask", None) is None:
        n = data.num_nodes
        perm = torch.randperm(n)
        tr, va = int(0.6 * n), int(0.2 * n)
        train_idx, val_idx, test_idx = perm[:tr], perm[tr : tr + va], perm[tr + va :]
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
    return data
