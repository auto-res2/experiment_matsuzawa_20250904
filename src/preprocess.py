"""
preprocess.py – dataset loading & synthetic graph helpers
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import torch
from torch_geometric.datasets import Planetoid, WebKB, LRGB
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

def load_dataset(name: str):
    if name in {"Cora", "CiteSeer", "PubMed"}:
        return Planetoid(root=f"data/{name}", name=name)[0]
    if name in {"Texas", "Cornell", "Wisconsin"}:
        return WebKB(root=f"data/{name}", name=name)[0]
    if name in {"Peptides-func", "Peptides-struct", "PCQM-Contact"}:
        key = name.replace("-", "")
        return LRGB(root="data/LRGB", name=key)[0]
    # --- synthetic fallback ---
    return synthetic_graph(name)
