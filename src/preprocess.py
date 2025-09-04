"""src/preprocess.py
Dataset loading helpers and other data-related utilities.
"""
from __future__ import annotations
from pathlib import Path

from torch_geometric.datasets import Planetoid, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset

# ------------------------------------------------------------------
#  Dataset builder
# ------------------------------------------------------------------

def build_dataset(name: str, cfg) -> "torch_geometric.data.InMemoryDataset":
    root = Path(cfg.data_root) / name
    if name in {"Cora", "Citeseer", "Pubmed"}:
        return Planetoid(root=str(root), name=name, split="public")
    if name.lower() == "chameleon":
        return WikipediaNetwork(root=str(root), name="chameleon")
    if name == "ogbn-arxiv":
        return PygNodePropPredDataset(name="ogbn-arxiv", root=str(root))
    raise RuntimeError(f"Dataset {name} not supported.")
