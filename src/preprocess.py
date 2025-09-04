"""src/preprocess.py
Data loading and curvature calculation utilities.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, coalesce
from torch_geometric.data import Data

import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci

__all__ = ["load_planetoid", "compute_or_curvature"]

DATA_DIR = Path(__file__).resolve().parent / ".." / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).resolve().parent / ".." / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
#  Load Planetoid dataset
# ----------------------------------------------------------------------------

def load_planetoid(name: str) -> Data:
    """Download/prepare Planetoid dataset (Cora, CiteSeer, PubMed)."""
    ds = Planetoid(root=str(DATA_DIR), name=name)
    data = ds[0]
    data.edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = coalesce(data.edge_index)
    return data

# ----------------------------------------------------------------------------
#  Ollivier–Ricci curvature
# ----------------------------------------------------------------------------

def compute_or_curvature(data: Data, cache_path: Path) -> torch.Tensor:
    """Compute (or load cached) Ollivier-Ricci curvature for every edge."""
    try:
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
    except (OSError, RuntimeError):
        # corrupted cache – ignore and recompute
        cache_path.unlink(missing_ok=True)

    # --- convert to undirected networkx graph ---
    g = nx.Graph()
    edge_list = data.edge_index.t().cpu().numpy()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(edge_list)

    orc = OllivierRicci(g, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    kappa_dict = {(u, v): k for u, v, k in orc.G.edges(data="ricciCurvature")}

    kappas = []
    for u, v in edge_list:
        kappas.append(kappa_dict.get((u, v), kappa_dict.get((v, u), 0.0)))
    kappa_t = torch.tensor(kappas, dtype=torch.float16)

    # safely cache to disk
    try:
        torch.save(kappa_t, cache_path)
    except (IOError, RuntimeError):
        pass  # non-fatal
    return kappa_t
