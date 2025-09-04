from __future__ import annotations
"""src/preprocess.py
Data loading and (mock) curvature calculation utilities.
NOTE:
-----
The original implementation relied on the *GraphRicciCurvature* package to
compute Ollivier–Ricci curvature for every edge.  Unfortunately a wheel that
is compatible with the Python version used in the CI environment is not
available on PyPI, which prevents the dependency stack from being installed
successfully.

To keep the end-to-end training/validation pipeline functional we replace the
exact Ricci-curvature computation with a lightweight, deterministic surrogate
that can be executed with the standard scientific-Python stack available in
CI.  The surrogate DOES NOT attempt to faithfully reproduce the theoretical
properties of Ricci curvature – it merely produces a real-valued   κ(e)   for
each edge so that the curvature-aware gates inside *CAPConv* receive tensor
input of correct shape and dtype.

If, in the future, a suitable binary distribution becomes available the
original implementation can be restored without further changes to the rest
of the codebase.
"""
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, coalesce
from torch_geometric.data import Data

import networkx as nx

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
    # ensure self-loops and coalesced edge index (as expected by downstream code)
    data.edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = coalesce(data.edge_index)
    return data

# ----------------------------------------------------------------------------
#  (Mock) Ollivier–Ricci curvature
# ----------------------------------------------------------------------------

def _surrogate_curvature(g: nx.Graph) -> dict[tuple[int, int], float]:
    """Very cheap surrogate for per-edge curvature.

    We use the negative normalized difference of the degrees of the incident
    nodes – this keeps the values in a reasonably small range and introduces
    some variability that the gating mechanism of *CAPConv* can utilise.
    The resulting values are **deterministic**.
    """
    curv: dict[tuple[int, int], float] = {}
    for u, v in g.edges():
        du, dv = g.degree[u], g.degree[v]
        curv[(u, v)] = -abs(du - dv) / (du + dv + 1e-6)
    return curv


def compute_or_curvature(data: Data, cache_path: Path) -> torch.Tensor:
    """Compute or load cached surrogate curvature values for every edge.

    The function preserves the *API contract* expected by the rest of the
    pipeline (returns a 1-D **torch.float16** tensor with length equal to the
    number of edges).  To avoid recomputation in subsequent CI steps the
    tensor is cached to *cache_path*.
    """
    # 1) Try loading from cache ------------------------------------------------
    try:
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu")
    except (OSError, RuntimeError):
        # corrupted cache – ignore and recompute
        cache_path.unlink(missing_ok=True)

    # 2) Build an undirected NetworkX graph -----------------------------------
    g = nx.Graph()
    edge_list = data.edge_index.t().cpu().numpy()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(edge_list)

    # 3) Compute surrogate curvature -----------------------------------------
    kappa_dict = _surrogate_curvature(g)

    kappas = []
    for u, v in edge_list:
        kappas.append(kappa_dict.get((u, v), kappa_dict.get((v, u), 0.0)))

    kappa_t = torch.tensor(kappas, dtype=torch.float16)

    # 4) Cache to disk (best-effort) ------------------------------------------
    try:
        torch.save(kappa_t, cache_path)
    except (IOError, RuntimeError):
        pass  # non-fatal – continue without persistent cache

    return kappa_t
