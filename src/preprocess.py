from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import from_networkx
import networkx as nx

################################################################################
#                               DATA HELPERS                                   #
################################################################################

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic clique/graph generators used in Exp-1.
# ---------------------------------------------------------------------------

def _clique(kind: str, nc: int = 30, cs: int = 10) -> Data:
    if kind == "poc":
        g = nx.connected_caveman_graph(nc, cs)
    elif kind == "roc":
        g = nx.ring_of_cliques(nc, cs)
    else:
        raise ValueError("Unknown synthetic kind: %s" % kind)

    for v in g.nodes:
        base = torch.zeros(cs)
        base[v % cs] = 1.0
        g.nodes[v]["x"] = (base + 0.01 * torch.randn_like(base)).float()
        g.nodes[v]["y"] = v // cs
    return from_networkx(g)


def _synthetic(name: str) -> Data:
    if name == "Path-of-Cliques":
        return _clique("poc")
    if name == "Ring-of-Cliques":
        return _clique("roc")
    if name == "Mixture":
        a = _clique("poc")
        b = _clique("roc")
        b.edge_index = b.edge_index + a.num_nodes  # shift indices
        return Data(
            x=torch.cat([a.x, b.x]),
            edge_index=torch.cat([a.edge_index, b.edge_index], 1),
            y=torch.cat([a.y, b.y]),
        )
    raise KeyError("Unrecognised synthetic dataset: %s" % name)


# ---------------------------------------------------------------------------
# Public loader – falls back to PyG datasets when necessary
# ---------------------------------------------------------------------------

def load_dataset(name: str, split_seed: int = 0) -> Data:
    """Return a single "Data" object with train/val/test boolean masks."""

    if name in {"Path-of-Cliques", "Ring-of-Cliques", "Mixture"}:
        data = _synthetic(name)
    elif name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(root=_DATA_ROOT / name, name=name)[0]
    elif name in {"Texas", "Cornell", "Wisconsin"}:
        data = WebKB(root=_DATA_ROOT / name, name=name)[0]
    else:
        # LRGB datasets
        from torch_geometric.datasets import LRGBDataset  # local import (heavy)

        key = name.replace("-", "")
        data = LRGBDataset(root=_DATA_ROOT / "LRGB", name=key)[0]

    # ---------------- train/val/test split ----------------
    torch.manual_seed(split_seed)
    if getattr(data, "train_mask", None) is None:
        n = data.num_nodes
        perm = torch.randperm(n)
        tr, va = int(0.6 * n), int(0.2 * n)
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        data.train_mask[perm[:tr]] = True
        data.val_mask[perm[tr : tr + va]] = True
        data.test_mask[perm[tr + va :]] = True
    return data