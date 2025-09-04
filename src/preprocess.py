"""src/preprocess.py – dataset loading utilities & cached sparse powers"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# ---------------------------------------------------------------------------
#  Paths                                                                      
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Small helpers                                                              
# ---------------------------------------------------------------------------

def _l2_normalise(X: torch.Tensor) -> torch.Tensor:
    return F.normalize(X, p=2, dim=-1)


# ---------------------------------------------------------------------------
#  Public API                                                                 
# ---------------------------------------------------------------------------

def load_data(name: str, out_dir: Path):
    """Return (graph, features, labels, split_idx, evaluator|None)."""
    if name.lower() in {"cora", "citeseer", "pubmed"}:
        if name.lower() == "cora":
            ds = CoraGraphDataset(raw_dir=DATA_ROOT)
        elif name.lower() == "citeseer":
            ds = CiteseerGraphDataset(raw_dir=DATA_ROOT)
        else:
            ds = PubmedGraphDataset(raw_dir=DATA_ROOT)
        g = dgl.add_self_loop(ds[0])
        X = _l2_normalise(g.ndata["feat"].float())
        y = g.ndata["label"].long()
        split = {
            "train": torch.nonzero(g.ndata["train_mask"]).squeeze(),
            "val": torch.nonzero(g.ndata["val_mask"]).squeeze(),
            "test": torch.nonzero(g.ndata["test_mask"]).squeeze(),
        }
        evaluator = None
    elif name.startswith("ogbn"):
        ds = DglNodePropPredDataset(name=name, root=DATA_ROOT)
        g, y = ds[0]
        g = dgl.add_self_loop(g)
        X = _l2_normalise(g.ndata["feat"].float())
        y = y.squeeze().long()
        split = ds.get_idx_split()
        evaluator = Evaluator(name)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return g, X, y, split, evaluator


def cached_powers(g: dgl.DGLGraph, K: int, cache_dir: Path) -> List[torch.Tensor]:
    """Return list [I, Â, Â², …, Â^K] in sparse COO format."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / f"powers_K{K}_{g.num_nodes()}N.pt"
    if fname.exists():
        try:
            return torch.load(fname)
        except Exception:
            pass  # fall-through to recompute if file is corrupted

    A = g.adj(scipy_fmt="coo").astype(np.float32)
    degs = np.maximum(A.sum(1).A1, 1)
    Dinv = sp.diags(1.0 / degs)
    A_norm = Dinv @ A

    mats = [sp.eye(g.num_nodes(), dtype=np.float32)]
    for _ in range(1, K + 1):
        mats.append(mats[-1] @ A_norm)

    res: List[torch.Tensor] = []
    for M in mats:
        coo = M.tocoo()
        idx = torch.tensor([coo.row, coo.col], dtype=torch.long)
        val = torch.tensor(coo.data, dtype=torch.float32)
        res.append(torch.sparse_coo_tensor(idx, val, torch.Size(M.shape)))

    try:
        torch.save(res, fname)
    except Exception:
        pass  # ignore I/O errors – computation is deterministic anyway
    return res
