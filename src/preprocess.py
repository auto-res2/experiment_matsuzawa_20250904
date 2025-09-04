"""src/preprocess.py
Data-loading, utility helpers and graph preprocessing.
"""
from __future__ import annotations

from typing import Tuple, Dict, List
from pathlib import Path

import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# ----------------------------------------------------------------------------
# 1)  MISCELLANEOUS UTILS -----------------------------------------------------
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_header(title: str):
    line = "-" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def fail(msg: str):
    print(f"[FATAL] {msg}")
    raise SystemExit(1)

# ----------------------------------------------------------------------------
# 2)  DATA LOADING ------------------------------------------------------------
# ----------------------------------------------------------------------------

def load_planetoid(name: str, data_root: str) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if name == "cora":
        ds = CoraGraphDataset(raw_dir=data_root)
    elif name == "citeseer":
        ds = CiteseerGraphDataset(raw_dir=data_root)
    elif name == "pubmed":
        ds = PubmedGraphDataset(raw_dir=data_root)
    else:
        fail(f"Unknown Planetoid dataset: {name}")

    g = ds[0]
    g = dgl.add_self_loop(g)
    feats = g.ndata["feat"].float()
    labels = g.ndata["label"].long()

    train_mask = g.ndata["train_mask"].bool()
    val_mask = g.ndata["val_mask"].bool()
    test_mask = g.ndata["test_mask"].bool()
    split = {
        "train": torch.nonzero(train_mask, as_tuple=False).squeeze(),
        "val": torch.nonzero(val_mask, as_tuple=False).squeeze(),
        "test": torch.nonzero(test_mask, as_tuple=False).squeeze(),
    }

    feats = F.normalize(feats, p=2, dim=1)
    return g, feats, labels, split


def load_ogb(name: str, data_root: str):
    dataset = DglNodePropPredDataset(name=name, root=data_root)
    g, labels = dataset[0]
    g = dgl.add_self_loop(g)
    labels = labels.squeeze().long()
    split_idx = dataset.get_idx_split()
    split = {k: v for k, v in split_idx.items()}
    feats = g.ndata["feat"].float()
    feats = F.normalize(feats, p=2, dim=1)
    evaluator = Evaluator(name=name)
    return g, feats, labels, split, evaluator

# ----------------------------------------------------------------------------
# 3)  GRAPH POWER SERIES ------------------------------------------------------
# ----------------------------------------------------------------------------

def sparse_power_series(g: dgl.DGLGraph, K: int) -> List[torch.Tensor]:
    """Pre-compute sparse powers \u00c2^k (row-normalised) as torch.sparse tensors."""
    device = torch.device("cpu")
    A = g.adj(scipy_fmt="coo").astype(np.float32)
    degs = np.maximum(A.sum(1).A1, 1)
    deg_inv = sp.diags(1.0 / degs)
    A_norm = deg_inv @ A

    powers = [sp.eye(g.num_nodes(), dtype=np.float32)]
    for _ in range(1, K + 1):
        powers.append(powers[-1] @ A_norm)

    sparse_tensors = []
    for mat in powers:
        coo = mat.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        sparse = torch.sparse.FloatTensor(indices, values, torch.Size(mat.shape)).to(device)
        sparse_tensors.append(sparse)
    return sparse_tensors
