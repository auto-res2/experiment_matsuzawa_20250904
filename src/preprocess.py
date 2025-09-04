"""src/preprocess.py
Data creation / loading and splitting utilities.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Callable

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# -----------------------------------------------------------------------------
#                        Synthetic graphs (PoC, RoC, Mix)
# -----------------------------------------------------------------------------
SYN_SEED = 1234
np.random.seed(SYN_SEED)
random.seed(SYN_SEED)


def _poc_graph() -> Data:
    """Path-of-Cliques."""
    fname = DATA_DIR / "synthetic/poc.npz"
    if fname.exists():
        arr = np.load(fname)
        return Data(
            x=torch.tensor(arr["x"], dtype=torch.float32),
            edge_index=torch.tensor(arr["edge_index"], dtype=torch.long),
            y=torch.tensor(arr["y"], dtype=torch.long),
        )
    g = nx.Graph()
    label = []
    vid = 0
    for c in range(30):
        nodes = list(range(vid, vid + 10))
        g.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                if i < j:
                    g.add_edge(i, j)
        if c > 0:
            g.add_edge(vid - 1, vid)
        label += [c] * 10
        vid += 10
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    edge_index = to_undirected(edge_index)
    x = torch.eye(10).repeat(30, 1) + 0.01 * torch.randn(300, 10)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.tensor(label, dtype=torch.long)
    fname.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fname, edge_index=edge_index.numpy(), x=x.numpy(), y=y.numpy())
    return Data(x=x, edge_index=edge_index, y=y)


def _roc_graph() -> Data:
    fname = DATA_DIR / "synthetic/roc.npz"
    if fname.exists():
        arr = np.load(fname)
        return Data(
            x=torch.tensor(arr["x"], dtype=torch.float32),
            edge_index=torch.tensor(arr["edge_index"], dtype=torch.long),
            y=torch.tensor(arr["y"], dtype=torch.long),
        )
    g = nx.cycle_graph(30)
    label = []
    vid_off = 0
    for c, clique_center in enumerate(g.nodes()):
        clique_nodes = list(range(vid_off, vid_off + 10))
        for i in clique_nodes:
            for j in clique_nodes:
                if i < j:
                    g.add_edge(i, j)
        g.add_edge(clique_center, vid_off)
        label += [c] * 10
        vid_off += 10
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    edge_index = to_undirected(edge_index)
    x = torch.eye(10).repeat(30, 1) + 0.01 * torch.randn(300, 10)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = torch.tensor(label, dtype=torch.long)
    fname.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fname, edge_index=edge_index.numpy(), x=x.numpy(), y=y.numpy())
    return Data(x=x, edge_index=edge_index, y=y)


def _mx_graph() -> Data:
    fname = DATA_DIR / "synthetic/mx.npz"
    if fname.exists():
        arr = np.load(fname)
        return Data(
            x=torch.tensor(arr["x"], dtype=torch.float32),
            edge_index=torch.tensor(arr["edge_index"], dtype=torch.long),
            y=torch.tensor(arr["y"], dtype=torch.long),
        )
    poc = _poc_graph()
    roc = _roc_graph()
    x = torch.cat([poc.x, roc.x], dim=0)
    y = torch.cat([poc.y, roc.y + 30], dim=0)
    edge_index = torch.cat([poc.edge_index, roc.edge_index + poc.x.size(0)], dim=1)
    rng = np.random.RandomState(SYN_SEED)
    bridges = torch.tensor([[rng.randint(0, poc.x.size(0)), rng.randint(poc.x.size(0), x.size(0))] for _ in range(10)]).t()
    edge_index = torch.cat([edge_index, bridges], dim=1)
    edge_index = to_undirected(edge_index)
    fname.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fname, edge_index=edge_index.numpy(), x=x.numpy(), y=y.numpy())
    return Data(x=x, edge_index=edge_index, y=y)

# -----------------------------------------------------------------------------
#                           Public loader map
# -----------------------------------------------------------------------------
dataset_map: Dict[str, Callable[[], Data]] = {
    "path_of_cliques": _poc_graph,
    "ring_of_cliques": _roc_graph,
    "mixture_graph": _mx_graph,
}

# -----------------------------------------------------------------------------
#                                 Splits
# -----------------------------------------------------------------------------

def random_split(data: Data, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 0):
    torch.manual_seed(seed)
    idx = torch.randperm(data.num_nodes)
    n = data.num_nodes
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[idx[: n - n_val - n_test]] = True
    data.val_mask[idx[n - n_val - n_test : n - n_test]] = True
    data.test_mask[idx[-n_test:]] = True
    return data
