"""src/preprocess.py
Dataset loading & synthetic-graph generation.
"""
from __future__ import annotations
from pathlib import Path
import typing, torch, networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, LRGBDataset
from torch_geometric.utils import from_networkx, add_self_loops

_DATA_ROOT = Path("data")
_DATA_ROOT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic generators (Path-/Ring-of-Cliques, Mixture)
# ---------------------------------------------------------------------------

def _clique_graph(kind:str, num_cliques:int = 30, clique_size:int = 10) -> nx.Graph:
    if kind == "poc":
        return nx.connected_caveman_graph(num_cliques, clique_size)
    if kind == "roc":
        return nx.ring_of_cliques(num_cliques, clique_size)
    raise KeyError(kind)

def _synthetic_dataset(name:str) -> Data:
    if name == "Path-of-Cliques":
        g = _clique_graph("poc")
    elif name == "Ring-of-Cliques":
        g = _clique_graph("roc")
    elif name == "Mixture":
        g1 = _clique_graph("poc"); g2 = _clique_graph("roc")
        mapping = {n: n + g1.number_of_nodes() for n in g2.nodes}
        g2 = nx.relabel_nodes(g2, mapping)
        g = nx.compose(g1, g2)
    else:
        raise KeyError(name)
    cs = 10
    for n in g.nodes:
        base = torch.zeros(cs)
        base[n % cs] = 1.0
        g.nodes[n]['x'] = (base + 0.01 * torch.randn_like(base)).float()
        g.nodes[n]['y'] = n // cs
    data = from_networkx(g)
    data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])
    data.y = torch.tensor([data.y[i] for i in range(data.num_nodes)], dtype=torch.long)
    return data

# ---------------------------------------------------------------------------
# Public loader returning a PyG Data object with train/val/test masks
# ---------------------------------------------------------------------------

def load_dataset(name:str, *, split_seed:int = 0):
    if name in {"Path-of-Cliques", "Ring-of-Cliques", "Mixture"}:
        data = _synthetic_dataset(name)
    elif name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(root=_DATA_ROOT / name, name=name)[0]
    elif name in {"Texas", "Cornell", "Wisconsin"}:
        data = WebKB(root=_DATA_ROOT / name, name=name)[0]
    else:
        key = name.replace("-", "")
        data = LRGBDataset(root=_DATA_ROOT / "LRGB", name=key)[0]

    if not hasattr(data, 'train_mask'):
        torch.manual_seed(split_seed)
        n = data.num_nodes
        perm = torch.randperm(n)
        tr, va = int(0.6 * n), int(0.2 * n)
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask   = torch.zeros(n, dtype=torch.bool)
        data.test_mask  = torch.zeros(n, dtype=torch.bool)
        data.train_mask[perm[:tr]]     = True
        data.val_mask[perm[tr:tr+va]]  = True
        data.test_mask[perm[tr+va:]]   = True

    data.x = (data.x - data.x.mean(1, keepdim=True)) / (data.x.std(1, keepdim=True) + 1e-8)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return data
