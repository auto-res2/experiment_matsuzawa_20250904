"""src/train.py
Training utilities, model definitions, and the full optimisation loop
extracted from the original monolithic experiment script.
"""
from __future__ import annotations
import math, random, time, contextlib, pathlib, typing, json
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops
import torch_sparse

# ---------------------------------------------------------------------------
# Configuration helper (light-weight replacement of the previous cfg(...))
# ---------------------------------------------------------------------------
import yaml
_CFG_PATH = pathlib.Path("config/config.yaml")
with _CFG_PATH.open() as fp:
    _CFG = yaml.safe_load(fp)

def cfg(*keys:str, default=None):
    node: typing.Any = _CFG
    for k in keys:
        if node is None:
            return default
        node = node.get(k, default)
    return node if node is not None else default

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seeds(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@contextlib.contextmanager
def cuda_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0=time.perf_counter()
    yield lambda: time.perf_counter()-t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Geometry – Forman curvature (needed by CurvAMP)
# ---------------------------------------------------------------------------

def _degree(index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row = index[0]
    return torch_sparse.degree(row, num_nodes=num_nodes).clamp_min_(1)

def forman_edge(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    deg = _degree(edge_index, num_nodes)
    return 4.0 - deg[row] - deg[col]

def forman_node(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    kappa_e = forman_edge(edge_index, num_nodes)
    row = edge_index[0]
    kappa_n = torch.zeros(num_nodes, device=row.device)
    kappa_n.index_add_(0, row, kappa_e)
    deg = _degree(edge_index, num_nodes)
    return kappa_n / deg

# ---------------------------------------------------------------------------
# Normalisation layers
# ---------------------------------------------------------------------------
class PairNorm(nn.Module):
    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(0, keepdim=True)
        x = x - mu
        row_norm = x.pow(2).sum(1, keepdim=True).sqrt().mean()
        return self.s * x / (row_norm + 1e-8)

class CurvPair(PairNorm):
    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(-kappa).unsqueeze(1)
        return super().forward(x) * scale

# ---------------------------------------------------------------------------
# CurvAMP convolution (multi-scale, curvature gate, online rewiring)
# ---------------------------------------------------------------------------
class CurvAMPConv(MessagePassing):
    def __init__(self, in_dim:int, out_dim:int, *, K:int=3,
                 rewired_ratio:float=0.02, rewired_weight:float=0.3):
        super().__init__(aggr="add")
        self.K, self.rr, self.w_re = K, rewired_ratio, rewired_weight
        self.proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(K)
        ])
        self.gate = nn.Sequential(
            nn.Linear(2,16), nn.GELU(), nn.Linear(16,K)
        )
        self.norm = CurvPair()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        N = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        with torch.no_grad():
            kappa_node = forman_node(edge_index, N)
            deg = torch.bincount(edge_index[0], minlength=N).float().clamp_min_(1)

        row, col = edge_index
        deg_inv_sqrt = deg.pow(-0.5)
        vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        A = torch.sparse_coo_tensor(edge_index, vals, (N, N))

        bank = [x]
        for _ in range(1, self.K):
            bank.append(torch.sparse.mm(A, bank[-1]))
        bank = [proj(b) for proj, b in zip(self.proj, bank)]

        alpha = torch.softmax(self.gate(torch.stack([kappa_node, deg], 1)), -1)
        h = sum(alpha[:, k:k+1] * bank[k] for k in range(self.K))

        # --------------------------------------------------
        # On-the-fly micro-rewiring (negative curvature pairs)
        # --------------------------------------------------
        if self.rr > 0:
            m = math.floor(self.rr * edge_index.size(1))
            if m > 0:
                with torch.no_grad():
                    cand_u = torch.randint(0, N, (3*m,), device=x.device)
                    cand_v = torch.randint(0, N, (3*m,), device=x.device)
                    neg_mask = (kappa_node[cand_u] + kappa_node[cand_v]) < 0
                    cand_u, cand_v = cand_u[neg_mask], cand_v[neg_mask]
                    if cand_u.numel():
                        score = torch.exp(-(kappa_node[cand_u] + kappa_node[cand_v]))
                        idx = torch.multinomial(score / score.sum(), min(m, cand_u.size(0)), False)
                        new_edges = torch.stack([cand_u[idx], cand_v[idx]])
                        edge_index = torch.cat([edge_index, new_edges], 1)
                        extra_vals = self.w_re * torch.ones(new_edges.size(1), device=x.device)
                        vals = torch.cat([vals, extra_vals], 0)
                        A = torch.sparse_coo_tensor(edge_index, vals, (N, N))

        out = self.norm(h, kappa_node)
        return out, edge_index

# ---------------------------------------------------------------------------
# Network wrappers
# ---------------------------------------------------------------------------
class GCNStack(nn.Module):
    def __init__(self, in_dim:int, hid:int, out_dim:int, depth:int, dropout:float=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([
            GCNConv(in_dim if i == 0 else hid, hid) for i in range(depth)
        ])
        self.head = nn.Linear(hid, out_dim)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)

class PairNormGCN(GCNStack):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pn = PairNorm()
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = self.pn(F.relu(conv(x, ei)))
        return self.head(x)

class CurvAMPNet(nn.Module):
    def __init__(self, in_dim:int, hid:int, out_dim:int, depth:int, *, K:int=3, rewired_ratio:float=0.02):
        super().__init__()
        self.layers = nn.ModuleList([
            CurvAMPConv(in_dim if i == 0 else hid, hid, K=K, rewired_ratio=rewired_ratio)
            for i in range(depth)
        ])
        self.head = nn.Linear(hid, out_dim)
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for layer in self.layers:
            x, ei = layer(x, ei)
            x = torch.relu(x)
        return self.head(x)

MODELS = {"GCN": GCNStack, "PairNorm": PairNormGCN, "CurvAMP": CurvAMPNet}

# ---------------------------------------------------------------------------
# Training loop (single split, early stopping)
# ---------------------------------------------------------------------------

def train_model(model: nn.Module, data, *, seed:int = 0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)

    set_seeds(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg('lr', default=5e-4),
                                  weight_decay=cfg('weight_decay', default=0.0))
    best_val = float('inf')
    best_state = None
    patience = cfg('patience', default=100)
    stale = 0
    history = []

    with cuda_timer() as total_time:
        for epoch in range(cfg('max_epochs', default=2000)):
            model.train(); optimiser.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward(); optimiser.step()

            model.eval()
            with torch.no_grad():
                val_out = model(data)
                val_loss = F.cross_entropy(val_out[data.val_mask], data.y[data.val_mask])
            history.append(float(val_loss))

            if val_loss < best_val:
                best_val = float(val_loss)
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale > patience:
                break
    train_seconds = total_time()
    model.load_state_dict(best_state)

    return {
        "model": model,
        "val_curve": history,
        "train_time": train_seconds
    }
