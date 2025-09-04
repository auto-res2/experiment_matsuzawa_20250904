import math
import random
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing

###########################################################################
#                               UTILITIES                                #
###########################################################################

def set_seeds(seed: int = 0) -> None:
    """Make all operations deterministic (CUDA, cudnn & numpy included)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CUDATimer:
    """A context-manager that returns the wall-clock time of a code block."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.sec = time.perf_counter() - self._t0

###########################################################################
#                       NORMALISATION LAYERS                              #
###########################################################################

class PairNorm(nn.Module):
    """PairNorm as proposed by Zhao & Akoglu (ICML 2020)."""

    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        col_mean = x.mean(0, keepdim=True)
        x = x - col_mean
        row_norm = x.pow(2).sum(1, keepdim=True).sqrt().mean()
        return self.s * x / (row_norm + 1e-8)


class CurvPair(PairNorm):
    """PairNorm variant whose scale depends on node-wise curvature κᵢ."""

    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        scale = torch.sigmoid(-kappa).unsqueeze(1)  # larger when κ is positive
        return out * scale

###########################################################################
#                       CurvAMP CONVOLUTION                               #
###########################################################################

class CurvAMPConv(MessagePassing):
    """One CurvAMP layer as described in the paper header."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 3, rewired_ratio: float = 0.02):
        super().__init__(aggr="add")
        self.K = K
        self.rr = rewired_ratio
        self.proj = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(K)]
        )
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, K))
        self.norm = CurvPair()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        N = x.size(0)
        row = edge_index[0]

        # Build sparse adjacency once
        A = torch.sparse_coo_tensor(
            edge_index, torch.ones(row.size(0), device=x.device), (N, N)
        )

        # Multi-scale bank h^{(k)}
        bank = [x]
        for _ in range(1, self.K):
            bank.append(torch.sparse.mm(A, bank[-1]))
        bank = [p(b) for p, b in zip(self.proj, bank)]

        # Cheap curvature surrogate: 1 / deg
        kappa_e = 1.0 / (torch.bincount(row, minlength=N)[row].clamp(min=1))
        kappa_n = torch.zeros(N, device=x.device).index_add_(0, row, kappa_e)
        deg = torch.bincount(row, minlength=N).float().clamp(min=1)
        kappa_n = kappa_n / deg

        # Attention over scales
        alpha = torch.softmax(self.gate(torch.stack([kappa_n, deg], 1)), -1)
        out = sum(alpha[:, k : k + 1] * bank[k] for k in range(self.K))

        # Online micro-rewiring (forward pass only)
        if self.rr > 0:
            m = int(self.rr * row.numel())
            if m > 0:  # safeguard small graphs
                cand_u = torch.randint(0, N, (m * 3,), device=x.device)
                cand_v = torch.randint(0, N, (m * 3,), device=x.device)
                cand_e = torch.stack([cand_u, cand_v])
                kappa_uv = kappa_n[cand_u] + kappa_n[cand_v]
                p = torch.softmax(-kappa_uv, 0)
                idx = torch.multinomial(p, m, replacement=False)
                edge_index = torch.cat([edge_index, cand_e[:, idx]], 1)

        out = self.norm(out, kappa_n)
        return out, edge_index

###########################################################################
#                               NETWORKS                                  #
###########################################################################

class GCNStack(nn.Module):
    """Vanilla deep GCN with ReLU + Dropout."""

    def __init__(self, in_dim: int, hid: int, out_dim: int, depth: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [GCNConv(in_dim if i == 0 else hid, hid, cached=True) for i in range(depth)]
        )
        self.head = nn.Linear(hid, out_dim)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        feats = []
        for c in self.convs:
            x = F.relu(c(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
            feats.append(x)
        return self.head(x), feats


class PairNormGCN(GCNStack):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pn = PairNorm()

    def forward(self, data):
        x, ei = data.x, data.edge_index
        feats = []
        for c in self.convs:
            x = self.pn(F.relu(c(x, ei)))
            feats.append(x)
        return self.head(x), feats


class CurvAMP(nn.Module):
    """Our proposed architecture (depth L).
    K and rewired_ratio are exposed for ablations.
    """

    def __init__(self, in_dim: int, hid: int, out_dim: int, depth: int, K: int = 3, rewired_ratio: float = 0.02):
        super().__init__()
        self.layers = nn.ModuleList(
            [CurvAMPConv(in_dim if i == 0 else hid, hid, K, rewired_ratio) for i in range(depth)]
        )
        self.head = nn.Linear(hid, out_dim)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        feats = []
        for layer in self.layers:
            x, ei = layer(x, ei)
            x = torch.relu(x)
            feats.append(x)
        return self.head(x), feats


# Registry ----------------------------------------------------------------
MODELS = {
    "GCN": GCNStack,
    "PairNorm": PairNormGCN,
    "CurvAMP": CurvAMP,
}

###########################################################################
#                          TRAINING ROUTINES                              #
###########################################################################

def train_model(model: nn.Module, data, cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Standard supervised training with early stopping on the validation loss."""

    model.to(device)
    data = data.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    best_val = float("inf")
    best_state = None
    patience = int(cfg["patience"])
    epochs_no_improve = 0

    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ------ validation ------
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(data)
            val_loss = F.cross_entropy(val_logits[data.val_mask], data.y[data.val_mask])

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            break

    # Roll back to best epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    return model