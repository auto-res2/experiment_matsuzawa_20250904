"""
src/train.py – model definitions, training utilities, optimisation loop
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, MessagePassing

# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Set python, numpy and torch RNG seeds (both CPU & all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Normalisation layers
# -----------------------------------------------------------------------------

class PairNorm(nn.Module):
    """Original PairNorm – keeps node features from collapsing (ICLR-19)."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.s = scale

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        col_mean = x.mean(0, keepdim=True)
        x = x - col_mean
        row_norm = x.pow(2).sum(1, keepdim=True).sqrt().mean()
        return self.s * x / (row_norm + 1e-8)


# -----------------------------------------------------------------------------
# Baseline GCN stacks
# -----------------------------------------------------------------------------

class GCNStack(nn.Module):
    """Vanilla GCN repeated *depth* times."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for layer in range(depth):
            self.convs.append(
                GCNConv(
                    in_dim if layer == 0 else hidden,
                    hidden,
                    cached=True,
                    add_self_loops=True,
                    normalize=True,
                )
            )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x, data.edge_index
        feats: List[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            feats.append(x)
        return self.head(x), feats


class PairNormGCN(GCNStack):
    """GCN + PairNorm after every layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pn = PairNorm()

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x, data.edge_index
        feats: List[torch.Tensor] = []
        for conv in self.convs:
            x = self.pn(conv(x, edge_index).relu())
            feats.append(x)
        return self.head(x), feats


# -----------------------------------------------------------------------------
# FoSR (approximated through PyG's GDC utility)
# -----------------------------------------------------------------------------

try:
    from torch_geometric.transforms import GDC
except ImportError:
    GDC = None


class FoSRGCN(GCNStack):
    """First-order Spectral Rewiring baseline (approximated with GDC)."""

    def __init__(self, *args, **kwargs):
        if GDC is None:
            raise RuntimeError("torch_geometric>=2.1 with GDC required for FoSR baseline")
        super().__init__(*args, **kwargs)

    def forward(self, data):  # type: ignore[override]
        if not hasattr(data, "_fosr_cached"):
            data._fosr_cached = True  # type: ignore[attr-defined]
            gdc = GDC(
                self_loop_weight=1,
                normalization_in="sym",
                normalization_out="sym",
                diffusion_kwargs=dict(method="ppr", alpha=0.15),
                sparsification_kwargs=dict(method="topk", k=64, dim=0),
                exact=True,
            )
            data = gdc(data)
        return super().forward(data)


# -----------------------------------------------------------------------------
# CurvAMP – minimal, self-contained implementation
# -----------------------------------------------------------------------------

class _SimpleCurvPair(PairNorm):
    """Curvature-conditioned PairNorm (scale depends on kappa)."""

    def forward(self, x: torch.Tensor, kappa: torch.Tensor):  # type: ignore[override]
        out = super().forward(x)
        scale = torch.sigmoid(-kappa).unsqueeze(1)
        return out * scale


class _CurvAMPConv(MessagePassing):
    """One CurvAMP message-passing layer (multi-scale + gate + micro-rewire)."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 3, rewired_ratio: float = 0.02):
        super().__init__(aggr="add")
        self.K = K
        self.rewired_ratio = rewired_ratio
        self.proj = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(K)])
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, K))
        self.norm = _SimpleCurvPair()

    # ---------------------------------------------------------------------
    @staticmethod
    def _deg(edge_index: torch.Tensor, n: int) -> torch.Tensor:
        row = edge_index[0]
        deg = torch.bincount(row, minlength=n).float()
        return deg

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # type: ignore[override]
        N = x.size(0)
        row = edge_index[0]
        # Build sparse adjacency once per forward pass
        A = torch.sparse_coo_tensor(
            edge_index, torch.ones(row.size(0), device=x.device), (N, N)
        )
        # Multi-scale bank (0..K hops)
        bank: List[torch.Tensor] = [x]
        for _ in range(1, self.K):
            bank.append(torch.sparse.mm(A, bank[-1]))
        bank = [proj(b) for proj, b in zip(self.proj, bank)]

        deg = self._deg(edge_index, N)
        kappa_edge = 1.0 / (deg[row] + 1e-6)  # very cheap curvature proxy
        kappa_node = torch.zeros(N, device=x.device).index_add_(0, row, kappa_edge)
        kappa_node = kappa_node / (deg + 1e-6)

        alpha = torch.softmax(self.gate(torch.stack([kappa_node, deg], 1)), -1)
        out = sum(alpha[:, k : k + 1] * bank[k] for k in range(self.K))

        # Online micro-rewiring (forward-only edges)
        m = int(self.rewired_ratio * row.numel())
        if m > 0:
            cand_u = torch.randint(0, N, (m * 2,), device=x.device)
            cand_v = torch.randint(0, N, (m * 2,), device=x.device)
            new_ei = torch.stack([cand_u, cand_v])
            edge_index = torch.cat([edge_index, new_ei], 1)

        out = self.norm(out, kappa_node)
        return out, edge_index


class CurvAMP(nn.Module):
    """Stack of *depth* CurvAMPConv layers."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int,
        K: int = 3,
        rewired_ratio: float = 0.02,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _CurvAMPConv(in_dim if i == 0 else hidden, hidden, K, rewired_ratio)
                for i in range(depth)
            ]
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x, data.edge_index
        feats: List[torch.Tensor] = []
        for conv in self.layers:
            x, edge_index = conv(x, edge_index)
            x = torch.relu(x)
            feats.append(x)
        return self.head(x), feats


# -----------------------------------------------------------------------------
# Public model factory
# -----------------------------------------------------------------------------

MODELS: Dict[str, nn.Module] = {
    "GCN": GCNStack,
    "PairNorm": PairNormGCN,
    "FoSR": FoSRGCN,
    "CurvAMP": CurvAMP,
}

# -----------------------------------------------------------------------------
# Optimisation utilities
# -----------------------------------------------------------------------------

def _clear_cached_adj(model: nn.Module) -> None:
    """Clear cached adjacency matrices inside GCNConv layers (if any)."""

    for module in model.modules():
        if isinstance(module, GCNConv):
            # PyG uses these private attributes for caching. They may not all be
            # present depending on the version, so we guard with hasattr.
            for attr in ("_cached_edge_index", "_cached_adj_t", "_cached_x"):
                if hasattr(module, attr):
                    setattr(module, attr, None)


def train_one(model: nn.Module, data, cfg: dict, device: torch.device):
    """Train *model* on *data* following hyper-parameters in *cfg*."""

    # Move tensors to the desired device *before* any forward pass so that
    # cached items (e.g. in GCNConv) are stored on the correct device.
    model.to(device)
    data = data.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_val, best_state = -1.0, None
    val_curve: List[float] = []

    for epoch in range(cfg["max_epochs"]):
        # -------- training step --------
        model.train()
        opt.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # -------- validation --------
        if epoch % 10 == 0 or epoch == cfg["max_epochs"] - 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                preds = logits.argmax(-1)
                val = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            val_curve.append(val)
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            # Early stopping
            if len(val_curve) > cfg["patience"] and best_val == max(val_curve[-cfg["patience"] :]):
                break

    # Load best weights
    assert best_state is not None, "Training failed to produce any validation score"
    model.load_state_dict(best_state)
    return model, best_val


def gradient_check(model: nn.Module, data):
    """Fail-fast tensor gradient sanity check (no silent collapse).

    This routine purposefully runs on CPU to be lightweight, *but* we must make
    sure we do not leave behind device-specific caches (e.g. inside GCNConv).
    Therefore we explicitly clear any such cached objects before returning so
    that subsequent training on GPU recomputes them on the correct device.
    """

    logits, _ = model(data)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    if total_grad < 1e-6:
        raise RuntimeError("❌ Gradients vanished (|grad|<1e-6) – aborting as per policy")

    # Remove potentially stale CPU-cached adjacency matrices so that later GPU
    # training does not run into cross-device errors.
    _clear_cached_adj(model)
