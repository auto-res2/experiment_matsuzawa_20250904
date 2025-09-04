"""
train.py – model definition and training utilities for CurvAMP
"""
from __future__ import annotations
import math, random, time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
# NOTE: We intentionally avoid importing torch_sparse / torch_scatter to
#       keep the dependency list lean and compilation-free.

# -----------------------------------------------------------------------------
# Utility helpers (self-contained, avoid external scatter dependencies)
# -----------------------------------------------------------------------------

def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Simple replacement for torch_scatter.scatter_mean for 1-D tensors.
    Args:
        src: Values to aggregate (E,)
        index: Indices indicating the target location of each *src* entry (E,)
        dim_size: Number of output rows (N)
    Returns:
        Tensor of shape (N,) where output[i] is the mean of src[j] for which
        index[j] == i. Empty rows get 0.
    """
    device = src.device
    sum_per_index = torch.zeros(dim_size, device=device).index_add_(0, index, src)
    count = torch.bincount(index, minlength=dim_size).clamp_min(1).to(src.dtype)
    return sum_per_index / count

# -----------------------------------------------------------------------------
# CurvAMP building blocks
# -----------------------------------------------------------------------------

def _approx_node_curvature(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Very fast proxy for Ollivier–Ricci curvature using degree difference."""
    row, col = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    curv = 1.0 / (deg[row] + 1e-6) + 1.0 / (deg[col] + 1e-6)
    return _scatter_mean(curv, row, num_nodes)


class CurvPair(nn.Module):
    """PairNorm scale factor modulated by node curvature."""

    def __init__(self, base_scale: float = 1.0):
        super().__init__()
        self.s0 = base_scale

    def forward(self, x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        col_mean = x.mean(dim=0, keepdim=True)
        x = x - col_mean
        row_norm = (x.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
        row_mean = row_norm.mean()
        s_i = self.s0 * torch.sigmoid(-kappa).unsqueeze(1)  # (N,1)
        return s_i * x / row_mean


class UniformityLoss(nn.Module):
    """Collapse-avoiding regulariser (similar to ContraNorm)."""

    def __init__(self, lam: float = 1e-2):
        super().__init__()
        self.lam = lam

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N,d)
        z = torch.nn.functional.normalize(z, dim=1)
        sim = torch.einsum("nd,md->nm", z, z)
        return self.lam * torch.exp(2 * sim).mean()


class CurvAMPConv(MessagePassing):
    """Single CurvAMP layer implementing multi-scale bank, curvature gate, online rewiring and CurvPair normalisation."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 3, rewired_ratio: float = 0.02):
        super().__init__(aggr="add")
        self.K, self.rewired_ratio = K, rewired_ratio
        self.lins = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(K)]
        )
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, K))
        self.norm = CurvPair()
        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):
        for l in self.lins:
            nn.init.xavier_uniform_(l.weight)
        for m in self.gate:  # type: ignore[assignment]
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: (updated features, possibly rewired edge_index)."""
        num_nodes = x.size(0)
        row, col = edge_index  # (E,)

        # 1) Build sparse adjacency once using torch native sparse COO
        values = torch.ones(row.size(0), device=x.device)
        A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

        # Multi-scale bank (cheap adjacency powers)
        bank = [x]
        h = x
        for _ in range(1, self.K):
            h = torch.sparse.mm(A, h)
            bank.append(h)
        bank_proj = [lin(hk) for lin, hk in zip(self.lins, bank)]

        # 2) curvature gate
        kappa = _approx_node_curvature(edge_index, num_nodes)
        deg = torch.bincount(row, minlength=num_nodes).float()
        gate_in = torch.stack([kappa, deg], dim=1)  # (N,2)
        alpha = torch.softmax(self.gate(gate_in), dim=-1)  # (N,K)
        out = sum(alpha[:, k : k + 1] * bank_proj[k] for k in range(self.K))

        # 3) online micro-rewiring (forward only)
        with torch.no_grad():
            curv_edges = -(kappa[row] + kappa[col])  # large negative curvature ⇒ candidate
            m = int(self.rewired_ratio * row.size(0))
            if m > 0:
                _, idx = torch.topk(curv_edges, k=m, largest=True)
                new_edges = torch.stack([row[idx], col[idx]], dim=0)
                edge_index = torch.cat([edge_index, new_edges], dim=1)

        # 4) CurvPair normalisation
        out = self.norm(out, kappa)
        return out, edge_index


# -----------------------------------------------------------------------------
# End-to-end model
# -----------------------------------------------------------------------------

class CurvAMPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int,
        K: int,
        rewired_ratio: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CurvAMPConv(
                    in_dim if i == 0 else hidden, hidden, K, rewired_ratio
                )
                for i in range(depth)
            ]
        )
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        feats: List[torch.Tensor] = []
        for conv in self.layers:
            x, ei = conv(x, ei)
            x = torch.nn.functional.gelu(x)
            feats.append(x)
        logits = self.head(x)
        return logits, feats


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def train_model(
    data, cfg: Dict[str, Any], device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains CurvAMP on a single split and returns the best model and logs."""

    in_dim = data.x.size(1)
    out_dim = int(data.y.max().item() + 1)

    model = CurvAMPNet(
        in_dim,
        cfg["hidden"],
        out_dim,
        cfg["depth"],
        cfg["K"],
        cfg["rewire_ratio"],
    ).to(device)

    uniformity = UniformityLoss(cfg["lambda_c"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=5e-4)

    best_state = None
    best_val = 0.0
    hist: List[float] = []

    for epoch in range(cfg["epochs"]):
        model.train()
        optimiser.zero_grad()
        logits, feats = model(data)
        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask]
        ) + uniformity(feats[-1])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        # --- validation ---
        if epoch % 20 == 0 or epoch == cfg["epochs"] - 1:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                val_acc = (
                    (logits[data.val_mask].argmax(dim=-1) == data.y[data.val_mask])
                ).float().mean().item()
            hist.append(val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            # early stopping
            if len(hist) > cfg["patience"] and best_val >= max(hist[-cfg["patience"] :]):
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": best_val, "val_curve": hist}
