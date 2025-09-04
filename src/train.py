"""src/train.py
Core model architectures and training-time utilities.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.nn import GCNConv

# -----------------------------------------------------------------------------
#                                PairNorm
# -----------------------------------------------------------------------------
class PairNorm(nn.Module):
    """Implementation of PairNorm (Zhao & Akoglu, 2020)."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.s = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        col_mean = x.mean(dim=0, keepdim=True)
        x = x - col_mean
        row_norm = (x.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
        row_mean = row_norm.mean()
        x = self.s * x / row_mean
        return x

# -----------------------------------------------------------------------------
#                                Deep GCN
# -----------------------------------------------------------------------------
class DeepGCN(nn.Module):
    """Deep vanilla-GCN stack with optional PairNorm."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden: int,
        depth: int,
        dropout_in: float = 0.2,
        dropout_h: float = 0.5,
        use_pairnorm: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.use_pairnorm = use_pairnorm
        self.pn = PairNorm() if use_pairnorm else None
        self.dropout_in = nn.Dropout(dropout_in)
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_dim = num_features
        for _ in range(depth):
            self.convs.append(GCNConv(in_dim, hidden, add_self_loops=False, normalize=True))
            self.activations.append(nn.GELU())
            self.dropouts.append(nn.Dropout(dropout_h))
            in_dim = hidden
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):  # noqa: D401
        x, edge_index = data.x, data.edge_index
        x = self.dropout_in(x)
        per_layer_feats: List[torch.Tensor] = []
        for l in range(self.depth):
            x = self.convs[l](x, edge_index)
            if self.use_pairnorm:
                x = self.pn(x)
            x = self.activations[l](x)
            x = self.dropouts[l](x)
            per_layer_feats.append(x)
        out = self.lin(x)
        return out, per_layer_feats

# -----------------------------------------------------------------------------
#                              Train loop
# -----------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    data,
    optimizer: Optimizer,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Run one optimisation step with optional AMP."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        out, _ = model(data)
        loss = nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.item()
