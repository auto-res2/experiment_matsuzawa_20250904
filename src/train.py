from __future__ import annotations

"""src/train.py
Model definitions and all training-related neural network components for the
FANS-GNN experimental suite.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, SAGEConv, PairNorm  # type: ignore
from torch_geometric.utils import degree

# ---------------------------------------------------------------------------
# 1.  Frequency–Adaptive Node-wise Smoothing Convolution (FANS-Conv)
# ---------------------------------------------------------------------------

class FANSConv(MessagePassing):
    """Frequency-Adaptive Node-wise Smoothing layer.

    h_{v}^{l+1} = (1-c_v)·H_v + c_v·L_v ,   c_v \in [0,1]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_hidden: int = 8,
        k_hop: int = 1,
        nyq_lambda: float = 0.05,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.lin_low = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_high = nn.Linear(in_channels, out_channels, bias=False)
        self.k_hop = k_hop
        self.nyq_lambda = nyq_lambda

        # tiny predictor g_θ : Dirichlet energy  ->  c_v
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1), nn.Sigmoid()
        )
        self.register_buffer("_dummy", torch.tensor(0.0))  # track device
        self.reg_loss: torch.Tensor = torch.tensor(0.0)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # high-pass residual (identity)
        H = self.lin_high(x)
        # low-pass message
        L = self.lin_low(self.propagate(edge_index, x=x, size=None))

        # local Dirichlet energy  E(v) = Σ_u ||h_v − h_u||²
        row, col = edge_index
        diff2 = (x[row] - x[col]).pow(2).sum(dim=1)
        E = torch.zeros(x.size(0), device=x.device).scatter_add_(0, row, diff2)
        c = self.mlp(E.view(-1, 1)).view(-1, 1)  # (N,1) in [0,1]

        out = (1.0 - c) * H + c * L

        # Nyquist safe cut-off  ĉ_v = 1/(1+deg_v)
        deg = degree(row, x.size(0), dtype=x.dtype).clamp(min=1.0)
        c_hat = 1.0 / (1.0 + deg.view(-1, 1))
        self.reg_loss = self.nyq_lambda * (c - c_hat).pow(2).mean()
        return out


# ---------------------------------------------------------------------------
# 2.  ContraNorm-D  (rank-preserving normalisation)
# ---------------------------------------------------------------------------

class ContraNormD(nn.Module):
    """Implementation of ContraNorm-D (ICLR-23)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        mean = x.mean(dim=0, keepdim=True)
        x = x - mean
        std = (x.pow(2).mean(dim=0, keepdim=True) + self.eps).sqrt()
        x = x / std
        return x * self.scale


# ---------------------------------------------------------------------------
# 3.  Minimal NDLS convolution (fallback)
# ---------------------------------------------------------------------------

class NDLSConv(MessagePassing):
    """Simplified NDLS layer:  h^{l+1} = (1−α_l)·I + α_l·A ."""

    def __init__(self, in_channels: int, out_channels: int, alpha: float = 0.2) -> None:  # noqa: D401,E501
        super().__init__(aggr="add", node_dim=0)
        self.alpha = alpha
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x
        out = self.propagate(edge_index, x=x, size=None)
        out = (1.0 - self.alpha) * identity + self.alpha * out
        return self.lin(out)


# ---------------------------------------------------------------------------
# 4.  Deep GNN factory that plugs different remedies
# ---------------------------------------------------------------------------

class DeepGNN(nn.Module):
    """General deep GNN backbone with optional remedy (PairNorm, FANS, …)."""

    def __init__(
        self,
        backbone: str,
        remedy: str,
        in_dim: int,
        hidden: int,
        out_dim: int,
        layers: int,
        fans_kwargs: Dict | None = None,
        ndls_alpha: float = 0.2,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        fans_kwargs = fans_kwargs or {}

        # select vanilla convolution
        _conv_dict = {
            "gcn": GCNConv,
            "sage": SAGEConv,
        }
        if backbone not in _conv_dict:
            raise ValueError(f"Backbone '{backbone}' is not supported.")
        VanillaConv = _conv_dict[backbone]

        self.layers: nn.ModuleList = nn.ModuleList()
        self.norms: nn.ModuleList = nn.ModuleList()
        self.remedy = remedy
        self.dropout_p = dropout_p

        for layer_idx in range(layers):
            in_c = in_dim if layer_idx == 0 else hidden
            out_c = out_dim if layer_idx == layers - 1 else hidden

            # choose convolution according to remedy
            if remedy == "fans":
                conv = FANSConv(in_c, out_c, **fans_kwargs)
            elif remedy == "ndls":
                conv = NDLSConv(in_c, out_c, alpha=ndls_alpha)
            else:  # vanilla, pairnorm, contranorm
                conv = VanillaConv(in_c, out_c)
            self.layers.append(conv)

            # choose normalisation layer
            if remedy == "pairnorm":
                self.norms.append(PairNorm())
            elif remedy in {"contranorm", "fans"}:
                self.norms.append(ContraNormD(out_c))
            else:
                self.norms.append(nn.Identity())

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # type: ignore[override]
        total_reg = 0.0
        for idx, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            if isinstance(conv, FANSConv):
                x = conv(x, edge_index)
                total_reg = total_reg + conv.reg_loss
            else:
                x = conv(x, edge_index)

            # final layer has no activation
            if idx < len(self.layers) - 1:
                x = F.relu(x)

            x = norm(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x, total_reg
