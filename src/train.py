"""src/train.py
Model definitions and training utilities for CAP-GNN experiments.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.data import Data

# local imports (relative – works with  `python -m src.main`)
from .evaluate import accuracy, row_diff

__all__ = [
    "ACTHalting",
    "CAPConv",
    "GCNStack",
    "train_one",
]

# ----------------------------------------------------------------------------
#  Adaptive Computation Time (per-node halting)
# ----------------------------------------------------------------------------
class ACTHalting(nn.Module):
    """Graves’ ACT mechanism, node-wise halting in GNNs.

    IMPORTANT MEMORY NOTE
    ---------------------
    The original implementation accumulated the computation graph of the
    *cum* and *ponder* statistics across **all** propagation layers.  With a
    depth of 16 this led to an exploding GPU memory footprint ( >12 GB for the
    tiny Cora graph) because every layer kept references to the full history
    of previous activations.

    To fix the resulting OOM we detach the running statistics from the graph
    after each update.  This preserves the mathematical behaviour of ACT (the
    values themselves are still carried over between layers) while preventing
    PyTorch from back-propagating through the entire history.
    """

    def __init__(self, in_dim: int, beta: float = 0.01, tau: float = 1.0):
        super().__init__()
        self.beta, self.tau = beta, tau
        self.halt_fc = nn.Linear(in_dim, 1)

    def forward(
        self,
        h_prev: torch.Tensor,
        h_new: torch.Tensor,
        state: Dict[str, Any],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        # Detach running statistics to avoid building a deep computation graph
        cum    = state["cum"].detach()
        ponder = state["ponder"].detach()

        p = torch.sigmoid(self.halt_fc(h_new))  # (N, 1)
        still_running = (cum < 1.0).float()
        add = (p * still_running).squeeze(-1)

        cum = cum + add
        ponder = ponder + still_running + self.beta * p.squeeze(-1)

        state["cum"], state["ponder"] = cum, ponder

        mask = (cum < 1.0).float().unsqueeze(-1)
        return mask * h_new + (1.0 - mask) * h_prev, state


# ----------------------------------------------------------------------------
#  Curvature-aware Adaptive Propagation convolution
# ----------------------------------------------------------------------------
class CAPConv(MessagePassing):
    """GCN-style layer with curvature gates + optional ACT."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kappa: torch.Tensor,
        act_cfg: Dict[str, float] | None = None,
    ):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.a1 = nn.Parameter(torch.randn(1) * 1e-1)
        self.a2 = nn.Parameter(torch.randn(1) * 1e-1)
        self.b = nn.Parameter(torch.zeros(1))
        # store curvature on correct device later via  register_buffer
        self.register_buffer("kappa", kappa)
        self.act = (
            ACTHalting(out_dim, beta=act_cfg["beta"], tau=act_cfg["tau"])
            if act_cfg
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        act_state: Dict[str, Any] | None = None,
    ):
        x_res = x
        x_lin = self.lin(x)

        # --- edge-wise similarity & gate -------------------------------------
        h_src, h_dst = x_lin[edge_index[0]], x_lin[edge_index[1]]
        sim = (h_src * h_dst).sum(dim=-1, keepdim=True)
        logits = self.a1 * self.kappa.unsqueeze(-1) + self.a2 * sim + self.b
        alpha = torch.sigmoid(logits).squeeze(-1)

        out = self.propagate(edge_index, x=x_lin, alpha=alpha)

        if self.act is not None:
            if act_state is None:
                raise ValueError("`act_state` dict must be supplied when ACT is enabled.")
            out, act_state = self.act(x_res, out, act_state)

        return out + x_res  # residual connection

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j


# ----------------------------------------------------------------------------
#  Stacked GCN / CAP-GNN network
# ----------------------------------------------------------------------------
class GCNStack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        depth: int,
        variant: str,
        kappa: torch.Tensor | None,
        act_cfg: Dict[str, float] | None,
    ):
        super().__init__()
        self.variant = variant
        self.depth = depth

        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        if variant == "cap":
            if kappa is None:
                raise ValueError("`kappa` tensor must be provided for CAP variant.")
            for _ in range(depth):
                self.convs.append(CAPConv(hidden, hidden, kappa, act_cfg))
        else:
            for _ in range(depth):
                self.convs.append(GCNConv(hidden, hidden))

        self.dropout = nn.Dropout(0.5)  # will be overwritten from config in main
        self.out_lin = nn.Linear(hidden, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.embed(x))

        if self.variant == "cap":
            act_state = {
                "cum": torch.zeros(x.size(0), device=x.device),
                "ponder": torch.zeros(x.size(0), device=x.device),
            }
            for conv in self.convs:
                x = conv(x, edge_index, act_state)
        else:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

        x = self.dropout(x)
        return self.out_lin(x)


# ----------------------------------------------------------------------------
#  Optimisation loop
# ----------------------------------------------------------------------------
@torch.no_grad()
def _evaluate(model: nn.Module, data: Data) -> Dict[str, float]:
    model.eval()
    logits = model(data)
    out = {
        "val": accuracy(logits[data.val_mask], data.y[data.val_mask]),
        "test": accuracy(logits[data.test_mask], data.y[data.test_mask]),
        "rowdiff": row_diff(logits),
    }
    return out, logits.detach().cpu()


def train_one(
    model: nn.Module,
    data: Data,
    epochs: int,
    patience: int,
    lr: float,
    wd: float,
    device: torch.device,
) -> Tuple[List[float], Dict[str, float], List[float]]:
    """Simple early-stopping training loop."""

    model.to(device)
    data = data.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad = 0.0, None, 0
    loss_curve: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimiser.step()
        loss_curve.append(loss.item())

        # --- validation ------------------------------------------------------
        metrics, _ = _evaluate(model, data)
        if metrics["val"] > best_val:
            best_val, best_state, bad = metrics["val"], model.state_dict(), 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, logits = _evaluate(model, data)
    return loss_curve, final_metrics, logits
