"""src/train.py – model definitions and generic training utilities for AdaProp experiments"""
from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

################################################################################
#  Functional building blocks                                                   #
################################################################################

EPS = 1e-9

def _gumbel_softmax_sample(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Differentiable sampling with the Gumbel-Softmax trick."""
    g = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    return F.softmax((logits + g) / tau, dim=-1)


class _GateMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, K: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, K + 1)
        )

    def forward(self, x: torch.Tensor, tau: float):  # (N,F)
        logits = self.mlp(x)
        return _gumbel_softmax_sample(logits, tau)


class AdaPropConv(nn.Module):
    """One AdaProp layer (cf. paper Section 3)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        K: int = 10,
        hid_gate: int = 32,
        tau: float = 0.5,
        lambda_: float = 0.2,
    ):
        super().__init__()
        self.K, self.tau, self.lambda_ = K, tau, lambda_
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = _GateMLP(in_dim, hid_gate, K)
        nn.init.xavier_uniform_(self.lin.weight)

    # ---------------------------------------------------------------------
    #  Regulariser from Li et al. (2020) controlling mix ↔ denoise balance
    # ---------------------------------------------------------------------

    @staticmethod
    def _mix_denoise_regulariser(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.numel() == 0:
            return h.new_tensor(0.0)
        classes = y.unique()
        cls_means, intra_var = [], 0.0
        for c in classes:
            idx = (y == c).nonzero(as_tuple=False).squeeze()
            if idx.numel() < 2:
                continue
            h_c = h[idx]
            mu = h_c.mean(0, keepdim=True)
            cls_means.append(mu)
            intra_var += ((h_c - mu) ** 2).mean()
        if len(cls_means) < 2:
            return h.new_tensor(0.0)
        cls_means = torch.cat(cls_means, 0)
        diffs = cls_means.unsqueeze(0) - cls_means.unsqueeze(1)
        mu_inter = diffs.pow(2).mean().sqrt()
        return -mu_inter / (intra_var + EPS)  # negative ‑> minimise

    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        powers: List[torch.Tensor],
        y: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parameters
        ----------
        X       : (N,F) input feature matrix
        powers  : list of sparse tensors  Â^k produced by preprocess.cached_powers
        y       : (N,) labels, only needed to compute regulariser during training
        Returns
        -------
        h_out   : (N, D) output features after linear projection
        pi      : (N, K+1) learned stopping probabilities
        reg     : scalar regularisation loss (already multiplied by λ)
        """
        msgs = [torch.sparse.mm(powers[k], X) for k in range(self.K + 1)]
        H_stack = torch.stack(msgs, dim=1)  # (N, K+1, F)
        pi = self.gate(X, self.tau)         # (N, K+1)
        h_mix = (pi.unsqueeze(-1) * H_stack).sum(1)
        h_out = self.lin(h_mix)
        reg = (
            self._mix_denoise_regulariser(h_out, y) * self.lambda_
            if self.training and y is not None and self.lambda_ > 0
            else h_out.new_tensor(0.0)
        )
        return h_out, pi, reg

################################################################################
#  Complete network definitions                                                 #
################################################################################

class AdaPropGCN(nn.Module):
    """Stack of AdaPropConv + linear classifier."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        *,
        depth: int = 4,
        K: int = 10,
        lambda_: float = 0.2,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            AdaPropConv(in_dim, hidden, K=K, lambda_=lambda_, tau=tau)
        )
        for _ in range(depth - 2):
            self.layers.append(
                AdaPropConv(hidden, hidden, K=K, lambda_=lambda_, tau=tau)
            )
        self.layers.append(nn.Linear(hidden, num_classes))

    def forward(
        self,
        g,  # kept for API compatibility
        X: torch.Tensor,
        powers: List[torch.Tensor],
        y: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        h, pis, reg_sum = X, [], 0.0
        for layer in self.layers[:-1]:
            h, pi, reg = layer(h, powers, y)
            h = F.relu(h)
            pis.append(pi)
            reg_sum = reg_sum + reg
        logits = self.layers[-1](h)
        return logits, pis, reg_sum


class GCN(nn.Module):
    """Standard GCN baseline from Kipf & Welling (2017)."""

    def __init__(
        self, in_dim: int, hidden: int, num_classes: int, *, depth: int = 2, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_dim, hidden, activation=F.relu))
        for _ in range(depth - 2):
            self.layers.append(dglnn.GraphConv(hidden, hidden, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hidden, num_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, X, *args, **kwargs):
        h = X
        for layer in self.layers[:-1]:
            h = self.dropout(layer(g, h))
        return self.layers[-1](g, h)


class SGC(nn.Module):
    """Simplifying GCN."""

    def __init__(self, in_dim: int, num_classes: int, *, K: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        self.K = K

    def forward(self, g, X, powers: List[torch.Tensor], *args, **kwargs):
        return self.fc(torch.sparse.mm(powers[self.K], X))

################################################################################
#  Generic training loop                                                        #
################################################################################

def train_model(
    model: nn.Module,
    g,
    X: torch.Tensor,
    y: torch.Tensor,
    split: Dict[str, torch.Tensor],
    powers: List[torch.Tensor] | None,
    *,
    cfg_train: Dict[str, Any],
    device: torch.device,
) -> Tuple[float, float]:
    """Train *model* with early stopping.

    Returns
    -------
    (best_val_acc, best_test_acc)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["wd"])
    best_val, best_test, patience_cnt = 0.0, 0.0, 0

    for epoch in range(cfg_train["max_epochs"]):
        model.train()
        opt.zero_grad()
        logits, _, reg = (
            model(g, X, powers, y) if powers is not None else model(g, X)
        )
        loss = F.cross_entropy(logits[split["train"]], y[split["train"]]) + reg
        loss.backward()
        opt.step()

        # --- validation -------------------------------------------------
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits, *_ = model(g, X, powers) if powers is not None else model(g, X)
            val_acc = float((logits[split["val"]].argmax(1) == y[split["val"]]).float().mean().item())
            if val_acc > best_val:
                best_val = val_acc
                best_test = float((logits[split["test"]].argmax(1) == y[split["test"]]).float().mean().item())
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= cfg_train["patience"]:
                break
    return best_val, best_test
