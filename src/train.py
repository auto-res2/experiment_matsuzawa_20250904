"""src/train.py
Model architectures, normalisation layers, and the generic training
loop that is re-used by all experiments.
"""
from __future__ import annotations
import time
from typing import Callable, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# ------------------------------------------------------------------
#  Utility – reproducibility
# ------------------------------------------------------------------

def set_global_seeds(seed: int):
    """Fix every random generator we can reasonably access."""
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Python / hash randomness (important for some dataloader shuffles)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ------------------------------------------------------------------
#  Mix-prop helper (ĀH) – custom autograd to be memory-light
# ------------------------------------------------------------------
class _MixProp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float().clamp_(min=1)
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col])
        out = out / deg.unsqueeze(1)
        ctx.save_for_backward(edge_index, deg)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        edge_index, deg = ctx.saved_tensors
        row, col = edge_index
        grad_x = torch.zeros_like(grad_out)
        grad_x.index_add_(0, col, grad_out[row] / deg[row].unsqueeze(1))
        return grad_x, None, None

def propagate_mean(x: torch.Tensor, edge_index: torch.Tensor):
    return _MixProp.apply(x, edge_index, x.size(0))

# ------------------------------------------------------------------
#  FRODO-Norm layer
# ------------------------------------------------------------------
class FRODONorm(nn.Module):
    """Frequency-aware Regulators Of Dimensional- and Over-smoothing."""
    def __init__(self, dim: int, hyper):
        super().__init__()
        self.g = hyper.gamma
        self.sigma = hyper.sigma
        self.k = hyper.k
        self.tau_m = hyper.tau_m
        self.tau_r = hyper.tau_r
        self.register_buffer("running_r", torch.tensor(1.0))

    # ------------------------------------------------------------------
    #  Helper – Hutchinson effective rank estimate
    # ------------------------------------------------------------------
    def _hutchinson_erank(self, x: torch.Tensor, iters: int = 8):
        d = x.size(1)
        sketch = torch.randn(x.size(0), 1, device=x.device)
        for _ in range(iters):
            sketch = x @ (x.t() @ sketch) / d
        trace = (sketch * x).sum()  # noqa: F841 – kept for completeness
        s = torch.linalg.svdvals(x) ** 2
        shares = s / (s.sum() + 1e-9)
        erank = torch.exp(-(shares * torch.log(shares + 1e-9)).sum())
        return erank / d

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # pylint: disable=arguments-differ
        with torch.no_grad():
            Px = propagate_mean(x, edge_index)
            M = (Px - x).norm(dim=1, keepdim=True)  # mix-index
            idx = torch.randint(0, x.size(0), (min(2048, x.size(0)),), device=x.device)
            R_est = self._hutchinson_erank(x[idx])
            self.running_r = 0.95 * self.running_r + 0.05 * R_est.detach()
        alpha = torch.sigmoid(self.g * (M.mean() - M))
        condition = (M < self.tau_m) | (self.running_r < self.tau_r)
        x = torch.where(condition, x + alpha * (x - Px), x)

        # contrastive perturbation (only 5 % of nodes)
        if self.training:
            num = max(1, int(0.05 * x.size(0)))
            perm = torch.randperm(x.size(0), device=x.device)[:num]
            noise = torch.randn_like(x[perm]) * self.sigma
            cos = 1 - F.cosine_similarity(x[perm], x[perm] + noise).mean()
            if not hasattr(self, "extra_losses"):
                self.extra_losses = []
            self.extra_losses.append(cos)
        return x

# ------------------------------------------------------------------
#  Back-bone GNNs (GCN / GAT / APPNP)
# ------------------------------------------------------------------
class GCNNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, depth: int,
                 dropout: float, norm_factory: Callable[[], nn.Module]):
        super().__init__()
        self.depth = depth
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        for i in range(depth):
            in_c = in_dim if i == 0 else hid_dim
            out_c = out_dim if i == depth - 1 else hid_dim
            self.convs.append(GCNConv(in_c, out_c, add_self_loops=False, cached=False))
            if i < depth - 1:
                self.norms.append(norm_factory())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # pylint: disable=arguments-differ
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.depth - 1:
                norm = self.norms[i]
                x = norm(x, edge_index) if isinstance(norm, FRODONorm) else norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GATNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, depth: int,
                 heads: int, dropout: float, norm_factory: Callable[[], nn.Module]):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            in_c = in_dim if i == 0 else hid_dim * heads
            out_c = out_dim if i == depth - 1 else hid_dim
            num_heads = 1 if i == depth - 1 else heads
            self.convs.append(GATConv(in_c, out_c, heads=num_heads, add_self_loops=False, dropout=dropout))
            if i < depth - 1:
                self.norms.append(norm_factory())
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # pylint: disable=arguments-differ
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                norm = self.norms[i]
                x = norm(x, edge_index) if isinstance(norm, FRODONorm) else norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class APPNPNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, depth: int,
                 K: int, dropout: float, norm_factory: Callable[[], nn.Module]):
        super().__init__()
        self.lins, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            in_c = in_dim if i == 0 else hid_dim
            out_c = out_dim if i == depth - 1 else hid_dim
            self.lins.append(nn.Linear(in_c, out_c))
            if i < depth - 1:
                self.norms.append(norm_factory())
        self.propagate = APPNP(K, alpha=0.1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # pylint: disable=arguments-differ
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                norm = self.norms[i]
                x = norm(x, edge_index) if isinstance(norm, FRODONorm) else norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.propagate(x, edge_index)

# ------------------------------------------------------------------
#  Factory for normalisation layer (variant is a string flag)
# ------------------------------------------------------------------

def norm_factory(variant: str, dim: int, frodo_hyper):
    if variant == "frodo":
        return lambda: FRODONorm(dim, frodo_hyper)
    if variant == "pairnorm":
        from pairnorm import PairNorm   # type: ignore
        return lambda: PairNorm("PN")
    if variant == "contranorm":
        from contranorm import ContraNorm  # type: ignore
        return lambda: ContraNorm(dim)
    if variant == "dgn":
        from dgn import DGN  # type: ignore
        return lambda: DGN(dim, groups=16)
    # vanilla / dropedge / ndls fall back to Identity
    return lambda: nn.Identity()

# ------------------------------------------------------------------
#  Training routine  (shared by all experiments)
# ------------------------------------------------------------------
from .evaluate import accuracy, effective_rank, group_distance_ratio  # noqa: E402 – circular-safe

def run_training(model: nn.Module, data: Data, split_idx: Dict[str, torch.Tensor],
                 cfg, exp_name: str) -> Tuple[float, Dict[str, Any]]:
    """Generic full-batch training with early stopping."""
    device = torch.device(cfg.device)
    model = model.to(device)
    x, y = data.x.to(device), data.y.view(-1).to(device)
    edge_index = data.edge_index.to(device)

    train_idx = split_idx["train"].to(device)
    val_idx = split_idx["valid"].to(device)
    test_idx = split_idx["test"].to(device)

    optimiser = optim.Adam(model.parameters(), lr=cfg.lr_grid[1], weight_decay=cfg.wd_grid[2])
    best_val, wait, best_state, start = -1.0, 0, None, time.time()

    for epoch in range(1, 10000):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        out = model(x, edge_index)
        loss_main = F.nll_loss(F.log_softmax(out[train_idx], dim=-1), y[train_idx])
        extra = 0.0
        for m in model.modules():
            if hasattr(m, "extra_losses") and m.extra_losses:
                extra += sum(m.extra_losses)
                m.extra_losses.clear()
        loss = loss_main + extra
        loss.backward()
        optimiser.step()

        # validation
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            val_acc = accuracy(logits[val_idx], y[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break

    train_time = (time.time() - start) / max(epoch, 1)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    test_acc = accuracy(logits[test_idx], y[test_idx])

    metrics = {
        "val_acc": best_val,
        "test_acc": test_acc,
        "train_sec_per_epoch": train_time,
        "epochs": epoch,
        "ER": effective_rank(logits.detach()[:2048]),
        "GDR": group_distance_ratio(logits.detach()[:2048], y[:2048]),
    }
    print({"experiment": exp_name, **metrics})
    return test_acc, metrics
