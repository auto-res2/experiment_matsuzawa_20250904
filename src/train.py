"""
train.py – model architectures, training loop and utility functions.
"""
from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GCNConv

# -----------------------------------------------------------------------------
#  Safety-check – we insist on running with a visible CUDA device.
# -----------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not visible – experiments require a GPU.")

# -----------------------------------------------------------------------------
#  Reproducibility helper
# -----------------------------------------------------------------------------

def set_global_seeds(seed: int):
    """Fix Python / NumPy / PyTorch RNG state for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
#  Configuration dataclasses (filled by src.main after reading YAML)
# -----------------------------------------------------------------------------

@dataclass
class GlobalCfg:
    device: str = "cuda:0"
    data_root: str = "data"
    hidden: int = 128
    dropout: float = 0.5
    patience: int = 50
    lr_grid: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02])
    wd_grid: List[float] = field(default_factory=lambda: [0.0, 1e-4, 5e-4])
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    datasets_e1: List[str] = field(default_factory=list)
    depths_e1: List[int] = field(default_factory=list)
    variants_e1: List[str] = field(default_factory=list)


@dataclass
class FrodoHyper:
    gamma: float = 5.0
    sigma: float = 0.1
    k: float = 0.15
    tau_m: float = 0.02
    tau_r: float = 0.35

# -----------------------------------------------------------------------------
#  Evaluation metrics  (kept here to avoid circular imports with evaluate.py)
# -----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


def effective_rank(x: torch.Tensor) -> float:
    if x.size(0) > 4096:  # memory-safety sampling
        idx = torch.randperm(x.size(0), device=x.device)[:4096]
        x = x[idx]
    with torch.no_grad():
        s = torch.linalg.svdvals(x.cpu())
        p = (s ** 2) / (s ** 2).sum()
        er = torch.exp(-(p * torch.log(p + 1e-9)).sum())
        return float(er / x.size(1))


def group_distance_ratio(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.size(0) > 2048:
        idx = torch.randperm(x.size(0), device=x.device)[:2048]
        x, y = x[idx].cpu(), y[idx].cpu()
    with torch.no_grad():
        dists = torch.cdist(x, x, p=2)
        same = y.unsqueeze(0) == y.unsqueeze(1)
        same.fill_diagonal_(False)
        diff = ~same
        return float((dists[same].mean() - dists[diff].mean()).abs() /
                     (dists[same].mean() + dists[diff].mean() + 1e-9))


def instance_information_gain(z_in: torch.Tensor, z_out: torch.Tensor) -> float:
    p = F.softmax(z_in, dim=-1)
    q = F.softmax(z_out, dim=-1)
    kl = F.kl_div(q.log(), p, reduction="batchmean")
    return float(kl)

# -----------------------------------------------------------------------------
#  Normalisation layers
# -----------------------------------------------------------------------------

class PairNorm(nn.Module):
    """PairNorm – PN-scale variant."""
    def forward(self, x: torch.Tensor, *_):  # type: ignore[override]
        mean = x.mean(dim=0, keepdim=True)
        x = x - mean
        norm = x.pow(2).sum(dim=1, keepdim=True).mean().sqrt()
        return x / (norm + 1e-6)


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor, *_):  # type: ignore[override]
        return x


# Hutchinson-trace helper ----------------------------------------------------------------
@torch.no_grad()
def _hutchinson_erank(x: torch.Tensor, iters: int = 8) -> torch.Tensor:
    n, d = x.shape
    v = torch.randn(n, 1, device=x.device)
    for _ in range(iters):
        v = x @ (x.T @ v) / d
    s = torch.linalg.svdvals(x)[:min(d, 64)]
    p = (s ** 2) / (s ** 2).sum()
    er = torch.exp(-(p * torch.log(p + 1e-9)).sum())
    return er / d


class FRODONorm(nn.Module):
    """Implementation of FRODO-Norm layer."""
    def __init__(self, dim: int, hyper: FrodoHyper):
        super().__init__()
        self.g = hyper.gamma
        self.sigma = hyper.sigma
        self.k = hyper.k
        self.tau_m = hyper.tau_m
        self.tau_r = hyper.tau_r
        self.register_buffer("running_r", torch.tensor(1.0))
        self.loss_contrastive = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):  # type: ignore[override]
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).float().to(x.device)
        Px = torch.zeros_like(x)
        Px.index_add_(0, row, x[col])
        Px = Px / deg.unsqueeze(1)

        M = (Px - x).norm(dim=1, keepdim=True)
        sample = torch.randperm(x.size(0), device=x.device)[:2048]
        R = _hutchinson_erank(x[sample])
        self.running_r.mul_(0.95).add_(0.05 * R)

        alpha = torch.sigmoid(self.g * (M.mean() - M))
        cond = (M < self.tau_m) | (self.running_r < self.tau_r)
        x = torch.where(cond, x + alpha * (x - Px), x)

        if self.training and self.sigma > 0:
            m = max(1, int(0.05 * x.size(0)))
            idx = torch.randperm(x.size(0), device=x.device)[:m]
            noise = torch.randn_like(x[idx]) * self.sigma
            self.loss_contrastive = 1 - F.cosine_similarity(x[idx], x[idx] + noise).mean()
        else:
            self.loss_contrastive = torch.tensor(0.0, device=x.device)
        return x

# -----------------------------------------------------------------------------
#  GCN backbone
# -----------------------------------------------------------------------------

class GCN(nn.Module):
    def __init__(self, in_dim: int, hid: int, out_dim: int, depth: int, dropout: float,
                 norm_factory: Callable[[], nn.Module]):
        super().__init__()
        self.depth = depth
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(depth):
            inp = in_dim if i == 0 else hid
            outp = out_dim if i == depth - 1 else hid
            self.convs.append(GCNConv(inp, outp, add_self_loops=False, cached=False, normalize=True))
            if i < depth - 1:
                self.norms.append(norm_factory())
        self.dropout = dropout

    def forward(self, x, edge_index):  # type: ignore[override]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.depth - 1:
                norm = self.norms[i]
                x = norm(x, edge_index) if isinstance(norm, FRODONorm) else norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# -----------------------------------------------------------------------------
#  Variant → normalisation layer factory helper
# -----------------------------------------------------------------------------

def norm_factory(variant: str, dim: int, frodo_hyper: FrodoHyper) -> Callable[[], nn.Module]:
    variant = variant.lower()
    if variant == "frodo":
        return lambda: FRODONorm(dim, frodo_hyper)
    if variant == "pairnorm":
        return lambda: PairNorm()
    if variant in {"vanilla", "dropedge"}:
        return lambda: IdentityNorm()
    warnings.warn(f"Variant {variant} not fully implemented; defaulting to IdentityNorm().")
    return lambda: IdentityNorm()

# -----------------------------------------------------------------------------
#  Full-batch training routine
# -----------------------------------------------------------------------------

def train_one(model: nn.Module, data, split: Dict[str, torch.Tensor], cfg: GlobalCfg):
    device = torch.device(cfg.device)
    model = model.to(device)
    x, y, edge_index = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    train_idx, val_idx, test_idx = split["train"].to(device), split["valid"].to(device), split["test"].to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr_grid[1], weight_decay=cfg.wd_grid[2])
    best_val, best_state, patience = -1.0, None, 0
    tic = time.time()

    for epoch in range(1, 10000):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        #  add contrastive term if provided by FRODO layers
        c_loss = sum(getattr(m, "loss_contrastive", 0.0) for m in model.modules())
        loss = loss + c_loss
        loss.backward()
        opt.step()

        # ---------------- validation ----------------
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
        val_acc = accuracy(logits[val_idx], y[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break

    train_time = (time.time() - tic) / epoch
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)

    metrics = {
        "test_acc": accuracy(logits[test_idx], y[test_idx]),
        "val_best": best_val,
        "epochs": epoch,
        "sec_per_ep": train_time,
        "ER": effective_rank(logits),
        "GDR": group_distance_ratio(logits, y),
    }
    return metrics
