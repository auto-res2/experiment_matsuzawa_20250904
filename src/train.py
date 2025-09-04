"""src/train.py
Model architectures and the training step used across experiments.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 0)  DEPENDENCY HOT-FIX ------------------------------------------------------
# ---------------------------------------------------------------------------
# DGL (>=2.1) internally relies on several optional sub-modules from the
# `torchdata` package.  Unfortunately, many pre-built wheels of torchdata that
# accompany newer Python/PyTorch versions ship without *any* of those
# sub-packages, leading to an `ImportError` already at `import dgl` time.  To
# keep the project self-contained (and to avoid heavyweight or version-locked
# dependencies) we create a *minimal* run-time stub that satisfies **all**
# import statements that DGL performs.  No actual functionality is required –
# the referenced symbols are used merely for static typing or in code paths
# that are executed only for distributed training, which is well outside the
# scope of this repository.

import sys
import types

# Utility --------------------------------------------------------------------

def _install_stub(module_name: str) -> types.ModuleType:
    """Ensure that *module_name* is import-able by registering a stub module.

    The function walks the fully-qualified module path (``a.b.c``) and makes
    sure that every parent package exists inside ``sys.modules``.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]  # pragma: no cover – already present

    parts = module_name.split(".")
    for idx in range(1, len(parts) + 1):
        sub_path = ".".join(parts[:idx])
        if sub_path not in sys.modules:
            sys.modules[sub_path] = types.ModuleType(sub_path)
    return sys.modules[module_name]


# 1)  Stub for ``torchdata.datapipes.iter`` ----------------------------------
try:
    from torchdata.datapipes.iter import IterDataPipe  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover – executed only in broken envs
    iter_mod = _install_stub("torchdata.datapipes.iter")

    class _IterDataPipe:  # minimal stand-in (no functionality required)
        pass

    iter_mod.IterDataPipe = _IterDataPipe

# 2)  Stub for ``torchdata.dataloader2.graph`` --------------------------------
# DGL ≥ 2.1 also tries to ``import torchdata.dataloader2.graph as dp_utils``.
# We expose a dummy sub-module whose *attributes* silently evaluate to no-op
# callables so that *any* access pattern remains safe.

def _make_no_op():
    def _fn(*_args, **_kwargs):  # noqa: D401, ANN001
        """No-op stub injected because full *torchdata* is absent."""
        return None

    return _fn

try:
    import torchdata.dataloader2.graph as _unused  # noqa: F401  # pylint: disable=import-error
except ModuleNotFoundError:  # pragma: no cover – executed only in broken envs
    graph_mod = _install_stub("torchdata.dataloader2.graph")

    class _GraphStub(types.ModuleType):
        def __getattr__(self, item):  # noqa: D401, ANN001
            return _make_no_op()

    # Replace plain ModuleType with fancy stub providing dynamic attributes
    sys.modules["torchdata.dataloader2.graph"] = _GraphStub("torchdata.dataloader2.graph")

# ---------------------------------------------------------------------------
# Standard library & third-party imports (after the patch) -------------------
# ---------------------------------------------------------------------------
from typing import List, Dict, Any  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import Adam  # noqa: E402
from torch.cuda.amp import GradScaler, autocast  # noqa: E402
import dgl  # noqa: E402

# ----------------------------------------------------------------------------
# 1)  MODEL COMPONENTS --------------------------------------------------------
# ----------------------------------------------------------------------------


class GateMLP(nn.Module):
    """Tiny MLP that outputs (K+1) logits per node."""

    def __init__(self, in_dim: int, hidden: int, K: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, in_dim) → (N, K+1)
        return self.mlp(x)


class AdaPropConv(nn.Module):
    """Adaptive propagation layer (see project description)."""

    def __init__(self, in_dim: int, out_dim: int, K: int = 10, lambda_reg: float = 0.2):
        super().__init__()
        self.K = K
        self.lambda_reg = lambda_reg
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = GateMLP(out_dim, hidden=32, K=K)
        self.temperature: float = 0.5  # Gumbel-Softmax temperature
        # cache containers filled during forward – used for analyses
        self.reg_loss: torch.Tensor | None = None
        self.pi_cache: torch.Tensor | None = None

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, sparse_powers: List[torch.Tensor]):
        # x: (N, in_dim)
        h = self.linear(x)  # (N, out_dim)
        logits = self.gate(h)  # (N, K+1)
        pi = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)  # (N,K+1)

        # aggregate K-hop messages
        agg: List[torch.Tensor] = []
        for k in range(self.K + 1):
            msg = torch.sparse.mm(sparse_powers[k], x)  # (N, in_dim)
            msg = self.linear(msg)
            agg.append(msg.unsqueeze(2))
        agg = torch.cat(agg, dim=2)  # (N, out_dim, K+1)
        out = (agg * pi.unsqueeze(1)).sum(dim=2)  # (N, out_dim)

        # ---------------- Regularizer --------------------
        if self.training and self.lambda_reg > 0.0:
            labels = g.ndata["label"]
            uniq = labels.unique()
            class_means, class_vars = [], []
            for c in uniq:
                m = h[labels == c].mean(dim=0)
                v = h[labels == c].var(dim=0, unbiased=False).mean()
                class_means.append(m)
                class_vars.append(v)
            mu_inter = torch.stack(class_means).var(dim=0, unbiased=False).mean()
            sigma_intra = torch.stack(class_vars).mean()
            self.reg_loss = self.lambda_reg * (mu_inter / (sigma_intra + 1e-6))
        else:
            self.reg_loss = torch.tensor(0.0, device=x.device)
        self.pi_cache = pi.detach()
        return out


class AdaPropGCN(nn.Module):
    """Stack of AdaPropConv layers with optional ReLU in between."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, depth: int, K: int, lambda_reg: float):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(
                AdaPropConv(
                    in_dim if i == 0 else hidden,
                    hidden if i < depth - 1 else num_classes,
                    K=K,
                    lambda_reg=lambda_reg,
                )
            )
            if i < depth - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, sparse_powers: List[torch.Tensor]):
        reg_loss_total = 0.0
        for mod in self.net:
            if isinstance(mod, AdaPropConv):
                x = mod(g, x, sparse_powers)
                reg_loss_total = reg_loss_total + mod.reg_loss
            else:  # activation
                x = mod(x)
        return x, reg_loss_total


# ----------------------- Baseline models ------------------------------------
class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.GraphConv(in_dim, hidden, activation=F.relu))
        for _ in range(depth - 2):
            self.layers.append(dgl.nn.GraphConv(hidden, hidden, activation=F.relu))
        self.layers.append(dgl.nn.GraphConv(hidden, num_classes))

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        for layer in self.layers:
            x = layer(g, x)
        return x


class SGC(nn.Module):
    """Simple Graph Convolution with collapsed K-hop propagation."""

    def __init__(self, in_dim: int, num_classes: int, K: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, bias=False)
        self.K = K

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, precomputed: torch.Tensor | None):
        if precomputed is None:
            with torch.no_grad():
                for _ in range(self.K):
                    x = dgl.ops.copy_u_sum(g, x) / g.in_degrees().clamp(min=1).unsqueeze(1)
        else:
            x = precomputed
        return self.linear(x)

# ----------------------------------------------------------------------------
# 2)  TRAINING LOOP -----------------------------------------------------------
# ----------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    g: dgl.DGLGraph,
    feats: torch.Tensor,
    labels: torch.Tensor,
    split: Dict[str, torch.Tensor],
    optimizer: Adam,
    scaler: GradScaler,
    cfg: Dict[str, Any],
    sparse_powers: List[torch.Tensor] | None = None,
):
    """One full optimisation step (full-batch for all experiments in this repo)."""

    model.train()
    optimizer.zero_grad()
    with autocast(enabled=cfg["mixed_precision"]):
        if isinstance(model, AdaPropGCN):
            logits, reg_loss = model(g, feats, sparse_powers)
            loss = F.cross_entropy(logits[split["train"]], labels[split["train"]]) + reg_loss
        elif isinstance(model, SGC):
            logits = model(g, feats, sparse_powers)
            loss = F.cross_entropy(logits[split["train"]], labels[split["train"]])
        else:
            logits = model(g, feats)
            loss = F.cross_entropy(logits[split["train"]], labels[split["train"]])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())
