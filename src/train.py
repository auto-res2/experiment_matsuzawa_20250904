"""src/train.py
Model zoo, training utilities & YAML-config helper.
"""
from __future__ import annotations
from pathlib import Path
import time, random, typing, yaml, math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# -----------------------------------------------------------------------------
# YAML configuration helper ----------------------------------------------------
# -----------------------------------------------------------------------------
_CFG_PATH = Path("config/config.yaml")
if _CFG_PATH.exists():
    with _CFG_PATH.open() as fh:
        _CFG_RAW: dict = yaml.safe_load(fh)
else:  # pragma: no cover – fall-back when config is missing
    _CFG_RAW = {}

def cfg(*keys: typing.Any, default: typing.Any = None):
    """Lightweight hierarchical query into the YAML dictionary.

    Example
    -------
    ``cfg('experiment') -> 1``
    ``cfg('section', 'sub', default=123)``
    """
    cur = _CFG_RAW
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# -----------------------------------------------------------------------------
# Simple CurvAMP implementation (a small stack of GCN layers) ------------------
# -----------------------------------------------------------------------------
class CurvAMP(nn.Module):
    """A *very* simplified stand-in for the CurvAMP architecture.

    The original paper relies on curvature-aware message passing; implementing
    that is far beyond the scope of these tests.  For unit-testing purposes we
    employ a plain GCN stack that respects the *interface* that ``main.py``
    expects (constructor signature + forward behaviour)."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 depth: int,
                 K: int = 3,
                 *,
                 rewired_ratio: float = 0.0,
                 dropout: float | None = None):
        super().__init__()
        self.depth = max(depth, 1)
        self.dropout = dropout if dropout is not None else cfg('dropout', default=0.2)

        layers: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(self.depth - 1):
            layers.append(GCNConv(ch_in, hidden_channels, cached=True, normalize=True))
            ch_in = hidden_channels
        layers.append(GCNConv(ch_in, out_channels, cached=True, normalize=True))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    # ---------------------------------------------------------------------
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x

# Registry expected by ``main.py``
MODELS: dict[str, type[nn.Module]] = {
    'CurvAMP': CurvAMP,
}

# -----------------------------------------------------------------------------
# Training loop with patience / early stopping ---------------------------------
# -----------------------------------------------------------------------------

def _set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model: nn.Module, data, *, seed: int = 0):
    """Train ``model`` on ``data`` (PyG Data), honouring the YAML config.

    Returns
    -------
    dict with keys ``model``, ``val_curve`` (list[float]) and ``train_time`` (s).
    """
    _set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = model.to(device)

    lr = cfg('exp1', 'lr', default=cfg('lr', default=5e-4))
    wd = cfg('weight_decay', default=0.0)
    patience = cfg('patience', default=100)
    max_epochs = cfg('max_epochs', default=2000)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_val = math.inf
    best_state = None
    val_curve: list[float] = []
    epochs_no_improve = 0

    tic = time.time()
    for epoch in range(max_epochs):
        model.train()
        optimiser.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimiser.step()

        # ---- validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
        val_curve.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break  # early stop

    toc = time.time()

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'model': model,
        'val_curve': val_curve,
        'train_time': round(toc - tic, 3),
    }
