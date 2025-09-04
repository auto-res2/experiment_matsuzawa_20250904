"""
train.py – model definitions and training loop
"""
from __future__ import annotations
import time, math, random, json, os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

from .evaluate import (
    row_diff,
    instance_information_gain,
    gpu_mem,
    save_lineplot,
)

# ----------------------------------------------------------------------------
#  Constants – all experiment figures MUST live in this folder (see task prompt)
# ----------------------------------------------------------------------------
IMAGE_DIR = Path(".research/iteration4/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------
#  Model building blocks
# ----------------------------------------------------------------------------------
class ACTHalting(nn.Module):
    """Node–wise Adaptive Computation Time (Graves, 2016)."""

    def __init__(self, hidden_dim: int, beta: float = 0.01, tau: float = 1.0):
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.h = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.h.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.sigmoid(self.h(x) / self.tau)  # (N,1)
        return p.squeeze(-1)


class CAPConv(nn.Module):
    """Curvature–aware Adaptive Propagation wrapper for PyG message-passing layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_curvatures: torch.Tensor,
        base_layer: str = "gcn",
        act_params: Dict | None = None,
        a1_init: float = 0.0,
        a2_init: float = 0.0,
        b_init: float = 0.0,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv, GINConv

        if base_layer == "gcn":
            self.base = GCNConv(in_channels, out_channels, add_self_loops=False, bias=False)
        elif base_layer == "gin":
            nn1 = nn.Sequential(
                nn.Linear(in_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)
            )
            self.base = GINConv(nn1)
        else:
            raise ValueError(f"Unsupported base_layer: {base_layer}")

        # edge-gating parameters
        self.a1 = nn.Parameter(torch.tensor(a1_init))
        self.a2 = nn.Parameter(torch.tensor(a2_init))
        self.b = nn.Parameter(torch.tensor(b_init))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("kappa", edge_curvatures)  # [E]

        self.use_act = act_params is not None
        self.act = ACTHalting(out_channels, **act_params) if self.use_act else None

    @torch.jit.ignore
    def _node_alpha(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index  # u -> v
        sim = (x[row] * x[col]).sum(-1)
        alpha = self.sigmoid(self.a1 * self.kappa + self.a2 * sim + self.b)  # (E,)
        deg = torch.zeros(x.size(0), device=x.device).index_add_(0, row, alpha).clamp_(min=1e-6)
        node_alpha = torch.zeros_like(deg).index_add_(0, row, alpha) / deg
        return node_alpha  # (N,)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, halting_state: dict | None):
        base_out = self.base(x, edge_index)
        node_alpha = self._node_alpha(x, edge_index)
        out = x + base_out * node_alpha.unsqueeze(-1)  # residual

        ponder_cost = None
        if self.use_act and halting_state is not None:
            p_i = self.act(out)
            still_running = (halting_state["halt"] < 1.0).float()
            halting_state["halt"] = (halting_state["halt"] + p_i * still_running).clamp_(max=1.0)
            halting_state["steps"] += still_running
            ponder_cost = (p_i * still_running).mean() * self.act.beta
        return out, ponder_cost


# ----------------------------------------------------------------------------------
#  Deep backbone encompassing different variants (baseline / CAP / etc.)
# ----------------------------------------------------------------------------------
class DeepBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        variant: str,
        edge_kappa: torch.Tensor,
        base: str = "gcn",
        act_params: Dict | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()

        from torch_geometric.nn import (
            GCNConv,
            GINConv,
            DropEdge,
            PairNorm,
        )

        for _ in range(num_layers):
            if variant == "vanilla":
                if base == "gcn":
                    self.layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
                else:
                    nn1 = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.layers.append(GINConv(nn1))
            elif variant == "dropedge":
                # We wrap the two-step procedure (edge-drop followed by GCN) into
                # a small callable to keep the forward logic simple.
                class _DropEdgeGCN(nn.Module):
                    def __init__(self, p: float, hidden_dim: int):
                        super().__init__()
                        self.de = DropEdge(p=p)
                        self.gcn = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)

                    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
                        ei = self.de(edge_index)
                        return self.gcn(x, ei)

                self.layers.append(_DropEdgeGCN(p=0.5, hidden_dim=hidden_dim))
            elif variant == "dgn":
                class _GCNPairNorm(nn.Module):
                    def __init__(self, hidden_dim: int):
                        super().__init__()
                        self.gcn = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
                        self.pn = PairNorm(scale=1.0)

                    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
                        return self.pn(self.gcn(x, edge_index))

                self.layers.append(_GCNPairNorm(hidden_dim))
            elif variant == "cap":
                self.layers.append(
                    CAPConv(
                        hidden_dim,
                        hidden_dim,
                        edge_kappa,
                        base_layer=base,
                        act_params=act_params,
                    )
                )
            else:
                raise ValueError(variant)
        self.output_lin = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.act(self.input_lin(x))
        halting, ponder_loss = None, 0.0
        for layer in self.layers:
            if self.variant == "cap":
                if halting is None:
                    halting = {
                        "halt": torch.zeros(x.size(0), device=x.device),
                        "steps": torch.zeros(x.size(0), device=x.device),
                    }
                x, p_cost = layer(x, edge_index, halting)
                ponder_loss += p_cost if p_cost is not None else 0.0
            else:
                x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        logits = self.output_lin(x)
        return logits, ponder_loss


# ----------------------------------------------------------------------------------
#  Public training API – orchestrates one dataset/depth/variant grid
# ----------------------------------------------------------------------------------
@torch.no_grad()
def _eval_split(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred[mask] == y[mask]).float().mean().item()


def train_on_graph(dataset_name: str, data, cfg: dict, results: list) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_kappa = cfg["edge_kappa"].to(device)
    data = data.to(device)

    for depth in cfg["depths"]:
        for variant in cfg["variants"]:
            tag = f"{dataset_name}_{variant}_{depth}L"
            # All figures saved under the mandatory IMAGE_DIR
            fig_path = IMAGE_DIR / f"rowdiff_{tag}.pdf"
            fig_path.parent.mkdir(parents=True, exist_ok=True)

            metrics_epoch = {"epoch": [], "val_acc": [], "row_diff": [], "iig": []}

            best_val_acc = 0.0
            for seed in cfg["seeds"]:
                _set_seed(seed)
                model = DeepBackbone(
                    in_dim=data.num_features,
                    hidden_dim=64,
                    out_dim=_num_classes(dataset_name),
                    num_layers=depth,
                    variant=variant,
                    edge_kappa=edge_kappa,
                    base="gcn",
                    act_params=cfg.get("act_params"),
                ).to(device)

                optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

                patience = 0
                for epoch in range(1, cfg["max_epochs"] + 1):
                    model.train(); optimiser.zero_grad()
                    with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
                        logits, p_loss = model(data.x, data.edge_index)
                        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]) + p_loss
                    scaler.scale(loss).backward(); scaler.step(optimiser); scaler.update()

                    if epoch % cfg["log_every"] == 0 or epoch == cfg["max_epochs"]:
                        model.eval()
                        logits, _ = model(data.x, data.edge_index)
                        val_acc = _eval_split(logits, data.y, data.val_mask)
                        rd = row_diff(logits.detach().float())
                        iig = instance_information_gain(logits.detach().float())
                        metrics_epoch["epoch"].append(epoch)
                        metrics_epoch["val_acc"].append(val_acc)
                        metrics_epoch["row_diff"].append(rd)
                        metrics_epoch["iig"].append(iig)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc; patience = 0
                        else:
                            patience += 1
                        if patience > cfg["patience"]:
                            break

                # ---- log per-seed result ----
                results.append(
                    {
                        "dataset": dataset_name,
                        "variant": variant,
                        "depth": depth,
                        "seed": seed,
                        "best_val_acc": best_val_acc,
                        "gpu_mem": gpu_mem(),
                    }
                )

            # ---- save learning curves (row-diff) ----
            save_lineplot(metrics_epoch["epoch"], metrics_epoch["row_diff"], "Epoch", "Row-Diff", tag, fig_path)
    return results


# ----------------------------------------------------------------------------------
#  Utility helpers
# ----------------------------------------------------------------------------------

def _num_classes(name: str) -> int:
    mapping = {
        "cora": 7,
        "citeseer": 6,
        "chameleon": 5,
        "squirrel": 5,
        "ogbn-arxiv": 40,
    }
    return mapping[name.lower()]


def _set_seed(seed: int):
    import numpy as np
    import random, os, torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
