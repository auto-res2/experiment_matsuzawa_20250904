"""src/evaluate.py
Utility functions for model evaluation and visualisation.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
import dgl

from .train import AdaPropGCN, SGC  # required for isinstance checks

# ----------------------------------------------------------------------------
# 1)  METRICS ----------------------------------------------------------------
# ----------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    """Accuracy of predictions at given node indices."""
    return float((logits[idx].argmax(dim=1) == labels[idx]).float().mean().item())


# ----------------------------------------------------------------------------
# 2)  EVALUATION --------------------------------------------------------------
# ----------------------------------------------------------------------------

def eval_model(
    model,
    g: dgl.DGLGraph,
    feats: torch.Tensor,
    labels: torch.Tensor,
    split: Dict[str, torch.Tensor],
    cfg: Dict,
    sparse_powers: List[torch.Tensor] | None = None,
):
    model.eval()
    with torch.no_grad():
        if isinstance(model, AdaPropGCN):
            logits, _ = model(g, feats, sparse_powers)
        elif isinstance(model, SGC):
            logits = model(g, feats, sparse_powers)
        else:
            logits = model(g, feats)
    accs = {k: compute_accuracy(logits, labels, v) for k, v in split.items()}
    return accs, logits.detach()
