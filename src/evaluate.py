"""
evaluate.py – generic metrics and aggregation utilities
------------------------------------------------------
Pure functions only – no side-effects, no heavy imports to avoid circular
references with train.py.
"""
from __future__ import annotations
from typing import List
import torch

# ---------------------------------------------------------------------------
#  Simple classification accuracy
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy expressed in percentage."""
    return (logits.argmax(dim=1) == target).float().mean().item() * 100.0

# ---------------------------------------------------------------------------
#  Average forgetting (Chaudhry et al.)
# ---------------------------------------------------------------------------

def compute_forgetting(acc_matrix: List[List[float]]) -> List[float]:
    """acc_matrix[t][k] = accuracy on task k after training task t (0-indexed)."""
    fgt = []
    for k in range(len(acc_matrix)):
        best = max(acc_matrix[t][k] for t in range(k, len(acc_matrix)))
        last = acc_matrix[-1][k]
        fgt.append(best - last)
    return fgt
