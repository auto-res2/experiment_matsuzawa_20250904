"""
src/train.py
---------------
Training-related utilities: CPQR replay buffer, strategy factory that plugs the
buffer into Avalanche Replay, model factory, random-seed and CUDA utilities.
This module is *import-only* – executed by src.main.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

# -----------------------------------------------------------------------------
#  Reproducibility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set Python / Numpy / Torch RNG state for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from contextlib import contextmanager

@contextmanager
def cuda_usage():
    """Context manager that prints the peak GPU memory allocated inside it."""
    if not torch.cuda.is_available():
        # CPU-only execution – silently yield
        yield
        return

    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        peak = torch.cuda.max_memory_allocated() - start_mem
        print(f"[GPU] peak_allocated(MiB)={peak / 1024 / 1024:.2f}")


# -----------------------------------------------------------------------------
#  CPQR BUFFER IMPLEMENTATION
# -----------------------------------------------------------------------------

@dataclass
class CPQRStats:
    items: int = 0
    bytes_per_sample: int = 0
    total_buffer: int = 0  # bytes


class CPQRBuffer:
    """Product-Quantised feature replay buffer as described in the paper."""

    def __init__(
        self,
        dim: int,
        M: int,
        codebook_size: int,
        max_bytes: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if dim % M != 0:
            raise ValueError("`dim` must be divisible by `M`.")
        self.dim, self.M, self.ks = dim, M, codebook_size
        self.subdim = dim // M
        self.device = device
        self.dtype = dtype

        # codebooks  [M, ks, subdim]
        self.codebooks = torch.randn(
            M, codebook_size, self.subdim, device=device, dtype=dtype
        )

        # reservoir storage – kept on CPU (uint8 tensors)
        self._indices: List[torch.ByteTensor] = []  # each element shape [M]
        self._labels: List[int] = []

        # reservoir sampling bookkeeping
        self.max_items: int = max_bytes // M  # 1 byte per sub-vector
        self._seen: int = 0  # total samples observed

    # ------------------------------------------------------  public interface
    def add_batch(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a mini-batch of feature vectors + labels using reservoir sampling."""
        feats = feats.detach().to(self.device, dtype=self.dtype)
        labels = labels.detach().cpu().tolist()
        for feat, lbl in zip(feats, labels):
            self._seen += 1
            if len(self._indices) < self.max_items:
                self._store(feat, lbl)
            elif random.random() < self.max_items / self._seen:
                # replace a random entry
                j = random.randrange(len(self._indices))
                self._store(feat, lbl, pos=j)

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return *up to* n decoded features + labels (tensors live on `device`)."""
        if len(self._indices) == 0:
            raise RuntimeError("CPQRBuffer is empty – cannot sample.")
        idx = random.sample(range(len(self._indices)), k=min(n, len(self._indices)))
        codes = torch.stack([self._indices[i] for i in idx]).to(self.device)
        labels = torch.tensor([self._labels[i] for i in idx], device=self.device)
        feats = self._decode_codes(codes)
        return feats, labels

    def stats(self) -> CPQRStats:
        bps = self.M  # one byte per sub-vector (ks == 256)
        total = len(self._indices) * bps
        return CPQRStats(len(self._indices), bps, total)

    # ------------------------------------------------------  internal helpers
    def _store(self, feat: torch.Tensor, label: int, pos: Optional[int] = None):
        codes = self._quantise(feat)  # ByteTensor [M]
        if pos is None:
            self._indices.append(codes.cpu())
            self._labels.append(label)
        else:
            self._indices[pos] = codes.cpu()
            self._labels[pos] = label

    def _quantise(self, feat: torch.Tensor) -> torch.ByteTensor:
        sub = feat.view(self.M, self.subdim)  # [M, subdim]
        # L2 distance to each codebook entry
        dist = (sub.unsqueeze(1) - self.codebooks).pow(2).sum(-1)  # [M, ks]
        codes = dist.argmin(-1).to(torch.uint8)  # [M]
        return codes

    def _decode_codes(self, codes: torch.ByteTensor) -> torch.Tensor:
        """Decode uint8 codes back to float16 features."""
        B = codes.size(0)
        idx = codes.long()
        # gather per sub-vector
        cb = self.codebooks[torch.arange(self.M).unsqueeze(0).repeat(B, 1), idx]
        return cb.reshape(B, -1).to(self.dtype)


# -----------------------------------------------------------------------------
#  STRATEGY FACTORY (wraps Avalanche Replay with CPQR integration)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#  Robust import for the `Replay` strategy and `EvaluationPlugin` across
#  Avalanche versions.  The package has reorganised its public API a few times
#  so we fall back to alternative locations when the preferred import fails.
# -----------------------------------------------------------------------------
try:
    from avalanche.training.strategies import Replay  # >=0.7 (old layout)
except ModuleNotFoundError:  # pragma: no cover – handled at runtime
    # Newer layout (observed in avalanche-lib v0.6.0+) where strategies are
    # re-exported directly under `avalanche.training`.
    from avalanche.training import Replay  # type: ignore

try:
    from avalanche.training.plugins import EvaluationPlugin
except ModuleNotFoundError:  # pragma: no cover
    # Fallback (older versions)
    from avalanche.evaluation.plugins import EvaluationPlugin  # type: ignore


def _logits_from_features(model: nn.Module, feats: torch.Tensor) -> torch.Tensor:
    """Utility that returns class logits from pre-computed *features*."""
    if hasattr(model, "fc"):
        return model.fc(feats)
    if hasattr(model, "classifier"):
        return model.classifier(feats)  # type: ignore[attr-defined]
    raise AttributeError("Model does not expose a classifier head named 'fc' or 'classifier'.")


def make_replay_strategy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mb_size: int,
    device: torch.device,
    buffer: CPQRBuffer,
    eval_plugin: EvaluationPlugin,
) -> Replay:
    """Return an Avalanche `Replay` strategy that uses `buffer` for experience replay."""

    if buffer is None:
        raise ValueError("`buffer` must be a CPQRBuffer instance.")

    class CPQRPlugin:
        """Tiny Avalanche plugin that stores & replays CPQR features."""

        def __init__(self, cpqr: CPQRBuffer, replay_size: int = 128):
            self.cpqr, self.replay_size = cpqr, replay_size

        # ----- called by Avalanche --------------------------------------------------
        def before_backward(self, strategy, **kwargs):  # noqa: D401, N802
            """(1) Store current mini-batch features; (2) add replay loss."""
            # 1) store features from current images
            with torch.no_grad():
                feats = strategy.model.get_features(strategy.mb_x)
            self.cpqr.add_batch(feats, strategy.mb_y)

            # 2) replay loss if buffer not empty
            if len(self.cpqr._indices) == 0:
                return
            rep_feats, rep_lbl = self.cpqr.sample(self.replay_size)
            rep_feats = rep_feats.to(next(strategy.model.parameters()).dtype)
            logits_rep = _logits_from_features(strategy.model, rep_feats)
            replay_loss = strategy.criterion(logits_rep, rep_lbl)
            # Combine with current loss
            strategy.loss = strategy.loss + replay_loss

    plugins = [CPQRPlugin(buffer)]
    return Replay(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        mem_size=1,  # unused but mandatory arg
        plugins=plugins,
        train_mb_size=mb_size,
        evaluator=eval_plugin,
        device=device,
    )


# -----------------------------------------------------------------------------
#  MODEL FACTORY  (adds `.get_features` hook needed by CPQR)
# -----------------------------------------------------------------------------


def build_model(dataset_key: str) -> nn.Module:
    """Return a backbone suitable for the dataset and expose `get_features`."""
    # Permuted-MNIST → small MLP
    if dataset_key.lower().startswith("permuted"):
        from avalanche.models import MLP

        model: nn.Module = MLP(num_classes=10, hidden_size=400)  # type: ignore[assignment]

        def _get_feats(x, m=model):  # noqa: E306
            return m.feature_extractor(x)  # type: ignore

        model.get_features = _get_feats  # type: ignore[attr-defined]
        return model

    # vision datasets → ResNet-18 w/ modified first & last layers
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    # lighten the stem for CIFAR-like images
    model.conv1.stride = (1, 1)
    model.maxpool = nn.Identity()
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, 100)  # will be adapted if needed by Avalanche

    def _get_feats(x, m=model):  # noqa: E306
        # Manually reproduce the forward pass up to (and incl.) the global pooling
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    model.get_features = _get_feats  # type: ignore[attr-defined]
    return model
