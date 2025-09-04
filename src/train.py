"""src/train.py
Module containing all model- and training-related code.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# Random Fourier Perturbation
# ------------------------------------------------------------------
class FourierPerturb(nn.Module):
    """Applies a mild Fourier domain phase / amplitude jitter.

    This is a direct copy of the logic from the original experimental
    script but without runtime FFT shape errors.  The input is expected
    to be in NCHW format with values in the typical [0, 1] or
    [-1, 1] range.
    """

    def __init__(self, phase_max: float = math.pi / 5,
                 amp_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        super().__init__()
        self.phase_max = float(phase_max)
        self.amp_lo, self.amp_hi = amp_range

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        # FFT and randomise phase/amplitude
        x_fft = torch.fft.rfft2(x, norm="ortho")
        phase = torch.angle(x_fft)
        mag = torch.abs(x_fft)

        phase = phase + (torch.rand_like(phase) * 2 - 1) * self.phase_max
        mag = mag * (torch.rand_like(mag) * (self.amp_hi - self.amp_lo) + self.amp_lo)

        x_new = torch.fft.irfft2(mag * torch.exp(1j * phase),
                                 s=x.shape[-2:], norm="ortho")
        return x_new.clamp(x.min(), x.max())


# ------------------------------------------------------------------
# Dummy mask-bank (placeholder for SAM)
# ------------------------------------------------------------------
class _RandMaskBank:
    """Returns a binary mask whose left half is 1 (object) and right half 0 (context)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        mask = torch.zeros_like(x)
        mask[..., :, : x.size(-1) // 2] = 1.0
        return mask


MASK_BANK = _RandMaskBank()


# ------------------------------------------------------------------
# AutoSpuSwap — counterfactual generation + consistency loss
# ------------------------------------------------------------------
class AutoSpuSwap(nn.Module):
    """Minimal, functional AutoSpuSwap implementation."""

    def __init__(self,
                 swap_prob: float = 1.0,
                 lambda_consistency: float = 1.0,
                 lambda_fourier: float = 0.5,
                 fourier_cfg: Dict[str, Any] | None = None,
                 device: str = "cpu") -> None:
        super().__init__()
        self.swap_prob = float(swap_prob)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_fourier = float(lambda_fourier)

        # Fourier aug
        phase_max = (fourier_cfg or {}).get("phase_max", math.pi / 5)
        amp_lo = (fourier_cfg or {}).get("amp_min", 0.8)
        amp_hi = (fourier_cfg or {}).get("amp_max", 1.2)
        self.fourier = FourierPerturb(phase_max, (amp_lo, amp_hi)).to(device)
        self.mse = nn.MSELoss()

    # --------------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------------
    @torch.no_grad()
    def make_counterfactual(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.swap_prob:
            return x.clone()
        x_cf = self._swap_context(x)
        if self.lambda_fourier > 0:
            x_cf = self.fourier(x_cf)
        return x_cf

    def consistency_loss(self, logits: torch.Tensor, logits_cf: torch.Tensor) -> torch.Tensor:
        return self.lambda_consistency * self.mse(logits, logits_cf)

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    @torch.no_grad()
    def _swap_context(self, x: torch.Tensor) -> torch.Tensor:
        # simple in-batch permutation
        perm = torch.randperm(x.size(0), device=x.device)
        mask = MASK_BANK(x).to(x.device)
        return x * mask + x[perm] * (1 - mask)


# ------------------------------------------------------------------
# Toy backbone for CI smoke test
# ------------------------------------------------------------------
class TinyConv(nn.Module):
    """Extremely small CNN – keeps CI wall-clock short."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(F.relu(self.conv2(x)), kernel_size=x.shape[-1])
        x = x.flatten(1)
        return self.fc(x)


# ------------------------------------------------------------------
# Generic helpers (train-test loop, seeding)
# ------------------------------------------------------------------

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    import numpy as np  # local import avoids requirements if not needed elsewhere

    np.random.seed(seed)


def run_epoch(model: nn.Module,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer | None,
              criterion: nn.Module,
              autospu: AutoSpuSwap | None,
              device: str) -> tuple[float, float]:
    """Runs one training or validation epoch and returns (loss, accuracy)."""

    is_train = optimizer is not None
    model.train(is_train)

    total, correct, loss_sum = 0, 0, 0.0

    for x, y, *_ in loader:  # loader may optionally return extra fields (group/id)
        x, y = x.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if autospu is not None:
            with torch.no_grad():
                x_cf = autospu.make_counterfactual(x)
            logits_cf = model(x_cf)
            loss = loss + autospu.consistency_loss(logits, logits_cf)

        if is_train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total
