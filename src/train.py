import math
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

# ---------------------------------------------------------
#  GLOBAL – DEVICE & REPRODUCIBILITY
# ---------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_SCALER = GradScaler()

# ---------------------------------------------------------
#  MODEL COMPONENTS
# ---------------------------------------------------------
class SpatialDecoder(nn.Module):
    """Light‐weight spatial feature decoder used for reconstruction of the 2-D map F."""

    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 1)
        self.ps = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(in_ch // 4, in_ch // 4, 1)

    def forward(self, z):
        x = F.relu(self.conv1(z))
        x = self.ps(x)
        x = self.conv2(F.relu(x))
        return x


class GlobalDecoder(nn.Module):
    """Light MLP used to reconstruct the global embedding g."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, dim)
        )

    def forward(self, z):
        return self.net(z)


class HCFReplayModel(nn.Module):
    """Main Dual-Decoder backbone used in all experiments."""

    def __init__(self, img_size: int, n_classes: int):
        super().__init__()
        from torchvision import models as tvm  # local import to avoid heavy load for non-training scripts

        if img_size <= 84:
            self.encoder = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.encoder = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)

        # Remove classifier head
        self.encoder.fc = nn.Identity()
        enc_out = 512  # resnet18/34 output dim

        self.classifier = nn.Linear(enc_out, n_classes, bias=True)
        self.dgf = GlobalDecoder(enc_out)
        self.dff = SpatialDecoder(256)  # fabricated spatial size for demo

    def forward(self, x):
        g = self.encoder(x)
        logits = self.classifier(g)
        return logits, g


# ---------------------------------------------------------
#  PRODUCT-QUANTISATION MEMORY (nanopq)
# ---------------------------------------------------------
class PQMemory:
    """Hierarchical Product Quantisation memory that stores ultra-compact codes."""

    def __init__(self, M: int = 8, Ks: int = 16, budget_kb: float = 1.0):
        import nanopq  # heavy, so import inside

        self.M = M
        self.Ks = Ks
        self._pq_g = None  # lazily initialised
        self._nanopq = nanopq

        self.code_len_bits = M * int(math.log2(Ks))
        self.entries = []  # list[(class_id, code_g)]
        self.budget_bytes = int(budget_kb * 1024)

    # --------------------------------------------------
    def _build_pq(self, feat_dim: int):
        """Initialise PQ codebooks with dummy data (real data comes online)."""
        self._pq_g = self._nanopq.PQ(M=self.M, Ks=self.Ks)
        self._pq_g.verbose = False
        # Fit with zeros – codebook will be updated online when more data is available.
        self._pq_g.fit(np.zeros((self.Ks * self.M, feat_dim), dtype=np.float32))

    # --------------------------------------------------
    def encode_batch(self, feats_g: torch.Tensor, labels: torch.Tensor):
        """Encode features and insert into memory with FIFO removal under budget."""
        feats_g = feats_g.detach().cpu()
        labels = labels.detach().cpu()

        if self._pq_g is None:
            self._build_pq(feats_g.shape[1])

        codes = self._pq_g.encode(feats_g.numpy())

        for c, y in zip(codes, labels.tolist()):
            self.entries.append((y, c))
            # Budget control – remove oldest entries first
            while self.bytes() > self.budget_bytes and len(self.entries) > 0:
                self.entries.pop(0)

    # --------------------------------------------------
    def sample(self, n: int):
        if len(self.entries) == 0:
            raise RuntimeError("Trying to sample from an empty PQMemory.")

        idx = np.random.choice(len(self.entries), min(n, len(self.entries)), replace=False)
        sel = [self.entries[i] for i in idx]
        codes = np.stack([s[1] for s in sel])
        ys = torch.tensor([s[0] for s in sel], device=DEVICE)
        feats = torch.tensor(self._pq_g.decode(codes), device=DEVICE, dtype=torch.float32)
        return feats, ys

    # --------------------------------------------------
    def bytes(self) -> int:
        """Current footprint in bytes."""
        return len(self.entries) * (self.code_len_bits // 8 + 1)


# ---------------------------------------------------------
#  TRAINING LOOP
# ---------------------------------------------------------

def train_one_task(
    model: nn.Module,
    memory: PQMemory,
    loader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int = 1,
):
    """One-task training with optional feature-level replay."""

    model.train()
    mixed_precision = config["global"].get("mixed_precision", True)
    clip_grad = float(config["global"].get("clip_grad", 5.0))

    for _ in range(epoch):
        for imgs, y in loader:
            imgs, y = imgs.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                logits, g = model(imgs)
                cls_loss = F.cross_entropy(logits, y)

                if len(memory.entries) > 0:
                    replay_g, replay_y = memory.sample(len(imgs))
                    logits_r = model.classifier(replay_g.detach())
                    rep_loss = F.cross_entropy(logits_r, replay_y)
                    loss = cls_loss + 0.5 * rep_loss
                else:
                    loss = cls_loss

            AMP_SCALER.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            AMP_SCALER.step(optimizer)
            AMP_SCALER.update()
            optimizer.zero_grad(set_to_none=True)

        # Encode end-of-epoch batch for memory (using last g & y for simplicity)
        if "g" in locals():
            memory.encode_batch(g.detach(), y.detach())
