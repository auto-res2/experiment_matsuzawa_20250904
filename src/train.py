"""
train.py – model architectures, memory buffers and the full training loop
--------------------------------------------------------------------------
All heavy-lifting (forward/back-prop, replay, buffer management, metrics,
plotting, checkpoint bookkeeping) lives here so that other modules can stay
light-weight.  Nothing was rewritten from scratch – the code is an exact
re-organisation of the original monolithic script with a few defensive
additions (CUDA availability checks, type safety casts, graceful download
failures) to guarantee smoother execution in a wider range of environments.
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (after Agg)

try:
    from torchinfo import summary  # optional, so we do not crash if missing
except Exception:  # pragma: no cover
    summary = None  # type: ignore

from .evaluate import accuracy, compute_forgetting  # no circular dep.
from .preprocess import DATA_ROOT, FIG_ROOT, CKPT_ROOT, download, SplitCIFAR100

# ---------------------------------------------------------------------------
#  Reproducibility helper
# ---------------------------------------------------------------------------
SEEDS: Tuple[int, ...] = (11, 13, 17, 19, 23)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
#  Vector-Quantiser layer (VQ-VAE bottleneck)
# ---------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        nn.init.uniform_(self.embedding.weight, -1 / n_e, 1 / n_e)

    def forward(self, z: torch.Tensor):  # z: (B,C,H,W)
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        flat = z_perm.view(-1, self.e_dim)
        # pair-wise (x-y)^2 = x^2 - 2xy + y^2
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idxs = dists.argmin(1)
        z_q = self.embedding(idxs).view_as(z_perm)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, idxs.view(z.shape[0], -1), loss

    # Used during replay when only integer codes are available
    def embed_code(self, codes: torch.Tensor):
        return self.embedding(codes)

# ---------------------------------------------------------------------------
#  ResNet18 backbone augmented with VQ bottleneck
# ---------------------------------------------------------------------------
class ResNet18_VQ(nn.Module):
    def __init__(self, n_classes: int, codebook: int = 512, dim: int = 64):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Encoder up to (and incl.) layer2
        self.enc = nn.Sequential(*(list(base.children())[:6]))  # conv1-layer2
        self.prepool = list(base.children())[6]  # layer3
        self.quant = VectorQuantizer(codebook, dim)
        # Lightweight decoder – only reconstructs feature maps, not pixels.
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.head = nn.Sequential(
            base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, n_classes),
        )
        self._freeze_decoder: bool = False

    def forward(self, x: torch.Tensor, *, replay_codes: Optional[torch.Tensor] = None):
        z = self.enc(x)  # (B,128,H/8,W/8)
        if replay_codes is not None:  # during replay we bypass quantiser
            z_q = self.quant.embed_code(replay_codes).view_as(z)
            code_idx = replay_codes
            quant_loss = torch.tensor(0.0, device=x.device)
        else:
            z_q, code_idx, quant_loss = self.quant(z)
        if not self._freeze_decoder:
            _ = self.dec(z_q)  # decoder branch – not used at inference time
        y = self.prepool(z_q)
        logits = self.head(y)
        return logits, code_idx, quant_loss

    def freeze_decoder(self):  # FLOP / memory optimiser
        if self._freeze_decoder:
            return
        for p in self.dec.parameters():
            p.requires_grad = False
        self._freeze_decoder = True

# ---------------------------------------------------------------------------
#  Memory buffer abstractions (raw, JPEG, latent codes)
# ---------------------------------------------------------------------------
class BaseMemory:  # minimal interface
    def add(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, batch_size: int):
        raise NotImplementedError

    def n_bytes(self) -> int:
        raise NotImplementedError

# --- Raw RGB tensors --------------------------------------------------------
class RawExemplarMemory(BaseMemory):
    """Store raw uint8 tensors on CPU; fast but very space hungry."""

    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.buffer: List[Tuple[torch.ByteTensor, int]] = []  # (img, label)
        self.bytes_per_ex = 32 * 32 * 3  # CIFAR uint8 per image

    def add(self, imgs: torch.Tensor, labels: torch.Tensor):
        for img, lbl in zip(imgs, labels):
            if self.n_bytes() + self.bytes_per_ex > self.capacity:
                break
            self.buffer.append((img.cpu().byte(), int(lbl)))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        imgs, labels = zip(*(self.buffer[i] for i in idx))
        return torch.stack([i.float() / 255.0 for i in imgs]), torch.tensor(labels)

    def n_bytes(self):
        return len(self.buffer) * self.bytes_per_ex

# --- JPEG-compressed --------------------------------------------------------
class JPEGExemplarMemory(BaseMemory):
    """JPEG compress each exemplar; good space/quality trade-off."""

    def __init__(self, capacity_bytes: int, jpeg_quality: int):
        from PIL import Image  # lazy import
        self.capacity = capacity_bytes
        self.buf: List[Tuple[bytes, int]] = []
        self.quality = jpeg_quality
        self._pil = Image
        self.transform = T.Compose([T.ToTensor()])

    def add(self, imgs: torch.Tensor, labels: torch.Tensor):
        import io
        for img, lbl in zip(imgs, labels):
            pil_img = T.ToPILImage()(img.cpu())
            byte_io = io.BytesIO()
            pil_img.save(byte_io, format="JPEG", quality=self.quality)
            data = byte_io.getvalue()
            if self.n_bytes() + len(data) > self.capacity:
                break
            self.buf.append((data, int(lbl)))

    def sample(self, batch_size: int):
        import io
        idx = np.random.choice(len(self.buf), size=min(batch_size, len(self.buf)), replace=False)
        tensors, labels = [], []
        for data, lbl in (self.buf[i] for i in idx):
            pil = self._pil.open(io.BytesIO(data)).convert("RGB")
            tensors.append(self.transform(pil))
            labels.append(lbl)
        return torch.stack(tensors), torch.tensor(labels)

    def n_bytes(self):
        return sum(len(d) for d, _ in self.buf)

# --- Latent integer codes ---------------------------------------------------
class LatentCodeMemory(BaseMemory):
    """Store only integer codebook indices (plus label). Extremely compact."""

    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.codes: List[Tuple[torch.IntTensor, int]] = []
        self.bytes_per_idx = 4  # int32 per code

    def add(self, codes: torch.Tensor, labels: torch.Tensor):
        for c, l in zip(codes, labels):
            needed = c.numel() * self.bytes_per_idx + 1  # +1 for label
            if self.n_bytes() + needed > self.capacity:
                break
            self.codes.append((c.int().cpu(), int(l)))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.codes), size=min(batch_size, len(self.codes)), replace=False)
        codes, labels = zip(*(self.codes[i] for i in idx))
        return torch.stack(codes), torch.tensor(labels)

    def n_bytes(self):
        return sum(c.numel() * self.bytes_per_idx + 1 for c, _ in self.codes)

# ---------------------------------------------------------------------------
#  Dataclasses for configuration – serialised to YAML for reproducibility
# ---------------------------------------------------------------------------
@dataclass
class OptimConfig:
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

@dataclass
class ContinualConfig:
    n_tasks: int
    classes_per_task: int
    buffer_bytes: int
    replay_ratio: float = 0.5
    epochs: int = 10
    batch_size: int = 128

@dataclass
class ExperimentConfig:
    name: str
    dataset: str
    model: str
    method: str  # LQR | ER | JPEG-ER
    buffer_quality: Optional[int] = None  # only for JPEG
    optim: OptimConfig = field(default_factory=OptimConfig)
    ctl: ContinualConfig | None = None
    seed: int = 0

# ---------------------------------------------------------------------------
#  Single experiment – one seed, one memory method, one buffer budget
# ---------------------------------------------------------------------------

def run_single_experiment(cfg: ExperimentConfig) -> Dict[str, float]:
    print(f"\n=== Running {cfg.name} | seed={cfg.seed} ===\n")
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if cfg.dataset == "split_cifar100":
        dataset = SplitCIFAR100(DATA_ROOT, seed=cfg.seed, n_tasks=cfg.ctl.n_tasks)
    else:
        raise NotImplementedError(cfg.dataset)

    # ------------------------------------------------------------------
    # Model & optimiser
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_VQ(n_classes=100).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )

    # ------------------------------------------------------------------
    # Replay memory
    # ------------------------------------------------------------------
    if cfg.method == "LQR":
        buffer: BaseMemory = LatentCodeMemory(cfg.ctl.buffer_bytes)
    elif cfg.method == "ER":
        buffer = RawExemplarMemory(cfg.ctl.buffer_bytes)
    elif cfg.method == "JPEG-ER":
        if cfg.buffer_quality is None:
            raise ValueError("JPEG quality must be set for JPEG-ER")
        buffer = JPEGExemplarMemory(cfg.ctl.buffer_bytes, cfg.buffer_quality)
    else:
        raise NotImplementedError(cfg.method)

    # Containers for logging
    acc_matrix: List[List[float]] = []
    task_times: List[float] = []

    # ------------------------------------------------------------------
    # Continual learning loop – iterate over tasks
    # ------------------------------------------------------------------
    for task_id in range(cfg.ctl.n_tasks):
        train_set, test_set = dataset.get_task(task_id)
        val_len = int(0.1 * len(train_set))
        train_len = len(train_set) - val_len
        train_split, val_split = random_split(train_set, [train_len, val_len])

        train_loader = DataLoader(
            train_split,
            batch_size=cfg.ctl.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(val_split, batch_size=256, shuffle=False, num_workers=4)
        print(f"Task {task_id+1}/{cfg.ctl.n_tasks}  (#train {len(train_split)})")

        t0 = time.perf_counter()
        # --------------- standard supervised training ------------------
        for epoch in range(cfg.ctl.epochs):
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, code_idx, qloss = model(imgs)
                loss = F.cross_entropy(logits, labels) + qloss

                # ---------------------- replay -------------------------
                if buffer.n_bytes() > 0 and random.random() < cfg.ctl.replay_ratio:
                    if cfg.method == "LQR":
                        rep_codes, rep_labels = buffer.sample(imgs.size(0))
                        rep_codes, rep_labels = rep_codes.to(device), rep_labels.to(device)
                        logits_rep, _, _ = model(imgs[: rep_codes.size(0)], replay_codes=rep_codes)
                    else:
                        rep_imgs, rep_labels = buffer.sample(imgs.size(0))
                        rep_imgs, rep_labels = rep_imgs.to(device), rep_labels.to(device)
                        logits_rep, _, _ = model(rep_imgs)
                    loss += F.cross_entropy(logits_rep, rep_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

            # ------------ validation (quick sanity) --------------------
            model.eval()
            with torch.no_grad():
                val_acc = np.mean([accuracy(model(x.to(device))[0].cpu(), y) for x, y in val_loader])
            print(f"  Epoch {epoch+1}/{cfg.ctl.epochs}   val-acc={val_acc:.2f}%")

        # ------------------------------------------------------------------
        # After finishing the task – freeze decoder, fill buffer, evaluate
        # ------------------------------------------------------------------
        model.freeze_decoder()
        model.eval()

        # --- add exemplars -------------------------------------------------
        all_imgs, all_codes, all_labels = [], [], []
        with torch.no_grad():
            for imgs, labels in DataLoader(train_split, batch_size=256, shuffle=False):
                imgs = imgs.to(device)
                _, codes, _ = model(imgs)
                all_imgs.append(imgs.cpu())
                all_labels.append(labels)
                all_codes.append(codes.cpu())
        imgs_cat = torch.cat(all_imgs)
        labels_cat = torch.cat(all_labels)
        codes_cat = torch.cat(all_codes)
        perm = torch.randperm(len(labels_cat))
        if cfg.method == "LQR":
            buffer.add(codes_cat[perm], labels_cat[perm])
        else:
            buffer.add(imgs_cat[perm], labels_cat[perm])

        # --- evaluate on all seen tasks -----------------------------------
        task_acc: List[float] = []
        for k in range(task_id + 1):
            _, te_loader = dataset.get_task(k)
            te_loader = DataLoader(te_loader, batch_size=256, shuffle=False)
            with torch.no_grad():
                tacc = np.mean([accuracy(model(x.to(device))[0].cpu(), y) for x, y in te_loader])
            task_acc.append(tacc)
        acc_matrix.append(task_acc)
        elapsed = time.perf_counter() - t0
        task_times.append(elapsed)
        print(f"Task {task_id} done. Elapsed {elapsed/60:.1f} min.  Acc={task_acc[-1]:.2f}%\n")

    # ------------------------------------------------------------------
    # Summary metrics & plots
    # ------------------------------------------------------------------
    fgt = float(np.mean(compute_forgetting(acc_matrix)))
    final_acc = float(np.mean(acc_matrix[-1]))
    print("\n*** Final Results ***")
    print(f"Average Accuracy  : {final_acc:.2f} %")
    print(f"Average Forgetting: {fgt:.2f} %")
    print(f"Memory budget     : {buffer.n_bytes()/1024:.1f} kB (cap {cfg.ctl.buffer_bytes/1024} kB)")
    print(f"Total train time  : {sum(task_times)/60:.1f} min")

    # quick line plot -------------------------------------------------------
    xs = list(range(1, cfg.ctl.n_tasks + 1))
    ys = [np.mean(acc_matrix[t]) for t in range(cfg.ctl.n_tasks)]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", label=cfg.name)
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.1f}", (x, y + 0.5))
    plt.xlabel("Task")
    plt.ylabel("Average Accuracy (%)")
    plt.title(cfg.name)
    plt.grid(True)
    plt.legend()
    fig_path = FIG_ROOT / f"accuracy_{cfg.name}.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure  → {fig_path.name}\n")

    return {
        "acc": final_acc,
        "fgt": fgt,
        "time": float(sum(task_times)),
        "bytes": buffer.n_bytes(),
    }
