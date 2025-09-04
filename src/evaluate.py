from __future__ import annotations

"""src/evaluate.py
Evaluation utilities, training loop, experiment definitions & plotting.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns  # noqa: F401  (imported for its style-side-effects)
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

from .preprocess import COMMON, load_dataset, set_global_seed, ensure_image_dir
from .train import DeepGNN

# ---------------------------------------------------------------------------
# 1.  Low-level helpers
# ---------------------------------------------------------------------------

def effective_rank(emb: torch.Tensor) -> float:
    """Shannon effective rank (Roy & Vetterli)."""
    with torch.no_grad():
        s = torch.linalg.svdvals(emb)
        if (s <= 0).all():  # all-zero embeddings
            return 0.0
        p = s / s.sum()
        H = -(p * (p + 1e-12).log()).sum()
        return float(torch.exp(H))


def save_lineplot(
    xs: List[int],
    ys_dict: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Utility to save a publication-ready line plot."""
    plt.figure(figsize=(6, 4))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", label=label)
        for x, y in zip(xs, ys):
            plt.text(x, y, f"{y:.2f}", fontsize=6, ha="center", va="bottom")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    # ensure directory exists before saving (defensive)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 2.  Single-seed train-and-evaluate routine
# ---------------------------------------------------------------------------

def run_single(
    data,
    model: DeepGNN,
    device: str,
    epochs: int = 1000,
    early: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
) -> Tuple[float, float, float]:
    """Train *model* on *data* and return (best_test_acc, eff_rank, sec/epoch)."""

    model.to(device)
    data = data.to(device)
    train_mask, val_mask, test_mask = (
        data.train_mask,
        data.val_mask,
        data.test_mask,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, epochs)

    best_val: float = 0.0
    best_test: float = 0.0
    best_epoch: int = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, reg_loss = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask]) + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # ---- validation & early stopping every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(data.x, data.edge_index)
                pred = logits.argmax(dim=1)
                accs = [
                    (pred[m] == data.y[m]).float().mean().item()
                    for m in (train_mask, val_mask, test_mask)
                ]
                val_acc = accs[1]
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = accs[2]
                    best_epoch = epoch
                if epoch - best_epoch > early:
                    break

    sec_per_epoch = (time.time() - t0) / max(1, epoch)

    # effective rank on penultimate representation
    model.eval()
    with torch.no_grad():
        if len(model.layers) > 1:
            penult_emb = model.layers[-2](data.x.to(device), data.edge_index)
            if isinstance(penult_emb, tuple):
                penult_emb = penult_emb[0]  # FANSConv returns (x, reg)
        else:
            penult_emb, _ = model(data.x.to(device), data.edge_index)
    r_e = effective_rank(penult_emb.detach().cpu())
    return best_test, r_e, sec_per_epoch


# ---------------------------------------------------------------------------
# 3.  EXP-1  (Depth scalability & over-smoothing stress test)
# ---------------------------------------------------------------------------

def run_exp1(exp_cfg: Dict[str, dict]) -> None:  # noqa: C901  (complexity fine for driver)
    print(exp_cfg["description"])

    # results will be dumped as pretty JSON for copy-paste into reports
    num_results: Dict[str, List] = {}
    generated_figs: List[str] = []

    for ds_name, ds_entry in exp_cfg["datasets"].items():
        data = load_dataset(ds_entry)
        for backbone in exp_cfg["backbones"]:
            x_depth = exp_cfg["depth_grid"]
            y_dict: Dict[str, List[float]] = {r: [] for r in exp_cfg["remedies"]}

            for depth in x_depth:
                for remedy in exp_cfg["remedies"]:
                    acc_runs, re_runs, wall_runs = [], [], []
                    for seed in COMMON.seed_list:
                        set_global_seed(seed)
                        model = DeepGNN(
                            backbone=backbone,
                            remedy=remedy,
                            in_dim=data.num_features,
                            hidden=COMMON.hidden,
                            out_dim=int(data.y.max().item()) + 1,
                            layers=depth,
                            fans_kwargs=dict(k_hop=1, nyq_lambda=0.05),
                        )
                        acc, r_e, wall = run_single(
                            data,
                            model,
                            device=COMMON.device,
                            epochs=exp_cfg["epochs"][ds_name],
                            early=COMMON.early_stop,
                            lr=COMMON.lr,
                            weight_decay=COMMON.weight_decay,
                        )
                        acc_runs.append(acc)
                        re_runs.append(r_e)
                        wall_runs.append(wall)

                    mean_acc = float(np.mean(acc_runs))
                    y_dict[remedy].append(mean_acc)

                    key = f"{ds_name}-{backbone}-{remedy}-{depth}"
                    num_results[key] = [
                        mean_acc,
                        float(np.std(acc_runs) / math.sqrt(len(acc_runs))),
                        float(np.mean(re_runs)),
                    ]

            # write figure for this (dataset, backbone)
            fig_name = f"accuracy_depth_{ds_name}_{backbone}.pdf"
            img_dir = ensure_image_dir()  # creates .research/iteration2/images
            fig_path = img_dir / fig_name
            save_lineplot(
                x_depth,
                y_dict,
                f"{ds_name} / {backbone.upper()} Accuracy vs Depth",
                "Depth (layers)",
                "Accuracy",
                fig_path,
            )
            generated_figs.append(str(fig_path))

    # ---------------------------------------------------------------------
    # STDOUT summary
    # ---------------------------------------------------------------------
    print(json.dumps(num_results, indent=2))
    print("Figures written →")
    for f in generated_figs:
        print("  ", f)
