"""Main orchestration script – entry point:  python -m src.main"""

import sys
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T, datasets as tvds

from .train import HCFReplayModel, PQMemory, train_one_task, DEVICE
from .evaluate import evaluate, plot_curve
from .preprocess import prepare_cifar100, split_classes

# ---------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------
_CFG_PATH = Path("config/config.yaml")
if not _CFG_PATH.exists():
    raise FileNotFoundError("config/config.yaml not found – please ensure it exists.")

with _CFG_PATH.open() as fh:
    CONFIG = yaml.safe_load(fh)


# ---------------------------------------------------------
#  EXPERIMENT-1: Memory-Accuracy Frontier
# ---------------------------------------------------------

def run_exp1():
    print("\n==========  EXPERIMENT-1  Memory-Accuracy Frontier  ==========")
    budgets = CONFIG["experiments"]["exp1"]["budgets_kb"]
    res_table = defaultdict(list)

    # Only CIFAR-100 implemented in this demo
    ds_name = "cifar100"
    print(f"\n--- Dataset: {ds_name} ---")

    droot = prepare_cifar100()
    full_ds = tvds.CIFAR100(
        root=droot,
        train=True,
        download=False,
        transform=T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
            ]
        ),
    )
    labels = np.array(full_ds.targets)
    n_tasks, cls_per = CONFIG["datasets"]["cifar100"]["split"]
    tasks = split_classes(labels, n_tasks, cls_per)

    for budget in budgets:
        print(f"Budget {budget} kB /class")
        # Fresh model per budget run
        model = HCFReplayModel(img_size=32, n_classes=100).to(DEVICE)
        optim_cfg = CONFIG["global"]["optim"]
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(optim_cfg["lr"]),
            momentum=float(optim_cfg["momentum"]),
            weight_decay=float(optim_cfg["weight_decay"]),
        )
        mem = PQMemory(M=CONFIG["global"]["pq"]["M"], Ks=16, budget_kb=budget)

        AA_list = []
        for t, cls in enumerate(tasks):
            idx = np.where(np.isin(labels, cls))[0]
            sub_ds = Subset(full_ds, idx)
            loader = DataLoader(
                sub_ds,
                batch_size=int(CONFIG["global"]["batch_size"]),
                shuffle=True,
                num_workers=4,
            )
            train_one_task(model, mem, loader, optimizer, CONFIG, epoch=1)
            acc = evaluate(model, loader)
            AA_list.append(acc)
            print(
                f"Task {t + 1}/{n_tasks} – Acc={acc:.2f}%  Mem={mem.bytes() / 1024:.1f} kB"
            )
        AA = float(np.mean(AA_list))
        res_table["budget"].append(budget)
        res_table["AA"].append(AA)

    # Plot & print results
    plot_curve(
        res_table["budget"],
        res_table["AA"],
        xlab="kB / class",
        ylab="Average Acc %",
        title="HCF-Replay on CIFAR-100",
        fname="accuracy_cifar100.pdf",
    )

    print("\n== Raw results (Experiment-1) ==")
    for b, a in zip(res_table["budget"], res_table["AA"]):
        print(f"Budget {b} kB : AA = {a:.2f}%")


# ---------------------------------------------------------
#  MAIN
# ---------------------------------------------------------

def main():
    print("===========================================================")
    print(" Hierarchical Compressed Feature Replay – Experiment Suite ")
    print("===========================================================")
    run_exp1()
    print("\nAll experiments finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user – exiting…", file=sys.stderr)
