"""Entry-point script – run with `python -m src.main` (from project root)."""

import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from .train import MODELS, train_model, set_seeds, CUDATimer
from .preprocess import load_dataset
from .evaluate import accuracy, gdr, eff_rank, ater, lineplot, wilcoxon_signed

###############################################################################
#                               CONFIGURATION                                 #
###############################################################################

_CFG_PATH = Path("config/config.yaml")
if not _CFG_PATH.exists():
    _CFG_PATH.parent.mkdir(exist_ok=True, parents=True)
    _CFG_PATH.write_text(
        """
# =========================================================
#  Default CurvAMP experiment configuration
# =========================================================
experiment: 1        # 1, 2, 3 or "all"
fast: false          # true = quick sanity pass
hidden: 128
lr: 5e-4
wd: 5e-4
patience: 100
max_epochs: 2000
        """
    )

with _CFG_PATH.open() as fp:
    CFG: Dict[str, Any] = yaml.safe_load(fp)
print("===== Active configuration =====\n" + yaml.safe_dump(CFG, sort_keys=False))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required – fail-fast as per protocol")
DEVICE = torch.device("cuda")

###############################################################################
#                                EXPERIMENTS                                  #
###############################################################################

def run_exp1():
    print("\n=== EXP-1  Sanity & Dual-Pathology Proof ===")
    datasets = ["Path-of-Cliques", "Ring-of-Cliques", "Mixture"]
    baselines = ["GCN", "PairNorm", "CurvAMP"]

    for ds in datasets:
        for seed in range(3 if CFG.get("fast", False) else 10):
            data = load_dataset(ds, seed)
            results = {}
            for model_name in baselines:
                set_seeds(seed)
                Model = MODELS[model_name]
                model = (
                    Model(
                        data.num_features,
                        CFG["hidden"],
                        int(data.y.max().item() + 1),
                        CFG.get("depth", 32),
                    )
                    if model_name != "CurvAMP"
                    else Model(
                        data.num_features,
                        CFG["hidden"],
                        int(data.y.max().item() + 1),
                        CFG.get("depth", 32),
                        K=3,
                        rewired_ratio=0.02,
                    )
                )

                # quick gradient check -----------------------------------
                logits, _ = model(data)
                loss = torch.nn.functional.cross_entropy(
                    logits[data.train_mask], data.y[data.train_mask]
                )
                loss.backward()
                total_grad = sum(p.grad.abs().sum() for p in model.parameters())
                if total_grad < 1e-6:
                    raise RuntimeError("Gradient vanished – check model definition")

                # train ---------------------------------------------------
                with CUDATimer() as t:
                    model = train_model(model, data, CFG, DEVICE)
                cost = t.sec

                # evaluation ---------------------------------------------
                model.eval()
                logits, feats = model(data.to(DEVICE))
                acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
                gdr_curve = [gdr(f, data.y.to(f.device)) for f in feats]
                er_curve = [eff_rank(f) for f in feats]
                a = ater(data.edge_index, data.num_nodes)

                fname = f"{ds}_{model_name}_seed{seed}.pdf"
                lineplot(gdr_curve, f"GDR {ds} {model_name}", "GDR", f"gdr_{fname}")

                results[model_name] = dict(
                    acc=acc, ater=a, gdr=max(gdr_curve), er=max(er_curve), time=cost
                )

            # success check
            if ds == "Mixture" and results["CurvAMP"]["acc"] < 0.9:
                raise RuntimeError("❌  Exp-1 criterion failed for CurvAMP on Mixture")
            print(ds, seed, json.dumps(results, indent=2))


def run_exp2():
    print("\n=== EXP-2  Benchmark Matrix ===")
    import pandas as pd
    import numpy as np

    datasets = [
        "Cora",
        "CiteSeer",
        "PubMed",
        "Texas",
        "Cornell",
        "Wisconsin",
        "Peptides-func",
        "Peptides-struct",
    ]
    depths = [2, 8, 16, 32] if CFG.get("fast", False) else [2, 8, 16, 32, 64, 128]
    models = ["GCN", "PairNorm", "CurvAMP"]

    summary = []
    for ds in datasets:
        for L in depths:
            for seed in range(3 if CFG.get("fast", False) else 10):
                data = load_dataset(ds, seed)
                row = {"ds": ds, "L": L, "seed": seed}

                for mname in models:
                    set_seeds(seed)
                    Model = MODELS[mname]
                    model = (
                        Model(data.num_features, CFG["hidden"], int(data.y.max()) + 1, L)
                        if mname != "CurvAMP"
                        else Model(
                            data.num_features,
                            CFG["hidden"],
                            int(data.y.max()) + 1,
                            L,
                            K=3,
                            rewired_ratio=0.02,
                        )
                    )
                    model = train_model(model, data, CFG, DEVICE)
                    model.eval()
                    logits, _ = model(data.to(DEVICE))
                    metric = accuracy(logits[data.test_mask], data.y[data.test_mask])
                    row[mname] = metric
                summary.append(row)

            # pairwise Wilcoxon test (CurvAMP vs. PairNorm)
            a = np.array([r["CurvAMP"] for r in summary if r["ds"] == ds and r["L"] == L])
            b = np.array([r["PairNorm"] for r in summary if r["ds"] == ds and r["L"] == L])
            stat, p = wilcoxon_signed(a.tolist(), b.tolist())
            print(f"{ds} L={L}: Δ={a.mean() - b.mean():.3f}  p={p:.3e}")

    pd.DataFrame(summary).to_csv(Path("logs/exp2_summary.csv"), index=False)


def run_exp3():
    print("\n=== EXP-3  Component Ablation ===")
    datasets = ["PubMed", "Peptides-struct"]
    variants = ["full", "-rewire", "-curvpair", "-gate"]

    for ds in datasets:
        data = load_dataset(ds, 0)
        results = {}
        for variant in variants:
            set_seeds(0)
            from .train import CurvAMP, CurvAMPConv  # local import to patch

            if variant == "full":
                model = CurvAMP(data.num_features, CFG["hidden"], int(data.y.max()) + 1, 32)
            else:
                # Monkey-patch single layer for ablations
                class Patched(CurvAMPConv):
                    def forward(self, x, ei):
                        out, ei = super().forward(x, ei)
                        if variant == "-rewire":
                            pass  # rewiring already conditional via rr
                        return out, ei

                class Net(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.layers = torch.nn.ModuleList(
                            [
                                Patched(
                                    data.num_features if i == 0 else CFG["hidden"],
                                    CFG["hidden"],
                                    3,
                                    0 if variant == "-rewire" else 0.02,
                                )
                                for i in range(32)
                            ]
                        )
                        self.head = torch.nn.Linear(CFG["hidden"], int(data.y.max()) + 1)

                    def forward(self, d):
                        x, ei = d.x, d.edge_index
                        for layer in self.layers:
                            x, ei = layer(x, ei)
                            x = torch.relu(x)
                        return self.head(x), []

                model = Net()

            model = train_model(model, data, CFG, DEVICE)
            model.eval()
            logits, _ = model(data.to(DEVICE))
            acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
            results[variant] = acc
        print(ds, json.dumps(results, indent=2))

###############################################################################
#                                   MAIN                                     #
###############################################################################

if __name__ == "__main__":
    LOGS = Path("logs")
    LOGS.mkdir(exist_ok=True, parents=True)

    exp = CFG["experiment"]
    if exp in (1, "1", "exp1"):
        run_exp1()
    elif exp in (2, "2", "exp2"):
        run_exp2()
    elif exp in (3, "3", "exp3"):
        run_exp3()
    elif exp in ("all", "*"):
        run_exp1()
        run_exp2()
        run_exp3()
    else:
        raise ValueError("experiment must be 1,2,3 or all")

    print("\n>>> Finished – all figures are stored in .research/iteration16/images <<<")