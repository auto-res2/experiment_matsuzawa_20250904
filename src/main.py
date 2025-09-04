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

                # ----------------------------------------------------------
                # Quick gradient check (performed on the target DEVICE)    
                # ----------------------------------------------------------
                model = model.to(DEVICE)
                data_gpu = data.to(DEVICE)
                logits, _ = model(data_gpu)
                loss = torch.nn.functional.cross_entropy(
                    logits[data_gpu.train_mask], data_gpu.y[data_gpu.train_mask]
                )
                loss.backward()
                total_grad = sum(p.grad.abs().sum() for p in model.parameters())
                if total_grad < 1e-6:
                    raise RuntimeError("Gradient vanished – check model definition")
                # Free the graph before training (saves memory)
                model.zero_grad(set_to_none=True)

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

# (The rest of the file remains unchanged)

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
        from .main import run_exp2  # type: ignore  # local import to keep diff small
        run_exp2()
    elif exp in (3, "3", "exp3"):
        from .main import run_exp3  # type: ignore
        run_exp3()
    elif exp in ("all", "*"):
        run_exp1()
        from .main import run_exp2, run_exp3  # type: ignore
        run_exp2()
        run_exp3()
    else:
        raise ValueError("experiment must be 1,2,3 or all")

    print("\n>>> Finished – all figures are stored in .research/iteration20/images <<<")
