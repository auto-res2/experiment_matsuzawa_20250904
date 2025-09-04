"""
src/main.py – single entry-point (run:  python -m src.main )
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from .evaluate import eval_full
from .preprocess import load_dataset
from .train import MODELS, gradient_check, set_seeds, train_one

# -----------------------------------------------------------------------------
# Config handling
# -----------------------------------------------------------------------------

CFG_DIR = Path("config")
CFG_PATH = CFG_DIR / "config.yaml"
CFG_DIR.mkdir(exist_ok=True)

# Write a minimal default config if none is present so the repo is runnable
if not CFG_PATH.exists():
    CFG_PATH.write_text(
        yaml.safe_dump(
            {
                "experiment": "default-exp",
                "datasets": ["Path-of-Cliques", "Ring-of-Cliques"],
                "models": ["GCN", "PairNorm", "CurvAMP"],
                "depth": 32,
                "hidden": 128,
                "lr": 5e-4,
                "wd": 5e-4,
                "patience": 100,
                "max_epochs": 400,
                "seeds": [0, 1, 2],
            }
        )
    )

# -----------------------------------------------------------------------------


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU mandatory – aborting (consistency item F).")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(CFG_PATH), help="Path to YAML config file")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    print("===== Experiment description =====\n", yaml.safe_dump(cfg, sort_keys=False))

    for dataset_name in cfg["datasets"]:
        for model_name in cfg["models"]:
            for seed in cfg["seeds"]:
                print(f"\n>>> {dataset_name} – {model_name} – seed {seed}")
                set_seeds(seed)
                data = load_dataset(dataset_name, seed)

                ModelCls = MODELS[model_name]
                if model_name == "CurvAMP":
                    model = ModelCls(
                        data.num_features,
                        cfg["hidden"],
                        int(data.y.max().item() + 1),
                        cfg["depth"],
                        K=3,
                        rewired_ratio=0.02,
                    )
                else:
                    model = ModelCls(
                        data.num_features,
                        cfg["hidden"],
                        int(data.y.max().item() + 1),
                        cfg["depth"],
                        dropout=0.2,
                    )

                # Quick gradient sanity check before expensive training
                gradient_check(model, data)

                model, best_val = train_one(model, data, cfg, torch.device("cuda"))
                results = eval_full(model, data, f"{dataset_name}_{model_name}_seed{seed}")

                print(f"Acc={results['acc']:.4f}  ATER={results['ater']:.4f}")
                print(
                    f"Figures saved: gdr_{dataset_name}_{model_name}_seed{seed}.pdf, "
                    f"er_{dataset_name}_{model_name}_seed{seed}.pdf"
                )


if __name__ == "__main__":
    main()
