"""src/main.py
Main orchestrator.  Run `python -m src.main`.
"""
from __future__ import annotations
import sys, json
import torch

from .train import MODELS, train_model, cfg
from .preprocess import load_dataset
from .evaluate import (
    accuracy, effective_rank, group_distance_ratio,
    ater, plot_curve
)

if not torch.cuda.is_available():
    sys.stderr.write("CUDA device required by the experiment.\n"); sys.exit(1)

# ------------------------------------------------------------
# Select experiment & hyper-parameters from YAML
# ------------------------------------------------------------
EXPERIMENT = cfg('experiment', default=1)
DATASETS   = ["Path-of-Cliques", "Ring-of-Cliques", "Mixture"]
DEPTH_GRID = cfg('exp1', 'depth_grid', default=[4,8,16])
SEEDS      = [0] if cfg('fast', default=False) else list(range(10))

success, total = 0, 0

for depth in DEPTH_GRID:
    for ds_name in DATASETS:
        for seed in SEEDS:
            total += 1
            print(f"\n[RUN] dataset={ds_name} depth={depth} seed={seed}")
            data = load_dataset(ds_name, split_seed=seed)

            # ---- build model ----
            Model = MODELS['CurvAMP']
            model = Model(data.num_features, cfg('hidden', default=128),
                          int(data.y.max().item())+1, depth,
                          K=3, rewired_ratio=len(data.edge_index[0])//50/len(data.edge_index[0]))

            # ---- train ----
            out = train_model(model, data, seed=seed)
            model = out['model']
            val_curve = out['val_curve']

            # ---- evaluate ----
            device = torch.device("cuda")
            data = data.to(device)
            model.eval()
            with torch.no_grad():
                logits = model(data)
            acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
            gdr = group_distance_ratio(logits.detach(), data.y)
            er  = effective_rank(logits)
            a_res = ater(data.edge_index, data.num_nodes)

            print(json.dumps({
                "accuracy": acc,
                "GDR": gdr,
                "eff_rank": er,
                "ATER": a_res,
                "train_time": out['train_time']
            }, indent=2))

            # ---- plots ----
            plot_curve(val_curve, f"Val-loss {ds_name} depth={depth}",
                       "CE loss", f"valloss_{ds_name}_d{depth}.pdf")

            if ds_name == "Mixture" and acc >= 0.90:
                success += 1

print(f"\nFinished – success {success}/{total}\n")
