"""src/main.py
Main orchestrator.  Run `python -m src.main`.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, subprocess, sys as _sys
import torch

# -----------------------------------------------------------------------------
# Ensure PyG (torch_geometric) & torch_scatter are present BEFORE importing
# training/evaluation modules that depend on them.
# -----------------------------------------------------------------------------
try:
    import torch_geometric  # noqa: F401 – just a presence check
except ImportError:  # pragma: no cover – install only when missing
    ver_base   = torch.__version__.split("+")[0]
    backend    = torch.__version__.split("+")[1] if "+" in torch.__version__ else "cpu"
    pyg_wheels = f"https://data.pyg.org/whl/torch-{ver_base}+{backend}.html"
    cmd = [
        _sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "torch_scatter==2.1.2", "torch_geometric==2.6.1", "-f", pyg_wheels
    ]
    subprocess.check_call(cmd)
    import importlib; importlib.invalidate_caches()  # noqa: E702, F401
    import torch_geometric  # noqa: F401 – re-import to confirm availability

# -----------------------------------------------------------------------------
# Now that dependencies are resolved we can safely import internals.
# -----------------------------------------------------------------------------
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

# Ensure the mandatory image directory exists (iteration26 as per spec)
IMG_DIR = Path(".research/iteration26/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

success, total = 0, 0

for depth in DEPTH_GRID:
    for ds_name in DATASETS:
        for seed in SEEDS:
            total += 1
            print(f"\n[RUN] dataset={ds_name} depth={depth} seed={seed}")
            data = load_dataset(ds_name, split_seed=seed)

            # ---- build model ----
            Model = MODELS['CurvAMP']
            model = Model(
                data.num_features,
                cfg('hidden', default=128),
                int(data.y.max().item())+1,
                depth,
                K=3,
                rewired_ratio=len(data.edge_index[0])//50/len(data.edge_index[0])
            )

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
            plot_path = IMG_DIR / f"valloss_{ds_name}_d{depth}.pdf"
            plot_curve(val_curve, f"Val-loss {ds_name} depth={depth}", "CE loss", str(plot_path))

            if ds_name == "Mixture" and acc >= 0.90:
                success += 1

print(f"\nFinished – success {success}/{total}\n")
