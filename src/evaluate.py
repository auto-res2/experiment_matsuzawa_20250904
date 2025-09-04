"""
evaluate.py – experiment orchestration, statistical analysis & plotting utilities.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import matplotlib.pyplot as plt

from .preprocess import load_dataset, random_split
from .train import (
    GlobalCfg,
    FrodoHyper,
    GCN,
    norm_factory,
    train_one,
    set_global_seeds,
)

plt.switch_backend("Agg")  # allow plotting on head-less servers

# -----------------------------------------------------------------------------
#  EXPERIMENT 1 – depth scaling benchmark
# -----------------------------------------------------------------------------

def experiment_depth(cfg: GlobalCfg, frodo_cfg: FrodoHyper):
    """Run the depth-scaling benchmark (Exp-1) end-to-end."""
    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "exp1_depth.json"
    fig_dir = Path("figures"); fig_dir.mkdir(exist_ok=True, parents=True)

    rows: List[Dict] = []
    for ds_name in cfg.datasets_e1:
        try:
            data = load_dataset(ds_name, cfg.data_root)
        except Exception as e:  # pragma: no cover – defensive
            warnings.warn(f"Failed to load dataset {ds_name}: {e}")
            continue

        for depth in cfg.depths_e1:
            for variant in cfg.variants_e1:
                for seed in cfg.seeds:
                    set_global_seeds(seed)
                    print(f"[Exp-1] dataset={ds_name} depth={depth} variant={variant} seed={seed}")

                    if ds_name != "ogbn-arxiv":
                        split = random_split(data.num_nodes, seed)
                    else:
                        # ogbn splits pre-defined
                        split = {
                            "train": data.train_mask.nonzero(as_tuple=False).view(-1),
                            "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
                            "test": data.test_mask.nonzero(as_tuple=False).view(-1),
                        }

                    model = GCN(
                        in_dim=data.num_features,
                        hid=cfg.hidden,
                        out_dim=int(data.y.max()) + 1,
                        depth=depth,
                        dropout=cfg.dropout,
                        norm_factory=norm_factory(variant, cfg.hidden, frodo_cfg),
                    )

                    metrics = train_one(model, data, split, cfg)
                    row = {"dataset": ds_name, "depth": depth, "variant": variant, "seed": seed, **metrics}
                    rows.append(row)

                    # persist continuously – safer on long runs
                    with open(csv_path, "w", encoding="utf-8") as fp:
                        json.dump(rows, fp)

    # ---------------- plots ----------------
    if not rows:
        print("[WARN] No results to plot – exiting experiment_depth early.")
        return

    df = pd.DataFrame(rows)
    for metric, fname in [
        ("test_acc", "accuracy_vs_depth.pdf"),
        ("ER", "er_vs_depth.pdf"),
        ("GDR", "gdr_vs_depth.pdf"),
    ]:
        for ds_name in cfg.datasets_e1:
            plt.figure()
            for variant in cfg.variants_e1:
                means: List[float] = []
                for depth in cfg.depths_e1:
                    m = df[(df.dataset == ds_name) & (df.depth == depth) & (df.variant == variant)][metric].mean()
                    means.append(m)
                plt.plot(cfg.depths_e1, means, marker="o", label=variant)
                for d, m in zip(cfg.depths_e1, means):
                    plt.text(d, m, f"{m:.2f}")
            plt.xlabel("Depth")
            plt.ylabel(metric)
            plt.title(f"{metric} – {ds_name}")
            plt.legend()
            plt.tight_layout()
            path = fig_dir / fname.replace(".pdf", f"_{ds_name}.pdf")
            plt.savefig(path, format="pdf", bbox_inches="tight")
            plt.close()
            print(f"Saved figure → {path}")

    # ---------------- console summary ----------------
    print("\nExperiment description: Depth-scaling benchmark (Exp-1) – real training "
          "on Pubmed, Chameleon and ogbn-arxiv with variants {vanilla, pairnorm, "
          "contranorm, ndls, dropedge, frodo} across depths {2,16,64,128} repeated "
          "for 10 seeds.")
    print("Numerical results (first 5 rows):")
    print(df.head())
    print("Figures stored in ./figures/*_vs_depth_*.pdf")
