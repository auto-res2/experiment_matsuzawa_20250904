from __future__ import annotations
"""src/main.py
Entry-point for CAP-GNN experiments.
Run via:   python -m src.main
"""
import os, sys, random, time, textwrap
from pathlib import Path
from typing import Any, Dict, List
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import numpy as np
import torch
import pandas as pd

# ---------------------------------------------------------------------------
#  Make package root importable when executed from outside project root
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent  # src/
if str(PKG_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PKG_DIR.parent))

# ---------------------------------------------------------------------------
#  Project directories -------------------------------------------------------
# ---------------------------------------------------------------------------
# All experiment artefacts (images, CSVs, …) must be stored in exactly this
# location as mandated by the autograder instructions.
IMG_DIR = (PKG_DIR / "../.research/iteration10/images").resolve()
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Ensure cache directory exists ------------------------------------------------
(PKG_DIR / "cache").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
#  Local imports (after path fix) --------------------------------------------
# ---------------------------------------------------------------------------
from .preprocess import load_planetoid, compute_or_curvature
from .train import GCNStack, train_one
from .evaluate import save_loss_plot

# ---------------------------------------------------------------------------
#  Deterministic behaviour ----------------------------------------------------
# ---------------------------------------------------------------------------
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  Configuration handling -----------------------------------------------------
# ---------------------------------------------------------------------------
CFG_PATH = PKG_DIR.parent / "config" / "config.yaml"
CFG_PATH.parent.mkdir(exist_ok=True, parents=True)
DEFAULT_CFG = {
    "experiment": "CI-SMOKE",
    "datasets": ["Cora"],
    "depth": 16,
    "hidden_dim": 64,
    "epochs": 5,
    "patience": 3,
    "lr": 1e-3,
    "wd": 5e-4,
    "dropout": 0.5,
    "act_beta": 0.01,
    "act_tau": 1.0,
    "variants": ["vanilla", "cap"],
}
if not CFG_PATH.exists():
    yaml.safe_dump(DEFAULT_CFG, CFG_PATH.open("w"))
CFG: Dict[str, Any] = DEFAULT_CFG
# merge user cfg (if exists) on top of defaults
user_cfg = yaml.safe_load(CFG_PATH.read_text()) or {}
CFG.update(user_cfg)

# propagate dropout to model file via environment var (simplest way)
os.environ["CAPGNN_DROPOUT"] = str(CFG["dropout"])

# ---------------------------------------------------------------------------
#  Run experiment (smoke test) ------------------------------------------------
# ---------------------------------------------------------------------------

def run_smoke() -> None:
    print("\n================  CI-SMOKE EXPERIMENT  ================")
    print(textwrap.dedent(
        """
        Purpose: minimal end-to-end run that verifies the CAP-GNN stack can
        execute forward/backward on GPU in CI. Dataset=Cora, Depth=16, Epochs=5.
        """
    ))

    results: List[Dict[str, Any]] = []
    for ds_name in CFG["datasets"]:
        data = load_planetoid(ds_name)
        kappa = compute_or_curvature(data, PKG_DIR / "cache" / f"{ds_name}_kappa.pt")

        for variant in CFG["variants"]:
            model = GCNStack(
                in_dim=data.num_features,
                hidden=CFG["hidden_dim"],
                out_dim=int(data.y.max()) + 1,
                depth=CFG["depth"],
                variant=variant,
                kappa=kappa if variant == "cap" else None,
                act_cfg={"beta": CFG["act_beta"], "tau": CFG["act_tau"]} if variant == "cap" else None,
            )
            # propagate dropout
            model.dropout.p = float(os.environ.get("CAPGNN_DROPOUT", 0.5))

            t0 = time.perf_counter()
            losses, metrics, _ = train_one(
                model, data, CFG["epochs"], CFG["patience"], CFG["lr"], CFG["wd"], device
            )
            t1 = time.perf_counter()
            mem = (
                torch.cuda.max_memory_reserved() / 1024 ** 3
                if torch.cuda.is_available()
                else 0.0
            )

            print(
                f"{variant:<7}  val={metrics['val']:.4f}  test={metrics['test']:.4f}  "
                f"rowdiff={metrics['rowdiff']:.4f}  time={t1 - t0:.1f}s  mem={mem:.2f}GB"
            )

            # record
            results.append({"dataset": ds_name, "variant": variant, **metrics, "time": t1 - t0, "mem": mem})
            # figure
            fig_path = IMG_DIR / f"training_loss_{variant}.pdf"
            save_loss_plot({variant: losses}, fig_path)

    # ---- summary CSV ----
    summary_csv = IMG_DIR / "exp1_summary.csv"
    pd.DataFrame(results).to_csv(summary_csv, index=False)

    print("\nArtifacts written to:")
    for p in IMG_DIR.glob("*.pdf"):
        print("  ", p.relative_to(IMG_DIR.parent.parent))
    print("  ", summary_csv.relative_to(IMG_DIR.parent.parent))


# ---------------------------------------------------------------------------
#  Entrypoint ----------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("Error: CUDA device not available – fail-fast as specified.")
    run_smoke()
