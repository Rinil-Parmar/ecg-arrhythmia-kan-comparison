"""
main.py — Run both experiments
-------------------------------
Experiment 1: CNN + MLP head (baseline)
Experiment 2: CNN + KAN head (proposed)

Usage:
    python main.py
    python main.py --model cnn
    python main.py --model kan
"""

import os
import argparse
import torch

from src.data.loader         import create_dataloaders
from src.models.cnn          import ECGCNN
from src.models.cnn_kan      import ECGCNNWithKAN
from src.training.train      import train


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_CONFIG = {
    "num_classes"   : 5,
    "epochs"        : 50,
    "lr"            : 1e-3,
    "weight_decay"  : 1e-4,
    "patience"      : 10,
    "batch_size"    : 128,
    "data_dir"      : "data/processed",
}

EXPERIMENTS = {
    "cnn": {
        "checkpoint_path": "checkpoints/cnn_mlp.pth",
        "log_path"       : "experiments/exp1_cnn_mlp/train_log.csv",
    },
    "kan": {
        "checkpoint_path": "checkpoints/cnn_kan.pth",
        "log_path"       : "experiments/exp2_cnn_kan/train_log.csv",
    },
}


# ─────────────────────────────────────────────
def run_experiment(model_type: str):
    print(f"\n{'='*65}")
    print(f"  Experiment : CNN + {'MLP' if model_type == 'cnn' else 'KAN'}")
    print(f"{'='*65}")

    train_loader, val_loader, _ = create_dataloaders(
        data_dir   = BASE_CONFIG["data_dir"],
        batch_size = BASE_CONFIG["batch_size"],
    )

    if model_type == "cnn":
        model = ECGCNN(num_classes=BASE_CONFIG["num_classes"], dropout=0.3)
    else:
        model = ECGCNNWithKAN(num_classes=BASE_CONFIG["num_classes"], dropout=0.3)

    config = {**BASE_CONFIG, **EXPERIMENTS[model_type]}

    history = train(model, train_loader, val_loader, config)
    return history


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cnn", "kan", "both"], default="both",
        help="Which model to train (default: both)"
    )
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments/exp1_cnn_mlp", exist_ok=True)
    os.makedirs("experiments/exp2_cnn_kan", exist_ok=True)

    if args.model in ("cnn", "both"):
        run_experiment("cnn")

    if args.model in ("kan", "both"):
        run_experiment("kan")