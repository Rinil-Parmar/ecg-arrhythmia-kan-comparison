"""
Evaluation — ECG Arrhythmia Classification
-------------------------------------------
Metrics:
  - Accuracy
  - Macro F1-score
  - Per-class Precision / Recall / F1
  - One-vs-rest ROC-AUC per class + Macro AUC
  - Confusion Matrix
  - Inference time
  - Parameter count

Usage:
    python -m src.training.evaluate
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────
# Class label mapping (MIT-BIH 5-class AAMI)
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


# ─────────────────────────────────────────────
# Collect all predictions on test set
# ─────────────────────────────────────────────
@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        y_true  : [N]        — ground truth labels
        y_pred  : [N]        — predicted labels
        y_proba : [N, C]     — softmax probabilities
    """
    model.eval()
    all_true, all_pred, all_proba = [], [], []

    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        pred   = proba.argmax(axis=1)

        all_true.append(y.numpy())
        all_pred.append(pred)
        all_proba.append(proba)

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )


# ─────────────────────────────────────────────
# Measure inference time
# ─────────────────────────────────────────────
@torch.no_grad()
def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_warmup: int = 5,
) -> dict:
    """
    Returns avg inference time per batch and per sample (ms).
    """
    model.eval()
    times = []

    for i, (X, _) in enumerate(loader):
        X = X.to(device)
        start = time.perf_counter()
        _ = model(X)
        end   = time.perf_counter()

        if i >= n_warmup:          # skip warmup batches
            times.append((end - start) * 1000)  # ms

        if i > n_warmup + 20:      # enough samples
            break

    batch_size     = loader.batch_size
    avg_batch_ms   = float(np.mean(times))
    avg_sample_ms  = avg_batch_ms / batch_size

    return {
        "avg_batch_ms" : round(avg_batch_ms,  3),
        "avg_sample_ms": round(avg_sample_ms, 4),
    }


# ─────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    config: dict,
) -> dict:
    """
    Run full evaluation on test set.

    Args:
        model       : trained ECGCNN or ECGCNNWithKAN
        test_loader : test DataLoader
        config      : dict with keys:
                        num_classes, results_path, model_name

    Returns:
        metrics dict (also saved to JSON)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    num_classes  = config["num_classes"]
    results_path = config["results_path"]
    model_name   = config["model_name"]

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Evaluating : {model_name}")
    print(f"{'='*65}")

    # ── Predictions ───────────────────────────
    y_true, y_pred, y_proba = get_predictions(model, test_loader, device)

    # ── Core metrics ──────────────────────────
    accuracy   = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_true, y_pred, average="weighted", zero_division=0)

    macro_auc  = roc_auc_score(
        y_true, y_proba, multi_class="ovr", average="macro"
    )

    # ── Per-class metrics ─────────────────────
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall    = recall_score(y_true, y_pred,    average=None, zero_division=0)
    per_class_f1        = f1_score(y_true, y_pred,        average=None, zero_division=0)

    per_class_auc = {}
    for c in range(num_classes):
        binary_true = (y_true == c).astype(int)
        try:
            auc = roc_auc_score(binary_true, y_proba[:, c])
        except ValueError:
            auc = float("nan")
        per_class_auc[CLASS_NAMES[c]] = round(float(auc), 4)

    # ── Confusion matrix ──────────────────────
    cm = confusion_matrix(y_true, y_pred)

    # ── Inference time ────────────────────────
    timing = measure_inference_time(model, test_loader, device)

    # ── Parameter count ───────────────────────
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Build results dict ────────────────────
    metrics = {
        "model_name"       : model_name,
        "accuracy"         : round(float(accuracy),    4),
        "macro_f1"         : round(float(macro_f1),    4),
        "weighted_f1"      : round(float(weighted_f1), 4),
        "macro_auc"        : round(float(macro_auc),   4),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": round(float(per_class_precision[c]), 4),
                "recall"   : round(float(per_class_recall[c]),    4),
                "f1"       : round(float(per_class_f1[c]),        4),
                "auc"      : per_class_auc[CLASS_NAMES[c]],
            }
            for c in range(num_classes)
        },
        "confusion_matrix" : cm.tolist(),
        "inference": {
            "avg_batch_ms" : timing["avg_batch_ms"],
            "avg_sample_ms": timing["avg_sample_ms"],
        },
        "parameters": {
            "total"    : total_params,
            "trainable": trainable_params,
        },
    }

    # ── Print summary ─────────────────────────
    print(f"\n  Accuracy    : {accuracy:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Macro AUC   : {macro_auc:.4f}")
    print(f"  Parameters  : {trainable_params:,}")
    print(f"  Inference   : {timing['avg_sample_ms']} ms/sample")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'-'*46}")
    for c in range(num_classes):
        name = CLASS_NAMES[c]
        print(
            f"  {name:<6}"
            f"  {per_class_precision[c]:>9.4f}"
            f"  {per_class_recall[c]:>9.4f}"
            f"  {per_class_f1[c]:>9.4f}"
            f"  {per_class_auc[name]:>9.4f}"
        )
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    print(f"\n{classification_report(y_true, y_pred, target_names=list(CLASS_NAMES.values()), zero_division=0)}")

    # ── Save to JSON ──────────────────────────
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved → {results_path}")

    return metrics


# ─────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────
def load_model(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load saved weights into model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {checkpoint_path}  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return model


# ─────────────────────────────────────────────
# Run evaluation for both models
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader    import create_dataloaders
    from src.models.cnn     import ECGCNN
    from src.models.cnn_kan import ECGCNNWithKAN
    from src.utils.plots    import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_training_history,
        plot_model_comparison,
        plot_perclass_f1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = create_dataloaders(data_dir="data/processed", batch_size=128)

    EXPERIMENTS = [
        {
            "model"          : ECGCNN(num_classes=5),
            "checkpoint_path": "checkpoints/cnn_mlp.pth",
            "model_name"     : "CNN_MLP",
            "results_path"   : "experiments/exp1_cnn_mlp/results.json",
            "plot_dir"       : "experiments/exp1_cnn_mlp",
            "log_path"       : "experiments/exp1_cnn_mlp/train_log.csv",
            "num_classes"    : 5,
        },
        {
            "model"          : ECGCNNWithKAN(num_classes=5),
            "checkpoint_path": "checkpoints/cnn_kan.pth",
            "model_name"     : "CNN_KAN",
            "results_path"   : "experiments/exp2_cnn_kan/results.json",
            "plot_dir"       : "experiments/exp2_cnn_kan",
            "log_path"       : "experiments/exp2_cnn_kan/train_log.csv",
            "num_classes"    : 5,
        },
    ]

    all_results  = {}
    all_y_true   = {}
    all_y_proba  = {}

    for exp in EXPERIMENTS:
        model      = load_model(exp["model"], exp["checkpoint_path"])
        config     = {k: v for k, v in exp.items() if k not in ("model", "plot_dir", "log_path")}
        result     = evaluate(model, test_loader, config)
        all_results[exp["model_name"]] = result

        # Collect predictions for plots
        model = model.to(device)
        y_true, _, y_proba = get_predictions(model, test_loader, device)
        all_y_true[exp["model_name"]]  = y_true
        all_y_proba[exp["model_name"]] = y_proba

        # ── Per-model plots ────────────────────
        print(f"\n  Generating plots for {exp['model_name']} ...")
        cm = np.array(result["confusion_matrix"])

        plot_confusion_matrix(cm, exp["model_name"], exp["plot_dir"])
        plot_roc_curves(y_true, y_proba, exp["model_name"], exp["plot_dir"])
        plot_training_history(exp["log_path"], exp["model_name"], exp["plot_dir"])

    # ── Comparison plots (both models) ────────
    print("\n  Generating comparison plots ...")
    compare_dir = "experiments"
    plot_model_comparison(all_results["CNN_MLP"], all_results["CNN_KAN"], compare_dir)
    plot_perclass_f1(all_results["CNN_MLP"],      all_results["CNN_KAN"], compare_dir)

    # ── Side-by-side comparison summary ───────
    print("\n" + "="*65)
    print("  FINAL COMPARISON")
    print("="*65)
    print(f"  {'Metric':<22} {'CNN+MLP':>12} {'CNN+KAN':>12}")
    print(f"  {'-'*46}")

    metrics_to_compare = ["accuracy", "macro_f1", "weighted_f1", "macro_auc"]
    for m in metrics_to_compare:
        cnn_val = all_results["CNN_MLP"][m]
        kan_val = all_results["CNN_KAN"][m]
        print(f"  {m:<22} {cnn_val:>12.4f} {kan_val:>12.4f}")

    print(f"  {'parameters':<22}"
          f"  {all_results['CNN_MLP']['parameters']['trainable']:>11,}"
          f"  {all_results['CNN_KAN']['parameters']['trainable']:>11,}")
    print(f"  {'inference (ms/sample)':<22}"
          f"  {all_results['CNN_MLP']['inference']['avg_sample_ms']:>12}"
          f"  {all_results['CNN_KAN']['inference']['avg_sample_ms']:>12}")
    print("="*65)
    print("\n  All plots saved to experiments/ folders.")