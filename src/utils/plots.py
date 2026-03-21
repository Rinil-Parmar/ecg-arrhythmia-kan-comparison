"""
Plots — ECG Arrhythmia Evaluation Graphs
-----------------------------------------
Generates and saves all evaluation plots:
  1. Confusion Matrix (per model)
  2. ROC Curve — one-vs-rest per class (per model)
  3. Training History — loss & accuracy curves (per model)
  4. Side-by-side metric comparison bar chart (both models)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# ─────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label",      fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# 2. ROC Curves (one-vs-rest per class)
# ─────────────────────────────────────────────
def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_dir: str,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    for c, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        binary_true = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_proba[:, c])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"Class {name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"roc_curves_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# 3. Training History (loss + accuracy)
# ─────────────────────────────────────────────
def plot_training_history(log_path: str, model_name: str, save_dir: str):
    if not os.path.exists(log_path):
        print(f"  No log file found at {log_path}, skipping.")
        return

    import csv
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training History — {model_name}", fontsize=14, fontweight="bold")

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color=COLORS[0], lw=2)
    ax1.plot(epochs, val_loss,   label="Val Loss",   color=COLORS[1], lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss",  fontsize=12)
    ax1.set_title("Loss Curve")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_acc, label="Train Acc", color=COLORS[2], lw=2)
    ax2.plot(epochs, val_acc,   label="Val Acc",   color=COLORS[3], lw=2, linestyle="--")
    ax2.set_xlabel("Epoch",    fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curve")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"training_history_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# 4. Side-by-side Model Comparison Bar Chart
# ─────────────────────────────────────────────
def plot_model_comparison(results_cnn: dict, results_kan: dict, save_dir: str):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "macro_auc"]
    labels  = ["Accuracy", "Macro F1", "Weighted F1", "Macro AUC"]

    cnn_vals = [results_cnn[m] for m in metrics]
    kan_vals = [results_kan[m] for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, cnn_vals, width, label="CNN + MLP", color=COLORS[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, kan_vals, width, label="CNN + KAN", color=COLORS[1], alpha=0.85)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.set_title("CNN+MLP vs CNN+KAN — Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# 5. Per-class F1 Comparison
# ─────────────────────────────────────────────
def plot_perclass_f1(results_cnn: dict, results_kan: dict, save_dir: str):
    cnn_f1 = [results_cnn["per_class"][c]["f1"] for c in CLASS_NAMES]
    kan_f1 = [results_kan["per_class"][c]["f1"] for c in CLASS_NAMES]

    x     = np.arange(len(CLASS_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, cnn_f1, width, label="CNN + MLP", color=COLORS[0], alpha=0.85)
    ax.bar(x + width/2, kan_f1, width, label="CNN + KAN", color=COLORS[1], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=13)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "perclass_f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")