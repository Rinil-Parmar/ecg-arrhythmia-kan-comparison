"""
Step 10 — Final Comparison: CNN+MLP vs CNN+KAN
------------------------------------------------
Reads both experiment result JSONs and produces:
  1. experiments/results.csv       — full metrics table
  2. experiments/final_summary.txt — human-readable report
  3. experiments/radar_chart.png   — radar plot comparison
  4. experiments/efficiency.png    — parameters vs performance plot

Usage:
    python -m src.compare
"""

import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
RESULTS = {
    "CNN_MLP": "experiments/exp1_cnn_mlp/results.json",
    "CNN_KAN": "experiments/exp2_cnn_kan/results.json",
}
TRAIN_LOGS = {
    "CNN_MLP": "experiments/exp1_cnn_mlp/train_log.csv",
    "CNN_KAN": "experiments/exp2_cnn_kan/train_log.csv",
}
OUTPUT_DIR   = "experiments"
CSV_PATH     = "experiments/results.csv"
SUMMARY_PATH = "experiments/final_summary.txt"

CLASS_NAMES  = ["N", "S", "V", "F", "Q"]
COLORS       = {"CNN_MLP": "#4C72B0", "CNN_KAN": "#DD8452"}


# ─────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────
def load_results() -> dict:
    results = {}
    for name, path in RESULTS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run evaluate.py first."
            )
        with open(path) as f:
            results[name] = json.load(f)
    return results


# ─────────────────────────────────────────────
# Load training time from log
# ─────────────────────────────────────────────
def get_training_epochs(log_path: str) -> int:
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    return len(rows)


# ─────────────────────────────────────────────
# Save results.csv
# ─────────────────────────────────────────────
def save_results_csv(results: dict):
    rows = []

    for model_name, r in results.items():
        # Overall metrics row
        row = {
            "model"              : model_name,
            "accuracy"           : r["accuracy"],
            "macro_f1"           : r["macro_f1"],
            "weighted_f1"        : r["weighted_f1"],
            "macro_auc"          : r["macro_auc"],
            "parameters"         : r["parameters"]["trainable"],
            "inference_ms_sample": r["inference"]["avg_sample_ms"],
        }
        # Per-class F1
        for c in CLASS_NAMES:
            row[f"f1_{c}"]        = r["per_class"][c]["f1"]
            row[f"precision_{c}"] = r["per_class"][c]["precision"]
            row[f"recall_{c}"]    = r["per_class"][c]["recall"]
            row[f"auc_{c}"]       = r["per_class"][c]["auc"]

        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved → {CSV_PATH}")


# ─────────────────────────────────────────────
# Save final_summary.txt
# ─────────────────────────────────────────────
def save_summary(results: dict):
    r_cnn = results["CNN_MLP"]
    r_kan = results["CNN_KAN"]

    cnn_ep = get_training_epochs(TRAIN_LOGS["CNN_MLP"])
    kan_ep = get_training_epochs(TRAIN_LOGS["CNN_KAN"])

    lines = []
    lines.append("=" * 65)
    lines.append("  FINAL COMPARISON REPORT")
    lines.append("  KAN vs MLP for ECG Arrhythmia Classification")
    lines.append("  Dataset: MIT-BIH Arrhythmia Database (PhysioNet)")
    lines.append("=" * 65)

    lines.append("\n  1. OVERALL METRICS")
    lines.append(f"  {'Metric':<25} {'CNN+MLP':>12} {'CNN+KAN':>12} {'Winner':>10}")
    lines.append(f"  {'-'*59}")

    metrics = [
        ("Accuracy",        "accuracy"),
        ("Macro F1",        "macro_f1"),
        ("Weighted F1",     "weighted_f1"),
        ("Macro AUC",       "macro_auc"),
    ]
    for label, key in metrics:
        cnn_v = r_cnn[key]
        kan_v = r_kan[key]
        winner = "CNN+MLP" if cnn_v >= kan_v else "CNN+KAN"
        lines.append(f"  {label:<25} {cnn_v:>12.4f} {kan_v:>12.4f} {winner:>10}")

    lines.append("\n  2. EFFICIENCY")
    lines.append(f"  {'Metric':<25} {'CNN+MLP':>12} {'CNN+KAN':>12} {'Winner':>10}")
    lines.append(f"  {'-'*59}")

    cnn_p = r_cnn["parameters"]["trainable"]
    kan_p = r_kan["parameters"]["trainable"]
    cnn_i = r_cnn["inference"]["avg_sample_ms"]
    kan_i = r_kan["inference"]["avg_sample_ms"]

    winner_p = "CNN+KAN" if kan_p <= cnn_p else "CNN+MLP"
    winner_i = "CNN+KAN" if kan_i <= cnn_i else "CNN+MLP"

    lines.append(f"  {'Parameters':<25} {cnn_p:>12,} {kan_p:>12,} {winner_p:>10}")
    lines.append(f"  {'Inference (ms/sample)':<25} {cnn_i:>12} {kan_i:>12} {winner_i:>10}")
    lines.append(f"  {'Training Epochs':<25} {cnn_ep:>12} {kan_ep:>12} {'—':>10}")

    lines.append("\n  3. PER-CLASS F1")
    lines.append(f"  {'Class':<10} {'CNN+MLP':>12} {'CNN+KAN':>12} {'Winner':>10}")
    lines.append(f"  {'-'*44}")
    for c in CLASS_NAMES:
        cnn_v = r_cnn["per_class"][c]["f1"]
        kan_v = r_kan["per_class"][c]["f1"]
        winner = "CNN+MLP" if cnn_v >= kan_v else "CNN+KAN"
        lines.append(f"  {c:<10} {cnn_v:>12.4f} {kan_v:>12.4f} {winner:>10}")

    lines.append("\n  4. PER-CLASS AUC")
    lines.append(f"  {'Class':<10} {'CNN+MLP':>12} {'CNN+KAN':>12} {'Winner':>10}")
    lines.append(f"  {'-'*44}")
    for c in CLASS_NAMES:
        cnn_v = r_cnn["per_class"][c]["auc"]
        kan_v = r_kan["per_class"][c]["auc"]
        winner = "CNN+MLP" if cnn_v >= kan_v else "CNN+KAN"
        lines.append(f"  {c:<10} {cnn_v:>12.4f} {kan_v:>12.4f} {winner:>10}")

    lines.append("\n  5. CONCLUSION")
    lines.append("  " + "-"*60)

    # Auto-generate conclusion based on results
    cnn_wins = sum(1 for _, k in metrics if r_cnn[k] >= r_kan[k])
    kan_wins = len(metrics) - cnn_wins

    if cnn_wins > kan_wins:
        conclusion = (
            f"  CNN+MLP outperforms CNN+KAN across {cnn_wins}/{len(metrics)} metrics.\n"
            f"  KAN achieves competitive Macro AUC ({r_kan['macro_auc']:.3f}) but lower\n"
            f"  Macro F1 ({r_kan['macro_f1']:.3f} vs {r_cnn['macro_f1']:.3f}), suggesting\n"
            f"  KAN requires more data or tuning to match MLP on this task.\n"
            f"  KAN's parameter count ({kan_p:,}) vs MLP ({cnn_p:,}) shows its\n"
            f"  efficiency trade-off potential with further optimization."
        )
    else:
        conclusion = (
            f"  CNN+KAN outperforms CNN+MLP across {kan_wins}/{len(metrics)} metrics.\n"
            f"  KAN's learnable spline activations provide better decision\n"
            f"  boundaries for ECG classification on this dataset.\n"
            f"  This validates KAN as a viable alternative to MLP heads\n"
            f"  in biomedical time-series classification."
        )

    lines.append(conclusion)
    lines.append("\n  [!] All results on held-out test set (never seen during training).")
    lines.append("=" * 65)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also print to terminal
    print("\n" + "\n".join(lines))
    print(f"\n  Saved → {SUMMARY_PATH}")


# ─────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────
def plot_radar_chart(results: dict):
    categories  = ["Accuracy", "Macro F1", "Weighted F1", "Macro AUC",
                   "F1-N", "F1-S", "F1-V", "F1-F", "F1-Q"]

    def get_values(r):
        return [
            r["accuracy"], r["macro_f1"], r["weighted_f1"], r["macro_auc"],
            r["per_class"]["N"]["f1"], r["per_class"]["S"]["f1"],
            r["per_class"]["V"]["f1"], r["per_class"]["F"]["f1"],
            r["per_class"]["Q"]["f1"],
        ]

    cnn_vals = get_values(results["CNN_MLP"])
    kan_vals = get_values(results["CNN_KAN"])

    N     = len(categories)
    angles= [n / float(N) * 2 * pi for n in range(N)]
    angles+= angles[:1]

    cnn_vals += cnn_vals[:1]
    kan_vals += kan_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, cnn_vals, "o-", lw=2, color=COLORS["CNN_MLP"], label="CNN+MLP")
    ax.fill(angles, cnn_vals, alpha=0.15, color=COLORS["CNN_MLP"])

    ax.plot(angles, kan_vals, "o-", lw=2, color=COLORS["CNN_KAN"], label="CNN+KAN")
    ax.fill(angles, kan_vals, alpha=0.15, color=COLORS["CNN_KAN"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("CNN+MLP vs CNN+KAN — Radar Comparison",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "radar_chart.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# Efficiency plot (params vs macro F1)
# ─────────────────────────────────────────────
def plot_efficiency(results: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, r in results.items():
        params = r["parameters"]["trainable"]
        f1     = r["macro_f1"]
        infer  = r["inference"]["avg_sample_ms"]
        color  = COLORS[name]

        ax.scatter(params, f1, s=infer * 300, color=color,
                   alpha=0.85, edgecolors="black", linewidths=1.5,
                   label=f"{name}  (params={params:,}, infer={infer}ms)")
        ax.annotate(name, (params, f1),
                    textcoords="offset points", xytext=(10, 5), fontsize=11)

    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Macro F1 Score",       fontsize=12)
    ax.set_title("Efficiency Comparison: Parameters vs Performance\n"
                 "(bubble size = inference time)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "efficiency.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading results ...")
    results = load_results()

    print("Saving results.csv ...")
    save_results_csv(results)

    print("Generating final summary ...")
    save_summary(results)

    print("\nGenerating plots ...")
    plot_radar_chart(results)
    plot_efficiency(results)

    print("\n" + "="*65)
    print("  Step 10 Complete. All outputs saved to experiments/")
    print("="*65)