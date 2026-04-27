"""
Streamlit UI — ECG Arrhythmia Classification
CNN+MLP vs CNN+KAN Comparison Dashboard
"""

import os
import json
import csv
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc as sk_auc
import streamlit as st
from datetime import datetime

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG Arrhythmia · KAN vs MLP",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark clinical theme */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0f1629 0%, #1a2744 50%, #0f1629 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #0066ff, #6600ff);
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.header-sub {
    font-size: 14px;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00d4ff44; }
.metric-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: #00d4ff;
    line-height: 1;
}
.metric-value.green  { color: #22c55e; }
.metric-value.orange { color: #f59e0b; }
.metric-value.red    { color: #ef4444; }
.metric-delta {
    font-size: 12px;
    color: #64748b;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 10px;
    margin: 28px 0 18px 0;
}

/* Clinical output cards */
.clinical-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-left: 4px solid #00d4ff;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.clinical-card.urgent    { border-left-color: #ef4444; }
.clinical-card.priority  { border-left-color: #f59e0b; }
.clinical-card.routine   { border-left-color: #22c55e; }
.clinical-card.warning   { border-left-color: #f59e0b; }

.clinical-label {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 6px;
}
.clinical-value {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
}
.clinical-sub {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Triage badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-urgent   { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }
.badge-priority { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
.badge-routine  { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }

/* Disclaimer */
.disclaimer {
    background: #1a1a0f;
    border: 1px solid #f59e0b44;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 12px;
    color: #f59e0b;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 20px;
}

/* Table styling */
.results-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.results-table th {
    background: #1e2d4a;
    color: #00d4ff;
    padding: 10px 16px;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.results-table td {
    padding: 10px 16px;
    border-bottom: 1px solid #1e2d4a;
    color: #e2e8f0;
}
.results-table tr:hover td { background: #1e2d4a44; }
.winner { color: #22c55e; font-weight: 600; }

/* Progress bar override */
.stProgress > div > div { background-color: #00d4ff; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1629;
    border-bottom: 1px solid #1e2d4a;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #64748b;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Selectbox / inputs */
.stSelectbox label, .stSlider label, .stRadio label {
    color: #94a3b8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Hide default streamlit menu */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLASS_DESCRIPTIONS = {
    "N": "Normal / Non-ectopic beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown / Unclassifiable beat",
}
CLASS_COLORS = {
    "N": "#22c55e",
    "S": "#f59e0b",
    "V": "#ef4444",
    "F": "#8b5cf6",
    "Q": "#64748b",
}
TRIAGE_POLICY = {
    "N": "Routine",
    "S": "Priority Review",
    "V": "Urgent Review",
    "F": "Urgent Review",
    "Q": "Priority Review",
}
CONFIDENCE_HIGH = 0.85
CONFIDENCE_LOW  = 0.60

RESULT_PATHS = {
    "CNN_MLP": "experiments/exp1_cnn_mlp/results.json",
    "CNN_KAN": "experiments/exp2_cnn_kan/results.json",
}
LOG_PATHS = {
    "CNN_MLP": "experiments/exp1_cnn_mlp/train_log.csv",
    "CNN_KAN": "experiments/exp2_cnn_kan/train_log.csv",
}
CHECKPOINT_PATHS = {
    "CNN_MLP": "checkpoints/cnn_mlp.pth",
    "CNN_KAN": "checkpoints/cnn_kan.pth",
}


# ══════════════════════════════════════════════
# Helpers — load data
# ══════════════════════════════════════════════
@st.cache_data
def load_results():
    results = {}
    for name, path in RESULT_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results


@st.cache_data
def load_train_log(model_name):
    path = LOG_PATHS[model_name]
    if not os.path.exists(path):
        return None
    epochs, tl, vl, ta, va = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tl.append(float(row["train_loss"]))
            vl.append(float(row["val_loss"]))
            ta.append(float(row["train_acc"]))
            va.append(float(row["val_acc"]))
    return {"epoch": epochs, "train_loss": tl, "val_loss": vl,
            "train_acc": ta, "val_acc": va}


@st.cache_data
def load_ecg_data():
    X_path = "data/processed/X.npy"
    y_path = "data/processed/y.npy"
    if os.path.exists(X_path) and os.path.exists(y_path):
        return np.load(X_path), np.load(y_path)
    return None, None


@st.cache_resource
def load_model(model_type: str):
    try:
        from src.models.cnn     import ECGCNN
        from src.models.cnn_kan import ECGCNNWithKAN

        if model_type == "CNN_MLP":
            model = ECGCNN(num_classes=5)
        else:
            model = ECGCNNWithKAN(num_classes=5)

        ckpt = torch.load(CHECKPOINT_PATHS[model_type],
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    except Exception:
        return None


# ══════════════════════════════════════════════
# Helpers — signal quality
# ══════════════════════════════════════════════
def assess_signal_quality(signal):
    std_val = float(np.std(signal))
    max_amp = float(np.max(np.abs(signal)))
    flatline = std_val < 0.01
    high_amp = max_amp > 4.0
    bw = float(np.std(np.diff(signal))) < 0.005
    if flatline or high_amp: sqi = "Poor"
    elif bw or std_val < 0.05: sqi = "Moderate"
    else: sqi = "Good"
    flags = []
    if flatline:  flags.append("Flatline")
    if high_amp:  flags.append("High amplitude")
    if bw:        flags.append("Baseline wander")
    return sqi, flags if flags else ["None"]


# ══════════════════════════════════════════════
# Helpers — prediction
# ══════════════════════════════════════════════
def run_prediction(model, signal):
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        t0     = time.perf_counter()
        logits = model(x)
        infer_ms = (time.perf_counter() - t0) * 1000
    proba    = torch.softmax(logits, dim=1).numpy()[0]
    pred_idx = int(proba.argmax())
    return proba, pred_idx, infer_ms


def get_saliency(model, signal):
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x.requires_grad_(True)
    logits = model(x)
    pred   = logits.argmax(dim=1).item()
    logits[0, pred].backward()
    sal = x.grad.abs().squeeze().detach().numpy()
    return sal


# ══════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════
PLOT_BG    = "#0a0e1a"
PLOT_FG    = "#e2e8f0"
PLOT_GRID  = "#1e2d4a"
PLOT_CYAN  = "#00d4ff"
PLOT_BLUE  = "#4C72B0"
PLOT_ORANGE= "#DD8452"

def fig_style(fig, ax_list):
    fig.patch.set_facecolor(PLOT_BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_FG, labelsize=9)
        ax.xaxis.label.set_color(PLOT_FG)
        ax.yaxis.label.set_color(PLOT_FG)
        ax.title.set_color(PLOT_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(PLOT_GRID)
        ax.grid(color=PLOT_GRID, linewidth=0.5, alpha=0.6)


def plot_ecg_signal(signal, saliency=None, pred_class="N", evidence_windows=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig_style(fig, ax)
    fs = 360
    t  = np.arange(len(signal)) / fs

    if saliency is not None:
        sal_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        ax.fill_between(t, signal - 0.05, signal + 0.05,
                        alpha=sal_norm * 0.5, color="#ef4444")

    ax.plot(t, signal, color=CLASS_COLORS.get(pred_class, PLOT_CYAN), lw=1.5)

    if evidence_windows:
        for i, w in enumerate(evidence_windows):
            ax.axvspan(w["start_s"], w["end_s"],
                       alpha=0.2,
                       color="#ef4444" if i == 0 else "#f59e0b",
                       label=f"Window {i+1}")

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_title(f"ECG Beat Segment — Class: {pred_class} ({CLASS_DESCRIPTIONS.get(pred_class,'')})",
                 fontsize=11, fontweight="bold")
    if evidence_windows:
        ax.legend(fontsize=9, facecolor=PLOT_BG,
                  labelcolor=PLOT_FG, edgecolor=PLOT_GRID)
    plt.tight_layout()
    return fig


def plot_prob_bars(proba):
    fig, ax = plt.subplots(figsize=(6, 3))
    fig_style(fig, ax)
    classes = [CLASS_NAMES[i] for i in range(5)]
    colors  = [CLASS_COLORS[c] for c in classes]
    bars    = ax.barh(classes, proba, color=colors, alpha=0.85, height=0.55)
    for bar, p in zip(bars, proba):
        ax.text(min(p + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=9,
                color=PLOT_FG, fontfamily="monospace")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Probability", fontsize=10)
    ax.set_title("Class Probabilities", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_training_curves(log_data, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig_style(fig, [ax1, ax2])
    color = PLOT_BLUE if model_name == "CNN_MLP" else PLOT_ORANGE
    ep = log_data["epoch"]
    ax1.plot(ep, log_data["train_loss"], color=color,    lw=2,  label="Train")
    ax1.plot(ep, log_data["val_loss"],   color=PLOT_CYAN, lw=2, ls="--", label="Val")
    ax1.set_title("Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(facecolor=PLOT_BG, labelcolor=PLOT_FG, edgecolor=PLOT_GRID, fontsize=9)
    ax2.plot(ep, log_data["train_acc"], color=color,    lw=2,  label="Train")
    ax2.plot(ep, log_data["val_acc"],   color=PLOT_CYAN, lw=2, ls="--", label="Val")
    ax2.set_title("Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(facecolor=PLOT_BG, labelcolor=PLOT_FG, edgecolor=PLOT_GRID, fontsize=9)
    fig.suptitle(f"Training History — {model_name}", fontsize=12,
                 fontweight="bold", color=PLOT_FG)
    plt.tight_layout()
    return fig


def plot_comparison_bars(results):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "macro_auc"]
    labels  = ["Accuracy", "Macro F1", "Weighted F1", "Macro AUC"]
    cnn_v   = [results["CNN_MLP"][m] for m in metrics]
    kan_v   = [results["CNN_KAN"][m] for m in metrics]
    x, w    = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig_style(fig, ax)
    b1 = ax.bar(x - w/2, cnn_v, w, label="CNN+MLP", color=PLOT_BLUE,   alpha=0.85)
    b2 = ax.bar(x + w/2, kan_v, w, label="CNN+KAN", color=PLOT_ORANGE, alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9,
                color=PLOT_FG, fontfamily="monospace")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_title("CNN+MLP vs CNN+KAN — Performance Comparison",
                 fontsize=12, fontweight="bold")
    ax.legend(facecolor=PLOT_BG, labelcolor=PLOT_FG, edgecolor=PLOT_GRID, fontsize=10)
    plt.tight_layout()
    return fig


def plot_perclass_f1(results):
    cls     = ["N", "S", "V", "F", "Q"]
    cnn_f1  = [results["CNN_MLP"]["per_class"][c]["f1"] for c in cls]
    kan_f1  = [results["CNN_KAN"]["per_class"][c]["f1"] for c in cls]
    x, w    = np.arange(len(cls)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    fig_style(fig, ax)
    ax.bar(x - w/2, cnn_f1, w, label="CNN+MLP", color=PLOT_BLUE,   alpha=0.85)
    ax.bar(x + w/2, kan_f1, w, label="CNN+KAN", color=PLOT_ORANGE, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cls, fontsize=13)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score", fontsize=12, fontweight="bold")
    ax.legend(facecolor=PLOT_BG, labelcolor=PLOT_FG, edgecolor=PLOT_GRID, fontsize=10)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm_list, model_name):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig_style(fig, ax)
    im     = ax.imshow(cm, cmap="Blues")
    cb     = plt.colorbar(im, ax=ax)
    cb.ax.tick_params(colors=PLOT_FG, labelsize=8)
    cls    = ["N", "S", "V", "F", "Q"]
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(cls, fontsize=11); ax.set_yticklabels(cls, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=9, color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    return fig


def plot_radar(results):
    cats  = ["Accuracy", "Macro F1", "Weighted F1", "Macro AUC",
             "F1-N", "F1-S", "F1-V", "F1-F", "F1-Q"]
    def vals(r):
        return [r["accuracy"], r["macro_f1"], r["weighted_f1"], r["macro_auc"],
                r["per_class"]["N"]["f1"], r["per_class"]["S"]["f1"],
                r["per_class"]["V"]["f1"], r["per_class"]["F"]["f1"],
                r["per_class"]["Q"]["f1"]]
    N      = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    cv     = vals(results["CNN_MLP"]) + [vals(results["CNN_MLP"])[0]]
    kv     = vals(results["CNN_KAN"]) + [vals(results["CNN_KAN"])[0]]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.plot(angles, cv, "o-", lw=2, color=PLOT_BLUE,   label="CNN+MLP")
    ax.fill(angles, cv, alpha=0.15, color=PLOT_BLUE)
    ax.plot(angles, kv, "o-", lw=2, color=PLOT_ORANGE, label="CNN+KAN")
    ax.fill(angles, kv, alpha=0.15, color=PLOT_ORANGE)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=9, color=PLOT_FG)
    ax.set_ylim(0, 1)
    ax.tick_params(colors=PLOT_FG)
    ax.grid(color=PLOT_GRID, linewidth=0.5)
    ax.spines["polar"].set_color(PLOT_GRID)
    ax.set_title("Radar Comparison", fontsize=12, fontweight="bold",
                 color=PLOT_FG, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=10, facecolor=PLOT_BG, labelcolor=PLOT_FG,
              edgecolor=PLOT_GRID)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px 0;'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:18px;
                    color:#00d4ff;font-weight:600;'>🫀 ECG·KAN</div>
        <div style='font-size:11px;color:#64748b;margin-top:4px;
                    font-family:IBM Plex Mono,monospace;letter-spacing:1px;'>
            ARRHYTHMIA DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏠  Overview",
         "🔬  Live Prediction",
         "📊  Model Comparison",
         "📈  Training Analysis",
         "🗂   Results Table"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#64748b;font-family:IBM Plex Mono,monospace;
                line-height:1.8;'>
        <b style='color:#94a3b8;'>DATASET</b><br>MIT-BIH Arrhythmia<br>PhysioNet<br><br>
        <b style='color:#94a3b8;'>MODELS</b><br>CNN + MLP (baseline)<br>CNN + KAN (proposed)<br><br>
        <b style='color:#94a3b8;'>CLASSES</b><br>
        <span style='color:#22c55e;'>■</span> N — Normal<br>
        <span style='color:#f59e0b;'>■</span> S — Supraventricular<br>
        <span style='color:#ef4444;'>■</span> V — Ventricular<br>
        <span style='color:#8b5cf6;'>■</span> F — Fusion<br>
        <span style='color:#64748b;'>■</span> Q — Unknown<br>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class='header-banner'>
    <div class='header-title'>ECG Arrhythmia Classification</div>
    <div class='header-sub'>KAN vs MLP · MIT-BIH Arrhythmia Database · PhysioNet</div>
    <div style='margin-top:14px;font-size:13px;color:#94a3b8;max-width:700px;line-height:1.6;'>
        Comparative study of <b style='color:#00d4ff;'>Kolmogorov-Arnold Networks (KAN)</b>
        vs traditional <b style='color:#4C72B0;'>MLP</b> classification heads on top of a
        shared 1D CNN encoder for automated ECG arrhythmia detection.
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════
if "Overview" in page:
    results = load_results()

    if not results:
        st.warning("No results found. Run `python -m src.training.evaluate` first.")
    else:
        # Top KPI cards
        r_cnn = results.get("CNN_MLP", {})
        r_kan = results.get("CNN_KAN", {})

        st.markdown("<div class='section-header'>KEY METRICS — CNN+MLP (BASELINE)</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics_display = [
            (c1, "Accuracy",    f"{r_cnn.get('accuracy',0):.3f}",    "green"),
            (c2, "Macro F1",    f"{r_cnn.get('macro_f1',0):.3f}",    "green"),
            (c3, "Weighted F1", f"{r_cnn.get('weighted_f1',0):.3f}", "green"),
            (c4, "Macro AUC",   f"{r_cnn.get('macro_auc',0):.3f}",   "green"),
            (c5, "Parameters",  f"{r_cnn.get('parameters',{}).get('trainable',0):,}", ""),
        ]
        for col, label, val, color in metrics_display:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value {color}'>{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>KEY METRICS — CNN+KAN (PROPOSED)</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics_display2 = [
            (c1, "Accuracy",    f"{r_kan.get('accuracy',0):.3f}",    "orange"),
            (c2, "Macro F1",    f"{r_kan.get('macro_f1',0):.3f}",    "orange"),
            (c3, "Weighted F1", f"{r_kan.get('weighted_f1',0):.3f}", "orange"),
            (c4, "Macro AUC",   f"{r_kan.get('macro_auc',0):.3f}",   "orange"),
            (c5, "Parameters",  f"{r_kan.get('parameters',{}).get('trainable',0):,}", ""),
        ]
        for col, label, val, color in metrics_display2:
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value {color}'>{val}</div>
            </div>""", unsafe_allow_html=True)

        # Comparison chart + radar
        st.markdown("<div class='section-header'>PERFORMANCE COMPARISON</div>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.pyplot(plot_comparison_bars(results))
        with col_b:
            st.pyplot(plot_radar(results))

        # Per-class F1
        st.markdown("<div class='section-header'>PER-CLASS F1 SCORE</div>",
                    unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.pyplot(plot_perclass_f1(results))
        with col_r:
            st.markdown("""
            <div style='padding:20px;'>
            <div class='section-header' style='margin-top:0;'>ABOUT THIS PROJECT</div>
            <div style='font-size:13px;color:#94a3b8;line-height:1.9;'>
            <b style='color:#e2e8f0;'>Dataset</b><br>
            MIT-BIH Arrhythmia Database (PhysioNet)<br>
            109,446 beat segments · 5 classes (AAMI standard)<br><br>
            <b style='color:#e2e8f0;'>Architecture</b><br>
            Shared 4-block 1D CNN encoder<br>
            Input: [B, 1, 256] → Features: [B, 256]<br><br>
            <b style='color:#e2e8f0;'>MLP Head</b><br>
            FC(256→128) → BN → ReLU → FC(128→64) → FC(64→5)<br><br>
            <b style='color:#e2e8f0;'>KAN Head</b><br>
            KANLayer(256→64) → KANLayer(64→5)<br>
            B-spline order: 3 · Grid size: 5<br><br>
            <b style='color:#e2e8f0;'>Training</b><br>
            Adam · LR=1e-3 · Class-weighted CrossEntropy<br>
            Early stopping (patience=10)
            </div>
            </div>
            """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class='disclaimer'>
        [!] CLINICAL DISCLAIMER — This tool is decision-support only and not a standalone
        diagnostic system. Final interpretation must be made by a licensed clinician.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: Live Prediction
# ══════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown("<div class='section-header'>LIVE ECG PREDICTION</div>",
                unsafe_allow_html=True)

    # ── Input source tabs ─────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📂  Dataset Sample",
        "📄  Upload .npy / .csv",
        "🖼   Upload ECG Image",
    ])

    signal      = None
    input_label = None   # ground truth if known

    # ─────────────────────────────────────────
    # TAB 1: Dataset sample (index slider)
    # ─────────────────────────────────────────
    with tab1:
        X, y = load_ecg_data()
        if X is None:
            st.info(
                "Dataset files are not available in this deployment "
                "(data files excluded to keep the repository lightweight). "
                "Use **Tab 2** to upload a .npy or .csv ECG signal file, "
                "or **Tab 3** to upload an ECG image. "
                "Sample files for each arrhythmia class are in `test_samples/` in the repository."
            )
        else:
            col_ctrl, col_info = st.columns([2, 1])
            with col_ctrl:
                sample_idx = st.slider(
                    "SAMPLE INDEX — pick any beat from your dataset",
                    min_value=0, max_value=len(X)-1, value=0, step=1,
                    help="Selects one ECG beat segment (256 samples) from your preprocessed MIT-BIH data"
                )
            with col_info:
                true_class = CLASS_NAMES[int(y[sample_idx])]
                st.markdown(f"""
                <div class='clinical-card'>
                    <div class='clinical-label'>Ground Truth Label</div>
                    <div class='clinical-value' style='color:{CLASS_COLORS[true_class]};
                         font-size:28px;font-family:IBM Plex Mono,monospace;'>
                        {true_class}
                    </div>
                    <div class='clinical-sub'>{CLASS_DESCRIPTIONS[true_class]}</div>
                </div>
                <div class='clinical-card'>
                    <div class='clinical-label'>Dataset Size</div>
                    <div class='clinical-value' style='font-size:22px;'>{len(X):,}</div>
                    <div class='clinical-sub'>Selected index: {sample_idx}</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("Use this sample", key="btn_dataset"):
                st.session_state["signal"]      = X[sample_idx]
                st.session_state["input_label"] = CLASS_NAMES[int(y[sample_idx])]
                st.session_state["input_source"]= f"Dataset sample #{sample_idx}"
                st.success(f"Sample {sample_idx} loaded. Scroll down or go back to predict.")

    # ─────────────────────────────────────────
    # TAB 2: Upload .npy or .csv
    # ─────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class='clinical-card' style='margin-bottom:16px;'>
            <div class='clinical-label'>Accepted Formats</div>
            <div class='clinical-sub' style='line-height:2;'>
                <b style='color:#e2e8f0;'>.npy</b> — NumPy array of shape [256] or [N, 256]<br>
                <b style='color:#e2e8f0;'>.csv</b> — One row of 256 comma-separated float values<br>
                <br>
                Signal must be <b style='color:#00d4ff;'>normalized</b> (z-score) and
                <b style='color:#00d4ff;'>256 samples</b> long — same preprocessing as training data.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload ECG signal file",
            type=["npy", "csv"],
            key="upload_signal",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".npy"):
                    arr = np.load(uploaded_file, allow_pickle=False)
                else:  # csv
                    import io
                    content = uploaded_file.read().decode("utf-8")
                    arr = np.array([float(x) for x in content.strip().split(",")])

                # Handle [N, 256] — let user pick row
                if arr.ndim == 2:
                    st.info(f"File contains {arr.shape[0]} segments. Pick one:")
                    row_idx = st.slider("Segment index", 0, arr.shape[0]-1, 0,
                                        key="csv_row")
                    arr = arr[row_idx]

                if arr.shape[0] != 256:
                    # Resample to 256
                    from scipy.signal import resample
                    arr = resample(arr, 256)
                    st.warning(f"Signal resampled from {arr.shape[0]} → 256 samples.")

                # Z-score normalize
                arr = (arr - arr.mean()) / (arr.std() + 1e-8)

                st.success(f"Signal loaded: shape {arr.shape}, "
                           f"mean={arr.mean():.3f}, std={arr.std():.3f}")

                # Preview plot
                fig_prev, ax_prev = plt.subplots(figsize=(10, 2.5))
                fig_style(fig_prev, ax_prev)
                ax_prev.plot(np.arange(256)/360, arr, color=PLOT_CYAN, lw=1.5)
                ax_prev.set_title("Uploaded Signal Preview", fontsize=11,
                                  fontweight="bold")
                ax_prev.set_xlabel("Time (s)"); ax_prev.set_ylabel("Amplitude")
                plt.tight_layout()
                st.pyplot(fig_prev)

                if st.button("Use this signal", key="btn_file"):
                    st.session_state["signal"]       = arr
                    st.session_state["input_label"]  = "Unknown (uploaded)"
                    st.session_state["input_source"] = f"Uploaded: {uploaded_file.name}"
                    st.success("Signal ready for prediction.")

            except Exception as e:
                st.error(f"Failed to load file: {e}")

    # ─────────────────────────────────────────
    # TAB 3: Upload ECG Image
    # ─────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class='clinical-card' style='margin-bottom:16px;'>
            <div class='clinical-label'>How Image Digitization Works</div>
            <div class='clinical-sub' style='line-height:2;'>
                1. Upload a <b style='color:#e2e8f0;'>clean single-lead ECG image</b> (PNG/JPG)<br>
                2. OpenCV detects the waveform trace by finding the darkest/most prominent curve<br>
                3. Signal is extracted column-by-column (y-position of trace per x-pixel)<br>
                4. Resampled to <b style='color:#00d4ff;'>256 samples</b> and z-score normalized<br>
                5. Passed to model for prediction<br><br>
                <b style='color:#f59e0b;'>Best results:</b> Clean background, single visible ECG trace,
                no grid lines or annotations overlapping the signal.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer' style='margin-bottom:16px;'>
        [!] Image digitization accuracy depends on image quality.
        Results from image input are less reliable than direct signal input.
        Always verify with raw signal data when available.
        </div>""", unsafe_allow_html=True)

        uploaded_img = st.file_uploader(
            "Upload ECG image",
            type=["png", "jpg", "jpeg"],
            key="upload_image",
        )

        if uploaded_img is not None:
            try:
                from PIL import Image
                import cv2

                # Load image
                pil_img = Image.open(uploaded_img).convert("RGB")
                img_arr = np.array(pil_img)

                # Show original
                st.image(pil_img, caption="Uploaded ECG Image",
                         use_column_width=True)

                # ── Digitization ──────────────────────
                # Convert to grayscale
                gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

                # Crop 8% from all edges — removes borders/spines/titles
                h, w   = gray.shape
                top    = int(h * 0.08)
                bot    = int(h * 0.92)
                left   = int(w * 0.04)
                right  = int(w * 0.96)
                gray   = gray[top:bot, left:right]

                # Invert if background is dark
                if gray.mean() < 128:
                    gray = 255 - gray

                # Threshold: isolate dark ECG trace
                _, binary = cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Morphological close — connect broken trace segments
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Extract signal: for each column find centroid of trace pixels
                signal_raw = []
                h_crop = binary.shape[0]
                prev_y = h_crop / 2.0

                for col in range(binary.shape[1]):
                    col_data = binary[:, col]
                    if col_data.max() > 0:
                        ys    = np.where(col_data > 0)[0]
                        y_pos = float(np.mean(ys))
                        # Reject if jump is too large (border artifact)
                        if abs(y_pos - prev_y) < h_crop * 0.4:
                            prev_y = y_pos
                        else:
                            y_pos = prev_y
                    else:
                        y_pos = prev_y
                    signal_raw.append(y_pos)

                signal_raw = np.array(signal_raw)

                # Invert y-axis
                signal_raw = h_crop - signal_raw

                # Resample to 256
                from scipy.signal import resample, medfilt
                signal_256 = resample(signal_raw, 256)

                # Smooth
                signal_256 = medfilt(signal_256, kernel_size=5)

                # Z-score normalize
                signal_256 = (signal_256 - signal_256.mean()) / (signal_256.std() + 1e-8)

                # Show extracted signal
                fig_ext, ax_ext = plt.subplots(figsize=(10, 2.5))
                fig_style(fig_ext, ax_ext)
                ax_ext.plot(np.arange(256)/360, signal_256,
                            color="#f59e0b", lw=1.5)
                ax_ext.set_title("Extracted Signal from Image (digitized)",
                                 fontsize=11, fontweight="bold")
                ax_ext.set_xlabel("Time (s)"); ax_ext.set_ylabel("Amplitude (normalized)")
                plt.tight_layout()
                st.pyplot(fig_ext)

                st.success(f"Signal digitized: 256 samples, "
                           f"mean={signal_256.mean():.3f}, std={signal_256.std():.3f}")

                if st.button("Use digitized signal", key="btn_img"):
                    st.session_state["signal"]       = signal_256
                    st.session_state["input_label"]  = "Unknown (image)"
                    st.session_state["input_source"] = f"Image: {uploaded_img.name}"
                    st.success("Digitized signal ready for prediction.")

            except ImportError:
                st.error("Missing libraries. Run: pip install opencv-python pillow scipy")
            except Exception as e:
                st.error(f"Image processing failed: {e}")

    # ─────────────────────────────────────────
    # PREDICTION BLOCK — runs for any input
    # ─────────────────────────────────────────
    st.markdown("---")

    # Also allow direct use from dataset slider without button
    if "signal" not in st.session_state:
        X, y = load_ecg_data()
        if X is not None:
            st.session_state["signal"]       = X[0]
            st.session_state["input_label"]  = CLASS_NAMES[int(y[0])]
            st.session_state["input_source"] = "Dataset sample #0 (default)"

    if "signal" in st.session_state:
        signal       = st.session_state["signal"]
        input_label  = st.session_state.get("input_label",  "Unknown")
        input_source = st.session_state.get("input_source", "Unknown")

        st.markdown(f"""
        <div class='section-header'>PREDICTION OUTPUT</div>
        <div style='font-size:12px;color:#64748b;font-family:IBM Plex Mono,monospace;
                    margin-bottom:16px;'>
            Input source: <span style='color:#94a3b8;'>{input_source}</span>
            &nbsp;|&nbsp; Ground truth: <span style='color:#94a3b8;'>{input_label}</span>
        </div>
        """, unsafe_allow_html=True)

        col_m, col_s = st.columns([2, 1])
        with col_m:
            model_choice = st.selectbox(
                "SELECT MODEL",
                ["CNN_MLP — Baseline", "CNN_KAN — Proposed", "Both Models"],
                key="pred_model_select"
            )
        with col_s:
            show_saliency = st.checkbox("Show saliency overlay", value=True,
                                        key="pred_saliency")

        models_to_run = []
        if "CNN_MLP" in model_choice or "Both" in model_choice:
            models_to_run.append("CNN_MLP")
        if "CNN_KAN" in model_choice or "Both" in model_choice:
            models_to_run.append("CNN_KAN")

        for model_name in models_to_run:
            model = load_model(model_name)
            if model is None:
                st.error(f"Could not load {model_name}. Check checkpoint path.")
                continue

            proba, pred_idx, infer_ms = run_prediction(model, signal)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(proba[pred_idx])
            triage     = TRIAGE_POLICY[pred_class]
            sqi, flags = assess_signal_quality(signal)
            entropy    = float(-np.sum(proba * np.log(proba + 1e-8)))

            if confidence >= CONFIDENCE_HIGH:
                unc_flag = "Confident"
            elif confidence >= CONFIDENCE_LOW:
                unc_flag = "Moderate — review recommended"
            else:
                unc_flag = "Low confidence — manual review required"

            badge_cls = {"Routine": "routine", "Priority Review": "priority",
                         "Urgent Review": "urgent"}[triage]

            st.markdown(f"""
            <div style='border-top:2px solid #1e2d4a;padding-top:20px;margin-top:20px;'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:13px;
                        color:#00d4ff;letter-spacing:2px;margin-bottom:16px;'>
                RESULT — {model_name}
            </div></div>""", unsafe_allow_html=True)

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Predicted</div>
                <div class='metric-value' style='color:{CLASS_COLORS[pred_class]};'>
                    {pred_class}</div>
                <div class='metric-delta'>{CLASS_DESCRIPTIONS[pred_class][:18]}...</div>
            </div>""", unsafe_allow_html=True)
            r2.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Confidence</div>
                <div class='metric-value {"green" if confidence>=CONFIDENCE_HIGH else "orange"}'>
                    {confidence*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            r3.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Triage</div>
                <div style='margin-top:8px;'>
                    <span class='badge badge-{badge_cls}'>{triage}</span></div>
            </div>""", unsafe_allow_html=True)
            r4.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Signal Quality</div>
                <div class='metric-value {"green" if sqi=="Good" else "orange" if sqi=="Moderate" else "red"}'>
                    {sqi}</div>
            </div>""", unsafe_allow_html=True)
            r5.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Inference</div>
                <div class='metric-value'>{infer_ms:.1f}</div>
                <div class='metric-delta'>ms</div>
            </div>""", unsafe_allow_html=True)

            # ECG plot
            saliency = None
            if show_saliency:
                try:
                    saliency = get_saliency(model, signal)
                except Exception:
                    pass

            st.pyplot(plot_ecg_signal(signal, saliency, pred_class))

            # Probability bars + clinical details
            col_prob, col_detail = st.columns([1, 1])
            with col_prob:
                st.pyplot(plot_prob_bars(proba))
            with col_detail:
                top3 = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:3]
                st.markdown("""<div class='clinical-card'>
                    <div class='clinical-label'>Top-3 Predictions</div>""",
                            unsafe_allow_html=True)
                for idx, p in top3:
                    c = CLASS_NAMES[idx]
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;
                                padding:6px 0;border-bottom:1px solid #1e2d4a;'>
                        <span style='color:{CLASS_COLORS[c]};
                                     font-family:IBM Plex Mono,monospace;
                                     font-weight:600;'>{c}</span>
                        <span style='color:#94a3b8;font-size:12px;'>
                            {CLASS_DESCRIPTIONS[c]}</span>
                        <span style='color:#e2e8f0;font-family:IBM Plex Mono,monospace;'>
                            {p*100:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='margin-top:12px;'>
                    <div class='clinical-label'>Uncertainty</div>
                    <div class='clinical-sub'>{unc_flag}</div>
                    <div class='clinical-label' style='margin-top:10px;'>Entropy</div>
                    <div class='clinical-sub'>{entropy:.4f}</div>
                    <div class='clinical-label' style='margin-top:10px;'>Artifacts</div>
                    <div class='clinical-sub'>{', '.join(flags)}</div>
                    </div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
    [!] CLINICAL DISCLAIMER — Decision-support only. Not for standalone diagnosis.
    Final interpretation must be made by a licensed clinician.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: Model Comparison
# ══════════════════════════════════════════════
elif "Comparison" in page:
    results = load_results()
    if not results:
        st.warning("Run evaluate.py first.")
    else:
        st.markdown("<div class='section-header'>OVERALL METRICS</div>",
                    unsafe_allow_html=True)
        st.pyplot(plot_comparison_bars(results))

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='section-header'>CONFUSION MATRIX — CNN+MLP</div>",
                        unsafe_allow_html=True)
            st.pyplot(plot_confusion_matrix(
                results["CNN_MLP"]["confusion_matrix"], "CNN_MLP"))
        with col_r:
            st.markdown("<div class='section-header'>CONFUSION MATRIX — CNN+KAN</div>",
                        unsafe_allow_html=True)
            st.pyplot(plot_confusion_matrix(
                results["CNN_KAN"]["confusion_matrix"], "CNN_KAN"))

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown("<div class='section-header'>PER-CLASS F1</div>",
                        unsafe_allow_html=True)
            st.pyplot(plot_perclass_f1(results))
        with col_r2:
            st.markdown("<div class='section-header'>RADAR COMPARISON</div>",
                        unsafe_allow_html=True)
            st.pyplot(plot_radar(results))

        # Efficiency
        st.markdown("<div class='section-header'>EFFICIENCY COMPARISON</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, name, color in [(c1, "CNN_MLP", "#4C72B0"), (c3, "CNN_KAN", "#DD8452")]:
            r = results[name]
            col.markdown(f"""
            <div class='clinical-card'>
                <div class='clinical-label' style='color:{color};'>{name}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:13px;
                            color:#94a3b8;line-height:2;'>
                    Parameters: <b style='color:#e2e8f0;'>
                        {r['parameters']['trainable']:,}</b><br>
                    Inference: <b style='color:#e2e8f0;'>
                        {r['inference']['avg_sample_ms']} ms/sample</b><br>
                    Accuracy: <b style='color:#e2e8f0;'>{r['accuracy']:.4f}</b><br>
                    Macro F1: <b style='color:#e2e8f0;'>{r['macro_f1']:.4f}</b><br>
                    Macro AUC: <b style='color:#e2e8f0;'>{r['macro_auc']:.4f}</b>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: Training Analysis
# ══════════════════════════════════════════════
elif "Training" in page:
    st.markdown("<div class='section-header'>TRAINING CURVES</div>",
                unsafe_allow_html=True)

    for model_name in ["CNN_MLP", "CNN_KAN"]:
        log = load_train_log(model_name)
        if log:
            st.pyplot(plot_training_curves(log, model_name))
            c1, c2, c3 = st.columns(3)
            best_val_loss = min(log["val_loss"])
            best_val_acc  = max(log["val_acc"])
            total_epochs  = len(log["epoch"])
            c1.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Best Val Loss</div>
                <div class='metric-value green'>{best_val_loss:.4f}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Best Val Acc</div>
                <div class='metric-value green'>{best_val_acc:.4f}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Epochs Trained</div>
                <div class='metric-value'>{total_epochs}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("---")
        else:
            st.warning(f"No training log found for {model_name}.")


# ══════════════════════════════════════════════
# PAGE: Results Table
# ══════════════════════════════════════════════
elif "Results" in page:
    results = load_results()
    if not results:
        st.warning("Run evaluate.py first.")
    else:
        st.markdown("<div class='section-header'>FULL RESULTS TABLE</div>",
                    unsafe_allow_html=True)

        cls_names = ["N", "S", "V", "F", "Q"]
        rows_html  = ""

        metrics_rows = [
            ("Accuracy",    "accuracy"),
            ("Macro F1",    "macro_f1"),
            ("Weighted F1", "weighted_f1"),
            ("Macro AUC",   "macro_auc"),
        ]

        for label, key in metrics_rows:
            cv = results["CNN_MLP"][key]
            kv = results["CNN_KAN"][key]
            w_cnn = "winner" if cv >= kv else ""
            w_kan = "winner" if kv >= cv else ""
            rows_html += f"""<tr>
                <td>{label}</td>
                <td class='{w_cnn}'>{cv:.4f}</td>
                <td class='{w_kan}'>{kv:.4f}</td>
            </tr>"""

        # Parameters
        cp = results["CNN_MLP"]["parameters"]["trainable"]
        kp = results["CNN_KAN"]["parameters"]["trainable"]
        w_cnn = "winner" if cp <= kp else ""
        w_kan = "winner" if kp <= cp else ""
        rows_html += f"""<tr>
            <td>Parameters (fewer = better)</td>
            <td class='{w_cnn}'>{cp:,}</td>
            <td class='{w_kan}'>{kp:,}</td>
        </tr>"""

        # Inference
        ci = results["CNN_MLP"]["inference"]["avg_sample_ms"]
        ki = results["CNN_KAN"]["inference"]["avg_sample_ms"]
        w_cnn = "winner" if ci <= ki else ""
        w_kan = "winner" if ki <= ci else ""
        rows_html += f"""<tr>
            <td>Inference ms/sample (lower = better)</td>
            <td class='{w_cnn}'>{ci}</td>
            <td class='{w_kan}'>{ki}</td>
        </tr>"""

        st.markdown(f"""
        <table class='results-table'>
            <thead><tr>
                <th>METRIC</th><th>CNN + MLP</th><th>CNN + KAN</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        # Per-class
        st.markdown("<div class='section-header' style='margin-top:32px;'>PER-CLASS METRICS</div>",
                    unsafe_allow_html=True)

        for metric_key, metric_label in [("f1","F1"), ("precision","Precision"),
                                          ("recall","Recall"), ("auc","AUC")]:
            rows_html = ""
            for c in cls_names:
                cv = results["CNN_MLP"]["per_class"][c][metric_key]
                kv = results["CNN_KAN"]["per_class"][c][metric_key]
                w_cnn = "winner" if cv >= kv else ""
                w_kan = "winner" if kv >= cv else ""
                desc  = CLASS_DESCRIPTIONS[c]
                rows_html += f"""<tr>
                    <td><span style='color:{CLASS_COLORS[c]};font-weight:600;'>{c}</span>
                        &nbsp; {desc}</td>
                    <td class='{w_cnn}'>{cv:.4f}</td>
                    <td class='{w_kan}'>{kv:.4f}</td>
                </tr>"""

            st.markdown(f"""
            <div style='margin-bottom:20px;'>
            <div style='font-size:11px;color:#64748b;font-family:IBM Plex Mono,monospace;
                        text-transform:uppercase;letter-spacing:1px;
                        margin-bottom:8px;'>{metric_label}</div>
            <table class='results-table'>
                <thead><tr>
                    <th>CLASS</th><th>CNN + MLP</th><th>CNN + KAN</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table></div>""", unsafe_allow_html=True)

        # Download CSV
        csv_path = "experiments/results.csv"
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download results.csv",
                    data=f,
                    file_name="ecg_kan_results.csv",
                    mime="text/csv",
                )