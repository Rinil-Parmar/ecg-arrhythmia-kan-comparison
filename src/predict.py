"""
Clinical Prediction — ECG Arrhythmia Classification
-----------------------------------------------------
Produces structured clinical output for a given ECG segment:
  - Predicted class + confidence
  - Top-3 predictions
  - Triage level
  - Uncertainty flag
  - Signal quality index
  - Arrhythmia burden
  - Evidence windows
  - JSON output + human-readable report

Usage:
    python -m src.predict --signal data/processed/X.npy --index 0 --model cnn
    python -m src.predict --signal data/processed/X.npy --index 0 --model kan
    python -m src.predict --signal data/processed/X.npy --index 0 --model both
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}

CLASS_DESCRIPTIONS = {
    "N": "Normal / Non-ectopic beat",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Unknown / Unclassifiable beat",
}

# Triage policy per class
TRIAGE_POLICY = {
    "N": "Routine",
    "S": "Priority Review",
    "V": "Urgent Review",
    "F": "Urgent Review",
    "Q": "Priority Review",
}

# Confidence thresholds
CONFIDENCE_HIGH   = 0.85
CONFIDENCE_LOW    = 0.60

# Signal quality thresholds (based on signal statistics)
SQI_GOOD_STD_MIN  = 0.05
SQI_POOR_STD_MAX  = 0.01
ARTIFACT_AMP_MAX  = 4.0


# ─────────────────────────────────────────────
# Signal Quality Assessment
# ─────────────────────────────────────────────
def assess_signal_quality(signal: np.ndarray) -> dict:
    """
    Simple signal quality index based on amplitude and noise stats.

    Args:
        signal: [256] normalized ECG segment

    Returns:
        dict with sqi label, artifact flags, action
    """
    std_val    = float(np.std(signal))
    max_amp    = float(np.max(np.abs(signal)))
    flatline   = std_val < SQI_POOR_STD_MAX
    high_amp   = max_amp > ARTIFACT_AMP_MAX

    # Baseline wander: low-freq energy check
    diff_signal      = np.diff(signal)
    baseline_wander  = float(np.std(diff_signal)) < 0.005

    # SQI label
    if flatline or high_amp:
        sqi = "Poor"
    elif baseline_wander or std_val < SQI_GOOD_STD_MIN:
        sqi = "Moderate"
    else:
        sqi = "Good"

    artifact_flags = []
    if flatline:         artifact_flags.append("Flatline / Lead-off")
    if high_amp:         artifact_flags.append("High amplitude artifact")
    if baseline_wander:  artifact_flags.append("Baseline wander")

    action = {
        "Poor"    : "Repeat ECG acquisition before interpretation.",
        "Moderate": "Interpret with caution. Signal quality may affect accuracy.",
        "Good"    : "Signal quality acceptable for automated analysis.",
    }[sqi]

    return {
        "sqi"           : sqi,
        "artifact_flags": artifact_flags if artifact_flags else ["None"],
        "action"        : action,
        "std"           : round(std_val, 4),
        "max_amplitude" : round(max_amp, 4),
    }


# ─────────────────────────────────────────────
# Evidence Windows (gradient-based attribution)
# ─────────────────────────────────────────────
def get_evidence_windows(
    model: nn.Module,
    x_tensor: torch.Tensor,
    device: torch.device,
    fs: int = 360,
    top_k: int = 2,
) -> list[dict]:
    """
    Compute input gradient saliency to find influential time windows.

    Args:
        model    : trained model
        x_tensor : [1, 1, 256] input tensor
        device   : torch device
        fs       : sampling frequency (MIT-BIH = 360 Hz)
        top_k    : number of top windows to return

    Returns:
        list of dicts with start_s, end_s, importance_score
    """
    model.eval()
    x = x_tensor.to(device).requires_grad_(True)

    logits = model(x)
    pred_class = logits.argmax(dim=1).item()

    # Backprop w.r.t. predicted class score
    logits[0, pred_class].backward()

    saliency = x.grad.abs().squeeze().cpu().numpy()  # [256]

    # Divide signal into windows of ~32 samples
    window_size = 32
    n_windows   = len(saliency) // window_size
    scores      = []
    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        score = float(saliency[start:end].mean())
        scores.append((start, end, score))

    # Top-k windows by importance
    scores.sort(key=lambda x: x[2], reverse=True)
    top_windows = scores[:top_k]

    # Convert samples → seconds
    results = []
    for start, end, score in top_windows:
        results.append({
            "start_s"         : round(start / fs, 2),
            "end_s"           : round(end   / fs, 2),
            "importance_score": round(score, 4),
        })

    return results


# ─────────────────────────────────────────────
# Core Prediction
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_single(
    model: nn.Module,
    signal: np.ndarray,
    device: torch.device,
    model_name: str,
    patient_id: str = "Unknown",
    recording_id: str = "Unknown",
) -> dict:
    """
    Full clinical prediction for one ECG segment.

    Args:
        model       : trained model
        signal      : [256] normalized ECG segment
        device      : torch device
        model_name  : "CNN_MLP" or "CNN_KAN"
        patient_id  : optional patient ID
        recording_id: optional recording ID

    Returns:
        structured clinical output dict
    """
    # ── Prepare input ─────────────────────────
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,256]
    x_device = x.to(device)

    # ── Inference ─────────────────────────────
    model.eval()
    start_time = time.perf_counter()
    logits     = model(x_device)
    infer_ms   = (time.perf_counter() - start_time) * 1000

    proba      = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [5]
    pred_idx   = int(proba.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])

    # ── Top-3 predictions ─────────────────────
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [
        {"class": CLASS_NAMES[i], "probability": round(float(proba[i]), 4)}
        for i in top3_idx
    ]

    # ── Uncertainty ───────────────────────────
    entropy = float(-np.sum(proba * np.log(proba + 1e-8)))
    if confidence >= CONFIDENCE_HIGH:
        uncertainty_flag = "Confident"
    elif confidence >= CONFIDENCE_LOW:
        uncertainty_flag = "Moderate confidence — review recommended"
    else:
        uncertainty_flag = "Low confidence — manual review required"

    # ── Triage ────────────────────────────────
    base_triage = TRIAGE_POLICY[pred_class]
    # Escalate if low confidence
    if confidence < CONFIDENCE_LOW and base_triage == "Routine":
        triage = "Priority Review"
    else:
        triage = base_triage

    # ── Signal quality ────────────────────────
    sqi_result = assess_signal_quality(signal)
    # Escalate triage if poor signal
    if sqi_result["sqi"] == "Poor" and triage == "Routine":
        triage = "Priority Review"

    # ── Evidence windows (uses gradients — re-enable grad) ──
    with torch.enable_grad():
        evidence = get_evidence_windows(model, x, device)

    # ── Clinical explanation ──────────────────
    explanation = _build_explanation(pred_class, confidence, sqi_result)

    # ── Arrhythmia burden (single beat) ───────
    is_abnormal    = pred_class != "N"
    arrhy_burden   = 100.0 if is_abnormal else 0.0

    # ── Assemble output ───────────────────────
    output = {
        "report_timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name"        : model_name,
        "patient_id"        : patient_id,
        "recording_id"      : recording_id,
        "prediction": {
            "predicted_class"      : pred_class,
            "class_description"    : CLASS_DESCRIPTIONS[pred_class],
            "confidence"           : round(confidence, 4),
            "confidence_pct"       : f"{confidence*100:.1f}%",
            "all_probabilities"    : {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(5)},
            "top_3_predictions"    : top3,
        },
        "triage": {
            "level"           : triage,
            "uncertainty_flag": uncertainty_flag,
            "entropy"         : round(entropy, 4),
        },
        "signal_quality": {
            "sqi"           : sqi_result["sqi"],
            "artifact_flags": sqi_result["artifact_flags"],
            "action"        : sqi_result["action"],
        },
        "evidence": {
            "windows"     : evidence,
            "explanation" : explanation,
        },
        "arrhythmia_burden_pct": arrhy_burden,
        "performance": {
            "inference_ms": round(infer_ms, 3),
        },
        "disclaimer": (
            "This tool is decision-support only and not a standalone diagnostic system. "
            "Final interpretation must be made by a licensed clinician."
        ),
    }

    return output


# ─────────────────────────────────────────────
# Build human-readable explanation
# ─────────────────────────────────────────────
def _build_explanation(pred_class: str, confidence: float, sqi: dict) -> str:
    templates = {
        "N": "Regular beat morphology detected. No significant ectopic activity observed.",
        "S": "Early beat with altered P-wave morphology. Possible supraventricular origin.",
        "V": "Wide, bizarre QRS complex detected. Possible ventricular ectopic origin.",
        "F": "Beat morphology intermediate between normal and ventricular. Possible fusion beat.",
        "Q": "Beat morphology does not match known patterns. Unclassifiable — manual review needed.",
    }
    base = templates[pred_class]
    conf_note = (
        f" Model confidence: {confidence*100:.1f}%."
        if confidence >= CONFIDENCE_HIGH
        else f" Model confidence is low ({confidence*100:.1f}%) — treat with caution."
    )
    sqi_note = (
        f" Signal quality: {sqi['sqi']}."
        + (f" Artifacts: {', '.join(sqi['artifact_flags'])}." if sqi["artifact_flags"] != ["None"] else "")
    )
    return base + conf_note + sqi_note


# ─────────────────────────────────────────────
# Print formatted clinical report
# ─────────────────────────────────────────────
def print_clinical_report(output: dict):
    p  = output["prediction"]
    t  = output["triage"]
    sq = output["signal_quality"]
    ev = output["evidence"]

    print("\n" + "="*65)
    print(f"  CLINICAL ECG ANALYSIS REPORT")
    print("="*65)
    print(f"  Timestamp     : {output['report_timestamp']}")
    print(f"  Model         : {output['model_name']}")
    print(f"  Patient ID    : {output['patient_id']}")
    print(f"  Recording ID  : {output['recording_id']}")
    print("-"*65)
    print(f"  Predicted Class  : {p['predicted_class']} — {p['class_description']}")
    print(f"  Confidence       : {p['confidence_pct']}")
    print(f"  Top-3 Predictions:")
    for item in p["top_3_predictions"]:
        print(f"      {item['class']} : {item['probability']*100:.1f}%")
    print("-"*65)
    print(f"  Triage Level     : {t['level']}")
    print(f"  Uncertainty      : {t['uncertainty_flag']}")
    print(f"  Signal Quality   : {sq['sqi']}")
    print(f"  Artifacts        : {', '.join(sq['artifact_flags'])}")
    print(f"  Action           : {sq['action']}")
    print("-"*65)
    print(f"  Evidence Windows :")
    for w in ev["windows"]:
        print(f"      {w['start_s']}s – {w['end_s']}s  (importance: {w['importance_score']})")
    print(f"  Explanation      : {ev['explanation']}")
    print("-"*65)
    print(f"  Arrhythmia Burden: {output['arrhythmia_burden_pct']}%")
    print(f"  Inference Time   : {output['performance']['inference_ms']} ms")
    print("-"*65)
    print(f"  ⚠  {output['disclaimer']}")
    print("="*65)


# ─────────────────────────────────────────────
# Save ECG plot with evidence windows
# ─────────────────────────────────────────────
def save_ecg_plot(signal: np.ndarray, output: dict, save_path: str, fs: int = 360):
    fig, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.arange(len(signal)) / fs

    ax.plot(time_axis, signal, color="#2c7bb6", lw=1.5, label="ECG Signal")

    # Highlight evidence windows
    colors = ["#d7191c", "#fdae61"]
    for i, w in enumerate(output["evidence"]["windows"]):
        ax.axvspan(w["start_s"], w["end_s"],
                   alpha=0.25, color=colors[i % len(colors)],
                   label=f"Evidence window {i+1}")

    pred  = output["prediction"]["predicted_class"]
    conf  = output["prediction"]["confidence_pct"]
    triage= output["triage"]["level"]

    ax.set_title(
        f"ECG Segment — Predicted: {pred} ({conf}) | Triage: {triage} | "
        f"Model: {output['model_name']}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Amplitude (normalized)", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ECG plot saved → {save_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.cnn          import ECGCNN
    from src.models.cnn_kan      import ECGCNNWithKAN
    from src.training.evaluate   import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--signal",  default="data/processed/X.npy", help="Path to X.npy")
    parser.add_argument("--labels",  default="data/processed/y.npy",  help="Path to y.npy")
    parser.add_argument("--index",   type=int, default=0,             help="Sample index to predict")
    parser.add_argument("--model",   choices=["cnn", "kan", "both"],  default="both")
    parser.add_argument("--out_dir", default="outputs/predictions",   help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load signal
    X = np.load(args.signal)
    y = np.load(args.labels)
    signal     = X[args.index]           # [256]
    true_label = CLASS_NAMES[int(y[args.index])]

    print(f"\nSample index  : {args.index}")
    print(f"True label    : {true_label} — {CLASS_DESCRIPTIONS[true_label]}")

    models_to_run = []
    if args.model in ("cnn", "both"):
        m = load_model(ECGCNN(num_classes=5), "checkpoints/cnn_mlp.pth")
        models_to_run.append(("CNN_MLP", m))
    if args.model in ("kan", "both"):
        m = load_model(ECGCNNWithKAN(num_classes=5), "checkpoints/cnn_kan.pth")
        models_to_run.append(("CNN_KAN", m))

    os.makedirs(args.out_dir, exist_ok=True)

    for model_name, model in models_to_run:
        model = model.to(device)

        output = predict_single(
            model       = model,
            signal      = signal,
            device      = device,
            model_name  = model_name,
            patient_id  = f"P_{args.index:05d}",
            recording_id= f"ECG_{args.index:05d}",
        )

        # Print report
        print_clinical_report(output)

        # Save JSON
        json_path = os.path.join(args.out_dir, f"prediction_{model_name}_idx{args.index}.json")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  JSON saved  → {json_path}")

        # Save ECG plot
        plot_path = os.path.join(args.out_dir, f"ecg_plot_{model_name}_idx{args.index}.png")
        save_ecg_plot(signal, output, plot_path)