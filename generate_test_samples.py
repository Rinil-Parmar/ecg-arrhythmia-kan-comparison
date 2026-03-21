"""
Generate Test Samples for ECG Arrhythmia UI
--------------------------------------------
Creates a 'test_samples/' folder with:
  - 5 x .npy files  (one per class)
  - 5 x .csv files  (one per class)
  - 10 x .png ECG images (2 per class, clean single-beat plots)

All samples taken from your real preprocessed data.
Ready to upload directly in the Streamlit UI.

Usage:
    python generate_test_samples.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OUTPUT_DIR  = "test_samples"

CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
CLASS_DESCRIPTIONS = {
    "N": "Normal",
    "S": "Supraventricular",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown",
}
CLASS_COLORS = {
    "N": "#22c55e",
    "S": "#f59e0b",
    "V": "#ef4444",
    "F": "#8b5cf6",
    "Q": "#64748b",
}

SAMPLES_PER_CLASS = 2   # number of images + npy + csv per class


# ─────────────────────────────────────────────
# Save ECG image — clean, white background
# ─────────────────────────────────────────────
def save_ecg_image(signal, class_name, sample_idx, save_path):
    fs  = 360
    t   = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Thick black signal line
    ax.plot(t, signal, color="black", lw=2.5)

    # Remove ALL spines, ticks, grid, labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # ── Load test split only ──────────────────
    print("Loading test split (held-out data only) ...")
    from src.data.loader import create_dataloaders

    _, _, test_loader = create_dataloaders(
        data_dir="data/processed", batch_size=128
    )

    X = test_loader.dataset.X.squeeze(1).numpy()  # [N, 256]
    y = test_loader.dataset.y.numpy()              # [N]

    print(f"Test set: X={X.shape}, y={y.shape} — never seen during training")

    # ── Create output dirs ─────────────────────
    dirs = {
        "npy"  : os.path.join(OUTPUT_DIR, "npy_files"),
        "csv"  : os.path.join(OUTPUT_DIR, "csv_files"),
        "images": os.path.join(OUTPUT_DIR, "ecg_images"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    summary = []

    # ── Per class ─────────────────────────────
    for class_idx, class_name in CLASS_NAMES.items():
        # Find all indices for this class
        indices = np.where(y == class_idx)[0]

        if len(indices) == 0:
            print(f"  [!] No samples found for class {class_name}, skipping.")
            continue

        # Pick evenly spaced samples for variety
        picked = indices[np.linspace(0, len(indices)-1,
                                     SAMPLES_PER_CLASS, dtype=int)]

        print(f"\nClass {class_name} ({CLASS_DESCRIPTIONS[class_name]}) "
              f"— {len(indices):,} total samples, saving {SAMPLES_PER_CLASS}:")

        for i, idx in enumerate(picked):
            signal = X[idx]   # [256]
            tag    = f"class{class_name}_sample{i+1}_idx{idx}"

            # ── Save .npy ─────────────────────
            npy_path = os.path.join(dirs["npy"], f"{tag}.npy")
            np.save(npy_path, signal)

            # ── Save .csv ─────────────────────
            csv_path = os.path.join(dirs["csv"], f"{tag}.csv")
            np.savetxt(csv_path, signal.reshape(1, -1), delimiter=",", fmt="%.6f")

            # ── Save ECG image ─────────────────
            img_path = os.path.join(dirs["images"], f"{tag}.png")
            save_ecg_image(signal, class_name, idx, img_path)

            print(f"  [{i+1}] index={idx:<7} → {tag}")
            summary.append({
                "class"      : class_name,
                "description": CLASS_DESCRIPTIONS[class_name],
                "dataset_idx": int(idx),
                "npy"        : npy_path,
                "csv"        : csv_path,
                "image"      : img_path,
            })

    # ── Save README ───────────────────────────
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("ECG TEST SAMPLES\n")
        f.write("=" * 60 + "\n")
        f.write("Generated from MIT-BIH Arrhythmia Database\n\n")
        f.write("FOLDER STRUCTURE:\n")
        f.write("  test_samples/\n")
        f.write("  |-- npy_files/    <- upload in UI Tab 2\n")
        f.write("  |-- csv_files/    <- upload in UI Tab 2\n")
        f.write("  +-- ecg_images/   <- upload in UI Tab 3\n\n")
        f.write("HOW TO USE IN STREAMLIT UI:\n")
        f.write("  Tab 2 → Upload .npy or .csv file → Predict\n")
        f.write("  Tab 3 → Upload .png image → Digitize → Predict\n\n")
        f.write("SAMPLES:\n")
        f.write("-" * 60 + "\n")
        for s in summary:
            f.write(f"Class {s['class']} ({s['description']})\n")
            f.write(f"  Dataset index : {s['dataset_idx']}\n")
            f.write(f"  .npy file     : {s['npy']}\n")
            f.write(f"  .csv file     : {s['csv']}\n")
            f.write(f"  image         : {s['image']}\n\n")
        f.write("=" * 60 + "\n")
        f.write("CLASSES:\n")
        f.write("  N — Normal / Non-ectopic beat\n")
        f.write("  S — Supraventricular ectopic beat\n")
        f.write("  V — Ventricular ectopic beat\n")
        f.write("  F — Fusion beat\n")
        f.write("  Q — Unknown / Unclassifiable beat\n")

    # ── Print summary ─────────────────────────
    print("\n" + "=" * 60)
    print(f"  DONE — {len(summary)} samples saved to '{OUTPUT_DIR}/'")
    print("=" * 60)
    print(f"  npy files  : {dirs['npy']}")
    print(f"  csv files  : {dirs['csv']}")
    print(f"  ecg images : {dirs['images']}")
    print(f"  readme     : {readme_path}")
    print("=" * 60)
    print("\nHOW TO TEST:")
    print("  streamlit run app.py")
    print("  → Tab 2: upload any .npy or .csv from test_samples/npy_files/")
    print("  → Tab 3: upload any .png  from test_samples/ecg_images/")
    print("=" * 60)


if __name__ == "__main__":
    main()