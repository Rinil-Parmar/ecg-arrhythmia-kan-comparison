"""
Download Real ECG Images for Testing
--------------------------------------
Downloads labeled ECG images from LITFL ECG Library
(Life in the Fast Lane — free clinical ECG resource)

Saves to: test_samples/real_ecg_images/

Usage:
    pip install requests pillow
    python download_ecg_images.py
"""

import os
import time
import requests
from PIL import Image
from io import BytesIO


# ─────────────────────────────────────────────
# Output folder
# ─────────────────────────────────────────────
OUTPUT_DIR = "test_samples/real_ecg_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─────────────────────────────────────────────
# ECG image URLs — direct image links
# Labeled by arrhythmia class (mapped to MIT-BIH)
# ─────────────────────────────────────────────
ECG_IMAGES = [
    # ── Normal (N) ────────────────────────────
    {
        "url"      : "https://litfl.com/wp-content/uploads/2018/08/Normal-sinus-rhythm-litfl-ECG-library.jpg",
        "filename" : "real_N_normal_sinus_rhythm.jpg",
        "class"    : "N",
        "label"    : "Normal Sinus Rhythm",
        "source"   : "LITFL",
    },
    {
        "url"      : "https://litfl.com/wp-content/uploads/2018/08/Normal-sinus-rhythm-2-litfl-ECG-library.jpg",
        "filename" : "real_N_normal_sinus_rhythm_2.jpg",
        "class"    : "N",
        "label"    : "Normal Sinus Rhythm 2",
        "source"   : "LITFL",
    },

    # ── Supraventricular (S) ──────────────────
    {
        "url"      : "https://litfl.com/wp-content/uploads/2018/08/Supraventricular-tachycardia-SVT-AVNRT-litfl-ECG-library.jpg",
        "filename" : "real_S_supraventricular_tachycardia.jpg",
        "class"    : "S",
        "label"    : "Supraventricular Tachycardia (SVT)",
        "source"   : "LITFL",
    },
    {
        "url"      : "https://litfl.com/wp-content/uploads/2018/08/Atrial-fibrillation-AF-fast-ventricular-rate-litfl-ECG-library.jpg",
        "filename" : "real_S_atrial_fibrillation.jpg",
        "class"    : "S",
        "label"    : "Atrial Fibrillation",
        "source"   : "LITFL",
    },

    # ── Ventricular (V) ───────────────────────
    {
        "url"      : "https://litfl.com/wp-content/uploads/2018/08/PVC-bigeminy-litfl-ECG-library.jpg",
        "filename" : "real_V_pvc_bigeminy.jpg",
        "class"    : "V",
        "label"    : "PVC Bigeminy",
        "source"   : "LITFL",
    },
    {
        "url"      : "https://litfl.com/wp-content/uploads/2018/08/Ventricular-tachycardia-VT-monomorphic-litfl-ECG-library.jpg",
        "filename" : "real_V_ventricular_tachycardia.jpg",
        "class"    : "V",
        "label"    : "Ventricular Tachycardia",
        "source"   : "LITFL",
    },

    # ── PhysioNet sample plots (PNG) ──────────
    {
        "url"      : "https://physionet.org/physiobank/database/mitdb/mitdbdir/figures/100s.png",
        "filename" : "real_physionet_record100.png",
        "class"    : "N",
        "label"    : "MIT-BIH Record 100 (Normal)",
        "source"   : "PhysioNet",
    },
    {
        "url"      : "https://physionet.org/physiobank/database/mitdb/mitdbdir/figures/119s.png",
        "filename" : "real_physionet_record119.png",
        "class"    : "V",
        "label"    : "MIT-BIH Record 119 (PVC)",
        "source"   : "PhysioNet",
    },
]


# ─────────────────────────────────────────────
# Download + save
# ─────────────────────────────────────────────
def download_image(entry: dict) -> bool:
    url      = entry["url"]
    filename = entry["filename"]
    save_path= os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(save_path):
        print(f"  [skip] already exists: {filename}")
        return True

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  [fail] HTTP {resp.status_code} — {url}")
            return False

        # Validate it's an image
        img = Image.open(BytesIO(resp.content)).convert("RGB")

        # Save as PNG for consistency
        png_path = save_path.replace(".jpg", ".png").replace(".jpeg", ".png")
        img.save(png_path, "PNG")

        print(f"  [ok]   {filename}  ({img.size[0]}x{img.size[1]}px)")
        return True

    except Exception as e:
        print(f"  [fail] {filename} — {e}")
        return False


# ─────────────────────────────────────────────
# Fallback — generate synthetic real-looking ECG
# if download fails
# ─────────────────────────────────────────────
def generate_fallback_ecg(entry: dict):
    """
    Generate a synthetic ECG waveform that looks like a real ECG.
    Used when real image download fails.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(hash(entry["filename"]) % 2**32)
    fs      = 360
    n       = 256
    t       = np.linspace(0, n/fs, n)
    signal  = np.zeros(n)

    class_name = entry["class"]

    # Build a beat based on class
    center = n // 2

    if class_name == "N":
        # P wave
        signal += 0.15 * np.exp(-((np.arange(n) - center*0.7)**2) / (2*8**2))
        # Q dip
        signal -= 0.1  * np.exp(-((np.arange(n) - center*0.92)**2) / (2*3**2))
        # R peak (tall, narrow)
        signal += 1.5  * np.exp(-((np.arange(n) - center)**2) / (2*4**2))
        # S dip
        signal -= 0.4  * np.exp(-((np.arange(n) - center*1.07)**2) / (2*3**2))
        # T wave
        signal += 0.3  * np.exp(-((np.arange(n) - center*1.3)**2) / (2*15**2))

    elif class_name == "S":
        # Narrow QRS, irregular P
        signal += 0.08 * np.exp(-((np.arange(n) - center*0.6)**2) / (2*5**2))
        signal += 1.2  * np.exp(-((np.arange(n) - center)**2) / (2*3**2))
        signal -= 0.3  * np.exp(-((np.arange(n) - center*1.05)**2) / (2*2**2))
        signal += 0.2  * np.exp(-((np.arange(n) - center*1.25)**2) / (2*12**2))

    elif class_name == "V":
        # Wide, bizarre QRS — no P wave
        signal += 0.8  * np.exp(-((np.arange(n) - center*0.9)**2) / (2*12**2))
        signal -= 1.2  * np.exp(-((np.arange(n) - center*1.05)**2) / (2*10**2))
        signal += 0.4  * np.exp(-((np.arange(n) - center*1.25)**2) / (2*18**2))

    elif class_name == "F":
        # Fusion — intermediate morphology
        signal += 0.1  * np.exp(-((np.arange(n) - center*0.7)**2) / (2*6**2))
        signal += 0.9  * np.exp(-((np.arange(n) - center)**2) / (2*6**2))
        signal -= 0.5  * np.exp(-((np.arange(n) - center*1.08)**2) / (2*5**2))
        signal += 0.25 * np.exp(-((np.arange(n) - center*1.3)**2) / (2*14**2))

    else:  # Q
        signal += 0.3  * np.random.randn(n) * 0.1
        signal += 0.5  * np.exp(-((np.arange(n) - center)**2) / (2*8**2))

    # Add mild baseline noise
    signal += np.random.randn(n) * 0.03

    # Normalize
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    # Save as clean image (no grid, no axes)
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(t, signal, color="black", lw=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0.3)

    fname    = entry["filename"].replace(".jpg", ".png")
    savepath = os.path.join(OUTPUT_DIR, f"synthetic_{fname}")
    plt.savefig(savepath, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [synth] generated fallback: synthetic_{fname}")


# ─────────────────────────────────────────────
# Save README
# ─────────────────────────────────────────────
def save_readme(results: list):
    path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("REAL ECG TEST IMAGES\n")
        f.write("=" * 60 + "\n\n")
        f.write("HOW TO USE:\n")
        f.write("  streamlit run app.py\n")
        f.write("  Go to: Live Prediction > Tab 3 (Upload ECG Image)\n")
        f.write("  Upload any .png file from this folder\n\n")
        f.write("FILES:\n")
        f.write("-" * 60 + "\n")
        for r in results:
            status = "downloaded" if r["ok"] else "fallback (synthetic)"
            f.write(f"  {r['filename']}\n")
            f.write(f"    Class  : {r['class']} — {r['label']}\n")
            f.write(f"    Source : {r['source']}\n")
            f.write(f"    Status : {status}\n\n")
        f.write("=" * 60 + "\n")
        f.write("CLASS REFERENCE:\n")
        f.write("  N — Normal beat\n")
        f.write("  S — Supraventricular ectopic\n")
        f.write("  V — Ventricular ectopic\n")
        f.write("  F — Fusion beat\n")
        f.write("  Q — Unknown\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"\nDownloading real ECG images to '{OUTPUT_DIR}/' ...\n")
    print("=" * 60)

    results = []
    ok_count = 0

    for entry in ECG_IMAGES:
        print(f"\n  {entry['class']} — {entry['label']}")
        ok = download_image(entry)

        if not ok:
            print(f"  Generating synthetic fallback ...")
            generate_fallback_ecg(entry)

        results.append({**entry, "ok": ok})
        if ok:
            ok_count += 1

        time.sleep(0.5)  # polite delay between requests

    save_readme(results)

    print("\n" + "=" * 60)
    print(f"  Done — {ok_count}/{len(ECG_IMAGES)} downloaded successfully")
    print(f"  Saved to: {OUTPUT_DIR}/")
    print("=" * 60)
    print("\nHOW TO TEST:")
    print("  streamlit run app.py")
    print("  -> Live Prediction -> Tab 3 -> Upload any .png from")
    print(f"     {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()