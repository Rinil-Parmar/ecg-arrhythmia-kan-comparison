# ECG Arrhythmia Classification — KAN vs MLP

> Comparing Kolmogorov-Arnold Networks (KAN) against MLP heads for ECG arrhythmia detection using the MIT-BIH Arrhythmia Database.

---

## Overview

This project investigates whether replacing the MLP classification head with a **KAN head** on top of a shared 1D CNN encoder improves ECG arrhythmia classification performance, parameter efficiency, or interpretability.

| Model | Accuracy | Macro F1 | Macro AUC | Parameters |
|-------|----------|----------|-----------|------------|
| CNN + MLP (baseline) | 0.980 | 0.902 | 0.997 | ~175,000 |
| CNN + KAN (proposed) | 0.948 | 0.817 | 0.992 | ~155,000 |

---

## Dataset

- **MIT-BIH Arrhythmia Database** — PhysioNet
- 109,451 beat segments — 256 samples each @ 360 Hz
- 5 classes (AAMI standard): **N, S, V, F, Q**
- Split: 70% train / 15% val / 15% test (stratified)

---

## Project Structure

```
ecg-arrhythmia-kan-comparison/
├── data/
│   ├── raw/                    # MIT-BIH .dat and .hea files
│   ├── processed/              # X.npy, y.npy
│   └── splits/                 # Saved train/val/test splits
├── src/
│   ├── data/
│   │   ├── preprocess.py       # Beat segmentation + normalization
│   │   └── loader.py           # Dataset class + DataLoaders
│   ├── models/
│   │   ├── cnn.py              # Baseline CNN + MLP head
│   │   ├── kan_head.py         # KAN layer (B-spline activations)
│   │   └── cnn_kan.py          # CNN + KAN combined model
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   └── evaluate.py         # Metrics + plots
│   ├── utils/
│   │   └── plots.py            # Confusion matrix, ROC, training curves
│   ├── predict.py              # Clinical prediction output
│   └── compare.py              # Final comparison table
├── experiments/
│   ├── exp1_cnn_mlp/           # CNN+MLP logs, results, plots
│   └── exp2_cnn_kan/           # CNN+KAN logs, results, plots
├── checkpoints/                # Saved model weights (.pth)
├── test_samples/               # Test .npy, .csv, .png files
├── app.py                      # Streamlit UI
├── main.py                     # Run both experiments
└── generate_test_samples.py    # Generate test data from test split
```

---

## Setup

```bash
git clone https://github.com/your-username/ecg-arrhythmia-kan-comparison.git
cd ecg-arrhythmia-kan-comparison
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## How to Run

**1. Download MIT-BIH data**
```bash
python data/download_mitbih.py
```

**2. Preprocess**
```bash
python src/data/preprocess.py
```

**3. Train both models**
```bash
python main.py               # trains CNN+MLP and CNN+KAN
python main.py --model cnn   # baseline only
python main.py --model kan   # proposed only
```

**4. Evaluate**
```bash
python -m src.training.evaluate
```

**5. Final comparison**
```bash
python -m src.compare
```

**6. Clinical prediction**
```bash
python -m src.predict --index 0 --model both
```

**7. Launch UI**
```bash
pip install streamlit
streamlit run app.py
```

---

## Streamlit UI

5-page dashboard:

| Page | Description |
|------|-------------|
| Overview | KPI cards, performance charts, radar comparison |
| Live Prediction | Dataset slider / upload .npy or .csv / upload ECG image |
| Model Comparison | Confusion matrices, per-class F1, efficiency |
| Training Analysis | Loss and accuracy curves per model |
| Results Table | Full metrics table with download |

---

## Architecture

```
Input [B, 1, 256]
      ↓
CNN Encoder (shared)
  Block 1: Conv1d(1→32)   + BN + ReLU + MaxPool → [B, 32, 128]
  Block 2: Conv1d(32→64)  + BN + ReLU + MaxPool → [B, 64, 64]
  Block 3: Conv1d(64→128) + BN + ReLU + MaxPool → [B, 128, 32]
  Block 4: Conv1d(128→256)+ BN + ReLU + MaxPool → [B, 256, 16]
  GAP                                            → [B, 256]
      ↓
MLP Head (baseline)          KAN Head (proposed)
FC(256→128)→BN→ReLU          KANLayer(256→64)
FC(128→64)→BN→ReLU           KANLayer(64→5)
FC(64→5)
      ↓
Logits [B, 5]
```

---

## Requirements

```
torch
numpy
scikit-learn
wfdb
pandas
matplotlib
tqdm
streamlit
opencv-python
pillow
scipy
```

---

## Results

**CNN+MLP outperforms CNN+KAN** on this dataset. KAN achieves competitive Macro AUC (0.992) with fewer parameters (~155K vs ~175K), but lower Macro F1 (0.817 vs 0.902) — suggesting KAN requires more data or longer training to match MLP on biomedical time-series.

---

## Clinical Disclaimer

> This tool is decision-support only and not a standalone diagnostic system. Final interpretation must be made by a licensed clinician.

---

## License

MIT License
