import os
import numpy as np
import wfdb


RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# AAMI-like grouped labels (simple 5-class mapping)
LABEL_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # Normal group
    "A": 1, "a": 1, "J": 1, "S": 1,           # Supraventricular group
    "V": 2, "E": 2,                           # Ventricular group
    "F": 3,                                   # Fusion
    "/": 4, "f": 4, "Q": 4                    # Unknown / paced
}

RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]


def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def extract_beats(signal, ann_samples, ann_symbols, window=128):
    X, y = [], []
    n = len(signal)

    for s, sym in zip(ann_samples, ann_symbols):
        if sym not in LABEL_MAP:
            continue
        left = s - window
        right = s + window
        if left < 0 or right >= n:
            continue
        beat = signal[left:right]
        beat = normalize(beat).astype(np.float32)
        X.append(beat)
        y.append(LABEL_MAP[sym])

    return X, y


def main():
    all_X, all_y = [], []

    for rec in RECORDS:
        rec_path = os.path.join(RAW_DIR, rec)
        if not os.path.exists(rec_path + ".dat"):
            print(f"Skipping {rec}: file not found")
            continue

        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        # Use lead 0 (MLII usually)
        signal = record.p_signal[:, 0]
        X, y = extract_beats(signal, ann.sample, ann.symbol, window=128)

        all_X.extend(X)
        all_y.extend(y)
        print(f"{rec}: beats={len(X)}")

    X_arr = np.array(all_X, dtype=np.float32)   # shape: [N, 256]
    y_arr = np.array(all_y, dtype=np.int64)     # shape: [N]

    np.save(os.path.join(OUT_DIR, "X.npy"), X_arr)
    np.save(os.path.join(OUT_DIR, "y.npy"), y_arr)

    print("\nSaved:")
    print(f"{OUT_DIR}/X.npy shape={X_arr.shape}")
    print(f"{OUT_DIR}/y.npy shape={y_arr.shape}")


if __name__ == "__main__":
    main()