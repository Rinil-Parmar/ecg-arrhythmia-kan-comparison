import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    def __init__(self, X, y):
        # X: [N, 256] -> [N, 1, 256] for Conv1D
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    data_dir="data/processed",
    splits_dir="data/splits",
    batch_size=128,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    num_workers=0
):
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing files in {data_dir}. Expected X.npy and y.npy")

    # ── Load splits if already saved ──────────
    splits_exist = all(
        os.path.exists(os.path.join(splits_dir, f))
        for f in ["X_train.npy", "y_train.npy",
                  "X_val.npy",   "y_val.npy",
                  "X_test.npy",  "y_test.npy"]
    )

    if splits_exist:
        print("Loading saved splits from data/splits/ ...")
        X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
        y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
        X_val   = np.load(os.path.join(splits_dir, "X_val.npy"))
        y_val   = np.load(os.path.join(splits_dir, "y_val.npy"))
        X_test  = np.load(os.path.join(splits_dir, "X_test.npy"))
        y_test  = np.load(os.path.join(splits_dir, "y_test.npy"))

    else:
        print("Splitting data and saving to data/splits/ ...")
        X = np.load(x_path)
        y = np.load(y_path)

        # Split 1: train_val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # Split 2: train vs val
        val_ratio_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio_adjusted,
            stratify=y_train_val,
            random_state=random_state
        )

        # ── Save splits ───────────────────────
        os.makedirs(splits_dir, exist_ok=True)
        np.save(os.path.join(splits_dir, "X_train.npy"), X_train)
        np.save(os.path.join(splits_dir, "y_train.npy"), y_train)
        np.save(os.path.join(splits_dir, "X_val.npy"),   X_val)
        np.save(os.path.join(splits_dir, "y_val.npy"),   y_val)
        np.save(os.path.join(splits_dir, "X_test.npy"),  X_test)
        np.save(os.path.join(splits_dir, "y_test.npy"),  y_test)
        print(f"Splits saved to {splits_dir}/")

    # ── Build datasets + loaders ──────────────
    train_ds = ECGDataset(X_train, y_train)
    val_ds   = ECGDataset(X_val,   y_val)
    test_ds  = ECGDataset(X_test,  y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader