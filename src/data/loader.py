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

    X = np.load(x_path)
    y = np.load(y_path)

    # Split 1: train_val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Split 2: train vs val (val fraction relative to train_val)
    val_ratio_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        stratify=y_train_val,
        random_state=random_state
    )

    train_ds = ECGDataset(X_train, y_train)
    val_ds = ECGDataset(X_val, y_val)
    test_ds = ECGDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader