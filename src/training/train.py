"""
Training Loop — ECG Arrhythmia Classification
----------------------------------------------
Handles training for both:
  - ECGCNN        (baseline)
  - ECGCNNWithKAN (proposed)

Features:
  - Class-weighted loss (handles MIT-BIH imbalance)
  - LR scheduler (ReduceLROnPlateau)
  - Early stopping
  - Best model checkpointing
  - Per-epoch logging to CSV
  - Training time tracking
"""

import os
import time
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# Compute class weights from training labels
# ─────────────────────────────────────────────
def compute_class_weights(loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle class imbalance.
    Returns: [num_classes] tensor on device.
    """
    counts = torch.zeros(num_classes)
    for _, y in loader:
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ─────────────────────────────────────────────
# One epoch — train
# ─────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Returns: (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping — important for KAN spline stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# One epoch — validate
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Returns: (avg_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss   = criterion(logits, y)

        total_loss += loss.item() * X.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += X.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> dict:
    """
    Full training loop with early stopping and checkpointing.

    Args:
        model        : ECGCNN or ECGCNNWithKAN
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        config       : dict with keys:
                         num_classes, epochs, lr, weight_decay,
                         patience, checkpoint_path, log_path,
                         device (optional, auto-detected if absent)

    Returns:
        history dict with train/val loss & accuracy per epoch
    """

    # ── Setup ─────────────────────────────────
    device = config.get("device") or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device)

    num_classes      = config["num_classes"]
    epochs           = config.get("epochs", 50)
    lr               = config.get("lr", 1e-3)
    weight_decay     = config.get("weight_decay", 1e-4)
    patience         = config.get("patience", 10)
    checkpoint_path  = config["checkpoint_path"]
    log_path         = config.get("log_path", None)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # ── Class-weighted loss ────────────────────
    print("Computing class weights ...")
    class_weights = compute_class_weights(train_loader, num_classes, device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + Scheduler ──────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── CSV logger ────────────────────────────
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", newline="")
        writer   = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
    else:
        log_file, writer = None, None

    # ── Training state ────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    total_start      = time.time()

    print(f"\nDevice     : {device}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Epochs     : {epochs}  |  LR: {lr}  |  Patience: {patience}")
    print("=" * 65)

    # ── Epoch loop ────────────────────────────
    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        ep_time    = time.time() - ep_start

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if writer:
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])
            log_file.flush()

        print(
            f"Ep {epoch:03d}/{epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {ep_time:.1f}s"
        )

        # ── Checkpoint best model ──────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "config":     config,
            }, checkpoint_path)
            print(f"           ✓ Saved best model  (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # ── Early stopping ────────────────────
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - total_start
    print("=" * 65)
    print(f"Training complete — {total_time:.1f}s total")
    print(f"Best val loss : {best_val_loss:.4f}")

    if log_file:
        log_file.close()

    history["total_time_sec"] = total_time
    history["best_val_loss"]  = best_val_loss
    return history