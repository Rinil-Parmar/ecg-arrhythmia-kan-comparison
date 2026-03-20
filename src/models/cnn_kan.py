"""
CNN + KAN Model for ECG Arrhythmia Classification
---------------------------------------------------
Same CNN encoder as baseline (cnn.py).
MLP head replaced with KAN head (kan_head.py).

This is the proposed model for comparison against CNN+MLP baseline.

Input : [B, 1, 256]
Output: [B, num_classes]  (logits)
"""

import torch
import torch.nn as nn

from src.models.cnn      import ECGCNN
from src.models.kan_head import KANHead


class ECGCNNWithKAN(nn.Module):
    """
    CNN encoder (frozen or trainable) + KAN classification head.

    Architecture
    ─────────────────────────────────────────────
    Input         : [B, 1, 256]
    CNN Encoder   : 4x ConvBlock → GAP → [B, 256]
    KAN Head      : KANLayer(256→64) → KANLayer(64→num_classes)
    Output        : [B, num_classes]  (raw logits)

    Args:
        num_classes     : number of output classes (default 5 for MIT-BIH)
        dropout         : dropout rate shared across encoder + head
        grid_size       : B-spline grid intervals in KAN layers
        spline_order    : B-spline order (3 = cubic)
        freeze_encoder  : if True, CNN weights are frozen (transfer-learning mode)
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
        grid_size: int = 5,
        spline_order: int = 3,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # ── Shared CNN encoder (no classifier head) ──
        _cnn = ECGCNN(num_classes=num_classes, dropout=dropout)
        self.encoder = _cnn.encoder   # 4 ConvBlocks
        self.gap     = _cnn.gap       # AdaptiveAvgPool1d(1)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ── KAN head replaces MLP head ──────────────
        self.kan_head = KANHead(
            in_features=256,
            hidden_features=64,
            num_classes=num_classes,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
        )

    # ─────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 256]
        Returns:
            logits: [B, num_classes]
        """
        x = self.encoder(x)           # [B, 256, 16]
        x = self.gap(x).squeeze(-1)   # [B, 256]
        x = self.kan_head(x)          # [B, num_classes]
        return x

    # ─────────────────────────────────────────────
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder features (for analysis / explainability)."""
        x = self.encoder(x)
        x = self.gap(x).squeeze(-1)
        return x  # [B, 256]

    # ─────────────────────────────────────────────
    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ─────────────────────────────────────────────
    def parameter_breakdown(self) -> dict:
        """Returns parameter count per sub-module (useful for Step 10 comparison)."""
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        kan_params = sum(p.numel() for p in self.kan_head.parameters() if p.requires_grad)
        return {
            "encoder"   : enc_params,
            "kan_head"  : kan_params,
            "total"     : enc_params + kan_params,
        }


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from src.models.cnn import ECGCNN

    # ── Baseline CNN ──
    cnn_model = ECGCNN(num_classes=5, dropout=0.3)

    # ── Proposed CNN + KAN ──
    kan_model = ECGCNNWithKAN(num_classes=5, dropout=0.3, grid_size=5, spline_order=3)

    x = torch.randn(32, 1, 256)

    cnn_logits = cnn_model(x)
    kan_logits = kan_model(x)

    print("=" * 45)
    print(f"Input shape         : {x.shape}")
    print("-" * 45)
    print(f"[CNN]  Output shape : {cnn_logits.shape}")
    print(f"[CNN]  Parameters   : {cnn_model.count_parameters():,}")
    print("-" * 45)
    print(f"[KAN]  Output shape : {kan_logits.shape}")
    print(f"[KAN]  Parameters   : {kan_model.count_parameters():,}")
    breakdown = kan_model.parameter_breakdown()
    print(f"       → encoder    : {breakdown['encoder']:,}")
    print(f"       → kan_head   : {breakdown['kan_head']:,}")
    print("=" * 45)
    print("CNN+KAN model — OK")