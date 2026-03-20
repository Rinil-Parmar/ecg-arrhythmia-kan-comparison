"""
Baseline 1D CNN for ECG Arrhythmia Classification
Input : [batch, 1, 256]
Output: [batch, num_classes]  (logits)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool (optional dropout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────
# Baseline CNN
# ─────────────────────────────────────────────
class ECGCNN(nn.Module):
    """
    4-block 1D CNN encoder + MLP classification head.

    Architecture
    ─────────────────────────────────────────────
    Input  : [B, 1, 256]

    Conv blocks (each halves the temporal dimension):
      Block-1 : 1  → 32   ch | L: 256 → 128
      Block-2 : 32 → 64   ch | L: 128 → 64
      Block-3 : 64 → 128  ch | L: 64  → 32
      Block-4 : 128→ 256  ch | L: 32  → 16

    Global Average Pooling : [B, 256, 16] → [B, 256]

    MLP head:
      FC(256→128) → BN → ReLU → Dropout
      FC(128→64)  → BN → ReLU → Dropout
      FC(64→num_classes)

    Output : [B, num_classes]  (raw logits)
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────
        self.encoder = nn.Sequential(
            ConvBlock(1,   32,  kernel_size=7, pool_size=2, dropout=dropout * 0.5),
            ConvBlock(32,  64,  kernel_size=5, pool_size=2, dropout=dropout * 0.5),
            ConvBlock(64,  128, kernel_size=3, pool_size=2, dropout=dropout),
            ConvBlock(128, 256, kernel_size=3, pool_size=2, dropout=dropout),
        )

        # Global average pooling → [B, 256]
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── MLP Classifier Head ───────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),

            nn.Linear(64, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    # ─────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 256]
        Returns:
            logits: [B, num_classes]
        """
        x = self.encoder(x)          # [B, 256, 16]
        x = self.gap(x).squeeze(-1)  # [B, 256]
        x = self.classifier(x)       # [B, num_classes]
        return x

    # ─────────────────────────────────────────────
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder features before classifier (for KAN head reuse)."""
        x = self.encoder(x)
        x = self.gap(x).squeeze(-1)
        return x  # [B, 256]

    # ─────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = ECGCNN(num_classes=5, dropout=0.3)
    x = torch.randn(32, 1, 256)          # batch of 32
    logits = model(x)
    features = model.get_features(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Parameters   : {model.count_parameters():,}")
    print("CNN baseline — OK")