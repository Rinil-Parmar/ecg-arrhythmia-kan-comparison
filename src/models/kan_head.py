"""
Kolmogorov-Arnold Network (KAN) Classification Head
-----------------------------------------------------
Replaces the MLP head in the CNN baseline.

Core idea:
  - MLP : y = sum( W * activation(x) )   — fixed activations, learnable weights
  - KAN  : y = sum( learnable_spline(x) ) — learnable activations on edges

Each KAN layer learns a B-spline activation per (input, output) pair.
This gives KAN its interpretability and parameter efficiency advantage.

Input : [B, in_features]   (encoder feature vector)
Output: [B, num_classes]   (logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# B-Spline basis computation
# ─────────────────────────────────────────────
def b_spline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute B-spline basis functions using Cox-de Boor recursion.

    Args:
        x    : [B, in_features]
        grid : [in_features, G+1]  — grid points
        k    : spline order

    Returns:
        basis: [B, in_features, G]  — G = num_intervals
    """
    # Clamp x to grid range
    x = x.unsqueeze(-1)  # [B, in_features, 1]

    # Order-0 basis: 1 if grid[i] <= x < grid[i+1] else 0
    basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()  # [B, in_f, G+k]

    # Cox-de Boor recursion
    for order in range(1, k + 1):
        left_num  = x - grid[:, :-(order + 1)]              # [B, in_f, G]
        left_den  = grid[:, order:-1] - grid[:, :-(order + 1)]
        right_num = grid[:, (order + 1):] - x
        right_den = grid[:, (order + 1):] - grid[:, 1:-order]

        # Safe division (avoid 0/0)
        left  = torch.where(left_den  != 0, left_num  / left_den,  torch.zeros_like(left_num))
        right = torch.where(right_den != 0, right_num / right_den, torch.zeros_like(right_num))

        basis = left * basis[:, :, :-1] + right * basis[:, :, 1:]

    return basis  # [B, in_features, num_intervals]


# ─────────────────────────────────────────────
# Single KAN Layer
# ─────────────────────────────────────────────
class KANLayer(nn.Module):
    """
    One KAN layer: learnable spline activation on every (input → output) edge.

    For each output neuron j and input neuron i:
        phi_{i,j}(x_i) = w_b * silu(x_i) + w_s * spline(x_i)

    Total parameters per layer:
        out_features * in_features * (grid_size + spline_order)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-1.0, 1.0),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # Number of B-spline basis functions
        self.num_basis = grid_size + spline_order  # G + k

        # Grid: [in_features, G + 2k + 1]  (extended grid for order-k splines)
        grid_pts = torch.linspace(grid_range[0], grid_range[1],
                                  grid_size + 1)  # G+1 interior points
        # Extend grid on both sides by spline_order steps
        step = (grid_range[1] - grid_range[0]) / grid_size
        left_ext  = grid_pts[0]  - step * torch.arange(spline_order, 0, -1)
        right_ext = grid_pts[-1] + step * torch.arange(1, spline_order + 1)
        full_grid = torch.cat([left_ext, grid_pts, right_ext])  # [G + 2k + 1]
        # Expand to [in_features, G + 2k + 1]
        self.register_buffer("grid", full_grid.unsqueeze(0).expand(in_features, -1).clone())

        # Learnable spline coefficients: [out_features, in_features, num_basis]
        self.spline_weight = nn.Parameter(
            torch.zeros(out_features, in_features, self.num_basis)
        )
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)

        # Residual scale (SiLU base activation weight)
        self.base_weight = nn.Parameter(torch.ones(out_features, in_features))

        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(in_features)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            out: [B, out_features]
        """
        x = self.layer_norm(x)                        # [B, in_f]

        # Base activation (SiLU residual connection)
        base = F.silu(x)                              # [B, in_f]
        base_out = torch.einsum("oi,bi->bo", self.base_weight, base)  # [B, out_f]

        # Spline activation
        basis = b_spline_basis(x, self.grid, self.spline_order)  # [B, in_f, num_basis]
        # spline_weight: [out_f, in_f, num_basis]
        spline_out = torch.einsum("oik,bik->bo", self.spline_weight, basis)  # [B, out_f]

        out = base_out + spline_out                   # [B, out_features]
        out = self.dropout(out)
        return out


# ─────────────────────────────────────────────
# KAN Classification Head
# ─────────────────────────────────────────────
class KANHead(nn.Module):
    """
    2-layer KAN head to replace the MLP classifier.

    Architecture:
        KANLayer(256 → 64) → KANLayer(64 → num_classes)

    Deliberately shallow — KAN's expressiveness comes from
    learnable activations, not depth.

    Input : [B, 256]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        in_features: int = 256,
        hidden_features: int = 64,
        num_classes: int = 5,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.kan1 = KANLayer(
            in_features=in_features,
            out_features=hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=dropout,
        )
        self.kan2 = KANLayer(
            in_features=hidden_features,
            out_features=num_classes,
            grid_size=grid_size,
            spline_order=spline_order,
            dropout=0.0,   # no dropout on final layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features]
        Returns:
            logits: [B, num_classes]
        """
        x = self.kan1(x)   # [B, hidden]
        x = self.kan2(x)   # [B, num_classes]
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    head = KANHead(in_features=256, hidden_features=64, num_classes=5)
    x = torch.randn(32, 256)
    out = head(x)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {head.count_parameters():,}")
    print("KAN head — OK")