"""Post-filter architectures for task-aware video compression.

All architectures share:
  - Residual connection: output = input + learned_correction
  - Zero-initialized output layer (starts as identity)
  - Forward takes (B, 3, H, W) float [0, 255], returns same
  - Compatible with int8 quantization via FakeQuant STE

Available variants:
  - PostFilter: standard 3-layer residual CNN
  - DilatedPostFilter: dilation=2 on middle layer (15x15 RF)
  - PixelShufflePostFilter: half-res 4-layer REN
  - PSDPostFilter: PixelShuffle + Dilated hybrid (council consensus)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostFilter(nn.Module):
    """Standard 3-layer residual CNN post-filter.

    Architecture: 3→h→h→3, 3×3 convolutions, ReLU, residual connection.
    Effective receptive field: 7×7.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class DilatedPostFilter(nn.Module):
    """PostFilter with dilation=2 on middle layer.

    Expands RF from 7×7 to 15×15 at zero param cost. Matches the
    receptive field of PoseNet's fastvit_t12 early layers.
    """

    def __init__(self, hidden: int = 16, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(3, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(
            hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True
        )
        self.conv3 = nn.Conv2d(hidden, 3, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act(self.conv1(x))
        residual = self.act(self.conv2(residual))
        residual = self.conv3(residual)
        return (x + residual).clamp(0, 255)


class PixelShufflePostFilter(nn.Module):
    """Half-resolution 4-layer REN using PixelUnshuffle/Shuffle.

    PixelUnshuffle(2) converts 3ch full-res to 12ch half-res.
    Four conv layers process at half-res where each 3×3 covers 6×6
    at full-res, aligning with scorer internal resolution.
    PixelShuffle(2) reconstructs full-res output.
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv2d(12, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True),
        )
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


class PSDPostFilter(nn.Module):
    """PixelShuffle + Dilated hybrid (expert council consensus architecture).

    Combines PixelShuffle half-res processing with dilation=2 on layer 2.
    Effective RF: 24×24 at full-res. Same params as PixelShufflePostFilter
    but with larger spatial reach from the dilated middle layer.

    This is the architecture unanimously selected by the expert panel
    (Tao, LeCun, Karpathy, Collier, Jensen Huang, Von Neumann) as the
    single best experiment to reach sub-1.6 score.
    """

    def __init__(self, hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        pad = kernel // 2
        self.conv1 = nn.Conv2d(12, hidden, kernel, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(
            hidden, hidden, kernel, padding=pad * 2, dilation=2, bias=True
        )
        self.conv3 = nn.Conv2d(hidden, hidden, kernel, padding=pad, bias=True)
        self.conv4 = nn.Conv2d(hidden, 12, kernel, padding=pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / 255.0
        h = self.down(x_norm)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        residual = self.conv4(h)
        residual = self.up(residual)
        return (x_norm + residual).clamp(0, 1) * 255.0


# ── Factory ──────────────────────────────────────────────────────────────


VARIANTS = {
    # Canonical names
    "standard": PostFilter,
    "dilated": DilatedPostFilter,
    "pixelshuffle": PixelShufflePostFilter,
    "psd": PSDPostFilter,
    # Legacy aliases (from deploy inflate_postfilter.py)
    "residual": PostFilter,
    "saliency_weighted": PostFilter,
    "segaware": PostFilter,
    "pixelshuffle_dilated": PSDPostFilter,
}


def build_postfilter(
    variant: str = "psd",
    hidden: int = 64,
    kernel: int = 3,
) -> nn.Module:
    """Build a post-filter by variant name.

    Args:
        variant: one of "standard", "dilated", "pixelshuffle", "psd",
                 or legacy aliases: "residual", "saliency_weighted",
                 "segaware", "pixelshuffle_dilated"
        hidden: hidden channel width
        kernel: conv kernel size (default 3)

    Returns:
        An nn.Module that takes (B, 3, H, W) float [0, 255] and returns same.
    """
    cls = VARIANTS.get(variant)
    if cls is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {list(VARIANTS.keys())}"
        )
    return cls(hidden=hidden, kernel=kernel)
