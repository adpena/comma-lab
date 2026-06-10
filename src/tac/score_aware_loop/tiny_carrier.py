# SPDX-License-Identifier: MIT
"""A tiny per-pair RGB carrier for the $0 score-aware-descent proof.

The carrier is deliberately minimal: a learnable per-pair latent -> small MLP/
conv that produces a ``(B, 2, 3, H, W)`` RGB pair in ``[0, 255]``. It exists ONLY
to demonstrate that the PR95-faithful loop's gradient ACTUALLY reduces the exact
SegNet ``d_seg`` on the live frozen scorer — the headline of task #76. It is not a
contest carrier (no archive grammar); the *loop* it exercises is what #74/#63/#65
consume, against their own real carriers.

To make the descent observable in seconds on CPU, the carrier renders at a small
scorer resolution. The trainer up/down-samples to the exact scorer input size via
the canonical eval roundtrip, so the gradient path is faithful even at tiny res.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyPairCarrier(nn.Module):
    """Per-pair learnable latent -> conv decoder -> ``(B, 2, 3, H, W)`` RGB pair.

    Args:
        n_pairs: number of trainable pairs (one latent row each).
        latent_dim: per-pair latent width.
        out_hw: ``(H, W)`` of the rendered frame.
        base_channels: decoder width.
    """

    def __init__(
        self,
        n_pairs: int,
        *,
        latent_dim: int = 16,
        out_hw: tuple[int, int] = (48, 64),
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        h, w = self.out_hw
        # Start grid is 1/8 of the output, upsampled x8 by two PixelShuffle x2 +
        # one bilinear x2 — cheap but enough spatial capacity to move argmax.
        self.start_h = max(h // 8, 1)
        self.start_w = max(w // 8, 1)
        self.latent_dim = int(latent_dim)
        self.latents = nn.Parameter(
            torch.randn(self.n_pairs, self.latent_dim) * 0.1
        )
        c = int(base_channels)
        self.proj = nn.Linear(self.latent_dim, c * self.start_h * self.start_w)
        # 2 frames per pair => 2 output paths share the trunk, differ in head.
        self.trunk = nn.Sequential(
            nn.Conv2d(c, c * 4, 3, padding=1),
            nn.PixelShuffle(2),  # x2
            nn.GELU(),
            nn.Conv2d(c, c * 4, 3, padding=1),
            nn.PixelShuffle(2),  # x4
            nn.GELU(),
        )
        self.head0 = nn.Conv2d(c, 3, 3, padding=1)
        self.head1 = nn.Conv2d(c, 3, 3, padding=1)

    def reconstruct_pair(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rgb_0, rgb_1)`` each ``(B, 3, H, W)`` in ``[0, 255]``."""
        b = idx.shape[0]
        z = self.latents[idx]
        c = self.proj(z).reshape(b, -1, self.start_h, self.start_w)
        feat = self.trunk(c)
        # trunk upsampled x4; bilinear x2 to reach out_hw.
        feat = torch.nn.functional.interpolate(
            feat, size=self.out_hw, mode="bilinear", align_corners=False
        )
        rgb0 = torch.sigmoid(self.head0(feat)) * 255.0
        rgb1 = torch.sigmoid(self.head1(feat)) * 255.0
        return rgb0, rgb1

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 2, 3, H, W)`` RGB pair in ``[0, 255]``."""
        rgb0, rgb1 = self.reconstruct_pair(idx)
        return torch.stack([rgb0, rgb1], dim=1)
