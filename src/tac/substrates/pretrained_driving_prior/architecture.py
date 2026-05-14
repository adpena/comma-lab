# SPDX-License-Identifier: MIT
"""Small contest-overfit renderer for the pre-trained driving prior substrate.

The renderer is intentionally TINY (under 50K params at FP16 ≈ 25-40 KB
after brotli). The score gains come from:

1. The codebook (frozen dashcam-statistical prior; offline distillation)
2. The score-aware Lagrangian (eval-roundtrip + Atick-Redlich cooperative-receiver)
3. The per-pair int8 residual encoding the contest-specific delta

The renderer maps ``(pair_idx, x, y, t)`` → RGB. Per HNeRV parity discipline L12,
every line in this file is reviewable in 30 seconds.

Architecture:

* Input: (B, 4) coordinate tensor (x_norm, y_norm, pair_idx_norm, foveation)
* Hidden: SIREN-style sine-activated MLP, 2-4 layers, hidden_dim ~ 64
* Output: (B, 3) RGB in [0, 1]

The renderer renders one full image at a time at scorer resolution (384, 512),
then inflate.py upsamples to camera resolution (874, 1164) via the canonical
``write_rgb_pair_to_raw`` helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EVAL_HW: tuple[int, int] = (384, 512)
"""Scorer resolution (height, width)."""


@dataclass(frozen=True)
class DrivingPriorRendererConfig:
    """Static design-time parameters for the renderer."""

    hidden_dim: int = 64
    num_hidden_layers: int = 3
    coord_dim: int = 4
    output_height: int = EVAL_HW[0]
    output_width: int = EVAL_HW[1]
    first_omega: float = 30.0
    """SIREN-style first-layer omega (Sitzmann signature)."""
    hidden_omega: float = 1.0


class _SineLayer(nn.Module):
    """Single SIREN-style sine-activated linear layer."""

    def __init__(self, in_features: int, out_features: int, *, omega: float, is_first: bool) -> None:
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        # Sitzmann initialization.
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = (6.0 / in_features) ** 0.5 / omega
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class DrivingPriorRenderer(nn.Module):
    """Small coordinate-MLP renderer with SIREN activations.

    Total params at hidden=64, layers=3 (≈ 4 → 64 → 64 → 64 → 3) ≈ ~12K params
    ≈ 25 KB at FP16. After brotli ~15-20 KB.
    """

    def __init__(self, cfg: DrivingPriorRendererConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_features = cfg.coord_dim
        # First layer.
        layers.append(
            _SineLayer(in_features, cfg.hidden_dim, omega=cfg.first_omega, is_first=True)
        )
        # Hidden layers.
        for _ in range(cfg.num_hidden_layers - 1):
            layers.append(
                _SineLayer(
                    cfg.hidden_dim, cfg.hidden_dim, omega=cfg.hidden_omega, is_first=False
                )
            )
        self.net = nn.Sequential(*layers)
        # Final linear layer (no sine).
        self.head = nn.Linear(cfg.hidden_dim, 3)

        # Cache coordinate grid (lazy; rebuilt on device when first used).
        self._coord_grid: torch.Tensor | None = None

    def _get_coord_grid(self, device: torch.device) -> torch.Tensor:
        """Build/cache the (H, W, 2) normalized (x, y) coordinate grid."""
        if self._coord_grid is not None and self._coord_grid.device == device:
            return self._coord_grid
        h, w = self.cfg.output_height, self.cfg.output_width
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        self._coord_grid = grid
        return grid

    def render_pair(self, pair_idx: int, num_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one pair of RGB images (frames 2k and 2k+1).

        Returns:
            (rgb_0, rgb_1) each shape (1, 3, H, W) in [0, 1].
        """
        device = next(self.parameters()).device
        grid = self._get_coord_grid(device)  # (H, W, 2)
        h, w = grid.shape[:2]
        # Per-pair t in [-1, 1].
        t0 = torch.tensor(
            [2.0 * (2 * pair_idx) / max(1, 2 * num_pairs - 1) - 1.0],
            device=device,
        )
        t1 = torch.tensor(
            [2.0 * (2 * pair_idx + 1) / max(1, 2 * num_pairs - 1) - 1.0],
            device=device,
        )
        # Foveation feature (placeholder): radial distance from center.
        radius = grid.pow(2).sum(dim=-1, keepdim=True).sqrt()  # (H, W, 1)

        coords_2d = grid.reshape(-1, 2)  # (H*W, 2)
        fov_flat = radius.reshape(-1, 1)  # (H*W, 1)

        def _render_one_frame(t_scalar: torch.Tensor) -> torch.Tensor:
            t_expanded = t_scalar.expand(coords_2d.shape[0], 1)
            features = torch.cat([coords_2d, t_expanded, fov_flat], dim=-1)
            hidden = self.net(features)
            rgb = torch.sigmoid(self.head(hidden))  # (H*W, 3) in [0, 1]
            return rgb.permute(1, 0).reshape(1, 3, h, w)

        rgb_0 = _render_one_frame(t0)
        rgb_1 = _render_one_frame(t1)
        return rgb_0, rgb_1


__all__ = [
    "EVAL_HW",
    "DrivingPriorRenderer",
    "DrivingPriorRendererConfig",
]
