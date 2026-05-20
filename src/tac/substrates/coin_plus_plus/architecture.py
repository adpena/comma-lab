# SPDX-License-Identifier: MIT
"""coin_plus_plus architecture — COIN++ (L0 SKETCH).

Modulation-based parameter-efficient INR. Operator 5-tier fit-ranking
MODERATE FIT ⭐⭐⭐: COIN++ ships a SHARED base coord-MLP + per-pair
modulation vectors. Different rate-tradeoff than NeRV-family.

Literature anchor: Dupont et al. ICML 2022 COIN++ (paper-ID literature
reference per BUILD task #1090). The modulation paradigm builds on Perez
et al. 2017 FiLM (Feature-wise Linear Modulation).

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair modulation m in R^MOD_DIM (default 64)
       |
       v
    For each pixel (x, y) in [0, H) x [0, W) and frame t in {0, 1}:
        c = (x_norm, y_norm, t_norm)  # normalized to [-1, 1]
        rgb_pixel = F_phi_mod_m(c)
       |
       v
    Stack: rgb_0 (B, 3, H, W), rgb_1 (B, 3, H, W)

The shared base network F_phi is a coord-MLP with NUM_LAYERS hidden
layers, HIDDEN_DIM channels each, SIREN activations. The modulation m
is broadcast across all pixels for that pair; it scales+shifts each
hidden layer's activations (FiLM-style).

CLAUDE.md compliance:
- No silent device defaults
- No scorer load
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2


@dataclass(frozen=True)
class CoinplusplusConfig:
    """Static design-time parameters for coin_plus_plus."""

    modulation_dim: int = 64
    """Per-pair modulation vector dim. THIS IS THE LATENT RATE.

    CARGO-CULTED at L0 (per cargo-cult audit in __init__.py); sweep at L1
    is critical because the per-pair modulation IS the variable-rate term.
    """

    hidden_dim: int = 96
    """Coord-MLP hidden channel count (SHARED base network)."""

    num_hidden_layers: int = 4
    """Coord-MLP depth (SHARED base network)."""

    sin_frequency: float = 30.0
    """SIREN activation frequency (NeRF default)."""

    coord_input_dim: int = 3
    """Input coordinate dim: (x_norm, y_norm, t_norm)."""

    output_channels: int = 3
    """RGB output channels."""

    num_pairs: int = _PAIRS

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _ModulatedLinear(nn.Module):
    """Linear layer with FiLM-style scale+shift modulation.

    out = sin(w * (gamma * (W @ x + b) + beta))

    where gamma = mod_gamma_proj(m), beta = mod_beta_proj(m).
    """

    def __init__(
        self, in_features: int, out_features: int, mod_dim: int, sin_freq: float
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mod_gamma_proj = nn.Linear(mod_dim, out_features)
        self.mod_beta_proj = nn.Linear(mod_dim, out_features)
        self.sin_freq = float(sin_freq)
        # Initialize modulation projections so gamma defaults to ~1.0 and
        # beta to ~0.0 (modulation is a perturbation around the unmodulated
        # base).
        with torch.no_grad():
            self.mod_gamma_proj.weight.zero_()
            self.mod_gamma_proj.bias.fill_(1.0)
            self.mod_beta_proj.weight.zero_()
            self.mod_beta_proj.bias.zero_()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: (B, N_pixels, in_features), m: (B, mod_dim)
        h = self.linear(x)  # (B, N_pixels, out_features)
        gamma = self.mod_gamma_proj(m).unsqueeze(1)  # (B, 1, out_features)
        beta = self.mod_beta_proj(m).unsqueeze(1)  # (B, 1, out_features)
        h = gamma * h + beta
        return torch.sin(self.sin_freq * h)


class CoinplusplusSubstrate(nn.Module):
    """COIN++ renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The forward path:
    1. Look up per-pair modulation m.
    2. Build pixel coordinate grid (x_norm, y_norm, t_norm) for t in {0, 1}.
    3. Forward through the modulated coord-MLP for ALL pixels in the grid.
    4. Reshape to (B, 3, H, W) for each frame.
    """

    def __init__(self, cfg: CoinplusplusConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair modulation vectors (THE LATENT TO BE SHIPPED).
        self.modulations = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.modulation_dim).normal_(std=0.02)
        )

        # Shared base coord-MLP (FiLM-modulated by m at each hidden layer).
        # Layer 0: coord_input_dim -> hidden_dim
        # Layers 1..num_hidden_layers-1: hidden_dim -> hidden_dim
        # Final layer: hidden_dim -> output_channels (unmodulated; sigmoid for RGB)
        layers: list[nn.Module] = [
            _ModulatedLinear(
                cfg.coord_input_dim, cfg.hidden_dim, cfg.modulation_dim, cfg.sin_frequency
            )
        ]
        for _ in range(cfg.num_hidden_layers - 1):
            layers.append(
                _ModulatedLinear(
                    cfg.hidden_dim, cfg.hidden_dim, cfg.modulation_dim, cfg.sin_frequency
                )
            )
        self.mod_layers = nn.ModuleList(layers)
        # Final unmodulated output head.
        self.output_head = nn.Linear(cfg.hidden_dim, cfg.output_channels)

        self._siren_init()

        # Pre-build the canonical pixel coordinate grid (lazy; built on first
        # forward call to the right device).
        self._coords_cache: torch.Tensor | None = None

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
            # Re-init mod projections to identity-ish (overrides SIREN init).
            for layer in self.mod_layers:
                if isinstance(layer, _ModulatedLinear):
                    layer.mod_gamma_proj.weight.zero_()
                    layer.mod_gamma_proj.bias.fill_(1.0)
                    layer.mod_beta_proj.weight.zero_()
                    layer.mod_beta_proj.bias.zero_()

    def _build_coord_grid(self, device: torch.device) -> torch.Tensor:
        """Build the canonical (H*W*2, 3) coordinate grid for frames {0, 1}.

        Cached on first forward call so subsequent calls reuse the buffer.
        """
        if self._coords_cache is not None and self._coords_cache.device == device:
            return self._coords_cache
        H, W = self.cfg.output_height, self.cfg.output_width
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Frame 0 coords + frame 1 coords stacked: (2*H*W, 3)
        coords_frame_0 = torch.stack(
            [grid_x.flatten(), grid_y.flatten(),
             torch.full((H * W,), -1.0, device=device)], dim=-1
        )
        coords_frame_1 = torch.stack(
            [grid_x.flatten(), grid_y.flatten(),
             torch.full((H * W,), 1.0, device=device)], dim=-1
        )
        coords = torch.cat([coords_frame_0, coords_frame_1], dim=0)  # (2*H*W, 3)
        self._coords_cache = coords
        return coords

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        B = pair_indices.shape[0]
        H, W = self.cfg.output_height, self.cfg.output_width
        m = self.modulations[pair_indices]  # (B, mod_dim)

        coords = self._build_coord_grid(self.modulations.device)  # (2*H*W, 3)
        # Expand coords across batch: (B, 2*H*W, 3)
        coords_batched = coords.unsqueeze(0).expand(B, -1, -1)

        h = coords_batched
        for layer in self.mod_layers:
            h = layer(h, m)
        rgb = torch.sigmoid(self.output_head(h))  # (B, 2*H*W, 3)

        # Split into frame 0 / frame 1
        N_pixels = H * W
        rgb_0 = rgb[:, :N_pixels, :].view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        rgb_1 = rgb[:, N_pixels:, :].view(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count (target ~140K with mod_dim=64)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
