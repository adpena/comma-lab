# SPDX-License-Identifier: MIT
"""ff_nerv architecture — Frequency-domain NeRV (L0 SKETCH).

The substrate predicts band-limited 2D DCT coefficient grids per frame-pair;
the inflate-time path reconstructs RGB frames via inverse 2D DCT (IDCT2).
The frequency-band structure is the principled prior — low-frequency
coefficients dominate perceptual content, so the natural rate term is a
banded entropy.

Architecture (council-approved SKETCH 2026-05-12; Carmack-style explicit):

    Per-pair latent z in R^16
       |
       v
    Linear 16 -> 384                              # latent embed
       |
       v
    Reshape (1, 384, 4, 4)                        # initial freq grid
       |
       v
    Block 0..3: Conv -> sin -> PixelShuffle(2)    # 8x8 -> 16x16 -> 32x32 -> 64x64 freq grid
       |
       v
    Head dct_0 / dct_1: Conv -> band-limited      # per-frame DCT grids (64x64 of 3 channels)
       |
       v
    inflate-time IDCT2 -> 384x512 RGB (interpolated from 64x64 reconstruction)

The 64x64 frequency grid stores the lowest 64 horizontal x 64 vertical DCT
basis functions of a 384x512 RGB; high-frequency coefficients are zero by
construction (band-limited).

~200K params target. SIREN initialization (omega-derived bounds).

Council notes:
- DCT basis functions are computed at inflate-time deterministically (no
  weights), so the decoder learns COEFFICIENTS not BASIS.
- sin_frequency: 30.0 (NeRF default)
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally)

CLAUDE.md compliance:
- No silent device defaults (caller passes device)
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
class FfnervConfig:
    """Static design-time parameters for ff_nerv.

    All fields explicit (no silent defaults beyond declared ones).
    """

    latent_dim: int = 16
    """Per-pair latent dimensionality."""

    embed_dim: int = 96
    """Channels of the initial freq-grid embedding."""

    initial_grid_h: int = 4
    """Initial freq-grid height before upsample blocks."""

    initial_grid_w: int = 4
    """Initial freq-grid width before upsample blocks."""

    decoder_channels: tuple[int, ...] = (64, 48, 32, 24)
    """Per-block output channels for the freq-domain decoder."""

    sin_frequency: float = 30.0
    """NeRF-style sin activation frequency (SIREN choice)."""

    num_upsample_blocks: int = 4
    """Number of PixelShuffle(2) blocks. 4 -> 4x4 -> 64x64 freq grid."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for the 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    freq_grid_h: int = 64
    """Number of vertical DCT basis functions retained (low-freq band)."""

    freq_grid_w: int = 64
    """Number of horizontal DCT basis functions retained."""


class _SinAct(nn.Module):
    """Sine activation (SIREN/NeRF-style)."""

    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    """One Conv -> sin -> PixelShuffle(2) block in the freq-decoder."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.conv(x)))


def _build_idct2_basis(
    grid_h: int,
    grid_w: int,
    out_h: int,
    out_w: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the 2D-IDCT basis matrices for a `grid_h x grid_w` band ->
    `out_h x out_w` reconstruction.

    Returns (basis_h, basis_w) such that for a coefficient tensor C of shape
    ``(B, C, grid_h, grid_w)``:

        rgb = basis_h @ C @ basis_w.T

    where basis_h is `(out_h, grid_h)` and basis_w is `(out_w, grid_w)`.

    DCT-II convention; ortho-normal scaling.
    """
    # basis_w: (out_w, grid_w)
    n = torch.arange(out_w, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(grid_w, device=device, dtype=dtype).unsqueeze(0)
    basis_w = torch.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * out_w))
    alpha_w = torch.full((grid_w,), math.sqrt(2.0 / out_w), device=device, dtype=dtype)
    alpha_w[0] = math.sqrt(1.0 / out_w)
    basis_w = basis_w * alpha_w.unsqueeze(0)

    # basis_h: (out_h, grid_h)
    n = torch.arange(out_h, device=device, dtype=dtype).unsqueeze(1)
    k = torch.arange(grid_h, device=device, dtype=dtype).unsqueeze(0)
    basis_h = torch.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * out_h))
    alpha_h = torch.full((grid_h,), math.sqrt(2.0 / out_h), device=device, dtype=dtype)
    alpha_h[0] = math.sqrt(1.0 / out_h)
    basis_h = basis_h * alpha_h.unsqueeze(0)

    return basis_h, basis_w


class FfnervSubstrate(nn.Module):
    """Frequency-domain NeRV renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1] via IDCT2.

    The decoder predicts a `(2, 3, freq_grid_h, freq_grid_w)` DCT coefficient
    bundle per pair (2 frames x 3 RGB channels x freq band), then the
    inflate-time path applies a deterministic IDCT2 to recover RGB.
    """

    def __init__(self, cfg: FfnervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair learned latents
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # latent -> initial freq grid embedding
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_UpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        # 2 frames per pair, 3 RGB channels each -> 6 channels of DCT coefficients
        self.head_dct = nn.Conv2d(final_ch, 6, kernel_size=3, padding=1)

        # Pre-compute IDCT2 basis (registered as buffers; not trainable)
        basis_h, basis_w = _build_idct2_basis(
            cfg.freq_grid_h, cfg.freq_grid_w, cfg.output_height, cfg.output_width,
        )
        self.register_buffer("idct_basis_h", basis_h, persistent=False)
        self.register_buffer("idct_basis_w", basis_w, persistent=False)

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN init: weights ~ Uniform(-c/fan_in, c/fan_in) with c = sqrt(6/fan_in)/w."""
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render frame-pairs at the given pair indices via IDCT2 of predicted coefficients."""
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z = self.latents[pair_indices]  # (B, latent_dim)
        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        for block in self.blocks:
            h = block(h)

        # h is now (B, final_ch, ~grid_h, ~grid_w); interpolate to exact freq grid if needed
        if h.shape[-2:] != (self.cfg.freq_grid_h, self.cfg.freq_grid_w):
            h = F.interpolate(
                h,
                size=(self.cfg.freq_grid_h, self.cfg.freq_grid_w),
                mode="bilinear",
                align_corners=False,
            )

        dct_coeffs = self.head_dct(h)  # (B, 6, freq_h, freq_w)

        # Split into 2 frames * 3 channels each
        # dct_coeffs[:, :3] -> rgb_0; dct_coeffs[:, 3:] -> rgb_1
        rgb_0 = self._idct2(dct_coeffs[:, :3])
        rgb_1 = self._idct2(dct_coeffs[:, 3:])
        return torch.sigmoid(rgb_0), torch.sigmoid(rgb_1)

    def _idct2(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply 2D inverse DCT-II to coefficients of shape (B, C, freq_h, freq_w).

        Returns (B, C, output_h, output_w).
        """
        # basis_h: (out_h, freq_h); basis_w: (out_w, freq_w)
        # rgb = basis_h @ coeffs @ basis_w.T
        # einsum: 'hi,bcij,wj->bchw'
        return torch.einsum(
            "hi,bcij,wj->bchw", self.idct_basis_h, coeffs, self.idct_basis_w
        )

    def num_parameters(self) -> int:
        """Total trainable parameter count (target ~200K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
