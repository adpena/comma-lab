# SPDX-License-Identifier: MIT
"""wavelet architecture — 2D DWT subbands + score-aware synthesis MLP.

L0 SKETCH scaffold per operator approval 2026-05-12. The substrate stores per-pair
DWT coefficients (LL, LH, HL, HH at depth-1) using Daubechies-4 filters, and
synthesizes RGB via a small MLP that consumes the IDWT-reconstructed feature
field plus a frame-conditional FiLM modulation.

Architecture (council-sketch 2026-05-12; not yet empirical-anchored):

    Per-pair subband coefficients:
        LL: R^(num_pairs, C, H/2, W/2)  — approximation
        LH: R^(num_pairs, C, H/2, W/2)  — horizontal detail
        HL: R^(num_pairs, C, H/2, W/2)  — vertical detail
        HH: R^(num_pairs, C, H/2, W/2)  — diagonal detail
       |
       v (IDWT via fixed Daubechies-4 synthesis filters; non-learnable)
       |
       v
    Feature field: R^(num_pairs, C, H, W)
       |
       v (FiLM modulation: per-frame gamma/beta from a small 2-row embedding)
       |
       v
    Synthesis MLP: 1x1 convs -> RGB
       |
       v
    Per-pair (rgb_0, rgb_1), each (B, 3, H, W) in [0, 1].

Council notes:
- Daubechies-4 (DB4) filters are FIXED at design-time (not learnable). This
  preserves the Mallat structure that the rate-axis depends on.
- Total param target: ~150K (synthesis + FiLM are tiny; subbands are the rate budget)
- The IDWT is a standard 2-channel-per-axis filter bank; we implement it as
  a transposed conv with fixed weights initialized from the DB4 coefficients.

CLAUDE.md compliance:
- No silent device defaults (caller passes device)
- No scorer loading inside this module
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
_PAIRS = _NUM_FRAMES // 2  # 600 pairs


# Daubechies-4 (DB4) wavelet filter coefficients (orthonormal, 4 taps)
# Reference: Daubechies (1992), "Ten Lectures on Wavelets", Table 6.1.
_DB4_LO: tuple[float, ...] = (
    (1.0 + math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
    (3.0 + math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
    (3.0 - math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
    (1.0 - math.sqrt(3.0)) / (4.0 * math.sqrt(2.0)),
)


def _db4_hi() -> tuple[float, ...]:
    """High-pass filter from QMF relationship: h_hi[k] = (-1)^k * h_lo[N-1-k]."""
    lo = _DB4_LO
    return tuple((-1.0) ** k * lo[len(lo) - 1 - k] for k in range(len(lo)))


@dataclass(frozen=True)
class WaveletConfig:
    """Static design-time parameters for the wavelet substrate (L0 SKETCH)."""

    coeff_channels: int = 8
    """Number of feature channels at the subband level."""

    synthesis_hidden: int = 32
    """Hidden size of the post-IDWT synthesis MLP."""

    synthesis_layers: int = 3
    """Layers of the synthesis MLP (incl. output)."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for 1200-frame contest video)."""

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SynthesisMLP(nn.Module):
    """Tiny shared synthesis MLP: per-pixel feature -> RGB."""

    def __init__(self, in_ch: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_ch
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(prev, hidden, kernel_size=1))
            layers.append(nn.GELU())
            prev = hidden
        layers.append(nn.Conv2d(prev, 3, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sigmoid(self.net(x))


def _build_db4_idwt_kernels(in_ch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-channel separable IDWT synthesis-filter kernels.

    Returns (kernel_lo, kernel_hi), each shape (in_ch, 1, 4) for 1-D filters
    applied per-axis. The caller is responsible for the upsample + filter
    bank composition.
    """
    lo = torch.tensor(_DB4_LO, dtype=torch.float32)
    hi = torch.tensor(_db4_hi(), dtype=torch.float32)
    # Tile to (in_ch, 1, 4) for grouped-conv-style application
    k_lo = lo.view(1, 1, 4).repeat(in_ch, 1, 1)
    k_hi = hi.view(1, 1, 4).repeat(in_ch, 1, 1)
    return k_lo, k_hi


def _idwt_2d(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
    """2D depth-1 inverse DWT using DB4 filters via separable 1-D filterbanks.

    Args:
        LL/LH/HL/HH: each (B, C, H/2, W/2)

    Returns:
        ``(B, C, H, W)``
    """
    B, C, h_half, w_half = LL.shape
    k_lo, k_hi = _build_db4_idwt_kernels(C)
    k_lo = k_lo.to(LL.device, LL.dtype)
    k_hi = k_hi.to(LL.device, LL.dtype)

    def _upsample_filter_1d(x_lo: torch.Tensor, x_hi: torch.Tensor, dim: int) -> torch.Tensor:
        """Upsample by 2 along `dim`, then filter with lo/hi and sum.

        We implement upsample-by-2 + filter as a transposed conv1d-equivalent
        by zero-stuffing + conv with circular-style padding (DB4 has 4 taps).
        """
        # Move target dim to -1
        x_lo_p = x_lo.movedim(dim, -1)
        x_hi_p = x_hi.movedim(dim, -1)
        orig_shape = list(x_lo_p.shape)
        # Zero-stuff: interleave with zeros to double the length
        zeros_lo = torch.zeros_like(x_lo_p)
        zeros_hi = torch.zeros_like(x_hi_p)
        up_lo = torch.stack([x_lo_p, zeros_lo], dim=-1).flatten(-2)
        up_hi = torch.stack([x_hi_p, zeros_hi], dim=-1).flatten(-2)
        # Reshape to (B*..., C, L) for grouped conv1d
        L = up_lo.shape[-1]
        bc_lo = up_lo.reshape(-1, C, L)
        bc_hi = up_hi.reshape(-1, C, L)
        # Pad with reflect to keep length L after a 4-tap filter
        pad = 3
        bc_lo_padded = F.pad(bc_lo, (pad, 0), mode="reflect")
        bc_hi_padded = F.pad(bc_hi, (pad, 0), mode="reflect")
        out_lo = F.conv1d(bc_lo_padded, k_lo, groups=C)
        out_hi = F.conv1d(bc_hi_padded, k_hi, groups=C)
        out = out_lo + out_hi  # length L
        # Reshape back
        new_shape = orig_shape[:-1] + [L]
        out_full = out.reshape(*new_shape)
        # Move dim back
        return out_full.movedim(-1, dim)

    # First: vertical filterbank — combine LL/LH (low-freq column) and HL/HH (high-freq column)
    low_col = _upsample_filter_1d(LL, LH, dim=-2)  # (B, C, H, W/2)
    high_col = _upsample_filter_1d(HL, HH, dim=-2)
    # Then: horizontal filterbank
    out = _upsample_filter_1d(low_col, high_col, dim=-1)  # (B, C, H, W)
    return out


class WaveletSubstrate(nn.Module):
    """Wavelet substrate: per-pair subband coefficients + IDWT + synthesis MLP.

    Forward signature mirrors sane_hnerv for trainer interop:
        forward(pair_indices) -> (rgb_0, rgb_1), each (B, 3, H, W).
    """

    def __init__(self, cfg: WaveletConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h_half = cfg.output_height // 2
        w_half = cfg.output_width // 2

        # Per-pair subbands (Mallat depth-1)
        self.coeff_ll = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.coeff_channels, h_half, w_half).normal_(std=0.05)
        )
        self.coeff_lh = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.coeff_channels, h_half, w_half).normal_(std=0.01)
        )
        self.coeff_hl = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.coeff_channels, h_half, w_half).normal_(std=0.01)
        )
        self.coeff_hh = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.coeff_channels, h_half, w_half).normal_(std=0.01)
        )

        # Frame-conditional FiLM: 2 rows (frame 0 vs 1), gamma + beta per channel
        self.film = nn.Parameter(torch.zeros(2, 2, cfg.coeff_channels, 1, 1))
        # film[:, 0, :, :, :] = gamma offset (added to 1.0); film[:, 1, :, :, :] = beta

        self.synthesis = _SynthesisMLP(cfg.coeff_channels, cfg.synthesis_hidden, cfg.synthesis_layers)

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render frame-pairs at the given pair indices.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1)``, each ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")

        LL = self.coeff_ll[pair_indices]
        LH = self.coeff_lh[pair_indices]
        HL = self.coeff_hl[pair_indices]
        HH = self.coeff_hh[pair_indices]

        # Reconstruct feature field via IDWT
        feat = _idwt_2d(LL, LH, HL, HH)  # (B, C, H, W)

        # Crop/pad to exact output size in case the filter introduced small mismatch
        if feat.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            feat = F.interpolate(
                feat,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )

        gamma_0 = 1.0 + self.film[0, 0]  # (C, 1, 1)
        beta_0 = self.film[0, 1]
        gamma_1 = 1.0 + self.film[1, 0]
        beta_1 = self.film[1, 1]

        feat_0 = feat * gamma_0 + beta_0
        feat_1 = feat * gamma_1 + beta_1

        rgb_0 = self.synthesis(feat_0)
        rgb_1 = self.synthesis(feat_1)
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
