# SPDX-License-Identifier: MIT
"""hybrid_renderer_residual architecture — γ HNeRV-class renderer + residual basis.

Per the Fields-medal grand council 2026-05-12 γ candidate (§4.1 + §8):

End-to-end-trainable substrate where the final frame is the SUM of a
canonical HNeRV-class renderer output AND a sparse residual produced by
decoding a per-pair set of (index, value) pairs against a learned residual
basis dictionary.

Architecture (council-approved 2026-05-12 SKETCH; substrate_engineering tag):

    Per-pair latent z in R^{C_z}          # learned from the contest video
       |
       v
    Renderer g_s(z):                       # HNeRV-class (PixelShuffle + sin)
       Linear -> reshape -> [Block_i: Conv -> sin -> PixelShuffle(2)] x N
       |
       v
    Pair of RGB heads (frame_0 / frame_1)
       |
       v
    (rgb_base_0, rgb_base_1) in [0, 1]
       +
    Residual basis decoder:                # ~50K params
       For each pair: gather k basis vectors from a learned residual basis
       D in R^{B x D_v} via the per-pair (index_i, value_i) coefficients,
       decode the linear combination via a tiny MLP head to a residual
       RGB delta of shape (3, H, W), and ADD to the renderer's frames.
       |
       v
    (rgb_0, rgb_1) = clamp(rgb_base + residual, 0, 1)

Per Dykstra's "Combined: build a substrate that has hyperprior side-info AND
score-aware pose-residual stream AND a re-architected renderer with lower R*"
— the residual stream IS the pose-axis attack vector at the PR106 r2 operating
point (2.71× pose-marginal).

Council notes:
- Param count target: ~250K total (renderer ~200K + residual basis + decoder ~50K)
- sin activation per α convention (SIREN/NeRF style)
- Residual basis dim D in [64, 256]; coeffs per pair k in [8, 32]
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally)
- Bolt-on <= 350 LOC; substrate_engineering exception per L7

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module (score-aware loss is a separate module)
- No /tmp paths
- Reviewable in 30 seconds per L12 (each method <= 30 LOC)
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


@dataclass(frozen=True)
class HybridRendererResidualConfig:
    """Static design-time parameters for hybrid_renderer_residual (γ).

    Defaults are council-calibrated 2026-05-12 SKETCH to hit ~250K params
    so we stay within the Selfcomp empirical ceiling. The full-mode subagent
    will retune after α's first anchor is in hand.
    """

    latent_dim: int = 28
    """Per-pair renderer latent dimensionality z."""

    embed_dim: int = 40
    """Channels of the initial spatial-grid embedding (decoder input)."""

    initial_grid_h: int = 3
    """Initial spatial-grid height before upsample blocks."""

    initial_grid_w: int = 4
    """Initial spatial-grid width before upsample blocks."""

    decoder_channels: tuple[int, ...] = (40, 32, 24, 20, 16, 12, 8)
    """Per-block output channels BEFORE the final RGB heads."""

    sin_frequency: float = 30.0
    """NeRF-style sin activation frequency (decoder)."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for the contest 1200-frame video)."""

    output_height: int = _CONTEST_H
    """Final RGB output height."""

    output_width: int = _CONTEST_W
    """Final RGB output width."""

    num_upsample_blocks: int = 7
    """Number of PixelShuffle(2) blocks. 7 -> 3x4 -> 384x512 ratio."""

    residual_basis_dim: int = 128
    """Size of the learned residual basis dictionary D."""

    residual_basis_value_dim: int = 16
    """Per-basis-entry vector dimensionality (D_v)."""

    residual_coeffs_per_pair: int = 12
    """Sparsity k — number of (index, value) pairs per frame-pair."""

    residual_decoder_hidden: tuple[int, ...] = (32, 16)
    """Hidden widths of the tiny residual MLP head before the RGB delta."""


class _SinAct(nn.Module):
    """Sine activation (SIREN/NeRF-style)."""

    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    """One Conv -> sin -> PixelShuffle(2) decoder block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # PixelShuffle(2) needs 4x output channels in the conv before shuffle
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.shuffle(self.act(self.conv(x)))


class _ResidualBasisDecoder(nn.Module):
    """Decode k sparse (index, value) coefficients into a per-pair RGB delta.

    The basis is a learnable dictionary ``D[B x D_v]``. Per-pair, we gather
    k rows by index, scale each by its coefficient value, sum, and decode
    via a small MLP -> RGB residual at a coarse grid that is bilinearly
    upsampled to the contest resolution.
    """

    def __init__(self, cfg: HybridRendererResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.basis = nn.Parameter(
            torch.empty(cfg.residual_basis_dim, cfg.residual_basis_value_dim).normal_(std=0.02)
        )

        # Small coarse-grid generator: D_v -> (small_h*small_w*small_ch)
        # The coarse grid is upsampled bilinearly to the contest H,W; this
        # keeps the residual decoder cheap (~50K params end-to-end).
        self._coarse_h = 6
        self._coarse_w = 8
        self._coarse_ch = 6  # 2 frames x 3 channels (small)
        coarse_total = self._coarse_h * self._coarse_w * self._coarse_ch

        layers: list[nn.Module] = []
        prev = cfg.residual_basis_value_dim
        for h in cfg.residual_decoder_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, coarse_total))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode per-pair residual deltas for frame_0 and frame_1.

        Args:
            indices: ``(B, k)`` long tensor of basis indices.
            values:  ``(B, k)`` float tensor of coefficients (continuous;
                quantized at archive-time).
            out_h, out_w: output spatial resolution.

        Returns:
            ``(res_0, res_1)`` each ``(B, 3, out_h, out_w)``.
        """
        if indices.shape != values.shape:
            raise ValueError(
                f"indices {tuple(indices.shape)} != values {tuple(values.shape)}"
            )
        if indices.dtype != torch.long:
            raise ValueError("indices must be torch.long")

        # gather basis rows: (B, k, D_v)
        gathered = self.basis[indices]
        # weight by values, then sum over k -> (B, D_v)
        combo = (gathered * values.unsqueeze(-1)).sum(dim=1)
        # decode coarse grid -> (B, 2*3, coarse_h, coarse_w)
        flat = self.head(combo)
        coarse = flat.view(-1, self._coarse_ch, self._coarse_h, self._coarse_w)
        # bilinear upsample to target
        up = F.interpolate(coarse, size=(out_h, out_w), mode="bilinear", align_corners=False)
        # split into frame_0 and frame_1 deltas
        res_0 = up[:, :3]
        res_1 = up[:, 3:]
        return res_0, res_1


class HybridRendererResidualSubstrate(nn.Module):
    """γ Hybrid renderer + residual basis substrate.

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The score-aware loss (separate module) consumes the rendered frames +
    the L1 norm of residual coefficients (sparsity term), runs frames
    through the differentiable eval-roundtrip (per CLAUDE.md eval_roundtrip
    non-negotiable), and backprops through SegNet/PoseNet on the SUM.
    """

    def __init__(self, cfg: HybridRendererResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair learned renderer latents
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # Per-pair sparse residual coefficients (continuous during training;
        # quantized + sparsified for archive). At SKETCH-level we keep the
        # FULL (num_pairs, residual_basis_dim) coefficient matrix and let the
        # follow-up subagent wire OMP / top-k to get the actual sparsity. At
        # inflate time, ONLY (num_pairs, residual_coeffs_per_pair) are stored.
        self.residual_coeff_full = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.residual_basis_dim).zero_()
        )

        # Renderer decoder: latent -> initial spatial grid -> up-blocks -> RGB heads
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim] + list(cfg.decoder_channels)
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at least "
                f"num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            blocks.append(_UpBlock(in_ch, out_ch, cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        # Residual basis decoder (the γ-specific contribution)
        self.residual_decoder = _ResidualBasisDecoder(cfg)

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN-style init on convs/linears with sin activation.

        The residual basis + residual head use plain Xavier (GELU
        activation; not sin-driven downstream).
        """
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
                    if m is self.latent_embed:
                        bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    else:
                        bound = math.sqrt(6.0 / fan_in)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def _topk_residual(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k sparsification of the per-pair residual coefficients.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.
        Returns:
            ``(indices, values)`` each ``(B, k)``; indices are long, values float.
        """
        coeffs = self.residual_coeff_full[pair_indices]  # (B, residual_basis_dim)
        k = self.cfg.residual_coeffs_per_pair
        # Top-k by absolute value (the typical sparse-coding selection)
        topk = torch.topk(coeffs.abs(), k=k, dim=1)
        indices = topk.indices
        # Gather signed values at the top-k positions
        values = coeffs.gather(1, indices)
        return indices, values

    def forward(
        self,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render frame-pairs at the given pair indices.

        Args:
            pair_indices: ``(B,)`` long tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1, residual_l1)`` where ``residual_l1`` is a scalar
            ``L_1`` norm of the active residual coefficients (the sparsity
            term consumed by the score-aware loss).
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z = self.latents[pair_indices]
        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )
        for block in self.blocks:
            h = block(h)
        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )
        rgb_base_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_base_1 = torch.sigmoid(self.head_rgb_1(h))

        # Residual basis decode (top-k sparse)
        res_idx, res_val = self._topk_residual(pair_indices)
        res_0, res_1 = self.residual_decoder(
            res_idx, res_val, self.cfg.output_height, self.cfg.output_width
        )

        rgb_0 = (rgb_base_0 + res_0).clamp(0.0, 1.0)
        rgb_1 = (rgb_base_1 + res_1).clamp(0.0, 1.0)

        # L1 norm of the ACTIVE coefficients (sparsity Lagrangian term)
        residual_l1 = res_val.abs().mean()

        return rgb_0, rgb_1, residual_l1

    def num_parameters(self) -> int:
        """Total trainable parameter count (council target ~250K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
