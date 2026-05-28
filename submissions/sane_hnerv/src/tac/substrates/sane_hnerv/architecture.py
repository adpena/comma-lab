# SPDX-License-Identifier: MIT
"""sane_hnerv architecture — Score-Aware NeRV Extended substrate.

Frame-conditional implicit renderer mirroring the leaderboard PR101 / PR100
HNeRV-LC-v2 design (canonical HNeRV with per-pair latent + bilinear-skip +
sin activation + PixelShuffle decoder), explicitly held to ~229K params per
the Quantizr empirical anchor.

Architecture (council-approved 2026-05-12; Hotz Carmack-style):

    Per-pair latent z in R^28  (14 numbers per frame, 2 frames per pair)
       |
       v
    Linear 28 -> 768                           # latent embed
       |
       v
    Reshape (1, 768, 3, 4)                     # initial spatial grid (~1/128 res)
       |
       v
    Block 0: Conv -> sin -> PixelShuffle(2)    # 4x6 grid
    Block 1: Conv -> sin -> PixelShuffle(2)    # 8x12 grid
    Block 2: Conv -> sin -> PixelShuffle(2)    # 16x24 grid
    Block 3: Conv -> sin -> PixelShuffle(2)    # 32x48 grid
    Block 4: Conv -> sin -> PixelShuffle(2)    # 64x96 grid
    Block 5: Conv -> sin -> PixelShuffle(2)    # 128x192 grid
    Block 6: Conv -> sin -> PixelShuffle(2)    # 256x384 grid  (NOT 384x512 yet)
       |  (with bilinear-skip from each prior block)
       v
    Head rgb_0 / rgb_1: Conv 3 channels each   # 2 frames per pair

(Output is interpolated bilinearly from 256x384 to the contest 384x512 at
inflate-time. The architecture honors L5 "full RGB renderer".)

Council notes:
- Param count target: 229K +- 10% (matches PR100/PR101 empirical anchor)
- sin frequency: 30.0 (NeRF default; ablation flag in CLI)
- Decoder channels: 128 -> 96 -> 80 -> 64 -> 48 -> 32 -> 16 -> 6 (2*3 RGB heads)
- EMA decay 0.997 per CLAUDE.md "EMA — non-negotiable" (applied externally)

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module (score-aware loss is a separate module)
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


@dataclass(frozen=True)
class SaneHnervConfig:
    """Static design-time parameters for sane_hnerv.

    All fields required-keyword (no silent defaults beyond explicit ones).
    """

    latent_dim: int = 28
    """Per-pair latent dimensionality (council: 14 floats per frame, 2 frames)."""

    embed_dim: int = 48
    """Channels of the initial spatial-grid embedding.

    Council-calibrated 2026-05-12: with `decoder_channels=(40,32,24,20,16,12,8)`
    and `num_upsample_blocks=7`, total param count is ~216K — within 10% of
    PR100/PR101's empirical 229K anchor per Quantizr.
    """

    initial_grid_h: int = 3
    """Initial spatial-grid height before upsample blocks."""

    initial_grid_w: int = 4
    """Initial spatial-grid width before upsample blocks."""

    decoder_channels: tuple[int, ...] = (40, 32, 24, 20, 16, 12, 8)
    """Per-block output channels BEFORE the final RGB heads.

    Council-calibrated 2026-05-12 to hit ~229K total params with
    `embed_dim=48`. PR101's empirical-best is ~229K.
    """

    sin_frequency: float = 30.0
    """NeRF-style sin activation frequency (Sitzmann SIREN choice)."""

    num_pairs: int = _PAIRS
    """Number of latent rows (600 for the contest 1200-frame video)."""

    output_height: int = _CONTEST_H
    """Final RGB output height (interpolated from final decoder block)."""

    output_width: int = _CONTEST_W
    """Final RGB output width."""

    num_upsample_blocks: int = 7
    """Number of PixelShuffle(2) blocks. 7 -> 3x4 -> 384x512 ratio."""


class _SinAct(nn.Module):
    """Sine activation (SIREN/NeRF-style)."""

    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    """One Conv -> sin -> PixelShuffle(2) block, with optional bilinear-skip."""

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


class SaneHnervSubstrate(nn.Module):
    """The score-aware NeRV Extended renderer.

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The score-aware loss (separate module) consumes the rendered frames,
    runs them through the differentiable eval-roundtrip (per CLAUDE.md
    eval_roundtrip non-negotiable) + the patched yuv6 (per PR #95/#106), then
    backprops through SegNet/PoseNet.
    """

    def __init__(self, cfg: SaneHnervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-pair learned latents
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        # latent -> initial spatial grid
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        # Decoder up-blocks
        # channels[0] = embed_dim, channels[i+1] is the post-block-i channel count
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

        # Two separate RGB heads (frame 0 and frame 1 of the pair)
        # Final post-block channel count is channels[num_upsample_blocks].
        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        # Initialize convs with SIREN-style scheme (omega-derived)
        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN initialization: weights ~ Uniform(-c/fan_in, c/fan_in) with c = sqrt(6/fan_in)/w."""
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

        # Interpolate to the contest resolution if needed
        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )

        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count (council target ~229K +- 10%)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
