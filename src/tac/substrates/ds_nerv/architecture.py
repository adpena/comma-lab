"""ds_nerv architecture — Depth-separable NeRV (L0 SKETCH).

Per-frame implicit renderer using depthwise-separable convolutions
throughout. The depthwise + 1x1 pointwise split factorizes a standard
KxK conv `O(C_in * C_out * K^2)` into `O(C_in * K^2 + C_in * C_out)`, a
~40% param reduction at K=3 / C=32. Quantizr-paradigm influence (PR101's
88K-param FiLM-conditioned depthwise-separable CNN).

Architecture (council-approved SKETCH 2026-05-12):

    Per-pair latent z in R^28
       |
       v
    Linear 28 -> 384                              # latent embed
       |
       v
    Reshape (1, 384, 3, 4)                        # initial spatial grid
       |
       v
    Block 0..6: DepthSepConv -> sin -> PixelShuffle(2)
       |     (depthwise 3x3 + pointwise 1x1; channels follow PR101 taper)
       v
    Head rgb_0 / rgb_1: 1x1 Conv 3 channels each

DepthSepConv unit (the substrate's distinctive primitive):
    1. Depthwise Conv2d(C_in, C_in, 3, groups=C_in)   -- spatial only
    2. Pointwise Conv2d(C_in, C_out, 1)                -- channel mix
    3. SIREN init on BOTH layers

~150K params target. SIREN initialization throughout.

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
class DsnervConfig:
    """Static design-time parameters for ds_nerv."""

    latent_dim: int = 28
    """Per-pair latent dimensionality."""

    embed_dim: int = 64
    """Channels of the initial spatial-grid embedding."""

    initial_grid_h: int = 3
    """Initial spatial-grid height."""

    initial_grid_w: int = 4

    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    """Per-block output channels for the depth-separable decoder."""

    sin_frequency: float = 30.0

    num_upsample_blocks: int = 7
    """Number of PixelShuffle(2) blocks; 7 -> 3x4 -> 384x512."""

    num_pairs: int = _PAIRS

    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _DepthSepConv(nn.Module):
    """Depthwise-3x3 + pointwise-1x1, SIREN-friendly."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    """DepthSep -> sin -> PixelShuffle(2)."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        # PixelShuffle(2) needs 4x channels in the pre-shuffle conv
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class DsnervSubstrate(nn.Module):
    """Depth-separable NeRV renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].
    """

    def __init__(self, cfg: DsnervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

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
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        # 1x1 RGB heads (depth-separable spirit: minimal head)
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
                        # Depthwise: fan_in is effectively kernel_size^2
                        fan_in = m.kernel_size[0] * m.kernel_size[1]
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

        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count (target ~150K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
