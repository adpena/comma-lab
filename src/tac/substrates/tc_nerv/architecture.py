# SPDX-License-Identifier: MIT
"""tc_nerv architecture — Temporal-Consistency NeRV substrate (L0 SKETCH).

Sibling of ``sane_hnerv``. Same SIREN + PixelShuffle + per-pair latent
backbone, but with smaller decoder channels to hit ~200K parameter target
(vs ~229K for sane_hnerv); the temporal-consistency contribution comes
purely from the score-aware loss (see ``score_aware_loss.py``) rather than
an architectural change to the renderer itself.

Architecture (L0 SKETCH, council pre-design only; concrete shapes mirror
``sane_hnerv`` for review-velocity reasons):

    Per-pair latent z in R^24                              (~14.4K latent params)
       |
       v
    Linear 24 -> 36 * 12                                   # initial embed
       |
       v
    Reshape (1, 36, 3, 4)                                  # 3x4 grid
       |
       v
    Block 0..6: Conv -> sin -> PixelShuffle(2)             # 384x512 final
       |
       v
    Head rgb_0 / rgb_1: Conv 3 channels each               # 2 frames per pair

Param-count budget:
    latents:        600 * 24 = 14_400
    latent_embed:   24 * (36 * 3 * 4) = 10_368 + bias 432 ~ 10_800
    blocks (7):     ~ 95_000 (smaller channel ladder vs sane_hnerv)
    heads (2):      ~ 1_000
    ── total ~ 121_000

    (Calibration during L1-promotion phase may bump channel widths up to
    hit the council-target ~200K. The L0 SKETCH commits to the architectural
    shape; param-count exact-match is a L1 calibration concern.)

L0 SKETCH disclaimers per CLAUDE.md "Lane maturity registry":
- research_only=true
- DEFERRED-pending-alpha-anchor
- score_claim/promotion_eligible/ready_for_exact_eval_dispatch all False

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
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


@dataclass(frozen=True)
class TCNervConfig:
    """Static design-time parameters for tc_nerv (L0 SKETCH).

    Mirrors ``SaneHnervConfig`` but with a smaller default channel ladder
    to land in the ~200K parameter band rather than ~229K.
    """

    latent_dim: int = 24
    """Per-pair latent dimensionality (slightly tighter than sane_hnerv's 28)."""

    embed_dim: int = 36
    """Channels of the initial spatial-grid embedding (smaller than sane_hnerv)."""

    initial_grid_h: int = 3
    initial_grid_w: int = 4

    decoder_channels: tuple[int, ...] = (32, 28, 22, 18, 14, 10, 8)
    """Per-block output channels BEFORE the final RGB heads.

    Calibrated 2026-05-12 (L0 SKETCH) to fall under sane_hnerv's ~229K target.
    L1 promotion may re-tune.
    """

    sin_frequency: float = 30.0
    """SIREN/NeRF default."""

    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W
    num_upsample_blocks: int = 7


class _SinAct(nn.Module):
    """Sine activation (SIREN/NeRF-style)."""

    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sin(self.w * x)


class _UpBlock(nn.Module):
    """One Conv -> sin -> PixelShuffle(2) block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2
        )
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.shuffle(self.act(self.conv(x)))


class TCNervSubstrate(nn.Module):
    """Temporal-Consistency NeRV renderer (L0 SKETCH).

    Forward signature matches ``SaneHnervSubstrate.forward``: takes a
    ``(B,)`` long tensor of pair indices and returns ``(rgb_0, rgb_1)`` each
    ``(B, 3, H, W)`` in ``[0, 1]``.

    The temporal-consistency regularizer is in ``score_aware_loss.py``;
    this module is just the renderer.
    """

    def __init__(self, cfg: TCNervConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim] + list(cfg.decoder_channels)
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have "
                f"at least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_UpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=3, padding=1)

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN initialization."""
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
        """Render frame-pairs at the given pair indices."""
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
        """Total trainable parameter count (L0 target ~200K)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
