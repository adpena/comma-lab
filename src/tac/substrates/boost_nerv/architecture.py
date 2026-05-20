# SPDX-License-Identifier: MIT
"""boost_nerv architecture — BoostNeRV (L0 SKETCH).

Per-frame implicit renderer with a NUM_BOOSTING_ROUNDS=2 iterative
residual-refinement chain on top of a DepthSep base decoder. Operator 5-tier
fit-ranking HIGH FIT ⭐⭐⭐⭐⭐: the boosting paradigm is paradigm-orthogonal
to existing NeRV variants and composes with any base substrate at L1+.

Literature anchor: Liu et al. ECCV 2024 BoostNeRV (paper-ID literature
reference per BUILD task #1090). The boosting paradigm draws on the
general gradient-boosting tradition: each round fits the residual of the
prior rounds, progressively reducing the worst-case error.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^24
       |
       v
    Base decoder (DepthSep + SIREN + PixelShuffle, mirrors ds_nerv)
       |
       v
    rgb_base in [0, 1]
       |
       v
    Boosting round 0..NUM_ROUNDS-1:
        TinyConv(rgb_iter, z) -> residual (gain clamped to [-0.1, 0.1])
        rgb_iter = clamp(rgb_iter + residual, 0, 1)
       |
       v
    Head rgb_0 / rgb_1: 1x1 Conv on rgb_iter

The boosting heads are TINY (per-round 3-channel residual prediction with a
shared latent-conditioning embedding). They add ~5-15K params per round at
the L0 SKETCH config; the base decoder dominates the parameter budget.

CLAUDE.md compliance:
- No silent device defaults
- No scorer load
- No /tmp paths
- Reviewable in 30 seconds per L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2


@dataclass(frozen=True)
class BoostnervConfig:
    """Static design-time parameters for boost_nerv."""

    latent_dim: int = 24
    """Per-pair latent dimensionality (shared across base + boosting rounds)."""

    embed_dim: int = 64
    """Channels of the initial spatial-grid embedding."""

    initial_grid_h: int = 3
    """Initial spatial-grid height."""

    initial_grid_w: int = 4

    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    """Per-block output channels for the depth-separable base decoder."""

    sin_frequency: float = 30.0

    num_upsample_blocks: int = 7
    """Number of PixelShuffle(2) blocks; 7 -> 3x4 -> 384x512."""

    num_boosting_rounds: int = 2
    """Number of iterative residual-refinement rounds.

    CARGO-CULTED at L0 (per cargo-cult audit in __init__.py); sweep at L1.
    Increasing this beyond 4 likely yields diminishing returns and
    exacerbates the rate term.
    """

    boosting_gain_clamp: float = 0.1
    """Per-round residual gain clamp magnitude.

    CARGO-CULTED at L0; chosen to prevent runaway residuals during early
    training. Needs empirical per-substrate tuning at L1.
    """

    boosting_hidden_dim: int = 12
    """Tiny boosting-head hidden channel count (keep heads cheap)."""

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
    """Depthwise-3x3 + pointwise-1x1, SIREN-friendly (mirrors ds_nerv)."""

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
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class _BoostingHead(nn.Module):
    """Tiny per-round residual head: TinyConv(rgb_in, z_proj) -> delta in [-1, 1].

    Each head consumes (rgb_in, z_proj) where z_proj is a small projection of
    the per-pair latent to the spatial grid. Output is 3-channel residual
    that we clamp to [-gain, +gain] before adding to rgb_in.
    """

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        self.conv1 = nn.Conv2d(3 + hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, kernel_size=1)

    def forward(self, rgb_in: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), rgb_in: (B, 3, H, W)
        z_emb = self.z_proj(z)  # (B, hidden_dim)
        z_grid = z_emb.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, rgb_in.shape[-2], rgb_in.shape[-1]
        )
        h = torch.cat([rgb_in, z_grid], dim=1)
        h = F.relu(self.conv1(h))
        residual = torch.tanh(self.conv2(h))  # in [-1, 1]
        return residual


class BoostnervSubstrate(nn.Module):
    """BoostNeRV renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)``
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The forward path:
    1. Base decoder produces rgb_base from latents.
    2. NUM_BOOSTING_ROUNDS residual rounds refine rgb_base iteratively;
       each residual is clamped to [-gain, +gain] before addition.
    3. Final 1x1 conv heads produce rgb_0 / rgb_1.
    """

    def __init__(self, cfg: BoostnervConfig) -> None:
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
        # Base RGB heads (pre-boosting)
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        # Boosting residual heads (one shared across frame_0 + frame_1 per round
        # to keep the rate term tight; per-round, per-frame variant is a
        # CARGO-CULTED choice that would inflate the head count 2x).
        self.boosting_heads = nn.ModuleList(
            [
                _BoostingHead(cfg.boosting_hidden_dim, cfg.latent_dim)
                for _ in range(cfg.num_boosting_rounds)
            ]
        )

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
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

        rgb_0_base = torch.sigmoid(self.head_rgb_0(h))
        rgb_1_base = torch.sigmoid(self.head_rgb_1(h))

        # Iterative boosting rounds (shared heads across frame_0/frame_1).
        gain = self.cfg.boosting_gain_clamp
        rgb_0 = rgb_0_base
        rgb_1 = rgb_1_base
        for head in self.boosting_heads:
            residual_0 = torch.clamp(head(rgb_0, z), -gain, gain)
            residual_1 = torch.clamp(head(rgb_1, z), -gain, gain)
            rgb_0 = torch.clamp(rgb_0 + residual_0, 0.0, 1.0)
            rgb_1 = torch.clamp(rgb_1 + residual_1, 0.0, 1.0)

        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        """Total trainable parameter count (target ~170K including boosting)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
