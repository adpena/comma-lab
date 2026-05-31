# SPDX-License-Identifier: MIT
"""Z4 Atick-Redlich cooperative-receiver substrate — PyTorch (inflate-time consumer).

Per the Atick-Redlich 1990 canonical retinal mutual-information theorem, the
optimal cooperative receiver decorrelates input statistics to maximize
``I(X; T)`` under a fixed coding budget B. This module realizes that primitive
as a PyTorch substrate consumed at inflate time (the MLX-native training pass
lives in :mod:`tac.substrates.time_traveler_l5_z4.mlx_renderer`; weights bridge
via :mod:`tac.substrates.time_traveler_l5_z4.archive_candidate`).

Z4 distinguishing primitive vs sister substrates per Catalog #272:

1. Per-pair learned latent ``z_i`` (dim ``latent_dim``, default 32)
2. Atick-Redlich 1990 spatial decorrelation filter ``W_AR`` (latent_dim x
   latent_dim) applied as ``z'_i = W_AR @ z_i`` before decoder forward.
   Initialized to identity (so untrained Z4 reduces to a vanilla latent-decoder
   baseline); the cooperative-receiver score-aware Lagrangian rotates it
   toward the decorrelating eigenbasis of the contest-scorer-conditional
   latent covariance.
3. Simple PixelShuffle decoder: latent_embed -> 5 upsample blocks
   (DepthwiseSeparable + sin + PixelShuffle(2)) -> 1x1 RGB heads -> sigmoid
   -> 255-domain output at (384, 512) camera resolution.

Per HNeRV parity L5: outputs full RGB at contest camera resolution (NOT a
mask codec). Per HNeRV parity L7: substrate-engineering tier (~50K params
target; ``lane_class=substrate_engineering`` declared in lane registry).

[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/architecture.py canonical PyTorch substrate pattern]
[verified-against: src/tac/substrates/_shared/score_aware_common.py Catalog #164 scorer routing]
[verified-against: Atick & Redlich 1990 "Towards a Theory of Early Visual Processing" decorrelation theorem]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class Z4AtickRedlichConfig:
    """Z4 Atick-Redlich cooperative-receiver substrate configuration.

    Defaults are chosen to mirror the Z6-v2 / Z7-Mamba-2 surface (384 x 512
    output resolution; 600 default num_pairs) while keeping Z4-specific
    parameters (latent_dim=32 + decorrelator + 5 PixelShuffle blocks) per
    Catalog #272 distinguishing-feature contract.
    """

    num_pairs: int = 600
    latent_dim: int = 32
    embed_dim: int = 48
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = field(
        default_factory=lambda: (48, 32, 24, 16, 16, 12)
    )
    num_upsample_blocks: int = 5
    sin_frequency: float = 30.0
    output_height: int = 384
    output_width: int = 512
    apply_decorrelator: bool = True
    """When False, ``W_AR`` is set to identity and never trained (ablation)."""
    cooperative_receiver_beta: float = 0.5
    """IB tradeoff weight per Tishby-Zaslavsky 2015 (carries to score-aware loss)."""


class _SinAct(nn.Module):
    """sin(w * x) activation (PyTorch sister of MLX ``_SinActMLX``)."""

    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _DepthSepConv(nn.Module):
    """Depth-separable convolution (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _UpsampleBlock(nn.Module):
    """DepthSep -> sin -> PixelShuffle(2) — Z4's UNIQUE per-block architecture.

    Unlike Z6-v2's FiLM-conditioned blocks, Z4 uses pure feed-forward
    DepthSeparable + sin convolutions WITHOUT FiLM modulation. The cooperative-
    receiver primitive lives entirely in the Atick-Redlich decorrelator at
    the latent surface (NOT distributed across decoder layers like Z6's FiLM).
    """

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dsc(x)
        h = self.act(h)
        return self.pixel_shuffle(h)


class _AtickRedlichDecorrelator(nn.Module):
    """Atick-Redlich 1990 spatial decorrelation filter on the per-pair latent.

    Implements ``z'_i = W_AR @ z_i + b_AR`` as a single learned Linear
    transformation (latent_dim -> latent_dim). Initialized to identity so
    untrained Z4 reduces to vanilla latent-decoder; the cooperative-receiver
    score-aware Lagrangian rotates ``W_AR`` toward the decorrelating
    eigenbasis of the contest-scorer-conditional latent covariance.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim)
        # Identity init (Atick-Redlich canonical: untrained = no decorrelation).
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(latent_dim))
            self.proj.bias.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class Z4AtickRedlichSubstrate(nn.Module):
    """Z4 Atick-Redlich cooperative-receiver substrate (PyTorch; inflate-time).

    Forward:

    1. Per-pair latent gather (Embedding[num_pairs, latent_dim]).
    2. Atick-Redlich decorrelation: ``z' = W_AR @ z + b_AR`` (identity at init).
    3. latent_embed: ``z' -> embed_dim * initial_grid_h * initial_grid_w``,
       reshape to (B, embed_dim, initial_grid_h, initial_grid_w).
    4. 5 ``_UpsampleBlock``s (DepthSep + sin + PixelShuffle(2)) → ~3x4 -> ~96x128
       (after 5 blocks at 2x each), then bilinear upsample to (384, 512).
    5. Two 1x1 RGB heads → ``sigmoid * 255`` → ``(B, 3, 384, 512)`` pair.

    Output: ``(rgb_0, rgb_1)`` each ``(B, 3, 384, 512)`` in ``[0, 255]``.
    """

    def __init__(self, cfg: Z4AtickRedlichConfig) -> None:
        super().__init__()
        self.cfg = cfg
        num_pairs = int(cfg.num_pairs)
        latent_dim = int(cfg.latent_dim)
        embed_dim = int(cfg.embed_dim)
        initial_h = int(cfg.initial_grid_h)
        initial_w = int(cfg.initial_grid_w)

        # Per-pair learnable latent (registered as a Parameter so .latents
        # appears in state_dict for the canonical archive bridge).
        self.latents = nn.Parameter(torch.randn(num_pairs, latent_dim) * 0.02)

        # Atick-Redlich 1990 spatial decorrelation filter (identity at init).
        self.decorrelator = _AtickRedlichDecorrelator(latent_dim)

        # Latent -> initial spatial grid.
        self.latent_embed = nn.Linear(
            latent_dim, embed_dim * initial_h * initial_w
        )

        # PixelShuffle upsample stack.
        channels = [embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        self.blocks = nn.ModuleList(
            [
                _UpsampleBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    sin_freq=cfg.sin_frequency,
                )
                for i in range(int(cfg.num_upsample_blocks))
            ]
        )

        final_ch = channels[int(cfg.num_upsample_blocks)]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN-style init (matches Z6-v2 sister canonical pattern)."""
        w = float(self.cfg.sin_frequency)

        def _siren_bound(fan_in: int) -> float:
            return math.sqrt(6.0 / max(int(fan_in), 1)) / max(w, 1.0)

        bound_e = _siren_bound(int(self.cfg.latent_dim))
        with torch.no_grad():
            nn.init.uniform_(self.latent_embed.weight, -bound_e, bound_e)
            self.latent_embed.bias.zero_()
            for block in self.blocks:
                d = block.dsc.depthwise
                fan_in_d = d.weight.shape[2] * d.weight.shape[3]
                bound_d = _siren_bound(fan_in_d)
                nn.init.uniform_(d.weight, -bound_d, bound_d)
                d.bias.zero_()
                p = block.dsc.pointwise
                fan_in_p = int(p.weight.shape[1])
                bound_p = _siren_bound(fan_in_p)
                nn.init.uniform_(p.weight, -bound_p, bound_p)
                p.bias.zero_()
            for head in (self.head_rgb_0, self.head_rgb_1):
                fan_in_h = int(head.weight.shape[1])
                bound_h = _siren_bound(fan_in_h)
                nn.init.uniform_(head.weight, -bound_h, bound_h)
                head.bias.zero_()

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning ``(rgb_0, rgb_1)`` each ``(B, 3, 384, 512)`` in [0, 255]."""
        z = torch.index_select(self.latents, 0, pair_indices)
        if self.cfg.apply_decorrelator:
            z = self.decorrelator(z)

        h = self.latent_embed(z)
        h = h.view(
            -1,
            int(self.cfg.embed_dim),
            int(self.cfg.initial_grid_h),
            int(self.cfg.initial_grid_w),
        )
        for block in self.blocks:
            h = block(h)
        # Final bilinear resize to camera resolution (384, 512).
        h = nn.functional.interpolate(
            h,
            size=(int(self.cfg.output_height), int(self.cfg.output_width)),
            mode="bilinear",
            align_corners=False,
        )
        rgb_0 = torch.sigmoid(self.head_rgb_0(h)) * 255.0
        rgb_1 = torch.sigmoid(self.head_rgb_1(h)) * 255.0
        return rgb_0, rgb_1

    def reconstruct_pair(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pair forward returning ``(rgb_0, rgb_1)`` in [0, 1] (NCHW).

        The MLX harness ``reconstruct_pair_nchw01`` convention requires the
        ``[0, 1]`` domain (the harness multiplies by 255 internally for the
        canonical Hinton-distilled SegNet teacher).
        """
        rgb_0_255, rgb_1_255 = self.forward(pair_indices)
        return rgb_0_255 / 255.0, rgb_1_255 / 255.0

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


__all__ = [
    "Z4AtickRedlichConfig",
    "Z4AtickRedlichSubstrate",
]
