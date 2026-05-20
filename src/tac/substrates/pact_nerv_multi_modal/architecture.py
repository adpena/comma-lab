# SPDX-License-Identifier: MIT
"""pact_nerv_multi_modal architecture — Pact-NeRV-MultiModal (L0 SKETCH).

HNeRV-class implicit renderer with 3-tower conditioning fusion: ego-pose +
SegNet-class-prior + odometry. The fusion vector conditions the decoder via
per-block channel-bias projection (sister of pact_nerv_distilled_scorer's
surrogate-feature conditioning pattern).

CLAUDE.md compliance: no MPS fallback, no inflate-time scorer load, no /tmp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_PAIRS = 600


@dataclass(frozen=True)
class PactNervMultiModalConfig:
    """Static design-time parameters for Pact-NeRV-MultiModal."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    pose_dim: int = 6
    """Ego-pose dimensionality (contest canonical: 6)."""
    class_prior_dim: int = 5
    """SegNet class prior dimensionality (upstream 5 classes)."""
    odometry_dim: int = 4
    """Odometry/IMU dimensionality at L0 (CARGO-CULTED; alternative: 9-DoF IMU)."""
    fusion_dim: int = 16
    """Fused conditioning vector dimensionality."""
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
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class MultiModalConditioningFusion(nn.Module):
    """3-tower fusion: pose + class_prior + odometry -> fusion_dim.

    Per Baltrušaitis 2019 multimodal-fusion taxonomy: concatenated late-fusion
    is the simplest baseline (CARGO-CULTED at L0; Stage 1 ablation against
    cross-attention + gated fusion).
    """

    def __init__(
        self,
        *,
        pose_dim: int = 6,
        class_prior_dim: int = 5,
        odometry_dim: int = 4,
        fusion_dim: int = 16,
    ) -> None:
        super().__init__()
        if pose_dim <= 0 or class_prior_dim <= 0 or odometry_dim <= 0:
            raise ValueError("all input dims must be positive")
        if fusion_dim <= 0:
            raise ValueError(f"fusion_dim must be positive; got {fusion_dim}")
        self.pose_dim = pose_dim
        self.class_prior_dim = class_prior_dim
        self.odometry_dim = odometry_dim
        self.fusion_dim = fusion_dim

        # Per-tower projection (lightweight; matches L0 LOC budget).
        self.pose_proj = nn.Linear(pose_dim, fusion_dim)
        self.class_proj = nn.Linear(class_prior_dim, fusion_dim)
        self.odo_proj = nn.Linear(odometry_dim, fusion_dim)
        # Concat-fusion: (B, 3*fusion_dim) -> (B, fusion_dim)
        self.fusion = nn.Linear(3 * fusion_dim, fusion_dim)

    def forward(
        self,
        pose: torch.Tensor,
        class_prior: torch.Tensor,
        odometry: torch.Tensor,
    ) -> torch.Tensor:
        if pose.dim() != 2 or pose.shape[1] != self.pose_dim:
            raise ValueError(
                f"pose must be (B, {self.pose_dim}); got shape {tuple(pose.shape)}"
            )
        if class_prior.dim() != 2 or class_prior.shape[1] != self.class_prior_dim:
            raise ValueError(
                f"class_prior must be (B, {self.class_prior_dim}); got shape {tuple(class_prior.shape)}"
            )
        if odometry.dim() != 2 or odometry.shape[1] != self.odometry_dim:
            raise ValueError(
                f"odometry must be (B, {self.odometry_dim}); got shape {tuple(odometry.shape)}"
            )

        h_p = F.relu(self.pose_proj(pose))
        h_c = F.relu(self.class_proj(class_prior))
        h_o = F.relu(self.odo_proj(odometry))
        h_cat = torch.cat([h_p, h_c, h_o], dim=1)
        return self.fusion(h_cat)


class PactNervMultiModalSubstrate(nn.Module):
    """Pact-NeRV-MultiModal renderer (L0 SKETCH).

    Forward: per-pair (latent + pose + class_prior + odometry) -> fusion vector
    -> per-block channel-bias conditioning of HNeRV decoder.

    Per HNeRV parity L5: outputs RGB at contest camera resolution.
    """

    def __init__(self, cfg: PactNervMultiModalConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.pose_data = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.pose_dim).normal_(std=0.02)
        )
        self.class_prior_data = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.class_prior_dim).normal_(std=0.02)
        )
        self.odometry_data = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.odometry_dim).normal_(std=0.02)
        )

        self.fusion = MultiModalConditioningFusion(
            pose_dim=cfg.pose_dim,
            class_prior_dim=cfg.class_prior_dim,
            odometry_dim=cfg.odometry_dim,
            fusion_dim=cfg.fusion_dim,
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim, cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        bias_projs: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
            bias_projs.append(nn.Linear(cfg.fusion_dim, channels[i + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.bias_projs = nn.ModuleList(bias_projs)

        final_ch = channels[cfg.num_upsample_blocks]
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
        pose = self.pose_data[pair_indices]
        class_prior = self.class_prior_data[pair_indices]
        odometry = self.odometry_data[pair_indices]
        fusion = self.fusion(pose, class_prior, odometry)

        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )

        for block, bias_proj in zip(self.blocks, self.bias_projs):
            h = block(h)
            bias = bias_proj(fusion)
            h = h + bias.view(-1, bias.shape[1], 1, 1)

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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
