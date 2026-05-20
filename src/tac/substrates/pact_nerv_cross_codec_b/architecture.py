# SPDX-License-Identifier: MIT
"""pact_nerv_cross_codec_b architecture — PR106 base + Pact-NeRV-IA3 side-info (L0 SKETCH).

CROSS-CODEC composition variant per PACT-NERV-ULTIMATE Variant #17.

The base codec (PR106 latent-score-table + format0d) is represented at L0
SCAFFOLD by ``Pr106BaseCodecPlaceholder`` — a deterministic byte-level
proxy that emits a per-pair (RGB) baseline render from a quantized
latent-score-table. The L1 full path will swap this for the actual PR106
runtime per ``submissions/pr106_*`` archive grammar.

The Pact-NeRV-IA3 side-info bolt-on uses the canonical HNeRV-class sister
architecture (depthwise-separable + SIREN + PixelShuffle decoder) with
IA3 γ-only ego-pose-conditioned per-block modulation per Liu 2022
arXiv:2205.05638 (sister pact_nerv_ia3 commit 9cf9bdb16).

Cross-codec composition at inflate time:
    rgb_0[pair] = clamp(pr106_render_0[pair] + alpha * ia3_residual_0[pair], 0, 1)
    rgb_1[pair] = clamp(pr106_render_1[pair] + alpha * ia3_residual_1[pair], 0, 1)
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
_POSE_DIM = 6  # contest canonical: upstream PoseNet first 6 dims


@dataclass(frozen=True)
class PactNervCrossCodecBConfig:
    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W
    pr106_score_table_size: int = 64
    """PR106 latent-score-table size (per CROSS-CANDIDATE finding #3 anchor)."""
    pose_dim: int = _POSE_DIM
    """IA3 γ-projection input dim (matches upstream PoseNet first 6 dims)."""
    ia3_init_delta_std: float = 0.01
    """IA3 γ_init = 1.0 + Δ where Δ ~ N(0, ia3_init_delta_std^2)."""
    composition_alpha: float = 0.1
    """Cross-codec composition coefficient (residual additive form)."""


class Pr106BaseCodecPlaceholder(nn.Module):
    """L0 SCAFFOLD placeholder for PR106 base codec render.

    Per HNeRV parity L4 + L9, the L1 path will replace this with the actual
    PR106 runtime invocation from `submissions/pr106_*/inflate.py`. At L0
    the placeholder emits a deterministic, per-pair render from a quantized
    latent-score-table so the cross-codec composition contract is testable.
    """

    def __init__(self, score_table_size: int, num_pairs: int, output_h: int, output_w: int) -> None:
        super().__init__()
        if score_table_size < 2:
            raise ValueError(f"score_table_size must be >= 2; got {score_table_size}")
        self.score_table_size = score_table_size
        self.num_pairs = num_pairs
        self.output_h = output_h
        self.output_w = output_w
        # Per-pair latent-score-table (quantized to score_table_size buckets)
        # Position-encoded color palette mirrors PR106 format0d per-bucket color
        self.register_buffer(
            "_score_table",
            torch.linspace(0.0, 1.0, score_table_size).unsqueeze(-1).repeat(1, 3),
        )

    def render(self, score_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render PR106-style base RGB from per-pair score-table indices.

        Args:
            score_indices: (B,) per-pair score-table index in [0, score_table_size).

        Returns:
            (rgb_0, rgb_1) each shape (B, 3, H, W) in [0, 1].
        """
        if score_indices.dtype != torch.long:
            raise ValueError("score_indices must be torch.long")
        if (
            score_indices.min().item() < 0
            or score_indices.max().item() >= self.score_table_size
        ):
            raise ValueError(
                f"score_indices out of table [0, {self.score_table_size})"
            )
        B = score_indices.shape[0]
        color = self._score_table[score_indices]  # (B, 3)
        rgb_0 = color.view(B, 3, 1, 1).expand(B, 3, self.output_h, self.output_w)
        rgb_1 = color.view(B, 3, 1, 1).expand(B, 3, self.output_h, self.output_w)
        return rgb_0.clone(), rgb_1.clone()


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


class _Ia3GammaBlock(nn.Module):
    """IA3 γ-only ego-pose-conditioned modulation per Liu 2022.

    γ = 1.0 + γ_proj(ego_pose); element-wise rescale on channel axis.
    """

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float, pose_dim: int,
                 ia3_init_delta_std: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)
        # IA3 γ-only projection from ego_pose; NO β bias
        self.gamma_proj = nn.Linear(pose_dim, out_ch, bias=True)
        with torch.no_grad():
            self.gamma_proj.weight.normal_(std=ia3_init_delta_std)
            self.gamma_proj.bias.zero_()

    def forward(self, x: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
        h = self.shuffle(self.act(self.dsc(x)))
        # γ = 1.0 + Δ (residual form per IA3 §3.2)
        gamma = 1.0 + self.gamma_proj(ego_pose)  # (B, out_ch)
        return h * gamma.view(h.shape[0], h.shape[1], 1, 1)


class PactNervCrossCodecBSubstrate(nn.Module):
    """Pact-NeRV-CROSS-CODEC-B renderer (L0 SKETCH; PR106 base + IA3 side-info)."""

    def __init__(self, cfg: PactNervCrossCodecBConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        # Per-pair ego-pose buffer (loaded at inflate time from archive meta)
        self.register_buffer(
            "ego_poses", torch.zeros(cfg.num_pairs, cfg.pose_dim)
        )
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )
        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        # IA3 γ-only modulated blocks (sister of pact_nerv_ia3)
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(
                _Ia3GammaBlock(
                    channels[i], channels[i + 1], cfg.sin_frequency,
                    cfg.pose_dim, cfg.ia3_init_delta_std,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        final_ch = channels[cfg.num_upsample_blocks]
        # Side-info heads emit RESIDUAL in [-1, 1] via tanh
        self.head_res_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_res_1 = nn.Conv2d(final_ch, 3, kernel_size=1)
        # Per-pair score indices for the PR106 base codec
        self.register_buffer("score_indices", torch.zeros(cfg.num_pairs, dtype=torch.long))
        # Base codec placeholder (L0; L1 replaces with actual PR106 runtime)
        self.base_codec = Pr106BaseCodecPlaceholder(
            cfg.pr106_score_table_size, cfg.num_pairs,
            cfg.output_height, cfg.output_width,
        )
        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, _Ia3GammaBlock):
                    # Skip IA3 gamma_proj; preserve its IA3-spec zero-init
                    continue
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
                        fan_in = m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear) and m is not None:
                    # Skip gamma_proj linears (already IA3-init'd inside _Ia3GammaBlock)
                    is_gamma = any(
                        m is blk.gamma_proj for blk in self.blocks
                    )
                    if is_gamma:
                        continue
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-codec composition: PR106 base render + alpha * IA3 side-info residual."""
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if (
            pair_indices.min().item() < 0
            or pair_indices.max().item() >= self.cfg.num_pairs
        ):
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )
        # IA3 side-info residual
        z = self.latents[pair_indices]
        ego_pose = self.ego_poses[pair_indices]
        h = self.latent_embed(z)
        h = h.view(-1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w)
        for block in self.blocks:
            h = block(h, ego_pose)
        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h, size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear", align_corners=False,
            )
        res_0 = torch.tanh(self.head_res_0(h))
        res_1 = torch.tanh(self.head_res_1(h))
        # PR106 base codec render (placeholder at L0)
        score_idx = self.score_indices[pair_indices]
        base_0, base_1 = self.base_codec.render(score_idx)
        # Cross-codec composition
        alpha = self.cfg.composition_alpha
        rgb_0 = torch.clamp(base_0 + alpha * res_0, 0.0, 1.0)
        rgb_1 = torch.clamp(base_1 + alpha * res_1, 0.0, 1.0)
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
