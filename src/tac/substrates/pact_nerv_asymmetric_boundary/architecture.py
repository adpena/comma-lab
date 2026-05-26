# SPDX-License-Identifier: MIT
"""pact_nerv_asymmetric_boundary architecture - asymmetric per-class boundary FiLM.

Sister of NSCS06 v7 44% improvement per-class chroma anchors (105.15 → 58.89  # DOCSTRING_PERCENT_CLAIM_OK:canonical_nscs06_v7_empirical_44pct_anchor_105_15_to_58_89_contest_CUDA_artifact_at_omx_research_nscs06_path_a_chroma_optical_flow_redesign_20260516_md
contest-CUDA via cargo-cult-unwind methodology). The distinguishing primitive:
asymmetric per-pair-per-SegNet-class boundary signal feeding FiLM γ+β
modulation at the final upsample block.
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
_NUM_SEGNET_CLASSES = 5


@dataclass(frozen=True)
class PactNervAsymmetricBoundaryConfig:
    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    num_segnet_classes: int = _NUM_SEGNET_CLASSES
    """Matches upstream SegNet 5 classes per canonical evaluator contract."""
    boundary_signal_dim: int = _NUM_SEGNET_CLASSES
    film_init_delta_std: float = 0.01
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


class AsymmetricBoundaryFilm(nn.Module):
    """Asymmetric per-class boundary FiLM (γ + β; sister NSCS06 v7 chroma anchors).

    Per the NSCS06 v6→v7 cargo-cult-unwind (44% improvement): per-class  # DOCSTRING_PERCENT_CLAIM_OK:canonical_nscs06_v6_to_v7_cargo_cult_unwind_methodology_empirical_44pct_anchor_artifact_at_omx_research_nscs06_path_a_chroma_optical_flow_redesign_20260516_md
    chroma anchors capture per-SegNet-class dispersion in the contest's
    response that per-pair-uniform conditioning misses. This module applies
    asymmetric FiLM γ+β modulation conditioned on a per-SegNet-class
    boundary signal (5 dims matching upstream SegNet 5 classes).

    γ_init = 1.0 + Δ; β_init = 0.0 + Δ (residual form for early-training stability).
    """

    def __init__(
        self,
        num_features: int,
        boundary_signal_dim: int = _NUM_SEGNET_CLASSES,
        init_delta_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive; got {num_features}")
        if boundary_signal_dim <= 0:
            raise ValueError(
                f"boundary_signal_dim must be positive; got {boundary_signal_dim}"
            )
        if init_delta_std < 0:
            raise ValueError(f"init_delta_std must be non-negative; got {init_delta_std}")
        self.num_features = num_features
        self.boundary_signal_dim = boundary_signal_dim
        # FiLM γ+β projections (asymmetric per-class boundary)
        self.gamma_proj = nn.Linear(boundary_signal_dim, num_features)
        self.beta_proj = nn.Linear(boundary_signal_dim, num_features)
        # γ_init = 1.0 + Δ; β_init = 0.0 + Δ (residual form)
        with torch.no_grad():
            self.gamma_proj.weight.normal_(mean=0.0, std=init_delta_std)
            self.gamma_proj.bias.zero_()
            self.beta_proj.weight.normal_(mean=0.0, std=init_delta_std)
            self.beta_proj.bias.zero_()

    def forward(self, x: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W); got {tuple(x.shape)}")
        if boundary.dim() != 2:
            raise ValueError(
                f"boundary must be (B, boundary_signal_dim); got {tuple(boundary.shape)}"
            )
        if boundary.shape[1] != self.boundary_signal_dim:
            raise ValueError(
                f"boundary dim {boundary.shape[1]} != {self.boundary_signal_dim}"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(f"x channels {x.shape[1]} != {self.num_features}")
        gamma = 1.0 + self.gamma_proj(boundary)
        beta = self.beta_proj(boundary)
        return x * gamma.view(-1, self.num_features, 1, 1) + beta.view(-1, self.num_features, 1, 1)


class PactNervAsymmetricBoundarySubstrate(nn.Module):
    """Pact-NeRV-AsymmetricBoundary renderer (L0 SKETCH; per-class FiLM)."""

    def __init__(self, cfg: PactNervAsymmetricBoundaryConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        # Per-pair-per-class boundary signal (5 SegNet classes)
        self.boundary_signals = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.boundary_signal_dim).normal_(std=0.02)
        )

        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )
        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)
        final_ch = channels[cfg.num_upsample_blocks]
        # Apply boundary FiLM at final upsample block only (asymmetric =
        # only-final-block; sister of NSCS06 v7 per-class chroma at output).
        self.boundary_film = AsymmetricBoundaryFilm(
            num_features=final_ch,
            boundary_signal_dim=cfg.boundary_signal_dim,
            init_delta_std=cfg.film_init_delta_std,
        )
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
                    # Skip boundary FiLM γ_proj + β_proj (already zero-init).
                    if m is self.boundary_film.gamma_proj or m is self.boundary_film.beta_proj:
                        continue
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self,
        pair_indices: torch.Tensor,
        boundary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")
        z = self.latents[pair_indices]
        b = self.boundary_signals[pair_indices] if boundary is None else boundary
        h = self.latent_embed(z)
        h = h.view(
            -1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w
        )
        for block in self.blocks:
            h = block(h)
        # Apply boundary FiLM after all upsample blocks (asymmetric design)
        h = self.boundary_film(h, b)
        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h, size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear", align_corners=False,
            )
        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_boundary_film_parameters(self) -> int:
        return sum(p.numel() for p in self.boundary_film.parameters() if p.requires_grad)
