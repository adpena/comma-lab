# SPDX-License-Identifier: MIT
"""pact_nerv_cross_codec_a architecture — fec6 base + Pact-NeRV side-info (L0 SKETCH).

CROSS-CODEC composition variant per PACT-NERV-ULTIMATE Variant #16.

The base codec (fec6 Huffman k=16 selector + frame-exploit) is represented
at L0 SCAFFOLD by ``Fec6BaseCodecPlaceholder`` — a deterministic byte-level
proxy that emits a per-pair (RGB) baseline render from a fixed-Huffman
selector vocabulary. The L1 full path will swap this for the actual fec6
runtime per ``submissions/pr101_*`` archive grammar.

The Pact-NeRV side-info bolt-on uses the canonical HNeRV-class sister
architecture (depthwise-separable + SIREN + PixelShuffle decoder per
pact_nerv_selector_v3 sister) to emit a per-pair RESIDUAL on top of the
base render.

Cross-codec composition at inflate time:
    rgb_0[pair] = clamp(base_render_0[pair] + alpha * pact_nerv_residual_0[pair], 0, 1)
    rgb_1[pair] = clamp(base_render_1[pair] + alpha * pact_nerv_residual_1[pair], 0, 1)

where alpha is a CARGO-CULTED static composition coefficient at L0 (L1
learned composition gate per Atick-Redlich 1990 cooperative-receiver).
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
class PactNervCrossCodecAConfig:
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
    fec6_palette_size: int = 16
    composition_alpha: float = 0.1
    """Cross-codec composition coefficient (residual additive form).

    HARD-EARNED-DEFAULT: small alpha keeps Pact-NeRV side-info as a residual
    correction on top of the fec6 base render. L1 sweep: learned composition
    gate per Atick-Redlich 1990 cooperative-receiver.
    """


class Fec6BaseCodecPlaceholder(nn.Module):
    """L0 SCAFFOLD placeholder for fec6 base codec render.

    Per HNeRV parity L4 (≤200 LOC inflate budget) + L9 runtime closure, the
    L1 path will replace this with the actual fec6 runtime invocation from
    `submissions/pr101_*/inflate.py`. At L0 the placeholder emits a
    deterministic, per-pair, position-encoded baseline render so the
    cross-codec composition contract is testable without the full fec6
    runtime dependency.
    """

    def __init__(self, palette_size: int, num_pairs: int, output_h: int, output_w: int) -> None:
        super().__init__()
        if palette_size < 2:
            raise ValueError(f"palette_size must be >= 2; got {palette_size}")
        self.palette_size = palette_size
        self.num_pairs = num_pairs
        self.output_h = output_h
        self.output_w = output_w
        # Per-mode color palette (FEC6 k=16 fixed colors per mode index)
        self.register_buffer(
            "_palette",
            torch.linspace(0.0, 1.0, palette_size).unsqueeze(-1).repeat(1, 3),
        )

    def render(self, selectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Render base RGB pairs from per-pair selector indices.

        Args:
            selectors: (B,) per-pair selector index in [0, palette_size).

        Returns:
            (rgb_0, rgb_1) each shape (B, 3, H, W) in [0, 1].
        """
        if selectors.dtype != torch.long:
            raise ValueError("selectors must be torch.long")
        if (
            selectors.min().item() < 0
            or selectors.max().item() >= self.palette_size
        ):
            raise ValueError(
                f"selectors out of palette [0, {self.palette_size})"
            )
        B = selectors.shape[0]
        color = self._palette[selectors]  # (B, 3)
        # Deterministic per-pair gradient flat-color field
        rgb_0 = color.view(B, 3, 1, 1).expand(B, 3, self.output_h, self.output_w)
        # rgb_1: same color (frame-warp symmetry placeholder at L0)
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


class _DsUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class PactNervCrossCodecASubstrate(nn.Module):
    """Pact-NeRV-CROSS-CODEC-A renderer (L0 SKETCH; fec6 base + side-info bolt-on)."""

    def __init__(self, cfg: PactNervCrossCodecAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Pact-NeRV side-info bolt-on (residual generator)
        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
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
        # Side-info heads emit RESIDUAL in [-1, 1] via tanh
        self.head_res_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_res_1 = nn.Conv2d(final_ch, 3, kernel_size=1)
        # Per-pair selectors for the fec6 base codec
        self.register_buffer("selectors", torch.zeros(cfg.num_pairs, dtype=torch.long))
        # Base codec placeholder (L0; L1 replaces with actual fec6 runtime)
        self.base_codec = Fec6BaseCodecPlaceholder(
            cfg.fec6_palette_size, cfg.num_pairs, cfg.output_height, cfg.output_width,
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

    def forward(self, pair_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-codec composition: base render + alpha * side-info residual."""
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if (
            pair_indices.min().item() < 0
            or pair_indices.max().item() >= self.cfg.num_pairs
        ):
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )
        # Side-info residual (Pact-NeRV decoder)
        z = self.latents[pair_indices]
        h = self.latent_embed(z)
        h = h.view(-1, self.cfg.embed_dim, self.cfg.initial_grid_h, self.cfg.initial_grid_w)
        for block in self.blocks:
            h = block(h)
        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h, size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear", align_corners=False,
            )
        res_0 = torch.tanh(self.head_res_0(h))
        res_1 = torch.tanh(self.head_res_1(h))
        # Base codec render (fec6 placeholder at L0)
        selectors = self.selectors[pair_indices]
        base_0, base_1 = self.base_codec.render(selectors)
        # Cross-codec composition: base + alpha * residual, clamped
        alpha = self.cfg.composition_alpha
        rgb_0 = torch.clamp(base_0 + alpha * res_0, 0.0, 1.0)
        rgb_1 = torch.clamp(base_1 + alpha * res_1, 0.0, 1.0)
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
