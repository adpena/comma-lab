# SPDX-License-Identifier: MIT
"""pact_nerv_selector_v3 architecture - Rice-Golomb selector coder (L0 SKETCH)."""

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
class PactNervSelectorV3Config:
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
    selector_palette_size: int = 16

    rice_golomb_k: int = 2
    """Rice-Golomb parameter k; HARD-EARNED-DEFAULT for geometric-decay
    distributions with mean ~ 2**k. Sweep at L1 dispatch."""


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


class RiceGolombSelectorCoder:
    """Rice-Golomb selector coder for FEC6 k=16 palette (Golomb 1966 + Rice 1971).

    Symbol n encoded as:
      - unary prefix: q = n >> k zeros + 1 terminator
      - binary suffix: low k bits of n
    Total bit-length per symbol: q + 1 + k bits.

    For geometric-decay distributions p(n) = (1-p) * p**n with mean
    M ~ p/(1-p), Rice-Golomb with k = max(0, floor(log2(M))) achieves
    near-optimal entropy.

    Static k at L0 = CARGO-CULTED; adaptive-k per stream is the L1 sweep.
    """

    def __init__(self, palette_size: int, k: int = 2) -> None:
        if palette_size < 2:
            raise ValueError(f"palette_size must be >= 2; got {palette_size}")
        if k < 0 or k > 8:
            raise ValueError(f"k must be in [0, 8]; got {k}")
        self.palette_size = palette_size
        self.k = k

    def encode(self, symbols: list[int]) -> bytes:
        """Encode a symbol stream into the Rice-Golomb bitstream."""
        if not symbols:
            return b""
        out_bits: list[int] = []
        for sym in symbols:
            if sym < 0 or sym >= self.palette_size:
                raise ValueError(
                    f"symbol {sym} out of palette [0, {self.palette_size})"
                )
            q = sym >> self.k
            # Unary prefix: q zeros + 1 terminator
            out_bits.extend([0] * q)
            out_bits.append(1)
            # Binary suffix: low k bits
            for j in range(self.k - 1, -1, -1):
                out_bits.append((sym >> j) & 1)
        while len(out_bits) % 8 != 0:
            out_bits.append(0)
        out = bytearray()
        for i in range(0, len(out_bits), 8):
            byte = 0
            for bit in out_bits[i:i + 8]:
                byte = (byte << 1) | (bit & 1)
            out.append(byte)
        return bytes(out)

    def encoded_bit_length(self, symbols: list[int]) -> int:
        """Return the exact Rice-Golomb code-length for a symbol stream."""
        if not symbols:
            return 0
        total = 0
        for sym in symbols:
            if sym < 0 or sym >= self.palette_size:
                raise ValueError(f"symbol {sym} out of palette")
            q = sym >> self.k
            total += q + 1 + self.k
        return total


class PactNervSelectorV3Substrate(nn.Module):
    """Pact-NeRV-SELECTOR-V3 renderer (L0 SKETCH; Rice-Golomb selector)."""

    def __init__(self, cfg: PactNervSelectorV3Config) -> None:
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
            raise ValueError("decoder_channels too short for num_upsample_blocks")
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)
        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.register_buffer("selectors", torch.zeros(cfg.num_pairs, dtype=torch.long))
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
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")
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
        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
