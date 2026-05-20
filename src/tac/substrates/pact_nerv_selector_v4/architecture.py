# SPDX-License-Identifier: MIT
"""pact_nerv_selector_v4 architecture - run-length-encoded selector (L0 SKETCH).

The distinguishing primitive vs FEC6 fixed-Huffman: RLE exploits temporal
runs of consecutive identical selectors. Per Robinson-Cherry 1967 + Capon
1959, RLE achieves rate savings proportional to mean run length.
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
class PactNervSelectorV4Config:
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


class RunLengthSelectorCoder:
    """Run-length selector coder over FEC6 k=16 palette (Robinson-Cherry 1967).

    Encoding format: for each maximal run of equal symbols, emit
    (value: u8, run_length: varint) pair. The varint encoding (LEB128-style)
    uses 7 bits per byte for the magnitude + 1 continuation bit; short runs
    (1-127) cost 2 bytes per run, longer runs cost more.

    Static varint = HARD-EARNED-LITERATURE; no value-distribution context
    at L0 = CARGO-CULTED (L1 sweep: RLE+Huffman hybrid for value encoding).
    """

    def __init__(self, palette_size: int) -> None:
        if palette_size < 2:
            raise ValueError(f"palette_size must be >= 2; got {palette_size}")
        if palette_size > 256:
            raise ValueError(f"palette_size must be <= 256; got {palette_size}")
        self.palette_size = palette_size

    @staticmethod
    def _encode_varint(n: int) -> bytes:
        """LEB128-style varint encoding (7 bits/byte + continuation)."""
        if n < 0:
            raise ValueError(f"run_length must be non-negative; got {n}")
        out = bytearray()
        if n == 0:
            return b"\x00"
        while n > 0:
            byte = n & 0x7F
            n >>= 7
            if n > 0:
                byte |= 0x80
            out.append(byte)
        return bytes(out)

    @staticmethod
    def _decode_varint(blob: bytes, pos: int) -> tuple[int, int]:
        """Decode varint starting at pos; return (value, next_pos)."""
        n = 0
        shift = 0
        while True:
            if pos >= len(blob):
                raise ValueError("truncated varint")
            byte = blob[pos]
            pos += 1
            n |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                return n, pos
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")

    def encode(self, symbols: list[int]) -> bytes:
        """Encode a symbol stream into the RLE bitstream."""
        if not symbols:
            return b""
        for sym in symbols:
            if sym < 0 or sym >= self.palette_size:
                raise ValueError(
                    f"symbol {sym} out of palette [0, {self.palette_size})"
                )
        out = bytearray()
        i = 0
        while i < len(symbols):
            value = symbols[i]
            run_length = 1
            while i + run_length < len(symbols) and symbols[i + run_length] == value:
                run_length += 1
            out.append(value & 0xFF)
            out.extend(self._encode_varint(run_length))
            i += run_length
        return bytes(out)

    def decode(self, blob: bytes) -> list[int]:
        """Decode the RLE bitstream into a symbol stream."""
        if not blob:
            return []
        out: list[int] = []
        pos = 0
        while pos < len(blob):
            value = blob[pos]
            pos += 1
            if value >= self.palette_size:
                raise ValueError(
                    f"decoded value {value} >= palette_size {self.palette_size}"
                )
            run_length, pos = self._decode_varint(blob, pos)
            out.extend([value] * run_length)
        return out

    def encoded_byte_length(self, symbols: list[int]) -> int:
        return len(self.encode(symbols))


class PactNervSelectorV4Substrate(nn.Module):
    """Pact-NeRV-SELECTOR-V4 renderer (L0 SKETCH; RLE selector)."""

    def __init__(self, cfg: PactNervSelectorV4Config) -> None:
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
            raise ValueError("decoder_channels too short")
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
