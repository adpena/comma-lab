#!/usr/bin/env python
# ruff: noqa: E402
"""FFNeRV substrate inflate (≤200 LOC; substrate-engineering budget).

Per CLAUDE.md HNeRV parity discipline lesson 4. Reads FFNeRV monolithic 0.bin,
reconstructs the Fourier-feature encoding (frequencies in archive) + NeRV
decoder + per-pair latents, then iterates per-pair forward and writes raw RGB.

Wire format matches src/tac/ffnerv_as_renderer.py ARCHIVE_GRAMMAR_FFNERV.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"FFNV"
FORMAT_ID = 0x61
CAMERA_H, CAMERA_W = 874, 1164


class FFNeRVDecoder(nn.Module):
    """Mirror of src/tac/ffnerv_as_renderer.FFNeRVRenderer (decoder side)."""

    def __init__(self, latent_dim: int, n_freq: int, base_channels: int,
                 base_h: int = 6, base_w: int = 8, n_stages: int = 6) -> None:
        super().__init__()
        self.latent_dim = latent_dim; self.n_freq = n_freq
        self.base_h = base_h; self.base_w = base_w
        self.encoded_dim = latent_dim * (1 + 2 * n_freq)
        C = base_channels
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        self.stem = nn.Linear(self.encoded_dim, self.channels[0] * base_h * base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_stages):
            in_ch = self.channels[i]; out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.register_buffer("frequencies", torch.zeros(n_freq))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        angles = z.unsqueeze(-1) * self.frequencies.view(1, 1, -1) * (2 * math.pi)
        sin_part = torch.sin(angles).reshape(z.shape[0], -1)
        cos_part = torch.cos(angles).reshape(z.shape[0], -1)
        return torch.cat([z, sin_part, cos_part], dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_enc = self.encode(z)
        B = z.shape[0]
        x = self.stem(z_enc).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def _decode_latents(blob: bytes) -> torch.Tensor:
    raw = brotli.decompress(blob)
    n, d = struct.unpack_from("<II", raw, 0); pos = 8
    mins = np.frombuffer(raw, dtype=np.float16, count=d, offset=pos); pos += 2 * d
    scales = np.frombuffer(raw, dtype=np.float16, count=d, offset=pos); pos += 2 * d
    delta_lo = np.frombuffer(raw, dtype=np.uint8, count=n * d, offset=pos).astype(np.int32); pos += n * d
    delta_hi = np.frombuffer(raw, dtype=np.uint8, count=n * d, offset=pos).astype(np.int32)
    delta_zz = (delta_hi << 8) | delta_lo
    delta = np.where(delta_zz % 2 == 0, delta_zz // 2, -((delta_zz + 1) // 2))
    q = np.cumsum(delta.reshape(n, d), axis=0)
    return torch.from_numpy(q.astype(np.float32) * scales.astype(np.float32) + mins.astype(np.float32))


def _schema_keys(decoder: FFNeRVDecoder, n_stages: int = 6) -> list[tuple[str, tuple[int, ...]]]:
    sd = decoder.state_dict()
    keys = ["stem.weight", "stem.bias"]
    for i in range(n_stages):
        keys += [f"blocks.{i}.weight", f"blocks.{i}.bias"]
    for i in range(n_stages):
        if isinstance(decoder.skips[i], nn.Conv2d):
            keys += [f"skips.{i}.weight", f"skips.{i}.bias"]
    keys += ["refine.0.weight", "refine.0.bias", "refine.1.weight", "refine.1.bias",
             "rgb_0.weight", "rgb_0.bias", "rgb_1.weight", "rgb_1.bias"]
    return [(k, tuple(sd[k].shape)) for k in keys if k in sd]


def inflate(src_bin: str, dst_raw: str) -> int:
    blob = Path(src_bin).read_bytes()
    if blob[:4] != MAGIC:
        raise ValueError(f"bad magic {blob[:4]!r}")
    (version, fid, latent_dim, n_pairs, n_freq, base_channels) = struct.unpack_from("<HHHHHH", blob, 4)  # DEAD_BYTES_AUDIT_OK: forward-compat version field; codec is v1 by construction (single MAGIC + format_id check below)
    if fid != FORMAT_ID:
        raise ValueError(f"unexpected format_id {fid:#x}")
    pos = 16
    (decoder_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    decoder_blob = blob[pos:pos + decoder_len]; pos += decoder_len
    (freq_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    freq_bytes = blob[pos:pos + freq_len]; pos += freq_len
    (scale_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    scale_table = blob[pos:pos + scale_len]; pos += scale_len
    (latent_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    latent_blob = blob[pos:pos + latent_len]

    decoder = FFNeRVDecoder(latent_dim, n_freq, base_channels)
    schema = _schema_keys(decoder)
    int8_codes = brotli.decompress(decoder_blob)
    sd_pos = scale_pos = 0
    for key, shape in schema:
        n_elem = 1
        for d_ in shape:
            n_elem *= d_
        codes = np.frombuffer(int8_codes, dtype=np.int8, count=n_elem, offset=sd_pos)
        sd_pos += n_elem
        scale = np.frombuffer(scale_table, dtype=np.float16, count=1, offset=scale_pos); scale_pos += 2
        decoder.state_dict()[key].copy_(
            torch.from_numpy(codes.astype(np.float32) * float(scale[0])).reshape(shape)
        )
    freqs = np.frombuffer(freq_bytes, dtype=np.float16, count=n_freq).astype(np.float32)
    decoder.frequencies.copy_(torch.from_numpy(freqs))
    decoder.eval()

    latents = _decode_latents(latent_blob)
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i:i + 1])
            up = F.interpolate(decoded.reshape(2, 3, decoded.shape[-2], decoded.shape[-1]),
                                size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            fout.write(frames.tobytes())
            written += 2
    print(f"saved {written} frames", file=sys.stderr)
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
