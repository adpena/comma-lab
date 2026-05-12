#!/usr/bin/env python
# ruff: noqa: E402
"""TCNeRV substrate inflate (≤200 LOC; substrate-engineering budget).

Per CLAUDE.md HNeRV parity discipline lesson 4. Reconstructs spatial decoder
+ temporal conv stack + per-pair latents, applies temporal conv ONCE to the
full latent sequence, then per-pair forward + write raw RGB.

Wire format matches src/tac/tcnerv_as_renderer.py ARCHIVE_GRAMMAR_TCNERV.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = b"TCNV"
FORMAT_ID = 0x64
CAMERA_H, CAMERA_W = 874, 1164


class TemporalConvStack(nn.Module):
    def __init__(self, latent_dim: int, kernel: int, n_layers: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList([
            nn.Conv1d(latent_dim, latent_dim, kernel, padding=kernel // 2)
            for _ in range(n_layers)
        ])

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        x = latent_seq.t().unsqueeze(0)
        identity = x
        for layer in self.layers:
            x = torch.tanh(layer(x))
        x = x + identity
        return x.squeeze(0).t().contiguous()


class TCNeRVDecoder(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int = 36,
                 base_h: int = 6, base_w: int = 8, n_stages: int = 6,
                 temporal_kernel: int = 3, temporal_n_layers: int = 2) -> None:
        super().__init__()
        self.base_h = base_h; self.base_w = base_w
        self.temporal_conv = TemporalConvStack(latent_dim, temporal_kernel, temporal_n_layers)
        C = base_channels
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        self.stem = nn.Linear(latent_dim, self.channels[0] * base_h * base_w)
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

    def spatial_forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
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


def _schema_keys(decoder: TCNeRVDecoder, n_stages: int = 6,
                 temporal_n_layers: int = 2) -> list[tuple[str, tuple[int, ...]]]:
    sd = decoder.state_dict()
    keys: list[str] = []
    for i in range(temporal_n_layers):
        keys += [f"temporal_conv.layers.{i}.weight", f"temporal_conv.layers.{i}.bias"]
    keys += ["stem.weight", "stem.bias"]
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
    (version, fid, latent_dim, n_pairs, t_kernel, t_layers) = struct.unpack_from("<HHHHHH", blob, 4)
    if fid != FORMAT_ID:
        raise ValueError(f"unexpected format_id {fid:#x}")
    pos = 16
    (spatial_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    spatial_blob = blob[pos:pos + spatial_len]; pos += spatial_len
    (temp_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    temp_blob = blob[pos:pos + temp_len]; pos += temp_len
    (scale_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    scale_table = blob[pos:pos + scale_len]; pos += scale_len
    (latent_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    latent_blob = blob[pos:pos + latent_len]

    base_channels = 36
    decoder = TCNeRVDecoder(latent_dim, base_channels=base_channels,
                             temporal_kernel=t_kernel, temporal_n_layers=t_layers)
    schema = _schema_keys(decoder, temporal_n_layers=t_layers)
    spatial_codes = brotli.decompress(spatial_blob)
    temp_codes = brotli.decompress(temp_blob) if temp_blob else b""
    spatial_pos = temp_pos = scale_pos = 0
    for key, shape in schema:
        n_elem = 1
        for d_ in shape:
            n_elem *= d_
        is_temporal = key.startswith("temporal_conv.")
        if is_temporal:
            codes = np.frombuffer(temp_codes, dtype=np.int8, count=n_elem, offset=temp_pos)
            temp_pos += n_elem
        else:
            codes = np.frombuffer(spatial_codes, dtype=np.int8, count=n_elem, offset=spatial_pos)
            spatial_pos += n_elem
        scale = np.frombuffer(scale_table, dtype=np.float16, count=1, offset=scale_pos); scale_pos += 2
        decoder.state_dict()[key].copy_(
            torch.from_numpy(codes.astype(np.float32) * float(scale[0])).reshape(shape)
        )
    decoder.eval()

    raw_latents = _decode_latents(latent_blob)
    # Apply temporal conv ONCE.
    z_temporal = decoder.temporal_conv(raw_latents)
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder.spatial_forward(z_temporal[i:i + 1])
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
