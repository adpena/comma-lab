#!/usr/bin/env python
# ruff: noqa: E402
"""HiNeRV substrate inflate (≤200 LOC; substrate-engineering budget).

Per CLAUDE.md HNeRV parity discipline lesson 4. Multi-stage decoder with
per-stage RGB heads; only the FINAL stage's RGB is written to dst.raw at
inference time (intermediate-stage RGBs are training auxiliaries).

Wire format matches src/tac/hinerv_as_renderer.py ARCHIVE_GRAMMAR_HINERV.
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

MAGIC = b"HiNV"
FORMAT_ID = 0x63
CAMERA_H, CAMERA_W = 874, 1164


class _StageDecoder(nn.Module):
    def __init__(self, latent_dim: int, prev_channels: int | None,
                 out_channels: int, init_h: int, init_w: int) -> None:
        super().__init__()
        self.out_channels = out_channels; self.init_h = init_h; self.init_w = init_w
        self.stem = nn.Linear(latent_dim, out_channels * init_h * init_w)
        self.has_prev = prev_channels is not None
        if self.has_prev:
            self.prev_proj = nn.Conv2d(prev_channels, out_channels, 1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.rgb_0 = nn.Conv2d(out_channels, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(out_channels, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        x = self.stem(z).view(B, self.out_channels, self.init_h, self.init_w)
        x = torch.sin(x)
        if self.has_prev and prev is not None:
            x = x + self.prev_proj(prev)
        x = torch.sin(self.conv1(x))
        x = torch.sin(self.conv2(x)) + x
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return x, torch.stack([f0, f1], dim=1)


class HiNeRVDecoder(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int, n_levels: int,
                 base_h: int, base_w: int) -> None:
        super().__init__()
        self.base_h = base_h; self.base_w = base_w; self.n_levels = n_levels
        c_taper = [base_channels for _ in range(n_levels)]
        c_taper[-1] = max(8, int(base_channels * 0.75))
        self.stages = nn.ModuleList()
        for i in range(n_levels):
            scale = 2 ** i
            stage_h = base_h * scale; stage_w = base_w * scale
            prev_channels = c_taper[i - 1] if i > 0 else None
            self.stages.append(_StageDecoder(
                latent_dim=latent_dim, prev_channels=prev_channels,
                out_channels=c_taper[i], init_h=stage_h, init_w=stage_w,
            ))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        prev = None; final_rgb = None
        for i, stage in enumerate(self.stages):
            if prev is not None:
                stage_h = self.base_h * (2 ** i); stage_w = self.base_w * (2 ** i)
                prev = F.interpolate(prev, size=(stage_h, stage_w),
                                     mode="bilinear", align_corners=False)
            features, rgb = stage(z, prev)
            prev = features; final_rgb = rgb
        return final_rgb


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


def _schema_keys(decoder: HiNeRVDecoder) -> list[tuple[str, tuple[int, ...]]]:
    sd = decoder.state_dict(); out = []
    for i in range(decoder.n_levels):
        keys = [f"stages.{i}.stem.weight", f"stages.{i}.stem.bias"]
        if i > 0:
            keys += [f"stages.{i}.prev_proj.weight", f"stages.{i}.prev_proj.bias"]
        keys += [f"stages.{i}.conv1.weight", f"stages.{i}.conv1.bias",
                 f"stages.{i}.conv2.weight", f"stages.{i}.conv2.bias",
                 f"stages.{i}.rgb_0.weight", f"stages.{i}.rgb_0.bias",
                 f"stages.{i}.rgb_1.weight", f"stages.{i}.rgb_1.bias"]
        for key in keys:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
    return out


def inflate(src_bin: str, dst_raw: str) -> int:
    blob = Path(src_bin).read_bytes()
    if blob[:4] != MAGIC:
        raise ValueError(f"bad magic {blob[:4]!r}")
    (version, fid, latent_dim, n_pairs, base_channels, n_levels) = struct.unpack_from("<HHHHHH", blob, 4)  # DEAD_BYTES_AUDIT_OK: forward-compat version field; codec is v1 by construction (single MAGIC + format_id check below)
    if fid != FORMAT_ID:
        raise ValueError(f"unexpected format_id {fid:#x}")
    pos = 16
    (decoder_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    decoder_blob = blob[pos:pos + decoder_len]; pos += decoder_len
    (scale_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    scale_table = blob[pos:pos + scale_len]; pos += scale_len
    (latent_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    latent_blob = blob[pos:pos + latent_len]

    # Defaults baked-in for Phase A (base_h=96, base_w=128 for n_levels=3 → 384x512).
    if n_levels == 3:
        base_h, base_w = 96, 128
    elif n_levels == 2:
        base_h, base_w = 192, 256
    else:
        raise ValueError(f"Phase A inflate supports n_levels=2 or 3, got {n_levels}")
    decoder = HiNeRVDecoder(latent_dim, base_channels, n_levels, base_h, base_w)
    schema = _schema_keys(decoder)
    int8_codes = brotli.decompress(decoder_blob)
    sd_pos = scale_pos = 0
    for key, shape in schema:
        n_elem = 1
        for d_ in shape:
            n_elem *= d_
        codes = np.frombuffer(int8_codes, dtype=np.int8, count=n_elem, offset=sd_pos); sd_pos += n_elem
        scale = np.frombuffer(scale_table, dtype=np.float16, count=1, offset=scale_pos); scale_pos += 2
        decoder.state_dict()[key].copy_(
            torch.from_numpy(codes.astype(np.float32) * float(scale[0])).reshape(shape)
        )
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
