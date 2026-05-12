#!/usr/bin/env python
# ruff: noqa: E402
"""NeRVdc substrate inflate (≤200 LOC; substrate-engineering budget).

Per CLAUDE.md HNeRV parity discipline lesson 4. Reads NeRVdc monolithic
0.bin, reconstructs decoder + per-pair latents, then iterates per-pair
forward — passing the previous pair's last frame as conditioning.

Wire format matches src/tac/nervdc_as_renderer.py ARCHIVE_GRAMMAR_NERVDC.
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

MAGIC = b"NRVc"
FORMAT_ID = 0x66
CAMERA_H, CAMERA_W = 874, 1164


class _CondSummary(nn.Module):
    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(8, cond_dim)

    def forward(self, prev: torch.Tensor) -> torch.Tensor:
        x = prev.float() / 127.5 - 1.0
        h = self.stem(x).flatten(1)
        return self.head(h)


class NeRVdcDecoder(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, base_channels: int,
                 base_h: int = 6, base_w: int = 8, n_stages: int = 6) -> None:
        super().__init__()
        self.latent_dim = latent_dim; self.cond_dim = cond_dim
        self.base_h = base_h; self.base_w = base_w
        self.cond_summary = _CondSummary(cond_dim)
        C = base_channels
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
        self.stem = nn.Linear(latent_dim + cond_dim, self.channels[0] * base_h * base_w)
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

    def forward(self, z: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        B = z.shape[0]
        if prev is None:
            cond = torch.zeros((B, self.cond_dim), dtype=z.dtype, device=z.device)
        else:
            cond = self.cond_summary(prev)
        z_cond = torch.cat([z, cond], dim=-1)
        x = self.stem(z_cond).view(B, self.channels[0], self.base_h, self.base_w)
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
    n, d = struct.unpack_from("<II", raw, 0); off = 8
    mins = np.frombuffer(raw, dtype=np.float16, count=d, offset=off); off += d * 2
    scales = np.frombuffer(raw, dtype=np.float16, count=d, offset=off); off += d * 2
    delta_lo = np.frombuffer(raw, dtype=np.uint8, count=n * d, offset=off); off += n * d
    delta_hi = np.frombuffer(raw, dtype=np.uint8, count=n * d, offset=off)
    delta_zz = (delta_hi.astype(np.int32) << 8) | delta_lo.astype(np.int32)
    delta_zz = delta_zz.reshape(n, d)
    delta = np.where(delta_zz % 2 == 0, delta_zz // 2, -((delta_zz + 1) // 2))
    q = np.cumsum(delta, axis=0)
    return torch.from_numpy(q.astype(np.float32) * scales.astype(np.float32) + mins.astype(np.float32))


def _decode_decoder(blob: bytes, scale_table: bytes,
                    schema: list[tuple[str, tuple[int, ...]]]) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd: dict[str, torch.Tensor] = {}
    off = 0
    for i, (key, shape) in enumerate(schema):
        n_elems = 1
        for s in shape:
            n_elems *= s
        chunk = np.frombuffer(raw, dtype=np.int8, count=n_elems, offset=off); off += n_elems
        scale = np.frombuffer(scale_table, dtype=np.float16, count=1, offset=i * 2)[0]
        sd[key] = torch.from_numpy(chunk.astype(np.float32) * float(scale)).reshape(shape)
    return sd


def _build_schema(latent_dim: int, cond_dim: int, base_channels: int, n_stages: int = 6
                  ) -> list[tuple[str, tuple[int, ...]]]:
    C = base_channels
    channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]
    base_h = 6; base_w = 8
    schema: list[tuple[str, tuple[int, ...]]] = []
    schema += [
        ("cond_summary.stem.0.weight", (8, 3, 3, 3)), ("cond_summary.stem.0.bias", (8,)),
        ("cond_summary.head.weight", (cond_dim, 8)), ("cond_summary.head.bias", (cond_dim,)),
        ("stem.weight", (channels[0] * base_h * base_w, latent_dim + cond_dim)),
        ("stem.bias", (channels[0] * base_h * base_w,)),
    ]
    for i in range(n_stages):
        in_ch = channels[i]; out_ch = channels[i + 1]
        schema += [(f"blocks.{i}.weight", (out_ch * 4, in_ch, 3, 3)),
                   (f"blocks.{i}.bias", (out_ch * 4,))]
    for i in range(n_stages):
        in_ch = channels[i]; out_ch = channels[i + 1]
        if in_ch != out_ch:
            schema += [(f"skips.{i}.weight", (out_ch, in_ch, 1, 1)),
                       (f"skips.{i}.bias", (out_ch,))]
    final_ch = channels[-1]
    schema += [
        ("refine.0.weight", (final_ch // 2, final_ch, 3, 3)),
        ("refine.0.bias", (final_ch // 2,)),
        ("refine.1.weight", (final_ch, final_ch // 2, 3, 3)),
        ("refine.1.bias", (final_ch,)),
        ("rgb_0.weight", (3, final_ch, 3, 3)), ("rgb_0.bias", (3,)),
        ("rgb_1.weight", (3, final_ch, 3, 3)), ("rgb_1.bias", (3,)),
    ]
    return schema


def inflate(src_bin: Path, dst_raw: Path) -> None:
    data = src_bin.read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"bad magic: {data[:4]!r} != {MAGIC!r}")
    format_id = struct.unpack_from("<H", data, 6)[0]
    if format_id != FORMAT_ID:
        raise ValueError(f"format_id={format_id} != {FORMAT_ID}")
    latent_dim = struct.unpack_from("<H", data, 8)[0]
    cond_dim = struct.unpack_from("<H", data, 10)[0]
    n_pairs = struct.unpack_from("<H", data, 12)[0]
    base_channels = struct.unpack_from("<H", data, 14)[0]
    off = 16
    dec_len = struct.unpack_from("<I", data, off)[0]; off += 4
    dec_blob = data[off:off + dec_len]; off += dec_len
    scale_len = struct.unpack_from("<I", data, off)[0]; off += 4
    scale_table = data[off:off + scale_len]; off += scale_len
    lat_len = struct.unpack_from("<I", data, off)[0]; off += 4
    lat_blob = data[off:off + lat_len]; off += lat_len

    schema = _build_schema(latent_dim, cond_dim, base_channels)
    decoder = NeRVdcDecoder(latent_dim, cond_dim, base_channels)
    sd = _decode_decoder(dec_blob, scale_table, schema)
    decoder.load_state_dict(sd, strict=False)
    decoder.eval()
    latents = _decode_latents(lat_blob)

    dst_raw.parent.mkdir(parents=True, exist_ok=True)
    prev_frame: torch.Tensor | None = None
    with dst_raw.open("wb") as f:
        with torch.no_grad():
            for i in range(n_pairs):
                z = latents[i:i + 1]
                # Pass previous pair's last frame at camera resolution as conditioning,
                # downsampled to match the encoder's expected input.
                cond_input = (
                    None if prev_frame is None else
                    F.interpolate(prev_frame.unsqueeze(0), size=(96, 128),
                                  mode="bilinear", align_corners=False)
                )
                pair = decoder(z, prev=cond_input)
                up = F.interpolate(
                    pair.reshape(2, 3, 384, 512), size=(CAMERA_H, CAMERA_W),
                    mode="bicubic", align_corners=False,
                )
                up_uint8 = up.clamp(0, 255).round().to(torch.uint8)
                out = up_uint8.permute(0, 2, 3, 1).contiguous().numpy()
                f.write(out.tobytes())
                prev_frame = up_uint8[1].float()  # second frame as next conditioning


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <src_bin> <dst_raw>", file=sys.stderr)
        return 2
    inflate(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
