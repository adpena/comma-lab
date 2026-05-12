#!/usr/bin/env python
# ruff: noqa: E402
"""BlockNeRV substrate inflate (≤200 LOC; substrate-engineering budget).

Per CLAUDE.md HNeRV parity discipline lesson 4 (inflate ≤200 LOC waiver) +
lesson 9 (runtime closure: torch + brotli only). Reads BlockNeRV monolithic
0.bin, reconstructs the tile-decoder + per-tile coord embeddings + per-pair
latents, then iterates per-pair forward through the tile decoder and writes
raw RGB frames at camera resolution.

Wire format (matches src/tac/blocknerv_as_renderer.py ARCHIVE_GRAMMAR_BLOCKNERV).
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

MAGIC = b"BNRV"
FORMAT_ID = 0x60
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


class TileDecoder(nn.Module):
    """Compact tile decoder (mirror of src/tac/blocknerv_as_renderer.BlockNeRVRenderer)."""

    def __init__(self, latent_dim: int, coord_embed_dim: int, base_channels: int,
                 tile_h: int, tile_w: int, n_stages: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.coord_embed_dim = coord_embed_dim
        self.tile_h = tile_h; self.tile_w = tile_w
        init_h = tile_h // (2 ** n_stages); init_w = tile_w // (2 ** n_stages)
        self._init_h = init_h; self._init_w = init_w
        C = base_channels
        self.channels = [C] + [max(8, int(C * (0.85 ** (i + 1)))) for i in range(n_stages)]
        self.stem = nn.Linear(latent_dim + coord_embed_dim, self.channels[0] * init_h * init_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_stages):
            in_ch = self.channels[i]; out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, padding=1),
            nn.Conv2d(final_ch, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, coord: torch.Tensor, tile_rows: int, tile_cols: int) -> torch.Tensor:
        B = z.shape[0]; n_tiles = tile_rows * tile_cols
        z_exp = z.unsqueeze(1).expand(B, n_tiles, -1)
        coord_exp = coord.unsqueeze(0).expand(B, n_tiles, -1)
        z_flat = torch.cat([z_exp, coord_exp], dim=-1).reshape(B * n_tiles, -1)
        x = self.stem(z_flat).view(B * n_tiles, self.channels[0], self._init_h, self._init_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        H = tile_rows * self.tile_h; W = tile_cols * self.tile_w
        f0 = f0.view(B, tile_rows, tile_cols, 3, self.tile_h, self.tile_w)
        f0 = f0.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, H, W)
        f1 = f1.view(B, tile_rows, tile_cols, 3, self.tile_h, self.tile_w)
        f1 = f1.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, H, W)
        return torch.stack([f0, f1], dim=1)


def _schema(latent_dim: int, coord_embed_dim: int, base_channels: int,
            tile_h: int, tile_w: int, n_stages: int = 3) -> list[tuple[str, tuple[int, ...]]]:
    """Replicate BlockNeRVRenderer.schema for known config."""
    decoder = TileDecoder(latent_dim, coord_embed_dim, base_channels, tile_h, tile_w, n_stages)
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


def _decode_latents(blob: bytes) -> torch.Tensor:
    raw = brotli.decompress(blob)
    n, d = struct.unpack_from("<II", raw, 0)
    pos = 8
    mins = np.frombuffer(raw, dtype=np.float16, count=d, offset=pos); pos += 2 * d
    scales = np.frombuffer(raw, dtype=np.float16, count=d, offset=pos); pos += 2 * d
    delta_lo = np.frombuffer(raw, dtype=np.uint8, count=n * d, offset=pos).astype(np.int32); pos += n * d
    delta_hi = np.frombuffer(raw, dtype=np.uint8, count=n * d, offset=pos).astype(np.int32)
    delta_zz = (delta_hi << 8) | delta_lo
    delta = np.where(delta_zz % 2 == 0, delta_zz // 2, -((delta_zz + 1) // 2))
    delta = delta.reshape(n, d)
    q = np.cumsum(delta, axis=0)
    return torch.from_numpy(q.astype(np.float32) * scales.astype(np.float32) + mins.astype(np.float32))


def inflate(src_bin: str, dst_raw: str) -> int:
    blob = Path(src_bin).read_bytes()
    if blob[:4] != MAGIC:
        raise ValueError(f"bad magic {blob[:4]!r}")
    (version, fid, latent_dim, n_pairs, tile_rows, tile_cols) = struct.unpack_from("<HHHHHH", blob, 4)
    if fid != FORMAT_ID:
        raise ValueError(f"unexpected format_id {fid:#x}")
    pos = 16
    (decoder_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    decoder_blob = blob[pos:pos + decoder_len]; pos += decoder_len
    (tile_coord_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    tile_coord_bytes = blob[pos:pos + tile_coord_len]; pos += tile_coord_len
    (scale_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    scale_table = blob[pos:pos + scale_len]; pos += scale_len
    (latent_len,) = struct.unpack_from("<I", blob, pos); pos += 4
    latent_blob = blob[pos:pos + latent_len]; pos += latent_len
    # Sidecar (Phase A: empty) intentionally not read past pos.

    # Reconstruct decoder using fixed config for known archive (Phase A: defaults).
    base_channels = 28; coord_embed_dim = 8
    tile_h = (CAMERA_H * 0 + 384) // tile_rows  # eval_size_h / tile_rows
    tile_w = (CAMERA_W * 0 + 512) // tile_cols
    decoder = TileDecoder(latent_dim, coord_embed_dim, base_channels, tile_h, tile_w, n_stages=3)
    schema = _schema(latent_dim, coord_embed_dim, base_channels, tile_h, tile_w)

    int8_codes = brotli.decompress(decoder_blob)
    sd_pos = 0
    scale_pos = 0
    for key, shape in schema:
        n_elem = 1
        for d_ in shape:
            n_elem *= d_
        codes = np.frombuffer(int8_codes, dtype=np.int8, count=n_elem, offset=sd_pos)
        sd_pos += n_elem
        scale = np.frombuffer(scale_table, dtype=np.float16, count=1, offset=scale_pos)
        scale_pos += 2
        tensor = torch.from_numpy(codes.astype(np.float32) * float(scale[0])).reshape(shape)
        decoder.state_dict()[key].copy_(tensor)
    n_tiles = tile_rows * tile_cols
    coord = torch.from_numpy(np.frombuffer(tile_coord_bytes, dtype=np.float16,
                                            count=n_tiles * coord_embed_dim).astype(np.float32))
    coord = coord.reshape(n_tiles, coord_embed_dim)
    decoder.eval()

    latents = _decode_latents(latent_blob)
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            z = latents[i:i + 1]
            decoded = decoder(z, coord, tile_rows, tile_cols)  # (1, 2, 3, H_native, W_native)
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
