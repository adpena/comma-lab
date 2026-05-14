#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate SC++ substrate archive.

Reads ``<src>.bin`` (SC++ wire format: 16-byte header + brotli config_json +
block-FP weights + brotli latents), reconstructs the SCPPSubstrate decoder,
runs forward at 384x512, bicubic-upsamples to 874x1164, rounds to uint8, and
writes contiguous ``(N, H, W, 3)`` bytes to ``<dst>``.

Wire-format details + decoder live in ``src/tac/scpp_substrate.py``. This
runtime is intentionally minimal: only ``torch`` + ``brotli`` runtime deps.
No scorer, no PoseNet, no SegNet, no EfficientNet, no FastViT, no DALI,
no NVDEC.

Invoked by inflate.sh as:
    python inflate.py <data_dir>/<base>.bin <output_dir>/<base>.raw

Per CLAUDE.md "Inflate.py ≤100 LOC" budget (substrate-engineering waiver:
≤200 LOC). This file is ~150 LOC; no waiver needed.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import torch
import torch.nn as nn
import torch.nn.functional as F


CAMERA_H, CAMERA_W = 874, 1164
SCPP_MAGIC = 0xFE
SCPP_FORMAT_ID = 0x40
SCPP_VERSION = 1


class _FiLMBlock(nn.Module):
    """Depthwise-separable conv with FiLM gamma/beta conditioning."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.film_gamma = nn.Linear(cond_dim, channels)
        self.film_beta = nn.Linear(cond_dim, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.pointwise(self.depthwise(x))
        gamma = self.film_gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_beta(cond).unsqueeze(-1).unsqueeze(-1)
        return self.act(h * (1.0 + gamma) + beta)


class _SCPPDecoder(nn.Module):
    """Inflate-only mirror of tac.scpp_substrate.SCPPSubstrate.

    Pure forward path; no training scaffolding. Matches the trained
    substrate's parameter shapes byte-for-byte.
    """

    def __init__(self, latent_dim: int, base_channels: int, eval_h: int, eval_w: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.eval_h = eval_h
        self.eval_w = eval_w
        self.h0, self.w0 = 8, 8
        self.proj = nn.Linear(latent_dim, base_channels * self.h0 * self.w0)
        self.block1 = _FiLMBlock(base_channels, latent_dim)
        self.block2 = _FiLMBlock(base_channels, latent_dim)
        self.block3 = _FiLMBlock(base_channels, latent_dim)
        self.block4 = _FiLMBlock(base_channels, latent_dim)
        self.pair_head = nn.Conv2d(base_channels, 6, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        B = latents.shape[0]
        h = self.proj(latents).reshape(B, self.base_channels, self.h0, self.w0)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.block1(h, latents)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.block2(h, latents)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.block3(h, latents)
        h = F.interpolate(h, size=(self.eval_h, self.eval_w), mode="bilinear", align_corners=False)
        h = self.block4(h, latents)
        return self.pair_head(h).reshape(B, 2, 3, self.eval_h, self.eval_w)


def _unpack_blockfp(packed_bytes: bytes, meta: dict) -> dict[str, torch.Tensor]:
    """Inverse of tac.scpp_substrate._pack_state_dict_blockfp."""
    sigma = float(meta["sigma"])
    out: dict[str, torch.Tensor] = {}
    for ts in meta["per_tensor"]:
        n, n_blocks, bs = ts["n_elements"], ts["n_blocks"], ts["block_size"]
        off = ts["byte_offset"]
        exp_slice = packed_bytes[off : off + n_blocks]
        qint_slice = packed_bytes[off + n_blocks : off + n_blocks + n]
        exponents = torch.frombuffer(bytearray(exp_slice), dtype=torch.int8).float()
        qint_flat = torch.frombuffer(bytearray(qint_slice), dtype=torch.int8).float()
        scales = sigma * torch.pow(2.0, exponents)
        pad = n_blocks * bs - n
        if pad > 0:
            qint_flat = torch.cat([qint_flat, torch.zeros(pad)])
        weights = (qint_flat.reshape(n_blocks, bs) * scales.unsqueeze(1)).flatten()[:n]
        out[ts["name"]] = weights.reshape(ts["shape"]).contiguous()
    return out


def select_inflate_device() -> torch.device:
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError("PACT_INFLATE_DEVICE=mps forbidden; use cpu or cuda")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin: str, dst_raw: str) -> int:
    archive = Path(src_bin).read_bytes()
    if len(archive) < 16:
        raise ValueError(f"SC++ archive truncated: {len(archive)} bytes")

    magic, format_id, version, cfg_len, bfp_len, lat_len = struct.unpack_from(
        "<BBHIII", archive, 0
    )
    if magic != SCPP_MAGIC:
        raise ValueError(f"SC++ magic mismatch: got 0x{magic:02X}")
    if format_id != SCPP_FORMAT_ID:
        raise ValueError(f"SC++ format_id mismatch: got 0x{format_id:02X}")
    if version != SCPP_VERSION:
        raise ValueError(f"SC++ version mismatch: got {version}")

    pos = 16
    cfg_payload = json.loads(brotli.decompress(archive[pos : pos + cfg_len]).decode("utf-8"))
    pos += cfg_len
    state_dict = _unpack_blockfp(archive[pos : pos + bfp_len], cfg_payload["blockfp_meta"])
    pos += bfp_len
    lat_raw = brotli.decompress(archive[pos : pos + lat_len])
    lat_int8 = torch.frombuffer(bytearray(lat_raw), dtype=torch.int8).float()
    lat_scale = float(cfg_payload["latent_scale"])
    lat_shape = cfg_payload["latent_shape"]
    latents = (lat_int8 * lat_scale).reshape(lat_shape)

    config = cfg_payload["config"]
    device = select_inflate_device()
    decoder = _SCPPDecoder(
        latent_dim=int(config["latent_dim"]),
        base_channels=int(config["base_channels"]),
        eval_h=int(config["eval_height"]),
        eval_w=int(config["eval_width"]),
    ).to(device)
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = int(config["n_pairs"])
    batch_size = int(os.environ.get("PACT_INFLATE_BATCH_PAIRS", "16"))
    print(f"[scpp inflate] device={device.type}, n_pairs={n_pairs}", file=sys.stderr)

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, batch_size):
            j = min(i + batch_size, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, int(config["eval_height"]), int(config["eval_width"]))
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
