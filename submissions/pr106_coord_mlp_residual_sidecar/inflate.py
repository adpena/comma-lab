#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + coordinate-MLP-family Laplacian-smoothness residual sidecar.

The generic coordinate-MLP family (SIREN/NeRV/HNeRV/Cool-Chic/C3/Fourier-feat
MLPs) shares ONE prior: the residual is locally smooth, so most variation lives
in a small set of low-Laplacian-magnitude pixels. We encode the residual at
1/8 resolution with INT8 quantisation + per-frame scale prefix, then bicubic-
upsample to camera resolution.

This is the simplest possible coordinate-MLP-family residual: it works as the
shared baseline that ANY future SIREN/Fourier-feat MLP can compete against.

Wire format: magic 0xFD + format_id 0x14 + PR106 bytes + length-prefixed
coord_mlp residual blob.

Family-scoped: rejects any archive whose format_id ≠ 0x14.

Per CLAUDE.md HNeRV parity lesson 4 (inflate ≤ 200 LOC). NO_NVDEC_NEEDED.
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from pr106_inner_sidecar import (  # type: ignore[import-not-found]
    apply_pr106_r2_sidecar_corrections,
    unwrap_pr106_r2_sidecar,
)

PR106_RESIDUAL_MAGIC = 0xFD
COORD_MLP_FORMAT_ID = 0x14
CAMERA_H, CAMERA_W = 874, 1164
DOWNSAMPLE_FACTOR = 8
LOW_H, LOW_W = CAMERA_H // DOWNSAMPLE_FACTOR, CAMERA_W // DOWNSAMPLE_FACTOR
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic 0x{magic:02X}")
    if format_id != COORD_MLP_FORMAT_ID:
        raise ValueError(
            f"format_id 0x{format_id:02X} != coord_mlp 0x{COORD_MLP_FORMAT_ID:02X}"
        )
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    residual_bytes = blob[pos : pos + residual_len]
    if pos + residual_len != len(blob):
        raise ValueError("trailing bytes")
    return bytes(pr106_bytes), bytes(residual_bytes)


def decode_coord_mlp_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode (T, H, W, 3) float residual from low-resolution INT8 stream.

    Wire format:
        per frame: 4B scale (float32) + int8 coeffs at LOW_H * LOW_W * 3.

    Returns float64 (T, H, W, 3); bicubic-upsampled to camera resolution.
    """
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    per_frame_bytes = 4 + LOW_H * LOW_W * RGB_CHANNELS
    expected = n_frames * per_frame_bytes
    if len(blob) != expected:
        raise ValueError(
            f"coord_mlp residual size mismatch: {len(blob)} != {expected} "
            f"(n_frames={n_frames}, per_frame={per_frame_bytes})"
        )
    coeffs = np.zeros((n_frames, LOW_H, LOW_W, RGB_CHANNELS), dtype=np.float64)
    pos = 0
    for t in range(n_frames):
        (scale,) = struct.unpack_from("<f", blob, pos)
        pos += 4
        arr = np.frombuffer(
            blob, dtype=np.int8, count=LOW_H * LOW_W * RGB_CHANNELS, offset=pos
        ).astype(np.float64)
        pos += LOW_H * LOW_W * RGB_CHANNELS
        coeffs[t] = arr.reshape(LOW_H, LOW_W, RGB_CHANNELS) * scale
    t = torch.from_numpy(coeffs).permute(0, 3, 1, 2)
    up = F.interpolate(t, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    return up.permute(0, 2, 3, 1).numpy()


def select_inflate_device() -> torch.device:
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError("MPS forbidden")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inflate(src_bin: str, dst_raw: str) -> int:
    blob = Path(src_bin).read_bytes()
    pr106_r2_bytes, residual_blob = parse_residual_archive(blob)
    raw_pr106, sidecar_blob = unwrap_pr106_r2_sidecar(pr106_r2_bytes)
    decoder_sd, latents, meta = parse_packed_archive(raw_pr106)
    apply_pr106_r2_sidecar_corrections(latents, sidecar_blob)
    device = select_inflate_device()
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]
    n_frames = n_pairs * 2
    residual = decode_coord_mlp_residual(residual_blob, n_frames)
    print(
        f"[inflate] PR106+coord_mlp residual: device={device.type}, "
        f"residual_bytes={len(residual_blob)}",
        file=sys.stderr,
    )
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i : i + 1]).reshape(2, 3, eval_h, eval_w)
            up = F.interpolate(decoded, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
            corrected = frames + residual[i * 2 : i * 2 + 2]
            out = np.clip(np.round(corrected), 0, 255).astype(np.uint8)
            fout.write(out.tobytes())
            written += 2
    print(f"saved {written} frames")
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
