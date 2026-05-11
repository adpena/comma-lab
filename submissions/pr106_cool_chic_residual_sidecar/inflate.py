#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + Cool-Chic-style hierarchical pyramid residual sidecar.

Wire format: magic 0xFD + format_id 0x11 + PR106 bytes + length-prefixed
Cool-Chic residual blob. The residual is an upsample-cascade pyramid: each
level L stores int8-quantised residual coefficients at H/2^L, W/2^L that are
2× bilinear-upsampled L times to camera resolution and summed.

Family-scoped: rejects any archive whose format_id ≠ 0x11.

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
COOL_CHIC_FORMAT_ID = 0x11  # dense
COOL_CHIC_SPARSE_FORMAT_ID = 0x21  # sparse PacketIR
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes, int]:
    """Returns (pr106_bytes, residual_bytes, format_id); accepts dense + sparse."""
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic 0x{magic:02X}")
    if format_id not in (COOL_CHIC_FORMAT_ID, COOL_CHIC_SPARSE_FORMAT_ID):
        raise ValueError(f"format_id 0x{format_id:02X} not in cool_chic set")
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    residual_bytes = blob[pos : pos + residual_len]
    if pos + residual_len != len(blob):
        raise ValueError("trailing bytes")
    return bytes(pr106_bytes), bytes(residual_bytes), int(format_id)


def decode_cool_chic_residual_sparse(blob: bytes, n_frames: int) -> np.ndarray:
    """Sparse: per-level RLE of int8 coefficient field at H/2^L * W/2^L * 3.

    Outer wire: 2B n_levels (LE u16); per level: 4B scale + 4B uncompressed
    coefficient count + RLE-of-zeros payload. The decoder reconstructs the
    same (n_frames, H/2^L, W/2^L, 3) tensor per level, bilinear-upsamples to
    camera res, and sums.
    """
    from sparse_packet_ir_inline import (  # type: ignore[import-not-found]
        decode_rle_of_zeros_bytes,
    )
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    (n_levels,) = struct.unpack_from("<H", blob, 0)
    pos = 2
    residual = np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    for L in range(n_levels):
        scale, rle_len = struct.unpack_from("<fI", blob, pos)
        pos += 8
        rle_blob = blob[pos : pos + rle_len]
        pos += rle_len
        flat = decode_rle_of_zeros_bytes(rle_blob)
        h_L = CAMERA_H // (2 ** L)
        w_L = CAMERA_W // (2 ** L)
        expected = n_frames * h_L * w_L * RGB_CHANNELS
        if flat.size != expected:
            raise ValueError(f"sparse cool_chic level {L} size {flat.size} != {expected}")
        arr = flat.reshape(n_frames, h_L, w_L, RGB_CHANNELS).astype(np.float64) * scale
        if L == 0:
            residual += arr
        else:
            t = torch.from_numpy(arr).permute(0, 3, 1, 2)
            up = F.interpolate(t, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
            residual += up.permute(0, 2, 3, 1).numpy()
    if pos != len(blob):
        raise ValueError(f"sparse cool_chic trailing bytes: pos={pos} total={len(blob)}")
    return residual


def decode_cool_chic_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode (T, H, W, 3) float residual from upsample-cascade pyramid.

    Wire format:
        2B n_levels (uint16 LE)
        per level: 4B scale (float32), then int8 coeffs at H/2^L * W/2^L * 3 per frame.

    Returns float64 (T, H, W, 3); summed across levels with bilinear upsample.
    """
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    (n_levels,) = struct.unpack_from("<H", blob, 0)
    pos = 2
    residual = np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    for L in range(n_levels):
        (scale,) = struct.unpack_from("<f", blob, pos)
        pos += 4
        h_L = CAMERA_H // (2 ** L)
        w_L = CAMERA_W // (2 ** L)
        level_bytes = n_frames * h_L * w_L * RGB_CHANNELS
        if pos + level_bytes > len(blob):
            raise ValueError(f"truncated level {L}")
        arr = np.frombuffer(blob, dtype=np.int8, count=level_bytes, offset=pos).astype(np.float64)
        pos += level_bytes
        arr = arr.reshape(n_frames, h_L, w_L, RGB_CHANNELS) * scale
        if L == 0:
            residual += arr
        else:
            # Bilinear-upsample (h_L, w_L) -> (CAMERA_H, CAMERA_W) per frame per channel.
            t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, 3, h_L, w_L)
            up = F.interpolate(t, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
            residual += up.permute(0, 2, 3, 1).numpy()
    if pos != len(blob):
        raise ValueError(f"trailing bytes after pyramid: pos={pos} total={len(blob)}")
    return residual


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
    pr106_r2_bytes, residual_blob, format_id = parse_residual_archive(blob)
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
    is_sparse = format_id == COOL_CHIC_SPARSE_FORMAT_ID
    decode_fn = decode_cool_chic_residual_sparse if is_sparse else decode_cool_chic_residual
    residual = decode_fn(residual_blob, n_frames)
    print(
        f"[inflate] PR106+cool_chic mode={'sparse' if is_sparse else 'dense'} "
        f"device={device.type} residual_bytes={len(residual_blob)}",
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
