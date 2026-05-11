#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + wavelet residual sidecar archive (research-only scaffold).

Wire format: magic 0xFD + format_id (0x10 dense | 0x20 sparse) + PR106 bytes
+ length-prefixed wavelet residual blob. Single-level 2D Haar over (T,H,W,3)
RGB at half camera resolution; inverse DWT in numpy.

Per CLAUDE.md HNeRV parity discipline lesson 4 (inflate ≤ 200 LOC waiver,
``lane_class=substrate_engineering``): family-scoped (refuses non-wavelet
format_ids); dense + sparse PacketIR dispatched on format_id byte.

NO_NVDEC_NEEDED — pure tensor + numpy decode. Invoked as
``python inflate.py <data_dir>/<base>.bin <output_dir>/<base>.raw``.
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
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from pr106_inner_sidecar import (  # type: ignore[import-not-found]
    apply_pr106_r2_sidecar_corrections,
    unwrap_pr106_r2_sidecar,
)

# Wire-format constants (must mirror tac.residual_basis.pr106_sidecar_packing).
PR106_RESIDUAL_MAGIC = 0xFD
WAVELET_FORMAT_ID = 0x10  # dense
WAVELET_SPARSE_FORMAT_ID = 0x20  # sparse PacketIR (RLE+temporal)
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes, int]:
    """Split (pr106_bytes, residual_bytes, format_id). Accepts dense OR sparse."""
    if not blob or len(blob) < 6:
        raise ValueError("archive too short for header")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic mismatch: 0x{magic:02X} != 0x{PR106_RESIDUAL_MAGIC:02X}")
    if format_id not in (WAVELET_FORMAT_ID, WAVELET_SPARSE_FORMAT_ID):
        raise ValueError(
            f"format_id 0x{format_id:02X} not in wavelet set "
            f"{{0x{WAVELET_FORMAT_ID:02X} dense, 0x{WAVELET_SPARSE_FORMAT_ID:02X} sparse}}"
        )
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    residual_bytes = blob[pos : pos + residual_len]
    pos += residual_len
    if pos != len(blob):
        raise ValueError(f"trailing bytes: pos={pos} total={len(blob)}")
    return bytes(pr106_bytes), bytes(residual_bytes), int(format_id)


def haar_inverse_2d_single_level(
    cA: np.ndarray, cH: np.ndarray, cV: np.ndarray, cD: np.ndarray
) -> np.ndarray:
    """Numpy-only single-level 2D Haar synthesis (orthonormal). Returns (2H, 2W)."""
    h, w = cA.shape
    out = np.empty((2 * h, 2 * w), dtype=cA.dtype)
    out[0::2, 0::2] = 0.5 * (cA + cH + cV + cD)
    out[0::2, 1::2] = 0.5 * (cA + cH - cV - cD)
    out[1::2, 0::2] = 0.5 * (cA - cH + cV - cD)
    out[1::2, 1::2] = 0.5 * (cA - cH - cV + cD)
    return out


def decode_wavelet_residual_sparse(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode sparse-PacketIR wavelet residual (temporal-subsampled + RLE inner)."""
    from sparse_packet_ir_inline import (  # type: ignore[import-not-found]
        decode_rle_of_zeros_bytes,
        decode_temporal_subsampled_bytes,
    )
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    half_h, half_w = CAMERA_H // 2, CAMERA_W // 2
    bands_per_frame = RGB_CHANNELS * half_h * half_w
    per_frame_residuals = decode_temporal_subsampled_bytes(blob, dtype=np.uint8)
    residual = np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    for t, raw in enumerate(per_frame_residuals):
        if raw is None or t >= n_frames:
            continue
        raw_bytes = raw.tobytes()
        if len(raw_bytes) < 16:
            raise ValueError(f"sparse wavelet frame {t}: payload < 16 (scales)")
        scales = struct.unpack_from("<4f", raw_bytes, 0)
        flat = decode_rle_of_zeros_bytes(raw_bytes[16:])
        if flat.size != 4 * bands_per_frame:
            raise ValueError(f"sparse wavelet frame {t}: bands size mismatch")
        bands = flat.reshape(4, RGB_CHANNELS, half_h, half_w).astype(np.float64)
        for c in range(RGB_CHANNELS):
            residual[t, : 2 * half_h, : 2 * half_w, c] = haar_inverse_2d_single_level(
                bands[0, c] * scales[0], bands[1, c] * scales[1],
                bands[2, c] * scales[2], bands[3, c] * scales[3],
            )
    return residual


def decode_wavelet_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Dense decode: per-frame 4x4B scales + 4x(C*H/2*W/2) int8 band coefficients."""
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    half_h, half_w = CAMERA_H // 2, CAMERA_W // 2
    band_size = half_h * half_w
    per_frame_bytes = 16 + 4 * RGB_CHANNELS * band_size
    if len(blob) != n_frames * per_frame_bytes:
        raise ValueError(
            f"wavelet residual size {len(blob)} != n_frames*per_frame {n_frames*per_frame_bytes}"
        )
    residual = np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    pos = 0
    for t in range(n_frames):
        scales = struct.unpack_from("<4f", blob, pos)
        pos += 16
        bands = []
        for _ in range(4):
            arr = np.frombuffer(blob, dtype=np.int8, count=RGB_CHANNELS * band_size, offset=pos)
            pos += RGB_CHANNELS * band_size
            bands.append(arr.reshape(RGB_CHANNELS, half_h, half_w).astype(np.float64))
        cA, cH, cV, cD = (b * s for b, s in zip(bands, scales))
        for c in range(RGB_CHANNELS):
            residual[t, :2 * half_h, :2 * half_w, c] = haar_inverse_2d_single_level(
                cA[c], cH[c], cV[c], cD[c]
            )
    return residual


def select_inflate_device() -> torch.device:
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError("PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
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
    is_sparse = format_id == WAVELET_SPARSE_FORMAT_ID
    decode_fn = decode_wavelet_residual_sparse if is_sparse else decode_wavelet_residual
    residual = decode_fn(residual_blob, n_frames)
    print(
        f"[inflate] PR106+wavelet device={device.type} "
        f"mode={'sparse' if is_sparse else 'dense'} "
        f"n_pairs={n_pairs} residual_bytes={len(residual_blob)}",
        file=sys.stderr,
    )
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i : i + 1]).reshape(2, 3, eval_h, eval_w)
            up = F.interpolate(decoded, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
            out = np.clip(np.round(frames + residual[i * 2 : i * 2 + 2]), 0, 255).astype(np.uint8)
            fout.write(out.tobytes())
            written += 2
    print(f"saved {written} frames")
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
