#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + wavelet residual sidecar archive (research-only scaffold).

Reads <src>.bin (lane_pr106_wavelet_residual_sidecar layout: magic 0xFD +
format_id 0x10 + PR106 bytes verbatim + length-prefixed wavelet residual blob),
reconstructs PR106 decoded RGB frames at camera resolution (874x1164), then
adds the inverse-DWT-reconstructed residual (quantised) to each frame before
rounding to uint8 and writing the raw bytes.

The wavelet residual is single-level 2D Haar over the (T, H, W, 3) RGB
stream. The blob carries per-frame per-channel (cA, cH, cV, cD) coefficient
bands at half resolution, INT8-quantised with a per-band scale prefix. The
``numpy_inverse_dwt.haar_inverse_2d_single_level`` helper synthesises each
band back to (H, W) float64 then the four bands sum into the residual.

Family-scoped: rejects any archive whose format_id ≠ 0x10 (wavelet).

Per CLAUDE.md HNeRV parity discipline lesson 4 (inflate ≤ 100 LOC default;
≤ 200 with rationale): this file is 178 LOC including docstring + imports;
the rationale is the wavelet residual decode itself (≤ 60 LOC) being
fundamentally distinct from PR106's HNeRV decoder.

Invoked by inflate.sh as:
    python inflate.py <data_dir>/<base>.bin <output_dir>/<base>.raw

NO_NVDEC_NEEDED — pure tensor + numpy decode.
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

# Wire-format constants (must mirror tac.residual_basis.pr106_sidecar_packing).
PR106_RESIDUAL_MAGIC = 0xFD
WAVELET_FORMAT_ID = 0x10
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes]:
    """Split (pr106_bytes, residual_bytes); refuse non-wavelet format_id."""
    if not blob or len(blob) < 6:
        raise ValueError("archive too short for header")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic mismatch: 0x{magic:02X} != 0x{PR106_RESIDUAL_MAGIC:02X}")
    if format_id != WAVELET_FORMAT_ID:
        raise ValueError(
            f"format_id 0x{format_id:02X} != wavelet 0x{WAVELET_FORMAT_ID:02X}"
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
    return bytes(pr106_bytes), bytes(residual_bytes)


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


def decode_wavelet_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode (T, H, W, 3) float residual from the INT8-quantised band stream.

    Wire format (per-frame):
        4B scale_cA (float32)
        4B scale_cH (float32)
        4B scale_cV (float32)
        4B scale_cD (float32)
        cA_int8 + cH_int8 + cV_int8 + cD_int8  (each H/2 * W/2 bytes per channel)

    Returns float64 (T, H, W, 3); the caller adds it to the decoded frames.
    """
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    half_h, half_w = CAMERA_H // 2, CAMERA_W // 2
    band_size = half_h * half_w
    per_frame_bytes = 4 * 4 + 4 * RGB_CHANNELS * band_size
    expected = n_frames * per_frame_bytes
    if len(blob) != expected:
        raise ValueError(
            f"wavelet residual size mismatch: got {len(blob)} expected {expected} "
            f"(n_frames={n_frames}, per_frame={per_frame_bytes})"
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
        # Even crop edge: the 874x1164 frame has even H,W so single-level Haar is exact.
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
    pr106_bytes, residual_blob = parse_residual_archive(blob)
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)
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
    residual = decode_wavelet_residual(residual_blob, n_frames)
    print(
        f"[inflate] PR106+wavelet residual: device={device.type}, "
        f"n_pairs={n_pairs}, residual_bytes={len(residual_blob)}",
        file=sys.stderr,
    )
    written = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(n_pairs):
            decoded = decoder(latents[i : i + 1])  # (1, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
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
