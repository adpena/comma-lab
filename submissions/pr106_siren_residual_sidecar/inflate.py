#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + SIREN-style sparse frequency-domain residual sidecar.

SIREN's coordinate-MLP with sinusoidal activations naturally encodes signal as
a sum of frequencies. Rather than embed an MLP runtime (which would inflate
the LOC budget), we store the SPARSE FFT coefficients directly: a small set
of (k_row, k_col, channel, real, imag) tuples with INT16 frequency indices
and INT8 quantised complex amplitudes, plus a per-coefficient float32 scale.

The inflate runtime reconstructs each frame as the inverse-2D-FFT of the
sparse coefficient set. This is the closest byte-closed analog of SIREN's
"smooth low-frequency-dominant residual" assumption.

Wire format: magic 0xFD + format_id 0x13 + PR106 bytes + length-prefixed
SIREN residual blob.

Family-scoped: rejects any archive whose format_id ≠ 0x13.

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
SIREN_FORMAT_ID = 0x13
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic 0x{magic:02X}")
    if format_id != SIREN_FORMAT_ID:
        raise ValueError(f"format_id 0x{format_id:02X} != siren 0x{SIREN_FORMAT_ID:02X}")
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    residual_bytes = blob[pos : pos + residual_len]
    if pos + residual_len != len(blob):
        raise ValueError("trailing bytes")
    return bytes(pr106_bytes), bytes(residual_bytes)


def decode_siren_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode (T, H, W, 3) float residual from sparse FFT coefficients.

    Wire format:
        4B scale (float32) — global amplitude scale for all coefs
        2B n_coefs (uint16 LE)
        per coef:
            2B frame_idx (uint16 LE)
            2B k_row     (int16 LE; freq index; can be negative)
            2B k_col     (int16 LE)
            1B channel   (uint8; 0/1/2)
            1B real_q    (int8)
            1B imag_q    (int8)
        Total per coef = 9B.

    The residual frequency-domain coefficients are inverse-FFT'd per frame
    per channel and the real part is taken.
    """
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    if len(blob) < 6:
        raise ValueError(f"siren header too short: {len(blob)}")
    (scale,) = struct.unpack_from("<f", blob, 0)
    (n_coefs,) = struct.unpack_from("<H", blob, 4)
    expected = 6 + n_coefs * 9
    if len(blob) != expected:
        raise ValueError(f"siren residual size mismatch: {len(blob)} != {expected}")
    # Frequency-domain accumulator (per-frame per-channel COMPLEX).
    spectrum = np.zeros(
        (n_frames, RGB_CHANNELS, CAMERA_H, CAMERA_W), dtype=np.complex128
    )
    pos = 6
    for _ in range(n_coefs):
        frame_idx, k_row, k_col = struct.unpack_from("<HhH", blob, pos)
        # NOTE: format string for k_col should be signed int16; redo correctly:
        frame_idx = struct.unpack_from("<H", blob, pos)[0]
        k_row = struct.unpack_from("<h", blob, pos + 2)[0]
        k_col = struct.unpack_from("<h", blob, pos + 4)[0]
        channel = blob[pos + 6]
        real_q = struct.unpack_from("<b", blob, pos + 7)[0]
        imag_q = struct.unpack_from("<b", blob, pos + 8)[0]
        pos += 9
        if frame_idx >= n_frames or channel >= RGB_CHANNELS:
            raise ValueError(f"coef out of range: frame={frame_idx} channel={channel}")
        # k_row, k_col are signed; map to FFT array indices via modulo.
        row_idx = int(k_row) % CAMERA_H
        col_idx = int(k_col) % CAMERA_W
        spectrum[frame_idx, channel, row_idx, col_idx] = (real_q + 1j * imag_q) * scale
    residual = np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    for t in range(n_frames):
        for c in range(RGB_CHANNELS):
            residual[t, :, :, c] = np.real(np.fft.ifft2(spectrum[t, c]))
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
    residual = decode_siren_residual(residual_blob, n_frames)
    print(
        f"[inflate] PR106+siren residual: device={device.type}, "
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
