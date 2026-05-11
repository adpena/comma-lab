#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + SIREN sparse FFT-coefficient residual sidecar.

Wire format: magic 0xFD + format_id (0x13 dense | 0x23 sparse) + PR106 bytes +
length-prefixed sparse FFT-coef residual blob. Inverse 2D-FFT per frame.

Per CLAUDE.md HNeRV parity lesson 4 (inflate ≤ 200 LOC waiver,
``lane_class=substrate_engineering``). NO_NVDEC_NEEDED.
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
SIREN_FORMAT_ID = 0x13  # dense (already sparse-FFT-style)
SIREN_SPARSE_FORMAT_ID = 0x23  # sparse PacketIR (RLE over (frame,k_row,k_col,ch) idx)
CAMERA_H, CAMERA_W = 874, 1164
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes, int]:
    """Returns (pr106_bytes, residual_bytes, format_id); accepts dense + sparse."""
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic 0x{magic:02X}")
    if format_id not in (SIREN_FORMAT_ID, SIREN_SPARSE_FORMAT_ID):
        raise ValueError(f"format_id 0x{format_id:02X} not in siren set")
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    residual_bytes = blob[pos : pos + residual_len]
    if pos + residual_len != len(blob):
        raise ValueError("trailing bytes")
    return bytes(pr106_bytes), bytes(residual_bytes), int(format_id)


def decode_siren_residual_sparse(blob: bytes, n_frames: int) -> np.ndarray:
    """Sparse: 4B scale + 4B n_coefs + RLE-of-zeros over packed (real_q,imag_q) bytes.

    Reuses the dense format's per-coef (frame,k_row,k_col,channel) addressing
    table written separately, then runs RLE over the int8 (real_q, imag_q)
    coefficient pair sequence (2 bytes per coef). The RLE saves bytes when
    most coefficients quantise to zero (Quantizr/Selfcomp pattern).
    Wire format:
        4B scale (float32)
        4B n_coefs (uint32 LE)
        n_coefs * 7B (frame_idx + k_row + k_col + channel) address table
        RLE-of-zeros blob over the int8 (real_q, imag_q) interleaved stream.
    """
    from sparse_packet_ir_inline import (  # type: ignore[import-not-found]
        decode_rle_of_zeros_bytes,
    )
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    if len(blob) < 8:
        raise ValueError(f"sparse siren header too short: {len(blob)}")
    (scale, n_coefs) = struct.unpack_from("<fI", blob, 0)
    pos = 8
    addr_size = n_coefs * 7
    if pos + addr_size > len(blob):
        raise ValueError(f"sparse siren address table truncated")
    addr_bytes = blob[pos : pos + addr_size]
    pos += addr_size
    rle_blob = blob[pos:]
    coef_stream = decode_rle_of_zeros_bytes(rle_blob)
    if coef_stream.size != 2 * n_coefs:
        raise ValueError(f"sparse siren coef stream size {coef_stream.size} != {2 * n_coefs}")
    spectrum = np.zeros((n_frames, RGB_CHANNELS, CAMERA_H, CAMERA_W), dtype=np.complex128)
    for i in range(n_coefs):
        a_pos = i * 7
        (frame_idx,) = struct.unpack_from("<H", addr_bytes, a_pos)
        (k_row,) = struct.unpack_from("<h", addr_bytes, a_pos + 2)
        (k_col,) = struct.unpack_from("<h", addr_bytes, a_pos + 4)
        channel = addr_bytes[a_pos + 6]
        if frame_idx >= n_frames or channel >= RGB_CHANNELS:
            raise ValueError(f"sparse siren coef out of range: frame={frame_idx} ch={channel}")
        real_q = int(coef_stream[2 * i])
        imag_q = int(coef_stream[2 * i + 1])
        row_idx = int(k_row) % CAMERA_H
        col_idx = int(k_col) % CAMERA_W
        spectrum[frame_idx, channel, row_idx, col_idx] = (real_q + 1j * imag_q) * scale
    residual = np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    for t in range(n_frames):
        for c in range(RGB_CHANNELS):
            residual[t, :, :, c] = np.real(np.fft.ifft2(spectrum[t, c]))
    return residual


def decode_siren_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Dense FFT-coef decode: 4B scale + 2B n_coefs + n_coefs*9B records."""
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    if len(blob) < 6:
        raise ValueError(f"siren header too short: {len(blob)}")
    (scale,) = struct.unpack_from("<f", blob, 0)
    (n_coefs,) = struct.unpack_from("<H", blob, 4)
    if len(blob) != 6 + n_coefs * 9:
        raise ValueError(f"siren size mismatch: {len(blob)} != {6 + n_coefs * 9}")
    spectrum = np.zeros((n_frames, RGB_CHANNELS, CAMERA_H, CAMERA_W), dtype=np.complex128)
    pos = 6
    for _ in range(n_coefs):
        frame_idx, k_row, k_col, channel, real_q, imag_q = struct.unpack_from(
            "<HhhBbb", blob, pos
        )
        pos += 9
        if frame_idx >= n_frames or channel >= RGB_CHANNELS:
            raise ValueError(f"coef out of range: frame={frame_idx} ch={channel}")
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
    is_sparse = format_id == SIREN_SPARSE_FORMAT_ID
    decode_fn = decode_siren_residual_sparse if is_sparse else decode_siren_residual
    residual = decode_fn(residual_blob, n_frames)
    print(
        f"[inflate] PR106+siren mode={'sparse' if is_sparse else 'dense'} "
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
