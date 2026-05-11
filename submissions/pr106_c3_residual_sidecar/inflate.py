#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate PR106 + C3-style conditional residual sidecar.

Wire format: magic 0xFD + format_id 0x12 + PR106 bytes + length-prefixed C3
residual blob. The C3 residual is a per-frame-delta encoding (frame[t]
- frame[t-1]) at quarter resolution, INT8-quantised, with per-frame scale
prefix. The decoder integrates: residual[t] = sum_{i<=t} delta_q[i] * scale[i],
then bilinear-upsamples to camera resolution and adds to PR106 decoded output.

Family-scoped: rejects any archive whose format_id ≠ 0x12.

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
C3_FORMAT_ID = 0x12  # dense
C3_SPARSE_FORMAT_ID = 0x22  # sparse PacketIR
CAMERA_H, CAMERA_W = 874, 1164
QUARTER_H, QUARTER_W = CAMERA_H // 4, CAMERA_W // 4
RGB_CHANNELS = 3


def parse_residual_archive(blob: bytes) -> tuple[bytes, bytes, int]:
    """Returns (pr106_bytes, residual_bytes, format_id); accepts dense + sparse."""
    if len(blob) < 6:
        raise ValueError("archive too short")
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ValueError(f"magic 0x{magic:02X}")
    if format_id not in (C3_FORMAT_ID, C3_SPARSE_FORMAT_ID):
        raise ValueError(f"format_id 0x{format_id:02X} not in c3 set")
    pos = 6
    pr106_bytes = blob[pos : pos + pr106_len]
    pos += pr106_len
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    residual_bytes = blob[pos : pos + residual_len]
    if pos + residual_len != len(blob):
        raise ValueError("trailing bytes")
    return bytes(pr106_bytes), bytes(residual_bytes), int(format_id)


def decode_c3_residual_sparse(blob: bytes, n_frames: int) -> np.ndarray:
    """Sparse: temporal-subsampled outer over per-frame (scale + RLE delta-q)."""
    from sparse_packet_ir_inline import (  # type: ignore[import-not-found]
        decode_rle_of_zeros_bytes,
        decode_temporal_subsampled_bytes,
    )
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    items_per_frame = QUARTER_H * QUARTER_W * RGB_CHANNELS
    per_frame_residuals = decode_temporal_subsampled_bytes(blob, dtype=np.uint8)
    deltas = np.zeros((n_frames, QUARTER_H, QUARTER_W, RGB_CHANNELS), dtype=np.float64)
    for t, raw in enumerate(per_frame_residuals):
        if raw is None or t >= n_frames:
            continue
        raw_bytes = raw.tobytes()
        if len(raw_bytes) < 4:
            raise ValueError(f"sparse c3 frame {t}: missing scale")
        (scale,) = struct.unpack_from("<f", raw_bytes, 0)
        flat = decode_rle_of_zeros_bytes(raw_bytes[4:])
        if flat.size != items_per_frame:
            raise ValueError(f"sparse c3 frame {t}: size {flat.size} != {items_per_frame}")
        deltas[t] = flat.reshape(QUARTER_H, QUARTER_W, RGB_CHANNELS).astype(np.float64) * scale
    integrated = np.cumsum(deltas, axis=0)
    t = torch.from_numpy(integrated).permute(0, 3, 1, 2)
    up = F.interpolate(t, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
    return up.permute(0, 2, 3, 1).numpy()


def decode_c3_residual(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode (T, H, W, 3) float residual from frame-delta conditional stream.

    Wire format:
        per frame: 4B scale (float32) + int8 coeffs at QUARTER_H * QUARTER_W * 3.
        Integration: residual_q[t] = sum_{i<=t} delta_q[i] (cumulative sum).

    Returns float64 (T, H, W, 3); bilinear-upsampled from quarter resolution.
    """
    if not blob:
        return np.zeros((n_frames, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    per_frame_bytes = 4 + QUARTER_H * QUARTER_W * RGB_CHANNELS
    expected = n_frames * per_frame_bytes
    if len(blob) != expected:
        raise ValueError(
            f"C3 residual size mismatch: {len(blob)} != expected {expected} "
            f"(n_frames={n_frames}, per_frame={per_frame_bytes})"
        )
    deltas = np.zeros((n_frames, QUARTER_H, QUARTER_W, RGB_CHANNELS), dtype=np.float64)
    pos = 0
    for t in range(n_frames):
        (scale,) = struct.unpack_from("<f", blob, pos)
        pos += 4
        arr = np.frombuffer(
            blob, dtype=np.int8, count=QUARTER_H * QUARTER_W * RGB_CHANNELS, offset=pos
        ).astype(np.float64)
        pos += QUARTER_H * QUARTER_W * RGB_CHANNELS
        deltas[t] = arr.reshape(QUARTER_H, QUARTER_W, RGB_CHANNELS) * scale
    # Integrate (cumulative-sum across time axis).
    integrated = np.cumsum(deltas, axis=0)
    # Bilinear-upsample to camera resolution per frame.
    t = torch.from_numpy(integrated).permute(0, 3, 1, 2)
    up = F.interpolate(t, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
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
    is_sparse = format_id == C3_SPARSE_FORMAT_ID
    decode_fn = decode_c3_residual_sparse if is_sparse else decode_c3_residual
    residual = decode_fn(residual_blob, n_frames)
    print(
        f"[inflate] PR106+c3 mode={'sparse' if is_sparse else 'dense'} "
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
