#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate lane_pr106_latent_sidecar archive: PR106 HNeRV decoder + per-pair
latent-correction sidecar.

Reads <src>.bin (lane_pr106_latent_sidecar layout: magic 0xFE + format_id 0x01 +
PR106 bytes verbatim + appended sidecar), reconstructs PR106 state_dict + latents,
applies the per-pair (dim, delta) corrections, runs the HNeRV decoder forward at
384x512, bicubic-upsamples to camera resolution (874x1164), rounds to uint8, and
writes contiguous (N, H, W, 3) bytes to <dst>.

Wire format details: see experiments/build_pr106_latent_sidecar.py.

Sister of submissions/apogee_intN/inflate.py (same PR106 base, different
orthogonal axis: sidecar adds 1.2KB but reduces distortion vs apogee_intN's
bit-width reduction).

Invoked by inflate.sh as:
    python -m submissions.pr106_latent_sidecar.inflate <data_dir>/<base>.bin <output_dir>/<base>.raw
"""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from tac.substrates._shared.inflate_runtime import select_inflate_device as _select_inflate_device_name


CAMERA_H, CAMERA_W = 874, 1164
SIDECAR_MAGIC = 0xFE
SIDECAR_FORMAT_ID = 0x01
DELTA_SCALE = 0.01
NO_OP_DIM = 255
DEFAULT_BATCH_PAIRS = 16


def parse_sidecar_archive(bin_bytes: bytes) -> tuple[bytes, bytes]:
    """Slice apart the (PR106 bytes, sidecar blob) tuple from the wrapper."""
    if not bin_bytes:
        raise ValueError("empty archive")
    if bin_bytes[0] != SIDECAR_MAGIC:
        raise ValueError(
            f"sidecar magic mismatch: got 0x{bin_bytes[0]:02X}, expected 0x{SIDECAR_MAGIC:02X}"
        )
    if bin_bytes[1] != SIDECAR_FORMAT_ID:
        raise ValueError(
            f"sidecar format_id mismatch: got 0x{bin_bytes[1]:02X}, expected 0x{SIDECAR_FORMAT_ID:02X}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    pr106_bytes = bin_bytes[pos : pos + pr106_len]
    pos += pr106_len
    if pos + 2 > len(bin_bytes):
        raise ValueError("sidecar archive truncated before sidecar_len")
    (sidecar_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    sidecar_blob = bin_bytes[pos : pos + sidecar_len]
    pos += sidecar_len
    if pos != len(bin_bytes):
        raise ValueError(
            f"sidecar archive trailing bytes: pos={pos} vs total={len(bin_bytes)}"
        )
    return pr106_bytes, sidecar_blob


def decode_sidecar_corrections(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of build_pr106_latent_sidecar.encode_sidecar_corrections."""
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2 : 2 + 2 * n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]  # uint8 with 255 sentinel
    delta_q = arr[:, 1].view(np.int8)  # signed view of same bytes
    return dim, delta_q


def apply_sidecar_corrections(
    latents: torch.Tensor,
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    scale: float = DELTA_SCALE,
) -> torch.Tensor:
    """In-place add per-pair correction to (n, latent_dim) latents tensor."""
    n = latents.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == NO_OP_DIM:
            continue
        latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * scale
    return latents


def select_inflate_device() -> torch.device:  # INLINE_DEVICE_FORK_OK:helper_delegates_to_canonical_tac_substrates_shared_inflate_runtime_select_inflate_device_helper_which_honors_PACT_INFLATE_DEVICE_env_var_per_comprehensive_bug_audit_cascade_20260526
    """Select an auth-eval-safe inflate device via the vendored shared helper."""
    return torch.device(_select_inflate_device_name())


def select_batch_pairs() -> int:
    """Return the deterministic decoder batch size for pair forwards."""
    raw = os.environ.get("PACT_INFLATE_BATCH_PAIRS")
    if raw is None:
        return DEFAULT_BATCH_PAIRS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"PACT_INFLATE_BATCH_PAIRS must be a positive integer (got {raw!r})"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"PACT_INFLATE_BATCH_PAIRS must be a positive integer (got {value})"
        )
    return value


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()

    pr106_bytes, sidecar_blob = parse_sidecar_archive(archive_bytes)
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)

    if sidecar_blob:
        dim_arr, delta_q_arr = decode_sidecar_corrections(sidecar_blob)
        n_corrections = int((dim_arr != NO_OP_DIM).sum())
        print(
            f"[inflate] sidecar applied: {n_corrections}/{len(dim_arr)} pairs corrected",
            file=sys.stderr,
        )
        apply_sidecar_corrections(latents, dim_arr, delta_q_arr)
    else:
        print("[inflate] sidecar empty (no corrections)", file=sys.stderr)

    try:
        device = select_inflate_device()
        batch_pairs = select_batch_pairs()
    except RuntimeError as exc:
        sys.exit(str(exc))
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
    print(
        f"[inflate] PR106+sidecar: decoder loaded, device={device.type}, "
        f"batch_pairs={batch_pairs}, running {n_pairs} pair forwards...",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, batch_pairs):
            j = min(i + batch_pairs, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.pr106_latent_sidecar_r2.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
