"""PR106 r2 inner-sidecar unwrap helper (shared across non-HNeRV residual families).

PR106 r2's canonical archive.zip contains a `0.bin` that is itself a sidecar-
wrapped archive (lane_pr106_latent_sidecar layout: magic 0xFE + format_id
0x01 + raw PR106 packed bytes + length-prefixed brotli-compressed sidecar
corrections blob). The 5 non-HNeRV residual families wrap PR106 r2's `0.bin`
verbatim inside their own (0xFD + format_id_0x10..0x14) wrapper, so each
family's inflate.py must perform TWO unwrap layers before calling
``parse_packed_archive`` on the raw PR106 packed bytes:

    blob = read(src.bin)
    pr106_r2_inner, family_residual = parse_residual_archive(blob)   # 0xFD unwrap
    raw_pr106, sidecar_blob = unwrap_pr106_r2_sidecar(pr106_r2_inner)  # 0xFE unwrap
    decoder_sd, latents, meta = parse_packed_archive(raw_pr106)
    apply_pr106_r2_sidecar_corrections(latents, sidecar_blob)  # per-pair latent fix

The sidecar corrections (575 bytes brotli-compressed) carry per-pair (dim,
delta_q) tuples applied with scale 0.01. Skipping them collapses the score
from PR106 r2's [contest-CUDA] 0.20665 anchor toward a worse baseline.

Mirrored from submissions/pr106_latent_sidecar_r2/inflate.py (commit
4db14b4d) to keep family inflate.py under HNeRV parity discipline lesson 4's
200-LOC inflate budget.
"""
from __future__ import annotations

import struct

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

PR106_R2_SIDECAR_MAGIC = 0xFE
PR106_R2_SIDECAR_FORMAT_ID = 0x01
PR106_R2_SIDECAR_DELTA_SCALE = 0.01
PR106_R2_SIDECAR_NO_OP_DIM = 255


def unwrap_pr106_r2_sidecar(blob: bytes) -> tuple[bytes, bytes]:
    """Split (raw_pr106_packed_bytes, sidecar_corrections_blob) from PR106 r2's `0.bin`."""
    if not blob:
        raise ValueError("empty pr106_r2 inner blob")
    if blob[0] != PR106_R2_SIDECAR_MAGIC:
        raise ValueError(
            f"PR106 r2 sidecar magic mismatch: 0x{blob[0]:02X} != 0x{PR106_R2_SIDECAR_MAGIC:02X}"
        )
    if blob[1] != PR106_R2_SIDECAR_FORMAT_ID:
        raise ValueError(
            f"PR106 r2 sidecar format_id mismatch: 0x{blob[1]:02X} "
            f"!= 0x{PR106_R2_SIDECAR_FORMAT_ID:02X}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    raw_pr106 = blob[pos : pos + pr106_len]
    pos += pr106_len
    if pos + 2 > len(blob):
        raise ValueError("PR106 r2 sidecar truncated before sidecar_len")
    (sidecar_len,) = struct.unpack_from("<H", blob, pos)
    pos += 2
    sidecar_blob = blob[pos : pos + sidecar_len]
    pos += sidecar_len
    if pos != len(blob):
        raise ValueError(f"PR106 r2 sidecar trailing bytes: pos={pos} total={len(blob)}")
    return bytes(raw_pr106), bytes(sidecar_blob)


def decode_pr106_r2_sidecar_corrections(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of build_pr106_latent_sidecar.encode_sidecar_corrections."""
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2 : 2 + 2 * n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]  # uint8 with 255 sentinel
    delta_q = arr[:, 1].view(np.int8)  # signed view of same bytes
    return dim, delta_q


def apply_pr106_r2_sidecar_corrections(
    latents: torch.Tensor,
    sidecar_blob: bytes,
    *,
    scale: float = PR106_R2_SIDECAR_DELTA_SCALE,
) -> torch.Tensor:
    """In-place add per-pair correction to (n, latent_dim) latents tensor.

    Returns ``latents`` for chaining. If ``sidecar_blob`` is empty the latents
    pass through unchanged.
    """
    if not sidecar_blob:
        return latents
    dim_arr, delta_q_arr = decode_pr106_r2_sidecar_corrections(sidecar_blob)
    n = latents.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == PR106_R2_SIDECAR_NO_OP_DIM:
            continue
        if p < dim_arr.shape[0]:
            latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * scale
    return latents


__all__ = [
    "PR106_R2_SIDECAR_DELTA_SCALE",
    "PR106_R2_SIDECAR_FORMAT_ID",
    "PR106_R2_SIDECAR_MAGIC",
    "PR106_R2_SIDECAR_NO_OP_DIM",
    "apply_pr106_r2_sidecar_corrections",
    "decode_pr106_r2_sidecar_corrections",
    "unwrap_pr106_r2_sidecar",
]
