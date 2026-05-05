#!/usr/bin/env python
"""Inflate pr106_lrl1_sidechannel: PR106 HNeRV decoder + per-frame Local-Region L1
(LRL1) luma low-rank correction sidechannel.

Wire format (single 0.bin in archive.zip):

    Offset  Bytes  Field                                    Notes
    ────────────────────────────────────────────────────────────────────────
    0       1      magic 0xFB                               lrl1 dispatch byte
    1       3      pr106_len (uint24 LE)                    bytes of inner PR106 archive
    4       N      pr106 packed archive (raw bytes)         starts with 0xFF magic
    4+N     1      sidechannel magic 0x1                    1 = LR01 v1
    5+N     2      sidechannel_len (uint16 LE)              brotli'd LR01 payload size
    7+N     M      brotli(LR01 sidechannel payload)         decompresses to header + raw

Inner LR01 payload (post-brotli-decompress) — mirrors codex_metric LRL1 mode 8:

    Offset  Bytes  Field                                    Notes
    ────────────────────────────────────────────────────────────────────────
    0       4      magic b"LR01"
    4       1      mode_id (= 8 = MODE_LRL1)
    5       1      K (coefficient count per frame, 1..255)
    6       2      low_h (uint16 LE)                        basis height
    8       2      low_w (uint16 LE)                        basis width
    10      4      n_frames (uint32 LE)                     number of frames covered
    14      4      coeff_step (float32 LE)                  scale for per-frame coefs
    18      4      basis_step (float32 LE)                  scale for basis values
    22      K*low_h*low_w  int8 basis stream                row-major (K, low_h, low_w)
    22+B    n_frames*K     int8 coefficient stream          row-major (n_frames, K)

At inflate time:
    1. Parse PR106 packed archive → decoder weights + latents.
    2. Forward decoder pair-by-pair → 384x512 RGB pair.
    3. Bicubic upsample → 874x1164.
    4. Round to uint8.
    5. Apply LRL1 correction per frame:
         a. Bilinear-upsample low-res basis (K, low_h, low_w) → (K, H, W)
            (cached across frames; done once per inflate).
         b. correction = einsum("k,khw->hw", coeffs[frame_idx]*coeff_step,
                                basis*basis_step)
         c. Add correction uniformly to R, G, B channels (luma broadcast).
         d. Clip to [0, 255], round to uint8.
    6. Write contiguous (N, H, W, 3) bytes.

CUDA-required (CLAUDE.md MPS-auth-eval-is-NOISE non-negotiable).

Sister of submissions/pr106_yshift_sidechannel/inflate.py and
submissions/apogee_intN/inflate.py; uses the SAME PR106 codec source
(submissions/apogee_intN/src/codec.py + model.py) — pr106_lrl1_sidechannel
is purely an additive luma-correction sidechannel on top of the PR106 packed
archive, the variant #6 of the score-aware sidechannel paradigm thread (see
docs/codex_metric_lrl1_audit_20260504.md).

Stack-on relationship: gated on lane_pr106_latent_sidecar (variant #1)
AND lane_pr106_yshift_sidechannel (variant #3) BOTH winning empirically
first; LRL1 is the natural 3rd stack-on per the paradigm decision pipeline.
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Reuse PR106 codec + model from sister submission to avoid duplication.
APOGEE_SRC = HERE.parent / "apogee_intN" / "src"
sys.path.insert(0, str(APOGEE_SRC))

from model import HNeRVDecoder  # type: ignore[import-not-found]
from codec import parse_packed_archive  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164

LRL1_MAGIC_BYTE = 0xFB  # outer dispatch byte for pr106_lrl1_sidechannel
SIDECHANNEL_VERSION = 1  # post-PR106-len, before LR01 brotli'd payload

LR01_MAGIC = b"LR01"
# magic(4) + mode_id(1) + K(1) + low_h(2) + low_w(2) + n_frames(4) +
# coeff_step(4) + basis_step(4) = 22 bytes
LR01_HEADER = struct.Struct("<4sBBHHIff")
SIDECHANNEL_MODE_LRL1 = 8  # mirrors codex_metric LRL1


def parse_lrl1_archive(archive_bytes: bytes) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict, dict | None]:
    """Parse pr106_lrl1_sidechannel 0.bin layout.

    Returns (state_dict, latents, meta, sidechannel_or_None).
    """
    if not archive_bytes:
        raise ValueError("empty archive")
    magic = archive_bytes[0]
    if magic != LRL1_MAGIC_BYTE:
        raise ValueError(
            f"pr106_lrl1 magic mismatch: got 0x{magic:02X}, expected 0x{LRL1_MAGIC_BYTE:02X}"
        )
    pr106_len = int.from_bytes(archive_bytes[1:4], "little")
    pr106_end = 4 + pr106_len
    if pr106_end > len(archive_bytes):
        raise ValueError(f"pr106_len {pr106_len} exceeds archive size {len(archive_bytes)}")
    pr106_bytes = archive_bytes[4:pr106_end]
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)

    sidechannel = None
    if pr106_end < len(archive_bytes):
        sc_version = archive_bytes[pr106_end]
        if sc_version != SIDECHANNEL_VERSION:
            raise ValueError(
                f"sidechannel version mismatch: got {sc_version}, expected {SIDECHANNEL_VERSION}"
            )
        sc_len = struct.unpack_from("<H", archive_bytes, pr106_end + 1)[0]
        sc_payload_start = pr106_end + 3
        sc_payload_end = sc_payload_start + sc_len
        if sc_payload_end != len(archive_bytes):
            raise ValueError(
                f"sidechannel trailing bytes: expected_end={sc_payload_end}, total={len(archive_bytes)}"
            )
        sidechannel = decode_sidechannel_blob(archive_bytes[sc_payload_start:sc_payload_end])
    return decoder_sd, latents, meta, sidechannel


def decode_sidechannel_blob(blob: bytes) -> dict:
    """Brotli-decompress + parse LR01 payload.

    Returns dict with keys: mode_id, K, low_h, low_w, n_frames, coeff_step,
    basis_step, basis (K, low_h, low_w) int8, coeffs (n_frames, K) int8.
    """
    raw = brotli.decompress(blob)
    if len(raw) < LR01_HEADER.size:
        raise ValueError(f"sidechannel blob too small: {len(raw)} < {LR01_HEADER.size}")
    magic, mode_id, K, low_h, low_w, n_frames, coeff_step, basis_step = LR01_HEADER.unpack_from(raw)
    if magic != LR01_MAGIC:
        raise ValueError(f"bad LR01 magic: {magic!r}")
    if mode_id != SIDECHANNEL_MODE_LRL1:
        raise ValueError(f"unsupported sidechannel mode_id={mode_id}, expected {SIDECHANNEL_MODE_LRL1}")
    if K < 1:
        raise ValueError(f"LRL1 expects K >= 1, got {K}")
    if low_h < 1 or low_w < 1:
        raise ValueError(f"LRL1 expects low_h,low_w >= 1, got {low_h}x{low_w}")
    basis_size = int(K) * int(low_h) * int(low_w)
    coeff_size = int(n_frames) * int(K)
    expected = LR01_HEADER.size + basis_size + coeff_size
    if len(raw) != expected:
        raise ValueError(
            f"bad LR01 length: expected {expected} "
            f"(header={LR01_HEADER.size} + basis={basis_size} + coeffs={coeff_size}), "
            f"got {len(raw)}"
        )
    basis_bytes = raw[LR01_HEADER.size:LR01_HEADER.size + basis_size]
    coeff_bytes = raw[LR01_HEADER.size + basis_size:]
    basis_int = np.frombuffer(basis_bytes, dtype=np.int8).copy().reshape(int(K), int(low_h), int(low_w))
    coeff_int = np.frombuffer(coeff_bytes, dtype=np.int8).copy().reshape(int(n_frames), int(K))
    return {
        "mode_id": int(mode_id),
        "K": int(K),
        "low_h": int(low_h),
        "low_w": int(low_w),
        "n_frames": int(n_frames),
        "coeff_step": float(coeff_step),
        "basis_step": float(basis_step),
        "basis": basis_int,    # (K, low_h, low_w) int8
        "coeffs": coeff_int,   # (n_frames, K) int8
    }


def upsample_basis(
    basis_int8: np.ndarray, basis_step: float, target_h: int, target_w: int,
    *, device: torch.device | None = None,
) -> torch.Tensor:
    """Bilinear-upsample (K, low_h, low_w) int8 basis to (K, target_h, target_w) float.

    Output is float32 with the basis_step scale already applied. This is computed
    once at inflate-load time and cached for all frames (codex pattern).
    """
    K, low_h, low_w = basis_int8.shape
    basis_f = torch.from_numpy(basis_int8.astype(np.float32) * float(basis_step))
    if device is not None:
        basis_f = basis_f.to(device)
    # F.interpolate wants (N, C, H, W) — wrap as (1, K, low_h, low_w)
    basis_4d = basis_f.unsqueeze(0)
    up = F.interpolate(basis_4d, size=(target_h, target_w),
                       mode="bilinear", align_corners=False)
    return up.squeeze(0)  # (K, H, W)


def apply_lrl1_to_frame(
    frame_u8: np.ndarray,
    upsampled_basis: torch.Tensor,
    coeffs_int8: np.ndarray,
    coeff_step: float,
) -> np.ndarray:
    """Apply per-frame LRL1 correction to a (H, W, 3) uint8 frame.

    correction(h,w) = sum_k coeffs[k]*coeff_step * upsampled_basis[k, h, w]
    out = clip(frame + correction[..., None], 0, 255).round().uint8

    Codex pattern: same correction broadcast across R, G, B (luma-only mechanism
    expressed as an isotropic RGB shift; effective Y component dominates because
    the scorer's PoseNet/SegNet front-end is luma-sensitive).
    """
    if frame_u8.ndim != 3 or frame_u8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8 frame, got shape {frame_u8.shape}")
    if coeffs_int8.shape != (upsampled_basis.shape[0],):
        raise ValueError(
            f"coeffs shape {coeffs_int8.shape} doesn't match basis K={upsampled_basis.shape[0]}"
        )
    coeffs_f = torch.from_numpy(coeffs_int8.astype(np.float32) * float(coeff_step))
    coeffs_f = coeffs_f.to(upsampled_basis.device)
    # einsum("k,khw->hw") — project per-frame coefs onto basis
    correction = torch.einsum("k,khw->hw", coeffs_f, upsampled_basis)  # (H, W)
    correction_np = correction.detach().cpu().numpy()
    out = frame_u8.astype(np.float32) + correction_np[..., None]
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta, sidechannel = parse_lrl1_archive(archive_bytes)

    if not torch.cuda.is_available():
        sys.exit("pr106_lrl1_sidechannel inflate requires GPU (per CLAUDE.md MPS-auth-eval-is-NOISE)")
    device = torch.device("cuda")
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
    expected_frames = n_pairs * 2

    # One-time basis upsample (cached across all frames)
    upsampled_basis: torch.Tensor | None = None
    sc_coeffs: np.ndarray | None = None
    sc_coeff_step: float = 0.0
    if sidechannel is not None:
        if sidechannel["n_frames"] != expected_frames:
            raise ValueError(
                f"sidechannel frame count mismatch: LR01 has {sidechannel['n_frames']}, "
                f"decoder produces {expected_frames}"
            )
        upsampled_basis = upsample_basis(
            sidechannel["basis"], sidechannel["basis_step"],
            CAMERA_H, CAMERA_W, device=device,
        )
        sc_coeffs = sidechannel["coeffs"]
        sc_coeff_step = sidechannel["coeff_step"]
    print(
        f"pr106_lrl1_sidechannel: decoder loaded; sidechannel="
        f"{'present' if sidechannel else 'ABSENT'}, "
        f"K={sidechannel['K'] if sidechannel else 0}, "
        f"basis={'%dx%d' % (sidechannel['low_h'], sidechannel['low_w']) if sidechannel else 'n/a'}, "
        f"n_pairs={n_pairs}",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )  # (B*2, H, W, 3) uint8
            if upsampled_basis is not None and sc_coeffs is not None:
                for k in range(B * 2):
                    frame_idx = i * 2 + k
                    frames[k] = apply_lrl1_to_frame(
                        frames[k], upsampled_basis, sc_coeffs[frame_idx], sc_coeff_step,
                    )
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.pr106_lrl1_sidechannel.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
