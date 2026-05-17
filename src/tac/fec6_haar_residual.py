# SPDX-License-Identifier: MIT
"""fec6 + Haar 1-level wavelet residual codec (Ext 5).

This module is the canonical encoder/decoder for the fec6+haar-residual
outer-wrapper extension (lane
``lane_fec6_stacking_wave_5_grammar_extensions_20260517``).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
lesson 7 (bolt-on vs substrate-engineering split): this is a BOLT-ON
extension to the fec6 outer wrapper.

The fec6 outer grammar is extended with a NEW trailing slot after any
prior format0d-EXTRA slot::

    [existing fec6 outer wrapper] | EXTRA_MAGIC=FE6W | u32 wavelet_payload_len | wavelet_payload

where ``wavelet_payload`` carries a 1-level Haar wavelet decomposition
of a per-frame additive correction.

Wire format of ``wavelet_payload``::

    u16 n_frames | u16 h_ll | u16 w_ll | u16 frame_h | u16 frame_w
  | fp16 scale_ll | fp16 scale_lh | fp16 scale_hl | fp16 scale_hh
  | i8  ll_quant[n_frames * h_ll * w_ll]   (LL band; lowpass-lowpass)
  | i8  lh_quant[n_frames * h_ll * w_ll]   (LH band; lowpass-highpass)
  | i8  hl_quant[n_frames * h_ll * w_ll]   (HL band; highpass-lowpass)
  | i8  hh_quant[n_frames * h_ll * w_ll]   (HH band; highpass-highpass)

where ``(h_ll, w_ll) = (residual_h // 2, residual_w // 2)`` and the
residual is computed at ``(residual_h, residual_w)`` typically
downsampled from full camera resolution ``(frame_h, frame_w)``.

At inflate time:
- Decode wavelet_payload
- For each frame, inverse-Haar to reconstruct the residual at
  ``(residual_h, residual_w)``
- Bilinear upsample residual to ``(frame_h, frame_w)``
- Add to the fec6-rendered RGB frame (clip to [0, 255], round to uint8)

Per CLAUDE.md "Bit-level deconstruction and entropy discipline":
total uncompressed size = 18 (header) + 4*n_frames*h_ll*w_ll (4 bands × n_frames frames × spatial extent)
At n_frames=600, h_ll=48, w_ll=64, total = 18 + 4*600*48*64 = ~7.4 MB uncompressed.
With brotli quality 11 + per-band fp16 quantization, expected ~3-5 KB after compression
because most band coefficients are near-zero after fec6 corrections.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
predicted ΔS band ``[-0.0005, -0.0020] [predicted, theoretical]`` on
contest-CPU axis.

Design memo: ``.omx/research/fec6_plus_haar_residual_design_20260517.md``
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

__all__ = [
    "WAVELET_MAGIC",
    "HaarResidualDecodeError",
    "HaarResidualEncodeError",
    "decode_haar_residual_payload",
    "dequantize_per_band",
    "encode_haar_residual_payload",
    "haar_forward_1level",
    "haar_inverse_1level",
    "quantize_per_band",
    "unwrap_fec6_archive_with_haar",
    "wrap_fec6_archive_with_haar",
]


WAVELET_MAGIC: bytes = b"FE6W"


class HaarResidualEncodeError(ValueError):
    """Raised when the Haar residual encoder cannot produce a valid payload."""


class HaarResidualDecodeError(ValueError):
    """Raised when the Haar residual decoder rejects a payload as malformed."""


def haar_forward_1level(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1-level orthonormal Haar wavelet transform of a 2-D frame.

    Returns (LL, LH, HL, HH) each of shape (H//2, W//2). Input must have
    even H and W. Uses the orthonormal Haar basis ((a+b)/√2, (a-b)/√2).
    """
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2-D; got shape {frame.shape}")
    h, w = frame.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"frame H={h} and W={w} must both be even")
    f = frame.astype(np.float32, copy=False)
    sqrt2 = np.float32(np.sqrt(2.0))
    # Row-wise transform: average + difference per consecutive pair.
    row_avg = (f[:, 0::2] + f[:, 1::2]) / sqrt2  # (H, W/2)
    row_dif = (f[:, 0::2] - f[:, 1::2]) / sqrt2  # (H, W/2)
    # Column-wise transform on each half.
    ll = (row_avg[0::2, :] + row_avg[1::2, :]) / sqrt2  # (H/2, W/2)
    lh = (row_avg[0::2, :] - row_avg[1::2, :]) / sqrt2  # (H/2, W/2)
    hl = (row_dif[0::2, :] + row_dif[1::2, :]) / sqrt2  # (H/2, W/2)
    hh = (row_dif[0::2, :] - row_dif[1::2, :]) / sqrt2  # (H/2, W/2)
    return ll, lh, hl, hh


def haar_inverse_1level(
    ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray
) -> np.ndarray:
    """Inverse 1-level Haar transform. Inputs each shape (H/2, W/2)."""
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError(
            f"Haar band shapes must match: ll={ll.shape}, lh={lh.shape}, "
            f"hl={hl.shape}, hh={hh.shape}"
        )
    if ll.ndim != 2:
        raise ValueError(f"Haar bands must be 2-D; got {ll.shape}")
    sqrt2 = np.float32(np.sqrt(2.0))
    h_ll, w_ll = ll.shape
    # Inverse column transform.
    row_avg = np.empty((2 * h_ll, w_ll), dtype=np.float32)
    row_avg[0::2, :] = (ll + lh) / sqrt2
    row_avg[1::2, :] = (ll - lh) / sqrt2
    row_dif = np.empty((2 * h_ll, w_ll), dtype=np.float32)
    row_dif[0::2, :] = (hl + hh) / sqrt2
    row_dif[1::2, :] = (hl - hh) / sqrt2
    # Inverse row transform.
    out = np.empty((2 * h_ll, 2 * w_ll), dtype=np.float32)
    out[:, 0::2] = (row_avg + row_dif) / sqrt2
    out[:, 1::2] = (row_avg - row_dif) / sqrt2
    return out


def quantize_per_band(band: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize a per-band Haar coefficient tensor to int8 with fp16 scale.

    Returns ``(quantized_int8, scale_fp16)`` where ``quantized * scale ≈ band``.
    """
    if band.size == 0:
        raise HaarResidualEncodeError("cannot quantize empty band")
    max_abs = float(np.abs(band).max())
    if max_abs == 0.0:
        # Pathological all-zero band; encode with scale=1.0 to avoid divide-by-zero
        return np.zeros_like(band, dtype=np.int8), 1.0
    # Scale so the most-extreme value lands at int8 max (~127).
    scale_target = max_abs / 127.0
    # Round to fp16 so encode/decode round-trip is byte-deterministic.
    scale_f16 = float(np.float16(scale_target))
    if scale_f16 <= 0.0 or not np.isfinite(scale_f16):
        raise HaarResidualEncodeError(
            f"scale_f16={scale_f16} is non-finite or non-positive (max_abs={max_abs})"
        )
    quant = np.clip(np.round(band / scale_f16), -127, 127).astype(np.int8)
    return quant, scale_f16


def dequantize_per_band(quant: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of quantize_per_band: int8 + fp16 scale → float32."""
    return quant.astype(np.float32) * np.float32(scale)


def encode_haar_residual_payload(
    *,
    residuals: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> bytes:
    """Encode per-frame residuals into the byte-deterministic Haar payload.

    Parameters
    ----------
    residuals : np.ndarray, shape (n_frames, residual_h, residual_w)
        Per-frame residual computed at the downsampled resolution. Will be
        Haar-transformed and per-band-quantized.
    frame_h, frame_w : int
        Full camera-resolution dimensions (for inflate-time upsampling).

    Returns
    -------
    bytes
        Byte-deterministic payload (without outer WAVELET_MAGIC or u32 length prefix).
    """
    if residuals.ndim != 3:
        raise HaarResidualEncodeError(
            f"residuals must be 3-D (n_frames, H, W); got shape {residuals.shape}"
        )
    n_frames, residual_h, residual_w = residuals.shape
    if residual_h % 2 != 0 or residual_w % 2 != 0:
        raise HaarResidualEncodeError(
            f"residual H={residual_h} and W={residual_w} must both be even"
        )
    h_ll = residual_h // 2
    w_ll = residual_w // 2

    # Compute Haar transform per frame; stack each band across frames.
    ll_all = np.empty((n_frames, h_ll, w_ll), dtype=np.float32)
    lh_all = np.empty((n_frames, h_ll, w_ll), dtype=np.float32)
    hl_all = np.empty((n_frames, h_ll, w_ll), dtype=np.float32)
    hh_all = np.empty((n_frames, h_ll, w_ll), dtype=np.float32)
    for i in range(n_frames):
        ll, lh, hl, hh = haar_forward_1level(residuals[i])
        ll_all[i] = ll
        lh_all[i] = lh
        hl_all[i] = hl
        hh_all[i] = hh

    # Quantize each band globally (one scale per band across all frames).
    ll_q, scale_ll = quantize_per_band(ll_all)
    lh_q, scale_lh = quantize_per_band(lh_all)
    hl_q, scale_hl = quantize_per_band(hl_all)
    hh_q, scale_hh = quantize_per_band(hh_all)

    # Header: u16 n_frames | u16 h_ll | u16 w_ll | u16 frame_h | u16 frame_w
    if n_frames > 2**16 - 1:
        raise HaarResidualEncodeError(f"n_frames={n_frames} exceeds u16 max")
    if h_ll > 2**16 - 1 or w_ll > 2**16 - 1:
        raise HaarResidualEncodeError(
            f"h_ll={h_ll} or w_ll={w_ll} exceeds u16 max"
        )
    if frame_h > 2**16 - 1 or frame_w > 2**16 - 1:
        raise HaarResidualEncodeError(
            f"frame_h={frame_h} or frame_w={frame_w} exceeds u16 max"
        )

    header = struct.pack("<HHHHH", n_frames, h_ll, w_ll, frame_h, frame_w)
    scales = (
        np.float16(scale_ll).tobytes()
        + np.float16(scale_lh).tobytes()
        + np.float16(scale_hl).tobytes()
        + np.float16(scale_hh).tobytes()
    )
    band_bytes = (
        ll_q.tobytes() + lh_q.tobytes() + hl_q.tobytes() + hh_q.tobytes()
    )
    return header + scales + band_bytes


def decode_haar_residual_payload(
    payload: bytes,
) -> dict[str, Any]:
    """Decode the byte-deterministic Haar payload back into per-frame bands.

    Returns dict with keys:
        n_frames, h_ll, w_ll, frame_h, frame_w
        scale_ll, scale_lh, scale_hl, scale_hh
        ll_quant, lh_quant, hl_quant, hh_quant  (np.ndarray int8 shape (n_frames, h_ll, w_ll))
    """
    if len(payload) < 18:
        raise HaarResidualDecodeError(
            f"payload too short for header+scales (need >= 18 bytes); got {len(payload)}"
        )
    n_frames, h_ll, w_ll, frame_h, frame_w = struct.unpack_from("<HHHHH", payload, 0)
    scale_ll = float(np.frombuffer(payload, dtype=np.float16, count=1, offset=10)[0])
    scale_lh = float(np.frombuffer(payload, dtype=np.float16, count=1, offset=12)[0])
    scale_hl = float(np.frombuffer(payload, dtype=np.float16, count=1, offset=14)[0])
    scale_hh = float(np.frombuffer(payload, dtype=np.float16, count=1, offset=16)[0])
    band_size = n_frames * h_ll * w_ll
    expected_size = 18 + 4 * band_size
    if len(payload) != expected_size:
        raise HaarResidualDecodeError(
            f"payload size mismatch: got {len(payload)} expected {expected_size} "
            f"for n_frames={n_frames}, h_ll={h_ll}, w_ll={w_ll}"
        )
    off = 18
    ll_q = np.frombuffer(payload, dtype=np.int8, count=band_size, offset=off).reshape(n_frames, h_ll, w_ll).copy()
    off += band_size
    lh_q = np.frombuffer(payload, dtype=np.int8, count=band_size, offset=off).reshape(n_frames, h_ll, w_ll).copy()
    off += band_size
    hl_q = np.frombuffer(payload, dtype=np.int8, count=band_size, offset=off).reshape(n_frames, h_ll, w_ll).copy()
    off += band_size
    hh_q = np.frombuffer(payload, dtype=np.int8, count=band_size, offset=off).reshape(n_frames, h_ll, w_ll).copy()

    return {
        "n_frames": n_frames,
        "h_ll": h_ll,
        "w_ll": w_ll,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "scale_ll": scale_ll,
        "scale_lh": scale_lh,
        "scale_hl": scale_hl,
        "scale_hh": scale_hh,
        "ll_quant": ll_q,
        "lh_quant": lh_q,
        "hl_quant": hl_q,
        "hh_quant": hh_q,
    }


def wrap_fec6_archive_with_haar(
    *,
    fec6_archive_bytes: bytes,
    haar_payload: bytes,
) -> bytes:
    """Append the Haar residual slot to existing fec6 archive inner bytes.

    The wrapped bytes are::

        <fec6_archive_bytes> | WAVELET_MAGIC | u32 haar_payload_len | haar_payload

    The u32 length (vs u16 for format0d-EXTRA) accommodates the larger
    wavelet payload (can exceed 64 KB for full-resolution residuals).
    """
    if len(haar_payload) > 2**32 - 1:
        raise HaarResidualEncodeError(
            f"haar_payload len {len(haar_payload)} exceeds u32 max"
        )
    return (
        fec6_archive_bytes
        + WAVELET_MAGIC
        + struct.pack("<I", len(haar_payload))
        + haar_payload
    )


def unwrap_fec6_archive_with_haar(
    fec6_archive_bytes: bytes,
) -> tuple[bytes, bytes | None]:
    """Split fec6 inner-member bytes into (base_fec6, haar_payload-or-None).

    Mirrors ``unwrap_fec6_archive_with_extra`` (Ext 4 sister) but with u32
    length prefix. Scans for trailing WAVELET_MAGIC slot.
    """
    if len(fec6_archive_bytes) < len(WAVELET_MAGIC) + 4:
        return fec6_archive_bytes, None
    max_window = min(len(fec6_archive_bytes), 4 + 4 + (2**32 - 1))
    base_offset = len(fec6_archive_bytes) - max_window
    tail = fec6_archive_bytes[-max_window:]
    pos = tail.rfind(WAVELET_MAGIC)
    while pos >= 0:
        global_pos = base_offset + pos
        if global_pos + 4 + 4 > len(fec6_archive_bytes):
            pos = tail.rfind(WAVELET_MAGIC, 0, pos)
            continue
        u32_candidate = struct.unpack_from("<I", fec6_archive_bytes, global_pos + 4)[0]
        if global_pos + 4 + 4 + u32_candidate == len(fec6_archive_bytes):
            base = fec6_archive_bytes[:global_pos]
            haar_payload = fec6_archive_bytes[global_pos + 4 + 4 :]
            return base, haar_payload
        pos = tail.rfind(WAVELET_MAGIC, 0, pos)
    return fec6_archive_bytes, None
