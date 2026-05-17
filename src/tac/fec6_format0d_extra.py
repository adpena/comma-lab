# SPDX-License-Identifier: MIT
"""fec6 + format0d-EXTRA additive latent correction codec (Ext 4).

This module is the canonical encoder/decoder for the fec6+format0d-EXTRA
outer-wrapper extension (lane
``lane_fec6_stacking_wave_5_grammar_extensions_20260517``).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
lesson 7 (bolt-on vs substrate-engineering split): this is a BOLT-ON
extension to the fec6 outer wrapper. fec6's outer grammar is::

    OUTER_MAGIC=FP11 | u32 source_len | source_payload | u16 selector_len | selector_payload

This module adds a NEW trailing slot after the selector_payload::

    [existing fec6 outer wrapper] | EXTRA_MAGIC=FE6E | u16 extra_payload_len | extra_payload

where ``extra_payload`` is a byte-deterministic per-pair (dim_arr,
delta_q_arr) correction stream, encoded as::

    u32 n_pairs | u8 dim_arr[n_pairs] | i8 delta_q_arr[n_pairs] | fp16 scale

At inflate time, the EXTRA correction is applied to PR101's decoded
latents via the same ``apply_sidecar_corrections`` pattern as PR106
format0d (verified at
``submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:578-592``)::

    latents[p, dim_arr[p]] += delta_q_arr[p] * scale  if dim_arr[p] != 255

Per CLAUDE.md "Bit-level deconstruction and entropy discipline":
- Total uncompressed payload size = 8 + n_pairs + n_pairs + 2 = 2*n_pairs + 10 bytes.
  At n_pairs=600 that's 1210 bytes.
- The byte-deterministic build does NOT brotli the extra_payload (the
  per-pair (dim, delta_q) pairs are not highly redundant in general; a
  Brotli wrapper adds 6-8 bytes of header overhead for sub-1KB streams
  with marginal compression gain).
- Future Phase 2 could add brotli compression of (dim_arr, delta_q_arr)
  separately for additional rate savings (~30-50% on the sparse stream
  if the operator decides to escalate).

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
predicted ΔS band ``[-0.0005, -0.0030] [predicted, council-consensus]``
on contest-CPU axis.

Per CLAUDE.md "Apples-to-apples evidence discipline": no contest-CUDA /
contest-CPU claims; only ``[predicted, theoretical]`` / ``[predicted,
council-consensus]`` until paired-axis dispatch lands.

Design memo: ``.omx/research/fec6_plus_format0d_extra_design_20260517.md``
"""
from __future__ import annotations

import struct

import numpy as np

__all__ = [
    "DEFAULT_DELTA_SCALE",
    "EXTRA_MAGIC",
    "NO_OP_DIM",
    "Format0dExtraDecodeError",
    "Format0dExtraEncodeError",
    "decode_format0d_extra_payload",
    "encode_format0d_extra_payload",
    "unwrap_fec6_archive_with_extra",
    "wrap_fec6_archive_with_extra",
]


EXTRA_MAGIC: bytes = b"FE6E"  # fec6 EXTRA correction slot
DEFAULT_DELTA_SCALE: float = 0.01  # mirrors PR106 format0d DELTA_SCALE
NO_OP_DIM: int = 255  # mirrors PR106 format0d NO_OP_DIM sentinel


class Format0dExtraEncodeError(ValueError):
    """Raised when the format0d-EXTRA encoder cannot produce a valid payload."""


class Format0dExtraDecodeError(ValueError):
    """Raised when the format0d-EXTRA decoder rejects a payload as malformed."""


def encode_format0d_extra_payload(
    *,
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    scale: float = DEFAULT_DELTA_SCALE,
) -> bytes:
    """Encode a per-pair (dim, delta_q) correction table to byte-deterministic payload.

    Format::

        u32 n_pairs (little-endian)
      | u8  dim_arr[n_pairs]    (255 = NO_OP)
      | i8  delta_q_arr[n_pairs]
      | fp16 scale (little-endian)

    Total size = 4 + n_pairs + n_pairs + 2 = 2 * n_pairs + 6 bytes.

    Parameters
    ----------
    dim_arr : np.ndarray (uint8)
        Per-pair correction dimension index. ``255`` is the NO_OP sentinel.
    delta_q_arr : np.ndarray (int8)
        Per-pair correction magnitude (signed 8-bit).
    scale : float
        Multiplicative scale factor applied at inflate time
        (``latents[p, d] += delta_q[p] * scale``). Stored as fp16.

    Returns
    -------
    bytes
        The byte-deterministic extra-payload (not including the outer
        wrapper EXTRA_MAGIC or u16 length prefix).

    Raises
    ------
    Format0dExtraEncodeError
        If ``dim_arr`` and ``delta_q_arr`` shapes mismatch, or dtypes
        don't match the wire format requirement.
    """
    if dim_arr.ndim != 1:
        raise Format0dExtraEncodeError(
            f"dim_arr must be 1-D; got shape {dim_arr.shape}"
        )
    if delta_q_arr.ndim != 1:
        raise Format0dExtraEncodeError(
            f"delta_q_arr must be 1-D; got shape {delta_q_arr.shape}"
        )
    if dim_arr.shape[0] != delta_q_arr.shape[0]:
        raise Format0dExtraEncodeError(
            f"dim_arr and delta_q_arr must have same length; "
            f"got {dim_arr.shape[0]} vs {delta_q_arr.shape[0]}"
        )
    n_pairs = int(dim_arr.shape[0])
    if n_pairs <= 0:
        raise Format0dExtraEncodeError(f"n_pairs must be positive; got {n_pairs}")
    if n_pairs > 2**32 - 1:
        raise Format0dExtraEncodeError(f"n_pairs={n_pairs} exceeds u32 range")
    # Coerce dtypes for byte-deterministic encoding.
    dim_u8 = np.ascontiguousarray(dim_arr.astype(np.uint8, copy=False))
    delta_i8 = np.ascontiguousarray(delta_q_arr.astype(np.int8, copy=False))
    scale_f16 = np.float16(scale)
    if not np.isfinite(float(scale_f16)):
        raise Format0dExtraEncodeError(
            f"scale={scale} is non-finite in fp16; refusing encode"
        )
    payload = (
        struct.pack("<I", n_pairs)
        + dim_u8.tobytes()
        + delta_i8.tobytes()
        + scale_f16.tobytes()
    )
    expected_size = 4 + n_pairs + n_pairs + 2
    if len(payload) != expected_size:
        raise Format0dExtraEncodeError(
            f"encoded payload size mismatch: got {len(payload)} expected {expected_size}"
        )
    return payload


def decode_format0d_extra_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode the byte-deterministic extra-payload back into (dim_arr, delta_q_arr, scale).

    Returns
    -------
    tuple
        ``(dim_arr: np.ndarray uint8, delta_q_arr: np.ndarray int8, scale: float)``.

    Raises
    ------
    Format0dExtraDecodeError
        If the payload is too short, n_pairs is implausible, or the
        total byte length doesn't match the inferred shape.
    """
    if len(payload) < 6:
        raise Format0dExtraDecodeError(
            f"payload too short for header+scale (need >= 6 bytes); got {len(payload)}"
        )
    n_pairs = struct.unpack_from("<I", payload, 0)[0]
    expected_size = 4 + n_pairs + n_pairs + 2
    if len(payload) != expected_size:
        raise Format0dExtraDecodeError(
            f"payload size mismatch: got {len(payload)} expected {expected_size} for n_pairs={n_pairs}"
        )
    dim_arr = np.frombuffer(payload, dtype=np.uint8, count=n_pairs, offset=4).copy()
    delta_q_arr = np.frombuffer(payload, dtype=np.int8, count=n_pairs, offset=4 + n_pairs).copy()
    scale_f16 = np.frombuffer(payload, dtype=np.float16, count=1, offset=4 + 2 * n_pairs)[0]
    return dim_arr, delta_q_arr, float(scale_f16)


def wrap_fec6_archive_with_extra(
    *,
    fec6_archive_bytes: bytes,
    extra_payload: bytes,
) -> bytes:
    """Append the format0d-EXTRA slot to an existing fec6 archive byte stream.

    The fec6 archive is a ZIP container with a single member ``x`` whose
    bytes are the OUTER_MAGIC wrapper. This helper appends ``EXTRA_MAGIC
    + u16 extra_len + extra_payload`` to the WRAPPER (inside the ZIP
    member) — NOT outside the ZIP container.

    The caller is responsible for re-zipping the result via the same
    ZipInfo + writestr pattern used by ``tac.zip_pack._zip_bytes`` (the
    canonical byte-deterministic ZIP writer per Catalog #158). This
    helper returns the inner-member bytes.

    Parameters
    ----------
    fec6_archive_bytes : bytes
        The existing fec6 outer-wrapper bytes (single ZIP member content,
        BEFORE the ZIP wrapping).
    extra_payload : bytes
        The output of ``encode_format0d_extra_payload``.

    Returns
    -------
    bytes
        The extended inner-member bytes::

            <fec6_archive_bytes> | EXTRA_MAGIC | u16 len(extra_payload) | extra_payload
    """
    if len(extra_payload) > 2**16 - 1:
        raise Format0dExtraEncodeError(
            f"extra_payload len {len(extra_payload)} exceeds u16 max 65535"
        )
    return fec6_archive_bytes + EXTRA_MAGIC + struct.pack("<H", len(extra_payload)) + extra_payload


def unwrap_fec6_archive_with_extra(
    fec6_archive_bytes: bytes,
) -> tuple[bytes, bytes | None]:
    """Split fec6 inner-member bytes into (base_fec6, extra_payload-or-None).

    The trailing format is ``EXTRA_MAGIC (4 bytes) | u16 extra_len (2 bytes) | extra_payload``.
    To deterministically locate the slot we scan backwards looking for
    ``EXTRA_MAGIC`` whose position is consistent with the trailing u16
    length. Since EXTRA_MAGIC is a 4-byte fixed sequence and the slot
    placement is uniquely determined by len(payload), this is O(N) in
    the worst case but typically finds the magic in constant time
    because the structure ANCHORS the magic at exactly
    ``len(bytes) - 4 - 2 - extra_len``.

    If the bytes don't end with a valid EXTRA_MAGIC + len-prefixed
    payload, returns ``(fec6_archive_bytes, None)`` indicating no extra
    slot is present.

    This is the inflate-side helper for the format0d-EXTRA wrapper.
    """
    if len(fec6_archive_bytes) < len(EXTRA_MAGIC) + 2:
        return fec6_archive_bytes, None
    # Bounded backward search: walk back from the end through the last
    # 2**16 + 6 bytes max (the largest possible EXTRA slot) looking for
    # EXTRA_MAGIC. The magic position is uniquely determined by the
    # u16 length immediately following it.
    max_window = min(len(fec6_archive_bytes), 4 + 2 + (2**16 - 1))
    tail = fec6_archive_bytes[-max_window:]
    # Find ALL occurrences of EXTRA_MAGIC in the tail window (rare; in
    # the canonical encoder there's exactly one). For each candidate,
    # verify the u16 length is consistent with the remaining bytes.
    base_offset = len(fec6_archive_bytes) - max_window
    pos = tail.rfind(EXTRA_MAGIC)
    while pos >= 0:
        global_pos = base_offset + pos
        if global_pos + 4 + 2 > len(fec6_archive_bytes):
            pos = tail.rfind(EXTRA_MAGIC, 0, pos)
            continue
        u16_candidate = struct.unpack_from("<H", fec6_archive_bytes, global_pos + 4)[0]
        if global_pos + 4 + 2 + u16_candidate == len(fec6_archive_bytes):
            # Found the canonical trailing slot.
            base = fec6_archive_bytes[:global_pos]
            extra_payload = fec6_archive_bytes[global_pos + 4 + 2 :]
            return base, extra_payload
        pos = tail.rfind(EXTRA_MAGIC, 0, pos)
    return fec6_archive_bytes, None
