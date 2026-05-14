# SPDX-License-Identifier: MIT
"""Dual-layer Filler-Pevný 2010 STC + magnitude-residual codec for mask deltas.

Background
----------
The single-layer Filler STC variant in
:mod:`tac.codec.syndrome_trellis_codec` projects 5-class mask deltas
(``{-4, -3, -2, -1, 0, +1, +2, +3, +4}``) to ternary sign
(``{-1, 0, +1}``) and encodes the sign stream alone. That projection is
LOSSY: a real ``+3`` becomes ``+1``, the ``+2`` of magnitude is dropped.

The dual-layer construction (Filler & Pevný 2010 "Wet paper codes with
improved embedding efficiency") restores LOSSLESS reconstruction by
splitting the original delta stream into two complementary layers:

* **Layer 1 — sign**: ternary ``sign(deltas) ∈ {-1, 0, +1}``, compressed
  by the canonical entropy coder. The "STC" name is preserved because
  this layer has the same per-symbol semantics that the STC trellis
  operates on; in the LOSSLESS dual-layer variant we transmit the cover
  directly (no flips), which is what STC reduces to when every cover
  position is wet (``ρ_i = WET_COST``).

  The STC trellis stats (block flips, embedding cost) on a UNIFORM-cost
  realization are recorded as **research signal** alongside the
  lossless encoding — they bound the additional rate that an
  embedding-efficiency variant would have available, but are NOT used
  to construct the bitstream. This keeps the dual-layer codec strictly
  lossless while preserving the Filler-Pevný 2010 measurement.

* **Layer 2 — magnitude residual**: ``abs(deltas)`` taken at positions
  where ``deltas != 0``. For 5-class masks this lives in
  ``{1, 2, 3, 4}``. The values are run-length compact (most non-zero
  deltas at class boundaries are ``±1``) and compressed by the same
  canonical entropy coder.

Joint reconstruction:
    ``deltas[i] = sign[i] * magnitude[i]``
where ``magnitude[i] = 0`` whenever ``sign[i] = 0``. The roundtrip is
exact — see ``test_dual_layer_stc_av1_codec.py``.

Wire format
-----------
Little-endian, single-blob:
    magic                 : 4 bytes  = ``b'DLST'`` (DualLayerSTc)
    version               : 1 byte   = 1
    flags                 : 1 byte   = bit0 set if magnitude payload is empty
    n_symbols             : uint32   = total symbols (including zeros)
    n_nonzero             : uint32   = count where sign != 0
    layer1_len            : uint32   = sign-layer payload length (bytes)
    layer1_payload        : raw bytes (brotli of int8 sign stream)
    layer2_len            : uint32   = magnitude-layer payload length (bytes)
    layer2_payload        : raw bytes (brotli of uint8 nonzero magnitudes)

The payloads are produced by :data:`tac.codec.per_tensor_codecs.BROTLI_PARAMS`
via :func:`tac.codec.per_tensor_codecs.encode_brotli_only` — i.e. the same
PR101-pinned brotli parameters that the rest of the cathedral uses.

Falsification scope
-------------------
``dual_layer_stc_av1_lossless_only``: only the 5-class lossless encode/decode
roundtrip with brotli on both layers is tested. Score-aware embedding costs
(detector-in-loop), the GF(q>2) STC variant, and AV1-monochrome layer-2
substitution (lossy with mask-quality budget) remain in
``reactivation_criteria_remaining``.

Reference
---------
Filler, T., & Pevný, T. (2010). "Wet paper codes with improved embedding
efficiency." IEEE TIFS 5(2): 388-393.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli
import numpy as np

from tac.codec.per_tensor_codecs import BROTLI_PARAMS
from tac.codec.syndrome_trellis_codec import (
    STCParams,
    ternary_stc_encode_stream,
)

MAGIC: bytes = b"DLST"
VERSION: int = 1
HEADER_STRUCT = struct.Struct("<4sBBII I")  # magic, ver, flags, n_sym, n_nz, layer1_len
LAYER2_LEN_STRUCT = struct.Struct("<I")
FLAG_EMPTY_MAGNITUDE: int = 1 << 0
KNOWN_FLAGS_MASK: int = FLAG_EMPTY_MAGNITUDE


def _brotli_compress(payload: bytes) -> bytes:
    """Compress ``payload`` under the canonical PR101-pinned brotli params."""
    return brotli.compress(payload, **BROTLI_PARAMS)


def _brotli_decompress(payload: bytes) -> bytes:
    return brotli.decompress(payload)


@dataclass(frozen=True)
class DualLayerStats:
    """Trellis-research statistics produced alongside the lossless encode.

    These are emitted as a side-channel for analysis only; they do NOT
    enter the wire format and have no effect on reconstruction.
    """

    layer1_bytes: int
    layer2_bytes: int
    header_bytes: int
    total_bytes: int
    n_symbols: int
    n_nonzero: int
    nonzero_fraction: float
    # Filler-Pevný 2010 trellis statistics on the UNIFORM-cost ternary
    # encoding of the sign stream (research-signal only — the lossless
    # bitstream stores the cover unchanged).
    stc_uniform_block_size: int
    stc_uniform_constraint_height: int
    stc_uniform_n_blocks: int
    stc_uniform_flips_soz: int
    stc_uniform_flips_sign: int
    stc_uniform_total_cost: float


# ---------------------------------------------------------------------------
# Encode / decode primitives
# ---------------------------------------------------------------------------


def _validate_deltas(deltas: np.ndarray) -> np.ndarray:
    arr = np.asarray(deltas)
    if arr.ndim != 1:
        raise ValueError(f"deltas must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("deltas must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"deltas must be integer-valued, got dtype {arr.dtype}")

    arr64 = arr.astype(np.int64, copy=False)
    if arr64.min() < -127 or arr64.max() > 127:
        raise ValueError(
            "deltas must fit in int8 range; "
            f"got [{int(arr64.min())}, {int(arr64.max())}]"
        )
    return arr64.astype(np.int16, copy=False)


def encode_dual_layer(
    deltas: np.ndarray,
    *,
    constraint_height: int = 8,
    block_size: int = 64,
    record_stc_stats: bool = False,
) -> bytes | tuple[bytes, DualLayerStats]:
    """Encode an integer delta stream as a lossless dual-layer payload.

    Args:
        deltas: 1-D integer array of mask deltas; values may be the full
            5-class delta range ``{-4..+4}`` or any int8 range. Positions
            where ``deltas[i] == 0`` consume zero bits in Layer 2.
        constraint_height: STC ``h`` (used only for the uniform-cost
            research-signal pass when ``record_stc_stats=True``). Default
            8 matches the Filler-Pevný 2010 reference.
        block_size: STC block size for the same research-signal pass.
        record_stc_stats: if True, also run the uniform-cost STC trellis
            on the sign stream and return ``(blob, DualLayerStats)``.
            The trellis is O(2^h · n) per block and may be expensive on
            long streams — gated off by default.

    Returns:
        Compressed wire-format bytes blob; with ``record_stc_stats=True``
        a ``(blob, DualLayerStats)`` tuple.
    """
    arr = _validate_deltas(deltas)
    sign = np.sign(arr).astype(np.int8)  # values in {-1, 0, +1}
    nonzero_mask = sign != 0
    magnitude_full = np.abs(arr).astype(np.int16)
    magnitude_nz = magnitude_full[nonzero_mask].astype(np.uint8)
    if magnitude_nz.size and magnitude_nz.max() > 127:
        raise ValueError(
            f"magnitude must fit in uint8 range [0, 127]; got max={int(magnitude_nz.max())}"
        )

    layer1_payload = _brotli_compress(sign.tobytes())
    if magnitude_nz.size > 0:
        layer2_payload = _brotli_compress(magnitude_nz.tobytes())
        flags = 0
    else:
        layer2_payload = b""
        flags = FLAG_EMPTY_MAGNITUDE

    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        flags,
        int(arr.size),
        int(magnitude_nz.size),
        len(layer1_payload),
    )
    layer2_len = LAYER2_LEN_STRUCT.pack(len(layer2_payload))
    blob = header + layer1_payload + layer2_len + layer2_payload

    if not record_stc_stats:
        return blob

    # --- Research signal: STC trellis on uniform costs (sign stream) ---
    sign_costs = np.ones_like(sign, dtype=np.float64)
    stc_res = ternary_stc_encode_stream(
        sign,
        sign_costs,
        block_size=block_size,
        params=STCParams(constraint_height=constraint_height, submatrix_seed=0),
    )

    stats = DualLayerStats(
        layer1_bytes=len(layer1_payload),
        layer2_bytes=len(layer2_payload),
        header_bytes=HEADER_STRUCT.size + LAYER2_LEN_STRUCT.size,
        total_bytes=len(blob),
        n_symbols=int(arr.size),
        n_nonzero=int(magnitude_nz.size),
        nonzero_fraction=float(magnitude_nz.size) / float(arr.size),
        stc_uniform_block_size=int(block_size),
        stc_uniform_constraint_height=int(constraint_height),
        stc_uniform_n_blocks=int(stc_res["n_blocks"]),
        stc_uniform_flips_soz=int(stc_res["flips_soz"]),
        stc_uniform_flips_sign=int(stc_res["flips_sign"]),
        stc_uniform_total_cost=float(stc_res["total_cost"]),
    )
    return blob, stats


def decode_dual_layer(blob: bytes) -> np.ndarray:
    """Inverse of :func:`encode_dual_layer`. Reconstructs the original
    delta stream losslessly.
    """
    if len(blob) < HEADER_STRUCT.size:
        raise ValueError(f"blob too short for header: {len(blob)} bytes")
    magic, version, flags, n_sym, n_nz, layer1_len = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic: expected {MAGIC!r}, got {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version} (expected {VERSION})")
    unknown_flags = flags & ~KNOWN_FLAGS_MASK
    if unknown_flags:
        raise ValueError(f"unsupported flags set: 0x{unknown_flags:02x}")
    pos = HEADER_STRUCT.size

    if pos + layer1_len > len(blob):
        raise ValueError("truncated blob: layer1 payload missing")
    layer1_payload = blob[pos : pos + layer1_len]
    pos += layer1_len

    if pos + LAYER2_LEN_STRUCT.size > len(blob):
        raise ValueError("truncated blob: layer2 length missing")
    (layer2_len,) = LAYER2_LEN_STRUCT.unpack_from(blob, pos)
    pos += LAYER2_LEN_STRUCT.size
    if pos + layer2_len != len(blob):
        raise ValueError(
            f"layer2 size mismatch: declared {layer2_len}, remaining {len(blob) - pos}"
        )
    layer2_payload = blob[pos : pos + layer2_len]

    sign_bytes = _brotli_decompress(layer1_payload)
    if len(sign_bytes) != n_sym:
        raise ValueError(
            f"layer1 decompressed length {len(sign_bytes)} != declared n_symbols {n_sym}"
        )
    sign = np.frombuffer(sign_bytes, dtype=np.int8).copy()
    if sign.size and not np.isin(sign, (-1, 0, 1)).all():
        raise ValueError("layer1 sign payload contains values outside {-1, 0, +1}")
    nonzero_mask = sign != 0
    if int(nonzero_mask.sum()) != n_nz:
        raise ValueError(
            f"layer1 sign nonzero count {int(nonzero_mask.sum())} != header n_nonzero {n_nz}"
        )

    if flags & FLAG_EMPTY_MAGNITUDE:
        if n_nz != 0:
            raise ValueError("FLAG_EMPTY_MAGNITUDE set but n_nonzero != 0")
        if layer2_len != 0:
            raise ValueError("FLAG_EMPTY_MAGNITUDE set but layer2 payload is non-empty")
        magnitudes = np.zeros(0, dtype=np.uint8)
    else:
        magnitudes = np.frombuffer(_brotli_decompress(layer2_payload), dtype=np.uint8).copy()
        if magnitudes.size != n_nz:
            raise ValueError(
                f"layer2 decompressed length {magnitudes.size} != declared n_nonzero {n_nz}"
            )

    out = np.zeros(n_sym, dtype=np.int16)
    if n_nz > 0:
        out[nonzero_mask] = sign[nonzero_mask].astype(np.int16) * magnitudes.astype(np.int16)
    return out


# ---------------------------------------------------------------------------
# Convenience: full 5-class mask delta extraction (preserves magnitude)
# ---------------------------------------------------------------------------


def extract_full_mask_deltas(masks: np.ndarray) -> np.ndarray:
    """Return the full 5-class delta stream between consecutive frames.

    Unlike :func:`tac.codec.syndrome_trellis_codec.extract_mask_deltas_ternary`,
    this preserves the magnitude — the result lives in ``[-(NUM_CLASSES-1),
    +(NUM_CLASSES-1)]`` (i.e. ``[-4, +4]`` for 5-class masks).
    """
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError("masks must be (N, H, W)")
    diffs = np.diff(masks.astype(np.int16), axis=0)
    return diffs.reshape(-1).astype(np.int16)


__all__ = [
    "MAGIC",
    "VERSION",
    "DualLayerStats",
    "decode_dual_layer",
    "encode_dual_layer",
    "extract_full_mask_deltas",
]
