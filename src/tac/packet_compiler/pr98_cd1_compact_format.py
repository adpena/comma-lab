"""PR98 CD1 compact decoder-format primitive.

This module ports the **CD1 compact archive grammar** from PR98 public
submission (``hnerv_muon_finetuned_from_pr95/src/codec.py``, lines 74-107)
into a typed, golden-vector-backed transducer.

PR98 grammar (literal bytes, little-endian throughout)::

    magic   = b'CD1'              # 3 bytes
    sb      = uint8               # 1 byte; scale dtype width: 16 (fp16) or 32 (fp32)
    n_t     = uint32              # 4 bytes; number of tensors
    tensor[i] = scale(sb/8 bytes) + zigzag_int8_body(prod(shape) bytes)

Key design property
===================

Per-tensor **NAMES** and **SHAPES** are NOT stored in the archive. The
decoder iterates ``HNeRVDecoder.state_dict()`` (or another caller-supplied
schema) in canonical insertion order and consumes ``prod(shape) * 1`` body
bytes per tensor.

This is the **schema contract**: encoder + decoder MUST share the
canonical state-dict iteration order. Any architecture change requires a
decoder source revision.

Byte savings
============

vs PR95 self-describing format (per design memo §1.1):
- PR95 per-tensor overhead: ~30 bytes (name_len + name + shape_dims +
  shape).
- 28 tensors * 30 bytes ≈ ~840 bytes saved per archive.

Predicted score Δ at PR106 r2: ~ -5.6e-4 (improvement) — but at the
saturated entropy frontier, empirical may regress. The probe
``tools/probe_schema_elision_disambiguator.py`` IS the empirical verdict.

Mutual exclusion with PR100 schema-driven decoder
=================================================

PR98 CD1 (this module) and PR100 schema-driven (the sibling module
``pr100_schema_driven_decoder``) are **fundamentally the same mechanism**
(eliminate per-tensor name + shape metadata by hardcoding the schema into
the decoder source). They are expressed differently but target the
**same** ~840 B/archive metadata region.

**Stacking them is double-counting**. The archive has only one set of
metadata bytes to elide; applying both does NOT add their savings.

Composition with PR105 size-sort
=================================

V3 (``pr105_packed_state_schema_size_sorted``, already landed) is a
**reorder** on top of V2 (PR100 schema-driven). It is NOT a reorder on
V1 (PR98 CD1 interleaved scale+body) — PR105 packs all bodies adjacent
then all scales adjacent, which V1's interleaved grammar doesn't expose.

So the composition matrix is:

* V1 alone: yes (-840 B)
* V2 alone: yes (-840 B)
* V1 + V2: MUTUALLY EXCLUSIVE
* V2 + V3: yes (V2 ~840 B + V3 ~30 B = ~870 B)
* V1 + V3: requires lifting V1 to V2 first (not implemented)
* V3 alone: N/A (V3 has no benefit without V2's bodies-then-scales substrate)

Source citation (read-only PR intake clone — Catalog #109 protected)
=====================================================================

``experiments/results/public_pr_archive_kaggle_mirror/public_pr98_intake_20260505_auto/source/submissions/hnerv_muon_finetuned_from_pr95/src/codec.py``
(lines 74-107).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import (intentional — the caller supplies the schema as
  a list of (name, shape) tuples and provides the float values
  separately).
* No ``/tmp`` paths.
* Frozen dataclass; bit-exact on the ``pr98_cd1_compact_format_v1``
  golden vector.
* Pure functional transducers — no global mutable state.
* No archive bytes mutated by this module — it is byte-grammar plumbing
  only.

[empirical:src/tac/packet_compiler/golden_vectors/pr98_cd1_compact_format_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


CD1_MAGIC: bytes = b"CD1"
"""The 3-byte CD1 magic prefix that distinguishes PR98's compact format
from PR95's self-describing format.

PR98 source line 84: ``if magic != b'CD1': raise ValueError(...)``.
"""

SUPPORTED_SCALE_BITS: tuple[int, ...] = (16, 32)
"""PR98 source line 96-100 supports two scale-width branches:

* ``sb == 16`` -> fp16 scale (2 bytes per tensor)
* ``sb == 32`` -> fp32 scale (4 bytes per tensor)
"""


ScaleBits = Literal[16, 32]


def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """Zigzag encode int8 -> uint8.

    PR98 source line 42-44::

        def zigzag_encode_i8(arr_i8):
            arr = arr_i8.astype(np.int32)
            return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)
    """
    arr = arr_i8.astype(np.int32)
    return np.where(arr >= 0, 2 * arr, -2 * arr - 1).astype(np.uint8)


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    """Zigzag decode uint8 -> int8 (PR98 source line 47-49)."""
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


@dataclass(frozen=True)
class CD1CompactFormat:
    """Serialised CD1 compact format header + payload.

    Attributes
    ----------
    scale_bits:
        Scale dtype width (16 or 32). 16 -> fp16, 32 -> fp32.
    n_tensors:
        Number of tensors in the archive.
    payload:
        The compact-format byte stream (excluding the magic prefix but
        including ``scale_bits``, ``n_tensors``, and per-tensor
        ``scale + body``).
    """

    scale_bits: ScaleBits
    n_tensors: int
    payload: bytes


def encode_cd1_compact(
    quantized_tensors: Iterable[tuple[np.ndarray, float]],
    *,
    scale_bits: ScaleBits = 16,
) -> bytes:
    """Encode a sequence of (zigzagged-int8-body, scale) pairs into CD1 bytes.

    The caller is responsible for:

    1. Iterating their state-dict in canonical order.
    2. Quantising each tensor to int8 (e.g., via per-tensor symmetric
       INT8 quant — PR98 source lines 30-39).
    3. Passing the int8 array + scale per tensor.

    The encoder concatenates ``CD1_MAGIC + scale_bits(u8) +
    n_tensors(u32_le) + for_each_tensor(scale + zigzag_encoded_body)``.

    Parameters
    ----------
    quantized_tensors:
        Iterable of ``(int8_array, scale_float)`` pairs. The int8 array
        is flattened (encoder does NOT preserve shape — the decoder
        recovers shape from the schema).
    scale_bits:
        Scale dtype width. 16 -> fp16, 32 -> fp32.

    Returns
    -------
    bytes
        The CD1-encoded compact archive bytes (caller wraps with brotli
        externally per PR98's design).

    Raises
    ------
    ValueError
        On unsupported ``scale_bits``, non-int8 input, or non-finite
        scale value.
    """
    if scale_bits not in SUPPORTED_SCALE_BITS:
        raise ValueError(
            f"scale_bits must be one of {SUPPORTED_SCALE_BITS}; got {scale_bits}"
        )
    buf = io.BytesIO()
    buf.write(CD1_MAGIC)
    buf.write(struct.pack("<B", scale_bits))
    # We need to write n_tensors. Materialise to a list to count.
    tensors_list = list(quantized_tensors)
    buf.write(struct.pack("<I", len(tensors_list)))
    for i, (q, scale) in enumerate(tensors_list):
        if not isinstance(q, np.ndarray):
            raise ValueError(
                f"tensor[{i}] body must be np.ndarray; got {type(q)!r}"
            )
        if q.dtype != np.int8:
            raise ValueError(
                f"tensor[{i}] body must be dtype=int8; got {q.dtype}"
            )
        scale_f = float(scale)
        if not np.isfinite(scale_f):
            raise ValueError(
                f"tensor[{i}] scale must be finite; got {scale!r}"
            )
        # PR98 source line 68: writes fp16 OR fp32 by branching on scale_bits.
        if scale_bits == 16:
            buf.write(
                np.array([scale_f], dtype=np.float16).tobytes()
            )
        else:  # scale_bits == 32
            buf.write(struct.pack("<f", scale_f))
        # PR98 source line 70: zigzag_encode_i8(q).tobytes(). q is flattened.
        buf.write(_zigzag_encode_i8(np.ascontiguousarray(q).reshape(-1)).tobytes())
    return buf.getvalue()


def decode_cd1_compact(
    data: bytes,
    schema: Iterable[tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[np.ndarray, float, tuple[int, ...]]]:
    """Decode CD1 compact bytes against a (name, shape) schema.

    Mirrors PR98 source ``_decode_compact_decoder`` (lines 74-107)
    exactly: the schema (HNeRVDecoder.state_dict iteration order +
    shapes) is supplied by the caller, NOT stored in the bytes.

    Parameters
    ----------
    data:
        The CD1-encoded compact archive bytes (post-brotli-decompress).
    schema:
        Iterable of ``(name, shape)`` tuples in canonical state-dict
        iteration order. The decoder consumes ``prod(shape) * 1`` body
        bytes per tensor.

    Returns
    -------
    dict[str, tuple[np.ndarray, float, tuple[int, ...]]]
        Mapping ``name -> (int8_body, scale, shape)``.

    Raises
    ------
    ValueError
        On bad magic, unsupported scale_bits, tensor-count mismatch, or
        truncated payload.
    TypeError
        On non-bytes input.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"data must be bytes-like; got {type(data)!r}")
    buf = io.BytesIO(bytes(data))
    magic = buf.read(3)
    if magic != CD1_MAGIC:
        raise ValueError(
            f"bad CD1 magic: got {magic!r}; expected {CD1_MAGIC!r}"
        )
    sb_raw = buf.read(1)
    if len(sb_raw) != 1:
        raise ValueError("truncated CD1 header (missing scale_bits)")
    scale_bits = struct.unpack("<B", sb_raw)[0]
    if scale_bits not in SUPPORTED_SCALE_BITS:
        raise ValueError(
            f"unsupported CD1 scale_bits: {scale_bits} "
            f"(expected one of {SUPPORTED_SCALE_BITS})"
        )
    n_raw = buf.read(4)
    if len(n_raw) != 4:
        raise ValueError("truncated CD1 header (missing n_tensors)")
    n_tensors = struct.unpack("<I", n_raw)[0]
    schema_list = list(schema)
    if n_tensors != len(schema_list):
        raise ValueError(
            f"CD1 tensor count mismatch: encoded n_t={n_tensors} != "
            f"schema_len={len(schema_list)}"
        )
    scale_byte_width = scale_bits // 8
    out: dict[str, tuple[np.ndarray, float, tuple[int, ...]]] = {}
    for name, shape in schema_list:
        shape_t = tuple(int(d) for d in shape)
        # Read scale.
        scale_raw = buf.read(scale_byte_width)
        if len(scale_raw) != scale_byte_width:
            raise ValueError(
                f"truncated CD1 payload reading scale for tensor {name!r}"
            )
        if scale_bits == 16:
            scale = float(np.frombuffer(scale_raw, dtype=np.float16)[0])
        else:  # 32
            scale = struct.unpack("<f", scale_raw)[0]
        # Read body — prod(shape) zigzag-encoded uint8 bytes.
        body_size = 1
        for d in shape_t:
            body_size *= d
        body_raw = buf.read(body_size)
        if len(body_raw) != body_size:
            raise ValueError(
                f"truncated CD1 payload reading body for tensor {name!r}: "
                f"expected {body_size} bytes, got {len(body_raw)}"
            )
        body_u8 = np.frombuffer(body_raw, dtype=np.uint8)
        body_i8 = _zigzag_decode_u8(body_u8)
        out[name] = (body_i8, scale, shape_t)
    return out


__all__ = [
    "CD1CompactFormat",
    "CD1_MAGIC",
    "SUPPORTED_SCALE_BITS",
    "ScaleBits",
    "decode_cd1_compact",
    "encode_cd1_compact",
]
