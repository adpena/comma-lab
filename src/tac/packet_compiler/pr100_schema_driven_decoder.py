# SPDX-License-Identifier: MIT
"""PR100 schema-driven decoder storage grammar.

This module ports the **schema-driven decoder storage grammar** from PR100
public submission (``hnerv_lc_v2/inflate.py``, lines 51-64, and
``hnerv_lc_v2/schema.py``) into a typed, golden-vector-backed transducer.

PR100 grammar (V2 — sister of PR98 V1 CD1 compact format)
=========================================================

The archive splits the decoder state into TWO parallel streams::

    decoder_blob = brotli(concat( int8_body[i] for i in schema_order ))
    scales_blob  = concat( fp16_scale[i] for i in schema_order )  # NOT brotli'd; raw fp16

* **Bodies** (int8) — all concatenated into ONE int8 blob, brotli'd.
  Bodies share statistical properties (per-tensor symmetric INT8
  quantisation), so the merged stream compresses better than per-tensor
  interleaving.
* **Scales** (fp16) — all concatenated into a separate parallel stream.
  Scales also share statistical properties (all small floats), so
  separating them from bodies improves brotli's context utilisation on
  both streams.

The decoder reconstructs section bounds **at parse time** from the
caller-supplied schema ``[(name, shape), ...]`` — bodies consume
``prod(shape) * 1`` bytes per tensor, scales consume 2 bytes per tensor.
No length prefixes are stored per tensor.

Key design property
===================

Per-tensor **NAMES** and **SHAPES** are NOT stored in the archive. The
caller-supplied schema is the wire-format contract — encoder and decoder
MUST share the canonical state-dict iteration order. Any architecture
change requires a decoder source revision.

This is **the same mechanism** as PR98 V1 CD1 (eliminate per-tensor name
+ shape metadata by hardcoding the schema into the decoder source). The
difference is **layout**:

* PR98 V1: per-tensor interleaved ``[scale][body]`` pairs.
* PR100 V2: all bodies concatenated, then all scales concatenated (also
  no zigzag — bodies are int8 directly, not zigzag-encoded).

Note: PR100's bodies are **raw int8** (``np.frombuffer(raw, dtype=np.int8)``
at inflate.py:53), not zigzag-encoded. This is a separate design choice
from PR98 (which DOES zigzag-encode). It is wire-compatible with PR101's
``twos`` sign-encoding strategy. The reason PR100 omits zigzag: when
training produces a roughly-symmetric distribution around zero in int8
space, brotli's context model already handles the bipolar histogram
reasonably well; the zigzag overhead becomes optional.

Mutual exclusion with PR98 V1
=============================

PR98 V1 (this module's sister ``pr98_cd1_compact_format``) and PR100 V2
(this module) are **fundamentally the same mechanism** (eliminate
per-tensor name + shape metadata by hardcoding the schema into the
decoder source). They are expressed differently but target the
**same** ~840 B/archive metadata region.

**Stacking them is double-counting**. The archive has only one set of
metadata bytes to elide; applying both does NOT add their savings.

The probe-disambiguator (``tools/probe_schema_elision_disambiguator.py``)
empirically measures which variant produces the smaller archive on a
target state-dict.

Composition with PR105 size-sort
=================================

PR105's V3 size-sort (already landed at ``pr105_packed_state_schema``) is
a **reorder** on top of V2. PR105 sorts the schema by descending
``prod(shape)`` BEFORE the V2 concatenation step, so the brotli stream
sees the largest-entropy bodies first while its entropy model is still
building. The composition V2 + V3 is **stackable** (different byte
regions) and saves ~30-60 B on top of V2's ~840 B.

Byte savings
============

vs PR95 self-describing format: ~840 B/archive (same as V1; mutually
exclusive savings region).
vs PR98 V1 (CD1 compact): ~30-100 B residual savings from
separate-stream brotli on the bodies-vs-scales split. Empirical — measure
via probe.

Predicted score Δ at PR106 r2: ~ -5.6e-4 (improvement) — but at the
saturated entropy frontier, empirical may regress. The probe IS the
empirical verdict.

Source citation (read-only PR intake clone — Catalog #109 protected)
=====================================================================

``experiments/results/public_pr_archive_kaggle_mirror/public_pr100_intake_20260505_auto/source/submissions/hnerv_lc_v2/inflate.py``
(lines 51-64) + ``schema.py`` (lines 9-38).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; bit-exact on the ``pr100_schema_driven_decoder_v1``
  golden vector.
* Pure functional transducers — no global mutable state.
* No archive bytes mutated by this module — it is byte-grammar plumbing
  only.

[empirical:src/tac/packet_compiler/golden_vectors/pr100_schema_driven_decoder_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SchemaDrivenPayload:
    """The two parallel streams of the V2 schema-driven decoder.

    Attributes
    ----------
    body_blob:
        Concatenated int8 bodies (one int8 per tensor element, in schema
        order). Length = ``sum(prod(shape) for name, shape in schema)``
        bytes.
        Caller wraps with brotli externally per PR100's design.
    scales_blob:
        Concatenated fp16 scales (2 bytes per tensor, in schema order).
        Length = ``2 * n_tensors`` bytes. NOT brotli'd in PR100's design;
        stored raw because fp16 is already a tight representation.
    """

    body_blob: bytes
    scales_blob: bytes


def encode_schema_driven(
    quantized_tensors: Iterable[tuple[np.ndarray, float]],
) -> SchemaDrivenPayload:
    """Encode (int8_body, scale) pairs into the V2 schema-driven layout.

    The caller is responsible for:

    1. Iterating their state-dict in canonical order (matching the
       schema the decoder will use).
    2. Quantising each tensor to int8.
    3. Passing the int8 array + scale per tensor.

    The encoder concatenates all bodies into one int8 stream + all
    scales into one fp16 stream. The streams are returned as a
    :class:`SchemaDrivenPayload`; the caller is responsible for wrapping
    ``body_blob`` with brotli before writing into the archive.

    The encoder erases tensor SHAPES — they must be recovered at decode
    time from the schema.

    Parameters
    ----------
    quantized_tensors:
        Iterable of ``(int8_array, scale_float)`` pairs.

    Returns
    -------
    SchemaDrivenPayload
        Two parallel byte streams (bodies + scales).

    Raises
    ------
    ValueError
        On non-int8 input or non-finite scale value.
    """
    body_parts: list[bytes] = []
    scale_parts: list[np.ndarray] = []
    for i, (q, scale) in enumerate(quantized_tensors):
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
        body_parts.append(np.ascontiguousarray(q).reshape(-1).tobytes())
        scale_parts.append(np.array([scale_f], dtype=np.float16))
    if scale_parts:
        scales_blob = np.concatenate(scale_parts).tobytes()
    else:
        scales_blob = b""
    return SchemaDrivenPayload(
        body_blob=b"".join(body_parts),
        scales_blob=scales_blob,
    )


def decode_schema_driven(
    body_blob: bytes,
    scales_blob: bytes,
    schema: Iterable[tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[np.ndarray, float, tuple[int, ...]]]:
    """Decode the V2 schema-driven bodies + scales against a schema.

    Mirrors PR100 source ``decode_decoder`` (inflate.py lines 51-64)
    exactly:

    1. Parse the bodies as one big int8 array.
    2. Parse the scales as one big fp16 array.
    3. For each ``(name, shape)`` in the schema:
       a. Slice ``prod(shape)`` int8 elements from the bodies stream.
       b. Reshape to the target shape.
       c. Look up the fp16 scale for that tensor index.

    Parameters
    ----------
    body_blob:
        The concatenated int8 bodies (post-brotli-decompress if the
        caller wrapped externally).
    scales_blob:
        The concatenated fp16 scales (raw, not brotli'd).
    schema:
        Iterable of ``(name, shape)`` tuples in canonical state-dict
        iteration order.

    Returns
    -------
    dict[str, tuple[np.ndarray, float, tuple[int, ...]]]
        Mapping ``name -> (int8_body, scale, shape)``.

    Raises
    ------
    ValueError
        On tensor-count / size mismatch or leftover bytes.
    TypeError
        On non-bytes input.
    """
    for nm, val in (("body_blob", body_blob), ("scales_blob", scales_blob)):
        if not isinstance(val, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"{nm} must be bytes-like; got {type(val)!r}"
            )
    schema_list = list(schema)
    n_tensors = len(schema_list)
    # Parse scales — fp16 (2 bytes each).
    expected_scales_bytes = 2 * n_tensors
    if len(scales_blob) != expected_scales_bytes:
        raise ValueError(
            f"scales_blob length {len(scales_blob)} != "
            f"expected {expected_scales_bytes} (2 * n_tensors)"
        )
    scales = np.frombuffer(bytes(scales_blob), dtype=np.float16)
    # Parse bodies — int8 stream.
    codes = np.frombuffer(bytes(body_blob), dtype=np.int8)
    expected_body_size = 0
    shapes_t: list[tuple[int, ...]] = []
    for name, shape in schema_list:
        shape_t = tuple(int(d) for d in shape)
        shapes_t.append(shape_t)
        n_el = 1
        for d in shape_t:
            n_el *= d
        expected_body_size += n_el
    if codes.size != expected_body_size:
        raise ValueError(
            f"body_blob size {codes.size} != expected {expected_body_size} "
            f"(sum of prod(shape) over schema)"
        )
    out: dict[str, tuple[np.ndarray, float, tuple[int, ...]]] = {}
    offset = 0
    for i, ((name, _shape), shape_t) in enumerate(zip(schema_list, shapes_t)):
        n_el = 1
        for d in shape_t:
            n_el *= d
        chunk = codes[offset : offset + n_el].reshape(shape_t).copy()
        scale = float(scales[i])
        out[name] = (chunk, scale, shape_t)
        offset += n_el
    return out


__all__ = [
    "SchemaDrivenPayload",
    "decode_schema_driven",
    "encode_schema_driven",
]
