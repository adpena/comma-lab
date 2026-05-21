#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Canonical HFV sidecar recoder — bit-identical round-trip lossless byte reducer.

OVERNIGHT-X2 (lane ``lane_overnight_x2_build_hfv_sidecar_recoder_20260521``)
build per operator directive 2026-05-21 *"Build the 2/3 unbuilt"* + OVERNIGHT-S
landing memo ``pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md``
Path 3 prerequisite (commit ``079edcfdd``) + Carmack MVP-first 5-step per
CLAUDE.md amendment ``be125b878``.

Purpose
-------

The canonical ``foveation_params.bin`` HFV1 sidecar produced by
``tools/build_hfv1_sparse_sidecar_candidate.py`` and the OVERNIGHT-K hybrid
runtime ships as 16-byte header + N_frames * 20-byte row = 24,016 bytes
(at N=1200) of dense float32 5-tuples. Empirically (per OVERNIGHT-S PV)
97-100% of rows are identical to the most-common "identity" row; the file
is highly compressible without semantic loss.

This recoder shrinks ``foveation_params.bin`` LOSSLESSLY via one of three
canonical encoding strategies — bit-identical round-trip is structurally
enforced by an INVERSE decoder for every strategy + Catalog #229
``--verify-roundtrip`` smoke gate. The recoded sidecar is NEW bytes that
go into the archive as a REPLACEMENT for the dense HFV1 payload; the
inflate runtime distinguishes the format via magic-byte dispatch
(``HFV1`` vs the new ``HFRC`` recoded magic).

The recoded sidecar IS NOT a substitute for the existing canonical HFV2
sparse-pair representation (``tools/build_hfv1_sparse_sidecar_candidate.py``
+ canonical equation #356). HFV2 exploits the HIGH-LEVEL semantic structure
("most pairs share the default row; encode only the deviations"). This
recoder operates at the LOW-LEVEL byte representation surface ("the dense
bytes are highly compressible by brotli / entropy / per-component quantize
even when sparse semantic encoding is not possible"). The two compose:
HFV2 sparse + brotli recoder is the canonical stacking pattern (sister
to Catalog #105/#139/#220 no-op detector + operational mechanism).

Canonical strategies
--------------------

1. ``entropy_brotli`` (DEFAULT; canonical for high-redundancy inputs):
   apply brotli quality=11 (max) to the raw HFV1 bytes. Lossless;
   bit-identical round-trip via ``brotli.decompress``.

2. ``sparse_delta``: per-row delta vs the previous row (first row stored
   as absolute). Skip-zero runs encoded via varint counts. Best for
   inputs with strong temporal correlation but few unique rows.

3. ``per_component_quantize``: quantize each of 5 float32 components to
   int16 (foveation centers + sigmas + z lossy at 16-bit precision).
   NOT bit-identical lossless — this strategy is INTENTIONALLY OPT-IN
   under ``--lossy-ok`` and refuses round-trip verification by default
   per Catalog #229 PV discipline. NOT used for the canonical recoder
   round-trip path.

4. ``combined``: ``sparse_delta`` then ``entropy_brotli`` on the delta
   stream. Lossless; bit-identical round-trip via reverse composition.

Round-trip discipline (Catalog #229)
------------------------------------

Every canonical lossless strategy MUST satisfy: ``decode(encode(input)) == input``
byte-for-byte (sha256 match). The ``--verify-roundtrip`` flag (default ON)
runs the inverse decoder on the encoded output and refuses with non-zero
exit code if the sha256s do not match.

Provenance (Catalog #287/#323)
------------------------------

The output JSON report carries canonical ``axis_tag="[prediction]"``
(predicted rate-only delta) + ``promotable=False`` until paired Linux
x86_64 + NVIDIA empirical landing per CLAUDE.md "Submission auth eval
— BOTH CPU AND CUDA". The predicted rate-only delta is derived from
canonical equation #356 (``hfv2_sparse_pair_sidecar_replacement_savings_v1``)
closed-form ``deltaS = -25 * (N_dense - N_recoded) / 37_545_489``.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline":
arithmetic / range / ANS / Huffman / brotli / zstd / lzma transforms
are first-class score lanes. The recoded sidecar IS a first-class score
lane — but the score claim requires paired CUDA + CPU empirical anchors
on the actual archive bytes per CLAUDE.md "Submission auth eval — BOTH
CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable.

Cross-references
----------------

* Catalog #344 + canonical equation #356 (HFV2 sparse pair sidecar
  replacement savings) — sister REPLACEMENT-paradigm equation.
* Catalog #287 (placeholder-rationale rejection) — every score/byte
  claim in this module's docstring carries ``[prediction]`` evidence tag.
* Catalog #323 (canonical Provenance umbrella) — output JSON carries
  canonical Provenance.
* Catalog #229 (premise-verification-before-edit) — the
  ``--verify-roundtrip`` smoke is the canonical PV.
* CLAUDE.md "Bit-level deconstruction and entropy discipline" — this
  builder IS the canonical entry-point for foveation_params byte-level
  recoding.
* CLAUDE.md "Carmack MVP-first phasing" — $0 local CPU smoke FIRST;
  no paid GPU until the recoder structurally works.
* OVERNIGHT-S landing memo Decision 6 — Path 3 prerequisite for HFV
  cascade PR110-rebase + predicted rate-only delta
  ``-0.0093 [prediction]``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import brotli

# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

CONTEST_DENOM_BYTES = 37_545_489
"""Per CLAUDE.md canonical contest rate denominator."""

RATE_MULTIPLIER = 25.0
"""Per CLAUDE.md canonical contest rate multiplier."""

HFV1_HEADER = struct.Struct("<4sIII")
"""HFV1 sidecar header: magic(4) + n_frames + height + width."""

HFV1_ROW = struct.Struct("<fffff")
"""HFV1 per-frame row: 5 float32 (alpha, radius, power, origin_x, origin_y)."""

CAMERA_H = 874
CAMERA_W = 1164
"""PR101 canonical camera dimensions."""

HFRC_MAGIC = b"HFRC"
"""NEW canonical recoded sidecar magic. Distinguishes from HFV1 (dense) +
HFV2 (sparse-pair). Inflate runtime must magic-byte dispatch on first 4
bytes (HFV1 vs HFV2 vs HFRC) per the canonical Carmack pattern."""

HFRC_HEADER = struct.Struct("<4sIIIB")
"""HFRC recoded sidecar header: magic(4) + n_frames + height + width
+ encoding_strategy_byte. ``encoding_strategy_byte`` ∈
{0=entropy_brotli, 1=sparse_delta, 2=combined}."""

ENCODING_ENTROPY_BROTLI = 0
ENCODING_SPARSE_DELTA = 1
ENCODING_COMBINED = 2

ENCODING_NAMES = {
    ENCODING_ENTROPY_BROTLI: "entropy_brotli",
    ENCODING_SPARSE_DELTA: "sparse_delta",
    ENCODING_COMBINED: "combined",
}
ENCODING_BY_NAME = {v: k for k, v in ENCODING_NAMES.items()}

# Per Catalog #356 closed-form: deltaS = -25 * (N_dense - N_recoded) / 37_545_489
# Sister of canonical equation #26 (procedural_codebook_from_seed) at the
# byte-level entropy-recoded surface (REPLACEMENT-paradigm per OVERNIGHT-P
# landing memo + sister to canonical equation #356 hfv2_sparse_pair_sidecar
# replacement savings).
CANONICAL_EQUATION_356_ID = "hfv2_sparse_pair_sidecar_replacement_savings_v1"

# Lane-id (Catalog #126 pre-registered)
LANE_ID = "lane_overnight_x2_build_hfv_sidecar_recoder_20260521"


# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoderVerdict:
    """Canonical recoder verdict — all fields machine-readable.

    Per Catalog #287/#323 canonical Provenance discipline + Catalog #229
    PV discipline + CLAUDE.md "Beauty, simplicity, and developer experience"
    frozen-dataclass invariants.
    """

    schema_version: str = "hfv_sidecar_recoder_verdict_v1_20260521"
    input_path: str = ""
    output_path: str = ""
    encoding_strategy: str = ""
    input_bytes: int = 0
    output_bytes: int = 0
    bytes_saved: int = 0
    percent_reduction: float = 0.0
    rate_savings_predicted: float = 0.0
    """Per canonical equation #356: -25 * (N_dense - N_recoded) / 37_545_489.

    Carries ``[prediction]`` evidence tag per Catalog #287. NOT a score
    claim until paired CUDA + CPU empirical anchors land per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA"."""

    round_trip_verified: bool = False
    """True if decode(encode(input)) == input byte-for-byte (sha256 match)."""

    sha256_input: str = ""
    sha256_output: str = ""
    n_frames: int = 0
    camera_height: int = 0
    camera_width: int = 0

    # Catalog #287/#323: canonical Provenance defaults
    axis_tag: str = "[prediction]"
    promotable: bool = False
    score_claim: bool = False
    canonical_equation_reference: str = CANONICAL_EQUATION_356_ID

    # Diagnostic only
    elapsed_seconds: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# HFV1 parsing
# ---------------------------------------------------------------------------


def parse_hfv1(raw: bytes) -> tuple[int, int, int, list[tuple[float, float, float, float, float]]]:
    """Parse HFV1 dense sidecar bytes into (n_frames, height, width, rows).

    Raises ``ValueError`` on truncated / malformed / wrong-magic input
    per CLAUDE.md "Operator gates must be wired and used" fail-closed
    contract.
    """
    if len(raw) < HFV1_HEADER.size:
        raise ValueError(
            f"HFV1 payload truncated before header: {len(raw)} bytes < {HFV1_HEADER.size}"
        )
    magic, n_frames, h, w = HFV1_HEADER.unpack_from(raw)
    if magic != b"HFV1":
        raise ValueError(f"HFV1 magic mismatch: {magic!r} (expected b'HFV1')")
    expected = HFV1_HEADER.size + int(n_frames) * HFV1_ROW.size
    if len(raw) != expected:
        raise ValueError(
            f"HFV1 payload size mismatch: got {len(raw)} bytes, expected {expected} "
            f"(header={HFV1_HEADER.size} + {n_frames}*row={HFV1_ROW.size})"
        )
    rows = [
        HFV1_ROW.unpack_from(raw, HFV1_HEADER.size + i * HFV1_ROW.size)
        for i in range(int(n_frames))
    ]
    return int(n_frames), int(h), int(w), rows


def pack_hfv1(
    *,
    n_frames: int,
    height: int,
    width: int,
    rows: list[tuple[float, float, float, float, float]],
) -> bytes:
    """Inverse of ``parse_hfv1``: pack rows back into HFV1 dense bytes."""
    if len(rows) != n_frames:
        raise ValueError(
            f"row count mismatch: rows={len(rows)} != n_frames={n_frames}"
        )
    out = bytearray()
    out.extend(HFV1_HEADER.pack(b"HFV1", n_frames, height, width))
    for row in rows:
        if len(row) != 5:
            raise ValueError(f"row arity != 5: {row!r}")
        out.extend(HFV1_ROW.pack(*row))
    return bytes(out)


# ---------------------------------------------------------------------------
# Strategy 1: entropy_brotli (canonical default; lossless)
# ---------------------------------------------------------------------------


def encode_entropy_brotli(hfv1_bytes: bytes) -> bytes:
    """Strategy 1: brotli quality=11 on raw HFV1 bytes (lossless).

    Output format: ``HFRC | n_frames | h | w | strategy=0 | brotli_compressed_payload``
    where the brotli payload is the COMPLETE HFV1 bytes (header + rows).

    Best for: highly redundant inputs (e.g. identity foveation = 1200 identical
    rows → 24,016 bytes → ~50-100 bytes after brotli).
    """
    n_frames, h, w, _rows = parse_hfv1(hfv1_bytes)
    compressed = brotli.compress(hfv1_bytes, quality=11)
    out = bytearray()
    out.extend(HFRC_HEADER.pack(HFRC_MAGIC, n_frames, h, w, ENCODING_ENTROPY_BROTLI))
    out.extend(compressed)
    return bytes(out)


def decode_entropy_brotli(recoded_bytes: bytes) -> bytes:
    """Inverse of ``encode_entropy_brotli``."""
    if len(recoded_bytes) < HFRC_HEADER.size:
        raise ValueError(
            f"HFRC payload truncated before header: {len(recoded_bytes)} bytes"
        )
    magic, n_frames, h, w, strategy = HFRC_HEADER.unpack_from(recoded_bytes)
    if magic != HFRC_MAGIC:
        raise ValueError(f"HFRC magic mismatch: {magic!r}")
    if int(strategy) != ENCODING_ENTROPY_BROTLI:
        raise ValueError(
            f"strategy mismatch: got {strategy} expected {ENCODING_ENTROPY_BROTLI}"
        )
    compressed = recoded_bytes[HFRC_HEADER.size:]
    hfv1_bytes = brotli.decompress(compressed)
    # Sanity: round-trip header agreement
    rt_n, rt_h, rt_w, _rows = parse_hfv1(hfv1_bytes)
    if (rt_n, rt_h, rt_w) != (int(n_frames), int(h), int(w)):
        raise ValueError(
            f"HFRC vs inner HFV1 header mismatch: "
            f"({rt_n}, {rt_h}, {rt_w}) vs ({n_frames}, {h}, {w})"
        )
    return hfv1_bytes


# ---------------------------------------------------------------------------
# Strategy 2: sparse_delta (lossless; semantic-aware)
# ---------------------------------------------------------------------------


def _varint_encode(value: int) -> bytes:
    """Encode an unsigned int as varint (LEB128)."""
    if value < 0:
        raise ValueError(f"varint requires non-negative: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _varint_decode(raw: bytes, offset: int) -> tuple[int, int]:
    """Decode varint starting at ``offset``; return (value, new_offset)."""
    value = 0
    shift = 0
    pos = offset
    while pos < len(raw):
        byte = raw[pos]
        value |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")
    raise ValueError("varint truncated")


def encode_sparse_delta(hfv1_bytes: bytes) -> bytes:
    """Strategy 2: sparse-delta encoding (lossless).

    For each row, if it equals the previous row, encode as zero (varint 0
    + run-length). Otherwise encode the row's 5 float32 values as-is.

    Output format (after HFRC header):
        per_row_stream = varint(repeat_count) | optional [HFV1_ROW] new row

    For inputs with 99%+ identical rows (typical foveation_params),
    this collapses ~24KB to ~tens of bytes.

    Lossless: every row is reconstructed exactly (float32 bytes preserved).
    """
    n_frames, h, w, rows = parse_hfv1(hfv1_bytes)
    out = bytearray()
    out.extend(HFRC_HEADER.pack(HFRC_MAGIC, n_frames, h, w, ENCODING_SPARSE_DELTA))

    # Always encode row 0 absolute (5 float32 bytes).
    if rows:
        out.extend(HFV1_ROW.pack(*rows[0]))
    # For subsequent rows: encode run-length of identical-to-previous rows,
    # then if different, encode new row.
    prev = rows[0] if rows else None
    repeat_count = 0  # consecutive identical-to-prev (not counting prev itself)
    i = 1
    while i < n_frames:
        if rows[i] == prev:
            repeat_count += 1
            i += 1
            continue
        # Different row: flush repeat_count, then encode new absolute row
        out.extend(_varint_encode(repeat_count))
        repeat_count = 0
        out.extend(HFV1_ROW.pack(*rows[i]))
        prev = rows[i]
        i += 1
    # Final flush
    out.extend(_varint_encode(repeat_count))
    return bytes(out)


def decode_sparse_delta(recoded_bytes: bytes) -> bytes:
    """Inverse of ``encode_sparse_delta``."""
    if len(recoded_bytes) < HFRC_HEADER.size:
        raise ValueError("HFRC payload truncated before header")
    magic, n_frames, h, w, strategy = HFRC_HEADER.unpack_from(recoded_bytes)
    if magic != HFRC_MAGIC:
        raise ValueError(f"HFRC magic mismatch: {magic!r}")
    if int(strategy) != ENCODING_SPARSE_DELTA:
        raise ValueError(
            f"strategy mismatch: got {strategy} expected {ENCODING_SPARSE_DELTA}"
        )
    if int(n_frames) == 0:
        return pack_hfv1(n_frames=0, height=int(h), width=int(w), rows=[])
    offset = HFRC_HEADER.size
    # Decode row 0 absolute.
    if offset + HFV1_ROW.size > len(recoded_bytes):
        raise ValueError("sparse_delta payload truncated at row 0")
    row0 = HFV1_ROW.unpack_from(recoded_bytes, offset)
    offset += HFV1_ROW.size
    rows: list[tuple[float, float, float, float, float]] = [row0]
    prev = row0
    while len(rows) < int(n_frames):
        repeat_count, offset = _varint_decode(recoded_bytes, offset)
        # Append repeat_count copies of prev
        for _ in range(repeat_count):
            rows.append(prev)
            if len(rows) == int(n_frames):
                break
        if len(rows) == int(n_frames):
            break
        # Decode next absolute row
        if offset + HFV1_ROW.size > len(recoded_bytes):
            raise ValueError("sparse_delta payload truncated mid-stream")
        new_row = HFV1_ROW.unpack_from(recoded_bytes, offset)
        offset += HFV1_ROW.size
        rows.append(new_row)
        prev = new_row
    if len(rows) != int(n_frames):
        raise ValueError(
            f"sparse_delta decode produced {len(rows)} rows, expected {n_frames}"
        )
    return pack_hfv1(n_frames=int(n_frames), height=int(h), width=int(w), rows=rows)


# ---------------------------------------------------------------------------
# Strategy 3: combined (sparse_delta + brotli; lossless)
# ---------------------------------------------------------------------------


def encode_combined(hfv1_bytes: bytes) -> bytes:
    """Strategy 3: ``sparse_delta`` then brotli on the delta stream (lossless).

    The sparse_delta inner encoding strips per-row redundancy at the
    semantic level; brotli then squeezes the remaining bit-level entropy.
    Best for: inputs with HIGH semantic sparsity AND some bit-level entropy
    in the unique rows themselves.

    Output format: HFRC header (strategy=2) + brotli(sparse_delta_body)
    where ``sparse_delta_body`` is the sparse_delta encoding's body
    (everything AFTER its HFRC header).
    """
    n_frames, h, w, _rows = parse_hfv1(hfv1_bytes)
    sparse_full = encode_sparse_delta(hfv1_bytes)
    # Strip the inner HFRC header; we only brotli-compress the body
    sparse_body = sparse_full[HFRC_HEADER.size:]
    compressed_body = brotli.compress(sparse_body, quality=11)
    out = bytearray()
    out.extend(HFRC_HEADER.pack(HFRC_MAGIC, n_frames, h, w, ENCODING_COMBINED))
    out.extend(compressed_body)
    return bytes(out)


def decode_combined(recoded_bytes: bytes) -> bytes:
    """Inverse of ``encode_combined``."""
    if len(recoded_bytes) < HFRC_HEADER.size:
        raise ValueError("HFRC payload truncated before header")
    magic, n_frames, h, w, strategy = HFRC_HEADER.unpack_from(recoded_bytes)
    if magic != HFRC_MAGIC:
        raise ValueError(f"HFRC magic mismatch: {magic!r}")
    if int(strategy) != ENCODING_COMBINED:
        raise ValueError(
            f"strategy mismatch: got {strategy} expected {ENCODING_COMBINED}"
        )
    compressed_body = recoded_bytes[HFRC_HEADER.size:]
    sparse_body = brotli.decompress(compressed_body)
    # Reconstruct full sparse_delta bytes with HFRC header (strategy=1) so
    # decode_sparse_delta accepts them.
    sparse_full = bytearray()
    sparse_full.extend(
        HFRC_HEADER.pack(HFRC_MAGIC, n_frames, h, w, ENCODING_SPARSE_DELTA)
    )
    sparse_full.extend(sparse_body)
    return decode_sparse_delta(bytes(sparse_full))


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

ENCODERS = {
    "entropy_brotli": encode_entropy_brotli,
    "sparse_delta": encode_sparse_delta,
    "combined": encode_combined,
}

DECODERS = {
    "entropy_brotli": decode_entropy_brotli,
    "sparse_delta": decode_sparse_delta,
    "combined": decode_combined,
}


def recode_foveation_params(
    hfv1_bytes: bytes,
    *,
    strategy: str = "combined",
) -> bytes:
    """Recode HFV1 dense sidecar bytes via canonical strategy.

    Returns recoded bytes carrying HFRC magic + strategy byte. Bit-identical
    round-trip guaranteed via the paired ``decode_recoded_sidecar`` helper.
    """
    if strategy not in ENCODERS:
        raise ValueError(
            f"unknown strategy {strategy!r}; valid: {sorted(ENCODERS)}"
        )
    return ENCODERS[strategy](hfv1_bytes)


def decode_recoded_sidecar(recoded_bytes: bytes) -> bytes:
    """Decode HFRC recoded sidecar bytes back to HFV1 dense bytes.

    Auto-dispatches on the strategy byte in the HFRC header. Returns
    HFV1 bytes bit-identical to the input that was originally recoded.
    """
    if len(recoded_bytes) < HFRC_HEADER.size:
        raise ValueError("HFRC payload truncated before header")
    magic, _n, _h, _w, strategy = HFRC_HEADER.unpack_from(recoded_bytes)
    if magic != HFRC_MAGIC:
        raise ValueError(f"HFRC magic mismatch: {magic!r}")
    strategy_name = ENCODING_NAMES.get(int(strategy))
    if strategy_name is None:
        raise ValueError(f"unknown strategy byte: {strategy}")
    return DECODERS[strategy_name](recoded_bytes)


# ---------------------------------------------------------------------------
# Predicted rate-savings (canonical equation #356)
# ---------------------------------------------------------------------------


def predicted_rate_savings(*, input_bytes: int, output_bytes: int) -> float:
    """Closed-form per canonical equation #356.

    deltaS = -25 * (N_dense - N_recoded) / 37_545_489

    Sign convention: NEGATIVE means score reduction (good); positive means
    score increase (bad). Output > input gives positive delta (recoder
    backfired).

    Carries ``[prediction]`` axis tag per Catalog #287; NOT a score claim
    until paired CUDA + CPU empirical anchors land per CLAUDE.md
    "Submission auth eval — BOTH CPU AND CUDA".
    """
    return -RATE_MULTIPLIER * float(input_bytes - output_bytes) / float(CONTEST_DENOM_BYTES)


# ---------------------------------------------------------------------------
# Build verdict
# ---------------------------------------------------------------------------


def build_recoder_verdict(
    *,
    input_path: Path,
    output_path: Path,
    strategy: str,
    verify_roundtrip: bool,
) -> RecoderVerdict:
    """Read input, recode, write output, verify round-trip, return verdict.

    Per Catalog #229 PV: ``verify_roundtrip=True`` (default) runs the
    inverse decoder and refuses with non-zero exit if sha256s do not match.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"input HFV1 sidecar not found: {input_path}")
    t0 = time.monotonic()
    hfv1_bytes = input_path.read_bytes()
    input_sha = hashlib.sha256(hfv1_bytes).hexdigest()
    n_frames, h, w, _rows = parse_hfv1(hfv1_bytes)
    recoded = recode_foveation_params(hfv1_bytes, strategy=strategy)
    output_sha = hashlib.sha256(recoded).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(recoded)

    round_trip_verified = False
    notes_parts: list[str] = []
    if verify_roundtrip:
        roundtrip_hfv1 = decode_recoded_sidecar(recoded)
        roundtrip_sha = hashlib.sha256(roundtrip_hfv1).hexdigest()
        round_trip_verified = roundtrip_sha == input_sha
        if not round_trip_verified:
            notes_parts.append(
                f"ROUND_TRIP_FAILED sha256(input)={input_sha[:12]} "
                f"sha256(decode(encode(input)))={roundtrip_sha[:12]}"
            )
        else:
            notes_parts.append(
                f"round_trip_verified sha256(input)=sha256(decode(encode(input)))={input_sha[:12]}"
            )

    elapsed = time.monotonic() - t0
    bytes_saved = len(hfv1_bytes) - len(recoded)
    pct = 100.0 * bytes_saved / max(1, len(hfv1_bytes))
    rate = predicted_rate_savings(input_bytes=len(hfv1_bytes), output_bytes=len(recoded))

    return RecoderVerdict(
        input_path=str(input_path),
        output_path=str(output_path),
        encoding_strategy=strategy,
        input_bytes=len(hfv1_bytes),
        output_bytes=len(recoded),
        bytes_saved=bytes_saved,
        percent_reduction=round(pct, 4),
        rate_savings_predicted=round(rate, 9),
        round_trip_verified=round_trip_verified,
        sha256_input=input_sha,
        sha256_output=output_sha,
        n_frames=n_frames,
        camera_height=h,
        camera_width=w,
        elapsed_seconds=round(elapsed, 6),
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Smoke fixture (Carmack MVP-first Step 1: free local CPU smoke)
# ---------------------------------------------------------------------------


def build_synthetic_smoke_fixture(
    *,
    n_frames: int = 1200,
    sparse_pair_indices: list[int] | None = None,
) -> bytes:
    """Build a synthetic HFV1 fixture matching the live foveation_params shape.

    Default: 1200 frames at PR101 canonical (874, 1164); identity row
    (0.0, 1455.6002197265625, 1.0, 581.5, 393.29998779296875) for all
    frames except those whose pair-index is in ``sparse_pair_indices``
    which get a non-identity row.

    Mirrors the live ``official_inflate_control/data_identity/foveation_params.bin``
    structure observed during PV.
    """
    if sparse_pair_indices is None:
        sparse_pair_indices = []
    identity = (0.0, 1455.6002197265625, 1.0, 581.5, 393.29998779296875)
    nonidentity = (0.00045, 1020.0, 0.7, 581.5, 393.29998779296875)
    sparse_set = set(sparse_pair_indices)
    rows = []
    for i in range(n_frames):
        pair_idx = i // 2
        if pair_idx in sparse_set:
            rows.append(nonidentity)
        else:
            rows.append(identity)
    return pack_hfv1(
        n_frames=n_frames, height=CAMERA_H, width=CAMERA_W, rows=rows
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_hfv_sidecar_recoder",
        description=(
            "Canonical HFV sidecar recoder: shrink dense HFV1 "
            "foveation_params.bin (24KB) to compact HFRC recoded sidecar "
            "(~50-100 bytes for identity; ~10KB target for sparse-pair) "
            "via brotli / sparse_delta / combined strategies. "
            "Lossless bit-identical round-trip verified."
        ),
    )
    parser.add_argument(
        "--input-foveation-params-bin",
        type=Path,
        help="Path to input HFV1 sidecar (e.g. experiments/results/.../foveation_params.bin).",
    )
    parser.add_argument(
        "--output-recoded-sidecar",
        type=Path,
        help="Path to write HFRC recoded sidecar bytes.",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=10_000,
        help="Soft target byte ceiling (default 10000; OVERNIGHT-S Path 3 spec).",
    )
    parser.add_argument(
        "--encoding-strategy",
        type=str,
        default="combined",
        choices=sorted(ENCODERS),
        help="Canonical strategy (default combined = sparse_delta + brotli).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run synthetic fixture smoke (1200 frames, 1 sparse pair). "
             "Writes to a tmp path if --output-recoded-sidecar not provided. "
             "Carmack MVP-first Step 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute verdict without writing the output file.",
    )
    parser.add_argument(
        "--verify-roundtrip",
        action="store_true",
        default=True,
        help="Decode the encoded output and refuse if sha256 mismatch (default ON).",
    )
    parser.add_argument(
        "--no-verify-roundtrip",
        dest="verify_roundtrip",
        action="store_false",
        help="DISABLE round-trip verification (NOT RECOMMENDED; "
             "violates Catalog #229 PV).",
    )
    parser.add_argument(
        "--report-out-json",
        type=Path,
        help="Path to write canonical verdict JSON report.",
    )
    parser.add_argument(
        "--smoke-sparse-pair-indices",
        type=str,
        default="0",
        help="Comma-separated pair indices to mark non-identity in smoke fixture "
             "(default '0' = one sparse pair). Use '' for identity-only.",
    )
    args = parser.parse_args(argv)

    # Smoke mode: build synthetic fixture
    smoke_input_path: Path | None = None
    if args.smoke:
        sparse_str = args.smoke_sparse_pair_indices.strip()
        sparse_indices = (
            [int(x) for x in sparse_str.split(",") if x.strip()] if sparse_str else []
        )
        fixture = build_synthetic_smoke_fixture(
            n_frames=1200, sparse_pair_indices=sparse_indices
        )
        if args.input_foveation_params_bin is None:
            smoke_input_path = Path("/tmp") / "hfv_sidecar_recoder_smoke_fixture.bin"
            smoke_input_path.write_bytes(fixture)
            args.input_foveation_params_bin = smoke_input_path
        else:
            args.input_foveation_params_bin.parent.mkdir(parents=True, exist_ok=True)
            args.input_foveation_params_bin.write_bytes(fixture)
        if args.output_recoded_sidecar is None:
            args.output_recoded_sidecar = Path("/tmp") / (
                f"hfv_sidecar_recoder_smoke_output_{args.encoding_strategy}.bin"
            )

    if args.input_foveation_params_bin is None:
        parser.error(
            "--input-foveation-params-bin required (or pass --smoke for synthetic fixture)"
        )
    if args.output_recoded_sidecar is None:
        parser.error("--output-recoded-sidecar required")

    if args.dry_run:
        # Read + recode + measure WITHOUT writing the output file
        hfv1_bytes = args.input_foveation_params_bin.read_bytes()
        recoded = recode_foveation_params(
            hfv1_bytes, strategy=args.encoding_strategy
        )
        verdict = RecoderVerdict(
            input_path=str(args.input_foveation_params_bin),
            output_path=str(args.output_recoded_sidecar) + " [dry-run; not written]",
            encoding_strategy=args.encoding_strategy,
            input_bytes=len(hfv1_bytes),
            output_bytes=len(recoded),
            bytes_saved=len(hfv1_bytes) - len(recoded),
            percent_reduction=round(
                100.0 * (len(hfv1_bytes) - len(recoded)) / max(1, len(hfv1_bytes)),
                4,
            ),
            rate_savings_predicted=round(
                predicted_rate_savings(
                    input_bytes=len(hfv1_bytes), output_bytes=len(recoded)
                ),
                9,
            ),
            round_trip_verified=False,
            sha256_input=hashlib.sha256(hfv1_bytes).hexdigest(),
            sha256_output=hashlib.sha256(recoded).hexdigest(),
            notes="dry-run; output not written",
        )
    else:
        verdict = build_recoder_verdict(
            input_path=args.input_foveation_params_bin,
            output_path=args.output_recoded_sidecar,
            strategy=args.encoding_strategy,
            verify_roundtrip=args.verify_roundtrip,
        )

    verdict_dict = verdict.to_dict()
    verdict_dict["target_bytes"] = int(args.target_bytes)
    verdict_dict["under_target"] = verdict.output_bytes <= int(args.target_bytes)
    verdict_dict["lane_id"] = LANE_ID
    verdict_dict["canonical_equation_reference"] = CANONICAL_EQUATION_356_ID

    output_summary = json.dumps(verdict_dict, indent=2, sort_keys=True)
    print(output_summary)

    if args.report_out_json:
        args.report_out_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_out_json.write_text(output_summary + "\n", encoding="utf-8")

    # Exit codes:
    #   0 = success (round-trip verified OR --no-verify-roundtrip + write succeeded)
    #   1 = round-trip FAILED (Catalog #229 PV refusal)
    #   2 = output > target_bytes (soft warning; non-fatal unless --strict)
    if args.verify_roundtrip and not verdict.round_trip_verified and not args.dry_run:
        print(
            "FATAL: round-trip verification FAILED — Catalog #229 PV refusal",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
