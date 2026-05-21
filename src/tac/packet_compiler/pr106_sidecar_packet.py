# SPDX-License-Identifier: MIT
"""PR106 sidecar PacketIR parser and identity emitter.

This module owns the byte-level wrapper used by PR106 latent-sidecar variants:

``magic(0xFE) | format_id | pr106_len(u32le) | pr106_bytes | sidecar...``

It is intentionally scorer-free and torch-free. The compiler layer can use it
to prove that a transform consumes and re-emits the same bytes before a new
grammar or sidecar mutation is allowed near exact eval.
"""

from __future__ import annotations

import hashlib
import io
import math
import string
import struct
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np

from tac.packet_compiler.pr101_sidecar_grammar import (
    RankedSidecarSchema,
    _huff_length_vector_count,
    decode_ranked_no_op_sidecar,
    encode_ranked_no_op_sidecar,
)

PR106_SIDECAR_MAGIC = 0xFE
PR106_SIDECAR_FORMAT_BROTLI = 0x01
PR106_SIDECAR_FORMAT_PR101_GRAMMAR = 0x02
PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED = 0x04
PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED = 0x05
PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED = 0x06
PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED = 0x07
PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED = 0x08
PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED = 0x09
PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED = (
    0x0A
)
PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED = 0x0B
PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED = (
    0x0C
)
PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA = 0x0D
PR106_SUPPORTED_SIDECAR_FORMATS = (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
)
PR106_HEADERLESS_INNER_FIRST_BYTE = 0xFF
PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES = 169_974
PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES = 169_950
PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES = 15_776
PR106_HDM8_HLM2_DECODER_MAGIC = b"HDM8"
PR106_HDM9_HLM2_DECODER_MAGIC = b"HDM9"
PR106_HDM8_HLM2_LATENT_MAGIC = b"HLM2"
PR106_HDM9_HLM3_LATENT_MAGIC = b"HLM3"
PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES = PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES - 2
PR106_HDM9_SCALE_HIGH_BASE = 0x3B
PR106_HDM9_SCALE_COUNT = 28
PR106_HDM9_SCALE_LOW3_BYTES = 3 * PR106_HDM9_SCALE_COUNT
PR106_HDM9_SCALE_HIGH_MASK_BYTES = (PR106_HDM9_SCALE_COUNT + 7) // 8
PR106_HDM9_Q_BROTLI_CHUNK_BYTES = (130_887, 2_769, 4_397, 31_805)
PR106_LATENT_N_PAIRS = 600
PR106_LATENT_N_DIMS = 28
PR106_DEFAULT_MEMBER_NAME = "0.bin"
PR106_ALLOWED_SINGLE_MEMBER_NAMES = (PR106_DEFAULT_MEMBER_NAME, "x")
PR106_PACKET_IR_SECTION_HASH_DOMAIN = (
    "pr106_sidecar_packet_ir_emitted_member_payload_section_bytes_v1"
)
PR106_NO_OP_DIM = 255
PR106_PR101_FIXED_META_NOOP_COUNT = 0
PR106_PR101_FIXED_META_DIM_BYTES = 375
PR106_PR101_FIXED_META_RANK_BYTES = 1
PR106_PR101_FIXED_META_NOOP_RANK_BYTES = 1
PR106_PR101_FIXED_META_RANK_BYTE = b"\x00"
PR106_PR101_FIXED_META_NOOP_RANK_BYTE = b"\x00"
PR106_PR101_FIXED_META_PAYLOAD_BYTES = 526
PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES = (
    PR106_PR101_FIXED_META_PAYLOAD_BYTES - PR106_PR101_FIXED_META_NOOP_RANK_BYTES
)
PR106_PR101_EXACT_RADIX_DIM_BYTES = 361
PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES = (
    PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
    - PR106_PR101_FIXED_META_DIM_BYTES
    + PR106_PR101_EXACT_RADIX_DIM_BYTES
)
PR106_FORMAT0D_EXTRA_FRAMING_META_BYTES = 6
PR106_PR101_RANKED_SCHEMA = RankedSidecarSchema(
    n_pairs=600,
    n_dims=28,
    deltas=(-2, -1, 1, 2),
    huff_min_len=2,
    huff_max_len=8,
    no_op_sentinel=PR106_NO_OP_DIM,
)
PR106_PR101_FIXED_META_FIELDS = {
    "noop_count": PR106_PR101_FIXED_META_NOOP_COUNT,
    "dim_bytes": PR106_PR101_FIXED_META_DIM_BYTES,
    "rank_bytes": PR106_PR101_FIXED_META_RANK_BYTES,
    "noop_rank_bytes": PR106_PR101_FIXED_META_NOOP_RANK_BYTES,
    "rank_byte_hex": PR106_PR101_FIXED_META_RANK_BYTE.hex(),
    "schema": {
        "n_pairs": PR106_LATENT_N_PAIRS,
        "n_dims": PR106_LATENT_N_DIMS,
        "deltas": tuple(int(value) for value in PR106_PR101_RANKED_SCHEMA.deltas),
        "huff_min_len": PR106_PR101_RANKED_SCHEMA.huff_min_len,
        "huff_max_len": PR106_PR101_RANKED_SCHEMA.huff_max_len,
        "no_op_sentinel": PR106_PR101_RANKED_SCHEMA.no_op_sentinel,
    },
}


@dataclass(frozen=True)
class StoredZipMember:
    """Single stored ZIP member plus enough metadata for byte-identical emit."""

    name: str
    payload: bytes
    date_time: tuple[int, int, int, int, int, int]
    external_attr: int
    create_system: int
    flag_bits: int
    comment: bytes
    extra: bytes
    archive_comment: bytes = b""


@dataclass(frozen=True)
class PR106SidecarPacket:
    """Typed PR106 sidecar wrapper.

    ``framing_meta`` is ``None`` for format ``0x01``. For format ``0x02`` it is
    the six-byte PR101 grammar footer:
    ``noop_count(u16le) | dim_bytes(u16le) | rank_bytes(u8) | noop_rank_bytes(u8)``.
    """

    format_id: int
    pr106_bytes: bytes
    sidecar_payload: bytes
    framing_meta: bytes | None = None
    extra_sidecar_payload: bytes = b""
    extra_framing_meta: bytes | None = None

    @property
    def sidecar_kind(self) -> str:
        if self.format_id == PR106_SIDECAR_FORMAT_BROTLI:
            return "brotli_dim_delta"
        if self.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
            return "pr101_ranked_no_op"
        if self.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
            return "pr101_ranked_no_op_rank_elided"
        if self.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED:
            return "pr101_ranked_no_op_fixed_meta_rank_elided"
        if (
            self.format_id
            == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        ):
            return "pr101_ranked_no_op_implicit_len_fixed_meta_rank_elided"
        if (
            self.format_id
            == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        ):
            return "pr101_ranked_no_op_headerless_implicit_len_fixed_meta_rank_elided"
        if (
            self.format_id
            == (
                PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
            )
        ):
            return "pr101_ranked_no_op_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided"
        if (
            self.format_id
            == (
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
            )
        ):
            return "pr101_ranked_no_op_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided"
        if (
            self.format_id
            == (
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
            )
        ):
            return "pr101_ranked_no_op_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided"
        if (
            self.format_id
            == (
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
            )
        ):
            return "pr101_ranked_no_op_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided"
        if (
            self.format_id
            == (
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
            )
        ):
            return (
                "pr101_ranked_no_op_hdm9_hlm3_magicless_exact_radix_dim_"
                "fixed_meta_noop_rank_elided"
            )
        if self.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
            return "format0c_base_plus_pr101_ranked_no_op_extra"
        return f"unknown_0x{self.format_id:02x}"


@dataclass(frozen=True)
class PR106SidecarMutation:
    """Description of a valid semantic sidecar mutation.

    The mutation is intentionally limited to the correction sidecar. It keeps
    the inner PR106 payload byte-identical and changes one per-pair correction
    in a way the runtime sidecar decoder can consume. This is a no-op-proof
    tool, not a score candidate generator.
    """

    section_name: str
    pair_index: int
    format_id: int
    old_dim: int
    new_dim: int
    old_delta_q: int
    new_delta_q: int
    old_delta_index: int | None = None
    new_delta_index: int | None = None


@dataclass(frozen=True)
class PR106SidecarRecodeCandidate:
    """Lossless alternative sidecar byte grammar candidate.

    These candidates are compiler-planning artifacts. They prove that the
    candidate byte string decodes to the same ``(dim, delta_q)`` correction
    arrays as the source sidecar, but they do not imply runtime support, archive
    emission, or score movement.
    """

    name: str
    encoded_bytes: bytes
    decoded_dims: np.ndarray
    decoded_delta_q: np.ndarray
    sidecar_format_id: int | None
    framing_meta_bytes: bytes = b""
    runtime_decoder_implemented: bool = False
    notes: tuple[str, ...] = ()

    @property
    def charged_bytes(self) -> int:
        """Bytes that would be charged inside the PR106 sidecar wrapper."""

        return len(self.encoded_bytes) + len(self.framing_meta_bytes)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_expected_sha256(expected_sha256: str | None) -> tuple[str | None, bool | None]:
    """Return normalized expected SHA-256 text plus a well-formed flag."""

    if expected_sha256 is None:
        return None, None
    canonical = expected_sha256.strip().lower()
    well_formed = (
        len(canonical) == 64
        and all(char in string.hexdigits.lower() for char in canonical)
    )
    return canonical, well_formed


def _sidecar_charged_bytes(packet: PR106SidecarPacket) -> int:
    base_bytes = len(packet.sidecar_payload) + (
        0 if packet.framing_meta is None else len(packet.framing_meta)
    )
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        return (
            base_bytes
            + 2
            + len(packet.extra_sidecar_payload)
            + (0 if packet.extra_framing_meta is None else len(packet.extra_framing_meta))
        )
    return base_bytes


def canonicalize_brotli_dim_delta_sidecar_arrays(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    n_dims: int = PR106_LATENT_N_DIMS,
    no_op_dim: int = PR106_NO_OP_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and canonicalize PR106 ``(dim, delta_q)`` correction arrays."""

    dims_in = np.asarray(dim_arr)
    deltas_in = np.asarray(delta_q_arr)
    if dims_in.ndim != 1 or deltas_in.ndim != 1:
        raise ValueError(
            f"dim/delta arrays must be 1D; got {dims_in.shape}/{deltas_in.shape}"
        )
    if dims_in.shape != deltas_in.shape:
        raise ValueError(f"dim/delta shapes differ: {dims_in.shape} vs {deltas_in.shape}")
    if dims_in.dtype.kind not in {"i", "u"}:
        raise TypeError(f"dim_arr must be integer typed; got {dims_in.dtype}")
    if deltas_in.dtype.kind not in {"i", "u"}:
        raise TypeError(f"delta_q_arr must be integer typed; got {deltas_in.dtype}")
    if dims_in.size > 0xFFFF:
        raise ValueError(f"sidecar has too many pairs for u16 length: {dims_in.size}")
    dims_i64 = dims_in.astype(np.int64, copy=False)
    deltas_i64 = deltas_in.astype(np.int64, copy=False)
    if dims_i64.size and (int(dims_i64.min()) < 0 or int(dims_i64.max()) > no_op_dim):
        raise ValueError(
            f"dim values must be in [0, {n_dims}) or {no_op_dim}; "
            f"got min={int(dims_i64.min())} max={int(dims_i64.max())}"
        )
    invalid_dim = (dims_i64 >= n_dims) & (dims_i64 != no_op_dim)
    if bool(np.any(invalid_dim)):
        bad = int(dims_i64[np.flatnonzero(invalid_dim)[0]])
        raise ValueError(f"invalid correction dim {bad}; expected [0, {n_dims}) or {no_op_dim}")
    if deltas_i64.size and (
        int(deltas_i64.min()) < -128 or int(deltas_i64.max()) > 127
    ):
        raise ValueError(
            "delta_q values must fit int8; "
            f"got min={int(deltas_i64.min())} max={int(deltas_i64.max())}"
        )
    no_op_with_delta = (dims_i64 == no_op_dim) & (deltas_i64 != 0)
    if bool(np.any(no_op_with_delta)):
        idx = int(np.flatnonzero(no_op_with_delta)[0])
        raise ValueError(f"pair {idx} has no-op dim {no_op_dim} but nonzero delta")
    dims = dims_i64.astype(np.uint8)
    deltas = deltas_i64.astype(np.int8)
    dims = np.where(deltas == 0, no_op_dim, dims).astype(np.uint8)
    return dims, deltas


def encode_brotli_dim_delta_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    quality: int = 11,
    n_dims: int = PR106_LATENT_N_DIMS,
    no_op_dim: int = PR106_NO_OP_DIM,
) -> bytes:
    """Encode PR106 format-0x01 sidecar payload.

    Layout matches the original PR100/PR106 sidecar grammar:
    ``brotli(u16 n_pairs | repeated u8 dim | i8 delta_q)``. Zero deltas are
    canonicalized to the ``no_op_dim`` sentinel before compression.
    """

    dims, deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        dim_arr,
        delta_q_arr,
        n_dims=n_dims,
        no_op_dim=no_op_dim,
    )
    payload = struct.pack("<H", int(dims.size)) + np.stack(
        [dims, deltas.view(np.uint8)], axis=1
    ).tobytes()
    return brotli.compress(payload, quality=quality)


def decode_brotli_dim_delta_sidecar_payload(
    payload: bytes,
    *,
    n_dims: int = PR106_LATENT_N_DIMS,
    no_op_dim: int = PR106_NO_OP_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PR106 format-0x01 sidecar payload into canonical arrays."""

    raw = brotli.decompress(payload)
    if len(raw) < 2:
        raise ValueError("brotli sidecar decompressed payload too short")
    n_pairs = struct.unpack_from("<H", raw, 0)[0]
    expected = 2 + 2 * n_pairs
    if len(raw) != expected:
        raise ValueError(
            f"brotli sidecar decompressed payload has trailing/truncated bytes: "
            f"got={len(raw)} expected={expected}"
        )
    arr = np.frombuffer(raw[2:], dtype=np.uint8).reshape(n_pairs, 2)
    return canonicalize_brotli_dim_delta_sidecar_arrays(
        arr[:, 0].copy(),
        arr[:, 1].copy().view(np.int8),
        n_dims=n_dims,
        no_op_dim=no_op_dim,
    )


def encode_pr101_ranked_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes]:
    """Encode PR106 corrections with the PR101 ranked/no-op grammar.

    Returns ``(sidecar_payload, framing_meta)`` where ``framing_meta`` is the
    six-byte trailer consumed by PR106 sidecar format ``0x02``.
    """

    dims_u8, deltas_i8 = canonicalize_brotli_dim_delta_sidecar_arrays(
        dim_arr,
        delta_q_arr,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    if int(dims_u8.size) != int(schema.n_pairs):
        raise ValueError(f"schema expects {schema.n_pairs} pairs; got {dims_u8.size}")
    dims = dims_u8.astype(np.int64)
    deltas = deltas_i8.astype(np.int64)
    valid = dims != schema.no_op_sentinel
    delta_to_index = {int(value): index for index, value in enumerate(schema.deltas)}
    delta_indices = np.zeros(schema.n_pairs, dtype=np.int64)
    for value in sorted({int(item) for item in deltas[valid].tolist()}):
        if value not in delta_to_index:
            raise ValueError(
                f"delta {value} not in ranked sidecar schema vocabulary "
                f"{list(schema.deltas)}"
            )
        delta_indices[(deltas == value) & valid] = delta_to_index[value]

    n_valid = int(valid.sum())
    noop_count = int(schema.n_pairs - n_valid)
    dim_bits = max(1, n_valid * math.ceil(math.log2(max(schema.n_dims, 2))))
    dim_bytes = (dim_bits + 7) // 8
    total_length_vectors = _huff_length_vector_count(
        0,
        schema.kraft_total,
        n_symbols=len(schema.deltas),
        huff_min_len=schema.huff_min_len,
        huff_max_len=schema.huff_max_len,
    )
    rank_bits = max(1, math.ceil(math.log2(max(total_length_vectors, 2))))
    rank_bytes = (rank_bits + 7) // 8
    noop_total = max(math.comb(schema.n_pairs, noop_count), 1)
    noop_rank_bits = max(1, math.ceil(math.log2(noop_total)))
    noop_rank_bytes = (noop_rank_bits + 7) // 8
    payload = encode_ranked_no_op_sidecar(
        dims=dims,
        delta_indices=delta_indices,
        schema=schema,
        dim_bytes=dim_bytes,
        noop_rank_bytes=noop_rank_bytes,
    )
    framing_meta = struct.pack("<HHBB", noop_count, dim_bytes, rank_bytes, noop_rank_bytes)
    decoded_dims, decoded_delta_indices = _decode_pr101_ranked_sidecar_payload(
        payload,
        framing_meta,
        schema=schema,
    )
    decoded_deltas = np.zeros(schema.n_pairs, dtype=np.int64)
    decoded_valid = decoded_dims != schema.no_op_sentinel
    decoded_deltas[decoded_valid] = np.asarray(schema.deltas, dtype=np.int64)[
        decoded_delta_indices[decoded_valid]
    ]
    expected_dims, expected_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        dims,
        deltas,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    if not np.array_equal(decoded_dims.astype(np.uint8), expected_dims):
        raise ValueError("ranked sidecar encode/decode dim mismatch")
    if not np.array_equal(decoded_deltas.astype(np.int8), expected_deltas):
        raise ValueError("ranked sidecar encode/decode delta mismatch")
    return payload, framing_meta


def decode_pr101_ranked_sidecar_payload_to_dim_delta(
    payload: bytes,
    framing_meta: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PR106 format-0x02 sidecar payload into ``(dim, delta_q)`` arrays."""

    dims, delta_indices = _decode_pr101_ranked_sidecar_payload(
        payload,
        framing_meta,
        schema=schema,
    )
    deltas = np.zeros(schema.n_pairs, dtype=np.int64)
    valid = dims != schema.no_op_sentinel
    deltas[valid] = np.asarray(schema.deltas, dtype=np.int64)[delta_indices[valid]]
    return canonicalize_brotli_dim_delta_sidecar_arrays(
        dims,
        deltas,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )


def _rank_elided_expected_payload_bytes(
    *,
    noop_count: int,
    dim_bytes: int,
    noop_rank_bytes: int,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> int:
    if not (0 <= int(noop_count) <= int(schema.n_pairs)):
        raise ValueError(f"noop_count must be in [0, {schema.n_pairs}]")
    n_valid = int(schema.n_pairs) - int(noop_count)
    expected_huff_bytes = (n_valid * int(schema.huff_min_len) + 7) // 8
    return int(dim_bytes) + expected_huff_bytes + int(noop_rank_bytes)


def reexpand_pr101_rank_elided_sidecar_payload(
    payload: bytes,
    framing_meta: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes]:
    """Reconstruct the equivalent format-0x02 PR101 payload/meta from format 0x04."""

    if len(framing_meta) != 5:
        raise ValueError(
            f"format_id=0x04 framing_meta must be 5 bytes; got {len(framing_meta)}"
        )
    noop_count, dim_bytes, noop_rank_bytes = struct.unpack("<HHB", framing_meta)
    expected_payload_bytes = _rank_elided_expected_payload_bytes(
        noop_count=int(noop_count),
        dim_bytes=int(dim_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
        schema=schema,
    )
    if len(payload) != expected_payload_bytes:
        raise ValueError(
            "format_id=0x04 rank-elided payload length mismatch: "
            f"got {len(payload)} bytes; expected {expected_payload_bytes}"
        )
    source_payload = payload[:dim_bytes] + b"\x00" + payload[dim_bytes:]
    source_meta = struct.pack("<HHBB", noop_count, dim_bytes, 1, noop_rank_bytes)
    return source_payload, source_meta


def _fixed_meta_framing_bytes() -> bytes:
    return struct.pack(
        "<HHBB",
        PR106_PR101_FIXED_META_NOOP_COUNT,
        PR106_PR101_FIXED_META_DIM_BYTES,
        PR106_PR101_FIXED_META_RANK_BYTES,
        PR106_PR101_FIXED_META_NOOP_RANK_BYTES,
    )


def _exact_radix_fixed_meta_framing_bytes() -> bytes:
    return struct.pack(
        "<HHBB",
        PR106_PR101_FIXED_META_NOOP_COUNT,
        PR106_PR101_EXACT_RADIX_DIM_BYTES,
        PR106_PR101_FIXED_META_RANK_BYTES,
        PR106_PR101_FIXED_META_NOOP_RANK_BYTES,
    )


def _validate_fixed_meta_schema(schema: RankedSidecarSchema) -> None:
    expected = PR106_PR101_RANKED_SCHEMA
    if (
        int(schema.n_pairs) != int(expected.n_pairs)
        or int(schema.n_dims) != int(expected.n_dims)
        or tuple(int(value) for value in schema.deltas)
        != tuple(int(value) for value in expected.deltas)
        or int(schema.huff_min_len) != int(expected.huff_min_len)
        or int(schema.huff_max_len) != int(expected.huff_max_len)
        or int(schema.no_op_sentinel) != int(expected.no_op_sentinel)
    ):
        raise ValueError("format_id=0x05 fixed meta requires the canonical PR106 PR101 schema")


def reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes]:
    """Reconstruct format-0x02 PR101 payload/meta from fixed-meta format 0x05.

    Format ``0x05`` is a byte-closed specialization for the current PR106/R2
    no-op-free ranked sidecar. The archive pays only the dim-radix payload,
    Huffman bits, and noop-rank byte; the fixed grammar metadata lives in the
    runtime schema.
    """

    _validate_fixed_meta_schema(schema)
    expected_payload_bytes = _rank_elided_expected_payload_bytes(
        noop_count=PR106_PR101_FIXED_META_NOOP_COUNT,
        dim_bytes=PR106_PR101_FIXED_META_DIM_BYTES,
        noop_rank_bytes=PR106_PR101_FIXED_META_NOOP_RANK_BYTES,
        schema=schema,
    )
    if expected_payload_bytes != PR106_PR101_FIXED_META_PAYLOAD_BYTES:
        raise ValueError(
            "format_id=0x05/0x06 fixed-meta payload constant is stale: "
            f"got {PR106_PR101_FIXED_META_PAYLOAD_BYTES}, expected {expected_payload_bytes}"
        )
    if len(payload) != expected_payload_bytes:
        raise ValueError(
            "format_id=0x05 fixed-meta rank-elided payload length mismatch: "
            f"got {len(payload)} bytes; expected {expected_payload_bytes}"
        )
    dim_bytes = PR106_PR101_FIXED_META_DIM_BYTES
    source_payload = payload[:dim_bytes] + PR106_PR101_FIXED_META_RANK_BYTE + payload[dim_bytes:]
    return source_payload, _fixed_meta_framing_bytes()


def reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes]:
    """Reconstruct format-0x02 PR101 payload/meta from format 0x0A sidecar bytes.

    Format ``0x0A`` specializes the fixed-meta/rank-elided sidecar one byte
    further by eliding the no-op combination rank. The current PR106/R2
    sidecar has ``noop_count=0``, so that rank is necessarily the one-byte
    zero value and can be derived from the committed runtime grammar.
    """

    _validate_fixed_meta_schema(schema)
    if PR106_PR101_FIXED_META_NOOP_COUNT != 0:
        raise ValueError("noop-rank elision requires fixed noop_count=0")
    expected_payload_bytes = PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
    if len(payload) != expected_payload_bytes:
        raise ValueError(
            "format_id=0x0A fixed-meta noop-rank-elided payload length mismatch: "
            f"got {len(payload)} bytes; expected {expected_payload_bytes}"
        )
    return reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(
        payload + PR106_PR101_FIXED_META_NOOP_RANK_BYTE,
        schema=schema,
    )


def _encode_exact_radix_dim_payload(
    dims: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> bytes:
    """Pack no-op-free PR101 dims as one tight base-``n_dims`` integer."""

    _validate_fixed_meta_schema(schema)
    values = np.asarray(dims, dtype=np.int64).reshape(-1)
    if values.size != schema.n_pairs:
        raise ValueError(
            f"exact-radix dim payload expects {schema.n_pairs} dims; got {values.size}"
        )
    if bool(np.any(values < 0)) or bool(np.any(values >= schema.n_dims)):
        raise ValueError("exact-radix dim payload requires no-op-free dims in [0, n_dims)")
    value = 0
    for dim in reversed(values.tolist()):
        value = value * int(schema.n_dims) + int(dim)
    return value.to_bytes(PR106_PR101_EXACT_RADIX_DIM_BYTES, "little")


def reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes]:
    """Reconstruct format-0x02 bytes from format-0x0C exact-radix sidecar.

    Format ``0x0C`` keeps the same fixed PR106/R2 sidecar semantics as 0x0B,
    but stores the 600 no-op-free dim symbols as one base-28 integer instead of
    the wider 375-byte PR101 dim container.
    """

    _validate_fixed_meta_schema(schema)
    if PR106_PR101_FIXED_META_NOOP_COUNT != 0:
        raise ValueError("exact-radix/noop-rank elision requires fixed noop_count=0")
    if len(payload) != PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES:
        raise ValueError(
            "format_id=0x0C exact-radix fixed-meta noop-rank-elided payload "
            "length mismatch: "
            f"got {len(payload)} bytes; "
            f"expected {PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES}"
        )
    dim_bytes = PR106_PR101_EXACT_RADIX_DIM_BYTES
    source_payload = (
        payload[:dim_bytes]
        + PR106_PR101_FIXED_META_RANK_BYTE
        + payload[dim_bytes:]
        + PR106_PR101_FIXED_META_NOOP_RANK_BYTE
    )
    return source_payload, _exact_radix_fixed_meta_framing_bytes()


def _decompress_one_brotli_stream(payload: bytes, *, label: str) -> tuple[bytes, int]:
    """Decode exactly one Brotli stream and return ``(raw, bytes_consumed)``."""

    decoder = brotli.Decompressor()
    chunks: list[bytes] = []
    cursor = 0
    try:
        while cursor < len(payload) and not decoder.is_finished():
            chunks.append(decoder.process(payload[cursor : cursor + 1]))
            cursor += 1
    except brotli.error as exc:
        raise ValueError(f"{label} Brotli stream decode failed") from exc
    if not decoder.is_finished():
        raise ValueError(f"{label} Brotli stream truncated")
    return b"".join(chunks), cursor


def encode_hlm3_latents_from_hlm2_payload(hlm2_latent_payload: bytes) -> bytes:
    """Return the HLM3 byte-closed latent recode for a fixed HLM2 payload.

    HLM3 removes HLM2's two-byte low-Brotli length prefix. The decoder finds
    the end of the low-byte Brotli stream from the Brotli frame itself, then
    reads the fixed 112-byte fp16 latent metadata and the high-byte sparse
    delta tail.
    """

    if not hlm2_latent_payload.startswith(PR106_HDM8_HLM2_LATENT_MAGIC):
        raise ValueError("HLM3 recode source latent payload must start with HLM2 magic")
    if len(hlm2_latent_payload) != PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES:
        raise ValueError(
            "HLM3 recode source HLM2 latent length mismatch: "
            f"got {len(hlm2_latent_payload)} bytes; "
            f"expected {PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES}"
        )
    total = PR106_LATENT_N_PAIRS * PR106_LATENT_N_DIMS
    meta_len = PR106_LATENT_N_DIMS * 4
    cursor = 4
    lo_len = int.from_bytes(hlm2_latent_payload[cursor : cursor + 2], "little")
    cursor += 2
    lo_end = cursor + lo_len
    meta_end = lo_end + meta_len
    if meta_end > len(hlm2_latent_payload):
        raise ValueError("HLM2 latent payload truncated before metadata")
    lo_brotli = hlm2_latent_payload[cursor:lo_end]
    meta = hlm2_latent_payload[lo_end:meta_end]
    hi_delta = hlm2_latent_payload[meta_end:]
    lo_raw, consumed = _decompress_one_brotli_stream(lo_brotli, label="HLM2 lo")
    if consumed != len(lo_brotli):
        raise ValueError("HLM2 lo Brotli stream has trailing bytes")
    if len(lo_raw) != total:
        raise ValueError("HLM2 lo stream length mismatch")
    hlm3 = PR106_HDM9_HLM3_LATENT_MAGIC + lo_brotli + meta + hi_delta
    if len(hlm3) != PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES:
        raise ValueError(
            "HLM3 recode length constant is stale: "
            f"got {len(hlm3)} bytes; expected {PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES}"
        )
    return hlm3


def decode_hlm3_latents_to_hlm2_payload(hlm3_latent_payload: bytes) -> bytes:
    """Expand HLM3 latent bytes to HLM2-equivalent bytes for proofing."""

    if not hlm3_latent_payload.startswith(PR106_HDM9_HLM3_LATENT_MAGIC):
        raise ValueError("HLM3 latent payload must start with HLM3 magic")
    if len(hlm3_latent_payload) != PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES:
        raise ValueError(
            "HLM3 latent length mismatch: "
            f"got {len(hlm3_latent_payload)} bytes; "
            f"expected {PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES}"
        )
    total = PR106_LATENT_N_PAIRS * PR106_LATENT_N_DIMS
    meta_len = PR106_LATENT_N_DIMS * 4
    lo_raw, lo_len = _decompress_one_brotli_stream(
        hlm3_latent_payload[4:],
        label="HLM3 lo",
    )
    if len(lo_raw) != total:
        raise ValueError("HLM3 lo stream length mismatch")
    lo_start = 4
    lo_end = lo_start + lo_len
    meta_end = lo_end + meta_len
    if meta_end > len(hlm3_latent_payload):
        raise ValueError("HLM3 latent payload truncated before metadata")
    lo_brotli = hlm3_latent_payload[lo_start:lo_end]
    meta = hlm3_latent_payload[lo_end:meta_end]
    hi_delta = hlm3_latent_payload[meta_end:]
    return (
        PR106_HDM8_HLM2_LATENT_MAGIC
        + len(lo_brotli).to_bytes(2, "little")
        + lo_brotli
        + meta
        + hi_delta
    )


def _hdm8_hlm2_inner_without_header_payload(pr106_bytes: bytes) -> bytes:
    """Return HDM8/HLM2 inner payload with the PR106 four-byte header elided."""

    expected_total = (
        4
        + PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES
    )
    if len(pr106_bytes) != expected_total:
        raise ValueError(
            "format_id=0x08 HDM8/HLM2 PR106 inner payload length mismatch: "
            f"got {len(pr106_bytes)} bytes; expected {expected_total}"
        )
    if pr106_bytes[0] != PR106_HEADERLESS_INNER_FIRST_BYTE:
        raise ValueError(
            "format_id=0x08 requires PR106 inner header byte "
            f"0x{PR106_HEADERLESS_INNER_FIRST_BYTE:02X}; got 0x{pr106_bytes[0]:02X}"
        )
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    if decoder_len != PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "format_id=0x08 requires fixed HDM8 decoder length "
            f"{PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES}; got {decoder_len}"
        )
    inner_without_header = pr106_bytes[4:]
    if inner_without_header[:4] != PR106_HDM8_HLM2_DECODER_MAGIC:
        raise ValueError(
            "format_id=0x08 requires HDM8 decoder section magic at inner offset 4"
        )
    latent_offset = PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES
    if (
        inner_without_header[latent_offset : latent_offset + 4]
        != PR106_HDM8_HLM2_LATENT_MAGIC
    ):
        raise ValueError(
            "format_id=0x08 requires HLM2 latent section magic at fixed decoder boundary"
        )
    return inner_without_header


def _reconstruct_hdm8_hlm2_inner_headerless_pr106_bytes(inner_without_header: bytes) -> bytes:
    """Reconstruct PR106 bytes from format 0x08's fixed HDM8/HLM2 payload."""

    expected_len = (
        PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES
    )
    if len(inner_without_header) != expected_len:
        raise ValueError(
            "format_id=0x08 HDM8/HLM2 headerless payload length mismatch: "
            f"got {len(inner_without_header)} bytes; expected {expected_len}"
        )
    if inner_without_header[:4] != PR106_HDM8_HLM2_DECODER_MAGIC:
        raise ValueError("format_id=0x08 headerless payload missing HDM8 magic")
    latent_offset = PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES
    if (
        inner_without_header[latent_offset : latent_offset + 4]
        != PR106_HDM8_HLM2_LATENT_MAGIC
    ):
        raise ValueError("format_id=0x08 headerless payload missing HLM2 magic")
    return (
        bytes([PR106_HEADERLESS_INNER_FIRST_BYTE])
        + PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES.to_bytes(3, "little")
        + inner_without_header
    )


def _pack_hdm9_scale_tail(scale_stream: bytes) -> bytes:
    """Pack 28 fp32 scale bytes as low3 stream plus 0x3B/0x3C high-byte mask."""

    expected = 4 * PR106_HDM9_SCALE_COUNT
    if len(scale_stream) != expected:
        raise ValueError(
            f"HDM9 scale stream must be {expected} bytes; got {len(scale_stream)}"
        )
    low3 = bytearray()
    mask = bytearray(PR106_HDM9_SCALE_HIGH_MASK_BYTES)
    for index in range(PR106_HDM9_SCALE_COUNT):
        start = 4 * index
        low3.extend(scale_stream[start : start + 3])
        high = scale_stream[start + 3]
        if high not in (PR106_HDM9_SCALE_HIGH_BASE, PR106_HDM9_SCALE_HIGH_BASE + 1):
            raise ValueError(
                "HDM9 scale high byte outside fixed two-symbol alphabet at "
                f"index {index}: 0x{high:02X}"
            )
        if high == PR106_HDM9_SCALE_HIGH_BASE + 1:
            mask[index // 8] |= 1 << (index % 8)
    return bytes(low3) + bytes(mask)


def _unpack_hdm9_scale_tail(scale_tail: bytes) -> bytes:
    """Unpack HDM9 low3+high-mask scale bytes back to the 112-byte f32 tail."""

    expected = PR106_HDM9_SCALE_LOW3_BYTES + PR106_HDM9_SCALE_HIGH_MASK_BYTES
    if len(scale_tail) != expected:
        raise ValueError(f"HDM9 scale tail must be {expected} bytes; got {len(scale_tail)}")
    low3 = scale_tail[:PR106_HDM9_SCALE_LOW3_BYTES]
    mask = scale_tail[PR106_HDM9_SCALE_LOW3_BYTES:]
    padding_bits = len(mask) * 8 - PR106_HDM9_SCALE_COUNT
    if padding_bits:
        padding_mask = ((1 << padding_bits) - 1) << (8 - padding_bits)
        if mask[-1] & padding_mask:
            raise ValueError("HDM9 scale high-byte mask padding must be zero")
    out = bytearray()
    for index in range(PR106_HDM9_SCALE_COUNT):
        start = 3 * index
        bit = (mask[index // 8] >> (index % 8)) & 1
        out.extend(low3[start : start + 3])
        out.append(PR106_HDM9_SCALE_HIGH_BASE + bit)
    return bytes(out)


def encode_hdm9_decoder_from_hdm8_payload(hdm8_decoder_payload: bytes) -> bytes:
    """Return the HDM9 byte-closed decoder recode for a fixed HDM8 payload.

    HDM9 keeps HDM8's fixed q-Brotli chunks byte-for-byte and recodes only the
    raw fp32 scale tail. The scale high-byte sequence is stored in a four-byte
    mask, so no model parameter bytes are moved into runtime source.
    """

    if not hdm8_decoder_payload.startswith(PR106_HDM8_HLM2_DECODER_MAGIC):
        raise ValueError("HDM9 recode source decoder must start with HDM8 magic")
    if len(hdm8_decoder_payload) != PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "HDM9 recode source HDM8 decoder length mismatch: "
            f"got {len(hdm8_decoder_payload)} bytes; "
            f"expected {PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES}"
        )
    q_payload_bytes = PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES - 4 - 4 * PR106_HDM9_SCALE_COUNT
    q_payload = hdm8_decoder_payload[4 : 4 + q_payload_bytes]
    scale_stream = hdm8_decoder_payload[4 + q_payload_bytes :]
    hdm9 = PR106_HDM9_HLM2_DECODER_MAGIC + q_payload + _pack_hdm9_scale_tail(scale_stream)
    if len(hdm9) != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "HDM9 recode length constant is stale: "
            f"got {len(hdm9)} bytes; expected {PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}"
        )
    return hdm9


def decode_hdm9_decoder_to_hdm8_payload(hdm9_decoder_payload: bytes) -> bytes:
    """Expand HDM9 decoder bytes to HDM8-equivalent bytes for compiler proofing."""

    if not hdm9_decoder_payload.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
        raise ValueError("HDM9 decoder payload must start with HDM9 magic")
    if len(hdm9_decoder_payload) != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "HDM9 decoder length mismatch: "
            f"got {len(hdm9_decoder_payload)} bytes; "
            f"expected {PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}"
        )
    q_payload_bytes = (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        - 4
        - PR106_HDM9_SCALE_LOW3_BYTES
        - PR106_HDM9_SCALE_HIGH_MASK_BYTES
    )
    q_payload = hdm9_decoder_payload[4 : 4 + q_payload_bytes]
    scale_tail = hdm9_decoder_payload[4 + q_payload_bytes :]
    return PR106_HDM8_HLM2_DECODER_MAGIC + q_payload + _unpack_hdm9_scale_tail(scale_tail)


def _reconstruct_hdm9_hlm2_inner_headerless_pr106_bytes(inner_without_header: bytes) -> bytes:
    """Reconstruct PR106 bytes from format 0x09's fixed HDM9/HLM2 payload."""

    expected_len = (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES
    )
    if len(inner_without_header) != expected_len:
        raise ValueError(
            "format_id=0x09 HDM9/HLM2 headerless payload length mismatch: "
            f"got {len(inner_without_header)} bytes; expected {expected_len}"
        )
    if inner_without_header[:4] != PR106_HDM9_HLM2_DECODER_MAGIC:
        raise ValueError("format_id=0x09 headerless payload missing HDM9 magic")
    latent_offset = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
    if (
        inner_without_header[latent_offset : latent_offset + 4]
        != PR106_HDM8_HLM2_LATENT_MAGIC
    ):
        raise ValueError("format_id=0x09 headerless payload missing HLM2 magic")
    return (
        bytes([PR106_HEADERLESS_INNER_FIRST_BYTE])
        + PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES.to_bytes(3, "little")
        + inner_without_header
    )


def _hdm9_hlm2_inner_without_header_payload(pr106_bytes: bytes) -> bytes:
    """Return HDM9/HLM2 inner payload with the PR106 four-byte header elided."""

    expected_total = (
        4
        + PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES
    )
    if len(pr106_bytes) != expected_total:
        raise ValueError(
            "format_id=0x09 HDM9/HLM2 PR106 inner payload length mismatch: "
            f"got {len(pr106_bytes)} bytes; expected {expected_total}"
        )
    if pr106_bytes[0] != PR106_HEADERLESS_INNER_FIRST_BYTE:
        raise ValueError(
            "format_id=0x09 requires PR106 inner header byte "
            f"0x{PR106_HEADERLESS_INNER_FIRST_BYTE:02X}; got 0x{pr106_bytes[0]:02X}"
        )
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    if decoder_len != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "format_id=0x09 requires fixed HDM9 decoder length "
            f"{PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}; got {decoder_len}"
        )
    inner_without_header = pr106_bytes[4:]
    if inner_without_header[:4] != PR106_HDM9_HLM2_DECODER_MAGIC:
        raise ValueError(
            "format_id=0x09 requires HDM9 decoder section magic at inner offset 4"
        )
    latent_offset = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
    if (
        inner_without_header[latent_offset : latent_offset + 4]
        != PR106_HDM8_HLM2_LATENT_MAGIC
    ):
        raise ValueError(
            "format_id=0x09 requires HLM2 latent section magic at fixed decoder boundary"
        )
    return inner_without_header


def _reconstruct_hdm9_hlm3_inner_headerless_pr106_bytes(inner_without_header: bytes) -> bytes:
    """Reconstruct PR106 bytes from format 0x0A's fixed HDM9/HLM3 payload."""

    expected_len = (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
    )
    if len(inner_without_header) != expected_len:
        raise ValueError(
            "format_id=0x0A HDM9/HLM3 headerless payload length mismatch: "
            f"got {len(inner_without_header)} bytes; expected {expected_len}"
        )
    if inner_without_header[:4] != PR106_HDM9_HLM2_DECODER_MAGIC:
        raise ValueError("format_id=0x0A headerless payload missing HDM9 magic")
    latent_offset = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
    if (
        inner_without_header[latent_offset : latent_offset + 4]
        != PR106_HDM9_HLM3_LATENT_MAGIC
    ):
        raise ValueError("format_id=0x0A headerless payload missing HLM3 magic")
    return (
        bytes([PR106_HEADERLESS_INNER_FIRST_BYTE])
        + PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES.to_bytes(3, "little")
        + inner_without_header
    )


def _reconstruct_hdm9_hlm3_magicless_pr106_bytes(magicless_payload: bytes) -> bytes:
    """Reconstruct PR106 bytes from format 0x0B's fixed magic-elided payload."""

    expected_len = (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        - len(PR106_HDM9_HLM2_DECODER_MAGIC)
        + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
        - len(PR106_HDM9_HLM3_LATENT_MAGIC)
    )
    if len(magicless_payload) != expected_len:
        raise ValueError(
            "format_id=0x0B HDM9/HLM3 magicless payload length mismatch: "
            f"got {len(magicless_payload)} bytes; expected {expected_len}"
        )
    decoder_tail_bytes = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES - len(
        PR106_HDM9_HLM2_DECODER_MAGIC
    )
    decoder_tail = magicless_payload[:decoder_tail_bytes]
    latent_tail = magicless_payload[decoder_tail_bytes:]
    inner_without_header = (
        PR106_HDM9_HLM2_DECODER_MAGIC
        + decoder_tail
        + PR106_HDM9_HLM3_LATENT_MAGIC
        + latent_tail
    )
    _validate_hdm9_hlm3_inner_payload_structure(
        inner_without_header,
        context="format_id=0x0B magicless payload",
    )
    return _reconstruct_hdm9_hlm3_inner_headerless_pr106_bytes(inner_without_header)


def _validate_hdm9_decoder_payload_structure(
    hdm9_decoder_payload: bytes,
    *,
    context: str,
) -> None:
    if not hdm9_decoder_payload.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
        raise ValueError(f"{context} missing HDM9 decoder magic")
    if len(hdm9_decoder_payload) != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            f"{context} HDM9 decoder length mismatch: "
            f"got {len(hdm9_decoder_payload)} bytes; "
            f"expected {PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}"
        )
    q_payload_bytes = (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        - len(PR106_HDM9_HLM2_DECODER_MAGIC)
        - PR106_HDM9_SCALE_LOW3_BYTES
        - PR106_HDM9_SCALE_HIGH_MASK_BYTES
    )
    if sum(PR106_HDM9_Q_BROTLI_CHUNK_BYTES) != q_payload_bytes:
        raise ValueError("HDM9 q-Brotli chunk constants are stale")
    cursor = len(PR106_HDM9_HLM2_DECODER_MAGIC)
    for index, chunk_len in enumerate(PR106_HDM9_Q_BROTLI_CHUNK_BYTES):
        chunk = hdm9_decoder_payload[cursor : cursor + chunk_len]
        cursor += chunk_len
        try:
            brotli.decompress(chunk)
        except brotli.error as exc:
            raise ValueError(
                f"{context} HDM9 q-Brotli chunk {index} failed structural decode"
            ) from exc
    _unpack_hdm9_scale_tail(hdm9_decoder_payload[cursor:])


def _validate_hdm9_hlm3_inner_payload_structure(
    inner_without_header: bytes,
    *,
    context: str,
) -> None:
    expected_len = (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
    )
    if len(inner_without_header) != expected_len:
        raise ValueError(
            f"{context} length mismatch: got {len(inner_without_header)} bytes; "
            f"expected {expected_len}"
        )
    decoder = inner_without_header[:PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES]
    latents = inner_without_header[PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:]
    _validate_hdm9_decoder_payload_structure(decoder, context=context)
    decode_hlm3_latents_to_hlm2_payload(latents)


def _validate_headerless_pr106_bytes(pr106_bytes: bytes, *, context: str) -> None:
    if len(pr106_bytes) < 4:
        raise ValueError(f"{context} truncated before inner header")
    if pr106_bytes[0] != PR106_HEADERLESS_INNER_FIRST_BYTE:
        raise ValueError(
            f"{context} requires PR106 inner payload to start with "
            f"0x{PR106_HEADERLESS_INNER_FIRST_BYTE:02X}"
        )
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    if decoder_len <= 0:
        raise ValueError(f"{context} decoder length must be positive")
    if 4 + decoder_len >= len(pr106_bytes):
        raise ValueError(f"{context} decoder length leaves no latent payload")


def _hdm9_hlm3_inner_without_header_payload(pr106_bytes: bytes) -> bytes:
    """Return HDM9/HLM3 inner payload with the PR106 four-byte header elided."""

    expected_total = (
        4
        + PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
    )
    if len(pr106_bytes) != expected_total:
        raise ValueError(
            "format_id=0x0A HDM9/HLM3 PR106 inner payload length mismatch: "
            f"got {len(pr106_bytes)} bytes; expected {expected_total}"
        )
    if pr106_bytes[0] != PR106_HEADERLESS_INNER_FIRST_BYTE:
        raise ValueError(
            "format_id=0x0A requires PR106 inner header byte "
            f"0x{PR106_HEADERLESS_INNER_FIRST_BYTE:02X}; got 0x{pr106_bytes[0]:02X}"
        )
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    if decoder_len != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "format_id=0x0A requires fixed HDM9 decoder length "
            f"{PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}; got {decoder_len}"
        )
    inner_without_header = pr106_bytes[4:]
    if inner_without_header[:4] != PR106_HDM9_HLM2_DECODER_MAGIC:
        raise ValueError(
            "format_id=0x0A requires HDM9 decoder section magic at inner offset 4"
        )
    latent_offset = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
    if (
        inner_without_header[latent_offset : latent_offset + 4]
        != PR106_HDM9_HLM3_LATENT_MAGIC
    ):
        raise ValueError(
            "format_id=0x0A requires HLM3 latent section magic at fixed decoder boundary"
        )
    return inner_without_header


def _hdm9_hlm3_magicless_payload(pr106_bytes: bytes) -> bytes:
    """Return HDM9/HLM3 payload with inner header and section magics elided."""

    inner_without_header = _hdm9_hlm3_inner_without_header_payload(pr106_bytes)
    decoder = inner_without_header[:PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES]
    latents = inner_without_header[PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:]
    if not decoder.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
        raise ValueError("format_id=0x0B requires HDM9 decoder magic")
    if not latents.startswith(PR106_HDM9_HLM3_LATENT_MAGIC):
        raise ValueError("format_id=0x0B requires HLM3 latent magic")
    return decoder[len(PR106_HDM9_HLM2_DECODER_MAGIC) :] + latents[
        len(PR106_HDM9_HLM3_LATENT_MAGIC) :
    ]


def recode_pr106_hdm8_hlm2_packet_to_hdm9(
    packet: PR106SidecarPacket,
) -> PR106SidecarPacket:
    """Recode a fixed HDM8/HLM2 PR106 packet to the HDM9 decoder prototype."""

    reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(packet.sidecar_payload)
    pr106_bytes = packet.pr106_bytes
    expected_total = (
        4
        + PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES
        + PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES
    )
    if len(pr106_bytes) != expected_total:
        raise ValueError(
            "HDM9 recode requires fixed HDM8/HLM2 PR106 bytes: "
            f"got {len(pr106_bytes)} bytes; expected {expected_total}"
        )
    if pr106_bytes[0] != PR106_HEADERLESS_INNER_FIRST_BYTE:
        raise ValueError("HDM9 recode requires a packed PR106 inner header")
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    if decoder_len != PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES:
        raise ValueError(
            "HDM9 recode requires HDM8 decoder length "
            f"{PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES}; got {decoder_len}"
        )
    decoder = pr106_bytes[4 : 4 + decoder_len]
    latents = pr106_bytes[4 + decoder_len :]
    if not latents.startswith(PR106_HDM8_HLM2_LATENT_MAGIC):
        raise ValueError("HDM9 recode requires HLM2 latent payload")
    hdm9_decoder = encode_hdm9_decoder_from_hdm8_payload(decoder)
    hdm9_pr106_bytes = (
        bytes([PR106_HEADERLESS_INNER_FIRST_BYTE])
        + len(hdm9_decoder).to_bytes(3, "little")
        + hdm9_decoder
        + latents
    )
    return PR106SidecarPacket(
        format_id=PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        pr106_bytes=hdm9_pr106_bytes,
        sidecar_payload=packet.sidecar_payload,
        framing_meta=None,
    )


def _recode_hdm8_or_hdm9_hlm2_pr106_bytes_to_hdm9_hlm3(pr106_bytes: bytes) -> bytes:
    if len(pr106_bytes) < 4 or pr106_bytes[0] != PR106_HEADERLESS_INNER_FIRST_BYTE:
        raise ValueError("HLM3 recode requires a packed PR106 inner header")
    decoder_len = int.from_bytes(pr106_bytes[1:4], "little")
    decoder = pr106_bytes[4 : 4 + decoder_len]
    latents = pr106_bytes[4 + decoder_len :]
    if decoder.startswith(PR106_HDM8_HLM2_DECODER_MAGIC):
        if decoder_len != PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES:
            raise ValueError(
                "HLM3 recode requires fixed HDM8 decoder length "
                f"{PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES}; got {decoder_len}"
            )
        hdm9_decoder = encode_hdm9_decoder_from_hdm8_payload(decoder)
    elif decoder.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
        if decoder_len != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
            raise ValueError(
                "HLM3 recode requires fixed HDM9 decoder length "
                f"{PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}; got {decoder_len}"
            )
        hdm9_decoder = decoder
    else:
        raise ValueError("HLM3 recode requires HDM8 or HDM9 decoder payload")
    if not latents.startswith(PR106_HDM8_HLM2_LATENT_MAGIC):
        raise ValueError("HLM3 recode requires HLM2 latent payload")
    hlm3_latents = encode_hlm3_latents_from_hlm2_payload(latents)
    return (
        bytes([PR106_HEADERLESS_INNER_FIRST_BYTE])
        + len(hdm9_decoder).to_bytes(3, "little")
        + hdm9_decoder
        + hlm3_latents
    )


def recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3(
    packet: PR106SidecarPacket,
) -> PR106SidecarPacket:
    """Recode fixed HDM8/HLM2 or HDM9/HLM2 bytes to format 0x0A.

    The caller supplies the already no-op-rank-elided sidecar payload. The
    inner PR106 payload is converted to HDM9/HLM3, preserving the same decoded
    HNeRV state and latent tensor.
    """

    reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(packet.sidecar_payload)
    hdm9_hlm3_pr106_bytes = _recode_hdm8_or_hdm9_hlm2_pr106_bytes_to_hdm9_hlm3(
        packet.pr106_bytes
    )
    return PR106SidecarPacket(
        format_id=(
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
        ),
        pr106_bytes=hdm9_hlm3_pr106_bytes,
        sidecar_payload=packet.sidecar_payload,
        framing_meta=None,
    )


def recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3_magicless(
    packet: PR106SidecarPacket,
) -> PR106SidecarPacket:
    """Recode fixed HDM8/HLM2 or HDM9/HLM2 bytes to format 0x0B."""

    hdm9_hlm3 = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3(packet)
    return replace(
        hdm9_hlm3,
        format_id=(
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
        ),
    )


def decode_pr101_rank_elided_sidecar_payload_to_dim_delta(
    payload: bytes,
    framing_meta: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PR106 format-0x04 rank-elided sidecar payload."""

    expanded_payload, expanded_meta = reexpand_pr101_rank_elided_sidecar_payload(
        payload,
        framing_meta,
        schema=schema,
    )
    return decode_pr101_ranked_sidecar_payload_to_dim_delta(
        expanded_payload,
        expanded_meta,
        schema=schema,
    )


def decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PR106 format-0x05 fixed-meta/rank-elided sidecar payload."""

    expanded_payload, expanded_meta = reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(
        payload,
        schema=schema,
    )
    return decode_pr101_ranked_sidecar_payload_to_dim_delta(
        expanded_payload,
        expanded_meta,
        schema=schema,
    )


def decode_pr101_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PR106 format-0x0A fixed-meta/noop-rank-elided sidecar payload."""

    expanded_payload, expanded_meta = (
        reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(
            payload,
            schema=schema,
        )
    )
    return decode_pr101_ranked_sidecar_payload_to_dim_delta(
        expanded_payload,
        expanded_meta,
        schema=schema,
    )


def decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode PR106 format-0x0C exact-radix fixed-meta sidecar payload."""

    expanded_payload, expanded_meta = (
        reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
            payload,
            schema=schema,
        )
    )
    return decode_pr101_ranked_sidecar_payload_to_dim_delta(
        expanded_payload,
        expanded_meta,
        schema=schema,
    )


def encode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> bytes:
    """Encode semantic corrections as the committed PR106 format-0x0C payload."""

    _validate_fixed_meta_schema(schema)
    dims, deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        dim_arr,
        delta_q_arr,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    if int(dims.size) != int(schema.n_pairs):
        raise ValueError(f"format_id=0x0C expects {schema.n_pairs} pairs; got {dims.size}")
    if bool(np.any(dims == schema.no_op_sentinel)) or bool(np.any(deltas == 0)):
        raise ValueError("format_id=0x0C fixed metadata requires no-op-free corrections")
    ranked_payload, ranked_meta = encode_pr101_ranked_sidecar_payload(
        dims,
        deltas,
        schema=schema,
    )
    if ranked_meta != _fixed_meta_framing_bytes():
        raise ValueError("format_id=0x0C requires canonical PR106 fixed metadata")
    if ranked_payload[-PR106_PR101_FIXED_META_NOOP_RANK_BYTES:] != (
        PR106_PR101_FIXED_META_NOOP_RANK_BYTE
    ):
        raise ValueError("format_id=0x0C requires zero fixed no-op rank")
    huff_start = PR106_PR101_FIXED_META_DIM_BYTES + PR106_PR101_FIXED_META_RANK_BYTES
    huff_end = -PR106_PR101_FIXED_META_NOOP_RANK_BYTES
    encoded = _encode_exact_radix_dim_payload(dims, schema=schema) + ranked_payload[
        huff_start:huff_end
    ]
    decoded_dims, decoded_deltas = (
        decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
            encoded,
            schema=schema,
        )
    )
    if not (np.array_equal(decoded_dims, dims) and np.array_equal(decoded_deltas, deltas)):
        raise ValueError("format_id=0x0C encode/decode semantic mismatch")
    return encoded


def encode_pr101_fixed_meta_rank_elided_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> bytes:
    """Encode semantic corrections as a fixed-meta/rank-elided PR101 payload."""

    _validate_fixed_meta_schema(schema)
    dims, deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        dim_arr,
        delta_q_arr,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    if int(dims.size) != int(schema.n_pairs):
        raise ValueError(
            f"fixed-meta rank-elided payload expects {schema.n_pairs} pairs; "
            f"got {dims.size}"
        )
    ranked_payload, ranked_meta = encode_pr101_ranked_sidecar_payload(
        dims,
        deltas,
        schema=schema,
    )
    if ranked_meta != _fixed_meta_framing_bytes():
        raise ValueError("fixed-meta rank-elided payload requires canonical PR106 metadata")
    encoded = (
        ranked_payload[:PR106_PR101_FIXED_META_DIM_BYTES]
        + ranked_payload[
            PR106_PR101_FIXED_META_DIM_BYTES
            + PR106_PR101_FIXED_META_RANK_BYTES :
        ]
    )
    decoded_dims, decoded_deltas = (
        decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
            encoded,
            schema=schema,
        )
    )
    if not (np.array_equal(decoded_dims, dims) and np.array_equal(decoded_deltas, deltas)):
        raise ValueError("fixed-meta rank-elided encode/decode semantic mismatch")
    return encoded


def decode_pr106_sidecar_packet_dim_delta(
    packet: PR106SidecarPacket,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a parsed PR106 sidecar packet into canonical correction arrays."""

    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
        return decode_brotli_dim_delta_sidecar_payload(packet.sidecar_payload)
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x02 requires framing_meta")
        return decode_pr101_ranked_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            packet.framing_meta,
            schema=schema,
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x04 requires framing_meta")
        return decode_pr101_rank_elided_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            packet.framing_meta,
            schema=schema,
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED:
        return decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            schema=schema,
        )
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        or packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
    ):
        return decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            schema=schema,
        )
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
        or packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        return decode_pr101_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            schema=schema,
        )
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        return decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
            packet.sidecar_payload,
            schema=schema,
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        raise ValueError(
            "format_id=0x0D carries two additive correction passes; use "
            "decode_pr106_sidecar_packet_correction_passes"
        )
    raise ValueError(f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}")


def decode_pr106_sidecar_packet_correction_passes(
    packet: PR106SidecarPacket,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Decode packet corrections as the runtime applies them.

    Most PacketIR formats carry one per-pair correction pass. Format ``0x0D``
    is deliberately multi-pass: the fixed format0C base stream is applied first,
    then the appended PR101 ranked/no-op extra stream is applied additively.
    """

    if packet.format_id != PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        return (decode_pr106_sidecar_packet_dim_delta(packet, schema=schema),)
    if packet.extra_framing_meta is None:
        raise ValueError("format_id=0x0D requires extra_framing_meta")
    base = decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
        packet.sidecar_payload,
        schema=schema,
    )
    extra = decode_pr101_ranked_sidecar_payload_to_dim_delta(
        packet.extra_sidecar_payload,
        packet.extra_framing_meta,
        schema=schema,
    )
    return (base, extra)


def encode_pr106_format0d_sidecar_payload(
    base_format0c_sidecar_payload: bytes,
    extra_dim_arr: np.ndarray,
    extra_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes, bytes]:
    """Encode format 0x0D's base-0C plus extra PR101 correction streams."""

    _validate_fixed_meta_schema(schema)
    reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
        base_format0c_sidecar_payload,
        schema=schema,
    )
    extra_payload, extra_meta = encode_pr101_ranked_sidecar_payload(
        extra_dim_arr,
        extra_delta_q_arr,
        schema=schema,
    )
    if len(extra_payload) > 0xFFFF:
        raise ValueError("format_id=0x0D extra PR101 payload too large for u16 length")
    return (
        base_format0c_sidecar_payload
        + struct.pack("<H", len(extra_payload))
        + extra_payload
        + extra_meta,
        extra_payload,
        extra_meta,
    )


def split_pr106_format0d_sidecar_payload(
    payload: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[bytes, bytes, bytes]:
    """Split and validate format 0x0D's stacked sidecar payload."""

    _validate_fixed_meta_schema(schema)
    base_len = PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
    if len(payload) < base_len + 2 + PR106_FORMAT0D_EXTRA_FRAMING_META_BYTES:
        raise ValueError(
            "format_id=0x0D sidecar payload truncated before base/extra streams"
        )
    base_payload = payload[:base_len]
    reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
        base_payload,
        schema=schema,
    )
    pos = base_len
    (extra_len,) = struct.unpack_from("<H", payload, pos)
    pos += 2
    extra_end = pos + int(extra_len)
    if extra_end + PR106_FORMAT0D_EXTRA_FRAMING_META_BYTES != len(payload):
        raise ValueError(
            "format_id=0x0D extra stream length mismatch: "
            f"extra_len={extra_len} total={len(payload)}"
        )
    extra_payload = payload[pos:extra_end]
    extra_meta = payload[extra_end:]
    decode_pr101_ranked_sidecar_payload_to_dim_delta(
        extra_payload,
        extra_meta,
        schema=schema,
    )
    return base_payload, extra_payload, extra_meta


def _pack_fixed_width_symbols(symbols: np.ndarray, *, bit_width: int) -> bytes:
    if bit_width <= 0:
        raise ValueError(f"bit_width must be positive; got {bit_width}")
    values = np.asarray(symbols, dtype=np.int64).reshape(-1)
    max_value = (1 << bit_width) - 1
    if values.size and (int(values.min()) < 0 or int(values.max()) > max_value):
        raise ValueError(
            f"symbols out of range for {bit_width}-bit packing: "
            f"min={int(values.min())} max={int(values.max())}"
        )
    out = bytearray()
    acc = 0
    acc_bits = 0
    for value in values.tolist():
        acc |= int(value) << acc_bits
        acc_bits += bit_width
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def _unpack_fixed_width_symbols(payload: bytes, *, n_symbols: int, bit_width: int) -> np.ndarray:
    if bit_width <= 0:
        raise ValueError(f"bit_width must be positive; got {bit_width}")
    if n_symbols < 0:
        raise ValueError(f"n_symbols must be non-negative; got {n_symbols}")
    expected_bytes = (n_symbols * bit_width + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"fixed-width payload byte length mismatch: got={len(payload)} "
            f"expected={expected_bytes}"
        )
    out = np.empty(n_symbols, dtype=np.int64)
    acc = 0
    acc_bits = 0
    pos = 0
    mask = (1 << bit_width) - 1
    for index in range(n_symbols):
        while acc_bits < bit_width:
            if pos >= len(payload):
                raise ValueError("truncated fixed-width payload")
            acc |= int(payload[pos]) << acc_bits
            acc_bits += 8
            pos += 1
        out[index] = acc & mask
        acc >>= bit_width
        acc_bits -= bit_width
    if acc and acc_bits:
        raise ValueError("non-zero fixed-width padding bits")
    return out


def _encode_vocab_bitpack_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    n_dims: int = PR106_LATENT_N_DIMS,
    no_op_dim: int = PR106_NO_OP_DIM,
) -> bytes:
    dims, deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        dim_arr,
        delta_q_arr,
        n_dims=n_dims,
        no_op_dim=no_op_dim,
    )
    vocab = tuple(sorted({int(value) for value in deltas.astype(np.int16).tolist()}))
    if len(vocab) > 255:
        raise ValueError(f"delta vocabulary too large for u8 length: {len(vocab)}")
    if n_dims >= no_op_dim:
        raise ValueError(f"n_dims must be below no_op_dim; got {n_dims}/{no_op_dim}")
    dim_width = max(1, math.ceil(math.log2(n_dims + 1)))
    delta_width = max(1, math.ceil(math.log2(max(len(vocab), 2))))
    dim_codes = np.where(dims == no_op_dim, n_dims, dims).astype(np.int64)
    delta_index = {value: index for index, value in enumerate(vocab)}
    delta_codes = np.asarray([delta_index[int(value)] for value in deltas], dtype=np.int64)
    return b"LSV1" + struct.pack(
        "<HBBBB",
        int(dims.size),
        int(n_dims),
        int(dim_width),
        int(delta_width),
        len(vocab),
    ) + np.asarray(vocab, dtype=np.int8).view(np.uint8).tobytes() + _pack_fixed_width_symbols(
        dim_codes,
        bit_width=dim_width,
    ) + _pack_fixed_width_symbols(
        delta_codes,
        bit_width=delta_width,
    )


def _decode_vocab_bitpack_sidecar_payload(
    payload: bytes,
    *,
    no_op_dim: int = PR106_NO_OP_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    if not payload.startswith(b"LSV1"):
        raise ValueError("vocab-bitpack sidecar magic mismatch")
    if len(payload) < 10:
        raise ValueError("vocab-bitpack sidecar payload too short")
    n_pairs, n_dims, dim_width, delta_width, vocab_len = struct.unpack_from(
        "<HBBBB",
        payload,
        4,
    )
    pos = 10
    vocab_end = pos + vocab_len
    if vocab_end > len(payload):
        raise ValueError("vocab-bitpack sidecar truncated before vocabulary")
    vocab = np.frombuffer(payload[pos:vocab_end], dtype=np.uint8).copy().view(np.int8)
    pos = vocab_end
    dim_bytes = (n_pairs * dim_width + 7) // 8
    delta_bytes = (n_pairs * delta_width + 7) // 8
    if pos + dim_bytes + delta_bytes != len(payload):
        raise ValueError(
            "vocab-bitpack sidecar byte length mismatch: "
            f"pos={pos} dim_bytes={dim_bytes} delta_bytes={delta_bytes} total={len(payload)}"
        )
    dim_codes = _unpack_fixed_width_symbols(
        payload[pos : pos + dim_bytes],
        n_symbols=n_pairs,
        bit_width=dim_width,
    )
    pos += dim_bytes
    delta_codes = _unpack_fixed_width_symbols(
        payload[pos : pos + delta_bytes],
        n_symbols=n_pairs,
        bit_width=delta_width,
    )
    if bool(np.any(dim_codes > n_dims)):
        bad = int(dim_codes[np.flatnonzero(dim_codes > n_dims)[0]])
        raise ValueError(f"vocab-bitpack dim code {bad} exceeds no-op code {n_dims}")
    if vocab_len == 0:
        raise ValueError("vocab-bitpack delta vocabulary is empty")
    if bool(np.any(delta_codes >= vocab_len)):
        bad = int(delta_codes[np.flatnonzero(delta_codes >= vocab_len)[0]])
        raise ValueError(f"vocab-bitpack delta code {bad} exceeds vocab_len {vocab_len}")
    dims = np.where(dim_codes == n_dims, no_op_dim, dim_codes).astype(np.uint8)
    deltas = vocab[delta_codes].astype(np.int8)
    return canonicalize_brotli_dim_delta_sidecar_arrays(
        dims,
        deltas,
        n_dims=int(n_dims),
        no_op_dim=no_op_dim,
    )


def _encode_split_stream_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    quality: int = 11,
) -> bytes:
    dims, deltas = canonicalize_brotli_dim_delta_sidecar_arrays(dim_arr, delta_q_arr)
    raw = struct.pack("<H", int(dims.size)) + dims.tobytes() + deltas.view(np.uint8).tobytes()
    return b"LSS1" + brotli.compress(raw, quality=quality)


def _decode_split_stream_sidecar_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if not payload.startswith(b"LSS1"):
        raise ValueError("split-stream sidecar magic mismatch")
    raw = brotli.decompress(payload[4:])
    if len(raw) < 2:
        raise ValueError("split-stream sidecar raw payload too short")
    n_pairs = struct.unpack_from("<H", raw, 0)[0]
    expected = 2 + 2 * n_pairs
    if len(raw) != expected:
        raise ValueError(
            f"split-stream sidecar raw length mismatch: got={len(raw)} expected={expected}"
        )
    dims = np.frombuffer(raw[2 : 2 + n_pairs], dtype=np.uint8).copy()
    deltas = np.frombuffer(raw[2 + n_pairs :], dtype=np.uint8).copy().view(np.int8)
    return canonicalize_brotli_dim_delta_sidecar_arrays(dims, deltas)


def _encode_sparse_indexed_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    quality: int = 11,
) -> bytes:
    dims, deltas = canonicalize_brotli_dim_delta_sidecar_arrays(dim_arr, delta_q_arr)
    valid = (dims != PR106_NO_OP_DIM) & (deltas != 0)
    indices = np.flatnonzero(valid)
    if len(indices) > 0xFFFF:
        raise ValueError(f"too many sparse corrections for u16 length: {len(indices)}")
    out = io.BytesIO()
    out.write(struct.pack("<HH", int(dims.size), len(indices)))
    for index in indices.tolist():
        delta_u8 = int(np.asarray([deltas[index]], dtype=np.int8).view(np.uint8)[0])
        out.write(struct.pack("<HBB", int(index), int(dims[index]), delta_u8))
    return b"LSP1" + brotli.compress(out.getvalue(), quality=quality)


def _decode_sparse_indexed_sidecar_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if not payload.startswith(b"LSP1"):
        raise ValueError("sparse-indexed sidecar magic mismatch")
    raw = brotli.decompress(payload[4:])
    if len(raw) < 4:
        raise ValueError("sparse-indexed sidecar raw payload too short")
    n_pairs, n_records = struct.unpack_from("<HH", raw, 0)
    expected = 4 + 4 * n_records
    if len(raw) != expected:
        raise ValueError(
            f"sparse-indexed sidecar raw length mismatch: got={len(raw)} expected={expected}"
        )
    dims = np.full(n_pairs, PR106_NO_OP_DIM, dtype=np.uint8)
    deltas = np.zeros(n_pairs, dtype=np.int8)
    seen: set[int] = set()
    pos = 4
    for _ in range(n_records):
        index, dim, delta_u8 = struct.unpack_from("<HBB", raw, pos)
        pos += 4
        if index >= n_pairs:
            raise ValueError(f"sparse-indexed sidecar index {index} out of range {n_pairs}")
        if index in seen:
            raise ValueError(f"sparse-indexed sidecar duplicate index {index}")
        seen.add(index)
        dims[index] = np.uint8(dim)
        deltas[index] = np.asarray([delta_u8], dtype=np.uint8).view(np.int8)[0]
    return canonicalize_brotli_dim_delta_sidecar_arrays(dims, deltas)


def _candidate(
    *,
    name: str,
    encoded_bytes: bytes,
    decoded: tuple[np.ndarray, np.ndarray],
    sidecar_format_id: int | None,
    framing_meta_bytes: bytes = b"",
    runtime_decoder_implemented: bool = False,
    notes: tuple[str, ...] = (),
) -> PR106SidecarRecodeCandidate:
    dims, deltas = decoded
    return PR106SidecarRecodeCandidate(
        name=name,
        encoded_bytes=encoded_bytes,
        decoded_dims=dims,
        decoded_delta_q=deltas,
        sidecar_format_id=sidecar_format_id,
        framing_meta_bytes=framing_meta_bytes,
        runtime_decoder_implemented=runtime_decoder_implemented,
        notes=notes,
    )


def lossless_pr106_sidecar_recode_candidates(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    brotli_quality: int = 11,
) -> list[PR106SidecarRecodeCandidate]:
    """Return lossless PR106 sidecar grammar alternatives for byte profiling."""

    source_dims, source_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        dim_arr,
        delta_q_arr,
    )
    candidates: list[PR106SidecarRecodeCandidate] = []

    current = encode_brotli_dim_delta_sidecar_payload(
        source_dims,
        source_deltas,
        quality=brotli_quality,
    )
    candidates.append(
        _candidate(
            name="current_pr100_dim_delta_brotli_q11",
            encoded_bytes=current,
            decoded=decode_brotli_dim_delta_sidecar_payload(current),
            sidecar_format_id=PR106_SIDECAR_FORMAT_BROTLI,
            runtime_decoder_implemented=True,
            notes=("current_runtime_consumed_format",),
        )
    )

    split = _encode_split_stream_sidecar_payload(
        source_dims,
        source_deltas,
        quality=brotli_quality,
    )
    candidates.append(
        _candidate(
            name="split_dim_stream_delta_stream_brotli_q11",
            encoded_bytes=split,
            decoded=_decode_split_stream_sidecar_payload(split),
            sidecar_format_id=None,
            notes=("candidate_requires_new_runtime_decoder",),
        )
    )

    sparse = _encode_sparse_indexed_sidecar_payload(
        source_dims,
        source_deltas,
        quality=brotli_quality,
    )
    candidates.append(
        _candidate(
            name="sparse_indexed_nonzero_brotli_q11",
            encoded_bytes=sparse,
            decoded=_decode_sparse_indexed_sidecar_payload(sparse),
            sidecar_format_id=None,
            notes=("candidate_requires_new_runtime_decoder",),
        )
    )

    vocab = _encode_vocab_bitpack_sidecar_payload(source_dims, source_deltas)
    candidates.append(
        _candidate(
            name="vocab_bitpack_dim_delta_raw",
            encoded_bytes=vocab,
            decoded=_decode_vocab_bitpack_sidecar_payload(vocab),
            sidecar_format_id=None,
            notes=("candidate_requires_new_runtime_decoder",),
        )
    )
    vocab_brotli = b"LSC1" + brotli.compress(vocab, quality=brotli_quality)
    decoded_vocab_brotli = _decode_vocab_bitpack_sidecar_payload(
        brotli.decompress(vocab_brotli[4:])
    )
    candidates.append(
        _candidate(
            name="vocab_bitpack_dim_delta_brotli_q11",
            encoded_bytes=vocab_brotli,
            decoded=decoded_vocab_brotli,
            sidecar_format_id=None,
            notes=("candidate_requires_new_runtime_decoder",),
        )
    )

    try:
        ranked_payload, ranked_meta = encode_pr101_ranked_sidecar_payload(
            source_dims,
            source_deltas,
        )
    except ValueError as exc:
        candidates.append(
            PR106SidecarRecodeCandidate(
                name="pr101_ranked_no_op_sidecar_format_0x02",
                encoded_bytes=b"",
                decoded_dims=source_dims,
                decoded_delta_q=source_deltas,
                sidecar_format_id=PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
                framing_meta_bytes=b"",
                runtime_decoder_implemented=False,
                notes=(f"not_applicable:{exc}",),
            )
        )
    else:
        candidates.append(
            _candidate(
                name="pr101_ranked_no_op_sidecar_format_0x02",
                encoded_bytes=ranked_payload,
                decoded=decode_pr101_ranked_sidecar_payload_to_dim_delta(
                    ranked_payload,
                    ranked_meta,
                ),
                sidecar_format_id=PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
                framing_meta_bytes=ranked_meta,
                runtime_decoder_implemented=True,
                notes=("existing_pr106_r2_pr101_grammar_runtime_consumed_format",),
            )
        )
        if ranked_meta == _fixed_meta_framing_bytes():
            fixed_payload = (
                ranked_payload[:PR106_PR101_FIXED_META_DIM_BYTES]
                + ranked_payload[
                    PR106_PR101_FIXED_META_DIM_BYTES
                    + PR106_PR101_FIXED_META_RANK_BYTES :
                ]
            )
            candidates.append(
                _candidate(
                    name="pr101_fixed_meta_rank_elided_sidecar_format_0x05",
                    encoded_bytes=fixed_payload,
                    decoded=decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
                        fixed_payload,
                    ),
                    sidecar_format_id=PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_derived_fixed_meta_consumed_format",
                        "fixed_meta:noops=0,dim_bytes=375,rank_bytes=1,noop_rank_bytes=1",
                    ),
                )
            )
            candidates.append(
                _candidate(
                    name=(
                        "pr101_implicit_len_fixed_meta_rank_elided_sidecar_"
                        "format_0x06"
                    ),
                    encoded_bytes=fixed_payload,
                    decoded=decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
                        fixed_payload,
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_derived_pr106_len_from_fixed_sidecar_tail",
                        "fixed_sidecar_bytes=526",
                    ),
                )
            )
            candidates.append(
                _candidate(
                    name=(
                        "pr101_headerless_implicit_len_fixed_meta_rank_elided_"
                        "sidecar_format_0x07"
                    ),
                    encoded_bytes=fixed_payload,
                    decoded=decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
                        fixed_payload,
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_derived_pr106_len_from_fixed_sidecar_tail",
                        "headerless_runtime_contract:payload_starts_with_pr106_inner_bytes",
                        "fixed_sidecar_bytes=526",
                    ),
                )
            )
            candidates.append(
                _candidate(
                    name=(
                        "pr101_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_"
                        "sidecar_format_0x08"
                    ),
                    encoded_bytes=fixed_payload,
                    decoded=decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
                        fixed_payload,
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_reconstructs_pr106_inner_header_from_fixed_hdm8_hlm2_lengths",
                        "fixed_decoder_bytes=169974",
                        "fixed_latent_bytes=15776",
                        "fixed_sidecar_bytes=526",
                    ),
                )
            )
            candidates.append(
                _candidate(
                    name=(
                        "pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_"
                        "sidecar_format_0x09"
                    ),
                    encoded_bytes=fixed_payload,
                    decoded=decode_pr101_fixed_meta_rank_elided_sidecar_payload_to_dim_delta(
                        fixed_payload,
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_reconstructs_pr106_inner_header_from_fixed_hdm9_hlm2_lengths",
                        "hdm9_requires_source_pr106_hdm8_hlm2_decoder_payload",
                        "fixed_decoder_bytes=169950",
                        "fixed_latent_bytes=15776",
                        "fixed_sidecar_bytes=526",
                    ),
                )
            )
            if fixed_payload[-PR106_PR101_FIXED_META_NOOP_RANK_BYTES:] != (
                PR106_PR101_FIXED_META_NOOP_RANK_BYTE
            ):
                raise ValueError("fixed-meta sidecar has non-zero no-op rank byte")
            noop_rank_elided_payload = fixed_payload[
                : -PR106_PR101_FIXED_META_NOOP_RANK_BYTES
            ]
            candidates.append(
                _candidate(
                    name=(
                        "pr101_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided_"
                        "sidecar_format_0x0a"
                    ),
                    encoded_bytes=noop_rank_elided_payload,
                    decoded=(
                        decode_pr101_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
                            noop_rank_elided_payload
                        )
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_reconstructs_pr106_inner_header_from_fixed_hdm9_hlm3_lengths",
                        "hdm9_requires_source_pr106_hdm8_or_hdm9_hlm2_decoder_payload",
                        "hlm3_elides_hlm2_lo_brotli_len",
                        "fixed_sidecar_noop_rank_elided_bytes=525",
                    ),
                )
            )
            candidates.append(
                _candidate(
                    name=(
                        "pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_"
                        "sidecar_format_0x0b"
                    ),
                    encoded_bytes=noop_rank_elided_payload,
                    decoded=(
                        decode_pr101_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
                            noop_rank_elided_payload
                        )
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_reconstructs_pr106_inner_header_from_fixed_hdm9_hlm3_lengths",
                        "runtime_reconstructs_fixed_hdm9_and_hlm3_section_magics",
                        "hdm9_requires_source_pr106_hdm8_or_hdm9_hlm2_decoder_payload",
                        "hlm3_elides_hlm2_lo_brotli_len",
                        "fixed_sidecar_noop_rank_elided_bytes=525",
                    ),
                )
            )
            exact_radix_dim = _encode_exact_radix_dim_payload(source_dims)
            huff_payload = noop_rank_elided_payload[PR106_PR101_FIXED_META_DIM_BYTES:]
            exact_radix_payload = exact_radix_dim + huff_payload
            candidates.append(
                _candidate(
                    name=(
                        "pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_"
                        "noop_rank_elided_sidecar_format_0x0c"
                    ),
                    encoded_bytes=exact_radix_payload,
                    decoded=(
                        decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload_to_dim_delta(
                            exact_radix_payload
                        )
                    ),
                    sidecar_format_id=(
                        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                    ),
                    runtime_decoder_implemented=True,
                    notes=(
                        "runtime_reconstructs_pr106_inner_header_from_fixed_hdm9_hlm3_lengths",
                        "runtime_reconstructs_fixed_hdm9_and_hlm3_section_magics",
                        "runtime_derives_exact_radix_dim_bytes=361",
                        "hdm9_requires_source_pr106_hdm8_or_hdm9_hlm2_decoder_payload",
                        "hlm3_elides_hlm2_lo_brotli_len",
                        "exact_radix_fixed_sidecar_noop_rank_elided_bytes=511",
                    ),
                )
            )

    for candidate in candidates:
        if candidate.encoded_bytes and not (
            np.array_equal(candidate.decoded_dims, source_dims)
            and np.array_equal(candidate.decoded_delta_q, source_deltas)
        ):
            raise ValueError(f"candidate {candidate.name} failed lossless sidecar equivalence")
    candidates.sort(key=lambda item: (item.charged_bytes if item.encoded_bytes else 10**9, item.name))
    return candidates


def read_single_stored_member_archive(
    archive_bytes: bytes,
    *,
    expected_member_name: str | None = None,
) -> StoredZipMember:
    """Return the only stored member from a PR106-style archive ZIP.

    Public HNeRV replay artifacts use both ``0.bin`` and single-member ``x``
    packets. When no explicit member name is provided, accept the known
    single-member packet names and preserve the original name on re-emit.
    """

    if (
        expected_member_name is not None
        and expected_member_name not in PR106_ALLOWED_SINGLE_MEMBER_NAMES
    ):
        raise ValueError(
            "unsupported expected PR106 ZIP member "
            f"{expected_member_name!r}; expected one of {PR106_ALLOWED_SINGLE_MEMBER_NAMES!r}"
        )

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"expected one ZIP member; got {len(infos)}")
        info = infos[0]
        allowed_names = (
            (expected_member_name,)
            if expected_member_name is not None
            else PR106_ALLOWED_SINGLE_MEMBER_NAMES
        )
        if info.filename not in allowed_names:
            raise ValueError(
                f"expected ZIP member in {allowed_names!r}; got {info.filename!r}"
            )
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"expected stored ZIP member; got method={info.compress_type}")
        return StoredZipMember(
            name=info.filename,
            payload=zf.read(info.filename),
            date_time=info.date_time,
            external_attr=info.external_attr,
            create_system=info.create_system,
            flag_bits=info.flag_bits,
            comment=info.comment,
            extra=info.extra,
            archive_comment=zf.comment,
        )


def emit_single_stored_member_archive(member: StoredZipMember) -> bytes:
    """Emit a single-member stored ZIP preserving source ZIP metadata."""

    out = io.BytesIO()
    info = zipfile.ZipInfo(member.name, date_time=member.date_time)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = member.external_attr
    info.create_system = member.create_system
    info.flag_bits = member.flag_bits
    info.comment = member.comment
    info.extra = member.extra
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(info, member.payload)
        zf.comment = member.archive_comment
    return out.getvalue()


def parse_pr106_sidecar_packet(
    payload: bytes,
    *,
    supported_formats: tuple[int, ...] = PR106_SUPPORTED_SIDECAR_FORMATS,
) -> PR106SidecarPacket:
    """Parse PR106 sidecar ``0.bin`` bytes into typed sections."""

    if len(payload) < 8:
        raise ValueError(f"PR106 sidecar payload too short: {len(payload)} bytes")
    if payload[0] != PR106_SIDECAR_MAGIC:
        if payload[0] == PR106_HEADERLESS_INNER_FIRST_BYTE:
            headerless_format = (
                PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
            )
            if headerless_format not in supported_formats:
                raise ValueError(
                    f"sidecar magic mismatch: got 0x{payload[0]:02X}, "
                    f"expected 0x{PR106_SIDECAR_MAGIC:02X}"
                )
            if len(payload) < PR106_PR101_FIXED_META_PAYLOAD_BYTES + 1:
                raise ValueError("headerless implicit-len sidecar truncated before payload")
            pr106_bytes = payload[:-PR106_PR101_FIXED_META_PAYLOAD_BYTES]
            sidecar = payload[-PR106_PR101_FIXED_META_PAYLOAD_BYTES:]
            _validate_headerless_pr106_bytes(
                pr106_bytes,
                context="format_id=0x07 headerless packet",
            )
            reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(sidecar)
            return PR106SidecarPacket(
                format_id=headerless_format,
                pr106_bytes=pr106_bytes,
                sidecar_payload=sidecar,
                framing_meta=None,
            )
        hdm8_inner_headerless_format = (
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        hdm9_inner_headerless_format = (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        hdm9_hlm3_inner_headerless_format = (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
        )
        hdm9_hlm3_magicless_format = (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
        )
        hdm9_hlm3_magicless_exact_radix_format = (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
        )
        if (
            hdm8_inner_headerless_format not in supported_formats
            and hdm9_inner_headerless_format not in supported_formats
            and hdm9_hlm3_inner_headerless_format not in supported_formats
            and hdm9_hlm3_magicless_format not in supported_formats
            and hdm9_hlm3_magicless_exact_radix_format not in supported_formats
        ):
            raise ValueError(
                f"sidecar magic mismatch: got 0x{payload[0]:02X}, "
                f"expected 0x{PR106_SIDECAR_MAGIC:02X}"
            )
        if (
            hdm9_hlm3_inner_headerless_format in supported_formats
            and len(payload) >= PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES + 1
        ):
            sidecar = payload[-PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES:]
            inner_without_header = payload[
                :-PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
            ]
            if inner_without_header.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
                latent_offset = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
                if (
                    len(inner_without_header)
                    == PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
                    + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
                    and inner_without_header[latent_offset : latent_offset + 4]
                    == PR106_HDM9_HLM3_LATENT_MAGIC
                ):
                    pr106_bytes = _reconstruct_hdm9_hlm3_inner_headerless_pr106_bytes(
                        inner_without_header
                    )
                    reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(sidecar)
                    return PR106SidecarPacket(
                        format_id=hdm9_hlm3_inner_headerless_format,
                        pr106_bytes=pr106_bytes,
                        sidecar_payload=sidecar,
                        framing_meta=None,
                    )
        if (
            hdm9_hlm3_magicless_format in supported_formats
            and len(payload)
            == (
                PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
                - len(PR106_HDM9_HLM2_DECODER_MAGIC)
                + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
                - len(PR106_HDM9_HLM3_LATENT_MAGIC)
                + PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
            )
        ):
            sidecar = payload[-PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES:]
            magicless = payload[
                :-PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
            ]
            pr106_bytes = _reconstruct_hdm9_hlm3_magicless_pr106_bytes(magicless)
            reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(sidecar)
            return PR106SidecarPacket(
                format_id=hdm9_hlm3_magicless_format,
                pr106_bytes=pr106_bytes,
                sidecar_payload=sidecar,
                framing_meta=None,
            )
        if (
            hdm9_hlm3_magicless_exact_radix_format in supported_formats
            and len(payload)
            == (
                PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
                - len(PR106_HDM9_HLM2_DECODER_MAGIC)
                + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
                - len(PR106_HDM9_HLM3_LATENT_MAGIC)
                + PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
            )
        ):
            sidecar = payload[
                -PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES:
            ]
            magicless = payload[
                :-PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
            ]
            pr106_bytes = _reconstruct_hdm9_hlm3_magicless_pr106_bytes(magicless)
            reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
                sidecar
            )
            return PR106SidecarPacket(
                format_id=hdm9_hlm3_magicless_exact_radix_format,
                pr106_bytes=pr106_bytes,
                sidecar_payload=sidecar,
                framing_meta=None,
            )
        if len(payload) < PR106_PR101_FIXED_META_PAYLOAD_BYTES + 1:
            raise ValueError("HDM inner-headerless sidecar truncated before payload")
        inner_without_header = payload[:-PR106_PR101_FIXED_META_PAYLOAD_BYTES]
        sidecar = payload[-PR106_PR101_FIXED_META_PAYLOAD_BYTES:]
        if inner_without_header.startswith(PR106_HDM8_HLM2_DECODER_MAGIC):
            if hdm8_inner_headerless_format not in supported_formats:
                raise ValueError("format_id=0x08 HDM8/HLM2 packet not supported")
            pr106_bytes = _reconstruct_hdm8_hlm2_inner_headerless_pr106_bytes(
                inner_without_header
            )
            format_id = hdm8_inner_headerless_format
        elif inner_without_header.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
            if hdm9_inner_headerless_format not in supported_formats:
                raise ValueError("format_id=0x09 HDM9/HLM2 packet not supported")
            pr106_bytes = _reconstruct_hdm9_hlm2_inner_headerless_pr106_bytes(
                inner_without_header
            )
            format_id = hdm9_inner_headerless_format
        else:
            raise ValueError("HDM inner-headerless payload missing HDM8/HDM9 magic")
        reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(sidecar)
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=None,
        )
    format_id = payload[1]
    if format_id not in supported_formats:
        expected = ", ".join(f"0x{value:02X}" for value in supported_formats)
        raise ValueError(
            f"unsupported PR106 sidecar format_id=0x{format_id:02X}; expected {expected}"
        )
    pos = 2
    if (
        format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    ):
        if len(payload) < pos + PR106_PR101_FIXED_META_PAYLOAD_BYTES:
            raise ValueError("implicit-len fixed-meta sidecar truncated before payload")
        pr106_bytes = payload[pos:-PR106_PR101_FIXED_META_PAYLOAD_BYTES]
        sidecar = payload[-PR106_PR101_FIXED_META_PAYLOAD_BYTES:]
        reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(sidecar)
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=None,
        )

    (pr106_len,) = struct.unpack_from("<I", payload, pos)
    pos += 4
    end_pr106 = pos + pr106_len
    if end_pr106 > len(payload):
        raise ValueError(
            f"PR106 inner payload truncated: pr106_len={pr106_len} total={len(payload)}"
        )
    pr106_bytes = payload[pos:end_pr106]
    pos = end_pr106

    if format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        base_end = pos + PR106_PR101_EXACT_RADIX_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
        if base_end > len(payload):
            raise ValueError("format0D packet truncated before base format0C sidecar")
        base_sidecar = payload[pos:base_end]
        pos = base_end
        reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
            base_sidecar
        )
        if pos + 2 > len(payload):
            raise ValueError("format0D packet truncated before extra_payload_len")
        (extra_len,) = struct.unpack_from("<H", payload, pos)
        pos += 2
        extra_end = pos + extra_len
        if extra_end > len(payload):
            raise ValueError(
                f"format0D extra stream truncated: extra_len={extra_len} total={len(payload)}"
            )
        extra_sidecar = payload[pos:extra_end]
        pos = extra_end
        if pos + 6 > len(payload):
            raise ValueError("format0D packet truncated before extra framing_meta")
        extra_meta = payload[pos : pos + 6]
        pos += 6
        if pos != len(payload):
            raise ValueError(
                f"format0D packet trailing bytes: pos={pos} total={len(payload)}"
            )
        decode_pr101_ranked_sidecar_payload_to_dim_delta(extra_sidecar, extra_meta)
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=base_sidecar,
            framing_meta=None,
            extra_sidecar_payload=extra_sidecar,
            extra_framing_meta=extra_meta,
        )

    if format_id == PR106_SIDECAR_FORMAT_BROTLI:
        if pos + 2 > len(payload):
            raise ValueError("brotli sidecar truncated before sidecar_len")
        (sidecar_len,) = struct.unpack_from("<H", payload, pos)
        pos += 2
        end_sidecar = pos + sidecar_len
        if end_sidecar > len(payload):
            raise ValueError(
                f"brotli sidecar truncated: sidecar_len={sidecar_len} total={len(payload)}"
            )
        sidecar = payload[pos:end_sidecar]
        if end_sidecar != len(payload):
            raise ValueError(
                f"brotli sidecar trailing bytes: pos={end_sidecar} total={len(payload)}"
            )
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=None,
        )

    if format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if pos + 2 > len(payload):
            raise ValueError("PR101 grammar sidecar truncated before payload_len")
        (sidecar_len,) = struct.unpack_from("<H", payload, pos)
        pos += 2
        end_sidecar = pos + sidecar_len
        if end_sidecar > len(payload):
            raise ValueError(
                f"PR101 grammar sidecar truncated: payload_len={sidecar_len} total={len(payload)}"
            )
        sidecar = payload[pos:end_sidecar]
        pos = end_sidecar
        if pos + 6 > len(payload):
            raise ValueError("PR101 grammar sidecar truncated before framing_meta")
        framing_meta = payload[pos : pos + 6]
        pos += 6
        if pos != len(payload):
            raise ValueError(
                f"PR101 grammar sidecar trailing bytes: pos={pos} total={len(payload)}"
            )
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=framing_meta,
        )

    if format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        if len(payload) - pos < 5:
            raise ValueError("rank-elided sidecar truncated before framing_meta")
        sidecar = payload[pos:-5]
        framing_meta = payload[-5:]
        reexpand_pr101_rank_elided_sidecar_payload(sidecar, framing_meta)
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=framing_meta,
        )

    if format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED:
        sidecar = payload[pos:]
        reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(sidecar)
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=None,
        )

    raise ValueError(f"unsupported PR106 sidecar format_id=0x{format_id:02X}")


def compose_pr106_sidecar_dim_delta_corrections(
    base_dim_arr: np.ndarray,
    base_delta_q_arr: np.ndarray,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compose score-table-selected deltas onto decoded base sidecar state.

    The runtime sidecar grammar can express at most one corrected latent
    dimension per pair. If a selected score-table correction would need a
    second latent dimension for the same pair, this helper fails closed instead
    of silently dropping one of the effects.
    """

    base_dims, base_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        base_dim_arr,
        base_delta_q_arr,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    selected_dims, selected_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        selected_dim_arr,
        selected_delta_q_arr,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    if base_dims.shape != selected_dims.shape:
        raise ValueError(
            "base and selected correction shapes differ: "
            f"{base_dims.shape} vs {selected_dims.shape}"
        )

    out_dims = base_dims.copy()
    out_deltas = base_deltas.copy()
    selected_nonzero = (selected_dims != schema.no_op_sentinel) & (selected_deltas != 0)
    base_nonzero = (base_dims != schema.no_op_sentinel) & (base_deltas != 0)
    same_dim_count = 0
    base_noop_count = 0
    changed_pairs = 0

    for pair_idx in np.flatnonzero(selected_nonzero).tolist():
        selected_dim = int(selected_dims[pair_idx])
        selected_delta = int(selected_deltas[pair_idx])
        if not bool(base_nonzero[pair_idx]):
            out_dims[pair_idx] = np.uint8(selected_dim)
            out_deltas[pair_idx] = np.int8(selected_delta)
            base_noop_count += 1
            changed_pairs += 1
            continue
        base_dim = int(base_dims[pair_idx])
        base_delta = int(base_deltas[pair_idx])
        if base_dim != selected_dim:
            raise ValueError(
                "cannot compose selected score-table correction for pair "
                f"{pair_idx}: base dim {base_dim} and selected dim {selected_dim} "
                "would require two latent dimensions in a one-dim sidecar grammar"
            )
        combined_delta = base_delta + selected_delta
        if not (-128 <= combined_delta <= 127):
            raise ValueError(
                f"composed delta for pair {pair_idx} overflows int8: {combined_delta}"
            )
        out_deltas[pair_idx] = np.int8(combined_delta)
        if combined_delta == 0:
            out_dims[pair_idx] = np.uint8(schema.no_op_sentinel)
        else:
            out_dims[pair_idx] = np.uint8(base_dim)
        same_dim_count += 1
        if combined_delta != base_delta:
            changed_pairs += 1

    out_dims, out_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        out_dims,
        out_deltas,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    diagnostics: dict[str, object] = {
        "schema": "pr106_sidecar_dim_delta_composition_v1",
        "composition_policy": "add_selected_delta_to_decoded_base_sidecar_state",
        "n_pairs": int(out_dims.size),
        "selected_nonzero_pair_count": int(selected_nonzero.sum()),
        "base_nonzero_pair_count": int(base_nonzero.sum()),
        "composed_same_dim_pair_count": int(same_dim_count),
        "selected_into_base_noop_pair_count": int(base_noop_count),
        "changed_pair_count": int(changed_pairs),
        "base_dim_sha256": sha256_hex(base_dims.astype(np.uint8).tobytes()),
        "base_delta_q_sha256": sha256_hex(base_deltas.astype(np.int8).tobytes()),
        "selected_dim_sha256": sha256_hex(selected_dims.astype(np.uint8).tobytes()),
        "selected_delta_q_sha256": sha256_hex(
            selected_deltas.astype(np.int8).tobytes()
        ),
        "composed_dim_sha256": sha256_hex(out_dims.astype(np.uint8).tobytes()),
        "composed_delta_q_sha256": sha256_hex(out_deltas.astype(np.int8).tobytes()),
    }
    return out_dims, out_deltas, diagnostics


def _decode_brotli_sidecar_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    return decode_brotli_dim_delta_sidecar_payload(payload)


def _encode_brotli_sidecar_payload(
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
) -> bytes:
    return encode_brotli_dim_delta_sidecar_payload(dim_arr, delta_q_arr)


def _decode_pr101_ranked_sidecar_payload(
    payload: bytes,
    framing_meta: bytes,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[np.ndarray, np.ndarray]:
    if len(framing_meta) != 6:
        raise ValueError(f"PR101 framing_meta must be 6 bytes; got {len(framing_meta)}")
    noop_count, dim_bytes, rank_bytes, noop_rank_bytes = struct.unpack(
        "<HHBB", framing_meta
    )
    dims, delta_indices = decode_ranked_no_op_sidecar(
        payload,
        schema=schema,
        dim_bytes=int(dim_bytes),
        rank_bytes=int(rank_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
        noop_count=int(noop_count),
    )
    return dims.astype(np.int64), delta_indices.astype(np.int64)


def _first_corrected_pair(dim_arr: np.ndarray, *, no_op_dim: int) -> int:
    corrected = np.where(np.asarray(dim_arr, dtype=np.int64) != int(no_op_dim))[0]
    if corrected.size == 0:
        raise ValueError("cannot mutate sidecar with zero corrected pairs")
    return int(corrected[0])


def _next_nonzero_int8(value: int) -> int:
    if value >= 127:
        return value - 1
    candidate = value + 1
    if candidate == 0:
        candidate = 1
    if not (-128 <= candidate <= 127):
        raise ValueError(f"int8 mutation out of range: {candidate}")
    return candidate


def mutate_pr106_sidecar_semantic_correction(
    packet: PR106SidecarPacket,
    *,
    pair_index: int | None = None,
    section_name: str | None = None,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[PR106SidecarPacket, PR106SidecarMutation]:
    """Return a valid packet with one runtime-visible sidecar correction changed.

    This helper is the deterministic mutation primitive for no-op/runtime
    consumption tests. It preserves the inner PR106 payload and only mutates
    ``sidecar_payload`` bytes through the sidecar's own grammar, so downstream
    tools can prove that the submission runtime's sidecar decoder sees the
    change before any exact-eval score is claimed.
    """

    if section_name not in (
        None,
        "sidecar_payload",
        "base_format0c_sidecar_payload",
        "extra_pr101_ranked_no_op_payload",
    ):
        raise ValueError(f"unsupported mutation section_name={section_name!r}")

    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
        if section_name not in (None, "sidecar_payload"):
            raise ValueError(
                f"format_id=0x{packet.format_id:02X} cannot mutate {section_name}"
            )
        dim_arr, delta_q_arr = _decode_brotli_sidecar_payload(packet.sidecar_payload)
        idx = (
            _first_corrected_pair(dim_arr, no_op_dim=PR106_NO_OP_DIM)
            if pair_index is None
            else int(pair_index)
        )
        if not (0 <= idx < dim_arr.size):
            raise ValueError(f"pair_index out of range: {idx} not in [0, {dim_arr.size})")
        if int(dim_arr[idx]) == PR106_NO_OP_DIM:
            raise ValueError(f"pair_index {idx} is a no-op correction")
        old_delta = int(delta_q_arr[idx])
        new_delta = _next_nonzero_int8(old_delta)
        mutated_delta = delta_q_arr.copy()
        mutated_delta[idx] = np.int8(new_delta)
        mutated_payload = _encode_brotli_sidecar_payload(dim_arr, mutated_delta)
        mutated_packet = PR106SidecarPacket(
            format_id=packet.format_id,
            pr106_bytes=packet.pr106_bytes,
            sidecar_payload=mutated_payload,
            framing_meta=packet.framing_meta,
        )
        return mutated_packet, PR106SidecarMutation(
            section_name="sidecar_payload",
            pair_index=idx,
            format_id=packet.format_id,
            old_dim=int(dim_arr[idx]),
            new_dim=int(dim_arr[idx]),
            old_delta_q=old_delta,
            new_delta_q=new_delta,
        )

    if packet.format_id in (
        PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
        PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    ):
        if section_name not in (None, "sidecar_payload"):
            raise ValueError(
                f"format_id=0x{packet.format_id:02X} cannot mutate {section_name}"
            )
        if packet.framing_meta is None:
            if packet.format_id not in (
                PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
            ):
                raise ValueError("format_id=0x02 requires framing_meta")
            if packet.format_id in (
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
            ):
                if (
                    packet.format_id
                    == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                ):
                    ranked_payload, ranked_meta = (
                        reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
                            packet.sidecar_payload,
                            schema=schema,
                        )
                    )
                else:
                    ranked_payload, ranked_meta = (
                        reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(
                            packet.sidecar_payload,
                            schema=schema,
                        )
                    )
            else:
                ranked_payload, ranked_meta = (
                    reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(
                        packet.sidecar_payload,
                        schema=schema,
                    )
                )
        else:
            if packet.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
                ranked_payload, ranked_meta = reexpand_pr101_rank_elided_sidecar_payload(
                    packet.sidecar_payload,
                    packet.framing_meta,
                    schema=schema,
                )
            else:
                ranked_payload = packet.sidecar_payload
                ranked_meta = packet.framing_meta
        dims, delta_indices = _decode_pr101_ranked_sidecar_payload(
            ranked_payload,
            ranked_meta,
            schema=schema,
        )
        idx = (
            _first_corrected_pair(dims, no_op_dim=schema.no_op_sentinel)
            if pair_index is None
            else int(pair_index)
        )
        if not (0 <= idx < dims.size):
            raise ValueError(f"pair_index out of range: {idx} not in [0, {dims.size})")
        if int(dims[idx]) == schema.no_op_sentinel:
            raise ValueError(f"pair_index {idx} is a no-op correction")
        noop_count, dim_bytes, rank_bytes, noop_rank_bytes = struct.unpack(
            "<HHBB", ranked_meta
        )
        old_delta_index = int(delta_indices[idx])
        new_delta_index = (old_delta_index + 1) % len(schema.deltas)
        mutated_delta_indices = delta_indices.copy()
        mutated_delta_indices[idx] = new_delta_index
        mutated_payload = encode_ranked_no_op_sidecar(
            dims=dims,
            delta_indices=mutated_delta_indices,
            schema=schema,
            dim_bytes=int(dim_bytes),
            noop_rank_bytes=int(noop_rank_bytes),
        )
        mutated_sidecar_payload = mutated_payload
        mutated_framing_meta = packet.framing_meta
        if packet.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
            if int(rank_bytes) != 1:
                raise ValueError(
                    f"format_id=0x04 rank-elision expects one rank byte; got {rank_bytes}"
                )
            mutated_sidecar_payload = (
                mutated_payload[:dim_bytes] + mutated_payload[dim_bytes + rank_bytes :]
            )
            mutated_framing_meta = struct.pack(
                "<HHB",
                int(noop_count),
                int(dim_bytes),
                int(noop_rank_bytes),
            )
            reexpand_pr101_rank_elided_sidecar_payload(
                mutated_sidecar_payload,
                mutated_framing_meta,
                schema=schema,
            )
        if packet.format_id in (
            PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
        ):
            exact_radix_format = (
                packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
            )
            mutated_meta = (
                _exact_radix_fixed_meta_framing_bytes()
                if exact_radix_format
                else _fixed_meta_framing_bytes()
            )
            if ranked_meta != mutated_meta:
                raise ValueError(
                    f"format_id=0x{packet.format_id:02X} expanded metadata "
                    "did not match fixed meta"
            )
            if exact_radix_format:
                huff_start = int(dim_bytes) + int(rank_bytes)
                huff_end = -int(noop_rank_bytes)
                mutated_sidecar_payload = (
                    _encode_exact_radix_dim_payload(dims, schema=schema)
                    + mutated_payload[huff_start:huff_end]
                    + PR106_PR101_FIXED_META_NOOP_RANK_BYTE
                )
            else:
                rank_elided_start = int(dim_bytes) + int(rank_bytes)
                mutated_sidecar_payload = (
                    mutated_payload[: int(dim_bytes)]
                    + mutated_payload[rank_elided_start:]
                )
            if (
                packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
                or packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
                or packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
            ):
                if mutated_sidecar_payload[-1:] != PR106_PR101_FIXED_META_NOOP_RANK_BYTE:
                    raise ValueError(
                        f"format_id=0x{packet.format_id:02X} mutated no-op rank must stay zero"
                    )
                mutated_sidecar_payload = mutated_sidecar_payload[:-1]
            mutated_framing_meta = None
        mutated_packet = PR106SidecarPacket(
            format_id=packet.format_id,
            pr106_bytes=packet.pr106_bytes,
            sidecar_payload=mutated_sidecar_payload,
            framing_meta=mutated_framing_meta,
        )
        return mutated_packet, PR106SidecarMutation(
            section_name="sidecar_payload",
            pair_index=idx,
            format_id=packet.format_id,
            old_dim=int(dims[idx]),
            new_dim=int(dims[idx]),
            old_delta_q=int(schema.deltas[old_delta_index]),
            new_delta_q=int(schema.deltas[new_delta_index]),
            old_delta_index=old_delta_index,
            new_delta_index=new_delta_index,
        )

    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        if packet.framing_meta is not None:
            raise ValueError("format_id=0x0D must not carry base framing_meta")
        if packet.extra_framing_meta is None:
            raise ValueError("format_id=0x0D requires extra_framing_meta")
        if section_name == "sidecar_payload":
            section_name = "extra_pr101_ranked_no_op_payload"
        if section_name in (None, "extra_pr101_ranked_no_op_payload"):
            dims, delta_indices = _decode_pr101_ranked_sidecar_payload(
                packet.extra_sidecar_payload,
                packet.extra_framing_meta,
                schema=schema,
            )
            idx = (
                _first_corrected_pair(dims, no_op_dim=schema.no_op_sentinel)
                if pair_index is None
                else int(pair_index)
            )
            if not (0 <= idx < dims.size):
                raise ValueError(f"pair_index out of range: {idx} not in [0, {dims.size})")
            if int(dims[idx]) == schema.no_op_sentinel:
                raise ValueError(f"pair_index {idx} is a no-op correction")
            old_delta_index = int(delta_indices[idx])
            new_delta_index = (old_delta_index + 1) % len(schema.deltas)
            mutated_delta_indices = delta_indices.copy()
            mutated_delta_indices[idx] = new_delta_index
            mutated_extra_payload = encode_ranked_no_op_sidecar(
                dims=dims,
                delta_indices=mutated_delta_indices,
                schema=schema,
                dim_bytes=int.from_bytes(packet.extra_framing_meta[2:4], "little"),
                noop_rank_bytes=int(packet.extra_framing_meta[5]),
            )
            mutated_packet = PR106SidecarPacket(
                format_id=packet.format_id,
                pr106_bytes=packet.pr106_bytes,
                sidecar_payload=packet.sidecar_payload,
                framing_meta=None,
                extra_sidecar_payload=mutated_extra_payload,
                extra_framing_meta=packet.extra_framing_meta,
            )
            return mutated_packet, PR106SidecarMutation(
                section_name="extra_pr101_ranked_no_op_payload",
                pair_index=idx,
                format_id=packet.format_id,
                old_dim=int(dims[idx]),
                new_dim=int(dims[idx]),
                old_delta_q=int(schema.deltas[old_delta_index]),
                new_delta_q=int(schema.deltas[new_delta_index]),
                old_delta_index=old_delta_index,
                new_delta_index=new_delta_index,
            )
        if section_name != "base_format0c_sidecar_payload":
            raise ValueError(f"format_id=0x0D cannot mutate {section_name}")
        dims, delta_indices = _decode_pr101_ranked_sidecar_payload(
            *reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
                packet.sidecar_payload,
                schema=schema,
            ),
            schema=schema,
        )
        idx = (
            _first_corrected_pair(dims, no_op_dim=schema.no_op_sentinel)
            if pair_index is None
            else int(pair_index)
        )
        if not (0 <= idx < dims.size):
            raise ValueError(f"pair_index out of range: {idx} not in [0, {dims.size})")
        if int(dims[idx]) == schema.no_op_sentinel:
            raise ValueError(f"pair_index {idx} is a no-op correction")
        old_delta_index = int(delta_indices[idx])
        new_delta_index = (old_delta_index + 1) % len(schema.deltas)
        mutated_delta_indices = delta_indices.copy()
        mutated_delta_indices[idx] = new_delta_index
        mutated_deltas = np.array(
            [schema.deltas[int(delta_index)] for delta_index in mutated_delta_indices],
            dtype=np.int8,
        )
        mutated_base_payload = encode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
            dims.astype(np.uint8, copy=False),
            mutated_deltas,
            schema=schema,
        )
        mutated_packet = PR106SidecarPacket(
            format_id=packet.format_id,
            pr106_bytes=packet.pr106_bytes,
            sidecar_payload=mutated_base_payload,
            framing_meta=None,
            extra_sidecar_payload=packet.extra_sidecar_payload,
            extra_framing_meta=packet.extra_framing_meta,
        )
        return mutated_packet, PR106SidecarMutation(
            section_name="base_format0c_sidecar_payload",
            pair_index=idx,
            format_id=packet.format_id,
            old_dim=int(dims[idx]),
            new_dim=int(dims[idx]),
            old_delta_q=int(schema.deltas[old_delta_index]),
            new_delta_q=int(schema.deltas[new_delta_index]),
            old_delta_index=old_delta_index,
            new_delta_index=new_delta_index,
        )

    raise ValueError(f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}")


def emit_pr106_sidecar_packet(packet: PR106SidecarPacket) -> bytes:
    """Emit ``0.bin`` bytes from a parsed PR106 sidecar packet."""

    if packet.format_id not in PR106_SUPPORTED_SIDECAR_FORMATS:
        raise ValueError(f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}")
    if len(packet.pr106_bytes) > 0xFFFF_FFFF:
        raise ValueError("PR106 inner payload too large for u32 length field")
    if packet.format_id not in (
        PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    ) and len(packet.sidecar_payload) > 0xFFFF:
        raise ValueError("sidecar payload too large for u16 length field")
    if (
        packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
        and len(packet.extra_sidecar_payload) > 0xFFFF
    ):
        raise ValueError("format0D extra sidecar payload too large for u16 length field")
    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI and packet.framing_meta is not None:
        raise ValueError("format_id=0x01 must not carry framing_meta")
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x02 requires framing_meta")
        if len(packet.framing_meta) != 6:
            raise ValueError(
                f"format_id=0x02 framing_meta must be 6 bytes; got {len(packet.framing_meta)}"
            )
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x04 requires framing_meta")
        reexpand_pr101_rank_elided_sidecar_payload(
            packet.sidecar_payload,
            packet.framing_meta,
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED:
        if packet.framing_meta is not None:
            raise ValueError("format_id=0x05 derives framing_meta and must not carry it")
        reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(packet.sidecar_payload)
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        or packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
        )
        or packet.format_id
        == (PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED)
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
        )
    ):
        if packet.framing_meta is not None:
            raise ValueError(
                f"format_id=0x{packet.format_id:02X} derives framing_meta and must not carry it"
            )
        if (
            packet.format_id
            == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
            or packet.format_id
            == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
        ):
            reexpand_pr101_fixed_meta_noop_rank_elided_sidecar_payload(
                packet.sidecar_payload
            )
        elif (
            packet.format_id
            == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
        ):
            reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
                packet.sidecar_payload
            )
        else:
            reexpand_pr101_fixed_meta_rank_elided_sidecar_payload(packet.sidecar_payload)
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        if packet.framing_meta is not None:
            raise ValueError("format_id=0x0D must not carry base framing_meta")
        if packet.extra_framing_meta is None:
            raise ValueError("format_id=0x0D requires extra_framing_meta")
        if len(packet.extra_framing_meta) != 6:
            raise ValueError(
                "format_id=0x0D extra_framing_meta must be 6 bytes; "
                f"got {len(packet.extra_framing_meta)}"
            )
        reexpand_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
            packet.sidecar_payload
        )
        decode_pr101_ranked_sidecar_payload_to_dim_delta(
            packet.extra_sidecar_payload,
            packet.extra_framing_meta,
        )
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    ):
        if not packet.pr106_bytes:
            raise ValueError("format_id=0x07 requires non-empty PR106 inner payload")
        _validate_headerless_pr106_bytes(
            packet.pr106_bytes,
            context="format_id=0x07 headerless packet",
        )
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        _hdm8_hlm2_inner_without_header_payload(packet.pr106_bytes)
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        _hdm9_hlm2_inner_without_header_payload(packet.pr106_bytes)
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        _hdm9_hlm3_inner_without_header_payload(packet.pr106_bytes)
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        _hdm9_hlm3_magicless_payload(packet.pr106_bytes)
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        _hdm9_hlm3_magicless_payload(packet.pr106_bytes)

    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    ):
        prefix = bytes([PR106_SIDECAR_MAGIC, packet.format_id]) + packet.pr106_bytes
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    ):
        prefix = packet.pr106_bytes
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        prefix = _hdm8_hlm2_inner_without_header_payload(packet.pr106_bytes)
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        prefix = _hdm9_hlm2_inner_without_header_payload(packet.pr106_bytes)
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        prefix = _hdm9_hlm3_inner_without_header_payload(packet.pr106_bytes)
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
        or packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        prefix = _hdm9_hlm3_magicless_payload(packet.pr106_bytes)
    else:
        prefix = (
            bytes([PR106_SIDECAR_MAGIC, packet.format_id])
            + struct.pack("<I", len(packet.pr106_bytes))
            + packet.pr106_bytes
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        assert packet.framing_meta is not None
        return (
            prefix
            + struct.pack("<H", len(packet.sidecar_payload))
            + packet.sidecar_payload
            + packet.framing_meta
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        assert packet.extra_framing_meta is not None
        return (
            prefix
            + packet.sidecar_payload
            + struct.pack("<H", len(packet.extra_sidecar_payload))
            + packet.extra_sidecar_payload
            + packet.extra_framing_meta
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
        return prefix + struct.pack("<H", len(packet.sidecar_payload)) + packet.sidecar_payload
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        assert packet.framing_meta is not None
        return prefix + packet.sidecar_payload + packet.framing_meta
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED:
        return prefix + packet.sidecar_payload
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        or packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        )
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
        )
        or packet.format_id
        == (PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED)
        or packet.format_id
        == (
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
        )
    ):
        return prefix + packet.sidecar_payload
    raise ValueError(f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}")


def _consumed_section_row(
    name: str,
    *,
    offset: int,
    payload: bytes,
    score_affecting: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "offset": offset,
        "offset_start": offset,
        "bytes": len(payload),
        "byte_count": len(payload),
        "end_offset": offset + len(payload),
        "offset_end_exclusive": offset + len(payload),
        "sha256": sha256_hex(payload),
        "hash_domain": PR106_PACKET_IR_SECTION_HASH_DOMAIN,
        "sha256_domain": PR106_PACKET_IR_SECTION_HASH_DOMAIN,
        "score_affecting": score_affecting,
    }


def pr106_sidecar_consumed_byte_proof(
    packet: PR106SidecarPacket,
) -> dict[str, object]:
    """Return parser-level proof that every emitted payload byte is accounted for.

    This is intentionally narrower than a runtime-inflate proof. It proves the
    PacketIR grammar consumes the emitted ``0.bin`` payload without gaps or
    trailing bytes; exact runtime consumption still needs an inflate mutation
    smoke or same-runtime eval artifact.
    """

    if packet.format_id not in PR106_SUPPORTED_SIDECAR_FORMATS:
        raise ValueError(
            f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}"
        )

    sections: list[dict[str, object]] = []
    offset = 0

    def add(name: str, payload: bytes, *, score_affecting: bool) -> None:
        nonlocal offset
        sections.append(
            _consumed_section_row(
                name,
                offset=offset,
                payload=payload,
                score_affecting=score_affecting,
            )
        )
        offset += len(payload)

    if (
        packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        and packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        and packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
        and packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
        and packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
        and packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        add("magic", bytes([PR106_SIDECAR_MAGIC]), score_affecting=False)
        add("format_id", bytes([packet.format_id]), score_affecting=False)
    if (
        packet.format_id
        not in (
            PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
        )
    ):
        add(
            "pr106_len_le_u32",
            struct.pack("<I", len(packet.pr106_bytes)),
            score_affecting=False,
        )
    if (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        add(
            "pr106_hdm8_hlm2_payload_without_inner_header",
            _hdm8_hlm2_inner_without_header_payload(packet.pr106_bytes),
            score_affecting=True,
        )
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        add(
            "pr106_hdm9_hlm2_payload_without_inner_header",
            _hdm9_hlm2_inner_without_header_payload(packet.pr106_bytes),
            score_affecting=True,
        )
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        add(
            "pr106_hdm9_hlm3_payload_without_inner_header",
            _hdm9_hlm3_inner_without_header_payload(packet.pr106_bytes),
            score_affecting=True,
        )
    elif (
        packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
        or packet.format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        add(
            "pr106_hdm9_hlm3_payload_without_inner_header_or_section_magic",
            _hdm9_hlm3_magicless_payload(packet.pr106_bytes),
            score_affecting=True,
        )
    else:
        add("pr106_payload", packet.pr106_bytes, score_affecting=True)
    if packet.format_id not in (
        PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
    ):
        add(
            "sidecar_len_le_u16",
            struct.pack("<H", len(packet.sidecar_payload)),
            score_affecting=False,
        )
    if packet.format_id == PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA:
        if packet.extra_framing_meta is None:
            raise ValueError("format_id=0x0D requires extra_framing_meta")
        add("base_format0c_sidecar_payload", packet.sidecar_payload, score_affecting=True)
        add(
            "extra_payload_len_le_u16",
            struct.pack("<H", len(packet.extra_sidecar_payload)),
            score_affecting=False,
        )
        add(
            "extra_pr101_ranked_no_op_payload",
            packet.extra_sidecar_payload,
            score_affecting=True,
        )
        add("extra_framing_meta", packet.extra_framing_meta, score_affecting=True)
    else:
        add("sidecar_payload", packet.sidecar_payload, score_affecting=True)
    if packet.format_id in (
        PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
        PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    ):
        if packet.framing_meta is None:
            raise ValueError(f"format_id=0x{packet.format_id:02X} requires framing_meta")
        add("framing_meta", packet.framing_meta, score_affecting=True)

    emitted = emit_pr106_sidecar_packet(packet)
    accounted = sum(int(row["bytes"]) for row in sections)
    cursor = 0
    gaps: list[dict[str, object]] = []
    for row in sections:
        row_offset = int(row["offset"])
        if row_offset != cursor:
            gaps.append({"expected_offset": cursor, "observed_offset": row_offset})
        cursor = int(row["end_offset"])

    return {
        "schema": "pr106_sidecar_packet_ir_consumed_byte_proof_v1",
        "proof_scope": "packet_ir_parser_accounting_not_runtime_inflate_consumption",
        "runtime_consumption_claim": False,
        "emitted_payload_bytes": len(emitted),
        "emitted_payload_sha256": sha256_hex(emitted),
        "accounted_payload_bytes": accounted,
        "all_payload_bytes_accounted": accounted == len(emitted) and not gaps,
        "unconsumed_trailing_bytes": max(0, len(emitted) - accounted),
        "section_gaps": gaps,
        "score_affecting_section_names": [
            str(row["name"]) for row in sections if bool(row["score_affecting"])
        ],
        "sections": sections,
    }


def pr106_sidecar_manifest(
    packet: PR106SidecarPacket,
    *,
    archive_sha256: str | None = None,
) -> dict[str, object]:
    """Return a small machine-checkable identity manifest for review artifacts."""

    emitted = emit_pr106_sidecar_packet(packet)
    manifest: dict[str, object] = {
        "schema": "pr106_sidecar_packet_ir_manifest_v1",
        "format_id": f"0x{packet.format_id:02X}",
        "sidecar_kind": packet.sidecar_kind,
        "pr106_bytes": len(packet.pr106_bytes),
        "pr106_sha256": sha256_hex(packet.pr106_bytes),
        "sidecar_bytes": len(packet.sidecar_payload),
        "sidecar_sha256": sha256_hex(packet.sidecar_payload),
        "framing_meta_bytes": 0 if packet.framing_meta is None else len(packet.framing_meta),
        "framing_meta_sha256": None
        if packet.framing_meta is None
        else sha256_hex(packet.framing_meta),
        "extra_sidecar_bytes": len(packet.extra_sidecar_payload),
        "extra_sidecar_sha256": sha256_hex(packet.extra_sidecar_payload)
        if packet.extra_sidecar_payload
        else None,
        "extra_framing_meta_bytes": 0
        if packet.extra_framing_meta is None
        else len(packet.extra_framing_meta),
        "extra_framing_meta_sha256": None
        if packet.extra_framing_meta is None
        else sha256_hex(packet.extra_framing_meta),
        "format0d_layout": None
        if packet.format_id != PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA
        else {
            "base_format": "0x0C",
            "base_sidecar_bytes": len(packet.sidecar_payload),
            "extra_payload_len_field_bytes": 2,
            "extra_payload_bytes": len(packet.extra_sidecar_payload),
            "extra_framing_meta_bytes": 0
            if packet.extra_framing_meta is None
            else len(packet.extra_framing_meta),
            "runtime_apply_order": [
                "base_format0c_corrections",
                "extra_pr101_ranked_no_op_corrections",
            ],
        },
        "derived_fixed_meta": None
        if packet.format_id
        not in (
            PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
            PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
        )
        else {
            "fields": (
                {
                    **PR106_PR101_FIXED_META_FIELDS,
                    "dim_bytes": PR106_PR101_EXACT_RADIX_DIM_BYTES,
                }
                if packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                else PR106_PR101_FIXED_META_FIELDS
            ),
            "framing_meta_bytes": (
                len(_exact_radix_fixed_meta_framing_bytes())
                if packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                else len(_fixed_meta_framing_bytes())
            ),
            "framing_meta_sha256": (
                sha256_hex(_exact_radix_fixed_meta_framing_bytes())
                if packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                else sha256_hex(_fixed_meta_framing_bytes())
            ),
            "sidecar_tail_bytes": len(packet.sidecar_payload),
            "noop_rank_elided": (
                packet.format_id
                in (
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
                )
            ),
            "exact_radix_dim_bytes": (
                PR106_PR101_EXACT_RADIX_DIM_BYTES
                if packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                else None
            ),
            "implicit_pr106_len": (
                packet.format_id
                in (
                    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
                    PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
                    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
            )
            ),
            "headerless_packet": (
                packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
            ),
            "hdm8_hlm2_inner_headerless_packet": (
                packet.format_id
                == (
                    PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
                )
            ),
            "hdm9_hlm2_inner_headerless_packet": (
                packet.format_id
                == (
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
                )
            ),
            "hdm9_hlm3_inner_headerless_packet": (
                packet.format_id
                == (
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
                )
            ),
            "hdm9_hlm3_magicless_packet": (
                packet.format_id
                == (
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
                )
            ),
            "hdm9_hlm3_magicless_exact_radix_packet": (
                packet.format_id
                == (
                    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
                )
            ),
            "hdm8_hlm2_fixed_lengths": None
            if packet.format_id
            != PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
            else {
                "decoder_payload_bytes": PR106_HDM8_HLM2_DECODER_PAYLOAD_BYTES,
                "latent_payload_bytes": PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES,
                "decoder_magic": PR106_HDM8_HLM2_DECODER_MAGIC.decode("ascii"),
                "latent_magic": PR106_HDM8_HLM2_LATENT_MAGIC.decode("ascii"),
                "elided_inner_header_bytes": 4,
            },
            "hdm9_hlm2_fixed_lengths": None
            if packet.format_id
            != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
            else {
                "decoder_payload_bytes": PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES,
                "latent_payload_bytes": PR106_HDM8_HLM2_LATENT_PAYLOAD_BYTES,
                "decoder_magic": PR106_HDM9_HLM2_DECODER_MAGIC.decode("ascii"),
                "latent_magic": PR106_HDM8_HLM2_LATENT_MAGIC.decode("ascii"),
                "elided_inner_header_bytes": 4,
                "scale_low3_bytes": PR106_HDM9_SCALE_LOW3_BYTES,
                "scale_high_mask_bytes": PR106_HDM9_SCALE_HIGH_MASK_BYTES,
                "scale_high_base": PR106_HDM9_SCALE_HIGH_BASE,
            },
            "hdm9_hlm3_fixed_lengths": None
            if packet.format_id
            not in (
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
                PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
            )
            else {
                "decoder_payload_bytes": PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES,
                "latent_payload_bytes": PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES,
                "decoder_magic": PR106_HDM9_HLM2_DECODER_MAGIC.decode("ascii"),
                "latent_magic": PR106_HDM9_HLM3_LATENT_MAGIC.decode("ascii"),
                "elided_inner_header_bytes": 4,
                "elided_section_magic_bytes": 0
                if packet.format_id
                == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
                else len(PR106_HDM9_HLM2_DECODER_MAGIC)
                + len(PR106_HDM9_HLM3_LATENT_MAGIC),
                "scale_low3_bytes": PR106_HDM9_SCALE_LOW3_BYTES,
                "scale_high_mask_bytes": PR106_HDM9_SCALE_HIGH_MASK_BYTES,
                "scale_high_base": PR106_HDM9_SCALE_HIGH_BASE,
                "elided_hlm2_lo_brotli_len_bytes": 2,
                "elided_noop_rank_bytes": PR106_PR101_FIXED_META_NOOP_RANK_BYTES,
            },
            "contest_compliance_rationale": (
                "metadata is fixed by the committed runtime grammar and runtime tree SHA; "
                "this manifest makes the implicit fields explicit for review"
            ),
        },
        "emitted_payload_bytes": len(emitted),
        "emitted_payload_sha256": sha256_hex(emitted),
        "packet_ir_consumed_byte_proof": pr106_sidecar_consumed_byte_proof(packet),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if archive_sha256 is not None:
        manifest["archive_sha256"] = archive_sha256
    return manifest


def prove_pr106_sidecar_packet_ir_identity(
    *,
    archive_path: Path,
    expected_member_name: str | None = None,
    expected_archive_sha256: str | None = None,
) -> dict[str, object]:
    """Prove PR106 sidecar PacketIR parse/emit identity for one archive.

    This is the canonical operator-facing identity proof for PR106/R2 sidecar
    packets. It proves only byte custody and parser accounting: it does not run
    the submission runtime, does not render frames, and never makes a score or
    promotion claim.
    """

    archive_path = Path(archive_path)
    archive_bytes = archive_path.read_bytes()
    archive_sha = sha256_hex(archive_bytes)
    expected_archive_sha, expected_archive_sha_well_formed = canonical_expected_sha256(
        expected_archive_sha256
    )
    member = read_single_stored_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    packet = parse_pr106_sidecar_packet(member.payload)
    emitted_payload = emit_pr106_sidecar_packet(packet)
    emitted_member = replace(member, payload=emitted_payload)
    emitted_archive = emit_single_stored_member_archive(emitted_member)
    emitted_archive_sha = sha256_hex(emitted_archive)
    packet_manifest = pr106_sidecar_manifest(packet, archive_sha256=archive_sha)
    consumed = packet_manifest["packet_ir_consumed_byte_proof"]
    if not isinstance(consumed, dict):
        raise TypeError("packet_ir_consumed_byte_proof must be a dict")

    blockers: list[str] = []
    if expected_archive_sha_well_formed is False:
        blockers.append("expected_archive_sha256_malformed")
    if (
        expected_archive_sha_well_formed is True
        and expected_archive_sha is not None
        and archive_sha != expected_archive_sha
    ):
        blockers.append("expected_archive_sha256_mismatch")
    if emitted_payload != member.payload:
        blockers.append("packet_ir_payload_parse_emit_not_identity")
    if emitted_archive != archive_bytes:
        blockers.append("single_member_zip_parse_emit_not_identity")
    if consumed.get("all_payload_bytes_accounted") is not True:
        blockers.append("packet_ir_consumed_byte_accounting_failed")
    if consumed.get("runtime_consumption_claim") is not False:
        blockers.append("packet_ir_overclaimed_runtime_consumption")

    identity_passed = not blockers
    return {
        "schema": "pr106_sidecar_packet_ir_identity_proof_v1",
        "proof_scope": (
            "packet_ir_parse_emit_identity_and_parser_consumed_byte_accounting_"
            "not_runtime_inflate"
        ),
        "archive": {
            "path": archive_path.as_posix(),
            "bytes": len(archive_bytes),
            "sha256": archive_sha,
            "zip_comment_bytes": len(member.archive_comment),
            "zip_comment_sha256": sha256_hex(member.archive_comment),
            "expected_sha256": expected_archive_sha,
            "expected_sha256_well_formed": expected_archive_sha_well_formed,
            "expected_sha256_matches": (
                None
                if expected_archive_sha is None or expected_archive_sha_well_formed is False
                else archive_sha == expected_archive_sha
            ),
        },
        "member": {
            "name": member.name,
            "expected_name": expected_member_name,
            "expected_name_matches": (
                None if expected_member_name is None else member.name == expected_member_name
            ),
            "payload_bytes": len(member.payload),
            "payload_sha256": sha256_hex(member.payload),
        },
        "packet": packet_manifest,
        "emitted_payload": {
            "bytes": len(emitted_payload),
            "sha256": sha256_hex(emitted_payload),
            "byte_identical_to_source_member": emitted_payload == member.payload,
        },
        "emitted_archive": {
            "bytes": len(emitted_archive),
            "sha256": emitted_archive_sha,
            "byte_identical_to_source_archive": emitted_archive == archive_bytes,
        },
        "byte_exact_identity": {
            "source_archive_bytes": len(archive_bytes),
            "source_archive_sha256": archive_sha,
            "source_member_name": member.name,
            "source_member_payload_bytes": len(member.payload),
            "source_member_payload_sha256": sha256_hex(member.payload),
            "emitted_payload_bytes": len(emitted_payload),
            "emitted_payload_sha256": sha256_hex(emitted_payload),
            "emitted_archive_bytes": len(emitted_archive),
            "emitted_archive_sha256": emitted_archive_sha,
            "payload_byte_identical": emitted_payload == member.payload,
            "archive_byte_identical": emitted_archive == archive_bytes,
            "expected_archive_sha256": expected_archive_sha,
            "expected_archive_sha256_matches": (
                None
                if expected_archive_sha is None or expected_archive_sha_well_formed is False
                else archive_sha == expected_archive_sha
            ),
            "expected_member_name": expected_member_name,
            "expected_member_name_matches": (
                None if expected_member_name is None else member.name == expected_member_name
            ),
        },
        "packet_ir_identity_passed": identity_passed,
        "blockers": blockers,
        "proof_not_score": True,
        "evidence_axis": "packet-ir-parser-local-no-score",
        "runtime_consumption_claim": False,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next_proof": (
            "runtime sidecar decode/apply proof, full-frame same-runtime parity, "
            "then exact contest auth eval with axis labels before score language"
        ),
    }


def build_pr106_sidecar_recode_candidate_packet(
    source_packet: PR106SidecarPacket,
    candidate: PR106SidecarRecodeCandidate,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> PR106SidecarPacket:
    """Build a PR106 wrapper packet for a lossless sidecar recode candidate.

    Only candidates with an existing PR106 ``format_id`` can be emitted into the
    current runtime wrapper. New experimental grammars still appear in profiler
    reports, but they remain parser-only until a runtime decoder exists.
    """

    if not candidate.encoded_bytes:
        raise ValueError(f"candidate {candidate.name!r} has no encoded bytes")
    if candidate.sidecar_format_id not in PR106_SUPPORTED_SIDECAR_FORMATS:
        raise ValueError(
            f"candidate {candidate.name!r} has no supported PR106 sidecar format_id"
        )
    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(
        source_packet,
        schema=schema,
    )
    if not (
        np.array_equal(candidate.decoded_dims, source_dims)
        and np.array_equal(candidate.decoded_delta_q, source_deltas)
    ):
        raise ValueError(f"candidate {candidate.name!r} is not lossless vs source packet")

    framing_meta: bytes | None = None
    if candidate.sidecar_format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if len(candidate.framing_meta_bytes) != 6:
            raise ValueError(
                f"candidate {candidate.name!r} requires six framing_meta bytes"
            )
        framing_meta = candidate.framing_meta_bytes
    elif candidate.sidecar_format_id in (
        PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM8_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    ):
        if candidate.framing_meta_bytes:
            raise ValueError(
                f"candidate {candidate.name!r} must not carry explicit framing_meta"
            )
    elif candidate.framing_meta_bytes:
        raise ValueError(
            f"candidate {candidate.name!r} carries framing_meta for non-PR101 format"
        )

    packet = PR106SidecarPacket(
        format_id=int(candidate.sidecar_format_id),
        pr106_bytes=source_packet.pr106_bytes,
        sidecar_payload=candidate.encoded_bytes,
        framing_meta=framing_meta,
    )
    if (
        candidate.sidecar_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM2_INNER_HEADERLESS_FIXED_META_RANK_ELIDED
    ):
        packet = recode_pr106_hdm8_hlm2_packet_to_hdm9(packet)
    if (
        candidate.sidecar_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_INNER_HEADERLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        packet = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3(packet)
    if (
        candidate.sidecar_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    ):
        packet = recode_pr106_hdm8_or_hdm9_hlm2_packet_to_hdm9_hlm3_magicless(packet)
    if (
        candidate.sidecar_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        try:
            _hdm9_hlm3_magicless_payload(packet.pr106_bytes)
        except ValueError:
            packet = replace(
                packet,
                pr106_bytes=_recode_hdm8_or_hdm9_hlm2_pr106_bytes_to_hdm9_hlm3(
                    packet.pr106_bytes
                ),
            )
    emitted = emit_pr106_sidecar_packet(packet)
    reparsed = parse_pr106_sidecar_packet(emitted)
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(
        reparsed,
        schema=schema,
    )
    if not (
        np.array_equal(reparsed_dims, source_dims)
        and np.array_equal(reparsed_deltas, source_deltas)
    ):
        raise ValueError(f"candidate {candidate.name!r} failed parse/reemit semantics")
    if emit_pr106_sidecar_packet(reparsed) != emitted:
        raise ValueError(f"candidate {candidate.name!r} failed parse/reemit identity")
    return packet


def emit_pr106_sidecar_recode_candidate_archive(
    source_member: StoredZipMember,
    source_packet: PR106SidecarPacket,
    candidate: PR106SidecarRecodeCandidate,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[StoredZipMember, bytes]:
    """Emit a single-member ZIP archive for an existing-runtime recode candidate."""

    candidate_packet = build_pr106_sidecar_recode_candidate_packet(
        source_packet,
        candidate,
        schema=schema,
    )
    candidate_member = replace(
        source_member,
        payload=emit_pr106_sidecar_packet(candidate_packet),
    )
    return candidate_member, emit_single_stored_member_archive(candidate_member)


def build_pr106_format0c_semantic_sidecar_packet(
    source_packet: PR106SidecarPacket,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[PR106SidecarPacket, dict[str, object]]:
    """Build a format-0x0C packet from score-table-selected semantic deltas."""

    if (
        source_packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        raise ValueError(
            "format0C semantic materialization requires source packet "
            "format_id=0x0C; got "
            f"0x{source_packet.format_id:02X}"
        )
    base_dims, base_deltas = decode_pr106_sidecar_packet_dim_delta(
        source_packet,
        schema=schema,
    )
    composed_dims, composed_deltas, diagnostics = (
        compose_pr106_sidecar_dim_delta_corrections(
            base_dims,
            base_deltas,
            selected_dim_arr,
            selected_delta_q_arr,
            schema=schema,
        )
    )
    if bool(np.any(composed_dims == schema.no_op_sentinel)) or bool(
        np.any(composed_deltas == 0)
    ):
        raise ValueError(
            "format0C semantic materialization cannot represent composed no-op "
            "pairs because the committed fixed metadata has noop_count=0"
        )
    allowed_deltas = {int(value) for value in schema.deltas}
    observed_deltas = {int(value) for value in composed_deltas.astype(np.int16).tolist()}
    unsupported_deltas = sorted(observed_deltas - allowed_deltas)
    if unsupported_deltas:
        raise ValueError(
            "format0C semantic materialization cannot represent composed delta "
            f"values outside {sorted(allowed_deltas)}: {unsupported_deltas}"
        )
    sidecar_payload = encode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar_payload(
        composed_dims,
        composed_deltas,
        schema=schema,
    )
    packet = PR106SidecarPacket(
        format_id=source_packet.format_id,
        pr106_bytes=source_packet.pr106_bytes,
        sidecar_payload=sidecar_payload,
        framing_meta=None,
    )
    reparsed = parse_pr106_sidecar_packet(emit_pr106_sidecar_packet(packet))
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(
        reparsed,
        schema=schema,
    )
    if not (
        np.array_equal(reparsed_dims, composed_dims)
        and np.array_equal(reparsed_deltas, composed_deltas)
    ):
        raise ValueError("format0C semantic materialization parse/reemit mismatch")
    diagnostics = {
        **diagnostics,
        "semantic_materializer": "format0c_packet_ir_native",
        "source_format_id": f"0x{source_packet.format_id:02X}",
        "output_format_id": f"0x{packet.format_id:02X}",
        "output_sidecar_payload_bytes": len(sidecar_payload),
        "output_sidecar_payload_sha256": sha256_hex(sidecar_payload),
        "output_packet_payload_sha256": sha256_hex(emit_pr106_sidecar_packet(packet)),
    }
    return packet, diagnostics


def build_pr106_format0d_semantic_sidecar_packet(
    source_packet: PR106SidecarPacket,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[PR106SidecarPacket, dict[str, object]]:
    """Build format-0x0D: format0C base plus an additive PR101 extra stream."""

    if (
        source_packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    ):
        raise ValueError(
            "format0D semantic materialization requires source packet "
            "format_id=0x0C; got "
            f"0x{source_packet.format_id:02X}"
        )
    base_dims, base_deltas = decode_pr106_sidecar_packet_dim_delta(
        source_packet,
        schema=schema,
    )
    extra_dims, extra_deltas = canonicalize_brotli_dim_delta_sidecar_arrays(
        selected_dim_arr,
        selected_delta_q_arr,
        n_dims=schema.n_dims,
        no_op_dim=schema.no_op_sentinel,
    )
    if base_dims.shape != extra_dims.shape:
        raise ValueError(
            "base and extra correction shapes differ: "
            f"{base_dims.shape} vs {extra_dims.shape}"
        )
    extra_payload, extra_meta = encode_pr101_ranked_sidecar_payload(
        extra_dims,
        extra_deltas,
        schema=schema,
    )
    base_nonzero = (base_dims != schema.no_op_sentinel) & (base_deltas != 0)
    extra_nonzero = (extra_dims != schema.no_op_sentinel) & (extra_deltas != 0)
    same_dim = extra_nonzero & base_nonzero & (extra_dims == base_dims)
    second_dim = extra_nonzero & base_nonzero & (extra_dims != base_dims)
    into_base_noop = extra_nonzero & ~base_nonzero
    combined = base_deltas.astype(np.int16, copy=False) + extra_deltas.astype(
        np.int16,
        copy=False,
    )
    allowed_deltas = {int(value) for value in schema.deltas}
    same_dim_out_of_vocab = same_dim & ~np.isin(combined, list(allowed_deltas))
    packet = PR106SidecarPacket(
        format_id=PR106_SIDECAR_FORMAT_FORMAT0C_PLUS_PR101_EXTRA,
        pr106_bytes=source_packet.pr106_bytes,
        sidecar_payload=source_packet.sidecar_payload,
        framing_meta=None,
        extra_sidecar_payload=extra_payload,
        extra_framing_meta=extra_meta,
    )
    reparsed = parse_pr106_sidecar_packet(emit_pr106_sidecar_packet(packet))
    passes = decode_pr106_sidecar_packet_correction_passes(reparsed, schema=schema)
    if len(passes) != 2:
        raise ValueError("format0D semantic materialization did not roundtrip two passes")
    (roundtrip_base_dims, roundtrip_base_deltas), (
        roundtrip_extra_dims,
        roundtrip_extra_deltas,
    ) = passes
    if not (
        np.array_equal(roundtrip_base_dims, base_dims)
        and np.array_equal(roundtrip_base_deltas, base_deltas)
        and np.array_equal(roundtrip_extra_dims, extra_dims)
        and np.array_equal(roundtrip_extra_deltas, extra_deltas)
    ):
        raise ValueError("format0D semantic materialization parse/reemit mismatch")
    diagnostics: dict[str, object] = {
        "schema": "pr106_format0d_semantic_materialization_v1",
        "semantic_materializer": "format0d_packet_ir_native",
        "source_format_id": f"0x{source_packet.format_id:02X}",
        "output_format_id": f"0x{packet.format_id:02X}",
        "runtime_apply_order": [
            "base_format0c_corrections",
            "extra_pr101_ranked_no_op_corrections",
        ],
        "n_pairs": int(base_dims.size),
        "base_nonzero_pair_count": int(base_nonzero.sum()),
        "extra_nonzero_pair_count": int(extra_nonzero.sum()),
        "extra_same_dim_pair_count": int(same_dim.sum()),
        "extra_second_dim_pair_count": int(second_dim.sum()),
        "extra_into_base_noop_pair_count": int(into_base_noop.sum()),
        "extra_same_dim_out_of_format0c_vocab_pair_count": int(
            same_dim_out_of_vocab.sum()
        ),
        "base_format0c_sidecar_payload_bytes": len(source_packet.sidecar_payload),
        "base_format0c_sidecar_payload_sha256": sha256_hex(
            source_packet.sidecar_payload
        ),
        "extra_sidecar_payload_bytes": len(extra_payload),
        "extra_sidecar_payload_sha256": sha256_hex(extra_payload),
        "extra_framing_meta_bytes": len(extra_meta),
        "extra_framing_meta_sha256": sha256_hex(extra_meta),
        "output_packet_payload_sha256": sha256_hex(emit_pr106_sidecar_packet(packet)),
        "base_dim_sha256": sha256_hex(base_dims.astype(np.uint8).tobytes()),
        "base_delta_q_sha256": sha256_hex(base_deltas.astype(np.int8).tobytes()),
        "extra_dim_sha256": sha256_hex(extra_dims.astype(np.uint8).tobytes()),
        "extra_delta_q_sha256": sha256_hex(extra_deltas.astype(np.int8).tobytes()),
    }
    return packet, diagnostics


def build_pr106_format07_semantic_sidecar_packet(
    source_packet: PR106SidecarPacket,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[PR106SidecarPacket, dict[str, object]]:
    """Build a format-0x07 packet from score-table-selected semantic deltas."""

    if (
        source_packet.format_id
        != PR106_SIDECAR_FORMAT_PR101_HEADERLESS_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
    ):
        raise ValueError(
            "format07 semantic materialization requires source packet "
            "format_id=0x07; got "
            f"0x{source_packet.format_id:02X}"
        )
    base_dims, base_deltas = decode_pr106_sidecar_packet_dim_delta(
        source_packet,
        schema=schema,
    )
    composed_dims, composed_deltas, diagnostics = (
        compose_pr106_sidecar_dim_delta_corrections(
            base_dims,
            base_deltas,
            selected_dim_arr,
            selected_delta_q_arr,
            schema=schema,
        )
    )
    allowed_deltas = {0, *[int(value) for value in schema.deltas]}
    observed_deltas = {int(value) for value in composed_deltas.astype(np.int16).tolist()}
    unsupported_deltas = sorted(observed_deltas - allowed_deltas)
    if unsupported_deltas:
        raise ValueError(
            "format07 semantic materialization cannot represent composed delta "
            f"values outside {sorted(allowed_deltas)}: {unsupported_deltas}"
        )
    sidecar_payload = encode_pr101_fixed_meta_rank_elided_sidecar_payload(
        composed_dims,
        composed_deltas,
        schema=schema,
    )
    packet = PR106SidecarPacket(
        format_id=source_packet.format_id,
        pr106_bytes=source_packet.pr106_bytes,
        sidecar_payload=sidecar_payload,
        framing_meta=None,
    )
    reparsed = parse_pr106_sidecar_packet(emit_pr106_sidecar_packet(packet))
    reparsed_dims, reparsed_deltas = decode_pr106_sidecar_packet_dim_delta(
        reparsed,
        schema=schema,
    )
    if not (
        np.array_equal(reparsed_dims, composed_dims)
        and np.array_equal(reparsed_deltas, composed_deltas)
    ):
        raise ValueError("format07 semantic materialization parse/reemit mismatch")
    diagnostics = {
        **diagnostics,
        "semantic_materializer": "format07_packet_ir_native",
        "source_format_id": f"0x{source_packet.format_id:02X}",
        "output_format_id": f"0x{packet.format_id:02X}",
        "output_sidecar_payload_bytes": len(sidecar_payload),
        "output_sidecar_payload_sha256": sha256_hex(sidecar_payload),
        "output_packet_payload_sha256": sha256_hex(emit_pr106_sidecar_packet(packet)),
    }
    return packet, diagnostics


def emit_pr106_format0c_semantic_candidate_archive(
    source_member: StoredZipMember,
    source_packet: PR106SidecarPacket,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[StoredZipMember, bytes, dict[str, object]]:
    """Emit a single-member ``x`` ZIP for a format-0x0C semantic candidate."""

    if source_member.name != "x":
        raise ValueError(
            "format0C semantic materialization requires source archive member "
            f"'x'; got {source_member.name!r}"
        )
    candidate_packet, diagnostics = build_pr106_format0c_semantic_sidecar_packet(
        source_packet,
        selected_dim_arr,
        selected_delta_q_arr,
        schema=schema,
    )
    candidate_member = replace(
        source_member,
        name="x",
        payload=emit_pr106_sidecar_packet(candidate_packet),
    )
    candidate_archive = emit_single_stored_member_archive(candidate_member)
    return candidate_member, candidate_archive, diagnostics


def emit_pr106_format0d_semantic_candidate_archive(
    source_member: StoredZipMember,
    source_packet: PR106SidecarPacket,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[StoredZipMember, bytes, dict[str, object]]:
    """Emit a single-member ZIP for a format-0x0D semantic candidate."""

    if source_member.name != "x":
        raise ValueError(
            "format0D semantic materialization requires source archive member "
            f"'x'; got {source_member.name!r}"
        )
    candidate_packet, diagnostics = build_pr106_format0d_semantic_sidecar_packet(
        source_packet,
        selected_dim_arr,
        selected_delta_q_arr,
        schema=schema,
    )
    candidate_member = replace(
        source_member,
        name="x",
        payload=emit_pr106_sidecar_packet(candidate_packet),
    )
    candidate_archive = emit_single_stored_member_archive(candidate_member)
    return candidate_member, candidate_archive, diagnostics


def emit_pr106_format07_semantic_candidate_archive(
    source_member: StoredZipMember,
    source_packet: PR106SidecarPacket,
    selected_dim_arr: np.ndarray,
    selected_delta_q_arr: np.ndarray,
    *,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[StoredZipMember, bytes, dict[str, object]]:
    """Emit a single-member ZIP for a format-0x07 semantic candidate."""

    candidate_packet, diagnostics = build_pr106_format07_semantic_sidecar_packet(
        source_packet,
        selected_dim_arr,
        selected_delta_q_arr,
        schema=schema,
    )
    candidate_member = replace(
        source_member,
        payload=emit_pr106_sidecar_packet(candidate_packet),
    )
    candidate_archive = emit_single_stored_member_archive(candidate_member)
    return candidate_member, candidate_archive, diagnostics


def pr106_sidecar_recode_candidate_manifest(
    source_packet: PR106SidecarPacket,
    candidate: PR106SidecarRecodeCandidate,
    *,
    source_archive_sha256: str | None = None,
    candidate_archive_sha256: str | None = None,
    candidate_archive_bytes: int | None = None,
    candidate_member_name: str | None = None,
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> dict[str, object]:
    """Return a no-score manifest for a PR106 sidecar compression candidate."""

    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(
        source_packet,
        schema=schema,
    )
    applicable = bool(candidate.encoded_bytes)
    source_charged_bytes = _sidecar_charged_bytes(source_packet)
    blockers: list[str] = []
    candidate_packet: PR106SidecarPacket | None = None
    candidate_proof: dict[str, object] | None = None
    candidate_payload: bytes | None = None

    if not applicable:
        blockers.append("candidate_not_applicable_to_source_arrays")
    if candidate.sidecar_format_id not in PR106_SUPPORTED_SIDECAR_FORMATS:
        blockers.append("candidate_runtime_decoder_missing")
    if applicable and candidate.sidecar_format_id in PR106_SUPPORTED_SIDECAR_FORMATS:
        candidate_packet = build_pr106_sidecar_recode_candidate_packet(
            source_packet,
            candidate,
            schema=schema,
        )
        candidate_payload = emit_pr106_sidecar_packet(candidate_packet)
        candidate_proof = pr106_sidecar_consumed_byte_proof(candidate_packet)
        if candidate_proof.get("all_payload_bytes_accounted") is not True:
            blockers.append("candidate_packet_ir_consumed_byte_accounting_failed")
        if candidate_proof.get("runtime_consumption_claim") is not False:
            blockers.append("candidate_packet_ir_overclaimed_runtime_consumption")
    if not candidate.runtime_decoder_implemented:
        blockers.append("candidate_runtime_decoder_not_implemented")

    source_payload = emit_pr106_sidecar_packet(source_packet)
    source_proof = pr106_sidecar_consumed_byte_proof(source_packet)
    lossless_semantic = applicable and (
        np.array_equal(candidate.decoded_dims, source_dims)
        and np.array_equal(candidate.decoded_delta_q, source_deltas)
    )
    delta_bytes = None
    if applicable:
        delta_bytes = candidate.charged_bytes - source_charged_bytes
    source_corrections = int(((source_dims != PR106_NO_OP_DIM) & (source_deltas != 0)).sum())
    return {
        "schema": "pr106_sidecar_recode_candidate_manifest_v1",
        "proof_scope": (
            "lossless_sidecar_compression_candidate_packet_ir_consumed_byte_"
            "accounting_not_runtime_inflate_not_score"
        ),
        "candidate_name": candidate.name,
        "applicable": applicable,
        "runtime_decoder_implemented": candidate.runtime_decoder_implemented,
        "lossless_semantic_equivalence_proven": lossless_semantic,
        "source_archive_sha256": source_archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_member_name": candidate_member_name,
        "source_packet": {
            "format_id": f"0x{source_packet.format_id:02X}",
            "payload_bytes": len(source_payload),
            "payload_sha256": sha256_hex(source_payload),
            "sidecar_charged_bytes": source_charged_bytes,
            "pr106_payload_sha256": sha256_hex(source_packet.pr106_bytes),
            "sidecar_payload_sha256": sha256_hex(source_packet.sidecar_payload),
            "framing_meta_sha256": None
            if source_packet.framing_meta is None
            else sha256_hex(source_packet.framing_meta),
        },
        "candidate_packet": None
        if candidate_packet is None or candidate_payload is None
        else {
            "format_id": f"0x{candidate_packet.format_id:02X}",
            "payload_bytes": len(candidate_payload),
            "payload_sha256": sha256_hex(candidate_payload),
            "sidecar_charged_bytes": candidate.charged_bytes,
            "delta_charged_sidecar_bytes_vs_source": delta_bytes,
            "pr106_payload_sha256": sha256_hex(candidate_packet.pr106_bytes),
            "sidecar_payload_sha256": sha256_hex(candidate_packet.sidecar_payload),
            "framing_meta_sha256": None
            if candidate_packet.framing_meta is None
            else sha256_hex(candidate_packet.framing_meta),
        },
        "semantic_arrays": {
            "n_pairs": int(source_dims.size),
            "n_corrections": source_corrections,
            "dim_sha256": sha256_hex(source_dims.astype(np.uint8).tobytes()),
            "delta_q_sha256": sha256_hex(source_deltas.astype(np.int8).tobytes()),
        },
        "source_packet_ir_consumed_byte_proof": source_proof,
        "candidate_packet_ir_consumed_byte_proof": candidate_proof,
        "candidate_packet_ir_identity_passed": (
            candidate_packet is not None
            and candidate_payload is not None
            and emit_pr106_sidecar_packet(parse_pr106_sidecar_packet(candidate_payload))
            == candidate_payload
        ),
        "runtime_consumption_claim": False,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "blockers": blockers,
        "exact_eval_blockers": [
            "runtime_decode_apply_proof_required_for_new_candidate_archive",
            "full_frame_same_runtime_parity_or_same_runtime_auth_eval_missing",
            "exact_cuda_auth_eval_missing",
            "contest_auth_eval_adjudication_missing",
        ],
    }


def pr106_sidecar_mutation_manifest(
    source_packet: PR106SidecarPacket,
    mutated_packet: PR106SidecarPacket,
    mutation: PR106SidecarMutation,
    *,
    source_archive_sha256: str | None = None,
    mutated_archive_sha256: str | None = None,
) -> dict[str, object]:
    """Return a no-score manifest for a runtime-consumption mutation smoke."""

    source_payload = emit_pr106_sidecar_packet(source_packet)
    mutated_payload = emit_pr106_sidecar_packet(mutated_packet)
    source_proof = pr106_sidecar_consumed_byte_proof(source_packet)
    mutated_proof = pr106_sidecar_consumed_byte_proof(mutated_packet)
    return {
        "schema": "pr106_sidecar_runtime_decode_mutation_manifest_v1",
        "proof_scope": "packet_ir_valid_mutation_for_runtime_decode_smoke_not_score",
        "format_id": f"0x{source_packet.format_id:02X}",
        "sidecar_kind": source_packet.sidecar_kind,
        "mutation": asdict(mutation),
        "source_archive_sha256": source_archive_sha256,
        "mutated_archive_sha256": mutated_archive_sha256,
        "source_payload_bytes": len(source_payload),
        "source_payload_sha256": sha256_hex(source_payload),
        "mutated_payload_bytes": len(mutated_payload),
        "mutated_payload_sha256": sha256_hex(mutated_payload),
        "payload_sha256_changed": sha256_hex(source_payload) != sha256_hex(mutated_payload),
        "inner_pr106_payload_sha256_unchanged": sha256_hex(source_packet.pr106_bytes)
        == sha256_hex(mutated_packet.pr106_bytes),
        "sidecar_payload_sha256_changed": sha256_hex(source_packet.sidecar_payload)
        != sha256_hex(mutated_packet.sidecar_payload),
        "extra_sidecar_payload_sha256_changed": sha256_hex(
            source_packet.extra_sidecar_payload
        )
        != sha256_hex(mutated_packet.extra_sidecar_payload),
        "extra_framing_meta_sha256_changed": (
            None
            if source_packet.extra_framing_meta is None
            and mutated_packet.extra_framing_meta is None
            else sha256_hex(source_packet.extra_framing_meta or b"")
            != sha256_hex(mutated_packet.extra_framing_meta or b"")
        ),
        "source_packet_ir_consumed_byte_proof": source_proof,
        "mutated_packet_ir_consumed_byte_proof": mutated_proof,
        "parser_consumed_byte_accounting_passed": (
            source_proof.get("all_payload_bytes_accounted") is True
            and mutated_proof.get("all_payload_bytes_accounted") is True
        ),
        "byte_exact_identity": {
            "inner_pr106_payload_sha256_unchanged": sha256_hex(source_packet.pr106_bytes)
            == sha256_hex(mutated_packet.pr106_bytes),
            "source_payload_sha256": sha256_hex(source_payload),
            "mutated_payload_sha256": sha256_hex(mutated_payload),
            "payload_sha256_changed": sha256_hex(source_payload)
            != sha256_hex(mutated_payload),
            "source_packet_accounted_payload_bytes": source_proof.get(
                "accounted_payload_bytes"
            ),
            "mutated_packet_accounted_payload_bytes": mutated_proof.get(
                "accounted_payload_bytes"
            ),
        },
        "runtime_sidecar_decode_consumption_claim": False,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "proof_not_score": True,
        "evidence_axis": "packet-ir-mutation-local-no-score",
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next_proof": (
            "run actual submission runtime sidecar decoder smoke; full-frame "
            "inflate parity or exact same-runtime eval still required before "
            "promotion language"
        ),
    }
