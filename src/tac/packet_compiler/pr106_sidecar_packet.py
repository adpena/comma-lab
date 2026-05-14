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
PR106_SUPPORTED_SIDECAR_FORMATS = (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
)
PR106_LATENT_N_PAIRS = 600
PR106_LATENT_N_DIMS = 28
PR106_DEFAULT_MEMBER_NAME = "0.bin"
PR106_ALLOWED_SINGLE_MEMBER_NAMES = (PR106_DEFAULT_MEMBER_NAME, "x")
PR106_NO_OP_DIM = 255
PR106_PR101_RANKED_SCHEMA = RankedSidecarSchema(
    n_pairs=600,
    n_dims=28,
    deltas=(-2, -1, 1, 2),
    huff_min_len=2,
    huff_max_len=8,
    no_op_sentinel=PR106_NO_OP_DIM,
)


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

    @property
    def sidecar_kind(self) -> str:
        if self.format_id == PR106_SIDECAR_FORMAT_BROTLI:
            return "brotli_dim_delta"
        if self.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
            return "pr101_ranked_no_op"
        if self.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
            return "pr101_ranked_no_op_rank_elided"
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
    return len(packet.sidecar_payload) + (
        0 if packet.framing_meta is None else len(packet.framing_meta)
    )


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

    payload = encode_ranked_no_op_sidecar(
        dims=dims,
        delta_indices=delta_indices,
        schema=schema,
    )
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
    raise ValueError(f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}")


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
        raise ValueError(
            f"sidecar magic mismatch: got 0x{payload[0]:02X}, "
            f"expected 0x{PR106_SIDECAR_MAGIC:02X}"
        )
    format_id = payload[1]
    if format_id not in supported_formats:
        expected = ", ".join(f"0x{value:02X}" for value in supported_formats)
        raise ValueError(
            f"unsupported PR106 sidecar format_id=0x{format_id:02X}; expected {expected}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", payload, pos)
    pos += 4
    end_pr106 = pos + pr106_len
    if end_pr106 > len(payload):
        raise ValueError(
            f"PR106 inner payload truncated: pr106_len={pr106_len} total={len(payload)}"
        )
    pr106_bytes = payload[pos:end_pr106]
    pos = end_pr106

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

    raise ValueError(f"unsupported PR106 sidecar format_id=0x{format_id:02X}")


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
    schema: RankedSidecarSchema = PR106_PR101_RANKED_SCHEMA,
) -> tuple[PR106SidecarPacket, PR106SidecarMutation]:
    """Return a valid packet with one runtime-visible sidecar correction changed.

    This helper is the deterministic mutation primitive for no-op/runtime
    consumption tests. It preserves the inner PR106 payload and only mutates
    ``sidecar_payload`` bytes through the sidecar's own grammar, so downstream
    tools can prove that the submission runtime's sidecar decoder sees the
    change before any exact-eval score is claimed.
    """

    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
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

    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x02 requires framing_meta")
        dims, delta_indices = _decode_pr101_ranked_sidecar_payload(
            packet.sidecar_payload,
            packet.framing_meta,
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
        mutated_payload = encode_ranked_no_op_sidecar(
            dims=dims,
            delta_indices=mutated_delta_indices,
            schema=schema,
        )
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
    if (
        packet.format_id != PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED
        and len(packet.sidecar_payload) > 0xFFFF
    ):
        raise ValueError("sidecar payload too large for u16 length field")
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
    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI:
        return prefix + struct.pack("<H", len(packet.sidecar_payload)) + packet.sidecar_payload
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        assert packet.framing_meta is not None
        return prefix + packet.sidecar_payload + packet.framing_meta
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

    add("magic", bytes([PR106_SIDECAR_MAGIC]), score_affecting=False)
    add("format_id", bytes([packet.format_id]), score_affecting=False)
    add(
        "pr106_len_le_u32",
        struct.pack("<I", len(packet.pr106_bytes)),
        score_affecting=False,
    )
    add("pr106_payload", packet.pr106_bytes, score_affecting=True)
    if packet.format_id != PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        add(
            "sidecar_len_le_u16",
            struct.pack("<H", len(packet.sidecar_payload)),
            score_affecting=False,
        )
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
