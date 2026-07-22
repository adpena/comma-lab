# SPDX-License-Identifier: MIT
"""Lossless variable-length six-stream grammar for the Task #603 DDM member.

The v2 chart archive stored canonical semantic records verbatim, so changing
record values could not change ``len(A(z))``.  This module keeps those exact
semantic records as the receiver contract while encoding each ZIP member with
a deterministic per-stream tournament over existing repository coders.

No entropy coder is implemented here.  The adapters reuse Brotli-Q11/LZMA
from :mod:`tac.optimization.arith_selfcomp_rate_coders`, AQc1 sparse
arithmetic coding, PR101 ranked canonical Huffman lengths and colex ranks, and
the existing Rice/Golomb residual stage.  Every selected stream decodes back
to the byte-identical v2 semantic payload before the established NumPy chart
receiver is invoked.
"""

from __future__ import annotations

import io
import math
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.arithmetic_qint_codec import (
    decode_qints_arithmetic_compact,
    encode_qints_arithmetic_compact,
)
from tac.boundary_math.xi_spline_residual_coder import (
    RESIDUAL_SCHEMES,
    decode_residual_matrix,
    encode_residual_matrix,
    measure_residual_schemes,
    residual_scheme_id,
    residual_scheme_name,
)
from tac.codec.pr101_polymorphic import (
    SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN,
    _bit_pack,
    _build_canonical_huffman_codebook,
    _build_optimal_huffman_lengths,
    _decode_canonical_huffman_n,
    decode_combination_colex,
    decode_huff_length_rank,
    encode_combination_colex,
    encode_huff_length_rank,
)
from tac.optimization.arith_selfcomp_rate_coders import (
    RateCoderError,
    decode_brotli_q11,
    decode_lzma,
    encode_brotli_q11,
    encode_lzma,
)
from tac.optimization.direct_description_measurement_ladder import (
    _ANCHOR_RECORD,
    _GRADIENT_RECORD,
    _POSE_RECORD,
    _RESIDUAL_RECORD,
    CHARTS_PER_PLANE,
    MEMBER_BY_STREAM,
    STREAM_BY_MEMBER,
    STREAM_MAGIC,
    STREAM_ORDER,
    CountedChartStreamV1,
    DirectDescriptionChartZV1,
    _deterministic_zip,
    _expected_record_count,
    _zip_unique_home_ledger,
    compile_chart_archive,
    receive_chart_archive,
)
from tac.optimization.direct_description_minimizer import (
    DirectDescriptionError,
    _read_regular_file_once,
    _sha256,
)
from tac.optimization.direct_description_polytope_membership import stream_decode_digest

GRAMMAR_SCHEMA: Final = "direct_description_entropy_chart_archive.v1"
RECEIVER_SCHEMA: Final = "direct_description_entropy_chart_receiver.v1"

TRANSFORM_TEMPORAL_DELTA: Final = 1
TRANSFORM_SPARSE_RECORDS: Final = 2
TRANSFORM_DENSE_TEMPORAL_BITMAP: Final = 3
TRANSFORM_DENSE_TEMPORAL_COLEX: Final = 4

TRANSFORM_NAME: Final = {
    TRANSFORM_TEMPORAL_DELTA: "pair_temporal_delta",
    TRANSFORM_SPARSE_RECORDS: "canonical_sparse_chart_records",
    TRANSFORM_DENSE_TEMPORAL_BITMAP: "chart_aligned_temporal_delta_bitmap",
    TRANSFORM_DENSE_TEMPORAL_COLEX: "chart_aligned_temporal_delta_colex",
}

CODER_BROTLI_Q11: Final = 1
CODER_LZMA_XZ9: Final = 2
CODER_AQC1: Final = 3
CODER_HUFFMAN_RANK16: Final = 4
CODER_SPLIT_RICE: Final = 5

CODER_NAME: Final = {
    CODER_BROTLI_Q11: "brotli_q11",
    CODER_LZMA_XZ9: "lzma_xz_preset9_extreme",
    CODER_AQC1: "aqc1_sparse_arithmetic_uint8",
    CODER_HUFFMAN_RANK16: "pr101_ranked_canonical_huffman_16",
    CODER_SPLIT_RICE: "split_metadata_plus_rice_golomb",
}

DTYPE_INT16: Final = 1
DTYPE_INT32: Final = 2
_DTYPE_FROM_ID: Final = {DTYPE_INT16: np.dtype("<i2"), DTYPE_INT32: np.dtype("<i4")}
_DTYPE_ID: Final = {np.dtype("int16"): DTYPE_INT16, np.dtype("int32"): DTYPE_INT32}

_TRANSFORM_HEADER = struct.Struct("<IIIB3x")
_ENTROPY_FRAME = struct.Struct("<8sHHBBHIIII32s")
_SPLIT_HEADER = struct.Struct("<BIBI")
_COLEX_BYTES = (math.comb(CHARTS_PER_PLANE, CHARTS_PER_PLANE // 3).bit_length() + 7) // 8
_ENTROPY_MAGIC: Final = {
    name: bytes((STREAM_MAGIC[name][0] ^ 0x20,)) + STREAM_MAGIC[name][1:7] + b"E" for name in STREAM_ORDER
}


@dataclass(frozen=True, slots=True)
class TransformBodyV1:
    transform_id: int
    metadata: bytes
    values: np.ndarray

    def canonical_bytes(self) -> bytes:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.dtype not in _DTYPE_ID:
            raise DirectDescriptionError("entropy transform values must be a 2-D int16/int32 matrix")
        body = values.astype(_DTYPE_FROM_ID[_DTYPE_ID[values.dtype]], copy=False).tobytes(order="C")
        return (
            _TRANSFORM_HEADER.pack(
                len(self.metadata),
                int(values.shape[0]),
                int(values.shape[1]),
                _DTYPE_ID[values.dtype],
            )
            + self.metadata
            + body
        )


@dataclass(frozen=True, slots=True)
class EntropyStreamBuildV1:
    stream: str
    transform_id: int
    coder_id: int
    frame: bytes
    semantic_payload: bytes
    canonical_transform: bytes
    coded_payload: bytes
    candidate_rows: tuple[Mapping[str, Any], ...]
    split_detail: Mapping[str, Any] | None

    def ledger_row(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "member": MEMBER_BY_STREAM[self.stream],
            "transform": TRANSFORM_NAME[self.transform_id],
            "coder": CODER_NAME[self.coder_id],
            "semantic_payload_bytes": len(self.semantic_payload),
            "semantic_payload_sha256": _sha256(self.semantic_payload),
            "canonical_transform_bytes": len(self.canonical_transform),
            "coded_payload_bytes": len(self.coded_payload),
            "member_frame_bytes": len(self.frame),
            "candidate_rows": [dict(row) for row in self.candidate_rows],
            "split_detail": dict(self.split_detail) if self.split_detail is not None else None,
            "exact_semantic_roundtrip": True,
        }


@dataclass(frozen=True, slots=True)
class EntropyChartArchiveBuildResultV1:
    archive: bytes
    framed_members: Mapping[str, bytes]
    z: DirectDescriptionChartZV1
    streams: Mapping[str, EntropyStreamBuildV1]

    def stream_byte_rows(self) -> list[dict[str, Any]]:
        homes = {row["owner"]: row for row in _zip_unique_home_ledger(self.archive)}
        rows: list[dict[str, Any]] = []
        for stream_name in STREAM_ORDER:
            row = self.streams[stream_name].ledger_row()
            home = homes[stream_name]
            row["unique_final_zip_home_bytes"] = int(home["home_bytes"])
            row["zip_member_payload_bytes"] = int(home["member_payload_range"]["bytes"])
            rows.append(row)
        return rows

    def custody(self) -> dict[str, Any]:
        homes = _zip_unique_home_ledger(self.archive)
        return {
            "schema": GRAMMAR_SCHEMA,
            "compiler": "ddm_six_stream_entropy_tournament.v1",
            "archive_bytes": len(self.archive),
            "archive_sha256": _sha256(self.archive),
            "member_count": len(self.framed_members),
            "member_order": [MEMBER_BY_STREAM[name] for name in STREAM_ORDER],
            "stream_ledger": self.stream_byte_rows(),
            "unique_final_zip_homes": homes,
            "unique_home_coverage_bytes": sum(int(row["home_bytes"]) for row in homes),
            "all_archive_bytes_have_one_home": sum(int(row["home_bytes"]) for row in homes) == len(self.archive),
            "container_only_bytes": sum(int(row["home_bytes"]) for row in homes if row["owner"] == "container_framing"),
            "receiver_consumption_verified": False,
        }


@dataclass(frozen=True, slots=True)
class EntropyChartReceiverResultV1:
    archive: bytes
    z: DirectDescriptionChartZV1
    anchors: np.ndarray
    gradients: np.ndarray
    residuals: np.ndarray
    pose6_codes: np.ndarray
    custody: Mapping[str, Any]
    _semantic_receiver: Any

    def render_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        return self._semantic_receiver.render_pairs(pair_ids)


def _parse_transform_body(payload: bytes) -> tuple[bytes, np.ndarray]:
    if len(payload) < _TRANSFORM_HEADER.size:
        raise DirectDescriptionError("entropy transform body is truncated")
    metadata_bytes, rows, columns, dtype_id = _TRANSFORM_HEADER.unpack_from(payload)
    dtype = _DTYPE_FROM_ID.get(dtype_id)
    if dtype is None or rows < 1 or columns < 1:
        raise DirectDescriptionError("entropy transform header is invalid")
    metadata_start = _TRANSFORM_HEADER.size
    values_start = metadata_start + metadata_bytes
    expected = values_start + rows * columns * dtype.itemsize
    if expected != len(payload):
        raise DirectDescriptionError("entropy transform body has truncated or trailing bytes")
    metadata = payload[metadata_start:values_start]
    values = np.frombuffer(payload, dtype=dtype, offset=values_start).reshape(rows, columns).copy()
    if (
        _TRANSFORM_HEADER.pack(metadata_bytes, rows, columns, dtype_id)
        + metadata
        + values.astype(dtype, copy=False).tobytes(order="C")
        != payload
    ):
        raise DirectDescriptionError("entropy transform body is noncanonical")
    return metadata, values


def _delta_matrix(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64)
    delta = np.empty_like(source, dtype=np.int64)
    delta[0] = source[0]
    if source.shape[0] > 1:
        delta[1:] = source[1:] - source[:-1]
    info = np.iinfo(dtype)
    if delta.min(initial=0) < info.min or delta.max(initial=0) > info.max:
        raise DirectDescriptionError(f"temporal delta exceeds {dtype}")
    return delta.astype(dtype)


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(values, dtype=np.int64), axis=0, dtype=np.int64)


def _parse_anchor_values(payload: bytes, n_pairs: int) -> np.ndarray:
    values = np.empty((n_pairs, 2, 3), dtype=np.int16)
    seen = np.zeros((n_pairs, 2), dtype=np.bool_)
    for offset in range(0, len(payload), _ANCHOR_RECORD.size):
        pair_id, plane_id, *rgb = _ANCHOR_RECORD.unpack_from(payload, offset)
        if pair_id >= n_pairs or plane_id >= 2 or seen[pair_id, plane_id]:
            raise DirectDescriptionError("entropy anchor coverage is noncanonical")
        values[pair_id, plane_id] = rgb
        seen[pair_id, plane_id] = True
    if not seen.all():
        raise DirectDescriptionError("entropy anchor stream is incomplete")
    return values


def _parse_gradient_values(payload: bytes, n_pairs: int) -> np.ndarray:
    values = np.empty((n_pairs, 2, 6), dtype=np.int16)
    seen = np.zeros((n_pairs, 2), dtype=np.bool_)
    for offset in range(0, len(payload), _GRADIENT_RECORD.size):
        pair_id, plane_id, *gradient = _GRADIENT_RECORD.unpack_from(payload, offset)
        if pair_id >= n_pairs or plane_id >= 2 or seen[pair_id, plane_id]:
            raise DirectDescriptionError("entropy gradient coverage is noncanonical")
        values[pair_id, plane_id] = gradient
        seen[pair_id, plane_id] = True
    if not seen.all():
        raise DirectDescriptionError("entropy gradient stream is incomplete")
    return values


def _parse_pose_values(payload: bytes, n_pairs: int) -> np.ndarray:
    values = np.empty((n_pairs, 6), dtype=np.int16)
    seen = np.zeros(n_pairs, dtype=np.bool_)
    for offset in range(0, len(payload), _POSE_RECORD.size):
        pair_id, *pose = _POSE_RECORD.unpack_from(payload, offset)
        if pair_id >= n_pairs or seen[pair_id]:
            raise DirectDescriptionError("entropy Pose6 coverage is noncanonical")
        values[pair_id] = pose
        seen[pair_id] = True
    if not seen.all():
        raise DirectDescriptionError("entropy Pose6 stream is incomplete")
    return values


def _parse_residual_values(payload: bytes, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    owned = CHARTS_PER_PLANE // 3
    occupancy = np.zeros((n_pairs, 2, CHARTS_PER_PLANE), dtype=np.bool_)
    dense = np.zeros((n_pairs, 2, CHARTS_PER_PLANE, 3), dtype=np.int16)
    counts = np.zeros((n_pairs, 2), dtype=np.uint16)
    observed_order: list[tuple[int, int, int]] = []
    for offset in range(0, len(payload), _RESIDUAL_RECORD.size):
        pair_id, plane_id, chart_id, *residual = _RESIDUAL_RECORD.unpack_from(payload, offset)
        if (
            pair_id >= n_pairs
            or plane_id >= 2
            or chart_id >= CHARTS_PER_PLANE
            or occupancy[pair_id, plane_id, chart_id]
        ):
            raise DirectDescriptionError("entropy residual coverage is noncanonical")
        occupancy[pair_id, plane_id, chart_id] = True
        dense[pair_id, plane_id, chart_id] = residual
        counts[pair_id, plane_id] += 1
        observed_order.append((pair_id, plane_id, chart_id))
    if not np.all(counts == owned):
        raise DirectDescriptionError("entropy residual stream must own one chart tertile")
    canonical_order = [
        (pair_id, plane_id, int(chart_id))
        for pair_id in range(n_pairs)
        for plane_id in range(2)
        for chart_id in np.flatnonzero(occupancy[pair_id, plane_id])
    ]
    if observed_order != canonical_order:
        raise DirectDescriptionError("entropy residual records are not in canonical chart order")
    return occupancy, dense


def _colex_metadata(occupancy: np.ndarray) -> bytes:
    out = bytearray()
    for pair_id in range(occupancy.shape[0]):
        for plane_id in range(2):
            positions = np.flatnonzero(occupancy[pair_id, plane_id]).astype(np.int64)
            rank = encode_combination_colex(positions, CHARTS_PER_PLANE)
            out.extend(rank.to_bytes(_COLEX_BYTES, "little"))
    return bytes(out)


def _decode_colex_metadata(metadata: bytes, n_pairs: int) -> np.ndarray:
    if len(metadata) != n_pairs * 2 * _COLEX_BYTES:
        raise DirectDescriptionError("entropy colex metadata length mismatch")
    occupancy = np.zeros((n_pairs, 2, CHARTS_PER_PLANE), dtype=np.bool_)
    cursor = 0
    for pair_id in range(n_pairs):
        for plane_id in range(2):
            rank = int.from_bytes(metadata[cursor : cursor + _COLEX_BYTES], "little")
            cursor += _COLEX_BYTES
            try:
                positions = decode_combination_colex(rank, CHARTS_PER_PLANE, CHARTS_PER_PLANE // 3)
            except ValueError as exc:
                raise DirectDescriptionError("entropy colex residual positions are invalid") from exc
            occupancy[pair_id, plane_id, positions] = True
    return occupancy


def _transform_candidates(stream_name: str, payload: bytes, n_pairs: int) -> tuple[TransformBodyV1, ...]:
    if stream_name == "global_chart_anchors":
        values = _parse_anchor_values(payload, n_pairs).reshape(n_pairs, 6)
        return (TransformBodyV1(TRANSFORM_TEMPORAL_DELTA, b"", _delta_matrix(values, np.dtype("int16"))),)
    if stream_name == "axial_chart_gradients":
        values = _parse_gradient_values(payload, n_pairs).reshape(n_pairs, 12)
        return (TransformBodyV1(TRANSFORM_TEMPORAL_DELTA, b"", _delta_matrix(values, np.dtype("int32"))),)
    if stream_name == "pose6_pair_codes":
        values = _parse_pose_values(payload, n_pairs)
        return (TransformBodyV1(TRANSFORM_TEMPORAL_DELTA, b"", _delta_matrix(values, np.dtype("int16"))),)
    if stream_name not in STREAM_ORDER[2:5]:
        raise DirectDescriptionError(f"unknown entropy stream {stream_name!r}")
    occupancy, dense = _parse_residual_values(payload, n_pairs)
    sparse_ids = np.concatenate(
        [
            np.flatnonzero(occupancy[pair_id, plane_id]).astype(np.uint8)
            for pair_id in range(n_pairs)
            for plane_id in range(2)
        ]
    ).tobytes()
    sparse_values = np.concatenate(
        [
            dense[pair_id, plane_id, np.flatnonzero(occupancy[pair_id, plane_id])].reshape(-1)
            for pair_id in range(n_pairs)
            for plane_id in range(2)
        ]
    ).reshape(n_pairs, -1)
    dense_matrix = dense.reshape(n_pairs, -1)
    dense_delta = _delta_matrix(dense_matrix, np.dtype("int32"))
    bitmap = np.packbits(occupancy.reshape(-1), bitorder="little").tobytes()
    candidates = (
        TransformBodyV1(TRANSFORM_SPARSE_RECORDS, sparse_ids, sparse_values.astype(np.int16)),
        TransformBodyV1(TRANSFORM_DENSE_TEMPORAL_BITMAP, bitmap, dense_delta),
        TransformBodyV1(TRANSFORM_DENSE_TEMPORAL_COLEX, _colex_metadata(occupancy), dense_delta),
    )
    for candidate in candidates:
        if _decode_transform(stream_name, n_pairs, candidate.transform_id, candidate.canonical_bytes()) != payload:
            raise DirectDescriptionError("entropy residual transform failed exact semantic roundtrip")
    return candidates


def _decode_transform(stream_name: str, n_pairs: int, transform_id: int, body: bytes) -> bytes:
    metadata, matrix = _parse_transform_body(body)
    if stream_name == "global_chart_anchors":
        if transform_id != TRANSFORM_TEMPORAL_DELTA or metadata or matrix.shape != (n_pairs, 6):
            raise DirectDescriptionError("entropy anchor transform identity mismatch")
        values = _prefix_sum(matrix)
        if values.min(initial=0) < 0 or values.max(initial=0) > 255:
            raise DirectDescriptionError("entropy anchor decode left uint8 domain")
        shaped = values.reshape(n_pairs, 2, 3)
        return b"".join(
            _ANCHOR_RECORD.pack(pair_id, plane_id, *(int(value) for value in shaped[pair_id, plane_id]))
            for pair_id in range(n_pairs)
            for plane_id in range(2)
        )
    if stream_name == "axial_chart_gradients":
        if transform_id != TRANSFORM_TEMPORAL_DELTA or metadata or matrix.shape != (n_pairs, 12):
            raise DirectDescriptionError("entropy gradient transform identity mismatch")
        values = _prefix_sum(matrix)
        info = np.iinfo(np.int16)
        if values.min(initial=0) < info.min or values.max(initial=0) > info.max:
            raise DirectDescriptionError("entropy gradient decode left int16 domain")
        shaped = values.reshape(n_pairs, 2, 6)
        return b"".join(
            _GRADIENT_RECORD.pack(pair_id, plane_id, *(int(value) for value in shaped[pair_id, plane_id]))
            for pair_id in range(n_pairs)
            for plane_id in range(2)
        )
    if stream_name == "pose6_pair_codes":
        if transform_id != TRANSFORM_TEMPORAL_DELTA or metadata or matrix.shape != (n_pairs, 6):
            raise DirectDescriptionError("entropy Pose6 transform identity mismatch")
        values = _prefix_sum(matrix)
        if values.min(initial=0) < 0 or values.max(initial=0) > 255:
            raise DirectDescriptionError("entropy Pose6 decode left uint8 domain")
        return b"".join(
            _POSE_RECORD.pack(pair_id, *(int(value) for value in values[pair_id])) for pair_id in range(n_pairs)
        )
    if stream_name not in STREAM_ORDER[2:5]:
        raise DirectDescriptionError(f"unknown entropy transform stream {stream_name!r}")
    owned = CHARTS_PER_PLANE // 3
    if transform_id == TRANSFORM_SPARSE_RECORDS:
        if len(metadata) != n_pairs * 2 * owned or matrix.shape != (n_pairs, 2 * owned * 3):
            raise DirectDescriptionError("entropy sparse residual transform geometry mismatch")
        chart_ids = np.frombuffer(metadata, dtype=np.uint8).reshape(n_pairs, 2, owned)
        residual = matrix.astype(np.int64).reshape(n_pairs, 2, owned, 3)
        out = bytearray()
        for pair_id in range(n_pairs):
            for plane_id in range(2):
                ids = chart_ids[pair_id, plane_id]
                if np.any(ids >= CHARTS_PER_PLANE) or (ids.size > 1 and not np.all(np.diff(ids.astype(np.int16)) > 0)):
                    raise DirectDescriptionError("entropy sparse residual chart ids are noncanonical")
                for local, chart_id in enumerate(ids):
                    out.extend(
                        _RESIDUAL_RECORD.pack(
                            pair_id,
                            plane_id,
                            int(chart_id),
                            *(int(value) for value in residual[pair_id, plane_id, local]),
                        )
                    )
        return bytes(out)
    if matrix.shape != (n_pairs, 2 * CHARTS_PER_PLANE * 3):
        raise DirectDescriptionError("entropy dense residual transform geometry mismatch")
    if transform_id == TRANSFORM_DENSE_TEMPORAL_BITMAP:
        expected = (n_pairs * 2 * CHARTS_PER_PLANE + 7) // 8
        if len(metadata) != expected:
            raise DirectDescriptionError("entropy residual bitmap length mismatch")
        bits = np.unpackbits(np.frombuffer(metadata, dtype=np.uint8), bitorder="little")
        if bits.size > n_pairs * 2 * CHARTS_PER_PLANE and np.any(bits[n_pairs * 2 * CHARTS_PER_PLANE :]):
            raise DirectDescriptionError("entropy residual bitmap has nonzero padding")
        occupancy = bits[: n_pairs * 2 * CHARTS_PER_PLANE].reshape(n_pairs, 2, CHARTS_PER_PLANE).astype(bool)
    elif transform_id == TRANSFORM_DENSE_TEMPORAL_COLEX:
        occupancy = _decode_colex_metadata(metadata, n_pairs)
    else:
        raise DirectDescriptionError("entropy residual transform id is invalid")
    if not np.all(occupancy.sum(axis=2) == owned):
        raise DirectDescriptionError("entropy dense residual occupancy is not one tertile")
    dense = _prefix_sum(matrix).reshape(n_pairs, 2, CHARTS_PER_PLANE, 3)
    info = np.iinfo(np.int16)
    if dense.min(initial=0) < info.min or dense.max(initial=0) > info.max:
        raise DirectDescriptionError("entropy dense residual decode left int16 domain")
    out = bytearray()
    for pair_id in range(n_pairs):
        for plane_id in range(2):
            for chart_id in np.flatnonzero(occupancy[pair_id, plane_id]):
                out.extend(
                    _RESIDUAL_RECORD.pack(
                        pair_id,
                        plane_id,
                        int(chart_id),
                        *(int(value) for value in dense[pair_id, plane_id, chart_id]),
                    )
                )
    return bytes(out)


def _encode_aqc1(payload: bytes) -> bytes:
    return encode_qints_arithmetic_compact(np.frombuffer(payload, dtype=np.uint8), num_symbols=256, offset=0)


def _decode_aqc1(payload: bytes) -> bytes:
    return decode_qints_arithmetic_compact(payload, expected_dtype=np.uint8).tobytes()


def _encode_huffman_rank16(payload: bytes) -> bytes:
    symbols = np.frombuffer(payload, dtype=np.uint8)
    alphabet = np.unique(symbols)
    if alphabet.size < 1 or alphabet.size > 16:
        raise DirectDescriptionError("ranked Huffman candidate requires 1..16 occupied byte symbols")
    mapped = np.searchsorted(alphabet, symbols).astype(np.uint8)
    lengths = _build_optimal_huffman_lengths(mapped)
    rank = encode_huff_length_rank(lengths)
    encoded = _bit_pack(mapped.tolist(), _build_canonical_huffman_codebook(lengths))
    return (
        bytes((int(alphabet.size),))
        + alphabet.tobytes()
        + rank.to_bytes(SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN, "little")
        + encoded
    )


def _decode_huffman_rank16(payload: bytes, expected_symbols: int) -> bytes:
    if not payload:
        raise DirectDescriptionError("ranked Huffman payload is empty")
    alphabet_size = payload[0]
    cursor = 1
    if not 1 <= alphabet_size <= 16 or len(payload) < cursor + alphabet_size + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN:
        raise DirectDescriptionError("ranked Huffman header is malformed")
    alphabet = np.frombuffer(payload[cursor : cursor + alphabet_size], dtype=np.uint8)
    if alphabet.size > 1 and not np.all(np.diff(alphabet.astype(np.int16)) > 0):
        raise DirectDescriptionError("ranked Huffman alphabet is noncanonical")
    cursor += alphabet_size
    rank = int.from_bytes(payload[cursor : cursor + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN], "little")
    cursor += SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN
    try:
        lengths = decode_huff_length_rank(rank)
        mapped = _decode_canonical_huffman_n(payload[cursor:], lengths, expected_symbols)
    except (IndexError, ValueError) as exc:
        raise DirectDescriptionError("ranked Huffman payload failed strict decode") from exc
    if mapped.size != expected_symbols or np.any(mapped >= alphabet_size):
        raise DirectDescriptionError("ranked Huffman decoded symbol is outside transmitted alphabet")
    bit_count = sum(int(lengths[int(symbol)]) for symbol in mapped)
    encoded = payload[cursor:]
    if len(encoded) != (bit_count + 7) // 8:
        raise DirectDescriptionError("ranked Huffman payload has truncated or trailing bytes")
    padding = len(encoded) * 8 - bit_count
    if padding and encoded[-1] & ((1 << padding) - 1):
        raise DirectDescriptionError("ranked Huffman payload has nonzero padding")
    return alphabet[mapped].tobytes()


def _generic_candidates(payload: bytes) -> tuple[tuple[int, bytes, str | None], ...]:
    rows: list[tuple[int, bytes, str | None]] = []
    for coder_id, encoder in (
        (CODER_BROTLI_Q11, encode_brotli_q11),
        (CODER_LZMA_XZ9, encode_lzma),
        (CODER_AQC1, _encode_aqc1),
        (CODER_HUFFMAN_RANK16, _encode_huffman_rank16),
    ):
        try:
            encoded = encoder(payload)
            if _decode_generic(coder_id, encoded, len(payload)) != payload:
                raise DirectDescriptionError("coder parse-back differs")
            rows.append((coder_id, encoded, None))
        except (DirectDescriptionError, RateCoderError, ValueError) as exc:
            rows.append((coder_id, b"", str(exc)))
    return tuple(rows)


def _decode_generic(coder_id: int, payload: bytes, expected_bytes: int) -> bytes:
    try:
        if coder_id == CODER_BROTLI_Q11:
            decoded = decode_brotli_q11(payload)
        elif coder_id == CODER_LZMA_XZ9:
            decoded = decode_lzma(payload)
        elif coder_id == CODER_AQC1:
            decoded = _decode_aqc1(payload)
        elif coder_id == CODER_HUFFMAN_RANK16:
            decoded = _decode_huffman_rank16(payload, expected_bytes)
        else:
            raise DirectDescriptionError("unknown generic entropy coder id")
    except (DirectDescriptionError, IndexError, RateCoderError, ValueError) as exc:
        raise DirectDescriptionError("entropy coder strict decode failed") from exc
    if len(decoded) != expected_bytes:
        raise DirectDescriptionError("entropy coder decoded byte count mismatch")
    return decoded


def _encode_split_rice(canonical: bytes) -> tuple[bytes, dict[str, Any]]:
    metadata, values = _parse_transform_body(canonical)
    header = canonical[: _TRANSFORM_HEADER.size]
    if metadata:
        meta_rows = [row for row in _generic_candidates(metadata) if row[2] is None]
        if not meta_rows:
            raise DirectDescriptionError("no exact metadata entropy coder is available")
        meta_coder, meta_payload, _ = min(meta_rows, key=lambda row: (len(row[1]), row[0]))
    else:
        meta_coder, meta_payload = 0, b""
    measurements = measure_residual_schemes(values.astype(np.int64))
    eligible = tuple(name for name in ("zlib9", "rice") if name in RESIDUAL_SCHEMES)
    scheme = min(eligible, key=lambda name: (measurements[name], residual_scheme_id(name)))
    encoded_values = encode_residual_matrix(values.astype(np.int64), scheme)
    payload = (
        header
        + _SPLIT_HEADER.pack(meta_coder, len(meta_payload), residual_scheme_id(scheme), len(encoded_values))
        + meta_payload
        + encoded_values
    )
    if _decode_split_rice(payload) != canonical:
        raise DirectDescriptionError("split Rice/Golomb candidate failed exact transform roundtrip")
    return payload, {
        "metadata_coder": "none" if meta_coder == 0 else CODER_NAME[meta_coder],
        "metadata_coded_bytes": len(meta_payload),
        "value_scheme": scheme,
        "value_coded_bytes": len(encoded_values),
        "measured_value_scheme_bytes": measurements,
        "raw_varint_measured_not_selectable_as_entropy_stream": measurements.get("varint"),
    }


def _decode_split_rice(payload: bytes) -> bytes:
    if len(payload) < _TRANSFORM_HEADER.size + _SPLIT_HEADER.size:
        raise DirectDescriptionError("split Rice/Golomb payload is truncated")
    transform_header = payload[: _TRANSFORM_HEADER.size]
    metadata_bytes, rows, columns, dtype_id = _TRANSFORM_HEADER.unpack(transform_header)
    dtype = _DTYPE_FROM_ID.get(dtype_id)
    if dtype is None or rows < 1 or columns < 1:
        raise DirectDescriptionError("split Rice/Golomb transform header is invalid")
    cursor = _TRANSFORM_HEADER.size
    meta_coder, meta_coded_bytes, scheme_id, value_coded_bytes = _SPLIT_HEADER.unpack_from(payload, cursor)
    cursor += _SPLIT_HEADER.size
    if cursor + meta_coded_bytes + value_coded_bytes != len(payload):
        raise DirectDescriptionError("split Rice/Golomb section lengths do not consume payload exactly")
    meta_payload = payload[cursor : cursor + meta_coded_bytes]
    cursor += meta_coded_bytes
    value_payload = payload[cursor : cursor + value_coded_bytes]
    if metadata_bytes == 0:
        if meta_coder != 0 or meta_payload:
            raise DirectDescriptionError("split Rice/Golomb empty metadata has a coded section")
        metadata = b""
    else:
        if meta_coder not in (CODER_BROTLI_Q11, CODER_LZMA_XZ9, CODER_AQC1, CODER_HUFFMAN_RANK16):
            raise DirectDescriptionError("split Rice/Golomb metadata coder is invalid")
        metadata = _decode_generic(meta_coder, meta_payload, metadata_bytes)
    try:
        scheme_name = residual_scheme_name(scheme_id)
        if scheme_name not in {"zlib9", "rice"}:
            raise DirectDescriptionError("split residual scheme is not entropy-coded")
        values64 = decode_residual_matrix(value_payload, scheme_id, rows, columns)
    except (IndexError, ValueError) as exc:
        raise DirectDescriptionError("split Rice/Golomb value section failed strict decode") from exc
    info = np.iinfo(dtype)
    if values64.min(initial=0) < info.min or values64.max(initial=0) > info.max:
        raise DirectDescriptionError("split Rice/Golomb values exceed declared dtype")
    values = values64.astype(dtype)
    return transform_header + metadata + values.tobytes(order="C")


def _decode_coder(coder_id: int, payload: bytes, expected_bytes: int) -> bytes:
    if coder_id == CODER_SPLIT_RICE:
        decoded = _decode_split_rice(payload)
        if len(decoded) != expected_bytes:
            raise DirectDescriptionError("split Rice/Golomb decoded byte count mismatch")
        return decoded
    return _decode_generic(coder_id, payload, expected_bytes)


@lru_cache(maxsize=256)
def _encode_entropy_stream_cached(stream_name: str, semantic_payload: bytes, n_pairs: int) -> EntropyStreamBuildV1:
    expected = _expected_record_count(stream_name, n_pairs)
    candidates: list[dict[str, Any]] = []
    encoded_candidates: list[tuple[int, int, bytes, bytes, Mapping[str, Any] | None]] = []
    for transform in _transform_candidates(stream_name, semantic_payload, n_pairs):
        canonical = transform.canonical_bytes()
        if _decode_transform(stream_name, n_pairs, transform.transform_id, canonical) != semantic_payload:
            raise DirectDescriptionError("entropy transform changed semantic payload")
        for coder_id, coded, refusal in _generic_candidates(canonical):
            candidates.append(
                {
                    "transform": TRANSFORM_NAME[transform.transform_id],
                    "coder": CODER_NAME[coder_id],
                    "canonical_transform_bytes": len(canonical),
                    "coded_payload_bytes": None if refusal is not None else len(coded),
                    "available": refusal is None,
                    "refusal_reason": refusal,
                }
            )
            if refusal is None:
                encoded_candidates.append((transform.transform_id, coder_id, canonical, coded, None))
        try:
            split, detail = _encode_split_rice(canonical)
            candidates.append(
                {
                    "transform": TRANSFORM_NAME[transform.transform_id],
                    "coder": CODER_NAME[CODER_SPLIT_RICE],
                    "canonical_transform_bytes": len(canonical),
                    "coded_payload_bytes": len(split),
                    "available": True,
                    "refusal_reason": None,
                    "split_detail": detail,
                }
            )
            encoded_candidates.append((transform.transform_id, CODER_SPLIT_RICE, canonical, split, detail))
        except (DirectDescriptionError, ValueError) as exc:
            candidates.append(
                {
                    "transform": TRANSFORM_NAME[transform.transform_id],
                    "coder": CODER_NAME[CODER_SPLIT_RICE],
                    "canonical_transform_bytes": len(canonical),
                    "coded_payload_bytes": None,
                    "available": False,
                    "refusal_reason": str(exc),
                }
            )
    if not encoded_candidates:
        raise DirectDescriptionError(f"no entropy candidate can encode {stream_name}")
    transform_id, coder_id, canonical, coded, detail = min(
        encoded_candidates,
        key=lambda row: (len(row[3]), row[0], row[1]),
    )
    if _decode_coder(coder_id, coded, len(canonical)) != canonical:
        raise DirectDescriptionError("selected entropy coder changed canonical transform")
    if _decode_transform(stream_name, n_pairs, transform_id, canonical) != semantic_payload:
        raise DirectDescriptionError("selected entropy stream changed semantic payload")
    frame = (
        _ENTROPY_FRAME.pack(
            _ENTROPY_MAGIC[stream_name],
            1,
            n_pairs,
            transform_id,
            coder_id,
            0,
            expected,
            len(semantic_payload),
            len(canonical),
            len(coded),
            bytes.fromhex(_sha256(semantic_payload)),
        )
        + coded
    )
    return EntropyStreamBuildV1(
        stream=stream_name,
        transform_id=transform_id,
        coder_id=coder_id,
        frame=frame,
        semantic_payload=semantic_payload,
        canonical_transform=canonical,
        coded_payload=coded,
        candidate_rows=tuple(candidates),
        split_detail=detail,
    )


def encode_entropy_stream(stream_name: str, semantic_payload: bytes, n_pairs: int) -> EntropyStreamBuildV1:
    if stream_name not in STREAM_ORDER or not isinstance(semantic_payload, bytes) or not semantic_payload:
        raise DirectDescriptionError("entropy stream input is invalid")
    return _encode_entropy_stream_cached(stream_name, semantic_payload, n_pairs)


def parse_entropy_stream(stream_name: str, frame: bytes, n_pairs: int | None = None) -> tuple[int, bytes]:
    if len(frame) < _ENTROPY_FRAME.size:
        raise DirectDescriptionError(f"{stream_name} entropy frame is truncated")
    (
        magic,
        version,
        observed_pairs,
        transform_id,
        coder_id,
        reserved,
        record_count,
        semantic_bytes,
        canonical_bytes,
        coded_bytes,
        digest,
    ) = _ENTROPY_FRAME.unpack_from(frame)
    if (
        magic != _ENTROPY_MAGIC[stream_name]
        or version != 1
        or not 1 <= observed_pairs <= 600
        or reserved != 0
        or transform_id not in TRANSFORM_NAME
        or coder_id not in CODER_NAME
    ):
        raise DirectDescriptionError(f"{stream_name} entropy frame identity mismatch")
    if n_pairs is not None and observed_pairs != n_pairs:
        raise DirectDescriptionError("entropy chart streams disagree on pair count")
    if record_count != _expected_record_count(stream_name, observed_pairs):
        raise DirectDescriptionError(f"{stream_name} entropy record count mismatch")
    coded = frame[_ENTROPY_FRAME.size :]
    if coded_bytes != len(coded) or _ENTROPY_FRAME.size + coded_bytes != len(frame):
        raise DirectDescriptionError(f"{stream_name} entropy frame has truncated or trailing coded bytes")
    canonical = _decode_coder(coder_id, coded, canonical_bytes)
    semantic = _decode_transform(stream_name, observed_pairs, transform_id, canonical)
    if len(semantic) != semantic_bytes or bytes.fromhex(_sha256(semantic)) != digest:
        raise DirectDescriptionError(f"{stream_name} entropy semantic length/hash mismatch")
    rebuilt = encode_entropy_stream(stream_name, semantic, observed_pairs)
    if rebuilt.frame != frame:
        raise DirectDescriptionError(f"{stream_name} entropy frame is not canonical tournament output")
    return observed_pairs, semantic


def compile_entropy_chart_archive(z: DirectDescriptionChartZV1) -> EntropyChartArchiveBuildResultV1:
    streams = {name: encode_entropy_stream(name, getattr(z, name).payload, z.n_pairs) for name in STREAM_ORDER}
    framed = {MEMBER_BY_STREAM[name]: streams[name].frame for name in STREAM_ORDER}
    archive = _deterministic_zip(framed)
    return EntropyChartArchiveBuildResultV1(archive=archive, framed_members=framed, z=z, streams=streams)


def _read_entropy_zip(archive_bytes: bytes) -> dict[str, bytes]:
    expected = tuple(MEMBER_BY_STREAM[name] for name in STREAM_ORDER)
    framed: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as handle:
            infos = handle.infolist()
            if tuple(row.filename for row in infos) != expected or handle.comment:
                raise DirectDescriptionError("entropy chart archive order/comment is noncanonical")
            for row in infos:
                if (
                    row.compress_type != zipfile.ZIP_STORED
                    or row.flag_bits != 0
                    or row.date_time != (1980, 1, 1, 0, 0, 0)
                    or row.extra
                    or row.comment
                    or row.compress_size != row.file_size
                    or row.external_attr != 0o100644 << 16
                    or row.create_system != 3
                ):
                    raise DirectDescriptionError("entropy chart ZIP member framing is noncanonical")
                framed[row.filename] = handle.read(row.filename)
    except DirectDescriptionError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise DirectDescriptionError("entropy chart archive ZIP is malformed") from exc
    return framed


def parse_entropy_chart_archive(archive: bytes | Path) -> EntropyChartArchiveBuildResultV1:
    archive_bytes = _read_regular_file_once(archive) if isinstance(archive, Path) else archive
    if not isinstance(archive_bytes, bytes) or not archive_bytes:
        raise DirectDescriptionError("entropy chart archive must be nonempty exact bytes")
    framed = _read_entropy_zip(archive_bytes)
    values: dict[str, CountedChartStreamV1] = {}
    n_pairs: int | None = None
    for member_name in (MEMBER_BY_STREAM[name] for name in STREAM_ORDER):
        stream_name = STREAM_BY_MEMBER[member_name]
        n_pairs, payload = parse_entropy_stream(stream_name, framed[member_name], n_pairs)
        values[stream_name] = CountedChartStreamV1(payload=payload)
    assert n_pairs is not None
    z = DirectDescriptionChartZV1(n_pairs=n_pairs, **values)
    rebuilt = compile_entropy_chart_archive(z)
    if rebuilt.framed_members != framed or rebuilt.archive != archive_bytes:
        raise DirectDescriptionError("entropy chart archive parse/re-encode identity failed")
    return rebuilt


def receive_entropy_chart_archive(archive: bytes | Path) -> EntropyChartReceiverResultV1:
    parsed = parse_entropy_chart_archive(archive)
    semantic_archive = compile_chart_archive(parsed.z).archive
    semantic = receive_chart_archive(semantic_archive)
    custody = {
        **parsed.custody(),
        "schema": RECEIVER_SCHEMA,
        "receiver": "entropy_decode_then_numpy_integer_uint8_chart_reference.v1",
        "semantic_receiver_archive_sha256": _sha256(semantic_archive),
        "semantic_receiver_archive_bytes": len(semantic_archive),
        "all_members_consumed_once": True,
        "all_coded_sections_consumed_exactly": True,
        "semantic_payloads_reconstructed_byte_identically": True,
        "receiver_consumption_verified": True,
        "source_raw_reference_used": False,
        "score_claim": False,
        "evidence_axis": "[macOS-CPU frozen-SegNet advisory]",
    }
    return EntropyChartReceiverResultV1(
        archive=parsed.archive,
        z=parsed.z,
        anchors=semantic.anchors,
        gradients=semantic.gradients,
        residuals=semantic.residuals,
        pose6_codes=semantic.pose6_codes,
        custody=custody,
        _semantic_receiver=semantic,
    )


def prove_entropy_home_fail_closed(z: DirectDescriptionChartZV1) -> dict[str, Any]:
    """Mutate every ZIP home class and require refusal or changed decode."""

    compiled = compile_entropy_chart_archive(z)
    baseline = receive_entropy_chart_archive(compiled.archive)
    digest = stream_decode_digest(baseline, n_pairs=z.n_pairs)
    positions: set[int] = set()
    homes = _zip_unique_home_ledger(compiled.archive)
    for row in homes:
        for span in row["home_ranges"]:
            positions.update({span["start"], (span["start"] + span["end"] - 1) // 2, span["end"] - 1})
    refused = 0
    changed = 0
    for position in sorted(positions):
        mutated = bytearray(compiled.archive)
        mutated[position] ^= 1
        try:
            observed = receive_entropy_chart_archive(bytes(mutated))
        except DirectDescriptionError:
            refused += 1
            continue
        if stream_decode_digest(observed, n_pairs=z.n_pairs) == digest:
            raise DirectDescriptionError("entropy archive-home mutation was silently inert")
        changed += 1
    return {
        "sampled_positions": len(positions),
        "fail_closed_refusals": refused,
        "changed_receiver_outputs": changed,
        "all_samples_effective_or_refused": refused + changed == len(positions),
    }


__all__ = [
    "CODER_NAME",
    "TRANSFORM_NAME",
    "EntropyChartArchiveBuildResultV1",
    "EntropyChartReceiverResultV1",
    "compile_entropy_chart_archive",
    "encode_entropy_stream",
    "parse_entropy_chart_archive",
    "parse_entropy_stream",
    "prove_entropy_home_fail_closed",
    "receive_entropy_chart_archive",
]
