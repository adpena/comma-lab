# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    RankedSidecarSchema,
    decode_centered_delta_uint8,
    decode_ranked_no_op_sidecar,
    encode_centered_delta_uint8,
    encode_ranked_no_op_sidecar,
    parse_split_brotli_self_delimiting,
    split_brotli_self_delimiting,
)
from tac.packet_compiler.pr101_fec6_packetir import (
    parse_pr101_fec6_packetir_member,
    read_single_stored_fec6_member_archive,
)
from tac.packet_compiler.pr101_fec6_source_anatomy import (
    PR101_DECODER_BLOB_LEN,
    PR101_LATENT_BLOB_LEN,
)
from tac.packet_compiler.pr101_sidecar_grammar import (
    _build_canonical_huffman_codebook,
    _decode_combination_colex,
    _decode_huff_length_rank,
    _encode_combination_colex,
    _encode_huff_length_rank,
    _huff_length_vector_count,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent
    / "packet_compiler"
    / "golden_vectors"
)
PR101_DELTAS = (-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10)
REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_FEC6_DIR = (
    REPO_ROOT
    / "experiments/results/"
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
    "submission_dir"
)
REAL_FEC6_ARCHIVE = REAL_FEC6_DIR / "archive.zip"


def _ranked_decode_widths(
    schema: RankedSidecarSchema,
    *,
    noop_count: int,
    dims: np.ndarray | None = None,
) -> tuple[int, int, int]:
    n_valid = schema.n_pairs - noop_count
    dim_bits = max(1, math.ceil(n_valid * math.log2(max(schema.n_dims, 2))))
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
    if dims is None:
        noop_total = max(math.comb(schema.n_pairs, noop_count), 1)
        noop_rank_bits = max(1, math.ceil(math.log2(noop_total)))
        noop_rank_bytes = (noop_rank_bits + 7) // 8
    else:
        noop_pos = np.where(np.asarray(dims) == schema.no_op_sentinel)[0]
        noop_rank = _encode_combination_colex(noop_pos.astype(np.int64), schema.n_pairs)
        noop_rank_bytes = max(1, (int(noop_rank).bit_length() + 7) // 8)
    return dim_bytes, rank_bytes, noop_rank_bytes


def test_pr101_ranked_sidecar_round_trips_sparse_corrections() -> None:
    schema = RankedSidecarSchema(
        n_pairs=8,
        n_dims=4,
        deltas=(-2, -1, 1, 2),
        huff_min_len=2,
        huff_max_len=4,
    )
    dims = np.array([0, 255, 2, 1, 255, 3, 0, 255], dtype=np.int64)
    delta_indices = np.array([0, 0, 1, 2, 0, 3, 1, 0], dtype=np.int64)

    payload = encode_ranked_no_op_sidecar(
        dims=dims,
        delta_indices=delta_indices,
        schema=schema,
    )
    noop_count = int((dims == schema.no_op_sentinel).sum())
    dim_bytes, rank_bytes, noop_rank_bytes = _ranked_decode_widths(
        schema,
        noop_count=noop_count,
        dims=dims,
    )

    decoded_dims, decoded_delta_indices = decode_ranked_no_op_sidecar(
        payload,
        schema=schema,
        dim_bytes=dim_bytes,
        rank_bytes=rank_bytes,
        noop_rank_bytes=noop_rank_bytes,
        noop_count=noop_count,
    )

    np.testing.assert_array_equal(decoded_dims, dims)
    np.testing.assert_array_equal(decoded_delta_indices, delta_indices)


def test_pr101_ranked_sidecar_fec6_widths_are_exact_not_legacy_worst_case() -> None:
    schema = RankedSidecarSchema(
        n_pairs=600,
        n_dims=28,
        deltas=PR101_DELTAS,
        huff_min_len=2,
        huff_max_len=8,
    )
    rng = np.random.default_rng(seed=20260520)
    dims = rng.integers(0, schema.n_dims, size=schema.n_pairs, dtype=np.int64)
    delta_indices = rng.integers(
        0, len(schema.deltas), size=schema.n_pairs, dtype=np.int64
    )
    # Same no-op coordinates as the live PR101/FEC6 sidecar.  This keeps the
    # rank-width guard permanent even when the experiment archive is absent.
    noop_pos = np.array([91, 428, 457], dtype=np.int64)
    dims[noop_pos] = schema.no_op_sentinel
    delta_indices[noop_pos] = 0

    payload = encode_ranked_no_op_sidecar(
        dims=dims,
        delta_indices=delta_indices,
        schema=schema,
    )
    dim_bytes, rank_bytes, noop_rank_bytes = _ranked_decode_widths(
        schema,
        noop_count=int(noop_pos.size),
        dims=dims,
    )

    assert dim_bytes == 359
    assert rank_bytes == 5
    assert noop_rank_bytes == 3
    assert math.ceil((schema.n_pairs - noop_pos.size) * math.log2(schema.n_dims)) == (
        359 * 8 - 2
    )
    noop_rank = _encode_combination_colex(noop_pos, schema.n_pairs)
    assert (
        math.ceil(math.log2(max(math.comb(schema.n_pairs, int(noop_pos.size)), 2)))
        == 26
    )
    assert (int(noop_rank).bit_length() + 7) // 8 == 3
    decoded_dims, decoded_delta_indices = decode_ranked_no_op_sidecar(
        payload,
        schema=schema,
        dim_bytes=dim_bytes,
        rank_bytes=rank_bytes,
        noop_rank_bytes=noop_rank_bytes,
        noop_count=int(noop_pos.size),
    )

    np.testing.assert_array_equal(decoded_dims, dims)
    np.testing.assert_array_equal(decoded_delta_indices, delta_indices)


def test_real_pr101_fec6_sidecar_reencodes_byte_identically_if_present() -> None:
    if not REAL_FEC6_ARCHIVE.exists():
        pytest.skip(f"missing real PR101/FEC6 archive: {REAL_FEC6_ARCHIVE}")

    member = read_single_stored_fec6_member_archive(REAL_FEC6_ARCHIVE.read_bytes())
    packet = parse_pr101_fec6_packetir_member(member.payload)
    sidecar = packet.source_pr101_payload[
        PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN :
    ]
    spec = importlib.util.spec_from_file_location(
        "pr101_fec6_runtime_codec_sidecar_for_test",
        REAL_FEC6_DIR / "src" / "codec_sidecar.py",
    )
    assert spec is not None and spec.loader is not None
    runtime_sidecar = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = runtime_sidecar
        spec.loader.exec_module(runtime_sidecar)
    finally:
        sys.modules.pop(spec.name, None)

    dims, codes = runtime_sidecar._decode_latent_sidecar_vectors(sidecar)
    schema = RankedSidecarSchema(
        n_pairs=600,
        n_dims=28,
        deltas=PR101_DELTAS,
        huff_min_len=2,
        huff_max_len=8,
        no_op_sentinel=255,
    )
    delta_to_index = {delta: index for index, delta in enumerate(PR101_DELTAS)}
    valid = dims != schema.no_op_sentinel
    delta_indices = np.zeros(schema.n_pairs, dtype=np.int64)
    for pair_index, code in enumerate(codes.astype(np.int64)):
        if valid[pair_index]:
            delta_indices[pair_index] = delta_to_index[int(code)]

    encoded = encode_ranked_no_op_sidecar(
        dims=dims.astype(np.int64),
        delta_indices=delta_indices,
        schema=schema,
    )

    assert len(sidecar) == 607
    assert encoded == sidecar


def test_pr101_centered_delta_uint8_round_trip_preserves_quantized_values() -> None:
    values = np.array(
        [
            [1.00, 2.00, -0.50],
            [1.25, 2.10, -0.45],
            [1.10, 2.40, -0.30],
            [1.60, 2.30, -0.20],
        ],
        dtype=np.float32,
    )

    stream = encode_centered_delta_uint8(values)
    decoded = decode_centered_delta_uint8(
        stream.lzma_bytes,
        n_pairs=values.shape[0],
        n_dims=values.shape[1],
    )

    assert decoded.shape == values.shape
    assert np.max(np.abs(decoded - values)) < 0.01


def test_pr101_split_brotli_self_delimiting_round_trips_and_rejects_tail() -> None:
    streams = [b"abc", b"hello" * 20, bytes(range(32))]

    encoded = split_brotli_self_delimiting(streams, quality=6)

    assert parse_split_brotli_self_delimiting(
        encoded.payload,
        n_streams=encoded.n_streams,
    ) == streams
    with pytest.raises(ValueError, match="trailing data"):
        parse_split_brotli_self_delimiting(
            encoded.payload + b"x",
            n_streams=encoded.n_streams,
        )


def test_pr101_schema_rejects_sentinel_colliding_with_dimensions() -> None:
    with pytest.raises(ValueError, match="no_op_sentinel"):
        RankedSidecarSchema(
            n_pairs=8,
            n_dims=4,
            deltas=(-1, 1),
            no_op_sentinel=3,
        )


# ── Extended RankedSidecarSchema validation ─────────────────────────────────


class TestRankedSidecarSchemaValidation:
    def test_rejects_zero_n_pairs(self) -> None:
        with pytest.raises(ValueError, match="n_pairs"):
            RankedSidecarSchema(n_pairs=0, n_dims=8, deltas=PR101_DELTAS)

    def test_rejects_zero_n_dims(self) -> None:
        with pytest.raises(ValueError, match="n_dims"):
            RankedSidecarSchema(n_pairs=10, n_dims=0, deltas=PR101_DELTAS)

    def test_rejects_too_few_deltas(self) -> None:
        with pytest.raises(ValueError, match="at least 2 delta"):
            RankedSidecarSchema(n_pairs=10, n_dims=8, deltas=(1,))

    def test_rejects_unsorted_deltas(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            RankedSidecarSchema(n_pairs=10, n_dims=8, deltas=(3, 1, 5))

    def test_rejects_invalid_huff_bounds(self) -> None:
        with pytest.raises(ValueError, match="huff_min_len"):
            RankedSidecarSchema(
                n_pairs=10,
                n_dims=8,
                deltas=PR101_DELTAS,
                huff_min_len=5,
                huff_max_len=3,
            )

    def test_kraft_total_matches_max_len(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=8,
            n_dims=4,
            deltas=(-1, 1),
            huff_min_len=1,
            huff_max_len=5,
        )
        assert schema.kraft_total == 32

    def test_frozen_schema(self) -> None:
        schema = RankedSidecarSchema(n_pairs=8, n_dims=4, deltas=PR101_DELTAS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            schema.n_pairs = 99  # type: ignore[misc]


# ── Combination co-lex rank ─────────────────────────────────────────────────


class TestCombinationColex:
    @pytest.mark.parametrize(
        "n,k", [(10, 3), (16, 5), (20, 0), (20, 20), (32, 7), (5, 2)]
    )
    def test_round_trip(self, n: int, k: int) -> None:
        rng = np.random.default_rng(seed=n * 100 + k)
        positions = np.empty(0, dtype=np.int64) if k == 0 else np.sort(rng.choice(n, size=k, replace=False))
        rank = _encode_combination_colex(positions.astype(np.int64), n)
        decoded = _decode_combination_colex(rank, n, k)
        np.testing.assert_array_equal(decoded, positions)

    def test_decode_invalid_rank(self) -> None:
        with pytest.raises(ValueError, match="combination rank"):
            _decode_combination_colex(10, 5, 2)  # C(5,2)=10

    def test_encode_rejects_dup(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _encode_combination_colex(np.array([1, 1, 3], dtype=np.int64), 10)


# ── Huffman length-vector rank ──────────────────────────────────────────────


class TestHuffmanLengthRank:
    def test_round_trip_kraft_tight(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=20,
            n_dims=8,
            deltas=PR101_DELTAS,
            huff_min_len=2,
            huff_max_len=8,
        )
        lengths = np.full(len(schema.deltas), 4, dtype=np.uint8)
        assert (
            sum(1 << (schema.huff_max_len - int(L)) for L in lengths)
            == schema.kraft_total
        )
        rank = _encode_huff_length_rank(lengths, schema)
        recovered = _decode_huff_length_rank(rank, schema)
        np.testing.assert_array_equal(recovered, lengths)

    def test_round_trip_uniform_short(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=10,
            n_dims=4,
            deltas=(-2, -1, 1, 2),
            huff_min_len=2,
            huff_max_len=4,
        )
        lengths = np.array([2, 2, 2, 2], dtype=np.uint8)
        rank = _encode_huff_length_rank(lengths, schema)
        recovered = _decode_huff_length_rank(rank, schema)
        np.testing.assert_array_equal(recovered, lengths)

    def test_decode_rejects_oversize_rank(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=10,
            n_dims=4,
            deltas=(-1, 1),
            huff_min_len=1,
            huff_max_len=3,
        )
        with pytest.raises(ValueError, match="length-vector rank"):
            _decode_huff_length_rank(10**6, schema)


# ── Canonical Huffman codebook ──────────────────────────────────────────────


class TestCanonicalHuffmanCodebook:
    def test_builds_prefix_free_codes(self) -> None:
        lengths = np.array([2, 2, 3, 3, 3, 3, 3, 3], dtype=np.uint8)
        cb = _build_canonical_huffman_codebook(lengths)
        assert set(cb.keys()) == set(range(8))
        items = sorted(cb.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        seen: list[tuple[int, int]] = []
        for sym, (length, code) in items:
            for L_prev, c_prev in seen:
                if L_prev < length and (code >> (length - L_prev)) == c_prev:
                    pytest.fail(
                        f"symbol {sym} (L={length},c={code}) prefix-collides "
                        f"with prior (L={L_prev},c={c_prev})"
                    )
            seen.append((length, code))

    def test_zero_length_symbols_excluded(self) -> None:
        lengths = np.array([0, 1, 1, 0], dtype=np.uint8)
        cb = _build_canonical_huffman_codebook(lengths)
        assert set(cb.keys()) == {1, 2}


# ── Centered-delta uint8 ────────────────────────────────────────────────────


class TestCenteredDeltaUint8Extended:
    def test_round_trip_random(self) -> None:
        rng = np.random.default_rng(seed=1234)
        values = rng.uniform(-2.0, 2.0, size=(30, 5)).astype(np.float32)
        stream = encode_centered_delta_uint8(values)
        recovered = decode_centered_delta_uint8(stream)
        assert recovered.shape == values.shape
        for c in range(values.shape[1]):
            quantum = float(stream.scales[c])
            np.testing.assert_allclose(
                recovered[:, c], values[:, c], atol=quantum * 1.5
            )

    def test_round_trip_constant_column(self) -> None:
        values = np.full((10, 3), 0.5, dtype=np.float32)
        stream = encode_centered_delta_uint8(values)
        recovered = decode_centered_delta_uint8(stream)
        np.testing.assert_allclose(recovered, values, atol=1.0)

    def test_round_trip_from_bytes(self) -> None:
        rng = np.random.default_rng(seed=42)
        values = rng.uniform(0.0, 1.0, size=(20, 4)).astype(np.float32)
        stream = encode_centered_delta_uint8(values)
        recovered = decode_centered_delta_uint8(
            stream.lzma_bytes, n_pairs=20, n_dims=4
        )
        for c in range(values.shape[1]):
            quantum = float(stream.scales[c])
            np.testing.assert_allclose(
                recovered[:, c], values[:, c], atol=quantum * 1.5
            )

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            encode_centered_delta_uint8(np.zeros((10,), dtype=np.float32))

    def test_rejects_zero_shape(self) -> None:
        with pytest.raises(ValueError, match="positive shape"):
            encode_centered_delta_uint8(np.zeros((0, 5), dtype=np.float32))

    def test_decode_raw_bytes_requires_dims(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.uniform(0.0, 1.0, size=(5, 3)).astype(np.float32)
        stream = encode_centered_delta_uint8(values)
        with pytest.raises(ValueError, match="explicit n_pairs and n_dims"):
            decode_centered_delta_uint8(stream.lzma_bytes)

    def test_explicit_mins_scales(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.uniform(0.0, 1.0, size=(8, 3)).astype(np.float32)
        mins = np.zeros(3, dtype=np.float16)
        scales = np.full(3, 1.0 / 255.0, dtype=np.float16)
        stream = encode_centered_delta_uint8(values, mins=mins, scales=scales)
        np.testing.assert_array_equal(stream.mins, mins)
        np.testing.assert_array_equal(stream.scales, scales)
        recovered = decode_centered_delta_uint8(stream)
        np.testing.assert_allclose(recovered, values, atol=2.0 / 255.0)

    def test_rejects_zero_scales(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            encode_centered_delta_uint8(
                np.zeros((5, 2), dtype=np.float32),
                mins=np.zeros(2, dtype=np.float16),
                scales=np.zeros(2, dtype=np.float16),
            )

    def test_concentrated_signal_compresses_well(self) -> None:
        # Mostly-constant signal should LZMA below raw size.
        values = np.zeros((60, 8), dtype=np.float32)
        values[10, 3] = 1.0
        stream = encode_centered_delta_uint8(values)
        raw = 2 * 8 * 2 + 60 * 8  # mins+scales bytes + column-major data
        assert len(stream.lzma_bytes) < raw

    def test_frozen_stream(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.uniform(0.0, 1.0, size=(5, 3)).astype(np.float32)
        stream = encode_centered_delta_uint8(values)
        with pytest.raises(dataclasses.FrozenInstanceError):
            stream.lzma_bytes = b""  # type: ignore[misc]


# ── Split-Brotli ────────────────────────────────────────────────────────────


class TestSplitBrotliExtended:
    def test_round_trip_three_streams(self) -> None:
        rng = np.random.default_rng(seed=7)
        streams = [rng.bytes(64), rng.bytes(128), rng.bytes(256)]
        packed = split_brotli_self_delimiting(streams, lgwin=22, quality=6)
        decoded = parse_split_brotli_self_delimiting(
            packed.payload, n_streams=len(streams)
        )
        assert decoded == streams
        assert packed.n_streams == len(streams)
        assert packed.stream_byte_offsets[-1] == len(packed.payload)

    def test_round_trip_single_stream(self) -> None:
        s = b"hello world" * 100
        packed = split_brotli_self_delimiting([s], lgwin=22, quality=11)
        decoded = parse_split_brotli_self_delimiting(packed.payload, n_streams=1)
        assert decoded == [s]

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="at least one substream"):
            split_brotli_self_delimiting([])

    def test_parse_rejects_zero_streams(self) -> None:
        with pytest.raises(ValueError, match="n_streams"):
            parse_split_brotli_self_delimiting(b"", n_streams=0)

    def test_parse_rejects_wrong_stream_count(self) -> None:
        packed = split_brotli_self_delimiting(
            [b"a", b"b"], quality=1, lgwin=10
        )
        with pytest.raises(ValueError, match="trailing data"):
            parse_split_brotli_self_delimiting(packed.payload, n_streams=1)

    def test_parse_rejects_truncated_payload(self) -> None:
        packed = split_brotli_self_delimiting([b"hello"], quality=1, lgwin=10)
        with pytest.raises(ValueError, match="truncated"):
            parse_split_brotli_self_delimiting(packed.payload[:-1], n_streams=1)

    def test_offsets_match_stream_boundaries(self) -> None:
        streams = [b"alpha" * 10, b"beta" * 20, b"gamma" * 5]
        packed = split_brotli_self_delimiting(streams, lgwin=22, quality=11)
        assert len(packed.stream_byte_offsets) == len(streams)
        # offsets are monotone non-decreasing.
        for a, b in zip(
            packed.stream_byte_offsets, packed.stream_byte_offsets[1:], strict=False
        ):
            assert a < b


# ── Ranked Huffman/no-op sidecar end-to-end ─────────────────────────────────


class TestRankedNoOpSidecarExtended:
    def _round_trip(
        self,
        dims: np.ndarray,
        delta_idx: np.ndarray,
        schema: RankedSidecarSchema,
    ) -> tuple[np.ndarray, np.ndarray]:
        payload = encode_ranked_no_op_sidecar(
            dims=dims, delta_indices=delta_idx, schema=schema
        )
        noop_count = int((dims == schema.no_op_sentinel).sum())
        dim_bytes, rank_bytes, noop_rank_bytes = _ranked_decode_widths(
            schema, noop_count=noop_count, dims=dims
        )
        out_dims, out_delta_idx = decode_ranked_no_op_sidecar(
            payload,
            schema=schema,
            dim_bytes=dim_bytes,
            rank_bytes=rank_bytes,
            noop_rank_bytes=noop_rank_bytes,
            noop_count=noop_count,
        )
        return out_dims, out_delta_idx

    def test_all_no_ops(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=16, n_dims=4, deltas=(-1, 1), huff_min_len=1, huff_max_len=2
        )
        dims = np.full(schema.n_pairs, schema.no_op_sentinel, dtype=np.int64)
        delta_idx = np.zeros(schema.n_pairs, dtype=np.int64)
        out_dims, out_delta_idx = self._round_trip(dims, delta_idx, schema)
        np.testing.assert_array_equal(out_dims, dims)
        np.testing.assert_array_equal(out_delta_idx, delta_idx)

    def test_no_no_ops(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=8, n_dims=4, deltas=(-1, 1), huff_min_len=1, huff_max_len=2
        )
        rng = np.random.default_rng(0)
        dims = rng.integers(0, schema.n_dims, size=schema.n_pairs).astype(np.int64)
        delta_idx = rng.integers(
            0, len(schema.deltas), size=schema.n_pairs
        ).astype(np.int64)
        out_dims, out_delta_idx = self._round_trip(dims, delta_idx, schema)
        np.testing.assert_array_equal(out_dims, dims)
        np.testing.assert_array_equal(out_delta_idx, delta_idx)

    def test_rejects_wrong_shape(self) -> None:
        schema = RankedSidecarSchema(n_pairs=16, n_dims=4, deltas=PR101_DELTAS)
        with pytest.raises(ValueError, match="shape"):
            encode_ranked_no_op_sidecar(
                dims=np.zeros(5, dtype=np.int64),
                delta_indices=np.zeros(5, dtype=np.int64),
                schema=schema,
            )

    def test_rejects_dim_out_of_range(self) -> None:
        schema = RankedSidecarSchema(n_pairs=16, n_dims=4, deltas=PR101_DELTAS)
        dims = np.zeros(schema.n_pairs, dtype=np.int64)
        dims[0] = 99
        delta_idx = np.zeros(schema.n_pairs, dtype=np.int64)
        with pytest.raises(ValueError, match="valid dims"):
            encode_ranked_no_op_sidecar(
                dims=dims, delta_indices=delta_idx, schema=schema
            )

    def test_rejects_delta_index_out_of_range(self) -> None:
        schema = RankedSidecarSchema(n_pairs=16, n_dims=4, deltas=PR101_DELTAS)
        dims = np.zeros(schema.n_pairs, dtype=np.int64)
        delta_idx = np.full(schema.n_pairs, 999, dtype=np.int64)
        with pytest.raises(ValueError, match="delta_indices"):
            encode_ranked_no_op_sidecar(
                dims=dims, delta_indices=delta_idx, schema=schema
            )


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR101GoldenVectors:
    """Persist + verify byte-level conformance vectors for native ports."""

    def test_golden_dir_exists(self) -> None:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        assert GOLDEN_DIR.exists()

    def test_centered_delta_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260511)
        mins = np.linspace(-1.0, -0.5, 6, dtype=np.float16)
        scales = np.full(6, 2.0 / 255.0, dtype=np.float16)
        values = (
            mins.astype(np.float32)[None, :]
            + rng.uniform(0.0, 1.8, size=(40, 6)).astype(np.float32)
        )
        stream = encode_centered_delta_uint8(values, mins=mins, scales=scales)
        # Round-trip to ensure values reconstruct under documented tolerance.
        recovered = decode_centered_delta_uint8(stream)
        # Quantum + half-bit headroom: scales 2/255, so allow up to 2.5 quanta
        # since we centre and apply per-row deltas.
        np.testing.assert_allclose(
            recovered, values, atol=2.5 * 2.0 / 255.0
        )

        digest = hashlib.sha256(stream.lzma_bytes).hexdigest()
        golden = GOLDEN_DIR / "centered_delta_uint8_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "centered-delta uint8 LZMA byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            golden.write_text(
                json.dumps(
                    {
                        "schema": "centered_delta_uint8.v1",
                        "n_pairs": 40,
                        "n_dims": 6,
                        "seed": 20260511,
                        "mins_fp16": [float(x) for x in mins.tolist()],
                        "scales_fp16": [float(x) for x in scales.tolist()],
                        "lzma_bytes_len": len(stream.lzma_bytes),
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_split_brotli_golden_vector(self) -> None:
        streams = [
            b"PR101 sidecar grammar conformance vector A " * 16,
            b"Reusable byte primitives, deterministic build " * 12,
            b"Future Rust/Zig port must match these bytes " * 8,
        ]
        result = split_brotli_self_delimiting(streams, lgwin=22, quality=11)
        out = parse_split_brotli_self_delimiting(result.payload, n_streams=3)
        assert out == streams
        digest = hashlib.sha256(result.payload).hexdigest()
        golden = GOLDEN_DIR / "split_brotli_self_delim_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "split-Brotli byte stream changed; verify Brotli library "
                "version and regenerate vector if intentional"
            )
        else:
            golden.write_text(
                json.dumps(
                    {
                        "schema": "split_brotli_self_delim.v1",
                        "n_streams": 3,
                        "lgwin": 22,
                        "quality": 11,
                        "payload_len": len(result.payload),
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_ranked_sidecar_golden_vector(self) -> None:
        schema = RankedSidecarSchema(
            n_pairs=24,
            n_dims=8,
            deltas=PR101_DELTAS,
            huff_min_len=2,
            huff_max_len=8,
        )
        dims = np.full(schema.n_pairs, schema.no_op_sentinel, dtype=np.int64)
        delta_idx = np.zeros(schema.n_pairs, dtype=np.int64)
        # Deterministic correction pattern.
        for i in range(0, schema.n_pairs, 3):
            dims[i] = (2 + i // 3) % schema.n_dims
            delta_idx[i] = (5 + (i // 3) * 3) % len(schema.deltas)
        payload = encode_ranked_no_op_sidecar(
            dims=dims, delta_indices=delta_idx, schema=schema
        )
        digest = hashlib.sha256(payload).hexdigest()
        golden = GOLDEN_DIR / "ranked_no_op_sidecar_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "ranked-no-op sidecar byte stream changed; if package-merge "
                "Huffman differs the golden must be regenerated"
            )
        else:
            golden.write_text(
                json.dumps(
                    {
                        "schema": "ranked_no_op_sidecar.v1",
                        "n_pairs": schema.n_pairs,
                        "n_dims": schema.n_dims,
                        "deltas": list(schema.deltas),
                        "huff_min_len": schema.huff_min_len,
                        "huff_max_len": schema.huff_max_len,
                        "payload_len": len(payload),
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
