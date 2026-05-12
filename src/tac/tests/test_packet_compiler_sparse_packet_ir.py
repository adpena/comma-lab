"""Focused conformance tests for the sparse PacketIR codec.

Tests cover the 3 primitives + serialisation round-trips + edge cases +
failure modes + integration composition. Per CLAUDE.md "Beauty, simplicity,
and DX": every test has a one-line docstring explaining the contract.

Closes O's L2 wire-format ceiling (memory:
``feedback_l2_score_aware_encoders_wavelet_c3_cool_chic_landed_20260511.md``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    ArithmeticCodedCoefficientStream,
    RleOfZerosStream,
    SparsePacketIRError,
    TemporalSubsampledResidualStream,
    decode_arithmetic_coefficients,
    decode_rle_of_zeros,
    decode_temporal_subsampled,
    deserialize_arithmetic_coefficients,
    deserialize_rle_of_zeros,
    deserialize_temporal_subsampled,
    encode_arithmetic_coefficients,
    encode_rle_of_zeros,
    encode_temporal_subsampled,
    pad_per_frame_to_uniform_size_with_length_prefix,
    serialize_arithmetic_coefficients,
    serialize_rle_of_zeros,
    serialize_temporal_subsampled,
)


# ── Primitive 1: RLE-of-zeros ───────────────────────────────────────────────


def test_rle_round_trip_zero_array() -> None:
    """RLE of an all-zero array has 0 non-zeros and round-trips."""
    dense = np.zeros(100, dtype=np.int8)
    stream = encode_rle_of_zeros(dense)
    assert stream.nonzero_indices.size == 0
    assert stream.total_length == 100
    assert stream.sparsity_ratio == 1.0
    recovered = decode_rle_of_zeros(stream)
    np.testing.assert_array_equal(recovered, dense)


def test_rle_round_trip_sparse_int8() -> None:
    """RLE round-trip on a sparse int8 array preserves all values + positions."""
    dense = np.zeros(1000, dtype=np.int8)
    dense[10] = 7
    dense[100] = -5
    dense[500] = 127
    dense[999] = -128
    stream = encode_rle_of_zeros(dense)
    assert stream.nonzero_indices.size == 4
    assert stream.total_length == 1000
    np.testing.assert_array_equal(decode_rle_of_zeros(stream), dense)


def test_rle_round_trip_dense_array() -> None:
    """RLE works (suboptimally) on a fully dense array."""
    rng = np.random.default_rng(0)
    dense = rng.integers(-20, 20, size=200, dtype=np.int8)
    # Replace any zeros to make it fully dense.
    dense[dense == 0] = 1
    stream = encode_rle_of_zeros(dense)
    assert stream.nonzero_indices.size == 200
    assert stream.sparsity_ratio == 0.0
    np.testing.assert_array_equal(decode_rle_of_zeros(stream), dense)


def test_rle_auto_dtype_selection() -> None:
    """RLE auto-picks int8 if all values fit in [-128, 127]."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 10
    dense[15] = -20
    stream = encode_rle_of_zeros(dense)
    assert stream.nonzero_values.dtype == np.int8


def test_rle_auto_dtype_int16() -> None:
    """RLE auto-picks int16 when int8 cannot hold values."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 200
    dense[15] = -300
    stream = encode_rle_of_zeros(dense)
    assert stream.nonzero_values.dtype == np.int16


def test_rle_auto_dtype_int32() -> None:
    """RLE auto-picks int32 when int16 cannot hold values."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 100_000
    stream = encode_rle_of_zeros(dense)
    assert stream.nonzero_values.dtype == np.int32


def test_rle_explicit_dtype_override() -> None:
    """Caller-supplied dtype_nonzero is honoured when values fit."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 10
    stream = encode_rle_of_zeros(dense, dtype_nonzero=np.int32)
    assert stream.nonzero_values.dtype == np.int32


def test_rle_explicit_dtype_overflow_rejected() -> None:
    """Caller-supplied dtype_nonzero refuses values that overflow."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 200
    with pytest.raises(SparsePacketIRError, match="out of range"):
        encode_rle_of_zeros(dense, dtype_nonzero=np.int8)


def test_rle_unsupported_dtype_rejected() -> None:
    """Caller-supplied dtype_nonzero outside the allowed set is refused."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 10
    with pytest.raises(SparsePacketIRError, match="not in"):
        encode_rle_of_zeros(dense, dtype_nonzero=np.int64)


def test_rle_float_input_rejected() -> None:
    """Float dense arrays are refused (quantise first)."""
    with pytest.raises(SparsePacketIRError, match="integer"):
        encode_rle_of_zeros(np.zeros(10, dtype=np.float32))


def test_rle_non_1d_input_rejected() -> None:
    """RLE refuses non-1D arrays."""
    with pytest.raises(SparsePacketIRError, match="1D"):
        encode_rle_of_zeros(np.zeros((3, 4), dtype=np.int8))


def test_rle_dataclass_rejects_zero_value() -> None:
    """RleOfZerosStream constructor refuses a zero in nonzero_values."""
    with pytest.raises(SparsePacketIRError, match="zero entry"):
        RleOfZerosStream(
            nonzero_indices=np.array([0, 1], dtype=np.uint32),
            nonzero_values=np.array([5, 0], dtype=np.int8),
            total_length=10,
        )


def test_rle_dataclass_rejects_out_of_range_index() -> None:
    """RleOfZerosStream constructor refuses an index >= total_length."""
    with pytest.raises(SparsePacketIRError, match="index >= total_length"):
        RleOfZerosStream(
            nonzero_indices=np.array([0, 100], dtype=np.uint32),
            nonzero_values=np.array([5, 7], dtype=np.int8),
            total_length=10,
        )


def test_rle_dataclass_rejects_duplicate_or_unsorted_indices() -> None:
    """RleOfZerosStream refuses non-canonical duplicate/out-of-order indices."""
    with pytest.raises(SparsePacketIRError, match="strictly increasing"):
        RleOfZerosStream(
            nonzero_indices=np.array([0, 2, 2], dtype=np.uint32),
            nonzero_values=np.array([5, 7, 9], dtype=np.int8),
            total_length=10,
        )
    with pytest.raises(SparsePacketIRError, match="strictly increasing"):
        RleOfZerosStream(
            nonzero_indices=np.array([0, 3, 1], dtype=np.uint32),
            nonzero_values=np.array([5, 7, 9], dtype=np.int8),
            total_length=10,
        )


def test_rle_dataclass_rejects_mismatched_lengths() -> None:
    """RleOfZerosStream constructor refuses index/value length mismatch."""
    with pytest.raises(SparsePacketIRError, match="size"):
        RleOfZerosStream(
            nonzero_indices=np.array([0, 1, 2], dtype=np.uint32),
            nonzero_values=np.array([5, 7], dtype=np.int8),
            total_length=10,
        )


def test_rle_serialize_round_trip_int8() -> None:
    """RLE serialize → deserialize is bit-exact for int8."""
    dense = np.zeros(1000, dtype=np.int8)
    dense[100] = -50
    dense[500] = 60
    stream = encode_rle_of_zeros(dense)
    blob = serialize_rle_of_zeros(stream)
    recovered_stream = deserialize_rle_of_zeros(blob)
    assert recovered_stream.total_length == stream.total_length
    np.testing.assert_array_equal(
        recovered_stream.nonzero_indices, stream.nonzero_indices
    )
    np.testing.assert_array_equal(
        recovered_stream.nonzero_values, stream.nonzero_values
    )


def test_rle_serialize_round_trip_int16() -> None:
    """RLE serialize → deserialize is bit-exact for int16."""
    dense = np.zeros(100, dtype=np.int32)
    dense[5] = 300
    dense[15] = -400
    stream = encode_rle_of_zeros(dense)
    blob = serialize_rle_of_zeros(stream)
    recovered = deserialize_rle_of_zeros(blob)
    np.testing.assert_array_equal(
        decode_rle_of_zeros(recovered), dense.astype(np.int16)
    )


def test_rle_serialize_round_trip_int32() -> None:
    """RLE serialize → deserialize is bit-exact for int32."""
    dense = np.zeros(50, dtype=np.int32)
    dense[5] = 100_000
    stream = encode_rle_of_zeros(dense)
    blob = serialize_rle_of_zeros(stream)
    recovered = deserialize_rle_of_zeros(blob)
    np.testing.assert_array_equal(
        decode_rle_of_zeros(recovered), dense.astype(np.int32)
    )


def test_rle_deserialize_rejects_wrong_magic() -> None:
    """RLE deserialize refuses a blob with the wrong magic bytes."""
    bad = b"XXXX" + b"\x00" * 20
    with pytest.raises(SparsePacketIRError, match="magic mismatch"):
        deserialize_rle_of_zeros(bad)


def test_rle_deserialize_rejects_truncated_blob() -> None:
    """RLE deserialize refuses a truncated blob."""
    dense = np.zeros(50, dtype=np.int8)
    dense[5] = 10
    blob = serialize_rle_of_zeros(encode_rle_of_zeros(dense))
    with pytest.raises(SparsePacketIRError, match="truncated"):
        deserialize_rle_of_zeros(blob[:-1])


def test_rle_byte_savings_vs_dense() -> None:
    """RLE byte cost < dense int8 cost when sparsity > ~50%."""
    # 1000-element array; 100 nonzeros (90% sparse).
    dense = np.zeros(1000, dtype=np.int8)
    indices = np.arange(100) * 10
    dense[indices] = np.arange(1, 101).astype(np.int8)
    stream = encode_rle_of_zeros(dense)
    blob = serialize_rle_of_zeros(stream)
    # Header (13) + 100*4 (indices) + 100*1 (values) = 513 < 1000
    assert len(blob) < dense.nbytes


# ── Primitive 2: Arithmetic-coded coefficient stream ───────────────────────


def test_ac_round_trip_peaked_distribution() -> None:
    """AC round-trip on peaked-at-zero distribution preserves all values."""
    rng = np.random.default_rng(0)
    # Laplacian-ish: most values are small, occasional larger values
    raw = rng.laplace(0, 2, size=1000).round().astype(np.int32)
    raw = np.clip(raw, -10, 10)
    stream = encode_arithmetic_coefficients(raw)
    recovered = decode_arithmetic_coefficients(stream)
    np.testing.assert_array_equal(recovered, raw)


def test_ac_round_trip_uniform_distribution() -> None:
    """AC round-trip on uniform distribution preserves values."""
    rng = np.random.default_rng(1)
    raw = rng.integers(-50, 51, size=500, dtype=np.int32)
    stream = encode_arithmetic_coefficients(raw)
    recovered = decode_arithmetic_coefficients(stream)
    np.testing.assert_array_equal(recovered, raw)


def test_ac_round_trip_all_zeros() -> None:
    """AC round-trip on all-zero stream returns all zeros (alphabet padded to 2)."""
    raw = np.zeros(100, dtype=np.int8)
    stream = encode_arithmetic_coefficients(raw)
    recovered = decode_arithmetic_coefficients(stream)
    np.testing.assert_array_equal(recovered, raw)


def test_ac_round_trip_empty_input() -> None:
    """AC round-trip handles empty input gracefully."""
    raw = np.zeros(0, dtype=np.int32)
    stream = encode_arithmetic_coefficients(raw)
    assert stream.n_symbols == 0
    recovered = decode_arithmetic_coefficients(stream)
    assert recovered.size == 0


def test_ac_explicit_histogram_round_trip() -> None:
    """AC with a caller-supplied histogram round-trips correctly."""
    raw = np.zeros(200, dtype=np.int32)
    raw[5:10] = 1
    raw[100] = -1
    # Need symbol_offset=1 and alphabet_size=3 to map [-1, 0, 1] to [0, 1, 2]
    hist = np.array([1.0, 100.0, 5.0], dtype=np.float32)
    stream = encode_arithmetic_coefficients(
        raw, histogram=hist, symbol_offset=1, alphabet_size=3
    )
    recovered = decode_arithmetic_coefficients(stream)
    np.testing.assert_array_equal(recovered, raw)


def test_ac_compression_beats_uncompressed_on_peaked_distribution() -> None:
    """AC encoded size < raw int8 size on peaked distributions."""
    rng = np.random.default_rng(2)
    # 95% zeros, 5% small ±values
    raw = np.zeros(5000, dtype=np.int8)
    n_nonzero = 250
    positions = rng.choice(5000, size=n_nonzero, replace=False)
    raw[positions] = rng.choice([-1, 1], size=n_nonzero).astype(np.int8)
    stream = encode_arithmetic_coefficients(raw)
    blob = serialize_arithmetic_coefficients(stream)
    # AC should compress well; allow some overhead but require beating raw.
    assert len(blob) < raw.nbytes


def test_ac_dataclass_rejects_negative_histogram() -> None:
    """ArithmeticCodedCoefficientStream constructor refuses negative histogram."""
    with pytest.raises(SparsePacketIRError, match="negative"):
        ArithmeticCodedCoefficientStream(
            encoded_bytes=b"",
            histogram=np.array([1.0, -1.0, 1.0], dtype=np.float32),
            n_symbols=0,
            alphabet_size=3,
            symbol_offset=0,
        )


def test_ac_dataclass_rejects_wrong_histogram_shape() -> None:
    """ArithmeticCodedCoefficientStream rejects histogram shape mismatch."""
    with pytest.raises(SparsePacketIRError, match="shape"):
        ArithmeticCodedCoefficientStream(
            encoded_bytes=b"",
            histogram=np.array([1.0, 1.0], dtype=np.float32),
            n_symbols=0,
            alphabet_size=3,
            symbol_offset=0,
        )


def test_ac_dataclass_rejects_alphabet_size_lt_2() -> None:
    """ArithmeticCodedCoefficientStream refuses alphabet_size < 2."""
    with pytest.raises(SparsePacketIRError, match="alphabet_size"):
        ArithmeticCodedCoefficientStream(
            encoded_bytes=b"",
            histogram=np.array([1.0], dtype=np.float32),
            n_symbols=0,
            alphabet_size=1,
            symbol_offset=0,
        )


def test_ac_non_integer_input_rejected() -> None:
    """encode_arithmetic_coefficients refuses non-integer inputs."""
    with pytest.raises(SparsePacketIRError, match="integer"):
        encode_arithmetic_coefficients(np.zeros(10, dtype=np.float32))


def test_ac_non_1d_input_rejected() -> None:
    """encode_arithmetic_coefficients refuses non-1D inputs."""
    with pytest.raises(SparsePacketIRError, match="1D"):
        encode_arithmetic_coefficients(np.zeros((3, 4), dtype=np.int32))


def test_ac_serialize_round_trip() -> None:
    """AC serialize → deserialize → decode is bit-exact."""
    rng = np.random.default_rng(3)
    raw = rng.integers(-10, 11, size=300, dtype=np.int32)
    stream = encode_arithmetic_coefficients(raw)
    blob = serialize_arithmetic_coefficients(stream)
    recovered_stream = deserialize_arithmetic_coefficients(blob)
    recovered = decode_arithmetic_coefficients(recovered_stream)
    np.testing.assert_array_equal(recovered, raw)


def test_ac_deserialize_rejects_wrong_magic() -> None:
    """AC deserialize refuses a blob with the wrong magic bytes."""
    bad = b"XXXX" + b"\x00" * 30
    with pytest.raises(SparsePacketIRError, match="magic"):
        deserialize_arithmetic_coefficients(bad)


def test_ac_deserialize_rejects_truncated_blob() -> None:
    """AC deserialize refuses a truncated blob."""
    raw = np.arange(100, dtype=np.int32) - 50
    blob = serialize_arithmetic_coefficients(encode_arithmetic_coefficients(raw))
    with pytest.raises(SparsePacketIRError, match="truncated|length mismatch"):
        deserialize_arithmetic_coefficients(blob[:-4])


def test_ac_alphabet_size_too_small_rejected() -> None:
    """encode_arithmetic_coefficients refuses alphabet_size < required."""
    raw = np.arange(-5, 6, dtype=np.int32)
    with pytest.raises(SparsePacketIRError, match="too small"):
        encode_arithmetic_coefficients(raw, alphabet_size=5)


# ── Primitive 3: Temporal-subsampling indicator vector ─────────────────────


def test_temporal_round_trip_k_of_n() -> None:
    """Temporal-subsampled round-trip preserves signal frames + None for skipped."""
    rng = np.random.default_rng(4)
    N = 20
    per_frame = 10
    frames: list[np.ndarray | None] = []
    for i in range(N):
        if i % 3 == 0:
            frames.append(rng.integers(-10, 10, size=per_frame, dtype=np.int8))
        else:
            frames.append(None)
    stream = encode_temporal_subsampled(frames)
    assert stream.N == N
    assert stream.K == sum(1 for f in frames if f is not None)
    recovered = decode_temporal_subsampled(stream, dtype=np.int8)
    assert len(recovered) == N
    for orig, rec in zip(frames, recovered, strict=False):
        if orig is None:
            assert rec is None
        else:
            np.testing.assert_array_equal(rec, orig)


def test_temporal_all_skipped() -> None:
    """Temporal round-trip with all frames skipped (K=0)."""
    frames: list[np.ndarray | None] = [None] * 10
    stream = encode_temporal_subsampled(frames)
    assert stream.K == 0
    assert stream.per_frame_bytes == 0
    assert stream.residuals_packed == b""
    recovered = decode_temporal_subsampled(stream)
    assert all(r is None for r in recovered)


def test_temporal_all_present() -> None:
    """Temporal round-trip with all frames present (K=N)."""
    frames: list[np.ndarray | None] = [
        np.array([i, i + 1, i + 2], dtype=np.int8) for i in range(5)
    ]
    stream = encode_temporal_subsampled(frames)
    assert stream.K == stream.N == 5
    recovered = decode_temporal_subsampled(stream, dtype=np.int8)
    for orig, rec in zip(frames, recovered, strict=False):
        np.testing.assert_array_equal(rec, orig)


def test_temporal_with_frame_shape() -> None:
    """Temporal decode honours frame_shape reshape."""
    frames: list[np.ndarray | None] = [
        np.arange(12, dtype=np.int8).reshape(3, 4),
        None,
        (np.arange(12, dtype=np.int8) + 50).reshape(3, 4),
    ]
    stream = encode_temporal_subsampled(frames)
    recovered = decode_temporal_subsampled(stream, dtype=np.int8, frame_shape=(3, 4))
    assert recovered[0] is not None and recovered[0].shape == (3, 4)
    np.testing.assert_array_equal(recovered[0], frames[0])
    assert recovered[1] is None
    np.testing.assert_array_equal(recovered[2], frames[2])


def test_temporal_non_uniform_size_rejected() -> None:
    """encode_temporal_subsampled refuses non-uniform per-frame sizes."""
    frames: list[np.ndarray | None] = [
        np.zeros(10, dtype=np.int8),
        np.zeros(20, dtype=np.int8),
    ]
    with pytest.raises(SparsePacketIRError, match="non-uniform per_frame_bytes"):
        encode_temporal_subsampled(frames)


def test_temporal_non_uniform_dtype_rejected() -> None:
    """encode_temporal_subsampled refuses non-uniform dtype."""
    frames: list[np.ndarray | None] = [
        np.zeros(10, dtype=np.int8),
        np.zeros(10, dtype=np.int16)[:5],  # 10 bytes total to keep size uniform
    ]
    with pytest.raises(SparsePacketIRError, match="non-uniform dtype"):
        encode_temporal_subsampled(frames)


# ── pad_per_frame_to_uniform_size_with_length_prefix + pad_to_uniform_size ─


def test_pad_per_frame_uniform_size_basic() -> None:
    """Variable-length payloads get zero-padded to max(len) with a <I prefix."""
    import struct as _s
    payloads: list[bytes | None] = [b"abc", None, b"abcdef", b"a"]
    framed = pad_per_frame_to_uniform_size_with_length_prefix(payloads)
    assert len(framed) == 4
    # Skipped frames stay None.
    assert framed[1] is None
    # Non-None frames are uint8 ndarrays of identical size = 4 + max_payload_len.
    max_len = 6  # max(len("abc"), len("abcdef"), len("a")) == 6
    expected_size = 4 + max_len
    for i in (0, 2, 3):
        assert framed[i] is not None
        assert framed[i].dtype == np.uint8
        assert framed[i].nbytes == expected_size
    # Length prefixes recover the original sizes.
    for i, expected_len in [(0, 3), (2, 6), (3, 1)]:
        prefix = framed[i].tobytes()[:4]
        (recovered_len,) = _s.unpack("<I", prefix)
        assert recovered_len == expected_len
    # Payload bytes immediately after the prefix are the original payload.
    assert framed[0].tobytes()[4 : 4 + 3] == b"abc"
    assert framed[2].tobytes()[4 : 4 + 6] == b"abcdef"
    assert framed[3].tobytes()[4 : 4 + 1] == b"a"
    # Padding bytes are pure zeros.
    assert framed[0].tobytes()[4 + 3 :] == b"\x00\x00\x00"
    assert framed[3].tobytes()[4 + 1 :] == b"\x00\x00\x00\x00\x00"


def test_pad_per_frame_uniform_size_all_none_returns_none_list() -> None:
    """All-None input returns identical-length all-None list (no error)."""
    framed = pad_per_frame_to_uniform_size_with_length_prefix([None, None, None])
    assert framed == [None, None, None]


def test_pad_per_frame_uniform_size_empty_input() -> None:
    """Empty input returns empty list."""
    framed = pad_per_frame_to_uniform_size_with_length_prefix([])
    assert framed == []


def test_pad_per_frame_uniform_size_single_frame() -> None:
    """Single non-None frame: padded to len(payload) (max == len)."""
    framed = pad_per_frame_to_uniform_size_with_length_prefix([b"hello"])
    assert len(framed) == 1
    assert framed[0] is not None
    assert framed[0].nbytes == 4 + 5  # prefix + payload
    assert framed[0].tobytes()[4:] == b"hello"


def test_pad_per_frame_uniform_size_composes_with_encode_temporal_subsampled() -> None:
    """The helper output is a valid input to encode_temporal_subsampled."""
    payloads: list[bytes | None] = [b"aa", None, b"bbbb", b"c"]
    framed = pad_per_frame_to_uniform_size_with_length_prefix(payloads)
    stream = encode_temporal_subsampled(framed)
    blob = serialize_temporal_subsampled(stream)
    recovered_stream = deserialize_temporal_subsampled(blob)
    decoded = decode_temporal_subsampled(recovered_stream, dtype=np.uint8)
    # Recover original payloads via the <I prefix.
    import struct as _s
    for i, original in enumerate(payloads):
        if original is None:
            assert decoded[i] is None
            continue
        raw_bytes = decoded[i].tobytes()
        (payload_len,) = _s.unpack_from("<I", raw_bytes, 0)
        assert payload_len == len(original)
        assert raw_bytes[4 : 4 + payload_len] == original


def test_encode_temporal_subsampled_pad_to_uniform_size_true() -> None:
    """pad_to_uniform_size=True flag accepts variable-size uint8 ndarrays."""
    import struct as _s
    rng = np.random.default_rng(7)
    frames: list[np.ndarray | None] = [
        rng.integers(0, 256, size=10, dtype=np.uint8),
        None,
        rng.integers(0, 256, size=20, dtype=np.uint8),
        rng.integers(0, 256, size=5, dtype=np.uint8),
    ]
    stream = encode_temporal_subsampled(frames, pad_to_uniform_size=True)
    # K = 3 non-None entries; per_frame_bytes = 4 (prefix) + max_payload_len = 4 + 20 = 24.
    assert stream.K == 3
    assert stream.N == 4
    assert stream.per_frame_bytes == 24
    decoded = decode_temporal_subsampled(stream, dtype=np.uint8)
    for i, orig in enumerate(frames):
        if orig is None:
            assert decoded[i] is None
            continue
        raw = decoded[i].tobytes()
        (payload_len,) = _s.unpack_from("<I", raw, 0)
        assert payload_len == orig.nbytes
        np.testing.assert_array_equal(
            np.frombuffer(raw, dtype=np.uint8, count=payload_len, offset=4),
            orig,
        )


def test_encode_temporal_subsampled_pad_to_uniform_size_requires_uint8() -> None:
    """pad_to_uniform_size=True refuses non-uint8 arrays (caller must cast)."""
    frames: list[np.ndarray | None] = [
        np.array([1, 2, 3], dtype=np.int8),
    ]
    with pytest.raises(SparsePacketIRError, match="requires uint8"):
        encode_temporal_subsampled(frames, pad_to_uniform_size=True)


def test_encode_temporal_subsampled_pad_to_uniform_size_default_false_back_compat() -> None:
    """Default pad_to_uniform_size=False preserves strict uniform-size contract."""
    # Pre-existing variable-size int8 input still raises.
    frames: list[np.ndarray | None] = [
        np.zeros(10, dtype=np.int8),
        np.zeros(20, dtype=np.int8),
    ]
    with pytest.raises(SparsePacketIRError, match="non-uniform per_frame_bytes"):
        encode_temporal_subsampled(frames)  # default False
    # Same input WITH pad_to_uniform_size=True (after cast to uint8) succeeds.
    framed = [f.view(np.uint8) for f in frames]
    stream = encode_temporal_subsampled(framed, pad_to_uniform_size=True)
    assert stream.K == 2
    assert stream.per_frame_bytes == 4 + 20


def test_encode_temporal_subsampled_pad_to_uniform_size_with_skipped_frames() -> None:
    """pad_to_uniform_size=True correctly preserves None entries."""
    frames: list[np.ndarray | None] = [
        np.array([1, 2], dtype=np.uint8),
        None,
        None,
        np.array([3, 4, 5, 6, 7], dtype=np.uint8),
    ]
    stream = encode_temporal_subsampled(frames, pad_to_uniform_size=True)
    assert stream.N == 4
    assert stream.K == 2
    decoded = decode_temporal_subsampled(stream, dtype=np.uint8)
    assert decoded[1] is None
    assert decoded[2] is None


def test_temporal_dataclass_rejects_k_gt_n() -> None:
    """TemporalSubsampledResidualStream constructor refuses K > N."""
    with pytest.raises(SparsePacketIRError, match="K must be in"):
        TemporalSubsampledResidualStream(
            indicator_bitmap=b"\x01",
            residuals_packed=b"",
            N=2,
            K=3,
            per_frame_bytes=0,
        )


def test_temporal_dataclass_rejects_bitmap_popcount_mismatch() -> None:
    """TemporalSubsampledResidualStream rejects bitmap popcount != K."""
    with pytest.raises(SparsePacketIRError, match="popcount"):
        TemporalSubsampledResidualStream(
            indicator_bitmap=b"\x00",
            residuals_packed=b"",
            N=4,
            K=1,
            per_frame_bytes=0,
        )


def test_temporal_dataclass_rejects_bitmap_size_mismatch() -> None:
    """TemporalSubsampledResidualStream rejects bitmap of wrong length."""
    with pytest.raises(SparsePacketIRError, match="indicator_bitmap len"):
        TemporalSubsampledResidualStream(
            indicator_bitmap=b"\x00\x00\x00",  # too long
            N=4,
            K=0,
            per_frame_bytes=0,
            residuals_packed=b"",
        )


def test_temporal_dataclass_rejects_nonzero_padding_bits() -> None:
    """TemporalSubsampledResidualStream refuses set bits beyond frame count N."""
    with pytest.raises(SparsePacketIRError, match="padding bits"):
        TemporalSubsampledResidualStream(
            indicator_bitmap=b"\x80",
            residuals_packed=b"\x01",
            N=1,
            K=1,
            per_frame_bytes=1,
        )


def test_temporal_serialize_round_trip() -> None:
    """Temporal serialize → deserialize → decode is bit-exact."""
    rng = np.random.default_rng(5)
    N = 30
    per_frame = 8
    frames: list[np.ndarray | None] = []
    for i in range(N):
        if i % 4 == 0:
            frames.append(rng.integers(-50, 50, size=per_frame, dtype=np.int8))
        else:
            frames.append(None)
    stream = encode_temporal_subsampled(frames)
    blob = serialize_temporal_subsampled(stream)
    recovered_stream = deserialize_temporal_subsampled(blob)
    assert recovered_stream.N == stream.N
    assert recovered_stream.K == stream.K
    assert recovered_stream.per_frame_bytes == stream.per_frame_bytes
    recovered = decode_temporal_subsampled(recovered_stream, dtype=np.int8)
    for orig, rec in zip(frames, recovered, strict=False):
        if orig is None:
            assert rec is None
        else:
            np.testing.assert_array_equal(rec, orig)


def test_temporal_deserialize_rejects_wrong_magic() -> None:
    """Temporal deserialize refuses a blob with the wrong magic bytes."""
    bad = b"XXXX" + b"\x00" * 20
    with pytest.raises(SparsePacketIRError, match="magic"):
        deserialize_temporal_subsampled(bad)


def test_temporal_deserialize_rejects_truncated_blob() -> None:
    """Temporal deserialize refuses a truncated blob."""
    frames: list[np.ndarray | None] = [
        np.array([1, 2, 3, 4], dtype=np.int8),
        None,
        np.array([5, 6, 7, 8], dtype=np.int8),
    ]
    blob = serialize_temporal_subsampled(encode_temporal_subsampled(frames))
    with pytest.raises(SparsePacketIRError, match="length mismatch|truncated"):
        deserialize_temporal_subsampled(blob[:-1])


def test_temporal_byte_savings_vs_dense() -> None:
    """Temporal-subsampled byte cost < dense per-frame storage for K << N."""
    N = 100
    per_frame = 100
    frames: list[np.ndarray | None] = []
    for i in range(N):
        if i < 5:  # only 5/100 frames carry signal
            frames.append(np.ones(per_frame, dtype=np.int8))
        else:
            frames.append(None)
    stream = encode_temporal_subsampled(frames)
    blob = serialize_temporal_subsampled(stream)
    # Dense would be N * per_frame = 10000 bytes
    # Sparse: 16 header + 13 bitmap + 5*100 = 529 bytes
    assert len(blob) < N * per_frame


# ── Composition: orthogonal primitives compose into a wire format ──────────


def test_composition_rle_then_ac() -> None:
    """RLE-of-zeros → AC on the non-zero values composes correctly."""
    rng = np.random.default_rng(6)
    dense = np.zeros(2000, dtype=np.int8)
    n_nonzero = 200
    positions = rng.choice(2000, size=n_nonzero, replace=False)
    values = rng.integers(-5, 6, size=n_nonzero, dtype=np.int8)
    # Avoid zeros in the nonzero values
    values[values == 0] = 1
    dense[positions] = values
    rle = encode_rle_of_zeros(dense)
    ac = encode_arithmetic_coefficients(rle.nonzero_values.astype(np.int32))
    recovered_values_int32 = decode_arithmetic_coefficients(ac)
    # Reconstruct via RLE with recovered values
    composed = RleOfZerosStream(
        nonzero_indices=rle.nonzero_indices,
        nonzero_values=recovered_values_int32.astype(np.int8),
        total_length=rle.total_length,
    )
    recovered = decode_rle_of_zeros(composed)
    np.testing.assert_array_equal(recovered, dense)


def test_composition_temporal_then_rle() -> None:
    """Temporal subsampling → RLE on each kept frame composes correctly."""
    rng = np.random.default_rng(7)
    N = 10
    per_frame = 100
    frames: list[np.ndarray | None] = []
    for i in range(N):
        if i % 3 == 0:
            arr = np.zeros(per_frame, dtype=np.int8)
            n_nz = rng.integers(5, 20)
            positions = rng.choice(per_frame, size=int(n_nz), replace=False)
            vals = rng.integers(1, 10, size=int(n_nz), dtype=np.int8)
            arr[positions] = vals
            frames.append(arr)
        else:
            frames.append(None)
    temporal = encode_temporal_subsampled(frames)
    # Apply RLE to each kept frame and verify round-trip
    kept_indices = [i for i, f in enumerate(frames) if f is not None]
    recovered_frames = decode_temporal_subsampled(temporal, dtype=np.int8)
    for idx in kept_indices:
        rle = encode_rle_of_zeros(recovered_frames[idx])
        re_dense = decode_rle_of_zeros(rle)
        np.testing.assert_array_equal(re_dense, frames[idx])


# ── 8 archive-grammar fields declared (HNeRV parity discipline lesson 3) ───


def test_archive_grammar_fields_declared() -> None:
    """The module docstring + golden vectors declare the 8 HNeRV-parity fields."""
    import tac.packet_compiler.sparse_packet_ir as m

    doc = m.__doc__ or ""
    # Per CLAUDE.md HNeRV parity discipline lesson 3 — 8 fields must appear.
    required_tokens = [
        "archive_grammar",
        "parser_section_manifest",
        "inflate_runtime_loc_budget",
        "runtime_dep_closure",
        "export_format",
        "score_aware_loss",
        "bolt_on_loc_budget",
        "no_op_detector_planned",
    ]
    for token in required_tokens:
        assert token in doc, f"missing archive-grammar field {token!r}"


# ── Golden vector parity (committed) ────────────────────────────────────────

GOLDEN_VECTORS_DIR = Path(__file__).resolve().parents[1] / "packet_compiler/golden_vectors"


def _gv_path(name: str) -> Path:
    return GOLDEN_VECTORS_DIR / f"{name}.json"


@pytest.mark.skipif(
    not _gv_path("sparse_rle_of_zeros_v1").exists(),
    reason="golden vector not yet committed",
)
def test_golden_vector_sparse_rle() -> None:
    """RLE golden vector SHA-256 matches the committed manifest."""
    manifest = json.loads(_gv_path("sparse_rle_of_zeros_v1").read_text())
    rng = np.random.default_rng(20260511)
    dense = np.zeros(1024, dtype=np.int8)
    positions = rng.choice(1024, size=64, replace=False)
    values = rng.integers(1, 32, size=64, dtype=np.int8)
    dense[positions] = values
    stream = encode_rle_of_zeros(dense)
    blob = serialize_rle_of_zeros(stream)
    sha = hashlib.sha256(blob).hexdigest()
    assert sha == manifest["sha256"], (
        f"RLE golden vector SHA mismatch: produced {sha} expected {manifest['sha256']}"
    )
    assert len(blob) == manifest["payload_len"]


@pytest.mark.skipif(
    not _gv_path("sparse_arithmetic_coefficients_v1").exists(),
    reason="golden vector not yet committed",
)
def test_golden_vector_sparse_arithmetic() -> None:
    """AC golden vector SHA-256 matches the committed manifest."""
    manifest = json.loads(_gv_path("sparse_arithmetic_coefficients_v1").read_text())
    rng = np.random.default_rng(20260511)
    raw = rng.integers(-8, 9, size=500, dtype=np.int32)
    stream = encode_arithmetic_coefficients(raw)
    blob = serialize_arithmetic_coefficients(stream)
    sha = hashlib.sha256(blob).hexdigest()
    assert sha == manifest["sha256"], (
        f"AC golden vector SHA mismatch: produced {sha} expected {manifest['sha256']}"
    )


@pytest.mark.skipif(
    not _gv_path("sparse_temporal_subsampled_v1").exists(),
    reason="golden vector not yet committed",
)
def test_golden_vector_sparse_temporal() -> None:
    """Temporal-subsampled golden vector SHA-256 matches the committed manifest."""
    manifest = json.loads(_gv_path("sparse_temporal_subsampled_v1").read_text())
    rng = np.random.default_rng(20260511)
    N = 50
    per_frame = 20
    frames: list[np.ndarray | None] = []
    for i in range(N):
        if i % 5 == 0:
            frames.append(rng.integers(-10, 11, size=per_frame, dtype=np.int8))
        else:
            frames.append(None)
    stream = encode_temporal_subsampled(frames)
    blob = serialize_temporal_subsampled(stream)
    sha = hashlib.sha256(blob).hexdigest()
    assert sha == manifest["sha256"], (
        f"Temporal golden vector SHA mismatch: produced {sha} expected {manifest['sha256']}"
    )


# ── Phase1 packet compiler transforms wired ────────────────────────────────


def test_phase1_packet_compiler_transforms_includes_sparse_tokens() -> None:
    """The 3 sparse transform tokens are registered in PACKET_COMPILER_TRANSFORMS."""
    from tac.phase1_packet_compiler import PACKET_COMPILER_TRANSFORMS

    assert "sparse_rle_of_zeros" in PACKET_COMPILER_TRANSFORMS
    assert "sparse_arithmetic_coefficients" in PACKET_COMPILER_TRANSFORMS
    assert "sparse_temporal_subsampled" in PACKET_COMPILER_TRANSFORMS


# ── PR106 sidecar packing sparse-format-id registry wired ──────────────────


def test_pr106_format_ids_include_sparse_siblings() -> None:
    """All 5 dense families have a 0x2N sparse sibling in the format_id registry."""
    from tac.residual_basis.pr106_sidecar_packing import (
        PR106_RESIDUAL_FORMAT_IDS,
        sparse_family_name,
    )

    for fam, dense_id in [
        ("wavelet", 0x10),
        ("cool_chic", 0x11),
        ("c3", 0x12),
        ("siren", 0x13),
        ("coord_mlp", 0x14),
    ]:
        sparse = sparse_family_name(fam)
        assert PR106_RESIDUAL_FORMAT_IDS[fam] == dense_id
        # Sparse format_id = dense + 0x10 (0x1N -> 0x2N).
        assert PR106_RESIDUAL_FORMAT_IDS[sparse] == dense_id + 0x10


def test_sparse_family_name_refuses_unknown() -> None:
    """sparse_family_name raises if the dense family has no sparse sibling."""
    from tac.residual_basis.pr106_sidecar_packing import (
        ResidualArchiveError,
        sparse_family_name,
    )

    with pytest.raises(ResidualArchiveError, match="no sparse sibling"):
        sparse_family_name("nonexistent_family")


# ── Materializer repack_dense_as_sparse round-trip (wavelet + inline decoder) ─


def test_repack_dense_as_sparse_wavelet_round_trip() -> None:
    """Wavelet dense → sparse repack + inline decode reproduces original bytes."""
    from tac.residual_basis.pr106_materializer_helpers import repack_dense_as_sparse
    import struct as _struct
    import sys as _sys

    inline_path = (
        Path(__file__).resolve().parents[3]
        / "submissions/pr106_wavelet_residual_sidecar/src/sparse_packet_ir_inline.py"
    )
    spec = importlib.util.spec_from_file_location("_inline_wavelet", inline_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    decode_temporal = mod.decode_temporal_subsampled_bytes
    decode_rle = mod.decode_rle_of_zeros_bytes

    n_frames = 4
    camera_h, camera_w, rgb = 874, 1164, 3
    half_h, half_w = camera_h // 2, camera_w // 2
    band_size = half_h * half_w
    per_frame_bytes = 16 + 4 * rgb * band_size
    rng = np.random.default_rng(42)
    parts: list[bytes] = []
    for t in range(n_frames):
        parts.append(_struct.pack("<4f", 0.1, 0.05, 0.05, 0.02))
        # Mostly-zero band coefs.
        band = np.zeros(4 * rgb * band_size, dtype=np.int8)
        positions = rng.choice(band.size, size=200, replace=False)
        band[positions] = rng.integers(1, 20, size=200, dtype=np.int8)
        parts.append(band.tobytes())
    dense_bytes = b"".join(parts)
    assert len(dense_bytes) == n_frames * per_frame_bytes

    sparse_bytes = repack_dense_as_sparse(
        family="wavelet", dense_residual_bytes=dense_bytes, n_frames=n_frames
    )
    # Sparse should be DRAMATICALLY smaller for sparse coefficients.
    assert len(sparse_bytes) < len(dense_bytes) // 4, (
        f"sparse {len(sparse_bytes)} should be << dense {len(dense_bytes)}"
    )

    # Decode the temporal-subsampled wrapper. Per the Sparse PacketIR
    # uniform-per-frame contract (see
    # `pad_per_frame_to_uniform_size_with_length_prefix`), each per-frame raw
    # payload begins with a 4-byte LE u32 length prefix capturing the actual
    # payload length, followed by the payload itself, followed by zero
    # padding to the longest-frame size. The decoder reads the <I prefix to
    # recover the actual payload length and slices off the trailing zeros.
    # This matches the family inflate.py path (e.g. wavelet inflate.py:99-102).
    import struct as _struct
    per_frame_raw = decode_temporal(sparse_bytes, dtype=np.uint8)
    assert len(per_frame_raw) == n_frames
    for t, raw in enumerate(per_frame_raw):
        assert raw is not None
        raw_bytes = raw.tobytes()
        # Recover the length-prefixed payload (the family inflate.py mirror).
        (payload_len,) = _struct.unpack_from("<I", raw_bytes, 0)
        assert 4 + payload_len <= len(raw_bytes), (
            f"frame {t}: declared payload_len={payload_len} > raw_bytes_len-4={len(raw_bytes)-4}"
        )
        payload = raw_bytes[4 : 4 + payload_len]
        # First 16 bytes of the payload are the scales.
        assert payload[:16] == dense_bytes[t * per_frame_bytes : t * per_frame_bytes + 16]
        # Remainder is RLE-encoded; decode to recover the int8 band coeffs.
        flat = decode_rle(payload[16:])
        original_bands = np.frombuffer(
            dense_bytes,
            dtype=np.int8,
            count=4 * rgb * band_size,
            offset=t * per_frame_bytes + 16,
        )
        np.testing.assert_array_equal(flat, original_bands)


def test_repack_dense_as_sparse_refuses_unknown_family() -> None:
    """repack_dense_as_sparse raises MaterializerError on unknown family."""
    from tac.residual_basis.pr106_materializer_helpers import (
        MaterializerError,
        repack_dense_as_sparse,
    )

    with pytest.raises(MaterializerError, match="sparse repack not implemented"):
        repack_dense_as_sparse(
            family="not_a_real_family",
            dense_residual_bytes=b"\x00\x01\x02",
            n_frames=1,
        )


def test_repack_dense_as_sparse_empty_short_circuits() -> None:
    """repack_dense_as_sparse returns empty bytes for empty input (any family)."""
    from tac.residual_basis.pr106_materializer_helpers import repack_dense_as_sparse

    for family in ["wavelet", "c3", "cool_chic", "siren", "coord_mlp"]:
        assert repack_dense_as_sparse(
            family=family, dense_residual_bytes=b"", n_frames=10
        ) == b""


# ── Inflate-inline decoder parity ───────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
INLINE_DECODER_PATHS = tuple(
    sorted(
        (REPO_ROOT / "submissions").glob(
            "pr106_*_residual_sidecar/src/sparse_packet_ir_inline.py"
        )
    )
)


def _load_inline_decoder(path: Path):
    spec = importlib.util.spec_from_file_location(f"inline_{path.parent.parent.name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not INLINE_DECODER_PATHS, reason="inline decoders not present")
def test_inline_decoders_match_python_oracle_for_rle_and_temporal() -> None:
    """Each submission inline decoder matches canonical sparse PacketIR bytes."""
    dense = np.zeros(128, dtype=np.int8)
    dense[[3, 19, 97]] = np.array([4, -7, 11], dtype=np.int8)
    rle_blob = serialize_rle_of_zeros(encode_rle_of_zeros(dense))

    frames: list[np.ndarray | None] = [
        np.arange(8, dtype=np.int8),
        None,
        (np.arange(8, dtype=np.int8) - 4),
        None,
        None,
    ]
    temporal_blob = serialize_temporal_subsampled(encode_temporal_subsampled(frames))

    for path in INLINE_DECODER_PATHS:
        module = _load_inline_decoder(path)
        np.testing.assert_array_equal(module.decode_rle_of_zeros_bytes(rle_blob), dense)
        recovered = module.decode_temporal_subsampled_bytes(
            temporal_blob,
            dtype=np.int8,
        )
        for expected, actual in zip(frames, recovered, strict=False):
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not INLINE_DECODER_PATHS, reason="inline decoders not present")
def test_inline_decoders_reject_noncanonical_rle_and_temporal_padding() -> None:
    """Inline decoders fail closed on malformed sparse PacketIR byte streams."""
    duplicate_rle = (
        struct.pack("<4sIIB", b"SRL1", 8, 2, 0)
        + np.array([3, 3], dtype="<u4").tobytes()
        + np.array([1, 2], dtype=np.int8).tobytes()
    )
    padding_temporal = struct.pack("<4sIII", b"STS1", 1, 1, 1) + b"\x80" + b"\x01"

    for path in INLINE_DECODER_PATHS:
        module = _load_inline_decoder(path)
        with pytest.raises(module.SparseDecodeError, match="strictly increasing"):
            module.decode_rle_of_zeros_bytes(duplicate_rle)
        with pytest.raises(module.SparseDecodeError, match="padding bits"):
            module.decode_temporal_subsampled_bytes(padding_temporal)
