"""Lane SH arithmetic codec roundtrip + entropy regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from tac.arithmetic_qint_codec import (
    build_freq_table,
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)


def test_ternary_roundtrip_uniform():
    rng = np.random.default_rng(0)
    qints = rng.choice([-1, 0, 1], size=10_000, p=[0.3, 0.4, 0.3]).astype(np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)


def test_ternary_roundtrip_skewed_low_entropy():
    rng = np.random.default_rng(1)
    qints = rng.choice([-1, 0, 1], size=20_000, p=[0.05, 0.9, 0.05]).astype(np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)
    # Shannon bound for this distribution is ~0.57 bits/symbol; arithmetic
    # coder should land within 5% of the bound.
    bits_per_symbol = len(blob) * 8 / len(qints)
    assert bits_per_symbol < 1.0, f"expected < 1.0 bits/symbol, got {bits_per_symbol:.3f}"


def test_septenary_roundtrip():
    rng = np.random.default_rng(2)
    qints = rng.integers(-3, 4, size=5000).astype(np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=7, offset=3)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)


def test_single_symbol_stream_handles_gracefully():
    qints = np.zeros(1000, dtype=np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)


def test_empty_stream_rejected():
    # Empty streams are rejected because numpy.min() has no identity on a
    # zero-size array — defensive guard, the encoder should never see one
    # in practice (a SegMap conv weight always has > 0 elements).
    qints = np.zeros(0, dtype=np.int8)
    with pytest.raises(ValueError):
        encode_qints_arithmetic(qints, num_symbols=3, offset=1)


def test_offset_overflow_rejected():
    qints = np.array([5, 0, -1], dtype=np.int8)
    with pytest.raises(ValueError, match="out of range"):
        encode_qints_arithmetic(qints, num_symbols=3, offset=1)


def test_freq_table_floor_is_one():
    qints = np.array([0, 0, 0, 1, 1], dtype=np.int8) + 1
    freq = build_freq_table(qints, num_symbols=3)
    # All counts must be >= 1 (defensive against zero-prob symbols at decode).
    assert (freq >= 1).all()
    # Counts: symbol 0 not present in input -> still 1. symbol 1 (originally 0)
    # appears 3 times. symbol 2 (originally 1) appears 2 times.
    assert int(freq[1]) == 3
    assert int(freq[2]) == 2


def test_short_stream_roundtrip():
    # Tiny streams stress the bit-flush path of the arithmetic coder.
    for n in [1, 2, 3, 7, 15, 16, 17]:
        rng = np.random.default_rng(n)
        qints = rng.choice([-1, 0, 1], size=n).astype(np.int8)
        blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
        recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
        assert np.array_equal(qints, recovered), f"failed at n={n}"
