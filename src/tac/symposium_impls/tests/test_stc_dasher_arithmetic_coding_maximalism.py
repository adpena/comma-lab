# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism`."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism import (
    DEFAULT_STC_CONSTRAINT_LENGTH,
    DasherContextModel,
    STCDasherSymbolStream,
    STCParityCheckMatrix,
    arithmetic_code_bit_estimate,
    build_default_stc_parity_matrix,
    compose_stc_dasher_encoded_bits,
    estimate_stc_dasher_rate_bound,
    stc_encode_to_syndrome,
    update_from_anchor,
)


# ----- parity matrix construction --------------------------------------------------------------


def test_default_constraint_length_matches_filler_canonical() -> None:
    assert DEFAULT_STC_CONSTRAINT_LENGTH == 12


def test_build_parity_returns_correct_shape() -> None:
    parity = build_default_stc_parity_matrix(n_cover=64, payload_bits=8)
    assert parity.matrix.shape == (8, 64)
    assert parity.n_cover == 64
    assert parity.n_syndrome == 8


def test_build_parity_invalid_dims_raise() -> None:
    with pytest.raises(ValueError):
        build_default_stc_parity_matrix(n_cover=0, payload_bits=4)
    with pytest.raises(ValueError):
        build_default_stc_parity_matrix(n_cover=4, payload_bits=0)
    with pytest.raises(ValueError):
        build_default_stc_parity_matrix(n_cover=4, payload_bits=8)
    with pytest.raises(ValueError):
        build_default_stc_parity_matrix(n_cover=8, payload_bits=2, constraint_length=0)


def test_build_parity_is_binary() -> None:
    parity = build_default_stc_parity_matrix(n_cover=32, payload_bits=4)
    assert ((parity.matrix == 0) | (parity.matrix == 1)).all()


def test_parity_matrix_validates_binary() -> None:
    bad = np.array([[0, 2], [1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError):
        STCParityCheckMatrix(matrix=bad, constraint_length=2)


def test_parity_matrix_validates_dim() -> None:
    bad = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        STCParityCheckMatrix(matrix=bad, constraint_length=2)


def test_parity_matrix_validates_dtype() -> None:
    bad = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        STCParityCheckMatrix(matrix=bad, constraint_length=2)


def test_parity_matrix_validates_constraint_length() -> None:
    matrix = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        STCParityCheckMatrix(matrix=matrix, constraint_length=0)


# ----- STC encode tests -----------------------------------------------------------------------


def test_stc_encode_matches_matrix_multiplication() -> None:
    """Per Filler 2011 §III: s = H @ x mod 2."""
    matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    parity = STCParityCheckMatrix(matrix=matrix, constraint_length=2)
    x = np.array([1, 1, 0, 1], dtype=np.uint8)
    result = stc_encode_to_syndrome(symbols=x, parity=parity)
    expected = (matrix @ x) % 2
    assert np.array_equal(result.syndrome, expected)


def test_stc_encode_zero_input_yields_zero_syndrome() -> None:
    parity = build_default_stc_parity_matrix(n_cover=16, payload_bits=4)
    x = np.zeros(16, dtype=np.uint8)
    result = stc_encode_to_syndrome(symbols=x, parity=parity)
    assert (result.syndrome == 0).all()


def test_stc_encode_invalid_dim_raises() -> None:
    parity = build_default_stc_parity_matrix(n_cover=4, payload_bits=2)
    with pytest.raises(ValueError):
        stc_encode_to_syndrome(symbols=np.zeros((2, 2), dtype=np.uint8), parity=parity)


def test_stc_encode_size_mismatch_raises() -> None:
    parity = build_default_stc_parity_matrix(n_cover=4, payload_bits=2)
    with pytest.raises(ValueError):
        stc_encode_to_syndrome(symbols=np.zeros(8, dtype=np.uint8), parity=parity)


def test_stc_encode_non_binary_raises() -> None:
    parity = build_default_stc_parity_matrix(n_cover=4, payload_bits=2)
    with pytest.raises(ValueError):
        stc_encode_to_syndrome(symbols=np.array([0, 1, 2, 0], dtype=np.uint8), parity=parity)


def test_stc_encode_rate_matches_payload_over_cover() -> None:
    parity = build_default_stc_parity_matrix(n_cover=64, payload_bits=8)
    x = np.zeros(64, dtype=np.uint8)
    result = stc_encode_to_syndrome(symbols=x, parity=parity)
    assert result.rate_bits_per_symbol == pytest.approx(8 / 64)


# ----- Dasher arithmetic coding tests ---------------------------------------------------------


def test_dasher_model_validation() -> None:
    DasherContextModel(context_length=2, symbol_alphabet_size=2)


def test_dasher_model_invalid_alphabet_raises() -> None:
    with pytest.raises(ValueError):
        DasherContextModel(context_length=2, symbol_alphabet_size=1)


def test_dasher_model_invalid_context_raises() -> None:
    with pytest.raises(ValueError):
        DasherContextModel(context_length=-1, symbol_alphabet_size=2)


def test_dasher_model_invalid_initial_count_raises() -> None:
    with pytest.raises(ValueError):
        DasherContextModel(context_length=2, symbol_alphabet_size=2, initial_count=0)


def test_arithmetic_code_empty_returns_zero() -> None:
    model = DasherContextModel(context_length=2, symbol_alphabet_size=2)
    assert arithmetic_code_bit_estimate(np.array([], dtype=np.uint8), model=model) == 0.0


def test_arithmetic_code_uniform_binary_close_to_one_bit_per_symbol() -> None:
    """For balanced binary symbols, AC achieves ~1 bit/symbol asymptotically."""
    rng = np.random.default_rng(42)
    symbols = rng.integers(0, 2, size=1000, dtype=np.uint8)
    model = DasherContextModel(context_length=0, symbol_alphabet_size=2)
    bits = arithmetic_code_bit_estimate(symbols, model=model)
    assert bits / symbols.size == pytest.approx(1.0, abs=0.05)


def test_arithmetic_code_constant_symbols_low_bit_cost() -> None:
    """Constant signal compresses to <<1 bit/symbol with adaptive context."""
    symbols = np.zeros(100, dtype=np.uint8)
    model = DasherContextModel(context_length=2, symbol_alphabet_size=2)
    bits = arithmetic_code_bit_estimate(symbols, model=model)
    assert bits / symbols.size < 0.2


def test_arithmetic_code_invalid_dim_raises() -> None:
    model = DasherContextModel(context_length=0, symbol_alphabet_size=2)
    with pytest.raises(ValueError):
        arithmetic_code_bit_estimate(np.zeros((2, 2), dtype=np.uint8), model=model)


def test_arithmetic_code_out_of_alphabet_raises() -> None:
    model = DasherContextModel(context_length=0, symbol_alphabet_size=2)
    with pytest.raises(ValueError):
        arithmetic_code_bit_estimate(np.array([0, 1, 5], dtype=np.uint8), model=model)


# ----- composition end-to-end ----------------------------------------------------------------


def test_compose_stc_dasher_returns_well_formed() -> None:
    parity = build_default_stc_parity_matrix(n_cover=64, payload_bits=8)
    model = DasherContextModel(context_length=2, symbol_alphabet_size=2)
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 2, size=64, dtype=np.uint8)
    stream = compose_stc_dasher_encoded_bits(symbols=symbols, parity=parity, model=model)
    assert isinstance(stream, STCDasherSymbolStream)
    assert stream.n_input_symbols == 64
    assert stream.n_syndrome_bits == 8
    assert stream.arithmetic_bits >= 0
    assert stream.rate_bits_per_input_symbol > 0


def test_compose_stc_dasher_invalid_alphabet_raises() -> None:
    parity = build_default_stc_parity_matrix(n_cover=8, payload_bits=2)
    model = DasherContextModel(context_length=0, symbol_alphabet_size=4)
    with pytest.raises(ValueError):
        compose_stc_dasher_encoded_bits(
            symbols=np.zeros(8, dtype=np.uint8), parity=parity, model=model
        )


def test_estimate_stc_dasher_rate_bound_filler_theorem() -> None:
    """R_STC(D) <= R_AC(D) + 1/h per Filler Theorem 4."""
    bound = estimate_stc_dasher_rate_bound(
        baseline_bits=1.0, syndrome_bits=0.5, constraint_length=12
    )
    # 1.0 + 1/12 - 0.5 = 0.5833...
    assert bound == pytest.approx(1.0 + 1.0 / 12 - 0.5, abs=1e-9)


def test_rate_bound_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        estimate_stc_dasher_rate_bound(baseline_bits=1.0, syndrome_bits=0.5, constraint_length=0)
    with pytest.raises(ValueError):
        estimate_stc_dasher_rate_bound(baseline_bits=-1.0, syndrome_bits=0.5, constraint_length=12)


# ----- continual learning hook -----------------------------------------------------------------


def test_update_from_anchor_no_symbols_returns_none() -> None:
    assert update_from_anchor({}) is None


def test_update_from_anchor_with_symbols_returns_stream() -> None:
    anchor = {"symbols": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)}
    stream = update_from_anchor(anchor)
    assert stream is not None
    assert stream.n_input_symbols == 8
