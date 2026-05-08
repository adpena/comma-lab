"""Unit tests for ``tac.codec.per_tensor_codecs``.

The byte counts are not asserted exactly (brotli is not byte-stable across
versions), but invariants are: ``encode_brotli_only`` is lossless;
``encode_sparsity_alpha`` agrees on rel_err with the historical formula;
``encode_lossy_K_coarsen`` honors the K-step rounding contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.per_tensor_codecs import (
    BROTLI_PARAMS,
    encode_brotli_only,
    encode_lossy_K_coarsen,
    encode_sparsity_alpha,
)


def _rand_int8(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-127, 128, size=n, dtype=np.int8)


def test_brotli_params_are_pinned() -> None:
    assert BROTLI_PARAMS == {"quality": 11, "lgwin": 16, "lgblock": 19}


def test_encode_brotli_only_lossless() -> None:
    syms = _rand_int8(1024)
    bytes_used, rel_err = encode_brotli_only(syms)
    assert bytes_used > 0
    assert rel_err == 0.0


def test_encode_brotli_only_smaller_for_low_entropy_input() -> None:
    syms = np.zeros(1024, dtype=np.int8)
    bytes_zeros, _ = encode_brotli_only(syms)
    bytes_random, _ = encode_brotli_only(_rand_int8(1024))
    assert bytes_zeros < bytes_random


def test_encode_sparsity_alpha_zero_falls_through_to_lossless() -> None:
    syms = _rand_int8(512, seed=7)
    b0, e0 = encode_brotli_only(syms)
    b1, e1 = encode_sparsity_alpha(syms, alpha=0.0)
    assert b0 == b1
    assert e0 == e1 == 0.0


def test_encode_sparsity_alpha_one_reduces_to_one_kept_value() -> None:
    syms = _rand_int8(64, seed=3)
    # alpha=0.99 keeps at least 1 value
    bytes_used, rel_err = encode_sparsity_alpha(syms, alpha=0.99)
    assert bytes_used > 0
    assert rel_err >= 0.0


def test_encode_sparsity_alpha_rel_err_monotone_in_alpha() -> None:
    syms = _rand_int8(512, seed=42)
    _, e_low = encode_sparsity_alpha(syms, alpha=0.1)
    _, e_high = encode_sparsity_alpha(syms, alpha=0.7)
    assert e_high >= e_low  # more aggressive sparsification → larger error


def test_encode_sparsity_alpha_known_recon() -> None:
    # Five values; drop the smallest two.
    syms = np.array([1, -2, 3, -4, 5], dtype=np.int8)
    # alpha=0.4 → keep 3 of 5 (the three largest by magnitude: 3, -4, 5)
    bytes_used, rel_err = encode_sparsity_alpha(syms, alpha=0.4)
    assert bytes_used > 0
    # Recon: [0, 0, 3, -4, 5] vs orig [1, -2, 3, -4, 5]; diff = [-1, 2, 0, 0, 0]
    # rel_err = sqrt(1+4) / sqrt(1+4+9+16+25) = sqrt(5) / sqrt(55)
    expected = float(np.sqrt(5.0) / np.sqrt(55.0))
    assert rel_err == pytest.approx(expected, rel=1e-6)


def test_encode_lossy_K_coarsen_K_eq_1_lossless() -> None:
    syms = _rand_int8(256, seed=11).astype(np.int32)
    bytes_used, rel_err = encode_lossy_K_coarsen(syms, K=1)
    assert bytes_used > 0
    assert rel_err == 0.0


def test_encode_lossy_K_coarsen_rel_err_monotone_in_K() -> None:
    syms = _rand_int8(256, seed=99).astype(np.int32)
    _, e_K2 = encode_lossy_K_coarsen(syms, K=2)
    _, e_K8 = encode_lossy_K_coarsen(syms, K=8)
    assert e_K8 >= e_K2


def test_encode_lossy_K_coarsen_rejects_K_below_one() -> None:
    syms = _rand_int8(64).astype(np.int32)
    with pytest.raises(ValueError):
        encode_lossy_K_coarsen(syms, K=0)
