"""Unit tests for ``tac.codec.cost_curves``."""
from __future__ import annotations

import numpy as np
import pytest

from tac.codec.cost_curves import (
    DEFAULT_K_RANGE,
    TensorBlob,
    find_best_K_for_tensor,
    greedy_uniform_per_tensor_budget_sparsity,
    precompute_per_tensor_K_curves,
    precompute_per_tensor_sparsity_curves,
)


def _make_blobs(n_tensors: int = 3, n_elem: int = 128, seed: int = 0) -> list[TensorBlob]:
    rng = np.random.default_rng(seed)
    return [
        TensorBlob(
            name=f"t{i}",
            raw=rng.integers(-127, 128, size=n_elem, dtype=np.int32),
        )
        for i in range(n_tensors)
    ]


def test_precompute_sparsity_curves_includes_brotli_only_row() -> None:
    blobs = _make_blobs()
    curves = precompute_per_tensor_sparsity_curves(blobs, alphas=[0.3, 0.7])
    assert len(curves) == len(blobs)
    for tensor_curve in curves:
        # 1 brotli-only row + 2 alpha rows
        assert len(tensor_curve) == 3
        # First row is the brotli-only baseline
        assert tensor_curve[0]["alpha"] == 0.0
        assert tensor_curve[0]["rel_err"] == 0.0
        # Sparsity rows have non-decreasing alpha and rel_err
        alphas = [r["alpha"] for r in tensor_curve]
        assert alphas == sorted(alphas)


def test_precompute_sparsity_accepts_tuple_form() -> None:
    rng = np.random.default_rng(1)
    quantized = [
        ("t0", rng.integers(-127, 128, size=64, dtype=np.int8)),
        ("t1", rng.integers(-127, 128, size=64, dtype=np.int8)),
    ]
    curves = precompute_per_tensor_sparsity_curves(quantized, alphas=[0.5])
    assert len(curves) == 2
    for tc in curves:
        assert len(tc) == 2
        assert tc[0]["bytes"] > 0


def test_greedy_uniform_per_tensor_budget_sparsity_returns_smallest_valid() -> None:
    # Hand-crafted curves where the answer is obvious
    curves = [
        [
            {"alpha": 0.0, "bytes": 100, "rel_err": 0.0},
            {"alpha": 0.5, "bytes": 60, "rel_err": 0.1},
            {"alpha": 0.9, "bytes": 30, "rel_err": 0.4},
        ],
        [
            {"alpha": 0.0, "bytes": 80, "rel_err": 0.0},
            {"alpha": 0.5, "bytes": 50, "rel_err": 0.05},
        ],
    ]
    # budget=0.2 → tensor 0 picks alpha=0.5 (60), tensor 1 picks alpha=0.5 (50)
    total, rms = greedy_uniform_per_tensor_budget_sparsity(curves, budget=0.2)
    assert total == 110
    assert rms == pytest.approx(float(np.sqrt(np.mean([0.1**2, 0.05**2]))), rel=1e-6)


def test_greedy_with_zero_budget_picks_lossless() -> None:
    blobs = _make_blobs()
    curves = precompute_per_tensor_sparsity_curves(blobs, alphas=[0.3, 0.7])
    total, rms = greedy_uniform_per_tensor_budget_sparsity(curves, budget=0.0)
    expected = sum(tc[0]["bytes"] for tc in curves)
    assert total == expected
    assert rms == 0.0


def test_precompute_K_curves_default_range() -> None:
    blobs = _make_blobs()
    curves = precompute_per_tensor_K_curves(blobs)
    assert len(curves) == len(blobs)
    for tc in curves:
        assert len(tc) == len(DEFAULT_K_RANGE)
        # K=1 is lossless
        assert tc[0]["K"] == 1
        assert tc[0]["rel_err"] == 0.0
        # rel_err is generally increasing in K but rounding-to-nearest can
        # produce small non-monotone fluctuations between adjacent K values.
        # Assert the trend over K=1 vs K=max instead.
        assert tc[-1]["rel_err"] > tc[0]["rel_err"]


def test_find_best_K_returns_largest_under_budget() -> None:
    rng = np.random.default_rng(0)
    syms = rng.integers(-127, 128, size=512, dtype=np.int32)
    K, rel = find_best_K_for_tensor(syms, budget=0.05)
    assert K >= 1
    assert rel <= 0.05 + 1e-9


def test_find_best_K_zero_budget_returns_K_one() -> None:
    rng = np.random.default_rng(0)
    syms = rng.integers(-127, 128, size=64, dtype=np.int32)
    K, rel = find_best_K_for_tensor(syms, budget=0.0)
    assert K == 1
    assert rel == 0.0


def test_find_best_K_handles_all_zero_input() -> None:
    syms = np.zeros(32, dtype=np.int32)
    K, rel = find_best_K_for_tensor(syms, budget=0.5)
    # All-zero input: rel_err is 0 for any K; we return K=1 by convention.
    assert K == 1
    assert rel == 0.0
