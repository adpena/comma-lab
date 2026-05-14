# SPDX-License-Identifier: MIT
"""Unit tests for the sensitivity-aware quantization allocator (no state_dict load)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pr101_sensitivity_aware_quantization import (  # noqa: E402
    SENSITIVITY_PROXY,
    TensorImportance,
    allocate_per_tensor_budgets,
    compute_xavier_l2_importance,
)


def _make_blob(name: str, values: list[float]):
    class _B:
        pass

    b = _B()
    b.name = name
    b.raw = np.array(values, dtype=np.int32)
    return b


def test_xavier_l2_importance_rms_is_zero_for_zero_tensor() -> None:
    blobs = [_make_blob("zero", [0, 0, 0, 0])]
    out = compute_xavier_l2_importance(blobs)
    assert out[0].importance == 0.0
    assert out[0].rms == 0.0
    assert out[0].numel == 4


def test_xavier_l2_importance_higher_amplitude_higher_score() -> None:
    blobs = [
        _make_blob("low", [1, -1, 1, -1]),
        _make_blob("high", [10, -10, 10, -10]),
    ]
    out = compute_xavier_l2_importance(blobs)
    assert out[0].importance < out[1].importance
    # rms("low") = 1, rms("high") = 10
    assert out[0].importance == pytest.approx(1.0)
    assert out[1].importance == pytest.approx(10.0)


def test_allocate_uniform_when_eta_zero() -> None:
    importances = [
        TensorImportance(name="a", importance=1.0, numel=10, rms=1.0),
        TensorImportance(name="b", importance=10.0, numel=10, rms=10.0),
    ]
    budgets = allocate_per_tensor_budgets(importances, average_budget=0.05, eta=0.0)
    # eta=0 → uniform; both tensors get 0.05
    assert all(b == pytest.approx(0.05, rel=1e-6) for b in budgets)


def test_allocate_inverse_proportional_when_eta_one() -> None:
    importances = [
        TensorImportance(name="a", importance=1.0, numel=10, rms=1.0),
        TensorImportance(name="b", importance=10.0, numel=10, rms=10.0),
    ]
    budgets = allocate_per_tensor_budgets(importances, average_budget=0.05, eta=1.0)
    # high-importance tensor gets tighter budget
    assert budgets[0] > budgets[1]


def test_allocate_respects_floor_and_cap() -> None:
    importances = [
        TensorImportance(name="a", importance=1.0, numel=10, rms=1.0),
        TensorImportance(name="b", importance=1000.0, numel=10, rms=1000.0),
    ]
    budgets = allocate_per_tensor_budgets(
        importances, average_budget=0.05, eta=2.0, floor=0.001, cap=0.20,
    )
    assert all(0.001 <= b <= 0.20 for b in budgets)


def test_allocate_negative_average_budget_raises() -> None:
    importances = [TensorImportance(name="a", importance=1.0, numel=10, rms=1.0)]
    with pytest.raises(ValueError):
        allocate_per_tensor_budgets(importances, average_budget=-0.01)


def test_allocate_negative_eta_raises() -> None:
    importances = [TensorImportance(name="a", importance=1.0, numel=10, rms=1.0)]
    with pytest.raises(ValueError):
        allocate_per_tensor_budgets(importances, average_budget=0.05, eta=-1.0)


def test_allocate_invalid_floor_cap_raises() -> None:
    importances = [TensorImportance(name="a", importance=1.0, numel=10, rms=1.0)]
    with pytest.raises(ValueError):
        allocate_per_tensor_budgets(
            importances, average_budget=0.05, floor=0.10, cap=0.05,
        )


def test_allocate_empty_returns_empty() -> None:
    assert allocate_per_tensor_budgets([], average_budget=0.05) == []


def test_sensitivity_proxy_constant_is_documented() -> None:
    assert SENSITIVITY_PROXY == "xavier_l2"
