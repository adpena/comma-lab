# SPDX-License-Identifier: MIT
"""Tests for Phase 4 (compressive-sensing coverage estimator)."""
from __future__ import annotations

import math

import pytest

from tac.preflight_rudin_daubechies import (
    CompressiveCoverageEstimator,
)


def test_estimator_rejects_empty_gate_numbers():
    with pytest.raises(ValueError):
        CompressiveCoverageEstimator(
            gate_numbers=[], fixture_classes=["a"], n_samples=4
        )


def test_estimator_rejects_empty_fixture_classes():
    with pytest.raises(ValueError):
        CompressiveCoverageEstimator(
            gate_numbers=[1], fixture_classes=[], n_samples=4
        )


def test_estimator_rejects_zero_samples():
    with pytest.raises(ValueError):
        CompressiveCoverageEstimator(
            gate_numbers=[1], fixture_classes=["a"], n_samples=0
        )


def test_estimator_initializes_full_cell_grid():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1, 5, 7, 146], fixture_classes=["trainer_a", "trainer_b"], n_samples=4
    )
    assert est.total_gates == 4
    assert est.total_fixture_classes == 2
    assert len(est.cells) == 8  # 4 * 2


def test_estimator_cells_default_uniform_prior():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1, 5], fixture_classes=["a"], n_samples=4
    )
    cell = est.cell(1, "a")
    assert cell is not None
    assert cell.activation_estimate == 0.5
    assert not cell.is_empirical


def test_uncertainty_bound_per_devore():
    # Per Daubechies-DeVore O(sqrt(N/K)) bound: with N=8, K=4, uncertainty
    # should be at most sqrt(8/4)/8 = 0.177...
    est = CompressiveCoverageEstimator(
        gate_numbers=list(range(1, 9)), fixture_classes=["a"], n_samples=4
    )
    cell = est.cell(1, "a")
    assert cell is not None
    expected_bound = math.sqrt(8 / 4) / 8
    assert cell.uncertainty <= expected_bound + 1e-9


def test_update_from_empirical_run_replaces_cell():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1, 5], fixture_classes=["a", "b"], n_samples=4
    )
    new_cell = est.update_from_empirical_run(1, "a", 1.0)
    assert new_cell.activation_estimate == 1.0
    assert new_cell.uncertainty == 0.0
    assert new_cell.is_empirical


def test_update_rejects_invalid_activation():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1], fixture_classes=["a"], n_samples=4
    )
    with pytest.raises(ValueError):
        est.update_from_empirical_run(1, "a", 1.5)
    with pytest.raises(ValueError):
        est.update_from_empirical_run(1, "a", -0.1)


def test_update_rejects_unknown_cell():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1], fixture_classes=["a"], n_samples=4
    )
    with pytest.raises(ValueError):
        est.update_from_empirical_run(999, "a", 0.5)


def test_reproject_sister_cells_inherits_empirical_mean():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1], fixture_classes=["a", "b", "c"], n_samples=4
    )
    est.update_from_empirical_run(1, "a", 1.0)
    # Sister cells (1, "b") and (1, "c") should now reflect the empirical mean.
    sister_b = est.cell(1, "b")
    sister_c = est.cell(1, "c")
    assert sister_b is not None
    assert sister_c is not None
    assert sister_b.activation_estimate == 1.0
    assert sister_c.activation_estimate == 1.0
    # Sister uncertainty is reduced from the cold-start prior.
    assert sister_b.uncertainty < 1.0


def test_remaining_uncertainty_decreases_after_observations():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1, 5], fixture_classes=["a", "b"], n_samples=4
    )
    initial_unc = est.remaining_uncertainty_total()
    est.update_from_empirical_run(1, "a", 1.0)
    after_unc = est.remaining_uncertainty_total()
    # After an empirical observation: that cell's uncertainty is 0 (excluded),
    # AND sister cells in the same gate are reduced.
    assert after_unc < initial_unc


def test_next_fixture_to_observe_is_highest_uncertainty():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1, 5], fixture_classes=["a", "b"], n_samples=4
    )
    next_probe = est.next_fixture_to_observe()
    assert next_probe is not None
    # All cells start at the same uncertainty; first by sort order.
    assert next_probe in [(1, "a"), (1, "b"), (5, "a"), (5, "b")]


def test_next_fixture_to_observe_returns_none_when_all_empirical():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1], fixture_classes=["a"], n_samples=4
    )
    est.update_from_empirical_run(1, "a", 0.5)
    assert est.next_fixture_to_observe() is None


def test_cell_explain_includes_kind_label():
    est = CompressiveCoverageEstimator(
        gate_numbers=[1], fixture_classes=["a"], n_samples=4
    )
    cell = est.cell(1, "a")
    assert "L1-reconstructed" in cell.explain()
    est.update_from_empirical_run(1, "a", 1.0)
    cell2 = est.cell(1, "a")
    assert "empirical" in cell2.explain()
