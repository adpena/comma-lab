# SPDX-License-Identifier: MIT
"""Tests for Row #6 ROC-optimal HIGH_PAIR_INVARIANT threshold solver."""

from __future__ import annotations

import pytest

from tac.analytical_solve_extinctions.roc_optimal_high_pair_invariant_threshold import (
    ROCThresholdInput,
    solve_roc_optimal_high_pair_invariant_threshold,
)


def test_separable_corpus_basic() -> None:
    # Positives have scores 0.8-1.0; negatives 0.0-0.3 — perfectly separable
    examples = [
        (0.05, False), (0.10, False), (0.15, False), (0.20, False),
        (0.25, False), (0.30, False),
        (0.80, True), (0.85, True), (0.90, True), (0.95, True), (1.00, True),
    ]
    res = solve_roc_optimal_high_pair_invariant_threshold(
        ROCThresholdInput(labeled_examples=examples),
    )
    assert 0.3 < res.solved_value < 0.8
    assert res.intermediate_values["best_tpr"] == pytest.approx(1.0)
    assert res.intermediate_values["best_fpr"] == pytest.approx(0.0)
    assert res.intermediate_values["youdens_j"] == pytest.approx(1.0)


def test_error_path_zero_positives_raises() -> None:
    with pytest.raises(ValueError, match="zero positives"):
        ROCThresholdInput(
            labeled_examples=[(0.1, False), (0.2, False), (0.3, False)],
        )


def test_error_path_zero_negatives_raises() -> None:
    with pytest.raises(ValueError, match="zero negatives"):
        ROCThresholdInput(
            labeled_examples=[(0.1, True), (0.2, True), (0.3, True)],
        )


def test_regression_overlapping_distributions_asymmetric_costs() -> None:
    # FP much more expensive than FN
    examples = [
        (0.1, False), (0.3, False), (0.5, False),
        (0.4, True), (0.6, True), (0.8, True),
    ]
    res_high_fp_cost = solve_roc_optimal_high_pair_invariant_threshold(
        ROCThresholdInput(
            labeled_examples=examples, fp_cost_multiplier=10.0, fn_cost_multiplier=1.0
        ),
    )
    # ROC curve preserved
    roc = res_high_fp_cost.intermediate_values["roc_curve"]
    assert len(roc) >= 4
    assert all(0 <= tpr <= 1 and 0 <= fpr <= 1 for _, tpr, fpr, _ in roc)


def test_integration_atom_emission_when_requested() -> None:
    examples = [(0.1, False), (0.9, True)]
    res = solve_roc_optimal_high_pair_invariant_threshold(
        ROCThresholdInput(labeled_examples=examples),
        emit_arbitrariness_atom=True,
    )
    atom = res.coupled_adjustments["atom"]
    assert atom.atom_id == "high_pair_invariant_threshold_roc_optimal"
    assert atom.resolution_path.value == "analytical_solve"
