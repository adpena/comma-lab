# SPDX-License-Identifier: MIT
"""Behavioral tests for the canonical FreSh initialization equations."""

from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.fresh_frequency_shift_init_20260712 import (
    EQUATION_ID,
    OPEN_FRESH_FROM_SCRATCH_SCOPE,
    SETTLED_WARM_START_ALONG26_SCOPE,
    build_fresh_frequency_shift_init_v1,
    fixed_quality_reduction_identity,
    fresh_antidiagonal_spectrum,
    fresh_bias_scale_candidates,
    fresh_wasserstein1_cdf_l1,
    populate_fresh_frequency_shift_init_equation,
    select_fresh_spectrum_candidate,
    tangent_frequency_candidates,
)
from tac.canonical_equations.registry import query_equations


def test_exact_unshifted_antidiagonal_definition_omits_dc() -> None:
    # For a 2x2 spatial impulse every DFT magnitude is one.  Degree 1 has
    # (0,1)+(1,0)=2 and degree 2 has (1,1)=1; (0,0) DC is omitted.
    impulse = np.array([[1.0, 0.0], [0.0, 0.0]])
    raw = fresh_antidiagonal_spectrum(impulse, 2, normalize=False)
    np.testing.assert_array_equal(raw, np.array([2.0, 1.0]))
    np.testing.assert_allclose(
        fresh_antidiagonal_spectrum(impulse, 2),
        np.array([2.0 / 3.0, 1.0 / 3.0]),
    )


def test_channels_sum_before_single_normalization_and_dc_offset_is_invariant() -> None:
    impulse = np.array([[1.0, 0.0], [0.0, 0.0]])
    channels = np.stack((impulse, 3.0 * impulse))
    np.testing.assert_array_equal(
        fresh_antidiagonal_spectrum(channels, 2, normalize=False),
        np.array([8.0, 4.0]),
    )
    np.testing.assert_allclose(
        fresh_antidiagonal_spectrum(impulse + 17.0, 2),
        fresh_antidiagonal_spectrum(impulse, 2),
        rtol=0.0,
        atol=2e-15,
    )


def test_wasserstein_is_exact_unit_bin_cdf_l1_and_scale_invariant() -> None:
    assert fresh_wasserstein1_cdf_l1([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]) == 2.0
    assert fresh_wasserstein1_cdf_l1([2.0, 2.0], [1.0, 1.0]) == 0.0
    expected = abs(0.5 - 0.25) + abs(1.0 - 1.0)
    assert fresh_wasserstein1_cdf_l1([1.0, 1.0], [1.0, 3.0]) == pytest.approx(expected)


def test_selector_minimizes_mean_across_targets_and_uses_stable_first_tie() -> None:
    targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],  # exact tie; first high-frequency candidate must win
        ]
    )
    selected = select_fresh_spectrum_candidate(targets, candidates)
    assert selected.index == 1
    assert selected.mean_wasserstein1 == pytest.approx(1.0 / 3.0)
    assert selected.per_target_wasserstein1 == pytest.approx((1.0, 0.0, 0.0))


def test_tangent_ladder_closes_measured_deficit_without_asserting_endpoint_optimum() -> None:
    candidates = tangent_frequency_candidates()
    assert candidates == pytest.approx((8.0, 8.0 * np.sqrt(3.2), 25.6))
    equation = build_fresh_frequency_shift_init_v1()
    assert equation.domain_of_validity["frequency_candidates"] == pytest.approx(candidates)
    assert "may retain baseline" in equation.domain_of_validity["frequency_candidate_semantics"]


def test_bias_scale_grid_is_exact_inclusive_and_has_no_float_drift_at_endpoint() -> None:
    candidates = fresh_bias_scale_candidates()
    assert len(candidates) == 31
    assert candidates[0] == 0.0
    assert candidates[1] == 0.1
    assert candidates[-1] == 3.0
    assert candidates == pytest.approx(tuple(index / 10 for index in range(31)))


def test_fixed_quality_identity_is_proportional_when_init_overheads_match() -> None:
    reduction = fixed_quality_reduction_identity(
        100,
        60,
        pairs_per_epoch=600,
        epoch_seconds=10.0,
    )
    assert reduction.baseline_scorer_calls == 60_000
    assert reduction.initialized_scorer_calls == 36_000
    assert reduction.epoch_reduction_fraction == pytest.approx(0.4)
    assert reduction.scorer_call_reduction_fraction == pytest.approx(0.4)
    assert reduction.wall_clock_reduction_fraction == pytest.approx(0.4)
    assert reduction.scorer_seconds_saved == pytest.approx(380.0)  # 40*10*0.95


def test_fixed_quality_identity_accounts_for_one_time_init_cost() -> None:
    reduction = fixed_quality_reduction_identity(
        100,
        60,
        epoch_seconds=10.0,
        initialized_init_seconds=100.0,
    )
    assert reduction.epoch_reduction_fraction == pytest.approx(0.4)
    assert reduction.wall_clock_reduction_fraction == pytest.approx(0.3)


def test_fixed_quality_identity_accounts_for_init_scorer_sweep() -> None:
    reduction = fixed_quality_reduction_identity(
        2,
        1,
        pairs_per_epoch=8,
        baseline_init_scorer_calls=2,
        initialized_init_scorer_calls=94,
    )
    assert reduction.baseline_scorer_calls == 16
    assert reduction.initialized_scorer_calls == 8
    assert reduction.baseline_total_scorer_calls == 18
    assert reduction.initialized_total_scorer_calls == 102
    assert reduction.scorer_call_reduction_fraction == pytest.approx(0.5)
    assert reduction.total_scorer_call_reduction_fraction < 0.0


def test_verdict_scopes_do_not_reopen_settled_warm_start_formulation() -> None:
    assert SETTLED_WARM_START_ALONG26_SCOPE.level == "formulation"
    assert OPEN_FRESH_FROM_SCRATCH_SCOPE.level == "formulation"
    assert "bounded_warm_start" in SETTLED_WARM_START_ALONG26_SCOPE.formulation
    assert "REFUTED" in SETTLED_WARM_START_ALONG26_SCOPE.status
    assert "from_scratch" in OPEN_FRESH_FROM_SCRATCH_SCOPE.formulation
    assert "OPEN/OWED" in OPEN_FRESH_FROM_SCRATCH_SCOPE.status
    assert SETTLED_WARM_START_ALONG26_SCOPE.formulation != OPEN_FRESH_FROM_SCRATCH_SCOPE.formulation


def test_equation_is_non_promotable_and_carries_measured_and_source_anchors() -> None:
    equation = build_fresh_frequency_shift_init_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.python_callable_module_path.endswith(":select_fresh_spectrum_candidate")
    assert len(equation.empirical_anchors) == 4
    assert equation.domain_of_validity["score_authority"].startswith("none")
    assert equation.provenance.promotion_eligible is False
    assert equation.provenance.score_claim_valid is False
    assert equation.domain_of_validity["settled_excluded_formulation"]["level"] == "formulation"
    assert equation.domain_of_validity["open_formulation"]["level"] == "formulation"
    assert "thin/dashed" in equation.domain_of_validity["selection_surface"]
    warm_start = next(
        anchor for anchor in equation.empirical_anchors if anchor.anchor_id.startswith("warm_start_along26")
    )
    assert warm_start.residual == 3.2e-05
    assert warm_start.empirical_output["delta_rebalanced_minus_off_ep700"] == 3.2e-05
    assert warm_start.noise_floor is None
    assert equation.predicted_vs_empirical_residual["warm_start_fixed_along26_hypothesis"] == 3.2e-05


def test_population_round_trips_only_through_explicit_isolated_registry(tmp_path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.jsonl.lock"
    populated = populate_fresh_frequency_shift_init_equation(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="fresh_equation",
    )
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [equation.equation_id for equation in loaded] == [EQUATION_ID]


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: fresh_antidiagonal_spectrum(np.ones((2, 2)), 0), "positive integer"),
        (lambda: fresh_antidiagonal_spectrum(np.ones((2, 2)), 3), "exceeds available"),
        (lambda: fresh_antidiagonal_spectrum(np.ones((2, 2)), 1), "positive non-DC mass"),
        (lambda: fresh_wasserstein1_cdf_l1([0.0, 0.0], [1.0, 0.0]), "positive mass"),
        (lambda: tangent_frequency_candidates(8.0, 0.5), "at least one"),
        (lambda: fixed_quality_reduction_identity(0, 0), "positive integer"),
    ],
)
def test_equations_fail_closed_on_invalid_domains(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
