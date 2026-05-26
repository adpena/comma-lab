# SPDX-License-Identifier: MIT
"""Tests for META-LAGRANGIAN-DUAL-SOLVER PHASE 2 per-axis dual-variable computation.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #355
Phase 2 advancement + Catalog #323 canonical Provenance + Catalog #341
canonical-routing markers.

Test classes:
1. Dykstra alternating projections convergence on synthetic 3D polytope
2. Per-axis KKT residual computation correctness
3. Per-candidate per-axis adjustment factor bounded [0.95, 1.05]
4. Canonical Provenance per Catalog #323
5. Cathedral autopilot ranker integration via Phase 2 module
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.findings_lagrangian.dual_solver_phase_2 import (
    AXIS_NAMES,
    DYKSTRA_DEFAULT_EPSILON,
    DYKSTRA_DEFAULT_MAX_ITERATIONS,
    MLX_AVAILABLE,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN,
    PHASE_2_DUAL_SOLVER_SCHEMA_VERSION,
    PerAxisDualSolverResult,
    Phase2SolverError,
    _dual_variable_from_correction,
    _exposed_scalar_adjustment_factor,
    compute_per_axis_dual_variables,
    dykstra_alternating_projections_3_axis,
    kkt_residuals_per_axis,
    per_axis_adjustment_factors,
)


# ─────────────────────────────────────────────────────────────────
# Dykstra alternating projections convergence
# ─────────────────────────────────────────────────────────────────


class TestDykstraConvergence:
    def test_already_inside_polytope_converges_immediately(self) -> None:
        x0 = [0.5, 0.005, 250_000]
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        x_conv, iters, conv, _ = dykstra_alternating_projections_3_axis(
            x0, budgets=budgets
        )
        assert conv
        assert iters == 1
        np.testing.assert_allclose(x_conv, x0, atol=1e-9)

    def test_outside_seg_axis_converges_to_boundary(self) -> None:
        x0 = [2.0, 0.005, 250_000]
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        x_conv, iters, conv, _ = dykstra_alternating_projections_3_axis(
            x0, budgets=budgets
        )
        assert conv
        assert iters <= 5
        assert x_conv[0] == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= x_conv[1] <= 0.01
        assert 0.0 <= x_conv[2] <= 350_000

    def test_outside_all_axes_converges_to_corner(self) -> None:
        x0 = [2.0, 0.05, 500_000]
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        x_conv, iters, conv, _ = dykstra_alternating_projections_3_axis(
            x0, budgets=budgets
        )
        assert conv
        assert iters <= 10
        assert x_conv[0] == pytest.approx(1.0, abs=1e-9)
        assert x_conv[1] == pytest.approx(0.01, abs=1e-9)
        assert x_conv[2] == pytest.approx(350_000, abs=1e-9)

    def test_negative_targets_clip_to_lower_bound(self) -> None:
        x0 = [-1.0, -0.001, -100]
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        x_conv, iters, conv, _ = dykstra_alternating_projections_3_axis(
            x0, budgets=budgets
        )
        assert conv
        np.testing.assert_allclose(x_conv, [0.0, 0.0, 0.0], atol=1e-9)

    def test_max_iterations_bound(self) -> None:
        # Confirm we never exceed the max iteration cap.
        x0 = [10.0, 1.0, 1_000_000]
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        x_conv, iters, conv, _ = dykstra_alternating_projections_3_axis(
            x0, budgets=budgets, max_iterations=5
        )
        assert iters <= 5

    def test_invalid_x0_length_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="length 3"):
            dykstra_alternating_projections_3_axis(
                [1.0, 2.0], budgets=[(0, 1), (0, 1), (0, 1)]
            )

    def test_invalid_budgets_length_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="length 3"):
            dykstra_alternating_projections_3_axis(
                [1.0, 2.0, 3.0], budgets=[(0, 1), (0, 1)]
            )

    def test_lower_above_upper_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="lower=.* > upper="):
            dykstra_alternating_projections_3_axis(
                [1.0, 2.0, 3.0],
                budgets=[(5.0, 1.0), (0, 1), (0, 1)],
            )

    def test_max_iterations_must_be_positive(self) -> None:
        with pytest.raises(Phase2SolverError, match="max_iterations must be > 0"):
            dykstra_alternating_projections_3_axis(
                [1, 2, 3], budgets=[(0, 1), (0, 1), (0, 1)], max_iterations=0
            )

    def test_epsilon_must_be_positive(self) -> None:
        with pytest.raises(Phase2SolverError, match="epsilon must be > 0"):
            dykstra_alternating_projections_3_axis(
                [1, 2, 3], budgets=[(0, 1), (0, 1), (0, 1)], epsilon=0.0
            )


# ─────────────────────────────────────────────────────────────────
# Per-axis KKT residual computation
# ─────────────────────────────────────────────────────────────────


class TestKKTResiduals:
    def test_interior_point_zero_residuals(self) -> None:
        x = np.array([0.5, 0.005, 250_000])
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        residuals = kkt_residuals_per_axis(x, budgets)
        for axis in AXIS_NAMES:
            assert residuals[axis] == pytest.approx(0.0)

    def test_on_upper_boundary_zero_residual(self) -> None:
        x = np.array([1.0, 0.005, 250_000])
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        residuals = kkt_residuals_per_axis(x, budgets)
        assert residuals["seg"] == pytest.approx(0.0)
        assert residuals["pose"] == pytest.approx(0.0)
        assert residuals["rate"] == pytest.approx(0.0)

    def test_above_upper_positive_residual(self) -> None:
        x = np.array([1.5, 0.005, 250_000])
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        residuals = kkt_residuals_per_axis(x, budgets)
        assert residuals["seg"] == pytest.approx(0.5)
        assert residuals["pose"] == pytest.approx(0.0)
        assert residuals["rate"] == pytest.approx(0.0)

    def test_below_lower_positive_residual(self) -> None:
        x = np.array([-0.5, 0.005, 250_000])
        budgets = [(0.0, 1.0), (0.0, 0.01), (0, 350_000)]
        residuals = kkt_residuals_per_axis(x, budgets)
        assert residuals["seg"] == pytest.approx(0.5)
        assert residuals["pose"] == pytest.approx(0.0)
        assert residuals["rate"] == pytest.approx(0.0)

    def test_invalid_x_length_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="x_converged must be length 3"):
            kkt_residuals_per_axis(np.array([1, 2]), [(0, 1)] * 3)


# ─────────────────────────────────────────────────────────────────
# Per-axis bounded adjustment factors
# ─────────────────────────────────────────────────────────────────


class TestPerAxisAdjustmentFactors:
    def test_zero_dual_gives_unity_factor(self) -> None:
        dual = {a: 0.0 for a in AXIS_NAMES}
        sigma = {a: 1.0 for a in AXIS_NAMES}
        factors = per_axis_adjustment_factors(dual, sigma)
        for a in AXIS_NAMES:
            assert factors[a] == pytest.approx(1.0)

    def test_positive_dual_downweights(self) -> None:
        # Positive dual = binding = should downweight (< 1.0).
        dual = {"seg": 5.0, "pose": 0.0, "rate": 0.0}
        sigma = {a: 0.5 for a in AXIS_NAMES}
        factors = per_axis_adjustment_factors(dual, sigma)
        assert factors["seg"] < 1.0
        assert factors["seg"] >= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN
        assert factors["pose"] == pytest.approx(1.0)
        assert factors["rate"] == pytest.approx(1.0)

    def test_all_factors_inside_bounded_envelope(self) -> None:
        # Adversarial: very large dual variables; factor must still be in [0.95, 1.05].
        dual = {"seg": 1e6, "pose": 1e6, "rate": 1e6}
        sigma = {a: 1e-9 for a in AXIS_NAMES}
        factors = per_axis_adjustment_factors(dual, sigma)
        for a in AXIS_NAMES:
            assert PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN <= factors[a] <= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX

    def test_higher_sigma_decreases_magnitude(self) -> None:
        dual = {a: 5.0 for a in AXIS_NAMES}
        sigma_low = {a: 0.1 for a in AXIS_NAMES}
        sigma_high = {a: 10.0 for a in AXIS_NAMES}
        factors_low = per_axis_adjustment_factors(dual, sigma_low)
        factors_high = per_axis_adjustment_factors(dual, sigma_high)
        # Higher sigma = more uncertainty = adjustment closer to 1.0.
        for a in AXIS_NAMES:
            assert abs(factors_high[a] - 1.0) < abs(factors_low[a] - 1.0)

    def test_negative_sigma_treated_as_zero(self) -> None:
        # Should NOT crash on negative sigma; just floor to 0.
        dual = {a: 5.0 for a in AXIS_NAMES}
        sigma = {a: -1.0 for a in AXIS_NAMES}
        factors = per_axis_adjustment_factors(dual, sigma)
        for a in AXIS_NAMES:
            assert PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN <= factors[a] <= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX


# ─────────────────────────────────────────────────────────────────
# Scalar adjustment factor aggregation
# ─────────────────────────────────────────────────────────────────


class TestScalarAdjustmentAggregation:
    def test_all_unity_gives_unity(self) -> None:
        per_axis = {a: 1.0 for a in AXIS_NAMES}
        assert _exposed_scalar_adjustment_factor(per_axis) == pytest.approx(1.0)

    def test_geometric_mean_inside_bounds(self) -> None:
        per_axis = {"seg": 0.95, "pose": 1.0, "rate": 1.05}
        result = _exposed_scalar_adjustment_factor(per_axis)
        assert PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN <= result <= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX
        # Geometric mean of (0.95, 1.0, 1.05) ≈ 0.9983...
        expected = (0.95 * 1.0 * 1.05) ** (1 / 3.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_all_min_yields_min(self) -> None:
        per_axis = {a: PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN for a in AXIS_NAMES}
        assert _exposed_scalar_adjustment_factor(per_axis) == pytest.approx(
            PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN
        )

    def test_all_max_yields_max(self) -> None:
        per_axis = {a: PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX for a in AXIS_NAMES}
        assert _exposed_scalar_adjustment_factor(per_axis) == pytest.approx(
            PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX
        )

    def test_missing_axis_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="must have all of"):
            _exposed_scalar_adjustment_factor({"seg": 1.0, "pose": 1.0})


# ─────────────────────────────────────────────────────────────────
# End-to-end compute_per_axis_dual_variables (cathedral integration)
# ─────────────────────────────────────────────────────────────────


class TestComputePerAxisDualVariables:
    def test_inside_polytope_no_binding_constraint(self) -> None:
        result = compute_per_axis_dual_variables(
            "cand_inside",
            predicted_axis_targets={"seg": 0.5, "pose": 0.005, "rate": 250_000},
            per_axis_budgets={
                "seg": (0.0, 1.0),
                "pose": (0.0, 0.01),
                "rate": (0, 350_000),
            },
        )
        assert result.converged
        # No binding constraint → duals near zero → scalar factor near 1.
        assert result.adjustment_factor == pytest.approx(1.0)
        for axis in AXIS_NAMES:
            assert result.dual_variables_per_axis[axis] == pytest.approx(0.0)

    def test_outside_polytope_binding_constraints(self) -> None:
        result = compute_per_axis_dual_variables(
            "cand_outside",
            predicted_axis_targets={"seg": 2.0, "pose": 0.05, "rate": 500_000},
            per_axis_budgets={
                "seg": (0.0, 1.0),
                "pose": (0.0, 0.01),
                "rate": (0, 350_000),
            },
            per_axis_posterior_sigma={"seg": 0.1, "pose": 0.1, "rate": 0.1},
        )
        assert result.converged
        # All 3 axes binding → all duals > 0 → factor < 1.0.
        for axis in AXIS_NAMES:
            assert result.dual_variables_per_axis[axis] > 0
        assert result.adjustment_factor < 1.0

    def test_scalar_factor_bounded_per_phase_1_envelope(self) -> None:
        # Adversarial: extreme far-outside targets and tiny sigma.
        result = compute_per_axis_dual_variables(
            "cand_extreme",
            predicted_axis_targets={"seg": 1e6, "pose": 1e6, "rate": 10_000_000},
            per_axis_budgets={
                "seg": (0.0, 1.0),
                "pose": (0.0, 0.01),
                "rate": (0, 350_000),
            },
            per_axis_posterior_sigma={"seg": 1e-9, "pose": 1e-9, "rate": 1e-9},
        )
        assert PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN <= result.adjustment_factor <= PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX

    def test_canonical_provenance_propagated(self) -> None:
        provenance = {"kind": "predicted", "model_id": "test_v1", "axis_tag": "[predicted]"}
        result = compute_per_axis_dual_variables(
            "cand_prov",
            predicted_axis_targets={"seg": 0.5, "pose": 0.005, "rate": 250_000},
            per_axis_budgets={
                "seg": (0.0, 1.0),
                "pose": (0.0, 0.01),
                "rate": (0, 350_000),
            },
            canonical_provenance=provenance,
        )
        assert result.canonical_provenance["model_id"] == "test_v1"

    def test_missing_axis_in_targets_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="missing canonical axis"):
            compute_per_axis_dual_variables(
                "cand_missing",
                predicted_axis_targets={"seg": 0.5, "pose": 0.005},
                per_axis_budgets={
                    "seg": (0.0, 1.0),
                    "pose": (0.0, 0.01),
                    "rate": (0, 350_000),
                },
            )

    def test_missing_axis_in_budgets_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="missing canonical axis"):
            compute_per_axis_dual_variables(
                "cand_missing_budget",
                predicted_axis_targets={"seg": 0.5, "pose": 0.005, "rate": 250_000},
                per_axis_budgets={"seg": (0.0, 1.0), "pose": (0.0, 0.01)},
            )

    def test_empty_candidate_id_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="candidate_id"):
            compute_per_axis_dual_variables(
                "",
                predicted_axis_targets={"seg": 0.5, "pose": 0.005, "rate": 250_000},
                per_axis_budgets={
                    "seg": (0.0, 1.0),
                    "pose": (0.0, 0.01),
                    "rate": (0, 350_000),
                },
            )


# ─────────────────────────────────────────────────────────────────
# PerAxisDualSolverResult invariants
# ─────────────────────────────────────────────────────────────────


class TestPerAxisDualSolverResult:
    def _valid_kwargs(self) -> dict:
        return dict(
            candidate_id="test",
            dual_variables_per_axis={a: 0.0 for a in AXIS_NAMES},
            kkt_residual_per_axis={a: 0.0 for a in AXIS_NAMES},
            adjustment_factor_per_axis={a: 1.0 for a in AXIS_NAMES},
            adjustment_factor=1.0,
            dykstra_iterations_to_convergence=1,
            converged=True,
            posterior_sigma_per_axis={a: 1.0 for a in AXIS_NAMES},
        )

    def test_default_axis_tag(self) -> None:
        result = PerAxisDualSolverResult(**self._valid_kwargs())
        assert result.axis_tag == "[predicted]"

    def test_default_score_claim_false(self) -> None:
        result = PerAxisDualSolverResult(**self._valid_kwargs())
        assert result.score_claim is False

    def test_default_promotable_false(self) -> None:
        result = PerAxisDualSolverResult(**self._valid_kwargs())
        assert result.promotable is False

    def test_score_claim_true_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="score_claim must be False"):
            PerAxisDualSolverResult(**{**self._valid_kwargs(), "score_claim": True})

    def test_promotable_true_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="promotable must be False"):
            PerAxisDualSolverResult(**{**self._valid_kwargs(), "promotable": True})

    def test_axis_tag_must_be_predicted(self) -> None:
        with pytest.raises(Phase2SolverError, match=r"\[predicted\]"):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "axis_tag": "[contest-CUDA]"}
            )

    def test_adjustment_factor_below_min_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="adjustment_factor=.* <"):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "adjustment_factor": 0.8}
            )

    def test_adjustment_factor_above_max_rejected(self) -> None:
        with pytest.raises(Phase2SolverError, match="adjustment_factor=.* >"):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "adjustment_factor": 1.2}
            )

    def test_per_axis_factor_below_min_rejected(self) -> None:
        bad = {a: 1.0 for a in AXIS_NAMES}
        bad["seg"] = 0.8
        with pytest.raises(Phase2SolverError):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "adjustment_factor_per_axis": bad}
            )

    def test_per_axis_factor_above_max_rejected(self) -> None:
        bad = {a: 1.0 for a in AXIS_NAMES}
        bad["pose"] = 1.2
        with pytest.raises(Phase2SolverError):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "adjustment_factor_per_axis": bad}
            )

    def test_missing_axis_in_mapping_rejected(self) -> None:
        bad = {a: 0.0 for a in AXIS_NAMES if a != "rate"}
        with pytest.raises(Phase2SolverError, match="missing canonical axis"):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "dual_variables_per_axis": bad}
            )

    def test_negative_posterior_sigma_rejected(self) -> None:
        bad = {a: 1.0 for a in AXIS_NAMES}
        bad["pose"] = -0.5
        with pytest.raises(Phase2SolverError, match="must be >= 0"):
            PerAxisDualSolverResult(
                **{**self._valid_kwargs(), "posterior_sigma_per_axis": bad}
            )

    def test_iterations_above_max_rejected(self) -> None:
        with pytest.raises(
            Phase2SolverError, match="dykstra_iterations_to_convergence"
        ):
            PerAxisDualSolverResult(
                **{
                    **self._valid_kwargs(),
                    "dykstra_iterations_to_convergence": DYKSTRA_DEFAULT_MAX_ITERATIONS
                    + 1,
                }
            )

    def test_as_dict_canonical_schema(self) -> None:
        result = PerAxisDualSolverResult(**self._valid_kwargs())
        d = result.as_dict()
        assert d["schema"] == PHASE_2_DUAL_SOLVER_SCHEMA_VERSION
        assert d["axis_tag"] == "[predicted]"
        assert d["score_claim"] is False
        assert d["promotable"] is False
        for axis in AXIS_NAMES:
            assert axis in d["dual_variables_per_axis"]
            assert axis in d["kkt_residual_per_axis"]
            assert axis in d["adjustment_factor_per_axis"]
            assert axis in d["posterior_sigma_per_axis"]


# ─────────────────────────────────────────────────────────────────
# MLX availability flag
# ─────────────────────────────────────────────────────────────────


class TestMLXAvailability:
    def test_mlx_available_on_m_series_local(self) -> None:
        # On M5 Max local this should be True per the standing directive.
        # Test passes regardless of platform; just asserts the flag is
        # well-typed and importable.
        assert isinstance(MLX_AVAILABLE, bool)


# ─────────────────────────────────────────────────────────────────
# Cathedral integration (Phase 1 → Phase 2 sanity)
# ─────────────────────────────────────────────────────────────────


class TestCathedralAutopilotIntegration:
    def test_result_consumable_by_phase_1_callsite_shape(self) -> None:
        """The Phase 2 result must expose the same scalar adjustment_factor
        field that the Phase 1 callsite already consumes per Catalog #355
        gate contract preservation."""
        result = compute_per_axis_dual_variables(
            "cand_compat",
            predicted_axis_targets={"seg": 0.5, "pose": 0.005, "rate": 250_000},
            per_axis_budgets={
                "seg": (0.0, 1.0),
                "pose": (0.0, 0.01),
                "rate": (0, 350_000),
            },
        )
        # Phase 1 callsite consumes 'adjustment_factor' and 'posterior_sigma_per_term'
        # (latter via the FindingsLagrangianResult). Phase 2 result has
        # 'adjustment_factor' (compatible) + 'adjustment_factor_per_axis'
        # (new observability dict).
        assert hasattr(result, "adjustment_factor")
        assert hasattr(result, "adjustment_factor_per_axis")
        assert hasattr(result, "dual_variables_per_axis")
        assert hasattr(result, "kkt_residual_per_axis")
        assert hasattr(result, "posterior_sigma_per_axis")
        assert hasattr(result, "dykstra_iterations_to_convergence")
        assert hasattr(result, "converged")

    def test_axis_ordering_matches_canonical_score_composition(self) -> None:
        """AXIS_NAMES must match tac.score_composition canonical ordering
        per Catalog #356 sister gate contract."""
        from tac.cathedral.consumer_contract import AxisDecomposition

        # Verify the per-axis ordering is consistent with AxisDecomposition.
        # AxisDecomposition has fields: predicted_d_seg_delta, predicted_d_pose_delta,
        # predicted_archive_bytes_delta.
        # AXIS_NAMES = ("seg", "pose", "rate") — matches.
        assert AXIS_NAMES == ("seg", "pose", "rate")
        # Confirm AxisDecomposition still has these fields.
        decomp = AxisDecomposition(
            predicted_d_seg_delta=-0.0001,
            predicted_d_pose_delta=1e-6,
            predicted_archive_bytes_delta=-200,
        )
        assert hasattr(decomp, "predicted_d_seg_delta")
        assert hasattr(decomp, "predicted_d_pose_delta")
        assert hasattr(decomp, "predicted_archive_bytes_delta")


# ─────────────────────────────────────────────────────────────────
# Determinism + reproducibility
# ─────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_repeated_invocation_yields_identical_result(self) -> None:
        kwargs = dict(
            candidate_id="cand_repro",
            predicted_axis_targets={"seg": 0.7, "pose": 0.02, "rate": 400_000},
            per_axis_budgets={
                "seg": (0.0, 1.0),
                "pose": (0.0, 0.01),
                "rate": (0, 350_000),
            },
            per_axis_posterior_sigma={"seg": 0.5, "pose": 0.5, "rate": 0.5},
        )
        r1 = compute_per_axis_dual_variables(**kwargs)
        r2 = compute_per_axis_dual_variables(**kwargs)
        for axis in AXIS_NAMES:
            assert r1.dual_variables_per_axis[axis] == r2.dual_variables_per_axis[axis]
            assert r1.kkt_residual_per_axis[axis] == r2.kkt_residual_per_axis[axis]
            assert r1.adjustment_factor_per_axis[axis] == r2.adjustment_factor_per_axis[axis]
        assert r1.adjustment_factor == r2.adjustment_factor
        assert r1.dykstra_iterations_to_convergence == r2.dykstra_iterations_to_convergence
