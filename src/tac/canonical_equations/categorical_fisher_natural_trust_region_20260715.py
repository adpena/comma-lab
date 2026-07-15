# SPDX-License-Identifier: MIT
"""Canonical equation for the categorical quotient ``H^-1`` trust solve."""

from __future__ import annotations

from tac.canonical_equations.equation import RECALIBRATE_ON_NEW_ANCHORS, CanonicalEquation
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "categorical_fisher_natural_trust_region_solve_v1"
METRIC_ID = "argmax_native_vjp_fidelity_v1"
AXIS = "DERIVED_NUMPY_FP32_MATH_SURFACE_NOT_SCORE"


def categorical_fisher_natural_trust_region_law(
    probabilities,
    cotangent,
    *,
    delta,
    delta_convention="delta_kl",
    damping=0.0,
    project_gauge=False,
):
    from tac.information_geometry.fisher_natural_solver import (
        solve_categorical_fisher_natural_step_numpy_fp32,
    )

    return solve_categorical_fisher_natural_step_numpy_fp32(
        probabilities,
        cotangent,
        delta=delta,
        delta_convention=delta_convention,
        damping=damping,
        project_gauge=project_gauge,
    )


def build_categorical_fisher_natural_trust_region_solve_v1() -> CanonicalEquation:
    provenance = build_provenance_for_predicted(
        model_id=EQUATION_ID,
        inputs_sha256="0" * 64,
        measurement_axis=AXIS,
        hardware_substrate="portable NumPy-fp32 reference; MLX parity surface",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Categorical-Fisher natural cotangent quotient trust solve",
        one_line_summary=(
            "Solve H^-1 on the additive-logit quotient, then scale the natural descent "
            "step into an explicit local categorical-KL trust ball."
        ),
        latex_form=(
            r"Q^T\mathbf1=0,\ Q^TQ=I,\ "
            r"(Q^TH(p)Q+\lambda I)v=-Q^Tg,\ u=Qv,\ "
            r"u\leftarrow u\min\{1,\sqrt{\Delta_{quad}/(u^THu)}\},\ "
            r"H=\operatorname{diag}(p)-pp^T"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.categorical_fisher_natural_trust_region_20260715:"
            "categorical_fisher_natural_trust_region_law"
        ),
        domain_of_validity={
            "metric_id": METRIC_ID,
            "probabilities": "strictly positive categorical simplex, K>=2",
            "cotangent": "finite zero-sum quotient cotangent; explicit projection opt-in only",
            "trust_region": "local categorical KL; delta_quad=2*delta_kl",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "measurement_status": "solver built; checkpoint A/B owed",
            "verdict_scope": (
                "DERIVED categorical quotient solver; no empirical optimizer or score verdict"
            ),
        },
        units_in={
            "probabilities": "dimensionless",
            "cotangent": "objective_per_logit",
            "delta": "local_KL_or_Fisher_quadratic",
            "damping": "Fisher_curvature",
        },
        units_out={"step": "centred_logit_displacement"},
        empirical_anchors=(),
        # Zero records the exact algebraic residual of the solved normal
        # equation.  The empirical A/B remains explicitly OWED in the domain
        # and must not be encoded as a non-numeric registry residual.
        predicted_vs_empirical_residual={"projected_normal_equation": 0.0},
        last_calibration_utc="2026-07-15T09:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.fisher_natural_solver_policy",
            "tac.canonical_equations.optimal_metric_unification_20260714",
        ),
        canonical_producers=(
            "tac.information_geometry.fisher_natural_solver",
            "tac.information_geometry.fisher_natural_solver_mlx",
        ),
        provenance=provenance,
    )


__all__ = [
    "EQUATION_ID",
    "METRIC_ID",
    "build_categorical_fisher_natural_trust_region_solve_v1",
    "categorical_fisher_natural_trust_region_law",
]
