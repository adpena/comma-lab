# SPDX-License-Identifier: MIT
"""Canonical law separating Fisher-natural and raw-dual Bregman metrics."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "bregman_dual_metric_squared_hessian_v1"
METRIC_ID = "argmax_native_vjp_fidelity_v1"
AXIS = "MEASURED_LOCAL_CPU_SYNTHETIC_MATH_FIXTURE_NOT_SCORE"
REAL_N600_SELECTION_STATUS = "NO_VERDICT_DATA_CUSTODY"
MEASUREMENT_UTC = "2026-07-14T14:16:00Z"
MEASUREMENT_ARTIFACT = ".omx/research/bregman_v9_all_surfaces_measurement_20260714.json"
BINDING_ARTIFACT = ".omx/research/bregman_v9_all_surfaces_binding_20260714.json"
INFORMATION_GEOMETRY_HELPER = "tac.information_geometry.bregman_v9_surfaces"
PREMISE_MEMO = (
    ".omx/research/"
    "codex_premise_falsification_bregman_dual_euclidean_20260714_codex.md"
)
STATE_COUNT = 600
RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES = 600
MAX_PRIMAL_EXACT_DUAL_ERROR = 5.684341886080802e-14
MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR = 9.094947017729282e-13


def bregman_dual_metric_squared_hessian_law(
    hessian, delta_theta, delta_eta
) -> dict[str, float]:
    """Evaluate the four local quantities through the validated NumPy helper."""

    from tac.information_geometry.bregman_v9_surfaces import (
        local_hessian_dual_geometry_summary,
    )

    return local_hessian_dual_geometry_summary(hessian, delta_theta, delta_eta)


def build_bregman_dual_metric_squared_hessian_v1() -> CanonicalEquation:
    """Build the corrected equation and retained deterministic n600 anchor."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEASUREMENT_ARTIFACT,
        reactivation_criteria=(
            "promotion is forbidden; real-n600 metric selection requires complete data "
            "custody and a separate authority-grade adoption decision"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS arm64 local CPU NumPy-fp64 synthetic SPD fixture",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="bregman_v9_dual_metric_correction_600_state_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "metric_id": METRIC_ID,
            "fixture": "synthetic_fixed_spd_local_chart_600_states",
            "state_count": STATE_COUNT,
            "dimension": 5,
            "axis": AXIS,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
        },
        predicted_output={
            "ordinary_hessian_identity": "dtheta.T@H@dtheta == deta.T@solve(H,deta)",
            "raw_dual_identity": "deta.T@deta == dtheta.T@H@H@dtheta",
            "raw_dual_equals_ordinary_hessian": False,
        },
        empirical_output={
            "raw_dual_vs_ordinary_hessian_mismatches": (
                RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES
            ),
            "state_count": STATE_COUNT,
            "max_primal_exact_dual_error": MAX_PRIMAL_EXACT_DUAL_ERROR,
            "max_raw_dual_squared_hessian_error": (
                MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR
            ),
            "dual_euclidean_no_solve_scope": "squared_hessian_H_squared_only",
            "binding_artifact": BINDING_ARTIFACT,
            "fisher_natural_cotangent_solve_elided": False,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
        },
        residual=MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR,
        source_artifact=MEASUREMENT_ARTIFACT,
        measurement_method=(
            "deterministic NumPy-fp64 evaluation of 600 five-dimensional SPD local charts; "
            "Fisher-natural cotangent evaluated with a typed linear solve"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Bregman dual metric squared-Hessian correction",
        one_line_summary=(
            "Fisher-natural cotangent length uses H^-1, while raw dual Euclidean length "
            "is the distinct H^2 primal quadratic form."
        ),
        latex_form=(
            r"\Delta\eta=H\Delta\theta,\quad "
            r"\Delta\theta^T H\Delta\theta="
            r"\Delta\eta^T H^{-1}\Delta\eta,\quad "
            r"\|\Delta\eta\|_2^2="
            r"\Delta\theta^T H^2\Delta\theta"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.bregman_v9_surfaces_20260714:"
            "bregman_dual_metric_squared_hessian_law"
        ),
        domain_of_validity={
            "metric_id": METRIC_ID,
            "included": "finite non-empty local charts with symmetric positive-definite H",
            "fisher_natural_cotangent_geometry": "inverse_hessian_H_inverse",
            "fisher_natural_cotangent_solve": "typed_linear_solve",
            "fisher_natural_cotangent_solve_elided": False,
            "dual_euclidean_no_solve_scope": "squared_hessian_H_squared_only",
            "binding_artifact": BINDING_ARTIFACT,
            "axis": AXIS,
            "real_n600_selection_status": REAL_N600_SELECTION_STATUS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "excluded": (
                "non-SPD or non-finite metrics; raw dual Euclidean aliases for the "
                "Fisher-natural cotangent metric; score and promotion claims"
            ),
            "verdict_scope": (
                "prompt identity falsified for general SPD H; squared-Hessian form valid; "
                "canonical metric family not rejected"
            ),
        },
        units_in={
            "hessian": "dual_coordinate_units_per_primal_coordinate_unit",
            "delta_theta": "primal_coordinate_units",
            "delta_eta": "dual_coordinate_units",
        },
        units_out={
            "primal_hessian": "local_metric_squared_distance",
            "fisher_natural_cotangent": "local_metric_squared_distance",
            "raw_dual_euclidean": "dual_coordinate_squared_distance",
            "squared_hessian": "dual_coordinate_squared_distance",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "primal_vs_exact_dual": MAX_PRIMAL_EXACT_DUAL_ERROR,
            "raw_dual_vs_squared_hessian": MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.bregman_dual_metric_guard",
            "tac.witness_dsl.optimal_basis_20260714",
        ),
        canonical_producers=(INFORMATION_GEOMETRY_HELPER,),
        provenance=provenance,
    )


def populate_bregman_dual_metric_squared_hessian_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicitly register the equation; importing this module never mutates state."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_bregman_dual_metric_squared_hessian_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "Bregman dual-metric correction; synthetic math fixture only; "
            "promotion forbidden"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "BINDING_ARTIFACT",
    "EQUATION_ID",
    "INFORMATION_GEOMETRY_HELPER",
    "MAX_PRIMAL_EXACT_DUAL_ERROR",
    "MAX_RAW_DUAL_SQUARED_HESSIAN_ERROR",
    "METRIC_ID",
    "RAW_DUAL_VS_ORDINARY_HESSIAN_MISMATCHES",
    "REAL_N600_SELECTION_STATUS",
    "STATE_COUNT",
    "bregman_dual_metric_squared_hessian_law",
    "build_bregman_dual_metric_squared_hessian_v1",
    "populate_bregman_dual_metric_squared_hessian_v1",
]
