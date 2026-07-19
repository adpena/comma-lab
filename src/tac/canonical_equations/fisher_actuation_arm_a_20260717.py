# SPDX-License-Identifier: MIT
"""Canonical equation: closed-form categorical-Fisher pseudo-inverse cotangent precondition.

Build-wave arm A (SPEC_v10 §13.4 surface 2, ``p0_build_fisher_actuation_20260717``): the
``--head-natural-grad`` lever's law, registered as its portable numpy-fp64 reference twin.

The law (DERIVED; the frozen SegNet head is EXACT rank-4 linear per
``segnet_head_rank4_linear_flipdist_v1``, so the logit-space Fisher g = diag(p) − p pᵀ is the
whole decision-geometry metric and its pseudo-inverse is closed-form on its range):

    for per-pixel cotangents v with 1ᵀv = 0 (every shift-invariant seg form),
    g⁺ v = v/p − mean_k(v_k/p_k) · 1            (min-norm; verify g(g⁺v) = v)

damped as ``v/(p+eps)`` with gauge re-projection before and after the division. O(K) per pixel —
no solve. Sister (same family, different realization): ``tac.information_geometry.
fisher_natural_solver`` (the Helmert-quotient H⁻¹ TRUST solve, ``categorical_fisher_natural_
trust_region_solve_v1``) — that surface solves a damped quotient system per pixel for a
trust-region STEP at analysis time; this law is the per-STEP in-graph training-force
preconditioner (mlx forward-identity/backward-g⁺ custom vjp in
``experiments/train_witness_realized_through_R_mlx.make_seg_logits_natural_grad_mlx``).

Status: DERIVED + unit-verified (inversion property; cross-checked against the sister quotient
solver at zero damping); the training A/B ($0 cached-ckpt) is OWED — no optimizer or score
verdict is encoded here. research_only; score_claim false; pointer UNMOVED.
"""
from __future__ import annotations

import numpy as np

from tac.canonical_equations.equation import RECALIBRATE_ON_NEW_ANCHORS, CanonicalEquation
from tac.provenance.builders import build_provenance_for_predicted

EQUATION_ID = "categorical_fisher_pseudoinverse_cotangent_precondition_v1"
AXIS = "DERIVED_NUMPY_FP64_MATH_SURFACE_NOT_SCORE"


def categorical_fisher_pseudoinverse_precondition_law(
    probabilities, cotangent, *, eps: float = 1e-3
):
    """Damped closed-form g⁺ cotangent precondition (numpy reference twin of the mlx vjp).

    Args:
        probabilities: (..., K) softmax rows (strictly positive).
        cotangent: (..., K) objective gradient w.r.t. logits (zero-sum up to fp for
            shift-invariant objectives; the gauge component is projected out).
        eps: damping added to p before the division (must be >= 0; 0 = exact g⁺ on
            strictly-positive rows).

    Returns:
        (..., K) preconditioned cotangent u with sum_k u_k = 0 per row and, at eps=0,
        g u = (gauge-projected) cotangent exactly.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    v = np.asarray(cotangent, dtype=np.float64)
    if p.shape != v.shape:
        raise ValueError(f"shape mismatch: probabilities {p.shape} vs cotangent {v.shape}")
    if not float(eps) >= 0.0:
        raise ValueError(f"eps must be >= 0, got {eps!r}")
    if np.any(p <= 0.0) and float(eps) == 0.0:
        raise ValueError("eps=0 requires strictly positive probabilities (exact 1/p)")
    v = v - v.mean(axis=-1, keepdims=True)
    u = v / (p + float(eps))
    return u - u.mean(axis=-1, keepdims=True)


def build_categorical_fisher_pseudoinverse_cotangent_precondition_v1() -> CanonicalEquation:
    provenance = build_provenance_for_predicted(
        model_id=EQUATION_ID,
        inputs_sha256="0" * 64,
        measurement_axis=AXIS,
        hardware_substrate="portable NumPy-fp64 reference; MLX custom-vjp parity surface",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Categorical-Fisher closed-form pseudo-inverse cotangent precondition",
        one_line_summary=(
            "g = diag(p)-pp^T has closed-form g+ v = v/p - mean(v/p) on zero-sum cotangents; "
            "O(K)/px natural-gradient precondition of the seg training force (A/B owed)."
        ),
        latex_form=(
            r"g=\operatorname{diag}(p)-pp^\top,\ \mathbf 1^\top v=0\Rightarrow "
            r"g^{+}v=\frac{v}{p}-\overline{\left(\frac{v}{p}\right)}\mathbf 1;\ "
            r"g\,(g^{+}v)=v,\ \mathbf 1^\top g^{+}v=0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.fisher_actuation_arm_a_20260717:"
            "categorical_fisher_pseudoinverse_precondition_law"
        ),
        domain_of_validity={
            "probabilities": "strictly positive categorical simplex rows, K>=2",
            "cotangent": "zero-sum (shift-invariant objective) per row; gauge projected",
            "damping": "eps>=0 added to p pre-division (eps>0 bounds 1/p at simplex corners)",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "measurement_status": "lever built default-OFF; $0 cached-ckpt training A/B OWED",
            "verdict_scope": (
                "DERIVED closed-form identity; NO training-efficacy or score verdict encoded"
            ),
            "sister_surfaces": (
                "categorical_fisher_natural_trust_region_solve_v1 (quotient H^-1 trust SOLVE); "
                "fisher_curvature_equals_categorical_fisher_trace_caustic_v1 (the trace law the "
                "--fisher-density-weight lever evaluates)"
            ),
        },
        units_in={
            "probabilities": "dimensionless",
            "cotangent": "objective_per_logit",
            "eps": "dimensionless_probability_damping",
        },
        units_out={"preconditioned_cotangent": "objective_per_fisher_logit"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={"pseudoinverse_identity": 0.0},
        last_calibration_utc="2026-07-17T20:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments.train_witness_realized_through_R_mlx:make_seg_logits_natural_grad_mlx",
            "tac.witness_dsl.curriculum_dsl:HeadNaturalGradient",
        ),
        canonical_producers=(
            "tac.canonical_equations.fisher_actuation_arm_a_20260717",
            "tac.information_geometry.fisher_natural_solver",
        ),
        provenance=provenance,
    )


__all__ = [
    "EQUATION_ID",
    "build_categorical_fisher_pseudoinverse_cotangent_precondition_v1",
    "categorical_fisher_pseudoinverse_precondition_law",
]
