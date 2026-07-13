# SPDX-License-Identifier: MIT
"""Decision-quotient target law for a differentiable frozen-SegNet surrogate.

This is a DESIGN-only equation definition.  It deliberately does not append the
shared canonical-equation registry: the registry was already modified by a live
sibling when this file was created.  ``populate_*`` is the explicit main-review
surface after collision review.
"""

from __future__ import annotations

import math

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "segnet_decision_quotient_surrogate_v1"
_MEMO = ".omx/research/jepa_latent_surrogate_20260713.md"
_UTC = "2026-07-13T21:55:00Z"
_AXIS = "[DERIVED/design-only; numpy-fp32 algebra; no empirical or score authority]"


def centered_logits_numpy(logits: np.ndarray, *, class_axis: int = 1) -> np.ndarray:
    """Return the float32 representative of logits modulo common class shift.

    For finite K-class logits ``u``, this applies
    ``P u = u - mean_class(u)``.  Argmax, softmax, and cross entropy are
    invariant to the removed common-class component.  The function is the
    deterministic NumPy-fp32 reference; a trainable implementation still owes
    framework parity and an input-VJP fidelity measurement.
    """

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    axis = int(class_axis)
    if not -values.ndim <= axis < values.ndim:
        raise ValueError(f"class_axis={axis} is invalid for shape {values.shape}")
    if values.shape[axis] < 2:
        raise ValueError("the class axis must contain at least two logits")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    return values - np.mean(values, axis=axis, keepdims=True, dtype=np.float32)


def costate_vjp_error_upper_bound(
    *,
    jacobian_operator_error: float,
    teacher_ce_residual_norm: float,
    student_jacobian_operator_norm: float,
    ce_residual_error_norm: float,
) -> float:
    """Return the derived two-term upper bound on input-costate error.

    If ``g=J_q^T r`` with ``r=softmax(q)-onehot(y)``, then

    ``||g_S-g_T|| <= ||J_S-J_T||op ||r_T|| + ||J_S||op ||r_S-r_T||``.

    This exposes why latent-value matching alone is not a VJP guarantee.
    """

    terms = (
        jacobian_operator_error,
        teacher_ce_residual_norm,
        student_jacobian_operator_norm,
        ce_residual_error_norm,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in terms):
        raise ValueError("all norms must be finite and non-negative")
    return float(
        float(jacobian_operator_error) * float(teacher_ce_residual_norm)
        + float(student_jacobian_operator_norm) * float(ce_residual_error_norm)
    )


def amortized_surrogate_teacher_slice_ms(
    *,
    surrogate_forward_backward_ms: float,
    exact_teacher_forward_backward_ms: float,
    cadence: int,
    anchor_update_ms: float = 0.0,
) -> float:
    """Return per-step teacher-slice cost with one exact anchor every ``cadence``.

    ``C_bar = C_student + (C_teacher + C_anchor_update) / K``.  Renderer VJP,
    PoseNet, and other epoch costs are outside this deliberately scoped slice.
    """

    if not isinstance(cadence, int) or cadence < 1:
        raise ValueError("cadence must be an integer >= 1")
    costs = (
        surrogate_forward_backward_ms,
        exact_teacher_forward_backward_ms,
        anchor_update_ms,
    )
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in costs):
        raise ValueError("costs must be finite and non-negative")
    return float(
        float(surrogate_forward_backward_ms)
        + (float(exact_teacher_forward_backward_ms) + float(anchor_update_ms)) / cadence
    )


def build_segnet_decision_quotient_surrogate_v1() -> CanonicalEquation:
    """Build the exact algebraic target law; empirical admission remains owed."""

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Frozen-SegNet decision quotient and surrogate input-costate law",
        one_line_summary=(
            "Center class logits to remove the softmax gauge, then require value and Jacobian/VJP fidelity before replacing the frozen-SegNet costate."
        ),
        latex_form=(
            r"P=I-\frac{1}{K}\mathbf1\mathbf1^\top,\ q=Pu,\ "
            r"\arg\max u=\arg\max q,\ \mathrm{softmax}(u)=\mathrm{softmax}(q);\quad "
            r"g=J_q^\top r,\ r=\mathrm{softmax}(q)-e_y;\quad "
            r"\|g_S-g_T\|\leq\|J_S-J_T\|_{op}\|r_T\|+\|J_S\|_{op}\|r_S-r_T\|"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.segnet_decision_quotient_surrogate_20260713:centered_logits_numpy"
        ),
        domain_of_validity={
            "included": (
                "finite K-class frozen-SegNet logits; per-pixel softmax cross entropy; "
                "small differentiable RGB-to-quotient student"
            ),
            "excluded": (
                "claim that latent-value fit implies Jacobian or input-costate fidelity; "
                "PoseNet; byte-close score authority; temporal world-model sufficiency"
            ),
            "vehicle": "frozen SegNet last-frame CE costate through the actual R surface",
            "verdict_scope": "TARGET-FORMULATION x DESIGN; no empirical arm was run",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "numpy_fp32_authority": True,
        },
        units_in={
            "u": "per_pixel_class_logits",
            "J_q": "quotient_logit_per_input_pixel",
            "cost": "milliseconds_per_teacher_slice_step",
        },
        units_out={
            "q": "centered_per_pixel_class_logits",
            "g": "loss_per_input_pixel_value",
            "vjp_error_bound": "loss_per_input_pixel_L2_norm",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.onpolicy_scorer_surrogate_policy",
            "tac.witness_dsl.pre_se_locus_policy_20260713",
            "tac.scorer_surrogate.onpolicy_matched_verdict",
        ),
        canonical_producers=(
            "tools.probe_onpolicy_scorer_surrogate",
            "tools.probe_pre_se_locus_20260713",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "append n600 real on-policy value, VJP, signed-descent, in-loop wall-time, and exact "
                "byte-closed anchors before any live activation or throughput claim"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="symbolic_derivation_plus_numpy_fp32_reference",
            captured_at_utc=_UTC,
        ),
    )


def populate_segnet_decision_quotient_surrogate_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Main-review append surface; do not call while the shared registry is held."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_segnet_decision_quotient_surrogate_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="FEED-455/484: centered-logit quotient plus explicit VJP-fidelity gate; design only",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "amortized_surrogate_teacher_slice_ms",
    "build_segnet_decision_quotient_surrogate_v1",
    "centered_logits_numpy",
    "costate_vjp_error_upper_bound",
    "populate_segnet_decision_quotient_surrogate_v1",
]
