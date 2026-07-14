# SPDX-License-Identifier: MIT
"""Whole-teacher student quotient, VJP-fidelity, and economics laws.

This standalone equation leg is research-only and carries no empirical anchor.
It imports the student's single NumPy-fp32 Helmert quotient implementation so
the math, model, and measurement harness cannot silently choose different
four-dimensional gauges.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar
from tac.scorer_surrogate.whole_teacher_distilled_student import (
    HELMERT_BASIS_5X4,
    logits5_from_quotient4_numpy,
    quotient4_from_logits5_numpy,
)

EQUATION_ID = "whole_teacher_distilled_student_fidelity_economics_v1"
MEMO = ".omx/research/whole_teacher_distilled_student_20260713.md"
MEASUREMENT_UTC = "2026-07-13T23:50:00Z"
AXIS = "[DERIVED; NumPy-fp32 algebra; n600 empirical anchor absent; no score authority]"
VERDICT_SCOPE = (
    "INSTANCE x INPUT-CACHE x STUDENT-SIZE x K_student; no negative from a missing or "
    "failed bundle closes the whole-teacher distilled-student family"
)
REQ_R = (
    "content-bound real n600 rendered states through the actual R surface, exact teacher "
    "decision quotients and full input VJPs, worst-pair gates, and matched-device fully "
    "charged in-loop timing"
)


def project_centered_quotient_numpy(logits5: np.ndarray, *, class_axis: int = 1) -> np.ndarray:
    """Return the canonical four-dimensional Helmert quotient coordinates.

    ``B.T @ u`` automatically removes the common-logit gauge because every
    Helmert column is orthogonal to the all-ones vector.
    """

    return quotient4_from_logits5_numpy(logits5, class_axis=class_axis)


def lift_centered_quotient_numpy(quotient4: np.ndarray, *, class_axis: int = 1) -> np.ndarray:
    """Lift four quotient coordinates to the canonical zero-sum five logits."""

    return logits5_from_quotient4_numpy(quotient4, class_axis=class_axis)


def _paired_finite_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    pair_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"reference/candidate shape mismatch: {ref.shape} != {cand.shape}")
    if ref.ndim < 1 or ref.size == 0:
        raise ValueError("fidelity arrays must be nonempty")
    axis = int(pair_axis)
    if not -ref.ndim <= axis < ref.ndim:
        raise ValueError(f"pair_axis={axis} is invalid for shape {ref.shape}")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("fidelity arrays must be finite")
    ref_pairs = np.moveaxis(ref, axis, 0).reshape(ref.shape[axis], -1)
    cand_pairs = np.moveaxis(cand, axis, 0).reshape(cand.shape[axis], -1)
    return ref_pairs, cand_pairs


def _pair_cosine_and_relative_l2(
    reference_pairs: np.ndarray, candidate_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ref64 = reference_pairs.astype(np.float64, copy=False)
    cand64 = candidate_pairs.astype(np.float64, copy=False)
    diff = cand64 - ref64
    ref_norm = np.linalg.norm(ref64, axis=1)
    cand_norm = np.linalg.norm(cand64, axis=1)
    if np.any(ref_norm == 0.0) or np.any(cand_norm == 0.0):
        raise ValueError("cosine/relative-L2 fidelity is undefined for a zero teacher or student vector")
    diff_norm = np.linalg.norm(diff, axis=1)
    product = ref_norm * cand_norm
    cosine = np.sum(ref64 * cand64, axis=1) / product
    relative_l2 = diff_norm / ref_norm
    return cosine, relative_l2


def pairwise_fidelity_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    pair_axis: int = 0,
) -> dict[str, float | int]:
    """Return aggregate and worst-pair cosine/relative-L2 fidelity.

    Accumulation is float64 only for stable metric reduction; both compared
    fields are first canonicalized to the NumPy-fp32 authority surface.
    Any zero teacher or student vector makes cosine undefined and therefore
    fails closed instead of manufacturing a perfect or finite fidelity row.
    """

    ref_pairs, cand_pairs = _paired_finite_arrays(reference, candidate, pair_axis=pair_axis)
    cosine, relative_l2 = _pair_cosine_and_relative_l2(ref_pairs, cand_pairs)
    worst_cosine_pair = int(np.argmin(cosine))
    worst_relative_l2_pair = int(np.argmax(relative_l2))
    return {
        "n_pairs": int(ref_pairs.shape[0]),
        "mean_cosine": float(np.mean(cosine, dtype=np.float64)),
        "worst_cosine": float(cosine[worst_cosine_pair]),
        "worst_cosine_pair": worst_cosine_pair,
        "mean_relative_l2": float(np.mean(relative_l2, dtype=np.float64)),
        "worst_relative_l2": float(relative_l2[worst_relative_l2_pair]),
        "worst_relative_l2_pair": worst_relative_l2_pair,
    }


def vjp_fidelity_summary(
    exact_teacher_input_vjp: np.ndarray,
    student_input_vjp: np.ndarray,
    *,
    pair_axis: int = 0,
) -> dict[str, float | int | str | bool]:
    """Return the decisive full-vector input-VJP fidelity summary."""

    summary: dict[str, float | int | str | bool] = pairwise_fidelity_summary(
        exact_teacher_input_vjp,
        student_input_vjp,
        pair_axis=pair_axis,
    )
    summary["authority_surface"] = "full_exact_teacher_input_vjp"
    summary["diagnostic_boundary_restriction_is_authority"] = False
    return summary


def _nonnegative_finite(value: float, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive_cadence(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("student_anchor_cadence must be an integer >= 1")
    return value


def amortized_student_teacher_cost_ms(
    *,
    student_cost_ms: float,
    exact_teacher_cost_ms: float,
    anchor_update_cost_ms: float,
    student_anchor_cadence: int,
) -> float:
    """Compute ``C_S + (C_T + U) / K_student`` with every charged term."""

    student = _nonnegative_finite(student_cost_ms, name="student_cost_ms")
    teacher = _nonnegative_finite(exact_teacher_cost_ms, name="exact_teacher_cost_ms")
    update = _nonnegative_finite(anchor_update_cost_ms, name="anchor_update_cost_ms")
    cadence = _positive_cadence(student_anchor_cadence)
    return student + (teacher + update) / cadence


def minimum_student_anchor_cadence_for_remaining_fraction(
    *,
    student_cost_ms: float,
    exact_teacher_cost_ms: float,
    anchor_update_cost_ms: float,
    target_remaining_fraction: float,
) -> dict[str, float | int | bool | None]:
    """Derive the minimum integer K for a requested teacher-slice remainder.

    The inequality is

    ``K >= (C_T + U) / (rho*C_T - C_S)``,

    and is infeasible for every finite cadence when the denominator is not
    strictly positive.
    """

    student = _nonnegative_finite(student_cost_ms, name="student_cost_ms")
    teacher = _nonnegative_finite(exact_teacher_cost_ms, name="exact_teacher_cost_ms")
    update = _nonnegative_finite(anchor_update_cost_ms, name="anchor_update_cost_ms")
    if teacher <= 0.0:
        raise ValueError("exact_teacher_cost_ms must be strictly positive")
    if (
        not isinstance(target_remaining_fraction, (int, float))
        or isinstance(target_remaining_fraction, bool)
        or not math.isfinite(float(target_remaining_fraction))
        or not 0.0 < float(target_remaining_fraction) < 1.0
    ):
        raise ValueError("target_remaining_fraction must lie in (0, 1)")
    remaining = float(target_remaining_fraction)
    denominator = remaining * teacher - student
    if denominator <= 0.0:
        return {
            "finite_cadence_feasible": False,
            "minimum_cadence_real": None,
            "minimum_cadence_integer": None,
            "remaining_budget_ms": remaining * teacher,
        }
    minimum_real = (teacher + update) / denominator
    return {
        "finite_cadence_feasible": True,
        "minimum_cadence_real": minimum_real,
        "minimum_cadence_integer": max(1, math.ceil(minimum_real)),
        "remaining_budget_ms": remaining * teacher,
    }


def surrogate_economics(
    *,
    tier: str,
    student_cost_ms: float,
    exact_teacher_cost_ms: float,
    anchor_update_cost_ms: float,
    student_anchor_cadence: int,
    fidelity_gate_passed: bool,
    charged_timing_measured: bool,
    exact_costate_reuse_kmax: int | None = None,
) -> dict[str, float | int | bool | str | None]:
    """Return one fully charged tier/K row without importing a #487 speedup.

    ``exact_costate_reuse_kmax`` is composition metadata only.  When present it
    must be exactly two and does not alter ``K_student`` or any cost in this
    law; the sibling controller must supply its own measured accounting.
    """

    if tier not in {"forward_advisory", "training_gradient"}:
        raise ValueError("tier must be forward_advisory or training_gradient")
    if not isinstance(fidelity_gate_passed, bool):
        raise ValueError("fidelity_gate_passed must be boolean")
    if not isinstance(charged_timing_measured, bool):
        raise ValueError("charged_timing_measured must be boolean")
    if exact_costate_reuse_kmax not in (None, 2):
        raise ValueError("optional exact-costate reuse composition is sealed to K_max=2")
    teacher = _nonnegative_finite(exact_teacher_cost_ms, name="exact_teacher_cost_ms")
    if teacher <= 0.0:
        raise ValueError("exact_teacher_cost_ms must be strictly positive")
    cadence = _positive_cadence(student_anchor_cadence)
    charged = amortized_student_teacher_cost_ms(
        student_cost_ms=student_cost_ms,
        exact_teacher_cost_ms=teacher,
        anchor_update_cost_ms=anchor_update_cost_ms,
        student_anchor_cadence=cadence,
    )
    cost_pays = charged < teacher
    inclusive_95_cost_feasible = charged <= 0.05 * teacher
    admitted_pays = charged_timing_measured and fidelity_gate_passed and cost_pays
    admitted_inclusive_95 = charged_timing_measured and fidelity_gate_passed and inclusive_95_cost_feasible
    return {
        "tier": tier,
        "student_anchor_cadence": cadence,
        "exact_costate_reuse_kmax": exact_costate_reuse_kmax,
        "exact_costate_reuse_speed_claim_imported": False,
        "charged_cost_ms_per_step": charged,
        "exact_teacher_baseline_ms_per_step": teacher,
        "charged_fraction_of_teacher": charged / teacher,
        "teacher_slice_speedup": teacher / charged if charged > 0.0 else math.inf,
        "cost_pays": cost_pays,
        "strict_pays": admitted_pays,
        "inclusive_95_cost_feasible": inclusive_95_cost_feasible,
        "inclusive_95": admitted_inclusive_95,
        "fidelity_gate_passed": fidelity_gate_passed,
        "charged_timing_measured": charged_timing_measured,
        "pays": admitted_pays,
        "status": "STUDENT_PAYS" if admitted_pays else "NO_PAY_AUTHORITY",
    }


def whole_teacher_distilled_student_laws(
    *,
    logits5: np.ndarray,
    exact_teacher_input_vjp: np.ndarray,
    student_input_vjp: np.ndarray,
    student_cost_ms: float,
    exact_teacher_cost_ms: float,
    anchor_update_cost_ms: float,
    student_anchor_cadence: int,
) -> dict[str, Any]:
    """Compose quotient, VJP, and charged economics into one callable law."""

    quotient = project_centered_quotient_numpy(logits5)
    lifted = lift_centered_quotient_numpy(quotient)
    return {
        "equation_id": EQUATION_ID,
        "quotient4": quotient,
        "zero_sum_logits5": lifted,
        "vjp_fidelity": vjp_fidelity_summary(exact_teacher_input_vjp, student_input_vjp),
        "charged_cost_ms": amortized_student_teacher_cost_ms(
            student_cost_ms=student_cost_ms,
            exact_teacher_cost_ms=exact_teacher_cost_ms,
            anchor_update_cost_ms=anchor_update_cost_ms,
            student_anchor_cadence=student_anchor_cadence,
        ),
    }


def build_whole_teacher_distilled_student_fidelity_economics_v1() -> CanonicalEquation:
    """Build the unanchored law; real n600 evidence remains explicitly owed."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "append a content-bound real n600 rendered-state receipt with worst-pair forward "
            "and full exact input-VJP metrics plus matched-device fully charged timing; then "
            "measure a governed in-loop matched window before any activation claim"
        ),
        measurement_axis=AXIS,
        hardware_substrate="symbolic_plus_numpy_fp32_no_empirical_student_receipt",
        captured_at_utc=MEASUREMENT_UTC,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Whole-teacher decision-quotient student fidelity and cadence economics",
        one_line_summary=(
            "Use one canonical four-dimensional softmax quotient, gate the full input VJP, "
            "and admit a tier only when C_S+(C_T+U)/K_student is fully measured and pays."
        ),
        latex_form=(
            r"B^TB=I_4,\ B^T\mathbf1=0,\ z=B^Tu,\ q=Bz;\quad "
            r"\cos(z_S,z_T),\ \epsilon_q=\|z_S-z_T\|_2/\|z_T\|_2;\quad "
            r"\cos(g_S,g_T),\ \epsilon_{VJP}=\|g_S-g_T\|_2/\|g_T\|_2;\quad "
            r"\bar C_t(s,K)=C_{S,t}(s)+(C_{T,t}+U(s))/K"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.whole_teacher_distilled_student_20260713:whole_teacher_distilled_student_laws"
        ),
        domain_of_validity={
            "research_only": True,
            "included": (
                "frozen five-class SegNet; canonical Helmert R5/<1> quotient; real rendered "
                "states through actual R; full exact teacher input VJP; separate forward and "
                "training-gradient tiers; periodic exact whole-teacher anchors"
            ),
            "excluded": (
                "value-only admission for training gradients; boundary-only VJP as authority; "
                "source-video substitution; n<600 evidence; MPS or MLX score authority; "
                "trainer activation; score or pointer movement"
            ),
            "student_anchor_cadence_law": ("K_student is independent of optional #487 exact_costate_reuse_kmax=2"),
            "semantic_custody": (
                "admission requires distinct hashes for actual R, frozen teacher weights, "
                "scalar objective/reduction, source custody, cache, policy, layout, and parameters"
            ),
            "economics_verdicts": ("strict_pays means C_bar<C_T; inclusive_95 separately means C_bar<=0.05*C_T"),
            "verdict_scope": VERDICT_SCOPE,
            "req_R": REQ_R,
            "empirical_status": "UNMEASURED_BLOCKED_INPUT_CACHE",
            "score_claim": False,
            "promotion_eligible": False,
            "numpy_fp32_authority": True,
        },
        units_in={
            "u": "five_class_logits_per_pixel",
            "g": "loss_per_render_frame_value",
            "C_S": "milliseconds_per_student_tier_step",
            "C_T": "milliseconds_per_exact_teacher_tier_step",
            "U": "milliseconds_per_exact_anchor_student_update",
            "K_student": "training_steps_per_exact_whole_teacher_anchor",
        },
        units_out={
            "z": "four_orthonormal_decision_quotient_coordinates_per_pixel",
            "vjp_cosine": "dimensionless",
            "vjp_relative_l2": "dimensionless",
            "C_bar": "milliseconds_per_amortized_tier_step",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.whole_teacher_distilled_student_policy",
            "tac.scorer_surrogate.whole_teacher_distilled_student",
            "tools.probe_whole_teacher_distilled_student",
        ),
        canonical_producers=("tools.probe_whole_teacher_distilled_student",),
        provenance=provenance,
    )


def populate_whole_teacher_distilled_student_fidelity_economics_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicit main-review registration surface; never invoked on import."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_whole_teacher_distilled_student_fidelity_economics_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "FEED-455 whole-teacher student; unanchored n600 VJP/economics law; "
            "shared registry append requires main review"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "EQUATION_ID",
    "HELMERT_BASIS_5X4",
    "MEASUREMENT_UTC",
    "MEMO",
    "REQ_R",
    "VERDICT_SCOPE",
    "amortized_student_teacher_cost_ms",
    "build_whole_teacher_distilled_student_fidelity_economics_v1",
    "lift_centered_quotient_numpy",
    "minimum_student_anchor_cadence_for_remaining_fraction",
    "pairwise_fidelity_summary",
    "populate_whole_teacher_distilled_student_fidelity_economics_v1",
    "project_centered_quotient_numpy",
    "surrogate_economics",
    "vjp_fidelity_summary",
    "whole_teacher_distilled_student_laws",
]
