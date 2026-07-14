# SPDX-License-Identifier: MIT
"""Canonical guarded K2 exact-costate reuse law (held from shared registry)."""

from __future__ import annotations

import math
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar
from tac.through_r.terminal_costate_skip import (
    EffectiveDimensionCertificate,
    TerminalAction,
    TerminalMethod,
    TerminalReceiptIdentity,
    decide_terminal_costate_skip,
)

EQUATION_ID = "exact_costate_reuse_k2_guarded_v1"
K2 = 2
N_PAIRS = 600
MEMO = ".omx/research/p0_backward_closer_20260713.md"
MEASUREMENT_UTC = "2026-07-14T02:24:28Z"
AXIS = "[macOS-CPU advisory; offline n600 training-signal only; no score authority]"
CORRECTED_WRAPPER = "experiments/results/p0_costate_reuse_k2_n600_v3_20260713/corrected_adjudication_receipt.json"
CORRECTED_WRAPPER_SHA256 = "2102912bc8bd9711f00869746414fb21ea723729bcd26e612274547c6ca73d59"
CORRECTED_RECEIPT = ".omx/research/p0_costate_reuse_k2_corrected_adjudication_receipt_20260714.json"
CORRECTED_RECEIPT_SHA256 = "30ce7e5e23b10cb15c52a89debc57b0bf5349be16ed9cb0e97c3974579465ff7"
FIDELITY_BLOCKED_STATUS = "FIDELITY_BLOCKED_PENDING_NEW_FORMULATION"
TIMING_ELIGIBILITY = "OWED_ONLY_AFTER_NEW_PREREGISTERED_FORMULATION_PASSES_FRESH_N600_FIDELITY"


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite non-boolean number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def amortized_cost_fraction(
    *,
    alpha: float,
    charged_nonaccept_rate: float | None = None,
    cadence: int = K2,
    fallback_rate: float | None = None,
) -> float:
    """Return counterfactual guarded-K2 cost for forward share ``alpha``.

    Every attempted reuse pays the exact forward guard.  Under the sealed
    all-state accounting convention, ``charged_nonaccept_rate`` includes both
    actual guard fallbacks and states that are terminal or blocked before a
    reuse decision.  It must therefore never be reported as a fallback rate.
    Each charged nonaccept pays a complete exact forward plus backward refresh;
    the stale-candidate guard forward cannot be recycled. ``fallback_rate`` is
    the v1 compatibility alias for this *charged nonaccept* rate; it is not an
    actual-guard-fallback rate.
    """

    if isinstance(cadence, bool) or not isinstance(cadence, int) or cadence != K2:
        raise ValueError("this canonical policy is sealed to K=2")
    alpha_value = _unit_interval(alpha, "alpha")
    charged_nonaccept_value = (
        0.0
        if charged_nonaccept_rate is None and fallback_rate is None
        else _unit_interval(
            charged_nonaccept_rate if charged_nonaccept_rate is not None else fallback_rate,
            "charged_nonaccept_rate",
        )
    )
    if fallback_rate is not None:
        legacy_value = _unit_interval(fallback_rate, "fallback_rate")
        if not math.isclose(legacy_value, charged_nonaccept_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fallback_rate alias must equal charged_nonaccept_rate")
    return (1.0 + alpha_value + charged_nonaccept_value) / float(cadence)


def exact_backward_call_amortization(*, reuse_accept_fraction: float) -> float:
    """Return counterfactual, nonadmitted call ratio ``2/(2-a)``."""

    accept_fraction = _unit_interval(reuse_accept_fraction, "reuse_accept_fraction")
    return 2.0 / (2.0 - accept_fraction)


def exact_backward_call_reduction(*, reuse_accept_fraction: float) -> float:
    """Return counterfactual, nonadmitted exact-backward reduction ``p/2``."""

    exact_backward_call_amortization(reuse_accept_fraction=reuse_accept_fraction)
    return _unit_interval(reuse_accept_fraction, "reuse_accept_fraction") / 2.0


def corrected_diagnostic_threshold(*, alpha: float) -> float:
    """Return strict diagnostic ceiling boundary ``p > 3*alpha``."""

    alpha_value = _unit_interval(alpha, "alpha")
    return 3.0 * alpha_value


def full_facet_guard(
    *,
    anchor_ce: float,
    candidate_ce: float,
    anchor_d_seg: float,
    candidate_d_seg: float,
    anchor_d_pose: float,
    candidate_d_pose: float,
) -> bool:
    """Exact admission inequality: strict CE descent and no facet regression."""

    values = (
        anchor_ce,
        candidate_ce,
        anchor_d_seg,
        candidate_d_seg,
        anchor_d_pose,
        candidate_d_pose,
    )
    if not all(
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("all guard values must be finite numbers")
    if any(float(value) < 0.0 for value in values):
        raise ValueError("all guard values must be non-negative")
    return candidate_ce < anchor_ce and candidate_d_seg <= anchor_d_seg and candidate_d_pose <= anchor_d_pose


def terminal_costate_skip_admitted(
    *,
    exact_metric_accept_reject: bool,
    terminal_receipt_path: str | Path | None = None,
    expected_receipt_sha256: str | None = None,
    dimension_certificate_path: str | Path | None = None,
    expected_dimension_certificate_sha256: str | None = None,
    # Legacy scalar claims remain accepted only so stale callers fail closed.
    effective_dimension: int | None = None,
    deterministic_dimension_certificate: bool = False,
    n_pairs: int = 0,
    receipt_custody_valid: bool = False,
    terminal_receipt_sha256: str | None = None,
    effective_dimension_certificate_sha256: str | None = None,
) -> bool:
    """Load durable evidence and delegate terminal admission to the runtime."""

    if exact_metric_accept_reject is True:
        method = TerminalMethod.EXACT_METRIC_MC_396
        admitted_action = TerminalAction.SKIP_COSTATE_EXACT_METRIC_MC
    elif exact_metric_accept_reject is False:
        method = TerminalMethod.SPSA
        admitted_action = TerminalAction.SKIP_COSTATE_DIMENSION_CERTIFIED
    else:
        return False
    if terminal_receipt_path is None or expected_receipt_sha256 is None:
        return False

    try:
        receipt = TerminalReceiptIdentity.from_path(
            terminal_receipt_path,
            expected_sha256=expected_receipt_sha256,
        )
        expected_receipt = TerminalReceiptIdentity.from_path(
            terminal_receipt_path,
            expected_sha256=expected_receipt_sha256,
        )
        dimension_certificate = None
        if method is not TerminalMethod.EXACT_METRIC_MC_396:
            if dimension_certificate_path is None or expected_dimension_certificate_sha256 is None:
                return False
            dimension_certificate = EffectiveDimensionCertificate.from_path(
                dimension_certificate_path,
                expected_sha256=expected_dimension_certificate_sha256,
            )
        decision = decide_terminal_costate_skip(
            method=method,
            receipt=receipt,
            expected_receipt=expected_receipt,
            expected_receipt_sha256=expected_receipt_sha256,
            dimension_certificate=dimension_certificate,
            expected_dimension_certificate_sha256=(
                expected_dimension_certificate_sha256 if dimension_certificate is not None else None
            ),
        )
    except (OSError, TypeError, ValueError):
        return False
    return decision.action is admitted_action and decision.costate_required is False


def exact_costate_reuse_k2_laws(
    *,
    alpha: float,
    anchor_ce: float,
    candidate_ce: float,
    anchor_d_seg: float,
    candidate_d_seg: float,
    anchor_d_pose: float,
    candidate_d_pose: float,
    charged_accept_fraction: float | None = None,
    actual_guard_fallback_fraction: float | None = None,
    terminal_or_blocked_fraction: float | None = None,
    # v1 compatibility aliases. ``fallback_rate`` means charged nonaccept.
    fallback_rate: float | None = None,
    reuse_accept_fraction: float | None = None,
    exact_metric_accept_reject: bool = False,
    terminal_receipt_path: str | Path | None = None,
    expected_receipt_sha256: str | None = None,
    dimension_certificate_path: str | Path | None = None,
    expected_dimension_certificate_sha256: str | None = None,
    # Deprecated claims are forwarded nowhere and cannot authorize a skip.
    effective_dimension: int | None = None,
    deterministic_dimension_certificate: bool = False,
    terminal_n_pairs: int = 0,
    terminal_receipt_custody_valid: bool = False,
    terminal_receipt_sha256: str | None = None,
    effective_dimension_certificate_sha256: str | None = None,
) -> dict[str, float | bool | int | None]:
    """Inject the cost, full-facet guard, and terminal-skip laws together."""

    if charged_accept_fraction is None and reuse_accept_fraction is None:
        accept_fraction = 0.0
    else:
        accept_fraction = _unit_interval(
            charged_accept_fraction if charged_accept_fraction is not None else reuse_accept_fraction,
            "charged_accept_fraction",
        )
    if reuse_accept_fraction is not None:
        legacy_accept_fraction = _unit_interval(reuse_accept_fraction, "reuse_accept_fraction")
        if not math.isclose(legacy_accept_fraction, accept_fraction, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reuse_accept_fraction alias must equal charged_accept_fraction")

    charged_nonaccept_fraction = 1.0 - accept_fraction
    if fallback_rate is not None:
        legacy_charged_nonaccept = _unit_interval(fallback_rate, "fallback_rate")
        if not math.isclose(
            legacy_charged_nonaccept,
            charged_nonaccept_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("fallback_rate alias must equal 1 - charged_accept_fraction")

    component_values = (actual_guard_fallback_fraction, terminal_or_blocked_fraction)
    if all(value is None for value in component_values):
        guard_fallback_fraction = None
        blocked_fraction = None
    elif any(value is None for value in component_values):
        raise ValueError("actual_guard_fallback_fraction and terminal_or_blocked_fraction must be supplied together")
    else:
        guard_fallback_fraction = _unit_interval(actual_guard_fallback_fraction, "actual_guard_fallback_fraction")
        blocked_fraction = _unit_interval(terminal_or_blocked_fraction, "terminal_or_blocked_fraction")
        if not math.isclose(
            guard_fallback_fraction + blocked_fraction,
            charged_nonaccept_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "actual_guard_fallback_fraction + terminal_or_blocked_fraction must equal 1 - charged_accept_fraction"
            )
    cost_fraction = amortized_cost_fraction(
        alpha=alpha,
        charged_nonaccept_rate=charged_nonaccept_fraction,
    )
    return {
        "cadence": K2,
        "n_pairs": N_PAIRS,
        "charged_accept_fraction": accept_fraction,
        "actual_guard_fallback_fraction": guard_fallback_fraction,
        "terminal_or_blocked_fraction": blocked_fraction,
        "charged_nonaccept_fraction": charged_nonaccept_fraction,
        "counterfactual_nonadmitted_amortized_cost_fraction": cost_fraction,
        "counterfactual_nonadmitted_teacher_slice_speedup": 1.0 / cost_fraction,
        "counterfactual_nonadmitted_exact_backward_call_amortization": exact_backward_call_amortization(
            reuse_accept_fraction=accept_fraction
        ),
        "counterfactual_nonadmitted_exact_backward_call_reduction": exact_backward_call_reduction(
            reuse_accept_fraction=accept_fraction
        ),
        "admitted_teacher_slice_speedup": 1.0,
        "admitted_exact_backward_call_reduction": 0.0,
        # v1 output aliases retained. ``fallback_rate`` is charged nonaccept,
        # never the actual-guard-fallback component.
        "reuse_accept_fraction": accept_fraction,
        "fallback_rate": charged_nonaccept_fraction,
        "amortized_cost_fraction": cost_fraction,
        "teacher_slice_speedup": 1.0 / cost_fraction,
        "exact_backward_call_amortization": exact_backward_call_amortization(reuse_accept_fraction=accept_fraction),
        "exact_backward_call_reduction": exact_backward_call_reduction(reuse_accept_fraction=accept_fraction),
        "diagnostic_accept_fraction_threshold_strict_gt": corrected_diagnostic_threshold(alpha=alpha),
        "full_facet_guard_admitted": full_facet_guard(
            anchor_ce=anchor_ce,
            candidate_ce=candidate_ce,
            anchor_d_seg=anchor_d_seg,
            candidate_d_seg=candidate_d_seg,
            anchor_d_pose=anchor_d_pose,
            candidate_d_pose=candidate_d_pose,
        ),
        "terminal_costate_skip_admitted": terminal_costate_skip_admitted(
            exact_metric_accept_reject=exact_metric_accept_reject,
            terminal_receipt_path=terminal_receipt_path,
            expected_receipt_sha256=expected_receipt_sha256,
            dimension_certificate_path=dimension_certificate_path,
            expected_dimension_certificate_sha256=(expected_dimension_certificate_sha256),
        ),
    }


def build_exact_costate_reuse_k2_guarded_v1() -> CanonicalEquation:
    """Build the measured offline NO-GO equation; provider-current stays false."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "a new provider formulation must factor the complete supported SegNet-backed scalar, "
            "preserve exact Pose/non-scorer gradients, pass provider-gradient parity, and earn a new "
            "content-bound n600 admission receipt; the sealed direct raw-ZOH K2 result remains NO-GO"
        ),
        measurement_axis=AXIS,
        hardware_substrate="macOS_CPU_Torch_NumPy_fp32_offline_sealed_replay",
        captured_at_utc=MEASUREMENT_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="p0_costate_reuse_k2_corrected_n600_no_go_20260714",
        measurement_utc=MEASUREMENT_UTC,
        inputs={
            "n_pairs": 600,
            "stage_counts": (200, 200, 200),
            "eligible": 523,
            "terminal_or_blocked": 77,
            "actual_guard_accept": 456,
            "actual_guard_fallback": 67,
            "charged_accept": 456,
            "charged_nonaccept": 144,
            "actual_guard_accept_fraction": 456.0 / 523.0,
            "charged_accept_fraction_p": 0.76,
            "actual_guard_fallback_fraction_all_states": 67.0 / 600.0,
            "terminal_or_blocked_fraction": 77.0 / 600.0,
            "charged_nonaccept_fraction_q": 0.24,
            "forward_share_alpha": 0.1784755863,
            "tracked_receipt_sha256": CORRECTED_RECEIPT_SHA256,
            "embedded_full_wrapper_path": CORRECTED_WRAPPER,
            "embedded_full_wrapper_sha256": CORRECTED_WRAPPER_SHA256,
        },
        predicted_output={
            "counterfactual_nonadmitted_guarded_expected_cost": 1.4184755862999998,
            "counterfactual_nonadmitted_diagnostic_teacher_slice_speedup_x": 1.4099643443401577,
            "counterfactual_nonadmitted_exact_backward_call_amortization_x": 1.6129032258064517,
            "counterfactual_nonadmitted_exact_backward_call_reduction_fraction": 0.38,
            "admitted_teacher_slice_speedup_x": 1.0,
            "admitted_exact_backward_call_reduction_fraction": 0.0,
            "required_accept_fraction_strict_gt": 0.5354267588999999,
            "timing_status": FIDELITY_BLOCKED_STATUS,
            "timing_eligibility": TIMING_ELIGIBILITY,
        },
        empirical_output={
            "corrected_gate_passed": False,
            "corrected_verdict": "NOT_ADMITTED",
            "accepted_d_seg_regret_lte_zero": "308/456",
            "accepted_renderer_gradient_rel_l2_lt_one": "456/456",
            "renderer_gradient_rel_l2_median": 0.03072912052372636,
            "renderer_gradient_rel_l2_p90": 0.0518675255971356,
            "renderer_gradient_rel_l2_max": 0.1432164947042975,
            "pointer_moved": False,
            "provider_current": False,
        },
        residual=0.0,
        source_artifact=CORRECTED_RECEIPT,
        measurement_method=(
            "sealed n600 row replay with recursive custody validation and arithmetic-only "
            "corrected adjudication; DERIVED_DIAGNOSTIC_NOT_IN_LOOP"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Guarded K2 exact-costate reuse with terminal gradient-free handoff",
        one_line_summary=(
            "Corrected n600 K=2 economics are favorable diagnostically, but the direct raw-ZOH policy is NOT_ADMITTED on accepted-row d_seg regret."
        ),
        latex_form=(
            r"q_c=q_{guard\_fallback}+q_{terminal\_or\_blocked}=1-p_c;\quad "
            r"C_2^{cf}(\alpha,q_c)=(1+\alpha+q_c)/2=(2+\alpha-p_c)/2;\quad "
            r"A_B^{cf}(p_c)=2/(2-p_c);\quad p_{c,ceiling}>3\alpha;\quad "
            r"\neg A_{fidelity}\Rightarrow A_B^{admitted}=1\land R_B^{admitted}=0;\quad "
            r"A_{reuse}\iff CE_1<CE_0\land d_{seg,1}\le d_{seg,0}\land d_{pose,1}\le d_{pose,0};\quad "
            r"A_{skip}\iff load_R(path_R,h_R)\land verified_R\land "
            r"[MC_{exact}\lor(load_D(path_D,h_D)\land verified_D\land r_{eff}\le2)]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.exact_costate_reuse_k2_20260713:exact_costate_reuse_k2_laws"
        ),
        domain_of_validity={
            "research_only": True,
            "included": (
                "one stale exact-input-costate attempt after each exact anchor; K=2; n600; "
                "content-bound costate/frame/objective/scorer identities; exact forward-only full-facet guard"
            ),
            "refresh_boundaries": ("event", "stage", "custody_change"),
            "terminal_skip": (
                "#396 exact-metric accept/reject only after runtime validation of durable n600 "
                "receipt bytes against a code-reviewed SHA-256; SPSA/ES additionally requires "
                "runtime validation of durable deterministic effective-dimension evidence <=2"
            ),
            "excluded": (
                "K>2; n<600 admission; cosine-only admission; proxy-only guard; blind cadence; "
                "caller-asserted custody/dimension/determinism without validated durable bytes; "
                "bulk SPSA/ES; provider-current, score, pointer, or promotion claims"
            ),
            "fallback": "rollback and full_teacher_refresh; #396 ordinary route at terminal",
            "verdict": "NO_GO_NOT_ADMITTED",
            "verdict_scope": (
                "bounded direct raw-input-costate zero-order-hold K2 policy on the sealed primary "
                "SegNet CE scalar; sibling costate/provider families and a factored full live scalar remain open"
            ),
            "diagnostic_economics_authority": "DERIVED_DIAGNOSTIC_NOT_IN_LOOP",
            "timing_status": FIDELITY_BLOCKED_STATUS,
            "timing_eligibility": TIMING_ELIGIBILITY,
            "provider_current": False,
            "pointer_moved": False,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "alpha": "forward_fraction_of_exact_teacher_call",
            "charged_accept_fraction": "accepted_states_per_all_charged_states",
            "actual_guard_fallback_fraction": "guard_fallback_states_per_all_charged_states",
            "terminal_or_blocked_fraction": "terminal_or_blocked_states_per_all_charged_states",
            "reuse_accept_fraction": "v1_alias_of_charged_accept_fraction",
            "fallback_rate": "v1_alias_of_charged_nonaccept_fraction_not_actual_guard_fallback",
            "CE": "cross_entropy_loss",
            "d_seg": "exact_through_R_segmentation_distance",
            "d_pose": "exact_through_R_pose_distance",
            "terminal_receipt_path": "durable_terminal_receipt_path",
            "expected_receipt_sha256": "code_reviewed_terminal_receipt_identity",
            "dimension_certificate_path": "durable_effective_dimension_certificate_path",
            "expected_dimension_certificate_sha256": ("code_reviewed_effective_dimension_certificate_identity"),
        },
        units_out={
            "C_2_counterfactual_nonadmitted": "fraction_of_exact_per_step_cost",
            "counterfactual_nonadmitted_teacher_slice_speedup": "dimensionless_ratio",
            "counterfactual_nonadmitted_exact_backward_call_amortization": "dimensionless_ratio",
            "counterfactual_nonadmitted_exact_backward_call_reduction": "fraction",
            "admitted_teacher_slice_speedup": "dimensionless_ratio_fixed_at_1_when_not_admitted",
            "admitted_exact_backward_call_reduction": "fraction_fixed_at_0_when_not_admitted",
            "amortized_cost_fraction": "v1_alias_of_counterfactual_nonadmitted_C_2",
            "teacher_slice_speedup": "v1_alias_of_counterfactual_nonadmitted_speedup",
            "exact_backward_call_amortization": "v1_alias_of_counterfactual_nonadmitted_call_ratio",
            "exact_backward_call_reduction": "v1_alias_of_counterfactual_nonadmitted_reduction",
            "full_facet_guard": "boolean",
            "terminal_costate_skip": "boolean",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "corrected_arithmetic_residual": 0.0,
            "accepted_d_seg_violation_fraction": 148.0 / 456.0,
        },
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.exact_costate_reuse",
            "tac.witness_dsl.exact_costate_reuse_policy",
            "tac.through_r.terminal_costate_skip",
        ),
        canonical_producers=(
            "tools.probe_p0_costate_reuse_k2",
            "tools.adjudicate_p0_costate_reuse_k2",
        ),
        provenance=provenance,
    )


def populate_exact_costate_reuse_k2_guarded_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Explicit main-review registration surface; never called at import time."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_exact_costate_reuse_k2_guarded_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "FEED-p0-backward-wave; corrected n600 NO-GO; provider-current=false; "
            "pointer_moved=false; fidelity-blocked pending a new preregistered formulation"
        ),
    )
    return equation


__all__ = [
    "AXIS",
    "CORRECTED_RECEIPT",
    "CORRECTED_RECEIPT_SHA256",
    "CORRECTED_WRAPPER",
    "CORRECTED_WRAPPER_SHA256",
    "EQUATION_ID",
    "FIDELITY_BLOCKED_STATUS",
    "K2",
    "MEMO",
    "N_PAIRS",
    "TIMING_ELIGIBILITY",
    "amortized_cost_fraction",
    "build_exact_costate_reuse_k2_guarded_v1",
    "corrected_diagnostic_threshold",
    "exact_backward_call_amortization",
    "exact_backward_call_reduction",
    "exact_costate_reuse_k2_laws",
    "full_facet_guard",
    "populate_exact_costate_reuse_k2_guarded_v1",
    "terminal_costate_skip_admitted",
]
