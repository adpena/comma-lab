# SPDX-License-Identifier: MIT
"""Canonical guarded K2 exact-costate reuse law (held from shared registry)."""

from __future__ import annotations

import math

from tac.canonical_equations.equation import RECALIBRATE_ON_NEW_ANCHORS, CanonicalEquation
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "exact_costate_reuse_k2_guarded_v1"
K2 = 2
N_PAIRS = 600
MEMO = ".omx/research/p0_backward_closer_20260713.md"
MEASUREMENT_UTC = "2026-07-13T22:30:00Z"
AXIS = "[DERIVED guarded policy; receipt admission pending; no score authority]"


def amortized_cost_fraction(
    *, alpha: float, fallback_rate: float = 0.0, cadence: int = K2
) -> float:
    """Return guarded K2 teacher-compute fraction for forward share ``alpha``.

    Every attempted reuse pays the exact forward guard.  A rejected fraction
    ``fallback_rate`` additionally pays the exact backward refresh.
    """

    if cadence != K2:
        raise ValueError("this canonical policy is sealed to K=2")
    if not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not isinstance(fallback_rate, (int, float)) or not math.isfinite(float(fallback_rate)):
        raise ValueError("fallback_rate must be finite")
    if not 0.0 <= float(fallback_rate) <= 1.0:
        raise ValueError("fallback_rate must be in [0, 1]")
    return float(alpha) + (1.0 - float(alpha)) * (1.0 + float(fallback_rate)) / float(cadence)


def exact_backward_call_amortization(*, reuse_accept_fraction: float) -> float:
    """Return exact-backward call ratio ``2/(2-a)`` for accept fraction ``a``."""

    if not isinstance(reuse_accept_fraction, (int, float)) or not math.isfinite(
        float(reuse_accept_fraction)
    ):
        raise ValueError("reuse_accept_fraction must be finite")
    if not 0.0 <= float(reuse_accept_fraction) <= 1.0:
        raise ValueError("reuse_accept_fraction must be in [0, 1]")
    return 2.0 / (2.0 - float(reuse_accept_fraction))


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
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        raise ValueError("all guard values must be finite numbers")
    if any(float(value) < 0.0 for value in values):
        raise ValueError("all guard values must be non-negative")
    return (
        candidate_ce < anchor_ce
        and candidate_d_seg <= anchor_d_seg
        and candidate_d_pose <= anchor_d_pose
    )


def terminal_costate_skip_admitted(
    *,
    exact_metric_accept_reject: bool,
    effective_dimension: int | None,
    deterministic_dimension_certificate: bool,
    n_pairs: int,
    receipt_custody_valid: bool,
) -> bool:
    """Encode the #396-versus-SPSA/ES distinction in the equation surface."""

    if n_pairs != N_PAIRS or receipt_custody_valid is not True:
        return False
    if exact_metric_accept_reject:
        return True
    return (
        not isinstance(effective_dimension, bool)
        and isinstance(effective_dimension, int)
        and 0 <= effective_dimension <= 2
        and deterministic_dimension_certificate is True
    )


def exact_costate_reuse_k2_laws(
    *,
    alpha: float,
    anchor_ce: float,
    candidate_ce: float,
    anchor_d_seg: float,
    candidate_d_seg: float,
    anchor_d_pose: float,
    candidate_d_pose: float,
    fallback_rate: float = 0.0,
    reuse_accept_fraction: float = 0.0,
    exact_metric_accept_reject: bool = False,
    effective_dimension: int | None = None,
    deterministic_dimension_certificate: bool = False,
    terminal_n_pairs: int = 0,
    terminal_receipt_custody_valid: bool = False,
) -> dict[str, float | bool | int]:
    """Inject the cost, full-facet guard, and terminal-skip laws together."""

    return {
        "cadence": K2,
        "n_pairs": N_PAIRS,
        "amortized_cost_fraction": amortized_cost_fraction(
            alpha=alpha, fallback_rate=fallback_rate
        ),
        "teacher_slice_speedup": 1.0
        / amortized_cost_fraction(alpha=alpha, fallback_rate=fallback_rate),
        "exact_backward_call_amortization": exact_backward_call_amortization(
            reuse_accept_fraction=reuse_accept_fraction
        ),
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
            effective_dimension=effective_dimension,
            deterministic_dimension_certificate=deterministic_dimension_certificate,
            n_pairs=terminal_n_pairs,
            receipt_custody_valid=terminal_receipt_custody_valid,
        ),
    }


def build_exact_costate_reuse_k2_guarded_v1() -> CanonicalEquation:
    """Build the held equation; no empirical provider-current claim is made."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "append completed content-bound n600 temporal-fidelity receipt with exact forward-only "
            "CE/d_seg/d_pose guards, charged in-loop timing, rollback/resume proof, and scorer/objective hashes"
        ),
        measurement_axis=AXIS,
        hardware_substrate="symbolic_derivation_only_no_provider_current",
        captured_at_utc=MEASUREMENT_UTC,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Guarded K2 exact-costate reuse with terminal gradient-free handoff",
        one_line_summary=(
            "K=2 reuse is admissible only under exact CE/d_seg/d_pose guards; #396 needs no gradient, while SPSA/ES needs certified dimension <=2."
        ),
        latex_form=(
            r"C_2(f,q)=f+(1-f)(1+q)/2;\quad A_B(a)=2/(2-a);\quad "
            r"A_{reuse}\iff CE_1<CE_0\land d_{seg,1}\le d_{seg,0}\land d_{pose,1}\le d_{pose,0};\quad "
            r"A_{skip}\iff MC_{exact}\lor(r_{eff}\le2\land cert_{det})"
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
                "#396 exact-metric accept/reject after matching n600 receipt; SPSA/ES only with "
                "deterministic effective_dimension_certificate <=2"
            ),
            "excluded": (
                "K>2; n<600 admission; cosine-only admission; proxy-only guard; blind cadence; "
                "bulk SPSA/ES; provider-current, score, pointer, or promotion claims"
            ),
            "fallback": "rollback and full_teacher_refresh; #396 ordinary route at terminal",
            "verdict_scope": "bounded direct raw-ZOH K2 policy; sibling provider families remain open",
            "provider_current": False,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "alpha": "forward_fraction_of_exact_teacher_call",
            "fallback_rate": "fraction_of_reuse_attempts_forcing_exact_backward_refresh",
            "reuse_accept_fraction": "fraction_of_reuse_attempts_accepted",
            "CE": "cross_entropy_loss",
            "d_seg": "exact_through_R_segmentation_distance",
            "d_pose": "exact_through_R_pose_distance",
            "effective_dimension": "active_search_coordinates",
        },
        units_out={
            "C_2": "fraction_of_exact_per_step_cost",
            "teacher_slice_speedup": "dimensionless_ratio",
            "exact_backward_call_amortization": "dimensionless_ratio",
            "full_facet_guard": "boolean",
            "terminal_costate_skip": "boolean",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc=MEASUREMENT_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_control.exact_costate_reuse",
            "tac.witness_dsl.exact_costate_reuse_policy",
            "tac.through_r.terminal_costate_skip",
        ),
        canonical_producers=("tools.probe_p0_costate_reuse_k2",),
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
        notes="FEED-p0-backward-wave; guarded K2 reuse; provider-current=false; receipt pending",
    )
    return equation


__all__ = [
    "AXIS",
    "EQUATION_ID",
    "K2",
    "MEMO",
    "N_PAIRS",
    "amortized_cost_fraction",
    "build_exact_costate_reuse_k2_guarded_v1",
    "exact_backward_call_amortization",
    "exact_costate_reuse_k2_laws",
    "full_facet_guard",
    "populate_exact_costate_reuse_k2_guarded_v1",
    "terminal_costate_skip_admitted",
]
