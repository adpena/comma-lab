# SPDX-License-Identifier: MIT
"""Conditional costate trust-region law and task-454 empirical anchor.

The law deliberately separates a rigorous certificate from the measured
margin/Fisher proxy.  The latter is a training-signal gate and never acquires
score, label-cell, or exact-descent authority by correlation.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "frozen_segnet_costate_trust_region_v1"
_UTC = "2026-07-13T03:10:02Z"
_MEMO = ".omx/research/trust_region_validation_95kill_20260713.md"
_RECEIPT = (
    "experiments/results/costate_trust_region_economics_20260713T032000Z/"
    "measurement_receipt.json"
)
_AXIS = "[macOS-CPU advisory; training-signal economics; no score authority]"

# MEASURED in the source-custodied task-454 receipt.
BASELINE_VALIDATION_FORWARDS = 402
BASELINE_TOTAL_TEACHER_CALLS = 48
NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR = 1.0
PROXY_REUSES = 1
PROXY_CANDIDATES = 64
ACCEPTED_EXACT_CE_DELTA = -1.1175870895385742e-08
ACCEPTED_EXACT_DSEG_DELTA = 0.0

# DERIVED from the measured counts above; these are not independent literals.
BASELINE_VALIDATIONS_PER_TEACHER_CALL = (
    BASELINE_VALIDATION_FORWARDS / BASELINE_TOTAL_TEACHER_CALLS
)
NORMALIZED_VALIDATION_REDUCTION_FACTOR = (
    BASELINE_VALIDATIONS_PER_TEACHER_CALL / NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR
)
NORMALIZED_VALIDATION_REDUCTION_FRACTION = (
    1.0 - NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR / BASELINE_VALIDATIONS_PER_TEACHER_CALL
)


def build_frozen_segnet_costate_trust_region_v1() -> CanonicalEquation:
    """Build the conditional theorem plus its scoped empirical falsifier."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "supply content-bound rigorous suffix pairwise-logit and costate Lipschitz bounds, "
            "a renderer-VJP norm upper bound, and a projected-gradient floor; or demonstrate "
            "nonzero admitted reuse in every registered regime under the same exact shadow gate"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="apple_macos_arm64_cpu_torch_fp32_saved_regime_probe",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="costate_trust_region_pair0_three_regimes_20260713",
        measurement_utc=_UTC,
        inputs={
            "pair": 0,
            "saved_regimes": ["early", "boundary", "late"],
            "baseline_validation_forwards": BASELINE_VALIDATION_FORWARDS,
            "baseline_total_teacher_calls": BASELINE_TOTAL_TEACHER_CALLS,
            "candidate_count": PROXY_CANDIDATES,
            "input_metric": "margin_fisher_rms",
            "validation_cadence": "one exact validation per anchor",
        },
        predicted_output={
            "rigorous_admission": (
                "q(r) remains inside the suffix margin ball and the renderer-projected "
                "costate-error envelope is strictly below the banked-gradient norm floor"
            ),
            "empirical_falsifier": (
                "NO-GO unless at least one proxy reuse is admitted in every early, boundary, "
                "and late regime and every admitted reuse passes a fresh exact-teacher shadow"
            ),
        },
        empirical_output={
            "baseline_validations_per_total_teacher_call": BASELINE_VALIDATIONS_PER_TEACHER_CALL,
            "new_operational_validations_per_anchor": NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR,
            "normalized_validation_reduction_factor": NORMALIZED_VALIDATION_REDUCTION_FACTOR,
            "normalized_validation_reduction_fraction": NORMALIZED_VALIDATION_REDUCTION_FRACTION,
            "proxy_reuses_by_regime": [1, 0, 0],
            "proxy_reuses": PROXY_REUSES,
            "accepted_exact_ce_delta": ACCEPTED_EXACT_CE_DELTA,
            "accepted_exact_dseg_delta": ACCEPTED_EXACT_DSEG_DELTA,
            "accepted_reuses_preserve_exact_teacher_descent": True,
            "rigorous_certificate": "BLOCKED_MISSING_BOUND_ARTIFACTS",
            "empirical_margin_fisher_formulation": "NO-GO_NO_BOUNDARY_OR_LATE_REUSE",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=2.0 / 3.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "three sealed pair-0 saved regimes; one exact anchor validation per regime; two "
            "prefix-only Jacobian probes; O(pixels) margin/Fisher RMS membership checks over "
            "a registered 64-candidate ladder; fresh exact SegNet shadows only for accepted reuse"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=(
            "no across-seed or composed acceptance floor; exact accepted-step CE and d_seg are "
            "reported directly, and the formulation fails regime coverage"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Frozen-SegNet banked-costate trust-region validation law",
        one_line_summary=(
            "Reuse an anchor costate only inside the intersection of its suffix-margin ball "
            "and its renderer-projected exact-descent ball."
        ),
        latex_form=(
            r"q(r)=J_0r+\tfrac12\beta r^2,\quad "
            r"r_m=q^{-1}(\rho_h)=\frac{2\rho_h}{J_0+\sqrt{J_0^2+2\beta\rho_h}},\quad "
            r"E(r)=(J_0+\beta r)\kappa q(r),\quad "
            r"r^*=\sup\{r\le r_m:B_R E(r)<\gamma_\theta\}"
        ),
        python_callable_module_path=(
            "tac.scorer_surrogate.costate_trust_region:derive_costate_trust_region"
        ),
        domain_of_validity={
            "scope_level": "conditional theorem plus registered empirical formulation",
            "research_only": True,
            "included": [
                "content-bound first-block Jacobian norm and neighborhood Lipschitz bounds",
                "content-bound suffix pairwise-logit and costate Lipschitz upper bounds",
                "renderer-VJP norm upper bound and strictly positive projected-gradient floor",
                "current cheap prefix VJP recomputed around one exact banked suffix costate",
                "pair0 empirical early, boundary, and late saved-regime falsification",
            ],
            "excluded": [
                "treating the measured margin/Fisher correlation as a rigorous upper bound",
                "MPS, CUDA, contest-CPU, score, promotion, or pointer authority",
                "live sequence-integrated trainer economics or unseen pairs and seeds",
                "a verdict against trust-region certificates with the missing rigorous bounds supplied",
                "direct reuse of a full input costate without its additional Jacobian-drift term",
            ],
            "fallback": "full exact-teacher refresh on any failed, stale, or non-authoritative gate",
            "authority": _AXIS,
            "review_status": "self-audited-UNREVIEWED_BY_MAIN",
        },
        units_in={
            "r": "input-metric units",
            "rho_h": "first-block feature-norm units",
            "J0": "feature norm per input-metric unit",
            "beta": "feature norm per squared input-metric unit",
            "kappa": "suffix costate norm per feature-norm unit",
            "B_R": "renderer parameter-gradient norm per scorer-input-costate norm",
            "gamma_theta": "banked renderer parameter-gradient norm lower bound",
        },
        units_out={"r_star": "input-metric units", "reuse_decision": "boolean"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"regimes_without_admitted_proxy_reuse_fraction": 2.0 / 3.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.costate_trust_region_policy",
            "tools.probe_costate_trust_region_economics",
        ),
        canonical_producers=("tools.probe_costate_trust_region_economics",),
        provenance=provenance,
    )


def populate_frozen_segnet_costate_trust_region_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked registry helper; never mutate registry bytes directly."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_frozen_segnet_costate_trust_region_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="task454; research_only; rigorous-blocked; empirical-formulation-NO-GO",
    )
    return equation


__all__ = [
    "ACCEPTED_EXACT_CE_DELTA",
    "ACCEPTED_EXACT_DSEG_DELTA",
    "BASELINE_VALIDATIONS_PER_TEACHER_CALL",
    "BASELINE_VALIDATION_FORWARDS",
    "EQUATION_ID",
    "NEW_OPERATIONAL_VALIDATIONS_PER_ANCHOR",
    "NORMALIZED_VALIDATION_REDUCTION_FACTOR",
    "NORMALIZED_VALIDATION_REDUCTION_FRACTION",
    "PROXY_CANDIDATES",
    "PROXY_REUSES",
    "build_frozen_segnet_costate_trust_region_v1",
    "populate_frozen_segnet_costate_trust_region_v1",
]
