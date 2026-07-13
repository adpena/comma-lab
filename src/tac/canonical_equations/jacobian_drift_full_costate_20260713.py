# SPDX-License-Identifier: MIT
"""Conditional Jacobian-drift law for direct full-input-costate reuse."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "jacobian_drift_full_costate_v1"
_MEMO = ".omx/research/jacobian_drift_certificate_95kill_20260713.md"
_UTC = "2026-07-13T04:08:30Z"
_RECEIPT = (
    "experiments/results/jacobian_drift_certificate_20260713T034951Z/"
    "measurement_receipt.json"
)
_RECEIPT_SHA256 = "c1a2431ebe9df21a370748f864f2da81a5f242544051986ed341d59fe1518d48"
_AXIS = "[macOS-CPU advisory; torch-fp32 training signal; numpy-fp32 d_seg shadow]"

# MEASURED fields recorded directly in the source-bundled terminal receipt.
# These are sampled points on one exact-gradient ray per regime, not
# sequence-integrated operational reuse.
SAMPLED_CANDIDATES_BY_REGIME = (22, 21, 21)
ORACLE_SAFE_PREFIX_BY_REGIME = (6, 10, 0)
ORACLE_SAFE_RADIUS_L2_BY_REGIME = (1.3887383937835693, 5.644960403442383, 0.0)
RIGOROUS_CERTIFIED_REUSES = 0
HVP_INCREMENTAL_MEDIAN_SECONDS = 3.350555353972595

# DERIVED from the measured terminal candidate/timing rows.
CURRENT_DESCENT_PREFIX_BY_REGIME = (17, 10, 17)
CURRENT_DESCENT_RADIUS_L2_BY_REGIME = (103.02526092529297, 5.644960403442383, 147.25413513183594)
MATCHED_THROUGH_R_VALIDATION_MEDIAN_SECONDS = 1.3124769580317661
EARLY_FAITHFUL_LOWER_BOUND_EQUIVALENTS_PER_ANCHOR = 16.32124488283467
BOUNDARY_FAITHFUL_LOWER_BOUND_EQUIVALENTS_PER_ANCHOR = 26.430512487347823

# DERIVED from the measured terminal receipt or its sealed task-454 parent.
BASELINE_VALIDATIONS_PER_TEACHER_CALL = 402 / 48
HVP_LOWER_BOUND_MATCHED_VALIDATION_EQUIVALENTS = (
    HVP_INCREMENTAL_MEDIAN_SECONDS / MATCHED_THROUGH_R_VALIDATION_MEDIAN_SECONDS
)
SAMPLED_SAFE_PREFIX_TOTAL = sum(ORACLE_SAFE_PREFIX_BY_REGIME)
SAMPLED_CANDIDATE_TOTAL = sum(SAMPLED_CANDIDATES_BY_REGIME)


def build_jacobian_drift_full_costate_v1() -> CanonicalEquation:
    """Build the conditional theorem and its scoped empirical characterization."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_MEMO,
        reactivation_criteria=(
            "supply custody-bearing whole-ball B_J, B_H, Lip(DJ), Q_a, L_q, L_c, "
            "current corrected renderer-gradient floor, rigorous geometry and norm conversion, "
            "full-SegNet C2,1 activation-cell proof, and correction numerical-error bound"
        ),
        measurement_axis=_AXIS,
        hardware_substrate="substrate_independent_conditional_bound",
        captured_at_utc=_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="jacobian_drift_pair0_three_regimes_20260713",
        measurement_utc=_UTC,
        inputs={
            "pair": 0,
            "saved_regimes": ["early", "boundary", "late"],
            "sampled_candidates_by_regime": list(SAMPLED_CANDIDATES_BY_REGIME),
            "candidate_geometry": "one exact-anchor-gradient ray per regime",
            "correction": "faithful per-candidate fixed-adjoint HVP on actual through-R displacement",
            "receipt_sha256": _RECEIPT_SHA256,
            "baseline_validations_per_teacher_call": BASELINE_VALIDATIONS_PER_TEACHER_CALL,
        },
        predicted_output={
            "rigorous_reuse": (
                "CERTIFIED_REUSE only inside the strict self-adjusting whole-ball radius with "
                "complete curvature, geometry, norm, correction, descent, and tensor custody"
            ),
            "economics_gate": (
                "total correction plus anchor-validation cost must be meaningfully below 8.375 "
                "matched validation equivalents per anchor"
            ),
        },
        empirical_output={
            "rigorous_certified_reuses": RIGOROUS_CERTIFIED_REUSES,
            "rigorous_certificate": "BLOCKED_MISSING_WHOLE_BALL_BOUND_CUSTODY",
            "oracle_safe_prefix_by_regime": list(ORACLE_SAFE_PREFIX_BY_REGIME),
            "oracle_safe_radius_l2_by_regime": list(ORACLE_SAFE_RADIUS_L2_BY_REGIME),
            "current_ce_dseg_descent_prefix_by_regime": list(CURRENT_DESCENT_PREFIX_BY_REGIME),
            "current_ce_dseg_descent_radius_l2_by_regime": list(
                CURRENT_DESCENT_RADIUS_L2_BY_REGIME
            ),
            "sampled_safe_prefix_total": SAMPLED_SAFE_PREFIX_TOTAL,
            "sampled_candidate_total": SAMPLED_CANDIDATE_TOTAL,
            "oracle_prefix_authority": "post_hoc_exact_shadow_characterization_not_certificate",
            "corrected_costate_error_improves_rows": 62,
            "positive_renderer_gradient_dot_rows": 64,
            "corrected_ce_descent_rows": 63,
            "corrected_dseg_nonworsening_rows": 49,
            "hvp_incremental_median_seconds": HVP_INCREMENTAL_MEDIAN_SECONDS,
            "matched_through_r_validation_median_seconds": (
                MATCHED_THROUGH_R_VALIDATION_MEDIAN_SECONDS
            ),
            "hvp_optimistic_lower_bound_matched_validation_equivalents_per_step": (
                HVP_LOWER_BOUND_MATCHED_VALIDATION_EQUIVALENTS
            ),
            "early_faithful_lower_bound_equivalents_per_anchor": (
                EARLY_FAITHFUL_LOWER_BOUND_EQUIVALENTS_PER_ANCHOR
            ),
            "boundary_faithful_lower_bound_equivalents_per_anchor": (
                BOUNDARY_FAITHFUL_LOWER_BOUND_EQUIVALENTS_PER_ANCHOR
            ),
            "faithful_economics": "NO_GO_INCREMENTAL_HVP_COST_EXCEEDS_8_375_AT_HIGH_REUSE",
            "collinear_amortization": "NO_GO_RENDERER_LINEARIZATION_RESIDUAL_APPROXIMATES_DISPLACEMENT",
            "score_claim": False,
            "pointer_moved": False,
        },
        residual=1.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "three sealed pair-0 saved regimes; exact through-R scorer inputs; detached-adjoint "
            "Jacobian HVP correction; fresh exact input-costate and renderer-gradient shadows; "
            "matched normalized one-step exact CE/numpy-fp32 d_seg controls; post-hoc L2-sorted "
            "safe-prefix characterization; correction economics conservatively rederived against "
            "matched full through-R validation time"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=1.0 / 196608.0,
        noise_floor_provenance=(
            "DERIVED one numpy-fp32 d_seg pixel over the measured 256x768 label field"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Direct full-costate Jacobian-drift certificate",
        one_line_summary=(
            "Reuse a first-order corrected full input costate only inside the strict "
            "self-adjusting ball controlled by Jacobian, adjoint, correction, and geometry bounds."
        ),
        latex_form=(
            r"E(r)=(B_JL_q+L_c)r+(B_HL_q+\tfrac12L_HQ_a)r^2+"
            r"\tfrac12L_HL_qr^3,\quad "
            r"r^*=\min\{\sup[E(r)<\gamma_\theta/B_R],r_{geometry},r_{cap}\}"
        ),
        python_callable_module_path=(
            "tac.scorer_surrogate.costate_trust_region:derive_direct_costate_certificate"
        ),
        domain_of_validity={
            "scope_level": "conditional local theorem plus registered empirical characterization",
            "research_only": True,
            "included": [
                "p(x)=J(x)^Tq(x) and p_hat(x)=p(a)+(DJ(a)[x-a])^Tq(a)",
                "one content-bound coercive L2 norm and its dual across every operator bound",
                "whole-ball upper bounds including Lip(DJ) and correction numerical error",
                "strict membership; equality refreshes",
                "current corrected renderer-gradient lower bound over the admitted ball",
            ],
            "excluded": [
                "promoting point HVP or power iteration to an operator-supremum upper bound",
                "using margin-Fisher RMS geometry as an L2 radius without a proved conversion",
                "calling a full CE Hessian-vector the fixed-adjoint Jacobian-drift correction",
                "crossing a full-SegNet activation boundary without a semismooth replacement theorem",
                "calling a post-hoc sampled-ray safe prefix a whole-ball or operational certificate",
                "score, contest-CPU, contest-CUDA, evaluator, or promotion authority",
            ],
            "fallback": "fresh exact input costate",
            "review_status": "worker-tested; independent theorem and authority audit incorporated",
        },
        units_in={
            "r": "scorer-input L2 units",
            "B_J": "logit-output norm per scorer-input L2 unit",
            "B_H": "Jacobian operator norm per scorer-input L2 unit",
            "L_H": "Jacobian-derivative operator norm per squared scorer-input L2 unit",
            "Q_a": "logit-adjoint dual norm",
            "L_q": "logit-adjoint dual norm per scorer-input L2 unit",
            "L_c": "input-costate norm per scorer-input L2 unit",
            "B_R": "renderer-gradient norm per input-costate norm",
            "gamma_theta": "current corrected reused renderer-gradient norm lower bound over ball",
        },
        units_out={"E_r": "input-costate norm", "r_star": "scorer-input L2 units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "missing_rigorous_bound_custody": 1.0,
            "regimes_without_nonzero_oracle_safe_prefix_fraction": 1.0 / 3.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.costate_trust_region_policy.DirectFullCostatePolicy",
            "tools.probe_jacobian_drift_certificate",
        ),
        canonical_producers=("tools.probe_jacobian_drift_certificate",),
        provenance=provenance,
    )


def populate_jacobian_drift_full_costate_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Append through the locked registry helper; never mutate registry bytes directly."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_jacobian_drift_full_costate_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="task454b; research_only; rigorous-blocked; faithful-economics-no-go",
    )
    return equation


__all__ = [
    "BASELINE_VALIDATIONS_PER_TEACHER_CALL",
    "BOUNDARY_FAITHFUL_LOWER_BOUND_EQUIVALENTS_PER_ANCHOR",
    "CURRENT_DESCENT_PREFIX_BY_REGIME",
    "EARLY_FAITHFUL_LOWER_BOUND_EQUIVALENTS_PER_ANCHOR",
    "EQUATION_ID",
    "HVP_LOWER_BOUND_MATCHED_VALIDATION_EQUIVALENTS",
    "ORACLE_SAFE_PREFIX_BY_REGIME",
    "RIGOROUS_CERTIFIED_REUSES",
    "build_jacobian_drift_full_costate_v1",
    "populate_jacobian_drift_full_costate_v1",
]
