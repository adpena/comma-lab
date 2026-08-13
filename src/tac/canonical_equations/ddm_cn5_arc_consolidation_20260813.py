# SPDX-License-Identifier: MIT
"""Canonical laws extracted by the 2026-08-13 DDM CN5 consolidation.

The four laws remain deliberately narrow:

* PO1 calibrates whether a local modeled step clears its own instrument floor.
* PZ4R separates deterministic receiver repeatability from PoseNet semantics.
* HV1 inverts the exact contest Pose term for one stack-level score allowance.
* JS5 measures the post-uint8 amplitude/leakage exponent on its n32 panel.

None of these equations promotes an advisory row, transfers a number across
vehicles, or claims that a closed instance closes a representation family.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import (
    eval_local_model_step_resolvability_ratio,
    eval_pose_stack_exact_budget,
    eval_receiver_lattice_leakage_exponent,
    eval_receiver_pose_semantic_preservation_ratio,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

REPO = Path(__file__).resolve().parents[3]

STEP_RESOLVABILITY_EQUATION_ID = "local_model_step_resolvability_ratio_v1"
POSE_SEMANTIC_EQUATION_ID = "receiver_pose_semantic_preservation_ratio_v1"
POSE_BUDGET_EQUATION_ID = "pose_stack_exact_budget_v1"
LATTICE_EXPONENT_EQUATION_ID = "receiver_lattice_leakage_exponent_v1"

PO1_MEMO = REPO / ".omx/research/ddm_po1_t4_error_feedback_pose_compensation_20260813.md"
PZ4R_MEMO = REPO / ".omx/research/ddm_pz4r_full_n600_eval_20260813.md"
HV1_MEMO = REPO / ".omx/research/ddm_hv1_fresh_eyes_hybrid_review_20260813.md"
JS5_MEMO = REPO / ".omx/research/ddm_js5_projector_distilled_conditioning_20260812.md"


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def build_local_model_step_resolvability_equation(*, source_receipt: Path = PO1_MEMO) -> CanonicalEquation:
    """Build the PO1 local-model instrument-resolution law."""

    inputs: Mapping[str, Any] = {
        "predicted_step_magnitude": 1.0e-8,
        "forward_mismatch_floor": 9.36e-6,
    }
    ratio = eval_local_model_step_resolvability_ratio(inputs)
    base_d_pose = 6.885642960696714e-6
    predicted_candidate_d_pose = 6.439082127459961e-6
    realized_candidate_d_pose = 5.6857839808799326e-5
    predicted_pose_ratio = predicted_candidate_d_pose / base_d_pose
    realized_pose_ratio = realized_candidate_d_pose / base_d_pose
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "reopen the local-step family only with a measured forward mismatch below "
            "every admitted step, or a representation-level model whose predicted step "
            "magnitude exceeds that floor"
        ),
        measurement_axis="[contest-CUDA T4 component-only, n600]",
        hardware_substrate="linux_x86_64_t4",
        captured_at_utc="2026-08-13T10:36:41Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="po1_round2b_local_step_below_forward_mismatch_20260813",
        measurement_utc="2026-08-13T10:36:41Z",
        inputs={
            **inputs,
            "accepted_pair_gain_range": [1.0e-9, 1.0e-8],
            "predicted_total_d_pose_gain": 4.46560833236753e-7,
            "forward_residual_rms": 1.589e-3,
            "base_d_pose": base_d_pose,
        },
        predicted_output={
            "instrument_resolved": False,
            "most_charitable_resolvability_ratio": ratio,
            "candidate_over_base_d_pose": predicted_pose_ratio,
        },
        empirical_output={
            "candidate_over_base_d_pose": realized_pose_ratio,
            "realized_d_pose": realized_candidate_d_pose,
            "repeat_noise_mse": 0.0,
            "round2b_realization_status": "REJECTED_INSTANCE",
        },
        residual=abs(realized_pose_ratio - predicted_pose_ratio),
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="po1_round2b_t4_component_replay_plus_model_forward_mismatch",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=STEP_RESOLVABILITY_EQUATION_ID,
        name="Local modeled-step resolvability against forward mismatch",
        one_line_summary=(
            "A local-model acceptance step is instrument-resolved only when its "
            "absolute predicted magnitude exceeds the model's measured forward mismatch."
        ),
        latex_form=r"\rho_{\mathrm{inst}}=|\widehat{\Delta}|/\epsilon_{\mathrm{fwd}},\quad \rho_{\mathrm{inst}}>1",
        python_callable_module_path=("tac.canonical_equations.evaluators:eval_local_model_step_resolvability_ratio"),
        domain_of_validity={
            "included": [
                "local linear or Jacobian proposal models with a measured same-object forward mismatch",
                "PO1 int16 error-feedback pose compensation on CP135",
            ],
            "excluded": [
                "models with no measured forward-mismatch floor",
                "representation-changing learned models",
                "claim that every below-floor proposal must worsen rather than remain unresolved",
            ],
            "admission_gate": "ratio_strictly_greater_than_one",
            "verdict_scope": "FORMULATION(PO1 local-J int16 error-feedback admission)",
            "score_claim": False,
        },
        units_in={
            "predicted_step_magnitude": "same units as modeled objective delta",
            "forward_mismatch_floor": "same units as modeled objective delta",
        },
        units_out={"resolvability_ratio": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"po1_round2b_candidate_over_base_d_pose": anchor.residual},
        last_calibration_utc="2026-08-13T10:36:41Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.local_jacobian_admission",
            ".omx.research.ddm_re1_realization_engineered_candidate_20260813",
        ),
        canonical_producers=(
            ".omx.research.ddm_po1_t4_error_feedback_pose_compensation_20260813",
            "tac.canonical_equations.ddm_cn5_arc_consolidation_20260813",
        ),
        provenance=provenance,
    )


def build_receiver_pose_semantic_preservation_equation(*, source_receipt: Path = PZ4R_MEMO) -> CanonicalEquation:
    """Build the PZ4R semantic Pose preservation ratio law."""

    inputs: Mapping[str, Any] = {
        "base_d_pose": 0.00014746535453014076,
        "candidate_d_pose": 0.6310142278671265,
    }
    ratio = eval_receiver_pose_semantic_preservation_ratio(inputs)
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "admit a receiver candidate only after a matched full-population PoseNet "
            "vector or d_pose gate passes in addition to byte/decode repeat identity"
        ),
        measurement_axis="[macOS-CPU advisory, frozen CPU-torch PoseNet, n600]",
        hardware_substrate="m5_max_macos_cpu",
        captured_at_utc="2026-08-13T16:16:59Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="pz4r_repeat_identity_without_pose_semantic_preservation_20260813",
        measurement_utc="2026-08-13T16:16:59Z",
        inputs={
            **inputs,
            "base_archive_bytes": 186_252,
            "candidate_archive_bytes": 183_137,
            "base_seg_flips": 50_394,
            "candidate_seg_flips": 50_412,
        },
        predicted_output={
            "repeat_identity_implies_semantic_preservation": True,
            "expected_candidate_over_base_d_pose": 1.0,
        },
        empirical_output={
            "candidate_over_base_d_pose": ratio,
            "repeat_identity": True,
            "semantic_pose_gate_passed": False,
            "retained_record_count": 619,
        },
        residual=abs(ratio - 1.0),
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="matched_n600_public_receiver_pose_replay_with_repeat_identity",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=POSE_SEMANTIC_EQUATION_ID,
        name="Receiver PoseNet semantic-preservation ratio",
        one_line_summary=(
            "Receiver byte identity and deterministic repeats do not establish PoseNet "
            "preservation; compare candidate d_pose with the matched base explicitly."
        ),
        latex_form=r"\rho_{\mathrm{pose}}=d_{\mathrm{pose}}(C)/d_{\mathrm{pose}}(B)",
        python_callable_module_path=(
            "tac.canonical_equations.evaluators:eval_receiver_pose_semantic_preservation_ratio"
        ),
        domain_of_validity={
            "included": [
                "matched-base receiver-closed candidates evaluated through frozen PoseNet",
                "PZ4R direct-v6 relative to the matched local CP135 base",
            ],
            "excluded": [
                "cross-axis T4-to-macOS component mixing",
                "semantic preservation inferred only from archive or decode hashes",
                "closure of joint-trained or different receiver representations",
            ],
            "verdict_scope": "INSTANCE(PZ4R direct-v6 full n600 local advisory row)",
            "score_claim": False,
        },
        units_in={"base_d_pose": "mse", "candidate_d_pose": "mse"},
        units_out={"candidate_over_base_d_pose": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"pz4r_repeat_identity_pose_ratio": anchor.residual},
        last_calibration_utc="2026-08-13T16:16:59Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.receiver_candidate_admission",
            ".omx.research.ddm_re1_realization_engineered_candidate_20260813",
        ),
        canonical_producers=(
            ".omx.research.ddm_pz4r_full_n600_eval_20260813",
            "tac.canonical_equations.ddm_cn5_arc_consolidation_20260813",
        ),
        provenance=provenance,
    )


def build_pose_stack_exact_budget_equation(*, source_receipt: Path = HV1_MEMO) -> CanonicalEquation:
    """Build the exact stack-level Pose degradation budget inversion."""

    inputs: Mapping[str, Any] = {
        "base_d_pose": 6.885642960696714e-6,
        "seg_credit_s": 0.000960,
        "archive_delta_bytes": 323,
    }
    exact_budget = eval_pose_stack_exact_budget(inputs)
    net_allowance = 0.000960 - 25.0 * 323 / 37_545_489.0
    first_order_budget = net_allowance * (10.0 * inputs["base_d_pose"]) ** 0.5 / 5.0
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "recompute at every new base d_pose and candidate-specific Seg/rate allowance; "
            "declare any extra safety factor separately"
        ),
        measurement_axis="[derived from CP135 contest-CUDA T4 n600 components]",
        hardware_substrate="source_inspection_exact_score_equation",
        captured_at_utc="2026-08-13T15:15:39Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="hv1_js7_corrected_candidate_specific_pose_stack_budget_20260813",
        measurement_utc="2026-08-13T15:15:39Z",
        inputs={
            **inputs,
            "seg_credit_provenance": "JS7 inferred n32 projected Seg credit",
            "archive_delta_bytes_provenance": "JS7 exact counted archive delta",
            "old_assumed_stack_budget": 1.3e-7,
        },
        predicted_output={"old_assumed_stack_budget": 1.3e-7},
        empirical_output={
            "net_score_allowance": net_allowance,
            "first_order_pose_budget": first_order_budget,
            "exact_pose_budget": exact_budget,
            "safety_factor_included": False,
        },
        residual=abs(exact_budget - 1.3e-7),
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="exact_symbolic_inversion_of_contest_pose_term_and_rate_term",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=POSE_BUDGET_EQUATION_ID,
        name="Exact stack-level Pose degradation budget",
        one_line_summary=(
            "Invert sqrt(10*d_pose) after candidate-specific Seg credit and exact "
            "archive-byte cost; budget the whole stack, never each proposal independently."
        ),
        latex_form=(
            r"A=G_{seg}-25\Delta B/37545489,\quad "
            r"\Delta p_{max}=((\sqrt{10p_0}+A)^2/10)-p_0"
        ),
        python_callable_module_path=("tac.canonical_equations.evaluators:eval_pose_stack_exact_budget"),
        domain_of_validity={
            "included": [
                "candidate-specific stack admission with a fixed measured base d_pose",
                "exact contest score arithmetic with a declared Seg credit and byte delta",
            ],
            "excluded": [
                "transfer of one candidate's budget to another base or vehicle",
                "per-proposal reuse of the entire stack budget",
                "undeclared safety factors",
                "promotion of the inferred JS7 Seg credit to a measured n600 gain",
            ],
            "verdict_scope": "FORMULATION(exact score-budget conversion; JS7 anchor is candidate-specific)",
            "score_claim": False,
        },
        units_in={
            "base_d_pose": "mse",
            "seg_credit_s": "score_units",
            "archive_delta_bytes": "bytes",
        },
        units_out={"maximum_d_pose_increase": "mse"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"hv1_old_assumption_to_exact_js7_budget": anchor.residual},
        last_calibration_utc="2026-08-13T15:15:39Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.joint_stack_admission",
            ".omx.research.ddm_js7_exact_row_verdict_20260812",
        ),
        canonical_producers=(
            ".omx.research.ddm_hv1_fresh_eyes_hybrid_review_20260813",
            "tac.canonical_equations.ddm_cn5_arc_consolidation_20260813",
        ),
        provenance=provenance,
    )


def build_receiver_lattice_leakage_exponent_equation(*, source_receipt: Path = JS5_MEMO) -> CanonicalEquation:
    """Build the JS5 post-uint8 log-log leakage exponent law."""

    amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]
    continuous_leakage = [
        8.8360566e-4,
        1.0038738e-4,
        1.6397902e-5,
        3.4996673e-6,
        7.8618451e-7,
    ]
    receiver_leakage = [
        8.5741907e-4,
        1.3687468e-4,
        6.6803035e-5,
        4.2786291e-5,
        5.5702489e-5,
    ]
    continuous_exponent = eval_receiver_lattice_leakage_exponent(
        {"amplitudes": amplitudes, "leakages": continuous_leakage}
    )
    receiver_exponent = eval_receiver_lattice_leakage_exponent({"amplitudes": amplitudes, "leakages": receiver_leakage})
    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "recalibrate on a content-distinct representation or a seeded larger panel "
            "that measures the same continuous and post-receiver amplitudes"
        ),
        measurement_axis="[macOS-CPU advisory, seeded stratified n32]",
        hardware_substrate="m5_max_macos_cpu",
        captured_at_utc="2026-08-12T13:56:40Z",
    )
    anchor = EmpiricalAnchor(
        anchor_id="js5_continuous_vs_uint8_amplitude_leakage_exponents_20260812",
        measurement_utc="2026-08-12T13:56:40Z",
        inputs={
            "amplitudes": amplitudes,
            "leakages": receiver_leakage,
            "continuous_leakages": continuous_leakage,
            "panel_pairs": 32,
        },
        predicted_output={
            "second_order_continuous_exponent": 2.0,
            "expected_receiver_exponent": 2.0,
        },
        empirical_output={
            "continuous_exponent": continuous_exponent,
            "receiver_exponent": receiver_exponent,
            "qualified_amplitude_count": 0,
            "proposal_count": 15,
        },
        residual=abs(receiver_exponent - 2.0),
        source_artifact=_relative_or_absolute(source_receipt),
        measurement_method="ols_loglog_fit_over_retained_js5_alpha_ladder",
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=LATTICE_EXPONENT_EQUATION_ID,
        name="Receiver-lattice amplitude/leakage exponent",
        one_line_summary=(
            "Fit leakage against amplitude after the real uint8 receiver; JS5's "
            "continuous second-order falloff collapsed to an approximately linear exponent."
        ),
        latex_form=(
            r"q=\frac{\sum_i(\log\alpha_i-\overline{\log\alpha})"
            r"(\log L_i-\overline{\log L})}{\sum_i(\log\alpha_i-\overline{\log\alpha})^2}"
        ),
        python_callable_module_path=("tac.canonical_equations.evaluators:eval_receiver_lattice_leakage_exponent"),
        domain_of_validity={
            "included": [
                "strictly positive amplitude/leakage ladders",
                "JS5 selected step-25 projector-distilled module after uint8 receiver",
            ],
            "excluded": [
                "n600 or contest-axis authority",
                "other representations or learned amplitude families",
                "extrapolation below the measured 1/16 amplitude",
            ],
            "verdict_scope": "INSTANCE(JS5 selected module, seeded stratified n32 alpha ladder)",
            "score_claim": False,
        },
        units_in={"amplitudes": "dimensionless", "leakages": "d_pose_mse"},
        units_out={"loglog_exponent": "dimensionless"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"js5_receiver_exponent_vs_second_order_prediction": anchor.residual},
        last_calibration_utc="2026-08-12T13:56:40Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.receiver_amplitude_admission",
            ".omx.research.ddm_js6_seg_representation_join_20260813",
        ),
        canonical_producers=(
            ".omx.research.ddm_js5_projector_distilled_conditioning_20260812",
            "tac.canonical_equations.ddm_cn5_arc_consolidation_20260813",
        ),
        provenance=provenance,
    )


def build_ddm_cn5_arc_equations() -> tuple[CanonicalEquation, ...]:
    """Build all four bounded CN5 equations without writing the registry."""

    return (
        build_local_model_step_resolvability_equation(),
        build_receiver_pose_semantic_preservation_equation(),
        build_pose_stack_exact_budget_equation(),
        build_receiver_lattice_leakage_exponent_equation(),
    )


def populate_ddm_cn5_arc_equations(
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str = "codex",
    subagent_id: str = "ddm_cn5",
) -> tuple[str, ...]:
    """Register the four CN5 laws through the locked append-only helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equations = build_ddm_cn5_arc_equations()
    for equation in equations:
        register_canonical_equation(
            equation,
            path=path,
            lock_path=lock_path,
            agent=agent,
            subagent_id=subagent_id,
            notes="ddm_cn5 bounded arc law registration",
        )
    return tuple(equation.equation_id for equation in equations)


__all__ = [
    "LATTICE_EXPONENT_EQUATION_ID",
    "POSE_BUDGET_EQUATION_ID",
    "POSE_SEMANTIC_EQUATION_ID",
    "STEP_RESOLVABILITY_EQUATION_ID",
    "build_ddm_cn5_arc_equations",
    "build_local_model_step_resolvability_equation",
    "build_pose_stack_exact_budget_equation",
    "build_receiver_lattice_leakage_exponent_equation",
    "build_receiver_pose_semantic_preservation_equation",
    "populate_ddm_cn5_arc_equations",
]
