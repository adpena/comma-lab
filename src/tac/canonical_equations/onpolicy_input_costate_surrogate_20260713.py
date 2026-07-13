# SPDX-License-Identifier: MIT
"""Derived mechanism law and evidence-limited first disposition for task #455."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_NEVER_AUTO,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "onpolicy_input_costate_surrogate_v1"
_DESIGN = ".omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md"
_RECEIPT = "experiments/results/onpolicy_scorer_surrogate_20260713T020600Z/measurement_receipt.json"
_UTC = "2026-07-13T02:35:08.485921Z"
_REVIEW_UTC = "2026-07-13T02:47:10.196415Z"
_RECEIPT_SHA256 = "2812cd3a984fb063845d0690d79e319f2875aad4954630a87b43cda94d22211b"
_CAMPAIGN = "experiments/results/onpolicy_costate_matched_campaign_20260713T031500Z.json"
_CAMPAIGN_SHA256 = "dc56941789b2746bb53a6cb652b9a42dcd1299e11cdaa8567aae96a930e9b8bf"
_CAMPAIGN_UTC = "2026-07-13T03:24:31.059268Z"
_TERMINAL = (
    "experiments/results/onpolicy_costate_symmetric_timing_20260713T034500Z/"
    "boundary/measurement_receipt.json"
)
_TERMINAL_SHA256 = "dc245467f9bd1fb63f3cfc0bbc1f092d133997a26d418a356738e8642c9abdf4"
_TERMINAL_UTC = "2026-07-13T03:44:30.630874Z"
_TERMINAL_MEMO = ".omx/research/onpolicy_surrogate_95kill_20260713.md"
_FINAL_CAMPAIGN = (
    "experiments/results/onpolicy_costate_matched_campaign_final_20260713T043000Z.json"
)
_FINAL_CAMPAIGN_SHA256 = "5b73396f4990a0d7d44fd358d64fc87d4b3e442dc7ac7a34f1264013fae5aff8"
_FINAL_CAMPAIGN_UTC = "2026-07-13T04:30:21.175155Z"


def build_onpolicy_input_costate_surrogate_v1() -> CanonicalEquation:
    """Build the mechanism law and its evidence-limited first measurement."""

    provenance = build_provenance_for_research_sidecar(
        sidecar_path=_DESIGN,
        reactivation_criteria=(
            "change the pre-registered capacity/control law, then rerun early/boundary/late until one cadence "
            "completes all regimes; only after admission wire the provider into the live trainer through typed DSL"
        ),
        measurement_axis="[macOS-CPU advisory; torch-fp32; n=1 pair0; research_only]",
        hardware_substrate="torch_cpu_research_scaffold",
        captured_at_utc=_REVIEW_UTC,
    )
    anchor = EmpiricalAnchor(
        anchor_id="onpolicy_costate_pair0_three_regime_partial_20260713",
        measurement_utc=_UTC,
        inputs={
            "pair_index": 0,
            "regimes": ["early", "boundary", "late"],
            "cadences": [1, 4, 20],
            "post_bootstrap_steps": 40,
            "seed": 455,
            "ema_decay": 0.997,
        },
        predicted_output={
            "admission": (
                "matched exact and surrogate d_seg trajectories share a common step schedule, the EMA provider "
                "passes, and isolated forward/costate/inference timings show positive economics"
            )
        },
        empirical_output={
            "measured_arms": 2,
            "blocked_arms": 7,
            "k4_timing_status": "diagnostic_only_not_isolated_forward_replacement_economics",
            "k20_regimes_completed": 0,
            "common_admitted_cadence": False,
            "raw_receipt_verdict": "NEEDS_MORE",
            "canonical_disposition": "NEEDS_MORE",
            "receipt_sha256": _RECEIPT_SHA256,
            "deterministic_reproduction": "BLOCKED_MISSING_UNCOMMITTED_LAUNCH_SOURCE_BYTES",
            "review_status": "fresh-eyes-reviewed(1)-finding-producing-evidence-limited",
            "inadmissible_claim_surfaces": (
                "matched_dseg_trajectory_parity",
                "isolated_forward_replacement_timing",
                "ema_provider_admission",
                "dense_temporal_onpolicy_supervision",
                "zero_teacher_resume",
            ),
        },
        residual=1.0,
        source_artifact=_RECEIPT,
        measurement_method=(
            "n=1 pair-0 early/boundary/late K={1,4,20} first probe; receipt contract/hash and 14 checkpoint "
            "records authenticated, but missing launch-source bytes block deterministic resume and the receipt "
            "cannot support the target gate"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance="deterministic repeat d_seg/d_pose floor is zero; across-seed variance remains UNKNOWN",
    )
    corrected_provenance = build_provenance_for_research_sidecar(
        sidecar_path=".omx/research/onpolicy_surrogate_95kill_20260713.md",
        reactivation_criteria=(
            "pre-register a changed capacity, EMA/update law, or exact d_seg-valid controller; retain dense "
            "student-owned labels and measure a full K20 matched trajectory in early/boundary/late"
        ),
        measurement_axis=(
            "[macOS-CPU advisory training-gradient; torch-fp32; pair0; seed455; "
            "early/boundary/late; W5 smoke prefixes]"
        ),
        hardware_substrate="torch_cpu_research_measurement",
        captured_at_utc=_CAMPAIGN_UTC,
    )
    corrected_anchor = EmpiricalAnchor(
        anchor_id="onpolicy_costate_pair0_three_regime_matched_w5_20260713",
        measurement_utc=_CAMPAIGN_UTC,
        inputs={
            "pair_index": 0,
            "regimes": ["early", "boundary", "late"],
            "seed": 455,
            "collection_steps": 5,
            "matched_window_steps": 5,
            "target_anchor_cadence": 20,
            "ema_decay": 0.8,
            "hidden_channels": 16,
            "branch_kernel_sizes": [3, 5],
        },
        predicted_output={
            "admission": (
                "EMA provider admitted and exact CE/d_seg/d_pose traces match deterministic repeats across "
                "all requested regimes"
            ),
            "target_teacher_skip_fraction": 0.95,
        },
        empirical_output={
            "campaign_verdict": "NEEDS_MORE",
            "early_verdict": "NO_GO",
            "boundary_verdict": "NEEDS_MORE_INVALID_EXACT_DSEG_CONTROL",
            "late_verdict": "NEEDS_MORE_INVALID_EXACT_DSEG_CONTROL",
            "deterministic_repeat_noise_floor": {"ce": 0.0, "d_seg": 0.0, "d_pose": 0.0},
            "measured_exact_forward_mean_seconds": 0.4345562303826834,
            "measured_surrogate_inference_mean_seconds": 0.13003257649446218,
            "measured_same_run_forward_speedup": 3.341902791576157,
            "derived_operator_1656ms_speedup": 12.735270227230524,
            "measured_anchored_window_speedup": 2.0364456861336944,
            "observed_teacher_skip_fraction": 0.8,
            "full_k20_fidelity": "UNKNOWN_NOT_MEASURED",
            "common_admitted_formulation": False,
            "receipt_sha256": _CAMPAIGN_SHA256,
            "review_status": "post-measurement-fresh-eyes-pending",
        },
        residual=1.0,
        source_artifact=_CAMPAIGN,
        measurement_method=(
            "three source-compatible pair0 seed455 saved-regime runs; dense exact labels on student-owned "
            "states; exact-derived common step norms; every-step exact CE/d_seg/d_pose; deterministic repeat; "
            "isolated timing and reconciled teacher-call accounting"
        ),
        provenance=corrected_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance=(
            "MEASURED two deterministic exact traces per regime: zero CE/d_seg/d_pose delta; "
            "across-seed and cross-hardware variance UNKNOWN"
        ),
    )
    terminal_provenance = build_provenance_for_research_sidecar(
        sidecar_path=_TERMINAL_MEMO,
        reactivation_criteria=(
            "pre-register a changed architecture, capacity, or update law; preserve the joint exact "
            "CE/d_seg/d_pose fractional control and require full K20 matched fidelity before activation"
        ),
        measurement_axis=(
            "[macOS-CPU advisory training-gradient; torch-fp32; pair0; seed455; boundary; "
            "event-conditioned three-update measured prefix; completion BLOCKED]"
        ),
        hardware_substrate="torch_cpu_research_measurement",
        captured_at_utc=_TERMINAL_UTC,
    )
    terminal_anchor = EmpiricalAnchor(
        anchor_id="onpolicy_costate_pair0_boundary_symmetric_timing_terminal_20260713",
        measurement_utc=_TERMINAL_UTC,
        inputs={
            "pair_index": 0,
            "regime": "boundary",
            "seed": 455,
            "target_anchor_cadence": 20,
            "maximum_step_fraction": 0.01,
            "learning_rate": 0.002,
            "ema_decay": 0.8,
            "hidden_channels": 16,
            "branch_kernel_sizes": [3, 5],
        },
        predicted_output={
            "admission": (
                "EMA provider preserves exact CE/d_seg/d_pose at the deterministic-repeat floor "
                "under the joint fractional common controller"
            ),
            "target_teacher_skip_fraction": 0.95,
        },
        empirical_output={
            "verdict": "NO_GO",
            "verdict_scope": "tested formulation on pair0 boundary regime seed455; not family or score authority",
            "accepted_exact_prefix_valid": True,
            "exact_completion_certified": False,
            "completion_reclassification": (
                "BLOCKED_PARAMETER_QUANTIZATION_EXHAUSTION; launch receipt terminal-floor label revoked"
            ),
            "observed_updates": 3,
            "observed_teacher_skip_fraction": 2.0 / 3.0,
            "deterministic_repeat_noise_floor": {"ce": 0.0, "d_seg": 0.0, "d_pose": 0.0},
            "first_failing_step": 2,
            "maximum_ce_regret": 1.7881393432617188e-7,
            "maximum_d_seg_regret": 0.0,
            "maximum_d_pose_regret": 6.329965646330038e-4,
            "measured_matched_window_speedup": 1.8398869525117787,
            "measured_matched_window_saved_fraction": 0.456488346398229,
            "projected_k20_speedup_non_authority": 2.7652661849351374,
            "full_k20_fidelity": "NOT_VALIDATED_BLOCKED_AFTER_MEASURED_PREFIX",
            "receipt_sha256": _TERMINAL_SHA256,
            "review_status": "fresh-eyes-review-pending",
        },
        residual=1.0,
        source_artifact=_TERMINAL,
        measurement_method=(
            "dense exact labels on student-owned states; exact joint CE/d_seg/d_pose fractional controller; "
            "exact-derived common step norms; deterministic repeat floor; symmetric complete per-step timing; "
            "hook-reconciled scorer calls; source-bundled resumable stage custody"
        ),
        provenance=terminal_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance=(
            "MEASURED paired deterministic repeat: zero CE/d_seg/d_pose delta; "
            "across-seed and cross-hardware variance UNKNOWN"
        ),
    )
    final_provenance = build_provenance_for_research_sidecar(
        sidecar_path=".omx/research/onpolicy_surrogate_95kill_20260713.md",
        reactivation_criteria=(
            "pre-register a changed architecture, capacity, EMA/update law, or provider representation; "
            "retain joint exact control and clear non-anchor fidelity in every saved regime before a full "
            "K20 matched replay or live-trainer activation"
        ),
        measurement_axis=(
            "[macOS-CPU advisory training-gradient; torch-fp32; pair0; seed455; "
            "early/boundary/late; blocked measured prefixes of 2/1/3 accepted updates]"
        ),
        hardware_substrate="torch_cpu_research_measurement",
        captured_at_utc=_FINAL_CAMPAIGN_UTC,
    )
    final_anchor = EmpiricalAnchor(
        anchor_id="onpolicy_costate_pair0_final_joint_campaign_20260713",
        measurement_utc=_FINAL_CAMPAIGN_UTC,
        inputs={
            "pair_index": 0,
            "regimes": ["early", "boundary", "late"],
            "seed": 455,
            "collection_steps": 5,
            "configured_smoke_window_steps": 5,
            "executable_decisive_window_steps": 20,
            "observed_updates_by_regime": {"early": 2, "boundary": 1, "late": 3},
            "target_anchor_cadence": 20,
            "ema_decay": 0.8,
            "hidden_channels": 16,
            "branch_kernel_sizes": [3, 5],
        },
        predicted_output={
            "admission": (
                "EMA provider admitted and surrogate-driven exact CE/d_seg/d_pose traces remain at the "
                "deterministic-repeat floor in every saved regime"
            ),
            "target_teacher_skip_fraction": 0.95,
        },
        empirical_output={
            "campaign_verdict": "NO_GO",
            "campaign_verdict_law": (
                "any non-anchor raw accepted-prefix failure rejects the tested formulation"
            ),
            "early_verdict": "NO_GO_EMA_NOT_ADMITTED",
            "boundary_verdict": "NEEDS_MORE_EXACT_ANCHOR_ONLY",
            "late_verdict": "NO_GO_DSEG_TRAJECTORY_DRIFT",
            "deterministic_repeat_noise_floor": {"ce": 0.0, "d_seg": 0.0, "d_pose": 0.0},
            "late_maximum_d_seg_delta": 2.0345052083333044e-5,
            "measured_exact_forward_mean_seconds": 0.5370454628977718,
            "measured_surrogate_inference_mean_seconds": 0.127524527721107,
            "measured_same_run_forward_speedup": 4.2113111296697125,
            "derived_operator_1656ms_speedup": 12.985737172237416,
            "measured_whole_window_speedup": 2.1077906826867006,
            "corrected_repeat_whole_window_speedup_range": [
                1.329915767799602,
                2.1077906826867006,
            ],
            "timing_repeat_variance": "MEASURED_NONZERO; shared-host cause unresolved",
            "observed_teacher_skip_fraction_by_regime": {
                "early": 0.5,
                "boundary": 0.0,
                "late": 2.0 / 3.0,
            },
            "full_k20_fidelity": "UNKNOWN_BLOCKED_MEASURED_PREFIXES_ONLY",
            "common_admitted_formulation": False,
            "receipt_sha256": _FINAL_CAMPAIGN_SHA256,
            "review_status": "post-fix-three-pass-seal-pending",
        },
        residual=1.0,
        source_artifact=_FINAL_CAMPAIGN,
        measurement_method=(
            "three source-compatible pair0 seed455 saved-regime runs; dense exact labels on student-owned "
            "states; joint CE/d_seg/d_pose fractional controller; exact-derived common step norms; exact "
            "deterministic repeats; symmetric complete-window timing; reconciled SegNet/PoseNet calls; "
            "source-bundled zero-teacher resume"
        ),
        provenance=final_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=0.0,
        noise_floor_provenance=(
            "MEASURED paired deterministic exact replay in every regime: zero CE/d_seg/d_pose delta; "
            "across-seed and cross-hardware variance UNKNOWN"
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="On-policy nonlinear frozen-SegNet input-costate surrogate",
        one_line_summary=(
            "Fit a nonlinear provider only on exact teacher-labeled witness transitions and inject its detached "
            "frame-shaped costate between exact anchors."
        ),
        latex_form=(
            r"\widehat\lambda_t=\lambda_a+s_a\tanh(\|x_t-x_a\|_{2,c}/255)"
            r"r_\phi(x_t/255,(x_t-x_a)/255,\lambda_a/s_a),\quad "
            r"L_{\rm inject}(\theta)=\langle\operatorname{stopgrad}(\widehat\lambda_t),x(\theta)\rangle,\quad "
            r"S_K=K t_{\rm exact}/(t_{\rm exact}+(K-1)t_{\rm surrogate})"
        ),
        python_callable_module_path=(
            "tac.scorer_surrogate.amortized_onpolicy_costate:predict_ema_detached_costate"
        ),
        domain_of_validity={
            "included": (
                "on-trajectory through-R frames with exact frozen-SegNet CE input-costate labels",
                "detached training-signal replacement only",
                "typed nonlinear K20 decisive target and W5 smoke measurement policy",
            ),
            "excluded": (
                "d_seg, d_pose, evaluator, score, or promotion authority",
                "offline/fixed-dataset distillation",
                "live-trainer activation or a positive three-regime fidelity verdict",
                "full-K20 fidelity, unseen pairs/seeds, or cross-hardware transfer",
            ),
            "first_receipt_status": (
                "MEASURED / raw NEEDS-MORE / canonical NEEDS-MORE: valid historical observations, "
                "but no matched-trajectory or isolated-forward verdict authority"
            ),
            "research_only": True,
            "full_build_blocker": (
                "early EMA admission fails and late non-anchor CE, d_pose, and d_seg trajectories "
                "leave their deterministic-repeat floors; boundary is exact-anchor-only NEEDS-MORE; "
                "full-K20 trajectory fidelity remains unvalidated; live-trainer activation is forbidden"
            ),
            "corrected_campaign": _CAMPAIGN,
            "corrected_campaign_sha256": _CAMPAIGN_SHA256,
            "corrected_campaign_status": (
                "MEASURED / campaign NEEDS-MORE / early formulation NO-GO / no admission"
            ),
            "terminal_receipt": _TERMINAL,
            "terminal_receipt_sha256": _TERMINAL_SHA256,
            "terminal_status": "MEASURED / formulation NO-GO / no admission",
            "final_campaign": _FINAL_CAMPAIGN,
            "final_campaign_sha256": _FINAL_CAMPAIGN_SHA256,
            "final_campaign_status": (
                "MEASURED / tested-formulation NO-GO / full-K20 fidelity UNKNOWN / no admission"
            ),
            "review_status": "externally_tracked_by_content_hash",
        },
        units_in={
            "x_t": "receiver_realized_frame_units",
            "lambda": "teacher_loss_per_frame_unit",
            "time": "seconds_per_whole_step",
        },
        units_out={"lambda_hat": "teacher_loss_per_frame_unit", "S_K": "dimensionless_ratio"},
        empirical_anchors=(anchor, corrected_anchor, terminal_anchor, final_anchor),
        predicted_vs_empirical_residual={"common_admitted_cadence_missing": 1.0},
        last_calibration_utc=_FINAL_CAMPAIGN_UTC,
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.onpolicy_scorer_surrogate_policy",
            "tac.scorer_surrogate.onpolicy_matched_verdict",
        ),
        canonical_producers=(
            "tac.scorer_surrogate.onpolicy_costate",
            "tac.scorer_surrogate.amortized_onpolicy_costate",
        ),
        provenance=final_provenance,
    )


def populate_onpolicy_input_costate_surrogate_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_onpolicy_input_costate_surrogate_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "task455 final joint-control campaign; research_only; tested formulation NO-GO; "
            "full-K20 fidelity UNKNOWN"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_onpolicy_input_costate_surrogate_v1",
    "populate_onpolicy_input_costate_surrogate_v1",
]
