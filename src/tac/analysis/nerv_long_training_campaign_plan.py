# SPDX-License-Identifier: MIT
"""Queue-safe long-training campaign plan for HiNeRV and SNeRV.

This module turns modelsize candidates into concrete MLX-first campaign rows.
It is not an executor and never grants score authority. Its job is to keep the
top-priority carrier race dynamic: modelsize, optimizer, scorer pressure, QAT,
archive proof, MLX prefilter, local CPU replay, and exact-auth promotion gates
are compiled into the same row so future agents do not hand-launch arbitrary
partial runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.adaptation.hard_pair_indices import (
    HardPairIndicesError,
    normalize_pair_indices,
    pair_indices_from_mapping,
)
from tac.analysis.action_commutator import build_commutator_ledger
from tac.analysis.action_effect import ACTION_EFFECT_V1_SCHEMA, ActionEffect
from tac.analysis.nerv_candidate_curriculum import (
    build_hinerv_candidate_curriculum_plan,
    build_snerv_candidate_curriculum_plan,
)
from tac.analysis.nerv_candidate_feedback import (
    FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
    build_nerv_candidate_feedback_row,
    recommend_segnet_distillation_weight_for_stagnation,
)
from tac.analysis.nerv_candidate_feedback import (
    SCHEMA as NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA,
)
from tac.analysis.nerv_decoder_weight_waterfill import (
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
    TRUSTED_RECEIVER_PROOF_STATUSES,
)
from tac.analysis.nerv_long_run_launch_gate import NERV_LONG_RUN_LAUNCH_GATE_SCHEMA
from tac.analysis.nerv_modelsize_budget import (
    NervModelSizeBudgetError,
    decoder_codec_nominal_bits,
    snerv_decoder_codec_nominal_bits,
    snerv_modelsize_candidate_id_from_controls,
)
from tac.analysis.nerv_scorer_objective import (
    PEIRCE_P1_CONTEST_SCORER_GEOMETRY,
)
from tac.analysis.nerv_source_parity_contract import build_nerv_source_parity_contract
from tac.analysis.nerv_witness_readiness_dag import (
    build_distortion_birth_before_rate_pressure_evidence,
)
from tac.analysis.pr95_distortion_practices_guard import (
    build_pr95_distortion_axis_trace_contract,
    build_pr95_distortion_practices_row_guard,
    build_pr95_distortion_source_inventory,
    build_pr95_evaluate_scorer_domain_telemetry_contract,
    build_pr95_posenet_marginal_telemetry_contract,
    build_pr95_scorer_atom_actuator_contract,
)
from tac.analysis.snerv_lf_hf_replacement_queue import (
    DEFAULT_QUEUE_ID as DEFAULT_SNERV_LF_HF_REPLACEMENT_QUEUE_ID,
)
from tac.analysis.snerv_lf_hf_replacement_queue import (
    build_snerv_lf_hf_replacement_queue,
    summarize_snerv_lf_hf_source_forward_evidence,
)
from tac.analysis.snerv_lf_over_ceiling_reroute_queue import (
    DEFAULT_QUEUE_ID as DEFAULT_SNERV_LF_REROUTE_QUEUE_ID,
)
from tac.analysis.snerv_lf_over_ceiling_reroute_queue import (
    build_snerv_lf_over_ceiling_reroute_queue,
)
from tac.analysis.snerv_lf_payload_archive_recode import (
    build_snerv_lf_payload_recode_admission_plan,
)
from tac.analysis.snerv_source_forward_proof import (
    SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA,
    validate_snerv_source_forward_proof_action_effect,
)
from tac.contest_eval_contract import build_score_allocation_contract
from tac.optimization.recon_pixel_weight_surface import (
    JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA,
)
from tac.substrates._shared.mlx_score_aware.adapter import (
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
)

SCHEMA = "nerv_long_training_campaign_plan.v1"
ROW_SCHEMA = "nerv_long_training_campaign_row.v1"
EXPERIMENT_QUEUE_SCHEMA = "experiment_queue.v1"
DEFAULT_EXPERIMENT_QUEUE_ID = "nerv_long_training_campaign_queue.v1"
SCORE_LOWERING_GATE_SCHEMA = "nerv_long_training_score_lowering_gate.v1"
RECEIVER_SURFACE_TRACE_CONTRACT_SCHEMA = "nerv_receiver_surface_trace_contract.v1"
ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA = (
    "nerv_archive_parseback_selection_contract.v1"
)
ACTION_EFFECT_PLANNING_BUNDLE_SCHEMA = "nerv_action_effect_planning_bundle.v1"
ACTION_EFFECT_ATLAS_SCHEMA = "nerv_action_effect_atlas.v1"
ACTION_EFFECT_SELECTOR_PLANNING_SCHEMA = "nerv_action_effect_selector_planning.v1"
HINERV_DISTORTION_BIRTH_RATE_GATE_SCHEMA = (
    "hinerv_distortion_birth_before_rate_pressure_gate.v1"
)
DISTORTION_BIRTH_RATE_EVIDENCE_SCHEMA = (
    "nerv_distortion_birth_before_rate_pressure_evidence.v1"
)
DEFAULT_OUTPUT_ROOT = "/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns"
DEFAULT_EPOCHS = 29_650
DEFAULT_BATCH_PAIRS = 8
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_SNERV_BOUNDED_PROOF_PAIR_COUNT = 16
DEFAULT_HINERV_TELEMETRY_FLUSH_INTERVAL_EPOCHS = 1
DEFAULT_CODER_QAT_QUANT_RESIDUAL_WEIGHT = 1.0e-3
DEFAULT_CODER_QAT_MAGNITUDE_WEIGHT = 1.0e-4
DEFAULT_CODER_QAT_DELTA_WEIGHT = 2.0e-4
DEFAULT_CODER_QAT_C1A_ENTROPY_WEIGHT = 1.0e-4
DEFAULT_CODER_QAT_C1A_SIGMA = 0.2
DEFAULT_CODER_QAT_C1A_SAMPLE_SIZE = 512
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_DISTILLATION_WEIGHT = 0.25
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_HISTOGRAM_WEIGHT = 0.25
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_HINGE_WEIGHT = 0.5
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_CE_WEIGHT = 0.25
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_SQUARED_HINGE_WEIGHT = 0.25
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_REGION_RECON_WEIGHT = 0.25
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_RARE_CLASS_LOGIT_WEIGHT = 16.0
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_TARGET_MASS_FLOOR_WEIGHT = 0.4
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_TARGET_MIN_RATIO_FLOOR_WEIGHT = 0.4
DEFAULT_HINERV_POSE_DIRECT_LIVE_DISTILLATION_WEIGHT = 0.25
DEFAULT_HINERV_SEGNET_DIRECT_LIVE_OBJECTIVE = "boundary_argmax_hinge"
DEFAULT_HINERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT = 2.0
DEFAULT_HINERV_SCORER_INPUT_CONTRAST_FLOOR_WEIGHT = 0.5
DEFAULT_HINERV_SCORER_INPUT_CONTRAST_FLOOR_SEGNET_MIN_STD_RATIO = 0.6
DEFAULT_HINERV_SCORER_INPUT_CONTRAST_FLOOR_POSENET_YUV6_MIN_STD_RATIO = 0.6
DEFAULT_HINERV_SCORER_INPUT_SHAPE_TETHER_WEIGHT = 0.25
DEFAULT_HINERV_SCORER_STEP_GUARD_TARGET_CLASS_COVERAGE_FRACTION = 1.0
DEFAULT_HINERV_SCORER_STEP_GUARD_TARGET_CLASS_MIN_RATIO = 0.2
DEFAULT_HINERV_SCORER_STEP_GUARD_TARGET_CLASS_MAX_RATIO_DROP = 0.05
# HiNeRV short smokes on 2026-06-06 showed the dense YUV6 geometry tether is a
# real actuator but a poor default: it improved PoseNet proxy terms while
# burning SegNet argmax/distribution score. Keep it opt-in until a late-stage
# or dynamic schedule proves positive value-per-byte.
DEFAULT_HINERV_POSENET_YUV6_GEOMETRY_TETHER_WEIGHT = 0.0
DEFAULT_HINERV_POSENET_TEMPORAL_SIGNAL_FLOOR_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_DISTILLATION_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_HISTOGRAM_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_HINGE_WEIGHT = 0.5
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_CE_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_SQUARED_HINGE_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_REGION_RECON_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_RARE_CLASS_LOGIT_WEIGHT = 4.0
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_TARGET_MASS_FLOOR_WEIGHT = 0.5
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_TARGET_MIN_RATIO_FLOOR_WEIGHT = 0.5
DEFAULT_SNERV_POSE_DIRECT_LIVE_DISTILLATION_WEIGHT = 0.25
DEFAULT_SNERV_SEGNET_DIRECT_LIVE_OBJECTIVE = "boundary_argmax_hinge"
DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT = 2.0
DEFAULT_SNERV_SCORER_INPUT_CONTRAST_FLOOR_WEIGHT = 0.5
DEFAULT_SNERV_SCORER_INPUT_CONTRAST_FLOOR_SEGNET_MIN_STD_RATIO = 0.6
DEFAULT_SNERV_SCORER_INPUT_CONTRAST_FLOOR_POSENET_YUV6_MIN_STD_RATIO = 0.6
DEFAULT_SNERV_SCORER_INPUT_SHAPE_TETHER_WEIGHT = 0.25
DEFAULT_SNERV_SCORER_STEP_GUARD_TARGET_CLASS_COVERAGE_FRACTION = 1.0
DEFAULT_SNERV_SCORER_STEP_GUARD_TARGET_CLASS_MIN_RATIO = 0.2
DEFAULT_SNERV_SCORER_STEP_GUARD_TARGET_CLASS_MAX_RATIO_DROP = 0.05
DEFAULT_SNERV_POSENET_YUV6_GEOMETRY_TETHER_WEIGHT = 0.5
DEFAULT_SNERV_POSENET_TEMPORAL_SIGNAL_FLOOR_WEIGHT = 0.25
# Do not classify the first 9e-5 pose-spike as repeated low-LR failure: its
# telemetry explicitly requested a 2.7e-5 recovery run. The Huber path is real,
# but it is reserved for repeated instability at or below that recovered regime;
# otherwise the planner skips a measured LR recovery and confounds two causes.
HINERV_POSE_INSTABILITY_LOW_LR_FLOOR = 3.0e-5
HINERV_POSE_INSTABILITY_POLICY_LOGIC = (
    "pose instability above low_learning_rate_floor applies the measured lower "
    "learning-rate recommendation; repeated instability at or below the floor "
    "switches to pose_distillation_loss=huber while preserving raw MSE telemetry; "
    "segnet stagnation raises segnet_distillation_weight for the next run without "
    "granting archive, replay, or score authority"
)
HINERV_POSE_PROTECTED_LOSS = "huber"
HINERV_POSE_PROTECTED_HUBER_DELTA = 1.0
_AUTHORITY_TRUE_KEYS: tuple[str, ...] = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "production_hardened_claim",
    "ready_for_exact_eval_dispatch",
)
HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION = "switch_to_hinerv_official_feature_grid_convnext_controls"
HINERV_WATERFILL_CANDIDATE_BINDING_FIELDS: tuple[str, ...] = (
    "num_pairs",
    "latent_dim",
    "latent_dim_coarse",
    "latent_dim_mid",
    "latent_dim_fine",
    "embed_dim",
    "decoder_channel",
    "decoder_channels",
    "decoder_codec",
    "use_hierarchical_feature_grid",
    "use_convnext_blocks",
    "local_grid_levels",
    "local_grid_channels",
    "convnext_mlp_ratio",
    "convnext_kernel_size",
    "mid_injection_block_index",
    "fine_injection_block_index",
    "modelsize_mparams",
    "target_modelsize_mparams",
    "hard_byte_ceiling",
)
HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA = "hinerv_archive_section_telemetry.v1"
DEFAULT_OPTIMIZER_KINDS = (
    "pact_muon_adamw",
    "adamw",
    "muon",
    "lion",
    "adamax",
    "rmsprop",
    "adafactor",
    "adam",
    "adagrad",
    "adadelta",
    "sgd",
)
AURORA_LIKE_OPTIMIZER_KIND = "aurora_like"
_OPTIMIZER_KIND_ALIASES = {
    "aurora": AURORA_LIKE_OPTIMIZER_KIND,
}
TIMING_SMOKE_OPTIMIZER_KINDS = (AURORA_LIKE_OPTIMIZER_KIND,)
_TIMING_SMOKE_OPTIMIZER_LAUNCH_BLOCKERS: dict[str, tuple[str, ...]] = {
    AURORA_LIKE_OPTIMIZER_KIND: ("aurora_requires_local_timing_convergence_smoke",)
}
FIRST_PASS_OPTIMIZER_KINDS = frozenset(("pact_muon_adamw", "adamw", "muon", "lion", "adamax"))
OPTIMIZER_CONTROL_SCHEMA = "nerv_optimizer_control_surface.v1"
HINERV_OPTIMIZER_POLICY_SCHEMA = "nerv_hinerv_optimizer_policy.v1"
UPSTREAM_EVALUATE_PRIORITY_CONTRACT_SCHEMA = "nerv_upstream_evaluate_priority_contract.v1"
ROW_UPSTREAM_EVALUATE_BINDING_SCHEMA = "nerv_row_upstream_evaluate_binding.v1"
TILDE_OSS_LEVERAGE_POLICY_SCHEMA = "nerv_tilde_oss_leverage_policy.v1"
ROW_TILDE_OSS_BINDING_SCHEMA = "nerv_row_tilde_oss_binding.v1"
PR95_BASELINE_IDENTITY_BINDING_SCHEMA = "nerv_pr95_baseline_identity_binding.v1"
SNERV_SCORER_TETHER_SMOKE_GATE_SCHEMA = "snerv_scorer_tether_smoke_gate.v1"
SNERV_RENDERER_NONDEGENERATE_GATE_SCHEMA = "snerv_renderer_nondegenerate_gate.v1"
SNERV_PRE_LONG_RUN_EVIDENCE_GATE_SCHEMA = "snerv_pre_long_run_evidence_gate.v1"
SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT = 16
SNERV_SCORER_TETHER_FEEDBACK_BLOCKERS = frozenset(
    {
        "snerv_scorer_domain_tether_missing_telemetry",
        "snerv_posenet_yuv6_pair_distill_metric_missing_telemetry",
        "snerv_segnet_last_frame_distill_metric_missing_telemetry",
        "snerv_scorer_domain_tether_lambda_inactive_telemetry",
        "snerv_score_aware_long_training_dual_segnet_metric_never_observed",
        "snerv_score_aware_long_training_dual_posenet_metric_never_observed",
        "snerv_score_aware_long_training_dual_segnet_lambda_never_active",
        "snerv_score_aware_long_training_dual_posenet_lambda_never_active",
    }
)


class NervLongTrainingCampaignPlanError(ValueError):
    """Raised when a long-training campaign plan is malformed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _upstream_evaluate_priority_contract() -> dict[str, Any]:
    score_allocation = build_score_allocation_contract()
    scorer_geometry = PEIRCE_P1_CONTEST_SCORER_GEOMETRY.as_false_authority_payload()
    return {
        "schema": UPSTREAM_EVALUATE_PRIORITY_CONTRACT_SCHEMA,
        "source": "upstream/evaluate.py",
        "baseline_to_beat": "full_pr95_fidelity_or_better_on_exact_upstream_evaluate_axes",
        "applies_to_families": ["hi_nerv", "snerv"],
        "objective": (
            "Allocate SNeRV/HiNeRV capacity against the official evaluator: "
            "SegNet last-frame hard argmax, PoseNet two-frame YUV6 pose MSE, "
            "and exact archive.zip byte price."
        ),
        "optimizer_target_terms": [
            "SegNet_last_frame_argmax_distortion",
            "PoseNet_two_frame_yuv6_first_six_pose_dims",
            "archive_zip_bytes_rate_term",
        ],
        "non_authority_terms": [
            "human_visual_fidelity",
            "PSNR_without_scorer_causal_evidence",
            "SSIM_without_scorer_causal_evidence",
            "inflated_raw_bytes_as_rate_denominator",
            "nominal_modelsize_as_archive_bytes",
        ],
        "crux": {
            "segnet_frame0_direct_weight": 0.0,
            "segnet_frame1_direct_weight": 1.0,
            "posenet_frame0_direct_weight": 1.0,
            "posenet_frame1_direct_weight": 1.0,
            "pose_marginal_formula": score_allocation["posenet"]["derivative_wrt_d_pose"],
            "rate_price_per_archive_byte": score_allocation["rate"]["rate_price_per_archive_byte"],
            "canonical_rate_denominator_bytes": score_allocation["rate"]["canonical_denominator_bytes"],
        },
        "row_binding_required": True,
        "promotion_boundary": "archive.zip plus deterministic inflate runtime through upstream evaluate.py",
        "score_allocation_contract": score_allocation,
        "scorer_geometry": scorer_geometry,
        **FALSE_AUTHORITY,
    }


def _row_upstream_evaluate_binding(
    *,
    family: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    score_allocation = contract.get("score_allocation_contract")
    if not isinstance(score_allocation, Mapping):
        score_allocation = build_score_allocation_contract()
    return {
        "schema": ROW_UPSTREAM_EVALUATE_BINDING_SCHEMA,
        "family": str(family),
        "contract_schema": contract.get("schema"),
        "source": contract.get("source"),
        "baseline_to_beat": contract.get("baseline_to_beat"),
        "optimizer_target_terms": list(contract.get("optimizer_target_terms") or ()),
        "rate": {
            "archive_authority": score_allocation["rate"]["archive_authority"],
            "canonical_denominator_bytes": score_allocation["rate"]["canonical_denominator_bytes"],
            "rate_price_per_archive_byte": score_allocation["rate"]["rate_price_per_archive_byte"],
            "raw_output_shape_bytes_are_not_rate_denominator": score_allocation["rate"][
                "raw_output_shape_bytes_are_not_rate_denominator"
            ],
        },
        "pair_geometry": {
            "seq_len": score_allocation["pair_geometry"]["seq_len"],
            "public_test_pair_count": score_allocation["pair_geometry"]["public_test_pair_count"],
            "camera_size_wh": list(score_allocation["pair_geometry"]["camera_size_wh"]),
            "candidate_raw_shape": list(score_allocation["pair_geometry"]["candidate_raw_shape"]),
        },
        "segnet": {
            "coefficient": score_allocation["segnet"]["coefficient"],
            "frame_scope": score_allocation["segnet"]["frame_scope"],
            "scored_frame_index_within_pair": score_allocation["segnet"]["scored_frame_index_within_pair"],
            "unscored_frame_index_within_pair": score_allocation["segnet"]["unscored_frame_index_within_pair"],
            "distortion": score_allocation["segnet"]["distortion"],
        },
        "posenet": {
            "frame_scope": score_allocation["posenet"]["frame_scope"],
            "scored_frame_indices_within_pair": list(score_allocation["posenet"]["scored_frame_indices_within_pair"]),
            "input_domain": score_allocation["posenet"]["input_domain"],
            "distortion": score_allocation["posenet"]["distortion"],
            "derivative_wrt_d_pose": score_allocation["posenet"]["derivative_wrt_d_pose"],
        },
        "authority": {
            "row_is_optimizer_guidance_only": True,
            "score_authority_requires": score_allocation["authority"]["score_authority_requires"],
            "receiver_contract": score_allocation["authority"]["receiver_contract"],
        },
        **FALSE_AUTHORITY,
    }


def _tilde_oss_leverage_policy() -> dict[str, Any]:
    return {
        "schema": TILDE_OSS_LEVERAGE_POLICY_SCHEMA,
        "source_intake": "xhigh_research_sidecar_20260604",
        "applies_to_families": ["hi_nerv", "snerv"],
        "implementation_order": [
            "reuse_or_refresh_exact_pr95_baseline_identity_on_upstream_evaluate_axes",
            "aurora_like_pr95_hinerv_timing_convergence_smoke",
            "pact_native_snerv_wall_style_lf_tub_gate_byte_charged_side_smoke",
        ],
        "aurora": {
            "official_tilde_surface": True,
            "allowed_use": "optimizer_timing_convergence_smoke_only",
            "planner_optimizer_kind": AURORA_LIKE_OPTIMIZER_KIND,
            "runtime_archive_payload_import_allowed": False,
            "score_claim": False,
        },
        "wall_attention": {
            "official_tilde_surface": True,
            "allowed_use": ("concept_only_pact_native_snerv_lf_tub_temporal_gate_with_receiver_bytes"),
            "direct_kernel_import_allowed": False,
            "byte_charged_receiver_replay_required": True,
            "score_claim": False,
        },
        "parallax": {
            "official_tilde_surface": False,
            "classification": "llm_local_linear_attention_not_video_parallax_geometry",
            "allowed_use": "concept_only_architecture_optimizer_codesign_signal",
            "direct_runtime_import_allowed": False,
            "runtime_blockers": [
                "torch_triton_hopper_cute_runtime_debt",
                "no_inflate_raw_rgb_contract",
                "no_archive_zip_byte_closed_component",
            ],
        },
        "nitrobrew": {
            "official_tilde_surface": True,
            "allowed_use": ("concept_only_streaming_accumulation_for_local_scorer_memory_reduction"),
            "codec_or_submission_claim_allowed": False,
        },
        "direct_import_policy": {
            "forbidden_repos": [
                "Yifei-Zuo/Parallax",
                "tilde-research/wall-attention-release",
            ],
            "reason": (
                "external LLM kernels are not contest inflate grammar and would "
                "add runtime dependency debt without archive.zip scorer evidence"
            ),
            "pact_native_reimplementation_required_for_receiver_runtime": True,
        },
        **FALSE_AUTHORITY,
    }


def _row_tilde_oss_binding(
    *,
    family: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ROW_TILDE_OSS_BINDING_SCHEMA,
        "family": str(family),
        "policy_schema": policy.get("schema"),
        "implementation_order": list(policy.get("implementation_order") or ()),
        "aurora_like_optimizer_smoke_allowed": True,
        "aurora_like_optimizer_kind": AURORA_LIKE_OPTIMIZER_KIND,
        "parallax_direct_runtime_import_allowed": False,
        "wall_attention_direct_kernel_import_allowed": False,
        "pact_native_receiver_byte_charged_required": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _pr95_baseline_identity_binding(
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {
            "schema": PR95_BASELINE_IDENTITY_BINDING_SCHEMA,
            "attached": False,
            "reason": "pr95_baseline_identity_missing",
            "selected_archive": None,
            "local_cpu_mlx_work_order": None,
            "modal_dispatch_policy": None,
            "paired_exact_eval_work_order": None,
            "blockers": ["pr95_baseline_identity_missing"],
            **FALSE_AUTHORITY,
        }
    selected_archive = source.get("selected_reusable_candidate_archive")
    exact_axis_status = source.get("exact_axis_status")
    local_work_order = source.get("local_cpu_mlx_work_order")
    modal_policy = source.get("modal_dispatch_policy")
    work_order = source.get("paired_exact_eval_work_order")
    source_blockers = [str(blocker) for blocker in source.get("blockers") or () if blocker]
    structural_blockers = []
    if source.get("schema") != "pr95_baseline_identity.v1":
        structural_blockers.append("pr95_baseline_identity_schema_mismatch")
    if not isinstance(selected_archive, Mapping):
        structural_blockers.append("pr95_baseline_identity_selected_archive_missing")
    if not isinstance(exact_axis_status, Mapping):
        structural_blockers.append("pr95_baseline_identity_exact_axis_status_missing")
    if not isinstance(local_work_order, Mapping):
        structural_blockers.append("pr95_baseline_identity_local_cpu_mlx_work_order_missing")
    if not isinstance(modal_policy, Mapping):
        structural_blockers.append("pr95_baseline_identity_modal_dispatch_policy_missing")
    if not isinstance(work_order, Mapping):
        structural_blockers.append("pr95_baseline_identity_paired_work_order_missing")
    blockers = _dedupe([*structural_blockers, *source_blockers])
    return {
        "schema": PR95_BASELINE_IDENTITY_BINDING_SCHEMA,
        "attached": not structural_blockers and isinstance(selected_archive, Mapping),
        "baseline_id": source.get("baseline_id"),
        "baseline_identity_reusable": bool(source.get("baseline_identity_reusable")),
        "selected_archive": (dict(selected_archive) if isinstance(selected_archive, Mapping) else None),
        "exact_axis_status": (dict(exact_axis_status) if isinstance(exact_axis_status, Mapping) else None),
        "local_cpu_mlx_work_order": (dict(local_work_order) if isinstance(local_work_order, Mapping) else None),
        "modal_dispatch_policy": (dict(modal_policy) if isinstance(modal_policy, Mapping) else None),
        "paired_exact_eval_work_order": (dict(work_order) if isinstance(work_order, Mapping) else None),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _snerv_scorer_tether_smoke_gate(
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {
            "schema": SNERV_SCORER_TETHER_SMOKE_GATE_SCHEMA,
            "attached": False,
            "passed": False,
            "required_by_default": True,
            "blockers": ["snerv_scorer_tether_smoke_report_missing"],
            **FALSE_AUTHORITY,
        }
    blockers: list[str] = []
    accepted_schemas = {
        "snerv_scorer_tether_smoke.v1",
        "snerv_score_aware_long_training_scorer_tether_gate.v1",
    }
    if source.get("schema") not in accepted_schemas:
        blockers.append("snerv_scorer_tether_smoke_schema_mismatch")
    if source.get("passed") is not True:
        blockers.append("snerv_scorer_tether_smoke_failed")
    blockers.extend(str(blocker) for blocker in source.get("blockers") or [] if blocker)
    return {
        "schema": SNERV_SCORER_TETHER_SMOKE_GATE_SCHEMA,
        "attached": True,
        "passed": not blockers,
        "required_by_default": True,
        "source_schema": source.get("schema"),
        "source_created_utc": source.get("created_utc"),
        "source_steps": source.get("steps"),
        "source_metric_summary": source.get("metric_summary"),
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _snerv_scorer_tether_smoke_gate_with_candidate_feedback(
    gate: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    """Let attached candidate feedback satisfy the tether gate, fail-closed."""

    out = dict(gate)
    if out.get("passed") is True:
        return out
    health = feedback.get("snerv_scorer_domain_tether_health")
    if not isinstance(health, Mapping) or health.get("passed") is not True:
        return out
    blockers = [
        str(blocker)
        for blocker in out.get("blockers") or []
        if str(blocker) != "snerv_scorer_tether_smoke_report_missing"
    ]
    out.update(
        {
            "attached": True,
            "passed": not blockers,
            "source_schema": health.get("schema"),
            "source_metric_summary": health.get("metric_health"),
            "source": "candidate_feedback_snerv_scorer_domain_tether_health",
            "candidate_feedback_source_report_path": feedback.get("source_report_path"),
            "blockers": _dedupe(blockers),
        }
    )
    return out


def _snerv_candidate_feedback_scorer_input_guard_passed(
    feedback: Mapping[str, Any],
) -> bool:
    proof = feedback.get("snerv_scorer_input_distribution_guard_proof")
    return bool(isinstance(proof, Mapping) and proof.get("passed") is True)


def _snerv_renderer_nondegenerate_gate(
    *,
    feedback: Mapping[str, Any],
    bounded_proof_only: bool,
) -> dict[str, Any]:
    proof = (
        dict(feedback.get("snerv_renderer_nondegenerate_proof"))
        if isinstance(feedback.get("snerv_renderer_nondegenerate_proof"), Mapping)
        else {}
    )
    blockers: list[str] = []
    required = not bool(bounded_proof_only)
    measured_pairs = _first_present_int(
        proof,
        ("measured_num_pairs", "candidate_num_pairs", "num_pairs"),
    )
    if required:
        if not proof:
            blockers.append("snerv_renderer_nondegenerate_smoke_missing")
        elif proof.get("passed") is not True:
            blockers.append("snerv_renderer_nondegenerate_smoke_failed")
        if measured_pairs is None or int(measured_pairs) < SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT:
            blockers.append("snerv_renderer_nondegenerate_smoke_min16_pairs_missing")
        blockers.extend(str(blocker) for blocker in proof.get("blockers") or () if blocker)
    blockers = _dedupe(blockers)
    return {
        "schema": SNERV_RENDERER_NONDEGENERATE_GATE_SCHEMA,
        "required": required,
        "proof_attached": bool(proof),
        "proof_passed": bool(proof.get("passed") is True) if proof else False,
        "passed": not blockers,
        "min_pair_count": SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT,
        "measured_num_pairs": measured_pairs,
        "proof": proof or None,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _snerv_pre_long_run_evidence_gate(
    *,
    feedback: Mapping[str, Any],
    bounded_proof_only: bool,
) -> dict[str, Any]:
    """Require byte-closed scorer evidence before launching real SNeRV long runs."""

    required = not bool(bounded_proof_only)
    direct_feedback_blockers = [
        str(blocker) for blocker in feedback.get("direct_feedback_blockers") or () if blocker
    ]
    feedback_kind = str(feedback.get("feedback_kind") or "").strip()
    byte_closed_feedback_kinds = {"full_video_mlx_scorer_response"}
    blockers: list[str] = []
    if required:
        if not feedback:
            blockers.append("snerv_pre_long_run_candidate_feedback_missing")
        elif feedback_kind == "training_telemetry":
            blockers.append("snerv_pre_long_run_candidate_feedback_not_byte_closed")
        elif feedback.get("context_only") is True:
            blockers.append("snerv_pre_long_run_candidate_feedback_context_only")
        elif feedback_kind not in byte_closed_feedback_kinds:
            blockers.append("snerv_pre_long_run_full_video_mlx_feedback_kind_missing")
        if feedback.get("feedback_ready") is False:
            blockers.append("snerv_pre_long_run_candidate_feedback_not_ready")
        if not (
            feedback.get("receiver_proof_attached") is True
            and feedback.get("native_mlx_receiver_proof_passed") is True
        ):
            blockers.append("snerv_pre_long_run_receiver_proof_missing_or_failed")
        if feedback.get("full_video_local_prefilter_attached") is not True:
            blockers.append("snerv_pre_long_run_full_video_mlx_prefilter_missing")
        if feedback.get("full_video_mlx_response_attached") is not True:
            blockers.append("snerv_pre_long_run_full_video_mlx_scorer_response_missing")
        if feedback.get("native_mlx_full600_campaign_ready") is not True:
            blockers.append("snerv_pre_long_run_native_full600_export_not_ready")
        if feedback.get("native_mlx_scorer_loop_qat_best_materialized") is not True:
            blockers.append("snerv_pre_long_run_scorer_loop_best_packet_not_materialized")
        if any(str(blocker).startswith("direct_feedback_") for blocker in direct_feedback_blockers):
            blockers.append("snerv_pre_long_run_feedback_custody_paths_missing")
        blockers.extend(
            blocker
            for blocker in direct_feedback_blockers
            if str(blocker).startswith("direct_feedback_")
        )
    blockers = _dedupe(blockers)
    return {
        "schema": SNERV_PRE_LONG_RUN_EVIDENCE_GATE_SCHEMA,
        "required": required,
        "passed": not blockers,
        "feedback_kind": feedback_kind,
        "context_only": bool(feedback.get("context_only")),
        "receiver_proof_attached": bool(feedback.get("receiver_proof_attached")),
        "native_mlx_receiver_proof_passed": bool(
            feedback.get("native_mlx_receiver_proof_passed")
        ),
        "full_video_local_prefilter_attached": bool(
            feedback.get("full_video_local_prefilter_attached")
        ),
        "full_video_mlx_response_attached": bool(
            feedback.get("full_video_mlx_response_attached")
        ),
        "native_mlx_full600_campaign_ready": bool(
            feedback.get("native_mlx_full600_campaign_ready")
        ),
        "native_mlx_scorer_loop_qat_best_materialized": bool(
            feedback.get("native_mlx_scorer_loop_qat_best_materialized")
        ),
        "direct_feedback_blockers": direct_feedback_blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def build_nerv_long_training_campaign_plan(
    *,
    hinerv_modelsize_budget: Mapping[str, Any],
    snerv_modelsize_budget: Mapping[str, Any],
    optimizer_kinds: Sequence[str] = DEFAULT_OPTIMIZER_KINDS,
    epochs: int = DEFAULT_EPOCHS,
    batch_pairs: int = DEFAULT_BATCH_PAIRS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_candidates_per_family: int = 3,
    joint_recon_weight_manifest_paths: Sequence[str | Path] = (),
    candidate_feedback_sources: Sequence[Mapping[str, Any]] = (),
    modelsize_byte_cap_feedback_paths: Sequence[str | Path] = (),
    hinerv_distortion_birth_evidence_sources: Sequence[Mapping[str, Any]] = (),
    decoder_weight_waterfill_sources: Sequence[Mapping[str, Any]] = (),
    archive_section_telemetry_sources: Sequence[Mapping[str, Any]] = (),
    snerv_lf_payload_recode_sources: Sequence[Mapping[str, Any]] = (),
    snerv_lf_payload_byte_report_sources: Sequence[Mapping[str, Any]] = (),
    snerv_snar_header_grammar_profile_sources: Sequence[Mapping[str, Any]] = (),
    snerv_snar_header_minimization_report_sources: Sequence[Mapping[str, Any]] = (),
    snerv_official_source_audit: Mapping[str, Any] | None = None,
    snerv_official_source_forward_artifacts: Sequence[Mapping[str, Any]] = (),
    snerv_long_run_launch_gate_verdict: Mapping[str, Any] | None = None,
    snerv_official_replacement_authority_gates: Sequence[Mapping[str, Any]] = (),
    snerv_value_domain_xray_reports: Sequence[Mapping[str, Any]] = (),
    snerv_hf_residual_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    snerv_joint_codebook_receiver_payload_proofs: Sequence[Mapping[str, Any]] = (),
    snerv_temporal_lf_predictor_receiver_payload_proofs: Sequence[
        Mapping[str, Any]
    ] = (),
    snerv_lf_super_resolution_receiver_payload_proofs: Sequence[
        Mapping[str, Any]
    ] = (),
    snerv_spectral_band_allocator_receiver_payload_proofs: Sequence[
        Mapping[str, Any]
    ] = (),
    snerv_lf_latent_hyperprior_receiver_payload_proofs: Sequence[
        Mapping[str, Any]
    ] = (),
    snerv_lf_hf_runtime_binding_proofs: Sequence[Mapping[str, Any]] = (),
    action_effect_sources: Sequence[Mapping[str, Any]] = (),
    pr95_baseline_identity: Mapping[str, Any] | None = None,
    snerv_scorer_tether_smoke_report: Mapping[str, Any] | None = None,
    snerv_bounded_proof_only: bool = False,
    snerv_bounded_proof_epochs: int = 3,
    snerv_bounded_proof_pair_count: int = DEFAULT_SNERV_BOUNDED_PROOF_PAIR_COUNT,
    experiment_queue_id: str = DEFAULT_EXPERIMENT_QUEUE_ID,
    planner_row_queue_artifact_path: str | Path | None = None,
    snerv_lf_hf_replacement_queue_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the shared HiNeRV/SNeRV long-training campaign matrix."""

    _require_schema(
        hinerv_modelsize_budget,
        "nerv_modelsize_budget.v1",
        "hinerv_modelsize_budget",
    )
    _require_schema(
        snerv_modelsize_budget,
        "snerv_modelsize_budget.v1",
        "snerv_modelsize_budget",
    )
    optimizers = _optimizer_tuple(optimizer_kinds)
    if int(epochs) < 8:
        raise NervLongTrainingCampaignPlanError("epochs must be >= 8")
    if int(batch_pairs) <= 0:
        raise NervLongTrainingCampaignPlanError("batch_pairs must be positive")
    if float(learning_rate) <= 0.0:
        raise NervLongTrainingCampaignPlanError("learning_rate must be positive")
    queue_id = str(experiment_queue_id or "").strip()
    if not queue_id:
        raise NervLongTrainingCampaignPlanError("experiment_queue_id must be non-empty")
    joint_recon_weight_artifacts = _load_verified_joint_recon_weight_artifacts(joint_recon_weight_manifest_paths)
    candidate_feedback_index = _candidate_feedback_index(candidate_feedback_sources)
    hinerv_distortion_birth_evidence = tuple(
        _normalize_hinerv_distortion_birth_evidence_source(source)
        for source in hinerv_distortion_birth_evidence_sources
    )
    decoder_weight_waterfill_index = _decoder_weight_waterfill_index(decoder_weight_waterfill_sources)
    archive_section_telemetry_index = _archive_section_telemetry_index(archive_section_telemetry_sources)
    byte_cap_feedback_paths = tuple(Path(path).as_posix() for path in modelsize_byte_cap_feedback_paths)
    planner_queue_artifact = (
        None if planner_row_queue_artifact_path is None else Path(planner_row_queue_artifact_path).as_posix()
    )
    source_parity_contract = build_nerv_source_parity_contract(
        repo_root=_repo_root(),
        families=("hi_nerv", "snerv"),
        snerv_official_source_audit=snerv_official_source_audit,
    )
    upstream_evaluate_priority_contract = _upstream_evaluate_priority_contract()
    tilde_oss_leverage_policy = _tilde_oss_leverage_policy()
    pr95_baseline_binding = _pr95_baseline_identity_binding(pr95_baseline_identity)
    pr95_distortion_source_inventory = build_pr95_distortion_source_inventory(_repo_root())
    snerv_scorer_tether_smoke_gate = _snerv_scorer_tether_smoke_gate(snerv_scorer_tether_smoke_report)
    snerv_source_forward_evidence = (
        summarize_snerv_lf_hf_source_forward_evidence(
            snerv_official_source_forward_artifacts
        )
        if snerv_official_source_forward_artifacts
        else None
    )
    snerv_official_replacement_authority_gate = (
        _select_snerv_official_replacement_authority_gate(
            snerv_official_replacement_authority_gates
        )
    )

    rows: list[dict[str, Any]] = []
    hi_candidates = _selected_candidates(
        hinerv_modelsize_budget,
        family="hi_nerv",
        limit=max_candidates_per_family,
    )
    hi_candidates = _merge_hinerv_waterfill_candidate_evidence(
        candidates=hi_candidates,
        decoder_weight_waterfill_index=decoder_weight_waterfill_index,
        limit=max_candidates_per_family,
    )
    hi_candidates = _merge_modelsize_byte_cap_feedback_candidates(
        selected_candidates=hi_candidates,
        modelsize_budget=hinerv_modelsize_budget,
        family="hi_nerv",
        feedback_paths=byte_cap_feedback_paths,
        limit=max_candidates_per_family,
    )
    snerv_candidates = _selected_candidates(
        snerv_modelsize_budget,
        family="snerv",
        limit=max_candidates_per_family,
    )
    snerv_candidates = _merge_modelsize_byte_cap_feedback_candidates(
        selected_candidates=snerv_candidates,
        modelsize_budget=snerv_modelsize_budget,
        family="snerv",
        feedback_paths=byte_cap_feedback_paths,
        limit=max_candidates_per_family,
    )
    for candidate in hi_candidates:
        for optimizer in optimizers:
            rows.append(
                _hinerv_campaign_row(
                    candidate=candidate,
                    optimizer_kind=optimizer,
                    epochs=int(epochs),
                    batch_pairs=int(batch_pairs),
                    learning_rate=float(learning_rate),
                    output_root=Path(output_root),
                    joint_recon_weight_artifacts=joint_recon_weight_artifacts,
                    candidate_feedback_index=candidate_feedback_index,
                    hinerv_distortion_birth_evidence_sources=(
                        hinerv_distortion_birth_evidence
                    ),
                    decoder_weight_waterfill_index=decoder_weight_waterfill_index,
                    archive_section_telemetry_index=archive_section_telemetry_index,
                    source_parity_contract=source_parity_contract,
                    upstream_evaluate_priority_contract=(upstream_evaluate_priority_contract),
                    tilde_oss_leverage_policy=tilde_oss_leverage_policy,
                    pr95_baseline_identity_binding=pr95_baseline_binding,
                    pr95_distortion_source_inventory=(pr95_distortion_source_inventory),
                    planner_row_queue_artifact_path=planner_queue_artifact,
                    modelsize_byte_cap_feedback_paths=byte_cap_feedback_paths,
                )
            )
    for candidate in snerv_candidates:
        rows.append(
            _snerv_campaign_row(
                candidate=candidate,
                epochs=int(epochs),
                batch_pairs=int(batch_pairs),
                learning_rate=float(learning_rate),
                optimizer_kind="pact_muon_adamw",
                output_root=Path(output_root),
                candidate_feedback_index=candidate_feedback_index,
                bounded_proof_only=bool(snerv_bounded_proof_only),
                bounded_proof_epochs=int(snerv_bounded_proof_epochs),
                source_parity_contract=source_parity_contract,
                upstream_evaluate_priority_contract=(upstream_evaluate_priority_contract),
                tilde_oss_leverage_policy=tilde_oss_leverage_policy,
                pr95_baseline_identity_binding=pr95_baseline_binding,
                pr95_distortion_source_inventory=pr95_distortion_source_inventory,
                snerv_scorer_tether_smoke_gate=snerv_scorer_tether_smoke_gate,
                snerv_source_forward_evidence=snerv_source_forward_evidence,
                snerv_official_replacement_authority_gate=(
                    snerv_official_replacement_authority_gate
                ),
                snerv_long_run_launch_gate_verdict=snerv_long_run_launch_gate_verdict,
                planner_row_queue_artifact_path=planner_queue_artifact,
                modelsize_byte_cap_feedback_paths=byte_cap_feedback_paths,
                snerv_lf_payload_recode_sources=snerv_lf_payload_recode_sources,
                bounded_proof_pair_count=int(snerv_bounded_proof_pair_count),
            )
        )

    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("family") or ""),
            str(row.get("row_id") or ""),
        ),
    )
    action_effect_planning_bundle = _action_effect_planning_bundle(
        rows,
        action_effect_sources=action_effect_sources,
    )
    experiment_queue = _experiment_queue(rows, queue_id=queue_id)
    experiment_queue["action_effect_planning_bundle"] = action_effect_planning_bundle
    snerv_lf_over_ceiling_reroute_queue = build_snerv_lf_over_ceiling_reroute_queue(
        campaign_rows=rows,
        measured_lf_payload_sources=(
            *tuple(snerv_lf_payload_recode_sources),
            *tuple(snerv_lf_payload_byte_report_sources),
        ),
        measured_lf_payload_paths=byte_cap_feedback_paths,
        snar_header_grammar_profiles=snerv_snar_header_grammar_profile_sources,
        snar_header_minimization_reports=(snerv_snar_header_minimization_report_sources),
        output_root=Path(output_root) / "snerv_lf_over_ceiling_reroutes",
        queue_id=DEFAULT_SNERV_LF_REROUTE_QUEUE_ID,
    )
    snerv_lf_hf_replacement_queue = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=(
            *tuple(snerv_lf_payload_recode_sources),
            *tuple(snerv_lf_payload_byte_report_sources),
        ),
        reroute_queues=(snerv_lf_over_ceiling_reroute_queue,),
        campaign_plans=({"campaign_rows": rows},),
        source_forward_artifacts=snerv_official_source_forward_artifacts,
        official_replacement_authority_gates=(
            snerv_official_replacement_authority_gates
        ),
        candidate_feedback_rows=candidate_feedback_sources,
        value_domain_xray_reports=snerv_value_domain_xray_reports,
        hf_residual_receiver_payload_proofs=(
            snerv_hf_residual_receiver_payload_proofs
        ),
        joint_codebook_receiver_payload_proofs=(
            snerv_joint_codebook_receiver_payload_proofs
        ),
        temporal_lf_predictor_receiver_payload_proofs=(
            snerv_temporal_lf_predictor_receiver_payload_proofs
        ),
        lf_super_resolution_receiver_payload_proofs=(
            snerv_lf_super_resolution_receiver_payload_proofs
        ),
        spectral_band_allocator_receiver_payload_proofs=(
            snerv_spectral_band_allocator_receiver_payload_proofs
        ),
        lf_latent_hyperprior_receiver_payload_proofs=(
            snerv_lf_latent_hyperprior_receiver_payload_proofs
        ),
        lf_hf_runtime_binding_proofs=snerv_lf_hf_runtime_binding_proofs,
        output_root=Path(output_root) / "snerv_lf_hf_replacements",
        queue_id=DEFAULT_SNERV_LF_HF_REPLACEMENT_QUEUE_ID,
        queue_artifact_path=snerv_lf_hf_replacement_queue_artifact_path,
        allow_local_output=True,
    )
    decoder_weight_waterfill_unattached_sources = _decoder_weight_waterfill_unattached_sources(
        index=decoder_weight_waterfill_index,
        campaign_rows=rows,
    )
    archive_section_telemetry_unattached_sources = _archive_section_telemetry_unattached_sources(
        index=archive_section_telemetry_index,
        campaign_rows=rows,
    )
    return {
        "schema": SCHEMA,
        "baseline_to_beat": "pr95_public_control_arm_plus_frontier_exact_axes",
        "objective": (
            "Make HiNeRV and SNeRV full-stack PR95-or-better carriers: "
            "modelsize-byte constrained, real SegNet/PoseNet scorer-bound, "
            "QAT/coder pressured, MLX-first, NumPy-portable, receiver-proven, "
            "CPU replay gated, and exact-auth only after local wins."
        ),
        "top_priority_families": ["hi_nerv", "snerv"],
        "optimizer_kinds": list(optimizers),
        "optimizer_control_policy": {
            "schema": OPTIMIZER_CONTROL_SCHEMA,
            "backend": "mixed_mlx_optimizers_and_pact_pr95_partition_adapter",
            "native_mlx_on_apple_silicon": True,
            "apple_specific_algorithm_claim": False,
            "applies_to": [
                "hi_nerv_shared_mlx_scoreaware_runner_rows",
                "snerv_shared_mlx_scoreaware_long_training_rows",
            ],
            "does_not_apply_to": [],
            "optimizer_kinds": list(optimizers),
            "native_mlx_optimizer_kinds": [
                kind
                for kind in optimizers
                if kind in SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS and kind != "pact_muon_adamw"
            ],
            "pact_partitioned_optimizer_kinds": [kind for kind in optimizers if kind == "pact_muon_adamw"],
            "selected_plan_only_optimizer_kinds": [],
            "available_plan_only_optimizer_kinds": [],
            "selected_timing_smoke_optimizer_kinds": [
                kind for kind in optimizers if _is_timing_smoke_optimizer_kind(kind)
            ],
            "available_timing_smoke_optimizer_kinds": list(TIMING_SMOKE_OPTIMIZER_KINDS),
            "first_pass_optimizer_kinds": sorted(FIRST_PASS_OPTIMIZER_KINDS),
            "default_optimizer_kind": "pact_muon_adamw",
            "default_optimizer_backend": "tac.local_acceleration.pr95_hnerv_mlx",
            "borrowed_from_pr95": ("Muon-vs-AdamW parameter partition and Newton-Schulz Muon update helper"),
            "original_pact_contest_adaptation": (
                "default score-aware NeRV optimizer control with false-authority "
                "MLX telemetry, byte/coder pressure, and per-run optimizer "
                "control metadata"
            ),
            "notes": (
                "pact_muon_adamw is Pact's PR95-derived partitioned default; "
                "Muon, Lion, AdamW, Adamax, and the other native controls are "
                "direct MLX optimizer baselines on Apple silicon, not "
                "Apple-invented optimizer algorithms."
            ),
        },
        "epochs": int(epochs),
        "batch_pairs": int(batch_pairs),
        "learning_rate": float(learning_rate),
        "output_root": Path(output_root).as_posix(),
        "joint_recon_weight_artifacts": list(joint_recon_weight_artifacts.values()),
        "joint_recon_weight_artifact_count": len(joint_recon_weight_artifacts),
        "candidate_feedback_source_count": len(candidate_feedback_sources),
        "modelsize_byte_cap_feedback_paths": list(byte_cap_feedback_paths),
        "modelsize_byte_cap_feedback_path_count": len(byte_cap_feedback_paths),
        "hinerv_distortion_birth_evidence_source_count": len(
            hinerv_distortion_birth_evidence_sources
        ),
        "snerv_bounded_proof_only": bool(snerv_bounded_proof_only),
        "snerv_bounded_proof_epochs": int(snerv_bounded_proof_epochs),
        "snerv_bounded_proof_pair_count": int(snerv_bounded_proof_pair_count),
        "candidate_feedback_row_count": _unique_index_row_count(candidate_feedback_index),
        "decoder_weight_waterfill_source_count": len(decoder_weight_waterfill_sources),
        "archive_section_telemetry_source_count": len(archive_section_telemetry_sources),
        "archive_section_telemetry_row_count": _unique_index_row_count(archive_section_telemetry_index),
        "action_effect_source_count": len(action_effect_sources),
        "snerv_lf_payload_recode_source_count": len(snerv_lf_payload_recode_sources),
        "snerv_lf_payload_byte_report_source_count": len(snerv_lf_payload_byte_report_sources),
        "snerv_snar_header_grammar_profile_source_count": len(snerv_snar_header_grammar_profile_sources),
        "snerv_snar_header_minimization_report_source_count": len(snerv_snar_header_minimization_report_sources),
        "decoder_weight_waterfill_row_count": _unique_index_row_count(decoder_weight_waterfill_index),
        "decoder_weight_waterfill_unattached_source_count": len(decoder_weight_waterfill_unattached_sources),
        "decoder_weight_waterfill_unattached_sources": (decoder_weight_waterfill_unattached_sources),
        "archive_section_telemetry_unattached_source_count": len(archive_section_telemetry_unattached_sources),
        "archive_section_telemetry_unattached_sources": archive_section_telemetry_unattached_sources,
        "action_effect_planning_bundle": action_effect_planning_bundle,
        "action_effect_planning_bundle_schema": action_effect_planning_bundle[
            "schema"
        ],
        "action_effect_row_count": action_effect_planning_bundle[
            "effect_count"
        ],
        "action_effect_commutator_measurement_queue_count": (
            action_effect_planning_bundle["commutator_ledger"][
                "needs_measurement_count"
            ]
        ),
        "action_effect_inline_measured_interaction_count": (
            action_effect_planning_bundle["inline_measured_interaction_count"]
        ),
        "action_effect_selector_planning_consumed_by_queue": (
            experiment_queue.get("action_effect_planning_bundle")
            == action_effect_planning_bundle
        ),
        "snerv_lf_over_ceiling_reroute_queue": snerv_lf_over_ceiling_reroute_queue,
        "snerv_lf_over_ceiling_reroute_queue_schema": snerv_lf_over_ceiling_reroute_queue["schema"],
        "snerv_lf_over_ceiling_reroute_queue_row_count": snerv_lf_over_ceiling_reroute_queue["queue_row_count"],
        "snerv_lf_hf_replacement_queue": snerv_lf_hf_replacement_queue,
        "snerv_lf_hf_replacement_queue_schema": snerv_lf_hf_replacement_queue["schema"],
        "snerv_lf_hf_replacement_queue_row_count": snerv_lf_hf_replacement_queue["queue_row_count"],
        "source_parity_contract": source_parity_contract,
        "snerv_official_source_audit_attached": isinstance(snerv_official_source_audit, Mapping),
        "snerv_official_source_forward_artifact_count": len(
            snerv_official_source_forward_artifacts
        ),
        "snerv_official_source_forward_evidence": (
            dict(snerv_source_forward_evidence)
            if isinstance(snerv_source_forward_evidence, Mapping)
            else None
        ),
        "source_parity_required_for_long_training_ready": bool(
            source_parity_contract.get("required_for_long_training_ready")
        ),
        "source_parity_blockers": list(source_parity_contract.get("blockers") or ()),
        "source_parity_nonblocking_gaps": list(source_parity_contract.get("nonblocking_gaps") or ()),
        "upstream_evaluate_priority_contract": upstream_evaluate_priority_contract,
        "upstream_evaluate_contract_consumed_by_rows": all(
            isinstance(row.get("upstream_evaluate_score_binding"), Mapping) for row in rows
        ),
        "tilde_oss_leverage_policy": tilde_oss_leverage_policy,
        "tilde_oss_policy_consumed_by_rows": all(
            isinstance(row.get("tilde_oss_leverage_binding"), Mapping) for row in rows
        ),
        "pr95_baseline_identity_binding": pr95_baseline_binding,
        "pr95_baseline_identity_consumed_by_rows": all(
            isinstance(row.get("pr95_baseline_identity_binding"), Mapping) for row in rows
        ),
        "pr95_distortion_source_inventory": pr95_distortion_source_inventory,
        "pr95_distortion_source_ready": bool(pr95_distortion_source_inventory.get("source_ready")),
        "pr95_distortion_practices_consumed_by_rows": all(
            isinstance(row.get("pr95_distortion_practices_guard"), Mapping) for row in rows
        ),
        "pr95_distortion_practices_blockers": _dedupe(
            [
                blocker
                for row in rows
                for blocker in (
                    row.get("pr95_distortion_practices_guard", {}).get(
                        "blockers",
                        [],
                    )
                    if isinstance(
                        row.get("pr95_distortion_practices_guard"),
                        Mapping,
                    )
                    else []
                )
            ]
        ),
        "snerv_scorer_tether_smoke_gate": snerv_scorer_tether_smoke_gate,
        "snerv_scorer_tether_smoke_report_attached": bool(snerv_scorer_tether_smoke_gate.get("attached")),
        "campaign_rows": rows,
        "campaign_row_count": len(rows),
        "experiment_queue": experiment_queue,
        "experiment_queue_schema": EXPERIMENT_QUEUE_SCHEMA,
        "experiment_queue_id": experiment_queue["queue_id"],
        "planner_row_queue_artifact_path": planner_queue_artifact,
        "experiment_queue_experiment_count": len(experiment_queue["experiments"]),
        "launchable_local_row_count": sum(
            1
            for row in rows
            if row["experiment_queue_entry"].get("blocked") is not True
            and row["experiment_queue_entry"].get("status") in {"queued", "runnable"}
        ),
        "blocked_row_count": sum(1 for row in rows if row["blockers"]),
        "family_counts": _family_counts(rows),
        "decoder_weight_waterfill_attached_row_count": sum(
            1
            for row in rows
            if isinstance(row.get("decoder_weight_waterfill_plan"), Mapping)
            and row["decoder_weight_waterfill_plan"].get("attached") is True
        ),
        "archive_section_telemetry_attached_row_count": sum(
            1
            for row in rows
            if isinstance(row.get("archive_section_telemetry"), Mapping)
            and row["archive_section_telemetry"].get("attached") is True
        ),
        "promotion_policy": {
            "schema": "nerv_long_training_campaign_promotion_policy.v1",
            "mlx_role": "fast acquisition and prefilter only",
            "local_cpu_role": "full local replay gate for MLX-filtered winners",
            "exact_cpu_role": "first auth axis for true local winners",
            "exact_cuda_role": "second auth axis only after exact CPU clears",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "blockers": _dedupe(
            [
                "campaign_plan_is_not_execution",
                "exact_cpu_cuda_not_launched_by_campaign_plan",
                *[blocker for row in rows for blocker in row.get("blockers", []) if _plan_level_blocker(blocker)],
            ]
        ),
        **FALSE_AUTHORITY,
    }


def render_nerv_long_training_campaign_plan_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing campaign summary."""

    lines = [
        "# NeRV Long-Training Campaign Plan",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Rows: `{report.get('campaign_row_count')}`",
        f"Launchable local rows: `{report.get('launchable_local_row_count')}`",
        f"Blocked rows: `{report.get('blocked_row_count')}`",
        f"Score claim: `{report.get('score_claim')}`",
        f"Ready for exact dispatch: `{report.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Authority Bindings",
        "",
    ]
    upstream_contract = report.get("upstream_evaluate_priority_contract")
    if isinstance(upstream_contract, Mapping):
        crux = upstream_contract.get("crux")
        crux = crux if isinstance(crux, Mapping) else {}
        lines.extend(
            [
                f"- upstream_evaluate: `{upstream_contract.get('source')}`",
                f"  baseline_to_beat: `{upstream_contract.get('baseline_to_beat')}`",
                f"  canonical_rate_denominator_bytes: `{crux.get('canonical_rate_denominator_bytes')}`",
                f"  pose_marginal_formula: `{crux.get('pose_marginal_formula')}`",
            ]
        )
    tilde_policy = report.get("tilde_oss_leverage_policy")
    if isinstance(tilde_policy, Mapping):
        parallax = tilde_policy.get("parallax")
        parallax = parallax if isinstance(parallax, Mapping) else {}
        wall_attention = tilde_policy.get("wall_attention")
        wall_attention = wall_attention if isinstance(wall_attention, Mapping) else {}
        lines.extend(
            [
                f"- tilde_oss_policy: `{tilde_policy.get('schema')}`",
                f"  parallax_official_tilde_surface: `{parallax.get('official_tilde_surface')}`",
                f"  parallax_direct_runtime_import_allowed: `{parallax.get('direct_runtime_import_allowed')}`",
                "  wall_attention_direct_kernel_import_allowed: "
                f"`{wall_attention.get('direct_kernel_import_allowed')}`",
            ]
        )
    pr95_binding = report.get("pr95_baseline_identity_binding")
    if isinstance(pr95_binding, Mapping):
        selected_archive = pr95_binding.get("selected_archive")
        selected_archive = selected_archive if isinstance(selected_archive, Mapping) else {}
        local_work_order = pr95_binding.get("local_cpu_mlx_work_order")
        local_work_order = local_work_order if isinstance(local_work_order, Mapping) else {}
        modal_policy = pr95_binding.get("modal_dispatch_policy")
        modal_policy = modal_policy if isinstance(modal_policy, Mapping) else {}
        paired_work_order = pr95_binding.get("paired_exact_eval_work_order")
        paired_work_order = paired_work_order if isinstance(paired_work_order, Mapping) else {}
        lines.extend(
            [
                f"- pr95_baseline_identity_attached: `{pr95_binding.get('attached')}`",
                f"  baseline_id: `{pr95_binding.get('baseline_id')}`",
                f"  selected_archive_sha256: `{selected_archive.get('sha256')}`",
                f"  selected_archive_bytes: `{selected_archive.get('bytes')}`",
                f"  local_cpu_mlx_ready: `{local_work_order.get('ready')}`",
                f"  local_cpu_axis: `{local_work_order.get('local_cpu_axis_tag')}`",
                f"  mlx_axis: `{local_work_order.get('mlx_axis_tag')}`",
                f"  modal_dispatch_allowed: `{modal_policy.get('modal_dispatch_allowed')}`",
                f"  paired_exact_eval_ready: `{paired_work_order.get('ready')}`",
                f"  exact_axis_blockers: `{', '.join(str(b) for b in pr95_binding.get('blockers') or ())}`",
            ]
        )
    lines.extend(["", "## Rows", ""])
    for row in report.get("campaign_rows") or ():
        if not isinstance(row, Mapping):
            continue
        lines.extend(
            [
                f"- `{row.get('row_id')}`",
                f"  family: `{row.get('family')}`",
                f"  launchable_mlx: `{row.get('local_mlx_launch_command_ready')}`",
                f"  optimizer: `{row.get('optimizer_kind')}`",
                f"  blockers: `{len(row.get('blockers') or [])}`",
            ]
        )
        split = row.get("snerv_official_runtime_authority_split")
        if isinstance(split, Mapping):
            lines.extend(
                [
                    f"  snerv_runtime_authority: `{split.get('launch_semantics')}`",
                    f"  snerv_receiver_training_evidence: `{split.get('receiver_bound_training_evidence_usable')}`",
                    f"  snerv_full_source_forward_authority: `{split.get('full_source_forward_authority_proven')}`",
                ]
            )
    lines.extend(["", "## Blockers", ""])
    blockers = list(report.get("blockers") or ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _hinerv_campaign_row(
    *,
    candidate: Mapping[str, Any],
    optimizer_kind: str,
    epochs: int,
    batch_pairs: int,
    learning_rate: float,
    output_root: Path,
    joint_recon_weight_artifacts: Mapping[int, Mapping[str, Any]] | None = None,
    candidate_feedback_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    hinerv_distortion_birth_evidence_sources: Sequence[Mapping[str, Any]] = (),
    decoder_weight_waterfill_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    archive_section_telemetry_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    source_parity_contract: Mapping[str, Any] | None = None,
    upstream_evaluate_priority_contract: Mapping[str, Any] | None = None,
    tilde_oss_leverage_policy: Mapping[str, Any] | None = None,
    pr95_baseline_identity_binding: Mapping[str, Any] | None = None,
    pr95_distortion_source_inventory: Mapping[str, Any] | None = None,
    planner_row_queue_artifact_path: str | None = None,
    modelsize_byte_cap_feedback_paths: Sequence[str] = (),
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "hinerv_candidate")
    runner_candidate_id = "auto" if modelsize_byte_cap_feedback_paths else candidate_id
    runner_candidate_label = (
        _auto_bytecap_candidate_label(candidate_id) if modelsize_byte_cap_feedback_paths else candidate_id
    )
    quant_bits = min(8, decoder_codec_nominal_bits(str(candidate.get("decoder_codec"))))
    num_pairs = int(candidate.get("num_pairs") or 600)
    joint_recon_weight = dict((joint_recon_weight_artifacts or {}).get(num_pairs) or {})
    feedback = _candidate_feedback_for(
        candidate=candidate,
        family="hi_nerv",
        index=candidate_feedback_index,
    )
    prioritized_pair_indices = _feedback_prioritized_pair_indices(feedback)
    feedback_evidence_blockers = _candidate_feedback_evidence_blockers(
        feedback,
        family="hi_nerv",
    )
    family_training_telemetry_context = _family_training_telemetry_context_for(
        candidate=candidate,
        family="hi_nerv",
        index=candidate_feedback_index,
    )
    decoder_weight_waterfill = _decoder_weight_waterfill_for(
        candidate=candidate,
        family="hi_nerv",
        index=decoder_weight_waterfill_index,
    )
    archive_section_telemetry = _archive_section_telemetry_for(
        candidate=candidate,
        family="hi_nerv",
        index=archive_section_telemetry_index,
    )
    modelsize_byte_cap_preflight = _modelsize_byte_cap_preflight(
        candidate=candidate,
        family="hi_nerv",
        feedback_paths=modelsize_byte_cap_feedback_paths,
    )
    modelsize_byte_cap_blockers = list(modelsize_byte_cap_preflight.get("blockers") or [])
    launch_feedback_adjustment = _hinerv_feedback_launch_adjustment(
        feedback=feedback,
        learning_rate=float(learning_rate),
    )
    source_faithfulness_controls = _hinerv_source_faithfulness_controls(
        candidate=candidate,
        feedback=feedback,
    )
    official_control_blockers = _hinerv_official_control_blockers(candidate)
    source_parity = _source_parity_family_report(
        family="hi_nerv",
        source_parity_contract=source_parity_contract,
    )
    upstream_evaluate_binding = _row_upstream_evaluate_binding(
        family="hi_nerv",
        contract=upstream_evaluate_priority_contract
        if isinstance(upstream_evaluate_priority_contract, Mapping)
        else _upstream_evaluate_priority_contract(),
    )
    tilde_oss_binding = _row_tilde_oss_binding(
        family="hi_nerv",
        policy=tilde_oss_leverage_policy
        if isinstance(tilde_oss_leverage_policy, Mapping)
        else _tilde_oss_leverage_policy(),
    )
    pr95_baseline_binding = (
        dict(pr95_baseline_identity_binding)
        if isinstance(pr95_baseline_identity_binding, Mapping)
        else _pr95_baseline_identity_binding(None)
    )
    optimizer_launch_blockers = _optimizer_launch_blockers(optimizer_kind)
    effective_learning_rate = float(launch_feedback_adjustment.get("learning_rate") or learning_rate)
    effective_segnet_distillation_weight = float(launch_feedback_adjustment.get("segnet_distillation_weight") or 1.0)
    effective_pose_distillation_weight = float(launch_feedback_adjustment.get("pose_distillation_weight") or 1.0)
    distortion_birth_gate = _hinerv_distortion_birth_before_rate_pressure_gate(
        candidate=candidate,
        evidence_sources=hinerv_distortion_birth_evidence_sources,
    )
    rate_pressure_allowed = bool(distortion_birth_gate.get("passed"))
    coder_qat_control = {
        **_coder_qat_control(quant_bits=int(quant_bits)),
        "enabled": rate_pressure_allowed,
        "blocked_until_distortion_birth_gate_passes": (
            not rate_pressure_allowed
        ),
        "distortion_birth_before_rate_pressure_gate_passed": (
            rate_pressure_allowed
        ),
    }
    output_dir_basename = _campaign_output_basename(
        row_id=f"hi_nerv::{candidate_id}::{optimizer_kind}",
        launch_feedback_adjustment=launch_feedback_adjustment,
    )
    curriculum = build_hinerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=int(epochs),
        num_pairs=num_pairs,
        segnet_distillation_weight=effective_segnet_distillation_weight,
        pose_distillation_weight=effective_pose_distillation_weight,
        coder_aware_qat=rate_pressure_allowed,
        coder_qat_quant_bits=int(quant_bits),
        recon_pixel_weight_attached=True,
        eval_roundtrip_ste_attached=True,
        differentiable_pose_preprocess_attached=True,
        ema_archive_selection_attached=True,
        scorer_input_distribution_guard_attached=True,
        receiver_proof_attached=bool(feedback.get("receiver_proof_attached")),
        full_video_local_prefilter_attached=bool(feedback.get("full_video_local_prefilter_attached")),
        local_cpu_replay_gate_attached=bool(feedback.get("local_cpu_replay_gate_attached")),
        measured_archive_bytes=feedback.get("measured_archive_bytes"),
        measured_num_pairs=feedback.get("measured_num_pairs"),
        archive_minus_nominal_bytes=feedback.get("archive_minus_nominal_bytes"),
        archive_to_nominal_ratio=feedback.get("archive_to_nominal_ratio"),
        calibrated_archive_overrun_bytes=feedback.get("calibrated_archive_overrun_bytes"),
        required_nominal_payload_bytes_max=feedback.get("required_nominal_payload_bytes_max"),
        hard_byte_ceiling_measurement_bypass_enabled=feedback.get("hard_byte_ceiling_measurement_bypass_enabled"),
        hard_byte_ceiling_checked_after_export=feedback.get("hard_byte_ceiling_checked_after_export"),
    )
    row_id = f"hi_nerv::{runner_candidate_label}::{optimizer_kind}"
    optimizer_policy = _hinerv_optimizer_policy_for_kind(optimizer_kind)
    command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "runtime",
        "--extra",
        "mlx",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "hi_nerv",
        "--planner-row-id",
        row_id,
        "--num-pairs",
        str(num_pairs),
        "--epochs",
        str(int(epochs)),
        "--batch-pairs",
        str(int(batch_pairs)),
        "--learning-rate",
        _float_token(effective_learning_rate),
        "--distillation-device",
        "gpu",
        "--modelsize-candidate-id",
        runner_candidate_id,
        "--hard-byte-ceiling",
        str(int(candidate.get("hard_byte_ceiling") or 0)),
        "--segnet-distillation-weight",
        _float_token(effective_segnet_distillation_weight),
        "--pose-distillation-weight",
        _float_token(effective_pose_distillation_weight),
        "--pose-direct-live-distillation-weight",
        _float_token(DEFAULT_HINERV_POSE_DIRECT_LIVE_DISTILLATION_WEIGHT),
        "--segnet-distillation-objective",
        DEFAULT_HINERV_SEGNET_DIRECT_LIVE_OBJECTIVE,
        "--segnet-direct-live-distillation-weight",
        _float_token(DEFAULT_HINERV_SEGNET_DIRECT_LIVE_DISTILLATION_WEIGHT),
        "--segnet-direct-live-class-histogram-weight",
        _float_token(DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_HISTOGRAM_WEIGHT),
        "--segnet-direct-live-class-balanced-hinge-weight",
        _float_token(DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_HINGE_WEIGHT),
        "--segnet-direct-live-class-balanced-ce-weight",
        _float_token(DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_CE_WEIGHT),
        "--segnet-direct-live-class-balanced-squared-hinge-weight",
        _float_token(
            DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_SQUARED_HINGE_WEIGHT
        ),
        "--segnet-direct-live-class-region-recon-weight",
        _float_token(DEFAULT_HINERV_SEGNET_DIRECT_LIVE_CLASS_REGION_RECON_WEIGHT),
        "--segnet-direct-live-rare-class-logit-weight",
        _float_token(DEFAULT_HINERV_SEGNET_DIRECT_LIVE_RARE_CLASS_LOGIT_WEIGHT),
        "--segnet-direct-live-target-mass-floor-weight",
        _float_token(
            DEFAULT_HINERV_SEGNET_DIRECT_LIVE_TARGET_MASS_FLOOR_WEIGHT
        ),
        "--segnet-direct-live-target-min-ratio-floor-weight",
        _float_token(
            DEFAULT_HINERV_SEGNET_DIRECT_LIVE_TARGET_MIN_RATIO_FLOOR_WEIGHT
        ),
        "--optimizer-kind",
        str(optimizer_kind),
        "--hi-nerv-optimizer-policy",
        optimizer_policy,
        "--mlx-prefilter-scorer-device",
        "gpu",
        "--mlx-prefilter-scorer-batch-pairs",
        "1",
        "--mlx-prefilter-progress-every",
        "10",
        "--telemetry-flush-interval-epochs",
        str(DEFAULT_HINERV_TELEMETRY_FLUSH_INTERVAL_EPOCHS),
        "--scorer-input-distribution-guard-weight",
        _float_token(DEFAULT_HINERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT),
        "--scorer-input-contrast-floor-weight",
        _float_token(DEFAULT_HINERV_SCORER_INPUT_CONTRAST_FLOOR_WEIGHT),
        "--scorer-input-contrast-floor-segnet-min-std-ratio",
        _float_token(DEFAULT_HINERV_SCORER_INPUT_CONTRAST_FLOOR_SEGNET_MIN_STD_RATIO),
        "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
        _float_token(DEFAULT_HINERV_SCORER_INPUT_CONTRAST_FLOOR_POSENET_YUV6_MIN_STD_RATIO),
        "--scorer-input-shape-tether-weight",
        _float_token(DEFAULT_HINERV_SCORER_INPUT_SHAPE_TETHER_WEIGHT),
        "--posenet-yuv6-geometry-tether-weight",
        _float_token(DEFAULT_HINERV_POSENET_YUV6_GEOMETRY_TETHER_WEIGHT),
        "--posenet-temporal-signal-floor-weight",
        _float_token(DEFAULT_HINERV_POSENET_TEMPORAL_SIGNAL_FLOOR_WEIGHT),
        "--scorer-space-step-guard-min-post-segnet-target-class-coverage-fraction",
        _float_token(
            DEFAULT_HINERV_SCORER_STEP_GUARD_TARGET_CLASS_COVERAGE_FRACTION
        ),
        "--scorer-space-step-guard-min-post-segnet-target-class-min-ratio",
        _float_token(DEFAULT_HINERV_SCORER_STEP_GUARD_TARGET_CLASS_MIN_RATIO),
        "--scorer-space-step-guard-max-post-segnet-target-class-ratio-drop",
        _float_token(DEFAULT_HINERV_SCORER_STEP_GUARD_TARGET_CLASS_MAX_RATIO_DROP),
        "--run-post-export-materializers",
        "--output-dir",
        (output_root / output_dir_basename).as_posix(),
    ]
    if rate_pressure_allowed:
        command.extend(
            [
                "--coder-aware-qat",
                "--coder-qat-quant-bits",
                str(int(quant_bits)),
                *_coder_qat_command_args(quant_bits=int(quant_bits)),
            ]
        )
    if planner_row_queue_artifact_path:
        command.extend(["--planner-row-queue-artifact", planner_row_queue_artifact_path])
    for path in modelsize_byte_cap_feedback_paths:
        command.extend(["--modelsize-byte-cap-feedback-json", str(path)])
    if prioritized_pair_indices:
        command.extend(
            [
                "--prioritized-pair-indices",
                _int_csv(prioritized_pair_indices),
            ]
        )
    if joint_recon_weight:
        command.extend(
            [
                "--recon-pixel-weight-path",
                str(joint_recon_weight["weight_path"]),
            ]
        )
    decoder_weight_waterfill_runner_admitted = (
        _decoder_weight_waterfill_runner_admitted(decoder_weight_waterfill) if decoder_weight_waterfill else False
    )
    if decoder_weight_waterfill_runner_admitted:
        command.extend(
            [
                "--decoder-weight-waterfill-plan-json",
                str(decoder_weight_waterfill["path"]),
            ]
        )
    archive_section_telemetry_metadata = (
        _archive_section_telemetry_row_metadata(archive_section_telemetry)
        if archive_section_telemetry
        else _archive_section_telemetry_missing_metadata()
    )
    if archive_section_telemetry_metadata.get("runner_admitted") is True:
        command.extend(
            [
                "--archive-section-telemetry-json",
                str(archive_section_telemetry_metadata["path"]),
            ]
        )
    archive_section_telemetry_runner_admitted = archive_section_telemetry_metadata.get("runner_admitted") is True
    archive_section_telemetry_gate_ready = (
        not archive_section_telemetry_metadata.get("attached") or archive_section_telemetry_runner_admitted
    )
    if launch_feedback_adjustment.get("pose_protected_pathway_applied") is True:
        command.extend(
            [
                "--pose-distillation-loss",
                str(launch_feedback_adjustment["pose_distillation_loss"]),
                "--pose-distillation-huber-delta",
                _float_token(float(launch_feedback_adjustment["pose_distillation_huber_delta"])),
            ]
        )
    pr95_telemetry_contract = build_pr95_evaluate_scorer_domain_telemetry_contract("hi_nerv")
    pr95_axis_trace_contract = build_pr95_distortion_axis_trace_contract("hi_nerv")
    pr95_axis_trace_measurements = _axis_trace_measurements_from_sources(
        candidate,
        feedback,
    )
    pr95_pose_marginal_contract = build_pr95_posenet_marginal_telemetry_contract("hi_nerv")
    pr95_actuator_contract = build_pr95_scorer_atom_actuator_contract("hi_nerv")
    explicit_pr95_actuator_execution_evidence = candidate.get(
        "pr95_scorer_atom_actuator_execution_evidence"
    )
    derived_pr95_actuator_execution_evidence = (
        _hinerv_pr95_actuator_execution_evidence_from_feedback(feedback)
    )
    pr95_actuator_execution_evidence = (
        explicit_pr95_actuator_execution_evidence
        if isinstance(explicit_pr95_actuator_execution_evidence, Mapping)
        and explicit_pr95_actuator_execution_evidence
        else derived_pr95_actuator_execution_evidence
    )
    pr95_distortion_guard = build_pr95_distortion_practices_row_guard(
        {
            "id": row_id,
            "family": "hi_nerv",
            "command_argv": command,
            "hard_byte_ceiling": int(candidate.get("hard_byte_ceiling") or 0),
            "upstream_evaluate_score_binding": upstream_evaluate_binding,
            "pr95_evaluate_scorer_domain_telemetry_contract": pr95_telemetry_contract,
            "pr95_distortion_axis_trace_contract": pr95_axis_trace_contract,
            "pr95_distortion_axis_trace_measurements": pr95_axis_trace_measurements,
            "pr95_posenet_marginal_telemetry_contract": pr95_pose_marginal_contract,
            "pr95_scorer_atom_actuator_contract": pr95_actuator_contract,
            "pr95_scorer_atom_actuator_execution_evidence": (
                pr95_actuator_execution_evidence
            ),
            "hinerv_distortion_birth_before_rate_pressure_gate": (
                distortion_birth_gate
            ),
            "coder_qat_control": coder_qat_control,
            "curriculum_plan": curriculum,
            "pr95_staged_curriculum": bool((curriculum.get("pr95_stage_plan") or {}).get("enabled")),
            "eval_roundtrip_ste_attached": bool(
                (curriculum.get("scorer_pressure") or {}).get("eval_roundtrip_ste_attached")
            ),
        },
        repo_root=_repo_root(),
        source_inventory=pr95_distortion_source_inventory,
    )
    pr95_distortion_blockers = list(pr95_distortion_guard.get("blockers") or [])
    candidate_authority_blockers = list(candidate.get("_candidate_authority_blockers") or [])
    blockers = [
        ("" if joint_recon_weight else "requires_verified_joint_p18_p19_recon_pixel_weight_artifact"),
        *list(distortion_birth_gate.get("blockers") or []),
        ("" if decoder_weight_waterfill else "hinerv_decoder_weight_waterfill_plan_missing"),
        (
            ""
            if not decoder_weight_waterfill or decoder_weight_waterfill_runner_admitted
            else "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted"
        ),
        (
            ""
            if not archive_section_telemetry_metadata.get("attached") or archive_section_telemetry_runner_admitted
            else "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted"
        ),
        "requires_full_video_mlx_prefilter_before_local_cpu_replay_unlock",
        "requires_local_cpu_replay_win_before_exact_cpu_auth",
        *candidate_authority_blockers,
        *official_control_blockers,
        *optimizer_launch_blockers,
        *modelsize_byte_cap_blockers,
        *pr95_distortion_blockers,
        *list(source_parity["required_blockers"]),
        *list(curriculum.get("blockers") or []),
        *feedback_evidence_blockers,
    ]
    if feedback.get("pose_instability_detected") is True and not (
        launch_feedback_adjustment.get("applied") or launch_feedback_adjustment.get("pose_protected_pathway_applied")
    ):
        blockers.append("hinerv_pose_instability_feedback_unapplied")
    if feedback.get("seg_stagnation_detected") is True and not launch_feedback_adjustment.get("segnet_weight_applied"):
        blockers.append("hinerv_segnet_stagnation_feedback_unapplied")
    if feedback.get("pose_tail_burst_detected") is True and not prioritized_pair_indices:
        blockers.append("hinerv_pose_tail_burst_requires_prioritized_pair_indices")
    if launch_feedback_adjustment.get(
        "repeated_low_lr_pose_instability"
    ) is True and not launch_feedback_adjustment.get("pose_protected_pathway_applied"):
        blockers.append("hinerv_repeated_low_lr_pose_instability_requires_pose_protected_pathway")
    if candidate.get("nominal_under_ceiling") is not True:
        blockers.append("hinerv_candidate_nominal_over_byte_ceiling")
    blockers = _dedupe(blockers)
    launch_blockers = _experiment_launch_blockers(blockers)
    prelaunch_gate = dict(curriculum.get("long_campaign_prelaunch_gate") or {})
    launch_ready = bool(
        prelaunch_gate.get("launch_allowed")
        and archive_section_telemetry_gate_ready
        and not launch_blockers
        and not candidate_authority_blockers
        and not official_control_blockers
        and not optimizer_launch_blockers
        and not modelsize_byte_cap_blockers
        and not pr95_distortion_blockers
        and not source_parity["required_blockers"]
    )
    if candidate_authority_blockers:
        implementation_status = "selected_candidate_authority_flags_block_launch"
    elif official_control_blockers:
        implementation_status = "hinerv_official_controls_required_for_launch"
    elif optimizer_launch_blockers:
        implementation_status = "optimizer_timing_smoke_required_before_campaign_launch"
    elif not distortion_birth_gate.get("passed"):
        implementation_status = "hinerv_distortion_birth_before_rate_pressure_blocked"
    elif pr95_distortion_blockers:
        implementation_status = "pr95_distortion_practices_required_for_launch"
    elif source_parity["required_blockers"]:
        implementation_status = "source_parity_required_gap_blocks_launch"
    elif decoder_weight_waterfill and not decoder_weight_waterfill_runner_admitted:
        implementation_status = "decoder_weight_waterfill_plan_advisory_only_blocks_launch"
    elif not archive_section_telemetry_gate_ready:
        implementation_status = "archive_section_telemetry_advisory_only_blocks_launch"
    elif launch_ready:
        implementation_status = (
            "shared_mlx_scoreaware_runner_launchable_without_optional_waterfill"
            if not decoder_weight_waterfill
            else "shared_mlx_scoreaware_runner_launchable"
        )
    else:
        implementation_status = "shared_mlx_scoreaware_runner_waiting_for_hard_gate"
    return _row(
        row_id=row_id,
        family="hi_nerv",
        priority=_optimizer_priority(optimizer_kind),
        candidate=candidate,
        curriculum_plan=curriculum,
        command_argv=command,
        local_mlx_launch_command_ready=launch_ready,
        implementation_status=implementation_status,
        blockers=blockers,
        extra={
            "optimizer_kind": str(optimizer_kind),
            "budget_candidate_id": candidate_id,
            "runner_modelsize_candidate_id": runner_candidate_id,
            "modelsize_candidate_selection_mode": (
                "calibrated_auto_from_modelsize_byte_cap_feedback"
                if modelsize_byte_cap_feedback_paths
                else "explicit_budget_candidate"
            ),
            "modelsize_byte_cap_feedback_paths": list(modelsize_byte_cap_feedback_paths),
            "modelsize_byte_cap_preflight": modelsize_byte_cap_preflight,
            "optimizer_control": _optimizer_control(optimizer_kind),
            "upstream_evaluate_score_binding": upstream_evaluate_binding,
            "tilde_oss_leverage_binding": tilde_oss_binding,
            "pr95_baseline_identity_binding": pr95_baseline_binding,
            "pr95_evaluate_scorer_domain_telemetry_contract": (pr95_telemetry_contract),
            "pr95_distortion_axis_trace_contract": pr95_axis_trace_contract,
            "pr95_distortion_axis_trace_measurements": pr95_axis_trace_measurements,
            "pr95_posenet_marginal_telemetry_contract": pr95_pose_marginal_contract,
            "pr95_scorer_atom_actuator_contract": pr95_actuator_contract,
            "pr95_scorer_atom_actuator_execution_evidence": (
                pr95_actuator_execution_evidence
            ),
            "pr95_distortion_practices_guard": pr95_distortion_guard,
            "optimizer_policy": _hinerv_optimizer_policy_control(
                optimizer_kind=optimizer_kind,
                optimizer_policy=optimizer_policy,
            ),
            "quant_bits": int(quant_bits),
            "hinerv_distortion_birth_before_rate_pressure_gate": (
                distortion_birth_gate
            ),
            "rate_pressure_controls_enabled": rate_pressure_allowed,
            "coder_qat_control": coder_qat_control,
            "joint_recon_pixel_weight_artifact": joint_recon_weight or None,
            "decoder_weight_waterfill_plan": (
                _decoder_weight_waterfill_row_metadata(decoder_weight_waterfill)
                if decoder_weight_waterfill
                else {
                    "schema": ("nerv_long_training_decoder_weight_waterfill_attachment.v1"),
                    "attached": False,
                    "reason": "no_matching_decoder_weight_waterfill_plan",
                    **FALSE_AUTHORITY,
                }
            ),
            "archive_section_telemetry": archive_section_telemetry_metadata,
            "feedback_launch_adjustment": launch_feedback_adjustment,
            "candidate_feedback": feedback or None,
            "prioritized_pair_training": _prioritized_pair_training_plan(prioritized_pair_indices),
            "candidate_feedback_evidence_blockers": feedback_evidence_blockers,
            "family_training_telemetry_context": family_training_telemetry_context or None,
            "source_faithfulness_controls": source_faithfulness_controls,
            "source_parity": source_parity,
            "output_dir_basename": output_dir_basename,
            "output_dir_reuse_policy": (
                "fresh_feedback_mutation_path"
                if launch_feedback_adjustment.get("applied")
                else "stable_candidate_optimizer_path"
            ),
        },
    )


def _snerv_campaign_row(
    *,
    candidate: Mapping[str, Any],
    epochs: int,
    batch_pairs: int,
    learning_rate: float,
    optimizer_kind: str,
    output_root: Path,
    candidate_feedback_index: (Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None) = None,
    bounded_proof_only: bool = False,
    bounded_proof_epochs: int = 3,
    source_parity_contract: Mapping[str, Any] | None = None,
    upstream_evaluate_priority_contract: Mapping[str, Any] | None = None,
    tilde_oss_leverage_policy: Mapping[str, Any] | None = None,
    pr95_baseline_identity_binding: Mapping[str, Any] | None = None,
    pr95_distortion_source_inventory: Mapping[str, Any] | None = None,
    snerv_scorer_tether_smoke_gate: Mapping[str, Any] | None = None,
    snerv_source_forward_evidence: Mapping[str, Any] | None = None,
    snerv_official_replacement_authority_gate: Mapping[str, Any] | None = None,
    snerv_long_run_launch_gate_verdict: Mapping[str, Any] | None = None,
    planner_row_queue_artifact_path: str | None = None,
    modelsize_byte_cap_feedback_paths: Sequence[str] = (),
    snerv_lf_payload_recode_sources: Sequence[Mapping[str, Any]] = (),
    bounded_proof_pair_count: int = DEFAULT_SNERV_BOUNDED_PROOF_PAIR_COUNT,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "snerv_candidate")
    runner_candidate_id = "auto" if modelsize_byte_cap_feedback_paths else candidate_id
    runner_candidate_label = (
        _auto_bytecap_candidate_label(candidate_id) if modelsize_byte_cap_feedback_paths else candidate_id
    )
    source_control_blockers = _snerv_source_bound_control_blockers(candidate)
    source_parity = _source_parity_family_report(
        family="snerv",
        source_parity_contract=source_parity_contract,
    )
    source_parity = _snerv_source_parity_with_source_forward_evidence(
        source_parity,
        snerv_source_forward_evidence,
    )
    official_runtime_authority_split = _snerv_official_runtime_authority_split(source_parity)
    official_runtime_authority_split = (
        _snerv_runtime_authority_split_with_source_forward_evidence(
            official_runtime_authority_split,
            snerv_source_forward_evidence,
        )
    )
    upstream_evaluate_binding = _row_upstream_evaluate_binding(
        family="snerv",
        contract=upstream_evaluate_priority_contract
        if isinstance(upstream_evaluate_priority_contract, Mapping)
        else _upstream_evaluate_priority_contract(),
    )
    tilde_oss_binding = _row_tilde_oss_binding(
        family="snerv",
        policy=tilde_oss_leverage_policy
        if isinstance(tilde_oss_leverage_policy, Mapping)
        else _tilde_oss_leverage_policy(),
    )
    pr95_baseline_binding = (
        dict(pr95_baseline_identity_binding)
        if isinstance(pr95_baseline_identity_binding, Mapping)
        else _pr95_baseline_identity_binding(None)
    )
    scorer_tether_smoke_gate = (
        dict(snerv_scorer_tether_smoke_gate)
        if isinstance(snerv_scorer_tether_smoke_gate, Mapping)
        else _snerv_scorer_tether_smoke_gate(None)
    )
    feedback = _candidate_feedback_for(
        candidate=candidate,
        family="snerv",
        index=candidate_feedback_index,
    )
    modelsize_byte_cap_preflight = _modelsize_byte_cap_preflight(
        candidate=candidate,
        family="snerv",
        feedback_paths=modelsize_byte_cap_feedback_paths,
    )
    modelsize_byte_cap_blockers = list(modelsize_byte_cap_preflight.get("blockers") or [])
    feedback = _snerv_feedback_with_modelsize_byte_cap_evidence(
        feedback=feedback,
        candidate=candidate,
        modelsize_byte_cap_preflight=modelsize_byte_cap_preflight,
    )
    feedback = _snerv_feedback_with_source_forward_evidence(
        feedback,
        snerv_source_forward_evidence,
    )
    scorer_tether_smoke_gate = (
        _snerv_scorer_tether_smoke_gate_with_candidate_feedback(
            scorer_tether_smoke_gate,
            feedback,
        )
    )
    scorer_input_guard_feedback_passed = (
        _snerv_candidate_feedback_scorer_input_guard_passed(feedback)
    )
    prioritized_pair_indices = _feedback_prioritized_pair_indices(feedback)
    raw_feedback_evidence_blockers = _candidate_feedback_evidence_blockers(
        feedback,
        family="snerv",
    )
    (
        feedback_evidence_blockers,
        smoke_suppressed_feedback_blockers,
    ) = _snerv_feedback_blockers_after_tether_smoke(
        raw_feedback_evidence_blockers,
        scorer_tether_smoke_gate,
    )
    renderer_nondegenerate_gate = _snerv_renderer_nondegenerate_gate(
        feedback=feedback,
        bounded_proof_only=bool(bounded_proof_only),
    )
    pre_long_run_evidence_gate = _snerv_pre_long_run_evidence_gate(
        feedback=feedback,
        bounded_proof_only=bool(bounded_proof_only),
    )
    execution_epochs = min(int(epochs), max(1, int(bounded_proof_epochs))) if bounded_proof_only else int(epochs)
    candidate_num_pairs = max(1, int(candidate.get("num_pairs") or 600))
    execution_num_pairs = (
        min(candidate_num_pairs, max(1, int(bounded_proof_pair_count)))
        if bounded_proof_only
        else candidate_num_pairs
    )
    quant_bits = min(
        8,
        snerv_decoder_codec_nominal_bits(str(candidate.get("decoder_payload_codec"))),
    )
    optimizer_control = _optimizer_control(optimizer_kind)
    pr95_curriculum_bound = not bool(bounded_proof_only) and int(execution_epochs) >= 8
    joint_teacher_bound = not bool(bounded_proof_only)
    coder_qat_bound = not bool(bounded_proof_only)
    eval_roundtrip_bound = not bool(bounded_proof_only)
    curriculum = build_snerv_candidate_curriculum_plan(
        candidate=candidate,
        requested_epochs=int(execution_epochs),
        num_pairs=int(execution_num_pairs),
        step_map_coder_mode="waterfill",
        native_mlx_train_export_attached=True,
        native_mlx_long_training_bound=not bool(bounded_proof_only),
        native_mlx_receiver_proof_passed=bool(feedback.get("native_mlx_receiver_proof_passed")),
        native_mlx_full600_campaign_ready=bool(feedback.get("native_mlx_full600_campaign_ready")),
        native_mlx_scorer_loop_qat_attached=True,
        native_mlx_scorer_loop_qat_receiver_contract_satisfied=bool(
            feedback.get("native_mlx_scorer_loop_qat_receiver_contract_satisfied")
        ),
        native_mlx_scorer_loop_qat_ready_for_pose_guard_gate=bool(
            feedback.get("native_mlx_scorer_loop_qat_ready_for_pose_guard_gate")
        ),
        native_mlx_scorer_loop_qat_accepted_improvement=bool(
            feedback.get("native_mlx_scorer_loop_qat_accepted_improvement")
        ),
        native_mlx_scorer_loop_qat_best_materialized=bool(feedback.get("native_mlx_scorer_loop_qat_best_materialized")),
        native_mlx_real_segnet_teacher_bound=joint_teacher_bound,
        native_mlx_real_posenet_teacher_bound=joint_teacher_bound,
        native_mlx_pr95_curriculum_bound=pr95_curriculum_bound,
        native_mlx_eval_roundtrip_ste_bound=eval_roundtrip_bound,
        native_mlx_differentiable_pose_preprocess_bound=joint_teacher_bound,
        native_mlx_coder_qat_bound=coder_qat_bound,
        native_mlx_muon_adamw_partition_bound=(str(optimizer_kind) == "pact_muon_adamw"),
        native_mlx_artifact_evidence=_snerv_native_artifact_evidence_from_feedback(feedback),
        receiver_proof_attached=bool(feedback.get("receiver_proof_attached")),
        full_video_local_prefilter_attached=bool(feedback.get("full_video_local_prefilter_attached")),
        local_cpu_replay_gate_attached=bool(feedback.get("local_cpu_replay_gate_attached")),
        measured_packet_bytes=feedback.get("measured_payload_bytes"),
        measured_archive_bytes=feedback.get("measured_archive_bytes"),
        measured_num_pairs=feedback.get("measured_num_pairs"),
        archive_minus_nominal_bytes=feedback.get("archive_minus_nominal_bytes"),
        archive_to_nominal_ratio=feedback.get("archive_to_nominal_ratio"),
        calibrated_archive_overrun_bytes=feedback.get("calibrated_archive_overrun_bytes"),
        required_nominal_payload_bytes_max=feedback.get("required_nominal_payload_bytes_max"),
        hard_byte_ceiling_measurement_bypass_enabled=feedback.get("hard_byte_ceiling_measurement_bypass_enabled"),
        hard_byte_ceiling_checked_after_export=feedback.get("hard_byte_ceiling_checked_after_export"),
    )
    curriculum = _snerv_curriculum_with_source_forward_evidence(
        curriculum,
        snerv_source_forward_evidence,
    )
    row_id = f"snerv::{runner_candidate_label}::native_rate_aware_training"
    native_mlx_decoder_train_steps = max(
        0,
        int(
            candidate.get(
                "snerv_native_mlx_decoder_train_steps",
                candidate.get("native_mlx_decoder_train_steps", 0),
            )
            or 0
        ),
    )
    native_mlx_decoder_train_lr = float(
        candidate.get(
            "snerv_native_mlx_decoder_train_lr",
            candidate.get("native_mlx_decoder_train_lr", 1.0e-5),
        )
        or 1.0e-5
    )
    native_mlx_decoder_train_ridge = float(
        candidate.get(
            "snerv_native_mlx_decoder_train_ridge",
            candidate.get("native_mlx_decoder_train_ridge", 1.0e-6),
        )
        or 1.0e-6
    )
    official_trained_checkpoint_state_dict_path = (
        _snerv_official_trained_checkpoint_state_dict_path_from_feedback(feedback)
    )
    command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "runtime",
        "--extra",
        "mlx",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "snerv",
        "--planner-row-id",
        row_id,
        "--num-pairs",
        str(int(execution_num_pairs)),
        "--epochs",
        str(int(execution_epochs)),
        "--snerv-score-aware-long-training-epochs",
        str(int(execution_epochs)),
        "--snerv-score-aware-long-training-lr",
        _float_token(float(learning_rate)),
        "--snerv-score-aware-long-training-batch-pairs",
        str(int(batch_pairs)),
        "--snerv-score-aware-long-training-optimizer",
        str(optimizer_kind),
        "--snerv-score-aware-long-training-eval-roundtrip-ste",
        "--snerv-score-aware-long-training-pr95-faithful-curriculum",
        "--modelsize-candidate-id",
        runner_candidate_id,
        "--hard-byte-ceiling",
        str(int(candidate.get("hard_byte_ceiling") or 0)),
        "--distillation-device",
        "gpu",
        "--segnet-distillation-weight",
        "1.0",
        "--pose-distillation-weight",
        "1.0",
        "--pose-direct-live-distillation-weight",
        _float_token(DEFAULT_SNERV_POSE_DIRECT_LIVE_DISTILLATION_WEIGHT),
        "--segnet-distillation-objective",
        DEFAULT_SNERV_SEGNET_DIRECT_LIVE_OBJECTIVE,
        "--segnet-direct-live-distillation-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_DISTILLATION_WEIGHT),
        "--segnet-direct-live-class-histogram-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_HISTOGRAM_WEIGHT),
        "--segnet-direct-live-class-balanced-hinge-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_HINGE_WEIGHT),
        "--segnet-direct-live-class-balanced-ce-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_CE_WEIGHT),
        "--segnet-direct-live-class-balanced-squared-hinge-weight",
        _float_token(
            DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_BALANCED_SQUARED_HINGE_WEIGHT
        ),
        "--segnet-direct-live-class-region-recon-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_CLASS_REGION_RECON_WEIGHT),
        "--segnet-direct-live-rare-class-logit-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_RARE_CLASS_LOGIT_WEIGHT),
        "--segnet-direct-live-target-mass-floor-weight",
        _float_token(DEFAULT_SNERV_SEGNET_DIRECT_LIVE_TARGET_MASS_FLOOR_WEIGHT),
        "--segnet-direct-live-target-min-ratio-floor-weight",
        _float_token(
            DEFAULT_SNERV_SEGNET_DIRECT_LIVE_TARGET_MIN_RATIO_FLOOR_WEIGHT
        ),
        "--scorer-input-distribution-guard-weight",
        _float_token(DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT),
        "--scorer-input-contrast-floor-weight",
        _float_token(DEFAULT_SNERV_SCORER_INPUT_CONTRAST_FLOOR_WEIGHT),
        "--scorer-input-contrast-floor-segnet-min-std-ratio",
        _float_token(DEFAULT_SNERV_SCORER_INPUT_CONTRAST_FLOOR_SEGNET_MIN_STD_RATIO),
        "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
        _float_token(
            DEFAULT_SNERV_SCORER_INPUT_CONTRAST_FLOOR_POSENET_YUV6_MIN_STD_RATIO
        ),
        "--scorer-input-shape-tether-weight",
        _float_token(DEFAULT_SNERV_SCORER_INPUT_SHAPE_TETHER_WEIGHT),
        "--posenet-yuv6-geometry-tether-weight",
        _float_token(DEFAULT_SNERV_POSENET_YUV6_GEOMETRY_TETHER_WEIGHT),
        "--posenet-temporal-signal-floor-weight",
        _float_token(DEFAULT_SNERV_POSENET_TEMPORAL_SIGNAL_FLOOR_WEIGHT),
        "--scorer-space-step-guard-min-post-segnet-target-class-coverage-fraction",
        _float_token(DEFAULT_SNERV_SCORER_STEP_GUARD_TARGET_CLASS_COVERAGE_FRACTION),
        "--scorer-space-step-guard-min-post-segnet-target-class-min-ratio",
        _float_token(DEFAULT_SNERV_SCORER_STEP_GUARD_TARGET_CLASS_MIN_RATIO),
        "--scorer-space-step-guard-max-post-segnet-target-class-ratio-drop",
        _float_token(DEFAULT_SNERV_SCORER_STEP_GUARD_TARGET_CLASS_MAX_RATIO_DROP),
        "--coder-aware-qat",
        "--coder-qat-quant-bits",
        str(int(quant_bits)),
        *_coder_qat_command_args(quant_bits=int(quant_bits)),
        "--snerv-scorer-loop-qat",
        "--snerv-scorer-loop-search-mode",
        "learned_random_subspace",
        "--snerv-fc-dim",
        str(int(candidate.get("fc_dim") or 0)),
        "--snerv-emb-size",
        str(int(candidate.get("emb_size") or 0)),
        "--snerv-patch-radius",
        str(int(candidate.get("patch_radius") or 0)),
        "--snerv-mfu-scales",
        _int_csv(candidate.get("mfu_scales") or (1, 2, 4)),
        "--snerv-hfr-gain",
        _float_token(float(candidate.get("hfr_gain") or 0.0)),
        "--snerv-temporal-context",
        str(int(candidate.get("temporal_context") or 0)),
        "--snerv-temporal-mode",
        str(candidate.get("temporal_mode") or "delta"),
        "--snerv-official-skip-high-mode",
        str(candidate.get("official_skip_high_mode", "full")),
        "--snerv-native-mlx-receiver-proof-timeout",
        str(int(candidate.get("snerv_native_mlx_receiver_proof_timeout", 1800))),
        "--snerv-native-mlx-decoder-train-steps",
        str(int(native_mlx_decoder_train_steps)),
        "--snerv-native-mlx-decoder-train-lr",
        _float_token(float(native_mlx_decoder_train_lr)),
        "--snerv-native-mlx-decoder-train-ridge",
        _float_token(float(native_mlx_decoder_train_ridge)),
        "--mlx-prefilter-scorer-device",
        "cpu",
        "--mlx-prefilter-scorer-batch-pairs",
        "1",
        "--mlx-prefilter-progress-every",
        "25",
        "--output-dir",
        (output_root / _safe_path_token(row_id)).as_posix(),
    ]
    if planner_row_queue_artifact_path:
        command.extend(["--planner-row-queue-artifact", planner_row_queue_artifact_path])
    for path in modelsize_byte_cap_feedback_paths:
        command.extend(["--modelsize-byte-cap-feedback-json", str(path)])
    if official_trained_checkpoint_state_dict_path is not None:
        command.extend(
            [
                "--snerv-official-trained-checkpoint-state-dict-path",
                official_trained_checkpoint_state_dict_path.as_posix(),
            ]
        )
    snerv_model_size_adapter = str(candidate.get("snerv_model_size_adapter") or "").strip()
    if snerv_model_size_adapter:
        command.extend(["--snerv-model-size-adapter", snerv_model_size_adapter])
    if snerv_model_size_adapter == SNERV_SPECTRA_PRESERVING_ADAPTER:
        command.append("--snerv-spectra-preserving-adapter")
    prioritized_pair_training = _snerv_prioritized_pair_training_plan(prioritized_pair_indices)
    lf_recode_admission_plan = _snerv_lf_payload_recode_admission_for_candidate(
        sources=snerv_lf_payload_recode_sources,
        candidate=candidate,
        candidate_id=candidate_id,
        full_video_coverage=bool(feedback.get("full_video_local_prefilter_attached")),
    )
    lf_recode_selected_mode = _snerv_lf_recode_selected_mode(lf_recode_admission_plan)
    if lf_recode_selected_mode:
        command.extend(
            [
                "--snerv-scorer-loop-lf-payload-codec",
                lf_recode_selected_mode,
            ]
        )
    pr95_telemetry_contract = build_pr95_evaluate_scorer_domain_telemetry_contract("snerv")
    pr95_axis_trace_contract = build_pr95_distortion_axis_trace_contract("snerv")
    pr95_axis_trace_measurements = _axis_trace_measurements_from_sources(
        candidate,
        feedback,
        snerv_source_forward_evidence,
    )
    pr95_pose_marginal_contract = build_pr95_posenet_marginal_telemetry_contract("snerv")
    pr95_actuator_contract = build_pr95_scorer_atom_actuator_contract("snerv")
    explicit_pr95_actuator_execution_evidence = candidate.get(
        "pr95_scorer_atom_actuator_execution_evidence"
    )
    derived_pr95_actuator_execution_evidence = (
        _snerv_pr95_actuator_execution_evidence_from_source_forward(
            snerv_source_forward_evidence,
            official_replacement_authority_gate=(
                snerv_official_replacement_authority_gate
            ),
        )
    )
    pr95_actuator_execution_evidence = (
        explicit_pr95_actuator_execution_evidence
        if isinstance(explicit_pr95_actuator_execution_evidence, Mapping)
        and explicit_pr95_actuator_execution_evidence
        else derived_pr95_actuator_execution_evidence
    )
    pr95_distortion_guard = build_pr95_distortion_practices_row_guard(
        {
            "id": row_id,
            "family": "snerv",
            "command_argv": command,
            "hard_byte_ceiling": int(candidate.get("hard_byte_ceiling") or 0),
            "upstream_evaluate_score_binding": upstream_evaluate_binding,
            "pr95_evaluate_scorer_domain_telemetry_contract": pr95_telemetry_contract,
            "pr95_distortion_axis_trace_contract": pr95_axis_trace_contract,
            "pr95_distortion_axis_trace_measurements": pr95_axis_trace_measurements,
            "pr95_posenet_marginal_telemetry_contract": pr95_pose_marginal_contract,
            "pr95_scorer_atom_actuator_contract": pr95_actuator_contract,
            "pr95_scorer_atom_actuator_execution_evidence": (
                pr95_actuator_execution_evidence
            ),
            "curriculum_plan": curriculum,
            "pr95_faithful_curriculum_enabled": bool(pr95_curriculum_bound),
            "eval_roundtrip_ste_attached": bool(eval_roundtrip_bound),
        },
        repo_root=_repo_root(),
        source_inventory=pr95_distortion_source_inventory,
    )
    pr95_distortion_blockers = list(pr95_distortion_guard.get("blockers") or [])
    curriculum_blockers = [str(v) for v in curriculum.get("blockers") or () if v]
    rate_plausible_for_long_training = _snerv_rate_plausible_for_long_training(candidate)
    hard_byte_ceiling_satisfied_for_long_training = _snerv_hard_byte_ceiling_satisfied_for_long_training(
        candidate,
        lf_recode_admission_plan=lf_recode_admission_plan,
    )
    snerv_launch_gate_status = _snerv_long_run_launch_gate_status(
        snerv_long_run_launch_gate_verdict,
        bounded_proof_only=bool(bounded_proof_only),
    )
    blockers = _dedupe(
        [
            ("snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only" if bounded_proof_only else ""),
            *list(prioritized_pair_training.get("blockers") or []),
            (
                "snerv_nominal_payload_far_over_ceiling_refuse_long_training"
                if not bounded_proof_only and not rate_plausible_for_long_training
                else ""
            ),
            "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes"
            if candidate.get("nominal_under_ceiling") is not True
            else "",
            "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training"
            if not bounded_proof_only and not hard_byte_ceiling_satisfied_for_long_training
            else "",
            *list(candidate.get("_candidate_authority_blockers") or []),
            *source_control_blockers,
            *modelsize_byte_cap_blockers,
            *pr95_distortion_blockers,
            *list(scorer_tether_smoke_gate.get("blockers") or []),
            *list(renderer_nondegenerate_gate.get("blockers") or []),
            *list(pre_long_run_evidence_gate.get("blockers") or []),
            *list(source_parity["required_blockers"]),
            *list(snerv_launch_gate_status.get("blockers") or []),
            *curriculum_blockers,
            *feedback_evidence_blockers,
            *_snerv_lf_payload_recode_campaign_blockers(lf_recode_admission_plan),
        ]
    )
    blockers = _without_closed_source_forward_blockers(
        blockers,
        snerv_source_forward_evidence,
    )
    if scorer_input_guard_feedback_passed:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "snerv_scorer_input_distribution_guard_missing"
        ]
    source_controls_ready = (
        not source_control_blockers
        and not candidate.get("_candidate_authority_blockers")
        and not modelsize_byte_cap_blockers
        and not pr95_distortion_blockers
        and not source_parity["required_blockers"]
    )
    bounded_source_controls_ready = (
        not source_control_blockers
        and not candidate.get("_candidate_authority_blockers")
        and not modelsize_byte_cap_blockers
    )
    curriculum_ready = not curriculum_blockers
    scorer_tether_smoke_ready = not scorer_tether_smoke_gate.get("blockers")
    renderer_nondegenerate_ready = not renderer_nondegenerate_gate.get("blockers")
    pre_long_run_evidence_ready = not pre_long_run_evidence_gate.get("blockers")
    launch_gate_ready = bool(
        bounded_proof_only or snerv_launch_gate_status.get("approved")
    )
    prelaunch_proof_ready = bool(
        scorer_tether_smoke_ready
        and renderer_nondegenerate_ready
        and pre_long_run_evidence_ready
        and launch_gate_ready
    )
    bounded_proof_launch_ready = bool(
        bounded_proof_only
        and bounded_source_controls_ready
        and scorer_tether_smoke_ready
    )
    launch_ready = (
        bounded_proof_launch_ready
        if bounded_proof_only
        else bool(
            source_controls_ready
            and curriculum_ready
            and prelaunch_proof_ready
            and bool(
                rate_plausible_for_long_training
                and hard_byte_ceiling_satisfied_for_long_training
            )
        )
    )
    if bounded_proof_only and not scorer_tether_smoke_ready:
        implementation_status = "snerv_scorer_tether_smoke_gate_blocked"
    elif bounded_proof_launch_ready:
        implementation_status = "bounded_native_export_scorer_loop_stage_ready"
    else:
        implementation_status = (
            "source_bound_capacity_controls_incomplete"
            if not source_controls_ready
            else "pr95_distortion_practices_required_for_launch"
            if pr95_distortion_blockers
            else "snerv_scorer_tether_smoke_gate_blocked"
            if not scorer_tether_smoke_ready
            else "native_rate_aware_long_training_renderer_proof_blocked"
            if not renderer_nondegenerate_ready
            else "native_rate_aware_long_training_evidence_gate_blocked"
            if not pre_long_run_evidence_ready
            else "snerv_long_run_launch_gate_blocked"
            if not launch_gate_ready
            else "snerv_scoreaware_curriculum_blocked"
            if not curriculum_ready
            else (
                "native_rate_aware_long_training_queue_ready"
                if (
                    rate_plausible_for_long_training
                    and hard_byte_ceiling_satisfied_for_long_training
                )
                else "native_rate_aware_long_training_rate_blocked"
            )
        )
    return _row(
        row_id=row_id,
        family="snerv",
        priority=12,
        candidate=candidate,
        curriculum_plan=curriculum,
        command_argv=command,
        local_mlx_launch_command_ready=launch_ready,
        implementation_status=implementation_status,
        blockers=blockers,
        bounded_proof_launch=bool(bounded_proof_only),
        extra={
            "optimizer_kind": str(optimizer_kind),
            "budget_candidate_id": candidate_id,
            "runner_modelsize_candidate_id": runner_candidate_id,
            "modelsize_candidate_selection_mode": (
                "calibrated_auto_from_modelsize_byte_cap_feedback"
                if modelsize_byte_cap_feedback_paths
                else "explicit_budget_candidate"
            ),
            "modelsize_byte_cap_feedback_paths": list(modelsize_byte_cap_feedback_paths),
            "modelsize_byte_cap_preflight": modelsize_byte_cap_preflight,
            "optimizer_control": optimizer_control,
            "upstream_evaluate_score_binding": upstream_evaluate_binding,
            "tilde_oss_leverage_binding": tilde_oss_binding,
            "pr95_baseline_identity_binding": pr95_baseline_binding,
            "pr95_evaluate_scorer_domain_telemetry_contract": (pr95_telemetry_contract),
            "pr95_distortion_axis_trace_contract": pr95_axis_trace_contract,
            "pr95_distortion_axis_trace_measurements": pr95_axis_trace_measurements,
            "pr95_posenet_marginal_telemetry_contract": pr95_pose_marginal_contract,
            "pr95_scorer_atom_actuator_contract": pr95_actuator_contract,
            "pr95_scorer_atom_actuator_execution_evidence": (
                pr95_actuator_execution_evidence
            ),
            "pr95_distortion_practices_guard": pr95_distortion_guard,
            "snerv_scorer_tether_smoke_gate": scorer_tether_smoke_gate,
            "snerv_renderer_nondegenerate_gate": renderer_nondegenerate_gate,
            "snerv_pre_long_run_evidence_gate": pre_long_run_evidence_gate,
            "snerv_long_run_launch_gate": snerv_launch_gate_status,
            "quant_bits": int(quant_bits),
            "coder_qat_control": _coder_qat_control(quant_bits=int(quant_bits)),
            "planned_long_training_epochs": int(epochs),
            "execution_epochs": int(execution_epochs),
            "execution_num_pairs": int(execution_num_pairs),
            "current_command_is_bounded_proof_not_long_training": bool(bounded_proof_only),
            "snerv_bounded_proof_epochs": int(bounded_proof_epochs),
            "snerv_bounded_proof_pair_count": int(bounded_proof_pair_count),
            "source_bound_capacity_controls": _snerv_source_bound_controls(candidate),
            "source_bound_capacity_control_blockers": source_control_blockers,
            "source_parity": source_parity,
            "snerv_official_runtime_authority_split": (official_runtime_authority_split),
            "snerv_official_source_forward_evidence": (
                dict(snerv_source_forward_evidence)
                if _snerv_source_forward_evidence_active(snerv_source_forward_evidence)
                else None
            ),
            "snerv_official_trained_checkpoint_mapping": feedback.get(
                "snerv_official_trained_checkpoint_mapping_manifest"
            ),
            "snerv_official_trained_checkpoint_state_dict_path_from_feedback": (
                None
                if official_trained_checkpoint_state_dict_path is None
                else official_trained_checkpoint_state_dict_path.as_posix()
            ),
            "snerv_official_trained_checkpoint_state_dict_path_consumed_by_command": bool(
                official_trained_checkpoint_state_dict_path is not None
            ),
            "candidate_feedback": feedback or None,
            "candidate_feedback_evidence_blockers_before_tether_smoke": (raw_feedback_evidence_blockers),
            "candidate_feedback_evidence_blockers": feedback_evidence_blockers,
            "snerv_scorer_tether_smoke_suppressed_feedback_blockers": (smoke_suppressed_feedback_blockers),
            "prioritized_pair_training": prioritized_pair_training,
            "snerv_lf_payload_recode_admission_plan": lf_recode_admission_plan,
            "snerv_lf_payload_codec_from_admission_plan": lf_recode_selected_mode,
            "hard_byte_ceiling_satisfied_for_long_training": (hard_byte_ceiling_satisfied_for_long_training),
            "native_mlx_decoder_training_plan": {
                "schema": "snerv_native_mlx_decoder_training_plan.v1",
                "candidate_conditioned": True,
                "planned_steps": int(native_mlx_decoder_train_steps),
                "learning_rate": float(native_mlx_decoder_train_lr),
                "ridge": float(native_mlx_decoder_train_ridge),
                "backend": "mlx_metal_full_batch_gradient_descent",
                "consumed_by_command": True,
                **FALSE_AUTHORITY,
            },
            "score_aware_long_training_plan": {
                "schema": "snerv_score_aware_long_training_plan.v1",
                "epochs": int(execution_epochs),
                "batch_pairs": int(batch_pairs),
                "learning_rate": float(learning_rate),
                "optimizer_kind": str(optimizer_kind),
                "segnet_distillation_weight": 1.0,
                "pose_distillation_weight": 1.0,
                "coder_aware_qat_bound": coder_qat_bound,
                "pr95_faithful_curriculum_bound": pr95_curriculum_bound,
                "eval_roundtrip_ste_bound": eval_roundtrip_bound,
                "muon_adamw_partition_bound": (str(optimizer_kind) == "pact_muon_adamw"),
                "upstream_evaluate_score_binding": upstream_evaluate_binding,
                "tilde_oss_leverage_binding": tilde_oss_binding,
                "pr95_baseline_identity_binding": pr95_baseline_binding,
                "pr95_distortion_axis_trace_contract": pr95_axis_trace_contract,
                "pr95_posenet_marginal_telemetry_contract": pr95_pose_marginal_contract,
                "pr95_scorer_atom_actuator_contract": pr95_actuator_contract,
                "pr95_distortion_practices_guard": pr95_distortion_guard,
                "snerv_scorer_tether_smoke_gate": scorer_tether_smoke_gate,
                "snerv_renderer_nondegenerate_gate": renderer_nondegenerate_gate,
                "snerv_pre_long_run_evidence_gate": pre_long_run_evidence_gate,
                **FALSE_AUTHORITY,
            },
        },
    )


def _row(
    *,
    row_id: str,
    family: str,
    priority: int,
    candidate: Mapping[str, Any],
    curriculum_plan: Mapping[str, Any],
    command_argv: Sequence[str],
    local_mlx_launch_command_ready: bool,
    implementation_status: str,
    blockers: Sequence[str],
    extra: Mapping[str, Any],
    bounded_proof_launch: bool = False,
) -> dict[str, Any]:
    family_optimal_strategy = (
        curriculum_plan.get("family_optimal_strategy")
        if isinstance(curriculum_plan.get("family_optimal_strategy"), Mapping)
        else None
    )
    row_extra = dict(extra)
    if isinstance(family_optimal_strategy, Mapping):
        row_extra.setdefault("family_optimal_strategy", dict(family_optimal_strategy))
    score_gate = _score_lowering_gate(
        family=family,
        local_mlx_launch_command_ready=local_mlx_launch_command_ready,
        curriculum_plan=curriculum_plan,
        blockers=blockers,
        bounded_proof_launch=bool(bounded_proof_launch),
    )
    return {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "family": family,
        "priority": int(priority),
        "candidate_id": candidate.get("candidate_id"),
        "candidate": dict(candidate),
        "hard_byte_ceiling": int(candidate.get("hard_byte_ceiling") or 0),
        "candidate_nominal_total_payload_bytes": int(candidate.get("nominal_total_payload_bytes") or 0),
        "candidate_nominal_under_ceiling": bool(candidate.get("nominal_under_ceiling")),
        "local_mlx_launch_command_ready": bool(local_mlx_launch_command_ready),
        "implementation_status": str(implementation_status),
        "command_argv": list(command_argv),
        "experiment_queue_entry": _experiment_for_row(
            row_id=row_id,
            family=family,
            priority=priority,
            command_argv=command_argv,
            local_mlx_launch_command_ready=local_mlx_launch_command_ready,
            blockers=blockers,
            score_lowering_gate=score_gate,
            row_metadata=_experiment_row_metadata(row_extra),
            bounded_proof_launch=bool(bounded_proof_launch),
        ),
        "curriculum_plan": dict(curriculum_plan),
        "family_optimal_strategy": (
            dict(family_optimal_strategy)
            if isinstance(family_optimal_strategy, Mapping)
            else None
        ),
        "score_lowering_gate": score_gate,
        "local_mlx_executable": bool(score_gate["local_mlx_executable"]),
        "cpu_replay_ready": bool(score_gate["cpu_replay_ready"]),
        "exact_gate_ready": bool(score_gate["exact_gate_ready"]),
        "promotion_blockers": list(score_gate["promotion_blockers"]),
        "blockers": _dedupe([str(blocker) for blocker in blockers if blocker]),
        **row_extra,
        **FALSE_AUTHORITY,
    }


def _experiment_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
) -> dict[str, Any]:
    return {
        "schema": EXPERIMENT_QUEUE_SCHEMA,
        "queue_id": queue_id,
        "owner": "nerv_long_training_campaign_plan",
        "description": (
            "Queue-owned MLX-first HiNeRV/SNeRV long-training campaign. "
            "Rows are false-authority until receiver proof, local CPU replay, "
            "and exact CPU/CUDA gates pass."
        ),
        "experiments": [
            dict(row["experiment_queue_entry"])
            for row in rows
            if isinstance(row.get("experiment_queue_entry"), Mapping)
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _action_effect_planning_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_effect_sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    effects_by_key: dict[tuple[Any, ...], ActionEffect] = {}
    source_refs_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    receiver_closed_by_key: dict[tuple[Any, ...], bool] = {}
    blockers: list[str] = []

    def add_effect(
        effect: ActionEffect,
        *,
        row_id: str,
        family: str,
        source_schema: Any,
        source: str,
    ) -> None:
        receiver_closed = _action_effect_is_receiver_closed(effect)
        if not receiver_closed:
            blockers.append(
                f"action_effect_not_receiver_closed:{row_id}:{effect.action_id}:{effect.action_kind}"
            )
        key = _action_effect_identity_key(effect)
        effects_by_key.setdefault(key, effect)
        receiver_closed_by_key[key] = receiver_closed_by_key.get(key, False) or receiver_closed
        source_refs_by_key.setdefault(key, []).append(
            {
                "row_id": row_id,
                "family": family,
                "source_schema": source_schema,
                "source": source,
                "receiver_closed": receiver_closed,
            }
        )

    for row in rows:
        row_id = str(row.get("row_id") or "")
        family = str(row.get("family") or "")
        for source in _row_pair_local_action_effect_sources(row):
            try:
                effect = ActionEffect.from_pair_local_admission(source)
            except (TypeError, ValueError) as exc:
                blockers.append(
                    f"action_effect_pair_local_receipt_invalid:{row_id}:{type(exc).__name__}"
                )
                continue
            add_effect(
                effect,
                row_id=row_id,
                family=family,
                source_schema=source.get("schema"),
                source=str(source.get("source") or "campaign_row_pair_local"),
            )
    for index, source in enumerate(action_effect_sources):
        source_id = str(
            source.get("row_id")
            or source.get("source")
            or source.get("action_id")
            or f"direct_action_effect_source_{index}"
        )
        family = str(source.get("family") or "")
        try:
            effects = _action_effects_from_direct_source(source)
        except (TypeError, ValueError) as exc:
            blockers.append(
                f"action_effect_direct_source_invalid:{source_id}:{type(exc).__name__}"
            )
            continue
        for effect in effects:
            add_effect(
                effect,
                row_id=source_id,
                family=family or effect.family,
                source_schema=source.get("schema"),
                source=str(source.get("source") or "direct_action_effect_source"),
            )
    effects = list(effects_by_key.values())
    atlas_rows = [
        _action_effect_atlas_row(
            effect,
            source_refs_by_key.get(_action_effect_identity_key(effect), []),
            receiver_closed=receiver_closed_by_key.get(
                _action_effect_identity_key(effect),
                False,
            ),
        )
        for effect in effects
    ]
    atlas_rows.sort(
        key=lambda item: (
            float("-inf")
            if item.get("value_per_byte") is None
            else float(item["value_per_byte"]),
            -abs(float(item.get("delta_score_total") or 0.0)),
            str(item.get("action_id") or ""),
        ),
        reverse=True,
    )
    commutator_ledger = build_commutator_ledger(effects)
    inline_commutator_count = sum(
        1 for effect in effects if effect.interaction_or_commutator is not None
    )
    receiver_closed_count = sum(
        1 for effect in effects if receiver_closed_by_key.get(_action_effect_identity_key(effect), False)
    )
    selector_planning = {
        "schema": ACTION_EFFECT_SELECTOR_PLANNING_SCHEMA,
        "action_effect_schema": ACTION_EFFECT_V1_SCHEMA,
        "commutator_ledger_schema": commutator_ledger["schema"],
        "independent_delta_assumption_allowed": False,
        "receiver_closed_action_count": receiver_closed_count,
        "advisory_false_authority_action_count": len(effects) - receiver_closed_count,
        "inline_measured_interaction_count": inline_commutator_count,
        "measured_commutator_count": commutator_ledger["measured_commutator_count"],
        "needs_measurement_count": commutator_ledger["needs_measurement_count"],
        "macro_action_candidate_count": len(commutator_ledger["macro_action_candidates"]),
        "conflict_pair_count": len(commutator_ledger["conflict_pairs"]),
        "measurement_queue": list(commutator_ledger["measurement_queue"]),
        "macro_action_candidates": list(commutator_ledger["macro_action_candidates"]),
        "conflict_pairs": list(commutator_ledger["conflict_pairs"]),
        "policy": {
            "receiver_closed_actions_are_score_currency": True,
            "composition_must_be_measured_before_additive_planning": True,
            "measurement_queue_routes_unmeasured_pairs": True,
        },
        **FALSE_AUTHORITY,
    }
    return {
        "schema": ACTION_EFFECT_PLANNING_BUNDLE_SCHEMA,
        "action_effect_schema": ACTION_EFFECT_V1_SCHEMA,
        "action_atlas": {
            "schema": ACTION_EFFECT_ATLAS_SCHEMA,
            "rows": atlas_rows,
            "row_count": len(atlas_rows),
            "ranked_by": "value_per_byte_desc_then_abs_delta_score_total_desc",
            **FALSE_AUTHORITY,
        },
        "effect_count": len(effects),
        "receiver_closed_effect_count": receiver_closed_count,
        "advisory_false_authority_effect_count": len(effects) - receiver_closed_count,
        "inline_measured_interaction_count": inline_commutator_count,
        "effects": [effect.as_dict() for effect in effects],
        "commutator_ledger": commutator_ledger,
        "selector_planning": selector_planning,
        "blockers": _dedupe(blockers),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _action_effects_from_direct_source(source: Mapping[str, Any]) -> tuple[ActionEffect, ...]:
    schema = str(source.get("schema") or "")
    if schema == ACTION_EFFECT_V1_SCHEMA:
        return (ActionEffect.from_dict(source),)
    if schema == "hi_nerv_target_region_birth_four_arm_ablation.v1" or (
        isinstance(source.get("four_arm_ablation"), Mapping)
    ):
        return ActionEffect.from_hinerv_four_arm_ablation(source)
    if schema in {
        "hi_nerv_target_region_birth_receipt.v1",
        "hi_nerv_target_region_birth_four_arm.v1",
    } or isinstance(source.get("exact_nonrate"), Mapping):
        return (ActionEffect.from_hinerv_birth_receipt(source),)
    raise ValueError(f"unsupported ActionEffect source schema: {schema!r}")


def _row_pair_local_action_effect_sources(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for evidence_key in (
        "pr95_scorer_atom_actuator_execution_evidence",
        "candidate_feedback",
        "snerv_official_source_forward_evidence",
    ):
        evidence = row.get(evidence_key)
        if isinstance(evidence, Mapping):
            receipt = evidence.get("pair_local_distortion_servo_receipt")
            if isinstance(receipt, Mapping):
                source = dict(receipt)
                source.setdefault("source", evidence_key)
                source.setdefault("family", row.get("family"))
                sources.append(source)
            nested = evidence.get("pr95_scorer_atom_actuator_execution_evidence")
            if isinstance(nested, Mapping):
                nested_receipt = nested.get("pair_local_distortion_servo_receipt")
                if isinstance(nested_receipt, Mapping):
                    source = dict(nested_receipt)
                    source.setdefault("source", f"{evidence_key}.pr95_scorer_atom_actuator_execution_evidence")
                    source.setdefault("family", row.get("family"))
                    sources.append(source)
    return sources


def _action_effect_is_receiver_closed(effect: ActionEffect) -> bool:
    authority = str(effect.authority or "").lower()
    return bool(
        effect.parseback_survived is True
        or effect.inflate_survived is True
        or "parseback" in authority
        or "inflate" in authority
        or "inflated" in authority
    )


def _action_effect_identity_key(effect: ActionEffect) -> tuple[Any, ...]:
    return (
        effect.action_id,
        effect.family,
        effect.action_kind,
        effect.arm,
        effect.authority,
        effect.old_d_seg,
        effect.new_d_seg,
        effect.old_d_pose,
        effect.new_d_pose,
        effect.old_bytes,
        effect.new_bytes,
    )


def _action_effect_atlas_row(
    effect: ActionEffect,
    source_refs: Sequence[Mapping[str, Any]],
    *,
    receiver_closed: bool,
) -> dict[str, Any]:
    out = effect.as_dict()
    out.update(
        {
            "schema": "nerv_action_effect_atlas_row.v1",
            "action_effect_schema": ACTION_EFFECT_V1_SCHEMA,
            "source_refs": [dict(ref) for ref in source_refs],
            "source_row_ids": _dedupe(
                [str(ref.get("row_id") or "") for ref in source_refs if ref.get("row_id")]
            ),
            "receiver_closed": bool(receiver_closed),
            "score_currency": (
                "contest_score_units"
                if bool(receiver_closed)
                else "advisory_false_authority_score_units"
            ),
            **FALSE_AUTHORITY,
        }
    )
    return out


def _experiment_for_row(
    *,
    row_id: str,
    family: str,
    priority: int,
    command_argv: Sequence[str],
    local_mlx_launch_command_ready: bool,
    blockers: Sequence[str],
    score_lowering_gate: Mapping[str, Any],
    row_metadata: Mapping[str, Any] | None = None,
    bounded_proof_launch: bool = False,
) -> dict[str, Any]:
    output_dir = _row_output_dir(command_argv)
    output_json = (output_dir / "compact_renderer_mlx_spine_runner_report.json").as_posix()
    receiver_surface_trace_contract = _receiver_surface_trace_contract(
        family=str(family),
        output_dir=output_dir,
    )
    receiver_surface_trace_path = str(
        receiver_surface_trace_contract["trace_artifact_path"]
    )
    telemetry_artifacts = _row_observable_artifacts(
        family=str(family),
        output_dir=output_dir,
    )
    postconditions = [
        {
            "type": "json_equals",
            "path": output_json,
            "key": "schema",
            "equals": "compact_renderer_mlx_spine_runner.v1",
        },
        {
            "type": "json_equals",
            "path": output_json,
            "key": "score_claim",
            "equals": False,
        },
        {
            "type": "json_equals",
            "path": output_json,
            "key": "ready_for_exact_eval_dispatch",
            "equals": False,
        },
        {
            "type": "json_equals",
            "path": output_json,
            "key": "promotion_eligible",
            "equals": False,
        },
        {
            "type": "path_exists",
            "path": receiver_surface_trace_path,
        },
    ]
    if str(family) == "hi_nerv":
        postconditions.extend(
            [
                {
                    "type": "json_equals",
                    "path": output_json,
                    "key": "execute_family",
                    "equals": "hi_nerv",
                },
                {
                    "type": "json_equals",
                    "path": output_json,
                    "key": "training_executed",
                    "equals": True,
                },
            ]
        )
    elif str(family) == "snerv":
        postconditions.append(
            {
                "type": "json_equals",
                "path": output_json,
                "key": "execute_family",
                "equals": "snerv",
            }
        )
        bounded_blocker = "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only"
        if bounded_blocker in blockers:
            postconditions.append(
                {
                    "type": "json_array_contains",
                    "path": output_json,
                    "key": "blockers",
                    "contains": bounded_blocker,
                }
            )
    launch_blockers = _experiment_launch_blockers(
        blockers,
        bounded_proof_launch=bool(bounded_proof_launch),
    )
    runnable = bool(local_mlx_launch_command_ready) and not launch_blockers
    metadata = dict(row_metadata or {})
    source_parity = metadata.get("source_parity")
    source_controls = metadata.get("source_bound_capacity_controls")
    source_control_blockers = metadata.get("source_bound_capacity_control_blockers")
    upstream_evaluate_binding = metadata.get("upstream_evaluate_score_binding")
    tilde_oss_binding = metadata.get("tilde_oss_leverage_binding")
    pr95_baseline_binding = metadata.get("pr95_baseline_identity_binding")
    pr95_telemetry_contract = metadata.get("pr95_evaluate_scorer_domain_telemetry_contract")
    pr95_axis_trace_contract = metadata.get("pr95_distortion_axis_trace_contract")
    archive_parseback_selection_contract = _archive_parseback_selection_contract(
        family=str(family),
        axis_trace_contract=pr95_axis_trace_contract,
    )
    pr95_axis_trace_measurements = metadata.get(
        "pr95_distortion_axis_trace_measurements"
    )
    hinerv_distortion_birth_gate = metadata.get(
        "hinerv_distortion_birth_before_rate_pressure_gate"
    )
    pr95_pose_marginal_contract = metadata.get("pr95_posenet_marginal_telemetry_contract")
    pr95_actuator_contract = metadata.get("pr95_scorer_atom_actuator_contract")
    pr95_actuator_execution_evidence = metadata.get(
        "pr95_scorer_atom_actuator_execution_evidence"
    )
    pr95_distortion_guard = metadata.get("pr95_distortion_practices_guard")
    family_optimal_strategy = metadata.get("family_optimal_strategy")
    snerv_runtime_authority_split = metadata.get("snerv_official_runtime_authority_split")
    snerv_renderer_nondegenerate_gate = metadata.get("snerv_renderer_nondegenerate_gate")
    snerv_pre_long_run_evidence_gate = metadata.get("snerv_pre_long_run_evidence_gate")
    current_command_is_bounded_proof = bool(metadata.get("current_command_is_bounded_proof_not_long_training"))
    pre_long_handoff = _snerv_pre_long_run_evidence_handoff(
        family=str(family),
        row_id=row_id,
        command_argv=command_argv,
        metadata=metadata,
        blockers=blockers,
        postconditions=postconditions,
    )
    queue_steps: list[dict[str, Any]] = []
    if pre_long_handoff is not None:
        queue_steps.append(pre_long_handoff)
        metadata["snerv_pre_long_run_evidence_handoff"] = dict(pre_long_handoff)
    queue_steps.append(
        {
            "id": "run_mlx_first_campaign_row",
            "command": list(command_argv),
            "resources": {
                "kind": "local_mlx",
                "max_parallel_group": "local_mlx_training",
            },
            "postconditions": postconditions,
            "telemetry": {
                "artifact_paths": telemetry_artifacts,
                "include_postcondition_paths": True,
            },
            "on_postcondition_failure": "failed",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    launch_steps = list(queue_steps) if runnable else []
    return {
        "id": _safe_path_token(row_id),
        "family": str(family),
        "priority": int(priority),
        "status": "queued" if runnable else "disabled",
        "blocked": not runnable,
        "launch_authority_contract": {
            "schema": "nerv_long_training_queue_launch_authority_contract.v1",
            "queue_status_is_local_mlx_plan": bool(local_mlx_launch_command_ready),
            "queue_status_is_runnable_plan": runnable,
            "queue_launch_step_count": len(launch_steps),
            "queue_steps_retained_as_post_unblock_handoff": not runnable,
            "queue_launch_blockers": list(launch_blockers),
            "queue_status_is_receiver_proof": False,
            "queue_status_is_cpu_replay_proof": False,
            "queue_status_is_exact_eval_authority": False,
            "source_parity_contract_consumed": isinstance(source_parity, Mapping),
            "upstream_evaluate_score_contract_consumed": isinstance(
                upstream_evaluate_binding,
                Mapping,
            ),
            "tilde_oss_leverage_policy_consumed": isinstance(
                tilde_oss_binding,
                Mapping,
            ),
            "pr95_baseline_identity_consumed": isinstance(
                pr95_baseline_binding,
                Mapping,
            ),
            "pr95_evaluate_scorer_domain_telemetry_contract_consumed": isinstance(
                pr95_telemetry_contract,
                Mapping,
            ),
            "pr95_distortion_axis_trace_contract_consumed": isinstance(
                pr95_axis_trace_contract,
                Mapping,
            ),
            "receiver_surface_trace_contract_consumed": True,
            "archive_parseback_selection_contract_consumed": True,
            "pr95_posenet_marginal_telemetry_contract_consumed": isinstance(
                pr95_pose_marginal_contract,
                Mapping,
            ),
            "pr95_scorer_atom_actuator_contract_consumed": isinstance(
                pr95_actuator_contract,
                Mapping,
            ),
            "pr95_distortion_practices_consumed": isinstance(
                pr95_distortion_guard,
                Mapping,
            ),
            "family_optimal_strategy_consumed": isinstance(
                family_optimal_strategy,
                Mapping,
            ),
            "family_optimal_strategy": (
                family_optimal_strategy
                if isinstance(family_optimal_strategy, Mapping)
                else None
            ),
            "source_bound_capacity_controls_consumed": isinstance(
                source_controls,
                Mapping,
            ),
            "source_bound_capacity_control_blockers": list(source_control_blockers or ()),
            "current_command_is_bounded_proof_not_long_training": (current_command_is_bounded_proof),
            "receiver_proof_required": bool(score_lowering_gate.get("receiver_proof_required")),
            "cpu_replay_ready": bool(score_lowering_gate["cpu_replay_ready"]),
            "exact_gate_ready": bool(score_lowering_gate["exact_gate_ready"]),
            "source_parity": source_parity if isinstance(source_parity, Mapping) else None,
            "upstream_evaluate_score_binding": (
                upstream_evaluate_binding if isinstance(upstream_evaluate_binding, Mapping) else None
            ),
            "tilde_oss_leverage_binding": (tilde_oss_binding if isinstance(tilde_oss_binding, Mapping) else None),
            "pr95_baseline_identity_binding": (
                pr95_baseline_binding if isinstance(pr95_baseline_binding, Mapping) else None
            ),
            "pr95_evaluate_scorer_domain_telemetry_contract": (
                pr95_telemetry_contract if isinstance(pr95_telemetry_contract, Mapping) else None
            ),
            "pr95_distortion_axis_trace_contract": (
                pr95_axis_trace_contract
                if isinstance(pr95_axis_trace_contract, Mapping)
                else None
            ),
            "receiver_surface_trace_contract": receiver_surface_trace_contract,
            "archive_parseback_selection_contract": (
                archive_parseback_selection_contract
            ),
            "pr95_distortion_axis_trace_measurements": (
                [
                    dict(item)
                    for item in pr95_axis_trace_measurements
                    if isinstance(item, Mapping)
                ]
                if isinstance(pr95_axis_trace_measurements, Sequence)
                and not isinstance(pr95_axis_trace_measurements, (str, bytes))
                else []
            ),
            "hinerv_distortion_birth_before_rate_pressure_gate": (
                hinerv_distortion_birth_gate
                if isinstance(hinerv_distortion_birth_gate, Mapping)
                else None
            ),
            "pr95_posenet_marginal_telemetry_contract": (
                pr95_pose_marginal_contract
                if isinstance(pr95_pose_marginal_contract, Mapping)
                else None
            ),
            "pr95_scorer_atom_actuator_contract": (
                pr95_actuator_contract
                if isinstance(pr95_actuator_contract, Mapping)
                else None
            ),
            "pr95_scorer_atom_actuator_execution_evidence": (
                pr95_actuator_execution_evidence
                if isinstance(pr95_actuator_execution_evidence, Mapping)
                else None
            ),
            "pr95_distortion_practices_guard": (
                pr95_distortion_guard if isinstance(pr95_distortion_guard, Mapping) else None
            ),
            "snerv_official_runtime_authority_split": (
                snerv_runtime_authority_split if isinstance(snerv_runtime_authority_split, Mapping) else None
            ),
            "snerv_renderer_nondegenerate_gate": (
                snerv_renderer_nondegenerate_gate if isinstance(snerv_renderer_nondegenerate_gate, Mapping) else None
            ),
            "snerv_pre_long_run_evidence_gate": (
                snerv_pre_long_run_evidence_gate
                if isinstance(snerv_pre_long_run_evidence_gate, Mapping)
                else None
            ),
            "source_bound_capacity_controls": (source_controls if isinstance(source_controls, Mapping) else None),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "blockers": _dedupe([str(blocker) for blocker in blockers if blocker]),
        "metadata": {
            "schema": "nerv_long_training_campaign_experiment_metadata.v1",
            **metadata,
            **FALSE_AUTHORITY,
        },
        "score_lowering_gate": dict(score_lowering_gate),
        "cpu_replay_ready": bool(score_lowering_gate["cpu_replay_ready"]),
        "exact_gate_ready": bool(score_lowering_gate["exact_gate_ready"]),
        "steps": queue_steps,
        "launch_steps": launch_steps,
        "launch_step_count": len(launch_steps),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_pre_long_run_evidence_handoff(
    *,
    family: str,
    row_id: str,
    command_argv: Sequence[str],
    metadata: Mapping[str, Any],
    blockers: Sequence[str],
    postconditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Expose the concrete pre-long SNeRV full-video feedback handoff."""

    if _family_key(family) != "snerv":
        return None
    if bool(metadata.get("current_command_is_bounded_proof_not_long_training")):
        return None
    gate = metadata.get("snerv_pre_long_run_evidence_gate")
    if not isinstance(gate, Mapping):
        return None
    if gate.get("required") is not True or gate.get("passed") is True:
        return None
    output_dir = _row_output_dir(command_argv)
    export_json = (
        output_dir
        / "snerv_mlx_native_export"
        / "native_train_export"
        / "snerv_mlx_native_train_export.json"
    )
    prefilter_profile = (
        output_dir
        / "snerv_mlx_native_export"
        / "native_train_export"
        / "local_mlx_prefilter"
        / "local_mlx_prefilter_profile.json"
    )
    feedback_dir = output_dir / "full_video_mlx_feedback"
    candidate_id = _first_non_auto_candidate_id(
        metadata.get("budget_candidate_id"),
        metadata.get("runner_modelsize_candidate_id"),
        _command_flag_value(command_argv, "--modelsize-candidate-id"),
    )
    hard_byte_ceiling = _command_flag_value(command_argv, "--hard-byte-ceiling")
    harvest_command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "runtime",
        "--extra",
        "mlx",
        "python",
        "tools/harvest_nerv_full_video_mlx_feedback.py",
        "--mlx-response",
        prefilter_profile.as_posix(),
        "--archive-export-json",
        export_json.as_posix(),
        "--family",
        "snerv",
        "--output-dir",
        feedback_dir.as_posix(),
    ]
    if candidate_id:
        harvest_command.extend(["--candidate-id", candidate_id])
    if hard_byte_ceiling:
        harvest_command.extend(["--hard-byte-ceiling", hard_byte_ceiling])
    gate_blockers = _dedupe(
        [
            str(blocker)
            for blocker in (
                *(gate.get("blockers") or ()),
                *blockers,
            )
            if str(blocker).startswith("snerv_pre_long_run_")
            or str(blocker).startswith("snerv_mlx_native_")
        ]
    )
    return {
        "id": "materialize_snerv_full600_prefilter_feedback_before_long_run",
        "schema": "snerv_pre_long_run_evidence_handoff_step.v1",
        "status": "blocked_until_full600_native_export_and_mlx_prefilter_feedback",
        "blocked": True,
        "blockers": gate_blockers,
        "row_id": row_id,
        "candidate_id": candidate_id,
        "command": list(command_argv),
        "resources": {
            "kind": "local_mlx",
            "max_parallel_group": "local_mlx_training",
        },
        "postconditions": [dict(condition) for condition in postconditions],
        "telemetry": {
            "artifact_paths": [
                export_json.as_posix(),
                prefilter_profile.as_posix(),
                (feedback_dir / "nerv_full_video_mlx_scorer_feedback_row.json").as_posix(),
            ],
            "include_postcondition_paths": False,
        },
        "on_postcondition_failure": "failed",
        "full600_materialization_command": list(command_argv),
        "full600_materialization_command_role": (
            "provenance for the canonical row command; do not treat as runnable "
            "while this queue experiment is disabled"
        ),
        "expected_artifacts": {
            "archive_export_json": export_json.as_posix(),
            "full_video_mlx_response": prefilter_profile.as_posix(),
            "candidate_feedback_row": (
                feedback_dir / "nerv_full_video_mlx_scorer_feedback_row.json"
            ).as_posix(),
            "candidate_feedback_manifest": (
                feedback_dir / "nerv_full_video_mlx_scorer_feedback.json"
            ).as_posix(),
        },
        "harvest_command": harvest_command,
        "harvest_tool": "tools/harvest_nerv_full_video_mlx_feedback.py",
        "next_planner_input_flag": "--candidate-feedback-row",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _command_flag_value(command_argv: Sequence[str], flag: str) -> str | None:
    argv = [str(item) for item in command_argv]
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return ""
    return argv[index + 1]


def _first_non_auto_candidate_id(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() != "auto":
            return text
    return ""


def _experiment_launch_blockers(
    blockers: Sequence[str],
    *,
    bounded_proof_launch: bool = False,
) -> list[str]:
    """Return blockers that should prevent a row from being runnable."""

    exact_names = {
        "aurora_requires_local_timing_convergence_smoke",
        "full600_or_hardpair_distortion_replay_required",
        "hi_nerv_archive_in_loop_byte_oracle_missing",
        "hinerv_distortion_birth_before_rate_pressure_missing_or_blocked",
        "hinerv_candidate_curriculum_recon_pixel_weight_missing",
        "hinerv_archive_section_telemetry_advisory_only_not_runner_admitted",
        "hinerv_archive_section_telemetry_archive_not_under_hard_byte_ceiling",
        "hinerv_archive_section_telemetry_receiver_cache_quality_gate_not_passed",
        "hinerv_archive_section_telemetry_receiver_cache_quality_not_bound",
        "hinerv_archive_section_telemetry_receiver_cache_quality_report_path_missing",
        "hinerv_archive_section_telemetry_receiver_cache_quality_report_path_not_file",
        "hinerv_archive_section_telemetry_receiver_cache_quality_report_sha256_mismatch",
        "hinerv_archive_section_telemetry_receiver_proof_not_bound",
        "hinerv_decoder_weight_waterfill_plan_missing",
        "hinerv_decoder_weight_waterfill_plan_advisory_only_not_runner_admitted",
        "hinerv_local_scorer_input_health_gate_failed",
        "hinerv_local_scorer_input_profile_missing",
        "hinerv_receiver_proven_archive_over_hard_byte_ceiling",
        "hinerv_trained_archive_byte_oracle_feedback_missing",
        "partial_pair_byte_feedback_only",
        "requires_verified_joint_p18_p19_recon_pixel_weight_artifact",
        "small_pair_distortion_smoke_only_not_representative",
        "snerv_candidate_id_source_bound_controls_mismatch",
        "snerv_candidate_id_source_bound_controls_unparseable",
        "snerv_hard_byte_ceiling_not_receiver_satisfied_for_long_training",
        "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes",
        "snerv_lf_payload_recode_still_over_hard_byte_ceiling",
        "snerv_lf_recode_selected_mode_still_over_byte_waterline",
        "snerv_lf_recode_no_receiver_proven_byte_saving_mode",
        "snerv_modelsize_auto_calibrated_byte_cap_over_ceiling",
        "snerv_nominal_payload_far_over_ceiling_refuse_long_training",
        "snerv_pre_long_run_candidate_feedback_context_only",
        "snerv_pre_long_run_candidate_feedback_missing",
        "snerv_pre_long_run_candidate_feedback_not_ready",
        "snerv_pre_long_run_candidate_feedback_not_byte_closed",
        "snerv_pre_long_run_feedback_custody_paths_missing",
        "snerv_pre_long_run_full_video_mlx_feedback_kind_missing",
        "snerv_pre_long_run_full_video_mlx_prefilter_missing",
        "snerv_pre_long_run_full_video_mlx_scorer_response_missing",
        "snerv_pre_long_run_native_full600_export_not_ready",
        "snerv_pre_long_run_receiver_proof_missing_or_failed",
        "snerv_pre_long_run_scorer_loop_best_packet_not_materialized",
        "snerv_receiver_proven_archive_over_hard_byte_ceiling",
        "snerv_receiver_proven_archive_over_hard_byte_ceiling_observed_demote_only",
        "snerv_renderer_nondegenerate_compact_skip_high_value_domain_not_passed",
        "snerv_renderer_nondegenerate_export_value_domain_not_passed",
        "snerv_renderer_nondegenerate_receiver_reconstruction_not_verified",
        "snerv_renderer_nondegenerate_smoke_failed",
        "snerv_renderer_nondegenerate_smoke_min16_pairs_missing",
        "snerv_renderer_nondegenerate_smoke_missing",
        "snerv_renderer_nondegenerate_target_value_domain_not_passed",
        "snerv_renderer_nondegenerate_telemetry_contract_missing_or_failed",
        "snerv_renderer_nondegenerate_tether_gate_missing_or_failed",
        "snerv_score_aware_long_training_dual_posenet_lambda_never_active",
        "snerv_score_aware_long_training_dual_segnet_lambda_never_active",
        "snerv_score_aware_long_training_telemetry_contract_failed",
        "snerv_scorer_domain_tether_lambda_inactive_telemetry",
        "snerv_scorer_domain_tether_missing_telemetry",
        "snerv_scorer_loop_qat_no_accepted_improvement",
        "snerv_scorer_loop_qat_pose_guard_not_ready",
        "snerv_scorer_loop_qat_receiver_contract_failed",
        "snerv_scorer_tether_smoke_failed",
        "snerv_scorer_tether_smoke_report_missing",
        "snerv_scorer_tether_smoke_schema_mismatch",
        "snerv_posenet_yuv6_pair_distill_metric_missing_telemetry",
        "snerv_segnet_last_frame_distill_metric_missing_telemetry",
        "snerv_upstream_eval_gate_failed",
        "snerv_upstream_eval_gate_score_bad",
        "snerv_upstream_eval_gate_score_missing",
        "pr95_distortion_source_inventory_incomplete",
    }
    prefixes = (
        "snerv_source_bound_control_missing:",
        "snerv_official_mfu_hfr_tub_",
        "snerv_official_trained_checkpoint_",
        "snerv_official_tub_",
        "snerv_long_run_launch_gate_",
        "source_parity:",
        "hi_nerv_pr95_distortion_",
        "snerv_pr95_distortion_",
    )
    launch_blockers = _dedupe(
        [
            str(blocker)
            for blocker in blockers
            if str(blocker)
            and (str(blocker) in exact_names or any(str(blocker).startswith(prefix) for prefix in prefixes))
        ]
    )
    if not bounded_proof_launch:
        return launch_blockers
    return [
        blocker
        for blocker in launch_blockers
        if _bounded_snerv_proof_launch_blocker(blocker)
    ]


def _bounded_snerv_proof_launch_blocker(blocker: str) -> bool:
    """Keep only defects that invalidate the bounded SNeRV smoke itself."""

    if blocker in {
        "snerv_candidate_id_source_bound_controls_mismatch",
        "snerv_candidate_id_source_bound_controls_unparseable",
        "snerv_scorer_tether_smoke_failed",
        "snerv_scorer_tether_smoke_report_missing",
        "snerv_scorer_tether_smoke_schema_mismatch",
    }:
        return True
    return blocker.startswith("snerv_source_bound_control_missing:")


def _candidate_feedback_evidence_blockers(
    feedback: Mapping[str, Any],
    *,
    family: str | None = None,
) -> list[str]:
    """Carry candidate-feedback evidence debt without making it launch-blocking."""

    if not feedback:
        return []
    blockers = [str(blocker) for blocker in feedback.get("sample_generalization_blockers") or [] if blocker]
    gate = feedback.get("sample_generalization_gate")
    if isinstance(gate, Mapping):
        blockers.extend(str(blocker) for blocker in gate.get("blockers") or [])
    blockers.extend(_feedback_prioritized_pair_index_blockers(feedback))
    blockers.extend(str(blocker) for blocker in feedback.get("direct_feedback_blockers") or [] if blocker)
    blockers.extend(
        str(blocker) for blocker in feedback.get("snerv_official_trained_checkpoint_mapping_blockers") or [] if blocker
    )
    blockers.extend(str(blocker) for blocker in feedback.get("snerv_renderer_nondegenerate_blockers") or [] if blocker)
    blockers.extend(
        str(blocker)
        for blocker in feedback.get("snerv_mlx_native_training_export_guard_blockers") or []
        if blocker
    )
    blockers.extend(
        str(blocker)
        for blocker in feedback.get("snerv_mlx_native_file_backed_export_blockers") or []
        if blocker
    )
    if (
        _family_key(str(feedback.get("family") or "")) == "snerv"
        and feedback.get("snerv_official_trained_checkpoint_state_dict_slice_path")
        and feedback.get("snerv_official_trained_checkpoint_state_dict_slice_file_present")
        is not True
    ):
        blockers.append("snerv_official_trained_checkpoint_state_dict_slice_file_missing")
    if (
        _family_key(str(feedback.get("family") or "")) == "snerv"
        and feedback.get("snerv_mlx_native_file_backed_export_proof_passed")
        is not True
        and not feedback.get("snerv_mlx_native_export_packet_path")
    ):
        blockers.append("snerv_mlx_native_packet_file_missing")
    return _filter_family_feedback_blockers(blockers, family=family)


def _filter_family_feedback_blockers(
    blockers: Sequence[Any],
    *,
    family: str | None,
) -> list[str]:
    family_key = _family_key(str(family or ""))
    filtered: list[str] = []
    for raw in blockers:
        blocker = str(raw)
        if not blocker:
            continue
        token = blocker.removeprefix("source_parity:")
        if family_key == "hi_nerv" and token.startswith("snerv_"):
            continue
        if family_key == "snerv" and (
            token.startswith("hi_nerv_") or token.startswith("hinerv_")
        ):
            continue
        filtered.append(blocker)
    return _dedupe(filtered)


def _snerv_official_trained_checkpoint_state_dict_path_from_feedback(
    feedback: Mapping[str, Any],
) -> Path | None:
    """Return the runner-consumable exported state slice, if it is real."""

    for key in (
        "snerv_official_trained_checkpoint_state_dict_path",
        "snerv_official_trained_checkpoint_state_dict_slice_path",
    ):
        path = _existing_path(feedback.get(key))
        if path is not None:
            return path
    return None


def _snerv_feedback_blockers_after_tether_smoke(
    blockers: Sequence[str],
    scorer_tether_smoke_gate: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Clear stale tether-path feedback blockers once the local smoke proves the path."""

    values = _dedupe(str(blocker) for blocker in blockers if str(blocker))
    if scorer_tether_smoke_gate.get("passed") is not True:
        return values, []
    kept: list[str] = []
    suppressed: list[str] = []
    for blocker in values:
        if blocker in SNERV_SCORER_TETHER_FEEDBACK_BLOCKERS:
            suppressed.append(blocker)
        else:
            kept.append(blocker)
    return kept, suppressed


def _feedback_prioritized_pair_indices(feedback: Mapping[str, Any]) -> tuple[int, ...]:
    if not isinstance(feedback, Mapping):
        return ()
    if not _feedback_prioritized_pair_indices_routable(feedback):
        return ()
    try:
        return pair_indices_from_mapping(feedback)
    except HardPairIndicesError:
        return ()
    return ()


def _feedback_prioritized_pair_index_blockers(
    feedback: Mapping[str, Any],
) -> list[str]:
    if not isinstance(feedback, Mapping):
        return []
    try:
        pair_indices = pair_indices_from_mapping(feedback)
    except HardPairIndicesError:
        return ["candidate_feedback_prioritized_pair_indices_parse_failed"]
    if pair_indices and not _feedback_prioritized_pair_indices_routable(feedback):
        return ["candidate_feedback_prioritized_pair_indices_not_launch_routable"]
    return []


def _feedback_prioritized_pair_indices_routable(feedback: Mapping[str, Any]) -> bool:
    hard_pair = feedback.get("hard_pair_coverage")
    if isinstance(hard_pair, Mapping) and _hard_pair_coverage_routable(hard_pair):
        return True
    gate = feedback.get("sample_generalization_gate")
    if isinstance(gate, Mapping):
        nested = gate.get("hard_pair_coverage")
        if isinstance(nested, Mapping) and _hard_pair_coverage_routable(nested):
            return True
    if str(feedback.get("feedback_kind") or "") == "training_telemetry":
        return bool(
            feedback.get("scope_matches_candidate") is True
            and int(feedback.get("measured_num_pairs") or 0) >= 600
            and (
                "launch_hard_pair_prioritized_sampler_successor"
                in {str(value) for value in feedback.get("recommended_launch_mutations") or ()}
            )
        )
    return False


def _hard_pair_coverage_routable(coverage: Mapping[str, Any]) -> bool:
    return bool(
        coverage.get("representative_distortion_evidence") is True
        or coverage.get("score_axis_hard_pair_coverage") is True
        or coverage.get("coverage_valid_for_distortion") is True
    )


def _normalize_pair_index_sequence(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        return normalize_pair_indices(value)
    except HardPairIndicesError:
        return ()


def _prioritized_pair_training_plan(pair_indices: Sequence[int]) -> dict[str, Any]:
    normalized = _normalize_pair_index_sequence(pair_indices)
    return {
        "schema": "nerv_prioritized_pair_training_plan.v1",
        "enabled": bool(normalized),
        "pair_indices": [int(value) for value in normalized],
        "pair_count": len(normalized),
        "sampling_scope": "local_mlx_training_batch_emphasis_only",
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _snerv_prioritized_pair_training_plan(pair_indices: Sequence[int]) -> dict[str, Any]:
    normalized = _normalize_pair_index_sequence(pair_indices)
    if not normalized:
        return _prioritized_pair_training_plan(())
    return {
        "schema": "nerv_prioritized_pair_training_plan.v1",
        "enabled": False,
        "requested": True,
        "command_routed": False,
        "pair_indices": [],
        "requested_pair_indices": [int(value) for value in normalized],
        "blocked_pair_indices": [int(value) for value in normalized],
        "pair_count": 0,
        "requested_pair_count": len(normalized),
        "sampling_scope": "blocked_current_snerv_path_hydrates_source_subset_not_full_training_emphasis",
        "routing_status": "recorded_not_routed_current_snerv_hydrates_subset_not_training_emphasis",
        "required_successor": "snerv_full_video_scoreaware_trainer_with_sampler_emphasis",
        "blockers": ["snerv_hardpair_indices_only_hydrated_subset_not_full_training"],
        "authority": "macos_mlx_research_signal_false_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _source_parity_family_report(
    *,
    family: str,
    source_parity_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return row-local source-parity debt without granting score authority."""

    if not isinstance(source_parity_contract, Mapping):
        return {
            "schema": "nerv_row_source_parity_binding.v1",
            "contract_schema": None,
            "family": str(family),
            "long_training_ready": False,
            "required_blockers": ["source_parity:source_parity_contract_missing"],
            "nonblocking_gaps": [],
            "feature_status_rows": [],
            "control_status_rows": [],
            **FALSE_AUTHORITY,
        }
    feature_rows = [
        row
        for row in source_parity_contract.get("feature_rows") or ()
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    control_rows = [
        row
        for row in source_parity_contract.get("control_rows") or ()
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    source_audits = [
        row
        for row in source_parity_contract.get("source_audits") or ()
        if isinstance(row, Mapping) and row.get("family") == family
    ]
    required_blockers = _dedupe(
        [
            f"source_parity:{blocker}"
            for row in (*feature_rows, *control_rows)
            if row.get("required_for_long_training") is True
            for blocker in row.get("blockers") or ()
            if str(blocker)
        ]
    )
    nonblocking_gaps = _dedupe(
        [
            f"source_parity:{blocker}"
            for row in (*feature_rows, *control_rows)
            if row.get("required_for_long_training") is not True
            for blocker in row.get("blockers") or ()
            if str(blocker)
        ]
    )
    family_summary = {}
    for row in source_parity_contract.get("family_rows") or ():
        if isinstance(row, Mapping) and row.get("family") == family:
            family_summary = dict(row)
            break
    return {
        "schema": "nerv_row_source_parity_binding.v1",
        "contract_schema": source_parity_contract.get("schema"),
        "contract_authority": source_parity_contract.get("authority"),
        "family": str(family),
        "long_training_ready": bool(family_summary.get("long_training_ready", not required_blockers)),
        "required_blockers": list(required_blockers),
        "nonblocking_gaps": list(nonblocking_gaps),
        "source_audit_rows": [dict(row) for row in source_audits],
        "feature_status_rows": [
            {
                "feature_id": row.get("feature_id"),
                "status": row.get("status"),
                "required_for_long_training": bool(row.get("required_for_long_training")),
                "source_audit_rows": [
                    dict(audit) for audit in row.get("source_audit_rows") or () if isinstance(audit, Mapping)
                ],
                "blockers": list(row.get("blockers") or ()),
            }
            for row in feature_rows
        ],
        "control_status_rows": [
            {
                "control_id": row.get("control_id"),
                "status": row.get("status"),
                "required_for_long_training": bool(row.get("required_for_long_training")),
                "blockers": list(row.get("blockers") or ()),
            }
            for row in control_rows
        ],
        **FALSE_AUTHORITY,
    }


def _snerv_official_runtime_authority_split(
    source_parity: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate receiver-usable SNeRV evidence from source-forward authority."""

    if str(source_parity.get("family") or "") != "snerv":
        return {
            "schema": "snerv_official_runtime_authority_split.v1",
            "family": "snerv",
            "source_parity_binding_consumed": False,
            "receiver_bound_training_evidence_usable": False,
            "full_source_forward_authority_proven": False,
            "blockers": ["snerv_source_parity_binding_missing_or_wrong_family"],
            **FALSE_AUTHORITY,
        }
    audit_rows = _snerv_official_audit_rows_from_source_parity(source_parity)
    receiver_primitive_replay_proven = any(
        row.get("official_mfu_hfr_tub_receiver_primitives_proven") is True for row in audit_rows
    )
    numeric_graph_replay_proven = any(
        row.get("official_mfu_hfr_tub_numeric_graph_replay_proven") is True for row in audit_rows
    )
    receiver_runtime_decode_proven = any(
        row.get("official_receiver_runtime_decode_proven") is True for row in audit_rows
    )
    receiver_source_forward_replay_bound = any(
        row.get("official_receiver_source_forward_replay_bound") is True for row in audit_rows
    )
    full_stack_source_forward_replay_proven = any(
        row.get("full_stack_source_forward_replay_proven") is True for row in audit_rows
    )
    official_mfu_hfr_tub_parity_proven = any(
        row.get("official_mfu_hfr_tub_parity_proven") is True for row in audit_rows
    )
    runtime_decode_blockers = _dedupe(
        [
            str(blocker)
            for row in audit_rows
            for blocker in row.get("official_receiver_runtime_decode_blockers") or ()
            if blocker
        ]
    )
    source_gap_blockers = _dedupe(
        [
            str(blocker).removeprefix("source_parity:")
            for blocker in (
                *(source_parity.get("required_blockers") or ()),
                *(source_parity.get("nonblocking_gaps") or ()),
            )
            if "snerv_official_mfu_hfr_tub" in str(blocker)
        ]
    )
    receiver_bound_training_evidence_usable = bool(
        receiver_primitive_replay_proven and numeric_graph_replay_proven and receiver_runtime_decode_proven
    )
    full_source_forward_authority_proven = bool(
        official_mfu_hfr_tub_parity_proven
        and full_stack_source_forward_replay_proven
        and receiver_source_forward_replay_bound
    )
    blockers: list[str] = []
    if not receiver_primitive_replay_proven:
        blockers.append("snerv_official_mfu_hfr_tub_receiver_primitive_replay_missing")
    if not numeric_graph_replay_proven:
        blockers.append("snerv_official_mfu_hfr_tub_numeric_graph_replay_missing")
    if not receiver_runtime_decode_proven:
        blockers.append("snerv_official_receiver_runtime_decode_missing")
    if not full_source_forward_authority_proven:
        blockers.append("snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing")
    if full_source_forward_authority_proven:
        launch_semantics = "official_source_forward_parity_available_false_authority_until_score_gate"
    elif receiver_bound_training_evidence_usable:
        launch_semantics = "receiver_bound_training_allowed_but_official_source_authority_false"
    else:
        launch_semantics = "receiver_bound_training_waits_on_required_primitive_rows"
    return {
        "schema": "snerv_official_runtime_authority_split.v1",
        "family": "snerv",
        "source_parity_binding_consumed": True,
        "source_audit_row_count": len(audit_rows),
        "receiver_primitive_replay_proven": receiver_primitive_replay_proven,
        "numeric_graph_replay_proven": numeric_graph_replay_proven,
        "receiver_runtime_decode_proven": receiver_runtime_decode_proven,
        "receiver_source_forward_replay_bound": receiver_source_forward_replay_bound,
        "full_stack_source_forward_replay_proven": (full_stack_source_forward_replay_proven),
        "official_mfu_hfr_tub_parity_proven": official_mfu_hfr_tub_parity_proven,
        "receiver_bound_training_evidence_usable": (receiver_bound_training_evidence_usable),
        "full_source_forward_authority_proven": (full_source_forward_authority_proven),
        "runtime_decode_blockers": runtime_decode_blockers,
        "source_gap_blockers": source_gap_blockers,
        "launch_semantics": launch_semantics,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _snerv_source_forward_evidence_active(
    source_forward_evidence: Mapping[str, Any] | None,
) -> bool:
    return bool(
        isinstance(source_forward_evidence, Mapping)
        and _positive_int_or_none(source_forward_evidence.get("artifact_count")) is not None
    )


def _select_snerv_official_replacement_authority_gate(
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    ready: list[Mapping[str, Any]] = []
    valid: list[Mapping[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, Mapping):
            continue
        if gate.get("schema") != "snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1":
            continue
        valid.append(gate)
        if (
            gate.get("official_tub_lf_hf_decoder_replacement_ready") is True
            and gate.get("official_checkpoint_export_binding_ready") is True
            and gate.get("receiver_output2_frame_replay_ready") is True
            and gate.get("tub_source_fixture_replay_ready") is True
            and gate.get("trained_checkpoint_state_dict_mapping_ready") is True
            and gate.get("tub_temporal_output2_weight_mapping_ready") is True
            and gate.get("full_tub_source_forward_replay_ready") is True
        ):
            ready.append(gate)
    selected = (ready or valid)[0] if (ready or valid) else None
    return dict(selected) if isinstance(selected, Mapping) else None


def _snerv_long_run_launch_gate_status(
    verdict: Mapping[str, Any] | None,
    *,
    bounded_proof_only: bool,
) -> dict[str, Any]:
    if bounded_proof_only:
        return {
            "schema": "snerv_long_run_launch_gate_consumption.v1",
            "required": False,
            "approved": True,
            "reason": "bounded_proof_only_rows_are_allowed_to_produce_gate_evidence",
            "blockers": [],
            **FALSE_AUTHORITY,
        }
    blockers: list[str] = []
    if not isinstance(verdict, Mapping):
        blockers.append("snerv_long_run_launch_gate_verdict_missing")
        return {
            "schema": "snerv_long_run_launch_gate_consumption.v1",
            "required": True,
            "approved": False,
            "verdict_schema": None,
            "gate_highest_level": None,
            "gate_blocking_evidence": [],
            "source_forward_action_effect_indexed": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
    if verdict.get("schema") != NERV_LONG_RUN_LAUNCH_GATE_SCHEMA:
        blockers.append("snerv_long_run_launch_gate_schema_invalid")
    if str(verdict.get("family") or "").strip().lower().replace("-", "_") != "snerv":
        blockers.append("snerv_long_run_launch_gate_family_invalid")
    if verdict.get("approved") is not True:
        blockers.append("snerv_long_run_launch_gate_not_approved")
    if str(verdict.get("highest_level") or "") != "L4":
        blockers.append("snerv_long_run_launch_gate_not_l4")
    gate_blockers = [str(v) for v in verdict.get("blocking_evidence") or [] if v]
    blockers.extend(f"snerv_long_run_launch_gate_blocker:{value}" for value in gate_blockers)
    evidence_index = verdict.get("evidence_index")
    source_forward_indexed = (
        isinstance(evidence_index, Mapping)
        and bool(evidence_index.get(SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA))
    )
    if not source_forward_indexed:
        blockers.append("snerv_long_run_launch_gate_source_forward_action_effect_missing")
    blockers = _dedupe(blockers)
    return {
        "schema": "snerv_long_run_launch_gate_consumption.v1",
        "required": True,
        "approved": not blockers,
        "verdict_schema": verdict.get("schema"),
        "gate_highest_level": verdict.get("highest_level"),
        "gate_blocking_evidence": gate_blockers,
        "source_forward_action_effect_indexed": bool(source_forward_indexed),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _looks_like_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _nonnegative_finite_float(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed >= 0.0 and parsed < float("inf")


def _snerv_source_forward_numerical_proof_complete(
    source_forward_evidence: Mapping[str, Any] | None,
) -> bool:
    """Require typed source-forward action-effect proof, not legacy metadata."""

    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return False
    assert source_forward_evidence is not None
    status = source_forward_evidence.get("source_forward_replay_proof_status")
    if isinstance(status, Mapping):
        action_effect = status.get("source_forward_proof_action_effect")
        if isinstance(action_effect, Mapping):
            validation = validate_snerv_source_forward_proof_action_effect(action_effect)
            if validation.get("passed") is not True:
                return False
        elif status.get("source_forward_replay_action_effect_valid") is not True:
            return False
        if status.get("source_forward_replay_numerical_proof_complete") is True:
            return True
        if status.get("source_forward_replay_required_fields_missing"):
            return False
        if status.get("source_forward_replay_invalid_fields"):
            return False
    proof = source_forward_evidence.get("source_forward_replay_proof")
    if isinstance(proof, Mapping) and proof.get("schema") == SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA:
        validation = validate_snerv_source_forward_proof_action_effect(proof)
        return bool(validation.get("passed") is True)
    return False


def _hinerv_pr95_actuator_execution_evidence_from_feedback(
    feedback: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(feedback, Mapping):
        return None
    direct = feedback.get("pr95_scorer_atom_actuator_execution_evidence")
    if isinstance(direct, Mapping):
        return dict(direct)
    bootstrap = feedback.get("hi_nerv_scorer_domain_bootstrap")
    if isinstance(bootstrap, Mapping):
        nested = bootstrap.get("pr95_scorer_atom_actuator_execution_evidence")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _snerv_pr95_actuator_execution_evidence_from_source_forward(
    source_forward_evidence: Mapping[str, Any] | None,
    *,
    official_replacement_authority_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Translate canonical SNeRV source-forward proof into PR95 actuator evidence.

    The PR95 guard expects family-local execution evidence, not another copy of
    the source-forward summary.  This bridge only materializes that guard input
    after the stricter SNeRV source-forward summary has already proven the
    byte-bound checkpoint artifact, full TUB source-forward parity, and output2
    receiver binding.
    """

    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return None
    assert source_forward_evidence is not None
    artifact_bytes = _positive_int_or_none(
        source_forward_evidence.get(
            "official_trained_checkpoint_state_dict_slice_bytes"
        )
    )
    artifact_sha256 = str(
        source_forward_evidence.get(
            "official_trained_checkpoint_state_dict_slice_sha256"
        )
        or ""
    ).strip()
    if (
        source_forward_evidence.get(
            "official_trained_checkpoint_state_dict_value_artifact_ready"
        )
        is not True
        or source_forward_evidence.get("source_forward_replay_authority") is not True
        or source_forward_evidence.get("full_tub_source_forward_parity_proven")
        is not True
        or not _snerv_source_forward_numerical_proof_complete(
            source_forward_evidence
        )
        or source_forward_evidence.get("receiver_frame_decode_consumes_output2")
        is not True
        or source_forward_evidence.get("official_checkpoint_export_bound")
        is not True
        or artifact_bytes is None
        or len(artifact_sha256) != 64
    ):
        return None
    evidence = {
        "schema": "pr95_scorer_atom_actuator_execution_evidence.v1",
        "family": "snerv",
        "source": "snerv_lf_hf_source_forward_evidence",
        "source_schema": source_forward_evidence.get("schema"),
        "source_path": source_forward_evidence.get("source_path"),
        "source_sha256": source_forward_evidence.get("source_sha256"),
        "state_artifact_schema": "snerv_official_source_forward_state_artifact.v1",
        "official_state_dict_value_artifact_bytes": int(artifact_bytes),
        "official_state_dict_value_artifact_sha256": artifact_sha256,
        "official_state_dict_value_artifact_path": source_forward_evidence.get(
            "official_trained_checkpoint_state_dict_slice_path"
        ),
        "official_state_dict_value_artifact_member_count": (
            source_forward_evidence.get(
                "official_trained_checkpoint_state_dict_slice_member_count"
            )
        ),
        "official_state_dict_value_artifact_member_names": list(
            source_forward_evidence.get(
                "official_trained_checkpoint_state_dict_slice_member_names"
            )
            or []
        ),
        "source_forward_replay_proof": (
            dict(source_forward_evidence["source_forward_replay_proof"])
            if isinstance(source_forward_evidence.get("source_forward_replay_proof"), Mapping)
            else None
        ),
        "source_forward_replay_proof_status": (
            dict(source_forward_evidence["source_forward_replay_proof_status"])
            if isinstance(
                source_forward_evidence.get("source_forward_replay_proof_status"),
                Mapping,
            )
            else None
        ),
        "checkpoint_export_lineage_bound": True,
        "mfu_hfr_tub_source_forward_parity_proven": True,
        "tub_output2_source_forward_parity_proven": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    servo_receipt = source_forward_evidence.get("pair_local_distortion_servo_receipt")
    if isinstance(servo_receipt, Mapping):
        evidence["pair_local_distortion_servo_receipt"] = dict(servo_receipt)
    if isinstance(official_replacement_authority_gate, Mapping):
        evidence["snerv_official_tub_lf_hf_decoder_replacement_authority_gate"] = dict(
            official_replacement_authority_gate
        )
    return evidence


def _snerv_source_forward_closed_blockers(
    source_forward_evidence: Mapping[str, Any] | None,
) -> set[str]:
    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return set()
    numerical_proof_complete = _snerv_source_forward_numerical_proof_complete(
        source_forward_evidence
    )
    source_authority_blockers = {
        "snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority",
        "snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing",
        "snerv_official_trained_checkpoint_source_forward_replay_missing",
    }
    return {
        str(blocker)
        for blocker in source_forward_evidence.get("closed_campaign_blockers") or ()
        if str(blocker)
        and (
            numerical_proof_complete
            or _source_forward_blocker_token(blocker) not in source_authority_blockers
        )
    }


def _snerv_source_forward_queue_blockers(
    source_forward_evidence: Mapping[str, Any] | None,
) -> list[str]:
    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return []
    blockers = _dedupe(
        [
            str(blocker)
            for blocker in source_forward_evidence.get("queue_blockers") or ()
            if str(blocker)
        ]
    )
    if not _snerv_source_forward_numerical_proof_complete(source_forward_evidence):
        blockers.append(
            "snerv_official_mfu_hfr_tub_numerical_source_forward_proof_missing"
        )
    return _dedupe(blockers)


def _source_forward_blocker_token(blocker: Any) -> str:
    return str(blocker).strip().removeprefix("source_parity:")


def _without_closed_source_forward_blockers(
    blockers: Sequence[Any],
    source_forward_evidence: Mapping[str, Any] | None,
) -> list[str]:
    closed = _snerv_source_forward_closed_blockers(source_forward_evidence)
    if not closed:
        return _dedupe([str(blocker) for blocker in blockers if str(blocker)])
    return _dedupe(
        [
            str(blocker)
            for blocker in blockers
            if str(blocker) and _source_forward_blocker_token(blocker) not in closed
        ]
    )


_SNERV_SOURCE_FORWARD_FEEDBACK_BLOCKER_KEYS = {
    "blockers",
    "direct_feedback_blockers",
    "snerv_official_trained_checkpoint_mapping_blockers",
    "snerv_mlx_native_export_blockers",
    "snerv_mlx_native_file_backed_export_blockers",
    "official_source_parity_blockers",
}


def _snerv_feedback_with_source_forward_evidence(
    feedback: Mapping[str, Any],
    source_forward_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply newer source-forward authority proof to stale candidate feedback.

    Candidate feedback rows can predate the source-forward authority artifact.
    Keep the row-local feedback useful for telemetry and byte evidence, but do
    not let old MFU/HFR/TUB source blockers survive as current planner debt once
    the source-forward proof closes them.
    """

    out = dict(feedback)
    closed = _snerv_source_forward_closed_blockers(source_forward_evidence)
    if not out or not closed:
        return out

    def scrub(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {str(k): scrub(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            if key in _SNERV_SOURCE_FORWARD_FEEDBACK_BLOCKER_KEYS:
                return _without_closed_source_forward_blockers(
                    value,
                    source_forward_evidence,
                )
            return [scrub(item) for item in value]
        return value

    scrubbed = scrub(out)
    assert isinstance(scrubbed, dict)
    superseded = sorted(
        {
            _source_forward_blocker_token(blocker)
            for key in _SNERV_SOURCE_FORWARD_FEEDBACK_BLOCKER_KEYS
            for blocker in out.get(key) or ()
            if _source_forward_blocker_token(blocker) in closed
        }
    )
    if superseded:
        scrubbed["snerv_official_source_forward_evidence_consumed"] = True
        scrubbed["snerv_official_source_forward_superseded_feedback_blockers"] = (
            superseded
        )
    return scrubbed


def _snerv_source_parity_with_source_forward_evidence(
    source_parity: Mapping[str, Any],
    source_forward_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = dict(source_parity)
    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return out
    out["required_blockers"] = _without_closed_source_forward_blockers(
        out.get("required_blockers") or (),
        source_forward_evidence,
    )
    out["nonblocking_gaps"] = _without_closed_source_forward_blockers(
        out.get("nonblocking_gaps") or (),
        source_forward_evidence,
    )
    out["snerv_official_source_forward_evidence_consumed"] = True
    out["snerv_official_source_forward_evidence"] = dict(source_forward_evidence)
    return out


def _snerv_runtime_authority_split_with_source_forward_evidence(
    split: Mapping[str, Any],
    source_forward_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out = dict(split)
    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return out
    queue_blockers = _snerv_source_forward_queue_blockers(source_forward_evidence)
    blockers = _dedupe(
        [
            *_without_closed_source_forward_blockers(
                out.get("blockers") or (),
                source_forward_evidence,
            ),
            *queue_blockers,
        ]
    )
    receiver_bound_export = bool(
        source_forward_evidence.get("receiver_bound_export_proven")
        or source_forward_evidence.get("receiver_payload_frame_replay_proven")
    )
    numerical_proof_complete = _snerv_source_forward_numerical_proof_complete(
        source_forward_evidence
    )
    source_authority = bool(
        source_forward_evidence.get("source_forward_replay_authority")
        and numerical_proof_complete
    )
    full_tub_parity = bool(
        source_forward_evidence.get("full_tub_source_forward_parity_proven")
        and numerical_proof_complete
    )
    out.update(
        {
            "snerv_official_source_forward_evidence_consumed": True,
            "snerv_official_source_forward_evidence": dict(source_forward_evidence),
            "official_checkpoint_export_bound": bool(
                source_forward_evidence.get("official_checkpoint_export_bound")
            ),
            "receiver_bound_export_proven": receiver_bound_export,
            "receiver_payload_frame_replay_proven": bool(
                source_forward_evidence.get("receiver_payload_frame_replay_proven")
            ),
            "receiver_frame_decode_consumes_output2": bool(
                source_forward_evidence.get("receiver_frame_decode_consumes_output2")
            ),
            "source_forward_numerical_proof_complete": numerical_proof_complete,
            "full_tub_source_forward_parity_proven": full_tub_parity,
            "source_forward_replay_authority": source_authority,
            "receiver_bound_training_evidence_usable": bool(
                out.get("receiver_bound_training_evidence_usable")
                or receiver_bound_export
            ),
            "full_source_forward_authority_proven": bool(
                source_authority and full_tub_parity and not blockers
            ),
            "blockers": blockers,
        }
    )
    if out["full_source_forward_authority_proven"]:
        out["launch_semantics"] = (
            "official_source_forward_parity_available_false_authority_until_score_gate"
        )
    elif out["receiver_bound_training_evidence_usable"]:
        out["launch_semantics"] = (
            "receiver_bound_training_allowed_but_official_source_authority_false"
        )
    else:
        out["launch_semantics"] = "receiver_bound_training_waits_on_required_primitive_rows"
    return out


def _snerv_curriculum_with_source_forward_evidence(
    curriculum: Mapping[str, Any],
    source_forward_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _snerv_source_forward_evidence_active(source_forward_evidence):
        return dict(curriculum)
    out = dict(curriculum)
    split = out.get("official_source_forward_authority_split")
    if isinstance(split, Mapping):
        updated_split = _snerv_source_forward_split_with_evidence(
            split,
            source_forward_evidence,
        )
        out["official_source_forward_authority_split"] = updated_split
    else:
        updated_split = None
    blockers = _without_closed_source_forward_blockers(
        out.get("blockers") or (),
        source_forward_evidence,
    )
    if isinstance(updated_split, Mapping):
        blockers = _dedupe([*blockers, *(updated_split.get("blockers") or ())])
    training_plan = out.get("training_plan")
    if isinstance(training_plan, Mapping):
        updated_training_plan = dict(training_plan)
        training_split = updated_training_plan.get("official_source_forward_authority_split")
        if isinstance(training_split, Mapping):
            updated_training_plan["official_source_forward_authority_split"] = (
                _snerv_source_forward_split_with_evidence(
                    training_split,
                    source_forward_evidence,
                )
            )
        out["training_plan"] = updated_training_plan
    out["blockers"] = blockers
    out["snerv_official_source_forward_evidence_consumed"] = True
    out["snerv_official_source_forward_evidence"] = dict(source_forward_evidence)
    return out


def _snerv_source_forward_split_with_evidence(
    split: Mapping[str, Any],
    source_forward_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(split)
    queue_blockers = _snerv_source_forward_queue_blockers(source_forward_evidence)
    blockers = _dedupe(
        [
            *_without_closed_source_forward_blockers(
                out.get("blockers") or (),
                source_forward_evidence,
            ),
            *queue_blockers,
        ]
    )
    official_blockers = _dedupe(
        [
            *_without_closed_source_forward_blockers(
                out.get("official_blockers") or (),
                source_forward_evidence,
            ),
            *queue_blockers,
        ]
    )
    export_bound = bool(
        out.get("export_bound") or source_forward_evidence.get("official_checkpoint_export_bound")
    )
    receiver_payload_bound = bool(
        out.get("receiver_payload_bound")
        or source_forward_evidence.get("receiver_payload_frame_replay_proven")
    )
    frame_producing_export = bool(
        out.get("frame_producing_export")
        or source_forward_evidence.get("frame_producing_official_payload_replay_proven")
    )
    numerical_proof_complete = _snerv_source_forward_numerical_proof_complete(
        source_forward_evidence
    )
    source_authority = bool(
        source_forward_evidence.get("source_forward_replay_authority")
        and numerical_proof_complete
    )
    full_tub_parity = bool(
        source_forward_evidence.get("full_tub_source_forward_parity_proven")
        and numerical_proof_complete
    )
    full_authority = bool(
        source_authority
        and full_tub_parity
        and export_bound
        and receiver_payload_bound
        and frame_producing_export
        and not blockers
    )
    if full_authority:
        launch_semantics = (
            "official_source_forward_parity_available_false_authority_until_score_gate"
        )
    elif receiver_payload_bound:
        launch_semantics = (
            "receiver_bound_training_allowed_but_official_source_authority_false"
        )
    else:
        launch_semantics = "official_training_waits_on_receiver_payload_binding"
    out.update(
        {
            "snerv_official_source_forward_evidence_consumed": True,
            "snerv_official_source_forward_evidence": dict(source_forward_evidence),
            "export_bound": export_bound,
            "receiver_payload_bound": receiver_payload_bound,
            "frame_producing_export": frame_producing_export,
            "source_forward_replay_bound": bool(
                out.get("source_forward_replay_bound") or source_authority
            ),
            "source_forward_replay_verified": bool(
                out.get("source_forward_replay_verified") or full_tub_parity
            ),
            "source_forward_replay_authority": bool(
                out.get("source_forward_replay_authority") or source_authority
            ),
            "source_faithful_stack": bool(
                out.get("source_faithful_stack") or source_authority
            ),
            "source_forward_numerical_proof_complete": numerical_proof_complete,
            "export_bound_semantics": out.get("export_bound_semantics")
            or (
                "official_checkpoint_export_bound_not_source_forward_parity"
                if export_bound
                else None
            ),
            "receiver_bound_training_evidence_usable": bool(receiver_payload_bound),
            "full_source_forward_authority_proven": full_authority,
            "official_blockers": official_blockers,
            "launch_semantics": launch_semantics,
            "blockers": blockers,
        }
    )
    return out


def _snerv_official_audit_rows_from_source_parity(
    source_parity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in source_parity.get("source_audit_rows") or ():
        if isinstance(raw, Mapping):
            rows.append(dict(raw))
    for feature in source_parity.get("feature_status_rows") or ():
        if not isinstance(feature, Mapping):
            continue
        if feature.get("feature_id") != "snerv_official_mfu_hfr_tub_parity":
            continue
        for raw in feature.get("source_audit_rows") or ():
            if isinstance(raw, Mapping):
                rows.append(dict(raw))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _row_output_report_path(command_argv: Sequence[str]) -> str:
    return (_row_output_dir(command_argv) / "compact_renderer_mlx_spine_runner_report.json").as_posix()


def _row_output_dir(command_argv: Sequence[str]) -> Path:
    argv = [str(value) for value in command_argv]
    try:
        out_dir = argv[argv.index("--output-dir") + 1]
    except (ValueError, IndexError):
        out_dir = DEFAULT_OUTPUT_ROOT
    return Path(out_dir)


def _row_receiver_surface_trace_path(*, family: str, output_dir: Path) -> Path:
    family_key = str(family).strip().lower().replace("-", "_")
    if family_key == "hinerv":
        family_key = "hi_nerv"
    if family_key == "snerv":
        return output_dir / "snerv_mlx_training" / "nerv_crux_trace_rows.json"
    return output_dir / "hi_nerv_mlx_training" / "nerv_crux_trace_rows.json"


def _receiver_surface_trace_contract(*, family: str, output_dir: Path) -> dict[str, Any]:
    family_key = str(family).strip().lower().replace("-", "_")
    if family_key == "hinerv":
        family_key = "hi_nerv"
    if family_key not in {"hi_nerv", "snerv"}:
        family_key = "hi_nerv"
    return {
        "schema": RECEIVER_SURFACE_TRACE_CONTRACT_SCHEMA,
        "family": family_key,
        "trace_artifact_path": _row_receiver_surface_trace_path(
            family=family_key,
            output_dir=output_dir,
        ).as_posix(),
        "trace_row_schema": "nerv_crux_trace_rows.v1",
        "trace_source": "accepted_live_mlx_updates_only",
        "canonical_metric_prefix": "receiver_surface_",
        "accepted_update_trace_required": True,
        "legacy_aliases_accepted": False,
        "required_canonical_metrics": [
            "receiver_surface_trace_present",
            "receiver_surface_loss_delta",
            "receiver_surface_uint8_changed_pixels",
            "receiver_surface_segnet_input_delta_linf",
            "receiver_surface_worst_region_margin_p50_delta",
            "receiver_surface_argmax_flipped_pixels",
            "receiver_surface_pose_output_delta",
            "receiver_surface_fakequant_argmax_flipped_pixels",
            "receiver_surface_parseback_argmax_flipped_pixels",
            "receiver_surface_inflated_argmax_flipped_pixels",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _archive_parseback_selection_contract(
    *,
    family: str,
    axis_trace_contract: Any,
) -> dict[str, Any]:
    family_key = str(family).strip().lower().replace("-", "_")
    if family_key == "hinerv":
        family_key = "hi_nerv"
    if family_key not in {"hi_nerv", "snerv"}:
        family_key = "hi_nerv"
    axis_contract = axis_trace_contract if isinstance(axis_trace_contract, Mapping) else {}
    axes = [str(axis) for axis in axis_contract.get("required_axes") or ()]
    policy = axis_contract.get("acceptance_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    return {
        "schema": ARCHIVE_PARSEBACK_SELECTION_CONTRACT_SCHEMA,
        "family": family_key,
        "source_axis_trace_contract_schema": axis_contract.get("schema"),
        "archive_parseback_axis_required": "archive_parseback" in set(axes),
        "parseback_selection_required": True,
        "parseback_score_delta_must_be_bounded_before_stage6": (
            policy.get("parseback_score_delta_must_be_bounded_before_stage6") is True
        ),
        "live_only_improvement_is_false_authority": (
            policy.get("live_only_improvement_is_false_authority") is True
        ),
        "fail_closed_on_axis_divergence": (
            policy.get("fail_closed_on_axis_divergence") is True
        ),
        "selection_authority_order": [
            "live_forward",
            "fakequant_forward",
            "archive_parseback",
            "inflate_replay",
            "official_evaluate_py",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _row_observable_artifacts(*, family: str, output_dir: Path) -> list[str]:
    artifacts = [(output_dir / "compact_renderer_mlx_spine_runner_startup.json").as_posix()]
    if str(family) == "hi_nerv":
        artifacts.extend(
            [
                (output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl").as_posix(),
                (output_dir / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl").as_posix(),
                (output_dir / "hi_nerv_mlx_training" / "nerv_crux_trace_rows.json").as_posix(),
            ]
        )
    elif str(family) == "snerv":
        artifacts.extend(
            [
                (output_dir / "compact_renderer_mlx_spine_runner_report.json").as_posix(),
                (output_dir / "nerv_candidate_byte_feedback_row.json").as_posix(),
                (output_dir / "nerv_candidate_byte_feedback.jsonl").as_posix(),
                (
                    output_dir
                    / "snerv_mlx_native_export"
                    / "snerv_mlx_native_export_attachment.json"
                ).as_posix(),
                (
                    output_dir
                    / "snerv_mlx_native_export"
                    / "native_train_export"
                    / "snerv_mlx_native_train_export.json"
                ).as_posix(),
                (
                    output_dir
                    / "snerv_mlx_native_export"
                    / "native_train_export"
                    / "snerv_score_aware_long_training"
                    / "long_training"
                    / "telemetry.jsonl"
                ).as_posix(),
                (output_dir / "snerv_mlx_training" / "nerv_crux_trace_rows.json").as_posix(),
                (
                    output_dir
                    / "snerv_mlx_native_export"
                    / "native_train_export"
                    / "snerv_score_aware_long_training"
                    / "long_training"
                    / "nerv_crux_trace_rows.json"
                ).as_posix(),
            ]
        )
    return artifacts


def _score_lowering_gate(
    *,
    family: str,
    local_mlx_launch_command_ready: bool,
    curriculum_plan: Mapping[str, Any],
    blockers: Sequence[str],
    bounded_proof_launch: bool = False,
) -> dict[str, Any]:
    """Separate launchability from score/promotion authority for campaign rows."""

    binding = (
        curriculum_plan.get("pr95_stack_binding")
        if isinstance(curriculum_plan.get("pr95_stack_binding"), Mapping)
        else {}
    )
    gate = (
        curriculum_plan.get("long_campaign_prelaunch_gate")
        if isinstance(curriculum_plan.get("long_campaign_prelaunch_gate"), Mapping)
        else {}
    )
    missing_rows = [
        dict(row) for row in binding.get("rows", []) if isinstance(row, Mapping) and row.get("satisfied") is not True
    ]
    missing_requirement_ids = [str(row.get("requirement_id")) for row in missing_rows if row.get("requirement_id")]
    post_run_requirements = [str(item) for item in gate.get("post_run_requirements_excluded", []) if item]
    post_run_missing = [requirement for requirement in missing_requirement_ids if requirement in post_run_requirements]
    promotion_blockers = _dedupe(
        [
            *(str(blocker) for blocker in blockers if blocker),
            *(f"{family}_{requirement}_missing" for requirement in post_run_missing),
        ]
    )
    launch_blockers = _experiment_launch_blockers(
        blockers,
        bounded_proof_launch=bool(bounded_proof_launch),
    )
    prelaunch_blockers = _dedupe(
        [
            *(str(blocker) for blocker in gate.get("blockers", []) if blocker),
            *launch_blockers,
        ]
    )
    local_proof_launch_allowed = bool(local_mlx_launch_command_ready) and not launch_blockers
    cpu_replay_ready = (
        bool(local_proof_launch_allowed)
        and "receiver_proof" not in post_run_missing
        and "full_video_local_prefilter" not in post_run_missing
        and "local_cpu_replay_gate" not in post_run_missing
    )
    exact_gate_ready = cpu_replay_ready and "exact_auth_gate_plan" not in post_run_missing and not promotion_blockers
    return {
        "schema": SCORE_LOWERING_GATE_SCHEMA,
        "family": str(family),
        "command_materialized": bool(local_mlx_launch_command_ready),
        "local_mlx_executable": bool(local_proof_launch_allowed),
        "prelaunch_allowed": bool(local_proof_launch_allowed),
        "promotion_prelaunch_allowed": bool(gate.get("launch_allowed")),
        "prelaunch_blockers": prelaunch_blockers,
        "launch_blockers": launch_blockers,
        "post_run_requirements": post_run_requirements,
        "missing_requirement_ids": _dedupe(missing_requirement_ids),
        "post_run_missing_requirement_ids": _dedupe(post_run_missing),
        "receiver_proof_required": "receiver_proof" in post_run_missing,
        "full_video_prefilter_required": "full_video_local_prefilter" in post_run_missing,
        "local_cpu_replay_required": "local_cpu_replay_gate" in post_run_missing,
        "exact_auth_gate_required": "exact_auth_gate_plan" in post_run_missing,
        "cpu_replay_ready": bool(cpu_replay_ready),
        "exact_gate_ready": bool(exact_gate_ready),
        "promotion_blockers": promotion_blockers,
        **FALSE_AUTHORITY,
    }


def _require_schema(payload: Mapping[str, Any], schema: str, name: str) -> None:
    if not isinstance(payload, Mapping):
        raise NervLongTrainingCampaignPlanError(f"{name} must be a mapping")
    if payload.get("schema") != schema:
        raise NervLongTrainingCampaignPlanError(f"{name} schema must be {schema}; got {payload.get('schema')}")


def _selected_candidates(
    payload: Mapping[str, Any],
    *,
    family: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("selected_candidates", []):
        if not isinstance(row, Mapping) or row.get("family") != family:
            continue
        authority_blockers = _candidate_authority_blockers(row)
        candidate = _scrub_candidate_authority_flags(row)
        if authority_blockers:
            candidate["_candidate_authority_blockers"] = authority_blockers
        rows.append(candidate)
    if family == "snerv":
        rows.sort(key=_snerv_long_training_candidate_sort_key)
    elif family == "hi_nerv":
        rows.sort(key=_hinerv_long_training_candidate_sort_key)
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        deduped.setdefault(candidate_id, row)
    rows = list(deduped.values())
    return rows[: max(1, int(limit))]


def _merge_modelsize_byte_cap_feedback_candidates(
    *,
    selected_candidates: Sequence[Mapping[str, Any]],
    modelsize_budget: Mapping[str, Any],
    family: str,
    feedback_paths: Sequence[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not feedback_paths:
        return [dict(row) for row in selected_candidates]
    observations = _modelsize_byte_cap_feedback_observations(
        family=family,
        feedback_paths=feedback_paths,
    )
    if not observations:
        return [dict(row) for row in selected_candidates]
    all_candidates = _selected_candidates(
        modelsize_budget,
        family=family,
        limit=max(
            len(tuple(modelsize_budget.get("selected_candidates", ()))),
            int(limit),
            1,
        ),
    )
    feedback_candidates: list[dict[str, Any]] = []
    for observation in observations:
        candidate = observation.get("modelsize_candidate")
        if not isinstance(candidate, Mapping) or candidate.get("family") != family:
            continue
        clean = _scrub_candidate_authority_flags(candidate)
        candidate_id = str(clean.get("candidate_id") or "")
        if not candidate_id:
            continue
        authority_blockers = _candidate_authority_blockers(candidate)
        if authority_blockers:
            clean["_candidate_authority_blockers"] = authority_blockers
        observed_archive_bytes = _first_present_int(
            observation,
            ("measured_archive_bytes",),
        )
        hard_byte_ceiling = _first_present_int(clean, ("hard_byte_ceiling",))
        if (
            observed_archive_bytes is not None
            and hard_byte_ceiling is not None
            and int(observed_archive_bytes) > int(hard_byte_ceiling)
        ):
            over_ceiling = _first_present_int(
                observation,
                ("calibrated_archive_overrun_bytes",),
            ) or int(observed_archive_bytes) - int(hard_byte_ceiling)
            clean["_candidate_authority_blockers"] = _dedupe(
                [
                    *list(clean.get("_candidate_authority_blockers") or []),
                    f"{family}_receiver_proven_archive_over_hard_byte_ceiling_observed_demote_only",
                ]
            )
            clean["_modelsize_feedback_demote_only"] = True
            clean["_modelsize_feedback_observed_archive_bytes"] = int(observed_archive_bytes)
            clean["_modelsize_feedback_archive_over_hard_byte_ceiling_bytes"] = int(over_ceiling)
            required_nominal_max = _first_present_int(
                observation,
                ("required_nominal_payload_bytes_max",),
            )
            if required_nominal_max is not None:
                clean["_modelsize_feedback_required_nominal_payload_bytes_max"] = int(required_nominal_max)
            if observation.get("hard_byte_ceiling_measurement_bypass_enabled") is not None:
                clean["_modelsize_feedback_measurement_bypass_enabled"] = bool(
                    observation.get("hard_byte_ceiling_measurement_bypass_enabled")
                )
        feedback_candidates.append(clean)
    all_candidates = [*all_candidates, *feedback_candidates]
    selected_by_id = {str(row.get("candidate_id") or ""): dict(row) for row in selected_candidates}
    merged: dict[str, dict[str, Any]] = {str(row.get("candidate_id") or ""): dict(row) for row in selected_candidates}
    for candidate in all_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        matching = _modelsize_byte_cap_matching_observations(
            observations,
            candidate=candidate,
            codec=_modelsize_byte_cap_codec(candidate),
        )
        if matching:
            merged.setdefault(candidate_id, dict(candidate))
    rows = list(merged.values())

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        preflight = _modelsize_byte_cap_preflight(
            candidate=row,
            family=family,
            feedback_paths=feedback_paths,
        )
        matched = int(preflight.get("matching_observation_count") or 0)
        predicted_under = preflight.get("predicted_under_hard_byte_ceiling") is True
        predicted_archive = int(preflight.get("predicted_archive_bytes") or 0)
        selected_rank = 0 if str(row.get("candidate_id") or "") in selected_by_id else 1
        family_key = (
            _snerv_long_training_candidate_sort_key(row)
            if family == "snerv"
            else _hinerv_long_training_candidate_sort_key(row)
        )
        if row.get("_modelsize_feedback_demote_only") is True:
            return (3, predicted_archive, selected_rank, *family_key)
        return (
            0 if matched and predicted_under else 1 if matched else 2,
            predicted_archive if matched and predicted_under else 0,
            selected_rank,
            *family_key,
        )

    rows.sort(key=sort_key)
    limited = rows[: max(1, int(limit))]
    limited_ids = {str(row.get("candidate_id") or "") for row in limited}
    demotion_rows = [
        row
        for row in rows[max(1, int(limit)) :]
        if row.get("_modelsize_feedback_demote_only") is True and str(row.get("candidate_id") or "") not in limited_ids
    ]
    return [*limited, *demotion_rows]


def _merge_hinerv_waterfill_candidate_evidence(
    *,
    candidates: Sequence[Mapping[str, Any]],
    decoder_weight_waterfill_index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Prefer source-bound HiNeRV candidates that already have waterfill evidence.

    Modelsize budgets and receiver-proof archive ladders can be generated by
    different sweeps. The campaign planner should not silently drop a
    receiver-backed waterfill row just because its candidate was not in the
    newest top-N budget. When the waterfill source preserves the exact
    ``modelsize_candidate`` row, it is safe to add it to the selection pool
    while keeping all launcher and promotion gates false-authority.
    """

    merged: dict[str, dict[str, Any]] = {}
    for row_obj in candidates:
        row = dict(row_obj)
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            merged.setdefault(candidate_id, row)
    for row in _hinerv_waterfill_source_candidates(decoder_weight_waterfill_index):
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id and candidate_id not in merged:
            merged[candidate_id] = row
    rows = list(merged.values())
    rows.sort(
        key=lambda row: _hinerv_waterfill_candidate_sort_key(
            row,
            decoder_weight_waterfill_index,
        )
    )
    return rows[: max(1, int(limit))]


def _hinerv_waterfill_source_candidates(
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (family, _candidate_key), source_rows in (index or {}).items():
        if _family_key(str(family)) != "hi_nerv":
            continue
        for source in source_rows:
            candidate = source.get("_modelsize_candidate")
            if not isinstance(candidate, Mapping):
                continue
            candidate_id = str(candidate.get("candidate_id") or source.get("candidate_id") or "").strip()
            if not candidate_id or candidate_id in seen:
                continue
            authority_blockers = _candidate_authority_blockers(candidate)
            clean = _scrub_candidate_authority_flags(candidate)
            if authority_blockers:
                clean["_candidate_authority_blockers"] = authority_blockers
            clean.setdefault("candidate_id", candidate_id)
            clean.setdefault("family", "hi_nerv")
            clean["_candidate_source"] = "decoder_weight_waterfill_modelsize_candidate"
            clean["_candidate_source_waterfill_path"] = source.get("path")
            clean["_candidate_source_waterfill_receiver_proof_ready"] = bool(source.get("receiver_proof_ready"))
            clean["_candidate_source_waterfill_runner_admitted"] = bool(
                _decoder_weight_waterfill_runner_admitted(source)
            )
            seen.add(candidate_id)
            rows.append(clean)
    return rows


def _hinerv_waterfill_candidate_sort_key(
    row: Mapping[str, Any],
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> tuple[Any, ...]:
    waterfill = _decoder_weight_waterfill_for(
        candidate=row,
        family="hi_nerv",
        index=index,
    )
    has_waterfill = bool(waterfill)
    waterfill_admitted = bool(waterfill and _decoder_weight_waterfill_runner_admitted(waterfill))
    waterfill_receiver_ready = bool(waterfill and waterfill.get("receiver_proof_ready") is True)
    return (
        0 if waterfill_admitted else 1,
        0 if waterfill_receiver_ready else 1,
        0 if has_waterfill else 1,
        *_hinerv_long_training_candidate_sort_key(row),
    )


def _candidate_authority_blockers(candidate: Mapping[str, Any]) -> list[str]:
    return _dedupe(
        [f"selected_candidate_authority_flag_true:{path}" for path in _iter_truthy_authority_paths(candidate)]
    )


def _scrub_candidate_authority_flags(value: Any) -> Any:
    if isinstance(value, Mapping):
        scrubbed: dict[str, Any] = {}
        for key, raw in value.items():
            text_key = str(key)
            if text_key in _AUTHORITY_TRUE_KEYS:
                scrubbed[text_key] = False
            else:
                scrubbed[text_key] = _scrub_candidate_authority_flags(raw)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_candidate_authority_flags(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_candidate_authority_flags(item) for item in value]
    return value


def _iter_truthy_authority_paths(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, raw in value.items():
            text_key = str(key)
            path = f"{prefix}.{text_key}" if prefix else text_key
            if text_key in _AUTHORITY_TRUE_KEYS and _truthy_authority_value(raw):
                paths.append(path)
            paths.extend(_iter_truthy_authority_paths(raw, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, raw in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_iter_truthy_authority_paths(raw, prefix=path))
    return paths


def _truthy_authority_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _find_nested_finite_number(value: Any, key: str) -> float | None:
    if isinstance(value, Mapping):
        if key in value:
            try:
                numeric = float(value[key])
            except (TypeError, ValueError):
                numeric = math.nan
            if math.isfinite(numeric):
                return numeric
        for child in value.values():
            found = _find_nested_finite_number(child, key)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found = _find_nested_finite_number(child, key)
            if found is not None:
                return found
    return None


def _snerv_rate_plausible_for_long_training(candidate: Mapping[str, Any]) -> bool:
    if candidate.get("nominal_under_ceiling") is True:
        return True
    ceiling = int(candidate.get("hard_byte_ceiling") or 0)
    byte_headroom = _candidate_byte_headroom(candidate)
    if ceiling <= 0:
        return False
    # Permit near-miss long-training rows because real SNAR1 bytes can move
    # after QAT/coding; refuse rows whose nominal payload is orders over budget.
    return abs(byte_headroom) <= max(int(ceiling), 65_536)


def _snerv_hard_byte_ceiling_satisfied_for_long_training(
    candidate: Mapping[str, Any],
    *,
    lf_recode_admission_plan: Mapping[str, Any] | None,
) -> bool:
    if candidate.get("nominal_under_ceiling") is True:
        return True
    if not isinstance(lf_recode_admission_plan, Mapping):
        return False
    return bool(
        lf_recode_admission_plan.get("local_planner_admitted") is True
        and lf_recode_admission_plan.get("waterline_satisfied_after_selected_recode") is True
    )


def _snerv_long_training_candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    under = row.get("nominal_under_ceiling") is True
    ceiling = int(row.get("hard_byte_ceiling") or 0)
    headroom = _candidate_byte_headroom(row)
    total = int(row.get("nominal_total_payload_bytes") or 0)
    return (
        0 if under else 1,
        headroom if under else abs(headroom),
        ceiling,
        total,
        str(row.get("candidate_id") or ""),
    )


def _hinerv_long_training_candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    under = row.get("nominal_under_ceiling") is True
    ceiling = int(row.get("hard_byte_ceiling") or 0)
    headroom = _candidate_byte_headroom(row)
    total = int(row.get("nominal_total_payload_bytes") or 0)
    return (
        0 if under else 1,
        -_hinerv_official_control_score(row),
        headroom if under else abs(headroom),
        ceiling,
        total,
        str(row.get("candidate_id") or ""),
    )


def _candidate_byte_headroom(row: Mapping[str, Any]) -> int:
    if row.get("byte_headroom") is not None:
        return int(row.get("byte_headroom") or 0)
    ceiling = int(row.get("hard_byte_ceiling") or 0)
    total = int(row.get("nominal_total_payload_bytes") or 0)
    return int(ceiling - total)


def _modelsize_byte_cap_preflight(
    *,
    candidate: Mapping[str, Any],
    family: str,
    feedback_paths: Sequence[str],
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    ceiling = _first_present_int(candidate, ("hard_byte_ceiling",))
    nominal = _first_present_int(
        candidate,
        (
            "nominal_total_payload_bytes",
            "total_payload_bytes",
            "estimated_total_payload_bytes",
        ),
    )
    codec = _modelsize_byte_cap_codec(candidate)
    observations = _modelsize_byte_cap_feedback_observations(
        family=family,
        feedback_paths=feedback_paths,
    )
    matching = _modelsize_byte_cap_matching_observations(
        observations,
        candidate=candidate,
        codec=codec,
    )
    blockers: list[str] = []
    predicted: int | None = nominal
    prediction_rule = "nominal_payload_bytes_no_feedback"
    missing_matching_feedback_is_blocking = bool(
        feedback_paths and not matching and not _observations_are_demote_only_byte_feedback(observations)
    )
    if missing_matching_feedback_is_blocking:
        blockers.append(f"{family}_modelsize_byte_cap_feedback_observation_missing")
    if nominal is None:
        blockers.append(f"{family}_modelsize_byte_cap_candidate_nominal_missing")
    if ceiling is None:
        blockers.append(f"{family}_modelsize_byte_cap_candidate_ceiling_missing")
    if nominal is not None and matching:
        max_ratio = max(float(row["archive_to_nominal_ratio"]) for row in matching)
        max_delta = max(int(row["archive_minus_nominal_bytes"]) for row in matching)
        predicted = max(
            int(float(nominal) * max_ratio + 0.999999),
            int(nominal) + int(max_delta),
        )
        prediction_rule = "max_observed_archive_to_nominal_ratio_or_additive_overhead"
    required_nominal_values = [
        int(value)
        for row in matching
        for value in [_first_present_int(row, ("required_nominal_payload_bytes_max",))]
        if value is not None
    ]
    overrun_values = [
        int(value)
        for row in matching
        for value in [_first_present_int(row, ("calibrated_archive_overrun_bytes",))]
        if value is not None
    ]
    dynamic_domain_blockers = _dedupe(
        [
            str(blocker)
            for row in matching
            for blocker in row.get("receiver_dynamic_domain_blockers") or ()
            if str(blocker).strip()
        ]
    )
    dynamic_domain_failed = any(
        row.get("receiver_dynamic_domain_stable") is False for row in matching
    )
    fit_scale_failed = any(
        row.get("receiver_fit_scale_guard_passed") is False for row in matching
    )
    cache_quality_failed = any(
        row.get("receiver_cache_quality_gate_passed") is False for row in matching
    )
    headroom = None
    predicted_under = None
    if ceiling is not None and predicted is not None:
        headroom = int(ceiling) - int(predicted)
        predicted_under = headroom >= 0
        if not predicted_under and matching:
            blockers.append(f"{family}_modelsize_auto_calibrated_byte_cap_over_ceiling")
    if str(family) == "hi_nerv" and matching:
        if dynamic_domain_failed:
            blockers.append(
                "hi_nerv_modelsize_byte_cap_feedback_receiver_dynamic_domain_unstable"
            )
        if fit_scale_failed:
            blockers.append(
                "hi_nerv_modelsize_byte_cap_feedback_fit_scale_guard_failed"
            )
        if cache_quality_failed:
            blockers.append(
                "hi_nerv_modelsize_byte_cap_feedback_receiver_cache_quality_gate_failed"
            )
    return {
        "schema": "nerv_long_training_modelsize_byte_cap_preflight.v1",
        "family": str(family),
        "candidate_id": candidate_id or None,
        "codec": codec,
        "hard_byte_ceiling": ceiling,
        "nominal_payload_bytes": nominal,
        "predicted_archive_bytes": predicted,
        "predicted_headroom_bytes": headroom,
        "predicted_under_hard_byte_ceiling": predicted_under,
        "prediction_rule": prediction_rule,
        "feedback_path_count": len(tuple(feedback_paths)),
        "observation_count": len(observations),
        "matching_observation_count": len(matching),
        "matching_required_nominal_payload_bytes_max": (
            min(required_nominal_values) if required_nominal_values else None
        ),
        "matching_calibrated_archive_overrun_bytes_max": (max(overrun_values) if overrun_values else None),
        "matching_measurement_bypass_observed": any(
            bool(row.get("hard_byte_ceiling_measurement_bypass_enabled")) for row in matching
        ),
        "matching_receiver_dynamic_domain_failed": dynamic_domain_failed,
        "matching_receiver_fit_scale_guard_failed": fit_scale_failed,
        "matching_receiver_cache_quality_gate_failed": cache_quality_failed,
        "matching_receiver_cache_quality_gate_verdicts": sorted(
            _dedupe(
                [
                    str(row.get("receiver_cache_quality_gate_verdict") or "")
                    for row in matching
                    if str(row.get("receiver_cache_quality_gate_verdict") or "").strip()
                ]
            )
        ),
        "matching_receiver_dynamic_domain_blockers": dynamic_domain_blockers,
        "missing_matching_feedback_is_blocking": missing_matching_feedback_is_blocking,
        "matching_observations": matching,
        "scope": "budget_candidate_preflight_runner_revalidates_auto_selection",
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _observations_are_demote_only_byte_feedback(
    observations: Sequence[Mapping[str, Any]],
) -> bool:
    if not observations:
        return False
    return all(_observation_is_over_own_hard_byte_ceiling(row) for row in observations)


def _observation_is_over_own_hard_byte_ceiling(row: Mapping[str, Any]) -> bool:
    overrun = _first_present_int(row, ("calibrated_archive_overrun_bytes",))
    if overrun is not None and int(overrun) > 0:
        return True
    archive_bytes = _first_present_int(row, ("measured_archive_bytes",))
    candidate = row.get("modelsize_candidate")
    candidate_mapping = candidate if isinstance(candidate, Mapping) else {}
    ceiling = _first_present_int(row, ("hard_byte_ceiling",)) or _first_present_int(
        candidate_mapping,
        ("hard_byte_ceiling",),
    )
    return bool(archive_bytes is not None and ceiling is not None and int(archive_bytes) > int(ceiling))


def _modelsize_byte_cap_feedback_observations(
    *,
    family: str,
    feedback_paths: Sequence[str],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for raw_path in feedback_paths:
        path = Path(str(raw_path)).expanduser().resolve(strict=False)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in _iter_modelsize_byte_cap_feedback_rows(payload):
            row_family = str(row.get("family") or family)
            if not row.get("family") and row.get("schema") == "snerv_binary_profile.v1":
                row_family = "snerv"
            if row_family != str(family):
                continue
            receiver_closed = _modelsize_byte_cap_row_receiver_closed(
                row,
                source_path=path,
            )
            if not receiver_closed:
                continue
            measured = _first_present_int(
                row,
                (
                    "charged_archive_bytes",
                    "measured_archive_bytes",
                    "archive_bytes",
                    "archive_zip_bytes",
                    "candidate_archive_bytes",
                ),
            )
            candidate = row.get("modelsize_candidate")
            candidate_mapping = candidate if isinstance(candidate, Mapping) else {}
            if not candidate_mapping:
                candidate_mapping = _modelsize_byte_cap_startup_candidate(
                    row,
                    source_path=path,
                )
            if not _modelsize_byte_cap_scope_matches_candidate(
                row,
                candidate_mapping,
            ):
                continue
            nominal = _first_present_int(
                candidate_mapping,
                (
                    "nominal_total_payload_bytes",
                    "total_payload_bytes",
                    "estimated_total_payload_bytes",
                    "packet_bytes",
                ),
            )
            if nominal is None:
                nominal = _first_present_int(
                    row,
                    (
                        "nominal_total_payload_bytes",
                        "total_payload_bytes",
                        "estimated_total_payload_bytes",
                        "packet_bytes",
                    ),
                )
            if measured is None or nominal is None or measured <= 0 or nominal <= 0:
                continue
            payload_bytes = _first_present_int(
                row,
                ("measured_payload_bytes", "packet_bytes", "snar1_packet_bytes"),
            )
            archive_path = _modelsize_byte_cap_first_path(
                row,
                source_path=path,
                keys=("archive_path", "input_path", "candidate_archive_path"),
            )
            packet_path = _modelsize_byte_cap_packet_path(
                row,
                source_path=path,
                archive_path=archive_path,
            )
            receiver_proof_path = _existing_path(
                receiver_closed.get("proof_path") or row.get("receiver_proof_path")
            )
            report_path = _modelsize_byte_cap_report_path(
                row,
                source_path=path,
                archive_path=archive_path,
            )
            measured_pairs = _modelsize_byte_cap_measured_pairs(
                row,
                candidate_mapping,
            )
            receiver_dynamic_domain = _modelsize_byte_cap_receiver_dynamic_domain(row)
            official_state_slice = _modelsize_byte_cap_official_state_slice(
                row,
                source_path=path,
            )
            observations.append(
                {
                    "source_path": path.as_posix(),
                    "row_id": row.get("row_id") or row.get("candidate_id"),
                    "family": row_family,
                    "codec": _modelsize_byte_cap_codec(row),
                    "measured_archive_bytes": int(measured),
                    "measured_payload_bytes": (None if payload_bytes is None else int(payload_bytes)),
                    "measured_num_pairs": measured_pairs,
                    "nominal_payload_bytes": int(nominal),
                    "archive_minus_nominal_bytes": int(measured) - int(nominal),
                    "archive_to_nominal_ratio": float(measured) / float(nominal),
                    "calibrated_archive_overrun_bytes": (
                        _first_present_int(
                            row,
                            ("calibrated_archive_overrun_bytes",),
                        )
                    ),
                    "required_nominal_payload_bytes_max": (
                        _first_present_int(
                            row,
                            ("required_nominal_payload_bytes_max",),
                        )
                    ),
                    "hard_byte_ceiling_measurement_bypass_enabled": bool(
                        row.get("hard_byte_ceiling_measurement_bypass_enabled")
                    ),
                    "hard_byte_ceiling_checked_after_export": (
                        None
                        if row.get("hard_byte_ceiling_checked_after_export") is None
                        else bool(row.get("hard_byte_ceiling_checked_after_export"))
                    ),
                    "receiver_dynamic_domain_feedback": receiver_dynamic_domain,
                    "receiver_dynamic_domain_stable": receiver_dynamic_domain.get(
                        "dynamic_domain_stable"
                    ),
                    "receiver_fit_scale_guard_passed": receiver_dynamic_domain.get(
                        "receiver_fit_scale_guard_passed"
                    ),
                    "receiver_cache_quality_gate_passed": receiver_dynamic_domain.get(
                        "receiver_cache_quality_gate_passed"
                    ),
                    "receiver_cache_quality_gate_verdict": receiver_dynamic_domain.get(
                        "receiver_cache_quality_gate_verdict"
                    ),
                    "receiver_dynamic_domain_blockers": list(
                        receiver_dynamic_domain.get("blockers") or []
                    ),
                    "receiver_closed": True,
                    "receiver_closed_status": receiver_closed.get("status"),
                    "receiver_proof_path": (
                        receiver_proof_path.as_posix()
                        if receiver_proof_path is not None
                        else None
                    ),
                    "receiver_contract_satisfied": True,
                    "archive_path": archive_path.as_posix() if archive_path else None,
                    "archive_sha256": row.get("input_sha256") or row.get("archive_sha256"),
                    "packet_path": packet_path.as_posix() if packet_path else None,
                    "packet_sha256": row.get("snar1_packet_sha256") or row.get("packet_sha256"),
                    "artifact_report_path": (report_path.as_posix() if report_path else None),
                    "snerv_official_trained_checkpoint_state_dict_path": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_path"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_path": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_path"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_present": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_present"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_file_present": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_file_present"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_bytes": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_bytes"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_sha256": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_sha256"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_member_count": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_member_count"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_member_names": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_member_names"
                    ),
                    "snerv_official_trained_checkpoint_state_dict_slice_runner_arg": official_state_slice.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_runner_arg"
                    ),
                    "candidate_id": (row.get("candidate_id") or candidate_mapping.get("candidate_id")),
                    "source_bound_controls": (
                        _modelsize_byte_cap_candidate_match_controls(candidate_mapping) if candidate_mapping else {}
                    ),
                    "modelsize_candidate": (
                        _scrub_candidate_authority_flags(candidate_mapping) if candidate_mapping else None
                    ),
                }
            )
    return observations


def _modelsize_byte_cap_scope_matches_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    if not candidate:
        return True
    measured_pairs = _first_present_int(
        row,
        ("measured_num_pairs", "candidate_num_pairs", "num_pairs"),
    )
    metadata = row.get("snar1_metadata")
    if measured_pairs is None and isinstance(metadata, Mapping):
        measured_pairs = _first_present_int(metadata, ("n_pairs", "num_pairs"))
    candidate_pairs = _first_present_int(candidate, ("num_pairs",))
    return not (
        measured_pairs is not None and candidate_pairs is not None and int(measured_pairs) != int(candidate_pairs)
    )


def _modelsize_byte_cap_measured_pairs(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> int | None:
    measured_pairs = _first_present_int(
        row,
        ("measured_num_pairs", "candidate_num_pairs", "num_pairs"),
    )
    metadata = row.get("snar1_metadata")
    if measured_pairs is None and isinstance(metadata, Mapping):
        measured_pairs = _first_present_int(metadata, ("n_pairs", "num_pairs"))
    if measured_pairs is None:
        measured_pairs = _first_present_int(candidate, ("num_pairs",))
    return measured_pairs


def _modelsize_byte_cap_receiver_dynamic_domain(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    raw = row.get("receiver_dynamic_domain_feedback")
    out = dict(raw) if isinstance(raw, Mapping) else {}
    out.setdefault("schema", "hinerv_receiver_dynamic_domain_feedback.v1")
    fit_passed = _maybe_bool(
        out.get("receiver_fit_scale_guard_passed")
        if out.get("receiver_fit_scale_guard_passed") is not None
        else row.get("receiver_fit_scale_guard_passed")
    )
    cache_passed = _maybe_bool(
        out.get("receiver_cache_quality_gate_passed")
        if out.get("receiver_cache_quality_gate_passed") is not None
        else row.get("receiver_cache_quality_gate_passed")
    )
    stable = _maybe_bool(
        out.get("dynamic_domain_stable")
        if out.get("dynamic_domain_stable") is not None
        else row.get("receiver_dynamic_domain_stable")
    )
    if stable is None:
        if fit_passed is False or cache_passed is False:
            stable = False
        elif fit_passed is True and cache_passed is True:
            stable = True
    blockers = _dedupe(
        [
            *(
                str(value)
                for value in out.get("blockers") or ()
                if str(value).strip()
            ),
            *(
                str(value)
                for value in row.get("receiver_dynamic_domain_blockers") or ()
                if str(value).strip()
            ),
            *(
                str(value)
                for value in out.get("receiver_fit_scale_guard_blockers") or ()
                if str(value).strip()
            ),
            *(
                str(value)
                for value in out.get("receiver_cache_quality_gate_blockers") or ()
                if str(value).strip()
                and str(value).strip()
                not in {
                    "mlx_cache_quality_gate_is_false_authority",
                    "hi_nerv_receiver_cache_quality_is_false_authority",
                }
            ),
        ]
    )
    out["dynamic_domain_stable"] = stable
    out["receiver_fit_scale_guard_passed"] = fit_passed
    out["receiver_cache_quality_gate_passed"] = cache_passed
    out["receiver_cache_quality_gate_verdict"] = (
        out.get("receiver_cache_quality_gate_verdict")
        or row.get("receiver_cache_quality_gate_verdict")
    )
    out["blockers"] = blockers
    out.update(FALSE_AUTHORITY)
    return out


def _modelsize_byte_cap_official_state_slice(
    row: Mapping[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    sources: list[Mapping[str, Any]] = [row]
    binding = row.get("official_checkpoint_export_binding")
    if isinstance(binding, Mapping):
        sources.append(binding)
    for source in sources:
        raw_path = source.get("official_trained_checkpoint_state_dict_slice_path")
        if not raw_path:
            continue
        raw = Path(str(raw_path)).expanduser()
        path = (
            raw.resolve(strict=False)
            if raw.is_absolute()
            else (source_path.parent / raw).resolve(strict=False)
        )
        out: dict[str, Any] = {
            "snerv_official_trained_checkpoint_state_dict_slice_path": path.as_posix(),
            "snerv_official_trained_checkpoint_state_dict_slice_present": bool(
                source.get("official_trained_checkpoint_state_dict_slice_present")
            ),
            "snerv_official_trained_checkpoint_state_dict_slice_file_present": path.is_file(),
            "snerv_official_trained_checkpoint_state_dict_slice_runner_arg": source.get(
                "official_trained_checkpoint_state_dict_slice_runner_arg"
            ),
        }
        if path.is_file():
            out["snerv_official_trained_checkpoint_state_dict_path"] = path.as_posix()
        for source_key, out_key in (
            (
                "official_trained_checkpoint_state_dict_slice_bytes",
                "snerv_official_trained_checkpoint_state_dict_slice_bytes",
            ),
            (
                "official_trained_checkpoint_state_dict_slice_sha256",
                "snerv_official_trained_checkpoint_state_dict_slice_sha256",
            ),
            (
                "official_trained_checkpoint_state_dict_slice_member_count",
                "snerv_official_trained_checkpoint_state_dict_slice_member_count",
            ),
        ):
            if source.get(source_key) is not None:
                out[out_key] = source.get(source_key)
        names = source.get("official_trained_checkpoint_state_dict_slice_member_names")
        if isinstance(names, Sequence) and not isinstance(
            names,
            (str, bytes, bytearray),
        ):
            out["snerv_official_trained_checkpoint_state_dict_slice_member_names"] = list(
                names
            )
        return out
    return {}


def _modelsize_byte_cap_first_path(
    row: Mapping[str, Any],
    *,
    source_path: Path | None,
    keys: Sequence[str],
) -> Path | None:
    for key in keys:
        value = row.get(key)
        if value:
            return Path(str(value)).expanduser().resolve(strict=False)
    return source_path


def _modelsize_byte_cap_packet_path(
    row: Mapping[str, Any],
    *,
    source_path: Path | None,
    archive_path: Path | None,
) -> Path | None:
    explicit = _modelsize_byte_cap_first_path(
        row,
        source_path=None,
        keys=("packet_path", "snar1_packet_path"),
    )
    if explicit and explicit.is_file():
        return explicit
    candidates: list[Path] = []
    for artifact_path in _modelsize_byte_cap_row_artifact_paths(
        row,
        source_path=source_path,
    ):
        for parent in _self_and_parents(artifact_path):
            candidates.append(parent / "snerv_mlx_native_packet.snar")
    if archive_path is not None:
        candidates.extend(
            [
                archive_path.parent.parent / "snerv_mlx_native_packet.snar",
                archive_path.parent / "snerv_mlx_native_packet.snar",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(strict=False)
    return explicit


def _modelsize_byte_cap_report_path(
    row: Mapping[str, Any],
    *,
    source_path: Path | None,
    archive_path: Path | None,
) -> Path | None:
    explicit = _modelsize_byte_cap_first_path(
        row,
        source_path=None,
        keys=("artifact_report_path", "report_path"),
    )
    if explicit and explicit.is_file():
        return explicit
    if source_path is not None and Path(source_path).is_file():
        return Path(source_path).resolve(strict=False)
    if archive_path is not None:
        for candidate in (
            archive_path.parent / "archive_bound_candidate_adapter_package.json",
            archive_path.parent.parent / "snerv_score_aware_long_training" / "snerv_score_aware_long_training.json",
        ):
            if candidate.is_file():
                return candidate.resolve(strict=False)
    return explicit


def _modelsize_byte_cap_startup_candidate(
    row: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    path_value = row.get("startup_json_path")
    startup_paths: list[Path] = []
    if path_value:
        startup_paths.append(Path(str(path_value)).expanduser().resolve(strict=False))
    for artifact_path in _modelsize_byte_cap_row_artifact_paths(
        row,
        source_path=source_path,
    ):
        startup_paths.extend(
            candidate
            for candidate in _candidate_startup_paths_from_artifact(artifact_path)
            if candidate not in startup_paths
        )
    for startup_path in startup_paths:
        try:
            startup = json.loads(startup_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate = startup.get("modelsize_candidate")
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _iter_modelsize_byte_cap_feedback_rows(node: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if (
                _first_present_int(
                    value,
                    (
                        "charged_archive_bytes",
                        "measured_archive_bytes",
                        "archive_bytes",
                        "archive_zip_bytes",
                        "candidate_archive_bytes",
                    ),
                )
                is not None
            ):
                rows.append(value)
            for key in (
                "row",
                "rows",
                "archive_rows",
                "ladder_rows",
                "receiver_closed_rows",
                "points",
                "selected_candidates",
                "family_rows",
                "trained_archive_byte_oracle",
            ):
                child = value.get(key)
                if child is not None:
                    visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                visit(item)

    visit(node)
    return rows


def _modelsize_byte_cap_row_receiver_closed(
    row: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    inline_runtime_ready = _truthy_any(
        row,
        (
            "receiver_closed",
            "receiver_proof_ready",
            "receiver_proof_passed",
            "runtime_consumption_proof_ready",
            "receiver_archive_replay_verified",
            "byte_closed_receiver_proof",
        ),
    )
    contract_ok = row.get("receiver_contract_satisfied") is not False
    if inline_runtime_ready and contract_ok:
        return {"status": "inline_receiver_closed"}
    for proof_path in _modelsize_byte_cap_receiver_proof_paths(
        row,
        source_path=source_path,
    ):
        try:
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runtime_ready = bool(
            proof.get("runtime_consumption_proof_ready")
            or proof.get("runtime_consumption_proof_passed")
            or proof.get("receiver_proof_ready")
            or proof.get("receiver_proof_passed")
        )
        if not (runtime_ready and proof.get("receiver_contract_satisfied") is True):
            continue
        proof_archive_sha = str(proof.get("archive_sha256") or "").strip().lower()
        row_archive_sha = str(row.get("input_sha256") or row.get("archive_sha256") or "").strip().lower()
        if proof_archive_sha and row_archive_sha and proof_archive_sha != row_archive_sha:
            continue
        proof_archive_bytes = _first_present_int(
            proof,
            ("archive_bytes", "measured_archive_bytes", "archive_zip_bytes"),
        )
        row_archive_bytes = _first_present_int(
            row,
            (
                "charged_archive_bytes",
                "measured_archive_bytes",
                "archive_bytes",
                "archive_zip_bytes",
                "candidate_archive_bytes",
            ),
        )
        if (
            proof_archive_bytes is not None
            and row_archive_bytes is not None
            and int(proof_archive_bytes) != int(row_archive_bytes)
        ):
            continue
        return {
            "status": "associated_receiver_proof",
            "proof_path": proof_path.as_posix(),
        }
    return {}


def _modelsize_byte_cap_receiver_proof_paths(
    row: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for artifact_path in _modelsize_byte_cap_row_artifact_paths(
        row,
        source_path=source_path,
    ):
        for parent in _self_and_parents(artifact_path):
            for name in (
                "snerv_inverse_steg_receiver_proof.json",
                "hi_nerv_mlx_receiver_proof.json",
            ):
                direct = parent / name
                nested = parent / "receiver_proof" / name
                for candidate in (direct, nested):
                    if candidate.is_file() and candidate not in paths:
                        paths.append(candidate)
    return paths


def _modelsize_byte_cap_row_artifact_paths(
    row: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> list[Path]:
    out: list[Path] = []
    if source_path is not None:
        out.append(Path(source_path).expanduser().resolve(strict=False))
    for key in (
        "input_path",
        "archive_path",
        "source_archive_path",
        "candidate_archive_path",
        "proof_path",
        "receiver_proof_path",
        "report_path",
    ):
        value = row.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser().resolve(strict=False)
        if candidate not in out:
            out.append(candidate)
    return out


def _candidate_startup_paths_from_artifact(path: Path) -> list[Path]:
    out: list[Path] = []
    for parent in _self_and_parents(path):
        candidate = parent / "compact_renderer_mlx_spine_runner_startup.json"
        if candidate.is_file() and candidate not in out:
            out.append(candidate)
    return out


def _self_and_parents(path: Path) -> list[Path]:
    base = path if path.is_dir() else path.parent
    return [base, *list(base.parents)]


def _modelsize_byte_cap_matching_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    candidate: Mapping[str, Any],
    codec: str | None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in observations]
    candidate_id = str(candidate.get("candidate_id") or "")
    if candidate_id:
        exact_id = [row for row in rows if str(row.get("candidate_id") or "") == candidate_id]
        if exact_id:
            return exact_id
    target_controls = _modelsize_byte_cap_candidate_match_controls(candidate)
    exact_controls = [
        row
        for row in rows
        if _modelsize_byte_cap_controls_match(
            row.get("source_bound_controls"),
            target_controls,
        )
    ]
    if exact_controls:
        return exact_controls
    scoped_rows = [
        row
        for row in rows
        if row.get("candidate_id")
        or _modelsize_byte_cap_controls_are_candidate_scoped(row.get("source_bound_controls"))
    ]
    if scoped_rows:
        return []
    if codec:
        exact = [row for row in rows if row.get("codec") == codec]
        if exact:
            return exact
    return rows


def _modelsize_byte_cap_candidate_match_controls(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "family",
        "num_pairs",
        "hard_byte_ceiling",
        "decoder_codec",
        "decoder_payload_codec",
        "wavelet",
        "levels",
        "bits_per_coeff",
        "step_map_bits_per_coeff",
        "snerv_model_size_adapter",
        "fc_dim",
        "emb_size",
        "patch_radius",
        "mfu_scales",
        "hfr_gain",
        "temporal_context",
        "temporal_mode",
        "official_skip_high_mode",
        "use_hierarchical_feature_grid",
        "use_convnext_blocks",
        "local_grid_levels",
        "local_grid_channels",
        "convnext_mlp_ratio",
        "convnext_kernel_size",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if key not in candidate:
            continue
        value = candidate[key]
        if isinstance(value, tuple):
            value = list(value)
        out[key] = value
    return out


def _modelsize_byte_cap_controls_match(
    source_controls: Any,
    target_controls: Mapping[str, Any],
) -> bool:
    if not isinstance(source_controls, Mapping) or not source_controls:
        return False
    if not target_controls:
        return False
    for key, source_value in source_controls.items():
        if key not in target_controls:
            return False
        target_value = target_controls[key]
        if isinstance(source_value, tuple):
            source_value = list(source_value)
        if isinstance(target_value, tuple):
            target_value = list(target_value)
        if source_value != target_value:
            return False
    return True


def _modelsize_byte_cap_controls_are_candidate_scoped(source_controls: Any) -> bool:
    if not isinstance(source_controls, Mapping) or not source_controls:
        return False
    generic_keys = {
        "family",
        "decoder_codec",
        "decoder_payload_codec",
        "hard_byte_ceiling",
    }
    return any(str(key) not in generic_keys for key in source_controls)


def _modelsize_byte_cap_codec(row: Mapping[str, Any]) -> str | None:
    for key in (
        "decoder_codec",
        "decoder_payload_codec",
        "codec",
        "latent_codec",
    ):
        value = row.get(key)
        if value:
            return str(value)
    candidate = row.get("modelsize_candidate")
    if isinstance(candidate, Mapping):
        return _modelsize_byte_cap_codec(candidate)
    return None


def _first_present_int(row: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _maybe_bool(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    return None


def _truthy_any(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(row.get(key) is True for key in keys)


def _snerv_source_bound_control_blockers(candidate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    candidate_id = str(candidate.get("candidate_id") or "")
    for key in (
        "num_pairs",
        "wavelet",
        "levels",
        "bits_per_coeff",
        "step_map_bits_per_coeff",
        "decoder_payload_codec",
        "snerv_model_size_adapter",
        "fc_dim",
        "emb_size",
        "patch_radius",
        "mfu_scales",
        "hfr_gain",
        "temporal_context",
        "temporal_mode",
        "hard_byte_ceiling",
    ):
        if key not in candidate:
            blockers.append(f"snerv_source_bound_control_missing:{key}")
    if not blockers:
        try:
            expected_candidate_id = _snerv_expected_candidate_id_from_controls(candidate)
        except (KeyError, NervModelSizeBudgetError, TypeError, ValueError):
            blockers.append("snerv_candidate_id_source_bound_controls_unparseable")
        else:
            if candidate_id != expected_candidate_id:
                blockers.append("snerv_candidate_id_source_bound_controls_mismatch")
    return blockers


def _snerv_source_bound_controls(candidate: Mapping[str, Any]) -> dict[str, Any]:
    expected_candidate_id = None
    candidate_id_matches_source_controls = False
    try:
        expected_candidate_id = _snerv_expected_candidate_id_from_controls(candidate)
        candidate_id_matches_source_controls = str(candidate.get("candidate_id") or "") == expected_candidate_id
    except (KeyError, NervModelSizeBudgetError, TypeError, ValueError):
        pass
    return {
        "schema": "snerv_source_bound_capacity_controls.v1",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "expected_candidate_id": expected_candidate_id,
        "candidate_id_matches_source_controls": candidate_id_matches_source_controls,
        "wavelet": candidate.get("wavelet"),
        "levels": candidate.get("levels"),
        "bits_per_coeff": candidate.get("bits_per_coeff"),
        "step_map_bits_per_coeff": candidate.get("step_map_bits_per_coeff"),
        "decoder_payload_codec": candidate.get("decoder_payload_codec"),
        "snerv_model_size_adapter": candidate.get("snerv_model_size_adapter"),
        "fc_dim": candidate.get("fc_dim"),
        "emb_size": candidate.get("emb_size"),
        "patch_radius": candidate.get("patch_radius"),
        "mfu_scales": list(candidate.get("mfu_scales") or ()),
        "hfr_gain": candidate.get("hfr_gain"),
        "temporal_context": candidate.get("temporal_context"),
        "temporal_mode": candidate.get("temporal_mode"),
        "official_skip_high_mode": candidate.get("official_skip_high_mode", "full"),
        "decoder_feature_count": candidate.get("decoder_feature_count"),
        **FALSE_AUTHORITY,
    }


def _snerv_lf_payload_recode_admission_for_candidate(
    *,
    sources: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    candidate_id: str,
    full_video_coverage: bool,
) -> dict[str, Any] | None:
    if not sources:
        return None
    plan = build_snerv_lf_payload_recode_admission_plan(
        sources,
        hard_byte_ceiling=int(candidate.get("hard_byte_ceiling") or 0),
        candidate_id=candidate_id,
        full_video_coverage=bool(full_video_coverage),
    )
    return plan


def _snerv_lf_recode_selected_mode(
    plan: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(plan, Mapping):
        return None
    selected = plan.get("selected_row")
    if not isinstance(selected, Mapping):
        return None
    if selected.get("local_planner_admitted") is not True:
        return None
    if _snerv_lf_payload_recode_campaign_blockers(plan):
        return None
    mode = str(selected.get("mode") or "").strip()
    return mode or None


def _snerv_lf_payload_recode_campaign_blockers(
    plan: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(plan, Mapping):
        return []
    selected = plan.get("selected_row")
    if not isinstance(selected, Mapping):
        return ["snerv_lf_payload_recode_no_receiver_proven_byte_saving_mode"]
    blockers: list[str] = [str(blocker) for blocker in plan.get("blockers") or () if str(blocker)]
    over_waterline = selected.get("post_recode_over_waterline_bytes")
    if over_waterline is not None and int(over_waterline) > 0:
        blockers.append("snerv_lf_payload_recode_still_over_hard_byte_ceiling")
    blockers.extend(str(blocker) for blocker in selected.get("local_admission_blockers") or () if str(blocker))
    blockers.extend(str(blocker) for blocker in selected.get("promotion_blockers") or () if str(blocker))
    return _dedupe(blockers)


def _snerv_expected_candidate_id_from_controls(candidate: Mapping[str, Any]) -> str:
    return snerv_modelsize_candidate_id_from_controls(
        num_pairs=int(candidate["num_pairs"]),
        wavelet=str(candidate["wavelet"]),
        levels=int(candidate["levels"]),
        bits_per_coeff=float(candidate["bits_per_coeff"]),
        step_map_bits_per_coeff=float(candidate["step_map_bits_per_coeff"]),
        fc_dim=int(candidate["fc_dim"]),
        emb_size=int(candidate["emb_size"]),
        patch_radius=int(candidate["patch_radius"]),
        mfu_scales=tuple(int(value) for value in candidate["mfu_scales"]),
        hfr_gain=float(candidate["hfr_gain"]),
        temporal_context=int(candidate["temporal_context"]),
        temporal_mode=str(candidate["temporal_mode"]),
        official_skip_high_mode=str(candidate.get("official_skip_high_mode", "full")),
        snerv_model_size_adapter=str(candidate["snerv_model_size_adapter"]),
        decoder_payload_codec=str(candidate["decoder_payload_codec"]),
        hard_byte_ceiling=int(candidate["hard_byte_ceiling"]),
        official_modelsize_mparams=(
            float(candidate["modelsize_mparams"])
            if candidate.get("capacity_source") == "official_snerv_modelsize"
            and candidate.get("modelsize_mparams") is not None
            else None
        ),
    )


def _snerv_optimizer_control_blocker() -> dict[str, Any]:
    return {
        "schema": OPTIMIZER_CONTROL_SCHEMA,
        "optimizer_kind": None,
        "backend": ("mlx_target_hydration_numpy_closed_form_decoder_fit_plus_scorer_loop_qat"),
        "native_mlx_on_apple_silicon": True,
        "apple_specific_algorithm_claim": False,
        "first_pass_priority": False,
        "borrowed_from_pr95": False,
        "original_pact_contest_adaptation": False,
        "pact_muon_adamw_default_inherited": False,
        "not_applicable_reason": (
            "Current SNeRV rows materialize source-bound closed-form/native "
            "packets plus optional scorer-loop QAT; they do not yet expose a "
            "learned optimizer-controlled decoder-weight training loop."
        ),
        "blocked_until": (
            "snerv_learned_nonlinear_or_shared_mlx_scoreaware_decoder_training_loop_bound_to_receiver_grammar"
        ),
        "required_next_implementation": [
            "source_faithful_snerv_mfu_hfr_tub_forward_parity",
            "receiver_visible_mixed_precision_or_decoder_delta_grammar",
            "pose_guarded_scorer_loop_decoder_weight_qat",
            "full600_byte_closed_receiver_archive_replay",
        ],
        **FALSE_AUTHORITY,
    }


def _int_csv(values: Any) -> str:
    if isinstance(values, str):
        return values
    if not isinstance(values, Sequence):
        return str(int(values))
    return ",".join(str(int(value)) for value in values)


def _normalize_optimizer_kind(value: Any) -> str:
    text = str(value).strip().lower()
    return _OPTIMIZER_KIND_ALIASES.get(text, text)


def _is_timing_smoke_optimizer_kind(optimizer_kind: str) -> bool:
    return _normalize_optimizer_kind(optimizer_kind) in TIMING_SMOKE_OPTIMIZER_KINDS


def _optimizer_launch_blockers(optimizer_kind: str) -> tuple[str, ...]:
    return _TIMING_SMOKE_OPTIMIZER_LAUNCH_BLOCKERS.get(
        _normalize_optimizer_kind(optimizer_kind),
        (),
    )


def _optimizer_tuple(values: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    supported = set(SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS) | set(TIMING_SMOKE_OPTIMIZER_KINDS)
    for value in values:
        text = _normalize_optimizer_kind(value)
        if not text:
            continue
        if text not in supported:
            raise NervLongTrainingCampaignPlanError(f"unsupported optimizer kind: {value!r}")
        if text not in out:
            out.append(text)
    if not out:
        raise NervLongTrainingCampaignPlanError("at least one optimizer is required")
    return tuple(out)


def _optimizer_priority(optimizer_kind: str) -> int:
    kind = _normalize_optimizer_kind(optimizer_kind)
    if kind == "pact_muon_adamw":
        return 9
    if _is_timing_smoke_optimizer_kind(kind):
        return 11
    return 10 if kind in FIRST_PASS_OPTIMIZER_KINDS else 11


def _optimizer_control(optimizer_kind: str) -> dict[str, Any]:
    kind = _normalize_optimizer_kind(optimizer_kind)
    if _is_timing_smoke_optimizer_kind(kind):
        return {
            "schema": OPTIMIZER_CONTROL_SCHEMA,
            "optimizer_kind": kind,
            "classification": "runnable_false_authority_timing_smoke_candidate",
            "backend": "tac.substrates._shared.mlx_score_aware.adapter.AuroraLikeMlxOptimizer",
            "source_ids": [
                "tilde-research/aurora-release@7303d8cb9999d735cb12c921f3651f04bf362524",
                "blog.tilderesearch.com/blog/aurora",
                "Yifei-Zuo/modded-nanogpt-plx@7698686df679a7990cf91571df64042c30168d5c",
            ],
            "native_mlx_on_apple_silicon": True,
            "native_mlx_optimizer_object": True,
            "pact_partitioned_muon_adamw": False,
            "apple_specific_algorithm_claim": False,
            "first_pass_priority": False,
            "borrowed_from_pr95": False,
            "original_pact_contest_adaptation": False,
            "implementation_status": ("mlx_score_aware_optimizer_contract_landed_timing_smoke_required"),
            "launch_blockers": list(_optimizer_launch_blockers(kind)),
            "authority_blockers": ["aurora_not_pr95_source_authority"],
            "blocked_until": "aurora_like_local_timing_convergence_smoke",
            "required_next_implementation": [
                "run Aurora-like against pact_muon_adamw and muon on a tiny PR95/HiNeRV smoke",
                "record seconds_per_epoch plus SegNet/PoseNet/loss telemetry with false-authority flags",
                "keep score_claim/promotion/rank authority false until byte-closed replay evidence exists",
            ],
            "provenance_note": (
                "Aurora-like is now a runnable Pact-local MLX optimizer port "
                "for timing and convergence smokes, not PR95 source authority "
                "or score/promotion authority."
            ),
            "default_hinerv_optimizer_policy": "native_optimizer",
            "pr95_curriculum_optimizer_swallow_guard": True,
            **FALSE_AUTHORITY,
        }
    if kind not in SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS:
        raise NervLongTrainingCampaignPlanError(f"unsupported optimizer kind: {optimizer_kind!r}")
    is_pact_default = kind == "pact_muon_adamw"
    return {
        "schema": OPTIMIZER_CONTROL_SCHEMA,
        "optimizer_kind": kind,
        "backend": ("tac.local_acceleration.pr95_hnerv_mlx" if is_pact_default else "mlx.optimizers"),
        "native_mlx_on_apple_silicon": True,
        "native_mlx_optimizer_object": not is_pact_default,
        "pact_partitioned_muon_adamw": is_pact_default,
        "apple_specific_algorithm_claim": False,
        "first_pass_priority": kind in FIRST_PASS_OPTIMIZER_KINDS,
        "borrowed_from_pr95": is_pact_default,
        "original_pact_contest_adaptation": is_pact_default,
        "provenance_note": (
            "Pact default borrows PR95's Muon-vs-AdamW partition rule and "
            "the existing MLX Newton-Schulz step, then applies it to the "
            "score-aware NeRV train loop with false-authority telemetry."
            if is_pact_default
            else "Direct native MLX optimizer control row."
        ),
        "default_hinerv_optimizer_policy": _hinerv_optimizer_policy_for_kind(kind),
        "pr95_curriculum_optimizer_swallow_guard": (kind not in {"adamw", "pact_muon_adamw"}),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _hinerv_optimizer_policy_for_kind(optimizer_kind: str) -> str:
    """Return the runner policy that makes this row's optimizer semantics real."""

    kind = _normalize_optimizer_kind(optimizer_kind)
    return "pr95_curriculum" if kind in {"adamw", "pact_muon_adamw"} else "native_optimizer"


def _hinerv_optimizer_policy_control(
    *,
    optimizer_kind: str,
    optimizer_policy: str,
) -> dict[str, Any]:
    kind = _normalize_optimizer_kind(optimizer_kind)
    policy = str(optimizer_policy).strip().lower()
    timing_smoke_only = _is_timing_smoke_optimizer_kind(kind)
    return {
        "schema": HINERV_OPTIMIZER_POLICY_SCHEMA,
        "optimizer_kind": kind,
        "requested_policy": policy,
        "classification": (
            "runnable_false_authority_timing_smoke_candidate" if timing_smoke_only else "runner_optimizer_policy"
        ),
        "is_plan_only_optimizer_control": False,
        "is_timing_smoke_optimizer_control": timing_smoke_only,
        "pr95_faithful_curriculum_expected": (policy == "pr95_curriculum" and not timing_smoke_only),
        "native_mlx_optimizer_expected": policy == "native_optimizer",
        "effective_optimizer_label": ("pr95_8stage_muon_adamw" if policy == "pr95_curriculum" else kind),
        "runner_policy_if_implemented": policy,
        "launch_blockers": list(_optimizer_launch_blockers(kind)),
        "why": (
            (
                "Aurora-like rows are runnable native MLX timing-smoke "
                "candidates, but full campaign launch stays blocked until the "
                "local convergence smoke exists."
            )
            if timing_smoke_only
            else (
                "adamw and pact_muon_adamw own PR95-faithful 8-stage "
                "Muon+AdamW control rows; other rows must run as native MLX "
                "optimizers so optimizer diversity is measured rather than "
                "swallowed by the curriculum"
            )
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _normalize_hinerv_distortion_birth_evidence_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(source)
    family = str(row.get("family") or row.get("execute_family") or "hi_nerv")
    candidate_id = _first_non_auto_candidate_id(
        row.get("candidate_id"),
        row.get("modelsize_candidate_id"),
        row.get("budget_candidate_id"),
    )
    source_path = row.get("_source_path") or row.get("report_path") or row.get("path")
    if row.get("schema") == HINERV_DISTORTION_BIRTH_RATE_GATE_SCHEMA:
        return row
    if row.get("schema") == DISTORTION_BIRTH_RATE_EVIDENCE_SCHEMA:
        evidence = dict(row)
    else:
        raw_metrics = row.get("metrics")
        raw_metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        metrics = {
            key: _find_nested_finite_number(row, key)
            for key in (
                "receiver_quantum_attempt_count",
                "hard_birth_argmax_progress_accepted_step_count",
                "max_candidate_segnet_worst_debt_reduction",
                "max_candidate_segnet_min_ratio_increase",
                "max_candidate_segnet_total_debt_spill_given_worst_improvement",
                "max_accepted_frame1_receiver_uint8_changed_count",
                "max_accepted_frame1_receiver_uint8_delta_abs",
                "max_candidate_pose_exact_delta",
                "max_candidate_segnet_target_min_ratio_increase_authoritative",
                "hard_birth_target_hard_won_count",
                "hard_birth_net_target_support_delta",
            )
        }
        for key in (
            "min_ratio_increase_by_source",
            "target_min_region_ratio_delta_by_source",
            "min_ratio_increase_authority_source",
            "target_support_by_source",
        ):
            if key in raw_metrics:
                metrics[key] = raw_metrics[key]
            elif key in row:
                metrics[key] = row[key]
        target_support_by_source = (
            row.get("target_support_by_source")
            if isinstance(row.get("target_support_by_source"), Mapping)
            else raw_metrics.get("target_support_by_source")
        )
        evidence = build_distortion_birth_before_rate_pressure_evidence(
            {
                "report_loaded": True,
                "metrics": metrics,
                "target_support_by_source": (
                    dict(target_support_by_source)
                    if isinstance(target_support_by_source, Mapping)
                    else {}
                ),
            }
        )
    blockers = [str(blocker) for blocker in evidence.get("blockers") or []]
    if family != "hi_nerv":
        blockers.append("hinerv_distortion_birth_evidence_family_mismatch")
    for path in _iter_truthy_authority_paths(row):
        blockers.append(f"hinerv_distortion_birth_authority_flag_true:{path}")
    passed = bool(
        evidence.get("distortion_birth_before_rate_pressure_satisfied")
    ) and not blockers
    return {
        "schema": HINERV_DISTORTION_BIRTH_RATE_GATE_SCHEMA,
        "family": "hi_nerv",
        "candidate_id": candidate_id or None,
        "candidate_binding": (
            "candidate_bound" if candidate_id else "global_hinerv_mechanism_gate"
        ),
        "source_path": None if source_path is None else str(source_path),
        "source_schema": row.get("schema"),
        "attached": True,
        "passed": passed,
        "distortion_birth_before_rate_pressure_satisfied": passed,
        "evidence": evidence,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _hinerv_distortion_birth_before_rate_pressure_gate(
    *,
    candidate: Mapping[str, Any],
    evidence_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    raw_sources: list[Mapping[str, Any]] = [
        raw for raw in evidence_sources if isinstance(raw, Mapping)
    ]
    for key in (
        "hinerv_distortion_birth_before_rate_pressure_gate",
        "distortion_birth_before_rate_pressure_evidence",
        "hinerv_distortion_birth_evidence",
    ):
        embedded = candidate.get(key)
        if isinstance(embedded, Mapping):
            raw_sources.append(embedded)
    for raw in raw_sources:
        gate = _normalize_hinerv_distortion_birth_evidence_source(raw)
        gate_candidate_id = str(gate.get("candidate_id") or "")
        if gate_candidate_id and gate_candidate_id != candidate_id:
            skipped.append(
                {
                    "candidate_id": gate_candidate_id,
                    "source_path": gate.get("source_path"),
                    "reason": "candidate_id_mismatch",
                }
            )
            continue
        candidates.append(gate)
    if not candidates:
        blockers = [
            "hinerv_distortion_birth_before_rate_pressure_missing_or_blocked",
            "distortion_birth_smoke_report_missing",
        ]
        return {
            "schema": HINERV_DISTORTION_BIRTH_RATE_GATE_SCHEMA,
            "family": "hi_nerv",
            "candidate_id": candidate_id or None,
            "required": True,
            "attached": False,
            "passed": False,
            "distortion_birth_before_rate_pressure_satisfied": False,
            "source_count": len(raw_sources),
            "skipped_sources": skipped,
            "rate_pressure_controls_blocked_until_satisfied": [
                "coder_qat",
                "section_byte_duals",
                "c1a_entropy_pressure",
                "byte_compiler_selection",
                "muon_late_polish",
            ],
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
    selected = next((gate for gate in candidates if gate.get("passed") is True), candidates[0])
    selected_blockers = [str(blocker) for blocker in selected.get("blockers") or []]
    blockers = (
        []
        if selected.get("passed") is True
        else [
            "hinerv_distortion_birth_before_rate_pressure_missing_or_blocked",
            *selected_blockers,
        ]
    )
    blockers = _dedupe(blockers)
    return {
        "schema": HINERV_DISTORTION_BIRTH_RATE_GATE_SCHEMA,
        "family": "hi_nerv",
        "candidate_id": candidate_id or None,
        "required": True,
        "attached": True,
        "passed": bool(selected.get("passed") is True and not blockers),
        "distortion_birth_before_rate_pressure_satisfied": bool(
            selected.get("passed") is True and not blockers
        ),
        "source_count": len(raw_sources),
        "matched_source_count": len(candidates),
        "selected_source_path": selected.get("source_path"),
        "selected_candidate_binding": selected.get("candidate_binding"),
        "selected_evidence": selected.get("evidence"),
        "skipped_sources": skipped,
        "rate_pressure_controls_blocked_until_satisfied": [
            "coder_qat",
            "section_byte_duals",
            "c1a_entropy_pressure",
            "byte_compiler_selection",
            "muon_late_polish",
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _candidate_feedback_index(
    sources: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        row = _normalize_candidate_feedback_source(source)
        family = _feedback_family(row)
        if not family:
            continue
        for candidate_key in _candidate_index_keys(row):
            index.setdefault((family, candidate_key), []).append(row)
    return {
        key: sorted(
            rows,
            key=_candidate_feedback_sort_key,
            reverse=True,
        )
        for key, rows in index.items()
    }


def _candidate_feedback_sort_key(
    row: Mapping[str, Any],
) -> tuple[
    bool,
    int,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    int,
    int,
    bool,
    bool,
    bool,
]:
    last_epoch, row_count = _candidate_feedback_training_progress(row)
    renderer_passed, tether_passed, guard_passed, direct_feedback_clean = (
        _candidate_feedback_proof_quality(row)
    )
    return (
        bool(row.get("scope_matches_candidate")),
        int(row.get("measured_num_pairs") or 0),
        renderer_passed,
        tether_passed,
        guard_passed,
        direct_feedback_clean,
        bool(row.get("full_video_mlx_response_attached")),
        row.get("training_stopped") is not True,
        last_epoch,
        row_count,
        bool(row.get("receiver_proof_attached")),
        bool(row.get("full_video_local_prefilter_attached")),
        bool(row.get("local_cpu_replay_gate_attached")),
    )


def _candidate_feedback_proof_quality(
    row: Mapping[str, Any],
) -> tuple[bool, bool, bool, bool]:
    proof = row.get("snerv_renderer_nondegenerate_proof")
    renderer_passed = bool(row.get("snerv_renderer_nondegenerate_proof_passed")) or (
        isinstance(proof, Mapping) and proof.get("passed") is True
    )
    return (
        renderer_passed,
        bool(row.get("snerv_scorer_domain_tether_passed")),
        bool(row.get("snerv_scorer_input_distribution_guard_proof_passed")),
        not bool(row.get("direct_feedback_blockers")),
    )


def _candidate_feedback_training_progress(row: Mapping[str, Any]) -> tuple[int, int]:
    for key in (
        "training_telemetry",
        "snerv_score_aware_long_training_telemetry_contract",
        "score_aware_long_training_telemetry_contract",
    ):
        telemetry = row.get(key)
        if not isinstance(telemetry, Mapping):
            continue
        return (
            int(_first_present_int(telemetry, ("last_epoch", "epoch")) or 0),
            int(_first_present_int(telemetry, ("row_count", "step_count")) or 0),
        )
    return (0, 0)


def _normalize_candidate_feedback_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("schema") == NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA:
        row = _sanitize_direct_candidate_feedback_row(source)
    elif source.get("schema") == FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA and isinstance(source.get("row"), Mapping):
        row = _sanitize_direct_candidate_feedback_row(source["row"])
    elif source.get("schema") == "hinerv_training_telemetry_feedback.v1":
        row = _normalize_hinerv_training_telemetry_feedback(source)
    elif source.get("schema") == "compact_renderer_mlx_spine_runner.v1":
        row = build_nerv_candidate_feedback_row(
            runner_report=source,
            source_report_path=source.get("_candidate_feedback_source_path"),
        )
    else:
        row = dict(source)
    return _augment_feedback_row(row, source)


def _sanitize_direct_candidate_feedback_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    if row.get("feedback_kind") == "training_telemetry":
        return row
    blockers = [str(blocker) for blocker in row.get("direct_feedback_blockers") or []]
    _guard_direct_feedback_bool(
        row,
        bool_key="receiver_proof_attached",
        path_keys=("receiver_proof_path", "receiver_proof_report_path"),
        sha_keys=("receiver_proof_sha256",),
        blocker="direct_feedback_receiver_proof_file_missing",
        blockers=blockers,
    )
    _guard_direct_feedback_bool(
        row,
        bool_key="full_video_local_prefilter_attached",
        path_keys=(
            "full_video_local_prefilter_path",
            "mlx_prefilter_path",
            "full_video_mlx_response_path",
        ),
        sha_keys=(
            "full_video_local_prefilter_sha256",
            "mlx_prefilter_sha256",
            "full_video_mlx_response_sha256",
        ),
        blocker="direct_feedback_full_video_prefilter_file_missing",
        blockers=blockers,
    )
    _guard_direct_feedback_bool(
        row,
        bool_key="local_cpu_replay_gate_attached",
        path_keys=("local_cpu_replay_summary_path", "local_cpu_replay_gate_path"),
        sha_keys=("local_cpu_replay_summary_sha256", "local_cpu_replay_gate_sha256"),
        blocker="direct_feedback_local_cpu_replay_file_missing",
        blockers=blockers,
    )
    _guard_direct_feedback_bool(
        row,
        bool_key="native_mlx_receiver_proof_passed",
        path_keys=(
            "snerv_mlx_native_export_receiver_proof_path",
            "receiver_proof_path",
        ),
        sha_keys=("snerv_mlx_native_export_receiver_proof_sha256",),
        blocker="direct_feedback_native_receiver_proof_file_missing",
        blockers=blockers,
    )
    evidence = row.get("snerv_mlx_native_file_backed_export_evidence")
    file_backed_ready = bool(
        isinstance(evidence, Mapping) and evidence.get("required_pair_file_backed_export_proof_passed") is True
    )
    if row.get("native_mlx_full600_campaign_ready") is True and not file_backed_ready:
        row["native_mlx_full600_campaign_ready"] = False
        blockers.append("direct_feedback_native_full600_file_backed_evidence_missing")
    row["direct_feedback_blockers"] = _dedupe(blockers)
    return row


def _guard_direct_feedback_bool(
    row: dict[str, Any],
    *,
    bool_key: str,
    path_keys: Sequence[str],
    sha_keys: Sequence[str],
    blocker: str,
    blockers: list[str],
) -> None:
    if row.get(bool_key) is not True:
        return
    if _direct_feedback_file_evidence_valid(row, path_keys=path_keys, sha_keys=sha_keys):
        return
    row[bool_key] = False
    blockers.append(blocker)


def _direct_feedback_file_evidence_valid(
    row: Mapping[str, Any],
    *,
    path_keys: Sequence[str],
    sha_keys: Sequence[str],
) -> bool:
    for path_key in path_keys:
        raw = row.get(path_key)
        if not raw:
            continue
        path = Path(str(raw)).expanduser().resolve(strict=False)
        if not path.is_file():
            continue
        expected_sha = next(
            (str(row.get(key)) for key in sha_keys if row.get(key)),
            "",
        )
        return not expected_sha or _sha256_file(path) == expected_sha
    return False


def _normalize_hinerv_training_telemetry_feedback(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert compact foreground/harvest telemetry into planner feedback.

    Foreground proof rows are not runner reports and intentionally carry no
    archive/replay authority. They are still high-value launch-pressure signal
    when they come from a full600 candidate and show a real optimization
    dynamic, such as PoseNet recovering while SegNet remains binding. Convert
    only that steering signal into the same false-authority feedback shape the
    campaign planner already consumes.
    """

    candidate_id = str(source.get("candidate_id") or "").strip()
    last_epoch = int(source.get("last_epoch") or 0)
    seg_still_binding = bool(source.get("segnet_still_binding") is True)
    pose_recovered = bool(source.get("pose_recovered_from_initial_spike") is True)
    pose_instability = bool(source.get("pose_instability_detected") is True)
    recommended_lr = _float_or_none(source.get("recommended_learning_rate"))
    observed_seg_weight = _float_or_none(source.get("observed_segnet_distillation_weight"))
    recommended_seg_weight = _float_or_none(source.get("recommended_segnet_distillation_weight"))
    if seg_still_binding and recommended_seg_weight is None:
        recommended_seg_weight = recommend_segnet_distillation_weight_for_stagnation(observed_seg_weight)
    recommended_mutations = list(source.get("recommended_next_mutations") or [])
    if seg_still_binding and (
        "increase_segnet_distillation_weight_from_stagnation_telemetry" not in recommended_mutations
    ):
        recommended_mutations.append("increase_segnet_distillation_weight_from_stagnation_telemetry")
    launch_control_feedback_ready = bool(
        candidate_id
        and last_epoch > 0
        and (
            (pose_instability and recommended_lr is not None and recommended_lr > 0.0)
            or (seg_still_binding and recommended_seg_weight is not None and recommended_seg_weight > 1.0)
        )
    )
    return {
        "schema": NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA,
        "telemetry_feedback_schema": str(source.get("schema")),
        "feedback_kind": "training_telemetry",
        "feedback_scope": "full600_training_telemetry",
        "feedback_ready": False,
        "launch_control_feedback_ready": launch_control_feedback_ready,
        "family": "hi_nerv",
        "candidate_id": candidate_id,
        "candidate_num_pairs": 600,
        "measured_num_pairs": 600,
        "scope_matches_candidate": True,
        "training_stopped": False,
        "source_report_path": source.get("telemetry_path"),
        "observed_learning_rate": source.get("learning_rate"),
        "pose_instability_detected": pose_instability,
        "recommended_learning_rate": recommended_lr,
        "pose_recovered_from_initial_spike": pose_recovered,
        "seg_stagnation_detected": seg_still_binding,
        "segnet_still_binding": seg_still_binding,
        "observed_segnet_distillation_weight": observed_seg_weight,
        "recommended_segnet_distillation_weight": recommended_seg_weight,
        "recommended_segnet_distillation_weight_multiplier": (2.0 if seg_still_binding else None),
        "recommended_launch_mutations": recommended_mutations,
        "training_telemetry": {
            "schema": str(source.get("schema")),
            "last_epoch": last_epoch,
            "row_count": int(source.get("row_count") or 0),
            "first_pose_axis": source.get("first_pose_axis"),
            "last_pose_axis": source.get("last_pose_axis"),
            "first_seg_axis": source.get("first_seg_axis"),
            "last_seg_axis": source.get("last_seg_axis"),
            "last_recon_aux": source.get("last_recon_aux"),
            "last_loss": source.get("last_loss"),
            "observed_segnet_distillation_weight": observed_seg_weight,
            "pose_recovered_from_initial_spike": pose_recovered,
            "segnet_still_binding": seg_still_binding,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _augment_feedback_row(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    native = source.get("snerv_mlx_native_export")
    native_file_evidence = source.get("snerv_mlx_native_file_backed_export_evidence")
    native_file_metadata = (
        native_file_evidence.get("packet_metadata_summary")
        if isinstance(native_file_evidence, Mapping)
        and isinstance(native_file_evidence.get("packet_metadata_summary"), Mapping)
        else {}
    )
    if isinstance(native_file_evidence, Mapping):
        out.setdefault(
            "snerv_mlx_native_file_backed_export_evidence",
            dict(native_file_evidence),
        )
        out.setdefault(
            "native_mlx_file_backed_export_proof_passed",
            bool(native_file_evidence.get("required_pair_file_backed_export_proof_passed")),
        )
    if isinstance(native, Mapping):
        scorer_loop_status = _native_scorer_loop_qat_status(native)
        out.setdefault(
            "native_mlx_receiver_proof_passed",
            bool(native.get("receiver_proof_passed") and native.get("receiver_contract_satisfied")),
        )
        out.setdefault(
            "native_mlx_full600_campaign_ready",
            bool(native.get("native_mlx_full600_campaign_ready")),
        )
        out["native_mlx_scorer_loop_qat_attached"] = bool(
            scorer_loop_status["attached"]
        )
        out["native_mlx_scorer_loop_qat_receiver_contract_satisfied"] = bool(
            scorer_loop_status["receiver_contract_satisfied"]
        )
        out["native_mlx_scorer_loop_qat_ready_for_pose_guard_gate"] = bool(
            scorer_loop_status["ready_for_pose_guard_gate"]
        )
        out["native_mlx_scorer_loop_qat_accepted_improvement"] = bool(
            scorer_loop_status["accepted_improvement"]
        )
        out["native_mlx_scorer_loop_qat_best_materialized"] = bool(
            scorer_loop_status["best_materialized"]
        )
        out.setdefault(
            "snerv_mlx_native_hf_decoder_training",
            native.get("native_mlx_hf_decoder_training"),
        )
        out.setdefault(
            "snerv_mlx_native_training_executed",
            native.get("native_mlx_training_executed"),
        )
        out.setdefault(
            "snerv_mlx_native_training_kind",
            native.get("native_mlx_training_kind"),
        )
        out.setdefault(
            "snerv_mlx_native_training_export_guard",
            native.get("native_mlx_training_export_guard"),
        )
        out.setdefault(
            "snerv_mlx_native_training_export_guard_passed",
            (
                dict(native.get("native_mlx_training_export_guard") or {}).get("export_guard_passed")
                if isinstance(native.get("native_mlx_training_export_guard"), Mapping)
                else None
            ),
        )
        out.setdefault(
            "snerv_mlx_native_training_export_guard_blockers",
            list(dict(native.get("native_mlx_training_export_guard") or {}).get("blockers") or [])
            if isinstance(native.get("native_mlx_training_export_guard"), Mapping)
            else [],
        )
        out.setdefault("snerv_mlx_native_export_executed", native.get("executed"))
        out.setdefault(
            "snerv_mlx_native_export_artifact_report_path",
            native.get("artifact_report_path") or native.get("report_path"),
        )
        out.setdefault("snerv_mlx_native_export_packet_path", native.get("packet_path"))
        out.setdefault("snerv_mlx_native_export_packet_sha256", native.get("packet_sha256"))
        out.setdefault("snerv_mlx_native_export_archive_path", native.get("archive_path"))
        out.setdefault("snerv_mlx_native_export_archive_sha256", native.get("archive_sha256"))
        out.setdefault(
            "snerv_mlx_native_export_receiver_proof_path",
            native.get("receiver_proof_path"),
        )
        for key in (
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested",
            "snerv_official_mfu_hfr_tub_export_bound",
            "snerv_official_mfu_hfr_tub_export_bound_semantics",
            "snerv_official_mfu_hfr_tub_receiver_payload_bound",
            "snerv_official_mfu_hfr_tub_frame_producing_export",
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound",
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority",
            "snerv_official_trained_checkpoint_loaded",
            "snerv_official_hfr_trained_checkpoint_weight_mapping_proven",
            "snerv_official_mfu_trained_checkpoint_weight_mapping_proven",
            "snerv_official_mfu_hfr_trained_checkpoint_weight_mapping_proven",
            "snerv_official_mfu_receiver_activation_payload_bound",
            "snerv_official_tub_receiver_activation_payload_bound",
            "snerv_official_native_receiver_state_mapping_proven",
            "snerv_official_tub_temporal_encoder_weight_mapping_proven",
            "snerv_official_trained_checkpoint_state_dict_mapping_verified",
            "snerv_official_trained_checkpoint_state_dict_path",
            "snerv_official_trained_checkpoint_state_dict_slice_path",
            "snerv_official_trained_checkpoint_state_dict_slice_present",
            "snerv_official_trained_checkpoint_state_dict_slice_file_present",
            "snerv_official_trained_checkpoint_state_dict_slice_bytes",
            "snerv_official_trained_checkpoint_state_dict_slice_sha256",
            "snerv_official_trained_checkpoint_state_dict_slice_member_count",
            "snerv_official_trained_checkpoint_state_dict_slice_member_names",
            "snerv_official_trained_checkpoint_state_dict_slice_runner_arg",
            "snerv_trained_state_exportable",
            "snerv_checkpoint_trained_state_exportable",
            "snerv_score_aware_long_training_trained_state_exportable",
            "checkpoint_trained_state_exportable",
            "score_aware_long_training_trained_state_exportable",
            "source_faithful_stack",
        ):
            out.setdefault(
                key,
                _first_mapping_value(
                    key,
                    native,
                    native_file_evidence,
                    native_file_metadata,
                ),
            )
        out.setdefault(
            "official_source_parity_blockers",
            list(
                native.get("official_source_parity_blockers")
                or native.get("snerv_official_mfu_hfr_tub_export_blockers")
                or (
                    native_file_evidence.get("official_source_parity_blockers")
                    if isinstance(native_file_evidence, Mapping)
                    else None
                )
                or (
                    native_file_evidence.get("snerv_official_mfu_hfr_tub_export_blockers")
                    if isinstance(native_file_evidence, Mapping)
                    else None
                )
                or (
                    native_file_metadata.get("official_source_parity_blockers")
                    if isinstance(native_file_metadata, Mapping)
                    else None
                )
                or []
            ),
        )
    if "receiver_proof_attached" not in out:
        receiver_paths = out.get("receiver_proof_report_paths")
        out["receiver_proof_attached"] = bool(
            out.get("scope_matches_candidate")
            and (
                out.get("native_mlx_receiver_proof_passed")
                or (isinstance(receiver_paths, list) and len(receiver_paths) > 0)
            )
        )
    if "full_video_local_prefilter_attached" not in out:
        out["full_video_local_prefilter_attached"] = bool(out.get("mlx_prefilter_has_full_video"))
    if "local_cpu_replay_gate_attached" not in out:
        out["local_cpu_replay_gate_attached"] = bool(
            out.get("local_cpu_replay_gate_executed") or out.get("local_cpu_replay_summary_present")
        )
    return out


def _native_scorer_loop_qat_status(native: Mapping[str, Any]) -> dict[str, Any]:
    """Return native SNeRV scorer-loop QAT status from the emitted export record.

    A nested ``scorer_loop_qat`` record is the packet/export truth surface. The
    legacy top-level flags are accepted only when no nested record exists.
    """

    scorer_loop_raw = native.get("scorer_loop_qat")
    if isinstance(scorer_loop_raw, Mapping):
        scorer_loop = dict(scorer_loop_raw)
        return {
            "attached": bool(scorer_loop.get("executed")),
            "receiver_contract_satisfied": bool(
                scorer_loop.get("receiver_contract_satisfied")
            ),
            "ready_for_pose_guard_gate": bool(
                scorer_loop.get("ready_for_pose_guard_gate")
            ),
            "accepted_improvement": bool(scorer_loop.get("accepted_improvement")),
            "best_materialized": bool(
                scorer_loop.get("emitted_packet_uses_scorer_loop_best_decoder")
            ),
        }
    return {
        "attached": bool(native.get("scorer_loop_qat_attached")),
        "receiver_contract_satisfied": bool(
            native.get("scorer_loop_qat_receiver_contract_satisfied")
        ),
        "ready_for_pose_guard_gate": bool(
            native.get("scorer_loop_qat_ready_for_pose_guard_gate")
        ),
        "accepted_improvement": bool(
            native.get("scorer_loop_qat_accepted_improvement")
        ),
        "best_materialized": bool(native.get("scorer_loop_qat_best_materialized")),
    }


def _first_mapping_value(key: str, *sources: Any) -> Any:
    for source in sources:
        if isinstance(source, Mapping) and source.get(key) is not None:
            return source.get(key)
    return None


def _axis_trace_measurements_from_sources(*sources: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in (
            "pr95_distortion_axis_trace_measurements",
            "distortion_axis_trace_measurements",
            "axis_trace_measurements",
            "axis_trace_rows",
        ):
            value = source.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                rows.extend(item for item in value if isinstance(item, Mapping))
        nested = source.get("pr95_distortion_axis_trace")
        if isinstance(nested, Mapping):
            value = nested.get("measurements") or nested.get("rows")
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                rows.extend(item for item in value if isinstance(item, Mapping))
    return rows


def _snerv_native_artifact_evidence_from_feedback(
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    embedded = feedback.get("snerv_mlx_native_file_backed_export_evidence")
    artifact = {
        "num_pairs": feedback.get("candidate_num_pairs")
        if feedback.get("scope_matches_candidate")
        else feedback.get("measured_num_pairs"),
        "executed": feedback.get("snerv_mlx_native_export_executed"),
        "artifact_report_path": feedback.get("snerv_mlx_native_export_artifact_report_path"),
        "packet_path": feedback.get("snerv_mlx_native_export_packet_path"),
        "packet_sha256": feedback.get("snerv_mlx_native_export_packet_sha256"),
        "archive_path": feedback.get("snerv_mlx_native_export_archive_path"),
        "archive_sha256": feedback.get("snerv_mlx_native_export_archive_sha256"),
        "receiver_proof_path": feedback.get("snerv_mlx_native_export_receiver_proof_path"),
        "receiver_proof_passed": feedback.get("snerv_mlx_native_export_receiver_proof_passed")
        or feedback.get("native_mlx_receiver_proof_passed"),
        "receiver_contract_satisfied": feedback.get("snerv_mlx_native_export_receiver_contract_satisfied"),
        "snerv_official_mfu_hfr_tub_numeric_primitives_requested": feedback.get(
            "snerv_official_mfu_hfr_tub_numeric_primitives_requested"
        ),
        "snerv_official_mfu_hfr_tub_export_bound": feedback.get("snerv_official_mfu_hfr_tub_export_bound"),
        "snerv_official_mfu_hfr_tub_export_bound_semantics": feedback.get(
            "snerv_official_mfu_hfr_tub_export_bound_semantics"
        ),
        "snerv_official_mfu_hfr_tub_receiver_payload_bound": feedback.get(
            "snerv_official_mfu_hfr_tub_receiver_payload_bound"
        ),
        "snerv_official_mfu_hfr_tub_frame_producing_export": feedback.get(
            "snerv_official_mfu_hfr_tub_frame_producing_export"
        ),
        "snerv_official_mfu_hfr_tub_source_forward_replay_bound": feedback.get(
            "snerv_official_mfu_hfr_tub_source_forward_replay_bound"
        ),
        "snerv_official_mfu_hfr_tub_source_forward_replay_authority": feedback.get(
            "snerv_official_mfu_hfr_tub_source_forward_replay_authority"
        ),
        "snerv_official_trained_checkpoint_mapping_manifest": feedback.get(
            "snerv_official_trained_checkpoint_mapping_manifest"
        ),
        "snerv_official_trained_checkpoint_loaded": feedback.get("snerv_official_trained_checkpoint_loaded"),
        "snerv_official_hfr_trained_checkpoint_weight_mapping_proven": feedback.get(
            "snerv_official_hfr_trained_checkpoint_weight_mapping_proven"
        ),
        "snerv_official_mfu_trained_checkpoint_weight_mapping_proven": feedback.get(
            "snerv_official_mfu_trained_checkpoint_weight_mapping_proven"
        ),
        "snerv_official_mfu_hfr_trained_checkpoint_weight_mapping_proven": feedback.get(
            "snerv_official_mfu_hfr_trained_checkpoint_weight_mapping_proven"
        ),
        "snerv_official_mfu_receiver_activation_payload_bound": feedback.get(
            "snerv_official_mfu_receiver_activation_payload_bound"
        ),
        "snerv_official_tub_receiver_activation_payload_bound": feedback.get(
            "snerv_official_tub_receiver_activation_payload_bound"
        ),
        "snerv_official_native_receiver_state_mapping_proven": feedback.get(
            "snerv_official_native_receiver_state_mapping_proven"
        ),
        "snerv_official_tub_temporal_encoder_weight_mapping_proven": feedback.get(
            "snerv_official_tub_temporal_encoder_weight_mapping_proven"
        ),
        "snerv_official_trained_checkpoint_state_dict_mapping_verified": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_mapping_verified"
        ),
        "snerv_official_trained_checkpoint_state_dict_path": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_path"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_path": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_path"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_present": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_present"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_file_present": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_file_present"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_bytes": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_bytes"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_sha256": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_sha256"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_member_count": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_member_count"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_member_names": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_member_names"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_runner_arg": feedback.get(
            "snerv_official_trained_checkpoint_state_dict_slice_runner_arg"
        ),
        "snerv_official_trained_checkpoint_mapping_blockers": feedback.get(
            "snerv_official_trained_checkpoint_mapping_blockers"
        ),
        "source_faithful_stack": feedback.get("source_faithful_stack"),
        "official_source_parity_blockers": feedback.get("official_source_parity_blockers"),
        "native_mlx_training_executed": feedback.get("snerv_mlx_native_training_executed"),
        "native_mlx_hf_decoder_training": feedback.get("snerv_mlx_native_hf_decoder_training"),
        "native_mlx_training_export_guard": feedback.get("snerv_mlx_native_training_export_guard"),
        "scorer_loop_qat": {
            "executed": feedback.get("native_mlx_scorer_loop_qat_attached"),
            "receiver_contract_satisfied": feedback.get("native_mlx_scorer_loop_qat_receiver_contract_satisfied"),
            "ready_for_pose_guard_gate": feedback.get("native_mlx_scorer_loop_qat_ready_for_pose_guard_gate"),
            "accepted_improvement": feedback.get("native_mlx_scorer_loop_qat_accepted_improvement"),
            "emitted_packet_uses_scorer_loop_best_decoder": feedback.get(
                "native_mlx_scorer_loop_qat_best_materialized"
            ),
        },
    }
    compact_artifact = {key: value for key, value in artifact.items() if value is not None}
    if isinstance(embedded, Mapping):
        merged = dict(embedded)
        merged.update(compact_artifact)
        return merged
    return compact_artifact


def _candidate_feedback_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    family_key = _family_key(family)
    rows = list((index or {}).get((family_key, candidate_id)) or [])
    if rows:
        out = dict(rows[0])
        out.setdefault("candidate_id_match", True)
        out.setdefault("feedback_match_scope", "candidate")
        return out
    fallback_rows = _family_level_candidate_feedback_rows(
        candidate=candidate,
        family=family_key,
        index=index,
    )
    if not fallback_rows:
        return {}
    return _sanitize_family_level_candidate_feedback(
        row=fallback_rows[0],
        target_candidate=candidate,
    )


def _snerv_feedback_with_modelsize_byte_cap_evidence(
    *,
    feedback: Mapping[str, Any],
    candidate: Mapping[str, Any],
    modelsize_byte_cap_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    if _family_key(str(candidate.get("family") or "")) != "snerv":
        return dict(feedback)
    matching = [
        row for row in modelsize_byte_cap_preflight.get("matching_observations") or () if isinstance(row, Mapping)
    ]
    if not matching:
        return dict(feedback)
    observation = dict(matching[0])
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return dict(feedback)
    candidate_pairs = _first_present_int(candidate, ("num_pairs",)) or 0
    measured_pairs = _first_present_int(observation, ("measured_num_pairs",)) or 0
    scope_matches = bool(candidate_pairs > 0 and measured_pairs == candidate_pairs)
    archive_bytes = _first_present_int(observation, ("measured_archive_bytes",))
    packet_bytes = _first_present_int(observation, ("measured_payload_bytes",))
    packet_path = _existing_path(observation.get("packet_path"))
    archive_path = _existing_path(observation.get("archive_path"))
    proof_path = _existing_path(observation.get("receiver_proof_path"))
    report_path = _existing_path(
        observation.get("artifact_report_path")
        or observation.get("report_path")
        or observation.get("source_path")
    )
    official_state_dict_path = _existing_path(
        observation.get("snerv_official_trained_checkpoint_state_dict_path")
        or observation.get("snerv_official_trained_checkpoint_state_dict_slice_path")
    )
    packet_sha = str(observation.get("packet_sha256") or "").strip()
    archive_sha = str(observation.get("archive_sha256") or "").strip()
    if packet_path is not None and not packet_sha:
        packet_sha = _sha256_file(packet_path)
    if archive_path is not None and not archive_sha:
        archive_sha = _sha256_file(archive_path)
    proof_sha = _sha256_file(proof_path) if proof_path is not None else None
    receiver_closed = bool(
        observation.get("receiver_closed") is True
        and observation.get("receiver_contract_satisfied") is True
        and str(observation.get("receiver_closed_status") or "")
        in {"inline_receiver_closed", "associated_receiver_proof"}
    )
    file_backed = bool(
        scope_matches
        and receiver_closed
        and packet_path is not None
        and archive_path is not None
        and proof_path is not None
        and report_path is not None
        and packet_bytes is not None
        and archive_bytes is not None
    )
    bridged: dict[str, Any] = {
        "schema": NERV_CANDIDATE_FEEDBACK_ROW_SCHEMA,
        "feedback_kind": "modelsize_byte_cap_receiver_proof",
        "feedback_scope": (
            "full600_native_file_backed_snar1_export" if file_backed else "modelsize_byte_cap_receiver_proof_bytes_only"
        ),
        "byte_feedback_source": "modelsize_byte_cap_receiver_proof",
        "family": "snerv",
        "candidate_id": candidate_id,
        "candidate_num_pairs": int(candidate_pairs),
        "measured_num_pairs": int(measured_pairs or candidate_pairs),
        "scope_matches_candidate": scope_matches,
        "feedback_ready": bool(scope_matches and archive_bytes is not None),
        "measured_payload_bytes": packet_bytes,
        "measured_archive_bytes": archive_bytes,
        "archive_minus_nominal_bytes": _first_present_int(
            observation,
            ("archive_minus_nominal_bytes",),
        ),
        "archive_to_nominal_ratio": observation.get("archive_to_nominal_ratio"),
        "calibrated_archive_overrun_bytes": _first_present_int(
            observation,
            ("calibrated_archive_overrun_bytes",),
        ),
        "required_nominal_payload_bytes_max": _first_present_int(
            observation,
            ("required_nominal_payload_bytes_max",),
        ),
        "hard_byte_ceiling_measurement_bypass_enabled": bool(
            observation.get("hard_byte_ceiling_measurement_bypass_enabled")
        ),
        "hard_byte_ceiling_checked_after_export": (
            None
            if observation.get("hard_byte_ceiling_checked_after_export") is None
            else bool(observation.get("hard_byte_ceiling_checked_after_export"))
        ),
        "receiver_proof_attached": proof_path is not None,
        "receiver_proof_path": proof_path.as_posix() if proof_path else None,
        "receiver_proof_sha256": proof_sha,
        "native_mlx_receiver_proof_passed": receiver_closed,
        "native_mlx_full600_campaign_ready": file_backed,
        "snerv_mlx_native_export_executed": file_backed,
        "snerv_mlx_native_export_artifact_report_path": (report_path.as_posix() if report_path else None),
        "snerv_mlx_native_export_packet_path": (packet_path.as_posix() if packet_path else None),
        "snerv_mlx_native_export_packet_sha256": packet_sha or None,
        "snerv_mlx_native_export_archive_path": (archive_path.as_posix() if archive_path else None),
        "snerv_mlx_native_export_archive_sha256": archive_sha or None,
        "snerv_mlx_native_export_receiver_proof_path": (proof_path.as_posix() if proof_path else None),
        "snerv_mlx_native_export_receiver_proof_sha256": proof_sha,
        "snerv_mlx_native_export_receiver_proof_passed": receiver_closed,
        "snerv_mlx_native_export_receiver_contract_satisfied": receiver_closed,
        "snerv_official_trained_checkpoint_state_dict_path": (
            official_state_dict_path.as_posix() if official_state_dict_path else None
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_path": (
            observation.get("snerv_official_trained_checkpoint_state_dict_slice_path")
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_present": observation.get(
            "snerv_official_trained_checkpoint_state_dict_slice_present"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_file_present": bool(
            official_state_dict_path is not None
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_bytes": observation.get(
            "snerv_official_trained_checkpoint_state_dict_slice_bytes"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_sha256": observation.get(
            "snerv_official_trained_checkpoint_state_dict_slice_sha256"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_member_count": observation.get(
            "snerv_official_trained_checkpoint_state_dict_slice_member_count"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_member_names": observation.get(
            "snerv_official_trained_checkpoint_state_dict_slice_member_names"
        ),
        "snerv_official_trained_checkpoint_state_dict_slice_runner_arg": observation.get(
            "snerv_official_trained_checkpoint_state_dict_slice_runner_arg"
        ),
        "snerv_mlx_native_file_backed_export_evidence": (
            {
                "schema": "snerv_mlx_native_train_export.v1",
                "executed": True,
                "num_pairs": int(measured_pairs or candidate_pairs),
                "candidate_id": candidate_id,
                "artifact_report_path": report_path.as_posix(),
                "packet_path": packet_path.as_posix(),
                "packet_sha256": packet_sha,
                "archive_path": archive_path.as_posix(),
                "archive_sha256": archive_sha,
                "receiver_proof_path": proof_path.as_posix(),
                "receiver_proof_passed": True,
                "receiver_contract_satisfied": True,
                "file_backed_export_proof_passed": True,
                "required_pair_file_backed_export_proof_passed": True,
                "official_checkpoint_export_binding": {
                    "schema": "snerv_official_checkpoint_export_binding.v1",
                    "official_trained_checkpoint_state_dict_slice_present": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_present"
                    ),
                    "official_trained_checkpoint_state_dict_slice_path": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_path"
                    ),
                    "official_trained_checkpoint_state_dict_slice_bytes": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_bytes"
                    ),
                    "official_trained_checkpoint_state_dict_slice_sha256": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_sha256"
                    ),
                    "official_trained_checkpoint_state_dict_slice_member_count": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_member_count"
                    ),
                    "official_trained_checkpoint_state_dict_slice_member_names": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_member_names"
                    ),
                    "official_trained_checkpoint_state_dict_slice_runner_arg": observation.get(
                        "snerv_official_trained_checkpoint_state_dict_slice_runner_arg"
                    ),
                },
                "blockers": [],
                **FALSE_AUTHORITY,
            }
            if file_backed
            else None
        ),
        **FALSE_AUTHORITY,
    }
    out = dict(feedback)
    for key, value in bridged.items():
        if value is None:
            continue
        if key not in out or out.get(key) in (None, False, [], {}):
            out[key] = value
    return out


def _existing_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve(strict=False)
    return path if path.is_file() else None


def _family_level_candidate_feedback_rows(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> list[dict[str, Any]]:
    family_key = _family_key(family)
    rows: list[dict[str, Any]] = []
    for (row_family, _row_candidate_id), candidate_rows in (index or {}).items():
        if row_family != family_key:
            continue
        for row in candidate_rows:
            if _family_level_candidate_feedback_applicable(
                candidate=candidate,
                family=family_key,
                row=row,
            ):
                rows.append(dict(row))
    return sorted(rows, key=_candidate_feedback_sort_key, reverse=True)


def _family_training_telemetry_context_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    family_key = _family_key(family)
    if family_key != "hi_nerv":
        return {}
    target_num_pairs = int(candidate.get("num_pairs") or 0)
    if target_num_pairs <= 0:
        return {}
    rows: list[dict[str, Any]] = []
    for (row_family, _row_candidate_id), candidate_rows in (index or {}).items():
        if row_family != family_key:
            continue
        for row in candidate_rows:
            if _family_training_telemetry_context_applicable(
                target_num_pairs=target_num_pairs,
                row=row,
            ):
                rows.append(dict(row))
    if not rows:
        return {}
    return _sanitize_family_training_telemetry_context(
        row=sorted(rows, key=_candidate_feedback_sort_key, reverse=True)[0],
        target_candidate=candidate,
    )


def _family_training_telemetry_context_applicable(
    *,
    target_num_pairs: int,
    row: Mapping[str, Any],
) -> bool:
    if str(row.get("feedback_kind") or "").strip() != "training_telemetry":
        return False
    if str(row.get("feedback_scope") or "").strip() != "full600_training_telemetry":
        return False
    measured_num_pairs = int(row.get("measured_num_pairs") or row.get("candidate_num_pairs") or 0)
    return measured_num_pairs == int(target_num_pairs)


def _sanitize_family_training_telemetry_context(
    *,
    row: Mapping[str, Any],
    target_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    source_candidate_id = str(row.get("candidate_id") or "").strip()
    target_candidate_id = str(target_candidate.get("candidate_id") or "").strip()
    candidate_match = bool(source_candidate_id and source_candidate_id == target_candidate_id)
    out["source_candidate_id"] = source_candidate_id
    out["target_candidate_id"] = target_candidate_id
    out["candidate_id_match"] = candidate_match
    out["feedback_match_scope"] = (
        "candidate_training_telemetry_context" if candidate_match else "family_training_telemetry_context"
    )
    out["scope_matches_candidate"] = candidate_match
    out["receiver_proof_attached"] = False
    out["full_video_local_prefilter_attached"] = False
    out["local_cpu_replay_gate_attached"] = False
    out["measured_archive_bytes"] = None
    out["measured_payload_bytes"] = None
    out["launch_control_feedback_ready"] = False
    out["context_only"] = True
    out["feedback_reuse_policy"] = "telemetry_context_only_no_launch_mutation_no_archive_receiver_or_replay_authority"
    out.update(FALSE_AUTHORITY)
    return out


def _family_level_candidate_feedback_applicable(
    *,
    candidate: Mapping[str, Any],
    family: str,
    row: Mapping[str, Any],
) -> bool:
    # Only reuse optimizer-stability telemetry across sibling HiNeRV candidates.
    # Archive, receiver, and replay evidence remain candidate-specific.
    family_key = _family_key(family)
    if family_key == "snerv":
        if _snerv_prelaunch_renderer_proof_applicable(
            candidate=candidate,
            row=row,
        ):
            return True
        feedback_kind = str(row.get("feedback_kind") or "").strip()
        if feedback_kind == "upstream_eval_gate":
            target_candidate_id = str(candidate.get("candidate_id") or "").strip()
            source_candidate_id = str(row.get("candidate_id") or "").strip()
            if not target_candidate_id or not source_candidate_id or source_candidate_id == target_candidate_id:
                return False
            measured_num_pairs = int(row.get("measured_num_pairs") or 0)
            target_num_pairs = int(candidate.get("num_pairs") or 0)
            return bool(
                target_num_pairs > 0
                and measured_num_pairs == target_num_pairs
                and str(row.get("feedback_scope") or "") in {"full600_upstream_cpu_eval", "full600_upstream_eval_gate"}
                and any(
                    str(blocker).startswith("snerv_upstream_eval_gate_")
                    for blocker in row.get("direct_feedback_blockers") or ()
                )
            )
        if feedback_kind == "training_telemetry":
            target_candidate_id = str(candidate.get("candidate_id") or "").strip()
            source_candidate_id = str(row.get("candidate_id") or "").strip()
            if not target_candidate_id or not source_candidate_id or source_candidate_id == target_candidate_id:
                return False
            measured_num_pairs = int(row.get("measured_num_pairs") or 0)
            target_num_pairs = int(candidate.get("num_pairs") or 0)
            return bool(
                target_num_pairs > 0
                and measured_num_pairs == target_num_pairs
                and str(row.get("feedback_scope") or "") == "full600_training_telemetry"
                and row.get("degenerate_renderer_risk_detected") is True
                and "snerv_scorer_domain_tether_missing_telemetry"
                in {str(blocker) for blocker in row.get("direct_feedback_blockers") or ()}
            )
        if feedback_kind != "full_video_mlx_scorer_response":
            return False
        target_candidate_id = str(candidate.get("candidate_id") or "").strip()
        source_candidate_id = str(row.get("candidate_id") or "").strip()
        if not target_candidate_id or not source_candidate_id or source_candidate_id == target_candidate_id:
            return False
        measured_num_pairs = int(row.get("measured_num_pairs") or 0)
        target_num_pairs = int(candidate.get("num_pairs") or 0)
        return bool(
            target_num_pairs > 0
            and measured_num_pairs == target_num_pairs
            and (
                row.get("full_video_mlx_response_attached") is True
                or str(row.get("feedback_scope") or "") == "full600_mlx_scorer_response"
            )
            and any(
                str(blocker).startswith("snerv_full_video_mlx_response_")
                for blocker in row.get("direct_feedback_blockers") or ()
            )
        )
    if family_key != "hi_nerv":
        return False
    target_candidate_id = str(candidate.get("candidate_id") or "").strip()
    source_candidate_id = str(row.get("candidate_id") or "").strip()
    if not target_candidate_id or not source_candidate_id:
        return False
    if source_candidate_id == target_candidate_id:
        return False
    if str(row.get("feedback_kind") or "").strip() != "training_telemetry":
        return False
    if str(row.get("feedback_scope") or "").strip() != "full600_training_telemetry":
        return False
    pose_feedback = bool(row.get("pose_instability_detected") is True)
    seg_feedback = bool(row.get("seg_stagnation_detected") is True)
    recommended = _float_or_none(row.get("recommended_learning_rate"))
    recommended_seg_weight = _float_or_none(row.get("recommended_segnet_distillation_weight"))
    if not (
        (pose_feedback and recommended is not None and recommended > 0.0)
        or (seg_feedback and recommended_seg_weight is not None and recommended_seg_weight > 1.0)
    ):
        return False
    target_num_pairs = int(candidate.get("num_pairs") or 0)
    measured_num_pairs = int(row.get("measured_num_pairs") or 0)
    return target_num_pairs > 0 and measured_num_pairs == target_num_pairs


def _snerv_prelaunch_renderer_proof_row(row: Mapping[str, Any]) -> bool:
    if _family_key(str(row.get("family") or "")) != "snerv":
        return False
    if str(row.get("candidate_id") or "").strip():
        return False
    proof = row.get("snerv_renderer_nondegenerate_proof")
    if not isinstance(proof, Mapping):
        return False
    if proof.get("passed") is not True:
        return False
    measured_pairs = _first_present_int(
        proof,
        ("measured_num_pairs", "candidate_num_pairs", "num_pairs"),
    )
    if measured_pairs is None:
        measured_pairs = _first_present_int(
            row,
            ("measured_num_pairs", "candidate_num_pairs", "num_pairs"),
        )
    proof_blockers = [blocker for blocker in row.get("snerv_renderer_nondegenerate_blockers") or () if blocker]
    return bool(
        measured_pairs is not None
        and int(measured_pairs) >= SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT
        and not proof_blockers
    )


def _snerv_prelaunch_renderer_proof_applicable(
    *,
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
) -> bool:
    if _family_key(str(candidate.get("family") or "")) != "snerv":
        return False
    if int(candidate.get("num_pairs") or 0) < SNERV_RENDERER_NONDEGENERATE_MIN_PAIR_COUNT:
        return False
    return _snerv_prelaunch_renderer_proof_row(row)


def _sanitize_family_level_candidate_feedback(
    *,
    row: Mapping[str, Any],
    target_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    source_candidate_id = str(row.get("candidate_id") or "").strip()
    target_candidate_id = str(target_candidate.get("candidate_id") or "").strip()
    if _family_key(str(row.get("family") or "")) != "hi_nerv":
        feedback_kind = str(row.get("feedback_kind") or "").strip()
        is_upstream_eval = feedback_kind == "upstream_eval_gate"
        is_training_telemetry = feedback_kind == "training_telemetry"
        is_renderer_prelaunch_proof = _snerv_prelaunch_renderer_proof_applicable(
            candidate=target_candidate,
            row=row,
        )
        out["source_candidate_id"] = source_candidate_id
        out["target_candidate_id"] = str(target_candidate_id)
        out["candidate_id_match"] = False
        out["feedback_match_scope"] = (
            "family_upstream_eval_gate_context"
            if is_upstream_eval
            else "family_snerv_degenerate_renderer_training_telemetry_context"
            if is_training_telemetry
            else "family_snerv_prelaunch_renderer_proof"
            if is_renderer_prelaunch_proof
            else "family_full_video_mlx_response_context"
        )
        out["family_scope_matches_target"] = True
        out["scope_matches_candidate"] = False
        out["receiver_proof_attached"] = False
        out["full_video_local_prefilter_attached"] = False
        out["local_cpu_replay_gate_attached"] = False
        out["measured_archive_bytes"] = None
        out["measured_payload_bytes"] = None
        out["feedback_ready"] = False
        out["launch_control_feedback_ready"] = False
        if is_renderer_prelaunch_proof:
            out["feedback_kind"] = "snerv_prelaunch_renderer_proof"
            out["feedback_scope"] = "family_prelaunch_renderer_proof_min16"
            out["context_only"] = False
        else:
            out["context_only"] = True
        out["feedback_reuse_policy"] = (
            "family_upstream_eval_context_only_no_archive_receiver_replay_or_launch_authority"
            if is_upstream_eval
            else "family_snerv_degenerate_renderer_context_only_no_archive_receiver_replay_or_launch_authority"
            if is_training_telemetry
            else "family_snerv_prelaunch_renderer_proof_only_no_archive_receiver_replay_or_launch_authority"
            if is_renderer_prelaunch_proof
            else "family_full_video_context_only_no_archive_receiver_replay_or_launch_authority"
        )
        out.update(FALSE_AUTHORITY)
        return out
    source_official_score = _hinerv_feedback_official_control_score(row)
    target_official_score = _hinerv_official_control_score(target_candidate)
    source_official_superseded = bool(target_official_score > source_official_score)
    out["source_candidate_id"] = source_candidate_id
    out["target_candidate_id"] = str(target_candidate_id)
    out["candidate_id_match"] = False
    out["feedback_match_scope"] = "family_training_telemetry"
    out["family_scope_matches_target"] = True
    out["scope_matches_candidate"] = False
    out["receiver_proof_attached"] = False
    out["full_video_local_prefilter_attached"] = False
    out["local_cpu_replay_gate_attached"] = False
    out["measured_archive_bytes"] = None
    out["measured_payload_bytes"] = None
    out["source_official_control_score"] = int(source_official_score)
    out["target_official_control_score"] = int(target_official_score)
    out["source_official_control_superseded"] = source_official_superseded
    out["feedback_reuse_policy"] = "optimizer_stability_only_no_archive_receiver_or_replay_authority"
    if source_official_superseded:
        mutations = [str(item) for item in (out.get("recommended_launch_mutations") or []) if str(item).strip()]
        if HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION not in mutations:
            mutations.append(HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION)
        out["recommended_launch_mutations"] = mutations
    return out


def _decoder_weight_waterfill_index(
    sources: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        row = _normalize_decoder_weight_waterfill_source(source)
        family = _family_key(str(row.get("family") or ""))
        if not family:
            continue
        for candidate_key in _candidate_index_keys(row):
            index.setdefault((family, candidate_key), []).append(row)
    return {
        key: sorted(
            rows,
            key=lambda row: (
                int(row.get("full_video_coverage") is True),
                int(row.get("receiver_proof_ready") is True),
                int(row.get("group_count") or 0),
            ),
            reverse=True,
        )
        for key, rows in index.items()
    }


def _normalize_decoder_weight_waterfill_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if source.get("schema") != NERV_DECODER_WEIGHT_WATERFILL_SCHEMA:
        raise NervLongTrainingCampaignPlanError(
            "decoder_weight_waterfill_sources must have schema "
            f"{NERV_DECODER_WEIGHT_WATERFILL_SCHEMA}; got {source.get('schema')!r}"
        )
    path = (
        source.get("_decoder_weight_waterfill_plan_path")
        or source.get("decoder_weight_waterfill_plan_path")
        or source.get("path")
    )
    if not path:
        raise NervLongTrainingCampaignPlanError(
            "decoder_weight_waterfill_source missing _decoder_weight_waterfill_plan_path"
        )
    rows = source.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise NervLongTrainingCampaignPlanError("decoder_weight_waterfill_source must contain non-empty rows")
    out = dict(source)
    out["path"] = str(path)
    out["family"] = _family_key(str(source.get("family") or "hi_nerv"))
    out["group_count"] = int(source.get("group_count") or len(rows))
    full_video_coverage = source.get("full_video_coverage")
    if full_video_coverage is None:
        full_video_coverage = (
            source.get("_archive_ladder_full_video_coverage")
            if source.get("_archive_ladder_full_video_coverage") is not None
            else source.get("_archive_size_ladder_full_video_coverage")
        )
    out["full_video_coverage"] = bool(full_video_coverage)
    archive_sha = (
        source.get("archive_sha256")
        or source.get("_archive_size_ladder_archive_sha256")
        or source.get("_archive_ladder_archive_sha256")
    )
    if archive_sha:
        out.setdefault("archive_sha256", archive_sha)
    receiver_proof_path = (
        source.get("receiver_proof_path")
        or source.get("receiver_proof_report_path")
        or source.get("_archive_size_ladder_receiver_proof_path")
        or source.get("_archive_ladder_receiver_proof_path")
    )
    if receiver_proof_path:
        out.setdefault("receiver_proof_path", receiver_proof_path)
    receiver_proof_sha = (
        source.get("receiver_proof_sha256")
        or source.get("_archive_size_ladder_receiver_proof_sha256")
        or source.get("_archive_ladder_receiver_proof_sha256")
    )
    if receiver_proof_sha:
        out.setdefault("receiver_proof_sha256", receiver_proof_sha)
    runtime_ready = (
        source.get("runtime_consumption_proof_ready")
        or source.get("_archive_size_ladder_runtime_consumption_proof_ready")
        or source.get("_archive_ladder_runtime_consumption_proof_ready")
    )
    if runtime_ready and not str(source.get("receiver_proof_status") or "").strip():
        out["receiver_proof_status"] = "runtime_consumption_proof_ready"
    receiver_binding = _decoder_weight_waterfill_receiver_proof_binding(out)
    out["receiver_proof_binding"] = receiver_binding
    out["receiver_proof_ready"] = bool(receiver_binding["bound"])
    out["blockers"] = _dedupe(
        [
            *(str(blocker) for blocker in source.get("blockers") or () if str(blocker)),
            *(str(blocker) for blocker in receiver_binding.get("blockers") or () if str(blocker)),
        ]
    )
    return out


def _decoder_weight_waterfill_receiver_proof_ready(source: Mapping[str, Any]) -> bool:
    return bool(_decoder_weight_waterfill_receiver_proof_binding(source)["bound"])


def _decoder_weight_waterfill_receiver_proof_binding(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind waterfill receiver-proof status to a file-backed archive identity."""

    status = str(source.get("receiver_proof_status") or "").strip().lower()
    archive_sha = str(source.get("archive_sha256") or "").strip().lower()
    source_archive_sha = (
        str(source.get("_archive_size_ladder_archive_sha256") or source.get("_archive_ladder_archive_sha256") or "")
        .strip()
        .lower()
    )
    proof_path_raw = str(
        source.get("receiver_proof_path")
        or source.get("receiver_proof_report_path")
        or source.get("_archive_size_ladder_receiver_proof_path")
        or source.get("_archive_ladder_receiver_proof_path")
        or ""
    ).strip()
    expected_proof_sha = (
        str(
            source.get("receiver_proof_sha256")
            or source.get("_archive_size_ladder_receiver_proof_sha256")
            or source.get("_archive_ladder_receiver_proof_sha256")
            or ""
        )
        .strip()
        .lower()
    )
    blockers: list[str] = []
    proof_payload: Mapping[str, Any] = {}
    proof_sha: str | None = None
    proof_path: Path | None = None
    proof_archive_sha: str | None = None
    proof_runtime_ready = False

    if status not in TRUSTED_RECEIVER_PROOF_STATUSES:
        blockers.append("receiver_proof_not_satisfied")
    if not _is_sha256_hex(archive_sha):
        blockers.append("archive_sha256_missing_or_invalid")
    if source_archive_sha and (not _is_sha256_hex(source_archive_sha) or source_archive_sha != archive_sha):
        blockers.append("decoder_weight_waterfill_archive_sha256_mismatch_with_source_ladder")
    if not proof_path_raw:
        blockers.append("decoder_weight_waterfill_receiver_proof_path_missing")
    else:
        proof_path = _decoder_weight_waterfill_resolve_proof_path(
            proof_path_raw,
            source=source,
        )
        if not proof_path.is_file():
            blockers.append("decoder_weight_waterfill_receiver_proof_path_not_file")
        else:
            proof_sha = _sha256_file(proof_path)
            if expected_proof_sha and expected_proof_sha != proof_sha:
                blockers.append("decoder_weight_waterfill_receiver_proof_sha256_mismatch")
            try:
                payload = json.loads(proof_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append(f"decoder_weight_waterfill_receiver_proof_unreadable:{type(exc).__name__}")
            else:
                if isinstance(payload, Mapping):
                    proof_payload = payload
                    proof_archive_sha = _first_sha_value(
                        proof_payload,
                        (
                            "archive_sha256",
                            "archive_zip_sha256",
                            "receiver_archive_sha256",
                            "candidate_archive_sha256",
                        ),
                    )
                    proof_runtime_ready = any(
                        proof_payload.get(key) is True
                        for key in (
                            "runtime_consumption_proof_ready",
                            "runtime_consumption_proof_passed",
                            "receiver_archive_replay_verified",
                            "receiver_proof_passed",
                        )
                    )
                    if proof_archive_sha != archive_sha:
                        blockers.append("decoder_weight_waterfill_receiver_proof_archive_sha256_mismatch")
                    if not proof_runtime_ready:
                        blockers.append("decoder_weight_waterfill_receiver_proof_runtime_consumption_not_ready")
                else:
                    blockers.append("decoder_weight_waterfill_receiver_proof_payload_not_object")

    blockers = _dedupe(blockers)
    return {
        "schema": "nerv_decoder_weight_waterfill_receiver_proof_binding.v1",
        "status": status or None,
        "bound": not blockers,
        "archive_sha256": archive_sha or None,
        "source_archive_sha256": source_archive_sha or None,
        "proof_path": None if proof_path is None else proof_path.as_posix(),
        "proof_sha256": proof_sha,
        "expected_proof_sha256": expected_proof_sha or None,
        "proof_archive_sha256": proof_archive_sha,
        "proof_runtime_consumption_ready": bool(proof_runtime_ready),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_resolve_proof_path(
    value: str,
    *,
    source: Mapping[str, Any],
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    base_raw = (
        source.get("_decoder_weight_waterfill_source_path")
        or source.get("_decoder_weight_waterfill_plan_path")
        or source.get("path")
    )
    base = Path(str(base_raw)).expanduser().resolve(strict=False).parent if base_raw else Path.cwd()
    return (base / path).resolve(strict=False)


def _first_sha_value(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = str(payload.get(key) or "").strip().lower()
        if _is_sha256_hex(value):
            return value
    return None


def _is_sha256_hex(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _candidate_index_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw_values = [
        row.get("candidate_id"),
        row.get("_candidate_id"),
        row.get("_candidate_key"),
        row.get("_modelsize_row_id"),
        row.get("row_id"),
        row.get("planner_row_id"),
    ]
    keys: list[str] = []
    for raw in raw_values:
        text = str(raw or "").strip()
        if not text:
            continue
        keys.extend(_candidate_id_aliases(text))
    if not keys and _snerv_prelaunch_renderer_proof_row(row):
        keys.append("__snerv_family_prelaunch_renderer_proof__")
    return tuple(_dedupe(keys))


def _unique_index_row_count(
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> int:
    seen: set[int] = set()
    for rows in index.values():
        for row in rows:
            seen.add(id(row))
    return len(seen)


def _candidate_id_aliases(value: str) -> tuple[str, ...]:
    """Return source-row aliases without dropping the modelsize candidate id."""

    text = str(value or "").strip()
    if not text:
        return ()
    aliases = [text]
    has_double_colon = "::" in text
    if has_double_colon:
        parts = [part.strip() for part in text.split("::") if part.strip()]
        if len(parts) >= 3 and _family_key(parts[0]) in {"hi_nerv", "snerv"}:
            aliases.append(parts[1])
        elif parts:
            aliases.append(parts[-1])
    for marker in (
        ":hi_nerv_decoder_weight_waterfill:",
        ":hinerv_decoder_weight_waterfill:",
        ":snerv_decoder_weight_waterfill:",
    ):
        if marker in text:
            aliases.append(text.split(marker, 1)[0])
    if ":" in text and not has_double_colon:
        aliases.append(text.rsplit(":", 1)[-1])
        aliases.append(text.split(":", 1)[0])
    return tuple(_dedupe(alias for alias in aliases if alias))


def _decoder_weight_waterfill_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    rows = list((index or {}).get((_family_key(family), candidate_id)) or [])
    mismatch_row: dict[str, Any] = {}
    for row in rows:
        source_candidate = row.get("_modelsize_candidate")
        mismatch_blockers = _waterfill_modelsize_candidate_mismatch_blockers(
            candidate=candidate,
            waterfill_candidate=source_candidate,
        )
        if not mismatch_blockers:
            return dict(row)
        if not mismatch_row:
            mismatch_row = dict(row)
            existing = [str(v) for v in mismatch_row.get("blockers") or () if str(v)]
            mismatch_row["blockers"] = _dedupe([*existing, *mismatch_blockers])
            mismatch_row["receiver_proof_ready"] = False
    return mismatch_row


def _waterfill_modelsize_candidate_mismatch_blockers(
    *,
    candidate: Mapping[str, Any],
    waterfill_candidate: Any,
) -> list[str]:
    if not isinstance(waterfill_candidate, Mapping):
        return []
    blockers: list[str] = []
    for field in HINERV_WATERFILL_CANDIDATE_BINDING_FIELDS:
        if field not in candidate or field not in waterfill_candidate:
            continue
        left = _normalized_candidate_binding_value(candidate.get(field))
        right = _normalized_candidate_binding_value(waterfill_candidate.get(field))
        if left != right:
            blockers.append(f"decoder_weight_waterfill_modelsize_mismatch:{field}")
    return _dedupe(blockers)


def _normalized_candidate_binding_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_candidate_binding_value(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), 12)
    if value is None:
        return None
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    if as_float.is_integer():
        return int(as_float)
    return round(as_float, 12)


def _decoder_weight_waterfill_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    runner_admission = _decoder_weight_waterfill_runner_admission(row)
    saliency_replay_work_order = _decoder_weight_waterfill_saliency_replay_work_order(
        row=row,
        runner_admission=runner_admission,
    )
    allocator_basin_recovery_work_order = _decoder_weight_waterfill_allocator_basin_recovery_work_order(
        row=row,
        runner_admission=runner_admission,
    )
    return {
        "schema": "nerv_long_training_decoder_weight_waterfill_attachment.v1",
        "attached": True,
        "path": str(row.get("path")),
        "sha256": row.get("_decoder_weight_waterfill_plan_sha256"),
        "source_path": row.get("_decoder_weight_waterfill_source_path"),
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "candidate_keys": list(_candidate_index_keys(row)),
        "group_count": int(row.get("group_count") or 0),
        "full_video_coverage": bool(row.get("full_video_coverage")),
        "receiver_proof_ready": bool(row.get("receiver_proof_ready")),
        "receiver_proof_binding": (
            dict(row["receiver_proof_binding"])
            if isinstance(row.get("receiver_proof_binding"), Mapping)
            else _decoder_weight_waterfill_receiver_proof_binding(row)
        ),
        "_archive_size_ladder_source_schema": row.get("_archive_size_ladder_source_schema"),
        "_archive_size_ladder_row_index": row.get("_archive_size_ladder_row_index"),
        "_archive_size_ladder_runtime_consumption_proof_ready": row.get(
            "_archive_size_ladder_runtime_consumption_proof_ready"
        ),
        "runner_admission": runner_admission,
        "runner_admitted": bool(runner_admission["admitted"]),
        "saliency_replay_work_order": saliency_replay_work_order,
        "allocator_basin_recovery_work_order": allocator_basin_recovery_work_order,
        "blockers": list(row.get("blockers") or []),
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_saliency_replay_work_order(
    *,
    row: Mapping[str, Any],
    runner_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the executable next step when decoder-weight saliency blocks launch."""

    refusal_reasons = tuple(str(v) for v in runner_admission.get("refusal_reasons") or ())
    needs_saliency_replay = any(
        reason
        in {
            "decoder_weight_waterfill_full_video_coverage_missing",
            "decoder_weight_saliency_missing_for_some_groups",
            "full_video_coverage_missing",
        }
        for reason in refusal_reasons
    )
    source_path = str(row.get("_decoder_weight_waterfill_source_path") or "").strip()
    source_schema = str(row.get("_archive_size_ladder_source_schema") or "").strip()
    row_id = str(row.get("_modelsize_row_id") or row.get("candidate_id") or "").strip()
    candidate_id = str(row.get("candidate_id") or row_id).strip()
    if not needs_saliency_replay:
        return {
            "schema": "nerv_decoder_weight_saliency_replay_work_order.v1",
            "required": False,
            "reason": "decoder_weight_waterfill_runner_admission_does_not_need_saliency_replay",
            **FALSE_AUTHORITY,
        }
    blockers: list[str] = []
    if source_schema != "hinerv_archive_size_ladder.v1":
        blockers.append("decoder_weight_saliency_replay_source_ladder_missing")
    if not source_path:
        blockers.append("decoder_weight_saliency_replay_source_ladder_missing")
    if not row_id:
        blockers.append("decoder_weight_saliency_replay_row_id_missing")
    blockers = _dedupe(blockers)
    slug = _safe_id(candidate_id or row_id or "hinerv_row")
    utc = _utc_compact_timestamp()
    saliency_json = f".omx/research/hinerv_decoder_weight_saliency_replay_{utc}_{slug}_full600_codex.json"
    saliency_md = f".omx/research/hinerv_decoder_weight_saliency_replay_{utc}_{slug}_full600_codex.md"
    waterfill_json = f".omx/research/hinerv_archive_ladder_waterfill_{utc}_{slug}_full600_saliency_codex.json"
    waterfill_md = f".omx/research/hinerv_archive_ladder_waterfill_{utc}_{slug}_full600_saliency_codex.md"
    saliency_command = [
        "uv",
        "run",
        "python",
        "tools/build_hinerv_decoder_weight_saliency_replay.py",
        "--archive-ladder-json",
        source_path,
        "--row-id",
        row_id,
        "--max-pairs",
        "600",
        "--start-pair",
        "0",
        "--pair-stride",
        "1",
        "--device",
        "mps",
        "--output-json",
        saliency_json,
        "--output-md",
        saliency_md,
    ]
    waterfill_command = [
        "uv",
        "run",
        "python",
        "tools/build_hinerv_archive_ladder_waterfill.py",
        "--archive-ladder-json",
        source_path,
        "--saliency-json",
        saliency_json,
        "--output-json",
        waterfill_json,
        "--output-md",
        waterfill_md,
    ]
    return {
        "schema": "nerv_decoder_weight_saliency_replay_work_order.v1",
        "required": True,
        "row_id": row_id or None,
        "candidate_id": candidate_id or None,
        "source_archive_ladder_path": source_path or None,
        "coverage_required": "full600_start0_stride1",
        "preferred_device": "mps",
        "cpu_fidelity_note": (
            "MPS/MLX-family saliency replay is allocation signal only; local CPU replay "
            "and exact auth gates remain mandatory for promotion."
        ),
        "saliency_replay_command_argv": [] if blockers else saliency_command,
        "waterfill_rebuild_command_argv": [] if blockers else waterfill_command,
        "campaign_rebuild_hint_argv": [],
        "campaign_rebuild_required_inputs": (
            []
            if blockers
            else [
                "--hinerv-modelsize-budget",
                "--snerv-modelsize-budget",
                "--output-json",
                "--decoder-weight-waterfill-source",
            ]
        ),
        "campaign_rebuild_decoder_weight_waterfill_source": (waterfill_json if not blockers else None),
        "expected_output_saliency_json": saliency_json if not blockers else None,
        "expected_output_waterfill_json": waterfill_json if not blockers else None,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_allocator_basin_recovery_work_order(
    *,
    row: Mapping[str, Any],
    runner_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Route unfit HiNeRV rows back to training before allocator mutation."""

    blockers = {str(blocker) for blocker in row.get("blockers") or () if str(blocker)}
    refusal_reasons = {str(reason) for reason in runner_admission.get("refusal_reasons") or () if str(reason)}
    outside_basin = "score_loss_proxy_outside_allocator_linearization_basin" in blockers
    unfit_waterfill = "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin" in blockers
    if not (outside_basin or unfit_waterfill):
        return {
            "schema": "nerv_decoder_weight_allocator_basin_recovery_work_order.v1",
            "required": False,
            "reason": "decoder_weight_waterfill_row_inside_allocator_basin_or_missing_basin_evidence",
            **FALSE_AUTHORITY,
        }
    candidate_id = str(row.get("_modelsize_row_id") or row.get("candidate_id") or "").strip()
    source_path = str(row.get("_decoder_weight_waterfill_source_path") or "").strip()
    row_id = str(row.get("_modelsize_row_id") or candidate_id).strip()
    recovery_blockers: list[str] = []
    if not candidate_id:
        recovery_blockers.append("allocator_basin_recovery_candidate_id_missing")
    command: list[str] = []
    if candidate_id:
        command = [
            "uv",
            "run",
            "--extra",
            "dev",
            "--extra",
            "runtime",
            "--extra",
            "mlx",
            "python",
            "tools/run_compact_renderer_mlx_spine_runner.py",
            "--execute-family",
            "hi_nerv",
            "--modelsize-candidate-id",
            candidate_id,
            "--hi-nerv-optimizer-policy",
            "pr95_curriculum",
            "--optimizer-kind",
            "adamw",
            "--coder-aware-qat",
            "--mlx-prefilter-scorer-device",
            "gpu",
            "--run-post-export-materializers",
        ]
    return {
        "schema": "nerv_decoder_weight_allocator_basin_recovery_work_order.v1",
        "required": True,
        "reason": "decoder_weight_waterfill_requires_fit_before_allocator_mutation",
        "candidate_id": candidate_id or None,
        "row_id": row_id or None,
        "source_decoder_weight_waterfill_path": str(row.get("path") or "") or None,
        "source_decoder_weight_waterfill_report_path": source_path or None,
        "observed_blockers": _dedupe([*sorted(blockers), *sorted(refusal_reasons)]),
        "next_stage": "pr95_grade_scoreaware_fit_then_replay_saliency_and_waterfill",
        "command_argv": [] if recovery_blockers else command,
        "blockers": recovery_blockers,
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_runner_admitted(row: Mapping[str, Any]) -> bool:
    return bool(_decoder_weight_waterfill_runner_admission(row)["admitted"])


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_")
    return out or "row"


def _auto_bytecap_candidate_label(candidate_id: str) -> str:
    """Make auto-bytecap planner rows unique without changing runner selection."""

    return f"auto_bytecap::{_safe_id(candidate_id or 'candidate')}"


def _utc_compact_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _decoder_weight_waterfill_runner_admission(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = {str(blocker) for blocker in row.get("blockers") or ()}
    refusal_reasons: list[str] = []
    if row.get("full_video_coverage") is not True:
        refusal_reasons.append("decoder_weight_waterfill_full_video_coverage_missing")
    if row.get("receiver_proof_ready") is not True:
        refusal_reasons.append("decoder_weight_waterfill_receiver_proof_not_ready")
    if not _is_sha256_hex(row.get("archive_sha256")):
        refusal_reasons.append("decoder_weight_waterfill_archive_sha256_missing_or_invalid")
    binding = row.get("receiver_proof_binding")
    if isinstance(binding, Mapping):
        refusal_reasons.extend(str(blocker) for blocker in binding.get("blockers") or ())
    for blocker in (
        "decoder_weight_saliency_missing_for_some_groups",
        "decoder_weight_saliency_replay_has_blockers",
        "score_loss_proxy_outside_allocator_linearization_basin",
        "decoder_weight_waterfill_not_admissible_from_unfit_scorer_basin",
        "full_video_coverage_missing",
        "receiver_proof_not_satisfied",
        "archive_sha256_missing",
        "archive_sha256_invalid",
    ):
        if blocker in blockers:
            refusal_reasons.append(blocker)
    refusal_reasons.extend(
        blocker
        for blocker in blockers
        if blocker.startswith("decoder_weight_waterfill_receiver_proof_")
        or blocker == "decoder_weight_waterfill_archive_sha256_mismatch_with_source_ladder"
    )
    return {
        "schema": "nerv_decoder_weight_waterfill_runner_admission.v1",
        "admitted": not refusal_reasons,
        "mode": (
            "runner_training_pressure_and_export_mutation" if not refusal_reasons else "advisory_learning_signal_only"
        ),
        "refusal_reasons": _dedupe(refusal_reasons),
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_unattached_sources(
    *,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
    campaign_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return source-backed waterfill rows that no campaign row consumed.

    Decoder-weight waterfill is candidate-shape specific. Dropping unmatched
    sources silently loses useful signal and invites false attachments later.
    The planner records them as non-authoritative diagnostics instead.
    """

    attached_paths = {
        str(plan.get("path") or "")
        for row in campaign_rows
        if isinstance((plan := row.get("decoder_weight_waterfill_plan")), Mapping) and plan.get("attached") is True
    }
    target_candidates_by_family: dict[str, list[str]] = {}
    for row in campaign_rows:
        family = _family_key(str(row.get("family") or ""))
        candidate_id = str(row.get("candidate_id") or "").strip()
        if family and candidate_id:
            target_candidates_by_family.setdefault(family, []).append(candidate_id)

    by_path: dict[str, dict[str, Any]] = {}
    for (family, _candidate_key), source_rows in (index or {}).items():
        for source in source_rows:
            path = str(source.get("path") or "")
            if not path or path in attached_paths or path in by_path:
                continue
            family_key = _family_key(str(source.get("family") or family))
            by_path[path] = {
                "schema": ("nerv_long_training_unattached_decoder_weight_waterfill_source.v1"),
                "attached": False,
                "reason": "no_matching_campaign_candidate_id",
                "path": path,
                "sha256": source.get("_decoder_weight_waterfill_plan_sha256"),
                "source_path": source.get("_decoder_weight_waterfill_source_path"),
                "family": family_key,
                "source_candidate_id": source.get("candidate_id"),
                "candidate_keys": list(_candidate_index_keys(source)),
                "target_candidate_ids": sorted(_dedupe(target_candidates_by_family.get(family_key, []))),
                "group_count": int(source.get("group_count") or 0),
                "full_video_coverage": bool(source.get("full_video_coverage")),
                "receiver_proof_ready": bool(source.get("receiver_proof_ready")),
                "blockers": list(source.get("blockers") or []),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
    return sorted(by_path.values(), key=lambda item: str(item.get("path") or ""))


def _archive_section_telemetry_index(
    sources: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source in sources:
        row = _normalize_archive_section_telemetry_source(source)
        family = _family_key(str(row.get("family") or "hi_nerv"))
        if family != "hi_nerv":
            continue
        for candidate_key in _candidate_index_keys(row):
            index.setdefault((family, candidate_key), []).append(row)
    return {
        key: sorted(
            rows,
            key=lambda row: (
                int(_archive_section_telemetry_runner_admitted(row)),
                int(row.get("receiver_cache_quality_gate_passed") is True),
                int(row.get("archive_under_hard_byte_ceiling") is True),
                int(row.get("num_pairs") or 0),
                -int(row.get("archive_zip_bytes") or row.get("inner_payload_bytes") or 0),
            ),
            reverse=True,
        )
        for key, rows in index.items()
    }


def _normalize_archive_section_telemetry_source(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if source.get("schema") != HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA:
        raise NervLongTrainingCampaignPlanError(
            "archive_section_telemetry_sources must have schema "
            f"{HINERV_ARCHIVE_SECTION_TELEMETRY_SCHEMA}; got {source.get('schema')!r}"
        )
    path = (
        source.get("_archive_section_telemetry_path")
        or source.get("archive_section_telemetry_path")
        or source.get("path")
    )
    if not path:
        raise NervLongTrainingCampaignPlanError(
            "archive_section_telemetry_source missing _archive_section_telemetry_path"
        )
    out = dict(source)
    out["path"] = str(path)
    out.setdefault("_archive_section_telemetry_source_path", str(path))
    out.setdefault("family", "hi_nerv")
    out.setdefault(
        "candidate_id",
        source.get("row_id") or source.get("_modelsize_row_id"),
    )
    if source.get("_archive_section_telemetry_sha256") is None:
        telemetry_path = Path(str(path)).expanduser().resolve(strict=False)
        if telemetry_path.is_file():
            out["_archive_section_telemetry_sha256"] = _sha256_file(telemetry_path)
    archive_zip_bytes = _positive_int_or_none(
        out.get("archive_zip_bytes") or out.get("measured_archive_bytes") or out.get("archive_bytes")
    )
    if archive_zip_bytes is not None:
        out["archive_zip_bytes"] = archive_zip_bytes
    inner_payload_bytes = _positive_int_or_none(out.get("inner_payload_bytes"))
    if inner_payload_bytes is not None:
        out["inner_payload_bytes"] = inner_payload_bytes
    section_payload_bytes = _positive_int_or_none(out.get("section_payload_bytes"))
    if section_payload_bytes is not None:
        out["section_payload_bytes"] = section_payload_bytes
    out["section_bytes"] = _archive_section_telemetry_section_bytes(out)
    if out.get("runtime_consumption_proof_ready") is True and not str(out.get("receiver_proof_status") or "").strip():
        out["receiver_proof_status"] = "runtime_consumption_proof_ready"
    if out.get("quality_gate_passed") is not None and out.get("receiver_cache_quality_gate_passed") is None:
        out["receiver_cache_quality_gate_passed"] = bool(out.get("quality_gate_passed"))
    if out.get("quality_gate_verdict") and out.get("receiver_cache_quality_gate_verdict") is None:
        out["receiver_cache_quality_gate_verdict"] = out.get("quality_gate_verdict")
    if out.get("report_path") and out.get("receiver_cache_quality_report_path") is None:
        out["receiver_cache_quality_report_path"] = out.get("report_path")
    out["receiver_proof_binding"] = _archive_section_telemetry_receiver_proof_binding(out)
    out["receiver_cache_quality_binding"] = _archive_section_telemetry_cache_quality_binding(out)
    out["archive_under_hard_byte_ceiling"] = _archive_section_telemetry_under_ceiling(out)
    blockers = [
        *(str(blocker) for blocker in source.get("blockers") or () if str(blocker)),
        *_archive_section_telemetry_static_blockers(out),
        *(str(blocker) for blocker in out["receiver_proof_binding"].get("blockers") or ()),
        *(str(blocker) for blocker in out["receiver_cache_quality_binding"].get("blockers") or ()),
    ]
    out["blockers"] = _dedupe(blockers)
    return out


def _archive_section_telemetry_static_blockers(
    row: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if row.get("profile_ready") is not True:
        blockers.append("hinerv_archive_section_telemetry_not_profile_ready")
    sections = row.get("sections")
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        blockers.append("hinerv_archive_section_telemetry_sections_missing")
    elif not any(
        isinstance(section, Mapping)
        and str(section.get("name") or "") == "decoder_state"
        and _positive_int_or_none(section.get("bytes")) is not None
        for section in sections
    ):
        blockers.append("hinerv_archive_section_telemetry_decoder_state_missing")
    if _positive_int_or_none(row.get("archive_zip_bytes")) is None:
        blockers.append("hinerv_archive_section_telemetry_archive_zip_bytes_missing")
    if _positive_int_or_none(row.get("num_pairs")) != 600:
        blockers.append("hinerv_archive_section_telemetry_full600_missing")
    for key in _AUTHORITY_TRUE_KEYS:
        if row.get(key) is True:
            blockers.append(f"hinerv_archive_section_telemetry_authority_flag_true:{key}")
    return blockers


def _archive_section_telemetry_section_bytes(row: Mapping[str, Any]) -> dict[str, int]:
    sections = row.get("sections_with_zip_overhead") or row.get("sections")
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes)):
        return {}
    out: dict[str, int] = {}
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            continue
        name = str(section.get("name") or f"section_{index:04d}").strip()
        nbytes = _positive_int_or_none(section.get("bytes"))
        if name and nbytes is not None:
            out[name] = nbytes
    return dict(sorted(out.items()))


def _archive_section_telemetry_under_ceiling(row: Mapping[str, Any]) -> bool | None:
    archive_bytes = _positive_int_or_none(row.get("archive_zip_bytes"))
    hard_ceiling = _positive_int_or_none(row.get("hard_byte_ceiling"))
    if archive_bytes is None or hard_ceiling is None:
        return None
    return int(archive_bytes) <= int(hard_ceiling)


def _archive_section_telemetry_receiver_proof_binding(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(source.get("receiver_proof_status") or "").strip().lower()
    archive_sha = str(source.get("archive_sha256") or "").strip().lower()
    proof_path_raw = str(source.get("receiver_proof_path") or source.get("receiver_proof_report_path") or "").strip()
    expected_proof_sha = str(source.get("receiver_proof_sha256") or "").strip().lower()
    blockers: list[str] = []
    proof_path: Path | None = None
    proof_sha: str | None = None
    proof_archive_sha: str | None = None
    proof_runtime_ready = False

    if status not in TRUSTED_RECEIVER_PROOF_STATUSES:
        blockers.append("hinerv_archive_section_telemetry_receiver_proof_not_satisfied")
    if not _is_sha256_hex(archive_sha):
        blockers.append("hinerv_archive_section_telemetry_archive_sha256_missing_or_invalid")
    if not proof_path_raw:
        blockers.append("hinerv_archive_section_telemetry_receiver_proof_path_missing")
    else:
        proof_path = _archive_section_telemetry_resolve_path(
            proof_path_raw,
            source=source,
        )
        if not proof_path.is_file():
            blockers.append("hinerv_archive_section_telemetry_receiver_proof_path_not_file")
        else:
            proof_sha = _sha256_file(proof_path)
            if expected_proof_sha and expected_proof_sha != proof_sha:
                blockers.append("hinerv_archive_section_telemetry_receiver_proof_sha256_mismatch")
            try:
                payload = json.loads(proof_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append(f"hinerv_archive_section_telemetry_receiver_proof_unreadable:{type(exc).__name__}")
            else:
                if not isinstance(payload, Mapping):
                    blockers.append("hinerv_archive_section_telemetry_receiver_proof_payload_not_object")
                else:
                    proof_archive_sha = _first_sha_value(
                        payload,
                        (
                            "archive_sha256",
                            "archive_zip_sha256",
                            "candidate_archive_sha256",
                        ),
                    )
                    proof_runtime_ready = any(
                        payload.get(key) is True
                        for key in (
                            "runtime_consumption_proof_ready",
                            "runtime_consumption_proof_passed",
                            "receiver_archive_replay_verified",
                            "receiver_proof_passed",
                        )
                    )
                    if proof_archive_sha != archive_sha:
                        blockers.append("hinerv_archive_section_telemetry_receiver_proof_archive_sha256_mismatch")
                    if not proof_runtime_ready:
                        blockers.append("hinerv_archive_section_telemetry_receiver_proof_runtime_consumption_not_ready")
    blockers = _dedupe(blockers)
    return {
        "schema": "hinerv_archive_section_telemetry_receiver_proof_binding.v1",
        "status": status or None,
        "bound": not blockers,
        "archive_sha256": archive_sha or None,
        "proof_path": None if proof_path is None else proof_path.as_posix(),
        "proof_sha256": proof_sha,
        "expected_proof_sha256": expected_proof_sha or None,
        "proof_archive_sha256": proof_archive_sha,
        "proof_runtime_consumption_ready": proof_runtime_ready,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _archive_section_telemetry_cache_quality_binding(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    path_raw = str(source.get("receiver_cache_quality_report_path") or "").strip()
    expected_sha = str(source.get("receiver_cache_quality_report_sha256") or "").strip().lower()
    blockers: list[str] = []
    path: Path | None = None
    actual_sha: str | None = None
    gate_passed = bool(source.get("receiver_cache_quality_gate_passed") is True)
    if not gate_passed:
        blockers.append("hinerv_archive_section_telemetry_receiver_cache_quality_gate_not_passed")
    if not path_raw:
        blockers.append("hinerv_archive_section_telemetry_receiver_cache_quality_report_path_missing")
    else:
        path = _archive_section_telemetry_resolve_path(path_raw, source=source)
        if not path.is_file():
            blockers.append("hinerv_archive_section_telemetry_receiver_cache_quality_report_path_not_file")
        else:
            actual_sha = _sha256_file(path)
            if expected_sha and expected_sha != actual_sha:
                blockers.append("hinerv_archive_section_telemetry_receiver_cache_quality_report_sha256_mismatch")
    blockers = _dedupe(blockers)
    return {
        "schema": "hinerv_archive_section_telemetry_receiver_cache_quality_binding.v1",
        "bound": not blockers,
        "quality_gate_passed": gate_passed,
        "quality_gate_verdict": source.get("receiver_cache_quality_gate_verdict"),
        "report_path": None if path is None else path.as_posix(),
        "report_sha256": actual_sha,
        "expected_report_sha256": expected_sha or None,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _archive_section_telemetry_resolve_path(
    value: str,
    *,
    source: Mapping[str, Any],
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    base_raw = source.get("_archive_section_telemetry_source_path") or source.get("path")
    base = Path(str(base_raw)).expanduser().resolve(strict=False).parent if base_raw else Path.cwd()
    return (base / path).resolve(strict=False)


def _archive_section_telemetry_for(
    *,
    candidate: Mapping[str, Any],
    family: str,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        return {}
    rows = list((index or {}).get((_family_key(family), candidate_id)) or [])
    mismatch_row: dict[str, Any] = {}
    for row in rows:
        candidate_row = dict(row)
        candidate_row.setdefault("hard_byte_ceiling", candidate.get("hard_byte_ceiling"))
        source_candidate = candidate_row.get("_modelsize_candidate")
        mismatch_blockers = _waterfill_modelsize_candidate_mismatch_blockers(
            candidate=candidate,
            waterfill_candidate=source_candidate,
        )
        if mismatch_blockers:
            if not mismatch_row:
                mismatch_row = dict(candidate_row)
                mismatch_row["blockers"] = _dedupe(
                    [
                        *(str(v) for v in mismatch_row.get("blockers") or () if str(v)),
                        *(
                            blocker.replace(
                                "decoder_weight_waterfill",
                                "archive_section_telemetry",
                            )
                            for blocker in mismatch_blockers
                        ),
                    ]
                )
            continue
        under_ceiling = _archive_section_telemetry_under_ceiling(candidate_row)
        candidate_row["archive_under_hard_byte_ceiling"] = under_ceiling
        if under_ceiling is False:
            candidate_row["blockers"] = _dedupe(
                [
                    *(str(v) for v in candidate_row.get("blockers") or () if str(v)),
                    "hinerv_archive_section_telemetry_archive_over_hard_byte_ceiling",
                ]
            )
        elif under_ceiling is None:
            candidate_row["blockers"] = _dedupe(
                [
                    *(str(v) for v in candidate_row.get("blockers") or () if str(v)),
                    "hinerv_archive_section_telemetry_hard_byte_ceiling_binding_missing",
                ]
            )
        return candidate_row
    return mismatch_row


def _archive_section_telemetry_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    runner_admission = _archive_section_telemetry_runner_admission(row)
    return {
        "schema": "nerv_long_training_archive_section_telemetry_attachment.v1",
        "attached": True,
        "path": str(row.get("path")),
        "sha256": row.get("_archive_section_telemetry_sha256"),
        "source_path": row.get("_archive_section_telemetry_source_path"),
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "candidate_keys": list(_candidate_index_keys(row)),
        "num_pairs": _positive_int_or_none(row.get("num_pairs")),
        "archive_sha256": row.get("archive_sha256"),
        "archive_zip_bytes": _positive_int_or_none(row.get("archive_zip_bytes")),
        "inner_payload_bytes": _positive_int_or_none(row.get("inner_payload_bytes")),
        "section_payload_bytes": _positive_int_or_none(row.get("section_payload_bytes")),
        "section_bytes": dict(row.get("section_bytes") or {}),
        "section_count": len(row.get("section_bytes") or {}),
        "section_names": list((row.get("section_bytes") or {}).keys()),
        "decoder_state_section_present": "decoder_state" in dict(row.get("section_bytes") or {}),
        "decoder_codec": row.get("decoder_codec"),
        "latent_codec": row.get("latent_codec"),
        "profile_ready": bool(row.get("profile_ready")),
        "hard_byte_ceiling": _positive_int_or_none(row.get("hard_byte_ceiling")),
        "archive_under_hard_byte_ceiling": row.get("archive_under_hard_byte_ceiling"),
        "receiver_proof_binding": (
            dict(row["receiver_proof_binding"])
            if isinstance(row.get("receiver_proof_binding"), Mapping)
            else _archive_section_telemetry_receiver_proof_binding(row)
        ),
        "receiver_cache_quality_binding": (
            dict(row["receiver_cache_quality_binding"])
            if isinstance(row.get("receiver_cache_quality_binding"), Mapping)
            else _archive_section_telemetry_cache_quality_binding(row)
        ),
        "runner_admission": runner_admission,
        "runner_admitted": bool(runner_admission["admitted"]),
        "blockers": list(row.get("blockers") or []),
        **FALSE_AUTHORITY,
    }


def _archive_section_telemetry_missing_metadata() -> dict[str, Any]:
    return {
        "schema": "nerv_long_training_archive_section_telemetry_attachment.v1",
        "attached": False,
        "reason": "no_matching_hinerv_archive_section_telemetry",
        **FALSE_AUTHORITY,
    }


def _archive_section_telemetry_runner_admitted(row: Mapping[str, Any]) -> bool:
    return bool(_archive_section_telemetry_runner_admission(row)["admitted"])


def _archive_section_telemetry_runner_admission(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    blockers = {str(blocker) for blocker in row.get("blockers") or () if str(blocker)}
    refusal_reasons: list[str] = []
    if row.get("profile_ready") is not True:
        refusal_reasons.append("hinerv_archive_section_telemetry_not_profile_ready")
    if not row.get("section_bytes"):
        refusal_reasons.append("hinerv_archive_section_telemetry_sections_missing")
    if row.get("archive_under_hard_byte_ceiling") is not True:
        refusal_reasons.append("hinerv_archive_section_telemetry_archive_not_under_hard_byte_ceiling")
    receiver_binding = row.get("receiver_proof_binding")
    if not isinstance(receiver_binding, Mapping) or receiver_binding.get("bound") is not True:
        refusal_reasons.append("hinerv_archive_section_telemetry_receiver_proof_not_bound")
    if isinstance(receiver_binding, Mapping):
        refusal_reasons.extend(str(v) for v in receiver_binding.get("blockers") or ())
    cache_binding = row.get("receiver_cache_quality_binding")
    if not isinstance(cache_binding, Mapping) or cache_binding.get("bound") is not True:
        refusal_reasons.append("hinerv_archive_section_telemetry_receiver_cache_quality_not_bound")
    if isinstance(cache_binding, Mapping):
        refusal_reasons.extend(str(v) for v in cache_binding.get("blockers") or ())
    refusal_reasons.extend(sorted(blockers))
    refusal_reasons = _dedupe(refusal_reasons)
    return {
        "schema": "hinerv_archive_section_telemetry_runner_admission.v1",
        "admitted": not refusal_reasons,
        "mode": ("runner_train_time_section_qat_pressure" if not refusal_reasons else "advisory_byte_profile_only"),
        "refusal_reasons": refusal_reasons,
        **FALSE_AUTHORITY,
    }


def _archive_section_telemetry_unattached_sources(
    *,
    index: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]] | None,
    campaign_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attached_paths = {
        str(plan.get("path") or "")
        for row in campaign_rows
        if isinstance((plan := row.get("archive_section_telemetry")), Mapping) and plan.get("attached") is True
    }
    target_candidates_by_family: dict[str, list[str]] = {}
    for row in campaign_rows:
        family = _family_key(str(row.get("family") or ""))
        candidate_id = str(row.get("candidate_id") or "").strip()
        if family and candidate_id:
            target_candidates_by_family.setdefault(family, []).append(candidate_id)

    by_path: dict[str, dict[str, Any]] = {}
    for (family, _candidate_key), source_rows in (index or {}).items():
        for source in source_rows:
            path = str(source.get("path") or "")
            if not path or path in attached_paths or path in by_path:
                continue
            family_key = _family_key(str(source.get("family") or family))
            by_path[path] = {
                "schema": ("nerv_long_training_unattached_archive_section_telemetry_source.v1"),
                "attached": False,
                "reason": "no_matching_campaign_candidate_id",
                "path": path,
                "sha256": source.get("_archive_section_telemetry_sha256"),
                "source_path": source.get("_archive_section_telemetry_source_path"),
                "family": family_key,
                "source_candidate_id": source.get("candidate_id"),
                "candidate_keys": list(_candidate_index_keys(source)),
                "target_candidate_ids": sorted(_dedupe(target_candidates_by_family.get(family_key, []))),
                "archive_zip_bytes": _positive_int_or_none(source.get("archive_zip_bytes")),
                "archive_under_hard_byte_ceiling": source.get("archive_under_hard_byte_ceiling"),
                "receiver_cache_quality_gate_passed": bool(source.get("receiver_cache_quality_gate_passed")),
                "blockers": list(source.get("blockers") or []),
                **FALSE_AUTHORITY,
            }
    return sorted(by_path.values(), key=lambda item: str(item.get("path") or ""))


def _hinerv_feedback_launch_adjustment(
    *,
    feedback: Mapping[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    if not feedback:
        return {
            "schema": "hinerv_feedback_launch_adjustment.v1",
            "applied": False,
            "reason": "no_candidate_feedback",
            "policy_logic": HINERV_POSE_INSTABILITY_POLICY_LOGIC,
            "learning_rate": float(learning_rate),
            "segnet_distillation_weight": 1.0,
            "pose_distillation_weight": 1.0,
            **FALSE_AUTHORITY,
        }
    launch_control_ready = _feedback_launch_control_ready(feedback)
    if not launch_control_ready:
        return {
            "schema": "hinerv_feedback_launch_adjustment.v1",
            "applied": False,
            "reason": "feedback_not_launch_control_ready",
            "policy_logic": HINERV_POSE_INSTABILITY_POLICY_LOGIC,
            "source_feedback_kind": feedback.get("feedback_kind"),
            "source_feedback_scope": feedback.get("feedback_scope"),
            "feedback_ready": feedback.get("feedback_ready"),
            "launch_control_feedback_ready": feedback.get("launch_control_feedback_ready"),
            "learning_rate": float(learning_rate),
            "segnet_distillation_weight": 1.0,
            "pose_distillation_weight": 1.0,
            **FALSE_AUTHORITY,
        }
    observed = _float_or_none(feedback.get("observed_learning_rate"))
    recommended = _float_or_none(feedback.get("recommended_learning_rate"))
    pose_instability = bool(feedback.get("pose_instability_detected"))
    lr_floor = HINERV_POSE_INSTABILITY_LOW_LR_FLOOR
    repeated_low_lr_instability = bool(pose_instability and observed is not None and observed <= lr_floor)
    lower_learning_rate_applied = bool(
        pose_instability
        and not repeated_low_lr_instability
        and recommended is not None
        and recommended > 0.0
        and recommended < float(learning_rate)
    )
    pose_protected_pathway_applied = bool(repeated_low_lr_instability)
    launch_mutations: list[str] = []
    receiver_class_survival_mutations = [
        str(mutation)
        for mutation in (feedback.get("recommended_launch_mutations") or [])
        if str(mutation)
        in {
            "increase_hi_nerv_receiver_class_survival_pressure",
            "disable_hi_nerv_byte_feedback_learning_from_receiver_collapsed_export",
            "rerun_hi_nerv_short_probe_with_receiver_cache_quality_gate",
        }
    ]
    if lower_learning_rate_applied:
        launch_mutations.extend(list(feedback.get("recommended_launch_mutations") or []))
    if pose_protected_pathway_applied:
        launch_mutations.append("enable_pose_distillation_huber_from_repeated_low_lr_instability")
    seg_stagnation = bool(feedback.get("seg_stagnation_detected"))
    pose_tail_burst = bool(feedback.get("pose_tail_burst_detected"))
    recommended_seg_weight = _float_or_none(feedback.get("recommended_segnet_distillation_weight"))
    recommended_pose_weight = _float_or_none(feedback.get("recommended_pose_distillation_weight"))
    segnet_weight_applied = bool(seg_stagnation and recommended_seg_weight is not None and recommended_seg_weight > 1.0)
    pose_weight_applied = bool(
        pose_tail_burst and recommended_pose_weight is not None and recommended_pose_weight > 1.0
    )
    if segnet_weight_applied:
        launch_mutations.extend(
            mutation
            for mutation in (feedback.get("recommended_launch_mutations") or [])
            if mutation not in launch_mutations
        )
    if pose_weight_applied:
        launch_mutations.extend(
            mutation
            for mutation in (feedback.get("recommended_launch_mutations") or [])
            if mutation not in launch_mutations
        )
    official_control_superseded = bool(feedback.get("source_official_control_superseded"))
    if official_control_superseded and (HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION not in launch_mutations):
        launch_mutations.append(HINERV_OFFICIAL_CONTROL_SUPERSESSION_MUTATION)
    for mutation in receiver_class_survival_mutations:
        if mutation not in launch_mutations:
            launch_mutations.append(mutation)
    receiver_class_survival_applied = bool(receiver_class_survival_mutations)
    applied = bool(
        lower_learning_rate_applied
        or pose_protected_pathway_applied
        or segnet_weight_applied
        or pose_weight_applied
        or official_control_superseded
        or receiver_class_survival_applied
    )
    return {
        "schema": "hinerv_feedback_launch_adjustment.v1",
        "applied": applied,
        "lower_learning_rate_applied": lower_learning_rate_applied,
        "pose_protected_pathway_applied": pose_protected_pathway_applied,
        "segnet_weight_applied": segnet_weight_applied,
        "pose_weight_applied": pose_weight_applied,
        "official_control_superseded": official_control_superseded,
        "receiver_class_survival_applied": receiver_class_survival_applied,
        "policy_logic": HINERV_POSE_INSTABILITY_POLICY_LOGIC,
        "reason": (
            "pose_instability_recommended_lower_learning_rate"
            if lower_learning_rate_applied
            else (
                "repeated_pose_instability_at_low_lr_pose_protected_pathway"
                if pose_protected_pathway_applied
                else (
                    "segnet_stagnation_recommended_higher_segnet_weight"
                    if segnet_weight_applied
                    else (
                        "pose_tail_recommended_higher_pose_weight"
                        if pose_weight_applied
                        else (
                            "official_hinerv_controls_supersede_source_feedback_run"
                            if official_control_superseded
                            else (
                                "receiver_class_survival_probe_mutation_applied"
                                if receiver_class_survival_applied
                                else (
                                    "pose_instability_feedback_without_lower_lr"
                                    if pose_instability
                                    else (
                                        "pose_tail_burst_requires_prioritized_pair_indices"
                                        if pose_tail_burst
                                        else ("feedback_does_not_request_launch_adjustment")
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "source_feedback_kind": feedback.get("feedback_kind"),
        "source_feedback_scope": feedback.get("feedback_scope"),
        "feedback_ready": feedback.get("feedback_ready"),
        "launch_control_feedback_ready": launch_control_ready,
        "pose_instability_detected": pose_instability,
        "pose_tail_burst_detected": pose_tail_burst,
        "seg_stagnation_detected": seg_stagnation,
        "observed_learning_rate": observed,
        "low_learning_rate_floor": lr_floor,
        "repeated_low_lr_pose_instability": repeated_low_lr_instability,
        "requested_learning_rate": float(learning_rate),
        "recommended_learning_rate": recommended,
        "learning_rate": float(recommended if lower_learning_rate_applied else learning_rate),
        "recommended_segnet_distillation_weight": recommended_seg_weight,
        "segnet_distillation_weight": float(recommended_seg_weight if segnet_weight_applied else 1.0),
        "recommended_pose_distillation_weight": recommended_pose_weight,
        "pose_distillation_weight": float(recommended_pose_weight if pose_weight_applied else 1.0),
        "pose_distillation_loss": (HINERV_POSE_PROTECTED_LOSS if pose_protected_pathway_applied else "mse"),
        "pose_distillation_huber_delta": (
            HINERV_POSE_PROTECTED_HUBER_DELTA if pose_protected_pathway_applied else None
        ),
        "launch_mutations": launch_mutations,
        **FALSE_AUTHORITY,
    }


def _feedback_launch_control_ready(feedback: Mapping[str, Any]) -> bool:
    if feedback.get("launch_control_feedback_ready") is True:
        return True
    if (
        str(feedback.get("feedback_kind") or "") == "training_telemetry"
        and str(feedback.get("feedback_scope") or "") == "full600_training_telemetry"
    ):
        recommended_lr = _float_or_none(feedback.get("recommended_learning_rate"))
        recommended_seg_weight = _float_or_none(feedback.get("recommended_segnet_distillation_weight"))
        recommended_pose_weight = _float_or_none(feedback.get("recommended_pose_distillation_weight"))
        pose_ready = (
            feedback.get("pose_instability_detected") is True and recommended_lr is not None and recommended_lr > 0.0
        )
        seg_ready = (
            feedback.get("seg_stagnation_detected") is True
            and recommended_seg_weight is not None
            and recommended_seg_weight > 1.0
        )
        tail_ready = feedback.get("pose_tail_burst_detected") is True
        pose_weight_ready = tail_ready and recommended_pose_weight is not None and recommended_pose_weight > 1.0
        return bool(pose_ready or seg_ready or tail_ready or pose_weight_ready)
    if feedback.get("feedback_ready") is False:
        return False
    return bool(feedback)


def _hinerv_source_faithfulness_controls(
    *,
    candidate: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    target_score = _hinerv_official_control_score(candidate)
    source_score = _hinerv_feedback_official_control_score(feedback) if feedback else target_score
    target_blockers = _hinerv_official_control_blockers(candidate)
    return {
        "schema": "hinerv_source_faithfulness_controls.v1",
        "target_candidate_id": str(candidate.get("candidate_id") or ""),
        "target_uses_hierarchical_feature_grid": bool(candidate.get("use_hierarchical_feature_grid")),
        "target_uses_convnext_blocks": bool(candidate.get("use_convnext_blocks")),
        "target_official_control_score": int(target_score),
        "target_official_control_blockers": target_blockers,
        "source_feedback_candidate_id": str(feedback.get("source_candidate_id") or feedback.get("candidate_id") or ""),
        "source_official_control_score": int(source_score),
        "source_official_control_superseded": bool(
            feedback.get("source_official_control_superseded") or source_score < target_score
        ),
        **FALSE_AUTHORITY,
    }


def _hinerv_official_control_blockers(candidate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not bool(candidate.get("use_hierarchical_feature_grid")):
        blockers.append("hinerv_official_hierarchical_feature_grid_not_enabled")
    if not bool(candidate.get("use_convnext_blocks")):
        blockers.append("hinerv_official_convnext_blocks_not_enabled")
    if blockers:
        return [
            "hinerv_official_control_required_for_top_priority_launch",
            *blockers,
        ]
    return []


def _hinerv_official_control_score(row: Mapping[str, Any]) -> int:
    return int(bool(row.get("use_hierarchical_feature_grid"))) + int(bool(row.get("use_convnext_blocks")))


def _hinerv_feedback_official_control_score(row: Mapping[str, Any]) -> int:
    nested = row.get("candidate")
    if isinstance(nested, Mapping):
        nested_score = _hinerv_official_control_score(nested)
        if nested_score:
            return nested_score
    explicit_score = _hinerv_official_control_score(row)
    if explicit_score:
        return explicit_score
    candidate_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "").lower()
    return int("_hfg" in candidate_id) + int("_cnx" in candidate_id)


def _experiment_row_metadata(extra: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "archive_section_telemetry",
        "coder_qat_control",
        "decoder_weight_waterfill_plan",
        "feedback_launch_adjustment",
        "family_optimal_strategy",
        "family_training_telemetry_context",
        "optimizer_control",
        "optimizer_policy",
        "budget_candidate_id",
        "runner_modelsize_candidate_id",
        "prioritized_pair_training",
        "upstream_evaluate_score_binding",
        "tilde_oss_leverage_binding",
        "pr95_baseline_identity_binding",
        "pr95_evaluate_scorer_domain_telemetry_contract",
        "pr95_distortion_axis_trace_contract",
        "pr95_distortion_axis_trace_measurements",
        "hinerv_distortion_birth_before_rate_pressure_gate",
        "pr95_posenet_marginal_telemetry_contract",
        "pr95_scorer_atom_actuator_contract",
        "pr95_scorer_atom_actuator_execution_evidence",
        "pr95_distortion_practices_guard",
        "source_faithfulness_controls",
        "source_bound_capacity_controls",
        "source_bound_capacity_control_blockers",
        "source_parity",
        "snerv_official_runtime_authority_split",
        "current_command_is_bounded_proof_not_long_training",
        "snerv_bounded_proof_epochs",
        "snerv_scorer_tether_smoke_gate",
        "snerv_renderer_nondegenerate_gate",
        "snerv_pre_long_run_evidence_gate",
        "snerv_long_run_launch_gate",
        "snerv_lf_payload_recode_admission_plan",
        "snerv_lf_payload_codec_from_admission_plan",
        "output_dir_reuse_policy",
    )
    return {
        key: dict(extra[key]) if isinstance(extra.get(key), Mapping) else extra[key] for key in keys if key in extra
    }


def _feedback_family(row: Mapping[str, Any]) -> str:
    return _family_key(str(row.get("family") or row.get("execute_family") or ""))


def _family_key(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text == "hinerv":
        return "hi_nerv"
    return text


def _load_verified_joint_recon_weight_artifacts(
    manifest_paths: Sequence[str | Path],
) -> dict[int, dict[str, Any]]:
    artifacts: dict[int, dict[str, Any]] = {}
    for path_value in manifest_paths:
        manifest_path = Path(path_value).expanduser().resolve(strict=False)
        if not manifest_path.is_file():
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NervLongTrainingCampaignPlanError(
                f"invalid joint recon weight manifest JSON: {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest must be an object: {manifest_path}")
        if manifest.get("schema") != JOINT_RECON_PIXEL_WEIGHT_MANIFEST_SCHEMA:
            raise NervLongTrainingCampaignPlanError(f"unsupported joint recon weight manifest schema: {manifest_path}")
        config = manifest.get("config")
        metadata = manifest.get("metadata")
        if not isinstance(config, Mapping) or not isinstance(metadata, Mapping):
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest missing config/metadata: {manifest_path}"
            )
        try:
            num_pairs = int(config.get("num_pairs"))
        except (TypeError, ValueError) as exc:
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest missing numeric num_pairs: {manifest_path}"
            ) from exc
        health = metadata.get("gradient_health")
        if not isinstance(health, Mapping) or health.get("status") != "pass_finite":
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest failed gradient health: {manifest_path}"
            )
        if metadata.get("training_consumption_recommended") is not True:
            raise NervLongTrainingCampaignPlanError(
                f"joint recon weight manifest is not recommended for training consumption: {manifest_path}"
            )
        blockers = [str(item) for item in metadata.get("blockers") or [] if item]
        if blockers:
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest has blockers: {manifest_path}")
        raw_weight_path = manifest.get("weight_path")
        if raw_weight_path is None:
            raise NervLongTrainingCampaignPlanError(f"joint recon weight manifest missing weight_path: {manifest_path}")
        weight_path = Path(str(raw_weight_path)).expanduser()
        if not weight_path.is_absolute():
            weight_path = manifest_path.parent / weight_path
        weight_path = weight_path.resolve(strict=False)
        if not weight_path.is_file():
            raise NervLongTrainingCampaignPlanError(f"joint recon weight file not found: {weight_path}")
        actual_sha = _sha256_file(weight_path)
        expected_sha = str(manifest.get("weight_sha256") or "")
        if expected_sha and actual_sha != expected_sha:
            raise NervLongTrainingCampaignPlanError(f"joint recon weight sha mismatch: {weight_path}")
        artifact = {
            "schema": "nerv_long_training_joint_recon_weight_artifact.v1",
            "num_pairs": int(num_pairs),
            "manifest_path": manifest_path.as_posix(),
            "weight_path": weight_path.as_posix(),
            "weight_sha256": actual_sha,
            "gradient_health": dict(health),
            "training_consumption_recommended": True,
            **FALSE_AUTHORITY,
        }
        artifacts[int(num_pairs)] = artifact
    return dict(sorted(artifacts.items()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family") or "unknown")
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def _plan_level_blocker(blocker: str) -> bool:
    return str(blocker) in {
        "snerv_scoreaware_long_training_not_bound_bounded_native_export_stage_only",
        "snerv_native_rate_pressure_in_loop_not_yet_training_authority",
        "snerv_lf_payload_rate_axis_over_ceiling_until_representation_changes",
    }


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _safe_path_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    token = "_".join(part for part in token.split("_") if part)
    return token.strip("._-")[:180] or "campaign"


def _campaign_output_basename(
    *,
    row_id: str,
    launch_feedback_adjustment: Mapping[str, Any],
) -> str:
    base = _safe_path_token(row_id)
    if not launch_feedback_adjustment.get("applied"):
        return base
    suffix_parts = ["feedback"]
    if launch_feedback_adjustment.get("pose_instability_detected"):
        suffix_parts.append("pose_instability")
    if launch_feedback_adjustment.get("seg_stagnation_detected"):
        suffix_parts.append("seg_stagnation")
    learning_rate = _float_or_none(launch_feedback_adjustment.get("learning_rate"))
    if learning_rate is not None:
        suffix_parts.append(f"lr{_safe_path_token(_float_token(learning_rate))}")
    segnet_weight = _float_or_none(launch_feedback_adjustment.get("segnet_distillation_weight"))
    if launch_feedback_adjustment.get("segnet_weight_applied") and segnet_weight is not None:
        suffix_parts.append(f"segw{_safe_path_token(_float_token(segnet_weight))}")
    mutations = [
        _safe_path_token(str(mutation))[:48]
        for mutation in (launch_feedback_adjustment.get("launch_mutations") or [])
        if str(mutation).strip()
    ]
    suffix_parts.extend(mutations[:2])
    suffix = "_".join(part for part in suffix_parts if part)
    return _safe_path_token(f"{base}__{suffix}")


def _float_token(value: float) -> str:
    return f"{float(value):.12g}"


def _coder_qat_control(*, quant_bits: int) -> dict[str, Any]:
    return {
        "schema": "nerv_long_training_coder_qat_control.v1",
        "quant_bits": int(quant_bits),
        "quant_residual_weight": float(DEFAULT_CODER_QAT_QUANT_RESIDUAL_WEIGHT),
        "magnitude_weight": float(DEFAULT_CODER_QAT_MAGNITUDE_WEIGHT),
        "delta_weight": float(DEFAULT_CODER_QAT_DELTA_WEIGHT),
        "c1a_entropy_weight": float(DEFAULT_CODER_QAT_C1A_ENTROPY_WEIGHT),
        "c1a_sigma": float(DEFAULT_CODER_QAT_C1A_SIGMA),
        "c1a_sample_size": int(DEFAULT_CODER_QAT_C1A_SAMPLE_SIZE),
        "c1a_source": ("PR95 cat_entropy_v2 soft categorical entropy adapted to selected decoder weights"),
        **FALSE_AUTHORITY,
    }


def _coder_qat_command_args(*, quant_bits: int) -> list[str]:
    control = _coder_qat_control(quant_bits=int(quant_bits))
    return [
        "--coder-qat-quant-residual-weight",
        _float_token(float(control["quant_residual_weight"])),
        "--coder-qat-magnitude-weight",
        _float_token(float(control["magnitude_weight"])),
        "--coder-qat-delta-weight",
        _float_token(float(control["delta_weight"])),
        "--coder-qat-c1a-entropy-weight",
        _float_token(float(control["c1a_entropy_weight"])),
        "--coder-qat-c1a-sigma",
        _float_token(float(control["c1a_sigma"])),
        "--coder-qat-c1a-sample-size",
        str(int(control["c1a_sample_size"])),
    ]


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    import math

    return number if math.isfinite(number) else None


__all__ = [
    "DEFAULT_OPTIMIZER_KINDS",
    "SCHEMA",
    "SCORE_LOWERING_GATE_SCHEMA",
    "NervLongTrainingCampaignPlanError",
    "build_nerv_long_training_campaign_plan",
    "render_nerv_long_training_campaign_plan_markdown",
]
