# SPDX-License-Identifier: MIT
"""Candidate-conditioned curriculum plans for NeRV-family campaigns.

Model-size candidates choose capacity and nominal byte budgets. This module
binds those candidates to the training pressure required to make the capacity
meaningful: PR95-style stages, real scorer teachers, coder-aware QAT, and
measured archive-byte feedback. It is planner/runner metadata, not score
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tac.analysis.nerv_modelsize_budget import RATE_SCORE_PER_BYTE
from tac.analysis.pr95_stack_binding_requirements import (
    build_pr95_long_campaign_prelaunch_gate,
    build_pr95_stack_binding_evidence,
    build_pr95_stack_binding_requirements,
)
from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
    CANONICAL_PR95_TOTAL_EPOCHS,
)

SCHEMA = "nerv_candidate_curriculum_plan.v1"
BYTE_FEEDBACK_SCHEMA = "nerv_candidate_byte_feedback.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
AUTHORITY_FIELD_NAMES = frozenset(FALSE_AUTHORITY)


def _codec_bits(codec: Any, *, default: int = 8) -> int:
    text = str(codec or "").strip().lower()
    if text.startswith("int2"):
        return 2
    if text.startswith("int4") or "mixed_magnitude" in text:
        return 4
    if text.startswith("int8") or text in {"portfolio_auto", "auto"}:
        return 8
    if text.startswith("fp16"):
        return 16
    if text.startswith("float32") or text.startswith("fp32"):
        return 32
    return int(default)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def strip_candidate_curriculum_authority_fields(value: Any) -> Any:
    """Return a metadata-safe copy without canonical authority/readiness keys."""

    if isinstance(value, Mapping):
        return {
            str(key): strip_candidate_curriculum_authority_fields(item)
            for key, item in value.items()
            if str(key) not in AUTHORITY_FIELD_NAMES
        }
    if isinstance(value, list):
        return [strip_candidate_curriculum_authority_fields(item) for item in value]
    if isinstance(value, tuple):
        return [strip_candidate_curriculum_authority_fields(item) for item in value]
    return value


def _base_byte_feedback(
    *,
    candidate: Mapping[str, Any],
    measured_num_pairs: int,
    measured_payload_bytes: int | None = None,
    measured_archive_bytes: int | None = None,
) -> dict[str, Any]:
    nominal = _int(candidate.get("nominal_total_payload_bytes"))
    measured = measured_archive_bytes if measured_archive_bytes is not None else measured_payload_bytes
    delta = None if measured is None else int(measured) - int(nominal)
    candidate_num_pairs = _int(candidate.get("num_pairs"))
    measured_pairs = int(measured_num_pairs)
    scope_matches = candidate_num_pairs > 0 and measured_pairs == candidate_num_pairs
    return {
        "schema": BYTE_FEEDBACK_SCHEMA,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_num_pairs": candidate_num_pairs,
        "measured_num_pairs": measured_pairs,
        "feedback_scope": (
            "candidate_full_scope" if scope_matches else "partial_pair_advisory"
        ),
        "scope_matches_candidate": scope_matches,
        "hard_byte_ceiling": _int(candidate.get("hard_byte_ceiling")),
        "nominal_total_payload_bytes": nominal,
        "measured_payload_bytes": (
            None if measured_payload_bytes is None else int(measured_payload_bytes)
        ),
        "measured_archive_bytes": (
            None if measured_archive_bytes is None else int(measured_archive_bytes)
        ),
        "measured_minus_nominal_bytes": delta,
        "measured_minus_nominal_rate_score_delta": (
            None if delta is None else float(delta * RATE_SCORE_PER_BYTE)
        ),
        "feedback_ready": measured is not None and scope_matches,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_hinerv_candidate_curriculum_plan(
    *,
    candidate: Mapping[str, Any] | None,
    requested_epochs: int,
    num_pairs: int,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    coder_aware_qat: bool,
    coder_qat_quant_bits: int,
    recon_pixel_weight_attached: bool,
    eval_roundtrip_ste_attached: bool = False,
    differentiable_pose_preprocess_attached: bool = False,
    ema_archive_selection_attached: bool = False,
    receiver_proof_attached: bool = False,
    full_video_local_prefilter_attached: bool = False,
    local_cpu_replay_gate_attached: bool = False,
    measured_archive_bytes: int | None = None,
) -> dict[str, Any]:
    """Bind a HiNeRV modelsize candidate to its required training pressure."""

    candidate_row = dict(candidate or {})
    candidate_selected = bool(candidate_row)
    codec = candidate_row.get("decoder_codec", "manual_cli")
    codec_bits = _codec_bits(codec, default=int(coder_qat_quant_bits))
    q_bits = min(max(1, int(coder_qat_quant_bits)), codec_bits)
    if candidate_selected:
        q_bits = codec_bits
    effective_coder_regularizer = bool(coder_aware_qat or candidate_selected)
    epochs = max(0, int(requested_epochs))
    full_video = int(num_pairs) >= 600
    blockers: list[str] = []
    launch_mutations: list[str] = []
    if candidate_selected and not coder_aware_qat:
        launch_mutations.append(
            "enabled_decoder_coder_regularizer_from_modelsize_candidate"
        )
    if candidate_selected and int(coder_qat_quant_bits) != q_bits:
        launch_mutations.append("aligned_coder_qat_quant_bits_to_candidate_codec")
    if epochs < 8:
        blockers.append("hinerv_candidate_curriculum_requires_min_8_epochs")
    if _num(segnet_distillation_weight) <= 0.0:
        blockers.append("hinerv_candidate_curriculum_requires_real_segnet_teacher")
    if _num(pose_distillation_weight) <= 0.0:
        blockers.append("hinerv_candidate_curriculum_requires_real_posenet_teacher")
    if not recon_pixel_weight_attached:
        blockers.append("hinerv_candidate_curriculum_recon_pixel_weight_missing")
    if not full_video:
        blockers.append("hinerv_candidate_curriculum_full600_required_for_promotion")
    if not candidate_selected:
        blockers.append("hinerv_modelsize_candidate_not_selected_manual_probe")
    byte_feedback = _base_byte_feedback(
        candidate=candidate_row,
        measured_num_pairs=int(num_pairs),
        measured_archive_bytes=measured_archive_bytes,
    ) if candidate_selected else {
        "schema": BYTE_FEEDBACK_SCHEMA,
        "candidate_id": None,
        "candidate_num_pairs": None,
        "measured_num_pairs": int(num_pairs),
        "feedback_scope": "manual_cli_probe",
        "scope_matches_candidate": False,
        "feedback_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if candidate_selected and measured_archive_bytes is not None and not byte_feedback["scope_matches_candidate"]:
        blockers.append("partial_pair_byte_feedback_only")
    elif candidate_selected and not byte_feedback["feedback_ready"]:
        blockers.append("hinerv_trained_archive_byte_oracle_feedback_missing")
    pr95_binding = build_pr95_stack_binding_requirements(
        family="hi_nerv",
        evidence=build_pr95_stack_binding_evidence(
            modelsize_archive_budget=candidate_selected,
            pr95_staged_curriculum=epochs >= 8,
            real_segnet_teacher=_num(segnet_distillation_weight) > 0.0,
            real_posenet_teacher=_num(pose_distillation_weight) > 0.0,
            differentiable_pose_preprocess=bool(
                differentiable_pose_preprocess_attached
            ),
            eval_roundtrip_ste=bool(eval_roundtrip_ste_attached),
            ema_archive_selection=bool(ema_archive_selection_attached),
            qat_forward=effective_coder_regularizer,
            coder_aware_regularizer=effective_coder_regularizer,
            muon_adamw_partition=epochs >= 8,
            archive_in_loop_byte_oracle=bool(byte_feedback.get("feedback_ready")),
            byte_closed_archive_export=measured_archive_bytes is not None,
            receiver_proof=bool(receiver_proof_attached),
            full_video_local_prefilter=bool(full_video_local_prefilter_attached),
            local_cpu_replay_gate=bool(local_cpu_replay_gate_attached),
        ),
    )
    long_campaign_prelaunch_gate = build_pr95_long_campaign_prelaunch_gate(
        pr95_binding
    )
    blockers.extend(pr95_binding["blockers"])
    return {
        "schema": SCHEMA,
        "family": "hi_nerv",
        "candidate_conditioned": candidate_selected,
        "candidate_id": candidate_row.get("candidate_id"),
        "num_pairs": int(num_pairs),
        "campaign_scope": "full600" if full_video else "partial_pair_smoke",
        "pr95_stage_plan": {
            "enabled": epochs >= 8,
            "requested_epochs": epochs,
            "minimum_candidate_epochs": 8,
            "canonical_full_epochs": CANONICAL_PR95_TOTAL_EPOCHS,
            "stage_count": 8,
            "stage_policy": "scaled_pr95_8_stage_curriculum",
        },
        "scorer_pressure": {
            "segnet_distillation_weight": float(segnet_distillation_weight),
            "pose_distillation_weight": float(pose_distillation_weight),
            "joint_p18_p19_weight_attached": bool(recon_pixel_weight_attached),
            "eval_roundtrip_ste_attached": bool(eval_roundtrip_ste_attached),
            "ema_archive_selection_attached": bool(
                ema_archive_selection_attached
            ),
            "receiver_proof_attached": bool(receiver_proof_attached),
            "differentiable_pose_preprocess_attached": bool(
                differentiable_pose_preprocess_attached
            ),
            "auto_teacher_weights_are_forbidden": True,
        },
        "coder_pressure": {
            "enabled": effective_coder_regularizer,
            "regularizer_enabled": effective_coder_regularizer,
            "fake_quant_forward_enabled": effective_coder_regularizer,
            "quant_bits": int(q_bits),
            "candidate_decoder_codec": str(codec),
            "candidate_decoder_codec_bits": int(codec_bits),
            "source": (
                "modelsize_candidate" if candidate_selected else "manual_cli_knobs"
            ),
            "implementation_status": (
                "decoder_weight_fake_quant_forward_plus_quant_residual_regularizer"
                if effective_coder_regularizer
                else "disabled"
            ),
        },
        "byte_oracle_logging": byte_feedback,
        "pr95_stack_binding": pr95_binding,
        "long_campaign_prelaunch_gate": long_campaign_prelaunch_gate,
        "launch_mutations": launch_mutations,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def build_snerv_candidate_curriculum_plan(
    *,
    candidate: Mapping[str, Any] | None,
    requested_epochs: int,
    num_pairs: int,
    step_map_coder_mode: str,
    measured_packet_bytes: int | None = None,
    measured_archive_bytes: int | None = None,
    scorer_loop_qat_attached: bool = False,
    scorer_loop_qat_receiver_contract_satisfied: bool = False,
    scorer_loop_qat_ready_for_pose_guard_gate: bool = False,
    scorer_loop_qat_accepted_improvement: bool = False,
) -> dict[str, Any]:
    """Bind a SNeRV receiver-grammar candidate to byte feedback and blockers."""

    candidate_row = dict(candidate or {})
    candidate_selected = bool(candidate_row)
    full_video = int(num_pairs) >= 600
    levels = _int(candidate_row.get("levels"), 3)
    lf_bits = _num(candidate_row.get("bits_per_coeff"), 2.5)
    step_bits = _num(candidate_row.get("step_map_bits_per_coeff"), 4.0)
    decoder_codec = str(candidate_row.get("decoder_payload_codec", "manual_cli"))
    byte_feedback = _base_byte_feedback(
        candidate=candidate_row,
        measured_num_pairs=int(num_pairs),
        measured_payload_bytes=measured_packet_bytes,
        measured_archive_bytes=measured_archive_bytes,
    ) if candidate_selected else {
        "schema": BYTE_FEEDBACK_SCHEMA,
        "candidate_id": None,
        "candidate_num_pairs": None,
        "measured_num_pairs": int(num_pairs),
        "feedback_scope": "manual_cli_probe",
        "scope_matches_candidate": False,
        "feedback_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    blockers: list[str] = [
        "snerv_mlx_native_train_export_adapter_missing",
    ]
    if not scorer_loop_qat_attached:
        blockers.append("snerv_scorer_loop_qat_not_attached")
    blockers.append("snerv_score_aware_curriculum_not_native_mlx_yet")
    if not candidate_selected:
        blockers.append("snerv_modelsize_candidate_not_selected_manual_probe")
    if not full_video:
        blockers.append("snerv_candidate_curriculum_full600_required_for_promotion")
    if str(step_map_coder_mode) != "waterfill":
        blockers.append("snerv_candidate_curriculum_requires_waterfill_step_maps")
    if (
        candidate_selected
        and (measured_packet_bytes is not None or measured_archive_bytes is not None)
        and not byte_feedback["scope_matches_candidate"]
    ):
        blockers.append("partial_pair_byte_feedback_only")
    elif candidate_selected and not byte_feedback["feedback_ready"]:
        blockers.append("snerv_snar1_byte_feedback_missing")
    if scorer_loop_qat_attached:
        if not scorer_loop_qat_receiver_contract_satisfied:
            blockers.append("snerv_scorer_loop_qat_receiver_contract_failed")
        if not scorer_loop_qat_ready_for_pose_guard_gate:
            blockers.append("snerv_scorer_loop_qat_pose_guard_not_ready")
        if not scorer_loop_qat_accepted_improvement:
            blockers.append("snerv_scorer_loop_qat_no_accepted_improvement")
    pr95_binding = build_pr95_stack_binding_requirements(
        family="snerv",
        evidence=build_pr95_stack_binding_evidence(
            modelsize_archive_budget=candidate_selected,
            real_segnet_teacher=bool(scorer_loop_qat_attached),
            real_posenet_teacher=bool(scorer_loop_qat_attached),
            qat_forward=bool(scorer_loop_qat_attached),
            coder_aware_regularizer=bool(scorer_loop_qat_attached),
            archive_in_loop_byte_oracle=bool(byte_feedback.get("feedback_ready")),
            byte_closed_archive_export=measured_archive_bytes is not None,
        ),
    )
    long_campaign_prelaunch_gate = build_pr95_long_campaign_prelaunch_gate(
        pr95_binding
    )
    blockers.extend(pr95_binding["blockers"])
    return {
        "schema": SCHEMA,
        "family": "snerv",
        "candidate_conditioned": candidate_selected,
        "candidate_id": candidate_row.get("candidate_id"),
        "num_pairs": int(num_pairs),
        "campaign_scope": "full600" if full_video else "partial_pair_smoke",
        "receiver_grammar_controls": {
            "levels": levels,
            "lf_bits_per_coeff": lf_bits,
            "step_map_bits_per_coeff": step_bits,
            "step_map_coder_mode": str(step_map_coder_mode),
            "decoder_payload_codec": decoder_codec,
        },
        "training_plan": {
            "requested_epochs": int(requested_epochs),
            "native_mlx_training_required": True,
            "current_execution_path": "cpu_advisory_receiver_bound_packet",
            "next_required_adapter": "snerv_mlx_native_train_export_archive",
            "scorer_loop_qat_attached": bool(scorer_loop_qat_attached),
            "scorer_loop_qat_receiver_contract_satisfied": bool(
                scorer_loop_qat_receiver_contract_satisfied
            ),
            "scorer_loop_qat_ready_for_pose_guard_gate": bool(
                scorer_loop_qat_ready_for_pose_guard_gate
            ),
            "scorer_loop_qat_accepted_improvement": bool(
                scorer_loop_qat_accepted_improvement
            ),
        },
        "byte_oracle_logging": byte_feedback,
        "pr95_stack_binding": pr95_binding,
        "long_campaign_prelaunch_gate": long_campaign_prelaunch_gate,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


__all__ = [
    "AUTHORITY_FIELD_NAMES",
    "BYTE_FEEDBACK_SCHEMA",
    "SCHEMA",
    "build_hinerv_candidate_curriculum_plan",
    "build_snerv_candidate_curriculum_plan",
    "strip_candidate_curriculum_authority_fields",
]
