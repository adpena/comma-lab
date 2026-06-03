# SPDX-License-Identifier: MIT
"""Fail-closed planner queue for NeRV rate/allocator work orders.

The rate allocator bridge normalizes what must be done. This module compiles
those work orders into a deterministic planner queue that final-rate attack,
bit allocator, and Cathedral consumers can ingest without treating the rows as
executable experiments or score evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from tac.analysis.nerv_rate_allocator_bridge import (
    FALSE_AUTHORITY,
)
from tac.analysis.nerv_rate_allocator_bridge import (
    SCHEMA as RATE_BRIDGE_SCHEMA,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)

SCHEMA = "nerv_rate_allocator_work_queue.v1"
AXIS_TAG = "[planning/control]"
DEFAULT_QUEUE_ID = "nerv_rate_allocator_work_queue"
DEFAULT_MLX_REFERENCE_CACHE = (
    "experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
)

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "exact_or_full_video_launched": False,
    "full_video_eval_allowed": False,
    "exact_or_full_video_cuda_allowed": False,
    "dispatch_allowed": False,
}


class NervRateAllocatorQueueError(ValueError):
    """Raised when a NeRV rate allocator queue cannot be built."""


def build_nerv_rate_allocator_work_queue(
    *,
    rate_bridge: Mapping[str, Any],
    section_value_artifacts: Sequence[Mapping[str, Any]] = (),
    queue_id: str = DEFAULT_QUEUE_ID,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Compile bridge work orders into a no-authority planner queue."""

    if not isinstance(rate_bridge, Mapping):
        raise NervRateAllocatorQueueError("rate_bridge must be a mapping")
    if rate_bridge.get("schema") != RATE_BRIDGE_SCHEMA:
        raise NervRateAllocatorQueueError(
            f"rate_bridge schema must be {RATE_BRIDGE_SCHEMA}, "
            f"got {rate_bridge.get('schema')}"
        )
    if not queue_id:
        raise NervRateAllocatorQueueError("queue_id must be non-empty")

    generated = generated_utc or datetime.now(UTC).isoformat()
    work_orders = _mapping_list(rate_bridge.get("rate_allocator_work_orders"))
    rows = [_queue_row(index, order) for index, order in enumerate(work_orders)]
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("queue_row_id") or ""),
        ),
    )
    blocked_rows = [row for row in rows if row["blocked"]]
    planning_ready_rows = [row for row in rows if not row["blocked"]]
    precision_modes = _precision_modes_from_policy(
        rate_bridge.get("receiver_precision_mode_policy")
    )
    admission_plans = [
        build_nerv_byte_price_plan(artifact)
        for artifact in section_value_artifacts
        if isinstance(artifact, Mapping)
    ]
    admission_plans.extend(_embedded_byte_price_plans(work_orders))
    admission_rows = _section_admission_queue_rows(admission_plans)

    return {
        "schema": SCHEMA,
        "queue_id": queue_id,
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "queue_kind": "planner_queue_not_experiment_queue",
        "verdict": "GO_LOCAL_PLANNER_INGEST__NO_GO_SCORE_PROMOTION_OR_EXACT_DISPATCH",
        "source_schema": rate_bridge.get("schema"),
        "source_candidate_id": rate_bridge.get("source_candidate_id"),
        "baseline_to_beat": rate_bridge.get("baseline_to_beat"),
        "top_priority_carriers": list(rate_bridge.get("top_priority_carriers") or []),
        "allowed_use": (
            "local final-rate, bit-allocator, sensitivity, and Cathedral "
            "planning only"
        ),
        "forbidden_use": (
            "score claim, rank/kill decision, promotion, exact/full-video/CUDA "
            "dispatch, or real bit assignment"
        ),
        "activation_policy": {
            "planner_rows_may_be_ranked": True,
            "planner_rows_may_open_local_source_parity_or_receiver_grammar_tasks": True,
            "planner_rows_are_executable_experiments": False,
            "dispatch_allowed": False,
            "exact_or_full_video_cuda_allowed": False,
            "requires_before_any_executable_queue": [
                "official_forward_parity",
                "source_faithful_contest_adapter",
                "measured_decoder_atom_sensitivity",
                "receiver_decoded_byte_accounting",
                "full600_byte_closed_receiver_proof",
                "paired_contest_CPU_CUDA_pass",
            ],
        },
        "receiver_precision_modes": precision_modes,
        "queue_rows": rows,
        "queue_row_count": len(rows),
        "blocked_queue_row_count": len(blocked_rows),
        "local_planning_ready_row_count": len(planning_ready_rows),
        "blocking_queue_row_ids": [row["queue_row_id"] for row in blocked_rows],
        "section_admission_plans": admission_plans,
        "section_admission_plan_count": len(admission_plans),
        "section_admission_queue_rows": admission_rows,
        "section_admission_queue_row_count": len(admission_rows),
        "section_admission_decision_counts": _section_admission_decision_counts(
            admission_rows
        ),
        "target_consumer_index": _target_consumer_index(rows),
        "precision_mode_index": _precision_mode_index(rows, precision_modes),
        "blockers": _dedupe_strings(
            [
                *_string_list(rate_bridge.get("blockers")),
                "nerv_rate_allocator_queue_is_false_authority",
                "exact_or_full_video_cuda_blocked_until_PR101_and_Z5_terminal",
                "real_bit_assignment_requires_measured_sensitivity_and_receiver_proof",
                *[
                    blocker
                    for plan in admission_plans
                    for blocker in _string_list(plan.get("blockers"))
                ],
            ]
        ),
        "predicted_delta_adjustment": 0.0,
        **QUEUE_FALSE_AUTHORITY,
    }


def _queue_row(index: int, work_order: Mapping[str, Any]) -> dict[str, Any]:
    work_order_id = str(work_order.get("work_order_id") or f"work_order_{index:04d}")
    blockers = _string_list(work_order.get("blockers"))
    target_consumers = _string_list(work_order.get("target_consumers"))
    precision_modes = _string_list(work_order.get("receiver_precision_modes"))
    blocked = bool(blockers)
    planner_ingest = _planner_ingest(work_order)
    return {
        "queue_row_id": f"nerv_rate_allocator_row_{index:04d}_{work_order_id}",
        "work_order_id": work_order_id,
        "source_unit_id": str(work_order.get("source_unit_id") or ""),
        "work_order_type": str(work_order.get("work_order_type") or "unknown"),
        "priority": int(work_order.get("priority") or 999),
        "status": (
            "blocked_until_prerequisite_evidence"
            if blocked
            else "local_planning_ready_no_exact_dispatch"
        ),
        "blocked": blocked,
        "blockers": blockers,
        "target_consumers": target_consumers,
        "planner_action": str(work_order.get("planner_action") or ""),
        "planner_ingest": planner_ingest,
        "pipeline_custody": _pipeline_custody(planner_ingest),
        "receiver_precision_modes": precision_modes,
        "payload": dict(work_order.get("payload") or {}),
        "rationale": str(work_order.get("rationale") or ""),
        "predicted_delta_adjustment": 0.0,
        **QUEUE_FALSE_AUTHORITY,
    }


def _pipeline_custody(planner_ingest: Mapping[str, Any]) -> dict[str, Any]:
    """Describe how a queue row enters canonical pipelines.

    This is intentionally attached to the queue row rather than hidden in a
    runbook: consumers can reject a row before it creates an orphan artifact.
    """

    producer_tool = str(planner_ingest.get("producer_tool") or "")
    intermediate_harvest_tool = str(
        planner_ingest.get("intermediate_harvest_tool") or ""
    )
    existing_tool_ingress = str(planner_ingest.get("existing_tool_ingress") or "")
    planning_context_tool = str(planner_ingest.get("planning_context_tool") or "")
    section_value_profile_tool = str(
        planner_ingest.get("section_value_profile_tool") or ""
    )
    downstream_ingest_tools = _string_list(
        planner_ingest.get("downstream_ingest_tools")
    )
    canonical_paths = _dedupe_strings(
        [
            producer_tool,
            intermediate_harvest_tool,
            existing_tool_ingress,
            planning_context_tool,
            section_value_profile_tool,
            *downstream_ingest_tools,
        ]
    )
    canonical_ingest_sequence = _dedupe_strings(
        [
            producer_tool,
            intermediate_harvest_tool,
            *downstream_ingest_tools,
            existing_tool_ingress,
        ]
    )
    has_ingress = bool(canonical_paths or planner_ingest.get("existing_surface_paths"))
    return {
        "schema": "nerv_rate_allocator_pipeline_custody.v1",
        "custody_mode": (
            "canonical_pipeline_ingest"
            if has_ingress
            else "blocked_until_canonical_ingest_path_exists"
        ),
        "canonical_ingress_paths": canonical_paths,
        "canonical_ingest_sequence": canonical_ingest_sequence,
        "existing_surface_paths": _string_list(
            planner_ingest.get("existing_surface_paths")
        ),
        "direct_ad_hoc_execution_allowed": False,
        "orphan_output_allowed": False,
        "output_must_reenter": (
            "archive_bound_contract_or_receiver_closed_ladder_or_section_value_profile"
        ),
        "promotion_authority_allowed": False,
        "runnable_now_is_authority": False,
        "blockers": (
            []
            if has_ingress
            else ["canonical_pipeline_ingest_path_missing_for_work_order"]
        ),
    }


def _planner_ingest(work_order: Mapping[str, Any]) -> dict[str, Any]:
    work_order_type = str(work_order.get("work_order_type") or "")
    planner_action = str(work_order.get("planner_action") or "")
    if work_order_type == "measured_modelsize_budget_ladder":
        return {
            "ingest_kind": "measured_modelsize_ladder_work_order",
            "planner_action": planner_action,
            "producer_tool": "tools/emit_nerv_trained_ladder_row.py",
            "intermediate_harvest_tool": (
                "tools/harvest_nerv_receiver_closed_ladder_rows.py"
            ),
            "existing_tool_ingress": (
                "tools/build_nerv_receiver_closed_modelsize_ladder.py"
            ),
            "downstream_ingest_tools": [
                "tools/harvest_nerv_receiver_closed_ladder_rows.py",
                "tools/build_nerv_receiver_closed_modelsize_ladder.py",
            ],
            "planning_context_tool": "tools/build_nerv_modelsize_archive_curve.py",
            "missing_tool_or_proof": (
                "trained_receiver_closed_archive_byte_ladder_rows"
            ),
            "runnable_now": False,
        }
    if work_order_type == "rate_allocator_control_binding":
        paths = (
            work_order.get("payload", {}).get("paths")
            if isinstance(work_order.get("payload"), Mapping)
            else []
        )
        return {
            "ingest_kind": "reuse_existing_control_binding",
            "planner_action": planner_action,
            "existing_surface_paths": _string_list(paths),
            "runnable_now": False,
        }
    if work_order_type == "receiver_rate_promotion_gate":
        return {
            "ingest_kind": "close_receiver_rate_promotion_gate",
            "planner_action": planner_action,
            "missing_tool_or_proof": "byte_closed_receiver_and_paired_axis_proof",
            "runnable_now": False,
        }
    if work_order_type == "decoder_weight_saliency_allocator_binding":
        return {
            "ingest_kind": "decoder_weight_saliency_waterfill_binding",
            "planner_action": planner_action,
            "producer_tool": "tools/build_hinerv_decoder_weight_saliency_replay.py",
            "existing_tool_ingress": "tools/build_hinerv_archive_ladder_waterfill.py",
            "missing_tool_or_proof": (
                "full_video_decoder_weight_saliency_replay_and_trainer_binding"
            ),
            "runnable_now": False,
        }
    if work_order_type == "decoder_weight_waterfill_archive_replay":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        replay_command_argv = _string_list(
            payload.get("archive_ladder_replay_command_argv")
        )
        replay_output_dir = str(payload.get("archive_ladder_replay_output_dir") or "")
        replay_runnable = bool(replay_command_argv and replay_output_dir)
        family = str(payload.get("family") or "")
        producer_tool, existing_tool_ingress = _waterfill_replay_tools(family)
        return {
            "ingest_kind": "decoder_weight_waterfill_archive_replay",
            "planner_action": planner_action,
            "producer_tool": producer_tool,
            "existing_tool_ingress": existing_tool_ingress,
            "missing_tool_or_proof": (
                "full_video_decoder_weight_saliency_replay_and_paired_exact_axes"
            ),
            "local_replay_runnable_now": replay_runnable,
            "local_replay_command_argv": replay_command_argv,
            "local_replay_command_hint": payload.get(
                "archive_ladder_replay_command_hint"
            ),
            "local_replay_axis_tag": payload.get(
                "archive_ladder_replay_command_axis_tag"
            ),
            "local_replay_output_dir": replay_output_dir or None,
            "local_replay_output_is_promotion_authority": False,
            "archive_bytes": payload.get("archive_bytes"),
            "archive_sha256": payload.get("archive_sha256"),
            "runnable_now": False,
        }
    if work_order_type == "receiver_proven_archive_full_video_mlx_replay":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        family = str(payload.get("family") or "")
        row_id = str(payload.get("row_id") or "unknown")
        archive_path = str(payload.get("archive_path") or "")
        submission_dir = str(payload.get("submission_dir") or "")
        archive_bytes = payload.get("archive_bytes")
        cache_command = _full_video_cache_command(
            family=family,
            row_id=row_id,
            archive_path=archive_path,
            submission_dir=submission_dir,
        )
        response_command = _full_video_response_command(
            family=family,
            row_id=row_id,
            archive_bytes=archive_bytes,
        )
        runnable = bool(
            cache_command
            and response_command
            and archive_path
            and submission_dir
            and archive_bytes
        )
        return {
            "ingest_kind": "receiver_proven_archive_full_video_mlx_replay",
            "planner_action": planner_action,
            "producer_tool": "tools/materialize_mlx_scorer_cache_from_submission.py",
            "existing_tool_ingress": "tools/run_mlx_scorer_response_cache.py",
            "section_value_profile_tool": "tools/profile_compact_renderer_mlx_section_value.py",
            "missing_tool_or_proof": (
                "full_video_mlx_response_and_section_value_profile"
            ),
            "local_full_video_mlx_replay_runnable_now": runnable,
            "local_full_video_cache_command_argv": cache_command,
            "local_full_video_response_command_argv": response_command,
            "local_full_video_response_cache_identity_mode": (
                "receiver_direct_unaudited_debug_override"
            ),
            "local_full_video_output_is_promotion_authority": False,
            "archive_bytes": archive_bytes,
            "archive_sha256": payload.get("archive_sha256"),
            "archive_path": archive_path or None,
            "submission_dir": submission_dir or None,
            "receiver_proof_ready": payload.get("receiver_proof_ready") is True,
            "runnable_now": False,
        }
    if work_order_type == "local_backend_drift_authority_guard":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        return {
            "ingest_kind": "local_backend_drift_authority_guard",
            "planner_action": planner_action,
            "producer_tool": "tools/build_hinerv_archive_backend_drift.py",
            "existing_tool_ingress": "tools/run_hinerv_archive_ladder_replay_actuator.py",
            "allowed_use": "local_iteration_velocity_only",
            "forbidden_use": "score_claim_rank_promotion_or_exact_dispatch",
            "missing_tool_or_proof": "paired_contest_cpu_cuda_auth_eval",
            "reference_label": payload.get("reference_label"),
            "candidate_label": payload.get("candidate_label"),
            "row_count": payload.get("row_count"),
            "matched_row_count": payload.get("matched_row_count"),
            "byte_ready_row_count": payload.get("byte_ready_row_count"),
            "max_abs_byte_delta_allowed": payload.get("max_abs_byte_delta_allowed"),
            "max_abs_byte_delta_observed": payload.get("max_abs_byte_delta_observed"),
            "sum_byte_delta_candidate_minus_reference": payload.get(
                "sum_byte_delta_candidate_minus_reference"
            ),
            "sum_rate_score_delta_candidate_minus_reference": payload.get(
                "sum_rate_score_delta_candidate_minus_reference"
            ),
            "within_byte_drift_tolerance": (
                payload.get("within_byte_drift_tolerance") is True
            ),
            "local_dev_velocity_ready": payload.get("local_dev_velocity_ready") is True,
            "ready_backend_for_local_iteration": payload.get(
                "ready_backend_for_local_iteration"
            ),
            "local_backend_output_is_promotion_authority": False,
            "runnable_now": False,
        }
    if work_order_type == "snerv_scorer_loop_qat_full600_followup":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        command = _snerv_scorer_loop_full600_command(payload)
        byte_price_plan = (
            payload.get("byte_price_plan")
            if isinstance(payload.get("byte_price_plan"), Mapping)
            else {}
        )
        return {
            "ingest_kind": "snerv_scorer_loop_qat_full600_followup",
            "planner_action": planner_action,
            "producer_tool": "experiments/train_substrate_snerv_scorer_loop_local.py",
            "existing_tool_ingress": "tools/build_nerv_control_inventory.py",
            "missing_tool_or_proof": (
                "full600_receiver_proof_section_value_and_paired_exact_axes"
            ),
            "source_report_path": payload.get("report_path"),
            "source_result_sha256": payload.get("result_sha256"),
            "source_n_pairs": payload.get("n_pairs"),
            "source_scorer_loop_evaluations": payload.get(
                "scorer_loop_evaluations"
            ),
            "source_history_count": payload.get("history_count"),
            "source_selection_policy": payload.get("selection_policy"),
            "source_score_delta_linf": payload.get("score_delta_linf"),
            "source_score_delta_fraction": payload.get("score_delta_fraction"),
            "source_candidate_count": payload.get("candidate_count"),
            "source_accepted_candidate_count": payload.get(
                "accepted_candidate_count"
            ),
            "source_rejected_candidate_count": payload.get(
                "rejected_candidate_count"
            ),
            "source_best_pair_deltas": _mapping_list(
                payload.get("best_pair_deltas")
            ),
            "source_section_value_rows": _mapping_list(
                payload.get("section_value_rows")
            ),
            "source_byte_price_plan_schema": byte_price_plan.get("schema"),
            "source_byte_price_decision_rows": _mapping_list(
                byte_price_plan.get("decision_rows")
            ),
            "source_accepted_improvement": payload.get("accepted_improvement") is True,
            "source_receiver_contract_satisfied": (
                payload.get("receiver_contract_satisfied") is True
            ),
            "local_full600_continuation_runnable_now": bool(command),
            "local_full600_continuation_command_argv": command,
            "local_full600_continuation_output_is_promotion_authority": False,
            "runnable_now": False,
        }
    if work_order_type == "snerv_lf_payload_codec_full_archive_replay":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        byte_price_plan = (
            payload.get("byte_price_plan")
            if isinstance(payload.get("byte_price_plan"), Mapping)
            else {}
        )
        return {
            "ingest_kind": "snerv_lf_payload_codec_full_archive_replay",
            "planner_action": planner_action,
            "producer_tool": "tools/build_snerv_lf_payload_codec_sweep.py",
            "existing_tool_ingress": "tools/build_nerv_control_inventory.py",
            "missing_tool_or_proof": (
                "full_archive_receiver_replay_full_video_section_value_and_paired_exact_axes"
            ),
            "source_report_path": payload.get("report_path"),
            "source_artifact_path": payload.get("source_artifact_path"),
            "source_artifact_bytes": payload.get("source_artifact_bytes"),
            "source_artifact_sha256": payload.get("source_artifact_sha256"),
            "source_selection_policy": payload.get("selection_policy"),
            "source_history_count": payload.get("history_count"),
            "source_plane_count": payload.get("plane_count"),
            "source_raw_i64_bytes": payload.get("raw_i64_bytes"),
            "source_baseline_mode": payload.get("baseline_mode"),
            "source_baseline_payload_bytes": payload.get(
                "baseline_payload_bytes"
            ),
            "source_selected_mode": payload.get("selected_mode"),
            "source_selected_payload_bytes": payload.get(
                "selected_payload_bytes"
            ),
            "source_selected_packet_schema": payload.get(
                "selected_packet_schema"
            ),
            "source_codec_rows": _mapping_list(payload.get("codec_rows")),
            "source_section_value_rows": _mapping_list(
                payload.get("section_value_rows")
            ),
            "source_byte_price_plan_schema": byte_price_plan.get("schema"),
            "source_byte_price_decision_rows": _mapping_list(
                byte_price_plan.get("decision_rows")
            ),
            "local_lf_codec_packet_sweep_is_promotion_authority": False,
            "local_full_archive_replay_runnable_now": False,
            "runnable_now": False,
        }
    if work_order_type == "receiver_visible_decoder_mode_assignment":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        probe_command_argv = _string_list(payload.get("probe_command_argv"))
        probe_packet_dir = str(payload.get("probe_receiver_packet_dir") or "")
        probe_runnable = bool(probe_command_argv and probe_packet_dir)
        return {
            "ingest_kind": "receiver_visible_decoder_mode_assignment",
            "planner_action": planner_action,
            "producer_tool": "tools/build_snerv_waterfill_mode_assignment.py",
            "existing_tool_ingress": "tools/probe_snerv_decoder_mode_assignments.py",
            "missing_tool_or_proof": (
                "receiver_visible_mixed_precision_decoder_grammar_export"
            ),
            "local_advisory_probe_runnable_now": probe_runnable,
            "local_advisory_probe_command_argv": probe_command_argv,
            "local_advisory_probe_command_hint": payload.get("probe_command_hint"),
            "local_advisory_probe_axis_tag": payload.get("probe_command_axis_tag"),
            "local_advisory_receiver_packet_dir": probe_packet_dir or None,
            "local_advisory_output_is_promotion_authority": False,
            "runnable_now": False,
        }
    if work_order_type == "decoder_mode_pair_robust_probe_followup":
        payload = (
            work_order.get("payload", {})
            if isinstance(work_order.get("payload"), Mapping)
            else {}
        )
        packet_path = str(payload.get("receiver_archive_packet_path") or "")
        replay_verified = payload.get("receiver_archive_replay_verified") is True
        return {
            "ingest_kind": "decoder_mode_pair_robust_probe_followup",
            "planner_action": planner_action,
            "producer_tool": "tools/probe_snerv_decoder_mode_assignments.py",
            "existing_tool_ingress": "tools/build_snerv_waterfill_mode_assignment.py",
            "missing_tool_or_proof": (
                "stratified_pair_pose_guard_replay_and_full600_receiver_proof"
            ),
            "source_receiver_packet_path": packet_path or None,
            "source_receiver_packet_bytes": payload.get(
                "receiver_archive_packet_bytes"
            ),
            "source_receiver_packet_sha256": payload.get(
                "receiver_archive_packet_sha256"
            ),
            "source_receiver_replay_verified": replay_verified,
            "source_receiver_packet_is_contest_archive_zip": (
                payload.get("receiver_archive_packet_is_contest_archive_zip") is True
            ),
            "local_pair_robust_replay_runnable_now": bool(
                packet_path and replay_verified
            ),
            "local_pair_robust_replay_is_promotion_authority": False,
            "runnable_now": False,
        }
    return {
        "ingest_kind": "unknown_rate_allocator_work_order",
        "planner_action": planner_action,
        "runnable_now": False,
    }


def _embedded_byte_price_plans(
    work_orders: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    plans: list[Mapping[str, Any]] = []
    for order in work_orders:
        payload = (
            order.get("payload", {})
            if isinstance(order.get("payload"), Mapping)
            else {}
        )
        plan = payload.get("byte_price_plan")
        if isinstance(plan, Mapping) and plan.get("schema") == (
            "compact_nerv_byte_price_controller.v1"
        ):
            plans.append(plan)
    return plans


def _precision_modes_from_policy(value: Any) -> list[str]:
    modes = []
    for row in _mapping_list(value):
        mode = row.get("mode")
        if mode:
            modes.append(str(mode))
    return _dedupe_strings(modes)


def _waterfill_replay_tools(family: str) -> tuple[str, str]:
    if family in {"hi_nerv", "hinerv"}:
        return (
            "tools/build_hinerv_archive_ladder_waterfill.py",
            "tools/build_hinerv_archive_size_ladder.py",
        )
    if family == "snerv":
        return (
            "tools/build_snerv_trained_ladder_waterfill.py",
            "tools/prove_snerv_receiver_archive.py",
        )
    return ("unknown_decoder_weight_waterfill_producer", "unknown_receiver_replay_tool")


def _full_video_cache_command(
    *,
    family: str,
    row_id: str,
    archive_path: str,
    submission_dir: str,
) -> list[str]:
    if family not in {"hi_nerv", "hinerv"} or not archive_path or not submission_dir:
        return []
    root = (
        "/Volumes/VertigoDataTier/pact/"
        f"hinerv_full_video_mlx_replay/{_safe_token(row_id)}"
    )
    return [
        ".venv/bin/python",
        "tools/materialize_mlx_scorer_cache_from_submission.py",
        "--archive",
        archive_path,
        "--submission-dir",
        submission_dir,
        "--output-cache-dir",
        f"{root}/candidate_cache",
        "--work-dir",
        f"{root}/cache_work",
        "--report-output",
        (
            ".omx/research/"
            f"hinerv_full_video_mlx_cache_{_safe_token(row_id)}_false_authority.json"
        ),
        "--receiver-direct-cache",
        "--batch-pairs",
        "1",
        "--max-pairs",
        "600",
        "--allow-large-tensor-cache",
        "--force",
    ]


def _full_video_response_command(
    *,
    family: str,
    row_id: str,
    archive_bytes: Any,
) -> list[str]:
    try:
        bytes_int = int(archive_bytes)
    except (TypeError, ValueError):
        return []
    if family not in {"hi_nerv", "hinerv"} or bytes_int <= 0:
        return []
    root = (
        "/Volumes/VertigoDataTier/pact/"
        f"hinerv_full_video_mlx_replay/{_safe_token(row_id)}"
    )
    return [
        ".venv/bin/python",
        "tools/run_mlx_scorer_response_cache.py",
        "--reference-cache-dir",
        DEFAULT_MLX_REFERENCE_CACHE,
        "--candidate-cache-dir",
        f"{root}/candidate_cache",
        "--archive-size-bytes",
        str(bytes_int),
        "--output",
        (
            ".omx/research/"
            f"hinerv_full_video_mlx_response_{_safe_token(row_id)}_false_authority.json"
        ),
        "--repo-root",
        ".",
        "--batch-pairs",
        "1",
        "--max-pairs",
        "600",
        "--device",
        "gpu",
        "--allow-gpu-research-signal",
        "--allow-unaudited-candidate-cache-debug",
        "--allow-local-cpu-advisory-cache-identity",
        "--cache-integrity-mode",
        "manifest",
        "--response-family",
        f"hi_nerv_{_safe_token(row_id)}",
    ]


def _snerv_scorer_loop_full600_command(payload: Mapping[str, Any]) -> list[str]:
    levels = _positive_int(payload.get("levels")) or 1
    qat_bits = _positive_int(payload.get("qat_bits")) or 8
    search_mode = str(payload.get("search_mode") or "nes_pair_robust")
    wavelet = str(payload.get("wavelet") or "haar")
    return [
        ".venv/bin/python",
        "experiments/train_substrate_snerv_scorer_loop_local.py",
        "--score-loop",
        "--n-pairs",
        "600",
        "--levels",
        str(levels),
        "--wavelet",
        wavelet,
        "--qat-bits",
        str(qat_bits),
        "--search-mode",
        search_mode,
        "--max-trials",
        "16",
        "--byte-pressure-multiplier",
        "2.0",
        "--pair-guard-min-score-improved-fraction",
        "0.75",
        "--pair-guard-max-pose-worsened-fraction",
        "0.0",
        "--storage-workload-subdir",
        "snerv_scorer_loop_qat_full600_followup",
    ]


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_token(value: str) -> str:
    text = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value)
    ).strip("_")
    return text or "row"


def _target_consumer_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for row in rows:
        row_id = str(row.get("queue_row_id") or "")
        for consumer in _string_list(row.get("target_consumers")):
            index.setdefault(consumer, []).append(row_id)
    return {key: sorted(values) for key, values in sorted(index.items())}


def _precision_mode_index(
    rows: Sequence[Mapping[str, Any]],
    precision_modes: Sequence[str],
) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {mode: [] for mode in precision_modes}
    for row in rows:
        row_id = str(row.get("queue_row_id") or "")
        for mode in _string_list(row.get("receiver_precision_modes")):
            index.setdefault(mode, []).append(row_id)
    return {key: sorted(values) for key, values in sorted(index.items())}


def _section_admission_queue_rows(
    admission_plans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for plan_index, plan in enumerate(admission_plans):
        for row_index, decision in enumerate(_mapping_list(plan.get("decision_rows"))):
            decision_id = str(decision.get("row_id") or f"decision_{row_index:04d}")
            final_decision = str(decision.get("decision") or "demote")
            blockers = _string_list(decision.get("blockers"))
            rows.append(
                {
                    "queue_row_id": (
                        f"nerv_section_admission_row_{plan_index:04d}_"
                        f"{row_index:04d}_{decision_id}"
                    ),
                    "source_plan_schema": plan.get("schema"),
                    "candidate_id": decision.get("candidate_id")
                    or plan.get("candidate_id"),
                    "section_id": decision.get("section_id"),
                    "row_id": decision_id,
                    "row_kind": decision.get("row_kind"),
                    "decision": final_decision,
                    "economic_decision": decision.get("economic_decision"),
                    "status": (
                        "blocked_fail_closed"
                        if blockers
                        else f"local_section_{final_decision}_ready_no_exact_dispatch"
                    ),
                    "blocked": bool(blockers),
                    "blockers": blockers,
                    "byte_delta": decision.get("byte_delta"),
                    "section_bytes": decision.get("section_bytes"),
                    "delta_nonrate_score": decision.get("delta_nonrate_score"),
                    "delta_rate_score": decision.get("delta_rate_score"),
                    "delta_total_score": decision.get("delta_total_score"),
                    "archive_sha256": decision.get("archive_sha256"),
                    "axis_labels": list(decision.get("axis_labels") or ()),
                    "receiver_proof_status": decision.get("receiver_proof_status"),
                    "full_video_coverage": bool(decision.get("full_video_coverage")),
                    "target_consumers": [
                        "final_rate_attack",
                        "bit_allocator",
                        "bounded_runner",
                    ],
                    "planner_action": _section_admission_planner_action(
                        final_decision
                    ),
                    "predicted_delta_adjustment": 0.0,
                    **QUEUE_FALSE_AUTHORITY,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            bool(row.get("blocked")),
            str(row.get("decision") or ""),
            str(row.get("queue_row_id") or ""),
        ),
    )


def _section_admission_decision_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision") or "unknown")
        counts[decision] = counts.get(decision, 0) + 1
    return dict(sorted(counts.items()))


def _section_admission_planner_action(decision: str) -> str:
    if decision == "cut":
        return "materialize_section_cut_candidate_after_receiver_replay"
    if decision == "admit":
        return "materialize_residual_or_sidecar_candidate_after_receiver_replay"
    if decision == "protect":
        return "protect_section_bytes_in_training_and_codec_sweep"
    if decision == "retrain":
        return "retrain_section_or_residual_until_value_exceeds_byte_price"
    return "demote_or_block_section_family_until_custody_repairs"


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "AXIS_TAG",
    "DEFAULT_QUEUE_ID",
    "QUEUE_FALSE_AUTHORITY",
    "SCHEMA",
    "NervRateAllocatorQueueError",
    "build_nerv_rate_allocator_work_queue",
]
