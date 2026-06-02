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
        "planner_ingest": _planner_ingest(work_order),
        "receiver_precision_modes": precision_modes,
        "payload": dict(work_order.get("payload") or {}),
        "rationale": str(work_order.get("rationale") or ""),
        "predicted_delta_adjustment": 0.0,
        **QUEUE_FALSE_AUTHORITY,
    }


def _planner_ingest(work_order: Mapping[str, Any]) -> dict[str, Any]:
    work_order_type = str(work_order.get("work_order_type") or "")
    planner_action = str(work_order.get("planner_action") or "")
    if work_order_type == "measured_modelsize_budget_ladder":
        return {
            "ingest_kind": "measured_modelsize_ladder_work_order",
            "planner_action": planner_action,
            "producer_tool": "tools/emit_nerv_trained_ladder_row.py",
            "existing_tool_ingress": (
                "tools/build_nerv_receiver_closed_modelsize_ladder.py"
            ),
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
    return {
        "ingest_kind": "unknown_rate_allocator_work_order",
        "planner_action": planner_action,
        "runnable_now": False,
    }


def _precision_modes_from_policy(value: Any) -> list[str]:
    modes = []
    for row in _mapping_list(value):
        mode = row.get("mode")
        if mode:
            modes.append(str(mode))
    return _dedupe_strings(modes)


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
