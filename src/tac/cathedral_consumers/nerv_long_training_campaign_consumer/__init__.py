# SPDX-License-Identifier: MIT
"""Cathedral consumer for HiNeRV/SNeRV long-training campaign queues.

This consumer ingests the queue-owned
``nerv_long_training_campaign_plan.v1`` artifact, or the extracted
``experiment_queue.v1`` it emits, and converts it into an autopilot-safe
routing verdict. It deliberately stays Tier A: local MLX campaign rows may be
recommended for acquisition, but score, CPU replay, exact auth, and promotion
authority remain closed until receiver proof and same-axis controls exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "nerv_long_training_campaign_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)

PLAN_SCHEMA = "nerv_long_training_campaign_plan.v1"
QUEUE_SCHEMA = "experiment_queue.v1"
RESULT_SCHEMA = "nerv_long_training_campaign_consumer_result.v1"
LAUNCH_AUTHORITY_CONTRACT_SCHEMA = (
    "nerv_long_training_queue_launch_authority_contract.v1"
)

_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "ready_for_operator_probe",
    "ready_for_provider_dispatch",
    "dispatch_attempted",
)


def update_from_anchor(anchor: Any) -> None:
    """Hook #5 placeholder: campaign artifacts carry their own gate state."""

    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Route a long-training plan or queue without granting score authority."""

    source_schema = str(candidate.get("schema") or "")
    blockers = _authority_blockers(candidate, label="campaign")
    source_blockers = _string_list(candidate.get("blockers"))
    blockers.extend(source_blockers)

    campaign_rows: list[Mapping[str, Any]] = []
    queue: Mapping[str, Any] = {}
    experiments: list[Mapping[str, Any]] = []
    if source_schema == PLAN_SCHEMA:
        campaign_rows = _mapping_list(candidate.get("campaign_rows"))
        queue = _mapping(candidate.get("experiment_queue"))
        if queue.get("schema") != QUEUE_SCHEMA:
            blockers.append("experiment_queue_missing_or_schema_mismatch")
        experiments = _mapping_list(queue.get("experiments"))
    elif source_schema == QUEUE_SCHEMA:
        queue = candidate
        experiments = _mapping_list(candidate.get("experiments"))
    else:
        blockers.append("campaign_plan_schema_mismatch")

    if source_schema in {PLAN_SCHEMA, QUEUE_SCHEMA} and not experiments:
        blockers.append("experiment_rows_missing")

    for index, row in enumerate(campaign_rows):
        blockers.extend(_authority_blockers(row, label=f"campaign_row_{index}"))
    for index, experiment in enumerate(experiments):
        experiment_id = _experiment_id(experiment, index)
        blockers.extend(
            _authority_blockers(experiment, label=f"experiment_{experiment_id}")
        )
        blockers.extend(
            _launch_authority_contract_blockers(
                experiment,
                label=f"experiment_{experiment_id}",
            )
        )
        gate = _mapping(experiment.get("score_lowering_gate"))
        blockers.extend(
            _authority_blockers(gate, label=f"experiment_gate_{experiment_id}")
        )

    ready_local = [
        _compact_experiment(experiment, index)
        for index, experiment in enumerate(experiments)
        if _local_mlx_ready(experiment)
    ]
    blocked_experiments = [
        experiment
        for experiment in experiments
        if experiment.get("blocked") is True
        or str(experiment.get("status") or "")
        in {"blocked_dependency", "blocked", "disabled"}
    ]
    gated_experiments = [
        experiment for experiment in experiments if _experiment_has_gates(experiment)
    ]
    cpu_ready = [
        experiment
        for experiment in experiments
        if _mapping(experiment.get("score_lowering_gate")).get("cpu_replay_ready")
        is True
    ]
    exact_ready = [
        experiment
        for experiment in experiments
        if _mapping(experiment.get("score_lowering_gate")).get("exact_gate_ready")
        is True
        or experiment.get("exact_gate_ready") is True
    ]
    if cpu_ready:
        blockers.append("local_cpu_replay_rows_require_receiver_proof_control_replay")
    if exact_ready:
        blockers.append(
            "exact_gate_rows_require_lane_claim_pr95_control_and_pr101_z5_adjudication"
        )

    if "campaign_plan_schema_mismatch" in blockers:
        planner_action = "repair_campaign_plan_schema_before_cathedral_consumption"
    elif _has_authority_overclaim(blockers):
        planner_action = "repair_campaign_false_authority_contract_before_consumption"
    elif ready_local:
        planner_action = "route_launchable_local_mlx_campaign_rows_without_exact_dispatch"
    elif _snerv_binding_gap(blockers, experiments):
        planner_action = "bind_snerv_shared_mlx_long_training_harness_then_reconsume"
    elif experiments:
        planner_action = "close_campaign_row_blockers_then_reconsume"
    else:
        planner_action = "rebuild_modelsize_ladder_and_campaign_plan"

    family_summary = _family_summary(experiments)
    blockers = _dedupe(blockers)
    return {
        "schema": RESULT_SCHEMA,
        "consumer_name": CONSUMER_NAME,
        "source_schema": source_schema,
        "baseline_to_beat": candidate.get("baseline_to_beat"),
        "queue_id": queue.get("queue_id"),
        "planner_action": planner_action,
        "top_priority_families": list(candidate.get("top_priority_families") or []),
        "campaign_row_count": len(campaign_rows),
        "experiment_count": len(experiments),
        "ready_local_mlx_experiment_count": len(ready_local),
        "blocked_experiment_count": len(blocked_experiments),
        "gated_experiment_count": len(gated_experiments),
        "cpu_replay_candidate_count": len(cpu_ready),
        "exact_gate_candidate_count": len(exact_ready),
        "family_summary": family_summary,
        "selected_local_mlx_experiment_ids": [
            str(row["id"]) for row in _sort_compact_experiments(ready_local)
        ],
        "selected_local_mlx_experiments": _sort_compact_experiments(ready_local),
        "blocked_exact_dispatch_dependencies": [
            "PR95_same_axis_control_replay_required_before_beat_claim",
            "PR101_and_Z5_terminal_adjudication_required_before_new_exact_full_video_cuda",
            "full600_byte_closed_receiver_proof_required_before_promotion",
            "paired_contest_cpu_cuda_pass_required_before_promotion",
        ],
        "local_mlx_route_recommended": bool(ready_local)
        and not _has_authority_overclaim(blockers)
        and "campaign_plan_schema_mismatch" not in blockers,
        "local_cpu_replay_recommended": False,
        "exact_auth_recommended": False,
        "storage_preflight_required": bool(ready_local),
        "lane_claim_required_before_execution": bool(ready_local),
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "Tier-A routing from tac.analysis.nerv_long_training_campaign_plan; "
            "local MLX acquisition only, no score or exact-dispatch authority."
        ),
        "axis_tag": "[planning/control]",
        "promotable": False,
        "score_claim": False,
        "score_claim_valid": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "confidence": 0.0
        if "campaign_plan_schema_mismatch" in blockers or _has_authority_overclaim(blockers)
        else (0.55 if ready_local else 0.2),
        "blockers": blockers,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render a concise operator artifact for the campaign consumer verdict."""

    selected = _mapping_list(result.get("selected_local_mlx_experiments"))
    lines = [
        "# NeRV Long-Training Campaign Consumer Verdict",
        "",
        f"Schema: `{result.get('schema')}`",
        f"Source schema: `{result.get('source_schema')}`",
        f"Planner action: `{result.get('planner_action')}`",
        f"Local MLX ready rows: `{result.get('ready_local_mlx_experiment_count')}`",
        f"Exact auth recommended: `{result.get('exact_auth_recommended')}`",
        f"Score claim: `{result.get('score_claim')}`",
        "",
        "## Selected Local MLX Rows",
        "",
    ]
    if selected:
        for row in selected:
            lines.append(
                f"- `{row.get('id')}` family=`{row.get('family')}` "
                f"priority=`{row.get('priority')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    blockers = _string_list(result.get("blockers"))
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append(
        "- none"
    )
    lines.append("")
    return "\n".join(lines)


def _local_mlx_ready(experiment: Mapping[str, Any]) -> bool:
    gate = _mapping(experiment.get("score_lowering_gate"))
    contract = _mapping(experiment.get("launch_authority_contract"))
    return (
        str(experiment.get("status") or "") in {"ready", "queued"}
        and experiment.get("blocked") is not True
        and contract.get("schema") == LAUNCH_AUTHORITY_CONTRACT_SCHEMA
        and contract.get("queue_status_is_local_mlx_plan") is True
        and contract.get("queue_status_is_runnable_plan") is True
        and not _string_list(contract.get("queue_launch_blockers"))
        and contract.get("queue_status_is_receiver_proof") is not True
        and contract.get("queue_status_is_cpu_replay_proof") is not True
        and contract.get("queue_status_is_exact_eval_authority") is not True
        and gate.get("local_mlx_executable") is True
        and gate.get("cpu_replay_ready") is not True
        and gate.get("exact_gate_ready") is not True
        and not _truthy_authority(experiment)
        and not _truthy_authority(gate)
        and bool(_mapping_list(experiment.get("steps")))
    )


def _compact_experiment(experiment: Mapping[str, Any], index: int) -> dict[str, Any]:
    steps = _mapping_list(experiment.get("steps"))
    first_step = steps[0] if steps else {}
    gate = _mapping(experiment.get("score_lowering_gate"))
    metadata = _mapping(experiment.get("metadata"))
    launch_contract = _mapping(experiment.get("launch_authority_contract"))
    return {
        "id": _experiment_id(experiment, index),
        "family": str(experiment.get("family") or "unknown"),
        "priority": _optional_int(experiment.get("priority")),
        "status": str(experiment.get("status") or ""),
        "command": list(first_step.get("command") or []),
        "postconditions": [
            dict(row)
            for row in _mapping_list(first_step.get("postconditions"))
        ],
        "score_lowering_gate": {
            "schema": gate.get("schema"),
            "local_mlx_executable": gate.get("local_mlx_executable") is True,
            "cpu_replay_ready": gate.get("cpu_replay_ready") is True,
            "exact_gate_ready": gate.get("exact_gate_ready") is True,
            "receiver_proof_required": gate.get("receiver_proof_required") is True,
            "full_video_prefilter_required": (
                gate.get("full_video_prefilter_required") is True
            ),
            "local_cpu_replay_required": gate.get("local_cpu_replay_required")
            is True,
            "exact_auth_gate_required": gate.get("exact_auth_gate_required") is True,
            "promotion_blockers": _string_list(gate.get("promotion_blockers")),
        },
        "launch_authority_contract": {
            "schema": launch_contract.get("schema"),
            "queue_status_is_local_mlx_plan": (
                launch_contract.get("queue_status_is_local_mlx_plan") is True
            ),
            "queue_status_is_runnable_plan": (
                launch_contract.get("queue_status_is_runnable_plan") is True
            ),
            "queue_launch_blockers": _string_list(
                launch_contract.get("queue_launch_blockers")
            ),
            "queue_status_is_receiver_proof": (
                launch_contract.get("queue_status_is_receiver_proof") is True
            ),
            "queue_status_is_cpu_replay_proof": (
                launch_contract.get("queue_status_is_cpu_replay_proof") is True
            ),
            "queue_status_is_exact_eval_authority": (
                launch_contract.get("queue_status_is_exact_eval_authority") is True
            ),
        },
        "metadata": _compact_metadata(metadata),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _compact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keep: dict[str, Any] = {
        "schema": "nerv_long_training_campaign_consumer_metadata.v1"
    }
    for key in (
        "feedback_launch_adjustment",
        "source_faithfulness_controls",
        "output_dir_reuse_policy",
    ):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            keep[key] = dict(value)
        elif value is not None:
            keep[key] = value
    keep.update(
        {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    return keep


def _family_summary(experiments: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for experiment in experiments:
        family = str(experiment.get("family") or "unknown")
        row = summary.setdefault(
            family,
            {
                "experiment_count": 0,
                "ready_local_mlx_count": 0,
                "blocked_count": 0,
                "gated_count": 0,
            },
        )
        row["experiment_count"] += 1
        if _local_mlx_ready(experiment):
            row["ready_local_mlx_count"] += 1
        if experiment.get("blocked") is True or str(experiment.get("status") or "") in {
            "blocked_dependency",
            "blocked",
            "disabled",
        }:
            row["blocked_count"] += 1
        if _experiment_has_gates(experiment):
            row["gated_count"] += 1
    return dict(sorted(summary.items()))


def _experiment_has_gates(experiment: Mapping[str, Any]) -> bool:
    gate = _mapping(experiment.get("score_lowering_gate"))
    return bool(
        _string_list(experiment.get("blockers"))
        or _string_list(gate.get("promotion_blockers"))
        or _string_list(gate.get("missing_requirement_ids"))
        or _string_list(gate.get("post_run_missing_requirement_ids"))
    )


def _launch_authority_contract_blockers(
    experiment: Mapping[str, Any],
    *,
    label: str,
) -> list[str]:
    contract = _mapping(experiment.get("launch_authority_contract"))
    blockers: list[str] = []
    if not contract:
        return [f"{label}_launch_authority_contract_missing"]
    if contract.get("schema") != LAUNCH_AUTHORITY_CONTRACT_SCHEMA:
        blockers.append(f"{label}_launch_authority_contract_schema_mismatch")
    if contract.get("queue_status_is_local_mlx_plan") is not True:
        blockers.append(f"{label}_launch_authority_contract_not_local_mlx_plan")
    if contract.get("queue_status_is_runnable_plan") is not True:
        blockers.append(f"{label}_launch_authority_contract_not_runnable_plan")
    launch_blockers = _string_list(contract.get("queue_launch_blockers"))
    if launch_blockers:
        blockers.append(f"{label}_launch_authority_contract_has_launch_blockers")
    for key in (
        "queue_status_is_receiver_proof",
        "queue_status_is_cpu_replay_proof",
        "queue_status_is_exact_eval_authority",
    ):
        if contract.get(key) is True:
            blockers.append(f"{label}_{key}_overclaimed")
    if _truthy_authority(contract):
        blockers.append(f"{label}_launch_authority_contract_false_authority")
    return blockers


def _sort_compact_experiments(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    family_rank = {"hi_nerv": 0, "snerv": 1}
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: (
                _optional_int(row.get("priority")) or 999,
                family_rank.get(str(row.get("family") or ""), 99),
                str(row.get("id") or ""),
            ),
        )
    ]


def _snerv_binding_gap(
    blockers: Sequence[str],
    experiments: Sequence[Mapping[str, Any]],
) -> bool:
    if any("snerv_shared_mlx_scoreaware_long_training_harness_not_bound" in b for b in blockers):
        return True
    return any(
        str(experiment.get("family") or "") == "snerv"
        and str(experiment.get("status") or "")
        in {"blocked_dependency", "blocked", "disabled"}
        for experiment in experiments
    )


def _authority_blockers(payload: Mapping[str, Any], *, label: str) -> list[str]:
    blockers: list[str] = []
    for field in _AUTHORITY_FIELDS:
        if payload.get(field) is True:
            blockers.append(f"{label}_{field}_overclaimed")
    return blockers


def _has_authority_overclaim(blockers: Sequence[str]) -> bool:
    return any(str(blocker).endswith("_overclaimed") for blocker in blockers)


def _truthy_authority(payload: Mapping[str, Any]) -> bool:
    return any(payload.get(field) is True for field in _AUTHORITY_FIELDS)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _experiment_id(experiment: Mapping[str, Any], index: int) -> str:
    return str(experiment.get("id") or f"experiment_{index}")


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "PLAN_SCHEMA",
    "QUEUE_SCHEMA",
    "RESULT_SCHEMA",
    "consume_candidate",
    "render_markdown",
    "update_from_anchor",
]
