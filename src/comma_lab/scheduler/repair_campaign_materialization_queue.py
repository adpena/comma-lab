# SPDX-License-Identifier: MIT
"""Build byte-closed materialization queues for repair-campaign allocations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from comma_lab.scheduler.frontier_rate_attack_feedback import (
    REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA,
    REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
    REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA,
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
)

REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA = (
    "repair_campaign_byte_closed_materialization_queue_metadata.v1"
)
REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA = (
    "repair_campaign_byte_closed_materialization_experiment_metadata.v1"
)
REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA = (
    "repair_campaign_byte_closed_materialization_gate.v1"
)


class RepairCampaignMaterializationQueueError(ValueError):
    """Raised when a repair campaign materialization queue cannot be built."""


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path)
    repo = Path(repo_root)
    try:
        return str(resolved.resolve(strict=False).relative_to(repo.resolve(strict=False)))
    except ValueError:
        return str(resolved)


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(repo_root) / value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _find_score_row(
    score_report: Mapping[str, Any],
    *,
    typed_response_id: str,
) -> Mapping[str, Any]:
    for row in score_report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("typed_response_id") or "") == typed_response_id:
            return row
    return {}


def _lineage_ready_for_materialization(lineage: Mapping[str, Any]) -> bool:
    return lineage.get("local_mlx_advisory_custody_ready") is True


def _allocation_blockers(
    *,
    allocation: Mapping[str, Any],
    score_row: Mapping[str, Any],
    work_order_path: Path,
) -> list[str]:
    blockers: list[str] = []
    if not str(allocation.get("typed_response_id") or "").strip():
        blockers.append("typed_response_id_missing")
    if not score_row:
        blockers.append("source_score_row_missing")
    if not work_order_path.is_file():
        blockers.append("repair_budget_waterfill_work_order_missing")
    lineage = _mapping(allocation.get("repair_materialization_lineage"))
    if not lineage:
        blockers.append("repair_materialization_lineage_missing")
    elif not _lineage_ready_for_materialization(lineage):
        blockers.append("local_mlx_advisory_custody_missing")
    return ordered_unique(
        [
            *blockers,
            *_string_list(
                _mapping(allocation.get("repair_materialization_lineage")).get(
                    "execution_gate_missing_artifacts"
                )
            ),
        ]
    )


def _gate_command(
    *,
    execution_report_ref: str,
    gate_ref: str,
    typed_response_id: str,
    candidate_id: str,
) -> list[str]:
    return [
        ".venv/bin/python",
        "-c",
        (
            "import json, sys; from pathlib import Path; "
            "report = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
            "out = Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True); "
            "payload = {"
            f"'schema': '{REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA}', "
            "'typed_response_id': sys.argv[3], "
            "'candidate_id': sys.argv[4], "
            "'source_execution_report_path': sys.argv[1], "
            "'candidate_archive_materialized': "
            "report.get('candidate_archive_materialized') is True, "
            "'archive_bound_runtime_consumption_proof_ready': "
            "report.get('runtime_consumption_proof_present') is True "
            "and report.get('receiver_consumed') is True, "
            "'component_response_replayed': "
            "report.get('component_response_replayed') is True, "
            "'exact_eval_handoff_eligible': False, "
            "'budget_spend_allowed': False, "
            "'ready_for_budget_spend': False, "
            "'ready_for_exact_eval_dispatch': False, "
            "'score_claim': False, "
            "'promotion_eligible': False, "
            "'rank_or_kill_eligible': False, "
            "'blockers': report.get('blockers') or []}; "
            "out.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\\n', "
            "encoding='utf-8')"
        ),
        execution_report_ref,
        gate_ref,
        typed_response_id,
        candidate_id,
    ]


def _materialization_experiment(
    *,
    score_report: Mapping[str, Any],
    score_report_path: str | Path,
    work_order_path: str | Path,
    allocation: Mapping[str, Any],
    repo_root: str | Path,
    queue_root: Path,
    priority: int,
    materializer_work_queue: str | Path | None,
    materializer_execution_queue: str | Path | None,
    repair_palette_modes: Sequence[str],
    posterior_path: str | Path,
    posterior_lock_path: str | Path,
) -> dict[str, Any]:
    typed_response_id = str(allocation.get("typed_response_id") or "").strip()
    candidate_id = str(allocation.get("candidate_id") or "").strip()
    score_row = _find_score_row(score_report, typed_response_id=typed_response_id)
    work_order = _resolve(work_order_path, repo_root)
    stem = _slug(typed_response_id or candidate_id or f"allocation_{priority}")
    row_root = queue_root / stem
    materialization_plan = row_root / "repair_budget_materialization_plan.json"
    child_replay_manifest = (
        row_root / "repair_budget_child_component_replay_manifests.json"
    )
    family_materializer_manifest = row_root / "repair_family_materializer_manifest.json"
    binding_report = row_root / "repair_budget_materializer_binding_report.json"
    execution_report = row_root / "repair_budget_materialization_execution_report.json"
    gate = row_root / "repair_campaign_byte_closed_materialization_gate.json"
    materialization_learning_signal = (
        row_root / "repair_materialization_learning_signal_report.json"
    )
    posterior_append_report = (
        row_root / "repair_materialization_posterior_append_report.json"
    )
    materialization_plan_ref = _repo_rel(materialization_plan, repo_root)
    child_replay_ref = _repo_rel(child_replay_manifest, repo_root)
    family_materializer_ref = _repo_rel(family_materializer_manifest, repo_root)
    binding_report_ref = _repo_rel(binding_report, repo_root)
    execution_report_ref = _repo_rel(execution_report, repo_root)
    gate_ref = _repo_rel(gate, repo_root)
    materialization_learning_signal_ref = _repo_rel(
        materialization_learning_signal,
        repo_root,
    )
    posterior_append_report_ref = _repo_rel(posterior_append_report, repo_root)
    posterior_ref = _repo_rel(posterior_path, repo_root)
    posterior_lock_ref = _repo_rel(posterior_lock_path, repo_root)
    work_order_ref = _repo_rel(work_order, repo_root)
    materializer_work_queue_ref = (
        _repo_rel(_resolve(materializer_work_queue, repo_root), repo_root)
        if materializer_work_queue is not None
        else None
    )
    materializer_execution_queue_ref = (
        _repo_rel(_resolve(materializer_execution_queue, repo_root), repo_root)
        if materializer_execution_queue is not None
        else None
    )
    blockers = _allocation_blockers(
        allocation=allocation,
        score_row=score_row,
        work_order_path=work_order,
    )
    lineage = _mapping(allocation.get("repair_materialization_lineage"))
    dynamics = _mapping(_mapping(allocation.get("multiscale_action_row")).get("interaction_dynamics"))
    status = "queued" if not blockers else "frozen"
    metadata = {
        "schema": REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA,
        "source_score_report_path": str(score_report_path),
        "source_work_order_path": work_order_ref,
        "typed_response_id": typed_response_id or None,
        "candidate_id": candidate_id or None,
        "family_id": allocation.get("family_id"),
        "entropy_position_label": allocation.get("entropy_position_label"),
        "allocated_repair_bytes": allocation.get("allocated_repair_bytes"),
        "repair_materialization_lineage": dict(lineage),
        "interaction_dynamics": dict(dynamics),
        "materialization_plan_path": materialization_plan_ref,
        "child_component_replay_manifests_path": child_replay_ref,
        "repair_family_materializer_manifest_path": family_materializer_ref,
        "repair_family_materializer_manifest_schema": (
            REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
        ),
        "materializer_binding_report_path": binding_report_ref,
        "materialization_execution_report_path": execution_report_ref,
        "byte_closed_materialization_gate_path": gate_ref,
        "byte_closed_materialization_gate_schema": (
            REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA
        ),
        "materialization_learning_signal_report_path": (
            materialization_learning_signal_ref
        ),
        "materialization_learning_signal_report_schema": (
            REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
        ),
        "posterior_path": posterior_ref,
        "posterior_lock_path": posterior_lock_ref,
        "posterior_append_report_path": posterior_append_report_ref,
        "posterior_append_report_schema": (
            REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA
        ),
        "materializer_work_queue_path": materializer_work_queue_ref,
        "materializer_execution_queue_path": materializer_execution_queue_ref,
        "queue_actuation_ready": not blockers,
        "queue_actuation_blockers": ordered_unique(blockers),
        "local_mlx_advisory_custody_required": True,
        "local_mlx_rows_are_advisory_only": True,
        "exact_eval_handoff_requires_complete_archive_runtime_component_custody": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_byte_closed_materialization_queue_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context=f"repair_campaign_byte_closed_materialization:{typed_response_id}",
    )
    if blockers:
        steps = [
            {
                "id": "inspect_missing_repair_materialization_prerequisites",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'schema': 'repair_campaign_materialization_blocker.v1', "
                        "'budget_spend_allowed': False, "
                        "'ready_for_exact_eval_dispatch': False, "
                        "'score_claim': False, "
                        "'promotion_eligible': False, "
                        "'rank_or_kill_eligible': False, "
                        f"'blockers': {blockers!r}"
                        "}, sort_keys=True))"
                    ),
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
                "telemetry": {"include_postcondition_paths": True},
            }
        ]
    else:
        steps = [
            {
                "id": "emit_repair_budget_materialization_plan",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_repair_budget_materialization_plan.py",
                    "--work-order",
                    work_order_ref,
                    "--materialization-plan-out",
                    materialization_plan_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": materialization_plan_ref,
                        "key": "schema",
                        "equals": REPAIR_BUDGET_MATERIALIZATION_PLAN_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": materialization_plan_ref},
                    {
                        "type": "json_equals",
                        "path": materialization_plan_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [materialization_plan_ref],
                    "input_artifact_paths": [work_order_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "emit_repair_family_materializer_manifest",
                "kind": "command",
                "requires": ["emit_repair_budget_materialization_plan"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_campaign_family_materializer_manifest.py",
                    "--materialization-plan",
                    materialization_plan_ref,
                    "--score-report",
                    str(score_report_path),
                    "--typed-response-id",
                    typed_response_id,
                    "--candidate-id",
                    candidate_id,
                    "--materializer-manifest-out",
                    family_materializer_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": family_materializer_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": family_materializer_ref},
                ],
                "telemetry": {
                    "artifact_paths": [family_materializer_ref],
                    "input_artifact_paths": [materialization_plan_ref, str(score_report_path)],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "emit_repair_budget_child_component_replay_manifests",
                "kind": "command",
                "requires": ["emit_repair_budget_materialization_plan"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_repair_budget_child_component_replay_manifests.py",
                    "--materialization-plan",
                    materialization_plan_ref,
                    "--output-manifest",
                    child_replay_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": child_replay_ref,
                        "key": "schema",
                        "equals": REPAIR_BUDGET_CHILD_COMPONENT_REPLAY_MANIFESTS_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": child_replay_ref},
                ],
                "telemetry": {
                    "artifact_paths": [child_replay_ref],
                    "input_artifact_paths": [materialization_plan_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "bind_repair_budget_materializer_execution",
                "kind": "command",
                "requires": [
                    "emit_repair_family_materializer_manifest",
                    "emit_repair_budget_child_component_replay_manifests",
                ],
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_repair_budget_materializer_binding_report.py",
                    "--materialization-plan",
                    materialization_plan_ref,
                    "--binding-report-out",
                    binding_report_ref,
                    "--materializer-manifest",
                    child_replay_ref,
                    "--materializer-manifest",
                    family_materializer_ref,
                    *(
                        ["--materializer-work-queue", materializer_work_queue_ref]
                        if materializer_work_queue_ref
                        else []
                    ),
                    *(
                        [
                            "--materializer-execution-queue",
                            materializer_execution_queue_ref,
                        ]
                        if materializer_execution_queue_ref
                        else []
                    ),
                    *[
                        item
                        for mode in repair_palette_modes
                        for item in ("--repair-palette-mode", mode)
                    ],
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": binding_report_ref,
                        "key": "schema",
                        "equals": REPAIR_BUDGET_MATERIALIZER_BINDING_REPORT_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": binding_report_ref},
                ],
                "telemetry": {
                    "artifact_paths": [binding_report_ref],
                    "input_artifact_paths": ordered_unique(
                        [
                            materialization_plan_ref,
                            child_replay_ref,
                            family_materializer_ref,
                            materializer_work_queue_ref,
                            materializer_execution_queue_ref,
                        ]
                    ),
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "audit_repair_budget_materialization_execution",
                "kind": "command",
                "requires": ["bind_repair_budget_materializer_execution"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_frontier_repair_budget_materialization_execution_report.py",
                    "--materialization-plan",
                    materialization_plan_ref,
                    "--materializer-binding-report",
                    binding_report_ref,
                    "--execution-report-out",
                    execution_report_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_io_heavy"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": execution_report_ref,
                        "key": "schema",
                        "equals": REPAIR_BUDGET_MATERIALIZATION_EXECUTION_REPORT_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": execution_report_ref},
                    {
                        "type": "json_equals",
                        "path": execution_report_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [execution_report_ref],
                    "input_artifact_paths": [materialization_plan_ref, binding_report_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "emit_selected_repair_materialization_gate",
                "kind": "command",
                "requires": ["audit_repair_budget_materialization_execution"],
                "command": _gate_command(
                    execution_report_ref=execution_report_ref,
                    gate_ref=gate_ref,
                    typed_response_id=typed_response_id,
                    candidate_id=candidate_id,
                ),
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": gate_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": gate_ref},
                    {
                        "type": "json_equals",
                        "path": gate_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [gate_ref],
                    "input_artifact_paths": [execution_report_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "build_repair_materialization_learning_signal",
                "kind": "command",
                "requires": ["emit_selected_repair_materialization_gate"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_campaign_blocked_learning_signals.py",
                    "--materialization-execution-report",
                    execution_report_ref,
                    "--materialization-gate",
                    gate_ref,
                    "--family-materializer-manifest",
                    family_materializer_ref,
                    "--blocked-signal-report-out",
                    materialization_learning_signal_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": materialization_learning_signal_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": materialization_learning_signal_ref,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [materialization_learning_signal_ref],
                    "input_artifact_paths": [
                        execution_report_ref,
                        gate_ref,
                        family_materializer_ref,
                    ],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "append_repair_materialization_posterior_signal",
                "kind": "command",
                "requires": ["build_repair_materialization_learning_signal"],
                "command": [
                    ".venv/bin/python",
                    "tools/append_repair_campaign_blocked_posterior.py",
                    "--blocked-learning-signal-report",
                    materialization_learning_signal_ref,
                    "--posterior-path",
                    posterior_ref,
                    "--lock-path",
                    posterior_lock_ref,
                    "--report-out",
                    posterior_append_report_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": posterior_append_report_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": posterior_append_report_ref},
                    {"type": "jsonl_false_authority", "path": posterior_ref},
                ],
                "telemetry": {
                    "artifact_paths": [posterior_append_report_ref, posterior_ref],
                    "input_artifact_paths": [materialization_learning_signal_ref],
                    "include_postcondition_paths": True,
                },
            },
        ]
    return {
        "id": f"repair_materialize_{stem}",
        "priority": priority,
        "status": status,
        "tags": [
            "frontier-rate-attack",
            "repair-campaign-byte-closed-materialization",
            "encoder-side-repair",
            "no-score-authority",
        ],
        "metadata": metadata,
        "steps": steps,
    }


def _empty_selection_experiment(
    *,
    score_report_path: str | Path,
    repo_root: str | Path,
    queue_root: Path,
) -> dict[str, Any]:
    gate = queue_root / "blocked_no_selected_allocations_materialization_gate.json"
    blockers = ["optimizer_selected_allocation_rows_empty"]
    metadata = {
        "schema": REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA,
        "source_score_report_path": str(score_report_path),
        "typed_response_id": None,
        "candidate_id": None,
        "byte_closed_materialization_gate_path": _repo_rel(gate, repo_root),
        "queue_actuation_ready": False,
        "queue_actuation_blockers": blockers,
        "local_mlx_rows_are_advisory_only": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "blocked_empty_repair_materialization_selection_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_campaign_byte_closed_materialization_empty_selection",
    )
    return {
        "id": "repair_materialization_no_selected_allocations",
        "priority": 1,
        "status": "frozen",
        "tags": [
            "frontier-rate-attack",
            "repair-campaign-byte-closed-materialization",
            "blocked",
            "no-score-authority",
        ],
        "metadata": metadata,
        "steps": [
            {
                "id": "inspect_empty_repair_materialization_selection",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'schema': 'repair_campaign_materialization_blocker.v1', "
                        "'budget_spend_allowed': False, "
                        "'ready_for_exact_eval_dispatch': False, "
                        f"'blockers': {blockers!r}"
                        "}, sort_keys=True))"
                    ),
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
                "telemetry": {"include_postcondition_paths": True},
            }
        ],
    }


def build_repair_campaign_byte_closed_materialization_queue(
    *,
    repo_root: str | Path,
    score_report: Mapping[str, Any],
    score_report_path: str | Path,
    work_order_path: str | Path,
    results_root: str | Path,
    queue_id: str = "repair_campaign_byte_closed_materialization_queue",
    experiment_limit: int | None = None,
    materializer_work_queue: str | Path | None = None,
    materializer_execution_queue: str | Path | None = None,
    repair_palette_modes: Sequence[str] = (),
    posterior_path: str | Path = DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    posterior_lock_path: str | Path = (
        DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH
    ),
) -> dict[str, Any]:
    """Build queue rows that attempt byte-closed repair candidate materialization."""

    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignMaterializationQueueError(
            "materialization queue requires repair_campaign_score_report.v1"
        )
    require_no_truthy_authority_fields(
        score_report,
        context="repair_campaign_byte_closed_materialization_queue_input",
    )
    decision = _mapping(score_report.get("optimizer_decision"))
    if decision.get("schema") != REPAIR_CAMPAIGN_OPTIMIZER_DECISION_SCHEMA:
        raise RepairCampaignMaterializationQueueError(
            "score report missing repair_campaign_optimizer_decision.v1"
        )
    allocations = [
        row
        for row in decision.get("selected_allocation_rows") or []
        if isinstance(row, Mapping)
    ]
    if experiment_limit is not None:
        allocations = allocations[: max(0, int(experiment_limit))]
    queue_root = (
        _resolve(results_root, repo_root)
        / "repair_campaign_byte_closed_materialization_queue"
        / _slug(queue_id)
    )
    resolved_posterior_path = _resolve(posterior_path, repo_root)
    resolved_posterior_lock_path = _resolve(posterior_lock_path, repo_root)
    experiments = (
        [
            _materialization_experiment(
                score_report=score_report,
                score_report_path=score_report_path,
                work_order_path=work_order_path,
                allocation=allocation,
                repo_root=repo_root,
                queue_root=queue_root,
                priority=priority,
                materializer_work_queue=materializer_work_queue,
                materializer_execution_queue=materializer_execution_queue,
                repair_palette_modes=repair_palette_modes,
                posterior_path=resolved_posterior_path,
                posterior_lock_path=resolved_posterior_lock_path,
            )
            for priority, allocation in enumerate(allocations, start=1)
        ]
        if allocations
        else [
            _empty_selection_experiment(
                score_report_path=score_report_path,
                repo_root=repo_root,
                queue_root=queue_root,
            )
        ]
    )
    ready_count = sum(
        1
        for experiment in experiments
        if experiment["metadata"].get("queue_actuation_ready") is True
    )
    blockers = ordered_unique(
        blocker
        for experiment in experiments
        for blocker in _string_list(
            experiment["metadata"].get("queue_actuation_blockers")
        )
    )
    if not allocations:
        blockers.append("optimizer_selected_allocation_rows_empty")
    metadata = {
        "schema": REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA,
        "source_score_report_path": str(score_report_path),
        "source_score_report_schema": score_report.get("schema"),
        "source_optimizer_decision_schema": decision.get("schema"),
        "source_work_order_path": _repo_rel(_resolve(work_order_path, repo_root), repo_root),
        "queue_id": queue_id,
        "experiment_count": len(experiments),
        "ready_experiment_count": ready_count,
        "blocked_experiment_count": len(experiments) - ready_count,
        "selected_allocation_count": len(allocations),
        "results_root": _repo_rel(queue_root, repo_root),
        "posterior_path": _repo_rel(resolved_posterior_path, repo_root),
        "posterior_lock_path": _repo_rel(resolved_posterior_lock_path, repo_root),
        "materialization_learning_signal_report_schema": (
            REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
        ),
        "posterior_append_report_schema": (
            REPAIR_CAMPAIGN_BLOCKED_POSTERIOR_APPEND_REPORT_SCHEMA
        ),
        "queue_actuation_blockers": ordered_unique(blockers),
        "local_mlx_rows_are_advisory_only": True,
        "exact_eval_handoff_requires_complete_archive_runtime_component_custody": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_byte_closed_materialization_queue",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_campaign_byte_closed_materialization_queue_metadata",
    )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_io_heavy": 1},
            },
            "metadata": metadata,
            "experiments": experiments,
        }
    )


__all__ = [
    "REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_EXPERIMENT_METADATA_SCHEMA",
    "REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_GATE_SCHEMA",
    "REPAIR_CAMPAIGN_BYTE_CLOSED_MATERIALIZATION_QUEUE_METADATA_SCHEMA",
    "REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA",
    "RepairCampaignMaterializationQueueError",
    "build_repair_campaign_byte_closed_materialization_queue",
]
