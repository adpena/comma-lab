# SPDX-License-Identifier: MIT
"""Build queue-owned local stackability probes for repair-campaign allocations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_chain_contract import (
    REPAIR_CAMPAIGN_REQUIRED_OPTIMIZER_SOLVER,
    require_interaction_aware_optimizer_decision,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
)
from tac.optimization.repair_campaign_posterior import (
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA,
)
from tac.optimization.repair_campaign_replay_bundle import (
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA,
)
from tac.optimization.repair_campaign_scorer import (
    REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
)

REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA = (
    "repair_campaign_stackability_queue_metadata.v1"
)
REPAIR_CAMPAIGN_STACKABILITY_EXPERIMENT_METADATA_SCHEMA = (
    "repair_campaign_stackability_experiment_metadata.v1"
)


class RepairCampaignStackabilityQueueError(ValueError):
    """Raised when a repair campaign stackability queue cannot be built."""


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


def _safe_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def _custody_blockers(score_row: Mapping[str, Any]) -> list[str]:
    gate = _mapping(score_row.get("execution_gate"))
    blockers = []
    if gate.get("recommended_queue_status") != "ready_for_local_mlx_advisory_execution":
        blockers.append("local_mlx_advisory_custody_missing")
    custody_paths = [
        item
        for item in gate.get("local_mlx_custody_paths") or []
        if isinstance(item, Mapping)
    ]
    required_keys = {"local_mlx_response_path", "reference_local_mlx_response_path"}
    required = [
        item for item in custody_paths if str(item.get("key") or "") in required_keys
    ]
    if not required:
        blockers.append("local_mlx_required_custody_paths_missing")
    for item in required:
        if item.get("exists") is not True:
            blockers.append(f"{item.get('key')}:missing_or_unverified")
    return ordered_unique([*blockers, *_string_list(gate.get("missing_artifacts"))])


def _stackability_experiment(
    *,
    score_report: Mapping[str, Any],
    score_report_path: str | Path,
    allocation: Mapping[str, Any],
    repo_root: str | Path,
    queue_root: Path,
    priority: int,
    posterior_path: str | Path,
    posterior_lock_path: str | Path,
) -> dict[str, Any]:
    typed_response_id = str(allocation.get("typed_response_id") or "").strip()
    score_row = _find_score_row(score_report, typed_response_id=typed_response_id)
    probe_path = (
        queue_root
        / _slug(typed_response_id)
        / "repair_campaign_stackability_probe.json"
    )
    probe_ref = _repo_rel(probe_path, repo_root)
    replay_bundle_path = (
        queue_root
        / _slug(typed_response_id)
        / "repair_campaign_stackability_replay_bundle.json"
    )
    replay_bundle_ref = _repo_rel(replay_bundle_path, repo_root)
    replay_rerun_dir = queue_root / _slug(typed_response_id) / "replay_rerun"
    replay_rerun_dir_ref = _repo_rel(replay_rerun_dir, repo_root)
    replay_rerun_summary_path = (
        queue_root
        / _slug(typed_response_id)
        / "repair_campaign_stackability_replay_rerun_summary.json"
    )
    replay_rerun_summary_ref = _repo_rel(replay_rerun_summary_path, repo_root)
    learning_signal_path = (
        queue_root
        / _slug(typed_response_id)
        / "repair_campaign_learning_signal.json"
    )
    learning_signal_ref = _repo_rel(learning_signal_path, repo_root)
    posterior_append_report_path = (
        queue_root
        / _slug(typed_response_id)
        / "repair_campaign_posterior_append_report.json"
    )
    posterior_append_report_ref = _repo_rel(posterior_append_report_path, repo_root)
    posterior_ref = _repo_rel(posterior_path, repo_root)
    posterior_lock_ref = _repo_rel(posterior_lock_path, repo_root)
    allocation_blockers: list[str] = []
    if not typed_response_id:
        allocation_blockers.append("typed_response_id_missing")
    if not score_row:
        allocation_blockers.append("source_score_row_missing")
    if _safe_int(allocation.get("allocated_repair_bytes")) <= 0:
        allocation_blockers.append("allocated_repair_bytes_missing")
    blockers = ordered_unique([*allocation_blockers, *_custody_blockers(score_row)])
    queue_actuation_ready = not blockers
    status = "queued" if queue_actuation_ready else "frozen"
    multiscale_action_row = _mapping(allocation.get("multiscale_action_row"))
    metadata = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_EXPERIMENT_METADATA_SCHEMA,
        "source_score_report_path": str(score_report_path),
        "typed_response_id": typed_response_id or None,
        "candidate_id": allocation.get("candidate_id"),
        "family_id": allocation.get("family_id"),
        "entropy_position_label": allocation.get("entropy_position_label"),
        "allocated_repair_bytes": _safe_int(allocation.get("allocated_repair_bytes")),
        "repair_materialization_lineage": dict(
            _mapping(allocation.get("repair_materialization_lineage"))
        ),
        "materialization_missing_artifacts": _string_list(
            allocation.get("materialization_missing_artifacts")
        ),
        "multiscale_action_row": dict(multiscale_action_row),
        "interaction_dynamics": dict(
            _mapping(multiscale_action_row.get("interaction_dynamics"))
        ),
        "probe_output_path": probe_ref,
        "stackability_probe_schema": REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
        "replay_bundle_path": replay_bundle_ref,
        "replay_bundle_schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
        "replay_rerun_dir": replay_rerun_dir_ref,
        "replay_rerun_summary_path": replay_rerun_summary_ref,
        "replay_rerun_summary_schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA,
        "learning_signal_path": learning_signal_ref,
        "learning_signal_schema": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
        "posterior_path": posterior_ref,
        "posterior_lock_path": posterior_lock_ref,
        "posterior_append_report_path": posterior_append_report_ref,
        "posterior_append_report_schema": (
            REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
        ),
        "queue_actuation_ready": queue_actuation_ready,
        "queue_actuation_blockers": blockers,
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "queue_owned_local_repair_stackability_probe_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context=f"repair_campaign_stackability_experiment:{typed_response_id}",
    )
    if blockers:
        steps = [
            {
                "id": "inspect_missing_repair_stackability_prerequisites",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'schema': 'repair_campaign_stackability_blocker.v1', "
                        "'score_claim': False, "
                        "'promotion_eligible': False, "
                        "'rank_or_kill_eligible': False, "
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
        ]
    else:
        probe_command = [
            ".venv/bin/python",
            "tools/run_repair_campaign_stackability_probe.py",
            "--score-report",
            str(score_report_path),
            "--typed-response-id",
            typed_response_id,
            "--output",
            probe_ref,
            "--overwrite",
        ]
        steps = [
            {
                "id": "emit_repair_campaign_stackability_probe",
                "kind": "command",
                "command": probe_command,
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": probe_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": probe_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": probe_ref,
                        "key": "stackability_ready",
                        "equals": True,
                    },
                    {
                        "type": "json_equals",
                        "path": probe_ref,
                        "key": "budget_spend_allowed",
                        "equals": False,
                    },
                    {
                        "type": "json_equals",
                        "path": probe_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [probe_ref],
                    "input_artifact_paths": [str(score_report_path)],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "build_repair_campaign_stackability_replay_bundle",
                "kind": "command",
                "requires": ["emit_repair_campaign_stackability_probe"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_campaign_stackability_replay_bundle.py",
                    "--score-report",
                    str(score_report_path),
                    "--probe",
                    probe_ref,
                    "--bundle-out",
                    replay_bundle_ref,
                    "--probe-command-json",
                    json.dumps(probe_command, separators=(",", ":")),
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": replay_bundle_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": replay_bundle_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": replay_bundle_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [replay_bundle_ref],
                    "input_artifact_paths": [str(score_report_path), probe_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "rerun_repair_campaign_stackability_replay_bundle",
                "kind": "command",
                "requires": ["build_repair_campaign_stackability_replay_bundle"],
                "command": [
                    ".venv/bin/python",
                    "tools/rerun_repair_campaign_stackability_replay_bundle.py",
                    "--bundle",
                    replay_bundle_ref,
                    "--output-dir",
                    replay_rerun_dir_ref,
                    "--summary-out",
                    replay_rerun_summary_ref,
                    "--run-id",
                    _slug(typed_response_id),
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": replay_rerun_summary_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": replay_rerun_summary_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": replay_rerun_summary_ref,
                        "key": "matched",
                        "equals": True,
                    },
                    {
                        "type": "json_equals",
                        "path": replay_rerun_summary_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [replay_rerun_summary_ref],
                    "input_artifact_paths": [replay_bundle_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "build_repair_campaign_learning_signal",
                "kind": "command",
                "requires": ["rerun_repair_campaign_stackability_replay_bundle"],
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_campaign_learning_signal.py",
                    "--score-report",
                    str(score_report_path),
                    "--probe",
                    probe_ref,
                    "--replay-bundle",
                    replay_bundle_ref,
                    "--signal-out",
                    learning_signal_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": learning_signal_ref,
                        "key": "schema",
                        "equals": REPAIR_CAMPAIGN_LEARNING_SIGNAL_SCHEMA,
                    },
                    {
                        "type": "json_false_authority",
                        "path": learning_signal_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": learning_signal_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [learning_signal_ref],
                    "input_artifact_paths": [
                        str(score_report_path),
                        probe_ref,
                        replay_bundle_ref,
                    ],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "append_repair_campaign_stackability_posterior",
                "kind": "command",
                "requires": ["build_repair_campaign_learning_signal"],
                "command": [
                    ".venv/bin/python",
                    "tools/append_repair_campaign_stackability_posterior.py",
                    "--learning-signal",
                    learning_signal_ref,
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
                        "equals": (
                            REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
                        ),
                    },
                    {
                        "type": "json_false_authority",
                        "path": posterior_append_report_ref,
                    },
                    {
                        "type": "jsonl_false_authority",
                        "path": posterior_ref,
                    },
                    {
                        "type": "json_equals",
                        "path": posterior_append_report_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [posterior_append_report_ref, posterior_ref],
                    "input_artifact_paths": [learning_signal_ref],
                    "include_postcondition_paths": True,
                },
            }
        ]
    return {
        "id": f"repair_stackability_{_slug(typed_response_id)}",
        "priority": priority,
        "status": status,
        "tags": [
            "frontier-rate-attack",
            "repair-campaign-stackability",
            "local-mlx-advisory-custody",
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
    probe_path = queue_root / "blocked_no_selected_allocations_probe.json"
    blockers = ["optimizer_selected_allocation_rows_empty"]
    metadata = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_EXPERIMENT_METADATA_SCHEMA,
        "source_score_report_path": str(score_report_path),
        "typed_response_id": None,
        "candidate_id": None,
        "family_id": None,
        "entropy_position_label": None,
        "allocated_repair_bytes": 0,
        "probe_output_path": _repo_rel(probe_path, repo_root),
        "stackability_probe_schema": REPAIR_CAMPAIGN_STACKABILITY_PROBE_SCHEMA,
        "queue_actuation_ready": False,
        "queue_actuation_blockers": blockers,
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "blocked_empty_repair_stackability_selection_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_campaign_stackability_empty_selection_experiment",
    )
    return {
        "id": "repair_stackability_no_selected_allocations",
        "priority": 1,
        "status": "frozen",
        "tags": [
            "frontier-rate-attack",
            "repair-campaign-stackability",
            "blocked-empty-selection",
            "no-score-authority",
        ],
        "metadata": metadata,
        "steps": [
            {
                "id": "inspect_empty_repair_stackability_selection",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'schema': 'repair_campaign_stackability_blocker.v1', "
                        "'score_claim': False, "
                        "'promotion_eligible': False, "
                        "'rank_or_kill_eligible': False, "
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


def build_repair_campaign_stackability_queue(
    *,
    repo_root: str | Path,
    score_report: Mapping[str, Any],
    score_report_path: str | Path,
    results_root: str | Path,
    queue_id: str = "repair_campaign_stackability_queue",
    experiment_limit: int | None = None,
    posterior_path: str | Path = DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    posterior_lock_path: str | Path = (
        DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH
    ),
) -> dict[str, Any]:
    """Build local probe rows for optimizer-selected repair allocations."""

    if score_report.get("schema") != REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        raise RepairCampaignStackabilityQueueError(
            "stackability queue requires repair_campaign_score_report.v1"
        )
    require_no_truthy_authority_fields(
        score_report,
        context="repair_campaign_stackability_queue_input",
    )
    decision = _mapping(score_report.get("optimizer_decision"))
    require_interaction_aware_optimizer_decision(
        decision,
        context="repair_campaign_stackability_queue",
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
        / "repair_campaign_stackability_queue"
        / _slug(queue_id)
    )
    resolved_posterior_path = _resolve(posterior_path, repo_root)
    resolved_posterior_lock_path = _resolve(posterior_lock_path, repo_root)
    experiments = (
        [
            _stackability_experiment(
                score_report=score_report,
                score_report_path=score_report_path,
                allocation=allocation,
                repo_root=repo_root,
                queue_root=queue_root,
                priority=priority,
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
    automation_rollup = {
        "schema": "repair_campaign_stackability_operator_automation_rollup.v1",
        "queue_id": queue_id,
        "selected_allocation_count": len(allocations),
        "ready_experiment_count": ready_count,
        "blocked_experiment_count": len(experiments) - ready_count,
        "required_optimizer_solver": REPAIR_CAMPAIGN_REQUIRED_OPTIMIZER_SOLVER,
        "source_optimizer_solver": decision.get("solver"),
        "stale_solver_contract_rejected": True,
        "local_mlx_rows_are_advisory_only": True,
        "posterior_append_default": True,
        "queue_actuation_blockers": ordered_unique(blockers),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    metadata = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA,
        "source_score_report_path": str(score_report_path),
        "source_score_report_schema": score_report.get("schema"),
        "source_optimizer_decision_schema": decision.get("schema"),
        "source_optimizer_solver": decision.get("solver"),
        "required_optimizer_solver": REPAIR_CAMPAIGN_REQUIRED_OPTIMIZER_SOLVER,
        "stale_solver_contract_rejected": True,
        "queue_id": queue_id,
        "experiment_count": len(experiments),
        "ready_experiment_count": ready_count,
        "blocked_experiment_count": len(experiments) - ready_count,
        "selected_allocation_count": len(allocations),
        "results_root": _repo_rel(queue_root, repo_root),
        "posterior_path": _repo_rel(resolved_posterior_path, repo_root),
        "posterior_lock_path": _repo_rel(resolved_posterior_lock_path, repo_root),
        "posterior_append_report_schema": (
            REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_APPEND_REPORT_SCHEMA
        ),
        "queue_actuation_blockers": ordered_unique(blockers),
        "operator_visible_automation_rollup": automation_rollup,
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_campaign_stackability_queue_for_local_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_campaign_stackability_queue_metadata",
    )
    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {"local_cpu": 1},
            },
            "metadata": metadata,
            "experiments": experiments,
        }
    )


__all__ = [
    "REPAIR_CAMPAIGN_STACKABILITY_EXPERIMENT_METADATA_SCHEMA",
    "REPAIR_CAMPAIGN_STACKABILITY_QUEUE_METADATA_SCHEMA",
    "RepairCampaignStackabilityQueueError",
    "build_repair_campaign_stackability_queue",
]
