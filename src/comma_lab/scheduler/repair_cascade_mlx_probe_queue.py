# SPDX-License-Identifier: MIT
"""Build queue-owned MLX-local probe specs for structural repair cascades."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import QUEUE_SCHEMA, normalize_queue_definition
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_scorer import REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA

REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA = (
    "repair_cascade_mlx_probe_queue_metadata.v1"
)
REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA = (
    "repair_cascade_mlx_probe_experiment_metadata.v1"
)
REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA = "repair_cascade_mlx_probe_spec.v1"
REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA = "repair_cascade_mlx_probe_result.v1"
REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA = (
    "frontier_rate_attack_repair_cascade_opportunity_row.v1"
)
REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA = (
    "frontier_rate_attack_repair_budget_waterfill_work_order.v1"
)


class RepairCascadeMlxProbeQueueError(ValueError):
    """Raised when a structural repair-cascade probe queue cannot be built."""


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path)
    repo = Path(repo_root)
    try:
        return str(resolved.resolve(strict=False).relative_to(repo.resolve(strict=False)))
    except ValueError:
        return str(resolved)


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_payload_schema(payload: Mapping[str, Any]) -> str:
    return str(payload.get("schema") or "")


def _cascade_rows_from_work_order(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("repair_cascade_opportunity_rows") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("schema") != REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA:
            continue
        rows.append(dict(row))
    return rows


def _cascade_rows_from_score_report(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        family_id = str(row.get("family_id") or "")
        if (
            family_id != "entropy_position_cascade"
            and row.get("source_row_kind") != "repair_cascade_opportunity"
        ):
            continue
        source = _mapping(row.get("source_structural_opportunity"))
        cascade = dict(source) if source else {}
        cascade.setdefault("schema", REPAIR_CASCADE_OPPORTUNITY_ROW_SCHEMA)
        cascade.setdefault("cascade_id", row.get("cascade_id") or row.get("candidate_id"))
        cascade.setdefault("label", row.get("cascade_label") or "structural repair cascade")
        cascade.setdefault("source_relation", row.get("source_relation"))
        cascade.setdefault(
            "pipeline_position",
            row.get("entropy_position_label")
            or "scorer_entropy_repair_before_selector_codec",
        )
        cascade.setdefault("blockers", _string_list(row.get("source_blockers")))
        cascade.setdefault(
            "required_probe_measurements",
            _string_list(row.get("source_missing_artifacts")),
        )
        rows.append(cascade)
    return rows


def repair_cascade_rows_from_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return structural repair-cascade rows from supported source payloads."""

    schema = _source_payload_schema(payload)
    if schema == REPAIR_BUDGET_WATERFILL_WORK_ORDER_SCHEMA:
        return _cascade_rows_from_work_order(payload)
    if schema == REPAIR_CAMPAIGN_SCORE_REPORT_SCHEMA:
        return _cascade_rows_from_score_report(payload)
    raise RepairCascadeMlxProbeQueueError(
        "repair cascade MLX probe queue requires a repair waterfill work order "
        "or repair campaign score report"
    )


def _targeted_positions(cascade: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in cascade.get("targeted_positions") or []
        if isinstance(row, Mapping)
    ]


def _required_probe_measurements(cascade: Mapping[str, Any]) -> list[str]:
    measurements = _string_list(cascade.get("required_probe_measurements"))
    if measurements:
        return ordered_unique(measurements)
    return [
        "posenet_null_bottom_decile_pair_ids",
        "segnet_class_region_mask_ids",
        "selector_payload_bits_per_region",
        "receiver_consumed_runtime_replay_proof",
    ]


def _artifact_status(
    cascade: Mapping[str, Any],
    key: str,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    text = str(cascade.get(key) or "").strip()
    path = _resolve(text, repo_root) if text else None
    exists = bool(path is not None and path.is_file())
    return {
        "key": key,
        "path": text or None,
        "exists": exists,
    }


def build_repair_cascade_mlx_probe_spec(
    *,
    source_payload: Mapping[str, Any],
    source_payload_path: str | Path,
    cascade_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build a false-authority MLX-local probe spec for one cascade row."""

    require_no_truthy_authority_fields(
        source_payload,
        context="repair_cascade_mlx_probe_source_payload",
    )
    rows = repair_cascade_rows_from_payload(source_payload)
    cascade = next(
        (row for row in rows if str(row.get("cascade_id") or "") == cascade_id),
        None,
    )
    if cascade is None:
        raise RepairCascadeMlxProbeQueueError(f"cascade id not found: {cascade_id}")
    measurements = _required_probe_measurements(cascade)
    artifact_keys = ordered_unique(
        [
            "local_mlx_response_path",
            "reference_local_mlx_response_path",
            "posenet_null_bottom_decile_pair_ids_path",
            "segnet_class_region_mask_ids_path",
            "selector_payload_bits_per_region_path",
            "runtime_consumption_proof_path",
        ]
    )
    artifact_status = [
        _artifact_status(cascade, key, repo_root=repo_root) for key in artifact_keys
    ]
    missing = ordered_unique(
        [
            f"{item['key']}:missing_or_unverified"
            for item in artifact_status
            if item.get("exists") is not True
        ]
    )
    blockers = ordered_unique(
        [
            *missing,
            *_string_list(cascade.get("blockers")),
            "exact_axis_component_response_required_before_budget_spend",
            "receiver_runtime_materialization_required_before_exact_dispatch",
        ]
    )
    spec = {
        "schema": REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
        "source_payload_path": str(source_payload_path),
        "source_payload_schema": source_payload.get("schema"),
        "cascade_id": cascade_id,
        "cascade_label": cascade.get("label"),
        "source_relation": cascade.get("source_relation"),
        "pipeline_position": cascade.get("pipeline_position"),
        "targeted_positions": [dict(row) for row in _targeted_positions(cascade)],
        "required_probe_measurements": measurements,
        "required_local_mlx_artifacts": artifact_keys,
        "local_mlx_artifact_status": artifact_status,
        "missing_local_mlx_artifacts": missing,
        "probe_measurement_plan": [
            {
                "measurement": measurement,
                "axis": "[macOS-MLX research-signal]",
                "authority": "local_repair_planning_only",
                **FALSE_AUTHORITY,
            }
            for measurement in measurements
        ],
        "local_mlx_probe_execution_ready": not missing,
        "recommended_next_action": (
            "run_cascade_mlx_local_component_response_probe"
            if not missing
            else "materialize_missing_cascade_mlx_probe_artifacts"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "cascade_mlx_local_probe_spec_for_repair_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        spec,
        context=f"repair_cascade_mlx_probe_spec:{cascade_id}",
    )
    return spec


def _artifact_status_from_record(
    record: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    key = str(record.get("key") or "unknown").strip() or "unknown"
    path_text = str(record.get("path") or "").strip()
    path = _resolve(path_text, repo_root) if path_text else None
    return {
        "key": key,
        "path": path_text or None,
        "exists": bool(path is not None and path.is_file()),
    }


def build_repair_cascade_mlx_probe_result(
    *,
    probe_spec: Mapping[str, Any],
    probe_spec_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Record deterministic readiness for one MLX-local cascade probe spec."""

    if probe_spec.get("schema") != REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA:
        raise RepairCascadeMlxProbeQueueError(
            "repair cascade MLX probe result requires a probe spec"
        )
    require_no_truthy_authority_fields(
        probe_spec,
        context="repair_cascade_mlx_probe_result_spec",
    )
    artifact_status = [
        _artifact_status_from_record(record, repo_root=repo_root)
        for record in probe_spec.get("local_mlx_artifact_status") or []
        if isinstance(record, Mapping)
    ]
    missing = ordered_unique(
        [
            f"{item['key']}:missing_or_unverified"
            for item in artifact_status
            if item.get("exists") is not True
        ]
    )
    spec_blockers = _string_list(probe_spec.get("blockers"))
    blockers = ordered_unique(
        [
            *missing,
            *spec_blockers,
            *(
                ["local_mlx_probe_artifacts_missing"]
                if missing
                else ["concrete_mlx_component_response_runner_required"]
            ),
            "exact_axis_component_response_required_before_budget_spend",
            "receiver_runtime_materialization_required_before_exact_dispatch",
        ]
    )
    result = {
        "schema": REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
        "probe_spec_path": str(probe_spec_path),
        "probe_spec_schema": probe_spec.get("schema"),
        "cascade_id": probe_spec.get("cascade_id"),
        "cascade_label": probe_spec.get("cascade_label"),
        "source_payload_path": probe_spec.get("source_payload_path"),
        "source_payload_schema": probe_spec.get("source_payload_schema"),
        "component_response_axis": "[macOS-MLX research-signal]",
        "local_mlx_artifact_status": artifact_status,
        "missing_local_mlx_artifacts": missing,
        "required_probe_measurements": _string_list(
            probe_spec.get("required_probe_measurements")
        ),
        "probe_measurement_plan": list(probe_spec.get("probe_measurement_plan") or []),
        "local_mlx_probe_execution_ready": not missing,
        "local_mlx_probe_executed": False,
        "component_response_row_emitted": False,
        "learning_signal_kind": (
            "blocked_repair_cascade_mlx_probe"
            if missing
            else "repair_cascade_mlx_probe_ready_for_component_response_runner"
        ),
        "recommended_next_action": (
            "materialize_missing_cascade_mlx_probe_artifacts"
            if missing
            else "run_concrete_mlx_component_response_probe_and_harvest_response_row"
        ),
        "blockers": blockers,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "cascade_mlx_local_probe_readiness_result_for_learning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        result,
        context=f"repair_cascade_mlx_probe_result:{probe_spec.get('cascade_id')}",
    )
    return result


def _cascade_experiment(
    *,
    source_payload: Mapping[str, Any],
    source_payload_path: str | Path,
    cascade: Mapping[str, Any],
    repo_root: str | Path,
    queue_root: Path,
    priority: int,
) -> dict[str, Any]:
    cascade_id = str(cascade.get("cascade_id") or f"cascade_{priority}").strip()
    slug = _slug(cascade_id)
    spec_path = queue_root / slug / "repair_cascade_mlx_probe_spec.json"
    result_path = queue_root / slug / "repair_cascade_mlx_probe_result.json"
    spec_ref = _repo_rel(spec_path, repo_root)
    result_ref = _repo_rel(result_path, repo_root)
    missing = _required_probe_measurements(cascade)
    metadata = {
        "schema": REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA,
        "source_payload_path": str(source_payload_path),
        "source_payload_schema": source_payload.get("schema"),
        "cascade_id": cascade_id,
        "cascade_label": cascade.get("label"),
        "source_relation": cascade.get("source_relation"),
        "pipeline_position": cascade.get("pipeline_position"),
        "probe_spec_path": spec_ref,
        "probe_spec_schema": REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
        "probe_result_path": result_ref,
        "probe_result_schema": REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
        "required_probe_measurements": missing,
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "queue_actuation_ready": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "queue_owned_cascade_mlx_probe_metadata",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context=f"repair_cascade_mlx_probe_experiment:{cascade_id}",
    )
    return {
        "id": f"repair_cascade_mlx_probe_{slug}",
        "priority": priority,
        "status": "queued",
        "tags": [
            "frontier-rate-attack",
            "repair-cascade",
            "local-mlx-advisory-custody",
            "no-score-authority",
        ],
        "metadata": metadata,
        "steps": [
            {
                "id": "emit_repair_cascade_mlx_probe_spec",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "tools/build_repair_cascade_mlx_probe_spec.py",
                    "--source-payload",
                    str(source_payload_path),
                    "--cascade-id",
                    cascade_id,
                    "--probe-spec-out",
                    spec_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": spec_ref,
                        "key": "schema",
                        "equals": REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": spec_ref},
                    {
                        "type": "json_equals",
                        "path": spec_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [spec_ref],
                    "input_artifact_paths": [str(source_payload_path)],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "inspect_repair_cascade_mlx_probe_missing_artifacts",
                "kind": "command",
                "requires": ["emit_repair_cascade_mlx_probe_spec"],
                "command": [".venv/bin/python", "-m", "json.tool", spec_ref],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 60,
                "telemetry": {
                    "input_artifact_paths": [spec_ref],
                    "include_postcondition_paths": True,
                },
            },
            {
                "id": "record_repair_cascade_mlx_probe_result",
                "kind": "command",
                "requires": ["emit_repair_cascade_mlx_probe_spec"],
                "command": [
                    ".venv/bin/python",
                    "tools/run_repair_cascade_mlx_probe.py",
                    "--probe-spec",
                    spec_ref,
                    "--output",
                    result_ref,
                    "--overwrite",
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 120,
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": result_ref,
                        "key": "schema",
                        "equals": REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA,
                    },
                    {"type": "json_false_authority", "path": result_ref},
                    {
                        "type": "json_equals",
                        "path": result_ref,
                        "key": "ready_for_exact_eval_dispatch",
                        "equals": False,
                    },
                ],
                "telemetry": {
                    "artifact_paths": [result_ref],
                    "input_artifact_paths": [spec_ref],
                    "include_postcondition_paths": True,
                },
            },
        ],
    }


def _empty_experiment(
    *,
    source_payload_path: str | Path,
) -> dict[str, Any]:
    return {
        "id": "repair_cascade_mlx_probe_no_structural_cascade_rows",
        "priority": 1,
        "status": "frozen",
        "tags": [
            "frontier-rate-attack",
            "repair-cascade",
            "blocked-empty-selection",
            "no-score-authority",
        ],
        "metadata": {
            "schema": REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA,
            "source_payload_path": str(source_payload_path),
            "cascade_id": None,
            "queue_actuation_ready": False,
            "queue_actuation_blockers": ["structural_repair_cascade_rows_empty"],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        },
        "steps": [
            {
                "id": "inspect_empty_repair_cascade_selection",
                "kind": "command",
                "command": [
                    ".venv/bin/python",
                    "-c",
                    (
                        "import json; print(json.dumps({"
                        "'schema':'repair_cascade_mlx_probe_blocker.v1',"
                        "'blockers':['structural_repair_cascade_rows_empty'],"
                        "'budget_spend_allowed':False,"
                        "'ready_for_exact_eval_dispatch':False,"
                        "'score_claim':False,"
                        "'promotion_eligible':False,"
                        "'rank_or_kill_eligible':False"
                        "}, sort_keys=True))"
                    ),
                ],
                "resources": {"kind": "local_cpu"},
                "timeout_seconds": 30,
            }
        ],
    }


def build_repair_cascade_mlx_probe_queue(
    *,
    repo_root: str | Path,
    source_payload: Mapping[str, Any],
    source_payload_path: str | Path,
    results_root: str | Path,
    queue_id: str = "repair_cascade_mlx_probe_queue",
    experiment_limit: int | None = None,
) -> dict[str, Any]:
    """Build local probe-spec rows for structural repair cascades."""

    require_no_truthy_authority_fields(
        source_payload,
        context="repair_cascade_mlx_probe_queue_input",
    )
    cascade_rows = repair_cascade_rows_from_payload(source_payload)
    if experiment_limit is not None:
        cascade_rows = cascade_rows[: max(0, int(experiment_limit))]
    queue_root = (
        _resolve(results_root, repo_root) / "repair_cascade_mlx_probe_queue" / _slug(queue_id)
    )
    experiments = (
        [
            _cascade_experiment(
                source_payload=source_payload,
                source_payload_path=source_payload_path,
                cascade=cascade,
                repo_root=repo_root,
                queue_root=queue_root,
                priority=priority,
            )
            for priority, cascade in enumerate(cascade_rows, start=1)
        ]
        if cascade_rows
        else [_empty_experiment(source_payload_path=source_payload_path)]
    )
    ready_count = sum(
        1
        for experiment in experiments
        if _mapping(experiment.get("metadata")).get("queue_actuation_ready") is True
    )
    blockers = ordered_unique(
        blocker
        for experiment in experiments
        for blocker in _string_list(
            _mapping(experiment.get("metadata")).get("queue_actuation_blockers")
        )
    )
    if not cascade_rows:
        blockers.append("structural_repair_cascade_rows_empty")
    metadata = {
        "schema": REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA,
        "source_payload_path": str(source_payload_path),
        "source_payload_schema": source_payload.get("schema"),
        "queue_id": queue_id,
        "structural_repair_cascade_count": len(cascade_rows),
        "experiment_count": len(experiments),
        "ready_experiment_count": ready_count,
        "blocked_experiment_count": len(experiments) - ready_count,
        "results_root": _repo_rel(queue_root, repo_root),
        "queue_actuation_blockers": blockers,
        "local_mlx_advisory_custody_required": True,
        "component_response_axis": "[macOS-MLX research-signal]",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "repair_cascade_mlx_probe_queue_for_local_planning",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        metadata,
        context="repair_cascade_mlx_probe_queue_metadata",
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
    "REPAIR_CASCADE_MLX_PROBE_EXPERIMENT_METADATA_SCHEMA",
    "REPAIR_CASCADE_MLX_PROBE_QUEUE_METADATA_SCHEMA",
    "REPAIR_CASCADE_MLX_PROBE_RESULT_SCHEMA",
    "REPAIR_CASCADE_MLX_PROBE_SPEC_SCHEMA",
    "RepairCascadeMlxProbeQueueError",
    "build_repair_cascade_mlx_probe_queue",
    "build_repair_cascade_mlx_probe_result",
    "build_repair_cascade_mlx_probe_spec",
    "repair_cascade_rows_from_payload",
]
