# SPDX-License-Identifier: MIT
"""Planning-only staircase DAG for local and distributed experiment queues.

The purpose of this module is to decide what can be run next, on which class of
machine, without turning a generic scheduler into score authority. Executors
such as Dask, SSH, Modal, or a local worker can consume the returned dispatch
plan, but score/promotion semantics stay in the repo-owned queue, ledger, and
auth-eval surfaces.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import (
    ExperimentQueueError,
    _is_cloud_resource,
    _step_hashes,
    connect_state_readonly,
    default_state_path,
    load_queue_definition,
    normalize_resource_kind,
)
from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    DEFAULT_WORKLOAD_SUBDIR,
    parse_storage_tier_specs,
    plan_experiment_storage,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact

STAIRCASE_DAG_SCHEMA = "staircase_dag.v1"
STAIRCASE_DISPATCH_PLAN_SCHEMA = "staircase_dispatch_plan.v1"
STORAGE_PREFLIGHT_DEPENDENCY_SCHEMA = "staircase_storage_preflight_dependency.v1"

DEFAULT_MACHINE_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "m5_max_128gb",
        "label": "M5 Max local workstation",
        "slots": {"local_cpu": 8, "local_mlx": 1},
        "memory_gb": 128.0,
        "disk_gb": 80.0,
        "tags": ["darwin", "arm64", "unified_memory", "primary"],
    },
    {
        "id": "m1_macbook_pro_8gb",
        "label": "M1 MacBook Pro",
        "slots": {"local_cpu": 2, "local_mlx": 1},
        "memory_gb": 8.0,
        "disk_gb": 12.0,
        "tags": ["darwin", "arm64", "edge"],
    },
    {
        "id": "raspberry_pi4_8gb",
        "label": "Raspberry Pi 4",
        "slots": {"local_cpu": 1},
        "memory_gb": 8.0,
        "disk_gb": 8.0,
        "tags": ["linux", "aarch64", "edge", "slow"],
    },
    {
        "id": "intel_mac_mini_8gb",
        "label": "Intel Mac mini",
        "slots": {"local_cpu": 2},
        "memory_gb": 8.0,
        "disk_gb": 16.0,
        "tags": ["darwin", "x86_64", "edge"],
    },
    {
        "id": "ryzen_3900x_rtx2070s",
        "label": "Ryzen 3900X + RTX 2070 Super",
        "slots": {"local_cpu": 8, "local_cuda": 1},
        "memory_gb": 32.0,
        "disk_gb": 64.0,
        "tags": ["linux", "x86_64", "cuda", "lan"],
    },
)


@dataclass(frozen=True)
class StaircaseReadyNode:
    node_id: str
    priority: int
    utility: float
    resource_kind: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    family: str
    stage: str
    machine_id: str | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "priority": self.priority,
            "utility": self.utility,
            "resource_kind": self.resource_kind,
            "command": list(self.command),
            "dependencies": list(self.dependencies),
            "family": self.family,
            "stage": self.stage,
            "machine_id": self.machine_id,
            "metadata": dict(self.metadata),
        }


def _json_text(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_json_text(payload).encode("utf-8")).hexdigest()


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentQueueError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ExperimentQueueError(f"{label} must be a list")
    return [_require_text(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _mapping(value: object, label: str, *, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if value is None:
        return dict(default or {})
    if not isinstance(value, Mapping):
        raise ExperimentQueueError(f"{label} must be an object")
    return dict(value)


def _optional_mapping(value: object, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, label)


def _non_negative_int(value: object, label: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ExperimentQueueError(f"{label} must be a non-negative integer")
    return value


def _positive_float(value: object, label: str, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
        raise ExperimentQueueError(f"{label} must be a positive number")
    return float(value)


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExperimentQueueError(f"{label} must be a number")
    return float(value)


def _queue_control_mode(value: object) -> str:
    mode = _require_text(value, "controls.mode")
    if mode not in {"running", "paused", "frozen"}:
        raise ExperimentQueueError("controls.mode must be one of ['frozen', 'paused', 'running']")
    return mode


def _normalize_storage_plan_payload(value: object) -> dict[str, Any] | None:
    storage = _optional_mapping(value, "storage")
    if storage is None:
        return None
    try:
        require_no_truthy_authority_fields(storage, context="staircase_dag.storage")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    schema = _require_text(storage.get("schema"), "storage.schema")
    selected = storage.get("selected_workload_root")
    if selected is not None and not isinstance(selected, str):
        raise ExperimentQueueError("storage.selected_workload_root must be a string or null")
    tiers = storage.get("tiers")
    if tiers is not None and not isinstance(tiers, list):
        raise ExperimentQueueError("storage.tiers must be a list when present")
    payload = apply_proxy_evidence_boundary(
        {
            **storage,
            "schema": schema,
            "storage_required": bool(storage.get("storage_required", True)),
            "executor_contract": {
                **_mapping(storage.get("executor_contract"), "storage.executor_contract"),
                "must_write_bulk_outputs_under_selected_workload_root": True,
                "local_disk_fallback_requires_explicit_allow_local_disk": True,
            },
        },
        dispatch_blockers=["storage_plan_is_planning_only"],
    )
    return payload


def build_storage_plan_payload(
    *,
    repo_root: str | Path,
    storage_tiers: Sequence[str] | None = None,
    workload_subdir: str = DEFAULT_WORKLOAD_SUBDIR,
    requested_bytes: int = 0,
    min_free_bytes: int = 0,
    reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    allow_local_disk: bool = False,
    create: bool = False,
) -> dict[str, Any]:
    """Build a false-authority storage waterfall payload for a staircase DAG."""

    specs = parse_storage_tier_specs(
        list(storage_tiers or []),
        repo_root=Path(repo_root),
        reserve_free_gb=reserve_free_gb,
        allow_local_disk=allow_local_disk,
    )
    plan = plan_experiment_storage(
        specs,
        workload_subdir=workload_subdir,
        requested_bytes=requested_bytes,
        min_free_bytes=min_free_bytes,
        create=create,
    )
    return _normalize_storage_plan_payload(
        {
            **plan.to_dict(),
            "storage_required": True,
            "executor_contract": {
                "must_write_bulk_outputs_under_selected_workload_root": True,
                "local_disk_fallback_requires_explicit_allow_local_disk": True,
            },
        }
    )


def _normalize_node(raw: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    node_id = _require_text(raw.get("id") or raw.get("node_id"), f"nodes[{index}].id")
    command = raw.get("command")
    if not isinstance(command, list) or not command:
        raise ExperimentQueueError(f"{node_id}.command must be a non-empty argv list")
    try:
        require_no_truthy_authority_fields(raw, context=f"staircase_dag.node[{node_id}]")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    row = apply_proxy_evidence_boundary(
        {
            "id": node_id,
            "node_id": node_id,
            "command": [_require_text(item, f"{node_id}.command[{idx}]") for idx, item in enumerate(command)],
            "dependencies": _string_list(
                raw.get("dependencies", raw.get("requires")),
                f"{node_id}.dependencies",
            ),
            "priority": _non_negative_int(raw.get("priority"), f"{node_id}.priority", default=100),
            "resource_kind": normalize_resource_kind(
                raw.get("resource_kind", "local_cpu"),
                f"{node_id}.resource_kind",
            ),
            "family": _require_text(raw.get("family", "default"), f"{node_id}.family"),
            "stage": _require_text(raw.get("stage", "candidate"), f"{node_id}.stage"),
            "expected_value": _optional_float(raw.get("expected_value"), f"{node_id}.expected_value"),
            "estimated_cost": _optional_float(raw.get("estimated_cost"), f"{node_id}.estimated_cost"),
            "tags": _string_list(raw.get("tags"), f"{node_id}.tags"),
            "metadata": _mapping(raw.get("metadata"), f"{node_id}.metadata"),
        },
        dispatch_blockers=["staircase_dag_is_planning_only"],
    )
    return row


def _queue_dependency_node_id(ref: str, *, default_experiment_id: str) -> str:
    text = _require_text(ref, "step dependency")
    if "." not in text:
        return f"{default_experiment_id}.{text}"
    experiment_id, step_id = text.split(".", 1)
    return (
        f"{_require_text(experiment_id, 'step dependency experiment_id')}."
        f"{_require_text(step_id, 'step dependency step_id')}"
    )


def _command_flag_value(command: object, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    for index, item in enumerate(command):
        if str(item) == flag and index + 1 < len(command):
            value = str(command[index + 1]).strip()
            return value or None
    return None


def _step_postcondition_paths(step: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    postconditions = step.get("postconditions")
    if not isinstance(postconditions, list):
        return paths
    for condition in postconditions:
        if not isinstance(condition, Mapping):
            continue
        path = condition.get("path")
        if isinstance(path, str) and path.strip():
            paths.append(path.strip())
    return paths


def _storage_preflight_artifact_path(
    step: Mapping[str, Any],
    *,
    command_flag: str,
) -> str | None:
    for path in _step_postcondition_paths(step):
        return path
    return _command_flag_value(step.get("command"), command_flag)


def _is_storage_preflight_experiment(experiment: Mapping[str, Any]) -> bool:
    tags = {str(tag) for tag in experiment.get("tags", []) if isinstance(tag, str)}
    if {"scheduler-preflight", "storage", "cleanup"}.issubset(tags):
        return True
    steps = [step for step in experiment.get("steps", []) if isinstance(step, Mapping)]
    has_storage_step = any(
        step.get("id") == "storage_tier_plan"
        and "tools/plan_experiment_storage.py"
        in [str(item) for item in step.get("command", [])]
        for step in steps
    )
    has_cleanup_step = any(
        step.get("id") == "proactive_cleanup"
        and "tools/compact_experiment_artifacts.py"
        in [str(item) for item in step.get("command", [])]
        for step in steps
    )
    return has_storage_step and has_cleanup_step


def _storage_preflight_dependencies_from_queue(queue: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    preflights: dict[str, dict[str, Any]] = {}
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping) or not _is_storage_preflight_experiment(experiment):
            continue
        experiment_id = _require_text(experiment.get("id"), "storage preflight experiment.id")
        steps = [step for step in experiment.get("steps", []) if isinstance(step, Mapping)]
        by_step_id = {str(step.get("id")): step for step in steps if isinstance(step.get("id"), str)}
        storage_step = by_step_id.get("storage_tier_plan")
        cleanup_step = by_step_id.get("proactive_cleanup")
        if storage_step is None or cleanup_step is None:
            raise ExperimentQueueError(
                f"{experiment_id}: storage preflight must include storage_tier_plan and proactive_cleanup steps"
            )
        storage_plan_path = _storage_preflight_artifact_path(
            storage_step,
            command_flag="--output",
        )
        cleanup_plan_path = _storage_preflight_artifact_path(
            cleanup_step,
            command_flag="--json-output",
        )
        if storage_plan_path is None or cleanup_plan_path is None:
            raise ExperimentQueueError(
                f"{experiment_id}: storage preflight steps must expose storage and cleanup artifact paths"
            )
        storage_node_id = f"{experiment_id}.storage_tier_plan"
        cleanup_node_id = f"{experiment_id}.proactive_cleanup"
        payload = apply_proxy_evidence_boundary(
            {
                "schema": STORAGE_PREFLIGHT_DEPENDENCY_SCHEMA,
                "dependency_node_id": cleanup_node_id,
                "cleanup_node_id": cleanup_node_id,
                "storage_plan_node_id": storage_node_id,
                "storage_plan_artifact_path": storage_plan_path,
                "cleanup_plan_artifact_path": cleanup_plan_path,
                "artifact_paths": [storage_plan_path, cleanup_plan_path],
                "advisory_scope": "storage_preflight_dependency_only",
                "executor_contract": {
                    "dependency_must_succeed_before_execution": True,
                    "executor_must_not_treat_preflight_as_score_authority": True,
                    "executor_must_read_artifacts_as_advisory_storage_plans": True,
                },
            },
            dispatch_blockers=[
                "storage_preflight_dependency_is_advisory_only",
                "storage_preflight_does_not_grant_score_authority",
            ],
        )
        preflights[cleanup_node_id] = payload
    return preflights


def normalize_resource_pools(raw_pools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    pools = list(raw_pools or default_local_resource_pools())
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_pool in enumerate(pools):
        if not isinstance(raw_pool, Mapping):
            raise ExperimentQueueError(f"resource_pools[{index}] must be an object")
        pool_id = _require_text(raw_pool.get("id"), f"resource_pools[{index}].id")
        if pool_id in seen:
            raise ExperimentQueueError(f"duplicate resource pool id: {pool_id}")
        seen.add(pool_id)
        slots = _mapping(raw_pool.get("slots"), f"{pool_id}.slots")
        normalized_slots: dict[str, int] = {}
        for kind, count in slots.items():
            normalized_slots[normalize_resource_kind(kind, f"{pool_id}.slots key")] = _non_negative_int(
                count,
                f"{pool_id}.slots.{kind}",
            )
        if not any(normalized_slots.values()):
            raise ExperimentQueueError(f"{pool_id}.slots must contain at least one positive slot")
        row = {
            "id": pool_id,
            "label": str(raw_pool.get("label") or pool_id),
            "slots": normalized_slots,
            "memory_gb": _positive_float(raw_pool.get("memory_gb"), f"{pool_id}.memory_gb", default=1.0),
            "disk_gb": _positive_float(raw_pool.get("disk_gb"), f"{pool_id}.disk_gb", default=1.0),
            "tags": _string_list(raw_pool.get("tags"), f"{pool_id}.tags"),
        }
        storage = _normalize_storage_plan_payload(raw_pool.get("storage"))
        if storage is not None:
            row["storage"] = storage
        normalized.append(row)
    return normalized


def _normalize_max_concurrency(value: object, label: str) -> dict[str, int]:
    if value is None:
        return {}
    raw = _mapping(value, label)
    out: dict[str, int] = {}
    for kind, count in raw.items():
        out[normalize_resource_kind(kind, f"{label} key")] = _non_negative_int(
            count,
            f"{label}.{kind}",
        )
    return out


def normalize_staircase_dag(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = _require_text(payload.get("schema", STAIRCASE_DAG_SCHEMA), "schema")
    if schema != STAIRCASE_DAG_SCHEMA:
        raise ExperimentQueueError(f"unsupported staircase DAG schema: {schema!r}")
    try:
        require_no_truthy_authority_fields(payload, context="staircase_dag")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    dag_id = _require_text(payload.get("dag_id"), "dag_id")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ExperimentQueueError("nodes must be a list")
    nodes = [_normalize_node(node, index=index) for index, node in enumerate(raw_nodes) if isinstance(node, Mapping)]
    if len(nodes) != len(raw_nodes):
        raise ExperimentQueueError("nodes must contain only objects")
    node_ids = [str(node["node_id"]) for node in nodes]
    duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    if duplicates:
        raise ExperimentQueueError(f"duplicate node id(s): {duplicates}")
    node_set = set(node_ids)
    for node in nodes:
        unknown = sorted(set(node["dependencies"]) - node_set)
        if unknown:
            raise ExperimentQueueError(f"{node['node_id']} depends on unknown node(s): {unknown}")
    controls = _mapping(payload.get("controls"), "controls")
    storage = _normalize_storage_plan_payload(payload.get("storage"))
    normalized = apply_proxy_evidence_boundary(
        {
            "schema": STAIRCASE_DAG_SCHEMA,
            "dag_id": dag_id,
            "controls": {
                "allow_cloud": bool(controls.get("allow_cloud", False)),
                "mode": _queue_control_mode(controls.get("mode", "running")),
                "max_ready_nodes": _non_negative_int(
                    controls.get("max_ready_nodes"),
                    "controls.max_ready_nodes",
                    default=32,
                ),
                "diversity_bucket_limit": _non_negative_int(
                    controls.get("diversity_bucket_limit"),
                    "controls.diversity_bucket_limit",
                    default=2,
                ),
                "max_concurrency": _normalize_max_concurrency(
                    controls.get("max_concurrency"),
                    "controls.max_concurrency",
                ),
            },
            "resource_pools": normalize_resource_pools(
                payload.get("resource_pools") if isinstance(payload.get("resource_pools"), list) else None
            ),
            "storage": storage,
            "nodes": nodes,
            "source_refs": list(payload.get("source_refs") or []),
        },
        dispatch_blockers=["staircase_dag_is_planning_only"],
    )
    normalized["dag_hash"] = _sha256_json(
        {
            "dag_id": normalized["dag_id"],
            "controls": normalized["controls"],
            "resource_pools": normalized["resource_pools"],
            "storage": normalized["storage"],
            "nodes": normalized["nodes"],
            "source_refs": normalized["source_refs"],
        }
    )
    return normalized


def default_local_resource_pools() -> list[dict[str, Any]]:
    disk = shutil.disk_usage(Path.cwd())
    cpu_slots = max(1, min(os.cpu_count() or 1, 8))
    slots: dict[str, int] = {"local_cpu": cpu_slots}
    tags = [platform.system().lower() or "unknown", platform.machine().lower() or "unknown"]
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        slots["local_mlx"] = 1
        tags.append("mlx")
    return [
        {
            "id": "local_default",
            "label": "Local host",
            "slots": slots,
            "memory_gb": 1.0,
            "disk_gb": max(1.0, float(disk.free) / (1024.0**3)),
            "tags": tags,
        }
    ]


def local_lab_resource_pools(*, live_primary_disk_gb: float | None = None) -> list[dict[str, Any]]:
    pools = [dict(pool) for pool in DEFAULT_MACHINE_PRESETS]
    if live_primary_disk_gb is not None:
        pools[0] = {**pools[0], "disk_gb": float(live_primary_disk_gb)}
    return pools


def build_staircase_dag_from_experiment_queue(
    queue: Mapping[str, Any],
    *,
    dag_id: str | None = None,
    source_path: str | Path | None = None,
    resource_pools: Sequence[Mapping[str, Any]] | None = None,
    storage_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    storage_preflights = _storage_preflight_dependencies_from_queue(queue)
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping):
            continue
        experiment_id = _require_text(experiment.get("id"), "experiment.id")
        experiment_status = _require_text(
            experiment.get("status", "queued"),
            f"{experiment_id}.status",
        )
        if experiment_status not in {"queued", "paused", "frozen", "disabled"}:
            raise ExperimentQueueError(f"{experiment_id}.status unsupported: {experiment_status!r}")
        if experiment_status != "queued":
            continue
        priority = _non_negative_int(experiment.get("priority"), f"{experiment_id}.priority", default=100)
        experiment_tags = _string_list(experiment.get("tags"), f"{experiment_id}.tags")
        family = str(experiment.get("lane_id") or (experiment_tags[0] if experiment_tags else experiment_id))
        experiment_metadata = dict(experiment.get("metadata") or {})
        for step in experiment.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            step_id = _require_text(step.get("id"), f"{experiment_id}.step.id")
            resources = _mapping(step.get("resources"), f"{experiment_id}.{step_id}.resources")
            step_telemetry = dict(step.get("telemetry") or {})
            step_hashes = _step_hashes(
                step,
                experiment_metadata=experiment_metadata,
            )
            metadata = {
                "queue_id": queue.get("queue_id"),
                "experiment_id": experiment_id,
                "step_id": step_id,
                "step_hashes": step_hashes,
                "postconditions": list(step.get("postconditions") or []),
                "timeout_seconds": step.get("timeout_seconds"),
                "resource_requirements": resources,
                "step_telemetry": step_telemetry,
                "experiment_metadata": experiment_metadata,
                "unit_kind": experiment_metadata.get("unit_kind"),
                "operation_family": experiment_metadata.get("operation_family"),
                "target_kind": experiment_metadata.get("target_kind"),
                "materializer_id": experiment_metadata.get("materializer_id"),
                "receiver_contract_id": experiment_metadata.get("receiver_contract_id"),
                "receiver_contract_kind": experiment_metadata.get("receiver_contract_kind"),
                "allowed_use": experiment_metadata.get("allowed_use"),
            }
            dependencies = [
                _queue_dependency_node_id(str(required), default_experiment_id=experiment_id)
                for required in _string_list(step.get("requires"), f"{experiment_id}.{step_id}.requires")
            ]
            storage_dependencies = [
                storage_preflights[dependency]
                for dependency in dependencies
                if dependency in storage_preflights
            ]
            if storage_dependencies:
                metadata["storage_preflight_dependencies"] = storage_dependencies
                if len(storage_dependencies) == 1:
                    metadata["storage_preflight_dependency"] = storage_dependencies[0]
            nodes.append(
                {
                    "id": f"{experiment_id}.{step_id}",
                    "command": list(step.get("command") or []),
                    "dependencies": dependencies,
                    "priority": priority,
                    "resource_kind": normalize_resource_kind(
                        resources.get("kind", "local_cpu"),
                        f"{experiment_id}.{step_id}.resources.kind",
                    ),
                    "family": family,
                    "stage": step_id,
                    "tags": experiment_tags,
                    "metadata": metadata,
                }
            )
    refs: list[dict[str, Any]] = [
        {
            "kind": "experiment_queue",
            "queue_id": queue.get("queue_id"),
            "queue_hash": _sha256_json(queue),
        }
    ]
    if source_path is not None:
        refs[0]["path"] = str(source_path)
    return normalize_staircase_dag(
        {
            "schema": STAIRCASE_DAG_SCHEMA,
            "dag_id": dag_id or f"{queue.get('queue_id')}_staircase",
            "controls": {
                "mode": str(_mapping(queue.get("controls"), "queue.controls").get("mode", "running")),
                "max_concurrency": dict(
                    _mapping(queue.get("controls"), "queue.controls").get(
                        "max_concurrency",
                        {},
                    )
                ),
            },
            "resource_pools": list(resource_pools or default_local_resource_pools()),
            "storage": dict(storage_plan) if storage_plan is not None else None,
            "nodes": nodes,
            "source_refs": refs,
        }
    )


def experiment_queue_status_map(
    *,
    queue_path: str | Path,
    repo_root: str | Path,
    state_path: str | Path | None = None,
) -> dict[str, str]:
    queue = load_queue_definition(queue_path)
    state = Path(state_path) if state_path else default_state_path(repo_root, str(queue["queue_id"]))
    with connect_state_readonly(state) as conn:
        rows = conn.execute(
            """
            SELECT experiment_id, step_id, status
            FROM step_state
            WHERE queue_id = ?
            """,
            (str(queue["queue_id"]),),
        ).fetchall()
    return {f"{row['experiment_id']}.{row['step_id']}": str(row["status"]) for row in rows}


def _node_status(node_id: str, status_map: Mapping[str, str]) -> str:
    return str(status_map.get(node_id) or "queued")


def _dependency_depths(nodes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    by_id = {str(node["node_id"]): node for node in nodes}
    memo: dict[str, int] = {}

    def depth(node_id: str, trail: set[str]) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in trail:
            raise ExperimentQueueError(f"cycle detected at node {node_id}")
        node = by_id[node_id]
        deps = [str(dep) for dep in node.get("dependencies", [])]
        value = 0 if not deps else 1 + max(depth(dep, {*trail, node_id}) for dep in deps)
        memo[node_id] = value
        return value

    for node_id in by_id:
        depth(node_id, set())
    return memo


def _utility(node: Mapping[str, Any]) -> float:
    expected_value = node.get("expected_value")
    estimated_cost = node.get("estimated_cost")
    if isinstance(expected_value, int | float) and not isinstance(expected_value, bool):
        value = float(expected_value)
    else:
        value = 1.0
    if isinstance(estimated_cost, int | float) and not isinstance(estimated_cost, bool) and float(estimated_cost) > 0:
        cost = float(estimated_cost)
    else:
        cost = 1.0
    return value / cost


def _machine_for_resource(
    resource_kind: str,
    pools: Sequence[Mapping[str, Any]],
    remaining: dict[tuple[str, str], int],
) -> str | None:
    best: tuple[float, str] | None = None
    for pool in pools:
        pool_id = str(pool["id"])
        slots = pool.get("slots") if isinstance(pool.get("slots"), Mapping) else {}
        if int(slots.get(resource_kind) or 0) <= 0:
            continue
        if remaining.get((pool_id, resource_kind), 0) <= 0:
            continue
        memory_gb = float(pool.get("memory_gb") or 0.0)
        disk_gb = float(pool.get("disk_gb") or 0.0)
        score = memory_gb + disk_gb * 0.05
        candidate = (score, pool_id)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    remaining[(best[1], resource_kind)] -= 1
    return best[1]


def _running_counts(
    nodes: Sequence[Mapping[str, Any]],
    status_map: Mapping[str, str],
    running_assignments: Mapping[str, str] | None,
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    if not running_assignments:
        return out
    by_id = {str(node["node_id"]): node for node in nodes}
    for node_id, machine_id in running_assignments.items():
        if _node_status(node_id, status_map) != "running":
            continue
        node = by_id.get(str(node_id))
        if node is None:
            continue
        key = (str(machine_id), str(node["resource_kind"]))
        out[key] = out.get(key, 0) + 1
    return out


def _load_json_artifact(path_text: object) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(path_text, str) or not path_text.strip():
        return None, "artifact_path_missing"
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return None, f"artifact_missing:{path_text}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"artifact_unreadable:{path_text}:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, f"artifact_not_json_object:{path_text}"
    return payload, None


def _storage_preflight_artifact_blockers(
    metadata: Mapping[str, Any],
) -> list[str]:
    deps = metadata.get("storage_preflight_dependencies")
    if not isinstance(deps, list):
        return []
    blockers: list[str] = []
    for index, dep in enumerate(deps):
        if not isinstance(dep, Mapping):
            blockers.append(f"dependency_{index}_not_object")
            continue
        storage_payload, storage_error = _load_json_artifact(
            dep.get("storage_plan_artifact_path")
        )
        if storage_error is not None:
            blockers.append(f"storage_plan_{storage_error}")
        elif storage_payload is not None:
            storage_blockers = storage_payload.get("blockers")
            if isinstance(storage_blockers, list) and storage_blockers:
                blockers.append(f"storage_plan_blockers_present:{storage_blockers!r}")
            elif storage_blockers not in (None, []):
                blockers.append(
                    f"storage_plan_blockers_malformed:{type(storage_blockers).__name__}"
                )
            if not isinstance(storage_payload.get("selected_workload_root"), str):
                blockers.append("storage_plan_selected_workload_root_missing")
            if storage_payload.get("selected_workload_root_matches_expected") is not True:
                blockers.append("storage_plan_selected_workload_root_not_expected")

        cleanup_payload, cleanup_error = _load_json_artifact(
            dep.get("cleanup_plan_artifact_path")
        )
        if cleanup_error is not None:
            blockers.append(f"cleanup_plan_{cleanup_error}")
        elif cleanup_payload is not None:
            plan = cleanup_payload.get("plan")
            if isinstance(plan, Mapping):
                for field in (
                    "score_claim",
                    "promotion_eligible",
                    "ready_for_exact_eval_dispatch",
                ):
                    if plan.get(field) is not False:
                        blockers.append(f"cleanup_plan_{field}_missing_or_not_false")
                if "candidate_count" not in plan:
                    blockers.append("cleanup_plan_candidate_count_missing")
                if "total_reclaimable_bytes" not in plan:
                    blockers.append("cleanup_plan_total_reclaimable_bytes_missing")
            else:
                blockers.append("cleanup_plan_plan_object_missing")
            execution = cleanup_payload.get("execution")
            if execution is None:
                pass
            elif isinstance(execution, Mapping):
                for field in (
                    "score_claim",
                    "promotion_eligible",
                    "ready_for_exact_eval_dispatch",
                ):
                    if execution.get(field) is not False:
                        blockers.append(
                            f"cleanup_execution_{field}_missing_or_not_false"
                        )
                if "executed_count" not in execution:
                    blockers.append("cleanup_execution_executed_count_missing")
                if "local_bytes_reclaimed" not in execution:
                    blockers.append("cleanup_execution_local_bytes_reclaimed_missing")
            else:
                blockers.append("cleanup_plan_execution_malformed")
    return blockers


def plan_staircase_dispatch(
    dag: Mapping[str, Any],
    *,
    status_map: Mapping[str, str] | None = None,
    running_assignments: Mapping[str, str] | None = None,
    max_nodes: int | None = None,
    allow_cloud: bool | None = None,
    diversity_bucket_limit: int | None = None,
) -> dict[str, Any]:
    normalized = normalize_staircase_dag(dag)
    statuses = dict(status_map or {})
    nodes = list(normalized["nodes"])
    depths = _dependency_depths(nodes)
    controls = normalized["controls"]
    storage = normalized.get("storage")
    storage_required = isinstance(storage, Mapping) and bool(storage.get("storage_required", True))
    storage_ready = (
        not storage_required
        or not isinstance(storage, Mapping)
        or isinstance(storage.get("selected_workload_root"), str)
    )
    cap = max_nodes if max_nodes is not None else int(controls["max_ready_nodes"])
    if cap <= 0:
        cap = int(controls["max_ready_nodes"])
    cloud_ok = bool(controls["allow_cloud"] if allow_cloud is None else allow_cloud)
    queue_mode = str(controls.get("mode") or "running")
    bucket_limit = (
        int(controls["diversity_bucket_limit"]) if diversity_bucket_limit is None else int(diversity_bucket_limit)
    )

    running = _running_counts(nodes, statuses, running_assignments)
    remaining: dict[tuple[str, str], int] = {}
    for pool in normalized["resource_pools"]:
        pool_id = str(pool["id"])
        slots = pool.get("slots") if isinstance(pool.get("slots"), Mapping) else {}
        for kind, slot_count in slots.items():
            key = (pool_id, str(kind))
            remaining[key] = max(0, int(slot_count) - running.get(key, 0))
    queue_limits = controls.get("max_concurrency")
    if isinstance(queue_limits, Mapping):
        running_by_kind: dict[str, int] = {}
        for (_pool_id, kind), count in running.items():
            running_by_kind[str(kind)] = running_by_kind.get(str(kind), 0) + int(count)
        for kind, limit in sorted(queue_limits.items()):
            kind_text = str(kind)
            budget = max(0, int(limit) - running_by_kind.get(kind_text, 0))
            for key in sorted(
                [item for item in remaining if item[1] == kind_text],
                key=lambda item: item[0],
            ):
                keep = min(remaining[key], budget)
                remaining[key] = keep
                budget -= keep

    ready_candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node["node_id"])
        status = _node_status(node_id, statuses)
        if status != "queued":
            continue
        if queue_mode != "running":
            blocked.append(
                {
                    "node_id": node_id,
                    "reason": "queue_control_not_running",
                    "mode": queue_mode,
                }
            )
            continue
        if not storage_ready:
            blocked.append(
                {
                    "node_id": node_id,
                    "reason": "no_eligible_storage_tier",
                    "storage": _storage_blocker_summary(storage),
                }
            )
            continue
        resource_kind = str(node["resource_kind"])
        if _is_cloud_resource(resource_kind) and not cloud_ok:
            blocked.append(
                {
                    "node_id": node_id,
                    "reason": "cloud_resource_requires_allow_cloud",
                    "resource_kind": resource_kind,
                }
            )
            continue
        missing_deps = [
            str(dep) for dep in node.get("dependencies", []) if _node_status(str(dep), statuses) != "succeeded"
        ]
        if missing_deps:
            blocked.append({"node_id": node_id, "reason": "dependencies_not_succeeded", "dependencies": missing_deps})
            continue
        preflight_blockers = _storage_preflight_artifact_blockers(
            node.get("metadata") if isinstance(node.get("metadata"), Mapping) else {}
        )
        if preflight_blockers:
            blocked.append(
                {
                    "node_id": node_id,
                    "reason": "storage_preflight_artifacts_not_valid",
                    "blockers": preflight_blockers,
                }
            )
            continue
        ready_candidates.append(dict(node))

    ready_candidates.sort(
        key=lambda node: (
            int(node["priority"]),
            depths[str(node["node_id"])],
            -_utility(node),
            str(node["family"]),
            str(node["stage"]),
            str(node["node_id"]),
        )
    )

    selected: list[StaircaseReadyNode] = []
    bucket_counts: dict[str, int] = {}
    deferred_for_diversity: list[dict[str, Any]] = []
    for node in ready_candidates:
        if len(selected) >= cap:
            break
        bucket = f"{node['family']}:{node['stage']}"
        if bucket_limit > 0 and bucket_counts.get(bucket, 0) >= bucket_limit:
            deferred_for_diversity.append(node)
            continue
        machine_id = _machine_for_resource(str(node["resource_kind"]), normalized["resource_pools"], remaining)
        if machine_id is None:
            blocked.append(
                {
                    "node_id": node["node_id"],
                    "reason": "no_available_resource_slot",
                    "resource_kind": node["resource_kind"],
                }
            )
            continue
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        selected.append(
            StaircaseReadyNode(
                node_id=str(node["node_id"]),
                priority=int(node["priority"]),
                utility=_utility(node),
                resource_kind=str(node["resource_kind"]),
                command=tuple(str(part) for part in node["command"]),
                dependencies=tuple(str(dep) for dep in node.get("dependencies", [])),
                family=str(node["family"]),
                stage=str(node["stage"]),
                machine_id=machine_id,
                metadata=dict(node.get("metadata") or {}),
            )
        )

    for node in deferred_for_diversity:
        if len(selected) >= cap:
            break
        machine_id = _machine_for_resource(str(node["resource_kind"]), normalized["resource_pools"], remaining)
        if machine_id is None:
            blocked.append(
                {
                    "node_id": node["node_id"],
                    "reason": "no_available_resource_slot",
                    "resource_kind": node["resource_kind"],
                }
            )
            continue
        selected.append(
            StaircaseReadyNode(
                node_id=str(node["node_id"]),
                priority=int(node["priority"]),
                utility=_utility(node),
                resource_kind=str(node["resource_kind"]),
                command=tuple(str(part) for part in node["command"]),
                dependencies=tuple(str(dep) for dep in node.get("dependencies", [])),
                family=str(node["family"]),
                stage=str(node["stage"]),
                machine_id=machine_id,
                metadata=dict(node.get("metadata") or {}),
            )
        )

    storage_hint = _storage_hint(storage if isinstance(storage, Mapping) else None)
    dask_tasks = []
    for node in selected:
        metadata = dict(node.metadata)
        task = {
            "key": f"{normalized['dag_id']}:{node.node_id}",
            "command": list(node.command),
            "resources": {node.resource_kind: 1, f"machine:{node.machine_id}": 1},
            "machine_hint": node.machine_id,
            "pure": False,
            "queue_id": metadata.get("queue_id"),
            "experiment_id": metadata.get("experiment_id"),
            "step_id": metadata.get("step_id"),
            "step_hashes": metadata.get("step_hashes"),
            "postconditions": list(metadata.get("postconditions") or []),
            "timeout_seconds": metadata.get("timeout_seconds"),
            "telemetry": dict(metadata.get("step_telemetry") or {}),
            "experiment_metadata": dict(metadata.get("experiment_metadata") or {}),
            "unit_kind": metadata.get("unit_kind"),
            "operation_family": metadata.get("operation_family"),
            "target_kind": metadata.get("target_kind"),
            "materializer_id": metadata.get("materializer_id"),
            "receiver_contract_id": metadata.get("receiver_contract_id"),
            "receiver_contract_kind": metadata.get("receiver_contract_kind"),
            "allowed_use": metadata.get("allowed_use"),
            "queue_state_writeback": {
                "required": True,
                "queue_id": metadata.get("queue_id"),
                "experiment_id": metadata.get("experiment_id"),
                "step_id": metadata.get("step_id"),
                "step_hashes": metadata.get("step_hashes"),
                "executor_must_claim_step_before_execution": True,
                "executor_must_record_terminal_step_event": True,
                "terminal_statuses": ["succeeded", "failed"],
            },
            "executor_boundary": "planning_only_task_must_write_back_to_experiment_queue_state",
        }
        if storage_hint is not None:
            task["storage_hint"] = storage_hint
        if isinstance(metadata.get("storage_preflight_dependency"), Mapping):
            task["storage_preflight_dependency"] = dict(metadata["storage_preflight_dependency"])
        storage_preflight_dependencies = metadata.get("storage_preflight_dependencies")
        if isinstance(storage_preflight_dependencies, list):
            task["storage_preflight_dependencies"] = [
                dict(dependency)
                for dependency in storage_preflight_dependencies
                if isinstance(dependency, Mapping)
            ]
        dask_tasks.append(task)
    plan = apply_proxy_evidence_boundary(
        {
            "schema": STAIRCASE_DISPATCH_PLAN_SCHEMA,
            "dag_id": normalized["dag_id"],
            "dag_hash": normalized["dag_hash"],
            "ready_count": len(ready_candidates),
            "selected_count": len(selected),
            "blocked_count": len(blocked),
            "selected_nodes": [node.to_dict() for node in selected],
            "blocked_nodes": blocked,
            "resource_pools": normalized["resource_pools"],
            "storage": storage,
            "dask_task_specs": dask_tasks,
            "executor_boundary": "planning_only_dask_or_local_executor_must_write_back_to_queue_state",
            "source_refs": normalized["source_refs"],
        },
        dispatch_blockers=[
            "staircase_dispatch_plan_is_planning_only",
            "executor_must_update_canonical_queue_state",
            "score_authority_requires_exact_auth_eval_axis",
        ],
    )
    plan["plan_hash"] = _sha256_json(
        {
            "dag_id": plan["dag_id"],
            "dag_hash": plan["dag_hash"],
            "selected_nodes": plan["selected_nodes"],
            "blocked_nodes": plan["blocked_nodes"],
            "resource_pools": plan["resource_pools"],
            "storage": plan["storage"],
        }
    )
    return plan


def _storage_hint(storage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if storage is None:
        return None
    selected = storage.get("selected_workload_root")
    if not isinstance(selected, str):
        return None
    return {
        "selected_tier": storage.get("selected_tier"),
        "selected_root": storage.get("selected_root"),
        "selected_workload_root": selected,
        "workload_subdir": storage.get("workload_subdir"),
        "executor_must_write_bulk_outputs_under_selected_workload_root": True,
    }


def _storage_blocker_summary(storage: object) -> dict[str, Any]:
    if not isinstance(storage, Mapping):
        return {"blockers": ["storage_plan_missing"]}
    blockers: list[str] = []
    tiers = storage.get("tiers")
    if isinstance(tiers, list):
        for tier in tiers:
            if not isinstance(tier, Mapping):
                continue
            tier_blockers = tier.get("blockers")
            if isinstance(tier_blockers, list):
                blockers.extend(f"{tier.get('name', 'tier')}:{item}" for item in tier_blockers)
    return {
        "schema": storage.get("schema"),
        "selected_tier": storage.get("selected_tier"),
        "selected_workload_root": storage.get("selected_workload_root"),
        "blockers": blockers or ["selected_workload_root_missing"],
    }


def parse_resource_pool_spec(spec: str) -> dict[str, Any]:
    """Parse ``id:local_cpu=4,local_mlx=1,memory_gb=128,disk_gb=80``."""

    pool_id, sep, rest = spec.partition(":")
    if not sep:
        raise ExperimentQueueError("resource pool spec must be id:key=value,...")
    slots: dict[str, int] = {}
    meta: dict[str, Any] = {"id": _require_text(pool_id, "resource_pool.id"), "tags": []}
    for part in rest.split(","):
        key, part_sep, value = part.partition("=")
        if not part_sep:
            raise ExperimentQueueError(f"invalid resource pool item: {part!r}")
        key = key.strip()
        value = value.strip()
        if key in {"memory_gb", "disk_gb"}:
            meta[key] = float(value)
        elif key == "label":
            meta[key] = value
        elif key == "tags":
            meta[key] = [item.strip() for item in value.split("+") if item.strip()]
        else:
            slots[key] = int(value)
    meta["slots"] = slots
    return normalize_resource_pools([meta])[0]


def load_staircase_dag(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentQueueError(f"{path}: DAG payload must be a JSON object")
    return normalize_staircase_dag(payload)


def write_staircase_dag(
    path: str | Path,
    dag: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
    min_free_bytes: int = 0,
) -> None:
    normalized = normalize_staircase_dag(dag)
    try:
        write_json_artifact(
            path,
            normalized,
            allow_overwrite=allow_overwrite,
            expected_existing_sha256=expected_existing_sha256,
            min_free_bytes=min_free_bytes,
        )
    except ArtifactWriteError as exc:
        raise ExperimentQueueError(str(exc)) from exc


__all__ = [
    "DEFAULT_MACHINE_PRESETS",
    "STAIRCASE_DAG_SCHEMA",
    "STAIRCASE_DISPATCH_PLAN_SCHEMA",
    "STORAGE_PREFLIGHT_DEPENDENCY_SCHEMA",
    "StaircaseReadyNode",
    "build_staircase_dag_from_experiment_queue",
    "build_storage_plan_payload",
    "default_local_resource_pools",
    "experiment_queue_status_map",
    "load_staircase_dag",
    "local_lab_resource_pools",
    "normalize_resource_pools",
    "normalize_staircase_dag",
    "parse_resource_pool_spec",
    "plan_staircase_dispatch",
    "write_staircase_dag",
]
