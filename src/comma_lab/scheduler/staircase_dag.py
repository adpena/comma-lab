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
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)

STAIRCASE_DAG_SCHEMA = "staircase_dag.v1"
STAIRCASE_DISPATCH_PLAN_SCHEMA = "staircase_dispatch_plan.v1"

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
        "slots": {"local_cpu": 8, "cuda_gpu": 1},
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
            "resource_kind": _require_text(raw.get("resource_kind", "local_cpu"), f"{node_id}.resource_kind"),
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
            normalized_slots[_require_text(kind, f"{pool_id}.slots key")] = _non_negative_int(
                count,
                f"{pool_id}.slots.{kind}",
            )
        if not any(normalized_slots.values()):
            raise ExperimentQueueError(f"{pool_id}.slots must contain at least one positive slot")
        normalized.append(
            {
                "id": pool_id,
                "label": str(raw_pool.get("label") or pool_id),
                "slots": normalized_slots,
                "memory_gb": _positive_float(raw_pool.get("memory_gb"), f"{pool_id}.memory_gb", default=1.0),
                "disk_gb": _positive_float(raw_pool.get("disk_gb"), f"{pool_id}.disk_gb", default=1.0),
                "tags": _string_list(raw_pool.get("tags"), f"{pool_id}.tags"),
            }
        )
    return normalized


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
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ExperimentQueueError("nodes must be a non-empty list")
    nodes = [
        _normalize_node(node, index=index)
        for index, node in enumerate(raw_nodes)
        if isinstance(node, Mapping)
    ]
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
    normalized = apply_proxy_evidence_boundary(
        {
            "schema": STAIRCASE_DAG_SCHEMA,
            "dag_id": dag_id,
            "controls": {
                "allow_cloud": bool(controls.get("allow_cloud", False)),
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
            },
            "resource_pools": normalize_resource_pools(
                payload.get("resource_pools") if isinstance(payload.get("resource_pools"), list) else None
            ),
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
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for experiment in queue.get("experiments", []):
        if not isinstance(experiment, Mapping):
            continue
        experiment_id = _require_text(experiment.get("id"), "experiment.id")
        priority = _non_negative_int(experiment.get("priority"), f"{experiment_id}.priority", default=100)
        experiment_tags = _string_list(experiment.get("tags"), f"{experiment_id}.tags")
        family = str(experiment.get("lane_id") or (experiment_tags[0] if experiment_tags else experiment_id))
        for step in experiment.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            step_id = _require_text(step.get("id"), f"{experiment_id}.step.id")
            resources = _mapping(step.get("resources"), f"{experiment_id}.{step_id}.resources")
            metadata = {
                "queue_id": queue.get("queue_id"),
                "experiment_id": experiment_id,
                "step_id": step_id,
                "step_hashes": _step_hashes(step),
                "postconditions": list(step.get("postconditions") or []),
            }
            nodes.append(
                {
                    "id": f"{experiment_id}.{step_id}",
                    "command": list(step.get("command") or []),
                    "dependencies": [
                        f"{experiment_id}.{required}"
                        for required in _string_list(step.get("requires"), f"{experiment_id}.{step_id}.requires")
                    ],
                    "priority": priority,
                    "resource_kind": str(resources.get("kind") or "local_cpu"),
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
            "resource_pools": list(resource_pools or default_local_resource_pools()),
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
    cap = max_nodes if max_nodes is not None else int(controls["max_ready_nodes"])
    if cap <= 0:
        cap = int(controls["max_ready_nodes"])
    cloud_ok = bool(controls["allow_cloud"] if allow_cloud is None else allow_cloud)
    bucket_limit = (
        int(controls["diversity_bucket_limit"])
        if diversity_bucket_limit is None
        else int(diversity_bucket_limit)
    )

    running = _running_counts(nodes, statuses, running_assignments)
    remaining: dict[tuple[str, str], int] = {}
    for pool in normalized["resource_pools"]:
        pool_id = str(pool["id"])
        slots = pool.get("slots") if isinstance(pool.get("slots"), Mapping) else {}
        for kind, slot_count in slots.items():
            key = (pool_id, str(kind))
            remaining[key] = max(0, int(slot_count) - running.get(key, 0))

    ready_candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node["node_id"])
        status = _node_status(node_id, statuses)
        if status != "queued":
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
            str(dep)
            for dep in node.get("dependencies", [])
            if _node_status(str(dep), statuses) != "succeeded"
        ]
        if missing_deps:
            blocked.append({"node_id": node_id, "reason": "dependencies_not_succeeded", "dependencies": missing_deps})
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
            )
        )

    dask_tasks = [
        {
            "key": f"{normalized['dag_id']}:{node.node_id}",
            "command": list(node.command),
            "resources": {node.resource_kind: 1},
            "machine_hint": node.machine_id,
            "pure": False,
        }
        for node in selected
    ]
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
        }
    )
    return plan


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


def write_staircase_dag(path: str | Path, dag: Mapping[str, Any]) -> None:
    normalized = normalize_staircase_dag(dag)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MACHINE_PRESETS",
    "STAIRCASE_DAG_SCHEMA",
    "STAIRCASE_DISPATCH_PLAN_SCHEMA",
    "StaircaseReadyNode",
    "build_staircase_dag_from_experiment_queue",
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
