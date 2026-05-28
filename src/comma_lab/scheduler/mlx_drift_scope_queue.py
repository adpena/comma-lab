# SPDX-License-Identifier: MIT
"""Compile local MLX drift-scope search plans into experiment_queue work."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

MLX_DRIFT_SCOPE_PLAN_SCHEMA = "local_mlx_drift_scope_search_plan.v1"
MLX_DRIFT_SCOPE_QUEUE_SCHEMA = "mlx_drift_scope_search_execution_queue_plan.v1"
TOOL_NAME = "comma_lab.scheduler.mlx_drift_scope_queue"
SUPPORTED_OUTPUT_SCHEMAS = frozenset(
    {
        "pr95_hnerv_mlx_conv2d_drift_scope_search.v1",
    }
)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_packet_ready": False,
}
LOCAL_RESOURCE_KINDS = frozenset({"local_cpu", "local_mlx"})


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "drift_scope"


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(value: Any, *, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or value.startswith("<"):
        raise ExperimentQueueError(f"{label} must be a concrete path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _flag_values(command: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item != flag:
            continue
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ExperimentQueueError(f"MLX drift-scope command flag {flag} requires a value")
        values.append(command[index + 1])
    return values


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    values = _flag_values(command, flag)
    if len(values) > 1:
        raise ExperimentQueueError(f"MLX drift-scope command flag {flag} appears multiple times")
    return values[0] if values else None


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)


def _execution_from_plan(plan: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    execution = plan.get("recommended_execution")
    if not isinstance(execution, Mapping):
        raise ExperimentQueueError(f"{label} missing recommended_execution")
    return execution


def _resource_kind(execution: Mapping[str, Any]) -> str:
    explicit = str(execution.get("resource_kind") or "").strip()
    if explicit:
        return explicit
    device = str(execution.get("mlx_device") or execution.get("device") or "").lower()
    return "local_mlx" if device in {"gpu", "mlx", "auto"} else "local_cpu"


def _command_args(
    execution: Mapping[str, Any],
    *,
    output_manifest: Path,
    repo_root: Path,
    label: str,
) -> list[str]:
    command = execution.get("python_command_args")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        raise ExperimentQueueError(f"{label}.recommended_execution missing python_command_args")
    if any(item.startswith("<") and item.endswith(">") for item in command):
        raise ExperimentQueueError(f"{label}.recommended_execution command contains placeholder argument")
    if len(command) < 2 or command[0] != ".venv/bin/python":
        raise ExperimentQueueError("MLX drift-scope command must run under .venv/bin/python")
    tool = str(execution.get("tool") or "")
    if tool and command[1] != tool:
        raise ExperimentQueueError(
            f"MLX drift-scope command tool {command[1]!r} does not match {tool!r}"
        )
    output_dir_value = _flag_value(command, "--output-dir")
    if output_dir_value is None:
        raise ExperimentQueueError("MLX drift-scope command must declare --output-dir")
    output_dir = _resolve_path(output_dir_value, repo_root=repo_root, label="--output-dir")
    if not _same_path(output_dir / "scope_search_summary.json", output_manifest):
        raise ExperimentQueueError(
            f"{label}: command --output-dir does not contain recommended output_manifest"
        )
    return list(command)


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    index: int,
    repo_root: Path,
) -> tuple[Mapping[str, Any], Path, list[str]]:
    label = f"drift_scope_plan[{index}]"
    schema = str(plan.get("schema") or "")
    if schema != MLX_DRIFT_SCOPE_PLAN_SCHEMA:
        raise ExperimentQueueError(f"{label}: unsupported schema {schema!r}")
    try:
        require_no_truthy_authority_fields(plan, context=label)
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    execution = _execution_from_plan(plan, label=label)
    try:
        require_no_truthy_authority_fields(execution, context=f"{label}.recommended_execution")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY.items():
        if key in execution and execution.get(key) is not expected:
            raise ExperimentQueueError(f"{label}.recommended_execution.{key} must be false")
    output_schema = str(execution.get("expected_output_schema") or plan.get("expected_output_schema") or "")
    if output_schema not in SUPPORTED_OUTPUT_SCHEMAS:
        raise ExperimentQueueError(f"{label}: unsupported expected_output_schema {output_schema!r}")
    resource_kind = _resource_kind(execution)
    if resource_kind not in LOCAL_RESOURCE_KINDS:
        raise ExperimentQueueError(
            f"{label}.recommended_execution.resource_kind must be local CPU/MLX, got {resource_kind!r}"
        )
    output_manifest = _resolve_path(
        execution.get("output_manifest"),
        repo_root=repo_root,
        label=f"{label}.recommended_execution.output_manifest",
    )
    command = _command_args(
        execution,
        output_manifest=output_manifest,
        repo_root=repo_root,
        label=label,
    )
    return execution, output_manifest, command


def _metadata_from_plan(
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    generated_at: str,
) -> dict[str, Any]:
    optimization_target = plan.get("optimization_target")
    if not isinstance(optimization_target, Mapping):
        optimization_target = {}
    return {
        "schema": MLX_DRIFT_SCOPE_QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": generated_at,
        "source_plan_schema": plan.get("schema"),
        "candidate_id": plan.get("candidate_id"),
        "lane_id": plan.get("lane_id"),
        "archive_family": plan.get("archive_family"),
        "candidate_family": plan.get("candidate_family"),
        "optimization_profile": optimization_target.get("profile"),
        "target_video": optimization_target.get("target_video"),
        "video_scope": optimization_target.get("video_scope"),
        "generalizable_against_videos": optimization_target.get("generalizable_against_videos"),
        "expected_output_schema": execution.get("expected_output_schema"),
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        **FALSE_AUTHORITY,
    }


def build_mlx_drift_scope_search_queue(
    plans: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str = "local_mlx_drift_scope_search",
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    timeout_seconds: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return ``experiment_queue.v1`` work for local MLX drift-scope plans."""

    if not plans:
        raise ExperimentQueueError("at least one MLX drift-scope plan is required")
    for label, value in (
        ("local_cpu_concurrency", local_cpu_concurrency),
        ("local_mlx_concurrency", local_mlx_concurrency),
    ):
        if isinstance(value, bool) or value < 1:
            raise ExperimentQueueError(f"{label} must be >= 1")
    if isinstance(timeout_seconds, bool) or timeout_seconds < 0:
        raise ExperimentQueueError("timeout_seconds must be non-negative")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ExperimentQueueError("limit must be >= 1 when provided")

    repo = Path(repo_root)
    selected = list(plans[:limit] if limit is not None else plans)
    generated_at = _utc_now()
    experiments: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    for index, plan in enumerate(selected):
        execution, output_manifest, command = _validate_plan(
            plan,
            index=index,
            repo_root=repo,
        )
        output_key = output_manifest.expanduser().resolve(strict=False).as_posix()
        if output_key in seen_outputs:
            raise ExperimentQueueError(
                f"drift_scope_plan[{index}]: duplicate output_manifest {output_key!r}"
            )
        seen_outputs.add(output_key)
        output_rel = _repo_rel(output_manifest, repo)
        output_schema = str(execution["expected_output_schema"])
        optimization_target = plan.get("optimization_target")
        if not isinstance(optimization_target, Mapping):
            optimization_target = {}
        experiment_id = f"mlx_drift_scope_{index:04d}_{_safe_id(str(plan.get('candidate_id') or index))}"
        postconditions: list[dict[str, Any]] = [
            {"type": "path_exists", "path": output_rel},
            {"type": "json_equals", "path": output_rel, "key": "schema", "equals": output_schema},
            {
                "type": "json_equals",
                "path": output_rel,
                "key": "exact_readiness_refusal.ready",
                "equals": False,
            },
            {"type": "json_false_authority", "path": output_rel},
        ]
        if "profile" in optimization_target:
            postconditions.append(
                {
                    "type": "json_equals",
                    "path": output_rel,
                    "key": "optimization_target.profile",
                    "equals": optimization_target["profile"],
                }
            )
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": str(plan.get("lane_id") or lane_id),
                "priority": 10 + index,
                "metadata": _metadata_from_plan(plan, execution, generated_at=generated_at),
                "steps": [
                    {
                        "id": "run_mlx_drift_scope_search",
                        "kind": "command",
                        "command": command,
                        "resources": {"kind": _resource_kind(execution)},
                        "timeout_seconds": timeout_seconds,
                        "postconditions": postconditions,
                        "telemetry": {
                            "artifact_paths": [output_rel],
                            "recursive": True,
                            "max_recursive_entries": 512,
                        },
                    }
                ],
            }
        )

    return normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": queue_id,
            "controls": {
                "mode": "running",
                "local_first": True,
                "max_concurrency": {
                    "local_cpu": local_cpu_concurrency,
                    "local_mlx": local_mlx_concurrency,
                },
            },
            "experiments": experiments,
        }
    )


__all__ = [
    "MLX_DRIFT_SCOPE_PLAN_SCHEMA",
    "MLX_DRIFT_SCOPE_QUEUE_SCHEMA",
    "build_mlx_drift_scope_search_queue",
]
