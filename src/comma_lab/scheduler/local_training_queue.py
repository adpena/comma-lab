# SPDX-License-Identifier: MIT
"""Compile local representation-training plans into experiment_queue work."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

LOCAL_TRAINING_QUEUE_SCHEMA = "local_training_execution_queue_plan.v1"
TOOL_NAME = "comma_lab.scheduler.local_training_queue"
SUPPORTED_PLAN_SCHEMAS = frozenset(
    {
        "local_training_execution_plan.v1",
        "representation_training_probe_plan_v1",
        "pr95_local_training_probe_plan_v1",
        "pr95_mlx_long_training_plan.v1",
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
    "reproduction_claim": False,
    "pr95_1to1_reproduction_claim": False,
    "reproduction_equivalence": False,
}
LOCAL_TRAINING_RESOURCE_KINDS = frozenset(
    {"local", "local_cpu", "local_mlx", "local_cuda", "local_mps"}
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "local_training"


def _resolve_output_path(value: Any, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or value.startswith("<"):
        raise ExperimentQueueError("local training execution requires concrete output_manifest")
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _false_authority_postcondition(path: str) -> dict[str, Any]:
    return {
        "type": "json_false_authority",
        "path": path,
        "false_or_missing": sorted(FALSE_AUTHORITY),
    }


def _json_equals_postcondition(path: str, key: str, value: Any) -> dict[str, Any]:
    return {"type": "json_equals", "path": path, "key": key, "equals": value}


def _extra_artifact_postconditions_from_execution(
    execution: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> list[dict[str, Any]]:
    raw_postconditions = execution.get("extra_artifact_postconditions")
    if raw_postconditions is None:
        return []
    if not isinstance(raw_postconditions, list):
        raise ExperimentQueueError(f"{label}.extra_artifact_postconditions must be a list")
    normalized: list[dict[str, Any]] = []
    for index, raw_condition in enumerate(raw_postconditions):
        if not isinstance(raw_condition, Mapping):
            raise ExperimentQueueError(
                f"{label}.extra_artifact_postconditions[{index}] must be an object"
            )
        condition = dict(raw_condition)
        path = condition.get("path")
        if not isinstance(path, str) or not path.strip() or path.startswith("<"):
            raise ExperimentQueueError(
                f"{label}.extra_artifact_postconditions[{index}].path "
                "must be a concrete path"
            )
        _resolve_output_path(path, repo_root=repo_root)
        normalized.append(condition)
    return normalized


def _execution_from_plan(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = plan.get("recommended_execution")
    if not isinstance(execution, Mapping):
        raise ExperimentQueueError(
            "local training plan missing recommended_execution; "
            "queue compilation requires a concrete runner command"
        )
    return execution


def _resource_kind(execution: Mapping[str, Any]) -> str:
    explicit = str(execution.get("resource_kind") or "").strip()
    if explicit:
        return explicit
    scheduler_resource = str(execution.get("scheduler_resource_kind") or "").strip()
    if scheduler_resource:
        return scheduler_resource
    backend = str(execution.get("training_backend") or execution.get("backend") or "").lower()
    device = str(execution.get("device") or "").lower()
    if backend == "mlx":
        return "local_mlx" if device in {"gpu", "auto", "mlx", ""} else "local_cpu"
    if backend in {"numpy", "np", "local_numpy", "macos_numpy"}:
        return "local_cpu"
    if device == "cuda":
        return "local_cuda"
    if device == "mps":
        return "local_mps"
    if device in {"cpu", "numpy", "local_numpy"}:
        return "local_cpu"
    return "local"


def _command_args(execution: Mapping[str, Any]) -> list[str]:
    command = execution.get("python_command_args")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        raise ExperimentQueueError("local training execution missing python_command_args")
    if any(item.startswith("<") and item.endswith(">") for item in command):
        raise ExperimentQueueError("local training command contains placeholder argument")
    if command[0] != ".venv/bin/python":
        raise ExperimentQueueError("local training command must run under .venv/bin/python")
    tool = str(execution.get("tool") or "")
    if tool and len(command) > 1 and command[1] != tool:
        raise ExperimentQueueError(
            f"local training command tool {command[1]!r} does not match {tool!r}"
        )
    return list(command)


def _flag_values(command: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item != flag:
            continue
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ExperimentQueueError(f"local training command flag {flag} requires a value")
        values.append(command[index + 1])
    return values


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    values = _flag_values(command, flag)
    if len(values) > 1:
        raise ExperimentQueueError(f"local training command flag {flag} appears multiple times")
    return values[0] if values else None


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)


def _identity_from_plan(
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer_recipe = plan.get("optimizer_recipe")
    if not isinstance(optimizer_recipe, Mapping):
        optimizer_recipe = {}
    candidate_params = plan.get("candidate_params")
    if not isinstance(candidate_params, Mapping):
        candidate_params = {}
    out: dict[str, Any] = {}
    for key in (
        "stage_index",
        "seed",
        "optimizer_descriptor_id",
        "optimizer_config_sha256",
        "optimizer_backend_status",
        "parameter_group_lr_policy_id",
        "parameter_group_lr_policy_sha256",
        "parameter_group_fingerprint_sha256",
    ):
        value = None
        for source in (candidate_params, optimizer_recipe, execution, plan):
            if key in source and source.get(key) is not None:
                value = source[key]
                break
        if value is not None:
            out[key] = value
    return out


def _resolve_command_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _validate_command_writes_declared_outputs(
    command: Sequence[str],
    execution: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> None:
    """Fail closed when command outputs can drift from watched postconditions."""

    output_manifest = _resolve_output_path(execution.get("output_manifest"), repo_root=repo_root)
    output_dir_value = _flag_value(command, "--output-dir")
    matching_output = False
    for flag in ("--output-manifest", "--manifest-out", "--output", "--output-report"):
        value = _flag_value(command, flag)
        if value is None:
            continue
        candidate = _resolve_command_path(value, repo_root=repo_root)
        if not _same_path(candidate, output_manifest):
            raise ExperimentQueueError(
                f"{label}: local training command {flag}={value!r} does not match "
                f"recommended_execution.output_manifest {execution.get('output_manifest')!r}"
            )
        matching_output = True
    if output_dir_value is not None:
        output_dir = _resolve_command_path(output_dir_value, repo_root=repo_root)
        if not _same_path(output_dir / "manifest.json", output_manifest):
            raise ExperimentQueueError(
                f"{label}: local training command --output-dir={output_dir_value!r} "
                "does not contain recommended_execution.output_manifest"
            )
        matching_output = True
    if not matching_output:
        raise ExperimentQueueError(
            f"{label}: local training command must declare --output, "
            "--output-manifest, --manifest-out, or --output-dir matching "
            "recommended_execution.output_manifest"
        )

    representation_manifest = execution.get("representation_manifest")
    if isinstance(representation_manifest, str) and representation_manifest.strip():
        representation_path = _resolve_output_path(representation_manifest, repo_root=repo_root)
        value = _flag_value(command, "--representation-manifest")
        if value is not None:
            candidate = _resolve_command_path(value, repo_root=repo_root)
            if not _same_path(candidate, representation_path):
                raise ExperimentQueueError(
                    f"{label}: local training command --representation-manifest={value!r} "
                    "does not match recommended_execution.representation_manifest"
                )
            return
        if output_dir_value is None or not _same_path(
            _resolve_command_path(output_dir_value, repo_root=repo_root)
            / "representation_training_manifest.json",
            representation_path,
        ):
            raise ExperimentQueueError(
                f"{label}: local training command must declare "
                "--representation-manifest or --output-dir matching "
                "recommended_execution.representation_manifest"
            )


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    index: int,
    repo_root: Path,
) -> Mapping[str, Any]:
    label = f"local_training_plan[{index}]"
    schema = str(plan.get("schema") or "")
    if schema not in SUPPORTED_PLAN_SCHEMAS:
        raise ExperimentQueueError(f"{label}: unsupported schema {schema!r}")
    try:
        require_no_truthy_authority_fields(plan, context=label)
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    execution = _execution_from_plan(plan)
    try:
        require_no_truthy_authority_fields(execution, context=f"{label}.recommended_execution")
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY.items():
        if key in execution and execution.get(key) is not expected:
            raise ExperimentQueueError(f"{label}.recommended_execution.{key} must be false")
    resource_kind = _resource_kind(execution)
    if resource_kind not in LOCAL_TRAINING_RESOURCE_KINDS:
        raise ExperimentQueueError(
            f"{label}.recommended_execution.resource_kind must be a local training "
            f"resource, got {resource_kind!r}"
        )
    command = _command_args(execution)
    _validate_command_writes_declared_outputs(
        command,
        execution,
        repo_root=repo_root,
        label=label,
    )
    return execution


def build_local_training_execution_queue(
    plans: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str = "local_representation_training",
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    local_cuda_concurrency: int = 1,
    local_mps_concurrency: int = 1,
    timeout_seconds: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return ``experiment_queue.v1`` work for local training execution plans."""

    if not plans:
        raise ExperimentQueueError("at least one local training plan is required")
    for label, value in (
        ("local_cpu_concurrency", local_cpu_concurrency),
        ("local_mlx_concurrency", local_mlx_concurrency),
        ("local_cuda_concurrency", local_cuda_concurrency),
        ("local_mps_concurrency", local_mps_concurrency),
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
    seen_output_paths: set[str] = set()
    for index, plan in enumerate(selected):
        execution = _validate_plan(plan, index=index, repo_root=repo)
        label = f"local_training_plan[{index}].recommended_execution"
        output_path = _resolve_output_path(execution.get("output_manifest"), repo_root=repo)
        output_key = output_path.expanduser().resolve(strict=False).as_posix()
        if output_key in seen_output_paths:
            raise ExperimentQueueError(
                f"local_training_plan[{index}]: duplicate output_manifest {output_key!r}"
            )
        seen_output_paths.add(output_key)
        output_rel = _repo_rel(output_path, repo)
        optimizer_identity = _identity_from_plan(plan, execution)
        representation_manifest = execution.get("representation_manifest")
        experiment_id = f"local_training_{index:04d}_{_safe_id(str(plan.get('candidate_id') or index))}"
        postconditions: list[dict[str, Any]] = [
            {"type": "path_exists", "path": output_rel},
            _false_authority_postcondition(output_rel),
        ]
        artifact_paths = [output_rel]
        extra_postconditions = _extra_artifact_postconditions_from_execution(
            execution,
            repo_root=repo,
            label=label,
        )
        if isinstance(representation_manifest, str) and representation_manifest:
            representation_path = _resolve_output_path(representation_manifest, repo_root=repo)
            representation_key = representation_path.expanduser().resolve(strict=False).as_posix()
            if representation_key in seen_output_paths:
                raise ExperimentQueueError(
                    f"local_training_plan[{index}]: duplicate representation_manifest "
                    f"{representation_key!r}"
                )
            seen_output_paths.add(representation_key)
            representation_rel = _repo_rel(representation_path, repo)
            artifact_paths.append(representation_rel)
            postconditions.extend(
                [
                    {"type": "path_exists", "path": representation_rel},
                    {
                        "type": "json_equals",
                        "path": representation_rel,
                        "key": "schema",
                        "equals": "representation_training_probe_manifest_v1",
                    },
                    _false_authority_postcondition(representation_rel),
                ]
            )
            candidate_id = str(plan.get("candidate_id") or "")
            if candidate_id:
                postconditions.append(
                    _json_equals_postcondition(
                        representation_rel,
                        "candidate_id",
                        candidate_id,
                    )
                )
            for identity_key, json_key in (
                ("stage_index", "candidate_params.stage_index"),
                ("seed", "seed"),
                ("optimizer_descriptor_id", "candidate_params.optimizer_descriptor_id"),
                ("optimizer_config_sha256", "candidate_params.optimizer_config_sha256"),
                (
                    "parameter_group_lr_policy_id",
                    "candidate_params.parameter_group_lr_policy_id",
                ),
                (
                    "parameter_group_lr_policy_sha256",
                    "candidate_params.parameter_group_lr_policy_sha256",
                ),
            ):
                if identity_key in optimizer_identity:
                    postconditions.append(
                        _json_equals_postcondition(
                            representation_rel,
                            json_key,
                            optimizer_identity[identity_key],
                        )
                    )
        extra_artifact_keys: set[str] = set()
        for condition in extra_postconditions:
            path_value = condition.get("path")
            if isinstance(path_value, str):
                extra_path = _resolve_output_path(path_value, repo_root=repo)
                extra_key = extra_path.expanduser().resolve(strict=False).as_posix()
                if (
                    extra_key in seen_output_paths
                    and extra_key != output_key
                    and extra_key not in extra_artifact_keys
                ):
                    raise ExperimentQueueError(
                        f"local_training_plan[{index}]: duplicate extra artifact "
                        f"{extra_key!r}"
                    )
                seen_output_paths.add(extra_key)
                extra_artifact_keys.add(extra_key)
                extra_rel = _repo_rel(extra_path, repo)
                condition = {**condition, "path": extra_rel}
                if extra_rel not in artifact_paths:
                    artifact_paths.append(extra_rel)
            postconditions.append(condition)
        resource_kind = _resource_kind(execution)
        source_dir = str(plan.get("source_dir") or "").strip()
        input_artifact_paths = [source_dir] if source_dir else []
        source_tree_sha256 = str(plan.get("source_tree_sha256") or "").strip()
        metadata: dict[str, Any] = {
            "schema": LOCAL_TRAINING_QUEUE_SCHEMA,
            "tool": TOOL_NAME,
            "generated_at_utc": generated_at,
            "source_plan_schema": plan.get("schema"),
            "candidate_id": plan.get("candidate_id"),
            "representation_family": plan.get("representation_family"),
            "substrate_family": plan.get("substrate_family"),
            "training_backend": execution.get("training_backend"),
            "device": execution.get("device"),
            "candidate_generation_only": True,
            "requires_exact_eval_before_promotion": True,
            **optimizer_identity,
            **FALSE_AUTHORITY,
        }
        if source_tree_sha256:
            metadata["source_tree_sha256"] = source_tree_sha256
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": str(plan.get("lane_id") or lane_id),
                "priority": 10 + index,
                "metadata": metadata,
                "steps": [
                    {
                        "id": "run_local_training",
                        "kind": "command",
                        "command": _command_args(execution),
                        "resources": {"kind": resource_kind},
                        "timeout_seconds": timeout_seconds,
                        "postconditions": postconditions,
                        "telemetry": {
                            "artifact_paths": artifact_paths,
                            "input_artifact_paths": input_artifact_paths,
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
                    "local_cuda": local_cuda_concurrency,
                    "local_mps": local_mps_concurrency,
                },
            },
            "experiments": experiments,
        }
    )


__all__ = [
    "LOCAL_TRAINING_QUEUE_SCHEMA",
    "build_local_training_execution_queue",
]
