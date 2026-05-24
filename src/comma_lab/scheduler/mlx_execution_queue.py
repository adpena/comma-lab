# SPDX-License-Identifier: MIT
"""Compile MLX scorer-response execution plans into experiment_queue work."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_execution_plan import SCHEMA_VERSION as MLX_EXECUTION_PLAN_SCHEMA
from tac.local_acceleration.mlx_scorer_response import SCHEMA_VERSION as MLX_RESPONSE_SCHEMA
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

from .experiment_queue import QUEUE_SCHEMA, ExperimentQueueError, normalize_queue_definition

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

MLX_EXECUTION_QUEUE_SCHEMA = "mlx_scorer_response_execution_queue_plan.v1"
TOOL_NAME = "comma_lab.scheduler.mlx_execution_queue"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "mlx_response"


def _resolve_output_path(value: Any, *, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or value.startswith("<"):
        raise ExperimentQueueError("mlx execution plan requires concrete response_output")
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _flag_values(command: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item != flag:
            continue
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ExperimentQueueError(f"mlx command flag {flag} requires a value")
        values.append(command[index + 1])
    return values


def _flag_value(command: Sequence[str], flag: str) -> str | None:
    values = _flag_values(command, flag)
    if len(values) > 1:
        raise ExperimentQueueError(f"mlx command flag {flag} appears multiple times")
    return values[0] if values else None


def _has_flag(command: Sequence[str], flag: str) -> bool:
    return flag in command


def _expect_flag(command: Sequence[str], flag: str, expected: Any) -> None:
    actual = _flag_value(command, flag)
    expected_text = str(expected)
    if actual is None:
        raise ExperimentQueueError(f"mlx command missing {flag}")
    if actual != expected_text:
        raise ExperimentQueueError(
            f"mlx command {flag}={actual!r} does not match "
            f"recommended_execution {expected_text!r}"
        )


def _expect_optional_flag(command: Sequence[str], flag: str, expected: Any) -> None:
    actual = _flag_value(command, flag)
    if expected is None:
        if actual is not None:
            raise ExperimentQueueError(f"mlx command unexpected {flag}")
        return
    expected_text = str(expected)
    if actual != expected_text:
        raise ExperimentQueueError(
            f"mlx command {flag}={actual!r} does not match "
            f"recommended_execution {expected_text!r}"
        )


def _validate_command_matches_execution(
    command: Sequence[str],
    execution: Mapping[str, Any],
) -> None:
    tool = str(execution.get("tool") or "")
    if len(command) < 2:
        raise ExperimentQueueError("mlx command is too short")
    if command[0] != ".venv/bin/python":
        raise ExperimentQueueError("mlx command must run under .venv/bin/python")
    if command[1] != tool:
        raise ExperimentQueueError(
            f"mlx command tool {command[1]!r} does not match recommended_execution.tool {tool!r}"
        )
    for flag, key in (
        ("--reference-cache-dir", "reference_cache_dir"),
        ("--candidate-cache-dir", "candidate_cache_dir"),
        ("--archive-size-bytes", "archive_size_bytes"),
        ("--repo-root", "repo_root"),
        ("--batch-pairs", "batch_pairs"),
        ("--start-pair", "start_pair"),
        ("--max-pairs", "max_pairs"),
        ("--device", "device"),
        ("--output", "response_output"),
    ):
        _expect_flag(command, flag, execution.get(key))
    progress_every = int(execution.get("progress_every") or 0)
    if progress_every > 0:
        _expect_flag(command, "--progress-every", progress_every)
    else:
        _expect_optional_flag(command, "--progress-every", None)
    _expect_optional_flag(command, "--components-dir", execution.get("components_dir"))

    device = str(execution.get("device") or "")
    has_gpu_flag = _has_flag(command, "--allow-gpu-research-signal")
    if device == "gpu" and not has_gpu_flag:
        raise ExperimentQueueError(
            "mlx gpu execution command missing --allow-gpu-research-signal"
        )
    if device == "cpu" and has_gpu_flag:
        raise ExperimentQueueError(
            "mlx cpu execution command must not include --allow-gpu-research-signal"
        )

    batch_pairs = int(execution.get("batch_pairs") or 0)
    has_batch_flag = _has_flag(command, "--allow-batch-shape-research-signal")
    if batch_pairs != 1 and not has_batch_flag:
        raise ExperimentQueueError(
            "mlx batch-shape execution command missing "
            "--allow-batch-shape-research-signal"
        )
    if batch_pairs == 1 and has_batch_flag:
        raise ExperimentQueueError(
            "mlx singleton-batch command must not include "
            "--allow-batch-shape-research-signal"
        )


def _command_args(plan: Mapping[str, Any], execution: Mapping[str, Any]) -> list[str]:
    command = execution.get("python_command_args")
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        raise ExperimentQueueError("mlx execution plan missing python_command_args")
    if "<required-response-output.json>" in command:
        raise ExperimentQueueError("mlx execution plan command uses placeholder response output")
    parsed = list(command)
    _validate_command_matches_execution(parsed, execution)
    return parsed


def _validate_plan(plan: Mapping[str, Any], *, index: int) -> Mapping[str, Any]:
    label = f"mlx_execution_plan[{index}]"
    if plan.get("schema_version") != MLX_EXECUTION_PLAN_SCHEMA:
        raise ExperimentQueueError(f"{label}: expected schema_version {MLX_EXECUTION_PLAN_SCHEMA}")
    try:
        require_no_truthy_authority_fields(plan, context=label)
    except ValueError as exc:
        raise ExperimentQueueError(str(exc)) from exc
    for key, expected in FALSE_AUTHORITY.items():
        if plan.get(key) is not expected:
            raise ExperimentQueueError(f"{label}: {key} must be explicit false")
    if plan.get("evidence_grade") != EVIDENCE_GRADE_MLX:
        raise ExperimentQueueError(f"{label}: evidence_grade must be {EVIDENCE_GRADE_MLX}")
    if plan.get("candidate_generation_only") is not True:
        raise ExperimentQueueError(f"{label}: candidate_generation_only must be true")
    execution = plan.get("recommended_execution")
    if not isinstance(execution, Mapping):
        raise ExperimentQueueError(f"{label}: recommended_execution must be an object")
    if execution.get("device") not in {"cpu", "gpu"}:
        raise ExperimentQueueError(f"{label}: recommended_execution.device must be cpu or gpu")
    return execution


def build_mlx_scorer_response_execution_queue(
    plans: Sequence[Mapping[str, Any]],
    *,
    queue_id: str,
    repo_root: str | Path,
    lane_id: str = "mlx_scorer_response_local_substrate",
    local_cpu_concurrency: int = 1,
    local_mlx_concurrency: int = 1,
    timeout_seconds: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return ``experiment_queue.v1`` work for concrete MLX execution plans."""

    if not plans:
        raise ExperimentQueueError("at least one MLX execution plan is required")
    if isinstance(local_cpu_concurrency, bool) or local_cpu_concurrency < 1:
        raise ExperimentQueueError("local_cpu_concurrency must be >= 1")
    if isinstance(local_mlx_concurrency, bool) or local_mlx_concurrency < 1:
        raise ExperimentQueueError("local_mlx_concurrency must be >= 1")
    if isinstance(timeout_seconds, bool) or timeout_seconds < 0:
        raise ExperimentQueueError("timeout_seconds must be non-negative")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ExperimentQueueError("limit must be >= 1 when provided")

    repo = Path(repo_root)
    selected = list(plans[:limit] if limit is not None else plans)
    queue_metadata = {
        "schema": MLX_EXECUTION_QUEUE_SCHEMA,
        "tool": TOOL_NAME,
        "generated_at_utc": _utc_now(),
        "plan_count": len(selected),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    experiments: list[dict[str, Any]] = []
    for index, plan in enumerate(selected):
        execution = _validate_plan(plan, index=index)
        output_path = _resolve_output_path(execution.get("response_output"), repo_root=repo)
        source_run_id = str(plan.get("source_run_id") or f"plan_{index:04d}")
        device = str(execution["device"])
        resource_kind = "local_mlx" if device == "gpu" else "local_cpu"
        experiment_id = f"mlx_response_{index:04d}_{_safe_id(source_run_id)}"
        step = {
            "id": "run_mlx_scorer_response",
            "kind": "command",
            "command": _command_args(plan, execution),
            "resources": {"kind": resource_kind},
            "timeout_seconds": timeout_seconds,
            "postconditions": [
                {"type": "path_exists", "path": _repo_rel(output_path, repo)},
                {
                    "type": "json_equals",
                    "path": _repo_rel(output_path, repo),
                    "key": "schema_version",
                    "equals": MLX_RESPONSE_SCHEMA,
                },
                {
                    "type": "json_equals",
                    "path": _repo_rel(output_path, repo),
                    "key": "evidence_tag",
                    "equals": EVIDENCE_TAG_MLX,
                },
                {
                    "type": "json_false_authority",
                    "path": _repo_rel(output_path, repo),
                    "axis_key": "score_axis",
                    "axis_equals": EVIDENCE_TAG_MLX,
                },
            ],
            "telemetry": {
                "artifact_paths": [_repo_rel(output_path, repo)],
                "input_artifact_paths": [
                    str(execution.get("reference_cache_dir")),
                    str(execution.get("candidate_cache_dir")),
                ],
                "recursive": False,
            },
        }
        components_dir = execution.get("components_dir")
        if isinstance(components_dir, str) and components_dir:
            step["telemetry"]["artifact_paths"].append(
                _repo_rel(_resolve_output_path(components_dir, repo_root=repo), repo)
            )
            step["telemetry"]["recursive"] = True
        experiments.append(
            {
                "id": experiment_id,
                "lane_id": lane_id,
                "priority": 10 + index,
                "metadata": {
                    **queue_metadata,
                    "source_run_id": plan.get("source_run_id"),
                    "source_profile_verdict": plan.get("source_profile_verdict"),
                    "evidence_grade": EVIDENCE_GRADE_MLX,
                    "evidence_tag": EVIDENCE_TAG_MLX,
                    "device": device,
                    "batch_pairs": execution.get("batch_pairs"),
                    "pair_window": execution.get("pair_window"),
                    "candidate_generation_only": True,
                    "requires_exact_eval_before_promotion": True,
                    **FALSE_AUTHORITY,
                },
                "steps": [step],
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
    "MLX_EXECUTION_QUEUE_SCHEMA",
    "build_mlx_scorer_response_execution_queue",
]
