# SPDX-License-Identifier: MIT
"""Small, typed migrations for experiment_queue.v1 definitions."""

from __future__ import annotations

import copy
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from typing import Any

MLX_RESPONSE_TOOLS = frozenset(
    {
        "tools/run_mlx_scorer_response_from_cache.py",
        "tools/run_mlx_scorer_response_from_local_advisory.py",
    }
)
SCORER_REGION_POLICY_COMPILER = "tools/build_scorer_region_selector_cascade_queue_from_policy.py"
MLX_CPU_SPEND_GATE_TOOL = "tools/gate_mlx_scorer_response_for_cpu_spend.py"
SCORER_REGION_OUTPUT_CHANGE_TOOL = "tools/prove_shell_inflate_output_change.py"
MLX_SUBMISSION_CACHE_TOOL = "tools/materialize_mlx_scorer_cache_from_submission.py"
MLX_RESPONSE_BATCH_TOOL = "tools/run_mlx_scorer_response_from_cache_batch.py"
JSON_ARTIFACT_VALIDATOR_TOOL = "tools/validate_json_artifact_contract.py"


def normalize_mlx_response_singleton_batches(
    queue: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Return a queue copy with production MLX response batches set to one.

    Cache construction may still use larger tensor batches. This migration only
    changes commands whose output is consumed as production scorer-response
    signal, plus nested scorer-region policy compiler commands that would emit
    the same invalid response batch shape downstream.
    """

    updated = copy.deepcopy(dict(queue))
    rewrites: list[dict[str, Any]] = []
    for experiment in updated.get("experiments", []):
        if not isinstance(experiment, MutableMapping):
            continue
        experiment_id = str(experiment.get("id", ""))
        for step in experiment.get("steps", []):
            if not isinstance(step, MutableMapping):
                continue
            command = step.get("command")
            if not isinstance(command, list):
                continue
            step_id = str(step.get("id", ""))
            if _command_contains_any(command, MLX_RESPONSE_TOOLS):
                rewrites.extend(
                    _set_flag_value(
                        command,
                        "--batch-pairs",
                        "1",
                        experiment_id=experiment_id,
                        step_id=step_id,
                    )
                )
            if _command_contains(command, SCORER_REGION_POLICY_COMPILER):
                rewrites.extend(
                    _set_flag_value(
                        command,
                        "--mlx-batch-pairs",
                        "1",
                        experiment_id=experiment_id,
                        step_id=step_id,
                    )
                )
            if _command_contains(command, MLX_CPU_SPEND_GATE_TOOL):
                old_value = step.get("on_postcondition_failure")
                if old_value != "skipped":
                    step["on_postcondition_failure"] = "skipped"
                    rewrites.append(
                        {
                            "experiment_id": experiment_id,
                            "step_id": step_id,
                            "field": "on_postcondition_failure",
                            "old_value": old_value,
                            "new_value": "skipped",
                        }
                    )
    metadata = updated.setdefault("metadata", {})
    if isinstance(metadata, MutableMapping):
        migrations = metadata.setdefault("queue_migrations", [])
        if isinstance(migrations, list):
            migrations.append(
                {
                    "schema": "experiment_queue_mlx_response_batch_migration.v1",
                    "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reason": reason,
                    "changed_command_count": len(rewrites),
                    "rewrites": rewrites,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return updated


def optimize_mlx_first_receiver_preinflated_cache_handoff(
    queue: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Reuse receiver output-change raw outputs for MLX-first cache materialization.

    Existing scorer-region queues often prove that the receiver patch changes
    ``inflate.sh`` output, then immediately run the same patched receiver again
    to build the MLX scorer-input cache. This migration preserves the
    output-change proof scratch and points the MLX cache step at
    ``scratch/right_out``. The score contract stays false-authority: this only
    removes duplicated local I/O and inflate work.
    """

    updated = copy.deepcopy(dict(queue))
    rewrites: list[dict[str, Any]] = []
    for experiment in updated.get("experiments", []):
        if not isinstance(experiment, MutableMapping):
            continue
        experiment_id = str(experiment.get("id", ""))
        steps = experiment.get("steps")
        if not isinstance(steps, list):
            continue
        by_id = {
            str(step.get("id", "")): step
            for step in steps
            if isinstance(step, MutableMapping)
        }
        proof_step = by_id.get("prove_receiver_patch_full_frame_output_change")
        cache_step = by_id.get("build_mlx_component_cache")
        if not isinstance(proof_step, MutableMapping) or not isinstance(
            cache_step,
            MutableMapping,
        ):
            continue
        proof_command = proof_step.get("command")
        cache_command = cache_step.get("command")
        if not isinstance(proof_command, list) or not isinstance(cache_command, list):
            continue
        if not _command_contains(proof_command, SCORER_REGION_OUTPUT_CHANGE_TOOL):
            continue
        if not _command_contains(cache_command, MLX_SUBMISSION_CACHE_TOOL):
            continue
        output_dir = _flag_value(proof_command, "--output-dir")
        if output_dir is None:
            continue
        right_out = f"{output_dir.rstrip('/')}/scratch/right_out"
        right_cache = _queue_root_right_cache(queue, output_dir)
        rewrites.extend(
            _ensure_flag(
                proof_command,
                "--right-cache-dir",
                right_cache,
                before_flag="--file-list-entry",
                experiment_id=experiment_id,
                step_id=str(proof_step.get("id", "")),
            )
        )
        rewrites.extend(
            _ensure_switch(
                proof_command,
                "--keep-scratch",
                before_flag="--file-list-entry",
                experiment_id=experiment_id,
                step_id=str(proof_step.get("id", "")),
            )
        )
        rewrites.extend(
            _ensure_flag(
                cache_command,
                "--preinflated-output-dir",
                right_out,
                before_flag="--inflate-timeout",
                experiment_id=experiment_id,
                step_id=str(cache_step.get("id", "")),
            )
        )
        rewrites.extend(
            _remove_flag_values(
                cache_command,
                "--local-acquisition-max-pairs",
                experiment_id=experiment_id,
                step_id=str(cache_step.get("id", "")),
            )
        )
        metadata = experiment.setdefault("metadata", {})
        if isinstance(metadata, MutableMapping):
            old_right_cache = metadata.get("receiver_patch_output_change_right_cache_dir")
            if old_right_cache != right_cache:
                metadata["receiver_patch_output_change_right_cache_dir"] = right_cache
                rewrites.append(
                    {
                        "experiment_id": experiment_id,
                        "step_id": "metadata",
                        "field": "receiver_patch_output_change_right_cache_dir",
                        "old_value": old_right_cache,
                        "new_value": right_cache,
                    }
                )
            old_preinflated = metadata.get("mlx_first_preinflated_receiver_output_dir")
            if old_preinflated != right_out:
                metadata["mlx_first_preinflated_receiver_output_dir"] = right_out
                rewrites.append(
                    {
                        "experiment_id": experiment_id,
                        "step_id": "metadata",
                        "field": "mlx_first_preinflated_receiver_output_dir",
                        "old_value": old_preinflated,
                        "new_value": right_out,
                    }
                )
    metadata = updated.setdefault("metadata", {})
    if isinstance(metadata, MutableMapping):
        migrations = metadata.setdefault("queue_migrations", [])
        if isinstance(migrations, list):
            migrations.append(
                {
                    "schema": "experiment_queue_mlx_receiver_cache_handoff_migration.v1",
                    "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reason": reason,
                    "changed_command_count": len(rewrites),
                    "rewrites": rewrites,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return updated


def batch_mlx_scorer_response_steps(
    queue: Mapping[str, Any],
    *,
    max_jobs_per_batch: int,
    reason: str,
) -> dict[str, Any]:
    """Amortize MLX scorer process/model setup across response steps.

    The original per-candidate MLX response step stays as the dependency target
    for downstream queue steps, but becomes a cheap JSON-contract validator that
    depends on a generated batch MLX step. The batch step writes the exact same
    response artifacts and remains false-authority.
    """

    if isinstance(max_jobs_per_batch, bool) or int(max_jobs_per_batch) < 2:
        raise ValueError("max_jobs_per_batch must be >= 2")
    updated = copy.deepcopy(dict(queue))
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    experiments = updated.get("experiments", [])
    for experiment in experiments:
        if not isinstance(experiment, MutableMapping):
            continue
        experiment_id = str(experiment.get("id", ""))
        steps = experiment.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, MutableMapping):
                continue
            command = step.get("command")
            if (
                str(step.get("id", "")) != "local_mlx_component_response"
                or not isinstance(command, list)
                or not _command_contains(command, "tools/run_mlx_scorer_response_from_cache.py")
            ):
                continue
            output = _flag_value(command, "--output")
            candidate_cache = _flag_value(command, "--candidate-cache-dir")
            archive = _flag_value(command, "--archive")
            archive_size_bytes = _flag_value(command, "--archive-size-bytes")
            if output is None or candidate_cache is None or (
                archive is None and archive_size_bytes is None
            ):
                continue
            key = _mlx_response_batch_group_key(command)
            groups.setdefault(key, []).append(
                {
                    "experiment": experiment,
                    "experiment_id": experiment_id,
                    "step": step,
                    "command": command,
                    "output": output,
                    "candidate_cache_dir": candidate_cache,
                    "archive": archive,
                    "archive_size_bytes": archive_size_bytes,
                    "components_dir": _flag_value(command, "--components-dir"),
                    "response_family": _flag_value(command, "--response-family"),
                }
            )

    batch_experiments: list[dict[str, Any]] = []
    rewrites: list[dict[str, Any]] = []
    batch_index = 0
    for key, jobs in sorted(groups.items(), key=lambda item: item[0]):
        if len(jobs) < 2:
            continue
        for offset in range(0, len(jobs), int(max_jobs_per_batch)):
            chunk = jobs[offset : offset + int(max_jobs_per_batch)]
            if len(chunk) < 2:
                continue
            batch_experiment_id = f"mlx_scorer_response_batch_{batch_index:04d}"
            batch_index += 1
            command = _mlx_response_batch_command(key=key, jobs=chunk, batch_id=batch_experiment_id)
            requires = sorted(
                {
                    _qualified_requirement(
                        str(job["experiment_id"]),
                        str(req),
                    )
                    for job in chunk
                    for req in (job["step"].get("requires") or [])
                }
            )
            postconditions = [
                {
                    "type": "json_equals",
                    "path": f"{_batch_summary_root(chunk[0]['output'])}/{batch_experiment_id}.json",
                    "key": "schema",
                    "equals": "mlx_scorer_response_batch_run.v1",
                },
                {
                    "type": "json_false_authority",
                    "path": f"{_batch_summary_root(chunk[0]['output'])}/{batch_experiment_id}.json",
                },
            ]
            for job in chunk:
                postconditions.extend(
                    [
                        {
                            "type": "json_equals",
                            "path": str(job["output"]),
                            "key": "schema_version",
                            "equals": "mlx_scorer_response.v1",
                        },
                        {
                            "type": "json_false_authority",
                            "path": str(job["output"]),
                        },
                    ]
                )
            batch_experiments.append(
                {
                    "id": batch_experiment_id,
                    "priority": 2,
                    "status": "queued",
                    "tags": [
                        "mlx-response-batch",
                        "local-mlx",
                        "false-authority",
                        "throughput-optimization",
                    ],
                    "metadata": {
                        "schema": "mlx_scorer_response_batch_experiment_metadata.v1",
                        "job_count": len(chunk),
                        "reason": reason,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    "steps": [
                        {
                            "id": "run_mlx_scorer_response_batch",
                            "kind": "command",
                            "command": command,
                            "requires": requires,
                            "resources": {"kind": "local_mlx"},
                            "timeout_seconds": 900 * len(chunk),
                            "postconditions": postconditions,
                        }
                    ],
                }
            )
            for job in chunk:
                step = job["step"]
                old_command = list(step.get("command") or [])
                old_requires = list(step.get("requires") or [])
                step["command"] = [
                    ".venv/bin/python",
                    JSON_ARTIFACT_VALIDATOR_TOOL,
                    "--path",
                    str(job["output"]),
                    "--schema-key",
                    "schema_version",
                    "--schema-equals",
                    "mlx_scorer_response.v1",
                    "--false-authority",
                ]
                step["resources"] = {"kind": "local_cpu"}
                step["requires"] = sorted(
                    set(old_requires + [f"{batch_experiment_id}.run_mlx_scorer_response_batch"])
                )
                rewrites.append(
                    {
                        "experiment_id": str(job["experiment_id"]),
                        "step_id": str(step.get("id", "")),
                        "field": "command",
                        "old_value": old_command,
                        "new_value": step["command"],
                    }
                )
                rewrites.append(
                    {
                        "experiment_id": str(job["experiment_id"]),
                        "step_id": str(step.get("id", "")),
                        "field": "requires",
                        "old_value": old_requires,
                        "new_value": step["requires"],
                    }
                )
    if batch_experiments and isinstance(experiments, list):
        experiments[0:0] = batch_experiments
    metadata = updated.setdefault("metadata", {})
    if isinstance(metadata, MutableMapping):
        migrations = metadata.setdefault("queue_migrations", [])
        if isinstance(migrations, list):
            migrations.append(
                {
                    "schema": "experiment_queue_mlx_response_batching_migration.v1",
                    "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reason": reason,
                    "batch_experiment_count": len(batch_experiments),
                    "changed_command_count": len(rewrites),
                    "max_jobs_per_batch": int(max_jobs_per_batch),
                    "rewrites": rewrites,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return updated


def _command_contains(command: Sequence[Any], needle: str) -> bool:
    return any(str(part) == needle for part in command)


def _command_contains_any(command: Sequence[Any], needles: frozenset[str]) -> bool:
    return any(str(part) in needles for part in command)


def _set_flag_value(
    command: list[Any],
    flag: str,
    value: str,
    *,
    experiment_id: str,
    step_id: str,
) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    for index, part in enumerate(command[:-1]):
        if str(part) != flag:
            continue
        old_value = str(command[index + 1])
        if old_value == value:
            continue
        command[index + 1] = value
        rewrites.append(
            {
                "experiment_id": experiment_id,
                "step_id": step_id,
                "flag": flag,
                "old_value": old_value,
                "new_value": value,
            }
        )
    return rewrites


def _flag_value(command: Sequence[Any], flag: str) -> str | None:
    for index, part in enumerate(command[:-1]):
        if str(part) == flag:
            return str(command[index + 1])
    return None


def _insert_index(command: Sequence[Any], before_flag: str | None) -> int:
    if before_flag is None:
        return len(command)
    for index, part in enumerate(command):
        if str(part) == before_flag:
            return index
    return len(command)


def _ensure_flag(
    command: list[Any],
    flag: str,
    value: str,
    *,
    before_flag: str | None,
    experiment_id: str,
    step_id: str,
) -> list[dict[str, Any]]:
    for index, part in enumerate(command[:-1]):
        if str(part) == flag:
            old_value = str(command[index + 1])
            if old_value == value:
                return []
            command[index + 1] = value
            return [
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "flag": flag,
                    "old_value": old_value,
                    "new_value": value,
                }
            ]
    index = _insert_index(command, before_flag)
    command[index:index] = [flag, value]
    return [
        {
            "experiment_id": experiment_id,
            "step_id": step_id,
            "flag": flag,
            "old_value": None,
            "new_value": value,
        }
    ]


def _ensure_switch(
    command: list[Any],
    flag: str,
    *,
    before_flag: str | None,
    experiment_id: str,
    step_id: str,
) -> list[dict[str, Any]]:
    if any(str(part) == flag for part in command):
        return []
    index = _insert_index(command, before_flag)
    command.insert(index, flag)
    return [
        {
            "experiment_id": experiment_id,
            "step_id": step_id,
            "flag": flag,
            "old_value": None,
            "new_value": True,
        }
    ]


def _remove_flag_values(
    command: list[Any],
    flag: str,
    *,
    experiment_id: str,
    step_id: str,
) -> list[dict[str, Any]]:
    rewrites: list[dict[str, Any]] = []
    index = 0
    while index < len(command):
        if str(command[index]) != flag:
            index += 1
            continue
        old_value = command[index + 1] if index + 1 < len(command) else None
        del command[index : min(index + 2, len(command))]
        rewrites.append(
            {
                "experiment_id": experiment_id,
                "step_id": step_id,
                "flag": flag,
                "old_value": old_value,
                "new_value": None,
            }
        )
    return rewrites


def _queue_root_right_cache(queue: Mapping[str, Any], output_dir: str) -> str:
    metadata = queue.get("metadata")
    root = metadata.get("output_root") if isinstance(metadata, Mapping) else None
    root_text = str(root or "").strip()
    if root_text:
        return f"{root_text.rstrip('/')}/_shell_inflate_right_cache"
    marker = "/frame1_region_waterfill_runtime_patch/full_frame_output_change_proof"
    if marker in output_dir:
        return output_dir.split(marker, 1)[0].rstrip("/") + "/_shell_inflate_right_cache"
    return output_dir.rstrip("/") + "/../_shell_inflate_right_cache"


def _mlx_response_batch_group_key(command: Sequence[Any]) -> tuple[str, ...]:
    flags = [
        "--reference-cache-dir",
        "--repo-root",
        "--batch-pairs",
        "--start-pair",
        "--max-pairs",
        "--device",
        "--progress-every",
    ]
    values: list[str] = []
    for flag in flags:
        values.extend([flag, _flag_value(command, flag) or ""])
    for switch in (
        "--allow-gpu-research-signal",
        "--allow-batch-shape-research-signal",
        "--allow-unaudited-candidate-cache-debug",
        "--allow-local-cpu-advisory-cache-identity",
    ):
        values.extend([switch, "1" if _command_contains(command, switch) else "0"])
    return tuple(values)


def _batch_summary_root(output: str) -> str:
    marker = "/frame1_region_waterfill_runtime_patch/local_component_spot_check/"
    if marker in output:
        return output.split(marker, 1)[0].rstrip("/") + "/mlx_response_batches"
    return output.rstrip("/") + ".batch"


def _mlx_response_batch_command(
    *,
    key: tuple[str, ...],
    jobs: Sequence[Mapping[str, Any]],
    batch_id: str,
) -> list[str]:
    key_map = {key[index]: key[index + 1] for index in range(0, len(key), 2)}
    first_output = str(jobs[0]["output"])
    summary_out = f"{_batch_summary_root(first_output)}/{batch_id}.json"
    command = [
        ".venv/bin/python",
        MLX_RESPONSE_BATCH_TOOL,
        "--reference-cache-dir",
        key_map["--reference-cache-dir"],
        "--summary-out",
        summary_out,
        "--repo-root",
        key_map["--repo-root"] or ".",
        "--batch-pairs",
        key_map["--batch-pairs"] or "1",
        "--device",
        key_map["--device"] or "gpu",
        "--overwrite",
    ]
    for flag in ("--start-pair", "--max-pairs", "--progress-every"):
        value = key_map.get(flag)
        if value:
            command.extend([flag, value])
    for switch in (
        "--allow-gpu-research-signal",
        "--allow-batch-shape-research-signal",
        "--allow-unaudited-candidate-cache-debug",
        "--allow-local-cpu-advisory-cache-identity",
    ):
        if key_map.get(switch) == "1":
            command.append(switch)
    for job in jobs:
        payload = {
            "candidate_cache_dir": str(job["candidate_cache_dir"]),
            "output": str(job["output"]),
        }
        if job.get("archive") is not None:
            payload["archive"] = str(job["archive"])
        if job.get("archive_size_bytes") is not None:
            payload["archive_size_bytes"] = int(str(job["archive_size_bytes"]))
        if job.get("components_dir") is not None:
            payload["components_dir"] = str(job["components_dir"])
        if job.get("response_family") is not None:
            payload["response_family"] = str(job["response_family"])
        command.extend(["--job-json", _json_dumps_compact(payload)])
    return command


def _qualified_requirement(experiment_id: str, requirement: str) -> str:
    return requirement if "." in requirement else f"{experiment_id}.{requirement}"


def _json_dumps_compact(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
