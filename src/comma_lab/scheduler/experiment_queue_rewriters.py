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
