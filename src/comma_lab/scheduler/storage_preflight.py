# SPDX-License-Identifier: MIT
"""Reusable storage and cleanup preflight for experiment queues."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB, DEFAULT_TIERS


def _false_authority_postcondition(path: str) -> dict[str, Any]:
    return {
        "type": "json_false_authority",
        "path": path,
        "required_false": ["score_claim", "promotion_eligible", "rank_or_kill_eligible"],
        "false_or_missing": [
            "ready_for_exact_eval_dispatch",
            "dispatch_attempted",
            "gpu_launched",
        ],
    }


def _results_root_storage_workload_subdir(results_root: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    path = Path(results_root).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    for _name, root in DEFAULT_TIERS:
        try:
            return path.resolve(strict=False).relative_to(
                Path(root).resolve(strict=False)
            ).as_posix()
        except ValueError:
            continue
    return path.name


def _expected_results_root(results_root: str) -> str:
    path = Path(results_root).expanduser()
    if not path.is_absolute():
        return str(path)
    return str(path.resolve(strict=False))


def build_scheduler_storage_preflight_experiment(
    *,
    experiment_id: str,
    lane_id: str,
    tags: list[str],
    artifact_prefix: str,
    date: str,
    results_root: str,
    storage_tiers: tuple[str, ...] = (),
    storage_workload_subdir: str | None = None,
    storage_expected_workload_root: str | None = None,
    storage_reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    storage_expected_bytes: int = 0,
    proactive_cleanup_roots: tuple[str, ...] = (),
    proactive_cleanup_execute: bool = False,
    proactive_cleanup_action: str = "move",
    proactive_cleanup_min_bytes: str = "1",
    proactive_cleanup_cold_store_roots: tuple[str, ...] = (),
    proactive_cleanup_cold_store_reserve_gb: float = DEFAULT_RESERVE_FREE_GB,
) -> dict[str, Any]:
    """Return a queue experiment that gates work on storage and cleanup.

    The returned experiment is false-authority plumbing: it plans the storage
    tier, optionally moves/deletes certified rebuildable artifacts, and exposes
    postconditions so downstream work can depend on the preflight succeeding.
    """

    storage_plan = f".omx/research/{artifact_prefix}_storage_plan_{date}.json"
    cleanup_plan = f".omx/research/{artifact_prefix}_proactive_cleanup_{date}.json"
    workload_subdir = _results_root_storage_workload_subdir(
        results_root,
        storage_workload_subdir,
    )
    expected_root = (
        storage_expected_workload_root
        if storage_expected_workload_root is not None
        else (
            _expected_results_root(results_root)
            if Path(results_root).expanduser().is_absolute()
            else None
        )
    )
    storage_command = [
        ".venv/bin/python",
        "tools/plan_experiment_storage.py",
        "--output",
        storage_plan,
        "--workload-subdir",
        workload_subdir,
        "--reserve-free-gb",
        str(storage_reserve_free_gb),
        "--requested-bytes",
        str(storage_expected_bytes),
        "--create",
    ]
    if expected_root is not None:
        storage_command.extend(["--expected-workload-root", expected_root])
    for spec in storage_tiers:
        storage_command.extend(["--storage-tier", spec])

    cleanup_roots = proactive_cleanup_roots or (
        "experiments/results",
        ".omx/tmp",
        "submissions/robust_current/eval_runs",
    )
    cleanup_command = [
        ".venv/bin/python",
        "tools/compact_experiment_artifacts.py",
        *cleanup_roots,
        "--min-bytes",
        str(proactive_cleanup_min_bytes),
        "--json-output",
        cleanup_plan,
    ]
    if proactive_cleanup_execute:
        cleanup_command.extend(["--execute", "--action", proactive_cleanup_action])
        cleanup_command.extend(
            ["--cold-store-reserve-gb", str(proactive_cleanup_cold_store_reserve_gb)]
        )
        for cold_store_root in proactive_cleanup_cold_store_roots:
            cleanup_command.extend(["--cold-store-root", cold_store_root])

    return {
        "id": experiment_id,
        "priority": 0,
        "lane_id": lane_id,
        "tags": tags,
        "steps": [
            {
                "id": "storage_tier_plan",
                "timeout_seconds": 120,
                "command": storage_command,
                "resources": {"kind": "local_cpu"},
                "postconditions": [
                    {
                        "type": "json_equals",
                        "path": storage_plan,
                        "key": "selected_workload_root_matches_expected",
                        "equals": True,
                    },
                    _false_authority_postcondition(storage_plan),
                ],
            },
            {
                "id": "proactive_cleanup",
                "requires": ["storage_tier_plan"],
                "timeout_seconds": 1200,
                "command": cleanup_command,
                "resources": {"kind": "local_io_heavy"},
                "postconditions": [
                    {
                        "type": "json_false_authority",
                        "path": cleanup_plan,
                        "required_false": [
                            "plan.score_claim",
                            "plan.promotion_eligible",
                            "plan.ready_for_exact_eval_dispatch",
                        ],
                        "false_or_missing": [],
                    }
                ],
            },
        ],
    }
