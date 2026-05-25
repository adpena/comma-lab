#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize local queue-health recovery artifacts for a byte-shaving campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import (  # noqa: E402
    default_state_path,
    load_queue_definition,
)
from comma_lab.scheduler.experiment_queue_observer import observe_experiment_queue  # noqa: E402
from comma_lab.scheduler.queue_feedback_replan_policy import (  # noqa: E402
    build_queue_feedback_replan_policy,
    build_queue_observation_recovery_plan,
    build_queue_observation_recovery_queue,
)
from tac.authority_contract import apply_false_authority_contract  # noqa: E402
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    read_json,
    write_json_artifact,
)

RECOVERY_SUMMARY_SCHEMA = "byte_shaving_materializer_campaign_queue_recovery_materialization.v1"


def _repo_rel(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(path_ref: object, *, repo_root: Path) -> Path | None:
    if not isinstance(path_ref, str) or not path_ref.strip():
        return None
    path = Path(path_ref)
    return path if path.is_absolute() else repo_root / path


def _run_dir_from_summary(summary_path: Path, payload: dict[str, Any], *, repo_root: Path) -> Path:
    run_dir = _resolve(payload.get("run_dir"), repo_root=repo_root)
    return run_dir if run_dir is not None else summary_path.parent


def _load_optional_json(path_ref: object, *, repo_root: Path) -> dict[str, Any] | None:
    path = _resolve(path_ref, repo_root=repo_root)
    if path is None or not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _write_artifact(
    path: Path,
    payload: dict[str, Any],
    *,
    write: bool,
    overwrite: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "write_requested": write,
        "exists_before": path.exists(),
        "written": False,
    }
    if not write:
        return record
    result = write_json_artifact(path, payload, allow_overwrite=overwrite)
    record.update(
        {
            "written": True,
            "bytes": result.bytes_written,
            "sha256": result.sha256,
            "allow_overwrite": result.allow_overwrite,
        }
    )
    return record


def build_recovery_materialization(
    *,
    repo_root: Path,
    run_summary_path: Path,
    lane_id: str | None,
    write: bool,
    overwrite: bool,
) -> dict[str, Any]:
    run_summary_path = run_summary_path if run_summary_path.is_absolute() else repo_root / run_summary_path
    run_summary = read_json(run_summary_path)
    if not isinstance(run_summary, dict):
        raise ValueError("run summary must be a JSON object")
    if run_summary.get("schema") != "byte_shaving_materializer_campaign_run.v1":
        raise ValueError("run summary is not byte_shaving_materializer_campaign_run.v1")

    queue_path = _resolve(run_summary.get("queue_path"), repo_root=repo_root)
    if queue_path is None or not queue_path.is_file():
        raise ValueError("run summary queue_path is missing or not a file")
    queue = load_queue_definition(queue_path)
    state_path = _resolve(run_summary.get("state_path"), repo_root=repo_root)
    if state_path is None:
        state_path = default_state_path(repo_root, str(queue["queue_id"]))

    run_dir = _run_dir_from_summary(run_summary_path, run_summary, repo_root=repo_root)
    observation_path = run_dir / "queue_observation.json"
    recovery_plan_path = run_dir / "queue_observation_recovery_plan.json"
    feedback_policy_path = run_dir / "queue_feedback_replan_policy.json"
    recovery_queue_path = run_dir / "queue_observation_recovery_queue.json"
    recovery_state_path = run_dir / "queue_observation_recovery_queue.sqlite"
    materialization_summary_path = run_dir / "queue_observation_recovery_materialization.json"

    observation = observe_experiment_queue(
        queue,
        state_path=state_path,
        repo_root=repo_root,
        tail_lines=0,
        include_orphans=True,
    )
    recovery_plan = build_queue_observation_recovery_plan(
        observation,
        queue_path=_repo_rel(queue_path, repo_root=repo_root),
        state_path=_repo_rel(state_path, repo_root=repo_root),
        reason="byte-shaving campaign live queue recovery materialization",
    )
    policy_input = dict(run_summary)
    policy_input.update(
        {
            "observation": observation,
            "queue_observation_path": _repo_rel(observation_path, repo_root=repo_root),
            "queue_observation_recovery_plan_path": _repo_rel(
                recovery_plan_path,
                repo_root=repo_root,
            ),
            "queue_observation_recovery_plan": recovery_plan,
            "queue_observation_recovery_required": recovery_plan.get(
                "recovery_required"
            )
            is True,
            "queue_observation_maintenance_recommended": recovery_plan.get(
                "maintenance_recommended"
            )
            is True,
            "queue_feedback_replan_policy_path": _repo_rel(
                feedback_policy_path,
                repo_root=repo_root,
            ),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    followup_queue = _load_optional_json(
        run_summary.get("queue_feedback_replan_followup_queue_path"),
        repo_root=repo_root,
    )
    policy = build_queue_feedback_replan_policy(
        policy_input,
        feedback_followup_queue=followup_queue,
        source_run_path=_repo_rel(run_summary_path, repo_root=repo_root),
    )
    effective_lane_id = lane_id or f"{queue['queue_id']}_live_queue_recovery"
    recovery_queue, recovery_queue_blockers = build_queue_observation_recovery_queue(
        policy,
        lane_id=effective_lane_id,
        source_policy_path=_repo_rel(feedback_policy_path, repo_root=repo_root),
    )

    artifact_records: list[dict[str, Any]] = []
    artifact_records.append(
        _write_artifact(
            observation_path,
            observation,
            write=write,
            overwrite=overwrite,
        )
    )
    artifact_records.append(
        _write_artifact(
            recovery_plan_path,
            recovery_plan,
            write=write,
            overwrite=overwrite,
        )
    )
    artifact_records.append(
        _write_artifact(
            feedback_policy_path,
            policy,
            write=write,
            overwrite=overwrite,
        )
    )
    if recovery_queue is not None:
        artifact_records.append(
            _write_artifact(
                recovery_queue_path,
                recovery_queue,
                write=write,
                overwrite=overwrite,
            )
        )

    payload = apply_false_authority_contract(
        {
            "schema": RECOVERY_SUMMARY_SCHEMA,
            "run_summary_path": _repo_rel(run_summary_path, repo_root=repo_root),
            "run_dir": _repo_rel(run_dir, repo_root=repo_root),
            "queue_id": str(queue["queue_id"]),
            "queue_path": _repo_rel(queue_path, repo_root=repo_root),
            "state_path": _repo_rel(state_path, repo_root=repo_root),
            "lane_id": effective_lane_id,
            "write_requested": write,
            "overwrite": overwrite,
            "queue_observation_path": _repo_rel(observation_path, repo_root=repo_root),
            "queue_observation_healthy": observation.get("healthy") is True,
            "queue_observation_blockers": list(observation.get("blockers") or []),
            "queue_observation_failed_step_count": len(
                list(observation.get("failed_steps") or [])
            ),
            "queue_observation_ready_step_count": len(
                list(observation.get("ready_steps") or [])
            ),
            "queue_observation_recovery_plan_path": _repo_rel(
                recovery_plan_path,
                repo_root=repo_root,
            ),
            "queue_observation_recovery_required": recovery_plan.get(
                "recovery_required"
            )
            is True,
            "queue_observation_recovery_action_count": recovery_plan.get(
                "action_count"
            ),
            "queue_feedback_replan_policy_path": _repo_rel(
                feedback_policy_path,
                repo_root=repo_root,
            ),
            "queue_feedback_replan_policy_decision": policy.get("decision"),
            "ready_for_queue_health_recovery": policy.get(
                "ready_for_queue_health_recovery"
            )
            is True,
            "queue_observation_recovery_queue_path": _repo_rel(
                recovery_queue_path,
                repo_root=repo_root,
            ),
            "queue_observation_recovery_queue_state_path": _repo_rel(
                recovery_state_path,
                repo_root=repo_root,
            ),
            "queue_observation_recovery_queue_emitted": recovery_queue is not None,
            "queue_observation_recovery_queue_blockers": recovery_queue_blockers,
            "artifact_records": artifact_records,
            "allowed_use": "local_queue_health_recovery_materialization_only",
            "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        },
        preserve_dispatch_ready=False,
        reason="byte_shaving_queue_recovery_materialization_no_score_authority",
    )
    artifact_records.append(
        _write_artifact(
            materialization_summary_path,
            payload,
            write=write,
            overwrite=overwrite,
        )
    )
    payload["materialization_summary_path"] = _repo_rel(
        materialization_summary_path,
        repo_root=repo_root,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--lane-id")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write recovery artifacts. Without this, print a plan only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = build_recovery_materialization(
            repo_root=args.repo_root.resolve(),
            run_summary_path=args.run_summary,
            lane_id=args.lane_id,
            write=args.write,
            overwrite=args.overwrite,
        )
    except (ArtifactWriteError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
