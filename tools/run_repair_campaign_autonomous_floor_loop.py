#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded autonomous floor loop for repair-family materialization artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.dqs1_materializer_feedback_bridge import (  # noqa: E402
    FALSE_AUTHORITY,
)
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_posterior import (  # noqa: E402
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    append_repair_campaign_blocked_learning_signal_report,
)
from tac.optimization.repair_family_byte_transform_executor import (  # noqa: E402
    REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
)
from tac.optimization.repair_family_exact_ready_bridge import (  # noqa: E402
    REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA,
    build_repair_family_exact_ready_bridge,
)
from tac.optimization.repair_family_stack_search import (  # noqa: E402
    REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
    REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA,
    REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
    RepairFamilyStackSearchError,
    build_repair_family_exact_handoff_plan,
    build_repair_family_stack_learning_signal_report,
    plan_repair_family_stack_search,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)

REPAIR_CAMPAIGN_AUTONOMOUS_FLOOR_LOOP_SCHEMA = "repair_campaign_autonomous_floor_loop.v1"
REPAIR_CAMPAIGN_AUTONOMOUS_FLOOR_LOOP_BLOCKER_REPORT_SCHEMA = "repair_campaign_autonomous_floor_loop_blocker_report.v1"


class RepairCampaignAutonomousFloorLoopError(ValueError):
    """Raised when the bounded autonomous floor loop cannot run."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-queue", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--posterior-path", type=Path)
    parser.add_argument(
        "--posterior-lock-path",
        type=Path,
        default=DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    )
    parser.add_argument("--byte-credit-budget", type=int)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--max-steps-per-iteration", type=int, default=32)
    parser.add_argument("--submission-dir", action="append", default=[], type=Path)
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairCampaignAutonomousFloorLoopError(f"{path} must contain a JSON object")
    return payload


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


def _run_worker(
    *,
    queue_path: Path,
    output_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "experiment_queue.py"),
        "--queue",
        str(queue_path),
        "run-worker",
        "--execute",
        "--max-steps",
        str(max_steps),
        "--max-experiments",
        "1",
        "--max-parallel",
        "1",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    return {
        "schema": "repair_campaign_floor_loop_worker_result.v1",
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "output_path": _repo_rel(output_path),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _discover_execution_reports(output_dir: Path) -> list[Path]:
    return sorted(
        path for path in output_dir.rglob("repair_family_byte_transform_execution_report.json") if path.is_file()
    )


def _load_execution_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        if payload.get("schema") != REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA:
            continue
        require_no_truthy_authority_fields(
            payload,
            context=f"autonomous_floor_loop_execution_report:{path}",
        )
        reports.append(payload)
    return reports


def _iteration_stop_reason(stack_plan: dict[str, Any], worker_result: dict[str, Any] | None) -> str:
    if worker_result is not None and worker_result.get("returncode") not in (0, None):
        return "exact_axis_blocker_or_local_worker_failure"
    primary_path = stack_plan.get("primary_stack_acquisition_path")
    terminal_outcome = str(primary_path.get("terminal_outcome_class") or "") if isinstance(primary_path, dict) else ""
    if terminal_outcome in {
        "strictly_better_archive_bound_candidate_exact_axis_blocked",
        "precise_exact_axis_blocker",
        "family_demoted_by_posterior_evidence",
    }:
        return terminal_outcome
    if stack_plan.get("candidate_improvement_observed") is True:
        return "candidate_improvement_observed"
    return "exact_axis_blocker"


def _precise_blocker_stop_reason(summary: dict[str, Any]) -> str:
    stack_plan = summary["stack_search_plan"]
    decision = stack_plan.get("budget_routing_decision") or {}
    route = str(decision.get("activation_action") or "")
    if any(
        _iteration.get("stop_reason") == "exact_axis_blocker_or_local_worker_failure"
        for _iteration in summary.get("iterations") or []
        if isinstance(_iteration, dict)
    ):
        return "local_worker_failure_or_exact_axis_blocker"
    primary_path = stack_plan.get("primary_stack_acquisition_path")
    terminal_outcome = str(primary_path.get("terminal_outcome_class") or "") if isinstance(primary_path, dict) else ""
    if terminal_outcome in {
        "strictly_better_archive_bound_candidate_exact_axis_blocked",
        "family_demoted_by_posterior_evidence",
    }:
        return terminal_outcome
    if int(summary.get("archive_bound_exact_handoff_candidate_count") or 0) > 0:
        return "archive_bound_candidate_exact_axis_blocked"
    if route == "demote_repair_family_until_new_component_signal":
        return "family_demoted_by_posterior_evidence"
    if int(stack_plan.get("execution_report_count") or 0) == 0:
        return "materialization_execution_reports_missing"
    if stack_plan.get("candidate_improvement_observed") is True:
        return "local_mlx_candidate_improvement_observed_exact_axis_blocked"
    if route:
        return f"{route}_blocked"
    return "exact_axis_blocker"


def _build_blocker_report(summary: dict[str, Any]) -> dict[str, Any]:
    stack_plan = summary["stack_search_plan"]
    bridge_report = summary["exact_ready_bridge_report"]
    decision = stack_plan.get("budget_routing_decision") or {}
    stop_reason = _precise_blocker_stop_reason(summary)
    selected_blocker = str(decision.get("selected_blocker_class") or "").strip()
    activation_action = str(decision.get("activation_action") or "").strip()
    priority_score = decision.get("priority_score")
    if not selected_blocker and int(stack_plan.get("execution_report_count") or 0) == 0:
        selected_blocker = "repair_family_byte_transform_execution_reports_missing"
        activation_action = "materialize_repair_family_byte_transform_reports"
        priority_score = 100
    blockers = ordered_unique(
        [
            *_string_list(summary.get("blockers")),
            *_string_list(stack_plan.get("exact_eval_handoff_gate", {}).get("blockers")),
            *_string_list(bridge_report.get("blockers")),
            selected_blocker,
        ]
    )
    report = {
        "schema": REPAIR_CAMPAIGN_AUTONOMOUS_FLOOR_LOOP_BLOCKER_REPORT_SCHEMA,
        "materialization_queue_path": summary["materialization_queue_path"],
        "stop_reason": stop_reason,
        "selected_blocker_class": selected_blocker,
        "activation_action": activation_action,
        "priority_score": priority_score,
        "execution_report_count": stack_plan.get("execution_report_count"),
        "exact_eval_handoff_candidate_count": summary["exact_eval_handoff_candidate_count"],
        "archive_bound_exact_handoff_candidate_count": summary["archive_bound_exact_handoff_candidate_count"],
        "exact_ready_bridge_candidate_count": summary["exact_ready_bridge_candidate_count"],
        "exact_ready_bridge_runtime_content_tree_custody_proven_count": summary[
            "exact_ready_bridge_runtime_content_tree_custody_proven_count"
        ],
        "posterior_learning_signal_count": summary["posterior_learning_signal_count"],
        "posterior_appended_count": summary["posterior_appended_count"],
        "blockers": [blocker for blocker in blockers if blocker],
        "required_next_authority": "contest_cpu_or_cuda_auth_axis_payload",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "repair_floor_loop_precise_blocker_and_routing_audit",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_autonomous_floor_loop_blocker_report",
    )
    return report


def _build_summary(
    *,
    queue_path: Path,
    output_dir: Path,
    posterior_path: Path | None,
    byte_credit_budget: int | None,
    max_iterations: int,
    max_steps_per_iteration: int,
    execute_local: bool,
    submission_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    queue = _load_json(queue_path)
    require_no_truthy_authority_fields(queue, context="autonomous_floor_loop_queue")
    iterations: list[dict[str, Any]] = []
    final_stack_plan: dict[str, Any] = {
        "schema": REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
        "execution_report_count": 0,
        "stack_rows": [],
        "candidate_improvement_observed": False,
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    for index in range(1, max(1, max_iterations) + 1):
        worker_result = None
        if execute_local:
            worker_result = _run_worker(
                queue_path=queue_path,
                output_path=output_dir / f"iteration_{index}_worker_result.json",
                max_steps=max_steps_per_iteration,
            )
        report_paths = _discover_execution_reports(output_dir)
        reports = _load_execution_reports(report_paths)
        if reports:
            final_stack_plan = plan_repair_family_stack_search(
                execution_reports=reports,
                execution_report_paths=tuple(_repo_rel(path) for path in report_paths),
                repo_root=REPO_ROOT,
                posterior_path=posterior_path,
                byte_credit_budget=byte_credit_budget,
            )
        stop_reason = _iteration_stop_reason(final_stack_plan, worker_result)
        iterations.append(
            {
                "schema": "repair_campaign_autonomous_floor_loop_iteration.v1",
                "iteration": index,
                "execute_local": execute_local,
                "worker_result": worker_result,
                "execution_report_count": len(reports),
                "execution_report_paths": [_repo_rel(path) for path in report_paths],
                "stack_plan_schema": final_stack_plan.get("schema"),
                "candidate_improvement_observed": final_stack_plan.get("candidate_improvement_observed") is True,
                "stop_reason": stop_reason,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
        )
        if stop_reason in {
            "candidate_improvement_observed",
            "exact_axis_blocker_or_local_worker_failure",
            "strictly_better_archive_bound_candidate_exact_axis_blocked",
            "precise_exact_axis_blocker",
            "family_demoted_by_posterior_evidence",
        }:
            break
    stack_plan_path = output_dir / "repair_family_stack_search_plan.json"
    exact_handoff_plan_path = output_dir / "repair_family_exact_handoff_plan.json"
    exact_handoff_plan = build_repair_family_exact_handoff_plan(
        stack_plan=final_stack_plan,
        stack_plan_path=_repo_rel(stack_plan_path),
    )
    exact_ready_bridge = build_repair_family_exact_ready_bridge(
        exact_handoff_plan=exact_handoff_plan,
        exact_handoff_plan_path=_repo_rel(exact_handoff_plan_path),
        submission_dirs=tuple(submission_dirs),
        repo_root=REPO_ROOT,
    )
    exact_ready_bridge_report = exact_ready_bridge["bridge_report"]
    exact_ready_source_queue_path = output_dir / "repair_family_exact_ready_source_queue.json"
    blocked_exact_ready_queue_path = output_dir / "repair_family_blocked_exact_ready_queue.json"
    exact_ready_bridge_report_path = output_dir / "repair_family_exact_ready_bridge_report.json"
    learning_signal_report = build_repair_family_stack_learning_signal_report(
        stack_plan=final_stack_plan,
        bridge_report=exact_ready_bridge_report,
    )
    learning_signal_report_path = output_dir / "repair_family_stack_learning_signal_report.json"
    primary_path = final_stack_plan.get("primary_stack_acquisition_path")
    primary_terminal_outcome = (
        str(primary_path.get("terminal_outcome_class") or "") if isinstance(primary_path, dict) else None
    )
    summary = {
        "schema": REPAIR_CAMPAIGN_AUTONOMOUS_FLOOR_LOOP_SCHEMA,
        "materialization_queue_path": _repo_rel(queue_path),
        "materialization_queue_schema": queue.get("schema"),
        "output_dir": _repo_rel(output_dir),
        "posterior_path": None if posterior_path is None else str(posterior_path),
        "byte_credit_budget": byte_credit_budget,
        "max_iterations": max_iterations,
        "max_steps_per_iteration": max_steps_per_iteration,
        "execute_local": execute_local,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "stack_search_plan_path": _repo_rel(stack_plan_path),
        "stack_search_plan": final_stack_plan,
        "primary_stack_acquisition_terminal_outcome": primary_terminal_outcome,
        "primary_stack_acquisition_path": primary_path,
        "exact_handoff_plan_path": _repo_rel(exact_handoff_plan_path),
        "exact_handoff_plan_schema": REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
        "exact_handoff_plan": exact_handoff_plan,
        "exact_eval_handoff_candidate_count": exact_handoff_plan["candidate_count"],
        "archive_bound_exact_handoff_candidate_count": exact_handoff_plan["archive_bound_candidate_count"],
        "exact_ready_bridge_source_queue_path": _repo_rel(exact_ready_source_queue_path),
        "blocked_exact_ready_queue_path": _repo_rel(blocked_exact_ready_queue_path),
        "exact_ready_bridge_report_path": _repo_rel(exact_ready_bridge_report_path),
        "exact_ready_bridge_report_schema": REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA,
        "exact_ready_bridge_report": exact_ready_bridge_report,
        "exact_ready_bridge_candidate_count": exact_ready_bridge_report["candidate_count"],
        "exact_ready_bridge_archive_custody_proven_count": exact_ready_bridge_report["archive_custody_proven_count"],
        "exact_ready_bridge_runtime_content_tree_custody_proven_count": (
            exact_ready_bridge_report["runtime_content_tree_custody_proven_count"]
        ),
        "exact_ready_bridge": exact_ready_bridge,
        "posterior_learning_signal_report_path": _repo_rel(learning_signal_report_path),
        "posterior_learning_signal_report_schema": (REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA),
        "posterior_learning_signal_count": learning_signal_report["learning_signal_count"],
        "posterior_learning_signal_report": learning_signal_report,
        "posterior_append_report": None,
        "posterior_appended_count": 0,
        "posterior_skipped_duplicate_count": 0,
        "stop_reason": iterations[-1]["stop_reason"] if iterations else "exact_axis_blocker",
        "autonomous_loop_closed": True,
        "loop_contract": [
            "planner_reads_queue_and_posterior",
            "local_worker_materializes_advisory_byte_transform_when_execute_local",
            "stack_search_routes_negative_results_and_byte_credit",
            "exact_eval_handoff_fails_closed_without_contest_cpu_or_cuda_axis",
            "exact_ready_bridge_emits_source_queue_and_blocked_exact_ready_queue",
            "stack_and_bridge_outcomes_emit_posterior_learning_signals",
            "precise_blocker_report_names_next_unblocked_action",
        ],
        "blockers": ordered_unique(
            [
                "contest_cpu_or_cuda_exact_axis_payload_required_before_score",
                "lane_dispatch_claim_required_before_exact_eval",
                *(
                    []
                    if final_stack_plan.get("execution_report_count")
                    else ["repair_family_byte_transform_execution_reports_missing"]
                ),
            ]
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "bounded_autonomous_repair_floor_loop_local_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        summary,
        context="repair_campaign_autonomous_floor_loop_summary",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_dir = _resolve(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = _build_summary(
            queue_path=_resolve(args.materialization_queue),
            output_dir=output_dir,
            posterior_path=args.posterior_path,
            byte_credit_budget=args.byte_credit_budget,
            max_iterations=args.max_iterations,
            max_steps_per_iteration=args.max_steps_per_iteration,
            execute_local=bool(args.execute_local),
            submission_dirs=tuple(args.submission_dir),
        )
        stack_plan_path = output_dir / "repair_family_stack_search_plan.json"
        expected_stack_sha = sha256_file(stack_plan_path) if stack_plan_path.exists() and args.overwrite else None
        write_json_artifact(
            stack_plan_path,
            summary["stack_search_plan"],
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_stack_sha,
        )
        exact_handoff_plan_path = output_dir / "repair_family_exact_handoff_plan.json"
        expected_exact_handoff_sha = (
            sha256_file(exact_handoff_plan_path) if exact_handoff_plan_path.exists() and args.overwrite else None
        )
        write_json_artifact(
            exact_handoff_plan_path,
            summary["exact_handoff_plan"],
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_exact_handoff_sha,
        )
        exact_ready_source_queue_path = output_dir / "repair_family_exact_ready_source_queue.json"
        blocked_exact_ready_queue_path = output_dir / "repair_family_blocked_exact_ready_queue.json"
        exact_ready_bridge_report_path = output_dir / "repair_family_exact_ready_bridge_report.json"
        for path, payload in (
            (
                exact_ready_source_queue_path,
                summary["exact_ready_bridge"]["source_optimizer_queue"],
            ),
            (
                blocked_exact_ready_queue_path,
                summary["exact_ready_bridge"]["blocked_exact_ready_queue"],
            ),
            (
                exact_ready_bridge_report_path,
                summary["exact_ready_bridge_report"],
            ),
        ):
            expected_bridge_sha = sha256_file(path) if path.exists() and args.overwrite else None
            write_json_artifact(
                path,
                payload,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_bridge_sha,
            )
        learning_signal_report_path = output_dir / "repair_family_stack_learning_signal_report.json"
        expected_learning_signal_sha = (
            sha256_file(learning_signal_report_path)
            if learning_signal_report_path.exists() and args.overwrite
            else None
        )
        write_json_artifact(
            learning_signal_report_path,
            summary["posterior_learning_signal_report"],
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_learning_signal_sha,
        )
        if args.posterior_path is not None:
            posterior_append_report = append_repair_campaign_blocked_learning_signal_report(
                blocked_learning_signal_report_path=_repo_rel(learning_signal_report_path),
                blocked_learning_signal_report=summary["posterior_learning_signal_report"],
                posterior_path=args.posterior_path,
                lock_path=args.posterior_lock_path,
                repo_root=REPO_ROOT,
            )
            summary["posterior_append_report"] = posterior_append_report
            summary["posterior_appended_count"] = posterior_append_report["appended_count"]
            summary["posterior_skipped_duplicate_count"] = posterior_append_report["skipped_duplicate_count"]
            require_no_truthy_authority_fields(
                summary,
                context="repair_campaign_autonomous_floor_loop_summary_after_posterior",
            )
        blocker_report = _build_blocker_report(summary)
        blocker_report_path = output_dir / "repair_family_floor_loop_blocker_report.json"
        expected_blocker_sha = (
            sha256_file(blocker_report_path) if blocker_report_path.exists() and args.overwrite else None
        )
        write_json_artifact(
            blocker_report_path,
            blocker_report,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_blocker_sha,
        )
        summary["exact_axis_blocker_report_path"] = _repo_rel(blocker_report_path)
        summary["exact_axis_blocker_report_schema"] = blocker_report["schema"]
        summary["exact_axis_blocker_report"] = blocker_report
        summary["exact_axis_blocker_class"] = blocker_report["selected_blocker_class"]
        summary["stop_reason"] = blocker_report["stop_reason"]
        require_no_truthy_authority_fields(
            summary,
            context="repair_campaign_autonomous_floor_loop_summary_after_blocker",
        )
        summary_out = _resolve(args.summary_out)
        expected_summary_sha = sha256_file(summary_out) if summary_out.exists() and args.overwrite else None
        write_result = write_json_artifact(
            summary_out,
            summary,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_summary_sha,
        )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignAutonomousFloorLoopError,
        RepairFamilyStackSearchError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"FATAL: repair campaign autonomous floor loop failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_autonomous_floor_loop_cli_result.v1",
                "summary_out": str(args.summary_out),
                "stop_reason": summary["stop_reason"],
                "primary_stack_acquisition_terminal_outcome": summary["primary_stack_acquisition_terminal_outcome"],
                "iteration_count": summary["iteration_count"],
                "execution_report_count": summary["stack_search_plan"]["execution_report_count"],
                "exact_eval_handoff_candidate_count": summary["stack_search_plan"].get(
                    "exact_eval_handoff_candidate_count",
                    0,
                ),
                "archive_bound_exact_handoff_candidate_count": (
                    summary["stack_search_plan"].get(
                        "archive_bound_exact_handoff_candidate_count",
                        0,
                    )
                ),
                "exact_ready_bridge_candidate_count": summary["exact_ready_bridge_candidate_count"],
                "exact_ready_bridge_runtime_content_tree_custody_proven_count": (
                    summary["exact_ready_bridge_runtime_content_tree_custody_proven_count"]
                ),
                "posterior_learning_signal_count": summary["posterior_learning_signal_count"],
                "posterior_appended_count": summary["posterior_appended_count"],
                "exact_axis_blocker_report_path": summary["exact_axis_blocker_report_path"],
                "candidate_improvement_observed": summary["stack_search_plan"]["candidate_improvement_observed"],
                "bytes_written": write_result.bytes_written,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
