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
from tac.optimization.repair_entropy_stage_chain_executor import (  # noqa: E402
    REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA,
    build_repair_entropy_stage_chain_execution_bundle,
)
from tac.optimization.repair_family_byte_transform_executor import (  # noqa: E402
    REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
    SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES,
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
from tools.check_exact_dispatch_provider_preclaim import (  # noqa: E402
    build_preclaim_check,
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
    parser.add_argument("--worker-max-experiments-per-iteration", type=int)
    parser.add_argument("--require-family-id", action="append", default=[])
    parser.add_argument("--require-all-queue-families", action="store_true")
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


def _slug(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "unknown"


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _family_from_experiment_metadata(metadata: dict[str, Any]) -> str:
    family = str(metadata.get("family_id") or "").strip()
    candidate = str(metadata.get("candidate_id") or "").strip()
    if (not family or family == "unclassified_repair_family") and candidate in SUPPORTED_REPAIR_BYTE_TRANSFORM_FAMILIES:
        return candidate
    return family


def _queue_family_ids(queue: dict[str, Any]) -> list[str]:
    families: list[str] = []
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        metadata = experiment.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if metadata.get("queue_actuation_ready") is not True:
            continue
        family = _family_from_experiment_metadata(metadata)
        if family:
            families.append(family)
    return ordered_unique(families)


def _execution_family_ids(reports: Sequence[dict[str, Any]]) -> list[str]:
    return ordered_unique(
        str(report.get("family_id") or "").strip()
        for report in reports
        if str(report.get("family_id") or "").strip()
    )


def _family_coverage_report(
    *,
    required_family_ids: Sequence[str],
    queue_family_ids: Sequence[str],
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    executed = _execution_family_ids(reports)
    required = ordered_unique(str(family).strip() for family in required_family_ids if str(family).strip())
    missing = [family for family in required if family not in set(executed)]
    report = {
        "schema": "repair_campaign_floor_loop_family_coverage.v1",
        "required_family_ids": required,
        "required_family_count": len(required),
        "queue_family_ids": ordered_unique(queue_family_ids),
        "queue_family_count": len(ordered_unique(queue_family_ids)),
        "executed_family_ids": executed,
        "executed_family_count": len(executed),
        "missing_required_family_ids": missing,
        "missing_required_family_count": len(missing),
        "coverage_satisfied": not missing,
        "blockers": (
            []
            if not missing
            else [
                "required_repair_family_coverage_incomplete",
                *[f"required_repair_family_not_executed:{family}" for family in missing],
            ]
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_floor_loop_family_coverage",
    )
    return report


def _run_worker(
    *,
    queue_path: Path,
    output_path: Path,
    max_steps: int,
    max_experiments: int,
) -> dict[str, Any]:
    state_path = output_path.parent / "repair_campaign_floor_loop_queue_state.sqlite"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "experiment_queue.py"),
        "--queue",
        str(queue_path),
        "--state",
        str(state_path),
        "run-worker",
        "--noncanonical-state-rationale",
        "repair_campaign_autonomous_floor_loop_uses_isolated_replay_state_to_avoid_shared_queue_collision",
        "--execute",
        "--max-steps",
        str(max_steps),
        "--max-experiments",
        str(max(1, max_experiments)),
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
        "state_path": _repo_rel(state_path),
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }


def _discover_execution_reports(output_dir: Path, queue: dict[str, Any]) -> list[Path]:
    paths = [
        path
        for path in output_dir.rglob("repair_family_byte_transform_execution_report.json")
        if path.is_file()
    ]
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        metadata = experiment.get("metadata")
        if not isinstance(metadata, dict):
            continue
        path_text = str(metadata.get("repair_family_byte_transform_execution_report_path") or "").strip()
        if not path_text:
            for step in experiment.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                command = step.get("command")
                if not isinstance(command, list):
                    continue
                for index, item in enumerate(command):
                    text = str(item)
                    if text == "--execution-report-out" and index + 1 < len(command):
                        candidate = Path(str(command[index + 1]))
                        if candidate.name == "repair_family_byte_transform_execution_report.json":
                            path_text = str(candidate)
                            break
                    if text.endswith("/repair_family_byte_transform_execution_report.json"):
                        path_text = text
                        break
                if path_text:
                    break
        if path_text:
            path = _resolve(Path(path_text))
            if path.is_file():
                paths.append(path)
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve(strict=False))] = path
    return sorted(unique.values())


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


def _metadata_identity(metadata: dict[str, Any]) -> dict[str, str]:
    family = _family_from_experiment_metadata(metadata)
    typed_response_id = str(metadata.get("typed_response_id") or "").strip()
    candidate_id = str(metadata.get("candidate_id") or "").strip()
    return {
        "family_id": family,
        "typed_response_id": typed_response_id,
        "candidate_id": candidate_id,
    }


def _frontier_execution_selection(
    stack_plan: dict[str, Any],
    *,
    max_paths: int = 1,
) -> dict[str, Any]:
    frontier_paths = [
        path
        for path in stack_plan.get("stack_acquisition_frontier") or []
        if isinstance(path, dict)
    ]
    if not frontier_paths:
        primary = stack_plan.get("primary_stack_acquisition_path")
        if isinstance(primary, dict):
            frontier_paths = [primary]
    selected_paths = frontier_paths[: max(1, max_paths)]
    families = ordered_unique(
        family
        for path in selected_paths
        for family in _string_list(path.get("family_order"))
    )
    typed_response_ids = ordered_unique(
        typed
        for path in selected_paths
        for typed in _string_list(path.get("typed_response_order"))
    )
    row_keys = ordered_unique(
        key for path in selected_paths for key in _string_list(path.get("row_keys"))
    )
    selection = {
        "schema": "repair_campaign_frontier_executable_selection.v1",
        "selection_active": bool(selected_paths and (families or typed_response_ids)),
        "source_stack_plan_schema": stack_plan.get("schema"),
        "source_frontier_path_count": len(frontier_paths),
        "selected_frontier_path_count": len(selected_paths),
        "selected_family_ids": families,
        "selected_typed_response_ids": typed_response_ids,
        "selected_stack_row_keys": row_keys,
        "selected_terminal_outcome_classes": ordered_unique(
            str(path.get("terminal_outcome_class") or "")
            for path in selected_paths
            if str(path.get("terminal_outcome_class") or "").strip()
        ),
        "selected_hyperedge_keys": ordered_unique(
            str(path.get("source_hyperedge_key") or "")
            for path in selected_paths
            if str(path.get("source_hyperedge_key") or "").strip()
        ),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        selection,
        context="repair_campaign_frontier_executable_selection",
    )
    return selection


def _experiment_matches_frontier_selection(
    experiment: dict[str, Any],
    selection: dict[str, Any],
) -> bool:
    metadata = experiment.get("metadata")
    if not isinstance(metadata, dict):
        return False
    identity = _metadata_identity(metadata)
    selected_families = set(_string_list(selection.get("selected_family_ids")))
    selected_typed = set(_string_list(selection.get("selected_typed_response_ids")))
    return bool(
        (identity["family_id"] and identity["family_id"] in selected_families)
        or (
            identity["typed_response_id"]
            and identity["typed_response_id"] in selected_typed
        )
        or (identity["candidate_id"] and identity["candidate_id"] in selected_families)
    )


def _experiment_archive_bound_default_ready(experiment: dict[str, Any]) -> bool:
    metadata = experiment.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if not str(
        metadata.get("repair_family_byte_transform_execution_report_path") or ""
    ).strip():
        return False
    for step in experiment.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("id") == "execute_repair_family_byte_transform":
            return True
        command = step.get("command")
        if isinstance(command, list) and any(
            str(item).endswith("run_repair_family_byte_transform_executor.py")
            for item in command
        ):
            return True
    return False


def _with_frontier_archive_bound_default(
    experiment: dict[str, Any],
) -> dict[str, Any]:
    cloned = dict(experiment)
    metadata = dict(cloned.get("metadata") or {})
    ready = _experiment_archive_bound_default_ready(experiment)
    metadata.update(
        {
            "frontier_selected_archive_bound_candidate_default": True,
            "frontier_selected_archive_bound_candidate_default_ready": ready,
            "frontier_selected_candidate_archive_required": True,
            "frontier_selected_runtime_consumption_proof_required": True,
            "frontier_selected_exact_bridge_preclaim_required": True,
            "frontier_selected_default_blockers": []
            if ready
            else [
                "frontier_selected_experiment_missing_archive_bound_materializer_steps"
            ],
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
    )
    cloned["metadata"] = metadata
    return cloned


def _frontier_archive_bound_default_contract(
    *,
    selected: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ready_ids: list[str] = []
    missing_ids: list[str] = []
    for experiment in selected:
        experiment_id = str(experiment.get("id") or "")
        metadata = experiment.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("frontier_selected_archive_bound_candidate_default_ready") is True:
            ready_ids.append(experiment_id)
        else:
            missing_ids.append(experiment_id)
    contract = {
        "schema": "repair_campaign_frontier_selected_archive_bound_default_contract.v1",
        "candidate_archive_emission_default": True,
        "receiver_runtime_proof_default": True,
        "exact_ready_bridge_preclaim_default": True,
        "selected_experiment_count": len(selected),
        "skipped_experiment_count": len(skipped),
        "selected_archive_bound_experiment_count": len(ready_ids),
        "selected_missing_archive_bound_default_count": len(missing_ids),
        "archive_bound_ready_experiment_ids": ready_ids,
        "archive_bound_missing_experiment_ids": missing_ids,
        "blockers": [
            f"frontier_selected_archive_bound_default_missing:{experiment_id}"
            for experiment_id in missing_ids
            if experiment_id
        ],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        contract,
        context="repair_campaign_frontier_selected_archive_bound_default_contract",
    )
    return contract


def _frontier_selected_queue(
    *,
    queue: dict[str, Any],
    selection: dict[str, Any],
    iteration: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiments = [
        experiment
        for experiment in queue.get("experiments") or []
        if isinstance(experiment, dict)
    ]
    selected_source = [
        experiment
        for experiment in experiments
        if _experiment_matches_frontier_selection(experiment, selection)
    ]
    selected_source_ids = {id(experiment) for experiment in selected_source}
    selected = [
        _with_frontier_archive_bound_default(experiment)
        for experiment in selected_source
    ]
    skipped = [
        experiment for experiment in experiments if id(experiment) not in selected_source_ids
    ]
    archive_bound_default_contract = _frontier_archive_bound_default_contract(
        selected=selected,
        skipped=skipped,
    )
    filtered_queue = dict(queue)
    filtered_queue["queue_id"] = (
        f"{queue.get('queue_id') or 'repair_campaign_queue'}"
        f"__frontier_selected_iter_{iteration}"
    )
    filtered_queue["experiments"] = selected
    metadata = dict(filtered_queue.get("metadata") or {})
    metadata.update(
        {
            "schema": "repair_campaign_frontier_selected_queue_metadata.v1",
            "source_queue_id": queue.get("queue_id"),
            "frontier_execution_selection": selection,
            "frontier_selected_experiment_count": len(selected),
            "frontier_skipped_experiment_count": len(skipped),
            "frontier_selected_queues_emit_archive_bound_candidates_by_default": True,
            "archive_bound_candidate_default_contract": (
                archive_bound_default_contract
            ),
            "budget_spend_allowed": False,
            "ready_for_exact_eval_dispatch": False,
            **FALSE_AUTHORITY,
        }
    )
    filtered_queue["metadata"] = metadata
    filtered_queue.update(FALSE_AUTHORITY)
    report = {
        "schema": "repair_campaign_frontier_selected_queue_report.v1",
        "iteration": iteration,
        "selection_active": selection.get("selection_active") is True,
        "source_queue_id": queue.get("queue_id"),
        "selected_queue_id": filtered_queue["queue_id"],
        "source_experiment_count": len(experiments),
        "selected_experiment_count": len(selected),
        "skipped_experiment_count": len(skipped),
        "selected_experiment_ids": [
            str(experiment.get("id") or "") for experiment in selected
        ],
        "skipped_experiment_ids": [
            str(experiment.get("id") or "") for experiment in skipped
        ],
        "frontier_execution_selection": selection,
        "archive_bound_candidate_default_contract": (
            archive_bound_default_contract
        ),
        "selected_archive_bound_experiment_count": (
            archive_bound_default_contract["selected_archive_bound_experiment_count"]
        ),
        "selected_missing_archive_bound_default_count": (
            archive_bound_default_contract[
                "selected_missing_archive_bound_default_count"
            ]
        ),
        "blockers": []
        if selected
        else ["frontier_selection_matched_no_queue_experiments"],
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        filtered_queue,
        context="repair_campaign_frontier_selected_queue",
    )
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_frontier_selected_queue_report",
    )
    report["blockers"] = ordered_unique(
        [
            *report["blockers"],
            *_string_list(archive_bound_default_contract.get("blockers")),
        ]
    )
    require_no_truthy_authority_fields(
        report,
        context="repair_campaign_frontier_selected_queue_report_after_contract",
    )
    return filtered_queue, report


def _entropy_compiler_stage(row: dict[str, Any]) -> str:
    label = str(row.get("entropy_position_label") or "").strip()
    order = int(row.get("entropy_stage_order") or 999)
    if label.startswith("before_entropy_coder") or order <= 10:
        return "before_coder"
    if (
        label.startswith("at_entropy_coder")
        or label == "selector_codec_entropy"
        or 10 < order <= 40
    ):
        return "coder_boundary"
    if label.startswith("after_entropy_coder") or order >= 50:
        return "after_coder"
    return "unknown_entropy_position"


def _compile_entropy_stage_materializer_work_orders(
    stack_plan: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row for row in stack_plan.get("stack_rows") or [] if isinstance(row, dict)
    ]
    selected_keys = set()
    primary = stack_plan.get("primary_stack_acquisition_path")
    if isinstance(primary, dict):
        selected_keys.update(_string_list(primary.get("row_keys")))
    if not selected_keys:
        selected_keys.update(
            str(row.get("stack_row_key") or "") for row in rows if row.get("stack_row_key")
        )
    selected_rows = [
        row
        for row in rows
        if str(row.get("stack_row_key") or "") in selected_keys
    ]
    stage_order = ("before_coder", "coder_boundary", "after_coder")
    work_orders: list[dict[str, Any]] = []
    for stage in stage_order:
        stage_rows = [row for row in selected_rows if _entropy_compiler_stage(row) == stage]
        if not stage_rows:
            continue
        stage_rows.sort(
            key=lambda row: (
                int(row.get("entropy_stage_order") or 999),
                int(row.get("planned_stack_order") or 999),
                str(row.get("typed_response_id") or ""),
            )
        )
        work_order = {
            "schema": "repair_campaign_entropy_stage_materializer_work_order.v1",
            "compiler_stage": stage,
            "stage_materialization_order": len(work_orders) + 1,
            "source_stack_plan_schema": stack_plan.get("schema"),
            "source_row_keys": [
                str(row.get("stack_row_key") or "") for row in stage_rows
            ],
            "family_order": [
                str(row.get("family_id") or "") for row in stage_rows
            ],
            "typed_response_order": [
                str(row.get("typed_response_id") or "") for row in stage_rows
            ],
            "entropy_stage_order": [
                int(row.get("entropy_stage_order") or 999) for row in stage_rows
            ],
            "fractal_scope_union_levels": ordered_unique(
                level
                for row in stage_rows
                for level in _string_list(row.get("fractal_scope_levels"))
            ),
            "total_delta_payload_bytes": sum(
                max(0, int(row.get("delta_payload_bytes") or 0))
                for row in stage_rows
            ),
            "total_local_mlx_expected_improvement_score_units": sum(
                float(row.get("local_mlx_expected_improvement_score_units") or 0.0)
                for row in stage_rows
            ),
            "candidate_archive_required": True,
            "receiver_decode_only_runtime_proof_required": True,
            "archive_bound_exact_ready_bridge_input_required": True,
            "preclaim_gate_required_before_exact_dispatch": True,
            "materializer_action": (
                "emit_byte_closed_archive_bound_candidates_for_entropy_stage"
            ),
            "blockers": ordered_unique(
                blocker
                for row in stage_rows
                for blocker in _string_list(row.get("blockers"))
            ),
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            work_order,
            context=f"repair_campaign_entropy_stage_materializer_work_order:{stage}",
        )
        work_orders.append(work_order)
    bundle = {
        "schema": "repair_campaign_entropy_stage_materializer_work_order_bundle.v1",
        "source_stack_plan_schema": stack_plan.get("schema"),
        "work_order_count": len(work_orders),
        "stage_order": list(stage_order),
        "work_orders": work_orders,
        "archive_bound_candidate_default": True,
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        bundle,
        context="repair_campaign_entropy_stage_materializer_work_order_bundle",
    )
    return bundle


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
    coverage = summary.get("repair_family_coverage") or {}
    if coverage.get("missing_required_family_count"):
        return "required_repair_family_coverage_incomplete"
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
        "precise_exact_axis_blocker",
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
            *[
                blocker
                for gate in summary.get("exact_dispatch_preclaim_gates") or []
                if isinstance(gate, dict)
                for blocker in _string_list(gate.get("blockers"))
            ],
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
        "exact_dispatch_preclaim_gate_count": summary.get(
            "exact_dispatch_preclaim_gate_count",
            0,
        ),
        "exact_dispatch_preclaim_blocker_count": summary.get(
            "exact_dispatch_preclaim_blocker_count",
            0,
        ),
        "failure_rebudgeting_update_count": summary.get(
            "failure_rebudgeting_update_count",
            0,
        ),
        "failure_rebudgeting_updates": summary.get("failure_rebudgeting_updates", []),
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


def _exact_dispatch_preclaim_gates(
    *,
    bridge_report: dict[str, Any],
    output_dir: Path,
    overwrite_artifacts: bool,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for index, row in enumerate(bridge_report.get("rows") or [], start=1):
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        output_path = (
            output_dir
            / f"{index:03d}_{_slug(candidate_id)}.provider_preclaim_check.json"
        )
        payload = build_preclaim_check(
            provider="lightning",
            job_id=candidate_id,
        )
        expected_sha = sha256_file(output_path) if output_path.exists() and overwrite_artifacts else None
        write_json_artifact(
            output_path,
            payload,
            allow_overwrite=overwrite_artifacts,
            expected_existing_sha256=expected_sha,
        )
        gate = {
            "schema": "repair_family_exact_dispatch_preclaim_gate.v1",
            "candidate_id": candidate_id,
            "family_id": row.get("family_id"),
            "typed_response_id": row.get("typed_response_id"),
            "candidate_chain_id": row.get("candidate_chain_id"),
            "candidate_chain_ids": _string_list(row.get("candidate_chain_ids")),
            "entropy_position_label": row.get("entropy_position_label"),
            "entropy_stage_order": row.get("entropy_stage_order"),
            "failure_rebudgeting_identity": dict(
                _mapping(row.get("failure_rebudgeting_identity"))
            ),
            "provider": "lightning",
            "preclaim_check_path": _repo_rel(output_path),
            "preclaim_ready": payload.get("preclaim_ready") is True,
            "preclaim_check": payload,
            "command": [
                sys.executable,
                "tools/check_exact_dispatch_provider_preclaim.py",
                "--provider",
                "lightning",
                "--job-id",
                candidate_id,
                "--output",
                _repo_rel(output_path),
                "--overwrite",
            ],
            "blockers": ordered_unique(
                [
                    *_string_list(payload.get("blockers")),
                    "lane_dispatch_claim_required_before_exact_eval",
                    "materializer_exact_eval_dispatch_plan_required_before_claim",
                ]
            ),
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            gate,
            context=f"repair_family_exact_dispatch_preclaim_gate:{candidate_id}",
        )
        gates.append(gate)
    return gates


def _failure_rebudgeting_updates(
    *,
    stack_plan: dict[str, Any],
    bridge_report: dict[str, Any],
    preclaim_gates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    stack_rows = [
        row for row in stack_plan.get("stack_rows") or [] if isinstance(row, dict)
    ]
    stack_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in stack_rows:
        stack_by_identity[
            (
                str(row.get("family_id") or ""),
                str(row.get("typed_response_id") or ""),
            )
        ] = row
    preclaim_by_candidate = {
        str(gate.get("candidate_id") or ""): gate for gate in preclaim_gates
    }
    updates: list[dict[str, Any]] = []
    for row in bridge_report.get("rows") or []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        family_id = str(row.get("family_id") or "")
        typed_response_id = str(row.get("typed_response_id") or "")
        stack_row = stack_by_identity.get((family_id, typed_response_id), {})
        preclaim_gate = preclaim_by_candidate.get(candidate_id, {})
        failure_identity = dict(_mapping(row.get("failure_rebudgeting_identity")))
        if not failure_identity:
            failure_identity = dict(
                _mapping(preclaim_gate.get("failure_rebudgeting_identity"))
            )
        if not failure_identity:
            failure_identity = {
                "schema": "repair_family_exact_failure_rebudgeting_identity.v1",
                "candidate_id": candidate_id,
                "family_id": family_id,
                "typed_response_id": typed_response_id,
                "candidate_chain_id": row.get("candidate_chain_id"),
                "candidate_chain_ids": _string_list(row.get("candidate_chain_ids")),
                "entropy_position_label": row.get("entropy_position_label"),
                "entropy_stage_order": (
                    row.get("entropy_stage_order")
                    or stack_row.get("entropy_stage_order")
                ),
                "fractal_scope_levels": stack_row.get("fractal_scope_levels", []),
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
        blockers = ordered_unique(
            [
                *_string_list(row.get("blockers")),
                *_string_list(preclaim_gate.get("blockers")),
            ]
        )
        blocker_text = " ".join(blockers)
        if "runtime_consumption_proof" in blocker_text or "receiver" in blocker_text:
            blocker_class = "receiver_runtime_proof_rebudget_required"
            policy = "prioritize_archive_bound_runtime_consumption_proof"
        elif "runtime_content_tree" in blocker_text or "submission_" in blocker_text:
            blocker_class = "runtime_content_tree_custody_rebuild_required"
            policy = "prioritize_runtime_content_tree_custody_before_exact_handoff"
        elif "candidate_archive" in blocker_text or "archive_custody" in blocker_text:
            blocker_class = "archive_custody_rebuild_required"
            policy = "prioritize_byte_closed_family_materializer_implementation"
        elif (
            "lightning_" in blocker_text
            or "preclaim" in blocker_text
            or "claim_required" in blocker_text
            or preclaim_gate
        ):
            blocker_class = "exact_dispatch_preclaim_failed"
            policy = "rebuild_exact_dispatch_provider_preclaim_before_budget_spend"
        else:
            blocker_class = "exact_axis_handoff_required"
            policy = "hold_until_byte_closed_exact_auth_handoff_available"
        update = {
            "schema": "repair_family_failure_rebudgeting_update.v1",
            "candidate_id": candidate_id,
            "family_id": family_id,
            "typed_response_id": typed_response_id,
            "candidate_chain_id": failure_identity.get("candidate_chain_id"),
            "candidate_chain_ids": _string_list(
                failure_identity.get("candidate_chain_ids")
            ),
            "entropy_position_label": failure_identity.get("entropy_position_label"),
            "entropy_stage_order": failure_identity.get("entropy_stage_order"),
            "fractal_scope_levels": failure_identity.get(
                "fractal_scope_levels",
                stack_row.get("fractal_scope_levels", []),
            ),
            "chain_stage_identities": [
                dict(item)
                for item in failure_identity.get("chain_stage_identities") or []
                if isinstance(item, dict)
            ],
            "source_archive_sha256": failure_identity.get("source_archive_sha256"),
            "candidate_archive_sha256": failure_identity.get(
                "candidate_archive_sha256"
            ),
            "runtime_consumption_proof_sha256": failure_identity.get(
                "runtime_consumption_proof_sha256"
            ),
            "runtime_content_tree_sha256": failure_identity.get(
                "runtime_content_tree_sha256"
            ),
            "failure_rebudgeting_identity": failure_identity,
            "selected_blocker_class": blocker_class,
            "recommended_acquisition_policy": policy,
            "responsible_failure_surface": (
                "exact_dispatch_preclaim"
                if blocker_class == "exact_dispatch_preclaim_failed"
                else (
                    "receiver_runtime_proof"
                    if blocker_class == "receiver_runtime_proof_rebudget_required"
                    else (
                        "runtime_content_tree"
                        if blocker_class
                        == "runtime_content_tree_custody_rebuild_required"
                        else (
                            "candidate_archive_custody"
                            if blocker_class == "archive_custody_rebuild_required"
                            else "exact_axis_handoff"
                        )
                    )
                )
            ),
            "demote_responsible_family_stage_scope": blocker_class
            in {
                "exact_dispatch_preclaim_failed",
                "receiver_runtime_proof_rebudget_required",
                "runtime_content_tree_custody_rebuild_required",
                "archive_custody_rebuild_required",
            },
            "rebudget_receiver_closed_credit": (
                blocker_class == "receiver_runtime_proof_rebudget_required"
            ),
            "preclaim_failed": blocker_class == "exact_dispatch_preclaim_failed",
            "receiver_or_runtime_failed": blocker_class
            in {
                "receiver_runtime_proof_rebudget_required",
                "runtime_content_tree_custody_rebuild_required",
            },
            "blockers": blockers,
            "budget_spend_allowed": False,
            "ready_for_budget_spend": False,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            **FALSE_AUTHORITY,
        }
        require_no_truthy_authority_fields(
            update,
            context=f"repair_family_failure_rebudgeting_update:{candidate_id}",
        )
        updates.append(update)
    return updates


def _build_summary(
    *,
    queue_path: Path,
    output_dir: Path,
    posterior_path: Path | None,
    byte_credit_budget: int | None,
    max_iterations: int,
    max_steps_per_iteration: int,
    worker_max_experiments_per_iteration: int | None,
    execute_local: bool,
    required_family_ids: Sequence[str] = (),
    require_all_queue_families: bool = False,
    submission_dirs: Sequence[Path] = (),
    overwrite_artifacts: bool = False,
) -> dict[str, Any]:
    queue = _load_json(queue_path)
    require_no_truthy_authority_fields(queue, context="autonomous_floor_loop_queue")
    queue_family_ids = _queue_family_ids(queue)
    required_families = ordered_unique(
        [
            *(_string_list(required_family_ids)),
            *(queue_family_ids if require_all_queue_families else []),
        ]
    )
    worker_experiment_limit = worker_max_experiments_per_iteration
    if worker_experiment_limit is None:
        worker_experiment_limit = max(1, len(required_families) if required_families else 1)
    iterations: list[dict[str, Any]] = []
    final_coverage = _family_coverage_report(
        required_family_ids=required_families,
        queue_family_ids=queue_family_ids,
        reports=[],
    )
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
        pre_worker_report_paths = _discover_execution_reports(output_dir, queue)
        pre_worker_reports = _load_execution_reports(pre_worker_report_paths)
        pre_worker_stack_plan = final_stack_plan
        if pre_worker_reports:
            pre_worker_stack_plan = plan_repair_family_stack_search(
                execution_reports=pre_worker_reports,
                execution_report_paths=tuple(
                    _repo_rel(path) for path in pre_worker_report_paths
                ),
                repo_root=REPO_ROOT,
                posterior_path=posterior_path,
                byte_credit_budget=byte_credit_budget,
            )
            final_stack_plan = pre_worker_stack_plan
        frontier_selection = _frontier_execution_selection(pre_worker_stack_plan)
        frontier_selected_queue_path = None
        frontier_selected_queue_report = None
        worker_queue_path = queue_path
        if execute_local and frontier_selection.get("selection_active") is True:
            selected_queue, frontier_selected_queue_report = _frontier_selected_queue(
                queue=queue,
                selection=frontier_selection,
                iteration=index,
            )
            frontier_selected_queue_path_obj = (
                output_dir / f"iteration_{index}_frontier_selected_queue.json"
            )
            expected_frontier_queue_sha = (
                sha256_file(frontier_selected_queue_path_obj)
                if frontier_selected_queue_path_obj.exists() and overwrite_artifacts
                else None
            )
            write_json_artifact(
                frontier_selected_queue_path_obj,
                selected_queue,
                allow_overwrite=overwrite_artifacts,
                expected_existing_sha256=expected_frontier_queue_sha,
            )
            frontier_selected_queue_path = _repo_rel(frontier_selected_queue_path_obj)
            if frontier_selected_queue_report["selected_experiment_count"]:
                worker_queue_path = frontier_selected_queue_path_obj
        worker_result = None
        if execute_local:
            worker_result = _run_worker(
                queue_path=worker_queue_path,
                output_path=output_dir / f"iteration_{index}_worker_result.json",
                max_steps=max_steps_per_iteration,
                max_experiments=worker_experiment_limit,
            )
        report_paths = _discover_execution_reports(output_dir, queue)
        reports = _load_execution_reports(report_paths)
        final_coverage = _family_coverage_report(
            required_family_ids=required_families,
            queue_family_ids=queue_family_ids,
            reports=reports,
        )
        if reports:
            final_stack_plan = plan_repair_family_stack_search(
                execution_reports=reports,
                execution_report_paths=tuple(_repo_rel(path) for path in report_paths),
                repo_root=REPO_ROOT,
                posterior_path=posterior_path,
                byte_credit_budget=byte_credit_budget,
            )
        stop_reason = _iteration_stop_reason(final_stack_plan, worker_result)
        if (
            worker_result is None
            or worker_result.get("returncode") in (0, None)
        ) and final_coverage["coverage_satisfied"] is not True:
            stop_reason = "required_repair_family_coverage_incomplete"
        iterations.append(
            {
                "schema": "repair_campaign_autonomous_floor_loop_iteration.v1",
                "iteration": index,
                "execute_local": execute_local,
                "worker_result": worker_result,
                "worker_queue_path": _repo_rel(worker_queue_path),
                "pre_worker_execution_report_count": len(pre_worker_reports),
                "pre_worker_stack_plan_schema": pre_worker_stack_plan.get("schema"),
                "frontier_execution_selection": frontier_selection,
                "frontier_selected_queue_path": frontier_selected_queue_path,
                "frontier_selected_queue_report": frontier_selected_queue_report,
                "execution_report_count": len(reports),
                "execution_report_paths": [_repo_rel(path) for path in report_paths],
                "stack_plan_schema": final_stack_plan.get("schema"),
                "candidate_improvement_observed": final_stack_plan.get("candidate_improvement_observed") is True,
                "repair_family_coverage": final_coverage,
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
    entropy_stage_materializer_work_orders = (
        _compile_entropy_stage_materializer_work_orders(final_stack_plan)
    )
    entropy_stage_materializer_work_orders_path = (
        output_dir / "repair_family_entropy_stage_materializer_work_orders.json"
    )
    entropy_stage_chain_execution_bundle = (
        build_repair_entropy_stage_chain_execution_bundle(
            execution_reports=reports,
            execution_report_paths=tuple(_repo_rel(path) for path in report_paths),
            work_order_bundle=entropy_stage_materializer_work_orders,
            output_dir=output_dir / "repair_family_entropy_stage_chain_execution",
            repo_root=REPO_ROOT,
            allow_overwrite=overwrite_artifacts,
        )
    )
    entropy_stage_chain_execution_bundle_path = (
        output_dir / "repair_family_entropy_stage_chain_execution_bundle.json"
    )
    exact_handoff_plan = build_repair_family_exact_handoff_plan(
        stack_plan=final_stack_plan,
        stack_plan_path=_repo_rel(stack_plan_path),
        chain_execution_bundle=entropy_stage_chain_execution_bundle,
    )
    exact_ready_bridge = build_repair_family_exact_ready_bridge(
        exact_handoff_plan=exact_handoff_plan,
        exact_handoff_plan_path=_repo_rel(exact_handoff_plan_path),
        submission_dirs=tuple(submission_dirs),
        repo_root=REPO_ROOT,
    )
    exact_ready_bridge_report = exact_ready_bridge["bridge_report"]
    exact_dispatch_preclaim_gates = _exact_dispatch_preclaim_gates(
        bridge_report=exact_ready_bridge_report,
        output_dir=output_dir,
        overwrite_artifacts=overwrite_artifacts,
    )
    failure_rebudgeting_updates = _failure_rebudgeting_updates(
        stack_plan=final_stack_plan,
        bridge_report=exact_ready_bridge_report,
        preclaim_gates=exact_dispatch_preclaim_gates,
    )
    exact_ready_source_queue_path = output_dir / "repair_family_exact_ready_source_queue.json"
    blocked_exact_ready_queue_path = output_dir / "repair_family_blocked_exact_ready_queue.json"
    exact_ready_bridge_report_path = output_dir / "repair_family_exact_ready_bridge_report.json"
    learning_signal_report_path = output_dir / "repair_family_stack_learning_signal_report.json"
    primary_path = final_stack_plan.get("primary_stack_acquisition_path")
    primary_terminal_outcome = (
        str(primary_path.get("terminal_outcome_class") or "") if isinstance(primary_path, dict) else None
    )
    fractal_surface = final_stack_plan.get("fractal_marginal_surface")
    fractal_surface = fractal_surface if isinstance(fractal_surface, dict) else {}
    fractal_cells = [
        cell for cell in fractal_surface.get("cells") or [] if isinstance(cell, dict)
    ]
    acquisition_frontier = [
        path
        for path in final_stack_plan.get("stack_acquisition_frontier") or []
        if isinstance(path, dict)
    ]
    primary_frontier_path = acquisition_frontier[0] if acquisition_frontier else None
    learning_signal_report = build_repair_family_stack_learning_signal_report(
        stack_plan=final_stack_plan,
        bridge_report=exact_ready_bridge_report,
        chain_execution_bundle=entropy_stage_chain_execution_bundle,
        failure_rebudgeting_updates=failure_rebudgeting_updates,
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
        "worker_max_experiments_per_iteration": worker_experiment_limit,
        "queue_family_ids": queue_family_ids,
        "required_family_ids": required_families,
        "require_all_queue_families": require_all_queue_families,
        "repair_family_coverage": final_coverage,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "stack_search_plan_path": _repo_rel(stack_plan_path),
        "stack_search_plan": final_stack_plan,
        "primary_stack_acquisition_terminal_outcome": primary_terminal_outcome,
        "primary_stack_acquisition_path": primary_path,
        "fractal_marginal_surface_schema": fractal_surface.get("schema"),
        "fractal_marginal_surface_cell_count": final_stack_plan.get(
            "fractal_marginal_surface_cell_count",
            0,
        ),
        "archive_entropy_substrate_gap_count": final_stack_plan.get(
            "archive_entropy_substrate_gap_count",
            0,
        ),
        "archive_entropy_substrate_blockers": final_stack_plan.get(
            "archive_entropy_substrate_blockers",
            [],
        ),
        "top_fractal_marginal_surface_cells": fractal_cells[:8],
        "stack_acquisition_frontier_count": final_stack_plan.get(
            "stack_acquisition_frontier_count",
            0,
        ),
        "stack_acquisition_frontier": acquisition_frontier,
        "primary_stack_acquisition_frontier_path": primary_frontier_path,
        "measured_mlx_posterior_budget_routing_update_count": (
            final_stack_plan.get(
                "measured_mlx_posterior_budget_routing_update_count",
                0,
            )
        ),
        "measured_mlx_posterior_budget_routing_updates": (
            final_stack_plan.get("measured_mlx_posterior_budget_routing_updates", [])
        ),
        "entropy_stage_materializer_work_orders_path": _repo_rel(
            entropy_stage_materializer_work_orders_path
        ),
        "entropy_stage_materializer_work_order_count": (
            entropy_stage_materializer_work_orders["work_order_count"]
        ),
        "entropy_stage_materializer_work_orders": (
            entropy_stage_materializer_work_orders
        ),
        "entropy_stage_chain_execution_bundle_path": _repo_rel(
            entropy_stage_chain_execution_bundle_path
        ),
        "entropy_stage_chain_execution_bundle_schema": (
            REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA
        ),
        "entropy_stage_chain_execution_bundle": (
            entropy_stage_chain_execution_bundle
        ),
        "entropy_stage_chain_count": (
            entropy_stage_chain_execution_bundle["chain_count"]
        ),
        "entropy_stage_chain_materialized_candidate_count": (
            entropy_stage_chain_execution_bundle[
                "materialized_chain_candidate_count"
            ]
        ),
        "entropy_stage_chain_runtime_consumption_proof_ready_count": (
            entropy_stage_chain_execution_bundle[
                "runtime_consumption_proof_ready_count"
            ]
        ),
        "archive_entropy_substrate_coverage_count": (
            entropy_stage_chain_execution_bundle.get(
                "archive_entropy_substrate_coverage_count",
                0,
            )
        ),
        "archive_entropy_substrate_coverages": (
            entropy_stage_chain_execution_bundle.get(
                "archive_entropy_substrate_coverages",
                [],
            )
        ),
        "frontier_executable_selection_consumed": any(
            isinstance(item, dict)
            and isinstance(item.get("frontier_selected_queue_report"), dict)
            and item["frontier_selected_queue_report"].get(
                "selected_experiment_count"
            )
            for item in iterations
        ),
        "exact_handoff_plan_path": _repo_rel(exact_handoff_plan_path),
        "exact_handoff_plan_schema": REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
        "exact_handoff_plan": exact_handoff_plan,
        "exact_eval_handoff_candidate_count": exact_handoff_plan["candidate_count"],
        "archive_bound_exact_handoff_candidate_count": exact_handoff_plan["archive_bound_candidate_count"],
        "entropy_stage_chain_exact_handoff_candidate_count": (
            exact_handoff_plan.get("entropy_stage_chain_candidate_count", 0)
        ),
        "entropy_stage_chain_archive_bound_exact_handoff_candidate_count": (
            exact_handoff_plan.get(
                "entropy_stage_chain_archive_bound_candidate_count",
                0,
            )
        ),
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
        "exact_dispatch_preclaim_gate_count": len(exact_dispatch_preclaim_gates),
        "exact_dispatch_preclaim_blocker_count": sum(
            len(_string_list(gate.get("blockers")))
            for gate in exact_dispatch_preclaim_gates
        ),
        "exact_dispatch_preclaim_gates": exact_dispatch_preclaim_gates,
        "failure_rebudgeting_update_count": len(failure_rebudgeting_updates),
        "failure_rebudgeting_updates": failure_rebudgeting_updates,
        "posterior_learning_signal_report_path": _repo_rel(learning_signal_report_path),
        "posterior_learning_signal_report_schema": (REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA),
        "posterior_stack_learning_signal_count": (
            learning_signal_report.get("stack_learning_signal_count", 0)
        ),
        "entropy_stage_chain_posterior_learning_signal_count": (
            learning_signal_report.get("entropy_stage_chain_learning_signal_count", 0)
        ),
        "exact_failure_rebudgeting_posterior_learning_signal_count": (
            learning_signal_report.get(
                "exact_failure_rebudgeting_learning_signal_count",
                0,
            )
        ),
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
            "archive_bound_bridge_rows_emit_exact_dispatch_preclaim_gate_artifacts",
            "exact_preclaim_receiver_and_runtime_failures_emit_rebudgeting_updates",
            "stack_and_bridge_outcomes_emit_posterior_learning_signals",
            "fractal_marginal_surface_and_ranked_frontier_are_operator_visible",
            "measured_mlx_marginals_update_posterior_budget_routing_directly",
            "frontier_paths_filter_executable_iteration_queues_when_available",
            "frontier_selected_queues_default_to_archive_bound_candidate_emission",
            "entropy_stage_chain_compiler_emits_materializer_work_orders",
            "entropy_stage_chain_compiler_executes_composed_archive_candidates",
            "precise_blocker_report_names_next_unblocked_action",
        ],
        "blockers": ordered_unique(
            [
                "contest_cpu_or_cuda_exact_axis_payload_required_before_score",
                "lane_dispatch_claim_required_before_exact_eval",
                *_string_list(entropy_stage_chain_execution_bundle.get("blockers")),
                *(
                    []
                    if final_stack_plan.get("execution_report_count")
                    else ["repair_family_byte_transform_execution_reports_missing"]
                ),
                *_string_list(final_coverage.get("blockers")),
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
            worker_max_experiments_per_iteration=args.worker_max_experiments_per_iteration,
            execute_local=bool(args.execute_local),
            required_family_ids=tuple(args.require_family_id),
            require_all_queue_families=bool(args.require_all_queue_families),
            submission_dirs=tuple(args.submission_dir),
            overwrite_artifacts=bool(args.overwrite),
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
        entropy_stage_materializer_work_orders_path = (
            output_dir / "repair_family_entropy_stage_materializer_work_orders.json"
        )
        entropy_stage_chain_execution_bundle_path = (
            output_dir / "repair_family_entropy_stage_chain_execution_bundle.json"
        )
        for path, payload in (
            (
                entropy_stage_materializer_work_orders_path,
                summary["entropy_stage_materializer_work_orders"],
            ),
            (
                entropy_stage_chain_execution_bundle_path,
                summary["entropy_stage_chain_execution_bundle"],
            ),
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
                "required_family_count": summary["repair_family_coverage"]["required_family_count"],
                "executed_family_count": summary["repair_family_coverage"]["executed_family_count"],
                "missing_required_family_ids": summary["repair_family_coverage"]["missing_required_family_ids"],
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
                "fractal_marginal_surface_cell_count": summary[
                    "fractal_marginal_surface_cell_count"
                ],
                "stack_acquisition_frontier_count": summary[
                    "stack_acquisition_frontier_count"
                ],
                "frontier_executable_selection_consumed": summary[
                    "frontier_executable_selection_consumed"
                ],
                "measured_mlx_posterior_budget_routing_update_count": summary[
                    "measured_mlx_posterior_budget_routing_update_count"
                ],
                "entropy_stage_materializer_work_order_count": summary[
                    "entropy_stage_materializer_work_order_count"
                ],
                "entropy_stage_chain_count": summary[
                    "entropy_stage_chain_count"
                ],
                "entropy_stage_chain_materialized_candidate_count": summary[
                    "entropy_stage_chain_materialized_candidate_count"
                ],
                "exact_dispatch_preclaim_gate_count": summary[
                    "exact_dispatch_preclaim_gate_count"
                ],
                "exact_dispatch_preclaim_blocker_count": summary[
                    "exact_dispatch_preclaim_blocker_count"
                ],
                "failure_rebudgeting_update_count": summary[
                    "failure_rebudgeting_update_count"
                ],
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
