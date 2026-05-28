# SPDX-License-Identifier: MIT
"""Autonomous multi-archive runner for repair-family floor campaigns."""

from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.repair_campaign_materialization_queue import (
    build_repair_campaign_byte_closed_materialization_queue,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_archive_candidate_intake import (
    build_repair_campaign_work_order_from_archives,
)
from tac.optimization.repair_campaign_posterior import (
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
)
from tac.optimization.repair_campaign_scorer import score_repair_campaign
from tac.optimizer.materializer_submission_closure import (
    MaterializerSubmissionClosureError,
    build_materializer_submission_runtime_closure,
)
from tac.repo_io import sha256_file, write_json_artifact

REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNNER_SCHEMA = (
    "repair_autonomous_multi_archive_runner.v1"
)
REPAIR_AUTONOMOUS_MULTI_ARCHIVE_DISCOVERY_SCHEMA = (
    "repair_autonomous_multi_archive_discovery.v1"
)
REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNTIME_CLOSURE_SCHEMA = (
    "repair_autonomous_multi_archive_runtime_closure.v1"
)


class RepairAutonomousMultiArchiveRunnerError(ValueError):
    """Raised when the multi-archive autonomous runner cannot complete."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(repo_root) / value


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve(strict=False)).as_posix()
    except ValueError:
        return value.as_posix()


def _slug(value: Any) -> str:
    text = str(value or "candidate").strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    return "_".join("".join(chars).split("_")) or "candidate"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairAutonomousMultiArchiveRunnerError(f"{path} must contain a JSON object")
    return payload


def _expected(path: Path, *, overwrite: bool) -> str | None:
    return sha256_file(path) if overwrite and path.is_file() else None


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    write_json_artifact(
        path,
        payload,
        allow_overwrite=overwrite,
        expected_existing_sha256=_expected(path, overwrite=overwrite),
    )


def _expand_archive_patterns(
    *,
    archives: Sequence[str | Path],
    archive_globs: Sequence[str],
    repo_root: str | Path,
) -> list[Path]:
    repo = Path(repo_root)
    paths = [_resolve(path, repo) for path in archives]
    for pattern in archive_globs:
        raw_pattern = str(pattern)
        glob_pattern = raw_pattern if Path(raw_pattern).is_absolute() else str(repo / raw_pattern)
        paths.extend(Path(match) for match in glob.glob(glob_pattern, recursive=True))
    return paths


def discover_repair_archive_candidates(
    *,
    archives: Sequence[str | Path] = (),
    archive_globs: Sequence[str] = (),
    repo_root: str | Path,
    source_labels: Sequence[str] = (),
    max_archives: int | None = None,
) -> dict[str, Any]:
    """Discover existing archive.zip candidates, deduping by archive SHA."""

    repo = Path(repo_root)
    raw_paths = _expand_archive_patterns(
        archives=archives,
        archive_globs=archive_globs,
        repo_root=repo,
    )
    if not raw_paths:
        raise RepairAutonomousMultiArchiveRunnerError("no archives or archive globs provided")
    if source_labels and len(source_labels) != len(raw_paths):
        raise RepairAutonomousMultiArchiveRunnerError(
            "source_labels must match expanded archive count"
        )
    rows: list[dict[str, Any]] = []
    seen_shas: set[str] = set()
    for index, path in enumerate(raw_paths):
        resolved = path.resolve(strict=False)
        if not resolved.is_file():
            continue
        sha = sha256_file(resolved)
        if sha in seen_shas:
            continue
        seen_shas.add(sha)
        label = (
            str(source_labels[index])
            if index < len(source_labels)
            else _slug(resolved.parent.name or resolved.stem or f"archive_{index + 1}")
        )
        if any(row["source_label"] == label for row in rows):
            label = f"{label}_{len(rows) + 1}"
        rows.append(
            {
                "schema": "repair_autonomous_multi_archive_candidate.v1",
                "source_label": label,
                "archive_path": _repo_rel(resolved, repo),
                "archive_sha256": sha,
                "archive_bytes": resolved.stat().st_size,
                "candidate_rank": len(rows) + 1,
                "training_artifact_path": _repo_rel(resolved.parent / "training_artifact.json", repo)
                if (resolved.parent / "training_artifact.json").is_file()
                else None,
                "equivalence_gate_path": _repo_rel(
                    resolved.parent / "pact_nerv_selector_v3_equivalence_gate.json",
                    repo,
                )
                if (resolved.parent / "pact_nerv_selector_v3_equivalence_gate.json").is_file()
                else None,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                **FALSE_AUTHORITY,
            }
        )
        if max_archives is not None and len(rows) >= max(0, max_archives):
            break
    if not rows:
        raise RepairAutonomousMultiArchiveRunnerError(
            "no existing archive files discovered"
        )
    payload = {
        "schema": REPAIR_AUTONOMOUS_MULTI_ARCHIVE_DISCOVERY_SCHEMA,
        "generated_at_utc": _utc_now(),
        "archive_count": len(rows),
        "archive_globs": list(archive_globs),
        "rows": rows,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="repair_autonomous_multi_archive_discovery",
    )
    return payload


def _run_command(
    command: Sequence[str],
    *,
    repo_root: Path,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    result = subprocess.run(
        list(command),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    payload = {
        "schema": "repair_autonomous_multi_archive_subprocess_result.v1",
        "command": list(command),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="repair_autonomous_multi_archive_subprocess_result",
    )
    return payload


def _runtime_mapping(
    raw_values: Sequence[str | Path],
    *,
    repo_root: Path,
) -> tuple[dict[str, Path], Path | None]:
    mapping: dict[str, Path] = {}
    default: Path | None = None
    for raw in raw_values:
        text = str(raw)
        if "=" in text:
            label, path_text = text.split("=", 1)
            label = _slug(label)
            if label:
                mapping[label] = _resolve(path_text, repo_root)
            continue
        default = _resolve(text, repo_root)
    return mapping, default


def _runtime_for_row(
    row: Mapping[str, Any],
    *,
    runtime_by_label: Mapping[str, Path],
    default_runtime: Path | None,
) -> Path | None:
    if default_runtime is not None:
        return default_runtime
    haystack = " ".join(
        str(row.get(key) or "")
        for key in ("candidate_id", "typed_response_id", "source_candidate_id")
    ).lower()
    for label, runtime in runtime_by_label.items():
        if label and label in haystack:
            return runtime
    return None


def _source_queue_rows(source_queue_path: Path) -> list[Mapping[str, Any]]:
    payload = _load_json(source_queue_path)
    rows = payload.get("top_k")
    if not isinstance(rows, list):
        raise RepairAutonomousMultiArchiveRunnerError(
            "repair exact-ready source queue missing top_k rows"
        )
    return [row for row in rows if isinstance(row, Mapping)]


def close_multi_archive_submission_runtime_custody(
    *,
    source_queue_path: str | Path,
    source_runtime_dirs: Sequence[str | Path],
    output_dir: str | Path,
    repo_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build contest-shaped submission runtime packets for source-queue rows."""

    repo = Path(repo_root)
    source_queue = _resolve(source_queue_path, repo)
    output = _resolve(output_dir, repo)
    output.mkdir(parents=True, exist_ok=True)
    runtime_by_label, default_runtime = _runtime_mapping(source_runtime_dirs, repo_root=repo)
    rows = _source_queue_rows(source_queue)
    reports: list[dict[str, Any]] = []
    submission_dirs: list[str] = []
    blockers: list[str] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            blockers.append("candidate_id_missing_for_runtime_closure")
            continue
        runtime_dir = _runtime_for_row(
            row,
            runtime_by_label=runtime_by_label,
            default_runtime=default_runtime,
        )
        if runtime_dir is None:
            blockers.append(f"source_runtime_dir_missing_for_candidate:{candidate_id}")
            continue
        slug = _slug(candidate_id)
        candidate_dir = output / "submissions" / slug
        sidecar_dir = output / "sidecars" / slug
        try:
            report = build_materializer_submission_runtime_closure(
                repo_root=repo,
                source_queue_path=source_queue,
                candidate_id=candidate_id,
                source_runtime_dir=runtime_dir,
                submission_dir_out=candidate_dir,
                closed_source_queue_out=sidecar_dir / "closed_source_queue.json",
                closure_report_out=sidecar_dir / "submission_closure_report.json",
                overwrite=overwrite,
            )
        except (MaterializerSubmissionClosureError, OSError, ValueError) as exc:
            blockers.append(f"submission_runtime_closure_failed:{candidate_id}:{exc}")
            continue
        reports.append(report)
        submission_dir = str(report.get("submission_dir") or "")
        if submission_dir:
            submission_dirs.append(submission_dir)
    payload = {
        "schema": REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNTIME_CLOSURE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "source_queue_path": _repo_rel(source_queue, repo),
        "candidate_count": len(rows),
        "closure_report_count": len(reports),
        "submission_dirs": ordered_unique(submission_dirs),
        "rows": reports,
        "blockers": ordered_unique(blockers),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "multi_archive_submission_runtime_custody_static_closure",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        payload,
        context="repair_autonomous_multi_archive_runtime_closure",
    )
    return payload


def run_repair_autonomous_multi_archive_runner(
    *,
    archives: Sequence[str | Path] = (),
    archive_globs: Sequence[str] = (),
    source_labels: Sequence[str] = (),
    source_runtime_dirs: Sequence[str | Path] = (),
    output_dir: str | Path,
    repo_root: str | Path,
    max_archives: int | None = None,
    chain_id: str = "repair_multi_archive_autonomous",
    queue_id: str = "repair_multi_archive_materialization",
    execute_local: bool = False,
    close_runtime_custody: bool = False,
    max_steps_per_iteration: int = 128,
    worker_max_experiments_per_iteration: int | None = None,
    byte_credit_budget: int | None = None,
    posterior_path: str | Path | None = DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_PATH,
    posterior_lock_path: str | Path = DEFAULT_REPAIR_CAMPAIGN_STACKABILITY_POSTERIOR_LOCK_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run archive discovery, materialization, stack search, and runtime closure."""

    repo = Path(repo_root)
    output = _resolve(output_dir, repo)
    output.mkdir(parents=True, exist_ok=True)
    discovery = discover_repair_archive_candidates(
        archives=archives,
        archive_globs=archive_globs,
        repo_root=repo,
        source_labels=source_labels,
        max_archives=max_archives,
    )
    _write_json(output / "archive_discovery.json", discovery, overwrite=overwrite)
    archive_paths = [row["archive_path"] for row in discovery["rows"]]
    labels = [row["source_label"] for row in discovery["rows"]]
    training_paths = [row.get("training_artifact_path") for row in discovery["rows"]]
    gate_paths = [row.get("equivalence_gate_path") for row in discovery["rows"]]
    work_order = build_repair_campaign_work_order_from_archives(
        archive_paths=archive_paths,
        output_dir=output / "intake",
        repo_root=repo,
        source_labels=labels,
        training_artifact_paths=training_paths,
        equivalence_gate_paths=gate_paths,
        chain_id=chain_id,
        overwrite=overwrite,
    )
    work_order_path = output / "work_order.json"
    _write_json(work_order_path, work_order, overwrite=overwrite)
    score_report = score_repair_campaign(payload=work_order, repo_root=repo)
    score_report_path = output / "score_report.json"
    _write_json(score_report_path, score_report, overwrite=overwrite)
    materialization_queue = build_repair_campaign_byte_closed_materialization_queue(
        repo_root=repo,
        score_report=score_report,
        score_report_path=score_report_path,
        work_order_path=work_order_path,
        results_root=output / "queue_results",
        queue_id=queue_id,
    )
    materialization_queue_path = output / "repair_materialization_queue.json"
    _write_json(materialization_queue_path, materialization_queue, overwrite=overwrite)
    validation = _run_command(
        [
            sys.executable,
            str(repo / "tools" / "experiment_queue.py"),
            "--queue",
            str(materialization_queue_path),
            "validate",
        ],
        repo_root=repo,
    )
    if validation["returncode"] != 0:
        raise RepairAutonomousMultiArchiveRunnerError("materialization queue validation failed")
    posterior = None if posterior_path is None else _resolve(posterior_path, repo)
    posterior_lock = _resolve(posterior_lock_path, repo)
    worker_limit = worker_max_experiments_per_iteration
    if worker_limit is None:
        worker_limit = int(materialization_queue.get("metadata", {}).get("ready_experiment_count") or 1)
    floor_loop_dir = output / "floor_loop"
    floor_loop_summary_path = output / "floor_loop_summary.json"
    floor_command = [
        sys.executable,
        str(repo / "tools" / "run_repair_campaign_autonomous_floor_loop.py"),
        "--materialization-queue",
        str(materialization_queue_path),
        "--output-dir",
        str(floor_loop_dir),
        "--summary-out",
        str(floor_loop_summary_path),
        "--posterior-lock-path",
        str(posterior_lock),
        "--require-all-queue-families",
        "--worker-max-experiments-per-iteration",
        str(worker_limit),
        "--max-steps-per-iteration",
        str(max_steps_per_iteration),
        "--max-iterations",
        "1",
        "--overwrite",
    ]
    if posterior is not None:
        floor_command.extend(["--posterior-path", str(posterior)])
    if byte_credit_budget is not None:
        floor_command.extend(["--byte-credit-budget", str(byte_credit_budget)])
    if execute_local:
        floor_command.append("--execute-local")
    floor_loop_result = _run_command(floor_command, repo_root=repo, timeout_seconds=900)
    if floor_loop_result["returncode"] != 0:
        raise RepairAutonomousMultiArchiveRunnerError("floor loop command failed")
    floor_loop_summary = _load_json(floor_loop_summary_path)
    runtime_closure: dict[str, Any] | None = None
    runtime_floor_summary: dict[str, Any] | None = None
    runtime_floor_result: dict[str, Any] | None = None
    if close_runtime_custody:
        source_queue_path = floor_loop_dir / "repair_family_exact_ready_source_queue.json"
        runtime_closure = close_multi_archive_submission_runtime_custody(
            source_queue_path=source_queue_path,
            source_runtime_dirs=source_runtime_dirs,
            output_dir=output / "submission_runtime_closure",
            repo_root=repo,
            overwrite=overwrite,
        )
        runtime_closure_path = output / "submission_runtime_closure_report.json"
        _write_json(runtime_closure_path, runtime_closure, overwrite=overwrite)
        runtime_floor_dir = output / "floor_loop_runtime_custody"
        runtime_floor_summary_path = output / "floor_loop_runtime_custody_summary.json"
        runtime_command = [
            sys.executable,
            str(repo / "tools" / "run_repair_campaign_autonomous_floor_loop.py"),
            "--materialization-queue",
            str(materialization_queue_path),
            "--output-dir",
            str(runtime_floor_dir),
            "--summary-out",
            str(runtime_floor_summary_path),
            "--posterior-lock-path",
            str(posterior_lock),
            "--require-all-queue-families",
            "--max-iterations",
            "1",
            "--overwrite",
        ]
        if posterior is not None:
            runtime_command.extend(["--posterior-path", str(posterior)])
        for submission_dir in runtime_closure.get("submission_dirs") or []:
            runtime_command.extend(["--submission-dir", str(submission_dir)])
        runtime_floor_result = _run_command(
            runtime_command,
            repo_root=repo,
            timeout_seconds=900,
        )
        if runtime_floor_result["returncode"] != 0:
            raise RepairAutonomousMultiArchiveRunnerError(
                "runtime-custody floor loop command failed"
            )
        runtime_floor_summary = _load_json(runtime_floor_summary_path)
    active_summary = runtime_floor_summary or floor_loop_summary
    floor_posterior_appended = int(floor_loop_summary.get("posterior_appended_count") or 0)
    runtime_posterior_appended = (
        0
        if runtime_floor_summary is None
        else int(runtime_floor_summary.get("posterior_appended_count") or 0)
    )
    runner_summary = {
        "schema": REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNNER_SCHEMA,
        "generated_at_utc": _utc_now(),
        "output_dir": _repo_rel(output, repo),
        "archive_discovery_path": _repo_rel(output / "archive_discovery.json", repo),
        "archive_count": discovery["archive_count"],
        "archive_rows": discovery["rows"],
        "work_order_path": _repo_rel(work_order_path, repo),
        "typed_response_count": len(work_order.get("typed_response_ledger", {}).get("rows") or []),
        "score_report_path": _repo_rel(score_report_path, repo),
        "selected_allocation_count": score_report.get("optimizer_decision", {}).get("selected_allocation_count"),
        "materialization_queue_path": _repo_rel(materialization_queue_path, repo),
        "ready_experiment_count": materialization_queue.get("metadata", {}).get("ready_experiment_count"),
        "queue_validation": validation,
        "floor_loop_command": floor_command,
        "floor_loop_result": floor_loop_result,
        "floor_loop_summary_path": _repo_rel(floor_loop_summary_path, repo),
        "floor_loop_stop_reason": floor_loop_summary.get("stop_reason"),
        "runtime_custody_requested": close_runtime_custody,
        "runtime_closure": runtime_closure,
        "runtime_floor_result": runtime_floor_result,
        "runtime_floor_summary_path": (
            None
            if runtime_floor_summary is None
            else _repo_rel(output / "floor_loop_runtime_custody_summary.json", repo)
        ),
        "exact_eval_handoff_candidate_count": active_summary.get("exact_eval_handoff_candidate_count"),
        "archive_bound_exact_handoff_candidate_count": active_summary.get("archive_bound_exact_handoff_candidate_count"),
        "exact_ready_bridge_candidate_count": active_summary.get("exact_ready_bridge_candidate_count"),
        "exact_ready_bridge_runtime_content_tree_custody_proven_count": active_summary.get(
            "exact_ready_bridge_runtime_content_tree_custody_proven_count"
        ),
        "posterior_learning_signal_count": active_summary.get("posterior_learning_signal_count"),
        "floor_loop_posterior_appended_count": floor_posterior_appended,
        "runtime_floor_posterior_appended_count": runtime_posterior_appended,
        "posterior_appended_count": floor_posterior_appended + runtime_posterior_appended,
        "posterior_appended_count_total": floor_posterior_appended + runtime_posterior_appended,
        "stop_reason": active_summary.get("stop_reason"),
        "primary_stack_acquisition_terminal_outcome": active_summary.get(
            "primary_stack_acquisition_terminal_outcome"
        ),
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "allowed_use": "multi_archive_repair_campaign_local_planning_and_exact_readiness_custody_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        runner_summary,
        context="repair_autonomous_multi_archive_runner_summary",
    )
    _write_json(output / "runner_summary.json", runner_summary, overwrite=overwrite)
    return runner_summary


__all__ = [
    "REPAIR_AUTONOMOUS_MULTI_ARCHIVE_DISCOVERY_SCHEMA",
    "REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNNER_SCHEMA",
    "REPAIR_AUTONOMOUS_MULTI_ARCHIVE_RUNTIME_CLOSURE_SCHEMA",
    "RepairAutonomousMultiArchiveRunnerError",
    "close_multi_archive_submission_runtime_custody",
    "discover_repair_archive_candidates",
    "run_repair_autonomous_multi_archive_runner",
]
