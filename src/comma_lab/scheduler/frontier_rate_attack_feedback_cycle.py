# SPDX-License-Identifier: MIT
"""Queue-owned frontier feedback cycle helpers.

This module keeps the frontier-rate attack loop out of chat glue:

materializer/local observations -> bounded DQS1 queue -> harvest observations ->
refreshed queue.  All artifacts are planning signal only unless a separate exact
auth path later grants authority.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.dqs1_local_first_harvest_observations import (
    build_harvest_observation_summary,
    build_observation_rows_from_harvests,
    render_markdown_summary,
    write_observation_jsonl,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import (
    ArtifactWriteError,
    json_text,
    write_json_artifact,
    write_text_artifact,
)

from .experiment_queue import ExperimentQueueError

FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA = "frontier_rate_attack_feedback_cycle.v1"
FRONTIER_RATE_ATTACK_DQS1_OBSERVATION_BUNDLE_SCHEMA = (
    "frontier_rate_attack_dqs1_observation_bundle.v1"
)
AUTOPILOT_RESULT_SCHEMA = "dqs1_local_first_autopilot_result.v1"


class FrontierRateAttackFeedbackCycleError(ExperimentQueueError):
    """Raised when a feedback cycle would lose signal or imply false authority."""


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def artifact_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def repo_rel(path: str | Path, repo_root: str | Path) -> str:
    value = Path(path)
    repo = Path(repo_root)
    try:
        return value.resolve(strict=False).relative_to(repo.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def resolve_repo_path(path: str | Path, *, repo_root: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path(repo_root) / value
    return value.resolve(strict=False)


def load_json_object(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierRateAttackFeedbackCycleError(
            f"{target}: could not load JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackFeedbackCycleError(f"{target}: expected JSON object")
    return payload


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.resolve(strict=False).as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return tuple(out)


def harvest_paths_from_autopilot_payload(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> tuple[Path, ...]:
    """Extract append-only DQS1 harvest JSON paths from an autopilot result."""

    if payload.get("schema") != AUTOPILOT_RESULT_SCHEMA:
        raise FrontierRateAttackFeedbackCycleError(
            f"autopilot result schema must be {AUTOPILOT_RESULT_SCHEMA}"
        )
    try:
        require_no_truthy_authority_fields(
            payload,
            context="frontier_rate_attack_feedback_cycle.autopilot_result",
        )
    except ValueError as exc:
        raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
    rounds = payload.get("rounds")
    if not isinstance(rounds, list):
        raise FrontierRateAttackFeedbackCycleError("autopilot result rounds must be a list")
    paths: list[Path] = []
    for round_index, round_row in enumerate(rounds):
        if not isinstance(round_row, Mapping):
            continue
        harvests = round_row.get("harvests")
        if harvests is None and isinstance(round_row.get("harvest"), Mapping):
            harvests = [round_row["harvest"]]
        if not isinstance(harvests, list):
            continue
        for harvest_index, harvest_row in enumerate(harvests):
            if not isinstance(harvest_row, Mapping):
                continue
            try:
                require_no_truthy_authority_fields(
                    harvest_row,
                    context=(
                        "frontier_rate_attack_feedback_cycle.autopilot_harvest"
                        f"[{round_index}][{harvest_index}]"
                    ),
                )
            except ValueError as exc:
                raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
            raw_path = harvest_row.get("harvest_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = resolve_repo_path(raw_path, repo_root=repo_root)
            if not path.is_file():
                raise FrontierRateAttackFeedbackCycleError(
                    f"autopilot harvest path not found: {path}"
                )
            paths.append(path)
    return _unique_paths(paths)


def harvest_paths_from_autopilot_result_files(
    paths: Sequence[str | Path],
    *,
    repo_root: str | Path,
) -> tuple[Path, ...]:
    """Load autopilot result files and return unique harvest JSON paths."""

    collected: list[Path] = []
    for value in paths:
        path = resolve_repo_path(value, repo_root=repo_root)
        collected.extend(
            harvest_paths_from_autopilot_payload(
                load_json_object(path),
                repo_root=repo_root,
            )
        )
    return _unique_paths(collected)


def write_frontier_refresh_artifacts(
    *,
    output_dir: str | Path,
    report: Mapping[str, Any],
    repo_root: str | Path,
    report_filename: str = "feedback_refresh_report.json",
    queue_filename: str = "dqs1_followup_queue.json",
) -> dict[str, str]:
    """Write the standard frontier feedback refresh artifact family."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    discovery = report.get("discovery")
    if isinstance(discovery, Mapping):
        path = out / "materializer_feedback_discovery.json"
        write_json_artifact(path, dict(discovery))
        artifacts["materializer_feedback_discovery"] = repo_rel(path, repo_root)
    bridge = report.get("materializer_feedback_bridge")
    if isinstance(bridge, Mapping):
        path = out / "materializer_feedback_bridge.json"
        write_json_artifact(path, dict(bridge))
        artifacts["materializer_feedback_bridge"] = repo_rel(path, repo_root)
    queue = report.get("queue")
    if isinstance(queue, Mapping):
        path = out / queue_filename
        write_json_artifact(path, dict(queue))
        artifacts["dqs1_followup_queue"] = repo_rel(path, repo_root)

    report_path = out / report_filename
    report_to_write = dict(report)
    report_to_write["artifacts"] = dict(artifacts)
    if "dqs1_followup_queue" in artifacts:
        report_to_write["operator_commands"] = {
            "validate_followup_queue": [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                artifacts["dqs1_followup_queue"],
                "validate",
            ],
            "run_frontier_feedback_cycle": [
                ".venv/bin/python",
                "tools/run_frontier_rate_attack_feedback_cycle.py",
                "--frontier-artifact-root",
                ".omx/research",
                "--action-summary",
                str(report.get("action_summary_path") or "latest"),
                "--results-root",
                str(report.get("results_root") or ""),
            ],
        }
    write_json_artifact(report_path, report_to_write)
    artifacts["feedback_refresh_report"] = repo_rel(report_path, repo_root)
    return artifacts


def write_dqs1_harvest_observation_bundle(
    *,
    harvest_paths: Sequence[str | Path],
    repo_root: str | Path,
    pairset_acquisition_path: str | Path,
    baseline_advisory_path: str | Path,
    baseline_archive_size_bytes: int,
    baseline_candidate_id: str,
    output_dir: str | Path,
    stamp: str | None = None,
) -> dict[str, Any]:
    """Canonicalize harvested DQS1 local results into dynamic observation rows."""

    repo = Path(repo_root)
    resolved_harvests = _unique_paths(
        resolve_repo_path(path, repo_root=repo) for path in harvest_paths
    )
    if not resolved_harvests:
        raise FrontierRateAttackFeedbackCycleError(
            "at least one DQS1 harvest path is required"
        )
    missing = [path.as_posix() for path in resolved_harvests if not path.is_file()]
    if missing:
        raise FrontierRateAttackFeedbackCycleError(
            "missing DQS1 harvest JSON: " + ", ".join(missing)
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_stamp = stamp or utc_stamp()
    jsonl_path = out / f"dqs1_local_first_harvest_observations_{run_stamp}.jsonl"
    summary_path = jsonl_path.with_suffix(".summary.json")
    markdown_path = jsonl_path.with_suffix(".md")
    rows = build_observation_rows_from_harvests(
        resolved_harvests,
        repo_root=repo,
        pairset_acquisition_path=resolve_repo_path(
            pairset_acquisition_path,
            repo_root=repo,
        ),
        baseline_advisory_path=resolve_repo_path(
            baseline_advisory_path,
            repo_root=repo,
        ),
        baseline_archive_size_bytes=baseline_archive_size_bytes,
        baseline_candidate_id=baseline_candidate_id,
    )
    write_observation_jsonl(rows, output_path=jsonl_path)
    summary = build_harvest_observation_summary(
        rows,
        jsonl_path=jsonl_path,
        repo_root=repo,
    )
    write_json_artifact(summary_path, summary)
    write_text_artifact(markdown_path, render_markdown_summary(summary))
    return {
        "schema": FRONTIER_RATE_ATTACK_DQS1_OBSERVATION_BUNDLE_SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harvest_paths": [repo_rel(path, repo) for path in resolved_harvests],
        "observation_jsonl": repo_rel(jsonl_path, repo),
        "observation_summary_json": repo_rel(summary_path, repo),
        "observation_markdown": repo_rel(markdown_path, repo),
        "row_count": len(rows),
        "best_local_advisory": summary.get("best_local_advisory"),
        **FALSE_AUTHORITY,
    }


def write_cycle_report(
    *,
    output_dir: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write the append-only cycle report."""

    path = Path(output_dir) / "frontier_rate_attack_feedback_cycle.json"
    try:
        write_json_artifact(path, dict(payload))
    except ArtifactWriteError as exc:
        raise FrontierRateAttackFeedbackCycleError(str(exc)) from exc
    return path


__all__ = [
    "AUTOPILOT_RESULT_SCHEMA",
    "FRONTIER_RATE_ATTACK_DQS1_OBSERVATION_BUNDLE_SCHEMA",
    "FRONTIER_RATE_ATTACK_FEEDBACK_CYCLE_SCHEMA",
    "FrontierRateAttackFeedbackCycleError",
    "artifact_token",
    "harvest_paths_from_autopilot_payload",
    "harvest_paths_from_autopilot_result_files",
    "json_text",
    "load_json_object",
    "repo_rel",
    "resolve_repo_path",
    "utc_stamp",
    "write_cycle_report",
    "write_dqs1_harvest_observation_bundle",
    "write_frontier_refresh_artifacts",
]
