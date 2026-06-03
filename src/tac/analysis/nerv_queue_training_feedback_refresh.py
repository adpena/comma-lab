# SPDX-License-Identifier: MIT
"""Refresh NeRV training feedback directly from queue-owned telemetry.

This is the durable bridge from long-running MLX queue rows back into the
planner.  It deliberately produces false-authority candidate feedback rows:
live telemetry may steer the next launch recipe, but it never grants archive,
receiver, local replay, or exact-eval authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.analysis.nerv_candidate_feedback import (
    write_nerv_training_telemetry_feedback_files,
)
from tac.repo_io import sha256_file, write_json_artifact, write_text_artifact

SCHEMA = "nerv_queue_training_feedback_refresh.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
_TRAINING_RUNNING_STOP_REASON = "training_running_midrun_feedback_snapshot"


class NervQueueTrainingFeedbackRefreshError(ValueError):
    """Raised when queue-owned telemetry cannot be refreshed safely."""


def refresh_nerv_queue_training_feedback(
    *,
    queue: Mapping[str, Any],
    queue_path: str | Path,
    queue_summary: Mapping[str, Any],
    output_dir: str | Path,
    include_statuses: Sequence[str] = ("running",),
) -> dict[str, Any]:
    """Harvest planner-consumable feedback rows from queue telemetry artifacts."""

    qpath = Path(queue_path).expanduser().resolve(strict=False)
    out_root = Path(output_dir).expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    status_map = _status_map(queue_summary)
    include = {str(status) for status in include_statuses}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for experiment in queue.get("experiments") or []:
        if not isinstance(experiment, Mapping):
            continue
        for step in experiment.get("steps") or []:
            if not isinstance(step, Mapping):
                continue
            step_id = str(step.get("id") or "")
            experiment_id = str(experiment.get("id") or "")
            command = [str(item) for item in (step.get("command") or [])]
            family = _family(
                experiment=experiment,
                command=command,
                experiment_id=experiment_id,
            )
            if family not in {"hi_nerv", "hinerv", "snerv"}:
                skipped.append(
                    {
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "reason": "not_nerv_training_family",
                    }
                )
                continue
            state = status_map.get((experiment_id, step_id), {})
            status = str(state.get("status") or "queued")
            if status not in include:
                skipped.append(
                    {
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "status": status,
                        "reason": "status_not_included",
                    }
                )
                continue
            telemetry = _telemetry_path(step=step, command=command, family=family)
            if telemetry is None or not telemetry.is_file():
                skipped.append(
                    {
                        "experiment_id": experiment_id,
                        "step_id": step_id,
                        "status": status,
                        "reason": "telemetry_missing",
                        "telemetry_path": None if telemetry is None else telemetry.as_posix(),
                    }
                )
                continue
            candidate_id = _candidate_id(command=command, experiment_id=experiment_id)
            candidate_pairs = _int_arg(command, "--num-pairs", default=600)
            row_dir = out_root / _safe_token(experiment_id)
            result = write_nerv_training_telemetry_feedback_files(
                telemetry_path=telemetry,
                output_dir=row_dir,
                family=family,
                candidate_id=candidate_id,
                candidate_num_pairs=candidate_pairs,
                source_queue_path=qpath,
                stop_reason=(
                    _TRAINING_RUNNING_STOP_REASON
                    if status == "running"
                    else f"queue_step_status_{status}"
                ),
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "step_id": step_id,
                    "status": status,
                    "family": family,
                    "candidate_id": candidate_id,
                    "telemetry_path": telemetry.as_posix(),
                    "row_path": result["row_path"],
                    "manifest_path": result["manifest_path"],
                    "row": result["row"],
                }
            )
    return {
        "schema": SCHEMA,
        "queue_id": queue.get("queue_id"),
        "queue_path": qpath.as_posix(),
        "queue_sha256": sha256_file(qpath),
        "queue_summary_schema": queue_summary.get("schema"),
        "included_statuses": sorted(include),
        "refreshed_row_count": len(rows),
        "skipped_count": len(skipped),
        "rows": rows,
        "skipped": skipped,
        **FALSE_AUTHORITY,
    }


def write_nerv_queue_training_feedback_refresh(
    *,
    report: Mapping[str, Any],
    output_json: str | Path,
    output_jsonl: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    """Write a refresh report plus row ledger for downstream planner ingestion."""

    artifact = write_json_artifact(output_json, dict(report))
    result = {
        "schema": "nerv_queue_training_feedback_refresh_write.v1",
        "report_path": artifact.path,
        "report_sha256": artifact.sha256,
        "report_bytes": artifact.bytes_written,
        "row_count": int(report.get("refreshed_row_count") or 0),
        **FALSE_AUTHORITY,
    }
    if output_jsonl is not None:
        lines = [
            json.dumps(row.get("row") or {}, sort_keys=True)
            for row in report.get("rows") or []
            if isinstance(row, Mapping)
        ]
        ledger = write_text_artifact(
            output_jsonl,
            "".join(f"{line}\n" for line in lines),
        )
        result["jsonl_path"] = ledger.path
        result["jsonl_sha256"] = ledger.sha256
        result["jsonl_bytes"] = ledger.bytes_written
    if output_md is not None:
        md = write_text_artifact(output_md, render_refresh_markdown(report))
        result["markdown_path"] = md.path
        result["markdown_sha256"] = md.sha256
        result["markdown_bytes"] = md.bytes_written
    return result


def render_refresh_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# NeRV Queue Training Feedback Refresh",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Queue: `{report.get('queue_path')}`",
        f"Rows refreshed: `{report.get('refreshed_row_count')}`",
        f"Skipped: `{report.get('skipped_count')}`",
        f"Score claim: `{report.get('score_claim')}`",
        f"Ready for exact dispatch: `{report.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Rows",
        "",
    ]
    for row in report.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        feedback = dict(row.get("row") or {})
        lines.extend(
            [
                f"- `{row.get('experiment_id')}`",
                f"  status: `{row.get('status')}`",
                f"  candidate: `{row.get('candidate_id')}`",
                f"  last_epoch: `{dict(feedback.get('training_telemetry') or {}).get('last_epoch')}`",
                f"  pose_tail_burst: `{feedback.get('pose_tail_burst_detected')}`",
                f"  pose_tail_recent_p95: `{feedback.get('pose_tail_burst_recent_p95')}`",
                f"  observed_segnet_weight: `{feedback.get('observed_segnet_distillation_weight')}`",
                f"  recommended_segnet_weight: `{feedback.get('recommended_segnet_distillation_weight')}`",
                f"  training_control_action: `{feedback.get('training_control_action')}`",
                f"  training_control_should_stop: `{feedback.get('training_control_should_stop_current_run')}`",
                f"  row: `{row.get('row_path')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _status_map(summary: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary.get("steps") or []:
        if not isinstance(row, Mapping):
            continue
        out[(str(row.get("experiment_id") or ""), str(row.get("step_id") or ""))] = dict(row)
    return out


def _telemetry_path(
    *,
    step: Mapping[str, Any],
    command: Sequence[str],
    family: str,
) -> Path | None:
    telemetry = step.get("telemetry")
    if isinstance(telemetry, Mapping):
        for value in telemetry.get("artifact_paths") or []:
            path = Path(str(value))
            if path.name == "telemetry.jsonl":
                return path
    output_dir = _output_dir(command)
    if output_dir is None:
        return None
    if family in {"hi_nerv", "hinerv"}:
        return output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl"
    if family == "snerv":
        return output_dir / "snerv_mlx_training" / "telemetry.jsonl"
    return None


def _family(
    *,
    experiment: Mapping[str, Any],
    command: Sequence[str],
    experiment_id: str,
) -> str:
    value = str(experiment.get("family") or "").strip()
    if value:
        return value
    value = _arg_value(command, "--execute-family")
    if value:
        return "hi_nerv" if value == "hinerv" else value
    if experiment_id.startswith("hi_nerv_") or experiment_id.startswith("hinerv_"):
        return "hi_nerv"
    if experiment_id.startswith("snerv_"):
        return "snerv"
    return ""


def _output_dir(command: Sequence[str]) -> Path | None:
    value = _arg_value(command, "--output-dir")
    return None if value is None else Path(value)


def _candidate_id(*, command: Sequence[str], experiment_id: str) -> str:
    for flag in (
        "--modelsize-candidate-id",
        "--snerv-modelsize-candidate-id",
        "--candidate-id",
    ):
        value = _arg_value(command, flag)
        if value:
            return value
    return experiment_id


def _int_arg(command: Sequence[str], flag: str, *, default: int) -> int:
    value = _arg_value(command, flag)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _arg_value(command: Sequence[str], flag: str) -> str | None:
    argv = [str(item) for item in command]
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


__all__ = [
    "SCHEMA",
    "NervQueueTrainingFeedbackRefreshError",
    "refresh_nerv_queue_training_feedback",
    "render_refresh_markdown",
    "write_nerv_queue_training_feedback_refresh",
]
