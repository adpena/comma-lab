# SPDX-License-Identifier: MIT
"""Execution-admission bridge for HiNeRV/SNeRV long-training campaigns.

The campaign planner and Cathedral consumer deliberately stop at local MLX
recommendations. This module turns a vetted consumer verdict into a scheduler
queue only after the selected rows have SSD-backed output roots, an active lane
claim, storage preflight, and false-authority custody. It is not an executor
and never unlocks score, CPU replay, exact auth, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.experiment_queue import (
    QUEUE_SCHEMA,
    normalize_queue_definition,
)
from comma_lab.scheduler.storage_preflight import (
    build_scheduler_storage_preflight_experiment,
)
from tac.deploy.claims import active_claim_row
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
)

CONSUMER_RESULT_SCHEMA = "nerv_long_training_campaign_consumer_result.v1"
ADMISSION_SCHEMA = "nerv_long_training_campaign_execution_admission.v1"
ADMITTED_EXPERIMENT_SCHEMA = "nerv_long_training_campaign_admitted_experiment.v1"
DEFAULT_QUEUE_ID = "nerv_manifest_pinned_long_training_local_mlx_admission.v1"
DEFAULT_STORAGE_EXPECTED_BYTES_PER_ROW = 8 * 1024**3
DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS = 12 * 60 * 60
DEFAULT_ALLOWED_OUTPUT_ROOTS = (
    "/Volumes/VertigoDataTier/pact",
    "/Volumes/APDataStore/pact",
)
_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "frontier_score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "promotable",
    "ready_for_exact_eval_dispatch",
    "ready_for_provider_dispatch",
    "dispatch_attempted",
    "gpu_launched",
    "exact_cuda_auth_eval",
    "contest_cuda_auth_eval",
)


class NervLongTrainingCampaignAdmissionError(ValueError):
    """Raised when a campaign verdict cannot be admitted for local execution."""


def build_nerv_long_training_campaign_execution_admission(
    consumer_verdict: Mapping[str, Any],
    *,
    repo_root: str | Path,
    active_claims_path: str | Path,
    lane_id: str,
    instance_job_id: str,
    queue_id: str = DEFAULT_QUEUE_ID,
    limit: int = 1,
    selected_experiment_ids: Sequence[str] = (),
    storage_expected_bytes_per_row: int = DEFAULT_STORAGE_EXPECTED_BYTES_PER_ROW,
    storage_reserve_free_gb: float = 40.0,
    local_mlx_timeout_seconds: int = DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS,
    allowed_output_roots: Sequence[str | Path] = DEFAULT_ALLOWED_OUTPUT_ROOTS,
    now_utc: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed local-MLX execution-admission artifact.

    ``consumer_verdict`` must come from
    :mod:`tac.cathedral_consumers.nerv_long_training_campaign_consumer`.
    The returned payload embeds an ``experiment_queue.v1`` only when the active
    claim and SSD-output checks pass. The queue itself starts with the reusable
    storage/cleanup preflight and then runs the selected local MLX rows.
    """

    repo = Path(repo_root).expanduser().resolve(strict=False)
    if consumer_verdict.get("schema") != CONSUMER_RESULT_SCHEMA:
        raise NervLongTrainingCampaignAdmissionError(
            f"consumer verdict schema must be {CONSUMER_RESULT_SCHEMA}"
        )
    _require_no_authority(consumer_verdict, label="consumer_verdict")
    if isinstance(limit, bool) or int(limit) < 1:
        raise NervLongTrainingCampaignAdmissionError("limit must be >= 1")
    if (
        isinstance(storage_expected_bytes_per_row, bool)
        or int(storage_expected_bytes_per_row) < 0
    ):
        raise NervLongTrainingCampaignAdmissionError(
            "storage_expected_bytes_per_row must be non-negative"
        )
    if isinstance(local_mlx_timeout_seconds, bool) or int(local_mlx_timeout_seconds) <= 0:
        raise NervLongTrainingCampaignAdmissionError(
            "local_mlx_timeout_seconds must be positive"
        )

    generated_at = now_utc or _utc_now()
    blockers: list[str] = []
    if consumer_verdict.get("local_mlx_route_recommended") is not True:
        blockers.append("consumer_did_not_recommend_local_mlx_route")

    selected_rows, selection_blockers = _select_rows(
        consumer_verdict,
        limit=int(limit),
        selected_experiment_ids=selected_experiment_ids,
    )
    blockers.extend(selection_blockers)

    claim_row: dict[str, str] | None = None
    try:
        claim_row = active_claim_row(
            _resolve_path(active_claims_path, repo_root=repo),
            lane_id=lane_id,
            instance_job_id=instance_job_id,
        )
    except ValueError as exc:
        blockers.append("active_lane_claim_missing_or_terminal")
        blockers.append(_safe_blocker_text(str(exc)))
    if claim_row is not None and claim_row.get("platform") != "local_mlx":
        blockers.append("active_lane_claim_platform_not_local_mlx")

    allowed_roots = tuple(
        Path(root).expanduser().resolve(strict=False) for root in allowed_output_roots
    )
    row_records: list[dict[str, Any]] = []
    admitted_rows: list[Mapping[str, Any]] = []
    output_dirs: list[Path] = []
    for row in selected_rows:
        record = _row_record(row, allowed_roots=allowed_roots)
        row_records.append(record)
        blockers.extend(record["blockers"])
        if record["admitted"]:
            admitted_rows.append(row)
            output_dirs.append(Path(record["output_dir"]).resolve(strict=False))

    output_parent = _common_output_parent(output_dirs)
    if output_parent is None and selected_rows:
        blockers.append("selected_output_parent_missing")
    storage_expected_bytes = int(storage_expected_bytes_per_row) * len(admitted_rows)

    queue: dict[str, Any] | None = None
    if not blockers and admitted_rows and output_parent is not None:
        queue = _execution_queue(
            admitted_rows,
            repo_root=repo,
            queue_id=queue_id,
            lane_id=lane_id,
            generated_at_utc=generated_at,
            output_parent=output_parent,
            storage_expected_bytes=storage_expected_bytes,
            storage_reserve_free_gb=storage_reserve_free_gb,
            local_mlx_timeout_seconds=int(local_mlx_timeout_seconds),
            source_verdict=consumer_verdict,
            claim_row=claim_row or {},
        )

    exact_dispatch_dependencies = [
        "PR95_same_axis_control_replay_required_before_beat_claim",
        "PR101_and_Z5_terminal_adjudication_required_before_new_exact_full_video_cuda",
        "full600_byte_closed_receiver_proof_required_before_promotion",
        "paired_contest_cpu_cuda_pass_required_before_promotion",
    ]
    return {
        "schema": ADMISSION_SCHEMA,
        "generated_at_utc": generated_at,
        "source_verdict": _source_record(consumer_verdict),
        "queue_id": queue_id,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "claim_row": claim_row,
        "claim_verified_active": claim_row is not None and not any(
            blocker.startswith("active_lane_claim_") for blocker in blockers
        ),
        "selection_limit": int(limit),
        "selected_experiment_ids_requested": list(selected_experiment_ids),
        "selected_row_count": len(selected_rows),
        "admitted_experiment_count": len(admitted_rows) if queue is not None else 0,
        "selected_rows": row_records,
        "storage_expected_bytes_per_row": int(storage_expected_bytes_per_row),
        "storage_expected_bytes": storage_expected_bytes,
        "storage_reserve_free_gb": float(storage_reserve_free_gb),
        "local_mlx_timeout_seconds": int(local_mlx_timeout_seconds),
        "output_parent": None if output_parent is None else output_parent.as_posix(),
        "allowed_output_roots": [root.as_posix() for root in allowed_roots],
        "experiment_queue": queue,
        "experiment_queue_ready": queue is not None,
        "local_mlx_execution_ready": queue is not None,
        "local_cpu_replay_recommended": False,
        "exact_auth_recommended": False,
        "blocked_exact_dispatch_dependencies": exact_dispatch_dependencies,
        "axis_tag": "[planning/control]",
        "rationale": (
            "Admission is limited to local MLX acquisition rows selected by the "
            "NeRV long-training campaign consumer. Storage and cleanup preflight "
            "are first-class scheduler steps; score/promotion authority stays "
            "closed until receiver proof and same-axis PR95/CPU/CUDA gates pass."
        ),
        "blockers": _dedupe(blockers),
        **PROXY_FALSE_AUTHORITY_FIELDS,
        "frontier_score_claim": False,
    }


def render_nerv_long_training_campaign_execution_admission_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing admission summary."""

    lines = [
        "# NeRV Long-Training Campaign Execution Admission",
        "",
        f"Schema: `{payload.get('schema')}`",
        f"Queue ready: `{payload.get('experiment_queue_ready')}`",
        f"Local MLX execution ready: `{payload.get('local_mlx_execution_ready')}`",
        f"Admitted experiments: `{payload.get('admitted_experiment_count')}`",
        f"Lane: `{payload.get('lane_id')}`",
        f"Job: `{payload.get('instance_job_id')}`",
        f"Score claim: `{payload.get('score_claim')}`",
        "",
        "## Selected Rows",
        "",
    ]
    rows = payload.get("selected_rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"- `{row.get('id')}` family=`{row.get('family')}` "
                f"admitted=`{row.get('admitted')}` output=`{row.get('output_dir')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    blockers = payload.get("blockers")
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)) and blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _execution_queue(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    queue_id: str,
    lane_id: str,
    generated_at_utc: str,
    output_parent: Path,
    storage_expected_bytes: int,
    storage_reserve_free_gb: float,
    local_mlx_timeout_seconds: int,
    source_verdict: Mapping[str, Any],
    claim_row: Mapping[str, str],
) -> dict[str, Any]:
    date = generated_at_utc.replace("-", "").replace(":", "").replace("Z", "Z")
    artifact_prefix = "nerv_long_training_campaign_admission"
    storage_experiment = build_scheduler_storage_preflight_experiment(
        experiment_id="nerv_campaign_storage_preflight",
        lane_id=lane_id,
        tags=["nerv", "snerv", "hinerv", "local-mlx", "storage-preflight"],
        artifact_prefix=artifact_prefix,
        date=date,
        results_root=output_parent.as_posix(),
        repo_root=repo_root,
        storage_expected_workload_root=output_parent.as_posix(),
        storage_expected_bytes=int(storage_expected_bytes),
        storage_reserve_free_gb=float(storage_reserve_free_gb),
        lifecycle_kind="HISTORICAL_PROVENANCE",
    )
    experiments = [storage_experiment]
    experiments.extend(
        _admitted_experiment(
            row,
            lane_id=lane_id,
            requires="nerv_campaign_storage_preflight.proactive_cleanup",
            local_mlx_timeout_seconds=int(local_mlx_timeout_seconds),
        )
        for row in rows
    )
    queue = {
        "schema": QUEUE_SCHEMA,
        "queue_id": queue_id,
        "controls": {
            "mode": "running",
            "local_first": True,
            "max_concurrency": {"local_mlx": 1, "local_io_heavy": 1, "local_cpu": 1},
        },
        "metadata": {
            "schema": "nerv_long_training_campaign_admission_queue_metadata.v1",
            "generated_at_utc": generated_at_utc,
            "source_verdict": _source_record(source_verdict),
            "lane_claim": dict(claim_row),
            "blocked_exact_dispatch_dependencies": [
                "PR95_same_axis_control_replay_required_before_beat_claim",
                "PR101_and_Z5_terminal_adjudication_required_before_new_exact_full_video_cuda",
                "full600_byte_closed_receiver_proof_required_before_promotion",
                "paired_contest_cpu_cuda_pass_required_before_promotion",
            ],
            **PROXY_FALSE_AUTHORITY_FIELDS,
            "frontier_score_claim": False,
        },
        "experiments": experiments,
        **PROXY_FALSE_AUTHORITY_FIELDS,
        "frontier_score_claim": False,
    }
    return normalize_queue_definition(queue)


def _admitted_experiment(
    row: Mapping[str, Any],
    *,
    lane_id: str,
    requires: str,
    local_mlx_timeout_seconds: int,
) -> dict[str, Any]:
    command = [str(item) for item in _list(row.get("command"))]
    report_path = _output_report_path(command)
    artifact_paths = _observable_artifact_paths(
        command,
        family=str(row.get("family") or "unknown"),
    )
    postconditions = [dict(item) for item in _mapping_list(row.get("postconditions"))]
    if not postconditions and report_path:
        postconditions = [
            {
                "type": "json_equals",
                "path": report_path,
                "key": "ready_for_exact_eval_dispatch",
                "equals": False,
            }
        ]
    return {
        "id": str(row.get("id") or "nerv_local_mlx_campaign_row"),
        "status": "queued",
        "priority": int(row.get("priority") or 100),
        "lane_id": lane_id,
        "tags": ["nerv", str(row.get("family") or "unknown"), "local-mlx", "false-authority"],
        "metadata": {
            "schema": ADMITTED_EXPERIMENT_SCHEMA,
            "family": str(row.get("family") or "unknown"),
            "source_selected_row": dict(row),
            "human_visual_fidelity_relevance": "irrelevant_unless_scorer_causal",
            **PROXY_FALSE_AUTHORITY_FIELDS,
            "frontier_score_claim": False,
        },
        "steps": [
            {
                "id": "run_mlx_first_campaign_row",
                "requires": [requires],
                "command": command,
                "resources": {
                    "kind": "local_mlx",
                    "max_parallel_group": "local_mlx_training",
                },
                "postconditions": postconditions,
                "on_postcondition_failure": "failed",
                "timeout_seconds": int(local_mlx_timeout_seconds),
                "telemetry": {
                    "artifact_paths": artifact_paths,
                    "include_postcondition_paths": True,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _select_rows(
    verdict: Mapping[str, Any],
    *,
    limit: int,
    selected_experiment_ids: Sequence[str],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    rows = _mapping_list(verdict.get("selected_local_mlx_experiments"))
    blockers: list[str] = []
    if not rows:
        return [], ["selected_local_mlx_experiments_missing"]
    requested = {str(value) for value in selected_experiment_ids if str(value)}
    if requested:
        rows = [row for row in rows if str(row.get("id") or "") in requested]
        missing = sorted(requested - {str(row.get("id") or "") for row in rows})
        blockers.extend(f"requested_experiment_id_missing:{item}" for item in missing)
    return list(rows[:limit]), blockers


def _row_record(
    row: Mapping[str, Any],
    *,
    allowed_roots: Sequence[Path],
) -> dict[str, Any]:
    blockers: list[str] = []
    row_id = str(row.get("id") or "")
    family = str(row.get("family") or "unknown")
    try:
        _require_no_authority(row, label=f"selected_row:{row_id or 'unknown'}")
    except NervLongTrainingCampaignAdmissionError as exc:
        blockers.append(_safe_blocker_text(str(exc)))
    command = [str(item) for item in _list(row.get("command"))]
    if not command:
        blockers.append("selected_row_command_missing")
    if _command_contains_forbidden_exact_or_remote_tokens(command):
        blockers.append("selected_row_command_contains_exact_or_remote_token")
    output_dir = _output_dir(command)
    if output_dir is None:
        blockers.append("selected_row_output_dir_missing")
    else:
        if not output_dir.is_absolute():
            blockers.append("selected_row_output_dir_not_absolute")
        resolved_output = output_dir.expanduser().resolve(strict=False)
        if not any(_is_relative_to(resolved_output, root) for root in allowed_roots):
            blockers.append("selected_row_output_dir_not_on_allowed_ssd_tier")
        collision_paths = _existing_output_artifact_paths(resolved_output)
        if collision_paths:
            blockers.append("selected_row_output_dir_contains_prior_training_artifacts")
    gate = row.get("score_lowering_gate")
    if isinstance(gate, Mapping):
        try:
            _require_no_authority(gate, label=f"selected_row_gate:{row_id or 'unknown'}")
        except NervLongTrainingCampaignAdmissionError as exc:
            blockers.append(_safe_blocker_text(str(exc)))
        if gate.get("local_mlx_executable") is not True:
            blockers.append("selected_row_gate_not_local_mlx_executable")
        if gate.get("cpu_replay_ready") is True or gate.get("exact_gate_ready") is True:
            blockers.append("selected_row_gate_overclaims_replay_or_exact_ready")
    return {
        "schema": "nerv_long_training_campaign_admission_row.v1",
        "id": row_id,
        "family": family,
        "priority": row.get("priority"),
        "output_dir": None if output_dir is None else output_dir.as_posix(),
        "output_report_path": _output_report_path(command),
        "existing_output_artifact_paths": (
            []
            if output_dir is None
            else [
                path.as_posix()
                for path in _existing_output_artifact_paths(
                    output_dir.expanduser().resolve(strict=False)
                )
            ]
        ),
        "command": command,
        "admitted": not blockers,
        "blockers": _dedupe(blockers),
        **PROXY_FALSE_AUTHORITY_FIELDS,
        "frontier_score_claim": False,
    }


def _require_no_authority(payload: Mapping[str, Any], *, label: str) -> None:
    for field in _AUTHORITY_FIELDS:
        if payload.get(field) is True:
            raise NervLongTrainingCampaignAdmissionError(
                f"{label}:{field}=truthy"
            )
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise NervLongTrainingCampaignAdmissionError(str(exc)) from exc


def _source_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema": str(payload.get("schema") or ""),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text.encode("utf-8")),
    }


def _output_dir(command: Sequence[str]) -> Path | None:
    try:
        return Path(command[command.index("--output-dir") + 1]).expanduser()
    except (ValueError, IndexError):
        return None


def _output_report_path(command: Sequence[str]) -> str:
    out_dir = _output_dir(command)
    if out_dir is None:
        return ""
    return (out_dir / "compact_renderer_mlx_spine_runner_report.json").as_posix()


def _observable_artifact_paths(command: Sequence[str], *, family: str) -> list[str]:
    out_dir = _output_dir(command)
    if out_dir is None:
        return []
    paths = [
        out_dir / "compact_renderer_mlx_spine_runner_report.json",
    ]
    if family in {"hi_nerv", "snerv"}:
        paths.append(out_dir / "compact_renderer_mlx_spine_runner_startup.json")
    if family == "hi_nerv":
        paths.extend(
            [
                out_dir / "hi_nerv_mlx_training" / "telemetry.jsonl",
                out_dir / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl",
            ]
        )
    elif family == "snerv":
        paths.extend(
            [
                out_dir / "snerv_mlx_training" / "telemetry.jsonl",
                out_dir / "snerv_mlx_training" / "local_mlx_prefilter_progress.jsonl",
            ]
        )
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = path.as_posix()
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _existing_output_artifact_paths(output_dir: Path) -> list[Path]:
    candidates = (
        output_dir / "compact_renderer_mlx_spine_runner_report.json",
        output_dir / "compact_renderer_mlx_spine_runner_startup.json",
        output_dir / "hi_nerv_mlx_training" / "telemetry.jsonl",
        output_dir / "hi_nerv_mlx_training" / "local_mlx_prefilter_progress.jsonl",
        output_dir / "snerv_mlx_training" / "telemetry.jsonl",
        output_dir / "snerv_mlx_training" / "local_mlx_prefilter_progress.jsonl",
        output_dir / "nerv_candidate_byte_feedback_row.json",
        output_dir / "nerv_candidate_byte_feedback.jsonl",
    )
    return [path for path in candidates if path.exists()]


def _command_contains_forbidden_exact_or_remote_tokens(command: Sequence[str]) -> bool:
    forbidden = {
        "modal",
        "lightning",
        "vastai",
        "exact",
        "auth_eval",
        "cuda_auth",
        "contest_cuda",
        "contest_cpu",
    }
    normalized = " ".join(str(item).lower() for item in command)
    return any(token in normalized for token in forbidden)


def _common_output_parent(output_dirs: Sequence[Path]) -> Path | None:
    if not output_dirs:
        return None
    parents = {path.parent.resolve(strict=False) for path in output_dirs}
    if len(parents) == 1:
        return next(iter(parents))
    # If multiple row directories were nested differently, storage preflight must
    # be split per parent rather than guessed here.
    return None


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else repo_root / value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _list(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _safe_blocker_text(value: str) -> str:
    return (
        value.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )[:240]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "ADMISSION_SCHEMA",
    "CONSUMER_RESULT_SCHEMA",
    "DEFAULT_ALLOWED_OUTPUT_ROOTS",
    "DEFAULT_LOCAL_MLX_LONG_TRAINING_TIMEOUT_SECONDS",
    "DEFAULT_QUEUE_ID",
    "DEFAULT_STORAGE_EXPECTED_BYTES_PER_ROW",
    "NervLongTrainingCampaignAdmissionError",
    "build_nerv_long_training_campaign_execution_admission",
    "render_nerv_long_training_campaign_execution_admission_markdown",
]
