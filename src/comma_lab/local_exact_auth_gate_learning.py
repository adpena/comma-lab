# SPDX-License-Identifier: MIT
"""Planner learning records from local exact-auth gate reports."""

from __future__ import annotations

import fcntl
import json
import subprocess
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from comma_lab.local_exact_auth_gate import LOCAL_EXACT_AUTH_GATE_SCHEMA
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import json_line, json_text, read_json, sha256_bytes, sha256_file

LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA = (
    "local_exact_auth_gate_learning_signal.v1"
)
LOCAL_EXACT_AUTH_GATE_POSTERIOR_ROW_SCHEMA = (
    "local_exact_auth_gate_posterior_row.v1"
)
LOCAL_EXACT_AUTH_GATE_POSTERIOR_APPEND_REPORT_SCHEMA = (
    "local_exact_auth_gate_posterior_append_report.v1"
)
DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_PATH = (
    Path(__file__).resolve().parents[2]
    / ".omx"
    / "state"
    / "local_exact_auth_gate_posterior.jsonl"
)
DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_LOCK_PATH = (
    Path(__file__).resolve().parents[2]
    / ".omx"
    / "state"
    / ".local_exact_auth_gate_posterior.lock"
)


class LocalExactAuthGateLearningError(ValueError):
    """Raised when local gate evidence cannot become a learning signal."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: str | Path, repo_root: str | Path) -> str:
    candidate = Path(path)
    root = Path(repo_root)
    try:
        return candidate.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return candidate.as_posix()


def _resolve(path: str | Path, repo_root: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path(repo_root) / candidate


def _stable_sha(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(json_text(payload).encode("utf-8"))


def _git_text(args: list[str], *, repo_root: str | Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _repo_provenance(repo_root: str | Path) -> dict[str, Any]:
    status = _git_text(["status", "--short", "--untracked-files=no"], repo_root=repo_root)
    return {
        "schema": "local_exact_auth_gate_repo_provenance.v1",
        "repo_root": str(Path(repo_root).resolve(strict=False)),
        "git_head": _git_text(["rev-parse", "--verify", "HEAD"], repo_root=repo_root),
        "git_dirty_tracked": bool(status),
        "git_status_short_tracked_sha256": (
            sha256_bytes((status + "\n").encode("utf-8")) if status is not None else None
        ),
        "git_status_short_tracked_line_count": (
            len(status.splitlines()) if status else 0
        ),
        "identity_excludes_worktree_status": True,
    }


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in {float("inf"), float("-inf")} else None


def _load_optional_json(path: str | Path | None, repo_root: str | Path) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise LocalExactAuthGateLearningError(f"JSON artifact missing: {path}")
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise LocalExactAuthGateLearningError(f"JSON artifact must be an object: {path}")
    return payload


def _artifact_record(label: str, path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    resolved = _resolve(path, repo_root)
    if not resolved.is_file():
        raise LocalExactAuthGateLearningError(f"required artifact missing: {label}={path}")
    return {
        "label": label,
        "path": _repo_rel(resolved, repo_root),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _policy_for_gate(gate: Mapping[str, Any]) -> str:
    next_action = str(gate.get("next_required_action") or "")
    blockers = {str(item) for item in gate.get("blockers") or []}
    if gate.get("exact_auth_dispatch_recommended") is True:
        return "exact_auth_dispatch_candidate_after_claim"
    if next_action == "run_local_cpu_replay":
        return "promote_mlx_prefilter_winner_to_local_cpu_replay"
    if (
        next_action == "do_not_run_local_cpu_replay"
        or "mlx_prefilter_action_not_below_target" in blockers
    ):
        return "demote_candidate_for_archive_until_allocator_or_actuator_changes"
    if "local_replay_required_for_exact_auth" in blockers:
        return "hold_until_local_cpu_replay_exists"
    return "hold_without_dispatch"


def _priority_delta(policy: str) -> str:
    if policy.startswith(("exact_auth_dispatch", "promote_")):
        return "increase"
    if policy.startswith("demote_"):
        return "decrease"
    return "hold"


def build_local_exact_auth_gate_learning_signal(
    *,
    gate_report: Mapping[str, Any],
    gate_report_path: str | Path,
    repo_root: str | Path,
    candidate_id: str,
    lane_id: str,
    family_id: str = "unclassified_local_candidate",
    replay_summary_path: str | Path | None = None,
    replay_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a false-authority planner signal from one local gate report."""

    if gate_report.get("schema") != LOCAL_EXACT_AUTH_GATE_SCHEMA:
        raise LocalExactAuthGateLearningError("gate report has unexpected schema")
    require_no_truthy_authority_fields(
        gate_report,
        context="local_exact_auth_gate_learning.gate_report",
    )
    replay_payload = dict(replay_summary or {})
    if replay_payload:
        require_no_truthy_authority_fields(
            replay_payload,
            context="local_exact_auth_gate_learning.replay_summary",
        )
    source_artifacts = [
        _artifact_record("local_exact_auth_gate_report", gate_report_path, repo_root)
    ]
    if replay_summary_path is not None:
        source_artifacts.append(
            _artifact_record("local_replay_or_mlx_prefilter_summary", replay_summary_path, repo_root)
        )
    blockers = ordered_unique(str(item) for item in gate_report.get("blockers") or [])
    policy = _policy_for_gate(gate_report)
    identity = {
        "schema": "local_exact_auth_gate_learning_identity.v1",
        "candidate_id": str(candidate_id),
        "lane_id": str(lane_id),
        "family_id": str(family_id),
        "gate_report_sha256": source_artifacts[0]["sha256"],
        "replay_summary_sha256": (
            source_artifacts[1]["sha256"] if len(source_artifacts) > 1 else None
        ),
        "next_required_action": gate_report.get("next_required_action"),
        "blockers": blockers,
    }
    feature_vector = {
        "schema": "local_exact_auth_gate_feature_vector.v1",
        "exact_auth_dispatch_recommended": gate_report.get("exact_auth_dispatch_recommended") is True,
        "exact_cpu_dispatch_recommended": gate_report.get("exact_cpu_dispatch_recommended") is True,
        "exact_cuda_dispatch_recommended": gate_report.get("exact_cuda_dispatch_recommended") is True,
        "next_required_action": gate_report.get("next_required_action"),
        "blocker_count": len(blockers),
        "mlx_action_proxy": _safe_float(gate_report.get("mlx_action_proxy")),
        "mlx_target_action": _safe_float(gate_report.get("mlx_target_action")),
        "local_score_estimate": _safe_float(gate_report.get("local_score_estimate")),
        "auth_target_score": _safe_float(gate_report.get("auth_target_score")),
        "mlx_prefilter_action_not_below_target": (
            "mlx_prefilter_action_not_below_target" in blockers
        ),
        "local_replay_required_for_exact_auth": (
            "local_replay_required_for_exact_auth" in blockers
        ),
        "local_axis_tag": gate_report.get("local_axis_tag"),
        "mlx_axis_tag": gate_report.get("mlx_axis_tag"),
        "exact_auth_axis": gate_report.get("exact_auth_axis"),
    }
    signal = {
        "schema": LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA,
        "signal_id": _stable_sha(identity),
        "created_at_utc": _utc_now(),
        "candidate_id": str(candidate_id),
        "lane_id": str(lane_id),
        "family_id": str(family_id),
        "evidence_grade": "local_gate_planning_signal_not_score_authority",
        "source_artifacts": source_artifacts,
        "reproducibility_provenance": {
            "schema": "local_exact_auth_gate_reproducibility_provenance.v1",
            "identity_is_stable_sha256_of_source_artifacts_and_gate_decision": True,
            "row_deduplication_excludes_wall_clock_time": True,
            "source_artifact_sha256s": [
                str(record["sha256"]) for record in source_artifacts
            ],
            "repo": _repo_provenance(repo_root),
        },
        "identity": identity,
        "planner_feature_vector": feature_vector,
        "recommended_acquisition_policy": policy,
        "priority_delta": _priority_delta(policy),
        "blockers": blockers,
        "warnings": ordered_unique(str(item) for item in gate_report.get("warnings") or []),
        "allowed_use": "local_acquisition_and_queue_routing_update_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        signal,
        context="local_exact_auth_gate_learning.signal",
    )
    return signal


def build_local_exact_auth_gate_posterior_row(
    *,
    learning_signal: Mapping[str, Any],
    learning_signal_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    if learning_signal.get("schema") != LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA:
        raise LocalExactAuthGateLearningError("learning signal has unexpected schema")
    require_no_truthy_authority_fields(
        learning_signal,
        context="local_exact_auth_gate_learning.posterior_input",
    )
    identity = dict(learning_signal.get("identity") or {})
    if not identity:
        raise LocalExactAuthGateLearningError("learning signal missing identity")
    row_identity = {
        "schema": "local_exact_auth_gate_posterior_row_identity.v1",
        "signal_id": learning_signal.get("signal_id"),
        "candidate_id": learning_signal.get("candidate_id"),
        "lane_id": learning_signal.get("lane_id"),
        "family_id": learning_signal.get("family_id"),
        "gate_report_sha256": identity.get("gate_report_sha256"),
        "replay_summary_sha256": identity.get("replay_summary_sha256"),
    }
    row = {
        "schema": LOCAL_EXACT_AUTH_GATE_POSTERIOR_ROW_SCHEMA,
        "row_id": _stable_sha(row_identity),
        "ingested_at_utc": _utc_now(),
        "row_identity": row_identity,
        "source_signal": _artifact_record(
            "local_exact_auth_gate_learning_signal",
            learning_signal_path,
            repo_root,
        ),
        "candidate_id": learning_signal.get("candidate_id"),
        "lane_id": learning_signal.get("lane_id"),
        "family_id": learning_signal.get("family_id"),
        "planner_feature_vector": dict(learning_signal.get("planner_feature_vector") or {}),
        "recommended_acquisition_policy": learning_signal.get(
            "recommended_acquisition_policy"
        ),
        "priority_delta": learning_signal.get("priority_delta"),
        "blockers": list(learning_signal.get("blockers") or []),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "local_exact_auth_gate_posterior_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context="local_exact_auth_gate_learning.posterior_row",
    )
    return row


def load_local_exact_auth_gate_posterior_rows(
    posterior_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    path = Path(posterior_path or DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_PATH)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise LocalExactAuthGateLearningError(
                f"{path}: posterior row {line_number} must be an object"
            )
        rows.append(payload)
    return rows


@contextmanager
def _posterior_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_local_exact_auth_gate_posterior_signal(
    *,
    learning_signal: Mapping[str, Any],
    learning_signal_path: str | Path,
    repo_root: str | Path,
    posterior_path: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    row = build_local_exact_auth_gate_posterior_row(
        learning_signal=learning_signal,
        learning_signal_path=learning_signal_path,
        repo_root=repo_root,
    )
    path = Path(posterior_path or DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_PATH)
    lock = Path(lock_path or DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_LOCK_PATH)
    with _posterior_lock(lock):
        rows = load_local_exact_auth_gate_posterior_rows(path)
        appended = not any(
            isinstance(existing, Mapping) and existing.get("row_id") == row["row_id"]
            for existing in rows
        )
        if appended:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json_line(row))
    report = {
        "schema": LOCAL_EXACT_AUTH_GATE_POSTERIOR_APPEND_REPORT_SCHEMA,
        "posterior_path": _repo_rel(path, repo_root),
        "lock_path": _repo_rel(lock, repo_root),
        "row_id": row["row_id"],
        "candidate_id": row.get("candidate_id"),
        "lane_id": row.get("lane_id"),
        "family_id": row.get("family_id"),
        "recommended_acquisition_policy": row.get("recommended_acquisition_policy"),
        "priority_delta": row.get("priority_delta"),
        "appended": appended,
        "skipped_duplicate": not appended,
        "existing_row_count": len(rows),
        "final_row_count": len(rows) + (1 if appended else 0),
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "local_exact_auth_gate_posterior_append_audit",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        report,
        context="local_exact_auth_gate_learning.posterior_append_report",
    )
    return report


def load_gate_and_build_signal(
    *,
    gate_report_path: str | Path,
    repo_root: str | Path,
    candidate_id: str,
    lane_id: str,
    family_id: str,
    replay_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    gate_report = _load_optional_json(gate_report_path, repo_root)
    assert gate_report is not None
    replay_summary = _load_optional_json(replay_summary_path, repo_root)
    return build_local_exact_auth_gate_learning_signal(
        gate_report=gate_report,
        gate_report_path=gate_report_path,
        repo_root=repo_root,
        candidate_id=candidate_id,
        lane_id=lane_id,
        family_id=family_id,
        replay_summary_path=replay_summary_path,
        replay_summary=replay_summary,
    )


__all__ = [
    "DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_LOCK_PATH",
    "DEFAULT_LOCAL_EXACT_AUTH_GATE_POSTERIOR_PATH",
    "LOCAL_EXACT_AUTH_GATE_LEARNING_SIGNAL_SCHEMA",
    "LOCAL_EXACT_AUTH_GATE_POSTERIOR_APPEND_REPORT_SCHEMA",
    "LOCAL_EXACT_AUTH_GATE_POSTERIOR_ROW_SCHEMA",
    "LocalExactAuthGateLearningError",
    "append_local_exact_auth_gate_posterior_signal",
    "build_local_exact_auth_gate_learning_signal",
    "build_local_exact_auth_gate_posterior_row",
    "load_gate_and_build_signal",
    "load_local_exact_auth_gate_posterior_rows",
]
