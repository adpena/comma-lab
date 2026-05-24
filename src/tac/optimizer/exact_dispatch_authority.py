# SPDX-License-Identifier: MIT
"""Shared exact-dispatch authority checks.

The ready flag is an input fact, not dispatch authority by itself. Paid exact
eval fan-out must also prove live archive/runtime custody at the moment the
actuator is about to fire.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.optimization.proxy_candidate_contract import (
    CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS,
    truthy_authority_field_violations,
)
from tac.optimizer.exact_readiness import (
    claim_status_terminal,
    is_sha256,
    parse_claim_rows,
    parse_utc,
    readiness_blockers,
    resolve_path,
)

CONTEST_EXACT_EVAL_TARGET_MODE = "contest_exact_eval"
ClaimPolicy = Literal["preclaim_conflict_check", "require_active_claim"]
PRE_DISPATCH_ALLOWED_TRUTHY_AUTHORITY_FIELDS = frozenset(
    {"dispatch_packet_ready", "ready_for_exact_eval_dispatch"}
)
SCORE_AXIS_FIELDS = (
    "score_axis",
    "target_score_axis",
    "exact_eval_axis",
    "auth_eval_axis",
)


@dataclass(frozen=True)
class ExactDispatchAuthorityVerdict:
    """Stable verdict for paid exact-eval dispatch authorization."""

    source: str
    authorized: bool
    blockers: tuple[str, ...]
    ready_for_exact_eval_dispatch: bool
    contest_exact_eval_target: bool
    facts: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly verdict dictionary."""

        return {
            "source": self.source,
            "authorized": self.authorized,
            "blockers": list(self.blockers),
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "contest_exact_eval_target": self.contest_exact_eval_target,
            "facts": _jsonable(self.facts),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _target_modes(row: Mapping[str, Any]) -> set[str]:
    modes: set[str] = set()
    for key in (
        "optimization_target",
        "target_mode",
        "target_modes",
        "dispatch_target",
        "deployment_target",
        "deployment_targets",
    ):
        for item in _as_text_items(row.get(key)):
            token = item.strip().lower().replace("-", "_").replace(" ", "_")
            if token:
                modes.add(token)
    if row.get("contest_mode") is True:
        modes.add("contest")
    return modes


def _normalize_score_axis(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _score_axis_blockers(
    row: Mapping[str, Any],
    *,
    required_score_axis: str | None,
) -> tuple[list[str], dict[str, str]]:
    declared: dict[str, str] = {}
    for key in SCORE_AXIS_FIELDS:
        axis = _normalize_score_axis(row.get(key))
        if axis:
            declared[key] = axis
    if required_score_axis is None:
        return [], declared
    required = _normalize_score_axis(required_score_axis)
    blockers: list[str] = []
    if not declared:
        blockers.append(f"score_axis_missing:required={required}")
        return blockers, declared
    values = set(declared.values())
    if len(values) > 1:
        details = ",".join(f"{key}={value}" for key, value in sorted(declared.items()))
        blockers.append(f"score_axis_field_mismatch:{details}")
    if required not in values:
        details = ",".join(f"{key}={value}" for key, value in sorted(declared.items()))
        blockers.append(f"score_axis_required:{required}:declared={details}")
    return blockers, declared


def _truthy_authority_blockers(row: Mapping[str, Any]) -> list[str]:
    fields = [
        field
        for field in CONSUMER_PAYLOAD_FORBIDDEN_TRUE_AUTHORITY_FIELDS
        if field not in PRE_DISPATCH_ALLOWED_TRUTHY_AUTHORITY_FIELDS
    ]
    return [
        f"truthy_authority_field:{violation}"
        for violation in truthy_authority_field_violations(row, fields=fields)
    ]


def _optional_resolved_path(
    row: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    repo_root: Path,
    queue_dir: Path | None,
) -> Path | None:
    for key in keys:
        value = row.get(key)
        path = resolve_path(value, repo_root=repo_root, queue_dir=queue_dir)
        if path is not None:
            return path
    return None


def active_dispatch_claim_present(
    *,
    lane_id: str,
    dispatch_claims_path: Path | None,
    platform: str | None = None,
    instance_job_ids: Iterable[str] = (),
    now_utc: dt.datetime | None = None,
    ttl_hours: float = 24.0,
) -> bool:
    if dispatch_claims_path is None or not dispatch_claims_path.is_file():
        return False
    now = now_utc or dt.datetime.now(tz=dt.UTC).replace(microsecond=0)
    platform_token = str(platform or "").strip().lower()
    allowed_job_ids = {
        str(item).strip()
        for item in instance_job_ids
        if str(item).strip()
    }
    for row in _latest_claim_rows_by_job(dispatch_claims_path).values():
        if row["lane_id"] != lane_id:
            continue
        if platform_token and row["platform"].strip().lower() != platform_token:
            continue
        job_id = row["instance_job_id"].strip()
        if allowed_job_ids and job_id not in allowed_job_ids:
            continue
        if claim_status_terminal(row["status"]):
            continue
        ts = parse_utc(row["timestamp_utc"])
        if ts is None:
            continue
        age_hours = max((now - ts).total_seconds() / 3600.0, 0.0)
        if age_hours > ttl_hours:
            continue
        return True
    return False


def _latest_claim_rows_by_job(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    latest_by_job: dict[tuple[str, str], dict[str, str]] = {}
    for row in parse_claim_rows(path):
        key = (row["lane_id"], row["instance_job_id"])
        prev = latest_by_job.get(key)
        row_ts = parse_utc(row["timestamp_utc"])
        prev_ts = parse_utc(prev["timestamp_utc"]) if prev is not None else None
        if prev is None or prev_ts is None or (row_ts is not None and row_ts > prev_ts):
            latest_by_job[key] = row
    return latest_by_job


def exact_dispatch_authority(
    row: Mapping[str, Any],
    *,
    repo_root: str | Path,
    queue_dir: str | Path | None = None,
    source: str = "unknown",
    require_ready_flag: bool = True,
    require_contest_target: bool = True,
    active_floor_archive_bytes: int | None = None,
    active_floor_score: float | None = None,
    allow_above_active_floor_dispatch: bool = False,
    operator_override_reason: str | None = None,
    extra_clearable_source_blockers: Iterable[str] = (),
    dispatch_claims_path: str | Path | None = None,
    claim_policy: ClaimPolicy = "preclaim_conflict_check",
    required_claim_platform: str | None = None,
    required_claim_instance_job_ids: Iterable[str] = (),
    claim_ttl_hours: float = 24.0,
    required_score_axis: str | None = None,
) -> ExactDispatchAuthorityVerdict:
    """Return fail-closed authority for a paid exact-eval dispatch row."""

    root = Path(repo_root).resolve()
    queue_root = Path(queue_dir).resolve() if queue_dir is not None else None
    claims_path = Path(dispatch_claims_path) if dispatch_claims_path is not None else None
    ready_flag = row.get("ready_for_exact_eval_dispatch") is True
    contest_target = CONTEST_EXACT_EVAL_TARGET_MODE in _target_modes(row)
    blockers: list[str] = []
    required_jobs = tuple(
        str(item).strip()
        for item in required_claim_instance_job_ids
        if str(item).strip()
    )
    required_platform = str(required_claim_platform or "").strip()
    if claim_policy not in {"preclaim_conflict_check", "require_active_claim"}:
        blockers.append(f"unknown_claim_policy:{claim_policy}")
    if claim_policy == "require_active_claim":
        if not required_platform:
            blockers.append("active_dispatch_claim_required_platform_missing")
        if not required_jobs:
            blockers.append("active_dispatch_claim_required_job_id_missing")

    if row.get("score_claim") is True:
        blockers.append("score_claim_true_requires_result_review")
    if row.get("promotion_eligible") is True:
        blockers.append("promotion_eligible_true_requires_result_review")
    blockers.extend(_truthy_authority_blockers(row))
    if require_ready_flag and not ready_flag:
        blockers.append("ready_for_exact_eval_dispatch_not_true")
    if require_contest_target and not contest_target:
        blockers.append("contest_exact_eval_target_mode_missing")
    score_axis_blockers, declared_score_axes = _score_axis_blockers(
        row,
        required_score_axis=required_score_axis,
    )
    blockers.extend(score_axis_blockers)

    submission_dir = _optional_resolved_path(
        row,
        ("submission_dir", "submission_path", "runtime_dir"),
        repo_root=root,
        queue_dir=queue_root,
    )
    archive_manifest_path = _optional_resolved_path(
        row,
        ("archive_manifest_path", "manifest_path", "runtime_packet_manifest_path"),
        repo_root=root,
        queue_dir=queue_root,
    )
    readiness, facts = readiness_blockers(
        row,
        repo_root=root,
        queue_dir=queue_root,
        submission_dir=submission_dir,
        archive_manifest_path=archive_manifest_path,
        active_floor_archive_bytes=active_floor_archive_bytes,
        active_floor_score=active_floor_score,
        allow_above_active_floor_dispatch=allow_above_active_floor_dispatch,
        operator_override_reason=operator_override_reason,
        extra_clearable_source_blockers=extra_clearable_source_blockers,
        dispatch_claims_path=claims_path,
        claim_ttl_hours=claim_ttl_hours,
        ignore_active_claim_conflicts=False,
        allowed_active_claim_platform=required_platform
        if claim_policy == "require_active_claim"
        else None,
        allowed_active_claim_instance_job_ids=required_jobs
        if claim_policy == "require_active_claim"
        else (),
    )
    blockers.extend(readiness)
    facts["claim_policy"] = claim_policy
    facts["required_score_axis"] = (
        _normalize_score_axis(required_score_axis)
        if required_score_axis is not None
        else None
    )
    facts["declared_score_axes"] = dict(declared_score_axes)

    lane_id = facts.get("lane_id")
    if claim_policy == "require_active_claim":
        if claims_path is None:
            blockers.append("active_dispatch_claim_required_missing_claims_path")
        elif not isinstance(lane_id, str) or not lane_id.strip():
            blockers.append("active_dispatch_claim_required_missing_lane_id")
        elif not active_dispatch_claim_present(
            lane_id=lane_id,
            dispatch_claims_path=claims_path,
            platform=required_platform,
            instance_job_ids=required_jobs,
            ttl_hours=claim_ttl_hours,
        ):
            suffix = ""
            if required_platform:
                suffix += f":platform={required_platform}"
            if required_jobs:
                suffix += ":job_id=" + ",".join(sorted(required_jobs))
            blockers.append("active_dispatch_claim_required_not_found" + suffix)

    declared_runtime_sha = row.get("runtime_tree_sha256")
    runtime_manifest = facts.get("runtime_manifest")
    actual_runtime_sha = (
        runtime_manifest.get("runtime_tree_sha256")
        if isinstance(runtime_manifest, Mapping)
        else None
    )
    if (
        is_sha256(declared_runtime_sha)
        and is_sha256(actual_runtime_sha)
        and str(declared_runtime_sha).lower() != str(actual_runtime_sha).lower()
    ):
        blockers.append("runtime_tree_sha256_mismatch")
    declared_runtime_content_sha = row.get("runtime_content_tree_sha256")
    actual_runtime_content_sha = (
        runtime_manifest.get("runtime_content_tree_sha256")
        if isinstance(runtime_manifest, Mapping)
        else None
    )
    if not is_sha256(declared_runtime_content_sha):
        blockers.append("runtime_content_tree_sha256_missing_or_invalid")
    elif (
        is_sha256(actual_runtime_content_sha)
        and str(declared_runtime_content_sha).lower()
        != str(actual_runtime_content_sha).lower()
    ):
        blockers.append("runtime_content_tree_sha256_mismatch")

    deduped = tuple(dict.fromkeys(str(blocker) for blocker in blockers if str(blocker)))
    return ExactDispatchAuthorityVerdict(
        source=source,
        authorized=not deduped,
        blockers=deduped,
        ready_for_exact_eval_dispatch=ready_flag,
        contest_exact_eval_target=contest_target,
        facts=dict(facts),
    )


__all__ = [
    "CONTEST_EXACT_EVAL_TARGET_MODE",
    "ClaimPolicy",
    "ExactDispatchAuthorityVerdict",
    "active_dispatch_claim_present",
    "exact_dispatch_authority",
]
