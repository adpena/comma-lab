# SPDX-License-Identifier: MIT
"""Shared exact-dispatch authority checks.

The ready flag is an input fact, not dispatch authority by itself. Paid exact
eval fan-out must also prove live archive/runtime custody at the moment the
actuator is about to fire.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.optimizer.exact_readiness import (
    claim_status_terminal,
    is_sha256,
    parse_claim_rows,
    readiness_blockers,
    resolve_path,
)

CONTEST_EXACT_EVAL_TARGET_MODE = "contest_exact_eval"
ClaimPolicy = Literal["preclaim_conflict_check", "require_active_claim"]


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
        modes.add("contest_exact_eval")
    return modes


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
) -> bool:
    if dispatch_claims_path is None or not dispatch_claims_path.is_file():
        return False
    platform_token = str(platform or "").strip().lower()
    allowed_job_ids = {
        str(item).strip()
        for item in instance_job_ids
        if str(item).strip()
    }
    closed_job_ids: set[str] = set()
    for row in parse_claim_rows(dispatch_claims_path):
        if row["lane_id"] != lane_id:
            continue
        if platform_token and row["platform"].strip().lower() != platform_token:
            continue
        job_id = row["instance_job_id"].strip()
        if allowed_job_ids and job_id not in allowed_job_ids:
            continue
        if claim_status_terminal(row["status"]):
            closed_job_ids.add(job_id)
            continue
        if job_id in closed_job_ids:
            continue
        return True
    return False


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
) -> ExactDispatchAuthorityVerdict:
    """Return fail-closed authority for a paid exact-eval dispatch row."""

    root = Path(repo_root).resolve()
    queue_root = Path(queue_dir).resolve() if queue_dir is not None else None
    claims_path = Path(dispatch_claims_path) if dispatch_claims_path is not None else None
    ready_flag = row.get("ready_for_exact_eval_dispatch") is True
    contest_target = CONTEST_EXACT_EVAL_TARGET_MODE in _target_modes(row)
    blockers: list[str] = []
    if claim_policy not in {"preclaim_conflict_check", "require_active_claim"}:
        blockers.append(f"unknown_claim_policy:{claim_policy}")

    if row.get("score_claim") is True:
        blockers.append("score_claim_true_requires_result_review")
    if row.get("promotion_eligible") is True:
        blockers.append("promotion_eligible_true_requires_result_review")
    if require_ready_flag and not ready_flag:
        blockers.append("ready_for_exact_eval_dispatch_not_true")
    if require_contest_target and not contest_target:
        blockers.append("contest_exact_eval_target_mode_missing")

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
        ignore_active_claim_conflicts=claim_policy == "require_active_claim",
    )
    blockers.extend(readiness)
    facts["claim_policy"] = claim_policy

    lane_id = facts.get("lane_id")
    if claim_policy == "require_active_claim":
        if claims_path is None:
            blockers.append("active_dispatch_claim_required_missing_claims_path")
        elif not isinstance(lane_id, str) or not lane_id.strip():
            blockers.append("active_dispatch_claim_required_missing_lane_id")
        elif not active_dispatch_claim_present(
            lane_id=lane_id,
            dispatch_claims_path=claims_path,
            platform=required_claim_platform,
            instance_job_ids=required_claim_instance_job_ids,
        ):
            suffix = ""
            required_jobs = [
                str(item).strip()
                for item in required_claim_instance_job_ids
                if str(item).strip()
            ]
            if required_claim_platform:
                suffix += f":platform={required_claim_platform}"
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
