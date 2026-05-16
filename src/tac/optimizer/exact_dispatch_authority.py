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
from typing import Any

from tac.optimizer.exact_readiness import (
    is_sha256,
    readiness_blockers,
    resolve_path,
)

CONTEST_EXACT_EVAL_TARGET_MODE = "contest_exact_eval"


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
) -> ExactDispatchAuthorityVerdict:
    """Return fail-closed authority for a paid exact-eval dispatch row."""

    root = Path(repo_root).resolve()
    queue_root = Path(queue_dir).resolve() if queue_dir is not None else None
    claims_path = Path(dispatch_claims_path) if dispatch_claims_path is not None else None
    ready_flag = row.get("ready_for_exact_eval_dispatch") is True
    contest_target = CONTEST_EXACT_EVAL_TARGET_MODE in _target_modes(row)
    blockers: list[str] = []

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
    )
    blockers.extend(readiness)

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
    "ExactDispatchAuthorityVerdict",
    "exact_dispatch_authority",
]
