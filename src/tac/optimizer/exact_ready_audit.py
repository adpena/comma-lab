"""Audit generated exact-ready queues against terminal dispatch evidence."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_SCORE,
    as_bool,
    is_sha256,
    read_json,
    repo_rel,
    terminal_claim_result_conflicts,
    utc_now,
)

AUDIT_SCHEMA = "optimizer_exact_ready_queue_terminal_evidence_audit_v1"


def _row_archive_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("candidate_archive_sha256", "archive_sha256", "expected_archive_sha256"):
        value = row.get(key)
        if is_sha256(value):
            return str(value).lower()
    return None


def _row_runtime_tree_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("runtime_tree_sha256", "candidate_runtime_tree_sha256"):
        value = row.get(key)
        if is_sha256(value):
            return str(value).lower()
    runtime_manifest = row.get("runtime_manifest")
    if isinstance(runtime_manifest, Mapping):
        value = runtime_manifest.get("runtime_tree_sha256")
        if is_sha256(value):
            return str(value).lower()
    return None


def _queue_rows(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for list_name in ("dispatch_ready", "top_k"):
        rows = payload.get(list_name)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                yield row


def audit_exact_ready_queue(
    queue_path: Path,
    *,
    repo_root: Path,
    dispatch_claims_path: Path,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
) -> dict[str, Any]:
    """Return stale-row findings for one generated exact-ready queue."""

    payload = read_json(queue_path)
    if not isinstance(payload, Mapping):
        return {
            "queue_path": repo_rel(queue_path, repo_root),
            "queue_schema": None,
            "row_count": 0,
            "stale_ready_rows": [
                {
                    "candidate_id": None,
                    "lane_id": None,
                    "archive_sha256": None,
                    "blockers": ["exact_ready_queue_not_object"],
                }
            ],
        }

    stale_rows: list[dict[str, Any]] = []
    seen: set[
        tuple[str | None, str | None, str | None, str | None, tuple[str, ...]]
    ] = set()
    row_count = 0
    for row in _queue_rows(payload):
        row_count += 1
        if row.get("ready_for_exact_eval_dispatch") is not True:
            continue
        lane_id = row.get("lane_id")
        archive_sha = _row_archive_sha(row)
        if not isinstance(lane_id, str) or not lane_id.strip():
            blockers = ["lane_id_missing"]
        elif archive_sha is None:
            blockers = ["archive_sha256_missing_or_invalid"]
        else:
            blockers = terminal_claim_result_conflicts(
                lane_id,
                archive_sha,
                dispatch_claims_path=dispatch_claims_path,
                active_floor_score=active_floor_score,
                runtime_tree_sha256=_row_runtime_tree_sha(row),
                score_affecting_runtime_changed=as_bool(
                    row.get("score_affecting_runtime_changed")
                ),
            )
        if not blockers:
            continue
        candidate_id = row.get("candidate_id")
        runtime_tree_sha = _row_runtime_tree_sha(row)
        runtime_changed = as_bool(row.get("score_affecting_runtime_changed"))
        key = (
            str(candidate_id) if candidate_id is not None else None,
            lane_id if isinstance(lane_id, str) else None,
            archive_sha,
            runtime_tree_sha,
            tuple(blockers),
        )
        if key in seen:
            continue
        seen.add(key)
        stale_rows.append(
            {
                "candidate_id": candidate_id,
                "lane_id": lane_id,
                "archive_sha256": archive_sha,
                "runtime_tree_sha256": runtime_tree_sha,
                "score_affecting_runtime_changed": runtime_changed,
                "ready_for_exact_eval_dispatch": True,
                "blockers": blockers,
            }
        )

    return {
        "queue_path": repo_rel(queue_path, repo_root),
        "queue_schema": payload.get("schema"),
        "row_count": row_count,
        "stale_ready_rows": stale_rows,
    }


def discover_exact_ready_queues(
    *,
    repo_root: Path,
    scan_root: Path,
    patterns: Iterable[str] = (
        "**/exact_ready_queue.json",
        "**/*exact_ready_queue.json",
    ),
) -> list[Path]:
    root = scan_root if scan_root.is_absolute() else repo_root / scan_root
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found[path.resolve().as_posix()] = path
    return [found[key] for key in sorted(found)]


def audit_exact_ready_queues(
    queue_paths: Iterable[Path],
    *,
    repo_root: Path,
    dispatch_claims_path: Path,
    active_floor_score: float | None = ACTIVE_FLOOR_SCORE,
) -> dict[str, Any]:
    queues = [
        audit_exact_ready_queue(
            path,
            repo_root=repo_root,
            dispatch_claims_path=dispatch_claims_path,
            active_floor_score=active_floor_score,
        )
        for path in queue_paths
    ]
    stale_count = sum(len(queue["stale_ready_rows"]) for queue in queues)
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "passed": stale_count == 0,
        "queue_count": len(queues),
        "stale_ready_row_count": stale_count,
        "queues": queues,
    }


__all__ = [
    "AUDIT_SCHEMA",
    "audit_exact_ready_queue",
    "audit_exact_ready_queues",
    "discover_exact_ready_queues",
]
