# SPDX-License-Identifier: MIT
"""Non-destructive repair for legacy exact-ready queues missing score axis."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimizer.exact_readiness import read_json, repo_rel, sha256_file, utc_now
from tac.optimizer.exact_ready_audit import audit_exact_ready_queue
from tac.repo_io import write_json

REPAIR_REPORT_SCHEMA = "optimizer_exact_ready_score_axis_repair_report_v1"
REPAIR_METADATA_SCHEMA = "optimizer_exact_ready_score_axis_repair_metadata_v1"
DEFAULT_SCORE_AXIS = "contest_cuda"
SUPPORTED_REPAIR_SCORE_AXES = frozenset({DEFAULT_SCORE_AXIS})
FALSE_AUTHORITY: dict[str, bool] = {
    "dispatch_attempted": False,
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def plan_exact_ready_score_axis_repairs_from_audit(
    audit_payload: Mapping[str, Any],
    *,
    score_axis: str = DEFAULT_SCORE_AXIS,
) -> dict[str, Any]:
    """Classify stale exact-ready rows that are score-axis-only repairable.

    The plan is advisory and non-mutating.  It exists so operator flows can
    distinguish a legacy metadata gap from a row that must be regenerated or
    retired by terminal evidence before anyone reaches for a JSON editor.
    """

    axis = _normalize_score_axis(score_axis)
    rows: list[dict[str, Any]] = []
    repairable = 0
    skipped = 0
    for queue in audit_payload.get("queues", []):
        if not isinstance(queue, Mapping):
            continue
        queue_path = str(queue.get("queue_path") or "")
        stale_rows = queue.get("stale_ready_rows", [])
        if not isinstance(stale_rows, list):
            continue
        for stale_row in stale_rows:
            if not isinstance(stale_row, Mapping):
                continue
            blockers = _unique_strings(stale_row.get("blockers", []))
            row: dict[str, Any] = {
                "source_queue_path": queue_path,
                "candidate_id": stale_row.get("candidate_id"),
                "lane_id": stale_row.get("lane_id"),
                "archive_sha256": stale_row.get("archive_sha256"),
                "runtime_tree_sha256": stale_row.get("runtime_tree_sha256"),
                "runtime_content_tree_sha256": stale_row.get(
                    "runtime_content_tree_sha256"
                ),
                "source_blockers": blockers,
                "score_axis": axis,
                "write_requested": False,
                "status": "skipped",
                "skip_reason": None,
                "proposed_fields": None,
                "automatic_mutation_allowed": False,
                **FALSE_AUTHORITY,
            }
            if blockers == ["score_axis_missing"]:
                row["status"] = "repairable"
                row["proposed_fields"] = {
                    "score_axis": axis,
                    "target_score_axis": axis,
                }
                repairable += 1
            else:
                row["skip_reason"] = _skip_reason_for_blockers(blockers)
                skipped += 1
            rows.append(row)
    return {
        "schema": REPAIR_REPORT_SCHEMA,
        "generated_at_utc": utc_now(),
        "source_audit_schema": audit_payload.get("schema"),
        "score_axis": axis,
        "write_repaired_queues": False,
        "queue_count": int(audit_payload.get("queue_count") or 0),
        "stale_ready_row_count": int(audit_payload.get("stale_ready_row_count") or 0),
        "repairable_or_repaired_count": repairable,
        "skipped_count": skipped,
        "automatic_mutation_count": 0,
        **FALSE_AUTHORITY,
        "rows": rows,
    }


def repair_exact_ready_score_axis_queues(
    queue_paths: Iterable[str | Path],
    *,
    repo_root: str | Path,
    out_dir: str | Path,
    dispatch_claims_path: str | Path,
    score_axis: str = DEFAULT_SCORE_AXIS,
    write_repaired_queues: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy repairable legacy queues with explicit score-axis metadata.

    Only rows whose current exact-ready audit blockers are exactly
    ``score_axis_missing`` are repaired. Rows with terminal evidence, runtime
    custody gaps, active claim conflicts, or any other blocker remain skipped.
    The source queue is never modified.
    """

    repo = Path(repo_root).resolve()
    output_dir = _resolve_path(out_dir, repo_root=repo)
    claims = _resolve_path(dispatch_claims_path, repo_root=repo)
    axis = _normalize_score_axis(score_axis)
    rows: list[dict[str, Any]] = []
    repaired = 0
    skipped = 0
    for raw_path in queue_paths:
        queue_path = _resolve_path(raw_path, repo_root=repo)
        source_audit = audit_exact_ready_queue(
            queue_path,
            repo_root=repo,
            dispatch_claims_path=claims,
        )
        source_rows = source_audit.get("stale_ready_rows", [])
        blockers = _unique_strings(
            blocker
            for row in source_rows
            if isinstance(row, Mapping)
            for blocker in row.get("blockers", [])
        )
        row: dict[str, Any] = {
            "source_queue_path": repo_rel(queue_path, repo),
            "source_queue_sha256": sha256_file(queue_path) if queue_path.is_file() else None,
            "source_stale_ready_row_count": len(source_rows)
            if isinstance(source_rows, list)
            else 0,
            "source_blockers": blockers,
            "score_axis": axis,
            "write_requested": write_repaired_queues,
            "repaired_queue_path": None,
            "repaired_queue_sha256": None,
            "repaired_audit_stale_ready_row_count": None,
            "status": "skipped",
            "skip_reason": None,
        }
        repairable, reason = _is_axis_only_repairable(source_rows)
        if not repairable:
            row["skip_reason"] = reason
            skipped += 1
            rows.append(row)
            continue
        payload = read_json(queue_path)
        if not isinstance(payload, dict):
            row["skip_reason"] = "source_queue_not_object"
            skipped += 1
            rows.append(row)
            continue
        patch_blockers = _patch_axis_fields(payload, score_axis=axis)
        if patch_blockers:
            row["skip_reason"] = ";".join(patch_blockers)
            skipped += 1
            rows.append(row)
            continue
        output_path = _repair_output_path(
            queue_path,
            payload=payload,
            repo_root=repo,
            out_dir=output_dir,
            score_axis=axis,
        )
        row["repaired_queue_path"] = repo_rel(output_path, repo)
        payload["score_axis_repair"] = {
            "schema": REPAIR_METADATA_SCHEMA,
            "generated_at_utc": utc_now(),
            "source_queue_path": repo_rel(queue_path, repo),
            "source_queue_sha256": row["source_queue_sha256"],
            "source_blockers": blockers,
            "score_axis": axis,
            "non_destructive_copy": True,
        }
        if write_repaired_queues:
            if output_path.exists() and not overwrite:
                row["skip_reason"] = "repaired_queue_exists"
                skipped += 1
                rows.append(row)
                continue
            write_json(output_path, payload)
            row["repaired_queue_sha256"] = sha256_file(output_path)
            repaired_audit = audit_exact_ready_queue(
                output_path,
                repo_root=repo,
                dispatch_claims_path=claims,
            )
            stale_rows = repaired_audit.get("stale_ready_rows", [])
            row["repaired_audit_stale_ready_row_count"] = (
                len(stale_rows) if isinstance(stale_rows, list) else None
            )
            row["repaired_audit_blockers"] = _unique_strings(
                blocker
                for stale_row in stale_rows
                if isinstance(stale_row, Mapping)
                for blocker in stale_row.get("blockers", [])
            )
            row["status"] = (
                "repaired"
                if row["repaired_audit_stale_ready_row_count"] == 0
                else "repair_still_blocked"
            )
        else:
            row["status"] = "repairable"
        if row["status"] in {"repairable", "repaired"}:
            repaired += 1
        else:
            skipped += 1
        rows.append(row)
    return {
        "schema": REPAIR_REPORT_SCHEMA,
        "generated_at_utc": utc_now(),
        "score_axis": axis,
        "write_repaired_queues": write_repaired_queues,
        "queue_count": len(rows),
        "repairable_or_repaired_count": repaired,
        "skipped_count": skipped,
        "automatic_mutation_count": 0,
        **FALSE_AUTHORITY,
        "rows": rows,
    }


def _is_axis_only_repairable(rows: object) -> tuple[bool, str | None]:
    if not isinstance(rows, list) or not rows:
        return False, "no_stale_axis_missing_rows"
    for row in rows:
        if not isinstance(row, Mapping):
            return False, "stale_row_not_object"
        blockers = row.get("blockers")
        blocker_list = _unique_strings(blockers if isinstance(blockers, list) else [])
        if blocker_list != ["score_axis_missing"]:
            return False, _skip_reason_for_blockers(blocker_list)
    return True, None


def _normalize_score_axis(score_axis: str) -> str:
    axis = score_axis.strip().lower()
    if axis not in SUPPORTED_REPAIR_SCORE_AXES:
        raise ValueError(
            "exact-ready score-axis repair only supports "
            f"{sorted(SUPPORTED_REPAIR_SCORE_AXES)!r}; got {score_axis!r}"
        )
    return axis


def _skip_reason_for_blockers(blockers: list[str]) -> str:
    if not blockers:
        return "no_stale_axis_missing_rows"
    if any(blocker.startswith("score_axis_unsupported:") for blocker in blockers):
        return "unsupported_score_axis"
    terminal_prefixes = (
        "same_lane_terminal_",
        "result_review_exact_cuda_",
        "packetir_exact_closure_",
    )
    if any(blocker.startswith(terminal_prefixes) for blocker in blockers):
        return "terminal_or_duplicate_exact_eval_evidence"
    custody_markers = (
        "ready_row_",
        "runtime_",
        "archive_",
    )
    if any(blocker.startswith(custody_markers) for blocker in blockers):
        return "live_custody_regeneration_required"
    if any("active_dispatch_claim" in blocker for blocker in blockers):
        return "active_dispatch_claim_conflict"
    if "score_axis_missing" in blockers:
        return "score_axis_missing_plus_other_blockers"
    return "non_axis_only_blockers"


def _patch_axis_fields(payload: dict[str, Any], *, score_axis: str) -> list[str]:
    blockers: list[str] = []
    for key in ("dispatch_ready", "top_k"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                blockers.append(f"{key}[{index}]_not_object")
                continue
            target_modes = row.get("target_modes")
            if not (
                isinstance(target_modes, list)
                and "contest_exact_eval" in {str(item) for item in target_modes}
            ):
                blockers.append(f"{key}[{index}]_missing_contest_exact_eval_target")
                continue
            for axis_key in ("score_axis", "target_score_axis"):
                existing = row.get(axis_key)
                if isinstance(existing, str) and existing.strip():
                    if existing.strip().lower() != score_axis:
                        blockers.append(
                            f"{key}[{index}]_{axis_key}_unsupported:{existing.strip()}"
                        )
                    continue
                row[axis_key] = score_axis
            runtime_manifest = row.get("runtime_manifest")
            if isinstance(runtime_manifest, Mapping):
                for sha_key in ("runtime_tree_sha256", "runtime_content_tree_sha256"):
                    existing = row.get(sha_key)
                    manifest_value = runtime_manifest.get(sha_key)
                    if not _is_sha256(existing) and _is_sha256(manifest_value):
                        row[sha_key] = str(manifest_value).lower()
            proof_ref = row.get("runtime_consumption_proof_path")
            if isinstance(proof_ref, str) and proof_ref.strip():
                row.setdefault("runtime_consumption_proof_required", True)
                row.setdefault("runtime_consumption_proof_status", "present")
    return blockers


def _repair_output_path(
    queue_path: Path,
    *,
    payload: Mapping[str, Any],
    repo_root: Path,
    out_dir: Path,
    score_axis: str,
) -> Path:
    candidate_id = ""
    rows = payload.get("dispatch_ready")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        candidate_id = str(rows[0].get("candidate_id") or "")
    stem = _safe_slug(candidate_id or queue_path.stem)[:72]
    digest = hashlib.sha256(
        f"{repo_rel(queue_path, repo_root)}:{sha256_file(queue_path)}:{score_axis}".encode()
    ).hexdigest()[:12]
    return out_dir / f"{stem}_{digest}_{score_axis}.exact_ready_queue.json"


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return slug or "exact_ready_queue"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _unique_strings(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "DEFAULT_SCORE_AXIS",
    "FALSE_AUTHORITY",
    "REPAIR_METADATA_SCHEMA",
    "REPAIR_REPORT_SCHEMA",
    "SUPPORTED_REPAIR_SCORE_AXES",
    "plan_exact_ready_score_axis_repairs_from_audit",
    "repair_exact_ready_score_axis_queues",
]
