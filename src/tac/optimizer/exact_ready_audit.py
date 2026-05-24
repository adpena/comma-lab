# SPDX-License-Identifier: MIT
"""Audit generated exact-ready queues against terminal dispatch evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.hdm8_selector_cuda_gate import validate_hdm8_selector_cuda_gate_context
from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_SCORE,
    active_claim_conflicts,
    as_bool,
    candidate_archive_byte_values,
    default_manifest_path,
    is_sha256,
    manifest_member_names,
    manifest_sha,
    manifest_size,
    read_json,
    repo_rel,
    resolve_path,
    runtime_dependency_manifest,
    sha256_file,
    terminal_claim_result_conflicts,
    utc_now,
    validate_serialized_archive_delta_contract,
)
from tac.zipwire_archive import inspect_zip_headers

AUDIT_SCHEMA = "optimizer_exact_ready_queue_terminal_evidence_audit_v1"
SUPPRESSION_MANIFEST_SCHEMA = "optimizer_exact_ready_queue_suppression_manifest_v1"
SUPPORTED_EXACT_READY_SCORE_AXES = frozenset({"contest_cuda"})


def _claim_lane_aliases(lane_id: str) -> tuple[str, ...]:
    """Return claim-ledger lane ids that legacy dispatch wrappers may use."""

    if lane_id.startswith("lane_"):
        return (lane_id, lane_id.removeprefix("lane_"))
    return (lane_id, f"lane_{lane_id}")


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


def _row_exact_eval_duplicate_key(row: Mapping[str, Any]) -> str | None:
    archive_sha = _row_archive_sha(row)
    runtime_sha = _row_runtime_content_sha(row)
    runtime_tree_sha = _row_runtime_tree_sha(row)
    score_axis, _ = _row_score_axis_with_blocker(row)
    if (
        archive_sha is None
        or runtime_sha is None
        or runtime_tree_sha is None
        or score_axis is None
    ):
        return None
    return f"{archive_sha}:{runtime_sha}:{runtime_tree_sha}:{score_axis}"


def _row_score_axis(row: Mapping[str, Any]) -> str | None:
    score_axis, _ = _row_score_axis_with_blocker(row)
    return score_axis


def _row_score_axis_with_blocker(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for key in ("score_axis", "target_score_axis"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            score_axis = value.strip().lower()
            if score_axis in SUPPORTED_EXACT_READY_SCORE_AXES:
                return score_axis, None
            return None, f"score_axis_unsupported:{value.strip()}"
    return None, "score_axis_missing"


def _row_runtime_content_sha(row: Mapping[str, Any]) -> str | None:
    for key in ("runtime_content_tree_sha256", "candidate_runtime_content_tree_sha256"):
        value = row.get(key)
        if is_sha256(value):
            return str(value).lower()
    runtime_manifest = row.get("runtime_manifest")
    if isinstance(runtime_manifest, Mapping):
        value = runtime_manifest.get("runtime_content_tree_sha256")
        if is_sha256(value):
            return str(value).lower()
    return None


def _row_archive_path(row: Mapping[str, Any], *, repo_root: Path, queue_dir: Path) -> Path | None:
    for key in ("candidate_archive_path", "archive_path"):
        path = resolve_path(row.get(key), repo_root=repo_root, queue_dir=queue_dir)
        if path is not None:
            return path
    return None


def _row_submission_dir(row: Mapping[str, Any], *, repo_root: Path, queue_dir: Path) -> Path | None:
    return resolve_path(row.get("submission_dir"), repo_root=repo_root, queue_dir=queue_dir)


def _row_inflate_sh_path(row: Mapping[str, Any], *, repo_root: Path, queue_dir: Path) -> Path | None:
    return resolve_path(row.get("inflate_sh_path"), repo_root=repo_root, queue_dir=queue_dir)


def _row_archive_manifest_path(
    row: Mapping[str, Any], *, repo_root: Path, queue_dir: Path
) -> Path | None:
    return resolve_path(row.get("archive_manifest_path"), repo_root=repo_root, queue_dir=queue_dir)


def _discover_packetir_exact_closure_records(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Return closure records keyed by candidate archive SHA.

    Closure manifests are generated after exact eval has already happened. They
    are not lane claims, so they need an archive-SHA guard here to keep a stale
    exact-ready queue from redispatching the same measured packet.
    """

    records: dict[str, list[dict[str, Any]]] = {}
    closure_paths = {
        *sorted((repo_root / "experiments" / "results").glob("**/closure.json")),
        *sorted((repo_root / ".omx" / "research").glob("*packetir*closure*.json")),
    }
    for path in sorted(closure_paths):
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("schema") != "packetir_exact_eval_closure_v1":
            continue
        archive = payload.get("archive")
        if not isinstance(archive, Mapping):
            continue
        archive_sha = archive.get("candidate_archive_sha256")
        if not is_sha256(archive_sha):
            continue
        exact_eval_duplicate_keys = [
            str(item.get("key"))
            for item in payload.get("exact_eval_duplicate_keys") or []
            if isinstance(item, Mapping) and item.get("key")
        ]
        blockers = payload.get("duplicate_dispatch_blockers")
        blockers = blockers if isinstance(blockers, list) else []
        if not blockers and not exact_eval_duplicate_keys:
            continue
        records.setdefault(str(archive_sha).lower(), []).append(
            {
                "closure_path": repo_rel(path, repo_root),
                "classification": payload.get("classification"),
                "lane_id": payload.get("lane_id"),
                "duplicate_dispatch_blockers": [str(blocker) for blocker in blockers],
                "exact_eval_duplicate_keys": exact_eval_duplicate_keys,
                "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch"),
                "score_claim": payload.get("score_claim"),
            }
        )
    return records


def _packetir_exact_closure_blockers(
    archive_sha: str | None,
    closure_records: Mapping[str, list[dict[str, Any]]],
    *,
    exact_eval_duplicate_key: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if archive_sha is None:
        return [], []
    records = list(closure_records.get(archive_sha.lower(), []))
    blockers: list[str] = []
    for record in records:
        classification = str(record.get("classification") or "unknown")
        closure_path = str(record.get("closure_path") or "<unknown>")
        for duplicate_blocker in record.get("duplicate_dispatch_blockers", []):
            blockers.append(
                "packetir_exact_closure_duplicate_dispatch:"
                f"{classification}:{duplicate_blocker}:{closure_path}"
            )
        for duplicate_key in record.get("exact_eval_duplicate_keys", []):
            if exact_eval_duplicate_key is not None and duplicate_key != exact_eval_duplicate_key:
                continue
            label = "packetir_exact_closure_exact_eval_duplicate_key"
            if exact_eval_duplicate_key is not None:
                label = "packetir_exact_closure_exact_eval_duplicate_key_match"
            blockers.append(f"{label}:{classification}:{duplicate_key}:{closure_path}")
    return blockers, records


def _discover_exact_cuda_result_review_records(
    repo_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Return exact-CUDA result-review records keyed by archive SHA.

    Terminal claim notes are intentionally compact and can miss runtime-content
    custody.  Result-review packets are the richer source of truth after eval
    recovery, so exact-ready queues should consult them before redispatching a
    same-archive packet whose wrapper runtime hash changed but whose consumed
    runtime content did not.
    """

    records: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((repo_root / ".omx" / "research").glob("*result_review*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("schema") != "tac_result_review_packet_v1":
            continue
        if payload.get("exact_cuda_evidence") is not True:
            continue
        if payload.get("score_claim_valid") is not True:
            continue
        custody = payload.get("custody")
        if not isinstance(custody, Mapping):
            continue
        archive_sha = custody.get("archive_sha256")
        if not is_sha256(archive_sha):
            continue
        runtime_custody = payload.get("runtime_custody")
        runtime_content_sha = None
        runtime_tree_sha = None
        if isinstance(runtime_custody, Mapping):
            value = runtime_custody.get("runtime_content_tree_sha256")
            if is_sha256(value):
                runtime_content_sha = str(value).lower()
            value = runtime_custody.get("runtime_tree_sha256")
            if is_sha256(value):
                runtime_tree_sha = str(value).lower()
        score_recomputation = payload.get("score_recomputation")
        score = None
        if isinstance(score_recomputation, Mapping):
            raw_score = score_recomputation.get("recomputed_score")
            if isinstance(raw_score, int | float):
                score = float(raw_score)
        if score is None and isinstance(payload.get("canonical_score"), int | float):
            score = float(payload["canonical_score"])
        score_axis = payload.get("score_axis")
        records.setdefault(str(archive_sha).lower(), []).append(
            {
                "review_path": repo_rel(path, repo_root),
                "lane_id": payload.get("lane_id"),
                "job_id": payload.get("job_id"),
                "archive_sha256": str(archive_sha).lower(),
                "archive_bytes": custody.get("archive_bytes"),
                "runtime_tree_sha256": runtime_tree_sha,
                "runtime_content_tree_sha256": runtime_content_sha,
                "score_axis": score_axis,
                "canonical_score": score,
                "measured_config_status": payload.get("measured_config_status"),
                "promotion_eligible": payload.get("promotion_eligible"),
                "ready_for_exact_eval_dispatch": payload.get(
                    "ready_for_exact_eval_dispatch"
                ),
                "score_claim": payload.get("score_claim"),
                "technique": payload.get("technique"),
            }
        )
    return records


def _exact_cuda_result_review_blockers(
    archive_sha: str | None,
    result_review_records: Mapping[str, list[dict[str, Any]]],
    *,
    lane_id: str | None,
    runtime_content_sha: str | None,
    score_axis: str | None,
    active_floor_score: float | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    if archive_sha is None:
        return [], []
    records = list(result_review_records.get(archive_sha.lower(), []))
    blockers: list[str] = []
    for record in records:
        record_axis = record.get("score_axis")
        record_axis_missing = not isinstance(record_axis, str) or not record_axis.strip()
        if isinstance(score_axis, str):
            if record_axis_missing:
                pass
            elif str(record_axis).strip().lower() != score_axis:
                continue
        record_lane = record.get("lane_id")
        same_lane = isinstance(lane_id, str) and record_lane == lane_id
        record_runtime_content = record.get("runtime_content_tree_sha256")
        same_runtime_content = (
            runtime_content_sha is not None
            and isinstance(record_runtime_content, str)
            and record_runtime_content == runtime_content_sha
        )
        if not same_lane and not same_runtime_content:
            continue
        review_path = str(record.get("review_path") or "<unknown>")
        record_id = f"{record_lane}:{record.get('job_id')}:{review_path}"
        if record_axis_missing:
            blockers.append(
                "result_review_exact_cuda_score_axis_missing_for_same_archive:"
                f"{record_id}"
            )
        score = record.get("canonical_score")
        if isinstance(score, int | float) and active_floor_score is not None:
            if score >= active_floor_score:
                blockers.append(
                    "result_review_exact_cuda_score_not_below_active_floor_"
                    f"for_same_archive:{score:.12g}>={active_floor_score:.12g}:"
                    f"{record_id}"
                )
            else:
                blockers.append(
                    "result_review_exact_cuda_score_already_below_active_floor_"
                    f"for_same_archive:{score:.12g}<{active_floor_score:.12g}:"
                    f"{record_id}"
                )
        if same_runtime_content:
            blockers.append(
                "result_review_exact_cuda_duplicate_runtime_content_for_same_archive:"
                f"{runtime_content_sha}:{record_id}"
            )
        elif same_lane:
            blockers.append(
                "result_review_exact_cuda_duplicate_lane_archive_for_same_archive:"
                f"{record_id}"
            )
    return blockers, records


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _ready_row_live_custody_blockers(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    queue_dir: Path,
    archive_sha: str | None,
    runtime_tree_sha: str | None,
    runtime_content_sha: str | None,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    facts: dict[str, Any] = {}

    archive_path = _row_archive_path(row, repo_root=repo_root, queue_dir=queue_dir)
    actual_archive_sha: str | None = None
    actual_archive_bytes: int | None = None
    zipwire: Mapping[str, Any] | None = None
    if archive_path is None:
        blockers.append("ready_row_archive_path_missing")
    else:
        facts["archive_path"] = repo_rel(archive_path, repo_root)
        if not archive_path.is_file():
            blockers.append("ready_row_archive_file_missing")
        else:
            actual_archive_sha = sha256_file(archive_path)
            actual_archive_bytes = archive_path.stat().st_size
            facts["actual_archive_sha256"] = actual_archive_sha
            facts["actual_archive_bytes"] = actual_archive_bytes
            if archive_sha is not None and actual_archive_sha != archive_sha:
                blockers.append(
                    "ready_row_archive_sha_mismatch:"
                    f"{actual_archive_sha}!={archive_sha}"
                )
            byte_values = candidate_archive_byte_values(row)
            if not byte_values:
                blockers.append("ready_row_archive_bytes_missing_or_invalid")
            elif len(set(byte_values.values())) > 1:
                details = ",".join(
                    f"{key}={value}" for key, value in sorted(byte_values.items())
                )
                blockers.append(f"ready_row_archive_bytes_field_mismatch:{details}")
            else:
                expected_bytes = next(iter(byte_values.values()))
                if actual_archive_bytes != expected_bytes:
                    blockers.append(
                        "ready_row_archive_bytes_mismatch:"
                        f"{actual_archive_bytes}!={expected_bytes}"
                    )
            delta_blockers, delta_facts = validate_serialized_archive_delta_contract(
                row,
                actual_candidate_archive_bytes=actual_archive_bytes,
            )
            blockers.extend(f"ready_row_{blocker}" for blocker in delta_blockers)
            if delta_facts:
                facts["serialized_archive_delta"] = delta_facts
            try:
                zipwire = inspect_zip_headers(archive_path)
            except (OSError, ValueError) as exc:
                blockers.append(f"ready_row_archive_zip_unreadable:{type(exc).__name__}")
                facts["archive_zip_error"] = str(exc)
            else:
                facts["archive_zip_strict"] = zipwire.get("zip_strict")
                facts["archive_zip_member_count"] = zipwire.get("member_count")
                if zipwire.get("zip_strict") is not True:
                    blockers.append("ready_row_archive_zip_not_strict")
                if int(zipwire.get("member_count") or 0) < 1:
                    blockers.append("ready_row_archive_zip_empty")

    submission_dir = _row_submission_dir(row, repo_root=repo_root, queue_dir=queue_dir)
    if submission_dir is None and archive_path is not None:
        submission_dir = archive_path.parent
    if submission_dir is None:
        blockers.append("ready_row_submission_dir_missing")
    else:
        facts["submission_dir"] = repo_rel(submission_dir, repo_root)
        if not submission_dir.is_dir():
            blockers.append("ready_row_submission_dir_missing")
        else:
            declared_inflate_sh = _row_inflate_sh_path(
                row, repo_root=repo_root, queue_dir=queue_dir
            )
            inflate_sh = submission_dir / "inflate.sh"
            facts["inflate_sh"] = repo_rel(inflate_sh, repo_root)
            if declared_inflate_sh is not None:
                facts["declared_inflate_sh_path"] = repo_rel(
                    declared_inflate_sh, repo_root
                )
                if not _same_resolved_path(declared_inflate_sh, inflate_sh):
                    blockers.append("ready_row_inflate_sh_path_mismatch")
            if not inflate_sh.is_file():
                blockers.append("ready_row_inflate_sh_missing")
            elif not os.access(inflate_sh, os.X_OK):
                blockers.append("ready_row_inflate_sh_not_executable")

            report_path = submission_dir / "report.txt"
            facts["report_path"] = repo_rel(report_path, repo_root)
            if not report_path.is_file():
                blockers.append("ready_row_report_txt_missing")

            manifest_path = _row_archive_manifest_path(
                row,
                repo_root=repo_root,
                queue_dir=queue_dir,
            ) or default_manifest_path(submission_dir)
            facts["archive_manifest_path"] = repo_rel(manifest_path, repo_root)
            if not manifest_path.is_file():
                blockers.append("ready_row_archive_manifest_missing")
            else:
                try:
                    raw_manifest = read_json(manifest_path)
                except (OSError, ValueError) as exc:
                    blockers.append(
                        f"ready_row_archive_manifest_json_invalid:{type(exc).__name__}"
                    )
                    facts["archive_manifest_error"] = str(exc)
                else:
                    if not isinstance(raw_manifest, Mapping):
                        blockers.append("ready_row_archive_manifest_not_object")
                    elif actual_archive_sha is not None:
                        manifest_archive_sha = manifest_sha(raw_manifest)
                        manifest_archive_size = manifest_size(raw_manifest)
                        facts["archive_manifest_sha256"] = manifest_archive_sha
                        facts["archive_manifest_bytes"] = manifest_archive_size
                        if manifest_archive_sha != actual_archive_sha:
                            blockers.append(
                                "ready_row_archive_manifest_sha_mismatch:"
                                f"{manifest_archive_sha}!={actual_archive_sha}"
                            )
                        if manifest_archive_size != actual_archive_bytes:
                            blockers.append(
                                "ready_row_archive_manifest_size_mismatch:"
                                f"{manifest_archive_size}!={actual_archive_bytes}"
                            )
                        if zipwire is not None:
                            member_names = {
                                str(member.get("name"))
                                for member in zipwire.get("members", [])
                                if isinstance(member, Mapping)
                            }
                            for name in manifest_member_names(raw_manifest):
                                if name not in member_names:
                                    blockers.append(
                                        f"ready_row_archive_manifest_member_mismatch:{name}"
                                    )
                        gate_blockers, gate_facts = validate_hdm8_selector_cuda_gate_context(
                            row,
                            raw_manifest,
                            expected_archive_sha256=actual_archive_sha,
                        )
                        blockers.extend(
                            f"ready_row_{blocker}" for blocker in gate_blockers
                        )
                        facts.update(gate_facts)
            try:
                runtime_manifest = runtime_dependency_manifest(submission_dir, repo_root)
            except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
                blockers.append(f"ready_row_runtime_manifest_error:{type(exc).__name__}")
                facts["runtime_manifest_error"] = str(exc)
            else:
                actual_runtime_sha = runtime_manifest.get("runtime_tree_sha256")
                actual_runtime_content_sha = runtime_manifest.get(
                    "runtime_content_tree_sha256"
                )
                facts["actual_runtime_tree_sha256"] = actual_runtime_sha
                facts["actual_runtime_content_tree_sha256"] = (
                    actual_runtime_content_sha
                )
                if runtime_tree_sha is None:
                    blockers.append("ready_row_runtime_tree_sha256_missing_or_invalid")
                elif actual_runtime_sha != runtime_tree_sha:
                    blockers.append(
                        "ready_row_runtime_tree_sha_mismatch:"
                        f"{actual_runtime_sha}!={runtime_tree_sha}"
                    )
                if runtime_content_sha is None:
                    blockers.append(
                        "ready_row_runtime_content_tree_sha256_missing_or_invalid"
                    )
                elif actual_runtime_content_sha != runtime_content_sha:
                    blockers.append(
                        "ready_row_runtime_content_tree_sha_mismatch:"
                        f"{actual_runtime_content_sha}!={runtime_content_sha}"
                    )

    return blockers, facts


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
    candidate_ids: Iterable[str] | None = None,
    claim_ttl_hours: float = 24.0,
    packetir_closure_records: Mapping[str, list[dict[str, Any]]] | None = None,
    result_review_records: Mapping[str, list[dict[str, Any]]] | None = None,
    ignore_active_claim_conflicts: bool = False,
    allowed_active_claim_platform: str | None = None,
    allowed_active_claim_instance_job_ids: Iterable[str] = (),
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
        tuple[
            str | None,
            str | None,
            str | None,
            str | None,
            str | None,
            tuple[str, ...],
        ]
    ] = set()
    candidate_id_filter = (
        {str(candidate_id) for candidate_id in candidate_ids}
        if candidate_ids is not None
        else None
    )
    closure_records = (
        packetir_closure_records
        if packetir_closure_records is not None
        else _discover_packetir_exact_closure_records(repo_root)
    )
    review_records = (
        result_review_records
        if result_review_records is not None
        else _discover_exact_cuda_result_review_records(repo_root)
    )
    row_count = 0
    queue_dir = queue_path.parent
    for row in _queue_rows(payload):
        row_count += 1
        if candidate_id_filter is not None:
            candidate_id = row.get("candidate_id")
            if str(candidate_id) not in candidate_id_filter:
                continue
        if row.get("ready_for_exact_eval_dispatch") is not True:
            continue
        lane_id = row.get("lane_id")
        archive_sha = _row_archive_sha(row)
        runtime_tree_sha = _row_runtime_tree_sha(row)
        runtime_content_sha = _row_runtime_content_sha(row)
        score_axis, score_axis_blocker = _row_score_axis_with_blocker(row)
        runtime_changed = as_bool(row.get("score_affecting_runtime_changed"))
        custody_blockers, custody_facts = _ready_row_live_custody_blockers(
            row,
            repo_root=repo_root,
            queue_dir=queue_dir,
            archive_sha=archive_sha,
            runtime_tree_sha=runtime_tree_sha,
            runtime_content_sha=runtime_content_sha,
        )
        if not isinstance(lane_id, str) or not lane_id.strip():
            blockers = ["lane_id_missing"]
        else:
            lane_aliases = _claim_lane_aliases(lane_id.strip())
            blockers = []
            if not ignore_active_claim_conflicts:
                for claim_lane_id in lane_aliases:
                    blockers.extend(
                        active_claim_conflicts(
                            claim_lane_id,
                            dispatch_claims_path=dispatch_claims_path,
                            ttl_hours=claim_ttl_hours,
                            allowed_active_claim_platform=allowed_active_claim_platform,
                            allowed_active_claim_instance_job_ids=(
                                allowed_active_claim_instance_job_ids
                            ),
                        )
                    )
            if archive_sha is None:
                blockers.append("archive_sha256_missing_or_invalid")
            else:
                if score_axis_blocker is not None:
                    blockers.append(score_axis_blocker)
                blockers.extend(custody_blockers)
                closure_blockers, closure_evidence = _packetir_exact_closure_blockers(
                    archive_sha,
                    closure_records,
                    exact_eval_duplicate_key=_row_exact_eval_duplicate_key(row),
                )
                if closure_evidence:
                    custody_facts["packetir_exact_closure_records"] = closure_evidence
                blockers.extend(closure_blockers)
                review_blockers, review_evidence = _exact_cuda_result_review_blockers(
                    archive_sha,
                    review_records,
                    lane_id=lane_id.strip(),
                    runtime_content_sha=_row_runtime_content_sha(row),
                    score_axis=score_axis,
                    active_floor_score=active_floor_score,
                )
                if review_evidence:
                    custody_facts["exact_cuda_result_review_records"] = review_evidence
                blockers.extend(review_blockers)
                for claim_lane_id in lane_aliases:
                    blockers.extend(
                        terminal_claim_result_conflicts(
                            claim_lane_id,
                            archive_sha,
                            dispatch_claims_path=dispatch_claims_path,
                            active_floor_score=active_floor_score,
                            runtime_tree_sha256=runtime_tree_sha,
                            runtime_content_tree_sha256=runtime_content_sha,
                            score_affecting_runtime_changed=runtime_changed,
                        )
                    )
        if not blockers:
            continue
        candidate_id = row.get("candidate_id")
        key = (
            str(candidate_id) if candidate_id is not None else None,
            lane_id if isinstance(lane_id, str) else None,
            archive_sha,
            runtime_tree_sha,
            runtime_content_sha,
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
                "runtime_content_tree_sha256": runtime_content_sha,
                "score_axis": score_axis,
                "score_affecting_runtime_changed": runtime_changed,
                "live_custody": custody_facts,
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
    scan_root: Path | str | Iterable[Path | str],
    patterns: Iterable[str] = (
        "**/exact_ready_queue.json",
        "**/*exact_ready_queue.json",
    ),
) -> list[Path]:
    found: dict[str, Path] = {}
    if isinstance(scan_root, (str, Path)):
        scan_roots: Iterable[Path | str] = (scan_root,)
    else:
        scan_roots = scan_root
    for root_item in scan_roots:
        root_path = Path(root_item)
        root = root_path if root_path.is_absolute() else repo_root / root_path
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
    candidate_ids: Iterable[str] | None = None,
    claim_ttl_hours: float = 24.0,
    ignore_active_claim_conflicts: bool = False,
    allowed_active_claim_platform: str | None = None,
    allowed_active_claim_instance_job_ids: Iterable[str] = (),
) -> dict[str, Any]:
    packetir_closure_records = _discover_packetir_exact_closure_records(repo_root)
    result_review_records = _discover_exact_cuda_result_review_records(repo_root)
    queues = [
        audit_exact_ready_queue(
            path,
            repo_root=repo_root,
            dispatch_claims_path=dispatch_claims_path,
            active_floor_score=active_floor_score,
            candidate_ids=candidate_ids,
            claim_ttl_hours=claim_ttl_hours,
            packetir_closure_records=packetir_closure_records,
            result_review_records=result_review_records,
            ignore_active_claim_conflicts=ignore_active_claim_conflicts,
            allowed_active_claim_platform=allowed_active_claim_platform,
            allowed_active_claim_instance_job_ids=(
                allowed_active_claim_instance_job_ids
            ),
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


def _norm(value: object) -> str:
    return "" if value is None else str(value)


def _row_key(queue_path: object, row: Mapping[str, object]) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    blockers = row.get("blockers", [])
    blocker_values = tuple(str(blocker) for blocker in blockers) if isinstance(blockers, list) else ()
    return (
        _norm(queue_path),
        _norm(row.get("candidate_id")),
        _norm(row.get("lane_id")),
        _norm(row.get("archive_sha256")),
        _norm(row.get("runtime_tree_sha256")),
        blocker_values,
    )


def _base_suppression_key(
    queue_path: object,
    row: Mapping[str, object],
) -> tuple[str, str, str, str, str]:
    return (
        _norm(queue_path),
        _norm(row.get("candidate_id")),
        _norm(row.get("lane_id")),
        _norm(row.get("archive_sha256")),
        _norm(row.get("runtime_tree_sha256")),
    )


def _manifest_entry_key(entry: Mapping[str, object]) -> tuple[str, str, str, str, str, tuple[str, ...]]:
    blockers = entry.get("blockers", [])
    blocker_values = tuple(str(blocker) for blocker in blockers) if isinstance(blockers, list) else ()
    return (
        _norm(entry.get("queue_path")),
        _norm(entry.get("candidate_id")),
        _norm(entry.get("lane_id")),
        _norm(entry.get("archive_sha256")),
        _norm(entry.get("runtime_tree_sha256")),
        blocker_values,
    )


def _terminal_score_blocker_key(blocker: str) -> tuple[str, str, str, str] | None:
    parts = blocker.split(":")
    if (
        blocker.startswith("same_lane_terminal_score_not_below_active_floor_for_same_archive:")
        and len(parts) >= 5
    ):
        return ("score_not_below_active_floor", parts[2], parts[3], parts[4])
    if (
        blocker.startswith("same_lane_terminal_score_already_below_active_floor_for_same_archive:")
        and len(parts) >= 5
    ):
        return ("score_already_below_active_floor", parts[2], parts[3], parts[4])
    return None


def _result_review_score_blocker_key(blocker: str) -> tuple[str, str, str, str] | None:
    parts = blocker.split(":")
    if (
        blocker.startswith(
            "result_review_exact_cuda_score_not_below_active_floor_for_same_archive:"
        )
        and len(parts) >= 5
    ):
        return ("review_score_not_below_active_floor", parts[2], parts[3], parts[4])
    if (
        blocker.startswith(
            "result_review_exact_cuda_score_already_below_active_floor_for_same_archive:"
        )
        and len(parts) >= 5
    ):
        return ("review_score_already_below_active_floor", parts[2], parts[3], parts[4])
    return None


def _terminal_score_blocker_keys(blockers: object) -> tuple[tuple[str, str, str, str], ...]:
    if not isinstance(blockers, list):
        return ()
    return tuple(
        sorted(
            key
            for blocker in blockers
            if (
                key := (
                    _terminal_score_blocker_key(str(blocker))
                    or _result_review_score_blocker_key(str(blocker))
                )
            )
            is not None
        )
    )


def _semantic_terminal_score_key(
    queue_path: object,
    row_or_entry: Mapping[str, object],
) -> tuple[str, str, str, str, str, tuple[tuple[str, str, str, str], ...]] | None:
    """Return a suppression key that ignores active-floor numeric drift.

    Terminal score blockers embed the active floor in text such as
    ``0.2265>=0.2089``.  That floor changes as the frontier improves, but the
    stale-row identity remains the same terminal same-lane/same-archive
    evaluation.  Suppression manifests must therefore match the terminal
    evidence tuple rather than the full blocker string for these blockers.
    """

    terminal_keys = _terminal_score_blocker_keys(row_or_entry.get("blockers", []))
    if not terminal_keys:
        return None
    return (*_base_suppression_key(queue_path, row_or_entry), terminal_keys)


def _terminal_evidence_from_blocker(blocker: str) -> dict[str, object] | None:
    parts = blocker.split(":")
    if blocker.startswith("same_lane_terminal_negative_for_same_archive:") and len(parts) >= 4:
        return {
            "kind": "terminal_exact_cuda_negative",
            "lane_id": parts[1],
            "job_id": parts[2],
            "status": parts[3],
        }
    if (
        blocker.startswith("same_lane_terminal_score_not_below_active_floor_for_same_archive:")
        and len(parts) >= 5
    ):
        return {
            "kind": "terminal_score_not_below_active_floor",
            "score_comparison": parts[1],
            "lane_id": parts[2],
            "job_id": parts[3],
            "status": parts[4],
        }
    if (
        blocker.startswith("same_lane_terminal_score_already_below_active_floor_for_same_archive:")
        and len(parts) >= 5
    ):
        return {
            "kind": "terminal_score_already_below_active_floor",
            "score_comparison": parts[1],
            "lane_id": parts[2],
            "job_id": parts[3],
            "status": parts[4],
        }
    if (
        blocker.startswith(
            "result_review_exact_cuda_score_not_below_active_floor_for_same_archive:"
        )
        and len(parts) >= 5
    ):
        return {
            "kind": "result_review_score_not_below_active_floor",
            "score_comparison": parts[1],
            "lane_id": parts[2],
            "job_id": parts[3],
            "status": parts[4],
        }
    if (
        blocker.startswith(
            "result_review_exact_cuda_score_already_below_active_floor_for_same_archive:"
        )
        and len(parts) >= 5
    ):
        return {
            "kind": "result_review_score_already_below_active_floor",
            "score_comparison": parts[1],
            "lane_id": parts[2],
            "job_id": parts[3],
            "status": parts[4],
        }
    if blocker.startswith("same_lane_terminal_refused_dispatch_for_same_archive:") and len(parts) >= 4:
        return {
            "kind": "terminal_refused_dispatch",
            "lane_id": parts[1],
            "job_id": parts[2],
            "status": parts[3],
        }
    return None


def _classify_blockers(blockers: Iterable[object]) -> tuple[str, str, list[str], dict[str, object] | None]:
    blocker_text = [str(blocker) for blocker in blockers]
    terminal_evidence = next(
        (evidence for blocker in blocker_text if (evidence := _terminal_evidence_from_blocker(blocker))),
        None,
    )
    if any(blocker.startswith("same_lane_terminal_negative_for_same_archive:") for blocker in blocker_text):
        return (
            "retired_by_terminal_exact_cuda_negative",
            "suppress_from_exact_ready_dispatch",
            [
                "Do not redispatch this same lane/archive/runtime identity.",
                "Reactivate only with a new archive SHA or a score-affecting runtime-tree change plus fresh custody.",
            ],
            terminal_evidence,
        )
    if any(
        blocker.startswith("same_lane_terminal_refused_dispatch_for_same_archive:")
        for blocker in blocker_text
    ):
        return (
            "retracted_by_terminal_refused_dispatch",
            "suppress_from_exact_ready_dispatch",
            [
                "Do not redispatch this same lane/archive identity after a terminal refused-dispatch finding.",
                "Reactivate only with a new archive SHA, a score-affecting runtime-tree change, or a cleared refusal blocker.",
            ],
            terminal_evidence,
        )
    if any(
        blocker.startswith("same_lane_terminal_score_not_below_active_floor_for_same_archive:")
        for blocker in blocker_text
    ):
        return (
            "retired_by_terminal_score_not_below_active_floor",
            "suppress_from_exact_ready_dispatch",
            [
                "Do not redispatch this measured configuration for score lowering.",
                "Reactivate only with a new charged-byte transform, score-affecting runtime change, or improved CUDA-calibrated objective.",
            ],
            terminal_evidence,
        )
    if any(
        blocker.startswith("same_lane_terminal_score_already_below_active_floor_for_same_archive:")
        for blocker in blocker_text
    ):
        return (
            "already_evaluated_by_terminal_exact_cuda_success",
            "suppress_duplicate_exact_ready_dispatch",
            [
                "Do not duplicate-dispatch this already evaluated identity.",
                "Use the terminal exact-CUDA artifact as the source of truth.",
            ],
            terminal_evidence,
        )
    if any(
        blocker.startswith(
            "result_review_exact_cuda_score_not_below_active_floor_for_same_archive:"
        )
        for blocker in blocker_text
    ):
        return (
            "retired_by_result_review_score_not_below_active_floor",
            "suppress_from_exact_ready_dispatch",
            [
                "Do not redispatch this result-reviewed exact-CUDA configuration.",
                "Reactivate only with a new charged-byte transform, score-affecting runtime-content change, or improved CUDA-calibrated objective.",
            ],
            terminal_evidence,
        )
    if any(
        blocker.startswith(
            "result_review_exact_cuda_score_already_below_active_floor_for_same_archive:"
        )
        for blocker in blocker_text
    ):
        return (
            "already_evaluated_by_result_review_exact_cuda_success",
            "suppress_duplicate_exact_ready_dispatch",
            [
                "Do not duplicate-dispatch this already result-reviewed exact-CUDA identity.",
                "Use the exact result-review packet as the source of truth.",
            ],
            terminal_evidence,
        )
    if any(
        blocker.startswith(
            "result_review_exact_cuda_duplicate_runtime_content_for_same_archive:"
        )
        for blocker in blocker_text
    ):
        return (
            "already_evaluated_by_result_review_runtime_content_duplicate",
            "suppress_duplicate_exact_ready_dispatch",
            [
                "Do not redispatch a same-archive packet with the same consumed runtime content.",
                "Reactivate only with a new archive SHA or a changed runtime-content tree plus fresh custody.",
            ],
            terminal_evidence,
        )
    if any(blocker.startswith("ready_row_runtime_tree_sha_mismatch:") for blocker in blocker_text):
        return (
            "retracted_stale_live_runtime_metadata",
            "retract_stale_exact_ready_row",
            [
                "Do not dispatch the persisted raw queue row.",
                "Regenerate the exact-ready queue from live packet custody before any future dispatch.",
            ],
            None,
        )
    return (
        "suppressed_by_exact_ready_audit_blocker",
        "suppress_from_exact_ready_dispatch",
        ["Do not dispatch until a newer queue row clears the exact-ready audit."],
        terminal_evidence,
    )


def build_suppression_manifest(payload: Mapping[str, object]) -> dict[str, object]:
    """Build a durable manifest classifying stale exact-ready rows.

    The manifest never mutates raw generated queue files.  It records why a
    row is no longer dispatchable and the evidence required to reactivate it.
    """

    entries: list[dict[str, object]] = []
    for queue in payload.get("queues", []):
        if not isinstance(queue, Mapping):
            continue
        queue_path = queue.get("queue_path")
        rows = queue.get("stale_ready_rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            blockers = row.get("blockers", [])
            blocker_list = blockers if isinstance(blockers, list) else []
            classification, action, criteria, terminal_evidence = _classify_blockers(blocker_list)
            entries.append(
                {
                    "queue_path": queue_path,
                    "candidate_id": row.get("candidate_id"),
                    "lane_id": row.get("lane_id"),
                    "archive_sha256": row.get("archive_sha256"),
                    "runtime_tree_sha256": row.get("runtime_tree_sha256"),
                    "score_affecting_runtime_changed": row.get("score_affecting_runtime_changed"),
                    "ready_for_exact_eval_dispatch_in_raw_queue": True,
                    "dispatch_allowed": False,
                    "classification": classification,
                    "operator_action": action,
                    "blockers": [str(blocker) for blocker in blocker_list],
                    "terminal_evidence": terminal_evidence,
                    "live_custody": row.get("live_custody", {}),
                    "reactivation_criteria": criteria,
                }
            )
    return {
        "schema": SUPPRESSION_MANIFEST_SCHEMA,
        "generated_at_utc": utc_now(),
        "source_audit_schema": payload.get("schema"),
        "raw_stale_ready_row_count": sum(
            len(queue.get("stale_ready_rows", []))
            for queue in payload.get("queues", [])
            if isinstance(queue, Mapping)
        ),
        "suppression_entry_count": len(entries),
        "entries": entries,
    }


def load_suppression_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("suppression manifest must be a JSON object")
    if payload.get("schema") != SUPPRESSION_MANIFEST_SCHEMA:
        raise ValueError(
            "suppression manifest schema mismatch: "
            f"{payload.get('schema')} != {SUPPRESSION_MANIFEST_SCHEMA}"
        )
    if not isinstance(payload.get("entries"), list):
        raise ValueError("suppression manifest entries must be a list")
    return payload


def apply_suppression_manifest(
    payload: dict[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Return an audit payload with manifest-classified stale rows suppressed."""

    entries = manifest.get("entries", [])
    suppression_index = {
        _manifest_entry_key(entry): entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("dispatch_allowed") is False
    }
    terminal_score_suppression_index = {
        key: entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("dispatch_allowed") is False
        and (key := _semantic_terminal_score_key(entry.get("queue_path"), entry)) is not None
    }
    raw_stale_count = int(payload.get("stale_ready_row_count") or 0)
    unresolved_count = 0
    suppressed_count = 0
    for queue in payload.get("queues", []):
        if not isinstance(queue, dict):
            continue
        queue_path = queue.get("queue_path")
        original_rows = queue.get("stale_ready_rows", [])
        if not isinstance(original_rows, list):
            continue
        unresolved_rows: list[dict[str, object]] = []
        suppressed_rows: list[dict[str, object]] = []
        for row in original_rows:
            if not isinstance(row, dict):
                continue
            match_basis = "exact_blockers"
            entry = suppression_index.get(_row_key(queue_path, row))
            if entry is None:
                semantic_key = _semantic_terminal_score_key(queue_path, row)
                if semantic_key is not None:
                    entry = terminal_score_suppression_index.get(semantic_key)
                    match_basis = "terminal_score_semantic"
            if entry is None:
                unresolved_rows.append(row)
                continue
            suppressed_row = dict(row)
            suppressed_row["suppression"] = {
                "manifest_path": repo_rel(manifest_path, repo_root),
                "match_basis": match_basis,
                "classification": entry.get("classification"),
                "operator_action": entry.get("operator_action"),
                "dispatch_allowed": entry.get("dispatch_allowed"),
                "reactivation_criteria": entry.get("reactivation_criteria", []),
            }
            suppressed_rows.append(suppressed_row)
        queue["raw_stale_ready_row_count"] = len(original_rows)
        queue["suppressed_ready_row_count"] = len(suppressed_rows)
        queue["suppressed_ready_rows"] = suppressed_rows
        queue["stale_ready_rows"] = unresolved_rows
        unresolved_count += len(unresolved_rows)
        suppressed_count += len(suppressed_rows)
    payload["raw_stale_ready_row_count"] = raw_stale_count
    payload["suppressed_ready_row_count"] = suppressed_count
    payload["stale_ready_row_count"] = unresolved_count
    payload["passed"] = unresolved_count == 0
    payload["suppression_manifest_path"] = repo_rel(manifest_path, repo_root)
    payload["suppression_manifest_schema"] = manifest.get("schema")
    return payload


__all__ = [
    "AUDIT_SCHEMA",
    "SUPPRESSION_MANIFEST_SCHEMA",
    "apply_suppression_manifest",
    "audit_exact_ready_queue",
    "audit_exact_ready_queues",
    "build_suppression_manifest",
    "discover_exact_ready_queues",
    "load_suppression_manifest",
]
