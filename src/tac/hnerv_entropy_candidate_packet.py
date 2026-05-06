"""Fail-closed HNeRV entropy candidate-packet readiness manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.entropy_codec_gap_audit import (
    EntropyCodecGapAuditError,
    build_entropy_codec_gap_audit,
)
from tac.repo_io import read_json, repo_relative, sha256_file

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_entropy_candidate_packet"

BLOCKER_TO_REQUIREMENT = {
    "missing_source_archive_manifest": "source_archive_manifest_with_archive_sha256_bytes_and_runtime_tree_sha256",
    "missing_source_stream_sha256_and_byte_range": "source_stream_section_sha256_byte_range_and_symbol_count",
    "missing_candidate_stream_sha256_and_byte_range": "candidate_stream_section_sha256_byte_range_and_byte_count",
    "missing_decoded_output_byte_equivalence_report": "old_new_decoded_output_sha256_equality_report",
    "missing_roundtrip_decode_validation_manifest": "roundtrip_decode_validation_manifest",
    "missing_candidate_archive_manifest": "candidate_archive_manifest_with_member_sha256s",
    "missing_runtime_tree_parity_manifest": "runtime_tree_parity_manifest",
}

PACKET_DISPATCH_BLOCKERS = [
    "packet_manifest_is_not_dispatch_authorization",
    "requires_operator_review_of_byte_equivalence_runtime_parity_and_archive_manifest",
    "requires_lane_dispatch_claim_before_gpu",
    "requires_exact_cuda_auth_eval",
]


class HnervEntropyCandidatePacketError(ValueError):
    """Raised when candidate-packet readiness input is malformed."""


def build_candidate_packet_manifest(
    entropy_audit_path: str | Path,
    *,
    target_rank: int | None = None,
    target_label: str | None = None,
    target_kind: str | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic readiness manifest for one entropy target row.

    The manifest is local custody/readiness state only. It never dispatches,
    never evaluates, and never turns a rate-only target into a score claim.
    """

    audit_path = Path(entropy_audit_path)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    payload = read_json(audit_path)
    audit, audit_source_kind = _normalize_audit(payload)
    target = _select_target(
        audit,
        target_rank=target_rank,
        target_label=target_label,
        target_kind=target_kind,
    )
    requirements = _requirements_for_target(target)
    artifacts = {str(key): Path(value) for key, value in (artifact_paths or {}).items()}
    unknown = sorted(set(artifacts) - set(requirements))
    if unknown:
        raise HnervEntropyCandidatePacketError(
            "artifact supplied for unknown selected-target requirement: " + ", ".join(unknown)
        )

    requirement_rows = [
        _requirement_record(requirement_id, artifacts.get(requirement_id), root)
        for requirement_id in requirements
    ]
    missing_artifacts = [
        str(row["id"])
        for row in requirement_rows
        if row["available"] is not True
    ]
    invalid_json_artifacts = [
        str(row["id"])
        for row in requirement_rows
        if row.get("json_parse_error")
    ]
    unsatisfied_byte_equivalence_blockers = _unsatisfied_byte_equivalence_blockers(
        target,
        available_ids={
            str(row["id"])
            for row in requirement_rows
            if row["available"] is True
        },
    )
    readiness_blockers = _unique_ordered(
        [
            *(f"missing_artifact:{artifact_id}" for artifact_id in missing_artifacts),
            *(f"invalid_json_artifact:{artifact_id}" for artifact_id in invalid_json_artifacts),
            *unsatisfied_byte_equivalence_blockers,
        ]
    )
    if target.get("ready_for_exact_eval_dispatch") is True:
        readiness_blockers.append("input_target_ready_for_exact_eval_claim_rejected")

    available_inputs = [
        _file_record("entropy_audit_json", audit_path, root),
        *[
            _file_record(str(row["id"]), Path(str(row["path"])), root, already_repo_relative=True)
            for row in requirement_rows
            if row["available"] is True and row.get("path")
        ],
    ]
    ready_for_local_packet_review = not missing_artifacts and not invalid_json_artifacts
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "score_evidence_grade": "invalid",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_byte_closed_candidate_build": ready_for_local_packet_review,
        "ready_for_archive_preflight": ready_for_local_packet_review,
        "ready_for_local_packet_review": ready_for_local_packet_review,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": _unique_ordered(
            [
                *readiness_blockers,
                *PACKET_DISPATCH_BLOCKERS,
            ]
        ),
        "readiness_blockers": readiness_blockers,
        "missing_artifacts": missing_artifacts,
        "invalid_json_artifacts": invalid_json_artifacts,
        "audit_source": _file_record("entropy_audit_json", audit_path, root),
        "audit_source_kind": audit_source_kind,
        "audit_summary": {
            "tool": audit.get("tool"),
            "source_label": audit.get("source_label"),
            "stream_count": audit.get("stream_count"),
            "total_actual_bytes": audit.get("total_actual_bytes"),
            "target_count": len(audit.get("entropy_overhead_target_ranking") or []),
        },
        "selected_target": {
            "rank": target.get("rank"),
            "label": target.get("label"),
            "target_kind": target.get("target_kind"),
            "target_bytes": target.get("target_bytes"),
            "target_bytes_field": target.get("target_bytes_field"),
            "required_next_artifact": target.get("required_next_artifact"),
            "target_action": target.get("target_action"),
            "row": target,
        },
        "packet_requirements": requirement_rows,
        "available_inputs": available_inputs,
        "source_artifact_requirements": [
            str(row["id"])
            for row in requirement_rows
            if row.get("requirement_group") == "source_custody"
        ],
        "byte_equivalence_requirements": [
            str(row["id"])
            for row in requirement_rows
            if row.get("requirement_group") == "byte_equivalence"
        ],
        "runtime_parity_requirements": [
            str(row["id"])
            for row in requirement_rows
            if row.get("requirement_group") == "runtime_parity"
        ],
        "archive_manifest_requirements": [
            str(row["id"])
            for row in requirement_rows
            if row.get("requirement_group") == "archive_manifest"
        ],
    }


def existing_artifact_input_paths(
    entropy_audit_path: str | Path,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> list[Path]:
    """Return existing input files for wrapper custody manifests."""

    out = [Path(entropy_audit_path)]
    for value in (artifact_paths or {}).values():
        path = Path(value)
        if path.is_file():
            out.append(path)
    return out


def _normalize_audit(payload: Any) -> tuple[dict[str, Any], str]:
    if isinstance(payload, Mapping) and isinstance(payload.get("entropy_overhead_target_ranking"), list):
        return dict(payload), "entropy_codec_gap_audit"
    if isinstance(payload, Mapping):
        streams = payload.get("streams")
        source_label = str(payload.get("source_label") or "")
        evidence_grade = str(payload.get("evidence_grade") or "empirical")
    else:
        streams = payload
        source_label = ""
        evidence_grade = "empirical"
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes, bytearray)):
        raise HnervEntropyCandidatePacketError(
            "entropy audit/profile JSON must contain entropy_overhead_target_ranking or streams"
        )
    try:
        audit = build_entropy_codec_gap_audit(
            streams,
            source_label=source_label,
            evidence_grade=evidence_grade,
        )
    except EntropyCodecGapAuditError as exc:
        raise HnervEntropyCandidatePacketError(f"stream profile rejected by entropy audit: {exc}") from exc
    return audit, "stream_profile_built_entropy_codec_gap_audit"


def _select_target(
    audit: Mapping[str, Any],
    *,
    target_rank: int | None,
    target_label: str | None,
    target_kind: str | None,
) -> dict[str, Any]:
    targets = audit.get("entropy_overhead_target_ranking")
    if not isinstance(targets, list) or not targets:
        raise HnervEntropyCandidatePacketError("audit missing entropy_overhead_target_ranking")
    filters_requested = target_rank is not None or target_label is not None or target_kind is not None
    effective_rank = 1 if not filters_requested else target_rank
    matches = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise HnervEntropyCandidatePacketError("entropy_overhead_target_ranking rows must be objects")
        if effective_rank is not None and int(target.get("rank") or -1) != int(effective_rank):
            continue
        if target_label is not None and str(target.get("label") or "") != str(target_label):
            continue
        if target_kind is not None and str(target.get("target_kind") or "") != str(target_kind):
            continue
        matches.append(dict(target))
    if not matches:
        raise HnervEntropyCandidatePacketError("no entropy-overhead target row matched the selection")
    if len(matches) > 1:
        raise HnervEntropyCandidatePacketError(
            "target selection matched multiple rows; add --target-rank or --target-kind"
        )
    return matches[0]


def _requirements_for_target(target: Mapping[str, Any]) -> list[str]:
    raw_requirements = target.get("exact_next_artifact_requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise HnervEntropyCandidatePacketError("selected target missing exact_next_artifact_requirements")
    blockers = target.get("byte_equivalence_blockers")
    if blockers is None:
        blockers = []
    if not isinstance(blockers, list):
        raise HnervEntropyCandidatePacketError("selected target byte_equivalence_blockers must be a list")
    return _unique_ordered(
        [
            *(str(item) for item in raw_requirements),
            *(BLOCKER_TO_REQUIREMENT[str(blocker)] for blocker in blockers if str(blocker) in BLOCKER_TO_REQUIREMENT),
        ]
    )


def _requirement_record(requirement_id: str, path: Path | None, repo_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": requirement_id,
        "requirement_group": _requirement_group(requirement_id),
        "provided": path is not None,
        "path": "",
        "available": False,
        "bytes": None,
        "sha256": None,
        "missing_reason": "not_provided",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if path is None:
        return row
    row["path"] = repo_relative(path, repo_root)
    if not path.is_file():
        row["missing_reason"] = "path_missing"
        return row
    row["bytes"] = path.stat().st_size
    row["sha256"] = sha256_file(path)
    row["missing_reason"] = ""
    row["available"] = True
    if path.suffix.lower() == ".json":
        try:
            read_json(path)
            row["json_parseable"] = True
        except Exception as exc:
            row["available"] = False
            row["json_parseable"] = False
            row["json_parse_error"] = str(exc)
            row["missing_reason"] = "json_parse_error"
    return row


def _file_record(
    record_id: str,
    path: Path,
    repo_root: Path,
    *,
    already_repo_relative: bool = False,
) -> dict[str, Any]:
    actual_path = path if not already_repo_relative else repo_root / path
    return {
        "id": record_id,
        "path": repo_relative(actual_path, repo_root),
        "bytes": actual_path.stat().st_size,
        "sha256": sha256_file(actual_path),
    }


def _unsatisfied_byte_equivalence_blockers(
    target: Mapping[str, Any],
    *,
    available_ids: set[str],
) -> list[str]:
    blockers = target.get("byte_equivalence_blockers") or []
    out: list[str] = []
    for blocker in blockers:
        blocker_text = str(blocker)
        requirement_id = BLOCKER_TO_REQUIREMENT.get(blocker_text)
        if requirement_id is None or requirement_id not in available_ids:
            out.append(blocker_text)
    return out


def _requirement_group(requirement_id: str) -> str:
    if requirement_id.startswith("source_"):
        return "source_custody"
    if requirement_id in {
        "candidate_archive_manifest_with_member_sha256s",
        "strict_pre_submission_compliance_json",
    }:
        return "archive_manifest"
    if requirement_id == "runtime_tree_parity_manifest":
        return "runtime_parity"
    if (
        requirement_id.startswith("candidate_stream_")
        or requirement_id.startswith("old_new_")
        or requirement_id.startswith("roundtrip_")
        or requirement_id.startswith("byte_equivalent_")
        or requirement_id.startswith("static_arithmetic_")
        or requirement_id.startswith("container_")
    ):
        return "byte_equivalence"
    return "target_specific"


def _unique_ordered(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "BLOCKER_TO_REQUIREMENT",
    "PACKET_DISPATCH_BLOCKERS",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "HnervEntropyCandidatePacketError",
    "build_candidate_packet_manifest",
    "existing_artifact_input_paths",
]
