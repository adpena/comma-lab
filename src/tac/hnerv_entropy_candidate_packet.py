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
DISCOVERY_SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_entropy_candidate_packet"
DISCOVERY_TOOL_NAME = "tac.hnerv_entropy_candidate_packet.discover_candidate_audit_inputs"
DEFAULT_DISCOVERY_ROOTS = (
    "experiments/results",
    ".omx/research/artifacts",
)
DISCOVERY_REQUIRED_SOURCE_ARTIFACTS = [
    "hnerv_entropy_codec_gap_audit_json_with_entropy_overhead_target_ranking",
    "or_hnerv_stream_profile_json_with_streams_actual_bytes_and_symbol_counts",
]
DISCOVERY_PATH_HINTS = (
    "audit",
    "entropy",
    "packing",
    "profile",
    "stream",
)
DISCOVERY_EXCLUDED_PATH_HINTS = (
    "hnerv_entropy_packet_discovery",
)

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


def discover_candidate_audit_inputs(
    *,
    search_roots: Sequence[str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Discover valid HNeRV entropy audit/profile inputs without inventing data."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    roots = [Path(item) for item in (search_roots or DEFAULT_DISCOVERY_ROOTS)]
    candidate_paths = _discover_candidate_json_paths(roots, root)
    candidates = [_candidate_discovery_record(path, root) for path in candidate_paths]
    valid_inputs = [row for row in candidates if row["valid"] is True]
    selected = valid_inputs[0] if valid_inputs else None
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "tool": DISCOVERY_TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "score_evidence_grade": "invalid",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_packet_materialization": selected is not None,
        "dispatch_blockers": [
            "discovery_report_is_not_dispatch_authorization",
            "requires_valid_entropy_audit_or_stream_profile_before_packet_materialization",
            "requires_packet_manifest_operator_review_before_archive_work",
            "requires_lane_dispatch_claim_before_gpu",
            "requires_exact_cuda_auth_eval",
        ],
        "candidate_selection_policy": {
            "path_filter": (
                "repo-relative JSON path must contain hnerv plus one of "
                + ", ".join(DISCOVERY_PATH_HINTS)
            ),
            "valid_input_contract": list(DISCOVERY_REQUIRED_SOURCE_ARTIFACTS),
            "selection": "first_valid_input_by_repo_relative_path",
        },
        "search_roots": [repo_relative(_resolve_search_root(path, root), root) for path in roots],
        "candidate_input_count": len(candidates),
        "valid_input_count": len(valid_inputs),
        "selected_entropy_audit": selected["source_json"] if selected is not None else None,
        "missing_source_artifacts": [] if selected is not None else list(DISCOVERY_REQUIRED_SOURCE_ARTIFACTS),
        "candidate_inputs": candidates,
    }


def discovery_report_input_paths(report: Mapping[str, Any], repo_root: str | Path) -> list[Path]:
    """Return existing source JSON paths referenced by a discovery report."""

    root = Path(repo_root)
    paths: list[Path] = []
    for row in report.get("candidate_inputs") or []:
        if not isinstance(row, Mapping):
            continue
        source_json = row.get("source_json")
        if not isinstance(source_json, Mapping):
            continue
        path_text = source_json.get("path")
        if not path_text:
            continue
        path = root / str(path_text)
        if path.is_file():
            paths.append(path)
    return paths


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


def _discover_candidate_json_paths(paths: Sequence[Path], repo_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for search_root in paths:
        resolved_root = _resolve_search_root(search_root, repo_root)
        if resolved_root.is_file():
            candidates = [resolved_root] if resolved_root.suffix.lower() == ".json" else []
        elif resolved_root.is_dir():
            candidates = list(resolved_root.rglob("*.json"))
        else:
            candidates = []
        for candidate in candidates:
            if not candidate.is_file() or not _looks_like_discovery_candidate(candidate, repo_root):
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                out.append(candidate)
                seen.add(resolved)
    return sorted(out, key=lambda path: repo_relative(path, repo_root))


def _resolve_search_root(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _looks_like_discovery_candidate(path: Path, repo_root: Path) -> bool:
    text = repo_relative(path, repo_root).lower()
    if any(hint in text for hint in DISCOVERY_EXCLUDED_PATH_HINTS):
        return False
    return "hnerv" in text and any(hint in text for hint in DISCOVERY_PATH_HINTS)


def _candidate_discovery_record(path: Path, repo_root: Path) -> dict[str, Any]:
    source_json = _file_record("candidate_entropy_audit_or_stream_profile_json", path, repo_root)
    row: dict[str, Any] = {
        "path": source_json["path"],
        "source_json": source_json,
        "valid": False,
        "audit_source_kind": None,
        "audit_summary": None,
        "selected_target": None,
        "missing_source_artifacts": list(DISCOVERY_REQUIRED_SOURCE_ARTIFACTS),
        "rejection_reason": "",
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
    }
    try:
        payload = read_json(path)
    except Exception as exc:
        row["rejection_reason"] = f"invalid_json:{exc}"
        return row

    row["json_shape"] = _json_shape(payload)
    try:
        audit, audit_source_kind = _normalize_audit(payload)
        target = _select_target(
            audit,
            target_rank=None,
            target_label=None,
            target_kind=None,
        )
        _requirements_for_target(target)
    except HnervEntropyCandidatePacketError as exc:
        row["rejection_reason"] = str(exc)
        row["missing_required_fields"] = _missing_discovery_source_fields(payload)
        return row

    row["valid"] = True
    row["audit_source_kind"] = audit_source_kind
    row["audit_summary"] = {
        "tool": audit.get("tool"),
        "source_label": audit.get("source_label"),
        "stream_count": audit.get("stream_count"),
        "total_actual_bytes": audit.get("total_actual_bytes"),
        "target_count": len(audit.get("entropy_overhead_target_ranking") or []),
    }
    row["selected_target"] = {
        "rank": target.get("rank"),
        "label": target.get("label"),
        "target_kind": target.get("target_kind"),
        "target_bytes": target.get("target_bytes"),
        "required_next_artifact": target.get("required_next_artifact"),
    }
    row["missing_source_artifacts"] = []
    return row


def _json_shape(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return {
            "kind": "object",
            "keys": sorted(str(key) for key in payload),
        }
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return {
            "kind": "array",
            "length": len(payload),
        }
    return {"kind": type(payload).__name__}


def _missing_discovery_source_fields(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return list(DISCOVERY_REQUIRED_SOURCE_ARTIFACTS)
    missing: list[str] = []
    if not isinstance(payload.get("entropy_overhead_target_ranking"), list):
        missing.append("entropy_overhead_target_ranking")
    streams = payload.get("streams")
    if not isinstance(streams, list):
        missing.append("streams")
    elif not streams:
        missing.append("streams_nonempty")
    else:
        for index, stream in enumerate(streams):
            if not isinstance(stream, Mapping):
                missing.append(f"streams[{index}]_object")
                continue
            if "actual_bytes" not in stream:
                missing.append(f"streams[{index}].actual_bytes")
            if "symbol_counts" not in stream:
                missing.append(f"streams[{index}].symbol_counts")
    return _unique_ordered(missing)


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
    "DISCOVERY_REQUIRED_SOURCE_ARTIFACTS",
    "PACKET_DISPATCH_BLOCKERS",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "HnervEntropyCandidatePacketError",
    "build_candidate_packet_manifest",
    "discover_candidate_audit_inputs",
    "discovery_report_input_paths",
    "existing_artifact_input_paths",
]
