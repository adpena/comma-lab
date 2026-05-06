"""Fail-closed HNeRV entropy candidate-packet readiness manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.entropy_codec_gap_audit import (
    BYTE_EQUIVALENCE_BLOCKERS,
    COMMON_EXACT_NEXT_ARTIFACT_REQUIREMENTS,
    ENTROPY_OVERHEAD_TARGET_ACTIONS,
    FAIL_CLOSED_CRITERIA,
    TARGET_KIND_ARTIFACT_REQUIREMENTS,
    EntropyCodecGapAuditError,
    build_entropy_codec_gap_audit,
)
from tac.optimization.entropy_codec_gap_audit import (
    DISPATCH_BLOCKERS as ENTROPY_DISPATCH_BLOCKERS,
)
from tac.repo_io import read_json, repo_relative, sha256_file

SCHEMA_VERSION = 1
DISCOVERY_SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_entropy_candidate_packet"
DISCOVERY_TOOL_NAME = "tac.hnerv_entropy_candidate_packet.discover_candidate_audit_inputs"
ADAPTED_AUDIT_TOOL_NAME = "tac.hnerv_entropy_candidate_packet.hnerv_profile_entropy_overhead_adapter"
STRUCTURAL_RECODE_TOOL_NAME = "tac.hnerv_decoder_recode.build_structural_recode_profile"
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


def normalize_entropy_audit_payload(payload: Any) -> tuple[dict[str, Any], str]:
    """Normalize a supported audit/profile payload into entropy audit shape."""

    return _normalize_audit(payload)


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
        "missing_data_report": _discovery_missing_data_report(candidates),
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
    adapted = _adapt_hnerv_structural_recode_profile(payload)
    if adapted is not None:
        return adapted, "hnerv_structural_recode_profile_adapted_entropy_overhead_audit"
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


def _adapt_hnerv_structural_recode_profile(payload: Any) -> dict[str, Any] | None:
    if not _looks_like_hnerv_structural_recode_profile(payload):
        return None
    assert isinstance(payload, Mapping)
    hdc2 = _variant_by_name(payload, "range_prev_symbol_global_q_streams_plus_raw_scales")
    if hdc2 is None:
        raise HnervEntropyCandidatePacketError(
            "HNeRV structural recode profile is missing "
            "range_prev_symbol_global_q_streams_plus_raw_scales variant"
        )
    _require_true(hdc2.get("raw_equal"), "hdc2.raw_equal")
    _require_true(hdc2.get("q_roundtrip_equal"), "hdc2.q_roundtrip_equal")
    _require_true(hdc2.get("scale_roundtrip_equal"), "hdc2.scale_roundtrip_equal")
    hdc2_bytes = _positive_int(hdc2.get("bytes"), "hdc2.bytes")
    header_bytes = _positive_int(hdc2.get("header_bytes"), "hdc2.header_bytes")
    range_payload_bytes = _positive_int(hdc2.get("range_payload_bytes"), "hdc2.range_payload_bytes")
    raw_scale_bytes = _nonnegative_int(hdc2.get("raw_scale_bytes"), "hdc2.raw_scale_bytes")
    accounted_bytes = header_bytes + range_payload_bytes + raw_scale_bytes
    if accounted_bytes != hdc2_bytes:
        raise HnervEntropyCandidatePacketError(
            "HNeRV structural recode profile HDC2 accounting is inconsistent: "
            f"header_bytes + range_payload_bytes + raw_scale_bytes = {accounted_bytes}, "
            f"bytes = {hdc2_bytes}"
        )

    source_label = str(payload.get("source_label") or "hnerv_structural_recode_profile")
    source_archive_sha256 = str(payload.get("source_archive_sha256") or "")
    source_decoder_sha256 = str(payload.get("source_decoder_section_sha256") or "")
    stream_label = _ascii_label(f"{source_label}:hdc2_global_prev_symbol_contexts")
    entropy_summary = payload.get("entropy_summary")
    entropy_floor_plus_scales = _optional_positive_int_from_mapping(
        entropy_summary,
        "per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes",
    )
    encoded_payload_with_scales = range_payload_bytes + raw_scale_bytes
    payload_gap = (
        encoded_payload_with_scales - entropy_floor_plus_scales
        if entropy_floor_plus_scales is not None
        else None
    )
    targets = [
        _adapted_target_row(
            label=stream_label,
            source_label=source_label,
            target_kind="known_model_overhead",
            target_bytes=header_bytes,
            target_bytes_field="hdc2.header_bytes",
            accounting_source="hnerv_decoder_structural_recode_profile.hdc2_variant",
            actual_bytes=hdc2_bytes,
            entropy_floor_bytes=entropy_floor_plus_scales,
            hdc2=hdc2,
            source_archive_sha256=source_archive_sha256,
            source_decoder_sha256=source_decoder_sha256,
        )
    ]
    if payload_gap is not None and payload_gap > 0:
        targets.append(
            _adapted_target_row(
                label=stream_label,
                source_label=source_label,
                target_kind="known_payload_entropy_gap",
                target_bytes=payload_gap,
                target_bytes_field=(
                    "hdc2.range_payload_bytes_plus_raw_scale_bytes_minus_"
                    "per_tensor_prev_symbol_entropy_floor_plus_raw_scales_bytes"
                ),
                accounting_source="hnerv_decoder_structural_recode_profile.entropy_summary",
                actual_bytes=hdc2_bytes,
                entropy_floor_bytes=entropy_floor_plus_scales,
                hdc2=hdc2,
                source_archive_sha256=source_archive_sha256,
                source_decoder_sha256=source_decoder_sha256,
            )
        )
    targets.sort(
        key=lambda row: (
            -float(row["target_bytes"]),
            str(row["label"]),
            str(row["target_kind"]),
        )
    )
    for rank, row in enumerate(targets, start=1):
        row["rank"] = rank

    stream_row = {
        "label": stream_label,
        "source": source_label,
        "codec_surface": "src/tac/hnerv_decoder_recode.py",
        "evidence_grade": str(payload.get("evidence_grade") or "empirical"),
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(ENTROPY_DISPATCH_BLOCKERS),
        "fail_closed_criteria": list(FAIL_CLOSED_CRITERIA),
        "actual_bytes": hdc2_bytes,
        "known_encoded_payload_bytes": encoded_payload_with_scales,
        "known_model_overhead_bytes": header_bytes,
        "known_container_overhead_bytes": 0,
        "known_overhead_bytes": header_bytes,
        "known_unattributed_bytes": 0,
        "known_overhead_accounting_complete": True,
        "entropy_floor_bytes": entropy_floor_plus_scales,
        "known_payload_gap_to_entropy_floor_bytes": payload_gap,
        "source_archive_sha256": source_archive_sha256,
        "source_decoder_section_sha256": source_decoder_sha256,
        "source_decoder_section_bytes": payload.get("source_decoder_section_bytes"),
        "source_decoder_raw_bytes": payload.get("source_decoder_raw_bytes"),
        "q_stream_bytes": payload.get("q_stream_bytes"),
        "scale_stream_bytes": payload.get("scale_stream_bytes"),
        "hdc2_variant": {
            "variant": hdc2.get("variant"),
            "codec": hdc2.get("codec"),
            "bytes": hdc2_bytes,
            "header_bytes": header_bytes,
            "range_payload_bytes": range_payload_bytes,
            "raw_scale_bytes": raw_scale_bytes,
            "sha256": hdc2.get("sha256"),
            "raw_equal": hdc2.get("raw_equal"),
            "q_roundtrip_equal": hdc2.get("q_roundtrip_equal"),
            "scale_roundtrip_equal": hdc2.get("scale_roundtrip_equal"),
        },
        "missing_for_full_symbol_count_audit": [
            "full_hdc2_range_payload_symbol_counts",
            "full_context_conditioned_symbol_counts",
        ],
    }
    overhead = {
        "streams_with_known_accounting": 1,
        "complete_stream_accounting_count": 1,
        "total_known_encoded_payload_bytes": encoded_payload_with_scales,
        "total_known_model_overhead_bytes": header_bytes,
        "total_known_container_overhead_bytes": 0,
        "total_known_overhead_bytes": header_bytes,
        "total_known_unattributed_bytes": 0,
        "total_known_payload_gap_to_entropy_floor_bytes": payload_gap,
        "largest_known_overhead_streams": [
            {
                "label": stream_label,
                "actual_bytes": hdc2_bytes,
                "entropy_floor_bytes": entropy_floor_plus_scales,
                "known_encoded_payload_bytes": encoded_payload_with_scales,
                "known_model_overhead_bytes": header_bytes,
                "known_container_overhead_bytes": 0,
                "known_overhead_bytes": header_bytes,
                "known_unattributed_bytes": 0,
                "known_overhead_accounting_complete": True,
                "known_payload_gap_to_entropy_floor_bytes": payload_gap,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_blockers": list(ENTROPY_DISPATCH_BLOCKERS),
            }
        ],
    }
    return {
        "schema_version": 1,
        "tool": ADAPTED_AUDIT_TOOL_NAME,
        "source_tool": str(payload.get("tool") or STRUCTURAL_RECODE_TOOL_NAME),
        "planning_only": True,
        "score_claim": False,
        "score_evidence_grade": "invalid",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(ENTROPY_DISPATCH_BLOCKERS),
        "fail_closed_criteria": [
            *list(FAIL_CLOSED_CRITERIA),
            "refuse_if_hdc2_raw_roundtrip_not_true",
            "refuse_if_hdc2_accounting_does_not_sum_to_variant_bytes",
            "refuse_if_profile_lacks_exact_hdc2_variant_bytes",
        ],
        "source_label": source_label,
        "source_archive_sha256": source_archive_sha256,
        "source_decoder_section_sha256": source_decoder_sha256,
        "evidence_grade": str(payload.get("evidence_grade") or "empirical"),
        "stream_count": 1,
        "total_actual_bytes": hdc2_bytes,
        "known_overhead_accounting": overhead,
        "streams": [stream_row],
        "entropy_overhead_target_ranking": targets,
        "adapter_notes": [
            "adapted_from_raw_equal_hdc2_structural_recode_profile",
            "targets_use_profile_recorded_bytes_only",
            "section_entropy_bits_per_byte_summaries_are_not_treated_as_symbol_counts",
        ],
    }


def _looks_like_hnerv_structural_recode_profile(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if str(payload.get("tool") or "") == STRUCTURAL_RECODE_TOOL_NAME:
        return True
    required_profile_fields = {
        "source_decoder_section_bytes",
        "source_decoder_raw_bytes",
        "q_stream_bytes",
        "scale_stream_bytes",
        "variants",
    }
    if not required_profile_fields.issubset({str(key) for key in payload}):
        return False
    haystack = " ".join(
        str(payload.get(key) or "") for key in ("tool", "source_label", "source_archive_sha256")
    ).lower()
    return "hnerv" in haystack or _variant_by_name(payload, "range_prev_symbol_global_q_streams_plus_raw_scales") is not None


def _variant_by_name(payload: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    variants = payload.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes, bytearray)):
        return None
    for variant in variants:
        if isinstance(variant, Mapping) and str(variant.get("variant") or "") == name:
            return variant
    return None


def _adapted_target_row(
    *,
    label: str,
    source_label: str,
    target_kind: str,
    target_bytes: int,
    target_bytes_field: str,
    accounting_source: str,
    actual_bytes: int,
    entropy_floor_bytes: int | None,
    hdc2: Mapping[str, Any],
    source_archive_sha256: str,
    source_decoder_sha256: str,
) -> dict[str, Any]:
    action = ENTROPY_OVERHEAD_TARGET_ACTIONS[target_kind]
    return {
        "label": label,
        "source": source_label,
        "codec_surface": "src/tac/hnerv_decoder_recode.py",
        "target_kind": target_kind,
        "target_bytes": int(target_bytes),
        "target_bytes_field": target_bytes_field,
        "accounting_source": accounting_source,
        "target_action": action["target_action"],
        "required_next_artifact": action["required_next_artifact"],
        "exact_next_artifact_requirements": _exact_next_artifact_requirements(target_kind),
        "byte_equivalence_blockers": list(BYTE_EQUIVALENCE_BLOCKERS),
        "actual_bytes": actual_bytes,
        "entropy_floor_bytes": entropy_floor_bytes,
        "best_static_arithmetic_container_kind": "",
        "best_static_arithmetic_container_floor_bytes": None,
        "known_overhead_accounting_complete": True,
        "source_archive_sha256": source_archive_sha256,
        "source_decoder_section_sha256": source_decoder_sha256,
        "source_variant": {
            "variant": hdc2.get("variant"),
            "codec": hdc2.get("codec"),
            "sha256": hdc2.get("sha256"),
        },
        "readiness_stage": "planning_target_requires_byte_equivalent_artifacts",
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_byte_closed_candidate_build": False,
        "ready_for_meta_lagrangian_atom_export": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(ENTROPY_DISPATCH_BLOCKERS),
        "fail_closed_criteria": list(FAIL_CLOSED_CRITERIA),
        "meta_lagrangian_atom_export": {
            "schema": "meta_lagrangian_atom_export_v1",
            "export_ready": False,
            "ready_for_meta_lagrangian_atom_export": False,
            "export_blockers": [
                "planning_target_not_byte_closed_candidate",
                *list(BYTE_EQUIVALENCE_BLOCKERS),
                "missing_archive_manifest_path",
                "missing_archive_manifest_sha256",
            ],
            "atom_template": {
                "atom_id": f"{_atom_id_fragment(label)}:{target_kind}",
                "family": f"hnerv_{target_kind}",
                "family_group": "hnerv_rate_equivalent_recode",
                "pareto_scope": f"hnerv_rate_equivalent_recode:{_atom_id_fragment(label)}",
                "byte_delta": -int(target_bytes),
                "estimated_byte_delta": -int(target_bytes),
                "target_bytes": int(target_bytes),
                "target_bytes_field": target_bytes_field,
                "expected_seg_dist_delta": 0.0,
                "expected_pose_dist_delta": 0.0,
                "confidence": 0.0,
                "evidence_grade": "invalid_planning_target_until_byte_equivalent_candidate",
                "raw_equal": False,
                "score_claim": False,
                "dispatchable": False,
                "ready_for_exact_eval_dispatch": False,
                "interaction_assumptions": [
                    "rate_only_decoded_output_equivalence_required",
                ],
                "archive_manifest_path": "",
                "archive_manifest_sha256": "",
            },
        },
    }


def _exact_next_artifact_requirements(target_kind: str) -> list[str]:
    action = ENTROPY_OVERHEAD_TARGET_ACTIONS[target_kind]
    return _unique_ordered(
        [
            action["required_next_artifact"],
            *TARGET_KIND_ARTIFACT_REQUIREMENTS[target_kind],
            *COMMON_EXACT_NEXT_ARTIFACT_REQUIREMENTS,
        ]
    )


def _optional_positive_int_from_mapping(payload: Any, key: str) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if value is None:
        return None
    return _positive_int(value, key)


def _require_true(value: Any, context: str) -> None:
    if value is not True:
        raise HnervEntropyCandidatePacketError(f"{context} must be true")


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HnervEntropyCandidatePacketError(f"{context} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, context: str) -> int:
    integer = _nonnegative_int(value, context)
    if integer <= 0:
        raise HnervEntropyCandidatePacketError(f"{context} must be a positive integer")
    return integer


def _ascii_label(value: str) -> str:
    label = value.strip()
    if not label:
        raise HnervEntropyCandidatePacketError("adapted profile label must be nonempty")
    if any(ord(char) > 127 for char in label):
        raise HnervEntropyCandidatePacketError("adapted profile label must be ASCII")
    return label


def _atom_id_fragment(value: str) -> str:
    pieces = []
    for char in value.lower():
        pieces.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(pieces).split("_") if part) or "stream"


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
        row["missing_data"] = {
            "classification": "invalid_json",
            "required_inputs": ["parseable_json"],
            "candidate_source_files": [source_json],
        }
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
        row["missing_data"] = _candidate_missing_data(payload, source_json)
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
    row["missing_data"] = {
        "classification": "sufficient_for_entropy_overhead_audit",
        "required_inputs": [],
        "candidate_source_files": [source_json],
    }
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


def _candidate_missing_data(payload: Any, source_json: Mapping[str, Any]) -> dict[str, Any]:
    required_stream_profile = [
        "streams[*].label",
        "streams[*].actual_bytes_or_bytes_charged",
        "streams[*].symbol_counts_full_histogram",
    ]
    if _looks_like_hnerv_structural_recode_profile(payload):
        return {
            "classification": "hnerv_decoder_structural_recode_profile_incomplete",
            "required_inputs": [
                "variants[].variant=range_prev_symbol_global_q_streams_plus_raw_scales",
                "variants[].raw_equal=true",
                "variants[].q_roundtrip_equal=true",
                "variants[].scale_roundtrip_equal=true",
                "variants[].bytes",
                "variants[].header_bytes",
                "variants[].range_payload_bytes",
                "variants[].raw_scale_bytes",
            ],
            "candidate_source_files": [dict(source_json)],
            "notes": [
                "HDC2 fixture accounting is sufficient for a known_model_overhead target only when all exact bytes and raw-equivalence flags are present.",
            ],
        }
    if _has_hnerv_section_profile_shape(payload):
        return {
            "classification": "hnerv_section_profile_summary_only",
            "required_inputs": [
                *required_stream_profile,
                "or_profile_with_entropy_overhead_target_ranking",
                "or_hnerv_decoder_structural_recode_profile_with_raw_equal_hdc2_accounting",
            ],
            "candidate_source_files": [dict(source_json)],
            "notes": [
                "sections[*].entropy_bits_per_byte is a lossy summary and cannot reconstruct symbol_counts.",
                "section byte size, SHA-256, and entropy summaries are preserved as forensics but are not enough for a stream-count entropy audit.",
            ],
        }
    if isinstance(payload, Mapping) and "entropy_summary" in payload:
        return {
            "classification": "entropy_summary_without_full_stream_contract",
            "required_inputs": [
                *required_stream_profile,
                "or_hdc2_variant_bytes_header_bytes_range_payload_bytes_raw_scale_bytes_raw_equal_flags",
            ],
            "candidate_source_files": [dict(source_json)],
            "notes": [
                "top-symbol summaries and entropy-floor byte totals are not a replacement for the full stream profile contract unless paired with exact HDC2 overhead accounting.",
            ],
        }
    return {
        "classification": "no_supported_entropy_audit_or_profile_contract",
        "required_inputs": [
            "entropy_overhead_target_ranking[*].exact_next_artifact_requirements",
            *required_stream_profile,
            "or_hnerv_decoder_structural_recode_profile_with_raw_equal_hdc2_accounting",
        ],
        "candidate_source_files": [dict(source_json)],
        "notes": [
            "No entropy data was inferred from unrelated JSON keys.",
        ],
    }


def _discovery_missing_data_report(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required_inputs: list[str] = []
    candidate_source_files: list[dict[str, Any]] = []
    classifications: list[str] = []
    for row in candidates:
        missing = row.get("missing_data")
        if not isinstance(missing, Mapping):
            continue
        classification = str(missing.get("classification") or "")
        if classification and classification != "sufficient_for_entropy_overhead_audit":
            classifications.append(classification)
        for requirement in missing.get("required_inputs") or []:
            required_inputs.append(str(requirement))
        for source_file in missing.get("candidate_source_files") or []:
            if isinstance(source_file, Mapping):
                candidate_source_files.append(dict(source_file))
    return {
        "valid_entropy_audit_available": any(row.get("valid") is True for row in candidates),
        "required_inputs_if_no_valid_adapter": _unique_ordered(required_inputs),
        "candidate_source_files": _unique_file_records(candidate_source_files),
        "invalid_candidate_classifications": _unique_ordered(classifications),
    }


def _has_hnerv_section_profile_shape(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return isinstance(payload.get("sections"), list)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return any(isinstance(item, Mapping) and isinstance(item.get("sections"), list) for item in payload)
    return False


def _unique_file_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        path = str(record.get("path") or "")
        if not path or path in seen:
            continue
        out.append(dict(record))
        seen.add(path)
    return out


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
    "ADAPTED_AUDIT_TOOL_NAME",
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
    "normalize_entropy_audit_payload",
]
