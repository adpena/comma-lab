# SPDX-License-Identifier: MIT
"""Compile final-byte operation artifact hints into materializer contexts."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.deterministic_compiler import (
    packetir_operation_set_bridge_contract,
)

from .byte_shaving_campaign_queue import (
    MATERIALIZER_BACKLOG_SCHEMA,
    MATERIALIZER_CONTEXTS_SCHEMA,
)
from .byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
)
from .experiment_queue import ExperimentQueueError

CONTEXT_COMPILER_SCHEMA = "final_byte_operation_context_compiler.v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_work_id(value: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return f"materializer_work_{safe or 'row'}"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return ordered_unique(str(item).strip() for item in value if str(item).strip())
    return []


def _int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, raw_count in value.items():
        if isinstance(raw_count, bool):
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            out[str(key)] = count
    return out


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _canonical_sha256(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    lowered = text.lower()
    if len(lowered) == 64 and all(char in "0123456789abcdef" for char in lowered):
        return lowered
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_text(context: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = _text(context.get(key))
        if value is not None:
            return value
    return None


def _first_text_with_key(
    context: Mapping[str, Any],
    keys: Sequence[str],
) -> tuple[str | None, str | None]:
    for key in keys:
        value = _text(context.get(key))
        if value is not None:
            return key, value
    return None, None


def _require_no_truthy_authority_recursive(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        require_no_truthy_authority_fields(value, context=context)
        for key, child in value.items():
            _require_no_truthy_authority_recursive(
                child,
                context=f"{context}.{key}",
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _require_no_truthy_authority_recursive(
                child,
                context=f"{context}[{index}]",
            )


def _packetir_operation_set_contract_context() -> dict[str, Any]:
    contract = packetir_operation_set_bridge_contract()
    return {
        "packetir_operation_set_contract": contract,
        "recommended_ir_schema": contract["recommended_ir_schema"],
        "required_order": list(contract["required_order"]),
        "required_proofs": list(contract["required_proofs"]),
    }


def _hint_mapping(artifact_map: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if artifact_map is None:
        return {}
    _require_no_truthy_authority_recursive(artifact_map, context="artifact_map")
    for key in ("contexts", "artifacts", "hints", "artifact_map"):
        value = artifact_map.get(key)
        if isinstance(value, Mapping):
            return value
    return artifact_map


def _merged_hints(
    hints: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    keys = [
        str(row.get("target_kind") or ""),
        str(row.get("materializer_id") or ""),
        f"{row.get('unit_kind') or ''}:{row.get('operation_family') or ''}",
        str(row.get("backlog_key") or ""),
    ]
    keys.extend(str(item) for item in _as_list(row.get("source_unit_ids")))
    merged: dict[str, Any] = _inline_row_hints(row)
    for key in ordered_unique(keys):
        value = hints.get(key)
        if isinstance(value, Mapping):
            merged.update(dict(value))
    return merged


def _inline_row_hints(row: Mapping[str, Any]) -> dict[str, Any]:
    params = row.get("operation_params")
    if not isinstance(params, Mapping):
        return {}
    hints = dict(params)
    if "archive_section" in hints and "target_sections" not in hints:
        hints["target_sections"] = _string_list(hints["archive_section"])
    if "section_name" in hints and "target_sections" not in hints:
        hints["target_sections"] = _string_list(hints["section_name"])
    if "target_section" in hints and "target_sections" not in hints:
        hints["target_sections"] = _string_list(hints["target_section"])
    if "packet_member" in hints and "member_name" not in hints:
        hints["member_name"] = hints["packet_member"]
    return hints


def _context_keys(row: Mapping[str, Any]) -> list[str]:
    keys = [
        str(row.get("backlog_key") or ""),
        str(row.get("materializer_id") or ""),
        str(row.get("target_kind") or ""),
    ]
    keys.extend(str(item) for item in _as_list(row.get("source_unit_ids")))
    return ordered_unique(keys)


def _output_paths(
    *,
    row: Mapping[str, Any],
    hints: Mapping[str, Any],
    repo_root: Path,
    default_output_root: Path | None,
) -> tuple[str | None, str | None]:
    output_archive = _text(hints.get("output_archive"))
    json_out = _text(hints.get("json_out")) or _text(hints.get("manifest_out"))
    if output_archive is not None:
        if json_out is None:
            json_out = Path(output_archive).with_suffix(".json").as_posix()
        return output_archive, json_out
    if default_output_root is None:
        return None, json_out
    root = default_output_root
    if not root.is_absolute():
        root = repo_root / root
    work_id = _safe_work_id(str(row.get("backlog_key") or "archive_section"))
    output_path = root / f"{work_id}.zip"
    manifest_path = root / f"{work_id}.json"
    return output_path.as_posix(), manifest_path.as_posix()


def _copy_common_materializer_controls(
    context: dict[str, Any],
    hints: Mapping[str, Any],
) -> None:
    """Carry controls consumed by the family materializer command builder."""

    for key, value in (
        ("runtime_consumption_proof", _first_text(hints, ("runtime_consumption_proof",))),
        ("expected_output_sha256", _first_text(hints, ("expected_output_sha256",))),
        ("expected_manifest_sha256", _first_text(hints, ("expected_manifest_sha256",))),
        (
            "expected_existing_output_sha256",
            _first_text(hints, ("expected_existing_output_sha256",)),
        ),
        (
            "expected_existing_manifest_sha256",
            _first_text(hints, ("expected_existing_manifest_sha256",)),
        ),
    ):
        if value is not None:
            context[key] = value
    for key in ("min_free_bytes",):
        value = hints.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            context[key] = value
    for key in ("allow_size_regression", "allow_rate_regression", "allow_overwrite"):
        if hints.get(key) is True:
            context[key] = True


def _archive_section_context_row(
    row: Mapping[str, Any],
    *,
    hints: Mapping[str, Any],
    repo_root: Path,
    default_output_root: Path | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    blockers: list[str] = []
    source_archive = _first_text(hints, ("source_archive", "archive_path"))
    section_manifest_key, section_manifest = _first_text_with_key(
        hints,
        (
            "section_manifest",
            "parser_section_manifest",
            "packet_ir_manifest",
        ),
    )
    target_sections = ordered_unique(
        [
            *_string_list(hints.get("target_sections")),
            *_string_list(hints.get("target_section")),
            *_string_list(hints.get("section_names")),
            *_string_list(hints.get("section_name")),
        ]
    )
    output_archive, json_out = _output_paths(
        row=row,
        hints=hints,
        repo_root=repo_root,
        default_output_root=default_output_root,
    )
    if source_archive is None:
        blockers.append("materializer_context_missing:archive_path")
    if output_archive is None:
        blockers.append("materializer_context_missing:output_archive")
    if section_manifest is None:
        blockers.append("materializer_context_missing:section_manifest")
    context.update(
        {
            "label": _first_text(hints, ("label", "candidate_id")) or _safe_work_id(str(row.get("backlog_key") or "")),
            "target_sections": target_sections,
            "context_blockers": blockers,
            **_packetir_operation_set_contract_context(),
            **FALSE_AUTHORITY,
        }
    )
    if source_archive is not None:
        context["source_archive"] = source_archive
        context["archive_path"] = source_archive
    if section_manifest is not None:
        context["section_manifest"] = section_manifest
        context["section_manifest_source_key"] = section_manifest_key
        if section_manifest_key != "section_manifest":
            context[str(section_manifest_key)] = section_manifest
    if output_archive is not None:
        context["output_archive"] = output_archive
    if json_out is not None:
        context["json_out"] = json_out
        context["output_manifest"] = json_out
    for key in ("quality", "qualities", "brotli_quality", "brotli_qualities"):
        values = _string_list(hints.get(key))
        if values:
            context[key] = values
    _copy_common_materializer_controls(context, hints)
    return _context_row_payload(row, context=context, blockers=blockers)


def _packet_member_context_row(
    row: Mapping[str, Any],
    *,
    hints: Mapping[str, Any],
    repo_root: Path,
    default_output_root: Path | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    blockers: list[str] = []
    archive_path = _first_text(hints, ("archive_path", "source_archive"))
    packet_member_manifest = _first_text(
        hints,
        ("packet_member_manifest", "member_manifest"),
    )
    member_name = _first_text(
        hints,
        ("member_name", "archive_member_name", "packet_member_name"),
    )
    output_archive, json_out = _output_paths(
        row=row,
        hints=hints,
        repo_root=repo_root,
        default_output_root=default_output_root,
    )
    if archive_path is None:
        blockers.append("materializer_context_missing:archive_path")
    if output_archive is None:
        blockers.append("materializer_context_missing:output_archive")
    context.update(
        {
            "context_blockers": blockers,
            **_packetir_operation_set_contract_context(),
            **FALSE_AUTHORITY,
        }
    )
    if archive_path is not None:
        context["archive_path"] = archive_path
    if packet_member_manifest is not None:
        context["packet_member_manifest"] = packet_member_manifest
    if member_name is not None:
        context["member_name"] = member_name
    member_names = _string_list(
        hints.get("member_names")
        or hints.get("archive_member_names")
        or hints.get("packet_member_names")
    )
    if member_names:
        context["member_names"] = member_names
    member_selection = _first_text(
        hints,
        ("member_selection", "zip_member_selection", "packet_member_selection"),
    )
    if member_selection is not None:
        context["member_selection"] = member_selection
    if any(hints.get(key) is True for key in ("all_members", "select_all_members")):
        context["all_members"] = True
    header_elision_contract = _first_text(
        hints,
        (
            "header_elision_contract",
            "zip_header_contract",
            "zip_header_elision_contract",
        ),
    )
    if header_elision_contract is not None:
        context["header_elision_contract"] = header_elision_contract
    merge_contract = _first_text(
        hints,
        (
            "merge_contract",
            "member_merge_contract",
            "packet_member_merge_contract",
        ),
    )
    if merge_contract is not None:
        context["merge_contract"] = merge_contract
    is_member_merge = (
        row.get("operation_family") == "member_merge"
        and row.get("target_kind") == PACKET_MEMBER_MERGE_TARGET_KIND
    )
    is_renderer_payload_dfl1 = (
        row.get("operation_family") == "native_renderer_payload"
        and row.get("target_kind") == RENDERER_PAYLOAD_DFL1_TARGET_KIND
    )
    source_runtime = _first_text(
        hints,
        (
            "renderer_payload_dfl1_source_runtime_dir",
            "renderer_payload_dfl1_inflate_runtime_dir",
            "packet_member_merge_source_runtime_dir",
            "source_runtime_dir",
            "inflate_runtime_dir",
        ),
    )
    if source_runtime is not None:
        context["source_runtime_dir"] = source_runtime
        if is_member_merge:
            context["packet_member_merge_source_runtime_dir"] = source_runtime
        if is_renderer_payload_dfl1:
            context["renderer_payload_dfl1_source_runtime_dir"] = source_runtime
            context["renderer_payload_dfl1_inflate_runtime_dir"] = source_runtime
            context["inflate_runtime_dir"] = source_runtime
    if is_renderer_payload_dfl1:
        candidate_runtime = _first_text(
            hints,
            (
                "renderer_payload_dfl1_candidate_runtime_dir",
                "candidate_runtime_dir",
            ),
        )
        if candidate_runtime is not None:
            context["renderer_payload_dfl1_candidate_runtime_dir"] = candidate_runtime
            context["candidate_runtime_dir"] = candidate_runtime
        file_list_key, file_list = _first_text_with_key(
            hints,
            (
                "renderer_payload_dfl1_full_frame_file_list",
                "full_frame_file_list",
                "inflate_file_list",
                "file_list",
            ),
        )
        if file_list is not None:
            context["renderer_payload_dfl1_full_frame_file_list"] = file_list
            context["full_frame_file_list"] = file_list
            if file_list_key not in (
                None,
                "renderer_payload_dfl1_full_frame_file_list",
                "full_frame_file_list",
            ):
                context[str(file_list_key)] = file_list
        file_list_entries = ordered_unique(
            [
                *_string_list(
                    hints.get("renderer_payload_dfl1_full_frame_file_list_entries")
                ),
                *_string_list(hints.get("full_frame_file_list_entries")),
                *_string_list(hints.get("file_list_entries")),
                *_string_list(hints.get("file_list_entry")),
            ]
        )
        if file_list_entries:
            context["renderer_payload_dfl1_full_frame_file_list_entries"] = file_list_entries
            context["full_frame_file_list_entries"] = file_list_entries
        parity_output_dir = _first_text(
            hints,
            (
                "renderer_payload_dfl1_inflate_parity_output_dir",
                "full_frame_inflate_parity_output_dir",
                "inflate_parity_output_dir",
            ),
        )
        if parity_output_dir is not None:
            context["renderer_payload_dfl1_inflate_parity_output_dir"] = parity_output_dir
            context["full_frame_inflate_parity_output_dir"] = parity_output_dir
        expected_file_list_sha = _canonical_sha256(
            _first_text(
                hints,
                (
                    "renderer_payload_dfl1_expected_full_frame_file_list_sha256",
                    "expected_full_frame_file_list_sha256",
                ),
            )
        )
        if expected_file_list_sha is not None:
            context["renderer_payload_dfl1_expected_full_frame_file_list_sha256"] = (
                expected_file_list_sha
            )
            context["expected_full_frame_file_list_sha256"] = expected_file_list_sha
        expected_entry_count = _positive_int(
            hints.get("renderer_payload_dfl1_expected_full_frame_entry_count")
        )
        if expected_entry_count is None:
            expected_entry_count = _positive_int(
                hints.get("expected_full_frame_entry_count")
            )
        if expected_entry_count is not None:
            context["renderer_payload_dfl1_expected_full_frame_entry_count"] = (
                expected_entry_count
            )
            context["expected_full_frame_entry_count"] = expected_entry_count
        file_list_source = _first_text(
            hints,
            (
                "renderer_payload_dfl1_full_frame_file_list_source",
                "full_frame_file_list_source",
            ),
        )
        if file_list_source is not None:
            context["renderer_payload_dfl1_full_frame_file_list_source"] = (
                file_list_source
            )
            context["full_frame_file_list_source"] = file_list_source
        if source_runtime is None:
            blockers.append(
                "materializer_context_missing:renderer_payload_dfl1_source_runtime_dir"
            )
        if file_list is None and not file_list_entries:
            blockers.append(
                "materializer_context_missing:renderer_payload_dfl1_full_frame_file_list_or_entries"
            )
        if expected_file_list_sha is None:
            blockers.append(
                "materializer_context_missing:renderer_payload_dfl1_expected_full_frame_file_list_sha256"
            )
        if expected_entry_count is None:
            blockers.append(
                "materializer_context_missing:renderer_payload_dfl1_expected_full_frame_entry_count"
            )
        if file_list_source is None:
            blockers.append(
                "materializer_context_missing:renderer_payload_dfl1_full_frame_file_list_source"
            )
    if is_member_merge and merge_contract is None:
        blockers.append("materializer_context_missing:merge_contract")
    if is_member_merge and source_runtime is None:
        blockers.append("materializer_context_missing:packet_member_merge_source_runtime_dir")
    merged_member_name = _first_text(
        hints,
        ("merged_member_name", "candidate_member_name", "output_member_name"),
    )
    if merged_member_name is not None:
        context["merged_member_name"] = merged_member_name
    payload_member_name = _first_text(
        hints,
        ("payload_member_name", "renderer_payload_member_name"),
    )
    if payload_member_name is not None:
        context["payload_member_name"] = payload_member_name
    if output_archive is not None:
        context["output_archive"] = output_archive
    if json_out is not None:
        context["output_manifest"] = json_out
    for key in ("zip_compression_method", "zip_compression_methods", "zip_compresslevel", "zip_compresslevels"):
        values = _string_list(hints.get(key))
        if values:
            context[key] = values
    _copy_common_materializer_controls(context, hints)
    return _context_row_payload(row, context=context, blockers=blockers)


def _tensor_factorize_context_row(
    row: Mapping[str, Any],
    *,
    hints: Mapping[str, Any],
    repo_root: Path,
    default_output_root: Path | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    blockers: list[str] = []
    archive_path = _first_text(hints, ("archive_path", "source_archive"))
    tensor_manifest = _first_text(hints, ("tensor_manifest",))
    factorization_contract = _first_text(
        hints,
        ("factorization_contract", "tensor_factorization_contract"),
    )
    rank = hints.get("rank")
    output_archive, json_out = _output_paths(
        row=row,
        hints=hints,
        repo_root=repo_root,
        default_output_root=default_output_root,
    )
    if archive_path is None:
        blockers.append("materializer_context_missing:archive_path")
    if tensor_manifest is None:
        blockers.append("materializer_context_missing:tensor_manifest")
    if factorization_contract is None and not (isinstance(rank, int) and not isinstance(rank, bool) and rank > 0):
        blockers.append("materializer_context_missing:factorization_contract_or_rank")
    if output_archive is None:
        blockers.append("materializer_context_missing:output_archive")
    context.update(
        {
            "context_blockers": blockers,
            **_packetir_operation_set_contract_context(),
            **FALSE_AUTHORITY,
        }
    )
    if archive_path is not None:
        context["archive_path"] = archive_path
    if tensor_manifest is not None:
        context["tensor_manifest"] = tensor_manifest
    if factorization_contract is not None:
        context["factorization_contract"] = factorization_contract
    if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
        context["rank"] = rank
    if output_archive is not None:
        context["output_archive"] = output_archive
    if json_out is not None:
        context["output_manifest"] = json_out
    _copy_common_materializer_controls(context, hints)
    return _context_row_payload(row, context=context, blockers=blockers)


def _inverse_scorer_cell_context_row(
    row: Mapping[str, Any],
    *,
    hints: Mapping[str, Any],
    repo_root: Path,
    default_output_root: Path | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    blockers: list[str] = []
    template = _first_text(
        hints,
        (
            "candidate_archive_template",
            "archive_template",
            "source_archive_template",
        ),
    )
    action_functional = _first_text(
        hints,
        (
            "inverse_action_functional",
            "action_functional",
            "inverse_action",
        ),
    )
    raw_digest = _first_text(
        hints,
        (
            "raw_contest_video_digest",
            "contest_video_digest",
            "raw_video_digest",
        ),
    )
    output_dir = _first_text(hints, ("chain_output_dir", "output_dir"))
    output_archive = None
    json_out = None
    if output_dir is None:
        output_archive, json_out = _output_paths(
            row=row,
            hints=hints,
            repo_root=repo_root,
            default_output_root=default_output_root,
        )
    if template is None:
        blockers.append("materializer_context_missing:candidate_archive_template")
    if action_functional is None:
        blockers.append("materializer_context_missing:inverse_action_functional")
    if raw_digest is None:
        blockers.append("materializer_context_missing:raw_contest_video_digest")
    if output_dir is None and output_archive is None:
        blockers.append("materializer_context_missing:output_archive")
    if output_dir is None and json_out is None:
        blockers.append("materializer_context_missing:manifest_out")

    source_inflate = _first_text(hints, ("source_inflate_output_dir",))
    candidate_inflate = _first_text(hints, ("candidate_inflate_output_dir",))
    inflate_runtime = _first_text(hints, ("inflate_runtime_dir",))
    descriptor_probe_only = hints.get("descriptor_probe_only") is True
    has_partial_parity_dirs = (source_inflate is None) != (candidate_inflate is None)
    if output_dir is not None and has_partial_parity_dirs:
        blockers.append("materializer_context_missing:inverse_scorer_cell_complete_inflate_parity_dirs")
    if (
        output_dir is not None
        and not descriptor_probe_only
        and not ((source_inflate is not None and candidate_inflate is not None) or inflate_runtime is not None)
    ):
        blockers.append("materializer_context_missing:inverse_scorer_cell_inflate_parity_context")

    context.update(
        {
            "context_blockers": blockers,
            **_packetir_operation_set_contract_context(),
            **FALSE_AUTHORITY,
        }
    )
    if template is not None:
        context["candidate_archive_template"] = template
    if action_functional is not None:
        context["inverse_action_functional"] = action_functional
    if raw_digest is not None:
        context["raw_contest_video_digest"] = raw_digest
    if output_dir is not None:
        context["output_dir"] = output_dir
        context["chain_output_dir"] = output_dir
    elif output_archive is not None:
        context["output_archive"] = output_archive
    if json_out is not None:
        context["manifest_out"] = json_out
        context["output_manifest"] = json_out
    for key, value in (
        ("source_inflate_output_dir", source_inflate),
        ("candidate_inflate_output_dir", candidate_inflate),
        ("inflate_runtime_dir", inflate_runtime),
        ("source_archive_for_parity", _first_text(hints, ("source_archive_for_parity",))),
        ("inflate_work_dir", _first_text(hints, ("inflate_work_dir",))),
        ("runtime_consumption_proof", _first_text(hints, ("runtime_consumption_proof",))),
    ):
        if value is not None:
            context[key] = value
    for key in ("atom_id", "atom_ids"):
        values = _string_list(hints.get(key))
        if values:
            context[key] = values
    for key in ("selected_limit", "min_free_bytes", "inflate_timeout_seconds"):
        value = hints.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            context[key] = value
    for key in (
        "allow_overwrite",
        "descriptor_probe_only",
        "fail_if_receiver_blocked",
        "fail_if_inflate_parity_blocked",
        "keep_inflate_work_dir",
    ):
        if hints.get(key) is True:
            context[key] = True
    for key in ("expected_output_sha256", "expected_manifest_sha256"):
        value = _text(hints.get(key))
        if value is not None:
            context[key] = value
    return _context_row_payload(row, context=context, blockers=blockers)


def _context_row_payload(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    context_payload = dict(context)
    for key in (
        "source_packet_ir_schemas",
        "source_packet_ir_operation_set_ids",
        "source_packet_ir_source_operation_set_ids",
    ):
        values = _as_list(row.get(key))
        if values:
            context_payload[key] = values
    packet_ir_blocker_counts = _int_mapping(row.get("packet_ir_blocker_counts"))
    if packet_ir_blocker_counts:
        context_payload["packet_ir_blocker_counts"] = packet_ir_blocker_counts
    return {
        "schema": "byte_shaving_materializer_context_row.v1",
        "backlog_key": row.get("backlog_key"),
        "backlog_rank": row.get("backlog_rank"),
        "materializer_id": row.get("materializer_id"),
        "target_kind": row.get("target_kind"),
        "unit_kind": row.get("unit_kind"),
        "operation_family": row.get("operation_family"),
        "source_unit_ids": _as_list(row.get("source_unit_ids")),
        "source_packet_ir_schemas": _as_list(row.get("source_packet_ir_schemas")),
        "source_packet_ir_operation_set_ids": _as_list(row.get("source_packet_ir_operation_set_ids")),
        "source_packet_ir_source_operation_set_ids": _as_list(row.get("source_packet_ir_source_operation_set_ids")),
        "packet_ir_blocker_counts": packet_ir_blocker_counts,
        "context_keys": _context_keys(row),
        "context": context_payload,
        "context_blockers": list(blockers),
        **FALSE_AUTHORITY,
    }


def _unsupported_context_row(row: Mapping[str, Any], *, fallback_key: str) -> dict[str, Any]:
    unit_kind = str(row.get("unit_kind") or "")
    operation_family = str(row.get("operation_family") or "")
    target_kind = str(row.get("target_kind") or "")
    blockers = [
        "final_byte_context_compiler_unsupported_backlog_row",
        (
            "materializer_context_compiler_missing:"
            f"{unit_kind or '<unknown>'}:"
            f"{operation_family or '<unknown>'}:"
            f"{target_kind or '<target_tbd>'}"
        ),
    ]
    return _context_row_payload(
        row,
        context={
            "context_blockers": blockers,
            "packetir_compiler_bridge": _packetir_compiler_bridge_hint(
                row,
                blockers=blockers,
            ),
            **FALSE_AUTHORITY,
        },
        blockers=blockers,
    ) | {
        "backlog_key": row.get("backlog_key") or fallback_key,
        "unsupported": True,
    }


def _packetir_compiler_bridge_hint(
    row: Mapping[str, Any],
    *,
    blockers: Sequence[str],
) -> dict[str, Any]:
    """Return a fail-closed PacketIR/compiler bridge hint for unsupported rows."""

    contract_context = _packetir_operation_set_contract_context()
    contract = contract_context["packetir_operation_set_contract"]
    return {
        "schema": "final_byte_packetir_compiler_bridge_hint.v1",
        **contract_context,
        "canonical_packet_compiler_module": contract["canonical_packet_compiler_module"],
        "canonical_packet_compiler_schema": contract["canonical_packet_compiler_schema"],
        "recommended_ir_schema": contract["recommended_ir_schema"],
        "bridge_role": "candidate_family_operation_compiler",
        "unit_kind": row.get("unit_kind"),
        "operation_family": row.get("operation_family"),
        "target_kind": row.get("target_kind"),
        "materializer_id": row.get("materializer_id"),
        "source_unit_ids": _as_list(row.get("source_unit_ids")),
        "required_order": list(contract["required_order"]),
        "required_proofs": list(contract["required_proofs"]),
        "blockers": ordered_unique(
            [
                *[str(item) for item in blockers],
                "packetir_bridge_requires_operation_set_ir",
                "packetir_bridge_requires_runtime_consumption_proof",
                "packetir_bridge_requires_exact_readiness_handoff",
            ]
        ),
        **FALSE_AUTHORITY,
    }


def build_final_byte_operation_contexts(
    backlog: Mapping[str, Any],
    *,
    artifact_map: Mapping[str, Any] | None,
    repo_root: str | Path,
    default_output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build queue-consumable materializer contexts from backlog + custody hints."""

    if backlog.get("schema") != MATERIALIZER_BACKLOG_SCHEMA:
        raise ExperimentQueueError(f"expected schema {MATERIALIZER_BACKLOG_SCHEMA}")
    _require_no_truthy_authority_recursive(backlog, context="backlog")
    repo = Path(repo_root)
    output_root = Path(default_output_root) if default_output_root is not None else None
    hints = _hint_mapping(artifact_map)
    rows: list[dict[str, Any]] = []
    unsupported_rows: list[str] = []
    for index, row in enumerate(_as_list(backlog.get("rows")), start=1):
        if not isinstance(row, Mapping):
            raise ExperimentQueueError(f"backlog row {index} must be an object")
        if (
            row.get("unit_kind") == "archive_section"
            and row.get("operation_family") == "section_entropy_recode"
            and row.get("target_kind") == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
        ):
            rows.append(
                _archive_section_context_row(
                    row,
                    hints=_merged_hints(hints, row),
                    repo_root=repo,
                    default_output_root=output_root,
                )
            )
        elif (
            row.get("unit_kind") == "packet_member"
            and (
                (
                    row.get("operation_family") == "member_recompress"
                    and row.get("target_kind") == PACKET_MEMBER_RECOMPRESS_TARGET_KIND
                )
                or (
                    row.get("operation_family") == "member_merge"
                    and row.get("target_kind") == PACKET_MEMBER_MERGE_TARGET_KIND
                )
                or (
                    row.get("operation_family") == "zip_header_elide"
                    and row.get("target_kind")
                    == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND
                )
                or (
                    row.get("operation_family") == "native_renderer_payload"
                    and row.get("target_kind") == RENDERER_PAYLOAD_DFL1_TARGET_KIND
                )
            )
        ):
            rows.append(
                _packet_member_context_row(
                    row,
                    hints=_merged_hints(hints, row),
                    repo_root=repo,
                    default_output_root=output_root,
                )
            )
        elif (
            row.get("unit_kind") == "tensor"
            and row.get("operation_family") == "factorize_tensor"
            and row.get("target_kind") == TENSOR_FACTORIZE_TARGET_KIND
        ):
            rows.append(
                _tensor_factorize_context_row(
                    row,
                    hints=_merged_hints(hints, row),
                    repo_root=repo,
                    default_output_root=output_root,
                )
            )
        elif (
            row.get("unit_kind") == "scorer_inverse_surface_cell"
            and row.get("operation_family") == "materialize_inverse_scorer_cell_candidate"
            and row.get("target_kind") == INVERSE_SCORER_CELL_TARGET_KIND
        ):
            rows.append(
                _inverse_scorer_cell_context_row(
                    row,
                    hints=_merged_hints(hints, row),
                    repo_root=repo,
                    default_output_root=output_root,
                )
            )
        else:
            fallback_key = str(index)
            unsupported_rows.append(str(row.get("backlog_key") or fallback_key))
            rows.append(_unsupported_context_row(row, fallback_key=fallback_key))
    blocked_context_count = sum(1 for row in rows if row["context_blockers"])
    return apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_CONTEXTS_SCHEMA,
            "generator_schema": CONTEXT_COMPILER_SCHEMA,
            "tool": "comma_lab.scheduler.final_byte_operation_contexts",
            "generated_at_utc": _utc_now(),
            "source_backlog_schema": backlog.get("schema"),
            "row_count": len(rows),
            "blocked_context_count": blocked_context_count,
            "unsupported_backlog_keys": unsupported_rows,
            "rows": rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            [
                "final_byte_context_compiler_has_blocked_contexts",
                "materializer_contexts_require_operator_or_custody_completion",
            ]
            if blocked_context_count
            else []
        ),
    )


__all__ = [
    "CONTEXT_COMPILER_SCHEMA",
    "build_final_byte_operation_contexts",
]
