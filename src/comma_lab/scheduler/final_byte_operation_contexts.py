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
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
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
    merged: dict[str, Any] = {}
    for key in ordered_unique(keys):
        value = hints.get(key)
        if isinstance(value, Mapping):
            merged.update(dict(value))
    return merged


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
            "label": _first_text(hints, ("label", "candidate_id"))
            or _safe_work_id(str(row.get("backlog_key") or "")),
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
    if hints.get("allow_rate_regression") is True:
        context["allow_rate_regression"] = True
    return {
        "schema": "byte_shaving_materializer_context_row.v1",
        "backlog_key": row.get("backlog_key"),
        "backlog_rank": row.get("backlog_rank"),
        "materializer_id": row.get("materializer_id"),
        "target_kind": row.get("target_kind"),
        "unit_kind": row.get("unit_kind"),
        "operation_family": row.get("operation_family"),
        "source_unit_ids": _as_list(row.get("source_unit_ids")),
        "context_keys": _context_keys(row),
        "context": context,
        "context_blockers": blockers,
        **FALSE_AUTHORITY,
    }


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
    if output_archive is not None:
        context["output_archive"] = output_archive
    if json_out is not None:
        context["output_manifest"] = json_out
    for key in ("zip_compression_method", "zip_compression_methods", "zip_compresslevel", "zip_compresslevels"):
        values = _string_list(hints.get(key))
        if values:
            context[key] = values
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
    if factorization_contract is None and not (
        isinstance(rank, int) and not isinstance(rank, bool) and rank > 0
    ):
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
    return _context_row_payload(row, context=context, blockers=blockers)


def _context_row_payload(
    row: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "byte_shaving_materializer_context_row.v1",
        "backlog_key": row.get("backlog_key"),
        "backlog_rank": row.get("backlog_rank"),
        "materializer_id": row.get("materializer_id"),
        "target_kind": row.get("target_kind"),
        "unit_kind": row.get("unit_kind"),
        "operation_family": row.get("operation_family"),
        "source_unit_ids": _as_list(row.get("source_unit_ids")),
        "context_keys": _context_keys(row),
        "context": dict(context),
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
        "canonical_packet_compiler_module": contract[
            "canonical_packet_compiler_module"
        ],
        "canonical_packet_compiler_schema": contract[
            "canonical_packet_compiler_schema"
        ],
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
            and row.get("operation_family") == "member_recompress"
            and row.get("target_kind") == PACKET_MEMBER_RECOMPRESS_TARGET_KIND
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
