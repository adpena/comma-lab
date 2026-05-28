# SPDX-License-Identifier: MIT
"""Bootstrap queue-owned final-rate attacks from frontier archive evidence.

This module is intentionally a thin compiler around existing scheduler and
family-agnostic materializer surfaces. It does not introduce a new executor or
score authority; it turns canonical frontier/archive evidence into an
``experiment_queue.v1`` that existing local workers can execute.
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.analysis.hnerv_packet_sections import (
    PARSER_AUTO,
    build_packet_section_manifest,
    validate_packet_section_manifest,
)
from tac.optimization.byte_shaving_campaign import FALSE_AUTHORITY
from tac.optimization.family_agnostic_materializers import (
    RENDERER_PAYLOAD_DFL1_MEMBER_NAMES,
)
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.packet_compiler.feca_selector_reparameterize import (
    FEC8_MAGIC,
    FECA_MAGIC,
    split_fp11_member,
)
from tac.packet_compiler.fp11_source_brotli_recode import (
    parse_decoder_blob_len,
)
from tac.repo_io import sha256_bytes, sha256_file, tree_sha256, write_json_artifact

from .byte_shaving_campaign_queue import (
    MATERIALIZER_BACKLOG_SCHEMA,
    MATERIALIZER_CONTEXTS_SCHEMA,
    build_materializer_execution_queue,
    build_materializer_work_queue,
    materializer_contexts_from_payload,
)
from .byte_shaving_materializer_registry import (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    ARCHIVE_ZIP_REPACK_TARGET_KIND,
    DQS1_PAIRSET_TARGET_KIND,
    FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND,
    FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
    INVERSE_SCORER_CELL_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
    registry_manifest,
)
from .experiment_queue import ExperimentQueueError, normalize_queue_definition
from .frontier_rate_attack_target_profile import (
    TARGET_OPTIMIZATION_PROFILE_SCHEMA,
    target_optimization_profile_queue_metadata,
)

BOOTSTRAP_SCHEMA = "frontier_final_rate_attack_bootstrap.v1"
TARGET_COVERAGE_SCHEMA = "frontier_final_rate_attack_target_coverage.v1"
DERIVED_PACKET_MEMBER_MERGE_CONTRACT_SCHEMA = (
    "frontier_rate_attack_derived_packet_member_merge_contract.v1"
)
DERIVED_SECTION_MANIFEST_SCHEMA = "frontier_rate_attack_derived_section_manifest.v1"
DERIVED_SECTION_MANIFEST_BATCH_SCHEMA = "frontier_rate_attack_derived_section_manifest_batch.v1"
FRONTIER_ARCHIVE_RESOLUTION_SCHEMA = "frontier_archive_resolution.v1"
FRONTIER_ARCHIVE_RECORD_SCHEMA = "frontier_rate_attack_archive_record.v1"
DEFAULT_FRONTIER_POINTER = ".omx/state/canonical_frontier_pointer.json"
DEFAULT_EXECUTABLE_TARGET_KINDS = (
    ARCHIVE_ZIP_REPACK_TARGET_KIND,
    PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND,
    PACKET_MEMBER_RECOMPRESS_TARGET_KIND,
    PACKET_MEMBER_MERGE_TARGET_KIND,
    RENDERER_PAYLOAD_DFL1_TARGET_KIND,
    FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND,
    FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
)
DEFAULT_OPTIONAL_TARGET_KINDS = (
    ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND,
    TENSOR_FACTORIZE_TARGET_KIND,
)
ARCHIVE_RATE_ATTACK_SUPPORTED_TARGET_KINDS = (
    *DEFAULT_EXECUTABLE_TARGET_KINDS,
    *DEFAULT_OPTIONAL_TARGET_KINDS,
)
FRONTIER_RATE_ATTACK_DEFERRED_TARGET_RATIONALES = {
    DQS1_PAIRSET_TARGET_KIND: (
        "DQS1 pairset drops require pair-index acquisition and scorer feedback; "
        "they are queued through the DQS1 local-first feedback cycle rather than "
        "the archive-only final-rate bootstrap."
    ),
    INVERSE_SCORER_CELL_TARGET_KIND: (
        "Inverse-scorer cell candidates require raw-video, action-functional, "
        "candidate-template, and inflate-parity context; they are queued through "
        "the inverse-steganalysis acquisition chain rather than the archive-only "
        "final-rate bootstrap."
    ),
}
_AXIS_POINTER_KEYS = {
    "contest_cpu": "our_local_frontier_contest_cpu",
    "contest-cpu": "our_local_frontier_contest_cpu",
    "cpu": "our_local_frontier_contest_cpu",
    "contest_cuda": "our_local_frontier_contest_cuda",
    "contest-cuda": "our_local_frontier_contest_cuda",
    "cuda": "our_local_frontier_contest_cuda",
}


class FrontierRateAttackBootstrapError(ExperimentQueueError):
    """Raised when frontier-rate bootstrap input would be ambiguous or unsafe."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _clean_id(value: str, *, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or fallback


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackBootstrapError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackBootstrapError(f"{path}: expected JSON object")
    return payload


def _zip_member_records(path: Path) -> list[dict[str, Any]]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            records = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                records.append(
                    {
                        "name": info.filename,
                        "compress_type": info.compress_type,
                        "compress_size": info.compress_size,
                        "file_size": info.file_size,
                        "crc": f"{info.CRC:08x}",
                        "extra_bytes": len(info.extra or b""),
                        "comment_bytes": len(info.comment or b""),
                    }
                )
    except zipfile.BadZipFile as exc:
        raise FrontierRateAttackBootstrapError(f"{path}: not a readable ZIP archive") from exc
    if not records:
        raise FrontierRateAttackBootstrapError(f"{path}: ZIP archive has no file members")
    return records


def _single_member_payload(path: Path, *, member_name: str) -> bytes:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.read(member_name)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise FrontierRateAttackBootstrapError(
            f"{path}: could not read ZIP member {member_name!r}"
        ) from exc


def _section_brotli_probe_rows(
    *,
    manifest: Mapping[str, Any],
    archive_path: Path,
) -> list[dict[str, Any]]:
    parser_input = manifest.get("parser_input")
    if isinstance(parser_input, Mapping) and parser_input.get("kind") != "member_payload":
        return [
            {
                "section_name": None,
                "brotli_decompressible": False,
                "blockers": ["section_manifest_parser_input_not_member_payload"],
                **FALSE_AUTHORITY,
            }
        ]
    member = manifest.get("member")
    if not isinstance(member, Mapping):
        return [
            {
                "section_name": None,
                "brotli_decompressible": False,
                "blockers": ["section_manifest_member_missing"],
                **FALSE_AUTHORITY,
            }
        ]
    member_name = str(member.get("name") or "")
    if not member_name:
        return [
            {
                "section_name": None,
                "brotli_decompressible": False,
                "blockers": ["section_manifest_member_name_missing"],
                **FALSE_AUTHORITY,
            }
        ]
    member_payload = _single_member_payload(archive_path, member_name=member_name)
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        return [
            {
                "section_name": None,
                "brotli_decompressible": False,
                "blockers": ["section_manifest_sections_missing"],
                **FALSE_AUTHORITY,
            }
        ]
    rows: list[dict[str, Any]] = []
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            continue
        name = str(section.get("name") or f"section_{index:04d}")
        blockers: list[str] = []
        try:
            offset = int(section["offset"])
            length = int(section["length"])
        except (KeyError, TypeError, ValueError):
            blockers.append("section_offset_or_length_invalid")
            offset = 0
            length = 0
        expected_sha = str(section.get("sha256") or "")
        payload = member_payload[offset : offset + length] if length > 0 else b""
        if length < 1 or offset < 0 or offset + length > len(member_payload):
            blockers.append("section_range_outside_member_payload")
        elif expected_sha and sha256_bytes(payload) != expected_sha:
            blockers.append("section_sha256_mismatch")
        raw_sha: str | None = None
        raw_bytes: int | None = None
        brotli_decompressible = False
        if not blockers:
            try:
                raw = brotli.decompress(payload)
            except brotli.error:
                blockers.append("section_not_brotli_decompressible")
            else:
                brotli_decompressible = True
                raw_sha = sha256_bytes(raw)
                raw_bytes = len(raw)
        rows.append(
            {
                "section_name": name,
                "section_index": int(section.get("index", index)),
                "offset": offset,
                "length": length,
                "sha256": expected_sha or None,
                "optimization_role": section.get("optimization_role"),
                "brotli_decompressible": brotli_decompressible,
                "raw_sha256": raw_sha,
                "raw_bytes": raw_bytes,
                "blockers": ordered_unique(blockers),
                **FALSE_AUTHORITY,
            }
        )
    return rows


def derive_archive_section_recode_manifest(
    *,
    archive_record: Mapping[str, Any],
    output_path: str | Path,
    repo_root: str | Path,
    parser: str = PARSER_AUTO,
    allow_overwrite: bool = False,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Derive a parser-section manifest and selected Brotli sections for one archive."""

    require_no_truthy_authority_fields(
        archive_record,
        context="derive_archive_section_recode_manifest.archive_record",
    )
    repo = Path(repo_root)
    label = _clean_id(str(archive_record.get("label") or "archive"), fallback="archive")
    archive_value = archive_record.get("absolute_path") or archive_record.get("path")
    if not isinstance(archive_value, str) or not archive_value.strip():
        raise FrontierRateAttackBootstrapError(f"{label}: archive record path missing")
    archive_path = _resolve_path(archive_value, repo_root=repo)
    target = _resolve_path(output_path, repo_root=repo)
    blockers: list[str] = []
    manifest: dict[str, Any] | None = None
    brotli_rows: list[dict[str, Any]] = []
    selected_names: list[str] = []
    try:
        manifest = build_packet_section_manifest(
            archive_path,
            label=label,
            parser=parser,
            repo_root=repo,
        )
        blockers.extend(validate_packet_section_manifest(manifest, repo_root=repo))
        brotli_rows = _section_brotli_probe_rows(
            manifest=manifest,
            archive_path=archive_path,
        )
        selected_names = [
            str(row["section_name"])
            for row in brotli_rows
            if row.get("brotli_decompressible") is True and row.get("section_name")
        ]
        if not selected_names:
            blockers.append("section_manifest_has_no_brotli_decompressible_sections")
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        blockers.append(f"section_manifest_derivation_failed:{exc}")
    if manifest is not None:
        manifest["frontier_rate_attack_section_recode"] = {
            "schema": DERIVED_SECTION_MANIFEST_SCHEMA,
            "archive_label": label,
            "archive_sha256": archive_record.get("sha256"),
            "selected_section_names": selected_names,
            "brotli_probe_rows": brotli_rows,
            "blockers": ordered_unique(blockers),
            **FALSE_AUTHORITY,
        }
        write = write_json_artifact(
            target,
            manifest,
            allow_overwrite=allow_overwrite,
            min_free_bytes=min_free_bytes,
        )
        manifest_path: str | None = _repo_rel(target, repo)
        manifest_sha: str | None = write.sha256
    else:
        manifest_path = None
        manifest_sha = None
    return apply_proxy_evidence_boundary(
        {
            "schema": DERIVED_SECTION_MANIFEST_SCHEMA,
            "archive_label": label,
            "archive_sha256": archive_record.get("sha256"),
            "archive_path": archive_record.get("path"),
            "section_manifest_path": manifest_path,
            "section_manifest_sha256": manifest_sha,
            "selected_section_names": selected_names,
            "selected_section_count": len(selected_names),
            "brotli_probe_rows": brotli_rows,
            "blockers": ordered_unique(blockers),
            "ready_for_materializer_target": not blockers,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            ordered_unique(blockers)
            if blockers
            else ("frontier_rate_attack_section_recode_is_local_only",)
        ),
    )


def derive_archive_section_recode_manifests(
    *,
    archive_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    repo_root: str | Path,
    parser: str = PARSER_AUTO,
    allow_overwrite: bool = False,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Derive per-archive section manifests for frontier-rate materializer queues."""

    repo = Path(repo_root)
    root = _resolve_path(output_dir, repo_root=repo)
    rows = [
        derive_archive_section_recode_manifest(
            archive_record=record,
            output_path=root / f"{_clean_id(str(record.get('label') or index), fallback='archive')}.section_manifest.json",
            repo_root=repo,
            parser=parser,
            allow_overwrite=allow_overwrite,
            min_free_bytes=min_free_bytes,
        )
        for index, record in enumerate(archive_records)
    ]
    return apply_proxy_evidence_boundary(
        {
            "schema": DERIVED_SECTION_MANIFEST_BATCH_SCHEMA,
            "generated_at_utc": _utc_now(),
            "manifest_count": len(rows),
            "ready_manifest_count": sum(
                1 for row in rows if row.get("ready_for_materializer_target") is True
            ),
            "rows": rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=("derived_section_manifests_are_local_materializer_inputs_only",),
    )


def _record_zip_member_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = record.get("zip_members")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _record_zip_member_names(record: Mapping[str, Any]) -> list[str]:
    return [
        str(row["name"])
        for row in _record_zip_member_rows(record)
        if isinstance(row.get("name"), str) and str(row.get("name"))
    ]


def _packet_member_merge_record_blockers(
    record: Mapping[str, Any],
    *,
    merged_member_name: str | None,
) -> list[str]:
    names = _record_zip_member_names(record)
    blockers: list[str] = []
    if len(names) < 2:
        blockers.append("packet_member_merge_requires_at_least_two_members")
    output_member = (merged_member_name or "__packet_member_merge_v1.bin").strip()
    if not output_member:
        blockers.append("packet_member_merge_merged_member_name_empty")
    elif output_member in set(names):
        blockers.append("packet_member_merge_merged_member_name_collides_with_source_member")
    return blockers


def _renderer_payload_dfl1_record_blockers(
    record: Mapping[str, Any],
    *,
    payload_member_name: str,
) -> list[str]:
    rows = _record_zip_member_rows(record)
    by_name = {str(row.get("name")): row for row in rows if isinstance(row.get("name"), str)}
    names = set(by_name)
    blockers: list[str] = []
    missing = [
        name for name in RENDERER_PAYLOAD_DFL1_MEMBER_NAMES if name not in names
    ]
    if missing:
        blockers.append("renderer_payload_dfl1_missing_members:" + ",".join(missing))
    unsupported = [
        name
        for name in RENDERER_PAYLOAD_DFL1_MEMBER_NAMES
        if name in by_name
        and int(by_name[name].get("compress_type", -1)) != zipfile.ZIP_DEFLATED
    ]
    if unsupported:
        blockers.append(
            "renderer_payload_dfl1_requires_deflated_members:" + ",".join(unsupported)
        )
    output_member = str(payload_member_name or "").strip() or "p"
    if not output_member or "/" in output_member or output_member.startswith("."):
        blockers.append("renderer_payload_dfl1_payload_member_name_unsafe")
    elif output_member in names:
        blockers.append("renderer_payload_dfl1_payload_member_name_collides_with_source_member")
    return blockers


def _record_absolute_archive_path(record: Mapping[str, Any]) -> Path | None:
    raw = record.get("absolute_path")
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    raw = record.get("path")
    if isinstance(raw, str) and raw.strip() and Path(raw).is_absolute():
        return Path(raw)
    return None


def _feca_selector_record_context(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    archive_path = _record_absolute_archive_path(record)
    blockers: list[str] = []
    if archive_path is None:
        return None, ["selector_context_recode_requires_absolute_archive_path"]
    if archive_path.name != "archive.zip":
        blockers.append("selector_context_recode_requires_submission_archive_zip")
    source_submission_dir = archive_path.parent
    if not _safe_exists(source_submission_dir):
        blockers.append("selector_context_recode_source_submission_dir_missing")
    required_paths = (
        source_submission_dir / "inflate.py",
        source_submission_dir / "inflate.sh",
        source_submission_dir
        / "encoder"
        / "build_pr101_frame_exploit_selector_packet_fec10_hybrid.py",
    )
    missing = [path.name for path in required_paths if not _safe_is_file(path)]
    if missing:
        blockers.append("selector_context_recode_missing_runtime_files:" + ",".join(missing))
    markov_module = (
        source_submission_dir
        / "encoder"
        / "build_pr101_frame_exploit_selector_packet_markov.py"
    )
    codec_families = ["fec10_adaptive_blend"]
    if _safe_is_file(markov_module):
        codec_families.extend(
            [
                "fec8_markov_static_order1",
                "fec8_markov_adaptive_order1",
                "fec8_markov_static_order2",
            ]
        )
    if len(_record_zip_member_names(record)) != 1:
        blockers.append("selector_context_recode_requires_single_fp11_member")
    else:
        try:
            member_name = _record_zip_member_names(record)[0]
            payload = _single_member_payload(archive_path, member_name=member_name)
        except (OSError, FrontierRateAttackBootstrapError):
            blockers.append("selector_context_recode_member_payload_unreadable")
        else:
            if not payload.startswith(b"FP11"):
                blockers.append("selector_context_recode_requires_fp11_archive_member")
            elif b"FECa" not in payload:
                blockers.append("selector_context_recode_requires_feca_selector_payload")
    if blockers:
        return None, ordered_unique(blockers)
    return (
        {
            "source_submission_dir": source_submission_dir.as_posix(),
            "selector_codec_families": codec_families,
            "upstream_entropy_positions": ["P19", "P18"],
            "downstream_materializer_targets": [ARCHIVE_ZIP_REPACK_TARGET_KIND],
            "chain_label": "frontier_final_rate_attack_p19_p18_to_p11_selector_then_p15_repack",
        },
        [],
    )


def _fp11_source_brotli_record_context(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    archive_path = _record_absolute_archive_path(record)
    blockers: list[str] = []
    if archive_path is None:
        return None, ["fp11_source_brotli_recode_requires_absolute_archive_path"]
    if archive_path.name != "archive.zip":
        blockers.append("fp11_source_brotli_recode_requires_submission_archive_zip")
    source_submission_dir = archive_path.parent
    if not _safe_exists(source_submission_dir):
        blockers.append("fp11_source_brotli_recode_source_submission_dir_missing")
    required_paths = (
        source_submission_dir / "inflate.py",
        source_submission_dir / "inflate.sh",
        source_submission_dir / "src" / "codec.py",
    )
    missing = [
        path.relative_to(source_submission_dir).as_posix()
        for path in required_paths
        if not _safe_is_file(path)
    ]
    if missing:
        blockers.append("fp11_source_brotli_recode_missing_runtime_files:" + ",".join(missing))
    if len(_record_zip_member_names(record)) != 1:
        blockers.append("fp11_source_brotli_recode_requires_single_fp11_member")
    else:
        try:
            member_name = _record_zip_member_names(record)[0]
            payload = _single_member_payload(archive_path, member_name=member_name)
        except (OSError, FrontierRateAttackBootstrapError):
            blockers.append("fp11_source_brotli_recode_member_payload_unreadable")
        else:
            if not payload.startswith(b"FP11"):
                blockers.append("fp11_source_brotli_recode_requires_fp11_archive_member")
            else:
                try:
                    parts = split_fp11_member(
                        payload,
                        allowed_selector_magics=(FECA_MAGIC, FEC8_MAGIC),
                    )
                except ValueError as exc:
                    blockers.append(f"fp11_source_brotli_recode_fp11_parse_failed:{exc}")
                else:
                    if _safe_is_file(source_submission_dir / "src" / "codec.py"):
                        try:
                            decoder_len = parse_decoder_blob_len(
                                (source_submission_dir / "src" / "codec.py").read_text(
                                    encoding="utf-8"
                                )
                            )
                        except (OSError, ValueError) as exc:
                            blockers.append(
                                f"fp11_source_brotli_recode_decoder_len_parse_failed:{exc}"
                            )
                        else:
                            if decoder_len >= len(parts["source_payload"]):
                                blockers.append(
                                    "fp11_source_brotli_recode_decoder_len_exhausts_source_payload"
                                )
    if blockers:
        return None, ordered_unique(blockers)
    return (
        {
            "source_submission_dir": source_submission_dir.as_posix(),
            "brotli_qualities": list(range(1, 12)),
            "brotli_lgwins": ["none", *list(range(16, 25))],
        },
        [],
    )


def derive_packet_member_merge_contract(
    *,
    archive_records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    repo_root: str | Path,
    merged_member_name: str | None = None,
    zip_compression_methods: Sequence[str] = ("stored", "deflated"),
    zip_compresslevels: Sequence[int] = (1, 6, 9),
    allow_overwrite: bool = False,
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    """Write a shared all-member merge contract for archive-local sweeps."""

    repo = Path(repo_root)
    target = _resolve_path(output_path, repo_root=repo)
    ready_labels: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(archive_records):
        require_no_truthy_authority_fields(
            record,
            context=f"derive_packet_member_merge_contract.archive_records[{index}]",
        )
        label = str(record.get("label") or f"archive_{index}")
        blockers = _packet_member_merge_record_blockers(
            record,
            merged_member_name=merged_member_name,
        )
        if not blockers:
            ready_labels.append(label)
        rows.append(
            apply_proxy_evidence_boundary(
                {
                    "schema": "frontier_rate_attack_packet_member_merge_contract_archive_row.v1",
                    "archive_label": label,
                    "archive_sha256": record.get("sha256"),
                    "zip_member_count": len(_record_zip_member_names(record)),
                    "selected_member_names": _record_zip_member_names(record),
                    "ready_for_materializer_target": not blockers,
                    "blockers": ordered_unique(blockers),
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=(
                    ordered_unique(blockers)
                    if blockers
                    else ("packet_member_merge_contract_row_is_local_only",)
                ),
            )
        )
    contract: dict[str, Any] = {
        "schema": DERIVED_PACKET_MEMBER_MERGE_CONTRACT_SCHEMA,
        "generated_at_utc": _utc_now(),
        "contract_kind": "packet_member_merge_all_members_local_sweep",
        "member_selection": "all_members",
        "all_members": True,
        "receiver_contract_kind": "family_agnostic_packet_member_merge",
        "cooperative_receiver_required": True,
        "zip_compression_methods": list(ordered_unique(zip_compression_methods)),
        "zip_compresslevels": [int(level) for level in zip_compresslevels],
        "archive_rows": rows,
        "ready_archive_labels": ready_labels,
        "ready_archive_count": len(ready_labels),
        "blockers": (
            []
            if ready_labels
            else ["packet_member_merge_contract_has_no_ready_archives"]
        ),
        **FALSE_AUTHORITY,
    }
    if merged_member_name is not None:
        contract["merged_member_name"] = merged_member_name
    write = write_json_artifact(
        target,
        contract,
        allow_overwrite=allow_overwrite,
        min_free_bytes=min_free_bytes,
    )
    return apply_proxy_evidence_boundary(
        {
            "schema": DERIVED_PACKET_MEMBER_MERGE_CONTRACT_SCHEMA,
            "generated_at_utc": contract["generated_at_utc"],
            "merge_contract_path": _repo_rel(target, repo),
            "merge_contract_sha256": write.sha256,
            "ready_archive_labels": ready_labels,
            "ready_archive_count": len(ready_labels),
            "archive_count": len(archive_records),
            "rows": rows,
            "blockers": contract["blockers"],
            "ready_for_materializer_target": bool(ready_labels),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            ("derived_packet_member_merge_contract_is_local_materializer_input_only",)
            if ready_labels
            else tuple(contract["blockers"])
        ),
    )


def archive_record(
    *,
    label: str,
    archive_path: str | Path,
    repo_root: str | Path,
    source_kind: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Return a checked archive record with ZIP member metadata."""

    repo = Path(repo_root)
    archive = _resolve_path(archive_path, repo_root=repo)
    if not archive.is_file():
        raise FrontierRateAttackBootstrapError(f"archive not found: {archive}")
    size = archive.stat().st_size
    if expected_bytes is not None and size != int(expected_bytes):
        raise FrontierRateAttackBootstrapError(
            f"{archive}: byte size mismatch expected={expected_bytes} actual={size}"
        )
    digest = sha256_file(archive)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FrontierRateAttackBootstrapError(
            f"{archive}: sha256 mismatch expected={expected_sha256} actual={digest}"
        )
    members = _zip_member_records(archive)
    return {
        "schema": FRONTIER_ARCHIVE_RECORD_SCHEMA,
        "label": _clean_id(label, fallback="archive"),
        "path": _repo_rel(archive, repo),
        "absolute_path": archive.as_posix(),
        "source_kind": source_kind,
        "bytes": size,
        "sha256": digest,
        "zip_member_count": len(members),
        "zip_members": members,
        **FALSE_AUTHORITY,
    }


def parse_archive_spec(spec: str, *, repo_root: str | Path) -> dict[str, Any]:
    """Parse ``label=path`` archive specs into checked archive records."""

    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise FrontierRateAttackBootstrapError(
                "archive specs must be label=path when '=' is present"
            )
    else:
        raw_path = spec.strip()
        if not raw_path:
            raise FrontierRateAttackBootstrapError("archive spec must not be empty")
        label = Path(raw_path).stem
    return archive_record(
        label=label,
        archive_path=raw_path,
        repo_root=repo_root,
        source_kind="explicit_archive_spec",
    )


def _frontier_pointer_entry(pointer: Mapping[str, Any], axis: str) -> Mapping[str, Any]:
    key = _AXIS_POINTER_KEYS.get(axis.strip().lower())
    if key is None:
        raise FrontierRateAttackBootstrapError(
            f"unsupported frontier axis {axis!r}; expected one of {sorted(_AXIS_POINTER_KEYS)}"
        )
    entry = pointer.get(key)
    if not isinstance(entry, Mapping):
        raise FrontierRateAttackBootstrapError(f"frontier pointer missing {key}")
    return entry


def _frontier_expected_bytes(entry: Mapping[str, Any]) -> int | None:
    extra = entry.get("extra")
    if isinstance(extra, Mapping):
        value = extra.get("archive_bytes")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _candidate_archive_path_from_request(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Path | None:
    for key in ("archive_path", "canonical_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidate = _resolve_path(value, repo_root=repo_root)
            if candidate.name == "archive.zip":
                return candidate
            if candidate.is_file():
                return candidate
    submission_dir = payload.get("submission_dir")
    if isinstance(submission_dir, str) and submission_dir.strip():
        candidate = _resolve_path(submission_dir, repo_root=repo_root) / "archive.zip"
        return candidate
    return None


def _request_file_matches_frontier(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str,
    expected_bytes: int | None,
) -> bool:
    sha_values = [
        payload.get("archive_sha256"),
        payload.get("expected_archive_sha256"),
        payload.get("submission_dir_zip_sha256"),
    ]
    if expected_sha256 not in {value for value in sha_values if isinstance(value, str)}:
        return False
    byte_values = [
        payload.get("archive_bytes"),
        payload.get("archive_size_bytes"),
        payload.get("bytes"),
    ]
    if expected_bytes is None:
        return True
    return any(value == expected_bytes for value in byte_values if isinstance(value, int))


def _request_roots(repo_root: Path, roots: Sequence[str | Path]) -> list[Path]:
    if roots:
        return [_resolve_path(root, repo_root=repo_root) for root in roots]
    return [
        repo_root / "experiments" / "results" / "modal_auth_eval_cpu",
        repo_root / "experiments" / "results" / "modal_auth_eval",
        repo_root / "experiments" / "results" / "modal_auth_eval_cuda",
    ]


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_rglob(root: Path, pattern: str) -> list[Path]:
    try:
        return list(root.rglob(pattern))
    except OSError:
        return []


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _safe_sha256_file(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def _safe_tree_sha256(path: Path) -> str | None:
    try:
        return tree_sha256(path)
    except OSError:
        return None


def _auth_eval_canonical_score(payload: Mapping[str, Any]) -> float | None:
    for key in ("canonical_score", "score_recomputed_from_components", "final_score"):
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _score_close(lhs: float | None, rhs: float | None) -> bool:
    if lhs is None or rhs is None:
        return False
    return abs(float(lhs) - float(rhs)) <= 1e-12


def _request_auth_eval_facts(request_file: Path) -> dict[str, Any]:
    eval_path = request_file.parent / "contest_auth_eval.json"
    facts: dict[str, Any] = {
        "request_mtime_ns": request_file.stat().st_mtime_ns,
    }
    try:
        payload = _load_json(eval_path)
    except (FrontierRateAttackBootstrapError, OSError):
        return facts
    score = _auth_eval_canonical_score(payload)
    if score is not None:
        facts["auth_eval_canonical_score"] = score
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        for key in ("archive_sha256", "archive_size_bytes"):
            value = provenance.get(key)
            if value is not None:
                facts[f"auth_eval_provenance_{key}"] = value
    facts["auth_eval_path"] = eval_path.as_posix()
    return facts


def _match_runtime_fingerprint(match: Mapping[str, Any]) -> str | None:
    for key in (
        "expected_runtime_content_tree_sha256",
        "expected_runtime_tree_sha256",
        "runtime_content_tree_sha256",
        "runtime_tree_sha256",
        "submission_dir_zip_sha256",
    ):
        value = match.get(key)
        if isinstance(value, str) and len(value) == 64:
            return f"{key}:{value}"
    archive_path = match.get("absolute_path")
    if isinstance(archive_path, str):
        tree = _safe_tree_sha256(Path(archive_path).parent)
        if tree is not None:
            return f"computed_submission_dir_tree_sha256:{tree}"
    return None


def _disambiguate_frontier_matches(
    matches: Sequence[Mapping[str, Any]],
    *,
    expected_score: Any,
) -> dict[str, Any]:
    unique = {str(item["absolute_path"]): dict(item) for item in matches}
    if not unique:
        raise FrontierRateAttackBootstrapError("no frontier archive matches to disambiguate")
    if len(unique) == 1:
        return next(iter(unique.values()))

    score = (
        float(expected_score)
        if isinstance(expected_score, int | float) and not isinstance(expected_score, bool)
        else None
    )
    score_matches = [
        item
        for item in unique.values()
        if _score_close(item.get("auth_eval_canonical_score"), score)
    ]
    if len(score_matches) == 1:
        chosen = dict(score_matches[0])
        chosen["disambiguation"] = {
            "strategy": "auth_eval_canonical_score_matches_frontier_pointer",
            "candidate_count_before": len(unique),
            "expected_score": score,
        }
        return chosen
    if len(score_matches) > 1:
        unique = {str(item["absolute_path"]): dict(item) for item in score_matches}

    fingerprints = {
        fingerprint
        for item in unique.values()
        if (fingerprint := _match_runtime_fingerprint(item)) is not None
    }
    if len(fingerprints) == 1 and len(fingerprints) == len({
        _match_runtime_fingerprint(item) for item in unique.values()
    }):
        candidates = sorted(
            unique.values(),
            key=lambda item: (
                int(item.get("request_mtime_ns") or 0),
                str(item.get("absolute_path") or ""),
            ),
            reverse=True,
        )
        chosen = dict(candidates[0])
        chosen["disambiguation"] = {
            "strategy": "identical_runtime_fingerprint_latest_request",
            "candidate_count_before": len(matches),
            "runtime_fingerprint": next(iter(fingerprints)),
        }
        return chosen

    details = []
    for item in sorted(unique.values(), key=lambda row: str(row.get("absolute_path") or "")):
        details.append(
            {
                "path": item.get("absolute_path"),
                "source": item.get("source"),
                "request_path": item.get("request_path"),
                "auth_eval_canonical_score": item.get("auth_eval_canonical_score"),
                "runtime_fingerprint": _match_runtime_fingerprint(item),
            }
        )
    raise FrontierRateAttackBootstrapError(
        "frontier archive resolution is ambiguous after score/runtime "
        "disambiguation: "
        + json.dumps(details, sort_keys=True)
    )


def _default_frontier_archive_candidates(repo_root: Path) -> list[Path]:
    """Return bounded canonical local archive locations for frontier replay.

    The repository can contain thousands of generated ``archive.zip`` files.
    The default resolver intentionally checks only source submission packets
    instead of broad recursive archive scans, keeping automatic current-frontier
    resolution deterministic and cheap while still covering promoted lanes.
    """

    patterns = (
        "experiments/results/*/submission_dir/archive.zip",
        "experiments/results/*/submission/archive.zip",
        "submissions/*/archive.zip",
    )
    candidates: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for candidate in sorted(repo_root.glob(pattern)):
            marker = candidate.resolve(strict=False).as_posix()
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(candidate)
    return candidates


def resolve_current_frontier_archive(
    *,
    repo_root: str | Path,
    frontier_axis: str = "contest_cpu",
    pointer_path: str | Path = DEFAULT_FRONTIER_POINTER,
    request_search_roots: Sequence[str | Path] = (),
    archive_search_roots: Sequence[str | Path] = (),
    max_archive_candidates: int = 512,
) -> dict[str, Any]:
    """Resolve the canonical frontier pointer to exactly one archive path."""

    repo = Path(repo_root)
    pointer_file = _resolve_path(pointer_path, repo_root=repo)
    pointer = _load_json(pointer_file)
    entry = _frontier_pointer_entry(pointer, frontier_axis)
    expected_sha256 = entry.get("archive_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise FrontierRateAttackBootstrapError(
            f"{pointer_file}: frontier entry for {frontier_axis} has no archive_sha256"
        )
    expected_bytes = _frontier_expected_bytes(entry)
    matches: list[dict[str, Any]] = []
    inspected_request_files = 0
    for root in _request_roots(repo, request_search_roots):
        if not _safe_exists(root):
            continue
        for request_file in _safe_rglob(root, "*.json"):
            if "request" not in request_file.name:
                continue
            inspected_request_files += 1
            try:
                payload = _load_json(request_file)
            except FrontierRateAttackBootstrapError:
                continue
            if not _request_file_matches_frontier(
                payload,
                expected_sha256=expected_sha256,
                expected_bytes=expected_bytes,
            ):
                continue
            archive_path = _candidate_archive_path_from_request(payload, repo_root=repo)
            if archive_path is None or not _safe_is_file(archive_path):
                continue
            if expected_bytes is not None and _safe_file_size(archive_path) != expected_bytes:
                continue
            if _safe_sha256_file(archive_path) != expected_sha256:
                continue
            request_facts = _request_auth_eval_facts(request_file)
            for key in (
                "expected_runtime_tree_sha256",
                "expected_runtime_content_tree_sha256",
                "submission_dir_zip_sha256",
            ):
                value = payload.get(key)
                if isinstance(value, str):
                    request_facts[key] = value
            matches.append(
                {
                    "path": _repo_rel(archive_path, repo),
                    "absolute_path": archive_path.as_posix(),
                    "source": "auth_eval_request",
                    "request_path": _repo_rel(request_file, repo),
                    **request_facts,
                }
            )

    if not matches:
        for archive_path in _default_frontier_archive_candidates(repo):
            if not _safe_is_file(archive_path):
                continue
            if expected_bytes is not None and _safe_file_size(archive_path) != expected_bytes:
                continue
            if _safe_sha256_file(archive_path) != expected_sha256:
                continue
            matches.append(
                {
                    "path": _repo_rel(archive_path, repo),
                    "absolute_path": archive_path.as_posix(),
                    "source": "default_submission_archive_search",
                    "request_path": None,
                }
            )

    if not matches:
        roots = [_resolve_path(root, repo_root=repo) for root in archive_search_roots]
        inspected_archives = 0
        for root in roots:
            if not _safe_exists(root):
                continue
            for archive_path in _safe_rglob(root, "archive.zip"):
                inspected_archives += 1
                if inspected_archives > max_archive_candidates:
                    raise FrontierRateAttackBootstrapError(
                        "frontier archive fallback search exceeded "
                        f"max_archive_candidates={max_archive_candidates}"
                    )
                if expected_bytes is not None and _safe_file_size(archive_path) != expected_bytes:
                    continue
                if _safe_sha256_file(archive_path) != expected_sha256:
                    continue
                matches.append(
                    {
                        "path": _repo_rel(archive_path, repo),
                        "absolute_path": archive_path.as_posix(),
                        "source": "bounded_archive_search",
                        "request_path": None,
                    }
                )

    if not matches:
        raise FrontierRateAttackBootstrapError(
            f"could not resolve current {frontier_axis} frontier archive "
            f"sha256={expected_sha256} bytes={expected_bytes}"
        )
    match = _disambiguate_frontier_matches(matches, expected_score=entry.get("score"))
    record = archive_record(
        label=f"current_{frontier_axis}_frontier",
        archive_path=match["absolute_path"],
        repo_root=repo,
        source_kind="canonical_frontier_pointer",
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    return {
        "schema": FRONTIER_ARCHIVE_RESOLUTION_SCHEMA,
        "frontier_axis": frontier_axis,
        "pointer_path": _repo_rel(pointer_file, repo),
        "archive_sha256": expected_sha256,
        "archive_bytes": expected_bytes,
        "score": entry.get("score"),
        "evidence_grade": entry.get("evidence_grade"),
        "hardware_substrate": entry.get("hardware_substrate"),
        "measured_at_utc": entry.get("measured_at_utc"),
        "inspected_request_files": inspected_request_files,
        "match": match,
        "archive_record": record,
        **FALSE_AUTHORITY,
    }


def _adapter_by_target_kind(target_kind: str) -> dict[str, Any]:
    adapters = _registry_adapters_by_target_kind()
    if target_kind in adapters:
        return dict(adapters[target_kind])
    raise FrontierRateAttackBootstrapError(f"materializer target kind is not registered: {target_kind}")


def _registry_adapters_by_target_kind() -> dict[str, dict[str, Any]]:
    return {
        str(row["target_kind"]): dict(row)
        for row in registry_manifest()["adapters"]
        if str(row.get("target_kind") or "")
    }


def _registry_candidate_executable_target_kinds() -> list[str]:
    return sorted(
        target_kind
        for target_kind, row in _registry_adapters_by_target_kind().items()
        if row.get("executable") is True
        and row.get("emits_candidate_archive") is True
        and row.get("planning_only") is not True
    )


def _target_coverage_report(
    *,
    requested_target_kinds: Sequence[str],
    backlog_rows: Sequence[Mapping[str, Any]],
    target_omissions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    registry_candidates = _registry_candidate_executable_target_kinds()
    supported = ordered_unique(ARCHIVE_RATE_ATTACK_SUPPORTED_TARGET_KINDS)
    included = ordered_unique(
        str(row.get("target_kind") or "")
        for row in backlog_rows
        if str(row.get("target_kind") or "")
    )
    context_omitted = ordered_unique(
        str(row.get("target_kind") or "")
        for row in target_omissions
        if str(row.get("target_kind") or "")
    )
    deferred_rows: list[dict[str, Any]] = []
    unclassified: list[str] = []
    for target_kind in registry_candidates:
        if target_kind in supported:
            continue
        rationale = FRONTIER_RATE_ATTACK_DEFERRED_TARGET_RATIONALES.get(target_kind)
        if rationale is None:
            unclassified.append(target_kind)
            continue
        deferred_rows.append(
            {
                "target_kind": target_kind,
                "rationale": rationale,
                "deferred_to": (
                    "dqs1_local_first_feedback_cycle"
                    if target_kind == DQS1_PAIRSET_TARGET_KIND
                    else "inverse_steganalysis_acquisition_chain"
                ),
            }
        )
    blockers = [
        f"unclassified_executable_candidate_materializer:{target_kind}"
        for target_kind in unclassified
    ]
    return apply_proxy_evidence_boundary(
        {
            "schema": TARGET_COVERAGE_SCHEMA,
            "requested_target_kinds": ordered_unique(requested_target_kinds),
            "archive_rate_supported_target_kinds": supported,
            "registry_candidate_executable_target_kinds": registry_candidates,
            "included_target_kinds": included,
            "context_omitted_target_kinds": context_omitted,
            "deferred_registry_target_rows": deferred_rows,
            "unclassified_executable_candidate_target_kinds": unclassified,
            "coverage_complete": not blockers,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=blockers
        or ("frontier_rate_attack_target_coverage_is_local_planning_signal",),
    )


def _archive_specs(records: Sequence[Mapping[str, Any]]) -> list[str]:
    specs = []
    for record in records:
        label = str(record.get("label") or "").strip()
        path = str(record.get("path") or "").strip()
        if not label or not path:
            raise FrontierRateAttackBootstrapError("archive records require label and path")
        specs.append(f"{label}={path}")
    return ordered_unique(specs)


def _shared_single_member_name(records: Sequence[Mapping[str, Any]]) -> str | None:
    names: list[str] = []
    for record in records:
        members = record.get("zip_members")
        if not isinstance(members, list) or len(members) != 1:
            return None
        member = members[0]
        if not isinstance(member, Mapping):
            return None
        name = member.get("name")
        if not isinstance(name, str) or not name:
            return None
        names.append(name)
    unique = ordered_unique(names)
    return unique[0] if len(unique) == 1 else None


def _target_context(
    *,
    target_kind: str,
    archive_records: Sequence[Mapping[str, Any]],
    output_root: Path,
    member_name: str | None,
    section_manifest: str | None,
    section_names: Sequence[str],
    merge_contract: str | None,
    merged_member_name: str | None,
    payload_member_name: str,
    full_frame_inflate_parity_proof: str | None,
    tensor_manifest: str | None,
    factorization_contract: str | None,
    tensor_factorize_rank: int | None,
    zip_compression_methods: Sequence[str],
    zip_compresslevels: Sequence[int],
    min_free_bytes: int,
    allow_overwrite: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    context: dict[str, Any] = {
        "sweep_archive_specs": _archive_specs(archive_records),
        "sweep_output_dir": (output_root / target_kind).as_posix(),
        "sweep_output_json": (output_root / target_kind / "sweep.json").as_posix(),
        "sweep_observation_jsonl": (
            output_root / target_kind / "observations.jsonl"
        ).as_posix(),
        "min_free_bytes": min_free_bytes,
        "allow_overwrite": allow_overwrite,
        **FALSE_AUTHORITY,
    }
    blockers: list[str] = []
    if target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND:
        if member_name is None:
            blockers.append("packet_member_recompress_requires_single_shared_member_or_member_name")
        else:
            context["member_name"] = member_name
        context["zip_compression_methods"] = list(zip_compression_methods)
        context["zip_compresslevels"] = [str(level) for level in zip_compresslevels]
    elif target_kind == ARCHIVE_ZIP_REPACK_TARGET_KIND:
        context["zip_compression_methods"] = list(zip_compression_methods)
        context["zip_compresslevels"] = [str(level) for level in zip_compresslevels]
    elif target_kind == PACKET_MEMBER_ZIP_HEADER_ELIDE_TARGET_KIND:
        if member_name is not None:
            context["member_name"] = member_name
        else:
            context["all_members"] = True
            context["member_selection"] = "all_members"
    elif target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
        merge_blockers: list[str] = []
        for record in archive_records:
            merge_blockers.extend(
                _packet_member_merge_record_blockers(
                    record,
                    merged_member_name=merged_member_name,
                )
            )
        blockers.extend(ordered_unique(merge_blockers))
        context["all_members"] = True
        context["member_selection"] = "all_members"
        if merge_contract is not None:
            context["merge_contract"] = merge_contract
        if merged_member_name is not None:
            context["merged_member_name"] = merged_member_name
    elif target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
        dfl1_blockers: list[str] = []
        for record in archive_records:
            dfl1_blockers.extend(
                _renderer_payload_dfl1_record_blockers(
                    record,
                    payload_member_name=payload_member_name,
                )
            )
        blockers.extend(ordered_unique(dfl1_blockers))
        context["member_names"] = list(RENDERER_PAYLOAD_DFL1_MEMBER_NAMES)
        context["payload_member_name"] = payload_member_name
        if full_frame_inflate_parity_proof is not None:
            context["full_frame_inflate_parity_proof"] = full_frame_inflate_parity_proof
    elif target_kind == FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND:
        if len(archive_records) != 1:
            blockers.append("selector_context_recode_requires_per_archive_context")
        else:
            selector_context, selector_blockers = _feca_selector_record_context(
                archive_records[0]
            )
            blockers.extend(selector_blockers)
            if selector_context is not None:
                context.update(selector_context)
                context["output_dir"] = (
                    output_root / target_kind / "feca_selector_context_recode"
                ).as_posix()
                if full_frame_inflate_parity_proof is not None:
                    context["full_frame_inflate_parity_proof"] = (
                        full_frame_inflate_parity_proof
                    )
    elif target_kind == FP11_SOURCE_BROTLI_RECODE_TARGET_KIND:
        if len(archive_records) != 1:
            blockers.append("fp11_source_brotli_recode_requires_per_archive_context")
        else:
            brotli_context, brotli_blockers = _fp11_source_brotli_record_context(
                archive_records[0]
            )
            blockers.extend(brotli_blockers)
            if brotli_context is not None:
                context.update(brotli_context)
                context["output_dir"] = (
                    output_root / target_kind / "fp11_source_brotli_recode"
                ).as_posix()
                if full_frame_inflate_parity_proof is not None:
                    context["full_frame_inflate_parity_proof"] = (
                        full_frame_inflate_parity_proof
                    )
    elif target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND:
        if section_manifest is None:
            blockers.append("archive_section_entropy_recode_requires_section_manifest")
        else:
            context["section_manifest"] = section_manifest
            context["section_names"] = list(section_names)
    elif target_kind == TENSOR_FACTORIZE_TARGET_KIND:
        if tensor_manifest is None:
            blockers.append("tensor_factorize_requires_tensor_manifest")
        else:
            context["tensor_manifest"] = tensor_manifest
        if factorization_contract is None and tensor_factorize_rank is None:
            blockers.append("tensor_factorize_requires_factorization_contract_or_rank")
        elif factorization_contract is not None:
            context["factorization_contract"] = factorization_contract
        else:
            context["rank"] = tensor_factorize_rank
    else:
        blockers.append(f"unsupported_frontier_rate_attack_target:{target_kind}")
    return (None, blockers) if blockers else (context, [])


def build_frontier_rate_attack_payloads(
    *,
    repo_root: str | Path,
    queue_id: str,
    archive_records: Sequence[Mapping[str, Any]],
    results_root: str | Path,
    target_kinds: Sequence[str] = DEFAULT_EXECUTABLE_TARGET_KINDS,
    include_optional_target_blockers: bool = True,
    member_name: str | None = None,
    section_manifest: str | None = None,
    section_manifest_by_archive_label: Mapping[str, str] | None = None,
    section_names: Sequence[str] = (),
    section_names_by_archive_label: Mapping[str, Sequence[str]] | None = None,
    merge_contract: str | None = None,
    merged_member_name: str | None = None,
    payload_member_name: str = "p",
    full_frame_inflate_parity_proof: str | None = None,
    tensor_manifest: str | None = None,
    factorization_contract: str | None = None,
    tensor_factorize_rank: int | None = None,
    zip_compression_methods: Sequence[str] = ("stored", "deflated"),
    zip_compresslevels: Sequence[int] = (1, 6, 9),
    min_free_bytes: int = 0,
    allow_overwrite: bool = False,
    local_cpu_concurrency: int = 1,
    lane_id: str | None = None,
    source_work_queue_path: str | Path | None = None,
    include_exact_readiness_followup: bool = False,
    exact_readiness_followup_require_ready: bool = False,
    target_optimization_profile: Mapping[str, Any] | None = None,
    require_target_profile_ready: bool = False,
) -> dict[str, Any]:
    """Compile archives and materializer targets into durable queue payloads."""

    repo = Path(repo_root)
    if not queue_id.strip():
        raise FrontierRateAttackBootstrapError("queue_id must be non-empty")
    if not archive_records:
        raise FrontierRateAttackBootstrapError("at least one archive record is required")
    if local_cpu_concurrency < 1:
        raise FrontierRateAttackBootstrapError("local_cpu_concurrency must be >= 1")
    if include_exact_readiness_followup and source_work_queue_path is None:
        raise FrontierRateAttackBootstrapError(
            "include_exact_readiness_followup requires source_work_queue_path"
        )
    checked_records = [dict(record) for record in archive_records]
    for index, record in enumerate(checked_records):
        require_no_truthy_authority_fields(record, context=f"archive_records[{index}]")
    target_profile_payload: dict[str, Any] | None = None
    target_profile_metadata: dict[str, Any] | None = None
    if target_optimization_profile is not None:
        if target_optimization_profile.get("schema") != TARGET_OPTIMIZATION_PROFILE_SCHEMA:
            raise FrontierRateAttackBootstrapError(
                "target_optimization_profile schema mismatch"
            )
        require_no_truthy_authority_fields(
            target_optimization_profile,
            context="frontier_rate_attack_bootstrap.target_optimization_profile",
        )
        target_profile_payload = dict(target_optimization_profile)
        target_profile_metadata = target_optimization_profile_queue_metadata(
            target_profile_payload
        )
        if (
            require_target_profile_ready
            and target_profile_metadata.get("profile_ready") is not True
        ):
            blockers = ", ".join(str(item) for item in target_profile_metadata.get("blockers") or [])
            raise FrontierRateAttackBootstrapError(
                "target optimization profile is not ready"
                + (f": {blockers}" if blockers else "")
            )
    output_root = _resolve_path(results_root, repo_root=repo) / queue_id
    shared_member = member_name or _shared_single_member_name(checked_records)
    requested_targets = ordered_unique(
        [
            *target_kinds,
            *(DEFAULT_OPTIONAL_TARGET_KINDS if include_optional_target_blockers else ()),
        ]
    )
    section_manifest_map = {
        str(key): str(value)
        for key, value in (section_manifest_by_archive_label or {}).items()
        if str(key) and str(value)
    }
    section_names_map = {
        str(key): tuple(str(name) for name in value if str(name))
        for key, value in (section_names_by_archive_label or {}).items()
        if str(key)
    }
    contexts_rows: list[dict[str, Any]] = []
    backlog_rows: list[dict[str, Any]] = []
    target_omissions: list[dict[str, Any]] = []

    def append_target_row(
        *,
        target_kind: str,
        adapter: Mapping[str, Any],
        backlog_key: str,
        context: Mapping[str, Any],
        source_records: Sequence[Mapping[str, Any]],
    ) -> None:
        contexts_rows.append(
            {
                "backlog_key": backlog_key,
                "target_kind": target_kind,
                "materializer_id": adapter.get("materializer_id"),
                "context": {
                    **dict(context),
                    **(
                        {"target_optimization_profile": target_profile_metadata}
                        if target_profile_metadata is not None
                        else {}
                    ),
                },
            }
        )
        backlog_rows.append(
            apply_proxy_evidence_boundary(
                {
                    "schema": "byte_shaving_materializer_backlog_row.v1",
                    "backlog_key": backlog_key,
                    "backlog_rank": len(backlog_rows) + 1,
                    "gap_class": "frontier_final_rate_attack_materializer_sweep",
                    "unit_kind": adapter.get("unit_kind"),
                    "operation_family": adapter.get("operation_family"),
                    "target_kind": target_kind,
                    "materializer_id": adapter.get("materializer_id"),
                    "receiver_contract_id": adapter.get("receiver_contract_id"),
                    "receiver_contract_kind": adapter.get("receiver_contract_kind"),
                    "receiver_contract_status": "local_receiver_proof_required",
                    "cooperative_receiver_required": adapter.get("cooperative_receiver_required"),
                    "materialization_resource_kind": adapter.get("materialization_resource_kind")
                    or "local_cpu",
                    "suggested_materializer_count": 1,
                    "suggested_materializers": [dict(adapter)],
                    "blocked_row_count": 1,
                    "blocked_resolution_count": 1,
                    "selected_operation_count": len(source_records),
                    "affected_unit_count": len(source_records),
                    "candidate_saved_bytes_sum": 0,
                    "expected_score_gain_sum": 0.0,
                    "source_unit_ids": [record["label"] for record in source_records],
                    "source_selection_ids": [record["label"] for record in source_records],
                    "source_selection_samples": [
                        {
                            "selection_id": record["label"],
                            "selection_kind": "frontier_archive",
                            "archive_sha256": record["sha256"],
                            "archive_bytes": record["bytes"],
                        }
                        for record in source_records[:8]
                    ],
                    **(
                        {
                            "target_profile_id": target_profile_metadata.get(
                                "target_profile_id"
                            ),
                            "target_mode": target_profile_metadata.get("target_mode"),
                            "declared_overfit_allowed": target_profile_metadata.get(
                                "declared_overfit_allowed"
                            )
                            is True,
                            "target_video_paths": target_profile_metadata.get(
                                "target_video_paths"
                            )
                            or [],
                            "target_corpus_manifest_path": target_profile_metadata.get(
                                "target_corpus_manifest_path"
                            ),
                        }
                        if target_profile_metadata is not None
                        else {}
                    ),
                    **FALSE_AUTHORITY,
                },
                dispatch_blockers=("frontier_rate_attack_local_materializer_sweep_only",),
            )
        )

    for target_kind in requested_targets:
        adapter = _adapter_by_target_kind(target_kind)
        if target_kind == PACKET_MEMBER_RECOMPRESS_TARGET_KIND and shared_member is None:
            for record in checked_records:
                label = str(record["label"])
                member_names = _record_zip_member_names(record)
                if not member_names:
                    target_omissions.append(
                        apply_proxy_evidence_boundary(
                            {
                                "schema": "frontier_rate_attack_target_omission.v1",
                                "target_kind": target_kind,
                                "archive_label": label,
                                "materializer_id": adapter.get("materializer_id"),
                                "blockers": ["packet_member_recompress_archive_has_no_members"],
                                **FALSE_AUTHORITY,
                            },
                            dispatch_blockers=("packet_member_recompress_archive_has_no_members",),
                        )
                    )
                    continue
                for member in member_names:
                    context, per_member_blockers = _target_context(
                        target_kind=target_kind,
                        archive_records=[record],
                        output_root=(
                            output_root
                            / "per_archive"
                            / _clean_id(label)
                            / "per_member"
                            / _clean_id(member, fallback="member")
                        ),
                        member_name=member,
                        section_manifest=section_manifest,
                        section_names=section_names,
                        merge_contract=merge_contract,
                        merged_member_name=merged_member_name,
                        payload_member_name=payload_member_name,
                        full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
                        tensor_manifest=tensor_manifest,
                        factorization_contract=factorization_contract,
                        tensor_factorize_rank=tensor_factorize_rank,
                        zip_compression_methods=zip_compression_methods,
                        zip_compresslevels=zip_compresslevels,
                        min_free_bytes=min_free_bytes,
                        allow_overwrite=allow_overwrite,
                    )
                    if per_member_blockers:
                        target_omissions.append(
                            apply_proxy_evidence_boundary(
                                {
                                    "schema": "frontier_rate_attack_target_omission.v1",
                                    "target_kind": target_kind,
                                    "archive_label": label,
                                    "member_name": member,
                                    "materializer_id": adapter.get("materializer_id"),
                                    "blockers": ordered_unique(per_member_blockers),
                                    **FALSE_AUTHORITY,
                                },
                                dispatch_blockers=per_member_blockers,
                            )
                        )
                        continue
                    assert context is not None
                    append_target_row(
                        target_kind=target_kind,
                        adapter=adapter,
                        backlog_key=(
                            f"frontier_rate_attack:{target_kind}:{label}:"
                            f"{_clean_id(member, fallback='member')}"
                        ),
                        context=context,
                        source_records=[record],
                    )
            continue
        if (
            target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
            and section_manifest is None
            and section_manifest_map
        ):
            for record in checked_records:
                label = str(record["label"])
                per_archive_blockers: list[str] = []
                manifest_path = section_manifest_map.get(label)
                per_archive_section_names = tuple(section_names_map.get(label, ()))
                if manifest_path is None:
                    per_archive_blockers.append(
                        "archive_section_entropy_recode_missing_derived_section_manifest"
                    )
                if not per_archive_section_names:
                    per_archive_blockers.append(
                        "archive_section_entropy_recode_requires_brotli_decompressible_section"
                    )
                context = None
                if not per_archive_blockers:
                    context, per_archive_blockers = _target_context(
                        target_kind=target_kind,
                        archive_records=[record],
                        output_root=output_root / "per_archive" / _clean_id(label),
                        member_name=shared_member,
                        section_manifest=manifest_path,
                        section_names=per_archive_section_names,
                        merge_contract=merge_contract,
                        merged_member_name=merged_member_name,
                        payload_member_name=payload_member_name,
                        full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
                        tensor_manifest=tensor_manifest,
                        factorization_contract=factorization_contract,
                        tensor_factorize_rank=tensor_factorize_rank,
                        zip_compression_methods=zip_compression_methods,
                        zip_compresslevels=zip_compresslevels,
                        min_free_bytes=min_free_bytes,
                        allow_overwrite=allow_overwrite,
                    )
                if per_archive_blockers:
                    target_omissions.append(
                        apply_proxy_evidence_boundary(
                            {
                                "schema": "frontier_rate_attack_target_omission.v1",
                                "target_kind": target_kind,
                                "archive_label": label,
                                "materializer_id": adapter.get("materializer_id"),
                                "blockers": ordered_unique(per_archive_blockers),
                                **FALSE_AUTHORITY,
                            },
                            dispatch_blockers=per_archive_blockers,
                        )
                    )
                    continue
                assert context is not None
                append_target_row(
                    target_kind=target_kind,
                    adapter=adapter,
                    backlog_key=f"frontier_rate_attack:{target_kind}:{label}",
                    context=context,
                    source_records=[record],
                )
            continue
        if target_kind == PACKET_MEMBER_MERGE_TARGET_KIND:
            for record in checked_records:
                label = str(record["label"])
                context, per_archive_blockers = _target_context(
                    target_kind=target_kind,
                    archive_records=[record],
                    output_root=output_root / "per_archive" / _clean_id(label),
                    member_name=shared_member,
                    section_manifest=section_manifest,
                    section_names=section_names,
                    merge_contract=merge_contract,
                    merged_member_name=merged_member_name,
                    payload_member_name=payload_member_name,
                    full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
                    tensor_manifest=tensor_manifest,
                    factorization_contract=factorization_contract,
                    tensor_factorize_rank=tensor_factorize_rank,
                    zip_compression_methods=zip_compression_methods,
                    zip_compresslevels=zip_compresslevels,
                    min_free_bytes=min_free_bytes,
                    allow_overwrite=allow_overwrite,
                )
                if per_archive_blockers:
                    target_omissions.append(
                        apply_proxy_evidence_boundary(
                            {
                                "schema": "frontier_rate_attack_target_omission.v1",
                                "target_kind": target_kind,
                                "archive_label": label,
                                "materializer_id": adapter.get("materializer_id"),
                                "blockers": ordered_unique(per_archive_blockers),
                                **FALSE_AUTHORITY,
                            },
                            dispatch_blockers=per_archive_blockers,
                        )
                    )
                    continue
                assert context is not None
                append_target_row(
                    target_kind=target_kind,
                    adapter=adapter,
                    backlog_key=f"frontier_rate_attack:{target_kind}:{label}",
                    context=context,
                    source_records=[record],
                )
            continue
        if target_kind == RENDERER_PAYLOAD_DFL1_TARGET_KIND:
            for record in checked_records:
                label = str(record["label"])
                context, per_archive_blockers = _target_context(
                    target_kind=target_kind,
                    archive_records=[record],
                    output_root=output_root / "per_archive" / _clean_id(label),
                    member_name=shared_member,
                    section_manifest=section_manifest,
                    section_names=section_names,
                    merge_contract=merge_contract,
                    merged_member_name=merged_member_name,
                    payload_member_name=payload_member_name,
                    full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
                    tensor_manifest=tensor_manifest,
                    factorization_contract=factorization_contract,
                    tensor_factorize_rank=tensor_factorize_rank,
                    zip_compression_methods=zip_compression_methods,
                    zip_compresslevels=zip_compresslevels,
                    min_free_bytes=min_free_bytes,
                    allow_overwrite=allow_overwrite,
                )
                if per_archive_blockers:
                    target_omissions.append(
                        apply_proxy_evidence_boundary(
                            {
                                "schema": "frontier_rate_attack_target_omission.v1",
                                "target_kind": target_kind,
                                "archive_label": label,
                                "materializer_id": adapter.get("materializer_id"),
                                "blockers": ordered_unique(per_archive_blockers),
                                **FALSE_AUTHORITY,
                            },
                            dispatch_blockers=per_archive_blockers,
                        )
                    )
                    continue
                assert context is not None
                append_target_row(
                    target_kind=target_kind,
                    adapter=adapter,
                    backlog_key=f"frontier_rate_attack:{target_kind}:{label}",
                    context=context,
                    source_records=[record],
                )
            continue
        if target_kind == FECA_SELECTOR_REPARAMETERIZE_TARGET_KIND:
            for record in checked_records:
                label = str(record["label"])
                context, per_archive_blockers = _target_context(
                    target_kind=target_kind,
                    archive_records=[record],
                    output_root=output_root / "per_archive" / _clean_id(label),
                    member_name=shared_member,
                    section_manifest=section_manifest,
                    section_names=section_names,
                    merge_contract=merge_contract,
                    merged_member_name=merged_member_name,
                    payload_member_name=payload_member_name,
                    full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
                    tensor_manifest=tensor_manifest,
                    factorization_contract=factorization_contract,
                    tensor_factorize_rank=tensor_factorize_rank,
                    zip_compression_methods=zip_compression_methods,
                    zip_compresslevels=zip_compresslevels,
                    min_free_bytes=min_free_bytes,
                    allow_overwrite=allow_overwrite,
                )
                if per_archive_blockers:
                    target_omissions.append(
                        apply_proxy_evidence_boundary(
                            {
                                "schema": "frontier_rate_attack_target_omission.v1",
                                "target_kind": target_kind,
                                "archive_label": label,
                                "materializer_id": adapter.get("materializer_id"),
                                "blockers": ordered_unique(per_archive_blockers),
                                **FALSE_AUTHORITY,
                            },
                            dispatch_blockers=per_archive_blockers,
                        )
                    )
                    continue
                assert context is not None
                append_target_row(
                    target_kind=target_kind,
                    adapter=adapter,
                    backlog_key=f"frontier_rate_attack:{target_kind}:{label}",
                    context=context,
                    source_records=[record],
                )
            continue
        if target_kind == FP11_SOURCE_BROTLI_RECODE_TARGET_KIND:
            for record in checked_records:
                label = str(record["label"])
                context, per_archive_blockers = _target_context(
                    target_kind=target_kind,
                    archive_records=[record],
                    output_root=output_root / "per_archive" / _clean_id(label),
                    member_name=shared_member,
                    section_manifest=section_manifest,
                    section_names=section_names,
                    merge_contract=merge_contract,
                    merged_member_name=merged_member_name,
                    payload_member_name=payload_member_name,
                    full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
                    tensor_manifest=tensor_manifest,
                    factorization_contract=factorization_contract,
                    tensor_factorize_rank=tensor_factorize_rank,
                    zip_compression_methods=zip_compression_methods,
                    zip_compresslevels=zip_compresslevels,
                    min_free_bytes=min_free_bytes,
                    allow_overwrite=allow_overwrite,
                )
                if per_archive_blockers:
                    target_omissions.append(
                        apply_proxy_evidence_boundary(
                            {
                                "schema": "frontier_rate_attack_target_omission.v1",
                                "target_kind": target_kind,
                                "archive_label": label,
                                "materializer_id": adapter.get("materializer_id"),
                                "blockers": ordered_unique(per_archive_blockers),
                                **FALSE_AUTHORITY,
                            },
                            dispatch_blockers=per_archive_blockers,
                        )
                    )
                    continue
                assert context is not None
                append_target_row(
                    target_kind=target_kind,
                    adapter=adapter,
                    backlog_key=f"frontier_rate_attack:{target_kind}:{label}",
                    context=context,
                    source_records=[record],
                )
            continue
        if (
            target_kind == ARCHIVE_SECTION_ENTROPY_RECODE_TARGET_KIND
            and section_manifest is not None
            and len(checked_records) > 1
        ):
            blockers = [
                "archive_section_entropy_recode_requires_per_archive_section_manifest_for_multi_archive_sweep"
            ]
            target_omissions.append(
                apply_proxy_evidence_boundary(
                    {
                        "schema": "frontier_rate_attack_target_omission.v1",
                        "target_kind": target_kind,
                        "materializer_id": adapter.get("materializer_id"),
                        "archive_labels": [str(record["label"]) for record in checked_records],
                        "blockers": blockers,
                        **FALSE_AUTHORITY,
                    },
                    dispatch_blockers=blockers,
                )
            )
            continue
        backlog_key = f"frontier_rate_attack:{target_kind}"
        context, blockers = _target_context(
            target_kind=target_kind,
            archive_records=checked_records,
            output_root=output_root,
            member_name=shared_member,
            section_manifest=section_manifest,
            section_names=section_names,
            merge_contract=merge_contract,
            merged_member_name=merged_member_name,
            payload_member_name=payload_member_name,
            full_frame_inflate_parity_proof=full_frame_inflate_parity_proof,
            tensor_manifest=tensor_manifest,
            factorization_contract=factorization_contract,
            tensor_factorize_rank=tensor_factorize_rank,
            zip_compression_methods=zip_compression_methods,
            zip_compresslevels=zip_compresslevels,
            min_free_bytes=min_free_bytes,
            allow_overwrite=allow_overwrite,
        )
        if blockers:
            target_omissions.append(
                apply_proxy_evidence_boundary(
                    {
                        "schema": "frontier_rate_attack_target_omission.v1",
                        "target_kind": target_kind,
                        "materializer_id": adapter.get("materializer_id"),
                        "blockers": ordered_unique(blockers),
                        **FALSE_AUTHORITY,
                    },
                    dispatch_blockers=blockers,
                )
            )
            continue
        assert context is not None
        append_target_row(
            target_kind=target_kind,
            adapter=adapter,
            backlog_key=backlog_key,
            context=context,
            source_records=checked_records,
        )
    if not backlog_rows:
        raise FrontierRateAttackBootstrapError(
            "no executable frontier-rate materializer targets; blockers: "
            + json.dumps(target_omissions, sort_keys=True)
        )
    target_coverage = _target_coverage_report(
        requested_target_kinds=requested_targets,
        backlog_rows=backlog_rows,
        target_omissions=target_omissions,
    )
    if target_coverage.get("coverage_complete") is not True:
        raise FrontierRateAttackBootstrapError(
            "unclassified executable candidate materializer targets: "
            + ", ".join(
                str(item)
                for item in target_coverage.get(
                    "unclassified_executable_candidate_target_kinds"
                )
                or []
            )
        )
    contexts_payload = apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_CONTEXTS_SCHEMA,
            "generated_at_utc": _utc_now(),
            "queue_id": queue_id,
            "rows": contexts_rows,
            **(
                {"target_optimization_profile": target_profile_metadata}
                if target_profile_metadata is not None
                else {}
            ),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=("frontier_rate_attack_contexts_are_local_only",),
    )
    contexts = materializer_contexts_from_payload(contexts_payload)
    backlog = apply_proxy_evidence_boundary(
        {
            "schema": MATERIALIZER_BACKLOG_SCHEMA,
            "tool": "comma_lab.scheduler.frontier_rate_attack_bootstrap",
            "generated_at_utc": _utc_now(),
            "backlog_row_count": len(backlog_rows),
            "rows": backlog_rows,
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=("frontier_rate_attack_backlog_is_planning_only",),
    )
    work_queue = build_materializer_work_queue(
        backlog,
        repo_root=repo,
        contexts=contexts,
        source_plan_path=None,
    )
    queue = build_materializer_execution_queue(
        work_queue,
        queue_id=queue_id,
        repo_root=repo,
        lane_id=lane_id,
        source_work_queue_path=source_work_queue_path,
        local_cpu_concurrency=local_cpu_concurrency,
        resource_concurrency={"local_cpu": local_cpu_concurrency},
        include_exact_readiness_followup=include_exact_readiness_followup,
        exact_readiness_followup_require_ready=(
            exact_readiness_followup_require_ready
        ),
    )
    queue_metadata = apply_proxy_evidence_boundary(
        {
            "schema": BOOTSTRAP_SCHEMA,
            "archive_count": len(checked_records),
            "archive_labels": [record["label"] for record in checked_records],
            "target_kinds": [row["target_kind"] for row in backlog_rows],
            "target_omissions": target_omissions,
            "target_coverage": target_coverage,
            **(
                {"target_optimization_profile": target_profile_metadata}
                if target_profile_metadata is not None
                else {}
            ),
            "exact_readiness_followup_requested": bool(
                include_exact_readiness_followup
            ),
            "exact_readiness_followup_require_ready": bool(
                exact_readiness_followup_require_ready
            ),
            "allowed_use": (
                "local_final_rate_attack_materializer_sweep_with_exact_readiness_handoff"
                if include_exact_readiness_followup
                else "local_final_rate_attack_materializer_sweep_only"
            ),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            "frontier_rate_attack_is_local_materializer_signal_only",
            "exact_auth_eval_required_before_score_claim",
        ),
    )
    queue = normalize_queue_definition(queue)
    for experiment in queue["experiments"]:
        metadata = dict(experiment.get("metadata") or {})
        metadata["frontier_rate_attack_bootstrap"] = queue_metadata
        if target_profile_metadata is not None:
            metadata["frontier_target_optimization_profile"] = target_profile_metadata
        experiment["metadata"] = metadata
    queue = normalize_queue_definition(queue)
    bootstrap = apply_proxy_evidence_boundary(
        {
            "schema": BOOTSTRAP_SCHEMA,
            "generated_at_utc": _utc_now(),
            "queue_id": queue_id,
            "archive_count": len(checked_records),
            "archives": checked_records,
            **(
                {
                    "target_optimization_profile": target_profile_payload,
                    "target_optimization_profile_metadata": target_profile_metadata,
                }
                if target_profile_payload is not None
                and target_profile_metadata is not None
                else {}
            ),
            "executable_target_count": len(backlog_rows),
            "executable_target_kinds": [row["target_kind"] for row in backlog_rows],
            "target_omissions": target_omissions,
            "target_coverage": target_coverage,
            "results_root": _repo_rel(output_root, repo),
            "shared_member_name": shared_member,
            "queue_schema": queue.get("schema"),
            "experiment_count": len(queue.get("experiments", [])),
            "step_count": sum(len(exp.get("steps", [])) for exp in queue.get("experiments", [])),
            "controls": queue.get("controls"),
            "exact_readiness_followup_requested": bool(
                include_exact_readiness_followup
            ),
            "exact_readiness_followup_require_ready": bool(
                exact_readiness_followup_require_ready
            ),
            "allowed_use": (
                "local_final_rate_attack_materializer_sweep_with_exact_readiness_handoff"
                if include_exact_readiness_followup
                else "local_final_rate_attack_materializer_sweep_only"
            ),
            **FALSE_AUTHORITY,
        },
        dispatch_blockers=(
            "frontier_rate_attack_is_local_materializer_signal_only",
            "exact_auth_eval_required_before_score_claim",
        ),
    )
    return {
        "bootstrap": bootstrap,
        "contexts": contexts_payload,
        "backlog": backlog,
        "work_queue": work_queue,
        "queue": queue,
        "target_coverage": target_coverage,
    }


__all__ = [
    "ARCHIVE_RATE_ATTACK_SUPPORTED_TARGET_KINDS",
    "BOOTSTRAP_SCHEMA",
    "DEFAULT_EXECUTABLE_TARGET_KINDS",
    "DEFAULT_FRONTIER_POINTER",
    "DEFAULT_OPTIONAL_TARGET_KINDS",
    "DERIVED_PACKET_MEMBER_MERGE_CONTRACT_SCHEMA",
    "DERIVED_SECTION_MANIFEST_BATCH_SCHEMA",
    "DERIVED_SECTION_MANIFEST_SCHEMA",
    "FRONTIER_ARCHIVE_RECORD_SCHEMA",
    "FRONTIER_ARCHIVE_RESOLUTION_SCHEMA",
    "TARGET_COVERAGE_SCHEMA",
    "FrontierRateAttackBootstrapError",
    "archive_record",
    "build_frontier_rate_attack_payloads",
    "derive_archive_section_recode_manifest",
    "derive_archive_section_recode_manifests",
    "derive_packet_member_merge_contract",
    "parse_archive_spec",
    "resolve_current_frontier_archive",
]
