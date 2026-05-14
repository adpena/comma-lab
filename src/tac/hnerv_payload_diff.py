# SPDX-License-Identifier: MIT
"""No-op-resistant section diffs for packed HNeRV archives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_lowlevel_packer import (
    REPACKABLE_SECTIONS,
    HnervLowlevelPackError,
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_payload_diff"


def build_hnerv_payload_diff(
    source_archive: str | Path,
    candidate_archive: str | Path,
    *,
    source_label: str = "source",
    candidate_label: str = "candidate",
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two strict single-member packed HNeRV archives.

    The result is byte-forensic optimizer input. It never claims score movement
    and it does not unlock exact eval; it only proves whether a candidate
    changed charged HNeRV sections and whether brotli raw payloads changed.
    """

    source = read_strict_single_member_zip(source_archive)
    candidate = read_strict_single_member_zip(candidate_archive)
    source_packed = parse_ff_packed_brotli_hnerv(source.payload)
    candidate_packed = parse_ff_packed_brotli_hnerv(candidate.payload)
    blockers = _source_manifest_blockers(source, source_packed, source_manifest)
    if source.member_name != candidate.member_name:
        blockers.append("zip_member_mismatch")

    sections = [
        _section_diff(name, source_packed, candidate_packed)
        for name in (
            "packed_header_ff_len24",
            "decoder_packed_brotli",
            "latents_and_sidecar_brotli",
        )
    ]
    changed_sections = [row for row in sections if row["changed"] is True]
    if not changed_sections:
        blockers.append("no_payload_section_changed")

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_archive_preflight": not blockers and bool(changed_sections),
        "source_label": source_label,
        "candidate_label": candidate_label,
        "source_archive_path": str(Path(source_archive)),
        "candidate_archive_path": str(Path(candidate_archive)),
        "source_archive_sha256": source.archive_sha256,
        "candidate_archive_sha256": candidate.archive_sha256,
        "source_archive_bytes": source.archive_bytes,
        "candidate_archive_bytes": candidate.archive_bytes,
        "archive_byte_delta": candidate.archive_bytes - source.archive_bytes,
        "source_member_name": source.member_name,
        "candidate_member_name": candidate.member_name,
        "source_payload_sha256": sha256_bytes(source.payload),
        "candidate_payload_sha256": sha256_bytes(candidate.payload),
        "source_payload_bytes": source.member_bytes,
        "candidate_payload_bytes": candidate.member_bytes,
        "payload_byte_delta": candidate.member_bytes - source.member_bytes,
        "changed_section_count": len(changed_sections),
        "sections": sections,
        "blockers": blockers,
        "dispatch_blockers": [
            "requires_archive_manifest_preflight",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def dump_json(payload: Mapping[str, Any]) -> str:
    """Return stable JSON for a payload diff."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _section_diff(
    name: str,
    source: PackedHnervPayload,
    candidate: PackedHnervPayload,
) -> dict[str, Any]:
    source_bytes = source.section_bytes(name)
    candidate_bytes = candidate.section_bytes(name)
    row: dict[str, Any] = {
        "name": name,
        "source_bytes": len(source_bytes),
        "candidate_bytes": len(candidate_bytes),
        "byte_delta": len(candidate_bytes) - len(source_bytes),
        "source_sha256": sha256_bytes(source_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "changed": source_bytes != candidate_bytes,
    }
    if name in REPACKABLE_SECTIONS:
        row.update(_brotli_raw_diff(source_bytes, candidate_bytes))
    return row


def _brotli_raw_diff(source_bytes: bytes, candidate_bytes: bytes) -> dict[str, Any]:
    try:
        source_raw = brotli.decompress(source_bytes)
        candidate_raw = brotli.decompress(candidate_bytes)
    except brotli.error as exc:
        return {
            "brotli_decode_ok": False,
            "brotli_error": str(exc),
            "raw_equal": False,
        }
    if len(source_raw) != len(candidate_raw):
        changed_positions = None
        abs_delta_sum = None
    else:
        pairs = zip(source_raw, candidate_raw, strict=True)
        deltas = [abs(int(a) - int(b)) for a, b in pairs if a != b]
        changed_positions = len(deltas)
        abs_delta_sum = sum(deltas)
    return {
        "brotli_decode_ok": True,
        "raw_equal": source_raw == candidate_raw,
        "raw_bytes": len(source_raw),
        "candidate_raw_bytes": len(candidate_raw),
        "source_raw_sha256": sha256_bytes(source_raw),
        "candidate_raw_sha256": sha256_bytes(candidate_raw),
        "raw_changed_positions": changed_positions,
        "raw_abs_delta_sum": abs_delta_sum,
    }


def _source_manifest_blockers(
    source_archive,
    source_packed: PackedHnervPayload,
    manifest: Mapping[str, Any] | None,
) -> list[str]:
    if manifest is None:
        return []
    blockers: list[str] = []
    if manifest.get("archive_sha256") != source_archive.archive_sha256:
        blockers.append("source_manifest_archive_sha256_mismatch")
    if manifest.get("archive_bytes") != source_archive.archive_bytes:
        blockers.append("source_manifest_archive_bytes_mismatch")
    if manifest.get("zip_member") != source_archive.member_name:
        blockers.append("source_manifest_zip_member_mismatch")
    if manifest.get("payload_sha256") != sha256_bytes(source_archive.payload):
        blockers.append("source_manifest_payload_sha256_mismatch")
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        blockers.append("source_manifest_missing_sections")
        return blockers
    by_name = {
        str(section.get("name")): section
        for section in sections
        if isinstance(section, Mapping)
    }
    for section_name in (
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ):
        section = by_name.get(section_name)
        if section is None:
            blockers.append(f"source_manifest_missing_section:{section_name}")
            continue
        actual = source_packed.section_bytes(section_name)
        if section.get("bytes") != len(actual):
            blockers.append(f"source_manifest_section_bytes_mismatch:{section_name}")
        if section.get("sha256") != sha256_bytes(actual):
            blockers.append(f"source_manifest_section_sha256_mismatch:{section_name}")
    return blockers


__all__ = [
    "HnervLowlevelPackError",
    "build_hnerv_payload_diff",
    "dump_json",
]
