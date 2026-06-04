# SPDX-License-Identifier: MIT
"""Profile SNeRV SNAR1 packet header grammar overhead.

Lossless LF recoding can make the LF section tiny while the outer SNAR1 JSON
header remains dominant.  This module makes that header mass explicit without
changing receiver bytes or granting score authority.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    HEADER_LEN_FMT,
    SECTION_ORDER,
    SNERV_ARCHIVE_MAGIC,
    SNERV_ARCHIVE_SCHEMA,
    unpack_snerv_archive,
)

SCHEMA = "snerv_snar_header_grammar_profile.v1"
AXIS_TAG = "[receiver-packet-grammar:false-authority]"
DEFAULT_ARCHIVE_MEMBER = "0.bin"
DEFAULT_TOP_CONTRIBUTOR_LIMIT = 40


class SnervSnarHeaderGrammarProfileError(ValueError):
    """Raised when a SNAR1 header grammar profile cannot be built."""


def build_snerv_snar_header_grammar_profile(
    *,
    input_path: str | Path,
    hard_byte_ceilings: Sequence[int] = (),
    top_contributor_limit: int = DEFAULT_TOP_CONTRIBUTOR_LIMIT,
    generated_utc: str | None = None,
    raw_argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Return byte accounting for the outer SNAR1 header and metadata grammar."""

    path = Path(input_path).expanduser().resolve(strict=False)
    packet, input_kind, archive_member = _read_packet(path)
    parsed = _parse_outer_header(packet)
    decoded = unpack_snerv_archive(packet)
    header = parsed["header"]
    header_payload = parsed["header_payload"]
    header_prefix_bytes = int(parsed["header_prefix_bytes"])
    header_payload_bytes = len(header_payload)
    header_bytes = int(parsed["header_bytes"])
    packet_bytes = len(packet)
    section_rows = _section_rows(header)
    section_total = sum(int(row["bytes"]) for row in section_rows)
    payload_bytes = packet_bytes - header_bytes
    metadata = header.get("metadata") if isinstance(header.get("metadata"), Mapping) else {}
    metadata_json_bytes = _json_len(metadata)
    component_rows = _header_component_rows(header)
    metadata_rows = _top_metadata_rows(
        metadata,
        limit=int(top_contributor_limit),
    )
    ceiling_rows = [
        _ceiling_row(
            ceiling=int(ceiling),
            packet_bytes=packet_bytes,
            header_bytes=header_bytes,
            header_payload_bytes=header_payload_bytes,
            metadata_json_bytes=metadata_json_bytes,
            section_total=section_total,
        )
        for ceiling in hard_byte_ceilings
    ]
    header_rewrite_needed = bool(
        ceiling_rows
        and any(row["packet_over_ceiling_bytes"] > 0 for row in ceiling_rows)
        and any(row["header_bytes_can_cover_overrun"] for row in ceiling_rows)
    )
    return {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "generated_utc": generated_utc or datetime.now(UTC).isoformat(),
        "input": {
            "path": path.as_posix(),
            "kind": input_kind,
            "archive_member": archive_member,
            "bytes": packet_bytes,
            "sha256": _sha256(packet),
        },
        "packet": {
            "schema": header.get("schema"),
            "schema_valid": header.get("schema") == SNERV_ARCHIVE_SCHEMA,
            "bytes": packet_bytes,
            "sha256": _sha256(packet),
            "decoded_packet_sha256": decoded.packet_sha256,
        },
        "header": {
            "bytes": header_bytes,
            "prefix_bytes": header_prefix_bytes,
            "json_payload_bytes": header_payload_bytes,
            "sha256": _sha256(header_payload),
            "metadata_json_bytes": metadata_json_bytes,
            "component_rows": component_rows,
            "metadata_top_contributor_rows": metadata_rows,
            "metadata_key_count": len(metadata),
        },
        "payload": {
            "bytes": payload_bytes,
            "section_total_bytes": section_total,
            "unreferenced_payload_bytes": payload_bytes - section_total,
            "section_rows": section_rows,
            "decoded_section_bytes": {
                name: len(decoded.sections[name]) for name in SECTION_ORDER
            },
        },
        "byte_accounting": {
            "header_bytes": header_bytes,
            "section_total_bytes": section_total,
            "packet_bytes": packet_bytes,
            "header_to_section_ratio": (
                None if section_total <= 0 else float(header_bytes) / float(section_total)
            ),
            "header_dominates_sections": header_bytes > section_total,
            "metadata_dominates_header_payload": (
                metadata_json_bytes > (header_payload_bytes - metadata_json_bytes)
            ),
        },
        "hard_byte_ceiling_rows": ceiling_rows,
        "header_rewrite_needed_for_any_ceiling": header_rewrite_needed,
        "next_actions": _next_actions(header_rewrite_needed=header_rewrite_needed),
        "blockers": _blockers(header_rewrite_needed=header_rewrite_needed),
        "raw_argv": list(raw_argv),
        **FALSE_AUTHORITY,
    }


def _parse_outer_header(packet: bytes) -> dict[str, Any]:
    blob = bytes(packet)
    if not blob.startswith(SNERV_ARCHIVE_MAGIC):
        raise SnervSnarHeaderGrammarProfileError("input is not a SNAR1 packet")
    offset = len(SNERV_ARCHIVE_MAGIC)
    len_size = struct.calcsize(HEADER_LEN_FMT)
    if len(blob) < offset + len_size:
        raise SnervSnarHeaderGrammarProfileError("truncated SNAR1 header length")
    (header_len,) = struct.unpack(HEADER_LEN_FMT, blob[offset : offset + len_size])
    header_start = offset + len_size
    header_end = header_start + int(header_len)
    if header_end > len(blob):
        raise SnervSnarHeaderGrammarProfileError(
            "declared SNAR1 header exceeds packet bytes"
        )
    header_payload = blob[header_start:header_end]
    try:
        header = json.loads(header_payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SnervSnarHeaderGrammarProfileError("SNAR1 header is not JSON") from exc
    if not isinstance(header, Mapping):
        raise SnervSnarHeaderGrammarProfileError("SNAR1 header JSON is not an object")
    return {
        "header": dict(header),
        "header_payload": header_payload,
        "header_prefix_bytes": len(SNERV_ARCHIVE_MAGIC) + len_size,
        "header_bytes": len(SNERV_ARCHIVE_MAGIC) + len_size + len(header_payload),
    }


def _section_rows(header: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_row in header.get("sections") or ():
        if not isinstance(raw_row, Mapping):
            continue
        rows.append(
            {
                "name": str(raw_row.get("name") or ""),
                "offset": _int_or_none(raw_row.get("offset")),
                "bytes": int(raw_row.get("bytes") or 0),
                "sha256": raw_row.get("sha256"),
                "header_json_bytes": _json_len(raw_row),
            }
        )
    return rows


def _header_component_rows(header: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "component": str(key),
            "json_bytes": _json_len(value),
            "type": type(value).__name__,
            "item_count": _item_count(value),
        }
        for key, value in header.items()
    ]
    rows.sort(key=lambda row: (-int(row["json_bytes"]), str(row["component"])))
    return rows


def _top_metadata_rows(
    metadata: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(path: str, value: Any, depth: int) -> None:
        rows.append(
            {
                "path": path,
                "json_bytes": _json_len(value),
                "type": type(value).__name__,
                "item_count": _item_count(value),
                "depth": depth,
            }
        )
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{path}.{key}", child, depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{path}[{index}]", child, depth + 1)

    visit("$.metadata", metadata, 0)
    rows.sort(key=lambda row: (-int(row["json_bytes"]), str(row["path"])))
    return rows[: max(int(limit), 1)]


def _ceiling_row(
    *,
    ceiling: int,
    packet_bytes: int,
    header_bytes: int,
    header_payload_bytes: int,
    metadata_json_bytes: int,
    section_total: int,
) -> dict[str, Any]:
    overrun = max(int(packet_bytes) - int(ceiling), 0)
    return {
        "hard_byte_ceiling": int(ceiling),
        "packet_over_ceiling_bytes": overrun,
        "packet_under_ceiling": overrun == 0,
        "header_bytes_can_cover_overrun": bool(overrun > 0 and header_bytes >= overrun),
        "header_payload_bytes_can_cover_overrun": bool(
            overrun > 0 and header_payload_bytes >= overrun
        ),
        "metadata_json_bytes_can_cover_overrun": bool(
            overrun > 0 and metadata_json_bytes >= overrun
        ),
        "sections_alone_under_ceiling": bool(section_total <= int(ceiling)),
    }


def _blockers(*, header_rewrite_needed: bool) -> list[str]:
    blockers = [
        "snerv_snar_header_grammar_profile_false_authority",
        "receiver_visible_header_rewrite_candidate_missing",
        "receiver_replay_proof_after_header_rewrite_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
    ]
    if header_rewrite_needed:
        blockers.append("snerv_snar_packet_header_grammar_rewrite_required")
    return blockers


def _next_actions(*, header_rewrite_needed: bool) -> list[str]:
    if header_rewrite_needed:
        return [
            "build_receiver_visible_snar_header_minimization_candidate",
            "prove_unpack_snerv_archive_accepts_minimized_header_packet",
            "rerun_snerv_lf_recode_admission_with_header_rewrite_report",
        ]
    return [
        "preserve_header_profile_as_packet_grammar_evidence",
        "continue_non_header_byte_accounting_before_long_training",
    ]


def _read_packet(path: Path) -> tuple[bytes, str, str | None]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SnervSnarHeaderGrammarProfileError(f"cannot read input: {path}") from exc
    if data.startswith(SNERV_ARCHIVE_MAGIC):
        return data, "snar1_packet", None
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            member = DEFAULT_ARCHIVE_MEMBER
            if member not in zf.namelist():
                raise SnervSnarHeaderGrammarProfileError(
                    f"archive missing {DEFAULT_ARCHIVE_MEMBER!r}"
                )
            return zf.read(member), "archive_zip_member", member
    raise SnervSnarHeaderGrammarProfileError(
        "input must be a raw SNAR1 packet or archive.zip containing 0.bin"
    )


def _json_len(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _item_count(value: Any) -> int | None:
    if isinstance(value, Mapping | list | tuple):
        return len(value)
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


__all__ = [
    "SCHEMA",
    "SnervSnarHeaderGrammarProfileError",
    "build_snerv_snar_header_grammar_profile",
]
