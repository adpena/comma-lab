#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove standalone runtime-style consumption of one HNGP archive member.

This is a non-score proof tool. It does not import or run the scorer, does not
dispatch GPU work, and does not touch provider or OMX state. The proof reads a
deterministic single-member archive produced by
``tools/build_hnerv_generated_schema_candidate.py``, parses the HNGP member
with ``tac.hnerv_generated_schema_packet`` as the oracle, independently parses
the same HNGP bytes with a minimal runtime-style checker, and compares the
section names, offsets, lengths, and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_generated_schema_packet import (  # noqa: E402
    HNeRVGeneratedSchemaPacketError,
    parse_hnerv_generated_schema_packet,
)

SCHEMA = "tac_hnerv_generated_schema_runtime_packet_proof_v1"
PROOF_FAMILY = "tac_runtime_consumption_proof_v1"
HNGP_MAGIC = b"HNGP"
HNGP_VERSION = 1
HNGP_HEADER_SCHEMA = "tac_hnerv_generated_schema_packet_header.v1"
HNGP_SECTION_ORDER = ("hngs_decoder", "latent_blob", "sidecar_blob")
HNGS_DECODER_MAGIC = b"HNGS"
PACKET_PREAMBLE = struct.Struct("<4sBI")
ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
ZIP_LOCAL_HEADER_SIGNATURE = 0x04034B50
EXPECTED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
EXPECTED_EXTERNAL_ATTR = 0o100644 << 16


class HNeRVGeneratedSchemaRuntimeProofError(ValueError):
    """Raised when a runtime-style HNGP parse fails closed."""


def build_hnerv_generated_schema_runtime_packet_proof(
    *,
    candidate_archive: Path,
    command_text: str | None = None,
) -> dict[str, Any]:
    """Return a non-score runtime proof JSON object for one HNGP archive."""

    candidate_archive = Path(candidate_archive)
    archive_bytes = candidate_archive.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    blockers: list[str] = []
    transcript: list[str] = [
        f"archive_path={candidate_archive}",
        f"archive_bytes={len(archive_bytes)}",
        f"archive_sha256={archive_sha256}",
    ]
    archive_info: dict[str, Any] = {
        "path": str(candidate_archive),
        "bytes": len(archive_bytes),
        "sha256": archive_sha256,
    }

    member_name = ""
    member_bytes = b""
    member_sha256 = ""
    try:
        extracted = _extract_single_hngp_member(
            candidate_archive=candidate_archive,
            archive_bytes=archive_bytes,
        )
        archive_info.update(extracted.archive_info)
        member_name = extracted.member_name
        member_bytes = extracted.member_bytes
        member_sha256 = _sha256_bytes(member_bytes)
        transcript.extend(
            [
                f"member_name={member_name}",
                f"member_bytes={len(member_bytes)}",
                f"member_sha256={member_sha256}",
            ]
        )
        blockers.extend(extracted.blockers)
    except (HNeRVGeneratedSchemaRuntimeProofError, zipfile.BadZipFile) as exc:
        blockers.append(f"archive_extract_failed:{exc}")
        transcript.append(f"archive_extract_failed={exc}")

    oracle_manifest: dict[str, Any] | None = None
    oracle_sections: list[dict[str, Any]] = []
    if member_bytes:
        try:
            oracle_packet = parse_hnerv_generated_schema_packet(member_bytes)
            oracle_manifest = oracle_packet.manifest
            oracle_sections = [
                _section_compare_row(section)
                for section in oracle_manifest.get("sections", [])
            ]
            transcript.append("oracle_parse=ok")
        except HNeRVGeneratedSchemaPacketError as exc:
            blockers.append(f"oracle_parse_failed:{exc}")
            transcript.append(f"oracle_parse_failed={exc}")
    else:
        blockers.append("member_bytes_missing")
        transcript.append("member_bytes_missing=true")

    runtime_manifest: dict[str, Any] | None = None
    runtime_sections: list[dict[str, Any]] = []
    if member_bytes:
        try:
            runtime_manifest = _standalone_runtime_parse_hngp(member_bytes)
            runtime_sections = [
                _section_compare_row(section)
                for section in runtime_manifest["sections"]
            ]
            transcript.append("standalone_runtime_parse=ok")
        except HNeRVGeneratedSchemaRuntimeProofError as exc:
            blockers.append(f"standalone_runtime_parse_failed:{exc}")
            transcript.append(f"standalone_runtime_parse_failed={exc}")

    comparisons = _compare_sections(
        oracle_sections=oracle_sections,
        runtime_sections=runtime_sections,
    )
    blockers.extend(comparisons["blockers"])
    transcript.extend(comparisons["transcript"])
    consumed_sections = comparisons["consumed_sections"]
    changed_sections = [
        {
            "section_name": section["section_name"],
            "new_sha256": section["sha256"],
            "runtime_consumed": section["runtime_consumed"],
            "offset": section["offset"],
            "length": section["length"],
        }
        for section in consumed_sections
        if section["section_name"] in HNGP_SECTION_ORDER
    ]

    command = command_text or (
        "tools/prove_hnerv_generated_schema_runtime_packet.py "
        f"--candidate-archive {candidate_archive}"
    )
    transcript_text = dumps_transcript(transcript)
    ready = not blockers
    return {
        "schema": SCHEMA,
        "proof_family": PROOF_FAMILY,
        "candidate_archive_path": str(candidate_archive),
        "candidate_archive_bytes": len(archive_bytes),
        "candidate_archive_sha256": archive_sha256,
        "archive": archive_info,
        "member_name": member_name,
        "member_bytes": len(member_bytes) if member_bytes else 0,
        "member_sha256": member_sha256,
        "new_member_sha256": member_sha256,
        "oracle_manifest_schema": (oracle_manifest or {}).get("schema", ""),
        "oracle_packet_sha256": (oracle_manifest or {}).get("packet_sha256", ""),
        "standalone_packet_sha256": (runtime_manifest or {}).get("packet_sha256", ""),
        "section_names_match": comparisons["section_names_match"],
        "section_offsets_match": comparisons["section_offsets_match"],
        "section_lengths_match": comparisons["section_lengths_match"],
        "section_sha256s_match": comparisons["section_sha256s_match"],
        "consumed_sections": consumed_sections,
        "changed_sections": changed_sections,
        "command_sha256": _sha256_text(command),
        "command_source": "inline" if command_text is not None else "generated",
        "proof_transcript": transcript,
        "proof_transcript_sha256": _sha256_text(transcript_text),
        "ready_for_exact_eval_runtime": ready,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "dispatch_blockers": [
            "non_score_hngp_runtime_proof_only",
            "contest_cuda_auth_eval_not_run",
            "lane_dispatch_claim_required_before_gpu",
        ],
        "promotion_blockers": [
            "non_score_generated_schema_runtime_proof",
            "contest_cuda_auth_eval_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "omx_state_touched": False,
    }


def dumps_json(payload: Mapping[str, Any]) -> str:
    """Return stable pretty JSON for the proof payload."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def dumps_transcript(transcript: list[str]) -> str:
    """Return stable transcript text hashed into the proof."""

    return "\n".join(transcript) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--command-text")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    try:
        payload = build_hnerv_generated_schema_runtime_packet_proof(
            candidate_archive=args.candidate_archive,
            command_text=args.command_text,
        )
    except OSError as exc:
        raise SystemExit(f"HNGP runtime proof failed: {exc}") from None

    text = dumps_json(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_not_ready and payload["ready_for_exact_eval_runtime"] is not True:
        return 1
    return 0


class _ExtractedMember:
    def __init__(
        self,
        *,
        archive_info: dict[str, Any],
        member_name: str,
        member_bytes: bytes,
        blockers: list[str],
    ) -> None:
        self.archive_info = archive_info
        self.member_name = member_name
        self.member_bytes = member_bytes
        self.blockers = blockers


def _extract_single_hngp_member(
    *,
    candidate_archive: Path,
    archive_bytes: bytes,
) -> _ExtractedMember:
    blockers: list[str] = []
    with zipfile.ZipFile(candidate_archive) as zf:
        infos = zf.infolist()
        archive_info: dict[str, Any] = {
            "member_count": len(infos),
            "zip_comment_sha256": _sha256_bytes(zf.comment),
            "zip_comment_bytes": len(zf.comment),
        }
        if zf.comment:
            blockers.append("zip_comment_present")
        if len(infos) != 1:
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"expected one ZIP member, got {len(infos)}"
            )
        info = infos[0]
        _validate_member_name(info.filename)
        member_bytes = zf.read(info.filename)

    member_sha256 = _sha256_bytes(member_bytes)
    local = _inspect_local_zip_header(
        archive_bytes=archive_bytes,
        header_offset=info.header_offset,
    )
    local_blockers = _zip_member_blockers(info=info, local=local)
    blockers.extend(local_blockers)
    archive_info.update(
        {
            "member_name": info.filename,
            "member_bytes": info.file_size,
            "member_sha256": member_sha256,
            "compression": "ZIP_STORED"
            if info.compress_type == zipfile.ZIP_STORED
            else str(info.compress_type),
            "crc32": f"{info.CRC:08x}",
            "date_time": list(info.date_time),
            "create_system": info.create_system,
            "external_attr": info.external_attr,
            "flag_bits": info.flag_bits,
            "local_header": local,
        }
    )
    if info.file_size != len(member_bytes):
        blockers.append("zip_member_file_size_mismatch")
    if member_sha256 != _sha256_bytes(member_bytes):
        blockers.append("zip_member_sha256_unstable")
    return _ExtractedMember(
        archive_info=archive_info,
        member_name=info.filename,
        member_bytes=member_bytes,
        blockers=blockers,
    )


def _validate_member_name(member_name: str) -> None:
    if not member_name:
        raise HNeRVGeneratedSchemaRuntimeProofError("ZIP member name is empty")
    if not member_name.endswith(".hngp"):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"ZIP member must end with .hngp: {member_name!r}"
        )
    if member_name.startswith(("/", "\\")) or "\\" in member_name:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"unsafe ZIP member name: {member_name!r}"
        )
    parts = member_name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"unsafe ZIP member name: {member_name!r}"
        )
    if any(part.startswith(".") for part in parts):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"hidden ZIP member name rejected: {member_name!r}"
        )


def _inspect_local_zip_header(
    *,
    archive_bytes: bytes,
    header_offset: int,
) -> dict[str, Any]:
    if header_offset < 0 or header_offset + ZIP_LOCAL_HEADER.size > len(archive_bytes):
        raise HNeRVGeneratedSchemaRuntimeProofError("local ZIP header offset is invalid")
    (
        signature,
        version_needed,
        flag_bits,
        compression,
        mod_time,
        mod_date,
        crc32,
        compressed_size,
        uncompressed_size,
        filename_len,
        extra_len,
    ) = ZIP_LOCAL_HEADER.unpack_from(archive_bytes, header_offset)
    if signature != ZIP_LOCAL_HEADER_SIGNATURE:
        raise HNeRVGeneratedSchemaRuntimeProofError("bad local ZIP header signature")
    name_start = header_offset + ZIP_LOCAL_HEADER.size
    name_end = name_start + filename_len
    extra_end = name_end + extra_len
    if extra_end > len(archive_bytes):
        raise HNeRVGeneratedSchemaRuntimeProofError("truncated local ZIP header")
    filename = archive_bytes[name_start:name_end].decode("utf-8")
    return {
        "header_offset": header_offset,
        "version_needed": version_needed,
        "flag_bits": flag_bits,
        "compression": compression,
        "mod_time": mod_time,
        "mod_date": mod_date,
        "crc32": f"{crc32:08x}",
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "filename": filename,
        "filename_len": filename_len,
        "extra_len": extra_len,
        "data_offset": extra_end,
    }


def _zip_member_blockers(info: zipfile.ZipInfo, local: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if info.compress_type != zipfile.ZIP_STORED:
        blockers.append("zip_member_not_stored")
    if info.date_time != EXPECTED_ZIP_DATE_TIME:
        blockers.append("zip_member_non_deterministic_timestamp")
    if info.create_system != 3:
        blockers.append("zip_member_create_system_not_unix")
    if info.external_attr != EXPECTED_EXTERNAL_ATTR:
        blockers.append("zip_member_external_attr_mismatch")
    if info.flag_bits != 0:
        blockers.append("zip_member_flag_bits_nonzero")
    if info.comment:
        blockers.append("zip_member_comment_present")
    if info.extra:
        blockers.append("zip_member_extra_present")
    if local.get("filename") != info.filename:
        blockers.append("zip_local_central_filename_mismatch")
    if local.get("compression") != info.compress_type:
        blockers.append("zip_local_central_compression_mismatch")
    if local.get("crc32") != f"{info.CRC:08x}":
        blockers.append("zip_local_central_crc32_mismatch")
    if local.get("compressed_size") != info.compress_size:
        blockers.append("zip_local_central_compressed_size_mismatch")
    if local.get("uncompressed_size") != info.file_size:
        blockers.append("zip_local_central_uncompressed_size_mismatch")
    if local.get("extra_len") != 0:
        blockers.append("zip_local_extra_present")
    return blockers


def _standalone_runtime_parse_hngp(packet: bytes) -> dict[str, Any]:
    if len(packet) < PACKET_PREAMBLE.size:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP packet too short for preamble"
        )
    magic, version, header_len = PACKET_PREAMBLE.unpack_from(packet, 0)
    if magic != HNGP_MAGIC:
        raise HNeRVGeneratedSchemaRuntimeProofError(f"bad HNGP magic: {magic!r}")
    if int(version) != HNGP_VERSION:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"unsupported HNGP version: {int(version)}"
        )
    if int(header_len) <= 0:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP header length must be positive"
        )
    header_end = PACKET_PREAMBLE.size + int(header_len)
    if header_end > len(packet):
        raise HNeRVGeneratedSchemaRuntimeProofError("truncated HNGP JSON header")
    header_bytes = packet[PACKET_PREAMBLE.size : header_end]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "invalid HNGP JSON header"
        ) from exc
    if not isinstance(header, dict):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP JSON header must be an object"
        )
    rows = _validate_runtime_header(header)
    cursor = header_end
    sections: list[dict[str, Any]] = [
        {
            "name": "header",
            "offset": 0,
            "length": header_end,
            "sha256": _sha256_bytes(packet[:header_end]),
        }
    ]
    for row in rows:
        name = row["name"]
        section_len = row["len"]
        section_end = cursor + section_len
        if section_end > len(packet):
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"truncated HNGP section {name!r}: "
                f"needs end offset {section_end}, packet has {len(packet)} bytes"
            )
        section = packet[cursor:section_end]
        section_sha = _sha256_bytes(section)
        if section_sha != row["sha256"]:
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"HNGP section {name!r} sha256 mismatch: "
                f"{section_sha} != {row['sha256']}"
            )
        sections.append(
            {
                "name": name,
                "offset": cursor,
                "length": section_len,
                "sha256": section_sha,
            }
        )
        cursor = section_end
    if cursor != len(packet):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"trailing bytes after HNGP sections: {len(packet) - cursor}"
        )
    hngs_decoder = packet[sections[1]["offset"] : sections[1]["offset"] + sections[1]["length"]]
    if not hngs_decoder.startswith(HNGS_DECODER_MAGIC):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "hngs_decoder section does not start with HNGS magic"
        )
    return {
        "packet_grammar": "hngp_v1",
        "packet_sha256": _sha256_bytes(packet),
        "packet_bytes": len(packet),
        "header": header,
        "sections": sections,
    }


def _validate_runtime_header(header: Mapping[str, Any]) -> list[dict[str, Any]]:
    if header.get("schema") != HNGP_HEADER_SCHEMA:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"unsupported HNGP header schema: {header.get('schema')!r}"
        )
    if header.get("magic") != HNGP_MAGIC.decode("ascii"):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"bad HNGP header magic: {header.get('magic')!r}"
        )
    if header.get("version") != HNGP_VERSION:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"bad HNGP header version: {header.get('version')!r}"
        )
    if header.get("packet_grammar") != "hngp_v1":
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"unsupported HNGP packet grammar: {header.get('packet_grammar')!r}"
        )
    if header.get("monolithic_packet") is not True:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP header must set monolithic_packet=true"
        )
    if header.get("score_claim") is not False:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP header must set score_claim=false"
        )
    if header.get("promotion_eligible") is not False:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP header must set promotion_eligible=false"
        )
    if header.get("ready_for_exact_eval_dispatch") is not False:
        raise HNeRVGeneratedSchemaRuntimeProofError(
            "HNGP header must set ready_for_exact_eval_dispatch=false"
        )
    if tuple(header.get("section_order") or ()) != HNGP_SECTION_ORDER:
        raise HNeRVGeneratedSchemaRuntimeProofError("HNGP section_order mismatch")
    rows = header.get("sections")
    if not isinstance(rows, list):
        raise HNeRVGeneratedSchemaRuntimeProofError("HNGP sections must be a list")
    if len(rows) != len(HNGP_SECTION_ORDER):
        raise HNeRVGeneratedSchemaRuntimeProofError(
            f"HNGP expected {len(HNGP_SECTION_ORDER)} sections, got {len(rows)}"
        )
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise HNeRVGeneratedSchemaRuntimeProofError(
                "HNGP section row must be an object"
            )
        name = row.get("name")
        if not isinstance(name, str):
            raise HNeRVGeneratedSchemaRuntimeProofError(
                "HNGP section name must be a string"
            )
        if name in seen:
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"duplicate HNGP section {name!r}"
            )
        expected = HNGP_SECTION_ORDER[index]
        if name != expected:
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"HNGP section order mismatch at index {index}: {name!r} != {expected!r}"
            )
        seen.add(name)
        section_len = row.get("len")
        if isinstance(section_len, bool) or not isinstance(section_len, int) or section_len < 0:
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"HNGP section {name!r} has invalid length {section_len!r}"
            )
        sha256 = row.get("sha256")
        if not _is_sha256(sha256):
            raise HNeRVGeneratedSchemaRuntimeProofError(
                f"HNGP section {name!r} has invalid sha256 {sha256!r}"
            )
        validated.append({"name": name, "len": section_len, "sha256": sha256})
    return validated


def _section_compare_row(section: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "section_name": str(section.get("name", "")),
        "offset": int(section.get("offset", -1)),
        "length": int(section.get("length", section.get("len", -1))),
        "sha256": str(section.get("sha256", "")),
    }


def _compare_sections(
    *,
    oracle_sections: list[dict[str, Any]],
    runtime_sections: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    transcript: list[str] = []
    oracle_by_name = {section["section_name"]: section for section in oracle_sections}
    runtime_by_name = {section["section_name"]: section for section in runtime_sections}
    names_match = [row["section_name"] for row in oracle_sections] == [
        row["section_name"] for row in runtime_sections
    ]
    offsets_match = _field_sequence(oracle_sections, "offset") == _field_sequence(
        runtime_sections, "offset"
    )
    lengths_match = _field_sequence(oracle_sections, "length") == _field_sequence(
        runtime_sections, "length"
    )
    sha256s_match = _field_sequence(oracle_sections, "sha256") == _field_sequence(
        runtime_sections, "sha256"
    )
    if oracle_sections and runtime_sections:
        if not names_match:
            blockers.append("section_names_mismatch")
        if not offsets_match:
            blockers.append("section_offsets_mismatch")
        if not lengths_match:
            blockers.append("section_lengths_mismatch")
        if not sha256s_match:
            blockers.append("section_sha256s_mismatch")
    elif not oracle_sections:
        blockers.append("oracle_sections_missing")
    elif not runtime_sections:
        blockers.append("standalone_runtime_sections_missing")

    consumed_sections: list[dict[str, Any]] = []
    for name in ["header", *HNGP_SECTION_ORDER]:
        oracle = oracle_by_name.get(name)
        runtime = runtime_by_name.get(name)
        if oracle is None or runtime is None:
            continue
        row = {
            "section_name": name,
            "offset": runtime["offset"],
            "length": runtime["length"],
            "sha256": runtime["sha256"],
            "oracle_sha256": oracle["sha256"],
            "standalone_sha256": runtime["sha256"],
            "runtime_consumed": oracle == runtime,
        }
        consumed_sections.append(row)
        transcript.append(
            "section "
            f"{name} offset={row['offset']} length={row['length']} "
            f"sha256={row['sha256']} runtime_consumed={row['runtime_consumed']}"
        )
    if any(section["runtime_consumed"] is not True for section in consumed_sections):
        blockers.append("consumed_section_comparison_failed")
    return {
        "section_names_match": names_match,
        "section_offsets_match": offsets_match,
        "section_lengths_match": lengths_match,
        "section_sha256s_match": sha256s_match,
        "consumed_sections": consumed_sections,
        "blockers": blockers,
        "transcript": transcript,
    }


def _field_sequence(rows: list[dict[str, Any]], field: str) -> list[Any]:
    return [row[field] for row in rows]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
