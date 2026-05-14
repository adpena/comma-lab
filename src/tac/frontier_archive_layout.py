# SPDX-License-Identifier: MIT
"""Physical and logical layout inspection for public frontier archives.

The medal-band HNeRV submissions are generally single-member ZIP packets. That
fact invalidates ZIP-member-level component budgets, but it does not by itself
prove that masks, poses, latents, sidecars, or renderer bytes are absent. Those
claims require an internal grammar parser with offsets, lengths, and hashes.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_generated_schema_packet import (
    HNGP_MAGIC,
    HNeRVGeneratedSchemaPacketError,
    inspect_hnerv_generated_schema_packet,
)

PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_INNER_MEMBER_NAME = "x"

PR106_INNER_MEMBER_NAME = "0.bin"
PR106_HEADER_LEN = 4
PR106_HEADER_MAGIC = 0xFF

A2K1_MAGIC = b"A2K1"
A2K1_HEADER_LEN = 8

CPLX1_MAGIC = b"CPLX"
CPLX1_HEADER_LEN = 8
CPLX1_BYTE_MAP_LEN_FIELD_LEN = 2


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _section(name: str, *, offset: int, data: bytes, role: str) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "offset": int(offset),
        "len": len(data),
        "sha256": _sha256_bytes(data),
    }


def _inspect_pr101_inner(member_name: str, blob: bytes) -> dict[str, Any] | None:
    if member_name != PR101_INNER_MEMBER_NAME:
        return None
    minimum = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    if len(blob) < minimum:
        return None
    decoder = blob[:PR101_DECODER_BLOB_LEN]
    latent = blob[PR101_DECODER_BLOB_LEN:minimum]
    sidecar = blob[minimum:]
    return {
        "grammar": "pr101_fixed_offset_hnerv_microcodec",
        "parser_confidence": "exact_member_name_and_fixed_offsets",
        "single_member_name": member_name,
        "member_bytes": len(blob),
        "sections": [
            _section("decoder_blob", offset=0, data=decoder, role="renderer_decoder_weights"),
            _section(
                "latent_blob",
                offset=PR101_DECODER_BLOB_LEN,
                data=latent,
                role="latent_motion_or_frame_conditioning",
            ),
            _section(
                "sidecar_blob",
                offset=minimum,
                data=sidecar,
                role="latent_sidecar_not_separate_pose_or_mask_member",
            ),
        ],
        "component_budget_implication": (
            "no_zip_member_mask_or_pose_budget; logical edits must target decoder_blob, "
            "latent_blob, or sidecar_blob under the PR101 parser"
        ),
    }


def _inspect_hngp_inner(member_name: str, blob: bytes) -> dict[str, Any] | None:
    if not blob.startswith(HNGP_MAGIC):
        return None
    try:
        packet = inspect_hnerv_generated_schema_packet(blob)
    except HNeRVGeneratedSchemaPacketError:
        return None
    return {
        "grammar": "hngp_v1",
        "parser_confidence": "hngp_magic_header_and_section_sha256",
        "parser_proof_strength": "canonical_hngp_parse",
        "parser_ambiguous": False,
        "parser_alternatives": ["hngp_v1"],
        "single_member_name": member_name,
        "member_bytes": len(blob),
        "packet_sha256": packet["packet_sha256"],
        "sections": [
            {
                "name": section["name"],
                "role": section["role"],
                "offset": int(section["offset"]),
                "len": int(section["len"]),
                "sha256": section["sha256"],
            }
            for section in packet["sections"]
        ],
        "component_budget_implication": (
            "generated-schema HNGP packet; logical edits must target hngs_decoder, "
            "latent_blob, or sidecar_blob under the HNGP parser"
        ),
    }


def _inspect_a2k1_inner(member_name: str, blob: bytes) -> dict[str, Any] | None:
    if not blob.startswith(A2K1_MAGIC):
        return None
    parsed = _parse_a2k1_blob(blob)
    if parsed is None:
        return None
    decoder_len = parsed["decoder_len"]
    return {
        "grammar": "a2k1_variable_decoder_pr101",
        "parser_confidence": "a2k1_magic_and_u32_decoder_len",
        "parser_proof_strength": "magic_len_and_pr101_latent_window",
        "parser_ambiguous": False,
        "parser_alternatives": ["a2k1_variable_decoder_pr101"],
        "single_member_name": member_name,
        "member_bytes": len(blob),
        "wire_format": "A2K1",
        "magic": A2K1_MAGIC.decode("ascii"),
        "decoder_len_field": decoder_len,
        "sections": [
            _section("a2k1_magic", offset=0, data=blob[:4], role="wire_format_magic"),
            _section("decoder_len_u32le", offset=4, data=blob[4:8], role="internal_length_header"),
            _section(
                "decoder_blob",
                offset=parsed["decoder_start"],
                data=parsed["decoder_blob"],
                role="renderer_decoder_weights",
            ),
            _section(
                "latent_blob",
                offset=parsed["latent_start"],
                data=parsed["latent_blob"],
                role="latent_motion_or_frame_conditioning",
            ),
            _section(
                "sidecar_blob",
                offset=parsed["sidecar_start"],
                data=parsed["sidecar_blob"],
                role="latent_sidecar_not_separate_pose_or_mask_member",
            ),
        ],
        "component_budget_implication": (
            "A2K1 byte-closed packet; logical edits must target decoder_blob, "
            "latent_blob, or sidecar_blob under the A2K1 parser"
        ),
    }


def _inspect_cplx1_inner(member_name: str, blob: bytes) -> dict[str, Any] | None:
    if not blob.startswith(CPLX1_MAGIC):
        return None
    parsed = _parse_cplx1_blob(blob)
    if parsed is None:
        return None
    return {
        "grammar": "cplx1_op1_byte_maps",
        "parser_confidence": "cplx_magic_section_len_and_byte_maps_json",
        "parser_proof_strength": "magic_len_json_and_pr101_latent_window",
        "parser_ambiguous": False,
        "parser_alternatives": ["cplx1_op1_byte_maps"],
        "single_member_name": member_name,
        "member_bytes": len(blob),
        "wire_format": "CPLX1",
        "magic": CPLX1_MAGIC.decode("ascii"),
        "decoder_section_total_field": parsed["section_total"],
        "byte_maps_json_len_field": parsed["byte_maps_json_len"],
        "byte_maps_entry_count": parsed["byte_maps_entry_count"],
        "sections": [
            _section("cplx_magic", offset=0, data=blob[:4], role="wire_format_magic"),
            _section(
                "decoder_section_len_u32le",
                offset=4,
                data=blob[4:8],
                role="internal_length_header",
            ),
            _section(
                "byte_maps_json_len_u16le",
                offset=8,
                data=blob[8:10],
                role="internal_length_header",
            ),
            _section(
                "byte_maps_json",
                offset=parsed["byte_maps_json_start"],
                data=parsed["byte_maps_json"],
                role="decoder_byte_map_metadata",
            ),
            _section(
                "op1_inner_blob",
                offset=parsed["op1_inner_start"],
                data=parsed["op1_inner_blob"],
                role="renderer_decoder_weights",
            ),
            _section(
                "latent_blob",
                offset=parsed["latent_start"],
                data=parsed["latent_blob"],
                role="latent_motion_or_frame_conditioning",
            ),
            _section(
                "sidecar_blob",
                offset=parsed["sidecar_start"],
                data=parsed["sidecar_blob"],
                role="latent_sidecar_not_separate_pose_or_mask_member",
            ),
        ],
        "component_budget_implication": (
            "CPLX1 byte-closed packet; logical edits must target byte_maps_json, "
            "op1_inner_blob, latent_blob, or sidecar_blob under the CPLX1 parser"
        ),
    }


def _inspect_pr106_inner(member_name: str, blob: bytes) -> dict[str, Any] | None:
    if len(blob) < PR106_HEADER_LEN or blob[0] != PR106_HEADER_MAGIC:
        return None
    decoder_len = int.from_bytes(blob[1:4], "little")
    decoder_start = PR106_HEADER_LEN
    decoder_end = decoder_start + decoder_len
    if decoder_len <= 0 or decoder_end > len(blob):
        return None
    header = blob[:PR106_HEADER_LEN]
    decoder = blob[decoder_start:decoder_end]
    tail = blob[decoder_end:]
    decoder_valid = _brotli_decompresses(decoder)
    tail_valid = _brotli_decompresses(tail)
    if not (decoder_valid and tail_valid):
        return None
    return {
        "grammar": "pr106_ff_packed_hnerv",
        "parser_confidence": "magic_and_24bit_decoder_len",
        "parser_proof_strength": "magic_len_and_brotli_streams",
        "validated_streams": {
            "decoder_packed_brotli": decoder_valid,
            "latents_and_sidecar_brotli": tail_valid,
        },
        "single_member_name": member_name,
        "member_bytes": len(blob),
        "decoder_len_field": decoder_len,
        "sections": [
            _section("ff_header", offset=0, data=header, role="internal_length_header"),
            _section(
                "decoder_packed_brotli",
                offset=decoder_start,
                data=decoder,
                role="renderer_decoder_weights",
            ),
            _section(
                "latents_and_sidecar_brotli",
                offset=decoder_end,
                data=tail,
                role="latent_sidecar_not_separate_pose_or_mask_member",
            ),
        ],
        "component_budget_implication": (
            "no_zip_member_mask_or_pose_budget; logical edits must target "
            "decoder_packed_brotli or latents_and_sidecar_brotli under the PR106 parser"
        ),
    }


def _pr106_header_like_failure(blob: bytes) -> str | None:
    """Return a fail-closed reason for PR106-looking bytes that do not parse.

    PR106 packets carry an explicit 0xff marker plus a 24-bit decoder-stream
    length. If that contract is present but the declared Brotli streams do not
    validate, the bytes are not safe to reinterpret as a fixed-offset PR101
    packet. That fallback would give parser authority to malformed PR106-like
    bytes.
    """

    if len(blob) < PR106_HEADER_LEN or blob[0] != PR106_HEADER_MAGIC:
        return None
    decoder_len = int.from_bytes(blob[1:4], "little")
    decoder_start = PR106_HEADER_LEN
    decoder_end = decoder_start + decoder_len
    if decoder_len <= 0:
        return "PR106-like 0xff header has non-positive decoder length; logical parser fails closed."
    if decoder_end > len(blob):
        return "PR106-like 0xff header decoder length exceeds payload; logical parser fails closed."
    decoder = blob[decoder_start:decoder_end]
    tail = blob[decoder_end:]
    decoder_valid = _brotli_decompresses(decoder)
    tail_valid = _brotli_decompresses(tail)
    if not (decoder_valid and tail_valid):
        return (
            "PR106-like 0xff header present but declared Brotli streams did not both decode; "
            "fallback fixed-offset parsers are suppressed as ambiguous/unvalidated."
        )
    return None


def _brotli_decompresses(data: bytes) -> bool:
    if not data:
        return False
    try:
        brotli.decompress(data)
    except brotli.error:
        return False
    return True


def _parse_a2k1_blob(blob: bytes) -> dict[str, Any] | None:
    if len(blob) < A2K1_HEADER_LEN + PR101_LATENT_BLOB_LEN:
        return None
    if blob[:4] != A2K1_MAGIC:
        return None
    decoder_len = int.from_bytes(blob[4:8], "little")
    if decoder_len <= 0:
        return None
    decoder_start = A2K1_HEADER_LEN
    decoder_end = decoder_start + decoder_len
    latent_start = decoder_end
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    if latent_end >= len(blob):
        return None
    return {
        "decoder_len": decoder_len,
        "decoder_start": decoder_start,
        "decoder_blob": blob[decoder_start:decoder_end],
        "latent_start": latent_start,
        "latent_blob": blob[latent_start:latent_end],
        "sidecar_start": latent_end,
        "sidecar_blob": blob[latent_end:],
    }


def _parse_cplx1_blob(blob: bytes) -> dict[str, Any] | None:
    minimum = CPLX1_HEADER_LEN + CPLX1_BYTE_MAP_LEN_FIELD_LEN + PR101_LATENT_BLOB_LEN
    if len(blob) < minimum:
        return None
    if blob[:4] != CPLX1_MAGIC:
        return None
    section_total = int.from_bytes(blob[4:8], "little")
    if section_total < CPLX1_HEADER_LEN + CPLX1_BYTE_MAP_LEN_FIELD_LEN:
        return None
    if section_total > len(blob) - PR101_LATENT_BLOB_LEN:
        return None
    byte_maps_json_len = int.from_bytes(blob[8:10], "little")
    byte_maps_json_start = 10
    byte_maps_json_end = byte_maps_json_start + byte_maps_json_len
    if byte_maps_json_end > section_total:
        return None
    op1_inner_start = byte_maps_json_end
    if op1_inner_start >= section_total:
        return None
    byte_maps_json = blob[byte_maps_json_start:byte_maps_json_end]
    try:
        byte_maps = json.loads(byte_maps_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(byte_maps, dict):
        return None
    try:
        {int(key): str(value) for key, value in byte_maps.items()}
    except (TypeError, ValueError):
        return None
    latent_start = section_total
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    if latent_end >= len(blob):
        return None
    return {
        "section_total": section_total,
        "byte_maps_json_len": byte_maps_json_len,
        "byte_maps_entry_count": len(byte_maps),
        "byte_maps_json_start": byte_maps_json_start,
        "byte_maps_json": byte_maps_json,
        "op1_inner_start": op1_inner_start,
        "op1_inner_blob": blob[op1_inner_start:section_total],
        "latent_start": latent_start,
        "latent_blob": blob[latent_start:latent_end],
        "sidecar_start": latent_end,
        "sidecar_blob": blob[latent_end:],
    }


def _resolve_logical_layout(member_name: str, blob: bytes) -> tuple[dict[str, Any] | None, list[str]]:
    pr106_failure = _pr106_header_like_failure(blob)
    if pr106_failure is not None:
        return None, [pr106_failure]
    candidates = [
        candidate
        for candidate in (
            _inspect_hngp_inner(member_name, blob),
            _inspect_a2k1_inner(member_name, blob),
            _inspect_cplx1_inner(member_name, blob),
            _inspect_pr106_inner(member_name, blob),
            _inspect_pr101_inner(member_name, blob),
        )
        if candidate is not None
    ]
    if not candidates:
        return None, ["No known internal grammar was proven; logical stream budgets are unavailable."]
    if len(candidates) == 1:
        candidate = candidates[0]
        candidate["parser_ambiguous"] = False
        candidate["parser_alternatives"] = [candidate["grammar"]]
        return candidate, []

    explicit_magic = next(
        (
            candidate
            for candidate in candidates
            if candidate["grammar"]
            in {"hngp_v1", "a2k1_variable_decoder_pr101", "cplx1_op1_byte_maps"}
        ),
        None,
    )
    if explicit_magic is not None:
        explicit_magic["parser_ambiguous"] = False
        explicit_magic["parser_alternatives"] = [str(candidate["grammar"]) for candidate in candidates]
        explicit_magic["ambiguity_resolution"] = (
            "explicit packet magic selected over fixed-offset fallback parser"
        )
        return explicit_magic, []

    pr106 = next((candidate for candidate in candidates if candidate["grammar"] == "pr106_ff_packed_hnerv"), None)
    if pr106 is not None and pr106.get("parser_proof_strength") == "magic_len_and_brotli_streams":
        pr106["parser_ambiguous"] = False
        pr106["parser_alternatives"] = [str(candidate["grammar"]) for candidate in candidates]
        pr106["ambiguity_resolution"] = (
            "PR106 selected over fixed-offset PR101 because both PR106 logical Brotli streams decode"
        )
        return pr106, []

    return None, [
        "Multiple internal grammars matched; logical layout is ambiguous and must fail closed.",
        "Resolve ambiguity with known source SHA/runtime adapter identity or validated PR106 stream decoding.",
    ]


def inspect_frontier_archive_layout(archive_path: Path) -> dict[str, Any]:
    """Return a deterministic no-score layout manifest for one archive ZIP."""
    archive_path = Path(archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)

    members: list[dict[str, Any]] = []
    member_payloads: list[tuple[str, bytes]] = []
    total_compressed = 0
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        for index, info in enumerate(zf.infolist()):
            with zf.open(info, "r") as f:
                blob = f.read()
            total_compressed += info.compress_size
            member_payloads.append((info.filename, blob))
            members.append(
                {
                    "index": index,
                    "name": info.filename,
                    "compress_type": info.compress_type,
                    "compress_size": info.compress_size,
                    "file_size": info.file_size,
                    "sha256": _sha256_bytes(blob),
                    "payload_share_of_archive": info.file_size / max(archive_bytes, 1),
                }
            )

    is_single_member = len(members) == 1
    monolithic = bool(is_single_member)
    zip_overhead = archive_bytes - total_compressed

    logical_layout = None
    logical_cautions: list[str] = []
    if is_single_member:
        member_name, blob = member_payloads[0]
        logical_layout, logical_cautions = _resolve_logical_layout(member_name, blob)

    cautions = [
        "ZIP-member layout is physical custody only; do not infer mask/pose budgets from member names.",
        "A single monolithic member falsifies separate ZIP-member budgets, not parser-proven logical sections.",
        "Score, promotion, and family-retirement claims remain forbidden without exact CUDA auth eval.",
    ]
    cautions.extend(logical_cautions)

    return {
        "schema": "tac_frontier_archive_layout_v1",
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "score_claim": False,
        "evidence_grade": "empirical_archive_layout_cpu_no_score",
        "physical_layout": {
            "n_members": len(members),
            "members": members,
            "zip_overhead_bytes": zip_overhead,
            "single_member_monolithic_packet": monolithic,
            "archive_member_level_component_budgets_valid": not monolithic,
            "member_level_mask_budget_valid": False if monolithic else None,
            "member_level_pose_budget_valid": False if monolithic else None,
        },
        "logical_layout": logical_layout,
        "cautions": cautions,
        "duplicate_member_names": duplicate_names,
    }


def render_frontier_archive_layout_summary(manifest: dict[str, Any]) -> str:
    """Render a short human-readable summary for operator logs."""
    physical = manifest["physical_layout"]
    lines = [
        "Frontier archive layout",
        f"archive: {manifest['archive_path']}",
        f"bytes:   {manifest['archive_bytes']}",
        f"sha256:  {manifest['archive_sha256']}",
        f"members: {physical['n_members']}",
        f"monolithic_single_member: {physical['single_member_monolithic_packet']}",
        f"member_level_component_budgets_valid: {physical['archive_member_level_component_budgets_valid']}",
    ]
    for member in physical["members"]:
        lines.append(
            f"  member[{member['index']}]: {member['name']} {member['file_size']} B "
            f"sha256={member['sha256']}"
        )
    logical = manifest.get("logical_layout")
    if logical:
        lines.append(f"logical grammar: {logical['grammar']}")
        for section in logical["sections"]:
            lines.append(
                f"  section: {section['name']} offset={section['offset']} "
                f"len={section['len']} sha256={section['sha256']}"
            )
        lines.append(f"implication: {logical['component_budget_implication']}")
    else:
        lines.append("logical grammar: unknown")
    lines.append("cautions:")
    for caution in manifest["cautions"]:
        lines.append(f"  - {caution}")
    return "\n".join(lines)


def dumps_manifest(payload: dict[str, Any]) -> str:
    """Return stable pretty JSON for layout reports."""
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
