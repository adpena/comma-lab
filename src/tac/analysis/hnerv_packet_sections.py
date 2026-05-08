"""Parser-section manifests for monolithic public HNeRV packets.

The public HNeRV-family frontier archives are usually one charged ZIP member.
This helper records the parser-proven sections inside that member: archive
identity, member identity, byte offsets, lengths, section names, and hashes.
It is custody/audit infrastructure only; it never emits a score claim.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
)
from tac.hnerv_pr103_lc_ac_schema import (
    PUBLIC_PR103_LAYOUT,
    HnervPr103LcAcSchemaError,
    parse_pr103_lc_ac_payload,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes

SCHEMA_VERSION = 1
TOOL_NAME = "tac.analysis.hnerv_packet_sections"
MANIFEST_SCHEMA = "tac_hnerv_packet_section_manifest.v1"
BATCH_SCHEMA = "tac_hnerv_packet_section_manifest_batch.v1"

PARSER_AUTO = "auto"
PARSER_PR101 = "pr101_microcodec_fixed"
PARSER_PR103 = "pr103_lc_ac"
PARSER_PR106 = "pr106_ff_packed_hnerv"
PARSER_CHOICES = (PARSER_AUTO, PARSER_PR101, PARSER_PR103, PARSER_PR106)

PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_TOTAL_KNOWN_LEN = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN + 607

PR101_SECTION_ROLES = {
    "decoder_compact_brotli_streams": "decoder_weight_stream",
    "latents_raw_lzma_delta_u8": "latent_stream",
    "sidecar_dim_delta_huffman_enum": "sidecar_or_correction_stream",
}
PR103_SECTION_ROLES = {
    "scales_fp16": "control_or_metadata",
    "non_ac_weights_brotli": "decoder_weight_stream",
    "ac_histograms_brotli": "entropy_model_or_range_stream",
    "merged_range_coded_weights_and_hi_latents": "entropy_model_or_range_stream",
    "latent_min_scale_fp16": "control_or_metadata",
    "latent_low_bytes_brotli": "latent_stream",
    "latent_hi_histogram_brotli": "entropy_model_or_range_stream",
    "sidecar_corrections_brotli": "sidecar_or_correction_stream",
}
PR106_SECTION_ROLES = {
    "packed_header_ff_len24": "control_or_metadata",
    "decoder_packed_brotli": "decoder_weight_stream",
    "latents_and_sidecar_brotli": "latent_stream",
}


class HnervPacketSectionManifestError(ValueError):
    """Raised when a packet-section manifest cannot be emitted or validated."""


def build_packet_section_manifest(
    archive_path: str | Path,
    *,
    label: str | None = None,
    parser: str = PARSER_AUTO,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Emit and validate one no-score parser-section manifest."""

    manifest = _emit_packet_section_manifest(
        archive_path,
        label=label,
        parser=parser,
        repo_root=repo_root,
    )
    blockers = validate_packet_section_manifest(manifest, repo_root=repo_root)
    manifest["parser_section_gate"] = _gate(blockers)
    return manifest


def build_packet_section_manifest_batch(
    archives: Iterable[tuple[str, str | Path, str]],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Emit a deterministic no-score batch manifest for labeled archives."""

    records = [
        build_packet_section_manifest(path, label=label, parser=parser, repo_root=repo_root)
        for label, path, parser in archives
    ]
    blockers: list[str] = []
    for record in records:
        label = str(record.get("label") or "unknown")
        for blocker in validate_packet_section_manifest(record, repo_root=repo_root):
            blockers.append(f"{label}:{blocker}")
    if not records:
        blockers.append("batch_missing_records")
    return {
        "schema": BATCH_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "parser_section_gate": _gate(blockers),
        "records": records,
    }


def validate_packet_section_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    recompute_archive: bool = True,
) -> list[str]:
    """Return fail-closed blockers for one parser-section manifest."""

    blockers = _validate_manifest_shape(manifest)
    if not recompute_archive:
        return blockers
    archive_path = _archive_path_from_manifest(manifest, repo_root=repo_root)
    if archive_path is None:
        return [*blockers, "archive_path_missing"]
    if not archive_path.is_file():
        return [*blockers, "archive_path_missing_on_disk"]
    parser_name = _parser_name_from_manifest(manifest)
    if parser_name is None:
        return [*blockers, "parser_name_missing"]
    try:
        rebuilt = _emit_packet_section_manifest(
            archive_path,
            label=str(manifest.get("label") or ""),
            parser=parser_name,
            repo_root=repo_root,
        )
    except (HnervPacketSectionManifestError, HnervLowlevelPackError, HnervPr103LcAcSchemaError) as exc:
        return [*blockers, f"archive_reparse_failed:{exc}"]
    blockers.extend(_compare_rebuilt_manifest(manifest, rebuilt))
    return blockers


def validate_packet_section_manifest_batch(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Return fail-closed blockers for a batch or single manifest payload."""

    if payload.get("schema") == MANIFEST_SCHEMA:
        return validate_packet_section_manifest(payload, repo_root=repo_root)
    blockers: list[str] = []
    if payload.get("schema") != BATCH_SCHEMA:
        blockers.append("batch_schema_mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("batch_schema_version_mismatch")
    if payload.get("score_claim") is not False:
        blockers.append("batch_score_claim_not_false")
    if payload.get("dispatch_attempted") is not False:
        blockers.append("batch_dispatch_attempted_not_false")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return [*blockers, "batch_records_missing"]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            blockers.append(f"record_not_object:{index}")
            continue
        label = str(record.get("label") or index)
        for blocker in validate_packet_section_manifest(record, repo_root=repo_root):
            blockers.append(f"{label}:{blocker}")
    return blockers


def render_manifest_summary(payload: Mapping[str, Any]) -> str:
    """Render a compact text summary for CLI output."""

    if payload.get("schema") == BATCH_SCHEMA:
        records = payload.get("records") if isinstance(payload.get("records"), list) else []
        lines = [
            "HNeRV packet-section manifest batch",
            f"score_claim: {payload.get('score_claim')}",
            f"parser_section_gate_ready: {_gate_ready(payload)}",
            f"records: {len(records)}",
        ]
        for record in records:
            if isinstance(record, Mapping):
                lines.append(_record_summary_line(record))
        return "\n".join(lines) + "\n"
    return "\n".join(
        [
            "HNeRV packet-section manifest",
            f"score_claim: {payload.get('score_claim')}",
            f"parser_section_gate_ready: {_gate_ready(payload)}",
            _record_summary_line(payload),
        ]
    ) + "\n"


def dumps_manifest(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON for a manifest or batch."""

    return json_text(payload)


def _emit_packet_section_manifest(
    archive_path: str | Path,
    *,
    label: str | None,
    parser: str,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    if parser not in PARSER_CHOICES:
        raise HnervPacketSectionManifestError(f"unknown parser {parser!r}")
    archive = Path(archive_path)
    single = read_strict_single_member_zip(archive)
    member_meta = _zip_member_metadata(archive)
    payload_sha256 = sha256_bytes(single.payload)
    parser_name = _infer_parser(
        parser,
        archive_path=archive,
        label=label or "",
        member_name=single.member_name,
        payload=single.payload,
    )
    sections = _parse_sections(parser_name, single.payload)
    coverage = _coverage_record(sections, payload_bytes=single.member_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "label": label or archive.stem,
        "archive": {
            "path": repo_relative(archive, repo_root or Path.cwd()),
            "bytes": single.archive_bytes,
            "sha256": single.archive_sha256,
        },
        "member": {
            "name": single.member_name,
            "bytes": single.member_bytes,
            "sha256": payload_sha256,
            "zip_compress_type": member_meta["compress_type"],
            "zip_compress_size": member_meta["compress_size"],
            "zip_crc": member_meta["crc"],
        },
        "parser": {
            "name": parser_name,
            "requested": parser,
            "confidence": _parser_confidence(parser_name),
        },
        "sections": sections,
        "coverage": coverage,
        "parser_section_gate": _gate(_validate_manifest_shape_without_gate(sections, coverage)),
        "notes": [
            "parser-section custody only",
            "no component score, score movement, or dispatch authorization is claimed",
        ],
    }
    return manifest


def _zip_member_metadata(archive: Path) -> dict[str, int]:
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise HnervPacketSectionManifestError(f"expected one ZIP member, got {len(infos)}")
        info = infos[0]
        return {
            "compress_type": int(info.compress_type),
            "compress_size": int(info.compress_size),
            "crc": int(info.CRC),
        }


def _infer_parser(
    requested: str,
    *,
    archive_path: Path,
    label: str,
    member_name: str,
    payload: bytes,
) -> str:
    if requested != PARSER_AUTO:
        return requested
    text = f"{label} {archive_path.as_posix()} {member_name}".lower()
    if "pr106" in text or "belt_and_suspenders" in text:
        return PARSER_PR106
    if "pr103" in text or "hnerv_lc_ac" in text:
        return PARSER_PR103
    if "pr101" in text or "hnerv_ft_microcodec" in text:
        return PARSER_PR101
    if len(payload) >= 4 and payload[0] == 0xFF:
        return PARSER_PR106
    if len(payload) == PR101_TOTAL_KNOWN_LEN:
        return PARSER_PR101
    if len(payload) >= PUBLIC_PR103_LAYOUT.fixed_bytes:
        return PARSER_PR103
    raise HnervPacketSectionManifestError("could not infer HNeRV packet parser")


def _parse_sections(parser_name: str, payload: bytes) -> list[dict[str, Any]]:
    if parser_name == PARSER_PR101:
        return _parse_pr101_sections(payload)
    if parser_name == PARSER_PR103:
        return _parse_pr103_sections(payload)
    if parser_name == PARSER_PR106:
        return _parse_pr106_sections(payload)
    raise HnervPacketSectionManifestError(f"unknown parser {parser_name!r}")


def _parse_pr101_sections(payload: bytes) -> list[dict[str, Any]]:
    minimum = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    if len(payload) < minimum:
        raise HnervPacketSectionManifestError(
            f"PR101 payload too short: expected at least {minimum} bytes, got {len(payload)}"
        )
    specs = [
        ("decoder_compact_brotli_streams", 0, PR101_DECODER_BLOB_LEN),
        ("latents_raw_lzma_delta_u8", PR101_DECODER_BLOB_LEN, minimum),
        ("sidecar_dim_delta_huffman_enum", minimum, len(payload)),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=start,
            data=payload[start:end],
            role=PR101_SECTION_ROLES[name],
        )
        for index, (name, start, end) in enumerate(specs)
    ]


def _parse_pr103_sections(payload: bytes) -> list[dict[str, Any]]:
    parsed = parse_pr103_lc_ac_payload(payload)
    records = []
    for index, section in enumerate(parsed.sections):
        records.append(
            _section_record(
                index,
                name=section.name,
                offset=section.start,
                data=section.data,
                role=PR103_SECTION_ROLES.get(section.name, "opaque_payload_stream"),
            )
        )
    return records


def _parse_pr106_sections(payload: bytes) -> list[dict[str, Any]]:
    packed = parse_ff_packed_brotli_hnerv(payload)
    decoder_offset = len(packed.header)
    tail_offset = decoder_offset + len(packed.decoder_packed_brotli)
    specs = [
        ("packed_header_ff_len24", 0, packed.header),
        ("decoder_packed_brotli", decoder_offset, packed.decoder_packed_brotli),
        ("latents_and_sidecar_brotli", tail_offset, packed.latents_and_sidecar_brotli),
    ]
    return [
        _section_record(
            index,
            name=name,
            offset=offset,
            data=data,
            role=PR106_SECTION_ROLES[name],
        )
        for index, (name, offset, data) in enumerate(specs)
    ]


def _section_record(index: int, *, name: str, offset: int, data: bytes, role: str) -> dict[str, Any]:
    end = offset + len(data)
    return {
        "index": index,
        "name": name,
        "offset": offset,
        "start": offset,
        "end": end,
        "length": len(data),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "optimization_role": role,
        "score_claim": False,
    }


def _coverage_record(sections: Sequence[Mapping[str, Any]], *, payload_bytes: int) -> dict[str, Any]:
    contiguous = True
    cursor = 0
    total = 0
    for section in sections:
        offset = int(section.get("offset") or 0)
        length = int(section.get("length") or 0)
        if offset != cursor or length < 0:
            contiguous = False
        cursor = offset + length
        total += length
    return {
        "payload_bytes": payload_bytes,
        "section_bytes": total,
        "covers_payload": total == payload_bytes and cursor == payload_bytes,
        "contiguous": contiguous and cursor == payload_bytes,
        "section_count": len(sections),
    }


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> list[str]:
    blockers = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        blockers.append("schema_mismatch")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("score_claim_not_false")
    if manifest.get("dispatch_attempted") is not False:
        blockers.append("dispatch_attempted_not_false")
    if manifest.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("ready_for_exact_eval_dispatch_not_false")
    archive = manifest.get("archive")
    member = manifest.get("member")
    parser = manifest.get("parser")
    sections = manifest.get("sections")
    coverage = manifest.get("coverage")
    if not isinstance(archive, Mapping):
        blockers.append("archive_identity_missing")
    else:
        if not isinstance(archive.get("bytes"), int) or int(archive.get("bytes") or 0) <= 0:
            blockers.append("archive_bytes_invalid")
        if not _is_sha256(archive.get("sha256")):
            blockers.append("archive_sha256_invalid")
    if not isinstance(member, Mapping):
        blockers.append("member_identity_missing")
    else:
        if not isinstance(member.get("name"), str) or not member.get("name"):
            blockers.append("member_name_missing")
        if not isinstance(member.get("bytes"), int) or int(member.get("bytes") or 0) <= 0:
            blockers.append("member_bytes_invalid")
        if not _is_sha256(member.get("sha256")):
            blockers.append("member_sha256_invalid")
    if not isinstance(parser, Mapping) or parser.get("name") not in PARSER_CHOICES[1:]:
        blockers.append("parser_name_invalid")
    if not isinstance(sections, list) or not sections:
        blockers.append("sections_missing")
        sections = []
    if not isinstance(coverage, Mapping):
        blockers.append("coverage_missing")
        coverage = {}
    blockers.extend(_validate_manifest_shape_without_gate(sections, coverage))
    return blockers


def _validate_manifest_shape_without_gate(
    sections: Sequence[Any],
    coverage: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    cursor = 0
    names: set[str] = set()
    for index, section in enumerate(sections):
        if not isinstance(section, Mapping):
            blockers.append(f"section_not_object:{index}")
            continue
        name = str(section.get("name") or "")
        if not name:
            blockers.append(f"section_name_missing:{index}")
        if name in names:
            blockers.append(f"section_name_duplicate:{name}")
        names.add(name)
        if section.get("index") != index:
            blockers.append(f"section_index_mismatch:{name or index}")
        offset = section.get("offset")
        length = section.get("length")
        end = section.get("end")
        if not isinstance(offset, int) or offset < 0:
            blockers.append(f"section_offset_invalid:{name or index}")
            offset = cursor
        if not isinstance(length, int) or length < 0:
            blockers.append(f"section_length_invalid:{name or index}")
            length = 0
        if section.get("bytes") != length:
            blockers.append(f"section_bytes_length_mismatch:{name or index}")
        if section.get("start") != offset:
            blockers.append(f"section_start_offset_mismatch:{name or index}")
        if end != offset + length:
            blockers.append(f"section_end_mismatch:{name or index}")
        if offset != cursor:
            blockers.append(f"section_not_contiguous:{name or index}")
        cursor = offset + length
        if not _is_sha256(section.get("sha256")):
            blockers.append(f"section_sha256_invalid:{name or index}")
        if section.get("score_claim") is not False:
            blockers.append(f"section_score_claim_not_false:{name or index}")
    if coverage.get("covers_payload") is not True:
        blockers.append("coverage_does_not_cover_payload")
    if coverage.get("contiguous") is not True:
        blockers.append("coverage_not_contiguous")
    if coverage.get("section_count") != len(sections):
        blockers.append("coverage_section_count_mismatch")
    if isinstance(coverage.get("payload_bytes"), int) and coverage.get("payload_bytes") != cursor:
        blockers.append("coverage_payload_bytes_mismatch")
    if isinstance(coverage.get("section_bytes"), int) and coverage.get("section_bytes") != cursor:
        blockers.append("coverage_section_bytes_mismatch")
    return blockers


def _archive_path_from_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path | None,
) -> Path | None:
    archive = manifest.get("archive")
    if not isinstance(archive, Mapping) or not isinstance(archive.get("path"), str):
        return None
    path = Path(str(archive["path"]))
    if path.is_absolute():
        return path
    if repo_root is not None:
        return Path(repo_root) / path
    return path


def _parser_name_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    parser = manifest.get("parser")
    if not isinstance(parser, Mapping):
        return None
    name = parser.get("name")
    return str(name) if name in PARSER_CHOICES[1:] else None


def _compare_rebuilt_manifest(manifest: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> list[str]:
    blockers = []
    for key in ("archive", "member", "parser", "sections", "coverage"):
        expected = rebuilt.get(key)
        actual = manifest.get(key)
        if key == "parser" and isinstance(expected, Mapping) and isinstance(actual, Mapping):
            expected = {"name": expected.get("name")}
            actual = {"name": actual.get("name")}
        if actual != expected:
            blockers.append(f"{key}_does_not_match_archive")
    return blockers


def _gate(blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "name": "parser-section gate",
        "ready": not blockers,
        "blockers": list(blockers),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _parser_confidence(parser_name: str) -> str:
    if parser_name == PARSER_PR101:
        return "fixed public PR101 offsets"
    if parser_name == PARSER_PR103:
        return "existing PR103 lc_ac parser"
    if parser_name == PARSER_PR106:
        return "0xff header plus 24-bit decoder length"
    return "unknown"


def _gate_ready(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("parser_section_gate")
    return isinstance(gate, Mapping) and gate.get("ready") is True


def _record_summary_line(record: Mapping[str, Any]) -> str:
    archive = record.get("archive") if isinstance(record.get("archive"), Mapping) else {}
    member = record.get("member") if isinstance(record.get("member"), Mapping) else {}
    parser = record.get("parser") if isinstance(record.get("parser"), Mapping) else {}
    sections = record.get("sections") if isinstance(record.get("sections"), list) else []
    return (
        f"- {record.get('label')}: parser={parser.get('name')} "
        f"archive_sha256={archive.get('sha256')} member={member.get('name')} "
        f"member_sha256={member.get('sha256')} sections={len(sections)}"
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


__all__ = [
    "BATCH_SCHEMA",
    "MANIFEST_SCHEMA",
    "PARSER_AUTO",
    "PARSER_CHOICES",
    "PARSER_PR101",
    "PARSER_PR103",
    "PARSER_PR106",
    "SCHEMA_VERSION",
    "TOOL_NAME",
    "HnervPacketSectionManifestError",
    "build_packet_section_manifest",
    "build_packet_section_manifest_batch",
    "dumps_manifest",
    "render_manifest_summary",
    "validate_packet_section_manifest",
    "validate_packet_section_manifest_batch",
]
