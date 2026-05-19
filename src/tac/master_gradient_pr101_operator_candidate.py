# SPDX-License-Identifier: MIT
"""Materialize PR101 pose-axis master-gradient packet candidates.

The OP-7 pose-axis manifest resolves diagnostic gradient-subject coordinates
onto a parser-proven PR101 decoder section. This module turns one resolved row
into a byte-closed archive candidate by losslessly recompressing the containing
split-Brotli stream at the same compressed length. That proves the packet
mechanics for the row while staying explicit that no score movement is claimed.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli

from tac.frontier_archive_layout import inspect_frontier_archive_layout
from tac.monolithic_packet_candidate import (
    ReplacementSection,
    build_monolithic_packet_candidate,
    sha256_bytes,
    sha256_file,
)
from tac.repo_io import write_json

MANIFEST_SCHEMA = "tac_pr101_pose_axis_decoder_recompression_candidate_v1"
SUPPORTED_SOURCE_SECTIONS: frozenset[str] = frozenset({"decoder", "decoder_blob"})
PR101_PACKET_SECTION_NAME = "decoder_blob"
SUPPORTED_MUTATION_OPERATOR = "decoder_codec_coordinate_response"
RATE_DENOMINATOR_BYTES = 37_545_489


class MasterGradientPR101OperatorError(ValueError):
    """Raised when a PR101 pose-axis operator row cannot be materialized."""


@dataclass(frozen=True)
class BrotliStreamSpan:
    index: int
    compressed_start: int
    compressed_end: int
    compressed_sha256: str
    raw_bytes: int
    raw_sha256: str
    payload: bytes
    raw_payload: bytes

    @property
    def compressed_bytes(self) -> int:
        return self.compressed_end - self.compressed_start

    def to_manifest(self) -> dict[str, object]:
        return {
            "stream_index": self.index,
            "compressed_start": self.compressed_start,
            "compressed_end": self.compressed_end,
            "compressed_bytes": self.compressed_bytes,
            "compressed_sha256": self.compressed_sha256,
            "raw_bytes": self.raw_bytes,
            "raw_sha256": self.raw_sha256,
        }


@dataclass(frozen=True)
class SameLengthBrotliCandidate:
    quality: int
    lgwin: int
    compressed_bytes: int
    compressed_sha256: str
    payload: bytes

    def to_manifest(self) -> dict[str, object]:
        return {
            "quality": self.quality,
            "lgwin": self.lgwin,
            "compressed_bytes": self.compressed_bytes,
            "compressed_sha256": self.compressed_sha256,
        }


def build_pr101_pose_axis_decoder_recompression_candidate(
    *,
    source_archive: Path,
    operator_manifest: Mapping[str, Any],
    output_dir: Path,
    candidate_id: str,
    candidate_rank: int = 1,
    operator_manifest_path: Path | None = None,
    qualities: tuple[int, ...] = tuple(range(12)),
    lgwin_values: tuple[int, ...] = tuple(range(10, 25)),
) -> dict[str, Any]:
    """Build a same-length split-Brotli recompression candidate for PR101.

    The output archive is a packet-construction artifact only. It proves that a
    resolved OP-7 grammar coordinate can be rebuilt with ZIP/header/CRC/parser
    closure. Because the selected stream is decompression-equivalent, runtime
    byte-consumption, inflate success, score response, and exact eval remain
    explicit blockers.
    """

    source_archive = Path(source_archive)
    output_dir = Path(output_dir)
    if not candidate_id:
        raise MasterGradientPR101OperatorError("candidate_id is required")
    if candidate_rank < 1:
        raise MasterGradientPR101OperatorError("candidate_rank must be >= 1")
    qualities = _normalize_int_tuple(qualities, name="qualities", lower=0, upper=11)
    lgwin_values = _normalize_int_tuple(lgwin_values, name="lgwin_values", lower=10, upper=24)

    selected, selected_spec, resolution = _select_resolved_candidate(
        operator_manifest,
        candidate_rank=candidate_rank,
    )
    _validate_resolved_candidate(selected)

    expected_source_sha = _expected_source_sha(operator_manifest, resolution, selected_spec)
    expected_source_bytes = _expected_source_bytes(operator_manifest, resolution, selected_spec)
    source_sha = sha256_file(source_archive)
    source_bytes = source_archive.stat().st_size
    if expected_source_sha and source_sha.lower() != expected_source_sha.lower():
        raise MasterGradientPR101OperatorError(
            f"source archive sha256 mismatch: {source_sha} != {expected_source_sha}"
        )
    if expected_source_bytes is not None and source_bytes != expected_source_bytes:
        raise MasterGradientPR101OperatorError(
            f"source archive bytes mismatch: {source_bytes} != {expected_source_bytes}"
        )

    layout = inspect_frontier_archive_layout(source_archive)
    logical = layout.get("logical_layout")
    if not isinstance(logical, dict) or logical.get("grammar") != "pr101_fixed_offset_hnerv_microcodec":
        raise MasterGradientPR101OperatorError(
            "PR101 pose-axis candidate requires parser-proven pr101_fixed_offset_hnerv_microcodec layout"
        )
    section = _find_section(logical, PR101_PACKET_SECTION_NAME)
    member_name, source_member = _single_member_payload(source_archive)
    if member_name != logical.get("single_member_name"):
        raise MasterGradientPR101OperatorError("layout member name does not match ZIP payload")

    section_offset = int(section["offset"])
    section_len = int(section["len"])
    source_decoder = source_member[section_offset: section_offset + section_len]
    source_decoder_sha = sha256_bytes(source_decoder)
    if source_decoder_sha != section.get("sha256"):
        raise MasterGradientPR101OperatorError("source decoder section SHA mismatch")

    relative_offset = int(selected["section_relative_offset"])
    if relative_offset < 0 or relative_offset >= len(source_decoder):
        raise MasterGradientPR101OperatorError(
            f"section_relative_offset outside decoder section: {relative_offset}"
        )
    streams = _split_concatenated_brotli(source_decoder)
    stream = _stream_for_relative_offset(streams, relative_offset)
    replacement_stream = _select_same_length_recompression(
        stream,
        qualities=qualities,
        lgwin_values=lgwin_values,
    )
    candidate_decoder = (
        source_decoder[: stream.compressed_start]
        + replacement_stream.payload
        + source_decoder[stream.compressed_end:]
    )
    if len(candidate_decoder) != len(source_decoder):
        raise MasterGradientPR101OperatorError("same-length recompression length drifted")
    if candidate_decoder == source_decoder:
        raise MasterGradientPR101OperatorError("same-length recompression selected a no-op decoder")
    candidate_decoder_sha = sha256_bytes(candidate_decoder)

    candidate_streams = _split_concatenated_brotli(candidate_decoder)
    raw_equivalence = _raw_stream_equivalence(streams, candidate_streams)
    if not raw_equivalence["all_stream_raw_sha256_match"]:
        raise MasterGradientPR101OperatorError("candidate decoder stream raw bytes changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    replacement_path = (
        output_dir
        / f"{PR101_PACKET_SECTION_NAME}.rank{candidate_rank:04d}.stream{stream.index:02d}.same_len.brotli_section"
    )
    replacement_path.write_bytes(candidate_decoder)

    candidate_archive = output_dir / "archive.zip"
    candidate_manifest_path = output_dir / "candidate_manifest.json"
    packet_manifest = build_monolithic_packet_candidate(
        source_archive=source_archive,
        output_archive=candidate_archive,
        candidate_id=candidate_id,
        replacements=[
            ReplacementSection(
                section_name=PR101_PACKET_SECTION_NAME,
                replacement_path=replacement_path,
                expected_old_sha256=source_decoder_sha,
                expected_old_bytes=len(source_decoder),
                expected_new_sha256=candidate_decoder_sha,
                expected_new_bytes=len(candidate_decoder),
            )
        ],
        expected_source_archive_sha256=source_sha,
        expected_source_archive_bytes=source_bytes,
        manifest_output=candidate_manifest_path,
    )
    closure = _packet_closure_summary(
        candidate_archive,
        expected_section_sha256=candidate_decoder_sha,
        expected_stream_count=len(streams),
    )
    operator_manifest_sha = (
        sha256_file(operator_manifest_path) if operator_manifest_path is not None else None
    )
    dispatch_blockers = tuple(
        dict.fromkeys(
            [
                *packet_manifest["dispatch_blockers"],
                "inflate_success_proof_missing",
                "runtime_byte_consumption_noop_detector_missing",
                "score_response_matrix_missing",
                "exact_eval_missing",
                "semantic_runtime_output_noop_expected_until_full_inflate_proof",
            ]
        )
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "candidate_id": candidate_id,
        "mutation_grain": "grammar_aware_operator",
        "mutation_operator": SUPPORTED_MUTATION_OPERATOR,
        "target_section": PR101_PACKET_SECTION_NAME,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "source_operator_manifest": {
            "path": str(operator_manifest_path) if operator_manifest_path is not None else None,
            "sha256": operator_manifest_sha,
            "schema": operator_manifest.get("schema"),
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_bytes,
            "sha256": source_sha,
        },
        "candidate_archive": packet_manifest["candidate_archive"],
        "selected_pose_axis_candidate": selected,
        "candidate_modification_spec": selected_spec,
        "source_decoder_section": {
            "name": PR101_PACKET_SECTION_NAME,
            "offset": section_offset,
            "bytes": len(source_decoder),
            "sha256": source_decoder_sha,
            "split_brotli_stream_count": len(streams),
        },
        "selected_stream": stream.to_manifest(),
        "replacement_stream": replacement_stream.to_manifest(),
        "replacement_decoder_section": {
            "path": str(replacement_path),
            "bytes": len(candidate_decoder),
            "sha256": candidate_decoder_sha,
            "section_byte_delta": len(candidate_decoder) - len(source_decoder),
            "changed": True,
        },
        "rate_delta_score_if_components_unchanged": (
            25.0 * float(packet_manifest["candidate_archive"]["archive_byte_delta"])
            / RATE_DENOMINATOR_BYTES
        ),
        "decoder_brotli_raw_equivalence": raw_equivalence,
        "candidate_manifest_path": str(candidate_manifest_path),
        "packet_closure": closure,
        "packet_proofs": {
            "repacked_archive": True,
            "updated_zip_headers": closure["zip_headers_bound"],
            "updated_zip_crc": closure["zip_crc_ok"],
            "parser_reparse_success": closure["parser_reparse_success"],
            "structural_non_noop_section_changed": closure["target_section_sha256_bound"],
            "decoder_brotli_stream_reparse_success": closure["split_brotli_stream_reparse_success"],
            "decoder_brotli_roundtrip_raw_equivalent": raw_equivalence["all_stream_raw_sha256_match"],
            "inflate_success_proof": False,
            "runtime_byte_consumption_noop_detector": False,
            "score_response_matrix": False,
        },
        "dispatch_blockers": dispatch_blockers,
        "promotion_blockers": packet_manifest["promotion_blockers"],
        "notes": (
            "This materializes one OP-7 PR101 pose-axis grammar coordinate as a byte-closed packet candidate.",
            "The changed decoder stream is Brotli raw-equivalent and same-length, so this is packet-mechanics proof, not score movement.",
            "Runtime inflate, byte-consumption no-op detection, score response, and exact CUDA auth eval remain mandatory blockers.",
        ),
    }
    write_json(output_dir / "operator_manifest.json", manifest)
    return manifest


def _select_resolved_candidate(
    operator_manifest: Mapping[str, Any],
    *,
    candidate_rank: int,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    resolution = operator_manifest.get("grammar_aware_operator_candidate_resolution")
    if not isinstance(resolution, Mapping):
        resolution = operator_manifest
    resolved_rows = resolution.get("resolved_pose_axis_candidates")
    if not isinstance(resolved_rows, list):
        raise MasterGradientPR101OperatorError("resolved_pose_axis_candidates missing")
    selected = None
    for row in resolved_rows:
        if isinstance(row, dict) and int(row.get("rank", -1)) == candidate_rank:
            selected = row
            break
    if selected is None:
        raise MasterGradientPR101OperatorError(
            f"resolved pose-axis candidate rank not found: {candidate_rank}"
        )
    spec = _find_candidate_spec(resolution, selected)
    return selected, spec, resolution


def _find_candidate_spec(
    resolution: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    selected_spec_id = selected.get("spec_id")
    specs = resolution.get("candidate_modification_specs")
    if not isinstance(specs, list):
        return {}
    for spec in specs:
        if isinstance(spec, dict) and spec.get("spec_id") == selected_spec_id:
            return spec
    return {}


def _validate_resolved_candidate(selected: Mapping[str, Any]) -> None:
    section_name = selected.get("section_name")
    if section_name not in SUPPORTED_SOURCE_SECTIONS:
        raise MasterGradientPR101OperatorError(
            f"unsupported resolved section for PR101 builder: {section_name}"
        )
    if selected.get("mutation_operator") != SUPPORTED_MUTATION_OPERATOR:
        raise MasterGradientPR101OperatorError(
            "PR101 builder only materializes decoder_codec_coordinate_response rows"
        )
    section_role = selected.get("section_role")
    if section_role != "brotli_streams_int8":
        raise MasterGradientPR101OperatorError(
            f"unsupported PR101 decoder role: {section_role}"
        )
    if "section_relative_offset" not in selected:
        raise MasterGradientPR101OperatorError("section_relative_offset missing")


def _expected_source_sha(
    operator_manifest: Mapping[str, Any],
    resolution: Mapping[str, Any],
    selected_spec: Mapping[str, Any],
) -> str | None:
    for payload in (selected_spec, resolution, operator_manifest):
        value = payload.get("source_archive_sha256") or payload.get("archive_sha256")
        if isinstance(value, str) and value:
            return value
    source_anchor = operator_manifest.get("source_anchor")
    if isinstance(source_anchor, Mapping):
        value = source_anchor.get("scored_archive_sha256") or source_anchor.get("archive_sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _expected_source_bytes(
    operator_manifest: Mapping[str, Any],
    resolution: Mapping[str, Any],
    selected_spec: Mapping[str, Any],
) -> int | None:
    for payload in (selected_spec, resolution, operator_manifest):
        value = payload.get("source_archive_bytes") or payload.get("archive_bytes")
        if isinstance(value, int):
            return value
    source_anchor = operator_manifest.get("source_anchor")
    if isinstance(source_anchor, Mapping):
        value = source_anchor.get("scored_archive_bytes")
        if isinstance(value, int):
            return value
    return None


def _split_concatenated_brotli(payload: bytes) -> list[BrotliStreamSpan]:
    streams: list[BrotliStreamSpan] = []
    cursor = 0
    while cursor < len(payload):
        start = cursor
        decoder = brotli.Decompressor()
        raw_parts: list[bytes] = []
        while cursor < len(payload):
            try:
                raw_parts.append(decoder.process(payload[cursor: cursor + 1]))
            except brotli.error as exc:
                raise MasterGradientPR101OperatorError(
                    f"Brotli split failed at byte {cursor}"
                ) from exc
            cursor += 1
            if decoder.is_finished():
                break
        if not decoder.is_finished():
            raise MasterGradientPR101OperatorError("unterminated Brotli stream")
        stream_payload = payload[start:cursor]
        raw_payload = b"".join(raw_parts)
        streams.append(
            BrotliStreamSpan(
                index=len(streams),
                compressed_start=start,
                compressed_end=cursor,
                compressed_sha256=sha256_bytes(stream_payload),
                raw_bytes=len(raw_payload),
                raw_sha256=sha256_bytes(raw_payload),
                payload=stream_payload,
                raw_payload=raw_payload,
            )
        )
    if not streams:
        raise MasterGradientPR101OperatorError("decoder section contains no Brotli streams")
    return streams


def _stream_for_relative_offset(
    streams: list[BrotliStreamSpan],
    relative_offset: int,
) -> BrotliStreamSpan:
    for stream in streams:
        if stream.compressed_start <= relative_offset < stream.compressed_end:
            return stream
    raise MasterGradientPR101OperatorError(
        f"no Brotli stream contains relative offset {relative_offset}"
    )


def _select_same_length_recompression(
    stream: BrotliStreamSpan,
    *,
    qualities: tuple[int, ...],
    lgwin_values: tuple[int, ...],
) -> SameLengthBrotliCandidate:
    candidates: list[SameLengthBrotliCandidate] = []
    source_sha = stream.compressed_sha256
    for quality in qualities:
        for lgwin in lgwin_values:
            try:
                payload = brotli.compress(stream.raw_payload, quality=quality, lgwin=lgwin)
            except brotli.error:
                continue
            payload_sha = sha256_bytes(payload)
            if len(payload) != stream.compressed_bytes or payload_sha == source_sha:
                continue
            try:
                raw = brotli.decompress(payload)
            except brotli.error:
                continue
            if raw != stream.raw_payload:
                continue
            candidates.append(
                SameLengthBrotliCandidate(
                    quality=quality,
                    lgwin=lgwin,
                    compressed_bytes=len(payload),
                    compressed_sha256=payload_sha,
                    payload=payload,
                )
            )
    if not candidates:
        raise MasterGradientPR101OperatorError(
            "no same-length byte-different Brotli recompression found for selected stream"
        )
    return min(candidates, key=lambda row: (row.quality, row.lgwin, row.compressed_sha256))


def _raw_stream_equivalence(
    source_streams: list[BrotliStreamSpan],
    candidate_streams: list[BrotliStreamSpan],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if len(source_streams) != len(candidate_streams):
        return {
            "source_stream_count": len(source_streams),
            "candidate_stream_count": len(candidate_streams),
            "all_stream_raw_sha256_match": False,
            "streams": rows,
        }
    for source, candidate in zip(source_streams, candidate_streams, strict=True):
        matched = source.raw_sha256 == candidate.raw_sha256 and source.raw_bytes == candidate.raw_bytes
        rows.append(
            {
                "stream_index": source.index,
                "source_raw_bytes": source.raw_bytes,
                "candidate_raw_bytes": candidate.raw_bytes,
                "source_raw_sha256": source.raw_sha256,
                "candidate_raw_sha256": candidate.raw_sha256,
                "raw_equivalent": matched,
            }
        )
    return {
        "source_stream_count": len(source_streams),
        "candidate_stream_count": len(candidate_streams),
        "all_stream_raw_sha256_match": all(row["raw_equivalent"] for row in rows),
        "streams": rows,
    }


def _packet_closure_summary(
    candidate_archive: Path,
    *,
    expected_section_sha256: str,
    expected_stream_count: int,
) -> dict[str, Any]:
    with zipfile.ZipFile(candidate_archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        bad_crc_member = zf.testzip()
        if len(infos) != 1:
            raise MasterGradientPR101OperatorError("candidate archive is not single-member")
        info = infos[0]
        payload = zf.read(info.filename)
    layout = inspect_frontier_archive_layout(candidate_archive)
    logical = layout.get("logical_layout")
    if not isinstance(logical, dict):
        raise MasterGradientPR101OperatorError("candidate logical layout does not reparse")
    section = _find_section(logical, PR101_PACKET_SECTION_NAME)
    offset = int(section["offset"])
    decoder_payload = payload[offset: offset + int(section["len"])]
    streams = _split_concatenated_brotli(decoder_payload)
    return {
        "zip_crc_ok": bad_crc_member is None,
        "bad_crc_member": bad_crc_member,
        "zip_headers_bound": info.file_size == len(payload),
        "zip_member_name": info.filename,
        "zip_member_file_size": info.file_size,
        "zip_member_crc32": f"{info.CRC:08x}",
        "rebuilt_member_bytes": len(payload),
        "parser_reparse_success": True,
        "logical_grammar": logical.get("grammar"),
        "target_section_sha256_bound": section.get("sha256") == expected_section_sha256,
        "split_brotli_stream_reparse_success": len(streams) == expected_stream_count,
        "split_brotli_stream_count": len(streams),
    }


def _find_section(logical: Mapping[str, Any], section_name: str) -> dict[str, Any]:
    sections = logical.get("sections")
    if not isinstance(sections, list):
        raise MasterGradientPR101OperatorError("logical layout has no sections")
    for section in sections:
        if isinstance(section, dict) and section.get("name") == section_name:
            return section
    raise MasterGradientPR101OperatorError(f"section not found: {section_name}")


def _single_member_payload(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise MasterGradientPR101OperatorError(
                f"expected one ZIP member, found {len(infos)}"
            )
        return infos[0].filename, zf.read(infos[0].filename)


def _normalize_int_tuple(
    values: tuple[int, ...],
    *,
    name: str,
    lower: int,
    upper: int,
) -> tuple[int, ...]:
    clean = tuple(dict.fromkeys(int(value) for value in values))
    if not clean:
        raise MasterGradientPR101OperatorError(f"{name} cannot be empty")
    for value in clean:
        if value < lower or value > upper:
            raise MasterGradientPR101OperatorError(
                f"{name} value {value} outside [{lower}, {upper}]"
            )
    return clean


__all__ = [
    "MANIFEST_SCHEMA",
    "MasterGradientPR101OperatorError",
    "build_pr101_pose_axis_decoder_recompression_candidate",
]
