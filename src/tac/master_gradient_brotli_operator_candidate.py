# SPDX-License-Identifier: MIT
"""Materialize a Brotli-section master-gradient operator candidate.

This is the first executable row type for the operator-response replacement of
raw archive-byte "master gradients": take a parser-proven PR106 Brotli section,
losslessly recompress the decompressed bytes, rebuild the monolithic packet, and
emit packet-closure evidence without claiming score movement.
"""

from __future__ import annotations

import zipfile
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

MANIFEST_SCHEMA = "tac_master_gradient_brotli_operator_candidate_v1"
SUPPORTED_TARGET_SECTIONS: frozenset[str] = frozenset(
    {"decoder_packed_brotli", "latents_and_sidecar_brotli"}
)
RATE_DENOMINATOR_BYTES = 37_545_489


class MasterGradientBrotliOperatorError(ValueError):
    """Raised when a Brotli operator row cannot be materialized safely."""


@dataclass(frozen=True)
class BrotliGridRow:
    quality: int
    lgwin: int
    bytes_out: int
    sha256: str
    section_byte_delta: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "quality": self.quality,
            "lgwin": self.lgwin,
            "bytes_out": self.bytes_out,
            "sha256": self.sha256,
            "section_byte_delta": self.section_byte_delta,
        }


def build_master_gradient_brotli_operator_candidate(
    *,
    source_archive: Path,
    output_dir: Path,
    target_section: str,
    candidate_id: str,
    qualities: tuple[int, ...] = tuple(range(12)),
    lgwin_values: tuple[int, ...] = tuple(range(10, 25)),
    require_smaller: bool = True,
) -> dict[str, Any]:
    """Build a byte-different archive candidate from a PR106 Brotli section."""

    source_archive = Path(source_archive)
    output_dir = Path(output_dir)
    if target_section not in SUPPORTED_TARGET_SECTIONS:
        raise MasterGradientBrotliOperatorError(
            f"unsupported Brotli operator target section: {target_section}"
        )
    if not candidate_id:
        raise MasterGradientBrotliOperatorError("candidate_id is required")
    qualities = _normalize_int_tuple(qualities, name="qualities", lower=0, upper=11)
    lgwin_values = _normalize_int_tuple(lgwin_values, name="lgwin_values", lower=10, upper=24)

    layout = inspect_frontier_archive_layout(source_archive)
    logical = layout.get("logical_layout")
    if not isinstance(logical, dict) or logical.get("grammar") != "pr106_ff_packed_hnerv":
        raise MasterGradientBrotliOperatorError(
            "Brotli operator candidate requires parser-proven pr106_ff_packed_hnerv layout"
        )
    section = _find_section(logical, target_section)
    member_name, source_member = _single_member_payload(source_archive)
    if member_name != logical.get("single_member_name"):
        raise MasterGradientBrotliOperatorError("layout member name does not match ZIP payload")
    offset = int(section["offset"])
    source_section = source_member[offset: offset + int(section["len"])]
    if sha256_bytes(source_section) != section.get("sha256"):
        raise MasterGradientBrotliOperatorError("source section SHA mismatch")
    try:
        raw = brotli.decompress(source_section)
    except brotli.error as exc:
        raise MasterGradientBrotliOperatorError(
            f"target section {target_section} does not Brotli-decompress"
        ) from exc

    grid = _brotli_recompression_grid(
        raw,
        source_bytes=len(source_section),
        qualities=qualities,
        lgwin_values=lgwin_values,
    )
    best = min(grid, key=lambda row: (row.bytes_out, row.quality, row.lgwin, row.sha256))
    if require_smaller and best.bytes_out >= len(source_section):
        raise MasterGradientBrotliOperatorError(
            "Brotli operator produced no byte-saving replacement; "
            f"best={best.bytes_out}, source={len(source_section)}"
        )
    if best.sha256 == sha256_bytes(source_section):
        raise MasterGradientBrotliOperatorError("Brotli operator selected a no-op payload")

    output_dir.mkdir(parents=True, exist_ok=True)
    replacement_path = output_dir / f"{target_section}.q{best.quality}.lgwin{best.lgwin}.brotli"
    replacement_payload = brotli.compress(raw, quality=best.quality, lgwin=best.lgwin)
    if sha256_bytes(replacement_payload) != best.sha256:
        raise MasterGradientBrotliOperatorError("selected replacement SHA drifted")
    replacement_path.write_bytes(replacement_payload)

    candidate_archive = output_dir / "archive.zip"
    candidate_manifest_path = output_dir / "candidate_manifest.json"
    candidate_manifest = build_monolithic_packet_candidate(
        source_archive=source_archive,
        output_archive=candidate_archive,
        candidate_id=candidate_id,
        replacements=[
            ReplacementSection(
                section_name=target_section,
                replacement_path=replacement_path,
                expected_old_sha256=sha256_bytes(source_section),
                expected_old_bytes=len(source_section),
                expected_new_sha256=best.sha256,
                expected_new_bytes=best.bytes_out,
            )
        ],
        expected_source_archive_sha256=sha256_file(source_archive),
        expected_source_archive_bytes=source_archive.stat().st_size,
        manifest_output=candidate_manifest_path,
    )
    closure = _packet_closure_summary(
        candidate_archive,
        target_section=target_section,
        expected_section_sha256=best.sha256,
    )
    operator_manifest = {
        "schema": MANIFEST_SCHEMA,
        "candidate_id": candidate_id,
        "mutation_grain": "grammar_aware_operator",
        "mutation_operator": "brotli_section_recompression_tournament",
        "target_section": target_section,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": sha256_file(source_archive),
        },
        "candidate_archive": candidate_manifest["candidate_archive"],
        "source_section": {
            "name": target_section,
            "offset": offset,
            "bytes": len(source_section),
            "sha256": sha256_bytes(source_section),
            "raw_decompressed_bytes": len(raw),
        },
        "replacement_payload": {
            "path": str(replacement_path),
            "bytes": best.bytes_out,
            "sha256": best.sha256,
            "quality": best.quality,
            "lgwin": best.lgwin,
            "section_byte_delta": best.section_byte_delta,
        },
        "rate_delta_score_if_components_unchanged": (
            25.0 * float(candidate_manifest["candidate_archive"]["archive_byte_delta"])
            / RATE_DENOMINATOR_BYTES
        ),
        "grid_top10": [row.to_manifest() for row in sorted(grid, key=lambda row: row.bytes_out)[:10]],
        "candidate_manifest_path": str(candidate_manifest_path),
        "packet_closure": closure,
        "packet_proofs": {
            "repacked_archive": True,
            "updated_zip_headers": closure["zip_headers_bound"],
            "updated_zip_crc": closure["zip_crc_ok"],
            "parser_reparse_success": closure["parser_reparse_success"],
            "brotli_decode_success": closure["target_section_brotli_decodes"],
            "structural_non_noop_section_changed": closure["target_section_sha256_bound"],
            "inflate_success_proof": False,
            "runtime_byte_consumption_noop_detector": False,
        },
        "dispatch_blockers": tuple(
            dict.fromkeys(
                [
                    *candidate_manifest["dispatch_blockers"],
                    "inflate_success_proof_missing",
                    "runtime_byte_consumption_noop_detector_missing",
                ]
            )
        ),
        "promotion_blockers": candidate_manifest["promotion_blockers"],
        "notes": (
            "This is a lossless packet mutation candidate, not a score claim.",
            "It proves ZIP/header/CRC/parser/Brotli closure for the changed section.",
            "Runtime inflate and byte-consumption proof remain required before dispatch readiness.",
        ),
    }
    write_json(output_dir / "operator_manifest.json", operator_manifest)
    return operator_manifest


def _brotli_recompression_grid(
    raw: bytes,
    *,
    source_bytes: int,
    qualities: tuple[int, ...],
    lgwin_values: tuple[int, ...],
) -> list[BrotliGridRow]:
    rows: list[BrotliGridRow] = []
    for quality in qualities:
        for lgwin in lgwin_values:
            try:
                payload = brotli.compress(raw, quality=quality, lgwin=lgwin)
            except brotli.error:
                continue
            rows.append(
                BrotliGridRow(
                    quality=quality,
                    lgwin=lgwin,
                    bytes_out=len(payload),
                    sha256=sha256_bytes(payload),
                    section_byte_delta=len(payload) - source_bytes,
                )
            )
    if not rows:
        raise MasterGradientBrotliOperatorError("Brotli recompression grid produced no rows")
    return rows


def _packet_closure_summary(
    candidate_archive: Path,
    *,
    target_section: str,
    expected_section_sha256: str,
) -> dict[str, Any]:
    with zipfile.ZipFile(candidate_archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        bad_crc_member = zf.testzip()
        if len(infos) != 1:
            raise MasterGradientBrotliOperatorError("candidate archive is not single-member")
        info = infos[0]
        payload = zf.read(info.filename)
    layout = inspect_frontier_archive_layout(candidate_archive)
    logical = layout.get("logical_layout")
    if not isinstance(logical, dict):
        raise MasterGradientBrotliOperatorError("candidate logical layout does not reparse")
    section = _find_section(logical, target_section)
    offset = int(section["offset"])
    section_payload = payload[offset: offset + int(section["len"])]
    try:
        brotli.decompress(section_payload)
        brotli_decodes = True
    except brotli.error:
        brotli_decodes = False
    return {
        "zip_crc_ok": bad_crc_member is None,
        "bad_crc_member": bad_crc_member,
        "zip_headers_bound": info.file_size == len(payload),
        "zip_member_name": info.filename,
        "zip_member_file_size": info.file_size,
        "rebuilt_member_bytes": len(payload),
        "parser_reparse_success": True,
        "logical_grammar": logical.get("grammar"),
        "target_section_sha256_bound": section.get("sha256") == expected_section_sha256,
        "target_section_brotli_decodes": brotli_decodes,
    }


def _find_section(logical: dict[str, Any], section_name: str) -> dict[str, Any]:
    sections = logical.get("sections")
    if not isinstance(sections, list):
        raise MasterGradientBrotliOperatorError("logical layout has no sections")
    for section in sections:
        if isinstance(section, dict) and section.get("name") == section_name:
            return section
    raise MasterGradientBrotliOperatorError(f"section not found: {section_name}")


def _single_member_payload(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise MasterGradientBrotliOperatorError(
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
        raise MasterGradientBrotliOperatorError(f"{name} cannot be empty")
    for value in clean:
        if value < lower or value > upper:
            raise MasterGradientBrotliOperatorError(
                f"{name} value {value} outside [{lower}, {upper}]"
            )
    return clean


__all__ = [
    "MANIFEST_SCHEMA",
    "SUPPORTED_TARGET_SECTIONS",
    "MasterGradientBrotliOperatorError",
    "build_master_gradient_brotli_operator_candidate",
]
