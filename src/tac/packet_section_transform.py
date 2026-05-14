# SPDX-License-Identifier: MIT
"""Typed packet-section transform bridge for contest archive candidates.

This module connects parser-section custody manifests to deterministic packet
rewrites. It is intentionally narrow: no scorer loads, no dispatch, no score
claim, and no hidden sidecars. The first supported compiler path is the public
HNeRV PR106 ``0xff + len24`` single-member grammar because it already has
byte-proved low-level packer support.
"""
from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import brotli

from tac.analysis.hnerv_packet_sections import (
    PARSER_PR106,
    PARSER_PR103,
    build_packet_section_manifest,
    validate_packet_section_manifest,
)
from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    brotli_recode_search,
    read_packed_archive_view,
    read_strict_single_member_zip,
    sha256_bytes,
)
from tac.hnerv_section_repack import RATE_SCORE_PER_BYTE
from tac.repo_io import sha256_file

SCHEMA = "packet_section_transform.v1"
OPPORTUNITY_SCHEMA = "hnerv_brotli_section_recode_opportunities.v1"
CERTIFICATION_SCHEMA = "hnerv_packet_transform_candidate_cert.v1"
CONTEST_CANDIDATE_BLOCKERS = (
    "requires_archive_manifest_preflight",
    "requires_lane_dispatch_claim",
    "requires_exact_cuda_auth_eval",
)
PR106_RUNTIME_RECODE_SECTIONS = frozenset(
    {"decoder_packed_brotli", "latents_and_sidecar_brotli"}
)
PR103_LAST_SECTION_RECODE_SECTIONS = frozenset({"sidecar_corrections_brotli"})
PR103_FIXED_LAYOUT_RECODE_SECTIONS = frozenset(
    {
        "non_ac_weights_brotli",
        "ac_histograms_brotli",
        "latent_low_bytes_brotli",
        "latent_hi_histogram_brotli",
    }
)


class PacketSectionTransformError(ValueError):
    """Raised when a packet-section transform cannot be compiled safely."""


@dataclass(frozen=True)
class SectionIR:
    """One parser-proven byte section inside a charged archive member."""

    name: str
    index: int
    offset: int
    length: int
    sha256: str
    optimization_role: str

    @classmethod
    def from_manifest_section(cls, section: Mapping[str, Any]) -> "SectionIR":
        return cls(
            name=str(section["name"]),
            index=int(section["index"]),
            offset=int(section["offset"]),
            length=int(section["length"]),
            sha256=str(section["sha256"]),
            optimization_role=str(section.get("optimization_role") or "unknown"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "index": self.index,
            "offset": self.offset,
            "length": self.length,
            "bytes": self.length,
            "sha256": self.sha256,
            "optimization_role": self.optimization_role,
        }


@dataclass(frozen=True)
class PacketIR:
    """Archive/member/section identity used by packet transform compilers."""

    label: str
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    member_name: str
    member_bytes: int
    member_sha256: str
    parser_name: str
    parser_input: Mapping[str, Any]
    pr106_sidecar_wrapper: Mapping[str, Any] | None
    sections: tuple[SectionIR, ...]
    parser_section_gate: Mapping[str, Any]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "PacketIR":
        archive = _mapping(manifest.get("archive"), "archive")
        member = _mapping(manifest.get("member"), "member")
        parser = _mapping(manifest.get("parser"), "parser")
        sections = manifest.get("sections")
        if not isinstance(sections, list) or not sections:
            raise PacketSectionTransformError("manifest sections must be a nonempty list")
        return cls(
            label=str(manifest.get("label") or ""),
            archive_path=str(archive["path"]),
            archive_bytes=int(archive["bytes"]),
            archive_sha256=str(archive["sha256"]),
            member_name=str(member["name"]),
            member_bytes=int(member["bytes"]),
            member_sha256=str(member["sha256"]),
            parser_name=str(parser["name"]),
            parser_input=_mapping(manifest.get("parser_input"), "parser_input"),
            pr106_sidecar_wrapper=(
                _mapping(manifest.get("pr106_sidecar_wrapper"), "pr106_sidecar_wrapper")
                if manifest.get("pr106_sidecar_wrapper") is not None
                else None
            ),
            sections=tuple(
                SectionIR.from_manifest_section(section)
                for section in sections
                if isinstance(section, Mapping)
            ),
            parser_section_gate=_mapping(
                manifest.get("parser_section_gate"), "parser_section_gate"
            ),
        )

    def section(self, name: str) -> SectionIR:
        for section in self.sections:
            if section.name == name:
                return section
        raise PacketSectionTransformError(f"section not found in packet IR: {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "archive_path": self.archive_path,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "member_name": self.member_name,
            "member_bytes": self.member_bytes,
            "member_sha256": self.member_sha256,
            "parser_name": self.parser_name,
            "parser_input": dict(self.parser_input),
            "pr106_sidecar_wrapper": (
                dict(self.pr106_sidecar_wrapper)
                if self.pr106_sidecar_wrapper is not None
                else None
            ),
            "sections": [section.to_dict() for section in self.sections],
            "parser_section_gate": dict(self.parser_section_gate),
        }


@dataclass(frozen=True)
class TransformOutput:
    """Output for one section transform before packet reassembly."""

    section_name: str
    payload: bytes
    metadata: Mapping[str, Any] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()


class PacketSectionTransform(Protocol):
    """Protocol for deterministic section transforms."""

    name: str

    def applies_to(self, section: SectionIR) -> bool:
        ...

    def transform(self, section: SectionIR, payload: bytes) -> TransformOutput:
        ...


@dataclass(frozen=True)
class CompositePacketSectionTransform:
    """Apply exactly one matching transform per section."""

    transforms: tuple[PacketSectionTransform, ...]
    name: str = "composite_packet_section_transform"

    def applies_to(self, section: SectionIR) -> bool:
        return any(transform.applies_to(section) for transform in self.transforms)

    def transform(self, section: SectionIR, payload: bytes) -> TransformOutput:
        matching = [transform for transform in self.transforms if transform.applies_to(section)]
        if not matching:
            return TransformOutput(
                section_name=section.name,
                payload=payload,
                blockers=(f"transform_matched_no_section:{section.name}",),
            )
        if len(matching) > 1:
            return TransformOutput(
                section_name=section.name,
                payload=payload,
                blockers=(f"multiple_transforms_match_section:{section.name}",),
                metadata={"matching_transform_count": len(matching)},
            )
        return matching[0].transform(section, payload)


@dataclass(frozen=True)
class BrotliRecodeSectionTransform:
    """Grammar-preserving Brotli recode for an existing compressed section."""

    target_section: str
    qualities: tuple[int, ...] = (9, 10, 11)
    lgwins: tuple[int | None, ...] = (None, 18, 20, 22, 24)
    lgblocks: tuple[int | None, ...] = (None,)
    jobs: int = 1
    allow_rate_regression: bool = False
    name: str = "brotli_recode_section"

    def applies_to(self, section: SectionIR) -> bool:
        return section.name == self.target_section

    def transform(self, section: SectionIR, payload: bytes) -> TransformOutput:
        try:
            raw = brotli.decompress(payload)
        except brotli.error:
            return TransformOutput(
                section_name=section.name,
                payload=payload,
                blockers=(f"section_not_brotli_decompressible:{section.name}",),
            )
        try:
            choice, candidate = brotli_recode_search(
                section.name,
                payload,
                qualities=self.qualities,
                lgwins=self.lgwins,
                lgblocks=self.lgblocks,
                jobs=self.jobs,
            )
        except HnervLowlevelPackError as exc:
            return TransformOutput(
                section_name=section.name,
                payload=payload,
                blockers=(f"brotli_recode_failed:{section.name}:{exc}",),
            )
        raw_equal = brotli.decompress(candidate) == raw
        byte_delta = len(candidate) - len(payload)
        blockers: list[str] = []
        if not raw_equal:
            blockers.append(f"brotli_raw_mismatch:{section.name}")
        if not choice.changed:
            blockers.append(f"candidate_section_noop:{section.name}")
        if byte_delta >= 0 and not self.allow_rate_regression:
            blockers.append(f"candidate_section_not_rate_positive:{section.name}")
        return TransformOutput(
            section_name=section.name,
            payload=candidate if not blockers else payload,
            metadata={
                "choice": {
                    "quality": choice.quality,
                    "lgwin": choice.lgwin,
                    "lgblock": choice.lgblock,
                    "source_bytes": len(payload),
                    "candidate_bytes": len(candidate),
                    "byte_delta": byte_delta,
                    "source_section_sha256": sha256_bytes(payload),
                    "candidate_section_sha256": sha256_bytes(candidate),
                    "changed": choice.changed,
                },
                "raw_equivalence": {
                    "section_name": section.name,
                    "raw_equal": raw_equal,
                    "raw_bytes": len(raw),
                    "source_raw_sha256": sha256_bytes(raw),
                    "candidate_raw_sha256": sha256_bytes(brotli.decompress(candidate)),
                },
            },
            blockers=tuple(blockers),
        )


def build_hnerv_packet_ir(
    archive_path: str | Path,
    *,
    label: str,
    parser: str = PARSER_PR106,
    repo_root: str | Path | None = None,
) -> PacketIR:
    """Build a validated packet IR from an HNeRV parser-section manifest."""

    manifest = build_packet_section_manifest(
        archive_path,
        label=label,
        parser=parser,
        repo_root=repo_root,
    )
    blockers = validate_packet_section_manifest(manifest, repo_root=repo_root)
    if blockers:
        raise PacketSectionTransformError(
            "packet-section manifest is blocked: " + ", ".join(blockers)
        )
    return PacketIR.from_manifest(manifest)


def compile_hnerv_pr106_section_transform_candidate(
    *,
    source_archive: str | Path,
    label: str,
    transform: PacketSectionTransform,
    output_archive: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Apply one transform to a PR106-style HNeRV packet and emit a candidate.

    This compiler updates the PR106 len24 header when the decoder section
    changes. It refuses exact-eval authority and records only byte-custody and
    raw-equivalence facts.
    """

    source_path = Path(source_archive)
    output_path = Path(output_archive)
    source_ir = build_hnerv_packet_ir(
        source_path,
        label=label,
        parser=PARSER_PR106,
        repo_root=repo_root,
    )
    if source_ir.parser_name != PARSER_PR106:
        raise PacketSectionTransformError(
            f"PR106 section-transform compiler requires parser {PARSER_PR106}, "
            f"got {source_ir.parser_name}"
        )
    source_view = read_packed_archive_view(source_path)
    packed = source_view.packed
    payload_by_section = {
        "packed_header_ff_len24": packed.header,
        "decoder_packed_brotli": packed.decoder_packed_brotli,
        "latents_and_sidecar_brotli": packed.latents_and_sidecar_brotli,
    }

    blockers: list[str] = []
    transform_outputs: list[TransformOutput] = []
    replacements: dict[str, bytes] = {}
    for section in source_ir.sections:
        if not transform.applies_to(section):
            continue
        payload = payload_by_section.get(section.name)
        if payload is None:
            blockers.append(f"section_payload_missing:{section.name}")
            continue
        output = transform.transform(section, payload)
        transform_outputs.append(output)
        blockers.extend(output.blockers)
        if not output.blockers and output.payload != payload:
            replacements[section.name] = output.payload

    if not transform_outputs:
        blockers.append("transform_matched_no_sections")
    if not replacements:
        blockers.append("transform_produced_no_changed_sections")

    candidate_payload = source_view.archive.payload
    candidate_ir: PacketIR | None = None
    candidate_archive_sha = None
    candidate_archive_bytes = None
    if not blockers:
        candidate_packed = packed.__class__(
            header=packed.header,
            decoder_packed_brotli=replacements.get(
                "decoder_packed_brotli",
                packed.decoder_packed_brotli,
            ),
            latents_and_sidecar_brotli=replacements.get(
                "latents_and_sidecar_brotli",
                packed.latents_and_sidecar_brotli,
            ),
        )
        candidate_payload = source_view.emit_payload(candidate_packed)
        source_view.write_archive(output_path, candidate_payload)
        candidate_archive_sha = sha256_file(output_path)
        candidate_archive_bytes = output_path.stat().st_size
        candidate_ir = build_hnerv_packet_ir(
            output_path,
            label=f"{label}_candidate",
            parser=PARSER_PR106,
            repo_root=repo_root,
        )

    changed_sections = _changed_sections(source_ir, candidate_ir)
    total_byte_delta = (
        int(candidate_archive_bytes) - source_ir.archive_bytes
        if candidate_archive_bytes is not None
        else 0
    )
    ready_for_archive_preflight = not blockers and bool(changed_sections)
    return {
        "schema": SCHEMA,
        "tool": "tac.packet_section_transform.compile_hnerv_pr106_section_transform_candidate",
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": ready_for_archive_preflight,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(CONTEST_CANDIDATE_BLOCKERS),
        "blockers": blockers,
        "source_packet_ir": source_ir.to_dict(),
        "candidate_packet_ir": candidate_ir.to_dict() if candidate_ir else None,
        "source_archive_path": str(source_path),
        "candidate_archive_path": str(output_path) if candidate_ir else None,
        "source_archive_sha256": source_ir.archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha,
        "source_archive_bytes": source_ir.archive_bytes,
        "candidate_archive_bytes": candidate_archive_bytes,
        "archive_byte_delta": total_byte_delta,
        "rate_score_delta_if_components_equal": round(
            total_byte_delta * RATE_SCORE_PER_BYTE,
            12,
        ),
        "transform": _transform_summary(transform, transform_outputs),
        "changed_sections": changed_sections,
    }


def certify_hnerv_grammar_preserving_candidate_pair(
    *,
    source_archive: str | Path,
    candidate_archive: str | Path,
    label: str,
    parser: str = PARSER_PR106,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Certify a source/candidate pair as grammar-preserving local evidence.

    This is a local PacketIR gate, not a dispatcher.  It proves that section
    changes are parser-accounted and either raw-equivalent recodes or accepted
    control-byte updates.  Edits that merely delete entropy-stream bytes or
    change decompressed Brotli payloads are blocked even if the candidate
    archive is smaller.
    """

    source_path = Path(source_archive)
    candidate_path = Path(candidate_archive)
    source_ir = build_hnerv_packet_ir(
        source_path,
        label=f"{label}_source",
        parser=parser,
        repo_root=repo_root,
    )
    candidate_ir = build_hnerv_packet_ir(
        candidate_path,
        label=f"{label}_candidate",
        parser=parser,
        repo_root=repo_root,
    )
    source_single = read_strict_single_member_zip(source_path)
    candidate_single = read_strict_single_member_zip(candidate_path)
    source_parser_payload = _parser_payload_for_archive(source_path, source_ir.parser_name)
    candidate_parser_payload = _parser_payload_for_archive(candidate_path, candidate_ir.parser_name)

    changed_sections = _changed_sections(source_ir, candidate_ir)
    section_equivalence = _candidate_pair_section_equivalence(
        parser_name=source_ir.parser_name,
        source_ir=source_ir,
        candidate_ir=candidate_ir,
        source_payload=source_parser_payload,
        candidate_payload=candidate_parser_payload,
    )
    blockers = _candidate_pair_blockers(
        source_ir=source_ir,
        candidate_ir=candidate_ir,
        changed_sections=changed_sections,
        section_equivalence=section_equivalence,
        source_member_name=source_single.member_name,
        candidate_member_name=candidate_single.member_name,
    )
    archive_byte_delta = candidate_ir.archive_bytes - source_ir.archive_bytes
    payload_byte_delta = candidate_ir.member_bytes - source_ir.member_bytes
    if archive_byte_delta >= 0:
        blockers.append("candidate_archive_not_rate_positive")
    if not changed_sections:
        blockers.append("candidate_payload_noop")

    readiness_blockers = _unique_ordered(blockers)
    return {
        "schema": CERTIFICATION_SCHEMA,
        "schema_version": 1,
        "tool": "tac.packet_section_transform.certify_hnerv_grammar_preserving_candidate_pair",
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "remote_gpu_run": False,
        "ready_for_archive_preflight": not readiness_blockers,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": _unique_ordered(
            [*readiness_blockers, *CONTEST_CANDIDATE_BLOCKERS]
        ),
        "readiness_blockers": readiness_blockers,
        "grammar_preserving": not readiness_blockers,
        "rate_positive": archive_byte_delta < 0,
        "score_affecting_payload_changed": source_ir.member_sha256 != candidate_ir.member_sha256,
        "charged_bits_changed": source_ir.member_sha256 != candidate_ir.member_sha256,
        "label": label,
        "parser_name": source_ir.parser_name,
        "source_packet_ir": source_ir.to_dict(),
        "candidate_packet_ir": candidate_ir.to_dict(),
        "source_archive_path": str(source_path),
        "candidate_archive_path": str(candidate_path),
        "source_archive_sha256": source_ir.archive_sha256,
        "candidate_archive_sha256": candidate_ir.archive_sha256,
        "source_archive_bytes": source_ir.archive_bytes,
        "candidate_archive_bytes": candidate_ir.archive_bytes,
        "archive_byte_delta": archive_byte_delta,
        "source_payload_sha256": source_ir.member_sha256,
        "candidate_payload_sha256": candidate_ir.member_sha256,
        "payload_byte_delta": payload_byte_delta,
        "rate_score_delta_if_components_equal": round(
            archive_byte_delta * RATE_SCORE_PER_BYTE,
            12,
        ),
        "changed_sections": changed_sections,
        "section_equivalence": section_equivalence,
        "exact_next_gate": [
            "operator approves exact CUDA promotion",
            "tools/claim_lane_dispatch.py claim creates a non-conflicting lane claim",
            "strict pre-submission compliance manifest is attached to the packet",
            "claimed exact CUDA auth eval runs against this archive SHA",
            "contest_auth_eval.adjudicated.json is harvested and formula-reviewed",
        ],
    }


def scan_hnerv_brotli_recode_opportunities(
    archives: Sequence[tuple[str, str | Path, str]],
    *,
    qualities: Sequence[int] = (9, 10, 11),
    lgwins: Sequence[int | None] = (None, 18, 20, 22, 24),
    lgblocks: Sequence[int | None] = (None,),
    jobs: int = 1,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rank parser-section Brotli recode opportunities without emitting candidates.

    PR101/PR103 sections may have rate-positive Brotli recodes, but their public
    runtimes use fixed section layouts. Those rows are deliberately blocked as
    runtime-adapter work rather than archive-preflight candidates.
    """

    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    if not archives:
        blockers.append("missing_archives")
    for label, archive_path, parser in archives:
        path = Path(archive_path)
        try:
            packet_ir = build_hnerv_packet_ir(
                path,
                label=label,
                parser=parser,
                repo_root=repo_root,
            )
            parser_payload = _parser_payload_for_archive(path, packet_ir.parser_name)
        except (PacketSectionTransformError, HnervLowlevelPackError) as exc:
            blockers.append(f"{label}:archive_unreadable:{exc}")
            continue
        for section in packet_ir.sections:
            payload = parser_payload[section.offset : section.offset + section.length]
            rows.append(
                _scan_one_brotli_section(
                    label=label,
                    packet_ir=packet_ir,
                    section=section,
                    payload=payload,
                    qualities=qualities,
                    lgwins=lgwins,
                    lgblocks=lgblocks,
                    jobs=jobs,
                )
            )
    rate_positive = [
        row
        for row in rows
        if row.get("brotli_decompressible") is True
        and row.get("rate_positive") is True
        and row.get("raw_equal") is True
    ]
    compilable = [
        row
        for row in rate_positive
        if row.get("candidate_compilable_by_existing_bridge") is True
        and row.get("runtime_adapter_required") is False
    ]
    adapter_required = [
        row
        for row in rate_positive
        if row.get("runtime_adapter_required") is True
    ]
    ranked = sorted(
        rate_positive,
        key=lambda row: (
            int(row.get("byte_delta", 0)),
            str(row.get("label", "")),
            str(row.get("section_name", "")),
        ),
    )
    return {
        "schema": OPPORTUNITY_SCHEMA,
        "tool": "tac.packet_section_transform.scan_hnerv_brotli_recode_opportunities",
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_candidate_build": bool(compilable) and not blockers,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(CONTEST_CANDIDATE_BLOCKERS),
        "blockers": blockers,
        "scan_config": {
            "archive_count": len(archives),
            "qualities": [int(quality) for quality in qualities],
            "lgwins": [_optional_json(value) for value in lgwins],
            "lgblocks": [_optional_json(value) for value in lgblocks],
            "jobs": jobs,
        },
        "summary": {
            "section_count": len(rows),
            "brotli_decompressible_count": sum(
                1 for row in rows if row.get("brotli_decompressible") is True
            ),
            "rate_positive_count": len(rate_positive),
            "candidate_compilable_by_existing_bridge_count": len(compilable),
            "runtime_adapter_required_count": len(adapter_required),
            "best_byte_delta": int(ranked[0]["byte_delta"]) if ranked else 0,
            "best_existing_bridge_byte_delta": (
                int(
                    min(
                        compilable,
                        key=lambda row: int(row.get("byte_delta", 0)),
                    )["byte_delta"]
                )
                if compilable
                else 0
            ),
        },
        "ranked_rate_positive": ranked,
        "sections": rows,
    }


def _parser_payload_for_archive(path: Path, parser_name: str) -> bytes:
    if parser_name == PARSER_PR106:
        return read_packed_archive_view(path).hnerv_payload
    return read_strict_single_member_zip(path).payload


def _candidate_pair_section_equivalence(
    *,
    parser_name: str,
    source_ir: PacketIR,
    candidate_ir: PacketIR,
    source_payload: bytes,
    candidate_payload: bytes,
) -> list[dict[str, Any]]:
    candidate_by_name = {section.name: section for section in candidate_ir.sections}
    rows: list[dict[str, Any]] = []
    for source_section in source_ir.sections:
        candidate_section = candidate_by_name.get(source_section.name)
        if candidate_section is None:
            rows.append(
                {
                    "section_name": source_section.name,
                    "status": "candidate_section_missing",
                    "grammar_equivalent": False,
                }
            )
            continue
        source_bytes = _section_bytes(source_payload, source_section)
        candidate_bytes = _section_bytes(candidate_payload, candidate_section)
        changed = (
            source_section.sha256 != candidate_section.sha256
            or source_section.length != candidate_section.length
            or source_section.offset != candidate_section.offset
        )
        row: dict[str, Any] = {
            "section_name": source_section.name,
            "optimization_role": source_section.optimization_role,
            "changed": changed,
            "content_changed": source_section.sha256 != candidate_section.sha256,
            "length_changed": source_section.length != candidate_section.length,
            "offset_changed": source_section.offset != candidate_section.offset,
            "source_bytes": source_section.length,
            "candidate_bytes": candidate_section.length,
            "byte_delta": candidate_section.length - source_section.length,
            "source_section_sha256": source_section.sha256,
            "candidate_section_sha256": candidate_section.sha256,
            "grammar_equivalent": True,
            "equivalence_kind": "unchanged_or_offset_only",
            "blockers": [],
        }
        if parser_name == PARSER_PR106 and source_section.name == "packed_header_ff_len24":
            row.update(_pr106_header_equivalence(source_ir, candidate_ir, source_bytes, candidate_bytes))
        elif _requires_brotli_raw_equivalence(parser_name, source_section.name) and changed:
            row.update(_brotli_raw_equivalence_record(source_section.name, source_bytes, candidate_bytes))
        elif changed:
            row["grammar_equivalent"] = False
            row["equivalence_kind"] = "runtime_adapter_required"
            row["blockers"] = [
                f"runtime_adapter_required:{parser_name}:{source_section.name}"
            ]
        rows.append(row)
    return rows


def _candidate_pair_blockers(
    *,
    source_ir: PacketIR,
    candidate_ir: PacketIR,
    changed_sections: Sequence[Mapping[str, Any]],
    section_equivalence: Sequence[Mapping[str, Any]],
    source_member_name: str,
    candidate_member_name: str,
) -> list[str]:
    blockers: list[str] = []
    if source_ir.parser_name != candidate_ir.parser_name:
        blockers.append("parser_name_changed")
    if source_member_name != candidate_member_name:
        blockers.append("zip_member_name_changed")
    if source_ir.member_sha256 == candidate_ir.member_sha256:
        blockers.append("candidate_member_sha256_unchanged")
    changed_names = {str(row.get("section_name")) for row in changed_sections}
    for row in section_equivalence:
        section_name = str(row.get("section_name"))
        if row.get("grammar_equivalent") is not True and section_name in changed_names:
            blockers.extend(str(item) for item in row.get("blockers") or [])
            if not row.get("blockers"):
                blockers.append(f"section_not_grammar_equivalent:{section_name}")
    if source_ir.parser_name == PARSER_PR103:
        blockers.extend(_pr103_fixed_layout_blockers(changed_sections))
    return _unique_ordered(blockers)


def _pr103_fixed_layout_blockers(
    changed_sections: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for row in changed_sections:
        name = str(row.get("section_name"))
        if name in PR103_FIXED_LAYOUT_RECODE_SECTIONS and (
            row.get("length_changed") is True or row.get("offset_changed") is True
        ):
            blockers.append(f"fixed_layout_section_length_change_requires_runtime_adapter:{name}")
        if name == "merged_range_coded_weights_and_hi_latents":
            blockers.append("range_stream_change_requires_symbol_roundtrip_proof")
    return blockers


def _requires_brotli_raw_equivalence(parser_name: str, section_name: str) -> bool:
    if parser_name == PARSER_PR106 and section_name in PR106_RUNTIME_RECODE_SECTIONS:
        return True
    if parser_name == PARSER_PR103 and (
        section_name in PR103_FIXED_LAYOUT_RECODE_SECTIONS
        or section_name in PR103_LAST_SECTION_RECODE_SECTIONS
    ):
        return True
    return False


def _brotli_raw_equivalence_record(
    section_name: str,
    source_bytes: bytes,
    candidate_bytes: bytes,
) -> dict[str, Any]:
    try:
        source_raw = brotli.decompress(source_bytes)
        candidate_raw = brotli.decompress(candidate_bytes)
    except brotli.error as exc:
        return {
            "equivalence_kind": "brotli_raw_equivalence",
            "grammar_equivalent": False,
            "raw_equal": False,
            "blockers": [f"brotli_raw_equivalence_unavailable:{section_name}:{exc}"],
        }
    raw_equal = source_raw == candidate_raw
    return {
        "equivalence_kind": "brotli_raw_equivalence",
        "grammar_equivalent": raw_equal,
        "raw_equal": raw_equal,
        "raw_bytes": len(source_raw),
        "source_raw_sha256": sha256_bytes(source_raw),
        "candidate_raw_sha256": sha256_bytes(candidate_raw),
        "blockers": [] if raw_equal else [f"brotli_raw_mismatch:{section_name}"],
    }


def _pr106_header_equivalence(
    source_ir: PacketIR,
    candidate_ir: PacketIR,
    source_header: bytes,
    candidate_header: bytes,
) -> dict[str, Any]:
    blockers: list[str] = []
    source_len = _pr106_header_decoder_len(source_header)
    candidate_len = _pr106_header_decoder_len(candidate_header)
    try:
        source_decoder_len = source_ir.section("decoder_packed_brotli").length
        candidate_decoder_len = candidate_ir.section("decoder_packed_brotli").length
    except PacketSectionTransformError as exc:
        return {
            "equivalence_kind": "pr106_len24_control",
            "grammar_equivalent": False,
            "blockers": [f"pr106_decoder_section_missing:{exc}"],
        }
    if source_len != source_decoder_len:
        blockers.append("source_pr106_len24_mismatches_decoder_section")
    if candidate_len != candidate_decoder_len:
        blockers.append("candidate_pr106_len24_mismatches_decoder_section")
    if len(candidate_header) != 4:
        blockers.append("candidate_pr106_header_length_changed")
    return {
        "equivalence_kind": "pr106_len24_control",
        "grammar_equivalent": not blockers,
        "source_decoder_len24": source_len,
        "candidate_decoder_len24": candidate_len,
        "source_decoder_section_bytes": source_decoder_len,
        "candidate_decoder_section_bytes": candidate_decoder_len,
        "blockers": blockers,
    }


def _pr106_header_decoder_len(header: bytes) -> int:
    if len(header) != 4 or header[:1] != b"\xff":
        return -1
    return int.from_bytes(header[1:4], "little")


def _section_bytes(payload: bytes, section: SectionIR) -> bytes:
    return payload[section.offset : section.offset + section.length]


def _unique_ordered(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _changed_sections(source_ir: PacketIR, candidate_ir: PacketIR | None) -> list[dict[str, Any]]:
    if candidate_ir is None:
        return []
    candidate_by_name = {section.name: section for section in candidate_ir.sections}
    rows: list[dict[str, Any]] = []
    for source_section in source_ir.sections:
        candidate = candidate_by_name.get(source_section.name)
        if candidate is None:
            rows.append(
                {
                    "section_name": source_section.name,
                    "blocker": "candidate_section_missing",
                }
            )
            continue
        changed = (
            source_section.sha256 != candidate.sha256
            or source_section.length != candidate.length
            or source_section.offset != candidate.offset
        )
        if changed:
            content_changed = source_section.sha256 != candidate.sha256
            length_changed = source_section.length != candidate.length
            offset_changed = source_section.offset != candidate.offset
            rows.append(
                {
                    "section_name": source_section.name,
                    "content_changed": content_changed,
                    "length_changed": length_changed,
                    "offset_changed": offset_changed,
                    "source_section_sha256": source_section.sha256,
                    "candidate_section_sha256": candidate.sha256,
                    "source_bytes": source_section.length,
                    "candidate_bytes": candidate.length,
                    "byte_delta": candidate.length - source_section.length,
                    "source_offset": source_section.offset,
                    "candidate_offset": candidate.offset,
                    "optimization_role": source_section.optimization_role,
                }
            )
    return rows


def _scan_one_brotli_section(
    *,
    label: str,
    packet_ir: PacketIR,
    section: SectionIR,
    payload: bytes,
    qualities: Sequence[int],
    lgwins: Sequence[int | None],
    lgblocks: Sequence[int | None],
    jobs: int,
) -> dict[str, Any]:
    base = {
        "label": label,
        "archive_sha256": packet_ir.archive_sha256,
        "archive_bytes": packet_ir.archive_bytes,
        "parser_name": packet_ir.parser_name,
        "section_name": section.name,
        "optimization_role": section.optimization_role,
        "section_offset": section.offset,
        "source_bytes": section.length,
        "source_section_sha256": section.sha256,
        "brotli_decompressible": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if len(payload) != section.length or sha256_bytes(payload) != section.sha256:
        return {
            **base,
            "blockers": [f"section_payload_mismatch:{section.name}"],
        }
    try:
        raw = brotli.decompress(payload)
    except brotli.error:
        return {
            **base,
            "blockers": [f"section_not_brotli_decompressible:{section.name}"],
        }
    try:
        attempt = _generic_brotli_recode_search(
            payload,
            raw,
            qualities=qualities,
            lgwins=lgwins,
            lgblocks=lgblocks,
            jobs=jobs,
        )
    except PacketSectionTransformError as exc:
        return {
            **base,
            "brotli_decompressible": True,
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "blockers": [f"brotli_recode_failed:{section.name}:{exc}"],
        }
    raw_equal = brotli.decompress(attempt["payload"]) == raw
    byte_delta = int(attempt["candidate_bytes"]) - section.length
    rate_positive = byte_delta < 0
    runtime_contract = _runtime_recode_contract(packet_ir.parser_name, section.name)
    candidate_compilable = (
        rate_positive
        and raw_equal
        and runtime_contract["candidate_compilable_by_existing_bridge"] is True
    )
    blockers = []
    if not raw_equal:
        blockers.append(f"brotli_raw_mismatch:{section.name}")
    if not rate_positive:
        blockers.append(f"candidate_section_not_rate_positive:{section.name}")
    if runtime_contract["runtime_adapter_required"]:
        blockers.append(f"runtime_adapter_required:{packet_ir.parser_name}:{section.name}")
    return {
        **base,
        "brotli_decompressible": True,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "candidate_section_sha256": attempt["candidate_section_sha256"],
        "candidate_bytes": attempt["candidate_bytes"],
        "byte_delta": byte_delta,
        "rate_positive": rate_positive,
        "raw_equal": raw_equal,
        "quality": attempt["quality"],
        "lgwin": attempt["lgwin"],
        "lgblock": attempt["lgblock"],
        "runtime_adapter_required": runtime_contract["runtime_adapter_required"],
        "candidate_compilable_by_existing_bridge": candidate_compilable,
        "runtime_contract": runtime_contract["runtime_contract"],
        "ready_for_archive_preflight": candidate_compilable,
        "rate_score_delta_if_components_equal": round(byte_delta * RATE_SCORE_PER_BYTE, 12),
        "blockers": blockers,
    }


def _generic_brotli_recode_search(
    source: bytes,
    raw: bytes,
    *,
    qualities: Sequence[int],
    lgwins: Sequence[int | None],
    lgblocks: Sequence[int | None],
    jobs: int,
) -> dict[str, Any]:
    attempts = [
        (int(quality), None if lgwin is None else int(lgwin), None if lgblock is None else int(lgblock))
        for quality in qualities
        for lgwin in lgwins
        for lgblock in lgblocks
    ]
    attempts = sorted(set(attempts), key=lambda item: (item[0], _optional_sort(item[1]), _optional_sort(item[2])))
    if not attempts:
        raise PacketSectionTransformError("brotli search did not evaluate any variants")
    for quality, lgwin, lgblock in attempts:
        if not 0 <= quality <= 11:
            raise PacketSectionTransformError(f"brotli quality out of range: {quality}")
        if lgwin is not None and not 10 <= lgwin <= 24:
            raise PacketSectionTransformError(f"brotli lgwin out of range: {lgwin}")
        if lgblock is not None and not 16 <= lgblock <= 24:
            raise PacketSectionTransformError(f"brotli lgblock out of range: {lgblock}")
    workers = _bounded_jobs(jobs, len(attempts))
    if workers == 1:
        results = [_generic_brotli_attempt(source, raw, attempt) for attempt in attempts]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda attempt: _generic_brotli_attempt(source, raw, attempt),
                    attempts,
                )
            )
    return min(
        results,
        key=lambda row: (
            int(row["candidate_bytes"]),
            0 if row["candidate_section_sha256"] != sha256_bytes(source) else 1,
            int(row["quality"]),
            _optional_sort(row["lgwin"]),
            _optional_sort(row["lgblock"]),
            str(row["candidate_section_sha256"]),
        ),
    )


def _generic_brotli_attempt(
    source: bytes,
    raw: bytes,
    attempt: tuple[int, int | None, int | None],
) -> dict[str, Any]:
    quality, lgwin, lgblock = attempt
    kwargs = {"quality": quality}
    if lgwin is not None:
        kwargs["lgwin"] = lgwin
    if lgblock is not None:
        kwargs["lgblock"] = lgblock
    candidate = brotli.compress(raw, **kwargs)
    return {
        "payload": candidate,
        "candidate_bytes": len(candidate),
        "candidate_section_sha256": sha256_bytes(candidate),
        "quality": quality,
        "lgwin": lgwin,
        "lgblock": lgblock,
        "changed": candidate != source,
    }


def _runtime_recode_contract(parser_name: str, section_name: str) -> dict[str, Any]:
    if parser_name == PARSER_PR106 and section_name in PR106_RUNTIME_RECODE_SECTIONS:
        return {
            "runtime_adapter_required": False,
            "candidate_compilable_by_existing_bridge": True,
            "runtime_contract": "pr106_len24_header_recomputed_by_packet_section_compiler",
        }
    return {
        "runtime_adapter_required": True,
        "candidate_compilable_by_existing_bridge": False,
        "runtime_contract": "fixed_layout_or_unknown_runtime_requires_adapter_update",
    }


def _bounded_jobs(jobs: int, attempt_count: int) -> int:
    if jobs < 1:
        raise PacketSectionTransformError(f"jobs must be >= 1, got {jobs}")
    return max(1, min(jobs, attempt_count, os.cpu_count() or 1))


def _optional_sort(value: int | None) -> int:
    return -1 if value is None else int(value)


def _optional_json(value: int | None) -> int | None:
    return None if value is None else int(value)


def _transform_summary(
    transform: PacketSectionTransform,
    outputs: Sequence[TransformOutput],
) -> dict[str, Any]:
    return {
        "name": getattr(transform, "name", transform.__class__.__name__),
        "output_count": len(outputs),
        "outputs": [
            {
                "section_name": output.section_name,
                "payload_sha256": sha256_bytes(output.payload),
                "payload_bytes": len(output.payload),
                "metadata": dict(output.metadata),
                "blockers": list(output.blockers),
            }
            for output in outputs
        ],
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PacketSectionTransformError(f"{name} must be an object")
    return value


__all__ = [
    "BrotliRecodeSectionTransform",
    "CompositePacketSectionTransform",
    "CONTEST_CANDIDATE_BLOCKERS",
    "PacketIR",
    "PacketSectionTransform",
    "PacketSectionTransformError",
    "SCHEMA",
    "SectionIR",
    "TransformOutput",
    "build_hnerv_packet_ir",
    "certify_hnerv_grammar_preserving_candidate_pair",
    "compile_hnerv_pr106_section_transform_candidate",
    "scan_hnerv_brotli_recode_opportunities",
]
