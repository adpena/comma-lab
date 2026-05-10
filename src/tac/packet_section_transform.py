"""Typed packet-section transform bridge for contest archive candidates.

This module connects parser-section custody manifests to deterministic packet
rewrites. It is intentionally narrow: no scorer loads, no dispatch, no score
claim, and no hidden sidecars. The first supported compiler path is the public
HNeRV PR106 ``0xff + len24`` single-member grammar because it already has
byte-proved low-level packer support.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import brotli

from tac.analysis.hnerv_packet_sections import (
    PARSER_PR106,
    build_packet_section_manifest,
    validate_packet_section_manifest,
)
from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    brotli_recode_search,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_section_repack import RATE_SCORE_PER_BYTE
from tac.repo_io import sha256_file

SCHEMA = "packet_section_transform.v1"
CONTEST_CANDIDATE_BLOCKERS = (
    "requires_archive_manifest_preflight",
    "requires_lane_dispatch_claim",
    "requires_exact_cuda_auth_eval",
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
    single = read_strict_single_member_zip(source_path)
    packed = parse_ff_packed_brotli_hnerv(single.payload)
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

    candidate_payload = single.payload
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
        candidate_payload = candidate_packed.to_bytes()
        write_stored_single_member_zip(
            output_path,
            member_name=single.member_name,
            payload=candidate_payload,
        )
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
    "compile_hnerv_pr106_section_transform_candidate",
]
