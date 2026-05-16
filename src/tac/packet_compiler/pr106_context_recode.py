# SPDX-License-Identifier: MIT
"""PR106 section-aware higher-order context recode planning.

This module is intentionally a PacketIR investigation surface, not an exact
eval actuator.  It parses PR106/HNeRV payload sections, measures section-local
empirical context floors for orders >= 2, and can emit a lossless prototype
section envelope.  The prototype proves target bytes changed and round-trip
back to the source section; it is still blocked from dispatch until an inflate
runtime decoder and same-runtime parity proof exist.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import PackedHnervPayload, parse_ff_packed_brotli_hnerv
from tac.lossless.range_coder import RangeDecoder, RangeEncoder, cumulative_frequencies
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_MAGIC,
    StoredZipMember,
    canonical_expected_sha256,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
    sha256_hex,
)

CONTEXT_RECODE_MAGIC = b"PCR1"
CONTEXT_RECODE_VERSION = 1
CONTEXT_RECODE_HEADER = "<4sBBBBIII32s"
CONTEXT_RECODE_HEADER_BYTES = struct.calcsize(CONTEXT_RECODE_HEADER)
HIGH_ORDER_MIN_CONTEXT_ORDER = 2
DEFAULT_CONTEXT_ORDERS = (0, 1, 2, 3, 4)
TARGETABLE_INNER_SECTIONS = (
    "decoder_packed_brotli",
    "latents_and_sidecar_brotli",
)
ALL_INNER_SECTIONS = (
    "packed_header_ff_len24",
    *TARGETABLE_INNER_SECTIONS,
)
ALWAYS_BLOCKERS = (
    "prototype_runtime_decoder_not_integrated",
    "full_frame_same_runtime_parity_missing",
    "exact_cuda_auth_eval_missing",
    "contest_auth_eval_adjudication_missing",
)


class PR106ContextRecodeError(ValueError):
    """Raised when a PR106 context-recode input is malformed or unsafe."""


@dataclass(frozen=True)
class SectionView:
    """Byte section in the PR106 PacketIR view."""

    name: str
    role: str
    offset: int
    data: bytes
    targetable: bool

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "offset": self.offset,
            "bytes": len(self.data),
            "sha256": sha256_hex(self.data),
            "targetable": self.targetable,
        }


@dataclass(frozen=True)
class PR106ContextSource:
    """Parsed PR106 payload plus source custody."""

    source: dict[str, Any]
    member: StoredZipMember | None
    payload: bytes
    inner_payload: bytes
    packed: PackedHnervPayload
    sections: tuple[SectionView, ...]
    wrapper: dict[str, Any] | None = None

    def section(self, name: str) -> SectionView:
        for section in self.sections:
            if section.name == name:
                return section
        names = ", ".join(section.name for section in self.sections)
        raise PR106ContextRecodeError(f"unknown PR106 section {name!r}; available: {names}")


@dataclass(frozen=True)
class ContextRecodeSection:
    """Lossless prototype envelope for one target section."""

    section_name: str
    context_order: int
    source_bytes: bytes
    encoded_bytes: bytes
    decoded_bytes: bytes
    prefix_bytes: int
    model_bytes: int
    range_stream_bytes: int
    context_count: int
    context_edge_count: int

    def manifest(self) -> dict[str, Any]:
        source_sha = sha256_hex(self.source_bytes)
        encoded_sha = sha256_hex(self.encoded_bytes)
        decoded_sha = sha256_hex(self.decoded_bytes)
        changed = encoded_sha != source_sha
        roundtrip = decoded_sha == source_sha
        blockers: list[str] = []
        if self.context_order < HIGH_ORDER_MIN_CONTEXT_ORDER:
            blockers.append("context_order_not_high_order")
        if self.context_order == 0:
            blockers.append("zero_order_arithmetic_control_falsified_not_candidate")
        if not changed:
            blockers.append("target_section_bytes_unchanged")
            blockers.append("no_op_detector_failed")
        if not roundtrip:
            blockers.append("context_recode_roundtrip_failed")
        if len(self.encoded_bytes) >= len(self.source_bytes):
            blockers.append("prototype_not_rate_positive_after_model_overhead")
        blockers.extend(ALWAYS_BLOCKERS)
        return {
            "section_name": self.section_name,
            "context_order": self.context_order,
            "codec": "static_section_local_context_range_prototype_v1",
            "source_section_bytes": len(self.source_bytes),
            "encoded_section_bytes": len(self.encoded_bytes),
            "delta_bytes_vs_source_section": len(self.encoded_bytes) - len(self.source_bytes),
            "source_section_sha256": source_sha,
            "encoded_section_sha256": encoded_sha,
            "decoded_section_sha256": decoded_sha,
            "target_section_bytes_changed": changed,
            "lossless_roundtrip_proven": roundtrip,
            "no_op_detector_passed": changed and roundtrip,
            "prefix_bytes": self.prefix_bytes,
            "context_model_bytes": self.model_bytes,
            "range_stream_bytes": self.range_stream_bytes,
            "context_count": self.context_count,
            "context_edge_count": self.context_edge_count,
            "runtime_consumption_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": blockers,
        }


@dataclass(frozen=True)
class PR106ContextRecodeBuildResult:
    """Report plus optional prototype section bytes for CLI materialization."""

    report: dict[str, Any]
    prototype_section_bytes: bytes | None


def load_pr106_context_source_from_archive(path: str | Path) -> PR106ContextSource:
    archive = Path(path)
    archive_bytes = archive.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    return parse_pr106_context_source(
        member.payload,
        member=member,
        source={
            "mode": "archive",
            "path": str(archive),
            "archive_bytes": len(archive_bytes),
            "archive_sha256": sha256_hex(archive_bytes),
            "member_name": member.name,
            "member_bytes": len(member.payload),
            "member_sha256": sha256_hex(member.payload),
            "zip_overhead_bytes": len(archive_bytes) - len(member.payload),
        },
    )


def load_pr106_context_source_from_payload(path: str | Path) -> PR106ContextSource:
    payload_path = Path(path)
    payload = payload_path.read_bytes()
    return parse_pr106_context_source(
        payload,
        source={
            "mode": "payload",
            "path": str(payload_path),
            "payload_bytes": len(payload),
            "payload_sha256": sha256_hex(payload),
        },
    )


def parse_pr106_context_source(
    payload: bytes,
    *,
    source: dict[str, Any] | None = None,
    member: StoredZipMember | None = None,
) -> PR106ContextSource:
    """Parse a PR106 PacketIR payload and expose section-local byte views."""

    if not payload:
        raise PR106ContextRecodeError("empty PR106 payload")
    base_source = dict(source or {})
    wrapper: dict[str, Any] | None = None
    inner = payload
    if payload[0] == PR106_SIDECAR_MAGIC:
        packet = parse_pr106_sidecar_packet(payload)
    elif payload[0] != 0xFF:
        try:
            packet = parse_pr106_sidecar_packet(payload)
        except ValueError:
            packet = None
    else:
        packet = None
    if packet is not None:
        inner = packet.pr106_bytes
        wrapper = {
            "format_id": f"0x{packet.format_id:02X}",
            "sidecar_kind": packet.sidecar_kind,
            "outer_payload_bytes": len(payload),
            "outer_payload_sha256": sha256_hex(payload),
            "inner_pr106_bytes": len(inner),
            "inner_pr106_sha256": sha256_hex(inner),
            "sidecar_payload_bytes": len(packet.sidecar_payload),
            "sidecar_payload_sha256": sha256_hex(packet.sidecar_payload),
            "framing_meta_bytes": 0 if packet.framing_meta is None else len(packet.framing_meta),
            "framing_meta_sha256": None
            if packet.framing_meta is None
            else sha256_hex(packet.framing_meta),
        }
        base_source["wrapper_unwrapped_for_section_context_model"] = True

    try:
        packed = parse_ff_packed_brotli_hnerv(inner)
    except Exception as exc:  # pragma: no cover - exception type varies by parser path.
        raise PR106ContextRecodeError(f"payload is not PR106 ff-packed HNeRV: {exc}") from exc

    decoder_offset = len(packed.header)
    latent_offset = decoder_offset + len(packed.decoder_packed_brotli)
    sections = [
        SectionView(
            "packed_header_ff_len24",
            "control_or_metadata",
            0,
            packed.header,
            False,
        ),
        SectionView(
            "decoder_packed_brotli",
            "decoder_weight_stream",
            decoder_offset,
            packed.decoder_packed_brotli,
            True,
        ),
        SectionView(
            "latents_and_sidecar_brotli",
            "latent_stream",
            latent_offset,
            packed.latents_and_sidecar_brotli,
            True,
        ),
    ]
    if wrapper is not None:
        packet = parse_pr106_sidecar_packet(payload)
        sections.append(
            SectionView(
                "wrapper_sidecar_payload",
                "sidecar_or_correction_stream",
                len(payload) - len(packet.sidecar_payload),
                packet.sidecar_payload,
                False,
            )
        )
    return PR106ContextSource(
        source=base_source,
        member=member,
        payload=payload,
        inner_payload=inner,
        packed=packed,
        sections=tuple(sections),
        wrapper=wrapper,
    )


def build_pr106_context_recode_report(
    source: PR106ContextSource,
    *,
    target_section: str = "auto",
    context_order: int = 2,
    context_orders: tuple[int, ...] = DEFAULT_CONTEXT_ORDERS,
    build_prototype: bool = True,
) -> PR106ContextRecodeBuildResult:
    """Build a measured section-aware context-recode report."""

    profiles = [profile_section(section, context_orders=context_orders) for section in source.sections]
    selected = (
        _select_auto_target(profiles)
        if target_section == "auto"
        else _find_profile(profiles, target_section)
    )
    prototype_bytes: bytes | None = None
    candidate_manifest: dict[str, Any] | None = None
    candidate_blockers: list[str] = []
    if selected is None:
        candidate_blockers.append("target_section_not_found")
    elif selected["section_name"] not in TARGETABLE_INNER_SECTIONS:
        candidate_blockers.append("target_section_not_targetable_in_pr106_inner_payload")
    elif not build_prototype:
        candidate_blockers.append("prototype_not_built")
    else:
        section = source.section(str(selected["section_name"]))
        prototype = encode_context_recode_section(
            section.name,
            section.data,
            context_order=context_order,
        )
        prototype_bytes = prototype.encoded_bytes
        candidate_manifest = prototype.manifest()
        candidate_blockers = list(candidate_manifest["blockers"])

    exact_blockers = list(dict.fromkeys([*candidate_blockers, *ALWAYS_BLOCKERS]))
    report = {
        "schema": "pr106_context_recode_profile_v1",
        "proof_scope": (
            "section_aware_high_order_context_entropy_profile_and_lossless_"
            "prototype_section_transform_not_runtime_inflate_not_score"
        ),
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "zero_order_arithmetic_baseline_verdict": (
            "falsified_control_not_candidate_high_order_context_required"
        ),
        "source": {
            **source.source,
            "payload_bytes": len(source.payload),
            "payload_sha256": sha256_hex(source.payload),
            "inner_pr106_bytes": len(source.inner_payload),
            "inner_pr106_sha256": sha256_hex(source.inner_payload),
            "wrapper": source.wrapper,
        },
        "sections": [section.manifest() for section in source.sections],
        "section_context_profiles": profiles,
        "selected_target": selected,
        "prototype_candidate": candidate_manifest,
        "readiness_blockers": exact_blockers,
        "dispatch_blockers": exact_blockers,
    }
    return PR106ContextRecodeBuildResult(report=report, prototype_section_bytes=prototype_bytes)


def emit_pr106_context_source_payload(source: PR106ContextSource) -> bytes:
    """Re-emit the payload consumed by the PR106 context view.

    This is identity plumbing for PacketIR custody. It does not emit a context
    recode candidate and it never claims runtime or score consumption.
    """

    inner_payload = source.packed.to_bytes()
    if source.wrapper is None:
        return inner_payload
    packet = parse_pr106_sidecar_packet(source.payload)
    return emit_pr106_sidecar_packet(replace(packet, pr106_bytes=inner_payload))


def prove_pr106_context_source_identity(
    source: PR106ContextSource,
) -> dict[str, object]:
    """Prove context-view parse/re-emit identity for a parsed PR106 source."""

    emitted_inner = source.packed.to_bytes()
    emitted_payload = emit_pr106_context_source_payload(source)
    blockers: list[str] = []
    if emitted_inner != source.inner_payload:
        blockers.append("context_inner_payload_parse_emit_not_identity")
    if emitted_payload != source.payload:
        blockers.append("context_payload_parse_emit_not_identity")

    return {
        "schema": "pr106_context_source_identity_proof_v1",
        "proof_scope": (
            "context_recode_source_parse_emit_identity_not_runtime_inflate_not_score"
        ),
        "source": {
            **source.source,
            "payload_bytes": len(source.payload),
            "payload_sha256": sha256_hex(source.payload),
            "inner_pr106_bytes": len(source.inner_payload),
            "inner_pr106_sha256": sha256_hex(source.inner_payload),
            "wrapper": source.wrapper,
        },
        "sections": [section.manifest() for section in source.sections],
        "emitted_inner_payload": {
            "bytes": len(emitted_inner),
            "sha256": sha256_hex(emitted_inner),
            "byte_identical_to_source_inner": emitted_inner == source.inner_payload,
        },
        "emitted_payload": {
            "bytes": len(emitted_payload),
            "sha256": sha256_hex(emitted_payload),
            "byte_identical_to_source_payload": emitted_payload == source.payload,
        },
        "context_packet_ir_identity_passed": not blockers,
        "blockers": blockers,
        "proof_not_score": True,
        "evidence_axis": "packet-ir-context-parser-local-no-score",
        "runtime_consumption_claim": False,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next_proof": (
            "prototype runtime decoder integration plus same-runtime full-frame "
            "parity and exact contest auth eval before score language"
        ),
    }


def prove_pr106_context_archive_identity(
    *,
    archive_path: str | Path,
    expected_member_name: str | None = None,
    expected_archive_sha256: str | None = None,
) -> dict[str, object]:
    """Prove context-view parse/re-emit identity for a single-member archive."""

    archive = Path(archive_path)
    archive_bytes = archive.read_bytes()
    archive_sha = sha256_hex(archive_bytes)
    expected_archive_sha, expected_archive_sha_well_formed = canonical_expected_sha256(
        expected_archive_sha256
    )
    member = read_single_stored_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    source = parse_pr106_context_source(
        member.payload,
        member=member,
        source={
            "mode": "archive",
            "path": str(archive),
            "archive_bytes": len(archive_bytes),
            "archive_sha256": archive_sha,
            "member_name": member.name,
            "member_bytes": len(member.payload),
            "member_sha256": sha256_hex(member.payload),
            "zip_overhead_bytes": len(archive_bytes) - len(member.payload),
        },
    )
    source_proof = prove_pr106_context_source_identity(source)
    emitted_payload = emit_pr106_context_source_payload(source)
    emitted_archive = emit_single_stored_member_archive(
        replace(member, payload=emitted_payload)
    )
    blockers = list(source_proof["blockers"])
    if expected_archive_sha_well_formed is False:
        blockers.append("expected_archive_sha256_malformed")
    if (
        expected_archive_sha_well_formed is True
        and expected_archive_sha is not None
        and archive_sha != expected_archive_sha
    ):
        blockers.append("expected_archive_sha256_mismatch")
    if emitted_archive != archive_bytes:
        blockers.append("context_single_member_zip_parse_emit_not_identity")

    proof = dict(source_proof)
    proof.update(
        {
            "archive": {
                "path": archive.as_posix(),
                "bytes": len(archive_bytes),
                "sha256": archive_sha,
                "expected_sha256": expected_archive_sha,
                "expected_sha256_well_formed": expected_archive_sha_well_formed,
                "expected_sha256_matches": (
                    None
                    if expected_archive_sha is None
                    or expected_archive_sha_well_formed is False
                    else archive_sha == expected_archive_sha
                ),
            },
            "member": {
                "name": member.name,
                "expected_name": expected_member_name,
                "expected_name_matches": (
                    None if expected_member_name is None else member.name == expected_member_name
                ),
                "payload_bytes": len(member.payload),
                "payload_sha256": sha256_hex(member.payload),
            },
            "emitted_archive": {
                "bytes": len(emitted_archive),
                "sha256": sha256_hex(emitted_archive),
                "byte_identical_to_source_archive": emitted_archive == archive_bytes,
            },
            "context_packet_ir_identity_passed": not blockers,
            "blockers": blockers,
        }
    )
    return proof


def profile_section(
    section: SectionView,
    *,
    context_orders: tuple[int, ...] = DEFAULT_CONTEXT_ORDERS,
) -> dict[str, Any]:
    rows = [
        _context_floor_row(section.data, order=order)
        for order in context_orders
    ]
    high_order_rows = [
        row for row in rows if int(row["context_order"]) >= HIGH_ORDER_MIN_CONTEXT_ORDER
    ]
    best = min(high_order_rows or rows, key=lambda row: int(row["floor_bytes"]))
    return {
        "section_name": section.name,
        "role": section.role,
        "targetable": section.targetable,
        "current_bytes": len(section.data),
        "sha256": sha256_hex(section.data),
        "floors": rows,
        "best_high_order_context_order": best["context_order"],
        "best_high_order_floor_bytes": best["floor_bytes"],
        "best_high_order_delta_vs_current_bytes": best["delta_vs_current_bytes"],
        "limitations": [
            "empirical_context_floor_only",
            "model_table_bytes_not_charged_in_floor_rows",
            "prototype_runtime_decoder_not_integrated",
            "no_exact_cuda_score_claim",
        ],
    }


def encode_context_recode_section(
    section_name: str,
    section_bytes: bytes,
    *,
    context_order: int,
) -> ContextRecodeSection:
    """Encode one section into a lossless static-context prototype envelope."""

    if not (0 <= context_order <= 8):
        raise PR106ContextRecodeError("context_order must be in range 0..8")
    if len(section_name.encode("utf-8")) > 255:
        raise PR106ContextRecodeError("section name too long for prototype header")

    prefix_len = min(context_order, len(section_bytes))
    prefix = section_bytes[:prefix_len]
    model = _build_context_model(section_bytes, context_order)
    model_bytes = _encode_context_model(model, context_order)
    stream = _encode_range_stream(section_bytes, context_order, model)
    name_bytes = section_name.encode("utf-8")
    header = struct.pack(
        CONTEXT_RECODE_HEADER,
        CONTEXT_RECODE_MAGIC,
        CONTEXT_RECODE_VERSION,
        context_order,
        len(name_bytes),
        prefix_len,
        len(section_bytes),
        len(model_bytes),
        len(stream),
        hashlib.sha256(section_bytes).digest(),
    )
    encoded = header + name_bytes + prefix + model_bytes + stream
    decoded = decode_context_recode_section(encoded)
    stats = _context_model_stats(model)
    return ContextRecodeSection(
        section_name=section_name,
        context_order=context_order,
        source_bytes=section_bytes,
        encoded_bytes=encoded,
        decoded_bytes=decoded,
        prefix_bytes=len(prefix),
        model_bytes=len(model_bytes),
        range_stream_bytes=len(stream),
        context_count=stats["contexts"],
        context_edge_count=stats["edges"],
    )


def decode_context_recode_section(encoded: bytes) -> bytes:
    """Decode a ``PCR1`` prototype section envelope."""

    if len(encoded) < CONTEXT_RECODE_HEADER_BYTES:
        raise PR106ContextRecodeError("context recode envelope truncated before header")
    (
        magic,
        version,
        order,
        name_len,
        prefix_len,
        source_len,
        model_len,
        stream_len,
        source_digest,
    ) = struct.unpack_from(CONTEXT_RECODE_HEADER, encoded, 0)
    if magic != CONTEXT_RECODE_MAGIC:
        raise PR106ContextRecodeError("context recode envelope magic mismatch")
    if version != CONTEXT_RECODE_VERSION:
        raise PR106ContextRecodeError(f"unsupported context recode version: {version}")
    pos = CONTEXT_RECODE_HEADER_BYTES
    end_name = pos + name_len
    end_prefix = end_name + prefix_len
    end_model = end_prefix + model_len
    end_stream = end_model + stream_len
    if end_stream != len(encoded):
        raise PR106ContextRecodeError(
            f"context recode envelope length mismatch: parsed={end_stream} total={len(encoded)}"
        )
    _section_name = encoded[pos:end_name].decode("utf-8")
    prefix = encoded[end_name:end_prefix]
    model = _decode_context_model(encoded[end_prefix:end_model], order)
    stream = encoded[end_model:end_stream]
    decoded = _decode_range_stream(
        prefix,
        source_len=source_len,
        context_order=order,
        model=model,
        stream=stream,
    )
    if hashlib.sha256(decoded).digest() != source_digest:
        raise PR106ContextRecodeError("context recode decoded sha256 mismatch")
    return decoded


def write_report_json(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_markdown(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = report.get("selected_target") or {}
    candidate = report.get("prototype_candidate") or {}
    lines = [
        "# PR106 Context Recode Profile",
        "",
        f"- score_claim: `{str(report.get('score_claim')).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(report.get('ready_for_exact_eval_dispatch')).lower()}`",
        f"- zero_order_arithmetic_baseline_verdict: `{report.get('zero_order_arithmetic_baseline_verdict')}`",
        f"- selected_target: `{selected.get('section_name')}`",
        f"- context_order: `{candidate.get('context_order')}`",
        f"- no_op_detector_passed: `{str(candidate.get('no_op_detector_passed')).lower()}`",
        f"- dispatch_blockers: `{', '.join(str(item) for item in report.get('dispatch_blockers', []))}`",
        "",
        "## Section Floors",
        "",
        "| section | bytes | best order | best floor bytes | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in report.get("section_context_profiles", []):
        lines.append(
            "| {section} | {current} | {order} | {floor} | {delta} |".format(
                section=profile.get("section_name"),
                current=profile.get("current_bytes"),
                order=profile.get("best_high_order_context_order"),
                floor=profile.get("best_high_order_floor_bytes"),
                delta=profile.get("best_high_order_delta_vs_current_bytes"),
            )
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _context_floor_row(data: bytes, *, order: int) -> dict[str, Any]:
    if order < 0:
        raise PR106ContextRecodeError("context order must be non-negative")
    bits = context_entropy_bits(data, order=order)
    floor_bytes = math.ceil(bits / 8.0) if bits > 0 else 0
    stats = context_stats(data, order=order)
    return {
        "context_order": order,
        "model_class": "iid_zero_order_control" if order == 0 else f"section_local_order_{order}",
        "source_symbols": len(data),
        "floor_bits": round(bits, 6),
        "floor_bytes": floor_bytes,
        "bits_per_symbol": round(bits / len(data), 12) if data else 0.0,
        "delta_vs_current_bytes": floor_bytes - len(data),
        "contexts_unpriced": stats["contexts"],
        "context_edges_unpriced": stats["edges"],
        "limitation": (
            "zero_order_arithmetic_control_not_candidate"
            if order == 0
            else "context_model_table_not_charged"
        ),
    }


def context_entropy_bits(data: bytes, *, order: int) -> float:
    if order < 0:
        raise PR106ContextRecodeError("context order must be non-negative")
    if not data:
        return 0.0
    values = list(data)
    marginal = _entropy_bits(Counter(values))
    if order == 0 or len(values) <= order:
        return len(values) * marginal
    prefix_bits = order * marginal
    contexts: dict[bytes, Counter[int]] = {}
    for index in range(order, len(values)):
        ctx = data[index - order : index]
        contexts.setdefault(ctx, Counter())[values[index]] += 1
    conditional_bits = 0.0
    for counter in contexts.values():
        conditional_bits += sum(counter.values()) * _entropy_bits(counter)
    return prefix_bits + conditional_bits


def context_stats(data: bytes, *, order: int) -> dict[str, int]:
    if order <= 0 or len(data) <= order:
        return {"contexts": 1 if data else 0, "edges": len(set(data))}
    contexts: dict[bytes, set[int]] = {}
    for index in range(order, len(data)):
        contexts.setdefault(data[index - order : index], set()).add(data[index])
    return {
        "contexts": len(contexts),
        "edges": sum(len(symbols) for symbols in contexts.values()),
    }


def _entropy_bits(counter: Counter[int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    bits = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = count / total
        bits -= p * math.log2(p)
    return bits


def _build_context_model(data: bytes, order: int) -> dict[bytes, Counter[int]]:
    if len(data) <= order:
        return {}
    model: dict[bytes, Counter[int]] = {}
    for index in range(order, len(data)):
        model.setdefault(data[index - order : index], Counter())[data[index]] += 1
    return model


def _context_model_stats(model: dict[bytes, Counter[int]]) -> dict[str, int]:
    return {
        "contexts": len(model),
        "edges": sum(len(counter) for counter in model.values()),
    }


def _encode_range_stream(
    data: bytes,
    order: int,
    model: dict[bytes, Counter[int]],
) -> bytes:
    if len(data) <= order:
        return b""
    tables = _context_tables(model)
    encoder = RangeEncoder()
    for index in range(order, len(data)):
        ctx = data[index - order : index]
        symbols, cumulative, total, symbol_to_index = tables[ctx]
        symbol_index = symbol_to_index[data[index]]
        encoder.encode(symbol=symbol_index, cumulative=cumulative, total=total)
    return encoder.finish()


def _decode_range_stream(
    prefix: bytes,
    *,
    source_len: int,
    context_order: int,
    model: dict[bytes, Counter[int]],
    stream: bytes,
) -> bytes:
    if source_len < len(prefix):
        raise PR106ContextRecodeError("source length shorter than prefix")
    if source_len == len(prefix):
        return prefix
    if len(prefix) != context_order:
        raise PR106ContextRecodeError("prefix length must equal context order for non-empty stream")
    if not stream:
        raise PR106ContextRecodeError("context range stream is empty")
    tables = _context_tables(model)
    decoder = RangeDecoder(stream)
    out = bytearray(prefix)
    while len(out) < source_len:
        ctx = bytes(out[-context_order:]) if context_order else b""
        if ctx not in tables:
            raise PR106ContextRecodeError("missing context model row during decode")
        symbols, cumulative, total, _symbol_to_index = tables[ctx]
        scaled = decoder.target(total)
        symbol_index = bisect_right(cumulative, scaled) - 1
        if symbol_index < 0 or symbol_index >= len(symbols):
            raise PR106ContextRecodeError("range stream symbol outside context row")
        decoder.update(
            low_count=cumulative[symbol_index],
            high_count=cumulative[symbol_index + 1],
            total=total,
        )
        out.append(symbols[symbol_index])
    return bytes(out)


def _context_tables(
    model: dict[bytes, Counter[int]]
) -> dict[bytes, tuple[tuple[int, ...], list[int], int, dict[int, int]]]:
    tables: dict[bytes, tuple[tuple[int, ...], list[int], int, dict[int, int]]] = {}
    for ctx, counter in model.items():
        pairs = sorted((int(symbol), int(count)) for symbol, count in counter.items())
        symbols = tuple(symbol for symbol, _count in pairs)
        frequencies = [count for _symbol, count in pairs]
        cumulative, total = cumulative_frequencies(frequencies)
        tables[ctx] = (
            symbols,
            cumulative,
            total,
            {symbol: index for index, symbol in enumerate(symbols)},
        )
    return tables


def _encode_context_model(model: dict[bytes, Counter[int]], order: int) -> bytes:
    out = bytearray()
    _write_varuint(out, len(model))
    for ctx in sorted(model):
        if len(ctx) != order:
            raise PR106ContextRecodeError("context length does not match order")
        out.extend(ctx)
        pairs = sorted((int(symbol), int(count)) for symbol, count in model[ctx].items())
        _write_varuint(out, len(pairs))
        for symbol, count in pairs:
            out.append(symbol)
            _write_varuint(out, count)
    return bytes(out)


def _decode_context_model(payload: bytes, order: int) -> dict[bytes, Counter[int]]:
    pos = 0
    context_count, pos = _read_varuint(payload, pos)
    model: dict[bytes, Counter[int]] = {}
    for _ in range(context_count):
        end_ctx = pos + order
        if end_ctx > len(payload):
            raise PR106ContextRecodeError("context model truncated in context bytes")
        ctx = payload[pos:end_ctx]
        pos = end_ctx
        symbol_count, pos = _read_varuint(payload, pos)
        counter: Counter[int] = Counter()
        for _symbol_index in range(symbol_count):
            if pos >= len(payload):
                raise PR106ContextRecodeError("context model truncated in symbol")
            symbol = payload[pos]
            pos += 1
            count, pos = _read_varuint(payload, pos)
            if count <= 0:
                raise PR106ContextRecodeError("context model count must be positive")
            counter[symbol] = count
        model[ctx] = counter
    if pos != len(payload):
        raise PR106ContextRecodeError("context model trailing bytes")
    return model


def _write_varuint(out: bytearray, value: int) -> None:
    if value < 0:
        raise PR106ContextRecodeError("varuint value must be non-negative")
    current = value
    while current >= 0x80:
        out.append((current & 0x7F) | 0x80)
        current >>= 7
    out.append(current)


def _read_varuint(payload: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if pos >= len(payload):
            raise PR106ContextRecodeError("truncated varuint")
        byte = payload[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 63:
            raise PR106ContextRecodeError("varuint too large")


def _select_auto_target(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        profile
        for profile in profiles
        if profile.get("section_name") in TARGETABLE_INNER_SECTIONS
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            int(item.get("best_high_order_delta_vs_current_bytes", 0)),
            -int(item.get("current_bytes", 0)),
        ),
    )


def _find_profile(
    profiles: list[dict[str, Any]],
    target_section: str,
) -> dict[str, Any] | None:
    for profile in profiles:
        if profile.get("section_name") == target_section:
            return profile
    return None


def build_stored_zip_for_tests(member_name: str, payload: bytes) -> bytes:
    """Small test helper for synthetic single-member PR106 archives."""

    out = io.BytesIO()
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(info, payload)
    return out.getvalue()
