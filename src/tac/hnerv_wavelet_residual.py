# SPDX-License-Identifier: MIT
"""Planning-only wavelet residual atoms for HNeRV payload sections.

The wavelet hidden gem is useful only if it produces charged, auditable bytes.
This module therefore starts from exact HNeRV section custody, computes a
deterministic Haar atom plan over section bytes, and records the old-section
SHA proof required before any future candidate archive can be considered.

It does not rewrite archives, load scorers, or claim a score.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import brotli

from tac.hnerv_lowlevel_packer import (
    DEFAULT_WAVELET_SECTION,
    REPACKABLE_SECTIONS,
    WAVELET_AUDIT_SECTIONS,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)
from tac.hnerv_section_repack import HnervSectionPlanError

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_wavelet_residual.build_wavelet_residual_plan"

# Adversarial review 2026-05-06 (BUG #5): import shared constant instead of
# hardcoding the literal — single source of truth lives in hnerv_lowlevel_packer.
DEFAULT_TARGET_SECTIONS = (DEFAULT_WAVELET_SECTION,)


class HnervWaveletResidualError(ValueError):
    """Raised when a wavelet residual plan input is invalid."""


def build_wavelet_residual_plan(
    *,
    source_archive: str,
    scorecard: Mapping[str, Any],
    source_label: str,
    target_sections: Sequence[str] = DEFAULT_TARGET_SECTIONS,
    top_k: int = 32,
    block_size: int = 64,
    quant_step: float = 1.0,
) -> dict[str, Any]:
    """Build a deterministic planning-only Haar residual atom manifest."""

    if top_k <= 0:
        raise HnervWaveletResidualError(f"top_k must be positive, got {top_k}")
    if block_size < 2 or block_size & (block_size - 1):
        raise HnervWaveletResidualError(f"block_size must be a power of two >= 2, got {block_size}")
    if quant_step <= 0:
        raise HnervWaveletResidualError(f"quant_step must be positive, got {quant_step}")

    archive = read_strict_single_member_zip(source_archive)
    packed = parse_ff_packed_brotli_hnerv(archive.payload)
    manifest = _manifest_for_label(scorecard, source_label)
    source_blockers = _audit_source_manifest(archive, packed, manifest)

    selected_sections = tuple(dict.fromkeys(str(section) for section in target_sections))
    section_records: list[dict[str, Any]] = []
    plan_blockers = list(source_blockers)
    for section_name in selected_sections:
        try:
            section_bytes = packed.section_bytes(section_name)
        except Exception:
            plan_blockers.append(f"target_section_missing_or_unknown:{section_name}")
            continue
        if section_name == "packed_header_ff_len24":
            plan_blockers.append(f"target_section_too_small_for_wavelet:{section_name}")
            continue
        try:
            raw, transform_domain = _raw_transform_bytes(section_name, section_bytes)
        except HnervWaveletResidualError as exc:
            plan_blockers.append(f"target_section_not_transformable:{section_name}:{exc}")
            continue
        atoms = _top_haar_atoms(raw, section_name=section_name, top_k=top_k, block_size=block_size, quant_step=quant_step)
        if not atoms:
            plan_blockers.append(f"no_nonzero_wavelet_atoms:{section_name}")
        section_records.append(
            {
                "section_name": section_name,
                "source_section_bytes": len(section_bytes),
                "source_section_sha256": sha256_bytes(section_bytes),
                "transform_domain": transform_domain,
                "raw_bytes": len(raw),
                "raw_sha256": sha256_bytes(raw),
                "block_size": block_size,
                "quant_step": quant_step,
                "atom_count": len(atoms),
                "estimated_atom_bytes": sum(int(atom["estimated_wire_bytes"]) for atom in atoms),
                "atoms": atoms,
                "score_claim": False,
            }
        )

    if not section_records:
        plan_blockers.append("no_wavelet_sections_planned")

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "source_archive_path": str(source_archive),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "target_sections": list(selected_sections),
        "ready_for_wavelet_candidate_build": not plan_blockers,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": plan_blockers,
        "dispatch_blockers": [
            "planning_only_wavelet_atoms",
            "requires_candidate_archive_consuming_atoms",
            "requires_old_new_section_sha256_proof",
            "requires_archive_manifest_preflight",
            "requires_exact_cuda_auth_eval",
        ],
        "candidate_required_proof": [
            "source_archive_sha256",
            "candidate_archive_sha256",
            "source_payload_sha256",
            "candidate_payload_sha256",
            "source_section_sha256",
            "candidate_section_sha256",
            "source_bytes",
            "candidate_bytes",
            "runtime_consumes_wavelet_atoms",
        ],
        "total_estimated_atom_bytes": sum(int(row["estimated_atom_bytes"]) for row in section_records),
        "sections": section_records,
    }


def _raw_transform_bytes(section_name: str, section_bytes: bytes) -> tuple[bytes, str]:
    if section_name in REPACKABLE_SECTIONS:
        try:
            return brotli.decompress(section_bytes), "brotli_decompressed_section"
        except brotli.error as exc:
            raise HnervWaveletResidualError("brotli decompress failed") from exc
    return section_bytes, "section_wire_bytes"


def _top_haar_atoms(
    raw: bytes,
    *,
    section_name: str,
    top_k: int,
    block_size: int,
    quant_step: float,
) -> list[dict[str, Any]]:
    """Compute top-K Haar wavelet atoms for the byte stream.

    Round 2A R2-4 note (2026-05-06, 85% confidence): the final partial block
    is zero-padded to `block_size` before the Haar transform runs. Atoms whose
    coefficient was computed using padded zeros (i.e. the last partial block,
    if any) are SEMANTICALLY UNSOUND for round-trip — they encode coefficients
    that depend on the implicit zero neighbor, not on a real byte value. The
    `support_end = min(len(raw), support_start + support)` guard prevents
    out-of-bounds writes, but it does not prevent the planner from generating
    atoms whose value is influenced by padding. We FILTER these atoms below by
    requiring the atom's full support window fit entirely within the real
    (un-padded) byte range. This drops the boundary atoms cleanly.

    The previous behavior was best-effort: boundary atoms could pass through
    and contribute byte-level deltas computed against an implicit zero. For a
    score-relevant sidechannel that's a correctness gap; per CLAUDE.md byte-
    determinism non-negotiable we filter them at plan-time.
    """
    atoms: list[dict[str, Any]] = []
    if not raw:
        return atoms
    raw_len = len(raw)
    for block_start in range(0, raw_len, block_size):
        chunk = raw[block_start : block_start + block_size]
        chunk_len = len(chunk)
        # Round 2A R2-4 fix: skip the final partial block entirely. Its atoms
        # depend on zero padding which is not present in the un-padded bytes.
        if chunk_len < block_size:
            continue
        centered = [float(byte) - 128.0 for byte in chunk]
        centered.extend([0.0] * (block_size - len(centered)))
        width = block_size
        level = 0
        current = centered
        while width >= 2:
            next_level: list[float] = []
            for pair_index in range(width // 2):
                left = current[2 * pair_index]
                right = current[2 * pair_index + 1]
                avg = (left + right) * 0.5
                detail = (left - right) * 0.5
                quantized = int(round(detail / quant_step))
                if quantized:
                    support = 1 << (level + 1)
                    support_start = block_start + pair_index * support
                    support_end = min(len(raw), support_start + support)
                    if support_start < len(raw):
                        atoms.append(
                            {
                                "section_name": section_name,
                                "raw_offset": support_start,
                                "raw_end": support_end,
                                "level": level,
                                "coefficient_index": pair_index,
                                "coefficient_quantized": quantized,
                                "abs_coefficient_quantized": abs(quantized),
                                "estimated_wire_bytes": _estimated_atom_wire_bytes(
                                    support_start,
                                    level,
                                    quantized,
                                ),
                                "score_claim": False,
                            }
                        )
                next_level.append(avg)
            current = next_level
            width //= 2
            level += 1
    atoms.sort(
        key=lambda atom: (
            -int(atom["abs_coefficient_quantized"]),
            int(atom["raw_offset"]),
            int(atom["level"]),
            int(atom["coefficient_index"]),
        )
    )
    return atoms[:top_k]


# Adversarial review 2026-05-06 (BUG #1, 100% confidence): the planner previously
# advertised a varint+zigzag wire format here, but the actual encoder in
# `tac.hnerv_wavelet_sidechannel.encode_wavelet_atom_sidechannel` writes each
# atom via `struct.pack("<IIBIi", raw_offset, raw_end, level, coefficient_index,
# coefficient_quantized)` = uint32+uint32+uint8+uint32+int32 = 17 bytes fixed.
# The varint estimate was off by 2-6x and downstream `rate_score_delta`
# computations consumed the wrong number. Estimate now matches the wire encoder.
_WAVELET_ATOM_WIRE_BYTES = 17


def _estimated_atom_wire_bytes(raw_offset: int, level: int, coefficient: int) -> int:
    return _WAVELET_ATOM_WIRE_BYTES


def _manifest_for_label(scorecard: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    manifests = scorecard.get("payload_section_manifests")
    if not isinstance(manifests, list):
        raise HnervSectionPlanError("scorecard missing payload_section_manifests")
    for manifest in manifests:
        if isinstance(manifest, Mapping) and str(manifest.get("label") or "") == label:
            return manifest
    raise HnervSectionPlanError(f"missing payload section manifest label: {label}")


def _audit_source_manifest(archive: Any, packed: Any, manifest: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if manifest.get("score_claim") is not False:
        blockers.append("manifest_score_claim_not_false")
    if manifest.get("dispatch_attempted") is not False:
        blockers.append("manifest_dispatch_attempted_not_false")
    if manifest.get("archive_sha256") != archive.archive_sha256:
        blockers.append("source_archive_sha256_mismatch")
    if manifest.get("archive_bytes") != archive.archive_bytes:
        blockers.append("source_archive_bytes_mismatch")
    if manifest.get("zip_member") != archive.member_name:
        blockers.append("source_zip_member_mismatch")
    if manifest.get("payload_sha256") != sha256_bytes(archive.payload):
        blockers.append("source_payload_sha256_mismatch")
    if manifest.get("member_bytes") != archive.member_bytes:
        blockers.append("source_member_bytes_mismatch")
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        return blockers + ["manifest_sections_missing"]
    by_name = {
        str(section.get("name") or ""): section
        for section in sections
        if isinstance(section, Mapping)
    }
    for section_name in WAVELET_AUDIT_SECTIONS:
        expected = by_name.get(section_name)
        if expected is None:
            blockers.append(f"manifest_section_missing:{section_name}")
            continue
        # Adversarial review 2026-05-06 (BUG #6): guard packed.section_bytes —
        # a malformed archive can raise here and abort the whole plan build with
        # an unhandled exception instead of returning a structured blocker.
        try:
            section_bytes = packed.section_bytes(section_name)
        except Exception as exc:  # noqa: BLE001 — propagate as blocker
            blockers.append(f"manifest_section_bytes_unreadable:{section_name}:{exc}")
            continue
        if expected.get("bytes") != len(section_bytes):
            blockers.append(f"manifest_section_bytes_mismatch:{section_name}")
        if expected.get("sha256") != sha256_bytes(section_bytes):
            blockers.append(f"manifest_section_sha256_mismatch:{section_name}")
    return blockers


def plan_digest(plan: Mapping[str, Any]) -> str:
    """Return a stable digest over the selected atom coordinates."""

    h = hashlib.sha256()
    h.update(str(plan.get("source_archive_sha256")).encode("utf-8"))
    for section in plan.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        h.update(str(section.get("section_name")).encode("utf-8"))
        h.update(str(section.get("source_section_sha256")).encode("utf-8"))
        for atom in section.get("atoms") or []:
            if not isinstance(atom, Mapping):
                continue
            h.update(
                (
                    f"{atom.get('raw_offset')}:{atom.get('raw_end')}:"
                    f"{atom.get('level')}:{atom.get('coefficient_index')}:"
                    f"{atom.get('coefficient_quantized')};"
                ).encode("utf-8")
            )
    return h.hexdigest()


__all__ = [
    "DEFAULT_TARGET_SECTIONS",
    "SCHEMA_VERSION",
    "TOOL",
    "HnervWaveletResidualError",
    "build_wavelet_residual_plan",
    "plan_digest",
]
