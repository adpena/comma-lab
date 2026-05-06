"""Offline WR01 wavelet transforms for PR106-style HNeRV sections.

This module consumes a byte-closed WR01 sidechannel candidate, applies the
decoded atoms to the referenced HNeRV brotli section, and emits a plain
PR106-format archive candidate. It does not load scorers or claim score
movement; exact CUDA auth eval remains the only score truth.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_lowlevel_packer import (
    DEFAULT_WAVELET_SECTION,
    REPACKABLE_SECTIONS,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_wavelet_sidechannel import parse_wavelet_sidechannel_archive_bytes
from tac.repo_io import json_text, sha256_file

SCHEMA_VERSION = 1
TOOL = "tac.hnerv_wavelet_apply_transform.build_wavelet_apply_transform_candidate"
# Adversarial review 2026-05-06 (BUG #5): re-export shared constant — this name
# stays stable for callers but its value comes from hnerv_lowlevel_packer.
DEFAULT_SECTION = DEFAULT_WAVELET_SECTION


class HnervWaveletApplyTransformError(ValueError):
    """Raised when a WR01 transform input is invalid."""


def build_wavelet_apply_transform_candidate(
    *,
    wavelet_archive: str | Path,
    output_dir: str | Path,
    source_label: str,
    section_name: str = DEFAULT_SECTION,
    strength_numerator: int = 1,
    strength_denominator: int = 2,
) -> dict[str, Any]:
    """Build a deterministic plain-HNeRV archive after applying WR01 atoms."""

    if strength_numerator < 0:
        raise HnervWaveletApplyTransformError("strength_numerator must be non-negative")
    if strength_denominator <= 0:
        raise HnervWaveletApplyTransformError("strength_denominator must be positive")
    archive = read_strict_single_member_zip(wavelet_archive)
    parsed_wrapper = parse_wavelet_sidechannel_archive_bytes(archive.payload)
    source_payload = parsed_wrapper.source_payload
    packed = parse_ff_packed_brotli_hnerv(source_payload)
    section = _single_section(parsed_wrapper.decoded_sidechannel, section_name)
    source_section = packed.section_bytes(section_name)
    if section.get("source_section_sha256") != sha256_bytes(source_section):
        raise HnervWaveletApplyTransformError(
            f"{section_name}: source section sha256 mismatch"
        )
    try:
        raw = brotli.decompress(source_section)
    except brotli.error as exc:
        raise HnervWaveletApplyTransformError(f"{section_name}: brotli decompress failed") from exc
    if int(section.get("raw_bytes") or -1) != len(raw):
        raise HnervWaveletApplyTransformError(
            f"{section_name}: raw byte count mismatch WR01={section.get('raw_bytes')} actual={len(raw)}"
        )

    transformed_raw, transform_stats = apply_wr01_atoms_to_raw(
        raw,
        section,
        strength_numerator=strength_numerator,
        strength_denominator=strength_denominator,
    )
    if transformed_raw == raw:
        raise HnervWaveletApplyTransformError(f"{section_name}: WR01 transform produced no byte change")
    transformed_section = brotli.compress(transformed_raw, quality=11)
    # Round 2 R2-5 fix (2026-05-06, 83% confidence): replace hardcoded section
    # name string literals in the comparison with the shared REPACKABLE_SECTIONS
    # constants. The dataclass FIELD names below (latents_and_sidecar_brotli=
    # and decoder_packed_brotli=) are tied to the dataclass definition, not
    # drift sources — they cannot use constants because Python kwargs require
    # literal identifiers. The comparison string literals previously WERE drift
    # sources; now they read from REPACKABLE_SECTIONS so any rename in the
    # packer module surfaces as a NameError, not silent drift.
    _LATENTS_SECTION = REPACKABLE_SECTIONS[1]  # "latents_and_sidecar_brotli"
    _DECODER_SECTION = REPACKABLE_SECTIONS[0]  # "decoder_packed_brotli"
    candidate_packed = dataclasses.replace(
        packed,
        latents_and_sidecar_brotli=(
            transformed_section if section_name == _LATENTS_SECTION else packed.latents_and_sidecar_brotli
        ),
        decoder_packed_brotli=(
            transformed_section if section_name == _DECODER_SECTION else packed.decoder_packed_brotli
        ),
    )
    candidate_payload = candidate_packed.to_bytes()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_archive = output_root / "hnerv_wavelet_apply_transform_candidate.zip"
    write_stored_single_member_zip(candidate_archive, member_name=archive.member_name, payload=candidate_payload)
    candidate_archive_bytes = candidate_archive.stat().st_size
    rate_score_delta = 25.0 * (
        candidate_archive_bytes - _source_archive_bytes_from_wrapper(archive.archive_bytes, archive.payload, source_payload)
    ) / 37_545_489.0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "wavelet_archive_path": str(wavelet_archive),
        "wavelet_archive_sha256": archive.archive_sha256,
        "wavelet_archive_bytes": archive.archive_bytes,
        "candidate_archive_path": str(candidate_archive),
        "candidate_archive_sha256": sha256_file(candidate_archive),
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_member_name": archive.member_name,
        "candidate_payload_sha256": sha256_bytes(candidate_payload),
        "candidate_payload_bytes": len(candidate_payload),
        "source_payload_sha256": sha256_bytes(source_payload),
        "source_payload_bytes": len(source_payload),
        "source_archive_bytes_estimated_from_wrapper": _source_archive_bytes_from_wrapper(
            archive.archive_bytes,
            archive.payload,
            source_payload,
        ),
        # Adversarial review 2026-05-06 (BUG #2, 95% confidence): these two fields
        # are advisory-only — the byte delta is computed from a wrapper-arithmetic
        # estimate of the source archive bytes, NOT from a re-measured contest
        # archive. The rate-score delta is therefore a [prediction], not a
        # [contest-CUDA] measurement. Per CLAUDE.md "Forbidden empirical-claim-
        # without-evidence-tag" we tag both via the field-name suffix and an
        # explicit `_advisory_note` adjacent so downstream gate code (e.g.
        # `tac.hnerv_wavelet_apply_gate`) cannot mistake them for empirical bytes.
        "candidate_archive_byte_delta_vs_source_estimate": candidate_archive_bytes
        - _source_archive_bytes_from_wrapper(archive.archive_bytes, archive.payload, source_payload),
        "rate_score_delta_vs_source_estimate": rate_score_delta,
        "rate_score_delta_advisory_note": (
            "byte delta and rate-score delta are estimated from "
            "_source_archive_bytes_from_wrapper, not from a re-measured contest "
            "archive — [prediction], not [contest-CUDA]. Treat as advisory only "
            "for break-even gating; do not promote to a score claim."
        ),
        "section_name": section_name,
        "source_section_sha256": sha256_bytes(source_section),
        "candidate_section_sha256": sha256_bytes(transformed_section),
        "source_section_bytes": len(source_section),
        "candidate_section_bytes": len(transformed_section),
        "section_byte_delta": len(transformed_section) - len(source_section),
        "source_raw_sha256": sha256_bytes(raw),
        "candidate_raw_sha256": sha256_bytes(transformed_raw),
        "source_raw_bytes": len(raw),
        "candidate_raw_bytes": len(transformed_raw),
        "strength_numerator": int(strength_numerator),
        "strength_denominator": int(strength_denominator),
        "transform_stats": transform_stats,
        # Round 5 R5-3 fix (2026-05-06, 80%): split-brain — old code set
        # `ready_for_archive_preflight: True` while keeping
        # `requires_archive_manifest_preflight` in dispatch_blockers. A
        # consumer reading `ready_for_archive_preflight` saw "ready" but the
        # blocker list said it isn't. Aligned to the apply_gate convention:
        # both ready_for_* flags are False; the blockers list is the canonical
        # "next required evidence" record (per R4-B fail-closed-by-design).
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        # Round 5 R5-5 fix (2026-05-06, 81%): expose source_archive_sha256 at
        # the top-level manifest key so the apply_gate's provenance chain
        # (which reads sidechannel_manifest.get("source_archive_sha256"))
        # works when an apply_transform manifest is the input.
        "source_archive_sha256": sha256_bytes(source_payload),
        "dispatch_blockers": [
            "requires_archive_manifest_preflight",
            "requires_component_response_or_exact_cuda_eval",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }
    manifest_path = output_root / "hnerv_wavelet_apply_transform_candidate.json"
    manifest_path.write_text(json_text(manifest), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def apply_wr01_atoms_to_raw(
    raw: bytes,
    section: Mapping[str, Any],
    *,
    strength_numerator: int,
    strength_denominator: int,
) -> tuple[bytes, dict[str, Any]]:
    """Return raw bytes with WR01 detail coefficients attenuated."""

    data = bytearray(raw)
    changed_positions: set[int] = set()
    applied = 0
    skipped = 0
    total_abs_delta = 0
    atoms = section.get("atoms")
    if not isinstance(atoms, list):
        raise HnervWaveletApplyTransformError("WR01 section atoms must be a list")
    # Adversarial review 2026-05-06 (BUG #4): atoms can have overlapping
    # [raw_offset, raw_end) ranges (Haar levels nest), and `_clamp_u8` is
    # non-linear at 0/255. Apply order changes output bytes when ranges overlap
    # AND a delta clamps. Sort by (raw_offset, level, coefficient_index) to make
    # apply deterministic regardless of caller's atom-list ordering.
    atoms = sorted(
        (a for a in atoms if isinstance(a, Mapping)),
        key=lambda a: (
            int(a.get("raw_offset", 0)),
            int(a.get("level", 0)),
            int(a.get("coefficient_index", 0)),
        ),
    )
    for atom in atoms:
        if not isinstance(atom, Mapping):
            skipped += 1
            continue
        start = int(atom.get("raw_offset"))
        end = int(atom.get("raw_end"))
        coefficient = int(atom.get("coefficient_quantized"))
        width = end - start
        if start < 0 or end > len(data) or width < 2:
            skipped += 1
            continue
        half = width // 2
        if half <= 0:
            skipped += 1
            continue
        delta = _scaled_int(coefficient, strength_numerator, strength_denominator)
        if delta == 0:
            skipped += 1
            continue
        left_delta = -delta
        right_delta = delta
        for idx in range(start, start + half):
            old = data[idx]
            new = _clamp_u8(old + left_delta)
            if new != old:
                data[idx] = new
                changed_positions.add(idx)
                total_abs_delta += abs(new - old)
        for idx in range(start + half, end):
            old = data[idx]
            new = _clamp_u8(old + right_delta)
            if new != old:
                data[idx] = new
                changed_positions.add(idx)
                total_abs_delta += abs(new - old)
        applied += 1
    return bytes(data), {
        "runtime_mode": "offline_wr01_detail_attenuation",
        "applied_atom_count": applied,
        "skipped_atom_count": skipped,
        "changed_raw_positions": len(changed_positions),
        "total_abs_byte_delta": total_abs_delta,
        "score_claim": False,
    }


def _single_section(decoded: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    sections = decoded.get("sections")
    if not isinstance(sections, list):
        raise HnervWaveletApplyTransformError("decoded WR01 sidechannel missing sections")
    matches = [
        section for section in sections
        if isinstance(section, Mapping) and section.get("section_name") == section_name
    ]
    if len(matches) != 1:
        raise HnervWaveletApplyTransformError(
            f"expected exactly one WR01 section {section_name!r}, got {len(matches)}"
        )
    return matches[0]


def _scaled_int(value: int, numerator: int, denominator: int) -> int:
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    return sign * ((magnitude * numerator + denominator // 2) // denominator)


def _clamp_u8(value: int) -> int:
    return max(0, min(255, int(value)))


def _source_archive_bytes_from_wrapper(wrapper_archive_bytes: int, wrapper_payload: bytes, source_payload: bytes) -> int:
    # Single stored ZIP framing is constant for a fixed member name, so removing
    # the 0xFA wrapper bytes estimates the original archive byte count exactly
    # for the deterministic writer used by this lane.
    return int(wrapper_archive_bytes) - (len(wrapper_payload) - len(source_payload))


__all__ = [
    "DEFAULT_SECTION",
    "SCHEMA_VERSION",
    "TOOL",
    "HnervWaveletApplyTransformError",
    "apply_wr01_atoms_to_raw",
    "build_wavelet_apply_transform_candidate",
]
