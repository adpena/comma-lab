"""Offline WR01 wavelet transforms for PR106-style HNeRV sections.

This module consumes a byte-closed WR01 sidechannel candidate, applies the
decoded atoms to the referenced HNeRV brotli section, and emits a plain
PR106-format archive candidate. It does not load scorers or claim score
movement; exact CUDA auth eval remains the only score truth.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
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
RUNTIME_DECODE_VALIDATION_SCHEMA = "hnerv_wavelet_runtime_decode_validation.v1"
RUNTIME_DECODE_VALIDATION_FILENAME = "hnerv_wavelet_runtime_decode_validation.json"
# Adversarial review 2026-05-06 (BUG #5): re-export shared constant — this name
# stays stable for callers but its value comes from hnerv_lowlevel_packer.
DEFAULT_SECTION = DEFAULT_WAVELET_SECTION


class HnervWaveletApplyTransformError(ValueError):
    """Raised when a WR01 transform input is invalid."""


def _guard_no_tmp_path(path: str | Path, label: str) -> None:
    """Reject persisted WR01 inputs/outputs under common tmp roots."""

    raw = Path(path).expanduser()
    try:
        resolved = raw.resolve(strict=False)
    except OSError as exc:
        raise HnervWaveletApplyTransformError(f"{label}: failed to resolve path {path!s}") from exc
    text = resolved.as_posix()
    for forbidden in ("/tmp", "/private/tmp", "/var/tmp"):
        if text == forbidden or text.startswith(forbidden + "/"):
            raise HnervWaveletApplyTransformError(
                f"{label}: persisted WR01 artifact path must not be under {forbidden}: {text}"
            )


def build_wavelet_apply_transform_candidate(
    *,
    wavelet_archive: str | Path,
    output_dir: str | Path,
    source_label: str,
    section_name: str = DEFAULT_SECTION,
    strength_numerator: int = 1,
    strength_denominator: int = 2,
    source_archive: str | Path | None = None,
    source_archive_sha256: str | None = None,
    source_archive_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic plain-HNeRV archive after applying WR01 atoms.

    Round 6 R6-1 fix (2026-05-06): the WR01 wire format does NOT encode the
    source archive's SHA-256 (only per-section brotli-section SHAs). R5-5
    populated `source_archive_sha256` with `sha256_bytes(source_payload)` —
    the INNER payload hash, not the archive file hash, which silently
    corrupts the provenance chain. The field is now caller-supplied: pass
    the actual source archive SHA-256 if you have it; otherwise the manifest
    records None so downstream gates can detect the missing-provenance state
    instead of trusting a bogus value.
    """

    # Adversarial review 2026-05-06 (PARADIGM-alpha CRITICAL #5): validate > 0.
    # strength_numerator=0 silently produces a zero-strength transform (no byte
    # change) and only raises AFTER wasting a full brotli roundtrip on the
    # transformed_section comparison. Gate at the entry point so the caller
    # gets a fast deterministic error.
    if strength_numerator <= 0:
        raise HnervWaveletApplyTransformError(
            f"strength_numerator must be > 0 (got {strength_numerator}); "
            "strength_numerator=0 produces zero-strength transform with no byte change"
        )
    if strength_denominator <= 0:
        raise HnervWaveletApplyTransformError("strength_denominator must be positive")
    # Adversarial review 2026-05-06 (PARADIGM-alpha IMPORTANT, /tmp path guard):
    # /tmp paths are FORBIDDEN in persisted manifests per CLAUDE.md
    # ("Forbidden /tmp paths in any persisted artifact"). Guard at the gate.
    _guard_no_tmp_path(wavelet_archive, "wavelet_archive")
    _guard_no_tmp_path(output_dir, "output_dir")
    if source_archive is not None:
        _guard_no_tmp_path(source_archive, "source_archive")
    _validate_sha256_or_none(source_archive_sha256, "source_archive_sha256")
    if source_archive_bytes is not None and int(source_archive_bytes) <= 0:
        raise HnervWaveletApplyTransformError("source_archive_bytes must be positive or None")
    archive = read_strict_single_member_zip(wavelet_archive)
    parsed_wrapper = parse_wavelet_sidechannel_archive_bytes(archive.payload)
    source_payload = parsed_wrapper.source_payload
    source_custody = _source_archive_custody(
        source_archive=source_archive,
        source_archive_sha256=source_archive_sha256,
        source_archive_bytes=source_archive_bytes,
        source_payload=source_payload,
    )
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
    # Adversarial review 2026-05-06 (PARADIGM-alpha CRITICAL #3): replace fragile
    # `REPACKABLE_SECTIONS[1]` / `[0]` index access with explicit string
    # constants gated by an `assert` membership check against the shared
    # constant tuple. Any reorder of REPACKABLE_SECTIONS in
    # `hnerv_lowlevel_packer` previously silently produced wrong-section
    # repacks (the index access compiled fine but mapped to the wrong
    # section). With the assert, a future rename or reorder fires loudly at
    # function entry instead of silently shipping a corrupt archive.
    # The dataclass FIELD names below (latents_and_sidecar_brotli= and
    # decoder_packed_brotli=) are tied to the dataclass definition; Python
    # kwargs require literal identifiers so they cannot themselves be
    # constants — but the comparison strings now are.
    _LATENTS_SECTION = "latents_and_sidecar_brotli"
    _DECODER_SECTION = "decoder_packed_brotli"
    if _LATENTS_SECTION not in REPACKABLE_SECTIONS:
        raise HnervWaveletApplyTransformError(
            f"_LATENTS_SECTION constant {_LATENTS_SECTION!r} not in REPACKABLE_SECTIONS "
            f"({REPACKABLE_SECTIONS!r}); hnerv_lowlevel_packer may have been "
            f"refactored; update this file."
        )
    if _DECODER_SECTION not in REPACKABLE_SECTIONS:
        raise HnervWaveletApplyTransformError(
            f"_DECODER_SECTION constant {_DECODER_SECTION!r} not in REPACKABLE_SECTIONS "
            f"({REPACKABLE_SECTIONS!r}); hnerv_lowlevel_packer may have been "
            f"refactored; update this file."
        )
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
    source_archive_bytes_estimate = _source_archive_bytes_from_wrapper(
        archive.archive_bytes,
        archive.payload,
        source_payload,
    )
    rate_score_delta = 25.0 * (candidate_archive_bytes - source_archive_bytes_estimate) / 37_545_489.0
    source_section_sha256 = sha256_bytes(source_section)
    candidate_section_sha256 = sha256_bytes(transformed_section)
    runtime_atom_records = _runtime_atom_records(
        section=section,
        section_name=section_name,
        source_section_sha256=source_section_sha256,
    )
    runtime_apply = {
        "schema": "hnerv_wavelet_runtime_apply.v1",
        "status": "applied",
        "ready_for_runtime_apply_review": True,
        "runtime_mode": transform_stats["runtime_mode"],
        "section_name": section_name,
        "source_section_sha256": source_section_sha256,
        "candidate_section_sha256": candidate_section_sha256,
        "source_raw_sha256": sha256_bytes(raw),
        "candidate_raw_sha256": sha256_bytes(transformed_raw),
        "source_raw_bytes": len(raw),
        "candidate_raw_bytes": len(transformed_raw),
        "strength_numerator": int(strength_numerator),
        "strength_denominator": int(strength_denominator),
        "applied_atom_count": int(transform_stats["applied_atom_count"]),
        "skipped_atom_count": int(transform_stats["skipped_atom_count"]),
        "applied_atom_ids": [str(atom["atom_id"]) for atom in runtime_atom_records],
        "applied_atoms": runtime_atom_records,
        "score_claim": False,
        "dispatch_attempted": False,
    }
    dispatch_blockers = [
        "requires_archive_manifest_preflight",
        "requires_component_response_or_exact_cuda_eval",
        "requires_lane_dispatch_claim",
        "requires_exact_cuda_auth_eval",
    ]
    if source_custody.get("source_archive_sha256") is None:
        dispatch_blockers.insert(0, "requires_source_archive_sha256")
    if source_custody.get("source_archive_bytes") is None:
        dispatch_blockers.insert(0, "requires_source_archive_bytes")
    runtime_decode_validation_path = output_root / RUNTIME_DECODE_VALIDATION_FILENAME
    runtime_decode_validation = _build_runtime_decode_validation_manifest(
        candidate_archive=candidate_archive,
        candidate_archive_bytes=candidate_archive_bytes,
        candidate_payload=candidate_payload,
        candidate_packed=candidate_packed,
        source_archive_sha256=source_custody.get("source_archive_sha256"),
        source_archive_bytes=source_custody.get("source_archive_bytes"),
        source_packed=packed,
        source_payload=source_payload,
        section_name=section_name,
        source_raw=raw,
        transformed_raw=transformed_raw,
        runtime_apply=runtime_apply,
        manifest_path=runtime_decode_validation_path,
    )
    runtime_decode_validation_path.write_text(
        json_text(runtime_decode_validation),
        encoding="utf-8",
    )
    manifest = {
        "schema": "hnerv_wavelet_apply_transform_candidate.v1",
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "wavelet_archive_path": str(wavelet_archive),
        "wavelet_archive_sha256": archive.archive_sha256,
        "wavelet_archive_bytes": archive.archive_bytes,
        "source_archive_path": source_custody.get("source_archive_path"),
        "source_archive_sha256": source_custody.get("source_archive_sha256"),
        "source_archive_bytes": source_custody.get("source_archive_bytes"),
        "source_archive_member_name": source_custody.get("source_archive_member_name"),
        "source_archive_custody_mode": source_custody["source_archive_custody_mode"],
        "candidate_archive_path": str(candidate_archive),
        "candidate_archive_sha256": sha256_file(candidate_archive),
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_member_name": archive.member_name,
        "candidate_payload_sha256": sha256_bytes(candidate_payload),
        "candidate_payload_bytes": len(candidate_payload),
        "source_payload_sha256": sha256_bytes(source_payload),
        "source_payload_bytes": len(source_payload),
        "source_archive_bytes_estimated_from_wrapper": source_archive_bytes_estimate,
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
        "source_section_sha256": source_section_sha256,
        "candidate_section_sha256": candidate_section_sha256,
        "source_section_bytes": len(source_section),
        "candidate_section_bytes": len(transformed_section),
        "section_byte_delta": len(transformed_section) - len(source_section),
        "changed_section_name": section_name,
        "changed_section_source_sha256": source_section_sha256,
        "changed_section_sha256": candidate_section_sha256,
        "changed_section_bytes": len(transformed_section),
        "changed_section": {
            "name": section_name,
            "source_sha256": source_section_sha256,
            "candidate_sha256": candidate_section_sha256,
            "source_bytes": len(source_section),
            "candidate_bytes": len(transformed_section),
            "byte_delta": len(transformed_section) - len(source_section),
        },
        "source_raw_sha256": sha256_bytes(raw),
        "candidate_raw_sha256": sha256_bytes(transformed_raw),
        "source_raw_bytes": len(raw),
        "candidate_raw_bytes": len(transformed_raw),
        "strength_numerator": int(strength_numerator),
        "strength_denominator": int(strength_denominator),
        "transform_stats": transform_stats,
        "runtime_apply": runtime_apply,
        "runtime_apply_atom_count": len(runtime_atom_records),
        "runtime_apply_atom_ids": [str(atom["atom_id"]) for atom in runtime_atom_records],
        "runtime_decode_validation_schema": RUNTIME_DECODE_VALIDATION_SCHEMA,
        "runtime_decode_validation_manifest_path": str(runtime_decode_validation_path),
        "runtime_decode_validation_manifest_sha256": runtime_decode_validation[
            "manifest_sha256_excluding_self"
        ],
        "runtime_decode_validation": runtime_decode_validation,
        # Round 5 R5-3 fix (2026-05-06, 80%): split-brain — old code set
        # `ready_for_archive_preflight: True` while keeping
        # `requires_archive_manifest_preflight` in dispatch_blockers. A
        # consumer reading `ready_for_archive_preflight` saw "ready" but the
        # blocker list said it isn't. Aligned to the apply_gate convention:
        # both ready_for_* flags are False; the blockers list is the canonical
        # "next required evidence" record (per R4-B fail-closed-by-design).
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": dispatch_blockers,
    }
    manifest_path = output_root / "hnerv_wavelet_apply_transform_candidate.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json_text(manifest), encoding="utf-8")
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
    # Round 6 R6-4 + Round 7 R7-2 fix (2026-05-06): pre-filter BOTH non-Mapping
    # atoms AND malformed-Mapping atoms (missing or non-int required keys) so
    # the sort key function and the inner loop both see only valid atoms. The
    # sort key calls int() on raw_offset/level/coefficient_index — a malformed
    # value would crash the whole sort before the inner loop's try/except
    # could catch it. Pre-filter solves it at the gate.
    valid_atoms: list[Mapping[str, Any]] = []
    for a in atoms:
        if not isinstance(a, Mapping):
            skipped += 1
            continue
        try:
            int(a.get("raw_offset"))
            int(a.get("raw_end"))
            int(a.get("coefficient_quantized"))
            int(a.get("level", 0))
            int(a.get("coefficient_index", 0))
        except (TypeError, ValueError):
            skipped += 1
            continue
        valid_atoms.append(a)
    atoms = sorted(
        valid_atoms,
        key=lambda a: (
            int(a.get("raw_offset", 0)),
            int(a.get("level", 0)),
            int(a.get("coefficient_index", 0)),
        ),
    )
    for atom in atoms:
        # All atoms here are well-formed Mappings by construction (the
        # pre-filter above checked Mapping membership AND that all five
        # required keys coerce to int); both classes were already added to
        # `skipped`.
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


def _validate_sha256_or_none(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not all(c in "0123456789abcdef" for c in value)
    ):
        raise HnervWaveletApplyTransformError(
            f"{field_name} must be a 64-char lowercase hex digest or None"
        )


def _source_archive_custody(
    *,
    source_archive: str | Path | None,
    source_archive_sha256: str | None,
    source_archive_bytes: int | None,
    source_payload: bytes,
) -> dict[str, Any]:
    if source_archive is not None:
        archive = read_strict_single_member_zip(source_archive)
        if archive.payload != source_payload:
            raise HnervWaveletApplyTransformError(
                "source_archive payload does not match WR01 wrapper source payload"
            )
        if source_archive_sha256 is not None and source_archive_sha256 != archive.archive_sha256:
            raise HnervWaveletApplyTransformError(
                "source_archive_sha256 does not match measured source archive"
            )
        if source_archive_bytes is not None and int(source_archive_bytes) != archive.archive_bytes:
            raise HnervWaveletApplyTransformError(
                "source_archive_bytes does not match measured source archive"
            )
        return {
            "source_archive_path": str(source_archive),
            "source_archive_sha256": archive.archive_sha256,
            "source_archive_bytes": archive.archive_bytes,
            "source_archive_member_name": archive.member_name,
            "source_archive_custody_mode": "verified_source_archive_payload_match",
        }
    if (source_archive_sha256 is None) != (source_archive_bytes is None):
        raise HnervWaveletApplyTransformError(
            "source_archive_sha256 and source_archive_bytes must be provided together"
        )
    if source_archive_sha256 is None:
        return {
            "source_archive_path": None,
            "source_archive_sha256": None,
            "source_archive_bytes": None,
            "source_archive_member_name": None,
            "source_archive_custody_mode": "missing_source_archive_identity_fail_closed",
        }
    return {
        "source_archive_path": None,
        "source_archive_sha256": source_archive_sha256,
        "source_archive_bytes": int(source_archive_bytes),
        "source_archive_member_name": None,
        "source_archive_custody_mode": "operator_supplied_source_archive_identity",
    }


def _runtime_atom_records(
    *,
    section: Mapping[str, Any],
    section_name: str,
    source_section_sha256: str,
) -> list[dict[str, Any]]:
    atoms = section.get("atoms")
    if not isinstance(atoms, list):
        raise HnervWaveletApplyTransformError("WR01 section atoms must be a list")
    records: list[dict[str, Any]] = []
    for atom in atoms:
        if not isinstance(atom, Mapping):
            continue
        try:
            raw_offset = int(atom.get("raw_offset"))
            raw_end = int(atom.get("raw_end"))
            level = int(atom.get("level", 0))
            coefficient_index = int(atom.get("coefficient_index", 0))
            coefficient_quantized = int(atom.get("coefficient_quantized"))
        except (TypeError, ValueError):
            continue
        atom_id = _runtime_atom_id(
            section_name=section_name,
            source_section_sha256=source_section_sha256,
            raw_offset=raw_offset,
            raw_end=raw_end,
            level=level,
            coefficient_index=coefficient_index,
            coefficient_quantized=coefficient_quantized,
        )
        records.append(
            {
                "atom_id": atom_id,
                "section_name": section_name,
                "source_section_sha256": source_section_sha256,
                "raw_offset": raw_offset,
                "raw_end": raw_end,
                "level": level,
                "coefficient_index": coefficient_index,
                "coefficient_quantized": coefficient_quantized,
                "score_claim": False,
                "dispatch_attempted": False,
            }
        )
    records.sort(
        key=lambda atom: (
            int(atom["raw_offset"]),
            int(atom["level"]),
            int(atom["coefficient_index"]),
            int(atom["coefficient_quantized"]),
            str(atom["atom_id"]),
        )
    )
    for apply_order, atom in enumerate(records, start=1):
        atom["runtime_apply_order"] = apply_order
    return records


def _runtime_atom_id(
    *,
    section_name: str,
    source_section_sha256: str,
    raw_offset: int,
    raw_end: int,
    level: int,
    coefficient_index: int,
    coefficient_quantized: int,
) -> str:
    section_slug = section_name.replace("_", "-")
    payload = (
        f"{section_name}:{source_section_sha256}:"
        f"{raw_offset}:{raw_end}:{level}:{coefficient_index}:{coefficient_quantized}"
    )
    return f"wr01-{section_slug}-{sha256_bytes(payload.encode('utf-8'))[:16]}"


def _build_runtime_decode_validation_manifest(
    *,
    candidate_archive: Path,
    candidate_archive_bytes: int,
    candidate_payload: bytes,
    candidate_packed: Any,
    source_archive_sha256: str | None,
    source_archive_bytes: int | None,
    source_packed: Any,
    source_payload: bytes,
    section_name: str,
    source_raw: bytes,
    transformed_raw: bytes,
    runtime_apply: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        reread = read_strict_single_member_zip(candidate_archive)
    except Exception as exc:  # pragma: no cover - defensive custody surface
        reread = None
        blockers.append(f"candidate_archive_strict_read_failed:{type(exc).__name__}")
    if reread is not None:
        if reread.archive_sha256 != sha256_file(candidate_archive):
            blockers.append("candidate_archive_sha256_reread_mismatch")
        if reread.archive_bytes != int(candidate_archive_bytes):
            blockers.append("candidate_archive_bytes_reread_mismatch")
        if reread.payload != candidate_payload:
            blockers.append("candidate_payload_reread_mismatch")
        try:
            parsed_candidate = parse_ff_packed_brotli_hnerv(reread.payload)
        except Exception as exc:  # pragma: no cover - defensive custody surface
            parsed_candidate = None
            blockers.append(f"candidate_payload_hnerv_parse_failed:{type(exc).__name__}")
    else:
        parsed_candidate = None

    sections = _runtime_decode_section_records(
        source_packed=source_packed,
        candidate_packed=parsed_candidate or candidate_packed,
        expected_changed_section=section_name,
        blockers=blockers,
    )
    changed_sections = [
        str(section["section_name"])
        for section in sections
        if section.get("section_changed") is True
    ]
    if changed_sections != [section_name]:
        blockers.append("runtime_decode_changed_section_set_mismatch")
    changed_record = next(
        (section for section in sections if section.get("section_name") == section_name),
        None,
    )
    if changed_record is None:
        blockers.append(f"runtime_decode_missing_changed_section:{section_name}")
    else:
        if changed_record.get("source_raw_sha256") != sha256_bytes(source_raw):
            blockers.append(f"runtime_decode_source_raw_sha256_mismatch:{section_name}")
        if changed_record.get("candidate_raw_sha256") != sha256_bytes(transformed_raw):
            blockers.append(f"runtime_decode_candidate_raw_sha256_mismatch:{section_name}")
    if runtime_apply.get("score_claim") is not False:
        blockers.append("runtime_apply_score_claim_not_false")
    if runtime_apply.get("dispatch_attempted") is not False:
        blockers.append("runtime_apply_dispatch_attempted_not_false")
    if int(runtime_apply.get("applied_atom_count") or 0) <= 0:
        blockers.append("runtime_apply_no_applied_atoms")
    if not runtime_apply.get("applied_atom_ids"):
        blockers.append("runtime_apply_missing_applied_atom_ids")

    manifest = {
        "schema": RUNTIME_DECODE_VALIDATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": f"{TOOL}.runtime_decode_validation",
        "validation_mode": "local_wr01_runtime_decode_validation_not_score",
        "status": "ready" if not blockers else "blocked",
        "ready_for_runtime_decode_review": not blockers,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "fail_closed": bool(blockers),
        "score_claim": False,
        "dispatch_attempted": False,
        "source_archive_sha256": source_archive_sha256,
        "source_archive_bytes": source_archive_bytes,
        "source_payload_sha256": sha256_bytes(source_payload),
        "source_payload_bytes": len(source_payload),
        "candidate_archive_path": str(candidate_archive),
        "candidate_archive_sha256": sha256_file(candidate_archive),
        "candidate_archive_bytes": int(candidate_archive_bytes),
        "candidate_payload_sha256": sha256_bytes(candidate_payload),
        "candidate_payload_bytes": len(candidate_payload),
        "section_name": section_name,
        "changed_section_names": changed_sections,
        "changed_section_only": changed_sections == [section_name],
        "sections": sections,
        "runtime_apply": dict(runtime_apply),
        "exact_cuda_auth_eval": False,
        "component_response_validation": False,
        "blockers": blockers,
        "dispatch_blockers": [
            "runtime_decode_validation_is_not_score_evidence",
            "requires_archive_manifest_preflight",
            "requires_component_response_or_exact_cuda_eval",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
        "manifest_path": str(manifest_path),
    }
    manifest["manifest_sha256_excluding_self"] = _manifest_sha256_excluding_self(manifest)
    return manifest


def _runtime_decode_section_records(
    *,
    source_packed: Any,
    candidate_packed: Any,
    expected_changed_section: str,
    blockers: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for section_name, source_section, candidate_section in _hnerv_section_bytes(
        source_packed,
        candidate_packed,
    ):
        section_changed = source_section != candidate_section
        if section_changed != (section_name == expected_changed_section):
            blockers.append(f"runtime_decode_unexpected_section_change:{section_name}")
        record: dict[str, Any] = {
            "section_name": section_name,
            "source_section_sha256": sha256_bytes(source_section),
            "candidate_section_sha256": sha256_bytes(candidate_section),
            "source_section_bytes": len(source_section),
            "candidate_section_bytes": len(candidate_section),
            "section_changed": section_changed,
            "expected_changed": section_name == expected_changed_section,
            "score_claim": False,
        }
        if section_name == "packed_header_ff_len24":
            record.update(
                {
                    "decode_probe_status": "not_required_for_header_section",
                    "source_raw_sha256": sha256_bytes(source_section),
                    "candidate_raw_sha256": sha256_bytes(candidate_section),
                    "source_raw_bytes": len(source_section),
                    "candidate_raw_bytes": len(candidate_section),
                }
            )
        else:
            try:
                source_raw = brotli.decompress(source_section)
                candidate_raw = brotli.decompress(candidate_section)
            except brotli.error:
                blockers.append(f"runtime_decode_brotli_decode_failed:{section_name}")
                record.update(
                    {
                        "decode_probe_status": "brotli_decode_failed",
                        "source_raw_sha256": None,
                        "candidate_raw_sha256": None,
                        "source_raw_bytes": None,
                        "candidate_raw_bytes": None,
                    }
                )
            else:
                record.update(
                    {
                        "decode_probe_status": "local_brotli_decode_ok_not_score",
                        "source_raw_sha256": sha256_bytes(source_raw),
                        "candidate_raw_sha256": sha256_bytes(candidate_raw),
                        "source_raw_bytes": len(source_raw),
                        "candidate_raw_bytes": len(candidate_raw),
                    }
                )
        records.append(record)
    return records


def _hnerv_section_bytes(source_packed: Any, candidate_packed: Any) -> Sequence[tuple[str, bytes, bytes]]:
    return (
        ("packed_header_ff_len24", source_packed.header, candidate_packed.header),
        (
            "decoder_packed_brotli",
            source_packed.decoder_packed_brotli,
            candidate_packed.decoder_packed_brotli,
        ),
        (
            "latents_and_sidecar_brotli",
            source_packed.latents_and_sidecar_brotli,
            candidate_packed.latents_and_sidecar_brotli,
        ),
    )


def _manifest_sha256_excluding_self(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256_excluding_self"}
    return sha256_bytes(json_text(payload).encode("utf-8"))


__all__ = [
    "DEFAULT_SECTION",
    "RUNTIME_DECODE_VALIDATION_FILENAME",
    "RUNTIME_DECODE_VALIDATION_SCHEMA",
    "SCHEMA_VERSION",
    "TOOL",
    "HnervWaveletApplyTransformError",
    "apply_wr01_atoms_to_raw",
    "build_wavelet_apply_transform_candidate",
]
