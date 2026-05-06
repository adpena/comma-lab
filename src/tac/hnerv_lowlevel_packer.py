"""Low-level HNeRV payload repacking helpers.

The public HNeRV frontier includes a compact single-member ZIP grammar whose
payload starts with ``0xff`` plus a little-endian 24-bit decoder section length.
The two charged brotli sections after that are natural byte-level optimization
targets. This module performs deterministic, no-op-resistant repack planning
and candidate emission only; it never claims score movement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_section_repack import (
    HnervSectionPlanError,
    audit_candidate_section_diff,
    build_section_repack_plan,
)
from tac.repo_io import sha256_file

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
SCHEMA_VERSION = 1

REPACKABLE_SECTIONS = ("decoder_packed_brotli", "latents_and_sidecar_brotli")

# Adversarial review 2026-05-06 (BUG #5): single source of truth for section
# names previously hardcoded across hnerv_wavelet_residual / sidechannel /
# apply_transform / apply_gate. Drift between these files would silently produce
# plans with no matching sections (a blocker, not a crash) and is hard to
# diagnose. Importers should reference these constants rather than the literal
# string.
DEFAULT_WAVELET_SECTION = "latents_and_sidecar_brotli"
WAVELET_AUDIT_SECTIONS = (
    "packed_header_ff_len24",
    "decoder_packed_brotli",
    "latents_and_sidecar_brotli",
)


class HnervLowlevelPackError(ValueError):
    """Raised when an HNeRV low-level pack/repack input is invalid."""


@dataclasses.dataclass(frozen=True)
class SingleMemberArchive:
    """A strict single-member ZIP payload."""

    member_name: str
    payload: bytes
    archive_sha256: str
    archive_bytes: int
    member_bytes: int


@dataclasses.dataclass(frozen=True)
class PackedHnervPayload:
    """Parsed ``0xff + len24 + brotli decoder + brotli latents`` payload."""

    header: bytes
    decoder_packed_brotli: bytes
    latents_and_sidecar_brotli: bytes

    def section_bytes(self, name: str) -> bytes:
        if name == "packed_header_ff_len24":
            return self.header
        if name == "decoder_packed_brotli":
            return self.decoder_packed_brotli
        if name == "latents_and_sidecar_brotli":
            return self.latents_and_sidecar_brotli
        raise HnervLowlevelPackError(f"unknown HNeRV packed section: {name}")

    def to_bytes(self) -> bytes:
        decoder_len = len(self.decoder_packed_brotli)
        if decoder_len > 0xFFFFFF:
            raise HnervLowlevelPackError(f"decoder section too large for len24: {decoder_len}")
        header = bytes([0xFF]) + decoder_len.to_bytes(3, "little")
        return header + self.decoder_packed_brotli + self.latents_and_sidecar_brotli


@dataclasses.dataclass(frozen=True)
class BrotliRecodeChoice:
    """One brotli recode attempt for a packed HNeRV section."""

    section_name: str
    raw_bytes: int
    source_bytes: int
    candidate_bytes: int
    quality: int
    lgwin: int | None
    candidate_sha256: str
    changed: bool

    @property
    def byte_delta(self) -> int:
        return self.candidate_bytes - self.source_bytes


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for ``data``."""

    return hashlib.sha256(data).hexdigest()


def read_strict_single_member_zip(path: str | Path) -> SingleMemberArchive:
    """Read a one-member ZIP archive without assuming the member name."""

    archive = Path(path)
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise HnervLowlevelPackError(f"expected exactly one ZIP entry, got {len(infos)}")
        info = infos[0]
        if info.is_dir():
            raise HnervLowlevelPackError(f"single ZIP entry must be a file, got directory {info.filename!r}")
        _validate_member_name(info.filename)
        bad = zf.testzip()
        if bad is not None:
            raise HnervLowlevelPackError(f"ZIP CRC validation failed for member {bad!r}")
        payload = zf.read(info.filename)
    return SingleMemberArchive(
        member_name=info.filename,
        payload=payload,
        archive_sha256=sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        member_bytes=len(payload),
    )


def write_stored_single_member_zip(path: str | Path, *, member_name: str, payload: bytes) -> None:
    """Write a deterministic stored ZIP with one validated member."""

    _validate_member_name(member_name)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name, date_time=FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def parse_ff_packed_brotli_hnerv(payload: bytes) -> PackedHnervPayload:
    """Parse the PR106-style ``0xff`` packed HNeRV payload."""

    if len(payload) < 4:
        raise HnervLowlevelPackError("packed HNeRV payload is shorter than 4-byte header")
    if payload[0] != 0xFF:
        raise HnervLowlevelPackError(f"expected packed HNeRV 0xff header, got 0x{payload[0]:02x}")
    decoder_len = int.from_bytes(payload[1:4], "little")
    decoder_start = 4
    decoder_end = decoder_start + decoder_len
    if decoder_end > len(payload):
        raise HnervLowlevelPackError(
            f"decoder section length {decoder_len} exceeds payload bytes {len(payload)}"
        )
    return PackedHnervPayload(
        header=payload[:4],
        decoder_packed_brotli=payload[decoder_start:decoder_end],
        latents_and_sidecar_brotli=payload[decoder_end:],
    )


def brotli_recode_search(
    section_name: str,
    compressed: bytes,
    *,
    qualities: Iterable[int] = (9, 10, 11),
    lgwins: Iterable[int | None] = (None, 18, 20, 22, 24),
) -> tuple[BrotliRecodeChoice, bytes]:
    """Return the smallest deterministic brotli recode for one section."""

    if section_name not in REPACKABLE_SECTIONS:
        raise HnervLowlevelPackError(f"section is not brotli-repackable: {section_name}")
    try:
        raw = brotli.decompress(compressed)
    except brotli.error as exc:
        raise HnervLowlevelPackError(f"{section_name} is not brotli-decompressible") from exc

    best_choice: BrotliRecodeChoice | None = None
    best_payload: bytes | None = None
    for quality in qualities:
        q = int(quality)
        if not 0 <= q <= 11:
            raise HnervLowlevelPackError(f"brotli quality out of range: {q}")
        for lgwin in lgwins:
            kwargs: dict[str, int] = {"quality": q}
            normalized_lgwin = None if lgwin is None else int(lgwin)
            if normalized_lgwin is not None:
                if not 10 <= normalized_lgwin <= 24:
                    raise HnervLowlevelPackError(f"brotli lgwin out of range: {normalized_lgwin}")
                kwargs["lgwin"] = normalized_lgwin
            candidate = brotli.compress(raw, **kwargs)
            choice = BrotliRecodeChoice(
                section_name=section_name,
                raw_bytes=len(raw),
                source_bytes=len(compressed),
                candidate_bytes=len(candidate),
                quality=q,
                lgwin=normalized_lgwin,
                candidate_sha256=sha256_bytes(candidate),
                changed=candidate != compressed,
            )
            if best_choice is None or _choice_sort_key(choice) < _choice_sort_key(best_choice):
                best_choice = choice
                best_payload = candidate
    if best_choice is None or best_payload is None:
        raise HnervLowlevelPackError("brotli search did not evaluate any variants")
    return best_choice, best_payload


def build_lowlevel_brotli_repack_candidate(
    *,
    source_archive: str | Path,
    scorecard: Mapping[str, Any],
    source_label: str,
    output_dir: str | Path,
    target_sections: Sequence[str] = REPACKABLE_SECTIONS,
    qualities: Iterable[int] = (9, 10, 11),
    lgwins: Iterable[int | None] = (None, 18, 20, 22, 24),
    allow_rate_regression: bool = False,
) -> dict[str, Any]:
    """Build a deterministic HNeRV brotli-repack candidate and proof manifest.

    The manifest may include a candidate archive when a targeted section changed.
    It remains non-promotable until normal archive preflight and exact CUDA auth
    eval are run on those bytes.
    """

    archive = read_strict_single_member_zip(source_archive)
    packed = parse_ff_packed_brotli_hnerv(archive.payload)
    manifest = _manifest_for_label(scorecard, source_label)
    plan = build_section_repack_plan(scorecard, labels=[source_label])
    blockers = _audit_source_against_manifest(archive, packed, manifest)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    replacements: dict[str, bytes] = {}
    attempts: list[dict[str, Any]] = []
    target_set = tuple(dict.fromkeys(str(section) for section in target_sections))
    for section_name in target_set:
        if section_name not in REPACKABLE_SECTIONS:
            blockers.append(f"target_section_not_repackable:{section_name}")
            continue
        source_bytes = packed.section_bytes(section_name)
        try:
            choice, candidate = brotli_recode_search(
                section_name,
                source_bytes,
                qualities=qualities,
                lgwins=lgwins,
            )
        except HnervLowlevelPackError as exc:
            blockers.append(f"brotli_recode_failed:{section_name}:{exc}")
            continue
        rate_positive = choice.byte_delta < 0
        dispatchable_change = choice.changed and (rate_positive or allow_rate_regression)
        attempts.append(
            {
                "section_name": section_name,
                "raw_bytes": choice.raw_bytes,
                "source_bytes": choice.source_bytes,
                "candidate_bytes": choice.candidate_bytes,
                "byte_delta": choice.byte_delta,
                "rate_positive": rate_positive,
                "quality": choice.quality,
                "lgwin": choice.lgwin,
                "source_section_sha256": sha256_bytes(source_bytes),
                "candidate_section_sha256": choice.candidate_sha256,
                "changed": choice.changed,
                "accepted_for_candidate": dispatchable_change,
            }
        )
        if dispatchable_change:
            replacements[section_name] = candidate

    if not replacements:
        blockers.append("no_rate_positive_section_recode")
        return _blocked_result(
            source_label=source_label,
            source_archive=archive,
            packed=packed,
            blockers=blockers,
            attempts=attempts,
        )

    candidate_packed = dataclasses.replace(
        packed,
        decoder_packed_brotli=replacements.get(
            "decoder_packed_brotli", packed.decoder_packed_brotli
        ),
        latents_and_sidecar_brotli=replacements.get(
            "latents_and_sidecar_brotli", packed.latents_and_sidecar_brotli
        ),
    )
    candidate_payload = candidate_packed.to_bytes()
    candidate_archive_path = output_root / f"{_slug(source_label)}_hnerv_brotli_repack_candidate.zip"
    write_stored_single_member_zip(
        candidate_archive_path,
        member_name=archive.member_name,
        payload=candidate_payload,
    )
    candidate_archive_sha = sha256_file(candidate_archive_path)
    candidate_archive_bytes = candidate_archive_path.stat().st_size
    candidate_packed_checked = parse_ff_packed_brotli_hnerv(candidate_payload)
    raw_equivalence = _brotli_raw_equivalence(packed, candidate_packed_checked)
    candidate_diff = _candidate_diff(
        source_label=source_label,
        source_archive=archive,
        candidate_archive_sha256=candidate_archive_sha,
        candidate_archive_bytes=candidate_archive_bytes,
        packed=packed,
        candidate_packed=candidate_packed_checked,
        brotli_raw_equivalence=raw_equivalence,
    )
    raw_equivalence_blockers = [
        f"brotli_raw_mismatch:{row['section_name']}"
        for row in raw_equivalence
        if row["raw_equal"] is not True
    ]
    blockers.extend(raw_equivalence_blockers)
    audit = audit_candidate_section_diff(plan, candidate_diff, require_raw_equivalence=True)
    if blockers:
        audit["blockers"] = list(audit.get("blockers") or []) + blockers
        audit["ready_for_archive_preflight"] = False
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_lowlevel_packer.build_lowlevel_brotli_repack_candidate",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": source_label,
        "source_archive_path": str(Path(source_archive)),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "candidate_archive_path": str(candidate_archive_path),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_member_name": archive.member_name,
        "candidate_payload_sha256": sha256_bytes(candidate_payload),
        "candidate_payload_bytes": len(candidate_payload),
        "brotli_raw_equivalence": raw_equivalence,
        "attempts": attempts,
        "candidate_diff": candidate_diff,
        "candidate_diff_audit": audit,
        "ready_for_archive_preflight": bool(audit["ready_for_archive_preflight"]),
        "dispatch_blockers": [
            "requires_archive_manifest_preflight",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _choice_sort_key(choice: BrotliRecodeChoice) -> tuple[int, int, int, int, str]:
    lgwin_sort = -1 if choice.lgwin is None else choice.lgwin
    return (
        choice.candidate_bytes,
        0 if choice.changed else 1,
        choice.quality,
        lgwin_sort,
        choice.candidate_sha256,
    )


def _manifest_for_label(scorecard: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    manifests = scorecard.get("payload_section_manifests")
    if not isinstance(manifests, list):
        raise HnervSectionPlanError("scorecard missing payload_section_manifests")
    for manifest in manifests:
        if isinstance(manifest, Mapping) and str(manifest.get("label") or "") == label:
            return manifest
    raise HnervSectionPlanError(f"missing payload section manifest label: {label}")


def _audit_source_against_manifest(
    archive: SingleMemberArchive,
    packed: PackedHnervPayload,
    manifest: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    expected_archive_sha = manifest.get("archive_sha256")
    if not _is_sha256(expected_archive_sha):
        blockers.append("source_manifest_archive_sha256_missing_or_invalid")
    elif expected_archive_sha != archive.archive_sha256:
        blockers.append("source_archive_sha256_mismatch")
    expected_archive_bytes = manifest.get("archive_bytes")
    if not isinstance(expected_archive_bytes, int) or expected_archive_bytes != archive.archive_bytes:
        blockers.append("source_archive_bytes_mismatch_or_missing")
    expected_member = manifest.get("zip_member")
    if not isinstance(expected_member, str) or not expected_member:
        blockers.append("source_zip_member_missing")
    elif expected_member != archive.member_name:
        blockers.append("source_zip_member_mismatch")
    expected_payload_sha = manifest.get("payload_sha256")
    if not _is_sha256(expected_payload_sha):
        blockers.append("source_payload_sha256_missing_or_invalid")
    elif expected_payload_sha != sha256_bytes(archive.payload):
        blockers.append("source_payload_sha256_mismatch")
    expected_member_bytes = manifest.get("member_bytes")
    if not isinstance(expected_member_bytes, int) or expected_member_bytes != archive.member_bytes:
        blockers.append("source_member_bytes_mismatch_or_missing")
    sections = manifest.get("sections")
    if not isinstance(sections, list):
        blockers.append("source_manifest_missing_sections")
        return blockers
    expected_sections = [
        ("packed_header_ff_len24", 0, 0, 4),
        ("decoder_packed_brotli", 1, 4, 4 + len(packed.decoder_packed_brotli)),
        (
            "latents_and_sidecar_brotli",
            2,
            4 + len(packed.decoder_packed_brotli),
            len(archive.payload),
        ),
    ]
    if len(sections) != len(expected_sections):
        blockers.append("source_manifest_section_count_mismatch")
    for expected, section in zip(expected_sections, sections, strict=False):
        if not isinstance(section, Mapping):
            blockers.append("source_manifest_section_not_object")
            continue
        expected_name, expected_index, expected_start, expected_end = expected
        section_name = str(section.get("name") or "")
        if section_name != expected_name:
            blockers.append(f"source_section_name_mismatch:{expected_name}")
        if section.get("index") != expected_index:
            blockers.append(f"source_section_index_mismatch:{expected_name}")
        if section.get("start") != expected_start:
            blockers.append(f"source_section_start_mismatch:{expected_name}")
        if section.get("end") != expected_end:
            blockers.append(f"source_section_end_mismatch:{expected_name}")
        try:
            actual = packed.section_bytes(expected_name)
        except HnervLowlevelPackError:
            blockers.append(f"source_section_unknown:{section_name}")
            continue
        if int(section.get("bytes") or -1) != len(actual):
            blockers.append(f"source_section_bytes_mismatch:{expected_name}")
        if section.get("sha256") != sha256_bytes(actual):
            blockers.append(f"source_section_sha256_mismatch:{expected_name}")
    return blockers


def _candidate_diff(
    *,
    source_label: str,
    source_archive: SingleMemberArchive,
    candidate_archive_sha256: str,
    candidate_archive_bytes: int,
    packed: PackedHnervPayload,
    candidate_packed: PackedHnervPayload,
    brotli_raw_equivalence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for section_name in (
        "packed_header_ff_len24",
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    ):
        source_section = packed.section_bytes(section_name)
        candidate_section = candidate_packed.section_bytes(section_name)
        if source_section == candidate_section:
            continue
        rows.append(
            {
                "label": source_label,
                "section_name": section_name,
                "source_section_sha256": sha256_bytes(source_section),
                "candidate_section_sha256": sha256_bytes(candidate_section),
                "source_bytes": len(source_section),
                "candidate_bytes": len(candidate_section),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_lowlevel_packer.candidate_diff",
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": source_label,
        "source_archive_sha256": source_archive.archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "source_archive_bytes": source_archive.archive_bytes,
        "candidate_archive_bytes": candidate_archive_bytes,
        "source_payload_sha256": sha256_bytes(packed.to_bytes()),
        "candidate_payload_sha256": sha256_bytes(candidate_packed.to_bytes()),
        "brotli_raw_equivalence": list(brotli_raw_equivalence),
        "sections": rows,
    }


def _brotli_raw_equivalence(
    source: PackedHnervPayload,
    candidate: PackedHnervPayload,
) -> list[dict[str, Any]]:
    rows = []
    for section_name in REPACKABLE_SECTIONS:
        source_section = source.section_bytes(section_name)
        candidate_section = candidate.section_bytes(section_name)
        try:
            source_raw = brotli.decompress(source_section)
            candidate_raw = brotli.decompress(candidate_section)
            raw_equal = source_raw == candidate_raw
            source_raw_sha = sha256_bytes(source_raw)
            candidate_raw_sha = sha256_bytes(candidate_raw)
            raw_bytes = len(source_raw)
        except brotli.error:
            raw_equal = False
            source_raw_sha = None
            candidate_raw_sha = None
            raw_bytes = None
        rows.append(
            {
                "section_name": section_name,
                "raw_equal": raw_equal,
                "raw_bytes": raw_bytes,
                "source_raw_sha256": source_raw_sha,
                "candidate_raw_sha256": candidate_raw_sha,
            }
        )
    return rows


def _blocked_result(
    *,
    source_label: str,
    source_archive: SingleMemberArchive,
    packed: PackedHnervPayload,
    blockers: Sequence[str],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_lowlevel_packer.build_lowlevel_brotli_repack_candidate",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_label": source_label,
        "source_archive_sha256": source_archive.archive_sha256,
        "source_archive_bytes": source_archive.archive_bytes,
        "source_member_name": source_archive.member_name,
        "source_payload_sha256": sha256_bytes(packed.to_bytes()),
        "attempts": list(attempts),
        "blockers": list(blockers),
        "dispatch_blockers": [
            "requires_byte_different_archive",
            "requires_archive_manifest_preflight",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _validate_member_name(name: str) -> None:
    if not name:
        raise HnervLowlevelPackError("ZIP member name must be nonempty")
    if "\\" in name:
        raise HnervLowlevelPackError(f"backslash ZIP member path is forbidden: {name!r}")
    path = Path(name)
    if path.is_absolute():
        raise HnervLowlevelPackError(f"absolute ZIP member path is forbidden: {name!r}")
    parts = path.parts
    if ".." in parts:
        raise HnervLowlevelPackError(f"parent traversal ZIP member path is forbidden: {name!r}")
    if any(part in ("", ".") for part in parts):
        raise HnervLowlevelPackError(f"unsafe ZIP member path is forbidden: {name!r}")
    if any(part.startswith(".") for part in parts):
        raise HnervLowlevelPackError(f"hidden ZIP member path is forbidden: {name!r}")
    if name.startswith("__MACOSX/") or "/__MACOSX/" in name:
        raise HnervLowlevelPackError(f"resource-fork ZIP member path is forbidden: {name!r}")


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in out.split("_") if part) or "hnerv"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


__all__ = [
    "FIXED_DATE_TIME",
    "REPACKABLE_SECTIONS",
    "SCHEMA_VERSION",
    "BrotliRecodeChoice",
    "HnervLowlevelPackError",
    "PackedHnervPayload",
    "SingleMemberArchive",
    "brotli_recode_search",
    "build_lowlevel_brotli_repack_candidate",
    "parse_ff_packed_brotli_hnerv",
    "read_strict_single_member_zip",
    "sha256_bytes",
    "write_stored_single_member_zip",
]
