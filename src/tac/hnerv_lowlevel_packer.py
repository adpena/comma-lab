# SPDX-License-Identifier: MIT
"""Low-level HNeRV payload repacking helpers.

The public HNeRV frontier includes a compact single-member ZIP grammar whose
payload starts with ``0xff`` plus a little-endian 24-bit decoder section length.
The two charged brotli sections after that are natural byte-level optimization
targets. This module performs deterministic, no-op-resistant repack planning
and candidate emission only; it never claims score movement.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import os
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
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_MAGIC,
    PR106SidecarPacket,
    StoredZipMember,
    emit_pr106_sidecar_packet,
    emit_single_stored_member_archive,
    parse_pr106_sidecar_packet,
    pr106_sidecar_consumed_byte_proof,
)
from tac.packet_compiler.pr106_sidecar_packet import (
    read_single_stored_member_archive as read_packet_single_stored_member_archive,
)
from tac.repo_io import sha256_file

FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
SCHEMA_VERSION = 1

REPACKABLE_SECTIONS = ("decoder_packed_brotli", "latents_and_sidecar_brotli")
A1_SPLIT_BROTLI_STREAM_COUNT = 7
HEADER_FORMAT_FF_LEN24 = "ff_len24"
HEADER_FORMAT_A1_U32_SECTION_TOTAL = "a1_u32_section_total"
HEADER_FORMAT_PR101_FIXED_OFFSETS = "pr101_fixed_offsets"
PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387

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
    header_format: str = HEADER_FORMAT_FF_LEN24

    @property
    def header_section_name(self) -> str:
        if self.header_format == HEADER_FORMAT_FF_LEN24:
            return "packed_header_ff_len24"
        if self.header_format == HEADER_FORMAT_A1_U32_SECTION_TOTAL:
            return "packed_header_a1_u32_section_total"
        if self.header_format == HEADER_FORMAT_PR101_FIXED_OFFSETS:
            return "packed_header_pr101_fixed_offsets"
        raise HnervLowlevelPackError(f"unknown packed HNeRV header format: {self.header_format}")

    def section_bytes(self, name: str) -> bytes:
        if name == self.header_section_name:
            return self.header
        if name == "decoder_packed_brotli":
            return self.decoder_packed_brotli
        if name == "latents_and_sidecar_brotli":
            return self.latents_and_sidecar_brotli
        raise HnervLowlevelPackError(f"unknown HNeRV packed section: {name}")

    def to_bytes(self) -> bytes:
        decoder_len = len(self.decoder_packed_brotli)
        if self.header_format == HEADER_FORMAT_FF_LEN24:
            if decoder_len > 0xFFFFFF:
                raise HnervLowlevelPackError(f"decoder section too large for len24: {decoder_len}")
            header = bytes([0xFF]) + decoder_len.to_bytes(3, "little")
        elif self.header_format == HEADER_FORMAT_A1_U32_SECTION_TOTAL:
            section_total = 4 + decoder_len
            header = section_total.to_bytes(4, "little")
        elif self.header_format == HEADER_FORMAT_PR101_FIXED_OFFSETS:
            return self.decoder_packed_brotli + self.latents_and_sidecar_brotli
        else:
            raise HnervLowlevelPackError(f"unknown packed HNeRV header format: {self.header_format}")
        return header + self.decoder_packed_brotli + self.latents_and_sidecar_brotli


@dataclasses.dataclass(frozen=True)
class PackedArchiveView:
    """Archive view whose repack target is an inner ``0xff`` HNeRV payload."""

    archive: SingleMemberArchive
    packed: PackedHnervPayload
    payload_kind: str
    hnerv_payload: bytes
    sidecar_packet: PR106SidecarPacket | None = None
    stored_member: StoredZipMember | None = None
    repackable_sections: tuple[str, ...] = REPACKABLE_SECTIONS
    decoder_brotli_stream_count: int | None = None
    hnerv_payload_start: int = 0

    def emit_payload(self, packed: PackedHnervPayload) -> bytes:
        inner = packed.to_bytes()
        if self.sidecar_packet is None:
            return inner
        return emit_pr106_sidecar_packet(
            dataclasses.replace(self.sidecar_packet, pr106_bytes=inner)
        )

    def write_archive(
        self,
        path: str | Path,
        payload: bytes,
        *,
        member_name: str | None = None,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        emitted_member_name = member_name or self.archive.member_name
        _validate_member_name(emitted_member_name)
        if self.stored_member is None:
            write_stored_single_member_zip(
                target,
                member_name=emitted_member_name,
                payload=payload,
            )
            return
        candidate_member = dataclasses.replace(
            self.stored_member,
            name=emitted_member_name,
            payload=payload,
        )
        target.write_bytes(emit_single_stored_member_archive(candidate_member))


@dataclasses.dataclass(frozen=True)
class BrotliRecodeChoice:
    """One brotli recode attempt for a packed HNeRV section."""

    section_name: str
    raw_bytes: int
    source_bytes: int
    candidate_bytes: int
    quality: int
    lgwin: int | None
    lgblock: int | None
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
    try:
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
    except zipfile.BadZipFile as exc:
        raise HnervLowlevelPackError(f"invalid ZIP archive: {archive}") from exc
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
        header_format=HEADER_FORMAT_FF_LEN24,
    )


def parse_a1_headered_split_brotli_hnerv(payload: bytes) -> PackedHnervPayload:
    """Parse the A1/PR101-style ``uint32 section_total + split-brotli`` payload."""

    if len(payload) < 4:
        raise HnervLowlevelPackError("A1 packed HNeRV payload is shorter than 4-byte header")
    section_total = int.from_bytes(payload[:4], "little")
    if section_total < 4 or section_total > len(payload):
        raise HnervLowlevelPackError(
            f"A1 decoder section total {section_total} exceeds payload bytes {len(payload)}"
        )
    decoder_blob = payload[4:section_total]
    if not decoder_blob:
        raise HnervLowlevelPackError("A1 decoder split-brotli section is empty")
    _split_brotli_streams(decoder_blob, A1_SPLIT_BROTLI_STREAM_COUNT)
    return PackedHnervPayload(
        header=payload[:4],
        decoder_packed_brotli=decoder_blob,
        latents_and_sidecar_brotli=payload[section_total:],
        header_format=HEADER_FORMAT_A1_U32_SECTION_TOTAL,
    )


def parse_pr101_fixed_offset_microcodec_hnerv(payload: bytes) -> PackedHnervPayload:
    """Parse PR101/PR106-R2 fixed-offset HNeRV microcodec payloads."""

    minimum = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    if len(payload) < minimum:
        raise HnervLowlevelPackError(
            f"PR101 microcodec payload {len(payload)} bytes is smaller than fixed minimum {minimum}"
        )
    decoder_blob = payload[:PR101_DECODER_BLOB_LEN]
    _split_brotli_streams(decoder_blob, A1_SPLIT_BROTLI_STREAM_COUNT)
    return PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=decoder_blob,
        latents_and_sidecar_brotli=payload[PR101_DECODER_BLOB_LEN:],
        header_format=HEADER_FORMAT_PR101_FIXED_OFFSETS,
    )


def read_packed_archive_view(path: str | Path) -> PackedArchiveView:
    """Read a raw HNeRV archive or PR106 sidecar wrapper as a repackable view."""

    archive = read_strict_single_member_zip(path)
    if archive.payload and archive.payload[0] == PR106_SIDECAR_MAGIC:
        try:
            packet = parse_pr106_sidecar_packet(archive.payload)
            packed = parse_ff_packed_brotli_hnerv(packet.pr106_bytes)
            stored_member = read_packet_single_stored_member_archive(
                Path(path).read_bytes(),
                expected_member_name=archive.member_name,
            )
        except ValueError as exc:
            raise HnervLowlevelPackError(f"invalid PR106 sidecar packet: {exc}") from exc
        return PackedArchiveView(
            archive=archive,
            packed=packed,
            payload_kind="pr106_sidecar_wrapper",
            hnerv_payload=packet.pr106_bytes,
            sidecar_packet=packet,
            stored_member=stored_member,
            decoder_brotli_stream_count=_detect_decoder_brotli_stream_count(
                packed.decoder_packed_brotli
            ),
            hnerv_payload_start=6,
        )
    if archive.payload and archive.payload[0] == 0xFF:
        packed = parse_ff_packed_brotli_hnerv(archive.payload)
        return PackedArchiveView(
            archive=archive,
            packed=packed,
            payload_kind="raw_ff_hnerv",
            hnerv_payload=archive.payload,
        )
    try:
        packed = parse_pr101_fixed_offset_microcodec_hnerv(archive.payload)
    except HnervLowlevelPackError:
        packed = None
    if packed is not None:
        return PackedArchiveView(
            archive=archive,
            packed=packed,
            payload_kind="pr101_fixed_offset_hnerv_microcodec",
            hnerv_payload=archive.payload,
            repackable_sections=("decoder_packed_brotli",),
            decoder_brotli_stream_count=A1_SPLIT_BROTLI_STREAM_COUNT,
        )
    packed = parse_a1_headered_split_brotli_hnerv(archive.payload)
    return PackedArchiveView(
        archive=archive,
        packed=packed,
        payload_kind="a1_headered_split_brotli_hnerv",
        hnerv_payload=archive.payload,
        repackable_sections=("decoder_packed_brotli",),
        decoder_brotli_stream_count=A1_SPLIT_BROTLI_STREAM_COUNT,
    )


def brotli_recode_search(
    section_name: str,
    compressed: bytes,
    *,
    qualities: Iterable[int] = (9, 10, 11),
    lgwins: Iterable[int | None] = (None, 18, 20, 22, 24),
    lgblocks: Iterable[int | None] = (None,),
    jobs: int = 1,
    stream_count: int | None = None,
) -> tuple[BrotliRecodeChoice, bytes]:
    """Return the smallest deterministic brotli recode for one section."""

    if section_name not in REPACKABLE_SECTIONS:
        raise HnervLowlevelPackError(f"section is not brotli-repackable: {section_name}")
    try:
        raw: bytes | tuple[bytes, ...]
        if stream_count is None:
            raw = brotli.decompress(compressed)
        else:
            raw = tuple(row[1] for row in _split_brotli_streams(compressed, stream_count))
    except brotli.error as exc:
        codec_hint = _codec_magic_hint(compressed)
        if codec_hint is not None:
            raise HnervLowlevelPackError(
                f"{section_name} is {codec_hint}, not a brotli stream; "
                "route through the codec-specific recoder"
            ) from exc
        raise HnervLowlevelPackError(f"{section_name} is not brotli-decompressible") from exc

    attempts: list[tuple[int, int | None, int | None]] = []
    for quality in qualities:
        q = int(quality)
        if not 0 <= q <= 11:
            raise HnervLowlevelPackError(f"brotli quality out of range: {q}")
        for lgwin in lgwins:
            normalized_lgwin = None if lgwin is None else int(lgwin)
            if normalized_lgwin is not None and not 10 <= normalized_lgwin <= 24:
                raise HnervLowlevelPackError(f"brotli lgwin out of range: {normalized_lgwin}")
            for lgblock in lgblocks:
                normalized_lgblock = None if lgblock is None else int(lgblock)
                if normalized_lgblock == 0:
                    normalized_lgblock = None
                if normalized_lgblock is not None and not 16 <= normalized_lgblock <= 24:
                    raise HnervLowlevelPackError(
                        f"brotli lgblock out of range: {normalized_lgblock}"
                    )
                attempts.append((q, normalized_lgwin, normalized_lgblock))
    attempts = sorted(set(attempts), key=lambda item: (item[0], _optional_sort(item[1]), _optional_sort(item[2])))
    if not attempts:
        raise HnervLowlevelPackError("brotli search did not evaluate any variants")

    best_choice: BrotliRecodeChoice | None = None
    best_payload: bytes | None = None
    max_workers = _bounded_jobs(jobs, len(attempts))
    if max_workers == 1:
        results = [
            _brotli_recode_attempt(section_name, compressed, raw, q, lgwin, lgblock)
            for q, lgwin, lgblock in attempts
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _brotli_recode_attempt,
                    section_name,
                    compressed,
                    raw,
                    q,
                    lgwin,
                    lgblock,
                )
                for q, lgwin, lgblock in attempts
            ]
            results = [future.result() for future in futures]
    for choice, candidate in results:
        if best_choice is None or _choice_sort_key(choice) < _choice_sort_key(best_choice):
            best_choice = choice
            best_payload = candidate
    assert best_choice is not None and best_payload is not None
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
    lgblocks: Iterable[int | None] = (None,),
    allow_rate_regression: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    """Build a deterministic HNeRV brotli-repack candidate and proof manifest.

    The manifest may include a candidate archive when a targeted section changed.
    It remains non-promotable until normal archive preflight and exact CUDA auth
    eval are run on those bytes.
    """

    view = read_packed_archive_view(source_archive)
    archive = view.archive
    packed = view.packed
    effective_scorecard, manifest, scorecard_anchor = _scorecard_for_source(
        scorecard,
        source_label=source_label,
        view=view,
    )
    plan = build_section_repack_plan(effective_scorecard, labels=[source_label])
    blockers = _audit_source_against_manifest(
        view,
        manifest,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    replacements: dict[str, bytes] = {}
    attempts: list[dict[str, Any]] = []
    target_set = tuple(dict.fromkeys(str(section) for section in target_sections))
    if target_set == REPACKABLE_SECTIONS and view.repackable_sections != REPACKABLE_SECTIONS:
        target_set = view.repackable_sections
    for section_name in target_set:
        if section_name not in view.repackable_sections:
            blockers.append(f"target_section_not_repackable:{section_name}")
            continue
        source_bytes = packed.section_bytes(section_name)
        try:
            choice, candidate = brotli_recode_search(
                section_name,
                source_bytes,
                qualities=qualities,
                lgwins=lgwins,
                lgblocks=lgblocks,
                jobs=jobs,
                stream_count=_section_brotli_stream_count(view, section_name),
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
                "lgblock": choice.lgblock,
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
            source_payload_kind=view.payload_kind,
            source_payload_sha256=sha256_bytes(archive.payload),
            source_inner_hnerv_payload_sha256=sha256_bytes(view.hnerv_payload),
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
    candidate_inner_payload = candidate_packed.to_bytes()
    candidate_payload = view.emit_payload(candidate_packed)
    candidate_archive_path = output_root / f"{_slug(source_label)}_hnerv_brotli_repack_candidate.zip"
    view.write_archive(candidate_archive_path, candidate_payload)
    candidate_archive_sha = sha256_file(candidate_archive_path)
    candidate_archive_bytes = candidate_archive_path.stat().st_size
    candidate_packed_checked = _parse_candidate_inner_payload(view, candidate_payload)
    section_name_aliases = _packed_section_name_aliases(view, manifest)
    raw_equivalence = _brotli_raw_equivalence(
        packed,
        candidate_packed_checked,
        repackable_sections=view.repackable_sections,
        decoder_brotli_stream_count=view.decoder_brotli_stream_count,
        section_name_aliases=section_name_aliases,
    )
    candidate_diff = _candidate_diff(
        source_label=source_label,
        source_archive=archive,
        candidate_archive_sha256=candidate_archive_sha,
        candidate_archive_bytes=candidate_archive_bytes,
        source_payload_kind=view.payload_kind,
        source_payload_sha256=sha256_bytes(archive.payload),
        candidate_payload_sha256=sha256_bytes(candidate_payload),
        packed=packed,
        candidate_packed=candidate_packed_checked,
        brotli_raw_equivalence=raw_equivalence,
        section_name_aliases=section_name_aliases,
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
        "source_payload_kind": view.payload_kind,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "source_payload_bytes": len(archive.payload),
        "source_inner_hnerv_payload_sha256": sha256_bytes(view.hnerv_payload),
        "source_inner_hnerv_payload_bytes": len(view.hnerv_payload),
        "candidate_archive_path": str(candidate_archive_path),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": candidate_archive_bytes,
        "candidate_member_name": archive.member_name,
        "candidate_payload_sha256": sha256_bytes(candidate_payload),
        "candidate_payload_bytes": len(candidate_payload),
        "candidate_inner_hnerv_payload_sha256": sha256_bytes(candidate_inner_payload),
        "candidate_inner_hnerv_payload_bytes": len(candidate_inner_payload),
        "scorecard_anchor": scorecard_anchor,
        "packet_ir_consumed_byte_proof": _packet_ir_consumed_byte_proof(view, candidate_payload),
        "sidecar_packet": _sidecar_packet_summary(view, candidate_payload),
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


def _choice_sort_key(choice: BrotliRecodeChoice) -> tuple[int, int, int, int, int, str]:
    return (
        choice.candidate_bytes,
        0 if choice.changed else 1,
        choice.quality,
        _optional_sort(choice.lgwin),
        _optional_sort(choice.lgblock),
        choice.candidate_sha256,
    )


def _optional_sort(value: int | None) -> int:
    return -1 if value is None else value


def _bounded_jobs(jobs: int, attempt_count: int) -> int:
    if jobs < 1:
        raise HnervLowlevelPackError(f"jobs must be >= 1, got {jobs}")
    return max(1, min(jobs, attempt_count, os.cpu_count() or 1))


def _brotli_recode_attempt(
    section_name: str,
    source: bytes,
    raw: bytes | Sequence[bytes],
    quality: int,
    lgwin: int | None,
    lgblock: int | None,
) -> tuple[BrotliRecodeChoice, bytes]:
    kwargs: dict[str, int] = {"quality": quality}
    if lgwin is not None:
        kwargs["lgwin"] = lgwin
    if lgblock is not None:
        kwargs["lgblock"] = lgblock
    if isinstance(raw, bytes):
        raw_bytes = len(raw)
        candidate = brotli.compress(raw, **kwargs)
    else:
        raw_bytes = sum(len(chunk) for chunk in raw)
        candidate = b"".join(brotli.compress(chunk, **kwargs) for chunk in raw)
    return (
        BrotliRecodeChoice(
            section_name=section_name,
            raw_bytes=raw_bytes,
            source_bytes=len(source),
            candidate_bytes=len(candidate),
            quality=quality,
            lgwin=lgwin,
            lgblock=lgblock,
            candidate_sha256=sha256_bytes(candidate),
            changed=candidate != source,
        ),
        candidate,
    )


def _scorecard_for_source(
    scorecard: Mapping[str, Any],
    *,
    source_label: str,
    view: PackedArchiveView,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    try:
        manifest = _manifest_for_label(scorecard, source_label)
        return scorecard, manifest, {
            "matched_scorecard_label": True,
            "derived_from_source_archive": False,
            "source_label": source_label,
        }
    except HnervSectionPlanError as exc:
        if "missing payload section manifest label" not in str(exc):
            raise
        manifest = _derive_payload_section_manifest(view, source_label)
        base = dict(scorecard)
        manifests = list(scorecard.get("payload_section_manifests") or [])
        manifests.append(manifest)
        base["payload_section_manifests"] = manifests
        return base, manifest, {
            "matched_scorecard_label": False,
            "derived_from_source_archive": True,
            "source_label": source_label,
            "source_scorecard_error": str(exc),
        }


def _manifest_for_label(scorecard: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    manifests = scorecard.get("payload_section_manifests")
    if not isinstance(manifests, list):
        raise HnervSectionPlanError("scorecard missing payload_section_manifests")
    for manifest in manifests:
        if isinstance(manifest, Mapping) and str(manifest.get("label") or "") == label:
            return manifest
    raise HnervSectionPlanError(f"missing payload section manifest label: {label}")


def _audit_source_against_manifest(
    view: PackedArchiveView,
    manifest: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    archive = view.archive
    packed = view.packed
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
    expected_sections = _expected_manifest_sections(view, manifest)
    section_by_name = {
        str(section.get("name") or ""): section
        for section in sections
        if isinstance(section, Mapping)
    }
    if all(expected[0] in section_by_name for expected in expected_sections):
        manifest_sections = [section_by_name[expected[0]] for expected in expected_sections]
    else:
        manifest_sections = list(sections)
        if len(manifest_sections) != len(expected_sections):
            blockers.append("source_manifest_section_count_mismatch")
    for expected, section in zip(expected_sections, manifest_sections, strict=False):
        if not isinstance(section, Mapping):
            blockers.append("source_manifest_section_not_object")
            continue
        expected_name, expected_packed_name, expected_index, expected_start, expected_end = expected
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
            actual = packed.section_bytes(expected_packed_name)
        except HnervLowlevelPackError:
            blockers.append(f"source_section_unknown:{section_name}")
            continue
        if int(section.get("bytes") or -1) != len(actual):
            blockers.append(f"source_section_bytes_mismatch:{expected_name}")
        if section.get("sha256") != sha256_bytes(actual):
            blockers.append(f"source_section_sha256_mismatch:{expected_name}")
    return blockers


def _expected_manifest_sections(
    view: PackedArchiveView,
    manifest: Mapping[str, Any],
) -> list[tuple[str, str, int, int, int]]:
    """Return manifest-name/packed-name/start/end tuples for the inner payload."""

    packed = view.packed
    aliases = _packed_section_name_aliases(view, manifest)
    wrapper_named = set(aliases.values()) != {
        packed.header_section_name,
        "decoder_packed_brotli",
        "latents_and_sidecar_brotli",
    }
    base = view.hnerv_payload_start if wrapper_named else 0
    index_base = 1 if wrapper_named else 0
    return [
        (
            aliases[packed.header_section_name],
            packed.header_section_name,
            index_base,
            base,
            base + 4,
        ),
        (
            aliases["decoder_packed_brotli"],
            "decoder_packed_brotli",
            index_base + 1,
            base + 4,
            base + 4 + len(packed.decoder_packed_brotli),
        ),
        (
            aliases["latents_and_sidecar_brotli"],
            "latents_and_sidecar_brotli",
            index_base + 2,
            base + 4 + len(packed.decoder_packed_brotli),
            base + len(view.hnerv_payload),
        ),
    ]


def _packed_section_name_aliases(
    view: PackedArchiveView,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Map internal packed-section names to scorecard manifest names."""

    packed = view.packed
    sections = manifest.get("sections")
    manifest_names = {
        str(section.get("name") or "")
        for section in sections
        if isinstance(section, Mapping)
    }
    aliases = {
        packed.header_section_name: packed.header_section_name,
        "decoder_packed_brotli": "decoder_packed_brotli",
        "latents_and_sidecar_brotli": "latents_and_sidecar_brotli",
    }
    if view.sidecar_packet is None:
        return aliases
    wrapper_aliases = {
        packed.header_section_name: f"inner_{packed.header_section_name}",
        "decoder_packed_brotli": "inner_decoder_packed_brotli",
        "latents_and_sidecar_brotli": "inner_latents_and_sidecar_brotli",
    }
    for packed_name, wrapper_name in wrapper_aliases.items():
        if wrapper_name in manifest_names:
            aliases[packed_name] = wrapper_name
    return aliases


def _candidate_diff(
    *,
    source_label: str,
    source_archive: SingleMemberArchive,
    candidate_archive_sha256: str,
    candidate_archive_bytes: int,
    source_payload_kind: str,
    source_payload_sha256: str,
    candidate_payload_sha256: str,
    packed: PackedHnervPayload,
    candidate_packed: PackedHnervPayload,
    brotli_raw_equivalence: Sequence[Mapping[str, Any]],
    section_name_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    aliases = dict(section_name_aliases or {})
    rows = []
    for section_name in (
        packed.header_section_name,
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
                "section_name": aliases.get(section_name, section_name),
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
        "source_payload_kind": source_payload_kind,
        "source_payload_sha256": source_payload_sha256,
        "candidate_payload_sha256": candidate_payload_sha256,
        "source_inner_hnerv_payload_sha256": sha256_bytes(packed.to_bytes()),
        "candidate_inner_hnerv_payload_sha256": sha256_bytes(candidate_packed.to_bytes()),
        "brotli_raw_equivalence": list(brotli_raw_equivalence),
        "sections": rows,
    }


def _brotli_raw_equivalence(
    source: PackedHnervPayload,
    candidate: PackedHnervPayload,
    *,
    repackable_sections: Sequence[str] = REPACKABLE_SECTIONS,
    decoder_brotli_stream_count: int | None = None,
    section_name_aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    aliases = dict(section_name_aliases or {})
    rows = []
    for section_name in repackable_sections:
        source_section = source.section_bytes(section_name)
        candidate_section = candidate.section_bytes(section_name)
        try:
            stream_count = decoder_brotli_stream_count if section_name == "decoder_packed_brotli" else None
            source_raw = _brotli_section_raw(source_section, stream_count=stream_count)
            candidate_raw = _brotli_section_raw(candidate_section, stream_count=stream_count)
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
                "section_name": aliases.get(section_name, section_name),
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
    source_payload_kind: str,
    source_payload_sha256: str,
    source_inner_hnerv_payload_sha256: str,
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
        "source_payload_kind": source_payload_kind,
        "source_payload_sha256": source_payload_sha256,
        "source_inner_hnerv_payload_sha256": source_inner_hnerv_payload_sha256,
        "attempts": list(attempts),
        "blockers": list(blockers),
        "dispatch_blockers": [
            "requires_byte_different_archive",
            "requires_archive_manifest_preflight",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _derive_payload_section_manifest(
    view: PackedArchiveView,
    label: str,
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    start = 0
    for index, (name, data, role) in enumerate(
        [
            (view.packed.header_section_name, view.packed.header, "control_or_metadata"),
            (
                "decoder_packed_brotli",
                view.packed.decoder_packed_brotli,
                "decoder_weight_stream",
            ),
            (
                "latents_and_sidecar_brotli",
                view.packed.latents_and_sidecar_brotli,
                "latent_stream",
            ),
        ]
    ):
        end = start + len(data)
        sections.append(
            {
                "index": index,
                "name": name,
                "start": start,
                "end": end,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "optimization_role": role,
                "entropy_bits_per_byte": None,
            }
        )
        start = end
    return {
        "label": label,
        "archive_sha256": view.archive.archive_sha256,
        "archive_bytes": view.archive.archive_bytes,
        "zip_member": view.archive.member_name,
        "payload_sha256": sha256_bytes(view.archive.payload),
        "member_bytes": view.archive.member_bytes,
        "payload_kind": view.payload_kind,
        "section_payload_kind": "inner_hnerv_ff_payload",
        "section_payload_sha256": sha256_bytes(view.hnerv_payload),
        "section_payload_bytes": len(view.hnerv_payload),
        "profile_match_key": "member_sha256",
        "score_claim": False,
        "dispatch_attempted": False,
        "sections": sections,
    }


def _parse_candidate_inner_payload(
    view: PackedArchiveView,
    candidate_payload: bytes,
) -> PackedHnervPayload:
    if view.sidecar_packet is None:
        if view.payload_kind == "a1_headered_split_brotli_hnerv":
            return parse_a1_headered_split_brotli_hnerv(candidate_payload)
        return parse_ff_packed_brotli_hnerv(candidate_payload)
    try:
        packet = parse_pr106_sidecar_packet(candidate_payload)
    except ValueError as exc:
        raise HnervLowlevelPackError(f"candidate PR106 sidecar packet invalid: {exc}") from exc
    return parse_ff_packed_brotli_hnerv(packet.pr106_bytes)


def _section_brotli_stream_count(view: PackedArchiveView, section_name: str) -> int | None:
    if section_name == "decoder_packed_brotli":
        return view.decoder_brotli_stream_count
    return None


def _detect_decoder_brotli_stream_count(data: bytes) -> int | None:
    try:
        brotli.decompress(data)
        return None
    except brotli.error:
        pass
    try:
        _split_brotli_streams(data, A1_SPLIT_BROTLI_STREAM_COUNT)
    except (brotli.error, HnervLowlevelPackError):
        return None
    return A1_SPLIT_BROTLI_STREAM_COUNT


def _codec_magic_hint(data: bytes) -> str | None:
    for magic, name in (
        (b"HDM6", "HDM6 decoder codec"),
        (b"HDM4", "HDM4 decoder codec"),
        (b"HDM3", "HDM3 decoder codec"),
        (b"HLM3", "HLM3 fixed-latent range codec"),
        (b"HLM2", "HLM2 fixed-latent codec"),
        (b"HLM1", "HLM1 fixed-latent codec"),
    ):
        if data.startswith(magic):
            return name
    return None


def _brotli_section_raw(data: bytes, *, stream_count: int | None) -> bytes:
    if stream_count is None:
        return brotli.decompress(data)
    return b"".join(raw for _compressed, raw in _split_brotli_streams(data, stream_count))


def _split_brotli_streams(data: bytes, stream_count: int) -> list[tuple[bytes, bytes]]:
    if stream_count < 1:
        raise HnervLowlevelPackError(f"stream_count must be >= 1, got {stream_count}")
    rows: list[tuple[bytes, bytes]] = []
    pos = 0
    for _ in range(stream_count):
        start = pos
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos : pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise HnervLowlevelPackError("truncated split-brotli HNeRV decoder stream")
        rows.append((data[start:pos], b"".join(chunks)))
    if pos != len(data):
        raise HnervLowlevelPackError("trailing split-brotli HNeRV decoder payload")
    return rows


def _packet_ir_consumed_byte_proof(
    view: PackedArchiveView,
    candidate_payload: bytes,
) -> dict[str, Any] | None:
    if view.sidecar_packet is None:
        return None
    packet = parse_pr106_sidecar_packet(candidate_payload)
    return pr106_sidecar_consumed_byte_proof(packet)


def _sidecar_packet_summary(
    view: PackedArchiveView,
    candidate_payload: bytes,
) -> dict[str, Any] | None:
    if view.sidecar_packet is None:
        return None
    candidate_packet = parse_pr106_sidecar_packet(candidate_payload)
    source_packet = view.sidecar_packet
    return {
        "format_id": f"0x{source_packet.format_id:02X}",
        "sidecar_kind": source_packet.sidecar_kind,
        "source_pr106_payload_sha256": sha256_bytes(source_packet.pr106_bytes),
        "candidate_pr106_payload_sha256": sha256_bytes(candidate_packet.pr106_bytes),
        "source_sidecar_payload_sha256": sha256_bytes(source_packet.sidecar_payload),
        "candidate_sidecar_payload_sha256": sha256_bytes(candidate_packet.sidecar_payload),
        "sidecar_payload_preserved": candidate_packet.sidecar_payload == source_packet.sidecar_payload,
        "source_framing_meta_sha256": None
        if source_packet.framing_meta is None
        else sha256_bytes(source_packet.framing_meta),
        "candidate_framing_meta_sha256": None
        if candidate_packet.framing_meta is None
        else sha256_bytes(candidate_packet.framing_meta),
        "framing_meta_preserved": candidate_packet.framing_meta == source_packet.framing_meta,
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
    "PR101_DECODER_BLOB_LEN",
    "PR101_LATENT_BLOB_LEN",
    "REPACKABLE_SECTIONS",
    "SCHEMA_VERSION",
    "BrotliRecodeChoice",
    "HnervLowlevelPackError",
    "PackedArchiveView",
    "PackedHnervPayload",
    "SingleMemberArchive",
    "brotli_recode_search",
    "build_lowlevel_brotli_repack_candidate",
    "parse_a1_headered_split_brotli_hnerv",
    "parse_ff_packed_brotli_hnerv",
    "parse_pr101_fixed_offset_microcodec_hnerv",
    "read_packed_archive_view",
    "read_strict_single_member_zip",
    "sha256_bytes",
    "write_stored_single_member_zip",
]
