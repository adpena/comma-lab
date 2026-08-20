# SPDX-License-Identifier: MIT
"""Strict outer ZIP pricing for one exact counted task-space member.

This module treats the inner P/G/A[/T] member as opaque bytes.  It constructs
one canonical ZIP_STORED archive and one canonical ZIP_DEFLATED archive, parses
both back under strict metadata and content custody, and selects the smaller
exact archive.  Parsing deliberately never recompresses DEFLATE data: custody
binds the received archive bytes and their decoded semantics, not a
zlib-version-dependent re-encoding.

The resulting evidence is structural and research-only.  It is not a contest
score, candidate, originality, or promotion claim, and it does not verify the
inner member's task-space semantics.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import re
import stat
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

MEMBER_NAME: Final = "0.bin"
MEMBER_NAME_BYTES: Final = MEMBER_NAME.encode("ascii")
FIXED_DOS_TIME: Final = 0
FIXED_DOS_DATE: Final = 0x0021  # 1980-01-01
VERSION_MADE_BY: Final = (3 << 8) | 20  # Unix, ZIP specification 2.0
STORE_VERSION_NEEDED: Final = 10
DEFLATE_VERSION_NEEDED: Final = 20
CANONICAL_EXTERNAL_ATTR: Final = (stat.S_IFREG | 0o644) << 16
DEFAULT_MAX_MEMBER_BYTES: Final = 64 * 1024 * 1024
MAX_ARCHIVE_OVERHEAD_BYTES: Final = 1024 * 1024

_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_CENTRAL_DIRECTORY_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
_END_OF_CENTRAL_DIRECTORY = struct.Struct("<IHHHHIIH")

_LOCAL_FILE_SIGNATURE: Final = 0x04034B50
_CENTRAL_DIRECTORY_SIGNATURE: Final = 0x02014B50
_END_OF_CENTRAL_DIRECTORY_SIGNATURE: Final = 0x06054B50
_UINT32_MAX: Final = (1 << 32) - 1
_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")


class TaskspaceOuterArchiveError(ValueError):
    """Raised when construction or strict parse-back custody fails."""


class OuterArchiveEncoding(StrEnum):
    """The two admissible outer ZIP encodings."""

    STORED = "zip_stored"
    DEFLATED = "zip_deflated"

    @property
    def zip_method(self) -> int:
        if self is OuterArchiveEncoding.STORED:
            return zipfile.ZIP_STORED
        return zipfile.ZIP_DEFLATED

    @property
    def version_needed(self) -> int:
        if self is OuterArchiveEncoding.STORED:
            return STORE_VERSION_NEEDED
        return DEFLATE_VERSION_NEEDED


@dataclass(frozen=True, slots=True)
class OuterArchiveTruthLabels:
    """Fail-closed authority labels for this structural codec."""

    authority: Literal["structural"] = field(default="structural", init=False)
    research_only: bool = field(default=True, init=False)
    inner_payload_semantics_verified: bool = field(default=False, init=False)
    candidate_claim: bool = field(default=False, init=False)
    score_claim: bool = field(default=False, init=False)
    originality_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ParsedTaskspaceOuterArchive:
    """Strictly parsed exact archive bytes and decoded member custody."""

    encoding: OuterArchiveEncoding
    archive_bytes: bytes
    archive_nbytes: int
    archive_sha256: str
    member_name: str
    member_bytes: bytes
    member_nbytes: int
    member_sha256: str
    member_crc32: int
    compressed_member_nbytes: int


@dataclass(frozen=True, slots=True)
class TaskspaceOuterArchiveBuild:
    """Both exact prices plus the deterministic minimum-price selection."""

    stored: ParsedTaskspaceOuterArchive
    deflated: ParsedTaskspaceOuterArchive
    selected: ParsedTaskspaceOuterArchive
    stored_price_bytes: int
    deflated_price_bytes: int
    selection_reason: Literal["lower_archive_nbytes", "equal_archive_nbytes_prefer_zip_stored"]
    tie_break_rule: Literal["prefer_zip_stored"]
    zlib_compile_version: str
    zlib_runtime_version: str
    deflate_profile: str
    determinism_scope: Literal[
        "exact bytes for recorded zlib compile/runtime/profile; receiver parse never recompresses"
    ]
    truth: OuterArchiveTruthLabels


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TaskspaceOuterArchiveError(f"{field} must be a lowercase 64-character SHA-256 hex digest")


def _as_encoding(value: OuterArchiveEncoding | str, *, field: str) -> OuterArchiveEncoding:
    try:
        return OuterArchiveEncoding(value)
    except ValueError as exc:
        choices = ", ".join(encoding.value for encoding in OuterArchiveEncoding)
        raise TaskspaceOuterArchiveError(f"{field} must be one of: {choices}") from exc


def _deflate_raw(member: bytes) -> bytes:
    compressor = zlib.compressobj(
        level=zlib.Z_BEST_COMPRESSION,
        method=zlib.DEFLATED,
        wbits=-zlib.MAX_WBITS,
        memLevel=9,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )
    return compressor.compress(member) + compressor.flush(zlib.Z_FINISH)


def _build_exact_archive(member: bytes, encoding: OuterArchiveEncoding) -> bytes:
    compressed_member = member if encoding is OuterArchiveEncoding.STORED else _deflate_raw(member)

    member_nbytes = len(member)
    compressed_nbytes = len(compressed_member)
    if member_nbytes > _UINT32_MAX or compressed_nbytes > _UINT32_MAX:
        raise TaskspaceOuterArchiveError("ZIP32 member size limit exceeded")

    crc32 = zlib.crc32(member) & _UINT32_MAX
    local_header = _LOCAL_FILE_HEADER.pack(
        _LOCAL_FILE_SIGNATURE,
        encoding.version_needed,
        0,
        encoding.zip_method,
        FIXED_DOS_TIME,
        FIXED_DOS_DATE,
        crc32,
        compressed_nbytes,
        member_nbytes,
        len(MEMBER_NAME_BYTES),
        0,
    )
    central_directory_offset = len(local_header) + len(MEMBER_NAME_BYTES) + compressed_nbytes
    central_header = _CENTRAL_DIRECTORY_HEADER.pack(
        _CENTRAL_DIRECTORY_SIGNATURE,
        VERSION_MADE_BY,
        encoding.version_needed,
        0,
        encoding.zip_method,
        FIXED_DOS_TIME,
        FIXED_DOS_DATE,
        crc32,
        compressed_nbytes,
        member_nbytes,
        len(MEMBER_NAME_BYTES),
        0,
        0,
        0,
        0,
        CANONICAL_EXTERNAL_ATTR,
        0,
    )
    central_directory_nbytes = len(central_header) + len(MEMBER_NAME_BYTES)
    end_record = _END_OF_CENTRAL_DIRECTORY.pack(
        _END_OF_CENTRAL_DIRECTORY_SIGNATURE,
        0,
        0,
        1,
        1,
        central_directory_nbytes,
        central_directory_offset,
        0,
    )
    return b"".join(
        (
            local_header,
            MEMBER_NAME_BYTES,
            compressed_member,
            central_header,
            MEMBER_NAME_BYTES,
            end_record,
        )
    )


def _inflate_raw_bounded(compressed_member: bytes, *, max_member_bytes: int) -> bytes:
    inflater = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    try:
        member = inflater.decompress(compressed_member, max_member_bytes + 1)
        if len(member) > max_member_bytes or inflater.unconsumed_tail:
            raise TaskspaceOuterArchiveError("DEFLATE member exceeds the configured decoded-size limit")
        member += inflater.flush(max_member_bytes - len(member) + 1)
    except zlib.error as exc:
        raise TaskspaceOuterArchiveError("invalid raw DEFLATE member") from exc

    if len(member) > max_member_bytes:
        raise TaskspaceOuterArchiveError("DEFLATE member exceeds the configured decoded-size limit")
    if not inflater.eof:
        raise TaskspaceOuterArchiveError("DEFLATE member did not reach stream EOF")
    if inflater.unused_data or inflater.unconsumed_tail:
        raise TaskspaceOuterArchiveError("DEFLATE member contains bytes after stream EOF")
    return member


def _parse_raw_archive(
    archive: bytes,
    *,
    max_member_bytes: int,
) -> tuple[OuterArchiveEncoding, bytes, int, int]:
    if len(archive) < _LOCAL_FILE_HEADER.size + _CENTRAL_DIRECTORY_HEADER.size + _END_OF_CENTRAL_DIRECTORY.size:
        raise TaskspaceOuterArchiveError("archive is too short to be a canonical one-member ZIP")

    end_offset = len(archive) - _END_OF_CENTRAL_DIRECTORY.size
    try:
        (
            end_signature,
            disk_number,
            central_disk,
            entries_on_disk,
            entries_total,
            central_nbytes,
            central_offset,
            archive_comment_nbytes,
        ) = _END_OF_CENTRAL_DIRECTORY.unpack_from(archive, end_offset)
    except struct.error as exc:
        raise TaskspaceOuterArchiveError("truncated end-of-central-directory record") from exc

    if end_signature != _END_OF_CENTRAL_DIRECTORY_SIGNATURE:
        raise TaskspaceOuterArchiveError("end-of-central-directory record is not at exact archive EOF")
    if archive_comment_nbytes != 0:
        raise TaskspaceOuterArchiveError("archive comments are forbidden")
    if (disk_number, central_disk, entries_on_disk, entries_total) != (0, 0, 1, 1):
        raise TaskspaceOuterArchiveError("archive must contain exactly one member on one disk")
    if central_offset + central_nbytes != end_offset:
        raise TaskspaceOuterArchiveError("central directory has a gap, overlap, or trailing bytes")
    if central_nbytes < _CENTRAL_DIRECTORY_HEADER.size:
        raise TaskspaceOuterArchiveError("central directory is truncated")

    try:
        (
            central_signature,
            version_made_by,
            version_needed,
            flags,
            method,
            modified_time,
            modified_date,
            crc32,
            compressed_nbytes,
            member_nbytes,
            name_nbytes,
            extra_nbytes,
            member_comment_nbytes,
            disk_start,
            internal_attr,
            external_attr,
            local_offset,
        ) = _CENTRAL_DIRECTORY_HEADER.unpack_from(archive, central_offset)
    except struct.error as exc:
        raise TaskspaceOuterArchiveError("truncated central-directory header") from exc

    if central_signature != _CENTRAL_DIRECTORY_SIGNATURE:
        raise TaskspaceOuterArchiveError("invalid central-directory signature")
    expected_central_nbytes = _CENTRAL_DIRECTORY_HEADER.size + name_nbytes + extra_nbytes + member_comment_nbytes
    if central_nbytes != expected_central_nbytes:
        raise TaskspaceOuterArchiveError("central directory contains extra records or malformed lengths")
    central_name_start = central_offset + _CENTRAL_DIRECTORY_HEADER.size
    central_name_end = central_name_start + name_nbytes
    central_extra_end = central_name_end + extra_nbytes
    central_comment_end = central_extra_end + member_comment_nbytes
    if central_comment_end != end_offset:
        raise TaskspaceOuterArchiveError("central-directory fields do not consume the directory exactly")
    central_name = archive[central_name_start:central_name_end]

    if central_name != MEMBER_NAME_BYTES:
        raise TaskspaceOuterArchiveError("member path must be the canonical path-safe name '0.bin'")
    if extra_nbytes != 0 or member_comment_nbytes != 0:
        raise TaskspaceOuterArchiveError("member extra fields and comments are forbidden")
    if version_made_by != VERSION_MADE_BY:
        raise TaskspaceOuterArchiveError("noncanonical creator-system metadata")
    if flags != 0:
        raise TaskspaceOuterArchiveError("general-purpose flags, encryption, and data descriptors are forbidden")
    if method == zipfile.ZIP_STORED:
        encoding = OuterArchiveEncoding.STORED
    elif method == zipfile.ZIP_DEFLATED:
        encoding = OuterArchiveEncoding.DEFLATED
    else:
        raise TaskspaceOuterArchiveError("only ZIP_STORED and ZIP_DEFLATED members are admissible")
    if version_needed != encoding.version_needed:
        raise TaskspaceOuterArchiveError("noncanonical version-needed metadata")
    if (modified_time, modified_date) != (FIXED_DOS_TIME, FIXED_DOS_DATE):
        raise TaskspaceOuterArchiveError("member timestamp is not canonical 1980-01-01 00:00:00")
    if disk_start != 0 or internal_attr != 0 or external_attr != CANONICAL_EXTERNAL_ATTR:
        raise TaskspaceOuterArchiveError("noncanonical member attribute metadata")
    if local_offset != 0:
        raise TaskspaceOuterArchiveError("local member header must begin at archive offset zero")
    if member_nbytes > max_member_bytes:
        raise TaskspaceOuterArchiveError("declared member size exceeds the configured limit")
    if member_nbytes == 0:
        raise TaskspaceOuterArchiveError("counted task-space member must not be empty")
    if encoding is OuterArchiveEncoding.STORED and compressed_nbytes != member_nbytes:
        raise TaskspaceOuterArchiveError("ZIP_STORED compressed and uncompressed sizes must match")

    try:
        (
            local_signature,
            local_version_needed,
            local_flags,
            local_method,
            local_modified_time,
            local_modified_date,
            local_crc32,
            local_compressed_nbytes,
            local_member_nbytes,
            local_name_nbytes,
            local_extra_nbytes,
        ) = _LOCAL_FILE_HEADER.unpack_from(archive, 0)
    except struct.error as exc:
        raise TaskspaceOuterArchiveError("truncated local member header") from exc

    if local_signature != _LOCAL_FILE_SIGNATURE:
        raise TaskspaceOuterArchiveError("invalid local member signature")
    local_name_start = _LOCAL_FILE_HEADER.size
    local_name_end = local_name_start + local_name_nbytes
    local_extra_end = local_name_end + local_extra_nbytes
    data_end = local_extra_end + local_compressed_nbytes
    if data_end != central_offset:
        raise TaskspaceOuterArchiveError("member data has a gap, overlap, descriptor, or unconsumed bytes")
    if archive[local_name_start:local_name_end] != MEMBER_NAME_BYTES:
        raise TaskspaceOuterArchiveError("local member path is not canonical or path-safe")
    if local_extra_nbytes != 0:
        raise TaskspaceOuterArchiveError("local member extra fields are forbidden")
    local_metadata = (
        local_version_needed,
        local_flags,
        local_method,
        local_modified_time,
        local_modified_date,
        local_crc32,
        local_compressed_nbytes,
        local_member_nbytes,
        local_name_nbytes,
    )
    central_metadata = (
        version_needed,
        flags,
        method,
        modified_time,
        modified_date,
        crc32,
        compressed_nbytes,
        member_nbytes,
        name_nbytes,
    )
    if local_metadata != central_metadata:
        raise TaskspaceOuterArchiveError("local and central member metadata disagree")

    compressed_member = archive[local_extra_end:data_end]
    if encoding is OuterArchiveEncoding.STORED:
        member = compressed_member
    else:
        member = _inflate_raw_bounded(compressed_member, max_member_bytes=max_member_bytes)
    if len(member) != member_nbytes:
        raise TaskspaceOuterArchiveError("decoded member size does not match ZIP metadata")
    if zlib.crc32(member) & _UINT32_MAX != crc32:
        raise TaskspaceOuterArchiveError("decoded member CRC-32 does not match ZIP metadata")

    return encoding, member, crc32, compressed_nbytes


def parse_taskspace_outer_archive(
    archive: bytes,
    *,
    expected_encoding: OuterArchiveEncoding | str | None = None,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> ParsedTaskspaceOuterArchive:
    """Parse one exact canonical outer archive without receiver-side recompression.

    The raw parser enforces exact EOF, one member, canonical metadata, path
    safety, decoded stream EOF, CRC-32, sizes, and optional SHA-256 custody.  A
    second read through :mod:`zipfile` confirms standard-library reopen
    semantics and CRC checking against the same exact archive bytes.
    """

    if not isinstance(archive, bytes):
        raise TaskspaceOuterArchiveError("archive must be immutable bytes")
    if not isinstance(max_member_bytes, int) or isinstance(max_member_bytes, bool) or max_member_bytes < 1:
        raise TaskspaceOuterArchiveError("max_member_bytes must be a positive integer")
    if len(archive) > max_member_bytes + MAX_ARCHIVE_OVERHEAD_BYTES:
        raise TaskspaceOuterArchiveError("archive exceeds the configured member limit plus bounded ZIP overhead")

    archive_sha256 = _sha256(archive)
    if expected_archive_sha256 is not None:
        _require_sha256(expected_archive_sha256, field="expected_archive_sha256")
        if not hmac.compare_digest(archive_sha256, expected_archive_sha256):
            raise TaskspaceOuterArchiveError("exact archive SHA-256 custody mismatch")

    encoding, member, crc32, compressed_nbytes = _parse_raw_archive(
        archive,
        max_member_bytes=max_member_bytes,
    )
    if expected_encoding is not None:
        required_encoding = _as_encoding(expected_encoding, field="expected_encoding")
        if encoding is not required_encoding:
            raise TaskspaceOuterArchiveError(
                f"archive encoding mismatch: expected {required_encoding.value}, observed {encoding.value}"
            )

    member_sha256 = _sha256(member)
    if expected_member_sha256 is not None:
        _require_sha256(expected_member_sha256, field="expected_member_sha256")
        if not hmac.compare_digest(member_sha256, expected_member_sha256):
            raise TaskspaceOuterArchiveError("decoded member SHA-256 custody mismatch")

    try:
        with zipfile.ZipFile(io.BytesIO(archive), mode="r") as reopened:
            infos = reopened.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME:
                raise TaskspaceOuterArchiveError(
                    "standard-library reopen did not observe exactly canonical member 0.bin"
                )
            reopened_member = reopened.read(infos[0])
            if reopened_member != member:
                raise TaskspaceOuterArchiveError("standard-library reopen bytes disagree with strict raw parse")
            if reopened.testzip() is not None:
                raise TaskspaceOuterArchiveError("standard-library CRC verification failed")
    except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise TaskspaceOuterArchiveError("standard-library ZIP reopen failed") from exc

    return ParsedTaskspaceOuterArchive(
        encoding=encoding,
        archive_bytes=archive,
        archive_nbytes=len(archive),
        archive_sha256=archive_sha256,
        member_name=MEMBER_NAME,
        member_bytes=member,
        member_nbytes=len(member),
        member_sha256=member_sha256,
        member_crc32=crc32,
        compressed_member_nbytes=compressed_nbytes,
    )


def build_taskspace_outer_archive(
    counted_member: bytes,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> TaskspaceOuterArchiveBuild:
    """Build, strictly parse, price, and select STORE/DEFLATE alternatives."""

    if not isinstance(counted_member, bytes):
        raise TaskspaceOuterArchiveError("counted_member must be immutable bytes")
    if not isinstance(max_member_bytes, int) or isinstance(max_member_bytes, bool) or max_member_bytes < 1:
        raise TaskspaceOuterArchiveError("max_member_bytes must be a positive integer")
    if len(counted_member) > max_member_bytes:
        raise TaskspaceOuterArchiveError("counted member exceeds the configured size limit")
    if not counted_member:
        raise TaskspaceOuterArchiveError("counted task-space member must not be empty")

    member_sha256 = _sha256(counted_member)
    stored_bytes = _build_exact_archive(counted_member, OuterArchiveEncoding.STORED)
    deflated_bytes = _build_exact_archive(counted_member, OuterArchiveEncoding.DEFLATED)
    stored = parse_taskspace_outer_archive(
        stored_bytes,
        expected_encoding=OuterArchiveEncoding.STORED,
        expected_archive_sha256=_sha256(stored_bytes),
        expected_member_sha256=member_sha256,
        max_member_bytes=max_member_bytes,
    )
    deflated = parse_taskspace_outer_archive(
        deflated_bytes,
        expected_encoding=OuterArchiveEncoding.DEFLATED,
        expected_archive_sha256=_sha256(deflated_bytes),
        expected_member_sha256=member_sha256,
        max_member_bytes=max_member_bytes,
    )

    if stored.archive_nbytes <= deflated.archive_nbytes:
        selected = stored
        selection_reason: Literal["lower_archive_nbytes", "equal_archive_nbytes_prefer_zip_stored"] = (
            "equal_archive_nbytes_prefer_zip_stored"
            if stored.archive_nbytes == deflated.archive_nbytes
            else "lower_archive_nbytes"
        )
    else:
        selected = deflated
        selection_reason = "lower_archive_nbytes"

    return TaskspaceOuterArchiveBuild(
        stored=stored,
        deflated=deflated,
        selected=selected,
        stored_price_bytes=stored.archive_nbytes,
        deflated_price_bytes=deflated.archive_nbytes,
        selection_reason=selection_reason,
        tie_break_rule="prefer_zip_stored",
        zlib_compile_version=zlib.ZLIB_VERSION,
        zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
        deflate_profile="raw-deflate level=9 memLevel=9 strategy=Z_DEFAULT_STRATEGY",
        determinism_scope="exact bytes for recorded zlib compile/runtime/profile; receiver parse never recompresses",
        truth=OuterArchiveTruthLabels(),
    )


__all__ = [
    "DEFAULT_MAX_MEMBER_BYTES",
    "MEMBER_NAME",
    "OuterArchiveEncoding",
    "OuterArchiveTruthLabels",
    "ParsedTaskspaceOuterArchive",
    "TaskspaceOuterArchiveBuild",
    "TaskspaceOuterArchiveError",
    "build_taskspace_outer_archive",
    "parse_taskspace_outer_archive",
]
