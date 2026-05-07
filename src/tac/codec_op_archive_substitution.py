"""Deterministic archive-member substitution for CodecOp byte streams.

This module is deliberately narrow: it replaces one physical ZIP member with a
new CodecOp-produced byte stream and records enough custody to route the
candidate into exact CUDA eval later. It does not unpack packed frontier
payloads or infer logical substream layouts; callers must provide a reviewed
substrate-specific packer for that.
"""

from __future__ import annotations

import binascii
import json
import platform
import re
import struct
import sys
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from tac.repo_io import json_text, sha256_bytes, sha256_file

SCHEMA_VERSION = "codec_op_archive_substitution_candidate.v1"
FIXED_SUPPORTED_COMPRESSION_METHODS = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)
PACKED_PAYLOAD_MEMBER_NAMES = frozenset(
    {"p", "renderer_payload.bin", "renderer_payload.bin.br"}
)
_LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ACTIVE_CLAIM_STATUSES = frozenset(
    {
        "active",
        "claimed",
        "queued",
        "pending",
        "running",
        "submitted",
        "in_progress",
    }
)


class ArchiveSubstitutionError(ValueError):
    """Raised when archive substitution would be unsafe or ambiguous."""


@dataclass(frozen=True)
class ArchiveMember:
    """One validated ZIP member and its source payload bytes."""

    name: str
    payload: bytes
    date_time: tuple[int, int, int, int, int, int]
    compress_type: int
    external_attr: int
    internal_attr: int
    create_system: int
    header_offset: int
    compress_size: int
    file_size: int
    crc32: int

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.payload)

    def manifest_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.file_size,
            "sha256": self.sha256,
            "compress_size": self.compress_size,
            "compress_type": self.compress_type,
            "compress_method": _compression_method_name(self.compress_type),
            "crc32": f"{self.crc32:08x}",
            "date_time": list(self.date_time),
            "external_attr": self.external_attr,
            "external_attr_mode": self.external_attr >> 16,
            "internal_attr": self.internal_attr,
            "create_system": self.create_system,
            "header_offset": self.header_offset,
        }


@dataclass(frozen=True)
class ArchiveInspection:
    """Validated archive bytes plus ordered member records."""

    path: Path
    bytes_len: int
    sha256: str
    members: tuple[ArchiveMember, ...]

    def member_names(self) -> list[str]:
        return [member.name for member in self.members]

    def find_member(self, name: str) -> ArchiveMember:
        matches = [member for member in self.members if member.name == name]
        if len(matches) != 1:
            raise ArchiveSubstitutionError(
                f"target member {name!r} matched {len(matches)} archive members"
            )
        return matches[0]


def validate_archive_member_name(name: str) -> str:
    """Validate strict ZIP member names used by contest archive tooling."""

    if not name:
        raise ArchiveSubstitutionError("archive member name is empty")
    if "\\" in name:
        raise ArchiveSubstitutionError(
            f"archive member uses backslashes: {name!r}"
        )
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        raise ArchiveSubstitutionError(
            f"archive member contains control character: {name!r}"
        )
    if re.match(r"^[A-Za-z]:", name):
        raise ArchiveSubstitutionError(
            f"archive member uses windows drive syntax: {name!r}"
        )
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ArchiveSubstitutionError(f"zip-slip archive member path: {name!r}")
    parts = posix_path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ArchiveSubstitutionError(f"zip-slip archive member path: {name!r}")
    if "__MACOSX" in parts:
        raise ArchiveSubstitutionError(f"macOS resource directory member: {name!r}")
    if any(part.startswith("._") for part in parts):
        raise ArchiveSubstitutionError(f"resource fork archive member: {name!r}")
    if any(part.startswith(".") for part in parts):
        raise ArchiveSubstitutionError(f"hidden archive member: {name!r}")
    if posix_path.name in {".DS_Store", "Thumbs.db", "desktop.ini"}:
        raise ArchiveSubstitutionError(f"resource sidecar archive member: {name!r}")
    return name


def inspect_archive_for_substitution(
    archive_path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> ArchiveInspection:
    """Read and validate a source or candidate archive without extracting it."""

    path = Path(archive_path)
    if not path.is_file():
        raise ArchiveSubstitutionError(f"archive path does not exist: {path}")
    archive_bytes = path.read_bytes()
    archive_sha = sha256_bytes(archive_bytes)
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected archive SHA-256")
        if archive_sha.lower() != expected_sha256.lower():
            raise ArchiveSubstitutionError(
                f"archive SHA-256 mismatch for {path}: "
                f"actual={archive_sha} expected={expected_sha256.lower()}"
            )
    if expected_bytes is not None and len(archive_bytes) != int(expected_bytes):
        raise ArchiveSubstitutionError(
            f"archive byte count mismatch for {path}: "
            f"actual={len(archive_bytes)} expected={int(expected_bytes)}"
        )

    try:
        with zipfile.ZipFile(path) as zf:
            if zf.comment:
                raise ArchiveSubstitutionError(
                    f"{path}: ZIP archive comments are unsupported for exact rewrite"
                )
            infos = zf.infolist()
            names = [info.filename for info in infos]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            if duplicate_names:
                raise ArchiveSubstitutionError(
                    f"{path}: duplicate archive member(s): {duplicate_names}"
                )

            members: list[ArchiveMember] = []
            for info in infos:
                if info.is_dir():
                    raise ArchiveSubstitutionError(
                        f"{path}: directory archive members are unsupported: {info.filename!r}"
                    )
                validate_archive_member_name(info.filename)
                local_name = _local_header_name(archive_bytes, info)
                if local_name != info.filename:
                    raise ArchiveSubstitutionError(
                        f"{path}: local header name mismatch: "
                        f"central={info.filename!r} local={local_name!r}"
                    )
                validate_archive_member_name(local_name)
                if info.flag_bits & 0x1:
                    raise ArchiveSubstitutionError(
                        f"{path}: encrypted archive member unsupported: {info.filename!r}"
                    )
                if info.compress_type not in FIXED_SUPPORTED_COMPRESSION_METHODS:
                    raise ArchiveSubstitutionError(
                        f"{path}: unsupported compression method "
                        f"{info.compress_type} for {info.filename!r}"
                    )
                if info.extra or info.comment:
                    raise ArchiveSubstitutionError(
                        f"{path}: member extra/comment metadata unsupported for "
                        f"exact rewrite: {info.filename!r}"
                    )
                payload = zf.read(info)
                crc = binascii.crc32(payload) & 0xFFFFFFFF
                if info.file_size != len(payload):
                    raise ArchiveSubstitutionError(
                        f"{path}: ZIP file_size mismatch for {info.filename!r}"
                    )
                if crc != info.CRC:
                    raise ArchiveSubstitutionError(
                        f"{path}: ZIP CRC mismatch for {info.filename!r}"
                    )
                members.append(
                    ArchiveMember(
                        name=info.filename,
                        payload=payload,
                        date_time=tuple(info.date_time),
                        compress_type=info.compress_type,
                        external_attr=info.external_attr,
                        internal_attr=info.internal_attr,
                        create_system=info.create_system,
                        header_offset=info.header_offset,
                        compress_size=info.compress_size,
                        file_size=info.file_size,
                        crc32=info.CRC,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise ArchiveSubstitutionError(f"{path}: not a readable ZIP archive: {exc}") from exc

    return ArchiveInspection(
        path=path,
        bytes_len=len(archive_bytes),
        sha256=archive_sha,
        members=tuple(members),
    )


def build_archive_substitution_candidate(
    *,
    source_archive: Path | str,
    expected_source_archive_sha256: str,
    expected_source_archive_bytes: int,
    target_member_name: str,
    expected_target_member_sha256: str,
    expected_target_member_bytes: int,
    replacement_substream: bytes,
    output_archive: Path | str,
    candidate_id: str,
    expected_replacement_sha256: str | None = None,
    expected_replacement_bytes: int | None = None,
    allow_packed_payload_container_replacement: bool = False,
    codec_op_manifest: Mapping[str, Any] | None = None,
    exact_runtime_parity: Mapping[str, Any] | None = None,
    lane_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace one validated source archive member and return a custody manifest."""

    target_member_name = validate_archive_member_name(target_member_name)
    if (
        target_member_name in PACKED_PAYLOAD_MEMBER_NAMES
        and not allow_packed_payload_container_replacement
    ):
        raise ArchiveSubstitutionError(
            f"{target_member_name!r} is a packed payload container, not a "
            "substrate substream. Use a reviewed substrate-specific unpack/repack "
            "builder, or pass allow_packed_payload_container_replacement only "
            "when the replacement is a complete packed payload container."
        )

    _require_sha256(expected_source_archive_sha256, "expected source archive SHA-256")
    _require_sha256(expected_target_member_sha256, "expected target member SHA-256")
    if expected_replacement_sha256 is not None:
        _require_sha256(expected_replacement_sha256, "expected replacement SHA-256")

    replacement_sha = sha256_bytes(replacement_substream)
    if expected_replacement_sha256 is not None and replacement_sha.lower() != expected_replacement_sha256.lower():
        raise ArchiveSubstitutionError(
            f"replacement SHA-256 mismatch: actual={replacement_sha} "
            f"expected={expected_replacement_sha256.lower()}"
        )
    if expected_replacement_bytes is not None and len(replacement_substream) != int(expected_replacement_bytes):
        raise ArchiveSubstitutionError(
            f"replacement byte count mismatch: actual={len(replacement_substream)} "
            f"expected={int(expected_replacement_bytes)}"
        )

    source = inspect_archive_for_substitution(
        source_archive,
        expected_sha256=expected_source_archive_sha256,
        expected_bytes=expected_source_archive_bytes,
    )
    old_member = source.find_member(target_member_name)
    if old_member.sha256.lower() != expected_target_member_sha256.lower():
        raise ArchiveSubstitutionError(
            f"target member SHA-256 mismatch for {target_member_name!r}: "
            f"actual={old_member.sha256} expected={expected_target_member_sha256.lower()}"
        )
    if old_member.file_size != int(expected_target_member_bytes):
        raise ArchiveSubstitutionError(
            f"target member byte count mismatch for {target_member_name!r}: "
            f"actual={old_member.file_size} expected={int(expected_target_member_bytes)}"
        )

    output_path = Path(output_archive)
    if output_path.resolve(strict=False) == Path(source_archive).resolve(strict=False):
        raise ArchiveSubstitutionError("refusing to overwrite source archive in place")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", allowZip64=False) as zf:
        for member in source.members:
            payload = (
                replacement_substream
                if member.name == target_member_name
                else member.payload
            )
            _write_member_with_preserved_metadata(zf, member, payload)

    candidate = inspect_archive_for_substitution(output_path)
    if candidate.member_names() != source.member_names():
        raise ArchiveSubstitutionError(
            "candidate archive member order changed unexpectedly: "
            f"source={source.member_names()} candidate={candidate.member_names()}"
        )
    new_member = candidate.find_member(target_member_name)
    if new_member.sha256 != replacement_sha:
        raise ArchiveSubstitutionError(
            f"candidate member payload SHA mismatch for {target_member_name!r}"
        )

    unchanged_metadata_mismatches = _member_metadata_mismatches(
        source.members,
        candidate.members,
        target_member_name=target_member_name,
    )
    no_op = old_member.sha256 == new_member.sha256 and old_member.file_size == new_member.file_size
    readiness = _dispatch_readiness(
        source_archive_sha256=source.sha256,
        candidate_archive_sha256=candidate.sha256,
        no_op=no_op,
        exact_runtime_parity=exact_runtime_parity,
        lane_claim=lane_claim,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "tool": "tac.codec_op_archive_substitution.build_archive_substitution_candidate",
        "score_claim": False,
        "evidence_grade": "archive_construction_only",
        "ready_for_exact_eval_dispatch": readiness["ready_for_exact_eval_dispatch"],
        "dispatch_readiness": readiness,
        "archive": {
            "source_path": Path(source_archive).as_posix(),
            "candidate_path": output_path.as_posix(),
            "old_archive_bytes": source.bytes_len,
            "new_archive_bytes": candidate.bytes_len,
            "archive_byte_delta": candidate.bytes_len - source.bytes_len,
            "old_archive_sha256": source.sha256,
            "new_archive_sha256": candidate.sha256,
            "member_order_preserved": candidate.member_names() == source.member_names(),
        },
        "target_member": {
            "name": target_member_name,
            "old_bytes": old_member.file_size,
            "new_bytes": new_member.file_size,
            "member_byte_delta": new_member.file_size - old_member.file_size,
            "old_compress_size": old_member.compress_size,
            "new_compress_size": new_member.compress_size,
            "member_compress_size_delta": new_member.compress_size - old_member.compress_size,
            "old_sha256": old_member.sha256,
            "new_sha256": new_member.sha256,
            "old_crc32": f"{old_member.crc32:08x}",
            "new_crc32": f"{new_member.crc32:08x}",
            "no_op_payload": no_op,
        },
        "metadata_policy": {
            "preserved_fields": [
                "member_order",
                "filename",
                "date_time",
                "compress_type",
                "external_attr",
                "internal_attr",
                "create_system",
            ],
            "recomputed_fields": [
                "crc32",
                "file_size",
                "compress_size",
                "header_offset",
            ],
            "unsupported_source_metadata": "member extra fields and comments fail closed",
            "unchanged_member_metadata_mismatches": unchanged_metadata_mismatches,
        },
        "replacement_substream": {
            "bytes": len(replacement_substream),
            "sha256": replacement_sha,
            "source": "codec_op_blob_or_substream",
        },
        "build_reproducibility": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "zlib_version": zlib.ZLIB_VERSION,
            "zip64": False,
            "member_order_source": "source_archive_order",
            "timestamp_policy": "preserve_source_member_timestamps",
            "compression_policy": "preserve_source_member_compress_type",
            "deflate_compresslevel": 9,
        },
        "source_members": [member.manifest_record() for member in source.members],
        "candidate_members": [member.manifest_record() for member in candidate.members],
        "remaining_exact_eval_blockers": _remaining_exact_eval_blockers(readiness),
    }
    if codec_op_manifest is not None:
        manifest["codec_op_manifest"] = dict(codec_op_manifest)
    return manifest


def build_archive_substitution_candidate_from_paths(
    *,
    source_archive: Path | str,
    expected_source_archive_sha256: str,
    expected_source_archive_bytes: int,
    target_member_name: str,
    expected_target_member_sha256: str,
    expected_target_member_bytes: int,
    replacement_substream_path: Path | str,
    output_archive: Path | str,
    candidate_id: str,
    expected_replacement_sha256: str | None = None,
    expected_replacement_bytes: int | None = None,
    manifest_output: Path | str | None = None,
    allow_packed_payload_container_replacement: bool = False,
    codec_op_manifest: Mapping[str, Any] | None = None,
    exact_runtime_parity: Mapping[str, Any] | None = None,
    lane_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Path-based wrapper that reads the replacement stream and optionally writes JSON."""

    replacement_path = Path(replacement_substream_path)
    if not replacement_path.is_file():
        raise ArchiveSubstitutionError(
            f"replacement substream path does not exist: {replacement_path}"
        )
    manifest = build_archive_substitution_candidate(
        source_archive=source_archive,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_source_archive_bytes=expected_source_archive_bytes,
        target_member_name=target_member_name,
        expected_target_member_sha256=expected_target_member_sha256,
        expected_target_member_bytes=expected_target_member_bytes,
        replacement_substream=replacement_path.read_bytes(),
        output_archive=output_archive,
        candidate_id=candidate_id,
        expected_replacement_sha256=expected_replacement_sha256,
        expected_replacement_bytes=expected_replacement_bytes,
        allow_packed_payload_container_replacement=allow_packed_payload_container_replacement,
        codec_op_manifest=codec_op_manifest,
        exact_runtime_parity=exact_runtime_parity,
        lane_claim=lane_claim,
    )
    manifest["replacement_substream"]["path"] = replacement_path.as_posix()
    if manifest_output is not None:
        manifest_path = Path(manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json_text(manifest), encoding="utf-8")
    return manifest


def codec_op_candidate_manifest_entry(
    manifest_path: Path | str,
    *,
    candidate_id: str | None,
    replacement_bytes: int,
) -> dict[str, Any]:
    """Load and verify one candidate row from a CodecOp sweep manifest."""

    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ArchiveSubstitutionError(f"{path}: CodecOp manifest must be a JSON object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ArchiveSubstitutionError(f"{path}: CodecOp manifest missing candidates list")
    if candidate_id is None:
        if len(candidates) != 1:
            raise ArchiveSubstitutionError(
                f"{path}: --codec-op-candidate-id is required when manifest "
                f"contains {len(candidates)} candidates"
            )
        selected = candidates[0]
    else:
        selected = next(
            (
                row for row in candidates
                if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
            ),
            None,
        )
    if not isinstance(selected, Mapping):
        raise ArchiveSubstitutionError(
            f"{path}: CodecOp candidate {candidate_id!r} not found"
        )
    manifest_bytes = (
        selected.get("candidate_substream_bytes")
        if selected.get("candidate_substream_bytes") is not None
        else selected.get("bytes_out")
    )
    if manifest_bytes is not None and int(manifest_bytes) != int(replacement_bytes):
        raise ArchiveSubstitutionError(
            f"{path}: CodecOp candidate byte count mismatch: "
            f"manifest={manifest_bytes} replacement={replacement_bytes}"
        )
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "candidate_id": selected.get("candidate_id"),
        "op_module": selected.get("op_module"),
        "op_class": selected.get("op_class"),
        "op_params": selected.get("op_params"),
        "candidate_substream_bytes": manifest_bytes,
        "ready_for_exact_eval_dispatch": selected.get("ready_for_exact_eval_dispatch"),
        "score_claim": selected.get("score_claim"),
        "evidence_semantics": selected.get("evidence_semantics"),
    }


def _write_member_with_preserved_metadata(
    zf: zipfile.ZipFile,
    source_member: ArchiveMember,
    payload: bytes,
) -> None:
    info = zipfile.ZipInfo(
        filename=source_member.name,
        date_time=source_member.date_time,
    )
    info.compress_type = source_member.compress_type
    info.external_attr = source_member.external_attr
    info.internal_attr = source_member.internal_attr
    info.create_system = source_member.create_system
    kwargs: dict[str, Any] = {"compress_type": source_member.compress_type}
    if source_member.compress_type == zipfile.ZIP_DEFLATED:
        kwargs["compresslevel"] = 9
    zf.writestr(info, payload, **kwargs)


def _local_header_name(archive_bytes: bytes, info: zipfile.ZipInfo) -> str:
    offset = info.header_offset
    if offset + 30 > len(archive_bytes):
        raise ArchiveSubstitutionError(
            f"local header extends beyond archive for {info.filename!r}"
        )
    signature = struct.unpack_from("<I", archive_bytes, offset)[0]
    if signature != _LOCAL_FILE_HEADER_SIGNATURE:
        raise ArchiveSubstitutionError(
            f"bad local header signature for {info.filename!r}: 0x{signature:08x}"
        )
    flag_bits = struct.unpack_from("<H", archive_bytes, offset + 6)[0]
    name_len, extra_len = struct.unpack_from("<HH", archive_bytes, offset + 26)
    name_start = offset + 30
    name_end = name_start + name_len
    if name_end + extra_len > len(archive_bytes):
        raise ArchiveSubstitutionError(
            f"local header name/extra extends beyond archive for {info.filename!r}"
        )
    if extra_len:
        raise ArchiveSubstitutionError(
            f"local header extra metadata unsupported for exact rewrite: {info.filename!r}"
        )
    raw_name = archive_bytes[name_start:name_end]
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw_name.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise ArchiveSubstitutionError(
            f"cannot decode local header name for {info.filename!r}: {exc}"
        ) from exc


def _member_metadata_mismatches(
    source_members: tuple[ArchiveMember, ...],
    candidate_members: tuple[ArchiveMember, ...],
    *,
    target_member_name: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for before, after in zip(source_members, candidate_members, strict=True):
        expected = {
            "name": before.name,
            "date_time": before.date_time,
            "compress_type": before.compress_type,
            "external_attr": before.external_attr,
            "internal_attr": before.internal_attr,
            "create_system": before.create_system,
        }
        actual = {
            "name": after.name,
            "date_time": after.date_time,
            "compress_type": after.compress_type,
            "external_attr": after.external_attr,
            "internal_attr": after.internal_attr,
            "create_system": after.create_system,
        }
        changed = {
            key: {"before": expected[key], "after": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        if changed:
            mismatches.append(
                {
                    "member": before.name,
                    "is_target_member": before.name == target_member_name,
                    "changed_fields": changed,
                }
            )
    return mismatches


def _dispatch_readiness(
    *,
    source_archive_sha256: str,
    candidate_archive_sha256: str,
    no_op: bool,
    exact_runtime_parity: Mapping[str, Any] | None,
    lane_claim: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if no_op:
        blockers.append("replacement_payload_matches_source_member")

    parity_supplied = exact_runtime_parity is not None
    lane_claim_supplied = lane_claim is not None
    parity_summary: dict[str, Any] = {"supplied": parity_supplied}
    lane_summary: dict[str, Any] = {"supplied": lane_claim_supplied}

    if exact_runtime_parity is None:
        blockers.append("exact_runtime_parity_not_supplied")
    else:
        parity_ok = _any_true(
            exact_runtime_parity,
            (
                "safe_for_exact_eval_dispatch",
                "exact_runtime_parity",
                "exact_runtime_parity_passed",
                "runtime_parity_passed",
            ),
        )
        parity_summary["passed"] = parity_ok
        if not parity_ok:
            blockers.append("exact_runtime_parity_report_not_passing")
        blockers.extend(
            _sha_mismatch_blockers(
                exact_runtime_parity,
                source_archive_sha256=source_archive_sha256,
                candidate_archive_sha256=candidate_archive_sha256,
                label="runtime_parity",
            )
        )

    if lane_claim is None:
        blockers.append("matching_lane_dispatch_claim_not_supplied")
    else:
        lane_id = lane_claim.get("lane_id") or lane_claim.get("claim_lane_id")
        status = str(lane_claim.get("status") or lane_claim.get("claim_status") or "").lower()
        lane_summary["lane_id"] = lane_id
        lane_summary["status"] = status
        if not lane_id:
            blockers.append("lane_claim_missing_lane_id")
        if status not in _ACTIVE_CLAIM_STATUSES:
            blockers.append(f"lane_claim_not_active:{status or '<missing>'}")
        blockers.extend(
            _sha_mismatch_blockers(
                lane_claim,
                source_archive_sha256=source_archive_sha256,
                candidate_archive_sha256=candidate_archive_sha256,
                label="lane_claim",
            )
        )

    return {
        "ready_for_exact_eval_dispatch": not blockers,
        "blockers": blockers,
        "runtime_parity": parity_summary,
        "lane_claim": lane_summary,
    }


def _remaining_exact_eval_blockers(readiness: Mapping[str, Any]) -> list[str]:
    blockers = list(readiness.get("blockers") or [])
    blockers.append("exact_cuda_auth_eval_not_run_for_candidate_archive")
    blockers.append("exact_auth_eval_adjudication_not_supplied")
    return blockers


def _sha_mismatch_blockers(
    payload: Mapping[str, Any],
    *,
    source_archive_sha256: str,
    candidate_archive_sha256: str,
    label: str,
) -> list[str]:
    blockers: list[str] = []
    source = _first_value(
        payload,
        ("source_archive_sha256", "old_archive_sha256", "baseline_archive_sha256"),
    )
    if isinstance(source, str) and source.lower() != source_archive_sha256.lower():
        blockers.append(f"{label}_source_archive_sha256_mismatch")
    candidate = _first_value(
        payload,
        ("candidate_archive_sha256", "new_archive_sha256", "archive_sha256"),
    )
    if isinstance(candidate, str) and candidate.lower() != candidate_archive_sha256.lower():
        blockers.append(f"{label}_candidate_archive_sha256_mismatch")
    return blockers


def _first_value(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in payload:
            return payload[key]
    for container in ("source_archive", "candidate_archive", "archive", "candidate"):
        nested = payload.get(container)
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
            if "sha256" in nested and any(key.endswith("sha256") for key in keys):
                return nested["sha256"]
    return None


def _any_true(payload: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if payload.get(key) is True:
            return True
    return any(
        isinstance(value, Mapping) and _any_true(value, keys)
        for value in payload.values()
    )


def _require_sha256(value: str, label: str) -> None:
    if not _SHA256_RE.match(value):
        raise ArchiveSubstitutionError(f"{label} must be 64 hex chars")


def _compression_method_name(method: int) -> str:
    if method == zipfile.ZIP_STORED:
        return "stored"
    if method == zipfile.ZIP_DEFLATED:
        return "deflated"
    return f"unsupported_{method}"


__all__ = [
    "PACKED_PAYLOAD_MEMBER_NAMES",
    "SCHEMA_VERSION",
    "ArchiveInspection",
    "ArchiveMember",
    "ArchiveSubstitutionError",
    "build_archive_substitution_candidate",
    "build_archive_substitution_candidate_from_paths",
    "codec_op_candidate_manifest_entry",
    "inspect_archive_for_substitution",
    "validate_archive_member_name",
]
