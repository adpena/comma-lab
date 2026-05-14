# SPDX-License-Identifier: MIT
"""Deterministic submission-packet compiler oracle.

This module is the Python reference for future Rust/Zig/C/ASM packet tooling.
It does not optimize score-affecting bytes yet. Its first job is to turn a
contest packet into a deterministic conformance manifest and to prove identity
re-emission, so native ports have exact golden vectors to match.
"""
from __future__ import annotations

import binascii
import hashlib
import json
import stat
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Any

from tac.submission_archive import (
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
    validate_archive_member_name,
)

SCHEMA_VERSION = "submission_packet_compiler.v1"
RUNTIME_TREE_SCHEMA_VERSION = "runtime_tree_manifest.v1"
MANIFEST_NAME = "submission_packet_compiler_manifest.json"
TARGET_PROFILE_POLICIES: dict[str, dict[str, Any]] = {
    "contest_one_video_replay": {
        "contest_dispatch_candidate": True,
        "allows_one_video_replay": True,
        "requires_cross_video_generalization": False,
        "allows_optional_device_learning": False,
        "description": (
            "Contest-only overfit replay; all score-affecting replay data and "
            "runtime code must be self-contained in the packet or fixed contest code."
        ),
    },
    "contest_generalized": {
        "contest_dispatch_candidate": True,
        "allows_one_video_replay": False,
        "requires_cross_video_generalization": True,
        "allows_optional_device_learning": False,
        "description": (
            "Contest-compliant generalized packet; no one-video lookup-table or "
            "fixed-replay assumptions may be required for decode."
        ),
    },
    "production_generalized": {
        "contest_dispatch_candidate": False,
        "allows_one_video_replay": False,
        "requires_cross_video_generalization": True,
        "allows_optional_device_learning": False,
        "description": (
            "comma-ai/openpilot production target with deterministic behavior across videos."
        ),
    },
    "production_edge_adaptive": {
        "contest_dispatch_candidate": False,
        "allows_one_video_replay": False,
        "requires_cross_video_generalization": True,
        "allows_optional_device_learning": True,
        "description": (
            "Production-only edge target; optional on-device learning is allowed "
            "only outside contest mode and must have deterministic fallbacks."
        ),
    },
}
TARGET_PROFILES = tuple(TARGET_PROFILE_POLICIES)
MODES = ("inspect", "identity", "canonicalize", "optimize")
SUPPORTED_ZIP_METHODS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
LOCAL_FILE_HEADER_SIG = 0x04034B50
ZIPWIRE_TIMEOUT_SECONDS = 30.0
ZIPWIRE_ARCHIVE_CORE_FIELDS = (
    "bytes",
    "sha256",
    "member_count",
    "duplicate_member_names",
    "blockers",
    "zip_strict",
)
ZIPWIRE_MEMBER_CORE_FIELDS = (
    "name",
    "local_header_name",
    "local_central_name_match",
    "member_order_index",
    "header_offset",
    "data_offset",
    "compress_type",
    "compressed_bytes",
    "uncompressed_bytes",
    "crc32",
    "payload_sha256",
    "compressed_payload_sha256",
    "date_time",
    "flag_bits",
    "external_attr",
    "create_system",
    "unix_permissions",
    "blockers",
)
SCORE_DISPATCH_BLOCKERS = (
    "score_claim_forbidden_without_exact_cuda_auth_eval",
    "dispatch_readiness_forbidden_without_byte_closed_archive_and_exact_cuda_auth_eval",
    "level2_dispatch_claim_required_before_any_remote_exact_eval",
    "pre_submission_compliance_check_required_before_release_or_promotion",
)


class PacketCompilerError(ValueError):
    """Raised when packet compilation cannot proceed deterministically."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_ready_path(path: Path) -> str:
    return path.as_posix()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _local_header_name(raw_zip: bytes, info: zipfile.ZipInfo) -> str:
    return str(_local_header_record(raw_zip, info)["name"])


def _local_header_record(raw_zip: bytes, info: zipfile.ZipInfo) -> dict[str, Any]:
    offset = int(info.header_offset)
    if offset < 0 or offset + 30 > len(raw_zip):
        raise PacketCompilerError(f"local header offset out of range: {offset}")
    (
        sig,
        version_needed,
        flag_bits,
        compress_type,
        mod_time,
        mod_date,
        crc32,
        compressed_size,
        uncompressed_size,
        name_len,
        extra_len,
    ) = struct.unpack_from("<IHHHHHIIIHH", raw_zip, offset)
    if sig != LOCAL_FILE_HEADER_SIG:
        raise PacketCompilerError(f"bad local header signature at offset {offset}")
    start = offset + 30
    end = start + name_len
    if end > len(raw_zip) or end + extra_len > len(raw_zip):
        raise PacketCompilerError(f"local header name/extra out of range at offset {offset}")
    encoding = "utf-8" if flag_bits & (1 << 11) else "cp437"
    extra_start = end
    extra_end = extra_start + extra_len
    return {
        "signature": f"{sig:08x}",
        "version_needed": int(version_needed),
        "flag_bits": int(flag_bits),
        "compress_type": int(compress_type),
        "mod_time": int(mod_time),
        "mod_date": int(mod_date),
        "crc32": f"{crc32:08x}",
        "compressed_bytes": int(compressed_size),
        "uncompressed_bytes": int(uncompressed_size),
        "name": raw_zip[start:end].decode(encoding),
        "name_bytes": int(name_len),
        "extra_bytes": int(extra_len),
        "extra_sha256": _sha256_bytes(raw_zip[extra_start:extra_end]),
        "data_offset": int(extra_end),
    }


def _member_safety_blockers(name: str) -> list[str]:
    try:
        validate_archive_member_name(name)
    except ValueError as exc:
        return [f"unsafe_member_name:{exc}"]
    return []


def _member_order_sha256(names: list[str]) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _zip_datetime_string(date_time: tuple[int, int, int, int, int, int]) -> str:
    year, month, day, hour, minute, second = date_time
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _inspect_archive(archive_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    raw_zip = archive_path.read_bytes()
    member_rows: list[dict[str, Any]] = []
    duplicate_names: list[str] = []
    archive_comment_bytes = 0
    try:
        with zipfile.ZipFile(archive_path) as zf:
            archive_comment_bytes = len(zf.comment)
            if zf.comment:
                blockers.append("archive_comment_not_empty")
            infos = zf.infolist()
            seen: set[str] = set()
            for info in infos:
                if info.filename in seen:
                    duplicate_names.append(info.filename)
                seen.add(info.filename)
            for member_index, info in enumerate(infos):
                row_blockers = _member_safety_blockers(info.filename)
                if info.is_dir():
                    row_blockers.append("directory_member_not_supported_for_contest_packet")
                if info.flag_bits & 1:
                    row_blockers.append("encrypted_member")
                if info.flag_bits & (1 << 3):
                    row_blockers.append("data_descriptor_not_supported_for_deterministic_packet")
                if info.compress_type not in SUPPORTED_ZIP_METHODS:
                    row_blockers.append(f"unsupported_zip_method:{info.compress_type}")
                unix_mode = (int(info.external_attr) >> 16) & 0o177777
                unix_permissions = unix_mode & 0o7777
                unix_file_type = unix_mode & 0o170000
                if unix_file_type and stat.S_ISLNK(unix_file_type):
                    row_blockers.append("symlink_member_not_supported_for_contest_packet")
                if tuple(info.date_time) != DETERMINISTIC_ZIP_DATE_TIME:
                    row_blockers.append(
                        "noncanonical_zip_timestamp:"
                        f"{_zip_datetime_string(tuple(info.date_time))}"
                    )
                if unix_permissions != DETERMINISTIC_ZIP_FILE_MODE:
                    row_blockers.append(
                        "noncanonical_zip_permissions:"
                        f"{oct(unix_permissions)}"
                    )
                if int(info.create_system) != 3:
                    row_blockers.append(f"noncanonical_zip_create_system:{info.create_system}")
                if info.extra:
                    row_blockers.append(f"central_directory_extra_not_empty:{len(info.extra)}")
                if getattr(info, "comment", b""):
                    row_blockers.append(
                        f"central_directory_comment_not_empty:{len(info.comment)}"
                    )
                try:
                    local_header = _local_header_record(raw_zip, info)
                    local_name = str(local_header["name"])
                except PacketCompilerError as exc:
                    local_header = {}
                    local_name = ""
                    row_blockers.append(f"local_header_error:{exc}")
                if local_name and local_name != info.filename:
                    row_blockers.append("local_central_name_mismatch")
                if local_header.get("extra_bytes", 0):
                    row_blockers.append(
                        f"local_header_extra_not_empty:{local_header['extra_bytes']}"
                    )
                payload = b""
                payload_sha256 = ""
                compressed_payload_sha256 = ""
                data_offset = local_header.get("data_offset")
                if isinstance(data_offset, int):
                    data_end = data_offset + int(info.compress_size)
                    if data_offset <= len(raw_zip) and data_end <= len(raw_zip):
                        compressed_payload_sha256 = _sha256_bytes(raw_zip[data_offset:data_end])
                    else:
                        row_blockers.append("compressed_payload_slice_out_of_range")
                computed_crc32 = None
                if not info.is_dir() and not (info.flag_bits & 1):
                    try:
                        payload = zf.read(info)
                        payload_sha256 = _sha256_bytes(payload)
                        computed_crc32 = binascii.crc32(payload) & 0xFFFFFFFF
                        if computed_crc32 != info.CRC:
                            row_blockers.append("crc32_mismatch")
                        if len(payload) != info.file_size:
                            row_blockers.append("uncompressed_size_mismatch")
                    except (RuntimeError, zipfile.BadZipFile, zlib.error) as exc:  # type: ignore[name-defined]
                        row_blockers.append(f"member_payload_read_error:{exc}")
                row = {
                    "name": info.filename,
                    "local_header_name": local_name,
                    "local_central_name_match": bool(local_name == info.filename),
                    "member_order_index": member_index,
                    "header_offset": int(info.header_offset),
                    "data_offset": data_offset,
                    "compress_type": int(info.compress_type),
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "crc32": f"{info.CRC:08x}",
                    "computed_crc32": (
                        f"{computed_crc32:08x}" if computed_crc32 is not None else None
                    ),
                    "payload_sha256": payload_sha256,
                    "payload_bytes": len(payload),
                    "compressed_payload_sha256": compressed_payload_sha256,
                    "date_time": list(info.date_time),
                    "flag_bits": int(info.flag_bits),
                    "external_attr": int(info.external_attr),
                    "external_attr_hex": f"{int(info.external_attr):08x}",
                    "create_system": int(info.create_system),
                    "unix_mode": unix_mode,
                    "unix_permissions": unix_permissions,
                    "unix_permissions_octal": oct(unix_permissions),
                    "local_header": local_header,
                    "central_directory": {
                        "extra_bytes": len(info.extra),
                        "extra_sha256": _sha256_bytes(info.extra),
                        "comment_bytes": len(getattr(info, "comment", b"")),
                    },
                    "blockers": row_blockers,
                }
                member_rows.append(row)
                blockers.extend(f"{info.filename}:{blocker}" for blocker in row_blockers)
    except zipfile.BadZipFile as exc:
        blockers.append(f"bad_zip:{exc}")
    for name in sorted(set(duplicate_names)):
        blockers.append(f"duplicate_archive_member:{name}")
    member_order = [str(row["name"]) for row in member_rows]
    return {
        "path": _json_ready_path(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "member_count": len(member_rows),
        "duplicate_member_names": sorted(set(duplicate_names)),
        "zip_metadata": {
            "canonical_timestamp": list(DETERMINISTIC_ZIP_DATE_TIME),
            "canonical_unix_permissions": DETERMINISTIC_ZIP_FILE_MODE,
            "canonical_unix_permissions_octal": oct(DETERMINISTIC_ZIP_FILE_MODE),
            "archive_comment_bytes": archive_comment_bytes,
            "member_order": member_order,
            "member_order_sha256": _member_order_sha256(member_order),
            "all_timestamps_canonical": all(
                tuple(row["date_time"]) == DETERMINISTIC_ZIP_DATE_TIME
                for row in member_rows
            ),
            "all_permissions_canonical": all(
                row["unix_permissions"] == DETERMINISTIC_ZIP_FILE_MODE
                for row in member_rows
            ),
            "all_local_central_names_match": all(
                bool(row["local_central_name_match"]) for row in member_rows
            ),
        },
        "members": member_rows,
        "blockers": sorted(set(blockers)),
        "zip_strict": not blockers,
    }


def _native_zipwire_not_requested() -> dict[str, Any]:
    return {
        "requested": False,
        "status": "not_requested",
        "matched": None,
        "blockers": [],
    }


def _value_for_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _zipwire_mismatch(path: str, python_value: Any, native_value: Any) -> dict[str, Any]:
    return {
        "path": path,
        "python": _value_for_json(python_value),
        "native": _value_for_json(native_value),
    }


def _compare_native_zipwire(
    python_archive: dict[str, Any],
    native_archive: dict[str, Any],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for field in ZIPWIRE_ARCHIVE_CORE_FIELDS:
        python_value = python_archive.get(field)
        native_value = native_archive.get(field)
        if native_value != python_value:
            mismatches.append(_zipwire_mismatch(f"archive.{field}", python_value, native_value))

    python_members = python_archive.get("members", [])
    native_members = native_archive.get("members", [])
    if not isinstance(native_members, list):
        mismatches.append(_zipwire_mismatch("archive.members", python_members, native_members))
        native_members = []
    if len(native_members) != len(python_members):
        mismatches.append(
            _zipwire_mismatch("archive.members.length", len(python_members), len(native_members))
        )

    for index, python_member in enumerate(python_members):
        if index >= len(native_members) or not isinstance(native_members[index], dict):
            mismatches.append(
                _zipwire_mismatch(f"members[{index}]", python_member, None)
            )
            continue
        native_member = native_members[index]
        for field in ZIPWIRE_MEMBER_CORE_FIELDS:
            python_value = python_member.get(field)
            native_value = native_member.get(field)
            if native_value != python_value:
                mismatches.append(
                    _zipwire_mismatch(
                        f"members[{index}].{field}",
                        python_value,
                        native_value,
                    )
                )

    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "compared_archive_fields": list(ZIPWIRE_ARCHIVE_CORE_FIELDS),
        "compared_member_fields": list(ZIPWIRE_MEMBER_CORE_FIELDS),
    }


def _run_native_zipwire(
    archive_path: Path,
    python_archive: dict[str, Any] | None,
    zipwire_bin: Path | str | None,
) -> tuple[dict[str, Any], list[str]]:
    if zipwire_bin is None:
        return _native_zipwire_not_requested(), []

    section: dict[str, Any] = {
        "requested": True,
        "zipwire_bin": _json_ready_path(Path(zipwire_bin)),
        "archive_path": _json_ready_path(archive_path),
        "status": "not_run",
        "matched": False,
        "blockers": [],
    }
    if python_archive is None:
        section["status"] = "archive_missing"
        blockers = ["native_zipwire:archive_zip_missing"]
        section["blockers"] = blockers
        return section, blockers

    try:
        completed = subprocess.run(
            [str(zipwire_bin), str(archive_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=ZIPWIRE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        blockers = ["native_zipwire:timeout"]
        section.update(
            {
                "status": "timeout",
                "timeout_seconds": ZIPWIRE_TIMEOUT_SECONDS,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "blockers": blockers,
            }
        )
        return section, blockers
    except OSError as exc:
        blockers = ["native_zipwire:execution_error"]
        section.update(
            {
                "status": "execution_error",
                "error": str(exc),
                "blockers": blockers,
            }
        )
        return section, blockers

    section["process"] = {
        "return_code": completed.returncode,
        "stderr": completed.stderr,
    }
    try:
        native_archive = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        blockers = ["native_zipwire:invalid_json"]
        section.update(
            {
                "status": "invalid_json",
                "error": str(exc),
                "stdout": completed.stdout,
                "blockers": blockers,
            }
        )
        return section, blockers
    if not isinstance(native_archive, dict):
        blockers = ["native_zipwire:invalid_json"]
        section.update(
            {
                "status": "invalid_json",
                "error": "zipwire stdout JSON must be an object",
                "archive": native_archive,
                "blockers": blockers,
            }
        )
        return section, blockers

    comparison = _compare_native_zipwire(python_archive, native_archive)
    section["archive"] = native_archive
    section["comparison"] = comparison
    section["matched"] = bool(comparison["matched"])

    blockers = [
        f"native_zipwire:mismatch:{item['path']}"
        for item in comparison["mismatches"]
    ]
    if completed.returncode != 0 and native_archive.get("zip_strict") is True:
        blockers.append(f"native_zipwire:unexpected_nonzero_exit:{completed.returncode}")

    section["status"] = "matched" if not blockers else "mismatch"
    section["blockers"] = blockers
    return section, blockers


def _iter_packet_files(packet_path: Path) -> list[Path]:
    if packet_path.is_file():
        return [packet_path]
    return sorted(
        path
        for path in packet_path.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _runtime_path_blockers(rel: str) -> list[str]:
    blockers: list[str] = []
    if not rel:
        return ["empty_runtime_path"]
    if "\\" in rel:
        blockers.append(f"runtime_path_uses_backslashes:{rel}")
    if "\x00" in rel or any(ord(ch) < 32 for ch in rel):
        blockers.append(f"runtime_path_control_character:{rel}")
    path = PurePosixPath(rel)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        blockers.append(f"runtime_path_not_relative_safe:{rel}")
    if "__MACOSX" in path.parts or any(part.startswith("._") for part in path.parts):
        blockers.append(f"runtime_path_resource_fork_or_macosx:{rel}")
    if any(part.startswith(".") for part in path.parts) or path.name in {".DS_Store", "Thumbs.db"}:
        blockers.append(f"runtime_path_hidden_or_system:{rel}")
    return blockers


def _runtime_tree_rows(packet_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if packet_path.is_file():
        return rows
    for path in _iter_packet_files(packet_path):
        rel = path.relative_to(packet_path).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode if path.is_symlink() else path.stat().st_mode)
        row_blockers = _runtime_path_blockers(rel)
        if path.is_symlink():
            row_blockers.append(f"runtime_symlink_not_supported:{rel}")
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size if not path.is_symlink() else None,
                "sha256": _sha256_file(path) if not path.is_symlink() else None,
                "mode": mode,
                "mode_octal": oct(mode),
                "executable": bool(mode & stat.S_IXUSR),
                "blockers": row_blockers,
            }
        )
    return rows


def _tree_sha256(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["path"])):
        h.update(str(row["path"]).encode("utf-8"))
        h.update(b"\0")
        h.update(str(row["bytes"]).encode("ascii"))
        h.update(b"\0")
        h.update(str(row["sha256"]).encode("ascii"))
        h.update(b"\0")
        h.update(str(row.get("mode", "")).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _runtime_tree_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers = sorted(
        {
            str(blocker)
            for row in rows
            for blocker in row.get("blockers", [])
        }
    )
    return {
        "schema_version": RUNTIME_TREE_SCHEMA_VERSION,
        "tree_sha256": _tree_sha256(rows),
        "file_count": len(rows),
        "files": rows,
        "blockers": blockers,
        "hooks": {
            "pre_submission_compliance_check_arg": "--expected-runtime-tree-sha256",
            "manifest_field": "runtime_tree_manifest.tree_sha256",
            "mode_semantics": "path + bytes + sha256 + POSIX mode",
        },
    }


def _resolve_packet_paths(packet_path: Path) -> tuple[Path, Path | None]:
    if packet_path.is_file():
        return packet_path, None
    archive = packet_path / "archive.zip"
    inflate = packet_path / "inflate.sh"
    return archive, inflate if inflate.exists() else None


def inspect_packet(
    packet_path: Path | str,
    *,
    target_profile: str = "contest_one_video_replay",
    zipwire_bin: Path | str | None = None,
) -> dict[str, Any]:
    """Inspect a contest packet and emit deterministic conformance vectors."""
    packet = Path(packet_path)
    if target_profile not in TARGET_PROFILE_POLICIES:
        raise PacketCompilerError(f"unknown target_profile: {target_profile}")
    if not packet.exists():
        raise PacketCompilerError(f"packet path does not exist: {packet}")

    archive_path, inflate_path = _resolve_packet_paths(packet)
    blockers: list[str] = []
    if not archive_path.is_file():
        blockers.append("archive_zip_missing")
        archive = None
    else:
        archive = _inspect_archive(archive_path)
        blockers.extend(f"archive:{blocker}" for blocker in archive["blockers"])

    if packet.is_dir() and inflate_path is None:
        blockers.append("inflate_sh_missing")
    if packet.is_file():
        blockers.append("archive_only_packet_runtime_missing")

    runtime_rows = _runtime_tree_rows(packet)
    runtime_tree = _runtime_tree_manifest(runtime_rows)
    blockers.extend(f"runtime_tree:{blocker}" for blocker in runtime_tree["blockers"])
    inflate_info = None
    if inflate_path is not None:
        inflate_info = {
            "path": _json_ready_path(inflate_path),
            "bytes": inflate_path.stat().st_size,
            "sha256": _sha256_file(inflate_path),
            "executable": bool(inflate_path.stat().st_mode & stat.S_IXUSR),
        }
        if not inflate_info["executable"]:
            blockers.append("inflate_sh_not_executable")
    native_zipwire, native_blockers = _run_native_zipwire(
        archive_path,
        archive,
        zipwire_bin,
    )
    blockers.extend(native_blockers)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "inspect",
        "target_profile": target_profile,
        "target_profile_policy": dict(TARGET_PROFILE_POLICIES[target_profile]),
        "score_claim": False,
        "promotion_eligible": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "packet": {
            "path": _json_ready_path(packet),
            "kind": "archive" if packet.is_file() else "directory",
            "tree_sha256": runtime_tree["tree_sha256"],
            "files": runtime_rows,
        },
        "runtime_tree_manifest": runtime_tree,
        "archive": archive,
        "native_zipwire": native_zipwire,
        "inflate_sh": inflate_info,
        "contest_compliance": {
            "checked": True,
            "contest_compliant_packet_shape": not blockers,
            "blockers": sorted(set(blockers)),
            "notes": [
                "inspect/identity proof only; no score claim, promotion, or dispatch readiness",
                "exact CUDA auth eval, byte-closed archive custody, strict compliance, and lane claim still required",
            ],
        },
        "score_dispatch_gate": {
            "score_claim": False,
            "promotion_eligible": False,
            "dispatchable": False,
            "ready_for_exact_eval_dispatch": False,
            "evidence_grade": "byte_custody_only",
            "blockers": list(SCORE_DISPATCH_BLOCKERS),
        },
        "golden_vectors": {
            "archive_sha256": archive["sha256"] if archive else None,
            "member_vectors": archive["members"] if archive else [],
            "runtime_tree_sha256": runtime_tree["tree_sha256"],
            "runtime_tree_manifest": runtime_tree,
        },
    }


def _copy_packet_identity(packet_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PacketCompilerError(f"identity output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    if packet_path.is_file():
        dst = output_dir / "archive.zip"
        dst.write_bytes(packet_path.read_bytes())
        copied.append(
            {
                "path": "archive.zip",
                "bytes": dst.stat().st_size,
                "sha256": _sha256_file(dst),
                "mode": stat.S_IMODE(dst.stat().st_mode),
            }
        )
        return copied
    for src in _iter_packet_files(packet_path):
        rel = src.relative_to(packet_path)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        dst.chmod(stat.S_IMODE(src.stat().st_mode))
        copied.append(
            {
                "path": rel.as_posix(),
                "bytes": dst.stat().st_size,
                "sha256": _sha256_file(dst),
                "mode": stat.S_IMODE(dst.stat().st_mode),
            }
        )
    return copied


def compile_packet(
    packet_path: Path | str,
    *,
    mode: str = "inspect",
    target_profile: str = "contest_one_video_replay",
    output_dir: Path | str | None = None,
    zipwire_bin: Path | str | None = None,
) -> dict[str, Any]:
    """Compile a packet in inspect or identity mode.

    ``canonicalize`` and ``optimize`` are intentionally reserved and fail
    closed until their rewrite contracts are implemented and reviewed.
    """
    if mode not in MODES:
        raise PacketCompilerError(f"unknown mode: {mode}")
    if mode in {"canonicalize", "optimize"}:
        raise PacketCompilerError(f"{mode} mode is not implemented; fail closed")
    packet = Path(packet_path)
    manifest = inspect_packet(packet, target_profile=target_profile, zipwire_bin=zipwire_bin)
    manifest["mode"] = mode
    if mode == "identity":
        if output_dir is None:
            raise PacketCompilerError("identity mode requires output_dir")
        out = Path(output_dir)
        copied = _copy_packet_identity(packet, out)
        copied_tree = _tree_sha256(copied)
        manifest["identity_rewrite"] = {
            "output_dir": _json_ready_path(out),
            "copied_files": copied,
            "copied_file_count": len(copied),
            "copied_tree_sha256": copied_tree,
            "byte_identical_to_input_tree": copied_tree
            == manifest["golden_vectors"]["runtime_tree_sha256"],
        }
        (out / MANIFEST_NAME).write_text(_canonical_json(manifest), encoding="utf-8")
    return manifest


def write_manifest(payload: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_canonical_json(payload), encoding="utf-8")


__all__ = [
    "MANIFEST_NAME",
    "MODES",
    "SCHEMA_VERSION",
    "TARGET_PROFILES",
    "PacketCompilerError",
    "compile_packet",
    "inspect_packet",
    "write_manifest",
]
