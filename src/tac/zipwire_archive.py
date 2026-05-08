"""Header-only ZIP archive inspection with optional Rust acceleration.

This module is a narrow bridge to ``runtime-rs/crates/zipwire``. It preserves a
Python fallback so submission/archive validators can adopt the native parser
incrementally and fail closed on parity mismatches before Rust becomes
authoritative for any contest gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from tac.submission_archive import validate_archive_member_name

ZIP_STORED = 0
ZIP_DEFLATED = 8
FLAG_ENCRYPTED = 1 << 0
FLAG_DATA_DESCRIPTOR = 1 << 3
FLAG_UTF8_NAME = 1 << 11


class ZipwireBridgeError(RuntimeError):
    """Raised when the optional Rust bridge is present but unusable."""


def inspect_zip_headers(
    archive_path: Path | str,
    *,
    prefer_rust: bool = True,
    zipwire_bin: Path | str | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Return deterministic header-only ZIP metadata.

    The returned dictionary matches the JSON shape emitted by the Rust
    ``zipwire`` binary. If the binary is absent or unusable, the Python parser
    runs instead. Neither path extracts or decompresses member payloads.
    """
    path = Path(archive_path)
    if prefer_rust:
        binary = resolve_zipwire_binary(zipwire_bin)
        if binary is not None:
            try:
                return inspect_zip_headers_rust(path, binary=Path(binary), timeout_s=timeout_s)
            except ZipwireBridgeError:
                pass
    return inspect_zip_headers_python(path)


def inspect_zip_headers_rust(
    archive_path: Path | str,
    *,
    binary: Path | str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run the Rust ``zipwire`` CLI and return its JSON inspect record."""
    path = Path(archive_path)
    bin_path = Path(binary)
    proc = subprocess.run(  # subprocess-no-check-OK: returncode 0/1 both carry inspect JSON; handled below.
        [str(bin_path), str(path)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise ZipwireBridgeError(
            f"zipwire exited {proc.returncode}: {proc.stderr.strip()}"
        )
    try:
        record = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ZipwireBridgeError("zipwire did not emit valid JSON") from exc
    if not isinstance(record, dict) or "members" not in record or "blockers" not in record:
        raise ZipwireBridgeError("zipwire JSON does not match inspect schema")
    return record


def resolve_zipwire_binary(zipwire_bin: Path | str | None = None) -> Path | None:
    """Resolve the optional ``zipwire`` binary without building Rust code."""
    if zipwire_bin is not None:
        candidate = Path(zipwire_bin)
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None

    candidates: list[Path] = []
    env_bin = os.environ.get("TAC_ZIPWIRE_BIN")
    if env_bin:
        candidates.append(Path(env_bin))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "runtime-rs" / "target" / "release" / "zipwire",
            repo_root / "runtime-rs" / "target" / "debug" / "zipwire",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def inspect_zip_headers_python(archive_path: Path | str) -> dict[str, Any]:
    """Pure-Python fallback for the Rust ``zipwire`` inspect schema."""
    path = Path(archive_path)
    raw = path.read_bytes()
    members: list[dict[str, Any]] = []
    duplicate_names: set[str] = set()
    blockers: list[str] = []

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        blockers.append(f"bad_zip:{exc}")
        return _archive_record(path, raw, [], [], blockers)

    seen: set[str] = set()
    for info in infos:
        if info.filename in seen:
            duplicate_names.add(info.filename)
        seen.add(info.filename)
        member = _inspect_member(raw, info)
        blockers.extend(f"{member['name']}:{blocker}" for blocker in member["blockers"])
        members.append(member)

    for name in sorted(duplicate_names):
        blockers.append(f"duplicate_archive_member:{name}")
    return _archive_record(path, raw, members, sorted(duplicate_names), blockers)


def _archive_record(
    path: Path,
    raw: bytes,
    members: list[dict[str, Any]],
    duplicate_names: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    unique_blockers = sorted(set(blockers))
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "member_count": len(members),
        "duplicate_member_names": duplicate_names,
        "members": members,
        "blockers": unique_blockers,
        "zip_strict": not unique_blockers,
    }


def _inspect_member(raw: bytes, info: zipfile.ZipInfo) -> dict[str, Any]:
    name = info.filename
    row_blockers = _member_safety_blockers(name)
    if name.endswith("/"):
        row_blockers.append("directory_member_not_supported_for_contest_packet")
    if info.flag_bits & FLAG_ENCRYPTED:
        row_blockers.append("encrypted_member")
    if info.flag_bits & FLAG_DATA_DESCRIPTOR:
        row_blockers.append("data_descriptor_member_not_supported")
    if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
        row_blockers.append(f"unsupported_zip_method:{info.compress_type}")

    local_header: dict[str, Any] | None = None
    local_header_name = ""
    payload_offset: int | None = None
    try:
        parsed = _parse_local_header(raw, int(info.header_offset))
    except ValueError as exc:
        row_blockers.append(f"local_header_error:{exc}")
    else:
        local_header_name = str(parsed["name"])
        payload_offset = int(parsed["data_offset"])
        local_header = {
            "flag_bits": int(parsed["flag_bits"]),
            "compress_type": int(parsed["compress_type"]),
            "crc32": f"{int(parsed['crc32']):08x}",
            "compressed_bytes": int(parsed["compressed_bytes"]),
            "uncompressed_bytes": int(parsed["uncompressed_bytes"]),
        }
        _append_mismatch_blockers(row_blockers, info, parsed)

    local_central_name_match = bool(local_header_name) and local_header_name == name
    if local_header_name and not local_central_name_match:
        row_blockers.append("local_central_name_mismatch")

    row_blockers = sorted(set(row_blockers))
    return {
        "name": name,
        "local_header_name": local_header_name,
        "local_central_name_match": local_central_name_match,
        "header_offset": int(info.header_offset),
        "payload_offset": payload_offset,
        "compress_type": int(info.compress_type),
        "compressed_bytes": int(info.compress_size),
        "uncompressed_bytes": int(info.file_size),
        "crc32": f"{int(info.CRC):08x}",
        "flag_bits": int(info.flag_bits),
        "blockers": row_blockers,
        "local_header": local_header,
    }


def _append_mismatch_blockers(
    row_blockers: list[str], info: zipfile.ZipInfo, parsed: dict[str, Any]
) -> None:
    comparisons = [
        ("flag_bits", int(info.flag_bits), int(parsed["flag_bits"])),
        ("compress_type", int(info.compress_type), int(parsed["compress_type"])),
        ("compressed_size", int(info.compress_size), int(parsed["compressed_bytes"])),
        ("uncompressed_size", int(info.file_size), int(parsed["uncompressed_bytes"])),
    ]
    for label, central, local in comparisons:
        if local != central:
            row_blockers.append(f"local_central_{label}_mismatch:{local}!={central}")
    local_crc = int(parsed["crc32"])
    central_crc = int(info.CRC)
    if local_crc != central_crc:
        row_blockers.append(
            f"local_central_crc32_mismatch:{local_crc:08x}!={central_crc:08x}"
        )


def _parse_local_header(raw: bytes, offset: int) -> dict[str, Any]:
    if offset + 30 > len(raw):
        raise ValueError(f"local header offset out of range: {offset}")
    (
        signature,
        _version_needed,
        flag_bits,
        compress_type,
        _mtime,
        _mdate,
        crc32,
        compressed_bytes,
        uncompressed_bytes,
        name_len,
        extra_len,
    ) = struct.unpack_from("<IHHHHHIIIHH", raw, offset)
    if signature != 0x0403_4B50:
        raise ValueError(f"bad local header signature at offset {offset}")
    name_start = offset + 30
    name_end = name_start + int(name_len)
    data_offset = name_end + int(extra_len)
    if name_end > len(raw) or data_offset > len(raw):
        raise ValueError(f"local header name/extra out of range at offset {offset}")
    name = _decode_zip_name(raw[name_start:name_end], int(flag_bits), offset)
    return {
        "flag_bits": int(flag_bits),
        "compress_type": int(compress_type),
        "crc32": int(crc32),
        "compressed_bytes": int(compressed_bytes),
        "uncompressed_bytes": int(uncompressed_bytes),
        "data_offset": data_offset,
        "name": name,
    }


def _decode_zip_name(raw_name: bytes, flag_bits: int, offset: int) -> str:
    if flag_bits & FLAG_UTF8_NAME:
        try:
            return raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"local header name decode error at offset {offset}: invalid utf-8 member name: {exc}"
            ) from exc
    if all(byte < 128 for byte in raw_name):
        return raw_name.decode("ascii")
    raise ValueError(
        f"local header name decode error at offset {offset}: cp437 non-ascii member names are not supported"
    )


def _member_safety_blockers(name: str) -> list[str]:
    if name.endswith("/") and name.count("/") == 1:
        # validate_archive_member_name rejects "." and hidden/system paths, but
        # accepts a normalized "dir" member. Keep directory blocking explicit in
        # _inspect_member and validate the normalized name here.
        normalized = name.rstrip("/")
    else:
        normalized = name.rstrip("/") if name.endswith("/") else name
    try:
        validate_archive_member_name(normalized)
    except ValueError as exc:
        return [f"unsafe_member_name:{exc}"]
    return []
