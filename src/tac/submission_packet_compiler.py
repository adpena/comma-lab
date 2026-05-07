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
import zipfile
import zlib
from pathlib import Path
from typing import Any

from tac.submission_archive import validate_archive_member_name

SCHEMA_VERSION = "submission_packet_compiler.v1"
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
    offset = int(info.header_offset)
    if offset < 0 or offset + 30 > len(raw_zip):
        raise PacketCompilerError(f"local header offset out of range: {offset}")
    sig = struct.unpack_from("<I", raw_zip, offset)[0]
    if sig != LOCAL_FILE_HEADER_SIG:
        raise PacketCompilerError(f"bad local header signature at offset {offset}")
    flag_bits = struct.unpack_from("<H", raw_zip, offset + 6)[0]
    name_len = struct.unpack_from("<H", raw_zip, offset + 26)[0]
    extra_len = struct.unpack_from("<H", raw_zip, offset + 28)[0]
    start = offset + 30
    end = start + name_len
    if end > len(raw_zip) or end + extra_len > len(raw_zip):
        raise PacketCompilerError(f"local header name/extra out of range at offset {offset}")
    encoding = "utf-8" if flag_bits & (1 << 11) else "cp437"
    return raw_zip[start:end].decode(encoding)


def _member_safety_blockers(name: str) -> list[str]:
    try:
        validate_archive_member_name(name)
    except ValueError as exc:
        return [f"unsafe_member_name:{exc}"]
    return []


def _inspect_archive(archive_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    raw_zip = archive_path.read_bytes()
    member_rows: list[dict[str, Any]] = []
    duplicate_names: list[str] = []
    try:
        with zipfile.ZipFile(archive_path) as zf:
            infos = zf.infolist()
            seen: set[str] = set()
            for info in infos:
                if info.filename in seen:
                    duplicate_names.append(info.filename)
                seen.add(info.filename)
            for info in infos:
                row_blockers = _member_safety_blockers(info.filename)
                if info.is_dir():
                    row_blockers.append("directory_member_not_supported_for_contest_packet")
                if info.flag_bits & 1:
                    row_blockers.append("encrypted_member")
                if info.compress_type not in SUPPORTED_ZIP_METHODS:
                    row_blockers.append(f"unsupported_zip_method:{info.compress_type}")
                try:
                    local_name = _local_header_name(raw_zip, info)
                except PacketCompilerError as exc:
                    local_name = ""
                    row_blockers.append(f"local_header_error:{exc}")
                if local_name and local_name != info.filename:
                    row_blockers.append("local_central_name_mismatch")
                payload = b""
                payload_sha256 = ""
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
                    "header_offset": int(info.header_offset),
                    "compress_type": int(info.compress_type),
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "crc32": f"{info.CRC:08x}",
                    "computed_crc32": (
                        f"{computed_crc32:08x}" if computed_crc32 is not None else None
                    ),
                    "payload_sha256": payload_sha256,
                    "date_time": list(info.date_time),
                    "flag_bits": int(info.flag_bits),
                    "external_attr": int(info.external_attr),
                    "create_system": int(info.create_system),
                    "blockers": row_blockers,
                }
                member_rows.append(row)
                blockers.extend(f"{info.filename}:{blocker}" for blocker in row_blockers)
    except zipfile.BadZipFile as exc:
        blockers.append(f"bad_zip:{exc}")
    for name in sorted(set(duplicate_names)):
        blockers.append(f"duplicate_archive_member:{name}")
    return {
        "path": _json_ready_path(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
        "member_count": len(member_rows),
        "duplicate_member_names": sorted(set(duplicate_names)),
        "members": member_rows,
        "blockers": sorted(set(blockers)),
        "zip_strict": not blockers,
    }


def _iter_packet_files(packet_path: Path) -> list[Path]:
    if packet_path.is_file():
        return [packet_path]
    return sorted(path for path in packet_path.rglob("*") if path.is_file())


def _runtime_tree_rows(packet_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if packet_path.is_file():
        return rows
    for path in _iter_packet_files(packet_path):
        rel = path.relative_to(packet_path).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "mode": stat.S_IMODE(path.stat().st_mode),
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
        h.update(b"\n")
    return h.hexdigest()


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
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "inspect",
        "target_profile": target_profile,
        "target_profile_policy": dict(TARGET_PROFILE_POLICIES[target_profile]),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "packet": {
            "path": _json_ready_path(packet),
            "kind": "archive" if packet.is_file() else "directory",
            "tree_sha256": _tree_sha256(runtime_rows),
            "files": runtime_rows,
        },
        "archive": archive,
        "inflate_sh": (
            {
                "path": _json_ready_path(inflate_path),
                "bytes": inflate_path.stat().st_size,
                "sha256": _sha256_file(inflate_path),
                "executable": bool(inflate_path.stat().st_mode & stat.S_IXUSR),
            }
            if inflate_path is not None else None
        ),
        "contest_compliance": {
            "checked": True,
            "contest_compliant_packet_shape": not blockers,
            "blockers": sorted(set(blockers)),
            "notes": [
                "inspect/identity proof only; exact CUDA auth eval and lane claim still required",
            ],
        },
        "golden_vectors": {
            "archive_sha256": archive["sha256"] if archive else None,
            "member_vectors": archive["members"] if archive else [],
            "runtime_tree_sha256": _tree_sha256(runtime_rows),
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
            }
        )
    return copied


def compile_packet(
    packet_path: Path | str,
    *,
    mode: str = "inspect",
    target_profile: str = "contest_one_video_replay",
    output_dir: Path | str | None = None,
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
    manifest = inspect_packet(packet, target_profile=target_profile)
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
