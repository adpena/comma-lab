# SPDX-License-Identifier: MIT
"""Canonical submission archive builder and validator.

This is the SINGLE SOURCE OF TRUTH for what goes into archive.zip.
Every auth eval, every compress run, every deployment MUST use this module
to build or validate the archive. Ad hoc archive construction is forbidden.

The archive measurement disaster of 2026-04-21 (119KB renderer-only archive
evaluated as if it were the 338KB full submission) happened because archive
construction was scattered across 6+ locations with no validation.
This module makes the wrong thing impossible by construction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import shutil
import stat
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # R41 fix: ruff F821 flagged torch.Tensor string annotations as undefined
    # because torch wasn't importable in this scope. TYPE_CHECKING import keeps
    # the runtime light (torch never loaded at module import) while letting type
    # checkers AND ruff resolve the symbol.
    import torch

logger = logging.getLogger(__name__)

DETERMINISTIC_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
DETERMINISTIC_ZIP_FILE_MODE = 0o644
_FORBIDDEN_ARCHIVE_MEMBER_NAMES = {
    "__MACOSX",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}


def validate_archive_member_name(name: str) -> str:
    """Validate a strict submission ZIP member name and return it unchanged."""
    if not name:
        raise ValueError("archive member name is empty")
    if "\x00" in name:
        raise ValueError(f"archive member contains NUL byte: {name!r}")
    if any(ord(ch) < 32 for ch in name):
        raise ValueError(f"archive member contains control character: {name!r}")
    if "\\" in name:
        raise ValueError(f"archive member uses backslashes: {name!r}")
    if re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"archive member uses Windows drive prefix: {name!r}")

    path = PurePosixPath(name)
    if path.is_absolute():
        raise ValueError(f"archive member path is absolute: {name!r}")

    parts = path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"zip-slip archive member path: {name!r}")
    if any(part in _FORBIDDEN_ARCHIVE_MEMBER_NAMES or part.startswith(".") for part in parts):
        raise ValueError(f"hidden/system archive member: {name!r}")

    return name


def deterministic_zip_info(
    arcname: str,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    mode: int = DETERMINISTIC_ZIP_FILE_MODE,
) -> zipfile.ZipInfo:
    """Create a fixed-metadata ZipInfo for byte-stable archive members."""
    arcname = validate_archive_member_name(arcname)
    info = zipfile.ZipInfo(arcname, date_time=DETERMINISTIC_ZIP_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = (mode & 0xFFFF) << 16
    info.create_system = 3
    return info


def write_deterministic_zip_member(
    zf: zipfile.ZipFile,
    arcname: str,
    data: bytes,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int | None = 9,
) -> None:
    """Write bytes with fixed timestamp, permissions, and host system metadata."""
    info = deterministic_zip_info(arcname, compress_type=compress_type)
    kwargs = {"compress_type": compress_type}
    if compresslevel is not None and compress_type != zipfile.ZIP_STORED:
        kwargs["compresslevel"] = compresslevel
    zf.writestr(info, data, **kwargs)


def write_deterministic_zip_file(
    zf: zipfile.ZipFile,
    source_path: Path | str,
    arcname: str,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int | None = 9,
) -> None:
    """Write a file to a ZIP using deterministic member metadata."""
    src = Path(source_path)
    if src.is_symlink():
        raise ValueError(f"Refusing to archive symlink: {src}")
    write_deterministic_zip_member(
        zf,
        arcname,
        src.read_bytes(),
        compress_type=compress_type,
        compresslevel=compresslevel,
    )


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _coerce_typed_sidechannel_member(raw: TypedSidechannelMember | dict) -> TypedSidechannelMember:
    if isinstance(raw, TypedSidechannelMember):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(
            "typed sidechannel entries must be TypedSidechannelMember or dict, "
            f"got {type(raw).__name__}"
        )
    return TypedSidechannelMember(**raw)


def _typed_sidechannel_contract_row(entry: TypedSidechannelMember) -> tuple[dict, Path]:
    member_name = validate_archive_member_name(entry.member_name)
    expected_name = _TYPED_SIDECHANNEL_ALLOWED_MEMBERS.get(entry.kind)
    if expected_name is None:
        allowed = ", ".join(sorted(_TYPED_SIDECHANNEL_ALLOWED_MEMBERS))
        raise ValueError(f"unknown typed sidechannel kind {entry.kind!r}; allowed: {allowed}")
    if member_name != expected_name:
        raise ValueError(
            f"typed sidechannel kind {entry.kind!r} must use member "
            f"{expected_name!r}, got {member_name!r}"
        )
    source_path = Path(entry.source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"typed sidechannel source missing: {source_path}")
    payload = source_path.read_bytes()
    dispatch_blockers: list[str] = []
    if entry.consumed_by_runtime and not entry.runtime_consumer:
        dispatch_blockers.append(f"{entry.kind}_runtime_consumer_missing")
    if entry.consumed_by_runtime and not _is_sha256(entry.runtime_consumption_proof_sha256):
        dispatch_blockers.append(f"{entry.kind}_runtime_consumption_proof_sha256_missing")
    if entry.score_affecting and not entry.consumed_by_runtime:
        dispatch_blockers.append(f"{entry.kind}_score_affecting_member_not_consumed_by_runtime")
    row = {
        "member_name": member_name,
        "kind": entry.kind,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "score_affecting": bool(entry.score_affecting),
        "consumed_by_runtime": bool(entry.consumed_by_runtime),
        "runtime_consumer": entry.runtime_consumer,
        "runtime_consumption_proof_sha256": entry.runtime_consumption_proof_sha256,
        "dispatch_ready": not dispatch_blockers,
        "dispatch_blockers": dispatch_blockers,
        "notes": entry.notes,
    }
    return row, source_path


def build_typed_sidechannel_contract(
    entries: list[TypedSidechannelMember | dict] | tuple[TypedSidechannelMember | dict, ...] | None,
) -> tuple[list[dict], dict[str, Path]]:
    """Normalize typed sidechannel archive metadata and source paths.

    The returned rows are deterministic and suitable for
    ``typed_sidechannels.json``. The path mapping is keyed by archive member
    name and is consumed by ``build_submission_archive``.
    """
    if not entries:
        return [], {}
    rows: list[dict] = []
    sources: dict[str, Path] = {}
    seen: set[str] = set()
    for raw in entries:
        row, source_path = _typed_sidechannel_contract_row(
            _coerce_typed_sidechannel_member(raw)
        )
        name = str(row["member_name"])
        if name in seen:
            raise ValueError(f"duplicate typed sidechannel member: {name!r}")
        seen.add(name)
        rows.append(row)
        sources[name] = source_path
    rows.sort(key=lambda row: (str(row["kind"]), str(row["member_name"])))
    return rows, sources


def _typed_sidechannel_contract_payload(rows: list[dict]) -> dict:
    blockers = sorted(
        {
            str(blocker)
            for row in rows
            for blocker in row.get("dispatch_blockers", [])
        }
    )
    return {
        "schema": TYPED_SIDECHANNEL_CONTRACT_SCHEMA,
        "score_claim": False,
        "dispatch_ready": not blockers,
        "dispatch_blockers": blockers,
        "members": rows,
    }


def _read_typed_sidechannel_contract(zf: zipfile.ZipFile) -> dict | None:
    names = zf.namelist()
    if TYPED_SIDECHANNEL_CONTRACT_MEMBER not in names:
        return None
    try:
        raw = zf.read(TYPED_SIDECHANNEL_CONTRACT_MEMBER)
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{TYPED_SIDECHANNEL_CONTRACT_MEMBER}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{TYPED_SIDECHANNEL_CONTRACT_MEMBER}: contract must be an object")
    if payload.get("schema") != TYPED_SIDECHANNEL_CONTRACT_SCHEMA:
        raise ValueError(
            f"{TYPED_SIDECHANNEL_CONTRACT_MEMBER}: unsupported schema "
            f"{payload.get('schema')!r}"
        )
    members = payload.get("members")
    if not isinstance(members, list):
        raise ValueError(f"{TYPED_SIDECHANNEL_CONTRACT_MEMBER}: members must be a list")
    return payload


def _validate_typed_sidechannel_contract(zf: zipfile.ZipFile) -> tuple[dict | None, list[str], list[str]]:
    """Validate typed sidechannel metadata.

    Returns ``(contract, allowed_member_names, errors)``. Structural errors are
    archive-validation errors; unconsumed score-affecting rows are dispatch
    blockers carried in the contract rather than silent score evidence.
    """
    errors: list[str] = []
    try:
        contract = _read_typed_sidechannel_contract(zf)
    except ValueError as exc:
        return None, [], [str(exc)]
    if contract is None:
        return None, [], []

    allowed: list[str] = [TYPED_SIDECHANNEL_CONTRACT_MEMBER]
    seen: set[str] = set()
    names = set(zf.namelist())
    members = contract["members"]
    for index, raw_row in enumerate(members):
        if not isinstance(raw_row, dict):
            errors.append(f"typed sidechannel row {index} is not an object")
            continue
        member_name = raw_row.get("member_name")
        kind = raw_row.get("kind")
        if not isinstance(member_name, str):
            errors.append(f"typed sidechannel row {index} has non-string member_name")
            continue
        try:
            member_name = validate_archive_member_name(member_name)
        except ValueError as exc:
            errors.append(f"typed sidechannel row {index}: {exc}")
            continue
        expected_name = _TYPED_SIDECHANNEL_ALLOWED_MEMBERS.get(str(kind))
        if expected_name is None:
            errors.append(f"typed sidechannel row {index} has unknown kind {kind!r}")
            continue
        if member_name != expected_name:
            errors.append(
                f"typed sidechannel row {index} kind {kind!r} must use "
                f"{expected_name!r}, got {member_name!r}"
            )
        if member_name in seen:
            errors.append(f"duplicate typed sidechannel member: {member_name!r}")
        seen.add(member_name)
        allowed.append(member_name)
        if member_name not in names:
            errors.append(f"typed sidechannel member missing from archive: {member_name}")
            continue
        data = zf.read(member_name)
        if raw_row.get("bytes") != len(data):
            errors.append(
                f"typed sidechannel {member_name} byte count mismatch: "
                f"contract={raw_row.get('bytes')!r} actual={len(data)}"
            )
        actual_sha = hashlib.sha256(data).hexdigest()
        if raw_row.get("sha256") != actual_sha:
            errors.append(
                f"typed sidechannel {member_name} sha256 mismatch: "
                f"contract={raw_row.get('sha256')!r} actual={actual_sha}"
            )
        if raw_row.get("score_affecting") and raw_row.get("consumed_by_runtime"):
            if not raw_row.get("runtime_consumer"):
                errors.append(f"typed sidechannel {member_name} claims runtime consumption without runtime_consumer")
            if not _is_sha256(raw_row.get("runtime_consumption_proof_sha256")):
                errors.append(
                    f"typed sidechannel {member_name} claims runtime consumption without "
                    "runtime_consumption_proof_sha256"
                )
    return contract, allowed, errors


def deterministic_zip_directory(
    source_dir: Path | str,
    output_path: Path | str,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    compresslevel: int | None = 9,
) -> list[str]:
    """Build a deterministic ZIP from a directory tree.

    Hidden files, macOS resource forks, zip-slip names, and symlinks fail
    closed instead of being included in contest archives.
    """
    src_dir = Path(source_dir)
    out = Path(output_path)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Archive source directory not found: {src_dir}")

    members: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for path in sorted(p for p in src_dir.rglob("*") if p.is_file() or p.is_symlink()):
        arcname = path.relative_to(src_dir).as_posix()
        validate_archive_member_name(arcname)
        if arcname in seen:
            raise ValueError(f"duplicate archive member: {arcname!r}")
        seen.add(arcname)
        members.append((arcname, path))

    out.parent.mkdir(parents=True, exist_ok=True)
    zip_kwargs: dict = {"compression": compress_type}
    if compresslevel is not None and compress_type != zipfile.ZIP_STORED:
        zip_kwargs["compresslevel"] = compresslevel
    with zipfile.ZipFile(out, "w", **zip_kwargs) as zf:
        for arcname, path in members:
            write_deterministic_zip_file(
                zf,
                path,
                arcname,
                compress_type=compress_type,
                compresslevel=compresslevel,
            )
    return [arcname for arcname, _ in members]


def validate_zip_member_infos(infos: list[zipfile.ZipInfo] | tuple[zipfile.ZipInfo, ...]) -> list[str]:
    """Validate strict submission ZIP member metadata and return member names.

    This is the read-side companion to deterministic archive construction.
    It rejects duplicate names, zip-slip paths, hidden/resource-fork members,
    and symlink entries before any extraction or member-level audit consumes
    archive bytes.
    """
    names: list[str] = []
    seen: set[str] = set()
    for info in infos:
        name = validate_archive_member_name(info.filename)
        if name in seen:
            raise ValueError(f"duplicate archive member: {name!r}")
        seen.add(name)
        mode = (info.external_attr >> 16) & 0o170000
        if mode and stat.S_ISLNK(mode):
            raise ValueError(f"archive member is a symlink: {name!r}")
        names.append(name)
    return names


def safe_extract_zip(archive_path: Path | str, destination: Path | str) -> list[str]:
    """Safely extract a ZIP after validating every member.

    Raw ``ZipFile.extractall`` is forbidden for contest archives because it can
    hide duplicate-member, zip-slip, hidden sidecar, and symlink bugs until much
    later in the pipeline. This helper validates the whole archive first, then
    streams each member into ``destination`` with an explicit path containment
    check.
    """
    archive = Path(archive_path)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    dest_root = dest.resolve()

    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        names = validate_zip_member_infos(infos)
        for info, name in zip(infos, names, strict=True):
            target = dest_root / name
            try:
                target.resolve().relative_to(dest_root)
            except ValueError as exc:
                raise ValueError(f"zip-slip extraction target: {name!r}") from exc
            if info.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f"archive directory conflicts with file: {name!r}")
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                raise ValueError(f"archive extraction target already exists: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return names


# ============================================================
# Brotli compression utilities
# ============================================================

def brotli_compress(data: bytes, quality: int = 11, lgwin: int = 24) -> bytes:
    """Compress bytes with Brotli (quality 11 = max, matches Quantizr)."""
    import brotli
    return brotli.compress(data, quality=quality, lgwin=lgwin)


def brotli_decompress(data: bytes) -> bytes:
    """Decompress Brotli-compressed bytes."""
    import brotli
    return brotli.decompress(data)


_SEG_TILE_ACTIONS_BIN = "seg_tile_actions.bin"
_SEG_TILE_ACTIONS_BR = "seg_tile_actions.br"
_SEG_TILE_ACTION_DICT_BIN = "seg_tile_action_dict.bin"
_SEG_TILE_ACTION_DICT_MAGIC = b"TAD1"
_SEG_TILE_ACTION_DICT_HEADER_STRUCT = struct.Struct("<4sHH")
_SEG_TILE_ACTION_SPLIT_MAGIC = b"S1"
_SEG_TILE_ACTION_SPLIT2_MAGIC = b"S2"
_SEG_TILE_ACTION_DEFAULT_COUNT = 108
_SEG_TILE_ACTION_MAX_FRAME_EXCLUSIVE = 10_000
_SEG_TILE_ACTION_DEFAULT_TILE_SIZE = 32
_SEG_TILE_ACTION_MAX_TILE_EXCLUSIVE = (
    384 // _SEG_TILE_ACTION_DEFAULT_TILE_SIZE
) * (512 // _SEG_TILE_ACTION_DEFAULT_TILE_SIZE)
_RENDERER_PAYLOAD_MEMBERS = ("renderer_payload.bin", "renderer_payload.bin.br", "p")
TYPED_SIDECHANNEL_CONTRACT_MEMBER = "typed_sidechannels.json"
TYPED_SIDECHANNEL_CONTRACT_SCHEMA = "typed_sidechannel_members_v1"
_TYPED_SIDECHANNEL_ALLOWED_MEMBERS = {
    "categorical_payload": "categorical_payload.bin",
    "lapose_lfv1": "lapose_foveation_tuples.lfv1",
    "sjkl_residual": "sjkl.bin",
    "jcsp_stream": "jcsp.bin",
    "hnerv_hdm3": "hdm3.bin",
}


@dataclass(frozen=True)
class TypedSidechannelMember:
    """Opt-in charged sidechannel member plus its runtime consumption claim.

    This is intentionally metadata-only. Adding one of these members to an
    archive records byte custody and stack intent, but a score-affecting member
    is not dispatch-ready until a reviewed runtime consumption proof is present.
    """

    member_name: str
    source_path: Path | str
    kind: str
    score_affecting: bool = True
    consumed_by_runtime: bool = False
    runtime_consumer: str = ""
    runtime_consumption_proof_sha256: str = ""
    notes: str = ""


def _read_seg_tile_action_uvarint(buf: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if cursor >= len(buf):
            raise ValueError("seg tile action varint payload ended unexpectedly")
        byte = int(buf[cursor])
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 28:
            raise ValueError("seg tile action varint is too large")


def _parse_sg2_seg_tile_action_records(buf: bytes) -> bytes:
    records: list[tuple[int, int, int]] = []
    cursor = 3 if buf.startswith(b"SG2") else 0
    while cursor < len(buf):
        tile, cursor = _read_seg_tile_action_uvarint(buf, cursor)
        count, cursor = _read_seg_tile_action_uvarint(buf, cursor)
        if count <= 0:
            raise ValueError("seg tile action SG2 group has zero records")
        frame = 0
        for idx in range(count):
            delta, cursor = _read_seg_tile_action_uvarint(buf, cursor)
            frame = delta if idx == 0 else frame + delta
            if frame >= 1 << 16:
                raise ValueError(f"seg tile action SG2 frame does not fit u16: {frame}")
            if cursor >= len(buf):
                raise ValueError("seg tile action SG2 payload ended inside record")
            action = int(buf[cursor])
            cursor += 1
            records.append((int(frame), int(tile), action))
    use_raw5 = any(tile >= 256 for _, tile, _ in records)
    out = bytearray()
    for frame, tile, action in records:
        out += int(frame).to_bytes(2, "little")
        if use_raw5:
            out += int(tile).to_bytes(2, "little")
        else:
            out.append(int(tile))
        out.append(action)
    return b"TA5" + bytes(out) if use_raw5 else bytes(out)


def _parse_split_seg_tile_action_records(buf: bytes) -> bytes:
    cursor = len(_SEG_TILE_ACTION_SPLIT_MAGIC)
    group_count, cursor = _read_seg_tile_action_uvarint(buf, cursor)
    if group_count <= 0 or group_count > 256:
        raise ValueError(f"seg tile action S1 group count out of bounds: {group_count}")

    groups: list[tuple[int, int]] = []
    tile = 0
    total_records = 0
    for group_index in range(group_count):
        tile_delta, cursor = _read_seg_tile_action_uvarint(buf, cursor)
        if group_index == 0:
            tile = tile_delta
        else:
            if tile_delta <= 0:
                raise ValueError("seg tile action S1 tile deltas must increase")
            tile += tile_delta
        if tile >= _SEG_TILE_ACTION_MAX_TILE_EXCLUSIVE:
            raise ValueError(f"seg tile action S1 tile out of bounds: {tile}")
        count, cursor = _read_seg_tile_action_uvarint(buf, cursor)
        if count <= 0:
            raise ValueError("seg tile action S1 group has zero records")
        total_records += count
        if total_records > _SEG_TILE_ACTION_MAX_FRAME_EXCLUSIVE:
            raise ValueError(f"seg tile action S1 record count out of bounds: {total_records}")
        groups.append((tile, count))

    pairs: list[tuple[int, int]] = []
    for tile, count in groups:
        frame = 0
        for idx in range(count):
            delta, cursor = _read_seg_tile_action_uvarint(buf, cursor)
            frame = delta if idx == 0 else frame + delta
            if frame >= _SEG_TILE_ACTION_MAX_FRAME_EXCLUSIVE:
                raise ValueError(f"seg tile action S1 frame out of bounds: {frame}")
            pairs.append((frame, tile))

    actions_end = cursor + total_records
    if actions_end != len(buf):
        raise ValueError(
            f"seg tile action S1 length mismatch: expected {actions_end}, got {len(buf)}"
        )
    out = bytearray()
    for frame, tile in pairs:
        action = int(buf[cursor])
        cursor += 1
        out += int(frame).to_bytes(2, "little")
        out.append(int(tile))
        out.append(action)
    return bytes(out)


def _seg_tile_action_record_size_is_semantically_valid(
    buf: bytes,
    size: int,
    *,
    action_count: int,
    tile_size: int = _SEG_TILE_ACTION_DEFAULT_TILE_SIZE,
) -> tuple[bool, str]:
    if size not in (4, 5):
        return False, f"unsupported record size {size}"
    if len(buf) % size != 0:
        return False, f"length {len(buf)} not divisible by {size}"
    for offset in range(0, len(buf), size):
        frame = int.from_bytes(buf[offset:offset + 2], "little")
        if size == 4:
            tile = int(buf[offset + 2])
            action = int(buf[offset + 3])
        else:
            tile = int.from_bytes(buf[offset + 2:offset + 4], "little")
            action = int(buf[offset + 4])
        if frame < 0 or frame >= _SEG_TILE_ACTION_MAX_FRAME_EXCLUSIVE:
            return False, f"frame out of bounds at offset {offset}: {frame}"
        max_tile = (SEG_H // tile_size) * (SEG_W // tile_size)
        if tile < 0 or tile >= max_tile:
            return False, f"tile out of bounds at offset {offset}: {tile}"
        if action < 0 or action >= action_count:
            return False, f"action out of bounds at offset {offset}: {action}"
    return True, "ok"


def validate_seg_tile_actions_payload(
    raw: bytes,
    *,
    action_count: int | None = None,
    source_name: str = _SEG_TILE_ACTIONS_BIN,
) -> dict[str, int | str]:
    """Validate runtime ``seg_tile_actions`` bytes and resolve raw4/raw5 safely.

    Untagged payloads whose byte length is divisible by both 4 and 5 must be
    semantically resolvable to exactly one wire layout before any exact eval
    dispatch. This mirrors the inflate-time loader guard without importing
    torch or scorer code.
    """
    if action_count is None:
        action_count = _SEG_TILE_ACTION_DEFAULT_COUNT
    if action_count <= 0:
        raise ValueError(f"{source_name}: action_count must be positive, got {action_count}")

    encoding = "raw"
    tile_size = _SEG_TILE_ACTION_DEFAULT_TILE_SIZE
    if raw.startswith(b"TG1"):
        if len(raw) < 5:
            raise ValueError(f"{source_name}: TG1 header is truncated")
        tile_size = int.from_bytes(raw[3:5], "little")
        if tile_size <= 0 or SEG_H % tile_size != 0 or SEG_W % tile_size != 0:
            raise ValueError(f"{source_name}: unsupported TG1 tile_size {tile_size}")
        raw = raw[5:]
        encoding = "TG1"
    if raw.startswith(b"TA4"):
        encoding = f"{encoding}+TA4" if encoding != "raw" else "TA4"
        raw = raw[3:]
        record_size = 4
    elif raw.startswith(b"TA5"):
        encoding = f"{encoding}+TA5" if encoding != "raw" else "TA5"
        raw = raw[3:]
        record_size = 5
    elif raw.startswith(_SEG_TILE_ACTION_SPLIT_MAGIC):
        encoding = f"{encoding}+S1" if encoding != "raw" else "S1"
        raw = _parse_split_seg_tile_action_records(raw)
        record_size = 4
    elif raw.startswith(_SEG_TILE_ACTION_SPLIT2_MAGIC):
        encoding = f"{encoding}+S2" if encoding != "raw" else "S2"
        unpacker = _load_renderer_payload_unpacker()
        raw = unpacker._decode_split2_seg_tile_actions(raw)
        record_size = 4
    elif raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        encoding = f"{encoding}+SG2" if encoding != "raw" else "SG2"
        raw = _parse_sg2_seg_tile_action_records(raw)
        if raw.startswith(b"TA5"):
            raw = raw[3:]
            record_size = 5
        else:
            record_size = 4
    elif len(raw) % 4 == 0 and len(raw) % 5 != 0:
        record_size = 4
    elif len(raw) % 5 == 0 and len(raw) % 4 != 0:
        record_size = 5
    elif not raw:
        record_size = 4
    else:
        valid4, reason4 = _seg_tile_action_record_size_is_semantically_valid(
            raw,
            4,
            action_count=action_count,
            tile_size=tile_size,
        )
        valid5, reason5 = _seg_tile_action_record_size_is_semantically_valid(
            raw,
            5,
            action_count=action_count,
            tile_size=tile_size,
        )
        if valid4 and not valid5:
            record_size = 4
        elif valid5 and not valid4:
            record_size = 5
        else:
            raise ValueError(
                f"{source_name}: ambiguous seg tile action payload length without "
                f"TA4/TA5 header: {len(raw)}; valid4={valid4} ({reason4}); "
                f"valid5={valid5} ({reason5})"
            )

    valid, reason = _seg_tile_action_record_size_is_semantically_valid(
        raw,
        record_size,
        action_count=action_count,
        tile_size=tile_size,
    )
    if not valid:
        raise ValueError(f"{source_name}: invalid seg tile action records: {reason}")
    return {
        "source_name": source_name,
        "encoding": encoding,
        "record_size": record_size,
        "record_count": len(raw) // record_size,
        "action_count": action_count,
        "tile_size": tile_size,
    }


def _seg_tile_action_dict_count(raw: bytes, *, source_name: str) -> int:
    header_size = _SEG_TILE_ACTION_DICT_HEADER_STRUCT.size
    if len(raw) < header_size:
        raise ValueError(f"{source_name} is too short")
    magic, version, count = _SEG_TILE_ACTION_DICT_HEADER_STRUCT.unpack_from(raw, 0)
    if magic != _SEG_TILE_ACTION_DICT_MAGIC or version != 1:
        raise ValueError(
            f"unsupported {source_name}: magic={magic!r} version={version}"
        )
    if count <= 0 or count > 256:
        raise ValueError(f"unreasonable {source_name} count: {count}")
    expected = header_size + count * 3 * 4
    if len(raw) != expected:
        raise ValueError(
            f"{source_name} length mismatch: expected {expected}, got {len(raw)}"
        )
    return count


def _load_renderer_payload_unpacker():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "submissions" / "robust_current" / "unpack_renderer_payload.py"
    if not path.is_file():
        raise FileNotFoundError(f"renderer payload unpacker not found: {path}")
    spec = importlib.util.spec_from_file_location("_tac_submission_archive_unpacker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load renderer payload unpacker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode_packed_renderer_payload_member(name: str, data: bytes) -> bytes:
    if name == "renderer_payload.bin":
        return data
    if name == "renderer_payload.bin.br":
        return brotli_decompress(data)
    try:
        return brotli_decompress(data)
    except Exception:
        # Public fixed-slice p payloads are concatenated Brotli streams, not a
        # single stream. The unpacker understands that raw container.
        return data


def _validate_archive_seg_tile_actions(zf: zipfile.ZipFile) -> list[str]:
    names = set(zf.namelist())
    errors: list[str] = []

    def validate_payload(raw: bytes, dict_raw: bytes | None, source_name: str) -> None:
        try:
            action_count = (
                _seg_tile_action_dict_count(
                    dict_raw,
                    source_name=_SEG_TILE_ACTION_DICT_BIN,
                )
                if dict_raw is not None
                else _SEG_TILE_ACTION_DEFAULT_COUNT
            )
            validate_seg_tile_actions_payload(
                raw,
                action_count=action_count,
                source_name=source_name,
            )
        except Exception as exc:
            errors.append(str(exc))

    if _SEG_TILE_ACTIONS_BIN in names and _SEG_TILE_ACTIONS_BR in names:
        errors.append(
            f"Archive contains both {_SEG_TILE_ACTIONS_BIN} and {_SEG_TILE_ACTIONS_BR}; "
            "refusing ambiguous tile-action payload."
        )
    elif _SEG_TILE_ACTIONS_BIN in names or _SEG_TILE_ACTIONS_BR in names:
        source_name = (
            _SEG_TILE_ACTIONS_BIN
            if _SEG_TILE_ACTIONS_BIN in names
            else _SEG_TILE_ACTIONS_BR
        )
        try:
            raw = zf.read(source_name)
            if source_name.endswith(".br"):
                raw = brotli_decompress(raw)
            dict_raw = (
                zf.read(_SEG_TILE_ACTION_DICT_BIN)
                if _SEG_TILE_ACTION_DICT_BIN in names
                else None
            )
            validate_payload(raw, dict_raw, source_name)
        except Exception as exc:
            errors.append(f"{source_name}: failed to read/decode payload: {exc}")
    elif _SEG_TILE_ACTION_DICT_BIN in names:
        errors.append(
            f"Archive contains {_SEG_TILE_ACTION_DICT_BIN} without "
            f"{_SEG_TILE_ACTIONS_BIN} or {_SEG_TILE_ACTIONS_BR}; refusing no-op dictionary."
        )

    packed_present = [name for name in _RENDERER_PAYLOAD_MEMBERS if name in names]
    if len(packed_present) > 1:
        errors.append(
            "Archive contains multiple packed renderer payload containers: "
            + ", ".join(packed_present)
        )
    elif packed_present:
        packed_name = packed_present[0]
        try:
            unpacker = _load_renderer_payload_unpacker()
            payload = _decode_packed_renderer_payload_member(packed_name, zf.read(packed_name))
            _header, members = unpacker._parse_payload(payload)
        except Exception as exc:
            errors.append(f"{packed_name}: renderer payload preflight failed: {exc}")
        else:
            required_groups = {
                "renderer": {"renderer.bin"},
                "mask": {
                    "masks.mkv",
                    "grayscale.mkv",
                    "masks.alpha4.mkv",
                    "masks.amrc",
                    "masks.nrv",
                    "masks.cmg2",
                    "masks.cmg3",
                    "masks.cdo1",
                    "masks.cdo1.xz",
                    "masks.cdo1.zlib",
                    "masks.cdo1.br",
                    "masks.qma9",
                },
                "pose": {
                    "optimized_poses.pt",
                    "optimized_poses.bin",
                    "optimized_poses.qp1",
                    "optimized_embedding.pt",
                    "poses.pt",
                    "zoom_scalars.bin",
                    "foveation_params.bin",
                },
            }
            member_names = set(members)
            for group, candidates in required_groups.items():
                group_members = member_names & candidates
                if not group_members:
                    errors.append(
                        f"{packed_name}: renderer payload missing logical {group} member"
                    )
                elif not any(members.get(name) for name in group_members):
                    errors.append(
                        f"{packed_name}: renderer payload logical {group} member is empty"
                    )
            actions = members.get(_SEG_TILE_ACTIONS_BIN)
            dict_raw = members.get(_SEG_TILE_ACTION_DICT_BIN)
            if actions is not None:
                validate_payload(actions, dict_raw, f"{packed_name}:{_SEG_TILE_ACTIONS_BIN}")
            elif dict_raw is not None:
                errors.append(
                    f"{packed_name}: contains {_SEG_TILE_ACTION_DICT_BIN} without "
                    f"{_SEG_TILE_ACTIONS_BIN}; refusing no-op dictionary."
                )

    return errors


def validate_archive_seg_tile_actions_payloads(archive_path: Path | str) -> list[str]:
    """Return seg-tile-action payload validation errors for a submission ZIP."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        return _validate_archive_seg_tile_actions(zf)


def compress_file_brotli(
    input_path: Path, output_path: Path | None = None, quality: int = 11
) -> Path:
    """Compress a file with Brotli, producing .br suffix.

    Args:
        input_path: file to compress
        output_path: destination (default: input_path + '.br')
        quality: Brotli quality 0-11 (11 = max, matches Quantizr)

    Returns:
        Path to the compressed file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(input_path.suffix + ".br")
    data = input_path.read_bytes()
    compressed = brotli_compress(data, quality=quality)
    output_path.write_bytes(compressed)
    ratio = len(compressed) / len(data) * 100
    logger.info(
        "Brotli: %s %d B -> %d B (%.1f%%)", input_path.name, len(data), len(compressed), ratio
    )
    print(f"  Brotli: {input_path.name} {len(data):,}B -> {len(compressed):,}B ({ratio:.1f}%)")
    return output_path


def decompress_file_brotli(input_path: Path, output_path: Path | None = None) -> Path:
    """Decompress a .br file.

    Args:
        input_path: Brotli-compressed file (typically *.br)
        output_path: destination (default: strip .br suffix)

    Returns:
        Path to the decompressed file.
    """
    input_path = Path(input_path)
    if output_path is None:
        # Strip .br suffix
        if input_path.suffix == ".br":
            output_path = input_path.with_suffix("")
        else:
            raise ValueError(
                f"Cannot infer output path: {input_path} does not end in .br. "
                f"Provide output_path explicitly."
            )
    data = input_path.read_bytes()
    decompressed = brotli_decompress(data)
    output_path.write_bytes(decompressed)
    logger.info(
        "Brotli decompress: %s %d B -> %s %d B",
        input_path.name, len(data), output_path.name, len(decompressed),
    )
    return output_path


def decompress_brotli_files_in_dir(directory: Path | str) -> int:
    """Decompress all .br files in a directory, removing the .br originals.

    This is the canonical inflate-time decompression step: after the archive
    is extracted, any .br files are decompressed in place before the inflate
    pipeline reads them.

    Args:
        directory: directory to scan for .br files

    Returns:
        Number of files decompressed.
    """
    directory = Path(directory)
    count = 0
    for br_file in sorted(directory.glob("*.br")):
        decompressed_path = decompress_file_brotli(br_file)
        br_file.unlink()  # remove .br, keep decompressed
        logger.info("Decompressed and removed: %s -> %s", br_file.name, decompressed_path.name)
        count += 1
    return count


# ============================================================
# Contest data contract — single source of truth
# Both builder (compress) and inflater (inflate) import these.
# Upstream evaluate.py expects these exact dimensions.
# ============================================================
ORIGINAL_VIDEO_BYTES = 37_545_489
NUM_FRAMES = 1200
NUM_PAIRS = 600
HALF_FRAMES = 600  # For half-frame mask encoding (Quantizr paradigm)
OUT_W, OUT_H = 1164, 874  # Camera output resolution
SEG_W, SEG_H = 512, 384  # SegNet / renderer resolution
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES  # 3,662,409,600
POSE_DIM = 6  # FiLM conditioning vector dimension


@dataclass
class ArchiveManifest:
    """Declares what the archive MUST contain for a given inflate mode."""

    renderer_bin: bool = False
    masks_mkv: bool = False
    # Lane MM / Alpha grayscale-LUT mask payload used by
    # inflate_renderer_grayscale.py.
    grayscale_mkv: bool = False
    # Alpha4 grayscale-LUT mask payload. This is intentionally distinct from
    # legacy masks.mkv because the class-to-gray mapping is not class*63.
    masks_alpha4_mkv: bool = False
    # Yousfi council #8 (2026-04-26): lossless argmax-RLE mask codec
    # (src/tac/lossless/argmax_codec.py). Mutually exclusive with
    # masks_mkv — see required_files() invariant.
    masks_amrc: bool = False
    # Lane 12 NeRV mask payload. Mutually exclusive with masks_mkv and
    # masks_amrc; the inflate path decodes it via tac.nerv_mask_codec.
    masks_nrv: bool = False
    # Charged decoded-mask overlay sidecar. This is not a standalone mask
    # format; it is applied after the selected base mask payload and therefore
    # must not participate in mask-format mutual exclusion.
    masks_cdo1: bool = False
    masks_cdo1_xz: bool = False
    masks_cdo1_zlib: bool = False
    masks_cdo1_br: bool = False
    optimized_poses_pt: bool = False
    optimized_poses_bin: bool = False  # raw fp16 binary (half the size of .pt)
    optimized_embedding_pt: bool = False
    poses_pt: bool = False
    corrections_bin: bool = False
    gradient_corrections_bin: bool = False
    mini_segnet_bin: bool = False
    mini_posenet_bin: bool = False
    posenet_targets_bin: bool = False
    # Unified lossless renderer payload. inflate.sh expands this single
    # member back into renderer.bin + mask payload + pose/zoom payload before
    # dispatching the existing renderer path.
    renderer_payload_bin: bool = False
    renderer_payload_bin_br: bool = False
    renderer_payload_p: bool = False
    # SegMap/Selfcomp-family archive payloads. These pair with grayscale.mkv
    # and optimized poses, but the model payload is not renderer.bin.
    segmap_weights_tar_xz: bool = False
    segmap_payload_bin: bool = False
    class_targets_fp16: bool = False
    # R-radial-zoom 2026-04-25 (Hotz council #94): per-pair scalar zoom from
    # the FoE — replaces 7KB poses.bin with 2.4KB zoom_scalars.bin for renderers
    # with use_zoom_flow=True. inflate_renderer.py already loads this from
    # archive_dir/zoom_scalars.bin (lines 2076-2090), but build_submission_archive
    # had no way to put it in the archive. Now wired.
    zoom_scalars_bin: bool = False
    foveation_params_bin: bool = False

    def required_files(self) -> list[str]:
        mapping = {
            "renderer_bin": "renderer.bin",
            "masks_mkv": "masks.mkv",
            "grayscale_mkv": "grayscale.mkv",
            "masks_alpha4_mkv": "masks.alpha4.mkv",
            "masks_amrc": "masks.amrc",
            "masks_nrv": "masks.nrv",
            "masks_cdo1": "masks.cdo1",
            "masks_cdo1_xz": "masks.cdo1.xz",
            "masks_cdo1_zlib": "masks.cdo1.zlib",
            "masks_cdo1_br": "masks.cdo1.br",
            "optimized_poses_pt": "optimized_poses.pt",
            "optimized_poses_bin": "optimized_poses.bin",
            "optimized_embedding_pt": "optimized_embedding.pt",
            "poses_pt": "poses.pt",
            "corrections_bin": "corrections.bin",
            "gradient_corrections_bin": "gradient_corrections.bin",
            "mini_segnet_bin": "mini_segnet.bin",
            "mini_posenet_bin": "mini_posenet.bin",
            "posenet_targets_bin": "posenet_targets.bin",
            "renderer_payload_bin": "renderer_payload.bin",
            "renderer_payload_bin_br": "renderer_payload.bin.br",
            "renderer_payload_p": "p",
            "segmap_weights_tar_xz": "segmap_weights.tar.xz",
            "segmap_payload_bin": "payload.bin",
            "class_targets_fp16": "class_targets.fp16",
            "zoom_scalars_bin": "zoom_scalars.bin",
            "foveation_params_bin": "foveation_params.bin",
        }
        n_renderer_payload_formats = sum(
            bool(v)
            for v in (
                self.renderer_payload_bin,
                self.renderer_payload_bin_br,
                self.renderer_payload_p,
            )
        )
        if n_renderer_payload_formats > 1:
            raise ValueError(
                "ArchiveManifest: renderer_payload_bin, renderer_payload_bin_br, "
                "and renderer_payload_p are mutually exclusive."
            )
        if n_renderer_payload_formats:
            return [
                mapping[k]
                for k, v in vars(self).items()
                if k in mapping and v
            ]
        n_segmap_payload_formats = sum(
            bool(v)
            for v in (
                self.segmap_weights_tar_xz,
                self.segmap_payload_bin,
            )
        )
        if n_segmap_payload_formats > 1:
            raise ValueError(
                "ArchiveManifest: segmap_weights_tar_xz and segmap_payload_bin "
                "are mutually exclusive."
            )
        if n_segmap_payload_formats and not self.grayscale_mkv:
            raise ValueError(
                "ArchiveManifest: SegMap payloads require grayscale_mkv=True."
            )
        if self.class_targets_fp16 and not n_segmap_payload_formats:
            raise ValueError(
                "ArchiveManifest: class_targets_fp16 requires a SegMap payload."
            )
        n_mask_formats = sum(
            bool(v)
            for v in (
                self.masks_mkv,
                self.grayscale_mkv,
                self.masks_alpha4_mkv,
                self.masks_amrc,
                self.masks_nrv,
            )
        )
        if n_mask_formats > 1:
            raise ValueError(
                "ArchiveManifest: masks_mkv, grayscale_mkv, masks_alpha4_mkv, "
                "masks_amrc, and masks_nrv are mutually exclusive — pick one "
                "mask format per submission archive."
            )
        return [
            mapping[k]
            for k, v in vars(self).items()
            if k in mapping and v
        ]


# The manifest for our current renderer-based submission
RENDERER_SUBMISSION_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_mkv=True,
    optimized_poses_pt=True,
)

# Compact manifest: raw binary poses instead of .pt (saves ~8KB)
RENDERER_COMPACT_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_mkv=True,
    optimized_poses_bin=True,
)

# Radial zoom manifest: replaces optimized_poses_bin with zoom_scalars_bin
# (2.4KB instead of 7KB; per-pair scalar zoom from FoE). Use this for
# renderers trained with use_zoom_flow=True (e.g. GREEN profile).
RENDERER_RADIAL_ZOOM_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_mkv=True,
    zoom_scalars_bin=True,
)

# AMRC mask manifest: drops masks.mkv in favour of the lossless argmax-RLE
# codec (Yousfi council #8, 2026-04-26). Use this when the renderer has
# been trained against a clean (non-AV1-noised) mask source — the AMRC blob
# is byte-identical to the pre-encoder masks, so the train/test mask
# distribution is exactly the same and the renderer cannot leak through
# AV1 dithering artifacts.
RENDERER_AMRC_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_amrc=True,
    optimized_poses_pt=True,
)

RENDERER_NRV_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_nrv=True,
    optimized_poses_pt=True,
)

RENDERER_NRV_COMPACT_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    masks_nrv=True,
    optimized_poses_bin=True,
)

RENDERER_ALPHA4_COMPACT_MANIFEST = ArchiveManifest(
    renderer_bin=True,
    grayscale_mkv=True,
    optimized_poses_bin=True,
)

RENDERER_PACKED_PAYLOAD_MANIFEST = ArchiveManifest(
    renderer_payload_bin=True,
)

RENDERER_PACKED_PAYLOAD_BROTLI_MANIFEST = ArchiveManifest(
    renderer_payload_bin_br=True,
)

RENDERER_PACKED_PAYLOAD_SHORT_BROTLI_MANIFEST = ArchiveManifest(
    renderer_payload_p=True,
)

SEGMAP_SUBMISSION_MANIFEST = ArchiveManifest(
    segmap_weights_tar_xz=True,
    grayscale_mkv=True,
    optimized_poses_pt=True,
)

SEGMAP_LCT_SUBMISSION_MANIFEST = ArchiveManifest(
    segmap_weights_tar_xz=True,
    grayscale_mkv=True,
    optimized_poses_pt=True,
    class_targets_fp16=True,
)

SEGMAP_ARITHMETIC_MANIFEST = ArchiveManifest(
    segmap_payload_bin=True,
    grayscale_mkv=True,
    optimized_poses_pt=True,
)

SEGMAP_ARITHMETIC_LCT_MANIFEST = ArchiveManifest(
    segmap_payload_bin=True,
    grayscale_mkv=True,
    optimized_poses_pt=True,
    class_targets_fp16=True,
)


def detect_pose_manifest(archive_path) -> ArchiveManifest:
    """R38 fix: inspect the archive and return whichever manifest matches
    the pose format actually present. Prior code at validation sites used
    a fixed manifest and reported false-negative MISSING for the other
    pose format. Use this helper at every validation site to auto-pick.
    """
    import zipfile
    try:
        with zipfile.ZipFile(str(archive_path), "r") as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, FileNotFoundError):
        # Default to .pt manifest; the validator will catch the missing zip.
        return RENDERER_SUBMISSION_MANIFEST
    is_brotli_members = any(name.endswith(".br") for name in names)
    logical_names = {
        name[:-3] if name.endswith(".br") else name
        for name in names
    }
    has_pose_bin = "optimized_poses.bin" in names
    if not has_pose_bin:
        has_pose_bin = "optimized_poses.bin" in logical_names
    pose_field = "optimized_poses_bin" if has_pose_bin else "optimized_poses_pt"
    overlay_fields: dict[str, bool] = {}
    if "masks.cdo1" in names:
        overlay_fields["masks_cdo1"] = True
    if "masks.cdo1.xz" in names:
        overlay_fields["masks_cdo1_xz"] = True
    if "masks.cdo1.zlib" in names:
        overlay_fields["masks_cdo1_zlib"] = True
    if "masks.cdo1.br" in names:
        overlay_fields["masks_cdo1_br"] = True
    if "renderer_payload.bin.br" in names:
        return RENDERER_PACKED_PAYLOAD_BROTLI_MANIFEST
    if "renderer_payload.bin" in names:
        return RENDERER_PACKED_PAYLOAD_MANIFEST
    if "p" in names:
        return RENDERER_PACKED_PAYLOAD_SHORT_BROTLI_MANIFEST
    has_lct = "class_targets.fp16" in logical_names
    if "grayscale.mkv" in logical_names and "payload.bin" in logical_names:
        manifest = ArchiveManifest(
            segmap_payload_bin=True,
            grayscale_mkv=True,
            class_targets_fp16=has_lct,
            **{pose_field: True},
            **overlay_fields,
        )
    elif "grayscale.mkv" in logical_names and "segmap_weights.tar.xz" in logical_names:
        manifest = ArchiveManifest(
            segmap_weights_tar_xz=True,
            grayscale_mkv=True,
            class_targets_fp16=has_lct,
            **{pose_field: True},
            **overlay_fields,
        )
    elif "grayscale.mkv" in logical_names:
        manifest = ArchiveManifest(renderer_bin=True, grayscale_mkv=True, **{pose_field: True}, **overlay_fields)
    elif "masks.alpha4.mkv" in logical_names:
        manifest = ArchiveManifest(renderer_bin=True, masks_alpha4_mkv=True, **{pose_field: True}, **overlay_fields)
    elif "masks.nrv" in logical_names:
        manifest = ArchiveManifest(renderer_bin=True, masks_nrv=True, **{pose_field: True}, **overlay_fields)
    elif "masks.amrc" in logical_names:
        manifest = ArchiveManifest(renderer_bin=True, masks_amrc=True, **{pose_field: True}, **overlay_fields)
    else:
        manifest = ArchiveManifest(renderer_bin=True, masks_mkv=True, **{pose_field: True}, **overlay_fields)
    return _brotli_manifest(manifest) if is_brotli_members else manifest


@dataclass
class ArchiveValidationResult:
    """Result of archive validation."""

    valid: bool
    archive_path: Path
    archive_bytes: int
    rate_term: float
    files_found: dict[str, int] = field(default_factory=dict)
    files_missing: list[str] = field(default_factory=list)
    files_unexpected: list[str] = field(default_factory=list)
    typed_sidechannel_contract: dict | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dispatch_ready: bool = True
    dispatch_blockers: list[str] = field(default_factory=list)
    md5: str = ""

    def summary(self) -> str:
        lines = [
            f"Archive: {self.archive_path}",
            f"  Size: {self.archive_bytes:,} bytes ({self.archive_bytes / 1024:.1f} KB)",
            f"  Rate term: 25 * {self.archive_bytes} / {ORIGINAL_VIDEO_BYTES} = {self.rate_term:.4f}",
            f"  MD5: {self.md5}",
            f"  Valid: {self.valid}",
        ]
        if self.files_found:
            lines.append("  Contents:")
            for name, size in sorted(self.files_found.items()):
                lines.append(f"    {name}: {size:,} bytes")
        if self.files_missing:
            lines.append(f"  MISSING: {', '.join(self.files_missing)}")
        if self.files_unexpected:
            lines.append(f"  UNEXPECTED: {', '.join(self.files_unexpected)}")
        if self.typed_sidechannel_contract is not None:
            lines.append(
                "  Typed sidechannels: "
                f"{len(self.typed_sidechannel_contract.get('members', []))}"
            )
        if not self.dispatch_ready:
            lines.append(f"  DISPATCH BLOCKERS: {', '.join(self.dispatch_blockers)}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)


def validate_archive(
    archive_path: Path | str,
    manifest: ArchiveManifest = RENDERER_SUBMISSION_MANIFEST,
    strict: bool = True,
) -> ArchiveValidationResult:
    """Validate a submission archive against a manifest.

    Args:
        archive_path: Path to archive.zip
        manifest: Expected contents declaration
        strict: If True, unexpected files are errors. If False, warnings.

    Returns:
        ArchiveValidationResult with full provenance.

    Raises:
        FileNotFoundError: if archive_path does not exist.
        zipfile.BadZipFile: if file is not a valid zip.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    archive_bytes = archive_path.stat().st_size
    md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    rate_term = 25 * archive_bytes / ORIGINAL_VIDEO_BYTES

    result = ArchiveValidationResult(
        valid=True,
        archive_path=archive_path,
        archive_bytes=archive_bytes,
        rate_term=rate_term,
        md5=md5,
    )

    required = set(manifest.required_files())

    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        archive_names_list = validate_zip_member_infos(infos)
        archive_names = set(archive_names_list)
        for info, name in zip(infos, archive_names_list, strict=True):
            result.files_found[name] = info.file_size
        (
            result.typed_sidechannel_contract,
            typed_sidechannel_allowed,
            typed_sidechannel_errors,
        ) = _validate_typed_sidechannel_contract(zf)
        for err in typed_sidechannel_errors:
            result.errors.append(err)
            result.valid = False
        if result.typed_sidechannel_contract is not None:
            blockers = sorted(
                {
                    str(blocker)
                    for row in result.typed_sidechannel_contract.get("members", [])
                    if isinstance(row, dict)
                    for blocker in row.get("dispatch_blockers", [])
                }
            )
            result.dispatch_blockers.extend(blockers)
            if blockers:
                result.dispatch_ready = False
                result.warnings.append(
                    "typed sidechannel members are structurally byte-closed but "
                    "not dispatch-ready: " + ", ".join(blockers)
                )

        # Check required files present
        for req in required:
            if req not in archive_names:
                result.files_missing.append(req)
                result.errors.append(f"Required file missing: {req}")
                result.valid = False

        # Check for unexpected files
        allowed_names = required | set(typed_sidechannel_allowed)
        for name in archive_names:
            if name not in allowed_names:
                result.files_unexpected.append(name)
                msg = f"Unexpected file in archive: {name}"
                if strict:
                    result.errors.append(msg)
                    result.valid = False
                else:
                    result.warnings.append(msg)

        for err in _validate_archive_seg_tile_actions(zf):
            result.errors.append(err)
            result.valid = False

    # Sanity checks on known file sizes (skip for Brotli-compressed archives
    # since .br files have unpredictable sizes after compression)
    is_brotli = any(name.endswith(".br") for name in result.files_found)

    if not is_brotli:
        if "renderer.bin" in result.files_found:
            size = result.files_found["renderer.bin"]
            if size < 10_000:
                result.errors.append(
                    f"renderer.bin suspiciously small: {size} bytes (expected ~150-300KB)"
                )
                result.valid = False
            elif size > 5_000_000:
                result.warnings.append(
                    f"renderer.bin unusually large: {size} bytes (expected ~150-300KB)"
                )

        for masks_name in ("masks.mkv", "grayscale.mkv", "masks.alpha4.mkv"):
            if masks_name not in result.files_found:
                continue
            size = result.files_found[masks_name]
            if size < 1_000:
                result.errors.append(
                    f"{masks_name} suspiciously small: {size} bytes (expected ~50-200KB)"
                )
                result.valid = False

        if "masks.amrc" in result.files_found:
            size = result.files_found["masks.amrc"]
            if size < 32:
                # AMRC header is ~32 bytes; anything smaller is malformed.
                result.errors.append(
                    f"masks.amrc suspiciously small: {size} bytes "
                    f"(header alone is ~32 bytes)"
                )
                result.valid = False
            elif size > 10_000_000:
                # 10MB is way past anything realistic for 1200 frames at
                # 384x512. Likely a corrupted blob or wrong file.
                result.warnings.append(
                    f"masks.amrc unusually large: {size:,} bytes "
                    f"(expected ~0.5-2MB for 1200 frames)"
                )

        if "masks.nrv" in result.files_found:
            size = result.files_found["masks.nrv"]
            if size < 16:
                result.errors.append(
                    f"masks.nrv suspiciously small: {size} bytes "
                    f"(NRV header alone is larger than this)"
                )
                result.valid = False
            elif size > 5_000_000:
                result.warnings.append(
                    f"masks.nrv unusually large: {size:,} bytes "
                    f"(expected tens of KB for Lane 12)"
                )
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    head = zf.read("masks.nrv")[:4]
                if head != b"NRV1":
                    result.errors.append(
                        f"masks.nrv has bad magic {head!r}; expected b'NRV1'"
                    )
                    result.valid = False
            except Exception as exc:
                result.errors.append(f"failed reading masks.nrv header: {exc!r}")
                result.valid = False

        if "segmap_weights.tar.xz" in result.files_found:
            size = result.files_found["segmap_weights.tar.xz"]
            if size < 6:
                result.errors.append(
                    f"segmap_weights.tar.xz suspiciously small: {size} bytes "
                    "(XZ magic alone is 6 bytes)"
                )
                result.valid = False
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    head = zf.read("segmap_weights.tar.xz")[:6]
                if head != b"\xfd7zXZ\x00":
                    result.errors.append(
                        f"segmap_weights.tar.xz has bad magic {head!r}; expected XZ magic"
                    )
                    result.valid = False
            except Exception as exc:
                result.errors.append(f"failed reading segmap_weights.tar.xz header: {exc!r}")
                result.valid = False

        if manifest.segmap_payload_bin and "payload.bin" in result.files_found:
            size = result.files_found["payload.bin"]
            if size < 4:
                result.errors.append(
                    f"payload.bin suspiciously small: {size} bytes "
                    "(SHv1 magic alone is 4 bytes)"
                )
                result.valid = False
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    head = zf.read("payload.bin")[:4]
                if head != b"SHv1":
                    result.errors.append(
                        f"payload.bin has bad magic {head!r}; expected b'SHv1'"
                    )
                    result.valid = False
            except Exception as exc:
                result.errors.append(f"failed reading payload.bin header: {exc!r}")
                result.valid = False

        if "class_targets.fp16" in result.files_found:
            size = result.files_found["class_targets.fp16"]
            if size != 10:
                result.errors.append(
                    f"class_targets.fp16 has {size} bytes; expected exactly 10 bytes "
                    "for five fp16 class targets"
                )
                result.valid = False

        for cdo1_name in ("masks.cdo1", "masks.cdo1.xz", "masks.cdo1.zlib", "masks.cdo1.br"):
            if cdo1_name not in result.files_found:
                continue
            size = result.files_found[cdo1_name]
            if size < 10:
                result.errors.append(
                    f"{cdo1_name} suspiciously small: {size} bytes "
                    f"(CDO1 fixed header is 10 bytes)"
                )
                result.valid = False
            elif size > 5_000_000:
                result.warnings.append(
                    f"{cdo1_name} unusually large: {size:,} bytes "
                    f"(expected small charged overlay sidecar)"
                )
            if cdo1_name == "masks.cdo1":
                try:
                    with zipfile.ZipFile(archive_path, "r") as zf:
                        head = zf.read("masks.cdo1")[:4]
                    if head != b"CDO1":
                        result.errors.append(
                            f"masks.cdo1 has bad magic {head!r}; expected b'CDO1'"
                        )
                        result.valid = False
                except Exception as exc:
                    result.errors.append(f"failed reading masks.cdo1 header: {exc!r}")
                    result.valid = False

        if "optimized_poses.pt" in result.files_found:
            size = result.files_found["optimized_poses.pt"]
            # 600 pairs * 6 values * 2 bytes (fp16) = 7.2KB minimum
            if size < 1_000:
                result.errors.append(
                    f"optimized_poses.pt suspiciously small: {size} bytes (expected ~7-20KB)"
                )
                result.valid = False

    return result


def compute_rate_term(archive_path: Path | str) -> float:
    """Compute the rate term for a submission archive.

    Returns: 25 * archive_bytes / original_video_bytes
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    return 25 * archive_path.stat().st_size / ORIGINAL_VIDEO_BYTES


def save_poses_binary(poses: torch.Tensor, output_path: Path | str) -> int:
    """Save poses as raw fp16 binary (minimal overhead, ~7.2KB for 600×6).

    Args:
        poses: (N, 6) float tensor of optimized pose vectors
        output_path: output .bin file path

    Returns:
        File size in bytes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = poses.half().cpu().numpy().tobytes()
    output_path.write_bytes(raw)
    return len(raw)


# Pickle / PyTorch checkpoint magic bytes. We detect format by content, not by
# filename suffix, because wrappers have repeatedly renamed `.pt` → `.bin` (the
# 2026-04-26 SHIRAZ auth-eval crash) and a suffix-only loader will silently
# torch.frombuffer() over pickle bytes and reshape into nonsense.
_PICKLE_MAGICS: tuple[bytes, ...] = (
    b"\x80\x02",      # pickle protocol 2
    b"\x80\x03",      # pickle protocol 3
    b"\x80\x04",      # pickle protocol 4
    b"\x80\x05",      # pickle protocol 5
    b"PK\x03\x04",    # ZIP (PyTorch >=1.6 default torch.save container)
)


def _looks_like_pickle(raw: bytes) -> bool:
    return any(raw.startswith(m) for m in _PICKLE_MAGICS)


def load_optimized_poses(
    path: Path | str,
    pose_dim: int = 6,
    expected_n_pairs: int | None = None,
) -> torch.Tensor:
    """Load optimized poses from EITHER a torch.save pickle (.pt) OR raw fp16
    binary (.bin), detected by content.

    This is the canonical loader for every consumer that touches a pose file
    (auth_eval_renderer, inflate_renderer, postfilter pipeline, etc.). Always
    use this — never call torch.frombuffer / torch.load directly on a pose
    artifact, because suffix-based dispatch has burned us repeatedly.

    Args:
        path: file path. Suffix is informational, NOT trusted.
        pose_dim: number of pose dimensions per pair (default 6).
        expected_n_pairs: if given, raise unless the loaded tensor has exactly
            this many rows. Pass 600 in eval contexts to catch partial-TTO
            artifacts being shipped as final.

    Returns:
        (N, pose_dim) float32 tensor.

    Raises:
        FileNotFoundError if path missing.
        ValueError with a specific, actionable diagnostic on any mismatch.
    """
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pose file not found: {p}")
    raw = p.read_bytes()
    n_bytes = len(raw)
    if n_bytes == 0:
        raise ValueError(f"Pose file is empty: {p}")

    # Branch A: pickle / torch.save container — load as object then validate.
    if _looks_like_pickle(raw):
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(
                f"Pose file {p.name} starts with pickle/zip magic but torch.load "
                f"failed: {e!r}. File size={n_bytes}B. If this was meant to be a "
                f"raw fp16 archive artifact, regenerate via save_poses_binary()."
            ) from e
        # Lane LR: LoRA-encoded pose pickle. The top-level object is a dict
        # with sentinel "format" == "lora_pose_v1" — materialise back to
        # (N, pose_dim) transparently so downstream consumers see a vanilla
        # tensor and need no LoRA awareness. See src/tac/lora_pose.py for
        # the on-disk schema.
        #
        # Lane LR-V2 (2026-04-27): "lora_pose_v2" is the LEARNABLE-rank
        # variant — same materialisation contract (base + U@V), but the
        # ranks were data-driven via per-rank gates during training and
        # gated/pruned at serialisation time. The reader does not need to
        # know about the gate; U has the gate already absorbed.
        from tac.lora_pose import decode_lora_poses_dict, is_lora_poses_dict
        from tac.lora_pose_v2 import (
            decode_lora_v2_poses_dict,
            is_lora_v2_poses_dict,
        )
        from tac.pose_delta_codec import (
            decode_pose_deltas,
            is_pose_delta_dict,
        )
        from tac.pose_delta_codec_v2 import (
            decode_pose_delta_v2,
            is_pose_delta_v2_dict,
        )
        if is_pose_delta_v2_dict(obj):
            # Lane PD-V2: arithmetic-coded per-pair pose deltas. The dict's
            # "blob" key holds the PDV2 binary container (magic-byte sniffed
            # against b"PDV2" in is_pose_delta_v2_dict). See
            # src/tac/pose_delta_codec_v2.py for the on-disk schema.
            poses = decode_pose_delta_v2(obj["blob"], pose_dim=pose_dim)
        elif is_pose_delta_dict(obj):
            # Lane PD: per-pair pose deltas + anchor + per-channel scale.
            # See src/tac/pose_delta_codec.py for the on-disk schema.
            poses = decode_pose_deltas(obj, pose_dim=pose_dim)
        elif is_lora_poses_dict(obj):
            poses = decode_lora_poses_dict(obj, pose_dim=pose_dim)
        elif is_lora_v2_poses_dict(obj):
            poses = decode_lora_v2_poses_dict(obj, pose_dim=pose_dim)
        elif not isinstance(obj, torch.Tensor):
            raise ValueError(
                f"Pose file {p.name} is a pickle but contains "
                f"{type(obj).__name__}, not Tensor or LoRA-encoded dict. "
                f"Wrappers must not pickle arbitrary dicts/lists into a "
                f"pose artifact."
            )
        else:
            poses = obj.detach().to(torch.float32).cpu()
            if poses.ndim != 2 or poses.shape[-1] != pose_dim:
                raise ValueError(
                    f"Pose tensor shape {tuple(poses.shape)} from {p.name} does "
                    f"not match expected (N, {pose_dim}). Wrong pose_dim or "
                    f"transposed export?"
                )

    # Branch B: raw fp16 buffer.
    else:
        elem_bytes = 2  # float16
        row_bytes = pose_dim * elem_bytes
        if n_bytes % row_bytes != 0:
            raise ValueError(
                f"Raw pose buffer {p.name}: file size {n_bytes}B is not a "
                f"multiple of pose_dim*{elem_bytes} ({row_bytes}B per row). "
                f"This usually means the file is a torch.save pickle that was "
                f"renamed to .bin without conversion (the 2026-04-26 SHIRAZ "
                f"bug), or a partial TTO write. Inspect the first 8 bytes: "
                f"{raw[:8]!r}."
            )
        poses = (
            torch.frombuffer(bytearray(raw), dtype=torch.float16)
            .reshape(-1, pose_dim)
            .float()
        )

    if expected_n_pairs is not None and poses.shape[0] != expected_n_pairs:
        raise ValueError(
            f"Pose count mismatch in {p.name}: got {poses.shape[0]} pairs, "
            f"expected {expected_n_pairs}. This is the partial-TTO-shipped-as-"
            f"final bug pattern (2026-04-26 SHIRAZ: 60 of 600 pairs saved "
            f"because TTO was killed mid-run, then the wrapper used "
            f"`*_partial.pt` as the archive artifact). Re-run optimize_poses "
            f"to completion or use the canonical .bin emit."
        )
    return poses


def load_poses_binary(path: Path | str, pose_dim: int = 6) -> torch.Tensor:
    """Load poses from raw fp16 binary.

    Now defers to load_optimized_poses() so callers get content-based format
    detection (pickle-renamed-to-.bin no longer silently corrupts) and a clear
    error on any malformed buffer. Kept for backward compatibility.

    Returns:
        (N, pose_dim) float32 tensor
    """
    return load_optimized_poses(path, pose_dim=pose_dim)


def build_submission_archive(
    output_path: Path | str,
    renderer_bin: Path | str | None = None,
    masks_mkv: Path | str | None = None,
    grayscale_mkv: Path | str | None = None,
    masks_alpha4_mkv: Path | str | None = None,
    masks_amrc: Path | str | None = None,
    masks_nrv: Path | str | None = None,
    masks_cdo1: Path | str | None = None,
    masks_cdo1_xz: Path | str | None = None,
    masks_cdo1_zlib: Path | str | None = None,
    masks_cdo1_br: Path | str | None = None,
    segmap_weights_tar_xz: Path | str | None = None,
    segmap_payload_bin: Path | str | None = None,
    class_targets_fp16: Path | str | None = None,
    renderer_payload_bin: Path | str | None = None,
    renderer_payload_bin_br: Path | str | None = None,
    renderer_payload_p: Path | str | None = None,
    optimized_poses_pt: Path | str | None = None,
    optimized_poses_bin: Path | str | None = None,
    optimized_embedding_pt: Path | str | None = None,
    gradient_corrections_bin: Path | str | None = None,
    zoom_scalars_bin: Path | str | None = None,
    foveation_params_bin: Path | str | None = None,
    typed_sidechannels: list[TypedSidechannelMember | dict] | tuple[TypedSidechannelMember | dict, ...] | None = None,
    manifest: ArchiveManifest = RENDERER_SUBMISSION_MANIFEST,
    validate: bool = True,
    use_brotli: bool = False,
    brotli_quality: int = 11,
) -> ArchiveValidationResult:
    """Build and validate a submission archive.

    This is the ONLY correct way to build an archive for submission or eval.

    Args:
        output_path: Where to write archive.zip
        renderer_bin: Path to renderer.bin
        masks_mkv: Path to masks.mkv
        grayscale_mkv: Path to grayscale.mkv
        masks_alpha4_mkv: Path to masks.alpha4.mkv
        masks_nrv: Path to masks.nrv
        masks_cdo1: Optional charged CDO1 overlay sidecar
        masks_cdo1_xz: Optional XZ-compressed charged CDO1 overlay sidecar
        masks_cdo1_zlib: Optional zlib-compressed charged CDO1 overlay sidecar
        masks_cdo1_br: Optional Brotli-compressed charged CDO1 overlay sidecar
        segmap_weights_tar_xz: Path to SegMap model payload
        segmap_payload_bin: Path to SHv1 arithmetic SegMap payload
        class_targets_fp16: Optional 5-class fp16 target payload
        renderer_payload_bin: Optional source-runtime-native renderer payload
        renderer_payload_bin_br: Optional Brotli-packed renderer payload
        renderer_payload_p: Optional short-name packed renderer payload
        optimized_poses_pt: Path to optimized_poses.pt
        optimized_poses_bin: Path to optimized_poses.bin (raw fp16, smaller)
        optimized_embedding_pt: Path to optimized_embedding.pt (optional)
        typed_sidechannels: Optional typed sidechannel members. These add
            member bytes plus ``typed_sidechannels.json`` metadata, but do not
            imply dispatch readiness unless runtime consumption proof is present.
        manifest: Expected contents manifest
        validate: If True, validate after building (default True)
        use_brotli: If True, Brotli-compress each artifact before adding
            to the ZIP. Archive will contain .br files (e.g. renderer.bin.br).
            The inflate side must call decompress_brotli_files_in_dir()
            after extraction.
        brotli_quality: Brotli quality level 0-11 (default 11 = max,
            matches Quantizr). Only used when use_brotli=True.

    Returns:
        ArchiveValidationResult with full provenance.

    Raises:
        FileNotFoundError: if any required source file is missing.
        ValueError: if validation fails.
    """
    output_path = Path(output_path)
    typed_sidechannel_rows, typed_sidechannel_sources = build_typed_sidechannel_contract(
        typed_sidechannels
    )

    # Map manifest fields to source paths
    source_map: dict[str, Path | None] = {
        "renderer.bin": Path(renderer_bin) if renderer_bin else None,
        "masks.mkv": Path(masks_mkv) if masks_mkv else None,
        "grayscale.mkv": Path(grayscale_mkv) if grayscale_mkv else None,
        "masks.alpha4.mkv": Path(masks_alpha4_mkv) if masks_alpha4_mkv else None,
        "masks.amrc": Path(masks_amrc) if masks_amrc else None,
        "masks.nrv": Path(masks_nrv) if masks_nrv else None,
        "masks.cdo1": Path(masks_cdo1) if masks_cdo1 else None,
        "masks.cdo1.xz": Path(masks_cdo1_xz) if masks_cdo1_xz else None,
        "masks.cdo1.zlib": Path(masks_cdo1_zlib) if masks_cdo1_zlib else None,
        "masks.cdo1.br": Path(masks_cdo1_br) if masks_cdo1_br else None,
        "segmap_weights.tar.xz": Path(segmap_weights_tar_xz) if segmap_weights_tar_xz else None,
        "payload.bin": Path(segmap_payload_bin) if segmap_payload_bin else None,
        "class_targets.fp16": Path(class_targets_fp16) if class_targets_fp16 else None,
        "renderer_payload.bin": Path(renderer_payload_bin) if renderer_payload_bin else None,
        "renderer_payload.bin.br": Path(renderer_payload_bin_br) if renderer_payload_bin_br else None,
        "p": Path(renderer_payload_p) if renderer_payload_p else None,
        "optimized_poses.pt": Path(optimized_poses_pt) if optimized_poses_pt else None,
        "optimized_poses.bin": Path(optimized_poses_bin) if optimized_poses_bin else None,
        "optimized_embedding.pt": Path(optimized_embedding_pt) if optimized_embedding_pt else None,
        "gradient_corrections.bin": Path(gradient_corrections_bin) if gradient_corrections_bin else None,
        "zoom_scalars.bin": Path(zoom_scalars_bin) if zoom_scalars_bin else None,
        "foveation_params.bin": Path(foveation_params_bin) if foveation_params_bin else None,
    }

    required = manifest.required_files()
    typed_sidechannel_contract = _typed_sidechannel_contract_payload(
        typed_sidechannel_rows
    ) if typed_sidechannel_rows else None

    # Verify all required source files exist BEFORE creating the archive
    for name in required:
        src = source_map.get(name)
        if src is None:
            raise FileNotFoundError(
                f"Required artifact {name} not provided. "
                f"Pass it as an argument to build_submission_archive()."
            )
        if not src.exists():
            raise FileNotFoundError(
                f"Required artifact {name} not found at {src}. "
                f"Build it first, then pass the path."
            )

    # ── Validate artifact integrity before building ──
    # Mask frame count: must be NUM_FRAMES (1200) or HALF_FRAMES (600)
    masks_sources = [
        ("masks.mkv", source_map.get("masks.mkv")),
        ("grayscale.mkv", source_map.get("grayscale.mkv")),
        ("masks.alpha4.mkv", source_map.get("masks.alpha4.mkv")),
    ]
    for masks_name, masks_src in masks_sources:
        if not masks_src or not masks_src.exists():
            continue
        import subprocess as _sp
        try:
            probe = _sp.run(
                ["ffprobe", "-v", "error", "-count_frames",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", str(masks_src)],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "ffprobe is required to validate mask frame count but was not found. "
                "Install ffmpeg (which includes ffprobe) to enable mask validation."
            ) from exc
        except _sp.TimeoutExpired:
            logger.warning(
                "ffprobe timed out on %s — mask frame count NOT validated. "
                "File may be corrupted or on a slow mount.", masks_src,
            )
            probe = None
        if probe is not None and probe.returncode != 0:
            logger.warning(
                "ffprobe returned non-zero (%d) for %s — mask frame count NOT validated. "
                "stderr: %s", probe.returncode, masks_src, probe.stderr.strip(),
            )
        elif probe is not None and not probe.stdout.strip():
            logger.warning(
                "ffprobe returned empty output for %s — mask frame count NOT validated.",
                masks_src,
            )
        else:
            n_mask_frames = int(probe.stdout.strip())
            if n_mask_frames not in (NUM_FRAMES, HALF_FRAMES):
                raise ValueError(
                    f"{masks_name} has {n_mask_frames} frames. "
                    f"Expected {NUM_FRAMES} (full) or {HALF_FRAMES} (half-frame). "
                    f"Rebuild masks with correct frame count."
                )
            logger.info("Mask frame count: %d (%s)",
                        n_mask_frames,
                        "full" if n_mask_frames == NUM_FRAMES else "half-frame")

    nrv_src = source_map.get("masks.nrv")
    if nrv_src and nrv_src.exists():
        head = nrv_src.read_bytes()[:4]
        if head != b"NRV1":
            raise ValueError(
                f"masks.nrv has bad magic {head!r}; expected b'NRV1'. "
                "Refusing to build an archive that would fail at inflate."
            )

    segmap_weights_src = source_map.get("segmap_weights.tar.xz")
    if segmap_weights_src and segmap_weights_src.exists():
        head = segmap_weights_src.read_bytes()[:6]
        if head != b"\xfd7zXZ\x00":
            raise ValueError(
                f"segmap_weights.tar.xz has bad magic {head!r}; expected XZ magic. "
                "Refusing to build an archive that would fail at inflate."
            )

    segmap_payload_src = source_map.get("payload.bin")
    if segmap_payload_src and segmap_payload_src.exists():
        head = segmap_payload_src.read_bytes()[:4]
        if head != b"SHv1":
            raise ValueError(
                f"payload.bin has bad magic {head!r}; expected b'SHv1'. "
                "Refusing to build an archive that would fail at inflate."
            )

    class_targets_src = source_map.get("class_targets.fp16")
    if class_targets_src and class_targets_src.exists():
        size = class_targets_src.stat().st_size
        if size != 10:
            raise ValueError(
                f"class_targets.fp16 has {size} bytes; expected exactly 10 bytes "
                "for five fp16 class targets."
            )

    # FP4 without QAT warning: check renderer.bin header
    renderer_src = source_map.get("renderer.bin")
    if renderer_src and renderer_src.exists():
        header = renderer_src.read_bytes()[:4]
        is_fp4 = header == b"FP4A"
        renderer_size = renderer_src.stat().st_size
        if is_fp4:
            logger.warning(
                "FP4 renderer detected (%d bytes). Ensure QAT was used — "
                "FP4 without QAT introduces ~11.6 pixel mean error (39x pose signal).",
                renderer_size,
            )

    # Build the archive
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # When using Brotli, we store pre-compressed .br blobs with ZIP_STORED
    # (no double compression). Without Brotli, use ZIP_DEFLATED as before.
    zip_method = zipfile.ZIP_STORED if use_brotli else zipfile.ZIP_DEFLATED
    zip_kwargs: dict = {} if use_brotli else {"compresslevel": 9}

    if use_brotli:
        print(f"Brotli compression enabled (quality={brotli_quality})")

    with zipfile.ZipFile(output_path, "w", zip_method, **zip_kwargs) as zf:
        for name in required:
            src = source_map[name]
            assert src is not None  # guaranteed by check above

            if use_brotli:
                # Brotli-compress the artifact data, store as name.br
                raw_data = src.read_bytes()
                compressed_data = brotli_compress(raw_data, quality=brotli_quality)
                br_name = name + ".br"
                write_deterministic_zip_member(
                    zf,
                    br_name,
                    compressed_data,
                    compress_type=zipfile.ZIP_STORED,
                    compresslevel=None,
                )
                ratio = len(compressed_data) / len(raw_data) * 100
                logger.info(
                    "  Added %s (%d -> %d bytes, %.1f%%)",
                    br_name, len(raw_data), len(compressed_data), ratio,
                )
                print(
                    f"  Brotli: {name} {len(raw_data):,}B -> "
                    f"{len(compressed_data):,}B ({ratio:.1f}%) as {br_name}"
                )
            else:
                write_deterministic_zip_file(
                    zf,
                    src,
                    name,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
                logger.info("  Added %s (%d bytes)", name, src.stat().st_size)
        for name in sorted(typed_sidechannel_sources):
            src = typed_sidechannel_sources[name]
            write_deterministic_zip_file(
                zf,
                src,
                name,
                compress_type=zipfile.ZIP_STORED,
                compresslevel=None,
            )
            logger.info("  Added typed sidechannel %s (%d bytes)", name, src.stat().st_size)
        if typed_sidechannel_contract is not None:
            write_deterministic_zip_member(
                zf,
                TYPED_SIDECHANNEL_CONTRACT_MEMBER,
                _canonical_json_bytes(typed_sidechannel_contract),
                compress_type=zipfile.ZIP_STORED,
                compresslevel=None,
            )

    # Determine expected archive names for validation
    if use_brotli:
        brotli_manifest = _brotli_manifest(manifest)
        result = validate_archive(output_path, manifest=brotli_manifest, strict=True)
    else:
        result = validate_archive(output_path, manifest=manifest, strict=True)

    logger.info("Archive built: %s", result.summary())

    if validate and not result.valid:
        raise ValueError(
            f"Archive validation FAILED after build:\n{result.summary()}"
        )

    return result


def _brotli_manifest(manifest: ArchiveManifest) -> ArchiveManifest:
    """Create a Brotli-aware manifest that expects .br suffixed filenames.

    This is used internally for validation of Brotli-compressed archives.
    The manifest mirrors the original but maps each file to name.br.
    """
    # We cannot reuse ArchiveManifest directly because required_files()
    # returns hardcoded names. Instead, create a thin wrapper manifest
    # that validates .br names.
    return _BrotliManifest(manifest)


class _BrotliManifest(ArchiveManifest):
    """Manifest wrapper that expects .br suffixed filenames."""

    def __init__(self, inner: ArchiveManifest):
        # Copy all fields from inner
        for k, v in vars(inner).items():
            setattr(self, k, v)
        self._inner = inner

    def required_files(self) -> list[str]:
        return [name + ".br" for name in self._inner.required_files()]


def require_valid_archive(
    archive_path: Path | str,
    manifest: ArchiveManifest = RENDERER_SUBMISSION_MANIFEST,
    context: str = "auth eval",
) -> ArchiveValidationResult:
    """Validate an archive and raise if invalid.

    Use this at the START of any auth eval, deployment, or submission pipeline.

    Args:
        archive_path: Path to archive.zip
        manifest: Expected contents
        context: Human-readable context for error messages

    Returns:
        ArchiveValidationResult (only if valid)

    Raises:
        FileNotFoundError: if archive does not exist
        ValueError: if archive is invalid
    """
    result = validate_archive(archive_path, manifest=manifest, strict=True)

    if not result.valid:
        raise ValueError(
            f"ARCHIVE VALIDATION FAILED for {context}.\n"
            f"This means the {context} would produce an INVALID score.\n"
            f"\n{result.summary()}\n"
            f"\nFix the archive before running {context}."
        )

    # Always print provenance
    logger.info("[%s] Archive validated:\n%s", context, result.summary())
    print(f"\n[{context}] Archive: {result.archive_bytes:,} bytes, "
          f"rate={result.rate_term:.4f}, md5={result.md5[:12]}...")

    return result


def score_from_components(
    segnet_dist: float,
    posenet_dist: float,
    archive_bytes: int,
) -> dict[str, float]:
    """Compute contest score from components with full breakdown.

    Returns dict with 'total', 'segnet_term', 'posenet_term', 'rate_term',
    and all input values for provenance.
    """
    import math

    rate = 25 * archive_bytes / ORIGINAL_VIDEO_BYTES
    segnet_term = 100 * segnet_dist
    posenet_term = math.sqrt(10 * posenet_dist)

    return {
        "total": segnet_term + posenet_term + rate,
        "segnet_term": segnet_term,
        "posenet_term": posenet_term,
        "rate_term": rate,
        "segnet_dist": segnet_dist,
        "posenet_dist": posenet_dist,
        "archive_bytes": archive_bytes,
        "original_video_bytes": ORIGINAL_VIDEO_BYTES,
    }
