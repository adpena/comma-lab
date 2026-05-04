#!/usr/bin/env python3
"""Strict pre-submission compliance gate for contest release packets.

This script is intentionally provider-agnostic. It validates the exact files
that would be uploaded or published, records deterministic expectations, and
fails closed on archive/runtime/auth-eval custody gaps. It does not run a
scorer and does not make a new score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from tac.public_submission_refs import parse_public_pr_refs_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ORIGINAL_VIDEO_BYTES = 37_545_489
SCHEMA = "pre_submission_compliance_check_v1"
TOOL = "scripts/pre_submission_compliance_check.py"
DEFAULT_REQUIRED_FILES = ("archive.zip", "inflate.sh", "report.txt")
SECRET_SCAN_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PACKED_PAYLOAD_MEMBER_NAMES = ("p", "renderer_payload.bin", "renderer_payload.bin.br")
TERMINAL_DISPATCH_STATUS_PREFIXES = (
    "completed",
    "failed",
    "stopped",
    "refused_dispatch",
    "stale_superseded",
)


class ComplianceError(RuntimeError):
    """Raised for malformed inputs before a report can be built."""


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    details: str
    severity: str = "error"

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "severity": self.severity,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _check(checks: list[Check], name: str, passed: bool, details: str, *, severity: str = "error") -> None:
    checks.append(Check(name=name, passed=bool(passed), details=details, severity=severity))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComplianceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComplianceError(f"{path} must contain a JSON object")
    return payload


def _score(seg_dist: float, pose_dist: float, archive_bytes: int) -> float:
    return 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * archive_bytes / ORIGINAL_VIDEO_BYTES


def _unsafe_zip_name(name: str) -> str | None:
    if not name:
        return "empty_member_name"
    if "\\" in name:
        return "backslash_member_name"
    if "\x00" in name or any(ord(ch) < 32 for ch in name):
        return "control_character_member_name"
    if re.match(r"^[A-Za-z]:", name):
        return "windows_drive_member_name"
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        return "zip_slip_member_name"
    if "__MACOSX" in member.parts:
        return "macosx_resource_directory"
    base = member.name
    if base in {".DS_Store", "Thumbs.db"} or any(part.startswith("._") for part in member.parts):
        return "resource_fork_or_hidden_sidecar"
    if any(part.startswith(".") for part in member.parts):
        return "hidden_sidecar_member_name"
    return None


def _decode_zip_name(raw: bytes, flag_bits: int) -> str | None:
    encoding = "utf-8" if (flag_bits & 0x800) else "cp437"
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        return None


def _local_header_name(path: Path, info: zipfile.ZipInfo) -> tuple[str | None, int | None]:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return None, None
        local_flag_bits = struct.unpack_from("<H", header, 6)[0]
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        _ = handle.read(extra_len)
    return _decode_zip_name(raw_name, local_flag_bits), local_flag_bits


def inspect_archive(path: Path, *, expect_single_member: str | None = None) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    record: dict[str, Any] = {
        "path": _rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
        "members": [],
    }
    _check(checks, "archive_exists", path.is_file(), f"archive={_rel(path)}")
    if not path.is_file():
        return record, checks

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad_crc = zf.testzip()
            infos = zf.infolist()
            names = [info.filename for info in infos]
            duplicate_names = sorted({name for name in names if names.count(name) > 1})
            packed_payload_members = [name for name in names if name in PACKED_PAYLOAD_MEMBER_NAMES]
            _check(checks, "archive_is_valid_zip", True, "zipfile opened and CRC scan completed")
            _check(checks, "archive_crc_ok", bad_crc is None, f"bad_crc_member={bad_crc!r}")
            _check(checks, "archive_no_duplicate_members", not duplicate_names, f"duplicates={duplicate_names}")
            _check(
                checks,
                "archive_packed_payload_singleton",
                len(packed_payload_members) <= 1,
                f"packed_payload_members={packed_payload_members}",
            )
            if expect_single_member:
                _check(
                    checks,
                    "archive_expected_single_member",
                    names == [expect_single_member],
                    f"expected={[expect_single_member]} observed={names}",
                )
            for info in infos:
                local_name, local_flag_bits = _local_header_name(path, info)
                unsafe = _unsafe_zip_name(info.filename)
                member_record = {
                    "name": info.filename,
                    "local_header_name": local_name,
                    "is_dir": info.is_dir(),
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "date_time": list(info.date_time),
                    "external_attr": int(info.external_attr),
                    "flag_bits": int(info.flag_bits),
                    "local_header_flag_bits": local_flag_bits,
                }
                if not info.is_dir():
                    with zf.open(info, "r") as handle:
                        member_record["sha256"] = hashlib.sha256(handle.read()).hexdigest()
                record["members"].append(member_record)
                _check(
                    checks,
                    f"archive_member_safe:{info.filename}",
                    unsafe is None,
                    unsafe or "safe member name",
                )
                _check(
                    checks,
                    f"archive_local_central_name_match:{info.filename}",
                    local_name == info.filename,
                    f"local={local_name!r} central={info.filename!r}",
                )
                _check(
                    checks,
                    f"archive_local_central_flag_bits_match:{info.filename}",
                    local_flag_bits == info.flag_bits,
                    f"local={local_flag_bits!r} central={info.flag_bits!r}",
                )
    except zipfile.BadZipFile as exc:
        _check(checks, "archive_is_valid_zip", False, str(exc))
    return record, checks


def _json_sha_candidates(payload: dict[str, Any]) -> dict[str, object]:
    archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
    frontier = payload.get("frontier_summary") if isinstance(payload.get("frontier_summary"), dict) else {}
    return {
        "archive_sha256": payload.get("archive_sha256"),
        "sha256": payload.get("sha256"),
        "archive.sha256": archive.get("sha256"),
        "archive.archive_sha256": archive.get("archive_sha256"),
        "frontier_summary.archive_sha256": frontier.get("archive_sha256"),
    }


def _json_size_candidates(payload: dict[str, Any]) -> dict[str, object]:
    archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
    frontier = payload.get("frontier_summary") if isinstance(payload.get("frontier_summary"), dict) else {}
    return {
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "size_bytes": payload.get("size_bytes"),
        "bytes": payload.get("bytes"),
        "archive.archive_size_bytes": archive.get("archive_size_bytes"),
        "archive.size_bytes": archive.get("size_bytes"),
        "archive.bytes": archive.get("bytes"),
        "frontier_summary.archive_size_bytes": frontier.get("archive_size_bytes"),
    }


def _first_present(mapping: dict[str, object]) -> object | None:
    for value in mapping.values():
        if value is not None:
            return value
    return None


def _manifest_member_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("members")
    if not isinstance(rows, list):
        archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
        rows = archive.get("members")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def inspect_archive_manifest(path: Path | None, archive: dict[str, Any]) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    record: dict[str, Any] = {"path": _rel(path) if path else None, "present": bool(path and path.is_file())}
    if path is None:
        _check(checks, "archive_manifest_present", False, "no archive manifest provided")
        return record, checks
    _check(checks, "archive_manifest_present", path.is_file(), _rel(path))
    if not path.is_file():
        return record, checks

    payload = _read_json(path)
    sha_candidates = _json_sha_candidates(payload)
    size_candidates = _json_size_candidates(payload)
    manifest_sha = _first_present(sha_candidates)
    manifest_size = _first_present(size_candidates)
    record.update(
        {
            "sha256": manifest_sha,
            "size_bytes": manifest_size,
            "sha256_candidates": {k: v for k, v in sha_candidates.items() if v is not None},
            "size_candidates": {k: v for k, v in size_candidates.items() if v is not None},
            "member_count": 0,
        }
    )
    present_shas = [value for value in sha_candidates.values() if value is not None]
    present_sizes = [value for value in size_candidates.values() if value is not None]
    _check(
        checks,
        "archive_manifest_sha256_present",
        manifest_sha is not None,
        f"sha256_candidates={record['sha256_candidates']}",
    )
    _check(
        checks,
        "archive_manifest_size_bytes_present",
        manifest_size is not None,
        f"size_candidates={record['size_candidates']}",
    )
    _check(
        checks,
        "archive_manifest_sha256_fields_consistent",
        len(set(present_shas)) <= 1,
        f"sha256_candidates={record['sha256_candidates']}",
    )
    _check(
        checks,
        "archive_manifest_size_bytes_fields_consistent",
        len(set(present_sizes)) <= 1,
        f"size_candidates={record['size_candidates']}",
    )
    _check(
        checks,
        "archive_manifest_sha256_format",
        _looks_like_sha256(manifest_sha),
        f"manifest_sha256={manifest_sha!r}",
    )
    _check(
        checks,
        "archive_manifest_size_bytes_integer",
        _is_nonnegative_int(manifest_size),
        f"manifest_size_bytes={manifest_size!r}",
    )
    _check(
        checks,
        "archive_manifest_sha256_matches_archive",
        manifest_sha == archive.get("sha256"),
        f"manifest={manifest_sha} archive={archive.get('sha256')}",
    )
    _check(
        checks,
        "archive_manifest_size_bytes_matches_archive",
        manifest_size == archive.get("bytes"),
        f"manifest={manifest_size} archive={archive.get('bytes')}",
    )

    rows = _manifest_member_rows(payload)
    record["member_count"] = len(rows)
    if rows:
        archive_members = {
            row["name"]: row
            for row in archive.get("members", [])
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        manifest_names = [row.get("name") for row in rows]
        duplicate_manifest_names = sorted(
            {name for name in manifest_names if isinstance(name, str) and manifest_names.count(name) > 1}
        )
        missing_from_archive = sorted(
            name for name in manifest_names if isinstance(name, str) and name not in archive_members
        )
        extra_archive_members = sorted(set(archive_members) - {name for name in manifest_names if isinstance(name, str)})
        _check(
            checks,
            "archive_manifest_members_unique",
            not duplicate_manifest_names,
            f"duplicates={duplicate_manifest_names}",
        )
        _check(
            checks,
            "archive_manifest_members_match_archive",
            not missing_from_archive and not extra_archive_members,
            f"missing_from_archive={missing_from_archive} extra_archive_members={extra_archive_members}",
        )
        for row in rows:
            name = row.get("name")
            check_suffix = str(name)
            if not isinstance(name, str) or not name:
                _check(checks, f"archive_manifest_member_name_valid:{check_suffix}", False, f"name={name!r}")
                continue
            observed = archive_members.get(name)
            if observed is None:
                continue
            manifest_bytes = row.get("file_size", row.get("bytes", row.get("size_bytes")))
            manifest_sha = row.get("sha256")
            if manifest_bytes is not None:
                _check(
                    checks,
                    f"archive_manifest_member_bytes_match:{name}",
                    manifest_bytes == observed.get("file_size"),
                    f"manifest={manifest_bytes} archive={observed.get('file_size')}",
                )
            if manifest_sha is not None:
                _check(
                    checks,
                    f"archive_manifest_member_sha256_match:{name}",
                    manifest_sha == observed.get("sha256"),
                    f"manifest={manifest_sha} archive={observed.get('sha256')}",
                )
    return record, checks


def inspect_submission_dir(submission_dir: Path, required_files: Iterable[str]) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    files: dict[str, Any] = {}
    for rel in required_files:
        path = submission_dir / rel
        exists = path.is_file()
        files[rel] = {
            "path": _rel(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
            "sha256": _sha256(path) if exists else None,
        }
        _check(checks, f"required_file_present:{rel}", exists, _rel(path))
    inflate = submission_dir / "inflate.sh"
    if inflate.is_file():
        mode = inflate.stat().st_mode
        executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        _check(checks, "inflate_sh_executable", executable, f"mode={oct(stat.S_IMODE(mode))}")
    return {"path": _rel(submission_dir), "required_files": files}, checks


def _runtime_tree_candidates_from_auth(payload: dict[str, Any]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        runtime = provenance.get("inflate_runtime_manifest")
        if isinstance(runtime, dict) and isinstance(runtime.get("runtime_tree_sha256"), str):
            candidates["provenance.inflate_runtime_manifest.runtime_tree_sha256"] = runtime[
                "runtime_tree_sha256"
            ]
        value = provenance.get("runtime_tree_sha256")
        if isinstance(value, str):
            candidates["provenance.runtime_tree_sha256"] = value
    runtime = payload.get("inflate_runtime_manifest")
    if isinstance(runtime, dict) and isinstance(runtime.get("runtime_tree_sha256"), str):
        candidates["inflate_runtime_manifest.runtime_tree_sha256"] = runtime["runtime_tree_sha256"]
    return candidates


def _runtime_file_manifest_candidates_from_auth(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        runtime = provenance.get("inflate_runtime_manifest")
        if isinstance(runtime, dict) and isinstance(runtime.get("files"), list):
            candidates["provenance.inflate_runtime_manifest.files"] = [
                row for row in runtime["files"] if isinstance(row, dict)
            ]
    runtime = payload.get("inflate_runtime_manifest")
    if isinstance(runtime, dict) and isinstance(runtime.get("files"), list):
        candidates["inflate_runtime_manifest.files"] = [
            row for row in runtime["files"] if isinstance(row, dict)
        ]
    return candidates


def _runtime_tree_from_auth(payload: dict[str, Any]) -> str | None:
    candidates = _runtime_tree_candidates_from_auth(payload)
    for key in (
        "provenance.inflate_runtime_manifest.runtime_tree_sha256",
        "provenance.runtime_tree_sha256",
        "inflate_runtime_manifest.runtime_tree_sha256",
    ):
        if key in candidates:
            return candidates[key]
    return None


def _runtime_files_from_auth(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = _runtime_file_manifest_candidates_from_auth(payload)
    for key in ("provenance.inflate_runtime_manifest.files", "inflate_runtime_manifest.files"):
        if key in candidates:
            return candidates[key]
    return []


def _runtime_file_identity(row: dict[str, Any]) -> tuple[object, object, object]:
    return row.get("relative_path"), row.get("bytes"), row.get("sha256")


def _runtime_file_manifest_values_consistent(
    candidates: dict[str, list[dict[str, Any]]]
) -> bool:
    normalized = {
        tuple(sorted(_runtime_file_identity(row) for row in rows))
        for rows in candidates.values()
    }
    return len(normalized) <= 1


def _looks_like_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _present_consistent(left: object, right: object) -> bool:
    return left is None or right is None or left == right


def _is_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def inspect_auth_eval(
    path: Path | None,
    *,
    archive: dict[str, Any],
    expected_samples: int | None,
    require_t4_equivalent: bool,
    expected_runtime_tree_sha256: str | None,
    require_submission_runtime_match: bool,
    submission_dir: Path,
) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    record: dict[str, Any] = {"path": _rel(path) if path else None, "present": bool(path and path.is_file())}
    if path is None:
        _check(checks, "auth_eval_json_present", False, "no --auth-eval-json provided")
        return record, checks
    _check(checks, "auth_eval_json_present", path.is_file(), _rel(path))
    if not path.is_file():
        return record, checks
    payload = _read_json(path)
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    payload_archive_sha = payload.get("archive_sha256")
    provenance_archive_sha = provenance.get("archive_sha256")
    payload_archive_bytes = payload.get("archive_size_bytes")
    provenance_archive_bytes = provenance.get("archive_size_bytes")
    archive_sha = provenance_archive_sha if provenance_archive_sha is not None else payload_archive_sha
    archive_bytes = provenance_archive_bytes if provenance_archive_bytes is not None else payload_archive_bytes
    n_samples = payload.get("n_samples")
    payload_device = payload.get("device")
    provenance_device = provenance.get("device")
    device = provenance_device if provenance_device is not None else payload_device
    cuda_available = provenance.get("cuda_available")
    gpu_t4_match = provenance.get("gpu_t4_match")
    seg = payload.get("avg_segnet_dist")
    pose = payload.get("avg_posenet_dist")
    recomputed = payload.get("score_recomputed_from_components")
    runtime_tree_candidates = _runtime_tree_candidates_from_auth(payload)
    runtime_tree = _runtime_tree_from_auth(payload)
    runtime_file_candidates = _runtime_file_manifest_candidates_from_auth(payload)
    runtime_files = _runtime_files_from_auth(payload)
    record.update(
        {
            "archive_sha256": archive_sha,
            "archive_size_bytes": archive_bytes,
            "payload_archive_sha256": payload_archive_sha,
            "provenance_archive_sha256": provenance_archive_sha,
            "payload_archive_size_bytes": payload_archive_bytes,
            "provenance_archive_size_bytes": provenance_archive_bytes,
            "n_samples": n_samples,
            "device": device,
            "payload_device": payload_device,
            "provenance_device": provenance_device,
            "cuda_available": cuda_available,
            "gpu_t4_match": gpu_t4_match,
            "runtime_tree_sha256": runtime_tree,
            "runtime_tree_candidates": runtime_tree_candidates,
            "runtime_file_manifest_candidates": {
                key: len(rows) for key, rows in runtime_file_candidates.items()
            },
            "runtime_files": runtime_files,
            "score_recomputed_from_components": recomputed,
        }
    )
    _check(
        checks,
        "auth_eval_archive_sha256_fields_consistent",
        _present_consistent(payload_archive_sha, provenance_archive_sha),
        f"payload={payload_archive_sha!r} provenance={provenance_archive_sha!r}",
    )
    _check(
        checks,
        "auth_eval_archive_size_bytes_fields_consistent",
        _present_consistent(payload_archive_bytes, provenance_archive_bytes),
        f"payload={payload_archive_bytes!r} provenance={provenance_archive_bytes!r}",
    )
    _check(
        checks,
        "auth_eval_archive_sha256_format",
        _looks_like_sha256(archive_sha),
        f"archive_sha256={archive_sha!r}",
    )
    _check(
        checks,
        "auth_eval_archive_size_bytes_integer",
        _is_nonnegative_int(archive_bytes),
        f"archive_size_bytes={archive_bytes!r}",
    )
    _check(
        checks,
        "auth_eval_device_fields_consistent",
        _present_consistent(payload_device, provenance_device),
        f"payload={payload_device!r} provenance={provenance_device!r}",
    )
    _check(checks, "auth_eval_archive_sha_matches", archive_sha == archive.get("sha256"), f"auth={archive_sha} archive={archive.get('sha256')}")
    _check(checks, "auth_eval_archive_bytes_match", archive_bytes == archive.get("bytes"), f"auth={archive_bytes} archive={archive.get('bytes')}")
    _check(checks, "auth_eval_cuda_device", str(device).lower() == "cuda", f"device={device!r}")
    _check(checks, "auth_eval_cuda_available", cuda_available is True, f"cuda_available={cuda_available!r}")
    _check(
        checks,
        "auth_eval_runtime_tree_recorded",
        _looks_like_sha256(runtime_tree),
        f"runtime_tree_sha256={runtime_tree!r}",
    )
    _check(
        checks,
        "auth_eval_runtime_tree_fields_consistent",
        len(set(runtime_tree_candidates.values())) <= 1,
        f"runtime_tree_candidates={runtime_tree_candidates}",
    )
    _check(
        checks,
        "auth_eval_runtime_file_manifest_fields_consistent",
        _runtime_file_manifest_values_consistent(runtime_file_candidates),
        f"runtime_file_manifest_candidates={list(runtime_file_candidates)}",
    )
    if expected_samples is not None:
        _check(checks, "auth_eval_expected_samples", n_samples == expected_samples, f"expected={expected_samples} observed={n_samples}")
    if require_t4_equivalent:
        _check(checks, "auth_eval_t4_equivalent", gpu_t4_match is True, f"gpu_t4_match={gpu_t4_match!r}")
    if expected_runtime_tree_sha256:
        _check(
            checks,
            "auth_eval_runtime_tree_expected",
            runtime_tree == expected_runtime_tree_sha256,
            f"expected={expected_runtime_tree_sha256} observed={runtime_tree}",
        )
    if require_submission_runtime_match:
        _check(
            checks,
            "auth_eval_runtime_file_manifest_present",
            bool(runtime_files),
            f"runtime_file_count={len(runtime_files)}",
        )
        rels = [row.get("relative_path") for row in runtime_files]
        duplicate_rels = sorted({rel for rel in rels if isinstance(rel, str) and rels.count(rel) > 1})
        _check(
            checks,
            "auth_eval_runtime_file_manifest_paths_unique",
            not duplicate_rels,
            f"duplicates={duplicate_rels}",
        )
        for row in runtime_files:
            rel = row.get("relative_path")
            expected_sha = row.get("sha256")
            expected_bytes = row.get("bytes")
            check_name = f"auth_eval_runtime_file_matches_submission:{rel}"
            if not isinstance(rel, str) or not rel:
                _check(checks, check_name, False, f"bad relative_path={rel!r}")
                continue
            if not _looks_like_sha256(expected_sha):
                _check(checks, check_name, False, f"bad expected sha256={expected_sha!r}")
                continue
            if not _is_nonnegative_int(expected_bytes):
                _check(checks, check_name, False, f"bad expected bytes={expected_bytes!r}")
                continue
            unsafe = _unsafe_zip_name(rel)
            candidate = (submission_dir / rel).resolve()
            if unsafe is not None:
                _check(checks, check_name, False, f"unsafe runtime path: {unsafe}")
                continue
            try:
                candidate.relative_to(submission_dir.resolve())
            except ValueError:
                _check(checks, check_name, False, f"runtime file escaped submission dir: {rel}")
                continue
            if not candidate.is_file():
                _check(checks, check_name, False, f"missing submission runtime file: {_rel(candidate)}")
                continue
            observed_sha = _sha256(candidate)
            observed_bytes = candidate.stat().st_size
            _check(
                checks,
                check_name,
                observed_sha == expected_sha and observed_bytes == expected_bytes,
                (
                    f"expected_sha={expected_sha} observed_sha={observed_sha} "
                    f"expected_bytes={expected_bytes} observed_bytes={observed_bytes}"
                ),
            )
    if all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (seg, pose, archive.get("bytes"), recomputed)):
        contribution_keys = (
            "score_seg_contribution",
            "score_pose_contribution",
            "score_rate_contribution",
        )
        if all(isinstance(payload.get(key), (int, float)) for key in contribution_keys):
            calc = sum(float(payload[key]) for key in contribution_keys)
            tolerance = 1e-9
            contribution_tolerance = 1e-6
            basis = "json_contribution_fields"
            expected_contributions = {
                "score_seg_contribution": 100.0 * float(seg),
                "score_pose_contribution": math.sqrt(10.0 * float(pose)),
                "score_rate_contribution": 25.0 * int(archive["bytes"]) / ORIGINAL_VIDEO_BYTES,
            }
            contribution_deltas = {
                key: abs(float(payload[key]) - expected)
                for key, expected in expected_contributions.items()
            }
            _check(
                checks,
                "auth_eval_score_contributions_match_components",
                all(delta < contribution_tolerance for delta in contribution_deltas.values()),
                f"deltas={contribution_deltas} tolerance={contribution_tolerance}",
            )
        else:
            calc = _score(float(seg), float(pose), int(archive["bytes"]))
            tolerance = 1e-6
            basis = "display_component_formula"
        _check(
            checks,
            "auth_eval_score_recomputes",
            abs(calc - float(recomputed)) < tolerance,
            f"basis={basis} computed={calc} json={recomputed} tolerance={tolerance}",
        )
        record["score_recomputed_independently"] = calc
    else:
        _check(checks, "auth_eval_score_recomputes", False, "missing finite seg/pose/bytes/recomputed score")
    return record, checks


def _iter_scan_files(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in SECRET_SCAN_SUFFIXES:
            out.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in SECRET_SCAN_SUFFIXES:
                    out.append(child)
    return sorted(dict.fromkeys(out))


def run_public_hygiene(paths: list[Path]) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    files = _iter_scan_files(paths)
    record = {"scanned_files": [_rel(path) for path in files], "violations": []}
    if not files:
        _check(checks, "public_hygiene_scan_nonempty", False, "no text files found to scan", severity="warning")
        return record, checks
    try:
        from tac.preflight import check_public_release_hygiene

        violations = check_public_release_hygiene(
            repo_root=REPO_ROOT,
            strict=False,
            verbose=False,
            scan_paths=files,
        )
    except Exception as exc:
        violations = [f"public hygiene check raised {type(exc).__name__}: {exc}"]
    record["violations"] = violations
    _check(checks, "public_release_hygiene", not violations, f"violations={len(violations)}")
    return record, checks


def inspect_report_linkage(
    report_path: Path,
    *,
    archive: dict[str, Any],
    auth_eval: dict[str, Any] | None,
    expected_lane_id: str | None,
    expected_job_id: str | None,
    require_archive_link: bool,
    require_auth_link: bool,
) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    record: dict[str, Any] = {"path": _rel(report_path), "present": report_path.is_file()}
    _check(checks, "report_txt_present", report_path.is_file(), _rel(report_path))
    if not report_path.is_file():
        return record, checks
    text = report_path.read_text(encoding="utf-8", errors="replace")
    archive_sha = archive.get("sha256")
    archive_bytes = archive.get("bytes")
    contains_archive_sha = isinstance(archive_sha, str) and archive_sha in text
    contains_archive_bytes = archive_bytes is not None and str(archive_bytes) in text
    record.update(
        {
            "contains_archive_sha256": contains_archive_sha,
            "contains_archive_size_bytes": contains_archive_bytes,
            "contains_expected_lane_id": bool(expected_lane_id and expected_lane_id in text),
            "contains_expected_job_id": bool(expected_job_id and expected_job_id in text),
        }
    )
    if require_archive_link:
        _check(
            checks,
            "report_links_exact_archive_sha256",
            contains_archive_sha,
            "report.txt must include the exact archive SHA-256",
        )
        _check(
            checks,
            "report_links_exact_archive_size_bytes",
            contains_archive_bytes,
            "report.txt must include the exact archive byte size",
        )
    if require_auth_link and auth_eval is not None:
        score = auth_eval.get("score_recomputed_from_components")
        contains_score = score is not None and str(score) in text
        record["contains_auth_eval_score"] = contains_score
        _check(
            checks,
            "report_links_auth_eval_score_field",
            contains_score,
            "report.txt must include the auth-eval recomputed score field",
        )
    if expected_lane_id:
        _check(
            checks,
            "report_links_expected_lane_id",
            expected_lane_id in text,
            f"expected_lane_id={expected_lane_id}",
        )
    if expected_job_id:
        _check(
            checks,
            "report_links_expected_job_id",
            expected_job_id in text,
            f"expected_job_id={expected_job_id}",
        )
    return record, checks


def inspect_dispatch_claim_linkage(
    claims_path: Path | None,
    *,
    expected_lane_id: str | None,
    expected_job_id: str | None,
) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    record: dict[str, Any] = {
        "path": _rel(claims_path) if claims_path else None,
        "present": bool(claims_path and claims_path.is_file()),
        "expected_lane_id": expected_lane_id,
        "expected_job_id": expected_job_id,
    }
    if not claims_path and not (expected_lane_id or expected_job_id):
        return record, checks
    _check(checks, "dispatch_claims_file_present", bool(claims_path and claims_path.is_file()), _rel(claims_path) if claims_path else "none")
    if not claims_path or not claims_path.is_file():
        return record, checks
    text = claims_path.read_text(encoding="utf-8", errors="replace")
    matching_lines = [
        line
        for line in text.splitlines()
        if (not expected_lane_id or expected_lane_id in line)
        and (not expected_job_id or expected_job_id in line)
    ]
    terminal_lines = [
        line
        for line in matching_lines
        if any(f"| {prefix}" in line or f"|{prefix}" in line for prefix in TERMINAL_DISPATCH_STATUS_PREFIXES)
    ]
    record["matching_lines"] = matching_lines
    record["terminal_lines"] = terminal_lines
    _check(
        checks,
        "dispatch_claim_matching_row_present",
        bool(matching_lines),
        f"matches={len(matching_lines)}",
    )
    _check(
        checks,
        "dispatch_claim_terminal_row_present",
        bool(terminal_lines),
        f"terminal_matches={len(terminal_lines)}",
    )
    return record, checks


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    submission_dir = args.submission_dir.resolve()
    archive_path = (args.archive or (submission_dir / "archive.zip")).resolve()
    if args.contest_final:
        args.require_auth_eval = True
        args.require_t4_equivalent = True
        args.expect_single_member = args.expect_single_member or "x"
        args.require_archive_manifest = True
        args.require_report_archive_link = True
        args.require_submission_runtime_match = True
        args.archive_manifest_json = args.archive_manifest_json or (submission_dir / "archive_manifest.json")
        if not args.public_scan_path:
            args.public_scan_path = [submission_dir]
    required_files = tuple(args.required_file or DEFAULT_REQUIRED_FILES)
    all_checks: list[Check] = []

    submission, checks = inspect_submission_dir(submission_dir, required_files)
    all_checks.extend(checks)
    archive, checks = inspect_archive(archive_path, expect_single_member=args.expect_single_member)
    all_checks.extend(checks)
    archive_manifest = None
    if args.archive_manifest_json or args.require_archive_manifest:
        archive_manifest, checks = inspect_archive_manifest(args.archive_manifest_json, archive)
        all_checks.extend(checks)
    submission_archive = submission["required_files"].get("archive.zip")
    if submission_archive and submission_archive.get("exists") and archive.get("exists"):
        _check(
            all_checks,
            "submission_archive_matches_inspected_archive",
            submission_archive.get("sha256") == archive.get("sha256")
            and submission_archive.get("bytes") == archive.get("bytes"),
            (
                f"submission_sha={submission_archive.get('sha256')} "
                f"submission_bytes={submission_archive.get('bytes')} "
                f"inspected_sha={archive.get('sha256')} "
                f"inspected_bytes={archive.get('bytes')}"
            ),
        )
    if args.expected_archive_sha256:
        _check(
            all_checks,
            "expected_archive_sha256_format",
            _looks_like_sha256(args.expected_archive_sha256),
            f"expected_archive_sha256={args.expected_archive_sha256!r}",
        )
        _check(
            all_checks,
            "expected_archive_sha256",
            archive.get("sha256") == args.expected_archive_sha256,
            f"expected={args.expected_archive_sha256} observed={archive.get('sha256')}",
        )
    elif args.contest_final:
        _check(
            all_checks,
            "contest_final_expected_archive_sha256_present",
            False,
            "--contest-final requires --expected-archive-sha256",
        )
    if args.expected_archive_size_bytes is not None:
        _check(
            all_checks,
            "expected_archive_size_bytes",
            archive.get("bytes") == args.expected_archive_size_bytes,
            f"expected={args.expected_archive_size_bytes} observed={archive.get('bytes')}",
        )
    elif args.contest_final:
        _check(
            all_checks,
            "contest_final_expected_archive_size_bytes_present",
            False,
            "--contest-final requires --expected-archive-size-bytes",
        )
    if args.expected_runtime_tree_sha256:
        _check(
            all_checks,
            "expected_runtime_tree_sha256_format",
            _looks_like_sha256(args.expected_runtime_tree_sha256),
            f"expected_runtime_tree_sha256={args.expected_runtime_tree_sha256!r}",
        )

    auth_eval = None
    if args.auth_eval_json or args.require_auth_eval:
        auth_eval, checks = inspect_auth_eval(
            args.auth_eval_json,
            archive=archive,
            expected_samples=args.expected_samples,
            require_t4_equivalent=args.require_t4_equivalent,
            expected_runtime_tree_sha256=args.expected_runtime_tree_sha256,
            require_submission_runtime_match=args.contest_final or args.require_submission_runtime_match,
            submission_dir=submission_dir,
        )
        all_checks.extend(checks)

    public_hygiene = None
    if args.public_scan_path:
        public_hygiene, checks = run_public_hygiene([path.resolve() for path in args.public_scan_path])
        all_checks.extend(checks)

    report_linkage, checks = inspect_report_linkage(
        submission_dir / "report.txt",
        archive=archive,
        auth_eval=auth_eval,
        expected_lane_id=args.expected_lane_id,
        expected_job_id=args.expected_job_id,
        require_archive_link=args.require_report_archive_link,
        require_auth_link=args.require_report_auth_score_link,
    )
    all_checks.extend(checks)
    dispatch_claim_linkage, checks = inspect_dispatch_claim_linkage(
        args.dispatch_claims_md,
        expected_lane_id=args.expected_lane_id,
        expected_job_id=args.expected_job_id,
    )
    all_checks.extend(checks)

    failed = [check for check in all_checks if not check.passed and check.severity == "error"]
    warnings = [check for check in all_checks if not check.passed and check.severity == "warning"]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "status": "passed" if not failed else "failed",
        "score_claim": False,
        "provider_agnostic": True,
        "submission": submission,
        "archive": archive,
        "archive_manifest": archive_manifest,
        "auth_eval": auth_eval,
        "public_hygiene": public_hygiene,
        "report_linkage": report_linkage,
        "dispatch_claim_linkage": dispatch_claim_linkage,
        "checks": [check.as_json() for check in all_checks],
        "failed_checks": [check.name for check in failed],
        "warning_checks": [check.name for check in warnings],
        "expectations": {
            "required_files": list(required_files),
            "expected_archive_sha256": args.expected_archive_sha256,
            "expected_archive_size_bytes": args.expected_archive_size_bytes,
            "expected_samples": args.expected_samples,
            "expected_runtime_tree_sha256": args.expected_runtime_tree_sha256,
            "require_auth_eval": args.require_auth_eval,
            "require_t4_equivalent": args.require_t4_equivalent,
            "require_submission_runtime_match": args.contest_final
            or args.require_submission_runtime_match,
            "expect_single_member": args.expect_single_member,
            "require_archive_manifest": args.require_archive_manifest,
            "archive_manifest_json": _rel(args.archive_manifest_json) if args.archive_manifest_json else None,
            "require_report_archive_link": args.require_report_archive_link,
            "require_report_auth_score_link": args.require_report_auth_score_link,
            "dispatch_claims_md": _rel(args.dispatch_claims_md) if args.dispatch_claims_md else None,
            "expected_lane_id": args.expected_lane_id,
            "expected_job_id": args.expected_job_id,
            "contest_final": args.contest_final,
        },
        "public_submission_refs": parse_public_pr_refs_csv(args.source_prs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--auth-eval-json", type=Path, default=None)
    parser.add_argument("--require-auth-eval", action="store_true")
    parser.add_argument("--require-t4-equivalent", action="store_true")
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument("--expected-archive-size-bytes", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=600)
    parser.add_argument("--expected-runtime-tree-sha256", default=None)
    parser.add_argument(
        "--require-submission-runtime-match",
        action="store_true",
        help="Require auth-eval runtime manifest files to match files in --submission-dir.",
    )
    parser.add_argument("--expect-single-member", default=None)
    parser.add_argument("--archive-manifest-json", type=Path, default=None)
    parser.add_argument("--require-archive-manifest", action="store_true")
    parser.add_argument("--require-report-archive-link", action="store_true")
    parser.add_argument("--require-report-auth-score-link", action="store_true")
    parser.add_argument("--dispatch-claims-md", type=Path, default=None)
    parser.add_argument("--expected-lane-id", default=None)
    parser.add_argument("--expected-job-id", default=None)
    parser.add_argument(
        "--contest-final",
        action="store_true",
        help=(
            "Enable final-submission strict mode: require auth eval, "
            "T4/equivalent custody, single member x, public hygiene scan, "
            "archive manifest, report archive linkage, submission-runtime "
            "match, and explicit expected archive SHA/bytes."
        ),
    )
    parser.add_argument("--required-file", action="append", default=None)
    parser.add_argument("--public-scan-path", action="append", type=Path, default=[])
    parser.add_argument(
        "--source-prs",
        default=None,
        help="Comma-separated public PR refs used as provenance signal, e.g. PR85,PR91.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
