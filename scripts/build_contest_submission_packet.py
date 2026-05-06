#!/usr/bin/env python3
# pyc-recovery pass2: rehydrated from git blob 3791a1ee111cb8c8b0d66ccb67b4d01632470be6 via `git fsck --lost-found`
# original path: scripts/build_contest_submission_packet.py
# This is OUR source, dropped during commit 66c59aae filter-repo cleanup; the .pyc was the only
# orphan left behind. Original blob SHA verified intact.
# Recovered: 2026-05-05 by Sherlock pass2
"""Build a deterministic contest-faithful submission packet manifest.

By default the packet is metadata-only: it records custody facts for an exact
eval artifact directory and writes a JSON manifest plus markdown checklist. When
``--runtime-dir`` is provided it also builds the concrete submission directory
from the auth-eval runtime manifest plus ``archive.zip`` and ``report.txt``.
Optional planner, visualization, and next-action files are recorded as non-score
supporting artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from tac.repo_io import json_text, read_json, sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_DENOMINATOR = 37_545_489
DEFAULT_MANIFEST_NAME = "submission_packet_manifest.json"
DEFAULT_CHECKLIST_NAME = "submission_packet_checklist.md"
DEFAULT_SUBMISSION_DIR_NAME = "submission"
KNOWN_OPTIONAL_ARTIFACTS = (
    "component_trace.json",
    "report.txt",
    "eval_provenance.json",
    "auth_eval.log",
    "contest_auth_eval.adjudicated.json",
    "adjudication_provenance.json",
)
CDO1_ARCHIVE_MEMBERS = (
    "masks.cdo1",
    "masks.cdo1.zlib",
    "masks.cdo1.xz",
    "masks.cdo1.br",
)
AMR1_ARCHIVE_MEMBERS = (
    "alpha4_residual_repair.amr1",
    "alpha4_residual_repair.amr1.xz",
    "alpha4_residual_repair.amr1.zlib",
    "alpha4_residual_repair.amr1.br",
)
PACKED_PAYLOAD_MEMBER_NAMES = ("renderer_payload.bin", "renderer_payload.bin.br", "p")
RPK1_MAGIC = b"RPK1"
RPK1_HEADER_STRUCT = struct.Struct("<I")
ZIP_LOCAL_HEADER_STRUCT = struct.Struct("<4sHHHHHIIIHH")
SUPPORTING_ARTIFACT_SECTIONS = {
    "planner_ledgers": "planning_or_proxy_only",
    "visualizations": "visual_audit_only",
    "next_action_tranches": "roadmap_only",
}


class PacketError(RuntimeError):
    """Raised when the source artifact directory is not packet-ready."""


_sha256 = sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise PacketError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PacketError(f"{path.name} must contain a JSON object")
    return payload


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_record(path: Path, repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, repo_root),
        "artifact_relative_path": _rel(path, artifact_dir),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _copied_file_record(path: Path, repo_root: Path, submission_dir: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, repo_root),
        "submission_relative_path": _rel(path, submission_dir),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "mode_octal": oct(path.stat().st_mode & 0o777),
    }


def _artifact_file(artifact_dir: Path, artifact_relative_path: str) -> Path:
    path = (artifact_dir / artifact_relative_path).resolve()
    try:
        path.relative_to(artifact_dir)
    except ValueError as exc:
        raise PacketError(
            f"artifact file must stay inside artifact directory: {artifact_relative_path}"
        ) from exc
    return path


def _safe_runtime_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PacketError(f"runtime manifest relative_path must be a nonempty string: {value!r}")
    if "\\" in value or "\x00" in value or any(ord(ch) < 32 for ch in value):
        raise PacketError(f"unsafe runtime manifest path: {value!r}")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise PacketError(f"runtime manifest path must stay inside runtime directory: {value!r}")
    if rel.name in {".DS_Store", "Thumbs.db"} or any(part.startswith("._") for part in rel.parts):
        raise PacketError(f"runtime manifest path is a resource-fork sidecar: {value!r}")
    return rel.as_posix()


def _unsafe_archive_member_name(name: str) -> str | None:
    if not name:
        return "empty_member_name"
    if "\\" in name or "\x00" in name or any(ord(ch) < 32 for ch in name):
        return "unsafe_member_name"
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        return "zip_slip_member_name"
    if "__MACOSX" in rel.parts:
        return "macosx_resource_directory"
    if rel.name in {".DS_Store", "Thumbs.db"} or any(part.startswith("._") for part in rel.parts):
        return "resource_fork_or_hidden_sidecar"
    if any(part.startswith(".") for part in rel.parts):
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
        header = handle.read(ZIP_LOCAL_HEADER_STRUCT.size)
        if len(header) != ZIP_LOCAL_HEADER_STRUCT.size:
            return None, None
        sig, _version, flag_bits, *_rest, name_len, extra_len = ZIP_LOCAL_HEADER_STRUCT.unpack(header)
        if sig != b"PK\x03\x04":
            return None, None
        raw_name = handle.read(name_len)
        _ = handle.read(extra_len)
    return _decode_zip_name(raw_name, flag_bits), flag_bits


def _inspect_archive_integrity(
    archive_path: Path,
    *,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "members": [],
        "duplicate_members": [],
        "packed_payload_members": [],
        "bad_crc_member": None,
    }
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        packed_payload_members = [name for name in names if name in PACKED_PAYLOAD_MEMBER_NAMES]
        bad_crc = zf.testzip()
        record["duplicate_members"] = duplicate_names
        record["packed_payload_members"] = packed_payload_members
        record["bad_crc_member"] = bad_crc
        _check(checks, "archive_crc_ok", bad_crc is None, f"bad_crc_member={bad_crc!r}")
        _check(checks, "archive_no_duplicate_members", not duplicate_names, f"duplicates={duplicate_names}")
        _check(
            checks,
            "archive_packed_payload_singleton",
            len(packed_payload_members) <= 1,
            f"packed_payload_members={packed_payload_members}",
        )
        for info in infos:
            local_name, local_flag_bits = _local_header_name(archive_path, info)
            unsafe = _unsafe_archive_member_name(info.filename)
            if not info.is_dir():
                with zf.open(info, "r") as handle:
                    member_sha = hashlib.sha256(handle.read()).hexdigest()
            else:
                member_sha = None
            record["members"].append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "sha256": member_sha,
                    "local_header_name": local_name,
                    "flag_bits": info.flag_bits,
                    "local_header_flag_bits": local_flag_bits,
                }
            )
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
    return record


def _runtime_manifest_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        runtime = provenance.get("inflate_runtime_manifest")
        if isinstance(runtime, dict):
            return runtime
    runtime = payload.get("inflate_runtime_manifest")
    if isinstance(runtime, dict):
        return runtime
    return {}


def _runtime_file_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = _runtime_manifest_from_payload(payload)
    rows = runtime.get("files")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PacketError(f"runtime manifest file row must be an object: {row!r}")
        rel = _safe_runtime_relative_path(row.get("relative_path"))
        if rel in seen:
            raise PacketError(f"duplicate runtime manifest path: {rel}")
        seen.add(rel)
        expected_sha = row.get("sha256")
        expected_bytes = row.get("bytes")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise PacketError(f"runtime manifest row has invalid sha256 for {rel}: {expected_sha!r}")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise PacketError(f"runtime manifest row has invalid bytes for {rel}: {expected_bytes!r}")
        out.append({"relative_path": rel, "sha256": expected_sha, "bytes": expected_bytes})
    return sorted(out, key=lambda item: item["relative_path"])


def _copy_checked_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _normalize_submission_mode(dst: Path, relative_path: str) -> None:
    if relative_path == "inflate.sh":
        dst.chmod(0o755)
    elif relative_path == "archive.zip" or relative_path == "report.txt":
        dst.chmod(0o644)


def _write_archive_manifest(
    submission_dir: Path,
    archive: dict[str, Any],
    archive_integrity: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    manifest_path = submission_dir / "archive_manifest.json"
    members = [
        {
            "name": row["name"],
            "file_size": row["file_size"],
            "compress_size": row["compress_size"],
            "sha256": row["sha256"],
        }
        for row in archive_integrity.get("members", [])
        if isinstance(row, dict) and row.get("sha256") is not None
    ]
    payload = {
        "schema_version": 1,
        "archive_sha256": archive["sha256"],
        "archive_size_bytes": archive["size_bytes"],
        "members": members,
    }
    write_json(manifest_path, payload)
    _normalize_submission_mode(manifest_path, "archive_manifest.json")
    return _copied_file_record(manifest_path, repo_root, submission_dir)


def _render_release_report(
    original_report: str,
    *,
    archive: dict[str, Any],
    contest_payload: dict[str, Any],
    expected_lane_id: str | None,
    expected_job_id: str | None,
) -> str:
    score = contest_payload.get("score_recomputed_from_components")
    samples = contest_payload.get("n_samples")
    seg = contest_payload.get("avg_segnet_dist")
    pose = contest_payload.get("avg_posenet_dist")
    lines = [
        original_report.rstrip(),
        "",
        "## Exact Custody",
        f"archive_sha256: {archive['sha256']}",
        f"archive_size_bytes: {archive['size_bytes']}",
        f"score_recomputed_from_components: {score}",
        f"n_samples: {samples}",
        f"avg_segnet_dist: {seg}",
        f"avg_posenet_dist: {pose}",
    ]
    if expected_lane_id:
        lines.append(f"dispatch_lane_id: {expected_lane_id}")
    if expected_job_id:
        lines.append(f"dispatch_job_id: {expected_job_id}")
    return "\n".join(lines).strip() + "\n"


def _copy_submission_packet(
    *,
    artifact_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
    submission_dir_name: str,
    contest_payload: dict[str, Any],
    archive: dict[str, Any],
    archive_integrity: dict[str, Any],
    expected_lane_id: str | None,
    expected_job_id: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    runtime_dir = runtime_dir.resolve()
    if not runtime_dir.is_dir():
        raise PacketError(f"runtime directory does not exist: {runtime_dir}")
    if not submission_dir_name or "/" in submission_dir_name or "\\" in submission_dir_name:
        raise PacketError(f"submission dir name must be a single safe path component: {submission_dir_name!r}")
    submission_dir = output_dir / submission_dir_name
    if submission_dir.exists() and any(submission_dir.iterdir()):
        raise PacketError(f"submission output directory already exists and is not empty: {submission_dir}")

    runtime_rows = _runtime_file_rows_from_payload(contest_payload)
    if not runtime_rows:
        raise PacketError("contest_auth_eval runtime manifest has no files to copy")

    copied_runtime_files: list[dict[str, Any]] = []
    for row in runtime_rows:
        rel = row["relative_path"]
        src = (runtime_dir / rel).resolve()
        try:
            src.relative_to(runtime_dir)
        except ValueError as exc:
            raise PacketError(f"runtime source escaped runtime directory: {rel}") from exc
        if not src.is_file():
            raise PacketError(f"missing runtime file selected by auth eval manifest: {src}")
        observed_sha = _sha256(src)
        observed_bytes = src.stat().st_size
        if observed_sha != row["sha256"] or observed_bytes != row["bytes"]:
            raise PacketError(
                "runtime file does not match auth eval manifest: "
                f"{rel} expected_sha={row['sha256']} observed_sha={observed_sha} "
                f"expected_bytes={row['bytes']} observed_bytes={observed_bytes}"
            )
        dst = submission_dir / rel
        _copy_checked_file(src, dst)
        _normalize_submission_mode(dst, rel)
        copied_runtime_files.append(_copied_file_record(dst, repo_root, submission_dir))

    copied_required_files: dict[str, dict[str, Any]] = {}
    for name in ("archive.zip", "report.txt"):
        src = artifact_dir / name
        if not src.is_file():
            raise PacketError(f"missing required packet file for copied submission: {src}")
        dst = submission_dir / name
        _copy_checked_file(src, dst)
        dst.chmod(0o644)
        if name == "report.txt":
            dst.write_text(
                _render_release_report(
                    dst.read_text(encoding="utf-8", errors="replace"),
                    archive=archive,
                    contest_payload=contest_payload,
                    expected_lane_id=expected_lane_id,
                    expected_job_id=expected_job_id,
                ),
                encoding="utf-8",
            )
        _normalize_submission_mode(dst, name)
        copied_required_files[name] = _copied_file_record(dst, repo_root, submission_dir)
    archive_manifest = _write_archive_manifest(submission_dir, archive, archive_integrity, repo_root)

    return {
        "path": _rel(submission_dir, repo_root),
        "source_runtime_dir": _rel(runtime_dir, repo_root),
        "runtime_manifest_source": "contest_auth_eval.provenance.inflate_runtime_manifest.files",
        "required_files": copied_required_files,
        "archive_manifest": archive_manifest,
        "runtime_files": copied_runtime_files,
        "file_count": len(copied_runtime_files) + len(copied_required_files) + 1,
        "archive_copied": True,
        "report_copied": True,
        "raw_frames_copied": False,
    }


def _supporting_records(
    paths: list[Path] | None,
    *,
    repo_root: Path,
    artifact_dir: Path,
    evidence_use: str,
    label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths or []:
        resolved = path.resolve()
        if not resolved.is_file():
            raise PacketError(f"missing {label}: {path}")
        try:
            artifact_relative_path: str | None = resolved.relative_to(artifact_dir).as_posix()
        except ValueError:
            artifact_relative_path = None
        records.append(
            {
                "path": _rel(resolved, repo_root),
                "artifact_relative_path": artifact_relative_path,
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
                "evidence_use": evidence_use,
                "score_claim": False,
                "ranking_claim": False,
                "promotion_claim": False,
            }
        )
    return records


def _numeric(payload: dict[str, Any], key: str) -> int | float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise PacketError(f"contest_auth_eval.json missing finite numeric {key!r}")
    return value


def _optional_numeric(payload: dict[str, Any], key: str) -> int | float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise PacketError(f"contest_auth_eval.json {key!r} is not finite numeric")
    return value


def _check(checks: list[dict[str, Any]], name: str, passed: bool, details: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _score_from_components(seg_dist: float, pose_dist: float, archive_bytes: int) -> float:
    return 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * archive_bytes / SCORE_DENOMINATOR


def _packed_payload_logical_names(name: str, payload: bytes) -> list[str]:
    if name.endswith(".br") or name == "p":
        try:
            import brotli  # type: ignore
        except Exception:
            return []
        try:
            payload = brotli.decompress(payload)
        except Exception:
            return []
    if not payload.startswith(RPK1_MAGIC) or len(payload) < len(RPK1_MAGIC) + RPK1_HEADER_STRUCT.size:
        return []
    header_len = RPK1_HEADER_STRUCT.unpack_from(payload, len(RPK1_MAGIC))[0]
    header_start = len(RPK1_MAGIC) + RPK1_HEADER_STRUCT.size
    header_end = header_start + header_len
    if header_len <= 0 or header_end > len(payload):
        return []
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except Exception:
        return []
    members = header.get("members")
    if not isinstance(members, list):
        return []
    names = []
    for row in members:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            names.append(row["name"])
    return names


def _archive_member_names(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            for name in tuple(names):
                if name in PACKED_PAYLOAD_MEMBER_NAMES:
                    names.update(_packed_payload_logical_names(name, zf.read(name)))
            return names
    except zipfile.BadZipFile:
        return set()


def _archive_contains_member(path: Path, member_name: str) -> bool:
    return member_name in _archive_member_names(path)


def _archive_present_members(path: Path, member_names: tuple[str, ...]) -> list[str]:
    names = _archive_member_names(path)
    return [member for member in member_names if member in names]


def _validate_auth_log_contract(
    *,
    contract_name: str,
    present: bool,
    artifact_dir: Path,
    repo_root: Path,
    checks: list[dict[str, Any]],
    log_checks: tuple[tuple[str, str, str], ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "present": present,
        "required_log": "auth_eval.log",
        "score_claim": False,
    }
    if not present:
        return record

    log_path = artifact_dir / "auth_eval.log"
    record["auth_eval_log"] = _rel(log_path, repo_root)
    if not log_path.is_file():
        _check(
            checks,
            f"{contract_name}_auth_eval_log_present",
            False,
            f"archive contains charged {contract_name} payload but auth_eval.log is missing",
        )
        return record

    log_text = log_path.read_text(errors="replace")
    record["auth_eval_log_sha256"] = _sha256(log_path)
    for field_name, needle, check_detail in log_checks:
        present_in_log = needle in log_text
        record[field_name] = present_in_log
        _check(
            checks,
            f"{contract_name}_{field_name}",
            present_in_log,
            check_detail,
        )
    return record


def _validate_sjkl_contract(
    archive_path: Path,
    artifact_dir: Path,
    repo_root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    present = _archive_contains_member(archive_path, "sjkl.bin")
    return _validate_auth_log_contract(
        contract_name="sjkl",
        present=present,
        artifact_dir=artifact_dir,
        repo_root=repo_root,
        checks=checks,
        log_checks=(
            (
                "loaded_payload_log_present",
                "Loaded SJ-KL residual payload:",
                "archive contains charged sjkl.bin and auth_eval.log proves payload load",
            ),
            (
                "apply_log_present",
                "Applying SJ-KL residuals to JointFrameGenerator fake1",
                "archive contains charged sjkl.bin and auth_eval.log proves runtime apply path",
            ),
            (
                "strict_contract_passed_log_present",
                "SJ-KL strict contract passed:",
                "archive contains charged sjkl.bin and auth_eval.log proves SJKL_REQUIRE_APPLIED strict pass",
            ),
        ),
    )


def _validate_cdo1_contract(
    archive_path: Path,
    artifact_dir: Path,
    repo_root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    present_members = _archive_present_members(archive_path, CDO1_ARCHIVE_MEMBERS)
    record = _validate_auth_log_contract(
        contract_name="cdo1",
        present=bool(present_members),
        artifact_dir=artifact_dir,
        repo_root=repo_root,
        checks=checks,
        log_checks=(
            (
                "apply_log_present",
                "Applied CDO1 decoded-mask overlay",
                "archive contains charged CDO1 overlay and auth_eval.log proves runtime apply path",
            ),
        ),
    )
    record["present_members"] = present_members
    return record


def _validate_amr1_contract(
    archive_path: Path,
    artifact_dir: Path,
    repo_root: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    present_members = _archive_present_members(archive_path, AMR1_ARCHIVE_MEMBERS)
    record = _validate_auth_log_contract(
        contract_name="amr1",
        present=bool(present_members),
        artifact_dir=artifact_dir,
        repo_root=repo_root,
        checks=checks,
        log_checks=(
            (
                "apply_log_present",
                "Applied Alpha residual repair",
                "archive contains charged AMR1 repair and auth_eval.log proves runtime apply path",
            ),
        ),
    )
    record["present_members"] = present_members
    return record


def _contest_eval_record(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise PacketError("contest_auth_eval.json provenance must be an object when present")
    provenance = provenance or {}
    return {
        "score_fields": {
            "final_score": _optional_numeric(payload, "final_score"),
            "score_recomputed_from_components": _numeric(payload, "score_recomputed_from_components"),
            "score_seg_contribution": _optional_numeric(payload, "score_seg_contribution"),
            "score_pose_contribution": _optional_numeric(payload, "score_pose_contribution"),
            "score_rate_contribution": _optional_numeric(payload, "score_rate_contribution"),
        },
        "component_fields": {
            "avg_segnet_dist": _numeric(payload, "avg_segnet_dist"),
            "avg_posenet_dist": _numeric(payload, "avg_posenet_dist"),
            "archive_size_bytes": int(_numeric(payload, "archive_size_bytes")),
            "n_samples": int(_numeric(payload, "n_samples")),
            "rate_unscaled": _optional_numeric(payload, "rate_unscaled"),
        },
        "runtime_fields": {
            "inflate_elapsed_seconds": _optional_numeric(payload, "inflate_elapsed_seconds"),
            "evaluate_elapsed_seconds": _optional_numeric(payload, "evaluate_elapsed_seconds"),
            "contest_auth_eval_elapsed_seconds": _optional_numeric(
                payload, "contest_auth_eval_elapsed_seconds"
            ),
        },
        "provenance_fields": {
            "tool": provenance.get("tool"),
            "archive_path": provenance.get("archive_path"),
            "archive_sha256": provenance.get("archive_sha256") or payload.get("archive_sha256"),
            "archive_size_bytes": provenance.get("archive_size_bytes"),
            "device": provenance.get("device"),
            "gpu_model": provenance.get("gpu_model"),
            "gpu_t4_match": provenance.get("gpu_t4_match"),
            "cuda_available": provenance.get("cuda_available"),
            "cuda_device_count": provenance.get("cuda_device_count"),
            "upstream_commit": provenance.get("upstream_commit"),
            "pact_commit": provenance.get("pact_commit"),
            "inflate_script": provenance.get("inflate_script"),
            "inflate_script_sha256": provenance.get("inflate_script_sha256"),
            "inflate_timeout_seconds": provenance.get("inflate_timeout_seconds"),
            "evaluate_timeout_seconds": provenance.get("evaluate_timeout_seconds"),
            "sys_argv": provenance.get("sys_argv"),
            "started_at_utc": provenance.get("started_at_utc"),
        },
    }


def _validate_trace(
    trace_payload: dict[str, Any],
    contest_record: dict[str, Any],
    accepted_contest_json_sha256s: set[str],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    component = contest_record["component_fields"]
    trace_record: dict[str, Any] = {
        "evidence_grade": trace_payload.get("evidence_grade"),
        "score_claim": trace_payload.get("score_claim"),
        "n_samples": trace_payload.get("n_samples"),
        "archive_size_bytes": trace_payload.get("archive_size_bytes"),
    }

    cross_check = trace_payload.get("contest_auth_eval_cross_check")
    if isinstance(cross_check, dict):
        trace_record["contest_auth_eval_cross_check"] = {
            "all_match": cross_check.get("all_match"),
            "contest_auth_eval_json_sha256": cross_check.get("contest_auth_eval_json_sha256"),
        }
        if cross_check.get("all_match") is False:
            _check(checks, "component_trace_cross_check", False, "component_trace all_match=false")
        else:
            _check(checks, "component_trace_cross_check", True, "component trace cross-check did not report a mismatch")
        trace_eval_sha = cross_check.get("contest_auth_eval_json_sha256")
        if trace_eval_sha is not None:
            _check(
                checks,
                "component_trace_contest_auth_eval_sha256",
                trace_eval_sha in accepted_contest_json_sha256s,
                "component trace cross-check points at an accepted contest_auth_eval JSON",
            )

    trace_samples = trace_payload.get("n_samples")
    if trace_samples is not None:
        _check(
            checks,
            "component_trace_n_samples",
            trace_samples == component["n_samples"],
            f"trace={trace_samples} contest={component['n_samples']}",
        )
    trace_bytes = trace_payload.get("archive_size_bytes")
    if trace_bytes is not None:
        _check(
            checks,
            "component_trace_archive_bytes",
            trace_bytes == component["archive_size_bytes"],
            f"trace={trace_bytes} contest={component['archive_size_bytes']}",
        )
    return trace_record


def _validate_eval_provenance(
    provenance_payload: dict[str, Any],
    archive: dict[str, Any],
    contest_record: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    record = {
        "tool": provenance_payload.get("tool"),
        "device": provenance_payload.get("device"),
        "gpu_model": provenance_payload.get("gpu_model"),
        "gpu_t4_match": provenance_payload.get("gpu_t4_match"),
        "archive_sha256": provenance_payload.get("archive_sha256"),
        "archive_size_bytes": provenance_payload.get("archive_size_bytes"),
    }
    if record["archive_sha256"] is not None:
        _check(
            checks,
            "eval_provenance_archive_sha256",
            record["archive_sha256"] == archive["sha256"],
            "eval_provenance archive_sha256 matches archive.zip",
        )
    if record["archive_size_bytes"] is not None:
        _check(
            checks,
            "eval_provenance_archive_bytes",
            record["archive_size_bytes"] == archive["size_bytes"],
            "eval_provenance archive_size_bytes matches archive.zip",
        )
    contest_prov = contest_record["provenance_fields"]
    if record["device"] is not None:
        _check(
            checks,
            "eval_provenance_device",
            record["device"] == contest_prov.get("device"),
            "eval_provenance device matches contest_auth_eval provenance",
        )
    return record


def _validate_adjudicated(
    adjudicated_payload: dict[str, Any],
    archive: dict[str, Any],
    contest_record: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance = adjudicated_payload.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise PacketError("contest_auth_eval.adjudicated.json provenance must be an object when present")
    provenance = provenance or {}
    record = {
        "score_recomputed_from_components": adjudicated_payload.get("score_recomputed_from_components"),
        "n_samples": adjudicated_payload.get("n_samples"),
        "archive_size_bytes": adjudicated_payload.get("archive_size_bytes"),
        "archive_sha256": provenance.get("archive_sha256") or adjudicated_payload.get("archive_sha256"),
        "device": provenance.get("device"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
    }
    if record["archive_sha256"] is not None:
        _check(
            checks,
            "adjudicated_archive_sha256",
            record["archive_sha256"] == archive["sha256"],
            "adjudicated JSON archive_sha256 matches archive.zip",
        )
    if record["archive_size_bytes"] is not None:
        _check(
            checks,
            "adjudicated_archive_bytes",
            record["archive_size_bytes"] == archive["size_bytes"],
            "adjudicated JSON archive_size_bytes matches archive.zip",
        )
    if record["n_samples"] is not None:
        _check(
            checks,
            "adjudicated_n_samples",
            record["n_samples"] == contest_record["component_fields"]["n_samples"],
            "adjudicated JSON n_samples matches contest_auth_eval.json",
        )
    return record


def _classify_evidence(
    checks_passed: bool,
    contest_record: dict[str, Any],
    optional_records: dict[str, Any],
) -> dict[str, Any]:
    component = contest_record["component_fields"]
    provenance = contest_record["provenance_fields"]
    reasons: list[str] = []
    blockers: list[str] = []

    if not checks_passed:
        return {
            "grade": "invalid",
            "basis": "field_check_failed",
            "reasons": [],
            "blockers": ["one or more packet validation checks failed"],
        }

    device = provenance.get("device")
    gpu_t4_match = provenance.get("gpu_t4_match")
    cuda_available = provenance.get("cuda_available")
    samples = component["n_samples"]
    adjudicated_present = optional_records.get("contest_auth_eval.adjudicated.json", {}).get("present")
    adjudication_provenance_present = optional_records.get("adjudication_provenance.json", {}).get("present")

    if device == "cuda":
        reasons.append("contest_auth_eval provenance reports device=cuda")
    else:
        blockers.append("contest_auth_eval provenance does not report device=cuda")
    if samples == 600:
        reasons.append("contest_auth_eval reports n_samples=600")
    else:
        blockers.append(f"contest_auth_eval reports n_samples={samples}")
    if cuda_available is True:
        reasons.append("contest_auth_eval provenance reports cuda_available=true")
    elif cuda_available is not None:
        blockers.append(f"contest_auth_eval provenance reports cuda_available={cuda_available!r}")
    if gpu_t4_match is True:
        reasons.append("contest_auth_eval provenance reports gpu_t4_match=true")
    elif gpu_t4_match is not None:
        blockers.append(f"contest_auth_eval provenance reports gpu_t4_match={gpu_t4_match!r}")
    if adjudicated_present and adjudication_provenance_present:
        reasons.append("adjudication JSON and adjudication provenance are present")

    if device == "cuda" and samples == 600 and gpu_t4_match is True and adjudicated_present:
        grade = "A++"
        basis = "cuda_t4_full_sample_adjudicated_fields"
    elif device == "cuda" and samples == 600:
        grade = "A"
        basis = "cuda_full_sample_fields"
    elif device == "cuda":
        grade = "B"
        basis = "cuda_incomplete_or_partial_fields"
    else:
        grade = "empirical"
        basis = "non_cuda_or_missing_cuda_fields"

    return {"grade": grade, "basis": basis, "reasons": reasons, "blockers": blockers}


def build_packet(
    artifact_dir: Path,
    output_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    score_authority: str = "contest_auth_eval.json",
    runtime_dir: Path | None = None,
    submission_dir_name: str = DEFAULT_SUBMISSION_DIR_NAME,
    expected_archive_sha256: str | None = None,
    expected_archive_size_bytes: int | None = None,
    expected_samples: int | None = None,
    expected_lane_id: str | None = None,
    expected_job_id: str | None = None,
    planner_ledgers: list[Path] | None = None,
    visualizations: list[Path] | None = None,
    next_action_tranches: list[Path] | None = None,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    archive_path = artifact_dir / "archive.zip"
    contest_json_path = _artifact_file(artifact_dir, score_authority)

    if not artifact_dir.is_dir():
        raise PacketError(f"artifact directory does not exist: {artifact_dir}")
    if not archive_path.is_file():
        raise PacketError(f"missing required artifact: {archive_path}")
    if not contest_json_path.is_file():
        raise PacketError(f"missing required artifact: {contest_json_path}")

    checks: list[dict[str, Any]] = []
    archive = {
        "path": _rel(archive_path, repo_root),
        "artifact_relative_path": "archive.zip",
        "size_bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
    }
    archive_integrity = _inspect_archive_integrity(archive_path, checks=checks)
    contest_payload = _read_json(contest_json_path)
    contest_record = _contest_eval_record(contest_payload)
    component = contest_record["component_fields"]
    provenance = contest_record["provenance_fields"]

    _check(
        checks,
        "archive_sha256_matches_contest_auth_eval",
        provenance.get("archive_sha256") == archive["sha256"],
        f"archive.zip sha256={archive['sha256']}",
    )
    _check(
        checks,
        "archive_bytes_matches_contest_auth_eval",
        component["archive_size_bytes"] == archive["size_bytes"],
        f"archive.zip bytes={archive['size_bytes']}",
    )
    if provenance.get("archive_size_bytes") is not None:
        _check(
            checks,
            "provenance_archive_bytes_matches_archive",
            provenance.get("archive_size_bytes") == archive["size_bytes"],
            "contest_auth_eval provenance archive_size_bytes matches archive.zip",
        )
    recomputed = _score_from_components(
        float(component["avg_segnet_dist"]),
        float(component["avg_posenet_dist"]),
        int(component["archive_size_bytes"]),
    )
    claimed_recomputed = float(contest_record["score_fields"]["score_recomputed_from_components"])
    _check(
        checks,
        "score_recomputes_from_components",
        math.isclose(recomputed, claimed_recomputed, rel_tol=0.0, abs_tol=1e-6),
        f"formula={recomputed:.17g} json={claimed_recomputed:.17g}",
    )
    if expected_archive_sha256 is not None:
        _check(
            checks,
            "expected_archive_sha256",
            archive["sha256"] == expected_archive_sha256,
            f"expected={expected_archive_sha256}",
        )
    if expected_archive_size_bytes is not None:
        _check(
            checks,
            "expected_archive_size_bytes",
            archive["size_bytes"] == expected_archive_size_bytes,
            f"expected={expected_archive_size_bytes}",
        )
    if expected_samples is not None:
        _check(
            checks,
            "expected_samples",
            component["n_samples"] == expected_samples,
            f"expected={expected_samples}",
        )

    eval_artifact = _file_record(contest_json_path, repo_root, artifact_dir)
    accepted_trace_eval_sha256s = {eval_artifact["sha256"]}
    canonical_contest_json_path = artifact_dir / "contest_auth_eval.json"
    if canonical_contest_json_path.is_file() and canonical_contest_json_path != contest_json_path:
        accepted_trace_eval_sha256s.add(_sha256(canonical_contest_json_path))
    optional_records: dict[str, Any] = {}
    for name in KNOWN_OPTIONAL_ARTIFACTS:
        path = artifact_dir / name
        record: dict[str, Any] = {"present": path.is_file()}
        if path.is_file():
            record.update(_file_record(path, repo_root, artifact_dir))
            if name == "component_trace.json":
                record["validated_fields"] = _validate_trace(
                    _read_json(path), contest_record, accepted_trace_eval_sha256s, checks
                )
            elif name == "eval_provenance.json":
                record["validated_fields"] = _validate_eval_provenance(
                    _read_json(path), archive, contest_record, checks
                )
            elif name == "contest_auth_eval.adjudicated.json":
                record["validated_fields"] = _validate_adjudicated(
                    _read_json(path), archive, contest_record, checks
                )
            elif name == "adjudication_provenance.json":
                payload = _read_json(path)
                record["validated_fields"] = {
                    "schema_version": payload.get("schema_version"),
                    "status": payload.get("status"),
                    "result_copy": payload.get("result_copy"),
                }
        optional_records[name] = record
    archive_payload_contracts = {
        "sjkl": _validate_sjkl_contract(archive_path, artifact_dir, repo_root, checks),
        "cdo1": _validate_cdo1_contract(archive_path, artifact_dir, repo_root, checks),
        "amr1": _validate_amr1_contract(archive_path, artifact_dir, repo_root, checks),
    }
    checks_passed = all(check["passed"] for check in checks)
    evidence_grade = _classify_evidence(checks_passed, contest_record, optional_records)
    supporting_artifacts = {
        "planner_ledgers": _supporting_records(
            planner_ledgers,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            evidence_use=SUPPORTING_ARTIFACT_SECTIONS["planner_ledgers"],
            label="planner ledger",
        ),
        "visualizations": _supporting_records(
            visualizations,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            evidence_use=SUPPORTING_ARTIFACT_SECTIONS["visualizations"],
            label="visualization",
        ),
        "next_action_tranches": _supporting_records(
            next_action_tranches,
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            evidence_use=SUPPORTING_ARTIFACT_SECTIONS["next_action_tranches"],
            label="next-action tranche",
        ),
        "claim_policy": {
            "score_claim": False,
            "ranking_claim": False,
            "promotion_claim": False,
            "supporting_artifacts_are_not_score_authorities": True,
        },
    }
    if not checks_passed:
        failed = [check["name"] for check in checks if not check["passed"]]
        raise PacketError(f"packet validation failed: {failed}")

    copied_submission: dict[str, Any] | None = None
    if runtime_dir is not None:
        copied_submission = _copy_submission_packet(
            artifact_dir=artifact_dir,
            runtime_dir=runtime_dir,
            output_dir=output_dir,
            submission_dir_name=submission_dir_name,
            contest_payload=contest_payload,
            archive=archive,
            archive_integrity=archive_integrity,
            expected_lane_id=expected_lane_id,
            expected_job_id=expected_job_id,
            repo_root=repo_root,
        )
    manifest = {
        "schema_version": 1,
        "packet_kind": "contest_faithful_submission_packet",
        "source_artifact_dir": _rel(artifact_dir, repo_root),
        "metadata_only": copied_submission is None,
        "copied_payloads": ["archive.zip", "report.txt"] if copied_submission is not None else [],
        "raw_frames_copied": False,
        "archive_copied": copied_submission is not None,
        "submission": copied_submission,
        "archive": archive,
        "archive_integrity": archive_integrity,
        "contest_auth_eval": {
            **eval_artifact,
            "validated_fields": contest_record,
        },
        "frontier_summary": {
            "score_authority": score_authority,
            "evidence_grade": evidence_grade["grade"],
            "score_recomputed_from_components": contest_record["score_fields"][
                "score_recomputed_from_components"
            ],
            "avg_segnet_dist": component["avg_segnet_dist"],
            "avg_posenet_dist": component["avg_posenet_dist"],
            "archive_size_bytes": component["archive_size_bytes"],
            "archive_sha256": archive["sha256"],
            "n_samples": component["n_samples"],
            "device": provenance.get("device"),
            "gpu_model": provenance.get("gpu_model"),
            "gpu_t4_match": provenance.get("gpu_t4_match"),
            "score_claim": False,
            "ranking_claim": False,
            "promotion_claim": False,
        },
        "optional_artifacts": optional_records,
        "archive_payload_contracts": archive_payload_contracts,
        "supporting_artifacts": supporting_artifacts,
        "evidence_grade": evidence_grade,
        "validation": {
            "status": "passed" if checks_passed else "failed",
            "checks": checks,
        },
        "claim_policy": {
            "score_claim": False,
            "ranking_claim": False,
            "promotion_claim": False,
            "score_fields_are_custody_metadata_only": True,
            "score_authority": score_authority,
        },
        "outputs": {
            "manifest": _rel(output_dir / DEFAULT_MANIFEST_NAME, repo_root),
            "checklist": _rel(output_dir / DEFAULT_CHECKLIST_NAME, repo_root),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    checklist_path = output_dir / DEFAULT_CHECKLIST_NAME
    write_json(manifest_path, manifest)
    checklist_path.write_text(render_checklist(manifest))
    return manifest


def render_checklist(manifest: dict[str, Any]) -> str:
    archive = manifest["archive"]
    eval_fields = manifest["contest_auth_eval"]["validated_fields"]
    component = eval_fields["component_fields"]
    provenance = eval_fields["provenance_fields"]
    grade = manifest["evidence_grade"]
    checks = manifest["validation"]["checks"]
    optional = manifest["optional_artifacts"]
    supporting = manifest["supporting_artifacts"]
    copied_submission = manifest.get("submission")

    lines = [
        "# Contest Submission Packet Checklist",
        "",
        (
            "This packet includes a concrete submission directory. It does not copy raw frames."
            if copied_submission
            else "This packet is metadata-only. It does not copy raw frames or archive payloads."
        ),
        "",
        "## Archive Custody",
        "",
        f"- [x] `archive.zip` exists at `{archive['path']}`.",
        f"- [x] SHA-256: `{archive['sha256']}`.",
        f"- [x] Byte size: `{archive['size_bytes']}`.",
        "",
    ]
    if copied_submission:
        lines.extend(
            [
                "## Copied Submission",
                "",
                f"- [x] Submission directory: `{copied_submission['path']}`.",
                f"- [x] Runtime source directory: `{copied_submission['source_runtime_dir']}`.",
                f"- [x] Runtime file count: `{len(copied_submission['runtime_files'])}`.",
                f"- [x] `archive.zip` and `report.txt` copied into the submission directory.",
                "",
            ]
        )
        for record in copied_submission["runtime_files"]:
            lines.append(
                f"- [x] Runtime `{record['submission_relative_path']}` "
                f"sha256 `{record['sha256']}` mode `{record['mode_octal']}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Frontier Snapshot",
            "",
            f"- Score authority JSON: `{manifest['claim_policy']['score_authority']}`.",
            f"- Field-supported grade: `{grade['grade']}`.",
            f"- Recomputed score field: `{eval_fields['score_fields']['score_recomputed_from_components']}`.",
            "- Score claim: `false`.",
            "- Ranking claim: `false`.",
            "- Promotion claim: `false`.",
            "",
            "## Auth Eval Fields",
            "",
            f"- [x] `{manifest['claim_policy']['score_authority']}` samples: `{component['n_samples']}`.",
            f"- [x] SegNet distance field present: `{component['avg_segnet_dist']}`.",
            f"- [x] PoseNet distance field present: `{component['avg_posenet_dist']}`.",
            f"- [x] Recomputed score field present: `{eval_fields['score_fields']['score_recomputed_from_components']}`.",
            f"- [x] Device field: `{provenance.get('device')}`.",
            f"- [x] GPU model field: `{provenance.get('gpu_model')}`.",
            "",
            "## Optional Evidence",
            "",
        ]
    )
    for name in KNOWN_OPTIONAL_ARTIFACTS:
        mark = "x" if optional[name]["present"] else " "
        lines.append(f"- [{mark}] `{name}`")
    lines.extend(
        [
            "",
            "## Non-Score Supporting Artifacts",
            "",
        ]
    )
    for section in ("planner_ledgers", "visualizations", "next_action_tranches"):
        evidence_use = SUPPORTING_ARTIFACT_SECTIONS[section]
        records = supporting[section]
        if records:
            for record in records:
                lines.append(f"- [x] `{section}` `{record['path']}`: `{evidence_use}`, score claim `false`.")
        else:
            lines.append(f"- [ ] `{section}`: none recorded.")
    lines.extend(
        [
            "",
            "## Validation",
            "",
        ]
    )
    for check in checks:
        mark = "x" if check["passed"] else " "
        lines.append(f"- [{mark}] `{check['name']}`: {check['details']}")
    lines.extend(
        [
            "",
            "## Evidence Classification",
            "",
            f"- Field-supported grade: `{grade['grade']}`.",
            f"- Basis: `{grade['basis']}`.",
            "- Score claim: `false`.",
            "- Ranking claim: `false`.",
            "- Promotion claim: `false`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--score-authority",
        default="contest_auth_eval.json",
        help="Artifact-directory JSON used as the exact-eval score authority.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help=(
            "Runtime directory to copy into output/submission using the "
            "contest_auth_eval runtime file manifest."
        ),
    )
    parser.add_argument(
        "--submission-dir-name",
        default=DEFAULT_SUBMISSION_DIR_NAME,
        help="Output subdirectory name for copied contest submission files.",
    )
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument("--expected-archive-size-bytes", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--expected-lane-id", default=None)
    parser.add_argument("--expected-job-id", default=None)
    parser.add_argument("--planner-ledger", type=Path, action="append", default=[])
    parser.add_argument("--visualization", type=Path, action="append", default=[])
    parser.add_argument("--next-action-tranche", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (args.artifact_dir / "submission_packet")
    manifest = build_packet(
        args.artifact_dir,
        output_dir,
        score_authority=args.score_authority,
        runtime_dir=args.runtime_dir,
        submission_dir_name=args.submission_dir_name,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_archive_size_bytes=args.expected_archive_size_bytes,
        expected_samples=args.expected_samples,
        expected_lane_id=args.expected_lane_id,
        expected_job_id=args.expected_job_id,
        planner_ledgers=args.planner_ledger,
        visualizations=args.visualization,
        next_action_tranches=args.next_action_tranche,
    )
    print(json_text(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
