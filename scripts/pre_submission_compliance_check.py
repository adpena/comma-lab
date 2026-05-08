#!/usr/bin/env python3
# ruff: noqa: I001
"""Strict pre-submission compliance gate for contest release packets.

This script validates the exact upload/publish surface. It does not run the
scorer and does not create a new score claim; it checks that an existing
auth-eval artifact, archive, manifest, report, and dispatch ledger agree.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import stat
import struct
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tools.auth_eval_records import parse_auth_eval_payload  # noqa: E402
from tac.preflight import check_public_release_hygiene  # noqa: E402
from tac.repo_io import json_text, read_json, repo_relative, sha256_file  # noqa: E402
from experiments.contest_auth_eval import (  # noqa: E402
    _repo_local_tac_import_manifest as _contest_repo_local_tac_import_manifest,
    _repo_rel as _contest_repo_rel,
    _runtime_dependency_extra_roots as _contest_runtime_dependency_extra_roots,
    _runtime_root_file_manifest as _contest_runtime_root_file_manifest,
    _sha256 as _contest_sha256,
)

ORIGINAL_VIDEO_BYTES = 37_545_489
SCHEMA = "pre_submission_compliance_check_v1"
PACKED_PAYLOAD_MEMBER_NAMES = ("p", "renderer_payload.bin", "renderer_payload.bin.br")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
TERMINAL_DISPATCH_STATUS_PREFIXES = (
    "completed",
    "failed",
    "stopped",
    "refused_dispatch",
    "stale_superseded",
)
PRIVATE_SURFACE_RE = re.compile(
    r"(/Users/|ssh\d+\.vast\.ai|fc-[A-Z0-9]{20,}|ap-[A-Za-z0-9]{12,}|sk-[A-Za-z0-9_-]{20,})"
)
SUBMISSION_RUNTIME_CUSTODY_FILENAMES = {
    "archive_manifest.json",
    "contest_auth_eval.json",
    "report.txt",
}
SUBMISSION_RUNTIME_CUSTODY_PREFIXES = ("pre_submission_compliance.",)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    severity: str
    details: str


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rel(path: Path) -> str:
    return repo_relative(path, REPO_ROOT)


def _add(checks: list[Check], name: str, passed: bool, details: str, *, severity: str = "error") -> None:
    checks.append(Check(name=name, passed=bool(passed), severity=severity, details=details))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _score(seg_dist: float, pose_dist: float, archive_bytes: int) -> float:
    return 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * archive_bytes / ORIGINAL_VIDEO_BYTES


def _rate_from_archive_bytes(archive_bytes: int) -> float:
    return archive_bytes / ORIGINAL_VIDEO_BYTES


def unsafe_zip_name(name: str) -> str | None:
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
    if any(part.startswith("._") for part in member.parts):
        return "resource_fork_member_name"
    if any(part.startswith(".") for part in member.parts):
        return "hidden_sidecar_member_name"
    if member.name in {".DS_Store", "Thumbs.db"}:
        return "resource_sidecar_member_name"
    return None


def _decode_zip_name(raw: bytes, flag_bits: int) -> str | None:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        return None


def _local_header_metadata(path: Path, info: zipfile.ZipInfo) -> dict[str, Any] | None:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return None
        version_needed = struct.unpack_from("<H", header, 4)[0]
        flag_bits = struct.unpack_from("<H", header, 6)[0]
        compress_type = struct.unpack_from("<H", header, 8)[0]
        mod_time, mod_date = struct.unpack_from("<HH", header, 10)
        crc32 = struct.unpack_from("<I", header, 14)[0]
        compress_size = struct.unpack_from("<I", header, 18)[0]
        file_size = struct.unpack_from("<I", header, 22)[0]
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        extra = handle.read(extra_len)
    return {
        "name": _decode_zip_name(raw_name, flag_bits),
        "version_needed": version_needed,
        "flag_bits": flag_bits,
        "compress_type": compress_type,
        "mod_time": mod_time,
        "mod_date": mod_date,
        "crc32": crc32,
        "compress_size": compress_size,
        "file_size": file_size,
        "extra": extra,
        "extra_len": extra_len,
    }


def _local_central_header_mismatches(
    local: dict[str, Any] | None,
    central: zipfile.ZipInfo,
) -> list[str]:
    if local is None:
        return ["local_header_missing_or_unreadable"]
    mismatches: list[str] = []
    local_name = local.get("name")
    if local_name != central.filename:
        mismatches.append(f"name:{local_name!r}!={central.filename!r}")
    local_reason = unsafe_zip_name(str(local_name or ""))
    if local_reason is not None:
        mismatches.append(f"local_name_unsafe:{local_reason}")
    if local.get("flag_bits") != central.flag_bits:
        mismatches.append(f"flag_bits:{local.get('flag_bits')}!={central.flag_bits}")
    if local.get("compress_type") != central.compress_type:
        mismatches.append(
            f"compress_type:{local.get('compress_type')}!={central.compress_type}"
        )
    year, month, day, hour, minute, second = central.date_time
    central_time = (hour << 11) | (minute << 5) | (second // 2)
    central_date = ((year - 1980) << 9) | (month << 5) | day
    if (local.get("mod_time"), local.get("mod_date")) != (central_time, central_date):
        mismatches.append("date_time_raw_mismatch")
    if local.get("extra") != central.extra:
        mismatches.append("extra_field_mismatch")
    uses_data_descriptor = bool(int(local.get("flag_bits") or 0) & 0x08)
    if not uses_data_descriptor:
        if local.get("crc32") != central.CRC:
            mismatches.append(f"crc32:{local.get('crc32')}!={central.CRC}")
        if local.get("compress_size") != central.compress_size:
            mismatches.append(
                f"compress_size:{local.get('compress_size')}!={central.compress_size}"
            )
        if local.get("file_size") != central.file_size:
            mismatches.append(f"file_size:{local.get('file_size')}!={central.file_size}")
    return mismatches


def inspect_archive(path: Path, *, expect_single_member: str | None = None) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    record: dict[str, Any] = {
        "path": _rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "members": [],
    }
    _add(checks, "archive_exists", path.is_file(), _rel(path))
    if not path.is_file():
        return record, checks

    names: list[str] = []
    packed_payload_count = 0
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            for info in infos:
                central = info.filename
                local_meta = _local_header_metadata(path, info)
                local = local_meta.get("name") if local_meta else None
                names.append(central)
                reason = unsafe_zip_name(central)
                local_reason = unsafe_zip_name(local or "")
                header_mismatches = _local_central_header_mismatches(local_meta, info)
                _add(checks, f"zip_member_safe:{central or '<empty>'}", reason is None, reason or central)
                _add(
                    checks,
                    f"zip_local_header_matches:{central or '<empty>'}",
                    local == central and local_reason is None,
                    f"central={central!r} local={local!r} local_reason={local_reason}",
                )
                _add(
                    checks,
                    f"zip_local_header_metadata_matches:{central or '<empty>'}",
                    not header_mismatches,
                    ", ".join(header_mismatches) or central,
                )
                try:
                    payload = zf.read(info)
                    member_sha = _bytes_sha256(payload)
                    member_readable = True
                    member_read_details = central
                except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
                    member_sha = None
                    member_readable = False
                    member_read_details = repr(exc)
                record["members"].append(
                    {
                        "name": central,
                        "local_header_name": local,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "sha256": member_sha,
                        "unsafe_reason": reason,
                        "local_header": {
                            key: value
                            for key, value in (local_meta or {}).items()
                            if key != "extra"
                        },
                    }
                )
                _add(
                    checks,
                    f"zip_member_payload_readable:{central or '<empty>'}",
                    member_readable,
                    member_read_details,
                )
                if central in PACKED_PAYLOAD_MEMBER_NAMES:
                    packed_payload_count += 1
    except zipfile.BadZipFile as exc:
        _add(checks, "archive_zip_readable", False, repr(exc))
        return record, checks

    _add(checks, "archive_zip_readable", True, f"members={len(names)}")
    _add(checks, "zip_no_duplicate_members", len(names) == len(set(names)), f"members={names}")
    if expect_single_member:
        _add(checks, "zip_expected_single_member", names == [expect_single_member], f"members={names}")
    _add(
        checks,
        "zip_at_most_one_packed_payload_container",
        packed_payload_count <= 1,
        f"packed_payload_count={packed_payload_count}",
    )
    return record, checks


def _candidate_sha(payload: dict[str, Any]) -> str | None:
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
    for value in (
        provenance.get("archive_sha256"),
        payload.get("candidate_archive_sha256"),
        payload.get("archive_sha256"),
        payload.get("sha256"),
        archive.get("candidate_archive_sha256"),
        archive.get("sha256"),
        archive.get("archive_sha256"),
    ):
        if isinstance(value, str) and SHA256_RE.match(value):
            return value.lower()
    return None


def _candidate_size(payload: dict[str, Any]) -> int | None:
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    archive = payload.get("archive") if isinstance(payload.get("archive"), dict) else {}
    for value in (
        provenance.get("archive_size_bytes"),
        payload.get("candidate_archive_bytes"),
        payload.get("candidate_archive_size_bytes"),
        payload.get("archive_size_bytes"),
        payload.get("archive_bytes"),
        payload.get("bytes"),
        archive.get("candidate_archive_bytes"),
        archive.get("candidate_archive_size_bytes"),
        archive.get("size_bytes"),
        archive.get("archive_size_bytes"),
    ):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _runtime_tree_candidates(payload: dict[str, Any]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    for scope, obj in (("root", payload), ("provenance", provenance)):
        runtime = obj.get("inflate_runtime_manifest") if isinstance(obj, dict) else None
        if isinstance(runtime, dict) and isinstance(runtime.get("runtime_tree_sha256"), str):
            candidates[f"{scope}.inflate_runtime_manifest.runtime_tree_sha256"] = runtime[
                "runtime_tree_sha256"
            ]
        if isinstance(obj.get("runtime_tree_sha256"), str):
            candidates[f"{scope}.runtime_tree_sha256"] = obj["runtime_tree_sha256"]
    return candidates


def _submission_runtime_manifest(submission_dir: Path) -> dict[str, Any]:
    inflate_sh = submission_dir / "inflate.sh"
    runtime_root = inflate_sh.parent.resolve()
    repo_root = REPO_ROOT.resolve()
    upstream_dir = repo_root / "upstream"

    def _is_custody_file(row: dict[str, Any]) -> bool:
        rel = str(row.get("relative_path") or "")
        name = PurePosixPath(rel).name
        return name in SUBMISSION_RUNTIME_CUSTODY_FILENAMES or any(
            name.startswith(prefix) for prefix in SUBMISSION_RUNTIME_CUSTODY_PREFIXES
        )

    files = [
        row
        for row in _contest_runtime_root_file_manifest(runtime_root, repo_root)
        if not _is_custody_file(row)
    ]
    extra_roots = _contest_runtime_dependency_extra_roots(inflate_sh, repo_root)
    external_dependency_roots = []
    for extra_root in extra_roots:
        external_dependency_roots.append(
            {
                "root": str(extra_root),
                "repo_relative_root": _contest_repo_rel(extra_root, repo_root),
                "exists": extra_root.exists(),
                "files": _contest_runtime_root_file_manifest(extra_root, repo_root),
            }
        )

    evaluate_py = (upstream_dir / "evaluate.py").resolve()
    upstream_eval = None
    if evaluate_py.exists():
        upstream_eval = {
            "relative_path": "evaluate.py",
            "bytes": evaluate_py.stat().st_size,
            "sha256": _contest_sha256(evaluate_py, prefix=0),
        }
    repo_local_tac = _contest_repo_local_tac_import_manifest(runtime_root, repo_root)
    tree_payload = {
        "runtime_root_name": runtime_root.name,
        "files": files,
        "external_dependency_roots": external_dependency_roots,
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": upstream_eval,
    }
    tree_sha = hashlib.sha256(
        json.dumps(tree_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "pre_submission_runtime_dependency_manifest_v1",
        "basis": "contest_auth_eval_runtime_dependency_manifest_v1_without_custody_files",
        "runtime_root": str(runtime_root),
        "runtime_file_count": len(files),
        "runtime_tree_sha256": tree_sha,
        "files": files,
        "external_dependency_roots": external_dependency_roots,
        "repo_local_tac_import_manifest": repo_local_tac,
        "upstream_evaluate_py": upstream_eval,
        "excluded_custody_filenames": sorted(SUBMISSION_RUNTIME_CUSTODY_FILENAMES),
        "excluded_custody_prefixes": list(SUBMISSION_RUNTIME_CUSTODY_PREFIXES),
    }


def _auth_runtime_candidates(auth_eval: dict[str, Any]) -> dict[str, str]:
    candidates = auth_eval.get("runtime_tree_candidates")
    if isinstance(candidates, dict):
        return {str(key): str(value) for key, value in candidates.items()}
    return {}


def inspect_submission_runtime(
    submission_dir: Path,
    auth_eval: dict[str, Any],
    *,
    required: bool,
) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    inflate_sh = submission_dir / "inflate.sh"
    exists = inflate_sh.is_file()
    _add(
        checks,
        "submission_runtime_inflate_exists",
        exists or not required,
        _rel(inflate_sh),
    )
    if not exists:
        return {"inflate_sh": _rel(inflate_sh), "exists": False}, checks

    try:
        manifest = _submission_runtime_manifest(submission_dir)
    except OSError as exc:
        _add(
            checks,
            "submission_runtime_manifest_computable",
            not required,
            f"{exc.__class__.__name__}: {exc}",
        )
        return {"inflate_sh": _rel(inflate_sh), "exists": True}, checks

    _add(checks, "submission_runtime_manifest_computable", True, manifest.get("schema", ""))
    runtime_sha = manifest.get("runtime_tree_sha256")
    _add(
        checks,
        "submission_runtime_tree_recorded",
        (not required) or bool(runtime_sha),
        f"runtime_tree_sha256={runtime_sha}",
    )
    candidates = _auth_runtime_candidates(auth_eval)
    expected = set(candidates.values())
    _add(
        checks,
        "submission_runtime_tree_matches_auth_eval",
        (not required) or (isinstance(runtime_sha, str) and runtime_sha in expected),
        f"submission={runtime_sha} auth_eval_candidates={candidates}",
    )
    return manifest, checks


def _safe_provenance_snapshot(
    payload: dict[str, Any],
    *,
    archive: dict[str, Any],
    record: Any,
    strict_formula: dict[str, Any] | None,
    auth_eval_path: Path,
    runtime_candidates: dict[str, str],
) -> dict[str, Any]:
    """Build a small clean-checkout proof from a local auth-eval artifact.

    The raw Lightning auth-eval JSON lives in ignored custody because it carries
    provider-local paths. This snapshot preserves the exact facts needed for
    archive/score/runtree reproducibility without committing provider logs.
    """

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    runtime_manifest = provenance.get("inflate_runtime_manifest")
    if not isinstance(runtime_manifest, dict):
        runtime_manifest = {}
    runtime_files = runtime_manifest.get("files")
    if not isinstance(runtime_files, list):
        runtime_files = []

    def _runtime_file(row: Any) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        rel = row.get("repo_relative_path") or row.get("relative_path")
        sha = row.get("sha256")
        size = row.get("bytes")
        if rel is None or sha is None or size is None:
            return None
        return {
            "repo_relative_path": str(rel),
            "bytes": int(size),
            "sha256": str(sha),
        }

    runtime_file_rows = [
        item for item in (_runtime_file(row) for row in runtime_files) if item is not None
    ]
    def _repoish(raw: Any, fallback: str) -> str:
        text = str(raw or "")
        if "/pact/" in text:
            return text.split("/pact/", 1)[1]
        if not text or text.startswith("/"):
            return fallback
        return text

    eval_args = [
        "experiments/contest_auth_eval.py",
        "--archive",
        "archive.zip",
        "--inflate-sh",
        _repoish(
            provenance.get("inflate_script"),
            "submissions/pr103_pr106_final_runtime/inflate.sh",
        ),
        "--upstream-dir",
        "upstream",
        "--device",
        str(record.device),
    ]
    upstream_eval = runtime_manifest.get("upstream_evaluate_py")
    return {
        "schema": "pre_submission_compliance_anchor_proof_v1",
        "auth_eval_local_custody_path": _rel(auth_eval_path),
        "auth_eval_local_custody_ignored_by_git": True,
        "score_basis": strict_formula,
        "record_score_source": "tools.auth_eval_records.parse_auth_eval_payload",
        "archive": {
            "path": archive.get("path"),
            "bytes": archive.get("bytes"),
            "sha256": archive.get("sha256"),
            "members": archive.get("members", []),
        },
        "auth_eval": {
            "evidence_grade": record.evidence_grade,
            "device": record.device,
            "samples": record.samples,
            "gpu_t4_match": record.gpu_t4_match,
            "gpu_model": provenance.get("gpu_model"),
            "gpu_driver": provenance.get("gpu_driver"),
            "cuda_version": provenance.get("cuda_version"),
            "torch_version": provenance.get("torch_version"),
            "contest_auth_eval_elapsed_seconds": payload.get(
                "contest_auth_eval_elapsed_seconds"
            ),
            "inflate_elapsed_seconds": payload.get("inflate_elapsed_seconds"),
            "evaluate_elapsed_seconds": payload.get("evaluate_elapsed_seconds"),
            "eval_command_sanitized": eval_args,
            "tool": provenance.get("tool"),
            "upstream_commit": provenance.get("upstream_commit"),
            "upstream_evaluate_py": upstream_eval
            if isinstance(upstream_eval, dict)
            else None,
        },
        "runtime": {
            "runtime_tree_sha256_candidates": runtime_candidates,
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "runtime_files": runtime_file_rows,
            "runtime_file_count": runtime_manifest.get(
                "runtime_file_count", len(runtime_file_rows)
            ),
        },
    }


def inspect_auth_eval(
    path: Path,
    archive: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    exists = path.is_file()
    _add(checks, "auth_eval_exists", exists or not args.require_auth_eval, _rel(path))
    if not exists:
        if not args.require_auth_eval:
            _add(
                checks,
                "auth_eval_optional_missing",
                False,
                (
                    f"{_rel(path)} missing; nonfinal compliance may pass, but "
                    "this packet has no auth-eval custody and is not score/rank/promote evidence"
                ),
                severity="warning",
            )
        return {"path": _rel(path), "exists": False}, checks

    payload = _load_json(path)
    _add(checks, "auth_eval_json_object", payload is not None, _rel(path))
    if payload is None:
        return {"path": _rel(path), "exists": True}, checks

    record = parse_auth_eval_payload(payload)
    _add(checks, "auth_eval_score_parseable", record is not None, "canonical auth eval parser")
    archive_sha = archive.get("sha256")
    archive_bytes = archive.get("bytes")
    claimed_sha = _candidate_sha(payload)
    claimed_size = _candidate_size(payload)
    _add(
        checks,
        "auth_eval_archive_sha_matches",
        claimed_sha == archive_sha,
        f"claimed={claimed_sha} actual={archive_sha}",
    )
    _add(
        checks,
        "auth_eval_archive_size_matches",
        claimed_size == archive_bytes,
        f"claimed={claimed_size} actual={archive_bytes}",
    )
    if record is not None:
        _add(
            checks,
            "auth_eval_has_components",
            record.avg_segnet_dist is not None
            and record.avg_posenet_dist is not None,
            "",
        )
        strict_formula: dict[str, Any] | None = None
        if (
            record.avg_segnet_dist is not None
            and record.avg_posenet_dist is not None
            and archive_bytes is not None
        ):
            recomputed = _score(record.avg_segnet_dist, record.avg_posenet_dist, int(archive_bytes))
            score_delta = recomputed - float(record.score)
            strict_formula = {
                "basis": "auth_eval_report_components_plus_exact_archive_bytes",
                "score": recomputed,
                "report_reconstructed_score": float(record.score),
                "score_delta_vs_report_reconstruction": score_delta,
                "archive_rate_unscaled": _rate_from_archive_bytes(int(archive_bytes)),
                "avg_segnet_dist": record.avg_segnet_dist,
                "avg_posenet_dist": record.avg_posenet_dist,
                "archive_bytes": int(archive_bytes),
                "note": (
                    "contest_auth_eval reconstructs score from upstream report fields; "
                    "this strict value recomputes the contest formula from those report "
                    "components and exact charged archive bytes for clean-checkout "
                    "anchor comparisons"
                ),
            }
            _add(
                checks,
                "auth_eval_score_recomputes",
                abs(recomputed - float(record.score)) < 1e-6,
                f"record={record.score} recomputed={recomputed}",
            )
            _add(
                checks,
                "auth_eval_strict_formula_score_recorded",
                math.isfinite(recomputed),
                f"strict={recomputed} record={record.score} delta={score_delta}",
            )
        _add(
            checks,
            "auth_eval_t4_equivalent",
            (not args.require_t4_equivalent)
            or (
                record.device == "cuda"
                and record.samples == 600
                and record.gpu_t4_match
            ),
            f"device={record.device} samples={record.samples} gpu_t4_match={record.gpu_t4_match}",
        )
        _add(
            checks,
            "auth_eval_promotable_stamp",
            (not args.require_t4_equivalent)
            or (record.promotion_eligible and record.score_claim_valid and record.evidence_grade == "A++"),
            f"promotion={record.promotion_eligible} claim={record.score_claim_valid} grade={record.evidence_grade}",
        )
        _add(
            checks,
            "auth_eval_explicit_promotable_stamp",
            (not args.contest_final)
            or (
                payload.get("promotion_eligible") is True
                and payload.get("score_claim_valid") is True
                and payload.get("evidence_grade") == "A++"
            ),
            "contest-final requires explicit promotion_eligible=true, "
            "score_claim_valid=true, evidence_grade=A++ in the auth-eval JSON",
        )

    runtime_candidates = _runtime_tree_candidates(payload)
    require_runtime = args.require_auth_eval or args.require_submission_runtime_match or args.contest_final
    _add(
        checks,
        "auth_eval_runtime_tree_recorded",
        (not require_runtime) or bool(runtime_candidates),
        f"candidates={runtime_candidates}",
    )
    if args.expected_runtime_tree_sha256:
        _add(
            checks,
            "auth_eval_runtime_tree_expected_match",
            args.expected_runtime_tree_sha256 in set(runtime_candidates.values()),
            f"expected={args.expected_runtime_tree_sha256} candidates={runtime_candidates}",
        )

    return {
        "path": _rel(path),
        "exists": True,
        "record": asdict(record) if record else None,
        "strict_formula": strict_formula if record else None,
        "anchor_proof": (
            _safe_provenance_snapshot(
                payload,
                archive=archive,
                record=record,
                strict_formula=strict_formula,
                auth_eval_path=path,
                runtime_candidates=runtime_candidates,
            )
            if record and strict_formula is not None
            else None
        ),
        "runtime_tree_candidates": runtime_candidates,
    }, checks


def inspect_archive_manifest(
    path: Path,
    archive: dict[str, Any],
    *,
    required: bool,
) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    exists = path.is_file()
    _add(checks, "archive_manifest_exists", exists or not required, _rel(path))
    if not exists:
        if not required:
            _add(
                checks,
                "archive_manifest_optional_missing",
                False,
                (
                    f"{_rel(path)} missing; nonfinal compliance may pass, but "
                    "archive identity is not manifest-closed"
                ),
                severity="warning",
            )
        return {"path": _rel(path), "exists": False}, checks
    payload = _load_json(path)
    _add(checks, "archive_manifest_json_object", payload is not None, _rel(path))
    if payload is None:
        return {"path": _rel(path), "exists": True}, checks
    claimed_sha = _candidate_sha(payload)
    claimed_size = _candidate_size(payload)
    _add(
        checks,
        "archive_manifest_sha_matches",
        claimed_sha == archive.get("sha256"),
        f"claimed={claimed_sha} actual={archive.get('sha256')}",
    )
    _add(
        checks,
        "archive_manifest_size_matches",
        claimed_size == archive.get("bytes"),
        f"claimed={claimed_size} actual={archive.get('bytes')}",
    )
    return {"path": _rel(path), "exists": True}, checks


def inspect_submission_dir(submission_dir: Path) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    files: dict[str, Any] = {}
    for rel in ("archive.zip", "inflate.sh", "report.txt"):
        path = submission_dir / rel
        exists = path.is_file()
        files[rel] = {"path": _rel(path), "exists": exists}
        _add(checks, f"required_file_present:{rel}", exists, _rel(path))
    inflate = submission_dir / "inflate.sh"
    if inflate.is_file():
        executable = bool(inflate.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        _add(checks, "inflate_sh_executable", executable, f"mode={oct(stat.S_IMODE(inflate.stat().st_mode))}")
    return {"path": _rel(submission_dir), "required_files": files}, checks


def inspect_report(path: Path, archive: dict[str, Any], *, required_link: bool) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    exists = path.is_file()
    _add(checks, "report_exists", exists, _rel(path))
    text = path.read_text(encoding="utf-8", errors="ignore") if exists else ""
    if required_link:
        sha = str(archive.get("sha256") or "")
        size = str(archive.get("bytes") or "")
        _add(checks, "report_mentions_archive_sha256", sha in text, sha[:16])
        _add(checks, "report_mentions_archive_size_bytes", size in text, size)
    return {"path": _rel(path), "exists": exists}, checks


def inspect_dispatch_claims(path: Path, lane_id: str | None, job_id: str | None) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    if not lane_id and not job_id:
        return {"required": False}, checks
    exists = path.is_file()
    _add(checks, "dispatch_claims_exists", exists, _rel(path))
    latest_matching_status: str | None = None
    latest_matching_row: str | None = None
    matching_rows = 0
    rows: list[str] = []
    if exists:
        rows = [
            line.strip()
            for line in path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            if line.strip().startswith("|")
        ]
        header: list[str] | None = None
        for row in rows:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if not cells or set(cells) <= {"---"}:
                continue
            lowered = [cell.lower() for cell in cells]
            if "lane_id" in lowered and "status" in lowered:
                header = lowered
                continue
            if header:
                lookup = {name: cells[idx] for idx, name in enumerate(header) if idx < len(cells)}
                row_lane = lookup.get("lane_id", "")
                row_job = lookup.get("instance/job_id", "") or lookup.get("job_id", "")
                status = lookup.get("status", "")
            elif len(cells) >= 6:
                row_lane = cells[1]
                row_job = cells[3]
                status = cells[4]
            else:
                continue
            if (
                (lane_id is None or row_lane == lane_id)
                and (job_id is None or row_job == job_id)
            ):
                matching_rows += 1
                if latest_matching_status is None:
                    latest_matching_status = status
                    latest_matching_row = row
    terminal = bool(
        latest_matching_status
        and latest_matching_status.startswith(TERMINAL_DISPATCH_STATUS_PREFIXES)
    )
    _add(
        checks,
        "dispatch_claim_terminal_row",
        terminal,
        (
            f"lane_id={lane_id} job_id={job_id} matching_rows={matching_rows} "
            f"latest_matching_status={latest_matching_status!r}"
        ),
    )
    return {
        "path": _rel(path),
        "exists": exists,
        "rows": len(rows),
        "matching_rows": matching_rows,
        "latest_matching_status": latest_matching_status,
        "latest_matching_row": latest_matching_row,
    }, checks


def inspect_public_hygiene(paths: list[Path]) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    violations = check_public_release_hygiene(
        repo_root=REPO_ROOT,
        scan_paths=paths,
        strict=False,
        verbose=False,
    )
    manual_hits: list[str] = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            if not candidate.is_file():
                continue
            for lineno, line in enumerate(candidate.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if PRIVATE_SURFACE_RE.search(line):
                    manual_hits.append(f"{_rel(candidate)}:{lineno}")
    hits = sorted(set(violations + manual_hits))
    _add(checks, "public_scan_has_no_private_surface", not hits, ", ".join(hits[:20]))
    return {"scan_paths": [_rel(path) for path in paths], "hits": hits}, checks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--auth-eval-json", type=Path)
    parser.add_argument("--archive-manifest-json", type=Path)
    parser.add_argument("--require-auth-eval", action="store_true")
    parser.add_argument("--require-t4-equivalent", action="store_true")
    parser.add_argument("--require-submission-runtime-match", action="store_true")
    parser.add_argument("--contest-final", action="store_true")
    parser.add_argument("--expect-single-member")
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--expected-archive-size-bytes", type=int)
    parser.add_argument("--expected-runtime-tree-sha256")
    parser.add_argument(
        "--dispatch-claims-md",
        type=Path,
        default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md",
    )
    parser.add_argument("--expected-lane-id")
    parser.add_argument("--expected-job-id")
    parser.add_argument("--public-scan-path", type=Path, action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.contest_final:
        args.require_auth_eval = True
        args.require_t4_equivalent = True
        args.require_submission_runtime_match = True

    submission_dir = args.submission_dir
    archive_path = args.archive or submission_dir / "archive.zip"
    auth_path = args.auth_eval_json or submission_dir / "contest_auth_eval.json"
    manifest_path = args.archive_manifest_json or submission_dir / "archive_manifest.json"

    sections: dict[str, Any] = {}
    checks: list[Check] = []
    for key, (record, section_checks) in {
        "submission_dir": inspect_submission_dir(submission_dir),
        "archive": inspect_archive(archive_path, expect_single_member=args.expect_single_member),
    }.items():
        sections[key] = record
        checks.extend(section_checks)

    archive = sections["archive"]
    if args.expected_archive_sha256:
        _add(
            checks,
            "expected_archive_sha256_is_well_formed",
            bool(SHA256_RE.match(args.expected_archive_sha256)),
            args.expected_archive_sha256,
        )
        _add(
            checks,
            "expected_archive_sha256_matches",
            archive.get("sha256") == args.expected_archive_sha256.lower(),
            f"expected={args.expected_archive_sha256} actual={archive.get('sha256')}",
        )
    elif args.contest_final:
        _add(checks, "expected_archive_sha256_supplied", False, "contest-final requires explicit archive SHA")
    if args.expected_archive_size_bytes is not None:
        _add(
            checks,
            "expected_archive_size_bytes_matches",
            archive.get("bytes") == args.expected_archive_size_bytes,
            f"expected={args.expected_archive_size_bytes} actual={archive.get('bytes')}",
        )
    elif args.contest_final:
        _add(checks, "expected_archive_size_bytes_supplied", False, "contest-final requires explicit archive bytes")

    auth_record, auth_checks = inspect_auth_eval(auth_path, archive, args)
    sections["auth_eval"] = auth_record
    checks.extend(auth_checks)
    runtime_record, runtime_checks = inspect_submission_runtime(
        submission_dir,
        auth_record,
        required=args.require_submission_runtime_match,
    )
    sections["submission_runtime"] = runtime_record
    checks.extend(runtime_checks)
    manifest_record, manifest_checks = inspect_archive_manifest(
        manifest_path,
        archive,
        required=args.contest_final or args.archive_manifest_json is not None,
    )
    sections["archive_manifest"] = manifest_record
    checks.extend(manifest_checks)
    report_record, report_checks = inspect_report(
        submission_dir / "report.txt",
        archive,
        required_link=args.contest_final,
    )
    sections["report"] = report_record
    checks.extend(report_checks)
    if args.contest_final:
        _add(
            checks,
            "contest_final_expected_lane_id_supplied",
            bool(args.expected_lane_id),
            "contest-final requires --expected-lane-id",
        )
        _add(
            checks,
            "contest_final_expected_job_id_supplied",
            bool(args.expected_job_id),
            "contest-final requires --expected-job-id",
        )
    claims_record, claims_checks = inspect_dispatch_claims(
        args.dispatch_claims_md,
        args.expected_lane_id,
        args.expected_job_id,
    )
    sections["dispatch_claims"] = claims_record
    checks.extend(claims_checks)
    if args.public_scan_path:
        hygiene_record, hygiene_checks = inspect_public_hygiene(args.public_scan_path)
        sections["public_hygiene"] = hygiene_record
        checks.extend(hygiene_checks)

    passed = all(check.passed or check.severity != "error" for check in checks)
    return {
        "schema": SCHEMA,
        "passed": passed,
        "checks": [asdict(check) for check in checks],
        **sections,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(args)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(report), encoding="utf-8")
    else:
        print(json_text(report), end="")
    if args.strict and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
