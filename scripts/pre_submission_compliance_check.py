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


def _local_header_name(path: Path, info: zipfile.ZipInfo) -> str | None:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            return None
        flag_bits = struct.unpack_from("<H", header, 6)[0]
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        handle.read(extra_len)
    return _decode_zip_name(raw_name, flag_bits)


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
                local = _local_header_name(path, info)
                names.append(central)
                reason = unsafe_zip_name(central)
                local_reason = unsafe_zip_name(local or "")
                try:
                    payload = zf.read(info)
                    member_sha = _bytes_sha256(payload)
                except RuntimeError:
                    member_sha = None
                record["members"].append(
                    {
                        "name": central,
                        "local_header_name": local,
                        "file_size": info.file_size,
                        "compress_size": info.compress_size,
                        "sha256": member_sha,
                        "unsafe_reason": reason,
                    }
                )
                _add(checks, f"zip_member_safe:{central or '<empty>'}", reason is None, reason or central)
                _add(
                    checks,
                    f"zip_local_header_matches:{central or '<empty>'}",
                    local == central and local_reason is None,
                    f"central={central!r} local={local!r} local_reason={local_reason}",
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


def inspect_auth_eval(path: Path, archive: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    exists = path.is_file()
    _add(checks, "auth_eval_exists", exists or not args.require_auth_eval, _rel(path))
    if not exists:
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
    _add(checks, "auth_eval_archive_sha_matches", claimed_sha == archive_sha, f"claimed={claimed_sha} actual={archive_sha}")
    _add(
        checks,
        "auth_eval_archive_size_matches",
        claimed_size == archive_bytes,
        f"claimed={claimed_size} actual={archive_bytes}",
    )
    if record is not None:
        _add(checks, "auth_eval_has_components", record.avg_segnet_dist is not None and record.avg_posenet_dist is not None, "")
        if record.avg_segnet_dist is not None and record.avg_posenet_dist is not None and archive_bytes is not None:
            recomputed = _score(record.avg_segnet_dist, record.avg_posenet_dist, int(archive_bytes))
            _add(checks, "auth_eval_score_recomputes", abs(recomputed - float(record.score)) < 1e-6, f"record={record.score} recomputed={recomputed}")
        _add(
            checks,
            "auth_eval_t4_equivalent",
            (not args.require_t4_equivalent) or (record.device == "cuda" and record.samples == 600 and record.gpu_t4_match),
            f"device={record.device} samples={record.samples} gpu_t4_match={record.gpu_t4_match}",
        )
        _add(
            checks,
            "auth_eval_promotable_stamp",
            (not args.require_t4_equivalent)
            or (record.promotion_eligible and record.score_claim_valid and record.evidence_grade == "A++"),
            f"promotion={record.promotion_eligible} claim={record.score_claim_valid} grade={record.evidence_grade}",
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
        "runtime_tree_candidates": runtime_candidates,
    }, checks


def inspect_archive_manifest(path: Path, archive: dict[str, Any], *, required: bool) -> tuple[dict[str, Any], list[Check]]:
    checks: list[Check] = []
    exists = path.is_file()
    _add(checks, "archive_manifest_exists", exists or not required, _rel(path))
    if not exists:
        return {"path": _rel(path), "exists": False}, checks
    payload = _load_json(path)
    _add(checks, "archive_manifest_json_object", payload is not None, _rel(path))
    if payload is None:
        return {"path": _rel(path), "exists": True}, checks
    claimed_sha = _candidate_sha(payload)
    claimed_size = _candidate_size(payload)
    _add(checks, "archive_manifest_sha_matches", claimed_sha == archive.get("sha256"), f"claimed={claimed_sha} actual={archive.get('sha256')}")
    _add(checks, "archive_manifest_size_matches", claimed_size == archive.get("bytes"), f"claimed={claimed_size} actual={archive.get('bytes')}")
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
    terminal = False
    rows: list[str] = []
    if exists:
        rows = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip().startswith("|")]
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
                and status.startswith(TERMINAL_DISPATCH_STATUS_PREFIXES)
            ):
                terminal = True
    _add(checks, "dispatch_claim_terminal_row", terminal, f"lane_id={lane_id} job_id={job_id}")
    return {"path": _rel(path), "exists": exists, "rows": len(rows)}, checks


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
    parser.add_argument("--dispatch-claims-md", type=Path, default=REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md")
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
        _add(checks, "expected_archive_sha256_is_well_formed", bool(SHA256_RE.match(args.expected_archive_sha256)), args.expected_archive_sha256)
        _add(checks, "expected_archive_sha256_matches", archive.get("sha256") == args.expected_archive_sha256.lower(), f"expected={args.expected_archive_sha256} actual={archive.get('sha256')}")
    elif args.contest_final:
        _add(checks, "expected_archive_sha256_supplied", False, "contest-final requires explicit archive SHA")
    if args.expected_archive_size_bytes is not None:
        _add(checks, "expected_archive_size_bytes_matches", archive.get("bytes") == args.expected_archive_size_bytes, f"expected={args.expected_archive_size_bytes} actual={archive.get('bytes')}")
    elif args.contest_final:
        _add(checks, "expected_archive_size_bytes_supplied", False, "contest-final requires explicit archive bytes")

    auth_record, auth_checks = inspect_auth_eval(auth_path, archive, args)
    sections["auth_eval"] = auth_record
    checks.extend(auth_checks)
    manifest_record, manifest_checks = inspect_archive_manifest(manifest_path, archive, required=args.contest_final or args.archive_manifest_json is not None)
    sections["archive_manifest"] = manifest_record
    checks.extend(manifest_checks)
    report_record, report_checks = inspect_report(submission_dir / "report.txt", archive, required_link=args.contest_final)
    sections["report"] = report_record
    checks.extend(report_checks)
    claims_record, claims_checks = inspect_dispatch_claims(args.dispatch_claims_md, args.expected_lane_id, args.expected_job_id)
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
