#!/usr/bin/env python3
"""Build sanitized local readiness artifacts for PR98/PR99 final packets.

This is intentionally local-only. It copies cleaned runtime snapshots, copies
the exact public archives for custody convenience, and writes deterministic
manifests/checklists. It never inflates video, loads the scorer, or dispatches
remote work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = REPO_ROOT / "experiments/results/leaderboard_intel_20260504_codex"
UPSTREAM_EVALUATE = REPO_ROOT / "upstream/evaluate.py"

SKIP_DIRS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
SKIP_FILE_NAMES = {".DS_Store", "Thumbs.db"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}
RUNTIME_SUFFIXES = {".c", ".cc", ".cpp", ".env", ".h", ".hpp", ".json", ".py", ".sh", ".toml", ".txt"}
ABS_PATH_RE = re.compile(r"(/Users/[A-Za-z0-9_.@+-]+/[^\\s'\"`]+|/teamspace/[^\\s'\"`]+|/workspace/[^\\s'\"`]+)")
SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|ghp_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]+|sk-[A-Za-z0-9]{20,})"
)


@dataclass(frozen=True)
class Candidate:
    pr: int
    archive: Path
    runtime: Path
    expected_archive_sha256: str
    expected_archive_bytes: int
    expected_runtime_tree_sha256: str
    original_job: str
    duplicate_job: str
    original_lane: str
    duplicate_lane: str


CANDIDATES = (
    Candidate(
        pr=98,
        archive=SOURCE_ROOT / "pr98_archive.zip",
        runtime=SOURCE_ROOT / "pr98_runtime",
        expected_archive_sha256="7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb",
        expected_archive_bytes=178392,
        expected_runtime_tree_sha256="4d71b5769e9c886e8a4e1be8997014ec47fe5d5ce5519619bf16bff0ae7f2738",
        original_job="exact_eval_public_pr98_hnerv_muon_finetuned_t4_20260504T0940Z",
        duplicate_job="exact_eval_public_pr98_hnerv_muon_finetuned_t4_dup_20260504T0944Z",
        original_lane="public_pr98_hnerv_muon_finetuned_t4_replay",
        duplicate_lane="public_pr98_hnerv_muon_finetuned_t4_replay_dup",
    ),
    Candidate(
        pr=99,
        archive=SOURCE_ROOT / "pr99_archive.zip",
        runtime=SOURCE_ROOT / "pr99_runtime",
        expected_archive_sha256="278b1c7a1bd6b03a5bceddafcb3489b2624c558ad22825d9211b701333b6eefb",
        expected_archive_bytes=178546,
        expected_runtime_tree_sha256="67fa8ef36f732be73d29053bc050a86a597b23d394ea07538451a8eb8303817f",
        original_job="exact_eval_public_pr99_hnerv_muon_lc_t4_20260504T0940Z",
        duplicate_job="exact_eval_public_pr99_hnerv_muon_lc_t4_dup_20260504T0944Z",
        original_lane="public_pr99_hnerv_muon_lc_t4_replay",
        duplicate_lane="public_pr99_hnerv_muon_lc_t4_replay_dup",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def should_skip(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    if any(part in SKIP_DIRS for part in parts):
        return "skip_cache_dir"
    if any(part.startswith(".") for part in parts):
        return "skip_hidden_path"
    if path.name.startswith("._") or path.name in SKIP_FILE_NAMES:
        return "skip_resource_fork_or_desktop_sidecar"
    if path.is_file() and path.suffix.lower() in SKIP_SUFFIXES:
        return "skip_non_runtime_sidecar_suffix"
    return None


def copy_sanitized_runtime(src: Path, dst: Path) -> dict[str, Any]:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    skipped: list[dict[str, str]] = []
    copied: list[dict[str, Any]] = []
    for path in sorted(src.rglob("*"), key=lambda p: p.relative_to(src).as_posix()):
        reason = should_skip(path, src)
        if reason:
            skipped.append({"path": path.relative_to(src).as_posix(), "reason": reason})
            continue
        if path.is_dir():
            (dst / path.relative_to(src)).mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            skipped.append({"path": path.relative_to(src).as_posix(), "reason": "skip_not_regular_file"})
            continue
        rel_path = path.relative_to(src)
        target = dst / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(target, mode)
        copied.append(
            {
                "relative_path": rel_path.as_posix(),
                "bytes": target.stat().st_size,
                "mode": oct(stat.S_IMODE(target.stat().st_mode)),
                "sha256": sha256_file(target),
            }
        )
    return {"copied": copied, "skipped": skipped}


def source_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    abs_path_hits: list[dict[str, Any]] = []
    secret_hits: list[dict[str, Any]] = []
    skipped_suffixes: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        if path.suffix.lower() not in RUNTIME_SUFFIXES:
            skipped_suffixes.append(rel_path)
            continue
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="ignore")
        abs_matches = sorted(set(ABS_PATH_RE.findall(text)))
        sec_matches = sorted(set(SECRET_RE.findall(text)))
        if abs_matches:
            abs_path_hits.append({"path": rel_path, "matches": abs_matches})
        if sec_matches:
            secret_hits.append({"path": rel_path, "matches": sec_matches})
        files.append(
            {
                "relative_path": rel_path,
                "bytes": len(payload),
                "mode": oct(stat.S_IMODE(path.stat().st_mode)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "files": files,
        "runtime_file_count": len(files),
        "source_bytes": sum(row["bytes"] for row in files),
        "skipped_non_runtime_files": skipped_suffixes,
        "absolute_path_hits": abs_path_hits,
        "secret_like_hits": secret_hits,
    }


def upstream_eval_record() -> dict[str, Any] | None:
    if not UPSTREAM_EVALUATE.exists():
        return None
    return {
        "relative_path": "evaluate.py",
        "bytes": UPSTREAM_EVALUATE.stat().st_size,
        "sha256": sha256_file(UPSTREAM_EVALUATE),
    }


def runtime_tree_sha256(runtime_root: Path) -> str:
    manifest = source_manifest(runtime_root)
    tree_files = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in manifest["files"]
    ]
    payload = {
        "runtime_root_name": runtime_root.name,
        "files": tree_files,
        "repo_local_tac_import_manifest": {
            "schema": "contest_auth_eval_repo_local_tac_import_manifest_v1",
            "runtime_root_name": runtime_root.name,
            "tac_root_relative_path": "src/tac",
            "discovery": "static_ast_recursive_import_closure",
            "root_import_modules": [],
            "module_count": 0,
            "file_count": 0,
            "files": [],
            "unresolved_modules": [],
            "parse_errors": [],
        },
        "upstream_evaluate_py": upstream_eval_record(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def unsafe_zip_name(name: str) -> str | None:
    if not name:
        return "empty_member_name"
    if "\\" in name or "\x00" in name:
        return "unsafe_member_name"
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        return "zip_slip_member_name"
    if any(part.startswith(".") for part in member.parts) or "__MACOSX" in member.parts:
        return "hidden_or_resource_member"
    return None


def archive_manifest(src_archive: Path, dst_archive: Path) -> dict[str, Any]:
    if dst_archive.exists():
        dst_archive.unlink()
    dst_archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_archive, dst_archive)
    members: list[dict[str, Any]] = []
    blockers: list[str] = []
    with zipfile.ZipFile(dst_archive, "r") as zf:
        bad_crc = zf.testzip()
        if bad_crc:
            blockers.append(f"bad_crc:{bad_crc}")
        names = [info.filename for info in zf.infolist()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            blockers.append(f"duplicate_members:{duplicates}")
        for info in zf.infolist():
            unsafe = unsafe_zip_name(info.filename)
            if unsafe:
                blockers.append(f"{info.filename}:{unsafe}")
            with zf.open(info, "r") as handle:
                member_sha = hashlib.sha256(handle.read()).hexdigest()
            members.append(
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "sha256": member_sha,
                }
            )
    return {
        "source_path": rel(src_archive),
        "readiness_copy_path": rel(dst_archive),
        "bytes": dst_archive.stat().st_size,
        "sha256": sha256_file(dst_archive),
        "member_count": len(members),
        "members": members,
        "blockers": blockers,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_checklist(summary: dict[str, Any]) -> str:
    lines = [
        "# PR98/PR99 Final-Packet Readiness",
        "",
        "This directory is local readiness only. It does not make a score claim and it did not dispatch GPU work.",
        "",
        "## Promotion Steps",
        "",
        "For whichever PR98/PR99 T4 exact eval wins:",
        "",
        "1. Harvest the terminal Lightning job through `scripts/launch_lightning_batch_job.py harvest-ssh`.",
        "2. Require `contest_auth_eval.adjudicated.json` with CUDA, 600 samples, promotion eligibility, archive SHA/bytes matching this manifest, and runtime tree SHA matching the selected sanitized runtime.",
        "3. Close the original and duplicate dispatch claims with terminal rows; stop the redundant duplicate if still pending/running.",
        "4. Build the public packet with the selected exact archive copy plus the selected sanitized runtime snapshot, preserving the runtime root name used by exact eval or recording the new runtime tree hash explicitly.",
        "5. Add `report.txt` only from the adjudicated JSON values: exact score, SegNet, PoseNet, archive bytes, archive SHA, runtime tree SHA, hardware, sample count, and eval command.",
        "6. Run `scripts/pre_submission_compliance_check.py` on the exact packet surface with `--require-auth-eval --require-t4-equivalent --require-submission-runtime-match --expect-single-member 0.bin --require-report-archive-link --require-report-auth-score-link --source-prs PR98,PR99` plus the selected expected archive SHA/bytes, expected runtime tree SHA, selected lane/job, and adjudicated JSON.",
        "7. Run public-release hygiene on the publish surface; do not include `.omx/state`, provider logs, local absolute paths, secrets, pycache, hidden files, or raw private manifests.",
        "",
        "## Candidate Snapshots",
        "",
    ]
    for record in summary["candidates"]:
        status = "ready" if record["checks"]["all_local_checks_passed"] else "blocked"
        lines.extend(
            [
                f"### PR{record['pr']} `{status}`",
                "",
                f"- archive: `{record['archive']['readiness_copy_path']}`",
                f"- archive bytes: `{record['archive']['bytes']}`",
                f"- archive sha256: `{record['archive']['sha256']}`",
                f"- sanitized runtime: `{record['runtime']['snapshot_root']}`",
                f"- runtime tree sha256: `{record['runtime']['runtime_tree_sha256']}`",
                f"- expected runtime tree sha256: `{record['runtime']['expected_runtime_tree_sha256']}`",
                f"- original T4 job: `{record['dispatch']['original_job']}`",
                f"- duplicate T4 job: `{record['dispatch']['duplicate_job']}`",
                f"- local blockers: `{record['checks']['blockers']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    generated_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    candidates: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        snapshot_root = OUT_DIR / "runtime_snapshots" / candidate.runtime.name
        archive_copy = OUT_DIR / "archives" / f"pr{candidate.pr}_archive.zip"
        copy_record = copy_sanitized_runtime(candidate.runtime, snapshot_root)
        source_record = source_manifest(snapshot_root)
        runtime_sha = runtime_tree_sha256(snapshot_root)
        archive_record = archive_manifest(candidate.archive, archive_copy)
        blockers: list[str] = []
        if archive_record["sha256"] != candidate.expected_archive_sha256:
            blockers.append("archive_sha256_mismatch")
        if archive_record["bytes"] != candidate.expected_archive_bytes:
            blockers.append("archive_size_mismatch")
        if archive_record["blockers"]:
            blockers.extend(archive_record["blockers"])
        if runtime_sha != candidate.expected_runtime_tree_sha256:
            blockers.append("runtime_tree_sha256_mismatch")
        if source_record["absolute_path_hits"]:
            blockers.append("runtime_absolute_path_hits")
        if source_record["secret_like_hits"]:
            blockers.append("runtime_secret_like_hits")
        if any(row["relative_path"].startswith(".") for row in source_record["files"]):
            blockers.append("hidden_runtime_file")

        record = {
            "schema": "final_packet_readiness_pr98_pr99_candidate_v1",
            "generated_utc": generated_utc,
            "pr": candidate.pr,
            "archive": archive_record,
            "runtime": {
                "source_root": rel(candidate.runtime),
                "snapshot_root": rel(snapshot_root),
                "expected_runtime_tree_sha256": candidate.expected_runtime_tree_sha256,
                "runtime_tree_sha256": runtime_sha,
                "copy_sanitization": copy_record,
                "manifest": source_record,
            },
            "dispatch": {
                "original_lane": candidate.original_lane,
                "duplicate_lane": candidate.duplicate_lane,
                "original_job": candidate.original_job,
                "duplicate_job": candidate.duplicate_job,
                "gpu_dispatch_performed_by_this_tool": False,
            },
            "checks": {
                "archive_sha256_matches_expected": archive_record["sha256"] == candidate.expected_archive_sha256,
                "archive_bytes_match_expected": archive_record["bytes"] == candidate.expected_archive_bytes,
                "archive_zip_has_no_blockers": not archive_record["blockers"],
                "runtime_tree_sha256_matches_expected": runtime_sha == candidate.expected_runtime_tree_sha256,
                "runtime_has_no_absolute_path_hits": not source_record["absolute_path_hits"],
                "runtime_has_no_secret_like_hits": not source_record["secret_like_hits"],
                "runtime_has_no_hidden_or_cache_files": all(
                    "__pycache__" not in row["relative_path"].split("/")
                    and not any(part.startswith(".") for part in row["relative_path"].split("/"))
                    for row in source_record["files"]
                ),
                "blockers": blockers,
                "all_local_checks_passed": not blockers,
            },
        }
        write_json(OUT_DIR / f"pr{candidate.pr}_readiness_manifest.json", record)
        candidates.append(record)

    summary = {
        "schema": "final_packet_readiness_pr98_pr99_summary_v1",
        "generated_utc": generated_utc,
        "score_claim": False,
        "remote_gpu_dispatch_performed": False,
        "output_root": rel(OUT_DIR),
        "candidates": candidates,
        "all_local_checks_passed": all(row["checks"]["all_local_checks_passed"] for row in candidates),
    }
    write_json(OUT_DIR / "readiness_summary.json", summary)
    (OUT_DIR / "promotion_checklist.md").write_text(build_checklist(summary) + "\n", encoding="utf-8")
    return 0 if summary["all_local_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
