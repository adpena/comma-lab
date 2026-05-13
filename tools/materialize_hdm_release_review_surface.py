#!/usr/bin/env python3
"""Materialize an HDM exact-eval release-review surface.

The HDM3/HDM4 archive builder can create a static wrapper surface, but final
pre-submission review must preserve the runtime files that were scored by exact
auth eval. This tool copies those runtime files unchanged, installs the scored
archive plus auth-eval JSON, and writes deterministic report/archive manifests.
It does not run scorers and does not mutate the source runtime tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file, write_json  # noqa: E402

SCHEMA = "hdm_release_review_surface_manifest_v1"
TOOL = "tools/materialize_hdm_release_review_surface.py"
DEFAULT_SOURCE_RUNTIME = Path("submissions/pr106_latent_sidecar_r2_pr101_grammar")
RUNTIME_CUSTODY_FILENAMES = {
    "archive_manifest.json",
    "contest_auth_eval.json",
    "report.txt",
}
RUNTIME_CUSTODY_PREFIXES = ("pre_submission_compliance.",)
RELEASE_REVIEW_MANIFEST = "pre_submission_compliance.release_review_manifest.json"
COPY_RUNTIME_SUFFIXES = {".py", ".sh", ".json", ".txt", ".toml", ".env", ".c", ".cc", ".cpp", ".h", ".hpp"}
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


class HdmReleaseReviewSurfaceError(ValueError):
    """Raised when an HDM release-review surface cannot be materialized."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _copy_file(src: Path, dst: Path, *, mode: int | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if mode is None:
        mode = stat.S_IMODE(src.stat().st_mode)
    os.chmod(dst, mode)


def _is_custody_path(rel: str) -> bool:
    name = Path(rel).name
    return name in RUNTIME_CUSTODY_FILENAMES or any(
        name.startswith(prefix) for prefix in RUNTIME_CUSTODY_PREFIXES
    )


def _runtime_source_files(source_runtime_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source_runtime_dir.rglob("*"), key=lambda item: item.relative_to(source_runtime_dir).as_posix()):
        if not path.is_file():
            continue
        rel = path.relative_to(source_runtime_dir)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if _is_custody_path(rel.as_posix()):
            continue
        if path.suffix.lower() not in COPY_RUNTIME_SUFFIXES:
            continue
        files.append(path)
    return files


def _auth_runtime_rows(auth_eval: Mapping[str, Any]) -> list[dict[str, Any]]:
    provenance = auth_eval.get("provenance")
    runtime = provenance.get("inflate_runtime_manifest") if isinstance(provenance, Mapping) else None
    rows = runtime.get("files") if isinstance(runtime, Mapping) else None
    if not isinstance(rows, list):
        raise HdmReleaseReviewSurfaceError("auth-eval runtime file manifest missing")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rel = row.get("relative_path")
        sha = row.get("sha256")
        byte_count = row.get("bytes")
        if not isinstance(rel, str) or not isinstance(sha, str):
            continue
        if _is_custody_path(rel):
            continue
        normalized.append(
            {
                "relative_path": rel,
                "bytes": int(byte_count),
                "sha256": sha,
            }
        )
    if not normalized:
        raise HdmReleaseReviewSurfaceError("auth-eval runtime manifest has no non-custody files")
    return sorted(normalized, key=lambda row: row["relative_path"])


def _copy_and_verify_runtime(
    *,
    source_runtime_dir: Path,
    output_dir: Path,
    auth_eval: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = {row["relative_path"]: row for row in _auth_runtime_rows(auth_eval)}
    copied: list[dict[str, Any]] = []
    for src in _runtime_source_files(source_runtime_dir):
        rel = src.relative_to(source_runtime_dir).as_posix()
        if rel not in expected:
            continue
        dst = output_dir / rel
        mode = 0o755 if rel == "inflate.sh" else stat.S_IMODE(src.stat().st_mode)
        _copy_file(src, dst, mode=mode)
        actual = {
            "relative_path": rel,
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
            "mode": f"{stat.S_IMODE(dst.stat().st_mode):04o}",
        }
        exp = expected[rel]
        if actual["bytes"] != exp["bytes"] or actual["sha256"] != exp["sha256"]:
            raise HdmReleaseReviewSurfaceError(
                f"runtime file mismatch after copy: {rel} "
                f"expected {exp['bytes']} {exp['sha256']} got {actual['bytes']} {actual['sha256']}"
            )
        copied.append(actual)
    copied_rels = {row["relative_path"] for row in copied}
    missing = sorted(set(expected) - copied_rels)
    if missing:
        raise HdmReleaseReviewSurfaceError(
            "source runtime is missing exact-CUDA runtime files: " + ", ".join(missing)
        )
    return copied


def _archive_member_rows(archive: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            rows.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": int(info.CRC),
                    "sha256": _bytes_sha256(zf.read(info)),
                }
            )
    return rows


def _bytes_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _score_summary(auth_eval: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "lane_tag",
        "score_axis",
        "final_score",
        "canonical_score",
        "avg_segnet_dist",
        "avg_posenet_dist",
        "archive_size_bytes",
        "n_samples",
        "evidence_grade",
        "exact_cuda_eval_complete",
        "score_claim_valid",
    )
    return {key: auth_eval.get(key) for key in keys if key in auth_eval}


def _runtime_tree_summary(auth_eval: Mapping[str, Any]) -> dict[str, Any]:
    provenance = auth_eval.get("provenance")
    runtime = provenance.get("inflate_runtime_manifest") if isinstance(provenance, Mapping) else None
    if not isinstance(runtime, Mapping):
        return {}
    return {
        "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": runtime.get("runtime_content_tree_sha256"),
        "runtime_file_count": runtime.get("runtime_file_count"),
    }


def _write_report(
    path: Path,
    *,
    candidate_label: str,
    lane_id: str,
    job_id: str,
    archive: dict[str, Any],
    auth_eval_path: Path,
    score_summary: Mapping[str, Any],
    runtime_tree: Mapping[str, Any],
) -> None:
    lines = [
        f"{candidate_label} exact-CUDA release-review surface",
        "",
        f"lane_id            : {lane_id}",
        f"job_id             : {job_id}",
        f"archive_sha256     : {archive['archive_sha256']}",
        f"archive_size_bytes : {archive['archive_size_bytes']}",
        f"member_count       : {archive['member_count']}",
        f"runtime_tree_sha256: {runtime_tree.get('runtime_tree_sha256')}",
        f"auth_eval_json     : {_repo_rel(auth_eval_path)}",
        "",
        "Custody",
        "=======",
        "This surface copies runtime files byte-for-byte from the source runtime",
        "and verifies them against the exact-CUDA auth-eval runtime manifest.",
        "archive_manifest.json, contest_auth_eval.json, report.txt, and",
        "pre_submission_compliance.* are custody files and are excluded from",
        "the pre-submission runtime-match hash by the compliance checker.",
        "",
        "Empirical [contest-CUDA] anchor",
        "===============================",
    ]
    for key in sorted(score_summary):
        lines.append(f"{key}: {score_summary[key]}")
    lines.extend(
        [
            "",
            "Submission-readiness status",
            "===========================",
            "Not asserted by this report alone. Submission readiness requires",
            "scripts/pre_submission_compliance_check.py --contest-final --strict",
            "to pass against this surface.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def materialize_release_review_surface(
    *,
    archive: Path,
    auth_eval_json: Path,
    source_runtime_dir: Path,
    output_dir: Path,
    lane_id: str,
    job_id: str,
    candidate_label: str,
    force: bool = False,
) -> dict[str, Any]:
    archive = _repo_path(archive)
    auth_eval_json = _repo_path(auth_eval_json)
    source_runtime_dir = _repo_path(source_runtime_dir)
    output_dir = _repo_path(output_dir)
    if not archive.is_file():
        raise FileNotFoundError(f"archive missing: {archive}")
    if not auth_eval_json.is_file():
        raise FileNotFoundError(f"auth-eval JSON missing: {auth_eval_json}")
    if not source_runtime_dir.is_dir():
        raise FileNotFoundError(f"source runtime dir missing: {source_runtime_dir}")
    if output_dir.exists():
        if not force:
            raise HdmReleaseReviewSurfaceError(f"output exists; pass --force: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    auth_eval = read_json(auth_eval_json)
    if not isinstance(auth_eval, Mapping):
        raise HdmReleaseReviewSurfaceError("auth-eval JSON must be an object")
    runtime_files = _copy_and_verify_runtime(
        source_runtime_dir=source_runtime_dir,
        output_dir=output_dir,
        auth_eval=auth_eval,
    )

    archive_dst = output_dir / "archive.zip"
    _copy_file(archive, archive_dst, mode=0o644)
    auth_dst = output_dir / "contest_auth_eval.json"
    _copy_file(auth_eval_json, auth_dst, mode=0o644)
    archive_sha = sha256_file(archive_dst)
    archive_bytes = archive_dst.stat().st_size
    expected_sha = auth_eval.get("provenance", {}).get("archive_sha256")
    expected_bytes = auth_eval.get("archive_size_bytes") or auth_eval.get("provenance", {}).get(
        "archive_size_bytes"
    )
    if expected_sha != archive_sha:
        raise HdmReleaseReviewSurfaceError(
            f"archive SHA mismatch with auth eval: expected={expected_sha} actual={archive_sha}"
        )
    if int(expected_bytes) != archive_bytes:
        raise HdmReleaseReviewSurfaceError(
            f"archive byte mismatch with auth eval: expected={expected_bytes} actual={archive_bytes}"
        )

    members = _archive_member_rows(archive_dst)
    archive_manifest = {
        "schema": "pre_submission_archive_manifest_v1",
        "tool": TOOL,
        "candidate_label": candidate_label,
        "lane_id": lane_id,
        "job_id": job_id,
        "archive_path": _repo_rel(archive_dst),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "member_count": len(members),
        "members": members,
        "score_claim": False,
        "score_claim_basis": "release-review custody only; score evidence remains contest_auth_eval.json",
    }
    write_json(output_dir / "archive_manifest.json", archive_manifest)

    runtime_tree = _runtime_tree_summary(auth_eval)
    report_path = output_dir / "report.txt"
    _write_report(
        report_path,
        candidate_label=candidate_label,
        lane_id=lane_id,
        job_id=job_id,
        archive=archive_manifest,
        auth_eval_path=auth_dst,
        score_summary=_score_summary(auth_eval),
        runtime_tree=runtime_tree,
    )

    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_label": candidate_label,
        "lane_id": lane_id,
        "job_id": job_id,
        "submission_dir": _repo_rel(output_dir),
        "source_runtime_dir": _repo_rel(source_runtime_dir),
        "archive": archive_manifest,
        "auth_eval_json": {
            "source_path": _repo_rel(auth_eval_json),
            "path": _repo_rel(auth_dst),
            "sha256": sha256_file(auth_dst),
            "bytes": auth_dst.stat().st_size,
        },
        "runtime_tree": runtime_tree,
        "runtime_files_verified_against_auth_eval": runtime_files,
        "report": {
            "path": _repo_rel(report_path),
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
        },
        "compliance_command": [
            ".venv/bin/python",
            "scripts/pre_submission_compliance_check.py",
            "--submission-dir",
            _repo_rel(output_dir),
            "--archive",
            _repo_rel(archive_dst),
            "--auth-eval-json",
            _repo_rel(auth_dst),
            "--require-auth-eval",
            "--require-t4-equivalent",
            "--require-submission-runtime-match",
            "--contest-final",
            "--expect-single-member",
            members[0]["name"] if len(members) == 1 else "<member>",
            "--expected-archive-sha256",
            archive_sha,
            "--expected-archive-size-bytes",
            str(archive_bytes),
            "--expected-runtime-tree-sha256",
            str(runtime_tree.get("runtime_tree_sha256") or ""),
            "--expected-lane-id",
            lane_id,
            "--expected-job-id",
            job_id,
            "--public-scan-path",
            _repo_rel(archive_dst),
            "--public-scan-path",
            _repo_rel(output_dir / "inflate.sh"),
            "--public-scan-path",
            _repo_rel(output_dir / "inflate.py"),
            "--public-scan-path",
            _repo_rel(output_dir / "src"),
            "--public-scan-path",
            _repo_rel(report_path),
            "--public-scan-path",
            _repo_rel(output_dir / "archive_manifest.json"),
            "--json-out",
            _repo_rel(output_dir / "pre_submission_compliance.contest_final.json"),
            "--strict",
        ],
    }
    write_json(output_dir / RELEASE_REVIEW_MANIFEST, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--auth-eval-json", required=True, type=Path)
    parser.add_argument("--source-runtime-dir", type=Path, default=DEFAULT_SOURCE_RUNTIME)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--candidate-label", default="HDM exact-CUDA candidate")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_release_review_surface(
        archive=args.archive,
        auth_eval_json=args.auth_eval_json,
        source_runtime_dir=args.source_runtime_dir,
        output_dir=args.output_dir,
        lane_id=args.lane_id,
        job_id=args.job_id,
        candidate_label=args.candidate_label,
        force=args.force,
    )
    print(
        json_text(
            {
                "schema": "hdm_release_review_surface_stdout_v1",
                "submission_dir": manifest["submission_dir"],
                "archive_sha256": manifest["archive"]["archive_sha256"],
                "archive_size_bytes": manifest["archive"]["archive_size_bytes"],
                "runtime_tree_sha256": manifest["runtime_tree"].get("runtime_tree_sha256"),
                "compliance_command": manifest["compliance_command"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
