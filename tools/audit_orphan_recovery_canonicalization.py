#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit staged orphan-recovery deletions for canonical tracked replacements.

The pyc recovery tree is allowed to shrink only after recovered source has
landed in its canonical repo path. This audit turns that rule into an
executable guard: source-like deletions under the orphan-recovery root must be
staged, and each deleted copy must map to a tracked canonical path after
stripping the recovery prefix.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, repo_relative  # noqa: E402
from tools.audit_untracked_source_artifacts import _is_source_like_path  # noqa: E402

ORPHAN_ROOT = "reverse_engineering/orphan_pyc_recovery_20260505_codex"


@dataclass(frozen=True)
class DeletionRecord:
    status: str
    path: str
    canonical_path: str | None
    canonical_tracked: bool
    staged_delete: bool


@dataclass(frozen=True)
class ShadowedModifiedRecord:
    status: str
    path: str
    canonical_path: str | None
    canonical_tracked: bool


def parse_git_status_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line:
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if path:
            records.append((status, path.replace("\\", "/")))
    return records


def canonical_path_for_orphan(path: str, *, orphan_root: str = ORPHAN_ROOT) -> str | None:
    prefix = orphan_root.rstrip("/") + "/"
    if not path.startswith(prefix):
        return None
    return path.removeprefix(prefix)


def _git_status(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "git status failed")
    return proc.stdout


def _tracked_files(repo_root: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        stdout = proc.stdout.decode(errors="replace").strip()
        raise RuntimeError(stderr or stdout or "git ls-files failed")
    return {path for path in proc.stdout.decode().split("\0") if path}


def build_deletion_records(
    status_records: list[tuple[str, str]],
    *,
    tracked_files: set[str],
    orphan_root: str = ORPHAN_ROOT,
) -> list[DeletionRecord]:
    deletion_records: list[DeletionRecord] = []
    for status, path in status_records:
        if "D" not in status or not _is_source_like_path(path):
            continue
        canonical = canonical_path_for_orphan(path, orphan_root=orphan_root)
        deletion_records.append(
            DeletionRecord(
                status=status,
                path=path,
                canonical_path=canonical,
                canonical_tracked=canonical in tracked_files if canonical is not None else False,
                staged_delete=status[0] == "D" and status[1] == " ",
            )
        )
    return deletion_records


def build_shadowed_modified_records(
    status_records: list[tuple[str, str]],
    *,
    tracked_files: set[str],
    orphan_root: str = ORPHAN_ROOT,
) -> list[ShadowedModifiedRecord]:
    modified_records: list[ShadowedModifiedRecord] = []
    for status, path in status_records:
        if "D" in status or not any(ch in status for ch in ("M", "A", "R", "C", "T", "U")):
            continue
        if not _is_source_like_path(path):
            continue
        canonical = canonical_path_for_orphan(path, orphan_root=orphan_root)
        if canonical is None:
            continue
        modified_records.append(
            ShadowedModifiedRecord(
                status=status,
                path=path,
                canonical_path=canonical,
                canonical_tracked=canonical in tracked_files,
            )
        )
    return modified_records


def audit_orphan_recovery_canonicalization(
    repo_root: Path,
    *,
    fail_on_shadowed_modified: bool = False,
) -> AuditReport:
    tracked = _tracked_files(repo_root)
    status_records = parse_git_status_records(_git_status(repo_root))
    records = build_deletion_records(status_records, tracked_files=tracked)
    shadowed_modified = build_shadowed_modified_records(status_records, tracked_files=tracked)
    blockers: list[str] = []
    for record in records:
        if record.canonical_path is None:
            blockers.append(f"{record.path}: source-like deletion is outside {ORPHAN_ROOT}")
            continue
        if not record.staged_delete:
            blockers.append(f"{record.path}: deletion is not staged cleanly (status={record.status!r})")
        if not record.canonical_tracked:
            blockers.append(
                f"{record.path}: canonical path is not tracked: {record.canonical_path}"
            )
    if fail_on_shadowed_modified:
        for record in shadowed_modified:
            if record.canonical_tracked:
                blockers.append(
                    f"{record.path}: modified orphan copy shadows tracked canonical path "
                    f"{record.canonical_path}; canonicalize or delete the duplicate"
                )
            else:
                blockers.append(
                    f"{record.path}: modified orphan copy has no tracked canonical path "
                    f"{record.canonical_path}"
                )

    canonicalized = [record for record in records if record.canonical_path is not None and record.canonical_tracked]
    return AuditReport(
        audit="orphan_recovery_canonicalization",
        readiness_key="ready_for_orphan_recovery_cleanup",
        ready=not blockers,
        blockers=tuple(blockers),
        summary={
            "orphan_root": ORPHAN_ROOT,
            "source_like_delete_count": len(records),
            "canonicalized_duplicate_delete_count": len(canonicalized),
            "non_orphan_delete_count": sum(1 for record in records if record.canonical_path is None),
            "unstaged_delete_count": sum(1 for record in records if not record.staged_delete),
            "missing_canonical_count": sum(
                1
                for record in records
                if record.canonical_path is not None and not record.canonical_tracked
            ),
            "source_like_modified_count": len(shadowed_modified),
            "shadowed_modified_count": sum(
                1 for record in shadowed_modified if record.canonical_tracked
            ),
            "modified_missing_canonical_count": sum(
                1 for record in shadowed_modified if not record.canonical_tracked
            ),
            "deleted_paths": [
                {
                    "path": record.path,
                    "canonical_path": record.canonical_path,
                    "canonical_tracked": record.canonical_tracked,
                    "status": record.status,
                }
                for record in records
            ],
            "modified_paths": [
                {
                    "path": record.path,
                    "canonical_path": record.canonical_path,
                    "canonical_tracked": record.canonical_tracked,
                    "status": record.status,
                }
                for record in shadowed_modified
            ],
        },
        metadata={"repo_root": repo_relative(repo_root, repo_root)},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="Accepted for all-lanes preflight symmetry.")
    parser.add_argument(
        "--fail-on-shadowed-modified",
        action="store_true",
        help=(
            "Fail when a modified source-like orphan copy shadows a tracked "
            "canonical path. Default is advisory inventory."
        ),
    )
    args = parser.parse_args(argv)

    report = audit_orphan_recovery_canonicalization(
        args.repo_root.resolve(),
        fail_on_shadowed_modified=args.fail_on_shadowed_modified,
    )
    payload = report.to_dict()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.format == "json":
        print(json_text(payload), end="")
    else:
        detail = (
            f"({report.summary['canonicalized_duplicate_delete_count']} duplicate delete(s) checked; "
            f"{report.summary['shadowed_modified_count']} shadowed modified orphan copy/copies; "
            f"{report.summary['modified_missing_canonical_count']} modified missing canonical)"
        )
        print(report.render_text(pass_detail=detail))
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
