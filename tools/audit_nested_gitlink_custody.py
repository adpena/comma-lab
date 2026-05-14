#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit dirty nested git checkouts used as local custody snapshots.

Public PR intakes and raw Kaggle ingests are often tracked as gitlinks. A
top-level `git status` line such as ` m path/to/source` is too opaque for
no-signal-loss review because the useful detail lives inside the nested repo.
This audit expands those gitlinks, reports their inner dirty paths, and applies
the same local-custody manifest used by the release index split guard.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text  # noqa: E402
from tools.audit_release_index_split import (  # noqa: E402
    IndexRecord,
    document_local_custody,
    load_local_custody_rules,
    parse_status_porcelain,
)


@dataclass(frozen=True)
class NestedGitlinkRecord:
    xy: str
    path: str
    head: str | None
    severity: str
    documented_by: str | None
    dirty_count: int
    dirty_entries: tuple[str, ...]


def _git(root: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_gitlink_paths(ls_files_stage: str) -> set[str]:
    paths: set[str] = set()
    for line in ls_files_stage.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[0] == "160000":
            paths.add(parts[3])
    return paths


def dirty_gitlink_statuses(status_text: str, gitlink_paths: set[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for xy, path in parse_status_porcelain(status_text):
        if path in gitlink_paths and xy != "  ":
            out.append((xy, path))
    return out


def nested_status(repo_root: Path, path: str) -> tuple[str | None, tuple[str, ...]]:
    full = repo_root / path
    if not full.is_dir():
        return None, (f"<missing nested gitlink directory: {path}>",)
    head_code, head_stdout, _head_stderr = _git(full, ["rev-parse", "--short", "HEAD"])
    head = head_stdout.strip() if head_code == 0 else None
    status_code, status_stdout, status_stderr = _git(
        full,
        ["status", "--short", "--untracked-files=all"],
    )
    if status_code != 0:
        detail = status_stderr.strip() or status_stdout.strip() or "nested git status failed"
        return head, (f"<nested status failed: {detail}>",)
    return head, tuple(line for line in status_stdout.splitlines() if line)


def audit_nested_gitlink_custody(
    repo_root: Path,
    *,
    local_custody_rules: list[dict[str, Any]] | None = None,
) -> list[NestedGitlinkRecord]:
    status_code, status_stdout, status_stderr = _git(repo_root, ["status", "--porcelain=v1"])
    if status_code != 0:
        raise RuntimeError(status_stderr.strip() or "git status failed")
    ls_code, ls_stdout, ls_stderr = _git(repo_root, ["ls-files", "-s"])
    if ls_code != 0:
        raise RuntimeError(ls_stderr.strip() or "git ls-files -s failed")

    gitlinks = parse_gitlink_paths(ls_stdout)
    records: list[NestedGitlinkRecord] = []
    for xy, path in dirty_gitlink_statuses(status_stdout, gitlinks):
        head, entries = nested_status(repo_root, path)
        custody = document_local_custody(
            IndexRecord(
                xy=xy,
                path=path,
                kind="unstaged_local_custody_snapshot",
                severity="warning",
                detail="nested gitlink has local inner status that must be custody-classified",
            ),
            local_custody_rules or [],
        )
        records.append(
            NestedGitlinkRecord(
                xy=xy,
                path=path,
                head=head,
                severity=custody.severity,
                documented_by=custody.documented_by,
                dirty_count=len(entries),
                dirty_entries=entries,
            )
        )
    return records


def render_payload(records: list[NestedGitlinkRecord]) -> dict[str, Any]:
    return {
        "audit": "nested_gitlink_custody",
        "ready_for_public_release_split": not any(record.severity == "warning" for record in records),
        "blockers": [
            f"{record.path}: dirty nested gitlink lacks a local-custody manifest rule"
            for record in records
            if record.severity == "warning"
        ],
        "dispatch_attempted": False,
        "score_claim": False,
        "summary": {
            "dirty_gitlink_count": len(records),
            "documented_count": sum(1 for record in records if record.severity == "info"),
            "warning_count": sum(1 for record in records if record.severity == "warning"),
        },
        "records": [asdict(record) for record in records],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-entries", type=int, default=12)
    parser.add_argument(
        "--local-custody-manifest",
        type=Path,
        help="Manifest documenting intentionally local raw/generated custody gitlinks.",
    )
    args = parser.parse_args(argv)

    rules = load_local_custody_rules(args.local_custody_manifest)
    records = audit_nested_gitlink_custody(args.repo_root.resolve(), local_custody_rules=rules)
    payload = render_payload(records)

    if args.format == "json":
        print(json_text(payload), end="")
    elif not records:
        print("nested gitlink custody: PASS (no dirty gitlinks)")
    elif payload["summary"]["warning_count"] == 0:
        print(
            "nested gitlink custody: PASS "
            f"({payload['summary']['dirty_gitlink_count']} documented dirty gitlink(s))"
        )
    else:
        print("nested gitlink custody: FAIL")
        for record in records:
            if record.severity != "warning":
                continue
            print(f"- {record.path}: {record.dirty_count} inner dirty entrie(s)")
            for entry in record.dirty_entries[: args.max_entries]:
                print(f"  {entry}")

    if args.strict and payload["summary"]["warning_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
