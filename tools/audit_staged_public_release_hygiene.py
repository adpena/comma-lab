#!/usr/bin/env python3
"""Audit staged public-release surfaces for private operator leakage.

This intentionally scans only files staged for commit under public-facing
paths. Broader forensic ledgers and generated custody surfaces may preserve
private paths locally, but staged docs/site/readme surfaces must be clean before
OSS publication.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.preflight import public_release_hygiene_violations_for_text  # noqa: E402
from tac.repo_io import json_text  # noqa: E402

PUBLIC_EXACT_PATHS = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "report.txt",
    "reports/silent_defaults.md",
    "reverse_engineering/README.md",
    "reverse_engineering/pr95_hnerv/README.md",
}

PUBLIC_PREFIXES = (
    ".github/",
    "docs/",
    "reports/graphs/",
    "reports/site/",
    "submissions/apogee/",
)


def is_public_release_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized in PUBLIC_EXACT_PATHS or normalized.startswith(PUBLIC_PREFIXES)


def select_staged_public_paths(paths: list[str]) -> list[str]:
    return sorted(path for path in paths if is_public_release_path(path))


def staged_paths(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff --cached failed: {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def staged_blob_text(root: Path, path: str) -> str | None:
    """Return staged blob text for ``path``.

    This deliberately reads ``:path`` from the git index instead of the
    worktree. A dirty public file can otherwise pass because the worktree was
    cleaned while the staged blob still contains private operator state.
    """
    proc = subprocess.run(
        ["git", "show", f":{path}"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        candidate = root / path
        if not candidate.is_file():
            return None
        try:
            return candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
    return proc.stdout


def audit_public_staged_hygiene(root: Path, paths: list[str]) -> dict[str, object]:
    scan_paths = select_staged_public_paths(paths)
    violations: list[str] = []
    for path in scan_paths:
        text = staged_blob_text(root, path)
        if text is None:
            continue
        violations.extend(public_release_hygiene_violations_for_text(path, text))
    return {
        "staged_path_count": len(paths),
        "public_scan_path_count": len(scan_paths),
        "public_scan_paths": scan_paths,
        "violation_count": len(violations),
        "violations": violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    payload = audit_public_staged_hygiene(root, staged_paths(root))
    if args.format == "json":
        print(json_text(payload), end="")
    elif payload["violation_count"]:
        print(
            "staged public release hygiene: FAIL "
            f"({payload['violation_count']} violation(s) across "
            f"{payload['public_scan_path_count']} staged public file(s))"
        )
        for violation in payload["violations"][:20]:
            print(f"  - {violation}")
        remaining = int(payload["violation_count"]) - 20
        if remaining > 0:
            print(f"  - ... {remaining} more")
    else:
        print(
            "staged public release hygiene: PASS "
            f"({payload['public_scan_path_count']} staged public file(s) scanned)"
        )

    if args.strict and payload["violation_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
