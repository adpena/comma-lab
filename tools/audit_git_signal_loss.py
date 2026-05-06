#!/usr/bin/env python3
"""Audit Git history for canonical files missing from the current checkout.

This is a recovery helper, not a cleanup tool. It scans historical Git paths,
filters to source/research surfaces that should normally survive OSS
canonicalization, and reports canonical files that existed in history but are
absent from the current working tree or HEAD.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_PREFIXES = (
    "src/tac/",
    "tools/",
    "scripts/",
    "docs/",
    ".omx/research/",
    "submissions/robust_current/",
    ".github/workflows/",
)
ROOT_CANONICAL_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}
EXPERIMENTS_CANONICAL_SUFFIXES = {
    ".py",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".md",
    ".json",
    ".toml",
    ".sh",
}
GENERATED_OR_BULKY_PARTS = (
    "/__pycache__/",
    "/.pytest_cache/",
    "/.ruff_cache/",
    "/.mypy_cache/",
    "/node_modules/",
    "/dist/",
    "/build/",
    "/.cache/",
)
GENERATED_OR_BULKY_PREFIXES = (
    "experiments/results/",
    "reports/raw/",
    "reports/private/",
    ".omx/state/",
    ".omx/tmp/",
)
GENERATED_OR_BULKY_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".pt",
    ".pth",
    ".safetensors",
    ".onnx",
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".npy",
    ".npz",
    ".bin",
    ".br",
    ".qma9",
    ".qmb1",
    ".qmc1",
    ".qmh1",
    ".qma9cb",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".metadata",
)


@dataclass(frozen=True)
class LatestCommit:
    commit: str
    date: str
    subject: str


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def _normalize(path: str) -> str:
    return path.strip().lstrip("./")


def _is_generated_or_bulky(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in GENERATED_OR_BULKY_PREFIXES):
        return True
    marked = f"/{path}"
    if any(part in marked for part in GENERATED_OR_BULKY_PARTS):
        return True
    return path.endswith(GENERATED_OR_BULKY_SUFFIXES)


def is_canonical_path(path: str) -> bool:
    path = _normalize(path)
    if not path or _is_generated_or_bulky(path):
        return False
    if path in ROOT_CANONICAL_FILES:
        return True
    if path.startswith(CANONICAL_PREFIXES):
        return True
    if path.startswith("experiments/") and "/" not in path.removeprefix("experiments/"):
        return Path(path).suffix in EXPERIMENTS_CANONICAL_SUFFIXES
    return False


def historical_canonical_paths() -> set[str]:
    output = _git(["log", "--all", "--full-history", "--no-renames", "--name-only", "--pretty=format:"])
    return {path for raw in output.splitlines() if is_canonical_path(path := _normalize(raw))}


def current_paths(*, mode: str) -> set[str]:
    if mode == "head":
        return {_normalize(path) for path in _git(["ls-tree", "-r", "--name-only", "HEAD"]).splitlines()}
    if mode == "worktree":
        output = _git(["ls-files", "--cached", "--others", "--exclude-standard"])
        return {_normalize(path) for path in output.splitlines()}
    raise ValueError(f"unknown comparison mode: {mode}")


def latest_commit_for_path(path: str) -> LatestCommit | None:
    output = _git(
        ["log", "--all", "--full-history", "--no-renames", "-n", "1", "--format=%H%x09%cs%x09%s", "--", path]
    ).strip()
    if not output:
        return None
    commit, date, subject = output.split("\t", 2)
    return LatestCommit(commit=commit, date=date, subject=subject)


def latest_commits_for_paths(paths: Iterable[str]) -> dict[str, LatestCommit]:
    """Return newest reachable commit evidence for each path.

    Path-limited ``git log`` can miss paths after aggressive filter/cleanup
    history. A commit-tree scan is slower but reliable for a recovery audit and
    avoids false ``latest_commit=null`` on precisely the files we care about.
    """

    unresolved = set(paths)
    if not unresolved:
        return {}
    latest: dict[str, LatestCommit] = {}
    commit_rows = _git(["log", "--all", "--date-order", "--format=%H%x09%cs%x09%s"]).splitlines()
    for row in commit_rows:
        if not unresolved:
            break
        commit, date, subject = row.split("\t", 2)
        try:
            tree_paths = set(_normalize(path) for path in _git(["ls-tree", "-r", "--name-only", commit]).splitlines())
        except subprocess.CalledProcessError:
            continue
        for path in sorted(unresolved.intersection(tree_paths)):
            latest[path] = LatestCommit(commit=commit, date=date, subject=subject)
            unresolved.remove(path)
    return latest


def classify_path(path: str) -> str:
    if path.startswith(("src/tac/", "tools/", "scripts/", "experiments/", "submissions/robust_current/")):
        return "source_or_tooling"
    if path.startswith(".omx/research/"):
        return "research_ledger"
    if path.startswith("docs/"):
        return "documentation"
    if path.startswith(".github/workflows/"):
        return "ci"
    return "root_control"


def audit_signal_loss(*, mode: str, max_paths: int | None = None) -> dict[str, Any]:
    historical = historical_canonical_paths()
    current = current_paths(mode=mode)
    missing = sorted(historical - current)
    if max_paths is not None:
        missing_for_detail = missing[:max_paths]
    else:
        missing_for_detail = missing

    latest_by_path = latest_commits_for_paths(missing_for_detail)
    rows = []
    for path in missing_for_detail:
        latest = latest_by_path.get(path) or latest_commit_for_path(path)
        rows.append(
            {
                "path": path,
                "class": classify_path(path),
                "latest_commit": None if latest is None else latest.__dict__,
            }
        )

    by_class: dict[str, int] = {}
    for path in missing:
        by_class[classify_path(path)] = by_class.get(classify_path(path), 0) + 1

    return {
        "schema": "git_signal_loss_audit_v1",
        "comparison_mode": mode,
        "historical_canonical_paths": len(historical),
        "current_paths": len(current),
        "missing_canonical_paths": len(missing),
        "missing_by_class": dict(sorted(by_class.items())),
        "detail_truncated": max_paths is not None and len(missing) > max_paths,
        "detail_count": len(rows),
        "missing": rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Git Signal-Loss Audit",
        "",
        f"- comparison_mode: `{report['comparison_mode']}`",
        f"- historical_canonical_paths: `{report['historical_canonical_paths']}`",
        f"- current_paths: `{report['current_paths']}`",
        f"- missing_canonical_paths: `{report['missing_canonical_paths']}`",
        "",
        "## Missing By Class",
        "",
    ]
    for cls, count in report["missing_by_class"].items():
        lines.append(f"- `{cls}`: `{count}`")
    lines.extend(["", "## Missing Paths", ""])
    for row in report["missing"]:
        latest = row["latest_commit"] or {}
        lines.append(
            f"- `{row['path']}` [{row['class']}] "
            f"{latest.get('commit', '')[:12]} {latest.get('date', '')} {latest.get('subject', '')}".rstrip()
        )
    if report["detail_truncated"]:
        lines.append("")
        lines.append("- Detail truncated; rerun without `--max-paths` for the full list.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worktree", "head"), default="worktree")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--max-paths", type=int)
    parser.add_argument("--format", choices=("json", "summary"), default="summary")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = audit_signal_loss(mode=args.mode, max_paths=args.max_paths)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out:
        _write_markdown(args.md_out, report)

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"missing={report['missing_canonical_paths']} "
            f"historical={report['historical_canonical_paths']} "
            f"mode={report['comparison_mode']} "
            f"by_class={report['missing_by_class']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
