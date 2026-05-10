#!/usr/bin/env python3
"""Inventory duplicate helper patterns across local audit/preflight tooling.

This is a planning audit, not a style gate. It identifies repeated local
patterns that should migrate toward ``tac.audit_contract`` and ``tac.repo_io``
when those files are next touched.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.audit_contract import AuditReport, audit_exit_code  # noqa: E402
from tac.repo_io import json_text, repo_relative  # noqa: E402

DEFAULT_SCAN_ROOTS = ("tools", "scripts", "experiments", "src/tac", "src/comma_lab")
DEFAULT_EXCLUDES = (
    ".git",
    ".omx",
    "__pycache__",
    "experiments/results",
    "reverse_engineering",
    "runtime-rs/target",
)
TEXT_SUFFIXES = {".py"}


class _SourceIndexLike(Protocol):
    def read_text(
        self,
        path: str | Path,
        *,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class Pattern:
    key: str
    regex: re.Pattern[str]
    canonical_target: str
    severity: str
    candidate_substrings: tuple[str, ...]


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        key="local_sha256_helper",
        regex=re.compile(r"def _?sha256(?:_file|_path|_of)?\("),
        canonical_target="tac.repo_io.sha256_file or a byte-specific helper",
        severity="medium",
        candidate_substrings=("sha256",),
    ),
    Pattern(
        key="local_json_dump",
        regex=re.compile(r"json\.dumps\([^\\n]+indent=2,\s*sort_keys=True"),
        canonical_target="tac.repo_io.json_text/write_json",
        severity="medium",
        candidate_substrings=("json.dumps",),
    ),
    Pattern(
        key="manual_sys_path_bootstrap",
        regex=re.compile(r"sys\.path\.insert\(0,\s*str\("),
        canonical_target="canonical tool bootstrap helper",
        severity="medium",
        candidate_substrings=("sys.path.insert",),
    ),
    Pattern(
        key="manual_repo_root_parents",
        regex=re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[[0-9]+\]"),
        canonical_target="comma_lab.paths.repo_root or tool bootstrap helper",
        severity="low",
        candidate_substrings=("Path(__file__).resolve().parents",),
    ),
    Pattern(
        key="manual_audit_score_dispatch_metadata",
        regex=re.compile(r"score_claim|dispatch_attempted"),
        canonical_target="tac.audit_contract.AuditReport",
        severity="high",
        candidate_substrings=("score_claim", "dispatch_attempted"),
    ),
)


def _is_excluded(path: Path, root: Path) -> bool:
    return _is_excluded_rel(repo_relative(path, root))


def _is_excluded_rel(rel: str) -> bool:
    parts = set(Path(rel).parts)
    for item in DEFAULT_EXCLUDES:
        if "/" in item:
            item = item.rstrip("/")
            if rel == item or rel.startswith(item + "/"):
                return True
        elif item in parts:
            return True
    return False


def iter_files(repo_root: Path, scan_roots: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for rel in scan_roots:
        base = repo_root / rel
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix in TEXT_SUFFIXES and not _is_excluded(base, repo_root):
                files.append(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=True):
            dirpath_path = Path(dirpath)
            dirnames[:] = [
                dirname
                for dirname in sorted(dirnames)
                if not _is_excluded_rel(repo_relative(dirpath_path / dirname, repo_root))
            ]
            for filename in sorted(filenames):
                candidate = dirpath_path / filename
                if candidate.suffix not in TEXT_SUFFIXES:
                    continue
                if _is_excluded_rel(repo_relative(candidate, repo_root)):
                    continue
                if not candidate.is_file():
                    continue
                files.append(candidate)
    return files


def _candidate_patterns(text: str) -> tuple[Pattern, ...]:
    return tuple(
        pattern
        for pattern in PATTERNS
        if any(needle in text for needle in pattern.candidate_substrings)
    )


def _pattern_matches(pattern: Pattern, line: str) -> bool:
    if pattern.key == "manual_audit_score_dispatch_metadata":
        return "score_claim" in line or "dispatch_attempted" in line
    return bool(pattern.regex.search(line))


def audit_tooling(
    repo_root: Path,
    scan_roots: tuple[str, ...],
    *,
    source_index: _SourceIndexLike | None = None,
) -> AuditReport:
    occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    per_file_counts: dict[str, Counter[str]] = defaultdict(Counter)
    files = iter_files(repo_root, scan_roots)
    for path in files:
        rel = repo_relative(path, repo_root)
        try:
            text = (
                source_index.read_text(path, encoding="utf-8")
                if source_index is not None
                else path.read_text(encoding="utf-8")
            )
        except UnicodeDecodeError:
            continue
        active_patterns = _candidate_patterns(text)
        if not active_patterns:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in active_patterns:
                if _pattern_matches(pattern, line):
                    per_file_counts[rel][pattern.key] += 1
                    occurrences[pattern.key].append(
                        {
                            "canonical_target": pattern.canonical_target,
                            "line": lineno,
                            "path": rel,
                            "severity": pattern.severity,
                        }
                    )
    summary = {
        "file_count": len(files),
        "pattern_counts": {
            pattern.key: len(occurrences.get(pattern.key, ()))
            for pattern in PATTERNS
        },
        "affected_file_count": len(per_file_counts),
        "patterns": {
            pattern.key: {
                "canonical_target": pattern.canonical_target,
                "severity": pattern.severity,
            }
            for pattern in PATTERNS
        },
    }
    return AuditReport(
        audit="tooling_consolidation_inventory",
        readiness_key="ready_for_incremental_consolidation",
        ready=True,
        summary=summary,
        metadata={
            "occurrence_limit_per_pattern": 50,
            "occurrences": {
                key: values[:50]
                for key, values in sorted(occurrences.items())
            },
            "per_file_counts": {
                path: dict(counts)
                for path, counts in sorted(per_file_counts.items())
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--scan-root", action="append", dest="scan_roots")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    scan_roots = tuple(args.scan_roots or DEFAULT_SCAN_ROOTS)
    report = audit_tooling(args.repo_root.resolve(), scan_roots)
    payload = report.to_dict()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")
    if args.format == "json":
        print(json_text(payload), end="")
    else:
        counts = payload["summary"]["pattern_counts"]
        print(
            "tooling consolidation inventory: PASS "
            f"({payload['summary']['file_count']} files scanned)"
        )
        for key, count in counts.items():
            print(f"  - {key}: {count}")
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
