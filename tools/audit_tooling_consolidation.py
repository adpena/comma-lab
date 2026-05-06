#!/usr/bin/env python3
"""Inventory duplicate helper patterns across local audit/preflight tooling.

This is a planning audit, not a style gate. It identifies repeated local
patterns that should migrate toward ``tac.audit_contract`` and ``tac.repo_io``
when those files are next touched.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tac.audit_contract import AuditReport, audit_exit_code
from tac.repo_io import json_text, repo_relative

REPO_ROOT = Path(__file__).resolve().parents[1]
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


@dataclass(frozen=True)
class Pattern:
    key: str
    regex: re.Pattern[str]
    canonical_target: str
    severity: str


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        key="local_sha256_helper",
        regex=re.compile(r"def _?sha256(?:_file|_path|_of)?\("),
        canonical_target="tac.repo_io.sha256_file or a byte-specific helper",
        severity="medium",
    ),
    Pattern(
        key="local_json_dump",
        regex=re.compile(r"json\.dumps\([^\\n]+indent=2,\s*sort_keys=True"),
        canonical_target="tac.repo_io.json_text/write_json",
        severity="medium",
    ),
    Pattern(
        key="manual_sys_path_bootstrap",
        regex=re.compile(r"sys\.path\.insert\(0,\s*str\("),
        canonical_target="canonical tool bootstrap helper",
        severity="medium",
    ),
    Pattern(
        key="manual_repo_root_parents",
        regex=re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[[0-9]+\]"),
        canonical_target="comma_lab.paths.repo_root or tool bootstrap helper",
        severity="low",
    ),
    Pattern(
        key="manual_audit_score_dispatch_metadata",
        regex=re.compile(r"score_claim|dispatch_attempted"),
        canonical_target="tac.audit_contract.AuditReport",
        severity="high",
    ),
)


def _is_excluded(path: Path, root: Path) -> bool:
    rel = repo_relative(path, root)
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
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix not in TEXT_SUFFIXES:
                continue
            if _is_excluded(candidate, repo_root):
                continue
            files.append(candidate)
    return files


def audit_tooling(repo_root: Path, scan_roots: tuple[str, ...]) -> AuditReport:
    occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    files = iter_files(repo_root, scan_roots)
    for path in files:
        rel = repo_relative(path, repo_root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                if pattern.regex.search(line):
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
            "occurrences": {
                key: values[:50]
                for key, values in sorted(occurrences.items())
            }
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
