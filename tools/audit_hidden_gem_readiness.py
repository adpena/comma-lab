#!/usr/bin/env python3
"""Audit hidden-gem registry rows against live repo files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_path()

from tac.hidden_gem_readiness import audit_hidden_gems, readiness_payload, render_markdown  # noqa: E402
from tac.hidden_gems import CATEGORIES, STATUSES, all_hidden_gems  # noqa: E402
from tac.repo_io import json_text  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to audit. Default: this checkout.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Default: markdown.",
    )
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORIES),
        help="Filter to one hidden-gem category.",
    )
    parser.add_argument(
        "--status",
        choices=sorted(STATUSES),
        help="Filter to one registry status.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this path instead of stdout.",
    )
    parser.add_argument(
        "--fail-if-missing-targets",
        action="store_true",
        help="Exit 2 when any selected row points at a missing integration target.",
    )
    parser.add_argument(
        "--fail-if-missing-evidence",
        action="store_true",
        help="Exit 2 when any selected row points at missing evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = all_hidden_gems(category=args.category, status=args.status)
    if args.format == "json":
        payload = readiness_payload(repo_root=args.repo_root, entries=entries)
        text = json_text(payload)
        missing_targets = payload["summary"]["missing_integration_target_count"]
        missing_evidence = payload["summary"]["missing_evidence_path_count"]
    else:
        rows = audit_hidden_gems(repo_root=args.repo_root, entries=entries)
        text = render_markdown(rows)
        missing_targets = sum(len(row.missing_integration_targets) for row in rows)
        missing_evidence = sum(len(row.missing_evidence_paths) for row in rows)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if args.fail_if_missing_targets and missing_targets:
        return 2
    if args.fail_if_missing_evidence and missing_evidence:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
