#!/usr/bin/env python3
"""Audit Paradigm-alpha mask codec readiness against the live checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.paradigm_alpha_readiness import (  # noqa: E402
    audit_paradigm_alpha,
    readiness_payload,
    render_markdown,
)
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
        "--output",
        type=Path,
        help="Write output to this path instead of stdout.",
    )
    parser.add_argument(
        "--fail-if-missing-core",
        action="store_true",
        help="Exit 2 when any candidate is missing required core code/tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = audit_paradigm_alpha(repo_root=args.repo_root)
    text = (
        json_text(readiness_payload(repo_root=args.repo_root))
        if args.format == "json"
        else render_markdown(rows)
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    if args.fail_if_missing_core and any(row.missing_core_components for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
