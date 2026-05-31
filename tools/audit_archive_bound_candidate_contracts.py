#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit archive-like candidate artifacts for shared runtime-bridge contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.archive_bound_candidate_contract_audit import (  # noqa: E402
    audit_archive_bound_candidate_contracts,
    format_archive_bound_candidate_contract_audit,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        action="append",
        dest="paths",
        help=(
            "Artifact file or directory to scan. Defaults to .omx/research and "
            "experiments/results."
        ),
    )
    parser.add_argument(
        "--include-markdown",
        action="store_true",
        help="Also scan JSON fenced blocks in Markdown memos.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Cap scanned files after deterministic path ordering.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=4_000_000,
        help="Skip individual files larger than this many bytes.",
    )
    parser.add_argument(
        "--fail-on-migration-required",
        action="store_true",
        help="Exit nonzero when archive-like rows still lack the shared contract.",
    )
    parser.add_argument(
        "--fail-on-advisory",
        action="store_true",
        help="Exit nonzero on advisory parse/read/prose findings.",
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Scan only files tracked by git under the selected paths.",
    )
    parser.add_argument("--output-json", help="Write the audit payload to a file.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--limit", type=int, default=12, help="Text finding limit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.paths or [".omx/research", "experiments/results"]
    result = audit_archive_bound_candidate_contracts(
        [Path(path) for path in paths],
        repo_root=REPO_ROOT,
        include_markdown=args.include_markdown,
        max_files=args.max_files,
        max_file_bytes=args.max_file_bytes,
        tracked_only=args.tracked_only,
    )
    payload = result.as_dict()
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_archive_bound_candidate_contract_audit(result, limit=args.limit))

    ok = result.passed
    if args.fail_on_migration_required and result.migration_required_findings:
        ok = False
    if args.fail_on_advisory and result.advisory_findings:
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
