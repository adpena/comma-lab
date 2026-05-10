#!/usr/bin/env python3
"""Certify a grammar-preserving HNeRV packet transform candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hnerv_packet_sections import PARSER_CHOICES, PARSER_PR106  # noqa: E402
from tac.packet_section_transform import (  # noqa: E402
    certify_hnerv_grammar_preserving_candidate_pair,
)
from tac.repo_io import json_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default=PARSER_PR106,
        help=f"Parser-section grammar to certify. Default: {PARSER_PR106}",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--fail-if-blocked",
        action="store_true",
        help="Exit nonzero if the candidate is not grammar-preserving archive-preflight evidence.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = certify_hnerv_grammar_preserving_candidate_pair(
        source_archive=args.source_archive,
        candidate_archive=args.candidate_archive,
        label=args.label,
        parser=args.parser,
        repo_root=REPO_ROOT,
    )
    text = json_text(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 1 if args.fail_if_blocked and not result["ready_for_archive_preflight"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
