#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile outer SNAR1 header grammar overhead for SNeRV packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_snar_header_grammar_profile import (  # noqa: E402
    SCHEMA,
    build_snerv_snar_header_grammar_profile,
)
from tac.repo_io import write_json_artifact  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--hard-byte-ceiling", action="append", type=int, default=[])
    parser.add_argument("--top-contributor-limit", type=int, default=40)
    parser.add_argument("--expected-output-json-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    report = build_snerv_snar_header_grammar_profile(
        input_path=args.input_path,
        hard_byte_ceilings=tuple(int(value) for value in args.hard_byte_ceiling),
        top_contributor_limit=int(args.top_contributor_limit),
        raw_argv=raw_argv,
    )
    result = write_json_artifact(
        args.output_json,
        report,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": result.path,
                "bytes": result.bytes_written,
                "sha256": result.sha256,
                "packet_bytes": report["packet"]["bytes"],
                "header_bytes": report["header"]["bytes"],
                "section_total_bytes": report["payload"]["section_total_bytes"],
                "header_rewrite_needed_for_any_ceiling": report[
                    "header_rewrite_needed_for_any_ceiling"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
