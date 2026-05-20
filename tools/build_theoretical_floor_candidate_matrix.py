#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit source-backed theoretical-floor candidate runtime contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text  # noqa: E402
from tac.theoretical_floor_candidates import (  # noqa: E402
    build_candidate_matrix,
    render_candidate_matrix_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    matrix = build_candidate_matrix(repo_root=REPO_ROOT)
    text = json_text(matrix)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_candidate_matrix_markdown(matrix), encoding="utf-8")
    if args.json_out is None and args.md_out is None:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
