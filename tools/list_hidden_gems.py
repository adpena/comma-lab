#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""List static hidden-gem techniques without dispatching GPU work."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _hidden_gems_module() -> ModuleType:
    from tac import hidden_gems

    return hidden_gems


def build_parser() -> argparse.ArgumentParser:
    hidden_gems = _hidden_gems_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format. Default: markdown.",
    )
    parser.add_argument(
        "--category",
        choices=sorted(hidden_gems.CATEGORIES),
        help="Filter to one registry category.",
    )
    parser.add_argument(
        "--status",
        choices=sorted(hidden_gems.STATUSES),
        help="Filter to one registry status.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to this path instead of stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    hidden_gems = _hidden_gems_module()
    args = build_parser().parse_args(argv)
    entries = hidden_gems.all_hidden_gems(category=args.category, status=args.status)

    if args.format == "json":
        from tac.repo_io import json_text

        text = json_text(hidden_gems.registry_payload(entries))
    else:
        text = hidden_gems.render_markdown(entries)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
