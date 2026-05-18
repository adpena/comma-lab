#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strict gate for canonical task-status schema, transitions, and memo pointers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from tac.canonical_task_status import (
        check_canonical_task_status_no_dangling_transitions,
    )

    violations = check_canonical_task_status_no_dangling_transitions(
        repo_root=args.repo_root,
        strict=args.strict,
        verbose=not args.json,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "status": "pass" if not violations else "fail",
                    "violations": violations,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 1 if violations and args.strict else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

