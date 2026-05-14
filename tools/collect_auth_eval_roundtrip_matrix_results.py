#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Collect recovered score rows for an auth-eval roundtrip matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.auth_eval_roundtrip_matrix import collect_auth_eval_roundtrip_results  # noqa: E402
from tac.repo_io import read_json, write_json  # noqa: E402


def _parse_target_result_dirs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"FATAL: --target-result-dir must be TARGET=DIR, got {value!r}")
        target_id, result_dir = value.split("=", 1)
        target_id = target_id.strip()
        result_dir = result_dir.strip()
        if not target_id or not result_dir:
            raise SystemExit(f"FATAL: --target-result-dir must be TARGET=DIR, got {value!r}")
        parsed[target_id] = result_dir
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--target-result-dir",
        action="append",
        default=[],
        metavar="TARGET=DIR",
        help="Override result directory for a matrix target; may be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = read_json(args.matrix)
    results = collect_auth_eval_roundtrip_results(
        matrix,
        target_result_dirs=_parse_target_result_dirs(args.target_result_dir),
    )
    write_json(args.json_out, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
