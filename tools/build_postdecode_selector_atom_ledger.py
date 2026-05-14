#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a meta-Lagrangian atom ledger from a postdecode selector sweep JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.postdecode_selector_waterfill import (  # noqa: E402
    build_postdecode_selector_waterfill_plan,
)
from tac.repo_io import read_json, repo_relative, write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--atom-ledger-output",
        type=Path,
        help="Optional path for the nested canonical atom ledger only.",
    )
    parser.add_argument(
        "--selector-byte-delta",
        type=int,
        required=True,
        help="Charged archive byte delta for the selector stream/runtime.",
    )
    parser.add_argument(
        "--mode-byte-delta",
        type=int,
        default=0,
        help="Charged archive byte delta for fixed global mode atoms.",
    )
    parser.add_argument("--confidence", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep = read_json(args.sweep_json)
    plan = build_postdecode_selector_waterfill_plan(
        sweep,
        selector_byte_delta=args.selector_byte_delta,
        mode_byte_delta=args.mode_byte_delta,
        confidence=args.confidence,
        evidence_source_path=repo_relative(args.sweep_json, REPO_ROOT),
    )
    write_json(args.output, plan)
    if args.atom_ledger_output is not None:
        write_json(args.atom_ledger_output, plan["atom_ledger"])
    print(f"wrote {args.output}")
    if args.atom_ledger_output is not None:
        print(f"wrote {args.atom_ledger_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
