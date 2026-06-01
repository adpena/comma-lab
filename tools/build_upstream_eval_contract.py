#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Emit the fail-closed contract for the pinned upstream contest evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.contest_eval_contract import build_upstream_eval_contract  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    contract = build_upstream_eval_contract(
        repo_root=args.repo_root,
        upstream_dir=args.upstream_dir,
    )
    body = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(body, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
        print(f"[upstream-eval-contract] wrote {args.output}")
    return 0 if contract["contract_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
