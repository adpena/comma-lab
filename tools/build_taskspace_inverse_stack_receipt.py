#!/usr/bin/env python3
"""Build the bounded research-only P/G/A/optional-T stack receipt with encoder-only E."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.witness_dsl.taskspace_inverse_stack_receipt import (  # noqa: E402
    DEFAULT_OUTPUT,
    build_stack_receipt,
    write_once_receipt,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict-source-reopen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reopen the teacher census and prior-harvest producers through their own full "
            "source-custody validators (default). --no-strict-source-reopen builds the receipt "
            "in the DECLARED degraded mode instead: source_reopen reads NOT_RUN and the "
            "strict_teacher_and_harvest_source_reopen_not_run exact blocker is added. That "
            "mode is diagnostic only and cannot be published -- see --dry-run below."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and report the receipt without publishing it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # A NOT_RUN receipt is an honest degraded record, but it is not a canonical
    # one: `write_once_receipt` re-validates by rebuilding with the strict
    # reopen, so publishing one is impossible by construction. Say that here
    # rather than letting the publisher fail with a confusing rebuild error --
    # the degraded mode exists so the stack can still be INSPECTED while an
    # upstream producer pin is drifted (ddm_ql1 root cause 3: the V9/PBR2
    # renderer manifest drift broke this tool, not only its test).
    dry_run = args.dry_run or not args.strict_source_reopen
    receipt = build_stack_receipt(repo_root=REPO, strict_source_reopen=args.strict_source_reopen)
    if not dry_run:
        write_once_receipt(args.output, receipt, repo_root=REPO)
    body = receipt["body"]
    print(
        json.dumps(
            {
                "output": None if dry_run else str(args.output),
                "published": not dry_run,
                "strict_source_reopen": args.strict_source_reopen,
                "source_reopen": body["source_reopen"],
                "body_sha256": receipt["body_sha256"],
                "verdict": body["verdict"],
                "exact_blockers": body["exact_blockers"],
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
