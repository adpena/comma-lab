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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_stack_receipt(repo_root=REPO, strict_source_reopen=True)
    write_once_receipt(args.output, receipt, repo_root=REPO)
    body = receipt["body"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "body_sha256": receipt["body_sha256"],
                "verdict": body["verdict"],
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
