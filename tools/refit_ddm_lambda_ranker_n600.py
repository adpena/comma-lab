#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the deterministic advisory DDM CO3 N600 lambda-ranker receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.ddm_lambda_ranker import (  # noqa: E402
    RUN_ID,
    build_n600_lambda_ranker_receipt,
    write_receipt_atomic,
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / ".omx"
    / "research"
    / RUN_ID
    / "ddm_co3_lambda_refit_full_join_receipt.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit and pair-held-out-evaluate the advisory N600 DDM lambda ranker"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="durable receipt path (default: canonical CO3 research directory)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="also print the compact selected-model/admission summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_n600_lambda_ranker_receipt(REPO_ROOT)
    write_receipt_atomic(args.output, payload)
    if args.stdout:
        print(
            json.dumps(
                {
                    "schema": payload["schema"],
                    "run_id": payload["run_id"],
                    "content_sha256": payload["content_sha256"],
                    "selected_model": payload["selected_model"],
                    "admission_gate": payload["admission_gate"],
                    "blocker_ids": payload["blocker_ids"],
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
