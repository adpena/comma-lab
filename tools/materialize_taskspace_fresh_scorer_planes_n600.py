#!/usr/bin/env python3
"""Preflight or materialize the fresh n600 scorer-plane operand stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (  # noqa: E402
    FreshScorerPlaneMaterializationError,
    file_identity,
    materialize,
    run_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--materialize", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.preflight_only:
            path, receipt = run_preflight(args.config)
            kind = "preflight"
            self_hash = receipt["preflight_sha256"]
        else:
            assert_governed_admission(
                "taskspace_fresh_scorer_plane_materializer_n600",
                on_refuse="raise",
            )
            path, receipt = materialize(args.config)
            kind = "aggregate"
            self_hash = receipt["aggregate_receipt_sha256"]
    except FreshScorerPlaneMaterializationError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "kind": kind,
                "receipt": file_identity(path),
                "sealed_self_sha256": self_hash,
                "pointer_moved": False,
                "score_claim": False,
                "candidate_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
