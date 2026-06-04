#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest a SNeRV upstream eval gate into planner candidate feedback."""

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

from tac.analysis.snerv_upstream_eval_feedback import (  # noqa: E402
    write_snerv_upstream_eval_candidate_feedback,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    row = write_snerv_upstream_eval_candidate_feedback(
        gate_report_path=args.gate_json,
        output_json=args.output_json,
    )
    print(
        json.dumps(
            {
                "schema": row["schema"],
                "feedback_kind": row["feedback_kind"],
                "output_json": args.output_json.as_posix(),
                "measured_archive_bytes": row.get("measured_archive_bytes"),
                "direct_feedback_blockers": row.get("direct_feedback_blockers"),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
