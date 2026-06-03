#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recover false-authority custody for interrupted compact NeRV runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tools.run_compact_renderer_mlx_spine_runner import (  # noqa: E402
    _write_compact_family_interrupted_report_from_startup_marker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reason",
        default="manual_recovery_after_missing_terminal_report",
        help="Durable reason recorded in the recovered false-authority report.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite an existing interrupted report. Default preserves it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _write_compact_family_interrupted_report_from_startup_marker(
        output_dir=args.output_dir,
        reason=args.reason,
        overwrite=bool(args.overwrite),
    )
    telemetry = dict(report.get("telemetry_summary") or {})
    print(
        json.dumps(
            {
                "schema": report.get("schema"),
                "mode": report.get("mode"),
                "report_path": report.get("report_path"),
                "recovered": report.get("recovered"),
                "execute_family": report.get("execute_family"),
                "planner_row_id": report.get("planner_row_id"),
                "last_epoch": telemetry.get("last_epoch"),
                "pr95_stage_index": dict(
                    telemetry.get("last_loss_components") or {}
                ).get("pr95_stage_index"),
                "score_claim": report.get("score_claim"),
                "ready_for_exact_eval_dispatch": report.get(
                    "ready_for_exact_eval_dispatch"
                ),
                "blockers": report.get("blockers"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
