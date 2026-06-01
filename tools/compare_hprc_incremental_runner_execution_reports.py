#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare singleton and batched HPRC incremental runner execution reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.hprc.incremental_runner_execution import (  # noqa: E402
    compare_hprc_incremental_runner_execution_reports,
    write_hprc_incremental_runner_execution_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--challenger-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--drift-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--min-default-speedup", type=float, default=1.10)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = compare_hprc_incremental_runner_execution_reports(
        reference_report_path=args.reference_report,
        challenger_report_path=args.challenger_report,
        drift_tolerance=float(args.drift_tolerance),
        min_default_speedup=float(args.min_default_speedup),
    )
    output_path = write_hprc_incremental_runner_execution_comparison(
        output_path=args.output,
        comparison=comparison,
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "default_execution_mode": comparison[
                    "default_execution_recommendation"
                ]["mode"],
                "total_speedup_reference_over_challenger": comparison["speed"][
                    "total_speedup_reference_over_challenger"
                ],
                "delta_total_mlx_score_advisory_drift": comparison["drift"][
                    "delta_total_mlx_score_advisory"
                ],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
