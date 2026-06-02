#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build false-authority SNeRV scorer-loop geometry analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_scorer_loop_geometry import (  # noqa: E402
    DEFAULT_PUBLIC_FRONTIER_REFERENCE,
    build_snerv_scorer_loop_geometry_report,
    render_snerv_scorer_loop_geometry_markdown,
)
from tac.repo_io import write_json, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        action="append",
        required=True,
        help="SNeRV scorer-loop result JSON. May be repeated.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--label", default="snerv_scorer_loop_geometry")
    parser.add_argument(
        "--frontier-reference-score",
        type=float,
        default=DEFAULT_PUBLIC_FRONTIER_REFERENCE,
    )
    args = parser.parse_args(argv)

    report = build_snerv_scorer_loop_geometry_report(
        args.result_json,
        label=args.label,
        frontier_reference_score=float(args.frontier_reference_score),
    )
    out_json = Path(args.output_json).expanduser().resolve(strict=False)
    write_json(out_json, report)
    if args.output_md:
        out_md = Path(args.output_md).expanduser().resolve(strict=False)
        write_text_artifact(
            out_md,
            render_snerv_scorer_loop_geometry_markdown(report),
            allow_overwrite=True,
        )

    aggregate = report["aggregate"]
    print("[SNeRV scorer-loop geometry] false-authority")
    print(f"  inputs: {report['input_count']}")
    print(f"  best_descent_score_linf: {report['best_descent_score_linf']}")
    print(f"  best_descent_score_delta_linf: {report['best_descent_score_delta_linf']}")
    print(f"  lowest_local_score_linf: {report['lowest_local_score_linf']}")
    print(f"  best_search_mode: {aggregate.get('best_search_mode')}")
    print(f"  dominant_lowering_axis: {aggregate.get('dominant_lowering_axis')}")
    print(f"  wrote: {out_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
