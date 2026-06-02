#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the false-authority NeRV-family control inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_control_inventory import (  # noqa: E402
    build_nerv_control_inventory,
    render_nerv_control_inventory_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--focus-family",
        action="append",
        default=None,
        help="Carrier family to include. Defaults to hi_nerv and snerv.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Durable JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional durable Markdown report path.",
    )
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        type=Path,
        help="Repository root used for the implementation sweep.",
    )
    args = parser.parse_args(argv)

    focus = tuple(args.focus_family or ("hi_nerv", "snerv"))
    report = build_nerv_control_inventory(
        focus_families=focus,
        repo_root=args.repo_root,
    )
    output = Path(args.output_json).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    if args.output_md:
        md_output = Path(args.output_md).expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
    write_json(output, report)
    if args.output_md:
        md_output.write_text(
            render_nerv_control_inventory_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "report_path": report.get("report_path"),
        "focus_families": report["focus_families"],
        "control_count": len(report["control_rows"]),
        "binding_gap_count": len(report["binding_gap_rows"]),
        "work_order_count": len(report["recommended_next_work_orders"]),
        "implementation_sweep_status": report["implementation_sweep"]["status"],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
