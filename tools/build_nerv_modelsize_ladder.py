#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the false-authority HiNeRV/SNeRV model-size ladder."""

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

from tac.analysis.nerv_modelsize_ladder import (  # noqa: E402
    NERV_MODELSIZE_LADDER_SCHEMA,
    build_nerv_modelsize_ladder,
    render_nerv_modelsize_ladder_markdown,
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
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--scorer-height", type=int, default=384)
    parser.add_argument("--scorer-width", type=int, default=512)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    args = parser.parse_args(argv)

    report = build_nerv_modelsize_ladder(
        focus_families=tuple(args.focus_family or ("hi_nerv", "snerv")),
        num_pairs=int(args.num_pairs),
        scorer_height=int(args.scorer_height),
        scorer_width=int(args.scorer_width),
    )
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(render_nerv_modelsize_ladder_markdown(report), encoding="utf-8")
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": NERV_MODELSIZE_LADDER_SCHEMA,
        "report_path": report.get("report_path"),
        "focus_families": report["focus_families"],
        "family_count": len(report["family_rows"]),
        "ladder_row_count": sum(
            len(row.get("ladder_rows") or ()) for row in report["family_rows"]
        ),
        "marginal_gate_count": sum(
            len(row.get("marginal_gates") or ()) for row in report["family_rows"]
        ),
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
