#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build decoder-weight waterfill plans from SNeRV trained ladder rows."""

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

from tac.analysis.nerv_decoder_weight_waterfill import load_saliency_json  # noqa: E402
from tac.analysis.snerv_trained_ladder_waterfill import (  # noqa: E402
    SNERV_TRAINED_LADDER_WATERFILL_SCHEMA,
    build_snerv_trained_ladder_waterfill,
    render_snerv_trained_ladder_waterfill_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-ladder-row-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--saliency-json", default=None, type=Path)
    parser.add_argument("--action-bits", default="0,2,4,8,16,32")
    parser.add_argument("--candidate-id", default=None)
    args = parser.parse_args(argv)

    trained_row = json.loads(args.trained_ladder_row_json.read_text(encoding="utf-8"))
    saliency = (
        None if args.saliency_json is None else load_saliency_json(args.saliency_json)
    )
    report = build_snerv_trained_ladder_waterfill(
        trained_row,
        saliency_by_name=saliency,
        action_bits=_parse_action_bits(args.action_bits),
        candidate_id=args.candidate_id,
    )
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_snerv_trained_ladder_waterfill_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _parse_action_bits(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SNERV_TRAINED_LADDER_WATERFILL_SCHEMA,
        "report_path": report.get("report_path"),
        "row_count": report["row_count"],
        "section_value_row_count": len(report["section_value_rows"]),
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
