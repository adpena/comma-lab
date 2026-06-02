#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build decoder-weight waterfill plans from a HiNeRV archive ladder report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hinerv_archive_ladder_waterfill import (  # noqa: E402
    HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA,
    build_hinerv_archive_ladder_waterfill,
    render_hinerv_archive_ladder_waterfill_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-ladder-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--saliency-json", default=None, type=Path)
    parser.add_argument("--action-bits", default="0,2,4,8,16,32")
    parser.add_argument("--candidate-id", default=None)
    args = parser.parse_args(argv)

    ladder = json.loads(args.archive_ladder_json.read_text(encoding="utf-8"))
    saliency_payload = (
        {}
        if args.saliency_json is None
        else json.loads(args.saliency_json.read_text(encoding="utf-8"))
    )
    report = build_hinerv_archive_ladder_waterfill(
        ladder,
        saliency_by_row_id=_row_saliency(saliency_payload),
        global_saliency_by_name=_global_saliency(saliency_payload),
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
            render_hinerv_archive_ladder_waterfill_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _row_saliency(payload: Any) -> dict[str, dict[str, float]]:
    if not isinstance(payload, Mapping):
        return {}
    rows = payload.get("row_saliency") or payload.get("saliency_by_row_id")
    if not isinstance(rows, Mapping):
        return {}
    out: dict[str, dict[str, float]] = {}
    for row_id, mapping in rows.items():
        if isinstance(mapping, Mapping):
            out[str(row_id)] = {
                str(key): float(value)
                for key, value in mapping.items()
                if _float_or_none(value) is not None
            }
    return out


def _global_saliency(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    mapping = payload.get("global_saliency") or payload.get("saliency_by_name")
    if not isinstance(mapping, Mapping):
        mapping = payload
    return {
        str(key): float(value)
        for key, value in mapping.items()
        if _float_or_none(value) is not None
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_action_bits(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": HINERV_ARCHIVE_LADDER_WATERFILL_SCHEMA,
        "report_path": report.get("report_path"),
        "row_count": report["row_count"],
        "section_value_row_count": len(report["section_value_rows"]),
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
        "blockers": report["blockers"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
