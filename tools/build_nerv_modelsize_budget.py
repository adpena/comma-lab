#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build planning-grade HiNeRV/SNeRV model-size budget artifacts."""

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

from tac.analysis.nerv_modelsize_budget import (  # noqa: E402
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-hinerv-json", required=True, type=Path)
    parser.add_argument("--output-snerv-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--hard-byte-ceiling", action="append", type=int)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--per-ceiling-limit", type=int, default=6)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Allow replacing existing output files, but only when the matching "
            "expected-output-* sha256 flag is supplied."
        ),
    )
    parser.add_argument("--expected-output-hinerv-json-sha256")
    parser.add_argument("--expected-output-snerv-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    args = parser.parse_args(argv)

    hard_byte_ceilings = tuple(
        int(value) for value in (args.hard_byte_ceiling or (216_000, 285_000, 360_000))
    )
    hinerv = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=int(args.num_pairs),
        per_ceiling_limit=int(args.per_ceiling_limit),
    )
    snerv = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=int(args.num_pairs),
        per_ceiling_limit=int(args.per_ceiling_limit),
    )
    hinerv_result = write_json_artifact(
        args.output_hinerv_json,
        hinerv,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=args.expected_output_hinerv_json_sha256,
    )
    snerv_result = write_json_artifact(
        args.output_snerv_json,
        snerv,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=args.expected_output_snerv_json_sha256,
    )
    md_result = None
    if args.output_md is not None:
        md_result = write_text_artifact(
            args.output_md,
            _render_markdown(hinerv, snerv),
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_sha256=args.expected_output_md_sha256,
        )

    summary = {
        "schema": "nerv_modelsize_budget_build.v1",
        "inputs": {
            "hard_byte_ceilings": list(hard_byte_ceilings),
            "num_pairs": int(args.num_pairs),
            "per_ceiling_limit": int(args.per_ceiling_limit),
        },
        "hinerv_output_json": hinerv_result.path,
        "hinerv_output_sha256": hinerv_result.sha256,
        "snerv_output_json": snerv_result.path,
        "snerv_output_sha256": snerv_result.sha256,
        "output_md": None if md_result is None else md_result.path,
        "output_md_sha256": None if md_result is None else md_result.sha256,
        "hinerv_selected_candidate_count": int(hinerv["selected_candidate_count"]),
        "snerv_selected_candidate_count": int(snerv["selected_candidate_count"]),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _render_markdown(hinerv: dict[str, Any], snerv: dict[str, Any]) -> str:
    lines = [
        "# NeRV Model-Size Budgets",
        "",
        "False-authority planning artifact. These rows price model-size controls",
        "before byte-closed archive export, receiver proof, local replay, or exact auth.",
        "",
        "## Summary",
        "",
        f"- HiNeRV selected candidates: `{hinerv['selected_candidate_count']}`",
        f"- SNeRV selected candidates: `{snerv['selected_candidate_count']}`",
        f"- Num pairs: `{hinerv['num_pairs']}`",
        f"- Score claim: `{hinerv['score_claim']}`",
        f"- Ready for exact eval: `{hinerv['ready_for_exact_eval_dispatch']}`",
        "",
        "## Top HiNeRV Candidates",
        "",
    ]
    lines.extend(_candidate_lines(hinerv.get("selected_candidates") or []))
    lines.extend(["", "## Top SNeRV Candidates", ""])
    lines.extend(_candidate_lines(snerv.get("selected_candidates") or []))
    return "\n".join(lines).rstrip() + "\n"


def _candidate_lines(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows[:8]:
        payload_bytes = row.get("nominal_total_payload_bytes", row.get("total_payload_bytes"))
        out.append(
            "- "
            f"`{row.get('candidate_id')}` "
            f"payload=`{payload_bytes}` "
            f"nominal_under_ceiling=`{row.get('nominal_under_ceiling')}`"
        )
    return out or ["- none"]


if __name__ == "__main__":
    raise SystemExit(main())
