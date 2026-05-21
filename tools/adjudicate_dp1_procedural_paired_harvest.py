#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adjudicate DP1 procedural-codebook paired CPU/CUDA harvests."""

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

from tac.optimization.dp1_procedural_paired_adjudication import (  # noqa: E402
    build_dp1_procedural_paired_adjudication,
    register_dp1_procedural_paired_adjudication,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--procedural-output-dir", type=Path, required=True)
    parser.add_argument("--baseline-cpu-dir", type=Path, required=True)
    parser.add_argument("--baseline-cuda-dir", type=Path, required=True)
    parser.add_argument("--procedural-cpu-dir", type=Path, required=True)
    parser.add_argument("--procedural-cuda-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument(
        "--register-probe-outcome",
        action="store_true",
        help="Append the adjudication to .omx/state/probe_outcomes.jsonl.",
    )
    args = parser.parse_args(argv)

    report = build_dp1_procedural_paired_adjudication(
        baseline_output_dir=args.baseline_output_dir,
        procedural_output_dir=args.procedural_output_dir,
        baseline_cpu_dir=args.baseline_cpu_dir,
        baseline_cuda_dir=args.baseline_cuda_dir,
        procedural_cpu_dir=args.procedural_cpu_dir,
        procedural_cuda_dir=args.procedural_cuda_dir,
        repo_root=REPO_ROOT,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(render_markdown(report), encoding="utf-8")
    if args.register_probe_outcome:
        evidence_path = args.json_out or args.md_out
        record = register_dp1_procedural_paired_adjudication(
            report,
            evidence_path=evidence_path,
        )
        report = {
            **report,
            "probe_outcome_registered": True,
            "probe_outcome_record": record,
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.json_out:
            args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("all_required_evidence_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
