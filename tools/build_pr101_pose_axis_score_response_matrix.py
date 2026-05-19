#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a paired score-response matrix for a PR101 pose-axis candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.master_gradient_pr101_score_response_matrix import (  # noqa: E402
    PR101ScoreResponseMatrixError,
    build_pr101_pose_axis_score_response_matrix,
    render_pr101_pose_axis_score_response_matrix_markdown,
)
from tac.repo_io import read_json, write_json  # noqa: E402


def _load_contest_auth_eval_module() -> Any:
    path = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location(
        "contest_auth_eval_runtime_manifest_pr101_pose_axis",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import runtime manifest helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-submission-dir", type=Path, required=True)
    parser.add_argument("--operator-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--runtime-proof", type=Path, required=True)
    parser.add_argument("--inflate-sh", default="inflate.sh")
    parser.add_argument("--label", default="pr101-op7-raw-byte-delta")
    parser.add_argument("--lane-id", default="pr101_pose_axis_op7_score_response")
    parser.add_argument(
        "--output-root",
        default="experiments/results/pr101_pose_axis_score_response_matrix",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--min-total-improvement", type=float, default=0.001)
    parser.add_argument("--min-scorer-term-improvement", type=float, default=0.0005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inflate_path = args.source_submission_dir / args.inflate_sh
    try:
        contest_auth_eval = _load_contest_auth_eval_module()
        runtime_manifest = contest_auth_eval._runtime_dependency_manifest(
            inflate_path,
            REPO_ROOT / "upstream",
            repo_root=REPO_ROOT,
        )
        matrix = build_pr101_pose_axis_score_response_matrix(
            source_archive=args.source_archive,
            source_submission_dir=args.source_submission_dir,
            operator_candidate_manifest=read_json(args.operator_manifest),
            packet_candidate_manifest=read_json(args.candidate_manifest),
            runtime_consumption_proof=read_json(args.runtime_proof),
            runtime_manifest=runtime_manifest,
            repo_root=REPO_ROOT,
            label=args.label,
            lane_id=args.lane_id,
            output_root=args.output_root,
            include_diagnostics=not args.no_diagnostics,
            min_total_improvement=args.min_total_improvement,
            min_scorer_term_improvement=args.min_scorer_term_improvement,
        )
    except (
        PR101ScoreResponseMatrixError,
        OSError,
        json.JSONDecodeError,
        RuntimeError,
    ) as exc:
        raise SystemExit(f"PR101 pose-axis score-response matrix failed: {exc}") from None

    write_json(args.json_out, matrix)
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            render_pr101_pose_axis_score_response_matrix_markdown(matrix),
            encoding="utf-8",
        )
    print(
        "score_claim=false promotion_eligible=false "
        f"ready_for_score_response_probe={matrix['ready_for_score_response_probe']}"
    )
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
