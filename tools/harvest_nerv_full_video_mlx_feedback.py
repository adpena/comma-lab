#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest full-video MLX scorer response into NeRV candidate feedback."""

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

from tac.analysis.nerv_candidate_feedback import (  # noqa: E402
    FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
    write_nerv_full_video_mlx_scorer_feedback_files,
)
from tac.repo_io import read_json, write_json_artifact  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-response", type=Path, required=True)
    parser.add_argument(
        "--archive-export-json",
        type=Path,
        required=True,
        help=(
            "Checkpoint/archive export JSON for the exact candidate measured by "
            "--mlx-response. Archive SHA and bytes must match."
        ),
    )
    parser.add_argument("--candidate-id")
    parser.add_argument("--family", choices=("hi_nerv", "hinerv", "snerv"))
    parser.add_argument("--hard-byte-ceiling", type=int)
    parser.add_argument("--current-segnet-distillation-weight", type=float)
    parser.add_argument("--max-mlx-score-for-local-replay", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    response_path = _resolve(args.mlx_response)
    export_path = _resolve(args.archive_export_json)
    result = write_nerv_full_video_mlx_scorer_feedback_files(
        mlx_response=read_json(response_path),
        archive_export_report=read_json(export_path),
        mlx_response_path=response_path,
        archive_export_report_path=export_path,
        candidate_id=args.candidate_id,
        family=args.family,
        hard_byte_ceiling=args.hard_byte_ceiling,
        current_segnet_distillation_weight=args.current_segnet_distillation_weight,
        max_mlx_score_for_local_replay=args.max_mlx_score_for_local_replay,
        output_dir=_resolve(args.output_dir),
    )
    if args.output_json is not None:
        write_json_artifact(_resolve(args.output_json), result)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))
    return 0


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.get("row") or {})
    response = dict(row.get("full_video_mlx_scorer_response") or {})
    control = dict(row.get("full_video_mlx_response_control") or {})
    return {
        "schema": FULL_VIDEO_MLX_SCORER_FEEDBACK_SCHEMA,
        "manifest_path": result.get("manifest_path"),
        "row_path": result.get("row_path"),
        "ledger_path": result.get("ledger_path"),
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "measured_num_pairs": row.get("measured_num_pairs"),
        "measured_archive_bytes": row.get("measured_archive_bytes"),
        "hard_byte_ceiling": row.get("hard_byte_ceiling"),
        "archive_under_hard_byte_ceiling": response.get(
            "archive_under_hard_byte_ceiling"
        ),
        "score_recomputed_from_components": response.get(
            "score_recomputed_from_components"
        ),
        "nonrate_score_estimate": response.get("nonrate_score_estimate"),
        "avg_segnet_dist": response.get("avg_segnet_dist"),
        "avg_posenet_dist": response.get("avg_posenet_dist"),
        "training_control_action": control.get("action"),
        "training_control_should_stop_current_run": control.get(
            "should_stop_current_run"
        ),
        "recommended_launch_mutations": control.get(
            "recommended_launch_mutations"
        ),
        "direct_feedback_blockers": row.get("direct_feedback_blockers"),
        "feedback_ready": row.get("feedback_ready"),
        "score_claim": result.get("score_claim"),
        "ready_for_exact_eval_dispatch": result.get("ready_for_exact_eval_dispatch"),
    }


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (REPO_ROOT / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
