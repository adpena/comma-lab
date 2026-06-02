#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest NeRV long-training telemetry into candidate feedback."""

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
    write_nerv_training_telemetry_feedback_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, help="training telemetry.jsonl")
    parser.add_argument("--family", required=True, choices=("hi_nerv", "hinerv", "snerv"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-num-pairs", type=int, default=600)
    parser.add_argument("--source-queue")
    parser.add_argument("--stop-reason")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", help="Optional copy of the manifest JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = write_nerv_training_telemetry_feedback_files(
        telemetry_path=args.telemetry,
        output_dir=args.output_dir,
        family=args.family,
        candidate_id=args.candidate_id,
        candidate_num_pairs=args.candidate_num_pairs,
        source_queue_path=args.source_queue,
        stop_reason=args.stop_reason,
    )
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve(strict=False)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(_summary(result), indent=2, sort_keys=True))
    return 0


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.get("row") or {})
    telemetry = dict(row.get("training_telemetry") or {})
    return {
        "schema": result.get("schema"),
        "row_path": result.get("row_path"),
        "manifest_path": result.get("manifest_path"),
        "family": row.get("family"),
        "candidate_id": row.get("candidate_id"),
        "measured_num_pairs": row.get("measured_num_pairs"),
        "pose_instability_detected": row.get("pose_instability_detected"),
        "pose_instability_ever_detected": row.get(
            "pose_instability_ever_detected"
        ),
        "pose_instability_recovered": row.get("pose_instability_recovered"),
        "pose_instability_active_latest_window": row.get(
            "pose_instability_active_latest_window"
        ),
        "pose_instability_first_epoch": row.get("pose_instability_first_epoch"),
        "recommended_learning_rate": row.get("recommended_learning_rate"),
        "seg_stagnation_detected": row.get("seg_stagnation_detected"),
        "seg_stagnation_relative_improvement": row.get(
            "seg_stagnation_relative_improvement"
        ),
        "recommended_segnet_distillation_weight": row.get(
            "recommended_segnet_distillation_weight"
        ),
        "training_stopped": row.get("training_stopped"),
        "last_epoch": telemetry.get("last_epoch"),
        "score_claim": result.get("score_claim"),
        "ready_for_exact_eval_dispatch": result.get("ready_for_exact_eval_dispatch"),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
