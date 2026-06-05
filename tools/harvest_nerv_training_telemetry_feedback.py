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
from tac.repo_io import write_json_artifact  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, help="training telemetry.jsonl")
    parser.add_argument("--family", required=True, choices=("hi_nerv", "hinerv", "snerv"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-num-pairs", type=int, default=600)
    parser.add_argument("--source-queue")
    parser.add_argument("--stop-reason")
    parser.add_argument(
        "--current-segnet-distillation-weight",
        type=float,
        help=(
            "Known launch/current SegNet pressure for PR95-curriculum "
            "telemetry, where loss-part ratios expose score-scale weights "
            "rather than the user-facing launch control."
        ),
    )
    parser.add_argument(
        "--training-running",
        action="store_true",
        help=(
            "Mark this as a midrun feedback snapshot. Without this flag, the "
            "row is treated as terminal unless --stop-reason is already a "
            "recognized midrun reason."
        ),
    )
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
        stop_reason=_effective_stop_reason(
            stop_reason=args.stop_reason,
            training_running=bool(args.training_running),
        ),
        current_segnet_distillation_weight=args.current_segnet_distillation_weight,
    )
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve(strict=False)
        manifest_path = Path(str(result.get("manifest_path") or "")).expanduser().resolve(
            strict=False
        )
        if output_json == manifest_path:
            # The writer above already emitted the canonical manifest. Treat an
            # identical --output-json as an alias, not a self-overwrite.
            pass
        else:
            write_json_artifact(
                output_json,
                result,
            )
    print(json.dumps(_summary(result), indent=2, sort_keys=True))
    return 0


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.get("row") or {})
    telemetry = dict(row.get("training_telemetry") or {})
    hinerv_control = dict(row.get("hinerv_train_time_control_health") or {})
    hinerv_section = dict(hinerv_control.get("section_byte_control_health") or {})
    gradient_control = dict(row.get("gradient_multiplier_control_health") or {})
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
        "pose_tail_burst_detected": row.get("pose_tail_burst_detected"),
        "pose_tail_burst_recent_p95": row.get("pose_tail_burst_recent_p95"),
        "pose_tail_burst_recent_max": row.get("pose_tail_burst_recent_max"),
        "pose_tail_burst_threshold": row.get("pose_tail_burst_threshold"),
        "recommended_learning_rate": row.get("recommended_learning_rate"),
        "seg_stagnation_detected": row.get("seg_stagnation_detected"),
        "seg_stagnation_relative_improvement": row.get(
            "seg_stagnation_relative_improvement"
        ),
        "observed_segnet_distillation_weight": row.get(
            "observed_segnet_distillation_weight"
        ),
        "segnet_distillation_weight_source": row.get(
            "segnet_distillation_weight_source"
        ),
        "recommended_segnet_distillation_weight": row.get(
            "recommended_segnet_distillation_weight"
        ),
        "training_control_action": row.get("training_control_action"),
        "training_control_should_stop_current_run": row.get(
            "training_control_should_stop_current_run"
        ),
        "direct_feedback_blockers": row.get("direct_feedback_blockers"),
        "hinerv_train_time_control_blockers": row.get(
            "hinerv_train_time_control_blockers"
        ),
        "hinerv_section_rate_metric_observed": hinerv_section.get(
            "section_rate_metric_observed"
        ),
        "hinerv_section_byte_dual_lambda_active_observed": hinerv_section.get(
            "section_byte_dual_lambda_active_observed"
        ),
        "hinerv_section_byte_dual_weight_applied_observed": hinerv_section.get(
            "section_byte_dual_weight_applied_observed"
        ),
        "hinerv_section_byte_dual_zero_base_masked_observed": hinerv_section.get(
            "section_byte_dual_zero_base_masked_observed"
        ),
        "hinerv_archive_rate_metric_observed": hinerv_section.get(
            "archive_rate_metric_observed"
        ),
        "hinerv_archive_byte_dual_lambda_active_observed": hinerv_section.get(
            "archive_byte_dual_lambda_active_observed"
        ),
        "hinerv_archive_byte_dual_weight_applied_observed": hinerv_section.get(
            "archive_byte_dual_weight_applied_observed"
        ),
        "gradient_multiplier_control_inert_risk_detected": row.get(
            "gradient_multiplier_control_inert_risk_detected"
        ),
        "gradient_multiplier_control_blockers": row.get(
            "gradient_multiplier_control_blockers"
        ),
        "gradient_multiplier_requested_observed": gradient_control.get(
            "requested_observed"
        ),
        "gradient_multiplier_applied_observed": gradient_control.get(
            "applied_observed"
        ),
        "gradient_multiplier_missing_requested_observed": gradient_control.get(
            "missing_requested_observed"
        ),
        "gradient_multiplier_requested_but_unapplied_observed": (
            gradient_control.get("requested_but_unapplied_observed")
        ),
        "training_stopped": row.get("training_stopped"),
        "last_epoch": telemetry.get("last_epoch"),
        "score_claim": result.get("score_claim"),
        "ready_for_exact_eval_dispatch": result.get("ready_for_exact_eval_dispatch"),
    }


def _effective_stop_reason(*, stop_reason: str | None, training_running: bool) -> str | None:
    if training_running and not str(stop_reason or "").strip():
        return "training_running_midrun_feedback_snapshot"
    return stop_reason


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
