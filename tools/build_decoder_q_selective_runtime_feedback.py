#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build fail-closed feedback from a DQS1 selective decoder-q probe."""

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

from tac.optimization.decoder_q_selective_runtime_feedback import (  # noqa: E402
    DecoderQSelectiveRuntimeFeedbackError,
    build_decoder_q_selective_runtime_feedback,
    build_sign_calibration_labels,
    load_json_object,
    load_packet_plan_for_materialization,
    render_decoder_q_selective_runtime_feedback_markdown,
    write_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-plan", type=Path, required=True)
    parser.add_argument("--packet-plan", type=Path)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--locality-controls", type=Path, required=True)
    parser.add_argument("--advisory-result", type=Path, required=True)
    parser.add_argument("--local-baseline-score", type=float, required=True)
    parser.add_argument("--min-dispatch-edge", type=float, required=True)
    parser.add_argument("--contest-cpu-frontier-score", type=float)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--signed-calibration-out", type=Path)
    parser.add_argument(
        "--additional-feedback",
        type=Path,
        action="append",
        default=[],
        help="Existing feedback JSON to include when writing --signed-calibration-out.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        materialization_manifest = load_json_object(args.materialization_manifest)
        packet_plan = (
            load_json_object(args.packet_plan)
            if args.packet_plan
            else load_packet_plan_for_materialization(
                materialization_manifest,
                manifest_path=args.materialization_manifest,
            )
        )
        feedback = build_decoder_q_selective_runtime_feedback(
            bridge_plan=load_json_object(args.bridge_plan),
            materialization_manifest=materialization_manifest,
            locality_controls=load_json_object(args.locality_controls),
            advisory_result=load_json_object(args.advisory_result),
            packet_plan=packet_plan,
            local_baseline_score=args.local_baseline_score,
            min_dispatch_edge=args.min_dispatch_edge,
            contest_cpu_frontier_score=args.contest_cpu_frontier_score,
        )
        write_json(args.json_out, feedback)
        if args.md_out is not None:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(
                render_decoder_q_selective_runtime_feedback_markdown(feedback),
                encoding="utf-8",
            )
        if args.signed_calibration_out is not None:
            feedback_rows = [feedback]
            feedback_rows.extend(load_json_object(path) for path in args.additional_feedback)
            source_paths = [args.json_out.resolve(), *(path.resolve() for path in args.additional_feedback)]
            write_json(
                args.signed_calibration_out,
                build_sign_calibration_labels(
                    feedback_rows,
                    source=";".join(str(path) for path in source_paths),
                ),
            )
    except (OSError, json.JSONDecodeError, DecoderQSelectiveRuntimeFeedbackError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "dispatch_recommended": feedback["decision"]["dispatch_recommended"],
                "recommended_next_action": feedback["decision"]["recommended_next_action"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
