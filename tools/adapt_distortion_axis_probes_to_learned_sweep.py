#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Adapt distortion-axis probe verdicts for learned-sweep planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.distortion_axis_probe_learned_sweep_adapter import (  # noqa: E402
    DEFAULT_VARIANCE_FLOOR,
    DistortionAxisProbeLearnedSweepAdapterError,
    build_distortion_axis_probe_learned_sweep_candidates,
    build_refusal_payload,
    dumps_json,
    load_json_object,
    source_artifact_metadata,
    write_json,
)
from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    MLXDynamicLearnedSweepError,
    build_mlx_dynamic_learned_sweep_plan,
    render_mlx_dynamic_learned_sweep_markdown,
)
from tac.repo_io import ArtifactWriteError  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdict",
        type=Path,
        action="append",
        required=True,
        help="Distortion-axis probe verdict JSON. May repeat.",
    )
    parser.add_argument("--incumbent-score", type=float, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=DEFAULT_VARIANCE_FLOOR,
    )
    parser.add_argument(
        "--plan-json-out",
        type=Path,
        help="Optional learned-sweep plan JSON to emit from the adapted candidates.",
    )
    parser.add_argument("--plan-md-out", type=Path)
    parser.add_argument("--plan-top-k", type=int, default=8)
    parser.add_argument("--plan-per-pass-top-k", type=int)
    parser.add_argument(
        "--failure-json-out",
        type=Path,
        help="Optional fail-closed refusal artifact written when adaptation fails.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--expected-plan-output-sha256")
    parser.add_argument("--expected-failure-output-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_artifacts: dict[str, Any] = {}
    try:
        source_artifacts = source_artifact_metadata(args.verdict)
        verdicts = [load_json_object(path) for path in args.verdict]
        payload = build_distortion_axis_probe_learned_sweep_candidates(
            verdicts,
            incumbent_score=args.incumbent_score,
            top_k=args.top_k,
            source_artifacts=source_artifacts,
            variance_floor=args.variance_floor,
        )
        write_json(
            args.json_out,
            payload,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
        plan = None
        if args.plan_json_out is not None:
            plan = build_mlx_dynamic_learned_sweep_plan(
                incumbent_score=args.incumbent_score,
                candidate_payloads=[payload],
                top_k=args.plan_top_k,
                per_pass_top_k=args.plan_per_pass_top_k,
                source_artifacts={
                    "distortion_axis_candidate_payload": {
                        "path": str(args.json_out),
                        "bytes": args.json_out.stat().st_size
                        if args.json_out.is_file()
                        else None,
                    },
                    "source_verdicts": source_artifacts,
                },
            )
            write_json(
                args.plan_json_out,
                plan,
                allow_overwrite=args.allow_overwrite,
                expected_existing_sha256=args.expected_plan_output_sha256,
            )
            if args.plan_md_out is not None:
                if args.plan_md_out.exists() and not args.allow_overwrite:
                    raise ArtifactWriteError(
                        f"{args.plan_md_out}: refusing to overwrite existing artifact"
                    )
                args.plan_md_out.parent.mkdir(parents=True, exist_ok=True)
                args.plan_md_out.write_text(
                    render_mlx_dynamic_learned_sweep_markdown(plan),
                    encoding="utf-8",
                )
    except (
        ArtifactWriteError,
        OSError,
        json.JSONDecodeError,
        DistortionAxisProbeLearnedSweepAdapterError,
        MLXDynamicLearnedSweepError,
        ValueError,
    ) as exc:
        if args.failure_json_out is not None:
            try:
                write_json(
                    args.failure_json_out,
                    build_refusal_payload(
                        error=str(exc),
                        source_artifacts=source_artifacts,
                    ),
                    allow_overwrite=args.allow_overwrite,
                    expected_existing_sha256=args.expected_failure_output_sha256,
                )
            except ArtifactWriteError as refusal_exc:
                print(
                    f"FATAL: {exc}; additionally failed to write refusal artifact: "
                    f"{refusal_exc}",
                    file=sys.stderr,
                )
                return 2
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    summary: dict[str, Any] = {
        "json_out": str(args.json_out),
        "adapted_candidate_count": payload["summary"]["adapted_candidate_count"],
        "suppressed_candidate_count": payload["summary"][
            "suppressed_candidate_count"
        ],
        "best_predicted_score_mean": payload["summary"][
            "best_predicted_score_mean"
        ],
        "best_non_authoritative_repair_budget_bytes_equivalent": payload[
            "summary"
        ]["best_non_authoritative_repair_budget_bytes_equivalent"],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if plan is not None:
        summary["plan_json_out"] = str(args.plan_json_out)
        summary["plan_md_out"] = (
            None if args.plan_md_out is None else str(args.plan_md_out)
        )
        summary["ranked_row_count"] = plan["summary"]["ranked_row_count"]
        summary["local_ready_row_count"] = plan["summary"]["local_ready_row_count"]
    print(dumps_json(summary), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
