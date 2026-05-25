#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest distortion learned-sweep probe signal into feedback observations."""

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

from tac.optimization.distortion_axis_probe_learned_sweep_feedback import (  # noqa: E402
    DEFAULT_OPTIMIZATION_PASS_ID,
    DEFAULT_SWEEP_CONFIG_ID,
    DistortionAxisProbeLearnedSweepFeedbackError,
    append_distortion_axis_probe_feedback_observation,
    build_feedback_summary,
    dumps_json,
    load_json_object,
)
from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    MLXDynamicLearnedSweepError,
    build_mlx_dynamic_learned_sweep_plan,
    render_mlx_dynamic_learned_sweep_markdown,
)
from tac.optimization.mlx_dynamic_sweep_observations import (  # noqa: E402
    load_observation_rows,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-payload", type=Path, required=True)
    parser.add_argument("--observation-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json-out", type=Path)
    parser.add_argument("--replan-json-out", type=Path)
    parser.add_argument("--replan-md-out", type=Path)
    parser.add_argument("--incumbent-score", type=float)
    parser.add_argument("--candidate-id")
    parser.add_argument("--sweep-config-id", default=DEFAULT_SWEEP_CONFIG_ID)
    parser.add_argument("--optimization-pass-id", default=DEFAULT_OPTIMIZATION_PASS_ID)
    parser.add_argument("--allow-duplicate-observation", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def _selection_policy_value(
    plan: dict[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    policy = plan.get("selection_policy")
    return policy.get(key, default) if isinstance(policy, dict) else default


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = load_json_object(args.plan)
        candidate_payload = load_json_object(args.candidate_payload)
        observation = append_distortion_axis_probe_feedback_observation(
            plan=plan,
            candidate_payload=candidate_payload,
            output_path=args.observation_jsonl,
            sweep_config_id=args.sweep_config_id,
            optimization_pass_id=args.optimization_pass_id,
            candidate_id=args.candidate_id,
            allow_duplicate_observation=args.allow_duplicate_observation,
        )
        replan = None
        if args.replan_json_out is not None:
            if args.incumbent_score is None:
                raise DistortionAxisProbeLearnedSweepFeedbackError(
                    "--incumbent-score is required with --replan-json-out"
                )
            observations = load_observation_rows(args.observation_jsonl)
            replan = build_mlx_dynamic_learned_sweep_plan(
                incumbent_score=float(args.incumbent_score),
                candidate_payloads=[candidate_payload],
                execution_configs=[
                    row for row in plan.get("execution_configs", []) if isinstance(row, dict)
                ],
                optimization_passes=[
                    row for row in plan.get("optimization_passes", []) if isinstance(row, dict)
                ],
                observations=observations,
                top_k=int(_selection_policy_value(plan, "top_k", default=32)),
                per_pass_top_k=_selection_policy_value(plan, "per_pass_top_k"),
                lcb_z=float(_selection_policy_value(plan, "lcb_z", default=1.0)),
                expected_improvement_weight=float(
                    _selection_policy_value(
                        plan,
                        "expected_improvement_weight",
                        default=1.0,
                    )
                ),
                exploration_weight=float(
                    _selection_policy_value(plan, "exploration_weight", default=1.0)
                ),
                source_artifacts={
                    "source_plan": str(args.plan),
                    "candidate_payload": str(args.candidate_payload),
                    "observation_jsonl": str(args.observation_jsonl),
                },
            )
            write_json_artifact(
                args.replan_json_out,
                replan,
                allow_overwrite=args.allow_overwrite,
            )
            if args.replan_md_out is not None:
                if args.replan_md_out.exists() and not args.allow_overwrite:
                    raise ArtifactWriteError(
                        f"{args.replan_md_out}: refusing to overwrite existing artifact"
                    )
                args.replan_md_out.parent.mkdir(parents=True, exist_ok=True)
                args.replan_md_out.write_text(
                    render_mlx_dynamic_learned_sweep_markdown(replan),
                    encoding="utf-8",
                )
        summary = build_feedback_summary(
            observation=observation,
            observation_jsonl=args.observation_jsonl,
            replan=replan,
        )
        if args.summary_json_out is not None:
            write_json_artifact(
                args.summary_json_out,
                summary,
                allow_overwrite=args.allow_overwrite,
            )
    except (
        ArtifactWriteError,
        OSError,
        json.JSONDecodeError,
        DistortionAxisProbeLearnedSweepFeedbackError,
        MLXDynamicLearnedSweepError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(dumps_json(summary), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
