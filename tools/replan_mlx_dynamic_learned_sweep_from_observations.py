#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replan MLX dynamic learned-sweep work from observation JSONL files."""

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

from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    FALSE_AUTHORITY,
    file_sha256,
)
from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    load_json_object as load_candidate_payload,
)
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (  # noqa: E402
    MLXDynamicLearnedSweepLocalActuatorError,
    load_json_object,
    replan_after_local_actuation,
)
from tac.optimization.mlx_dynamic_sweep_observations import (  # noqa: E402
    deduplicate_observation_rows,
    json_text,
    load_observation_rows,
    summarize_observations,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

SCHEMA = "mlx_dynamic_learned_sweep_replan_from_observations.v1"
TOOL = "tools/replan_mlx_dynamic_learned_sweep_from_observations.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--candidate-payload",
        action="append",
        default=[],
        type=Path,
        help="Candidate payload consumed by the learned-sweep planner. May repeat.",
    )
    parser.add_argument(
        "--observation-jsonl",
        action="append",
        default=[],
        type=Path,
        help="Observation JSONL produced by local MLX actuation. May repeat.",
    )
    parser.add_argument("--incumbent-score", type=float, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--summary-json-out", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--per-pass-top-k", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def _require_paths(paths: list[Path], *, label: str) -> None:
    if not paths:
        raise MLXDynamicLearnedSweepLocalActuatorError(f"{label} is required")
    for index, path in enumerate(paths):
        if not path.is_file():
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"{label}[{index}] does not exist: {path}"
            )


def _require_output_available(path: Path | None, *, allow_overwrite: bool) -> None:
    if path is None:
        return
    if path.exists() and not allow_overwrite:
        raise MLXDynamicLearnedSweepLocalActuatorError(
            f"refusing to overwrite existing artifact: {path}"
        )


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _summary(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    raw_observations: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    replanned: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "producer": TOOL,
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "observation_only": True,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "allowed_use": "local_mlx_observation_ledger_replanning_only",
        "source_plan_schema": plan.get("schema"),
        "source_artifacts": {
            "plan": _artifact(args.plan),
            "candidate_payloads": [_artifact(path) for path in args.candidate_payload],
            "observation_jsonl": [_artifact(path) for path in args.observation_jsonl],
        },
        "observation_jsonl_count": len(args.observation_jsonl),
        "raw_observation_row_count": len(raw_observations),
        "observation_row_count": len(observations),
        "duplicate_observation_row_count": len(raw_observations) - len(observations),
        "observation_summary": summarize_observations(observations),
        "replan": {
            "json_out": str(args.json_out),
            "md_out": None if args.md_out is None else str(args.md_out),
            "schema": replanned.get("schema"),
            "ranked_row_count": replanned["summary"]["ranked_row_count"],
            "local_ready_row_count": replanned["summary"]["local_ready_row_count"],
            "suppressed_observed_row_count": replanned["summary"][
                "suppressed_observed_row_count"
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _require_paths(args.candidate_payload, label="--candidate-payload")
        _require_paths(args.observation_jsonl, label="--observation-jsonl")
        if not args.plan.is_file():
            raise MLXDynamicLearnedSweepLocalActuatorError(
                f"--plan does not exist: {args.plan}"
            )
        _require_output_available(args.json_out, allow_overwrite=args.allow_overwrite)
        _require_output_available(args.md_out, allow_overwrite=args.allow_overwrite)
        _require_output_available(
            args.summary_json_out,
            allow_overwrite=args.allow_overwrite,
        )
        plan = load_json_object(args.plan)
        raw_observations: list[dict[str, Any]] = []
        for path in args.observation_jsonl:
            raw_observations.extend(load_observation_rows(path))
        observations = deduplicate_observation_rows(raw_observations)
        replanned = replan_after_local_actuation(
            incumbent_score=float(args.incumbent_score),
            candidate_payloads=[
                load_candidate_payload(path) for path in args.candidate_payload
            ],
            source_plan=plan,
            observation_jsonl_paths=args.observation_jsonl,
            json_out=args.json_out,
            md_out=args.md_out,
            top_k=args.top_k,
            per_pass_top_k=args.per_pass_top_k,
        )
        summary = _summary(
            args=args,
            plan=plan,
            raw_observations=raw_observations,
            observations=observations,
            replanned=replanned,
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
        MLXDynamicLearnedSweepLocalActuatorError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": SCHEMA,
                "observation_jsonl_count": len(args.observation_jsonl),
                "raw_observation_row_count": len(raw_observations),
                "observation_row_count": len(observations),
                "duplicate_observation_row_count": len(raw_observations)
                - len(observations),
                "replan_json_out": str(args.json_out),
                "summary_json_out": None
                if args.summary_json_out is None
                else str(args.summary_json_out),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
