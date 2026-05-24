#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute local MLX dynamic learned-sweep rows and append observations."""

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
    file_sha256,
)
from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    load_json_object as load_candidate_payload,
)
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (  # noqa: E402
    MLXDynamicLearnedSweepLocalActuatorError,
    execute_local_mlx_sweep_rows,
    load_json_object,
    replan_after_local_actuation,
)
from tac.optimization.mlx_dynamic_sweep_observations import json_text  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json-out", type=Path)
    parser.add_argument("--observation-jsonl", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--sweep-config-id", default="mlx_local_response")
    parser.add_argument("--optimization-pass-id")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Restrict execution to candidate_id. May repeat.",
    )
    parser.add_argument(
        "--queue-candidate-id",
        action="append",
        default=[],
        help="Restrict execution to queue_candidate_id. May repeat.",
    )
    parser.add_argument("--source-artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--allow-gpu-research-signal", action="store_true")
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--allow-duplicate-observation", action="store_true")
    parser.add_argument(
        "--replan-json-out",
        type=Path,
        help="Optional learned-sweep plan path to write after appended observations.",
    )
    parser.add_argument("--replan-md-out", type=Path)
    parser.add_argument(
        "--candidate-payload",
        action="append",
        default=[],
        type=Path,
        help="Candidate payloads needed when --replan-json-out is set. May repeat.",
    )
    parser.add_argument(
        "--incumbent-score",
        type=float,
        help="Required when --replan-json-out is set.",
    )
    parser.add_argument("--replan-top-k", type=int)
    parser.add_argument("--replan-per-pass-top-k", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def _source_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for label, path in (
        ("plan", args.plan),
        ("selection", args.selection),
        ("observation_jsonl", args.observation_jsonl),
    ):
        out[label] = {
            "path": str(path),
            "sha256": file_sha256(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    out["candidate_payloads"] = [
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in args.candidate_payload
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.replan_json_out is not None:
            if args.incumbent_score is None:
                raise MLXDynamicLearnedSweepLocalActuatorError(
                    "--incumbent-score is required with --replan-json-out"
                )
            if not args.candidate_payload:
                raise MLXDynamicLearnedSweepLocalActuatorError(
                    "--candidate-payload is required with --replan-json-out"
                )
        plan = load_json_object(args.plan)
        selection = load_json_object(args.selection)
        summary = execute_local_mlx_sweep_rows(
            plan=plan,
            selection=selection,
            output_dir=args.output_dir,
            observation_jsonl=args.observation_jsonl,
            max_rows=args.max_rows,
            sweep_config_id=args.sweep_config_id,
            optimization_pass_id=args.optimization_pass_id,
            candidate_ids=args.candidate_id or None,
            queue_candidate_ids=args.queue_candidate_id or None,
            source_artifact_root=args.source_artifact_root,
            device_type=args.device,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
            batch_pairs=args.batch_pairs,
            progress_every=args.progress_every,
            allow_duplicate_observation=args.allow_duplicate_observation,
        )
        summary["source_artifacts"] = _source_artifacts(args)
        replan = None
        if args.replan_json_out is not None:
            replan = replan_after_local_actuation(
                incumbent_score=float(args.incumbent_score),
                candidate_payloads=[
                    load_candidate_payload(path) for path in args.candidate_payload
                ],
                source_plan=plan,
                observation_jsonl_paths=[args.observation_jsonl],
                json_out=args.replan_json_out,
                md_out=args.replan_md_out,
                top_k=args.replan_top_k,
                per_pass_top_k=args.replan_per_pass_top_k,
            )
            summary["replan"] = {
                "json_out": str(args.replan_json_out),
                "md_out": None if args.replan_md_out is None else str(args.replan_md_out),
                "ranked_row_count": replan["summary"]["ranked_row_count"],
                "local_ready_row_count": replan["summary"]["local_ready_row_count"],
                "suppressed_observed_row_count": replan["summary"][
                    "suppressed_observed_row_count"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
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
                "executed_row_count": summary["executed_row_count"],
                "observation_row_count": summary["observation_row_count"],
                "observation_jsonl": str(args.observation_jsonl),
                "summary_json_out": None
                if args.summary_json_out is None
                else str(args.summary_json_out),
                "replan_json_out": None
                if args.replan_json_out is None
                else str(args.replan_json_out),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
