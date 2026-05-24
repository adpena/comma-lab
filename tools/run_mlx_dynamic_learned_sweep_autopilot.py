#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded local MLX learned-sweep actuation/replan cycles."""

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

from tac.optimization.mlx_dynamic_learned_sweep import file_sha256  # noqa: E402
from tac.optimization.mlx_dynamic_learned_sweep import (  # noqa: E402
    load_json_object as load_candidate_payload,
)
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (  # noqa: E402
    load_json_object,
)
from tac.optimization.mlx_dynamic_learned_sweep_local_autopilot import (  # noqa: E402
    MLXDynamicLearnedSweepLocalAutopilotError,
    run_local_mlx_sweep_autopilot,
)
from tac.optimization.mlx_dynamic_sweep_observations import json_text  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--candidate-payload",
        action="append",
        default=[],
        type=Path,
        required=True,
        help="Learned-sweep candidate payload. May repeat.",
    )
    parser.add_argument("--incumbent-score", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json-out", type=Path)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--max-new-observations", type=int, default=1)
    parser.add_argument("--rows-per-replan", type=int, default=1)
    parser.add_argument("--sweep-config-id", default="mlx_local_response")
    parser.add_argument("--optimization-pass-id")
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Restrict execution to one candidate_id. May repeat.",
    )
    parser.add_argument(
        "--queue-candidate-id",
        action="append",
        default=[],
        help="Restrict execution to one queue_candidate_id. May repeat.",
    )
    parser.add_argument("--source-artifact-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--allow-gpu-research-signal", action="store_true")
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--replan-top-k", type=int)
    parser.add_argument("--replan-per-pass-top-k", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def _source_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
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
        summary = run_local_mlx_sweep_autopilot(
            initial_plan=load_json_object(args.plan),
            selection=load_json_object(args.selection),
            candidate_payloads=[
                load_candidate_payload(path) for path in args.candidate_payload
            ],
            incumbent_score=args.incumbent_score,
            output_dir=args.output_dir,
            observation_jsonl=args.observation_jsonl,
            max_iterations=args.max_iterations,
            max_new_observations=args.max_new_observations,
            rows_per_replan=args.rows_per_replan,
            sweep_config_id=args.sweep_config_id,
            optimization_pass_id=args.optimization_pass_id,
            candidate_ids=args.candidate_id or None,
            queue_candidate_ids=args.queue_candidate_id or None,
            source_artifact_root=args.source_artifact_root,
            device_type=args.device,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
            batch_pairs=args.batch_pairs,
            progress_every=args.progress_every,
            max_seconds=args.max_seconds,
            replan_top_k=args.replan_top_k,
            replan_per_pass_top_k=args.replan_per_pass_top_k,
            source_artifacts=_source_artifacts(args),
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
        MLXDynamicLearnedSweepLocalAutopilotError,
        ValueError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "cycle_count": summary["cycle_count"],
                "executed_row_count": summary["executed_row_count"],
                "new_observation_row_count": summary["new_observation_row_count"],
                "final_observation_row_count": summary["final_observation_row_count"],
                "observation_jsonl": str(args.observation_jsonl),
                "summary_json_out": None
                if args.summary_json_out is None
                else str(args.summary_json_out),
                "stopping_reason": summary["stopping_reason"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
