#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build targeted drop-many stage inputs from a chain stage plan."""

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

from comma_lab.scheduler.experiment_queue import ExperimentQueueError  # noqa: E402
from comma_lab.scheduler.frontier_rate_attack_feedback import (  # noqa: E402
    FrontierRateAttackFeedbackError,
    build_frontier_targeted_drop_many_stage_inputs,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation-chain-stage-plan", required=True, type=Path)
    parser.add_argument("--stage-inputs-out", required=True, type=Path)
    parser.add_argument("--stage-id", default="scorer_sensitive_operation_selection")
    parser.add_argument("--selector-pareto", type=Path)
    parser.add_argument("--pair-frame-geometry-lattice-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--drop-many-counts", default="3,4,6,8,12,16")
    parser.add_argument("--max-drop-many", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stage_plan_path = args.operation_chain_stage_plan
        if not stage_plan_path.is_absolute():
            stage_plan_path = REPO_ROOT / stage_plan_path
        stage_plan = json.loads(stage_plan_path.read_text(encoding="utf-8"))
        if not isinstance(stage_plan, dict):
            raise FrontierRateAttackFeedbackError(
                "operation chain stage plan must be a JSON object"
            )
        stage_inputs = build_frontier_targeted_drop_many_stage_inputs(
            repo_root=REPO_ROOT,
            operation_chain_stage_plan=stage_plan,
            stage_id=args.stage_id,
            selector_pareto=args.selector_pareto,
            pair_frame_geometry_lattice=args.pair_frame_geometry_lattice_json,
            output_dir=args.output_dir,
            drop_many_counts=args.drop_many_counts,
            max_drop_many=args.max_drop_many,
        )
        out = args.stage_inputs_out
        if not out.is_absolute():
            out = REPO_ROOT / out
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if out.exists() and args.overwrite:
            existing_text = out.read_text(encoding="utf-8")
            next_text = json_text(stage_inputs)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                out,
                stage_inputs,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        ExperimentQueueError,
        FrontierRateAttackFeedbackError,
        OSError,
        ValueError,
    ) as exc:
        print(f"FATAL: targeted drop-many stage input build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": (
                    "frontier_rate_attack_targeted_drop_many_stage_inputs_"
                    "cli_result.v1"
                ),
                "stage_id": args.stage_id,
                "stage_inputs_out": str(args.stage_inputs_out),
                "local_plan_queueable": stage_inputs.get("local_plan_queueable")
                is True,
                "pairset_acquisition_path": stage_inputs.get(
                    "pairset_acquisition_path"
                ),
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
