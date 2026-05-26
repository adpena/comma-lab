#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a grouped targeted component correction materialization request."""

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
    build_frontier_targeted_component_correction_materialization_request,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targeted-component-correction-response-harvest",
        required=True,
        type=Path,
    )
    parser.add_argument("--materialization-request-id", required=True)
    parser.add_argument("--request-out", required=True, type=Path)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--family-limit-per-candidate", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        response_harvest_path = args.targeted_component_correction_response_harvest
        if not response_harvest_path.is_absolute():
            response_harvest_path = REPO_ROOT / response_harvest_path
        response_harvest = json.loads(
            response_harvest_path.read_text(encoding="utf-8")
        )
        if not isinstance(response_harvest, dict):
            raise FrontierRateAttackFeedbackError(
                "targeted component correction response harvest must be a JSON object"
            )
        request = build_frontier_targeted_component_correction_materialization_request(
            targeted_component_correction_response_harvest=response_harvest,
            materialization_request_id=args.materialization_request_id,
            candidate_limit=args.candidate_limit,
            family_limit_per_candidate=args.family_limit_per_candidate,
        )
        request_out = args.request_out
        if not request_out.is_absolute():
            request_out = REPO_ROOT / request_out
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if request_out.exists() and args.overwrite:
            existing_text = request_out.read_text(encoding="utf-8")
            next_text = json_text(request)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(request_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                request_out,
                request,
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
        print(
            "FATAL: targeted component correction materialization request failed: "
            f"{exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": (
                    "frontier_rate_attack_targeted_component_correction_"
                    "materialization_request_cli_result.v1"
                ),
                "materialization_request_id": args.materialization_request_id,
                "request_out": str(args.request_out),
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
