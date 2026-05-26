#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned targeted component correction work order."""

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
    build_frontier_targeted_component_correction_work_order,
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
        "--targeted-component-correction-acquisition",
        required=True,
        type=Path,
    )
    parser.add_argument("--acquisition-id", required=True)
    parser.add_argument("--work-order-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        acquisition_path = args.targeted_component_correction_acquisition
        if not acquisition_path.is_absolute():
            acquisition_path = REPO_ROOT / acquisition_path
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        if not isinstance(acquisition, dict):
            raise FrontierRateAttackFeedbackError(
                "targeted component correction acquisition must be a JSON object"
            )
        work_order = build_frontier_targeted_component_correction_work_order(
            targeted_component_correction_acquisition=acquisition,
            acquisition_id=args.acquisition_id,
            repo_root=REPO_ROOT,
        )
        work_order_out = args.work_order_out
        if not work_order_out.is_absolute():
            work_order_out = REPO_ROOT / work_order_out
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if work_order_out.exists() and args.overwrite:
            existing_text = work_order_out.read_text(encoding="utf-8")
            next_text = json_text(work_order)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(work_order_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                work_order_out,
                work_order,
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
            f"FATAL: targeted component correction work order failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": (
                    "frontier_rate_attack_targeted_component_correction_"
                    "work_order_cli_result.v1"
                ),
                "acquisition_id": args.acquisition_id,
                "work_order_out": str(args.work_order_out),
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
