#!/usr/bin/env python3
"""Run one custody-bound G120 rank-zero archive through public exact eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.witness_control.taskspace_single_stage_score_attempt_v1 import (
    RankZeroScoreAttemptConfigV1,
    SingleStageScoreAttemptError,
    run_rank_zero_score_attempt,
    write_blocker_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="SSD run root; required for cold start and crash resume",
    )
    parser.add_argument(
        "--g121-stage-ledger",
        type=Path,
        required=True,
        help="append-only G121 stage measurement ledger",
    )
    parser.add_argument(
        "--expected-g121-stage-ledger-sha256",
        required=True,
        help="exact immutable ledger file identity consumed by this attempt",
    )
    parser.add_argument(
        "--g121-attempt-identity-sha256",
        required=True,
        help="one explicit self-hashed COMPLETED retained G121 row",
    )
    parser.add_argument(
        "--expected-runtime-tree-sha256",
        required=True,
        help="G120 sealed public runtime content-tree identity",
    )
    parser.add_argument(
        "--video-names-file",
        type=Path,
        default=Path("upstream/public_test_video_names.txt"),
    )
    parser.add_argument(
        "--evaluator-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--authority-axis",
        choices=("macOS-CPU advisory", "contest-CPU", "contest-CUDA"),
        default="macOS-CPU advisory",
    )
    parser.add_argument(
        "--competitive-target",
        default="0.172",
        help="comparison only; never mutates the canonical pointer",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    config = RankZeroScoreAttemptConfigV1(
        run_id=args.run_id,
        resume_from=args.resume_from,
        g121_stage_ledger=args.g121_stage_ledger,
        expected_g121_stage_ledger_sha256=(
            args.expected_g121_stage_ledger_sha256
        ),
        g121_attempt_identity_sha256=(
            args.g121_attempt_identity_sha256
        ),
        expected_runtime_tree_sha256=(
            args.expected_runtime_tree_sha256
        ),
        evaluator_device=args.evaluator_device,
        video_names_file=args.video_names_file,
        authority_axis=args.authority_axis,
        competitive_target=args.competitive_target,
    )
    try:
        result = run_rank_zero_score_attempt(
            config=config,
            command=command,
        )
    except (
        OSError,
        ValueError,
        SingleStageScoreAttemptError,
    ) as exc:
        blocker = write_blocker_receipt(
            resume_from=args.resume_from,
            command=command,
            error=exc,
        )
        detail = f"; blocker={blocker}" if blocker is not None else ""
        print(f"REFUSE: {exc}{detail}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "final_receipt_path": str(result.final_receipt_path),
                "final_receipt_sha256": result.final_receipt_sha256,
                "submission_dir": str(result.submission_dir),
                "archive_bytes": result.archive_bytes,
                "archive_sha256": result.archive_sha256,
                "report_path": str(result.report_path),
                "d_pose": result.d_pose,
                "d_seg": result.d_seg,
                "recomputed_score_from_reported_8dp_components": (
                    result.recomputed_score
                ),
                "authority_axis": result.authority_axis,
                "explicit_single_stage_attempt": True,
                "pose_refit_run": False,
                "exhaustive_stage_coverage_claim": False,
                "pareto_claim": False,
                "cross_stage_winner_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
