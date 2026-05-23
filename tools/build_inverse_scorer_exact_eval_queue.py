#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an exact-readiness source queue from a verified IAS1 chain manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.inverse_scorer_exact_eval_queue import (  # noqa: E402
    DEFAULT_LANE_ID,
    InverseScorerExactEvalQueueError,
    build_inverse_scorer_exact_eval_source_queue,
    dumps_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-manifest", type=Path, required=True)
    parser.add_argument("--runtime-submission-dir", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--archive-manifest-output", type=Path, required=True)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    try:
        result = build_inverse_scorer_exact_eval_source_queue(
            chain_manifest_path=args.chain_manifest,
            runtime_submission_dir=args.runtime_submission_dir,
            archive_manifest_path=args.archive_manifest_output,
            repo_root=args.repo_root,
            candidate_id=args.candidate_id,
            lane_id=args.lane_id,
        )
    except (OSError, ValueError, InverseScorerExactEvalQueueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    args.archive_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.archive_manifest_output.write_text(dumps_json(result.archive_manifest), encoding="utf-8")
    args.output_queue.parent.mkdir(parents=True, exist_ok=True)
    args.output_queue.write_text(dumps_json(result.queue), encoding="utf-8")
    candidate_id = result.queue["top_k"][0]["candidate_id"]
    print(
        f"wrote {args.output_queue} and {args.archive_manifest_output} "
        f"(candidate_id={candidate_id}, score_claim=false)"
    )
    print(
        "next: tools/promote_optimizer_candidate_for_exact_eval.py "
        f"--queue {args.output_queue} --candidate-id {candidate_id} "
        "--output <exact-ready-queue.json> "
        f"--archive-manifest {args.archive_manifest_output} "
        f"--submission-dir {args.runtime_submission_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
