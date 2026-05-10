#!/usr/bin/env python3
"""Promote one optimizer queue row after byte-closed exact-eval readiness checks.

This is a local custody gate, not a dispatcher and not a score promoter. It
does not create lane claims, launch remote jobs, or turn proxy evidence into a
rank claim. The output queue is suitable for exact-eval dispatch only after the
operator creates the required lane claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimizer.exact_readiness import (  # noqa: E402
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
    ExactReadinessError,
    json_dumps,
    promote_candidate_for_exact_eval,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--submission-dir", type=Path, default=None)
    parser.add_argument("--archive-manifest", type=Path, default=None)
    parser.add_argument("--lane-id", default=None)
    parser.add_argument(
        "--allow-source-blocker",
        action="append",
        default=[],
        help="Additional source dispatch_blocker string to clear after local custody checks.",
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=REPO_ROOT / ".omx" / "state" / "active_lane_dispatch_claims.md",
        help=(
            "Read-only lane-claim ledger check. Existing active same-lane "
            "claims block promotion."
        ),
    )
    parser.add_argument("--claim-ttl-hours", type=float, default=24.0)
    parser.add_argument(
        "--skip-active-claim-check",
        action="store_true",
        help=(
            "Skip read-only active-claim conflict check. This does not remove "
            "the dispatch-time claim requirement."
        ),
    )
    parser.add_argument(
        "--active-floor-archive-bytes",
        type=int,
        default=ACTIVE_FLOOR_ARCHIVE_BYTES,
    )
    parser.add_argument("--active-floor-score", type=float, default=ACTIVE_FLOOR_SCORE)
    parser.add_argument("--allow-above-active-floor-dispatch", action="store_true")
    parser.add_argument("--operator-override-reason", default=None)
    args = parser.parse_args(argv)

    if args.allow_above_active_floor_dispatch and not args.operator_override_reason:
        print(
            "FATAL: --allow-above-active-floor-dispatch requires "
            "--operator-override-reason",
            file=sys.stderr,
        )
        return 2

    try:
        result = promote_candidate_for_exact_eval(
            args.queue,
            args.candidate_id,
            repo_root=args.repo_root,
            submission_dir=args.submission_dir,
            archive_manifest_path=args.archive_manifest,
            lane_id=args.lane_id,
            active_floor_archive_bytes=args.active_floor_archive_bytes,
            active_floor_score=args.active_floor_score,
            allow_above_active_floor_dispatch=args.allow_above_active_floor_dispatch,
            operator_override_reason=args.operator_override_reason,
            extra_clearable_source_blockers=args.allow_source_blocker,
            dispatch_claims_path=None
            if args.skip_active_claim_check
            else args.dispatch_claims_path,
            claim_ttl_hours=args.claim_ttl_hours,
        )
    except ExactReadinessError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    report = result["report"]
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json_dumps(report), encoding="utf-8")

    promoted_queue = result["promoted_queue"]
    if promoted_queue is None:
        print(
            "FATAL: candidate is not exact-eval dispatch ready:\n  - "
            + "\n  - ".join(report["blockers"][:40]),
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json_dumps(promoted_queue), encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"(candidate_id={args.candidate_id}, dispatch_ready_count=1, "
        "score_claim=false)"
    )
    print(
        "next required action before GPU/provider launch: "
        "tools/claim_lane_dispatch.py claim ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
