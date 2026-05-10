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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimizer.exact_readiness import (  # noqa: E402
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
    ExactReadinessError,
    active_claim_conflicts,
    as_bool,
    find_candidate,
    json_dumps,
    promote_candidate_for_exact_eval,
    read_json,
    terminal_claim_result_conflicts,
)


def _resolve_for_guard(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _refuses_output_inside_submission(
    output_paths: list[Path],
    *,
    submission_dir_text: object,
    repo_root: Path,
) -> str | None:
    if not isinstance(submission_dir_text, str) or not submission_dir_text:
        return None
    submission_dir = _resolve_for_guard(Path(submission_dir_text), repo_root)
    for output_path in output_paths:
        resolved_output = _resolve_for_guard(output_path, repo_root)
        if _is_relative_to(resolved_output, submission_dir):
            return (
                "output_inside_submission_dir_would_mutate_runtime_tree:"
                f"{resolved_output}"
            )
    return None


def _dedupe_blockers(blockers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for blocker in blockers:
        if blocker in seen:
            continue
        seen.add(blocker)
        out.append(blocker)
    return out


def _claim_lane_aliases(lane_id: str) -> tuple[str, ...]:
    """Return canonical claim ids plus the legacy lane_ alias spelling."""

    clean = lane_id.strip()
    if not clean:
        return ()
    aliases = [clean]
    if clean.startswith("lane_"):
        suffix = clean.removeprefix("lane_")
        if suffix:
            aliases.append(suffix)
    else:
        aliases.append(f"lane_{clean}")
    return tuple(dict.fromkeys(aliases))


def _source_lane_id(
    queue_path: Path,
    candidate_id: str,
    *,
    lane_id_override: str | None,
) -> str | None:
    if isinstance(lane_id_override, str) and lane_id_override.strip():
        return lane_id_override.strip()
    try:
        queue_payload = read_json(queue_path)
    except (OSError, ValueError):
        return None
    if not isinstance(queue_payload, dict):
        return None
    row, _source_list = find_candidate(queue_payload, candidate_id)
    if row is None:
        return None
    lane_id = row.get("lane_id")
    return lane_id.strip() if isinstance(lane_id, str) and lane_id.strip() else None


def _active_claim_alias_blockers(
    lane_id: str | None,
    *,
    dispatch_claims_path: Path,
    claim_ttl_hours: float,
) -> list[str]:
    if not isinstance(lane_id, str) or not lane_id.strip():
        return []
    blockers: list[str] = []
    for claim_lane_id in _claim_lane_aliases(lane_id):
        blockers.extend(
            active_claim_conflicts(
                claim_lane_id,
                dispatch_claims_path=dispatch_claims_path,
                ttl_hours=claim_ttl_hours,
            )
        )
    return _dedupe_blockers(blockers)


def _first_promoted_row(promoted_queue: object) -> Mapping[str, Any] | None:
    if not isinstance(promoted_queue, Mapping):
        return None
    for key in ("dispatch_ready", "top_k"):
        rows = promoted_queue.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                return row
    return None


def _terminal_claim_alias_blockers(
    promoted_queue: object,
    *,
    fallback_lane_id: str | None,
    report: Mapping[str, Any],
    dispatch_claims_path: Path,
    active_floor_score: float | None,
) -> list[str]:
    row = _first_promoted_row(promoted_queue)
    facts = report.get("facts") if isinstance(report.get("facts"), Mapping) else {}
    lane_id = row.get("lane_id") if row is not None else None
    if not isinstance(lane_id, str) or not lane_id.strip():
        lane_id = fallback_lane_id
    if not isinstance(lane_id, str) or not lane_id.strip():
        return []

    archive_sha = row.get("archive_sha256") if row is not None else None
    if not isinstance(archive_sha, str):
        archive_sha = facts.get("archive_sha256") if isinstance(facts, Mapping) else None
    runtime_tree_sha = row.get("runtime_tree_sha256") if row is not None else None
    if not isinstance(runtime_tree_sha, str):
        runtime_tree_sha = (
            facts.get("runtime_tree_sha256") if isinstance(facts, Mapping) else None
        )
    runtime_changed = (
        as_bool(row.get("score_affecting_runtime_changed")) if row is not None else None
    )

    blockers: list[str] = []
    for claim_lane_id in _claim_lane_aliases(lane_id):
        blockers.extend(
            terminal_claim_result_conflicts(
                claim_lane_id,
                archive_sha,
                dispatch_claims_path=dispatch_claims_path,
                active_floor_score=active_floor_score,
                runtime_tree_sha256=runtime_tree_sha
                if isinstance(runtime_tree_sha, str)
                else None,
                score_affecting_runtime_changed=runtime_changed,
            )
        )
    return _dedupe_blockers(blockers)


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
            "Deprecated: exact-eval dispatch promotion now always requires "
            "the read-only active-claim conflict check."
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

    if args.skip_active_claim_check:
        print(
            "FATAL: --skip-active-claim-check is disabled for exact-eval "
            "dispatch promotion; use a research-only/non-dispatch tool path "
            "instead of writing an exact-ready queue.",
            file=sys.stderr,
        )
        return 2

    dispatch_claims_path = _resolve_for_guard(args.dispatch_claims_path, args.repo_root)
    if not dispatch_claims_path.is_file():
        print(
            "FATAL: dispatch claim ledger missing or not a file: "
            f"{dispatch_claims_path}",
            file=sys.stderr,
        )
        return 2

    if args.allow_above_active_floor_dispatch and not args.operator_override_reason:
        print(
            "FATAL: --allow-above-active-floor-dispatch requires "
            "--operator-override-reason",
            file=sys.stderr,
        )
        return 2

    effective_lane_id = _source_lane_id(
        args.queue,
        args.candidate_id,
        lane_id_override=args.lane_id,
    )
    active_claim_blockers = _active_claim_alias_blockers(
        effective_lane_id,
        dispatch_claims_path=dispatch_claims_path,
        claim_ttl_hours=args.claim_ttl_hours,
    )
    if active_claim_blockers:
        print(
            "FATAL: active dispatch claim check blocked exact-eval promotion:\n  - "
            + "\n  - ".join(active_claim_blockers[:40]),
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
            dispatch_claims_path=dispatch_claims_path,
            claim_ttl_hours=args.claim_ttl_hours,
        )
    except ExactReadinessError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    report = result["report"]
    promoted_queue = result["promoted_queue"]
    terminal_alias_blockers = _terminal_claim_alias_blockers(
        promoted_queue,
        fallback_lane_id=effective_lane_id,
        report=report,
        dispatch_claims_path=dispatch_claims_path,
        active_floor_score=args.active_floor_score,
    )
    if terminal_alias_blockers:
        report["blockers"] = _dedupe_blockers(
            list(report.get("blockers") or []) + terminal_alias_blockers
        )
        report["ready_for_exact_eval_dispatch"] = False
        promoted_queue = None

    output_paths = [args.output]
    if args.report_output is not None:
        output_paths.append(args.report_output)
    output_guard = _refuses_output_inside_submission(
        output_paths,
        submission_dir_text=(report.get("facts") or {}).get("submission_dir")
        if isinstance(report.get("facts"), dict)
        else None,
        repo_root=args.repo_root,
    )
    if output_guard:
        print(f"FATAL: {output_guard}", file=sys.stderr)
        return 2

    if promoted_queue is None:
        print(
            "FATAL: candidate is not exact-eval dispatch ready:\n  - "
            + "\n  - ".join(report["blockers"][:40]),
            file=sys.stderr,
        )
        return 2

    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json_dumps(report), encoding="utf-8")

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
