#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for the recursive adversarial review ledger.

Canonical entry per .omx/research/reusable_recursive_adversarial_review_canonical_design_20260517.md.
Wraps tac.recursive_adversarial_review per Catalog #265 canonical-helper-routing.

Examples:
    # Register bundle + return bundle_id
    .venv/bin/python tools/run_recursive_adversarial_review.py \\
        --scope-paths a.md b.md c.md --start-bundle-only

    # Record a round whose findings are pre-built in a JSONL file
    .venv/bin/python tools/run_recursive_adversarial_review.py \\
        --bundle-id <id> --round 1 --rotation Z_fresh_eyes \\
        --findings-jsonl /tmp/round1.jsonl --verdict PROCEED_WITH_REVISIONS

    # Query counter
    .venv/bin/python tools/run_recursive_adversarial_review.py \\
        --bundle-id <id> --query-counter --json
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from tac.recursive_adversarial_review import (
    SEAL_THRESHOLD,
    RecursiveReviewRound,
    ReviewFinding,
    append_round_locked,
    clean_pass_counter_for_bundle,
    compute_bundle_id,
    compute_scope_content_sha256,
    latest_round_by_bundle_id,
    query_rounds_by_bundle_id,
    query_unresolved_critical_findings,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(obj: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, sort_keys=True))
    else:
        for k, v in obj.items():
            print(f"{k}: {v}")


def _load_findings_jsonl(path: Path) -> tuple[ReviewFinding, ...]:
    out: list[ReviewFinding] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(ReviewFinding(**row))
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--scope-paths", nargs="+", help="Relative repo paths comprising the bundle")
    parser.add_argument("--bundle-id", help="Pre-computed bundle id (skip --scope-paths)")
    parser.add_argument("--round", type=int, help="Round number (1, 2, 3, ...)")
    parser.add_argument("--rotation", help="Council rotation name (e.g., Z_fresh_eyes)")
    parser.add_argument(
        "--attendees", nargs="*", default=None, help="Explicit attendee list; defaults from rotation name"
    )
    parser.add_argument("--reviewer-agent", default="claude-in-context", help="Reviewer identity")
    parser.add_argument("--verdict", help="PROCEED / PROCEED_WITH_REVISIONS / DEFER / KILL_CANDIDATE")
    parser.add_argument("--findings-jsonl", type=Path, help="Read findings from this JSONL file")
    parser.add_argument("--start-bundle-only", action="store_true", help="Just compute + return bundle id")
    parser.add_argument("--query-counter", action="store_true", help="Print clean-pass counter and exit")
    parser.add_argument("--query-latest", action="store_true", help="Print latest round and exit")
    parser.add_argument("--repo-root", default=".", help="Repo root (default cwd)")
    parser.add_argument("--ledger-path", type=Path, default=None, help="Override canonical ledger path")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if counter_after < 3 after appending")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)

    # Bundle-only registration
    if args.start_bundle_only:
        if not args.scope_paths:
            parser.error("--start-bundle-only requires --scope-paths")
        bundle_id = compute_bundle_id(args.scope_paths)
        content_sha = compute_scope_content_sha256(args.scope_paths, repo_root=repo_root)
        counter = clean_pass_counter_for_bundle(
            bundle_id,
            path=args.ledger_path,
            scope_content_sha256=content_sha,
        )
        _emit(
            {
                "bundle_id": bundle_id,
                "scope_content_sha256": content_sha,
                "scope_paths": sorted(args.scope_paths),
                "counter": counter,
                "sealed": counter >= SEAL_THRESHOLD,
            },
            as_json=args.json,
        )
        return 0

    # Resolve bundle_id
    if args.bundle_id:
        bundle_id = args.bundle_id
    elif args.scope_paths:
        bundle_id = compute_bundle_id(args.scope_paths)
    else:
        parser.error("require --bundle-id or --scope-paths")

    if args.query_counter:
        content_sha = (
            compute_scope_content_sha256(args.scope_paths, repo_root=repo_root)
            if args.scope_paths
            else None
        )
        counter = clean_pass_counter_for_bundle(
            bundle_id,
            path=args.ledger_path,
            scope_content_sha256=content_sha,
        )
        latest = latest_round_by_bundle_id(bundle_id, path=args.ledger_path)
        critical = query_unresolved_critical_findings(bundle_id, path=args.ledger_path)
        _emit(
            {
                "bundle_id": bundle_id,
                "counter": counter,
                "content_freshness_checked": content_sha is not None,
                "content_matches_latest": (
                    latest is not None
                    and content_sha is not None
                    and latest.get("scope_content_sha256") == content_sha
                ),
                "sealed": counter >= SEAL_THRESHOLD,
                "unresolved_critical_count": len(critical),
            },
            as_json=args.json,
        )
        return 0 if counter >= SEAL_THRESHOLD or not args.strict else 1

    if args.query_latest:
        latest = latest_round_by_bundle_id(bundle_id, path=args.ledger_path)
        if latest is None:
            _emit({"bundle_id": bundle_id, "latest": None}, as_json=args.json)
            return 0
        if args.json:
            print(json.dumps(latest, sort_keys=True))
        else:
            print(json.dumps(latest, indent=2, sort_keys=True))
        return 0

    # Append a new round
    required = ("round", "rotation", "verdict")
    missing = [f for f in required if getattr(args, f) is None]
    if missing:
        parser.error(f"missing required flags for new round: {missing}")
    if not args.scope_paths:
        parser.error("require --scope-paths to compute scope_content_sha256")
    findings = _load_findings_jsonl(args.findings_jsonl) if args.findings_jsonl else ()
    attendees = tuple(args.attendees) if args.attendees else (args.rotation,)
    content_sha = compute_scope_content_sha256(args.scope_paths, repo_root=repo_root)
    counter_before = clean_pass_counter_for_bundle(
        bundle_id,
        path=args.ledger_path,
        scope_content_sha256=content_sha,
    )
    non_confirms = [f for f in findings if f.severity != "CONFIRMS"]
    counter_after = 0 if non_confirms else counter_before + 1
    prior_rounds = query_rounds_by_bundle_id(bundle_id, path=args.ledger_path)
    related = tuple(r.get("review_id") for r in prior_rounds if r.get("review_id"))
    record = RecursiveReviewRound(
        review_id=uuid.uuid4().hex[:12],
        bundle_id=bundle_id,
        scope_paths=tuple(sorted(args.scope_paths)),
        scope_content_sha256=content_sha,
        round_number=int(args.round),
        council_rotation=args.rotation,
        council_attendees=attendees,
        findings=findings,
        verdict=args.verdict,
        counter_before=counter_before,
        counter_after=counter_after,
        reviewed_at_utc=_utc_now(),
        reviewer_agent=args.reviewer_agent,
        related_round_ids=related,
    )
    append_round_locked(record, path=args.ledger_path)
    _emit(
        {
            "bundle_id": bundle_id,
            "review_id": record.review_id,
            "counter_before": counter_before,
            "counter_after": counter_after,
            "sealed": counter_after >= SEAL_THRESHOLD,
            "verdict": args.verdict,
            "findings_count": len(findings),
        },
        as_json=args.json,
    )
    if args.strict and counter_after < SEAL_THRESHOLD:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
