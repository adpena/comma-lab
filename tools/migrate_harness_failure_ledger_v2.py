#!/usr/bin/env python3
"""Migrate ``.omx/state/harness_failure_ledger.jsonl`` to the canonical FailureEventV2
lifecycle (A1 of the harness-engineering crosswalk,
``.omx/research/harness_engineering_crosswalk_20260719_codex.md``).

WHAT it does — APPEND-ONLY, never rewrites history (Catalog #110/#113 HISTORICAL_PROVENANCE):

  * reads every JSON-object row (all writer generations: harness_failure.v1 +
    schemaless ``failure_class`` / ``class_id`` rows);
  * projects them into ONE canonical ``harness_failure.v2`` row per SEMANTIC class
    (the two known aliases collapsed: ``codex_probe_token_limit_death_incomplete_wip`` and
    ``dashboard_false_FAIL_at_init``);
  * derives the typed ``resolution_state`` from STRUCTURED markers only (never free prose —
    a prose-only closure lands as ``VERIFY_PENDING``, not ``CLOSED``);
  * appends those V2 rows AFTER the originals. The originals are preserved as provenance;
    each V2 row's ``migrated_from`` records the folded legacy span.

IDEMPOTENT: a class that already has a V2 row is skipped. Re-running after a partial run
only fills the gap. ``--dry-run`` (default) prints the plan and writes nothing; ``--apply``
appends.

VERIFY after ``--apply``:
    python tools/migrate_harness_failure_ledger_v2.py --verify
prints the canonical V2 summary (class count, per-class resolution_state) so the crosswalk's
falsifiable gate ("20 normalized classes, no '?', correct states for phantom-death / provider
lexical-trigger / review-spiral / SIGURG") can be asserted directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac import harness_failure_ledger as L  # noqa: E402


def _plan(path: Path | None) -> tuple[list, set[str]]:
    """Return (v2 projections needed, class_ids that already have a V2 row)."""
    rows = L.load_raw_rows(path)
    already = {ev.class_id for ev in L.load_failure_events_v2(path)}
    projections = [p for p in L.project_legacy_rows_to_v2(rows) if p.class_id not in already]
    return projections, already


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true",
                    help="append the V2 rows (default is a dry-run plan)")
    ap.add_argument("--verify", action="store_true",
                    help="print the canonical V2 summary and exit (no migration)")
    ap.add_argument("--ledger-path", type=Path, default=None,
                    help="override the ledger path (default: canonical)")
    args = ap.parse_args(argv)
    path = args.ledger_path

    if args.verify:
        summary = L.summarize_v2(path)
        print(f"V2 summary: {summary['classes']} class(es), "
              f"{len(summary['unresolved'])} OPEN, "
              f"{len(summary['not_closed'])} not-closed, "
              f"{len(summary['recurrent'])} recurrent")
        for cid, state in summary["states"].items():
            print(f"  {state:14} {cid}")
        return 0

    projections, already = _plan(path)
    total_raw = len(L.load_raw_rows(path))
    print(f"raw rows in ledger: {total_raw}")
    print(f"V2 rows already present: {len(already)}")
    print(f"V2 projections to append: {len(projections)}")
    for p in sorted(projections, key=lambda p: (p.resolution_state, p.class_id)):
        alias = f"  [alias<-{p.legacy_alias}]" if p.legacy_alias else ""
        print(f"  {p.resolution_state:14} {p.class_id}{alias}")

    if not args.apply:
        print("\n(dry-run — no rows written; pass --apply to append)")
        return 0

    for p in projections:
        L.append_failure_event_v2(p, path=path)
    print(f"\nAPPLIED: appended {len(projections)} canonical V2 row(s) (append-only; "
          f"legacy rows preserved).")
    summary = L.summarize_v2(path)
    print(f"post-migration V2 summary: {summary['classes']} class(es), "
          f"{len(summary['unresolved'])} OPEN, {len(summary['not_closed'])} not-closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
