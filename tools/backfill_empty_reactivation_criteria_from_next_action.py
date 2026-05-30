#!/usr/bin/env -S uv run --quiet --script
# SPDX-License-Identifier: MIT
"""Backfill EMPTY reactivation_criteria from next_action — META Finding A Landing 2.

Canonical operator-facing tool for the META Finding A canonical 2-landing pattern
landed 2026-05-30 per the deferred-items feeder audit at commit ``a9d45b171``
(memo ``.omx/research/deferred_items_feeder_audit_landed_20260530.md``).

Why
───
Per the audit's META Finding A: 104 of 105 DEFER probe outcomes in
``.omx/state/probe_outcomes.jsonl`` have EMPTY ``reactivation_criteria`` field.
Callers consistently populated the ``next_action`` field instead. The canonical
feeder consumer at ``tac.cathedral_consumers`` queries ``reactivation_criteria``;
the EMPTY field means the feeder can NEVER auto-pickup historical probes
structurally.

Landing 1 (canonical helper extension at ``tac.probe_outcomes_ledger``) closes
the FORWARD surface — future ``register_probe_outcome`` calls auto-derive
``reactivation_criteria`` from ``next_action``.

Landing 2 (this tool) closes the HISTORICAL surface — it scans the existing
ledger for rows with EMPTY ``reactivation_criteria`` AND substantive
``next_action``, and for each appends a NEW ``EVENT_BACKFILL`` event row that
carries the auto-derived ``reactivation_criteria`` + the canonical
``AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL`` provenance. This is
APPEND-ONLY per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113/#132 — the
original adjudicated row is NEVER mutated; the canonical latest-row-wins
semantics surface the auto-derived criteria to the feeder consumer.

NO FAKE IMPLEMENTATIONS discipline
──────────────────────────────────
Per the new CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable: rows where
BOTH ``reactivation_criteria`` AND ``next_action`` are empty/placeholder are
SKIPPED — they preserve HONEST emptiness rather than fabricating fake content
from synthetic markers. The 4 historical rows in this category remain empty
until the operator manually populates them per the standing directive.

Path discipline
───────────────
Default --dry-run (no mutation). --apply REQUIRES --operator-approved
'<handle>:<UTC_timestamp>' per the canonical pattern from
``tools/gc_experiments_results.py`` (Catalog #154 sister destructive-helper
discipline).

CLI exit codes
──────────────
- 0  — dry-run succeeded OR apply succeeded
- 2  — validation error (missing/malformed --operator-approved on --apply)
- 3  — refusing --apply without --operator-approved

Usage
─────
::

    # Preview the backfill plan (default; safe; no mutation):
    .venv/bin/python tools/backfill_empty_reactivation_criteria_from_next_action.py

    # Execute the backfill:
    .venv/bin/python tools/backfill_empty_reactivation_criteria_from_next_action.py \\
        --apply \\
        --operator-approved 'adpena:2026-05-30T19:00:00Z'

    # JSON output for downstream tooling:
    .venv/bin/python tools/backfill_empty_reactivation_criteria_from_next_action.py --json

Cross-references
────────────────
- Catalog #313 (canonical probe-outcomes ledger gate)
- Catalog #245 (canonical Modal call_id ledger 4-layer exemplar)
- Catalog #131/#138 (fcntl-locked + strict-load discipline)
- Catalog #154 (canonical operator-approved-handle CLI pattern)
- Catalog #287 (placeholder-rationale rejection sister discipline)
- Catalog #371 (auto-recalibrator sister pattern)
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" non-negotiable (sister discipline)
- Memo: ``feedback_meta_finding_a_auto_derive_reactivation_criteria_canonical_2_landing_landed_20260530.md``
"""

# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure the canonical package is importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.probe_outcomes_ledger import (  # noqa: E402
    AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL,
    EVENT_BACKFILL,
    PROBE_OUTCOMES_LEDGER_LOCK,
    PROBE_OUTCOMES_LEDGER_PATH,
    SCHEMA_VERSION,
    _append_event_locked,
    _auto_derive_reactivation_criteria_from_next_action,
    _is_substantive_string,
    _ledger_lock,
    _save_ledger,
    load_outcomes,
    load_outcomes_strict,
)


def _now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format with microsecond precision."""
    return (
        _dt.datetime.now(_dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def compute_backfill_plan(
    *,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Scan ``ledger_path`` and return a structured backfill plan.

    Returns a dict with the following shape::

        {
            "schema": "pact.probe_outcomes_reactivation_criteria_backfill_plan.v1",
            "generated_at_utc": "<UTC>",
            "ledger_path": "<path>",
            "ledger_rows_total": N,
            "candidates_to_backfill": [
                {
                    "probe_id": "<probe_id>",
                    "substrate": "<substrate>",
                    "verdict": "DEFER",
                    "next_action": "<verbatim next_action>",
                    "derived_reactivation_criteria": ["next_action_satisfied: <X>"],
                    "current_blocker_status": "blocking" | "advisory" | "expired",
                },
                ...
            ],
            "skipped_both_empty": [<probe_id>, ...],
            "skipped_already_populated": [<probe_id>, ...],
            "total_candidates": M,
        }

    The plan is LATEST-ROW-WINS per the canonical ledger semantics; backfill is
    triggered only when the LATEST row for a probe_id has EMPTY
    ``reactivation_criteria``.
    """
    path = ledger_path or PROBE_OUTCOMES_LEDGER_PATH
    rows = load_outcomes(path)

    # Group by probe_id; latest row wins (JSONL append order).
    latest_by_probe_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row.get("probe_id")
        if isinstance(pid, str):
            latest_by_probe_id[pid] = row

    candidates: list[dict[str, Any]] = []
    skipped_both_empty: list[str] = []
    skipped_already_populated: list[str] = []

    for probe_id, row in latest_by_probe_id.items():
        existing_criteria = row.get("reactivation_criteria")
        # Treat already-populated rows (non-empty list[str] or substantive str)
        # as skip per the canonical "do not mutate" semantics.
        if existing_criteria:
            if isinstance(existing_criteria, list) and any(
                _is_substantive_string(e) for e in existing_criteria
            ):
                skipped_already_populated.append(probe_id)
                continue
            if isinstance(existing_criteria, str) and _is_substantive_string(
                existing_criteria
            ):
                skipped_already_populated.append(probe_id)
                continue

        # At this point reactivation_criteria is None/empty/all-placeholder.
        next_action = row.get("next_action")
        derived_criteria, _ = _auto_derive_reactivation_criteria_from_next_action(
            next_action
        )
        if derived_criteria is None:
            # BOTH empty — HONEST emptiness per NO FAKE IMPLEMENTATIONS.
            skipped_both_empty.append(probe_id)
            continue

        candidates.append(
            {
                "probe_id": probe_id,
                "substrate": row.get("substrate"),
                "verdict": row.get("verdict"),
                "next_action": next_action,
                "derived_reactivation_criteria": derived_criteria,
                "current_blocker_status": row.get("blocker_status"),
            }
        )

    return {
        "schema": "pact.probe_outcomes_reactivation_criteria_backfill_plan.v1",
        "generated_at_utc": _now_iso(),
        "ledger_path": str(path),
        "ledger_rows_total": len(rows),
        "unique_probe_ids": len(latest_by_probe_id),
        "candidates_to_backfill": candidates,
        "skipped_both_empty": sorted(skipped_both_empty),
        "skipped_already_populated": sorted(skipped_already_populated),
        "total_candidates": len(candidates),
    }


def execute_backfill(
    plan: dict[str, Any],
    *,
    operator_approved: str,
    ledger_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the backfill plan — APPEND new EVENT_BACKFILL rows under lock.

    Returns an execution summary including count of rows appended and the
    operator-approved handle. The append is fcntl-locked per Catalog #131
    sister discipline — concurrent backfill runs cannot corrupt the ledger.

    Per CLAUDE.md "Forbidden premature KILL": this tool does NOT mutate
    historical rows; it ONLY appends new EVENT_BACKFILL rows. The original
    adjudicated rows are preserved verbatim per HISTORICAL_PROVENANCE Catalog
    #110/#113.
    """
    p_path = ledger_path or PROBE_OUTCOMES_LEDGER_PATH
    l_path = lock_path or PROBE_OUTCOMES_LEDGER_LOCK

    candidates = plan.get("candidates_to_backfill", [])
    if not candidates:
        return {
            "schema": "pact.probe_outcomes_reactivation_criteria_backfill_execution.v1",
            "applied_at_utc": _now_iso(),
            "operator_approved": operator_approved,
            "appended_count": 0,
            "skipped_count": 0,
            "appended_probe_ids": [],
            "skipped_probe_ids": [],
            "notes": "no candidates in plan; nothing to do",
        }

    appended_probe_ids: list[str] = []
    skipped_probe_ids: list[tuple[str, str]] = []

    # Acquire the lock ONCE for the entire batch — sister of Catalog #131
    # transactional discipline; avoids re-acquiring the fcntl lock per append.
    with _ledger_lock(l_path):
        try:
            current_rows = load_outcomes_strict(p_path)
        except Exception as exc:
            # Surface as execution summary error; do NOT silently no-op
            # (per CLAUDE.md "NO FAKE IMPLEMENTATIONS" + Catalog #339
            # fail-closed-on-corrupt-state pattern).
            raise RuntimeError(
                f"refusing backfill — ledger at {p_path} is corrupt or "
                f"unreadable: {exc}. Inspect the file and re-run."
            ) from exc

        # Build latest-row-wins map again INSIDE the lock so concurrent writes
        # since plan-generation are honored.
        latest_by_probe_id_inside_lock: dict[str, dict[str, Any]] = {}
        for row in current_rows:
            pid = row.get("probe_id")
            if isinstance(pid, str):
                latest_by_probe_id_inside_lock[pid] = row

        new_rows = list(current_rows)
        now_iso = _now_iso()
        pid_int = os.getpid()
        host = socket.gethostname()

        for candidate in candidates:
            probe_id = candidate["probe_id"]
            existing = latest_by_probe_id_inside_lock.get(probe_id)
            if existing is None:
                skipped_probe_ids.append(
                    (probe_id, "no_existing_row_inside_lock")
                )
                continue

            # Re-check inside the lock: if reactivation_criteria became
            # populated since plan generation (e.g. via concurrent backfill or
            # operator manual edit), SKIP per latest-row-wins semantics.
            existing_criteria = existing.get("reactivation_criteria")
            if existing_criteria:
                if isinstance(existing_criteria, list) and any(
                    _is_substantive_string(e) for e in existing_criteria
                ):
                    skipped_probe_ids.append(
                        (probe_id, "already_populated_concurrent_backfill")
                    )
                    continue
                if isinstance(existing_criteria, str) and _is_substantive_string(
                    existing_criteria
                ):
                    skipped_probe_ids.append(
                        (probe_id, "already_populated_concurrent_backfill")
                    )
                    continue

            # Construct the backfill event row — APPEND-ONLY; inherits all
            # immutable identifying fields from the original adjudicated row.
            backfill_record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "event_type": EVENT_BACKFILL,
                "probe_id": probe_id,
                "substrate": existing.get("substrate"),
                "recipe_path": existing.get("recipe_path"),
                "probe_kind": existing.get("probe_kind"),
                "verdict": existing.get("verdict"),
                "metric_name": existing.get("metric_name"),
                "metric_value": existing.get("metric_value"),
                "threshold": existing.get("threshold"),
                "threshold_token": existing.get("threshold_token"),
                "evidence_path": existing.get("evidence_path"),
                "next_action": existing.get("next_action"),
                "reactivation_criteria": candidate[
                    "derived_reactivation_criteria"
                ],
                "reactivation_criteria_derivation_provenance": (
                    AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL
                ),
                "blocker_status": existing.get("blocker_status"),
                "dispatched_at_utc": existing.get("dispatched_at_utc"),
                "adjudicated_at_utc": existing.get("adjudicated_at_utc"),
                "expires_at_utc": existing.get("expires_at_utc"),
                "staleness_window_days": existing.get("staleness_window_days"),
                "agent": "backfill_tool",
                "subagent_id": (
                    "meta-finding-a-auto-derive-reactivation-criteria-"
                    "canonical-helper-fix-20260530"
                ),
                "session_id": None,
                "notes": (
                    "META Finding A canonical 2-landing pattern backfill per "
                    "feedback_meta_finding_a_auto_derive_reactivation_criteria_"
                    "canonical_2_landing_landed_20260530.md; operator-approved="
                    f"{operator_approved}"
                ),
                "written_at_utc": now_iso,
                "written_pid": pid_int,
                "written_host": host,
                "backfill_operator_approved": operator_approved,
            }
            new_rows.append(backfill_record)
            # Update the per-probe latest snapshot so a second candidate
            # targeting the same probe_id (shouldn't happen in the canonical
            # plan but defense-in-depth) sees the new row.
            latest_by_probe_id_inside_lock[probe_id] = backfill_record
            appended_probe_ids.append(probe_id)

        # Single atomic write of the entire ledger under the lock.
        _save_ledger(new_rows, p_path)

    return {
        "schema": "pact.probe_outcomes_reactivation_criteria_backfill_execution.v1",
        "applied_at_utc": _now_iso(),
        "operator_approved": operator_approved,
        "appended_count": len(appended_probe_ids),
        "skipped_count": len(skipped_probe_ids),
        "appended_probe_ids": appended_probe_ids,
        "skipped_probe_ids": [
            {"probe_id": pid, "reason": reason} for pid, reason in skipped_probe_ids
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_operator_handle(value: str) -> str:
    """Validate the --operator-approved value per canonical Catalog #154 pattern."""
    value = value.strip()
    if not value:
        raise SystemExit("VALIDATION_ERROR: --operator-approved must not be empty")
    if ":" not in value:
        raise SystemExit(
            "VALIDATION_ERROR: --operator-approved must be "
            "'<handle>:<UTC_timestamp>' (e.g. 'adpena:2026-05-30T19:00:00Z')"
        )
    handle, _, ts = value.partition(":")
    if not handle.strip():
        raise SystemExit("VALIDATION_ERROR: --operator-approved handle is empty")
    if not ts.strip():
        raise SystemExit("VALIDATION_ERROR: --operator-approved timestamp is empty")
    return value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help=(
            "Path to probe_outcomes.jsonl "
            f"(default: {PROBE_OUTCOMES_LEDGER_PATH})"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the backfill plan but do NOT mutate the ledger (default).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Execute the backfill plan — APPEND EVENT_BACKFILL rows. "
            "REQUIRES --operator-approved."
        ),
    )
    p.add_argument(
        "--operator-approved",
        default="",
        help=(
            "Operator handle:UTC_timestamp authorizing --apply "
            "(REQUIRED with --apply)"
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable summary.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-candidate detail in the human-readable summary.",
    )
    return p


def _render_human_summary(plan: dict[str, Any], verbose: bool) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("META Finding A canonical 2-landing pattern — backfill plan")
    lines.append("=" * 70)
    lines.append(f"Ledger:                  {plan['ledger_path']}")
    lines.append(f"Total rows in ledger:    {plan['ledger_rows_total']}")
    lines.append(f"Unique probe_ids:        {plan['unique_probe_ids']}")
    lines.append(f"Candidates to backfill:  {plan['total_candidates']}")
    lines.append(
        f"Skipped (both empty):    {len(plan['skipped_both_empty'])} "
        "(HONEST emptiness per NO FAKE IMPLEMENTATIONS)"
    )
    lines.append(
        f"Skipped (already pop'd): {len(plan['skipped_already_populated'])}"
    )
    lines.append("")
    if plan["total_candidates"] > 0 and verbose:
        lines.append("Candidates (first 20):")
        for cand in plan["candidates_to_backfill"][:20]:
            lines.append(f"  • {cand['probe_id']}")
            lines.append(f"      verdict:    {cand['verdict']}")
            lines.append(f"      substrate:  {cand['substrate']}")
            next_action_short = (
                cand['next_action'][:80] + "..."
                if cand.get("next_action") and len(cand["next_action"]) > 80
                else cand.get("next_action")
            )
            lines.append(f"      next_action: {next_action_short}")
        if plan["total_candidates"] > 20:
            lines.append(
                f"  ... ({plan['total_candidates'] - 20} more candidates; use "
                "--json for the full list)"
            )
        lines.append("")
    if plan["total_candidates"] > 0:
        lines.append(
            "To apply: re-run with --apply --operator-approved "
            "'<handle>:<UTC>'"
        )
    else:
        lines.append("No backfill candidates — ledger is fully populated.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.apply and not args.operator_approved:
        print(
            "REFUSING_APPLY: --apply REQUIRES --operator-approved "
            "'<handle>:<UTC_timestamp>'. Run with --dry-run first.",
            file=sys.stderr,
        )
        return 3
    if args.apply:
        _validate_operator_handle(args.operator_approved)
    if not args.apply and not args.dry_run:
        # Default mode: dry-run.
        args.dry_run = True

    plan = compute_backfill_plan(ledger_path=args.ledger_path)

    if args.apply:
        execution = execute_backfill(
            plan,
            operator_approved=args.operator_approved,
            ledger_path=args.ledger_path,
        )
        if args.json:
            print(json.dumps({"plan": plan, "execution": execution}, indent=2))
        else:
            print(_render_human_summary(plan, args.verbose))
            print()
            print("=" * 70)
            print("EXECUTION SUMMARY")
            print("=" * 70)
            print(f"Applied at:            {execution['applied_at_utc']}")
            print(f"Operator-approved:     {execution['operator_approved']}")
            print(f"Backfill rows appended: {execution['appended_count']}")
            print(f"Skipped rows:           {execution['skipped_count']}")
        return 0

    # Dry-run path.
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(_render_human_summary(plan, args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
