# SPDX-License-Identifier: MIT
"""Audit the per-window KILL-verdict compliance rate vs CLAUDE.md "Forbidden premature KILL".

ML3 apparatus-process change of META-RESURRECTION-AUDIT-V2 op-routables
(`.omx/research/meta_resurrection_audit_v2_op_routables_canonicalization_wave_landed_20260527.md`
§Item #4). Operator-facing surface for the 85%-over-kill empirical insight: scans
the canonical probe-outcomes ledger (`.omx/state/probe_outcomes.jsonl`; the
machine-readable source of adjudicated KILL/DEFER/FALSIFIED verdicts per Catalog
#313) and computes the COMPLIANCE RATE of negative-result verdicts against
CLAUDE.md "Forbidden premature KILL without research exhaustion".

A KILL/DEFER/FALSIFIED verdict is COMPLIANT when it carries the canonical rigor
evidence per the 5 NEW rigor gates (Catalog #307 paradigm-vs-implementation +
#308 alternative-reducer enumeration + #313 reactivation criteria + #324 / #325
post-training validation) per `feedback_pre_rigor_kill_defer_falsified_inventory
_landed_20260517.md`. The compliance signal:

* COMPLIANT          - verdict carries reactivation_criteria + next_action +
                       advisory blocker_status (DEFER not unconditional KILL) OR
                       PROCEED/PROMOTE/PARTIAL (not a kill at all).
* NON_COMPLIANT_KILL - VERDICT_KILL with blocking status AND no reactivation
                       criteria (a kill-too-fast candidate per Forbidden
                       premature KILL).
* INDETERMINATE      - blocking INDEPENDENT/DEFER verdict missing a next_action
                       (operator-routable for re-classification).

Per the audit's §4.4 META-lesson #4 proposal: if the compliance rate over a
window drops below the 90% threshold, surface an operator-visible STOP AND
CONSOLIDATE alert (sister of Catalog #300 Mission alignment Consequence 5
pattern). The 2026-05-17 inventory empirically established ~85% over-kill across
the historical corpus (~15% non-compliance), which is operator-visible but the
90% threshold is the structurally-correct STOP-AND-CONSOLIDATE trigger.

Per CLAUDE.md "Forbidden premature KILL": this tool does NOT reopen lanes, does
NOT change verdicts, and does NOT propose new kills. It is the OBSERVABILITY
surface that makes the compliance rate queryable + AUTOMATED per the 7th META
AUTOMATED+COMPOUNDING+OPTIMAL standing directive.

Sister of:
  - `tools/audit_stale_l1_substrates.py` (Catalog #298 operator-facing surface)
  - `tools/audit_council_tier_cadence.py` (Catalog #300 cadence audit)
  - `tac.probe_outcomes_ledger` (Catalog #313 canonical ledger this tool reads)
  - META-RESURRECTION-AUDIT-V2 op-routables Items #2 (5 amplification equations)
    + #3 (cathedral consumer) - this tool is Item #4 ML3.

Usage:
    .venv/bin/python tools/audit_kill_verdict_compliance_rate.py
    .venv/bin/python tools/audit_kill_verdict_compliance_rate.py --json
    .venv/bin/python tools/audit_kill_verdict_compliance_rate.py --threshold 0.90
    .venv/bin/python tools/audit_kill_verdict_compliance_rate.py --window-days 30
    .venv/bin/python tools/audit_kill_verdict_compliance_rate.py --strict  # rc=1 below threshold
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import tac.probe_outcomes_ledger as pol

# Canonical STOP-AND-CONSOLIDATE threshold per audit §4.4 META-lesson #4.
DEFAULT_COMPLIANCE_THRESHOLD = 0.90

# Verdict tokens that are negative-result adjudications (KILL/DEFER/FALSIFIED).
# PROCEED / PROMOTE / PARTIAL are NOT negative-result verdicts (advisory or
# positive); they are excluded from the kill-compliance denominator.
_NEGATIVE_RESULT_VERDICTS = (
    pol.VERDICT_KILL,
    pol.VERDICT_DEFER,
)


def _parse_utc(s: str | None) -> _dt.datetime | None:
    if not s or not isinstance(s, str):
        return None
    try:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _classify_compliance(row: dict[str, Any]) -> str:
    """Return COMPLIANT / NON_COMPLIANT_KILL / INDETERMINATE for one verdict row."""
    verdict = str(row.get("verdict", ""))
    blocker_status = str(row.get("blocker_status", ""))
    reactivation = row.get("reactivation_criteria") or row.get("notes") or ""
    next_action = row.get("next_action") or ""

    # Positive / advisory verdicts are compliant by construction.
    if verdict not in _NEGATIVE_RESULT_VERDICTS:
        return "COMPLIANT"

    has_reactivation = bool(str(reactivation).strip())
    has_next_action = bool(str(next_action).strip())

    if verdict == pol.VERDICT_KILL and blocker_status == pol.BLOCKER_STATUS_BLOCKING:
        # Unconditional blocking KILL: COMPLIANT only with reactivation criteria
        # per CLAUDE.md "Forbidden premature KILL" (c) reactivation pinned.
        if has_reactivation:
            return "COMPLIANT"
        return "NON_COMPLIANT_KILL"

    # DEFER verdicts: COMPLIANT with reactivation criteria OR a next_action
    # (operator-routable) per CLAUDE.md "default verdict is DEFERRED-pending-
    # research". Missing both -> INDETERMINATE (operator-routable).
    if has_reactivation or has_next_action:
        return "COMPLIANT"
    return "INDETERMINATE"


def audit_kill_verdict_compliance(
    *,
    repo_root: Path | str | None = None,
    window_days: int | None = None,
    threshold: float = DEFAULT_COMPLIANCE_THRESHOLD,
    now_utc: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Compute the KILL-verdict compliance rate over the latest-per-probe corpus.

    Returns a typed report dict (JSON-safe) with per-verdict-row classification
    + aggregate compliance rate + STOP-AND-CONSOLIDATE alert flag.
    """
    root = Path(repo_root) if repo_root is not None else None
    ledger_path = (
        (root / ".omx/state/probe_outcomes.jsonl")
        if root is not None
        else None
    )
    rows = pol.load_outcomes(path=ledger_path)

    now = now_utc or _dt.datetime.now(_dt.timezone.utc)
    cutoff = (
        now - _dt.timedelta(days=window_days) if window_days is not None else None
    )

    # Collapse to latest-per-probe_id (the canonical adjudication state).
    latest_by_probe: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = row.get("probe_id")
        if not pid:
            continue
        latest_by_probe[pid] = row  # JSONL append-only => last wins

    classified: list[dict[str, Any]] = []
    for pid, row in latest_by_probe.items():
        verdict = str(row.get("verdict", ""))
        if verdict not in _NEGATIVE_RESULT_VERDICTS:
            continue  # only kill/defer rows count toward kill-compliance
        if cutoff is not None:
            ts = _parse_utc(row.get("adjudicated_at_utc") or row.get("written_at_utc"))
            if ts is not None and ts < cutoff:
                continue
        compliance = _classify_compliance(row)
        classified.append(
            {
                "probe_id": pid,
                "substrate": row.get("substrate"),
                "verdict": verdict,
                "blocker_status": row.get("blocker_status"),
                "compliance": compliance,
            }
        )

    total = len(classified)
    non_compliant = [c for c in classified if c["compliance"] == "NON_COMPLIANT_KILL"]
    indeterminate = [c for c in classified if c["compliance"] == "INDETERMINATE"]
    compliant = [c for c in classified if c["compliance"] == "COMPLIANT"]

    compliance_rate = (len(compliant) / total) if total else 1.0
    stop_and_consolidate = compliance_rate < threshold and total > 0

    return {
        "schema_version": "kill_verdict_compliance_audit_v1_20260527",
        "audit_utc": now.isoformat(),
        "window_days": window_days,
        "threshold": threshold,
        "total_negative_result_verdicts": total,
        "n_compliant": len(compliant),
        "n_non_compliant_kill": len(non_compliant),
        "n_indeterminate": len(indeterminate),
        "compliance_rate": round(compliance_rate, 4),
        "stop_and_consolidate_alert": stop_and_consolidate,
        "non_compliant_kill_rows": non_compliant[:50],
        "indeterminate_rows": indeterminate[:50],
        "note": (
            "Per CLAUDE.md 'Forbidden premature KILL': this audit does NOT reopen "
            "lanes or change verdicts. NON_COMPLIANT_KILL + INDETERMINATE rows are "
            "operator-routable for re-classification per Catalog #307/#308; "
            "default re-verdict is DEFERRED-pending-research, NEVER confirmed-kill."
        ),
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = []
    lines.append("=== KILL-verdict compliance rate audit (ML3; META-RESURRECTION-AUDIT-V2) ===")
    lines.append(f"audit_utc: {report['audit_utc']}")
    win = report["window_days"]
    lines.append(f"window: {'all-time' if win is None else str(win) + ' days'}")
    lines.append(f"threshold: {report['threshold']:.2%}")
    lines.append(f"total negative-result verdicts: {report['total_negative_result_verdicts']}")
    lines.append(f"  COMPLIANT:          {report['n_compliant']}")
    lines.append(f"  NON_COMPLIANT_KILL: {report['n_non_compliant_kill']}")
    lines.append(f"  INDETERMINATE:      {report['n_indeterminate']}")
    lines.append(f"compliance_rate: {report['compliance_rate']:.2%}")
    if report["stop_and_consolidate_alert"]:
        lines.append("")
        lines.append(
            "*** STOP AND CONSOLIDATE: compliance rate below threshold. "
            "Review NON_COMPLIANT_KILL + INDETERMINATE rows per Catalog #307/#308. "
            "Per CLAUDE.md 'Forbidden premature KILL': re-classify, do NOT confirm-kill. ***"
        )
    else:
        lines.append("WITHIN_THRESHOLD: no STOP AND CONSOLIDATE alert.")
    if report["non_compliant_kill_rows"]:
        lines.append("")
        lines.append("NON_COMPLIANT_KILL rows (kill-too-fast candidates):")
        for r in report["non_compliant_kill_rows"]:
            lines.append(f"  - {r['probe_id']} (substrate={r['substrate']})")
    if report["indeterminate_rows"]:
        lines.append("")
        lines.append("INDETERMINATE rows (missing next_action; operator-routable):")
        for r in report["indeterminate_rows"]:
            lines.append(f"  - {r['probe_id']} (substrate={r['substrate']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit KILL-verdict compliance rate vs CLAUDE.md 'Forbidden premature KILL'.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COMPLIANCE_THRESHOLD,
        help="STOP-AND-CONSOLIDATE compliance threshold (default 0.90)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="restrict to verdicts adjudicated within the last N days (default all-time)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit rc=1 if compliance rate is below the threshold",
    )
    parser.add_argument("--repo-root", type=str, default=None)
    args = parser.parse_args(argv)

    report = audit_kill_verdict_compliance(
        repo_root=args.repo_root,
        window_days=args.window_days,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report))

    if args.strict and report["stop_and_consolidate_alert"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
