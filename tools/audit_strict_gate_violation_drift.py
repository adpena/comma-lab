#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing STRICT-GATE VIOLATION-COUNT DRIFT ALARM.

The bug class (diagnosed 2026-07-02, anchor: Catalog #344 drifted 0 -> 480
with NO alarm): a STRICT preflight gate whose backlog silently decays red
without anyone noticing — apparatus blindness recursing one layer up. The
existing Catalog #185 drift gate
(`check_strict_flipped_catalog_entries_have_live_count_zero`) is
structurally blind to it: #185 only invokes a gate when that gate's
CLAUDE.md catalog row LITERALLY self-claims "live count: 0". #344's row
says "strict-flipped 2026-05-19" (no "live count: 0"), so #344 was never
invoked and its 0 -> 480 decay went unwatched.

This tool is the rc-bearing ALARM (Catalog #185 scope extension; no new
catalog number). It reads the committed baseline manifest
`.omx/state/strict_gate_violation_baseline.json` and, for each WATCHED
strict gate, invokes it (strict=False) and compares its LIVE violation
count against the DECLARED `baseline_max`. Intentional backlogs
(#344 -> 480) are declared so they do not spuriously alarm; a NEW drift
PAST the declared count fires immediately (rc=1).

It shares the ONE canonical evaluator
`tac.preflight.evaluate_strict_gate_violation_drift` with the preflight
CHECK (`check_strict_gate_violation_counts_within_declared_baseline`) so
the tool and the gate never drift.

Modes:
  (default)          -- snapshot only the manifest-declared watched gates
                        (fast). rc=1 if any is OVER_BASELINE or a declared
                        gate is MISSING_CALLABLE.
  --full             -- best-effort snapshot of EVERY strict callsite in
                        preflight_all (slow; the genuine blind-spot closer
                        that finds NOT-YET-declared drifting gates). rc=1
                        if any strict gate's count exceeds its declared
                        baseline (0 if undeclared).
  --json             -- machine-readable output.
  --update-baseline  -- re-baseline the manifest to current live counts
                        (operator convenience after a burndown, or to seed
                        newly-declared gates). With --full also ADDS any
                        undeclared nonzero strict gate.

Usage:
    .venv/bin/python tools/audit_strict_gate_violation_drift.py
    .venv/bin/python tools/audit_strict_gate_violation_drift.py --json
    .venv/bin/python tools/audit_strict_gate_violation_drift.py --full
    .venv/bin/python tools/audit_strict_gate_violation_drift.py --update-baseline
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.preflight import (  # noqa: E402
    STRICT_GATE_DRIFT_MISSING_CALLABLE,
    STRICT_GATE_DRIFT_OVER_BASELINE,
    STRICT_GATE_DRIFT_UNDER_BASELINE,
    _STRICT_GATE_BASELINE_RELPATH,
    _check_176_collect_strict_callsites,
    evaluate_strict_gate_violation_drift,
    load_strict_gate_violation_baseline,
    snapshot_strict_gate_violation_count,
)

_ALARM_VERDICTS = frozenset(
    {STRICT_GATE_DRIFT_OVER_BASELINE, STRICT_GATE_DRIFT_MISSING_CALLABLE}
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _full_sweep(repo_root: Path) -> list[dict]:
    """Best-effort snapshot of EVERY strict callsite in preflight_all.

    Returns one record per unique strict callsite:
    ``{check_name, live_count, baseline_max, verdict}``. ``live_count`` is
    None for gates that are not cleanly invokable from here. Baseline is the
    manifest value when declared, else 0. This is the blind-spot closer: it
    finds strict gates that have drifted above 0 but are NOT yet declared.
    """
    preflight_py = repo_root / "src" / "tac" / "preflight.py"
    baseline = load_strict_gate_violation_baseline(repo_root)
    try:
        text = preflight_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    names = sorted({n for _, n in _check_176_collect_strict_callsites(text)})
    records: list[dict] = []
    for name in names:
        bmax = baseline.get(name, {}).get("baseline_max", 0)
        live = snapshot_strict_gate_violation_count(name, repo_root=repo_root)
        if live is None:
            verdict = "NOT_INVOKABLE"
        elif live > bmax:
            verdict = STRICT_GATE_DRIFT_OVER_BASELINE
        elif live < bmax:
            verdict = STRICT_GATE_DRIFT_UNDER_BASELINE
        else:
            verdict = "AT_BASELINE"
        records.append({
            "check_name": name,
            "catalog": baseline.get(name, {}).get("catalog"),
            "live_count": live,
            "baseline_max": bmax,
            "declared": name in baseline,
            "verdict": verdict,
        })
    return records


def _update_baseline(repo_root: Path, records: list[dict], full: bool) -> Path:
    """Re-baseline the manifest to current live counts (operator action)."""
    path = repo_root / _STRICT_GATE_BASELINE_RELPATH
    existing: dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    gates = existing.get("gates")
    if not isinstance(gates, dict):
        gates = {}
    for rec in records:
        name = rec["check_name"]
        live = rec["live_count"]
        if live is None:
            continue
        if name in gates:
            gates[name]["baseline_max"] = live
        elif full and live > 0:
            gates[name] = {
                "catalog": rec.get("catalog"),
                "baseline_max": live,
                "first_seen_utc": _utc_now(),
                "reason": (
                    "AUTO-SEEDED by --update-baseline --full: undeclared "
                    "strict gate found with a nonzero live count. Replace this "
                    "with a real reason (why is this non-zero intentional?) "
                    "and a burndown plan."
                ),
            }
    existing["gates"] = gates
    existing.setdefault(
        "_doc",
        "DECLARED per-gate baseline max live-violation count for watched "
        "STRICT preflight gates. See CLAUDE.md Catalog #185 scope extension.",
    )
    existing["_catalog"] = 185
    existing["generated_at_utc"] = _utc_now()
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=str(_REPO_ROOT))
    ap.add_argument(
        "--full", action="store_true",
        help="best-effort snapshot of EVERY strict callsite (slow; finds "
             "not-yet-declared drifting gates).",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--update-baseline", action="store_true",
        help="re-baseline the manifest to current live counts (operator "
             "action after a burndown / to seed).",
    )
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()

    if args.full:
        records = _full_sweep(repo_root)
    else:
        records = evaluate_strict_gate_violation_drift(repo_root)

    if args.update_baseline:
        path = _update_baseline(repo_root, records, full=args.full)
        if not args.json:
            print(f"Re-baselined manifest written: "
                  f"{path.relative_to(repo_root)}")
        # After re-baseline, re-evaluate so the exit code reflects the new
        # state (should be clean unless a MISSING_CALLABLE remains).
        records = (
            _full_sweep(repo_root) if args.full
            else evaluate_strict_gate_violation_drift(repo_root)
        )

    alarms = [r for r in records if r["verdict"] in _ALARM_VERDICTS]

    if args.json:
        counts: dict[str, int] = {}
        for r in records:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(json.dumps({
            "mode": "full" if args.full else "declared",
            "manifest": _STRICT_GATE_BASELINE_RELPATH,
            "total_watched": len(records),
            "alarms": len(alarms),
            "verdict_counts": counts,
            "records": records,
        }, indent=2))
        return 1 if alarms else 0

    mode = "FULL (every strict callsite)" if args.full else "declared watchlist"
    print(f"Strict-gate violation-count DRIFT ALARM — mode: {mode}")
    print(f"  manifest: {_STRICT_GATE_BASELINE_RELPATH}")
    print(f"  watched gates: {len(records)}   alarms: {len(alarms)}\n")
    # Alarms first, then advisories, then the rest.
    order = {
        STRICT_GATE_DRIFT_OVER_BASELINE: 0,
        STRICT_GATE_DRIFT_MISSING_CALLABLE: 0,
        STRICT_GATE_DRIFT_UNDER_BASELINE: 1,
    }
    for r in sorted(records, key=lambda x: (order.get(x["verdict"], 2),
                                            -(x["live_count"] or 0))):
        cat = r.get("catalog")
        cat_tag = f"#{cat}" if cat is not None else "#? "
        flag = "  ALARM ->" if r["verdict"] in _ALARM_VERDICTS else "         "
        if not args.full and r["verdict"] not in _ALARM_VERDICTS \
                and r["verdict"] != STRICT_GATE_DRIFT_UNDER_BASELINE:
            # default mode: keep AT_BASELINE lines (context is small)
            pass
        print(f"{flag} [{r['verdict']:<16}] {cat_tag:<6} "
              f"live={str(r['live_count']):<5} baseline={r['baseline_max']:<5} "
              f"{r['check_name']}")

    if alarms:
        print(f"\n  {len(alarms)} ALARM(s): a watched strict gate drifted "
              "above its declared baseline (or a declared gate vanished). "
              "Fix the new violations OR raise baseline_max with a reason.")
    else:
        print("\n  OK: every watched strict gate is within its declared "
              "baseline. (Known backlogs like #344 -> 480 are DECLARED, not "
              "alarmed; burn them down and LOWER the baseline.)")
    if not args.full:
        print("  Tip: run with --full to sweep EVERY strict callsite for "
              "not-yet-declared drift (slow).")
    return 1 if alarms else 0


if __name__ == "__main__":
    raise SystemExit(main())
