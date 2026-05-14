#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single-command fleet triage: health, burn, runway, redeploy recommendations.

Wraps `scripts/verify_vast_instances.py` and adds:
  - Total $/hr burn across all reachable instances
  - Hours-of-budget runway at the current burn rate
  - Top 3 lane-redeploy recommendations if any died

Usage:
    .venv/bin/python tools/triage_fleet.py                # report only
    .venv/bin/python tools/triage_fleet.py --auto-destroy-stale
    .venv/bin/python tools/triage_fleet.py --budget-cap 24

Deep hardening pass 3, dimension 4. Reference:
  - feedback_per_instance_verify_pattern_20260428
  - feedback_vastai_cost_paranoia
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_verify(auto_destroy: bool, stale_minutes: int, setup_stale_minutes: int) -> dict:
    """Invoke scripts/verify_vast_instances.py and parse its JSON output."""
    cmd = [
        ".venv/bin/python",
        str(REPO_ROOT / "scripts" / "verify_vast_instances.py"),
        "--json-out",
        "--stale-minutes", str(stale_minutes),
        "--setup-stale-minutes", str(setup_stale_minutes),
    ]
    if auto_destroy:
        cmd.append("--auto-destroy-stale")
    result = subprocess.run(  # subprocess-no-check-OK: we surface returncode + parse stdout below
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"[triage] verify_vast_instances exited {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    # The verifier prints non-JSON status lines first; find the trailing JSON
    # by parsing line-by-line until a {…} block is found.
    text = result.stdout
    parsed: dict = {}
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not parsed:
        # Fallback: full-stdout parse attempt
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
    return parsed


def _summarize(report: dict) -> dict:
    """Aggregate counts + total burn ($/hr) from the verifier report."""
    instances = report.get("instances", []) or []
    counts: dict[str, int] = {}
    total_burn = 0.0
    healthy_lanes: list[str] = []
    dead_lanes: list[str] = []
    for inst in instances:
        status = inst.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
        burn = float(inst.get("dph", 0.0) or 0.0)
        total_burn += burn
        label = inst.get("label", "")
        if status == "HEALTHY":
            healthy_lanes.append(label)
        elif status in ("CRASHED", "GONE", "IDLE"):
            dead_lanes.append(label)
    return {
        "counts": counts,
        "total_burn_per_hour": round(total_burn, 4),
        "healthy_lanes": healthy_lanes,
        "dead_lanes": dead_lanes,
        "n_total": len(instances),
    }


def _redeploy_recommendations(dead_lanes: list[str], top_n: int = 3) -> list[str]:
    """Heuristic: priority order = recently-dispatched > older lanes.

    Without explicit lane priority metadata, returns the first N dead lanes
    so the operator's eye picks them up. Replace with a smarter ranker if
    we add per-lane EV scores in .omx/state/lane_priority.json.
    """
    return dead_lanes[:top_n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--auto-destroy-stale", action="store_true",
                        help="Destroy CRASHED/IDLE/SETUP-stuck instances")
    parser.add_argument("--stale-minutes", type=int, default=30)
    parser.add_argument("--setup-stale-minutes", type=int, default=90)
    parser.add_argument("--budget-cap", type=float, default=24.0,
                        help="Hard budget cap in $ — runway is hours of "
                             "remaining burn before this is hit")
    parser.add_argument("--budget-spent", type=float, default=None,
                        help="Already-spent $; runway = (cap - spent) / burn")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON to stdout (suppresses pretty print)")
    args = parser.parse_args()

    report = _run_verify(
        args.auto_destroy_stale, args.stale_minutes, args.setup_stale_minutes,
    )
    summary = _summarize(report)
    burn = summary["total_burn_per_hour"]
    spent = args.budget_spent if args.budget_spent is not None else 0.0
    remaining = max(0.0, args.budget_cap - spent)
    if burn > 0:
        runway_hours = remaining / burn
    else:
        runway_hours = float("inf")
    summary["runway_hours"] = (
        round(runway_hours, 2) if runway_hours != float("inf") else None
    )
    summary["budget_cap_dollars"] = args.budget_cap
    summary["budget_remaining_dollars"] = round(remaining, 2)

    redeploy = _redeploy_recommendations(summary["dead_lanes"], top_n=3)
    summary["redeploy_recommendations"] = redeploy

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    # Pretty-print
    print("=" * 60)
    print("FLEET TRIAGE REPORT")
    print("=" * 60)
    print(f"Total instances: {summary['n_total']}")
    print(f"Status counts:   {summary['counts']}")
    print(f"Healthy lanes:   {summary['healthy_lanes']}")
    print(f"Dead lanes:      {summary['dead_lanes']}")
    print(f"Burn rate:       ${burn:.4f}/hr")
    print(f"Budget remaining: ${remaining:.2f} (cap=${args.budget_cap:.2f})")
    if runway_hours == float("inf"):
        print("Runway:          ∞ (no active burn)")
    else:
        print(f"Runway:          {runway_hours:.1f} hours @ current burn")
    if redeploy:
        print(f"Top redeploy:    {redeploy}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
