#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit council-tier deliberation cadence vs sustainable budget (v2 spec).

Per the COUNCIL-HIERARCHY-V2 landing 2026-05-16, each council tier has an
operator-attention budget. Going over budget is the "stop and consolidate"
brake signal — the operator may be deliberating too often at the wrong tier
(e.g. landing T3 every day when only T2 was warranted).

Sustainable cadence per tier (per the v2 spec):

* **T1 Working Group** — UNBOUNDED (many/day OK; bounded by elevation
  triggers that promote crossing-finding T1 outputs into T2/T3).
* **T2 Inner-Skunkworks** — ≤3/day (5-of-6 sextet must convene; over budget
  means design tradeoffs are coming faster than 5 humans can deliberate
  rigorously).
* **T3 Full Grand Council** — ≤3/week (≥12-of-20 grand council + 5-of-6
  sextet must convene; over budget means CLAUDE.md non-negotiable changes /
  cross-cutting wire-ins are coming too fast for council coherence).
* **T4 Symposium** — ≤2/month (≥16-of-20 + 6-of-6 sextet + ≥1 specialist
  per affected paradigm; over budget means strategic redirection is
  happening too often — strong signal of unstable directional commitment).

Verdicts per tier per window:

* **WITHIN_BUDGET** — count <= 80% of budget.
* **APPROACHING_LIMIT** — 80% < count <= 100% of budget.
* **OVER_CADENCE** — count > 100% of budget (operator-visible alert).

CLI:

    python tools/audit_council_tier_cadence.py             # human-readable
    python tools/audit_council_tier_cadence.py --json      # machine output

Exit codes:

* 0 = no tier OVER_CADENCE in window.
* 1 = at least one tier OVER_CADENCE — operator should review the cadence
  alert and consider tier-elevation review / stop-and-consolidate pause.

Verified against:

* COUNCIL-HIERARCHY-V2 spec [verified-against: feedback_council_hierarchy_v2_landed_20260516.md].
* :mod:`tac.council_continual_learning` canonical posterior path.
* Sister of :mod:`tools.audit_stale_l1_substrates` (premortem #1 pattern).
* CLAUDE.md "Council hierarchy: 4-tier protocol" non-negotiable.

Memory: ``feedback_council_hierarchy_v2_landed_20260516.md`` Catalog #300 sister.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root is two parents up (`tools/` is a direct child of repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))  # noqa: E402

from tac.council_continual_learning import (  # noqa: E402
    CouncilDeliberationRecord,
    CouncilTier,
    DEFAULT_COUNCIL_POSTERIOR_PATH,
    load_council_anchors,
)


# Sustainable cadence budgets per tier in DELIBERATIONS PER WINDOW.
# Window is the lookback period (default 30 days); per-tier numeric budgets
# are normalized to the window so direct count comparison applies.
SUSTAINABLE_CADENCE: dict[str, dict[str, Any]] = {
    CouncilTier.T1: {
        # T1 is UNBOUNDED by design (bounded-scope working groups; elevation
        # triggers promote crossing-finding outputs into T2/T3). Use None
        # as the sentinel "always WITHIN_BUDGET" value.
        "per_day": None,
        "per_week": None,
        "per_month": None,
        "window_30d_budget": None,
    },
    CouncilTier.T2: {
        "per_day": 3,
        "per_week": 21,
        "per_month": 90,
        "window_30d_budget": 90,
    },
    CouncilTier.T3: {
        "per_day": None,  # cadence measured per-week not per-day
        "per_week": 3,
        "per_month": 13,
        "window_30d_budget": 13,
    },
    CouncilTier.T4: {
        "per_day": None,
        "per_week": None,
        "per_month": 2,
        "window_30d_budget": 2,
    },
}


WITHIN_BUDGET = "WITHIN_BUDGET"
APPROACHING_LIMIT = "APPROACHING_LIMIT"
OVER_CADENCE = "OVER_CADENCE"
UNBOUNDED = "UNBOUNDED"


@dataclass(frozen=True)
class TierVerdict:
    tier: str
    count: int
    budget: int | None
    verdict: str
    pct_of_budget: float | None
    alert_message: str


def _parse_utc_iso(s: str) -> _dt.datetime | None:
    """Tolerant ISO-8601 parse; returns None on malformed input."""
    if not s:
        return None
    try:
        # Python's datetime parser is strict about trailing Z; normalize.
        normalized = s.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None


def _count_anchors_by_tier(
    anchors: list[CouncilDeliberationRecord],
    *,
    since: _dt.datetime,
    until: _dt.datetime,
) -> dict[str, int]:
    """Count distinct deliberation_ids per tier within [since, until]."""
    counts: dict[str, int] = {t: 0 for t in (
        CouncilTier.T1, CouncilTier.T2, CouncilTier.T3, CouncilTier.T4
    )}
    seen_ids_by_tier: dict[str, set[str]] = {t: set() for t in counts}
    for anchor in anchors:
        # Only count `dispatched` events (outcomes are not separate deliberations).
        if anchor.event_type != "dispatched":
            continue
        ts = _parse_utc_iso(anchor.written_at_utc)
        if ts is None:
            continue
        # Normalize to UTC-aware.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.UTC)
        if ts < since or ts > until:
            continue
        tier = anchor.council_tier
        if tier not in counts:
            continue
        if anchor.deliberation_id in seen_ids_by_tier[tier]:
            continue
        seen_ids_by_tier[tier].add(anchor.deliberation_id)
        counts[tier] += 1
    return counts


def classify_tier_count(tier: str, count: int) -> TierVerdict:
    """Apply the cadence-budget formula and produce a verdict for one tier."""
    cadence = SUSTAINABLE_CADENCE.get(tier, {})
    budget = cadence.get("window_30d_budget")
    if budget is None:
        return TierVerdict(
            tier=tier,
            count=count,
            budget=None,
            verdict=UNBOUNDED,
            pct_of_budget=None,
            alert_message=(
                f"Tier {tier} is UNBOUNDED by design (elevation triggers route "
                "crossing-finding outputs to higher tiers)."
            ),
        )
    pct = (count / budget) * 100.0 if budget else 0.0
    if count > budget:
        verdict = OVER_CADENCE
        msg = (
            f"Tier {tier} OVER_CADENCE: {count} deliberations in last 30d vs "
            f"budget {budget} ({pct:.0f}% of budget). Per CLAUDE.md "
            "'Council hierarchy: 4-tier protocol' non-negotiable + 'Gate "
            "consolidation discipline' sister principle: STOP AND CONSOLIDATE. "
            "Review whether recent deliberations could have been resolved at a "
            "LOWER tier (e.g. T3 → T2 via Shannon tie-break, T4 → T3 via "
            "specialist consult). Re-cadence the operator-attention budget."
        )
    elif count > 0.8 * budget:
        verdict = APPROACHING_LIMIT
        msg = (
            f"Tier {tier} APPROACHING_LIMIT: {count}/{budget} ({pct:.0f}% of "
            "budget). Watch for next deliberation; if it'd push over, "
            "consider tier-elevation review."
        )
    else:
        verdict = WITHIN_BUDGET
        msg = (
            f"Tier {tier} WITHIN_BUDGET: {count}/{budget} ({pct:.0f}% of "
            "budget)."
        )
    return TierVerdict(
        tier=tier,
        count=count,
        budget=budget,
        verdict=verdict,
        pct_of_budget=pct,
        alert_message=msg,
    )


def audit_cadence(
    *,
    posterior_path: Path | None = None,
    lookback_days: int = 30,
    now_utc: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Run the cadence audit over the canonical posterior.

    Returns a dict with keys:
      - schema: 'council_tier_cadence_audit_v1'
      - lookback_days: int
      - window_start_utc / window_end_utc: ISO timestamps
      - per_tier: list of TierVerdict-as-dict
      - any_over_cadence: bool
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    now = now_utc or _dt.datetime.now(_dt.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.UTC)
    window_start = now - _dt.timedelta(days=lookback_days)
    anchors = load_council_anchors(posterior_path=posterior)
    counts = _count_anchors_by_tier(anchors, since=window_start, until=now)
    per_tier: list[TierVerdict] = []
    for tier in (CouncilTier.T1, CouncilTier.T2, CouncilTier.T3, CouncilTier.T4):
        per_tier.append(classify_tier_count(tier, counts.get(tier, 0)))
    any_over = any(v.verdict == OVER_CADENCE for v in per_tier)
    return {
        "schema": "council_tier_cadence_audit_v1",
        "lookback_days": lookback_days,
        "window_start_utc": window_start.isoformat(timespec="seconds"),
        "window_end_utc": now.isoformat(timespec="seconds"),
        "posterior_path": str(posterior),
        "per_tier": [
            {
                "tier": v.tier,
                "count": v.count,
                "budget": v.budget,
                "verdict": v.verdict,
                "pct_of_budget": v.pct_of_budget,
                "alert_message": v.alert_message,
            }
            for v in per_tier
        ],
        "any_over_cadence": any_over,
    }


def render_text(report: dict[str, Any]) -> str:
    """Human-readable rendering of the audit report."""
    lines: list[str] = []
    lines.append("Council-tier cadence audit (v2)")
    lines.append("=" * 50)
    lines.append(
        f"Window: last {report['lookback_days']}d "
        f"({report['window_start_utc']} → {report['window_end_utc']})"
    )
    lines.append(f"Posterior: {report['posterior_path']}")
    lines.append("")
    lines.append(f"{'Tier':<6} {'Count':>6} {'Budget':>8} {'Pct':>6}  Verdict")
    lines.append("-" * 50)
    for entry in report["per_tier"]:
        pct = (
            f"{entry['pct_of_budget']:.0f}%"
            if entry["pct_of_budget"] is not None
            else "n/a"
        )
        budget_str = str(entry["budget"]) if entry["budget"] is not None else "∞"
        lines.append(
            f"{entry['tier']:<6} {entry['count']:>6} {budget_str:>8} "
            f"{pct:>6}  {entry['verdict']}"
        )
    lines.append("")
    for entry in report["per_tier"]:
        if entry["verdict"] in (OVER_CADENCE, APPROACHING_LIMIT):
            lines.append(f"⚠ {entry['alert_message']}")
    if report["any_over_cadence"]:
        lines.append("")
        lines.append(
            "OPERATOR ACTION REQUESTED: at least one tier is OVER_CADENCE. "
            "Review per-tier alerts above and consider stop-and-consolidate."
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit_council_tier_cadence",
        description=(
            "Audit council-tier deliberation cadence vs sustainable budget "
            "(COUNCIL-HIERARCHY-V2 spec). Exit 1 if any tier OVER_CADENCE."
        ),
    )
    p.add_argument(
        "--posterior-path",
        type=Path,
        default=None,
        help="Override the council deliberation posterior JSONL path.",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Lookback window in days (default 30).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON report instead of human-readable text.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = audit_cadence(
        posterior_path=args.posterior_path,
        lookback_days=args.lookback_days,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        print(render_text(report))
    return 1 if report["any_over_cadence"] else 0


if __name__ == "__main__":
    sys.exit(main())
