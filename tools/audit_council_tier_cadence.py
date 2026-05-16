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
    is_rigor_dominant,
    load_council_anchors,
    query_due_retrospectives,
    query_mission_contribution_distribution,
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


# ───────────────────────────────────────────────────────────────────
# Mission-alignment alert classes (operator binding directive 2026-05-16).
# Per CLAUDE.md "Mission alignment — non-negotiable" subsection of
# "Council hierarchy: 4-tier protocol":
#   - mission_contribution_distribution_alert (operational consequence 5)
#   - overdue_retrospective_alert (operational consequence 3)
#   - annual_gate_audit_alert (operational consequence 2)
# ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MissionAlignmentAlert:
    """One mission-alignment alert.

    The alert taxonomy maps the 3 operational consequences of the mission-
    alignment binding directive 2026-05-16 directly to operator-visible
    audit surfaces.
    """

    alert_class: str
    fired: bool
    summary: str
    details: dict[str, Any]


def compute_mission_contribution_distribution_alert(
    *,
    posterior_path: Path | None = None,
    lookback_days: int = 30,
    now_utc: _dt.datetime | None = None,
) -> MissionAlignmentAlert:
    """Operational consequence 5: rigor-dominance alert.

    Fires when `(rigor_overhead + apparatus_maintenance) / total > 60%` in
    the lookback window across all T2+ deliberations. The breakdown
    surfaces whether the council apparatus is producing innovation-
    enabling or innovation-blocking verdicts.

    Recommendation when fired: "council is producing more apparatus-
    maintenance than frontier-breaking work; re-evaluate per the mission-
    alignment standing directive".
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    now = now_utc or _dt.datetime.now(_dt.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.UTC)
    window_start = now - _dt.timedelta(days=lookback_days)
    since_iso = window_start.isoformat(timespec="seconds")
    dist = query_mission_contribution_distribution(
        since_utc=since_iso, posterior_path=posterior
    )
    fired = is_rigor_dominant(since_utc=since_iso, posterior_path=posterior)
    total = sum(dist.values())
    overhead = dist.get("rigor_overhead", 0) + dist.get("apparatus_maintenance", 0)
    overhead_pct = (overhead / total * 100.0) if total > 0 else 0.0
    if fired:
        summary = (
            f"MISSION-ALIGNMENT RIGOR-DOMINANT ALERT: in the last {lookback_days}d, "
            f"{overhead}/{total} T2+ deliberations ({overhead_pct:.0f}%) were "
            "classified `rigor_overhead` or `apparatus_maintenance` (vs the "
            "60% threshold). The council apparatus is producing more apparatus-"
            "maintenance than frontier-breaking work; re-evaluate per the "
            "mission-alignment standing directive "
            "(`feedback_council_apparatus_in_service_of_innovation_rigor_optimization_score_lowering_20260516.md`)."
        )
    else:
        summary = (
            f"Mission-alignment OK: {overhead}/{total} T2+ deliberations "
            f"({overhead_pct:.0f}%) classified rigor-overhead+apparatus-maintenance "
            f"in the last {lookback_days}d (below 60% threshold)."
        )
    return MissionAlignmentAlert(
        alert_class="mission_contribution_distribution_alert",
        fired=fired,
        summary=summary,
        details={
            "distribution": dict(dist),
            "total_t2_plus_deliberations": total,
            "rigor_overhead_plus_apparatus_count": overhead,
            "rigor_overhead_plus_apparatus_pct": overhead_pct,
            "threshold_pct": 60.0,
            "lookback_days": lookback_days,
            "window_start_utc": since_iso,
        },
    )


def compute_overdue_retrospective_alert(
    *,
    posterior_path: Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> MissionAlignmentAlert:
    """Operational consequence 3: overdue 30-day retrospective alert.

    Fires when any deferred / killed substrate's
    `deferred_substrate_retrospective_due_utc` is older than now.

    Recommendation when fired: "operator must run 30-day retrospective on
    deferred substrates" (per CLAUDE.md "Forbidden premature KILL" non-
    negotiable + "Mission alignment — non-negotiable" operational
    consequence 3).
    """
    posterior = posterior_path or DEFAULT_COUNCIL_POSTERIOR_PATH
    now = now_utc or _dt.datetime.now(_dt.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.UTC)
    due = query_due_retrospectives(
        as_of_utc=now.isoformat(timespec="seconds"), posterior_path=posterior
    )
    fired = len(due) > 0
    overdue_entries = [
        {
            "deliberation_id": d.deliberation_id,
            "deferred_substrate_id": d.deferred_substrate_id,
            "retrospective_due_utc": d.deferred_substrate_retrospective_due_utc,
            "council_verdict": d.council_verdict,
            "memory_path": d.memory_path,
            "related_deliberation_ids": list(d.related_deliberation_ids),
        }
        for d in due
    ]
    if fired:
        summary = (
            f"OVERDUE RETROSPECTIVE ALERT: {len(due)} deferred/killed substrate(s) "
            "have a 30-day retrospective DUE per CLAUDE.md 'Mission alignment "
            "— non-negotiable' operational consequence 3. Operator must run "
            "30-day retrospective on the deferred substrate(s) to evaluate "
            "whether the deferral cost actual score improvement, whether a "
            "sister substrate captured the same gain, OR whether the deferral "
            "should be reconsidered."
        )
    else:
        summary = "Mission-alignment retrospectives OK: no overdue 30-day retrospectives."
    return MissionAlignmentAlert(
        alert_class="overdue_retrospective_alert",
        fired=fired,
        summary=summary,
        details={
            "overdue_count": len(due),
            "overdue_substrates": overdue_entries,
            "as_of_utc": now.isoformat(timespec="seconds"),
        },
    )


# Canonical paths for the annual gate audit alert.
_CLAUDE_MD_REL_PATH = "CLAUDE.md"
_ANNUAL_AUDIT_MEMO_DIR_REL = ".omx/research"
_ANNUAL_AUDIT_MEMO_PATTERN = "annual_gate_audit_catalog_{n}_{year}.md"
_ANNUAL_AUDIT_WINDOW_DAYS = 365


def _extract_catalog_landing_years(claude_md_path: Path) -> dict[int, int]:
    """Best-effort extraction of {catalog_number: landing_year}.

    Scans CLAUDE.md for canonical catalog entry lines like
    ``267. `check_<name>` — ... 2026-05-15 ...``. Returns the first year
    matched on the catalog's row (the landing date). Tolerant of multiple
    dates on the same row (uses the first one).
    """
    import re as _re

    if not claude_md_path.exists():
        return {}
    out: dict[int, int] = {}
    catalog_re = _re.compile(r"^\s*(\d{1,4})\.\s+`check_")
    year_re = _re.compile(r"(\d{4})-\d{2}-\d{2}")
    try:
        text = claude_md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    for line in text.splitlines():
        m = catalog_re.match(line)
        if not m:
            continue
        n = int(m.group(1))
        ym = year_re.search(line)
        if ym:
            year = int(ym.group(1))
            # Keep first (earliest) year seen for each catalog #.
            if n not in out:
                out[n] = year
    return out


def compute_annual_gate_audit_alert(
    *,
    repo_root: Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> MissionAlignmentAlert:
    """Operational consequence 2: annual gate audit alert.

    Fires when any catalog # STRICT gate's landing date is >= 365 days
    ago AND no audit-verdict memo exists at
    `.omx/research/annual_gate_audit_catalog_<N>_<YYYY>.md`.

    Recommendation when fired: "Catalog #<N> annual audit overdue per
    Catalog #299 gate consolidation discipline".
    """
    now = now_utc or _dt.datetime.now(_dt.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.UTC)
    root = repo_root or _REPO_ROOT
    claude_md = Path(root) / _CLAUDE_MD_REL_PATH
    memo_dir = Path(root) / _ANNUAL_AUDIT_MEMO_DIR_REL
    landings = _extract_catalog_landing_years(claude_md)
    overdue: list[dict[str, Any]] = []
    current_year = now.year
    for n, year in landings.items():
        # A gate is "due for audit" if its landing year + 1 <= current year
        # AND the calendar gap from landing is >= 365 days. Use a fixed-day
        # heuristic by comparing year-only to keep behavior deterministic.
        age_years = current_year - year
        if age_years < 1:
            continue
        # Look for an audit memo for either the current year OR any year
        # >= landing_year + 1 (operator may have run the audit late OR
        # backfilled it; either satisfies the gate).
        audit_satisfied = False
        for candidate_year in range(year + 1, current_year + 1):
            candidate_memo = memo_dir / _ANNUAL_AUDIT_MEMO_PATTERN.format(
                n=n, year=candidate_year
            )
            if candidate_memo.exists():
                audit_satisfied = True
                break
        if not audit_satisfied:
            overdue.append({
                "catalog_number": n,
                "landed_year": year,
                "age_years": age_years,
                "expected_memo_path": str(
                    memo_dir / _ANNUAL_AUDIT_MEMO_PATTERN.format(
                        n=n, year=current_year
                    )
                ),
            })
    overdue.sort(key=lambda d: (-d["age_years"], d["catalog_number"]))
    fired = len(overdue) > 0
    if fired:
        cite_ns = ", ".join(f"#{e['catalog_number']}" for e in overdue[:10])
        if len(overdue) > 10:
            cite_ns += f", … (+{len(overdue) - 10} more)"
        summary = (
            f"ANNUAL GATE AUDIT ALERT: {len(overdue)} Catalog # STRICT gate(s) "
            f"overdue for annual audit per CLAUDE.md 'Mission alignment — "
            "non-negotiable' operational consequence 2 + Catalog #299 (gate "
            f"consolidation discipline): {cite_ns}. Operator should review "
            "each gate's empirical contribution to score lowering (incidents "
            "prevented vs false positives blocking innovation) and decide "
            "retire / scope-narrow / preserve per net contribution."
        )
    else:
        summary = "Annual gate audit OK: no overdue catalog-gate audits."
    return MissionAlignmentAlert(
        alert_class="annual_gate_audit_alert",
        fired=fired,
        summary=summary,
        details={
            "overdue_count": len(overdue),
            "overdue_gates": overdue,
            "audit_window_days": _ANNUAL_AUDIT_WINDOW_DAYS,
        },
    )


def audit_cadence(
    *,
    posterior_path: Path | None = None,
    lookback_days: int = 30,
    now_utc: _dt.datetime | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the cadence audit over the canonical posterior.

    Returns a dict with keys:
      - schema: 'council_tier_cadence_audit_v2'
      - lookback_days: int
      - window_start_utc / window_end_utc: ISO timestamps
      - per_tier: list of TierVerdict-as-dict
      - any_over_cadence: bool
      - mission_alignment_alerts: list of 3 MissionAlignmentAlert-as-dict
      - any_mission_alignment_alert_fired: bool
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
    # Mission-alignment alerts (operator binding directive 2026-05-16).
    mission_alerts = [
        compute_mission_contribution_distribution_alert(
            posterior_path=posterior, lookback_days=lookback_days, now_utc=now,
        ),
        compute_overdue_retrospective_alert(
            posterior_path=posterior, now_utc=now,
        ),
        compute_annual_gate_audit_alert(
            repo_root=repo_root, now_utc=now,
        ),
    ]
    any_mission_alert = any(a.fired for a in mission_alerts)
    return {
        "schema": "council_tier_cadence_audit_v2",
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
        "mission_alignment_alerts": [
            {
                "alert_class": a.alert_class,
                "fired": a.fired,
                "summary": a.summary,
                "details": a.details,
            }
            for a in mission_alerts
        ],
        "any_mission_alignment_alert_fired": any_mission_alert,
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
    # Mission-alignment alerts (operator binding directive 2026-05-16).
    mission_alerts = report.get("mission_alignment_alerts", [])
    if mission_alerts:
        lines.append("")
        lines.append("Mission-alignment alerts (CLAUDE.md 'Mission alignment — non-negotiable')")
        lines.append("-" * 50)
        for alert in mission_alerts:
            marker = "⚠ FIRED" if alert["fired"] else "✓ OK"
            lines.append(f"{marker} [{alert['alert_class']}] {alert['summary']}")
    if report.get("any_mission_alignment_alert_fired"):
        lines.append("")
        lines.append(
            "OPERATOR ACTION REQUESTED: at least one mission-alignment alert "
            "is FIRED. Review per-alert detail above per CLAUDE.md 'Mission "
            "alignment — non-negotiable' subsection."
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
    # Exit 1 when EITHER (a) any tier is OVER_CADENCE OR (b) any mission-
    # alignment alert is FIRED per the operator binding directive 2026-05-16.
    if report["any_over_cadence"] or report.get("any_mission_alignment_alert_fired"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
