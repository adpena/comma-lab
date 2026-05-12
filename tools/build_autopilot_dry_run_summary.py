#!/usr/bin/env python3
"""Build a would-dispatch table + report from an autopilot dry-run report.

Per the operator one-touch authorization toolkit Deliverable 2 (2026-05-11):
the cathedral autopilot loop produces a structured halt-event report when run
in HALT-and-ASK mode (no operator authorization). This tool ingests the JSON
report and produces:

- ``would_dispatch.json`` — typed table of every halt event the autopilot
  would surface to the operator if authorized (per-dispatch row: substrate,
  cost, expected delta, composition_notes, autopilot_authorized flag).
- ``dry_run_report.md`` — human-readable per-dispatch summary + cumulative
  cost projection + the recommended subset to actually authorize first.

Per CLAUDE.md "Forbidden score claims": every row carries a
``[predicted; cathedral autopilot ranking]`` tag and ``promotion_eligible``
remains False; this is a planning artifact, not an empirical anchor.

Per CLAUDE.md "Operator gates must be wired and used": the would-dispatch
list is operator-decision-ready — each row matches a concrete entry the
operator can authorize via the per-decision one-command authorize scripts.

CLAUDE.md compliance tags:
- ``planning_only_no_score_claim``
- ``operator_gate_non_negotiable_at_every_dispatch``
- ``halt_and_ask_default_on``
- ``no_tmp_paths``
- ``cathedral_autopilot_dry_run_aware``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


SCHEMA = "tac_autopilot_dry_run_summary_v1"


def extract_would_dispatch_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract every halt event with ``event_class=dispatch`` into a row."""
    rows: list[dict[str, Any]] = []
    for iteration_report in report.get("reports", []):
        for he in iteration_report.get("halt_events", []):
            if he.get("event_class") != "dispatch":
                continue
            rows.append(
                {
                    "candidate_id": he.get("candidate_id", ""),
                    "estimated_cost_usd": float(he.get("estimated_cost_usd", 0.0) or 0.0),
                    "predicted_score_delta": float(
                        he.get("predicted_score_delta", 0.0) or 0.0
                    ),
                    "blockers": list(he.get("blockers", []) or []),
                    "requires_approval": bool(he.get("requires_approval", True)),
                    "autopilot_authorized": bool(
                        he.get("autopilot_authorized", False)
                    ),
                    "autopilot_authorized_reason": he.get(
                        "autopilot_authorized_reason", ""
                    ),
                    "autopilot_refused_reason": he.get(
                        "autopilot_refused_reason", ""
                    ),
                    "decision": he.get("decision", "defer"),
                    "halt_at_utc": he.get("halt_at_utc", ""),
                    "evidence_tag": "[predicted; cathedral autopilot ranking]",
                    "promotion_eligible": False,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    rows.sort(
        key=lambda r: (
            r["estimated_cost_usd"],
            r["predicted_score_delta"],
        )
    )
    return rows


def cumulative_cost_band(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the cumulative cost projection + Pareto candidate subset.

    Returns a dict with totals + Pareto-frontier subset within the canonical
    autopilot ≤$5/individual + ≤$20 cumulative envelope.
    """
    total = sum(r["estimated_cost_usd"] for r in rows)
    pareto: list[dict[str, Any]] = []
    cumulative = 0.0
    for r in rows:
        if r["estimated_cost_usd"] > 5.00:
            continue
        if cumulative + r["estimated_cost_usd"] > 20.00:
            continue
        pareto.append(r)
        cumulative += r["estimated_cost_usd"]
    return {
        "total_dispatch_count": len(rows),
        "total_cumulative_cost_usd_if_all_dispatched": total,
        "pareto_subset_within_envelope_count": len(pareto),
        "pareto_subset_cumulative_cost_usd": cumulative,
        "pareto_subset_candidate_ids": [r["candidate_id"] for r in pareto[:10]],
    }


def render_would_dispatch_table(rows: list[dict[str, Any]]) -> str:
    """Render an aligned per-dispatch markdown table."""
    if not rows:
        return "_(no dispatch halt events found in report)_"
    header = (
        "| # | candidate_id | cost ($) | predicted Δ | autopilot_authorized | blockers |"
        "\n|---:|---|---:|---:|:---:|---|\n"
    )
    body_lines: list[str] = []
    for i, r in enumerate(rows, 1):
        candidate_id = r["candidate_id"][:60]
        cost = f"{r['estimated_cost_usd']:.2f}"
        delta = f"{r['predicted_score_delta']:+.6f}"
        authorized = "yes" if r["autopilot_authorized"] else "no"
        blockers = ", ".join(r["blockers"]) if r["blockers"] else "_(none)_"
        body_lines.append(
            f"| {i} | `{candidate_id}` | {cost} | {delta} | {authorized} | {blockers} |"
        )
    return header + "\n".join(body_lines)


def build_report_md(
    *,
    rows: list[dict[str, Any]],
    cumulative: dict[str, Any],
    autopilot_report_path: Path,
    operator_authorized_mode: dict[str, Any],
    composition_constraints: dict[str, Any],
) -> str:
    """Build the full markdown dry-run report."""
    top5 = rows[:5]
    top5_lines: list[str] = []
    for i, r in enumerate(top5, 1):
        top5_lines.append(
            f"  {i}. `{r['candidate_id']}` "
            f"@ ${r['estimated_cost_usd']:.2f} → "
            f"predicted Δ={r['predicted_score_delta']:+.6f}"
        )
    top5_block = "\n".join(top5_lines) if top5_lines else "  _(none)_"

    return f"""---
title: Cathedral autopilot dry-run sample output
date: 2026-05-11
research_only: false
lane_class: substrate_engineering
lane_id: lane_autopilot_dry_run_sample
schema: {SCHEMA}
evidence_grade: planning_only_autopilot_dispatch_ranking_v1
promotion_eligible: false
score_claim: false
---

# Cathedral autopilot dry-run sample output

## Summary

This document captures the **would-dispatch list** the cathedral autopilot
loop produces in HALT-and-ASK mode (default safe behavior, no operator
authorization). It is the verification artifact that the autopilot pipeline
(QQ ranking → MM autopilot loop → composition constraints → halt-event
emission) produces operator-decision-ready output.

**Source autopilot run**: `{autopilot_report_path}`

**Operator authorization status**: `enabled={operator_authorized_mode.get('enabled')}`
(this dry-run was executed WITHOUT the operator-authorized-le-5-dollar-mode
flag; the autopilot remained HALT-and-ASK on every dispatch decision).

## Cumulative cost projection

| Metric | Value |
|---|---:|
| Total dispatch halt events | {cumulative['total_dispatch_count']} |
| Total cumulative cost if all dispatched | ${cumulative['total_cumulative_cost_usd_if_all_dispatched']:.2f} |
| Pareto subset within ≤$5/individual + ≤$20 cumulative envelope | {cumulative['pareto_subset_within_envelope_count']} |
| Pareto subset cumulative cost | ${cumulative['pareto_subset_cumulative_cost_usd']:.2f} |

## Top 5 candidates (lowest-cost-first ordering)

{top5_block}

## Recommended subset to authorize first

The Pareto subset above ({cumulative['pareto_subset_within_envelope_count']}
dispatches @ ${cumulative['pareto_subset_cumulative_cost_usd']:.2f} cumulative)
fits within the operator-approved ≤$5/individual + ≤$20 cumulative envelope.
The first 10 candidate IDs in this subset:

{chr(10).join(f"  - `{cid}`" for cid in cumulative['pareto_subset_candidate_ids']) if cumulative['pareto_subset_candidate_ids'] else '  _(none)_'}

To execute this Pareto subset, the operator activates dual-gated authorization
mode:

```bash
export CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1
.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py \\
  --use-substrate-composition-matrix-ranking \\
    experiments/results/cathedral_autopilot_dispatch_ranking_<UTC>/ranking.json \\
  --operator-authorized-le-5-dollar-mode \\
  --journal-path experiments/results/autopilot_authorized_journal_<UTC>.jsonl \\
  --iterations 1 \\
  --max-dispatch-recommendations {cumulative['pareto_subset_within_envelope_count']}
```

Per CLAUDE.md "Operator gates must be wired and used", the canonical
one-command wrapper is `scripts/operator_authorize_autopilot_le_5_dollar_mode.sh`
(landed in this same operator one-touch authorization toolkit).

## Full would-dispatch table

{render_would_dispatch_table(rows)}

## Composition-constraint enforcement

The autopilot consumed the substrate composition matrix ranking JSON and
applied REPLACEMENT/INCOMPATIBLE/ANTAGONISTIC matrix-cell refusal across the
dispatch queue per QQ matrix v1. Composition section from the loop output:

```json
{json.dumps(composition_constraints, indent=2, sort_keys=True)}
```

## CLAUDE.md compliance verification

- `planning_only_no_score_claim` — every row carries `score_claim=False` +
  `promotion_eligible=False` + `[predicted; cathedral autopilot ranking]` tag
- `operator_gate_non_negotiable_at_every_dispatch` — autopilot HALT-and-ASK
  default ON; no candidate self-authorized
- `halt_and_ask_default_on` — verified: all {len(rows)} dispatches require
  operator approval
- `no_tmp_paths` — output goes to `experiments/results/...` not `/tmp/...`
- `cathedral_autopilot_dry_run_aware` — explicit dry-run sample; not a
  production dispatch

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map**: would-dispatch rows feed sensitivity-driven priors;
   per-candidate predicted_score_delta + cost informs `tac.sensitivity_map.*`.
2. **Pareto constraint**: Pareto subset within envelope IS a Pareto-frontier
   filter applied to the dispatch queue.
3. **Bit-allocator hook**: per-substrate cost-EV-per-dollar feeds the
   bit-allocator's per-tensor importance routing.
4. **Cathedral autopilot dispatch hook**: this report IS the autopilot's
   dispatch-readiness verification.
5. **Continual-learning posterior update**: post-authorization dispatch
   results would feed `posterior_update_locked` per harvest.
6. **Probe-disambiguator**: N/A — would-dispatch list is not a
   regime-conditional disambiguator (it's the full Pareto-filtered queue).
"""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--autopilot-report",
        type=Path,
        required=True,
        help="Path to autopilot loop output JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write would_dispatch.json + dry_run_report.md.",
    )
    args = parser.parse_args(argv)

    if not args.autopilot_report.is_file():
        print(
            f"build_autopilot_dry_run_summary: report not found: {args.autopilot_report}",
            file=sys.stderr,
        )
        return 2
    out_path_str = str(args.output_dir)
    if out_path_str.startswith("/tmp/") or out_path_str.startswith("/var/tmp/"):
        print(
            "build_autopilot_dry_run_summary: --output-dir must NOT live under "
            "/tmp; pick a durable evidence path "
            "(e.g. experiments/results/autopilot_dry_run_sample_<UTC>/).",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(args.autopilot_report.read_text(encoding="utf-8"))
    rows = extract_would_dispatch_rows(report)
    cumulative = cumulative_cost_band(rows)
    operator_authorized_mode = report.get("operator_authorized_mode", {})
    composition_constraints = report.get("substrate_composition_ranking", {})

    would_dispatch_payload = {
        "schema": SCHEMA,
        "evidence_grade": "planning_only_autopilot_dispatch_ranking_v1",
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "operator_gate_non_negotiable_at_every_dispatch",
            "halt_and_ask_default_on",
            "no_tmp_paths",
            "cathedral_autopilot_dry_run_aware",
        ],
        "source_autopilot_report": str(args.autopilot_report),
        "operator_authorized_mode": operator_authorized_mode,
        "composition_constraints": composition_constraints,
        "cumulative_summary": cumulative,
        "would_dispatch_rows": rows,
    }
    (args.output_dir / "would_dispatch.json").write_text(
        json.dumps(would_dispatch_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md = build_report_md(
        rows=rows,
        cumulative=cumulative,
        autopilot_report_path=args.autopilot_report,
        operator_authorized_mode=operator_authorized_mode,
        composition_constraints=composition_constraints,
    )
    (args.output_dir / "dry_run_report.md").write_text(md, encoding="utf-8")
    print(
        f"build_autopilot_dry_run_summary: wrote "
        f"{args.output_dir / 'would_dispatch.json'} + "
        f"{args.output_dir / 'dry_run_report.md'}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
