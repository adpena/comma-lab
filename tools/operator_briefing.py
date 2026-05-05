#!/usr/bin/env python3
"""Operator briefing: runs the dispatch trio in sequence — one command, full state.

Phases:
  1. Pre-dispatch (Pareto frontier + dispatch one-liners per non-dominated bits)
  2. Post-dispatch general (sorted view of every contest_auth_eval.json on disk)
  3. Post-dispatch apogee_intN (predicted-vs-actual reconciliation)

Use cases:
  - Start of session: see what's ready to dispatch + what's already landed
  - After a dispatch lands: see the new score in dashboard + reconciler verdict
  - Quick "where am I?" between deep work blocks

Usage:
  .venv/bin/python tools/operator_briefing.py                   # all 3 phases
  .venv/bin/python tools/operator_briefing.py --top 10           # cap dashboard rows
  .venv/bin/python tools/operator_briefing.py --skip-pareto      # only dashboard + reconciler
  .venv/bin/python tools/operator_briefing.py --json             # machine-readable composite
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"

PARETO = TOOLS / "apogee_intN_pareto.py"
DASHBOARD = TOOLS / "score_dashboard.py"
RECONCILER = TOOLS / "predicted_vs_actual_reconciler.py"


# Phase-1 supplementary lanes: pre-registered dispatches that don't fit the
# apogee_intN Pareto matrix but are operator-launchable now. Each entry is
# (lane_id, predicted_band, estimated_cost, council_priority, one-liner).
PHASE_1_SUPPLEMENTARY_LANES = [
    {
        "lane_id": "lane_pr106_latent_sidecar",
        "name": "PR106 + per-pair latent-correction sidecar (PR100 hnerv_lc_v2 pattern)",
        "predicted_band": (0.205, 0.208),
        "estimated_cost_usd": 0.60,
        "council_priority": 1,
        "max_dph": 0.30,
        "one_liner": (
            ".venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            "  --lane-script scripts/remote_lane_pr106_latent_sidecar.sh \\\n"
            "  --label lane_pr106_latent_sidecar \\\n"
            "  --predicted-band 0.205 0.208 \\\n"
            "  --estimated-cost 0.60 --council-priority 1 --max-dph 0.30"
        ),
    },
]


# Phase-4 gated lanes: pre-registered dispatches that REQUIRE a prior empirical
# result before launch. Per docs/INDEX_score_aware_sidechannel_thread_20260504.md
# decision pipeline, sequential validation prevents wasting GPU spend on
# stacking lanes that interact unexpectedly. Each entry adds a `gate_condition`
# string operator must satisfy before running the one-liner.
PHASE_4_GATED_LANES = [
    {
        "lane_id": "lane_pr106_yshift_sidechannel",
        "name": "PR106 + per-frame Y-shift sidechannel (codex_metric_yshift SC01 mode-7 pattern)",
        "predicted_band": (0.2065, 0.2080),
        "estimated_cost_usd": 0.40,
        "council_priority": 2,
        "max_dph": 0.30,
        "gate_condition": (
            "DISPATCH ONLY IF lane_pr106_latent_sidecar lands < 0.20800 "
            "[contest-CUDA] (per docs/INDEX_score_aware_sidechannel_thread_20260504.md "
            "TICK 2). Verify via: `tools/score_dashboard.py --filter pr106_latent_sidecar`."
        ),
        "one_liner": (
            ".venv/bin/python scripts/launch_lane_on_vastai.py full \\\n"
            "  --lane-script scripts/remote_lane_pr106_yshift_sidechannel.sh \\\n"
            "  --label lane_pr106_yshift_sidechannel \\\n"
            "  --predicted-band 0.2065 0.2080 \\\n"
            "  --estimated-cost 0.40 --council-priority 2 --max-dph 0.30 \\\n"
            "  --env PR106_YSHIFT_MODE=brute_force"
        ),
    },
]


def _format_supplementary_lanes() -> str:
    lines = []
    for lane in PHASE_1_SUPPLEMENTARY_LANES:
        lo, hi = lane["predicted_band"]
        lines.append(
            f"  • {lane['lane_id']} — {lane['name']}\n"
            f"    predicted [{lo:.4f}, {hi:.4f}]   est ${lane['estimated_cost_usd']:.2f}   "
            f"council priority {lane['council_priority']}\n"
            f"    Operator one-liner:\n"
            f"      {lane['one_liner']}"
        )
    return "\n\n".join(lines) if lines else "  (none)"


def _format_gated_lanes() -> str:
    lines = []
    for lane in PHASE_4_GATED_LANES:
        lo, hi = lane["predicted_band"]
        lines.append(
            f"  • {lane['lane_id']} — {lane['name']}\n"
            f"    predicted [{lo:.4f}, {hi:.4f}]   est ${lane['estimated_cost_usd']:.2f}   "
            f"council priority {lane['council_priority']}\n"
            f"    GATE: {lane['gate_condition']}\n"
            f"    Operator one-liner (post-gate):\n"
            f"      {lane['one_liner']}"
        )
    return "\n\n".join(lines) if lines else "  (none)"


def _run(script: Path, extra_args: list[str] | None = None) -> str:
    args = [sys.executable, str(script)]
    if extra_args:
        args.extend(extra_args)
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return (
            f"(tool exited with code {proc.returncode}; stderr: "
            f"{proc.stderr.strip()[:200]})"
        )
    return proc.stdout


def _run_json(script: Path, extra_args: list[str] | None = None) -> dict:
    text = _run(script, (extra_args or []) + ["--json"])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_error": "non-JSON output", "_stdout": text[:500]}


def _section(title: str, body: str) -> str:
    bar = "═" * len(title)
    return f"\n{bar}\n{title}\n{bar}\n\n{body}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20,
                        help="Cap dashboard rows (default: 20).")
    parser.add_argument("--skip-pareto", action="store_true",
                        help="Skip Phase 1 (Pareto pre-dispatch matrix).")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip Phase 2 (general score dashboard).")
    parser.add_argument("--skip-reconciler", action="store_true",
                        help="Skip Phase 3 (apogee_intN predicted-vs-actual).")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip Phase 4 (gated next-tick lanes).")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable composite JSON output.")
    args = parser.parse_args(argv)

    # Verify all 3 tools exist
    for tool in (PARETO, DASHBOARD, RECONCILER):
        if not tool.is_file():
            print(f"FATAL: missing dependency tool {tool.relative_to(REPO_ROOT)}",
                  file=sys.stderr)
            return 2

    if args.json:
        out = {}
        if not args.skip_pareto:
            out["pareto"] = _run_json(PARETO)
            out["supplementary_lanes"] = PHASE_1_SUPPLEMENTARY_LANES
        if not args.skip_dashboard:
            out["dashboard"] = _run_json(DASHBOARD, ["--top", str(args.top)])
        if not args.skip_reconciler:
            out["reconciler"] = _run_json(RECONCILER)
        if not args.skip_gated:
            out["gated_lanes"] = PHASE_4_GATED_LANES
        print(json.dumps(out, indent=2, default=str))
        return 0

    # Human-readable composite
    parts: list[str] = ["OPERATOR BRIEFING — dispatch trio"]
    if not args.skip_pareto:
        parts.append(_section(
            "Phase 1 — Pre-dispatch: apogee_intN Pareto frontier",
            _run(PARETO).strip(),
        ))
        parts.append(_section(
            "Phase 1 supplementary — pre-registered non-Pareto lanes",
            _format_supplementary_lanes(),
        ))
    if not args.skip_dashboard:
        parts.append(_section(
            f"Phase 2 — Post-dispatch (general): top {args.top} contest scores on disk",
            _run(DASHBOARD, ["--top", str(args.top)]).strip(),
        ))
    if not args.skip_reconciler:
        parts.append(_section(
            "Phase 3 — Post-dispatch (apogee_intN): predicted-vs-actual reconciliation",
            _run(RECONCILER).strip(),
        ))
    if not args.skip_gated:
        parts.append(_section(
            "Phase 4 — Gated next-tick lanes (sequential validation)",
            _format_gated_lanes(),
        ))
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
