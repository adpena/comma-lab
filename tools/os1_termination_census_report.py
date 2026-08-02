#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit a solver receipt's TERMINATION CENSUS retroactively, at zero new compute.

Reads a per-item JSONL receipt that already recorded a forward/evaluation count and
reports how many solves CONVERGED versus how many stopped on a BOUND, using the
registered law ``ddm_os1_termination_census_from_cost_proxy_v1``.

The point is that this costs NOTHING: the receipt is already on disk.  ``ddm_sv1`` spent
1,385 scorer evaluations answering the same question on a sister solve that recorded no
cost proxy -- which is the whole argument for recording one.

Example (the measured anchor)::

    .venv/bin/python tools/os1_termination_census_report.py \\
        --receipt /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl \\
        --cost-field n_forwards --objective-field d_pose_solved --tolerance 1e-6 \\
        --relin-bound 4 --fd-per-relin 6 --ladder-levels 4 --line-search-points 2

Advisory only: ``score_claim=false``, ``promotable=false``.  A bound verdict is a request
for ONE measurement (free the bound, re-measure), never a claim that freeing it will pay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tac.canonical_equations.ddm_os1_termination_census_from_cost_proxy_20260802 import (
    termination_census,
)

AXIS = "[macOS-CPU advisory] reconstruction — score_claim=false, promotable=false"


def load_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return parsed rows and the count of unparsable lines (the denominator matters)."""
    rows: list[dict[str, Any]] = []
    bad = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    # Try WHOLE-FILE json first. Sniffing on the suffix or on a "\n{" substring
    # misroutes a pretty-printed .json (whose nested objects contain "\n{") into the
    # line-by-line branch, where every line fails to parse and the file reads as empty
    # — a silent vacuous PASS.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(obj, list):
            rows = [o for o in obj if isinstance(o, dict)]
            return rows, len(obj) - len(rows)
        if isinstance(obj, dict):
            return [obj], 0
        return [], 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            bad += 1
    return rows, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receipt", type=Path, required=True,
                    help="per-item JSON/JSONL receipt carrying the cost proxy")
    ap.add_argument("--cost-field", required=True,
                    help="field holding the forward/evaluation count (e.g. n_forwards)")
    ap.add_argument("--objective-field", default=None,
                    help="field holding the final objective; needed to decide convergence")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="convergence tolerance in objective units")
    # No defaults on the SHAPE: these describe a specific loop and guessing them would
    # silently produce a confident wrong census. Required, per never-invent-flags.
    ap.add_argument("--relin-bound", type=int, required=True)
    ap.add_argument("--fd-per-relin", type=int, required=True)
    ap.add_argument("--ladder-levels", type=int, required=True)
    ap.add_argument("--line-search-points", type=int, required=True)
    ap.add_argument("--init-cost", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    if not args.receipt.exists():
        print(f"receipt not found: {args.receipt}", file=sys.stderr)
        return 2
    if (args.objective_field is None) != (args.tolerance is None):
        ap.error("--objective-field and --tolerance must be given together; a tolerance "
                 "with no objective cannot decide convergence")

    rows, unparsable = load_rows(args.receipt)
    # a null cost is as unusable as an absent one; counting it as present would
    # traceback later instead of showing up in the denominator
    missing_cost = sum(1 for r in rows if r.get(args.cost_field) is None)
    usable = [r for r in rows if r.get(args.cost_field) is not None]
    if not usable:
        print(f"NO rows carry {args.cost_field!r} — VACUOUS scope, not a clean bill "
              f"(rows={len(rows)}, unparsable={unparsable})", file=sys.stderr)
        return 3

    objective = None
    if args.objective_field is not None:
        if any(r.get(args.objective_field) is None for r in usable):
            print(f"some rows lack {args.objective_field!r}; convergence would be "
                  f"undecidable for them — refusing rather than partially guessing",
                  file=sys.stderr)
            return 4
        objective = [float(r[args.objective_field]) for r in usable]

    out = termination_census(
        [int(r[args.cost_field]) for r in usable],
        relin_bound=args.relin_bound,
        fd_per_relin=args.fd_per_relin,
        ladder_levels=args.ladder_levels,
        line_search_points=args.line_search_points,
        init_cost=args.init_cost,
        objective=objective,
        tolerance=args.tolerance,
    )
    report = {
        "axis": AXIS,
        "receipt": str(args.receipt),
        "rows_total": len(rows),
        "rows_unparsable": unparsable,
        "rows_missing_cost_field": missing_cost,
        "rows_used": len(usable),
        "new_scorer_evaluations": 0,
        **{k: v for k, v in out.items() if k != "states"},
    }
    if args.json:
        print(json.dumps(report, indent=1))
        return 0

    print(f"{AXIS}\n{args.receipt}")
    print(f"  DENOMINATOR rows={len(rows)} unparsable={unparsable} "
          f"missing_cost_field={missing_cost} used={len(usable)}")
    print(f"  verdict={out['verdict']}  sufficient_for_verdict={out['sufficient_for_verdict']}")
    if out.get("insufficiency_reason"):
        print(f"  insufficiency: {out['insufficiency_reason']}")
    for state, agg in out["census"].items():
        mass = agg["objective_mass_fraction"]
        mass_s = "     —" if mass is None else f"{100 * mass:5.1f}%"
        print(f"    {state:24s} n={agg['count']:5d} ({100 * agg['fraction']:5.1f}%)"
              f"  objective-mass {mass_s}")
    print("  a bound verdict is a MEASUREMENT REQUEST, never a claim that freeing pays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
