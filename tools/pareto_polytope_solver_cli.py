#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""META-LIFT-2 Pareto polytope unified solver CLI.

Per the 11th standing directive ORDER discipline: this is the operator-
facing sister of :mod:`tac.pareto_polytope_unified_solver`. Loads the
canonical META-LIFT-1 cross-substrate analysis from
``.omx/state/cross_substrate_master_gradient_analyses.jsonl``, invokes
the canonical Pareto polytope Dykstra solver, and emits the unified
bit-budget allocation (human-readable summary or JSON).

Usage:
    .venv/bin/python tools/pareto_polytope_solver_cli.py
    .venv/bin/python tools/pareto_polytope_solver_cli.py --target-aggregate-delta-s -0.001
    .venv/bin/python tools/pareto_polytope_solver_cli.py --max-iterations 200 --tol 1e-8
    .venv/bin/python tools/pareto_polytope_solver_cli.py --json
    .venv/bin/python tools/pareto_polytope_solver_cli.py --persist-to-ledger

Exit codes:
    0  CLEAN — solution emitted (stdout)
    1  INFEASIBLE_WITH_BOUND — Cauchy-Schwarz aggregate bound or budget
       caps make the problem infeasible (e.g. target ΔS requires more
       bytes than the cross-substrate corpus allows under the canonical
       constraints)
    2  CLI error (invalid arguments / unrecognized flag value /
       no META-LIFT-1 analyses available)

Per Catalog #341 routing markers: this CLI's output is OBSERVABILITY-
ONLY by construction. Promotion of any allocation entry to a contest
dispatch decision REQUIRES paired-CUDA empirical anchor per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable.

Sister of:
  - ``tools/cross_substrate_master_gradient_cli.py`` (META-LIFT-1 —
    produces the canonical analysis this CLI consumes)
  - ``tools/extract_master_gradient.py`` (canonical per-substrate
    gradient extractor — upstream of META-LIFT-1)
  - ``tools/list_canonical_equations.py`` (canonical equations registry)
  - ``tools/cathedral_autopilot_autonomous_loop.py`` (downstream ranker
    consumer via the sister cathedral consumer)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.cross_substrate_master_gradient_analyzer import (  # noqa: E402
    CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH,
    load_analyses_strict,
)
from tac.pareto_polytope_unified_solver import (  # noqa: E402
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    append_solution_locked,
    build_problem_spec_from_meta_lift_1_analysis,
    solve_pareto_polytope_via_dykstra_projections,
)


def _load_latest_meta_lift_1_analysis(ledger_path: Path | None = None) -> dict | None:
    """Load the most-recent META-LIFT-1 analysis from the canonical ledger.

    Returns ``None`` if the ledger is missing or empty (signals
    UPSTREAM_GAP to the CLI).
    """
    target = ledger_path or CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH
    if not target.exists():
        return None
    rows = load_analyses_strict(target)
    if not rows:
        return None
    # Most-recent row wins (latest written_at_utc).
    return max(rows, key=lambda r: r.get("written_at_utc", ""))


def _format_human_summary(solution_dict: dict) -> str:
    """Format the solution as a human-readable summary."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("META-LIFT-2 PARETO POLYTOPE UNIFIED BIT-BUDGET ALLOCATION")
    lines.append("=" * 78)
    lines.append(f"solution_id:                  {solution_dict['solution_id']}")
    lines.append(f"measurement_utc:              {solution_dict['measurement_utc']}")
    lines.append(f"axis_tag:                     {solution_dict['axis_tag']}")
    lines.append(f"evidence_grade:               {solution_dict['evidence_grade']}")
    lines.append(f"score_claim:                  {solution_dict['score_claim']}")
    lines.append(f"promotable:                   {solution_dict['promotable']}")
    lines.append(f"canonical_equation_id:        {solution_dict['canonical_equation_id']}")
    lines.append(f"canonical_equation_status:    {solution_dict['canonical_equation_status']}")
    if solution_dict.get("upstream_meta_lift_1_analysis_id"):
        lines.append(
            f"upstream_meta_lift_1_id:      {solution_dict['upstream_meta_lift_1_analysis_id']}"
        )
    lines.append("")
    lines.append(
        f"converged:                    {solution_dict['converged']} "
        f"(after {solution_dict['n_iterations_to_convergence']} iterations)"
    )
    allocation = solution_dict["allocation"]
    lines.append(
        f"feasible:                     {allocation['feasible']} "
        f"(residual={allocation['feasibility_residual']:.6e})"
    )
    lines.append("")
    lines.append(
        f"aggregate_total_bytes:        {allocation['aggregate_total_bytes_allocated']:.2f}"
    )
    lines.append(
        f"aggregate_predicted_delta_s:  {allocation['aggregate_predicted_delta_s']:+.6e}"
    )
    lines.append("  (Cauchy-Schwarz upper bound on |ΔS|; signed via gradient direction)")
    lines.append("")
    lines.append("Per-substrate allocations:")
    lines.append("  " + " " * 14 + "    seg          pose         rate         aggregate")
    shas = allocation["substrate_archive_sha256s"]
    per_axis = allocation["per_substrate_per_axis_allocations"]
    aggregate = allocation["per_substrate_aggregate_allocations"]
    for idx, sha in enumerate(shas):
        sha_short = sha[:12]
        seg = per_axis[idx][0] if idx < len(per_axis) else 0.0
        pose = per_axis[idx][1] if idx < len(per_axis) else 0.0
        rate = per_axis[idx][2] if idx < len(per_axis) else 0.0
        agg = aggregate[idx] if idx < len(aggregate) else 0.0
        lines.append(
            f"  {sha_short}    "
            f"{seg:>10.2f}  {pose:>10.2f}  {rate:>10.2f}  {agg:>10.2f}"
        )
    lines.append("")
    if solution_dict.get("convergence_history"):
        history = solution_dict["convergence_history"]
        if len(history) >= 1:
            lines.append(
                f"Convergence history (first→last delta): {history[0]:.6e} → {history[-1]:.6e}"
            )
    lines.append("")
    lines.append("Per Catalog #341 + CLAUDE.md 'Apples-to-apples evidence discipline':")
    lines.append("  EVERY allocation is observability-only. Promotion REQUIRES paired-CUDA")
    lines.append("  empirical anchor per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA'.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pareto_polytope_solver_cli",
        description=(
            "META-LIFT-2 Pareto polytope unified solver CLI. Loads the latest "
            "META-LIFT-1 cross-substrate analysis from "
            ".omx/state/cross_substrate_master_gradient_analyses.jsonl, runs the "
            "canonical Dykstra alternating projections solver per Boyd 2004 §7.2, "
            "and emits the unified per-substrate per-axis bit-budget allocation."
        ),
    )
    parser.add_argument(
        "--target-aggregate-delta-s",
        type=float,
        default=None,
        help=(
            "Optional target aggregate ΔS for budget-driven allocation. When "
            "provided, the solver searches for an allocation achieving this ΔS "
            "subject to constraints. When None, the solver minimizes ΔS subject "
            "to constraints."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            f"Max Dykstra alternating projections iterations "
            f"(default: {DEFAULT_MAX_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            f"Convergence tolerance on ||x^(k+1) - x^k||_2 "
            f"(default: {DEFAULT_TOLERANCE})."
        ),
    )
    parser.add_argument(
        "--per-axis-cap-fraction",
        type=float,
        default=0.10,
        help=(
            "Per-substrate per-axis byte-budget cap as fraction of total "
            "archive bytes (default: 0.10 = 10 pct per axis)."
        ),
    )
    parser.add_argument(
        "--aggregate-cap-fraction",
        type=float,
        default=0.20,
        help=(
            "Per-substrate aggregate byte-budget cap as fraction of total "
            "archive bytes (default: 0.20 = 20 pct summed across axes)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human summary.",
    )
    parser.add_argument(
        "--persist-to-ledger",
        action="store_true",
        help=(
            "Append the solution row to "
            ".omx/state/pareto_polytope_solutions.jsonl "
            "(default: do not persist; CLI is observability-only)."
        ),
    )
    parser.add_argument(
        "--meta-lift-1-ledger-path",
        type=str,
        default=None,
        help=(
            "Override path to "
            ".omx/state/cross_substrate_master_gradient_analyses.jsonl "
            "(default: canonical repo location)."
        ),
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse calls sys.exit(2) for invalid args; preserve canonical rc.
        return int(exc.code) if exc.code is not None else 2

    if args.max_iterations < 1:
        sys.stderr.write("ERROR: --max-iterations must be >= 1\n")
        return 2
    if args.tol <= 0:
        sys.stderr.write("ERROR: --tol must be positive\n")
        return 2
    if args.per_axis_cap_fraction <= 0 or args.per_axis_cap_fraction > 1:
        sys.stderr.write("ERROR: --per-axis-cap-fraction must be in (0, 1]\n")
        return 2
    if args.aggregate_cap_fraction <= 0 or args.aggregate_cap_fraction > 1:
        sys.stderr.write("ERROR: --aggregate-cap-fraction must be in (0, 1]\n")
        return 2

    ledger_path = (
        Path(args.meta_lift_1_ledger_path) if args.meta_lift_1_ledger_path else None
    )

    analysis = _load_latest_meta_lift_1_analysis(ledger_path)
    if analysis is None:
        sys.stderr.write(
            "ERROR: no META-LIFT-1 cross-substrate analysis found in canonical "
            "ledger at .omx/state/cross_substrate_master_gradient_analyses.jsonl "
            "(use tools/cross_substrate_master_gradient_cli.py --persist-to-ledger "
            "to populate the ledger first).\n"
        )
        return 2

    try:
        problem_spec = build_problem_spec_from_meta_lift_1_analysis(
            analysis,
            per_substrate_per_axis_byte_budget_cap_fraction=args.per_axis_cap_fraction,
            per_substrate_aggregate_byte_budget_cap_fraction=args.aggregate_cap_fraction,
            target_aggregate_delta_s=args.target_aggregate_delta_s,
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: problem spec construction raised: {exc}\n")
        return 2

    try:
        solution = solve_pareto_polytope_via_dykstra_projections(
            problem_spec,
            max_iterations=args.max_iterations,
            tol=args.tol,
            upstream_meta_lift_1_analysis_id=analysis.get("analysis_id"),
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: solver raised: {exc}\n")
        return 2

    if not solution.allocation.feasible:
        sys.stderr.write(
            f"WARNING: solution is INFEASIBLE_WITH_BOUND "
            f"(residual={solution.allocation.feasibility_residual:.6e}); "
            f"problem may have empty constraint intersection. "
            f"Try increasing --per-axis-cap-fraction or --aggregate-cap-fraction.\n"
        )
        infeasible_exit = 1
    else:
        infeasible_exit = 0

    if args.persist_to_ledger:
        append_solution_locked(solution)

    solution_dict = solution.as_dict()
    if args.json:
        sys.stdout.write(json.dumps(solution_dict, sort_keys=True, indent=2, allow_nan=False))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_human_summary(solution_dict))
        sys.stdout.write("\n")

    return infeasible_exit


if __name__ == "__main__":
    raise SystemExit(main())
