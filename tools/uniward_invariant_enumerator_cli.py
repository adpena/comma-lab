#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""META-LIFT-4 UNIWARD canonical-application-surface invariant enumerator CLI.

Per the 11th standing directive ORDER discipline: this is the operator-facing
sister of :mod:`tac.uniward_invariant_enumerator`. Iterates ALL known
canonical-application surfaces in our codebase (DCT analog / chroma LUT /
scorer class softmax / master-gradient per-byte / FEC selector indices /
VQ-VAE indices_blob / Wyner-Ziv codec layer / etc.), emits per-surface
UNIWARD applicability verdicts + per-axis rankings (human-readable summary
or JSON).

Usage:
    .venv/bin/python tools/uniward_invariant_enumerator_cli.py --enumerate-all
    .venv/bin/python tools/uniward_invariant_enumerator_cli.py --verify-surface nscs06_v8_chroma_lut
    .venv/bin/python tools/uniward_invariant_enumerator_cli.py --rank-by-predicted-delta-s --contest-axis seg
    .venv/bin/python tools/uniward_invariant_enumerator_cli.py --enumerate-all --json
    .venv/bin/python tools/uniward_invariant_enumerator_cli.py --enumerate-all --persist-to-ledger

Exit codes:
    0  CLEAN — enumeration emitted (stdout)
    1  NO_APPLICABLE_SURFACES — every surface failed canonical-application test
    2  CLI error (invalid arguments / unrecognized flag value)

Per Catalog #341 routing markers: this CLI's output is OBSERVABILITY-ONLY
by construction. Promotion of any surface ranking to a contest score signal
REQUIRES paired-CUDA empirical anchor on that specific surface per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable.

Sister of:
  - :mod:`tools.cross_substrate_master_gradient_cli` (META-LIFT-1 CLI)
  - :mod:`tools.pareto_polytope_solver_cli` (META-LIFT-2 CLI)
  - :mod:`tools.list_canonical_equations` (canonical equations registry)
  - :mod:`tools.cathedral_autopilot_autonomous_loop` (downstream ranker
    consumer via the sister cathedral consumer auto-discovered per
    Catalog #335 / #336 / #337)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.uniward_invariant_enumerator import (  # noqa: E402
    UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH,
    VALID_AXIS_LABELS,
    append_enumeration_locked,
    enumerate_uniward_canonical_application_surfaces,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="META-LIFT-4 UNIWARD canonical-application-surface invariant enumerator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--enumerate-all",
        action="store_true",
        help="Enumerate ALL canonical-application surfaces + emit per-axis rankings.",
    )
    mode.add_argument(
        "--verify-surface",
        type=str,
        default=None,
        help="Verify per-surface UNIWARD applicability (surface_id; "
        "e.g. 'nscs06_v8_chroma_lut').",
    )
    mode.add_argument(
        "--rank-by-predicted-delta-s",
        action="store_true",
        help="Rank applicable surfaces by predicted ΔS per Cauchy-Schwarz bound.",
    )
    parser.add_argument(
        "--contest-axis",
        type=str,
        choices=("seg", "pose", "rate"),
        default=None,
        help="Per Catalog #356 per-axis decomposition: filter ranking to one axis "
        "(seg / pose / rate). Required when --rank-by-predicted-delta-s.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Show top-N surfaces in ranking (default: all).",
    )
    parser.add_argument(
        "--persist-to-ledger",
        action="store_true",
        help="Persist the enumeration to the canonical fcntl-locked JSONL "
        f"ledger at {UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable summary.",
    )
    return parser


def _emit_summary_text(enumeration) -> None:
    """Human-readable summary text emit."""
    print("=" * 72)
    print(f"META-LIFT-4 UNIWARD canonical-application-surface invariant enumeration")
    print(f"  enumeration_id: {enumeration.enumeration_id}")
    print(f"  measurement_utc: {enumeration.measurement_utc}")
    print(f"  canonical_equation_id: {enumeration.canonical_equation_id}")
    print(f"  canonical_equation_status: {enumeration.canonical_equation_status}")
    print(f"  evidence_grade: {enumeration.evidence_grade}")
    print(f"  axis_tag: {enumeration.axis_tag}  score_claim: {enumeration.score_claim}  promotable: {enumeration.promotable}")
    print("=" * 72)
    print()
    print(f"Surface counts:")
    print(f"  APPLICABLE:   {enumeration.n_applicable_surfaces}")
    print(f"  INAPPLICABLE: {enumeration.n_inapplicable_surfaces}")
    print(f"  UNKNOWN:      {enumeration.n_unknown_surfaces}")
    print()
    print("Per-surface UNIWARD applicability verdicts:")
    print("-" * 72)
    for surface, verdict in zip(enumeration.surfaces, enumeration.verdicts):
        status_marker = "[OK]" if verdict.all_conditions_pass() else "[FAIL]"
        print(f"  {status_marker:>7}  {surface.surface_id}")
        print(f"           kind={surface.surface_kind}  layer={surface.architecture_layer}")
        print(f"           verdict={verdict.verdict}")
        print(f"           cond_1_entropy={verdict.condition_1_entropy_coded}  "
              f"cond_2_quant={verdict.condition_2_quantized}  "
              f"cond_3_routable={verdict.condition_3_per_symbol_routable}  "
              f"cond_4_canonical={verdict.condition_4_canonical_formula_grounded}")
    print()
    print("Per-axis ranked surfaces (DESC by predicted ΔS upper bound):")
    print("-" * 72)
    for ranking in enumeration.rankings_per_axis:
        print(f"  axis={ranking.axis}:")
        for i, sid in enumerate(ranking.ranked_surface_ids[:10], start=1):
            upper = ranking.per_surface_predicted_delta_s_upper_bound[i - 1]
            lever = ranking.per_surface_per_byte_leverage[i - 1]
            print(f"    #{i:2d}: {sid:<50s}  upper={upper:.4f}  leverage={lever:.6e}")
        print()
    print("Notes:")
    print(f"  - All outputs OBSERVABILITY-ONLY per Catalog #341.")
    print(f"  - Promotion requires paired-CUDA empirical anchor per CLAUDE.md")
    print(f"    'Submission auth eval — BOTH CPU AND CUDA'.")
    print(f"  - Canonical equation #344 status: {enumeration.canonical_equation_status}")
    print()


def _emit_verify_surface_text(enumeration, surface_id: str) -> bool:
    """Verify per-surface UNIWARD applicability + emit verdict text.

    Returns True if surface_id was found in the canonical registry.
    """
    for surface, verdict in zip(enumeration.surfaces, enumeration.verdicts):
        if surface.surface_id == surface_id:
            print(f"Surface: {surface.surface_id}")
            print(f"  kind:              {surface.surface_kind}")
            print(f"  substrate_id:      {surface.substrate_id}")
            print(f"  entropy_coded:     {surface.entropy_coded_axis}")
            print(f"  quantization:      {surface.quantization_axis}")
            print(f"  per_symbol_route:  {surface.per_symbol_routable_axis}")
            print(f"  canonical_ref:     {surface.canonical_formula_reference}")
            print(f"  n_symbols:         {surface.n_symbols_estimated}")
            print(f"  arch_layer:        {surface.architecture_layer}")
            print(f"  notes:             {surface.notes}")
            print()
            print(f"VERDICT: {verdict.verdict}")
            print(f"  condition_1_entropy_coded:           {verdict.condition_1_entropy_coded}")
            print(f"  condition_2_quantized:               {verdict.condition_2_quantized}")
            print(f"  condition_3_per_symbol_routable:     {verdict.condition_3_per_symbol_routable}")
            print(f"  condition_4_canonical_grounded:      {verdict.condition_4_canonical_formula_grounded}")
            print(f"  all_pass:                            {verdict.all_conditions_pass()}")
            print(f"  rationale: {verdict.rationale}")
            print()
            return True
    return False


def _emit_ranking_text(enumeration, axis: str, top_n: int | None) -> None:
    """Per-axis ranking text emit."""
    for ranking in enumeration.rankings_per_axis:
        if ranking.axis != axis:
            continue
        print(f"Per-axis ranking (axis={axis}; DESC by predicted ΔS upper bound):")
        print(f"  canonical_equation_reference: {ranking.canonical_equation_reference}")
        print("-" * 72)
        n = top_n if top_n else len(ranking.ranked_surface_ids)
        for i, sid in enumerate(ranking.ranked_surface_ids[:n], start=1):
            upper = ranking.per_surface_predicted_delta_s_upper_bound[i - 1]
            lever = ranking.per_surface_per_byte_leverage[i - 1]
            print(f"  #{i:2d}: {sid:<50s}  upper={upper:.4f}  leverage={lever:.6e}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # If no mode flag, default to --enumerate-all.
    if not args.enumerate_all and args.verify_surface is None and not args.rank_by_predicted_delta_s:
        args.enumerate_all = True

    if args.rank_by_predicted_delta_s and args.contest_axis is None:
        print(
            "ERROR: --rank-by-predicted-delta-s requires --contest-axis (one of seg/pose/rate)",
            file=sys.stderr,
        )
        return 2

    enumeration = enumerate_uniward_canonical_application_surfaces()

    if args.persist_to_ledger:
        append_enumeration_locked(enumeration)

    if args.json:
        print(json.dumps(enumeration.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.verify_surface is not None:
        found = _emit_verify_surface_text(enumeration, args.verify_surface)
        if not found:
            print(
                f"ERROR: surface_id={args.verify_surface!r} not found in canonical registry. "
                f"Run --enumerate-all to list all known surfaces.",
                file=sys.stderr,
            )
            return 2
        return 0

    if args.rank_by_predicted_delta_s:
        _emit_ranking_text(enumeration, args.contest_axis, args.top_n)
        return 0

    # Default: --enumerate-all summary.
    _emit_summary_text(enumeration)

    if enumeration.n_applicable_surfaces == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
