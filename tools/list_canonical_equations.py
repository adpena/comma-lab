#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI: list canonical equations + per-axis residuals.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable.
Reads ``.omx/state/canonical_equations_registry.jsonl`` via the canonical
``tac.canonical_equations.query_equations`` helper and emits a
human-readable summary OR JSON view.

Usage:

    .venv/bin/python tools/list_canonical_equations.py
    .venv/bin/python tools/list_canonical_equations.py --json
    .venv/bin/python tools/list_canonical_equations.py --equation-id mps_drift_architecture_class_dependent_v1
    .venv/bin/python tools/list_canonical_equations.py --consumer tac.master_gradient_iterative_refinement
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (  # noqa: E402
    get_equation_by_id,
    query_equations,
    query_equations_by_consumer,
)


def _format_one(eq, *, verbose: bool) -> list[str]:
    lines = [
        f"equation_id: {eq.equation_id}",
        f"  name:           {eq.name}",
        f"  summary:        {eq.one_line_summary}",
        f"  callable:       {eq.python_callable_module_path}",
        f"  anchors:        {len(eq.empirical_anchors)}",
        f"  well-calibrated: {eq.is_well_calibrated}",
        f"  last calibrated: {eq.last_calibration_utc}",
        f"  trigger:        {eq.next_recalibration_trigger}",
        f"  consumers ({len(eq.canonical_consumers)}): {', '.join(eq.canonical_consumers) or '(none)'}",
        f"  producers ({len(eq.canonical_producers)}): {', '.join(eq.canonical_producers) or '(none)'}",
        "  residuals:",
    ]
    if not eq.predicted_vs_empirical_residual:
        lines.append("    (no empirical anchors yet)")
    else:
        for axis, residual in sorted(eq.predicted_vs_empirical_residual.items()):
            tag = "OK" if residual < 2.0 else "DRIFT"
            lines.append(f"    [{tag}] {axis}: residual={residual:.4f}")
    if verbose:
        lines.append("  latex:")
        lines.append(f"    {eq.latex_form}")
        lines.append("  empirical anchors:")
        for a in eq.empirical_anchors:
            lines.append(
                f"    - {a.anchor_id} ({a.measurement_method}, residual={a.residual:.4f})"
            )
            lines.append(f"        source: {a.source_artifact}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List canonical equations from the registry"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--equation-id", default=None, help="filter to one equation_id")
    parser.add_argument(
        "--consumer",
        default=None,
        help="filter to equations whose canonical_consumers contains this token",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include latex_form and per-anchor details",
    )
    args = parser.parse_args()

    if args.equation_id:
        eq = get_equation_by_id(args.equation_id)
        equations = [eq] if eq is not None else []
    elif args.consumer:
        equations = query_equations_by_consumer(args.consumer)
    else:
        equations = query_equations()

    if args.json:
        print(json.dumps([eq.to_dict() for eq in equations], indent=2, sort_keys=True))
        return 0

    if not equations:
        print("(no canonical equations registered yet OR no match for filter)")
        return 0

    print(f"# Canonical equations registry ({len(equations)} entries)")
    print()
    for eq in equations:
        for line in _format_one(eq, verbose=args.verbose):
            print(line)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
