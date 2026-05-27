#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI: trigger auto-recalibration for canonical equations.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable.
Iterates current equations + emits a no-op refresh report when no new
continual-learning anchors are present. Per Catalog #287/#323, actual
recalibration requires explicit signed measurement provenance via
``tac.canonical_equations.update_equation_with_empirical_anchor`` — this
CLI is the operator-facing audit + trigger surface.

Usage:

    .venv/bin/python tools/recalibrate_equation.py
    .venv/bin/python tools/recalibrate_equation.py --equation-id mps_drift_architecture_class_dependent_v1
    .venv/bin/python tools/recalibrate_equation.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (  # noqa: E402
    auto_recalibrate_from_continual_learning_posterior,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trigger calibration refresh from continual-learning posterior"
    )
    parser.add_argument("--equation-id", default=None, help="filter to one equation_id")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    report = auto_recalibrate_from_continual_learning_posterior(args.equation_id)

    if args.json:
        out = {
            "equations_checked": report.equations_checked,
            "equations_recalibrated": report.equations_recalibrated,
            "new_anchors_absorbed": report.new_anchors_absorbed,
            "per_equation_summary": report.per_equation_summary,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(
        f"# Recalibration report (checked={report.equations_checked}, "
        f"recalibrated={report.equations_recalibrated}, "
        f"new_anchors_absorbed={report.new_anchors_absorbed})"
    )
    print()
    for eq_id, summary in sorted(report.per_equation_summary.items()):
        ok = summary.get("well_calibrated", False)
        marker = "OK" if ok else "DRIFT"
        recal = " [RECALIBRATED]" if summary.get("recalibrated") else ""
        print(f"  [{marker}]{recal} {eq_id}")
        for axis, residual in sorted(summary.get("current_residuals", {}).items()):
            try:
                print(f"      {axis}: residual={float(residual):.4f}")
            except (TypeError, ValueError):
                print(f"      {axis}: residual={residual}")
    print()
    print(
        "Note: auto-refit re-derives the residual SUMMARY from anchors already "
        "landed via signed update_equation_with_empirical_anchor() calls when an "
        "equation's when_3+_new_empirical_anchors_in_domain trigger is satisfied. "
        "It never synthesizes anchors (Catalog #287/#323); a NEW measurement still "
        "requires an explicit update_equation_with_empirical_anchor() call backed "
        "by a measured artifact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
