"""Register the 2026-07-05 boundary-distance weight calibration law into the canonical-equations
JSONL registry (``.omx/state/canonical_equations_registry.jsonl``) — the EQUATIONS leg of the
CE-window intervention pre-stage (``.omx/research/ce_window_intervention_package_20260705.md``),
anchored by the MEASURED ep100 phi-surface calibration.

Idempotent (append-only 'registered' event keyed by equation_id — safe to re-run). MEANS;
pointer 0.19110 UNMOVED.

    .venv/bin/python tools/register_bd_calibration_20260705.py --dry-run   # build + validate only
    .venv/bin/python tools/register_bd_calibration_20260705.py             # flush to the registry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.boundary_distance_calibration_20260705 import (  # noqa: E402
    build_boundary_distance_weight_calibration_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402

_BUILDERS = (build_boundary_distance_weight_calibration_v1,)
_SUBAGENT_ID = "CE-WINDOW-PRESTAGE"
_NOTES = (
    "CE-window intervention pre-stage 2026-07-05 equations leg; anchor = the MEASURED ep100 bd "
    "calibration (12-pair, phi surface): ratio(w)=w*M_bd/(M_ce+w*M_bd) with M_ce=35.61 M_bd=17.72 "
    "=> 5-15% window = w in [0.106,0.355], w*=0.2 @9.05%; shares monotone; no island collapse; "
    "live/EMA CE 0.928 (live weights not seg-damaged)."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    results = []
    for build in _BUILDERS:
        eq = build()
        if args.dry_run:
            results.append({"equation_id": eq.equation_id, "dry_run": True})
            continue
        register_canonical_equation(eq, subagent_id=_SUBAGENT_ID, notes=_NOTES)
        results.append({"equation_id": eq.equation_id, "registered": True})
    print(json.dumps({"stage": "register_bd_calibration_20260705", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
