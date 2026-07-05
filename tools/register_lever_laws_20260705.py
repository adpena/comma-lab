"""Register the 2026-07-05 focal-gamma concentration law into the canonical-equations JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) — the EQUATIONS leg of the
levelset-loss-geometry symposium (``council_grand_symposium_levelset_loss_geometry_20260705.md``
§triality updates), anchored by the MEASURED ep50 calibration
(``.omx/research/focal_boundary_calibration_20260705.md``).

The ``levelset_energy_loss_equivalence_v1`` DESIGN-STAGE equation named by the symposium is NOT
registered here (run-3/theta* design item, no measured anchor yet — FORMALIZATION deferred with
the active-contour design doc per symposium resolution 3).

Idempotent (append-only 'registered' event keyed by equation_id — safe to re-run). MEANS;
pointer 0.19110 UNMOVED.

    .venv/bin/python tools/register_lever_laws_20260705.py --dry-run   # build + validate only
    .venv/bin/python tools/register_lever_laws_20260705.py             # flush to the registry
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

from tac.canonical_equations.focal_gradient_concentration_20260705 import (  # noqa: E402
    build_focal_gradient_concentration_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402

_BUILDERS = (build_focal_gradient_concentration_v1,)
_SUBAGENT_ID = "focal-boundary-calibrate-build-20260705"
_NOTES = (
    "council levelset-loss-geometry symposium 2026-07-05 equations leg; anchor = the MEASURED "
    "ep50 focal calibration (12-pair, witness-alone surface): island share NON-monotone in gamma "
    "(peak ~1), focal = weak bulk-boundary reallocator, gamma*=0 (HOLD) on the live trajectory."
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
    print(json.dumps({"stage": "register_lever_laws_20260705", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
