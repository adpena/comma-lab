"""Register the 2026-07-05 curriculum-derivation laws into the canonical-equations JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) — the EQUATIONS leg of the
T3 curriculum-derivation symposium (task #302;
``.omx/research/council_grand_symposium_curriculum_derivation_20260705.md`` §D).

Four laws: curriculum_handoff_critical_nucleus_v1 · ema_window_pi_group_v1 ·
muon_switch_conditioning_criterion_v1 · rewarmup_beta2_memory_window_v1 (the last is
PROVISIONAL-PENDING-VERIFICATION per Catalog #363 Round-3, recorded as such in its domain).

Idempotent (append-only 'registered' event keyed by equation_id — safe to re-run). MEANS;
pointer 0.19110 UNMOVED.

    .venv/bin/python tools/register_lever_laws_curriculum_20260705.py --dry-run
    .venv/bin/python tools/register_lever_laws_curriculum_20260705.py
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

from tac.canonical_equations.curriculum_derivation_laws_20260705 import (  # noqa: E402
    build_curriculum_handoff_critical_nucleus_v1,
    build_ema_window_pi_group_v1,
    build_muon_switch_conditioning_criterion_v1,
    build_rewarmup_beta2_memory_window_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402

_BUILDERS = (
    build_curriculum_handoff_critical_nucleus_v1,
    build_ema_window_pi_group_v1,
    build_muon_switch_conditioning_criterion_v1,
    build_rewarmup_beta2_memory_window_v1,
)
_SUBAGENT_ID = "CURRICULUM-DERIVATION-T3"
_NOTES = (
    "T3 curriculum-derivation symposium 2026-07-05 (task #302) equations leg; anchors = $0 "
    "log mining of #205/FIRE-arm/cert traces (no new runs): C1 plateau-eps miscalibration, "
    "#205 tau-onset erosion, Muon -32% fork + cold-switch transient, verified per-step EMA "
    "cadence + measured 78x lag, rewarmup-vs-beta2-memory arithmetic."
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
    print(json.dumps({"stage": "register_lever_laws_curriculum_20260705",
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
