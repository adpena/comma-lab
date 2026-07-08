"""Register the TAIL PowerPlay attribution-floor law s* = ν·forfeit
(``forfeit_matched_exit_v1``) into the canonical-equations JSONL registry
(``.omx/state/canonical_equations_registry.jsonl``) — the EQUATIONS leg of
SEAL-v7-r1 MAJOR-1's reconcile (the seal found the law was NOT in the registry,
so nothing bound the TAIL stop to the DERIVED 6.897e-6 floor; it hardcoded 1e-4).

Idempotent by inspection of prior events (skips equation_ids already present).
MEANS; pointer contest-CPU 0.19110 UNMOVED.

    .venv/bin/python tools/register_tail_stop_forfeit_floor_20260708.py --dry-run
    .venv/bin/python tools/register_tail_stop_forfeit_floor_20260708.py
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

from tac.canonical_equations.registry import (  # noqa: E402
    load_registry_events_lenient,
    register_canonical_equation,
)
from tac.canonical_equations.tail_stop_forfeit_floor_20260708 import (  # noqa: E402
    ALL_BUILDERS,
)

_SUBAGENT_ID = "tail-majors-fix-seal-v7-r1-major1-20260708"
_NOTES = (
    "SEAL-v7-r1 MAJOR-1 reconcile: s* = ν(tau_softplus)·forfeit = 6.897e-6 S/ep DERIVED "
    "PowerPlay attribution floor (DRAFT v6 §2.2f/g); replaces the struck HARDCODED 1e-4 "
    "(14.5× coarser). [macOS-MLX research-signal] NON-PROMOTABLE; pointer 0.19110 UNMOVED"
)


def _registered_ids() -> set[str]:
    return {ev.get("equation_id", "") for ev in load_registry_events_lenient()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    eq_ids = _registered_ids()
    results: list[dict[str, object]] = []
    for build in ALL_BUILDERS:
        eq = build()  # builds + validates (raises on schema violation)
        if eq.equation_id in eq_ids:
            results.append({"equation_id": eq.equation_id, "skipped": "already_registered"})
            continue
        if args.dry_run:
            results.append({"equation_id": eq.equation_id, "dry_run": True})
            continue
        register_canonical_equation(eq, subagent_id=_SUBAGENT_ID, notes=_NOTES)
        results.append({"equation_id": eq.equation_id, "registered": True})

    print(json.dumps({"stage": "register_tail_stop_forfeit_floor_20260708",
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
