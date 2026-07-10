"""Register the 2026-07-10 SegNet texture-perception canonical equations (the EQUATIONS leg of the
stem-perception recovery: perceiver read + through-R per-class texture price list).

Idempotent (append-only 'registered' event keyed by equation_id — safe to re-run). MEANS;
pointer 0.19110 UNMOVED.

    .venv/bin/python tools/register_segnet_texture_perception_equations_20260710.py --dry-run
    .venv/bin/python tools/register_segnet_texture_perception_equations_20260710.py
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

from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402
from tac.canonical_equations.segnet_texture_perception_20260710 import (  # noqa: E402
    build_segnet_stem_nyquist_alias_wall_v1,
    build_segnet_through_r_texture_price_list_v1,
)

_BUILDERS = (
    build_segnet_stem_nyquist_alias_wall_v1,
    build_segnet_through_r_texture_price_list_v1,
)
_SUBAGENT_ID = "stem-perception-recovery-20260710"
_NOTES = (
    "SegNet texture-perception recovery 2026-07-10 equations leg. MEASURED through-R (frozen "
    "CPU-SegNet, 568-tile context-free sweep): B2 stem is colour+low-pass dominated with a period-4 "
    "(~9.1 cam-px) Nyquist alias wall; Road+Lane are TEXTURE-defined (negative flat floor, 0/64 "
    "flats win) and reachable ONLY by a period-4 high-contrast luminance grating. Advisory, "
    "NON-PROMOTABLE; artifact experiments/results/stem_perception_20260710/."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    results = []
    for build in _BUILDERS:
        eq = build()
        if args.dry_run:
            results.append({"equation_id": eq.equation_id, "dry_run": True,
                            "n_anchors": len(eq.empirical_anchors)})
            continue
        register_canonical_equation(eq, subagent_id=_SUBAGENT_ID, notes=_NOTES)
        results.append({"equation_id": eq.equation_id, "registered": True})
    print(json.dumps({"stage": "register_segnet_texture_perception_20260710",
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
