"""Register the #395 texture-trunk band-design canonical equation (equations leg of the P0 build).

Idempotent (append-only 'registered' event keyed by equation_id — safe to re-run). MEANS;
pointer 0.19110 UNMOVED.

    .venv/bin/python tools/register_texture_trunk_band_equation_20260710.py --dry-run
    .venv/bin/python tools/register_texture_trunk_band_equation_20260710.py
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
from tac.canonical_equations.texture_trunk_band_20260710 import (  # noqa: E402
    build_texture_trunk_band_is_stem_passband_v1,
)

_BUILDERS = (build_texture_trunk_band_is_stem_passband_v1,)
_SUBAGENT_ID = "textrunk-p0-20260710"
_NOTES = (
    "#395 P0 texture-trunk band-design law. DERIVED from segnet_stem_nyquist_alias_wall_v1 + the "
    "price-list breadth; VERIFIED_VIA_SOURCE_INSPECTION (TextureBandSpec refuses out-of-band periods; "
    "band_limit_report spectral proof). The band-designed per-class stationary texture trunk's bank "
    "support = the frozen stem transfer pass-band [period-4 Nyquist .. band_hi=8] render-px (clause-B "
    "minimal-dim); bank is rule-118 free, only W_tex+bias counted. Advisory, NON-PROMOTABLE."
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
    print(json.dumps({"stage": "register_texture_trunk_band_20260710", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
