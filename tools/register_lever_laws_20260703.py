"""Register the 2026-07-03 lever laws into the canonical-equations JSONL registry
(``.omx/state/canonical_equations_registry.jsonl``) so the EQUATIONS leg of the triality
AGREES with the code — closes the sweep-B apparatus finding (2026-07-04): the 07-03 lever
law modules are code-live (imported in ``canonical_equations/__init__.py``, consumed by
``witness_dsl/gauge.py``) but were never flushed to the registry (JSONL=0), leaving
``list_canonical_equations.py`` + every cathedral/DSL JSONL consumer BLIND to the exact
levers the fresh seeded run turns on (along-tangent, margin-saliency, Muon-finisher, chroma,
separatrix-asymmetry, MTF, flicker-floor, scalar-margin, Lever-D economics).

Idempotent (append-only 'registered' event keyed by equation_id — safe to re-run). MEANS;
pointer 0.19110 UNMOVED; every lever's verdict remains net-score #205-gated through the
byte-close, NOT the registration.

    .venv/bin/python tools/register_lever_laws_20260703.py --dry-run   # build + validate only
    .venv/bin/python tools/register_lever_laws_20260703.py             # flush to the registry
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

from tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703 import (  # noqa: E402
    build_margin_saliency_reachability_replaces_texture_proxy_v1,
    build_muon_finisher_schedule_warmstart_and_lr_anneal_v1,
)
from tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703 import (  # noqa: E402
    build_anisotropic_basis_along_tangent_frequency_deficit_v1,
    build_chroma_decides_lane_and_movable_at_annulus_v1,
    build_contest_R_operator_mtf_allpass_to_2px_v1,
    build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1,
    build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1,
    build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1,
)
from tac.canonical_equations.leverd_flicker_residual_reactivation_economics_20260703 import (  # noqa: E402
    build_leverd_flicker_residual_reactivation_economics_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402

# The 9 code-live 07-03 lever-law builders (2 + 6 + 1 across the 3 modules).
ALL_LEVER_LAW_BUILDERS_20260703 = (
    build_margin_saliency_reachability_replaces_texture_proxy_v1,
    build_muon_finisher_schedule_warmstart_and_lr_anneal_v1,
    build_contest_R_operator_mtf_allpass_to_2px_v1,
    build_anisotropic_basis_along_tangent_frequency_deficit_v1,
    build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1,
    build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1,
    build_chroma_decides_lane_and_movable_at_annulus_v1,
    build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1,
    build_leverd_flicker_residual_reactivation_economics_v1,
)

_SUBAGENT_ID = "sweep-B-apparatus-flush-07-03-lever-laws"
_NOTES = (
    "sweep-#290-B apparatus flush (2026-07-04): the 9 code-live 07-03 lever laws "
    "(margin-saliency-reachability / Muon-finisher / R-MTF-allpass / along-tangent-deficit / "
    "separatrix-asymmetry-t / flicker-jitter-floor / chroma-decides-lane / scalar-margin / "
    "Lever-D economics) were never flushed to the registry — the equations leg was blind to the "
    "fresh-run levers. Registered so the triality AGREES. MEANS; pointer 0.19110 UNMOVED; "
    "each lever's verdict stays net-score #205-gated through the byte-close."
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="build + validate the equations but do NOT write the registry")
    args = ap.parse_args(argv)

    registered: list[str] = []
    for builder in ALL_LEVER_LAW_BUILDERS_20260703:
        eq = builder()  # __post_init__ validates the full contract (producers/consumers/anchors)
        summary = {
            "equation_id": eq.equation_id,
            "n_anchors": len(eq.empirical_anchors),
            "consumers": list(eq.canonical_consumers),
            "producers": list(eq.canonical_producers),
        }
        print(json.dumps(summary, indent=2))
        if not args.dry_run:
            register_canonical_equation(eq, subagent_id=_SUBAGENT_ID, notes=_NOTES)
            registered.append(eq.equation_id)

    print(json.dumps({
        "stage": "dry_run" if args.dry_run else "registered",
        "n_laws": len(ALL_LEVER_LAW_BUILDERS_20260703),
        "registered": registered,
        "pointer": "0.19110 UNMOVED (MEANS; net-score #205-gated)",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
