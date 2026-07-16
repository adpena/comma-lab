# SPDX-License-Identifier: MIT
"""Per-class × per-stratum residual carrier taxonomy + n600 separatrix asymmetry.

Decomposes the necessity-calibration min-S residual (d_seg 0.01328 at ε=0-lossless +
static hood-tex; ``necessity_generator_seed_dseg_calibration_v1``) per (class × stratum)
on n600, measures its static/dynamic character, the one-sided separatrix margin-slope
asymmetry per class pair (upgrading ``separatrix_asymmetry_t_subpixel_boundary_localizer_v1``
from n6 to n600), the input-space cure drivers (VJP through the real frozen SegNet), and
the carrier-form smokes. Four MEASURED laws (2026-07-16, ``[macOS-CPU advisory]``):

  1. RESIDUAL CONCENTRATION: Road|edge 35.3% + Movable|far 24.3% + Road|near 11.9% +
     Movable|near 8.3% + Lane|edge 8.3% + Movable|edge 4.6% = 92.6% of the residual.
     No static bucket remains (occupancy <= 0.26); Movable interiors are object-persistent
     (persist_next 0.865 -> xi-transportable), boundaries semi-static (0.27-0.46).
  2. SEPARATRIX ASYMMETRY IS PER-PAIR: one-sided margin slope ratio 2.78x Road-Undrivable
     (Road shallow), 2.33x Lane-MyCar (Lane), 1.36x Road-Lane (Lane), ~1.0 both Movable
     pairs (symmetric AND lowest absolute margins m1 ~0.83-0.98: both-sides-fragile).
  3. CURE DRIVERS: luma 92-94% of cure-gradient energy in every bucket; non-local
     (<=15% within r4, <=67% within r36); flat-shift coherence <= 0.24 (texture, not
     palette) — Lane is the single flat-coherent exception (brighten lane side, +10/1).
  4. ASYMMETRIC ONE-SIDED DOMINANCE + REGION-FROM-BOUNDARY: a 2px one-sided Movable-side
     border-contrast band at beta=2 cures 29.5% of total d_seg at ZERO counted bytes
     (83% of Movable|far cured from the band alone — SegNet reads region identity from
     border contrast); symmetric +73%, deep-side +98% (categorical failures); blurry
     low-frequency realism (ds16 GT oracle) LOSES to the crisp cartoon (+78%).

Verdict scope: laws 2-4 are frozen-scorer-facing (vehicle-agnostic); law 1's weights are
the palette vehicle's. Smokes are stride-5 rankings (winner owed n600 + byte-closed A/B).
NOT a score claim; pointer 0.19108 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "perclass_stratum_residual_carrier_taxonomy_v1"
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached n600 GT"
TOOL = "tools/c2_perclass_stratum_carrier_analysis.py"
ART_DIR = "experiments/results/c2_perclass_stratum_20260716"
MEMO = ".omx/research/c2_perclass_stratum_carrier_taxonomy_20260716.md"

# ---- MEASURED constants (artifacts: temporal.json / slope.json / sens.json / smoke_*.json)
RESIDUAL_DSEG = 0.01328              # the decomposed parent residual (n600)
BUCKET_FRACS = {                      # % of residual, n600 (top buckets; sum of ALL = 1.0)
    "Road|edge": 0.3528, "Movable|far": 0.2426, "Road|near": 0.1194,
    "Movable|near": 0.0829, "Lane|edge": 0.0827, "Movable|edge": 0.0462,
}
MOVABLE_FAR_PERSIST_NEXT = 0.865      # object-persistent -> xi-transportable
MAX_BUCKET_OCCUPANCY = 0.26           # no static bucket remains after the hood seed
SLOPE_ASYM_RATIO = {                  # one-sided margin slope max/min per pair (stride-5)
    "Road-Undrivable": 2.78, "Lane-MyCar": 2.33, "Road-Lane": 1.36,
    "Road-MyCar": 1.20, "Undrivable-Movable": 1.08, "Road-Movable": 1.04,
}
SHALLOW_SIDE = {
    "Road-Undrivable": "Road", "Lane-MyCar": "Lane", "Road-Lane": "Lane",
    "Road-MyCar": "MyCar", "Undrivable-Movable": None, "Road-Movable": None,
}
LUMA_CURE_ENERGY_FRAC_MED = 0.93      # every bucket 0.92-0.94
FLAT_COHERENCE_MAX = 0.24             # Lane gt-side; all other buckets <= 0.06
SMOKE_BASELINE_SUBSET = 0.013044      # stride-5 subset baseline
SMOKE_ONESIDED_MOVABLE_B2 = 0.009196  # winner: -29.5% at 0 counted bytes
SMOKE_SYMMETRIC_B05 = 0.022572        # +73% (categorical failure)
SMOKE_DEEPSIDE_B05 = 0.025854         # +98% (categorical failure)
SMOKE_TEX_GLOBAL_DS16 = 0.023194      # blurry realism loses to the crisp cartoon (+78%)
MOVABLE_FAR_CURED_FRAC_BY_BAND = 0.83  # region-from-boundary law


def one_sided_carrier_gain(dseg_variant: float,
                           dseg_baseline: float = SMOKE_BASELINE_SUBSET) -> float:
    """Fractional d_seg reduction of a carrier variant (negative = a win)."""
    if dseg_baseline <= 0.0:
        raise ValueError(f"dseg_baseline must be > 0, got {dseg_baseline!r}")
    return (dseg_variant - dseg_baseline) / dseg_baseline


def carrier_side_for_pair(pair: str) -> str | None:
    """Measured shallow (cheap-crossing) side per class pair; None = symmetric pair."""
    if pair not in SHALLOW_SIDE:
        raise KeyError(f"unknown pair {pair!r}; known: {sorted(SHALLOW_SIDE)}")
    return SHALLOW_SIDE[pair]


def build_perclass_stratum_residual_carrier_taxonomy_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256="0a4b9a41a6c9d0bbfed4015f58d49889691aba573c13ee504a9ed07edc866776",
        source_path=f"{ART_DIR}/temporal.json",
        captured_at_utc="2026-07-16T06:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="c2_perclass_stratum_decomposition_n600_20260716",
            measurement_utc="2026-07-16T06:00:00Z",
            inputs={"frames": "n600 (ALL scored pairs)", "vehicle": "eps0+hood palette render",
                    "strata": "saddle > edge(1px) > near(<=3px) > far, class = pixel GT class"},
            predicted_output={"buckets_carry_residual": True},
            empirical_output={"bucket_fracs_top6": BUCKET_FRACS,
                              "movable_far_persist_next": MOVABLE_FAR_PERSIST_NEXT,
                              "max_bucket_occupancy": MAX_BUCKET_OCCUPANCY,
                              "erasure_direction": "Road-side flips (lane/hood DILATION on "
                                                   "this vehicle; mirror of witness erosion)"},
            residual=0.0,
            source_artifact=f"{ART_DIR}/temporal.json",
            measurement_method="per-frame disagreement masks + stratum/class attribution + "
                               "consecutive-frame persistence + occupancy",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="c2_separatrix_asymmetry_slope_n600_20260716",
            measurement_utc="2026-07-16T06:05:00Z",
            inputs={"field": "cached bit-exact margins (gt_n600.npz)",
                    "profile": "m_med(k), k=1..6px per (pair, side); slope=(m3-m1)/2",
                    "frames": "stride-5 (120 of n600; 6k-0.9M px per pair)"},
            predicted_output={"fine_class_always_shallow": False,
                              "note": "operator INFERRED reading refined: per-pair, not global"},
            empirical_output={"asym_ratio": SLOPE_ASYM_RATIO, "shallow_side": SHALLOW_SIDE,
                              "movable_pairs": "symmetric AND lowest m1 (0.83-0.98): "
                                               "both-sides-fragile annulus"},
            residual=0.0,
            source_artifact=f"{ART_DIR}/slope.json",
            measurement_method="EDT distance-to-boundary + nearest-pair attribution + "
                               "streamed margin histograms",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="c2_onesided_carrier_dominance_smoke_20260716",
            measurement_utc="2026-07-16T06:20:00Z",
            inputs={"form": "2px one-sided border-contrast band, push away from partner "
                            "palette, beta swept 0.5-2.0; 0 counted bytes (constants)",
                    "frames": "stride-5 subset (120) — RANKING ONLY, n600 re-measure owed"},
            predicted_output={"asymmetric_dominates_symmetric": True},
            empirical_output={"baseline": SMOKE_BASELINE_SUBSET,
                              "oneside_movable_b2": SMOKE_ONESIDED_MOVABLE_B2,
                              "gain_frac": -0.295,
                              "symmetric_b05": SMOKE_SYMMETRIC_B05,
                              "deepside_b05": SMOKE_DEEPSIDE_B05,
                              "movable_far_cured_frac": MOVABLE_FAR_CURED_FRAC_BY_BAND,
                              "tex_global_ds16_oracle": SMOKE_TEX_GLOBAL_DS16,
                              "luma_cure_energy_frac": LUMA_CURE_ENERGY_FRAC_MED,
                              "flat_coherence_max": FLAT_COHERENCE_MAX},
            residual=0.0,
            source_artifact=f"{ART_DIR}/smoke_oneside_movable_b2.0.json",
            measurement_method="carrier-variant realization through the real frozen SegNet "
                               "argmax; VJP cure-driver decomposition (sens.json)",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Per-class × per-stratum residual carrier taxonomy + separatrix asymmetry (n600)",
        one_line_summary=(
            "residual 92.6% in 6 buckets; separatrix asymmetry PER-PAIR (max 2.78x); cure = "
            "non-local luma texture; one-sided Movable band -29.5% @ 0B vs symmetric +73% "
            "(region identity is border-driven)"
        ),
        latex_form=(
            r"\mathrm{asym}(p,q)=\frac{\partial m/\partial n\,\big|_{deep}}"
            r"{\partial m/\partial n\,\big|_{shallow}},\quad"
            r"\Delta d_{seg}^{carrier} = d_{seg}(G+\beta\,\mathbb{1}_{band\cap side}\,"
            r"(v_c - v_{partner})) - d_{seg}(G)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.perclass_stratum_carrier_taxonomy_20260716:"
            "one_sided_carrier_gain"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet, exact SegnetWrapper preprocess",
            "vehicle": ("bucket weights: palette+hood render; slope/cure-driver/one-sided "
                        "laws: frozen-scorer-facing (vehicle-agnostic)"),
            "verdict_scope": ("smokes = stride-5 subset RANKING (n600 + byte-closed A/B owed "
                              "before composition); tex_band/tex_global negatives are "
                              "INSTANCE-scoped naive-paste forms"),
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"dseg_variant": "argmax disagreement fraction",
                  "dseg_baseline": "argmax disagreement fraction"},
        units_out={"gain_frac": "fractional d_seg change (negative = win)"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"asymmetric_dominates_symmetric": 0.0,
                                         "fine_class_always_shallow_refined": 0.0},
        last_calibration_utc="2026-07-16T06:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "c2 witness design (#515): boundary-annulus luma-texture budget + one-sided "
            "per-pair band carriers + xi-tracked Movable border carriers",
            "separatrix_asymmetry_t_subpixel_boundary_localizer_v1 (n6 -> n600 upgrade)",
            "necessity_generator_seed_dseg_calibration_v1 (residual decomposed)",
        ),
        canonical_producers=(TOOL, MEMO, f"{ART_DIR}/temporal.json", f"{ART_DIR}/slope.json",
                             f"{ART_DIR}/sens.json"),
        provenance=provenance,
    )


def populate_perclass_stratum_residual_carrier_taxonomy_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of FEED-c2-taxonomy)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_perclass_stratum_residual_carrier_taxonomy_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "RESIDUAL_DSEG",
    "BUCKET_FRACS",
    "SLOPE_ASYM_RATIO",
    "SHALLOW_SIDE",
    "LUMA_CURE_ENERGY_FRAC_MED",
    "FLAT_COHERENCE_MAX",
    "SMOKE_ONESIDED_MOVABLE_B2",
    "MOVABLE_FAR_CURED_FRAC_BY_BAND",
    "one_sided_carrier_gain",
    "carrier_side_for_pair",
    "build_perclass_stratum_residual_carrier_taxonomy_v1",
    "populate_perclass_stratum_residual_carrier_taxonomy_equation",
]
