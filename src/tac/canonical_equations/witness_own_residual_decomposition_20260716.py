# SPDX-License-Identifier: MIT
"""Witness-own per-class × per-stratum residual decomposition + law-transfer verdicts.

Sibling of ``perclass_stratum_residual_carrier_taxonomy_v1`` (whose #1 caveat — palette
vehicle DILATES, witness ERODES — this closes). Decomposes the TRAINED witness's own
residual (frozen mod32cap EMA-best ep650, d_seg 0.003146 n600 through the exact contest
R + frozen CPU SegNet) per (class × stratum), and re-verifies each taxonomy law on the
witness frames. Five MEASURED results (2026-07-16, ``[macOS-CPU advisory]``):

  1. WITNESS RESIDUAL = LANE-DOMINATED EDGE FLICKER: Lane|edge 38.0% + Road|edge 29.2%
     + Undrivable|edge 9.8% + Movable|edge 9.6% + MyCar|edge 4.0% -> edge stratum 90.6%
     of residual; the Road-Lane pair alone = 66.0% (pair-side, edge+near). NO Movable|far
     bucket exists (<100 px total over n600; the palette's #2 bucket at 24.3% is ABSENT).
     persist_next <= 0.25 and occupancy <= 0.07 in EVERY bucket -> per-frame boundary
     FLICKER, not static sites nor object misses (consistent with L85 GT sub-pixel
     advection phase + the evasion realization floor "sub-pixel geometry not amplitude").
  2. EROSION CONFIRMED: edge|Lane->Road 165k vs edge|Road->Lane 62k (2.64:1 on the Lane
     side) — the exact mirror of the palette vehicle's dilation (Road-side 240k vs 123k).
     Bucket WEIGHTS do not transfer between vehicles; the frozen-scorer LAWS do.
  3. LAW TRANSFER (holds): luma cure driver (median energy 0.85 witness vs 0.93 palette;
     chroma share doubles on this chroma-active vehicle, luma still dominant); non-local
     (r4 <= 0.26, r36 0.31-0.72); flat-orthogonal (coh <= 0.07) with the Lane
     flat-coherent brighten exception REPLICATED (cohGT 0.28-0.37, sign +9/3) — now on
     the #1 bucket; crisp>blurry (blur sigma=2 -> +20.3%); one-sided ORDERING
     (correct-side < symmetric < deep-side at beta=0.5: +35.8%/+123.8%/+141.6%).
  4. LAW TRANSFER (breaks): the taxonomy's carrier VALUES. The palette winner
     (oneside_movable beta=2, -29.5% @ 0B) is +80.1% on the witness; EVERY flat band
     variant is net-negative (best = beta->0 ~ neutral: oneside_lane beta=0.1 +1.2%).
     beta-monotone WORSENING on the witness vs monotone improving on the palette.
  5. FLAT-AMPLITUDE EXHAUSTION (NEW, formulation-scoped): the trained witness sits at
     the flat-basis optimum of its own residual — a finite flat push cures its target
     flip pixels (oneside_lane beta=0.25: Lane|edge -19%) but buys MORE collateral
     crossings at the ~30x more numerous correctly-classified boundary pixels
     (Road|edge +70%). Post-hoc render-time amplitude carriers pay on UNTRAINED
     vehicles only; witness d_seg bytes must buy joint-trained boundary profile /
     appearance-phase (xi,R) sub-pixel geometry / per-pair boundary offsets.

Verdict scope: result 5 is FORMULATION-level (flat palette-delta band, kpx=2, post-hoc,
5 sides x beta 0.1-2.0) — trained band profiles / sub-pixel placement remain open.
Smokes are stride-5 rankings. NOT a score claim; pointer 0.19108 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "witness_own_residual_decomposition_v1"
AXIS = ("[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached n600 GT; frozen "
        "mod32cap EMA-best ep650 witness rendered through exact R")
TOOL = "tools/c2_witness_own_decomp.py"
ART_DIR = "experiments/results/c2_witness_own_decomp_20260716"
MEMO = ".omx/research/c2_witness_own_decomp_20260716.md"

# ---- MEASURED constants (artifacts: temporal.json / sens.json / smoke_*.json)
WITNESS_DSEG_N600 = 0.003146          # frozen mod32cap ep650 through R (n600)
BUCKET_FRACS = {                      # % of the WITNESS residual, n600
    "Lane|edge": 0.380, "Road|edge": 0.292, "Undrivable|edge": 0.098,
    "Movable|edge": 0.096, "MyCar|edge": 0.040, "Road|near": 0.038,
}
EDGE_STRATUM_FRAC = 0.906             # edge(1px) share of the witness residual
ROAD_LANE_PAIR_FRAC = 0.660           # Road-Lane pair share (pair-side, edge+near)
EROSION_RATIO_LANE = 2.64             # edge|Lane->Road : edge|Road->Lane (165026:62459)
MAX_PERSIST_NEXT = 0.251              # every bucket <= 0.25 -> flicker, not static
MAX_OCCUPANCY = 0.067
LUMA_CURE_ENERGY_FRAC_MED = 0.85      # vs 0.93 on the palette (chroma-active vehicle)
LANE_FLAT_COHERENCE_GT = 0.28         # Lane exception replicated (sign +9/3 brighten)
SMOKE_BASELINE_SUBSET = 0.0031068     # stride-5 subset witness baseline
SMOKE_ONESIDE_MOVABLE_B2 = 0.0055950  # +80.1% (palette winner FAILS on the witness)
SMOKE_ONESIDE_LANE_B01 = 0.0031448    # +1.2% (best flat variant ~ neutral at beta->0)
SMOKE_ONESIDE_MOVABLE_B05 = 0.0042193  # +35.8%
SMOKE_SYMMETRIC_B05 = 0.0069512       # +123.8%
SMOKE_ONESIDE_DEEP_B05 = 0.0075044    # +141.6%
SMOKE_BLUR_S2 = 0.0037374             # +20.3% (crisp>blurry holds)


def flat_band_gain_on_witness(dseg_variant: float,
                              dseg_baseline: float = SMOKE_BASELINE_SUBSET) -> float:
    """Fractional d_seg change of a flat band variant on the witness (negative = win).

    MEASURED law: no flat palette-delta band variant achieves a negative value on the
    trained witness (flat-amplitude exhaustion; formulation-scoped).
    """
    if dseg_baseline <= 0.0:
        raise ValueError(f"dseg_baseline must be > 0, got {dseg_baseline!r}")
    return (dseg_variant - dseg_baseline) / dseg_baseline


def witness_bucket_cure_value_S(bucket_frac: float,
                                dseg_total: float = WITNESS_DSEG_N600) -> float:
    """S-units value of fully curing a witness residual bucket (100 * d_seg share)."""
    if not 0.0 <= bucket_frac <= 1.0:
        raise ValueError(f"bucket_frac must be in [0,1], got {bucket_frac!r}")
    return 100.0 * bucket_frac * dseg_total


def build_witness_own_residual_decomposition_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256="77f760afb33ac3786d9dc3b31669e515489f4e4531cec058f25b57b408fde203",
        source_path=f"{ART_DIR}/temporal.json",
        captured_at_utc="2026-07-16T12:00:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="c2w_witness_own_decomposition_n600_20260716",
            measurement_utc="2026-07-16T11:00:00Z",
            inputs={"frames": "n600 (ALL scored pairs)",
                    "vehicle": "FROZEN mod32cap EMA-best ep650 witness through exact R "
                               "(bicubic^ 874x1164 -> round/clamp/uint8)",
                    "strata": "saddle > edge(1px) > near(<=3px) > far, class = pixel GT"},
            predicted_output={"bucket_weights_differ_from_palette": True,
                              "erosion_not_dilation": True},
            empirical_output={"dseg_n600": WITNESS_DSEG_N600,
                              "bucket_fracs_top6": BUCKET_FRACS,
                              "edge_stratum_frac": EDGE_STRATUM_FRAC,
                              "road_lane_pair_frac": ROAD_LANE_PAIR_FRAC,
                              "erosion_ratio_lane": EROSION_RATIO_LANE,
                              "max_persist_next": MAX_PERSIST_NEXT,
                              "max_occupancy": MAX_OCCUPANCY,
                              "movable_far_bucket": "ABSENT (<100 px total over n600)"},
            residual=0.0,
            source_artifact=f"{ART_DIR}/temporal.json",
            measurement_method="witness render+R+SegNet argmax per frame -> disagreement "
                               "masks + stratum/class attribution + persistence/occupancy "
                               "(schema identical to the taxonomy arm)",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="c2w_law_transfer_cure_drivers_20260716",
            measurement_utc="2026-07-16T11:30:00Z",
            inputs={"probe": "VJP of logit_gt - logit_pred at DISAGREEING px of the "
                             "witness render through the real frozen SegNet (120 samples, "
                             "frame stride 50)"},
            predicted_output={"luma_dominant": True, "non_local": True,
                              "flat_orthogonal_with_lane_exception": True},
            empirical_output={"luma_energy_frac_med": LUMA_CURE_ENERGY_FRAC_MED,
                              "lane_flat_coherence_gt": LANE_FLAT_COHERENCE_GT,
                              "lane_sign": "+9/3 brighten (replicates the palette arm)",
                              "e_frac_r36_range": [0.31, 0.72]},
            residual=0.0,
            source_artifact=f"{ART_DIR}/sens.json",
            measurement_method="same sens decomposition as the taxonomy arm, on witness "
                               "frames",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="c2w_flat_amplitude_exhaustion_smoke_20260716",
            measurement_utc="2026-07-16T11:50:00Z",
            inputs={"form": "2px one-sided palette-delta border band ON the cached "
                            "witness frames; sides {Lane,Movable,shallow,deep,symmetric}; "
                            "beta 0.1-2.0; 0 counted bytes",
                    "frames": "stride-5 subset (120) — RANKING ONLY"},
            predicted_output={"ordering_transfers": True,
                              "palette_winner_transfers": False},
            empirical_output={"baseline": SMOKE_BASELINE_SUBSET,
                              "oneside_movable_b2": SMOKE_ONESIDE_MOVABLE_B2,
                              "oneside_movable_b2_gain": +0.801,
                              "oneside_lane_b01": SMOKE_ONESIDE_LANE_B01,
                              "oneside_movable_b05": SMOKE_ONESIDE_MOVABLE_B05,
                              "symmetric_b05": SMOKE_SYMMETRIC_B05,
                              "oneside_deep_b05": SMOKE_ONESIDE_DEEP_B05,
                              "blur_s2": SMOKE_BLUR_S2,
                              "ordering": "correct-side < symmetric < deep-side HOLDS; "
                                          "net sign FLIPS (all variants >= baseline)",
                              "mechanism": "oneside_lane b0.25 cures Lane|edge -19% but "
                                           "buys Road|edge +70% collateral"},
            residual=0.0,
            source_artifact=f"{ART_DIR}/smoke_oneside_movable_b2.0.json",
            measurement_method="carrier variants composited on cached witness camera "
                               "frames -> frozen SegNet argmax vs GT",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Witness-own residual decomposition + flat-amplitude exhaustion (n600)",
        one_line_summary=(
            "witness residual = Lane-dominated edge flicker (edge 90.6%, Road-Lane 66%, "
            "erosion 2.64:1); scorer laws transfer, carrier values do NOT — every flat "
            "band variant net-negative (exhaustion)"
        ),
        latex_form=(
            r"d_{seg}^{wit}=\sum_{c,s} w_{c,s},\ w_{Lane,edge}=0.38;\quad"
            r"\min_{\beta,side}\ d_{seg}\big(W+\beta\,\mathbb{1}_{band}\,"
            r"(v_c-v_{p})\big)\ \text{at}\ \beta\to 0\ (\text{exhaustion})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.witness_own_residual_decomposition_20260716:"
            "flat_band_gain_on_witness"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet, exact SegnetWrapper preprocess",
            "vehicle": ("bucket weights + exhaustion: frozen mod32cap EMA-best ep650 "
                        "(best available trained witness, d_seg 0.003146 n600); "
                        "transfer-verdicts compare against the palette taxonomy arm"),
            "verdict_scope": ("exhaustion is FORMULATION-level: flat palette-delta band, "
                              "kpx=2, post-hoc composite; trained band profiles / "
                              "sub-pixel placement / joint-trained carriers remain open; "
                              "smokes = stride-5 subset RANKING"),
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"dseg_variant": "argmax disagreement fraction",
                  "bucket_frac": "fraction of witness residual"},
        units_out={"gain_frac": "fractional d_seg change (negative = win)",
                   "cure_value_S": "contest S units (100 * d_seg)"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"erosion_not_dilation": 0.0,
                                         "ordering_transfers": 0.0,
                                         "palette_winner_transfers": 0.0},
        last_calibration_utc="2026-07-16T12:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "c2 witness design (#515): d_seg bytes -> joint-trained boundary profile + "
            "appearance-phase (xi,R) sub-pixel geometry (L85/L86 #424/#425), NOT post-hoc "
            "flat bands on the trained trunk",
            "perclass_stratum_residual_carrier_taxonomy_v1 (its #1 vehicle caveat closed)",
            "oneside_border_band_carrier DSL Lever (OWED): vehicle gate = palette/"
            "necessity ON pending n600 A/B, trained-witness OFF (measured net-negative)",
        ),
        canonical_producers=(TOOL, MEMO, f"{ART_DIR}/temporal.json", f"{ART_DIR}/sens.json",
                             f"{ART_DIR}/decomp_rows.jsonl"),
        provenance=provenance,
    )


def populate_witness_own_residual_decomposition_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of FEED-c2w-witness-own)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_witness_own_residual_decomposition_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "WITNESS_DSEG_N600",
    "BUCKET_FRACS",
    "EDGE_STRATUM_FRAC",
    "ROAD_LANE_PAIR_FRAC",
    "EROSION_RATIO_LANE",
    "LUMA_CURE_ENERGY_FRAC_MED",
    "SMOKE_ONESIDE_MOVABLE_B2",
    "flat_band_gain_on_witness",
    "witness_bucket_cure_value_S",
    "build_witness_own_residual_decomposition_v1",
    "populate_witness_own_residual_decomposition_equation",
]
