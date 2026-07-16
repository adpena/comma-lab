# SPDX-License-Identifier: MIT
"""Canonical equation: the composed Road-Lane gain chain — decision-amplified, control-attenuated.

MEASURED 2026-07-16 on the real frozen SegNet weights (``upstream/models/segnet.safetensors``,
sha256 ``68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6``), sibling of
``segnet_head_rank4_linear_flipdist_v1`` (the head half of the same chain).

The law
-------
Let ``G_cc'(x) = || d(z_c - z_c') / d(input) ||_2`` be the COMPOSED input->pair-margin gain of the
frozen scorer at a (c,c') boundary pixel (full encoder + skips + decoder + head), and
``||w_c - w_c'||`` the head pair normal (feature-space decision gain).  Measured at real GT boundary
pixels:

  * HEAD (feature space): Lane pair normals are the LARGEST (3.75-4.01 vs 2.60-2.95) — Lane feature
    noise flips cheapest (the sibling law).
  * COMPOSED (pixel space): the Road-Lane gain is the LOWEST — G_RL med 0.0212 vs 0.040-0.058 for
    every other major pair (1.9-2.5x attenuation), with the largest input flip distance
    (|m|/G med 16.0 input-L2 units vs 3.4-8.8).

  ==> the frozen net is a HIGH-decision-gain head reading a LOW-pixel-gain lane-evidence channel:
  Road-Lane is simultaneously FLICKER-PRONE (feature noise, low island persistence) and
  AMPLITUDE-CURE-RESISTANT (attenuated pixel control).  The only efficient control coordinate for
  Road-Lane is sub-pixel boundary GEOMETRY/PHASE — this derives the measured flat-amplitude
  exhaustion (c2 witness-own decomp §3) from the frozen weights, and yields the per-pair weighting
  prior ``lambda_cc' ∝ 1/G_cc'`` (first-order equalization of margin-space progress per unit
  training pressure: lambda_RL ≈ 1.9-2.5x other pairs).

Supporting measured structure (same chain): the stride-2 16-ch skip carries the largest RL gain
share (skip-detach ratio 1.344 vs 1.03-1.11 other pairs); no PRIVATE lane channel exists in the skip
(RL and R-Undrivable per-channel grad profiles near-identical; top RL channel ch10 is chroma-tuned,
99.6% chroma-plane filter); input-gradient luma share ~90% at all boundaries.  Dynamics side: Lane
island persistence median 0.625 (42% < 0.5) vs Movable 2.63 / MyCar 9.13; GT birth+death events
~9.4+9.5 per scored step on ~19.1 interior islands (upper bound at the 5%-overlap tolerance).

Scope / honesty: gains measured at (384,512) scorer-input space (NOT through R); n=20 boundary px
per major pair (4 frames of n96); events/persistence at full n600 from the bit-exact GT cache.
Axis ``[macOS-CPU advisory]``; research_only; no score claim.

Artifacts: ``experiments/results/lane_channel_refactor_20260716/{s1_gain_chain,s2_margins_flicker,``
``s3_dash_geometry,s4_events_t1_audit}.json``
Memo: ``.omx/research/lane_channel_deep_refactorization_20260716.md``
"""

from __future__ import annotations

from collections.abc import Mapping

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "lane_gain_chain_composed_v1"
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; real frozen weights"
SEGNET_WEIGHTS_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
S1_ARTIFACT = "experiments/results/lane_channel_refactor_20260716/s1_gain_chain.json"
S3_ARTIFACT = "experiments/results/lane_channel_refactor_20260716/s3_dash_geometry.json"
S4_ARTIFACT = "experiments/results/lane_channel_refactor_20260716/s4_events_t1_audit.json"

#: MEASURED composed input->pair-margin gain medians (s1_gain_chain.json per_pair_gain_table).
COMPOSED_GAIN_MED: Mapping[str, float] = {
    "Road-Lane": 0.0212, "Road-Undrivable": 0.0404, "Road-Movable": 0.0471,
    "Road-MyCar": 0.0537, "Undrivable-Movable": 0.0415,
}
#: MEASURED skip-detach gain ratios (share of gain carried by the stride-2 skip path).
SKIP_GAIN_RATIO_MED: Mapping[str, float] = {
    "Road-Lane": 1.344, "Road-Undrivable": 1.052, "Road-Movable": 1.082,
    "Road-MyCar": 1.027, "Undrivable-Movable": 1.044,
}
#: MEASURED Lane island persistence stats (s3_dash_geometry.json persistence_by_class).
LANE_ISLAND_PERSISTENCE_MEDIAN = 0.625
LANE_ISLAND_FRAC_BELOW_05 = 0.419
#: MEASURED n600 island-event rates (s4_events_t1_audit.json).  These are upper
#: bounds under the documented 5%-overlap/interpolation tolerance, not exact
#: saddle-node identities.  Exported so controller consumers do not copy numbers.
BIRTHS_PER_FRAMESTEP = 9.43
DEATHS_PER_FRAMESTEP = 9.50
INTERIOR_ISLANDS_PER_FRAME = 19.1


def perpair_lambda_prior(gain_cc: float, gain_ref: float) -> float:
    """DERIVED per-pair weighting prior: lambda_cc' / lambda_ref = gain_ref / gain_cc'.

    First-order equalization of margin-space progress per unit training pressure across
    class pairs, given the measured composed gains.  ``gain_cc`` / ``gain_ref`` are composed
    input->margin gain medians (e.g. from :data:`COMPOSED_GAIN_MED`).  Recorded as a SEED for
    the next per-pair lambda sweep (duty-to-measure) — not an asserted optimum.
    """
    if gain_cc <= 0 or gain_ref <= 0:
        raise ValueError("gains must be > 0")
    return float(gain_ref) / float(gain_cc)


def build_lane_gain_chain_composed_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SEGNET_WEIGHTS_SHA256,
        source_path=S1_ARTIFACT,
        captured_at_utc="2026-07-16T14:10:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="lane_composed_gain_inversion_20260716",
            measurement_utc="2026-07-16T14:05:00Z",
            inputs={
                "weights_sha256": SEGNET_WEIGHTS_SHA256,
                "frames": "4 of n96 real GT frame_1; 5 boundary px per pair per frame",
                "space": "(384,512) scorer-input; full frozen chain autograd",
            },
            predicted_output={
                "law": "head-largest Lane normals need NOT survive composition; "
                       "G_RL measured lowest",
            },
            empirical_output={
                "composed_gain_med": dict(COMPOSED_GAIN_MED),
                "skip_gain_ratio_med": dict(SKIP_GAIN_RATIO_MED),
                "input_flipdist_med_RoadLane": 16.0,
                "luma_share_med_all_pairs": "~0.90",
            },
            residual=0.0,
            source_artifact=S1_ARTIFACT,
            measurement_method=(
                "exact pairwise-margin backward through the real frozen forward at real GT "
                "boundary pixels; second backward with the stride-2 skip detached; BT.601 "
                "luma projection of the per-pixel input gradient"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="lane_island_persistence_histogram_n600_20260716",
            measurement_utc="2026-07-16T13:45:00Z",
            inputs={"gt_cache": "gt_n600.npz lstars+margins (bit-exact frozen SegNet cache)"},
            predicted_output={"expectation": "degraded dash islands sit near the argmax threshold"},
            empirical_output={
                "lane_pers_median": LANE_ISLAND_PERSISTENCE_MEDIAN,
                "lane_frac_below_0.5": LANE_ISLAND_FRAC_BELOW_05,
                "movable_pers_median": 2.63,
                "mycar_pers_median": 9.13,
                "islands_per_frame": 20.6,
            },
            residual=0.0,
            source_artifact=S3_ARTIFACT,
            measurement_method=(
                "8-conn connected components of the Lane class over all 600 cached GT argmax "
                "frames; persistence = max top1-top2 margin inside each island"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="lane_island_birth_death_rates_t1_cadence_20260716",
            measurement_utc="2026-07-16T13:50:00Z",
            inputs={
                "matching": "exact T1 machinery (cross_scored_frame_xi_interp + ground-homography "
                            "advection); interior islands area>=3; <5%-overlap events",
            },
            predicted_output={"expectation": "non-trivial GT event rate on the low-persistence layer"},
            empirical_output={
                "births_per_framestep": 9.43,
                "deaths_per_framestep": 9.50,
                "interior_islands_per_frame": 19.1,
                "t1_weight_stale_death_frac": 0.051,
                "t1_birth_uncovered_frac_of_candidates": 0.263,
                "t1_birth_uncovered_lane_adjacent_px_per_frame": 354.2,
            },
            residual=0.0,
            source_artifact=S4_ARTIFACT,
            measurement_method=(
                "xi-advected island matching at the scored-pair cadence (n600) + exact "
                "reconstruction of the T1 weight mask (trainer formulation) on 60 pairs; "
                "event counts are upper bounds at the 5%-overlap/interp-gap tolerance"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Composed Road-Lane gain chain: decision-amplified, control-attenuated; phase is the efficient control",
        one_line_summary=(
            "Head amplifies Lane but composed pixel gain G_RL is LOWEST (0.021 vs 0.040-0.058): "
            "flicker-prone AND amplitude-cure-resistant; phase is the efficient control; "
            "lambda_cc' ∝ 1/G_cc'."
        ),
        latex_form=(
            r"G_{cc'}(x)=\big\|\nabla_x\,(z_c-z_{c'})\big\|_2,\quad "
            r"G_{RL}\approx\tfrac{1}{2}G_{cc'\neq RL}\ \text{while}\ \|w_L-w_{c'}\|=\max,\quad "
            r"\lambda_{cc'}\propto G_{cc'}^{-1}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_gain_chain_composed_20260716:perpair_lambda_prior"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet (smp.Unet tu-efficientnet_b2, classes=5, activation=None)",
            "weights_sha256": SEGNET_WEIGHTS_SHA256,
            "space": "(384,512) scorer-input space; NOT re-measured through R (owed if a lever "
                     "consumes the exact ratio through-R)",
            "sampling": "n=20 boundary px per major pair (4 frames of n96); events/persistence n600",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "verdict_scope": "frozen-weight chain law + this video's GT dynamics; lambda prior is "
                             "DERIVED first-order, duty-to-measure",
        },
        units_in={"gain_cc": "logit_per_input_L2", "gain_ref": "logit_per_input_L2"},
        units_out={"lambda_ratio": "dimensionless"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"gain_inversion_sign": 0.0},
        last_calibration_utc="2026-07-16T14:10:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "c2 phase-stage event-fallback amendment (spec_c2_surgical, owed before ep700)",
            "next per-pair lambda sweep seed (duty-to-measure)",
            "world-frame per-dash anchor carrier design (L71/#425 archive shape)",
        ),
        canonical_producers=(
            "experiments/results/lane_channel_refactor_20260716/s1_gain_chain.py",
            "experiments/results/lane_channel_refactor_20260716/s3_dash_geometry.py",
            "experiments/results/lane_channel_refactor_20260716/s4_events_t1_audit.py",
        ),
        provenance=provenance,
    )


__all__ = [
    "BIRTHS_PER_FRAMESTEP",
    "COMPOSED_GAIN_MED",
    "DEATHS_PER_FRAMESTEP",
    "EQUATION_ID",
    "INTERIOR_ISLANDS_PER_FRAME",
    "LANE_ISLAND_FRAC_BELOW_05",
    "LANE_ISLAND_PERSISTENCE_MEDIAN",
    "SKIP_GAIN_RATIO_MED",
    "build_lane_gain_chain_composed_v1",
    "perpair_lambda_prior",
]
