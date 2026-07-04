# SPDX-License-Identifier: MIT
"""Canonical equation: LEVER-D flicker-residual reactivation ECONOMICS (the FEED-03w synthesis).

The Lever-D reactivation (#279) reduces to ONE Stage-0 byte-measurement. This equation formalizes the
go/no-go so the equations leg AGREES with DAG FEED-03u (the design net-S band) + FEED-03v (the 6.897
coder floor) + the DSL ``FlickerTreatmentGauge.STORE_REGIONAL_LEVERD`` chart:

  net ΔS(witness) = −r · D_ceiling + 25 · coded_bytes / N ,   D_ceiling = 100 · d_seg_now = 0.4964

where ``r`` = the net-of-collateral recovery fraction of the current witness d_seg (the deciding unknown),
``coded_bytes`` = the realized COUNTED residual sidecar bytes, N = 37,545,489 (the contest rate denom).

TWO composed facts (both advisory / SPEC — NO byte-closed exact row; pointer 0.19110 UNMOVED; MEANS):

1. THE NET-S BAND (design memo ``leverd_flicker_residual_reactivation_design_20260703.md``, FEED-03u).
   d_seg_now = 0.004964 (#205 @ep225, a949ff63 n600) → the addressable ceiling is −0.4964 S (driving
   d_seg→0, unreachable). Break-even recovery r* = (25·coded_bytes/N) / 0.4964 = 34% @250 KB / 40% @300 KB
   / 54% @400 KB. Stated band: −0.35 OPTIMISTIC (the popout-floor-magnitude corner) / −0.048 EXPECTED /
   +0.117 pessimistic-WORSE. Even the optimistic corner leaves witness S ~0.40 (~2x above the 0.19110
   pointer) — a d_seg-competitiveness increment, NOT a pointer move. The lever touches ONLY d_seg.

2. THE CODER FLOOR (6.897 review ``6897_advanced_data_structures_rate_axis_review_20260703.md``, FEED-03v).
   Lever-D GO requires coded-subset survival σ_eff > σ* = b / WATERLINE, WATERLINE = 1.273108 B/flip
   (imported from the #72 MCR codec). Current hand-coded b ≈ 0.99 → σ* ≈ 0.778; measured best-decile
   σ_eff ≈ 0.51 → NO-GO. Flipping GO at that σ_eff needs b < 0.51·1.273108 ≈ 0.65 B/flip (a ~34% cut).
   The ~250 KB regional-context entropy ≈ SPATIAL-ONLY 0.90 B/flip > 0.65 → NO-GO on the spatial axis
   EVEN at a perfect entropy coder. ONLY the JOINT temporal+spatial conditional entropy (frame-to-frame
   flip persistence → few BWT runs → RLE collapse) can cross 0.65 — a byte-MEASUREMENT #279 must run, NOT
   a guarantee. Coder = H_k-context arithmetic (in-tree RangeEncoder) OR BWT+MTF+RLE (stdlib bz2),
   take min(b); zero new deps.

THE COMBINED VERDICT (FEED-03w): Lever-D = a single Stage-0 byte-measurement on a FROZEN #205 checkpoint
(read-only, 32-pair subset). GO iff min(b) < 0.65 B/flip AND subset net ΔS < 0; else d_seg stays
IN-TRAINING (keep the BUILT #274 down-weight lever as the seg play). Await operator GO for any build/heavy.

means != ends: every net-S number is a SPEC/derivation, advisory, NOT a byte-closed upstream/evaluate.py
row. The pointer (0.19110) moves ONLY through a byte-closed exact eval. #205 SACRED, untouched.

Consumers: the DSL ``FlickerTreatmentGauge`` (STORE_REGIONAL_LEVERD chart) + the gauge cost table.
Producers: the #72 MCR codec (the honest coder + WATERLINE) + the #202 byte-close tool (the Stage-1 A/B).
"""
from __future__ import annotations

from tac.boundary_math.margin_conditional_residual import (
    SEG_VALUE_PER_FLIP,
    WATERLINE_BYTES_PER_FLIP,
)
from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "leverd_flicker_residual_reactivation_economics_v1"

_UTC = "2026-07-03T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"

_DESIGN_MEMO = ".omx/research/leverd_flicker_residual_reactivation_design_20260703.md"
_CODER_MEMO = ".omx/research/6897_advanced_data_structures_rate_axis_review_20260703.md"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- grounding constants (advisory; the deciding r + b are #205-class byte-measurements) -------------
RATE_DENOM = 37_545_489                         # contest rate denominator (bytes)
BYTES_PER_SCORE = RATE_DENOM / 25.0             # 1,501,819.56 B per unit of rate-S
DSEG_NOW = 0.004964                             # #205 d_seg @ep225 (a949ff63, n600) = the flicker floor
D_CEILING_S = 100.0 * DSEG_NOW                  # 0.4964 = max recoverable seg-S (drive d_seg->0; unreachable)
# The #72 MCR codec economics (imported, exact): the survival waterline + the seg value per flip.
WATERLINE = WATERLINE_BYTES_PER_FLIP            # 1.273108 B/flip (== SEG_VALUE_PER_FLIP * BYTES_PER_SCORE)
CURRENT_B = 0.99                                # current hand-coded MCR B/flip (spatial per-pair)
SPATIAL_ONLY_B = 0.90                           # ~250 KB regional-context entropy = spatial-only floor
SIGMA_EFF_BEST_DECILE = 0.51                    # measured best-decile coded-subset survival
CODER_GO_THRESHOLD_B = 0.65                     # == SIGMA_EFF_BEST_DECILE * WATERLINE (0.6493 ~ 0.65)


def leverd_net_delta_s(r: float, coded_bytes: float) -> float:
    """Lever-D net contest ΔS on the witness: ``-r * D_ceiling + 25 * coded_bytes / N``.

    ``r`` = the net-of-collateral recovery fraction of the current witness d_seg (in [0, 1]); the seg
    term recovers ``-r * 0.4964``. ``coded_bytes`` = the realized COUNTED residual sidecar bytes; the
    rate term ADDS ``25 * coded_bytes / 37,545,489``. Negative = a win (lowers the witness S). The lever
    touches ONLY d_seg (pose + base INR rate unchanged)."""
    return -float(r) * D_CEILING_S + 25.0 * float(coded_bytes) / RATE_DENOM


def leverd_break_even_recovery(coded_bytes: float) -> float:
    """The net-of-collateral recovery fraction ``r*`` at which Lever-D breaks even (net ΔS = 0):
    ``r* = (25 * coded_bytes / N) / D_ceiling``. == 0.34 @250 KB / 0.40 @300 KB / 0.54 @400 KB. If the
    realized net recovery clears r*, the lever is a (marginal) win; below it, net-NEGATIVE (worse)."""
    return (25.0 * float(coded_bytes) / RATE_DENOM) / D_CEILING_S


def leverd_survival_threshold(bytes_per_flip: float) -> float:
    """The coded-subset survival break-even ``σ* = b / WATERLINE`` (WATERLINE = 1.273108 B/flip, #72
    MCR). Lever-D GO requires measured σ_eff > σ*. At b=0.99 → σ*≈0.778; measured σ_eff≈0.51 → NO-GO."""
    return float(bytes_per_flip) / WATERLINE


def leverd_coder_go(bytes_per_flip: float) -> bool:
    """The FEED-03w coder go/no-go on the RATE axis: True iff the realized per-flip byte cost ``b`` is
    below the GO threshold (0.65 B/flip == σ_eff_best_decile * WATERLINE). The ~250 KB spatial-only
    entropy (~0.90 B/flip) does NOT clear it; ONLY the joint temporal+spatial conditional entropy can.
    This is a NECESSARY (rate-axis) condition; the FULL GO also requires subset net ΔS < 0."""
    return float(bytes_per_flip) < CODER_GO_THRESHOLD_B


def build_leverd_flicker_residual_reactivation_economics_v1() -> CanonicalEquation:
    """Build the Lever-D flicker-residual reactivation economics canonical equation (FEED-03w synthesis
    of the FEED-03u net-S band + the FEED-03v 6.897 coder floor)."""

    anchor_band = EmpiricalAnchor(
        anchor_id="leverd_net_s_band_design_stress_test_20260703",
        measurement_utc=_UTC,
        inputs={"objective": "net ΔS(witness) = -r*0.4964 + 25*coded_bytes/N",
                "d_seg_now": DSEG_NOW, "d_ceiling_s": D_CEILING_S,
                "r": "net-of-collateral recovery fraction (the deciding unknown)",
                "coded_bytes": "realized COUNTED residual sidecar bytes"},
        predicted_output={"break_even_recovery_at_250kb": leverd_break_even_recovery(250_000),
                          "break_even_recovery_at_300kb": leverd_break_even_recovery(300_000),
                          "break_even_recovery_at_400kb": leverd_break_even_recovery(400_000)},
        empirical_output={
            "corner_optimistic_net_s": -0.35,       # popout-floor-magnitude corner (design headline)
            "corner_expected_net_s": -0.048,        # r=0.50, 300 KB
            "corner_pessimistic_net_s": 0.117,      # r=0.30, 400 KB -> net-NEGATIVE (WORSE)
            "corner_optimistic_r070_250kb_net_s": -0.177,  # r=0.70 intermediate (memo rounds rate 0.170)
            "break_even_recovery_pct": "34% @250 KB / 40% @300 KB / 54% @400 KB",
            "note": ("SPEC band (design memo). -0.35 is the OPTIMISTIC corner; EXPECTED ~-0.05; "
                     "pessimistic net-NEGATIVE. Even the optimistic corner leaves witness S ~0.40 "
                     "(~2x above the 0.19110 pointer) -> a d_seg-competitiveness increment, NOT a pointer "
                     "move. The lever touches ONLY d_seg. rate_S = 25*bytes/N exactly (250 KB -> 0.1665; "
                     "the memo rounds to 0.170)."),
        },
        residual=0.0,
        source_artifact=_DESIGN_MEMO,
        measurement_method="design_stress_test_net_s_band_spec_advisory",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DESIGN_MEMO,
            reactivation_criteria=("Stage-0 subset proof: measure the realized net recovery r + coded "
                                   "bytes on a FROZEN #205 checkpoint (32-pair subset); Stage-1 = the "
                                   "byte-closed n600 A/B through #202 levelset_byte_close_and_eval.py"),
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    anchor_coder = EmpiricalAnchor(
        anchor_id="leverd_coder_floor_6897_go_threshold_20260703",
        measurement_utc=_UTC,
        inputs={"survival_rule": "GO requires σ_eff > σ* = b / WATERLINE",
                "waterline_bytes_per_flip": WATERLINE,
                "current_b": CURRENT_B, "sigma_eff_best_decile": SIGMA_EFF_BEST_DECILE,
                "coders": "H_k-context arithmetic (in-tree RangeEncoder) OR BWT+MTF+RLE (stdlib bz2)"},
        predicted_output={"go_threshold_b": CODER_GO_THRESHOLD_B,
                          "survival_threshold_at_current_b": leverd_survival_threshold(CURRENT_B),
                          "spatial_only_go": leverd_coder_go(SPATIAL_ONLY_B)},
        empirical_output={
            "current_survival_threshold": leverd_survival_threshold(CURRENT_B),  # 0.778 @b=0.99
            "measured_sigma_eff_best_decile": SIGMA_EFF_BEST_DECILE,             # 0.51 -> NO-GO now
            "go_threshold_b": CODER_GO_THRESHOLD_B,                              # b<0.65 (~-34%)
            "spatial_only_b": SPATIAL_ONLY_B,                                    # 0.90 > 0.65 -> NO-GO
            "spatial_only_go": leverd_coder_go(SPATIAL_ONLY_B),                  # False
            "note": ("current b~0.99 -> σ*~0.778 > σ_eff~0.51 = NO-GO. GO needs b<0.65 B/flip (~-34%). "
                     "The ~250 KB regional entropy ~0.90 B/flip is SPATIAL-ONLY > 0.65 -> NO-GO EVEN at a "
                     "perfect entropy coder; ONLY the JOINT temporal+spatial conditional entropy (flip "
                     "persistence -> BWT runs -> RLE collapse) can cross 0.65 -- a byte-MEASUREMENT #279 "
                     "must run, NOT a guarantee. positions already at the L12/L31 colex-succinct optimum. "
                     "zero new deps (in-tree RangeEncoder / stdlib bz2)."),
        },
        residual=0.0,
        source_artifact=_CODER_MEMO,
        measurement_method="6897_rate_coding_toolkit_review_byte_reasoned_advisory",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_CODER_MEMO,
            reactivation_criteria=("byte-close the frame-major residual through (a) the temporal-context "
                                   "RangeEncoder and (b) BWT+MTF+RLE; read min(b); GO iff min(b)<0.65"),
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Lever-D flicker-residual reactivation economics: net ΔS = -r*0.4964 + 25*bytes/N, GO iff "
              "coder min(b)<0.65 B/flip AND subset net ΔS<0"),
        one_line_summary=(
            "Lever-D = ONE Stage-0 byte-measurement: GO iff min(b)<0.65 B/flip AND subset net ΔS<0 "
            "(band -0.35/-0.048/+0.117); else d_seg stays IN-TRAINING (keep #274 down-weight)."
        ),
        latex_form=(
            r"\Delta S = -r\,D_{\text{ceil}} + \tfrac{25\,b_{\text{bytes}}}{N},\ D_{\text{ceil}}=0.4964;\ "
            r"\text{GO}\iff \min b<0.65\ \tfrac{B}{\text{flip}}\ \big(\sigma^*=\tfrac{b}{1.2731}\big)"
            r"\ \wedge\ \Delta S_{\text{subset}}<0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.leverd_flicker_residual_reactivation_economics_20260703:"
            "leverd_net_delta_s"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": "Lever-D (#279) temporal flip-residual STORE (COUNTED 7th archive block)",
            "scope": "d_seg ONLY (pose + base INR rate unchanged); the residual is COUNTED, nudge FREE",
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "note": ("the net-S band + b-floor are a SPEC/derivation; the deciding r (net recovery) + "
                     "min(b) are the Stage-0 byte-measurements #279 must produce; even the optimistic "
                     "corner is ~2x above the 0.19110 pointer (a d_seg increment, not a pointer move)"),
        },
        units_in={"r": "net_recovery_fraction_0_1", "coded_bytes": "archive_bytes"},
        units_out={"net_delta_s": "contest_score_delta"},
        empirical_anchors=(anchor_band, anchor_coder),
        predicted_vs_empirical_residual={"leverd_net_s_band_spec": 0.0, "leverd_coder_go_threshold": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=("tac.witness_dsl.gauge",),
        canonical_producers=(
            "src/tac/boundary_math/margin_conditional_residual.py",
            "tools/levelset_byte_close_and_eval.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="leverd_flicker_residual_reactivation_economics.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="unknown",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "CODER_GO_THRESHOLD_B",
    "DSEG_NOW",
    "D_CEILING_S",
    "WATERLINE",
    "build_leverd_flicker_residual_reactivation_economics_v1",
    "leverd_break_even_recovery",
    "leverd_coder_go",
    "leverd_net_delta_s",
    "leverd_survival_threshold",
]
