# SPDX-License-Identifier: MIT
"""Canonical equation: through-R margin-annulus waterfill — flip-CONCENTRATION (positive,
routing-key valid) vs STORE-side paste REALIZATION penalty (negative), and the store-vs-
witness-INR-capacity NON-TRANSFER (FEED-wf, the KKT-capacity-routing $0 gate).

THE GATE (operator: "run the EXISTING measure_waterfill_through_R.py at n600 to confirm the
modeled waterfill win holds THROUGH R"). ``tools/measure_waterfill_through_R.py`` measures the
STORE-side targeted waterfill: store/correct only the low-margin boundary annulus (bytes) and
leave the high-margin bulk un-stored, vs the naive whole-bulk store. It is a decode-free
economics measurement (warps/pastes GT RGB, NO witness INR, NO conv-weight allocation).

TWO coupled MEASURED facts (n600, all 600 pairs, through the ACTUAL contest R + frozen
CPU-torch SegNet argmax ``lstars``, NO-FAKE self-check PASS SegNet(gt_f1)==lstars,
``[macOS-CPU advisory]`` NON-PROMOTABLE, NEVER MPS):

1. FLIP-CONCENTRATION (POSITIVE — the routing key is VALID). Through-R bulk argmax flips are
   ~17x ENRICHED in the low-margin annulus: at margin<3 (only 5.7% of bulk pixels) the annulus
   captures 0.982 of all through-R flips; residual OUTSIDE = 7.9e-5 << the d_seg budget 1.23e-3.
   So the d_seg residual through R LIVES on the low-margin band -> ``sal = exp(-margin/tau)``
   (the inline LEVER-4 / DSL MarginSaliency routing key) is the CORRECT place to route effort.
   The IDEALIZED targeted-annulus store (perfect-correction assumption) S_bulk=0.09365 at tau=3
   BEATS the naive whole-bulk 0.1185 and the FEED-kd model 0.108.

2. STORE-side REALIZATION penalty (NEGATIVE — for the STORE lever only). When the annulus
   correction is REALIZED (paste GT RGB at the annulus -> R -> SegNet), the realized residual
   does NOT drop toward 0 as the annulus grows; the REALIZED best is FULL-STORE (tau=1e6,
   S_bulk=0.10575), every finite targeted tau is dominated. Realization efficiency
   ``eta_R = (res_noov - res_realized)/(res_noov - res_idealized) ~= 0.35`` at tau=12: only
   ~35% of the idealized correction survives R. Mechanism: R (bicubic^->uint8->bilinear_) + the
   SegNet stride-2 stem re-sample means a LOCALLY-pasted correction at pixel p does NOT
   deterministically fix the through-R argmax at p (the FEED-* R-low-pass/Gibbs wall, spatial
   dual of the erasure tail); the paste also adds seam FP.

THE STORE-vs-INR NON-TRANSFER (the decisive distinction). This gate measures the STORE-side
waterfill: ``S_bulk = 100*residual_realized + rate_STORED_BYTES``. It does NOT train a witness
INR nor allocate the witness's ~83K conv weights. The store-side realized NEGATIVE therefore
does NOT transfer to the witness INR-capacity-routing lever (``boundary_routing.py`` ported as a
TRAINING-TIME allocator riding LEVER-4): a training-time capacity/loss allocator RE-ADAPTS the
whole coordinate field THROUGH R (no paste, no seam, no stored-correction rate). What DOES
transfer: fact 1 (the flip-concentration precondition is met). What is UNTESTED here: (b) the
DAG basis-precondition ("capacity ALONE on an isotropic basis HURTS +6%; pays only after
directional/self-orient basis-match") and the ACTUAL witness d_seg movement — both require a
witness TRAINING A/B, not this decode-free store gate.

VERDICT (honest, NO-FAKE, kill-is-last-resort): the STORE-side targeted waterfill is a NEGATIVE
through R at n600 (realized full-store dominates) -> do NOT build a store-side annulus
flip-correction coder. The witness INR-capacity-routing lever is DEFER-pending-research (NOT
kill, NOT build): its precondition is confirmed but its decisive evidence is a different,
unmeasured gate. REACTIVATION = a witness training A/B (n600, through R, byte-closed): with the
self-orient directional basis engaged, (capacity-route/margin-saliency-capacity ON) vs (OFF),
compare d_seg through R; build + fold ONLY if the ON arm lowers d_seg after basis-match.

Consumers: the DSL MarginSaliency lever (routing-key validity) + boundary_routing (the DEFERRED
port whose reactivation criterion this equation IS). Producer: the n600 gate tool.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "waterfill_annulus_through_r_store_realization_vs_witness_capacity_v1"

_UTC = "2026-07-01T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_REPORT = "experiments/results/waterfill_through_R_n600_r2/results.json"

# MEASURED anchors (n600, through R, frozen CPU-torch SegNet argmax; _REPORT).
_RES_NOOV = 0.004487        # through-R no-override BULK d_seg (flip fraction; N=115.8M bulk px)
# fact 1: margin-annulus flip concentration (b_compliant curve, tau=3)
_TAU_MARGIN = 3.0
_FLIP_CAPTURE = 0.9823      # fraction of through-R bulk flips inside margin<3
_ANNULUS_FRACTION = 0.0571  # 1 - safe_fraction (0.9429) = annulus px fraction at tau=3
_RES_OUT_IDEAL = 0.000079   # idealized residual OUTSIDE the annulus at tau=3 (< budget 1.23e-3)
_S_IDEAL_TARGETED = 0.09365  # idealized targeted S_bulk at tau=3 (< naive 0.1185, < model 0.108)
_S_NAIVE = 0.1185           # naive whole-bulk store rate/S reference
# fact 2: store-side REALIZATION (GT-paste sweep, sdf annulus tau=12 vs full-store)
_RES_IDEAL_T12 = 0.000489   # idealized A residual at tau=12
_RES_REALIZED_T12 = 0.003088  # realized B (GT-paste) residual at tau=12
_S_REALIZED_FULLSTORE = 0.10575  # realized best = FULL-STORE (tau=1e6): 100*0 + rate 0.1058


def flip_concentration_enrichment(flip_capture: float, annulus_fraction: float) -> float:
    """Enrichment ratio = (fraction of flips captured) / (fraction of pixels in the annulus).

    >1 means the through-R flips are DENSER in the low-margin annulus than uniform. MEASURED
    ~17x at margin<3 -> the margin-saliency routing key is valid (flips concentrate on the band)."""
    if annulus_fraction <= 0.0:
        raise ValueError(f"annulus_fraction must be > 0; got {annulus_fraction}")
    return float(flip_capture) / float(annulus_fraction)


def predict_store_realized_residual(
    residual_noov: float, residual_idealized_out: float, realization_efficiency: float,
) -> float:
    """Predicted REALIZED through-R residual of a targeted-annulus paste correction.

    ``residual_realized = residual_noov - eta_R * (residual_noov - residual_idealized_out)``.
    ``eta_R`` (the through-R paste realization efficiency) is MEASURED ~0.35 (only ~35% of the
    idealized correction survives R) -> the realized best is FULL-STORE, targeted is dominated.
    Returns the realized flip fraction (the d_seg contribution is 100x this)."""
    eta = min(max(float(realization_efficiency), 0.0), 1.0)
    return float(residual_noov) - eta * (float(residual_noov) - float(residual_idealized_out))


# eta_R derived from the measured tau=12 point (consistency: reproduces _RES_REALIZED_T12).
_ETA_R = (_RES_NOOV - _RES_REALIZED_T12) / (_RES_NOOV - _RES_IDEAL_T12)  # ~0.350


def build_waterfill_annulus_through_R_store_vs_capacity_v1() -> CanonicalEquation:
    """Build the FEED-wf through-R waterfill store-vs-capacity canonical equation."""

    anchor_concentration = EmpiricalAnchor(
        anchor_id="margin_annulus_flip_concentration_through_R_n600_20260701",
        measurement_utc=_UTC,
        inputs={"condition": "b_compliant_warped_canonical_margin", "tau_margin": _TAU_MARGIN,
                "n_pairs": 600, "bulk_classes": ["Road", "Undriv", "MyCar"]},
        predicted_output={"flip_capture": _FLIP_CAPTURE,
                          "enrichment": flip_concentration_enrichment(_FLIP_CAPTURE, _ANNULUS_FRACTION)},
        empirical_output={"flip_capture": _FLIP_CAPTURE, "annulus_fraction": _ANNULUS_FRACTION,
                          "residual_outside": _RES_OUT_IDEAL, "s_idealized_targeted": _S_IDEAL_TARGETED,
                          "s_naive_whole_bulk": _S_NAIVE,
                          "verdict": "POSITIVE: through-R flips ~17x enriched in low-margin annulus "
                                     "-> margin-saliency routing key VALID"},
        residual=0.0,
        source_artifact=_REPORT,
        measurement_method="n600_through_R_frozen_cpu_torch_segnet_argmax",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_REPORT,
            reactivation_criteria="re-measure through R if the R operator or SegNet snapshot changes",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_realization = EmpiricalAnchor(
        anchor_id="store_side_targeted_waterfill_realization_negative_through_R_n600_20260701",
        measurement_utc=_UTC,
        inputs={"condition": "MEASUREMENT_B_GTpaste_sdf_annulus", "tau_sdf": 12.0, "n_pairs": 600,
                "realization_efficiency_eta_R": _ETA_R},
        predicted_output={"residual_realized": predict_store_realized_residual(
            _RES_NOOV, _RES_IDEAL_T12, _ETA_R)},
        empirical_output={"residual_realized_tau12": _RES_REALIZED_T12,
                          "s_realized_best": _S_REALIZED_FULLSTORE, "realized_best_tau": "full_store_1e6",
                          "verdict": "NEGATIVE (store-side): realized best = FULL-STORE; targeted paste "
                                     "dominated (eta_R~0.35). Does NOT transfer to witness INR capacity."},
        residual=abs(predict_store_realized_residual(_RES_NOOV, _RES_IDEAL_T12, _ETA_R) - _RES_REALIZED_T12),
        source_artifact=_REPORT,
        measurement_method="n600_through_R_frozen_cpu_torch_segnet_argmax",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_REPORT,
            reactivation_criteria=("witness TRAINING A/B (n600, through R, byte-closed): self-orient "
                                   "basis ON x capacity-route ON-vs-OFF -> build+fold ONLY if ON lowers d_seg"),
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Through-R margin-annulus waterfill: flip-concentration (positive) vs store-side "
              "paste realization (negative); store-vs-witness-INR-capacity non-transfer"),
        one_line_summary=(
            "n600 through R: flips ~17x enriched in margin<3 annulus (routing key VALID) but store-side "
            "targeted paste best=FULL-STORE (eta_R~0.35) -> STORE lever NEGATIVE; witness INR-capacity DEFER."
        ),
        latex_form=(
            r"E=\frac{\mathrm{capture}}{\mathrm{annulus\ frac}}\approx17;\ "
            r"r_{real}=r_{noov}-\eta_R\,(r_{noov}-r_{ideal}),\ \eta_R\approx0.35\Rightarrow \min_\tau S_{bulk}=\text{full-store}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.waterfill_annulus_through_r_store_vs_capacity_20260701:"
            "predict_store_realized_residual"
        ),
        domain_of_validity={
            "vehicle": ["decode_free_store_side_waterfill"],
            "measurement_axis": [_ADVISORY],
            "result_type": "MIXED: POSITIVE concentration precondition + NEGATIVE store realization",
            "store_vs_inr": ("this gate measures the STORE-side (bytes) waterfill, NOT witness INR "
                             "capacity allocation; the store-side negative does NOT transfer to the "
                             "training-time INR-capacity lever, which re-adapts through R"),
            "untested": ["basis_precondition_capacity_pays_only_after_directional_basis_match",
                         "actual_witness_dseg_movement_from_capacity_routing"],
        },
        units_in={
            "residual_noov": "flip_fraction",
            "residual_idealized_out": "flip_fraction",
            "realization_efficiency": "dimensionless_0_to_1",
        },
        units_out={"residual_realized": "flip_fraction"},
        empirical_anchors=(anchor_concentration, anchor_realization),
        predicted_vs_empirical_residual={
            "n600_through_R_frozen_cpu_torch_segnet_argmax": abs(
                predict_store_realized_residual(_RES_NOOV, _RES_IDEAL_T12, _ETA_R) - _RES_REALIZED_T12),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "tac.torch_vehicle.boundary_routing",
        ),
        canonical_producers=(
            "tools/measure_waterfill_through_R.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="waterfill_annulus_through_R_store_vs_capacity.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "flip_concentration_enrichment",
    "predict_store_realized_residual",
    "build_waterfill_annulus_through_R_store_vs_capacity_v1",
]
