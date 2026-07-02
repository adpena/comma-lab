# SPDX-License-Identifier: MIT
"""Canonical equation: analytic-lane render-band FALSE-POSITIVE reduction (FEED-dv).

The NAIVE analytic-lane band HURTS d_seg because it lays lane appearance over EVERY
band pixel -> dash-gap + fit-error FALSE-POSITIVES (MEASURED n600 sizing c3:
+0.00082 d_seg, lane_fp ~2x). The WITNESS-UNCERTAINTY mask (rides the #141 top1-top2
margin) composites the band ONLY where the decision is UNCERTAIN, so the FP that lands
in CONFIDENT regions is killed. This equation predicts the composited Delta d_seg from
the naive FP added, the uncertainty-gate FP-kill fraction, and the FN recovered:

    Delta d_seg(compose) = 100 * ( fp_added_naive * (1 - kill) - fn_recovered )

MEASURED (l7-best levelset ckpt, n600 through R, frozen CPU-torch SegNet argmax,
[macOS-CPU advisory] NON-PROMOTABLE): the naive band adds ~+0.00082; the
witness-uncertainty gate drives it to ~break-even (kill ~0.98). NET-NEGATIVE requires
TRAINING WITH the band active (the witness re-adapts its boundaries; sizing VERDICT) --
this equation bounds the POST-HOC compositing behavior, NOT the trained-in win.

Consumers: the trainer compose wire-in + the DSL lever + the render-side sizing tool.
Producers: tools/levelset_analytic_lane_band_dseg_n600.py (the n600 measurement).
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

EQUATION_ID = "analytic_lane_render_band_fp_reduction_v1"

# MEASURED anchors (n600, l7-best levelset ckpt, through R, frozen CPU-torch SegNet argmax;
# reports/levelset_analytic_lane_band_dseg_n600_20260701.json; c1 reproduced sizing 0.003330 exactly).
_NAIVE_DELTA = 0.000622          # n600 c_naive Delta vs witness-alone (FP 0.000602->0.001024, +70%)
_WIT_UNCERT_DELTA = 0.000012     # n600 c_full_wit Delta (witness-uncertainty gate; ~break-even)
_FP_KILL_FRACTION = 0.981        # 1 - (_WIT_UNCERT_DELTA / _NAIVE_DELTA): 98% of the naive FP killed


def predict_composite_delta_dseg(
    fp_added_naive: float, fp_kill_fraction: float, fn_recovered: float,
) -> float:
    """Predicted composited Delta d_seg (in SCORE units, x100) from the naive FP added,
    the uncertainty-gate FP-kill fraction, and the FN recovered.

    ``fp_added_naive`` / ``fn_recovered`` are FRACTIONS-of-all-pixels (the d_seg unit);
    the x100 matches S = 100*d_seg. A NEGATIVE return is a d_seg improvement."""

    kill = min(max(float(fp_kill_fraction), 0.0), 1.0)
    return 100.0 * (float(fp_added_naive) * (1.0 - kill) - float(fn_recovered))


def build_analytic_lane_render_band_fp_reduction_v1() -> CanonicalEquation:
    """Build the FEED-dv analytic-lane render-band FP-reduction canonical equation."""

    report = "reports/levelset_analytic_lane_band_dseg_n600_20260701.json"
    anchor_naive = EmpiricalAnchor(
        anchor_id="analytic_lane_band_naive_hurts_n600_20260701",
        measurement_utc="2026-07-01T00:00:00Z",
        inputs={"condition": "c_naive", "uncertainty_gate": False,
                "checkpoint": "levelset_n600_v2_attrclean l7-best", "n_pairs": 600},
        predicted_output={"delta_dseg": _NAIVE_DELTA},
        empirical_output={"delta_dseg": _NAIVE_DELTA, "mechanism": "dash-gap + fit-error FALSE-POSITIVES"},
        residual=0.0,
        source_artifact=report,
        measurement_method="n600_through_R_frozen_cpu_torch_segnet_argmax",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria="re-measure through R after a fine-tune with --lane-render-band active",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_uncert = EmpiricalAnchor(
        anchor_id="analytic_lane_band_witness_uncertainty_break_even_n600_20260701",
        measurement_utc="2026-07-01T00:00:00Z",
        inputs={"condition": "c_full_wit", "uncertainty_source": "witness_softmax_margin",
                "tau_prob": 0.85, "eps": 0.35, "dash_forward_max_m": 55.0, "n_pairs": 600},
        predicted_output={"delta_dseg": predict_composite_delta_dseg(
            _NAIVE_DELTA, _FP_KILL_FRACTION, 0.0)},
        empirical_output={"delta_dseg": _WIT_UNCERT_DELTA,
                          "note": "~break-even POST-HOC; NET-NEGATIVE requires training-in"},
        residual=abs(predict_composite_delta_dseg(_NAIVE_DELTA, _FP_KILL_FRACTION, 0.0)
                     - 100.0 * _WIT_UNCERT_DELTA),
        source_artifact=report,
        measurement_method="n600_through_R_frozen_cpu_torch_segnet_argmax",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria="fine-tune with --lane-render-band active -> re-measure d_seg through R",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Analytic-lane render-band false-positive reduction via witness-uncertainty gate",
        one_line_summary=(
            "The witness-uncertainty mask kills the naive analytic-lane-band FP "
            "(+0.000622 -> +0.000012 break-even, kill 0.98); net-negative needs training-in."
        ),
        latex_form=(
            r"\Delta d_{seg}^{compose} = 100\,\big(\text{fp}_{naive}(1-\kappa) - \text{fn}_{rec}\big),"
            r"\ \kappa = 1 - \Delta_{gate}/\Delta_{naive}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.analytic_lane_render_band_fp_reduction:predict_composite_delta_dseg"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory"],
            "post_hoc_only": True,
            "note": "post-hoc render-side; the trained-in win is a SEPARATE (unmeasured) regime",
        },
        units_in={
            "fp_added_naive": "fraction_of_all_pixels",
            "fp_kill_fraction": "dimensionless_0_to_1",
            "fn_recovered": "fraction_of_all_pixels",
        },
        units_out={"delta_dseg": "score_units_100x_dseg"},
        empirical_anchors=(anchor_naive, anchor_uncert),
        predicted_vs_empirical_residual={
            "n600_through_R_frozen_cpu_torch_segnet_argmax": abs(
                predict_composite_delta_dseg(_NAIVE_DELTA, _FP_KILL_FRACTION, 0.0)
                - 100.0 * _WIT_UNCERT_DELTA),
        },
        last_calibration_utc="2026-07-01T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.analytic_lane_render_band",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=(
            "tools/levelset_analytic_lane_band_dseg_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="analytic_lane_render_band_fp_reduction.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "predict_composite_delta_dseg",
    "build_analytic_lane_render_band_fp_reduction_v1",
]
