# SPDX-License-Identifier: MIT
"""Canonical equation: lane-band ego-factorization source re-parameterization (Wave-F unified-xi).

The Wave-F Stage-1 RD codec plateaus at the delta-stream Shannon floor (26,179 B, rate 0.01743)
because it fits each of the 600 pairs INDEPENDENTLY in the CAMERA frame. The measured
per-dim-type floor decomposes EXACTLY (arithmetic, residual 0.0) into two INFORMATION-bound
components:

  * EGO-SWEPT CENTERLINE CURVATURE ~= 11,268 B (c3+c2+c1+c0): the lane geometry in the camera
    frame changes every frame because the ego drives forward (the lane sweeps toward the car).
  * PER-FRAME FIT JITTER ~= 14,911 B (hw0/hw1 + dash*3 + fwd0/fwd1): each pair is fit
    INDEPENDENTLY from the SegNet argmax, so the "same" world lane refits slightly differently.

The DERIVED next lever is a SOURCE RE-PARAMETERIZATION, not a better coder: transform all 600
frames' class-1 pixels to the STATIC world frame via ONE metric ego-trajectory xi_ego(t)
(tac.lie.se3_bspline, ~O(10-48) DOF for the whole drive), fit the world lane geometry ONCE +
code the tiny per-frame innovation. This collapses BOTH the ego sweep AND the fit jitter =>
DERIVED target ~1-4 KB (~0.001 rate). Scorer-free (pure geometry).

THE UNIFICATION (Morse-Smale dual-axis advection): the argmax partition is a Morse-Smale
complex (separatrices = class boundaries; lanes = class-1 separatrices); xi_ego(t) is the
VECTOR FIELD that ADVECTS the whole complex frame-to-frame (xi-pushforward via the ground
homography). So the ONE stored xi is TRIPLE-USE and DUAL-AXIS: (i) IS the pose (warp real
keyframe luma -> d_pose; warp_real_luma_frame0), (ii) advects the lanes to the static world
frame (L1: lane RATE), (iii) partition temporal-consistency. It REPLACES the Quantizr 6-vector
FiLM pose sidecar (that 6-vector store is the FALLBACK). One SE(3) B-spline xi, both scorers --
its marginal rate lowers d_seg AND d_pose. This is Wyner-Ziv side-information / motion-
compensated prediction: the ego-pose is the side-info BOTH scorers share.

    bytes_reparam = static_world_frame_bytes + ego_trajectory_bytes  (DERIVED << 26179 floor)

CORRECTION (R2 audit, honored): the stored #206 pose sidecar holds the PoseNet 6-vector
OUTPUTS, NOT a metric se(3) twist usable to warp lanes; the metric ego-motion must be either
DERIVED-free-bit-exact-at-decode OR STORED-counted -- rule-118 accounting MEASURED both ways,
NEVER asserted free. The ~1-4 KB target and the ~0.001 rate are DERIVED [prediction] pending
the Stage-2 build + its bit-exact-inverse determinism gate.

Consumers: the Stage-2 byte-close ego-warp codec + the DSL WarpGauge/PoseGauge + warp_real_luma.
Producers: the Wave-F design memo + the wave_f n600 measured per-dim floor breakdown.
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

EQUATION_ID = "lane_band_ego_factorization_source_reparam_v1"

# MEASURED per-dim-type floor breakdown (n600; the decomposition is EXACT arithmetic).
_EGO_SWEPT_CENTERLINE_BYTES = 11_268   # c3+c2+c1+c0 (ego-swept curvature)
_FIT_JITTER_BYTES = 14_911             # hw0/hw1 + dash*3 + fwd0/fwd1 (per-frame independent fit)
_SHANNON_FLOOR_BYTES = 26_179          # ego + jitter == the delta-stream Shannon floor (exact)
# DERIVED [prediction] target after the SE(3) ego-factorization source re-param.
_REPARAM_TARGET_BYTES_LO = 1_000
_REPARAM_TARGET_BYTES_HI = 4_000


def predict_source_reparam_bytes(static_world_frame_bytes: float,
                                 ego_trajectory_bytes: float) -> float:
    """DERIVED post-re-parameterization byte cost = static world-frame geometry (fit ONCE)
    + the compact metric ego-trajectory (xi_ego(t)). Collapses the camera-frame ego-sweep
    (~11268 B) + per-frame fit jitter (~14911 B) that sum EXACTLY to the 26179 B floor."""

    return float(static_world_frame_bytes) + float(ego_trajectory_bytes)


def build_lane_band_ego_factorization_source_reparam_v1() -> CanonicalEquation:
    """Build the Wave-F unified-xi ego-factorization source-reparam canonical equation."""

    design_memo = ".omx/research/wave_f_optimal_lane_band_rd_code_design_20260702.md"
    report = ".omx/research/wave_f_lane_band_rd_rate_n600_measured.json"

    anchor_decomp = EmpiricalAnchor(
        anchor_id="lane_band_floor_decomposition_ego_plus_jitter_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"quantity": "delta_stream_shannon_floor_decomposition", "n_pairs": 600,
                "decomposition": "exact_arithmetic"},
        predicted_output={"floor_bytes": _EGO_SWEPT_CENTERLINE_BYTES + _FIT_JITTER_BYTES},
        empirical_output={"ego_swept_centerline_bytes": _EGO_SWEPT_CENTERLINE_BYTES,
                          "fit_jitter_bytes": _FIT_JITTER_BYTES,
                          "sum_bytes": _SHANNON_FLOOR_BYTES,
                          "note": "ego 11268 + jitter 14911 == floor 26179 EXACTLY"},
        residual=abs((_EGO_SWEPT_CENTERLINE_BYTES + _FIT_JITTER_BYTES) - _SHANNON_FLOOR_BYTES),
        source_artifact=report,
        measurement_method="n600_real_gt_per_dim_type_shannon_floor",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria="re-measure the per-dim floor if the lane parameterization changes",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_target = EmpiricalAnchor(
        anchor_id="lane_band_ego_factorization_derived_target_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"lever": "L1_SE3_bspline_ego_factorization", "engine": "tac.lie.se3_bspline",
                "unified_xi": "triple_use_dual_axis (pose + lane advection + temporal consistency)"},
        predicted_output={"reparam_bytes_lo": _REPARAM_TARGET_BYTES_LO,
                          "reparam_bytes_hi": _REPARAM_TARGET_BYTES_HI,
                          "rate_term_target": 25.0 * _REPARAM_TARGET_BYTES_HI / 37_545_489},
        empirical_output={"note": "DERIVED [prediction]; not yet measured through the Stage-2 build"},
        residual=0.0,
        source_artifact=design_memo,
        measurement_method="derived_prediction_pending_stage2_byte_close",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=design_memo,
            reactivation_criteria="Stage-2 SE(3) ego-warp byte-close: bit-exact-inverse gate + MEASURE reparam bytes at n600",
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Lane-band ego-factorization source re-parameterization (unified xi, Morse-Smale dual-axis)",
        one_line_summary=(
            "The 26179 B lane floor = ego-sweep 11268 + fit-jitter 14911 (exact); ONE SE(3) xi "
            "re-param (triple-use, dual-axis) DERIVED-collapses both to ~1-4 KB (~0.001)."
        ),
        latex_form=(
            r"b_{floor} = b_{ego} + b_{jitter} = 11268 + 14911 = 26179;\ "
            r"b_{reparam} = b_{static} + b_{\xi} \ll b_{floor}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_ego_factorization_source_reparam:predict_source_reparam_bytes"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band", "softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "derived": True,
            "note": "the DECOMPOSITION is measured-exact; the ~1-4 KB target is DERIVED pending Stage-2 byte-close",
        },
        units_in={
            "static_world_frame_bytes": "bytes",
            "ego_trajectory_bytes": "bytes",
        },
        units_out={"reparam_bytes": "bytes"},
        empirical_anchors=(anchor_decomp, anchor_target),
        predicted_vs_empirical_residual={
            "n600_real_gt_per_dim_type_shannon_floor": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/levelset_byte_close_and_eval.py",
            "tac.witness_dsl.gauge",
            "tac.boundary_math.warp_real_luma_frame0",
        ),
        canonical_producers=(
            "tools/wave_f_lane_band_rd_rate_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="lane_band_ego_factorization_source_reparam.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "predict_source_reparam_bytes",
    "build_lane_band_ego_factorization_source_reparam_v1",
]
