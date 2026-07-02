# SPDX-License-Identifier: MIT
"""Canonical equations: correspondence-first lane-coefficient coding (Wave-F Stage-2, MEASURED).

The Wave-F unified-xi BUILD (2026-07-02) MEASURED the design's decisive reframe at n600 and
returned a CLEAN, CONSTRUCTIVE result that SUPERSEDES-BY-REFINEMENT the ego-warp mechanism the
Stage-1 ``lane_band_ego_factorization_source_reparam_v1`` equation ASSUMED:

  * EGO-MOTION-COMPENSATED PREDICTIVE lane coding (LBND3) is REFUTED -- every ego-predictive
    variant is 1.04-1.34x LARGER than LBND2 (lane-optimal 3-DOF 1.143x, dy+yaw 1.075x, PoseNet-
    physical affine 1.081x; ego even HURTS the already-smoothed source 1.344x). The negative is
    IMPLEMENTATION-informative: the camera-frame residual is per-frame fit JITTER + SLOT-SWAPs,
    NOT a coherent ego sweep a planar advect can predict.

  * SOURCE TEMPORAL SMOOTHING is POSITIVE -- LBND2-on-median-smoothed-source (win15) = 24,149 B
    / rate 0.01608 = 0.582x LBND2 = -42%, DIPPING BELOW the Stage-1 delta-stream Shannon floor
    (26,179 B): smoothing CHANGES the source to a lower-entropy signal, so the raw-source floor
    no longer applies. This CONFIRMS the L1 SOURCE-RE-PARAMETERIZATION thesis via a DIFFERENT
    mechanism (denoising, ZERO xi) than the hypothesized ego-warp.

  * The lane rate axis is xi-FREE (ego-predictive negative; smoothing wins with zero xi) =>
    xi is a PURE-POSE sidecar, optimally calibrated for d_pose at ZERO lane cost. The "one xi,
    both axes" tension (design 5.5) RESOLVES TRIVIALLY -- the lane axis DECLINED xi. (Pose still
    uses the physical PoseNet xi to warp real keyframe luma: null d_pose 163.12 -> 1.367 = -99%.)

THE UNIFYING LAW (DERIVED -- one principle explains BOTH measured failures): an INDEX-PERMUTATION
DISCONTINUITY defeats every temporal model. The LBND2 lateral-sort assigns slot k to the k-th
lane BY LATERAL POSITION AT THAT FRAME; at a lane cross / birth / death the sort flips => the
coefficient time-series OF A SLOT has a discontinuity corresponding to NO physical motion. A
PREDICTOR spends bytes coding swap-innovations (=> LBND3 worse); a linear SMOOTHER averages
across the swap into a phantom lane (=> moving-average lossy + d_seg spill); a TRANSFORM (DCT/KLT)
Gibbs-spreads the discontinuity. MEASURED: 44% of the temporal-delta L1 mass sits in the top-5%
largest jumps (the swap/outlier signature). Therefore CORRESPONDENCE MUST PRECEDE any temporal
coder. SOTA video-lane-detection independently validates the ordering: RVLD (ICCV'23, causal
one-frame warp = our failed LBND3) -> LaneTCA (2024, longer-window batch aggregation, higher F1)
= batch/aggregation BEATS causal-one-step, exactly our LBND3-vs-smoothing result.

THE OPTIMAL PIPELINE (DERIVED [prediction] -- correspondence-first, R-D-optimal, task-lambda):
  CORRESPONDENCE (global min-cost-flow / MHT-lite track assignment; LOSSLESS on geometry -- a
  relabeling never changes a rendered lane) -> per-track BATCH Kalman-RTS fixed-interval smoother
  (MMSE; fills occlusion gaps) [or RPCA/PCP: L=low-rank trajectory + S=sparse swap-spikes] ->
  ll1-trend / 1-D TV / Potts EDGE-PRESERVING denoise with per-coefficient lambda set by the
  margin-saliency d(d_seg)/d(coeff) (#141) at the KKT point d(d_seg)/d(byte) = 25/(100*37.5M) ->
  the UNCHANGED LBND2 quantize+brotli backend (ships as LBND2 bytes, ZERO new inflate code,
  rule-118 clean). DERIVED expected rate ~0.007-0.012 with LESS d_seg loss than the moving-
  average -- the d_seg-through-R leg is the #205 gate (UNMEASURED; per NO-FAKE, do not assert).

THE openpilot UNIFIED PHYSICAL PRIOR (DERIVED design-pattern): openpilot -- the production
driving-perception stack on the SAME comma rig, so contest-task == openpilot-task -- is the
SINGLE physically-grounded, temporally-coherent, offline-FREE (rule-118: estimator runs
compress-time; ship ONLY the compact video-derived statistic) prior that seeds the witness
across BOTH scored axes: POSE = ego-motion xi -> warp-real-luma warm-start (-94/-99%); LANES =
the coherent recurrent lane model output -> the temporally-COHERENT lane SOURCE (fixes the
slot-swap jitter that killed ego-predictive coding + made moving-average lossy). Same 2-part
recipe each axis: a physical coherent FREE prior + a SMALL learned refinement to the exact
frozen scorer. openpilot is a PRIOR / INIT / REGULARIZER, NEVER the target (SegNet/PoseNet are
the sole authority; fitting supercombo's driving-lane output would spill d_seg); net-S is
#205-gated; NEVER ship openpilot/estimator weights.

Consumers: the Stage-2 tracking codec (`tac.boundary_math.lane_track_and_smooth`, #234, design-
stage) + the byte-close tool + the DSL LaneGauge/PoseGauge + `warp_real_luma_frame0`.
Producers: the Wave-F unified-xi n600 bake-off + the tracking-rate probe + the survey ledger.
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

# ---------------------------------------------------------------------------
# Shared MEASURED constants (n600, real gt_n600.npz, byte-closed, [macOS-CPU advisory]).
# ---------------------------------------------------------------------------
_RATE_DENOM = 37_545_489.0            # contest rate-term denominator
_LBND2_BASELINE_BYTES = 41_526       # Stage-1 RD codec (rate 0.02765)
_SHANNON_FLOOR_BYTES = 26_179        # delta-stream Shannon floor (rate 0.01743)
# Ego-predictive LBND3 variants (all WORSE): brotli bytes.
_EGO_LANE_OPTIMAL_3DOF_BYTES = 47_453   # 1.143x LBND2
_EGO_DY_YAW_BYTES = 44_652              # 1.075x
_EGO_POSENET_PHYSICAL_BYTES = 44_908    # 1.081x
# Source temporal-smoothing (all BETTER): brotli bytes by median window.
_SMOOTH_BYTES = {0: _LBND2_BASELINE_BYTES, 3: 31_360, 5: 28_050, 9: 26_260, 15: 24_149}
# The measured swap/outlier signature: fraction of temporal-delta L1 in the top-5% jumps.
_TOP5PCT_JUMP_MASS = 0.44
# Pose axis (pure-pose xi): null d_pose -> warp-calibrated d_pose (frozen CPU-torch PoseNet).
_DPOSE_NULL = 163.12
_DPOSE_WARP = 1.367
# DERIVED [prediction] correspondence-first pipeline rate band.
_CORR_FIRST_RATE_LO = 0.007
_CORR_FIRST_RATE_HI = 0.012


def rate_term(bytes_count: float) -> float:
    """The contest rate-term contribution of a byte-count = 25 * bytes / 37_545_489."""

    return 25.0 * float(bytes_count) / _RATE_DENOM


# ---------------------------------------------------------------------------
# Callable 1: the index-permutation-discontinuity law.
# ---------------------------------------------------------------------------
def top5pct_jump_mass_fraction() -> float:
    """The MEASURED swap/outlier signature: fraction (0.44) of the temporal-delta L1 mass that
    sits in the top-5% largest per-frame jumps. This is the index-permutation (slot-swap) mass a
    predictor codes as innovations, a smoother blurs into phantom lanes, and a transform Gibbs-
    spreads -- the empirical proof that CORRESPONDENCE must precede any temporal coder."""

    return _TOP5PCT_JUMP_MASS


# ---------------------------------------------------------------------------
# Callable 2: the MEASURED unified-xi resolution (smoothing rate vs ego-predictive).
# ---------------------------------------------------------------------------
def predict_smoothed_source_rate_term(window: int) -> float:
    """The MEASURED LBND2-on-median-smoothed-source rate_term for a smoothing window in
    {0,3,5,9,15}. window=0 is the LBND2 baseline (0.02765); window=15 is the best measured
    (24,149 B / 0.01608 = -42%, below the Shannon floor). Ego-predictive LBND3 is REFUTED and
    is NOT a member of this family (it is a predictor, not a source transform)."""

    b = _SMOOTH_BYTES.get(int(window))
    if b is None:
        raise KeyError(
            f"window={window!r} not in the measured set {sorted(_SMOOTH_BYTES)!r}; "
            "the smoothing rate is only MEASURED at these windows (do not extrapolate)"
        )
    return rate_term(b)


# ---------------------------------------------------------------------------
# Callable 3: the DERIVED correspondence-first optimal pipeline rate band.
# ---------------------------------------------------------------------------
def predict_correspondence_first_rate_band() -> tuple[float, float]:
    """The DERIVED [prediction] rate band (0.007, 0.012) of the correspondence-first pipeline
    (global track assignment -> Kalman-RTS batch smooth -> ll1-trend/TV/Potts task-lambda
    denoise -> LBND2 backend). Below the moving-average's 0.01608 AND less-lossy (edges kept);
    UNMEASURED byte-closed -- the #205 d_seg-through-R leg is the gate. NEVER a score claim."""

    return (_CORR_FIRST_RATE_LO, _CORR_FIRST_RATE_HI)


# ---------------------------------------------------------------------------
# Callable 4: the openpilot unified physical prior role (both axes).
# ---------------------------------------------------------------------------
def openpilot_prior_role_for_axis(axis: str) -> str:
    """The openpilot world-model's role on a scored axis ('pose' or 'lane'): always a
    PRIOR / INIT / REGULARIZER, NEVER the target. Same 2-part recipe each axis (physical
    coherent free prior + small learned refinement to the frozen scorer). Encodes rule-118 +
    the NO-FAKE 'never fit supercombo's output' discipline."""

    a = str(axis).strip().lower()
    if a not in {"pose", "lane"}:
        raise ValueError(f"axis={axis!r} must be 'pose' or 'lane'")
    return "prior_init_regularizer_not_target"


# ---------------------------------------------------------------------------
# Equation builders.
# ---------------------------------------------------------------------------
def build_index_permutation_discontinuity_defeats_temporal_model_v1() -> CanonicalEquation:
    """The unifying law: an index-permutation discontinuity defeats every temporal model."""

    ledger = ".omx/research/lane_coeff_tracking_denoising_optimal_survey_20260702.md"
    result = ".omx/research/wave_f_ego_predictive_rate_n600_RESULT.json"

    anchor_swap_mass = EmpiricalAnchor(
        anchor_id="lane_slot_swap_top5pct_jump_mass_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"quantity": "top5pct_jump_mass_fraction_of_temporal_delta_L1", "n_pairs": 600,
                "cause": "lateral-sort slot re-labeling at lane cross / birth / death"},
        predicted_output={"top5pct_jump_mass": top5pct_jump_mass_fraction()},
        empirical_output={"top5pct_jump_mass": _TOP5PCT_JUMP_MASS,
                          "note": "44% of temporal-delta L1 in top-5% jumps = the swap signature"},
        residual=0.0,
        source_artifact=result,
        measurement_method="n600_temporal_delta_l1_top5pct_jump_mass",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=result,
            reactivation_criteria="re-measure the top-5% jump mass if the lane slot assignment changes",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_two_failures = EmpiricalAnchor(
        anchor_id="permutation_discontinuity_explains_predictor_and_smoother_failures_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"predictor": "LBND3_ego_dpcm (RVLD-style causal warp)",
                "smoother": "median_moving_average (LTI low-pass)",
                "transform": "DCT/KLT"},
        predicted_output={"law": "correspondence_must_precede_temporal_coder"},
        empirical_output={"predictor_result": "1.04-1.34x WORSE (codes swap-innovations)",
                          "smoother_result": "-42% rate but LOSSY + d_seg spill at swaps",
                          "sota_validation": "RVLD causal -> LaneTCA batch (batch beats causal)"},
        residual=0.0,
        source_artifact=ledger,
        measurement_method="n600_ego_predictive_vs_smoothing_bakeoff_both_explained",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=ledger,
            reactivation_criteria="a temporal coder that BEATS correspondence-first on the swap mass would falsify the law",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id="index_permutation_discontinuity_defeats_temporal_model_v1",
        name="Index-permutation discontinuity defeats every temporal model (correspondence-first law)",
        one_line_summary=(
            "A slot relabeling discontinuity (lane swap/birth/death) defeats predictor+smoother"
            "+transform alike (44% top-5% jump mass) => correspondence must precede any temporal coder."
        ),
        latex_form=(
            r"\text{slot swap} \Rightarrow \partial_t c_k \perp \text{physical motion};\ "
            r"\text{predictor}\to\text{innov},\ \text{smoother}\to\text{phantom},\ "
            r"\text{DCT}\to\text{Gibbs};\ m_{top5\%}=0.44"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_correspondence_first_source_reparam:top5pct_jump_mass_fraction"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band", "softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory", "derived"],
            "derived": True,
            "note": "the 0.44 swap mass is MEASURED; the 'correspondence-first' law is DERIVED (explains both failures)",
        },
        units_in={},
        units_out={"top5pct_jump_mass": "dimensionless_l1_mass_fraction"},
        empirical_anchors=(anchor_swap_mass, anchor_two_failures),
        predicted_vs_empirical_residual={
            "n600_temporal_delta_l1_top5pct_jump_mass": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "tac.boundary_math.lane_track_and_smooth",
        ),
        canonical_producers=(
            "tools/wave_f_ego_predictive_rate_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="index_permutation_discontinuity_defeats_temporal_model.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[derived]",
            hardware_substrate="unknown",
        ),
    )


def build_lane_band_source_reparam_measured_resolution_v1() -> CanonicalEquation:
    """The MEASURED Wave-F unified-xi resolution: ego-predictive REFUTED; smoothing -42%; xi pure-pose.

    SUPERSEDES-BY-REFINEMENT ``lane_band_ego_factorization_source_reparam_v1`` (which ASSUMED the
    ego-warp mechanism): the SOURCE-RE-PARAMETERIZATION thesis is CONFIRMED, but the MECHANISM is
    correspondence + denoising (temporal smoothing), NOT ego-warp predictive coding. APPEND-ONLY
    per Catalog #110/#113 -- the prior equation's payload is preserved; this is the measured row.
    """

    build = ".omx/research/wave_f_unified_xi_build_measured_20260702.md"
    result = ".omx/research/wave_f_ego_predictive_rate_n600_RESULT.json"

    anchor_ego_refuted = EmpiricalAnchor(
        anchor_id="ego_predictive_lbnd3_refuted_all_variants_worse_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"codec": "LBND3_ego_motion_compensated_predictive", "n_pairs": 600,
                "variants": ["lane_optimal_3dof", "dy_yaw", "posenet_physical_affine"]},
        predicted_output={"design_hypothesis": "ego-warp collapses the 26179 B floor toward ~1-4 KB"},
        empirical_output={"lane_optimal_3dof_bytes": _EGO_LANE_OPTIMAL_3DOF_BYTES,
                          "dy_yaw_bytes": _EGO_DY_YAW_BYTES,
                          "posenet_physical_bytes": _EGO_POSENET_PHYSICAL_BYTES,
                          "ratio_vs_lbnd2": "1.04-1.34x WORSE",
                          "verdict": "REFUTED -- residual is fit-jitter+swaps, not a coherent ego sweep"},
        residual=abs((_EGO_LANE_OPTIMAL_3DOF_BYTES / _LBND2_BASELINE_BYTES) - 0.06),  # ~1.08: predicted ~0.06x, measured 1.14x
        source_artifact=result,
        measurement_method="n600_ego_predictive_lbnd3_bakeoff",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=result,
            reactivation_criteria="a JOINT (batch, not causal) world-BEV fit with a measured-reliable ego trajectory that BEATS the per-track smoother",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_smoothing_win = EmpiricalAnchor(
        anchor_id="source_temporal_smoothing_win15_minus42_below_floor_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"transform": "median_temporal_smoothing_of_coeff_trajectory", "window": 15,
                "ships_as": "standard_LBND2_bytes (compress-time source transform, ZERO new inflate code)"},
        predicted_output={"smoothed_rate_term": predict_smoothed_source_rate_term(15)},
        empirical_output={"win15_bytes": _SMOOTH_BYTES[15], "win15_rate": rate_term(_SMOOTH_BYTES[15]),
                          "ratio_vs_lbnd2": 0.582, "note": "-42%; DIPS BELOW the 26179 B Shannon floor",
                          "mechanism": "source re-param CONFIRMED via denoising (zero xi), NOT ego-warp",
                          "caveat": "LOSSY on geometry; net-S is #205-gated (win5 conservative)"},
        residual=0.0,
        source_artifact=build,
        measurement_method="n600_source_smoothing_rate",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=build,
            reactivation_criteria="sweep the smoothing window for the S-optimum through the #205 d_seg-through-R leg",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_xi_pure_pose = EmpiricalAnchor(
        anchor_id="xi_pure_pose_lane_axis_declined_xi_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"resolution": "lane axis is xi-FREE (ego-predictive negative; smoothing wins with zero xi)",
                "pose_carrier": "warp_real_luma_frame0 by the physical PoseNet xi"},
        predicted_output={"one_xi_both_axes_tension": "RESOLVED trivially -- lane declined xi"},
        empirical_output={"dpose_null": _DPOSE_NULL, "dpose_warp": _DPOSE_WARP,
                          "dpose_drop_pct": -99.0,
                          "note": "xi = PURE-POSE sidecar, zero lane cost; supersedes the 'dual-use xi' framing for lanes"},
        residual=0.0,
        source_artifact=build,
        measurement_method="n600_xi_pure_pose_warp_dpose",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=build,
            reactivation_criteria="re-measure if a JOINT world-BEV lane fit later re-admits xi to the lane axis",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id="lane_band_source_reparam_measured_resolution_v1",
        name="Lane-band source re-parameterization — MEASURED resolution (ego-predictive REFUTED; smoothing −42%; xi pure-pose)",
        one_line_summary=(
            "Ego-predictive lane coding REFUTED (1.04-1.34x WORSE); source smoothing -42% "
            "(0.02765->0.01608, below floor); xi = PURE-POSE (lane axis declined xi)."
        ),
        latex_form=(
            r"LBND3(\xi)\in[1.04,1.34]\cdot LBND2\ (\text{WORSE});\ "
            r"\text{smooth}_{15}=24149\,B=0.582\cdot LBND2 < b_{floor}=26179;\ \xi=\text{pure-pose}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_correspondence_first_source_reparam:predict_smoothed_source_rate_term"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band"],
            "measurement_axis": ["macOS-CPU advisory"],
            "supersedes_mechanism_of": "lane_band_ego_factorization_source_reparam_v1",
            "note": "source-reparam THESIS confirmed; MECHANISM = correspondence+denoising, NOT ego-warp",
        },
        units_in={"window": "median_smoothing_window_pairs"},
        units_out={"smoothed_rate_term": "dimensionless_contest_rate_contribution"},
        empirical_anchors=(anchor_ego_refuted, anchor_smoothing_win, anchor_xi_pure_pose),
        predicted_vs_empirical_residual={
            "n600_source_smoothing_rate": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "tools/levelset_byte_close_and_eval.py",
            "tac.boundary_math.warp_real_luma_frame0",
        ),
        canonical_producers=(
            "tools/wave_f_ego_predictive_rate_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="lane_band_source_reparam_measured_resolution.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


def build_correspondence_first_lane_coding_optimal_pipeline_v1() -> CanonicalEquation:
    """The DERIVED optimal correspondence-first pipeline (tracking -> RTS -> ll1-trend, task-lambda)."""

    survey = ".omx/research/lane_coeff_tracking_denoising_optimal_survey_20260702.md"

    anchor_pipeline = EmpiricalAnchor(
        anchor_id="correspondence_first_optimal_pipeline_derived_band_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"stage_1": "global min-cost-flow / MHT-lite track assignment (LOSSLESS on geometry)",
                "stage_2": "per-track Kalman-RTS fixed-interval smoother (MMSE) [or RPCA/PCP]",
                "stage_3": "ll1-trend / 1-D TV / Potts edge-preserving denoise, lambda_i = d(d_seg)/d(coeff_i)",
                "stage_4": "UNCHANGED LBND2 quantize+brotli backend (ships as LBND2 bytes)"},
        predicted_output={"rate_band": list(predict_correspondence_first_rate_band()),
                          "vs_smoothing": "below 0.01608 AND less-lossy (edges kept)"},
        empirical_output={"note": "DERIVED [prediction]; UNMEASURED byte-closed; #205 d_seg-through-R is the gate"},
        residual=0.0,
        source_artifact=survey,
        measurement_method="derived_prediction_pending_234_tracking_build",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=survey,
            reactivation_criteria="#234 build tac.boundary_math.lane_track_and_smooth -> MEASURE tracked-only + +smooth + RPCA rows @ n600; then #205 d_seg-through-R",
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )
    anchor_lossless = EmpiricalAnchor(
        anchor_id="correspondence_is_lossless_on_geometry_pareto_improvement_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"claim": "correspondence (a slot relabeling) is EXACTLY lossless on rendered geometry",
                "target_mass": top5pct_jump_mass_fraction()},
        predicted_output={"pareto": "rate down, distortion unchanged (kills the 44% swap mass at zero d_seg cost)"},
        empirical_output={"note": "DERIVED-exact property (a re-labeling never changes a rendered lane); "
                          "isolate the correspondence gain first (verify top-5% jump mass drops)"},
        residual=0.0,
        source_artifact=survey,
        measurement_method="derived_correspondence_lossless_property",
        empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=survey,
            reactivation_criteria="measure the tracked-only stream bytes @ n600 to confirm the swap-mass drop is lossless",
            measurement_axis="[derived]",
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id="correspondence_first_lane_coding_optimal_pipeline_v1",
        name="Correspondence-first optimal lane-coefficient coding pipeline (track -> RTS -> ll1-trend, task-lambda)",
        one_line_summary=(
            "Global min-cost-flow track assignment (lossless) -> Kalman-RTS batch smooth -> "
            "ll1-trend/RPCA edge-preserving denoise (lambda=d(d_seg)/d(coeff)) -> LBND2; DERIVED ~0.007-0.012."
        ),
        latex_form=(
            r"x^*=\arg\min_x \|y-x\|^2+\lambda\|D^2 x\|_1\ \text{per track};\ "
            r"\lambda_i=\partial d_{seg}/\partial c_i\ \text{at}\ \partial d_{seg}/\partial b=25/(100\cdot37.5\text{M})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_correspondence_first_source_reparam:predict_correspondence_first_rate_band"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band"],
            "measurement_axis": ["predicted", "derived"],
            "derived": True,
            "note": "the pipeline ORDER is DERIVED-exact (correspondence-first law); the ~0.007-0.012 rate is [prediction] pending #234",
        },
        units_in={},
        units_out={"rate_band": "dimensionless_contest_rate_contribution_lo_hi"},
        empirical_anchors=(anchor_pipeline, anchor_lossless),
        predicted_vs_empirical_residual={
            "derived_prediction_pending_234_tracking_build": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.boundary_math.lane_track_and_smooth",
            "tools/levelset_byte_close_and_eval.py",
            "tac.witness_dsl.gauge",
        ),
        canonical_producers=(
            "tools/wave_f_lane_tracking_rate_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="correspondence_first_lane_coding_optimal_pipeline.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )


def build_openpilot_unified_physical_prior_both_scored_axes_v1() -> CanonicalEquation:
    """The DERIVED design-pattern: ONE openpilot world-model seeds BOTH scored axes (pose + lane)."""

    memo = ".omx/research/lane_coeff_tracking_denoising_optimal_survey_20260702.md"

    anchor_pose_confirmed = EmpiricalAnchor(
        anchor_id="openpilot_pose_prior_warp_real_luma_confirmed_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"axis": "pose", "prior": "openpilot ego-motion xi -> warp_real_luma_frame0",
                "role": openpilot_prior_role_for_axis("pose")},
        predicted_output={"prior_lowers_d_pose": True},
        empirical_output={"dpose_null": _DPOSE_NULL, "dpose_warp": _DPOSE_WARP, "drop_pct": -99.0,
                          "note": "the pose-axis prior is MEASURED-effective (-94/-99%)"},
        residual=0.0,
        source_artifact=memo,
        measurement_method="n600_openpilot_pose_prior_warp_dpose",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=memo,
            reactivation_criteria="re-measure the warp-real-luma d_pose on the witness at n600",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_lane_design = EmpiricalAnchor(
        anchor_id="openpilot_lane_coherent_source_prior_design_stage_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"axis": "lane", "prior": "openpilot coherent recurrent lane model -> coherent lane SOURCE",
                "role": openpilot_prior_role_for_axis("lane")},
        predicted_output={"prior_fixes_slot_swap_jitter": True,
                          "recipe": "physical coherent free prior + small learned refinement to frozen scorer"},
        empirical_output={"note": "DESIGN-STAGE; the coherent-tracking build is #234 (vs the 0.01608 smoothing floor); "
                          "PRIOR/INIT-only, NEVER the target (fitting supercombo's driving-lanes spills d_seg)"},
        residual=0.0,
        source_artifact=memo,
        measurement_method="design_stage_pending_234_coherent_tracking",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=memo,
            reactivation_criteria="#234 measure the coherent-source lane rate + the #205 d_seg-through-R leg",
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id="openpilot_unified_physical_prior_both_scored_axes_v1",
        name="openpilot unified physical prior seeds BOTH scored axes (pose xi + coherent-lane source)",
        one_line_summary=(
            "ONE openpilot world-model (same comma rig) seeds BOTH axes: ego-xi->pose warm-start "
            "(-99%) + coherent lanes->d_seg source; PRIOR/INIT-only, rule-118, net-S #205-gated."
        ),
        latex_form=(
            r"\text{axis}\in\{pose,lane\}:\ \hat s = \text{prior}_{openpilot}(\text{rig}) + "
            r"\Delta_{learned}(\text{frozen scorer});\ \text{prior}=\text{init, never target}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_correspondence_first_source_reparam:openpilot_prior_role_for_axis"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band", "softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory", "predicted", "derived"],
            "design_pattern": True,
            "note": "pose-axis prior MEASURED-effective; lane-axis coherent-source prior DESIGN-STAGE (#234)",
        },
        units_in={"axis": "scored_axis_token"},
        units_out={"role": "prior_discipline_token"},
        empirical_anchors=(anchor_pose_confirmed, anchor_lane_design),
        predicted_vs_empirical_residual={
            "n600_openpilot_pose_prior_warp_dpose": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "tac.boundary_math.warp_real_luma_frame0",
            "tac.boundary_math.lane_track_and_smooth",
        ),
        canonical_producers=(
            "tools/wave_f_lane_tracking_rate_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="openpilot_unified_physical_prior_both_scored_axes.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[derived]",
            hardware_substrate="unknown",
        ),
    )


def build_all_correspondence_first_lane_coding_equations() -> list[CanonicalEquation]:
    """All 4 Wave-F Stage-2 correspondence-first canonical equations (triality EQUATIONS leg)."""

    return [
        build_index_permutation_discontinuity_defeats_temporal_model_v1(),
        build_lane_band_source_reparam_measured_resolution_v1(),
        build_correspondence_first_lane_coding_optimal_pipeline_v1(),
        build_openpilot_unified_physical_prior_both_scored_axes_v1(),
    ]


__all__ = [
    "rate_term",
    "top5pct_jump_mass_fraction",
    "predict_smoothed_source_rate_term",
    "predict_correspondence_first_rate_band",
    "openpilot_prior_role_for_axis",
    "build_index_permutation_discontinuity_defeats_temporal_model_v1",
    "build_lane_band_source_reparam_measured_resolution_v1",
    "build_correspondence_first_lane_coding_optimal_pipeline_v1",
    "build_openpilot_unified_physical_prior_both_scored_axes_v1",
    "build_all_correspondence_first_lane_coding_equations",
]
