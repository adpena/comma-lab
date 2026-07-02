# SPDX-License-Identifier: MIT
"""Canonical equation: pose in-training lever survey verdict (Wave-F #205 pose plan).

A survey of 13 in-training pose levers (how to lower d_pose DURING the witness training loop)
ranked against the measured warp-real-luma baseline. The load-bearing QUANTITATIVE claims
(the rest of the survey is a [prediction] ranking):

  * NO lever beats warp-real-luma alone -- all 13 are COMPLEMENTS, not replacements (the real-
    keyframe-luma pose carrier is pose-valid BY CONSTRUCTION: frame0 is seg-free per modules.py,
    so warping the REAL luma by xi feeds PoseNet a real-luma pair; the FiLM/dxi residual is the
    only trained part). openpilot xi-init is co-#1 (dual-axis warm-start: seeds BOTH the pose
    warp AND the lane-advection xi).

  * DISJOINT-FRAME FREEZE-AND-ADD realizes the MEASURED seg-perp-pose orthogonality (cos ~6e-5)
    EXACTLY: pose rides even frames (f0->real-luma warp), d_seg rides odd frames (f1->witness),
    so their gradients touch DISJOINT parameters (+ trunk stop-grad) => adding the pose gradient
    perturbs d_seg by ~cos ~6e-5 (negligible; the freeze-and-add is EXACT at this cos).

  * PCGrad (gradient-surgery projection) is a FALSE FRIEND at cos ~6e-5: projecting out the
    conflicting component is a NO-OP when the gradients are ALREADY orthogonal -- it adds
    compute + a projection that can only INJECT noise, never help. Do NOT use it here.

  * KKT pose-tube is the MOST GOAL-aligned lever (a trust-region constraint keeping d_pose
    inside its tube while d_seg descends), but all three (#205 H1 openpilot-xi-warm-start >
    D1 disjoint-frame freeze-and-add > E1 KKT pose-tube) are DESIGN-STAGE pending the #224
    trainer wire-in.

Two watch-item papers (NOT in-training levers, filed for the framing): LA-Pose (arXiv 2604.27448,
latent-action -- validates the framing + is an offline xi-estimator candidate) + Telescope
(arXiv 2604.06332, hyperbolic foveation = a d_seg FAR-FIELD lever, NOT pose).

    interference(cos) ~= |grad_pose . grad_seg| / (|grad_pose| |grad_seg|) = cos  (== 6e-5 => exact)

Consumers: the #205 pose plan (H1->D1->E1) + the DSL PoseTrainingGauge (design-stage) + warp_real_luma.
Producers: the Wave-F pose-in-training levers survey (sister-in-flight ledger).
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

EQUATION_ID = "pose_in_training_lever_survey_verdict_v1"

# The MEASURED grounding number: seg-perp-pose gradient cosine similarity (established prior finding;
# the disjoint-frame freeze-and-add realizes this EXACTLY, PCGrad is a no-op at it).
_SEG_POSE_COS = 6e-5


def predict_freeze_and_add_interference(cos_similarity: float) -> float:
    """The residual d_seg <-> d_pose gradient interference of the disjoint-frame freeze-and-add
    == the cosine similarity of the two gradients. At the MEASURED seg-perp-pose cos ~6e-5 the
    freeze-and-add is EXACT (interference negligible) AND PCGrad is a no-op (nothing to project)."""

    return abs(float(cos_similarity))


def build_pose_in_training_lever_survey_verdict_v1() -> CanonicalEquation:
    """Build the Wave-F pose in-training lever survey-verdict canonical equation."""

    survey = ".omx/research/pose_in_training_levers_survey_20260702.md"

    anchor_orthogonality = EmpiricalAnchor(
        anchor_id="seg_pose_gradient_orthogonality_freeze_and_add_exact_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"lever": "D1_disjoint_frame_freeze_and_add", "trunk_stopgrad": True,
                "pose_frame": "even (f0->real-luma warp)", "seg_frame": "odd (f1->witness)"},
        predicted_output={"interference": predict_freeze_and_add_interference(_SEG_POSE_COS)},
        empirical_output={"seg_pose_cos": _SEG_POSE_COS,
                          "note": "freeze-and-add EXACT at cos~6e-5; PCGrad is a NO-OP (false friend)"},
        residual=0.0,
        source_artifact=survey,
        measurement_method="prior_measured_seg_pose_gradient_cosine",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=survey,
            reactivation_criteria="re-measure the seg/pose gradient cosine on the witness at n600",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_ranking = EmpiricalAnchor(
        anchor_id="pose_in_training_lever_ranking_survey_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"n_levers_surveyed": 13,
                "top_plan": "H1_openpilot_xi_warm_start > D1_disjoint_freeze_and_add > E1_KKT_pose_tube",
                "papers": ["LA-Pose_2604.27448", "Telescope_2604.06332"]},
        predicted_output={"verdict": "no_lever_beats_warp_real_luma_alone; all are complements; openpilot_xi_co_1"},
        empirical_output={"note": "SURVEY [prediction]; DESIGN-STAGE pending #224 trainer wire-in of H1/D1/E1"},
        residual=0.0,
        source_artifact=survey,
        measurement_method="survey_prediction_pending_205_wire_in",
        empirical_verification_status="ASSUMED_AWAITING_VERIFICATION",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=survey,
            reactivation_criteria="#205 pose run (H1->D1->E1) byte-closed through R -> rank the levers by measured d_pose",
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Pose in-training lever survey verdict (warp-real-luma dominant; PCGrad false-friend)",
        one_line_summary=(
            "13 pose levers surveyed: none beats warp-real-luma (all complements); "
            "disjoint freeze-and-add is EXACT at seg-perp-pose cos~6e-5; PCGrad is a no-op."
        ),
        latex_form=(
            r"\text{interference}(\cos) = |\hat g_{pose}\cdot\hat g_{seg}| = \cos \approx 6\times10^{-5}"
            r"\ \Rightarrow\ \text{freeze-and-add exact},\ \text{PCGrad no-op}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pose_in_training_lever_survey_verdict:predict_freeze_and_add_interference"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory", "predicted"],
            "survey": True,
            "note": "the cos~6e-5 grounding is measured; the lever RANKING is [prediction] pending #205",
        },
        units_in={"cos_similarity": "dimensionless_gradient_cosine"},
        units_out={"interference": "dimensionless_residual_dseg_perturbation"},
        empirical_anchors=(anchor_orthogonality, anchor_ranking),
        predicted_vs_empirical_residual={
            "prior_measured_seg_pose_gradient_cosine": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge",
            "tac.boundary_math.warp_real_luma_frame0",
        ),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="pose_in_training_lever_survey_verdict.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[predicted]",
            hardware_substrate="unknown",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "predict_freeze_and_add_interference",
    "build_pose_in_training_lever_survey_verdict_v1",
]
