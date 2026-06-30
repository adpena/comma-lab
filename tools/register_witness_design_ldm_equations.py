#!/usr/bin/env python
"""Refresh the canonical equations registry (operator-flagged stale) — fold in
(a) the 2026-06-30 design refinements (pose=screw, canonicalize-to-ground-frame,
SE(3) B-spline ego-trajectory, dual-quaternion screw-blend, movables-stored,
residual intrinsic-dim) and (b) the LDM theoretical grounding (Mikulasch &
Zenke, "Understanding SSL via Latent Distribution Matching", ICML 2026).

APPEND-ONLY (Catalog #110/#113 HISTORICAL_PROVENANCE): every equation below is a
NEW equation_id that REFERENCES (annotates / supersedes) the prior E0-E12 rows
via ``annotates``/``supersedes`` keys in ``domain_of_validity``. The original
rows (witness_unified_action_fixed_fisher_background_v1, pose_sqrt_concave_
coupling_sidecar_v1, rate_mdl_cosmological_constant_reverse_waterfill_v1, ...)
are NOT mutated and NOT re-registered — they stay verbatim as HISTORICAL_PROVENANCE.

Honest tiers (NO-FAKE — theoretical != measured):
  * THEORETICAL_ANCHOR — cited from the LDM paper, NOT a measurement (empty
    anchors; axis "[predicted]").
  * DERIVED — standard textbook math / Whitney embedding / literature-grounded
    engineering (screw memo); anchors pending where unmeasured.
  * SOLVED — structural design refinement (the deeper reformulation); pending
    realized-through-R measurement.
  * MEASURED — carries a real research-signal EmpiricalAnchor (the grok pose-warp
    $0 confirmation; the ~8-dim lane-orbit manifold), advisory/pre-R.

means != ends: the equations are the MATH VIEW (a MEANS). The pointer (0.19110)
moves only on the byte-closed exact row. No GPU / launch / pipeline touch here.
"""
from __future__ import annotations

import hashlib

from tac.canonical_equations import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    RECALIBRATE_NEVER_AUTO,
    register_canonical_equation,
    query_equations,
)
from tac.canonical_equations.equation import (
    INFERRED_FROM_DOMAIN_LITERATURE,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

UTC = "2026-06-30T22:30:00Z"
DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
SCREW = ".omx/research/screw_twist_se3_literature_enrichment_20260630T220000Z.md"
DESIGN = ".omx/research/thetastar_residual_inr_config_update_requirements_20260630T220000Z.md"
COLO_JSON = "experiments/results/colocation_test_20260629T160343Z/colocation_results.json"

# The LDM theoretical anchor (just-read paper). Cited, NOT measured.
LDM_CITE = "Mikulasch & Zenke, 'Understanding SSL via Latent Distribution Matching', ICML 2026"


def _spec_sha(eqid: str, latex: str) -> str:
    """sha256 of the equation's defining spec (honest provenance, not a measurement)."""
    return hashlib.sha256((eqid + "\n" + latex).encode("utf-8")).hexdigest()


def _pred_prov(eqid: str, latex: str, axis: str = "[predicted]", hw: str = "unknown"):
    return build_provenance_for_predicted(
        model_id=f"witness_design_ldm.{eqid}",
        inputs_sha256=_spec_sha(eqid, latex),
        measurement_axis=axis,
        hardware_substrate=hw,
    )


def eq(eqid, name, summary, latex, callable_path, domain, units_in, units_out,
       anchors, residual, consumers, producers, trigger=RECALIBRATE_ON_NEW_ANCHORS,
       axis="[predicted]", hw="unknown"):
    return CanonicalEquation(
        equation_id=eqid, name=name, one_line_summary=summary, latex_form=latex,
        python_callable_module_path=callable_path, domain_of_validity=domain,
        units_in=units_in, units_out=units_out, empirical_anchors=anchors,
        predicted_vs_empirical_residual=residual, last_calibration_utc=UTC,
        next_recalibration_trigger=trigger, canonical_consumers=consumers,
        canonical_producers=producers, provenance=_pred_prov(eqid, latex, axis, hw),
    )


equations: list[CanonicalEquation] = []

# ============================================================================
# 1. MASTER ACTION <-> LDM alignment+uniformity correspondence (THEORETICAL_ANCHOR)
#    Annotates E0 (witness_unified_action_fixed_fisher_background_v1) + E7 rate.
# ============================================================================
equations.append(eq(
    "witness_action_ldm_alignment_uniformity_correspondence_v1",
    "Master action S_tau = LDM (alignment+uniformity) <-> (distortion+rate) [THEORETICAL ANCHOR]",
    "LDM F=-KL[R||P]=alignment+uniformity maps onto our action S_tau=distortion(d_seg/d_pose)+rate(entropy/MDL): alignment<->distortion, uniformity<->rate.",
    r"\mathcal{F}_{LDM} = -D_{KL}[R(z,z')\,\|\,P_\theta(z,z')] = \underbrace{\langle\log P_\theta(z'|z)\rangle}_{\text{alignment}\,\leftrightarrow\,D_{seg},D_{pose}} + \underbrace{H_R[z,z']}_{\text{uniformity}\,\leftrightarrow\,25B/N\,(\text{rate/entropy})}",
    "tac.contest_score:compute_contest_score",
    {"role": "ldm_grounding_of_E0_master_and_E7_rate",
     "provenance_tag": "THEORETICAL_ANCHOR",
     "tier": "THEORETICAL_ANCHOR_LDM_cited_not_measured",
     "annotates": ["witness_unified_action_fixed_fisher_background_v1",
                   "rate_mdl_cosmological_constant_reverse_waterfill_v1",
                   "indirect_rd_logloss_equals_information_bottleneck_v1"],
     "citation": LDM_CITE,
     "correspondence": {"alignment": "distortion (d_seg + d_pose, the predictive-match term)",
                        "uniformity": "rate (entropy/MDL, the H_R[z,z'] entropy-estimator term)"},
     "ldm_discussion": "SSL latents intrinsic-dim ~few-tens; SSL = geometric reparameterization NOT lossy compression; the entropy estimator is the key term -> grounds rate-as-entropy-estimator",
     "no_fake": "THEORETICAL correspondence cited from the LDM paper; NOT an exact-eval measurement"},
    {"R_joint": "true_positive_pair_distribution", "P_theta": "model_predictor_distribution"},
    {"F_ldm": "negative_kl_decomposed_alignment_plus_uniformity"},
    (), {},
    ("tac.unified_action", "tac.contest_score", "tac.boundary_math.lever_b_levelset_generator"),
    ("tools.register_witness_design_ldm_equations",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ============================================================================
# 2. POSE = ego screw/twist xi in se(3), identifiable up-to-affine (LDM Thm 1)
#    Annotates E9 (pose_sqrt_concave_coupling_sidecar_v1). DERIVED + MEASURED + THEORETICAL.
# ============================================================================
e2_grokwarp = EmpiricalAnchor(
    anchor_id="grok_pose_warp_dual_use_dseg_modulation_feed_ja_20260630",
    measurement_utc=UTC,
    inputs={"method": "stratified per-class SE(3) warp from STORED pose",
            "classes": {"road": "ground_homography(pose)", "hood": "identity", "sky": "rotation_only_KRKinv"},
            "test": "$0 grok pose-warp advisory (necessary-not-sufficient, pre-R)"},
    predicted_output={"pose_is_free_dual_use_dseg_modulation": True},
    empirical_output={"road_ground_homography_dseg_gain_pct": 15.0, "calibration": "closes via EON intrinsics",
                      "commit": "2f83e0b9e", "feed": "FEED-ja", "axis": "[macOS-MLX research-signal]"},
    residual=0.0,
    source_artifact=DAG,
    measurement_method="[macOS-MLX research-signal]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=DAG,
        reactivation_criteria="realized_through_R_dseg_on_byte_closed_warp_archive",
        measurement_axis="[macOS-MLX research-signal]", hardware_substrate="macos_arm64_mlx",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
e2_affine = EmpiricalAnchor(
    anchor_id="pose_identifiable_up_to_affine_ldm_theorem1_20260630",
    measurement_utc=UTC,
    inputs={"theorem": "LDM Thm 1: predictive model + Gaussian predictor + invertible encoder covering latent => recovers true latent UP TO AFFINE",
            "instance": "physical screw xi <-> PoseNet 6-vector gap is AFFINE"},
    predicted_output={"posenet6_approx_affine_of_xi": True, "readback_validation": "PENDING"},
    empirical_output={"status": "THEORETICAL_ANCHOR_cited_not_measured", "citation": LDM_CITE},
    residual=0.0,
    source_artifact=SCREW,
    measurement_method="[predicted]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=SCREW,
        reactivation_criteria="measure_posenet_readback_of_stored_xi_affine_fit_residual",
        measurement_axis="[predicted]", hardware_substrate="theoretical",
    ),
    empirical_verification_status=INFERRED_FROM_DOMAIN_LITERATURE,
)
equations.append(eq(
    "pose_ego_screw_twist_identifiable_up_to_affine_v1",
    "Pose = ego screw/twist xi in se(3), identifiable UP-TO-AFFINE; d_pose target = PoseNet[:6] ~ affine(xi)",
    "The d_pose latent is the ego-motion twist xi in se(3) (dual-use: warps the partition for d_seg AND is the pose); PoseNet(pair)[:6] ~ affine(xi) per LDM Thm 1 (identifiable up-to-affine).",
    r"\xi=(\rho,\omega)\in\mathfrak{se}(3),\; T=\exp([\xi]^\wedge)\in SE(3);\quad \text{PoseNet}(pair)[:6] \approx A\,\xi + b\ (\text{LDM Thm 1: up-to-affine})",
    "tac.lie.se3",
    {"role": "pose_term_refinement_of_E9",
     "provenance_tag": "DERIVED+MEASURED(warp,research-signal)+THEORETICAL_ANCHOR(affine identifiability)",
     "tier": "DERIVED_screw_plus_LDM_Thm1_affine_anchor",
     "annotates": ["pose_sqrt_concave_coupling_sidecar_v1",
                   "wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1"],
     "twist_convention": "xi=(rho,omega) translation-first (Sola micro-Lie arXiv:1812.01537)",
     "dual_use": "xi is FREE: warps per-class partition for d_seg (E.2 grok warp) AND is the pose for d_pose",
     "identifiability_citation": LDM_CITE + " (Theorem 1: nonlinear-Gaussian predictor + invertible encoder + latent coverage => recover true latent up to affine)",
     "readback_validation": "PENDING — does PoseNet read xi back off the witness; fallback = store the PoseNet-6-vector directly",
     "screw_memo": SCREW, "measurement_axes": ["[macOS-MLX research-signal]", "[predicted]"]},
    {"xi": "ego_twist_se3_6vector", "pair": "two_frame_yuv6"},
    {"posenet6": "first_6_posenet_scalars", "affine_map": "A_xi_plus_b"},
    (e2_grokwarp, e2_affine), {"[macOS-MLX research-signal]": 0.0, "[predicted]": 0.0},
    ("tac.lie.se3", "tac.scorer_targets", "tac.boundary_math.lever_b_levelset_generator"),
    ("tac.lie.se3", "tools.register_witness_design_ldm_equations"),
    axis="[macOS-MLX research-signal]", hw="macos_arm64_mlx",
))

# ============================================================================
# 3. CANONICALIZE-TO-GROUND-FRAME residual (SOLVED, structural; LDM geom-reparam)
# ============================================================================
equations.append(eq(
    "witness_canonicalize_to_ground_frame_residual_v1",
    "Canonicalize S_tau to the ground frame: residual = GT - W(xi_ego(t)) o Phi_canon, through R",
    "Write the action in the ego-removed ground frame (screw encoded ONCE); the residual INR carries only what does NOT fall out of the screw action -> code collapses ~2-4 dims, image-space rate vanishes.",
    r"S_\tau \text{ in ground frame};\quad r(x,t) = \mathrm{GT}(x,t) - R\big[\,W(\xi_{ego}(t))\circ\Phi_{canon}(x)\,\big],\; \xi_{ego}(t)\ \text{encoded once}",
    "tac.v2_compose.residual_compose",
    {"role": "structural_reformulation_of_E0_residual",
     "provenance_tag": "SOLVED",
     "tier": "SOLVED_structural_design_pending_realized_through_R",
     "annotates": ["witness_unified_action_fixed_fisher_background_v1",
                   "rate_mdl_cosmological_constant_reverse_waterfill_v1"],
     "mechanism": "ego-motion removed via the once-encoded screw xi_ego(t); residual = what the screw action does NOT explain",
     "ldm_grounding": "LDM Discussion: SSL = geometric reparameterization NOT lossy compression -> removing ego-motion IS the reparameterization that collapses the code dim; " + LDM_CITE,
     "expected_effect": "residual code ~2-4 dims; ~65KB image-space rate vanishes (DESIGN estimate, unmeasured)",
     "R_operator": "bicubic^384->874 -> uint8-STE -> bilinear^512x384 (the eval round-trip; authority)",
     "design_memo": DESIGN, "status": "design-stage; no realized-through-R anchor yet"},
    {"GT": "ground_truth_frames", "xi_ego": "once_encoded_ego_screw", "Phi_canon": "canonical_ground_frame_witness"},
    {"residual_r": "ground_frame_residual_through_R", "code_dim": "collapsed_2_to_4"},
    (), {},
    ("tac.v2_compose.residual_compose", "tac.boundary_math.margin_conditional_residual", "tac.lie.se3"),
    ("tools.register_witness_design_ldm_equations",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ============================================================================
# 4a. xi_ego(t) = cumulative SE(3) cubic B-spline (DERIVED; Sommer-Usenko)
# ============================================================================
equations.append(eq(
    "ego_motion_cumulative_se3_bspline_v1",
    "xi_ego(t) = cumulative SE(3) cubic B-spline: 600 pose 6-vectors -> ~48-96 control floats + C2 prior",
    "Represent the whole-clip ego trajectory as a uniform cubic cumulative B-spline on SE(3) (Sommer-Usenko); 3600 floats -> ~48-96 control floats + a C2 smoothness prior (the counted-byte temporal win).",
    r"T(u)=T_i\prod_{j=1}^{3}\exp\!\big(\tilde B_j(u)\,\Omega_j\big),\ \Omega_j=\log_{SE3}(T_{i+j-1}^{-1}T_{i+j}),\ \tilde B_1=\tfrac{5+3u-3u^2+u^3}{6},\ \tilde B_2=\tfrac{1+3u+3u^2-2u^3}{6},\ \tilde B_3=\tfrac{u^3}{6}",
    "tac.lie.se3_bspline",
    {"role": "temporal_factor_of_canonicalize_to_ground_frame",
     "provenance_tag": "DERIVED",
     "tier": "DERIVED_literature_grounded_anchors_pending",
     "citation": "Sommer, Usenko et al., 'Efficient Derivative Computation for Cumulative B-Splines on Lie Groups' (arXiv:1911.08860, CVPR 2020); Kim-Kim-Shin 1995 (SO(3) cumulative basis)",
     "convention": "uniform cubic k=4; cumulative blending matrix; translation-first twist",
     "byte_accounting": "control poses are COUNTED video-derived payload; the spline-eval algorithm is FREE (rule-118 generic in inflate.py)",
     "differentiability": "product of exp of constant twists scaled by scalar bases -> MLX autodiff gives dT/du AND control-point grads; no custom VJP for the spline layer",
     "license_note": "implement FRESH from paper formulas (basalt impl is MPL-2.0); cross-check numerically only",
     "screw_memo": SCREW, "status": "design-stage; byte/d_pose anchors pending"},
    {"control_poses": "se3_control_point_set", "u": "local_segment_time"},
    {"T_u": "se3_pose_at_time_u", "control_floats": "approx_48_to_96"},
    (), {},
    ("tac.lie.se3_bspline", "tac.scorer_targets", "tac.v2_compose.residual_compose"),
    ("tac.lie.se3_bspline", "tools.register_witness_design_ldm_equations"),
))

# ============================================================================
# 4b. Dual-quaternion screw-blend at the boundary annulus (DERIVED/ASPIRATIONAL; Kavan)
# ============================================================================
equations.append(eq(
    "dual_quaternion_screw_blend_annulus_seam_v1",
    "Dual-quaternion screw-blend (DLB/ScLERP) at the class-boundary annulus removes the per-class-warp seam tear",
    "At the annulus (where d_seg is scored) DLB-blend per-class warps via dual quaternions so the warp field is continuous -> no seam-tear argmax flips. ASPIRATIONAL: measure through R.",
    r"\hat q = \hat b/\lVert\hat b_r\rVert,\ \hat b=\sum_i w_i(x)\,\hat q_i,\ \hat q_i=q_{r,i}+\tfrac{\epsilon}{2}(0,t_i)\!\otimes\! q_{r,i};\quad \text{ScLERP: } \hat q(u)=\hat q_A\otimes(\hat q_A^*\otimes\hat q_B)^u",
    "tac.lie.screw_blend",
    {"role": "boundary_seam_continuity_term_of_per_class_warp",
     "provenance_tag": "DERIVED_ASPIRATIONAL",
     "tier": "DERIVED_geometry_ASPIRATIONAL_must_measure_through_R",
     "citation": "Kavan-Collins-Zara-O'Sullivan, 'Geometric Skinning with Approximate Dual Quaternion Blending' (ACM TOG 2008); pytransform3d dual_quaternion_sclerp (BSD-3) reference oracle",
     "annulus": "codim-1 inter-class boundary band = exactly where d_seg is scored",
     "soft_weights": "w_c(pixel) from SegNet softmax OR signed distance to boundary",
     "aspirational_flag": "geometrically the right tool but UNMEASURED through R; current screw probes used hard regime + persist-fallback; open measurement whether blend reduces through-R d_seg on annulus (screw memo S4 item 3)",
     "screw_memo": SCREW, "status": "ASPIRATIONAL; anchors pending realized-through-R"},
    {"per_class_warps": "T_road_T_sky_T_hood_dualquats", "soft_weights": "w_c_per_pixel"},
    {"blended_warp": "continuous_dualquat_field_across_annulus"},
    (), {},
    ("tac.lie.screw_blend", "tac.boundary_math.lever_b_levelset_generator"),
    ("tac.lie.screw_blend", "tools.register_witness_design_ldm_equations"),
))

# ============================================================================
# 4c. Movables stored OUT of the INR (SOLVED design; d_seg target ~0.0008)
# ============================================================================
equations.append(eq(
    "movables_stored_out_of_inr_multibody_v1",
    "Movables STORED (multibody codec <=2.7KB -> d_seg ~0.0008); the INR carries lane-survival ONLY",
    "Small movable bodies (cars) are stored via a multibody codec (<=2.7KB) rather than fit by the INR; the trained INR shrinks to ONLY the lane-survival residual (the binding R-survival wall).",
    r"\Sigma = \underbrace{\Phi_{canon}\circ W(\xi_{ego})}_{\text{deterministic bulk}} \,\cup\, \underbrace{\mathcal{M}_{stored}}_{\le 2.7\text{KB},\,\Delta d_{seg}\sim 0.0008} \,\cup\, \underbrace{\mathrm{INR}_{lane\text{-}survival}}_{\text{trained residual only}}",
    "tac.v2_compose.residual_compose",
    {"role": "capacity_allocation_refinement_of_E0_E7",
     "provenance_tag": "SOLVED+DERIVED(dseg estimate)",
     "tier": "SOLVED_design_dseg_0008_estimate_pending_through_R",
     "annotates": ["rate_mdl_cosmological_constant_reverse_waterfill_v1"],
     "mechanism": "store small rigid movable bodies (out of the INR); INR budget spent only on the lane-survival long-tail",
     "byte_estimate": "multibody codec <=2.7KB (DESIGN estimate)",
     "dseg_estimate": "~0.0008 (DERIVED estimate, unmeasured through R)",
     "binding_wall": "lane-survival residual = the R-survival F1 wall (the real INR job)",
     "design_memo": DESIGN, "status": "design-stage; byte/d_seg anchors pending realized-through-R"},
    {"movable_bodies": "small_rigid_car_segments", "inr_budget": "residual_capacity"},
    {"stored_movable_bytes": "le_2700", "inr_scope": "lane_survival_only", "movable_dseg": "approx_0.0008"},
    (), {},
    ("tac.v2_compose.residual_compose", "tac.scorer_targets"),
    ("tools.register_witness_design_ldm_equations",),
))

# ============================================================================
# 5. Residual manifold intrinsic-dim -> mod-dim DERIVED (Whitney 2m+1; LDM few-tens)
# ============================================================================
e5_laneorbit = EmpiricalAnchor(
    anchor_id="lane_orbit_manifold_dim_8_decisive_20260623",
    measurement_utc=UTC,
    inputs={"object": "d_seg islands = lane markings", "method": "shuffle-controlled manifold-dim probe",
            "decisive": "9 measured lines converge"},
    predicted_output={"lane_orbit_is_low_dim_nonlinear_manifold": True},
    empirical_output={"lane_orbit_dim_approx": 8, "nature": "NONLINEAR manifold (go-generator, shuffle-controlled)",
                      "feed": "DECISIVE 2026-06-23"},
    residual=0.0,
    source_artifact=DAG,
    measurement_method="[macOS-MLX research-signal]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=DAG,
        reactivation_criteria="run_residual_specific_TwoNN_MLE_ID_on_residual_target_npz",
        measurement_axis="[macOS-MLX research-signal]", hardware_substrate="macos_arm64_mlx",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
equations.append(eq(
    "residual_manifold_intrinsic_dim_whitney_v1",
    "mod-dim = Whitney 2m+1 of the MEASURED residual intrinsic-dim (NOT inherited mod-16); LDM few-tens manifold",
    "Derive the witness mod-dim from the measured residual intrinsic-dim m via Whitney 2m+1 (NOT inherited mod-16 which under-embeds); LDM grounds the few-tens intrinsic-dim + geometric-reparam.",
    r"\mathrm{mod\text{-}dim} = 2m+1\ (\text{Whitney}),\ m=\mathrm{ID}(\text{residual sub-manifold});\ \text{measured lane-orbit } m\approx 8 \Rightarrow \mathrm{mod\text{-}dim}\approx 17",
    "tac.witness_autoconfig",
    {"role": "architecture_derivation_of_E0_residual",
     "provenance_tag": "DERIVED+MEASURED(lane-orbit dim)+THEORETICAL_ANCHOR(LDM few-tens)",
     "tier": "DERIVED_Whitney_plus_measured_8dim_plus_LDM_anchor",
     "citation": "Whitney embedding theorem (2m+1); " + LDM_CITE + " (Discussion: SSL intrinsic-dim ~a few tens; SSL = geometric reparameterization, the entropy estimator is the key term)",
     "measured": "lane-orbit manifold dim ~8 (DECISIVE 2026-06-23, 9 lines converge); residual-specific TwoNN/MLE ID measurement PENDING ($0)",
     "derived": "mod-dim = Whitney 2m+1; m~8 -> ~17; image-space ~19-21 pre-ground-frame, lower after canonicalization",
     "anti_pattern": "inherited mod-16 UNDER-embeds the residual sub-manifold",
     "design_memo": DESIGN, "measurement_axes": ["[macOS-MLX research-signal]"]},
    {"residual_id_m": "measured_residual_intrinsic_dim", "lane_orbit_dim": "approx_8"},
    {"mod_dim": "whitney_2m_plus_1", "embed_dim": "derived_not_inherited"},
    (e5_laneorbit,), {"[macOS-MLX research-signal]": 0.0},
    ("tac.witness_autoconfig", "tac.boundary_math.lever_b_levelset_generator"),
    ("tac.witness_autoconfig", "tools.register_witness_design_ldm_equations"),
    axis="[macOS-MLX research-signal]", hw="macos_arm64_mlx",
))

# ============================================================================
# REGISTER ALL (APPEND-ONLY; skip any already-present id, never mutate originals)
# ============================================================================
existing = {e.equation_id for e in query_equations()}
registered, skipped = [], []
for e in equations:
    if e.equation_id in existing:
        skipped.append(e.equation_id)
        continue
    register_canonical_equation(
        e, agent="canonical-eq-refresh", subagent_id="canonical_eq_refresh_ldm_20260630",
        notes="design-refine + LDM grounding refresh (operator 2026-06-30); APPEND-ONLY, new ids annotate E0-E12",
    )
    registered.append(e.equation_id)

print("REGISTERED (%d):" % len(registered))
for r in registered:
    print("  +", r)
if skipped:
    print("SKIPPED already-present (%d):" % len(skipped))
    for s in skipped:
        print("  =", s)

# verify round-trip — every new id present, every annotated original UNTOUCHED
post = {e.equation_id for e in query_equations()}
missing = [e.equation_id for e in equations if e.equation_id not in post]
print("VERIFY: all 7 new equations in registry =", not missing,
      ("MISSING=" + str(missing)) if missing else "")
ORIGINALS = [
    "witness_unified_action_fixed_fisher_background_v1",
    "pose_sqrt_concave_coupling_sidecar_v1",
    "rate_mdl_cosmological_constant_reverse_waterfill_v1",
    "indirect_rd_logloss_equals_information_bottleneck_v1",
]
print("VERIFY: annotated originals still present (untouched) =",
      all(o in post for o in ORIGINALS))
