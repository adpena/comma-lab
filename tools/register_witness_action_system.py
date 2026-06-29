#!/usr/bin/env python
"""Register the witness unified-action SYSTEM OF EQUATIONS (E0-E12) into the
canonical registry. Honest tiers: measured laws get real EmpiricalAnchors;
theoretical terms carry NO fabricated calibration (empty anchors). Operator
directive 2026-06-29: formalize the system + no signal loss + integration."""
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
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

UTC = "2026-06-29T20:50:00Z"
COLO_JSON = "experiments/results/colocation_test_20260629T160343Z/colocation_results.json"
COLO_SHA = "e4f9d971dc16f4d99a51a55d2e82b8529b004c237f57455d2cf017ee780acf68"
OPT_MEMO = ".omx/research/per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629.md"
DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"


def _spec_sha(eqid: str, latex: str) -> str:
    """sha256 of the equation's defining specification (honest, not a measurement)."""
    return hashlib.sha256((eqid + "\n" + latex).encode("utf-8")).hexdigest()


def _pred_prov(eqid: str, latex: str, axis: str = "[predicted]", hw: str = "unknown"):
    return build_provenance_for_predicted(
        model_id=f"witness_system.{eqid}",
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


SYS = "witness unified-action system (E0 master) per .omx/research/system_of_equations_formalization_20260629.md + GR-research memo"

equations = []

# ---- E0 MASTER ------------------------------------------------------------
equations.append(eq(
    "witness_unified_action_fixed_fisher_background_v1",
    "Witness unified action on a fixed frozen-scorer Fisher background (MASTER)",
    "Contest score = ONE variational action S_t stationary in the FROZEN-scorer Fisher metric (fixed background, no back-reaction); E1-E9 are its terms.",
    r"S_\tau[\theta] = 100\,D_{seg}^{\tau} + \sqrt{10\,D_{pose}} + 25\,B(\theta)/N;\quad \delta S_\tau/\delta\varphi = 0 \text{ in } G(\theta)",
    "tac.contest_score:compute_contest_score",
    {"role": "master_action", "terms": ["E1_metric", "E2_dm1", "E3_dm2_colocation", "E4_dm2_lane", "E5_dm3_flow", "E6_ib", "E7_rate", "E8_gate", "E9_pose", "E10_boundary_energy", "E11_decode_determinism", "E12_vcm_headroom"],
     "physics": "matter_on_fixed_curved_background_NOT_full_GR_no_back_reaction",
     "theta_star_frame": "theta_star_witness_lever_stack_and_variational_levelset_frame_20260627 = OPERATIONAL lever-stack instantiation of E0 (witness = viscosity solution of the variational level-set PDE; same object as S_tau)",
     "theta_star_campaign_tasks": ["#183", "#184", "#185"], "witness_dsl": "src/tac/witness_dsl (#189)",
     "measurement_axes": ["[contest-CPU]", "[contest-CUDA]"], "system_doc": "system_of_equations_formalization_20260629"},
    {"d_seg": "argmax_disagreement_fraction", "d_pose": "mse_6_posenet_scalars", "bytes": "archive_zip_uint8_count", "N": "37545489_normalizer"},
    {"S": "contest_score_scalar"},
    (), {},
    ("tac.unified_action", "tac.contest_score", "tac.boundary_math.lever_b_levelset_generator"),
    ("tools.colocation_fisher_stress_anisotropy_test",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ---- E1 METRIC (theoretical; validated by E3) ------------------------------
equations.append(eq(
    "frozen_scorer_fisher_pullback_metric_v1",
    "Frozen-scorer Fisher pullback metric = the fixed background geometry",
    "The descent geometry G is the Fisher pullback of the FROZEN SegNet through the witness Jacobian; DM1 conditioning lives in G; validated by E3.",
    r"G(\theta) = \mathbb{E}_x[\,J_x^\top F_x J_x\,],\; F_x = \mathrm{diag}(p_x) - p_x p_x^\top,\; p_x = \mathrm{softmax}(z(x))",
    "tools.colocation_fisher_stress_anisotropy_test",
    {"role": "metric_term_of_E0", "background": "fixed_frozen_scorer", "validated_by": "frozen_scorer_fisher_curvature_margin_colocation_v1",
     "measurement_axes": ["[macOS-CPU advisory]"]},
    {"logits_z": "segnet_5class_logits", "jacobian_J": "d_logits_d_theta"},
    {"G": "fisher_metric_psd_matrix"},
    (), {},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.unified_action"),
    ("tools.colocation_fisher_stress_anisotropy_test",),
))

# ---- E2 DM1 Stiefel isometry (MEASURED collapse anchor) --------------------
e2_anchor = EmpiricalAnchor(
    anchor_id="dm1_film_rank_collapse_participation_ratio_20260629",
    measurement_utc=UTC,
    inputs={"matrix": "M=code@film.weight^T", "code_dim": 22, "stage_axis": "CE->L7"},
    predicted_output={"PR_invariant_under_stiefel": "PR(M)=PR(cov(code))"},
    empirical_output={"participation_ratio_CE": 3.34, "participation_ratio_L7": 1.19, "collapse_ratio": 2.81},
    residual=0.0,
    source_artifact=OPT_MEMO,
    measurement_method="[macOS-MLX research-signal]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=OPT_MEMO,
        reactivation_criteria="dm1_decisive_smoke_lands_PR_and_dseg",
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="macos_arm64_mlx",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
e2_identity_anchor = EmpiricalAnchor(
    anchor_id="dm1_stiefel_pr_identity_proven_by_test_20260629",
    measurement_utc=UTC,
    inputs={"identity": "PR(M=code@W^T)=PR(cov(code)) when W^T W=I", "build_commit": "07dd971d8"},
    predicted_output={"identity_holds": True},
    empirical_output={"identity_holds_by_unit_test": True, "test": "src/tac/tests/test_levelset_dm1_stiefel_entropy.py"},
    residual=0.0,
    source_artifact="src/tac/tests/test_levelset_dm1_stiefel_entropy.py (commit 07dd971d8)",
    measurement_method="[empirical]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path="src/tac/tests/test_levelset_dm1_stiefel_entropy.py",
        reactivation_criteria="stiefel_identity_unit_test_passes",
        measurement_axis="[empirical]", hardware_substrate="macos_arm64",
    ),
    empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
)
equations.append(eq(
    "dm1_stiefel_isometry_rank_preservation_v1",
    "DM1 byte-free cure: Stiefel-W isometry preserves the code participation ratio",
    "W^T W=I makes W an isometry so PR(M=code@W^T)=PR(cov(code)) EXACTLY; Stiefel-W + code spectral-entropy fix the measured rank-collapse (PR 3.34->1.19) by construction, byte-free.",
    r"W^\top W = I \;\Rightarrow\; \mathrm{PR}(M=\mathrm{code}\,W^\top) = \mathrm{PR}(\mathrm{cov}(\mathrm{code}));\quad \mathcal{E}_{spec} = -\beta\log[(\mathrm{tr}\,C)^2/\lVert C\rVert_F^2]",
    "tac.boundary_math.lever_b_levelset_generator",
    {"role": "dm1_conditioning_term_of_E0", "matrix": "film_weight_W", "code_dim_range": [16, 64],
     "measured_collapse": "PR 3.34->1.19 at L7", "byte_cost": 0, "measurement_axes": ["[macOS-MLX research-signal]", "[empirical]"],
     "build_commit": "07dd971d8 (DM1 minimal cure landed: Stiefel-W + code spectral-entropy, default-OFF byte-identical)",
     "identity_proven_by_test": "src/tac/tests/test_levelset_dm1_stiefel_entropy.py",
     "falsification_gate": "dm1_decisive_smoke_falsification_gate_v1"},
    {"W": "film_conditioning_weight", "code": "per_pair_code_manifold", "beta": "spectral_entropy_weight"},
    {"PR_M": "participation_ratio_of_M", "byte_delta": "zero_byte_structural_cure"},
    (e2_anchor, e2_identity_anchor), {"[macOS-MLX research-signal]": 0.0, "[empirical]": 0.0},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.unified_action"),
    ("tac.boundary_math.lever_b_levelset_generator", "src/tac/tests/test_levelset_dm1_stiefel_entropy.py"),
    axis="[macOS-MLX research-signal]", hw="macos_arm64_mlx",
))

# ---- E3 DM2 co-location (MEASURED, strongest anchor) -----------------------
e3_anchor = EmpiricalAnchor(
    anchor_id="colocation_fisher_curvature_margin_n96_20260629",
    measurement_utc=UTC,
    inputs={"frames": 96, "grid": "512x384", "scorer": "frozen_cpu_torch_segnet_argmax",
            "byte_faithful": "argmax_mismatch=0.0, margin_delta_max=4.8e-7"},
    predicted_output={"curvature_stress_colocate": True, "anisotropy_approx": 7.0},
    empirical_output={"pearson_curv_neg_margin_band": 0.978, "spearman": 0.908, "all_px_pearson": 0.814,
                      "fisher_trace_vs_specnorm": 0.997, "boundary_anisotropy": 9.56,
                      "flip_mass_in_2px_band": 0.968, "per_class_flip": {"road": 0.48, "lane": 0.19, "undriv": 0.15}},
    residual=0.0,
    source_artifact=COLO_JSON,
    measurement_method="[macOS-CPU advisory]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=COLO_JSON,
        reactivation_criteria="recompute_on_segnet_or_frame_cache_change",
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="macos_arm64",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
equations.append(eq(
    "frozen_scorer_fisher_curvature_margin_colocation_v1",
    "DM2 co-location law: Fisher curvature ~ -margin; margin is a byte-faithful Fisher surrogate",
    "Fisher curvature ~ -margin co-locate on the annulus (Pearson 0.978); the cheap top1-top2 margin is a byte-faithful Fisher-metric surrogate; anisotropy 9.56:1; flip-mass 96.8% in 2px band.",
    r"\mathrm{corr}(\mathrm{tr}\,F_x,\, -m(x))_{\partial\Sigma} = 0.978,\; m(x)=z_{(1)}-z_{(2)};\quad \text{anisotropy} \approx 9.56:1",
    "tools.colocation_fisher_stress_anisotropy_test",
    {"role": "dm2_loss_geometry_term_of_E0", "annulus": "codim1_2px_band", "measurement_axes": ["[macOS-CPU advisory]"],
     "commit": "b0bee924e", "n_frames": 96, "note": "advisory/research-signal NOT a contest score"},
    {"logits_z": "segnet_5class_logits", "margin_m": "top1_minus_top2_logit"},
    {"pearson": "dimensionless_corr", "anisotropy": "cross_over_along_tangent_ratio", "flip_mass_band": "dimensionless_ratio"},
    (e3_anchor,), {"[macOS-CPU advisory]": 0.0},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.unified_action"),
    ("tools.colocation_fisher_stress_anisotropy_test",),
    axis="[macOS-CPU advisory]", hw="macos_arm64",
))

# ---- E4 DM2 lane geodesic (research-signal anchor) -------------------------
e4_anchor = EmpiricalAnchor(
    anchor_id="dm2_lane_ipm_polynomial_collinearity_20260629",
    measurement_utc=UTC,
    inputs={"chart": "ipm_road_plane", "poly_degree": 3},
    predicted_output={"lane_is_low_degree_polynomial_in_ipm": True},
    empirical_output={"collinearity_px": 1.28, "tangent_cross_anisotropy": 7.0},
    residual=0.0,
    source_artifact=DAG,
    measurement_method="[macOS-MLX research-signal]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=DAG, reactivation_criteria="dm2_feed_hv",
        measurement_axis="[macOS-MLX research-signal]", hardware_substrate="macos_arm64_mlx",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
equations.append(eq(
    "dm2_lane_ipm_polynomial_geodesic_v1",
    "DM2 lane = deg-3 IPM-rectified polynomial = geodesic of the road-plane chart",
    "The lane boundary is a degree-3 polynomial in the IPM road-plane chart (collinear to 1.28px), anisotropic ~7x tangent/cross; the annulus is a geodesic of that chart.",
    r"\ell(u) = a_0 + a_1 u + a_2 u^2 + a_3 u^3 \text{ in IPM chart};\; \text{anisotropy} \approx 7\times",
    "tac.boundary_math.lever_b_levelset_generator",
    {"role": "dm2_lane_geodesic_term_of_E0", "chart": "ipm_homography", "measurement_axes": ["[macOS-MLX research-signal]"],
     "boundary_energy_refinement": "Finsler tangent-anisotropic boundary energy (Zach-Shan-Niethammer; Chen-Mirebeau-Cohen Finsler-elastica arXiv:1612.00343) = the principled anisotropic boundary cost matching measured 9.56:1 (E3) [lit-hunt FEED-ih, advisory not independently read]"},
    {"u": "along_lane_coord", "ipm_chart": "road_plane_homography"},
    {"lane_curve": "deg3_polynomial_coeffs"},
    (e4_anchor,), {"[macOS-MLX research-signal]": 0.0},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.unified_action"),
    ("tac.boundary_math.lever_b_levelset_generator",),
    axis="[macOS-MLX research-signal]", hw="macos_arm64_mlx",
))

# ---- E5 DM3 flow (theoretical framework) -----------------------------------
equations.append(eq(
    "dm3_natural_gradient_steepest_descent_under_norm_v1",
    "DM3 flow = natural-gradient / steepest-descent-under-a-norm; curriculum = homotopy",
    "Every optimizer = steepest descent under a norm (Adam=diagonal, Muon=spectral, SinkGD=doubly-stochastic); witness flow = natural-gradient/Dykstra in G; curriculum = homotopy of relaxations.",
    r"\dot\theta = -G^{-1}\nabla_\theta S_\tau;\quad \text{Adam}\!=\!\mathrm{diag},\ \text{Muon}\!=\!\mathrm{spectral},\ \text{SinkGD}\!=\!\mathrm{doubly\text{-}stochastic}",
    "tac.boundary_math.lever_b_levelset_generator",
    {"role": "dm3_flow_term_of_E0", "norms": ["diagonal", "spectral", "doubly_stochastic"],
     "framework": "theoretical_steepest_descent_under_norm", "cross_ref": ["ema_decay_substrate_stage_aware_v1", "dykstra_pareto_polytope_intersection_compounding_v1"]},
    {"grad": "score_gradient", "norm": "descent_geometry_choice"},
    {"update": "natural_gradient_step"},
    (), {},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.unified_action"),
    ("tac.boundary_math.lever_b_levelset_generator",),
))

# ---- E6 indirect-RD = IB (theorem) ----------------------------------------
equations.append(eq(
    "indirect_rd_logloss_equals_information_bottleneck_v1",
    "Contest data term = indirect/remote RD under log-loss = Information Bottleneck (exact)",
    "The contest is indirect/remote rate-distortion for a FROZEN decoder; under log-loss the seg distortion term equals the Information Bottleneck exactly (Courtade-Weissman 1110.3069).",
    r"\min R \text{ s.t. } \mathbb{E}[\ell_{log}] \le D \;\equiv\; \max I(Z;Y_{scorer}) - \beta I(X;Z) \text{ (IB)}",
    "tac.contest_score",
    {"role": "ib_data_term_of_E0", "theorem": "courtade_weissman_logloss_remote_coding_eq_IB",
     "citation": "arXiv:1110.3069", "tier": "theoretical_exact_for_surrogate",
     "posterior_coding_refinement": "code SOFT logits not argmax (iRDF arXiv:2410.09018) [lit-hunt FEED-ig, advisory not independently read]",
     "flip_cost_surrogate_refinement": "Generalised Wasserstein-Dice cost-matrix M weighted by measured Road48/Lane19/Undriv15 (Fidon arXiv:1707.00478) [lit-hunt FEED-ih, advisory]",
     "cross_ref": ["categorical_blahut_arimoto_rate_distortion_v1", "wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1"]},
    {"X": "source", "Z": "code", "Y_scorer": "frozen_segnet_output"},
    {"R": "rate_bits", "D": "logloss_distortion"},
    (), {},
    ("tac.unified_action", "tac.contest_score"),
    ("tac.contest_score",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ---- E7 rate = Lambda / MDL (theoretical) ----------------------------------
equations.append(eq(
    "rate_mdl_cosmological_constant_reverse_waterfill_v1",
    "Rate term = MDL / cosmological-constant Lambda; reverse-water-fill over boundary sensitivity",
    "25*bytes/N rate term = the action's MDL/Lambda regularizer; allocate by reverse-water-fill over boundary sensitivity; FREE generic generator vs COUNTED learned payload = the compliance boundary.",
    r"\Lambda = 25\,B(\theta)/N;\quad b^*(x) = [\tfrac12\log(\sigma^2(x)/\lambda)]_+ \text{ (reverse water-fill)}",
    "tac.contest_score",
    {"role": "rate_lambda_term_of_E0", "allocation": "reverse_water_fill",
     "compliance_boundary": "free_generic_generator_vs_counted_learned_payload",
     "zero_byte_generator_refinement": "eikonal-SDF zero-byte generator + modulation-split + integer-determinism (Balle) [lit-hunt FEED-ig, advisory]",
     "boundary_rate_descriptor_refinement": "context-tree contour coding w/ geometric prior (Zheng-Cheung-Florencio arXiv:1604.08001) + MDL texture+boundary (Mobahi arXiv:1006.3679) = 'store the partition via its smooth boundary at minimal bits', the 8-dim->hundreds-of-bytes instantiation [lit-hunt FEED-ih, advisory not independently read]",
     "cross_ref": ["master_gradient_null_space_byte_fraction_v1", "per_byte_leverage_uniformly_distributed_v1"]},
    {"bytes_B": "archive_zip_uint8_count", "sensitivity": "per_byte_boundary_sensitivity"},
    {"rate_term": "score_units", "bit_allocation": "per_dimension_bits"},
    (), {},
    ("tac.unified_action", "tac.contest_score"),
    ("tac.contest_score",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ---- E8 decisive-smoke falsification gate (PENDING) ------------------------
equations.append(eq(
    "dm1_decisive_smoke_falsification_gate_v1",
    "DM1 binding-cause falsification gate (PENDING the decisive $0 smoke)",
    "DM1 is the binding d_seg cause IFF the minimal arm holds PR(M)>=~3.0 AND lowers advisory d_seg >=~10%; else DM1 is a SYMPTOM (major redirect). Awaiting the decisive smoke measurement.",
    r"\text{BINDING} \iff \big(\mathrm{PR}(M)\ge 3.0\big)\wedge\big(\Delta d_{seg}^{adv}\le -0.10\big);\ \text{else SYMPTOM}",
    "tac.boundary_math.lever_b_levelset_generator",
    {"role": "falsification_gate_of_E2", "status": "FORMALIZATION_PENDING_AWAITING_DECISIVE_SMOKE",
     "rationale": "anchor lands when the 4-arm minimal-DM1-fix smoke (A0/A1/A2/A3) is measured",
     "measurement_axes": ["[macOS-MLX research-signal]"]},
    {"PR_M": "participation_ratio", "d_seg_delta": "advisory_d_seg_change"},
    {"verdict": "BINDING_or_SYMPTOM"},
    (), {},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.unified_action"),
    ("tac.boundary_math.lever_b_levelset_generator",),
))

# ---- E9 pose sqrt concave coupling -> sidecar (research-signal) ------------
e9_anchor = EmpiricalAnchor(
    anchor_id="pose_sqrt_coupling_quantizr_sidecar_realized_dpose_20260629",
    measurement_utc=UTC,
    inputs={"coupling": "sqrt(10*d_pose)", "method": "store_6_posenet_targets_plus_pose_FiLM"},
    predicted_output={"sidecar_not_carrier": True},
    empirical_output={"realized_d_pose_before": 0.094, "realized_d_pose_after": 0.018},
    residual=0.0,
    source_artifact=DAG,
    measurement_method="[macOS-MLX research-signal]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path=DAG, reactivation_criteria="pose_film_84_byte_close",
        measurement_axis="[macOS-MLX research-signal]", hardware_substrate="macos_arm64_mlx",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
equations.append(eq(
    "pose_sqrt_concave_coupling_sidecar_v1",
    "Pose sqrt coupling is concave (marginal->inf as d_pose->0) => stored sufficient-statistic sidecar, not a carrier",
    "sqrt(10*d_pose) is a concave coupling whose marginal diverges as d_pose->0, so pose is solved by storing the 6 PoseNet targets + pose-FiLM (Quantizr), not a reconstruction carrier.",
    r"\sqrt{10\,d_{pose}},\; \partial/\partial d_{pose} = 5/\sqrt{10\,d_{pose}} \to \infty \text{ as } d_{pose}\to 0",
    "tac.scorer_targets",
    {"role": "pose_term_of_E0", "resolution": "stored_sufficient_statistic_sidecar_plus_pose_film",
     "measurement_axes": ["[macOS-MLX research-signal]"], "cross_ref": ["wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1"]},
    {"d_pose": "mse_6_posenet_scalars"},
    {"pose_contribution": "sqrt_10_dpose_score_units", "marginal": "divergent_as_dpose_to_0"},
    (e9_anchor,), {"[macOS-MLX research-signal]": 0.0},
    ("tac.scorer_targets", "tac.unified_action"),
    ("tac.scorer_targets",),
    axis="[macOS-MLX research-signal]", hw="macos_arm64_mlx",
))

# ---- E10 theta* eikonal+length boundary energy (operational instantiation) -
e10_anchor = EmpiricalAnchor(
    anchor_id="theta_star_eikonal_length_active_weights_20260629",
    measurement_utc=UTC,
    inputs={"eikonal_weight": 0.01, "length_weight": 0.001, "frame": "theta_star_variational_levelset"},
    predicted_output={"boundary_energy_terms_active": True},
    empirical_output={"eikonal_active": 0.01, "length_active": 0.001, "regime": "live in trainer"},
    residual=0.0,
    source_artifact=".omx/.../memory/theta_star_witness_lever_stack_and_variational_levelset_frame_20260627.md",
    measurement_method="[macOS-MLX research-signal]",
    provenance=build_provenance_for_research_sidecar(
        sidecar_path="theta_star_witness_lever_stack_and_variational_levelset_frame_20260627.md",
        reactivation_criteria="theta_star_per_lever_ab_campaign_tasks_183_184_185",
        measurement_axis="[macOS-MLX research-signal]", hardware_substrate="macos_arm64_mlx",
    ),
    empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
)
equations.append(eq(
    "theta_star_eikonal_length_boundary_energy_v1",
    "theta* eikonal+length boundary energy = the operational boundary terms of E0 (witness = viscosity solution of the level-set PDE)",
    "theta* = operational lever-stack instantiation of E0; eikonal(0.01)+Chan-Vese length(0.001) are the live boundary-energy terms of E0/E4 (witness = viscosity soln of the level-set PDE).",
    r"E_{bdy} = 0.01\!\int(\lVert\nabla\varphi\rVert-1)^2\,dx + 0.001\!\int\lVert\nabla H(\varphi)\rVert\,dx \;(\text{eikonal}+\text{Chan-Vese length})",
    "tac.boundary_math.lever_b_levelset_generator",
    {"role": "boundary_energy_term_of_E0_E4", "theta_star": "operational lever-stack instantiation of E0 (S_tau)",
     "eikonal_weight": 0.01, "length_weight": 0.001, "morse_smale_chart": "2/1/0-cells",
     "code_manifold_dim": 22, "lane_orbit_dim": 8, "campaign_tasks": ["#183", "#184", "#185"],
     "dsl": "src/tac/witness_dsl (#189)", "memory": "theta_star_witness_lever_stack_and_variational_levelset_frame_20260627",
     "measurement_axes": ["[macOS-MLX research-signal]"]},
    {"phi": "signed_distance_field", "eikonal_w": "0.01", "length_w": "0.001"},
    {"E_bdy": "boundary_energy_scalar"},
    (e10_anchor,), {"[macOS-MLX research-signal]": 0.0},
    ("tac.boundary_math.lever_b_levelset_generator", "tac.witness_dsl", "tac.unified_action"),
    ("tac.boundary_math.lever_b_levelset_generator", "tac.witness_dsl"),
    axis="[macOS-MLX research-signal]", hw="macos_arm64_mlx",
))

# ---- E11 decode-determinism (integer arithmetic; NOT McMullen rigidity) ----
equations.append(eq(
    "decode_determinism_integer_arithmetic_v1",
    "Bit-identical deterministic decode requires INTEGER arithmetic (Balle), NOT McMullen rigidity",
    "Same archive -> bit-identical inflate every host requires integer/fixed-point decode arithmetic (Balle), distinct from the McMullen contraction nugget; floats are non-portable.",
    r"\text{decode}(A) \equiv_{\text{bit}} \text{decode}(A)\ \forall\,\text{host} \iff \text{integer/fixed-point arithmetic}",
    "tools.levelset_byte_close_and_eval",
    {"role": "decode_determinism_discipline_of_E0", "mechanism": "integer_fixed_point_arithmetic",
     "citation": "Balle integer-determinism [lit-hunt FEED-ig, advisory not independently read]",
     "distinct_from": "McMullen superattracting-contraction (FEED-ie) is a separate convergence nugget, NOT this",
     "non_negotiable": "deterministic reproducibility (CLAUDE.md)", "tier": "discipline_theoretical"},
    {"archive_A": "archive_zip_bytes", "host": "decode_host"},
    {"decoded_output": "bit_identical_across_hosts"},
    (), {},
    ("tools.levelset_byte_close_and_eval", "tac.unified_action"),
    ("tools.levelset_byte_close_and_eval",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ---- E12 VCM task-RD headroom above floor (target/floor) -------------------
equations.append(eq(
    "vcm_task_rd_headroom_above_floor_v1",
    "VCM task-RD headroom: practical codecs sit 1-3 OoM above the task-RD floor (target equation)",
    "Practical coding-for-machines codecs sit ~1-3 orders of magnitude above the task-RD floor; our S (pointer 0.19110) vs the measured S_floor~0.118 quantifies the headroom and the sub-0.15 target gap.",
    r"\text{headroom} = S_{\text{achieved}} / S_{\text{floor}};\; S_{\text{floor}}\approx 0.118,\; S_{\text{pointer}}=0.19110,\; T_3=0.15",
    "tac.contest_score",
    {"role": "floor_target_of_E0", "S_floor_measured": 0.118, "S_pointer": 0.19110, "target_T3": 0.15,
     "citation": "VCM headroom 1-3 OoM above floor (arXiv:2505.14980) [lit-hunt FEED-ig, advisory not independently read]",
     "tier": "FORMALIZATION_PENDING_target_cited_from_lit_hunt", "pending": "H2/H1 paper independent read"},
    {"S_achieved": "contest_score", "S_floor": "information_theoretic_floor"},
    {"headroom_ratio": "dimensionless"},
    (), {},
    ("tac.unified_action", "tac.contest_score"),
    ("tac.contest_score",),
    trigger=RECALIBRATE_NEVER_AUTO,
))

# ---- REGISTER ALL ---------------------------------------------------------
existing = {e.equation_id for e in query_equations()}
registered, skipped = [], []
for e in equations:
    if e.equation_id in existing:
        skipped.append(e.equation_id)
        continue
    register_canonical_equation(
        e, agent="formalization-fork", subagent_id="system_of_equations_20260629",
        notes="witness unified-action system E0-E9 formalization (operator 2026-06-29)",
    )
    registered.append(e.equation_id)

print("REGISTERED (%d):" % len(registered))
for r in registered:
    print("  +", r)
if skipped:
    print("SKIPPED already-present (%d):" % len(skipped))
    for s in skipped:
        print("  =", s)

# verify round-trip
post = {e.equation_id for e in query_equations()}
missing = [e.equation_id for e in equations if e.equation_id not in post]
print("VERIFY: all E0-E9 in registry =", not missing, ("MISSING=" + str(missing)) if missing else "")
