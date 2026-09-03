# SPDX-License-Identifier: MIT
"""tac.canonical_equations — typed registry of formalized empirical findings.

Per operator NON-NEGOTIABLE 2026-05-19 verbatim: *"we need to formalize all
of this and canonicalize and operationalize because I am afraid we are
learning but if we don't have systems of equations and models and such we
are just gaining tribal knowledge"*.

The package codifies the session's accumulated empirical findings into a
typed + auditable + operator-callable + auto-recalibrating system of
equations + predictive models. Without this framework, every new finding
becomes orphan knowledge invisible to future agents, cathedral autopilot,
and research subagents.

Quick start:

    from tac.canonical_equations import (
        CanonicalEquation,
        EmpiricalAnchor,
        register_canonical_equation,
        update_equation_with_empirical_anchor,
        query_equations,
        query_equations_by_consumer,
        get_equation_by_id,
        populate_initial_equations,
    )

    # Inspect the registry (returns list of latest-payload-per-equation_id)
    for eq in query_equations():
        print(eq.equation_id, eq.is_well_calibrated, eq.predicted_vs_empirical_residual)

    # Add an empirical anchor (e.g., after a Modal smoke lands)
    update_equation_with_empirical_anchor(
        "mps_drift_architecture_class_dependent_v1",
        anchor_for_segnet_class_validation,
    )

The 6 initial equations are documented in ``builtins.py`` + the landing
memo at ``feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md``.

Cross-references:
  * CLAUDE.md "Canonical equations + models registry — non-negotiable"
  * CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable (typed atom discipline)
  * Catalog #344 STRICT preflight gate (refuses new empirical-finding memos
    without equation reference)
  * Catalog #323 canonical Provenance umbrella (every equation + anchor
    carries Provenance)
  * Catalog #125 6-hook wire-in non-negotiable (hook #5 continual-learning
    posterior auto-recalibrates)
  * Catalog #245/#313 canonical 4-layer pattern this registry follows
"""
from __future__ import annotations

from tac.canonical_equations.adaptive_ceiling_admission_control_20260703 import (
    adaptive_ceiling_gib,
    admits,
    build_adaptive_ceiling_admission_control_v1,
)
from tac.canonical_equations.adaptive_eps_cfl_edge_tracking_20260705 import (
    build_adaptive_eps_cfl_edge_tracking_v1,
    populate_adaptive_eps_cfl_edge_tracking_equation,
)
from tac.canonical_equations.anisotropic_basis_two_regime_allocation_20260707 import (
    build_anisotropic_basis_two_regime_allocation_v1,
    freq_along_for_regime,
    populate_anisotropic_basis_two_regime_allocation_equation,
)
from tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904 import (
    bias_amplification,
    build_annulus_restricted_prefix_bias_detector_v1,
    global_check_is_blind,
    prefix_constant_is_suspect,
)
from tac.canonical_equations.bayesian_posterior_update import (
    DEFAULT_NIG_PRIOR,
    BayesianPosterior,
    NormalInverseGammaHyperparameters,
    PosteriorUpdateError,
    append_empirical_anchor_to_equation_with_posterior_update,
    bootstrap_posterior_from_anchor_residuals,
    compute_predicted_band_from_posterior,
    update_equation_with_anchor_via_conjugate_prior,
)
from tac.canonical_equations.blind_coordinate_rate_lever_20260711 import (
    build_blind_coordinate_rate_lever_v1,
    populate_blind_coordinate_rate_lever_equation,
)
from tac.canonical_equations.boundary_distance_calibration_20260705 import (
    bd_weight_for_ratio,
    build_boundary_distance_weight_calibration_v1,
    populate_boundary_distance_weight_calibration_equation,
)
from tac.canonical_equations.bregman_v9_surfaces_20260714 import (
    bregman_closed_form_dual_cancellation_law,
    bregman_nonnegative_convexity_invariant_law,
    bregman_positive_unscented_propagation_law,
    bregman_right_data_centroid_dual_mean_law,
    build_bregman_cgauge_application_equations_v1,
    build_bregman_closed_form_dual_cancellation_v1,
    build_bregman_dual_metric_squared_hessian_v1,
    build_bregman_nonnegative_convexity_invariant_v1,
    build_bregman_positive_unscented_propagation_v1,
    build_bregman_right_data_centroid_dual_mean_v1,
    build_cgauge_categorical_bregman_hessian_covariance_v1,
    cgauge_categorical_bregman_hessian_covariance_law,
    populate_bregman_cgauge_application_equations_v1,
    populate_bregman_dual_metric_squared_hessian_v1,
)
from tac.canonical_equations.builtins import (
    build_all_initial_equations,
    populate_initial_equations,
)
from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
    area_constraint_lambda,
    build_chan_vese_area_constraint_birth_balance_v1,
    build_isoperimetric_birth_weight_scaling_v1,
    isoperimetric_birth_weight,
    populate_chan_vese_area_constraint_birth_balance_equation,
    populate_isoperimetric_birth_weight_scaling_equation,
)
from tac.canonical_equations.chroma_boundary_match_20260709 import (
    build_chroma_boundary_annulus_match_hinge_v1,
    populate_chroma_boundary_annulus_match_equation,
)
from tac.canonical_equations.compact_shearlet_parabolic_capacity_20260714 import (
    build_compact_shearlet_parabolic_capacity_v1,
    compact_shearlet_sigma_pair,
    compact_shearlet_structural_certificate_law,
    populate_compact_shearlet_parabolic_capacity_v1,
)
from tac.canonical_equations.costate_lambda_marginal_ds_20260705 import (
    build_costate_lambda_marginal_ds_v1,
    chained_ds_depoch,
    costate_vector,
    populate_costate_lambda_marginal_ds_equation,
)
from tac.canonical_equations.curriculum_derivation_laws_20260705 import (
    ALL_CURRICULUM_DERIVATION_BUILDERS,
    build_curriculum_handoff_critical_nucleus_v1,
    build_ema_window_pi_group_v1,
    build_label_floor_to_phase_tail_handoff_v1,
    build_muon_switch_conditioning_criterion_v1,
    build_rewarmup_beta2_memory_window_v1,
    populate_curriculum_derivation_laws_equations,
)
from tac.canonical_equations.dash_erasure_homogenization_20260707 import (
    build_dash_erasure_homogenization_v1,
    populate_dash_erasure_homogenization_equation,
    smoothing_crossover_ok,
)
from tac.canonical_equations.ddm_b2b_rowband_flip_mass_20260731 import (
    build_rowband_flip_mass_foveation_v1,
    flip_band_render_rows,
    populate_rowband_flip_mass_foveation_v1,
)
from tac.canonical_equations.ddm_cf2_token_price_laws_20260821 import (
    ALL_CF2_TOKEN_PRICE_BUILDERS,
    AWAY_TRUST_VS_ACTUAL_PRICE_BAND,
    JG1_LOGIT_RANKER_BITS_PER_TOKEN,
    JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN,
    TOWARD_TRUST_VS_MODEL_U7P75,
    TOWARD_TRUST_VS_MODEL_U12,
    bits_per_token,
    build_greedy_set_average_vs_marginal_price_v1,
    build_token_rate_model_direction_dependence_v1,
    direction_trust_factor,
    greedy_margin_degrades_both_terms,
    marginal_over_average_ratio,
    populate_cf2_token_price_laws,
    predict_realized_bits_per_token,
    predict_reselection_delta_bits,
    ranker_relative_ratio,
    second_order_recapture_fraction,
)
from tac.canonical_equations.ddm_cr1_seg_only_base_pose_degradation_20260801 import (
    build_seg_only_base_pose_degradation_v1,
    populate_seg_only_base_pose_degradation_v1,
    pose_degradation_ratio,
)
from tac.canonical_equations.ddm_cw1_win_family_laws_20260819 import (
    ALL_CW1_WIN_FAMILY_BUILDERS,
    build_container_archive_vs_payload_delta_v1,
    build_gt_lineage_additive_pose_offset_v1,
    build_realized_acceptance_monotonicity_v1,
    container_attributable_bytes,
    populate_cw1_win_family_laws,
    pose_pyav_from_dali,
    realized_acceptance_worsened_count,
)
from tac.canonical_equations.ddm_gc9_seg_rate_product_law_20260730 import (
    build_seg_rate_product_law_v1,
    populate_seg_rate_product_law_v1,
    product_c,
)
from tac.canonical_equations.ddm_lv3_current_arc_laws_20260901 import (
    ALL_LV3_CURRENT_ARC_BUILDERS,
    build_byte_distortion_cross_intersection_count_v1,
    build_context_model_reorder_savings_v1,
    build_decoder_derivable_ideal_savings_ceiling_v1,
    build_field_change_bhw_decomposition_v1,
    build_generator_form_fit_error_entanglement_v1,
    build_roundtrip_token_to_argmax_affine_v1,
    build_same_basin_sharp_optimum_v1,
    populate_lv3_current_arc_laws,
)
from tac.canonical_equations.ddm_v4b_composed_gate_fidelity_20260730 import (
    build_ddm_v4b_composed_gate_instrument_fidelity_v1,
    composed_gate_fidelity,
    populate_ddm_v4b_composed_gate_instrument_fidelity_v1,
)
from tac.canonical_equations.decoder_causal_condition_transport_20260901 import (
    build_decoder_causal_condition_transport_v1,
    populate_decoder_causal_condition_transport_v1,
    receiver_causal_context_is_free,
    transport_floor_bytes,
)
from tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704 import (
    ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS,
    annulus_anisotropy_ratio,
    annulus_fisher_trace,
    build_annulus_anisotropy_magnitude_disputed_v1,
    build_ce_softmax_mirror_descent_natural_gradient_v1,
    build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1,
    build_maslov_dequantization_bound_v1,
    build_mcf_minority_erasure_inevitability_v1,
    build_multiphase_modica_mortola_perimeter_gamma_limit_v1,
    build_shearlet_nterm_upper_bounds_task_rate_v1,
    build_tau_eps_hbar_one_dequantization_two_scales_v1,
    categorical_bregman_divergence,
    categorical_fisher_trace_two_class,
    herring_triple_junction_angle_deg,
    maslov_dequantization_bound,
    mbo_smoothing_cost_lane_fraction,
    shearlet_nterm_error,
    tau_interface_halfwidth,
)
from tac.canonical_equations.defect_network_rate_code_20260712 import (
    build_defect_network_component_delta_rate_v1,
    populate_defect_network_component_delta_rate_v1,
)
from tac.canonical_equations.dseg_aware_fourier_taper_20260709 import (
    build_dseg_aware_fourier_taper_v1,
    populate_dseg_aware_fourier_taper_equation,
)
from tac.canonical_equations.dsl_custodied_scalar_identity_20260717 import (
    build_dsl_custodied_scalar_identity_v1,
    custodied_scalar_identity,
)
from tac.canonical_equations.eikonal_retention_tau_rung_20260713 import (
    build_eikonal_retention_couples_to_tau_rung_v1,
    eikonal_retention_for_rung,
    populate_eikonal_retention_couples_to_tau_rung_v1,
)
from tac.canonical_equations.equation import (
    CANONICAL_EQUATION_SCHEMA_VERSION,
    RECALIBRATE_NEVER_AUTO,
    RECALIBRATE_ON_NEW_ANCHORS,
    RECALIBRATE_ON_PARAMETER_REFIT,
    RECALIBRATE_ON_RESIDUAL_DRIFT,
    VALID_RECALIBRATION_TRIGGERS,
    CanonicalEquation,
    DomainOfValidityViolation,
    EmpiricalAnchor,
    InvalidEquationError,
    delta_exceeds_floor,
)
from tac.canonical_equations.evaluators import (
    LAWREF_BUILTIN_EVALUATORS,
    EvaluatorError,
    EvaluatorNotRegisteredError,
    eval_isoperimetric_birth_weight,
    get_evaluator,
    has_evaluator,
    populate_lawref_evaluators,
    register_evaluator,
    registered_equation_ids,
    resolve_equation_value,
)
from tac.canonical_equations.exchange_ratio_noise_floor_20260903 import (
    build_exchange_ratio_noise_floor_v1,
    near_win_is_admissible,
)
from tac.canonical_equations.focal_gradient_concentration_20260705 import (
    build_focal_gradient_concentration_v1,
    focal_region_share,
    focal_weight_ratio,
    populate_focal_gradient_concentration_equation,
)
from tac.canonical_equations.fullstack_home_assignment_20260710 import (
    build_fullstack_unique_home_assignment_v1,
    populate_fullstack_unique_home_assignment_equation,
)
from tac.canonical_equations.horizon_weighted_margin_20260709 import (
    build_horizon_weighted_margin_v1,
    populate_horizon_weighted_margin_equation,
)
from tac.canonical_equations.hybrid_factorized_costate_adjoint_20260716 import (
    build_hybrid_exact_factorized_costate_adjoint_v1,
    populate_hybrid_exact_factorized_costate_adjoint_equation,
)
from tac.canonical_equations.instant_projected_input_adjoint_20260712 import (
    build_instant_projected_input_adjoint_v1,
    populate_instant_projected_input_adjoint_v1,
)
from tac.canonical_equations.laguerre_ot_head_offset_20260709 import (
    build_laguerre_ot_head_offset_v1,
    populate_laguerre_ot_head_offset_equation,
)
from tac.canonical_equations.lane_band_res_entropy_stage_20260707 import (
    build_lane_band_res_entropy_stage_v1,
    measured_entropy_stage_delta_bytes,
    populate_lane_band_res_entropy_stage_equation,
)
from tac.canonical_equations.lane_dash_residual_root_cause_findings_20260703 import (
    along_tangent_frequency_deficit_ratio,
    boundary_asymmetry_t,
    build_anisotropic_basis_along_tangent_frequency_deficit_v1,
    build_chroma_decides_lane_and_movable_at_annulus_v1,
    build_contest_R_operator_mtf_allpass_to_2px_v1,
    build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1,
    build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1,
    build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1,
    chroma_margin_gradient_energy_fraction,
    gap13_minus_gap12_min,
    independent_jitter_dseg,
    r_mtf_amplitude_at,
)
from tac.canonical_equations.lane_groundframe_xi_transport_no_collapse_20260709 import (
    build_lane_groundframe_xi_transport_no_collapse_v1,
    populate_lane_groundframe_xi_transport_no_collapse_equation,
)
from tac.canonical_equations.leverd_flicker_residual_reactivation_economics_20260703 import (
    build_leverd_flicker_residual_reactivation_economics_v1,
    leverd_break_even_recovery,
    leverd_coder_go,
    leverd_net_delta_s,
    leverd_survival_threshold,
)
from tac.canonical_equations.logit_adjustment_class_prior_20260707 import (
    build_logit_adjustment_class_prior_law_v1,
    logit_adjust_offsets,
    populate_logit_adjustment_class_prior_equation,
)
from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
    build_margin_band_satisficing_threshold_v1,
    margin_safe_threshold,
    minimum_integer_headroom,
    populate_margin_band_satisficing_threshold_equation,
    resolve_margin_band_threshold,
)
from tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703 import (
    build_margin_saliency_reachability_replaces_texture_proxy_v1,
    build_muon_finisher_schedule_warmstart_and_lr_anneal_v1,
    muon_cold_start_transition_spike,
    through_r_saliency,
)
from tac.canonical_equations.mlx_matmul_m_series_floor import (
    build_mlx_matmul_drift_m_series_canonical_floor_v1,
    classify_mlx_matmul_drift,
)
from tac.canonical_equations.mlx_pytorch_drift import (
    build_equation_from_result_json as build_mlx_pytorch_drift_equation_from_result_json,
)
from tac.canonical_equations.mlx_pytorch_drift import (
    build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1,
    mlx_pytorch_full_decoder_downstream_scorer_drift_bound,
)
from tac.canonical_equations.onpolicy_input_costate_surrogate_20260713 import (
    build_onpolicy_input_costate_surrogate_v1,
    populate_onpolicy_input_costate_surrogate_v1,
)
from tac.canonical_equations.oom_verdict_batch_spike_peak_rss_20260702 import (
    build_oom_verdict_batch_spike_peak_rss_v1,
    verdict_transient_gib,
)
from tac.canonical_equations.pairset_component_marginal import (
    build_pairset_component_marginal_score_decomposition_v1,
    pairset_component_marginal_payload,
    pairset_component_marginal_score_delta,
)
from tac.canonical_equations.palette_realization_ceiling_20260710 import (
    build_palette_realization_ceiling_v1,
    populate_palette_realization_ceiling_equation,
)
from tac.canonical_equations.perclass_stratum_carrier_taxonomy_20260716 import (
    build_perclass_stratum_residual_carrier_taxonomy_v1,
    one_sided_carrier_gain,
    populate_perclass_stratum_residual_carrier_taxonomy_equation,
)
from tac.canonical_equations.powerplay_variant_ii_cost_isomorphism_20260702 import (
    build_powerplay_variant_ii_cost_isomorphism_v1,
    contest_score_as_powerplay_cost,
)
from tac.canonical_equations.procedural_predictor_residual_savings import (
    build_procedural_predictor_plus_residual_correction_savings_v1,
    predict_procedural_predictor_plus_residual_correction_savings,
    validate_residual_hybrid_context,
)
from tac.canonical_equations.quadratic_head_chart_subset_solve_gap_20260707 import (
    build_quadratic_head_chart_subset_solve_gap_v1,
    populate_quadratic_head_chart_subset_solve_gap_equation,
    subset_solve_net_transfer,
)
from tac.canonical_equations.realization_necessity_preimage_20260715 import (
    build_realization_necessity_preimage_per_stratum_v1,
    populate_realization_necessity_preimage_equation,
    stratum_rate_floor_bytes,
)
from tac.canonical_equations.registry import (
    CANONICAL_EQUATIONS_REGISTRY_LOCK,
    CANONICAL_EQUATIONS_REGISTRY_PATH,
    EVENT_ANCHOR_APPENDED,
    EVENT_DEPRECATED,
    EVENT_DOMAIN_REFINED,
    EVENT_RECALIBRATED,
    EVENT_REGISTERED,
    VALID_EVENT_TYPES,
    CanonicalEquationsRegistryCorruptError,
    RecalibrationReport,
    auto_recalibrate_from_continual_learning_posterior,
    get_equation_by_id,
    load_equation_registry_strict,
    load_registry_events_lenient,
    query_equations,
    query_equations_by_consumer,
    query_equations_by_domain,
    query_equations_by_producer,
    register_canonical_equation,
    update_equation_with_domain_refinement,
    update_equation_with_empirical_anchor,
)
from tac.canonical_equations.renderer_seg_pose_coupling_20260903 import (
    build_renderer_seg_pose_coupling_shipped_object_v1,
    payable_pose_ceiling,
    seg_only_move_is_payable,
)
from tac.canonical_equations.resize_exploit_flip_fix_frontier_20260709 import (
    build_resize_exploit_flip_fix_frontier_v1,
    populate_resize_exploit_flip_fix_frontier_equation,
)
from tac.canonical_equations.resize_full_kernel_structure_20260720 import (
    build_separable_resize_full_kernel_direct_sum_v1,
    full_resize_kernel_direct_sum,
    populate_separable_resize_full_kernel_direct_sum_equation,
)
from tac.canonical_equations.roadlane_grating_composition_refuted_20260710 import (
    build_roadlane_grating_composition_refuted_v1,
    populate_roadlane_grating_composition_refuted_equation,
)
from tac.canonical_equations.safe_compile_device_bitidentity_20260708 import (
    build_safe_compile_hosc_device_bitidentity_v1,
    populate_safe_compile_device_bitidentity_equation,
)
from tac.canonical_equations.scorer_batch_dependence_20260708 import (
    build_frozen_scorer_forward_batch_dependence_v1,
    populate_frozen_scorer_forward_batch_dependence_equation,
)
from tac.canonical_equations.scorer_input_cache_hash_identity import (
    build_scorer_input_cache_hash_identity_v1,
    scorer_input_cache_hash_identity,
)
from tac.canonical_equations.segnet_exact_forward_cpu_thread_law_20260713 import (
    build_segnet_exact_forward_cpu_thread_control_v1,
    populate_segnet_exact_forward_cpu_thread_control_v1,
)
from tac.canonical_equations.segnet_margin_trust_region_20260713 import (
    build_segnet_margin_trust_region_v1,
    populate_segnet_margin_trust_region_v1,
)
from tac.canonical_equations.sfess_k_subset_cached_replay_20260712 import (
    build_sfess_fixed_k_cached_replay_ranking_v1,
    populate_sfess_fixed_k_cached_replay_equation,
)
from tac.canonical_equations.step_native_activation_edge_optimality_20260707 import (
    build_step_native_activation_edge_optimality_v1,
    hosc_step_limit_beta_ratio,
    populate_step_native_activation_edge_optimality_equation,
)
from tac.canonical_equations.store_nothing_pose_carrier_rate_dpose_20260702 import (
    build_store_nothing_pose_carrier_rate_collapse_vs_dpose_v1,
    store_nothing_marginal_bytes,
    store_nothing_rate_term,
)
from tac.canonical_equations.task_rd_dominates_reconstruction_rd_20260702 import (
    build_task_rd_dominates_reconstruction_rd_v1,
    task_rd_dominance_gap,
)
from tac.canonical_equations.tropnnc_dense_trunk_exact_reduction_20260712 import (
    build_tropnnc_dense_trunk_exact_reduction_empty_v1,
    populate_tropnnc_dense_trunk_exact_reduction_equation,
)
from tac.canonical_equations.v8_geometric_rate_decomposition_20260709 import (
    build_v8_geometric_rate_decomposition_v1,
    populate_v8_geometric_rate_decomposition_equation,
)
from tac.canonical_equations.windowed_curvelet_parabolic_capacity_20260714 import (
    build_windowed_curvelet_parabolic_capacity_v1,
    parabolic_sigma_pair,
    populate_windowed_curvelet_parabolic_capacity_equation,
)
from tac.canonical_equations.witness_measured_findings_20260701 import (
    build_all_witness_measured_findings_20260701,
)
from tac.canonical_equations.witness_own_residual_decomposition_20260716 import (
    build_witness_own_residual_decomposition_v1,
    flat_band_gain_on_witness,
    populate_witness_own_residual_decomposition_equation,
)
from tac.canonical_equations.witness_pose_grad_coeff_stability_20260709 import (
    build_witness_pose_grad_coeff_stability_v1,
    populate_witness_pose_grad_coeff_stability_equation,
)
from tac.canonical_equations.wyner_ziv_decoder_side_posenet_side_information import (
    build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1,
    predict_wyner_ziv_posenet_side_info_savings,
)
from tac.canonical_equations.yopo_first_layer_costate_20260712 import (
    build_yopo_first_layer_costate_v1,
    populate_yopo_first_layer_costate_v1,
)

__all__ = [
    "ALL_CF2_TOKEN_PRICE_BUILDERS",
    "ALL_CURRICULUM_DERIVATION_BUILDERS",
    "ALL_CW1_WIN_FAMILY_BUILDERS",
    "ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS",
    "ALL_LV3_CURRENT_ARC_BUILDERS",
    "AWAY_TRUST_VS_ACTUAL_PRICE_BAND",
    "bias_amplification",
    "build_annulus_restricted_prefix_bias_detector_v1",
    "build_renderer_seg_pose_coupling_shipped_object_v1",
    "CANONICAL_EQUATIONS_REGISTRY_LOCK",
    "CANONICAL_EQUATIONS_REGISTRY_PATH",
    "CANONICAL_EQUATION_SCHEMA_VERSION",
    "DEFAULT_NIG_PRIOR",
    "EVENT_ANCHOR_APPENDED",
    "EVENT_DEPRECATED",
    "EVENT_DOMAIN_REFINED",
    "EVENT_RECALIBRATED",
    "EVENT_REGISTERED",
    "global_check_is_blind",
    "JG1_LOGIT_RANKER_BITS_PER_TOKEN",
    "JG3_ACTUAL_FLAT_PRICE_BITS_PER_TOKEN",
    "LAWREF_BUILTIN_EVALUATORS",
    "payable_pose_ceiling",
    "prefix_constant_is_suspect",
    "RECALIBRATE_NEVER_AUTO",
    "RECALIBRATE_ON_NEW_ANCHORS",
    "RECALIBRATE_ON_PARAMETER_REFIT",
    "RECALIBRATE_ON_RESIDUAL_DRIFT",
    "seg_only_move_is_payable",
    "TOWARD_TRUST_VS_MODEL_U7P75",
    "TOWARD_TRUST_VS_MODEL_U12",
    "VALID_EVENT_TYPES",
    "VALID_RECALIBRATION_TRIGGERS",
    "BayesianPosterior",
    "CanonicalEquation",
    "CanonicalEquationsRegistryCorruptError",
    "DomainOfValidityViolation",
    "EmpiricalAnchor",
    "EvaluatorError",
    "EvaluatorNotRegisteredError",
    "InvalidEquationError",
    "NormalInverseGammaHyperparameters",
    "PosteriorUpdateError",
    "RecalibrationReport",
    "adaptive_ceiling_gib",
    "admits",
    "along_tangent_frequency_deficit_ratio",
    "annulus_anisotropy_ratio",
    "annulus_fisher_trace",
    "append_empirical_anchor_to_equation_with_posterior_update",
    "area_constraint_lambda",
    "auto_recalibrate_from_continual_learning_posterior",
    "bd_weight_for_ratio",
    "bits_per_token",
    "bootstrap_posterior_from_anchor_residuals",
    "boundary_asymmetry_t",
    "bregman_closed_form_dual_cancellation_law",
    "bregman_nonnegative_convexity_invariant_law",
    "bregman_positive_unscented_propagation_law",
    "bregman_right_data_centroid_dual_mean_law",
    "build_adaptive_ceiling_admission_control_v1",
    "build_byte_distortion_cross_intersection_count_v1",
    "build_context_model_reorder_savings_v1",
    "build_decoder_derivable_ideal_savings_ceiling_v1",
    "build_field_change_bhw_decomposition_v1",
    "build_generator_form_fit_error_entanglement_v1",
    "build_roundtrip_token_to_argmax_affine_v1",
    "build_same_basin_sharp_optimum_v1",
    "build_adaptive_eps_cfl_edge_tracking_v1",
    "build_all_initial_equations",
    "build_all_witness_measured_findings_20260701",
    "build_anisotropic_basis_along_tangent_frequency_deficit_v1",
    "build_anisotropic_basis_two_regime_allocation_v1",
    "build_annulus_anisotropy_magnitude_disputed_v1",
    "build_blind_coordinate_rate_lever_v1",
    "build_boundary_distance_weight_calibration_v1",
    "build_bregman_cgauge_application_equations_v1",
    "build_bregman_closed_form_dual_cancellation_v1",
    "build_bregman_dual_metric_squared_hessian_v1",
    "build_bregman_nonnegative_convexity_invariant_v1",
    "build_bregman_positive_unscented_propagation_v1",
    "build_bregman_right_data_centroid_dual_mean_v1",
    "build_ce_softmax_mirror_descent_natural_gradient_v1",
    "build_cgauge_categorical_bregman_hessian_covariance_v1",
    "build_chan_vese_area_constraint_birth_balance_v1",
    "build_chroma_boundary_annulus_match_hinge_v1",
    "build_chroma_decides_lane_and_movable_at_annulus_v1",
    "build_compact_shearlet_parabolic_capacity_v1",
    "build_contest_R_operator_mtf_allpass_to_2px_v1",
    "build_costate_lambda_marginal_ds_v1",
    "build_curriculum_handoff_critical_nucleus_v1",
    "build_dash_erasure_homogenization_v1",
    "build_decoder_causal_condition_transport_v1",
    "build_defect_network_component_delta_rate_v1",
    "build_dseg_aware_fourier_taper_v1",
    "build_dsl_custodied_scalar_identity_v1",
    "build_eikonal_retention_couples_to_tau_rung_v1",
    "build_ema_window_pi_group_v1",
    "build_exchange_ratio_noise_floor_v1",
    "build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
    "build_focal_gradient_concentration_v1",
    "build_frozen_scorer_forward_batch_dependence_v1",
    "build_fullstack_unique_home_assignment_v1",
    "build_greedy_set_average_vs_marginal_price_v1",
    "build_horizon_weighted_margin_v1",
    "build_hybrid_exact_factorized_costate_adjoint_v1",
    "build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1",
    "build_instant_projected_input_adjoint_v1",
    "build_isoperimetric_birth_weight_scaling_v1",
    "build_label_floor_to_phase_tail_handoff_v1",
    "build_laguerre_ot_head_offset_v1",
    "build_lane_band_res_entropy_stage_v1",
    "build_lane_groundframe_xi_transport_no_collapse_v1",
    "build_leverd_flicker_residual_reactivation_economics_v1",
    "build_logit_adjustment_class_prior_law_v1",
    "build_margin_band_satisficing_threshold_v1",
    "build_margin_saliency_reachability_replaces_texture_proxy_v1",
    "build_maslov_dequantization_bound_v1",
    "build_mcf_minority_erasure_inevitability_v1",
    "build_mlx_matmul_drift_m_series_canonical_floor_v1",
    "build_mlx_pytorch_drift_equation_from_result_json",
    "build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
    "build_multiphase_modica_mortola_perimeter_gamma_limit_v1",
    "build_muon_finisher_schedule_warmstart_and_lr_anneal_v1",
    "build_muon_switch_conditioning_criterion_v1",
    "build_onpolicy_input_costate_surrogate_v1",
    "build_oom_verdict_batch_spike_peak_rss_v1",
    "build_pairset_component_marginal_score_decomposition_v1",
    "build_palette_realization_ceiling_v1",
    "build_perclass_stratum_residual_carrier_taxonomy_v1",
    "build_powerplay_variant_ii_cost_isomorphism_v1",
    "build_procedural_predictor_plus_residual_correction_savings_v1",
    "build_quadratic_head_chart_subset_solve_gap_v1",
    "build_realization_necessity_preimage_per_stratum_v1",
    "build_resize_exploit_flip_fix_frontier_v1",
    "build_rewarmup_beta2_memory_window_v1",
    "build_roadlane_grating_composition_refuted_v1",
    "build_safe_compile_hosc_device_bitidentity_v1",
    "build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1",
    "build_scorer_input_cache_hash_identity_v1",
    "build_segnet_exact_forward_cpu_thread_control_v1",
    "build_segnet_margin_trust_region_v1",
    "build_separable_resize_full_kernel_direct_sum_v1",
    "build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1",
    "build_sfess_fixed_k_cached_replay_ranking_v1",
    "build_shearlet_nterm_upper_bounds_task_rate_v1",
    "build_step_native_activation_edge_optimality_v1",
    "build_store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
    "build_task_rd_dominates_reconstruction_rd_v1",
    "build_tau_eps_hbar_one_dequantization_two_scales_v1",
    "build_token_rate_model_direction_dependence_v1",
    "build_tropnnc_dense_trunk_exact_reduction_empty_v1",
    "build_v8_geometric_rate_decomposition_v1",
    "build_windowed_curvelet_parabolic_capacity_v1",
    "build_witness_own_residual_decomposition_v1",
    "build_witness_pose_grad_coeff_stability_v1",
    "build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1",
    "build_yopo_first_layer_costate_v1",
    "categorical_bregman_divergence",
    "categorical_fisher_trace_two_class",
    "cgauge_categorical_bregman_hessian_covariance_law",
    "chained_ds_depoch",
    "chroma_margin_gradient_energy_fraction",
    "classify_mlx_matmul_drift",
    "compact_shearlet_sigma_pair",
    "compact_shearlet_structural_certificate_law",
    "compute_predicted_band_from_posterior",
    "contest_score_as_powerplay_cost",
    "costate_vector",
    "custodied_scalar_identity",
    "delta_exceeds_floor",
    "direction_trust_factor",
    "eikonal_retention_for_rung",
    "eval_isoperimetric_birth_weight",
    "flat_band_gain_on_witness",
    "focal_region_share",
    "focal_weight_ratio",
    "freq_along_for_regime",
    "full_resize_kernel_direct_sum",
    "gap13_minus_gap12_min",
    "get_equation_by_id",
    "get_evaluator",
    "greedy_margin_degrades_both_terms",
    "has_evaluator",
    "herring_triple_junction_angle_deg",
    "hosc_step_limit_beta_ratio",
    "independent_jitter_dseg",
    "isoperimetric_birth_weight",
    "leverd_break_even_recovery",
    "leverd_coder_go",
    "leverd_net_delta_s",
    "leverd_survival_threshold",
    "load_equation_registry_strict",
    "load_registry_events_lenient",
    "logit_adjust_offsets",
    "margin_safe_threshold",
    "marginal_over_average_ratio",
    "maslov_dequantization_bound",
    "mbo_smoothing_cost_lane_fraction",
    "measured_entropy_stage_delta_bytes",
    "minimum_integer_headroom",
    "mlx_pytorch_full_decoder_downstream_scorer_drift_bound",
    "muon_cold_start_transition_spike",
    "near_win_is_admissible",
    "one_sided_carrier_gain",
    "pairset_component_marginal_payload",
    "pairset_component_marginal_score_delta",
    "parabolic_sigma_pair",
    "populate_adaptive_eps_cfl_edge_tracking_equation",
    "populate_anisotropic_basis_two_regime_allocation_equation",
    "populate_blind_coordinate_rate_lever_equation",
    "populate_boundary_distance_weight_calibration_equation",
    "populate_bregman_cgauge_application_equations_v1",
    "populate_bregman_dual_metric_squared_hessian_v1",
    "populate_cf2_token_price_laws",
    "populate_chan_vese_area_constraint_birth_balance_equation",
    "populate_chroma_boundary_annulus_match_equation",
    "populate_compact_shearlet_parabolic_capacity_v1",
    "populate_costate_lambda_marginal_ds_equation",
    "populate_curriculum_derivation_laws_equations",
    "populate_dash_erasure_homogenization_equation",
    "populate_decoder_causal_condition_transport_v1",
    "populate_defect_network_component_delta_rate_v1",
    "populate_dseg_aware_fourier_taper_equation",
    "populate_eikonal_retention_couples_to_tau_rung_v1",
    "populate_focal_gradient_concentration_equation",
    "populate_frozen_scorer_forward_batch_dependence_equation",
    "populate_fullstack_unique_home_assignment_equation",
    "populate_horizon_weighted_margin_equation",
    "populate_hybrid_exact_factorized_costate_adjoint_equation",
    "populate_initial_equations",
    "populate_instant_projected_input_adjoint_v1",
    "populate_isoperimetric_birth_weight_scaling_equation",
    "populate_laguerre_ot_head_offset_equation",
    "populate_lane_band_res_entropy_stage_equation",
    "populate_lane_groundframe_xi_transport_no_collapse_equation",
    "populate_lawref_evaluators",
    "populate_logit_adjustment_class_prior_equation",
    "populate_lv3_current_arc_laws",
    "populate_margin_band_satisficing_threshold_equation",
    "populate_onpolicy_input_costate_surrogate_v1",
    "populate_palette_realization_ceiling_equation",
    "populate_perclass_stratum_residual_carrier_taxonomy_equation",
    "populate_quadratic_head_chart_subset_solve_gap_equation",
    "populate_realization_necessity_preimage_equation",
    "populate_resize_exploit_flip_fix_frontier_equation",
    "populate_roadlane_grating_composition_refuted_equation",
    "populate_safe_compile_device_bitidentity_equation",
    "populate_segnet_exact_forward_cpu_thread_control_v1",
    "populate_segnet_margin_trust_region_v1",
    "populate_separable_resize_full_kernel_direct_sum_equation",
    "populate_sfess_fixed_k_cached_replay_equation",
    "populate_step_native_activation_edge_optimality_equation",
    "populate_tropnnc_dense_trunk_exact_reduction_equation",
    "populate_v8_geometric_rate_decomposition_equation",
    "populate_windowed_curvelet_parabolic_capacity_equation",
    "populate_witness_own_residual_decomposition_equation",
    "populate_witness_pose_grad_coeff_stability_equation",
    "populate_yopo_first_layer_costate_v1",
    "predict_procedural_predictor_plus_residual_correction_savings",
    "predict_realized_bits_per_token",
    "predict_reselection_delta_bits",
    "predict_wyner_ziv_posenet_side_info_savings",
    "query_equations",
    "query_equations_by_consumer",
    "query_equations_by_domain",
    "query_equations_by_producer",
    "r_mtf_amplitude_at",
    "ranker_relative_ratio",
    "register_canonical_equation",
    "register_evaluator",
    "registered_equation_ids",
    "receiver_causal_context_is_free",
    "resolve_equation_value",
    "resolve_margin_band_threshold",
    "scorer_input_cache_hash_identity",
    "second_order_recapture_fraction",
    "shearlet_nterm_error",
    "smoothing_crossover_ok",
    "store_nothing_marginal_bytes",
    "store_nothing_rate_term",
    "stratum_rate_floor_bytes",
    "subset_solve_net_transfer",
    "task_rd_dominance_gap",
    "tau_interface_halfwidth",
    "through_r_saliency",
    "transport_floor_bytes",
    "update_equation_with_anchor_via_conjugate_prior",
    "update_equation_with_domain_refinement",
    "update_equation_with_empirical_anchor",
    "validate_residual_hybrid_context",
    "verdict_transient_gib",
]
