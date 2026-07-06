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
from tac.canonical_equations.builtins import (
    build_all_initial_equations,
    populate_initial_equations,
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
from tac.canonical_equations.pairset_component_marginal import (
    build_pairset_component_marginal_score_decomposition_v1,
    pairset_component_marginal_payload,
    pairset_component_marginal_score_delta,
)
from tac.canonical_equations.procedural_predictor_residual_savings import (
    build_procedural_predictor_plus_residual_correction_savings_v1,
    predict_procedural_predictor_plus_residual_correction_savings,
    validate_residual_hybrid_context,
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
from tac.canonical_equations.scorer_input_cache_hash_identity import (
    build_scorer_input_cache_hash_identity_v1,
    scorer_input_cache_hash_identity,
)
from tac.canonical_equations.oom_verdict_batch_spike_peak_rss_20260702 import (
    build_oom_verdict_batch_spike_peak_rss_v1,
    verdict_transient_gib,
)
from tac.canonical_equations.adaptive_ceiling_admission_control_20260703 import (
    adaptive_ceiling_gib,
    admits,
    build_adaptive_ceiling_admission_control_v1,
)
from tac.canonical_equations.powerplay_variant_ii_cost_isomorphism_20260702 import (
    build_powerplay_variant_ii_cost_isomorphism_v1,
    contest_score_as_powerplay_cost,
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
from tac.canonical_equations.leverd_flicker_residual_reactivation_economics_20260703 import (
    build_leverd_flicker_residual_reactivation_economics_v1,
    leverd_break_even_recovery,
    leverd_coder_go,
    leverd_net_delta_s,
    leverd_survival_threshold,
)
from tac.canonical_equations.adaptive_eps_cfl_edge_tracking_20260705 import (
    build_adaptive_eps_cfl_edge_tracking_v1,
    populate_adaptive_eps_cfl_edge_tracking_equation,
)
from tac.canonical_equations.curriculum_derivation_laws_20260705 import (
    ALL_CURRICULUM_DERIVATION_BUILDERS,
    build_curriculum_handoff_critical_nucleus_v1,
    build_ema_window_pi_group_v1,
    build_muon_switch_conditioning_criterion_v1,
    build_rewarmup_beta2_memory_window_v1,
    populate_curriculum_derivation_laws_equations,
)
from tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703 import (
    build_margin_saliency_reachability_replaces_texture_proxy_v1,
    build_muon_finisher_schedule_warmstart_and_lr_anneal_v1,
    muon_cold_start_transition_spike,
    through_r_saliency,
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
from tac.canonical_equations.witness_measured_findings_20260701 import (
    build_all_witness_measured_findings_20260701,
)
from tac.canonical_equations.wyner_ziv_decoder_side_posenet_side_information import (
    build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1,
    predict_wyner_ziv_posenet_side_info_savings,
)

__all__ = [
    "CANONICAL_EQUATIONS_REGISTRY_LOCK",
    "CANONICAL_EQUATIONS_REGISTRY_PATH",
    "CANONICAL_EQUATION_SCHEMA_VERSION",
    "DEFAULT_NIG_PRIOR",
    "EVENT_ANCHOR_APPENDED",
    "EVENT_DEPRECATED",
    "EVENT_DOMAIN_REFINED",
    "EVENT_RECALIBRATED",
    "EVENT_REGISTERED",
    "RECALIBRATE_NEVER_AUTO",
    "RECALIBRATE_ON_NEW_ANCHORS",
    "RECALIBRATE_ON_PARAMETER_REFIT",
    "RECALIBRATE_ON_RESIDUAL_DRIFT",
    "VALID_EVENT_TYPES",
    "VALID_RECALIBRATION_TRIGGERS",
    "BayesianPosterior",
    "CanonicalEquation",
    "CanonicalEquationsRegistryCorruptError",
    "DomainOfValidityViolation",
    "EmpiricalAnchor",
    "ALL_DEEPMATH_AMORTIZING_ARGMAX_BUILDERS",
    "InvalidEquationError",
    "NormalInverseGammaHyperparameters",
    "PosteriorUpdateError",
    "RecalibrationReport",
    "along_tangent_frequency_deficit_ratio",
    "annulus_anisotropy_ratio",
    "annulus_fisher_trace",
    "append_empirical_anchor_to_equation_with_posterior_update",
    "auto_recalibrate_from_continual_learning_posterior",
    "bootstrap_posterior_from_anchor_residuals",
    "boundary_asymmetry_t",
    "build_adaptive_ceiling_admission_control_v1",
    "build_all_initial_equations",
    "build_all_witness_measured_findings_20260701",
    "build_anisotropic_basis_along_tangent_frequency_deficit_v1",
    "build_annulus_anisotropy_magnitude_disputed_v1",
    "build_ce_softmax_mirror_descent_natural_gradient_v1",
    "build_chroma_decides_lane_and_movable_at_annulus_v1",
    "build_contest_R_operator_mtf_allpass_to_2px_v1",
    "build_fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
    "build_independent_flicker_jitter_dseg_floor_smooth_optimal_v1",
    "build_leverd_flicker_residual_reactivation_economics_v1",
    "build_maslov_dequantization_bound_v1",
    "build_mcf_minority_erasure_inevitability_v1",
    "build_mlx_matmul_drift_m_series_canonical_floor_v1",
    "build_mlx_pytorch_drift_equation_from_result_json",
    "build_mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
    "build_oom_verdict_batch_spike_peak_rss_v1",
    "build_pairset_component_marginal_score_decomposition_v1",
    "build_adaptive_eps_cfl_edge_tracking_v1",
    "populate_adaptive_eps_cfl_edge_tracking_equation",
    "ALL_CURRICULUM_DERIVATION_BUILDERS",
    "build_curriculum_handoff_critical_nucleus_v1",
    "build_ema_window_pi_group_v1",
    "build_muon_switch_conditioning_criterion_v1",
    "build_rewarmup_beta2_memory_window_v1",
    "populate_curriculum_derivation_laws_equations",
    "build_margin_saliency_reachability_replaces_texture_proxy_v1",
    "build_multiphase_modica_mortola_perimeter_gamma_limit_v1",
    "build_muon_finisher_schedule_warmstart_and_lr_anneal_v1",
    "build_powerplay_variant_ii_cost_isomorphism_v1",
    "build_procedural_predictor_plus_residual_correction_savings_v1",
    "build_scalar_top1_top2_margin_is_exact_distance_to_flip_v1",
    "build_scorer_input_cache_hash_identity_v1",
    "build_separatrix_asymmetry_t_subpixel_boundary_localizer_v1",
    "build_shearlet_nterm_upper_bounds_task_rate_v1",
    "build_store_nothing_pose_carrier_rate_collapse_vs_dpose_v1",
    "build_task_rd_dominates_reconstruction_rd_v1",
    "build_tau_eps_hbar_one_dequantization_two_scales_v1",
    "build_wyner_ziv_decoder_side_posenet_side_information_conditional_entropy_reduction_v1",
    "categorical_bregman_divergence",
    "categorical_fisher_trace_two_class",
    "chroma_margin_gradient_energy_fraction",
    "classify_mlx_matmul_drift",
    "contest_score_as_powerplay_cost",
    "compute_predicted_band_from_posterior",
    "gap13_minus_gap12_min",
    "get_equation_by_id",
    "herring_triple_junction_angle_deg",
    "independent_jitter_dseg",
    "leverd_break_even_recovery",
    "leverd_coder_go",
    "leverd_net_delta_s",
    "leverd_survival_threshold",
    "load_equation_registry_strict",
    "load_registry_events_lenient",
    "maslov_dequantization_bound",
    "mbo_smoothing_cost_lane_fraction",
    "mlx_pytorch_full_decoder_downstream_scorer_drift_bound",
    "muon_cold_start_transition_spike",
    "pairset_component_marginal_payload",
    "pairset_component_marginal_score_delta",
    "populate_initial_equations",
    "predict_procedural_predictor_plus_residual_correction_savings",
    "predict_wyner_ziv_posenet_side_info_savings",
    "r_mtf_amplitude_at",
    "query_equations",
    "query_equations_by_consumer",
    "query_equations_by_domain",
    "query_equations_by_producer",
    "register_canonical_equation",
    "scorer_input_cache_hash_identity",
    "shearlet_nterm_error",
    "store_nothing_marginal_bytes",
    "store_nothing_rate_term",
    "task_rd_dominance_gap",
    "tau_interface_halfwidth",
    "through_r_saliency",
    "update_equation_with_anchor_via_conjugate_prior",
    "update_equation_with_domain_refinement",
    "update_equation_with_empirical_anchor",
    "validate_residual_hybrid_context",
    "verdict_transient_gib",
]
