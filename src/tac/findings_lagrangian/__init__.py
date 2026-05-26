# SPDX-License-Identifier: MIT
"""tac.findings_lagrangian — 4-term scalar findings Lagrangian (TRACK A; hand-rolled Gaussian).

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19) + operator-frontier-override 2026-05-19
verbatim *"we shoud pursue PP in parallel"*:

This package implements TRACK A of the dual-track build per slot 20 Q5
binding decision (RATIFIED across 3 rounds): closed-form Gaussian posteriors
+ scipy.stats.multivariate_normal sampling + ZERO new PP framework
dependency. The sister TRACK B (NumPyro hierarchical posteriors) lives at
``tac.findings_lagrangian_pp``; both tracks emit predictions through the
canonical ``tac.findings_lagrangian.unified.UnifiedPrediction`` interface.

The 4 Lagrangian terms (per slot 20 + supplemental Q3 amendment):

    L = data_fit
      + lambda_Occam_complexity * complexity_penalty
      + lambda_Occam_interpretability * interpretability_penalty
      + lambda_partition * partition_penalty
      - mu_explore * E[KL info gain from hypothetical experiment]

Quick start::

    from tac.findings_lagrangian import (
        GaussianPosterior, posterior_update_from_anchors,
        kl_divergence_gaussians, expected_information_gain,
        FindingsLagrangianWeights, default_initial_weights,
        CanonicalPartition, build_initial_partition,
        compute_findings_lagrangian, FindingsLagrangianResult,
        ScalarPrediction, UnifiedPrediction,
        recommend_next_action_via_expected_information_gain,
    )

    # Update posterior with empirical residuals
    posterior = posterior_update_from_anchors(
        "mps_drift_architecture_class_dependent_v1",
        prior_mu=(1.0,),
        prior_sigma_diagonal=(0.5,),
        anchor_residuals=[0.03, 0.05, -0.02],
    )

    # Compute Lagrangian for the equation
    result = compute_findings_lagrangian(
        "mps_drift_architecture_class_dependent_v1",
        posterior=posterior,
        partition=build_initial_partition(),
        anchor_residuals=[0.03, 0.05, -0.02],
    )

    # Recommend next action via expected info gain per dollar
    recommendation = recommend_next_action_via_expected_information_gain(
        [candidate_action_1, candidate_action_2, ...],
        posteriors_by_equation_id={"mps_drift_architecture_class_dependent_v1": posterior},
        budget_usd=5.0,
    )

Cross-references:
- CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable (this IS the
  living solver for the FINDINGS surface)
- CLAUDE.md "Canonical equations + models registry" non-negotiable
  (``tac.canonical_equations`` is the substrate this Lagrangian operates on)
- Slot 20 + supplemental + second-supplemental council memos under
  ``.omx/research/grand_council_t3_*findings_lagrangian*.md``
- Operator-frontier-override at
  ``.omx/research/operator_authorizations/findings_lagrangian_pp_parallel_pursuit_plus_all_voices_matter_override_20260519T080000Z.md``
- Catalog #347 STRICT preflight gate (extends Catalog #344;
  accepts findings_lagrangian reference in empirical-finding memos)
- Catalog #125 6-hook wire-in (ALL 6 hooks ACTIVE per landing memo)
- Catalog #287 + #323 (canonical [predicted] axis tag + Provenance)
- Sister ``tac.findings_lagrangian_pp`` TRACK B per operator override
- Sister ``tac.cathedral_consumers.findings_lagrangian_consumer`` per
  Catalog #335 canonical consumer contract
"""
from __future__ import annotations

from tac.findings_lagrangian.posterior import (
    GaussianPosterior,
    PosteriorInvalidError,
    posterior_update_from_anchors,
    posterior_sample,
    posterior_predict,
)
from tac.findings_lagrangian.info_gain import (
    kl_divergence_gaussians,
    expected_information_gain,
    monte_carlo_kl_fallback,
)
from tac.findings_lagrangian.weights import (
    FindingsLagrangianWeights,
    default_initial_weights,
    adapt_weights_from_residuals,
    WEIGHT_BUDGET_TOTAL,
    MU_EXPLORE_CAP,
    LAMBDA_OCCAM_COMPLEXITY_INITIAL,
    LAMBDA_OCCAM_INTERPRETABILITY_INITIAL,
    LAMBDA_PARTITION_INITIAL,
    MU_EXPLORE_INITIAL,
    ADAPTIVE_FACTOR,
    RESIDUAL_DRIFT_THRESHOLD_SIGMA,
    WeightInvariantError,
)
from tac.findings_lagrangian.partition import (
    CanonicalPartition,
    PartitionClass,
    PartitionInvalidError,
    INITIAL_4_CLASS_CASCADE_TAXONOMY,
    INITIAL_PARTITION_ID,
    MDL_SPLIT_THRESHOLD_BITS,
    WAVELET_DEPTH_WEIGHT_BASE,
    compute_mdl_gain_with_wavelet_prior,
    should_split_class,
    explain_split,
    build_initial_partition,
)
from tac.findings_lagrangian.interpretability import (
    FallingRule,
    FallingRuleListExplanation,
    build_falling_rule_list_for_downweight,
    explain_decision_per_candidate,
    compute_downweight_factor,
    MAX_RULES_PER_EXPLANATION,
    InterpretabilityError,
)
from tac.findings_lagrangian.lagrangian import (
    FindingsLagrangianResult,
    compute_findings_lagrangian,
    decompose_lagrangian,
    LagrangianInvalidError,
)
from tac.findings_lagrangian.action_selector import (
    CandidateAction,
    ActionRecommendation,
    recommend_next_action_via_expected_information_gain,
    ActionSelectorError,
)
from tac.findings_lagrangian.unified import (
    ScalarPrediction,
    UnifiedPrediction,
    ensemble_prediction_from_tracks,
    EnsembleError,
)
from tac.findings_lagrangian.dual_solver_phase_2 import (
    PerAxisDualSolverResult,
    Phase2SolverError,
    compute_per_axis_dual_variables,
    dykstra_alternating_projections_3_axis,
    kkt_residuals_per_axis,
    per_axis_adjustment_factors,
    PHASE_2_DUAL_SOLVER_SCHEMA_VERSION,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN,
    PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX,
    DYKSTRA_DEFAULT_MAX_ITERATIONS,
    DYKSTRA_DEFAULT_EPSILON,
    AXIS_NAMES,
    MLX_AVAILABLE,
)


__all__ = [
    # posterior
    "GaussianPosterior",
    "PosteriorInvalidError",
    "posterior_update_from_anchors",
    "posterior_sample",
    "posterior_predict",
    # info_gain
    "kl_divergence_gaussians",
    "expected_information_gain",
    "monte_carlo_kl_fallback",
    # weights
    "FindingsLagrangianWeights",
    "default_initial_weights",
    "adapt_weights_from_residuals",
    "WEIGHT_BUDGET_TOTAL",
    "MU_EXPLORE_CAP",
    "LAMBDA_OCCAM_COMPLEXITY_INITIAL",
    "LAMBDA_OCCAM_INTERPRETABILITY_INITIAL",
    "LAMBDA_PARTITION_INITIAL",
    "MU_EXPLORE_INITIAL",
    "ADAPTIVE_FACTOR",
    "RESIDUAL_DRIFT_THRESHOLD_SIGMA",
    "WeightInvariantError",
    # partition
    "CanonicalPartition",
    "PartitionClass",
    "PartitionInvalidError",
    "INITIAL_4_CLASS_CASCADE_TAXONOMY",
    "INITIAL_PARTITION_ID",
    "MDL_SPLIT_THRESHOLD_BITS",
    "WAVELET_DEPTH_WEIGHT_BASE",
    "compute_mdl_gain_with_wavelet_prior",
    "should_split_class",
    "explain_split",
    "build_initial_partition",
    # interpretability
    "FallingRule",
    "FallingRuleListExplanation",
    "build_falling_rule_list_for_downweight",
    "explain_decision_per_candidate",
    "compute_downweight_factor",
    "MAX_RULES_PER_EXPLANATION",
    "InterpretabilityError",
    # lagrangian
    "FindingsLagrangianResult",
    "compute_findings_lagrangian",
    "decompose_lagrangian",
    "LagrangianInvalidError",
    # action_selector
    "CandidateAction",
    "ActionRecommendation",
    "recommend_next_action_via_expected_information_gain",
    "ActionSelectorError",
    # unified
    "ScalarPrediction",
    "UnifiedPrediction",
    "ensemble_prediction_from_tracks",
    "EnsembleError",
    # dual_solver_phase_2
    "PerAxisDualSolverResult",
    "Phase2SolverError",
    "compute_per_axis_dual_variables",
    "dykstra_alternating_projections_3_axis",
    "kkt_residuals_per_axis",
    "per_axis_adjustment_factors",
    "PHASE_2_DUAL_SOLVER_SCHEMA_VERSION",
    "PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN",
    "PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX",
    "DYKSTRA_DEFAULT_MAX_ITERATIONS",
    "DYKSTRA_DEFAULT_EPSILON",
    "AXIS_NAMES",
    "MLX_AVAILABLE",
]
