# SPDX-License-Identifier: MIT
"""Canonical contest-oracle package -- operationalizing the 15 implications.

The 5 contest_fixed values (rate-term denominator, sqrt(10) pose coefficient,
5-class SegNet, 600 pairs, 384x512 resolution) are NOT constraints to respect
-- they are ORACLES to mine. Every constant embeds a piece of the closed-form
analytical solution that we would otherwise have to learn via PATH 4 (neural
surrogate). The contest GIVES US the gradient; this package surfaces it as
canonical typed helpers.

Per the operator standing directive 2026-05-18 verbatim
*"the contest information defines the problem and the solution... or the
path to the solution"* + the design memo
``.omx/research/contest_fixed_as_oracles_15_implications_design_memo_20260518.md``.

The package operationalizes 14 implications + the META principle (the 15th):

| # | Module                                | Implication                                       |
|---|----------------------------------------|--------------------------------------------------|
| 1 | ``gradient``                          | Contest formula IS the gradient oracle           |
| 2 | ``phase_classifier``                  | Phase-transition map across operating points     |
| 3 | ``pareto_frontier``                   | Closed-form Pareto frontier                      |
| 4 | ``per_pair_decomposition``            | 600-pair additive decomposition                  |
| 5 | ``per_class_lagrangian``              | 5-class imbalance-corrected lambda               |
| 6+7 | ``pose_axis_canonical``             | sqrt(10) pose curvature + frontier-free EV       |
| 8 | ``theoretical_floor``                 | Closed-form theoretical floor                    |
| 9 | ``pixel_budget_allocator``            | Pixel-budget allocator (foveation sister)        |
| 10| ``cell_allocator``                    | 588M-cell sparse water-filling                   |
| 11| ``score_predictor``                   | Differentiable contest-formula teacher           |
| 12| ``substrate_alignment``               | Substrate-shape contest-alignment score          |
| 13| ``bandit_per_pair``                   | Bandit-optimal per-pair Thompson sampling        |
| 14| ``arithmetic_coder_class_conditional``| Class-conditional CDF priors                     |

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125 6-hook
wire-in: every public helper EITHER feeds canonical infrastructure
(sensitivity_map / pareto_constraint / bit_allocator / cathedral_autopilot_dispatch
/ continual_learning_posterior / probe_disambiguator) OR carries a documented
opt-out rationale in its module docstring.

Per CLAUDE.md "Max observability" + Catalog #305: every module declares
its observability surface in the module docstring (subset of the canonical
6: inspectable_per_layer / decomposable_per_signal / diff_able_across_runs
/ queryable_post_hoc / cite_able / counterfactual_able).

Composition-not-duplication discipline (per design memo APPENDIX A): only
Impl 2 / 5 / 12 are genuinely missing; the other 11 implications are
canonical-alias / extend / compose / wire-in. Total NEW core LOC ~ 1300
across the 14 modules + constants + __init__ (under the 2200 LOC budget).
"""
from __future__ import annotations

# --- constants ---
from .constants import (
    CONTEST_INPUT_HEIGHT,
    CONTEST_INPUT_WIDTH,
    CONTEST_NUM_PAIRS,
    CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
    CONTEST_PER_ARCHIVE_PIXEL_CELLS,
    CONTEST_PIXELS_PER_FRAME,
    CONTEST_POSE_SQRT_INNER,
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_RATE_DENOM_BYTES,
    CONTEST_RATE_PER_BYTE,
    CONTEST_RATE_WEIGHT,
    CONTEST_SEG_WEIGHT,
    SCORE_AXIS_LABELS,
    SEGNET_NUM_CLASSES,
)

# --- Impl 1: gradient ---
from .gradient import (
    ContestGradientError,
    ContestScoreGradient,
    compute_score,
    compute_score_gradient,
)

# --- Impl 2: phase classifier ---
from .phase_classifier import (
    CROSSOVER_POSE_AVG,
    PHASE_BOUNDARY_CROSSOVER_TO_FRONTIER,
    PHASE_BOUNDARY_MID_TO_CROSSOVER,
    PHASE_BOUNDARY_OLD_1X_TO_MID,
    ContestPhase,
    OptimalAttackRecommendation,
    PhaseClassification,
    classify_phase,
    recommend_attack,
)

# --- Impl 3: Pareto frontier ---
from .pareto_frontier import (
    AnalyticalParetoPoint,
    analytical_optimum,
    trace_pareto_frontier,
)

# --- Impl 4 + 13: per-pair decomposition + bandit alias ---
from .per_pair_decomposition import (
    CANONICAL_PER_PAIR_PLAN_AVAILABLE,
    PerPairThompsonSamplingPlan,
    per_pair_optimal_treatment_plan,
    thompson_sample_per_pair_assignment,
)

# --- Impl 5: per-class Lagrangian ---
from .per_class_lagrangian import (
    DEFAULT_EFFECTIVE_NUMBER_BETA,
    PerClassLagrangianError,
    PerClassLambdaSeg,
    apply_per_class_lambda_to_seg_loss,
    compute_per_class_lambda_seg,
)

# --- Impl 6 + 7: pose-axis canonical ---
from .pose_axis_canonical import (
    PoseAxisAnalysis,
    PoseAxisError,
    analyze_pose_axis,
    contest_curvature_pose_loss,
)

# --- Impl 8: theoretical floor ---
from .theoretical_floor import (
    CANONICAL_BLAHUT_AVAILABLE,
    compute_contest_floor,
)

# --- Impl 9: pixel-budget allocator ---
from .pixel_budget_allocator import (
    CANONICAL_INTERNAL_RESOLUTION_MULTIPLES,
    PixelBudgetRecommendation,
    recommend_internal_resolution,
)

# --- Impl 10: 588M-cell sparse water-filling ---
from .cell_allocator import (
    CellAllocation,
    CellAllocatorError,
    SparseWaterFillingAllocation,
    sparse_water_fill,
)

# --- Impl 11: differentiable score predictor ---
from .score_predictor import (
    ContestScorePrediction,
    build_contest_action,
    predict_score,
    validate_against_canonical_formula,
)

# --- Impl 12: substrate alignment ---
from .substrate_alignment import (
    PR101_GOLD_ALIGNED_FACETS,
    AlignmentFacet,
    SubstrateAlignmentScore,
    pr101_gold_reference,
    score_substrate_alignment,
)

# --- Impl 13: per-pair bandit ---
from .bandit_per_pair import (
    BanditError,
    BetaBernoulliPosterior,
    PerPairBanditAssignment,
    PerPairBanditPlan,
    thompson_sample_per_pair_with_posterior,
    update_beta_bernoulli_posterior,
)

# --- Impl 14: class-conditional CDF ---
from .arithmetic_coder_class_conditional import (
    ArithmeticCoderError,
    ClassConditionalCodebook,
    ClassConditionalPrior,
    build_class_conditional_codebook,
)

__all__ = [
    # constants
    "CONTEST_INPUT_HEIGHT",
    "CONTEST_INPUT_WIDTH",
    "CONTEST_NUM_PAIRS",
    "CONTEST_PER_ARCHIVE_PER_CLASS_CELLS",
    "CONTEST_PER_ARCHIVE_PIXEL_CELLS",
    "CONTEST_PIXELS_PER_FRAME",
    "CONTEST_POSE_SQRT_INNER",
    "CONTEST_POSE_SQRT_WEIGHT",
    "CONTEST_RATE_DENOM_BYTES",
    "CONTEST_RATE_PER_BYTE",
    "CONTEST_RATE_WEIGHT",
    "CONTEST_SEG_WEIGHT",
    "SCORE_AXIS_LABELS",
    "SEGNET_NUM_CLASSES",
    # gradient (Impl 1)
    "ContestGradientError",
    "ContestScoreGradient",
    "compute_score",
    "compute_score_gradient",
    # phase classifier (Impl 2)
    "CROSSOVER_POSE_AVG",
    "PHASE_BOUNDARY_CROSSOVER_TO_FRONTIER",
    "PHASE_BOUNDARY_MID_TO_CROSSOVER",
    "PHASE_BOUNDARY_OLD_1X_TO_MID",
    "ContestPhase",
    "OptimalAttackRecommendation",
    "PhaseClassification",
    "classify_phase",
    "recommend_attack",
    # Pareto frontier (Impl 3)
    "AnalyticalParetoPoint",
    "analytical_optimum",
    "trace_pareto_frontier",
    # per-pair decomposition (Impl 4)
    "CANONICAL_PER_PAIR_PLAN_AVAILABLE",
    "PerPairThompsonSamplingPlan",
    "per_pair_optimal_treatment_plan",
    "thompson_sample_per_pair_assignment",
    # per-class Lagrangian (Impl 5)
    "DEFAULT_EFFECTIVE_NUMBER_BETA",
    "PerClassLagrangianError",
    "PerClassLambdaSeg",
    "apply_per_class_lambda_to_seg_loss",
    "compute_per_class_lambda_seg",
    # pose axis (Impl 6 + 7)
    "PoseAxisAnalysis",
    "PoseAxisError",
    "analyze_pose_axis",
    "contest_curvature_pose_loss",
    # theoretical floor (Impl 8)
    "CANONICAL_BLAHUT_AVAILABLE",
    "compute_contest_floor",
    # pixel-budget allocator (Impl 9)
    "CANONICAL_INTERNAL_RESOLUTION_MULTIPLES",
    "PixelBudgetRecommendation",
    "recommend_internal_resolution",
    # cell allocator (Impl 10)
    "CellAllocation",
    "CellAllocatorError",
    "SparseWaterFillingAllocation",
    "sparse_water_fill",
    # score predictor (Impl 11)
    "ContestScorePrediction",
    "build_contest_action",
    "predict_score",
    "validate_against_canonical_formula",
    # substrate alignment (Impl 12)
    "AlignmentFacet",
    "PR101_GOLD_ALIGNED_FACETS",
    "SubstrateAlignmentScore",
    "pr101_gold_reference",
    "score_substrate_alignment",
    # bandit per pair (Impl 13)
    "BanditError",
    "BetaBernoulliPosterior",
    "PerPairBanditAssignment",
    "PerPairBanditPlan",
    "thompson_sample_per_pair_with_posterior",
    "update_beta_bernoulli_posterior",
    # class-conditional CDF (Impl 14)
    "ArithmeticCoderError",
    "ClassConditionalCodebook",
    "ClassConditionalPrior",
    "build_class_conditional_codebook",
]
