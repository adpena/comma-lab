# SPDX-License-Identifier: MIT
"""Adaptive lambda / mu weight schedule per Q3 binding decision + amendments.

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19):

- Q3 PROCEED_WITH_REVISIONS (slot 20): fixed initial weights
  lambda_Occam=0.1, lambda_partition=0.1, mu_explore=0.05 (capped at 0.1).
  Adaptive schedule per Catalog #167 sister.
- Q3 AMEND (slot 20-supplemental, Rudin amendment): lambda_Occam DECOMPOSED
  into lambda_Occam_complexity (penalizes high-dim posterior) AND
  lambda_Occam_interpretability (penalizes non-falling-rule-list-explainable
  posterior structure). Each sub-weight starts at 0.05 (sum = 0.1 preserves
  slot 20 budget).
- Q3 ratified with Hotz extension (slot 20-second-supplemental): weights MUST
  be hand-tunable from empirical-anchor evidence; abstract priors REQUIRE
  empirical residual evidence to fire.

Per Tishby operating-within (slot 20): the IB Lagrangian beta coefficient is
canonically annealed; mu_explore similarly annealed from initial 0.05 upward
only when partition-discovery has stabilized.

Per Schmidhuber: mu_explore upper-bound capped at 0.1 (10% exploration budget).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = [
    "FindingsLagrangianWeights",
    "default_initial_weights",
    "adapt_weights_from_residuals",
    "WEIGHT_BUDGET_TOTAL",
    "MU_EXPLORE_CAP",
    "LAMBDA_OCCAM_SUM_INITIAL",
    "LAMBDA_OCCAM_COMPLEXITY_INITIAL",
    "LAMBDA_OCCAM_INTERPRETABILITY_INITIAL",
    "LAMBDA_PARTITION_INITIAL",
    "MU_EXPLORE_INITIAL",
    "ADAPTIVE_FACTOR",
    "RESIDUAL_DRIFT_THRESHOLD_SIGMA",
    "WeightInvariantError",
]


# Canonical initial values per Q3 binding decision (slot 20) + supplemental
# decomposition (slot 20-supplemental Rudin amendment).
LAMBDA_OCCAM_COMPLEXITY_INITIAL = 0.05
"""Initial penalty on posterior dimensionality (Rudin: complexity sub-weight)."""

LAMBDA_OCCAM_INTERPRETABILITY_INITIAL = 0.05
"""Initial penalty on non-falling-rule-list-explainable posteriors (Rudin)."""

LAMBDA_OCCAM_SUM_INITIAL = (
    LAMBDA_OCCAM_COMPLEXITY_INITIAL + LAMBDA_OCCAM_INTERPRETABILITY_INITIAL
)
"""Sum of the two Occam sub-weights (preserves slot 20's 0.1 budget)."""

LAMBDA_PARTITION_INITIAL = 0.1
"""Initial partition penalty (slot 20 unchanged)."""

MU_EXPLORE_INITIAL = 0.05
"""Initial exploration budget (slot 20 unchanged)."""

MU_EXPLORE_CAP = 0.1
"""Upper bound on mu_explore per Schmidhuber dissent + Q3 binding decision."""

WEIGHT_BUDGET_TOTAL = (
    LAMBDA_OCCAM_SUM_INITIAL + LAMBDA_PARTITION_INITIAL + MU_EXPLORE_INITIAL
)
"""Total weight budget at initialization (operator-facing invariant)."""

ADAPTIVE_FACTOR = 1.5
"""Multiplicative bump when adaptive trigger fires (per Catalog #167 sister)."""

RESIDUAL_DRIFT_THRESHOLD_SIGMA = 2.0
"""Residual must exceed N*sigma to fire lambda_partition bump."""


class WeightInvariantError(ValueError):
    """Raised when weights violate invariants (e.g. mu_explore > cap)."""


@dataclass(frozen=True)
class FindingsLagrangianWeights:
    """4-term Lagrangian weights with Rudin Q3 decomposition.

    Per slot 20 + supplemental Q3 amendment: lambda_Occam is split into
    complexity + interpretability sub-weights so posterior structure that
    is non-interpretable (high-dim full covariance) is penalized differently
    than posterior structure that is interpretable (diagonal + falling-rule).
    """

    lambda_occam_complexity: float = LAMBDA_OCCAM_COMPLEXITY_INITIAL
    lambda_occam_interpretability: float = LAMBDA_OCCAM_INTERPRETABILITY_INITIAL
    lambda_partition: float = LAMBDA_PARTITION_INITIAL
    mu_explore: float = MU_EXPLORE_INITIAL

    def __post_init__(self) -> None:
        for name, val in [
            ("lambda_occam_complexity", self.lambda_occam_complexity),
            ("lambda_occam_interpretability", self.lambda_occam_interpretability),
            ("lambda_partition", self.lambda_partition),
            ("mu_explore", self.mu_explore),
        ]:
            if not isinstance(val, (int, float)):
                raise WeightInvariantError(f"{name} must be numeric, got {type(val).__name__}")
            if val != val:  # NaN
                raise WeightInvariantError(f"{name} is NaN")
            if val < 0:
                raise WeightInvariantError(f"{name}={val} must be >= 0")
        if self.mu_explore > MU_EXPLORE_CAP:
            raise WeightInvariantError(
                f"mu_explore={self.mu_explore} exceeds cap {MU_EXPLORE_CAP} "
                "per Schmidhuber dissent + Q3 binding decision"
            )

    @property
    def lambda_occam_sum(self) -> float:
        """Total Occam penalty (sum of the two sub-weights)."""
        return self.lambda_occam_complexity + self.lambda_occam_interpretability

    @property
    def total_weight_budget(self) -> float:
        """Sum of all 4 sub-weights — operator-facing scalar."""
        return (
            self.lambda_occam_sum
            + self.lambda_partition
            + self.mu_explore
        )

    def as_dict(self) -> dict[str, float]:
        """Serialize to JSON-safe dict for posterior persistence."""
        return {
            "lambda_occam_complexity": self.lambda_occam_complexity,
            "lambda_occam_interpretability": self.lambda_occam_interpretability,
            "lambda_partition": self.lambda_partition,
            "mu_explore": self.mu_explore,
            "lambda_occam_sum_derived": self.lambda_occam_sum,
            "total_weight_budget_derived": self.total_weight_budget,
        }


def default_initial_weights() -> FindingsLagrangianWeights:
    """Return the canonical initial weight vector per Q3 binding decision."""
    return FindingsLagrangianWeights()


def adapt_weights_from_residuals(
    current: FindingsLagrangianWeights,
    *,
    residual_magnitude: float,
    posterior_sigma: float,
    interpretability_failures: int = 0,
) -> FindingsLagrangianWeights:
    """Adaptive schedule per Catalog #167 sister + Q3 amendments.

    Adaptive rules (per slot 20 + supplemental Q3 amendment):

    - lambda_partition: bump by ADAPTIVE_FACTOR (1.5x) if
      residual_magnitude > RESIDUAL_DRIFT_THRESHOLD_SIGMA * posterior_sigma.
      Signal that the partition is wrong and needs refinement (slot 20).
    - lambda_occam_interpretability: bump by ADAPTIVE_FACTOR (1.5x) if any
      downstream consumer emits an interpretability-failure-flag in its
      decision explanation (supplemental Rudin amendment).

    Per Hotz operating-within (slot 20-second-supplemental): weights MUST be
    hand-tunable from empirical-anchor evidence; this helper IS the bridge
    between abstract priors + empirical residual evidence.

    Args:
        current: current weight vector.
        residual_magnitude: magnitude of latest residual (predicted vs empirical).
        posterior_sigma: posterior std-dev on the same axis.
        interpretability_failures: count of downstream consumers flagging
            interpretability failure since last call.

    Returns:
        New FindingsLagrangianWeights with adapted values (frozen-safe).
    """
    if posterior_sigma <= 0:
        raise WeightInvariantError(f"posterior_sigma={posterior_sigma} must be > 0")
    if residual_magnitude < 0:
        raise WeightInvariantError(
            f"residual_magnitude={residual_magnitude} must be >= 0"
        )
    if interpretability_failures < 0:
        raise WeightInvariantError(
            f"interpretability_failures={interpretability_failures} must be >= 0"
        )

    new_lambda_partition = current.lambda_partition
    # Partition bump if residual exceeds 2*sigma.
    if residual_magnitude > RESIDUAL_DRIFT_THRESHOLD_SIGMA * posterior_sigma:
        new_lambda_partition = current.lambda_partition * ADAPTIVE_FACTOR

    new_lambda_occam_interpretability = current.lambda_occam_interpretability
    # Interpretability bump per Rudin amendment.
    if interpretability_failures > 0:
        bump_factor = ADAPTIVE_FACTOR**interpretability_failures
        new_lambda_occam_interpretability = (
            current.lambda_occam_interpretability * bump_factor
        )

    # mu_explore unchanged in this adaptation; annealed by separate IB schedule.
    return FindingsLagrangianWeights(
        lambda_occam_complexity=current.lambda_occam_complexity,
        lambda_occam_interpretability=new_lambda_occam_interpretability,
        lambda_partition=new_lambda_partition,
        mu_explore=min(current.mu_explore, MU_EXPLORE_CAP),
    )
