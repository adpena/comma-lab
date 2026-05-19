# SPDX-License-Identifier: MIT
"""4-term scalar findings Lagrangian + canonical result dataclass.

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19):

    L_findings = data_fit
                 + lambda_Occam_complexity * complexity_penalty
                 + lambda_Occam_interpretability * interpretability_penalty
                 + lambda_partition * partition_penalty
                 - mu_explore * expected_info_gain

The 4 terms (per slot 20 + amendments):

1. **data_fit**: Gaussian likelihood sum over empirical anchors
   (per equation's posterior; Q1 binding decision).
2. **Occam complexity**: posterior dimensionality penalty (Rudin amendment).
3. **Occam interpretability**: penalty on non-falling-rule-list-explainable
   posteriors (Rudin Q3 amendment).
4. **Partition**: MDL penalty for the current partition refinement state
   (Daubechies Q4 amendment with wavelet-multi-scale prior).

Plus the info-gain TERM (subtracted as a reward for exploration):

5. **mu_explore * E[KL info gain]**: active-inference bridge per
   Schmidhuber operating-within + Lindley 1956. Subtracted because
   higher info gain = better candidate = lower Lagrangian.

The result dataclass FindingsLagrangianResult emits posterior_sigma_per_term
as the canonical sensitivity-map signal for downstream consumers
(Catalog #125 hook #1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tac.findings_lagrangian.info_gain import (
    expected_information_gain,
    kl_divergence_gaussians,
)
from tac.findings_lagrangian.partition import (
    CanonicalPartition,
    PartitionInvalidError,
)
from tac.findings_lagrangian.posterior import (
    GaussianPosterior,
    PosteriorInvalidError,
)
from tac.findings_lagrangian.weights import (
    FindingsLagrangianWeights,
    default_initial_weights,
)


__all__ = [
    "FindingsLagrangianResult",
    "compute_findings_lagrangian",
    "decompose_lagrangian",
    "LagrangianInvalidError",
]


class LagrangianInvalidError(ValueError):
    """Raised when a FindingsLagrangianResult violates invariants."""


@dataclass(frozen=True)
class FindingsLagrangianResult:
    """4-term findings Lagrangian computation result.

    Per Catalog #305 observability surface: every term is decomposable,
    inspectable, queryable, citable, diff-able, and counterfactual-able.

    Per Catalog #125 hook #1: ``posterior_sigma_per_term`` IS the
    sensitivity-map signal that downstream ``tac.sensitivity_map.*``
    consumers route through cathedral autopilot ranker.
    """

    equation_id: str
    partition_id: str
    data_fit: float
    occam_complexity: float
    occam_interpretability: float
    partition_penalty: float
    info_gain_nats: float
    weights_snapshot: FindingsLagrangianWeights
    posterior_sigma_per_term: tuple[float, ...]
    n_anchors_used: int
    computation_utc: str

    def __post_init__(self) -> None:
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise LagrangianInvalidError("equation_id must be non-empty string")
        if not isinstance(self.partition_id, str) or not self.partition_id.strip():
            raise LagrangianInvalidError("partition_id must be non-empty string")
        for name, val in [
            ("data_fit", self.data_fit),
            ("occam_complexity", self.occam_complexity),
            ("occam_interpretability", self.occam_interpretability),
            ("partition_penalty", self.partition_penalty),
            ("info_gain_nats", self.info_gain_nats),
        ]:
            if not isinstance(val, (int, float)):
                raise LagrangianInvalidError(f"{name} must be numeric")
            if val != val:  # NaN
                raise LagrangianInvalidError(f"{name} is NaN")
        if self.data_fit < 0:
            raise LagrangianInvalidError(
                f"data_fit={self.data_fit} must be >= 0 (likelihood log-loss)"
            )
        if self.occam_complexity < 0:
            raise LagrangianInvalidError(
                f"occam_complexity={self.occam_complexity} must be >= 0"
            )
        if self.occam_interpretability < 0:
            raise LagrangianInvalidError(
                f"occam_interpretability={self.occam_interpretability} must be >= 0"
            )
        if self.partition_penalty < 0:
            raise LagrangianInvalidError(
                f"partition_penalty={self.partition_penalty} must be >= 0"
            )
        if self.info_gain_nats < 0:
            raise LagrangianInvalidError(
                f"info_gain_nats={self.info_gain_nats} must be >= 0"
            )
        if not isinstance(self.weights_snapshot, FindingsLagrangianWeights):
            raise LagrangianInvalidError("weights_snapshot must be FindingsLagrangianWeights")
        if not isinstance(self.posterior_sigma_per_term, tuple):
            raise LagrangianInvalidError("posterior_sigma_per_term must be tuple")
        for i, s in enumerate(self.posterior_sigma_per_term):
            if not isinstance(s, (int, float)) or s < 0:
                raise LagrangianInvalidError(
                    f"posterior_sigma_per_term[{i}]={s} must be >= 0"
                )
        if not isinstance(self.n_anchors_used, int) or self.n_anchors_used < 0:
            raise LagrangianInvalidError(
                f"n_anchors_used={self.n_anchors_used} must be non-negative int"
            )
        if not isinstance(self.computation_utc, str) or not self.computation_utc:
            raise LagrangianInvalidError("computation_utc must be non-empty string")

    @property
    def scalar(self) -> float:
        """Scalar Lagrangian value (the optimization target).

        Lower is better; the mu_explore term is SUBTRACTED so higher
        info gain pushes the Lagrangian down.
        """
        w = self.weights_snapshot
        return (
            self.data_fit
            + w.lambda_occam_complexity * self.occam_complexity
            + w.lambda_occam_interpretability * self.occam_interpretability
            + w.lambda_partition * self.partition_penalty
            - w.mu_explore * self.info_gain_nats
        )

    def decompose(self) -> Mapping[str, float]:
        """Return per-term decomposition for operator-facing display.

        Per Catalog #305 observability surface: decomposable per signal.
        """
        w = self.weights_snapshot
        return {
            "data_fit": float(self.data_fit),
            "occam_complexity_weighted": float(
                w.lambda_occam_complexity * self.occam_complexity
            ),
            "occam_interpretability_weighted": float(
                w.lambda_occam_interpretability * self.occam_interpretability
            ),
            "partition_penalty_weighted": float(
                w.lambda_partition * self.partition_penalty
            ),
            "info_gain_reward_weighted": float(
                -w.mu_explore * self.info_gain_nats
            ),
            "scalar": float(self.scalar),
        }

    def as_dict(self) -> dict[str, object]:
        """Serialize for posterior persistence + operator display."""
        return {
            "equation_id": self.equation_id,
            "partition_id": self.partition_id,
            "data_fit": float(self.data_fit),
            "occam_complexity": float(self.occam_complexity),
            "occam_interpretability": float(self.occam_interpretability),
            "partition_penalty": float(self.partition_penalty),
            "info_gain_nats": float(self.info_gain_nats),
            "weights_snapshot": self.weights_snapshot.as_dict(),
            "posterior_sigma_per_term": list(self.posterior_sigma_per_term),
            "n_anchors_used": int(self.n_anchors_used),
            "computation_utc": self.computation_utc,
            "scalar": float(self.scalar),
            "decompose": dict(self.decompose()),
        }


def _gaussian_negative_log_likelihood(
    residuals: Sequence[float], sigma_obs: float
) -> float:
    """Negative log-likelihood of residuals under Gaussian(0, sigma_obs**2).

    Used as the data_fit term. Higher residuals → larger NLL → larger
    data_fit cost (which the Lagrangian wants to minimize).
    """
    if sigma_obs <= 0:
        raise LagrangianInvalidError(f"sigma_obs={sigma_obs} must be > 0")
    n = len(residuals)
    if n == 0:
        return 0.0
    sum_sq = sum(r * r for r in residuals)
    # -log p = 0.5 * n * log(2 pi sigma**2) + 0.5 * sum_sq / sigma**2
    return 0.5 * n * math.log(2.0 * math.pi * sigma_obs**2) + 0.5 * sum_sq / (
        sigma_obs**2
    )


def _complexity_penalty(posterior: GaussianPosterior) -> float:
    """Posterior-dimensionality penalty (Rudin Q3 complexity sub-weight).

    Larger dim = higher complexity cost. Rudin's canonical interpretability
    principle: prefer low-dim posteriors when sufficient.
    """
    return float(posterior.dim)


def _interpretability_penalty(posterior: GaussianPosterior) -> float:
    """Penalty on non-falling-rule-list-explainable posteriors (Rudin Q3 amend).

    For diagonal-covariance Gaussian (our canonical case) the penalty is 0
    because diagonal posteriors are trivially explainable per dim. For
    full-covariance posteriors the penalty scales with off-diagonal mass
    (encodes inter-dimension correlation that breaks per-dim explainability).
    """
    d = posterior.dim
    if d == 1:
        return 0.0
    sigma_array = posterior.sigma
    off_diag_mass = 0.0
    diag_mass = 0.0
    for i in range(d):
        for j in range(d):
            v = abs(float(sigma_array[i][j]))
            if i == j:
                diag_mass += v
            else:
                off_diag_mass += v
    if diag_mass <= 0:
        return float(d)  # degenerate; max penalty
    return off_diag_mass / diag_mass  # 0 for purely diagonal


def _partition_penalty(partition: CanonicalPartition) -> float:
    """MDL-style partition complexity penalty per Q4 amendment.

    More classes = higher penalty; deeper tree = higher penalty (wavelet
    prior per Daubechies amendment).
    """
    n_classes = len(partition.classes)
    avg_depth = sum(c.tree_depth for c in partition.classes) / max(n_classes, 1)
    # log(n_classes) base cost + linear in average depth (wavelet prior)
    return math.log(n_classes + 1.0) + 0.5 * avg_depth


def compute_findings_lagrangian(
    equation_id: str,
    *,
    posterior: GaussianPosterior,
    partition: CanonicalPartition,
    anchor_residuals: Sequence[float],
    weights: FindingsLagrangianWeights | None = None,
    hypothetical_residuals_for_info_gain: Sequence[float] | None = None,
    sigma_obs: float = 1.0,
    computation_utc: str | None = None,
) -> FindingsLagrangianResult:
    """Compute the 4-term findings Lagrangian for one equation + partition.

    Per slot 20 binding + supplemental amendments:

    L = data_fit
      + lambda_Occam_complexity * complexity_penalty
      + lambda_Occam_interpretability * interpretability_penalty
      + lambda_partition * partition_penalty
      - mu_explore * E[KL info gain from hypothetical experiment]

    Args:
        equation_id: which canonical_equation this Lagrangian computes.
        posterior: current GaussianPosterior for that equation.
        partition: current CanonicalPartition.
        anchor_residuals: actual residuals from observed empirical anchors.
        weights: weight vector; defaults to canonical initial values.
        hypothetical_residuals_for_info_gain: optional simulated residuals
            from a hypothetical next experiment (used for E[KL] term).
            If None, info_gain term is 0.
        sigma_obs: observation noise std-dev for likelihood.
        computation_utc: timestamp for the computation result.

    Returns:
        FindingsLagrangianResult with all 4 terms + scalar.
    """
    if not isinstance(posterior, GaussianPosterior):
        raise LagrangianInvalidError(
            f"posterior must be GaussianPosterior, got {type(posterior).__name__}"
        )
    if not isinstance(partition, CanonicalPartition):
        raise LagrangianInvalidError(
            f"partition must be CanonicalPartition, got {type(partition).__name__}"
        )
    if equation_id != posterior.equation_id:
        raise LagrangianInvalidError(
            f"equation_id={equation_id!r} != posterior.equation_id={posterior.equation_id!r}"
        )

    w = weights if weights is not None else default_initial_weights()

    data_fit = _gaussian_negative_log_likelihood(anchor_residuals, sigma_obs)
    complexity = _complexity_penalty(posterior)
    interpretability = _interpretability_penalty(posterior)
    partition_pen = _partition_penalty(partition)

    if hypothetical_residuals_for_info_gain is not None:
        info_gain = expected_information_gain(
            posterior,
            hypothetical_residuals=hypothetical_residuals_for_info_gain,
            sigma_obs=sigma_obs,
        )
    else:
        info_gain = 0.0

    if computation_utc is None:
        import datetime as _dt

        computation_utc = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    return FindingsLagrangianResult(
        equation_id=equation_id,
        partition_id=partition.partition_id,
        data_fit=data_fit,
        occam_complexity=complexity,
        occam_interpretability=interpretability,
        partition_penalty=partition_pen,
        info_gain_nats=info_gain,
        weights_snapshot=w,
        posterior_sigma_per_term=posterior.posterior_sigma_per_term,
        n_anchors_used=len(anchor_residuals),
        computation_utc=computation_utc,
    )


def decompose_lagrangian(
    result: FindingsLagrangianResult,
) -> Mapping[str, float]:
    """Operator-facing decomposition (delegates to FindingsLagrangianResult.decompose)."""
    return result.decompose()
