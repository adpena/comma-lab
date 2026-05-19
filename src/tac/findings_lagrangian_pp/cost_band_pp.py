# SPDX-License-Identifier: MIT
"""Q8 rank #2-A — Gaussian posterior over cost-band p50 (TRACK B).

Per T3 grand council 3-round consolidated verdict Q8 ratified rank-order:
cost band calibration is in the Phase 1 TRACK B scope per operator
override (slot 20 originally ranked it #3; supplemental moved to #4 after
composition matrix elevated to #1; operator parallel-pursuit override
brings it back into Phase 1).

The implementation: closed-form Gaussian posterior over per-class cost
p50 (median) + uncertainty sigma. Sister of TRACK A's
``tac.findings_lagrangian.posterior.GaussianPosterior`` but specialized
for the existing ``.omx/state/cost_band_posterior.jsonl`` schema.

Per Boyd operating-within slot 20: closed-form Gaussian preserves convex
feasibility structure that admits alternating-projections solution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


__all__ = [
    "CostBandGaussianPosterior",
    "update_cost_band_posterior_from_dispatch_anchors",
    "predict_cost_for_class_with_uncertainty",
    "CostBandPPError",
]


class CostBandPPError(ValueError):
    """Raised when cost-band PP inputs are malformed."""


@dataclass(frozen=True)
class CostBandGaussianPosterior:
    """Per-class cost posterior with explicit uncertainty.

    Attributes:
        class_id: dispatch class (e.g. 'smoke' / 'full' / 'long_burn').
        posterior_mu_log_cost: posterior mean of log(cost_usd).
        posterior_sigma_log_cost: posterior std-dev of log(cost_usd).
        n_dispatches_observed: count of empirical dispatches.
    """

    class_id: str
    posterior_mu_log_cost: float
    posterior_sigma_log_cost: float
    n_dispatches_observed: int

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id.strip():
            raise CostBandPPError("class_id must be non-empty string")
        if not isinstance(self.posterior_mu_log_cost, (int, float)):
            raise CostBandPPError("posterior_mu_log_cost must be numeric")
        if self.posterior_mu_log_cost != self.posterior_mu_log_cost:  # NaN
            raise CostBandPPError("posterior_mu_log_cost is NaN")
        if not isinstance(self.posterior_sigma_log_cost, (int, float)):
            raise CostBandPPError("posterior_sigma_log_cost must be numeric")
        if self.posterior_sigma_log_cost < 0:
            raise CostBandPPError(
                f"posterior_sigma_log_cost={self.posterior_sigma_log_cost} must be >= 0"
            )
        if not isinstance(self.n_dispatches_observed, int) or self.n_dispatches_observed < 0:
            raise CostBandPPError(
                f"n_dispatches_observed={self.n_dispatches_observed} must be non-negative int"
            )

    @property
    def predicted_p50_cost_usd(self) -> float:
        """Predicted median cost (back-transformed from log-Gaussian)."""
        return math.exp(self.posterior_mu_log_cost)

    @property
    def predicted_p90_cost_usd(self) -> float:
        """Predicted 90th percentile cost (log-Gaussian + 1.28 sigma)."""
        return math.exp(self.posterior_mu_log_cost + 1.28 * self.posterior_sigma_log_cost)


def update_cost_band_posterior_from_dispatch_anchors(
    class_id: str,
    *,
    observed_costs_usd: Sequence[float],
    prior_mu_log_cost: float = 0.0,  # exp(0) = $1 prior median
    prior_sigma_log_cost: float = 1.0,  # ~3x cost-band uncertainty
    sigma_obs_log: float = 0.5,  # ~50% per-dispatch log-cost noise
) -> CostBandGaussianPosterior:
    """Closed-form Bayesian update for log-Gaussian cost model.

    Args:
        class_id: dispatch class.
        observed_costs_usd: empirical per-dispatch costs (USD, > 0).
        prior_mu_log_cost: prior mean of log(cost) (default 0 = $1 median).
        prior_sigma_log_cost: prior std-dev of log(cost).
        sigma_obs_log: per-dispatch log-cost noise std-dev.

    Returns:
        CostBandGaussianPosterior with refreshed mean + sigma.
    """
    costs = list(observed_costs_usd)
    if not costs:
        return CostBandGaussianPosterior(
            class_id=class_id,
            posterior_mu_log_cost=prior_mu_log_cost,
            posterior_sigma_log_cost=prior_sigma_log_cost,
            n_dispatches_observed=0,
        )
    for i, c in enumerate(costs):
        if not isinstance(c, (int, float)) or c <= 0:
            raise CostBandPPError(
                f"observed_costs_usd[{i}]={c} must be > 0"
            )
    if sigma_obs_log <= 0:
        raise CostBandPPError(f"sigma_obs_log={sigma_obs_log} must be > 0")
    if prior_sigma_log_cost <= 0:
        raise CostBandPPError(f"prior_sigma_log_cost={prior_sigma_log_cost} must be > 0")

    log_costs = [math.log(c) for c in costs]
    n = len(log_costs)
    mean_log_obs = sum(log_costs) / n

    prior_precision = 1.0 / (prior_sigma_log_cost**2)
    likelihood_precision = n / (sigma_obs_log**2)
    posterior_precision = prior_precision + likelihood_precision
    posterior_variance = 1.0 / posterior_precision
    posterior_sigma = math.sqrt(posterior_variance)
    posterior_mu = posterior_variance * (
        prior_precision * prior_mu_log_cost
        + likelihood_precision * mean_log_obs
    )

    return CostBandGaussianPosterior(
        class_id=class_id,
        posterior_mu_log_cost=posterior_mu,
        posterior_sigma_log_cost=posterior_sigma,
        n_dispatches_observed=n,
    )


def predict_cost_for_class_with_uncertainty(
    posterior: CostBandGaussianPosterior,
) -> tuple[float, float]:
    """Return (predicted_p50_cost_usd, uncertainty_factor).

    The uncertainty_factor is the multiplicative range corresponding to
    +/-1 sigma in log-cost: e.g. uncertainty_factor=1.5 means
    p50 * 1.5 = +1 sigma upper bound.

    Per Catalog #125 hook #4: cathedral autopilot can consume this for
    asymmetric-cost dispatch decisions (high-uncertainty + paid-class →
    consider cheap probe first).
    """
    p50 = posterior.predicted_p50_cost_usd
    uncertainty_factor = math.exp(posterior.posterior_sigma_log_cost)
    return (p50, uncertainty_factor)
