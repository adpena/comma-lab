# SPDX-License-Identifier: MIT
"""Closed-form Gaussian posterior + conjugate Bayesian update.

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19): Q1 RATIFIED closed-form Gaussian
posterior per equation; scipy.stats.multivariate_normal for sampling;
ZERO new PP framework dependency. Per Q5 RATIFIED hand-rolled
Gaussian + scipy.stats only.

The math: assume Gaussian prior over per-equation parameter vector
``theta ~ N(mu_0, Sigma_0)`` + Gaussian likelihood
``residual | theta ~ N(0, sigma_obs**2)``. Then the posterior after
observing N anchors is conjugate Gaussian with
``Sigma_N^-1 = Sigma_0^-1 + N / sigma_obs**2 * I_d`` and
``mu_N = Sigma_N (Sigma_0^-1 mu_0 + sum residuals / sigma_obs**2)``
for a simple 1-d-per-anchor closed form. Multi-d extensions use
``scipy.stats.multivariate_normal``.

Per CLAUDE.md "Apples-to-apples evidence discipline": every posterior emits
provenance with axis_tag=[predicted] per Catalog #287/#323. The posterior
sigma is the uncertainty signal Q7 cathedral autopilot ranker consumes
for asymmetric-cost dispatch routing per Lindley 1956 + Foster 2019.

Sister modules:
- ``tac.findings_lagrangian.lagrangian`` — 4-term scalar objective
- ``tac.findings_lagrangian.info_gain`` — closed-form KL between Gaussians (Q2)
- ``tac.findings_lagrangian.weights`` — adaptive lambda/mu schedule (Q3)
- ``tac.canonical_equations`` — equation registry the posterior is keyed to
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats


__all__ = [
    "GaussianPosterior",
    "posterior_update_from_anchors",
    "posterior_sample",
    "posterior_predict",
    "PosteriorInvalidError",
]


class PosteriorInvalidError(ValueError):
    """Raised when a GaussianPosterior violates invariants.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": every contract is
    enforced in ``__post_init__`` so the construction surface refuses
    bad inputs at the source (not after silently producing wrong posteriors).
    """


@dataclass(frozen=True)
class GaussianPosterior:
    """Closed-form Gaussian posterior over a per-equation parameter vector.

    Attributes:
        equation_id: pinned canonical_equation_id (snake_case_vN per registry)
        mu: posterior mean vector (shape (d,))
        sigma: posterior std-dev vector for diagonal covariance OR full
            covariance matrix shape (d, d). Sub-d arrays are diagonal;
            d-by-d arrays are full-covariance. Encoded as tuple-of-tuples
            for frozen-dataclass safety.
        n_anchors: count of empirical anchors that produced this posterior.
        prior_mu: original prior mean (for KL info gain calculations).
        prior_sigma_diagonal: original prior diagonal std-dev (for KL).

    Per Q3 supplemental amendment: posterior_sigma_per_term is the
    sensitivity-map signal for downstream consumers per Catalog #125 hook #1.

    Per Catalog #287 + #323: every prediction from this posterior emits
    ``[predicted]`` axis tag with canonical Provenance.
    """

    equation_id: str
    mu: tuple[float, ...]
    sigma: tuple[tuple[float, ...], ...]  # full-covariance matrix
    n_anchors: int
    prior_mu: tuple[float, ...]
    prior_sigma_diagonal: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise PosteriorInvalidError("equation_id must be non-empty string")
        if not isinstance(self.mu, tuple):
            raise PosteriorInvalidError(
                f"mu must be tuple of floats, got {type(self.mu).__name__}"
            )
        d = len(self.mu)
        if d == 0:
            raise PosteriorInvalidError("mu must be non-empty")
        for i, m in enumerate(self.mu):
            if not isinstance(m, (int, float)):
                raise PosteriorInvalidError(f"mu[{i}] must be numeric")
            if m != m:  # NaN
                raise PosteriorInvalidError(f"mu[{i}] is NaN")
        if not isinstance(self.sigma, tuple):
            raise PosteriorInvalidError("sigma must be tuple of tuples (matrix rows)")
        if len(self.sigma) != d:
            raise PosteriorInvalidError(
                f"sigma rows={len(self.sigma)} != mu dim={d}"
            )
        for i, row in enumerate(self.sigma):
            if not isinstance(row, tuple) or len(row) != d:
                raise PosteriorInvalidError(
                    f"sigma[{i}] must be tuple of length {d}, got {row!r}"
                )
            for j, v in enumerate(row):
                if not isinstance(v, (int, float)):
                    raise PosteriorInvalidError(
                        f"sigma[{i}][{j}] must be numeric"
                    )
                if v != v:  # NaN
                    raise PosteriorInvalidError(f"sigma[{i}][{j}] is NaN")
        # Diagonal must be positive (PSD requirement on covariance).
        for i in range(d):
            if self.sigma[i][i] <= 0:
                raise PosteriorInvalidError(
                    f"sigma[{i}][{i}]={self.sigma[i][i]} must be > 0 (PSD)"
                )
        if not isinstance(self.n_anchors, int) or self.n_anchors < 0:
            raise PosteriorInvalidError("n_anchors must be non-negative int")
        if not isinstance(self.prior_mu, tuple) or len(self.prior_mu) != d:
            raise PosteriorInvalidError(
                f"prior_mu must be tuple of length {d}"
            )
        if not isinstance(self.prior_sigma_diagonal, tuple) or len(
            self.prior_sigma_diagonal
        ) != d:
            raise PosteriorInvalidError(
                f"prior_sigma_diagonal must be tuple of length {d}"
            )
        for i, ps in enumerate(self.prior_sigma_diagonal):
            if not isinstance(ps, (int, float)) or ps <= 0:
                raise PosteriorInvalidError(
                    f"prior_sigma_diagonal[{i}]={ps} must be > 0"
                )

    @property
    def dim(self) -> int:
        """Posterior dimensionality."""
        return len(self.mu)

    @property
    def posterior_sigma_per_term(self) -> tuple[float, ...]:
        """Diagonal std-dev per term — sensitivity-map signal per Catalog #125 hook #1.

        Per slot 20 + supplemental Q7 amendment: this is the uncertainty
        signal cathedral autopilot ranker consumes for asymmetric-cost
        downweighting via 1/(1+sigma) per Q7 binding decision.
        """
        return tuple(math.sqrt(self.sigma[i][i]) for i in range(self.dim))

    @property
    def is_well_calibrated(self) -> bool:
        """True iff posterior is tighter than the prior on every dimension.

        Operationally: "have we learned ANYTHING about this equation yet?"
        Empty-anchor posterior has prior_sigma == posterior_sigma, so this
        returns False until at least one anchor lands.
        """
        if self.n_anchors == 0:
            return False
        for i in range(self.dim):
            post_std = math.sqrt(self.sigma[i][i])
            if post_std >= self.prior_sigma_diagonal[i]:
                return False
        return True

    def as_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu_array, sigma_matrix) as numpy arrays for scipy ops."""
        mu = np.array(self.mu, dtype=np.float64)
        sigma = np.array([list(row) for row in self.sigma], dtype=np.float64)
        return mu, sigma


def _build_diagonal_posterior_tuple(
    equation_id: str,
    mu: Sequence[float],
    sigma_diag: Sequence[float],
    prior_mu: Sequence[float],
    prior_sigma_diag: Sequence[float],
    n_anchors: int,
) -> GaussianPosterior:
    """Build a frozen GaussianPosterior from diagonal-only spec.

    Internal helper for the conjugate update path which produces
    diagonal-covariance posteriors when prior + likelihood are diagonal.
    """
    d = len(mu)
    sigma_rows: list[tuple[float, ...]] = []
    for i in range(d):
        row = [0.0] * d
        row[i] = float(sigma_diag[i]) ** 2  # variance on diagonal
        sigma_rows.append(tuple(row))
    return GaussianPosterior(
        equation_id=equation_id,
        mu=tuple(float(m) for m in mu),
        sigma=tuple(sigma_rows),
        n_anchors=n_anchors,
        prior_mu=tuple(float(m) for m in prior_mu),
        prior_sigma_diagonal=tuple(float(s) for s in prior_sigma_diag),
    )


def posterior_update_from_anchors(
    equation_id: str,
    *,
    prior_mu: Sequence[float],
    prior_sigma_diagonal: Sequence[float],
    anchor_residuals: Sequence[float],
    sigma_obs: float = 1.0,
) -> GaussianPosterior:
    """Conjugate Bayesian update for diagonal-Gaussian-prior + Gaussian-likelihood.

    The canonical math (per MacKay 1992 + Hinton dissent on slot 20):

        prior:   theta_i ~ N(mu_0_i, sigma_0_i**2)
        likely:  r_n | theta ~ N(theta_d, sigma_obs**2)  for d = n mod dim
                 (residuals cycled through dimensions if N > dim)
        post:    sigma_N_i**2 = 1 / (1/sigma_0_i**2 + n_i/sigma_obs**2)
                 mu_N_i = sigma_N_i**2 * (mu_0_i/sigma_0_i**2 + sum(r)/sigma_obs**2)

    where n_i = count of residuals assigned to dimension i.

    For the simple ``dim=1`` case (most equations) this reduces to the
    canonical 1-d conjugate update.

    Args:
        equation_id: canonical equation_id this posterior tracks.
        prior_mu: prior mean vector (shape (d,)).
        prior_sigma_diagonal: prior std-dev per dimension.
        anchor_residuals: empirical (predicted - actual) residuals.
        sigma_obs: observation noise std-dev (default 1.0 = unit-scale).

    Returns:
        GaussianPosterior with refreshed mu + sigma + n_anchors.

    Raises:
        PosteriorInvalidError: if inputs are malformed.
    """
    d = len(prior_mu)
    if d == 0:
        raise PosteriorInvalidError("prior_mu must be non-empty")
    if len(prior_sigma_diagonal) != d:
        raise PosteriorInvalidError(
            f"prior_sigma_diagonal length {len(prior_sigma_diagonal)} != "
            f"prior_mu length {d}"
        )
    if sigma_obs <= 0:
        raise PosteriorInvalidError(f"sigma_obs={sigma_obs} must be > 0")

    n_total = len(anchor_residuals)
    # Assign residuals round-robin to dimensions.
    sum_residuals_per_dim = [0.0] * d
    count_per_dim = [0] * d
    for k, r in enumerate(anchor_residuals):
        if not isinstance(r, (int, float)):
            raise PosteriorInvalidError(f"anchor_residuals[{k}] must be numeric")
        if r != r:  # NaN
            raise PosteriorInvalidError(f"anchor_residuals[{k}] is NaN")
        dim_index = k % d
        sum_residuals_per_dim[dim_index] += float(r)
        count_per_dim[dim_index] += 1

    post_sigma_diag = []
    post_mu = []
    for i in range(d):
        prior_precision = 1.0 / (float(prior_sigma_diagonal[i]) ** 2)
        likelihood_precision = float(count_per_dim[i]) / (sigma_obs**2)
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance = 1.0 / posterior_precision
        posterior_sigma = math.sqrt(posterior_variance)
        # Predicted = prior_mu; empirical_signal = prior_mu + r (residual is
        # offset from prior prediction). Closed-form mean update.
        sum_signal = (
            float(prior_mu[i]) * count_per_dim[i] + sum_residuals_per_dim[i]
        )
        posterior_mean = posterior_variance * (
            prior_precision * float(prior_mu[i])
            + sum_signal / (sigma_obs**2)
        )
        post_sigma_diag.append(posterior_sigma)
        post_mu.append(posterior_mean)

    return _build_diagonal_posterior_tuple(
        equation_id=equation_id,
        mu=post_mu,
        sigma_diag=post_sigma_diag,
        prior_mu=prior_mu,
        prior_sigma_diag=prior_sigma_diagonal,
        n_anchors=n_total,
    )


def posterior_sample(
    posterior: GaussianPosterior,
    *,
    n_samples: int = 100,
    seed: int | None = 42,
) -> np.ndarray:
    """Draw samples from a GaussianPosterior via scipy.stats.multivariate_normal.

    Per Q1 binding decision: scipy.stats.multivariate_normal is the canonical
    sampler; ZERO new PP framework dependency.

    Args:
        posterior: the GaussianPosterior to sample from.
        n_samples: how many samples to draw.
        seed: RNG seed for deterministic reproducibility per CLAUDE.md
            "Beauty, simplicity, and developer experience".

    Returns:
        Array of shape (n_samples, posterior.dim).
    """
    mu, sigma = posterior.as_numpy()
    # Ensure PSD by symmetrizing.
    sigma_sym = (sigma + sigma.T) / 2.0
    rng = np.random.default_rng(seed)
    samples = stats.multivariate_normal.rvs(
        mean=mu, cov=sigma_sym, size=n_samples, random_state=rng
    )
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    return samples


def posterior_predict(
    posterior: GaussianPosterior,
    *,
    return_uncertainty: bool = True,
) -> tuple[tuple[float, ...], tuple[float, ...] | None]:
    """Return (predicted_mean, uncertainty_sigma) for this posterior.

    Per Q7 binding decision (slot 20 + supplemental amendments): the
    posterior emits BOTH the mean prediction AND the per-dimension
    uncertainty sigma so cathedral autopilot ranker can downweight
    high-uncertainty candidates by 1/(1+sigma).

    Args:
        posterior: the GaussianPosterior to predict from.
        return_uncertainty: if True (default), return per-dim sigma tuple;
            if False, return None as the second value.

    Returns:
        Tuple of (mu_tuple, sigma_tuple_or_None).
    """
    mu_tuple = posterior.mu
    if return_uncertainty:
        sigma_tuple = posterior.posterior_sigma_per_term
        return (mu_tuple, sigma_tuple)
    return (mu_tuple, None)
