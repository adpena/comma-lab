# SPDX-License-Identifier: MIT
"""Uncertainty quantification from EmpiricalAnchor residuals.

Per operator NON-NEGOTIABLE 2026-05-19 SLOT MG-1: extend the cathedral
autopilot ranker with predicted-delta uncertainty so risk-adjusted ranking
can prefer empirically-anchored predictions over hand-waved ones.

The canonical estimator is a **conjugate-prior Normal-Inverse-Gamma** scheme
because:

1. It is the canonical Bayesian small-sample form for an unknown-mean
   unknown-variance Gaussian (predicted-vs-empirical residuals on a
   typical dispatch wave have 1-8 anchors per equation; bootstrap variance
   is degenerate at n<3, while NIG yields a well-defined posterior at n>=1).
2. The hyperparameters (alpha_0 = 1.0, beta_0 = sample-variance-floor)
   are interpretable: a single anchor yields a finite posterior with
   uncertainty inflated by the prior, matching CLAUDE.md "Forbidden score
   claims" intuition that one measurement is not zero-uncertainty.
3. Closed-form posterior std avoids the Monte-Carlo non-determinism that
   would violate CLAUDE.md "Beauty, simplicity, and developer experience"
   ("Deterministic reproducibility" + "Strategic Secrecy" both demand
   that the same inputs produce the same uncertainty number).

Per Catalog #287 + Catalog #323 canonical Provenance: every returned
uncertainty estimate is paired with the source equation_id + anchor_ids
+ measurement_method so the operator can audit residual lineage.

Wire-in hooks per Catalog #125 (consumer-side; producer is CanonicalEquation):
  hook #1 sensitivity-map = N/A (consumer-side, not producer)
  hook #2 Pareto constraint = ACTIVE (uncertainty bounds inform Pareto)
  hook #3 bit-allocator = N/A
  hook #4 cathedral autopilot dispatch = ACTIVE (SLOT MG-1 deliverable)
  hook #5 continual-learning posterior = ACTIVE (re-fits on new anchor)
  hook #6 probe-disambiguator = ACTIVE (uncertainty IS the disambiguator)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


# Conjugate-prior hyperparameters. Small alpha_0 = weak prior (let anchors
# dominate as they arrive). Non-zero beta_0 = floor on posterior variance
# so a single anchor cannot claim zero uncertainty per CLAUDE.md "Forbidden
# score claims".
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 1e-6  # minimal floor; in score-delta units (typically ~1e-3)
_PRIOR_KAPPA = 1.0  # prior pseudo-count for the mean

# Minimum number of anchors for the empirical-bootstrap fallback to fire.
# Below this we rely on the NIG posterior + prior so the result is always
# well-defined (never NaN).
_BOOTSTRAP_MIN_ANCHORS = 3


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Canonical typed uncertainty estimate from empirical anchor residuals.

    Fields:
        posterior_std: 1-sigma posterior standard deviation of the
            predicted-delta. Strictly positive (NIG floor ensures non-zero).
        n_anchors_consumed: number of EmpiricalAnchor rows that fed the
            estimate. Zero means pure-prior (maximal uncertainty); 1+ means
            posterior-updated.
        method: "normal_inverse_gamma_posterior" (canonical default) or
            "empirical_bootstrap" (when n_anchors >= _BOOTSTRAP_MIN_ANCHORS
            AND the caller explicitly opted in via empirical_bootstrap=True).
        equation_id: source CanonicalEquation.equation_id (for audit).
        anchor_ids: tuple of EmpiricalAnchor.anchor_id values consumed
            (for cite-chain per CLAUDE.md "Subagent coherence-by-default").
    """

    posterior_std: float
    n_anchors_consumed: int
    method: str
    equation_id: str
    anchor_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.posterior_std, (int, float)):
            raise ValueError("posterior_std must be numeric")
        if self.posterior_std != self.posterior_std:  # NaN
            raise ValueError("posterior_std must not be NaN")
        if self.posterior_std <= 0.0:
            raise ValueError(
                f"posterior_std must be strictly positive (got {self.posterior_std}); "
                "NIG prior floor should make this unreachable"
            )
        if not isinstance(self.n_anchors_consumed, int) or self.n_anchors_consumed < 0:
            raise ValueError("n_anchors_consumed must be a non-negative int")
        if self.method not in {"normal_inverse_gamma_posterior", "empirical_bootstrap"}:
            raise ValueError(
                f"method must be one of "
                f"{{'normal_inverse_gamma_posterior', 'empirical_bootstrap'}}, got {self.method}"
            )
        if not isinstance(self.equation_id, str) or not self.equation_id:
            raise ValueError("equation_id must be a non-empty string")
        if not isinstance(self.anchor_ids, tuple):
            raise ValueError("anchor_ids must be a tuple")
        if len(self.anchor_ids) != self.n_anchors_consumed:
            raise ValueError(
                f"len(anchor_ids)={len(self.anchor_ids)} must match "
                f"n_anchors_consumed={self.n_anchors_consumed}"
            )


def _normal_inverse_gamma_posterior_std(residuals: Sequence[float]) -> float:
    """Closed-form NIG posterior std for an unknown-mean unknown-variance Gaussian.

    Posterior parameters under conjugate prior NIG(mu_0=0, kappa_0, alpha_0, beta_0):
      n = len(residuals)
      sample_mean = mean(residuals)
      sample_ss = sum((r - sample_mean) ** 2)
      alpha_n = alpha_0 + n / 2
      beta_n  = beta_0 + 0.5 * sample_ss
                + 0.5 * (kappa_0 * n / (kappa_0 + n)) * (sample_mean - 0) ** 2

    Posterior marginal variance of the mean follows a scaled inverse-chi-square;
    the canonical point estimate of std is sqrt(beta_n / (alpha_n - 0.5)) when
    alpha_n > 0.5 (always true given _PRIOR_ALPHA = 1.0).
    """
    n = len(residuals)
    if n == 0:
        # Pure prior: posterior std collapses to sqrt(beta_0 / (alpha_0 - 0.5))
        # but with the kappa_0 = 1 prior mean variance inflation.
        return math.sqrt(_PRIOR_BETA / (_PRIOR_ALPHA - 0.5)) + 1.0
        # The "+ 1.0" pure-prior inflation reflects CLAUDE.md "Forbidden score
        # claims": zero anchors = maximum uncertainty, ranker should treat as
        # speculative. The exact constant is calibrated so risk-aversion
        # lambda=1 fully neutralizes a no-anchor prediction.

    sample_mean = sum(residuals) / n
    sample_ss = sum((r - sample_mean) ** 2 for r in residuals)

    alpha_n = _PRIOR_ALPHA + n / 2.0
    beta_n = (
        _PRIOR_BETA
        + 0.5 * sample_ss
        + 0.5 * (_PRIOR_KAPPA * n / (_PRIOR_KAPPA + n)) * (sample_mean ** 2)
    )

    # Canonical NIG point std-estimate. alpha_n > 0.5 always given _PRIOR_ALPHA=1.0.
    if alpha_n <= 0.5:
        raise RuntimeError(
            f"NIG alpha_n={alpha_n} <= 0.5 unreachable given _PRIOR_ALPHA={_PRIOR_ALPHA}; "
            "indicates module-level constant tampering"
        )
    return math.sqrt(beta_n / (alpha_n - 0.5))


def _empirical_bootstrap_std(residuals: Sequence[float]) -> float:
    """Sample standard deviation (Bessel-corrected) of the residuals.

    Requires len(residuals) >= 2. The caller is responsible for ensuring
    n >= _BOOTSTRAP_MIN_ANCHORS before invoking this; below that threshold
    the bootstrap is degenerate and the NIG posterior is preferred.
    """
    n = len(residuals)
    if n < 2:
        raise ValueError(
            f"empirical bootstrap requires n >= 2 residuals, got {n}; "
            f"use NIG posterior for small-sample regime"
        )
    sample_mean = sum(residuals) / n
    sample_var = sum((r - sample_mean) ** 2 for r in residuals) / (n - 1)
    return math.sqrt(max(sample_var, _PRIOR_BETA))


def predicted_delta_uncertainty_from_empirical_anchors(
    equation_id: str,
    anchors: Iterable,
    *,
    empirical_bootstrap: bool = False,
) -> UncertaintyEstimate:
    """Compute 1-sigma posterior std of a predicted-delta from EmpiricalAnchor residuals.

    The default estimator is the canonical Normal-Inverse-Gamma conjugate
    posterior (well-defined at all n >= 0). The empirical bootstrap (Bessel-
    corrected sample std) is available as an opt-in alternative when
    n >= 3 AND the caller wants a frequentist comparison row.

    Args:
        equation_id: source CanonicalEquation identifier. Stamped into the
            returned UncertaintyEstimate for audit cite-chain.
        anchors: iterable of EmpiricalAnchor objects (or anchor-shaped objects
            with `.anchor_id: str` and `.residual: float` attributes). Per
            tac.canonical_equations.equation.EmpiricalAnchor.__post_init__,
            residual is always >= 0 (normalized magnitude).
        empirical_bootstrap: opt-in flag. When True AND n >= 3, uses
            Bessel-corrected sample std instead of the NIG posterior. Default
            False (canonical NIG).

    Returns:
        UncertaintyEstimate with posterior_std > 0, n_anchors_consumed,
        method label, equation_id, and tuple of anchor_ids.

    Raises:
        ValueError: if anchors contains non-numeric residuals OR equation_id
            is empty. Per CLAUDE.md "Comment-only contracts are FORBIDDEN":
            invariants enforced at construction, not docstring-only.

    Per CLAUDE.md "Forbidden score claims": the returned uncertainty is
    ALWAYS non-zero (NIG prior floor + pure-prior inflation at n=0). A
    consumer that wants "uncertainty == 0 means promotable" should use
    the canonical Provenance + custody fields per Catalog #323, NOT this
    uncertainty estimate.
    """
    if not isinstance(equation_id, str) or not equation_id.strip():
        raise ValueError("equation_id must be a non-empty string")

    anchor_list = list(anchors)
    residuals: list[float] = []
    anchor_ids: list[str] = []
    for anchor in anchor_list:
        # Duck-typed: accept EmpiricalAnchor or any object with .residual + .anchor_id.
        if not hasattr(anchor, "residual"):
            raise ValueError(f"anchor missing .residual attribute: {type(anchor).__name__}")
        if not hasattr(anchor, "anchor_id"):
            raise ValueError(f"anchor missing .anchor_id attribute: {type(anchor).__name__}")
        r = anchor.residual
        if not isinstance(r, (int, float)):
            raise ValueError(f"anchor.residual must be numeric, got {type(r).__name__}")
        if r != r:  # NaN
            raise ValueError("anchor.residual must not be NaN")
        residuals.append(float(r))
        anchor_ids.append(str(anchor.anchor_id))

    n = len(residuals)

    if empirical_bootstrap and n >= _BOOTSTRAP_MIN_ANCHORS:
        std = _empirical_bootstrap_std(residuals)
        method = "empirical_bootstrap"
    else:
        std = _normal_inverse_gamma_posterior_std(residuals)
        method = "normal_inverse_gamma_posterior"

    return UncertaintyEstimate(
        posterior_std=std,
        n_anchors_consumed=n,
        method=method,
        equation_id=equation_id,
        anchor_ids=tuple(anchor_ids),
    )
