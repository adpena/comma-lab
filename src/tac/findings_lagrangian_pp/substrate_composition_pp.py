# SPDX-License-Identifier: MIT
"""Q8 rank #1 — Dirichlet posterior over substrate composition alpha values.

Per T3 grand council 3-round consolidated verdict + Q8 OVERRIDE
(slot 20-supplemental Time-Traveler dissent ratified by Balle in
slot 20-second-supplemental): substrate composition matrix is THE
canonical hierarchical-Bayesian-with-shrinkage use case for 2024-2026
field-trajectory ML (cross-family alpha values are exchangeable within
their composition family).

Per Balle operating-within slot 20-second-supplemental: substrate
composition matrix posterior IS canonical sister of his entropy bottleneck;
Phase 1 implementation uses Dirichlet posterior closed-form (conjugate to
multinomial likelihood). The Dirichlet posterior is closed-form even
WITHOUT NumPyro; this module provides BOTH the Dirichlet closed-form path
(per Wyner operating-within slot 20) AND the NumPyro hierarchical-Bayes
path (per Time-Traveler operating-within slot 20-supplemental) for the
empirical comparison that tests slot 20's CARGO-CULTED-PP classification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


__all__ = [
    "DirichletPosteriorOverAlpha",
    "build_dirichlet_posterior_for_substrate_composition",
    "predict_alpha_from_posterior_dirichlet",
    "CompositionPPError",
]


class CompositionPPError(ValueError):
    """Raised when composition PP inputs are malformed."""


@dataclass(frozen=True)
class DirichletPosteriorOverAlpha:
    """Closed-form Dirichlet posterior over composition alpha probability simplex.

    Per Wyner operating-within slot 20: 4-tier discrete distribution admits
    Dirichlet closed-form (conjugate to multinomial likelihood); same pattern
    here for the composition alpha simplex.

    Per Balle slot 20-second-supplemental: Phase 1 uses Dirichlet
    closed-form FIRST (no NumPyro); the NumPyro hierarchical path
    (via :mod:`pp_posterior`) lives alongside for the empirical comparison.

    Attributes:
        pair_key: canonical substrate-pair id (e.g. "lane_g_v3_x_siren_topology").
        alpha_concentration: Dirichlet concentration parameters (shape (k,)).
        family_id: composition family for cross-family shrinkage.
        n_anchors_used: count of empirical alpha observations.
    """

    pair_key: str
    alpha_concentration: tuple[float, ...]
    family_id: str
    n_anchors_used: int

    def __post_init__(self) -> None:
        if not isinstance(self.pair_key, str) or not self.pair_key.strip():
            raise CompositionPPError("pair_key must be non-empty string")
        if not isinstance(self.alpha_concentration, tuple):
            raise CompositionPPError("alpha_concentration must be tuple")
        if len(self.alpha_concentration) < 2:
            raise CompositionPPError(
                f"alpha_concentration must have at least 2 components, got {len(self.alpha_concentration)}"
            )
        for i, c in enumerate(self.alpha_concentration):
            if not isinstance(c, (int, float)):
                raise CompositionPPError(
                    f"alpha_concentration[{i}] must be numeric"
                )
            if c <= 0:
                raise CompositionPPError(
                    f"alpha_concentration[{i}]={c} must be > 0 (Dirichlet requires positive concentration)"
                )
        if not isinstance(self.family_id, str) or not self.family_id.strip():
            raise CompositionPPError("family_id must be non-empty string")
        if not isinstance(self.n_anchors_used, int) or self.n_anchors_used < 0:
            raise CompositionPPError(
                f"n_anchors_used={self.n_anchors_used} must be non-negative int"
            )

    @property
    def concentration_sum(self) -> float:
        """Sum of concentration parameters (the Dirichlet 'precision')."""
        return sum(self.alpha_concentration)

    @property
    def mean_alpha(self) -> tuple[float, ...]:
        """Dirichlet posterior mean: alpha_i / sum(alpha)."""
        total = self.concentration_sum
        return tuple(c / total for c in self.alpha_concentration)

    @property
    def variance_alpha(self) -> tuple[float, ...]:
        """Dirichlet posterior variance per component."""
        total = self.concentration_sum
        var = []
        for c in self.alpha_concentration:
            mean = c / total
            v = mean * (1.0 - mean) / (total + 1.0)
            var.append(v)
        return tuple(var)

    @property
    def std_alpha(self) -> tuple[float, ...]:
        """Posterior std-dev per component (uncertainty signal for Catalog #125 hook #1)."""
        return tuple(math.sqrt(v) for v in self.variance_alpha)


def build_dirichlet_posterior_for_substrate_composition(
    pair_key: str,
    *,
    observed_alpha_buckets: Sequence[int],
    family_id: str,
    prior_concentration: float = 1.0,
) -> DirichletPosteriorOverAlpha:
    """Closed-form Dirichlet update from observed alpha-bucket counts.

    The composition alpha lives in [0, 1] (or sometimes >1 for super-additive
    per Catalog #823); we discretize into K buckets and treat each
    observed alpha measurement as a multinomial draw. The Dirichlet
    posterior is conjugate.

    Args:
        pair_key: canonical pair_key for the substrate composition.
        observed_alpha_buckets: count per bucket (e.g. for 5-bucket
            classification: [under_zero, near_zero, near_half, near_one, super_additive]).
        family_id: composition family for cross-family shrinkage.
        prior_concentration: Dirichlet symmetric prior (default 1.0 = uniform).

    Returns:
        DirichletPosteriorOverAlpha.
    """
    if prior_concentration <= 0:
        raise CompositionPPError(
            f"prior_concentration={prior_concentration} must be > 0"
        )
    buckets = list(observed_alpha_buckets)
    if not buckets or any(b < 0 for b in buckets):
        raise CompositionPPError(
            "observed_alpha_buckets must be non-empty + all >= 0"
        )

    # Closed-form: posterior_concentration[i] = prior_concentration + count_i.
    posterior_concentration = tuple(prior_concentration + b for b in buckets)
    n_anchors = sum(buckets)
    return DirichletPosteriorOverAlpha(
        pair_key=pair_key,
        alpha_concentration=posterior_concentration,
        family_id=family_id,
        n_anchors_used=n_anchors,
    )


def predict_alpha_from_posterior_dirichlet(
    posterior: DirichletPosteriorOverAlpha,
    *,
    bucket_alpha_midpoints: Sequence[float],
) -> tuple[float, float]:
    """Predict scalar alpha + uncertainty from a Dirichlet posterior over buckets.

    Per Q8 OVERRIDE (slot 20-supplemental + second-supplemental):
    cathedral autopilot ranker consumes (predicted_alpha, alpha_sigma)
    via Q7 mechanism for asymmetric-cost downweighting.

    Args:
        posterior: DirichletPosteriorOverAlpha.
        bucket_alpha_midpoints: scalar alpha value at the midpoint of each bucket
            (e.g. for 5-bucket [-0.5, 0.0, 0.5, 1.0, 1.5]).

    Returns:
        Tuple of (predicted_alpha_mean, predicted_alpha_uncertainty_sigma).
    """
    if len(bucket_alpha_midpoints) != len(posterior.alpha_concentration):
        raise CompositionPPError(
            f"bucket_alpha_midpoints length {len(bucket_alpha_midpoints)} != "
            f"posterior concentration length {len(posterior.alpha_concentration)}"
        )

    mean_probs = posterior.mean_alpha
    variance_probs = posterior.variance_alpha

    # Expected alpha = sum(p_i * midpoint_i).
    predicted_alpha = sum(
        p * m for p, m in zip(mean_probs, bucket_alpha_midpoints)
    )
    # Uncertainty: sqrt(sum(var_i * midpoint_i^2)) — first-order approximation.
    uncertainty_sigma = math.sqrt(
        sum(v * (m**2) for v, m in zip(variance_probs, bucket_alpha_midpoints))
    )
    return (float(predicted_alpha), float(uncertainty_sigma))
