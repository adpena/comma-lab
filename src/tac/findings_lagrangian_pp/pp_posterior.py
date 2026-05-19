# SPDX-License-Identifier: MIT
"""NumPyro hierarchical posteriors over architectures (TRACK B per operator override).

Per operator-frontier-override 2026-05-19 verbatim *"we shoud pursue PP in
parallel"*: TRACK B implements hierarchical Bayesian posteriors via NumPyro
+ JAX-native HMC (NUTS) for parameter inference. Sister of TRACK A's
closed-form Gaussian (slot 20 binding); both emit through canonical
``tac.findings_lagrangian.unified.UnifiedPrediction`` interface.

The hierarchical model (per Time-Traveler operating-within
slot 20-supplemental + Balle operating-within slot 20-second-supplemental):

    # Hyperprior over architecture-family means
    family_mu ~ Normal(0, 1)            # per architecture family (PR101/PR106/A1/etc.)
    family_sigma ~ HalfCauchy(0, 1)     # family-level uncertainty
    # Per-architecture parameters (shrinkage toward family mean)
    theta_arch[i] ~ Normal(family_mu[arch_family(i)], family_sigma[arch_family(i)])
    # Per-anchor likelihood
    residual[n] ~ Normal(theta_arch[arch_id(n)], sigma_obs)

This is the canonical hierarchical-Bayes-with-shrinkage use case Time-Traveler
flagged as the highest-EV PP integration (Q8 OVERRIDE rank #1).

Per CLAUDE.md "Apples-to-apples evidence discipline": every TRACK B
prediction emits ``[predicted]`` axis tag; promotion to ``[contest-CUDA]`` /
``[contest-CPU]`` requires paired empirical anchor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from tac.findings_lagrangian.unified import ScalarPrediction
from tac.findings_lagrangian_pp._optional_numpyro import (
    TRACK_B_NUMPYRO_AVAILABLE,
    TrackBNumPyroUnavailableError,
    require_numpyro,
)


__all__ = [
    "PPHierarchicalPosterior",
    "hierarchical_posterior_via_nuts",
    "predict_from_hierarchical_posterior",
    "PPPosteriorError",
]


class PPPosteriorError(ValueError):
    """Raised when PP posterior inputs are malformed."""


@dataclass(frozen=True)
class PPHierarchicalPosterior:
    """NumPyro-derived hierarchical posterior approximation.

    Per Time-Traveler operating-within: cross-family alpha values are
    exchangeable within their composition family (A1xB family != PR101xB
    family != DP1xB family); the hierarchical Bayes posterior captures
    cross-family variance + within-family shrinkage that point estimates
    HIDE.

    Per Balle operating-within: this is canonical sister of his entropy
    bottleneck (Phase 3 endpoint per slot 20 Q5 reactivation criterion).

    Attributes:
        equation_id: pinned canonical_equation_id.
        family_mu_posterior_samples: shape (n_samples, n_families) hyperprior samples.
        theta_arch_posterior_samples: shape (n_samples, n_architectures) per-arch samples.
        architecture_family_map: mapping arch_id -> family_id for cite-chain.
        n_samples_used: NUTS sample count.
        sampler_method: e.g. "NUTS" / "variational_advi" / etc.
    """

    equation_id: str
    family_mu_posterior_samples: tuple[tuple[float, ...], ...]
    theta_arch_posterior_samples: tuple[tuple[float, ...], ...]
    architecture_family_map: Mapping[str, str]
    n_samples_used: int
    sampler_method: str

    def __post_init__(self) -> None:
        if not isinstance(self.equation_id, str) or not self.equation_id.strip():
            raise PPPosteriorError("equation_id must be non-empty string")
        if not isinstance(self.family_mu_posterior_samples, tuple):
            raise PPPosteriorError("family_mu_posterior_samples must be tuple")
        if not isinstance(self.theta_arch_posterior_samples, tuple):
            raise PPPosteriorError("theta_arch_posterior_samples must be tuple")
        if not isinstance(self.architecture_family_map, Mapping):
            raise PPPosteriorError("architecture_family_map must be mapping")
        if not isinstance(self.n_samples_used, int) or self.n_samples_used <= 0:
            raise PPPosteriorError(
                f"n_samples_used={self.n_samples_used} must be > 0"
            )
        if not isinstance(self.sampler_method, str) or not self.sampler_method.strip():
            raise PPPosteriorError("sampler_method must be non-empty string")

    @property
    def n_families(self) -> int:
        """Number of architecture families (top-level hyperprior dim)."""
        if not self.family_mu_posterior_samples:
            return 0
        return len(self.family_mu_posterior_samples[0])

    @property
    def n_architectures(self) -> int:
        """Number of per-architecture parameters."""
        if not self.theta_arch_posterior_samples:
            return 0
        return len(self.theta_arch_posterior_samples[0])


def hierarchical_posterior_via_nuts(
    equation_id: str,
    *,
    anchor_residuals: Sequence[float],
    architecture_ids_per_anchor: Sequence[str],
    architecture_family_map: Mapping[str, str],
    n_samples: int = 500,
    n_warmup: int = 250,
    sigma_obs: float = 1.0,
    seed: int = 42,
) -> PPHierarchicalPosterior:
    """Run NumPyro NUTS to infer hierarchical posterior over architectures.

    Per Time-Traveler operating-within + Q8 OVERRIDE (slot 20-supplemental):
    composition matrix is the canonical hierarchical-Bayes-with-shrinkage
    use case. NUTS is the gold-standard HMC sampler for posterior inference;
    NumPyro provides JAX-native NUTS that runs ~100x faster than vanilla
    PyMC for this problem class.

    Args:
        equation_id: which canonical_equation this posterior tracks.
        anchor_residuals: empirical residuals per anchor.
        architecture_ids_per_anchor: arch_id for each anchor (same length).
        architecture_family_map: mapping arch_id -> family_id for hyperprior.
        n_samples: NUTS samples after warmup.
        n_warmup: NUTS warmup steps.
        sigma_obs: observation noise std-dev.
        seed: JAX PRNG seed for deterministic reproducibility.

    Returns:
        PPHierarchicalPosterior with sample matrices.

    Raises:
        TrackBNumPyroUnavailableError: if NumPyro / JAX not installed.
        PPPosteriorError: if inputs malformed.
    """
    numpyro, jax = require_numpyro()
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    import jax.numpy as jnp
    from jax import random as jrandom
    import numpy as np

    if len(anchor_residuals) != len(architecture_ids_per_anchor):
        raise PPPosteriorError(
            f"anchor_residuals length {len(anchor_residuals)} != "
            f"architecture_ids_per_anchor length {len(architecture_ids_per_anchor)}"
        )
    if len(anchor_residuals) == 0:
        raise PPPosteriorError("Cannot run NUTS on zero anchors")
    if sigma_obs <= 0:
        raise PPPosteriorError(f"sigma_obs={sigma_obs} must be > 0")

    # Canonicalize architecture + family IDs to integer indices.
    arch_ids_sorted = sorted(set(architecture_ids_per_anchor))
    arch_to_idx = {a: i for i, a in enumerate(arch_ids_sorted)}
    family_ids_sorted = sorted(set(architecture_family_map.get(a, "DEFAULT") for a in arch_ids_sorted))
    family_to_idx = {f: i for i, f in enumerate(family_ids_sorted)}
    n_archs = len(arch_ids_sorted)
    n_families = len(family_ids_sorted)
    arch_family_idx = np.array(
        [family_to_idx[architecture_family_map.get(a, "DEFAULT")] for a in arch_ids_sorted]
    )
    anchor_arch_idx = np.array([arch_to_idx[a] for a in architecture_ids_per_anchor])
    residuals_arr = np.array(list(anchor_residuals), dtype=np.float64)

    def hierarchical_model(residuals, anchor_arch_idx, arch_family_idx, n_archs, n_families):
        # Hyperprior over family means
        family_mu = numpyro.sample(
            "family_mu",
            dist.Normal(jnp.zeros(n_families), jnp.ones(n_families)),
        )
        family_sigma = numpyro.sample(
            "family_sigma",
            dist.HalfCauchy(scale=jnp.ones(n_families) * 1.0),
        )
        # Per-architecture parameters with family-level shrinkage
        theta_arch = numpyro.sample(
            "theta_arch",
            dist.Normal(
                family_mu[arch_family_idx],
                family_sigma[arch_family_idx],
            ),
        )
        # Per-anchor likelihood
        numpyro.sample(
            "obs",
            dist.Normal(theta_arch[anchor_arch_idx], sigma_obs),
            obs=residuals,
        )

    rng_key = jrandom.PRNGKey(seed)
    nuts_kernel = NUTS(hierarchical_model)
    mcmc = MCMC(
        nuts_kernel, num_samples=n_samples, num_warmup=n_warmup, progress_bar=False
    )
    mcmc.run(
        rng_key,
        residuals=residuals_arr,
        anchor_arch_idx=anchor_arch_idx,
        arch_family_idx=arch_family_idx,
        n_archs=n_archs,
        n_families=n_families,
    )

    samples = mcmc.get_samples()
    family_mu_samples = np.array(samples["family_mu"])  # (n_samples, n_families)
    theta_arch_samples = np.array(samples["theta_arch"])  # (n_samples, n_archs)

    return PPHierarchicalPosterior(
        equation_id=equation_id,
        family_mu_posterior_samples=tuple(
            tuple(float(x) for x in row) for row in family_mu_samples
        ),
        theta_arch_posterior_samples=tuple(
            tuple(float(x) for x in row) for row in theta_arch_samples
        ),
        architecture_family_map=dict(architecture_family_map),
        n_samples_used=n_samples,
        sampler_method="NUTS",
    )


def predict_from_hierarchical_posterior(
    posterior: PPHierarchicalPosterior,
    *,
    architecture_id: str | None = None,
) -> ScalarPrediction:
    """Return a TRACK B ScalarPrediction (median + posterior std-dev).

    Per Catalog #287 + #323: emits ``[predicted]`` axis tag with explicit
    source_track="track_b_numpyro" so the unified ensemble can distinguish.

    Args:
        posterior: PPHierarchicalPosterior from `hierarchical_posterior_via_nuts`.
        architecture_id: optional specific arch to predict for. If None,
            returns marginal prediction over ALL architectures (cross-family mean).

    Returns:
        ScalarPrediction with source_track="track_b_numpyro".
    """
    import numpy as np

    theta_samples = np.array(
        [list(row) for row in posterior.theta_arch_posterior_samples]
    )
    # If no specific arch, return marginal over all arch samples (flatten).
    if architecture_id is None:
        all_values = theta_samples.flatten()
    else:
        # Find architecture index in canonical sorted order.
        arch_ids_sorted = sorted(set(posterior.architecture_family_map.keys()))
        try:
            arch_idx = arch_ids_sorted.index(architecture_id)
        except ValueError:
            raise PPPosteriorError(
                f"architecture_id={architecture_id!r} not in posterior's architecture set"
            )
        all_values = theta_samples[:, arch_idx]

    predicted_value = float(np.median(all_values))
    uncertainty_sigma = float(np.std(all_values))

    return ScalarPrediction(
        predicted_value=predicted_value,
        uncertainty_sigma=uncertainty_sigma,
        axis_tag="[predicted]",
        source_track="track_b_numpyro",
        equation_id=posterior.equation_id,
        n_anchors_consumed=posterior.n_samples_used,
        rationale=(
            f"TRACK B hierarchical posterior median (NumPyro NUTS, "
            f"n_samples={posterior.n_samples_used}, "
            f"n_families={posterior.n_families}, "
            f"n_architectures={posterior.n_architectures}). "
            f"Per operator-frontier-override 2026-05-19 + Time-Traveler "
            f"operating-within slot 20-supplemental."
        ),
    )
