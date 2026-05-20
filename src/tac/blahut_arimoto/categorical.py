"""Categorical-alphabet Blahut-Arimoto R(D) specialization for RSSM-class substrates.

Per WAVE-3-PATH-A.2 closure of PATH-A's deeper gap for the DreamerV3 RSSM
categorical-posterior paradigm: PATH-A's
``categorical_posterior_capacity_vs_continuous_gaussian_v1`` registered the
**capacity ceiling** ``H(T) = G * log2(K)`` plus the Shannon necessary-condition
R(D) lower bound; this module computes the **achievable** R(D) curve over the
categorical alphabet by composing the canonical BA iteration with the
``G`` independent groups × ``K`` categories structure.

Math contract
=============

The DreamerV3 RSSM categorical posterior is a product of ``G`` independent
categorical variables, each over ``K`` categories. For an independent-source
product, the joint rate is the sum of per-group rates::

    R_joint(D) = sum_{g=1}^{G} R_g(D_g)         (for D = sum_g D_g)

Per Cover & Thomas Theorem 9.6.1 (sum-rate property for independent sources)
the achievable rate over the product distribution decomposes additively. For
a homogeneous configuration (every group has the same K + same
distortion-measure) this simplifies to::

    R_joint(D) = G * R_single_group(D / G)

The per-group R(D) is the canonical Blahut-Arimoto output on the ``|X| = K``
categorical source with the user-supplied per-symbol distortion measure.

Per CLAUDE.md "Subagent coherence-by-default" hook #2 Pareto constraint:
this curve is the structural achievability bound any RSSM-class substrate
must satisfy at its chosen ``(G, K, distortion_measure)`` configuration.

Cross-references
================

* PATH-A canonical equation ``categorical_posterior_capacity_vs_continuous_gaussian_v1``
  registers ``H(T) = G * log2(K)`` capacity ceiling.
* This module's sister sub-equation
  ``categorical_blahut_arimoto_rate_distortion_v1`` registers the
  achievable R(D) iteration that REFINES PATH-A's lower bound into an
  exact achievable curve for any specified distortion measure.
* Cover & Thomas 2nd ed Theorem 9.6.1 (sum-rate property).
* Jang et al. 2017 arXiv:1611.01144 (Gumbel-Softmax categorical relaxation).
* Hafner et al. 2024 arXiv:2301.04104 (DreamerV3 canonical 32x32 config).

[verified-against: Cover & Thomas Theorem 9.6.1; PATH-A sister equation
canonical-config table (Hafner 32x32 + C6 Path B2 24x256);
[prediction] iteration output is deterministic, not empirical anchor]
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

from tac.blahut_arimoto.canonical import (
    DEFAULT_LAMBDA_SWEEP_POINTS,
    RateDistortionCurve,
    iterate_rate_distortion,
)


__all__ = (
    "DEFAULT_DREAMERV3_HAFNER_G",
    "DEFAULT_DREAMERV3_HAFNER_K",
    "DEFAULT_C6_PATH_B2_G",
    "DEFAULT_C6_PATH_B2_K",
    "default_categorical_msl_distortion",
    "iterate_categorical_rd",
)


# Per PATH-A canonical-config table — see equation
# ``categorical_posterior_capacity_vs_continuous_gaussian_v1.domain_of_validity.canonical_configs``
DEFAULT_DREAMERV3_HAFNER_G: Final[int] = 32
DEFAULT_DREAMERV3_HAFNER_K: Final[int] = 32
DEFAULT_C6_PATH_B2_G: Final[int] = 24
DEFAULT_C6_PATH_B2_K: Final[int] = 256


def default_categorical_msl_distortion(K: int) -> np.ndarray:
    """Canonical per-symbol categorical distortion: ``d(i, j) = 1 - delta_{ij}``.

    The canonical default distortion for a categorical-alphabet source is
    Hamming distortion: ``d(i, j) = 0`` if ``i == j``, else ``1``. This is
    the Cover & Thomas §10.5 canonical example and the maximum-likelihood-
    surrogate proxy for a softmax cross-entropy categorical posterior.

    Parameters
    ----------
    K: number of categories.

    Returns
    -------
    ``(K, K)`` symmetric 0/1 Hamming distortion matrix.

    Notes
    -----
    Callers with a substrate-specific distortion (e.g. the contest's
    SegNet+PoseNet pair-wise distance) should pass their own
    ``distortion_fn`` to ``iterate_categorical_rd`` rather than using
    this default.

    [verified-against: Cover & Thomas 2nd ed §10.5 (Hamming distortion
    canonical example); Jang et al. 2017 §3.1 (categorical relaxation
    surrogate cost).]
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    return 1.0 - np.eye(K, dtype=np.float64)


def iterate_categorical_rd(
    G: int,
    K: int,
    *,
    distortion_fn: Callable[[int], np.ndarray] | None = None,
    source_prior: np.ndarray | Sequence[float] | None = None,
    lambda_values: Sequence[float] | None = None,
    homogeneous_groups: bool = True,
    n_lambda_points: int = DEFAULT_LAMBDA_SWEEP_POINTS,
) -> RateDistortionCurve:
    """R(D) curve for a ``G``-group × ``K``-category source via BA iteration.

    Per the categorical specialization of Cover & Thomas §10.8 + Theorem 9.6.1:
    for ``G`` independent homogeneous categorical groups, the joint R(D)
    is ``G * R_single(D / G)``. This helper computes the single-group BA
    curve then scales it canonically.

    Per PATH-A's sister equation
    ``categorical_posterior_capacity_vs_continuous_gaussian_v1``: the
    capacity ceiling is ``H(T) = G * log2(K)`` (e.g. 160 bits for
    Hafner 32x32; 192 bits for C6 Path B2 24x256). The achievable R(D)
    returned here is bounded above by ``H(T)``.

    Parameters
    ----------
    G: number of independent categorical groups (Hafner canonical = 32;
        C6 Path B2 = 24).
    K: number of categories per group (Hafner canonical = 32;
        C6 Path B2 = 256).
    distortion_fn: optional callable ``K -> (K, K) ndarray``. If None,
        defaults to canonical Hamming distortion per
        ``default_categorical_msl_distortion``.
    source_prior: optional ``(K,)`` per-group prior. Defaults to uniform
        ``1/K`` per Catalog #292 max-entropy assumption (HARD-EARNED per
        T3 symposium Decision D for random-init categorical posteriors).
    lambda_values: optional explicit Lagrange-multiplier sweep; if None,
        defaults to the canonical adaptive sweep.
    homogeneous_groups: when True, scale single-group R(D) by ``G`` per
        the sum-rate property. When False, future API extension would
        accept heterogeneous per-group distortion_fn (not implemented;
        raises NotImplementedError).
    n_lambda_points: number of slope-sweep samples.

    Returns
    -------
    Typed ``RateDistortionCurve`` parameterized over the ``G * K`` joint
    alphabet, with ``rate_values`` and ``distortion_values`` already
    scaled by ``G`` (so they represent the per-sample joint quantities,
    matching PATH-A equation's ``H_total`` units).

    [verified-against: Cover & Thomas Theorem 9.6.1 sum-rate property;
    PATH-A canonical-config table; [prediction] not empirical anchor]
    """
    if G < 1:
        raise ValueError(f"G must be >= 1, got {G}")
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    if not homogeneous_groups:
        raise NotImplementedError(
            "heterogeneous_groups=False not yet implemented; future API "
            "extension would accept a sequence of per-group distortion matrices"
        )
    if n_lambda_points < 2:
        raise ValueError(f"n_lambda_points must be >= 2, got {n_lambda_points}")

    if distortion_fn is None:
        distortion_matrix = default_categorical_msl_distortion(K)
    else:
        distortion_matrix = np.asarray(distortion_fn(K), dtype=np.float64)
        if distortion_matrix.shape != (K, K):
            raise ValueError(
                f"distortion_fn returned shape {distortion_matrix.shape}, expected ({K}, {K})"
            )

    if source_prior is None:
        prior = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        prior = np.asarray(source_prior, dtype=np.float64)
        if prior.shape != (K,):
            raise ValueError(
                f"source_prior shape {prior.shape} != (K,) = ({K},)"
            )
        if not math.isclose(float(prior.sum()), 1.0, abs_tol=1e-9):
            raise ValueError(f"source_prior must sum to 1, got {prior.sum()}")

    # Single-group BA sweep — honor n_lambda_points if caller didn't pass
    # an explicit lambda_values sequence
    if lambda_values is None:
        from tac.blahut_arimoto.canonical import build_default_lambda_sweep

        lambda_values = build_default_lambda_sweep(
            distortion_matrix, n_points=n_lambda_points
        )

    summary = (
        f"categorical G={G} K={K} (single-group iterated then scaled by G); "
        f"distortion={'default_hamming' if distortion_fn is None else 'user_supplied'}; "
        f"prior={'uniform_max_entropy' if source_prior is None else 'user_supplied'}"
    )
    single = iterate_rate_distortion(
        distortion_matrix,
        prior,
        lambda_values=lambda_values,
        source_distribution_summary=summary,
        derivation_method=f"blahut_arimoto_categorical_v1_G{G}_K{K}",
    )

    # Scale by G per Theorem 9.6.1 sum-rate; lambda + distortion + converged
    # carry over unchanged (the per-group operating point), but rate scales
    # by G AND distortion scales by G (since D_joint = G * D_single in the
    # homogeneous-groups case).
    scaled_rates = tuple(G * r for r in single.rate_values)
    scaled_distortions = tuple(G * d for d in single.distortion_values)

    return RateDistortionCurve(
        lambda_values=single.lambda_values,
        rate_values=scaled_rates,
        distortion_values=scaled_distortions,
        converged=single.converged,
        canonical_provenance=single.canonical_provenance,
        derivation_method=f"blahut_arimoto_categorical_v1_G{G}_K{K}",
        source_distribution_summary=summary,
    )
