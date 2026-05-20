"""Contest-scorer Blahut-Arimoto R(D) specialization.

Per WAVE-3-PATH-A.2 closure of PATH-A's documented deeper gap: PATH-A's
``categorical_posterior_capacity_vs_continuous_gaussian_v1`` registered the
**necessary-condition** Shannon R(D) lower bound (proxied via MSE under
Gumbel-Softmax low-temperature asymptote per Cover & Thomas §13.3.2);
this module computes the **actual achievable** R(D) curve over the
contest's specific seg+pose distortion measure by querying a distortion
oracle (a callable that maps per-source-symbol reproductions to the
contest scorer's actual SegNet + PoseNet response).

Math contract
=============

The contest score (per ``upstream/evaluate.py:92`` + CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" non-negotiable)::

    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489

is NOT a single MSE per source symbol; it is a per-pair composite that
the contest scorer computes by running SegNet + PoseNet on the
reconstructed frames. To compute the achievable R(D) curve for an
encoder with reproduction alphabet ``Y_alphabet``, we need a
**distortion oracle** ``oracle : (x, y) -> R_+`` that returns the
expected contribution to the contest distortion for each
(source-symbol, reproduction-symbol) pair.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog
#318 (master-gradient raw-byte-authority guard): the oracle MUST NOT
expose raw archive-byte / bit responses; it MUST be a TYPED operator on
the source alphabet that returns a finite distortion measure.

For practical use, the oracle is typically built by sampling a small
``(x, y)`` table with paired contest auth-eval runs OR a Hinton-distilled
surrogate scorer (per CLAUDE.md "eval_roundtrip" non-negotiable's
gradient-reachability requirement).

This module emits the R(D) curve given that oracle; producing the
oracle itself is a separately operator-routable Phase B2 task per
CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical
dispatch" non-negotiable.

Cross-references
================

* PATH-A canonical equation
  ``categorical_posterior_capacity_vs_continuous_gaussian_v1`` (necessary-
  condition lower bound).
* This module's sister sub-equation
  ``categorical_blahut_arimoto_rate_distortion_v1`` (actual achievable
  R(D) curve).
* Cover & Thomas §10.8 + Theorem 13.4.1 + §13.3.2.
* Catalog #318 master-gradient raw-byte-authority guard.
* CLAUDE.md "Apples-to-apples evidence discipline" + "Submission auth
  eval — BOTH CPU AND CUDA" + "eval_roundtrip" non-negotiables.

[verified-against: contest formula at upstream/evaluate.py:92;
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
    "ContestScorerDistortionOracle",
    "build_distortion_matrix_from_oracle",
    "iterate_contest_scorer_rd",
)


# Type alias: oracle maps (source_index, reproduction_index) -> distortion.
# The canonical contract is a callable accepting (x_index, y_index) and
# returning a non-negative float in the contest distortion units.
ContestScorerDistortionOracle = Callable[[int, int], float]


def build_distortion_matrix_from_oracle(
    oracle: ContestScorerDistortionOracle,
    n_source_symbols: int,
    n_reproduction_symbols: int,
) -> np.ndarray:
    """Materialize a ``(|X|, |Y|)`` distortion matrix by querying the oracle.

    Per Cover & Thomas §10.8 the BA iteration requires the full distortion
    matrix at iteration time. For substrates with large alphabets the
    oracle queries can be expensive (each is a contest scorer forward);
    callers may want to memoize / pre-compute / sample-and-extrapolate
    rather than calling this eagerly.

    Parameters
    ----------
    oracle:
        Callable returning the per-(x, y) distortion. Must be non-negative.
    n_source_symbols: ``|X|`` (e.g. K=256 for C6 Path B2 single group).
    n_reproduction_symbols: ``|Y|`` (typically equal to |X| for
        autoencoder-style codecs).

    Returns
    -------
    ``(|X|, |Y|)`` ``float64`` distortion matrix.

    Notes
    -----
    The oracle is called ``|X| * |Y|`` times. For ``K=256`` this is
    65,536 calls — operationally significant if the oracle is a paired
    contest auth-eval. Production callers should typically prepare a
    Hinton-distilled surrogate per CLAUDE.md "differentiable_eval_roundtrip"
    before invoking this helper.

    [verified-against: contest formula composition at upstream/evaluate.py;
    canonical oracle-driven distortion-matrix pattern.]
    """
    if n_source_symbols < 2:
        raise ValueError(f"n_source_symbols must be >= 2, got {n_source_symbols}")
    if n_reproduction_symbols < 1:
        raise ValueError(f"n_reproduction_symbols must be >= 1, got {n_reproduction_symbols}")
    matrix = np.empty((n_source_symbols, n_reproduction_symbols), dtype=np.float64)
    for i in range(n_source_symbols):
        for j in range(n_reproduction_symbols):
            d = oracle(i, j)
            if not isinstance(d, (int, float)) or math.isnan(d):
                raise ValueError(
                    f"oracle({i}, {j}) returned {d!r}; must be finite numeric"
                )
            if d < 0:
                raise ValueError(
                    f"oracle({i}, {j}) returned {d}; must be >= 0 (distortion)"
                )
            matrix[i, j] = float(d)
    return matrix


def iterate_contest_scorer_rd(
    distortion_oracle: ContestScorerDistortionOracle,
    *,
    n_source_symbols: int,
    n_reproduction_symbols: int | None = None,
    source_distribution: np.ndarray | Sequence[float] | None = None,
    lambda_values: Sequence[float] | None = None,
    n_lambda_points: int = DEFAULT_LAMBDA_SWEEP_POINTS,
    source_distribution_summary: str | None = None,
) -> RateDistortionCurve:
    """R(D) curve via BA iteration on the contest scorer's distortion oracle.

    Per Cover & Thomas §10.8 + the contest's specific seg+pose distortion
    composition: the achievable R(D) curve depends on (a) the source
    distribution (typically the per-pair distribution of contest frames)
    and (b) the distortion oracle (the contest scorer's response to each
    reproduction symbol).

    This helper is the operator-routable closure path for PATH-A's
    deeper gap. PATH-A's necessary-condition lower bound (Shannon R(D)
    via MSE proxy) is conservative; the BA iteration on the actual
    contest oracle returns the curve a substrate's encoder + decoder
    MUST satisfy to be Pareto-feasible per CLAUDE.md
    "Meta-Lagrangian/Pareto solver" non-negotiable.

    Parameters
    ----------
    distortion_oracle:
        Callable ``(x_index, y_index) -> non-negative float``.
        Caller's responsibility to provide a deterministic, reviewable
        oracle (canonical patterns: Hinton-distilled surrogate scorer
        per CLAUDE.md "differentiable_eval_roundtrip"; OR sampled
        empirical distortion table from a Modal smoke).
    n_source_symbols: ``|X|`` (e.g. K=256 for C6 Path B2 single group).
    n_reproduction_symbols: ``|Y|``; defaults to ``n_source_symbols`` for
        autoencoder-style codec (encoder + decoder share alphabet).
    source_distribution: optional ``(|X|,)`` prior; defaults to uniform.
    lambda_values: optional explicit slope sweep.
    n_lambda_points: number of sweep samples; defaults to
        ``DEFAULT_LAMBDA_SWEEP_POINTS`` = 16.
    source_distribution_summary: optional human-readable citation.

    Returns
    -------
    Typed ``RateDistortionCurve`` with canonical Provenance per Catalog #323;
    ``derivation_method = "blahut_arimoto_contest_scorer_v1"``.

    [verified-against: contest formula at upstream/evaluate.py:92;
    canonical R(D) iteration per Cover & Thomas §10.8;
    [prediction] not empirical anchor]
    """
    if n_source_symbols < 2:
        raise ValueError(f"n_source_symbols must be >= 2, got {n_source_symbols}")
    if n_reproduction_symbols is None:
        n_reproduction_symbols = n_source_symbols
    if n_reproduction_symbols < 1:
        raise ValueError(f"n_reproduction_symbols must be >= 1")

    # Materialize the distortion matrix via oracle queries
    distortion_matrix = build_distortion_matrix_from_oracle(
        distortion_oracle, n_source_symbols, n_reproduction_symbols
    )

    if source_distribution is None:
        prior = np.full(n_source_symbols, 1.0 / n_source_symbols, dtype=np.float64)
    else:
        prior = np.asarray(source_distribution, dtype=np.float64)
        if prior.shape != (n_source_symbols,):
            raise ValueError(
                f"source_distribution shape {prior.shape} != ({n_source_symbols},)"
            )
        if not math.isclose(float(prior.sum()), 1.0, abs_tol=1e-9):
            raise ValueError(f"source_distribution must sum to 1, got {prior.sum()}")

    if source_distribution_summary is None:
        source_distribution_summary = (
            f"contest-scorer-oracle |X|={n_source_symbols} "
            f"|Y|={n_reproduction_symbols} / "
            f"distortion-matrix-mean={float(distortion_matrix.mean()):.6f} / "
            f"oracle={getattr(distortion_oracle, '__name__', repr(distortion_oracle))}"
        )

    return iterate_rate_distortion(
        distortion_matrix,
        prior,
        lambda_values=lambda_values,
        source_distribution_summary=source_distribution_summary,
        derivation_method="blahut_arimoto_contest_scorer_v1",
    )
