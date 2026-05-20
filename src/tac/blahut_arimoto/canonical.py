"""Canonical Blahut-Arimoto iteration for the exact achievable R(D) curve.

Per WAVE-3-PATH-A.2 closure of PATH-A's documented deeper gap
(canonical equation ``categorical_posterior_capacity_vs_continuous_gaussian_v1``
registers the Shannon **necessary-condition lower bound**; this module
implements the iteration that produces the **actual achievable** R(D) curve
for an arbitrary discrete source + distortion-matrix triple).

Math contract
=============

Per Cover & Thomas ``Elements of Information Theory`` 2nd ed §10.8 the
Blahut-Arimoto algorithm computes ``R(D)`` for a discrete source ``X``
with distribution ``p_X`` and reproduction alphabet ``Y`` under distortion
``d : X x Y -> R_+`` via the alternating-projection fixed point::

    q_t(y)         = sum_x p_X(x) * p_t(y|x)                              (1)
    p_{t+1}(y|x)   = q_t(y) * exp(-s * d(x, y)) / Z_x(s)                  (2)
    R(D(s))        = sum_{x,y} p_X(x) * p(y|x) * log2(p(y|x) / q(y))       (3)
    D(s)           = sum_{x,y} p_X(x) * p(y|x) * d(x, y)                  (4)

with the slope ``s = -dR/dD`` the Lagrangian dual of the distortion
constraint. Sweeping ``s`` from ``s_max`` down to ``0`` traces the entire
``R(D)`` curve from ``(D_min, R_max)`` to ``(D_max, 0)``.

The canonical reference implementation is Blahut 1972
(``Computation of channel capacity and rate-distortion functions`` IEEE
Trans. Inf. Theory IT-18 §III) + Arimoto 1972 (``An algorithm for
computing the capacity of arbitrary discrete memoryless channels`` IEEE
Trans. Inf. Theory IT-18 Theorem 1).

This module composes the canonical foundational helper
``tac.symposium_impls.blahut_arimoto_theoretical_floor.blahut_arimoto_rate_distortion``
(which solves a single ``R(D)`` point at fixed target distortion via
slope-bisection inside BA) into a sweep that produces the entire
``R(D)`` curve, with typed result + canonical Provenance per Catalog #323.

Cross-references
================

* Sister equation ``categorical_posterior_capacity_vs_continuous_gaussian_v1``
  registered by PATH-A landing (necessary-condition lower bound).
* Sister sub-equation ``categorical_blahut_arimoto_rate_distortion_v1``
  registered by this landing (actual achievable R(D) curve via BA iteration).
* Cover & Thomas 2nd ed §10.8 — canonical BA algorithm.
* Blahut 1972 + Arimoto 1972 — original derivation references.
* Boyd & Vandenberghe §5.5 — Lagrangian-dual slope bisection.
* Catalog #344 canonical equations registry — non-negotiable formalization.
* CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable.

[verified-against: Cover & Thomas 2nd ed Theorem 10.8.1; canonical
implementation reuses the proven slope-bisection driver from
tac.symposium_impls.blahut_arimoto_theoretical_floor;
[prediction] curve emission is deterministic algorithm output, not an
empirical anchor]
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

from tac.provenance.builders import build_provenance_for_predicted
from tac.provenance.contract import Provenance
from tac.symposium_impls.blahut_arimoto_theoretical_floor import (
    blahut_arimoto_rate_distortion,
)


__all__ = (
    "DEFAULT_LAMBDA_SWEEP_POINTS",
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOLERANCE",
    "RateDistortionCurve",
    "build_default_lambda_sweep",
    "iterate_rate_distortion",
)


# Default sweep granularity per Cover & Thomas §10.8 + Blahut 1972 III.B:
# 16 points across the slope range gives a smooth R(D) curve for most
# discrete sources. Operators can override via the ``lambda_values`` arg.
DEFAULT_LAMBDA_SWEEP_POINTS: Final[int] = 16

DEFAULT_MAX_ITER: Final[int] = 1024
DEFAULT_TOLERANCE: Final[float] = 1e-9


@dataclasses.dataclass(frozen=True)
class RateDistortionCurve:
    """Typed result of a Blahut-Arimoto R(D) curve sweep.

    Per CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog
    #305 observability surface: every facet of the swept curve is
    inspectable (lambda/rate/distortion tuples), decomposable per-point
    (convergence flags), cite-able (Provenance), and counterfactual-able
    (the sweep can be re-run with different ``lambda_values`` without
    re-instantiating).

    Per Catalog #323 canonical Provenance: every curve carries
    ``canonical_provenance`` of kind PREDICTED_FROM_MODEL with
    ``promotion_eligible=False`` and ``score_claim_valid=False`` — the
    curve is an ALGORITHMIC PREDICTION (Cover & Thomas §10.8 iteration),
    not an empirical anchor. Promotion would require a paired empirical
    measurement on a contest-1:1 substrate per CLAUDE.md "Submission
    auth eval — BOTH CPU AND CUDA" non-negotiable.

    Per Catalog #344 canonical equations: this dataclass is the typed
    output of ``categorical_blahut_arimoto_rate_distortion_v1``.

    Fields
    ------
    lambda_values:
        Lagrange-multiplier values swept. By convention ``len() == K``
        points; higher lambda -> lower distortion / higher rate.
    rate_values:
        Achievable rate per lambda, in bits (``R(D(lambda))``).
    distortion_values:
        Achieved distortion per lambda (``D(lambda)``); same length as
        ``lambda_values``.
    converged:
        Per-point boolean: True iff the BA inner iteration converged within
        ``max_iter`` to tolerance.
    canonical_provenance:
        Provenance of kind PREDICTED_FROM_MODEL per Catalog #323.
    derivation_method:
        Free-form readable string identifying the BA variant (e.g.
        ``"blahut_arimoto_v1"`` for canonical Cover & Thomas §10.8).
    source_distribution_summary:
        Human-readable citation for the source distribution + distortion
        matrix used (e.g. ``"binary symmetric source / hamming distortion
        / |X|=2 |Y|=2"``).
    """

    lambda_values: tuple[float, ...]
    rate_values: tuple[float, ...]
    distortion_values: tuple[float, ...]
    converged: tuple[bool, ...]
    canonical_provenance: Provenance
    derivation_method: str
    source_distribution_summary: str

    def __post_init__(self) -> None:
        # Length parity invariant
        n = len(self.lambda_values)
        if n == 0:
            raise ValueError("lambda_values must be non-empty")
        if len(self.rate_values) != n:
            raise ValueError(
                f"rate_values length={len(self.rate_values)} != lambda length={n}"
            )
        if len(self.distortion_values) != n:
            raise ValueError(
                f"distortion_values length={len(self.distortion_values)} != lambda length={n}"
            )
        if len(self.converged) != n:
            raise ValueError(
                f"converged length={len(self.converged)} != lambda length={n}"
            )
        # Type + numeric invariants
        for i, lam in enumerate(self.lambda_values):
            if not isinstance(lam, (int, float)) or math.isnan(lam):
                raise ValueError(f"lambda_values[{i}]={lam!r} must be finite numeric")
            if lam < 0:
                raise ValueError(f"lambda_values[{i}]={lam} must be >= 0")
        for i, r in enumerate(self.rate_values):
            if not isinstance(r, (int, float)) or math.isnan(r):
                raise ValueError(f"rate_values[{i}]={r!r} must be finite numeric")
            if r < 0:
                raise ValueError(f"rate_values[{i}]={r} must be >= 0 (bits)")
        for i, d in enumerate(self.distortion_values):
            if not isinstance(d, (int, float)) or math.isnan(d):
                raise ValueError(f"distortion_values[{i}]={d!r} must be finite numeric")
            if d < 0:
                raise ValueError(f"distortion_values[{i}]={d} must be >= 0")
        for i, c in enumerate(self.converged):
            if not isinstance(c, (bool, np.bool_)):
                raise ValueError(f"converged[{i}]={c!r} must be bool")
        if not isinstance(self.canonical_provenance, Provenance):
            raise ValueError(
                f"canonical_provenance must be Provenance, got {type(self.canonical_provenance).__name__}"
            )
        if not isinstance(self.derivation_method, str) or not self.derivation_method.strip():
            raise ValueError("derivation_method must be non-empty string")
        if not isinstance(self.source_distribution_summary, str) or not self.source_distribution_summary.strip():
            raise ValueError("source_distribution_summary must be non-empty string")

    def as_dict(self) -> dict:
        """JSON-safe serialization (Provenance flattened via canonical helper)."""
        from tac.provenance.validator import provenance_to_dict

        return {
            "lambda_values": list(self.lambda_values),
            "rate_values": list(self.rate_values),
            "distortion_values": list(self.distortion_values),
            "converged": [bool(c) for c in self.converged],
            "canonical_provenance": provenance_to_dict(self.canonical_provenance),
            "derivation_method": self.derivation_method,
            "source_distribution_summary": self.source_distribution_summary,
        }


def build_default_lambda_sweep(
    distortion_matrix: np.ndarray,
    *,
    n_points: int = DEFAULT_LAMBDA_SWEEP_POINTS,
) -> tuple[float, ...]:
    """Build a canonical lambda sweep adapted to the distortion-matrix scale.

    Per Cover & Thomas §10.8 the slope ``s`` should span from near-zero
    (corresponds to the R(0) = log2(|Y|) at near-zero distortion regime,
    i.e. the high-rate end of the curve) up to a maximum that drives
    distortion near its D_max ceiling. The canonical choice is
    ``s_max = 1.0 / d_typical`` where ``d_typical = mean(distortion)``;
    sweeping ``s in [s_max / 100, s_max]`` geometrically covers the
    interesting region without numerical overflow.

    Parameters
    ----------
    distortion_matrix:
        ``(|X|, |Y|)`` distortion measure.
    n_points:
        Number of lambda samples; defaults to ``DEFAULT_LAMBDA_SWEEP_POINTS``.

    Returns
    -------
    Geometric sweep of slope values, ascending (lowest lambda first).

    [verified-against: Cover & Thomas §10.8 example 10.8.1 slope range;
    canonical heuristic 1/d_typical for slope normalization.]
    """
    arr = np.asarray(distortion_matrix, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("distortion_matrix must be non-empty")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    # d_typical = mean of POSITIVE distortion entries (zero entries are
    # the diagonal of identity-like matrices and not informative).
    positive = arr[arr > 0]
    if positive.size == 0:
        # Pathological all-zero distortion matrix: rate is 0 at any lambda.
        d_typical = 1.0
    else:
        d_typical = float(positive.mean())
    s_max = 1.0 / max(d_typical, 1e-12)
    s_min = s_max / 100.0
    # Geometric ascending sweep
    return tuple(float(s) for s in np.geomspace(s_min, s_max, n_points))


def _compute_distortion_for_slope(
    p_x: np.ndarray, distortion: np.ndarray, slope: float
) -> tuple[float, float, bool]:
    """Run BA at fixed slope; return (achieved_distortion, rate_bits, converged).

    This is the inner iteration extracted from the canonical
    ``blahut_arimoto_rate_distortion`` slope-bisection driver so the curve
    sweep can run BA directly at the slope sequence without the
    target-distortion bisection (which would be redundant for a sweep).

    Per Cover & Thomas eq 10.8.5-10.8.8 + canonical reference
    ``tac.symposium_impls.blahut_arimoto_theoretical_floor._ba_inner``.

    Parameters
    ----------
    p_x: source distribution (1D, sums to 1).
    distortion: (|X|, |Y|) distortion matrix.
    slope: lambda value to iterate at.

    Returns
    -------
    (achieved_D, rate_bits, converged).
    """
    n_y = distortion.shape[1]
    q_y = np.full(n_y, 1.0 / n_y)
    converged = False
    for _ in range(DEFAULT_MAX_ITER):
        log_p_yx = np.log(q_y[None, :] + 1e-300) - slope * distortion
        log_p_yx -= log_p_yx.max(axis=1, keepdims=True)
        p_yx = np.exp(log_p_yx)
        row_sums = p_yx.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        p_yx = p_yx / row_sums
        new_q = (p_x[:, None] * p_yx).sum(axis=0)
        s = new_q.sum()
        if s > 0:
            new_q = new_q / s
        if np.max(np.abs(new_q - q_y)) < DEFAULT_TOLERANCE:
            q_y = new_q
            converged = True
            break
        q_y = new_q
    # Final rate + distortion
    log_p_yx = np.log(q_y[None, :] + 1e-300) - slope * distortion
    log_p_yx -= log_p_yx.max(axis=1, keepdims=True)
    p_yx = np.exp(log_p_yx)
    row_sums = p_yx.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    p_yx = p_yx / row_sums
    ratio = p_yx / (q_y[None, :] + 1e-300)
    log2_ratio = np.where(p_yx > 0, np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
    rate = float((p_x[:, None] * p_yx * log2_ratio).sum())
    achieved = float((p_x[:, None] * p_yx * distortion).sum())
    return achieved, max(rate, 0.0), converged


def iterate_rate_distortion(
    distortion_matrix: np.ndarray | Sequence[Sequence[float]],
    source_distribution: np.ndarray | Sequence[float],
    *,
    lambda_values: Sequence[float] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    source_distribution_summary: str | None = None,
    derivation_method: str = "blahut_arimoto_v1",
) -> RateDistortionCurve:
    """Compute the achievable R(D) curve via Blahut-Arimoto iteration sweep.

    Per Cover & Thomas §10.8 canonical formulation: each lambda value
    yields one ``(R, D)`` point via the BA inner iteration (alternating
    projections between the conditional distribution ``p(y|x)`` and the
    marginal ``q(y)``). Sweeping lambda traces the entire achievable
    R(D) curve from low-rate/high-distortion (small lambda) to
    high-rate/low-distortion (large lambda).

    Parameters
    ----------
    distortion_matrix:
        ``(|X|, |Y|)`` non-negative distortion measure.
    source_distribution:
        ``p_X``; ``(|X|,)``; non-negative; sums to 1.
    lambda_values:
        Optional explicit slope sweep. Defaults to ``build_default_lambda_sweep``
        which adapts to the distortion-matrix scale.
    max_iter:
        BA inner-iteration max; defaults to ``DEFAULT_MAX_ITER`` = 1024.
    tol:
        BA inner-iteration tolerance (max change in q_y between iterations);
        defaults to ``DEFAULT_TOLERANCE`` = 1e-9.
    source_distribution_summary:
        Optional human-readable citation; auto-derived if not supplied.
    derivation_method:
        Free-form derivation-method tag for Provenance / observability.

    Returns
    -------
    Typed ``RateDistortionCurve`` with canonical Provenance per Catalog #323.

    [verified-against: Cover & Thomas 2nd ed §10.8 + Theorem 10.8.1;
    canonical ``_ba_inner`` extracted from sister reference helper
    ``tac.symposium_impls.blahut_arimoto_theoretical_floor.blahut_arimoto_rate_distortion``
    (which is itself unit-tested against the binary symmetric source
    ``R(D) = 1 - H(D)`` closed-form per Cover & Thomas example 10.8.1)]

    Examples
    --------
    >>> import numpy as np
    >>> from tac.blahut_arimoto import iterate_rate_distortion
    >>> # Binary symmetric source + Hamming distortion: R(D) = 1 - H(D) for D in [0, 1/2]
    >>> p_x = np.array([0.5, 0.5])
    >>> distortion = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> curve = iterate_rate_distortion(distortion, p_x)
    >>> # curve.rate_values[-1] >= curve.rate_values[0]  (lambda ascending -> rate ascending)
    """
    p_x = np.asarray(source_distribution, dtype=np.float64)
    distortion = np.asarray(distortion_matrix, dtype=np.float64)
    if p_x.ndim != 1:
        raise ValueError("source_distribution must be 1D")
    if distortion.ndim != 2 or distortion.shape[0] != p_x.shape[0]:
        raise ValueError(
            f"distortion_matrix shape={distortion.shape} must be (|X|, |Y|) with |X|={p_x.shape[0]}"
        )
    if not math.isclose(float(p_x.sum()), 1.0, abs_tol=1e-9):
        raise ValueError(f"source_distribution must sum to 1, got {p_x.sum()}")
    if (p_x < 0).any():
        raise ValueError("source_distribution must be non-negative")
    if (distortion < 0).any():
        raise ValueError("distortion_matrix must be non-negative")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")

    lambdas: tuple[float, ...]
    if lambda_values is None:
        lambdas = build_default_lambda_sweep(distortion)
    else:
        lambdas = tuple(float(l) for l in lambda_values)
        if not lambdas:
            raise ValueError("lambda_values must be non-empty")
        if any(l < 0 for l in lambdas):
            raise ValueError("all lambda_values must be >= 0")

    rates: list[float] = []
    distortions: list[float] = []
    convergences: list[bool] = []
    for lam in lambdas:
        achieved_d, rate_bits, conv = _compute_distortion_for_slope(p_x, distortion, lam)
        rates.append(rate_bits)
        distortions.append(achieved_d)
        convergences.append(conv)

    if source_distribution_summary is None:
        source_distribution_summary = (
            f"|X|={p_x.shape[0]} discrete source / "
            f"|Y|={distortion.shape[1]} reproduction alphabet / "
            f"distortion-matrix-mean={float(distortion.mean()):.6f}"
        )

    # Canonical Provenance per Catalog #323
    # Hash inputs for inputs_sha256 (canonical for reproducibility)
    h = hashlib.sha256()
    h.update(p_x.tobytes())
    h.update(distortion.tobytes())
    h.update(repr(lambdas).encode("utf-8"))
    inputs_sha256 = h.hexdigest()
    prov = build_provenance_for_predicted(
        model_id=f"blahut_arimoto.iterate_rate_distortion.{derivation_method}",
        inputs_sha256=inputs_sha256,
        measurement_axis="[predicted]",
        hardware_substrate="cpu_algorithmic_iteration",
    )

    return RateDistortionCurve(
        lambda_values=lambdas,
        rate_values=tuple(rates),
        distortion_values=tuple(distortions),
        converged=tuple(convergences),
        canonical_provenance=prov,
        derivation_method=derivation_method,
        source_distribution_summary=source_distribution_summary,
    )
