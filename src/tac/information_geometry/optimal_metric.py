# SPDX-License-Identifier: MIT
"""The single reachable-decision-geometry metric ``g = ∇²F`` and its role reductions.

This is the *unification* module for operator task #500: one metric that is,
simultaneously, the **fidelity** predicate (surrogate admission at the argmax
boundary), the **training-loss** geometry (what the witness descends), and a
**curriculum-varying** metric (it changes with the softmax temperature ``τ``).

``F(θ) = logsumexp(θ)`` is the convex Bregman generator of the categorical
exponential family (softmax). Its Hessian is the categorical Fisher metric::

    g(p) = ∇²F = diag(p) − p pᵀ,   p = softmax(θ)

Every role below is a *reduction of this ONE object*, computed here (not asserted)
so a test can prove the reduction is bit-real:

* **fidelity** — the winner↔rival directional curvature
  ``C_wr = uᵀ g u`` with ``u = e_w − e_r`` reduces to
  ``C_wr = p_w + p_r − (p_w − p_r)²`` (the RIPO trust-region law
  ``|t| ≤ √(8·δ_KL / C_wr)``). This is a *primal / tangent-space* quadratic form
  — no ``H⁻¹`` solve — so it stays Fisher-natural in logit coordinates.
* **training-loss** — the two-class annulus specialization
  ``tr g|_{2-class} = 2 p (1−p) = ½ sech²(m/2)`` is a monotone function of the
  logit margin ``m``, which is *why* the measured curvature↔(−margin) Pearson is
  0.978 (the margin field is the Fisher surrogate the witness descends). This is a
  SCALAR-TRACE SURROGATE (a measured band calibration), not the full directional
  metric — see :func:`annulus_fisher_trace_surrogate` and the honesty note below.
* **curriculum-varying** — at temperature ``τ`` the metric in fixed-logit
  coordinates is ``g(τ) = (1/τ²)(diag(p_τ) − p_τ p_τᵀ)`` with
  ``p_τ = softmax(θ/τ)``. Both the ``1/τ²`` chain-rule prefactor AND the operating
  point ``p_τ`` (hence ``C_wr(τ)`` and ``p_w(τ)``) vary with ``τ`` — the metric
  concentrates onto the separatrix as ``τ ↓`` (the curriculum anneal facet).

**NO-FAKE honesty (the landed Bregman guard, `bregman_dual_metric_squared_hessian_v1`).**
Everything above uses the metric in its **primal / tangent (logit-displacement)**
quadratic form ``Δθᵀ g Δθ`` — that IS Fisher-natural, no inverse needed. The DUAL
raw-mean Euclidean *no-solve* length is ``Δθᵀ g² Δθ`` (the **squared** Hessian),
which is NOT the Fisher-natural cotangent length ``Δηᵀ g⁻¹ Δη`` (that one needs an
``H⁻¹`` solve). This module never conflates them; :func:`squared_metric_quadratic`
is provided only to make the distinction testable and cross-references the
canonical Bregman helper as the single source of that truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "MetricGeometryError",
    "annulus_fisher_trace_surrogate",
    "log_partition_hessian",
    "metric_directional_quadratic",
    "optimal_metric_unification_law",
    "softmax",
    "squared_metric_quadratic",
    "tempered_metric",
    "tempered_winner_rival_curvature",
    "winner_rival_curvature_via_metric",
]


class MetricGeometryError(ValueError):
    """Raised when a metric input violates the simplex / finiteness contract."""


def _as_prob_vector(p: Any) -> NDArray[np.float64]:
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 2:
        raise MetricGeometryError("p must be a 1-D categorical vector with K >= 2 entries")
    if not np.all(np.isfinite(arr)):
        raise MetricGeometryError("p must be finite")
    if np.any(arr < 0.0):
        raise MetricGeometryError("p must be non-negative (a probability vector)")
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise MetricGeometryError(f"p must sum to 1 (got {total!r}); pass a normalized simplex vector")
    return arr


def softmax(logits: Any, tau: float = 1.0) -> NDArray[np.float64]:
    """``softmax(logits / tau)`` with a numerically stable max-shift. ``tau > 0``."""

    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or z.shape[0] < 2:
        raise MetricGeometryError("logits must be a 1-D vector with K >= 2 entries")
    if not np.all(np.isfinite(z)):
        raise MetricGeometryError("logits must be finite")
    if not (isinstance(tau, (int, float)) and np.isfinite(tau) and tau > 0.0):
        raise MetricGeometryError("tau must be a finite positive temperature")
    scaled = z / float(tau)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def log_partition_hessian(p: Any) -> NDArray[np.float64]:
    """The metric ``g = ∇²F = diag(p) − p pᵀ`` (categorical Fisher information).

    ``F(θ) = logsumexp(θ)`` ⇒ ``∇F = p = softmax(θ)`` ⇒ ``∇²F = diag(p) − p pᵀ``.
    This is the ONE object every role below reduces from.
    """

    prob = _as_prob_vector(p)
    return np.diag(prob) - np.outer(prob, prob)


def metric_directional_quadratic(p: Any, direction: Any) -> float:
    """The primal / tangent Fisher-natural quadratic form ``uᵀ g u`` for logit
    displacement ``u = direction``. No ``H⁻¹`` solve (this IS the natural length in
    logit coordinates)."""

    g = log_partition_hessian(p)
    u = np.asarray(direction, dtype=np.float64)
    if u.shape != (g.shape[0],):
        raise MetricGeometryError("direction must have the same length K as p")
    if not np.all(np.isfinite(u)):
        raise MetricGeometryError("direction must be finite")
    return float(u @ g @ u)


def winner_rival_curvature_via_metric(p: Any) -> float:
    """The **fidelity** reduction: ``C_wr = (e_w − e_r)ᵀ g (e_w − e_r)`` computed
    directly from the metric ``g``. Equals ``p_w + p_r − (p_w − p_r)²`` — proven
    bit-equal to :func:`tac.optimization.ripo_fisher_trust_region.winner_rival_curvature`
    in the tests. This is the surrogate-admission trust-region curvature."""

    prob = _as_prob_vector(p)
    order = np.argsort(-prob, kind="stable")
    w, r = int(order[0]), int(order[1])
    u = np.zeros_like(prob)
    u[w] = 1.0
    u[r] = -1.0
    return metric_directional_quadratic(prob, u)


def annulus_fisher_trace_surrogate(margin: float) -> float:
    """The **training-loss** reduction (two-class annulus): ``tr g|_{2-class} =
    2 p (1−p) = ½ sech²(m/2)`` with ``p = σ(m)``.

    A monotone function of the logit margin ``m`` — this is WHY the measured
    curvature↔(−margin) Pearson is 0.978 (the margin field the witness descends is
    the Fisher surrogate). Delegates to the canonical deepmath identity so there is
    ONE source of this reduction. HONEST SCOPE: this is a scalar TRACE surrogate
    (two-class band), a measured 0.978 calibration to the full K=5 directional
    metric — NOT an exact global identity (see the module docstring)."""

    from tac.canonical_equations.deepmath_amortizing_argmax_laws_20260704 import (
        annulus_fisher_trace,
    )

    return float(annulus_fisher_trace(float(margin)))


def tempered_metric(logits: Any, tau: float) -> NDArray[np.float64]:
    """The **curriculum-varying** reduction: the metric in fixed-logit coordinates
    at temperature ``τ``::

        g(τ) = (1/τ²) (diag(p_τ) − p_τ p_τᵀ),   p_τ = softmax(θ/τ)

    The ``1/τ²`` prefactor is the chain-rule pullback of the natural-coordinate
    Fisher metric through ``η = θ/τ``; the operating point ``p_τ`` also varies with
    ``τ``. As ``τ ↓`` the metric concentrates onto the separatrix (interior mass
    ``p(1−p) ~ e^{−m/τ}`` decays faster than the ``1/τ²`` blow-up while the boundary
    ``p≈0.5`` grows like ``1/τ²``)."""

    if not (isinstance(tau, (int, float)) and np.isfinite(tau) and tau > 0.0):
        raise MetricGeometryError("tau must be a finite positive temperature")
    p_tau = softmax(logits, tau)
    g_natural = np.diag(p_tau) - np.outer(p_tau, p_tau)
    return g_natural / (float(tau) ** 2)


def tempered_winner_rival_curvature(logits: Any, tau: float) -> dict[str, float]:
    """The τ-dependence of the fidelity curvature: ``C_wr(τ)`` and ``p_w(τ)`` at the
    tempered operating point ``p_τ = softmax(θ/τ)``. Returns the natural-coordinate
    ``C_wr`` (the trust-region curvature the RIPO radius uses, at p_τ) plus the
    fixed-logit-coordinate ``C_wr / τ²`` so the τ-scaling is explicit."""

    p_tau = softmax(logits, tau)
    c_wr_natural = winner_rival_curvature_via_metric(p_tau)
    order = np.argsort(-p_tau, kind="stable")
    return {
        "tau": float(tau),
        "p_w": float(p_tau[int(order[0])]),
        "p_r": float(p_tau[int(order[1])]),
        "c_wr_natural": float(c_wr_natural),
        "c_wr_fixed_logit": float(c_wr_natural / (float(tau) ** 2)),
    }


def squared_metric_quadratic(p: Any, direction: Any) -> float:
    """The DUAL raw-mean *no-solve* Euclidean length ``uᵀ g² u`` (the **squared**
    Hessian). Provided ONLY to make the NO-FAKE distinction testable: this is NOT the
    Fisher-natural cotangent length ``ηᵀ g⁻¹ η`` (that needs an ``H⁻¹`` solve). Cross-
    references the canonical Bregman guard `bregman_dual_metric_squared_hessian_v1`
    as the single source of that truth."""

    from tac.information_geometry.bregman_v9_surfaces import squared_hessian_quadratic

    g = log_partition_hessian(p)
    return float(squared_hessian_quadratic(g, direction))


def optimal_metric_unification_law(
    logits: Any, tau: float = 1.0, direction: Any | None = None
) -> dict[str, Any]:
    """Evaluate all three role readings of the single metric ``g = ∇²F`` at one
    operating point ``(logits, τ)``. This is the canonical-equation callable.

    Returns a dict with the fidelity curvature (directional), the training-loss
    margin surrogate (annulus trace at the winner↔rival logit margin), and the
    curriculum τ-reading (tempered C_wr + operating point)."""

    p_tau = softmax(logits, tau)
    order = np.argsort(-p_tau, kind="stable")
    w, r = int(order[0]), int(order[1])
    z = np.asarray(logits, dtype=np.float64) / float(tau)
    logit_margin = float(z[w] - z[r])
    if direction is None:
        u = np.zeros_like(p_tau)
        u[w] = 1.0
        u[r] = -1.0
    else:
        u = np.asarray(direction, dtype=np.float64)
    return {
        "fidelity_directional_curvature": metric_directional_quadratic(p_tau, u),
        "training_loss_margin_surrogate_trace": annulus_fisher_trace_surrogate(logit_margin),
        "curriculum_tau_reading": tempered_winner_rival_curvature(logits, tau),
        "logit_margin": logit_margin,
    }
