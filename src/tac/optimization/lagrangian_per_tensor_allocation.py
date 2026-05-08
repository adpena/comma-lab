"""Canonical Lagrangian per-tensor allocator.

This module is the SINGLE canonical home for the per-tensor allocation
mechanism that historically lived (in three near-identical copies) in
``tools/pr101_omega_opt_joint_admm_allocation_empirical.py``,
``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py``, and
``tools/pr101_omega_opt_uniward_weighted_allocation.py``.

Naming
------

Per the Dykstra council finding (REVIEW-MATH 2026-05-08, commit ``f4f6270c``)
the technique is **Lagrangian per-tensor allocation** — λ-bisection over
INDEPENDENT per-tensor argmin problems — not the iterative primal-dual ADMM
with consensus that lives in :mod:`tac.joint_admm_coordinator`. The earlier
"Joint-ADMM" name is preserved only as a historical reference.

Mechanism
---------

For Lagrangian multiplier λ each tensor independently picks the codec choice
``c`` that minimizes::

    cost_t(c) = bytes_t(c) + λ · w_t · rel_err_t(c)²

where ``w_t`` defaults to 1.0 and may be set via the UNIWARD inverse-variance
hook (``UniwardWeightedAllocator``) to bias error away from
low-variance/smooth tensors. λ is then bisected to satisfy a global RMS
``rel_err`` target.

Public API
----------

* :class:`LagrangianPerTensorAllocator` — base mechanism. Pluggable encoder
  hook for joint-bytes evaluation (default: per-tensor sum).
* :class:`UniwardWeightedAllocator` — subclass with inverse-variance weights.

Both classes return a result ``dict`` with at minimum::

    {"lambda": float, "total_bytes": int, "rel_err": float, "selections": list}

and additional fields supplied by the joint-encoder hook (e.g. per-component
byte breakdown for ``encode_with_per_tensor_K``).

Score-relevance: the allocator's choice of codec/K per tensor changes archive
bytes; tag downstream evidence rows with the joint-encoder identity when
reporting numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


EPS_VARIANCE: float = 1e-12
"""Numerical floor for inverse-variance weight computation."""


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


def _select_min_cost(curve: Sequence[dict], lam: float, weight: float) -> dict:
    """Pick the row of ``curve`` minimizing ``bytes_or_proxy + λ · w · rel_err²``."""
    bytes_key = "bytes" if "bytes" in curve[0] else "byte_proxy"
    cost = [r[bytes_key] + lam * weight * r["rel_err"] ** 2 for r in curve]
    return curve[int(np.argmin(cost))]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """Outcome of a single Lagrangian allocation pass.

    Attributes:
        lam: chosen Lagrangian multiplier.
        total_bytes: sum of selected per-tensor bytes (or joint-encoder bytes
            when a joint encoder hook is supplied).
        rel_err: achieved aggregate ``rel_err`` (RMS of per-tensor rel_err if
            no joint encoder, else the joint encoder's reported value).
        selections: per-tensor selected curve row.
        per_tensor_rel_errs: per-tensor selected ``rel_err``.
        joint_extras: extra fields from a joint-encoder hook (empty dict if
            none).
    """
    lam: float
    total_bytes: int
    rel_err: float
    selections: list[dict]
    per_tensor_rel_errs: list[float]
    joint_extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (preserves historical key names)."""
        out: dict[str, Any] = {
            "lambda": self.lam,
            "total_bytes": int(self.total_bytes),
            "rel_err": float(self.rel_err),
            "rms_rel_err": float(self.rel_err),
            "selections": list(self.selections),
            "per_tensor_rel_errs": list(self.per_tensor_rel_errs),
        }
        out.update(self.joint_extras)
        return out


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


JointEncoderHook = Callable[[list[dict]], dict[str, Any]]
"""Optional joint-encoder hook signature.

Accepts the per-tensor selected curve rows; returns a dict that MUST include
``"total_bytes": int`` and ``"rel_err": float`` (joint achieved RMS). Any
extra keys are forwarded to :attr:`AllocationResult.joint_extras` and to the
final dict-form result.

Example: a wrapper around :func:`tools/pr101_lossy_coarsening_analytical.encode_with_per_tensor_K`
that maps the selected rows back to ``Ks`` and returns the joint brotli
result.
"""


class LagrangianPerTensorAllocator:
    """λ-bisection per-tensor allocator with optional joint encoder.

    Works over either sparsity curves (``{"alpha", "bytes", "rel_err"}``) or
    K-curves (``{"K", "byte_proxy", "rel_err"}``); the row's first ``bytes``
    or ``byte_proxy`` key is auto-detected.

    Args:
        weights: optional per-tensor non-negative weights (default ``1.0`` per
            tensor). Used in the cost function as
            ``cost = bytes + λ · w_t · rel_err²``.
        joint_encoder: optional joint-encoder hook (see :data:`JointEncoderHook`).
            When supplied, ``total_bytes`` and ``rel_err`` come from the hook
            instead of the per-tensor sum + RMS.
    """

    def __init__(
        self,
        *,
        weights: Sequence[float] | None = None,
        joint_encoder: JointEncoderHook | None = None,
    ) -> None:
        self._weights = list(weights) if weights is not None else None
        self._joint_encoder = joint_encoder

    # -- core ---------------------------------------------------------------

    def allocate(self, curves: Sequence[Sequence[dict]], lam: float) -> AllocationResult:
        """One-shot allocation at fixed ``λ``.

        Args:
            curves: per-tensor curves (list-of-lists of dict rows).
            lam: Lagrangian multiplier.

        Returns:
            :class:`AllocationResult`.
        """
        weights = self._resolve_weights(len(curves))
        selections: list[dict] = []
        rel_errs: list[float] = []
        for tensor_curve, w in zip(curves, weights, strict=True):
            chosen = _select_min_cost(tensor_curve, lam, w)
            selections.append(dict(chosen))
            rel_errs.append(float(chosen["rel_err"]))

        if self._joint_encoder is not None:
            joint = self._joint_encoder(selections)
            total_bytes = int(joint["total_bytes"])
            rel_err = float(joint["rel_err"])
            joint_extras = {k: v for k, v in joint.items() if k not in {"total_bytes", "rel_err"}}
        else:
            bytes_key = "bytes" if "bytes" in selections[0] else "byte_proxy"
            total_bytes = int(sum(int(s[bytes_key]) for s in selections))
            rel_err = float(np.sqrt(np.mean([e ** 2 for e in rel_errs]))) if rel_errs else 0.0
            joint_extras = {}

        return AllocationResult(
            lam=lam,
            total_bytes=total_bytes,
            rel_err=rel_err,
            selections=selections,
            per_tensor_rel_errs=rel_errs,
            joint_extras=joint_extras,
        )

    def bisect_for_rms_target(
        self,
        curves: Sequence[Sequence[dict]],
        rms_target: float,
        *,
        max_iter: int = 80,
        lam_lo: float = 0.0,
        lam_hi: float = 1e15,
        tol: float = 1e-12,
        memoize: bool = True,
    ) -> AllocationResult:
        """Bisect ``λ`` so the achieved aggregate ``rel_err <= rms_target``.

        Strategy: large λ → all tensors pick lossless (``rel_err=0``); small
        λ → most-aggressive codec. Bisection narrows the interval; the
        smallest λ satisfying the target is returned.

        Memoization: when ``memoize=True`` (default) repeated allocations
        with the same selection vector are cached. This matters for the
        K-curve substrate where many λ values map to identical K vectors;
        empirically gives ~3× wall-clock speedup over the K-grid (matches
        REVIEW-ENG S2 commit ``0c736176``).

        Args:
            curves: per-tensor curves.
            rms_target: target aggregate ``rel_err``.
            max_iter: bisection budget.
            lam_lo: initial lower bracket.
            lam_hi: initial upper bracket (a sentinel-large value triggers
                exponential ramping until the bracket has finite width).
            tol: bisection-stopping width tolerance.
            memoize: cache by selection vector identity.

        Returns:
            :class:`AllocationResult` corresponding to the chosen ``λ`` (the
            tightest feasible one found).
        """
        cache: dict[tuple[Any, ...], AllocationResult] = {}
        SENTINEL = lam_hi
        lo, hi = lam_lo, lam_hi
        best: AllocationResult | None = None

        def _alloc(lam: float) -> AllocationResult:
            if not memoize:
                return self.allocate(curves, lam)
            # We can only cache by selection vector AFTER one allocation;
            # do the allocate, then re-key if seen.
            res = self.allocate(curves, lam)
            key = tuple(_selection_key(s) for s in res.selections)
            cached = cache.get(key)
            if cached is not None:
                # Selection vector seen before — reuse its result (joint
                # encoder is a pure function of the selection vector).
                return cached
            cache[key] = res
            return res

        for _ in range(max_iter):
            if hi == SENTINEL:
                mid = lo * 10 + 1.0
            else:
                mid = 0.5 * (lo + hi)
            res = _alloc(mid)
            if res.rel_err <= rms_target:
                hi = mid
                best = res  # tightest feasible found so far
            else:
                lo = mid
            if hi != SENTINEL and abs(hi - lo) <= tol:
                break

        if best is None:
            # No feasible λ found within budget → return the last attempt at
            # the largest λ tried (closest to feasibility).
            best = _alloc(hi if hi != SENTINEL else lo)
        return best

    # -- helpers ------------------------------------------------------------

    def _resolve_weights(self, n: int) -> list[float]:
        if self._weights is None:
            return [1.0] * n
        if len(self._weights) != n:
            raise ValueError(
                f"weights length {len(self._weights)} does not match curves length {n}"
            )
        return list(self._weights)


# ---------------------------------------------------------------------------
# UNIWARD-weighted variant
# ---------------------------------------------------------------------------


def compute_local_variance_proxy(symbols_iter: Sequence[np.ndarray]) -> list[float]:
    """Return per-tensor variance of the int8 symbol distribution.

    This is the CPU analogue of UNIWARD's local-variance residual: a tensor
    whose symbols span a wide dynamic range is "textured" and absorbs error
    well; a tensor whose symbols cluster near zero is "smooth" and distortion
    there is detector-visible.

    Args:
        symbols_iter: sequence of 1-D int symbol vectors.

    Returns:
        Per-tensor variance.
    """
    out: list[float] = []
    for syms in symbols_iter:
        arr = syms.astype(np.float64)
        out.append(float(np.var(arr)) if arr.size > 0 else 0.0)
    return out


def compute_uniward_weights(variances: Sequence[float]) -> list[float]:
    """Inverse-variance UNIWARD weight: ``w(t) = 1 / (var(t) + ε)``.

    Higher variance → lower weight → more error budget.

    Args:
        variances: per-tensor variance.

    Returns:
        Per-tensor weight.
    """
    return [1.0 / (float(v) + EPS_VARIANCE) for v in variances]


class UniwardWeightedAllocator(LagrangianPerTensorAllocator):
    """Lagrangian allocator with UNIWARD inverse-variance weights.

    Args:
        symbols_iter: sequence of per-tensor int symbol vectors used to
            compute the inverse-variance weights.
        joint_encoder: see :class:`LagrangianPerTensorAllocator`.
    """

    def __init__(
        self,
        symbols_iter: Sequence[np.ndarray],
        *,
        joint_encoder: JointEncoderHook | None = None,
    ) -> None:
        variances = compute_local_variance_proxy(symbols_iter)
        weights = compute_uniward_weights(variances)
        super().__init__(weights=weights, joint_encoder=joint_encoder)
        self._variances = variances

    @property
    def variances(self) -> list[float]:
        """Per-tensor variances used to derive the UNIWARD weights."""
        return list(self._variances)


# ---------------------------------------------------------------------------
# Selection key (memoization)
# ---------------------------------------------------------------------------


def _selection_key(row: dict) -> tuple[Any, ...]:
    """Stable hashable key for a curve row (used for memoization)."""
    if "K" in row:
        return ("K", int(row["K"]))
    if "alpha" in row:
        return ("alpha", round(float(row["alpha"]), 9))
    return tuple(sorted(row.items()))


__all__ = [
    "EPS_VARIANCE",
    "AllocationResult",
    "JointEncoderHook",
    "LagrangianPerTensorAllocator",
    "UniwardWeightedAllocator",
    "compute_local_variance_proxy",
    "compute_uniward_weights",
]
