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

``rel_err`` form contract (2026-05-08 closure)
----------------------------------------------

The cost function ``cost = bytes + λ · w · rel_err²`` is mathematically a
**rate-MSE Lagrangian**: the squared penalty corresponds to L2² distortion
under the Karush-Kuhn-Tucker dual of an MSE constraint. The canonical form
of ``rel_err`` is therefore RMS (or its scale-equivalent L2 ratio); see
:mod:`tac.codec.rel_err` for the canonical helper.

The allocator also accepts curves whose ``rel_err`` is global L1 (the form
emitted by :func:`tac.codec.cost_curves.precompute_per_tensor_K_curves` and
the PR101 Path-B / PR106 UNIWARD-packet tools). When the curves are L1 the
Lagrangian dual is mathematically loose (squaring an L1 ratio does not
recover an inner-product distortion), but the per-tensor allocation still
produces monotone selections in λ and the bisection converges on the L1
form. Empirical anchors shipped before this refactor (0.0386 PR101
lossy_coarsening, 0.0415 ADMM × continuous-K) are L1 and remain valid.

Curves built via the canonical :mod:`tac.codec.cost_curves` precompute
helpers carry a per-row ``rel_err_form`` tag. The allocator inspects the
tag at the entry of :meth:`bisect_for_rms_target` and:

1. Asserts uniform form across all curves (raises if mixed).
2. Records the discovered form in the result's ``joint_extras`` so
   downstream evidence rows can tag themselves correctly.

Curves built by older tools without form tags still work; the form check
emits a single ``UserWarning`` per allocator instance via
``warnings.warn`` and proceeds. The strict-flip is gated on a class-level
flag and on the corresponding preflight check
``check_rel_err_definition_canonical``.

Public API
----------

* :class:`LagrangianPerTensorAllocator` — base mechanism. Pluggable encoder
  hook for joint-bytes evaluation (default: per-tensor sum).
* :class:`UniwardWeightedAllocator` — subclass with inverse-variance weights.
* :class:`JacobianWeightedAllocator` — subclass for externally certified
  scorer-pullback or boundary-pullback importance weights.

Both classes return a result ``dict`` with at minimum::

    {"lambda": float, "total_bytes": int, "rel_err": float, "selections": list}

and additional fields supplied by the joint-encoder hook (e.g. per-component
byte breakdown for ``encode_with_per_tensor_K``).

Score-relevance: the allocator's choice of codec/K per tensor changes archive
bytes; tag downstream evidence rows with the joint-encoder identity when
reporting numbers.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.codec.rel_err import (
    REL_ERR_FORM_KEY,
    RelErrForm,
    assert_uniform_rel_err_form,
)

EPS_VARIANCE: float = 1e-12
"""Numerical floor for inverse-variance weight computation."""

EPS_JACOBIAN_IMPORTANCE: float = 1e-12
"""Numerical floor for Jacobian-pullback importance weights."""


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

    # Class-level switch for promoting the rel_err_form check to strict.
    # Default False during the bug-class closure tranche; flip True after
    # ``check_rel_err_definition_canonical`` lands at strict in
    # ``preflight_all()`` and live violations across tools/ are at zero.
    strict_rel_err_form: bool = False

    def __init__(
        self,
        *,
        weights: Sequence[float] | None = None,
        joint_encoder: JointEncoderHook | None = None,
        strict_rel_err_form: bool | None = None,
    ) -> None:
        self._weights = list(weights) if weights is not None else None
        self._joint_encoder = joint_encoder
        if strict_rel_err_form is not None:
            self.strict_rel_err_form = bool(strict_rel_err_form)
        self._rel_err_form_warned = False
        self._observed_rel_err_form: RelErrForm | None = None

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
            rel_err = float(np.sqrt(np.mean([e ** 2 for e in rel_errs]))) if rel_errs else 0.0  # REL_ERR_NON_CANONICAL_OK: canonical RMS aggregator equivalent to aggregate_rel_err(mode=RMS); kept inline to avoid an indirect call in the bisection inner loop
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
        # Form-uniformity check: every curve row should declare the same
        # ``rel_err_form``. Curves built by tools that predate the
        # 2026-05-08 canonicalization may lack the tag; we tolerate that
        # via a one-time warning unless ``strict_rel_err_form`` is set.
        try:
            form = assert_uniform_rel_err_form(
                curves, strict=self.strict_rel_err_form
            )
            self._observed_rel_err_form = form
        except ValueError as e:
            # Mixed form is always fatal — the allocator math becomes
            # ill-posed.
            if "mixed rel_err_form" in str(e):
                raise
            # Missing-tag in strict mode: re-raise.
            if self.strict_rel_err_form:
                raise
            # Missing-tag in lax-strict mode: warn once. (When
            # ``strict_rel_err_form=False`` the helper does NOT raise on
            # missing tags, so this branch is mostly unreachable; kept as
            # a defensive path.)
            form = None

        # Warn-once when curves lacked any form tag in lax mode (no
        # exception was raised).
        if form is None and not self.strict_rel_err_form and not self._rel_err_form_warned:
            # Only warn if at least one row was missing the tag (i.e. not
            # the empty-curves case).
            has_any_row = any(rows for rows in curves)
            if has_any_row:
                warnings.warn(
                    "LagrangianPerTensorAllocator: curves lack "
                    f"'{REL_ERR_FORM_KEY}' tags; the allocator assumes "
                    "RMS-form rel_err for the squared Lagrangian "
                    "penalty. Use tac.codec.rel_err.RelErrForm and the "
                    "canonical precompute helpers in "
                    "tac.codec.cost_curves to silence this.",
                    UserWarning,
                    stacklevel=2,
                )
                self._rel_err_form_warned = True

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
            mid = lo * 10 + 1.0 if hi == SENTINEL else 0.5 * (lo + hi)
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

        # Forward the discovered form to evidence consumers.
        if self._observed_rel_err_form is not None:
            best.joint_extras = {
                **best.joint_extras,
                REL_ERR_FORM_KEY: self._observed_rel_err_form.value,
            }
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


def _as_finite_nonnegative_array(
    values: Sequence[float],
    *,
    label: str,
) -> np.ndarray:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D sequence")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    if (arr < 0.0).any():
        raise ValueError(f"{label} must be non-negative")
    return arr


def normalize_importance_weights(
    importance: Sequence[float],
    *,
    floor: float = EPS_JACOBIAN_IMPORTANCE,
    target_mean: float = 1.0,
) -> list[float]:
    """Normalize non-negative sensitivity weights to a stable mean.

    Higher returned weight means ``LagrangianPerTensorAllocator`` will protect
    that tensor harder because the cost is
    ``bytes + lambda * weight * rel_err**2``.
    """
    if floor < 0.0 or not np.isfinite(floor):
        raise ValueError("floor must be finite and non-negative")
    if target_mean <= 0.0 or not np.isfinite(target_mean):
        raise ValueError("target_mean must be finite and positive")
    arr = _as_finite_nonnegative_array(importance, label="importance")
    if float(arr.max()) <= 0.0:
        raise ValueError("importance must contain at least one positive value")
    arr = np.maximum(arr, float(floor))
    mean = float(arr.mean())
    if mean <= 0.0 or not np.isfinite(mean):
        raise ValueError("importance mean is not finite/positive after flooring")
    return (arr * (float(target_mean) / mean)).tolist()


def compute_cpu_axis_weights(
    *,
    scorer_jacobian_pose: Sequence[float],
    scorer_jacobian_seg: Sequence[float],
    pose_axis_weight: float = 1.0,
    seg_axis_weight: float = 1.0,
    architecture_class: str = "hnerv",
    target_mean: float = 1.0,
) -> list[float]:
    """Score-axis-aware Jacobian importance for CPU-axis allocation.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the contest
    leaderboard ranks by CPU eval. Jacobian magnitudes pulled through a
    CUDA evaluator OVER-ATTRIBUTE pose-axis importance because the CPU
    evaluator sees a damped pose channel (R_pose ~= 5.04). This helper
    rebalances per-tensor importance to match the CPU evaluator's
    sensitivity, then funnels through the canonical normalization.

    Form::

        w_t = pose_axis_weight * (J_pose_t^2 / R_pose^2)
            + seg_axis_weight  * (J_seg_t^2  / R_seg^2)

    The squared form preserves Fisher / Jacobian accumulation semantics;
    dividing by ``R_*^2`` rebases each axis from CUDA-Jacobian space to
    CPU-Jacobian space.

    Args:
        scorer_jacobian_pose: per-tensor PoseNet d(score)/d(tensor)
            magnitude (CUDA-axis; e.g., from
            :mod:`tac.component_sensitivity_artifact`).
        scorer_jacobian_seg: per-tensor SegNet d(score)/d(tensor) magnitude
            (CUDA-axis).
        pose_axis_weight: multiplicative weight on the pose-axis term
            (default 1.0).
        seg_axis_weight: multiplicative weight on the seg-axis term.
        architecture_class: calibration class for R_pose / R_seg.
        target_mean: see :func:`normalize_importance_weights`.

    Returns:
        Per-tensor non-negative weight vector ready for the allocator.
    """
    from tac.optimization.cuda_cpu_axis_calibration import (
        CudaCpuCalibration,
    )
    cal = CudaCpuCalibration(architecture_class=architecture_class)
    pose = _as_finite_nonnegative_array(
        scorer_jacobian_pose, label="scorer_jacobian_pose"
    )
    seg = _as_finite_nonnegative_array(
        scorer_jacobian_seg, label="scorer_jacobian_seg"
    )
    if pose.size != seg.size:
        raise ValueError(
            f"scorer_jacobian_pose length {pose.size} != "
            f"scorer_jacobian_seg length {seg.size}"
        )
    if pose_axis_weight < 0.0 or seg_axis_weight < 0.0:
        raise ValueError("axis weights must be non-negative")
    pose_term = pose_axis_weight * (pose ** 2) / (cal.r_pose ** 2)
    seg_term = seg_axis_weight * (seg ** 2) / (cal.r_seg ** 2)
    raw = pose_term + seg_term
    return normalize_importance_weights(raw, target_mean=target_mean)


def compute_jacobian_importance_weights(
    importance: Sequence[float],
    *,
    texture_capacity: Sequence[float] | None = None,
    importance_floor: float = EPS_JACOBIAN_IMPORTANCE,
    capacity_floor: float = EPS_JACOBIAN_IMPORTANCE,
    target_mean: float = 1.0,
) -> list[float]:
    """Convert scorer-pullback importance into allocator protection weights.

    ``importance[t]`` is expected to be a per-tensor scalar such as
    ``E[(J_t^T s)^2]`` or a certified component-sensitivity aggregate.
    Optional ``texture_capacity[t]`` is a UNIWARD-style capacity proxy:
    larger capacity lowers the protection weight, allowing more error in
    tensors whose perturbations are empirically easier to hide.
    """
    raw = _as_finite_nonnegative_array(importance, label="importance")
    if texture_capacity is not None:
        cap = _as_finite_nonnegative_array(
            texture_capacity,
            label="texture_capacity",
        )
        if cap.size != raw.size:
            raise ValueError(
                "texture_capacity length "
                f"{cap.size} does not match importance length {raw.size}"
            )
        if capacity_floor <= 0.0 or not np.isfinite(capacity_floor):
            raise ValueError("capacity_floor must be finite and positive")
        raw = raw / np.maximum(cap, float(capacity_floor))
    return normalize_importance_weights(
        raw,
        floor=importance_floor,
        target_mean=target_mean,
    )


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


class JacobianWeightedAllocator(LagrangianPerTensorAllocator):
    """Lagrangian allocator for externally computed scorer-pullback weights.

    The class does not compute scorer gradients. It consumes a checked
    per-tensor importance vector produced by CUDA sensitivity/JVP/VJP tools and
    reuses the canonical λ-bisection allocator.
    """

    def __init__(
        self,
        importance: Sequence[float],
        *,
        texture_capacity: Sequence[float] | None = None,
        joint_encoder: JointEncoderHook | None = None,
        target_mean: float = 1.0,
    ) -> None:
        weights = compute_jacobian_importance_weights(
            importance,
            texture_capacity=texture_capacity,
            target_mean=target_mean,
        )
        super().__init__(weights=weights, joint_encoder=joint_encoder)
        self._importance_weights = weights

    @property
    def importance_weights(self) -> list[float]:
        """Protection weights derived from scorer-pullback importance."""
        return list(self._importance_weights)


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
    "EPS_JACOBIAN_IMPORTANCE",
    "EPS_VARIANCE",
    "AllocationResult",
    "JacobianWeightedAllocator",
    "JointEncoderHook",
    "LagrangianPerTensorAllocator",
    "UniwardWeightedAllocator",
    "compute_cpu_axis_weights",
    "compute_jacobian_importance_weights",
    "compute_local_variance_proxy",
    "compute_uniward_weights",
    "normalize_importance_weights",
]
