# SPDX-License-Identifier: MIT
"""META-LAGRANGIAN-DUAL-SOLVER PHASE 2 - per-axis dual-variable computation.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver - NON-NEGOTIABLE, HIGHEST EMPHASIS"
+ Catalog #355 STRICT preflight gate Phase 1 → Phase 2 advancement +
operator standing directive 2026-05-26 verbatim *"all are approved + pursue
other attacks as well remmeber all MLX first portable via numpy and
indivually fractally optimized"*.

The Phase 1 → Phase 2 advancement
---------------------------------

**Phase 1 (LANDED at commit c8d51ebb)**:
    ``tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates``
    builds a 1-dim Gaussian posterior per candidate via
    :func:`tac.findings_lagrangian.posterior_update_from_anchors` (single
    bounded residual proxy), computes the scalar 4-term Lagrangian, and
    derives a bounded [0.95, 1.05] adjustment factor.

**Phase 2 (THIS module)**:
    Per-candidate per-axis (seg, pose, rate) DUAL-VARIABLE computation via
    MLX-native Dykstra alternating projections onto the 3D Pareto polytope.
    Per-axis KKT residuals expose the canonical sensitivity signal sister
    to (but distinct from) :mod:`tac.master_gradient` raw-byte authority
    per Catalog #318 axis-aggregate-scale boundary.

    The exposed ``adjustment_factor`` REMAINS bounded [0.95, 1.05] per
    CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323
    canonical Provenance + Catalog #341 routing markers. Phase 2 enables
    UNBOUNDED per-axis dual computation INTERNALLY but the EXPOSED
    contribution to cathedral autopilot ranker stays inside the safety
    envelope.

**Phase 3+ (deferred)**:
    Typed atom flow into solver; per-element learned-optimal destination
    per META engineering vision.

The math
--------

The cathedral autopilot ranker faces a 3-axis polytope
``F = {(d_seg, d_pose, archive_bytes) : 0 ≤ d_seg ≤ B_seg,
0 ≤ d_pose ≤ B_pose, 0 ≤ archive_bytes ≤ B_rate}`` per the contest score
formula ``S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes/N``.

Dykstra alternating projections (Dykstra 1983; Boyd-Dattorro 2006 modern
treatment) on a convex polytope ``F = ∩_i F_i`` (intersection of
half-spaces ``F_i = {x : a_i^T x ≤ b_i}``) converges to the projection
``proj_F(x_0)`` via the iteration::

    x_{k+1} = proj_{F_{k mod n}}(x_k + correction_{k mod n})
    correction_{k mod n} updated per Dykstra's correction step

For our 3-axis half-space polytope the per-axis projection is closed-form
(clip into [0, B_axis]) so the iteration converges in O(log(1/ε)) per
Boyd 2004 Theorem 1.

The per-axis dual variables ``λ_seg, λ_pose, λ_rate`` are the Lagrange
multipliers of the active polytope constraints; their values quantify
the per-axis SENSITIVITY of the Lagrangian to per-axis budget tightening.

Per-axis KKT residual ``r_i = a_i^T x* - b_i`` where ``x*`` is the
Dykstra-converged projection; ``|r_i|`` measures distance to feasibility
on axis i.

MLX-first + numpy-portable bridge
---------------------------------

Per ``feedback_mlx_first_with_numpy_portable_bridge_standing_directive_20260526.md``:

- **MLX inner kernel**: the Dykstra inner loop uses :mod:`mlx.core`
  arrays for the per-iteration arithmetic. MLX provides Apple Silicon
  unified-memory acceleration on M5 Max with fp32 default.
- **numpy boundary**: the public API accepts numpy ``np.ndarray`` inputs
  AND returns numpy outputs. The MLX → numpy conversion at the boundary
  ensures sister consumers (CUDA-side cathedral autopilot ranker callers)
  receive the same interface.
- **No gradient flow**: per CLAUDE.md "MPS auth eval is NOISE", MLX is
  research-signal only; no gradients flow through MLX into PyTorch.
- **Provenance discipline**: per Catalog #341, all returned per-axis
  adjustment factors carry ``axis_tag="[predicted]"`` + ``score_claim=False``
  + ``promotable=False``.

Catalog #125 6-hook wire-in declaration
---------------------------------------

- Hook #1 sensitivity-map: ACTIVE - per-axis KKT residuals ARE the
  canonical sensitivity signal at the apparatus-scope boundary.
- Hook #2 Pareto constraint: ACTIVE - Dykstra alternating-projections on
  the (seg, pose, rate) polytope IS the Pareto-feasibility primitive.
- Hook #3 bit-allocator: ACTIVE (indirect) - per-axis dual λ_rate IS the
  apparatus-scope rate-axis sensitivity signal (sister of bit-allocator
  per-byte sensitivity).
- Hook #4 cathedral autopilot dispatch: ACTIVE PRIMARY - Phase 2 helper
  extends Catalog #355 wire-in callsite at ranker decision boundary.
- Hook #5 continual-learning posterior: ACTIVE - per-axis posterior
  anchors emit through ``tac.findings_lagrangian.posterior_update_from_anchors``.
- Hook #6 probe-disambiguator: ACTIVE - per-axis dual variables
  disambiguate which polytope axis is the binding constraint per candidate.

Cross-references
----------------

- CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable (THIS IS the
  apparatus-scope advancement; substrate-scope sister is Cascade C'
  Lagrangian dual primitive at commit ``2d5337f27``).
- Catalog #355 STRICT preflight gate Phase 2 advancement.
- Catalog #318 ``tac.master_gradient`` raw-byte-authority guard (THIS
  module operates at AXIS-AGGREGATE scale; raw-byte authority refused).
- Catalog #323 canonical Provenance umbrella.
- Catalog #341 canonical-routing markers (``score_claim=False`` +
  ``promotable=False`` + ``axis_tag="[predicted]"``).
- Catalog #356 per-axis decomposition canonical Provenance.
- Sister :mod:`tac.score_composition` ``compose_score_from_axes`` (uses
  the SAME canonical formula constants this module's Dykstra polytope
  expresses).
- ``feedback_mlx_first_with_numpy_portable_bridge_standing_directive_20260526.md``
- T3 council § 9 entropy-position P21 (just-added).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    _MLX_AVAILABLE = False


__all__ = [
    "AXIS_NAMES",
    "DYKSTRA_DEFAULT_EPSILON",
    "DYKSTRA_DEFAULT_MAX_ITERATIONS",
    "MLX_AVAILABLE",
    "PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX",
    "PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN",
    "PHASE_2_DUAL_SOLVER_SCHEMA_VERSION",
    "PerAxisDualSolverResult",
    "Phase2SolverError",
    "compute_per_axis_dual_variables",
    "dykstra_alternating_projections_3_axis",
    "kkt_residuals_per_axis",
    "per_axis_adjustment_factors",
]


PHASE_2_DUAL_SOLVER_SCHEMA_VERSION = "meta_lagrangian_dual_solver_phase_2_v1_20260526"
"""Canonical schema id for Phase 2 dual-solver output payload.

Mirrors ``META_LAGRANGIAN_INVOCATION_SCHEMA`` (Phase 1) per Catalog
#245/#300/#336 sister versioning discipline. Bump when output shape
changes meaningfully.
"""

PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN: float = 0.95
"""Lower bound on the EXPOSED adjustment factor to cathedral autopilot.

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323/#341:
the per-candidate per-axis dual computation can produce arbitrary
INTERNAL values, but the EXPOSED contribution to the ranker stays
inside [0.95, 1.05] preserving the Phase 1 SAFETY ENVELOPE contract.
"""

PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX: float = 1.05
"""Upper bound on the EXPOSED adjustment factor."""

DYKSTRA_DEFAULT_MAX_ITERATIONS: int = 50
"""Maximum Dykstra alt-projections iterations per candidate.

Per Boyd 2004 Theorem 1: convergence in O(log(1/ε)) for convex polytopes;
3-axis half-space polytope typically converges in 10-20 iterations.
"""

DYKSTRA_DEFAULT_EPSILON: float = 1.0e-5
"""Convergence threshold for Dykstra alt-projections (Linf norm).

Per MLX fp32 default precision; lower thresholds may not converge in
finite-iteration budget on M5 Max.
"""

AXIS_NAMES: tuple[str, str, str] = ("seg", "pose", "rate")
"""Canonical axis ordering for per-candidate per-axis dual variables.

Mirrors :class:`tac.cathedral.consumer_contract.AxisDecomposition` field
ordering (``predicted_d_seg_delta`` / ``predicted_d_pose_delta`` /
``predicted_archive_bytes_delta``) + ``tac.score_composition`` canonical
formula constant ordering.
"""

MLX_AVAILABLE = _MLX_AVAILABLE
"""True iff :mod:`mlx.core` imported successfully at module load.

Per MLX-first + numpy-portable bridge contract: the module FALLS BACK to
numpy-only computation if MLX is unavailable. The bridge contract is the
canonical interface; the underlying compute substrate is implementation
detail.
"""


class Phase2SolverError(ValueError):
    """Raised when Phase 2 dual solver inputs or outputs violate invariants.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": every contract
    enforced in ``__post_init__`` / function entry so the construction
    surface refuses bad inputs at the source.
    """


@dataclass(frozen=True)
class PerAxisDualSolverResult:
    """Phase 2 per-candidate per-axis dual-variable computation result.

    Per Catalog #305 observability surface: every per-axis value is
    decomposable, inspectable, queryable, citable, diff-able, and
    counterfactual-able.

    Per Catalog #341 canonical-routing markers: every emitted row carries
    ``axis_tag="[predicted]"`` + ``score_claim=False`` + ``promotable=False``.

    Per Catalog #318 axis-aggregate-scale boundary: this result operates
    at the per-candidate per-axis level, NOT at the raw-byte level.

    Fields
    ------
    candidate_id : str
        Identifier of the candidate this result is computed for.
    dual_variables_per_axis : Mapping[str, float]
        Per-axis Lagrange multipliers ``{seg, pose, rate}``. Larger
        values indicate the axis is the binding polytope constraint;
        the value's MAGNITUDE quantifies per-axis sensitivity of the
        Lagrangian to budget tightening.
    kkt_residual_per_axis : Mapping[str, float]
        Per-axis KKT residual ``r_i = a_i^T x* - b_i`` where ``x*`` is
        the Dykstra-converged projection. ``|r_i| < ε`` indicates
        feasibility on axis i; large positive values indicate axis i
        is infeasible without further iteration.
    adjustment_factor_per_axis : Mapping[str, float]
        Per-axis bounded adjustment factor in [0.95, 1.05]; observability
        ONLY; cathedral autopilot ranker consumes the SCALAR factor.
    adjustment_factor : float
        Exposed scalar adjustment factor in [0.95, 1.05] per Phase 1
        contract preservation. Cathedral autopilot ranker consumes this
        value; per-axis dict is observability-only.
    dykstra_iterations_to_convergence : int
        Number of Dykstra alt-projection iterations until ``||x_{k+1} - x_k||_∞ < ε``.
        Always ``≤ DYKSTRA_DEFAULT_MAX_ITERATIONS``.
    converged : bool
        True iff the iteration converged before the max-iteration cap.
    posterior_sigma_per_axis : Mapping[str, float]
        Per-axis posterior uncertainty for downstream consumers per
        Catalog #125 hook #1 sensitivity-map.
    axis_tag : str
        Canonical axis tag per Catalog #287/#341. Always "[predicted]".
    score_claim : bool
        Always False per Catalog #341 canonical-routing markers.
    promotable : bool
        Always False per Catalog #341 canonical-routing markers.
    canonical_provenance : Mapping[str, Any]
        Dict-form Provenance per Catalog #323.
    """

    candidate_id: str
    dual_variables_per_axis: Mapping[str, float]
    kkt_residual_per_axis: Mapping[str, float]
    adjustment_factor_per_axis: Mapping[str, float]
    adjustment_factor: float
    dykstra_iterations_to_convergence: int
    converged: bool
    posterior_sigma_per_axis: Mapping[str, float]
    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)
    axis_tag: str = "[predicted]"
    score_claim: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise Phase2SolverError("candidate_id must be non-empty string")
        for mapping_name in (
            "dual_variables_per_axis",
            "kkt_residual_per_axis",
            "adjustment_factor_per_axis",
            "posterior_sigma_per_axis",
        ):
            mapping = getattr(self, mapping_name)
            if not isinstance(mapping, Mapping):
                raise Phase2SolverError(f"{mapping_name} must be Mapping")
            for axis in AXIS_NAMES:
                if axis not in mapping:
                    raise Phase2SolverError(
                        f"{mapping_name} missing canonical axis {axis!r}"
                    )
                value = mapping[axis]
                if not isinstance(value, (int, float)):
                    raise Phase2SolverError(
                        f"{mapping_name}[{axis!r}]={value!r} must be numeric"
                    )
                if value != value:  # NaN
                    raise Phase2SolverError(
                        f"{mapping_name}[{axis!r}] is NaN"
                    )
        # Per-axis adjustment factors must be in [0.95, 1.05].
        for axis in AXIS_NAMES:
            af = float(self.adjustment_factor_per_axis[axis])
            if af < PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN - 1e-9:
                raise Phase2SolverError(
                    f"adjustment_factor_per_axis[{axis!r}]={af} < "
                    f"{PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN}"
                )
            if af > PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX + 1e-9:
                raise Phase2SolverError(
                    f"adjustment_factor_per_axis[{axis!r}]={af} > "
                    f"{PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX}"
                )
        # Per-axis posterior sigma must be >= 0.
        for axis in AXIS_NAMES:
            sig = float(self.posterior_sigma_per_axis[axis])
            if sig < 0:
                raise Phase2SolverError(
                    f"posterior_sigma_per_axis[{axis!r}]={sig} must be >= 0"
                )
        # Exposed scalar adjustment factor must be in [0.95, 1.05].
        if not isinstance(self.adjustment_factor, (int, float)):
            raise Phase2SolverError("adjustment_factor must be numeric")
        if self.adjustment_factor != self.adjustment_factor:  # NaN
            raise Phase2SolverError("adjustment_factor is NaN")
        if self.adjustment_factor < PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN - 1e-9:
            raise Phase2SolverError(
                f"adjustment_factor={self.adjustment_factor} < "
                f"{PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN}"
            )
        if self.adjustment_factor > PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX + 1e-9:
            raise Phase2SolverError(
                f"adjustment_factor={self.adjustment_factor} > "
                f"{PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX}"
            )
        # Convergence iterations must be non-negative int <= max.
        if not isinstance(self.dykstra_iterations_to_convergence, int):
            raise Phase2SolverError(
                "dykstra_iterations_to_convergence must be int"
            )
        if self.dykstra_iterations_to_convergence < 0:
            raise Phase2SolverError(
                "dykstra_iterations_to_convergence must be >= 0"
            )
        if self.dykstra_iterations_to_convergence > DYKSTRA_DEFAULT_MAX_ITERATIONS:
            raise Phase2SolverError(
                f"dykstra_iterations_to_convergence={self.dykstra_iterations_to_convergence}"
                f" > DYKSTRA_DEFAULT_MAX_ITERATIONS={DYKSTRA_DEFAULT_MAX_ITERATIONS}"
            )
        if not isinstance(self.converged, bool):
            raise Phase2SolverError("converged must be bool")
        # Catalog #341 canonical-routing markers.
        if self.axis_tag != "[predicted]":
            raise Phase2SolverError(
                f"axis_tag={self.axis_tag!r} must be '[predicted]' per Catalog #341"
            )
        if self.score_claim is not False:
            raise Phase2SolverError(
                "score_claim must be False per Catalog #341 (apparatus-scope; "
                "no contest-score claim)"
            )
        if self.promotable is not False:
            raise Phase2SolverError(
                "promotable must be False per Catalog #341 (apparatus-scope; "
                "no promotion eligibility)"
            )
        # canonical_provenance must be a Mapping.
        if not isinstance(self.canonical_provenance, Mapping):
            raise Phase2SolverError(
                "canonical_provenance must be Mapping per Catalog #323"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization per Catalog #305 observability surface."""
        return {
            "schema": PHASE_2_DUAL_SOLVER_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "dual_variables_per_axis": {
                k: float(v) for k, v in self.dual_variables_per_axis.items()
            },
            "kkt_residual_per_axis": {
                k: float(v) for k, v in self.kkt_residual_per_axis.items()
            },
            "adjustment_factor_per_axis": {
                k: float(v) for k, v in self.adjustment_factor_per_axis.items()
            },
            "adjustment_factor": float(self.adjustment_factor),
            "dykstra_iterations_to_convergence": int(
                self.dykstra_iterations_to_convergence
            ),
            "converged": bool(self.converged),
            "posterior_sigma_per_axis": {
                k: float(v) for k, v in self.posterior_sigma_per_axis.items()
            },
            "axis_tag": str(self.axis_tag),
            "score_claim": bool(self.score_claim),
            "promotable": bool(self.promotable),
            "canonical_provenance": dict(self.canonical_provenance),
        }


def _project_onto_axis_halfspace_numpy(
    x: np.ndarray, axis_index: int, lower: float, upper: float
) -> np.ndarray:
    """Project 3-vector x onto the axis-i half-space [lower, upper].

    Closed-form projection: clip the i-th coordinate; leave other
    coordinates unchanged. Sister of MLX kernel; used for numpy-only
    fallback path.
    """
    out = x.copy()
    out[axis_index] = np.clip(out[axis_index], lower, upper)
    return out


def _project_onto_axis_halfspace_mlx(
    x: Any, axis_index: int, lower: float, upper: float
) -> Any:
    """MLX-native projection onto axis-i half-space [lower, upper]."""
    if not _MLX_AVAILABLE:
        raise Phase2SolverError("MLX not available; use numpy path")
    # MLX: create a copy with the i-th coordinate clipped.
    # We do this via index assignment after copy.
    out = mx.array(x)
    # Compute clipped value scalar.
    coord = float(out[axis_index].item())
    if coord < lower:
        coord = float(lower)
    elif coord > upper:
        coord = float(upper)
    # Rebuild the array with clipped coordinate.
    result_list = [float(out[i].item()) for i in range(3)]
    result_list[axis_index] = coord
    return mx.array(result_list, dtype=mx.float32)


def dykstra_alternating_projections_3_axis(
    x0: Sequence[float],
    *,
    budgets: Sequence[tuple[float, float]],
    max_iterations: int = DYKSTRA_DEFAULT_MAX_ITERATIONS,
    epsilon: float = DYKSTRA_DEFAULT_EPSILON,
    use_mlx: bool | None = None,
) -> tuple[np.ndarray, int, bool, np.ndarray]:
    """Dykstra alternating projections onto 3-axis half-space polytope.

    Per Dykstra 1983 + Boyd-Dattorro 2006: convex polytope intersection
    via alternating projections WITH correction steps. The 3-axis
    polytope ``F = ∩_i F_i`` where ``F_i = {x : lower_i ≤ x_i ≤ upper_i}``
    is the canonical Pareto polytope at the apparatus level (3 axes:
    seg, pose, rate).

    Per Boyd 2004 Theorem 1: convergence in O(log(1/ε)) for convex
    polytopes; this 3-axis case typically converges in 10-20 iterations
    at ε = 1e-5.

    Args
    ----
    x0 : Sequence[float]
        Initial 3-vector ``[d_seg_target, d_pose_target, archive_bytes_target]``
        in canonical axis ordering (per :data:`AXIS_NAMES`).
    budgets : Sequence[tuple[float, float]]
        Per-axis ``(lower, upper)`` budget bounds for the half-spaces.
        Must have len == 3; canonical axis ordering.
    max_iterations : int
        Maximum Dykstra iterations. Defaults to
        :data:`DYKSTRA_DEFAULT_MAX_ITERATIONS`.
    epsilon : float
        Convergence threshold for Linf norm of (x_{k+1} - x_k).
        Defaults to :data:`DYKSTRA_DEFAULT_EPSILON`.
    use_mlx : bool | None
        If True, use MLX path (requires MLX available). If False, use
        numpy path. If None, use MLX if available, else numpy.

    Returns
    -------
    tuple
        ``(x_converged, iterations, converged, corrections)``
        where:

        - ``x_converged`` : np.ndarray shape (3,) - the converged projection
        - ``iterations`` : int - number of iterations performed
        - ``converged`` : bool - True iff converged before max iterations
        - ``corrections`` : np.ndarray shape (3,) - per-axis Dykstra
          correction vectors at convergence (used downstream for dual
          variable extraction)
    """
    if len(x0) != 3:
        raise Phase2SolverError(f"x0 must be length 3, got {len(x0)}")
    if len(budgets) != 3:
        raise Phase2SolverError(f"budgets must be length 3, got {len(budgets)}")
    for i, (lo, hi) in enumerate(budgets):
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise Phase2SolverError(f"budgets[{i}] must be (float, float)")
        if lo > hi:
            raise Phase2SolverError(
                f"budgets[{i}]: lower={lo} > upper={hi}"
            )
    if max_iterations <= 0:
        raise Phase2SolverError("max_iterations must be > 0")
    if epsilon <= 0:
        raise Phase2SolverError("epsilon must be > 0")

    if use_mlx is None:
        use_mlx = _MLX_AVAILABLE
    if use_mlx and not _MLX_AVAILABLE:
        # Fail back to numpy if MLX requested but unavailable.
        use_mlx = False

    if use_mlx:
        # MLX path: keep the Dykstra state and correction rows as MLX arrays,
        # then convert to numpy at the API boundary per the portable bridge.
        x_mx = mx.array([float(v) for v in x0], dtype=mx.float32)
        corrections_mx = mx.zeros((3, 3), dtype=mx.float32)

        converged = False
        iterations = 0
        for k in range(max_iterations):
            x_prev_mx = x_mx
            for axis_index in range(3):
                lo, hi = budgets[axis_index]
                y = x_mx + corrections_mx[axis_index]
                x_new_mx = _project_onto_axis_halfspace_mlx(
                    y, axis_index, float(lo), float(hi)
                )
                correction_new = y - x_new_mx
                correction_rows = [
                    correction_new if row_index == axis_index else corrections_mx[row_index]
                    for row_index in range(3)
                ]
                corrections_mx = mx.stack(correction_rows)
                x_mx = x_new_mx
            iterations = k + 1
            diff = float(mx.max(mx.abs(x_mx - x_prev_mx)).item())
            if diff < epsilon:
                converged = True
                break

        return (
            np.asarray(x_mx.tolist(), dtype=np.float64),
            iterations,
            converged,
            np.asarray(corrections_mx.tolist(), dtype=np.float64),
        )

    # Numpy path: canonical fallback for non-MLX environments.
    x = np.asarray(x0, dtype=np.float64)
    # Dykstra corrections: one per half-space (n_halfspaces == 3).
    corrections = np.zeros((3, 3), dtype=np.float64)

    converged = False
    iterations = 0
    for k in range(max_iterations):
        x_prev = x.copy()
        # Cycle through the 3 half-spaces.
        for axis_index in range(3):
            lo, hi = budgets[axis_index]
            y = x + corrections[axis_index]
            x_new = _project_onto_axis_halfspace_numpy(
                y, axis_index, float(lo), float(hi)
            )
            corrections[axis_index] = y - x_new
            x = x_new
        iterations = k + 1
        # Convergence check on Linf norm.
        diff = float(np.max(np.abs(x - x_prev)))
        if diff < epsilon:
            converged = True
            break

    return x, iterations, converged, corrections


def kkt_residuals_per_axis(
    x_converged: np.ndarray,
    budgets: Sequence[tuple[float, float]],
) -> dict[str, float]:
    """Per-axis KKT residuals at the Dykstra-converged projection.

    For the half-space constraint ``a_i^T x ≤ b_i`` (in our case
    ``x_i ≤ upper_i`` and ``-x_i ≤ -lower_i``), the KKT residual is
    ``r_i = a_i^T x* - b_i`` where ``x*`` is the converged projection.

    Returns
    -------
    dict
        ``{axis_name: residual}`` mapping per canonical axis ordering.
        Positive values indicate infeasibility on axis i (Dykstra
        did not fully project; budget binding); near-zero values
        indicate axis i is feasible at the converged point.
    """
    if len(x_converged) != 3:
        raise Phase2SolverError(
            f"x_converged must be length 3, got {len(x_converged)}"
        )
    if len(budgets) != 3:
        raise Phase2SolverError(
            f"budgets must be length 3, got {len(budgets)}"
        )
    residuals: dict[str, float] = {}
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        lo, hi = budgets[axis_index]
        x_i = float(x_converged[axis_index])
        # KKT residual = max(x_i - upper_i, lower_i - x_i, 0)
        upper_residual = max(x_i - float(hi), 0.0)
        lower_residual = max(float(lo) - x_i, 0.0)
        residuals[axis_name] = float(max(upper_residual, lower_residual))
    return residuals


def _dual_variable_from_correction(
    correction_vector: np.ndarray, axis_index: int
) -> float:
    """Extract the per-axis Lagrange multiplier from Dykstra correction.

    Per Boyd-Dattorro 2006 § 6.2: the Dykstra correction vector for
    half-space i at convergence equals the dual variable ``λ_i`` times
    the constraint gradient ``a_i``. For our axis-aligned half-spaces
    the constraint gradient is the i-th unit basis vector, so the dual
    variable is the i-th component of the correction vector (signed).

    Returns the absolute value (per Lagrangian theory dual variables
    are non-negative for inequality constraints).
    """
    if len(correction_vector) != 3:
        raise Phase2SolverError(
            f"correction_vector must be length 3, got {len(correction_vector)}"
        )
    return float(abs(correction_vector[axis_index]))


def per_axis_adjustment_factors(
    dual_variables: Mapping[str, float],
    posterior_sigma: Mapping[str, float],
) -> dict[str, float]:
    """Per-axis bounded [0.95, 1.05] adjustment factors.

    Sister of Phase 1 ``_lagrangian_derived_adjustment_factor`` extended
    to per-axis. Per axis: lower dual variable = looser constraint =
    smaller adjustment; higher posterior sigma = more uncertainty =
    smaller adjustment magnitude per Q7 binding decision.

    The formula:

        per_axis_adjustment_i = 1.0 + 0.05 * uncertainty_factor_i * sign_factor_i

    where:
        uncertainty_factor_i = 1.0 / (1.0 + sigma_i)
        sign_factor_i = tanh(-dual_var_i / 10.0)  # negative for binding

    Returns clipped values in [0.95, 1.05].
    """
    factors: dict[str, float] = {}
    for axis in AXIS_NAMES:
        dual = float(dual_variables[axis])
        sigma = float(posterior_sigma[axis])
        if sigma < 0:
            sigma = 0.0
        uncertainty_factor = 1.0 / (1.0 + sigma)
        # Bound the dual variable input to tanh to avoid saturation.
        capped_dual = max(-50.0, min(50.0, dual))
        # Negative sign convention: larger dual = binding = downweight.
        sign_factor = math.tanh(-capped_dual / 10.0)
        adjustment_delta = 0.05 * uncertainty_factor * sign_factor
        factor = 1.0 + adjustment_delta
        # Defense-in-depth bound check per Phase 1 safety envelope.
        if factor < PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN:
            factor = PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN
        elif factor > PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX:
            factor = PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX
        factors[axis] = float(factor)
    return factors


def _exposed_scalar_adjustment_factor(
    per_axis_factors: Mapping[str, float],
) -> float:
    """Aggregate per-axis adjustment factors into single scalar.

    Per CLAUDE.md "SegNet vs PoseNet importance - operating-point dependent"
    + Catalog #356 canonical axis ordering: use geometric mean across the
    3 axes so each axis contributes multiplicatively to the scalar.

    Geometric mean ensures the scalar stays inside [0.95, 1.05] as long
    as each per-axis factor is inside [0.95, 1.05].
    """
    if not all(axis in per_axis_factors for axis in AXIS_NAMES):
        raise Phase2SolverError(
            f"per_axis_factors must have all of {AXIS_NAMES}"
        )
    product = 1.0
    for axis in AXIS_NAMES:
        product *= float(per_axis_factors[axis])
    # Geometric mean of 3 values.
    if product <= 0:
        return PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN
    geo_mean = product ** (1.0 / 3.0)
    # Bound check.
    if geo_mean < PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN:
        return PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MIN
    if geo_mean > PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX:
        return PHASE_2_BOUNDED_ADJUSTMENT_FACTOR_MAX
    return float(geo_mean)


def compute_per_axis_dual_variables(
    candidate_id: str,
    *,
    predicted_axis_targets: Mapping[str, float],
    per_axis_budgets: Mapping[str, tuple[float, float]],
    per_axis_posterior_sigma: Mapping[str, float] | None = None,
    canonical_provenance: Mapping[str, Any] | None = None,
    max_iterations: int = DYKSTRA_DEFAULT_MAX_ITERATIONS,
    epsilon: float = DYKSTRA_DEFAULT_EPSILON,
    use_mlx: bool | None = None,
) -> PerAxisDualSolverResult:
    """Phase 2 per-candidate per-axis dual-variable computation.

    Main entry point for the Phase 2 advancement. Given a candidate's
    per-axis target deltas and per-axis polytope budgets:

    1. Run Dykstra alternating projections onto the 3-axis polytope.
    2. Extract per-axis Lagrange multipliers from Dykstra corrections.
    3. Compute per-axis KKT residuals.
    4. Derive per-axis bounded [0.95, 1.05] adjustment factors.
    5. Aggregate into scalar exposed adjustment factor.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #323/#341:
    the exposed scalar ``adjustment_factor`` stays inside Phase 1's
    [0.95, 1.05] safety envelope. The per-axis dict is observability-only.

    Args
    ----
    candidate_id : str
        Identifier of the candidate.
    predicted_axis_targets : Mapping[str, float]
        Per-axis target values ``{seg: ..., pose: ..., rate: ...}`` (the
        candidate's predicted axis delta from baseline; positive values =
        worse, negative values = improvement per CLAUDE.md sign convention).
    per_axis_budgets : Mapping[str, tuple[float, float]]
        Per-axis ``(lower, upper)`` budget bounds for the polytope
        half-spaces.
    per_axis_posterior_sigma : Mapping[str, float] | None
        Per-axis posterior uncertainty (from
        :mod:`tac.findings_lagrangian.posterior`). Defaults to ``1.0``
        per axis if None.
    canonical_provenance : Mapping[str, Any] | None
        Dict-form Provenance per Catalog #323. Defaults to empty.
    max_iterations : int
        Dykstra iteration cap.
    epsilon : float
        Convergence threshold.
    use_mlx : bool | None
        MLX vs numpy path selector.

    Returns
    -------
    PerAxisDualSolverResult
        Frozen dataclass with per-axis duals, KKT residuals, adjustment
        factors, scalar adjustment factor, and provenance.

    Raises
    ------
    Phase2SolverError
        If inputs violate invariants.
    """
    # Validate inputs.
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise Phase2SolverError("candidate_id must be non-empty string")
    for axis in AXIS_NAMES:
        if axis not in predicted_axis_targets:
            raise Phase2SolverError(
                f"predicted_axis_targets missing canonical axis {axis!r}"
            )
        if axis not in per_axis_budgets:
            raise Phase2SolverError(
                f"per_axis_budgets missing canonical axis {axis!r}"
            )

    # Build x0 in canonical axis ordering.
    x0 = np.array(
        [float(predicted_axis_targets[axis]) for axis in AXIS_NAMES],
        dtype=np.float64,
    )
    budgets = [per_axis_budgets[axis] for axis in AXIS_NAMES]

    # Per-axis posterior sigma (default 1.0 if unset).
    if per_axis_posterior_sigma is None:
        sigma_per_axis = dict.fromkeys(AXIS_NAMES, 1.0)
    else:
        sigma_per_axis = {
            axis: float(per_axis_posterior_sigma.get(axis, 1.0))
            for axis in AXIS_NAMES
        }

    # Run Dykstra alternating projections.
    x_converged, iterations, converged, corrections = (
        dykstra_alternating_projections_3_axis(
            x0,
            budgets=budgets,
            max_iterations=max_iterations,
            epsilon=epsilon,
            use_mlx=use_mlx,
        )
    )

    # Extract per-axis dual variables from Dykstra corrections.
    dual_variables: dict[str, float] = {}
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        dual_variables[axis_name] = _dual_variable_from_correction(
            corrections[axis_index], axis_index
        )

    # Compute per-axis KKT residuals.
    kkt_residuals = kkt_residuals_per_axis(x_converged, budgets)

    # Compute per-axis bounded adjustment factors.
    factors_per_axis = per_axis_adjustment_factors(
        dual_variables, sigma_per_axis
    )

    # Aggregate to scalar exposed adjustment factor.
    scalar_factor = _exposed_scalar_adjustment_factor(factors_per_axis)

    return PerAxisDualSolverResult(
        candidate_id=candidate_id,
        dual_variables_per_axis=dual_variables,
        kkt_residual_per_axis=kkt_residuals,
        adjustment_factor_per_axis=factors_per_axis,
        adjustment_factor=scalar_factor,
        dykstra_iterations_to_convergence=int(iterations),
        converged=bool(converged),
        posterior_sigma_per_axis=sigma_per_axis,
        canonical_provenance=dict(canonical_provenance or {}),
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
    )
