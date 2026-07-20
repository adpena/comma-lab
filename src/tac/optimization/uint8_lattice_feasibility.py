# SPDX-License-Identifier: MIT
"""Bounded uint8 preimages for a disjoint separable resize lattice.

This module places the camera-frame integer lattice *inside* the inverse solve.
For the contest downsample every scorer pixel has a disjoint camera-space
support, so the affine equality separates into small bounded Diophantine
problems.  Each block is searched exhaustively with interval and gcd pruning;
if a node budget is reached the result is explicitly ``NOT_FOUND_BUDGET``.

The optional hard-oracle repair is deliberately weaker: a nonlinear frozen
scorer supplies proposals, while fresh hard uint8 evaluations alone admit
moves.  A stall is therefore ``STALLED_UNKNOWN``, never an infeasibility proof.
No Dykstra or global nonlinear-optimum guarantee is claimed.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import ceil, floor, gcd, isfinite, ulp
from numbers import Integral, Real

import numpy as np

_RATIONAL_FLOAT_MATCH_ULPS = 32


class Uint8LatticeError(ValueError):
    """Fail-closed malformed-operator, target, or payload error."""


class BlockSolveStatus(StrEnum):
    FEASIBLE_EXACT = "FEASIBLE_EXACT"
    INFEASIBLE_EXHAUSTIVE = "INFEASIBLE_EXHAUSTIVE"
    NOT_FOUND_BUDGET = "NOT_FOUND_BUDGET"
    HEURISTIC_CANDIDATE = "HEURISTIC_CANDIDATE"


class CandidateProvenance(StrEnum):
    """How the returned values were constructed, independent of proof status."""

    EXACT_FEASIBLE_POINT = "EXACT_FEASIBLE_POINT"
    ADJACENT_CORNER_FALLBACK = "ADJACENT_CORNER_FALLBACK"


class RepairStatus(StrEnum):
    FEASIBLE = "FEASIBLE"
    STALLED_UNKNOWN = "STALLED_UNKNOWN"
    CYCLE_DETECTED_UNKNOWN = "CYCLE_DETECTED_UNKNOWN"
    BUDGET_EXHAUSTED_UNKNOWN = "BUDGET_EXHAUSTED_UNKNOWN"


@dataclass(frozen=True)
class AxisSupport:
    indices: tuple[int, ...]
    numerators: tuple[int, ...]
    denominator: int
    weights: tuple[float, ...]


@dataclass(frozen=True)
class BlockSolveResult:
    values: tuple[int, ...]
    status: BlockSolveStatus
    candidate_provenance: CandidateProvenance
    nodes_visited: int
    target_integer: int | None
    common_denominator: int
    exact_target_rational: bool
    projection_residual: float


@dataclass(frozen=True)
class LatticeSolveDiagnostics:
    exact_blocks: int
    heuristic_blocks: int
    budget_blocks: int
    proven_affine_infeasible_blocks: int
    nodes_visited: int
    max_projection_discrepancy: float
    mean_projection_discrepancy: float
    out_of_gamut_before_bounded_solve: int
    exact_candidate_blocks: int
    certified_exact: bool
    aggregate_status: BlockSolveStatus


@dataclass(frozen=True)
class LatticeFrameResult:
    frame: np.ndarray
    diagnostics: LatticeSolveDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _immutable_array_copy(self.frame))

    @property
    def certified_exact(self) -> bool:
        return self.diagnostics.certified_exact

    @property
    def aggregate_status(self) -> BlockSolveStatus:
        return self.diagnostics.aggregate_status


@dataclass(frozen=True)
class Factor2ExactVerification:
    """Exact proof for the canonical disjoint-support uint8 realization."""

    scorer_values: int
    owned_camera_values: int
    unowned_camera_values: int
    numerator_equal_values: int
    canonical_equal_values: int
    denominator: int
    numerator_exact: bool
    certified_exact: bool


@dataclass(frozen=True, order=True)
class IntegerMove:
    row: int
    col: int
    channel: int
    delta: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row", _require_integral_scalar(self.row, "move.row", minimum=0)
        )
        object.__setattr__(
            self, "col", _require_integral_scalar(self.col, "move.col", minimum=0)
        )
        object.__setattr__(
            self,
            "channel",
            _require_integral_scalar(self.channel, "move.channel", minimum=0),
        )
        delta = _require_integral_scalar(self.delta, "move.delta")
        if delta not in (-1, 1):
            raise Uint8LatticeError("move.delta must be exactly -1 or +1")
        object.__setattr__(self, "delta", delta)


@dataclass(frozen=True)
class HardOracleEvaluation:
    """One exact hard-oracle evaluation plus non-authoritative proposals."""

    satisfied: np.ndarray
    margins: np.ndarray
    proposals: tuple[IntegerMove, ...] = ()

    def __post_init__(self) -> None:
        raw_sat = np.asarray(self.satisfied)
        raw_margins = np.asarray(self.margins)
        if raw_sat.dtype.kind != "b":
            raise Uint8LatticeError(
                "hard-oracle satisfied must contain actual booleans"
            )
        if raw_margins.dtype.kind not in ("i", "u", "f"):
            raise Uint8LatticeError(
                "hard-oracle margins must contain real numeric scalars"
            )
        if not isinstance(self.proposals, tuple) or any(
            not isinstance(move, IntegerMove) for move in self.proposals
        ):
            raise Uint8LatticeError(
                "hard-oracle proposals must be a tuple of IntegerMove values"
            )
        sat = raw_sat.astype(bool, copy=False)
        margins = raw_margins.astype(np.float64, copy=False)
        if sat.shape != margins.shape or sat.size == 0:
            raise Uint8LatticeError("hard satisfied/margins must have one nonempty shape")
        if not np.all(np.isfinite(margins)):
            raise Uint8LatticeError("hard-oracle margins must be finite")
        if np.any(sat & (margins < 0.0)) or np.any(~sat & (margins > 0.0)):
            raise Uint8LatticeError(
                "hard-oracle satisfied flags and signed margins disagree"
            )
        _finite_hard_oracle_debt(margins)
        object.__setattr__(self, "satisfied", _immutable_array_copy(sat))
        object.__setattr__(self, "margins", _immutable_array_copy(margins))

    @property
    def key(self) -> tuple[int, float, float]:
        violations = int(np.count_nonzero(~self.satisfied))
        debt = _finite_hard_oracle_debt(self.margins)
        return violations, debt, -float(np.min(self.margins))


@dataclass(frozen=True)
class RepairIteration:
    iteration: int
    key_before: tuple[int, float, float]
    key_after: tuple[int, float, float]
    accepted_move: IntegerMove | None
    max_projection_drift: float


@dataclass(frozen=True)
class HardRepairResult:
    frame: np.ndarray
    status: RepairStatus
    evaluation: HardOracleEvaluation
    iterations: tuple[RepairIteration, ...]
    changed_lattice_coordinates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _immutable_array_copy(self.frame))


@dataclass(frozen=True)
class DisjointResizeOperator:
    """Certified disjoint-support separable bilinear resize."""

    camera_h: int
    camera_w: int
    scorer_h: int
    scorer_w: int
    row_supports: tuple[AxisSupport, ...]
    col_supports: tuple[AxisSupport, ...]

    def __post_init__(self) -> None:
        camera_h = _require_integral_scalar(self.camera_h, "camera_h", minimum=1)
        camera_w = _require_integral_scalar(self.camera_w, "camera_w", minimum=1)
        scorer_h = _require_integral_scalar(self.scorer_h, "scorer_h", minimum=1)
        scorer_w = _require_integral_scalar(self.scorer_w, "scorer_w", minimum=1)
        object.__setattr__(self, "camera_h", camera_h)
        object.__setattr__(self, "camera_w", camera_w)
        object.__setattr__(self, "scorer_h", scorer_h)
        object.__setattr__(self, "scorer_w", scorer_w)
        expected_rows = _exact_half_pixel_axis_supports(camera_h, scorer_h, "row")
        expected_cols = _exact_half_pixel_axis_supports(camera_w, scorer_w, "column")
        if not _supports_exactly_match(self.row_supports, expected_rows):
            raise Uint8LatticeError(
                "row_supports must exactly match derived half-pixel integer geometry"
            )
        if not _supports_exactly_match(self.col_supports, expected_cols):
            raise Uint8LatticeError(
                "col_supports must exactly match derived half-pixel integer geometry"
            )
        _refuse_overlapping_supports(self.row_supports, "row")
        _refuse_overlapping_supports(self.col_supports, "column")

    @classmethod
    def build(
        cls,
        *,
        camera_h: int,
        camera_w: int,
        scorer_h: int,
        scorer_w: int,
    ) -> DisjointResizeOperator:
        camera_h = _require_integral_scalar(camera_h, "camera_h", minimum=1)
        camera_w = _require_integral_scalar(camera_w, "camera_w", minimum=1)
        scorer_h = _require_integral_scalar(scorer_h, "scorer_h", minimum=1)
        scorer_w = _require_integral_scalar(scorer_w, "scorer_w", minimum=1)
        rows = _exact_half_pixel_axis_supports(camera_h, scorer_h, "row")
        cols = _exact_half_pixel_axis_supports(camera_w, scorer_w, "column")
        _refuse_overlapping_supports(rows, "row")
        _refuse_overlapping_supports(cols, "column")
        return cls(camera_h, camera_w, scorer_h, scorer_w, rows, cols)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        raw = np.asarray(frame)
        if raw.dtype.kind not in ("i", "u", "f"):
            raise Uint8LatticeError(
                "camera-frame values must have a real numeric non-boolean dtype"
            )
        x = raw.astype(np.float64, copy=False)
        if x.ndim == 2:
            x = x[:, :, None]
            squeeze = True
        elif x.ndim == 3:
            squeeze = False
        else:
            raise Uint8LatticeError("frame must have shape (H,W) or (H,W,C)")
        if x.shape[:2] != (self.camera_h, self.camera_w) or x.shape[2] < 1:
            raise Uint8LatticeError(
                "camera-frame shape must match the operator with nonempty channels"
            )
        if not np.all(np.isfinite(x)):
            raise Uint8LatticeError("camera-frame values must be finite")
        uniform = _uniform_support_arrays(self.row_supports, self.col_supports)
        if uniform is not None:
            row_indices, col_indices, weights = uniform
            blocks = x[
                row_indices[:, None, :, None],
                col_indices[None, :, None, :],
                :,
            ]
            out = np.sum(blocks * weights[..., None], axis=(2, 3))
            return _finite_apply_output(out, squeeze=squeeze)
        out = np.empty((self.scorer_h, self.scorer_w, x.shape[2]), dtype=np.float64)
        for oi, rs in enumerate(self.row_supports):
            for oj, cs in enumerate(self.col_supports):
                block = x[np.ix_(rs.indices, cs.indices, range(x.shape[2]))]
                weights = np.outer(rs.weights, cs.weights)
                out[oi, oj] = np.tensordot(weights, block, axes=((0, 1), (0, 1)))
        return _finite_apply_output(out, squeeze=squeeze)

    def apply_numerators(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """Return the exact integer ``A(frame)`` numerators and denominator."""
        x = np.asarray(frame)
        if x.ndim == 2:
            x = x[:, :, None]
            squeeze = True
        elif x.ndim == 3:
            squeeze = False
        else:
            raise Uint8LatticeError("integer frame must have shape (H,W) or (H,W,C)")
        if x.shape[:2] != (self.camera_h, self.camera_w) or x.shape[2] < 1:
            raise Uint8LatticeError(
                "apply_numerators requires matching geometry and nonempty channels"
            )
        if x.dtype.kind not in ("i", "u"):
            raise Uint8LatticeError(
                "apply_numerators requires a non-boolean integer camera frame"
            )
        if np.any(x < 0) or np.any(x > 255):
            raise Uint8LatticeError(
                "apply_numerators input must stay inside the uint8 lattice [0,255]"
            )
        denominators = {
            int(rs.denominator) * int(cs.denominator)
            for rs in self.row_supports
            for cs in self.col_supports
        }
        if len(denominators) != 1:
            raise Uint8LatticeError("operator lacks a single scorer-plane denominator")
        denominator = denominators.pop()
        if denominator > np.iinfo(np.int64).max // 255:
            raise Uint8LatticeError(
                "exact numerator geometry exceeds safe int64 accumulation range"
            )
        uniform = _uniform_support_arrays(self.row_supports, self.col_supports, integer_weights=True)
        if uniform is not None:
            row_indices, col_indices, coefficients = uniform
            blocks = x.astype(np.int64, copy=False)[
                row_indices[:, None, :, None],
                col_indices[None, :, None, :],
                :,
            ]
            out = np.sum(blocks * coefficients[..., None], axis=(2, 3), dtype=np.int64)
            return (out[:, :, 0] if squeeze else out), denominator
        out = np.empty((self.scorer_h, self.scorer_w, x.shape[2]), dtype=np.int64)
        x64 = x.astype(np.int64, copy=False)
        for oi, rs in enumerate(self.row_supports):
            for oj, cs in enumerate(self.col_supports):
                block = x64[np.ix_(rs.indices, cs.indices, range(x.shape[2]))]
                coefficients = np.outer(rs.numerators, cs.numerators).astype(np.int64)
                out[oi, oj] = np.tensordot(coefficients, block, axes=((0, 1), (0, 1)))
        return (out[:, :, 0] if squeeze else out), denominator

    def realize_factor2_uint8(self, target: np.ndarray) -> np.ndarray:
        """Return the canonical exact uint8 preimage for an integer scorer plane.

        Each scorer value owns a disjoint camera-space support.  Assigning the
        target byte to every tap in that support is exact because the tap
        coefficients sum to the common denominator.  Camera coordinates not
        owned by any support remain zero.  This path is integer-only and does
        not consult a source frame, preference, scorer, or floating-point
        arithmetic.
        """

        y = _factor2_uint8_target(target, self.scorer_h, self.scorer_w)
        out = np.zeros((self.camera_h, self.camera_w, y.shape[2]), dtype=np.uint8)
        uniform = _uniform_support_arrays(self.row_supports, self.col_supports)
        if uniform is not None:
            row_indices, col_indices, _weights = uniform
            if row_indices.shape[1] > 2 or col_indices.shape[1] > 2:
                raise Uint8LatticeError("factor-2 support exceeds two taps")
            if any(
                sum(support.numerators) != support.denominator
                for support in (*self.row_supports, *self.col_supports)
            ):
                raise Uint8LatticeError(
                    "factor-2 support coefficients do not sum to the denominator"
                )
            if (
                np.unique(row_indices).size != row_indices.size
                or np.unique(col_indices).size != col_indices.size
            ):
                raise Uint8LatticeError("factor-2 camera supports overlap")
            # The certified factor-2 geometry has uniform disjoint supports.
            # Four whole-plane assignments replace one Python iteration per
            # scorer pixel while preserving the exact canonical support fill.
            for row_offset in range(row_indices.shape[1]):
                for col_offset in range(col_indices.shape[1]):
                    out[
                        row_indices[:, row_offset, None],
                        col_indices[None, :, col_offset],
                        :,
                    ] = y
            return out if np.asarray(target).ndim == 3 else out[:, :, 0]

        owned = np.zeros((self.camera_h, self.camera_w), dtype=bool)
        for oi, row_support in enumerate(self.row_supports):
            if len(row_support.indices) > 2:
                raise Uint8LatticeError("factor-2 row support exceeds two taps")
            for oj, col_support in enumerate(self.col_supports):
                if len(col_support.indices) > 2:
                    raise Uint8LatticeError("factor-2 column support exceeds two taps")
                if sum(row_support.numerators) != row_support.denominator or sum(
                    col_support.numerators
                ) != col_support.denominator:
                    raise Uint8LatticeError(
                        "factor-2 support coefficients do not sum to the denominator"
                    )
                index = np.ix_(row_support.indices, col_support.indices)
                if np.any(owned[index]):
                    raise Uint8LatticeError("factor-2 camera supports overlap")
                owned[index] = True
                out[np.ix_(row_support.indices, col_support.indices, range(y.shape[2]))] = (
                    y[oi, oj]
                )
        return out if np.asarray(target).ndim == 3 else out[:, :, 0]

    def verify_factor2_uint8(
        self, frame: np.ndarray, target: np.ndarray
    ) -> Factor2ExactVerification:
        """Verify numerator equality and the deterministic canonical feasible point."""

        y = _factor2_uint8_target(target, self.scorer_h, self.scorer_w)
        raw_frame = np.asarray(frame)
        canonical = self.realize_factor2_uint8(y)
        if raw_frame.ndim == 2:
            raw_frame_3d = raw_frame[:, :, None]
        elif raw_frame.ndim == 3:
            raw_frame_3d = raw_frame
        else:
            raise Uint8LatticeError("factor-2 frame must have shape (H,W) or (H,W,C)")
        if raw_frame_3d.dtype != np.uint8 or raw_frame_3d.shape != canonical.shape:
            raise Uint8LatticeError(
                "factor-2 frame must be uint8 with the operator camera geometry"
            )
        numerators, denominator = self.apply_numerators(raw_frame_3d)
        expected = y.astype(np.int64) * denominator
        canonical_equal = raw_frame_3d == canonical
        numerator_equal = numerators == expected
        owned = np.any(canonical != 0, axis=2)
        # A target byte of zero makes owned and unowned coordinates
        # indistinguishable from values alone, so derive ownership from geometry.
        owned.fill(False)
        uniform = _uniform_support_arrays(self.row_supports, self.col_supports)
        if uniform is not None:
            row_indices, col_indices, _weights = uniform
            owned[np.ix_(np.unique(row_indices), np.unique(col_indices))] = True
        else:
            for row_support in self.row_supports:
                for col_support in self.col_supports:
                    owned[np.ix_(row_support.indices, col_support.indices)] = True
        certified = bool(np.all(numerator_equal) and np.all(canonical_equal))
        return Factor2ExactVerification(
            scorer_values=int(y.size),
            owned_camera_values=int(np.count_nonzero(owned) * y.shape[2]),
            unowned_camera_values=int(np.count_nonzero(~owned) * y.shape[2]),
            numerator_equal_values=int(np.count_nonzero(numerator_equal)),
            canonical_equal_values=int(np.count_nonzero(canonical_equal)),
            denominator=int(denominator),
            numerator_exact=bool(np.all(numerator_equal)),
            certified_exact=certified,
        )

    def minimum_norm_real_preimage(self, target: np.ndarray) -> np.ndarray:
        y = _target_3d(target, self.scorer_h, self.scorer_w)
        out = np.zeros((self.camera_h, self.camera_w, y.shape[2]), dtype=np.float64)
        uniform = _uniform_support_arrays(self.row_supports, self.col_supports)
        if uniform is not None:
            row_indices, col_indices, weights = uniform
            denominator = np.sum(np.square(weights), axis=(2, 3))
            with np.errstate(over="ignore", invalid="ignore"):
                values = weights[..., None] * (
                    y[:, :, None, None, :]
                    / denominator[:, :, None, None, None]
                )
            out[
                row_indices[:, None, :, None],
                col_indices[None, :, None, :],
                :,
            ] = values
            return _finite_minimum_norm_output(
                out, squeeze=np.asarray(target).ndim != 3
            )
        for oi, rs in enumerate(self.row_supports):
            for oj, cs in enumerate(self.col_supports):
                weights = np.outer(rs.weights, cs.weights)
                denom = float(np.square(weights).sum())
                for ch in range(y.shape[2]):
                    with np.errstate(over="ignore", invalid="ignore"):
                        values = weights[:, :, None] * (y[oi, oj, ch] / denom)
                    out[np.ix_(rs.indices, cs.indices, (ch,))] = values
        return _finite_minimum_norm_output(
            out, squeeze=np.asarray(target).ndim != 3
        )

    def bounded_continuous_preimage(self, target: np.ndarray, *, reference: np.ndarray | None = None) -> np.ndarray:
        y = _target_3d(target, self.scorer_h, self.scorer_w)
        if reference is None:
            reference = self.minimum_norm_real_preimage(y)
        z = _camera_3d(reference, self.camera_h, self.camera_w, y.shape[2])
        out = np.clip(z, 0.0, 255.0)
        uniform = _uniform_support_arrays(self.row_supports, self.col_supports)
        if uniform is not None:
            row_indices, col_indices, weights_grid = uniform
            blocks = z[
                row_indices[:, None, :, None],
                col_indices[None, :, None, :],
                :,
            ]
            weights = weights_grid.reshape(self.scorer_h, self.scorer_w, -1)
            values = blocks.reshape(self.scorer_h, self.scorer_w, -1, y.shape[2])
            lo = np.min((-values) / weights[..., None], axis=2) - 1.0
            hi = np.max((255.0 - values) / weights[..., None], axis=2) + 1.0
            for _ in range(56):
                mid = (lo + hi) / 2.0
                projected = np.sum(
                    weights[..., None] * np.clip(values + mid[:, :, None, :] * weights[..., None], 0.0, 255.0),
                    axis=2,
                )
                below = projected < y
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            solved = np.clip(
                values + ((lo + hi) / 2.0)[:, :, None, :] * weights[..., None],
                0.0,
                255.0,
            ).reshape(blocks.shape)
            out[
                row_indices[:, None, :, None],
                col_indices[None, :, None, :],
                :,
            ] = solved
            if float(np.max(np.abs(self.apply(out) - y))) > 2e-9:
                raise Uint8LatticeError("vectorized bounded affine projection did not converge")
            return out if np.asarray(target).ndim == 3 else out[:, :, 0]
        for oi, rs in enumerate(self.row_supports):
            for oj, cs in enumerate(self.col_supports):
                weights = np.outer(rs.weights, cs.weights).reshape(-1)
                for ch in range(y.shape[2]):
                    idx = np.ix_(rs.indices, cs.indices, (ch,))
                    base = z[idx].reshape(-1)
                    solved = _bounded_affine_projection(weights, base, float(y[oi, oj, ch]))
                    out[idx] = solved.reshape(len(rs.indices), len(cs.indices), 1)
        return out if np.asarray(target).ndim == 3 else out[:, :, 0]

    def solve_uint8(
        self,
        target: np.ndarray,
        *,
        target_numerators: np.ndarray | None = None,
        reference: np.ndarray | None = None,
        preferred_preimage: np.ndarray | None = None,
        max_nodes_per_block: int = 4096,
        target_verification_tolerance: float = 5e-10,
    ) -> LatticeFrameResult:
        """Construct a uint8 preimage with one unchanged exact block solver.

        ``reference`` is retained as a compatibility-time integrity assertion:
        when supplied it must be bit-identical to ``B(target)``.  It is never
        used to choose values.  This fails closed if a hidden/source camera
        frame is accidentally passed through the public solve boundary.

        ``preferred_preimage`` is an explicit, bounded continuous proposal used
        only to order feasible values inside each call to
        :func:`solve_bounded_integer_block`.  It cannot change the authoritative
        target numerator, proof status, or acceptance rule.  Callers must keep
        custody for how this proposal was produced; decoded hard-oracle
        acceptance remains a separate mandatory step.
        """

        max_nodes_per_block = _require_integral_scalar(
            max_nodes_per_block, "max_nodes_per_block", minimum=1
        )
        target_verification_tolerance = _require_finite_real_scalar(
            target_verification_tolerance,
            "target_verification_tolerance",
            minimum=0.0,
        )
        y = _target_3d(target, self.scorer_h, self.scorer_w)
        authoritative_numerators: np.ndarray | None = None
        if target_numerators is not None:
            authoritative_numerators = np.asarray(target_numerators)
            if authoritative_numerators.ndim == 2:
                authoritative_numerators = authoritative_numerators[:, :, None]
            if authoritative_numerators.shape != y.shape or not np.issubdtype(
                authoritative_numerators.dtype, np.integer
            ):
                raise Uint8LatticeError("target_numerators must be an integer scorer plane")
        derived_real = self.minimum_norm_real_preimage(y)
        if reference is not None:
            asserted_reference = _camera_3d(
                reference, self.camera_h, self.camera_w, y.shape[2]
            )
            derived_reference = _camera_3d(
                derived_real, self.camera_h, self.camera_w, y.shape[2]
            )
            if not np.array_equal(asserted_reference, derived_reference):
                raise Uint8LatticeError(
                    "solve reference must equal target-derived minimum-norm preimage; "
                    "source-dependent preferences are forbidden"
                )
        z = _camera_3d(derived_real, self.camera_h, self.camera_w, y.shape[2])
        if preferred_preimage is None:
            bounded = _camera_3d(
                self.bounded_continuous_preimage(y, reference=z),
                self.camera_h,
                self.camera_w,
                y.shape[2],
            )
        else:
            bounded = _camera_3d(
                preferred_preimage,
                self.camera_h,
                self.camera_w,
                y.shape[2],
            )
            if np.any(bounded < 0.0) or np.any(bounded > 255.0):
                raise Uint8LatticeError(
                    "preferred_preimage must stay inside the uint8 box [0,255]"
                )
        out = np.rint(np.clip(bounded, 0.0, 255.0)).astype(np.uint8)
        counts = dict.fromkeys(BlockSolveStatus, 0)
        candidate_counts = dict.fromkeys(CandidateProvenance, 0)
        total_nodes = 0

        for oi, rs in enumerate(self.row_supports):
            for oj, cs in enumerate(self.col_supports):
                denominator = rs.denominator * cs.denominator
                coefficients = tuple(
                    np.outer(rs.numerators, cs.numerators).reshape(-1)
                )
                for ch in range(y.shape[2]):
                    idx = np.ix_(rs.indices, cs.indices, (ch,))
                    preferred = bounded[idx].reshape(-1)
                    result = solve_bounded_integer_block(
                        coefficients,
                        denominator,
                        float(y[oi, oj, ch]),
                        target_integer=(
                            None if authoritative_numerators is None else int(authoritative_numerators[oi, oj, ch])
                        ),
                        preferred=preferred,
                        max_nodes=max_nodes_per_block,
                        target_verification_tolerance=target_verification_tolerance,
                    )
                    out[idx] = np.asarray(result.values, dtype=np.uint8).reshape(len(rs.indices), len(cs.indices), 1)
                    counts[result.status] += 1
                    candidate_counts[result.candidate_provenance] += 1
                    total_nodes += result.nodes_visited

        projected = _target_3d(self.apply(out), self.scorer_h, self.scorer_w)
        residual = np.abs(projected - y)
        certified_exact = (
            authoritative_numerators is not None
            and counts[BlockSolveStatus.FEASIBLE_EXACT] == y.size
        )
        if certified_exact:
            aggregate_status = BlockSolveStatus.FEASIBLE_EXACT
        elif counts[BlockSolveStatus.INFEASIBLE_EXHAUSTIVE]:
            aggregate_status = BlockSolveStatus.INFEASIBLE_EXHAUSTIVE
        elif counts[BlockSolveStatus.NOT_FOUND_BUDGET]:
            aggregate_status = BlockSolveStatus.NOT_FOUND_BUDGET
        else:
            aggregate_status = BlockSolveStatus.HEURISTIC_CANDIDATE
        diagnostics = LatticeSolveDiagnostics(
            exact_blocks=counts[BlockSolveStatus.FEASIBLE_EXACT],
            heuristic_blocks=candidate_counts[
                CandidateProvenance.ADJACENT_CORNER_FALLBACK
            ],
            budget_blocks=counts[BlockSolveStatus.NOT_FOUND_BUDGET],
            proven_affine_infeasible_blocks=counts[
                BlockSolveStatus.INFEASIBLE_EXHAUSTIVE
            ],
            nodes_visited=total_nodes,
            max_projection_discrepancy=float(residual.max(initial=0.0)),
            mean_projection_discrepancy=float(residual.mean()),
            out_of_gamut_before_bounded_solve=int(np.count_nonzero((z < 0.0) | (z > 255.0))),
            exact_candidate_blocks=candidate_counts[
                CandidateProvenance.EXACT_FEASIBLE_POINT
            ],
            certified_exact=certified_exact,
            aggregate_status=aggregate_status,
        )
        result_frame = out if np.asarray(target).ndim == 3 else out[:, :, 0]
        return LatticeFrameResult(result_frame, diagnostics)


def solve_bounded_integer_block(
    coefficients: Sequence[int],
    common_denominator: int,
    target: float,
    *,
    target_integer: int | None = None,
    preferred: Sequence[float] | np.ndarray | None = None,
    max_nodes: int = 4096,
    target_verification_tolerance: float = 5e-10,
) -> BlockSolveResult:
    """Solve ``c dot u = target_integer``, ``u in [0,255]^n``.

    Gcd-pruned DFS exhausts the bounded block unless ``max_nodes`` interrupts it.
    Only an explicit integer numerator can receive an exact/proven verdict.  A
    float-only target returns an explicitly heuristic adjacent-corner candidate.
    """

    try:
        raw_coefficients = tuple(coefficients)
    except TypeError as exc:
        raise Uint8LatticeError("coefficients must be a finite integer sequence") from exc
    if not raw_coefficients:
        raise Uint8LatticeError("coefficients must be nonempty")
    if len(raw_coefficients) > 4:
        raise Uint8LatticeError(
            "factor-2 block solver supports at most four disjoint 2x2 taps"
        )
    coeff = tuple(
        _require_integral_scalar(value, f"coefficients[{index}]", minimum=1)
        for index, value in enumerate(raw_coefficients)
    )
    common_denominator = _require_integral_scalar(
        common_denominator, "common_denominator", minimum=1
    )
    max_nodes = _require_integral_scalar(max_nodes, "max_nodes", minimum=1)
    target = _require_finite_real_scalar(target, "target")
    target_verification_tolerance = _require_finite_real_scalar(
        target_verification_tolerance,
        "target_verification_tolerance",
        minimum=0.0,
    )
    if target_integer is not None:
        target_integer = _require_integral_scalar(target_integer, "target_integer")
    if preferred is None:
        pref = np.full(len(coeff), 127.5)
    else:
        raw_preferred = np.asarray(preferred)
        if raw_preferred.dtype.kind not in ("i", "u", "f"):
            raise Uint8LatticeError(
                "preferred must have a real numeric non-boolean dtype"
            )
        pref = raw_preferred.astype(np.float64, copy=False)
    if pref.shape != (len(coeff),) or not np.all(np.isfinite(pref)):
        raise Uint8LatticeError("preferred must be finite and match coefficients")
    if np.any(pref < 0.0) or np.any(pref > 255.0):
        raise Uint8LatticeError("preferred must stay inside the uint8 box [0,255]")

    if target_integer is None:
        values = _adjacent_corner_candidate(coeff, common_denominator, target, pref)
        residual = abs(sum(c * v for c, v in zip(coeff, values, strict=True)) / common_denominator - target)
        return BlockSolveResult(
            values,
            BlockSolveStatus.HEURISTIC_CANDIDATE,
            CandidateProvenance.ADJACENT_CORNER_FALLBACK,
            0,
            None,
            common_denominator,
            False,
            float(residual),
        )
    try:
        rational_target = target_integer / common_denominator
    except OverflowError as exc:
        raise Uint8LatticeError(
            "authoritative integer target ratio is outside finite float range"
        ) from exc
    if not isfinite(rational_target):
        raise Uint8LatticeError(
            "authoritative integer target ratio is outside finite float range"
        )
    fixed_match_tolerance = _fixed_rational_float_match_tolerance(rational_target)
    effective_match_tolerance = min(
        target_verification_tolerance, fixed_match_tolerance
    )
    if abs(target - rational_target) > effective_match_tolerance:
        raise Uint8LatticeError(
            "float target disagrees with authoritative integer numerator under "
            "the fixed machine-derived bound"
        )

    order = tuple(sorted(range(len(coeff)), key=lambda i: (-coeff[i], i)))
    sorted_c = tuple(coeff[i] for i in order)
    sorted_p = tuple(float(pref[i]) for i in order)
    suffix_sum = [0] * (len(coeff) + 1)
    suffix_gcd = [0] * (len(coeff) + 1)
    for i in range(len(coeff) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_c[i]
        suffix_gcd[i] = gcd(sorted_c[i], suffix_gcd[i + 1])

    nodes = 0
    budget_hit = False

    def search(i: int, residual: int) -> tuple[int, ...] | None:
        nonlocal nodes, budget_hit
        if i == len(sorted_c) - 1:
            c = sorted_c[i]
            if residual % c:
                return None
            value = residual // c
            if not 0 <= value <= 255:
                return None
            if nodes >= max_nodes:
                budget_hit = True
                return None
            nodes += 1
            return (value,)

        c = sorted_c[i]
        remaining_max = 255 * suffix_sum[i + 1]
        lo = max(0, _ceil_div(residual - remaining_max, c))
        hi = min(255, residual // c)
        if lo > hi:
            return None
        g_rest = suffix_gcd[i + 1]
        d = gcd(c, g_rest)
        if residual % d:
            return None
        modulus = g_rest // d
        residue_class = 0 if modulus == 1 else ((residual // d) * pow(c // d, -1, modulus)) % modulus
        first = residue_class + _ceil_div(lo - residue_class, modulus) * modulus
        candidates = list(range(first, hi + 1, modulus))
        candidates.sort(key=lambda value: (abs(value - sorted_p[i]), value))
        for value in candidates:
            if nodes >= max_nodes:
                budget_hit = True
                return None
            nodes += 1
            tail = search(i + 1, residual - c * value)
            if tail is not None:
                return (value, *tail)
        return None

    g_all = suffix_gcd[0]
    sorted_values: tuple[int, ...] | None = None
    impossible_by_gcd = target_integer % g_all != 0
    if not impossible_by_gcd:
        sorted_values = search(0, target_integer)

    if sorted_values is None:
        heuristic = _adjacent_corner_candidate(coeff, common_denominator, target, pref)
        status = (
            BlockSolveStatus.NOT_FOUND_BUDGET
            if budget_hit
            else BlockSolveStatus.INFEASIBLE_EXHAUSTIVE
        )
        residual = abs(sum(c * value for c, value in zip(coeff, heuristic, strict=True)) / common_denominator - target)
        return BlockSolveResult(
            heuristic,
            status,
            CandidateProvenance.ADJACENT_CORNER_FALLBACK,
            nodes,
            target_integer,
            common_denominator,
            True,
            float(residual),
        )

    values = [0] * len(coeff)
    for sorted_index, original_index in enumerate(order):
        values[original_index] = sorted_values[sorted_index]
    residual = abs(sum(c * value for c, value in zip(coeff, values, strict=True)) / common_denominator - target)
    return BlockSolveResult(
        tuple(values),
        BlockSolveStatus.FEASIBLE_EXACT,
        CandidateProvenance.EXACT_FEASIBLE_POINT,
        nodes,
        target_integer,
        common_denominator,
        True,
        float(residual),
    )


def realize_factor2_uint8_scorer_plane(
    operator: DisjointResizeOperator, target: np.ndarray
) -> np.ndarray:
    """Public functional form of :meth:`DisjointResizeOperator.realize_factor2_uint8`."""

    if not isinstance(operator, DisjointResizeOperator):
        raise Uint8LatticeError("factor-2 realization requires DisjointResizeOperator")
    return operator.realize_factor2_uint8(target)


def verify_factor2_uint8_scorer_plane(
    operator: DisjointResizeOperator, frame: np.ndarray, target: np.ndarray
) -> Factor2ExactVerification:
    """Public functional form of :meth:`DisjointResizeOperator.verify_factor2_uint8`."""

    if not isinstance(operator, DisjointResizeOperator):
        raise Uint8LatticeError("factor-2 verification requires DisjointResizeOperator")
    return operator.verify_factor2_uint8(frame, target)


def repair_with_hard_oracle(
    frame: np.ndarray,
    operator: DisjointResizeOperator,
    oracle: Callable[[np.ndarray], HardOracleEvaluation],
    *,
    max_iterations: int = 16,
    max_proposals_per_iteration: int = 64,
    max_projection_drift: float = 255.0,
) -> HardRepairResult:
    """Deterministic +/-1 repair admitted only by fresh hard-oracle keys."""

    initial = np.asarray(frame)
    if initial.dtype != np.uint8 or initial.ndim != 3:
        raise Uint8LatticeError("repair frame must be HxWxC uint8")
    max_iterations = _require_integral_scalar(
        max_iterations, "max_iterations", minimum=0
    )
    max_proposals_per_iteration = _require_integral_scalar(
        max_proposals_per_iteration,
        "max_proposals_per_iteration",
        minimum=1,
    )
    max_projection_drift = _require_finite_real_scalar(
        max_projection_drift,
        "max_projection_drift",
        minimum=0.0,
    )
    current = initial.copy()
    baseline_projection = operator.apply(current)
    evaluation = _evaluate_hard_oracle(oracle, current)
    obligation_shape = evaluation.satisfied.shape
    if evaluation.key[0] == 0:
        return HardRepairResult(current, RepairStatus.FEASIBLE, evaluation, (), 0)
    seen = {sha256(current.tobytes()).digest()}
    history: list[RepairIteration] = []

    for iteration in range(max_iterations):
        accepted = False
        admissible_proposals = 0
        seen_admissible_proposals = 0
        evaluated_unseen_proposals = 0
        proposal_budget_exhausted = False
        proposals = sorted(set(evaluation.proposals))
        if not proposals:
            return _repair_result(initial, current, RepairStatus.STALLED_UNKNOWN, evaluation, history)
        for move in proposals:
            if move.delta not in (-1, 1):
                raise Uint8LatticeError("hard-oracle proposals must be unit +/-1 moves")
            if not (
                0 <= move.row < current.shape[0]
                and 0 <= move.col < current.shape[1]
                and 0 <= move.channel < current.shape[2]
            ):
                raise Uint8LatticeError("hard-oracle proposal is out of frame bounds")
            old_value = int(current[move.row, move.col, move.channel])
            new_value = old_value + move.delta
            if not 0 <= new_value <= 255:
                continue
            candidate = current.copy()
            candidate[move.row, move.col, move.channel] = new_value
            digest = sha256(candidate.tobytes()).digest()
            drift = float(np.max(np.abs(operator.apply(candidate) - baseline_projection)))
            if drift > max_projection_drift:
                continue
            admissible_proposals += 1
            if digest in seen:
                seen_admissible_proposals += 1
                continue
            if evaluated_unseen_proposals >= max_proposals_per_iteration:
                proposal_budget_exhausted = True
                continue
            evaluated_unseen_proposals += 1
            candidate_evaluation = _evaluate_hard_oracle(
                oracle, candidate, expected_shape=obligation_shape
            )
            if candidate_evaluation.key < evaluation.key:
                history.append(RepairIteration(iteration, evaluation.key, candidate_evaluation.key, move, drift))
                current = candidate
                evaluation = candidate_evaluation
                seen.add(digest)
                accepted = True
                break
        if not accepted:
            history.append(RepairIteration(iteration, evaluation.key, evaluation.key, None, 0.0))
            if proposal_budget_exhausted:
                return _repair_result(
                    initial,
                    current,
                    RepairStatus.BUDGET_EXHAUSTED_UNKNOWN,
                    evaluation,
                    history,
                )
            if (
                admissible_proposals > 0
                and seen_admissible_proposals == admissible_proposals
                and evaluated_unseen_proposals == 0
            ):
                return _repair_result(
                    initial,
                    current,
                    RepairStatus.CYCLE_DETECTED_UNKNOWN,
                    evaluation,
                    history,
                )
            return _repair_result(initial, current, RepairStatus.STALLED_UNKNOWN, evaluation, history)
        if evaluation.key[0] == 0:
            return _repair_result(initial, current, RepairStatus.FEASIBLE, evaluation, history)
    return _repair_result(initial, current, RepairStatus.BUDGET_EXHAUSTED_UNKNOWN, evaluation, history)


_PAYLOAD_MAGIC = b"U8LF1"
_PAYLOAD_HEADER = struct.Struct(">5sIII32s")
MAX_UINT8_FRAME_BYTES = 874 * 1164 * 3


def serialize_uint8_frame(frame: np.ndarray) -> bytes:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8 or arr.ndim != 3 or any(dim <= 0 for dim in arr.shape):
        raise Uint8LatticeError("payload frame must be nonempty HxWxC uint8")
    if arr.nbytes > MAX_UINT8_FRAME_BYTES:
        raise Uint8LatticeError(
            "payload frame exceeds the default parser byte contract"
        )
    raw = np.ascontiguousarray(arr).tobytes()
    return _PAYLOAD_HEADER.pack(_PAYLOAD_MAGIC, *arr.shape, sha256(raw).digest()) + zlib.compress(raw, level=9)


def parse_uint8_frame(payload: bytes, *, max_frame_bytes: int = MAX_UINT8_FRAME_BYTES) -> np.ndarray:
    if len(payload) < _PAYLOAD_HEADER.size:
        raise Uint8LatticeError("truncated uint8 lattice payload")
    max_frame_bytes = _require_integral_scalar(
        max_frame_bytes, "max_frame_bytes", minimum=1
    )
    magic, height, width, channels, expected_hash = _PAYLOAD_HEADER.unpack_from(payload)
    if magic != _PAYLOAD_MAGIC or min(height, width, channels) <= 0:
        raise Uint8LatticeError("invalid uint8 lattice payload header")
    expected_bytes = height * width * channels
    if expected_bytes > max_frame_bytes:
        raise Uint8LatticeError("payload frame exceeds configured byte cap")
    compressed = payload[_PAYLOAD_HEADER.size :]
    if len(compressed) > max_frame_bytes + 1024:
        raise Uint8LatticeError("compressed payload exceeds configured byte cap")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(compressed, expected_bytes + 1)
    except zlib.error as exc:
        raise Uint8LatticeError("invalid compressed uint8 lattice payload") from exc
    if len(raw) > expected_bytes:
        raise Uint8LatticeError("decompressed payload exceeds declared frame size")
    if decompressor.unused_data or decompressor.unconsumed_tail or not decompressor.eof:
        raise Uint8LatticeError("payload has trailing, oversized, or incomplete compressed data")
    if len(raw) != expected_bytes or sha256(raw).digest() != expected_hash:
        raise Uint8LatticeError("payload size/hash custody failure")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels).copy()


def _uniform_support_arrays(
    row_supports: Sequence[AxisSupport],
    col_supports: Sequence[AxisSupport],
    *,
    integer_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    row_sizes = {len(support.indices) for support in row_supports}
    col_sizes = {len(support.indices) for support in col_supports}
    if len(row_sizes) != 1 or len(col_sizes) != 1:
        return None
    row_indices = np.asarray([support.indices for support in row_supports], dtype=np.intp)
    col_indices = np.asarray([support.indices for support in col_supports], dtype=np.intp)
    attribute = "numerators" if integer_weights else "weights"
    dtype = np.int64 if integer_weights else np.float64
    row_weights = np.asarray([getattr(support, attribute) for support in row_supports], dtype=dtype)
    col_weights = np.asarray([getattr(support, attribute) for support in col_supports], dtype=dtype)
    weights = row_weights[:, None, :, None] * col_weights[None, :, None, :]
    return row_indices, col_indices, weights


def _exact_half_pixel_axis_supports(input_size: int, output_size: int, axis_name: str) -> tuple[AxisSupport, ...]:
    """Derive align_corners=False taps as integers, never from float weights."""
    if input_size < 1 or output_size < 1:
        raise Uint8LatticeError(f"invalid {axis_name} resize geometry")
    denominator = 2 * output_size
    supports: list[AxisSupport] = []
    for output_index in range(output_size):
        coordinate_numerator = (2 * output_index + 1) * input_size - output_size
        left = coordinate_numerator // denominator
        fraction_numerator = coordinate_numerator - left * denominator
        taps: dict[int, int] = {}
        for raw_index, numerator in (
            (left, denominator - fraction_numerator),
            (left + 1, fraction_numerator),
        ):
            if numerator == 0:
                continue
            index = min(max(raw_index, 0), input_size - 1)
            taps[index] = taps.get(index, 0) + numerator
        indices = tuple(sorted(taps))
        numerators = tuple(taps[index] for index in indices)
        if not indices or len(indices) > 2 or sum(numerators) != denominator:
            raise Uint8LatticeError(f"invalid exact {axis_name} half-pixel taps")
        weights = tuple(numerator / denominator for numerator in numerators)
        supports.append(AxisSupport(indices, numerators, denominator, weights))
    return tuple(supports)


def _refuse_overlapping_supports(supports: Sequence[AxisSupport], axis_name: str) -> None:
    claimed: set[int] = set()
    for support in supports:
        if claimed.intersection(support.indices):
            raise Uint8LatticeError(f"overlapping {axis_name} resize-row supports are unsupported")
        claimed.update(support.indices)


def _bounded_affine_projection(weights: np.ndarray, reference: np.ndarray, target: float) -> np.ndarray:
    total = float(weights.sum())
    if not np.isfinite(target) or target < -1e-10 or target > 255.0 * total + 1e-10:
        raise Uint8LatticeError("target is outside bounded affine image")
    if target <= 1e-12:
        return np.zeros_like(reference)
    if target >= 255.0 * total - 1e-12:
        return np.full_like(reference, 255.0)

    # f(lambda)=w dot clip(z+lambda*w) is monotone piecewise affine.  Its only
    # breakpoints are where one of the at-most-four block coordinates enters or
    # leaves the box.  Locate the target segment, then solve it in closed form.
    breakpoints = sorted({*((0.0 - reference) / weights), *((255.0 - reference) / weights)})
    lower = breakpoints[0]
    for upper in breakpoints[1:]:
        upper_value = float(np.dot(weights, np.clip(reference + upper * weights, 0.0, 255.0)))
        if upper_value + 1e-12 < target:
            lower = upper
            continue
        midpoint = (lower + upper) / 2.0
        midpoint_values = reference + midpoint * weights
        free = (midpoint_values > 0.0) & (midpoint_values < 255.0)
        fixed = np.clip(midpoint_values, 0.0, 255.0)
        if not np.any(free):
            solved = np.clip(reference + upper * weights, 0.0, 255.0)
        else:
            fixed[free] = 0.0
            lam = (target - float(np.dot(weights, fixed)) - float(np.dot(weights[free], reference[free]))) / float(
                np.dot(weights[free], weights[free])
            )
            solved = np.clip(reference + lam * weights, 0.0, 255.0)
        break
    else:  # pragma: no cover - guarded by the bounded target check above
        raise Uint8LatticeError("failed to bracket bounded affine target")
    if abs(float(np.dot(weights, solved)) - target) > 2e-9:
        raise Uint8LatticeError("bounded affine projection did not converge")
    return solved


def _adjacent_corner_candidate(
    coefficients: Sequence[int], denominator: int, target: float, preferred: np.ndarray
) -> tuple[int, ...]:
    choices = [sorted({floor(v), ceil(v)}) for v in np.clip(preferred, 0, 255)]
    best: tuple[float, float, tuple[int, ...]] | None = None
    for values in _product(choices):
        projection = sum(c * v for c, v in zip(coefficients, values, strict=True)) / denominator
        key = (
            abs(projection - target),
            float(np.square(np.asarray(values, dtype=float) - preferred).sum()),
            values,
        )
        if best is None or key < best:
            best = key
    assert best is not None
    return best[2]


def _product(choices: Sequence[Sequence[int]]) -> Iterable[tuple[int, ...]]:
    values: list[tuple[int, ...]] = [()]
    for options in choices:
        values = [(*prefix, value) for prefix in values for value in options]
    return values


def _target_3d(target: np.ndarray, height: int, width: int) -> np.ndarray:
    raw = np.asarray(target)
    if raw.dtype.kind not in ("i", "u", "f"):
        raise Uint8LatticeError(
            "target must have a real numeric non-boolean dtype"
        )
    y = raw.astype(np.float64, copy=False)
    if y.ndim == 2:
        y = y[:, :, None]
    if (
        y.ndim != 3
        or y.shape[:2] != (height, width)
        or y.shape[2] < 1
        or not np.all(np.isfinite(y))
    ):
        raise Uint8LatticeError("target shape/value does not match scorer plane")
    return y


def _factor2_uint8_target(target: np.ndarray, height: int, width: int) -> np.ndarray:
    raw = np.asarray(target)
    if raw.dtype != np.uint8:
        raise Uint8LatticeError("factor-2 scorer plane must have exact uint8 dtype")
    if raw.ndim == 2:
        y = raw[:, :, None]
    elif raw.ndim == 3:
        y = raw
    else:
        raise Uint8LatticeError("factor-2 scorer plane must have shape (H,W) or (H,W,C)")
    if y.shape[:2] != (height, width) or y.shape[2] < 1:
        raise Uint8LatticeError(
            "factor-2 scorer plane must match scorer geometry with nonempty channels"
        )
    return np.ascontiguousarray(y)


def _camera_3d(frame: np.ndarray, height: int, width: int, channels: int) -> np.ndarray:
    if channels < 1:
        raise Uint8LatticeError("reference must declare at least one channel")
    raw = np.asarray(frame)
    if raw.dtype.kind not in ("i", "u", "f"):
        raise Uint8LatticeError(
            "reference must have a real numeric non-boolean dtype"
        )
    x = raw.astype(np.float64, copy=False)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.shape != (height, width, channels) or not np.all(np.isfinite(x)):
        raise Uint8LatticeError("reference shape/value does not match camera frame")
    return x


def _finite_apply_output(out: np.ndarray, *, squeeze: bool) -> np.ndarray:
    if not np.all(np.isfinite(out)):
        raise Uint8LatticeError("resize operator produced non-finite output")
    return out[:, :, 0] if squeeze else out


def _finite_minimum_norm_output(out: np.ndarray, *, squeeze: bool) -> np.ndarray:
    if not np.all(np.isfinite(out)):
        raise Uint8LatticeError(
            "minimum-norm real preimage produced non-finite values"
        )
    return out[:, :, 0] if squeeze else out


def _finite_hard_oracle_debt(margins: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        debt = float(np.maximum(-margins, 0.0).sum(dtype=np.float64))
    if not isfinite(debt):
        raise Uint8LatticeError("hard-oracle aggregate margin debt must be finite")
    return debt


def _immutable_array_copy(value: np.ndarray) -> np.ndarray:
    """Own values behind a read-only bytes buffer whose flag cannot be reopened."""

    contiguous = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _repair_result(
    initial: np.ndarray,
    current: np.ndarray,
    status: RepairStatus,
    evaluation: HardOracleEvaluation,
    history: Sequence[RepairIteration],
) -> HardRepairResult:
    changed = int(np.count_nonzero(initial != current))
    return HardRepairResult(current.copy(), status, evaluation, tuple(history), changed)


def _evaluate_hard_oracle(
    oracle: Callable[[np.ndarray], HardOracleEvaluation],
    frame: np.ndarray,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> HardOracleEvaluation:
    evaluation = oracle(frame.copy())
    if not isinstance(evaluation, HardOracleEvaluation):
        raise Uint8LatticeError(
            "hard oracle must return a HardOracleEvaluation instance"
        )
    if expected_shape is not None and evaluation.satisfied.shape != expected_shape:
        raise Uint8LatticeError(
            "hard-oracle obligation shape changed across repair evaluations"
        )
    return evaluation


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _fixed_rational_float_match_tolerance(
    expected: float,
) -> float:
    """Bound equality by a fixed float64 operation budget, never caller width."""

    tolerance = _RATIONAL_FLOAT_MATCH_ULPS * ulp(expected)
    if not isfinite(tolerance):  # pragma: no cover - float64 max stays finite
        raise Uint8LatticeError("machine-derived rational match bound overflowed")
    return tolerance


def _require_integral_scalar(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    """Return an exact Python int without accepting bools or float coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise Uint8LatticeError(f"{label} must be an integer scalar, not coerced")
    result = int(value)
    if minimum is not None and result < minimum:
        raise Uint8LatticeError(f"{label} must be at least {minimum}")
    return result


def _require_finite_real_scalar(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float:
    """Return a finite real scalar without accepting booleans or strings."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise Uint8LatticeError(f"{label} must be a finite real scalar")
    result = float(value)
    if not isfinite(result):
        raise Uint8LatticeError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise Uint8LatticeError(f"{label} must be at least {minimum}")
    return result


def _supports_exactly_match(
    actual: object,
    expected: tuple[AxisSupport, ...],
) -> bool:
    """Require typed integer support custody as well as numeric equality."""

    if not isinstance(actual, tuple) or len(actual) != len(expected):
        return False
    for candidate, derived in zip(actual, expected, strict=True):
        if not isinstance(candidate, AxisSupport):
            return False
        if (
            not isinstance(candidate.indices, tuple)
            or not isinstance(candidate.numerators, tuple)
            or not isinstance(candidate.weights, tuple)
        ):
            return False
        integer_fields = (
            *candidate.indices,
            *candidate.numerators,
            candidate.denominator,
        )
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in integer_fields
        ):
            return False
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not isfinite(float(value))
            for value in candidate.weights
        ):
            return False
        if candidate != derived:
            return False
    return True
