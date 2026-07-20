# SPDX-License-Identifier: MIT
"""Implicit compiler for the complete kernel of the shared scorer resize.

The contest resize is separable and, for the canonical factor-2 downsample,
has disjoint two-tap supports.  This module represents the full real-linear
kernel without materializing a ``(H*W) x nullity`` matrix.  Float projection
and exact uint8 claims deliberately use different authorities:

* ``FullResizeKernel.project_kernel`` applies the fp32/fp64 orthogonal
  projector ``I - Q_h (.) Q_w``;
* every uint8 equality is certified by
  :class:`tac.optimization.uint8_lattice_feasibility.DisjointResizeOperator`
  integer numerators.

The implicit nonredundant parameterization is

``K(U,V) = N_h U + A_h.T V N_w.T``

with ``U`` shaped ``(H-h, W[, C])`` and ``V`` shaped
``(h, W-w[, C])``.  Its parameter count is exactly
``(H-h)W + h(W-w) = HW-hw``.

Evidence scope: structural math plus local CPU advisory coder measurements.
No score, promotion, dispatch, or pointer authority is conferred here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd
from typing import Any, Literal

import numpy as np

from tac.optimization.evaluator_invisibility_basis import (
    CAMERA_H,
    CAMERA_W,
    SCORER_INPUT_H,
    SCORER_INPUT_W,
)
from tac.optimization.resize_null_preimage import (
    ResizeProjector,
    apply_tier1_zero_weight_fill,
    coded_size_both,
)
from tac.optimization.uint8_lattice_feasibility import (
    AxisSupport,
    BlockSolveStatus,
    DisjointResizeOperator,
    solve_bounded_integer_block,
)

FULL_RESIZE_KERNEL_SCHEMA = "resize_null_preimage_full_kernel.v1"
UINT8_REACHABILITY_SEMANTICS = (
    "canonical_tensor_primitive_basis_bounded_reachability_lower_bound"
)


class FullResizeKernelError(ValueError):
    """Fail-closed invalid geometry, coefficients, or exactness claim."""


def _primitive_pair(first: int, second: int) -> tuple[int, int]:
    divisor = gcd(abs(int(first)), abs(int(second)))
    if divisor == 0:
        raise FullResizeKernelError("zero two-tap support has no primitive pair")
    return int(first) // divisor, int(second) // divisor


def _as_float_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise FullResizeKernelError("projection input must contain real numeric values")
    out = raw.astype(dtype, copy=False)
    if not np.all(np.isfinite(out)):
        raise FullResizeKernelError("projection input must be finite")
    return out


def _select_coder_admitted_name(
    sizes: dict[str, dict[str, int]], *, baseline_name: str
) -> str:
    """Return the Pareto-safe coder winner, retaining the baseline on ties."""

    baseline = sizes[baseline_name]
    coders = ("brotli", "lzma")
    eligible = [
        name
        for name, candidate in sizes.items()
        if all(candidate[coder] <= baseline[coder] for coder in coders)
    ]
    return min(
        eligible,
        key=lambda name: (
            sizes[name]["brotli"],
            sizes[name]["lzma"],
            name != baseline_name,
            name,
        ),
    )


@dataclass(frozen=True)
class ImplicitAxisKernel:
    """One-axis row-space projector and exact implicit kernel basis.

    Basis order is deterministic: one primitive two-tap null atom per output
    support, followed by coordinate atoms for unowned input indices.
    """

    n_in: int
    n_out: int
    supports: tuple[AxisSupport, ...]
    unowned_indices: tuple[int, ...]

    @classmethod
    def from_supports(
        cls, n_in: int, supports: Sequence[AxisSupport]
    ) -> ImplicitAxisKernel:
        n_in = int(n_in)
        supports_tuple = tuple(supports)
        if n_in <= 0 or not supports_tuple:
            raise FullResizeKernelError("axis geometry must be nonempty")
        claimed: set[int] = set()
        for support in supports_tuple:
            if len(support.indices) != 2 or len(support.numerators) != 2:
                raise FullResizeKernelError(
                    "full-kernel compiler requires disjoint two-tap supports"
                )
            if claimed.intersection(support.indices):
                raise FullResizeKernelError("axis supports must be disjoint")
            claimed.update(support.indices)
        if min(claimed) < 0 or max(claimed) >= n_in:
            raise FullResizeKernelError("axis support index is out of bounds")
        unowned = tuple(index for index in range(n_in) if index not in claimed)
        return cls(n_in, len(supports_tuple), supports_tuple, unowned)

    @property
    def nullity(self) -> int:
        return self.n_in - self.n_out

    @property
    def local_atom_count(self) -> int:
        return self.n_out

    def primitive_null_atoms(self) -> tuple[tuple[int, int], ...]:
        """Primitive integer atom values at each support's two indices."""

        atoms: list[tuple[int, int]] = []
        for support in self.supports:
            a, b = support.numerators
            b_primitive, a_primitive = _primitive_pair(b, a)
            atoms.append((b_primitive, -a_primitive))
        return tuple(atoms)

    def _move_axis_to_front(self, value: np.ndarray, axis: int) -> np.ndarray:
        raw = np.asarray(value)
        normalized_axis = np.core.numeric.normalize_axis_index(axis, raw.ndim)
        if raw.shape[normalized_axis] != self.n_in:
            raise FullResizeKernelError(
                f"axis length {raw.shape[normalized_axis]} != expected {self.n_in}"
            )
        return np.moveaxis(raw, normalized_axis, 0)

    def project_row_space(
        self,
        value: np.ndarray,
        *,
        axis: int = 0,
        dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
    ) -> np.ndarray:
        """Apply ``Q=A.T(AA.T)^-1A`` along one array axis."""

        target_dtype = np.dtype(dtype)
        if target_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise FullResizeKernelError("projector dtype must be float32 or float64")
        x = _as_float_array(value, target_dtype)
        normalized_axis = np.core.numeric.normalize_axis_index(axis, x.ndim)
        front = self._move_axis_to_front(x, normalized_axis)
        indices = np.asarray([support.indices for support in self.supports], dtype=np.intp)
        weights = np.asarray(
            [support.weights for support in self.supports], dtype=target_dtype
        )
        blocks = front[indices]
        expand = (slice(None), slice(None)) + (None,) * (blocks.ndim - 2)
        weighted = weights[expand]
        numerator = np.sum(weighted * blocks, axis=1, dtype=target_dtype)
        denominator = np.sum(weights * weights, axis=1, dtype=target_dtype)
        denom_expand = (slice(None),) + (None,) * (numerator.ndim - 1)
        coefficient = numerator / denominator[denom_expand]
        projected_blocks = weighted * coefficient[:, None, ...]
        out = np.zeros_like(front, dtype=target_dtype)
        out[indices[:, 0]] = projected_blocks[:, 0]
        out[indices[:, 1]] = projected_blocks[:, 1]
        return np.moveaxis(out, 0, normalized_axis)

    def project_kernel(
        self,
        value: np.ndarray,
        *,
        axis: int = 0,
        dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
    ) -> np.ndarray:
        x = _as_float_array(value, np.dtype(dtype))
        return x - self.project_row_space(x, axis=axis, dtype=dtype)

    def synthesize_kernel(self, coefficients: np.ndarray, *, axis: int = 0) -> np.ndarray:
        """Apply the implicit exact basis ``N`` along ``axis``."""

        raw = np.asarray(coefficients)
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            raise FullResizeKernelError("kernel coefficients must be real numeric")
        normalized_axis = np.core.numeric.normalize_axis_index(axis, raw.ndim)
        if raw.shape[normalized_axis] != self.nullity:
            raise FullResizeKernelError(
                f"kernel coefficient axis {raw.shape[normalized_axis]} != {self.nullity}"
            )
        front = np.moveaxis(raw, normalized_axis, 0)
        output_dtype = (
            raw.dtype if raw.dtype.kind == "f" else np.result_type(raw.dtype, np.int64)
        )
        out = np.zeros((self.n_in, *front.shape[1:]), dtype=output_dtype)
        atoms = self.primitive_null_atoms()
        for support_index, (support, atom) in enumerate(
            zip(self.supports, atoms, strict=True)
        ):
            coefficient = front[support_index].astype(output_dtype, copy=False)
            out[support.indices[0]] = coefficient * atom[0]
            out[support.indices[1]] = coefficient * atom[1]
        for offset, input_index in enumerate(self.unowned_indices, start=self.n_out):
            out[input_index] = front[offset].astype(output_dtype, copy=False)
        return np.moveaxis(out, 0, normalized_axis)

    def synthesize_row_space(
        self, coefficients: np.ndarray, *, axis: int = 0
    ) -> np.ndarray:
        """Apply ``A.T`` as an implicit row-space basis along ``axis``."""

        raw = np.asarray(coefficients)
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            raise FullResizeKernelError("row-space coefficients must be real numeric")
        normalized_axis = np.core.numeric.normalize_axis_index(axis, raw.ndim)
        if raw.shape[normalized_axis] != self.n_out:
            raise FullResizeKernelError(
                f"row-space coefficient axis {raw.shape[normalized_axis]} != {self.n_out}"
            )
        front = np.moveaxis(raw, normalized_axis, 0)
        output_dtype = (
            raw.dtype
            if raw.dtype in (np.dtype(np.float32), np.dtype(np.float64))
            else np.dtype(np.float64)
        )
        out = np.zeros((self.n_in, *front.shape[1:]), dtype=output_dtype)
        for support_index, support in enumerate(self.supports):
            coefficient = front[support_index].astype(output_dtype, copy=False)
            out[support.indices[0]] = coefficient * support.weights[0]
            out[support.indices[1]] = coefficient * support.weights[1]
        return np.moveaxis(out, 0, normalized_axis)


@dataclass(frozen=True)
class KernelCoverage:
    height_input_dimension: int
    width_input_dimension: int
    domain_dimension: int
    resize_rank: int
    full_nullity: int
    old_zero_weight_nullity: int
    added_full_kernel_dimensions: int
    height_axis_nullity: int
    width_axis_nullity: int
    left_tensor_dimensions: int
    right_tensor_dimensions: int

    def to_dict(self) -> dict[str, Any]:
        domain = self.domain_dimension
        return {
            "domain_dimension_per_channel": domain,
            "resize_rank_per_channel": self.resize_rank,
            "full_nullity_per_channel": self.full_nullity,
            "full_nullity_fraction": self.full_nullity / domain,
            "full_nullity_percent": 100.0 * self.full_nullity / domain,
            "old_zero_weight_nullity_per_channel": self.old_zero_weight_nullity,
            "old_zero_weight_fraction": self.old_zero_weight_nullity / domain,
            "old_zero_weight_percent": 100.0 * self.old_zero_weight_nullity / domain,
            "added_full_kernel_dimensions_per_channel": (
                self.added_full_kernel_dimensions
            ),
            "added_coverage_percentage_points": (
                100.0 * self.added_full_kernel_dimensions / domain
            ),
            "height_axis_nullity": self.height_axis_nullity,
            "height_axis_nullity_fraction": (
                self.height_axis_nullity / self.height_input_dimension
            ),
            "height_axis_nullity_percent": (
                100.0 * self.height_axis_nullity / self.height_input_dimension
            ),
            "width_axis_nullity": self.width_axis_nullity,
            "width_axis_nullity_fraction": (
                self.width_axis_nullity / self.width_input_dimension
            ),
            "width_axis_nullity_percent": (
                100.0 * self.width_axis_nullity / self.width_input_dimension
            ),
            "left_tensor_dimensions": self.left_tensor_dimensions,
            "right_tensor_dimensions": self.right_tensor_dimensions,
            "identity_check": (
                self.left_tensor_dimensions + self.right_tensor_dimensions
                == self.full_nullity
            ),
        }


@dataclass(frozen=True)
class Uint8Reachability:
    channels: int
    zero_weight_coordinate_directions: int
    active_tensor_directions: int
    feasible_height_col0_directions: int
    feasible_height_col1_directions: int
    feasible_width_tensor_directions: int
    active_cell_channel_rank_histogram: tuple[int, int, int, int]

    @property
    def feasible_active_directions(self) -> int:
        return (
            self.feasible_height_col0_directions
            + self.feasible_height_col1_directions
            + self.feasible_width_tensor_directions
        )

    @property
    def full_basis_directions(self) -> int:
        return self.zero_weight_coordinate_directions + self.active_tensor_directions

    @property
    def feasible_basis_directions_lower_bound(self) -> int:
        return self.zero_weight_coordinate_directions + self.feasible_active_directions

    def to_dict(self) -> dict[str, Any]:
        total = self.full_basis_directions
        return {
            "semantics": UINT8_REACHABILITY_SEMANTICS,
            "is_lower_bound_on_full_bounded_lattice_intersection": True,
            "channels": self.channels,
            "zero_weight_coordinate_directions": self.zero_weight_coordinate_directions,
            "active_tensor_directions": self.active_tensor_directions,
            "feasible_height_col0_directions": self.feasible_height_col0_directions,
            "feasible_height_col1_directions": self.feasible_height_col1_directions,
            "feasible_width_tensor_directions": self.feasible_width_tensor_directions,
            "feasible_active_directions": self.feasible_active_directions,
            "full_basis_directions": total,
            "feasible_basis_directions_lower_bound": (
                self.feasible_basis_directions_lower_bound
            ),
            "feasible_basis_fraction_lower_bound": (
                self.feasible_basis_directions_lower_bound / total if total else 0.0
            ),
            "feasible_basis_percent_lower_bound": (
                100.0 * self.feasible_basis_directions_lower_bound / total
                if total
                else 0.0
            ),
            "active_cell_channel_rank_histogram": {
                str(rank): int(count)
                for rank, count in enumerate(self.active_cell_channel_rank_histogram)
            },
        }


@dataclass(frozen=True)
class CandidateSolveDiagnostics:
    preference: str
    exact_blocks: int
    fallback_blocks: int
    budget_blocks: int
    proven_infeasible_blocks: int
    nodes_visited: int
    exact_numerator_equal: bool
    max_float_projection_residual: float
    bytes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference": self.preference,
            "exact_blocks": self.exact_blocks,
            "fallback_blocks": self.fallback_blocks,
            "budget_blocks": self.budget_blocks,
            "proven_infeasible_blocks": self.proven_infeasible_blocks,
            "nodes_visited": self.nodes_visited,
            "exact_numerator_equal": self.exact_numerator_equal,
            "max_float_projection_residual": self.max_float_projection_residual,
            "bytes": dict(self.bytes),
        }


@dataclass(frozen=True)
class FullKernelFillResult:
    frame: np.ndarray
    old_mask_frame: np.ndarray
    selected_name: str
    original_bytes: dict[str, int]
    old_mask_bytes: dict[str, int]
    selected_bytes: dict[str, int]
    candidates: tuple[CandidateSolveDiagnostics, ...]

    @property
    def selected_uses_full_kernel(self) -> bool:
        return self.selected_name != "old_zero_weight_mask"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FULL_RESIZE_KERNEL_SCHEMA,
            "selected_name": self.selected_name,
            "selected_uses_full_kernel": self.selected_uses_full_kernel,
            "original_bytes": dict(self.original_bytes),
            "old_mask_bytes": dict(self.old_mask_bytes),
            "selected_bytes": dict(self.selected_bytes),
            "delta_selected_vs_old_mask": {
                coder: self.selected_bytes[coder] - self.old_mask_bytes[coder]
                for coder in self.old_mask_bytes
            },
            "delta_selected_vs_original": {
                coder: self.selected_bytes[coder] - self.original_bytes[coder]
                for coder in self.original_bytes
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "global_mdl_optimum_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
        }


Preference = Literal["constant", "horizontal", "vertical", "neighbor_mean"]


@dataclass(frozen=True)
class FullResizeKernel:
    """Complete implicit kernel/compiler for one separable resize geometry."""

    operator: DisjointResizeOperator
    height: ImplicitAxisKernel
    width: ImplicitAxisKernel

    @classmethod
    def build(
        cls,
        *,
        camera_h: int = CAMERA_H,
        camera_w: int = CAMERA_W,
        scorer_h: int = SCORER_INPUT_H,
        scorer_w: int = SCORER_INPUT_W,
    ) -> FullResizeKernel:
        operator = DisjointResizeOperator.build(
            camera_h=camera_h,
            camera_w=camera_w,
            scorer_h=scorer_h,
            scorer_w=scorer_w,
        )
        return cls(
            operator=operator,
            height=ImplicitAxisKernel.from_supports(camera_h, operator.row_supports),
            width=ImplicitAxisKernel.from_supports(camera_w, operator.col_supports),
        )

    @property
    def camera_h(self) -> int:
        return self.operator.camera_h

    @property
    def camera_w(self) -> int:
        return self.operator.camera_w

    @property
    def scorer_h(self) -> int:
        return self.operator.scorer_h

    @property
    def scorer_w(self) -> int:
        return self.operator.scorer_w

    def project_range(
        self,
        value: np.ndarray,
        *,
        dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
    ) -> np.ndarray:
        x = _as_float_array(value, np.dtype(dtype))
        if x.ndim not in (2, 3) or x.shape[:2] != (self.camera_h, self.camera_w):
            raise FullResizeKernelError("image must have shape (camera_h,camera_w[,C])")
        q_height = self.height.project_row_space(x, axis=0, dtype=dtype)
        return self.width.project_row_space(q_height, axis=1, dtype=dtype)

    def project_kernel(
        self,
        value: np.ndarray,
        *,
        dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
    ) -> np.ndarray:
        x = _as_float_array(value, np.dtype(dtype))
        return x - self.project_range(x, dtype=dtype)

    def synthesize(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Synthesize ``N_h U + A_h.T V N_w.T`` without a dense basis."""

        left_raw = np.asarray(left)
        right_raw = np.asarray(right)
        if left_raw.ndim not in (2, 3) or right_raw.ndim != left_raw.ndim:
            raise FullResizeKernelError("left/right coefficients must both be 2D or 3D")
        if left_raw.shape[0:2] != (self.height.nullity, self.camera_w):
            raise FullResizeKernelError(
                f"left shape {left_raw.shape[:2]} != {(self.height.nullity, self.camera_w)}"
            )
        if right_raw.shape[0:2] != (self.scorer_h, self.width.nullity):
            raise FullResizeKernelError(
                f"right shape {right_raw.shape[:2]} != {(self.scorer_h, self.width.nullity)}"
            )
        if left_raw.ndim == 3 and left_raw.shape[2:] != right_raw.shape[2:]:
            raise FullResizeKernelError("left/right channel shapes differ")
        dtype = np.result_type(left_raw.dtype, right_raw.dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            dtype = np.dtype(np.float64)
        left_term = self.height.synthesize_kernel(left_raw.astype(dtype), axis=0)
        width_term = self.width.synthesize_kernel(right_raw.astype(dtype), axis=1)
        right_term = self.height.synthesize_row_space(width_term, axis=0)
        return left_term + right_term

    def coverage(self) -> KernelCoverage:
        domain = self.camera_h * self.camera_w
        rank = self.scorer_h * self.scorer_w
        full = domain - rank
        old = (
            len(self.height.unowned_indices) * self.camera_w
            + len(self.width.unowned_indices) * self.camera_h
            - len(self.height.unowned_indices) * len(self.width.unowned_indices)
        )
        left = self.height.nullity * self.camera_w
        right = self.scorer_h * self.width.nullity
        if left + right != full:
            raise FullResizeKernelError("tensor parameterization dimension mismatch")
        return KernelCoverage(
            self.camera_h,
            self.camera_w,
            domain,
            rank,
            full,
            old,
            full - old,
            self.height.nullity,
            self.width.nullity,
            left,
            right,
        )

    @staticmethod
    def _move_fits(block: np.ndarray, atom: np.ndarray) -> np.ndarray:
        plus = np.all((block + atom >= 0) & (block + atom <= 255), axis=-1)
        minus = np.all((block - atom >= 0) & (block - atom <= 255), axis=-1)
        return plus | minus

    def uint8_reachability(self, frame: np.ndarray) -> Uint8Reachability:
        """Measure bounded reachability of the canonical tensor primitive basis.

        This is an exact lower bound for the chosen nonredundant primitive
        basis, not an equality claim for every integer combination in the
        bounded lattice intersection.
        """

        x = np.asarray(frame)
        if x.dtype != np.uint8 or x.ndim != 3 or x.shape[:2] != (
            self.camera_h,
            self.camera_w,
        ):
            raise FullResizeKernelError("reachability frame must be camera-shaped uint8 HWC")
        row_indices = np.asarray(
            [support.indices for support in self.operator.row_supports], dtype=np.intp
        )
        col_indices = np.asarray(
            [support.indices for support in self.operator.col_supports], dtype=np.intp
        )
        blocks = x[
            row_indices[:, None, :, None],
            col_indices[None, :, None, :],
            :,
        ].astype(np.int64)
        # (h,w,2,2,C) -> (h,w,C,4), order 00,01,10,11.
        blocks_flat = np.moveaxis(blocks.reshape(self.scorer_h, self.scorer_w, 4, -1), 2, -1)

        row_null = np.asarray(self.height.primitive_null_atoms(), dtype=np.int64)
        row_space = np.asarray(
            [_primitive_pair(*support.numerators) for support in self.operator.row_supports],
            dtype=np.int64,
        )
        col_null = np.asarray(self.width.primitive_null_atoms(), dtype=np.int64)

        atom_h0 = np.zeros((self.scorer_h, 4), dtype=np.int64)
        atom_h0[:, 0] = row_null[:, 0]
        atom_h0[:, 2] = row_null[:, 1]
        atom_h1 = np.zeros((self.scorer_h, 4), dtype=np.int64)
        atom_h1[:, 1] = row_null[:, 0]
        atom_h1[:, 3] = row_null[:, 1]
        atom_w = (
            row_space[:, None, :, None] * col_null[None, :, None, :]
        ).reshape(self.scorer_h, self.scorer_w, 4)

        feasible_h0 = self._move_fits(blocks_flat, atom_h0[:, None, None, :])
        feasible_h1 = self._move_fits(blocks_flat, atom_h1[:, None, None, :])
        feasible_w = self._move_fits(blocks_flat, atom_w[:, :, None, :])
        ranks = (
            feasible_h0.astype(np.uint8)
            + feasible_h1.astype(np.uint8)
            + feasible_w.astype(np.uint8)
        )
        histogram = tuple(int(np.count_nonzero(ranks == rank)) for rank in range(4))
        channels = x.shape[2]
        zero_directions = self.coverage().old_zero_weight_nullity * channels
        active_directions = 3 * self.scorer_h * self.scorer_w * channels
        return Uint8Reachability(
            channels=channels,
            zero_weight_coordinate_directions=zero_directions,
            active_tensor_directions=active_directions,
            feasible_height_col0_directions=int(np.count_nonzero(feasible_h0)),
            feasible_height_col1_directions=int(np.count_nonzero(feasible_h1)),
            feasible_width_tensor_directions=int(np.count_nonzero(feasible_w)),
            active_cell_channel_rank_histogram=histogram,
        )

    @staticmethod
    def _preference(frame: np.ndarray, preference: Preference) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64)
        if preference == "constant":
            value = np.median(x, axis=(0, 1), keepdims=True)
            return np.broadcast_to(value, x.shape).copy()
        if preference == "horizontal":
            return np.concatenate((x[:, :1], x[:, :-1]), axis=1)
        if preference == "vertical":
            return np.concatenate((x[:1], x[:-1]), axis=0)
        if preference == "neighbor_mean":
            return (
                np.roll(x, 1, axis=0)
                + np.roll(x, -1, axis=0)
                + np.roll(x, 1, axis=1)
                + np.roll(x, -1, axis=1)
            ) / 4.0
        raise FullResizeKernelError(f"unknown preference {preference!r}")

    def _solve_candidate(
        self,
        source: np.ndarray,
        baseline: np.ndarray,
        *,
        preference: Preference,
        max_nodes_per_block: int,
    ) -> tuple[np.ndarray, CandidateSolveDiagnostics]:
        target_numerators, denominator = self.operator.apply_numerators(source)
        preferred = self._preference(baseline, preference)
        out = baseline.copy()
        exact_blocks = budget_blocks = infeasible_blocks = nodes = 0
        for row_out, row_support in enumerate(self.operator.row_supports):
            for col_out, col_support in enumerate(self.operator.col_supports):
                coefficients = tuple(
                    int(value)
                    for value in np.outer(
                        row_support.numerators, col_support.numerators
                    ).reshape(-1)
                )
                index = np.ix_(row_support.indices, col_support.indices, range(source.shape[2]))
                preferred_block = preferred[index]
                for channel in range(source.shape[2]):
                    target_integer = int(target_numerators[row_out, col_out, channel])
                    result = solve_bounded_integer_block(
                        coefficients,
                        denominator,
                        target_integer / denominator,
                        target_integer=target_integer,
                        preferred=preferred_block[:, :, channel].reshape(-1),
                        max_nodes=max_nodes_per_block,
                    )
                    nodes += result.nodes_visited
                    if result.status == BlockSolveStatus.FEASIBLE_EXACT:
                        out[
                            np.ix_(
                                row_support.indices,
                                col_support.indices,
                                (channel,),
                            )
                        ] = np.asarray(result.values, dtype=np.uint8).reshape(2, 2, 1)
                        exact_blocks += 1
                    elif result.status == BlockSolveStatus.NOT_FOUND_BUDGET:
                        budget_blocks += 1
                    elif result.status == BlockSolveStatus.INFEASIBLE_EXHAUSTIVE:
                        infeasible_blocks += 1
        realized_numerators, realized_denominator = self.operator.apply_numerators(out)
        numerator_equal = bool(
            denominator == realized_denominator
            and np.array_equal(realized_numerators, target_numerators)
        )
        if not numerator_equal:
            raise FullResizeKernelError("candidate failed exact numerator verification")
        float_residual = float(
            np.max(np.abs(self.operator.apply(out) - self.operator.apply(source)))
        )
        exact_total = self.scorer_h * self.scorer_w * source.shape[2]
        diagnostics = CandidateSolveDiagnostics(
            preference=preference,
            exact_blocks=exact_blocks,
            fallback_blocks=exact_total - exact_blocks,
            budget_blocks=budget_blocks,
            proven_infeasible_blocks=infeasible_blocks,
            nodes_visited=nodes,
            exact_numerator_equal=True,
            max_float_projection_residual=float_residual,
            bytes=coded_size_both(out),
        )
        return out, diagnostics

    def compile_min_description_preimage(
        self,
        frame: np.ndarray,
        *,
        preferences: Sequence[Preference] = ("constant",),
        max_nodes_per_block: int = 128,
    ) -> FullKernelFillResult:
        """Coder-admitted exact uint8 fill over full affine resize cells.

        This is a deterministic bounded heuristic, not a global MDL optimum.
        The legacy #49 measured-best mask fill is always included and wins on
        ties or whenever every full-kernel candidate is larger.
        """

        source = np.asarray(frame)
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[:2] != (
            self.camera_h,
            self.camera_w,
        ):
            raise FullResizeKernelError("compiler frame must be camera-shaped uint8 HWC")
        if max_nodes_per_block < 1:
            raise FullResizeKernelError("max_nodes_per_block must be positive")
        preferences_tuple = tuple(preferences)
        if not preferences_tuple:
            raise FullResizeKernelError("at least one full-kernel preference is required")
        projector = ResizeProjector.build(
            camera_h=self.camera_h,
            camera_w=self.camera_w,
            scorer_h=self.scorer_h,
            scorer_w=self.scorer_w,
        )
        old_mask, _proof = apply_tier1_zero_weight_fill(
            source, projector=projector, strategy="measured_best"
        )
        original_bytes = coded_size_both(source)
        old_bytes = coded_size_both(old_mask)
        candidates: list[CandidateSolveDiagnostics] = []
        frames: dict[str, np.ndarray] = {"old_zero_weight_mask": old_mask}
        sizes: dict[str, dict[str, int]] = {"old_zero_weight_mask": old_bytes}
        for preference in preferences_tuple:
            candidate, diagnostics = self._solve_candidate(
                source,
                old_mask,
                preference=preference,
                max_nodes_per_block=max_nodes_per_block,
            )
            name = f"full_kernel_{preference}"
            candidates.append(diagnostics)
            frames[name] = candidate
            sizes[name] = diagnostics.bytes
        selected_name = _select_coder_admitted_name(
            sizes,
            baseline_name="old_zero_weight_mask",
        )
        selected = frames[selected_name]
        return FullKernelFillResult(
            frame=selected,
            old_mask_frame=old_mask,
            selected_name=selected_name,
            original_bytes=original_bytes,
            old_mask_bytes=old_bytes,
            selected_bytes=sizes[selected_name],
            candidates=tuple(candidates),
        )


__all__ = [
    "FULL_RESIZE_KERNEL_SCHEMA",
    "UINT8_REACHABILITY_SEMANTICS",
    "CandidateSolveDiagnostics",
    "FullKernelFillResult",
    "FullResizeKernel",
    "FullResizeKernelError",
    "ImplicitAxisKernel",
    "KernelCoverage",
    "Uint8Reachability",
]
