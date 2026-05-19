# SPDX-License-Identifier: MIT
"""Impl 10 -- 588M-cell sparse water-filling decomposition.

Extends ``tac.optimization.bit_allocator_end_to_end`` to 588M-cell (per-pixel
* per-pair * per-class) sparse water-filling per CLAUDE.md "Meta-Lagrangian/
Pareto solver" non-negotiable.

The contest scoring surface is 588M-dim convex in rate (588_824_000 =
117_964_800 * 5; per ``constants.CONTEST_PER_ARCHIVE_PER_CLASS_CELLS``).
Decomposable per cell via canonical Lagrangian dual: each cell gets bits
in proportion to its analytical per-cell utility ``dS/d(cell_quality)``.

Since the surface is SPARSE (most cells contribute negligibly), the
588M-dim KKT closed-form reduces to a top-K sparse selection over the
empirically-non-negligible cells. ``tac.master_gradient`` provides the
per-cell sensitivity vectors; this module is the canonical allocator that
consumes them.

Citations:
  - Cover & Thomas 2006 Section 10.3 -- water-filling for parallel
    Gaussian channels (canonical algorithm).
  - Boyd & Vandenberghe 2004 Section 5.5 -- KKT in convex optimization.
  - ``tac.optimization.bit_allocator_end_to_end.EndToEndBitAllocator`` --
    canonical sister that allocates at the per-tensor granularity; this
    module extends to per-cell granularity.
  - ``tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual``
    -- canonical 600-pair sister; cell allocator is per-cell extension.

Catalog #125 hook 3 (bit_allocator): ACTIVE -- this IS a bit allocator.
Catalog #125 hook 1 (sensitivity_map): ACTIVE -- per-cell sensitivities
are the input to the allocator.
Catalog #305 observability surface: decomposable_per_signal, queryable_post_hoc,
cite_able.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .constants import CONTEST_PER_ARCHIVE_PER_CLASS_CELLS


class CellAllocatorError(ValueError):
    """Raised when allocator inputs are invalid."""


@dataclass(frozen=True, slots=True)
class CellAllocation:
    """One cell's bit allocation."""

    cell_index: int
    """Canonical 0-based cell index in the 588M-cell flat space."""

    per_cell_sensitivity: float
    """Analytical sensitivity ``dS/d(cell_quality)`` at this cell."""

    bits_allocated: int
    """Bits allocated to this cell."""


@dataclass(frozen=True, slots=True)
class SparseWaterFillingAllocation:
    """Sparse water-filling allocation over 588M-cell space."""

    total_bits_budget: int
    """Total bit budget across all cells."""

    total_cells_in_space: int
    """Total cells in the canonical space (= CONTEST_PER_ARCHIVE_PER_CLASS_CELLS)."""

    num_nonzero_cells: int
    """Number of cells with non-zero allocation (sparse)."""

    allocations: tuple[CellAllocation, ...]
    """Tuple of cell allocations, sorted by bits_allocated DESCENDING."""

    total_bits_allocated: int
    """Actual total bits allocated (may differ slightly from budget due to int rounding)."""

    sparsity_ratio: float
    """``num_nonzero_cells / total_cells_in_space``; near-1 indicates dense
    allocation, near-0 indicates sparse allocation (the canonical regime)."""


def sparse_water_fill(
    *,
    per_cell_sensitivity: Sequence[tuple[int, float]],
    total_bits_budget: int,
    min_bits_per_nonzero_cell: int = 1,
) -> SparseWaterFillingAllocation:
    """Closed-form sparse water-filling over per-cell sensitivities.

    The canonical water-filling allocates bits in proportion to per-cell
    sensitivity, with a noise-floor analog (``min_bits_per_nonzero_cell``)
    that determines the cutoff between "allocated" and "zero" cells.

    Args:
        per_cell_sensitivity: Sequence of ``(cell_index, sensitivity)`` tuples.
            Only cells with positive sensitivity are eligible for allocation;
            zero/negative are silently dropped (they have no marginal value).
        total_bits_budget: Total bits to distribute.
        min_bits_per_nonzero_cell: Floor on per-cell allocation. Cells whose
            proportional share would fall below this floor are dropped from
            allocation entirely (canonical sparsity-inducing cutoff).

    Returns:
        ``SparseWaterFillingAllocation`` with the allocation result.

    Raises:
        CellAllocatorError: if inputs are invalid.
    """
    if total_bits_budget < 0:
        raise CellAllocatorError(
            f"total_bits_budget must be >= 0 (got {total_bits_budget})"
        )
    if min_bits_per_nonzero_cell < 1:
        raise CellAllocatorError(
            f"min_bits_per_nonzero_cell must be >= 1 "
            f"(got {min_bits_per_nonzero_cell})"
        )

    # Filter to positive-sensitivity cells only.
    positive = [
        (idx, sens) for idx, sens in per_cell_sensitivity if sens > 0
    ]
    if not positive:
        return SparseWaterFillingAllocation(
            total_bits_budget=int(total_bits_budget),
            total_cells_in_space=CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
            num_nonzero_cells=0,
            allocations=(),
            total_bits_allocated=0,
            sparsity_ratio=0.0,
        )

    # Sort by sensitivity descending; canonical water-filling fills the
    # highest-sensitivity cells first.
    positive.sort(key=lambda x: x[1], reverse=True)

    # Sum of sensitivities for proportional allocation.
    total_sensitivity = sum(s for _, s in positive)

    # Greedy proportional allocation with the min-bits floor cutoff.
    # We allocate top-K cells where K is the largest count satisfying the
    # min_bits constraint.
    allocations: list[CellAllocation] = []
    remaining = total_bits_budget

    # First pass: determine cutoff K such that all K cells get >= min_bits.
    # K satisfies: per_cell_share_of_smallest >= min_bits, i.e.
    #   (sens_K / sum_top_K) * total_budget >= min_bits
    # We do a linear search since 588M is too large to brute-force; in practice
    # the sensitivity vector is sparse (~thousands of nonzero entries).
    for k in range(len(positive), 0, -1):
        top_k = positive[:k]
        sub_sum = sum(s for _, s in top_k)
        if sub_sum <= 0:
            continue
        # Smallest cell in top_k gets:
        smallest_sens = top_k[-1][1]
        smallest_alloc = int(
            (smallest_sens / sub_sum) * total_bits_budget
        )
        if smallest_alloc >= min_bits_per_nonzero_cell:
            chosen_k = k
            chosen_sub_sum = sub_sum
            break
    else:
        # Budget too tight for even top-1 to meet the floor; allocate min_bits
        # to top-1 only (if budget admits).
        if total_bits_budget >= min_bits_per_nonzero_cell:
            top_idx, top_sens = positive[0]
            allocations.append(CellAllocation(
                cell_index=int(top_idx),
                per_cell_sensitivity=float(top_sens),
                bits_allocated=int(min_bits_per_nonzero_cell),
            ))
            return SparseWaterFillingAllocation(
                total_bits_budget=int(total_bits_budget),
                total_cells_in_space=CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
                num_nonzero_cells=1,
                allocations=tuple(allocations),
                total_bits_allocated=int(min_bits_per_nonzero_cell),
                sparsity_ratio=1.0 / CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
            )
        return SparseWaterFillingAllocation(
            total_bits_budget=int(total_bits_budget),
            total_cells_in_space=CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
            num_nonzero_cells=0,
            allocations=(),
            total_bits_allocated=0,
            sparsity_ratio=0.0,
        )

    # Second pass: allocate proportionally to top-K.
    top_k = positive[:chosen_k]
    for idx, sens in top_k:
        alloc = int((sens / chosen_sub_sum) * total_bits_budget)
        alloc = max(alloc, min_bits_per_nonzero_cell)
        allocations.append(CellAllocation(
            cell_index=int(idx),
            per_cell_sensitivity=float(sens),
            bits_allocated=int(alloc),
        ))

    # Recompute total bits actually allocated (may differ from budget due to
    # int rounding + min-bits flooring).
    total_actually = sum(a.bits_allocated for a in allocations)

    return SparseWaterFillingAllocation(
        total_bits_budget=int(total_bits_budget),
        total_cells_in_space=CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
        num_nonzero_cells=len(allocations),
        allocations=tuple(allocations),
        total_bits_allocated=int(total_actually),
        sparsity_ratio=float(len(allocations)) / CONTEST_PER_ARCHIVE_PER_CLASS_CELLS,
    )


__all__ = [
    "CellAllocation",
    "CellAllocatorError",
    "SparseWaterFillingAllocation",
    "sparse_water_fill",
]
