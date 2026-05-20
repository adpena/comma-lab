# SPDX-License-Identifier: MIT
"""``tac.bit_allocator`` — canonical bit-allocator primitives.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #125
hook #3 (bit-allocator). This package hosts deterministic, observability-only
bit-allocation helpers that consume sensitivity / difficulty signals from
upstream consumers (e.g. ``tac.cathedral_consumers.per_pair_difficulty_atlas_consumer``)
and emit canonical bit-allocation manifests with Provenance per Catalog #323.

Bit-allocator outputs are predicted, observability-only signals — they do
NOT make score claims. The downstream packet compiler / canonical
deterministic compiler (Catalog #158) is responsible for realizing the
allocation as actual archive bytes, and only paired-axis Linux x86_64
auth-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" can
promote an allocation to a contest-grade score claim.

Quick start:

    from tac.bit_allocator import allocate_bits_per_pair, AllocationStrategy

    allocation = allocate_bits_per_pair(
        total_bits=1024,
        difficulty_per_pair={0: 1.0, 1: 4.0, 2: 0.5},
        strategy=AllocationStrategy.SQRT,
    )
    # allocation.bits_per_pair: {0: 290, 1: 580, 2: 154}
    # allocation.provenance.evidence_grade == PREDICTED
    # allocation.score_claim is False

Hooks per Catalog #125:
- #3 BIT_ALLOCATOR (PRIMARY): this is the canonical bit-allocator surface
- #1 SENSITIVITY_MAP: consumes per-pair difficulty signal
- #2 PARETO_CONSTRAINT: total_bits is the canonical Pareto constraint
- #6 PROBE_DISAMBIGUATOR: linear vs sqrt vs uniform strategies are the
  canonical probe-disambiguator surface for empirical comparison
"""
from __future__ import annotations

from tac.bit_allocator.per_pair_difficulty_weighted import (
    AllocationStrategy,
    BitAllocationResult,
    BitAllocationStrategyError,
    allocate_bits_per_pair,
)

__all__ = (
    "AllocationStrategy",
    "BitAllocationResult",
    "BitAllocationStrategyError",
    "allocate_bits_per_pair",
)
