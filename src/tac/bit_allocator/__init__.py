# SPDX-License-Identifier: MIT
"""``tac.bit_allocator`` — canonical bit-allocator primitives.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #125
hook #3 (bit-allocator). This package hosts deterministic, observability-only
bit-allocation helpers that consume sensitivity / difficulty signals from
upstream consumers and emit canonical bit-allocation manifests with
Provenance per Catalog #323.

Five orthogonal allocator surfaces ship in this package:

* ``allocate_bits_per_pair``  (per-pair difficulty-weighted)
* ``allocate_per_byte``       (per-byte top-K + uniform baseline)
* ``allocate_per_class``      (per-SegNet-class prior-weighted)
* ``allocate_per_axis``       (per seg/pose/rate axis-weighted)
* ``allocate_via_lagrangian_dual`` (Pareto-feasibility KKT bisection /
  Dykstra alternating projection)

Per Dykstra's co-lead position on the inner council quintet pact (CLAUDE.md
"Council conduct"), the Lagrangian-dual + Dykstra-projection paths in
:mod:`tac.bit_allocator.pareto_dual` realize the canonical Pareto-feasibility
arbitration for multi-constraint bit-allocation problems.

Bit-allocator outputs are predicted, observability-only signals — they do
NOT make score claims. The downstream packet compiler / canonical
deterministic compiler (Catalog #158) is responsible for realizing the
allocation as actual archive bytes, and only paired-axis Linux x86_64
auth-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" can
promote an allocation to a contest-grade score claim.

Quick start — per-pair difficulty:

    from tac.bit_allocator import allocate_bits_per_pair, AllocationStrategy

    allocation = allocate_bits_per_pair(
        total_bits=1024,
        difficulty_per_pair={0: 1.0, 1: 4.0, 2: 0.5},
        strategy=AllocationStrategy.SQRT,
    )

Quick start — per-byte top-K (canonical equation
``per_byte_leverage_uniformly_distributed_v1``):

    from tac.bit_allocator import allocate_per_byte, PerByteAllocationMethod

    plan = allocate_per_byte(
        total_budget_bits=128,
        sensitivity_per_byte={0: 5.0, 1: 0.1, 2: 9.0, 3: 0.5},
        method=PerByteAllocationMethod.TOP_K_BY_SENSITIVITY,
        top_k=2,
    )

Quick start — per-SegNet-class (canonical 5-class taxonomy):

    from tac.bit_allocator import allocate_per_class, PerClassAllocationStrategy

    plan = allocate_per_class(
        total_budget_bits=500,
        prior_per_class={0: 1.0, 1: 2.0, 2: 1.5, 3: 3.0, 4: 0.5},
        strategy=PerClassAllocationStrategy.SQRT,
    )

Quick start — per-axis seg/pose/rate:

    from tac.bit_allocator import allocate_per_axis, PerAxisAllocationStrategy

    plan = allocate_per_axis(
        total_budget_bits=300,
        strategy=PerAxisAllocationStrategy.SENSITIVITY_WEIGHTED,
        sensitivity_per_axis={"seg": 100.0, "pose": 271.0, "rate": 25.0},
    )

Quick start — Lagrangian-dual Pareto allocation:

    from tac.bit_allocator import allocate_via_lagrangian_dual, ParetoDualMethod

    plan = allocate_via_lagrangian_dual(
        total_budget_bits=100,
        sensitivity_per_element={0: 5.0, 1: 0.1, 2: 9.0, 3: 0.5},
        min_bits=0,
        max_bits=8,
    )
    # plan.is_pareto_feasible: True; plan.kkt_residual: ~0.0
    # plan.lagrangian_lambda: KKT dual variable

Hooks per Catalog #125 (across all 5 allocators):

* hook #1 SENSITIVITY_MAP: consumes per-element sensitivity / difficulty
  / priority priors
* hook #2 PARETO_CONSTRAINT: total_bits is the primary Pareto constraint;
  Lagrangian dual / Dykstra projection are the canonical KKT-derived
  solvers
* hook #3 BIT_ALLOCATOR (PRIMARY): this package IS the canonical
  bit-allocator surface
* hook #4 CATHEDRAL_AUTOPILOT_DISPATCH: allocator outputs are consumable
  by future cathedral consumers; ``per_byte_sensitivity_consumer`` +
  ``per_pair_difficulty_atlas_consumer`` are the canonical inputs
* hook #5 CONTINUAL_LEARNING_POSTERIOR: canonical equation
  ``per_byte_leverage_uniformly_distributed_v1`` recalibrates on new
  empirical anchors per :mod:`tac.canonical_equations`
* hook #6 PROBE_DISAMBIGUATOR: each allocator's strategy/method enum is
  the canonical probe-disambiguator surface for empirical comparison

Canonical-vs-unique decision per layer (CLAUDE.md Catalog #290):

* per-pair difficulty (UNIFORM / LINEAR / SQRT) — adopted canonical
  ``tac.bit_allocator.per_pair_difficulty_weighted`` from prior subagent
  landing (sister-extend; reused).
* per-byte (TOP_K / UNIFORM) — canonically implemented in
  :mod:`tac.bit_allocator.per_byte`; cites canonical equation
  ``per_byte_leverage_uniformly_distributed_v1`` per
  :mod:`tac.canonical_equations`.
* per-class (UNIFORM / PROPORTIONAL / SQRT) — canonically implemented in
  :mod:`tac.bit_allocator.per_class`; NSCS06 v6→v7 unwind anchor.
* per-axis (UNIFORM / SENSITIVITY_WEIGHTED / SCORER_FORMULA_WEIGHTED) —
  canonically implemented in :mod:`tac.bit_allocator.per_axis`; ties to
  CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent".
* pareto_dual (LAGRANGIAN_DUAL / DYKSTRA_PROJECTION) — canonically
  implemented in :mod:`tac.bit_allocator.pareto_dual`; cites
  :mod:`tac.findings_lagrangian` at META Lagrangian level (observability-
  only at this phase; downstream meta-Lagrangian-wire-in is a sister wave).

Lane Ω water-fill per-weight Hessian importance allocator is re-exported
from :mod:`tac.bit_allocator.lane_omega` (originally at the legacy module
location ``src/tac/bit_allocator.py``; renamed by WAVE-3-FORENSIC-FIX-2
2026-05-20 to extinct the package-shadows-legacy-module bug class). The
canonical caller API ``from tac.bit_allocator import allocate_bits`` /
``allocation_report`` / ``DEFAULT_ALPHA`` / ``DEFAULT_MIN_BITS`` /
``DEFAULT_MAX_BITS`` continues to work without modification.

Quick start — Lane Ω water-fill per-weight Fisher importance:

    from tac.bit_allocator import allocate_bits, allocation_report

    bits = allocate_bits(
        importance={"l.weight": torch.tensor([0.1, 1.0, 10.0])},
        total_bits=20,
        alpha=0.5,
        min_bits=1,
        max_bits=8,
    )
    report = allocation_report(bits)
"""
from __future__ import annotations

from tac.bit_allocator.lane_omega import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_BITS,
    DEFAULT_MIN_BITS,
    allocate_bits,
    allocation_report,
)
from tac.bit_allocator.per_axis import (
    CANONICAL_MODEL_ID_PER_AXIS,
    CANONICAL_SCORER_AXES,
    CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED,
    PerAxisAllocationError,
    PerAxisAllocationStrategy,
    PerAxisBitAllocationPlan,
    allocate_per_axis,
)
from tac.bit_allocator.per_byte import (
    CANONICAL_EQUATION_ID as PER_BYTE_CANONICAL_EQUATION_ID,
    CANONICAL_MODEL_ID_PER_BYTE,
    PerByteAllocationError,
    PerByteAllocationMethod,
    PerByteAllocationPlan,
    allocate_per_byte,
)
from tac.bit_allocator.per_class import (
    CANONICAL_MODEL_ID_PER_CLASS,
    CANONICAL_SEGNET_CLASS_NAMES,
    PerClassAllocationError,
    PerClassAllocationStrategy,
    PerClassBitAllocationPlan,
    SEGNET_CLASS_COUNT,
    allocate_per_class,
)
from tac.bit_allocator.pareto_dual import (
    CANONICAL_MODEL_ID_PARETO_DUAL,
    DEFAULT_BISECTION_ITERS,
    ParetoDualBitAllocationPlan,
    ParetoDualError,
    ParetoDualMethod,
    allocate_via_lagrangian_dual,
)
from tac.bit_allocator.per_pair_difficulty_weighted import (
    AllocationStrategy,
    BitAllocationResult,
    BitAllocationStrategyError,
    allocate_bits_per_pair,
)

__all__ = (
    # lane_omega (Lane Ω per-weight Fisher water-fill; renamed from legacy
    # src/tac/bit_allocator.py by WAVE-3-FORENSIC-FIX-2 2026-05-20)
    "DEFAULT_ALPHA",
    "DEFAULT_MAX_BITS",
    "DEFAULT_MIN_BITS",
    "allocate_bits",
    "allocation_report",
    # per_pair (legacy sister; existing canonical landing)
    "AllocationStrategy",
    "BitAllocationResult",
    "BitAllocationStrategyError",
    "allocate_bits_per_pair",
    # per_byte
    "CANONICAL_MODEL_ID_PER_BYTE",
    "PER_BYTE_CANONICAL_EQUATION_ID",
    "PerByteAllocationError",
    "PerByteAllocationMethod",
    "PerByteAllocationPlan",
    "allocate_per_byte",
    # per_class
    "CANONICAL_MODEL_ID_PER_CLASS",
    "CANONICAL_SEGNET_CLASS_NAMES",
    "PerClassAllocationError",
    "PerClassAllocationStrategy",
    "PerClassBitAllocationPlan",
    "SEGNET_CLASS_COUNT",
    "allocate_per_class",
    # per_axis
    "CANONICAL_MODEL_ID_PER_AXIS",
    "CANONICAL_SCORER_AXES",
    "CANONICAL_SCORER_COEFFICIENTS_FORMULA_WEIGHTED",
    "PerAxisAllocationError",
    "PerAxisAllocationStrategy",
    "PerAxisBitAllocationPlan",
    "allocate_per_axis",
    # pareto_dual
    "CANONICAL_MODEL_ID_PARETO_DUAL",
    "DEFAULT_BISECTION_ITERS",
    "ParetoDualBitAllocationPlan",
    "ParetoDualError",
    "ParetoDualMethod",
    "allocate_via_lagrangian_dual",
)
