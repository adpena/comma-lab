# SPDX-License-Identifier: MIT
"""tac.compress_time_optimization — decorator-based composable compress-time
pass namespace.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5 the canonical-helper namespace design + §G compress-time techniques.
Sister of ``tac.boosting`` (same decorator + frozen-dataclass + composition-
operator pattern at a different stage surface).

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  Decorator:
    - ``compress_time_pass(contract)``: register a pass's contract + function
    - ``CompressTimePassContract``: ~24-field frozen dataclass contract
    - ``CompressTimePassContractError`` / ``DeterminismViolation`` /
      ``SeedRequiredViolation`` / ``InflatePhaseForbiddenError``:
      decoration-time error classes

  Registry read API:
    - ``get_registered_passes()``: id → contract map
    - ``get_pass_function(pass_id)``: id → callable
    - ``validate_all_registered_passes()``: re-validate every contract

  Composition:
    - ``ComposableCompressPipeline``: pipe-operator pipeline builder
    - ``PipelineStageRef``: single pass reference
    - ``CompressTimePipelineResult``: run output

  Builders (one per §G compress-time row):
    - ``GenericTTOHarness`` + ``GenericTTOHarnessSpec``
    - ``MultipassRefinement`` + ``MultipassRefinementSpec``
    - ``SimulatedAnnealingOnDiscreteCodes`` + ``SimulatedAnnealingSpec``
    - ``PerPairCoordinateSearch`` + ``PerPairCoordinateSearchSpec``
    - ``IteratedBisectionRateKnee`` + ``IteratedBisectionRateKneeSpec``

  Persistence (opt-in):
    - ``append_pass_outcome_locked(record)``: fcntl-locked JSONL append
    - ``load_pass_outcomes`` / ``load_pass_outcomes_strict``: read API
    - ``CompressTimeLedgerCorruptError``: corruption signal (Catalog #138)
    - ``COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH`` /
      ``COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK``

  Errors:
    - ``CompressTimeOptimizationError``: root exception
    - ``CompressTimePipelineError`` / ``AmbiguousCompositionError`` /
      ``RateBudgetViolation`` / ``CompressTimeBudgetExceededError``:
      pipeline-build / run errors

Cross-references:
  - Design memo: `.omx/research/tac_compress_time_optimization_namespace_design_20260517.md`
  - Premise verification (Catalog #229): `.omx/tmp/tac_compress_time_optimization_premise_verifier.txt`
  - Sister namespace: ``tac.boosting``
  - Spec provenance: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
    §5.3-§5.5 + §G compress-time
  - Empirical anchor (TTO): ``experiments/optimize_poses.py`` (canonical TTO
    template that GenericTTOHarness generalizes)
  - Empirical anchor (multipass): ``src/tac/multipass_compressor.py`` (Lane 8
    substrate-specific compressor that MultipassRefinement generalizes)
"""

from __future__ import annotations

from tac.compress_time_optimization.contract import (
    LEGAL_CORRECTION_KIND,
    LEGAL_CORRECTION_RESOLUTION,
    LEGAL_HOOK_AUTOPILOT,
    LEGAL_HOOK_BIT_ALLOCATOR,
    LEGAL_HOOK_CONTINUAL_LEARNING,
    LEGAL_HOOK_PARETO,
    LEGAL_HOOK_SENSITIVITY,
    LEGAL_MERGE_POLICY,
    LEGAL_STAGE_PHASE,
    NOT_APPLICABLE_WITH_RATIONALE,
    CompressTimePassContract,
)
from tac.compress_time_optimization.decorator import (
    _clear_pass_registry_for_tests,
    _REGISTERED_PASSES,
    compress_time_pass,
    get_pass_function,
    get_registered_passes,
    validate_all_registered_passes,
)
from tac.compress_time_optimization.errors import (
    AmbiguousCompositionError,
    CompressTimeBudgetExceededError,
    CompressTimeLedgerCorruptError,
    CompressTimeOptimizationError,
    CompressTimePassContractError,
    CompressTimePipelineError,
    DeterminismViolation,
    InflatePhaseForbiddenError,
    RateBudgetViolation,
    SeedRequiredViolation,
)
from tac.compress_time_optimization.generic_tto_harness import (
    LEGAL_TTO_TARGET_KIND,
    GenericTTOHarness,
    GenericTTOHarnessSpec,
)
from tac.compress_time_optimization.iterated_bisection import (
    LEGAL_BISECTION_GRANULARITY,
    IteratedBisectionRateKnee,
    IteratedBisectionRateKneeSpec,
)
from tac.compress_time_optimization.multipass_refinement import (
    MultipassRefinement,
    MultipassRefinementSpec,
)
from tac.compress_time_optimization.per_pair_coordinate_search import (
    PerPairCoordinateSearch,
    PerPairCoordinateSearchSpec,
)
from tac.compress_time_optimization.persistence import (
    COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK,
    COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH,
    COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION,
    append_pass_outcome_locked,
    load_pass_outcomes,
    load_pass_outcomes_strict,
)
from tac.compress_time_optimization.pipeline import (
    CompressTimePipelineResult,
    ComposableCompressPipeline,
    PipelineStageRef,
)
from tac.compress_time_optimization.simulated_annealing import (
    LEGAL_SA_DISCRETE_TARGET,
    LEGAL_SA_TEMP_SCHEDULE,
    SimulatedAnnealingOnDiscreteCodes,
    SimulatedAnnealingSpec,
)
from tac.compress_time_optimization.per_pair_master_gradient_wire_in import (
    CompressTimeOptimizationPerPairWireInOutcome,
    compose_compress_time_optimization_per_pair_wire_in,
)

__all__ = [
    # Per-pair master gradient wire-in (LOW gap closure widened wave 2026-05-17)
    "CompressTimeOptimizationPerPairWireInOutcome",
    "compose_compress_time_optimization_per_pair_wire_in",
    # Decorator + registry
    "compress_time_pass",
    "get_registered_passes",
    "get_pass_function",
    "validate_all_registered_passes",
    "_REGISTERED_PASSES",
    "_clear_pass_registry_for_tests",
    # Contract + enums
    "CompressTimePassContract",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_STAGE_PHASE",
    "LEGAL_MERGE_POLICY",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    # Errors
    "CompressTimeOptimizationError",
    "CompressTimePassContractError",
    "DeterminismViolation",
    "SeedRequiredViolation",
    "InflatePhaseForbiddenError",
    "CompressTimePipelineError",
    "AmbiguousCompositionError",
    "RateBudgetViolation",
    "CompressTimeBudgetExceededError",
    "CompressTimeLedgerCorruptError",
    # Composition
    "ComposableCompressPipeline",
    "PipelineStageRef",
    "CompressTimePipelineResult",
    # Builders + specs + enums
    "GenericTTOHarness",
    "GenericTTOHarnessSpec",
    "LEGAL_TTO_TARGET_KIND",
    "MultipassRefinement",
    "MultipassRefinementSpec",
    "SimulatedAnnealingOnDiscreteCodes",
    "SimulatedAnnealingSpec",
    "LEGAL_SA_TEMP_SCHEDULE",
    "LEGAL_SA_DISCRETE_TARGET",
    "PerPairCoordinateSearch",
    "PerPairCoordinateSearchSpec",
    "IteratedBisectionRateKnee",
    "IteratedBisectionRateKneeSpec",
    "LEGAL_BISECTION_GRANULARITY",
    # Persistence
    "append_pass_outcome_locked",
    "load_pass_outcomes",
    "load_pass_outcomes_strict",
    "COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH",
    "COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK",
    "COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION",
]
