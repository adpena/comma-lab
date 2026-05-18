# SPDX-License-Identifier: MIT
"""tac.inflate_time_post_processing — decorator-based composable inflate-time
post-processing pass namespace.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5 the canonical-helper namespace design + §G inflate-time techniques.
Sister of ``tac.boosting`` and ``tac.compress_time_optimization`` (same
decorator + frozen-dataclass + composition-operator pattern at a different
stage surface — namely the 30-min T4 inflate-time wall budget per spec
§G).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + the operator standing
directive 2026-05-15: this namespace is STRUCTURALLY INDEPENDENT from
``tac.boosting`` and ``tac.compress_time_optimization``. It does NOT
re-export their decorators, contracts, errors, or pipelines. Each
namespace evolves independently.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  Decorator:
    - ``inflate_time_post_filter(contract)``: register a pass's contract +
      function
    - ``InflateTimePostProcessingContract``: ~28-field frozen dataclass
      contract
    - ``InflateTimePassContractError`` / ``CompressPhaseForbiddenError`` /
      ``ScorerAccessForbiddenError`` / ``ArchiveBytesViolation`` /
      ``WallclockBudgetRequiredError`` / ``SeedRequiredViolation``:
      decoration-time error classes

  Registry read API:
    - ``get_registered_passes()``: id → contract map
    - ``get_pass_function(pass_id)``: id → callable
    - ``validate_all_registered_passes()``: re-validate every contract

  Composition:
    - ``ComposableInflatePipeline``: pipe-operator pipeline builder
    - ``PipelineStageRef``: single pass reference
    - ``InflateTimePipelineResult``: run output

  Builders (one per §G inflate-time row, collapsed to 5 first-class):
    - ``BilateralFilterPostProcessor`` + ``BilateralFilterSpec``
    - ``NLMDenoisingPostProcessor`` + ``NLMDenoisingSpec``
    - ``LearnedPostFilterApplier`` + ``LearnedPostFilterSpec``
    - ``SuperResolutionUpscaler`` + ``SuperResolutionUpscalerSpec``
    - ``MultiPassInflateRefinement`` + ``MultiPassInflateRefinementSpec``

  Persistence (opt-in):
    - ``append_pass_outcome_locked(record)``: fcntl-locked JSONL append
    - ``load_pass_outcomes`` / ``load_pass_outcomes_strict``: read API
    - ``InflateTimeLedgerCorruptError``: corruption signal (Catalog #138)
    - ``INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_PATH`` /
      ``INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_LOCK``

  Errors:
    - ``InflateTimePostProcessingError``: root exception
    - ``InflateTimePipelineError`` / ``AmbiguousCompositionError`` /
      ``InflateBudgetExceededError``: pipeline-build / run errors

Cross-references:
  - Design memo:
    `.omx/research/tac_inflate_time_post_processing_namespace_design_20260517.md`
  - Premise verification (Catalog #229):
    `.omx/tmp/tac_inflate_time_post_processing_premise_verifier.txt`
  - Sister namespaces: ``tac.boosting`` /
    ``tac.compress_time_optimization``
  - Spec provenance:
    `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
    §5.3-§5.5 + §G inflate-time
  - Empirical anchor (existing post-filter):
    ``experiments/train_postfilter_on_renderer.py``
  - Empirical anchor (inflate runtime):
    ``submissions/exact_current/inflate.py`` + Catalog #146 (canonical
    inflate.py budget ≤ 100 LOC + ≤ 2 dependencies)
"""

from __future__ import annotations

from tac.inflate_time_post_processing.bilateral_filter import (
    BilateralFilterPostProcessor,
    BilateralFilterSpec,
)
from tac.inflate_time_post_processing.contract import (
    LEGAL_APPLIES_TO_FRAMES,
    LEGAL_CORRECTION_KIND,
    LEGAL_CORRECTION_RESOLUTION,
    LEGAL_HOOK_AUTOPILOT,
    LEGAL_HOOK_BIT_ALLOCATOR,
    LEGAL_HOOK_CONTINUAL_LEARNING,
    LEGAL_HOOK_PARETO,
    LEGAL_HOOK_SENSITIVITY,
    LEGAL_MERGE_POLICY,
    LEGAL_SCORE_AXIS,
    LEGAL_STAGE_PHASE,
    MAX_INFLATE_COMPUTE_BUDGET_SECONDS,
    NOT_APPLICABLE_WITH_RATIONALE,
    InflateTimePostProcessingContract,
)
from tac.inflate_time_post_processing.decorator import (
    _clear_pass_registry_for_tests,
    _REGISTERED_PASSES,
    get_pass_function,
    get_registered_passes,
    inflate_time_post_filter,
    validate_all_registered_passes,
)
from tac.inflate_time_post_processing.errors import (
    AmbiguousCompositionError,
    ArchiveBytesViolation,
    CompressPhaseForbiddenError,
    InflateBudgetExceededError,
    InflateTimeLedgerCorruptError,
    InflateTimePassContractError,
    InflateTimePipelineError,
    InflateTimePostProcessingError,
    ScorerAccessForbiddenError,
    SeedRequiredViolation,
    WallclockBudgetRequiredError,
)
from tac.inflate_time_post_processing.learned_post_filter import (
    LearnedPostFilterApplier,
    LearnedPostFilterSpec,
)
from tac.inflate_time_post_processing.multi_pass_refinement import (
    MultiPassInflateRefinement,
    MultiPassInflateRefinementSpec,
)
from tac.inflate_time_post_processing.nlm_denoising import (
    NLMDenoisingPostProcessor,
    NLMDenoisingSpec,
)
from tac.inflate_time_post_processing.persistence import (
    INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_LOCK,
    INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_PATH,
    INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_SCHEMA_VERSION,
    append_pass_outcome_locked,
    load_pass_outcomes,
    load_pass_outcomes_strict,
)
from tac.inflate_time_post_processing.pipeline import (
    ComposableInflatePipeline,
    InflateTimePipelineResult,
    PipelineStageRef,
)
from tac.inflate_time_post_processing.super_resolution_upscaler import (
    LEGAL_UPSCALER_KIND,
    SuperResolutionUpscaler,
    SuperResolutionUpscalerSpec,
)
from tac.inflate_time_post_processing.per_pair_master_gradient_wire_in import (
    InflateTimePostProcessingPerPairWireInOutcome,
    compose_inflate_time_post_processing_per_pair_wire_in,
)

__all__ = [
    # Per-pair master gradient wire-in (LOW gap closure widened wave 2026-05-17)
    "InflateTimePostProcessingPerPairWireInOutcome",
    "compose_inflate_time_post_processing_per_pair_wire_in",
    # Decorator + registry
    "inflate_time_post_filter",
    "get_registered_passes",
    "get_pass_function",
    "validate_all_registered_passes",
    "_REGISTERED_PASSES",
    "_clear_pass_registry_for_tests",
    # Contract + enums
    "InflateTimePostProcessingContract",
    "MAX_INFLATE_COMPUTE_BUDGET_SECONDS",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_APPLIES_TO_FRAMES",
    "LEGAL_SCORE_AXIS",
    "LEGAL_STAGE_PHASE",
    "LEGAL_MERGE_POLICY",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    # Errors
    "InflateTimePostProcessingError",
    "InflateTimePassContractError",
    "CompressPhaseForbiddenError",
    "ScorerAccessForbiddenError",
    "ArchiveBytesViolation",
    "WallclockBudgetRequiredError",
    "SeedRequiredViolation",
    "InflateTimePipelineError",
    "AmbiguousCompositionError",
    "InflateBudgetExceededError",
    "InflateTimeLedgerCorruptError",
    # Composition
    "ComposableInflatePipeline",
    "PipelineStageRef",
    "InflateTimePipelineResult",
    # Builders + specs + enums
    "BilateralFilterPostProcessor",
    "BilateralFilterSpec",
    "NLMDenoisingPostProcessor",
    "NLMDenoisingSpec",
    "LearnedPostFilterApplier",
    "LearnedPostFilterSpec",
    "SuperResolutionUpscaler",
    "SuperResolutionUpscalerSpec",
    "LEGAL_UPSCALER_KIND",
    "MultiPassInflateRefinement",
    "MultiPassInflateRefinementSpec",
    # Persistence
    "append_pass_outcome_locked",
    "load_pass_outcomes",
    "load_pass_outcomes_strict",
    "INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_PATH",
    "INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_LOCK",
    "INFLATE_TIME_POST_PROCESSING_PASS_OUTCOMES_SCHEMA_VERSION",
]
