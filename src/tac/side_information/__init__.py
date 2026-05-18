# SPDX-License-Identifier: MIT
"""tac.side_information — decorator-based composable side-information baker
namespace.

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§5 the canonical-helper namespace design + §J side-information / pre-
processing / per-pair input conditioning. Sister of ``tac.boosting`` and
``tac.compress_time_optimization`` (same decorator + frozen-dataclass +
composition-operator pattern at a different stage surface).

Per CLAUDE.md "Subagent coherence-by-default" anti-fragmentation primitive
+ Wyner-Ziv 1976 source-coding-with-side-information theorem: every piece
of side information used by the contest decoder MUST be derivable from
public sources (the strict-scorer-rule forbids loading scorer at inflate;
the publicly-reproducible-prior contract enforces structural reproducibility
of any other side-info source).

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  Decorator:
    - ``side_info_baker(contract)``: register a baker's contract + function
    - ``SideInfoBakerContract``: ~28-field frozen dataclass contract
    - ``SideInfoBakerContractError`` / ``NonReproducibleSideInfoViolation``
      / ``CanonicalComma2k19CacheRequiredViolation`` /
      ``WynerZivCorrelationInvalidError``: decoration-time error classes

  Registry read API:
    - ``get_registered_bakers()``: id → contract map
    - ``get_baker_function(baker_id)``: id → callable
    - ``validate_all_registered_bakers()``: re-validate every contract

  Composition:
    - ``ComposableSideInfoPipeline``: pipe-operator pipeline builder
    - ``PipelineBakerRef``: single baker reference
    - ``SideInfoPipelineResult``: run output

  Builders (one per parent-prompt canonical-primitive row):
    - ``ScorerWeightsAsSharedPrior`` + ``ScorerWeightsAsSharedPriorSpec``
    - ``Comma2k19DerivedPriorPalette`` +
      ``Comma2k19DerivedPriorPaletteSpec``
    - ``ImageNetStatisticsPrior`` + ``ImageNetStatisticsPriorSpec``
    - ``DashcamDomainPrior`` + ``DashcamDomainPriorSpec``
    - ``WynerZivResidualEncoder`` + ``WynerZivResidualEncoderSpec``

  Persistence (opt-in):
    - ``append_baker_outcome_locked(record)``: fcntl-locked JSONL append
    - ``load_baker_outcomes`` / ``load_baker_outcomes_strict``: read API
    - ``SideInfoLedgerCorruptError``: corruption signal (Catalog #138)
    - ``SIDE_INFO_BAKER_OUTCOMES_PATH`` / ``SIDE_INFO_BAKER_OUTCOMES_LOCK``

  Errors:
    - ``SideInformationError``: root exception
    - ``SideInfoPipelineError`` / ``AmbiguousCompositionError`` /
      ``SideInfoArchiveBudgetViolation`` /
      ``InflateRuntimeBudgetExceededError``: pipeline-build / run errors

Cross-references:
  - Design memo: `.omx/research/tac_side_information_namespace_design_20260517.md`
  - Premise verification (Catalog #229): `.omx/tmp/tac_side_information_premise_verifier.txt`
  - Sister namespaces: ``tac.boosting``, ``tac.compress_time_optimization``
  - Spec provenance: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
    §5.3-§5.5 + §J side-information
  - Comma2k19 canonical helper (Catalog #213):
    ``tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache``
  - DP1 codebook provenance (Catalog #210):
    ``tac.substrates.pretrained_driving_prior.codebook``
  - Empirical anchor (cooperative-receiver):
    ``feedback_dp1_phase_2_landed_20260514.md`` + Atick-Redlich 1990 +
    Wyner-Ziv 1976.
"""

from __future__ import annotations

from tac.side_information.comma2k19_derived_prior_palette import (
    LEGAL_PALETTE_KIND,
    Comma2k19DerivedPriorPalette,
    Comma2k19DerivedPriorPaletteSpec,
)
from tac.side_information.contract import (
    LEGAL_CORRECTION_KIND,
    LEGAL_CORRECTION_RESOLUTION,
    LEGAL_HOOK_AUTOPILOT,
    LEGAL_HOOK_BIT_ALLOCATOR,
    LEGAL_HOOK_CONTINUAL_LEARNING,
    LEGAL_HOOK_PARETO,
    LEGAL_HOOK_SENSITIVITY,
    LEGAL_MERGE_POLICY,
    LEGAL_SIDE_INFO_SOURCE,
    LEGAL_STAGE_PHASE,
    NOT_APPLICABLE_WITH_RATIONALE,
    SideInfoBakerContract,
)
from tac.side_information.dashcam_domain_prior import (
    LEGAL_DASHCAM_PRIOR_KIND,
    LEGAL_SOURCE_DATASET,
    DashcamDomainPrior,
    DashcamDomainPriorSpec,
)
from tac.side_information.decorator import (
    _clear_baker_registry_for_tests,
    _REGISTERED_BAKERS,
    get_baker_function,
    get_registered_bakers,
    side_info_baker,
    validate_all_registered_bakers,
)
from tac.side_information.errors import (
    AmbiguousCompositionError,
    CanonicalComma2k19CacheRequiredViolation,
    InflateRuntimeBudgetExceededError,
    NonReproducibleSideInfoViolation,
    SideInfoArchiveBudgetViolation,
    SideInfoBakerContractError,
    SideInfoLedgerCorruptError,
    SideInfoPipelineError,
    SideInformationError,
    WynerZivCorrelationInvalidError,
)
from tac.side_information.imagenet_statistics_prior import (
    LEGAL_STATISTIC_KIND,
    ImageNetStatisticsPrior,
    ImageNetStatisticsPriorSpec,
)
from tac.side_information.persistence import (
    SIDE_INFO_BAKER_OUTCOMES_LOCK,
    SIDE_INFO_BAKER_OUTCOMES_PATH,
    SIDE_INFO_BAKER_OUTCOMES_SCHEMA_VERSION,
    append_baker_outcome_locked,
    load_baker_outcomes,
    load_baker_outcomes_strict,
)
from tac.side_information.pipeline import (
    ComposableSideInfoPipeline,
    PipelineBakerRef,
    SideInfoPipelineResult,
)
from tac.side_information.scorer_weights_as_shared_prior import (
    LEGAL_FEATURE_EXTRACTION_KIND,
    ScorerWeightsAsSharedPrior,
    ScorerWeightsAsSharedPriorSpec,
)
from tac.side_information.wyner_ziv_residual_encoder import (
    LEGAL_RECONSTRUCTION_FN,
    LEGAL_RESIDUAL_CODE,
    WynerZivResidualEncoder,
    WynerZivResidualEncoderSpec,
)
from tac.side_information.per_pair_master_gradient_wire_in import (
    SideInformationPerPairWireInOutcome,
    compose_side_information_per_pair_wire_in,
)

__all__ = [
    # Per-pair master gradient wire-in (LOW gap closure widened wave 2026-05-17)
    "SideInformationPerPairWireInOutcome",
    "compose_side_information_per_pair_wire_in",
    # Decorator + registry
    "side_info_baker",
    "get_registered_bakers",
    "get_baker_function",
    "validate_all_registered_bakers",
    "_REGISTERED_BAKERS",
    "_clear_baker_registry_for_tests",
    # Contract + enums
    "SideInfoBakerContract",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_SIDE_INFO_SOURCE",
    "LEGAL_STAGE_PHASE",
    "LEGAL_MERGE_POLICY",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    # Errors
    "SideInformationError",
    "SideInfoBakerContractError",
    "NonReproducibleSideInfoViolation",
    "CanonicalComma2k19CacheRequiredViolation",
    "WynerZivCorrelationInvalidError",
    "SideInfoPipelineError",
    "AmbiguousCompositionError",
    "SideInfoArchiveBudgetViolation",
    "InflateRuntimeBudgetExceededError",
    "SideInfoLedgerCorruptError",
    # Composition
    "ComposableSideInfoPipeline",
    "PipelineBakerRef",
    "SideInfoPipelineResult",
    # Builders + specs + enums
    "ScorerWeightsAsSharedPrior",
    "ScorerWeightsAsSharedPriorSpec",
    "LEGAL_FEATURE_EXTRACTION_KIND",
    "Comma2k19DerivedPriorPalette",
    "Comma2k19DerivedPriorPaletteSpec",
    "LEGAL_PALETTE_KIND",
    "ImageNetStatisticsPrior",
    "ImageNetStatisticsPriorSpec",
    "LEGAL_STATISTIC_KIND",
    "DashcamDomainPrior",
    "DashcamDomainPriorSpec",
    "LEGAL_DASHCAM_PRIOR_KIND",
    "LEGAL_SOURCE_DATASET",
    "WynerZivResidualEncoder",
    "WynerZivResidualEncoderSpec",
    "LEGAL_RECONSTRUCTION_FN",
    "LEGAL_RESIDUAL_CODE",
    # Persistence
    "append_baker_outcome_locked",
    "load_baker_outcomes",
    "load_baker_outcomes_strict",
    "SIDE_INFO_BAKER_OUTCOMES_PATH",
    "SIDE_INFO_BAKER_OUTCOMES_LOCK",
    "SIDE_INFO_BAKER_OUTCOMES_SCHEMA_VERSION",
]
