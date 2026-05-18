# SPDX-License-Identifier: MIT
"""tac.boosting — decorator-based composable boost stage namespace.

Per `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5 the canonical-helper namespace design + §I.6 boosting / stack-of-
stacks framework. Sister of ``tac.substrate_registry`` (same decorator-
plus-contract pattern at the substrate-registry surface).

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  Decorator:
    - ``boost_stage(contract)``: register a stage's contract + function
    - ``BoostStageContract``: 23-field frozen dataclass contract
    - ``BoostStageContractError`` / ``DeterminismViolation`` /
      ``ScorerFreedomViolation``: decoration-time error classes

  Registry read API:
    - ``get_registered_stages()``: id → contract map
    - ``get_stage_function(stage_id)``: id → callable
    - ``validate_all_registered_stages()``: re-validate every contract

  Composition:
    - ``ComposableBoostingPipeline``: pipe-operator pipeline builder
    - ``PipelineStageRef``: single stage reference
    - ``BoostingPipelineResult``: run output (final_state +
      per_stage_outcomes + rejected_stages + pareto_snapshot)

  Builders:
    - ``ResidualCascadeBuilder`` + ``ResidualCascadeStageSpec``
    - ``PerPairDecoderEnsembleSelector`` + ``PerPairDecoderEnsembleSpec``
    - ``ModeEnsembleDispatch`` + ``ModeEnsembleDispatchSpec``

  Pareto frontier:
    - ``ParetoFrontTracker`` + ``ParetoAnchor``
    - ``ParetoFrontTrackerError``

  Persistence (opt-in):
    - ``append_stage_outcome_locked(record)``: fcntl-locked JSONL append
    - ``load_stage_outcomes`` / ``load_stage_outcomes_strict``: read API
    - ``BoostingLedgerCorruptError``: corruption signal (Catalog #138 sister)
    - ``BOOSTING_STAGE_OUTCOMES_PATH`` / ``BOOSTING_STAGE_OUTCOMES_LOCK``

  Errors:
    - ``BoostingNamespaceError``: root exception
    - ``BoostingPipelineError`` / ``AmbiguousCompositionError``: pipeline errors

Cross-references:
  - Design memo: `.omx/research/tac_boosting_namespace_design_20260517.md`
  - Premise verification (Catalog #229): `.omx/tmp/tac_boosting_premise_verifier.txt`
  - Sister namespace: ``tac.substrate_registry``
  - Empirical anchor: PR106 format0d 2-pass additive correction at
    ``submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575``
"""

from __future__ import annotations

from tac.boosting.contract import (
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
    BoostStageContract,
)
from tac.boosting.decorator import (
    _REGISTERED_STAGES,
    _clear_stage_registry_for_tests,
    boost_stage,
    get_registered_stages,
    get_stage_function,
    validate_all_registered_stages,
)
from tac.boosting.errors import (
    AmbiguousCompositionError,
    BoostingLedgerCorruptError,
    BoostingNamespaceError,
    BoostingPipelineError,
    BoostStageContractError,
    DeterminismViolation,
    ScorerFreedomViolation,
)
from tac.boosting.mode_ensemble_dispatch import (
    ModeEnsembleDispatch,
    ModeEnsembleDispatchSpec,
)
from tac.boosting.pareto_front import (
    ParetoAnchor,
    ParetoFrontTracker,
    ParetoFrontTrackerError,
)
from tac.boosting.per_pair_decoder_ensemble import (
    PerPairDecoderEnsembleSelector,
    PerPairDecoderEnsembleSpec,
)
from tac.boosting.persistence import (
    BOOSTING_STAGE_OUTCOMES_LOCK,
    BOOSTING_STAGE_OUTCOMES_PATH,
    BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION,
    append_stage_outcome_locked,
    load_stage_outcomes,
    load_stage_outcomes_strict,
)
from tac.boosting.pipeline import (
    BoostingPipelineResult,
    ComposableBoostingPipeline,
    PipelineStageRef,
)
from tac.boosting.residual_cascade import (
    ResidualCascadeBuilder,
    ResidualCascadeStageSpec,
)
from tac.boosting.per_pair_master_gradient_wire_in import (
    BoostingPerPairWireInOutcome,
    compose_boosting_per_pair_wire_in,
)

__all__ = [
    "BOOSTING_STAGE_OUTCOMES_LOCK",
    "BOOSTING_STAGE_OUTCOMES_PATH",
    "BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION",
    "LEGAL_CORRECTION_KIND",
    "LEGAL_CORRECTION_RESOLUTION",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_MERGE_POLICY",
    "LEGAL_STAGE_PHASE",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "_REGISTERED_STAGES",
    "AmbiguousCompositionError",
    "BoostStageContract",
    "BoostStageContractError",
    "BoostingLedgerCorruptError",
    "BoostingNamespaceError",
    "BoostingPipelineError",
    "BoostingPipelineResult",
    "BoostingPerPairWireInOutcome",
    "ComposableBoostingPipeline",
    "DeterminismViolation",
    "ModeEnsembleDispatch",
    "ModeEnsembleDispatchSpec",
    "ParetoAnchor",
    "ParetoFrontTracker",
    "ParetoFrontTrackerError",
    "PerPairDecoderEnsembleSelector",
    "PerPairDecoderEnsembleSpec",
    "PipelineStageRef",
    "ResidualCascadeBuilder",
    "ResidualCascadeStageSpec",
    "ScorerFreedomViolation",
    "_clear_stage_registry_for_tests",
    "append_stage_outcome_locked",
    "boost_stage",
    "compose_boosting_per_pair_wire_in",
    "get_registered_stages",
    "get_stage_function",
    "load_stage_outcomes",
    "load_stage_outcomes_strict",
    "validate_all_registered_stages",
]
