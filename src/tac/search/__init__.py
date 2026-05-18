# SPDX-License-Identifier: MIT
"""tac.search — decorator-based composable search strategy namespace.

Per `.omx/research/tac_search_namespace_design_20260517.md` + the §7.6
spec at
`.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5 the canonical-helper namespace design + §F search-method coverage.
Sister of ``tac.boosting`` + ``tac.compress_time_optimization`` (same
decorator-plus-contract pattern at the search-strategy surface).

This namespace is the ENGINE for the `@` operator on sister pipelines
(``ComposableBoostingPipeline`` + ``ComposableCompressPipeline``). The
sister pipelines store the search descriptor as opaque metadata via
``__matmul__``; ``tac.search.run_search_over_pipeline`` resolves the
descriptor and executes the actual search.

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  Decorator:
    - ``search_strategy(contract)``: register a strategy's contract +
      function
    - ``SearchContract``: ~20-field frozen dataclass contract
    - ``SearchContractError`` / ``DeterminismViolation`` /
      ``SeedRequiredViolation`` / ``SearchEngineNotInstalledError``:
      decoration-time / run-time error classes

  Registry read API:
    - ``get_registered_strategies()``: id → contract map
    - ``get_strategy_function(strategy_id)``: id → callable
    - ``validate_all_registered_strategies()``: re-validate every contract

  Composition:
    - ``ComposableSearchPipeline``: pipe-operator pipeline builder for
      stand-alone search composition (RandomSearch warmup | TPE refinement)
    - ``SearchPipelineStrategyRef``: single strategy reference
    - ``SearchResult`` / ``SearchHistory`` / ``SearchTrial``: result
      dataclasses
    - ``run_search_over_pipeline(sister_pipeline, objective_fn)``:
      canonical helper for `@`-attached search execution on sister
      pipelines

  Builders (one per §6 builder row):
    - ``CMAESCandidateSearcher`` + ``CMAESCandidateSearcherSpec``
    - ``OptunaTPESampler`` + ``OptunaTPESamplerSpec``
    - ``BayesianOptimizationGP`` + ``BayesianOptimizationGPSpec``
    - ``MCTSCodebookSearcher`` + ``MCTSCodebookSearcherSpec``
    - ``RashomonEnsembleCommittee`` + ``RashomonEnsembleCommitteeSpec``

  Persistence (opt-in):
    - ``append_search_outcome_locked(record)``: fcntl-locked JSONL append
    - ``load_search_outcomes`` / ``load_search_outcomes_strict``: read API
    - ``SearchLedgerCorruptError``: corruption signal (Catalog #138)
    - ``SEARCH_STRATEGY_OUTCOMES_PATH`` / ``SEARCH_STRATEGY_OUTCOMES_LOCK``
    - ``query_outcomes_by_strategy_id`` / ``query_outcomes_by_objective_label``
      / ``latest_best_score_by_strategy``

  Errors:
    - ``SearchNamespaceError``: root exception
    - ``SearchPipelineError`` / ``SearchAmbiguousCompositionError`` /
      ``SearchBudgetExceededError`` / ``ObjectiveFunctionError``:
      pipeline-build / run errors
    - ``SearchStrategyNotRegisteredError``: missing strategy lookup

Cross-references:
  - Design memo: `.omx/research/tac_search_namespace_design_20260517.md`
  - Premise verification (Catalog #229): `.omx/tmp/tac_search_premise_verifier.txt`
  - Sister namespaces: ``tac.boosting`` + ``tac.compress_time_optimization``
  - Canonical Rashomon re-export: ``tac.autopilot_rudin_daubechies.RashomonEnsembleRanker`` (Catalog #252)
"""

from __future__ import annotations

from tac.search.bayesian_optimization_gp import (
    BayesianOptimizationGP,
    BayesianOptimizationGPSpec,
)
from tac.search.cma_es_searcher import (
    CMAESCandidateSearcher,
    CMAESCandidateSearcherSpec,
)
from tac.search.contract import (
    LEGAL_HOOK_AUTOPILOT,
    LEGAL_HOOK_BIT_ALLOCATOR,
    LEGAL_HOOK_CONTINUAL_LEARNING,
    LEGAL_HOOK_PARETO,
    LEGAL_HOOK_SENSITIVITY,
    LEGAL_PARALLELISM,
    LEGAL_SEARCH_KIND,
    NOT_APPLICABLE_WITH_RATIONALE,
    SearchContract,
)
from tac.search.decorator import (
    _REGISTERED_STRATEGIES,
    _clear_strategy_registry_for_tests,
    get_registered_strategies,
    get_strategy_function,
    search_strategy,
    validate_all_registered_strategies,
)
from tac.search.errors import (
    DeterminismViolation,
    ObjectiveFunctionError,
    SearchAmbiguousCompositionError,
    SearchBudgetExceededError,
    SearchContractError,
    SearchEngineNotInstalledError,
    SearchLedgerCorruptError,
    SearchNamespaceError,
    SearchPipelineError,
    SearchStrategyNotRegisteredError,
    SeedRequiredViolation,
)
from tac.search.mcts_codebook_searcher import (
    MCTSCodebookSearcher,
    MCTSCodebookSearcherSpec,
)
from tac.search.optuna_tpe_sampler import (
    OptunaTPESampler,
    OptunaTPESamplerSpec,
)
from tac.search.persistence import (
    SEARCH_STRATEGY_OUTCOMES_LOCK,
    SEARCH_STRATEGY_OUTCOMES_PATH,
    SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION,
    append_search_outcome_locked,
    latest_best_score_by_strategy,
    load_search_outcomes,
    load_search_outcomes_strict,
    query_outcomes_by_objective_label,
    query_outcomes_by_strategy_id,
)
from tac.search.pipeline import (
    ComposableSearchPipeline,
    SearchHistory,
    SearchPipelineStrategyRef,
    SearchResult,
    SearchTrial,
    run_search_over_pipeline,
)
from tac.search.rashomon_ensemble_committee import (
    RashomonEnsembleCommittee,
    RashomonEnsembleCommitteeSpec,
)
from tac.search.per_pair_master_gradient_wire_in import (
    SearchPerPairWireInOutcome,
    compose_search_per_pair_wire_in,
)

__all__ = [
    # Per-pair master gradient wire-in (LOW gap closure widened wave 2026-05-17)
    "SearchPerPairWireInOutcome",
    "compose_search_per_pair_wire_in",
    # Decorator + registry
    "search_strategy",
    "get_registered_strategies",
    "get_strategy_function",
    "validate_all_registered_strategies",
    "_REGISTERED_STRATEGIES",
    "_clear_strategy_registry_for_tests",
    # Contract + enums
    "SearchContract",
    "NOT_APPLICABLE_WITH_RATIONALE",
    "LEGAL_SEARCH_KIND",
    "LEGAL_PARALLELISM",
    "LEGAL_HOOK_SENSITIVITY",
    "LEGAL_HOOK_PARETO",
    "LEGAL_HOOK_BIT_ALLOCATOR",
    "LEGAL_HOOK_AUTOPILOT",
    "LEGAL_HOOK_CONTINUAL_LEARNING",
    # Errors
    "SearchNamespaceError",
    "SearchContractError",
    "DeterminismViolation",
    "SeedRequiredViolation",
    "SearchStrategyNotRegisteredError",
    "SearchEngineNotInstalledError",
    "SearchPipelineError",
    "SearchAmbiguousCompositionError",
    "SearchBudgetExceededError",
    "ObjectiveFunctionError",
    "SearchLedgerCorruptError",
    # Composition
    "ComposableSearchPipeline",
    "SearchPipelineStrategyRef",
    "SearchResult",
    "SearchHistory",
    "SearchTrial",
    "run_search_over_pipeline",
    # Builders + specs
    "CMAESCandidateSearcher",
    "CMAESCandidateSearcherSpec",
    "OptunaTPESampler",
    "OptunaTPESamplerSpec",
    "BayesianOptimizationGP",
    "BayesianOptimizationGPSpec",
    "MCTSCodebookSearcher",
    "MCTSCodebookSearcherSpec",
    "RashomonEnsembleCommittee",
    "RashomonEnsembleCommitteeSpec",
    # Persistence
    "append_search_outcome_locked",
    "load_search_outcomes",
    "load_search_outcomes_strict",
    "query_outcomes_by_strategy_id",
    "query_outcomes_by_objective_label",
    "latest_best_score_by_strategy",
    "SEARCH_STRATEGY_OUTCOMES_PATH",
    "SEARCH_STRATEGY_OUTCOMES_LOCK",
    "SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION",
]
