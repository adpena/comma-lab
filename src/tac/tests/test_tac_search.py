# SPDX-License-Identifier: MIT
"""Tests for the tac.search namespace.

Covers per design memo §8 9-dim checklist:
  - Contract validation (~20 tests)
  - Decorator behavior (~15 tests)
  - Pipeline composition (~15 tests)
  - 5 builders (~8 tests each = 40)
  - The `@` operator integration with sibling tac.boosting (~5 tests)
  - SeedRequiredViolation (~3 tests)
  - Search budget enforcement (~3 tests)
  - Rashomon re-export integrity NOT re-implementation (~3 tests)
  - Surrogate-vs-real-objective routing (~3 tests)
  - Persistence (~5 tests)

Target total: ≥ 90 tests.

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" + Catalog #229 +
Catalog #294: every claim in the design memo is backed by an executable
test in this file.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from tac.search import (
    BayesianOptimizationGP,
    BayesianOptimizationGPSpec,
    CMAESCandidateSearcher,
    CMAESCandidateSearcherSpec,
    ComposableSearchPipeline,
    DeterminismViolation,
    LEGAL_HOOK_AUTOPILOT,
    LEGAL_HOOK_BIT_ALLOCATOR,
    LEGAL_HOOK_CONTINUAL_LEARNING,
    LEGAL_HOOK_PARETO,
    LEGAL_HOOK_SENSITIVITY,
    LEGAL_PARALLELISM,
    LEGAL_SEARCH_KIND,
    MCTSCodebookSearcher,
    MCTSCodebookSearcherSpec,
    NOT_APPLICABLE_WITH_RATIONALE,
    ObjectiveFunctionError,
    OptunaTPESampler,
    OptunaTPESamplerSpec,
    RashomonEnsembleCommittee,
    RashomonEnsembleCommitteeSpec,
    SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION,
    SearchAmbiguousCompositionError,
    SearchBudgetExceededError,
    SearchContract,
    SearchContractError,
    SearchEngineNotInstalledError,
    SearchHistory,
    SearchLedgerCorruptError,
    SearchNamespaceError,
    SearchPipelineError,
    SearchPipelineStrategyRef,
    SearchResult,
    SearchStrategyNotRegisteredError,
    SearchTrial,
    SeedRequiredViolation,
    _clear_strategy_registry_for_tests,
    append_search_outcome_locked,
    get_registered_strategies,
    get_strategy_function,
    latest_best_score_by_strategy,
    load_search_outcomes,
    load_search_outcomes_strict,
    query_outcomes_by_objective_label,
    query_outcomes_by_strategy_id,
    run_search_over_pipeline,
    search_strategy,
    validate_all_registered_strategies,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty strategy registry."""
    _clear_strategy_registry_for_tests()
    yield
    _clear_strategy_registry_for_tests()


def _minimal_contract(**overrides):
    """Helper to build a minimal valid SearchContract for tests."""
    base = dict(
        id="t",
        search_kind="continuous",
        n_candidate_evaluations_max=10,
        parallelism="serial",
        deterministic=True,
        seed=42,
        hook_probe_disambiguator=None,
        hook_not_applicable_rationale={
            "hook_sensitivity_contribution": "test",
            "hook_pareto_constraint": "test",
            "hook_bit_allocator_class": "test",
            "hook_probe_disambiguator": "test",
        },
    )
    base.update(overrides)
    return SearchContract(**base)


def _quadratic_objective(target=0.3):
    """Toy quadratic objective: minimum at params['x'] = target."""
    def _obj(params):
        return (params["x"] - target) ** 2
    return _obj


# ---------------------------------------------------------------------------
# §1 SearchContract validation tests
# ---------------------------------------------------------------------------


def test_contract_minimal_construction_succeeds():
    c = _minimal_contract()
    assert c.id == "t"
    assert c.seed == 42
    assert c.requires_objective_function is True


def test_contract_rejects_kebab_case_id():
    with pytest.raises(SearchContractError, match="snake_case"):
        _minimal_contract(id="bad-kebab-id")


def test_contract_rejects_uppercase_id():
    with pytest.raises(SearchContractError, match=r"\^\[a-z\]"):
        _minimal_contract(id="UpperCase")


def test_contract_rejects_empty_id():
    with pytest.raises(SearchContractError):
        _minimal_contract(id="")


def test_contract_rejects_self_parent():
    with pytest.raises(SearchContractError, match="cannot be its own parent"):
        _minimal_contract(parent_strategy_id="t")


def test_contract_accepts_none_parent():
    c = _minimal_contract(parent_strategy_id=None)
    assert c.parent_strategy_id is None


def test_contract_rejects_bad_search_kind():
    with pytest.raises(SearchContractError, match="search_kind"):
        _minimal_contract(search_kind="invalid_kind")


def test_contract_rejects_bad_parallelism():
    with pytest.raises(SearchContractError, match="parallelism"):
        _minimal_contract(parallelism="invalid_parallelism")


def test_contract_rejects_zero_evaluation_budget():
    with pytest.raises(SearchContractError, match="must be >= 1"):
        _minimal_contract(n_candidate_evaluations_max=0)


def test_contract_rejects_negative_evaluation_budget():
    with pytest.raises(SearchContractError):
        _minimal_contract(n_candidate_evaluations_max=-1)


def test_contract_rejects_bool_evaluation_budget():
    with pytest.raises(SearchContractError):
        _minimal_contract(n_candidate_evaluations_max=True)


def test_contract_hard_pins_requires_objective_true():
    with pytest.raises(SearchContractError, match="FORBIDDEN"):
        _minimal_contract(requires_objective_function=False)


def test_contract_rejects_negative_seed():
    with pytest.raises(SearchContractError):
        _minimal_contract(seed=-1)


def test_contract_accepts_none_seed():
    c = _minimal_contract(seed=None, deterministic=False)
    assert c.seed is None


def test_contract_rejects_negative_predicted_cost():
    with pytest.raises(SearchContractError):
        _minimal_contract(predicted_search_cost_usd=-0.1)


def test_contract_rejects_bad_hook_value():
    with pytest.raises(SearchContractError, match="hook_autopilot_ranker"):
        _minimal_contract(hook_autopilot_ranker="invalid_hook_value")


def test_contract_rejects_na_hook_without_rationale():
    with pytest.raises(SearchContractError, match="not_applicable_with_rationale"):
        _minimal_contract(
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_not_applicable_rationale={
                # missing sensitivity rationale
                "hook_pareto_constraint": "x",
                "hook_bit_allocator_class": "x",
                "hook_probe_disambiguator": "x",
            },
        )


def test_contract_rejects_illegal_rationale_key():
    with pytest.raises(SearchContractError, match="illegal key"):
        _minimal_contract(
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": "x",
                "hook_pareto_constraint": "x",
                "hook_bit_allocator_class": "x",
                "hook_probe_disambiguator": "x",
                "hook_invalid_name": "should reject",
            },
        )


def test_contract_to_dict_round_trip():
    c = _minimal_contract()
    d = c.to_dict()
    c2 = SearchContract.from_dict(d)
    assert c.id == c2.id
    assert c.seed == c2.seed
    assert c.hook_not_applicable_rationale == c2.hook_not_applicable_rationale


def test_contract_to_dict_sorted_keys_for_byte_stability():
    c = _minimal_contract()
    d1 = json.dumps(c.to_dict(), sort_keys=True)
    d2 = json.dumps(c.to_dict(), sort_keys=True)
    assert d1 == d2


def test_legal_enum_sets_are_frozensets():
    assert isinstance(LEGAL_SEARCH_KIND, frozenset)
    assert isinstance(LEGAL_PARALLELISM, frozenset)
    assert isinstance(LEGAL_HOOK_AUTOPILOT, frozenset)
    assert isinstance(LEGAL_HOOK_PARETO, frozenset)
    assert isinstance(LEGAL_HOOK_SENSITIVITY, frozenset)
    assert isinstance(LEGAL_HOOK_BIT_ALLOCATOR, frozenset)
    assert isinstance(LEGAL_HOOK_CONTINUAL_LEARNING, frozenset)


def test_contract_probe_disambiguator_none_requires_rationale():
    with pytest.raises(SearchContractError, match="hook_probe_disambiguator"):
        _minimal_contract(
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": "x",
                "hook_pareto_constraint": "x",
                "hook_bit_allocator_class": "x",
                # missing probe rationale
            },
        )


def test_contract_probe_disambiguator_non_string_rejected():
    with pytest.raises(SearchContractError, match="non-empty path"):
        _minimal_contract(hook_probe_disambiguator="")


# ---------------------------------------------------------------------------
# §2 @search_strategy decorator behavior tests
# ---------------------------------------------------------------------------


def test_decorator_registers_strategy():
    contract = _minimal_contract(id="decorator_test")

    @search_strategy(contract)
    def my_fn(objective_fn, *, seed=42, **_ignored):
        return SearchResult(
            strategy_id="decorator_test",
            best_params={"x": 0.0},
            best_score=0.0,
            n_evaluations=1,
        )

    assert "decorator_test" in get_registered_strategies()
    assert get_registered_strategies()["decorator_test"] is contract


def test_decorator_attaches_contract_to_function():
    contract = _minimal_contract(id="decorator_attach")

    @search_strategy(contract)
    def my_fn(objective_fn, *, seed=42, **_ignored):
        return SearchResult(
            strategy_id="decorator_attach",
            best_params={},
            best_score=0.0,
            n_evaluations=0,
        )

    assert my_fn.__search_strategy_contract__ is contract


def test_decorator_rejects_non_contract():
    with pytest.raises(SearchContractError, match="expects a SearchContract"):
        search_strategy("not_a_contract")  # type: ignore[arg-type]


def test_decorator_rejects_duplicate_id():
    c1 = _minimal_contract(id="dup", description="first")

    @search_strategy(c1)
    def fn1(*args, **kwargs):
        return SearchResult(strategy_id="dup", best_params={}, best_score=0.0, n_evaluations=0)

    c2 = _minimal_contract(id="dup", description="second")
    with pytest.raises(SearchContractError, match="Duplicate"):
        @search_strategy(c2)
        def fn2(*args, **kwargs):
            return SearchResult(strategy_id="dup", best_params={}, best_score=0.0, n_evaluations=0)


def test_decorator_idempotent_on_identity():
    c = _minimal_contract(id="idempotent")

    # First registration
    decorator = search_strategy(c)
    @decorator
    def fn(*args, **kwargs):
        return SearchResult(strategy_id="idempotent", best_params={}, best_score=0.0, n_evaluations=0)

    # Re-decorate with the SAME contract object (identity-equal) — must succeed
    @search_strategy(c)
    def fn2(*args, **kwargs):
        return SearchResult(strategy_id="idempotent", best_params={}, best_score=0.0, n_evaluations=0)

    assert "idempotent" in get_registered_strategies()


def test_decorator_rejects_non_callable_target():
    c = _minimal_contract(id="non_callable_test")
    with pytest.raises(SearchContractError, match="non-callable"):
        search_strategy(c)("not_a_function")  # type: ignore[arg-type]


def test_decorator_rolls_back_on_non_callable():
    c = _minimal_contract(id="rollback_test")
    with pytest.raises(SearchContractError):
        search_strategy(c)(42)  # type: ignore[arg-type]
    # Registry should not contain the failed registration
    assert "rollback_test" not in get_registered_strategies()


def test_decorator_determinism_violation_on_rng_param():
    c = _minimal_contract(id="det_viol", deterministic=True, seed=42)
    with pytest.raises(DeterminismViolation, match="randomness parameter"):
        @search_strategy(c)
        def bad_fn(objective_fn, *, rng=None, **_ignored):
            return SearchResult(strategy_id="det_viol", best_params={}, best_score=0.0, n_evaluations=0)


def test_decorator_seed_required_violation():
    c = _minimal_contract(id="seed_req", deterministic=True, seed=None)
    # Function signature has no `seed` param AND contract.seed is None
    with pytest.raises(SeedRequiredViolation, match="neither the contract nor"):
        @search_strategy(c)
        def bad_fn(objective_fn, **_ignored):
            return SearchResult(strategy_id="seed_req", best_params={}, best_score=0.0, n_evaluations=0)


def test_decorator_deterministic_with_seed_in_signature_ok():
    c = _minimal_contract(id="det_sig_ok", deterministic=True, seed=None)

    @search_strategy(c)
    def ok_fn(objective_fn, *, seed=0, **_ignored):
        return SearchResult(strategy_id="det_sig_ok", best_params={}, best_score=0.0, n_evaluations=0)

    assert "det_sig_ok" in get_registered_strategies()


def test_decorator_non_deterministic_skips_seed_check():
    c = _minimal_contract(id="non_det", deterministic=False, seed=None)

    @search_strategy(c)
    def fn(objective_fn, **_ignored):
        return SearchResult(strategy_id="non_det", best_params={}, best_score=0.0, n_evaluations=0)

    assert "non_det" in get_registered_strategies()


def test_get_strategy_function_returns_callable():
    c = _minimal_contract(id="get_fn")

    @search_strategy(c)
    def my_fn(objective_fn, *, seed=42, **_ignored):
        return SearchResult(strategy_id="get_fn", best_params={}, best_score=0.0, n_evaluations=0)

    fn = get_strategy_function("get_fn")
    assert fn is my_fn


def test_get_strategy_function_raises_on_missing():
    with pytest.raises(SearchStrategyNotRegisteredError):
        get_strategy_function("never_registered")


def test_validate_all_registered_strategies_clean():
    c = _minimal_contract(id="validate_clean")

    @search_strategy(c)
    def fn(objective_fn, *, seed=42, **_ignored):
        return SearchResult(strategy_id="validate_clean", best_params={}, best_score=0.0, n_evaluations=0)

    errors = validate_all_registered_strategies()
    assert errors == []


def test_validate_all_detects_back_door_mutation():
    from tac.search import _REGISTERED_STRATEGIES
    _REGISTERED_STRATEGIES["bogus"] = "not_a_contract"  # type: ignore[assignment]
    errors = validate_all_registered_strategies()
    assert any("bogus" in e for e in errors)


def test_validate_all_prune_corrupt_removes_bad_entries():
    from tac.search import _REGISTERED_STRATEGIES
    _REGISTERED_STRATEGIES["bogus"] = "not_a_contract"  # type: ignore[assignment]
    errors = validate_all_registered_strategies(prune_corrupt=True)
    assert any("bogus" in e for e in errors)
    assert "bogus" not in get_registered_strategies()


# ---------------------------------------------------------------------------
# §3 ComposableSearchPipeline composition tests
# ---------------------------------------------------------------------------


def _register_simple_strategy(strategy_id="s1", parent=None, score=0.5):
    c = _minimal_contract(id=strategy_id, parent_strategy_id=parent)

    @search_strategy(c)
    def fn(objective_fn, *, bounds=None, seed=42, warm_start=None, **_ignored):
        return SearchResult(
            strategy_id=strategy_id,
            best_params={"x": score},
            best_score=score,
            n_evaluations=1,
            seed=seed,
        )

    return c


def test_pipeline_empty_construction():
    p = ComposableSearchPipeline()
    assert p.strategies == ()
    assert str(p) == "ComposableSearchPipeline(<empty>)"


def test_pipeline_or_appends_strategy():
    _register_simple_strategy("s1")
    p = ComposableSearchPipeline() | "s1"
    assert len(p.strategies) == 1
    assert p.strategies[0].strategy_id == "s1"
    assert p.strategies[0].composition_kind == "sequential"


def test_pipeline_or_immutable_returns_new():
    _register_simple_strategy("s1")
    p1 = ComposableSearchPipeline()
    p2 = p1 | "s1"
    assert p1.strategies == ()
    assert len(p2.strategies) == 1


def test_pipeline_from_strategy_ids():
    _register_simple_strategy("s1")
    _register_simple_strategy("s2", parent="s1")
    p = ComposableSearchPipeline.from_strategy_ids(["s1", "s2"])
    assert len(p.strategies) == 2


def test_pipeline_to_json_roundtrip():
    _register_simple_strategy("s1")
    p = ComposableSearchPipeline() | "s1"
    p2 = ComposableSearchPipeline.from_dict(json.loads(p.to_json()))
    assert p2.strategies[0].strategy_id == "s1"


def test_pipeline_to_dict_includes_objective_label():
    p = ComposableSearchPipeline().with_shared_objective_label("test_label")
    d = p.to_dict()
    assert d["shared_objective_function_label"] == "test_label"


def test_pipeline_build_rejects_unknown_id():
    p = ComposableSearchPipeline() | "never_registered"
    with pytest.raises(SearchPipelineError, match="not registered"):
        p.build()


def test_pipeline_build_detects_parent_cycle():
    # Create two strategies referring to each other via parent
    c1 = _minimal_contract(id="cycle_a", parent_strategy_id="cycle_b")
    c2 = _minimal_contract(id="cycle_b", parent_strategy_id="cycle_a")

    @search_strategy(c1)
    def fn_a(objective_fn, *, seed=42, **_ignored):
        return SearchResult(strategy_id="cycle_a", best_params={}, best_score=0.0, n_evaluations=0, seed=seed)

    @search_strategy(c2)
    def fn_b(objective_fn, *, seed=42, **_ignored):
        return SearchResult(strategy_id="cycle_b", best_params={}, best_score=0.0, n_evaluations=0, seed=seed)

    p = ComposableSearchPipeline() | "cycle_a"
    with pytest.raises(SearchPipelineError, match="Cycle"):
        p.build()


def test_pipeline_run_invokes_strategy_with_objective():
    _register_simple_strategy("s1")
    p = ComposableSearchPipeline() | "s1"
    result = p.run(lambda params: 0.5)
    assert result.best_score == 0.5


def test_pipeline_run_aggregates_best_across_chain():
    _register_simple_strategy("s_a", score=0.7)
    _register_simple_strategy("s_b", parent="s_a", score=0.2)
    p = ComposableSearchPipeline() | "s_a" | "s_b"
    result = p.run(lambda params: 0.0)
    assert result.best_score == 0.2


def test_pipeline_run_empty_raises():
    p = ComposableSearchPipeline()
    with pytest.raises(SearchPipelineError, match="empty pipeline"):
        p.run(lambda params: 0.0)


def test_pipeline_run_rejects_non_callable_objective():
    _register_simple_strategy("s1")
    p = ComposableSearchPipeline() | "s1"
    with pytest.raises(ObjectiveFunctionError, match="must be callable"):
        p.run("not_a_function")  # type: ignore[arg-type]


def test_pipeline_and_requires_prior_strategy():
    _register_simple_strategy("s1")
    p = ComposableSearchPipeline()
    with pytest.raises(SearchPipelineError, match="requires at least one prior"):
        _ = p & "s1"


def test_pipeline_strategy_ref_to_dict_round_trip():
    ref = SearchPipelineStrategyRef(
        strategy_id="x", parameters=(("k", 1),), composition_kind="parallel"
    )
    d = ref.to_dict()
    ref2 = SearchPipelineStrategyRef.from_dict(d)
    assert ref == ref2


def test_pipeline_ambiguous_composition_detected():
    _register_simple_strategy("s_a")
    _register_simple_strategy("s_b")  # no parent — ambiguous chain
    p = ComposableSearchPipeline() | "s_a" | "s_b"
    with pytest.raises(SearchAmbiguousCompositionError, match="parent_strategy_id"):
        p.build()


# ---------------------------------------------------------------------------
# §4 SearchResult / SearchTrial / SearchHistory tests
# ---------------------------------------------------------------------------


def test_search_trial_to_dict():
    t = SearchTrial(trial_index=0, params={"x": 0.1}, score=0.5)
    assert t.to_dict() == {
        "trial_index": 0,
        "params": {"x": 0.1},
        "score": 0.5,
        "elapsed_seconds": 0.0,
        "score_axis": "[proxy]",
    }


def test_search_history_length_and_iter():
    h = SearchHistory(trials=(
        SearchTrial(trial_index=0, params={"x": 0}, score=1.0),
        SearchTrial(trial_index=1, params={"x": 1}, score=0.5),
    ))
    assert len(h) == 2
    assert list(h)[0].score == 1.0


def test_search_result_to_dict_complete():
    r = SearchResult(
        strategy_id="r",
        best_params={"x": 0.3},
        best_score=0.01,
        n_evaluations=10,
        elapsed_seconds=1.5,
        seed=42,
    )
    d = r.to_dict()
    assert d["strategy_id"] == "r"
    assert d["best_score"] == 0.01
    assert d["seed"] == 42


# ---------------------------------------------------------------------------
# §5 CMAESCandidateSearcher tests
# ---------------------------------------------------------------------------


def test_cma_es_spec_validation_rejects_empty_bounds():
    with pytest.raises(ValueError, match="non-empty mapping"):
        CMAESCandidateSearcherSpec(strategy_id="t", bounds={})


def test_cma_es_spec_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="low < high"):
        CMAESCandidateSearcherSpec(strategy_id="t", bounds={"x": (5.0, 1.0)})


def test_cma_es_spec_rejects_zero_population():
    with pytest.raises(ValueError, match="population_size"):
        CMAESCandidateSearcherSpec(
            strategy_id="t", bounds={"x": (0.0, 1.0)}, population_size=0
        )


def test_cma_es_spec_rejects_zero_sigma():
    with pytest.raises(ValueError, match="sigma_init"):
        CMAESCandidateSearcherSpec(
            strategy_id="t", bounds={"x": (0.0, 1.0)}, sigma_init=0.0
        )


def test_cma_es_build_contract_canonical():
    spec = CMAESCandidateSearcherSpec(
        strategy_id="cma_t", bounds={"x": (-1.0, 1.0)}
    )
    c = CMAESCandidateSearcher(spec=spec).build_contract()
    assert c.search_kind == "continuous"
    assert c.parallelism == "vectorized"
    assert c.deterministic is True


def test_cma_es_register_succeeds():
    spec = CMAESCandidateSearcherSpec(
        strategy_id="cma_register", bounds={"x": (-1.0, 1.0)}
    )
    contract = CMAESCandidateSearcher(spec=spec).register()
    assert contract.id == "cma_register"
    assert "cma_register" in get_registered_strategies()


def test_cma_es_run_raises_engine_not_installed_when_lib_absent():
    spec = CMAESCandidateSearcherSpec(
        strategy_id="cma_lib_absent", bounds={"x": (-1.0, 1.0)}
    )
    CMAESCandidateSearcher(spec=spec).register()
    fn = get_strategy_function("cma_lib_absent")
    # The `cma` library is intentionally NOT installed in this env per PV-5
    with pytest.raises(SearchEngineNotInstalledError, match="cma"):
        fn(lambda p: 0.0)


def test_cma_es_rejects_bad_spec_type():
    with pytest.raises(TypeError, match="CMAESCandidateSearcherSpec"):
        CMAESCandidateSearcher(spec="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §6 OptunaTPESampler tests
# ---------------------------------------------------------------------------


def test_optuna_spec_rejects_empty_bounds():
    with pytest.raises(ValueError, match="non-empty mapping"):
        OptunaTPESamplerSpec(strategy_id="t", bounds={})


def test_optuna_spec_rejects_bad_kind():
    with pytest.raises(ValueError, match="not in"):
        OptunaTPESamplerSpec(
            strategy_id="t", bounds={"x": ("invalid_kind", 0, 1)}
        )


def test_optuna_spec_accepts_all_legal_kinds():
    spec = OptunaTPESamplerSpec(
        strategy_id="t",
        bounds={
            "i": ("int", 0, 10),
            "f": ("float", 0.0, 1.0),
            "lf": ("log_float", 1e-3, 1.0),
            "c": ("categorical", ["a", "b", "c"]),
            "b": ("bool",),
        },
    )
    assert len(spec.bounds) == 5


def test_optuna_spec_rejects_zero_trials():
    with pytest.raises(ValueError, match="n_trials"):
        OptunaTPESamplerSpec(
            strategy_id="t", bounds={"x": ("float", 0.0, 1.0)}, n_trials=0
        )


def test_optuna_build_contract_kind_is_mixed():
    spec = OptunaTPESamplerSpec(
        strategy_id="opt_t", bounds={"x": ("float", 0.0, 1.0)}
    )
    c = OptunaTPESampler(spec=spec).build_contract()
    assert c.search_kind == "mixed"


def test_optuna_register_succeeds():
    spec = OptunaTPESamplerSpec(
        strategy_id="opt_register", bounds={"x": ("float", 0.0, 1.0)}
    )
    OptunaTPESampler(spec=spec).register()
    assert "opt_register" in get_registered_strategies()


def test_optuna_run_succeeds_when_lib_installed_or_raises_if_absent():
    """Optuna's lazy-import surface: raises SearchEngineNotInstalledError if
    `optuna` is missing; runs end-to-end if present.
    """
    spec = OptunaTPESamplerSpec(
        strategy_id="opt_run",
        bounds={"x": ("float", 0.0, 1.0)},
        n_trials=5,
        n_startup_trials=2,
        seed=42,
    )
    OptunaTPESampler(spec=spec).register()
    fn = get_strategy_function("opt_run")
    try:
        import optuna  # noqa: F401
        # Library available — run end-to-end; quiet optuna's INFO logger
        import logging
        logging.getLogger("optuna").setLevel(logging.WARNING)
        result = fn(lambda p: (p["x"] - 0.3) ** 2)
        assert result.n_evaluations == 5
        assert result.best_score >= 0.0
    except ImportError:
        # Library absent — must raise the typed error
        with pytest.raises(SearchEngineNotInstalledError, match="optuna"):
            fn(lambda p: 0.0)


def test_optuna_raises_search_engine_not_installed_when_lazy_import_fails(
    monkeypatch,
):
    """Independent of host env, verify the lazy-import branch raises the
    typed SearchEngineNotInstalledError when `optuna` is missing.
    """
    spec = OptunaTPESamplerSpec(
        strategy_id="opt_lazy_absent",
        bounds={"x": ("float", 0.0, 1.0)},
    )
    OptunaTPESampler(spec=spec).register()
    # Force ImportError on optuna import
    import sys
    monkeypatch.setitem(sys.modules, "optuna", None)
    fn = get_strategy_function("opt_lazy_absent")
    with pytest.raises(SearchEngineNotInstalledError, match="optuna"):
        fn(lambda p: 0.0)


# ---------------------------------------------------------------------------
# §7 BayesianOptimizationGP tests
# ---------------------------------------------------------------------------


def test_gp_bo_spec_rejects_zero_initial():
    with pytest.raises(ValueError, match="n_initial_points"):
        BayesianOptimizationGPSpec(
            strategy_id="t", bounds={"x": (0.0, 1.0)}, n_initial_points=0
        )


def test_gp_bo_spec_rejects_n_calls_le_initial():
    with pytest.raises(ValueError, match="n_calls"):
        BayesianOptimizationGPSpec(
            strategy_id="t",
            bounds={"x": (0.0, 1.0)},
            n_initial_points=10,
            n_calls=10,
        )


def test_gp_bo_spec_rejects_bad_acquisition():
    with pytest.raises(ValueError, match="acquisition_function"):
        BayesianOptimizationGPSpec(
            strategy_id="t",
            bounds={"x": (0.0, 1.0)},
            acquisition_function="INVALID",
        )


def test_gp_bo_build_contract_continuous():
    spec = BayesianOptimizationGPSpec(
        strategy_id="gp_t", bounds={"x": (-1.0, 1.0)}
    )
    c = BayesianOptimizationGP(spec=spec).build_contract()
    assert c.search_kind == "continuous"


def test_gp_bo_register_succeeds():
    spec = BayesianOptimizationGPSpec(
        strategy_id="gp_register", bounds={"x": (-1.0, 1.0)}
    )
    BayesianOptimizationGP(spec=spec).register()
    assert "gp_register" in get_registered_strategies()


def test_gp_bo_run_raises_engine_not_installed():
    spec = BayesianOptimizationGPSpec(
        strategy_id="gp_lib_absent", bounds={"x": (-1.0, 1.0)}
    )
    BayesianOptimizationGP(spec=spec).register()
    fn = get_strategy_function("gp_lib_absent")
    with pytest.raises(SearchEngineNotInstalledError, match="scikit-optimize"):
        fn(lambda p: 0.0)


# ---------------------------------------------------------------------------
# §8 MCTSCodebookSearcher tests (in-house — fully runnable)
# ---------------------------------------------------------------------------


def test_mcts_spec_rejects_empty_bounds():
    with pytest.raises(ValueError, match="non-empty mapping"):
        MCTSCodebookSearcherSpec(strategy_id="t", bounds={})


def test_mcts_spec_rejects_empty_choices():
    with pytest.raises(ValueError, match="non-empty"):
        MCTSCodebookSearcherSpec(strategy_id="t", bounds={"K": []})


def test_mcts_spec_rejects_zero_simulations():
    with pytest.raises(ValueError, match="max_simulations"):
        MCTSCodebookSearcherSpec(
            strategy_id="t", bounds={"K": [1, 2]}, max_simulations=0
        )


def test_mcts_spec_rejects_zero_uct_constant():
    with pytest.raises(ValueError, match="exploration_constant"):
        MCTSCodebookSearcherSpec(
            strategy_id="t",
            bounds={"K": [1, 2]},
            exploration_constant_c_uct=0.0,
        )


def test_mcts_build_contract_discrete():
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_t", bounds={"K": [4, 8, 16]}
    )
    c = MCTSCodebookSearcher(spec=spec).build_contract()
    assert c.search_kind == "discrete"
    assert c.parallelism == "serial"


def test_mcts_register_and_run_end_to_end():
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_e2e",
        bounds={"K": [4, 8, 16, 32]},
        max_simulations=20,
        seed=42,
    )
    MCTSCodebookSearcher(spec=spec).register()
    fn = get_strategy_function("mcts_e2e")
    # Objective: minimize K
    result = fn(lambda p: float(p["K"]))
    assert result.best_params["K"] == 4
    assert result.best_score == 4.0


def test_mcts_deterministic_with_same_seed():
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_det1",
        bounds={"K": [1, 2, 3, 4, 5], "L": [10, 20, 30]},
        max_simulations=20,
        seed=42,
    )
    MCTSCodebookSearcher(spec=spec).register()
    fn = get_strategy_function("mcts_det1")
    r1 = fn(lambda p: float(p["K"]) * float(p["L"]))
    r2 = fn(lambda p: float(p["K"]) * float(p["L"]))
    # Same seed → same trial sequence
    assert r1.history.trials == r2.history.trials


def test_mcts_run_with_warm_start():
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_warm",
        bounds={"K": [1, 2, 3]},
        max_simulations=10,
        seed=42,
    )
    MCTSCodebookSearcher(spec=spec).register()
    fn = get_strategy_function("mcts_warm")
    # Warm-start with K=1; should be the first trial
    result = fn(lambda p: float(p["K"]), warm_start={"K": 1})
    assert result.history.trials[0].params["K"] == 1


def test_mcts_objective_function_error_propagates():
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_obj_err",
        bounds={"K": [1, 2]},
        max_simulations=5,
        seed=42,
    )
    MCTSCodebookSearcher(spec=spec).register()
    fn = get_strategy_function("mcts_obj_err")

    def bad_obj(p):
        raise RuntimeError("boom")

    with pytest.raises(ObjectiveFunctionError, match="boom"):
        fn(bad_obj)


def test_mcts_rejects_nan_score():
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_nan",
        bounds={"K": [1, 2]},
        max_simulations=5,
        seed=42,
    )
    MCTSCodebookSearcher(spec=spec).register()
    fn = get_strategy_function("mcts_nan")
    with pytest.raises(ObjectiveFunctionError, match="non-finite"):
        fn(lambda p: float("nan"))


# ---------------------------------------------------------------------------
# §9 RashomonEnsembleCommittee tests (re-export integrity)
# ---------------------------------------------------------------------------


def test_rashomon_spec_rejects_empty_pool():
    with pytest.raises(ValueError, match="non-empty list"):
        RashomonEnsembleCommitteeSpec(
            strategy_id="t", candidate_pool=[]
        )


def test_rashomon_spec_rejects_missing_candidate_id():
    with pytest.raises(ValueError, match="candidate_id"):
        RashomonEnsembleCommitteeSpec(
            strategy_id="t", candidate_pool=[{"K": 8}]
        )


def test_rashomon_build_contract_canonical_decision():
    spec = RashomonEnsembleCommitteeSpec(
        strategy_id="rash_t",
        candidate_pool=[{"candidate_id": "c0"}],
    )
    c = RashomonEnsembleCommittee(spec=spec).build_contract()
    # The ONLY builder that ADOPTS canonical (per design memo §10)
    assert "ADOPT_CANONICAL_BECAUSE_SERVES" in c.canonical_vs_unique_decision


def test_rashomon_reuses_canonical_implementation_not_reimplemented():
    """Per the design memo §9 PV-4: the Rashomon ranker MUST be the
    canonical class from tac.autopilot_rudin_daubechies, not a copy.

    Verified by importing both and asserting identity at the class level.
    """
    from tac.autopilot_rudin_daubechies import RashomonEnsembleRanker as _canonical
    import tac.search.rashomon_ensemble_committee as wrapper_module
    # The wrapper module's body does `from tac.autopilot_rudin_daubechies import RashomonEnsembleRanker`
    # inside the runner — so we check the runner function uses it.
    import inspect
    src = inspect.getsource(wrapper_module._run_rashomon_committee)
    assert "from tac.autopilot_rudin_daubechies import RashomonEnsembleRanker" in src
    # And the wrapper module does NOT redefine the class
    assert not hasattr(wrapper_module, "RashomonEnsembleRanker")


def test_rashomon_register_and_run_end_to_end():
    spec = RashomonEnsembleCommitteeSpec(
        strategy_id="rash_e2e",
        candidate_pool=[
            {"candidate_id": f"c{i}", "K": 4 + i * 4}
            for i in range(4)
        ],
        ensemble_size=4,
        sparsity_target=2,
        integer_coefficient_bound=5,
    )
    RashomonEnsembleCommittee(spec=spec).register()
    fn = get_strategy_function("rash_e2e")
    # Objective returns ProxyPanel-like dict (with seg_p0 etc fields)
    def panel_extractor(candidate):
        return {
            "seg_p0": 0.01 * candidate["K"],
            "pose_p0": 0.001,
            "rate_p0": 0.1,
        }
    result = fn(panel_extractor)
    assert result.n_evaluations == 4
    assert "rashomon_consensus_score" in result.best_params


# ---------------------------------------------------------------------------
# §10 The `@` operator integration with tac.boosting tests
# ---------------------------------------------------------------------------


def test_matmul_integration_boosting_pipeline():
    """The `@` operator on tac.boosting's pipeline stores the descriptor
    and tac.search's run_search_over_pipeline executes it."""
    _register_simple_strategy("matmul_test", score=0.05)

    from tac.boosting import ComposableBoostingPipeline
    pipeline = ComposableBoostingPipeline() @ "matmul_test"
    assert pipeline.search_strategy_descriptor == "matmul_test"

    result = run_search_over_pipeline(pipeline, lambda p: 0.0)
    assert result.best_score == 0.05


def test_matmul_integration_compress_time_pipeline():
    """The `@` operator on tac.compress_time_optimization's pipeline works too."""
    _register_simple_strategy("matmul_compress", score=0.07)

    from tac.compress_time_optimization import ComposableCompressPipeline
    pipeline = ComposableCompressPipeline() @ "matmul_compress"
    assert pipeline.search_strategy_descriptor == "matmul_compress"

    result = run_search_over_pipeline(pipeline, lambda p: 0.0)
    assert result.best_score == 0.07


def test_run_search_over_pipeline_rejects_no_descriptor():
    """Sister pipeline without a @-attached descriptor must raise."""
    from tac.boosting import ComposableBoostingPipeline
    pipeline = ComposableBoostingPipeline()
    with pytest.raises(SearchPipelineError, match="search_strategy_descriptor"):
        run_search_over_pipeline(pipeline, lambda p: 0.0)


def test_run_search_over_pipeline_rejects_unregistered_strategy():
    from tac.boosting import ComposableBoostingPipeline
    pipeline = ComposableBoostingPipeline() @ "never_registered_in_search"
    with pytest.raises(SearchStrategyNotRegisteredError):
        run_search_over_pipeline(pipeline, lambda p: 0.0)


def test_run_search_over_pipeline_passes_bounds_through():
    contract = _minimal_contract(id="bounds_pass")

    @search_strategy(contract)
    def fn(objective_fn, *, bounds=None, seed=42, warm_start=None, **_ignored):
        assert bounds == {"x": (0.0, 1.0)}
        return SearchResult(
            strategy_id="bounds_pass",
            best_params={"x": 0.5},
            best_score=0.25,
            n_evaluations=1,
            seed=seed,
        )

    from tac.boosting import ComposableBoostingPipeline
    pipeline = ComposableBoostingPipeline() @ "bounds_pass"
    result = run_search_over_pipeline(
        pipeline, lambda p: 0.0, bounds={"x": (0.0, 1.0)}
    )
    assert result.best_score == 0.25


# ---------------------------------------------------------------------------
# §11 SeedRequiredViolation discipline tests
# ---------------------------------------------------------------------------


def test_seed_required_violation_when_signature_and_contract_both_missing():
    """Both function signature AND contract.seed are None → SeedRequiredViolation."""
    c = _minimal_contract(id="seed_v1", deterministic=True, seed=None)
    with pytest.raises(SeedRequiredViolation):
        @search_strategy(c)
        def fn(objective_fn, **_ignored):
            return SearchResult(strategy_id="seed_v1", best_params={}, best_score=0.0, n_evaluations=0)


def test_no_seed_violation_when_contract_pins_seed():
    """Contract.seed pinned → OK even if signature has no `seed` param."""
    c = _minimal_contract(id="seed_v2", deterministic=True, seed=42)

    @search_strategy(c)
    def fn(objective_fn, **_ignored):
        return SearchResult(strategy_id="seed_v2", best_params={}, best_score=0.0, n_evaluations=0)


def test_no_seed_violation_when_signature_has_seed():
    """Function signature has `seed` param → OK even if contract.seed is None."""
    c = _minimal_contract(id="seed_v3", deterministic=True, seed=None)

    @search_strategy(c)
    def fn(objective_fn, *, seed=0, **_ignored):
        return SearchResult(strategy_id="seed_v3", best_params={}, best_score=0.0, n_evaluations=0, seed=seed)


# ---------------------------------------------------------------------------
# §12 Search budget enforcement tests
# ---------------------------------------------------------------------------


def test_search_budget_field_pinned_at_contract():
    c = _minimal_contract(n_candidate_evaluations_max=50)
    assert c.n_candidate_evaluations_max == 50


def test_search_budget_exceeded_error_class_exists():
    """SearchBudgetExceededError is a typed exception, child of SearchPipelineError."""
    assert issubclass(SearchBudgetExceededError, SearchPipelineError)
    assert issubclass(SearchBudgetExceededError, SearchNamespaceError)


def test_mcts_budget_respected():
    """MCTS stops at max_simulations regardless of state-space size."""
    spec = MCTSCodebookSearcherSpec(
        strategy_id="mcts_budget",
        bounds={"K": list(range(100))},  # huge state space
        max_simulations=5,
        seed=42,
    )
    MCTSCodebookSearcher(spec=spec).register()
    fn = get_strategy_function("mcts_budget")
    result = fn(lambda p: float(p["K"]))
    assert result.n_evaluations == 5


# ---------------------------------------------------------------------------
# §13 Surrogate-vs-real-objective routing tests
# ---------------------------------------------------------------------------


def test_objective_is_surrogate_flag_propagates_to_result():
    c = _minimal_contract(
        id="surrogate_true", objective_is_surrogate=True
    )

    @search_strategy(c)
    def fn(objective_fn, *, seed=42, **_ignored):
        return SearchResult(
            strategy_id="surrogate_true",
            best_params={},
            best_score=0.0,
            n_evaluations=1,
            objective_is_surrogate=c.objective_is_surrogate,
            seed=seed,
        )

    result = fn(lambda p: 0.0)
    assert result.objective_is_surrogate is True


def test_objective_is_surrogate_default_false():
    c = _minimal_contract(id="surrogate_false")
    assert c.objective_is_surrogate is False


def test_rashomon_marks_objective_as_surrogate():
    """RashomonEnsembleCommittee uses a SLIM panel surrogate by construction."""
    spec = RashomonEnsembleCommitteeSpec(
        strategy_id="rash_surr",
        candidate_pool=[{"candidate_id": "c0"}],
    )
    c = RashomonEnsembleCommittee(spec=spec).build_contract()
    assert c.objective_is_surrogate is True


# ---------------------------------------------------------------------------
# §14 Persistence tests
# ---------------------------------------------------------------------------


def test_append_search_outcome_roundtrip(tmp_path: Path):
    path = tmp_path / "outcomes.jsonl"
    lock = tmp_path / "outcomes.jsonl.lock"
    record = {
        "strategy_id": "test_strat",
        "best_params": {"x": 0.1},
        "best_score": 0.5,
        "n_evaluations": 10,
    }
    append_search_outcome_locked(record, path=path, lock_path=lock)
    rows = load_search_outcomes(path)
    assert len(rows) == 1
    assert rows[0]["strategy_id"] == "test_strat"
    assert rows[0]["best_score"] == 0.5
    assert rows[0]["schema_version"] == SEARCH_STRATEGY_OUTCOMES_SCHEMA_VERSION


def test_load_search_outcomes_strict_raises_on_malformed(tmp_path: Path):
    path = tmp_path / "corrupt.jsonl"
    path.write_text('not valid json\n')
    with pytest.raises(SearchLedgerCorruptError):
        load_search_outcomes_strict(path)


def test_load_search_outcomes_lenient_skips_malformed(tmp_path: Path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"strategy_id": "x", "best_score": 1.0, "schema_version": "search_strategy_outcomes_v1"}\n'
        'not valid json\n'
    )
    rows = load_search_outcomes(path)
    assert len(rows) == 1


def test_append_rejects_nan_score(tmp_path: Path):
    path = tmp_path / "out.jsonl"
    lock = tmp_path / "out.jsonl.lock"
    with pytest.raises(SearchLedgerCorruptError, match="NaN"):
        append_search_outcome_locked(
            {"strategy_id": "x", "best_score": float("nan")},
            path=path, lock_path=lock,
        )


def test_query_outcomes_by_strategy_id(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    lock = tmp_path / "q.jsonl.lock"
    append_search_outcome_locked(
        {"strategy_id": "a", "best_score": 1.0}, path=path, lock_path=lock
    )
    append_search_outcome_locked(
        {"strategy_id": "b", "best_score": 0.5}, path=path, lock_path=lock
    )
    append_search_outcome_locked(
        {"strategy_id": "a", "best_score": 0.8}, path=path, lock_path=lock
    )
    a_rows = query_outcomes_by_strategy_id("a", path=path)
    assert len(a_rows) == 2
    assert latest_best_score_by_strategy("a", path=path) == 0.8
    assert latest_best_score_by_strategy("b", path=path) == 0.5
    assert latest_best_score_by_strategy("missing", path=path) is None


def test_query_outcomes_by_objective_label(tmp_path: Path):
    path = tmp_path / "obj.jsonl"
    lock = tmp_path / "obj.jsonl.lock"
    append_search_outcome_locked(
        {"strategy_id": "a", "best_score": 1.0, "objective_function_label": "x"},
        path=path, lock_path=lock,
    )
    append_search_outcome_locked(
        {"strategy_id": "a", "best_score": 0.5, "objective_function_label": "y"},
        path=path, lock_path=lock,
    )
    x_rows = query_outcomes_by_objective_label("x", path=path)
    assert len(x_rows) == 1
    assert x_rows[0]["best_score"] == 1.0


# ---------------------------------------------------------------------------
# §15 Example registry tests (post-clean smoke)
# ---------------------------------------------------------------------------


def test_examples_register_all_six():
    from tac.search.examples import register_example_searches
    contracts = register_example_searches()
    assert len(contracts) == 6
    registered = get_registered_strategies()
    for cid in (
        "random_search_baseline_example",
        "cma_es_over_x_example",
        "tpe_mixed_x_y_example",
        "gp_bo_over_x_example",
        "mcts_codebook_example",
        "rashomon_committee_example",
    ):
        assert cid in registered


def test_random_search_baseline_example_runs():
    from tac.search.examples import register_example_searches
    register_example_searches()
    fn = get_strategy_function("random_search_baseline_example")
    result = fn(_quadratic_objective(0.3), bounds={"x": (0.0, 1.0)})
    assert result.best_score < 0.1


def test_mcts_codebook_example_runs():
    from tac.search.examples import register_example_searches
    register_example_searches()
    fn = get_strategy_function("mcts_codebook_example")
    result = fn(lambda p: float(p["K"]))
    assert result.best_params["K"] == 4


def test_examples_idempotent_re_register():
    from tac.search.examples import register_example_searches
    c1 = register_example_searches()
    c2 = register_example_searches()
    assert len(c1) == len(c2) == 6


# ---------------------------------------------------------------------------
# §16 Error hierarchy + import-surface tests
# ---------------------------------------------------------------------------


def test_all_errors_inherit_from_search_namespace_error():
    for exc_class in [
        SearchContractError,
        DeterminismViolation,
        SeedRequiredViolation,
        SearchEngineNotInstalledError,
        SearchPipelineError,
        SearchAmbiguousCompositionError,
        SearchBudgetExceededError,
        ObjectiveFunctionError,
        SearchLedgerCorruptError,
        SearchStrategyNotRegisteredError,
    ]:
        assert issubclass(exc_class, SearchNamespaceError)


def test_legal_enum_membership():
    assert "continuous" in LEGAL_SEARCH_KIND
    assert "discrete" in LEGAL_SEARCH_KIND
    assert "mixed" in LEGAL_SEARCH_KIND
    assert "multi_objective" in LEGAL_SEARCH_KIND
    assert "serial" in LEGAL_PARALLELISM
    assert "vectorized" in LEGAL_PARALLELISM
    assert "process_pool" in LEGAL_PARALLELISM


def test_not_applicable_with_rationale_constant():
    assert NOT_APPLICABLE_WITH_RATIONALE == "not_applicable_with_rationale"
    assert NOT_APPLICABLE_WITH_RATIONALE in LEGAL_HOOK_SENSITIVITY
    assert NOT_APPLICABLE_WITH_RATIONALE in LEGAL_HOOK_PARETO
    assert NOT_APPLICABLE_WITH_RATIONALE in LEGAL_HOOK_BIT_ALLOCATOR
    assert NOT_APPLICABLE_WITH_RATIONALE in LEGAL_HOOK_AUTOPILOT
    assert NOT_APPLICABLE_WITH_RATIONALE in LEGAL_HOOK_CONTINUAL_LEARNING


# ---------------------------------------------------------------------------
# §17 Pipeline composition edge-case tests
# ---------------------------------------------------------------------------


def test_pipeline_warm_start_propagates_through_chain():
    """Sequential pipeline threads best_params as warm_start to next strategy."""
    contract_a = _minimal_contract(id="warm_a")

    @search_strategy(contract_a)
    def fn_a(objective_fn, *, bounds=None, seed=42, warm_start=None, **_ignored):
        return SearchResult(
            strategy_id="warm_a",
            best_params={"x": 0.99},
            best_score=0.99,
            n_evaluations=1,
            seed=seed,
        )

    seen_warm_start = {}
    contract_b = _minimal_contract(id="warm_b", parent_strategy_id="warm_a")

    @search_strategy(contract_b)
    def fn_b(objective_fn, *, bounds=None, seed=42, warm_start=None, **_ignored):
        seen_warm_start["got"] = warm_start
        return SearchResult(
            strategy_id="warm_b",
            best_params={"x": 0.01},
            best_score=0.01,
            n_evaluations=1,
            seed=seed,
        )

    p = ComposableSearchPipeline() | "warm_a" | "warm_b"
    p.run(lambda params: 0.0)
    assert seen_warm_start["got"] == {"x": 0.99}


def test_pipeline_shared_objective_label_propagates():
    contract = _minimal_contract(id="obj_label")

    @search_strategy(contract)
    def fn(objective_fn, *, seed=42, **_ignored):
        return SearchResult(
            strategy_id="obj_label",
            best_params={},
            best_score=0.0,
            n_evaluations=0,
            seed=seed,
        )

    p = (
        ComposableSearchPipeline()
        .with_shared_objective_label("my_label")
        | "obj_label"
    )
    result = p.run(lambda params: 0.0)
    assert result.objective_function_label == "my_label"


def test_pipeline_strategy_contracts_returns_in_order():
    _register_simple_strategy("contract_a")
    _register_simple_strategy("contract_b", parent="contract_a")
    p = ComposableSearchPipeline() | "contract_a" | "contract_b"
    contracts = p.strategy_contracts()
    assert [c.id for c in contracts] == ["contract_a", "contract_b"]


# ---------------------------------------------------------------------------
# §18 Live design-memo / contract regression guards (Catalog #305 / #294 / #303 / #290)
# ---------------------------------------------------------------------------


def test_design_memo_exists():
    """The canonical design memo must exist + carry the required headers."""
    memo = Path(__file__).resolve().parents[3] / ".omx" / "research" / (
        "tac_search_namespace_design_20260517.md"
    )
    assert memo.exists()
    body = memo.read_text(encoding="utf-8").lower()
    assert "## canonical-vs-unique decision per layer" in body
    assert "## 9-dimension success checklist evidence" in body
    assert "## cargo-cult audit per assumption" in body
    assert "## observability surface" in body


def test_premise_verifier_exists():
    pv = Path(__file__).resolve().parents[3] / ".omx" / "tmp" / (
        "tac_search_premise_verifier.txt"
    )
    assert pv.exists()
    body = pv.read_text(encoding="utf-8")
    # PV-1 through PV-16 sections
    for i in range(1, 17):
        assert f"PV-{i}" in body
