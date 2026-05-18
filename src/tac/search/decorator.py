# SPDX-License-Identifier: MIT
"""@search_strategy decorator + in-memory strategy registry.

Mirrors ``tac.boosting.decorator.boost_stage`` /
``tac.compress_time_optimization.decorator.compress_time_pass`` at the
search-strategy surface — pass-through decorator that:

  1. Validates the contract via ``SearchContract.__post_init__``.
  2. Inspects the wrapped function for determinism + seed-presence
     violations (per CLAUDE.md "Bit-level deconstruction" + Catalog #158).
  3. Refuses duplicate strategy ids (the registry IS the deduplication
     layer per CLAUDE.md "Subagent coherence-by-default").
  4. Registers the strategy into ``_REGISTERED_STRATEGIES`` keyed by ``id``.
  5. Returns the original callable unmodified (no runtime wrap; the
     strategy function runs at full speed).

Adversarial-review-anchored discipline (mirroring tac.boosting K1 + Q2):
  - K1 anti-pattern: non-callable target ⇒ rollback registration + raise
    so a strategy file cannot create a 'ghost' registry entry.
  - Q2 anti-pattern: out-of-band registry mutation ⇒
    ``validate_all_registered_strategies`` catches it.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this decorator does
NOT import sister-namespace decorators. The three namespaces have
independent registries.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from tac.search.contract import SearchContract
from tac.search.errors import (
    DeterminismViolation,
    SearchContractError,
    SearchStrategyNotRegisteredError,
    SeedRequiredViolation,
)

__all__ = [
    "_REGISTERED_STRATEGIES",
    "_clear_strategy_registry_for_tests",
    "get_registered_strategies",
    "get_strategy_function",
    "search_strategy",
    "validate_all_registered_strategies",
]

# Module-level registry keyed by contract.id.
_REGISTERED_STRATEGIES: dict[str, SearchContract] = {}

# Side registry: id → callable. Decorators populate both atomically.
_REGISTERED_STRATEGY_FUNCTIONS: dict[str, Callable[..., Any]] = {}


F = TypeVar("F", bound=Callable[..., Any])


# Token sets used by the determinism auto-detector.
_FORBIDDEN_RANDOMNESS_PARAM_NAMES: frozenset[str] = frozenset(
    {"rng", "random_state", "noise", "random_generator", "torch_generator"}
)


def _check_determinism_invariant(
    fn: Callable[..., Any], contract: SearchContract
) -> None:
    """Inspect the wrapped function for determinism violations.

    Two distinct checks mirroring sister namespaces:

    A. If the contract claims ``deterministic=True`` and the function
       signature declares a forbidden randomness parameter without an
       accompanying ``seed=`` parameter, raise DeterminismViolation.

    B. If the contract claims ``deterministic=True`` and ``seed=None`` AND
       the function signature lacks a ``seed`` parameter, raise
       SeedRequiredViolation.

    Builtins / C-extensions / lambdas with no inspectable signature pass
    through silently — ``inspect.signature`` raises ValueError in those
    cases and we tolerate it.
    """
    if not contract.deterministic:
        return
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return
    param_names = set(sig.parameters)
    forbidden_present = param_names & _FORBIDDEN_RANDOMNESS_PARAM_NAMES
    has_seed_param = "seed" in param_names
    if forbidden_present and not has_seed_param:
        raise DeterminismViolation(
            f"Strategy id={contract.id!r} declares deterministic=True but "
            f"its function signature includes randomness parameter(s) "
            f"{sorted(forbidden_present)!r} WITHOUT an accompanying 'seed=' "
            "parameter. Per Catalog #158 deterministic-compiler discipline + "
            "CLAUDE.md 'Bit-level deconstruction' non-negotiable, "
            "deterministic strategies MUST accept a 'seed=' kwarg so the "
            "trial sequence is reproducible. Either add 'seed' to the "
            "signature OR set deterministic=False."
        )
    if not has_seed_param and contract.seed is None:
        if forbidden_present:
            return  # pragma: no cover — defensive
        raise SeedRequiredViolation(
            f"Strategy id={contract.id!r} declares deterministic=True but "
            "neither the contract nor the function signature pins a seed. "
            "Per Catalog #158 deterministic-compiler discipline, "
            "deterministic search strategies MUST either set contract.seed "
            "OR accept a 'seed=' kwarg with a stable default. This prevents "
            "search code from becoming non-reproducible by later adding "
            "hidden randomness."
        )


def search_strategy(
    contract: SearchContract,
) -> Callable[[F], F]:
    """Register a search strategy's contract into the namespace registry.

    Usage::

        from tac.search import search_strategy, SearchContract

        @search_strategy(SearchContract(
            id="random_search_baseline",
            search_kind="mixed",
            n_candidate_evaluations_max=100,
            parallelism="process_pool",
            seed=42,
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind="search_strategy_outcomes_v1",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution":
                    "RandomSearch is sensitivity-blind by design.",
                "hook_pareto_constraint":
                    "Single-objective; Pareto undefined.",
                "hook_bit_allocator_class":
                    "RandomSearch discovers parameter values, not bit allocations.",
                "hook_probe_disambiguator":
                    "Single canonical interpretation.",
            },
        ))
        def random_search_baseline(objective_fn, *, bounds, seed=42, ...):
            ...
            return SearchResult(...)

    The decorator is pass-through (the wrapped function runs at full
    speed). The contract is captured at decoration time so consumers
    (Pipeline, persistence) can introspect without importing or executing
    the strategy function.

    Raises:
        SearchContractError: contract is not a SearchContract / id collides
            with a previously registered strategy / non-callable target.
        DeterminismViolation: contract claims deterministic=True but the
            function signature includes randomness param without `seed=`.
        SeedRequiredViolation: contract claims deterministic=True with
            seed=None AND function signature lacks seed param.
    """
    if not isinstance(contract, SearchContract):
        raise SearchContractError(
            f"search_strategy expects a SearchContract, got "
            f"{type(contract).__name__}"
        )

    existing = _REGISTERED_STRATEGIES.get(contract.id)
    if existing is not None and existing is not contract:
        raise SearchContractError(
            f"Duplicate search strategy id={contract.id!r}: already "
            "registered with a different contract. If this is the same "
            "module being re-imported, the registration is idempotent on "
            "identity. If two strategies legitimately need the same name, "
            "rename one."
        )

    fresh_registration = existing is None
    _REGISTERED_STRATEGIES[contract.id] = contract

    def _rollback_failed_registration() -> None:
        if fresh_registration:
            _REGISTERED_STRATEGIES.pop(contract.id, None)
            _REGISTERED_STRATEGY_FUNCTIONS.pop(contract.id, None)
        else:
            _REGISTERED_STRATEGIES[contract.id] = contract

    def _wrap(fn: F) -> F:
        if not callable(fn):
            _rollback_failed_registration()
            raise SearchContractError(
                f"@search_strategy({contract.id!r}) must decorate a "
                f"callable (function or class); got {type(fn).__name__}. "
                "Per adversarial-review discipline (mirroring sister "
                "tac.boosting K1 2026-05-15), the decorator refuses "
                "non-callable targets so a strategy file cannot create a "
                "'ghost' registry entry without an actual handler."
            )

        try:
            _check_determinism_invariant(fn, contract)
        except (DeterminismViolation, SeedRequiredViolation):
            _rollback_failed_registration()
            raise

        # Attach the contract to the function for introspection sugar
        try:
            fn.__search_strategy_contract__ = contract  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

        _REGISTERED_STRATEGY_FUNCTIONS[contract.id] = fn
        return fn

    return _wrap


def get_registered_strategies() -> dict[str, SearchContract]:
    """Return a shallow copy of the strategy-contract registry.

    Returns a copy so callers cannot mutate the registry through the
    returned dict; mutation requires invoking ``@search_strategy`` again.
    """
    return dict(_REGISTERED_STRATEGIES)


def get_strategy_function(strategy_id: str) -> Callable[..., Any]:
    """Look up a registered strategy's wrapped function by id.

    Raises:
        SearchStrategyNotRegisteredError: if no strategy with the given id
            is registered.
    """
    fn = _REGISTERED_STRATEGY_FUNCTIONS.get(strategy_id)
    if fn is None:
        raise SearchStrategyNotRegisteredError(
            f"No search strategy registered with id={strategy_id!r}. "
            f"Registered ids: {sorted(_REGISTERED_STRATEGIES)}"
        )
    return fn


def validate_all_registered_strategies(
    *, prune_corrupt: bool = False
) -> list[str]:
    """Re-run validation on every registered strategy and return any errors.

    Defensive helper for test fixtures + preflight gates that want to
    confirm the registry's state is internally consistent (no contract has
    been frozen-then-mutated through a back door per Q2-style adversarial
    scenarios).

    Args:
        prune_corrupt: when True, ALSO remove any registry entry whose
            value is not a SearchContract instance OR whose contract fails
            re-validation. Mirrors tac.boosting Q2 discipline.

    Returns:
        list[str]: zero-length on success; otherwise per-id error strings.
    """
    errors: list[str] = []
    to_prune: list[str] = []
    for sid, contract in _REGISTERED_STRATEGIES.items():
        if not isinstance(contract, SearchContract):
            errors.append(
                f"{sid}: registry value is {type(contract).__name__!r}, "
                "not a SearchContract (likely out-of-band mutation)"
            )
            to_prune.append(sid)
            continue
        try:
            SearchContract(**contract.to_dict())
        except SearchContractError as exc:
            errors.append(f"{sid}: {exc}")
            to_prune.append(sid)
        except TypeError as exc:
            errors.append(f"{sid}: contract construction failed: {exc}")
            to_prune.append(sid)
        except Exception as exc:  # pragma: no cover - defensive catch
            errors.append(f"{sid}: unexpected validation error: {exc}")
            to_prune.append(sid)
    if prune_corrupt:
        for sid in to_prune:
            _REGISTERED_STRATEGIES.pop(sid, None)
            _REGISTERED_STRATEGY_FUNCTIONS.pop(sid, None)
    return errors


def _clear_strategy_registry_for_tests() -> None:
    """Test-only helper to clear both registries between fixture runs.

    Intended for pytest fixtures that want a clean slate. Production code
    MUST NOT call this.
    """
    _REGISTERED_STRATEGIES.clear()
    _REGISTERED_STRATEGY_FUNCTIONS.clear()
