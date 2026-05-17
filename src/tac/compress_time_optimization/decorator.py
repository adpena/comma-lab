# SPDX-License-Identifier: MIT
"""@compress_time_pass decorator + in-memory pass registry.

Mirrors ``tac.boosting.decorator.boost_stage`` at the compress-time-pass
surface — pass-through decorator that:

  1. Validates the contract via ``CompressTimePassContract.__post_init__``.
  2. Inspects the wrapped function for determinism + seed-presence violations
     (per CLAUDE.md "Bit-level deconstruction" + Catalog #158).
  3. Refuses duplicate pass ids (the registry IS the deduplication layer per
     CLAUDE.md "Subagent coherence-by-default").
  4. Registers the pass into ``_REGISTERED_PASSES`` keyed by ``id``.
  5. Returns the original callable unmodified (no runtime wrap; the pass
     function runs at full speed).

Adversarial-review-anchored discipline (mirroring tac.boosting K1 + Q2):
  - K1 anti-pattern: non-callable target ⇒ rollback registration + raise so
    a pass file cannot create a 'ghost' registry entry.
  - Q2 anti-pattern: out-of-band registry mutation ⇒
    ``validate_all_registered_passes`` catches it.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this decorator does NOT
import tac.boosting.decorator. The two namespaces have independent registries.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from tac.compress_time_optimization.contract import CompressTimePassContract
from tac.compress_time_optimization.errors import (
    CompressTimePassContractError,
    DeterminismViolation,
    SeedRequiredViolation,
)

__all__ = [
    "_REGISTERED_PASSES",
    "_clear_pass_registry_for_tests",
    "compress_time_pass",
    "get_pass_function",
    "get_registered_passes",
    "validate_all_registered_passes",
]

# Module-level registry keyed by contract.id.
_REGISTERED_PASSES: dict[str, CompressTimePassContract] = {}

# Side registry: id → callable. Separate from the contract dict so the
# contract registry is JSON-serializable independently. Decorators populate
# both atomically.
_REGISTERED_PASS_FUNCTIONS: dict[str, Callable[..., Any]] = {}


F = TypeVar("F", bound=Callable[..., Any])


# Token sets used by the determinism auto-detector.
_FORBIDDEN_RANDOMNESS_PARAM_NAMES: frozenset[str] = frozenset(
    {"rng", "random_state", "noise", "random_generator", "torch_generator"}
)


def _check_determinism_invariant(
    fn: Callable[..., Any], contract: CompressTimePassContract
) -> None:
    """Inspect the wrapped function for determinism violations.

    Two distinct checks per CLAUDE.md "Bit-level deconstruction" + Catalog
    #158 sister discipline:

    A. If the contract claims ``deterministic=True`` and the function signature
       declares a forbidden randomness parameter (``rng`` / ``random_state`` /
       ``noise`` / etc.) WITHOUT an accompanying ``seed=`` parameter, raise
       DeterminismViolation.

    B. If the contract claims ``deterministic=True`` and ``seed=None`` AND the
       function signature lacks a ``seed`` parameter, raise SeedRequiredViolation.
       A deterministic stage MUST either pin a seed on the contract OR accept
       seed as a kwarg so the byte stream is reproducible.

    Builtins / C-extensions / lambdas with no inspectable signature pass
    through silently — ``inspect.signature`` raises ValueError in those cases
    and we tolerate it.
    """
    if not contract.deterministic:
        return
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Non-inspectable callable; tolerate.
        return
    param_names = set(sig.parameters)
    forbidden_present = param_names & _FORBIDDEN_RANDOMNESS_PARAM_NAMES
    has_seed_param = "seed" in param_names
    if forbidden_present and not has_seed_param:
        raise DeterminismViolation(
            f"Pass id={contract.id!r} declares deterministic=True but its "
            f"function signature includes randomness parameter(s) "
            f"{sorted(forbidden_present)!r} WITHOUT an accompanying 'seed=' "
            f"parameter. Per Catalog #158 deterministic-compiler discipline + "
            f"CLAUDE.md 'Bit-level deconstruction' non-negotiable, deterministic "
            f"passes MUST accept a 'seed=' kwarg so byte-identical output is "
            f"reproducible. Either add 'seed' to the signature OR set "
            f"deterministic=False (which forbids stage_phase='archive_build')."
        )
    # B-check: deterministic + no seed on contract + no seed in signature.
    # The error class existed before this check was enforced, which made the
    # namespace look more deterministic than it was. Keep the rule simple:
    # deterministic passes either pin contract.seed or expose seed= in the
    # callable signature. Pure deterministic fixtures can use seed=0 default.
    if not has_seed_param and contract.seed is None:
        if forbidden_present:
            # Already handled by the earlier raise.
            return  # pragma: no cover - defensive
        raise SeedRequiredViolation(
            f"Pass id={contract.id!r} declares deterministic=True but neither "
            f"the contract nor the function signature pins a seed. Per Catalog "
            f"#158 deterministic-compiler discipline, deterministic "
            f"compress-time passes MUST either set contract.seed or accept a "
            f"'seed=' kwarg with a stable default. This prevents archive-build "
            f"and compress-time search code from becoming non-reproducible by "
            f"later adding hidden randomness."
        )


def compress_time_pass(
    contract: CompressTimePassContract,
) -> Callable[[F], F]:
    """Register a compress-time pass's contract into the namespace registry.

    Usage::

        from tac.compress_time_optimization import (
            compress_time_pass, CompressTimePassContract,
        )

        @compress_time_pass(CompressTimePassContract(
            id="sensitivity_weighted_tto_refinement",
            stage_phase="compress",
            stage_order=2,
            consumes=frozenset({"archive_bytes_v0", "master_gradient"}),
            emits=frozenset({"archive_bytes_v1"}),
            correction_kind="refinement",
            correction_resolution="per_byte",
            deterministic=True,
            sensitivity_weighted=True,
            max_wallclock_seconds=None,
            seed=42,
            hook_sensitivity_contribution="master_gradient_v1",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_probe_disambiguator": "single canonical refinement; "
                                            "no defensible alternative interpretation.",
            },
        ))
        def refine_quant_by_sensitivity(state, *, master_gradient, policy, seed=42):
            ...
            return {"archive_bytes_v1": ...}

    The decorator is pass-through (the wrapped function runs at full speed
    with no runtime overhead). The contract is captured at decoration time
    so consumers (Pipeline, persistence) can introspect without importing
    or executing the pass function.

    Raises:
        CompressTimePassContractError: contract is not a CompressTimePassContract
            / id collides with previously registered pass / non-callable target.
        InflatePhaseForbiddenError: stage_phase='inflate' (raised in contract
            validation; this decorator surfaces it).
        DeterminismViolation: contract claims deterministic=True but function
            signature includes randomness param without `seed=`.
        SeedRequiredViolation: contract claims deterministic=True with seed=None
            AND function signature lacks seed param.
    """
    if not isinstance(contract, CompressTimePassContract):
        raise CompressTimePassContractError(
            f"compress_time_pass expects a CompressTimePassContract, got "
            f"{type(contract).__name__}"
        )

    existing = _REGISTERED_PASSES.get(contract.id)
    if existing is not None and existing is not contract:
        raise CompressTimePassContractError(
            f"Duplicate compress-time pass id={contract.id!r}: already "
            f"registered with a different contract. If this is the same module "
            f"being re-imported, the registration is idempotent on identity. "
            f"If two passes legitimately need the same name, rename one."
        )

    fresh_registration = existing is None
    _REGISTERED_PASSES[contract.id] = contract

    def _rollback_failed_registration() -> None:
        if fresh_registration:
            _REGISTERED_PASSES.pop(contract.id, None)
            _REGISTERED_PASS_FUNCTIONS.pop(contract.id, None)
        else:
            _REGISTERED_PASSES[contract.id] = contract

    def _wrap(fn: F) -> F:
        if not callable(fn):
            # Roll back only a fresh registry write. A failed re-decoration
            # with the same contract must not erase a previously valid pass.
            _rollback_failed_registration()
            raise CompressTimePassContractError(
                f"@compress_time_pass({contract.id!r}) must decorate a callable "
                f"(function or class); got {type(fn).__name__}. Per "
                f"adversarial-review discipline (mirroring tac.boosting K1 "
                f"2026-05-15), the decorator refuses non-callable targets so "
                f"a pass file cannot create a 'ghost' registry entry without "
                f"an actual handler."
            )

        # Determinism check must run AFTER we have the callable. If it fires
        # we also roll back the registry write.
        try:
            _check_determinism_invariant(fn, contract)
        except (DeterminismViolation, SeedRequiredViolation):
            _rollback_failed_registration()
            raise

        # Attach the contract to the function for introspection sugar
        try:
            fn.__compress_time_pass_contract__ = contract  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Builtin / non-attribute-settable callable — still register.
            pass

        _REGISTERED_PASS_FUNCTIONS[contract.id] = fn
        return fn

    return _wrap


def get_registered_passes() -> dict[str, CompressTimePassContract]:
    """Return a shallow copy of the pass-contract registry.

    Returns a copy so callers cannot mutate the registry through the
    returned dict; mutation requires invoking ``@compress_time_pass`` again.
    """
    return dict(_REGISTERED_PASSES)


def get_pass_function(pass_id: str) -> Callable[..., Any]:
    """Look up a registered pass's wrapped function by id.

    Raises:
        CompressTimePassContractError: if no pass with the given id is
            registered.
    """
    fn = _REGISTERED_PASS_FUNCTIONS.get(pass_id)
    if fn is None:
        raise CompressTimePassContractError(
            f"No compress-time pass registered with id={pass_id!r}. "
            f"Registered ids: {sorted(_REGISTERED_PASSES)}"
        )
    return fn


def validate_all_registered_passes(
    *, prune_corrupt: bool = False
) -> list[str]:
    """Re-run validation on every registered pass and return any errors.

    Defensive helper for test fixtures + preflight gates that want to
    confirm the registry's state is internally consistent (no contract has
    been frozen-then-mutated through a back door per Q2-style adversarial
    scenarios).

    Args:
        prune_corrupt: when True, ALSO remove any registry entry whose value
            is not a CompressTimePassContract instance OR whose contract
            fails re-validation. Mirrors tac.boosting Q2 discipline.

    Returns:
        list[str]: zero-length on success; otherwise per-id error strings.
    """
    errors: list[str] = []
    to_prune: list[str] = []
    for pid, contract in _REGISTERED_PASSES.items():
        if not isinstance(contract, CompressTimePassContract):
            errors.append(
                f"{pid}: registry value is {type(contract).__name__!r}, "
                f"not a CompressTimePassContract (likely out-of-band mutation)"
            )
            to_prune.append(pid)
            continue
        try:
            CompressTimePassContract(**contract.to_dict())
        except CompressTimePassContractError as exc:
            errors.append(f"{pid}: {exc}")
            to_prune.append(pid)
        except TypeError as exc:
            errors.append(f"{pid}: contract construction failed: {exc}")
            to_prune.append(pid)
        except Exception as exc:  # pragma: no cover - defensive catch
            errors.append(f"{pid}: unexpected validation error: {exc}")
            to_prune.append(pid)
    if prune_corrupt:
        for pid in to_prune:
            _REGISTERED_PASSES.pop(pid, None)
            _REGISTERED_PASS_FUNCTIONS.pop(pid, None)
    return errors


def _clear_pass_registry_for_tests() -> None:
    """Test-only helper to clear both registries between fixture runs.

    Intended for pytest fixtures that want a clean slate. Production code
    MUST NOT call this.
    """
    _REGISTERED_PASSES.clear()
    _REGISTERED_PASS_FUNCTIONS.clear()
