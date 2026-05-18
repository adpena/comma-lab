# SPDX-License-Identifier: MIT
"""@inflate_time_post_filter decorator + in-memory pass registry.

Mirrors ``tac.compress_time_optimization.decorator.compress_time_pass`` at
the inflate-time-pass surface — pass-through decorator that:

  1. Validates the contract via ``InflateTimePostProcessingContract.__post_init__``.
  2. Inspects the wrapped function for determinism + seed-presence violations
     (per CLAUDE.md "Bit-level deconstruction" + Catalog #158).
  3. Refuses duplicate pass ids (the registry IS the deduplication layer per
     CLAUDE.md "Subagent coherence-by-default").
  4. Registers the pass into ``_REGISTERED_PASSES`` keyed by ``id``.
  5. Returns the original callable unmodified (no runtime wrap; the pass
     function runs at full speed).

Adversarial-review-anchored discipline (mirroring tac.boosting K1 + Q2 +
tac.compress_time_optimization sister rules):
  - K1 anti-pattern: non-callable target ⇒ rollback registration + raise so
    a pass file cannot create a 'ghost' registry entry.
  - Q2 anti-pattern: out-of-band registry mutation ⇒
    ``validate_all_registered_passes`` catches it.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this decorator does
NOT import tac.compress_time_optimization.decorator. The three sister
namespaces (tac.boosting / tac.compress_time_optimization /
tac.inflate_time_post_processing) have independent registries.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from tac.inflate_time_post_processing.contract import (
    InflateTimePostProcessingContract,
)
from tac.inflate_time_post_processing.errors import (
    InflateTimePassContractError,
    SeedRequiredViolation,
)

__all__ = [
    "_REGISTERED_PASSES",
    "_clear_pass_registry_for_tests",
    "get_pass_function",
    "get_registered_passes",
    "inflate_time_post_filter",
    "validate_all_registered_passes",
]

# Module-level registry keyed by contract.id.
_REGISTERED_PASSES: dict[str, InflateTimePostProcessingContract] = {}

# Side registry: id → callable. Separate from the contract dict so the
# contract registry is JSON-serializable independently. Decorators populate
# both atomically.
_REGISTERED_PASS_FUNCTIONS: dict[str, Callable[..., Any]] = {}


F = TypeVar("F", bound=Callable[..., Any])


# Token sets used by the determinism auto-detector. Same names as sister
# namespaces (rng/random_state/noise/etc.).
_FORBIDDEN_RANDOMNESS_PARAM_NAMES: frozenset[str] = frozenset(
    {"rng", "random_state", "noise", "random_generator", "torch_generator"}
)


def _check_determinism_invariant(
    fn: Callable[..., Any], contract: InflateTimePostProcessingContract
) -> None:
    """Inspect the wrapped function for determinism violations.

    Inflate-time passes MUST be deterministic per the contract invariant
    (deterministic=False raises at decoration). This check verifies the
    function signature is consistent with the deterministic contract:

    A. If the function signature declares a forbidden randomness parameter
       (``rng`` / ``random_state`` / ``noise`` / etc.) WITHOUT an
       accompanying ``seed=`` parameter, raise SeedRequiredViolation.

    B. If the contract pins ``seed=None`` AND the function signature lacks
       a ``seed`` parameter, raise SeedRequiredViolation. A deterministic
       stage MUST either pin a seed on the contract OR accept seed as a
       kwarg so the byte stream is reproducible.

    Builtins / C-extensions / lambdas with no inspectable signature pass
    through silently — ``inspect.signature`` raises ValueError in those
    cases and we tolerate it.
    """
    # deterministic=False is already refused by the contract; defensive guard:
    if not contract.deterministic:
        return  # pragma: no cover - defensive
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return
    param_names = set(sig.parameters)
    forbidden_present = param_names & _FORBIDDEN_RANDOMNESS_PARAM_NAMES
    has_seed_param = "seed" in param_names
    if forbidden_present and not has_seed_param:
        raise SeedRequiredViolation(
            f"Pass id={contract.id!r} declares deterministic=True but its "
            f"function signature includes randomness parameter(s) "
            f"{sorted(forbidden_present)!r} WITHOUT an accompanying 'seed=' "
            f"parameter. Per Catalog #158 deterministic-compiler discipline + "
            f"CLAUDE.md 'Bit-level deconstruction' non-negotiable, "
            f"deterministic inflate passes MUST accept a 'seed=' kwarg so "
            f"byte-identical frame output is reproducible. Either add 'seed' "
            f"to the signature OR remove the randomness parameter."
        )
    if not has_seed_param and contract.seed is None:
        if forbidden_present:  # already raised above; defensive
            return  # pragma: no cover
        raise SeedRequiredViolation(
            f"Pass id={contract.id!r} declares deterministic=True but neither "
            f"the contract nor the function signature pins a seed. Per "
            f"Catalog #158 deterministic-compiler discipline, deterministic "
            f"inflate-time passes MUST either set contract.seed or accept a "
            f"'seed=' kwarg with a stable default. This prevents inflate "
            f"frames from becoming non-reproducible by later adding hidden "
            f"randomness."
        )


def inflate_time_post_filter(
    contract: InflateTimePostProcessingContract,
) -> Callable[[F], F]:
    """Register an inflate-time post-processing pass into the namespace
    registry.

    Usage::

        from tac.inflate_time_post_processing import (
            inflate_time_post_filter, InflateTimePostProcessingContract,
        )

        @inflate_time_post_filter(InflateTimePostProcessingContract(
            id="bilateral_denoise_per_frame",
            stage_phase="inflate",
            stage_order=1,
            consumes=frozenset({"frames_v0"}),
            emits=frozenset({"frames_v1"}),
            correction_kind="denoise",
            correction_resolution="per_frame",
            applies_to_frames="all",
            deterministic=True,
            scorer_free=True,
            max_wallclock_seconds=60.0,
            inflate_compute_budget_seconds=1800.0,
            archive_bytes_added=0,
            score_axis_affected=("seg",),
            requires_scorer_surrogate=False,
            requires_cpu_only=True,
            seed=42,
            hook_pareto_constraint="inflate_wallclock_envelope_v1",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": (
                    "Per-frame bilateral filter does not consume "
                    "sensitivity gradients; the operator opted for a uniform "
                    "image-domain prior."
                ),
                "hook_bit_allocator_class": (
                    "Inflate-time passes do not allocate bits; the archive "
                    "byte budget was already spent at compress time."
                ),
                "hook_autopilot_ranker": (
                    "Initial wire-in landing; ranker integration deferred to "
                    "a follow-on subagent slot."
                ),
                "hook_probe_disambiguator": (
                    "Single canonical bilateral kernel; no defensible "
                    "alternative interpretation at the pass level."
                ),
            },
        ))
        def bilateral_denoise_per_frame(state, *, policy, seed=42):
            ...
            return {"frames_v1": ...}

    The decorator is pass-through (the wrapped function runs at full speed
    with no runtime overhead). The contract is captured at decoration time
    so consumers (Pipeline, persistence) can introspect without importing
    or executing the pass function.

    Raises:
        InflateTimePassContractError: contract is not an
            InflateTimePostProcessingContract / id collides with previously
            registered pass / non-callable target.
        CompressPhaseForbiddenError: stage_phase='compress' or sister
            forbidden values (raised in contract validation; this decorator
            surfaces it).
        ScorerAccessForbiddenError: scorer_free=False (raised in contract
            validation; surfaced here).
        ArchiveBytesViolation: archive_bytes_added > 0 (raised in contract
            validation; surfaced here).
        WallclockBudgetRequiredError: max_wallclock_seconds is None
            (REQUIRED per spec §G; surfaced from contract).
        SeedRequiredViolation: contract claims deterministic=True with
            seed=None AND function signature lacks seed param.
    """
    if not isinstance(contract, InflateTimePostProcessingContract):
        raise InflateTimePassContractError(
            f"inflate_time_post_filter expects an "
            f"InflateTimePostProcessingContract, got "
            f"{type(contract).__name__}"
        )

    existing = _REGISTERED_PASSES.get(contract.id)
    if existing is not None and existing is not contract:
        raise InflateTimePassContractError(
            f"Duplicate inflate-time pass id={contract.id!r}: already "
            f"registered with a different contract. If this is the same "
            f"module being re-imported, the registration is idempotent on "
            f"identity. If two passes legitimately need the same name, "
            f"rename one."
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
            _rollback_failed_registration()
            raise InflateTimePassContractError(
                f"@inflate_time_post_filter({contract.id!r}) must decorate a "
                f"callable (function or class); got {type(fn).__name__}. Per "
                f"adversarial-review discipline (mirroring tac.boosting K1 "
                f"2026-05-15), the decorator refuses non-callable targets so "
                f"a pass file cannot create a 'ghost' registry entry without "
                f"an actual handler."
            )

        # Determinism check must run AFTER we have the callable. If it fires
        # we also roll back the registry write.
        try:
            _check_determinism_invariant(fn, contract)
        except SeedRequiredViolation:
            _rollback_failed_registration()
            raise

        # Attach the contract to the function for introspection sugar.
        try:
            fn.__inflate_time_post_filter_contract__ = contract  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

        _REGISTERED_PASS_FUNCTIONS[contract.id] = fn
        return fn

    return _wrap


def get_registered_passes() -> dict[str, InflateTimePostProcessingContract]:
    """Return a shallow copy of the pass-contract registry.

    Returns a copy so callers cannot mutate the registry through the
    returned dict; mutation requires invoking ``@inflate_time_post_filter``
    again.
    """
    return dict(_REGISTERED_PASSES)


def get_pass_function(pass_id: str) -> Callable[..., Any]:
    """Look up a registered pass's wrapped function by id.

    Raises:
        InflateTimePassContractError: if no pass with the given id is
            registered.
    """
    fn = _REGISTERED_PASS_FUNCTIONS.get(pass_id)
    if fn is None:
        raise InflateTimePassContractError(
            f"No inflate-time pass registered with id={pass_id!r}. "
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
            is not an InflateTimePostProcessingContract instance OR whose
            contract fails re-validation. Mirrors tac.boosting Q2
            discipline.

    Returns:
        list[str]: zero-length on success; otherwise per-id error strings.
    """
    errors: list[str] = []
    to_prune: list[str] = []
    for pid, contract in _REGISTERED_PASSES.items():
        if not isinstance(contract, InflateTimePostProcessingContract):
            errors.append(
                f"{pid}: registry value is {type(contract).__name__!r}, "
                f"not an InflateTimePostProcessingContract (likely "
                f"out-of-band mutation)"
            )
            to_prune.append(pid)
            continue
        try:
            InflateTimePostProcessingContract(**contract.to_dict())
        except InflateTimePassContractError as exc:
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
