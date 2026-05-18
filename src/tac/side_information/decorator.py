# SPDX-License-Identifier: MIT
"""@side_info_baker decorator + in-memory baker registry.

Mirrors ``tac.boosting.decorator.boost_stage`` and
``tac.compress_time_optimization.decorator.compress_time_pass`` at the
side-info-baker surface — pass-through decorator that:

  1. Validates the contract via ``SideInfoBakerContract.__post_init__``.
  2. Inspects the wrapped function for determinism + seed-presence
     violations (per CLAUDE.md "Bit-level deconstruction" + Catalog #158).
  3. Refuses duplicate baker ids (the registry IS the deduplication layer
     per CLAUDE.md "Subagent coherence-by-default").
  4. Registers the baker into ``_REGISTERED_BAKERS`` keyed by ``id``.
  5. Returns the original callable unmodified (no runtime wrap; the baker
     function runs at full speed).

Adversarial-review-anchored discipline (mirroring tac.boosting K1 + Q2):
  - K1 anti-pattern: non-callable target ⇒ rollback registration + raise
    so a baker file cannot create a 'ghost' registry entry.
  - Q2 anti-pattern: out-of-band registry mutation ⇒
    ``validate_all_registered_bakers`` catches it.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" PV-7: this decorator does
NOT import tac.boosting.decorator or
tac.compress_time_optimization.decorator. The three namespaces have
independent registries.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from tac.side_information.contract import SideInfoBakerContract
from tac.side_information.errors import (
    SideInfoBakerContractError,
)

__all__ = [
    "_REGISTERED_BAKERS",
    "_clear_baker_registry_for_tests",
    "side_info_baker",
    "get_baker_function",
    "get_registered_bakers",
    "validate_all_registered_bakers",
]

# Module-level registry keyed by contract.id.
_REGISTERED_BAKERS: dict[str, SideInfoBakerContract] = {}

# Side registry: id → callable. Separate from the contract dict so the
# contract registry is JSON-serializable independently. Decorators populate
# both atomically.
_REGISTERED_BAKER_FUNCTIONS: dict[str, Callable[..., Any]] = {}


F = TypeVar("F", bound=Callable[..., Any])


# Note: side-information bakers MAY be non-deterministic in principle (a
# stochastic feature extractor could pick a random subset of input frames),
# but per CLAUDE.md "Bit-level deconstruction" non-negotiable + Catalog
# #158 deterministic-compiler discipline, bakers that contribute archive
# bytes MUST be deterministic. The contract-level cross-field invariant
# (archive_bytes_added>0 ⇒ deterministic=True) already enforces this at
# construction. The decorator's determinism check below catches the
# orthogonal case: a deterministic baker whose function signature includes
# randomness but no seed.
_FORBIDDEN_RANDOMNESS_PARAM_NAMES: frozenset[str] = frozenset(
    {"rng", "random_state", "noise", "random_generator", "torch_generator"}
)


def _check_determinism_invariant(
    fn: Callable[..., Any], contract: SideInfoBakerContract
) -> None:
    """Inspect the wrapped function for determinism violations.

    Per CLAUDE.md "Bit-level deconstruction" non-negotiable + Catalog #158
    deterministic-compiler discipline:

    If the contract claims ``deterministic=True`` and the function signature
    declares a forbidden randomness parameter (``rng`` / ``random_state`` /
    ``noise`` / etc.) WITHOUT an accompanying ``seed=`` parameter, raise
    ``SideInfoBakerContractError``. A deterministic baker MUST be
    reproducible — either pin seed on the contract OR accept seed as a
    kwarg.

    Builtins / C-extensions / lambdas with no inspectable signature pass
    through silently — ``inspect.signature`` raises ValueError in those
    cases and we tolerate it.
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
        raise SideInfoBakerContractError(
            f"Baker id={contract.id!r} declares deterministic=True but its "
            f"function signature includes randomness parameter(s) "
            f"{sorted(forbidden_present)!r} WITHOUT an accompanying 'seed=' "
            f"parameter. Per Catalog #158 deterministic-compiler discipline "
            f"+ CLAUDE.md 'Bit-level deconstruction' non-negotiable, "
            f"deterministic bakers MUST accept a 'seed=' kwarg so the side-"
            f"info bytes are reproducible. Either add 'seed' to the "
            f"signature OR set deterministic=False (which requires "
            f"archive_bytes_added==0)."
        )


def side_info_baker(
    contract: SideInfoBakerContract,
) -> Callable[[F], F]:
    """Register a side-information baker's contract into the namespace registry.

    Usage::

        from tac.side_information import (
            side_info_baker, SideInfoBakerContract,
        )

        @side_info_baker(SideInfoBakerContract(
            id="comma2k19_chroma_palette_k16",
            stage_phase="both",
            stage_order=1,
            consumes=frozenset({"comma2k19_chunks"}),
            emits=frozenset({"side_info_palette_v1"}),
            correction_kind="palette_distillation",
            correction_resolution="global",
            side_info_source="comma2k19_distilled",
            side_info_reproducible=True,
            requires_canonical_comma2k19_cache=True,
            wyner_ziv_correlation_estimate=0.42,
            archive_bytes_added=0,  # palette baked into inflate.py constant
            inflate_runtime_bytes_added=48,  # 16 chroma anchors * 3 bytes
            deterministic=True,
            seed=42,
            hook_sensitivity_contribution="not_applicable_with_rationale",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_sensitivity_contribution": "palette is a fixed prior; "
                                                  "sensitivity weighting N/A.",
                "hook_probe_disambiguator": "single canonical interpretation.",
            },
        ))
        def comma2k19_chroma_palette_k16(state, *, policy, seed=42):
            ...
            return {"side_info_palette_v1": palette_bytes}

    The decorator is pass-through (the wrapped function runs at full speed
    with no runtime overhead). The contract is captured at decoration time
    so consumers (Pipeline, persistence) can introspect without importing
    or executing the baker function.

    Raises:
        SideInfoBakerContractError: contract is not a SideInfoBakerContract
            / id collides with previously registered baker / non-callable
            target.
        NonReproducibleSideInfoViolation: side_info_reproducible=False
            (raised by contract validation; this decorator surfaces it).
        CanonicalComma2k19CacheRequiredViolation: requires_canonical_comma2k19_cache
            =True but the canonical helper is not importable in this
            runtime (raised by contract validation; this decorator surfaces
            it).
        WynerZivCorrelationInvalidError: wyner_ziv_correlation_estimate is
            outside [0.0, 1.0] or NaN/inf (raised by contract validation;
            this decorator surfaces it).
    """
    if not isinstance(contract, SideInfoBakerContract):
        raise SideInfoBakerContractError(
            f"side_info_baker expects a SideInfoBakerContract, got "
            f"{type(contract).__name__}"
        )

    existing = _REGISTERED_BAKERS.get(contract.id)
    if existing is not None and existing is not contract:
        raise SideInfoBakerContractError(
            f"Duplicate side-info baker id={contract.id!r}: already "
            f"registered with a different contract. If this is the same "
            f"module being re-imported, the registration is idempotent on "
            f"identity. If two bakers legitimately need the same name, "
            f"rename one."
        )

    fresh_registration = existing is None
    _REGISTERED_BAKERS[contract.id] = contract

    def _rollback_failed_registration() -> None:
        if fresh_registration:
            _REGISTERED_BAKERS.pop(contract.id, None)
            _REGISTERED_BAKER_FUNCTIONS.pop(contract.id, None)
        else:
            _REGISTERED_BAKERS[contract.id] = contract

    def _wrap(fn: F) -> F:
        if not callable(fn):
            # Roll back only a fresh registry write. A failed re-decoration
            # with the same contract must not erase a previously valid baker.
            _rollback_failed_registration()
            raise SideInfoBakerContractError(
                f"@side_info_baker({contract.id!r}) must decorate a "
                f"callable (function or class); got {type(fn).__name__}. "
                f"Per adversarial-review discipline (mirroring tac.boosting "
                f"K1 2026-05-15), the decorator refuses non-callable "
                f"targets so a baker file cannot create a 'ghost' registry "
                f"entry without an actual handler."
            )

        # Determinism check must run AFTER we have the callable. If it
        # fires we also roll back the registry write.
        try:
            _check_determinism_invariant(fn, contract)
        except SideInfoBakerContractError:
            _rollback_failed_registration()
            raise

        # Attach the contract to the function for introspection sugar
        try:
            fn.__side_info_baker_contract__ = contract  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Builtin / non-attribute-settable callable — still register.
            pass

        _REGISTERED_BAKER_FUNCTIONS[contract.id] = fn
        return fn

    return _wrap


def get_registered_bakers() -> dict[str, SideInfoBakerContract]:
    """Return a shallow copy of the baker-contract registry.

    Returns a copy so callers cannot mutate the registry through the
    returned dict; mutation requires invoking ``@side_info_baker`` again.
    """
    return dict(_REGISTERED_BAKERS)


def get_baker_function(baker_id: str) -> Callable[..., Any]:
    """Look up a registered baker's wrapped function by id.

    Raises:
        SideInfoBakerContractError: if no baker with the given id is
            registered.
    """
    fn = _REGISTERED_BAKER_FUNCTIONS.get(baker_id)
    if fn is None:
        raise SideInfoBakerContractError(
            f"No side-info baker registered with id={baker_id!r}. "
            f"Registered ids: {sorted(_REGISTERED_BAKERS)}"
        )
    return fn


def validate_all_registered_bakers(
    *, prune_corrupt: bool = False
) -> list[str]:
    """Re-run validation on every registered baker and return any errors.

    Defensive helper for test fixtures + preflight gates that want to
    confirm the registry's state is internally consistent (no contract has
    been frozen-then-mutated through a back door per Q2-style adversarial
    scenarios).

    Args:
        prune_corrupt: when True, ALSO remove any registry entry whose
            value is not a SideInfoBakerContract instance OR whose
            contract fails re-validation. Mirrors tac.boosting Q2
            discipline.

    Returns:
        list[str]: zero-length on success; otherwise per-id error strings.
    """
    errors: list[str] = []
    to_prune: list[str] = []
    for bid, contract in _REGISTERED_BAKERS.items():
        if not isinstance(contract, SideInfoBakerContract):
            errors.append(
                f"{bid}: registry value is {type(contract).__name__!r}, "
                f"not a SideInfoBakerContract (likely out-of-band mutation)"
            )
            to_prune.append(bid)
            continue
        try:
            SideInfoBakerContract(**contract.to_dict())
        except SideInfoBakerContractError as exc:
            errors.append(f"{bid}: {exc}")
            to_prune.append(bid)
        except TypeError as exc:
            errors.append(f"{bid}: contract construction failed: {exc}")
            to_prune.append(bid)
        except Exception as exc:  # pragma: no cover - defensive catch
            errors.append(f"{bid}: unexpected validation error: {exc}")
            to_prune.append(bid)
    if prune_corrupt:
        for bid in to_prune:
            _REGISTERED_BAKERS.pop(bid, None)
            _REGISTERED_BAKER_FUNCTIONS.pop(bid, None)
    return errors


def _clear_baker_registry_for_tests() -> None:
    """Test-only helper to clear both registries between fixture runs.

    Intended for pytest fixtures that want a clean slate. Production code
    MUST NOT call this.
    """
    _REGISTERED_BAKERS.clear()
    _REGISTERED_BAKER_FUNCTIONS.clear()
