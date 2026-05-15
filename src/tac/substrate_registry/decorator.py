"""@register_substrate decorator + in-memory registry.

Per `.omx/research/substrate_meta_layer_design_20260515.md`, the decorator:

  1. Validates the contract at import time (via ``SubstrateContract.__post_init__``).
  2. Registers the substrate into ``_REGISTERED_SUBSTRATES`` keyed by ``id``.
  3. Returns the original callable unmodified (pass-through; no runtime wrap).

Duplicate ids raise ``SubstrateContractError`` so accidental copy-paste between
substrate trainers fails loud at import time, not silently in a downstream
consumer.

Per CLAUDE.md "Subagent coherence-by-default" the registry IS the
deduplication layer for substrate ids — the equivalent of the lane registry's
duplicate-id refusal at the substrate level.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tac.substrate_registry.contract import (
    SubstrateContract,
    SubstrateContractError,
)

__all__ = [
    "_REGISTERED_SUBSTRATES",
    "_clear_registry_for_tests",
    "get_registered_substrates",
    "register_substrate",
    "validate_all_registered",
]

# Module-level registry. Keyed by ``SubstrateContract.id`` (which is
# guaranteed unique by the decorator).
_REGISTERED_SUBSTRATES: dict[str, SubstrateContract] = {}

F = TypeVar("F", bound=Callable[..., Any])


def register_substrate(contract: SubstrateContract) -> Callable[[F], F]:
    """Register a substrate's contract into the META layer registry.

    Usage::

        from tac.substrate_registry import register_substrate, SubstrateContract

        @register_substrate(SubstrateContract(id=..., ...))
        def main(argv: list[str] | None = None) -> int:
            ...

    The decorator is a pass-through — it does not wrap or modify the
    decorated callable. The contract is captured at decoration time so
    consumers can read it via ``get_registered_substrates()`` without
    importing or executing the trainer.

    Raises:
        SubstrateContractError: when the contract is not a SubstrateContract
            instance OR when ``contract.id`` collides with a previously-
            registered substrate.
    """
    if not isinstance(contract, SubstrateContract):
        raise SubstrateContractError(
            f"register_substrate expects a SubstrateContract, got {type(contract).__name__}"
        )

    existing = _REGISTERED_SUBSTRATES.get(contract.id)
    if existing is not None and existing is not contract:
        raise SubstrateContractError(
            f"Duplicate substrate id={contract.id!r}: already registered with a different contract. "
            "If this is the same module being re-imported, the registration is idempotent on identity. "
            "If two substrates legitimately need the same name, rename one."
        )

    _REGISTERED_SUBSTRATES[contract.id] = contract

    def _wrap(fn: F) -> F:
        # Pass-through; attach contract for introspection but do not modify
        # the call signature.
        # Adversarial-review finding K1 (2026-05-15): refuse decoration of
        # non-callable objects. The Carmack probe showed
        # ``register_substrate(contract)(42)`` silently registered a contract
        # with no actual training entry point ("ghost registration"). The
        # decorator now demands a callable so a substrate file that decorates
        # a constant or a module-level value fails loud at import time.
        # Classes are callables (instantiation), so ``@register_substrate``
        # on a class still works.
        if not callable(fn):
            # Roll back the registration we performed above to avoid a partial
            # registry write that survives the raise.
            registered = _REGISTERED_SUBSTRATES.get(contract.id)
            if registered is contract:
                _REGISTERED_SUBSTRATES.pop(contract.id, None)
            raise SubstrateContractError(
                f"@register_substrate({contract.id!r}) must decorate a callable "
                f"(function or class); got {type(fn).__name__}. Per "
                "adversarial-review finding K1 (2026-05-15) the decorator "
                "refuses non-callable targets so a substrate file cannot "
                "create a 'ghost' registry entry without an actual training "
                "entry point."
            )
        try:
            fn.__substrate_contract__ = contract  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Builtin / non-attribute-settable callables — still register the
            # contract; just skip the introspection sugar.
            pass
        return fn

    return _wrap


def get_registered_substrates() -> dict[str, SubstrateContract]:
    """Return a shallow copy of the registry (id → contract).

    Returns a copy so callers cannot mutate the registry through the
    returned dict; mutation requires calling ``register_substrate(...)``
    again with a fresh contract.
    """
    return dict(_REGISTERED_SUBSTRATES)


def validate_all_registered(*, prune_corrupt: bool = False) -> list[str]:
    """Re-run validation on every registered contract and return any errors.

    Each contract was validated at decoration time, but ``validate_all_registered``
    is a defensive helper for test fixtures + preflight gates that want to
    confirm the registry's state is internally consistent (e.g., no contract
    has been frozen-then-mutated through a back door).

    Args:
        prune_corrupt: when True, ALSO remove any registry entry whose value
            is not a ``SubstrateContract`` instance OR whose contract fails
            re-validation. Per adversarial-review finding Q2 (2026-05-15),
            corrupt registry rows propagate to the auto_wire query helpers
            as ``AttributeError``; preflight gates and operator workflows
            that want a clean registry can call this with ``prune_corrupt=True``.
            Default ``False`` preserves backward-compatible read-only behavior.

    Returns:
        list[str]: zero-length on success; otherwise a list of error strings
        formatted as ``"<id>: <error_message>"``.
    """
    errors: list[str] = []
    to_prune: list[str] = []
    for sid, contract in _REGISTERED_SUBSTRATES.items():
        # Adversarial-review finding Q2 (2026-05-15): the registry can be
        # mutated out-of-band (e.g., a test that injects a fake object).
        # Reject anything that isn't a SubstrateContract so the consumer
        # query helpers don't AttributeError on missing fields.
        if not isinstance(contract, SubstrateContract):
            errors.append(
                f"{sid}: registry value is {type(contract).__name__!r}, "
                "not a SubstrateContract (likely out-of-band mutation)"
            )
            to_prune.append(sid)
            continue
        try:
            # Reconstruct via to_dict→Contract roundtrip. This re-runs
            # __post_init__ validators against the current snapshot.
            SubstrateContract(**contract.to_dict())
        except SubstrateContractError as exc:
            errors.append(f"{sid}: {exc}")
            to_prune.append(sid)
        except TypeError as exc:
            # Missing-field / wrong-type errors from dataclass construction
            # surface here; wrap as contract-error string for consumers.
            errors.append(f"{sid}: contract construction failed: {exc}")
            to_prune.append(sid)
        except Exception as exc:  # pragma: no cover - defensive catch
            errors.append(f"{sid}: unexpected validation error: {exc}")
            to_prune.append(sid)
    if prune_corrupt:
        for sid in to_prune:
            _REGISTERED_SUBSTRATES.pop(sid, None)
    return errors


def _clear_registry_for_tests() -> None:
    """Test-only helper to clear the registry between fixture runs.

    Intended for use in pytest fixtures that want a clean slate. Production
    code MUST NOT call this.
    """
    _REGISTERED_SUBSTRATES.clear()
