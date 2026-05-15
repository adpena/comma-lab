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

from typing import Any, Callable, TypeVar

from tac.substrate_registry.contract import (
    SubstrateContract,
    SubstrateContractError,
)

__all__ = [
    "register_substrate",
    "get_registered_substrates",
    "validate_all_registered",
    "_REGISTERED_SUBSTRATES",
    "_clear_registry_for_tests",
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


def validate_all_registered() -> list[str]:
    """Re-run validation on every registered contract and return any errors.

    Each contract was validated at decoration time, but ``validate_all_registered``
    is a defensive helper for test fixtures + preflight gates that want to
    confirm the registry's state is internally consistent (e.g., no contract
    has been frozen-then-mutated through a back door).

    Returns:
        list[str]: zero-length on success; otherwise a list of error strings
        formatted as ``"<id>: <error_message>"``.
    """
    errors: list[str] = []
    for sid, contract in _REGISTERED_SUBSTRATES.items():
        try:
            # Reconstruct via to_dict→Contract roundtrip. This re-runs
            # __post_init__ validators against the current snapshot.
            SubstrateContract(**contract.to_dict())
        except SubstrateContractError as exc:
            errors.append(f"{sid}: {exc}")
        except TypeError as exc:
            # Missing-field / wrong-type errors from dataclass construction
            # surface here; wrap as contract-error string for consumers.
            errors.append(f"{sid}: contract construction failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive catch
            errors.append(f"{sid}: unexpected validation error: {exc}")
    return errors


def _clear_registry_for_tests() -> None:
    """Test-only helper to clear the registry between fixture runs.

    Intended for use in pytest fixtures that want a clean slate. Production
    code MUST NOT call this.
    """
    _REGISTERED_SUBSTRATES.clear()
