# SPDX-License-Identifier: MIT
"""@boost_stage decorator + in-memory stage registry.

Mirrors ``tac.substrate_registry.decorator.register_substrate`` at the
boost-stage surface — pass-through decorator that:

  1. Validates the contract via ``BoostStageContract.__post_init__``.
  2. Inspects the wrapped function for determinism + scorer-freedom
     violations (per CLAUDE.md "Strict scorer rule" + Catalog #158).
  3. Refuses duplicate stage ids (the registry IS the deduplication layer
     per CLAUDE.md "Subagent coherence-by-default").
  4. Registers the stage into ``_REGISTERED_STAGES`` keyed by ``id``.
  5. Returns the original callable unmodified (no runtime wrap; the stage
     function runs at full speed).

Adversarial-review-anchored discipline (mirroring SubstrateContract K1 +
Q2):
  - K1 anti-pattern: non-callable target ⇒ rollback registration + raise
    so a stage file cannot create a 'ghost' registry entry.
  - Q2 anti-pattern: out-of-band registry mutation ⇒ ``validate_all_registered``
    catches it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from tac.boosting.contract import BoostStageContract
from tac.boosting.errors import (
    BoostStageContractError,
    DeterminismViolation,
    ScorerFreedomViolation,
)

__all__ = [
    "_REGISTERED_STAGES",
    "_clear_stage_registry_for_tests",
    "boost_stage",
    "get_registered_stages",
    "get_stage_function",
    "validate_all_registered_stages",
]

# Module-level registry keyed by contract.id.
_REGISTERED_STAGES: dict[str, BoostStageContract] = {}

# Side registry: id → callable. Separate from the contract dict so the
# contract registry is JSON-serializable independently. Decorators populate
# both atomically.
_REGISTERED_STAGE_FUNCTIONS: dict[str, Callable[..., Any]] = {}


F = TypeVar("F", bound=Callable[..., Any])


# Token sets used by the determinism + scorer-freedom auto-detectors.
_FORBIDDEN_RANDOMNESS_PARAM_NAMES: frozenset[str] = frozenset(
    {"rng", "random_state", "noise", "random_generator", "torch_generator"}
)
_FORBIDDEN_SCORER_CONSUMES_TOKENS: frozenset[str] = frozenset(
    {
        "segnet",
        "posenet",
        "scorer_state",
        "scorer_weights",
        "segnet_state_dict",
        "posenet_state_dict",
    }
)


def _check_determinism_invariant(
    fn: Callable[..., Any], contract: BoostStageContract
) -> None:
    """Inspect the wrapped function for determinism violations.

    If the contract claims ``deterministic=True`` and the function signature
    declares a forbidden randomness parameter (``rng`` / ``random_state`` /
    ``noise`` / etc.) WITHOUT an accompanying ``seed=`` parameter, raise
    DeterminismViolation at decoration time so the contract / implementation
    drift is caught before any dispatch.

    Builtins / C-extensions / lambdas with no inspectable signature pass
    through silently — ``inspect.signature`` raises ValueError in those
    cases and we tolerate it.
    """
    if not contract.deterministic:
        return
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Non-inspectable callable; tolerate (the runtime will surface any
        # determinism issue empirically).
        return
    param_names = set(sig.parameters)
    forbidden_present = param_names & _FORBIDDEN_RANDOMNESS_PARAM_NAMES
    if forbidden_present and "seed" not in param_names:
        raise DeterminismViolation(
            f"Stage id={contract.id!r} declares deterministic=True but its "
            f"function signature includes randomness parameter(s) "
            f"{sorted(forbidden_present)!r} WITHOUT an accompanying 'seed=' "
            f"parameter. Per Catalog #158 deterministic-compiler discipline + "
            f"CLAUDE.md 'Bit-level deconstruction' non-negotiable, deterministic "
            f"stages MUST accept a 'seed=' kwarg so byte-identical output is "
            f"reproducible. Either add 'seed' to the signature OR set "
            f"deterministic=False (which forbids stage_phase='archive_build')."
        )


def _check_scorer_freedom_invariant(contract: BoostStageContract) -> None:
    """Inspect the contract's `consumes` set for scorer-dependency tokens.

    If the contract declares ``stage_phase='inflate'`` AND ``consumes``
    references any forbidden scorer-state token, raise ScorerFreedomViolation
    at decoration time. Per CLAUDE.md "Strict scorer rule" non-negotiable.

    This is a STRUCTURAL check at the contract surface; the per-line scanner
    that catches scorer-at-inflate in the source code itself is sister-owned
    by Catalog #6 (`check_no_scorer_load_at_inflate`).
    """
    if contract.stage_phase != "inflate":
        return
    forbidden_present: set[str] = set()
    for token in contract.consumes:
        lower = token.lower()
        for forbidden in _FORBIDDEN_SCORER_CONSUMES_TOKENS:
            if forbidden in lower:
                forbidden_present.add(token)
                break
    if forbidden_present:
        raise ScorerFreedomViolation(
            f"Stage id={contract.id!r} declares stage_phase='inflate' but "
            f"consumes scorer-state token(s) {sorted(forbidden_present)!r}. "
            f"Per CLAUDE.md 'Strict scorer rule' non-negotiable + Catalog #6 "
            f"(`check_no_scorer_load_at_inflate`), inflate-time stages MUST NOT "
            f"depend on PoseNet/SegNet/scorer weights. Loading scorers at "
            f"inflate destroys the rate term and produces non-compliant "
            f"artifacts. Move the scorer-dependent computation to stage_phase="
            f"'compress' OR replace the dependency with a deterministic "
            f"image-domain proxy."
        )


def boost_stage(contract: BoostStageContract) -> Callable[[F], F]:
    """Register a boost stage's contract into the namespace registry.

    Usage::

        from tac.boosting import boost_stage, BoostStageContract

        @boost_stage(BoostStageContract(
            id="cascade_pose_residual_v1",
            parent_stage_id="raw_decoder",
            stage_phase="compress",
            consumes=frozenset({"frames_v0", "predicted_distortion_v0"}),
            emits=frozenset({"residual_correction_v1", "frames_v1"}),
            correction_kind="additive",
            correction_resolution="per_pair",
            deterministic=True,
            max_bytes_added=512,
            hook_sensitivity_contribution="master_gradient_v1",
            hook_probe_disambiguator=None,
            hook_not_applicable_rationale={
                "hook_probe_disambiguator": "single canonical residual cascade; "
                                            "no defensible alternative interpretation."
            },
        ))
        def cascade_pose_residual(state, *, policy):
            ...
            return state.replace(frames_v1=..., residual_correction_v1=...)

    The decorator is pass-through (the wrapped function runs at full speed
    with no runtime overhead). The contract is captured at decoration time
    so consumers (Pipeline, ParetoFrontTracker, persistence) can introspect
    without importing or executing the stage function.

    Raises:
        BoostStageContractError: contract is not a BoostStageContract / id
            collides with previously registered stage / non-callable target.
        DeterminismViolation: contract claims deterministic=True but function
            signature includes randomness param without `seed=`.
        ScorerFreedomViolation: stage_phase='inflate' AND consumes references
            scorer-state tokens.
    """
    if not isinstance(contract, BoostStageContract):
        raise BoostStageContractError(
            f"boost_stage expects a BoostStageContract, got "
            f"{type(contract).__name__}"
        )

    # Scorer-freedom check fires BEFORE the registry write so a forbidden
    # contract never gets a registry slot.
    _check_scorer_freedom_invariant(contract)

    existing = _REGISTERED_STAGES.get(contract.id)
    if existing is not None and existing is not contract:
        raise BoostStageContractError(
            f"Duplicate boost stage id={contract.id!r}: already registered "
            f"with a different contract. If this is the same module being "
            f"re-imported, the registration is idempotent on identity. If two "
            f"stages legitimately need the same name, rename one."
        )

    _REGISTERED_STAGES[contract.id] = contract

    def _wrap(fn: F) -> F:
        if not callable(fn):
            # Rollback the registry write so a partial registration does NOT
            # survive the raise (mirrors substrate_registry K1 discipline).
            registered = _REGISTERED_STAGES.get(contract.id)
            if registered is contract:
                _REGISTERED_STAGES.pop(contract.id, None)
            raise BoostStageContractError(
                f"@boost_stage({contract.id!r}) must decorate a callable "
                f"(function or class); got {type(fn).__name__}. Per "
                f"adversarial-review discipline (mirroring SubstrateContract "
                f"K1 2026-05-15), the decorator refuses non-callable targets "
                f"so a stage file cannot create a 'ghost' registry entry "
                f"without an actual handler."
            )

        # Determinism check must run AFTER we have the callable. If it fires
        # we also roll back the registry write.
        try:
            _check_determinism_invariant(fn, contract)
        except DeterminismViolation:
            registered = _REGISTERED_STAGES.get(contract.id)
            if registered is contract:
                _REGISTERED_STAGES.pop(contract.id, None)
            raise

        # Attach the contract to the function for introspection sugar
        try:
            fn.__boost_stage_contract__ = contract  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Builtin / non-attribute-settable callable — still register.
            pass

        _REGISTERED_STAGE_FUNCTIONS[contract.id] = fn
        return fn

    return _wrap


def get_registered_stages() -> dict[str, BoostStageContract]:
    """Return a shallow copy of the stage-contract registry.

    Returns a copy so callers cannot mutate the registry through the
    returned dict; mutation requires invoking ``@boost_stage`` again.
    """
    return dict(_REGISTERED_STAGES)


def get_stage_function(stage_id: str) -> Callable[..., Any]:
    """Look up a registered stage's wrapped function by id.

    Raises:
        BoostStageContractError: if no stage with the given id is registered.
    """
    fn = _REGISTERED_STAGE_FUNCTIONS.get(stage_id)
    if fn is None:
        raise BoostStageContractError(
            f"No boost stage registered with id={stage_id!r}. Registered ids: "
            f"{sorted(_REGISTERED_STAGES)}"
        )
    return fn


def validate_all_registered_stages(*, prune_corrupt: bool = False) -> list[str]:
    """Re-run validation on every registered stage and return any errors.

    Defensive helper for test fixtures + preflight gates that want to confirm
    the registry's state is internally consistent (no contract has been
    frozen-then-mutated through a back door per Q2-style adversarial scenarios).

    Args:
        prune_corrupt: when True, ALSO remove any registry entry whose value
            is not a BoostStageContract instance OR whose contract fails
            re-validation. Mirrors substrate_registry Q2 discipline.

    Returns:
        list[str]: zero-length on success; otherwise per-id error strings.
    """
    errors: list[str] = []
    to_prune: list[str] = []
    for sid, contract in _REGISTERED_STAGES.items():
        if not isinstance(contract, BoostStageContract):
            errors.append(
                f"{sid}: registry value is {type(contract).__name__!r}, "
                f"not a BoostStageContract (likely out-of-band mutation)"
            )
            to_prune.append(sid)
            continue
        try:
            BoostStageContract(**contract.to_dict())
        except BoostStageContractError as exc:
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
            _REGISTERED_STAGES.pop(sid, None)
            _REGISTERED_STAGE_FUNCTIONS.pop(sid, None)
    return errors


def _clear_stage_registry_for_tests() -> None:
    """Test-only helper to clear both registries between fixture runs.

    Intended for pytest fixtures that want a clean slate. Production code
    MUST NOT call this.
    """
    _REGISTERED_STAGES.clear()
    _REGISTERED_STAGE_FUNCTIONS.clear()
    _REGISTERED_STAGE_FUNCTIONS.clear()
