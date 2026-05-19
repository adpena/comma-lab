# SPDX-License-Identifier: MIT
"""Canonical contract for cathedral autopilot auto-ingested consumers.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125
6-hook wire-in + operator directive 2026-05-19 *"What if we change the
paradigm by making cathedral autopilot ingest by default if within a certain
directory and exposing/respecting a certain contract or schema. Fix
permanently and self protect against"*.

Every package in ``src/tac/cathedral_consumers/`` MUST expose this contract
OR carry ``# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>`` waiver in
``__init__.py`` first 30 lines per Catalog #335 STRICT preflight gate.

Sister of:
- Catalog #265 ``check_symposium_impls_canonical_contract`` (same canonical
  contract pattern at the symposium-impl surface)
- Catalog #125 6-hook wire-in non-negotiable (this contract IS the
  structural extinction of hook #4 cathedral autopilot dispatch)
- ``tac.atom.linguistic_extensions`` + ``tac.council_continual_learning``
  (both expose ``update_from_anchor`` as the canonical contract token)

The contract is intentionally minimal:
1. Module-level metadata (CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS)
2. ``update_from_anchor(anchor)`` — Catalog #125 hook #5 (continual-learning posterior)
3. ``consume_candidate(candidate) -> dict`` — Catalog #125 hook #4 (cathedral dispatch)

Validation is structural (importable, satisfies Protocol, fields well-typed).
Runtime correctness is the consumer's responsibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


WAIVER_TOKEN = "CATHEDRAL_CONSUMER_DEFERRED_OK"
"""Canonical same-line waiver token for non-compliant consumer packages.

Format: ``# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>`` in __init__.py first
30 lines. Placeholder rationales (``<rationale>`` / ``<reason>`` literals,
empty, or <4 chars) rejected per Catalog #287 sister discipline.
"""

# Placeholder waiver rationale literals refused per Catalog #287 sister.
_PLACEHOLDER_RATIONALES = ("<rationale>", "<reason>", "rationale", "reason")
_MIN_RATIONALE_LEN = 4
_WAIVER_LOOKBACK_LINES = 30


class HookNumber(IntEnum):
    """Catalog #125 6-hook wire-in surfaces.

    Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, every
    landing must declare which of the 6 canonical hooks it consumes (or
    explicitly mark N/A with rationale).
    """

    SENSITIVITY_MAP = 1
    PARETO_CONSTRAINT = 2
    BIT_ALLOCATOR = 3
    CATHEDRAL_AUTOPILOT_DISPATCH = 4
    CONTINUAL_LEARNING_POSTERIOR = 5
    PROBE_DISAMBIGUATOR = 6


@dataclass(frozen=True)
class ConsumerRegistration:
    """Canonical registration record for an auto-ingested consumer.

    Emitted by :func:`validate_consumer_module` after successful contract
    verification. Consumed by
    ``tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers``
    to populate the ranker cascade.
    """

    consumer_name: str
    consumer_version: str  # semver-like; not strictly enforced
    consumer_hook_numbers: tuple[HookNumber, ...]
    consumer_module_path: str
    contract_compliant: bool
    waiver_rationale: str | None = None
    waiver_active: bool = False
    validation_errors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Frozen-dataclass invariants.
        if not self.consumer_name or not isinstance(self.consumer_name, str):
            raise ValueError("consumer_name must be a non-empty string")
        if not isinstance(self.consumer_hook_numbers, tuple):
            raise ValueError("consumer_hook_numbers must be a tuple")
        for hook in self.consumer_hook_numbers:
            if not isinstance(hook, HookNumber):
                raise ValueError(
                    f"consumer_hook_numbers entries must be HookNumber, got {type(hook).__name__}"
                )
        if self.waiver_active and not self.waiver_rationale:
            raise ValueError(
                "waiver_active=True requires non-empty waiver_rationale"
            )
        if self.waiver_active and self.contract_compliant:
            raise ValueError(
                "waiver_active=True is incompatible with contract_compliant=True "
                "(waiver is the explicit non-compliance escape)"
            )


@runtime_checkable
class CathedralConsumerContract(Protocol):
    """Canonical Protocol every auto-ingested cathedral consumer must satisfy.

    Per Catalog #265 canonical contract pattern (sister of
    ``tac.atom.linguistic_extensions`` + ``tac.council_continual_learning``).

    The 3 module-level fields plus 2 callable surfaces are the minimum the
    auto-discovery loop needs to register + invoke the consumer.

    Consumers MAY expose additional attributes; the contract is structurally
    minimal so future hooks can extend without breaking the canonical surface.
    """

    CONSUMER_NAME: str
    CONSUMER_VERSION: str
    CONSUMER_HOOK_NUMBERS: tuple[HookNumber, ...]

    @staticmethod
    def update_from_anchor(anchor: Any) -> None:
        """Continual-learning posterior hook (Catalog #125 hook #5).

        Called when a new empirical anchor (contest-CUDA / contest-CPU /
        diagnostic) is appended to the canonical posterior store. Consumer
        updates its internal state (e.g. refits a SLIM ranker, updates a
        Rashomon ensemble member, recomputes a sensitivity prior).

        Per CLAUDE.md "Apples-to-apples evidence discipline": the anchor's
        evidence_grade / axis_tag / hardware_substrate must be honored;
        consumers may not silently promote diagnostic to contest-grade.
        """
        ...

    @staticmethod
    def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        """Cathedral autopilot dispatch hook (Catalog #125 hook #4).

        Called per-candidate by the autopilot ranker cascade. Returns the
        consumer's contribution to ranking as a dict with at minimum:

        - ``predicted_delta_adjustment``: float (additive to candidate's
          predicted score delta; bounded; non-NaN)
        - ``rationale``: str (≥4 chars; human-readable why)
        - ``axis_tag``: str (one of ``[contest-CPU]`` / ``[contest-CUDA]`` /
          ``[diagnostic-CPU]`` / ``[diagnostic-CUDA]`` / ``[predicted]`` /
          ``[advisory only]`` per CLAUDE.md "Apples-to-apples")

        Optional fields the autopilot loop honors:
        - ``promotable``: bool (default False; True only with paired-axis evidence)
        - ``provenance``: dict (Catalog #323 canonical Provenance payload)
        - ``confidence``: float in [0, 1]
        """
        ...


class CathedralConsumerContractError(Exception):
    """Raised when a package fails to satisfy the canonical contract.

    Includes explicit field-by-field rationale so the operator can
    distinguish missing CONSUMER_NAME from wrong-type CONSUMER_HOOK_NUMBERS
    from missing update_from_anchor without re-reading the source.
    """


# Same-line waiver detection regex. Mirrors Catalog #287 / #303 / #305 pattern:
# the gate's docstring example token (``CATHEDRAL_CONSUMER_DEFERRED_OK``) cannot
# self-waive because placeholder rationales are rejected.
_WAIVER_PATTERN = re.compile(
    r"#\s*" + re.escape(WAIVER_TOKEN) + r":\s*(?P<rationale>[^\n]+)"
)


def discover_waiver_in_init(init_path: Path) -> tuple[str | None, bool]:
    """Read first 30 lines of ``__init__.py`` looking for canonical waiver.

    Returns ``(rationale, active)`` where:
    - ``rationale`` is the raw rationale string (or None if no waiver line)
    - ``active`` is True only if rationale is well-formed (non-placeholder,
      ≥4 chars, non-empty after strip)

    Per Catalog #287 sister discipline: placeholder rationales
    (``<rationale>`` / ``<reason>`` / empty / <4 chars) reject the waiver so
    the gate's docstring example cannot self-waive.
    """
    if not init_path.exists() or not init_path.is_file():
        return (None, False)
    try:
        with init_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for i, line in enumerate(fh):
                if i >= _WAIVER_LOOKBACK_LINES:
                    break
                lines.append(line)
    except OSError:
        return (None, False)
    text = "".join(lines)
    match = _WAIVER_PATTERN.search(text)
    if match is None:
        return (None, False)
    raw_rationale = match.group("rationale").strip()
    # Strip trailing comment characters / quotes / etc.
    rationale = raw_rationale.rstrip("'\"`*")
    if not rationale:
        return (raw_rationale, False)
    if rationale.lower() in _PLACEHOLDER_RATIONALES:
        return (raw_rationale, False)
    if len(rationale) < _MIN_RATIONALE_LEN:
        return (raw_rationale, False)
    return (rationale, True)


def _validate_field(
    module: Any,
    field_name: str,
    expected_type: type | tuple[type, ...],
    errors: list[str],
) -> bool:
    """Helper: check a module-level field exists + is correctly typed."""
    if not hasattr(module, field_name):
        errors.append(f"missing module-level field: {field_name}")
        return False
    value = getattr(module, field_name)
    if not isinstance(value, expected_type):
        type_names = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else "/".join(t.__name__ for t in expected_type)
        )
        errors.append(
            f"{field_name} must be {type_names}, got {type(value).__name__}"
        )
        return False
    return True


def validate_consumer_module(
    module: Any, *, module_path: str | None = None
) -> ConsumerRegistration:
    """Verify a module implements :class:`CathedralConsumerContract`.

    Returns :class:`ConsumerRegistration` with ``contract_compliant=True``
    on success.

    Returns :class:`ConsumerRegistration` with ``contract_compliant=False``
    and populated ``validation_errors`` on failure (does NOT raise; the
    caller decides whether to refuse or apply a waiver per the auto-discovery
    loop's strict-mode semantics).

    The single hard-raise case is when ``module`` is None or not a Python
    module object (programming error, not contract failure).
    """
    if module is None:
        raise CathedralConsumerContractError("module is None")

    errors: list[str] = []
    resolved_path = module_path or getattr(module, "__name__", "<unknown>")

    # Field validation.
    name_ok = _validate_field(module, "CONSUMER_NAME", str, errors)
    version_ok = _validate_field(module, "CONSUMER_VERSION", str, errors)
    hooks_ok = _validate_field(
        module, "CONSUMER_HOOK_NUMBERS", tuple, errors
    )

    # Hook number element validation.
    hook_numbers: tuple[HookNumber, ...] = ()
    if hooks_ok:
        raw_hooks = getattr(module, "CONSUMER_HOOK_NUMBERS")
        validated_hooks: list[HookNumber] = []
        for i, hook in enumerate(raw_hooks):
            if not isinstance(hook, HookNumber):
                errors.append(
                    f"CONSUMER_HOOK_NUMBERS[{i}] must be HookNumber, "
                    f"got {type(hook).__name__}"
                )
            else:
                validated_hooks.append(hook)
        if not validated_hooks and not errors:
            errors.append("CONSUMER_HOOK_NUMBERS must not be empty")
        hook_numbers = tuple(validated_hooks)

    # Callable surface validation.
    for callable_name in ("update_from_anchor", "consume_candidate"):
        if not hasattr(module, callable_name):
            errors.append(f"missing callable: {callable_name}")
            continue
        attr = getattr(module, callable_name)
        if not callable(attr):
            errors.append(
                f"{callable_name} must be callable, got {type(attr).__name__}"
            )

    consumer_name = (
        getattr(module, "CONSUMER_NAME", "") if name_ok else ""
    )
    consumer_version = (
        getattr(module, "CONSUMER_VERSION", "") if version_ok else "unknown"
    )

    return ConsumerRegistration(
        consumer_name=consumer_name or "<invalid>",
        consumer_version=consumer_version or "unknown",
        consumer_hook_numbers=hook_numbers,
        consumer_module_path=resolved_path,
        contract_compliant=(not errors),
        waiver_rationale=None,
        waiver_active=False,
        validation_errors=tuple(errors),
    )
