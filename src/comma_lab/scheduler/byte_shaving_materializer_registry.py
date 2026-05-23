# SPDX-License-Identifier: MIT
"""Materializer registry for byte-shaving campaign queue compilation.

The byte-shaving planner is intentionally broader than the current executable
surface. This registry is the fail-closed boundary between planning rows and
queueable local work: every selected operation resolves to either a concrete
adapter or an auditable missing-materializer blocker.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tac.optimization.byte_shaving_campaign import DEFAULT_OPERATION_FAMILIES
from tac.optimization.proxy_candidate_contract import ordered_unique

REGISTRY_SCHEMA = "byte_shaving_materializer_registry.v1"
DQS1_DROP_PAIR_MATERIALIZER = "dqs1_pairset_drop_pair_adapter"
DQS1_PAIRSET_TARGET_KIND = "dqs1_pairset_drop_pair"


@dataclass(frozen=True)
class MaterializerAdapter:
    """A concrete operation materializer known to the queue compiler."""

    materializer_id: str
    unit_kind: str
    operation_family: str
    target_kind: str
    executable: bool
    description: str
    required_context_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterializerResolution:
    """Resolution of one selected byte-shaving operation against the registry."""

    unit_id: str
    unit_kind: str
    operation_id: str
    operation_family: str
    explicit_materializer: str | None
    materializer_id: str | None
    target_kind: str | None
    executable: bool
    blockers: tuple[str, ...]
    adapter: MaterializerAdapter | None = None


_ADAPTERS: tuple[MaterializerAdapter, ...] = (
    MaterializerAdapter(
        materializer_id=DQS1_DROP_PAIR_MATERIALIZER,
        unit_kind="pair",
        operation_family="drop_pair",
        target_kind=DQS1_PAIRSET_TARGET_KIND,
        executable=True,
        description="Compile pair-unit drop operations into DQS1 pairset local-first queue rows.",
        required_context_fields=("dqs1_base_pair_indices",),
    ),
)

_ADAPTERS_BY_TARGET_KEY: dict[tuple[str, str, str], MaterializerAdapter] = {
    (adapter.target_kind, adapter.unit_kind, adapter.operation_family): adapter
    for adapter in _ADAPTERS
}
_ADAPTERS_BY_ID: dict[str, MaterializerAdapter] = {
    adapter.materializer_id: adapter
    for adapter in _ADAPTERS
}
KNOWN_OPERATION_FAMILIES: frozenset[str] = frozenset(
    family
    for families in DEFAULT_OPERATION_FAMILIES.values()
    for family in families
)


def _nonempty_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _operation_materializer(operation: Mapping[str, Any]) -> str | None:
    value = _nonempty_str(operation.get("materializer"))
    return value or None


def _operation_target_kind(operation: Mapping[str, Any]) -> str | None:
    value = _nonempty_str(operation.get("target_kind"))
    if value:
        return value
    params = operation.get("params")
    if isinstance(params, Mapping):
        value = _nonempty_str(
            params.get("target_kind") or params.get("materializer_target_kind")
        )
        if value:
            return value
    return None


def resolve_materializer(
    *,
    operation: Mapping[str, Any],
    unit: Mapping[str, Any] | None,
) -> MaterializerResolution:
    """Resolve a selected operation into a concrete adapter or blockers."""

    unit_id = _nonempty_str(operation.get("unit_id"))
    operation_id = _nonempty_str(operation.get("operation_id"))
    operation_family = _nonempty_str(operation.get("operation_family"))
    explicit_materializer = _operation_materializer(operation)
    explicit_target_kind = _operation_target_kind(operation)
    blockers: list[str] = []

    if unit is None:
        unit_kind = ""
        blockers.append(f"selected_unit_missing_from_ranked_units:{unit_id or '<missing>'}")
    else:
        unit_kind = _nonempty_str(unit.get("unit_kind") or unit.get("kind"))
        if not unit_kind:
            blockers.append(f"unit_kind_missing:{unit_id or '<missing>'}")

    adapter: MaterializerAdapter | None = None
    if explicit_materializer is not None:
        adapter = _ADAPTERS_BY_ID.get(explicit_materializer)
        if adapter is None:
            blockers.append(f"materializer_not_registered:{explicit_materializer}")
    elif explicit_target_kind and unit_kind and operation_family:
        adapter = _ADAPTERS_BY_TARGET_KEY.get(
            (explicit_target_kind, unit_kind, operation_family)
        )
        if adapter is None:
            blockers.append(
                f"materializer_not_registered:{explicit_target_kind}:"
                f"{unit_kind}:{operation_family}"
            )
    elif unit_kind and operation_family:
        blockers.append(
            f"materializer_target_kind_required:{unit_kind}:{operation_family}"
        )

    if not operation_family:
        blockers.append(f"operation_family_missing:{unit_id or operation_id or '<missing>'}")
    elif operation_family not in KNOWN_OPERATION_FAMILIES:
        blockers.append(f"unknown_operation_family:{operation_family}")
    elif adapter is None and explicit_materializer is not None:
        blockers.append(
            f"materializer_not_registered:{unit_kind or '<missing>'}:{operation_family}"
        )

    if adapter is not None:
        if explicit_target_kind and explicit_target_kind != adapter.target_kind:
            blockers.append(
                f"materializer_target_kind_mismatch:{adapter.materializer_id}:"
                f"{explicit_target_kind}:expected_{adapter.target_kind}"
            )
        if unit_kind and unit_kind != adapter.unit_kind:
            blockers.append(
                f"materializer_unit_kind_mismatch:{adapter.materializer_id}:"
                f"{unit_id or '<missing>'}:{unit_kind}:expected_{adapter.unit_kind}"
            )
        if operation_family and operation_family != adapter.operation_family:
            blockers.append(
                f"materializer_operation_family_mismatch:{adapter.materializer_id}:"
                f"{operation_family}:expected_{adapter.operation_family}"
            )
        if not adapter.executable:
            blockers.append(f"materializer_not_executable:{adapter.materializer_id}")

    blockers = ordered_unique(blockers)
    return MaterializerResolution(
        unit_id=unit_id,
        unit_kind=unit_kind,
        operation_id=operation_id,
        operation_family=operation_family,
        explicit_materializer=explicit_materializer,
        materializer_id=adapter.materializer_id if adapter is not None else explicit_materializer,
        target_kind=adapter.target_kind if adapter is not None else explicit_target_kind,
        executable=adapter is not None and adapter.executable and not blockers,
        blockers=tuple(blockers),
        adapter=adapter,
    )


def registry_manifest() -> dict[str, Any]:
    """Return a machine-readable registry view for tests and runbooks."""

    return {
        "schema": REGISTRY_SCHEMA,
        "adapters": [
            {
                "materializer_id": adapter.materializer_id,
                "unit_kind": adapter.unit_kind,
                "operation_family": adapter.operation_family,
                "target_kind": adapter.target_kind,
                "executable": adapter.executable,
                "required_context_fields": list(adapter.required_context_fields),
                "description": adapter.description,
            }
            for adapter in _ADAPTERS
        ],
        "known_operation_families": sorted(KNOWN_OPERATION_FAMILIES),
    }


__all__ = [
    "DQS1_DROP_PAIR_MATERIALIZER",
    "DQS1_PAIRSET_TARGET_KIND",
    "KNOWN_OPERATION_FAMILIES",
    "REGISTRY_SCHEMA",
    "MaterializerAdapter",
    "MaterializerResolution",
    "registry_manifest",
    "resolve_materializer",
]
