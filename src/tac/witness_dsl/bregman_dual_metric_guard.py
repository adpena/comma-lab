# SPDX-License-Identifier: MIT
"""Fail-closed DSL adoption guard for the corrected Bregman dual metric.

This is a non-trainer policy surface.  It emits no argv and resolves its metric
through :mod:`tac.witness_dsl.lever_registry`, preserving one DSL registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from tac.witness_dsl.lever_registry import (
    CanonicalMetricDescriptor,
    resolve_canonical_metric,
)

CANONICAL_METRIC_ID = "argmax_native_vjp_fidelity_v1"
FISHER_NATURAL_COTANGENT_GEOMETRY = "inverse_hessian_H_inverse"
TYPED_LINEAR_SOLVE = "typed_linear_solve"
RAW_DUAL_EUCLIDEAN_SCOPE = "squared_hessian_H_squared_only"


class BregmanDualMetricAdoptionError(ValueError):
    """Raised when a metric adoption is missing, duplicated, or shortcuts H^-1."""


@dataclass(frozen=True, slots=True)
class BregmanDualMetricBinding:
    """Complete declaration required to adopt the canonical Bregman metric."""

    metric_id: str
    fisher_natural_cotangent_geometry: str
    fisher_natural_cotangent_solve: str
    fisher_natural_cotangent_solve_elided: bool
    dual_euclidean_no_solve_scope: str

    def __post_init__(self) -> None:
        if self.fisher_natural_cotangent_geometry != FISHER_NATURAL_COTANGENT_GEOMETRY:
            raise BregmanDualMetricAdoptionError(
                "canonical adoption requires Fisher-natural inverse-Hessian cotangent geometry"
            )
        if self.fisher_natural_cotangent_solve != TYPED_LINEAR_SOLVE:
            raise BregmanDualMetricAdoptionError(
                "canonical adoption requires a typed H^-1 linear solve"
            )
        if self.fisher_natural_cotangent_solve_elided is not False:
            raise BregmanDualMetricAdoptionError(
                "Fisher-natural cotangent solve_elided must be false"
            )
        if self.dual_euclidean_no_solve_scope != RAW_DUAL_EUCLIDEAN_SCOPE:
            raise BregmanDualMetricAdoptionError(
                "raw dual Euclidean is allowed only for squared_hessian_H_squared_only"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BregmanDualMetricBinding:
        """Build from an exact-key mapping; missing and extra declarations fail closed."""

        if not isinstance(payload, Mapping):
            raise BregmanDualMetricAdoptionError("metric binding must be a mapping or typed binding")
        expected = {field.name for field in fields(cls)}
        actual = set(payload)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise BregmanDualMetricAdoptionError(
                f"metric binding keys do not close: missing={missing}, unknown={extra}"
            )
        return cls(**{key: payload[key] for key in expected})


@dataclass(frozen=True, slots=True)
class ResolvedBregmanDualMetricAdoption:
    """A validated binding paired with its exact existing-registry entry."""

    binding: BregmanDualMetricBinding
    registry_entry: CanonicalMetricDescriptor


def canonical_bregman_dual_metric_binding() -> BregmanDualMetricBinding:
    """Return the complete, canonical single-metric adoption declaration."""

    return BregmanDualMetricBinding(
        metric_id=CANONICAL_METRIC_ID,
        fisher_natural_cotangent_geometry=FISHER_NATURAL_COTANGENT_GEOMETRY,
        fisher_natural_cotangent_solve=TYPED_LINEAR_SOLVE,
        fisher_natural_cotangent_solve_elided=False,
        dual_euclidean_no_solve_scope=RAW_DUAL_EUCLIDEAN_SCOPE,
    )


def resolve_bregman_dual_metric_adoption(
    bindings: Sequence[BregmanDualMetricBinding | Mapping[str, Any]],
) -> ResolvedBregmanDualMetricAdoption:
    """Validate exactly one binding and resolve it through ``lever_registry``.

    Empty input, repeated bindings, unknown metric IDs, incomplete mappings,
    and any no-solve shortcut are all refused before an adoption is returned.
    """

    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise BregmanDualMetricAdoptionError("bindings must be a sequence")
    if len(bindings) == 0:
        raise BregmanDualMetricAdoptionError("canonical metric binding is missing")
    if len(bindings) != 1:
        raise BregmanDualMetricAdoptionError(
            f"canonical metric binding must appear exactly once, got {len(bindings)}"
        )
    candidate = bindings[0]
    binding = (
        candidate
        if isinstance(candidate, BregmanDualMetricBinding)
        else BregmanDualMetricBinding.from_mapping(candidate)
    )
    try:
        registry_entry = resolve_canonical_metric(binding.metric_id)
    except ValueError as exc:
        raise BregmanDualMetricAdoptionError(str(exc)) from exc
    if registry_entry.metric_id != CANONICAL_METRIC_ID:
        raise BregmanDualMetricAdoptionError(
            f"guard resolves only {CANONICAL_METRIC_ID!r}, got {registry_entry.metric_id!r}"
        )
    return ResolvedBregmanDualMetricAdoption(
        binding=binding,
        registry_entry=registry_entry,
    )


__all__ = [
    "CANONICAL_METRIC_ID",
    "FISHER_NATURAL_COTANGENT_GEOMETRY",
    "RAW_DUAL_EUCLIDEAN_SCOPE",
    "TYPED_LINEAR_SOLVE",
    "BregmanDualMetricAdoptionError",
    "BregmanDualMetricBinding",
    "ResolvedBregmanDualMetricAdoption",
    "canonical_bregman_dual_metric_binding",
    "resolve_bregman_dual_metric_adoption",
]
