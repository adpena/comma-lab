# SPDX-License-Identifier: MIT
"""Public query dispatcher for the Lens Engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from .core import QueryError, T, TypedResult
from .graph import GraphLens
from .protocols import ComplexAdapter, Lens
from .spatial import SpatialLens
from .statistics import StatisticsLens
from .topology import TopologyLens

TOPOLOGY = TopologyLens()
GRAPH = GraphLens()
SPATIAL = SpatialLens()
STATISTICS = StatisticsLens()

_LENS_MAP: dict[str, Lens] = {
    TOPOLOGY.name: TOPOLOGY,
    GRAPH.name: GRAPH,
    SPATIAL.name: SPATIAL,
    STATISTICS.name: STATISTICS,
}
LENSES: Mapping[str, Lens] = MappingProxyType(_LENS_MAP)


def query(
    adapter: ComplexAdapter | T,
    lens: str | Lens,
    op: str,
    **args: Any,
) -> TypedResult[Any]:
    """Run ``lens.op`` over an adapter's typed attributed complex.

    A raw ``T`` is accepted for internal composition and unit tests.  Public
    adapter calls stamp the adapter name into the immutable result envelope.
    """

    if isinstance(adapter, T):
        complex_ = adapter
        adapter_name = str(adapter.metadata.get("adapter", "typed_complex"))
    else:
        if not isinstance(adapter, ComplexAdapter):
            raise QueryError("adapter must implement name and to_complex()")
        try:
            complex_ = adapter.to_complex()
        except QueryError:
            raise
        if not isinstance(complex_, T):
            raise QueryError("adapter.to_complex() must return TypedAttributedComplex")
        adapter_name = str(adapter.name)
    resolved: Lens
    if isinstance(lens, str):
        try:
            resolved = LENSES[lens]
        except KeyError as exc:
            raise QueryError(f"unknown lens {lens!r}; choose {sorted(LENSES)}") from exc
    else:
        if not isinstance(lens, Lens):
            raise QueryError("lens must be a registered name or implement the Lens protocol")
        resolved = lens
    if op not in resolved.operations:
        raise QueryError(
            f"lens {resolved.name!r} has no operation {op!r}; "
            f"choose {sorted(resolved.operations)}"
        )
    result = resolved.apply(complex_, op, **args)
    if not isinstance(result, TypedResult):
        raise QueryError(f"lens {resolved.name!r} returned an untyped result")
    return replace(result, adapter=adapter_name)
