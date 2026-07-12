# SPDX-License-Identifier: MIT
"""Typed attributed complex shared by the corpus and witness lens surfaces.

The Lens Engine treats the research-corpus graph and witness geometry as the
same abstract object ``T = (E, G, Phi, S, L, R, X)``.  This module intentionally
contains only data contracts and validation; lens algorithms live in sibling
modules and adapters preserve the authority (or absence) of source fields.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import numpy as np


class LensEngineError(ValueError):
    """Base error for malformed complexes, adapters, and lens queries."""


class ComplexValidationError(LensEngineError):
    """Raised when a :class:`TypedAttributedComplex` violates its contract."""


class AdapterError(LensEngineError):
    """Raised when source data cannot be truthfully exposed as a complex."""


class LensOperationError(LensEngineError):
    """Raised when a lens operation or its arguments are invalid."""


class QueryError(LensEngineError):
    """Raised when query dispatch cannot resolve an adapter, lens, or operation."""


def immutable_array(value: np.ndarray) -> np.ndarray:
    """Return an array backed by immutable bytes, not a reversible write flag."""

    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise ComplexValidationError("object arrays cannot be frozen safely")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    return frozen


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return immutable_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a recursively immutable snapshot of a string-keyed mapping."""

    return MappingProxyType(
        {key: _freeze_value(item) for key, item in (value or {}).items()}
    )


def _finite_tuple(values: tuple[float, ...], *, owner: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in out):
        raise ComplexValidationError(f"{owner} must contain only finite values")
    return out


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """Opaque source-preserving interval; temporal algebra is increment 2."""

    start: str
    end: str | None = None

    def __post_init__(self) -> None:
        if not self.start:
            raise ComplexValidationError("time interval start must be non-empty")
        if self.end == "":
            raise ComplexValidationError("time interval end must be non-empty when present")


@dataclass(frozen=True, slots=True)
class SpatialGeometry:
    """Typed spatial support for a point, polygon, grid pixel, or region cell.

    Coordinates preserve the adapter's native convention.  Witness adapters use
    ``(row, column)`` while general polygon calls may use ``(x, y)``; ``axes``
    makes that convention explicit rather than silently swapping coordinates.
    """

    kind: str
    coordinates: tuple[tuple[float, ...], ...]
    axes: tuple[str, ...] = ("x", "y")
    bounds: tuple[float, ...] | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ComplexValidationError("spatial geometry kind must be non-empty")
        if not self.axes:
            raise ComplexValidationError("spatial geometry axes must be non-empty")
        coords = tuple(
            _finite_tuple(tuple(coord), owner="spatial coordinates")
            for coord in self.coordinates
        )
        if any(len(coord) != len(self.axes) for coord in coords):
            raise ComplexValidationError("each spatial coordinate must match the axes arity")
        bounds = None
        if self.bounds is not None:
            bounds = _finite_tuple(tuple(self.bounds), owner="spatial bounds")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "attrs", _frozen_mapping(self.attrs))


@dataclass(frozen=True, slots=True)
class ComplexElement:
    """One typed element carrying the per-element maps of ``T``."""

    id: str
    kind: str
    phi: float = 0.0
    vec: tuple[float, ...] = ()
    scopes: frozenset[str] = frozenset()
    spatial: SpatialGeometry | None = None
    interval: TimeInterval | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ComplexValidationError("element id must be non-empty")
        if not self.kind:
            raise ComplexValidationError(f"element {self.id!r} kind must be non-empty")
        phi = float(self.phi)
        if not math.isfinite(phi):
            raise ComplexValidationError(f"element {self.id!r} phi must be finite")
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "vec", _finite_tuple(tuple(self.vec), owner=f"{self.id}.vec"))
        object.__setattr__(self, "scopes", frozenset(str(scope) for scope in self.scopes))
        object.__setattr__(self, "attrs", _frozen_mapping(self.attrs))


@dataclass(frozen=True, slots=True)
class ComplexEdge:
    """A typed graph edge in ``G``."""

    source: str
    target: str
    kind: str
    directed: bool = True
    weight: float = 1.0
    attrs: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ComplexValidationError("edge endpoints must be non-empty")
        if not self.kind:
            raise ComplexValidationError("edge kind must be non-empty")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ComplexValidationError("edge weight must be finite and non-negative")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "attrs", _frozen_mapping(self.attrs))


@dataclass(frozen=True, slots=True)
class TypedRelation:
    """A typed relation over one or more elements of ``E``."""

    kind: str
    members: tuple[str, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ComplexValidationError("relation kind must be non-empty")
        members = tuple(str(member) for member in self.members)
        if not members or any(not member for member in members):
            raise ComplexValidationError("relation members must be non-empty")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "attrs", _frozen_mapping(self.attrs))


@dataclass(frozen=True, slots=True)
class TypedAttributedComplex:
    """Immutable, validated representation of ``T = (E, G, Phi, S, L, R, X)``."""

    elements: tuple[ComplexElement, ...]
    edges: tuple[ComplexEdge, ...] = ()
    lineage: tuple[ComplexEdge, ...] = ()
    relations: tuple[TypedRelation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    _by_id: Mapping[str, ComplexElement] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        edges = tuple(self.edges)
        lineage = tuple(self.lineage)
        relations = tuple(self.relations)
        by_id: dict[str, ComplexElement] = {}
        for element in elements:
            if element.id in by_id:
                raise ComplexValidationError(f"duplicate element id: {element.id!r}")
            by_id[element.id] = element
        known = frozenset(by_id)
        for edge in (*edges, *lineage):
            missing = {edge.source, edge.target} - known
            if missing:
                raise ComplexValidationError(
                    f"edge {edge.kind!r} has unknown endpoint(s): {sorted(missing)}"
                )
        for relation in relations:
            missing = set(relation.members) - known
            if missing:
                raise ComplexValidationError(
                    f"relation {relation.kind!r} has unknown member(s): {sorted(missing)}"
                )
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    def __len__(self) -> int:
        return len(self.elements)

    def __iter__(self) -> Iterator[ComplexElement]:
        return iter(self.elements)

    def element(self, element_id: str) -> ComplexElement:
        """Return one element or fail with a domain-specific error."""

        try:
            return self._by_id[element_id]
        except KeyError as exc:
            raise LensOperationError(f"unknown element id: {element_id!r}") from exc

    def neighbours(self, element_id: str, *, directed: bool = False) -> tuple[str, ...]:
        """Return deterministic graph neighbours.

        ``directed=False`` treats every graph edge as undirected.  With
        ``directed=True``, directed edges contribute only ``source -> target``;
        explicitly undirected edges still contribute in both directions.
        """

        self.element(element_id)
        out: set[str] = set()
        for edge in self.edges:
            if edge.source == element_id:
                out.add(edge.target)
            if edge.target == element_id and (not directed or not edge.directed):
                out.add(edge.source)
        return tuple(sorted(out))

    @property
    def E(self) -> tuple[ComplexElement, ...]:
        return self.elements

    @property
    def G(self) -> tuple[ComplexEdge, ...]:
        return self.edges

    @property
    def Phi(self) -> Mapping[str, float]:
        return MappingProxyType({element.id: element.phi for element in self.elements})

    @property
    def S(self) -> Mapping[str, frozenset[str]]:
        return MappingProxyType({element.id: element.scopes for element in self.elements})

    @property
    def X(self) -> Mapping[str, SpatialGeometry | None]:
        return MappingProxyType({element.id: element.spatial for element in self.elements})

    @property
    def t(self) -> Mapping[str, TimeInterval | None]:
        return MappingProxyType({element.id: element.interval for element in self.elements})

    @property
    def L(self) -> tuple[ComplexEdge, ...]:
        return self.lineage

    @property
    def R(self) -> tuple[TypedRelation, ...]:
        return self.relations


T = TypedAttributedComplex

ResultValue = TypeVar("ResultValue")


@dataclass(frozen=True, slots=True)
class TypedResult(Generic[ResultValue]):
    """Uniform result envelope returned by every lens operation."""

    lens: str
    op: str
    value: ResultValue
    adapter: str = ""
    element_ids: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.lens or not self.op:
            raise ComplexValidationError("typed result lens and op must be non-empty")
        object.__setattr__(self, "element_ids", tuple(self.element_ids))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
