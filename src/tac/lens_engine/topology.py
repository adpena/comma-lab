# SPDX-License-Identifier: MIT
"""Discrete Morse-style topology lens over an arbitrary attributed complex.

The witness adapter reuses the repository's real connected-component/region
adjacency substrate.  This module supplies the missing generic critical-point,
watershed, integral-route, and zero-dimensional persistence operations over that
graph.  It does not claim a smooth Morse-Smale reconstruction from the historical
polygon codec: every result records the discrete graph filtration it actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .core import ComplexEdge, LensOperationError, T, TypedResult


@dataclass(frozen=True, slots=True)
class CriticalPoint:
    element_id: str
    kind: str
    phi: float
    upper_link_components: int = 0
    lower_link_components: int = 0


@dataclass(frozen=True, slots=True)
class Basin:
    root_id: str
    member_ids: tuple[str, ...]
    mode: str
    relief: float


@dataclass(frozen=True, slots=True)
class PersistencePair:
    birth_id: str
    death_id: str | None
    birth: float
    death: float | None
    persistence: float
    essential: bool
    mode: str


@dataclass(frozen=True, slots=True)
class IntegralRoute:
    start_id: str
    end_id: str
    element_ids: tuple[str, ...]
    mode: str


@dataclass(frozen=True, slots=True)
class Watershed:
    basins: tuple[Basin, ...]
    separatrices: tuple[ComplexEdge, ...]
    mode: str


def _adjacency(complex_: T) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {element.id: set() for element in complex_.elements}
    for edge in complex_.edges:
        out[edge.source].add(edge.target)
        out[edge.target].add(edge.source)
    return out


def _induced_component_count(nodes: set[str], adjacency: dict[str, set[str]]) -> int:
    unseen = set(nodes)
    count = 0
    while unseen:
        count += 1
        stack = [min(unseen)]
        unseen.remove(stack[0])
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current] & unseen, reverse=True):
                unseen.remove(neighbour)
                stack.append(neighbour)
    return count


def _mode_key(phi: float, rank: int, mode: Literal["max", "min"]) -> tuple[float, int]:
    primary = phi if mode == "max" else -phi
    return primary, -rank


def _flow_map(
    complex_: T,
    mode: Literal["max", "min"],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    adjacency = _adjacency(complex_)
    ranks = {element.id: rank for rank, element in enumerate(complex_.elements)}
    nxt: dict[str, str] = {}
    for element in complex_.elements:
        candidates = {element.id, *adjacency[element.id]}
        nxt[element.id] = max(
            candidates,
            key=lambda element_id: _mode_key(
                complex_.element(element_id).phi,
                ranks[element_id],
                mode,
            ),
        )
    return nxt, adjacency


def _basins(complex_: T, mode: Literal["max", "min"]) -> tuple[Basin, ...]:
    nxt, _ = _flow_map(complex_, mode)

    def root(start: str) -> str:
        trail: list[str] = []
        current = start
        while nxt[current] != current:
            trail.append(current)
            current = nxt[current]
        for element_id in trail:
            nxt[element_id] = current
        return current

    members: dict[str, list[str]] = {}
    for element in complex_.elements:
        members.setdefault(root(element.id), []).append(element.id)
    out: list[Basin] = []
    for root_id in sorted(members):
        member_ids = tuple(members[root_id])
        values = [complex_.element(element_id).phi for element_id in member_ids]
        root_phi = complex_.element(root_id).phi
        relief = root_phi - min(values) if mode == "max" else max(values) - root_phi
        out.append(
            Basin(
                root_id=root_id,
                member_ids=member_ids,
                mode=mode,
                relief=float(relief),
            )
        )
    return tuple(out)


def _separatrices(complex_: T, basins: tuple[Basin, ...]) -> tuple[ComplexEdge, ...]:
    basin_of = {
        member_id: basin.root_id
        for basin in basins
        for member_id in basin.member_ids
    }
    out: list[ComplexEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in complex_.edges:
        if basin_of[edge.source] == basin_of[edge.target]:
            continue
        a, b = sorted((edge.source, edge.target))
        key = (a, b, edge.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ComplexEdge(
                source=a,
                target=b,
                kind="separatrix",
                directed=False,
                weight=edge.weight,
                attrs={"source_edge_kind": edge.kind},
            )
        )
    return tuple(sorted(out, key=lambda edge: (edge.source, edge.target, edge.kind)))


def _persistence_pairs(
    complex_: T,
    mode: Literal["max", "min"],
) -> tuple[PersistencePair, ...]:
    """Compute deterministic zero-dimensional graph-filtration persistence."""

    elements = sorted(
        complex_.elements,
        key=lambda element: (
            -element.phi if mode == "max" else element.phi,
            element.id,
        ),
    )
    adjacency = _adjacency(complex_)
    parent: dict[str, str] = {}
    birth_id: dict[str, str] = {}
    active: set[str] = set()
    pairs: list[PersistencePair] = []

    def find(element_id: str) -> str:
        root = element_id
        while parent[root] != root:
            root = parent[root]
        while parent[element_id] != element_id:
            element_id, parent[element_id] = parent[element_id], root
        return root

    def birth_key(root: str) -> tuple[float, str]:
        bid = birth_id[find(root)]
        value = complex_.element(bid).phi
        return ((value if mode == "max" else -value), bid)

    for element in elements:
        active.add(element.id)
        parent[element.id] = element.id
        birth_id[element.id] = element.id
        for neighbour in sorted(adjacency[element.id] & active):
            left = find(element.id)
            right = find(neighbour)
            if left == right:
                continue
            left_key = birth_key(left)
            right_key = birth_key(right)
            if left_key > right_key:
                survivor, dying = left, right
            else:
                survivor, dying = right, left
            dying_birth_id = birth_id[dying]
            birth_value = complex_.element(dying_birth_id).phi
            death_value = element.phi
            persistence = (
                birth_value - death_value
                if mode == "max"
                else death_value - birth_value
            )
            pairs.append(
                PersistencePair(
                    birth_id=dying_birth_id,
                    death_id=element.id,
                    birth=birth_value,
                    death=death_value,
                    persistence=float(max(persistence, 0.0)),
                    essential=False,
                    mode=mode,
                )
            )
            parent[dying] = survivor
            birth_id[survivor] = birth_id[survivor]

    essential_roots = sorted({find(element.id) for element in elements})
    for root_id in essential_roots:
        bid = birth_id[root_id]
        pairs.append(
            PersistencePair(
                birth_id=bid,
                death_id=None,
                birth=complex_.element(bid).phi,
                death=None,
                persistence=float("inf"),
                essential=True,
                mode=mode,
            )
        )
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                pair.essential,
                pair.persistence,
                pair.birth_id,
                pair.death_id or "",
            ),
        )
    )


class TopologyLens:
    """Critical points, watersheds, routes, and persistence over ``Phi`` on ``G``."""

    name = "topology"
    operations = frozenset(
        {
            "peaks",
            "saddles",
            "basins",
            "separatrices",
            "watershed",
            "routes",
            "persistence",
        }
    )
    _provenance = (
        "discrete graph filtration over T.Phi and T.G",
        "witness G supplied by tac.boundary_math.partition region adjacency",
    )

    def apply(self, complex_: T, op: str, **args: Any) -> TypedResult[Any]:
        if op not in self.operations:
            raise LensOperationError(
                f"topology operation {op!r} is unsupported; choose {sorted(self.operations)}"
            )
        mode = args.pop("mode", "max")
        if mode not in {"max", "min"}:
            raise LensOperationError("topology mode must be 'max' or 'min'")
        value: Any
        if op == "peaks":
            strict = bool(args.pop("strict", False))
            self._reject_extra(args)
            adjacency = _adjacency(complex_)
            out: list[CriticalPoint] = []
            for element in complex_.elements:
                values = [complex_.element(nid).phi for nid in adjacency[element.id]]
                is_peak = not values or (
                    all(element.phi > value for value in values)
                    if strict
                    else all(element.phi >= value for value in values)
                )
                if is_peak:
                    out.append(CriticalPoint(element.id, "peak", element.phi))
            value = tuple(out)
        elif op == "saddles":
            self._reject_extra(args)
            adjacency = _adjacency(complex_)
            out = []
            for element in complex_.elements:
                upper = {
                    nid
                    for nid in adjacency[element.id]
                    if complex_.element(nid).phi > element.phi
                }
                lower = {
                    nid
                    for nid in adjacency[element.id]
                    if complex_.element(nid).phi < element.phi
                }
                upper_n = _induced_component_count(upper, adjacency)
                lower_n = _induced_component_count(lower, adjacency)
                if upper and lower and (upper_n >= 2 or lower_n >= 2):
                    out.append(
                        CriticalPoint(
                            element_id=element.id,
                            kind="saddle",
                            phi=element.phi,
                            upper_link_components=upper_n,
                            lower_link_components=lower_n,
                        )
                    )
            value = tuple(out)
        elif op == "basins":
            self._reject_extra(args)
            value = _basins(complex_, mode)
        elif op == "separatrices":
            self._reject_extra(args)
            value = _separatrices(complex_, _basins(complex_, mode))
        elif op == "watershed":
            self._reject_extra(args)
            basins = _basins(complex_, mode)
            value = Watershed(basins, _separatrices(complex_, basins), mode)
        elif op == "routes":
            starts = args.pop("starts", None)
            self._reject_extra(args)
            nxt, _ = _flow_map(complex_, mode)
            if starts is None:
                starts = tuple(element.id for element in complex_.elements)
            starts = tuple(str(start) for start in starts)
            for start in starts:
                complex_.element(start)
            routes: list[IntegralRoute] = []
            for start in starts:
                path = [start]
                while nxt[path[-1]] != path[-1]:
                    path.append(nxt[path[-1]])
                routes.append(IntegralRoute(start, path[-1], tuple(path), mode))
            value = tuple(routes)
        else:
            self._reject_extra(args)
            value = _persistence_pairs(complex_, mode)

        element_ids = self._element_ids(value)
        return TypedResult(
            lens=self.name,
            op=op,
            value=value,
            element_ids=element_ids,
            provenance=self._provenance,
            metadata={"mode": mode, "empty": not bool(complex_.elements)},
        )

    @staticmethod
    def _reject_extra(args: dict[str, Any]) -> None:
        if args:
            raise LensOperationError(f"unexpected topology arguments: {sorted(args)}")

    @staticmethod
    def _element_ids(value: Any) -> tuple[str, ...]:
        if isinstance(value, tuple):
            ids: list[str] = []
            for item in value:
                if isinstance(item, CriticalPoint):
                    ids.append(item.element_id)
                elif isinstance(item, Basin):
                    ids.extend(item.member_ids)
                elif isinstance(item, IntegralRoute):
                    ids.extend(item.element_ids)
                elif isinstance(item, ComplexEdge):
                    ids.extend((item.source, item.target))
                elif isinstance(item, PersistencePair):
                    ids.append(item.birth_id)
                    if item.death_id is not None:
                        ids.append(item.death_id)
            return tuple(dict.fromkeys(ids))
        if isinstance(value, Watershed):
            return tuple(
                dict.fromkeys(
                    member
                    for basin in value.basins
                    for member in basin.member_ids
                )
            )
        return ()
