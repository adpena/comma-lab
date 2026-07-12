# SPDX-License-Identifier: MIT
"""Deterministic graph lens over ``T.G``.

``CorpusAdapter`` maps the canonical :mod:`tac.graph_memory` nodes and edges
into ``T``; ``WitnessAdapter`` maps its region adjacency graph the same way.
The algorithms here are deliberately thin and dependency-free because graph
memory itself does not expose general DFS, shortest-path, centrality, component,
or community APIs.
"""

from __future__ import annotations

import heapq
from collections import Counter, deque
from dataclasses import dataclass
from math import inf
from typing import Any

from .core import LensOperationError, T, TypedResult


@dataclass(frozen=True, slots=True)
class Traversal:
    order: tuple[str, ...]
    parents: tuple[tuple[str, str | None], ...]
    depths: tuple[tuple[str, int], ...]
    directed: bool


@dataclass(frozen=True, slots=True)
class ShortestPath:
    nodes: tuple[str, ...]
    distance: float
    reachable: bool
    directed: bool


@dataclass(frozen=True, slots=True)
class Centrality:
    method: str
    scores: tuple[tuple[str, float], ...]
    directed: bool
    weighted: bool


@dataclass(frozen=True, slots=True)
class ComponentPartition:
    components: tuple[tuple[str, ...], ...]
    kind: str


@dataclass(frozen=True, slots=True)
class CommunityPartition:
    communities: tuple[tuple[str, ...], ...]
    method: str
    iterations: int


def _adjacency(
    complex_: T,
    *,
    directed: bool,
    edge_kinds: frozenset[str] | None,
) -> dict[str, tuple[tuple[str, float], ...]]:
    work: dict[str, dict[str, float]] = {element.id: {} for element in complex_.elements}
    for edge in complex_.edges:
        if edge_kinds is not None and edge.kind not in edge_kinds:
            continue
        prior = work[edge.source].get(edge.target, inf)
        work[edge.source][edge.target] = min(prior, edge.weight)
        if not directed or not edge.directed:
            prior = work[edge.target].get(edge.source, inf)
            work[edge.target][edge.source] = min(prior, edge.weight)
    return {
        element_id: tuple(sorted(neighbours.items()))
        for element_id, neighbours in work.items()
    }


def _edge_kinds(value: Any) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(item) for item in value)


def _bfs(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    start: str,
    max_depth: int | None,
) -> Traversal:
    if start not in adjacency:
        raise LensOperationError(f"unknown BFS start element: {start!r}")
    if max_depth is not None and max_depth < 0:
        raise LensOperationError("max_depth must be non-negative")
    queue = deque([start])
    parent: dict[str, str | None] = {start: None}
    depth = {start: 0}
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        if max_depth is not None and depth[current] >= max_depth:
            continue
        for neighbour, _ in adjacency[current]:
            if neighbour in parent:
                continue
            parent[neighbour] = current
            depth[neighbour] = depth[current] + 1
            queue.append(neighbour)
    return Traversal(
        order=tuple(order),
        parents=tuple((element_id, parent[element_id]) for element_id in order),
        depths=tuple((element_id, depth[element_id]) for element_id in order),
        directed=False,
    )


def _dfs(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    start: str,
    max_depth: int | None,
) -> Traversal:
    if start not in adjacency:
        raise LensOperationError(f"unknown DFS start element: {start!r}")
    if max_depth is not None and max_depth < 0:
        raise LensOperationError("max_depth must be non-negative")
    stack: list[tuple[str, str | None, int]] = [(start, None, 0)]
    seen: set[str] = set()
    parent: dict[str, str | None] = {}
    depth: dict[str, int] = {}
    order: list[str] = []
    while stack:
        current, predecessor, current_depth = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        parent[current] = predecessor
        depth[current] = current_depth
        order.append(current)
        if max_depth is not None and current_depth >= max_depth:
            continue
        for neighbour, _ in reversed(adjacency[current]):
            if neighbour not in seen:
                stack.append((neighbour, current, current_depth + 1))
    return Traversal(
        order=tuple(order),
        parents=tuple((element_id, parent[element_id]) for element_id in order),
        depths=tuple((element_id, depth[element_id]) for element_id in order),
        directed=False,
    )


def _shortest_path(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    start: str,
    target: str,
) -> ShortestPath:
    if start not in adjacency or target not in adjacency:
        missing = start if start not in adjacency else target
        raise LensOperationError(f"unknown shortest-path element: {missing!r}")
    distance = dict.fromkeys(adjacency, inf)
    distance[start] = 0.0
    predecessor: dict[str, str] = {}
    heap: list[tuple[float, str]] = [(0.0, start)]
    while heap:
        current_distance, current = heapq.heappop(heap)
        if current_distance != distance[current]:
            continue
        if current == target:
            break
        for neighbour, weight in adjacency[current]:
            candidate = current_distance + weight
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                predecessor[neighbour] = current
                heapq.heappush(heap, (candidate, neighbour))
    if distance[target] == inf:
        return ShortestPath((), inf, False, False)
    path = [target]
    while path[-1] != start:
        path.append(predecessor[path[-1]])
    path.reverse()
    return ShortestPath(tuple(path), float(distance[target]), True, False)


def _components(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
) -> tuple[tuple[str, ...], ...]:
    unseen = set(adjacency)
    components: list[tuple[str, ...]] = []
    while unseen:
        start = min(unseen)
        traversal = _bfs(adjacency, start, None)
        component = tuple(sorted(traversal.order))
        unseen.difference_update(component)
        components.append(component)
    return tuple(sorted(components, key=lambda component: (component[0], len(component))))


def _degree_centrality(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    *,
    weighted: bool,
) -> dict[str, float]:
    scale = max(len(adjacency) - 1, 1)
    return {
        element_id: (
            sum(weight for _, weight in neighbours) / scale
            if weighted
            else len(neighbours) / scale
        )
        for element_id, neighbours in adjacency.items()
    }


def _closeness_centrality(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    *,
    weighted: bool,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    n = len(adjacency)
    for start in adjacency:
        if weighted:
            distances_float = dict.fromkeys(adjacency, inf)
            distances_float[start] = 0.0
            heap: list[tuple[float, str]] = [(0.0, start)]
            while heap:
                current_distance, current = heapq.heappop(heap)
                if current_distance != distances_float[current]:
                    continue
                for neighbour, weight in adjacency[current]:
                    candidate = current_distance + weight
                    if candidate < distances_float[neighbour]:
                        distances_float[neighbour] = candidate
                        heapq.heappush(heap, (candidate, neighbour))
            distances = {
                element_id: distance
                for element_id, distance in distances_float.items()
                if distance < inf
            }
        else:
            distances = {start: 0.0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbour, _ in adjacency[current]:
                    if neighbour not in distances:
                        distances[neighbour] = distances[current] + 1.0
                        queue.append(neighbour)
        reachable = len(distances) - 1
        total = sum(distances.values())
        if reachable == 0 or total == 0 or n <= 1:
            scores[start] = 0.0
        else:
            scores[start] = (reachable / total) * (reachable / (n - 1))
    return scores


def _betweenness_centrality(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
) -> dict[str, float]:
    scores = dict.fromkeys(adjacency, 0.0)
    for source in adjacency:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {element_id: [] for element_id in adjacency}
        sigma = dict.fromkeys(adjacency, 0.0)
        sigma[source] = 1.0
        distance = dict.fromkeys(adjacency, -1)
        distance[source] = 0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbour, _ in adjacency[vertex]:
                if distance[neighbour] < 0:
                    queue.append(neighbour)
                    distance[neighbour] = distance[vertex] + 1
                if distance[neighbour] == distance[vertex] + 1:
                    sigma[neighbour] += sigma[vertex]
                    predecessors[neighbour].append(vertex)
        dependency = dict.fromkeys(adjacency, 0.0)
        while stack:
            vertex = stack.pop()
            for predecessor in predecessors[vertex]:
                if sigma[vertex] > 0:
                    dependency[predecessor] += (
                        sigma[predecessor] / sigma[vertex]
                    ) * (1.0 + dependency[vertex])
            if vertex != source:
                scores[vertex] += dependency[vertex]
    n = len(adjacency)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        scores = {element_id: value * scale for element_id, value in scores.items()}
    return scores


def _label_propagation(
    adjacency: dict[str, tuple[tuple[str, float], ...]],
    *,
    max_iterations: int,
) -> CommunityPartition:
    if max_iterations <= 0:
        raise LensOperationError("max_iterations must be positive")
    labels = {element_id: element_id for element_id in adjacency}
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        changed = False
        for element_id in sorted(adjacency):
            neighbour_labels = [labels[neighbour] for neighbour, _ in adjacency[element_id]]
            if not neighbour_labels:
                continue
            counts = Counter(neighbour_labels)
            best_count = max(counts.values())
            candidates = sorted(label for label, count in counts.items() if count == best_count)
            chosen = labels[element_id] if labels[element_id] in candidates else candidates[0]
            if chosen != labels[element_id]:
                labels[element_id] = chosen
                changed = True
        if not changed:
            break
    groups: dict[str, list[str]] = {}
    for element_id, label in labels.items():
        groups.setdefault(label, []).append(element_id)
    communities = tuple(
        sorted(
            (tuple(sorted(group)) for group in groups.values()),
            key=lambda group: (group[0], len(group)),
        )
    )
    return CommunityPartition(communities, "deterministic_label_propagation", iterations)


class GraphLens:
    """Traversal and network analysis over the graph edge set of ``T``."""

    name = "graph"
    operations = frozenset(
        {"bfs", "dfs", "shortest_path", "centrality", "components", "community"}
    )
    _provenance = ("algorithms over canonical T.G edges",)

    def apply(self, complex_: T, op: str, **args: Any) -> TypedResult[Any]:
        if op not in self.operations:
            raise LensOperationError(
                f"graph operation {op!r} is unsupported; choose {sorted(self.operations)}"
            )
        directed = bool(args.pop("directed", False))
        kinds = _edge_kinds(args.pop("edge_kinds", None))
        adjacency = _adjacency(complex_, directed=directed, edge_kinds=kinds)
        value: Any
        element_ids: tuple[str, ...]
        if op in {"bfs", "dfs"}:
            start = str(args.pop("start", ""))
            if not start:
                raise LensOperationError(f"{op} requires start=<element id>")
            max_depth = args.pop("max_depth", None)
            if max_depth is not None:
                max_depth = int(max_depth)
            self._reject_extra(args)
            value = (
                _bfs(adjacency, start, max_depth)
                if op == "bfs"
                else _dfs(adjacency, start, max_depth)
            )
            value = Traversal(value.order, value.parents, value.depths, directed)
            element_ids = value.order
        elif op == "shortest_path":
            start = str(args.pop("start", ""))
            target = str(args.pop("target", ""))
            if not start or not target:
                raise LensOperationError("shortest_path requires start= and target=")
            self._reject_extra(args)
            raw = _shortest_path(adjacency, start, target)
            value = ShortestPath(raw.nodes, raw.distance, raw.reachable, directed)
            element_ids = value.nodes
        elif op == "components":
            if directed:
                raise LensOperationError("components kind='weak' requires directed=False")
            kind = str(args.pop("kind", "weak"))
            if kind != "weak":
                raise LensOperationError("increment-1 components supports kind='weak' only")
            self._reject_extra(args)
            weak = _adjacency(complex_, directed=False, edge_kinds=kinds)
            value = ComponentPartition(_components(weak), kind)
            element_ids = tuple(element.id for element in complex_.elements)
        elif op == "community":
            if directed:
                raise LensOperationError("community requires directed=False")
            max_iterations = int(args.pop("max_iterations", 100))
            self._reject_extra(args)
            value = _label_propagation(adjacency, max_iterations=max_iterations)
            element_ids = tuple(element.id for element in complex_.elements)
        else:
            method = str(args.pop("method", "degree"))
            weighted = bool(args.pop("weighted", False))
            self._reject_extra(args)
            if method == "degree":
                scores = _degree_centrality(adjacency, weighted=weighted)
            elif method == "closeness":
                scores = _closeness_centrality(adjacency, weighted=weighted)
            elif method == "betweenness":
                if weighted:
                    raise LensOperationError(
                        "weighted betweenness is not implemented; pass weighted=False explicitly"
                    )
                scores = _betweenness_centrality(adjacency)
            else:
                raise LensOperationError(
                    "centrality method must be 'degree', 'closeness', or 'betweenness'"
                )
            ranked = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
            value = Centrality(method, ranked, directed, weighted)
            element_ids = tuple(element_id for element_id, _ in ranked)
        return TypedResult(
            lens=self.name,
            op=op,
            value=value,
            element_ids=element_ids,
            provenance=self._provenance,
            metadata={
                "directed": directed,
                "edge_kinds": tuple(sorted(kinds)) if kinds is not None else None,
                "centrality_weighted": value.weighted if isinstance(value, Centrality) else None,
            },
        )

    @staticmethod
    def _reject_extra(args: dict[str, Any]) -> None:
        if args:
            raise LensOperationError(f"unexpected graph arguments: {sorted(args)}")
