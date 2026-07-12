# SPDX-License-Identifier: MIT
"""Reconstruction recall over the graph-memory (task #411).

MRAgent thesis: "Memory is Reconstructed, Not Retrieved." Rather than returning
the top-k flat chunks (what tools/corpus_query.py does — the #346 retrieval layer),
this ASSEMBLES an answer by (1) finding seed nodes whose id/title/summary match the
query, then (2) traversing the graph outward from the seeds and (3) reconstructing
a coherent context from the connected subgraph + the edge paths that link it.

Deterministic: the same graph + query always yields the same reconstruction (pure
token scoring + BFS with stable tie-breaks on node id).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .model import Graph

if TYPE_CHECKING:
    from .model import Node

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_HASHREF_RE = re.compile(r"#(\d{2,4})\b")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_STOP = frozenset(
    ["the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "is", "at", "by", "with", "vs", "from", "as", "it", "its", "our", "we", "be", "are", "was", "what", "do", "we", "know", "about", "that", "this", "these", "those", "when", "how", "why", "the", "incident"]
)
# distance decay for BFS reconstruction weight
_DECAY = 0.45
_MAX_DEPTH = 2


def tokenize(query: str) -> list[str]:
    toks = [t for t in _TOKEN_RE.findall(query.lower()) if t not in _STOP and len(t) > 1]
    # preserve #refs + dates as high-value exact tokens
    return toks


@dataclass
class Reconstruction:
    query: str
    seeds: list[tuple[str, float]]  # (node_id, score)
    nodes: list[str]  # ordered node ids in the reconstruction
    paths: dict[str, list[str]]  # node_id -> human path from a seed
    graph_counts: dict = field(default_factory=dict)

    def to_dict(self, graph: Graph) -> dict:
        return {
            "query": self.query,
            "seeds": [{"id": s, "score": round(sc, 3),
                       "title": graph.nodes[s].title if s in graph.nodes else s}
                      for s, sc in self.seeds],
            "reconstructed_nodes": [
                {
                    "id": nid,
                    "ntype": graph.nodes[nid].ntype if nid in graph.nodes else "?",
                    "title": graph.nodes[nid].title if nid in graph.nodes else nid,
                    "summary": graph.nodes[nid].summary if nid in graph.nodes else "",
                    "source": graph.nodes[nid].source if nid in graph.nodes else "",
                    "path": self.paths.get(nid, []),
                }
                for nid in self.nodes
            ],
            "graph_counts": self.graph_counts,
        }


def _score_node(node: Node, terms: list[str], hashrefs: set[str], dates: set[str]) -> float:
    hay = (node.id + " \n " + node.title + " \n " + node.summary).lower()
    if not hay.strip():
        return 0.0
    score = 0.0
    distinct = 0
    for t in terms:
        c = hay.count(t)
        if c:
            distinct += 1
            score += 1.0 + 0.25 * min(c, 4)  # term-hit density, capped
    # distinct-term coverage bonus (rewards matching MORE of the query)
    if terms:
        score *= 1.0 + 0.6 * (distinct / len(terms))
    # exact #ref match is high-value (entity anchor)
    for h in hashrefs:
        if "#" + h in hay or node.id == "ref:#" + h:
            score += 3.0
    # exact date match
    ndate = str(node.attrs.get("date", ""))
    for d in dates:
        if d in hay or d == ndate:
            score += 2.0
    return score


def _select_seeds(graph: Graph, query: str, max_seeds: int) -> list[tuple[float, str]]:
    """Seed-selection step of `reconstruct()`, factored so `reconstruct_lensed()`
    (the #346 retrieval-first lensed path) shares the IDENTICAL seed scorer
    rather than re-deriving it. Pure function of (graph, query, max_seeds).
    """
    terms = tokenize(query)
    hashrefs = set(_HASHREF_RE.findall(query))
    dates = set(_DATE_RE.findall(query))
    scored: list[tuple[float, str]] = []
    for nid, node in graph.nodes.items():
        s = _score_node(node, terms, hashrefs, dates)
        if s > 0:
            scored.append((s, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[:max_seeds]


def _bfs_reconstruct(
    graph: Graph,
    seeds: list[tuple[float, str]],
    max_depth: int,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """BFS-reconstruction step of `reconstruct()`, factored for reuse.

    Accumulates weight = seed_score * decay^dist across all seed-rooted BFS
    trees (a node reached by multiple seeds accumulates multiple contributions,
    by design) and keeps a representative shortest human-readable path per
    node. Pure function of (graph, seeds, max_depth); `reconstruct()` and
    `reconstruct_lensed()` both call this so their base traversal is identical
    — the lensed path never re-derives (and cannot silently diverge from) the
    naive BFS pool it augments.
    """
    weight: dict[str, float] = {}
    best_path: dict[str, list[str]] = {}
    for seed_score, seed_id in seeds:
        # frontier BFS with per-node first-seen path
        frontier: list[tuple[str, int, list[str]]] = [(seed_id, 0, [_label(graph, seed_id)])]
        seen_local: set[str] = set()
        while frontier:
            nid, dist, path = frontier.pop(0)
            if nid in seen_local:
                continue
            seen_local.add(nid)
            w = seed_score * (_DECAY ** dist)
            weight[nid] = weight.get(nid, 0.0) + w
            if nid not in best_path or len(path) < len(best_path[nid]):
                best_path[nid] = path
            if dist >= max_depth:
                continue
            # expand: both out and in edges (backlinks), stable order
            nbrs = sorted(graph.neighbors(nid), key=lambda t: (t[0].etype, t[2]))
            for edge, direction, other in nbrs:
                if other in seen_local:
                    continue
                arrow = f"--{edge.etype}-->" if direction == "out" else f"<--{edge.etype}--"
                frontier.append((other, dist + 1, [*path, arrow, _label(graph, other)]))
    return weight, best_path


def reconstruct(
    graph: Graph,
    query: str,
    *,
    max_seeds: int = 4,
    max_nodes: int = 18,
    max_depth: int = _MAX_DEPTH,
) -> Reconstruction:
    """Traverse the graph from query-matched seeds and assemble connected context."""
    # 1) seed selection — score every node, keep the best (stable tie-break on id)
    seeds = _select_seeds(graph, query, max_seeds)

    # 2) BFS reconstruction — accumulate weight = seed_score * decay^dist across
    #    all paths; keep a representative human path (shortest, then best seed).
    weight, best_path = _bfs_reconstruct(graph, seeds, max_depth)

    # 3) rank the reconstructed subgraph; seeds first, then by accumulated weight
    seed_ids = {s for _, s in seeds}
    ranked = sorted(
        weight.items(),
        key=lambda kv: (0 if kv[0] in seed_ids else 1, -kv[1], kv[0]),
    )
    ordered = [nid for nid, _ in ranked[:max_nodes]]

    return Reconstruction(
        query=query,
        seeds=[(sid, sc) for sc, sid in seeds],
        nodes=ordered,
        paths={nid: best_path.get(nid, []) for nid in ordered},
        graph_counts={
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "by_type": graph.counts_by_type(),
        },
    )


def _label(graph: Graph, nid: str) -> str:
    if nid in graph.nodes:
        n = graph.nodes[nid]
        return f"[{n.ntype}] {n.title[:60]}"
    return nid


def format_human(recon: Reconstruction, graph: Graph, *, max_summary: int = 260) -> str:
    """Render the reconstruction as an assembled context block (not raw chunks)."""
    lines: list[str] = []
    lines.append(f"RECONSTRUCTED CONTEXT for: {recon.query!r}")
    gc = recon.graph_counts
    lines.append(f"(graph: {gc.get('nodes', '?')} nodes, {gc.get('edges', '?')} edges)")
    if not recon.seeds:
        lines.append("  no matching seed nodes — query terms absent from the graph.")
        return "\n".join(lines)
    lines.append("SEEDS (query-matched anchors):")
    for sid, sc in recon.seeds:
        title = graph.nodes[sid].title if sid in graph.nodes else sid
        lines.append(f"  • {sid}  (score {sc:.2f})  {title[:70]}")
    lines.append("ASSEMBLED from the connected subgraph:")
    for nid in recon.nodes:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        summ = node.summary[:max_summary].strip()
        tag = f"[{node.ntype}]"
        lines.append(f"  ── {tag} {node.title[:80]}")
        if summ:
            lines.append(f"      {summ}")
        if node.source:
            lines.append(f"      src: {node.source}")
        path = recon.paths.get(nid, [])
        if path and len(path) > 1:
            lines.append("      via: " + " ".join(path))
    return "\n".join(lines)


# ------------------------------------------------------- lensed recall (#346) --
# Additive, opt-in augmentation of `reconstruct()` via `tac.lens_engine` (the
# increment-1 typed multi-lens analyzer over T=(E,G,Phi,S,L,R,X)). Keyword-BFS
# reconstruction only ever asks "which nodes are near a seed by hop count and
# accumulated keyword weight" — it cannot see (a) a node that structurally sits
# on the shortest path connecting two seeds but scored low on keyword-weight,
# (b) which of the BFS-reached pool is a genuine network hub/bridge by
# centrality, or (c) a graph-topological crux (a discrete saddle in the Phi
# field) near the query. `tac.lens_engine`'s GRAPH and TOPOLOGY lenses already
# implement centrality/shortest-path/saddle detection over exactly the same
# `tac.graph_memory.Graph`; this module composes them rather than
# reimplementing them. `tac.lens_engine` is imported LAZILY inside
# `reconstruct_lensed()` (not at module scope) because
# `tac.lens_engine.adapters` imports FROM `tac.graph_memory` — a top-level
# import here would circular-import the very first time anything imports
# `tac.graph_memory` before `tac.lens_engine` (confirmed by direct
# reproduction 2026-07-12: `cache_paths` is defined after the `from .recall
# import ...` line in `tac/graph_memory/__init__.py`, so a partially
# initialized module would be missing it).


@dataclass
class LensedReconstruction:
    """Additive augmentation of a base `Reconstruction` via `tac.lens_engine`.

    `nodes` is `base.nodes` PLUS any bridge/hub/crux additions — a strict
    superset by construction; nothing the naive keyword-BFS reconstruction
    found is ever dropped or reordered. Each addition records WHY it was added
    so a reader (or a downstream SENSE/costate consumer) can audit the lens's
    marginal contribution independently of the base reconstruction.
    """

    base: Reconstruction
    bridge_nodes: list[tuple[str, tuple[str, str]]]  # (node_id, (seed_a, seed_b) path)
    hub_nodes: list[tuple[str, float]]  # (node_id, centrality score)
    crux_nodes: list[tuple[str, float]]  # (node_id, phi at the saddle)
    nodes: list[str]  # base.nodes + new additions, in that priority order
    phi_mode: str
    centrality_method: str
    local_pool_size: int
    local_pool_capped: bool

    def to_dict(self, graph: Graph) -> dict:
        base_dict = self.base.to_dict(graph)

        def _entry(nid: str, why: str) -> dict:
            node = graph.nodes.get(nid)
            return {
                "id": nid,
                "ntype": node.ntype if node else "?",
                "title": node.title if node else nid,
                "why": why,
            }

        return {
            **base_dict,
            "lensed": True,
            "phi_mode": self.phi_mode,
            "centrality_method": self.centrality_method,
            "local_pool_size": self.local_pool_size,
            "local_pool_capped": self.local_pool_capped,
            "bridge_nodes": [
                _entry(nid, f"GRAPH shortest_path {a}<->{b}") for nid, (a, b) in self.bridge_nodes
            ],
            "hub_nodes": [
                _entry(nid, f"GRAPH centrality[{self.centrality_method}]={score:.4f}")
                for nid, score in self.hub_nodes
            ],
            "crux_nodes": [
                _entry(nid, f"TOPOLOGY saddle phi={phi:.2f}") for nid, phi in self.crux_nodes
            ],
            "nodes_lensed": self.nodes,
        }


def _local_pool_graph(
    graph: Graph,
    pool_ids: set[str],
    weight: dict[str, float],
    *,
    pool_cap: int,
) -> tuple[Graph, bool]:
    """Build the bounded induced subgraph `tac.lens_engine` centrality/topology
    run over. Full-corpus centrality was MEASURED at ~91s for closeness over
    all 8,842 nodes / 30,616 edges of the live corpus cache (2026-07-12) — far
    too slow for an interactive recall call — while betweenness over a
    ~950-node query-local BFS pool measured ~1.8s. `pool_ids` is therefore
    always the query's own BFS-reached candidate set (already query-relevant
    by construction), never the whole graph; when it still exceeds `pool_cap`
    it is deterministically truncated to the highest-BFS-weight members (never
    dropping a node already required to stay in the pool).
    """
    capped = False
    if len(pool_ids) > pool_cap:
        required = {nid for nid in pool_ids if nid not in weight}
        ranked_pool = sorted(
            (nid for nid in pool_ids if nid in weight),
            key=lambda nid: (-weight[nid], nid),
        )
        budget = max(pool_cap - len(required), 0)
        pool_ids = required | set(ranked_pool[:budget])
        capped = True
    local = Graph()
    for nid in sorted(pool_ids):
        if nid in graph.nodes:
            local.add_node(graph.nodes[nid])
    for key in sorted(graph.edges):
        edge = graph.edges[key]
        if edge.src in pool_ids and edge.dst in pool_ids:
            local.add_edge(edge)
    return local, capped


def reconstruct_lensed(
    graph: Graph,
    query: str,
    *,
    max_seeds: int = 4,
    max_nodes: int = 18,
    max_depth: int = _MAX_DEPTH,
    phi: str = "degree",
    centrality_method: str = "betweenness",
    max_bridge_nodes: int = 8,
    max_hub_nodes: int = 4,
    max_crux_nodes: int = 3,
    centrality_pool_cap: int = 800,
) -> LensedReconstruction:
    """Additive, opt-in lensed recall: base keyword-BFS reconstruction (byte-
    identical to `reconstruct()`) augmented with three `tac.lens_engine`
    findings keyword-BFS structurally cannot see:

    1. **BRIDGE nodes** — GRAPH lens `shortest_path` between every pair of
       seeds on the FULL corpus graph (not depth-limited). This rescues nodes
       that ARE reachable within `max_depth` but got truncated by the naive
       weight ranking (measured 2026-07-12: for the query "lane d_seg", the
       node `feed:phase-advect-build` sits on the actual shortest path linking
       3 of 4 seeds through `person:Yousfi`, yet ranked #20 by keyword-BFS
       weight — one past the default `max_nodes=18` cutoff). The mechanism is
       also unbounded in principle (a bridge can lie beyond `max_depth`
       entirely), though that was not observed on the three canonical probe
       queries in the landing memo.
    2. **HUB nodes** — GRAPH lens `centrality` (betweenness by default) over
       the BOUNDED induced subgraph of every node the base BFS actually
       reached (the untruncated pool; see `_local_pool_graph` for why this
       must be query-local rather than corpus-wide).
    3. **CRUX nodes** — TOPOLOGY lens `saddles` on the same local pool: nodes
       structurally between two-or-more "sides" of the local Phi field. This
       is a discrete graph-filtration saddle per `tac.lens_engine.topology`'s
       own honesty boundary (a connectivity/degree crux), not a claim of
       smooth Morse-Smale reconstruction.

    Phi is explicit and custodied exactly as `tac.lens_engine.CorpusAdapter`
    requires: `"degree"` (the safe structural default) is the only mode this
    function accepts without a caller-supplied mapping/callable, because the
    graph-memory cache does not yet custody `citation_salience`/`recency`
    values for every node (per the lens_engine increment-1 landing memo).
    Passing an unsupported string raises the SAME `AdapterError`
    `CorpusAdapter` raises; this function never fabricates a fallback value.

    `nodes` is `base.nodes` plus any new bridge/hub/crux ids, in that priority
    order — a strict superset; nothing from the base reconstruction is ever
    dropped. `reconstruct()` itself is neither called nor mutated by this
    function beyond sharing its `_select_seeds`/`_bfs_reconstruct` helpers
    (see the `test_reconstruct_unchanged_after_lensed_refactor` regression
    guard), so this path is strictly additive and opt-in.
    """
    from tac.lens_engine import (
        GRAPH,
        TOPOLOGY,
        AdapterError,
        CorpusAdapter,
        LensOperationError,
        QueryError,
    )
    from tac.lens_engine import query as lens_query

    seeds = _select_seeds(graph, query, max_seeds)
    weight, best_path = _bfs_reconstruct(graph, seeds, max_depth)
    seed_ids_ordered = [sid for _, sid in seeds]
    seed_id_set = {s for _, s in seeds}

    ranked = sorted(
        weight.items(),
        key=lambda kv: (0 if kv[0] in seed_id_set else 1, -kv[1], kv[0]),
    )
    base_ordered = [nid for nid, _ in ranked[:max_nodes]]
    base = Reconstruction(
        query=query,
        seeds=[(sid, sc) for sc, sid in seeds],
        nodes=base_ordered,
        paths={nid: best_path.get(nid, []) for nid in base_ordered},
        graph_counts={
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "by_type": graph.counts_by_type(),
        },
    )
    base_ids = set(base_ordered)

    if not seed_ids_ordered:
        return LensedReconstruction(
            base=base, bridge_nodes=[], hub_nodes=[], crux_nodes=[],
            nodes=list(base_ordered), phi_mode=phi, centrality_method=centrality_method,
            local_pool_size=0, local_pool_capped=False,
        )

    # ---- 1) bridge nodes: pairwise GRAPH shortest_path between seeds, full graph --
    # (measured 2026-07-12: ~40ms per call on the live 8,842-node/30,616-edge
    # corpus cache — cheap enough to run on the full graph unbounded.)
    full_adapter = CorpusAdapter(graph, phi=phi)
    raw_bridges: list[tuple[str, tuple[str, str]]] = []
    seen_bridge: set[str] = set()
    for i, seed_a in enumerate(seed_ids_ordered):
        for seed_b in seed_ids_ordered[i + 1:]:
            try:
                res = lens_query(
                    full_adapter, GRAPH, "shortest_path",
                    start=seed_a, target=seed_b, directed=False,
                )
            except (QueryError, LensOperationError):
                continue
            sp = res.value
            if not sp.reachable:
                continue
            for nid in sp.nodes:
                if nid in base_ids or nid in seen_bridge:
                    continue
                seen_bridge.add(nid)
                raw_bridges.append((nid, (seed_a, seed_b)))
    raw_bridges.sort(key=lambda t: t[0])
    bridge_nodes = raw_bridges[:max_bridge_nodes]
    included_ids = base_ids | {nid for nid, _ in bridge_nodes}

    # ---- 2/3) local bounded pool for centrality + topology (see docstring) --
    pool_ids = set(weight) | included_ids
    local_graph, local_pool_capped = _local_pool_graph(
        graph, pool_ids, weight, pool_cap=centrality_pool_cap,
    )
    try:
        local_adapter = CorpusAdapter(local_graph, phi=phi)
    except AdapterError:
        raise

    hub_nodes: list[tuple[str, float]] = []
    try:
        cres = lens_query(local_adapter, GRAPH, "centrality", method=centrality_method, weighted=False)
        for nid, score in cres.value.scores:
            if len(hub_nodes) >= max_hub_nodes:
                break
            if nid in included_ids:
                continue
            hub_nodes.append((nid, score))
    except (QueryError, LensOperationError):
        hub_nodes = []
    included_ids |= {nid for nid, _ in hub_nodes}

    crux_nodes: list[tuple[str, float]] = []
    try:
        tres = lens_query(local_adapter, TOPOLOGY, "saddles")
        near = included_ids | seed_id_set
        candidates = []
        for cp in tres.value:
            if cp.element_id in included_ids:
                continue
            is_near = cp.element_id in near or any(
                other in near for _edge, _direction, other in graph.neighbors(cp.element_id)
            )
            if is_near:
                candidates.append(cp)
        candidates.sort(key=lambda cp: (-cp.phi, cp.element_id))
        crux_nodes = [(cp.element_id, cp.phi) for cp in candidates[:max_crux_nodes]]
    except (QueryError, LensOperationError):
        crux_nodes = []

    final_nodes = list(base_ordered)
    for nid, _ in (*bridge_nodes, *hub_nodes, *crux_nodes):
        if nid not in final_nodes:
            final_nodes.append(nid)

    return LensedReconstruction(
        base=base,
        bridge_nodes=bridge_nodes,
        hub_nodes=hub_nodes,
        crux_nodes=crux_nodes,
        nodes=final_nodes,
        phi_mode=str(phi),
        centrality_method=centrality_method,
        local_pool_size=len(local_graph.nodes),
        local_pool_capped=local_pool_capped,
    )


def format_human_lensed(lensed: LensedReconstruction, graph: Graph, *, max_summary: int = 260) -> str:
    """Render `format_human()`'s base block plus the lensed additions."""
    lines = [format_human(lensed.base, graph, max_summary=max_summary)]
    lines.append("")
    lines.append(
        f"LENS ENGINE augmentation (phi={lensed.phi_mode!r}, "
        f"centrality={lensed.centrality_method!r}, "
        f"local_pool={lensed.local_pool_size}{' capped' if lensed.local_pool_capped else ''}):"
    )
    if not any((lensed.bridge_nodes, lensed.hub_nodes, lensed.crux_nodes)):
        lines.append("  (no new nodes beyond the base reconstruction on this query)")
        return "\n".join(lines)

    def _line(nid: str, why: str) -> str:
        node = graph.nodes.get(nid)
        title = node.title[:70] if node else nid
        ntype = node.ntype if node else "?"
        return f"  + [{ntype}] {title}  ({why})"

    for nid, (a, b) in lensed.bridge_nodes:
        lines.append(_line(nid, f"BRIDGE {a} <-> {b}"))
    for nid, score in lensed.hub_nodes:
        lines.append(_line(nid, f"HUB {lensed.centrality_method}={score:.4f}"))
    for nid, phi_value in lensed.crux_nodes:
        lines.append(_line(nid, f"CRUX saddle phi={phi_value:.2f}"))
    return "\n".join(lines)
