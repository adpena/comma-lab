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

from .model import Graph, Node

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


def reconstruct(
    graph: Graph,
    query: str,
    *,
    max_seeds: int = 4,
    max_nodes: int = 18,
    max_depth: int = _MAX_DEPTH,
) -> Reconstruction:
    """Traverse the graph from query-matched seeds and assemble connected context."""
    terms = tokenize(query)
    hashrefs = set(_HASHREF_RE.findall(query))
    dates = set(_DATE_RE.findall(query))

    # 1) seed selection — score every node, keep the best (stable tie-break on id)
    scored: list[tuple[float, str]] = []
    for nid, node in graph.nodes.items():
        s = _score_node(node, terms, hashrefs, dates)
        if s > 0:
            scored.append((s, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    seeds = scored[:max_seeds]

    # 2) BFS reconstruction — accumulate weight = seed_score * decay^dist across
    #    all paths; keep a representative human path (shortest, then best seed).
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
                frontier.append((other, dist + 1, path + [arrow, _label(graph, other)]))

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
