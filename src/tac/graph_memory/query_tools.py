# SPDX-License-Identifier: MIT
"""Typed query-tools over the graph-memory (task #411 increment-2).

MRAgent's QA phase is a tool-calling loop over typed query tools, not one
generic search: the agent asks a PRECISE question ("what happened around
2026-07-02?", "what references #205?", "what supersedes X?") and the graph
answers by a narrow deterministic traversal. This module gives the graph-memory
that same typed surface — seven tools, each answering ONE kind of question:

    query_by_time            -- findings/decisions in a date or date-range
    query_by_keywords        -- token-scored nodes (the reconstruct() seed scorer)
    query_by_entity          -- a lever #NNN / file / person + everything about it
    query_by_topic           -- nodes tagged to a topic/discipline
    query_by_decision        -- verdict-carrying nodes (optionally by verdict)
    query_neighbors          -- the 1-hop typed neighborhood of a node
    query_supersession_chain -- follow the supersedes edges transitively

Each returns a QueryResult (a typed, deterministic, JSON-able hit list) rather
than prose — so a recall loop, the costate digest, or a human can compose them.
Deterministic by construction: pure scoring + sorted traversal, stable id
tie-breaks, zero RNG.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .recall import _score_node, tokenize

if TYPE_CHECKING:
    from .model import Graph, Node

_HASHREF_RE = re.compile(r"#?(\d{2,4})\b")
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


@dataclass(frozen=True, slots=True)
class QueryHit:
    """One node returned by a typed query, with WHY it matched."""

    id: str
    ntype: str
    title: str
    summary: str = ""
    source: str = ""
    why: str = ""  # the edge/attr that made this a hit (provenance for the reader)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "ntype": self.ntype, "title": self.title,
            "summary": self.summary, "source": self.source, "why": self.why,
        }


@dataclass
class QueryResult:
    """The typed result of one query-tool call."""

    kind: str
    query: str
    hits: list[QueryHit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "query": self.query,
                "hits": [h.to_dict() for h in self.hits]}

    def ids(self) -> list[str]:
        return [h.id for h in self.hits]


def _hit(node: Node, why: str = "") -> QueryHit:
    return QueryHit(
        id=node.id, ntype=node.ntype, title=node.title,
        summary=node.summary, source=node.source, why=why,
    )


# --------------------------------------------------------------- 1) by time --
def query_by_time(
    graph: Graph, *, date: str = "", since: str = "", until: str = "", limit: int = 25,
) -> QueryResult:
    """Nodes whose recorded date matches `date` OR falls in [`since`, `until`].

    Dates come from node.attrs['date'] (DAG FEED blocks stamp one) or, failing
    that, a YYYY-MM-DD found in the id/title/summary. Lexicographic compare is
    correct for ISO dates. Ordered newest-first, then by id.
    """
    q = date or f"{since}..{until}"
    hits: list[tuple[str, QueryHit]] = []
    for nid, node in graph.nodes.items():
        nd = str(node.attrs.get("date", ""))
        if not nd:
            m = _DATE_RE.search(f"{nid} {node.title} {node.summary}")
            nd = m.group(1) if m else ""
        if not nd:
            continue
        ok = (nd == date) if date else True
        if since:
            ok = ok and nd >= since
        if until:
            ok = ok and nd <= until
        if ok and (date or since or until):
            hits.append((nd, _hit(node, why=f"date={nd}")))
    hits.sort(key=lambda t: (t[0], t[1].id), reverse=True)
    return QueryResult("by_time", q, [h for _, h in hits[:limit]])


# ----------------------------------------------------------- 2) by keywords --
def query_by_keywords(graph: Graph, text: str, *, limit: int = 15) -> QueryResult:
    """Token-scored nodes (the same seed scorer reconstruct() uses), ranked."""
    terms = tokenize(text)
    hashrefs = set(re.findall(r"#(\d{2,4})\b", text))
    dates = set(_DATE_RE.findall(text))
    scored: list[tuple[float, str, Node]] = []
    for nid, node in graph.nodes.items():
        s = _score_node(node, terms, hashrefs, dates)
        if s > 0:
            scored.append((s, nid, node))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return QueryResult(
        "by_keywords", text,
        [_hit(n, why=f"score={s:.2f}") for s, _, n in scored[:limit]],
    )


# ------------------------------------------------------------- 3) by entity --
def _resolve_entity(graph: Graph, entity: str) -> str | None:
    """Map a human entity string to its node id (or None)."""
    e = entity.strip()
    if e in graph.nodes:
        return e
    # a bare/hashed number -> ref:#NNN
    m = _HASHREF_RE.fullmatch(e)
    if m and ("ref:#" + m.group(1)) in graph.nodes:
        return "ref:#" + m.group(1)
    # a file path
    if ("file:" + e) in graph.nodes:
        return "file:" + e
    # a person (case-insensitive title match)
    for pid, node in graph.nodes.items():
        if node.ntype == "person" and node.title.lower() == e.lower():
            return pid
    # a memory slug
    if ("memory:" + e) in graph.nodes:
        return "memory:" + e
    return None


def query_by_entity(graph: Graph, entity: str, *, limit: int = 30) -> QueryResult:
    """Everything about an entity (a lever #NNN / file / person / memory slug).

    Returns the entity node first, then every node that references / produces /
    consumes / blocks it (its in- and out-neighborhood), so the reader gets the
    full local context around one anchor.
    """
    eid = _resolve_entity(graph, entity)
    if eid is None:
        return QueryResult("by_entity", entity, [])
    hits: list[QueryHit] = [_hit(graph.nodes[eid], why="the entity")]
    seen = {eid}
    nbrs: list[tuple[str, QueryHit]] = []
    for edge, direction, other in graph.neighbors(eid):
        if other in seen or other not in graph.nodes:
            continue
        seen.add(other)
        arrow = "->" if direction == "out" else "<-"
        nbrs.append((other, _hit(graph.nodes[other], why=f"{arrow}{edge.etype}")))
    nbrs.sort(key=lambda t: t[0])
    hits.extend(h for _, h in nbrs[:limit])
    return QueryResult("by_entity", entity, hits)


# -------------------------------------------------------------- 4) by topic --
def query_by_topic(graph: Graph, topic: str, *, limit: int = 30) -> QueryResult:
    """Nodes tagged to a topic (frontmatter `type`/discipline grouping)."""
    tid = topic if topic.startswith("topic:") else "topic:" + topic
    if tid not in graph.nodes:
        # tolerant: match a topic node by suffix
        cands = [n for n in graph.nodes if n.startswith("topic:") and topic.lower() in n.lower()]
        if not cands:
            return QueryResult("by_topic", topic, [])
        tid = sorted(cands)[0]
    tagged: list[tuple[str, QueryHit]] = []
    for edge in graph.in_edges(tid):
        if edge.etype == "tagged" and edge.src in graph.nodes:
            tagged.append((edge.src, _hit(graph.nodes[edge.src], why=f"tagged:{tid}")))
    tagged.sort(key=lambda t: t[0])
    return QueryResult("by_topic", topic, [h for _, h in tagged[:limit]])


# ----------------------------------------------------------- 5) by decision --
def query_by_decision(graph: Graph, *, verdict: str = "", limit: int = 30) -> QueryResult:
    """Verdict-carrying nodes (decision-type OR has_verdict), optional filter.

    `verdict` (e.g. NO-GO, CONFIRMED, DEFERRED) filters on the node's recorded
    verdict attr (case-insensitive substring). Ordered newest-first by date.
    """
    want = verdict.strip().upper()
    hits: list[tuple[str, QueryHit]] = []
    for node in graph.nodes.values():
        v = str(node.attrs.get("verdict", "")).upper()
        is_decision = node.ntype == "decision" or node.attrs.get("has_verdict")
        if not is_decision:
            continue
        if want and want not in v and want not in (node.title + " " + node.summary).upper():
            continue
        nd = str(node.attrs.get("date", ""))
        hits.append((nd, _hit(node, why=f"verdict={v or 'yes'}")))
    hits.sort(key=lambda t: (t[0], t[1].id), reverse=True)
    return QueryResult("by_decision", verdict or "*", [h for _, h in hits[:limit]])


# ---------------------------------------------------------- 6) by neighbors --
def query_neighbors(
    graph: Graph, node_id: str, *, etypes: tuple[str, ...] = (), limit: int = 40,
) -> QueryResult:
    """The 1-hop typed neighborhood of a node (both directions), optional filter."""
    start = _resolve_entity(graph, node_id) or node_id
    if start not in graph.nodes:
        return QueryResult("neighbors", node_id, [])
    want = frozenset(etypes)
    out: list[tuple[str, str, QueryHit]] = []
    for edge, direction, other in graph.neighbors(start):
        if want and edge.etype not in want:
            continue
        if other not in graph.nodes:
            continue
        arrow = f"--{edge.etype}-->" if direction == "out" else f"<--{edge.etype}--"
        out.append((edge.etype, other, _hit(graph.nodes[other], why=arrow)))
    out.sort(key=lambda t: (t[0], t[1]))
    return QueryResult("neighbors", start, [h for _, _, h in out[:limit]])


# -------------------------------------------------- 7) supersession chain ----
def query_supersession_chain(graph: Graph, node_id: str, *, max_hops: int = 12) -> QueryResult:
    """Follow `supersedes` edges from a node to its latest successor.

    Returns the chain in order [start, ..., newest]. Cycle-safe (visited set),
    bounded by max_hops. Deterministic: at a fork, take the lexicographically
    smallest successor so the walk is reproducible.
    """
    start = _resolve_entity(graph, node_id) or node_id
    if start not in graph.nodes:
        return QueryResult("supersession_chain", node_id, [])
    chain: list[QueryHit] = [_hit(graph.nodes[start], why="chain start")]
    seen = {start}
    cur = start
    for _ in range(max_hops):
        nxts = sorted(
            e.dst for e in graph.out_edges(cur)
            if e.etype == "supersedes" and e.dst in graph.nodes and e.dst not in seen
        )
        if not nxts:
            break
        cur = nxts[0]
        seen.add(cur)
        chain.append(_hit(graph.nodes[cur], why="supersedes prior"))
    return QueryResult("supersession_chain", start, chain)


__all__ = [
    "QueryHit",
    "QueryResult",
    "query_by_decision",
    "query_by_entity",
    "query_by_keywords",
    "query_by_time",
    "query_by_topic",
    "query_neighbors",
    "query_supersession_chain",
]
