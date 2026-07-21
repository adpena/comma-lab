# SPDX-License-Identifier: MIT
"""Graph-memory data model — nodes + edges over the durable corpus (task #411).

The substrate is the SHARED WIKILINK MARKDOWN GRAPH (Obsidian-compatible): the
memory files already carry `[[wikilink]]` edges + YAML frontmatter, and the DAG
FEED blocks / equation registry / task ledger carry references we synthesize into
edges. This module holds the typed node/edge model + a small deterministic graph
container with save/load (JSONL) + traversal. The markdown FEEDs / memory files
REMAIN the source of truth; this graph INDEXES them (does not replace them).

MRAgent thesis ("Memory is Reconstructed, Not Retrieved"): recall is graph
traversal + on-demand context assembly, NOT flat chunk retrieval. `model` is the
graph; `recall` is the reconstruction loop over it.

Deterministic by construction: node/edge ids are derived from source content, the
save format sorts by id, and there are zero external dependencies + zero RNG.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Canonical node types (namespaced id prefix == type where practical).
NODE_TYPES: tuple[str, ...] = (
    "memory",    # a ~/.claude/.../memory/*.md file (frontmatter name == id)
    "finding",   # a DAG FEED-* block
    "topic",     # a frontmatter `type` / discipline grouping
    "entity",    # a lever #NNN, a file path, a task ref
    "person",    # a council member / named person
    "equation",  # a canonical_equations registry entry
    "task",      # a canonical_task_status entry
    "deferral",  # a deferral-ledger row
    "decision",  # a finding/memory carrying a verdict (CONFIRMED/NO-GO/VERDICT/...)
    "regime",    # a #426 costate-organ prototype regime (costate_organ_trajectory_ledger)
    "index",     # MEMORY.md / cluster / full-index navigation document
    "section",   # doctrine/document section addressable by a stable anchor slug
    "research",  # a dated .omx/research memo outside the monolithic DAG
    "lane",      # a canonical lane_registry lane
    "catalog",   # a Catalog #NNN doctrine row
)

# Canonical edge types (directed src -> dst).
EDGE_TYPES: tuple[str, ...] = (
    "links",       # [[wikilink]] : source note -> target note
    "references",  # FEED/memo -> task/equation/memory/file/lever it mentions
    "supersedes",  # supersession chain
    "blocks",      # task -> task it blocks (blocker -> blocked)
    "produces",    # tool/module -> equation it produces
    "consumes",    # module -> equation it consumes
    "sister",      # "sister of" relation (symmetric-intent, stored directed)
    "tagged",      # node -> topic
    "indexed_by",  # memory note -> MEMORY/cluster/full-index document
    "aliases",     # wikilink-compatible alias -> canonical section/task node
    "memo_link",   # research memo -> local markdown-linked memo
    "equation_ref",# memo/document -> canonical equation
    "task_ref",    # memo/document -> #NNN entity/task alias
    "feed_ref",    # memo/document -> DAG FEED-* node
    "lane_ref",    # memo/document -> lane_registry lane
    "catalog_ref", # memo/document -> Catalog #NNN row
)


@dataclass(frozen=True, slots=True)
class Node:
    """A single graph node. `id` is unique + deterministic from source content."""

    id: str
    ntype: str
    title: str
    summary: str = ""
    source: str = ""  # path or path#Lstart-Lend
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> Node:
        return Node(
            id=d["id"],
            ntype=d["ntype"],
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            source=d.get("source", ""),
            attrs=dict(d.get("attrs", {})),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """A directed typed edge with provenance (where the edge was found)."""

    src: str
    dst: str
    etype: str
    source: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.etype)

    def to_dict(self) -> dict:
        return {"src": self.src, "dst": self.dst, "etype": self.etype, "source": self.source}

    @staticmethod
    def from_dict(d: dict) -> Edge:
        return Edge(src=d["src"], dst=d["dst"], etype=d["etype"], source=d.get("source", ""))


class Graph:
    """A small in-memory node/edge graph with deterministic save/load + traversal.

    Adjacency is maintained both forward (out-edges) and backward (in-edges) so
    reconstruction can walk both directions (a finding -> the memory it cites, AND
    a memory -> every finding that cites it = backlinks, the Obsidian view).
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}

    # ---- mutation -----------------------------------------------------------
    def add_node(self, node: Node) -> None:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return
        # Merge: prefer a richer (longer summary / real type over placeholder)
        # so that a node first seen as a bare reference is upgraded when its
        # canonical source is parsed. Deterministic: longer summary wins; on tie
        # keep the more specific (non-entity) type.
        better_summary = node.summary if len(node.summary) > len(existing.summary) else existing.summary
        ntype = existing.ntype
        placeholder = not existing.source and not existing.summary
        if (existing.ntype == "entity" and node.ntype != "entity") or placeholder:
            ntype = node.ntype
        title = node.title if placeholder and node.title else (existing.title or node.title)
        source = existing.source or node.source
        merged_attrs = {**node.attrs, **existing.attrs}
        self.nodes[node.id] = Node(
            id=node.id, ntype=ntype, title=title, summary=better_summary,
            source=source, attrs=merged_attrs,
        )

    def add_edge(self, edge: Edge) -> None:
        if edge.src == edge.dst:
            return  # no self-loops
        k = edge.key()
        if k in self.edges:
            return
        self.edges[k] = edge
        self._out.setdefault(edge.src, []).append(edge)
        self._in.setdefault(edge.dst, []).append(edge)

    def ensure_stub(self, node_id: str, ntype: str = "entity", title: str = "") -> None:
        """Register a bare reference target if it does not yet exist."""
        if node_id not in self.nodes:
            self.add_node(Node(id=node_id, ntype=ntype, title=title or node_id))

    # ---- queries ------------------------------------------------------------
    def out_edges(self, node_id: str) -> list[Edge]:
        return self._out.get(node_id, [])

    def in_edges(self, node_id: str) -> list[Edge]:
        return self._in.get(node_id, [])

    def neighbors(self, node_id: str) -> Iterator[tuple[Edge, str, str]]:
        """Yield (edge, direction, other_id) over both out- and in-edges."""
        for e in self._out.get(node_id, []):
            yield e, "out", e.dst
        for e in self._in.get(node_id, []):
            yield e, "in", e.src

    def degree(self, node_id: str) -> int:
        return len(self._out.get(node_id, [])) + len(self._in.get(node_id, []))

    def counts_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self.nodes.values():
            out[n.ntype] = out.get(n.ntype, 0) + 1
        return dict(sorted(out.items()))

    def edge_counts_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.edges.values():
            out[e.etype] = out.get(e.etype, 0) + 1
        return dict(sorted(out.items()))

    # ---- persistence (deterministic; rebuildable) ---------------------------
    def save(self, nodes_path: Path, edges_path: Path) -> None:
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        with nodes_path.open("w", encoding="utf-8") as fh:
            for nid in sorted(self.nodes):
                fh.write(json.dumps(self.nodes[nid].to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        with edges_path.open("w", encoding="utf-8") as fh:
            for k in sorted(self.edges):
                fh.write(json.dumps(self.edges[k].to_dict(), sort_keys=True, ensure_ascii=False) + "\n")

    @staticmethod
    def load(nodes_path: Path, edges_path: Path) -> Graph:
        g = Graph()
        with Path(nodes_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    g.add_node(Node.from_dict(json.loads(line)))
        with Path(edges_path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    g.add_edge(Edge.from_dict(json.loads(line)))
        return g

    def extend(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        for n in nodes:
            self.add_node(n)
        for e in edges:
            self.add_edge(e)
