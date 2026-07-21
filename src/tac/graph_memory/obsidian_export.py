# SPDX-License-Identifier: MIT
"""Obsidian round-trip export of the SYNTHESIZED graph edges (task #411 inc-2).

The corpus is already an Obsidian-compatible vault: memory notes carry
`[[wikilink]]` edges Obsidian reads natively. But the graph-memory SYNTHESIZES
many edges that live ONLY in the graph, not in the markdown — a DAG FEED that
`references` a memory/lever/file, an equation's `produces`/`consumes`, a
`supersedes` chain, `sister` relations, task `blocks`. Those synthesized edges
are invisible in Obsidian.

This exporter writes them back as ONE generated Obsidian-navigable index note:
every synthesized edge whose target is a memory becomes a real `[[wikilink]]`
(so Obsidian's graph view + backlinks light up), and non-memory targets are
listed with their type. The round-trip: corpus -> graph (synthesize) -> markdown
(re-materialize) -> Obsidian graph. It is a GENERATED, REBUILDABLE index (written
to the cache dir, NOT into the source-of-truth memory files) — deterministic and
idempotent, so re-exporting an unchanged graph yields byte-identical output.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .build import REPO_ROOT

if TYPE_CHECKING:
    from .model import Graph

# The edge kinds NOT already present as [[wikilinks]] in the markdown corpus.
# (`links` IS the wikilink edge — already visible in Obsidian, so it is excluded.)
_SYNTHESIZED_ETYPES: tuple[str, ...] = (
    "references", "supersedes", "sister", "produces", "consumes", "blocks", "tagged",
    "indexed_by", "aliases", "memo_link", "equation_ref", "task_ref", "feed_ref",
    "lane_ref", "catalog_ref",
)

_DEFAULT_OUT = REPO_ROOT / ".omx" / "state" / "graph_memory" / "synthesized_edges.md"


def _wikilink_or_label(graph: Graph, node_id: str) -> str:
    """A memory target -> `[[slug]]` (Obsidian-navigable); else a typed label."""
    node = graph.nodes.get(node_id)
    if node is None:
        return f"`{node_id}`"
    if node.ntype == "memory":
        # Obsidian resolves [[slug]] the same way the corpus already links memories.
        return f"[[{node.title}]]"
    return f"`{node.title}` ({node.ntype})"


def render_obsidian(graph: Graph, *, etypes: tuple[str, ...] = _SYNTHESIZED_ETYPES) -> str:
    """Render the synthesized edges as a deterministic Obsidian index note."""
    want = frozenset(etypes)
    # group edges by source node -> {etype: [dst, ...]}
    by_src: dict[str, dict[str, list[str]]] = {}
    for edge in graph.edges.values():
        if edge.etype not in want:
            continue
        by_src.setdefault(edge.src, {}).setdefault(edge.etype, []).append(edge.dst)

    n_edges = sum(len(dsts) for rels in by_src.values() for dsts in rels.values())
    lines: list[str] = []
    lines.append("---")
    lines.append("generated: tac.graph_memory.obsidian_export  # DO NOT EDIT — rebuildable")
    lines.append("kind: synthesized-edge-index")
    lines.append("---")
    lines.append("")
    lines.append("# Graph-memory — synthesized edges (Obsidian round-trip)")
    lines.append("")
    lines.append(
        f"Materialized from {len(graph.nodes)} nodes / {len(graph.edges)} edges: "
        f"{n_edges} SYNTHESIZED edges (the ones NOT already `[[wikilinks]]` in the corpus). "
        "Open in Obsidian to browse the reconstructed graph; `[[links]]` resolve to memory notes."
    )
    lines.append("")
    for src in sorted(by_src):
        src_node = graph.nodes.get(src)
        heading = src_node.title if src_node else src
        stype = f" ({src_node.ntype})" if src_node else ""
        lines.append(f"## {heading}{stype}")
        if src_node and src_node.source:
            lines.append(f"<!-- src: {src_node.source} -->")
        rels = by_src[src]
        for etype in sorted(rels):
            targets = sorted(set(rels[etype]))
            rendered = ", ".join(_wikilink_or_label(graph, t) for t in targets)
            lines.append(f"- **{etype}**: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_obsidian(
    graph: Graph,
    out_path: Path | None = None,
    *,
    etypes: tuple[str, ...] = _SYNTHESIZED_ETYPES,
) -> Path:
    """Write the synthesized-edge Obsidian index; returns the path written."""
    path = Path(out_path) if out_path is not None else _DEFAULT_OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_obsidian(graph, etypes=etypes), encoding="utf-8")
    return path


__all__ = ["export_obsidian", "render_obsidian"]
