# SPDX-License-Identifier: MIT
"""Deterministic link-hygiene measurements over the graph-memory corpus."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from .build import (
    _RESEARCH_DIR,
    _WIKILINK_RE,
    _memory_dir,
    _memory_node_id_for_path,
    wikilink_target,
)

if TYPE_CHECKING:
    from .model import Graph


def canonical_sweep_files(
    *, memory_dir: Path | None = None, research_dir: Path | None = None,
) -> list[Path]:
    """Return the operator-audit corpus: memory/*.md + top-level research/*.md."""
    mdir = memory_dir or _memory_dir()
    rdir = research_dir or _RESEARCH_DIR
    return sorted(mdir.glob("*.md")) + sorted(rdir.glob("*.md"))


def _alias_resolves(graph: Graph, node_id: str) -> bool:
    return any(
        edge.etype == "aliases"
        and (target := graph.nodes.get(edge.dst)) is not None
        and bool(target.source)
        for edge in graph.out_edges(node_id)
    )


def measure_link_hygiene(
    graph: Graph,
    *,
    memory_dir: Path | None = None,
    research_dir: Path | None = None,
) -> dict:
    """Measure resolution, reachability, dangling classes, and typed families."""
    mdir = memory_dir or _memory_dir()
    files = canonical_sweep_files(memory_dir=mdir, research_dir=research_dir)
    raw_targets: list[str] = []
    semantic_targets: list[str] = []
    filtered_targets: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _WIKILINK_RE.finditer(text):
            raw = match.group(1).split("|", 1)[0].strip()
            raw_targets.append(raw)
            target = wikilink_target(match.group(1))
            if target is None:
                filtered_targets.append(raw)
            else:
                semantic_targets.append(target)

    direct: list[str] = []
    alias: list[str] = []
    unwritten: list[str] = []
    for target in semantic_targets:
        node_id = "memory:" + target
        node = graph.nodes.get(node_id)
        if node is not None and node.source:
            direct.append(target)
        elif node is not None and _alias_resolves(graph, node_id):
            alias.append(target)
        else:
            unwritten.append(target)

    notes = [p for p in sorted(mdir.glob("*.md")) if p.name != "MEMORY.md"]
    note_ids = [_memory_node_id_for_path(path) for path in notes]
    unreachable: list[str] = []
    for node_id in note_ids:
        incoming = graph.in_edges(node_id)
        discoverable = any(e.etype in {"links", "references", "memo_link"} for e in incoming)
        if not discoverable:
            # A filename alias is useful only when some corpus node actually
            # points at that alias.  Counting an unreferenced alias stub would
            # manufacture a zero-orphan result without adding reachability.
            discoverable = any(
                e.etype == "aliases"
                and any(parent.etype != "aliases" for parent in graph.in_edges(e.src))
                for e in incoming
            )
        indexed = any(e.etype == "indexed_by" for e in graph.out_edges(node_id))
        if not discoverable and not indexed:
            unreachable.append(node_id.removeprefix("memory:"))

    family_names = (
        "indexed_by", "aliases", "memo_link", "equation_ref", "task_ref",
        "feed_ref", "lane_ref", "catalog_ref",
    )
    edge_counts = graph.edge_counts_by_type()
    resolved_total = len(direct) + len(alias)
    semantic_total = len(semantic_targets)
    return {
        "schema": "graph_memory_link_hygiene.v1",
        "files_scanned": len(files),
        "raw_wikilinks": len(raw_targets),
        "semantic_wikilinks": semantic_total,
        "resolved_direct": len(direct),
        "resolved_by_alias": len(alias),
        "resolved_total": resolved_total,
        "resolution_pct_semantic": round(100.0 * resolved_total / semantic_total, 6)
        if semantic_total else 0.0,
        "false_positive_filtered": len(filtered_targets),
        "false_positive_distinct": len(set(filtered_targets)),
        "truly_unwritten": len(unwritten),
        "truly_unwritten_distinct": len(set(unwritten)),
        "memory_notes": len(note_ids),
        "unreachable_orphans": len(unreachable),
        "unreachable_orphan_pct": round(100.0 * len(unreachable) / len(note_ids), 6)
        if note_ids else 0.0,
        "graph_nodes": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "edge_counts_by_family": {name: edge_counts.get(name, 0) for name in family_names},
        "top_truly_unwritten": Counter(unwritten).most_common(25),
        "top_filtered_false_positives": Counter(filtered_targets).most_common(25),
    }


__all__ = ["canonical_sweep_files", "measure_link_hygiene"]
