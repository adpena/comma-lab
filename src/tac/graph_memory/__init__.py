# SPDX-License-Identifier: MIT
"""tac.graph_memory — the DAG as a RECONSTRUCTABLE GRAPH MEMORY (task #411).

The DAG was always meant to be a reconstructable graph memory (MRAgent thesis:
"Memory is Reconstructed, Not Retrieved"), not the append-only folder of markdown
FEEDs it degraded into — the root cause of the goldfish-memory problem (memory L83:
"apparatus WRITES better than it READS"). This package builds a real NODE/EDGE
graph over the EXISTING corpus (the wikilink markdown IS the Obsidian-compatible
substrate; the FEEDs remain the source of truth) and RECONSTRUCTS context on demand
by traversing it — which dissolves the MEMORY.md flat-load cap (recall becomes
reconstruct-on-demand, not load-the-whole-index).

Public API:
    build_graph(...)        -> Graph          # parse the corpus into the graph
    load_or_build(...)      -> Graph          # load the cached graph, rebuild if stale
    reconstruct(graph, q)   -> Reconstruction # traverse + assemble context for a query
    recall(query)           -> Reconstruction # convenience: load_or_build + reconstruct

Storage: `.omx/state/graph_memory/{nodes,edges}.jsonl` — deterministic + rebuildable
(NOT the source of truth; a cache/index over the markdown corpus).
"""
from __future__ import annotations

from pathlib import Path

from .build import REPO_ROOT, build_graph, corpus_mtime
from .model import Edge, Graph, Node
from .obsidian_export import export_obsidian
from .query_tools import (
    QueryHit,
    QueryResult,
    query_by_decision,
    query_by_entity,
    query_by_keywords,
    query_by_time,
    query_by_topic,
    query_neighbors,
    query_supersession_chain,
)
from .recall import Reconstruction, format_human, reconstruct

__all__ = [
    "Edge",
    "Graph",
    "Node",
    "QueryHit",
    "QueryResult",
    "Reconstruction",
    "build_graph",
    "cache_paths",
    "export_obsidian",
    "format_human",
    "load_or_build",
    "query_by_decision",
    "query_by_entity",
    "query_by_keywords",
    "query_by_time",
    "query_by_topic",
    "query_neighbors",
    "query_supersession_chain",
    "recall",
    "reconstruct",
    "save_graph",
]

_CACHE_DIR = REPO_ROOT / ".omx" / "state" / "graph_memory"


def cache_paths() -> tuple[Path, Path]:
    return _CACHE_DIR / "nodes.jsonl", _CACHE_DIR / "edges.jsonl"


def save_graph(graph: Graph) -> tuple[Path, Path]:
    nodes_path, edges_path = cache_paths()
    graph.save(nodes_path, edges_path)
    return nodes_path, edges_path


def _cache_is_stale(nodes_path: Path) -> bool:
    """True if any corpus source is newer than the cached graph (auto-rebuild)."""
    try:
        cache_mtime = nodes_path.stat().st_mtime
    except OSError:
        return True
    return corpus_mtime() > cache_mtime


def load_or_build(*, rebuild: bool = False) -> Graph:
    """Load the cached graph; auto-(re)build if missing, stale, or `rebuild`.

    The cache is stale the moment the corpus (memory files, DAG, ledgers) is
    newer than the saved graph — increment-2 auto-invalidation, so recall never
    serves a graph that has fallen behind the markdown source of truth.
    """
    nodes_path, edges_path = cache_paths()
    if (
        not rebuild
        and nodes_path.is_file()
        and edges_path.is_file()
        and not _cache_is_stale(nodes_path)
    ):
        try:
            return Graph.load(nodes_path, edges_path)
        except (OSError, ValueError):
            pass  # fall through to rebuild
    graph = build_graph()
    try:
        graph.save(nodes_path, edges_path)
    except OSError:
        pass  # cache is best-effort; the in-memory graph is still usable
    return graph


def recall(query: str, *, rebuild: bool = False, **kwargs) -> Reconstruction:
    """Reconstruct-on-demand: load/build the graph, then traverse it for `query`."""
    graph = load_or_build(rebuild=rebuild)
    return reconstruct(graph, query, **kwargs)
