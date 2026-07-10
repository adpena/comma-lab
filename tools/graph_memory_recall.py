#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""graph_memory_recall — reconstruct-on-demand over the corpus graph (task #411).

The DAG was meant to be a RECONSTRUCTABLE GRAPH MEMORY (MRAgent "reconstruct, not
retrieve"), not the append-only markdown FEED log that caused the goldfish-memory
problem (memory L83: "apparatus WRITES better than it READS"). This is the recall
entry point: it builds a node/edge graph over the existing corpus (memory files'
[[wikilinks]] + DAG FEED refs + equation producers/consumers + tasks + deferrals)
and ASSEMBLES context for a query by traversing the connected subgraph — NOT flat
chunk retrieval (that is tools/corpus_query.py, the complementary retrieval layer).

Usage:
  .venv/bin/python tools/graph_memory_recall.py "what do we know about lane d_seg"
  .venv/bin/python tools/graph_memory_recall.py "2026-07-10 naive-launch incident" --json
  .venv/bin/python tools/graph_memory_recall.py --rebuild "muon warm start"
  .venv/bin/python tools/graph_memory_recall.py --stats   # graph node/edge counts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# make `tac` importable when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tac.graph_memory import (  # noqa: E402
    cache_paths,
    format_human,
    load_or_build,
    reconstruct,
    save_graph,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reconstruct-on-demand over the corpus graph memory.")
    ap.add_argument("query", nargs="*", help="free-text query / entity / #ref / date")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the graph cache from source")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--stats", action="store_true", help="print graph node/edge counts and exit")
    ap.add_argument("--max-seeds", type=int, default=4)
    ap.add_argument("--max-nodes", type=int, default=18)
    ap.add_argument("--max-depth", type=int, default=2)
    args = ap.parse_args(argv)

    graph = load_or_build(rebuild=args.rebuild)
    if args.rebuild:
        save_graph(graph)

    if args.stats or not args.query:
        stats = {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "by_node_type": graph.counts_by_type(),
            "by_edge_type": graph.edge_counts_by_type(),
            "cache": [str(p) for p in cache_paths()],
        }
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"graph memory: {stats['nodes']} nodes, {stats['edges']} edges")
            print(f"  node types: {stats['by_node_type']}")
            print(f"  edge types: {stats['by_edge_type']}")
            print(f"  cache: {cache_paths()[0]}")
            if not args.query:
                print("\n(pass a query to reconstruct context, e.g. \"lane d_seg\")")
        return 0

    query = " ".join(args.query)
    recon = reconstruct(
        graph, query,
        max_seeds=args.max_seeds, max_nodes=args.max_nodes, max_depth=args.max_depth,
    )
    if args.json:
        print(json.dumps(recon.to_dict(graph), indent=2, ensure_ascii=False))
    else:
        print(format_human(recon, graph))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
