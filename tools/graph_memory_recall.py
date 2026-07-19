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

  # opt-in lensed recall (task #346): the SAME base reconstruction above, plus
  # tac.lens_engine GRAPH (bridge/hub) + TOPOLOGY (saddle/crux) augmentation.
  .venv/bin/python tools/graph_memory_recall.py --lens "lane d_seg"
  .venv/bin/python tools/graph_memory_recall.py --lens --lens-centrality closeness "muon warm start"
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
    build_source_manifest,
    cache_paths,
    check_completeness,
    export_obsidian,
    format_human,
    format_human_lensed,
    load_latest_manifest,
    load_or_build,
    query_by_decision,
    query_by_entity,
    query_by_keywords,
    query_by_time,
    query_by_topic,
    query_neighbors,
    query_supersession_chain,
    reconstruct,
    reconstruct_lensed,
    save_graph,
)

# The 7 typed MRAgent query-tools, keyed by their CLI name (increment-2).
_TOOLS = {
    "time": lambda g, q: query_by_time(g, date=q) if "-" in q and ".." not in q
    else query_by_time(g, since=q.split("..")[0], until=q.split("..")[-1]),
    "keywords": lambda g, q: query_by_keywords(g, q),
    "entity": lambda g, q: query_by_entity(g, q),
    "topic": lambda g, q: query_by_topic(g, q),
    "decision": lambda g, q: query_by_decision(g, verdict=q),
    "neighbors": lambda g, q: query_neighbors(g, q),
    "supersedes": lambda g, q: query_supersession_chain(g, q),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reconstruct-on-demand over the corpus graph memory.")
    ap.add_argument("query", nargs="*", help="free-text query / entity / #ref / date")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the graph cache from source")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--stats", action="store_true", help="print graph node/edge counts and exit")
    ap.add_argument("--manifest", action="store_true",
                    help="print the source/completeness manifest (per-source node "
                         "counts + warn-only diff vs the last published manifest) and exit")
    ap.add_argument("--tool", choices=sorted(_TOOLS), default=None,
                    help="run a typed query-tool over the positional query (increment-2)")
    ap.add_argument("--export-obsidian", nargs="?", const="", default=None, metavar="PATH",
                    help="write the synthesized-edge Obsidian index (default cache path) and exit")
    ap.add_argument("--max-seeds", type=int, default=4)
    ap.add_argument("--max-nodes", type=int, default=18)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument(
        "--lens", action="store_true",
        help="opt-in: augment the reconstruction with tac.lens_engine GRAPH "
             "(bridge/hub) + TOPOLOGY (saddle/crux) findings on top of the "
             "unchanged base reconstruction (task #346 retrieval-first wire-in)",
    )
    ap.add_argument(
        "--lens-phi", default="degree",
        help="tac.lens_engine.CorpusAdapter Phi mode for --lens (default: "
             "'degree', the only custody-safe default; see CorpusAdapter for "
             "other modes and their custody requirements)",
    )
    ap.add_argument(
        "--lens-centrality", default="betweenness", choices=("betweenness", "closeness", "degree"),
        help="GRAPH lens centrality method used to rank hub nodes for --lens",
    )
    args = ap.parse_args(argv)

    graph = load_or_build(rebuild=args.rebuild)
    if args.rebuild:
        save_graph(graph)

    if args.manifest:
        manifest = build_source_manifest(graph)
        prev = load_latest_manifest()
        findings = check_completeness(prev, manifest)
        if args.json:
            print(json.dumps({
                "manifest": manifest.to_dict(),
                "completeness_warnings": [w.to_dict() for w in findings],
            }, indent=2, ensure_ascii=False))
        else:
            print(f"source manifest: {manifest.node_count} nodes, "
                  f"{manifest.edge_count} edges  (git {manifest.git_rev[:9] or '?'})")
            for s in manifest.sources:
                flag = "REQUIRED" if s.required else "optional"
                warn = "  <-- COLLAPSED" if (s.required and s.exists and s.nodes_emitted == 0) else ""
                print(f"  [{flag:>8}] {s.name:<14} "
                      f"units={s.units_present:<6} nodes={s.nodes_emitted:<6} "
                      f"exists={s.exists} {s.skip_reason}{warn}")
            if findings:
                print("\nCOMPLETENESS WARNINGS (warn-only, vs last published manifest):")
                for w in findings:
                    src = f" [{w.source}]" if w.source else ""
                    print(f"  ! {w.kind}{src}: {w.detail}")
            else:
                print("\ncompleteness: OK (no drop vs last published manifest)")
        return 0

    if args.export_obsidian is not None:
        out = Path(args.export_obsidian) if args.export_obsidian else None
        path = export_obsidian(graph, out)
        print(json.dumps({"exported": str(path)}) if args.json else f"exported: {path}")
        return 0

    if args.tool is not None:
        q = " ".join(args.query)
        result = _TOOLS[args.tool](graph, q)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"[{result.kind}] {result.query!r} -> {len(result.hits)} hits")
            for h in result.hits:
                print(f"  ── [{h.ntype}] {h.title[:78]}  ({h.why})")
        return 0

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
    if args.lens:
        lensed = reconstruct_lensed(
            graph, query,
            max_seeds=args.max_seeds, max_nodes=args.max_nodes, max_depth=args.max_depth,
            phi=args.lens_phi, centrality_method=args.lens_centrality,
        )
        if args.json:
            print(json.dumps(lensed.to_dict(graph), indent=2, ensure_ascii=False))
        else:
            print(format_human_lensed(lensed, graph))
        return 0

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
