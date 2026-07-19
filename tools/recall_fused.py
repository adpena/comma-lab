#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""recall_fused — ONE ranked recall surface across lexical + graph memory (#569 P0-2).

The two durable recall surfaces (``tools/corpus_query.py`` lexical density and
``tools/graph_memory_recall.py`` reconstruct-not-retrieve) returned separate,
incomparable result lists. This tool runs both and reciprocal-rank fuses them
into one typed ``RecallEvidence`` ranking, so a consumer ranks ACROSS surfaces
instead of reading two lists.

Usage:
  .venv/bin/python tools/recall_fused.py "dash comb registration"
  .venv/bin/python tools/recall_fused.py "muon warm start" --top 10 --json
  .venv/bin/python tools/recall_fused.py "pose carrier" --stores dag,equations
  .venv/bin/python tools/recall_fused.py "lane d_seg" --surface corpus   # single surface
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import corpus_query as cq  # noqa: E402
from tac.graph_memory import load_or_build, reconstruct  # noqa: E402
from tac.recall_evidence import DEFAULT_RRF_K, fuse_recall  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", help="free-text query")
    ap.add_argument("--top", type=int, default=15, help="max fused hits (default 15)")
    ap.add_argument("--stores", default=None,
                    help=f"corpus store filter from: {','.join(cq.STORE_NAMES)}")
    ap.add_argument("--surface", choices=("both", "corpus", "graph"), default="both",
                    help="which retrieval surfaces to fuse (default both)")
    ap.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K,
                    help=f"reciprocal-rank-fusion smoothing k (default {DEFAULT_RRF_K}; candidate not sacred)")
    ap.add_argument("--corpus-max-seconds", type=float, default=None,
                    help="optional wall-clock budget for the lexical scan")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    stores = None
    if args.stores:
        stores = [s.strip() for s in args.stores.split(",") if s.strip()]
        unknown = [s for s in stores if s not in cq.STORE_NAMES]
        if unknown:
            print(f"unknown store(s): {', '.join(unknown)} — valid: {', '.join(cq.STORE_NAMES)}",
                  file=sys.stderr)
            return 2

    corpus_result = None
    if args.surface in ("both", "corpus"):
        corpus_result = cq.run_query(
            args.query, stores=stores, top=max(args.top, 25),
            max_seconds=args.corpus_max_seconds,
        )

    reconstruction = None
    graph = None
    if args.surface in ("both", "graph"):
        graph = load_or_build()
        reconstruction = reconstruct(graph, args.query, max_nodes=max(args.top, 25))

    fused = fuse_recall(
        corpus_result=corpus_result,
        reconstruction=reconstruction,
        graph=graph,
        k=args.rrf_k,
    )
    top = fused[: args.top]

    if args.json:
        print(json.dumps({
            "query": args.query,
            "rrf_k": args.rrf_k,
            "surfaces": args.surface,
            "hits": [e.to_dict() for e in top],
        }, indent=2, ensure_ascii=False))
        return 0

    if not top:
        print(f"(no fused hits for {args.query!r})")
        return 0
    print(f"FUSED RECALL for {args.query!r}  (rrf_k={args.rrf_k}, surfaces={args.surface})")
    for i, e in enumerate(top, 1):
        surfaces = "+".join(e.contributing_surfaces)
        print(f"{i:2d}. [{surfaces}] rrf={e.rrf_score:.5f}  "
              f"[{e.source_surface}:{e.store}] {e.ref}")
        if e.hook_line:
            print(f"      | {e.hook_line[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
