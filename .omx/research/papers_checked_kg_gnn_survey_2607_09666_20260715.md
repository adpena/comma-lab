# PAPER CHECKED — arXiv 2607.09666 "Knowledge Graphs Meet Graph Neural Networks: A Comprehensive Survey" (Sun, Tian, … Philip S. Yu — ACM Computing Surveys)

**Assessed 2026-07-15** (operator-supplied). Recall-first: NOT previously in corpus. MEANS-layer, pointer UNMOVED 0.19108 / 0.18804. Paper-list repo: github.com/sunxiaobei/awesome-gnn-based-knowledge-graphs (curated list, no method code).

## What it says
SURVEY (no novel method, no results). Two-level taxonomy of GNN-based KG tech: KG construction → embedding → reasoning → applications; architectures GCN/GAT/HGNN. No INR / level-set / SDF / wavelet / entropy-coding / neural-compression / ego-motion / pose / segmentation / argmax / quantization content.

## Honest-fork vs OUR stack
**DIRECT = NOT-APPLICABLE.** We have no knowledge graph and no GNN; the contest is a single-clip witness INR compression problem. Nothing in the survey touches d_seg / d_pose / rate / the witness / the frozen scorer.

**Faint apparatus-tangential hook (note-only, NO build):** our `[[#411]]` DAG-as-reconstructable-graph-memory (9,273 nodes / 31,191 edges) IS a knowledge-graph-like structure, and GNN-based reasoning over a corpus graph is this survey's subject. But at ~9K nodes our traversal + keyword recall (`tools/graph_memory_recall.py`) is adequate and a GNN embedding layer over the corpus graph would be over-engineering with no score relevance (it's apparatus, and the recall path already works). If the graph-memory ever grows to a scale where traversal recall degrades, this survey is the reference for GNN-over-KG reasoning — a future-conditional apparatus note, not a lever.

## Verdict
NOT-APPLICABLE (KG+GNN survey; no compression/INR/witness/pose/seg content, no method, no results). Faint hook = GNN reasoning over our #411 graph-memory is over-engineering at current scale (traversal suffices); note-only, future-conditional apparatus reference, NO build. NOT a pointer-mover, NOT a lever. Sisters: `[[#411]]` (graph memory), `[[paper_warm_start_from_assumption_divergence_not_route_or_dismiss_20260714]]`. MEANS.
