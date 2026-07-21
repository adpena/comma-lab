---
title: "Graph-memory link hygiene sweep"
date_utc: "2026-07-21T02:17:04Z"
task: 594
lane_id: link_hygiene_sweep_20260721T015324Z
research_only: true
verdict: LANDED_MEASURED_WITH_SOURCE_INDEX_LIMIT
verdict_scope: "This worktree snapshot's graph-memory indexing and advisory tooling; no score, frontier, or MAIN-landing authority."
receipt: ".omx/research/link_hygiene_sweep_20260721T021704Z.json"
code_commit: "0f607748c108a9ccb598b06b5787010fcbcebdf5"
---

# Graph-memory link hygiene sweep

## Verdict

**LANDED / MEASURED, with a source-index-custody limit.** Task #594 now synthesizes typed
index, alias, research-memo, equation, task, FEED, lane, and Catalog edges without rewriting
the source corpus. The claim that index synthesis alone would collapse *most* orphans is
**FALSIFIED on this worktree snapshot**: the eligible root/cluster/full indexes contain only
303 valid local markdown-note links, so inventing more `indexed_by` edges would create false
authority.

## STORES CONSULTED

- Delegated authority file, SHA-256
  `3b7af2e276315d5c91e3152c82126ce64f75d9ade21a87c9b23969e0ddce5f46`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, operating manual, top-10 Claude memory entries,
  latest sister-agent/design memos, and the current canonical pointer surfaces named by the
  operating contract.
- All 8,007 markdown files in the comparable audit corpus: memory `*.md` plus top-level
  `.omx/research/*.md`.
- Canonical equation registry, task-status ledger, lane registry, latest DAG, docs tree, and
  graph-memory cache/Obsidian export.
- Delegation inboxes; no task-specific stop or superseding directive was present.

## Before/after decomposition

Two before snapshots exist and must not be conflated. The operator baseline is the authority
prompt's later audit. This isolated branch contains 60 fewer raw wikilinks and therefore cannot
honestly support direct subtraction from that baseline. A same-worktree pre-edit scan is included
for causal comparison.

| Metric | Operator before (authority) | Worktree before (same snapshot) | Worktree after |
|---|---:|---:|---:|
| Files scanned | 8,007 | 8,007 | **8,009** (includes this memo + FEED) |
| Raw wikilinks | 2,200 | 2,140 | **2,140** |
| Resolved | 1,308 (59.5%) | 1,150 (53.738%) | **1,531 / 2,138 semantic (71.609%)** |
| Alias-resolved | not decomposed | 0 | **381** |
| FP-filtered | numeric tuple class observed x17 | 0 | **2 occurrences / 1 distinct target** |
| Truly unwritten | 892 dangling / 393 distinct | 990 / 440 distinct | **607 / 290 distinct** |
| Memory notes unreachable | 1,827 / 1,976 (92.5%) | 1,707 / 1,976 (86.387%) | **1,526 / 1,976 (77.227%)** |

Same-snapshot changes are **DERIVED**: resolution +17.871 percentage points; unreachable notes
-181 (-9.160 points); distinct unwritten targets -150. The after numerator decomposes exactly as
1,150 direct + 381 alias = 1,531 resolved; the semantic denominator is 2,140 raw - 2 filtered =
2,138. The remaining orphan verdict is narrowly scoped to discoverability by a real incoming
corpus edge or a real index edge; an unreferenced filename-alias stub is deliberately not counted
as reachability.

## Comprehensive family sweep

All requested cheap families were synthesized; none was silently marked N/A.

| Family | Typed edge | MEASURED edges | Decision |
|---|---|---:|---|
| Root/cluster/full indexes | `indexed_by` | 303 | synthesize only verified local `.md` targets |
| Filename + doctrine anchors | `aliases` | 9,991 | synthesize deterministic aliases; no fuzzy guessing |
| Research memo markdown links | `memo_link` | 8 | synthesize verified local memo targets |
| Canonical equation IDs | `equation_ref` | 2,521 | synthesize only IDs present in canonical registry |
| Task references | `task_ref` | 54,010 | synthesize `#NNN`; alias canonical task rows when present |
| DAG FEED references | `feed_ref` | 2,199 | synthesize `FEED-*` mentions |
| Lane IDs | `lane_ref` | 10,103 | synthesize `lane_*`; hydrate canonical lane rows |
| Catalog references | `catalog_ref` | 32,812 | synthesize `Catalog #NNN`; hydrate doctrine rows |

The final report-inclusive rebuild materialized 2,341 section nodes, four index nodes, 6,289
research nodes, 368 Catalog nodes, and 3,874 lane nodes: 33,050 nodes / 144,732 edges total. These
are index/cache records, not new corpus claims.

## Landed apparatus and proof

- `build.py` now filters pure numbers, numeric comma tuples, and named coordinate tuples; parses
  index links; materializes doctrine sections/anchors and research memos; and synthesizes all
  requested typed entity families.
- `tools/suggest_sister_links.py <note.md>` fuses both RecallEvidence #569 surfaces, prints only,
  excludes self/already-linked/navigation-index candidates, and verifies input bytes did not
  change. A live smoke returned semantic candidates from corpus+graph and left the note unchanged.
- `tools/audit_graph_link_hygiene.py --rebuild --output ...` produces the machine-readable
  decomposition receipt and triggers the cache plus Obsidian synthesized-edge refresh.
- Final report-inclusive cache receipts: nodes 14,906,061 bytes / SHA-256
  `0bb40b45d842b44c87330d208a72f18e54132b52da0cd1e114f4f9046a7046bb`; edges 44,809,839 bytes /
  `d4018640da51f248e006b94d692beccd264d241b08fbfc51d808c9d5638b9e04`; Obsidian export
  7,653,897 bytes / `56a24e5a51ab6e450954c2c7170b4161a1bb9507466041b605c6149c0db1dcb8`.
- Final regression command covered the existing graph-memory/manifest/lensed-recall/RecallEvidence
  suites plus the new hygiene suite: **98 passed in 59.03s**. Ruff and `py_compile` passed. Two
  named review-tracker passes were recorded for every touched Python file. The new 11-test suite
  includes one test for each excluded false-positive class and a guard proving that an unreferenced
  alias cannot fake orphan closure.
- Code commit: `0f607748c108a9ccb598b06b5787010fcbcebdf5`, serialized with post-edit SHA checks and no
  co-author trailer.
- The new lane is internally consistent at L1 with `impl_complete` evidence. A whole-registry
  `lane_maturity.py validate` remains blocked by 110 pre-existing evidence paths absent from this
  worktree; this sweep did not mutate or claim those unrelated lanes.

## Triality and pointer delta

- DAG: this memo has a standalone FEED companion.
- DSL: N/A with reason — this is retrieval/indexing apparatus and changes no training control.
- Equations: N/A with reason — canonical law IDs are indexed but no law or equation is changed.
- Score/frontier pointer: **UNCHANGED**. No contest score or promotion claim is made.
- MAIN landing: **REQUIRED REVIEW** of the branch diff, especially alias precision, honest orphan
  reachability semantics, cache-size/performance cost, and the worktree/operator snapshot mismatch.
