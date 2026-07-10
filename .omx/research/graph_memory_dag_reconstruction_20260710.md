# Graph-memory: the DAG as a RECONSTRUCTABLE GRAPH MEMORY (task #411, increment 1)

**Date:** 2026-07-10 · **Task:** #411 (FOUNDATIONAL P0) · **Pointer:** 0.19108282 [contest-CPU]
**UNMOVED** — this is APPARATUS (MEANS), not a score move. Say so plainly.

**Operator correction (verbatim):** *"Pursue it now as p0, that was my original intent for the DAG
which you turned into just a folder of markdown files."* + steer: *"You can use other tools and software
too … like obsidian … LEAN ON the [[wikilink]] markdown graph as the substrate — do NOT build a bespoke
graph DB."*

## The miss this repairs

The DAG (`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`) was ALWAYS meant to be a
**reconstructable graph memory** — the MRAgent *"Memory is Reconstructed, Not Retrieved"* thesis. It
degraded into an append-only markdown FEED log I could only grep/tail. That degradation IS the concrete
mechanism of the goldfish-memory root cause (memory `L83`: *"apparatus WRITES better than it READS"*): a
flat log can be grepped, never RECONSTRUCTED, so I re-forget what the corpus already holds and the operator
catches it. The MEMORY.md flat-load-cap struggle is the same bug at the index layer.

## STORES CONSULTED

- Durable memories `dag_was_meant_to_be_reconstructable_graph_memory_not_markdown_folder_20260710`,
  `papers_checked_mragent_reconstruct_not_retrieve_20260710`, `L83` (retrieval-first nexus root cause).
- MRAgent source (https://github.com/Ji-shuo/MRAgent) — `memory/` graph store + `agent/` 2-phase pipeline
  (construction → 7-tool reconstruction QA loop); node types {episode, topic, personal-fact}.
- The existing #346 retrieval-first nexus: `tools/corpus_query.py` (FLAT token retrieval),
  `tools/costate_digest.py` (session-start SENSE surface, SessionStart hook), `tools/convene.py`.
- Corpus formats: memory `*.md` frontmatter + `[[wikilinks]]` (1344 links across 1885 files),
  DAG `### / ## FEED-*` blocks (525 raw headers → 496 distinct slugs), `canonical_equations_registry.jsonl`
  (573 equations, producers/consumers), `canonical_task_status.jsonl`, `deferral_ledger.md`.

## The design (grounded in MRAgent, on our Obsidian-compatible substrate)

MRAgent's transferable idea is the **RETRIEVAL phase**: a graph of {episode/finding, topic, fact/entity}
+ a query loop that RECONSTRUCTS an answer by traversing the graph, rather than returning flat chunks. Our
construction phase is largely N/A (we have structured stores, not a chat log) — and the operator's steer
nails why it's cheap: **the `[[wikilink]]` markdown IS already an Obsidian vault** (308+ wikilinked files).
So the substrate is the shared wikilink markdown graph; the builder MATERIALIZES the graph over it and
SYNTHESIZES the cross-store edges (FEED→#task/eq/file, producer/consumer, sister, supersedes) — it does
**not** rewrite any FEED markdown, and it is **not** a bespoke DB (a gitignored JSONL cache/index).

`tac.graph_memory`:
- `model.py` — typed `Node` (id/ntype/title/summary/source/attrs) + `Edge` (src/dst/etype/source) +
  `Graph` (forward + BACKWARD adjacency = backlinks, the Obsidian view) with deterministic save/load
  (sorted JSONL) + traversal. **9 node types** {memory, finding, topic, entity, person, equation, task,
  deferral, decision}; **8 edge types** {links, references, supersedes, blocks, produces, consumes,
  sister, tagged}.
- `build.py` — deterministic parsers over the REAL corpus (capped reads, content-derived ids, no RNG):
  memory frontmatter + `[[wikilinks]]` + body #ref/file/eq refs; DAG FEED blocks (duplicate slugs
  disambiguated so no block is merged away); equation producers/consumers; task blockers; deferral rows.
- `recall.py` — the reconstruction loop: (1) score every node by query-token overlap on id+title+summary
  (+ exact `#ref`/date anchors) → SEEDS; (2) BFS from seeds to depth-2 over BOTH edge directions,
  accumulating weight = seed_score·decay^dist; (3) rank the connected subgraph (seeds first) and ASSEMBLE
  the context with the edge PATH that links each node to a seed. Deterministic (stable id tie-breaks).
- `__init__.py` — `build_graph` / `load_or_build` (cached, rebuild-if-stale) / `reconstruct` / `recall`.
- `tools/graph_memory_recall.py` — CLI (`--rebuild`, `--json`, `--stats`).

**Honest mechanism note:** seed selection uses the same deterministic token-scoring as `corpus_query`
(flat retrieval). The GRAPH is what's new: the traversal + backlink assembly + edge-path reconstruction
surface CONNECTED nodes that flat retrieval never links. This is faithful to MRAgent (its QA loop also
finds seeds via keyword tools, then traverses). `graph_memory` (reconstruct) and `corpus_query` (retrieve)
are complementary layers of #346.

## Built from the ACTUAL corpus (NO-FAKE)

`build_graph()` in 0.4s → **8556 nodes / 29971 edges**:
`{memory 2180, finding 410, decision 116, equation 280, task 24, deferral 29, person 19, topic 5,
entity 5493}` · edges `{references 26425, links 1150, tagged 1128, consumes 643, produces 588, sister 37}`.
Rebuild is byte-identical (deterministic). Cache: `.omx/state/graph_memory/{nodes,edges}.jsonl`
(gitignored — rebuildable index, NOT source of truth).

## Demonstrated reconstructions (proof it works, real queries)

**Q1 "what do we know about lane d_seg"** → seeds `feed:auditA` (v7.5 d_seg lever synthesis),
`eq:analytic_lane_band_dseg_recon_floor_v1`, `feed:dc`/`feed:de` (Yousfi lever build/land); traversal
assembles `person:Yousfi` (via references) and `decision:FEED-poseladder` (via a SHARED `#238` reference)
— context flat retrieval would not connect. The reconstruction prints each node's summary + source line-
range + the edge path (`… --references--> #238 <--references-- FEED-poseladder`).

**Q2 "2026-07-02 naive launch OOM incident verdict-batch"** → seeds
`memory:review-seals-borrowed-numbers-and-unrun-configs-measure-at-real-config`, `feed:oom`,
`memory:orphaned-measured-win-…`, `memory:machine-crashing-risk-is-P0-hard-gate-…`; traversal cross-links
these incident memories via the `#205` entity + surfaces `eq:oom_verdict_batch_spike_peak_rss_v1` and the
`save-memories-not-apologies` discipline via a `[[link]]` — **the incident reconstructed from the connected
subgraph WITHOUT loading MEMORY.md wholesale.** This is the read-side fix for L83.

## Wire-in (recall becomes structural, not grep-by-volition)

`tools/costate_digest.py` (the SessionStart hook + #346 SENSE surface) gains `section_graph_memory()` — a
read-only, fast (JSONL line-count, never rebuilds), score-neutral line every session start:
`graph-memory: 8556 nodes / 29971 edges — RECONSTRUCT before grepping: tools/graph_memory_recall.py
"<query>" (#411 DAG-as-graph)`. On a fresh checkout (no cache) it prints the `--rebuild` hint (fail-open).
costate_digest still runs ~1s, rc=0. This is the observability-defaults-ON principle: the recall
affordance is surfaced every session so reconstruction is the default before grepping.

## Rigor

`src/tac/tests/test_graph_memory.py` — 18 tests: deterministic fixture corpus (parsers produce exact
expected nodes/edges: frontmatter, wikilinks, sister, tag→topic, #ref, producers/consumers, verdict→
decision, duplicate-slug disambiguation, task blockers, deferrals) + reconstruction (seed-finding,
traversal-path presence, entity anchor, determinism, graceful no-match) + save/load round-trip +
save-determinism + a real-corpus smoke (structurally-valid graph, robust to count drift). `ruff --select F`
clean. Round-1 adversarial self-review done (below).

## Triality

- **DAG leg (this IS the leg being repaired):** DAG FEED `### FEED-graph-memory` records the build.
- **DSL / equations:** N/A-with-rationale — this is META recall APPARATUS (reconstruction over the corpus),
  not a witness lever or a measured score-law. No `WitnessProgram` lever, no `EmpiricalAnchor` (nothing
  measured about the contest score). If a future increment measures a recall-quality law, register it then.

## Increment-1 covers vs OWED

**Covers:** real node/edge graph over the whole corpus; deterministic build + cache; reconstruct-on-demand
recall proven on 2 real queries; session-start wire-in; tests; Obsidian-compatible substrate (the wikilink
vault is already openable in Obsidian; the builder indexes it + synthesizes cross-store edges).

**OWED (increment 2+):**
1. **Edge weighting / typing quality** — `references` edges (26425) include incidental `#NNN` mentions;
   down-weight incidental refs vs load-bearing ones (BFS is seed-anchored + depth-bounded so noise is
   contained, but weighting sharpens reconstruction).
2. **Obsidian round-trip export** — the SYNTHESIZED cross-store edges (FEED→eq, producer/consumer) are not
   yet written back as `[[links]]` into FEEDs/memos, so the operator's Obsidian graph view shows the
   memory-file wikilinks but not the synthesized edges. Export a companion vault (or a graph.json Obsidian
   plugins read) so the operator SEES the full reconstructable graph in the GUI.
3. **Auto-build the cache on first session-start** (currently prints a `--rebuild` hint; fail-open).
4. **MRAgent's 7 typed query tools** — I ship ONE `reconstruct()`; add typed tools (by_topic / by_time /
   by_entity / by_decision) so an agent (or a costate DECIDE step) can compose multi-hop reconstructions.
5. **Cache freshness/incremental rebuild** — currently full rebuild (0.4s, fine now); incremental on
   corpus change if it grows.

## Round-1 adversarial self-review

- **NO-FAKE:** graph is parsed from the real 1885 memory files + real DAG + real equation registry; recall
  demonstrated end-to-end on real queries; not a scaffold. ✓
- **Reconstruct ≠ retrieve relabeled:** honest — seeds use token-scoring, but the graph traversal +
  backlink assembly + edge-path is the genuinely new (MRAgent) part; surfaces connected context flat
  retrieval cannot. Stated plainly, not oversold. ✓
- **Constraint honored:** does NOT rewrite the DAG markdown (read-only → gitignored cache); only new files
  + `costate_digest.py` (clean, not a sibling-modified file); did not touch the git-status-modified
  siblings (`tools/memory_guard.py`, `.omx/state/*`). ✓
- **Determinism:** byte-identical rebuild verified. ✓
- **Means not ends:** pointer UNMOVED 0.19108282; this is recall apparatus. Stated. ✓
