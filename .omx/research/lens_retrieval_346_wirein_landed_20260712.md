# Lens Engine wired into #346 retrieval-first corpus recall — `reconstruct_lensed`

**Date:** 2026-07-12
**Lane:** `lane_lens_retrieval_346_wirein_20260712`
**Status:** LANDED / ANALYSIS-RETRIEVAL APPARATUS / MEANS
**Authority:** no training actuation, no archive candidate, no evaluator replay, no score claim
**Pointer delta:** `0.18804` UNMOVED

## Scope

This is the single named consumer directed by the operator to make the just-landed **Lens Engine**
increment 1 (`tac.lens_engine`, commit `eb3a4fa494`, memo
`.omx/research/lens_engine_inc1_landed_20260712.md`) pay rent on the P0 anti-forgetting goal: wiring it
into the **#346/#411 retrieval-first corpus recall path**. Scope is corpus retrieval ONLY — the
dashboard/SENSE consumer, the unified query language, and cross-lens composition are inc-1's explicitly
owed inc-3 items and were **not** started here.

## What today's naive recall could not see

`tac.graph_memory.reconstruct(graph, query, max_seeds, max_nodes, max_depth)` seeds by keyword match then
traverses a depth-limited (`max_depth=2` default) BFS, accumulating a weight = `seed_score * decay^dist`
per seed and keeping the top `max_nodes` by that weight. Structurally this can never see: (a) a node that
sits on the actual connecting path between two seeds but that BFS-weight ranking pushed just past the
`max_nodes` cutoff; (b) a node beyond `max_depth` from every seed entirely, even if it is the one thing
that ties two findings together; (c) which of the reached candidates is a genuine network hub/bridge by
graph centrality rather than by raw keyword-hit count; (d) a graph-topological crux (a discrete saddle in
the corpus connectivity field) near the query.

## What was built

### `src/tac/graph_memory/recall.py`

- **Refactor (behavior-preserving).** Extracted `_select_seeds(graph, query, max_seeds)` and
  `_bfs_reconstruct(graph, seeds, max_depth)` from `reconstruct()`'s body. `reconstruct()` itself now
  calls these two helpers and does the identical ranking/truncation it always did — same seed scorer,
  same BFS, same tie-breaks, same output. Pinned by
  `test_reconstruct_unchanged_after_lensed_refactor` (a hand-derived synthetic fixture whose exact
  seeds/nodes/paths are computed by hand and asserted) plus a real-corpus smoke that reruns `build_graph()`.
- **`reconstruct_lensed(graph, query, *, max_seeds, max_nodes, max_depth, phi="degree",
  centrality_method="betweenness", max_bridge_nodes=8, max_hub_nodes=4, max_crux_nodes=3,
  centrality_pool_cap=800) -> LensedReconstruction`.** Computes the IDENTICAL base reconstruction (calls
  the same two shared helpers) and then, strictly additively:
  1. **BRIDGE** — `tac.lens_engine`'s GRAPH lens `shortest_path` between every pair of the base's seeds,
     run on a `CorpusAdapter` over the FULL corpus graph (not depth-limited). Any node on a discovered
     shortest path that is not already in the base reconstruction is added, recording which seed pair it
     bridges.
  2. **HUB** — GRAPH lens `centrality` (betweenness by default; closeness/degree selectable) over a
     **query-local bounded pool**: the union of every node the base BFS actually reached (the untruncated
     weight-dict keys) plus whatever the bridge step already added. This is NOT the whole 8,842-node
     corpus graph (see below for why) — new nodes ranked highest by centrality and not already included
     are added, in rank order.
  3. **CRUX** — TOPOLOGY lens `saddles` over the same local pool (same `CorpusAdapter`, same `Phi`),
     filtered to saddles adjacent to (or equal to) an already-included/seed node — i.e. topological cruxes
     "near" the query, not the whole graph's saddle set.
  `nodes` is `base.nodes` plus these additions, in that priority order — **base.nodes is always an
  unmodified prefix**; nothing the naive path found is ever reordered or dropped.
- **`_local_pool_graph(graph, pool_ids, weight, pool_cap)`.** Builds the bounded induced subgraph GRAPH/
  TOPOLOGY run over. **Full-corpus closeness centrality was MEASURED at ~91.4s over the live 8,842-node /
  30,616-edge corpus cache (2026-07-12)** — far too slow for an interactive recall call — while betweenness
  over a real query's ~950-node local BFS pool measured ~1.8s. The pool is therefore always the query's own
  BFS-reached candidates (already query-relevant by construction), never the whole graph; if it still
  exceeds `pool_cap` (default 800) it is deterministically truncated to the highest-BFS-weight members,
  never dropping an id the caller marked `required` (already-included base/bridge nodes).
- **`format_human_lensed`** renders the unchanged `format_human` base block plus a
  `LENS ENGINE augmentation (phi=..., centrality=..., local_pool=...)` section listing each addition with
  its reason (`BRIDGE a<->b`, `HUB betweenness=...`, `CRUX saddle phi=...`).
- **`LensedReconstruction.to_dict`** carries `lensed: true`, `phi_mode`, `centrality_method`,
  `local_pool_size`, `local_pool_capped`, and per-category `bridge_nodes`/`hub_nodes`/`crux_nodes` lists
  with `why` provenance strings, plus `nodes_lensed` (the final superset list).

### `tools/graph_memory_recall.py`

Added `--lens` (opt-in; naive path is the unchanged default), `--lens-phi` (default `degree`),
`--lens-centrality` (`betweenness`/`closeness`/`degree`). `--lens --json` emits the `to_dict()` schema
above; `--lens` (human mode) prints `format_human_lensed`.

## Reuse map (no reimplementation)

| Surface | Reused from `tac.lens_engine` | Not reimplemented |
|---|---|---|
| Seed scoring / BFS | `tac.graph_memory.recall._select_seeds` / `_bfs_reconstruct` (own module, factored not duplicated) | n/a |
| Corpus → typed complex | `tac.lens_engine.CorpusAdapter` (constructed directly over an in-memory `Graph`, not `.from_cache()`, so the caller's already-loaded/rebuilt graph is used verbatim) | n/a |
| Shortest path | `tac.lens_engine.GRAPH` (`query(..., "shortest_path", directed=False)`) | Dijkstra was NOT reimplemented in `graph_memory` |
| Centrality | `tac.lens_engine.GRAPH` (`query(..., "centrality", method=...)`) | betweenness/closeness/degree were NOT reimplemented |
| Saddle/crux | `tac.lens_engine.TOPOLOGY` (`query(..., "saddles")`) | the discrete graph-filtration saddle definition (upper/lower link component counts) lives ONLY in `tac.lens_engine.topology` |

`src/tac/lens_engine/*` was **not edited** (inc-1 is sealed; imported, not modified). The one non-obvious
engineering constraint: `tac.lens_engine.adapters` imports FROM `tac.graph_memory` (`Graph`, `Node`,
`cache_paths`), so a **module-level** `from tac.lens_engine import ...` inside `tac/graph_memory/recall.py`
would circular-import the first time anything imports `tac.graph_memory` before `tac.lens_engine` (`cache_paths`
is defined in `tac/graph_memory/__init__.py` AFTER the `from .recall import ...` line, so a partially
initialized module would be missing it). **Confirmed by direct reproduction** with a throwaway two-package
harness before writing the fix: the import is done LAZILY inside `reconstruct_lensed()`'s body — the same
pattern `tac.lens_engine.adapters` itself already uses for its own optional `tac.boundary_math` imports.

## MEASURED before/after — the pay-rent proof (real corpus, real queries, 2026-07-12)

All three canonical probe queries, `max_seeds=4, max_nodes=18, max_depth=2` (the CLI defaults), superset
property verified programmatically (`base_ids.issubset(lensed_ids)` — never fails; also pinned by
`test_real_corpus_lensed_is_superset_of_naive`, parametrized over all 3 queries):

### "lane d_seg" — base 18 → lensed 26 (local pool 504, uncapped)

| New node | Type | Why |
|---|---|---|
| `feed:phase-advect-build` | finding | **BRIDGE** `feed:auditA <-> feed:dc` — ranked #20 by naive keyword-BFS weight (one past the `max_nodes=18` cutoff) yet sits on the ACTUAL shortest path connecting 3 of the 4 seeds through `person:Yousfi` |
| `ref:#205` | entity | HUB betweenness=0.1435 |
| `ref:#346` | entity | HUB betweenness=0.0572 — **this very task, surfaced as a structural hub of its own query** |
| `eq:dash_erasure_homogenization_v1` | equation | HUB betweenness=0.0357 |
| `feed:relsig` | finding | HUB betweenness=0.0284 |
| `memory:memos-must-be-acted-upon-...` | memory | CRUX saddle phi=20.00 |
| `memory:slot-l-slot-h-top-3-super-additive-...` | memory | CRUX saddle phi=8.00 |
| `file:tools/costate_digest.py` | entity | CRUX saddle phi=8.00 |

### "muon warm start" — base 18 → lensed 25 (local pool 948→capped to 800)

| New node | Type | Why |
|---|---|---|
| `memory:deepmath-amortizing-argmax-maslov-caustic-tau-eps-hbar` | memory | HUB betweenness=0.0307 |
| `memory:pose-solved-screw-twist-dual-use-film-conditioned-sidecar` | memory | HUB betweenness=0.0267 |
| `ref:#157` | entity | HUB betweenness=0.0244 |
| `memory:msal-uni-texture-proxy-inert-build-exact-sR-reachability-weight` | memory | HUB betweenness=0.0240 |
| `ref:#229` | entity | CRUX saddle phi=70.00 |
| `ref:#206` | entity | CRUX saddle phi=68.00 |
| `ref:#220` | entity | CRUX saddle phi=60.00 |

Bridge step found no rescue for this query: only 1 of the 6 seed-pair shortest paths was reachable
(`feed:ch` is disconnected in the corpus graph from the muon memory nodes), and its one intermediate node
(`file:experiments/train_witness_realized_through_R_mlx.py`) was already in the naive top-18.

### "naive-launch incident" — base 18 → lensed 25 (local pool 815→capped to 800)

| New node | Type | Why |
|---|---|---|
| `memory:assumptions-classification-hard-earned-vs-cargo-culted-critical-addendum-20260515` | memory | HUB betweenness=0.0154 |
| `memory:consolidate-everything-into-meta-layer-or-canonical-helpers-20260515` | memory | HUB betweenness=0.0151 |
| `memory:z3-g1-variant-modal-dispatch-paired-landed-20260515` | memory | HUB betweenness=0.0146 |
| `memory:feedback_recursive_review_r1_LANDED_20260513` | memory | HUB betweenness=0.0143 |
| `memory:memos-must-be-acted-upon-...` | memory | CRUX saddle phi=25.00 |
| `memory:save-memories-not-apologies-anti-forgetfulness` | memory | CRUX saddle phi=24.00 |
| `memory:canonical-ev-metric-trichotomy-...` | memory | CRUX saddle phi=17.00 |

Only 2 seeds here (1 pair), and that pair's shortest path's one intermediate node
(`file:tools/witness_launch_readiness_gate.py`) was already in the naive top-18 — bridge again finds
nothing new, honestly reported as empty rather than fabricated.

### Honest reading of the result

On all 3 probes the lensed path IS a genuine superset with real, verifiable additions. The bridge
mechanism only paid off on 1 of 3 (a real, structurally interesting rescue — a node one rank below the
cutoff, on the literal connecting path through a named person-node). The hub/crux mechanisms paid off on
all 3, but with a recurring honest pattern: with the safe default `Phi="degree"`, the crux/hub findings
skew toward **generic well-cited standing-directive memories and catalog `ref:#NNN` numbers** (e.g.
`memory:memos-must-be-acted-upon-...` appears as a crux in 2 of 3 queries) rather than query-specific
tension points — because degree-based "betweenness"/"saddle" naturally surfaces whatever is broadly
cited within the query-local pool, and several such standing-directive memories are cited very broadly.
This is a real, disclosed property of the safe default, not a failure to hide: richer `Phi` custody
(`citation_salience`/`recency`, per the inc-1 memo's fail-closed contract) would very likely surface more
query-specific tension points, but the graph-memory cache does not yet custody those fields for every
node — that gap is source-level, and `reconstruct_lensed` correctly refuses to fabricate a fallback for
it (see the `phi` custody test below).

## Correctness findings caught in review (both fixed before landing)

1. **Off-by-one in `max_hub_nodes=0`.** The original loop appended a candidate BEFORE checking the cap
   (`hub_nodes.append(...); if len(hub_nodes) >= max_hub_nodes: break`), so `max_hub_nodes=0` still let
   exactly one node through. Fixed to check the cap first. Caught by
   `test_lensed_max_hub_nodes_zero_adds_nothing` (a positive regression test for the exact bug), reproduced
   and confirmed via a hand-derived synthetic "articulation point" fixture before and after the fix.
2. **Misleading `local_pool_size` metadata.** The returned `local_pool_size` was `len(pool_ids)` — the
   PRE-cap set size — even when `local_pool_capped=True`, so a capped result could report a larger number
   than the pool that was actually queried (a silent-misleading-claim bug, the exact class the operating
   manual's §4/§8 warns about: "does my fix repair the CLASS or just the instance?"). Fixed to read
   `len(local_graph.nodes)` — the actually-built local graph — after capping. Caught by
   `test_real_corpus_lensed_bounded_local_pool_for_cost` on the real corpus (a broad query genuinely
   exceeded the 800 cap and the pre-fix code reported 1,627, not ≤800).

## Tests

`src/tac/tests/test_graph_memory_lensed_recall.py` — 24 test items (22 test functions, one
parametrized ×3 over the canonical probe queries): the `reconstruct()`-unchanged
regression pin (hand-derived fixture), a bridge-rescue positive test (hand-verified reachability at
`max_depth=1`: two nodes exist in NEITHER seed's BFS tree at all, only the full-graph shortest path finds
them), a hub-ranking positive test (hand-verified betweenness on a graph with one true articulation
point vs. two plain leaves, values pinned to 0.7/0.0), a saddle positive test (hand-verified degree-based
saddle: one node with two mutually-disconnected higher-local-degree fans plus two lower-degree
neighbors), the two regression-guarded bugs above, phi-custody fail-closed (`AdapterError` propagates,
never silently degrades to 0), local-pool-cap never-drops-required-ids, determinism, graceful no-match,
`to_dict`/`format_human_lensed` rendering, a real-corpus superset smoke parametrized over the 3 canonical
queries, a real-corpus pool-cap-engagement smoke, and an end-to-end CLI subprocess test of `--lens --json`
against the real corpus cache. Full suite (this file + the existing `test_graph_memory*` +
`test_lens_engine.py`): **133 passed**, `ruff check` clean, `ruff check --select F` clean, `mypy` clean on
both touched modules.

## Triality disposition

This landing IS the retrieval-first CONSUMER leg the inc-1 memo left owed — not `[no-triality]`.
`reconstruct_lensed`'s bridge/hub/crux findings are a new SENSE-layer input a future costate/dashboard
consumer (`#247`/`#219`, still inc-3-owed — NOT built here) can query for structurally-connecting or
crux-adjacent nodes near a live decision. The DAG FEED row
(`### FEED-lens-retrieval-346` in `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`)
is this landing's trajectory leg. No new canonical equation was registered (this is retrieval apparatus,
not a measured score-law) and no DSL lever was added (no training actuation, no config surface). Six-hook
disposition, per the subagent-landing wire-in discipline:

- **Sensitivity-map / Pareto / bit-allocator:** N/A — no byte/score effect.
- **Cathedral autopilot dispatch:** N/A in this increment; inc-3 owes the dashboard/SENSE consumer hook.
- **Continual-learning posterior:** N/A — no empirical score anchor produced.
- **Probe-disambiguator:** N/A — the 3 additions (bridge/hub/crux) are independently labeled and
  additive, not a hidden blended verdict; the honest-reading section above documents where the default
  `Phi="degree"` mode is and is not decisive.

## Explicitly still owed (inc-3 — do not infer as started)

- Dashboard/SENSE governed consumer that surfaces `reconstruct_lensed`'s bridge/crux findings to the
  costate controller or the operator-facing dashboard.
- The small unified query language / cross-lens composition (chaining GRAPH → TOPOLOGY → SPATIAL etc.
  through one typed DSL surface).
- Richer, custodied `Phi` modes (`citation_salience`/`recency`) for the graph-memory cache, which the
  honest-reading section above shows would likely sharpen crux specificity beyond the generic-hub pattern
  observed with the safe `degree` default.
- A durable registration path (probe outcome / DAG / equation) for the rare case a future lensed query
  surfaces a load-bearing finding that should itself become a measured, cited anchor.
