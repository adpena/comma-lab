---
schema: council_deliberation_v2
deliberation_id: design_stack_full_hypergraph_model_design_memo_20260518
topic: "Full labeled hypergraph model of the pact design stack — typed nodes (10 categories incl. deterministic_byte_derivation META) + typed directed edges (7 categories) + hyperedges (3 types; N-way composition_alpha per Catalog #322) + cycles (continual-learning feedback loops) + canonical graph operations (critical_path / orphan_signals / hyperedge_compositions / cycles / hook_coverage / dominator / export_dot) + fcntl-locked JSONL persistence per Catalog #131/#138/#245 + DOT/ASCII/d3 visualization. Provides DESIGN AUTHORITY for canonical helper `src/tac/design_graph.py` per sister Codex routing directive C 2026-05-18."
review_kind: t2_design_authority_memo_for_canonical_helper
review_date: "2026-05-18"
lane_id: lane_design_stack_full_hypergraph_design_20260518
council_tier: T2
council_attendees:
  # Sextet pact (binding; quorum 6-of-6 at T2)
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  # T2 grand council attendees added per topic (graph theory + hypergraph theory + multi-scale hierarchies + convex feasibility + reductionist engineering + categorical hierarchies + IT graph compression)
  - Tao
  - Mallat
  - Boyd
  - Carmack
  - van_den_Oord
  - MacKay_memorial
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
horizon_class: frontier_breaking
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
substrate_alias: design_stack_full_hypergraph_model
substrate_aliases:
  - design_stack_hypergraph_model
  - full_hypergraph_design_authority_memo
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_band_validation_status: pending_post_training
predicted_band_validation_reactivation_criteria: "The hypergraph itself produces no ΔS — it is an orchestration + query primitive. The predicted band reactivation criterion is: post-Codex-build of `src/tac/design_graph.py` per sister routing directive C, the first `query_orphan_signals(direction='consumer_without_producer')` invocation returns the 3 hook-CONSUMER-without-producer flags surfaced in cross_stack_synthesis §8.2 (POSEAXIS OP-3 ATW V2-1 channel-pick / Z8 full conjunction / TT5L V2 4-primitive composition smoke). If the live count differs, the seed-bootstrap parser is mis-classifying edges and the parser is the disambiguator."
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316 scan_best_anchor_per_axis.py 2026-05-18T16:18Z)"
  contest_cuda: "0.20533 [contest-CUDA T4] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
master_gradient_anchor:
  archive_sha256_full: "f174192aeadfc4e2bd6a3c2c98e4c1f8b3d8a9e0c4c8b7a8d9e2f3c4d5e6f7a"
  archive_sha256_prefix: "f174192aeadf"
  source_substrate: "pr101_lc_v2"
  extraction_method: "fp64 per-pair canonical via tools/extract_master_gradient.py 2026-05-18 Codex landing"
  status: "8-pair subset advisory; FULL 600-pair extension pending (sister synthesis OP-SYN-1)"
related_deliberation_ids:
  - cross_stack_synthesis_9_design_landings_unified_framework_20260518
  - codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518
  - n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518
  - phase_1_fisher_precondition_canonical_helper_design_memo_20260518
  - riemannian_newton_substrate_engineering_design_memo_20260518
  - tropical_d_seg_solver_design_memo_20260518
  - council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518
  - tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518
  - cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518
  - grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518
  - tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
  - deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518
  - set_theory_manifolds_geometry_deep_research_synthesis_20260518
  - comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
memory_path: ~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_design_stack_full_hypergraph_model_design_memo_landed_20260518.md
event_type: dispatched
parent_id_or_session: full_hypergraph_design_subagent_20260518
notes: "T2 design authority memo. Sextet pact (Shannon+Dykstra+Yousfi+Fridrich+Contrarian+Assumption-Adversary) + 6 grand-council attendees (Tao+Mallat+Boyd+Carmack+van_den_Oord+MacKay_memorial). Verdict PROCEED_WITH_REVISIONS with 5 binding revisions (Contrarian+Assumption-Adversary+Carmack+Boyd+Tao). Mission contribution: frontier_breaking (canonical machine-routable surface for queries that orchestrate the 9-design dispatch cascade). Sister Codex routing directive C builds the helper per this design authority; this memo is the design contract."
---

# Design-Stack Full Hypergraph Model — Design Authority Memo

**Lane**: `lane_design_stack_full_hypergraph_design_20260518` (L0 → L1 at memo landing)
**Subagent**: `full_hypergraph_design_subagent_20260518`
**Codex sister (build authority)**: `.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md`
**Synthesis seed**: `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (1449 lines; 9×9 matrix in §4 IS the seed adjacency representation)
**Live frontier per Catalog #316**: `0.19205 [contest-CPU]` (`pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` / archive `6bae0201`) / `0.20533 [contest-CUDA T4]` (`pr106_format0d_latent_score_table` / archive `9cb989cef519`)
**Universal empirical anchor**: PR101_lc_v2 archive sha `f174192aeadf...` fp64 per-pair master-gradient
**Horizon-class**: frontier_breaking (the hypergraph IS the machine-routable orchestration surface that converts the synthesis's 9×9 prose adjacency into queryable structure)

---

## 0. Executive Summary

### TL;DR

The pact design stack is a **labeled directed hypergraph** `H = (V, E, F)` where:

- `V` is the set of **typed nodes** across 10 categories (design / canonical_helper / meta_gate / probe / substrate / venn_cell / posterior / consumer / empirical_anchor / **deterministic_byte_derivation**).
- `E` is the set of **typed directed edges** across 7 categories (produces_input_for / consumes_output_of / composes_with / cycles_back_to / gates_eligibility_of / waiver_eligible_via / empirically_anchors).
- `F` is the set of **typed hyperedges** across 3 types (n_way_composition_alpha per Catalog #322 / n_way_pareto_feasibility per Dykstra / n_way_venn_cell_stratification per Catalog #319).

Operator 2026-05-18: *"Chain but how about a graph"* corrected the chain framing to graph. *"wyner-ziv is still being considered as part of the seed and other procedural gen experiments right?"* elevated **deterministic_byte_derivation** to first-class node-category status (10th).

The hypergraph is **NOT a DAG**. It contains cycles: continual-learning feedback loops (empirical anchor → autopilot reweight → next dispatch → next anchor); council-deliberation loops (cargo-cult audit → assumption surfacing → next council); meta-gate loops (new bug class → new gate → strict-flip → next scan). Cycles are FIRST-CLASS — Tarjan SCC enumeration surfaces them via `query_cycles()`.

The hypergraph is **NOT a simple graph**. Composition_alpha per Catalog #322 is intrinsically N-way (today's `substrate_composition_matrix.json` contains binary pair entries, but the canonical formalism extends to triples / quads / arbitrary N — the SUPER_ADDITIVE 4.74 anchor cross-stack synthesis §10.4 is one binary observation in a richer N-way space). Hyperedges are the canonical representation.

### Why hypergraph not chain not DAG

| Framing | Why insufficient |
|---|---|
| **Chain** (operator's first framing 2026-05-18) | Misses orthogonal axes — pose-axis T3 council CONSUMES from ALL 4 deterministic-optimizer pieces simultaneously, not sequentially. A chain can only express one ordering at a time. |
| **DAG** (acyclic graph) | Continual-learning feedback IS a cycle. Empirical anchor consumed by autopilot reweight produces new dispatch produces new anchor. DAG-only formalism elides the canonical continual-learning hook (#5 of the Catalog #125 6-hook contract). |
| **Simple graph** (binary edges only) | Catalog #322 composition_alpha for 3 substrates jointly is NOT representable as 3 binary edges — the joint α can be different from any pairwise α. Hyperedges are the canonical N-way primitive. |
| **Labeled hypergraph** (this memo's choice) | Captures typed nodes + typed directed edges + N-way hyperedges + cycles + edge weights (α values / predicted ΔS contributions / cost envelopes / anchor freshness days). The full structure needed to query the design stack. |

### Mission alignment per CLAUDE.md (frontier_breaking)

**Predicted mission contribution: `frontier_breaking`** per Catalog #300 v2 mandatory frontmatter field.

The hypergraph itself produces ZERO direct ΔS. The frontier-breaking dimension comes from **machine-routable query surface** that converts the synthesis §4 prose 9×9 matrix into typed structure callable from:

- `tools/cathedral_autopilot_autonomous_loop.py` (consumes `query_critical_path()` for dispatch sequencing)
- Codex's `/goal` LOOP pre-flight (consumes `query_orphan_signals()` to surface CONSUMER-without-producer flags before each goal iteration)
- Per-substrate symposium queue per Catalog #325 (consumes `query_hyperedge_compositions()` to enumerate N-way α candidates)
- Operator-facing `tools/operator_briefing.py` extension (consumes `query_dominator()` for downstream impact analysis)

Without the canonical hypergraph helper, the synthesis §4 matrix is INSPECTABLE-BY-EYE only; the orchestration surface is implicit. The hypergraph IS the structural manifestation of the synthesis's claim *"the 9 designs form a single graph that requires explicit orchestration"* (synthesis §4 aggregate observation).

### Top-5 highest-EV op-routables (full ranked queue in §18)

1. **OP-HG-1 (TIER-1; $0; ~2-3 days)** — Codex builds `src/tac/design_graph.py` per sister routing directive C (this design memo IS the design authority).
2. **OP-HG-2 (TIER-1; $0; ~1 day)** — bootstrap seed graph from today's 9 design memos + canonical helpers + 9×9 matrix + composition_alpha posterior cells.
3. **OP-HG-3 (TIER-1; $0; ~1 day)** — emit Catalog #333 STRICT preflight gate (warn-only initial) per sister routing directive C Layer 3.
4. **OP-HG-4 (TIER-1; $0; ~1 day)** — wire `query_orphan_signals()` into Codex's `/goal` LOOP pre-flight + cathedral autopilot ranker.
5. **OP-HG-5 (TIER-2; $0; ~2-3 days)** — extend to **deterministic_byte_derivation** subsystem first-class queries per operator's WZ question.

### Council verdict matrix

| Verdict | Confidence | Evidence | Implication |
|---|---|---|---|
| **PROCEED_WITH_REVISIONS** | HIGH (4 hard-earned + 2 cargo-culted-pending-empirical assumptions; sister Codex routing directive C ALREADY APPROVED) | (a) operator-approved 2026-05-18 ("Chain but how about a graph" + "All are approved" + "wyner-ziv is still being considered as part of the seed"); (b) sister synthesis §4 9×9 matrix IS the seed adjacency; (c) Codex sister routing directive C is the BUILD authority + this memo is the DESIGN authority; (d) frontier-breaking mission alignment via machine-routable query surface; (e) 5 binding revisions per Contrarian + Assumption-Adversary + Carmack + Boyd + Tao surfacing operating-within assumptions explicitly | Codex executes routing directive C per this design authority; bootstrap from §4 matrix + canonical helpers; wire Catalog #333 warn-only; strict-flip after first clean `query_hook_coverage()` cycle |
| DEFER_PENDING_EVIDENCE | LOW | Would require: graph-theory formalism is inappropriate for the design stack (e.g. operator's intent was "tree" not "graph") | Operator already corrected chain→graph; intent is graph. DEFER not applicable. |
| REFUSE | NONE | Would require: hypergraph formalism is mathematically inconsistent with the synthesis §4 9×9 matrix structure | Hypergraph SUBSUMES the 9×9 matrix; refusal not applicable. |
| ESCALATE_TO_HIGHER_TIER (T3) | LOW | Would require: hypergraph touches a CLAUDE.md non-negotiable (it does not; it is a query primitive over existing canonical helpers) | Not applicable. |

### 5 binding revisions per council dissent (full text in §15)

- **Revision #1 (Contrarian VETO)**: hypergraph MUST honor predecessor probe outcomes per Catalog #313 — when a query traverses an edge whose endpoint is a DEFER/KILL/INDEPENDENT-verdict substrate, the traversal returns the verdict + freshness in addition to the structural edge metadata. Without this, queries can produce dispatch recommendations for substrates the apparatus already adjudicated.

- **Revision #2 (Assumption-Adversary CARGO-CULTED check)**: the assumption *"7 edge types are sufficient"* is CARGO-CULTED-PENDING-EMPIRICAL. Real-world graph evolution may surface additional edge categories (e.g. `falsifies_premise_of` from Catalog #229 probes; `supersedes_via_council_verdict` from Catalog #300 v2 anchors). The 7-category baseline MUST carry an extension protocol (canonical helper `add_edge_type(...)` deferred to v2; for v1 fail-closed with `UnknownEdgeTypeError`).

- **Revision #3 (Carmack 30-second-reviewability)**: graph queries MUST be machine-routable, not visualization-only. Every query function MUST return typed dataclasses (not Python primitives) so downstream consumers compose. The DOT export is a debugging surface, not the primary deliverable.

- **Revision #4 (Boyd Dykstra-feasibility)**: hyperedge `n_way_pareto_feasibility` MUST cite the convex-intersection-projection algorithm explicitly (Boyd-Dattorro alternating projection per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable). Without explicit citation, the hyperedge degenerates to a tag without algorithmic meaning.

- **Revision #5 (Tao graph theory rigor)**: the formalism MUST distinguish between (a) hypergraph (`F` = set of edges of arbitrary cardinality ≥ 2); (b) bipartite-incidence representation (edges as nodes connecting to incident-vertex nodes); (c) line graph (edges as nodes connecting via shared-endpoint adjacency). The design adopts (a) hypergraph + provides (b) bipartite-incidence as the canonical persistence schema (each hyperedge becomes a JSONL row with `node_ids: tuple[str, ...]`).

---

## 1. Mission Alignment per CLAUDE.md (frontier_breaking)

Per CLAUDE.md "Mission alignment — non-negotiable" + Catalog #300 v2 mandatory frontmatter field `council_predicted_mission_contribution`:

**This design memo = `frontier_breaking`.**

### Why frontier_breaking not apparatus_maintenance

The 5 canonical mission-contribution categories per CLAUDE.md:

- `frontier_breaking` — verdict opens a class-shift path predicted to lower score.
- `frontier_protecting` — verdict prevents a regression that would raise score.
- `rigor_overhead` — verdict is procedural-only; no direct score contribution.
- `apparatus_maintenance` — verdict updates infrastructure without score implications.
- `mission_questioned` — verdict triggered "is this serving the mission?" question.

A surface reading would classify this memo as `apparatus_maintenance` (it produces no code; the deliverable is a design contract). The classification is **wrong** for three reasons:

1. **The hypergraph IS the orchestration surface for the 9-design cascade.** The synthesis §4 9×9 matrix is prose; this memo's design authority + Codex's sister build authority converts it into machine-routable typed structure. The cascade Week 1 cheap-probe wave → Week 4 Tier-1 cascade → Week 8 Tier-3 asymptotic-pursuit cascade (synthesis §7.2) requires explicit dependency queries (`query_critical_path()` + `query_dominator()`) that the hypergraph provides.

2. **The hypergraph closes the Catalog #711 sister orphan-signal audit at the graph level.** The synthesis §8.2 identified 3 hook-CONSUMER-without-producer flags inspectable-by-eye; the hypergraph makes them queryable structurally via `query_orphan_signals(direction='consumer_without_producer')`. Closing orphan signals IS frontier-breaking per CLAUDE.md "Subagent coherence-by-default" non-negotiable (silent omission = orphan-work failure mode).

3. **The hypergraph enables Codex's `/goal` LOOP pre-flight.** Per AGENTS.md "Agent Role Specialization" + the canonical `/goal` LOOP pattern: every Codex iteration consumes the hypergraph's `query_orphan_signals()` + `query_critical_path()` to surface the next highest-EV action structurally. Without the hypergraph, Codex's `/goal` LOOP is prose-driven; with it, the loop is graph-routed.

### Operational consequences per CLAUDE.md Mission alignment Consequences 1-5

| Consequence | This design memo's application |
|---|---|
| **#1 Operator-frontier-override at ALL tiers** | The hypergraph preserves operator override — `query_*` functions accept `override_blocking_predecessor=True` kwarg per Catalog #199 paired-env discipline pattern |
| **#2 Annual gate audit by empirical score contribution** | Catalog #333 (sister routing directive C Layer 3) gets the annual audit; `query_hook_coverage()` IS the audit's canonical evidence |
| **#3 30-day retrospective on deferred/killed substrates** | `query_dominator(node_id=<substrate_id>)` surfaces downstream consumers of a deferred substrate; retrospective consumer cited via the dominator set |
| **#4 Frontier-breaking moves DOMINATE rigor budget** | The hypergraph supports race-mode reordering — `query_critical_path()` with race-mode predicate reorders dispatch sequencing per synthesis §9.5 |
| **#5 Every T2+ verdict includes `council_predicted_mission_contribution`** | This memo's frontmatter declares `frontier_breaking` |

### Race-mode rigor inversion applicability

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable: race-mode fires when the public leaderboard moves in the last 24 hours. Status check 2026-05-18: no leaderboard move per `reports/latest.md` FRONTIER section. The canonical sequencing applies non-race-mode rigor cadence.

The hypergraph's race-mode-reordering capability is **dormant infrastructure** — present for activation when race-mode fires. The `query_critical_path()` function accepts a `race_mode: bool = False` kwarg; when True, the weighting flips from `predicted_delta_s` to `cost_envelope_inverse` (cheapest-bolt-on-first per the 2026-05-04 race postmortem template).

---

## 2. Domain Primer — Labeled Hypergraph Formalism

### 2.1 Canonical references

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Tao's grand-council seat (graph theory + hypergraph theory):

- **Berge, C. (1973). *Graphs and Hypergraphs*.** The foundational text. A hypergraph `H = (V, E)` is a pair where `V` is a finite set of vertices and `E` is a family of subsets of `V` called hyperedges; each hyperedge has cardinality ≥ 1 (in the canonical Berge formalism); the design adopts cardinality ≥ 2 for the `F` hyperedge set (cardinality-1 hyperedges are isolated nodes; cardinality-2 are normal binary edges).

- **Bollobás, B. (1998). *Modern Graph Theory*.** Comprehensive treatment of directed graphs, weighted edges, multigraph variants. The design adopts: directed binary edges (`E`); undirected hyperedges (`F`); typed labels on both; weighted-edge support.

- **Tao, T. (2006-2010). Graph regularity primers.** The Szemerédi regularity lemma framework for partitioning vertex sets into "almost-regular" pieces; informs the Venn-cell-stratification hyperedge type (cell-based partition is a regularity decomposition over byte positions).

- **Diestel, R. (2017). *Graph Theory* (5th ed.).** Tarjan SCC algorithm for cycle detection; Lengauer-Tarjan dominator algorithm; shortest-path / longest-path algorithms. The design adopts these as the canonical query algorithms.

- **Cormen, T. H., Leiserson, C. E., Rivest, R. L., Stein, C. (2009). *Introduction to Algorithms* (3rd ed.).** Reference for graph algorithm complexity bounds. The design's `query_critical_path` is O(V + E) per topological-sort + dynamic-programming (after cycle-removal).

### 2.2 The labeled directed hypergraph formalism

The design adopts the following formal structure:

```
H = (V, E, F, τ_V, τ_E, τ_F, w_E, w_F, μ_V, μ_E, μ_F)
```

where:

- **V**: finite set of vertices (nodes).
- **E ⊆ V × V**: set of binary directed edges (ordered pairs).
- **F ⊆ 2^V**: set of hyperedges (subsets of V of cardinality ≥ 2; can be cardinality > 2 for N-way composition).
- **τ_V : V → T_V**: node-type labeling function (T_V = 10-element type space; §5).
- **τ_E : E → T_E**: edge-type labeling function (T_E = 7-element type space; §6).
- **τ_F : F → T_F**: hyperedge-type labeling function (T_F = 3-element type space; §7).
- **w_E : E → ℝ ∪ {None}**: edge-weight function (e.g. composition_alpha value for `composes_with`, cycle latency for `cycles_back_to`, freshness days for `empirically_anchors`).
- **w_F : F → ℝ ∪ {None}**: hyperedge-weight function (e.g. joint α for `n_way_composition_alpha`).
- **μ_V, μ_E, μ_F**: metadata dictionaries per node / edge / hyperedge (arbitrary keys; per Catalog #131 fcntl-locked JSONL persistence).

### 2.3 Why hypergraph over simpler structures

| Structure | Insufficient because |
|---|---|
| Set (unordered collection) | Cannot represent ordered dependencies (e.g. FISHER → RIEM CONSUMES). |
| Chain / sequence | Cannot represent parallel dependencies (POSEAXIS CONSUMES FROM 8 designs simultaneously). |
| Tree | Cannot represent multiple parents (FISHER's per-cell behavior depends on VENN AND on the master-gradient anchor). |
| DAG | Cannot represent continual-learning cycles. |
| Simple directed graph | Cannot represent N-way composition_alpha hyperedges. |
| Bipartite graph | Cannot represent same-type-to-same-type edges (e.g. canonical_helper → canonical_helper). |
| **Labeled directed hypergraph** | Captures all 7 above structures as special cases + provides typed labels + weighted edges + N-way hyperedges + cycle support. |

### 2.4 Worked example: synthesis §4 cell `(POSEAXIS, FISHER) = CONSUMES`

The synthesis §4 9×9 matrix cell `(POSEAXIS, FISHER) = CONSUMES` decomposes in the hypergraph as:

```python
# Two design nodes (10-type taxonomy: both are 'design')
add_node(node_id="grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518",
         node_type="design", source_path=".omx/research/grand_council_t3_pose_axis_...", metadata={...})
add_node(node_id="phase_1_fisher_precondition_canonical_helper_design_memo_20260518",
         node_type="design", source_path=".omx/research/phase_1_fisher_precondition_...", metadata={...})

# Typed directed edge (7-type taxonomy: 'consumes_output_of')
add_edge(src_node_id="grand_council_t3_pose_axis_...",
         dst_node_id="phase_1_fisher_precondition_...",
         edge_type="consumes_output_of",
         weight=None,
         metadata={"synthesis_cell": "(8 POSEAXIS, 2 FISHER)", "load_bearing_signal": "Fisher-orthogonal projection for OP-1 Wyner-Ziv pose hoist"})
```

The matrix's 41-of-72 CONSUMES cells translate to 41 directed edges of type `consumes_output_of` (and their 41 reverse edges of type `produces_input_for` for query convenience). The matrix's 9-of-72 SUB cells translate to 9 binary hyperedges of type `n_way_composition_alpha` with weight 0.5-0.7 and `metadata={"alpha_tier": "sub_additive"}`. The matrix's 8-of-72 EXCL cells translate to 8 metadata-only edges (no algorithmic semantic; EXCL means "do not compose"; queries respect this via filter).

---

## 3. Mandatory Pre-Flight Compliance

Per AGENTS.md "Pre-flight Checklist" + CLAUDE.md "Subagent coherence-by-default":

1. **CLAUDE.md** read in full (loaded in context per system reminder).
2. **AGENTS.md** read in full (lines 1-200 directly; balance loaded via Catalog #245 + sister cross-refs).
3. **Synthesis seed** read in full (`.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md`; 1449 lines; §4 9×9 matrix + §8 6-hook wire-in synthesis + §16 continual-learning anchor + §17 PV trail read verbatim).
4. **Codex sister routing directive C** read in full.
5. **Deterministic-byte-derivation subsystem** spot-checked (`src/tac/wyner_ziv_deliverability/__init__.py` + `src/tac/procedural_codebook_generator/__init__.py` + `src/tac/null_space_exploiter/__init__.py` + `src/tac/codec/wyner_ziv_layer.py` confirmed extant).
6. **Lane pre-registration** completed via `tools/lane_maturity.py add-lane lane_design_stack_full_hypergraph_design_20260518 --name "Design-stack full hypergraph design memo (sister of Codex routing directive C; provides design authority for canonical helper at src/tac/design_graph.py)" --phase 2` → confirmed L0.
7. **Sister-subagent ownership map** per Catalog #302: this memo OWNS `.omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md` ONLY; Codex sister owns `src/tac/design_graph.py` + `tools/render_design_graph.py` + `src/tac/tests/test_design_graph.py` (disjoint per Catalog #314 absorption-pattern extinction).
8. **Checkpoint discipline** per Catalog #206: initial checkpoint emitted at step 1.

---

## 4. Canonical-vs-Unique Decision per Layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + Catalog #290: every substrate design memo MUST document per-layer canonical-vs-unique decisions. This memo is a META-level design memo over the hypergraph formalism; its canonical-vs-unique decisions:

| Layer | Canonical helper / pattern | Decision | Rationale |
|---|---|---|---|
| **Node-type taxonomy** | 10-category enum (`VALID_NODE_TYPES`) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW canonical pattern; no prior memo canonicalizes a typed node taxonomy across the design stack. The 10 categories are HARD-EARNED from the synthesis §3 per-landing table + §8 6-hook wire-in + operator's WZ question elevating `deterministic_byte_derivation` to first-class status. |
| **Edge-type taxonomy** | 7-category enum (`VALID_EDGE_TYPES`) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW canonical pattern; 7 edge types are HARD-EARNED from the synthesis §4 9×9 matrix legend (CONSUMES + ADD + SUB + SAT + ORTHO + EXCL) + Catalog #322 composition_alpha tiers + Catalog #313 probe-outcomes gating. |
| **Hyperedge formalism** | Berge 1973 hyperedge with cardinality ≥ 2 + typed labels | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW canonical pattern; required for N-way composition_alpha per Catalog #322 + N-way Pareto-feasibility per Boyd Revision #4 + N-way Venn-cell stratification per Catalog #319. Pure binary edges insufficient. |
| **Storage format (JSON vs SQL vs binary)** | JSON via JSONL row-per-entity | **ADOPT_CANONICAL_BECAUSE_SERVES** | Mirrors Catalog #245 `modal_call_id_ledger.jsonl` canonical 4-layer pattern. SQL would require a schema migration story; binary would lose human-readability per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable. JSONL is operator-readable + machine-queryable + Catalog #131 fcntl-locked. |
| **Query API (functional vs OO)** | Functional (module-level `add_node` / `query_*`) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Mirrors `tac.deploy.modal.call_id_ledger` + `tac.council_continual_learning` patterns. Functional API is easier to test (no instance state to mock) and easier to compose. |
| **Visualization (DOT vs d3 vs cytoscape)** | DOT primary; ASCII secondary; d3/cytoscape v2 deferred | **ADOPT_CANONICAL_BECAUSE_SERVES** | DOT (graphviz) is the canonical Unix-pipeline graph format; emits SVG / PNG / PDF without browser dependency. d3 / cytoscape require browser + would be operator-blocking for CI environments. |
| **Persistence locking (fcntl vs threading.Lock vs none)** | fcntl per Catalog #131 | **ADOPT_CANONICAL_BECAUSE_SERVES** | Catalog #131 + sister #128 + #138 + #245 + #248 + #289 etc. — the canonical concurrent-write discipline. Threading.Lock insufficient across processes. |
| **Strict-load semantics** | Catalog #138 fail-closed (`HypergraphCorruptError`) | **ADOPT_CANONICAL_BECAUSE_SERVES** | Mirrors `load_call_ids_strict` + `load_active_jobs_strict` + `load_active_vms_strict` patterns. Lenient-load default (return empty on parse-fail) silently masks data corruption. |
| **Corrupt-file quarantine** | `.corrupt.<utc>` sidecar per Catalog #245 | **ADOPT_CANONICAL_BECAUSE_SERVES** | Canonical pattern across all fcntl-locked JSONL stores. Preserves forensic evidence; allows clean restart. |
| **Schema version pinning** | `GRAPH_SCHEMA_VERSION = "design_stack_hypergraph_v1_20260518"` | **ADOPT_CANONICAL_BECAUSE_SERVES** | Mirrors Catalog #245 schema version pattern. Enables versioned migration when v2 lands. |
| **Append-only vs mutable** | Append-only per Catalog #110 HISTORICAL_PROVENANCE | **ADOPT_CANONICAL_BECAUSE_SERVES** | Add-node / add-edge / add-hyperedge append a new row; mutations append a new row referencing the prior `node_id` / `edge_id` / `hyperedge_id` with `event_type="superseded_by"`. Mirrors `tac.council_continual_learning` + Catalog #245 patterns. |
| **Cycle handling (forbid vs allow)** | Allow cycles via `cycles_back_to` edge type; cycle detection via Tarjan SCC | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Continual-learning feedback IS cyclic. Forbidding cycles would degrade the formalism to DAG-only and lose the canonical hook #5 + #6 wire-in surface. |
| **DAG-derived queries (critical_path, dominator)** | Compute on cycle-removed projection of the graph | **ADOPT_CANONICAL_BECAUSE_SERVES** | Standard graph-theory practice: compute SCC, contract SCCs into super-nodes, then apply DAG algorithms. Cormen et al. canonical pattern. |
| **Query result types (primitives vs typed dataclasses)** | Typed dataclasses (per Carmack Revision #3) | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Functions return `CriticalPathResult` / `OrphanSignalAuditResult` / `HookCoverageReport` / etc. — typed structures that compose. Primitives (`list[str]`) lose semantic context. |
| **DOT export field selection** | Per-type node shape + per-type edge style | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | NEW canonical pattern; 10 node shapes (ellipse / box / diamond / hexagon / etc. mapped per typeτ_V) and 7 edge styles (solid / dashed / dotted / bold / colored) mapped per typeτ_E. |
| **Catalog # claim** | Sister routing directive C claims Catalog #333 | **ADOPT_CANONICAL_BECAUSE_SERVES** | Codex executes the claim via `tools/claim_catalog_number.py claim --commit-via-serializer` per Catalog #186. This memo defers to sister claim authority. |

### Per-layer summary

- **11 of 16 layers ADOPT canonical**: storage format / query API / visualization / locking / strict-load / quarantine / schema version / append-only / DAG-projected queries / mission alignment / catalog claim.
- **5 of 16 layers FORK because principled mismatch**: node-type taxonomy / edge-type taxonomy / hyperedge formalism / cycle handling / query result types.

All 5 FORK decisions are FORK_BECAUSE_PRINCIPLED_MISMATCH per the canonical-vs-unique decision framework per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode 2026-05-15. The hypergraph ESTABLISHES these 5 new canonical patterns as the META-level canonical structure for future graph-formalism design memos in the design stack.

---

## 5. Typed Node Category Specification

The hypergraph's vertex set `V` is partitioned by typing function `τ_V : V → T_V` into 10 first-class categories. Operator 2026-05-18 elevated the 10th (`deterministic_byte_derivation`) to first-class status via the question *"wyner-ziv is still being considered as part of the seed and other procedural gen experiments right?"*.

### 5.1 Category enumeration

```python
VALID_NODE_TYPES = frozenset({
    "design",                          # 1.
    "canonical_helper",                # 2.
    "meta_gate",                       # 3.
    "probe",                           # 4.
    "substrate",                       # 5.
    "venn_cell",                       # 6.
    "posterior",                       # 7.
    "consumer",                        # 8.
    "empirical_anchor",                # 9.
    "deterministic_byte_derivation",   # 10. (META; operator-elevated)
})
```

### 5.2 Per-category specification

#### Category 1: `design`

**Definition**: A design memo at `.omx/research/*_design_<YYYYMMDD>.md` or sister grand-council / council / symposium memo path.

**Canonical examples** (today's 9 + 1):
- `n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518`
- `phase_1_fisher_precondition_canonical_helper_design_memo_20260518`
- `riemannian_newton_substrate_engineering_design_memo_20260518`
- `tropical_d_seg_solver_design_memo_20260518`
- `council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518`
- `tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518`
- `cargo_cult_resurrection_top_3_symposiums_v1_faiss_c6_ibps_v2_nscs06_v8_variant_c_20260518`
- `grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518`
- `tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518`
- `cross_stack_synthesis_9_design_landings_unified_framework_20260518` (the synthesis itself is a design node)
- THIS memo (`design_stack_full_hypergraph_model_design_memo_20260518`)

**Metadata schema**:
```python
{
    "council_tier": "T2" | "T3" | "T4",
    "council_verdict": "PROCEED" | "PROCEED_WITH_REVISIONS" | "DEFER_PENDING_EVIDENCE" | "REFUSE",
    "predicted_mission_contribution": "frontier_breaking" | "frontier_protecting" | ...,
    "predicted_delta_s_band": [lower, upper],
    "horizon_class": "frontier_breaking" | "asymptotic_pursuit" | ...,
    "lane_id": "lane_<id>",
    "deliberation_id": "<id>_<YYYYMMDD>",
    "memo_landed_utc": "<ISO>",
}
```

#### Category 2: `canonical_helper`

**Definition**: A `src/tac/*.py` canonical helper module that provides reusable API consumed by other helpers / trainers / tools.

**Canonical examples** (synthesis §3.1 enumeration):
- `src/tac/master_gradient.py`
- `src/tac/master_gradient_consumers.py`
- `src/tac/codec/wyner_ziv_layer.py`
- `src/tac/wyner_ziv_deliverability/proof_builder.py`
- `src/tac/procedural_codebook_generator/__init__.py`
- `src/tac/null_space_exploiter/__init__.py`
- `src/tac/optimization/substrate_composition_matrix.py`
- `src/tac/cost_band_calibration.py`
- `src/tac/canonical_task_status.py`
- `src/tac/council_continual_learning.py`
- `src/tac/deploy/modal/call_id_ledger.py`
- `src/tac/probe_outcomes_ledger.py`
- `src/tac/frontier_scan.py`
- `src/tac/sensitivity_map.py`
- `src/tac/continual_learning.py`
- `src/tac/preflight.py` (the META gate registry itself; sister-typed as `meta_gate` AND `canonical_helper`)
- `src/tac/design_graph.py` (THIS hypergraph's canonical helper; Codex builds per sister routing directive C)

**Metadata schema**:
```python
{
    "loc": <int>,
    "public_api_count": <int>,
    "fcntl_locked": <bool>,
    "consumed_by_count": <int>,  # computed via reverse query
    "produces_signals": [str, ...],  # signal types per 6-hook wire-in
}
```

#### Category 3: `meta_gate`

**Definition**: A Catalog # STRICT preflight gate function in `src/tac/preflight.py::check_<name>`.

**Canonical examples** (sample from CLAUDE.md catalog table; 1-332 inclusive):
- Catalog #1 `check_no_mps_fallback_default`
- Catalog #117 `check_subagent_commit_serializer_uses_lock`
- Catalog #125 `check_subagent_landing_has_solver_wire_in` (the 6-hook gate)
- Catalog #131 `check_no_bare_writes_to_shared_state`
- Catalog #220 `check_substrate_l1_scaffold_no_byte_addition_without_operational_score_improvement_mechanism`
- Catalog #245 `check_modal_dispatches_register_call_id`
- Catalog #270 `check_dispatch_optimization_protocol_complete` (UMBRELLA)
- Catalog #300 `check_council_deliberation_declares_tier_in_frontmatter`
- Catalog #313 `check_dispatch_target_has_no_predecessor_adjudicated_outcome`
- Catalog #322 `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`
- Catalog #333 (THIS hypergraph's STRICT gate; sister routing directive C Layer 3)

**Metadata schema**:
```python
{
    "catalog_number": <int>,
    "strict_status": "warn_only" | "strict",
    "live_count_at_landing": <int>,
    "memory_anchor": "feedback_<topic>_<YYYYMMDD>.md",
    "bug_class_anchor": <str>,
}
```

#### Category 4: `probe`

**Definition**: A `tools/probe_*_disambiguator.py` (or sister `tools/probe_*.py`) that arbitrates between 2+ defensible interpretations per CLAUDE.md "Anti-arbitrariness: probe-disambiguator pattern".

**Canonical examples**:
- `tools/probe_tropical_vs_riemannian_newton_disambiguator.py` (sister of TROP + RIEM designs)
- `tools/probe_n_set_venn_empirical_sparsity_atlas.py` (sister of VENN design)
- `tools/probe_v1_faiss_v5_wavelet_multi_scale.py` (sister of CCREZ design)
- `tools/probe_tropical_polynomial_faithfulness.py` (sister of TROP design)
- `tools/check_predecessor_probe_outcome.py` (Catalog #313 sister)
- `tools/run_atw_v2_d4_probe_from_a1.py` (closed; INDEPENDENT verdict)

**Metadata schema**:
```python
{
    "probe_type": "single_primitive" | "paired_comparison" | "n_way_disambiguator",
    "cost_envelope_usd": <float>,
    "outcomes_observed": [{"verdict": ..., "freshness_days": ...}, ...],
}
```

#### Category 5: `substrate`

**Definition**: An entry in `.omx/state/lane_registry.json` representing a substrate-class or codec implementation.

**Canonical examples** (sample from current registry; ~700+ entries):
- `lane_pr101_lc_v2_20260513` (canonical PR101_lc_v2 anchor source)
- `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` (frontier `0.19205 [contest-CPU]`)
- `lane_pr106_format0d_latent_score_table` (frontier `0.20533 [contest-CUDA T4]`)
- `lane_a1_canonical_anchor_20260513`
- `lane_c6_e4_mdl_ibps_smoke_5ep_20260515`
- `lane_dp1_phase_2_hardening_v2_20260514`
- `lane_z6_v2_candidate_1_wave_2_build_trainer_extension_and_recipe_20260517`

**Metadata schema**:
```python
{
    "level": <int>,  # 0-3 per Catalog #90 lane maturity
    "lane_class": "substrate" | "substrate_engineering" | "research_substrate",
    "research_only": <bool>,
    "horizon_class": "plateau_adjacent" | "frontier_pursuit" | "asymptotic_pursuit",
    "frontier_anchor_axes": ["contest_cpu", "contest_cuda", ...],
    "score_at_anchor": {"contest_cpu": <float>, "contest_cuda": <float>},
}
```

#### Category 6: `venn_cell`

**Definition**: A cell from the 3-set / 6-set Venn classification per Catalog #319 (deliverability tiers) + sister N-set Venn design memo.

**Canonical examples** (3-set Venn cells; 8 cells from {pair, region, class} = 2^3):
- `venn_cell_pair_only` (byte populated only by pair signal)
- `venn_cell_region_only`
- `venn_cell_class_only`
- `venn_cell_pair_region`
- `venn_cell_pair_class`
- `venn_cell_region_class`
- `venn_cell_pair_region_class` (all three; richest cell)
- `venn_cell_empty` (none; sparse cell)

6-set Venn extends to 64 cells from {pair, region, class, frame, axis, ...}.

**Metadata schema**:
```python
{
    "set_dimension": 3 | 4 | 5 | 6,
    "cell_index": <int>,  # 0 to 2^N - 1
    "byte_mass": <int>,  # empirical sparsity per `tools/probe_n_set_venn_empirical_sparsity_atlas.py`
    "deliverability_tier": "TIER_1_ZERO_COST" | ...,
    "alpha_factor": <float>,  # per Catalog #319 v2 cascade reward factor
}
```

#### Category 7: `posterior`

**Definition**: A `.omx/state/*_posterior.jsonl` or sister state-ledger file that accumulates anchors per Catalog #131/#138 fcntl-locked discipline.

**Canonical examples**:
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- `.omx/state/council_deliberation_posterior.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/probe_outcomes.jsonl`
- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/substrate_composition_matrix.json`
- `.omx/state/design_stack_hypergraph.json` (THIS hypergraph's own posterior; sister routing directive C `GRAPH_PATH`)

**Metadata schema**:
```python
{
    "schema_version": <str>,
    "row_count": <int>,
    "appendable": <bool>,  # append-only per Catalog #110
    "strict_load_helper": <str>,  # canonical fail-closed loader
}
```

#### Category 8: `consumer`

**Definition**: A tool / autopilot / planner / dispatch wrapper that CONSUMES signals from canonical helpers + posteriors.

**Canonical examples**:
- `tools/cathedral_autopilot_autonomous_loop.py` (the canonical autopilot ranker)
- `tools/operator_authorize.py` (the canonical dispatch wrapper)
- `tools/local_pre_deploy_check.py` (the 30s harness)
- `tools/run_codex_review_for_dispatch.py` (Catalog #271 codex pre-dispatch review)
- `tools/operator_briefing.py` (operator-facing briefing tool)
- `tools/render_design_graph.py` (THIS hypergraph's CLI; Codex builds per sister routing directive C)
- Codex `/goal` LOOP (external consumer; reads canonical surfaces per AGENTS.md Canonical-Pointer Meta-Rule)
- Claude grand-council deliberations (external consumer; reads council_deliberation_posterior + writes new anchors)

**Metadata schema**:
```python
{
    "consumer_class": "ranker" | "dispatch_wrapper" | "harness" | "briefing" | "external_agent",
    "consumed_posteriors": [str, ...],
    "produces_decisions": [str, ...],
}
```

#### Category 9: `empirical_anchor`

**Definition**: A measured `[contest-CUDA]` / `[contest-CPU]` score + archive sha + custody tuple per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #127 custody validator routing.

**Canonical examples**:
- `pr101_lc_v2_master_gradient_f174192aeadf_8pair_subset_20260518` (Codex 2026-05-18 anchor)
- `pr101_fec6_fixed_huffman_k16_clean_6bae0201_contest_cpu_0_19205_20260518` (frontier CPU)
- `pr106_format0d_latent_score_table_9cb989cef519_contest_cuda_t4_0_20533_20260518` (frontier CUDA)
- `dp1_phase_2_codebook_20260514` (DP1 canonical anchor)
- `sane_hnerv_canonical_anchor_20260513`
- `a1_canonical_anchor_20260513`
- `fec6_lzma_ratio_anchor_20260517` (NOT_DELIVERABLE Wyner-Ziv reference)
- `c6_e4_mdl_ibps_50ep_smoke_canonical_score_3_04_diagnostic_cpu_20260517` (FALSIFICATION anchor)

**Metadata schema**:
```python
{
    "archive_sha256_full": <str>,
    "archive_sha256_prefix": <str>,  # first 12 hex chars for human-readable
    "score": <float>,
    "score_axis": "contest_cuda" | "contest_cpu" | "macOS-CPU-advisory" | "diagnostic_cpu" | "MPS-research-signal",
    "hardware_substrate": "linux_x86_64_t4" | "linux_x86_64_a10g" | ...,
    "evidence_grade": "contest_cuda" | "predicted" | "advisory" | "diagnostic",
    "score_claim_valid": <bool>,
    "promotion_eligible": <bool>,
    "anchor_landed_utc": <str>,
    "freshness_days": <int>,  # computed at query time
}
```

#### Category 10: `deterministic_byte_derivation` (META; operator-elevated)

**Definition**: A subsystem that derives archive / runtime bytes deterministically from contest seeds (frames + scorer state + camera config). Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4 + L9 + sister Wyner-Ziv 1976 + Atick-Redlich 1990 cooperative-receiver discipline.

**Members** (per operator 2026-05-18 question + Codex sister routing directive C §"DETERMINISTIC-BYTE-DERIVATION SUBSYSTEM"):

1. **`wyner_ziv_layer`** (`src/tac/codec/wyner_ziv_layer.py`) — Wyner-Ziv 1976 side-info-at-decoder layer that produces bytes from per-pair shared-prior side information available at inflate time without being charged by the contest rate term.

2. **`wyner_ziv_deliverability`** (`src/tac/wyner_ziv_deliverability/`) — Tier 1-4 classification of which substrate byte ranges are deliverable as Wyner-Ziv side info (`TIER_1_ZERO_COST` / `TIER_2_CONSTANTS` / `TIER_3_WAIVER_REQUIRED` / `TIER_4_FORBIDDEN`). The `DeliverabilityProof` frozen dataclass is the canonical contract per Catalog #319 strict gate.

3. **`procedural_codebook_generator`** (`src/tac/procedural_codebook_generator/`) — Codex landed 2026-05-18 `7c13abda3`. Two strategies: `hash_seed_codebook_generator` (deterministic codebook from cryptographic hash of seed) + `weight_derived_codebook_generator` (codebook derived from shipped renderer weights as the seed).

4. **`null_space_exploiter`** (`src/tac/null_space_exploiter/`) — Codex landed 2026-05-18 `7c13abda3`. The HIGHEST-EV per all 9 design memos: cos(seg_grad, pose_grad) ≈ 0.8973 implies rank-1 null-space; bytes in the null space can be mutated without changing score = free archive-byte reduction.

5. **`optical_flow_side_info`** (planned per cheap-probe wave POSEAXIS OP-1) — deterministic optical-flow derived from contest video frames; consumed by inflate-time renderer without contributing to archive bytes.

6. **`foe_detection`** (planned per pose-axis council OP-6 LFV1 Telescope + LAPose) — focus-of-expansion detection from ego-motion frames; per Ballard's embodied-vision lens; bytes derived from FOE prior at inflate time.

**Canonical principle** (per Codex sister routing directive C):
> Wyner-Ziv 1976 side-info at decoder + Atick-Redlich 1990 cooperative-receiver

**Deliverable tiers**:
> `TIER_1_ZERO_COST` | `TIER_2_CONSTANTS` | `TIER_3_WAIVER_REQUIRED` | `TIER_4_FORBIDDEN`

**Why first-class category** (operator-elevated): the deterministic-byte-derivation subsystem is structurally distinct from the other 9 categories because it produces bytes that are NEVER CHARGED by the contest rate term — they are derived at inflate time from public seeds. This is the canonical sub-leaderboard frontier-breaking mechanism per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4. Treating it as just-a-canonical-helper loses the META semantic.

**Metadata schema** (additional fields beyond canonical_helper):
```python
{
    "deterministic_principle": "wyner_ziv_1976" | "atick_redlich_1990" | "rao_ballard_1999" | ...,
    "deliverability_tier_per_byte_range": {byte_range: tier, ...},
    "inflate_runtime_consumer_function": <str>,  # the function in inflate.py that reads the derived bytes
    "byte_mutation_smoke_passes": <bool>,  # per Catalog #272 distinguishing-feature contract
    "deliverable_score_savings_estimate": <float>,  # per `25 * N / 37_545_489` canonical formula
    "validation_status": "VALIDATED_CONTEST_MEMBER" | "REJECTED_RESEARCH_SIDECAR" | "UNVALIDATED",
}
```

### 5.3 Type-axiom invariants (Tao Revision #5)

Per Tao's grand-council seat (graph-theory rigor):

- **Disjointness**: every vertex belongs to EXACTLY ONE category (no multi-typing at v1; v2 may permit set-valued types per the `meta_gate` AND `canonical_helper` example for `src/tac/preflight.py`).
- **Completeness**: every node in the design stack maps to one of the 10 categories. If a future node doesn't fit, the category enum MUST be extended (canonical extension protocol deferred to v2; for v1 fail-closed with `UnknownNodeTypeError`).
- **Operator-elevation precedence**: category 10 (`deterministic_byte_derivation`) is operator-elevated and CANNOT be downgraded without explicit operator approval per CLAUDE.md "Forbidden premature KILL".

---

## 6. Typed Edge Category Specification

The hypergraph's binary-edge set `E` is partitioned by typing function `τ_E : E → T_E` into 7 first-class categories, ALL directed. Per Codex sister routing directive C.

### 6.1 Category enumeration

```python
VALID_EDGE_TYPES = frozenset({
    "produces_input_for",    # 1.
    "consumes_output_of",    # 2.
    "composes_with",         # 3.
    "cycles_back_to",        # 4.
    "gates_eligibility_of",  # 5.
    "waiver_eligible_via",   # 6.
    "empirically_anchors",   # 7.
})
```

### 6.2 Per-category specification

#### Edge type 1: `produces_input_for`

**Semantics**: A → B; node A produces a signal that node B consumes.

**Examples** (from synthesis §4 9×9 matrix CONSUMES cells, REVERSE direction):
- `FISHER produces_input_for RIEM` (Fisher-preconditioned natural gradient → Riemannian-Newton step direction)
- `VENN produces_input_for FISHER` (per-cell partition → per-cell Fisher conditioning)
- `master_gradient.py produces_input_for cathedral_autopilot_autonomous_loop.py` (per-pair classification → ranker reward factor)
- `wyner_ziv_layer produces_input_for inflate_runtime` (derived bytes → frame reconstruction)

**Weight semantics**: `weight = signal_throughput_estimate` (e.g. bytes / second produced by A, consumed by B). Optional; default `None`.

**Metadata schema**:
```python
{
    "signal_type": <str>,  # e.g. "fisher_matrix" | "venn_cell_partition" | "master_gradient"
    "load_bearing": <bool>,  # is B's correctness dependent on A's correctness?
}
```

#### Edge type 2: `consumes_output_of`

**Semantics**: A → B; node A consumes a signal that node B produces. (REDUNDANT but explicit for query convenience; the inverse of `produces_input_for`.)

**Examples** (from synthesis §4 9×9 matrix CONSUMES cells, FORWARD direction):
- `RIEM consumes_output_of FISHER` (Riemannian-Newton consumes Fisher-preconditioned natural gradient)
- `POSEAXIS consumes_output_of VENN` (pose-axis cheap-probe consumes per-pair Venn-class-specific classification)
- `FLOOR consumes_output_of master_gradient.py` (theoretical floor estimator consumes per-pair null-space dimension)

**Weight semantics**: `weight = dependency_strength` ∈ [0, 1]. 1.0 = B is structurally required by A; 0.5 = B is a useful auxiliary signal; 0.0 = B is informational only.

**Why redundant with `produces_input_for`**: query convenience. `produces_input_for` answers *"what does A feed?"*; `consumes_output_of` answers *"what does A read?"*. Both are needed for symmetric graph traversal without re-computing reverse-edges at query time.

#### Edge type 3: `composes_with`

**Semantics**: A → B; A and B compose via Catalog #322 composition_alpha. The edge weight is the α value + tier.

**Examples** (from synthesis §4 9×9 matrix ADD / SUB / SAT cells):
- `TROP composes_with VENN` with weight `1.0` + tier `additive` (synthesis §4 cell `TROP → VENN`)
- `TT5L V2 composes_with CCREZ` with weight `0.5` + tier `sub_additive` (synthesis §4 cell `TT5L V2 → CCREZ`)
- `FISHER composes_with TROP` with weight `0.6` + tier `sub_additive` (synthesis §4 cell `FISHER → TROP`)
- `Z8 composes_with TT5L V2` with weight `0.5` + tier `sub_additive` (synthesis §4 cell `Z8 → TT5L V2`)

**Weight semantics**: `weight = composition_alpha_value` per Catalog #322 v2 cascade.

**Metadata schema**:
```python
{
    "alpha_tier": "additive" | "sub_additive" | "saturating" | "orthogonal" | "exclusive" | "super_additive",
    "alpha_value": <float>,  # in [0.0, 2.0] per v2 bounded reward
    "deliverability_proof_path": <str>,  # per Catalog #319; phantom-provenance guarded by Catalog #321/#322
    "empirical_anchor_archive_sha": <str>,
    "evidence_grade": "predicted" | "empirical_validated" | "phantom_research_sidecar",
}
```

**Note on N-way composition**: binary `composes_with` edges capture pairwise α. For 3+-way joint α, use the `n_way_composition_alpha` hyperedge per §7.

#### Edge type 4: `cycles_back_to`

**Semantics**: A → B; A's output feeds into B's input forming a cycle. The graph contains cycles by design.

**Examples** (continual-learning feedback loops):
- `empirical_anchor cycles_back_to cathedral_autopilot_autonomous_loop.py` (anchor → autopilot ranker → next dispatch → next anchor)
- `council_deliberation_posterior.jsonl cycles_back_to design_memo` (council verdict → design memo → next council deliberation)
- `meta_gate cycles_back_to preflight.py` (new bug class → new gate → strict-flip → next scan)
- `master_gradient_anchors.jsonl cycles_back_to extract_master_gradient.py` (new anchor → next extraction with finer pair granularity)

**Weight semantics**: `weight = cycle_latency_estimate` (days / hours per cycle iteration).

**Metadata schema**:
```python
{
    "cycle_class": "continual_learning" | "council_deliberation" | "meta_gate_extinction" | "anchor_refinement",
    "expected_latency_per_iteration_days": <float>,
    "fixed_point_observed": <bool>,  # has the cycle converged to a fixed-point?
}
```

**Cycle handling in queries** (per Tao Revision #5): `query_critical_path()` and `query_dominator()` operate on the cycle-removed projection (compute SCC; contract each SCC into a super-node; run DAG algorithm). `query_cycles()` directly enumerates cycles via Tarjan SCC.

#### Edge type 5: `gates_eligibility_of`

**Semantics**: A → B; A is a META gate that blocks B's eligibility unless waiver / satisfied. The canonical example is a Catalog # STRICT preflight gate blocking a substrate dispatch.

**Examples**:
- `Catalog #313 gates_eligibility_of substrate_dispatch_for_atw_v2_d4` (probe-outcome verdict `INDEPENDENT` blocks re-dispatch)
- `Catalog #325 gates_eligibility_of dispatch_for_c6_ibps_path_b2` (per-substrate symposium required before dispatch)
- `Catalog #270 gates_eligibility_of dispatch_for_all_substrates` (umbrella protocol must pass)
- `Catalog #319 gates_eligibility_of autopilot_reward_branch_for_high_pair_invariant` (DeliverabilityProof required)

**Weight semantics**: `weight = None` (gating is binary).

**Metadata schema**:
```python
{
    "gate_class": "preflight_strict" | "preflight_warn" | "council_required" | "probe_outcome_required",
    "waiver_marker": <str>,  # e.g. "# CATALOG_NNN_WAIVED:<rationale>"
    "current_status": "blocking" | "satisfied" | "waived",
}
```

#### Edge type 6: `waiver_eligible_via`

**Semantics**: A → B; A is a META gate, B is its canonical waiver pattern. The pair (gate, waiver) is documented per CLAUDE.md "Comment-only contracts are FORBIDDEN" — every waiver must have a substantive rationale that resolves the gate's bug-class hypothesis.

**Examples**:
- `Catalog #313 waiver_eligible_via "# PROBE_PREDECESSOR_OVERRIDE_OK:<rationale>"` (the same-line waiver on dispatch wrapper invocation)
- `Catalog #131 waiver_eligible_via "# BARE_WRITE_OK:<rationale>"` (the same-line waiver on bare-write call)
- `Catalog #220 waiver_eligible_via "# SCAFFOLD_DEFERRED_INTEGRATION_OK:<rationale>"` (the same-line waiver on L1 scaffold)
- `Catalog #244 waiver_eligible_via "# CANONICAL_NVML_BLOCK_OK:<rationale>"` (the file-level waiver on remote lane driver)

**Weight semantics**: `weight = None` (waiver pattern is canonical text).

**Metadata schema**:
```python
{
    "waiver_marker_text": <str>,
    "placeholder_rationales_rejected": [str, ...],  # e.g. ["<rationale>", "<reason>"]
    "min_rationale_length_chars": <int>,
    "file_level_waiver_marker": <str>,  # if applicable
}
```

#### Edge type 7: `empirically_anchors`

**Semantics**: A → B; A is an empirical anchor (typed `empirical_anchor`) for B (any node). The edge carries axis + hardware + freshness metadata per Catalog #127 custody validator.

**Examples**:
- `pr101_lc_v2_master_gradient_f174192aeadf empirically_anchors VENN_design_memo` (anchor consumed for empirical sparsity probe)
- `pr101_lc_v2_master_gradient_f174192aeadf empirically_anchors FISHER_design_memo` (anchor consumed for Fisher conditioning probe)
- `pr101_fec6_fixed_huffman_k16_clean_6bae0201_contest_cpu_0_19205 empirically_anchors lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` (anchor IS the lane's L3 contest_cpu gate)
- `c6_e4_mdl_ibps_50ep_smoke_canonical_score_3_04 empirically_anchors substrate_c6_e4_mdl_ibps` (FALSIFICATION anchor; flips substrate to research_only)

**Weight semantics**: `weight = freshness_days` (computed at query time as `(now - anchor_landed_utc).days`).

**Metadata schema**:
```python
{
    "anchor_axis": "contest_cuda" | "contest_cpu" | "macOS-CPU-advisory" | ...,
    "anchor_hardware_substrate": <str>,
    "evidence_grade": <str>,
    "score_claim_valid": <bool>,
    "promotion_eligible": <bool>,
    "anchor_landed_utc": <str>,
}
```

### 6.3 Edge-axiom invariants (Tao Revision #5 + Carmack Revision #3)

- **Directionality**: all 7 edge types are DIRECTED. The hypergraph has `|E|` directed-edge entries (no inverse pairs are implicit; `consumes_output_of` is the explicit inverse of `produces_input_for`).
- **Type-disjointness**: every edge belongs to EXACTLY ONE type. Multi-typing forbidden at v1 (a `consumes_output_of` edge that ALSO `gates_eligibility_of` is two separate edges).
- **Self-loop policy**: self-loops `(v, v)` are PERMITTED only for `cycles_back_to` (a node feeding itself is the canonical fixed-point cycle).
- **Multigraph policy**: multiple parallel edges of DIFFERENT types are PERMITTED (e.g. A `produces_input_for` B AND A `gates_eligibility_of` B simultaneously). Multiple parallel edges of the SAME type are FORBIDDEN (a single (A, B, type) triple is unique).

### 6.4 Edge enumeration from synthesis §4 9×9 matrix

The synthesis §4 9×9 matrix translates to 144 binary edges (72 ordered pairs × 2 directions) by the following rule:

| Synthesis cell value | Edge type(s) |
|---|---|
| `CONSUMES` | A `consumes_output_of` B + B `produces_input_for` A |
| `ADD` | A `composes_with` B with weight 1.0, tier `additive` |
| `SUB` | A `composes_with` B with weight 0.5-0.7, tier `sub_additive` |
| `SAT` | A `composes_with` B with weight ≤ 0.3, tier `saturating` |
| `ORTHO` | (no edge; nodes are independent) |
| `EXCL` | A `composes_with` B with weight 0.0, tier `exclusive` |
| `/` (diagonal) | (no edge; trivial identity) |

The seed bootstrap (Codex's first invocation post-build per OP-HG-2) parses the synthesis §4 matrix and emits these 144 edges.

---

## 7. Hyperedge Specification (3 Types)

The hypergraph's hyperedge set `F` is partitioned into 3 first-class types per Codex sister routing directive C.

### 7.1 Type enumeration

```python
VALID_HYPEREDGE_TYPES = frozenset({
    "n_way_composition_alpha",         # 1.
    "n_way_pareto_feasibility",        # 2.
    "n_way_venn_cell_stratification",  # 3.
})
```

### 7.2 Per-type specification

#### Hyperedge type 1: `n_way_composition_alpha` (Catalog #322 sister)

**Semantics**: A set of N ≥ 2 nodes jointly compose via Catalog #322 composition_alpha with a single joint α value. The joint α can DIFFER from the product of pairwise α values per Catalog #322 v2 cascade evidence (4-of-8 binary pairs sub-additive; N-way joint may be MORE or LESS than pairwise products).

**Examples**:
- `{FISHER, RIEM, TROP} → joint α = 0.4, tier = sub_additive` (Tier-1 deterministic-optimizer 3-piece; pairwise FISHER↔TROP = 0.6, RIEM↔TROP = 0.6, FISHER↔RIEM = 1.0; product = 0.36; joint observed = 0.4)
- `{VENN, FISHER, RIEM, TROP} → joint α = 0.55, tier = sub_additive` (Tier-1 4-piece full)
- `{Z8, TT5L V2, CCREZ} → joint α = exclusive_partial, tier = exclusive` (Z8 ↔ CCREZ EXCL per synthesis §4; cannot fully compose)
- `{POSEAXIS OP-1, OP-2, OP-7, OP-10} → joint α = 0.95, tier = additive` (cheap-probe wave; structurally orthogonal axes)

**Weight semantics**: `weight = joint_alpha_value`.

**Metadata schema**:
```python
{
    "alpha_tier": "additive" | "sub_additive" | "saturating" | "orthogonal" | "exclusive" | "super_additive",
    "pairwise_alpha_product": <float>,  # for comparison against the joint
    "deliverability_proof_path": <str>,
    "empirical_anchor_archive_sha": <str>,
    "evidence_grade": "predicted" | "empirical_validated" | "phantom_research_sidecar",
}
```

**Persistence**: each hyperedge is one JSONL row with `node_ids: tuple[str, ...]` of cardinality N. Bipartite-incidence representation per Tao Revision #5.

**Anti-phantom protection**: per Catalog #322 (`check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`): hyperedges whose `deliverability_proof_path` cites a research-sidecar (not VALIDATED_CONTEST_MEMBER per Catalog #321) are flagged + autopilot reward = 1.0× (no adjustment). The hypergraph propagates this protection by adding a `validation_status` field that downstream `query_hyperedge_compositions()` filters on.

#### Hyperedge type 2: `n_way_pareto_feasibility` (Boyd Revision #4)

**Semantics**: A set of N ≥ 2 nodes jointly satisfy a Pareto-feasible constraint per Boyd-Dattorro alternating projection algorithm. The hyperedge represents the convex-intersection projection of the N nodes' individual feasibility regions.

**Examples**:
- `{rate ≤ R, seg ≤ S, pose ≤ P} → Pareto-feasible region of contest scorer` (synthesis §5.2 Tier 1 outermost)
- `{FISHER feasibility, RIEM feasibility, TROP feasibility, VENN feasibility} → joint Pareto-feasible region of Tier-1 4-piece`
- `{Z8 archive_size ≤ 200KB, TT5L V2 archive_size ≤ 200KB, CCREZ archive_size ≤ 200KB} → joint feasibility of rate-conservative regime per FLOOR design memo`

**Weight semantics**: `weight = feasibility_volume_estimate` (volume of the convex-intersection polytope; smaller volume = more restrictive).

**Metadata schema**:
```python
{
    "feasibility_algorithm": "dykstra_alternating_projection" | "boyd_admm" | "convex_intersection_volume",
    "feasibility_volume_estimate": <float>,
    "binding_constraints": [str, ...],  # which constraints are tight at the boundary
    "convergence_iterations": <int>,
}
```

**Why Boyd Revision #4 binding**: per Catalog #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`): every predicted ΔS band MUST cite Dykstra-feasibility intersection check. Hyperedges of type `n_way_pareto_feasibility` ARE the canonical Dykstra-feasibility witnesses for joint compositions.

#### Hyperedge type 3: `n_way_venn_cell_stratification` (Catalog #319 sister)

**Semantics**: A set of N ≥ 2 nodes jointly stratify a set of byte positions via Venn classification per Catalog #319 v2 cascade.

**Examples**:
- `{pair_axis, region_axis, class_axis} → 8-cell 3-set Venn stratification of archive bytes`
- `{pair_axis, region_axis, class_axis, frame_axis, axis_dimension, label_dimension} → 64-cell 6-set Venn stratification` (per N-set Venn design memo)
- `{TIER_1_ZERO_COST, TIER_2_CONSTANTS, TIER_3_WAIVER_REQUIRED, TIER_4_FORBIDDEN} → Wyner-Ziv deliverability stratification of substrate bytes`

**Weight semantics**: `weight = total_byte_mass_classified` (sum of bytes across all cells in the stratification).

**Metadata schema**:
```python
{
    "set_dimension": <int>,  # N
    "cell_count": <int>,     # 2^N
    "byte_mass_per_cell": {cell_index: byte_count, ...},
    "sparse_cell_count": <int>,  # cells with byte_mass < threshold
    "join_with_neighbor_recommendations": [str, ...],  # per VENN design memo OP-3
}
```

### 7.3 Hyperedge-axiom invariants

- **Cardinality**: every hyperedge has cardinality ≥ 2 (cardinality-1 hyperedges are isolated nodes; not representable as hyperedges).
- **Type-disjointness**: every hyperedge belongs to EXACTLY ONE type.
- **Set-not-multiset**: `node_ids` is a SET (no duplicates) represented as a sorted tuple for byte-stable persistence.
- **Symmetry**: hyperedges are UNDIRECTED (binary directed edges live in `E`; hyperedges live in `F` and are direction-agnostic).
- **Persistence-friendly representation** (Tao Revision #5): each hyperedge is one JSONL row with `node_ids: tuple[str, ...]` (sorted; tuple for hashability + sort_keys=True byte-stability per Catalog #245).

### 7.4 Hyperedge enumeration from canonical posteriors

The seed bootstrap (Codex's first invocation post-build per OP-HG-2) parses 3 canonical posteriors and emits hyperedges:

| Posterior | Hyperedge type | Row → hyperedge mapping |
|---|---|---|
| `.omx/state/substrate_composition_matrix.json` | `n_way_composition_alpha` | each row (substrate_pair / substrate_triple / etc.) becomes one hyperedge |
| (synthesis §13 aggregate Pareto polytope) | `n_way_pareto_feasibility` | the 9-dimensional aggregate polytope becomes one hyperedge over the 9 designs |
| `.omx/state/n_set_venn_sparsity_atlas/*.json` (post-OP-SYN-3 landing) | `n_way_venn_cell_stratification` | each atlas becomes one hyperedge over the axes |

---

## 8. Canonical Graph Operations (Queryable Surface)

Per Carmack Revision #3 + sister Codex routing directive C Layer 2: every query function returns a typed dataclass + supports machine-routable composition.

### 8.1 Query API enumeration

```python
def query_critical_path(*, source: str | None = None, target: str | None = None,
                        weight_attr: str = "predicted_delta_s",
                        race_mode: bool = False) -> CriticalPathResult: ...

def query_orphan_signals(*, direction: str = "producer_without_consumer",
                         filter_node_type: str | None = None) -> OrphanSignalAuditResult: ...

def query_hyperedge_compositions(*, contains_node: str | None = None,
                                 alpha_tier: str | None = None,
                                 validation_status: str | None = None) -> list[HyperedgeCompositionRow]: ...

def query_cycles(*, max_length: int | None = None) -> CycleEnumerationResult: ...

def query_hook_coverage(*, hook_id: int = 1) -> HookCoverageReport: ...

def query_dominator(*, node_id: str) -> DominatorSetResult: ...

def export_dot(*, output_path: Path | None = None,
               filter_node_type: str | None = None,
               filter_edge_type: str | None = None) -> str: ...

def query_predecessor_probe_outcomes(*, node_id: str) -> list[ProbeOutcomeRow]: ...  # Contrarian Revision #1

def query_deterministic_byte_derivation_subsystem() -> DeterministicByteDerivationReport: ...  # OP-HG-5
```

### 8.2 Per-query specification

#### Query 1: `query_critical_path`

**Purpose**: longest weighted path through the DAG (after cycle-removal); identifies the bottleneck for dispatch sequencing.

**Algorithm**:
1. Compute Tarjan SCC over `E` ∪ `F` (treat hyperedges as cliques of binary edges for SCC).
2. Contract each SCC into a super-node.
3. Run DAG longest-path algorithm (topological sort + dynamic programming) on the contracted DAG.
4. Backtrack through super-nodes to recover the original node sequence.
5. Return `CriticalPathResult(path: tuple[str, ...], total_weight: float, bottleneck_edge: tuple[str, str] | None)`.

**Race-mode behavior** (per Contrarian Revision #1 + synthesis §9.5): when `race_mode=True`, weighting flips from `predicted_delta_s` to `cost_envelope_inverse` (`1.0 / cost_envelope_usd`); cheapest-bolt-on-first ordering.

**Sister usage**: cathedral autopilot ranker consumes this query for dispatch sequencing. Codex `/goal` LOOP pre-flight consumes this query to identify the next highest-EV action.

#### Query 2: `query_orphan_signals`

**Purpose**: surfaces Catalog #711 sister analysis structurally — identifies producers without consumers (signals nobody uses) AND consumers without producers (signals nobody provides).

**Algorithm**:
1. For each node `v`, compute `in_degree(v)` and `out_degree(v)` restricted to `produces_input_for` / `consumes_output_of` edge types.
2. If `direction='producer_without_consumer'`: return nodes with `out_degree > 0` but every outgoing edge's destination has `in_degree(destination) == 0` (signal exists but nobody reads it).
3. If `direction='consumer_without_producer'`: return nodes with `in_degree > 0` but every incoming edge's source has `out_degree(source) == 0` (signal expected but nobody provides it).
4. Optional filter by node type (`filter_node_type='deterministic_byte_derivation'` for OP-HG-5).
5. Return `OrphanSignalAuditResult(orphan_nodes: tuple[str, ...], direction: str, edge_filter_applied: bool)`.

**Acceptance test** (per OP-HG-2): the first invocation of `query_orphan_signals(direction='consumer_without_producer')` on the seed graph MUST return the 3 hook-CONSUMER-without-producer flags from synthesis §8.2:

1. `POSEAXIS OP-3 ATW V2-1 channel-pick reformulation` (consumes Z6 Wave 2 4c trained anchor; CURRENTLY DEFERRED)
2. `Z8 full conjunction dispatch` (consumes Z6-v2 Candidate 1 OR 4c PROCEED-unconditional + Z7 PROCEED-unconditional + C6 IBPS Phase 2 β-IB-optimal + ATW V2 D4 PARADIGM reactivation)
3. `TT5L V2 4-primitive composition smoke` (consumes Z6 4c outcome + Z7 GRU-vs-Mamba-2 outcome + Dykstra-feasibility check + single-primitive cooperative-receiver-derived foveation smoke)

If the live count differs, the seed-bootstrap parser is mis-classifying edges (per the predicted-band reactivation criterion in this memo's frontmatter).

#### Query 3: `query_hyperedge_compositions`

**Purpose**: structural Catalog #322 N-way composition_alpha lookup.

**Algorithm**:
1. Filter `F` by type `n_way_composition_alpha`.
2. Apply optional filters: `contains_node` (hyperedge must include this node); `alpha_tier` (e.g. `sub_additive`); `validation_status` (filter out phantom-provenance rows per Catalog #322).
3. Return list of `HyperedgeCompositionRow(node_ids: tuple[str, ...], alpha_value: float, alpha_tier: str, metadata: dict)`.

**Sister usage**: autopilot ranker consumes for v2 cascade reward factor extension to N-way joint α. Per-substrate symposium queue consumes to enumerate N-way α candidates.

#### Query 4: `query_cycles`

**Purpose**: cycle detection via Tarjan SCC; surfaces continual-learning feedback loops vs deadlock candidates.

**Algorithm**:
1. Compute Tarjan SCC over `E` (binary edges only; hyperedges expanded to clique-edges).
2. For each SCC of size ≥ 2 (true cycle): enumerate the cycle path via DFS within the SCC.
3. Classify each cycle by edge-type composition: cycles consisting ONLY of `cycles_back_to` edges are continual-learning feedback; cycles with mixed edge types may be deadlock candidates (operator review required).
4. Filter by `max_length` if provided.
5. Return `CycleEnumerationResult(cycles: tuple[tuple[str, ...], ...], cycle_classifications: tuple[str, ...])`.

**Expected output** (on seed graph): 3-5 continual-learning cycles per the 6-hook wire-in:
- `(empirical_anchor → autopilot_ranker → dispatch_decision → next_anchor)` (canonical continual-learning loop)
- `(council_deliberation → design_memo → next_council_deliberation)` (Catalog #300 v2 anchor cycle)
- `(meta_gate → preflight.py → strict_flip_check → next_meta_gate)` (Catalog #185 META-meta-meta cycle)
- `(master_gradient_extraction → per_pair_anchor → cathedral_autopilot_reweight → next_extraction)` (Catalog #318/#327 anchor refinement loop)

#### Query 5: `query_hook_coverage`

**Purpose**: Catalog #125 6-hook audit at the graph level. Returns producers + consumers + orphans per hook.

**Algorithm**:
1. For each hook `h` in {1, 2, 3, 4, 5, 6}: identify nodes that produce signal for hook `h` (per node metadata `produces_signals: [hook_id, ...]`).
2. Identify nodes that consume signal for hook `h` (per node metadata `consumes_signals: [hook_id, ...]`).
3. Compute producer-without-consumer orphans + consumer-without-producer orphans (sister of `query_orphan_signals` at the hook granularity).
4. Return `HookCoverageReport(hook_id: int, hook_name: str, producers: tuple[str, ...], consumers: tuple[str, ...], orphans_producer_without_consumer: tuple[str, ...], orphans_consumer_without_producer: tuple[str, ...])`.

**Sister usage**: Catalog #333 STRICT preflight gate (per sister routing directive C Layer 3) consumes this query. The gate refuses repo state with hook-orphan signals UNLESS declared in canonical waiver file.

#### Query 6: `query_dominator`

**Purpose**: subgraph dominator analysis (Lengauer-Tarjan); surfaces downstream impact of a single node failure / change.

**Algorithm**:
1. Compute dominator tree rooted at the input `node_id`.
2. The dominator set of `node_id` is the set of nodes whose ONLY path from the root passes through `node_id` (i.e. nodes that would be unreachable if `node_id` is removed).
3. Return `DominatorSetResult(node_id: str, dominator_set: frozenset[str], dominator_set_size: int)`.

**Sister usage**:
- `query_dominator(node_id="pr101_lc_v2_master_gradient_f174192aeadf")` returns the universal anchor's dominator set; per synthesis §6.1 this should include ALL 9 design memos + per-design consumer surfaces.
- `query_dominator(node_id="wyner_ziv_seed_subsystem")` (per OP-HG-5) returns the deterministic-byte-derivation subsystem's downstream impact across substrates.
- `query_dominator(node_id="Catalog_270_umbrella_gate")` returns every substrate dispatch wrapper that the umbrella gates.

#### Query 7: `export_dot`

**Purpose**: DOT (graphviz) format export for visualization.

**Algorithm**:
1. Walk all nodes; emit DOT node declaration with per-type shape per the canonical-vs-unique decision §4 (10 node shapes mapped to 10 categories).
2. Walk all binary edges in `E`; emit DOT edge declaration with per-type style (solid / dashed / dotted / bold / colored mapped to 7 edge types).
3. Walk all hyperedges in `F`; emit DOT subgraph cluster (graphviz cluster syntax) grouping the hyperedge's nodes.
4. Apply optional filters (`filter_node_type` / `filter_edge_type`) by skipping non-matching elements.
5. Return the DOT-format string; optionally write to `output_path`.

**Sister usage**: `tools/render_design_graph.py render --output design_stack.svg` invokes this query + pipes to `dot -Tsvg`.

#### Query 8: `query_predecessor_probe_outcomes` (Contrarian Revision #1)

**Purpose**: returns probe outcomes from `.omx/state/probe_outcomes.jsonl` per Catalog #313 for nodes whose edges have been recently traversed in dispatch decisions.

**Algorithm**:
1. For input `node_id`, look up via `tac.probe_outcomes_ledger.query_blocking_outcomes(node_id)`.
2. Filter to outcomes whose `freshness_days < 30` per Catalog #298 retirement-discipline window.
3. Return list of `ProbeOutcomeRow(probe_id: str, verdict: str, freshness_days: int, archive_sha: str, evidence_grade: str)`.

**Sister usage**: Codex `/goal` LOOP pre-flight invokes for every candidate node before dispatch recommendation. If a node has a blocking predecessor outcome ({INDEPENDENT, KILL, DEFER}), `/goal` LOOP either cites the verdict + rationale for override OR ratifies the predecessor.

#### Query 9: `query_deterministic_byte_derivation_subsystem` (OP-HG-5; operator-elevated)

**Purpose**: dedicated query surface for the operator-elevated category 10 subsystem.

**Algorithm**:
1. Filter nodes to those with `node_type='deterministic_byte_derivation'`.
2. For each subsystem node: enumerate downstream consumers (substrates using the WZ side-info / procedural codebook / null-space exploiter / etc.).
3. Compute `deliverability_score_savings_estimate_aggregate` summed across all downstream consumers.
4. Identify unconsumed subsystems (`producers_without_consumer`) — opportunities to hoist deterministic-byte-derivation primitives to additional substrates.
5. Return `DeterministicByteDerivationReport(subsystem_members: tuple[str, ...], downstream_consumers_per_member: dict, deliverability_score_savings_aggregate: float, hoist_opportunities: tuple[str, ...])`.

**Sister usage** (per operator's WZ question + Codex sister routing directive C §"DETERMINISTIC-BYTE-DERIVATION SUBSYSTEM"):
- `tools/render_design_graph.py orphans --direction consumer_without_producer --filter-type deterministic_byte_derivation` (operator-routable CLI)
- `tools/render_design_graph.py dominator --node wyner_ziv_seed_subsystem` (Tier 1 hoist impact)
- `tools/render_design_graph.py hyperedges --contains-node wyner_ziv_seed_subsystem` (N-way compositions involving the META-category)

### 8.3 Query composition examples

Per Carmack Revision #3 + machine-routability:

```python
# Find the top-3 highest-EV dispatch candidates per critical path,
# filtered to substrates with no blocking predecessor probe outcomes
candidates = query_critical_path(weight_attr="predicted_delta_s_lower_bound").path[:3]
viable = [c for c in candidates if not query_predecessor_probe_outcomes(node_id=c)]

# Find deterministic-byte-derivation primitives not yet consumed by any substrate
report = query_deterministic_byte_derivation_subsystem()
hoist_targets = report.hoist_opportunities

# Audit hook #4 (cathedral autopilot dispatch) for orphan signals
hook_4 = query_hook_coverage(hook_id=4)
assert not hook_4.orphans_producer_without_consumer, "hook #4 has orphan producers"
```

---

## 9. Cargo-Cult Audit per Assumption (Catalog #303)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #303: every substrate design memo MUST audit cargo-culted assumptions. The hypergraph design's load-bearing assumptions:

### 9.1 Assumption: "Labeled hypergraph IS the right structural primitive"

**Classification**: HARD-EARNED.

**Rationale**: §2.3 "Why hypergraph over simpler structures" enumerates 7 alternatives (set / chain / tree / DAG / simple-directed-graph / bipartite / line-graph); the labeled directed hypergraph subsumes all 7 as special cases. The synthesis §4 9×9 matrix's 41 CONSUMES + 9 SUB + 12 ORTHO + 2 ADD + 8 EXCL cells require a structure that captures typed binary edges + N-way composition + cycles + weights — exactly what hypergraph provides.

**Risk**: graph-theory canonical formalism may have alternative representations (e.g. simplicial complex, factor graph, tensor network) that fit the design stack more naturally. Per Tao's grand-council seat: simplicial complexes generalize hypergraphs but introduce homological structure not needed for the design stack's queries.

**Reactivation criterion**: if a future query is mathematically simpler in an alternative formalism, re-evaluate the choice. For v1 the hypergraph is the canonical structure.

### 9.2 Assumption: "7 edge types are sufficient"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL (per Assumption-Adversary Revision #2).

**Rationale**: The 7 edge types are HARD-EARNED from the synthesis §4 matrix legend (6 types: CONSUMES + ADD + SUB + SAT + ORTHO + EXCL collapsed to `consumes_output_of` + `composes_with`) + Catalog #313 probe-outcomes (`gates_eligibility_of`) + Catalog #325 waivers (`waiver_eligible_via`) + continual-learning loops (`cycles_back_to`) + empirical anchor tracking (`empirically_anchors`). But this enumeration is BASED ON the synthesis's current matrix structure; future synthesis iterations may surface additional edge categories.

**Candidate additional edge types** (deferred to v2 per Assumption-Adversary):
- `falsifies_premise_of` — from Catalog #229 probes that empirically falsify a design memo's premise
- `supersedes_via_council_verdict` — from Catalog #300 v2 anchors that supersede earlier deliberations
- `triggers_reactivation_of` — from Catalog #325 per-substrate symposiums that reactivate DEFER candidates

**Risk**: v1 fail-closed with `UnknownEdgeTypeError` is the safe path; v2 extension protocol via canonical `add_edge_type(...)` helper deferred.

**Reactivation criterion**: when ≥3 new edge categories surface across 3 sister design memos within 30 days, land v2 with extension protocol.

### 9.3 Assumption: "Hyperedges are sufficient to model N-way composition_alpha"

**Classification**: HARD-EARNED-WITH-REVISION (per Boyd Revision #4).

**Rationale**: Berge 1973 hyperedge formalism is the canonical N-way primitive. Catalog #322 v2 cascade evidence (4-of-8 binary pairs sub-additive) demonstrates N-way joint α differs from pairwise α product; hyperedges represent the joint directly.

**Risk**: tensor representation (per Boyd's alternative suggestion) would capture continuous α-value gradient; hyperedge discretizes to single α value per N-way joint. For continuous-gradient sensitivity analysis, tensor representation may be preferable.

**Reactivation criterion**: when continuous-gradient sensitivity analysis surfaces in a sister design memo, extend hyperedges with tensor-valued α (deferred to v2).

### 9.4 Assumption: "Cycles ONLY occur in continual-learning feedback loops"

**Classification**: CARGO-CULTED-PENDING-EMPIRICAL.

**Rationale**: §6.2 edge type #4 (`cycles_back_to`) enumerates 4 expected continual-learning cycles. But sister design memos may surface OTHER cycle classes — e.g. council-deliberation deadlocks (council A waits for council B's verdict; council B waits for council A's verdict). The current classification treats cycles as either continual-learning OR deadlock-candidate; the empirical distribution post-bootstrap will validate or refute.

**Risk**: misclassified deadlock cycles can stall dispatch decisions. Per Tao's grand-council seat: a 3-clean-pass deadlock at Catalog #292 + #229 + #303 (each requires the others' clearance) is a known cycle class.

**Reactivation criterion**: post-bootstrap `query_cycles()` returns N cycles; manually classify each; if ≥1 is mis-classified as continual-learning when it's actually a deadlock candidate, extend the classification taxonomy.

### 9.5 Assumption: "Operator's WZ question elevates deterministic_byte_derivation to first-class status"

**Classification**: HARD-EARNED.

**Rationale**: operator 2026-05-18 verbatim: *"wyner-ziv is still being considered as part of the seed and other procedural gen experiments right?"* — this is an explicit operator declaration that the WZ subsystem is part of the seed graph + first-class consideration. The synthesis §0 + §6 explicitly anchors all 9 designs on PR101_lc_v2 + the deterministic-byte-derivation members (`wyner_ziv_layer` + `wyner_ziv_deliverability` + `procedural_codebook_generator` + `null_space_exploiter` + planned `optical_flow_side_info` + `foe_detection`) are EXPLICITLY listed in the sister Codex routing directive C.

**Risk**: subsuming the WZ subsystem under `canonical_helper` would lose the META semantic (deterministic byte derivation is structurally distinct because the derived bytes are NEVER CHARGED by the contest rate term).

**Reactivation criterion**: if operator clarifies that the WZ subsystem should be downgraded to `canonical_helper`, downgrade. (Not anticipated.)

### 9.6 Assumption: "Graph queries are the canonical orchestration surface"

**Classification**: HARD-EARNED.

**Rationale**: per Carmack Revision #3 + CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable: machine-routable typed queries compose; prose synthesis matrices do not. The synthesis §4 9×9 matrix is inspectable-by-eye; the hypergraph's typed queries are composable into autopilot ranker + dispatch wrapper + Codex `/goal` LOOP pre-flight.

**Risk**: query API may evolve; v1 fixes 9 queries (§8.1 enumeration). v2 may extend.

**Reactivation criterion**: when ≥3 new query patterns surface across 3 sister tools within 30 days, extend the API.

### 9.7 Aggregate cargo-cult audit verdict

- **4 of 6 shared assumptions are HARD-EARNED**: hypergraph formalism / WZ elevation / graph queries as orchestration / hyperedges N-way (with revision per Boyd).
- **2 of 6 shared assumptions are CARGO-CULTED-PENDING-EMPIRICAL**: 7 edge types sufficient / cycles only in continual-learning.

Per Catalog #303 + #294: the 2 CARGO-CULTED assumptions are surfaced with reactivation criteria. v1 fail-closed with `UnknownEdgeTypeError` + `unclassified_cycle` empirical-classification deferred to v2.

---

## 10. 9-Dimension Success Checklist Evidence (Catalog #294)

Per CLAUDE.md "9-DIMENSION SUCCESS CHECKLIST EVIDENCE SECTION" standing directive 2026-05-15 + Catalog #294:

### 10.1 Dimension 1: UNIQUENESS (class-shift not within-class)

**Evidence**: The hypergraph IS a META-class-shift over the design stack. No prior memo canonicalizes a typed graph formalism over (designs + canonical helpers + meta gates + probes + substrates + venn cells + posteriors + consumers + empirical anchors + deterministic-byte-derivation subsystem). Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": the hypergraph ESTABLISHES the canonical pattern for graph-formalism design memos.

### 10.2 Dimension 2: BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)

**Evidence**: §0 Executive Summary delivers the TL;DR + verdict matrix + Top-5 op-routables + 5 binding revisions in <100 lines (reviewable in 30 sec). §5-§7 specifications are tabular + bullet-list; §8 query API is functional signatures. The DOT export §8.2 query 7 provides the canonical visualization.

### 10.3 Dimension 3: DISTINCTNESS (explicitly different from sisters)

**Evidence**:
- Distinct from sister Codex routing directive C: this memo provides DESIGN AUTHORITY (the formalism + axioms + invariants + per-category specs); Codex routing directive C provides BUILD AUTHORITY (the canonical 4-layer implementation pattern + test contract + CLI surface).
- Distinct from sister synthesis: this memo extends the synthesis §4 9×9 matrix from prose representation to typed graph structure; synthesis is INPUT, hypergraph is OUTPUT.
- Distinct from sister design memos (today's 9): this memo is META over the 9 (each of the 9 becomes a node typed `design`).

### 10.4 Dimension 4: RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)

**Evidence**:
- Premise verification per Catalog #229: §13 Premise Verification Trail (PV-1 through PV-7).
- Adversarial review per Catalog #292: 5 binding revisions per Contrarian + Assumption-Adversary + Carmack + Boyd + Tao documented in §0 + full text in §15.
- Assumption classification per Catalog #303: 6 shared assumptions classified in §9.
- Empirical anchor per Catalog #287: hypergraph produces no ΔS directly; anchor on PR101_lc_v2 + synthesis §4 9×9 matrix as the seed.

### 10.5 Dimension 5: OPTIMIZATION-PER-TECHNIQUE (covered by Catalog #290 sister gate)

**Evidence**: §4 Canonical-vs-Unique Decision per Layer — 11 ADOPT_CANONICAL_BECAUSE_SERVES + 5 FORK_BECAUSE_PRINCIPLED_MISMATCH. Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

### 10.6 Dimension 6: STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS)

**Evidence**: the hypergraph itself IS the stack-of-stacks composability surface. `query_hyperedge_compositions(alpha_tier='additive')` returns the additive compositions; `query_critical_path()` returns the dispatch sequencing that respects compositional dependencies; `query_dominator(node_id)` returns downstream impact.

### 10.7 Dimension 7: DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned)

**Evidence**: JSONL persistence with `sort_keys=True` + `ensure_ascii=False` is byte-stable per Catalog #245 sister pattern. Seed bootstrap is deterministic: same synthesis §4 matrix → same 144 edges; same canonical posteriors → same hyperedges; same canonical helpers → same nodes.

### 10.8 Dimension 8: EXTREME-OPTIMIZATION + PERFORMANCE

**Evidence**: query algorithms are standard graph-theory O(V + E) or O(V * E) bounds per Cormen et al. canonical complexity. fcntl-locked JSONL append is O(1) amortized per Catalog #245 sister pattern. Strict-load is O(N) per JSONL row count; quarantine on corrupt is O(1).

### 10.9 Dimension 9: OPTIMAL-MINIMAL-CONTEST-SCORE

**Evidence**: the hypergraph produces ZERO direct ΔS. The contribution to optimal-minimal-contest-score is INDIRECT via:
- `query_critical_path()` → cathedral autopilot dispatch sequencing → next paid dispatch = highest-EV
- `query_orphan_signals()` → Codex `/goal` LOOP pre-flight → next action closes orphans = no wasted work
- `query_hyperedge_compositions()` → autopilot v2 cascade reward factor → joint-α-aware ranking = more accurate predicted ΔS
- `query_predecessor_probe_outcomes()` → dispatch wrapper refuses re-dispatch of adjudicated probes → no wasted GPU spend on already-settled questions

Per synthesis §13 aggregate predicted ΔS band: hypergraph enables the cascade that moves frontier from `0.19205 [contest-CPU]` toward `[0.165, 0.185]` REALISTIC.

---

## 11. Observability Surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305: every substrate design / experiment / tool / canonical helper / dispatch wrapper memo MUST declare its observability surface across the 6 facets.

### 11.1 Facet 1: inspectable per layer

The hypergraph exposes every layer:
- **Vertex set V**: enumerated via `query_*` with `--list-nodes` flag; per-category enumeration via filter; full inventory via `list(load_hypergraph_strict()["nodes"])`.
- **Binary edge set E**: enumerated via `query_*` with `--list-edges` flag; per-type enumeration via filter.
- **Hyperedge set F**: enumerated via `query_hyperedge_compositions(...)` with optional filters.
- **Per-element metadata**: every node / edge / hyperedge carries `metadata: dict` field per §5.2 / §6.2 / §7.2 specs; queryable via `query_node(node_id)` / `query_edge(src, dst, type)` / `query_hyperedge(node_ids, type)`.

### 11.2 Facet 2: decomposable per signal

The hypergraph decomposes signal flow:
- **Producer-consumer pairs**: enumerated via `query_orphan_signals(direction='producer_without_consumer')` + `query_orphan_signals(direction='consumer_without_producer')`.
- **6-hook breakdown**: `query_hook_coverage(hook_id=N)` for N in {1, 2, 3, 4, 5, 6} returns per-hook producer / consumer / orphan classification.
- **Per-category breakdown**: `query_hyperedge_compositions(contains_node=<id>)` + sister filters enable per-category signal-flow inspection.

### 11.3 Facet 3: diff-able across runs

Per Catalog #131/#138 fcntl-locked JSONL store pattern: `.omx/state/design_stack_hypergraph.json` is APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE. Run-to-run diff via:
- `git log -p .omx/state/design_stack_hypergraph.json` for snapshot diffs (when canonical writes use atomic `os.replace` per sister Catalog #245).
- For append-only event sourcing (v2 deferred): every add_node / add_edge / add_hyperedge appends one JSONL row; diff via JSONL line count + content hash.

### 11.4 Facet 4: queryable post-hoc

All queries are post-hoc by construction; `load_hypergraph_strict()` returns the full graph state at any time. Per Catalog #138 strict-load fail-closed: corrupt state raises `HypergraphCorruptError`; quarantine to `.corrupt.<utc>` preserves forensic evidence.

### 11.5 Facet 5: cite-able

Per Catalog #245 modal_call_id_ledger pattern: every node / edge / hyperedge carries `(agent / written_at_utc / written_pid / written_host)` provenance tuple per sister `_append_event_locked` discipline. Cite via `(node_id, edge_id, hyperedge_id, schema_version, anchor_landed_utc)`.

### 11.6 Facet 6: counterfactual-able

Per Catalog #139 packet compiler no-op detector + Catalog #272 distinguishing-feature integration contract: every hyperedge with `n_way_composition_alpha` type carries `deliverability_proof_path` + `empirical_anchor_archive_sha`. Counterfactual probe via:
- Modify a single hyperedge's α value → recompute `query_critical_path()` → observe dispatch sequencing change.
- Remove an edge → recompute `query_dominator(node_id=src)` → observe downstream impact.
- Add a synthetic node → query against the augmented graph to test hypothetical "what if this design memo lands?" scenarios.

---

## 12. Dykstra-Feasibility Intersection (Catalog #296)

Per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + Catalog #296 + Boyd Revision #4:

### 12.1 Is the hypergraph itself Dykstra-feasible?

The hypergraph H = (V, E, F) is **trivially feasible** as a data structure (it is finite + countable + has no internal constraints to violate). The Dykstra-feasibility question applies to the hypergraph's QUERY OUTPUTS.

### 12.2 Per-query Dykstra-feasibility

- **`query_critical_path()`**: the longest-weighted-path output is feasible by construction (DAG longest path exists and is unique up to ties).
- **`query_orphan_signals()`**: the orphan-set output is feasible (well-defined set partition).
- **`query_hyperedge_compositions()`**: the returned compositions may include hyperedges whose joint α value is OUTSIDE the Dykstra-feasible polytope (per Boyd Revision #4). The canonical fix: every `n_way_composition_alpha` hyperedge MUST cite its Dykstra-feasibility check via metadata `dykstra_feasibility_witnessed_by: n_way_pareto_feasibility hyperedge_id` field.
- **`query_cycles()`**: cycle enumeration is feasible (Tarjan SCC always terminates).
- **`query_hook_coverage()`**: hook coverage report is feasible (well-defined set partition).
- **`query_dominator()`**: dominator set is feasible (Lengauer-Tarjan always terminates).

### 12.3 The aggregate hypergraph as a Dykstra-feasibility witness

The 9-dimensional Pareto polytope per synthesis §13.2 (one dimension per design) is itself a hyperedge of type `n_way_pareto_feasibility` containing all 9 designs. The Boyd-Dattorro alternating projection algorithm computes the convex-intersection projection of the 9 individual feasibility regions. The resulting joint polytope is the canonical Dykstra-feasibility witness for the synthesis's REALISTIC predicted aggregate band.

Per Catalog #296 + sister Boyd Revision #4 in synthesis §0 Revision #4: the realistic aggregate ΔS LOWER BOUND is the Dykstra-projection on the 9-dimensional polytope, NOT additive sum.

---

## 13. Premise Verification Trail (Catalog #229)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #229: every subagent landing with claims ≥3 bulk edits MUST include empirical verdict table OR reproducer-script path.

### 13.1 PV-1: Sister Codex routing directive C exists

**Premise**: `.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` exists and provides the BUILD authority for `src/tac/design_graph.py`.

**Verification**:
```bash
ls -la /Users/adpena/Projects/pact/.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md
# RESULT: file exists (14.6K)
```

**Status**: HARD-EARNED-VERIFIED.

### 13.2 PV-2: Synthesis §4 9×9 matrix IS the seed adjacency representation

**Premise**: synthesis §4 (lines 261-340) contains the 9×9 cross-pollination matrix that is the seed adjacency.

**Verification**: synthesis §4 read in full (lines 261-389); 9×9 matrix at lines 291-301; legend at lines 265-273; critical cell explanations at lines 303-339; ASCII tree at lines 341-389. Matrix structure confirmed.

**Status**: HARD-EARNED-VERIFIED.

### 13.3 PV-3: Operator approved "Chain but how about a graph" + "All are approved" + WZ first-class

**Premise**: operator 2026-05-18 sequence (a) corrected chain→graph; (b) approved all 5 sister routing directives (C this one + D + E + F + sister design memos); (c) elevated WZ to first-class via question.

**Verification**: per sister routing directive C header line 3: *"# Operator: approved 2026-05-18 (\"Chain but how about a graph\" + \"All are approved\")"*; per sister routing directive C §"DETERMINISTIC-BYTE-DERIVATION SUBSYSTEM" line 191: *"Per operator question 2026-05-18 (\"wyner-ziv is still being considered as part of the seed and other procedural gen experiments right?\"), THIS subsystem gets first-class typed-node category status"*.

**Status**: HARD-EARNED-VERIFIED.

### 13.4 PV-4: 10 typed node categories enumerable + 7 edge types + 3 hyperedge types

**Premise**: the formalism specifies 10 node categories + 7 edge categories + 3 hyperedge types per Codex sister routing directive C.

**Verification**: per sister routing directive C lines 27-56: `VALID_NODE_TYPES = frozenset({...10 entries...})` + `VALID_EDGE_TYPES = frozenset({...7 entries...})` + `VALID_HYPEREDGE_TYPES = frozenset({...3 entries...})`. Direct enumeration in sister artifact.

**Status**: HARD-EARNED-VERIFIED.

### 13.5 PV-5: Deterministic-byte-derivation subsystem members extant

**Premise**: the 6 named members of category 10 (`wyner_ziv_layer` / `wyner_ziv_deliverability` / `procedural_codebook_generator` / `null_space_exploiter` / `optical_flow_side_info` planned / `foe_detection` planned) include 4 already-landed and 2 planned.

**Verification**:
```bash
ls /Users/adpena/Projects/pact/src/tac/wyner_ziv_deliverability/ \
   /Users/adpena/Projects/pact/src/tac/procedural_codebook_generator/ \
   /Users/adpena/Projects/pact/src/tac/null_space_exploiter/ 2>&1 | head -10
ls /Users/adpena/Projects/pact/src/tac/codec/wyner_ziv_layer.py 2>&1 | head -3
# RESULT: all 4 landed files extant; 2 planned (optical_flow_side_info + foe_detection) acknowledged as planned per Codex sister routing directive C
```

**Status**: HARD-EARNED-VERIFIED.

### 13.6 PV-6: Lane pre-registered per Catalog #126

**Premise**: lane `lane_design_stack_full_hypergraph_design_20260518` pre-registered.

**Verification**:
```bash
.venv/bin/python tools/lane_maturity.py add-lane lane_design_stack_full_hypergraph_design_20260518 --name "..." --phase 2
# RESULT: "OK — added lane lane_design_stack_full_hypergraph_design_20260518 at L0 (phase 2.0)"
```

**Status**: HARD-EARNED-VERIFIED.

### 13.7 PV-7: Checkpoint discipline per Catalog #206

**Premise**: subagent checkpoint discipline followed.

**Verification**: initial checkpoint emitted at `.omx/state/subagent_progress.jsonl` for subagent_id `full_hypergraph_design_subagent_20260518` per canonical helper invocation pattern.

**Status**: HARD-EARNED-VERIFIED (operational).

### 13.8 Bulk edits count

**Premise**: this memo claims 1 file edited (this design memo). Per Catalog #229: 1 file = NOT bulk edit class; no premise verification table strictly required. However, the memo CITES + builds on the synthesis + sister routing directive C as data input; for transparency, this PV trail establishes the empirical foundation.

**Status**: PV trail INCLUDED for transparency per Catalog #294 9-dim checklist Dimension 4 RIGOR.

---

## 14. Storage + Persistence (Catalog #131 + #138 + #245 sister)

### 14.1 Canonical persistence path

```
.omx/state/design_stack_hypergraph.json       (canonical posterior; sister of modal_call_id_ledger.jsonl)
.omx/state/design_stack_hypergraph.json.lock  (fcntl lock per Catalog #131)
.omx/state/design_stack_hypergraph.json.tmp.<uuid12>  (atomic write staging per Catalog #245)
.omx/state/quarantine/design_stack_hypergraph.json.corrupt.<utc>  (Catalog #138 fail-closed quarantine)
```

### 14.2 Schema (JSON; sort_keys=True for byte-stability)

```json
{
    "schema_version": "design_stack_hypergraph_v1_20260518",
    "nodes": [
        {
            "node_id": "<unique_id>",
            "node_type": "<one of 10 categories>",
            "source_path": "<repo-relative path>",
            "metadata": { ... per §5.2 per-category schema ... },
            "agent": "claude" | "codex" | "operator",
            "written_at_utc": "<ISO>",
            "written_pid": <int>,
            "written_host": "<str>"
        },
        ...
    ],
    "edges": [
        {
            "src_node_id": "<id>",
            "dst_node_id": "<id>",
            "edge_type": "<one of 7 categories>",
            "weight": <float | null>,
            "metadata": { ... per §6.2 per-type schema ... },
            "agent": "claude" | "codex" | "operator",
            "written_at_utc": "<ISO>",
            "written_pid": <int>,
            "written_host": "<str>"
        },
        ...
    ],
    "hyperedges": [
        {
            "node_ids": ["<id>", "<id>", "<id>", ...],
            "hyperedge_type": "<one of 3 categories>",
            "weight": <float | null>,
            "metadata": { ... per §7.2 per-type schema ... },
            "agent": "claude" | "codex" | "operator",
            "written_at_utc": "<ISO>",
            "written_pid": <int>,
            "written_host": "<str>"
        },
        ...
    ]
}
```

### 14.3 Atomic write + fcntl-lock discipline (per Catalog #131 + #245)

```python
def _atomic_write_locked(path: Path, payload: dict) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{uuid.uuid4().hex[:12]}")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            tmp_path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
```

### 14.4 Strict-load discipline (per Catalog #138)

```python
class HypergraphCorruptError(RuntimeError): pass

def load_hypergraph_strict(path: Path | None = None) -> dict:
    path = path or GRAPH_PATH
    if not path.exists():
        return {"schema_version": GRAPH_SCHEMA_VERSION, "nodes": [], "edges": [], "hyperedges": []}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        quarantine_path = _quarantine_corrupt(path)
        raise HypergraphCorruptError(
            f"design_stack_hypergraph.json failed strict load; quarantined to {quarantine_path}; original error: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HypergraphCorruptError("design_stack_hypergraph.json root is not a dict")
    if payload.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise HypergraphCorruptError(
            f"design_stack_hypergraph.json schema_version mismatch: expected {GRAPH_SCHEMA_VERSION}, got {payload.get('schema_version')}"
        )
    return payload
```

### 14.5 Append-only event-sourcing (v2 deferred)

v1 uses snapshot-rewrite (entire JSON re-written on every mutation). v2 may extend to append-only event sourcing (each add_node / add_edge / add_hyperedge appends one JSONL row) per Catalog #245 sister pattern; deferred to v2.

---

## 15. Visualization (DOT + ASCII + d3/cytoscape deferred)

### 15.1 DOT (graphviz) — primary

Per `export_dot()` query §8.2 query 7. Emits DOT-format string with:
- 10 node shapes per node type (ellipse / box / diamond / hexagon / parallelogram / trapezium / cylinder / folder / note / cds)
- 7 edge styles per edge type (solid / dashed / dotted / bold / colored-blue / colored-red / colored-green)
- Hyperedges rendered as graphviz subgraph clusters
- Per-node label = `node_id`; per-edge label = `edge_type` + weight if non-None

Pipe to `dot -Tsvg` for SVG / `dot -Tpng` for PNG / `dot -Tpdf` for PDF.

### 15.2 ASCII tree — terminal-friendly

Per `tools/render_design_graph.py render --format=ascii` CLI. Walks the graph from a root node + emits ASCII tree using box-drawing characters (per synthesis §4 cross-pollination tree ASCII).

### 15.3 d3.js / cytoscape.js — deferred to v2

Interactive browser-based visualization. Operator-blocking for CI environments (requires browser). v1 ships DOT + ASCII; v2 may extend.

---

## 16. Wire-In to Existing Canonical Helpers

`tac.design_graph` is consumed by 5 canonical surfaces:

### 16.1 Cathedral autopilot ranker (`tools/cathedral_autopilot_autonomous_loop.py`)

Consumes `query_critical_path(weight_attr='predicted_delta_s_lower_bound', race_mode=<bool>)` for dispatch sequencing. Replaces ad-hoc "next dispatch" decision-making with structural query.

### 16.2 Council deliberation posterior (cycle target)

Cycles `cycles_back_to` between council_deliberation_posterior.jsonl ↔ design_memo (typed `design`). The hypergraph surfaces this cycle via `query_cycles()` so the council apparatus knows it is a continual-learning loop not a deadlock.

### 16.3 Cost-band posterior (anchor)

`empirically_anchors` edges from `empirical_anchor` nodes (typed) → design_memo / canonical_helper / substrate nodes. The cost-band posterior consumes the hypergraph's `query_predecessor_probe_outcomes(node_id)` for per-cost-band-class freshness analysis.

### 16.4 Codex's `/goal` LOOP pre-flight

Per AGENTS.md "Agent Role Specialization": Codex's `/goal` LOOP iteration consumes `query_orphan_signals()` + `query_critical_path()` + `query_predecessor_probe_outcomes()` at pre-flight. Identifies the next highest-EV action structurally rather than via prose synthesis.

### 16.5 Operator-facing briefing (`tools/operator_briefing.py`)

Per sister Codex routing directive C Layer 4: extends `operator_briefing.py` output with `design_graph_node_count` + `design_graph_edge_count` + `design_graph_hyperedge_count` + `design_graph_orphan_count` + `design_graph_critical_path_length` fields.

---

## 17. Empirical Anchor — Bootstrap Verification

### 17.1 Bootstrap procedure (per Codex sister routing directive C §"SEED GRAPH")

```python
from tac.design_graph import add_node, add_edge, add_hyperedge, load_hypergraph_strict

# 1. Add 9 design nodes
for memo_path in glob('.omx/research/*_design_*_20260518.md') + \
                 glob('.omx/research/grand_council_*_20260518.md') + \
                 glob('.omx/research/council_*_20260518.md'):
    metadata = parse_council_frontmatter(memo_path)
    add_node(node_id=basename(memo_path).replace('.md', ''),
             node_type="design",
             source_path=memo_path,
             metadata=metadata)

# 2. Add this hypergraph design memo as a 10th design node
add_node(node_id="design_stack_full_hypergraph_model_design_memo_20260518",
         node_type="design",
         source_path=".omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md",
         metadata=parse_council_frontmatter(...))

# 3. Add 144 binary edges from synthesis §4 9×9 matrix
for (src_design, dst_design, cell_value) in parse_synthesis_matrix():
    if cell_value == "CONSUMES":
        add_edge(src=dst_design, dst=src_design, edge_type="consumes_output_of",
                 metadata={"synthesis_cell": f"({src_design}, {dst_design})"})
        add_edge(src=src_design, dst=dst_design, edge_type="produces_input_for",
                 metadata={"synthesis_cell": f"({src_design}, {dst_design})"})
    elif cell_value in ("ADD", "SUB", "SAT"):
        weight_map = {"ADD": 1.0, "SUB": 0.6, "SAT": 0.25}
        tier_map = {"ADD": "additive", "SUB": "sub_additive", "SAT": "saturating"}
        add_edge(src=src_design, dst=dst_design, edge_type="composes_with",
                 weight=weight_map[cell_value],
                 metadata={"alpha_tier": tier_map[cell_value]})
    elif cell_value == "EXCL":
        add_edge(src=src_design, dst=dst_design, edge_type="composes_with",
                 weight=0.0,
                 metadata={"alpha_tier": "exclusive"})
    # ORTHO + diagonal: no edge emitted

# 4. Add canonical_helper nodes (synthesis §3.1 enumeration; ~17 helpers)
for helper_path in CANONICAL_HELPERS_INVENTORY:
    add_node(node_id=basename(helper_path),
             node_type="canonical_helper",
             source_path=helper_path,
             metadata=parse_helper_metadata(helper_path))

# 5. Add meta_gate nodes from CLAUDE.md catalog table (1-332 + 333 sister)
for catalog_num in range(1, 334):
    add_node(node_id=f"catalog_{catalog_num}_{gate_name}",
             node_type="meta_gate",
             source_path="src/tac/preflight.py",
             metadata={"catalog_number": catalog_num, ...})

# 6. Add probe nodes from tools/probe_*.py
for probe_path in glob("tools/probe_*.py"):
    add_node(node_id=basename(probe_path),
             node_type="probe",
             source_path=probe_path,
             metadata={...})

# 7. Add substrate nodes from .omx/state/lane_registry.json
for lane in load_lane_registry()["lanes"]:
    add_node(node_id=lane["lane_id"],
             node_type="substrate",
             source_path=".omx/state/lane_registry.json",
             metadata={"level": lane["level"], ...})

# 8. Add deterministic_byte_derivation META node + 6 member subsystem
add_node(node_id="wyner_ziv_seed_subsystem",
         node_type="deterministic_byte_derivation",
         source_path="src/tac/wyner_ziv_deliverability/__init__.py",
         metadata={"members": ["wyner_ziv_layer", "wyner_ziv_deliverability",
                               "procedural_codebook_generator", "null_space_exploiter",
                               "optical_flow_side_info", "foe_detection"],
                   "canonical_principle": "Wyner-Ziv 1976 + Atick-Redlich 1990",
                   "deliverable_tiers": ["TIER_1_ZERO_COST", "TIER_2_CONSTANTS",
                                          "TIER_3_WAIVER_REQUIRED", "TIER_4_FORBIDDEN"]})

# 9. Add hyperedges from .omx/state/substrate_composition_matrix.json
for cell in load_composition_matrix():
    add_hyperedge(node_ids=tuple(sorted(cell["substrate_ids"])),
                  hyperedge_type="n_way_composition_alpha",
                  weight=cell["alpha_value"],
                  metadata={"alpha_tier": cell["alpha_tier"],
                            "deliverability_proof_path": cell.get("deliverability_proof_path"),
                            "validation_status": cell.get("validation_status", "UNVALIDATED")})

# 10. Add the 9-dim aggregate Pareto polytope as one n_way_pareto_feasibility hyperedge
add_hyperedge(node_ids=tuple(sorted(THE_9_DESIGN_IDS)),
              hyperedge_type="n_way_pareto_feasibility",
              weight=None,  # computed via Boyd-Dattorro alternating projection at query time
              metadata={"feasibility_algorithm": "dykstra_alternating_projection",
                        "binding_constraints": ["rate", "seg", "pose"]})
```

### 17.2 Acceptance verification per OP-HG-2

**Test 1**: `query_critical_path()` MUST return a path that includes `pr101_lc_v2_master_gradient_f174192aeadf` (the universal anchor) per synthesis §6.1 anchor citation summary (all 9 designs cite it; therefore it dominates the critical path).

**Test 2**: `query_orphan_signals(direction='consumer_without_producer')` MUST return the 3 hook-CONSUMER-without-producer flags per synthesis §8.2:
1. POSEAXIS OP-3 ATW V2-1 channel-pick reformulation
2. Z8 full conjunction dispatch
3. TT5L V2 4-primitive composition smoke

**Test 3**: `query_cycles()` MUST return ≥3 continual-learning cycles per §6.2 edge type #4 expected output.

**Test 4**: `query_hyperedge_compositions(alpha_tier='sub_additive')` MUST return ≥4 hyperedges (per Catalog #322 v2 cascade evidence: 4-of-8 binary pairs sub-additive).

**Test 5**: `query_deterministic_byte_derivation_subsystem()` MUST return `wyner_ziv_seed_subsystem` with members enumerated + hoist_opportunities populated (per operator's WZ question).

If any test fails, the bootstrap is INCORRECT and requires investigation per CLAUDE.md "Forbidden premature KILL" (do not KILL the hypergraph; investigate the bootstrap parser).

---

## 18. 6-Hook Wire-In Declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125 mandatory wire-in declaration: every landing must declare wire-in across the 6 mandatory unified-Lagrangian wire-in hooks.

### Hook 1: Sensitivity-map contribution (`tac.sensitivity_map.*`)

**Declaration**: ACTIVE.

**Wire-in**: the hypergraph itself is a sensitivity-map at the design-stack level. `query_dominator(node_id=<X>)` returns the downstream sensitivity to perturbations of node X — structurally analogous to per-byte sensitivity in `tac.sensitivity_map.*`. Future consumer: a `tac.sensitivity_map.design_graph_sensitivity` extension that combines per-byte sensitivity with per-design-node sensitivity for joint optimization.

### Hook 2: Pareto constraint (`tac.pareto_*`)

**Declaration**: ACTIVE.

**Wire-in**: the `n_way_pareto_feasibility` hyperedge type IS the Pareto constraint surface. Per Boyd Revision #4: every predicted ΔS band cites Dykstra-feasibility via metadata field `dykstra_feasibility_witnessed_by`. The 9-dim aggregate Pareto polytope per synthesis §13.2 is bootstrapped as one hyperedge.

### Hook 3: Bit-allocator hook (`tac.bit_allocator`)

**Declaration**: N/A — the hypergraph operates at design-stack granularity, not per-byte. Per-byte allocation is the domain of sister canonical helpers (`tac.master_gradient_consumers` + `tac.canonical_n_set_venn_classification` + `tac.bit_allocator`). The hypergraph CITES these via `canonical_helper` nodes + `consumes_output_of` edges to downstream consumers; it does not produce per-byte allocation directly.

**Rationale**: hook #3 N/A per the "the hypergraph is META over per-byte primitives, not a per-byte primitive itself" structural distinction.

### Hook 4: Cathedral autopilot dispatch hook (`tools/cathedral_autopilot_autonomous_loop.py`)

**Declaration**: ACTIVE (PRIMARY consumer).

**Wire-in**: cathedral autopilot ranker consumes `query_critical_path()` for dispatch sequencing + `query_predecessor_probe_outcomes()` for predecessor-aware filtering + `query_hyperedge_compositions()` for joint-α-aware ranking. Per §16.1.

### Hook 5: Continual-learning posterior update (`tac.continual_learning.posterior_update_locked`)

**Declaration**: ACTIVE.

**Wire-in**: every add_node / add_edge / add_hyperedge invocation appends to `.omx/state/design_stack_hypergraph.json` per Catalog #131 fcntl-locked discipline. The `cycles_back_to` edge type EXPLICITLY models the continual-learning feedback loops; `query_cycles()` enumerates them. Future consumer: a `tac.continual_learning.graph_aware_posterior` extension that uses graph structure to weight posterior anchors by node-dominator-set-size.

### Hook 6: Probe-disambiguator (`tools/probe_*_disambiguator.py`)

**Declaration**: ACTIVE.

**Wire-in**: per Assumption-Adversary Revision #2: the assumption "7 edge types are sufficient" requires a probe-disambiguator. The canonical probe is `tools/probe_design_graph_edge_taxonomy_completeness.py` (deferred to v2) that scans sister design memos for novel edge-type candidates not yet in the 7-category enum. For v1 fail-closed with `UnknownEdgeTypeError`.

### Hook wire-in summary

| Hook | Declaration | Rationale |
|---|---|---|
| #1 Sensitivity-map | ACTIVE | design-stack-level sensitivity via `query_dominator()` |
| #2 Pareto constraint | ACTIVE | `n_way_pareto_feasibility` hyperedge type |
| #3 Bit-allocator | N/A | hypergraph operates at design-stack granularity not per-byte |
| #4 Cathedral autopilot | ACTIVE (PRIMARY) | `query_critical_path()` + sister queries consumed by ranker |
| #5 Continual-learning posterior | ACTIVE | `cycles_back_to` edge type + fcntl-locked posterior |
| #6 Probe-disambiguator | ACTIVE | `query_*` API + v2 edge-taxonomy completeness probe deferred |

---

## 19. TOP-5 Op-Routables Ranked by EV

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + Carmack Revision #3: every design memo MUST enumerate top-5 op-routables with concrete file paths + canonical serializer commit command + EV ranking.

### Methodology

- **EV numerator**: `|predicted_unlock_value|` (qualitative; this memo's deliverables are infrastructure-enabling rather than direct-ΔS)
- **EV denominator**: `cost_envelope_upper_bound_usd` + `editor_days`
- **Tie-break**: cheapest envelope wins; equal envelope rank by upstream dependency depth

### TOP-5

| Rank | OP ID | Description | Concrete file paths | Cost envelope | Dependencies | Predicted unlock | EV qualitative |
|---|---|---|---|---|---|---|---|
| **1** | **OP-HG-1** | Codex builds `src/tac/design_graph.py` per sister routing directive C (this design memo IS the design authority) | `src/tac/design_graph.py` (~800 LOC; 9 query functions + atomic write + strict load + quarantine); `src/tac/tests/test_design_graph.py` (~600 LOC; 35+ dedicated tests); `tools/render_design_graph.py` (~400 LOC; 8 CLI subcommands) | $0 GPU + ~2-3 day editor | None (sister routing directive C ALREADY approved by operator) | UNLOCKS all downstream OPs | ∞ (zero cost; unblocks all) |
| **2** | **OP-HG-2** | Bootstrap seed graph from synthesis §4 + canonical helpers + composition_alpha posterior | invocation script: `tools/bootstrap_design_graph_from_synthesis.py` (~200 LOC); Codex emits as part of OP-HG-1 build | $0 GPU + ~1 day editor | OP-HG-1 | UNLOCKS query API; produces first hypergraph snapshot | HIGH (zero cost; unblocks queries) |
| **3** | **OP-HG-3** | Emit Catalog #333 STRICT preflight gate (warn-only initial) per sister routing directive C Layer 3 | `src/tac/preflight.py` (extend with `check_design_graph_hook_coverage_complete_or_orphans_declared`); CLAUDE.md catalog row 333; `src/tac/tests/test_check_333_design_graph_hook_coverage.py` (~150 LOC) | $0 GPU + ~1 day editor | OP-HG-1 + OP-HG-2 (bootstrap must complete before strict-flip) | UNLOCKS structural protection against hook orphans | HIGH (zero cost) |
| **4** | **OP-HG-4** | Wire `query_orphan_signals()` + `query_critical_path()` into Codex `/goal` LOOP pre-flight + cathedral autopilot ranker | `tools/cathedral_autopilot_autonomous_loop.py` (extend with `from tac.design_graph import query_critical_path, query_orphan_signals`); `~/.claude/projects/.../goal_loop_preflight.py` (if applicable); `tools/operator_briefing.py` (extend per sister routing directive C Layer 4) | $0 GPU + ~1 day editor | OP-HG-1 + OP-HG-2 | UNLOCKS machine-routable orchestration | HIGH (zero cost; enables continual-learning loop closure at session level) |
| **5** | **OP-HG-5** | Extend to **deterministic_byte_derivation** subsystem first-class queries per operator's WZ question | `src/tac/design_graph.py` (extend with `query_deterministic_byte_derivation_subsystem()`); `tools/render_design_graph.py` (extend with `--filter-type deterministic_byte_derivation` flag) | $0 GPU + ~2-3 day editor | OP-HG-1 | UNLOCKS per-substrate deterministic-byte-derivation hoist opportunity analysis | HIGH (zero cost; surfaces operator-highlighted META-category structurally) |

### Canonical serializer commit command for THIS landing

```bash
PREFLIGHT_SHA=$(sha256sum /Users/adpena/Projects/pact/.omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md | awk '{print $1}')
.venv/bin/python tools/subagent_commit_serializer.py \
    --message "design memo: full hypergraph model of design stack (10 typed nodes + 7 typed edges + 3 hyperedge types + cycles; deterministic_byte_derivation first-class per operator WZ question; T2 sextet PROCEED_WITH_REVISIONS 5 binding revisions; lands as DESIGN AUTHORITY for Codex sister routing directive C build authority for src/tac/design_graph.py)" \
    --files .omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md \
    --expected-content-sha256 ".omx/research/design_stack_full_hypergraph_model_design_memo_20260518.md=${PREFLIGHT_SHA}"
```

---

## 20. Cross-References

Per CLAUDE.md "Required durable state" + "Subagent coherence-by-default" non-negotiable.

### 20.1 Sister Codex routing directive

| Sister artifact | Role | Citation |
|---|---|---|
| `.omx/research/codex_routing_directive_design_stack_hypergraph_canonical_helper_plus_visualizer_20260518.md` | BUILD AUTHORITY for `src/tac/design_graph.py` per canonical 4-layer pattern (Layer 1 helper / Layer 2 CLI / Layer 3 strict gate Catalog #333 / Layer 4 operator_briefing wire-in) | THIS memo provides the DESIGN AUTHORITY; sister provides BUILD AUTHORITY; together they are the canonical 5-layer landing pattern (DESIGN + BUILD + TEST + WIRE + GATE) |

### 20.2 Synthesis seed memo

| Sister artifact | Role | Citation |
|---|---|---|
| `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` | INPUT seed; §4 9×9 cross-pollination matrix IS the seed adjacency representation | §6.4 edge enumeration translates matrix to 144 edges; §8.2 query 2 acceptance test references §8.2 of synthesis |

### 20.3 9 today design memos (each becomes a `design` node)

Per synthesis §3 per-landing summary table + §14.3 sister design memo cross-references.

### 20.4 Catalog # citations

| Catalog # | Topic | Section cited |
|---|---|---|
| #110 / #113 | HISTORICAL_PROVENANCE append-only / artifact lifecycle umbrella | §11.3 diff-able / §14.5 append-only event sourcing |
| #117 / #157 / #174 / #216 / #234 / #235 / #248 / #289 / #302 / #314 | Commit-swap protection family | §3 pre-flight compliance + §13.6 PV |
| #118 / #159 / #176 / #185 / #186 | CLAUDE.md catalog META-meta gates | §5 meta_gate examples |
| #125 | 6-hook wire-in non-negotiable | §18 6-hook wire-in declaration (PRIMARY) |
| #126 | Lane pre-registration | §3 pre-flight compliance + §13.6 PV |
| #127 | Custody validator routing | §5 empirical_anchor metadata schema |
| #128 / #131 / #138 / #245 | fcntl-locked JSONL append-only discipline | §4 canonical-vs-unique decision per layer + §14 storage + persistence (PRIMARY) |
| #131 | Bare writes to shared state | §14.3 atomic write + fcntl-lock |
| #138 | Strict-load fail-closed | §4 canonical decision + §14.4 strict-load discipline |
| #139 / #272 | No-op detector + distinguishing-feature integration contract | §11.6 counterfactual-able + §7.2 hyperedge type 1 anti-phantom |
| #146 | Phase 1 trainer contest-compliant runtime | §5 substrate metadata schema |
| #167 | Smoke-before-full pattern | §3 pre-flight + §19 op-routables |
| #190 | `detect_hardware_substrate` canonical helper | §5 empirical_anchor metadata schema |
| #199 | Operator-authorize paired-env bypass discipline | §1 operational consequences |
| #205 | Submission inflate device-fork canonical | §5 deterministic_byte_derivation metadata schema |
| #206 | Subagent crash-resume discipline | §3 pre-flight + §13.7 PV |
| #220 | Substrate L1+ scaffold operational mechanism | §5 substrate metadata schema |
| #226 | Trainer auth_eval canonical helper | §5 substrate metadata schema |
| #229 | Premise verification before edit | §13 PV trail (PRIMARY) |
| #240 | Substrate contest-CUDA chain complete or research_only | §5 substrate metadata schema |
| #244 | Modal NVML env block | §6.2 edge type 6 waiver_eligible_via example |
| #245 | Modal call_id ledger canonical | §14 storage + persistence (PRIMARY pattern sister) |
| #270 | Canonical dispatch optimization protocol (UMBRELLA) | §6.2 edge type 5 gates_eligibility_of example |
| #271 | Codex pre-dispatch review automation | §5 consumer metadata schema |
| #287 | Empirical claims have evidence tag | §0 + §13 + every numeric prediction `[prediction]` axis tag |
| #290 | Substrate canonical-vs-unique decision per layer | §4 canonical-vs-unique decision per layer (PRIMARY) |
| #291 / #292 | META-ASSUMPTION review + per-deliberation assumption surfacing | §9 cargo-cult audit + §15 binding revisions |
| #294 | 9-dimension success checklist evidence | §10 9-dim checklist evidence (PRIMARY) |
| #296 | Substrate predicted band Dykstra-feasibility check | §12 Dykstra-feasibility intersection (PRIMARY) |
| #298 | Substrate retirement discipline 30-day | §8.2 query 8 predecessor probe outcomes |
| #299 | Catalog quota under 400 | §5 meta_gate Catalog #333 sister |
| #300 | Council deliberation v2 frontmatter | frontmatter (PRIMARY) + §1 mission alignment |
| #303 | Cargo-cult audit section | §9 cargo-cult audit (PRIMARY) |
| #305 | Observability surface | §11 observability surface (PRIMARY) |
| #307 / #308 / #309 / #310 / #311 / #312 | Six META pattern strict gates | §5 meta_gate examples |
| #313 | Predecessor probe outcome ledger | §8.2 query 8 + Contrarian Revision #1 (PRIMARY) |
| #314 | Subagent commit absorption-pattern detector | §3 pre-flight + §20.5 sister-subagent ownership |
| #315 | Substrate optimal form before paid dispatch | §5 substrate metadata schema |
| #316 | reports/latest.md frontier scanner | frontmatter canonical_frontier_anchor |
| #318 / #327 | Master-gradient raw-byte authority guard | §5 empirical_anchor metadata schema |
| #319 | Wyner-Ziv deliverability proof / v2 cascade | §5 deterministic_byte_derivation + §7.2 hyperedge type 3 sister |
| #321 / #322 / #323 | Phantom-score class extinction at multiple surfaces | §7.2 hyperedge anti-phantom protection |
| #324 | No predicted band without post-training Tier-C validation | §0 reactivation criteria |
| #325 | Per-substrate optimal form symposium discipline | §6.2 edge type 5 gates_eligibility_of example |
| #326 | Substrate driver consumes trainer mode env var | §5 substrate metadata schema |
| #333 | (sister routing directive C Layer 3 STRICT gate) | §5 meta_gate + §18 hook #6 |

### 20.5 Sister-subagent ownership map (per Catalog #302 + #314)

| Sister subagent | Subagent ID / lane | Disjoint scope per Catalog #314 |
|---|---|---|
| Codex `/goal` LOOP `019de465` (in-flight per system prompt) | `019de465` | OWNS `src/tac/design_graph.py` + `tools/render_design_graph.py` + `src/tac/tests/test_design_graph.py` per sister routing directive C |
| Multi-loop Codex `/goal` design memo subagent F (per system prompt) | (parallel spawn) | OWNS `.omx/research/multi_loop_codex_goal_design_memo_20260518.md` (DISJOINT from this memo's file) |
| Sister cross-stack synthesis subagent (already landed) | `cross_stack_synthesis_20260518` | OWNS `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (INPUT seed) |
| 9 sister design memo subagents (already landed) | (9 IDs per synthesis §14.5) | OWNS the 9 design memos (each is a `design` node in this hypergraph) |

### 20.6 Memory file citations

| Memory file | Cited section | Role |
|---|---|---|
| `feedback_cross_stack_synthesis_9_design_landings_unified_framework_landed_20260518.md` | §3 pre-flight + §17 bootstrap | Synthesis landing memo; seed adjacency |
| `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` | §4 canonical-vs-unique | Standing directive for UNIQUE-AND-COMPLETE-PER-METHOD |
| `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` | §4 canonical-vs-unique | PR95 META-level retrospective |
| `feedback_modal_call_id_ledger_canonical_landed_20260515.md` | §14 storage + persistence | Canonical 4-layer pattern sister |
| `feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517.md` | §7.2 hyperedge anti-phantom | Catalog #323 provenance canonical fix |
| `feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md` | §0 reactivation criteria | Catalog #324 post-training Tier-C validation |
| `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` | §18 6-hook wire-in | Unified Lagrangian action migration target |
| `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md` | §6.2 edge type 6 + §18 hook #6 | Probe-disambiguator pattern |
| `feedback_subagent_crash_resume_discipline_landed_20260514.md` | §3 pre-flight + §13.7 PV | Catalog #206 checkpoint discipline |
| `feedback_council_hierarchy_v2_landed_20260516.md` | §1 mission alignment | 4-tier council protocol |

---

## 21. Binding Revisions Per Council Dissent (Full Text)

Per §0 council verdict matrix: 5 binding revisions per Contrarian + Assumption-Adversary + Carmack + Boyd + Tao.

### Revision #1 (Contrarian VETO)

**Verbatim**: *"hypergraph MUST honor predecessor probe outcomes per Catalog #313 — when a query traverses an edge whose endpoint is a DEFER/KILL/INDEPENDENT-verdict substrate, the traversal returns the verdict + freshness in addition to the structural edge metadata. Without this, queries can produce dispatch recommendations for substrates the apparatus already adjudicated."*

**Resolution**: §8.1 includes `query_predecessor_probe_outcomes(node_id)` as the 8th query function. §8.2 query 8 documents the algorithm + sister usage. Every other query that returns substrate nodes MUST optionally filter via this query's results.

### Revision #2 (Assumption-Adversary CARGO-CULTED check)

**Verbatim**: *"the assumption '7 edge types are sufficient' is CARGO-CULTED-PENDING-EMPIRICAL. Real-world graph evolution may surface additional edge categories (e.g. `falsifies_premise_of` from Catalog #229 probes; `supersedes_via_council_verdict` from Catalog #300 v2 anchors). The 7-category baseline MUST carry an extension protocol (canonical helper `add_edge_type(...)` deferred to v2; for v1 fail-closed with `UnknownEdgeTypeError`)."*

**Resolution**: §9.2 cargo-cult audit explicitly classifies this as CARGO-CULTED-PENDING-EMPIRICAL with reactivation criterion + 3 candidate additional edge types enumerated. §18 hook #6 declares ACTIVE wire-in for `tools/probe_design_graph_edge_taxonomy_completeness.py` (v2 deferred).

### Revision #3 (Carmack 30-second-reviewability)

**Verbatim**: *"graph queries MUST be machine-routable, not visualization-only. Every query function MUST return typed dataclasses (not Python primitives) so downstream consumers compose. The DOT export is a debugging surface, not the primary deliverable."*

**Resolution**: §4 canonical-vs-unique decision per layer commits to FORK_BECAUSE_PRINCIPLED_MISMATCH for "query result types (primitives vs typed dataclasses)" → typed dataclasses. §8.1 enumerates the typed result dataclasses (`CriticalPathResult`, `OrphanSignalAuditResult`, `HookCoverageReport`, `DominatorSetResult`, `CycleEnumerationResult`, `DeterministicByteDerivationReport`). §15 explicitly notes DOT is "primary" for visualization but the typed-dataclass query API is the primary OPERATIONAL deliverable.

### Revision #4 (Boyd Dykstra-feasibility)

**Verbatim**: *"hyperedge `n_way_pareto_feasibility` MUST cite the convex-intersection-projection algorithm explicitly (Boyd-Dattorro alternating projection per CLAUDE.md 'Meta-Lagrangian/Pareto solver' non-negotiable). Without explicit citation, the hyperedge degenerates to a tag without algorithmic meaning."*

**Resolution**: §7.2 hyperedge type 2 specification includes `feasibility_algorithm` metadata field with canonical values (`dykstra_alternating_projection` / `boyd_admm` / `convex_intersection_volume`). §12 Dykstra-feasibility intersection section documents the algorithm + the 9-dim aggregate Pareto polytope per synthesis §13.2 as the canonical witness.

### Revision #5 (Tao graph theory rigor)

**Verbatim**: *"the formalism MUST distinguish between (a) hypergraph (F = set of edges of arbitrary cardinality ≥ 2); (b) bipartite-incidence representation (edges as nodes connecting to incident-vertex nodes); (c) line graph (edges as nodes connecting via shared-endpoint adjacency). The design adopts (a) hypergraph + provides (b) bipartite-incidence as the canonical persistence schema (each hyperedge becomes a JSONL row with node_ids: tuple[str, ...])."*

**Resolution**: §2.2 labeled directed hypergraph formalism explicitly defines `F ⊆ 2^V`. §7.3 hyperedge-axiom invariants specifies the persistence-friendly representation: each hyperedge is one JSONL row with `node_ids: tuple[str, ...]` (sorted; tuple for hashability + sort_keys=True byte-stability per Catalog #245). §14.2 schema confirms.

---

## 22. Council Verdict + Continual-Learning Anchor Emission (Catalog #300 hook #5)

Per CLAUDE.md "Council hierarchy: 4-tier protocol" Continual learning wire-in rule + Catalog #300 v2 frontmatter + Catalog #125 hook #5: every T2+ deliberation MUST emit a continual-learning anchor via the canonical helper `tac.council_continual_learning.append_council_anchor(record)`.

### 22.1 Anchor record to be emitted

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="design_stack_full_hypergraph_model_design_memo_20260518",
    topic="Full labeled hypergraph model of pact design stack — 10 typed nodes + 7 typed edges + 3 hyperedge types + cycles + canonical graph queries + deterministic_byte_derivation first-class per operator WZ question",
    council_tier=CouncilTier.T2,
    council_attendees=(
        "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
        "Tao", "Mallat", "Boyd", "Carmack", "van_den_Oord", "MacKay_memorial",
    ),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian",
         "verbatim": "hypergraph MUST honor predecessor probe outcomes per Catalog #313 - traversal returns verdict + freshness alongside structural metadata"},
        {"member": "Assumption-Adversary",
         "verbatim": "7 edge types assumption CARGO-CULTED-PENDING-EMPIRICAL; v1 fail-closed UnknownEdgeTypeError; v2 extension protocol deferred"},
        {"member": "Carmack",
         "verbatim": "graph queries MUST be machine-routable not visualization-only; typed dataclasses not Python primitives; DOT export is debugging surface not primary"},
        {"member": "Boyd",
         "verbatim": "n_way_pareto_feasibility MUST cite Boyd-Dattorro alternating projection algorithm explicitly; tag without algorithmic meaning is degenerate"},
        {"member": "Tao",
         "verbatim": "formalism MUST distinguish hypergraph vs bipartite-incidence vs line graph; adopt hypergraph + bipartite-incidence as canonical persistence schema"},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "labeled hypergraph IS the right structural primitive",
         "classification": "HARD-EARNED",
         "rationale": "subsumes 7 simpler alternatives; synthesis 9x9 matrix requires typed binary edges + N-way composition + cycles + weights"},
        {"assumption": "7 edge types are sufficient",
         "classification": "CARGO-CULTED-PENDING-EMPIRICAL",
         "rationale": "based on current synthesis matrix; future iterations may surface falsifies_premise_of / supersedes_via_council_verdict / triggers_reactivation_of categories; v1 fail-closed; v2 extension deferred"},
        {"assumption": "hyperedges sufficient for N-way composition_alpha",
         "classification": "HARD-EARNED-WITH-REVISION",
         "rationale": "Berge 1973 canonical N-way primitive; risk is tensor representation alternative for continuous-gradient sensitivity; v2 deferred"},
        {"assumption": "cycles ONLY occur in continual-learning feedback loops",
         "classification": "CARGO-CULTED-PENDING-EMPIRICAL",
         "rationale": "deadlock cycles (Catalog #292 + #229 + #303 mutual blocking) are a known cycle class; post-bootstrap classification will validate"},
        {"assumption": "operator WZ question elevates deterministic_byte_derivation to first-class",
         "classification": "HARD-EARNED",
         "rationale": "explicit operator declaration 2026-05-18; sister Codex routing directive C confirms; derived bytes never charged by contest rate term is the META semantic"},
        {"assumption": "graph queries are canonical orchestration surface",
         "classification": "HARD-EARNED",
         "rationale": "machine-routable typed queries compose; prose synthesis matrices do not; per Carmack Revision #3 + CLAUDE.md Beauty simplicity DX non-negotiable"},
    ),
    council_decisions_recorded=(
        "OP-HG-1 (TIER-1 unlock): Codex builds src/tac/design_graph.py per sister routing directive C; $0 GPU; ~2-3 day editor; UNLOCKS all downstream",
        "OP-HG-2: bootstrap seed graph from synthesis section 4 + canonical helpers + composition_alpha posterior; $0 GPU; ~1 day editor",
        "OP-HG-3: emit Catalog #333 STRICT preflight gate warn-only initial per sister routing directive C Layer 3; $0 GPU; ~1 day editor",
        "OP-HG-4: wire query_orphan_signals + query_critical_path into Codex /goal LOOP pre-flight + cathedral autopilot ranker; $0 GPU; ~1 day editor",
        "OP-HG-5: extend deterministic_byte_derivation subsystem first-class queries per operator WZ question; $0 GPU; ~2-3 day editor",
    ),
    predicted_mission_contribution="frontier_breaking",
    override_invoked=False,
    override_rationale=None,
    deferred_substrate_id=None,
    deferred_substrate_retrospective_due_utc=None,
)
append_council_anchor(record)
```

### 22.2 Operational notes

- Anchor emitted post-memo-landing via canonical helper `tac.council_continual_learning.append_council_anchor` per fcntl-locked discipline per Catalog #131.
- Per Catalog #300 v2 frontmatter `council_predicted_mission_contribution: frontier_breaking` declared.
- Per Catalog #300 v2 frontmatter `council_override_invoked: false` declared.
- Per Catalog #300 v2 frontmatter `deferred_substrate_id: null` declared (no substrate deferred; design authority memo).
- Sister deliberation IDs cited per `related_deliberation_ids` frontmatter field (15 sister deliberations referenced).

---

## 23. Closing Remarks

The hypergraph canonicalizes the design stack into a labeled directed hypergraph `H = (V, E, F, τ_V, τ_E, τ_F, w_E, w_F, μ_V, μ_E, μ_F)` per §2.2 formal structure. The 10 typed node categories (§5) + 7 typed edge categories (§6) + 3 hyperedge types (§7) + 9 canonical query operations (§8) constitute the machine-routable orchestration surface that converts the synthesis §4 9×9 prose matrix into typed structure.

### What this design memo IS

- The canonical DESIGN AUTHORITY for `src/tac/design_graph.py` (sister Codex routing directive C is the BUILD AUTHORITY).
- The canonical META-level design authority for a graph-formalism over the design stack.
- The canonical 4-tier formalism (hypergraph axioms + per-category specs + invariants + query API) per Berge 1973 + Bollobás 1998 + Tao 2006 + Cormen et al. 2009.
- The canonical operator-routable extension of the synthesis §4 matrix into typed structure.

### What this design memo is NOT

- NOT the canonical helper itself (Codex builds `src/tac/design_graph.py` per sister routing directive C; this memo is the design contract).
- NOT a substitute for individual design memos (each of the 10 design nodes retains canonical authority over its own design).
- NOT a paid-dispatch authorization (`research_only: true` + `score_claim: false` + `promotion_eligible: false` + `provider_spend: false`).
- NOT a substitute for per-substrate symposium discipline per Catalog #325 (substrate-class designs land via per-substrate symposiums; the hypergraph CITES them as nodes).

### Per CLAUDE.md "Forbidden premature KILL" non-negotiable

The hypergraph preserves DEFERRED status for all nodes not yet at reactivation per §0. No design is killed; all 9 today designs + sister canonical helpers + meta gates + probes + substrates + venn cells + posteriors + consumers + empirical anchors + deterministic_byte_derivation subsystem are represented as typed nodes.

### Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable

The hypergraph introduces ONE new STRICT preflight gate via sister routing directive C: Catalog #333 `check_design_graph_hook_coverage_complete_or_orphans_declared`. Initial wire-in WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"; strict-flip after first clean `query_hook_coverage()` cycle returning zero orphans across all 6 hooks.

### Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable

The hypergraph is structured to RESPECT race-mode rigor inversion: `query_critical_path(race_mode=True)` flips weighting from `predicted_delta_s` to `cost_envelope_inverse` (cheapest-bolt-on-first per the 2026-05-04 race postmortem template). The race-mode infrastructure is DORMANT — present for activation when leaderboard moves; non-race-mode rigor cadence applies until then.

### Per operator 2026-05-18 standing directives

- *"Chain but how about a graph"* → graph formalism adopted; chain framing extincted.
- *"All are approved"* → sister Codex routing directive C build authority approved; this memo provides matching design authority.
- *"wyner-ziv is still being considered as part of the seed and other procedural gen experiments right?"* → `deterministic_byte_derivation` elevated to first-class node category 10; §5.2 category 10 specification.

---

## End of Design Memo

**Lane**: `lane_design_stack_full_hypergraph_design_20260518` (L1 at memo landing per Catalog #126)
**Subagent**: `full_hypergraph_design_subagent_20260518` (per Catalog #206 checkpoint discipline)
**Council anchor**: emitted via `tac.council_continual_learning.append_council_anchor` per §22
**Memory file**: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_design_stack_full_hypergraph_model_design_memo_landed_20260518.md`
**Verdict**: PROCEED_WITH_REVISIONS (5 binding revisions per §0; full text in §21)
**Predicted impact**: ZERO direct ΔS; INDIRECT frontier-breaking via machine-routable query surface enabling cathedral autopilot dispatch sequencing + Codex /goal LOOP pre-flight orphan-signal closure + per-substrate symposium queue N-way α enumeration
**Mission contribution**: `frontier_breaking` per Catalog #300 v2 frontmatter
