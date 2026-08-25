# The Lens Engine — one multi-lens analyzer over the campaign's double one-object

**Source:** operator directive 2026-07-12 — the 8-lens framework (vector-geometry / topology-terrain /
graph / set / spatial / temporal / statistics / relational-algebra), built as ONE unified engine serving
BOTH the research-corpus knowledge-graph AND the witness Morse-Smale geometry.

**Status:** DESIGN (increment-1 build to follow). MEANS — this is apparatus that makes the corpus as
navigable as it is writable (the P0 anti-forgetting / retrieval-first goal) AND turns the witness geometry
into a queryable instrument. Pointer 0.19108282 UNMOVED.

## 1. The unification — a Typed Attributed Complex `T`

Both surfaces are instances of ONE abstract structure. Let `T = (E, G, Φ, S, L, R, X)`:
- `E` — ELEMENTS (nodes). Corpus: memos/findings/levers/measurements. Witness: pixels/cells/regions/critical-points.
- `vec: E → ℝ^d` — a semantic / spatial VECTOR per element (→ vector-geometry lens).
- `Φ: E → ℝ` — a scalar FIELD ("height") per element. Corpus: relevance/recency/citation-salience. Witness: margin/loss. (→ topology lens.)
- `S` — SCOPES: set memberships. Corpus: verdict_scope (instance/formulation/family/paradigm). Witness: class-regions. (→ set lens.)
- `X: E → ℝ^k` — SPATIAL embedding (polygons/points). Corpus: region-of-relevance. Witness: 384×512 grid / Laguerre cells. (→ spatial lens.)
- `t: E → interval`, `L` — TIMESTAMP + LINEAGE edges (supersession / se(3) screw). (→ temporal lens.)
- `R` — typed RELATIONS across element-types (memo↔lever↔measurement / class↔region↔pixel). (→ relational lens.)
- `G = (E, edges)` — the GRAPH (citations/lineage / region-adjacency). (→ graph lens.)
- collectively a DISTRIBUTION over `(vec, Φ, X)` (→ statistics lens).

The claim (the double one-object): the corpus and the witness are the SAME `T` under two adapters. One
engine, two adapters, eight lenses.

## 2. The 8 lenses (typed operations on `T`) + what each REUSES

| Lens | Ops | Reuses (don't rebuild) |
|---|---|---|
| **vector** (semantic similarity) | cosine k-NN, cluster, project | graph-memory embeddings (#415); dual θ↔η coords |
| **topology** (peaks/saddles/basins/routes) | Morse-Smale on `Φ`, persistence diagram, watershed basins, integral-line "routes" | Morse-Smale codec (#180), margin field (#141), persistence loss |
| **graph** (networks/traversal) | BFS/DFS, shortest-path, centrality, components, community | DAG-graph-memory (#411, 8788 nodes/30516 edges), Obsidian round-trip (#415) |
| **set** (scopes ∪∩∖) | scope algebra over `S`-subsets | verdict_scope ladder; class-region masks |
| **spatial** (polygons/containment/overlap/distance) | point-in-polygon, IoU-overlap, distance, Voronoi/Laguerre cells | Laguerre power-diagram (#284), lane band, class polygons |
| **temporal** (interval/lineage) | Allen interval algebra, lineage/supersession DAG walk | DAG provenance, se(3) screw lineage (tac.lie) |
| **statistics** (density/persistence/drift/anisotropy/change) | KDE density, topological persistence, distribution-drift, structure-tensor anisotropy, change-point | anisotropy (#277 along-tangent), drift (EMA/costate), persistence (#180) |
| **relational** (project scopes → articles/entities/evidence) | select/project/join across `R` | triality DAG↔DSL↔equations joins |

## 3. The 2 adapters (expose existing data as `T`)

- `CorpusAdapter` — over `tac.graph_memory` (#411/#415): `E`=nodes, `Φ`=relevance/recency, `S`=verdict_scope,
  `L`=supersession, `R`=triality legs, `vec`=node embeddings.
- `WitnessAdapter` — over the live level-set field: `E`=pixels/cells/critical-points, `Φ`=margin/loss,
  `S`=class-regions, `X`=grid/Laguerre, `L`=pair-index/se(3), stats from the run telemetry.

## 4. Query API (composable)

`lens_engine.query(adapter, lens, op, **args) → TypedResult`. Lenses COMPOSE via the relational lens:
e.g. `relational.join(topology.saddles(Φ), set.scope("formulation"))` = "which cruxes are formulation-scoped."
Same call shape over the witness: `topology.basins(margin) ∩ spatial.laguerre_cells` = "which power-cells are stable basins."

## 5. Build increments (MVP-first)

- **inc-1** (we already have the pieces): core `T` dataclass + `Lens` Protocol + `CorpusAdapter` + `WitnessAdapter`
  + the 4 reuse-heavy lenses {topology, graph, spatial, statistics}. Tests + one worked query per surface.
- **inc-2**: {vector, set, temporal, relational} lenses.
- **inc-3**: cross-lens composition + a small unified query language + wire into #346 retrieval-first (corpus)
  and the dashboard/costate SENSE layer (witness).

## 6. Triality + double-use

The engine is the campaign's double one-object made COMPUTABLE (witness physics facets + campaign
representational views, one substrate). Module `tac.lens_engine`. New capability → DSL surfacing where a
query becomes a lever/telemetry; measured witness-geometry facts → equations/DAG. Reuses, does not
duplicate: #411 graph memory, #180 Morse-Smale, #284 Laguerre, #141 margin, #277 anisotropy, #415 query-tools.

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §2 "The 8 lenses (typed operations on `T`) + what each REUSES" makes every lens a separately-invocable typed operation over the Typed Attributed Complex `T` defined in §1.
2. **Per-signal decomposition** — §1's typed attributes ARE the decomposition: each lens reads named attributes of `T` rather than a composite headline.
3. **Run-to-run diff** — `T` is a typed complex, so two campaign states are diffable node-by-node under the same type; §3 "The 2 adapters (expose existing data as `T`)" is what makes existing stores comparable.
4. **Post-hoc query** — §4 "Query API (composable)" is precisely the post-hoc query interface; the adapters in §3 are what it queries.
5. **Cite-chain** — §6 "Triality + double-use" ties the engine to the DAG / DSL / equations legs, which carry the citations.
6. **Counterfactual hooks** — §5 "Build increments (MVP-first)" stages the engine so each lens can be added or withheld independently.

**Scope honesty:** this memo designs an ANALYZER over campaign state, not a substrate that contributes archive bytes. It has no runtime forward pass and names no trainer flags; facets 1-3 above describe the analyzer's own structure, not a training run's.
