# Lens Engine increment 1 landed — unified corpus/witness analysis substrate

**Date:** 2026-07-12  
**Lane:** `lane_lens_engine_inc1_20260712`  
**Status:** LANDED LOCALLY / ANALYSIS APPARATUS / MEANS  
**Authority:** no training actuation, no archive candidate, no evaluator replay, no score claim  
**Pointer delta:** `0.19108282` UNMOVED

## What increment 1 builds

`tac.lens_engine` now exposes the design's double one-object as a validated typed attributed complex

\[
T = (E, G, \Phi, S, L, R, X, t, \mathrm{vec}).
\]

- `TypedAttributedComplex` (`T`) with frozen typed elements, graph/lineage edges, relations, spatial
  supports, intervals, vectors, scopes, scalar field values, recursively frozen metadata/attributes,
  immutable-bytes-backed arrays, endpoint validation, unique-id validation, and finite-number guards.
- Runtime-checkable `Lens` and `ComplexAdapter` protocols plus the uniform
  `TypedResult[value]` envelope (`lens`, `op`, `adapter`, element ids, method provenance, metadata).
- Four increment-1 lenses:
  - `topology`: peaks, discrete branching saddles, ascent/descent basins, separatrices, watershed,
    integral routes, basin relief, and deterministic graph-filtration H0 persistence;
  - `graph`: BFS, DFS, weighted shortest path, degree/closeness/betweenness centrality, weak
    components, and explicitly named deterministic label-propagation communities;
  - `spatial`: point-in-polygon, exact mask/set IoU, point distance, and Laguerre/power cells;
  - `statistics`: scalar KDE, delegated topological persistence, Wasserstein/KS drift,
    structure-tensor anisotropy, and EMA-smoothed two-segment change-point analysis.
- `CorpusAdapter` over an injected `tac.graph_memory.Graph` or the existing cache through read-only
  `Graph.load`. `Phi` modes are explicit: degree is the safe structural default; recency, stored
  citation salience, stored relevance/phi, mappings, and callables require custodied values for every
  adapted node. Missing field values fail closed rather than becoming extrema-producing zeros. Missing
  embeddings and structured verdict scopes stay empty; the adapter never fabricates hash vectors or
  authority scopes.
- `WitnessAdapter` over one 2-D margin/loss field and `lstars` partition in either connected-cell or
  pixel mode. Cell mode uses the real 4-connected region-adjacency graph. `from_npz` reads only the
  requested leading-axis pair from each NPY member, so the canonical multi-gigabyte NPZ remains
  read-only and peak memory is bounded to one pair rather than one whole member.
- Pixel mode refuses materialization above its typed element-count limit unless the caller explicitly
  opts into the large object graph. Spatial operations preserve coordinate-axis names and fail closed
  on row/column versus x/y mismatches without an explicit transform.
- `lens_engine.query(adapter, lens, op, **args) -> TypedResult` with registered singleton lenses and
  raw-`T` support for internal use.
- SciPy-backed operations are lazy and declared in the `analysis` optional dependency extra. Core,
  corpus, graph, topology, and package imports stay clean in a simulated base install without SciPy;
  an invoked SciPy-backed operation gives an explicit `install tac[analysis]` error.

## Reuse map and authority boundary

| Increment-1 surface | Reused implementation | Honest boundary |
|---|---|---|
| Corpus graph | `tac.graph_memory.Graph` / `Node` / `Edge` / deterministic `Graph.load` | General DFS, shortest path, centrality, components, and communities did not exist as public graph-memory operations; the lens supplies thin deterministic algorithms over the reused graph model. |
| Witness regions | `tac.boundary_math.partition.build_region_adjacency_graph` | The adapter uses its actual connected components and symmetric RAG; it does not infer smooth regions from prose or telemetry. |
| Class scopes | `tac.boundary_math.bitmask_dseg.class_masks_from_argmax` | Class ids are validated; absent/invalid classes fail closed. |
| Laguerre cells | `tac.boundary_math.partition_collapse.PowerDiagram` and `power_assign` | Cell assignment is the existing power-diagram implementation; generic containment/IoU/distance were absent and are small typed lens primitives. |
| Anisotropy | `tac.boundary_math.partition_anisotropy_map.structure_tensor_dH` | Returned arrays are analysis-only, method-tagged, and read-only. |
| Telemetry trend primitives | `tac.witness_control.sigma_min_plateau.ema_smooth` and `tac.witness_control.costate_estimator.slope_with_stderr` | Change-point selection is explicitly the Lens Engine's deterministic maximum two-segment mean-shift rule, not a renamed existing detector. |
| Density/drift | SciPy `gaussian_kde`, `wasserstein_distance`, `ks_2samp` | No repository KDE or generic distribution-drift primitive existed. |
| Topology | witness RAG plus a typed discrete graph-field filtration | The historical #180 feasibility landing is a connected-component/contour/polygon codec, not an importable full Morse-Smale critical-point/separatrix API. Increment 1 therefore labels its method `discrete graph filtration over T.Phi and T.G`; it does not claim a smooth Morse-Smale reconstruction. `persistence_topology_loss` is not relabeled as persistent homology. |

## Two worked queries — one lens, both surfaces

The same registered `TOPOLOGY` lens instance is used in both calls:

```python
from tac.lens_engine import CorpusAdapter, TOPOLOGY, WitnessAdapter, query

corpus = CorpusAdapter.from_cache(phi="degree")
corpus_cruxes = query(corpus, TOPOLOGY, "saddles")

witness = WitnessAdapter.from_npz(
    "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
    pair_index=0,
    mode="cells",
)
stable_regions = query(witness, TOPOLOGY, "basins")
```

The dedicated tests use a corpus branch with two higher-field arms to prove the saddle classification,
and a labeled witness field to prove that all connected regions are partitioned into stable ascent
basins. A read-only smoke also ran these call shapes against the canonical graph-memory cache and pair 0
of the canonical witness cache. It recorded no geometry statistic or score claim.

## Verification

- `69 passed` — dedicated Lens Engine positive/negative/edge/cross-surface tests.
- `179 passed` — dedicated tests plus graph-memory, boundary partition, Laguerre, anisotropy, and
  telemetry plateau reuse-target regressions.
- `mypy src/tac/lens_engine`: clean across all 10 Python files.
- `ruff check src/tac/lens_engine`: clean.
- `ty check src/tac/lens_engine`: no Lens Engine diagnostic; command exit 1 is solely the pre-existing
  project-config warning for the unsupported `possibly-unbound` rule.
- `uv lock --check --offline`: clean after adding SciPy to the `analysis` extra.
- Import smoke from `/tmp`: clean; there is no installed-package dependency on repository-only
  `tools.*` modules.
- Read-only canonical corpus + witness cross-surface smoke: clean.
- Independent adversarial re-review: no remaining blocker or high-severity finding after executable
  attacks on dependency laziness, deep immutability, axes, Phi custody, topology contracts, graph
  semantics, pixel guarding, and the bounded NPZ loader.

The protected live `experiments/results/v9_cgauge_432_coherent_arm_20260711` run, its process, trainer,
and result files were not modified or actuated.

## Triality and six-hook disposition

This landing is pure analysis apparatus. It produces no measured witness-geometry fact, no training
lever, no candidate bytes, and no empirical posterior row; therefore the code commit is explicitly
`[no-triality]` rather than manufacturing DAG/equation/control claims.

- **DSL/control leg:** increment 1 supplies the typed callable query surface only. Increment 3 owes the
  small unified query language and the explicit mapping from a selected query result to a governed DSL
  lever or SENSE telemetry consumer. No query currently actuates training or changes a loss weight.
- **DAG/trajectory leg:** `TypedResult.element_ids` and method provenance make future results
  registerable. If a later run surfaces a measured saddle/basin/drift fact that changes action, that
  fact must land in the canonical result/probe ledger and DAG with its exact adapter, pair, field, and
  method. This increment records none.
- **Equation/law leg:** the structural law `T=(E,G,Phi,S,L,R,X,t,vec)` is implemented. No new score law,
  measured coefficient, or witness-energy term is asserted, so the canonical equation registry is
  unchanged.
- **Sensitivity map:** no empirical sensitivity is measured; future measured query results may become
  sensitivity-map evidence only after authority/provenance registration.
- **Pareto constraint / bit allocator:** non-binding in increment 1 because queries neither allocate
  bytes nor admit candidates. A future consumer must carry score-unit value per byte before actuation.
- **Cathedral/autopilot dispatch:** no dispatch hook in increment 1. Increment 3 owes retrieval-first
  and dashboard/SENSE consumers behind existing governance.
- **Continual-learning posterior:** no empirical anchor, hence no posterior mutation. Corpus retrieval
  is improved as an analysis surface, not silently treated as learned evidence.
- **Probe disambiguation:** multiple defensible corpus field meanings are callable modes rather than a
  hidden weighted blend; callers can compare explicit degree, custodied citation salience/recency,
  stored fields, or an explicit mapping. Missing values fail rather than acquiring a synthetic zero.
  No mode is promoted by this scaffold.

## Explicitly still owed

### Increment 2 — do not infer from this landing

- vector lens: custodied embedding cosine k-NN, clustering, projection;
- set lens: scope/class-region union, intersection, difference;
- temporal lens: Allen interval algebra plus lineage/supersession walks;
- relational lens: select/project/join across typed relations and triality legs.

The corpus cache's current lack of guaranteed embeddings and structured verdict scopes must be solved
at the source or by separately custodied inputs; increment 2 must not backfill invented values.

### Increment 3 — composition and consumers

- cross-lens composition with typed compatibility checks;
- the small unified query language / DSL;
- retrieval-first integration with #346 on the corpus surface;
- governed witness dashboard/costate SENSE integration;
- durable registration path for measured query facts into probe outcomes, DAG, equations, sensitivity,
  and continual-learning consumers.

No increment-2 or increment-3 implementation was started in this pass.
