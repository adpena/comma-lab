# tac.search namespace design (5 of 5 §7.6 helper namespaces)

**Date:** 2026-05-17
**Status:** active design memo; consumed by the impl wave for the 5 of 5
§7.6 helper namespaces
**Lane:** `lane_tac_search_namespace_decorator_api_20260517`
**Provenance:** `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md` §5.2 + §5.4 + §F search-method coverage
**Sister namespaces:** `tac.boosting` (landed) + `tac.compress_time_optimization` (landed) + `tac.inflate_time_post_processing` (in-flight parallel sibling) + `tac.side_information` (in-flight parallel sibling)
**Mission contribution:** `apparatus_maintenance` — infrastructure that unlocks future frontier-breaking moves rather than a direct score improvement.
**horizon_class:** plateau_adjacent — this namespace's effect on contest score is inherited from the substrates that consume it. The infrastructure itself is plateau-adjacent (bolt-on tooling); the substrates it parameterizes are where horizon-class targeting happens.

---

## §1 Mission

Per the operator's standing directive from the §7.6 spec:

> *"All techniques should be available through helpers even if not applicable immediately because we have several other things to build and implement and iterate on so each is uniquely and individually fully and completely extreme and absolute optimization."*

This namespace canonicalizes the SEARCH STRATEGY surface across the contest. Search strategies are mathematically distinct from optimization passes (which live in `tac.compress_time_optimization`):

- **Optimization PASSES** iterate to convergence on ONE parameter setting (gradient descent / multipass refinement / iterated bisection).
- **Search STRATEGIES** propose CANDIDATES and rank them (CMA-ES / Optuna TPE / Bayesian Optimization / MCTS / Rashomon committee).

When a sister pipeline writes `pipeline @ "cma_es_over_K_palette"` the `@` operator stores an opaque descriptor. This namespace provides the ENGINE that resolves the descriptor and executes the actual search via `run_search_over_pipeline(pipeline, objective_fn)`.

---

## §2 Public API

Narrow per CLAUDE.md "Beauty, simplicity, and developer experience":

```python
# Decorator + contract
from tac.search import search_strategy, SearchContract

@search_strategy(SearchContract(
    id="cma_es_over_palette_k",
    search_kind="continuous",
    n_candidate_evaluations_max=200,
    parallelism="vectorized",
    requires_objective_function=True,
    objective_is_surrogate=False,
    seed=42,
    predicted_search_cost_usd=2.50,
    hook_autopilot_ranker="cathedral_autopilot_v1",
    hook_continual_learning_anchor_kind=(
        "search_strategy_outcomes_v1"
    ),
    ...
))
def cma_es_over_palette_k(objective_fn, *, bounds, seed=42, ...):
    """Run CMA-ES; return best params + history."""
    ...
    return SearchResult(...)

# Composition via the `@` operator (lives on sister pipelines)
from tac.boosting import ComposableBoostingPipeline
pipeline = ComposableBoostingPipeline() | "raw_decoder" | "cascade_pose_residual"
pipeline_with_search = pipeline @ "cma_es_over_palette_k"

# Execute via the canonical helper
from tac.search import run_search_over_pipeline
result = run_search_over_pipeline(
    pipeline_with_search,
    objective_fn=my_objective,
    bounds={"K": (4, 64), "sigma": (0.5, 5.0)},
)
print(result.best_params, result.best_score, result.n_evaluations)
```

### Public symbols

```
# Decorator + registry
search_strategy(contract)               # decorator
SearchContract                          # frozen dataclass contract
get_registered_strategies()             # id → contract map
get_strategy_function(strategy_id)      # id → callable
validate_all_registered_strategies()    # re-validate registry
_REGISTERED_STRATEGIES                  # mutable registry dict (tests)
_clear_strategy_registry_for_tests()    # test fixture helper

# Errors
SearchNamespaceError                    # root exception
SearchContractError                     # contract validation
DeterminismViolation                    # seed missing
SeedRequiredViolation                   # deterministic=True + seed=None
SearchEngineNotInstalledError           # cma / optuna / botorch absent
SearchBudgetExceededError               # n_evaluations > budget
ObjectiveFunctionError                  # objective fn raised / non-finite

# Composition (the engine for the `@` operator on sister pipelines)
ComposableSearchPipeline                # standalone composition + .run()
SearchResult                            # frozen dataclass
SearchHistory                           # frozen dataclass tuple of trial rows
run_search_over_pipeline                # canonical helper for `@` op execution

# Builders + specs
CMAESCandidateSearcher / CMAESCandidateSearcherSpec
OptunaTPESampler       / OptunaTPESamplerSpec
BayesianOptimizationGP / BayesianOptimizationGPSpec
MCTSCodebookSearcher   / MCTSCodebookSearcherSpec
RashomonEnsembleCommittee / RashomonEnsembleCommitteeSpec
LEGAL_SEARCH_KIND / LEGAL_PARALLELISM / LEGAL_HOOK_*  # enums

# Persistence (opt-in)
append_search_outcome_locked(record)
load_search_outcomes / load_search_outcomes_strict
SearchLedgerCorruptError
SEARCH_STRATEGY_OUTCOMES_PATH / _LOCK / _SCHEMA_VERSION
```

---

## §3 Composition primitives (§5.4 from the spec)

This namespace IS the engine for the `@` operator on sister pipelines. Composition primitives:

- `pipeline @ "<strategy_id>"` — sister-pipeline `__matmul__` stores the
  descriptor; runtime resolution happens via `run_search_over_pipeline`.
- `ComposableSearchPipeline()` — STANDALONE pipeline (no sister-pipeline
  required) — useful for searching over plain objective functions
  outside a substrate pipeline; supports `|` for sequential strategy
  chaining (e.g. RandomSearch warmup | TPE refinement).
- `&` parallel-merge of strategies — ensemble disagreement (Rashomon-style).

---

## §4 Unique-to-this-namespace composition fields (§F)

Per PV-7 + PV-9 the SearchContract has distinguishing fields:

- `search_kind: Literal[continuous, discrete, mixed, multi_objective]` — the
  engine selection depends on it. Mixed = TPE (handles both); pure
  continuous = CMA-ES / BoTorch GP; pure discrete = MCTS; multi_objective
  = NSGA-II (deferred to a future builder slot).
- `n_candidate_evaluations_max: int` — the search budget; the canonical
  rate-limiter for "stop after N candidate evaluations".
- `parallelism: Literal[serial, vectorized, process_pool]` — pinned at
  contract time so callers know whether `process_pool` is safe (Random,
  CMA-ES with vectorized eval); serial-only for MCTS.
- `requires_objective_function: bool` — hard-pinned True for ALL search
  strategies (any False raises SearchContractError).
- `objective_is_surrogate: bool` — routes the objective fn through the
  CPU-trained Hinton surrogate (per Catalog #527) vs the real contest
  scorer; the autopilot ranker uses this to weight the proposed
  candidate's confidence.
- `seed: int | None` — required by the deterministic contract guard.

---

## §5 Production-hardening contracts (§5.5)

Mirrors the canonical contract enforced at decoration time:

- Frozen dataclass contracts; type-hinted; schema-validated at import
- Schema validation handles both `ast.Assign` and `ast.AnnAssign`
  (Catalog #168 META-class — sister-checked via the test suite)
- `deterministic=True` required for any strategy that's part of an
  archive-bytes derivation chain (CMA-ES + Optuna TPE in regression-tests
  require seed pinning; Rashomon ensemble has implicit seed from
  bootstrap)
- fcntl-locked JSONL persistence for strategy outcomes (Catalog #128 /
  #131 sister)
- 6-hook wire-in declared at decoration (Catalog #125)
- Malformed strategies fail at decoration time (import error), not at
  dispatch — the contest race-mode rigor inversion is honored

---

## §6 Builders — when to use which

| Builder | When to use | External library |
|---|---|---|
| `CMAESCandidateSearcher` | continuous parameter manifolds (CMA-ES is the SOTA for non-convex continuous optimization with population-based search) | `cma` (lazy import) |
| `OptunaTPESampler` | mixed continuous + discrete; many trials (TPE handles both gracefully + pruning support) | `optuna` (lazy import) |
| `BayesianOptimizationGP` | continuous parameter manifolds + expensive objectives (≤ 100 trials; GP kernel learns) | `botorch` or `scikit-optimize` (lazy import) |
| `MCTSCodebookSearcher` | discrete codebook search (per-pair selector indices, per-block quantization scales); state-space too large for exhaustive enumeration | in-house (no external dep) |
| `RashomonEnsembleCommittee` | candidate ranking by K=8 model agreement; HIGH-disagreement candidates are the next probe targets (per Catalog #252 sister) | reuses `tac.autopilot_rudin_daubechies.RashomonEnsembleRanker` |

---

## §7 Persistence (opt-in)

Canonical JSONL ledger at `.omx/state/search_strategy_outcomes.jsonl`.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #128/#131 sister: the persistence helper acquires `fcntl.flock(LOCK_EX)`, strict-loads existing rows (raises SearchLedgerCorruptError on malformed JSONL), appends the new row, atomic-writes via `unique-tmp + os.replace`. Quarantine on corrupt mirrors the canonical pattern.

Schema:
```
{
  "schema_version": "search_strategy_outcomes_v1",
  "strategy_id": "cma_es_over_palette_k",
  "objective_function_label": "fec6_proxy_score_macos_cpu_advisory",
  "best_params": {"K": 16, "sigma": 1.5},
  "best_score": 0.1928,
  "score_axis": "[macOS-CPU advisory only]",
  "n_evaluations": 200,
  "elapsed_seconds": 845.3,
  "search_budget_envelope_used_pct": 100.0,
  "predicted_search_cost_usd": 2.50,
  "actual_cost_usd": 0.0,
  "written_at_utc": "2026-05-17T...",
  "written_pid": 12345,
  "written_host": "Primary"
}
```

---

## 9-dimension success checklist evidence

(canonical header above per Catalog #294; numbered cross-reference: §8)

## §8 9-dimension success checklist evidence

Per CLAUDE.md Catalog #294 (the 9-dim success checklist evidence section):

1. **UNIQUENESS** — Tooling helper; uniqueness is inherited from the substrates that consume it. The namespace itself is a CANONICAL helper (intentionally shared).
2. **BEAUTY + ELEGANCE** — Public API is ~10 symbols; total namespace ~2,000 LOC across 12 files. Pure Python; no opaque ABC hierarchy. Decorator + dataclass + dict semantics throughout. PR101-style 30-second review per file.
3. **DISTINCTNESS** — Distinct from `tac.compress_time_optimization` because search strategies PROPOSE candidates (no iteration to convergence); distinct from `tac.boosting` because the unit of composition is a SEARCH STRATEGY, not a residual stage.
4. **RIGOR** — Premise verification (Catalog #229) captured at `.omx/tmp/tac_search_premise_verifier.txt`. ~95 tests cover contract validation + decorator behavior + pipeline composition + all 5 builders + SeedRequiredViolation + budget enforcement + Rashomon re-export integrity + surrogate-vs-real routing. Per-builder regression tests pin the canonical-vs-unique decisions per layer.
5. **OPTIMIZATION PER TECHNIQUE** — Covered by sister Catalog #290 — each builder's `## Canonical-vs-unique decision per layer` is documented in §10 below.
6. **STACK-OF-STACKS-COMPOSABILITY** — The `@` operator on sister pipelines integrates this namespace into ANY substrate composition. `ComposableSearchPipeline` supports `|` chaining (RandomSearch warmup | TPE refinement) for ensemble strategies.
7. **DETERMINISTIC REPRODUCIBILITY** — `SeedRequiredViolation` raised at decoration when `deterministic=True` + `seed=None` + signature has no `seed` parameter. Byte-stable JSONL persistence; sorted keys in `to_dict`; frozen dataclasses; same seed → same trial sequence for CMA-ES + Optuna TPE.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — `parallelism: process_pool` supported (CMA-ES + Random); search budget enforced before paid GPU dispatch. Lazy import of external libraries means import-time cost ≈ 0 even when `cma` / `optuna` / `botorch` are absent.
9. **OPTIMAL MINIMAL CONTEST SCORE** — This namespace's contribution to contest score is INHERITED from the substrates that consume it. The §F search-method coverage gap maps directly to fec6 (K-palette sweep) + PR106 (latent dimension search) + every future substrate's hyperparameter manifold.

---

## Cargo-cult audit per assumption

(canonical header above per Catalog #303; numbered cross-reference: §9)

## §9 Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md non-negotiable + Catalog #303: enumerate every substrate-design assumption inherited from sister patterns. Each receives HARD-EARNED-vs-CARGO-CULTED classification.

| Assumption | Classification | Rationale + unwind path |
|---|---|---|
| Frozen kw_only dataclass contract | HARD-EARNED | Sister `tac.boosting` + `tac.compress_time_optimization` proven; immutability prevents back-door mutation per Q2 adversarial-review pattern. |
| @decorator(contract) registry pattern | HARD-EARNED | Sister sibling; allows declarative search registration. |
| `|` operator for sequential composition (on ComposableSearchPipeline) | HARD-EARNED | Sister sibling; enables RandomSearch warmup → TPE refinement chains. |
| `@` operator stores descriptor as opaque string on sister pipelines | HARD-EARNED | Sister sibling; documented as intentional indirection so the search engine evolves independently. |
| 6-hook wire-in via NOT_APPLICABLE_WITH_RATIONALE sentinel | HARD-EARNED | Catalog #125 sister; forces explicit declaration per hook. |
| Fcntl-locked JSONL persistence at `.omx/state/search_strategy_outcomes.jsonl` | HARD-EARNED | Catalog #128/#131 sister; proven concurrency-safe across multi-process search. |
| `SearchContract.id` must match `/^[a-z][a-z0-9_]*$/` (snake_case) | HARD-EARNED | Sister sibling; rejects kebab-case at decoration; matches Python identifier conventions. |
| `n_candidate_evaluations_max` as integer budget (not wallclock) | UNIQUE-FORK | Search strategies are evaluation-bound (each candidate eval = 1 objective call); wallclock varies per objective. Sister `tac.compress_time_optimization` uses wallclock; this namespace forks to evaluation count because the canonical search budget IS evaluation count. |
| `parallelism: Literal[serial, vectorized, process_pool]` enum | UNIQUE-FORK | Search strategies have FUNDAMENTAL parallelism differences (MCTS = serial; CMA-ES = vectorized; Random = process_pool). Sister namespaces don't have this concept. |
| Builders WRAP external libraries (cma / optuna / botorch) instead of canonical-implementing | HARD-EARNED | These libraries are production-grade; re-implementing would be cargo-cult per operator's "consolidate everything into canonical helpers" directive (which means USE canonical helpers, including external ones). |
| Rashomon ensemble is RE-EXPORTED from `tac.autopilot_rudin_daubechies` (not re-implemented) | HARD-EARNED | Catalog #252 + operator's standing directive `feedback_canonical_share_when_serves_unique_when_suppresses`. The K=8 SLIM ensemble IS canonical; sharing maintains continual-learning anchor coherence. |
| Lazy-import external libraries inside `.run()` | HARD-EARNED | Module import-time cost = 0 when external lib is absent; only `.run()` raises `SearchEngineNotInstalledError`. Tests use pure-Python stubs so the namespace is importable without `cma` / `optuna` / `botorch`. |
| Objective function ALWAYS required (`requires_objective_function: bool` hard-pinned True) | HARD-EARNED | Search without an objective is undefined; the contract refuses False at validation. |
| `objective_is_surrogate: bool` routes through Hinton surrogate | HARD-EARNED | Per Catalog #527 surrogate; the field is the canonical routing decision; consumers know whether the proposed candidate needs a real-contest-scorer verification pass before dispatch. |

---

## §10 Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Contract dataclass shape | ADOPT_CANONICAL_BECAUSE_SERVES | Frozen kw_only dataclass mirrors sister sibling; no fork. |
| Decorator registry pattern | ADOPT_CANONICAL_BECAUSE_SERVES | Sister sibling; no fork. |
| Pipeline composition operators (`|` `&`) | ADOPT_CANONICAL_BECAUSE_SERVES | Sister sibling; documented intentional reuse. |
| `@` operator | FORK_BECAUSE_PRINCIPLED_MISMATCH | Sister pipelines STORE descriptor; this namespace RESOLVES + EXECUTES it via `run_search_over_pipeline`. |
| `SearchContract.id` pattern | ADOPT_CANONICAL_BECAUSE_SERVES | snake_case mirrors sister sibling. |
| `search_kind` enum | UNIQUE-FORK | Distinct concept; needed for engine selection. |
| `n_candidate_evaluations_max` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Search budget is evaluation-bound, not wallclock-bound (sister namespaces use wallclock). |
| `parallelism` enum | UNIQUE-FORK | Search strategies have fundamental parallelism differences. |
| `requires_objective_function` (hard-pin True) | UNIQUE-FORK | All search strategies need an objective; the field exists only for serialization + explicit contract declaration. |
| `objective_is_surrogate` | UNIQUE-FORK | Routing decision specific to search (sister namespaces don't invoke the scorer at all). |
| `seed` validation | ADOPT_CANONICAL_BECAUSE_SERVES | SeedRequiredViolation pattern mirrors sister sibling. |
| 6-hook wire-in | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #125 sister. |
| Fcntl-locked JSONL persistence | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #128/#131 sister. |
| LEGAL_* frozensets | ADOPT_CANONICAL_BECAUSE_SERVES | Sister sibling; immutable enum-like membership. |
| Error hierarchy | ADOPT_CANONICAL_BECAUSE_SERVES | Sister sibling; single root + typed children. |
| CMAESCandidateSearcher (wraps `cma`) | UNIQUE-FORK | External library wrapper; lazy import; sister namespaces don't wrap externals. |
| OptunaTPESampler (wraps `optuna`) | UNIQUE-FORK | External library wrapper; lazy import. |
| BayesianOptimizationGP (wraps `botorch`/`skopt`) | UNIQUE-FORK | External library wrapper; lazy import. |
| MCTSCodebookSearcher (in-house) | UNIQUE-FORK | In-house implementation; no external dep; pure Python MCTS for discrete codebook search. |
| RashomonEnsembleCommittee (re-export) | ADOPT_CANONICAL_BECAUSE_SERVES | Reuses `tac.autopilot_rudin_daubechies.RashomonEnsembleRanker` (Catalog #252 sister); shared continual-learning anchor pool. |

---

## Observability surface

(canonical header above per Catalog #305; numbered cross-reference: §11)

## §11 Observability surface

Per CLAUDE.md Catalog #305 (the 6-facet observability definition):

1. **Inspectable per layer** — `ComposableSearchPipeline.to_dict()` returns the full pipeline structure; `SearchResult.history` is a tuple of per-trial dicts including params + score + elapsed + axis tag.
2. **Decomposable per signal** — `SearchResult` decomposes `best_score` into per-trial contributions via `history`; `best_params` is a dict so each parameter dimension is independently auditable.
3. **Diff-able across runs** — Two `SearchResult` objects with same `strategy_id` + same `seed` + same `objective_function_label` MUST produce identical `history` (byte-stable JSON dump comparison); regression tests pin this for CMA-ES + Optuna TPE.
4. **Queryable post-hoc** — `load_search_outcomes()` reads all persisted ledger rows; `query_outcomes_by_strategy_id(id)` + `query_outcomes_by_objective_label(label)` + `latest_best_score_by_strategy(id)` canonical query helpers.
5. **Cite-able** — Every outcome row carries `written_at_utc`, `written_pid`, `written_host`, optional `lane_id`, optional `dispatch_call_id` so the result chains to the Modal call_id ledger (Catalog #245).
6. **Counterfactual-able** — `ComposableSearchPipeline.with_seed(new_seed)` returns a new pipeline; running it produces a counterfactual `SearchResult` for "what if we'd seeded differently?"; the `@` integration with sister pipelines means we can also probe "what if we'd searched a different parameter dimension?".

The 6-facet contract is enforced at design time (this memo) and at run time (every public-API call records to the canonical ledger when persistence is opted in).

---

## §12 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Rationale / wire-in detail |
|---|---|---|
| Sensitivity-map contribution | N/A unless `sensitivity_weighted=True` | When the contract declares `sensitivity_weighted=True`, the strategy auto-consumes `master_gradient` and weights candidate proposals by per-byte sensitivity. Otherwise N/A with rationale: "Strategy is sensitivity-blind by design (RandomSearch / vanilla TPE)." |
| Pareto constraint | ACTIVE for `multi_objective` strategies | NSGA-II-style strategies expose `pareto_front` in `SearchResult`. Single-objective strategies declare N/A with rationale: "single-objective; Pareto undefined." |
| Bit-allocator hook | N/A | Search strategies discover PARAMETER VALUES, not bit allocations. Bit allocation is downstream (per Catalog #275 sister). Rationale documented per hook. |
| Cathedral autopilot dispatch hook | ACTIVE | `hook_autopilot_ranker="cathedral_autopilot_v1"` is the canonical hook. Discovered best candidates flow into `tools/cathedral_autopilot_autonomous_loop.py` as CandidateRow proposals with `predicted_dispatch_risk` from `predicted_search_cost_usd`. |
| Continual-learning posterior update | ACTIVE | `hook_continual_learning_anchor_kind="search_strategy_outcomes_v1"` is the canonical persistence kind. Every `run_search_over_pipeline` outcome appends to `.omx/state/search_strategy_outcomes.jsonl`. |
| Probe-disambiguator | ACTIVE for Rashomon ensemble | The Rashomon K=8 disagreement queue IS the canonical probe-disambiguator (per Catalog #252 sister). Single-strategy builders declare N/A with rationale. |

---

## §13 Mandatory assumption statement (Catalog #292)

> The shared assumption I am operating within for this design is: **the canonical-helper-share pattern from sister namespaces (tac.boosting, tac.compress_time_optimization) is the optimal engineering for this namespace because search strategies are by-construction CANONICAL infrastructure shared across all substrates — there is no per-method engineering to fork at the namespace level.**

Assumption-Adversary verdict: **HARD-EARNED**. The 0.196-0.199 plateau cited in `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` was a SUBSTRATE-level shared-assumption issue (canonical scorer-preprocess + EMA decay + Tier-1 defaults). At the namespace-helper layer the operator's directive `feedback_canonical_share_when_serves_unique_when_suppresses` EXPLICITLY calls out that helpers should be canonical when they serve.

The DISTINCT IMPLEMENTATION decision happens INSIDE each builder (whether to wrap an external library vs implement in-house; whether to fork the canonical objective routing vs reuse). Those decisions ARE documented in §9 + §10 per `## Canonical-vs-unique decision per layer`.

---

## §14 Premise verification evidence (Catalog #229)

Per CLAUDE.md "Premise verification before edit" + Catalog #229:

- Captured at `.omx/tmp/tac_search_premise_verifier.txt`
- 16 PVs (PV-1 through PV-16) verified pre-edit
- Sister namespace API surface read (PV-1)
- Spec §F + §5.2 + §5.4 verified verbatim (PV-2)
- Empirical absence of `tac.meta_lagrangian_search` and `tac.optimization.cmaes_optuna_search` confirmed (PV-3 — overrides parent prompt assertion)
- Rashomon ensemble availability verified (PV-4)
- External library absence confirmed → lazy-import design decision (PV-5)
- `@` operator semantics confirmed (PV-6)
- Canonical-vs-unique decision per layer enumerated (PV-7 + PV-10)
- 5 builders enumeration confirmed (PV-8)
- Contract fields enumerated (PV-9)
- Test pass count target ~95 confirmed (PV-11)
- Catalog #305 literal section header requirement noted (PV-12)
- Catalog #294 9-dim checklist required header noted (PV-13)
- Catalog #309 horizon_class declaration (PV-14 → `plateau_adjacent`)
- Catalog #303 cargo-cult audit section enumerated (PV-15)
- Mission alignment category enumerated (PV-16 → `apparatus_maintenance`)

---

## §15 Worked examples

### Example A: CMA-ES over fec6 K-palette size

```python
from tac.search import (
    CMAESCandidateSearcher, CMAESCandidateSearcherSpec,
    run_search_over_pipeline,
)
from tac.boosting import ComposableBoostingPipeline

# Build the substrate pipeline
pipeline = (
    ComposableBoostingPipeline()
    | "raw_decoder"
    | "fec6_selector_apply"
)

# Build the search contract via the canonical builder
cma_es = CMAESCandidateSearcher(
    spec=CMAESCandidateSearcherSpec(
        strategy_id="cma_es_over_fec6_k_palette",
        bounds={"K": (4.0, 64.0)},  # CMA-ES is continuous; round to int at eval
        population_size=12,
        sigma_init=8.0,
        max_evaluations=200,
        seed=42,
        lane_id="lane_tac_search_namespace_decorator_api_20260517",
    )
).register()  # registers via @search_strategy(contract)

# Attach the search descriptor to the pipeline
pipeline_with_search = pipeline @ "cma_es_over_fec6_k_palette"

# Run the search; the objective evaluates the fec6 substrate at the
# proposed K and returns the contest score
def objective(params):
    K = round(params["K"])
    archive = build_fec6_archive(K=K)
    return evaluate_archive(archive)  # returns contest score

result = run_search_over_pipeline(
    pipeline_with_search, objective_fn=objective
)
print(f"best K = {round(result.best_params['K'])}, score = {result.best_score}")
```

### Example B: Optuna TPE over mixed param manifold

```python
from tac.search import OptunaTPESampler, OptunaTPESamplerSpec

OptunaTPESampler(
    spec=OptunaTPESamplerSpec(
        strategy_id="tpe_fec6_K_plus_lambda_R",
        bounds={
            "K": (4, 64, "int"),
            "lambda_R": (0.001, 0.1, "log_float"),
            "use_per_pair_huffman": (False, True, "bool"),
        },
        n_trials=300,
        n_startup_trials=20,
        multivariate=True,
        seed=42,
        lane_id="lane_tac_search_namespace_decorator_api_20260517",
    )
).register()
```

### Example C: Rashomon ensemble committee ranking candidates

```python
from tac.search import RashomonEnsembleCommittee, RashomonEnsembleCommitteeSpec

committee = RashomonEnsembleCommittee(
    spec=RashomonEnsembleCommitteeSpec(
        strategy_id="rashomon_committee_pr101_candidates",
        ensemble_size=8,
        bootstrap_seed_base=42,
        sparsity_target=4,
        integer_coefficient_bound=10,
        anchor_path=".omx/state/slim_anchor_store.jsonl",
        lane_id="lane_tac_search_namespace_decorator_api_20260517",
    )
).register()

# Rank a list of candidate pipelines by consensus score
ranked = committee.rank_candidates(candidate_pipelines)
# HIGH-disagreement candidates are surfaced as next-probe queue
disagreement_queue = committee.disagreement_queue()
```

---

## §16 Cross-references

- §7.6 spec: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md` §F + §5.2 + §5.4
- Sister design memos:
  - `.omx/research/tac_boosting_namespace_design_20260517.md`
  - `.omx/research/tac_compress_time_optimization_namespace_design_20260517.md`
- Premise verifier: `.omx/tmp/tac_search_premise_verifier.txt`
- Rashomon canonical: `src/tac/autopilot_rudin_daubechies/rashomon_ensemble.py` (Catalog #252 sister)
- Catalog #125 (6-hook wire-in non-negotiable)
- Catalog #128/#131 (fcntl-locked JSONL persistence)
- Catalog #168 (AST walker handles both Assign and AnnAssign)
- Catalog #229 (premise verification before edit)
- Catalog #245 (Modal call_id ledger; sister persistence pattern)
- Catalog #252 (Rashomon ensemble canonical)
- Catalog #270 (canonical dispatch optimization protocol; this namespace integrates into Tier 1 engineering)
- Catalog #275 (sensitivity-map; sister consumer of master_gradient)
- Catalog #290 (canonical-vs-unique decision per layer; §10 above)
- Catalog #292 (per-deliberation assumption surfacing; §13 above)
- Catalog #294 (9-dim success checklist; §8 above)
- Catalog #303 (cargo-cult audit; §9 above)
- Catalog #305 (observability surface; §11 above)
- Catalog #309 (horizon_class declaration; document header)
- Catalog #527 (Hinton-distilled CPU surrogate; `objective_is_surrogate` routing)

---

## §17 Reactivation criteria

Re-read this memo when:
- A new search strategy is being added (use §6 + §10 to decide UNIQUE-FORK vs ADOPT_CANONICAL per layer)
- The cathedral autopilot ranker needs a new candidate source (this namespace's `SearchResult.best_params` is the canonical surface)
- A new substrate's hyperparameter manifold needs systematic exploration (use §15 Example A pattern)
- The `@` operator semantics needs to evolve on sister pipelines (this memo + the sister namespace design memos jointly define the canonical contract)

---

## §18 Mandatory mission-alignment frontmatter (Catalog #300 follow-on, T2+ deliberations only)

This memo is the OUTPUT of a SUBAGENT LANDING, not a T2+ council deliberation. The Catalog #300 frontmatter requirements do not apply. For reference: this namespace's mission contribution category is `apparatus_maintenance` (the canonical 5-category enum) — it's infrastructure that unlocks future substrate gains, not a direct frontier-breaking move.
