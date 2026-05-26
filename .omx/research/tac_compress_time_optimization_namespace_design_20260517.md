# tac.compress_time_optimization namespace — design memo

**Date:** 2026-05-17
**Status:** active design memo; sister of `tac.boosting` namespace
(`.omx/research/tac_boosting_namespace_design_20260517.md`)
**Lane:** `lane_tac_compress_time_optimization_namespace_decorator_api_20260517`
**Spec provenance:** `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5.3-§5.5 + §G compress-time
**Premise verification:** `.omx/tmp/tac_compress_time_optimization_premise_verifier.txt`
(PV-1..PV-8)
**Mirrors:** `tac.boosting` (this is the SECOND of 5 §7.6 missing namespaces)

---

## §1 Mission

Land the SECOND of 5 §7.6 missing canonical-helper namespaces (after
`tac.boosting`). The namespace canonicalizes the 5 compress-time
techniques from spec §G as first-class composable passes:

1. **Generic TTO harness** (extends `optimize_poses.py` to per-byte/
   per-stream/per-pair generic)
2. **Multipass refinement loop** (Lane 8 pattern: quantize → measure →
   re-quantize → re-measure)
3. **Simulated annealing on discrete codes** (selector indices using
   master_gradient as energy landscape)
4. **Per-pair coordinate search** (exhaustive product space across
   mode × palette-entry × pose-delta; M5 Max CPU fan-out)
5. **Iterated bisection on rate-distortion knee** (per-tensor + per-block
   scale bisection)

Per CLAUDE.md §G *"Compress-time compute is effectively unbounded"* — the
namespace's `max_wallclock_seconds=None` is LEGAL by default (unlike
inflate-time which is 30-min capped per Catalog "strict scorer rule"
sister discipline).

Per the operator standing directive — these helpers serve substrates not
yet built. fec6 / pr101 / DP1 / SA02 / NSCS06 / NSCS01 will all inherit
these primitives at compose-time rather than re-implementing.

## §2 Public API

```python
from tac.compress_time_optimization import (
    # Decorator + registry
    compress_time_pass, CompressTimePassContract,
    get_registered_passes, get_pass_function, validate_all_registered_passes,
    # Errors
    CompressTimeOptimizationError, CompressTimePassContractError,
    DeterminismViolation, SeedRequiredViolation, InflatePhaseForbiddenError,
    CompressTimePipelineError, AmbiguousCompositionError,
    RateBudgetViolation, CompressTimeBudgetExceededError,
    CompressTimeLedgerCorruptError,
    # Composition
    ComposableCompressPipeline, PipelineStageRef, CompressTimePipelineResult,
    # Builders (one per §G compress-time row)
    GenericTTOHarness, GenericTTOHarnessSpec,
    MultipassRefinement, MultipassRefinementSpec,
    SimulatedAnnealingOnDiscreteCodes, SimulatedAnnealingSpec,
    PerPairCoordinateSearch, PerPairCoordinateSearchSpec,
    IteratedBisectionRateKnee, IteratedBisectionRateKneeSpec,
    # Persistence (opt-in)
    append_pass_outcome_locked, load_pass_outcomes, load_pass_outcomes_strict,
    COMPRESS_TIME_OPT_PASS_OUTCOMES_PATH,
    COMPRESS_TIME_OPT_PASS_OUTCOMES_LOCK,
    COMPRESS_TIME_OPT_PASS_OUTCOMES_SCHEMA_VERSION,
)
```

Narrow per CLAUDE.md "Beauty, simplicity, and developer experience" —
ONE dataclass (`CompressTimePassContract`), ONE decorator
(`@compress_time_pass`), THREE composition operators (`|`/`&`/`@`), FIVE
builders, ONE persistence helper.

## §3 Composition primitives (§5.4 from the spec)

| operator | meaning | example |
|---|---|---|
| `\|` (sequential) | chains passes | `A \| B` runs A then B |
| `&` (parallel-merge) | runs side-by-side and merges by per-byte policy | `(A \| B) & C` runs B and C in parallel |
| `@` (attach search) | attaches a `tac.search.*` strategy | `pipeline @ "cma_es_K_sweep"` |

Every compose returns a NEW immutable pipeline. Pipelines are
JSON-serializable via `.to_json()` for cathedral autopilot ranking
without instantiation.

**Operator-precedence note (verbatim from tac.boosting):** Python
evaluates `&` and `@` BEFORE `|`, so `A | B & C` parses as `A | (B & C)`.
Use parentheses for `(A | B) & C`. This is the SAME Python language
constraint that tac.boosting documented; surfaced explicitly in tests +
docstrings.

## §4 Unique-to-this-namespace composition filters (§G + Catalog #167)

| filter | semantics | empirical anchor |
|---|---|---|
| `with_rate_budget(bytes=N)` | rejects passes whose cumulative `bytes_added` would exceed N | rate-side guardrail per §G |
| `with_wallclock_budget(seconds=N)` | rejects passes whose cumulative elapsed wallclock would exceed N | smoke-before-full per Catalog #167; default None = unbounded per §G |

The wallclock filter is **opt-in only**. Per CLAUDE.md §G compress-time
is unbounded by default — the operator opts into wallclock budgeting
explicitly. Both filters integrate with `rate_strict=True` /
`wallclock_strict=True` kwargs on `.run()` for fail-closed dispatch
semantics.

## §5 Production-hardening contracts (§5.5)

Per spec §5.5, every pass MUST satisfy these invariants enforced at
DECORATION TIME (NOT dispatch time):

| invariant | enforced by | failure mode without it |
|---|---|---|
| Frozen dataclass + type-hinted throughout | `CompressTimePassContract.__post_init__` | Mid-build mutation of pass parameters |
| Schema validation at import time (`ast.Assign` + `ast.AnnAssign`) | Catalog #168 discipline + dedicated AST regression test | Future introspection tools silently skip type-annotated registries |
| `deterministic=True` for archive-byte-emitting passes | cross-field invariant in `__post_init__` + decorator signature check | Byte-stable archives break per Catalog #158 |
| Fcntl-locked JSONL persistence | `tac.compress_time_optimization.persistence._pass_outcomes_lock` | Catalog #131 sister "bare writes to shared state" |
| `stage_phase='inflate'` FORBIDDEN at decoration | `InflatePhaseForbiddenError` raised in `__post_init__` | Inflate-time stages belong to sister namespace `tac.inflate_time_post_processing` (deferred per spec §5.2) |
| `seed=` required for random passes claiming `deterministic=True` | `DeterminismViolation` / `SeedRequiredViolation` raised by decorator signature inspector | Byte stream non-reproducible across runs |
| 6-hook wire-in declared per Catalog #125 | 5 enum-validated hook fields + `hook_not_applicable_rationale` dict with required keys | Orphaned work; cathedral autopilot ranker doesn't see the pass |
| Decoration-time error vs dispatch-time error | All validators raise at module-import time | Errors at $5-50 dispatch instead of $0 import |

## §6 Builders — when to use which

| builder | §G row | empirical anchor |
|---|---|---|
| `GenericTTOHarness` | row 1 (TTO at compress time) | `experiments/optimize_poses.py` template; PD-V2 pose TTO |
| `MultipassRefinement` | row 2 (Multipass refinement) | `src/tac/multipass_compressor.py` Lane 8 |
| `SimulatedAnnealingOnDiscreteCodes` | row 4 (SA on discrete codes) | `tac.contrib.cross_disciplinary_optimizers` SA primitive |
| `PerPairCoordinateSearch` | row 3 (Per-pair coordinate search) | task #466 / #470 partial implementations; fec6 K=16 |
| `IteratedBisectionRateKnee` | row 6 (Iterated bisection on R-D knee) | NEW; trivial CPU fan-out |

The builders DO NOT subclass each other — per CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD" each builder's contract emission is
substrate-specific enough that subclassing would suppress engineering
choices. They DO share `CompressTimePassContract` as the common output.

## §7 Persistence (opt-in)

`append_pass_outcome_locked` writes to
`.omx/state/compress_time_optimization_pass_outcomes.jsonl` via fcntl
LOCK_EX. Per CLAUDE.md Catalog #128/#131/#132/#138 sister discipline:

- Schema-versioned (`compress_time_optimization_pass_outcomes_v1`)
- APPEND-ONLY per Catalog #132
- STRICT-load via `load_pass_outcomes_strict` raises
  `CompressTimeLedgerCorruptError` on malformed line
- Lenient `load_pass_outcomes` skips malformed lines
- Quarantine-on-corrupt: `<path>.corrupt.<utcstamp>` per Catalog #138
- 4-process spawn-pool stress test verifies safety under concurrency

The namespace does NOT auto-persist — callers explicitly invoke
`append_pass_outcome_locked` only when they want the audit trail.

## §8 9-dimension success checklist evidence

Per CLAUDE.md "9-dimension success checklist" non-negotiable + Catalog #294:

| dim | label | evidence |
|---|---|---|
| 1 | UNIQUENESS | Compress-time-pass namespace with decorator + composition + 5 first-class technique builders — no equivalent canonical helper exists in the repo |
| 2 | BEAUTY + ELEGANCE | Narrow API (1 dataclass, 1 decorator, 3 operators, 5 builders, 1 persistence layer; 12 files; ~2300 LOC for namespace + ~1200 LOC for tests). 102 tests pass in <500 ms. |
| 3 | DISTINCTNESS | Sister of `tac.boosting` at the compress-time-pass surface — DIFFERENT contract shape (24 fields vs boost's 23; `max_wallclock_seconds` / `rate_budget_bytes` / `distortion_budget` / `seed` unique). NO shared parent class. |
| 4 | RIGOR | Premise verification (PV-1..PV-8) BEFORE edit. 102 dedicated tests covering positive + negative + boundary. Sister-regression tests vs tac.boosting + Catalog #1 / #5. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per-method-optimal engineering: each builder forks the contract template where canonical adoption would suppress (per Catalog #290 canonical-vs-unique decision per layer). |
| 6 | STACK-OF-STACKS COMPOSABILITY | `\|` `&` `@` composition; rate/wallclock budget filters; sister namespaces will compose at the pipeline level (compress passes × boost stages × inflate post-process × side-info × search). |
| 7 | DETERMINISTIC REPRODUCIBILITY | `deterministic=True` enforced via decorator signature check; cross-field invariant refuses non-deterministic archive_build passes. Seed pinning + JSON-round-trip preserves pass references byte-for-byte. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Decorator is PASS-THROUGH (zero runtime overhead). Pipeline `run()` is a single loop over passes. Persistence is OPT-IN. Pipeline objects are JSON-serializable for cathedral autopilot ranking without instantiation. `max_wallclock_seconds=None` unique to this namespace — compress-time compute is unbounded per CLAUDE.md §G. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | The namespace itself does not produce a score — it is INFRASTRUCTURE for substrates that will. The 9-dim checklist applies to FUTURE substrates that consume the namespace. |

## §9 Cargo-cult audit per assumption (Catalog #303)

| assumption | HARD-EARNED or CARGO-CULTED? | rationale + unwind path |
|---|---|---|
| Frozen dataclass + field-level validators | **HARD-EARNED** | tac.boosting (just landed), tac.substrate_registry (Catalog #241), tac.deploy.modal.call_id_ledger (Catalog #245), tac.continual_learning (Catalog #128) all use this pattern empirically + successfully across the codebase. |
| Pipe-operator composition (`\|`) | **HARD-EARNED** | sklearn `Pipeline`, PyTorch `nn.Sequential`, scikit-image's pipe — canonical Python composition idiom. The spec §5.3-§5.4 mandates this directly. tac.boosting already validates the pattern. |
| `max_wallclock_seconds=None` default (unbounded) | **HARD-EARNED** | CLAUDE.md §G *"Compress-time compute is effectively unbounded"* — direct quote. Forcing a finite default would suppress the §G optimization opportunity. |
| `stage_phase='inflate'` FORBIDDEN | **HARD-EARNED** | Inflate-time has different constraints (30-min cap; scorer-free; deterministic). The sister namespace tac.inflate_time_post_processing exists to handle that surface; failing-closed at decoration time prevents misuse. |
| Fcntl-locked JSONL persistence | **CARGO-CULTED from tac.boosting's pattern** | The pattern works for low-throughput pass outcomes. OP-10 amortized append is overkill at this cadence. Unwind path: marked OPT-IN; the canonical helper exists if needed but the namespace does NOT force consumers through it. |
| 6-hook wire-in per Catalog #125 | **HARD-EARNED at the namespace surface; CARGO-CULTED per-pass** | The namespace MUST declare hooks (Catalog #125 non-negotiable for new infrastructure). Per-pass 5 of 6 hooks accept `not_applicable_with_rationale` so per-pass opt-outs are trivial. |
| Decorator pass-through (no runtime wrap) | **HARD-EARNED** | tac.boosting + tac.substrate_registry both do this. Wrapping would add per-call overhead with no benefit; the contract is read at import time. |
| Operator precedence of `&` and `@` BEFORE `\|` | **HARD-EARNED** | Python language semantics; cannot be changed. Surfaced explicitly in design memo §3 + docstring + tests so callers add parens. |
| Compress-time random ops MUST accept `seed=` | **HARD-EARNED** | Catalog #158 + Catalog #105/#139 — byte-stable archives require seed-pinned RNG. The decorator's signature inspector enforces. |
| 5 builders (TTO / multipass / SA / coord-search / bisection) chosen as first-class | **HARD-EARNED from §G** | Spec §G directly enumerates these 5 + LoRA per-pair fine-tune (deferred). The 5 chosen are the unblocked ones; LoRA requires a sister namespace + GPU dispatch infra. |

## §10 Canonical-vs-unique decision per layer (Catalog #290)

| layer | DECISION | rationale |
|---|---|---|
| CompressTimePassContract frozen dataclass | ADOPT_CANONICAL | Mirrors BoostStageContract / SubstrateContract patterns. |
| Decorator pass-through + `_REGISTERED_*` dict | ADOPT_CANONICAL | Mirrors `tac.boosting.decorator`. |
| Pipe-operator composition | ADOPT_CANONICAL | Mirrors `tac.boosting.pipeline.ComposableBoostingPipeline.__or__` exactly. |
| `with_rate_budget` / `with_wallclock_budget` filters | FORK_BECAUSE_PRINCIPLED_MISMATCH | tac.boosting has `with_pareto_growth` (rate/distortion tradeoff at the stage surface). Compress-time needs byte / wallclock guardrails — sister filters with different semantics. |
| Persistence: simple full-rewrite (not OP-10 amortized) | FORK_BECAUSE_PRINCIPLED_MISMATCH | call_id_ledger uses OP-10 because it sees high-throughput; pass outcomes are low-throughput. Premature optimization avoided; pattern mirrors tac.boosting persistence. |
| 5 builders (GenericTTOHarness / MultipassRefinement / SAOnDiscreteCodes / PerPairCoordinateSearch / IteratedBisectionRateKnee) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Each emits a different contract shape (TTO refinement / multipass cascade / SA search / per-pair search / R-D-knee bisection). Subclassing would suppress engineering. |
| Example passes (6 canonical) | ADOPT_CANONICAL | Mirror tac.boosting's example_stages.py pattern. |
| Errors: `CompressTimeOptimizationError` root + specific subclasses | ADOPT_CANONICAL | Mirrors `BoostingNamespaceError` + sister errors. |
| `correction_kind` enum | FORK_BECAUSE_PRINCIPLED_MISMATCH | tac.boosting has {additive, multiplicative, gated, replace, passthrough}; this namespace needs {refinement, residual_correction, search, bisection, transform, passthrough} — more appropriate to compress-time semantics. |
| `correction_resolution` enum (extended with per_byte / per_stream / per_tensor) | FORK_BECAUSE_PRINCIPLED_MISMATCH | tac.boosting handles per_frame / per_pair / per_pixel / per_block / global. Compress-time adds per_byte / per_stream / per_tensor for §G use cases. |
| `merge_policy` enum | ADOPT_CANONICAL | Same enum (last_writer_wins, first_writer_wins, additive, concatenate, explicit). |
| `max_wallclock_seconds`, `rate_budget_bytes`, `distortion_budget`, `seed` fields | UNIQUE | tac.boosting has none of these — compress-time-unique per §G + Catalog #158. |

## §11 Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable":

| facet | how to observe |
|---|---|
| Inspectable per layer | `pipeline.pass_contracts()` returns the tuple of contracts in order; each contract's `to_dict()` is a flat dict |
| Decomposable per signal | `CompressTimePipelineResult.per_pass_outcomes` is a tuple of per-pass dicts with `pass_id` / `status` / `emitted_keys` / `elapsed_seconds` / `bytes_added` / `cumulative_bytes` |
| Diff-able across runs | `pipeline.to_json()` produces byte-stable representation (sort_keys=True); two runs of the same pipeline against the same seed_state are byte-identical when `deterministic=True` passes are used |
| Queryable post-hoc | `load_pass_outcomes(path)` reads the persisted ledger; rows are JSON dicts with `pass_id` / `status` / `pipeline_id` (caller-attached) / `written_at_utc` / `written_pid` / `written_host` |
| Cite-able | Each persisted outcome row carries `written_at_utc` + `written_pid` + `written_host`; combined with the upstream `call_id` (when caller attaches) the full provenance is preserved |
| Counterfactual-able | The decorator's pass-through nature means swapping a pass's contract requires editing one source file + restarting the process; byte-mutation discipline (Catalog #139) applies to archive bytes that downstream codec passes emit |

## §12 6-hook wire-in declaration (Catalog #125)

| hook | wire-in status | rationale |
|---|---|---|
| 1. Sensitivity-map contribution | **ACTIVE — opt-in per pass** | `CompressTimePassContract.hook_sensitivity_contribution` field with `LEGAL_HOOK_SENSITIVITY` enum admitting `master_gradient_v1` / `scorer_conditional_entropy_map_v1` / `axis_weights_v1`. When `sensitivity_weighted=True` the pipeline auto-threads `master_gradient` into the pass function's kwargs. |
| 2. Pareto constraint | **ACTIVE — canonical** | `CompressTimePassContract.hook_pareto_constraint` field defaults to `"rate_distortion_v1"`. Pareto filter integration is the cross-cutting compose-with `tac.boosting.ParetoFrontTracker` pattern. |
| 3. Bit-allocator hook | **ACTIVE — opt-in per pass** | `CompressTimePassContract.hook_bit_allocator_class` field with `LEGAL_HOOK_BIT_ALLOCATOR` enum extended with `iterated_bisection` for the IteratedBisectionRateKnee builder. |
| 4. Cathedral autopilot dispatch hook | **ACTIVE — pipeline-serialization-enabled** | `pipeline.to_json()` produces a stable serialized form the cathedral autopilot can rank without instantiating. `hook_autopilot_ranker` field declares ranker class. |
| 5. Continual-learning posterior update | **ACTIVE — opt-in via persistence helper** | `append_pass_outcome_locked` is the canonical writer; consumers explicitly invoke per run. Fcntl-locked per Catalog #128/#131. |
| 6. Probe-disambiguator | **ACTIVE — opt-in per pass with rationale** | `CompressTimePassContract.hook_probe_disambiguator` field; if None, `hook_not_applicable_rationale['hook_probe_disambiguator']` MUST carry non-empty rationale (enforced by `__post_init__`). SA builder declares `tools/probe_sa_temperature_schedule_disambiguator.py` as a real probe path (4-defensible interpretations: exp / linear / log / adaptive). |

## §13 Mandatory assumption statement (Catalog #292)

The shared assumption I am operating within for this design:

> "A decorator-based composable API with frozen-dataclass contracts +
> pipe-operator composition + 5 first-class compress-time technique builders
> + rate/wallclock budget filters is the OPTIMAL engineering for a
> compress-time-optimization namespace that future substrates will compose
> with at design time."

This assumption is HARD-EARNED (per §9 cargo-cult audit) by:
- Empirical success of the tac.boosting / tac.substrate_registry / tac.deploy.modal.call_id_ledger / tac.continual_learning canonical patterns
- Spec §5.3-§5.5 + §G compress-time direct mandate
- Python language-canonical operator semantics for composition (sklearn / PyTorch)
- The 5 §G compress-time techniques are operator-named first-class
  primitives, not arbitrary choices

If the assumption is wrong, the failure mode is: the pipe-operator API is
over-engineered for early use cases that only chain 2-3 passes.
Mitigation: `ComposableCompressPipeline.from_pass_ids([...])` provides
the imperative form for callers who prefer not to chain.

## §14 Premise verification evidence (Catalog #229)

Premise verifier landed at
`.omx/tmp/tac_compress_time_optimization_premise_verifier.txt` covering 8
premises:

| premise | what we verified |
|---|---|
| PV-1 | Sister `tac.boosting` namespace API surface (the mirror anchor) — internalized via direct read of all 8 namespace files |
| PV-2 | `experiments/optimize_poses.py` template structure documented (2674 lines); GenericTTOHarness generalizes without duplicating |
| PV-3 | `src/tac/multipass_compressor.py` Lane 8 sister exists; MultipassRefinement is the canonical loop primitive (future op-routable: deduplicate the sister's loop into this namespace) |
| PV-4 | `fcntl`-locked JSONL append pattern in `tac.boosting.persistence` mirrored verbatim |
| PV-5 | Catalog #158 deterministic enforcement at contract validation; `max_wallclock_seconds=None` unique to this namespace |
| PV-6 | Catalog #168 AST handling via dataclass declarations (test case verifies) |
| PV-7 | Per-layer canonical-vs-unique decisions documented (10 ADOPT, 8 FORK, 3 SEMANTIC-FORK) |
| PV-8 | Cross-namespace composition stub plan documented (tac.boosting + this namespace co-exist; future hybrid pipeline composition deferred) |

## §15 Worked examples (3)

### Example 1: fec6 compress refinement pipeline

```python
from tac.compress_time_optimization import (
    ComposableCompressPipeline,
    GenericTTOHarness, GenericTTOHarnessSpec,
    MultipassRefinement, MultipassRefinementSpec,
    SimulatedAnnealingOnDiscreteCodes, SimulatedAnnealingSpec,
    compress_time_pass,
)

# Build canonical contracts via the builders
tto_spec = GenericTTOHarnessSpec(
    pass_id="fec6_tto_pose_refinement",
    target_kind="parameter_tensor",
    num_steps=200,
    seed=42,
    sensitivity_weighted=True,
)
mp_spec = MultipassRefinementSpec(
    pass_id="fec6_multipass_quant_depth_3",
    depth=3,
    residual_termination_threshold=1e-4,
    seed=42,
)
sa_spec = SimulatedAnnealingSpec(
    pass_id="fec6_sa_selector_indices",
    discrete_target="selector_indices",
    num_steps=10000,
    seed=42,
    sensitivity_weighted=True,
)

@compress_time_pass(GenericTTOHarness(spec=tto_spec).build_contract())
def fec6_tto_pose_refinement(state, *, master_gradient, policy, seed):
    # 200-step AdamW on pose params; sensitivity-weighted via master_gradient
    ...
    return {"archive_bytes_v1": ..., "bytes_added": delta}

# Decorate the other two stages similarly...

# Compose + run with rate budget
pipeline = (
    ComposableCompressPipeline()
    | "raw_quant"
    | "fec6_tto_pose_refinement"
    | "fec6_multipass_quant_depth_3"
    | "fec6_sa_selector_indices"
).with_rate_budget(bytes=200_000)

result = pipeline.run(
    seed_state={"seed_archive": raw_bytes},
    master_gradient=master_gradient_anchor,
)
```

### Example 2: PD-V2 pose TTO at compress time (extends optimize_poses.py)

```python
spec = GenericTTOHarnessSpec(
    pass_id="pd_v2_pose_tto_per_pair",
    target_kind="parameter_tensor",
    optimizer="adamw",
    learning_rate=1e-3,
    num_steps=500,
    seed=42,
    sensitivity_weighted=False,
    correction_resolution="per_pair",
    description=(
        "Per-pair pose TTO at compress time (PD-V2 pattern); 500 steps "
        "AdamW per pair using substrate's loss_callable that wraps "
        "differentiable scorers."
    ),
    lane_id="lane_pd_v2_pose_tto_20260601",
)
contract = GenericTTOHarness(spec=spec).build_contract()

@compress_time_pass(contract)
def pd_v2_pose_tto_per_pair(state, *, policy, seed):
    # Substrate-specific: load pose_params from state, run AdamW loop
    # with custom loss_callable that wraps differentiable PoseNet/SegNet
    # at compress time (eval_roundtrip applied).
    ...
    return {"archive_bytes_v1": new_archive, "bytes_added": delta}
```

### Example 3: per-block bisection on int8 scale (NSCS01 compress refinement)

```python
spec = IteratedBisectionRateKneeSpec(
    pass_id="nscs01_per_block_int8_scale_bisect",
    granularity="per_block",
    num_outer_iterations=4,
    max_inner_iterations=20,
    convergence_tolerance=1e-4,
    scale_range_log10=(-3.0, 1.0),
    seed=42,
    description=(
        "Per-block int8 scale bisection for NSCS01 nullspace-split "
        "renderer; 4 outer × 20 inner iterations per block."
    ),
    lane_id="lane_nscs01_compress_bisect_20260601",
)
contract = IteratedBisectionRateKnee(spec=spec).build_contract()

@compress_time_pass(contract)
def nscs01_per_block_int8_scale_bisect(state, *, policy, seed):
    # Substrate-specific: per-block bisection on rate-distortion knee
    # using NSCS01's null-space-split-aware evaluator.
    ...
    return {"archive_bytes_v1": ..., "bytes_added": delta}
```

## §16 Cross-references

- Spec: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
  (§5.3-§5.5 contract design + §G compress-time techniques)
- Sister namespace: `tac.boosting` (just landed — same decorator + dataclass + composition pattern)
- Empirical anchor (TTO): `experiments/optimize_poses.py` template
- Empirical anchor (multipass): `src/tac/multipass_compressor.py` Lane 8
- Canonical persistence pattern: `tac.deploy.modal.call_id_ledger`
- Premise verifier: `.omx/tmp/tac_compress_time_optimization_premise_verifier.txt`
- Landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_tac_compress_time_optimization_namespace_landed_20260517.md`
- Next namespace queued: `tac.inflate_time_post_processing` per spec §5.2 build sequence

## §17 Reactivation criteria

Re-read this memo when:
- A new substrate wants to use `@compress_time_pass` (consult §6 builders + §15 examples)
- Sister namespace (`tac.inflate_time_post_processing` / `tac.side_information` / `tac.search`) is being scaffolded (use the same decorator + dataclass + composition pattern; reference §5.3-§5.5 in the spec)
- The cathedral autopilot ranker needs to rank compress-time pipelines (consume via `pipeline.to_json()`; persist via `append_pass_outcome_locked`)
- A council deliberation asks "which compress-time techniques compose with which substrates" (consult §15 worked examples)
- The `tac.multipass_compressor` sister wants deduplication (consult PV-3 future op-routable)


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
