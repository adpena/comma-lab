# tac.boosting namespace — design memo

**Date:** 2026-05-17
**Status:** active design memo; consumed by sister namespace subagents
(`tac.compress_time_optimization` / `tac.inflate_time_post_processing` /
`tac.side_information` / `tac.search` per spec §5.2 build queue)
**Lane:** `lane_tac_boosting_namespace_decorator_api_20260517`
**Spec provenance:** `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5.3-§5.5 + §I.6
**Premise verification:** `.omx/tmp/tac_boosting_premise_verifier.txt` (PV-1..PV-6)

---

## §1 Mission

Land the FIRST of 5 §7.6 missing canonical-helper namespaces. The
namespace is the FOUNDATION other §7.6 namespaces will compose with
(`tac.compress_time_optimization`, `tac.inflate_time_post_processing`,
`tac.side_information`, `tac.search`). Design with that future composition
in mind.

Per the operator standing directive *"All techniques should be available
through helpers even if not applicable immediately because we have several
other things to build and implement and iterate on so each is uniquely and
individually fully and completely extreme and absolute optimization"* — the
namespace serves substrates not yet built. The empirical anchor (PR106
format0d's 2-pass additive correction) is the PROOF-OF-CONCEPT; the
namespace generalizes it.

## §2 Public API

```python
from tac.boosting import (
    # Decorator + registry
    boost_stage, BoostStageContract,
    get_registered_stages, get_stage_function, validate_all_registered_stages,
    # Errors
    BoostingNamespaceError, BoostStageContractError,
    DeterminismViolation, ScorerFreedomViolation,
    BoostingPipelineError, AmbiguousCompositionError,
    BoostingLedgerCorruptError,
    # Composition
    ComposableBoostingPipeline, PipelineStageRef, BoostingPipelineResult,
    # Builders
    ResidualCascadeBuilder, ResidualCascadeStageSpec,
    PerPairDecoderEnsembleSelector, PerPairDecoderEnsembleSpec,
    ModeEnsembleDispatch, ModeEnsembleDispatchSpec,
    # Pareto frontier
    ParetoFrontTracker, ParetoAnchor, ParetoFrontTrackerError,
    # Persistence (opt-in)
    append_stage_outcome_locked, load_stage_outcomes, load_stage_outcomes_strict,
    BOOSTING_STAGE_OUTCOMES_PATH, BOOSTING_STAGE_OUTCOMES_LOCK,
    BOOSTING_STAGE_OUTCOMES_SCHEMA_VERSION,
)
```

The API is narrow per CLAUDE.md "Beauty, simplicity, and developer
experience" — ONE dataclass (`BoostStageContract`), ONE decorator
(`@boost_stage`), THREE composition operators (`|` `&` `@`), THREE
builders, ONE Pareto tracker, ONE persistence helper. Every callsite
visible.

## §3 Composition primitives (§5.4 from the spec)

| operator | meaning | example |
|---|---|---|
| `\|` (sequential) | chains stages | `A \| B` runs A then B |
| `&` (parallel-merge) | runs side-by-side and merges by per-byte policy | `A & B` runs A and B in parallel; merge_policy decides conflicts |
| `@` (attach search) | attaches a `tac.search.*` strategy | `pipeline @ "cma_es_K_sweep"` |

Every compose operation returns a NEW immutable pipeline (no mutation, no
surprise side-effects). Pipelines are JSON-serializable via `.to_json()`
so the cathedral autopilot can rank candidate pipelines without
instantiating them and the operator can audit ranked candidates as plain
text.

Critical operator-precedence note: Python evaluates `&` BEFORE `|`, so
`A | B & C` parses as `A | (B & C)`. Use parentheses for the common
pattern `(A | B) & C`.

## §4 Production-hardening contracts (§5.5)

Per spec §5.5, every stage MUST satisfy these invariants enforced at
DECORATION TIME (NOT dispatch time):

| invariant | enforced by | failure mode without it |
|---|---|---|
| Frozen dataclass + type-hinted throughout | `BoostStageContract.__post_init__` | Mid-build mutation of stage parameters |
| Schema validation at import time (`ast.Assign` + `ast.AnnAssign`) | Catalog #168 discipline + test in `test_tac_boosting.py` | Future introspection tools silently skip type-annotated registries |
| `deterministic=True` for archive-byte-emitting stages | cross-field invariant in `__post_init__` + decorator signature check | Byte-stable archives break; Catalog #105/#139 no-op detector misfires |
| Fcntl-locked JSONL persistence | `tac.boosting.persistence._stage_outcomes_lock` | Catalog #131 sister "bare writes to shared state" |
| `scorer_free=True` for inflate-time stages | cross-field invariant + decorator scan for forbidden tokens | CLAUDE.md "Strict scorer rule" non-negotiable; ~73 MB rate inflation |
| 6-hook wire-in declared per Catalog #125 | 5 enum-validated hook fields + `hook_not_applicable_rationale` dict with required keys | Orphaned work; cathedral autopilot ranker doesn't see the stage |
| Decoration-time error vs dispatch-time error | All validators raise at module-import time | Errors at $5-50 dispatch instead of $0 import |

## §5 Builders — when to use which

| builder | composition shape | empirical anchor |
|---|---|---|
| `ResidualCascadeBuilder` | depth-N additive cascade; stage N consumes stage N-1's residual | PR106 format0d 2-pass (extends to depth N) |
| `PerPairDecoderEnsembleSelector` | M decoders in archive; per-pair selector picks one | extends fec6's K-mode selector to DECODERS |
| `ModeEnsembleDispatch` | K modes × M decoders product space (K*M candidates per pair) | composition of fec6's K + per-pair-ensemble's M |

The builders DO NOT subclass each other — per CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD" each builder's contract emission logic is
substrate-specific enough that subclassing would suppress engineering
choices (e.g. the index-bit budget is `log2(M)` vs `log2(K*M)`; the
emission key versioning differs; the consumed keys differ). They DO share
the `BoostStageContract` dataclass as the common output type.

## §6 Pareto frontier — apples-to-apples discipline

`ParetoFrontTracker` enforces axis-pinning per CLAUDE.md "Apples-to-apples
evidence discipline":

- Construction REQUIRES `axis: str` in `_LEGAL_AXIS_LABELS` (`[contest-CUDA]` /
  `[contest-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]` / `[proxy]` /
  `[advisory only]` / `[prediction]`).
- Each `ParetoAnchor` carries the tracker's axis; cross-axis mixing is
  refused at construction.
- `with_pareto_growth(reject_if_worsens_axis=...)` returns a filter
  callable that rejects stages whose contribution worsens the chosen axis
  vs the prior best.

The filter's behavior:
- `reject_if_worsens_axis='rate'`: candidate must have rate ≤ best_rate
- `reject_if_worsens_axis='distortion'`: ≤ best_distortion
- `reject_if_worsens_axis='both'`: candidate must be Pareto-non-dominated

## §7 Persistence (opt-in)

`append_stage_outcome_locked` writes to `.omx/state/boosting_stage_outcomes.jsonl`
via fcntl LOCK_EX. Per CLAUDE.md Catalog #128/#131/#132/#138 sister discipline:

- Schema-versioned (`boosting_stage_outcomes_v1`)
- APPEND-ONLY per Catalog #132
- STRICT-load via `load_stage_outcomes_strict` raises
  `BoostingLedgerCorruptError` on malformed line
- Lenient `load_stage_outcomes` skips malformed lines (for read-only
  aggregations)
- Quarantine-on-corrupt: `<path>.corrupt.<utcstamp>` per Catalog #138
- 4-process spawn-pool stress test verifies safety under concurrency

The namespace does NOT auto-persist — callers explicitly invoke
`append_stage_outcome_locked` only when they want the audit trail.

## §8 9-dimension success checklist evidence

Per CLAUDE.md "9-dimension success checklist" non-negotiable + Catalog #294:

| dim | label | evidence |
|---|---|---|
| 1 | UNIQUENESS | Decorator-based composable pipeline with operator semantics for boost stages — no equivalent canonical helper exists in the repo |
| 2 | BEAUTY + ELEGANCE | Narrow API (1 dataclass, 1 decorator, 3 operators, 3 builders, 1 tracker, 1 persistence layer; 12 files; ~2300 LOC for namespace + ~830 LOC for tests). 83 tests pass in <500 ms. |
| 3 | DISTINCTNESS | Sister of `tac.substrate_registry` at the boost-stage surface — DIFFERENT contract shape (23 fields vs substrate's 36; cascade ordering vs identity-only). NO shared parent class. |
| 4 | RIGOR | Premise verification (PV-1..PV-6) BEFORE edit. Assumption statement per Catalog #292. 83 dedicated tests covering positive + negative + boundary. Sister-regression tests vs Catalog #1 + #5. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per-method-optimal engineering: each builder forks the contract template where canonical adoption would suppress (per Catalog #290 canonical-vs-unique decision per layer). |
| 6 | STACK-OF-STACKS COMPOSABILITY | `|` `&` `@` composition; Pareto growth filter; sister namespaces will compose at the pipeline level (boost stages × compress-time passes × inflate-time post-process × side-info × search). |
| 7 | DETERMINISTIC REPRODUCIBILITY | `deterministic=True` enforced via decorator signature check; cross-field invariant refuses non-deterministic archive_build stages. JSON-round-trip preserves stage references byte-for-byte. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Decorator is PASS-THROUGH (zero runtime overhead). Pipeline `run()` is a single loop over stages. Persistence is OPT-IN. Pipeline objects are JSON-serializable for cathedral autopilot ranking without instantiation. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | The namespace itself does not produce a score — it is INFRASTRUCTURE for substrates that will. The 9-dim checklist applies to FUTURE substrates that consume the namespace. |

## §9 Cargo-cult audit per assumption (Catalog #303)

| assumption | HARD-EARNED or CARGO-CULTED? | rationale + unwind path |
|---|---|---|
| Frozen dataclass + field-level validators | **HARD-EARNED** | `tac.substrate_registry` (Catalog #241), `tac.deploy.modal.call_id_ledger` (Catalog #245), `tac.continual_learning` (Catalog #128) all use this pattern empirically + successfully across the codebase |
| Pipe-operator composition (`\|`) | **HARD-EARNED** | sklearn `Pipeline`, PyTorch `nn.Sequential`, scikit-image's pipe — canonical Python composition idiom. The spec §5.3-§5.4 mandates this directly. |
| Pareto-front-aware growth filter | **HARD-EARNED** | Spec §5.4 + §I.6 explicit framework. Greedy stack-by-largest-predicted-ΔS produced the 0.196-0.199 plateau per the assumptions audit. |
| Byte-stable archives via Catalog #158 | **HARD-EARNED** | `tac.packet_compiler.deterministic_compiler` is the canonical byte-stability surface. Catalog #158 STRICT @ 0. |
| Fcntl-locked JSONL persistence | **CARGO-CULTED from the canonical pattern** | The pattern works for high-throughput state (call_id_ledger has 100s of writes/hr; boost stage outcomes are 10s/day). Unwind path: marked as OPT-IN; the canonical helper exists if needed but the namespace does NOT force consumers through it. |
| 6-hook wire-in per Catalog #125 | **HARD-EARNED at the namespace surface; CARGO-CULTED per-stage** | The namespace MUST declare hooks (Catalog #125 non-negotiable for new infrastructure). But forcing every stage to declare hooks may be over-engineered. Unwind path: 5 of 6 hooks accept `not_applicable_with_rationale` so per-stage opt-outs are trivial. |
| Decorator pass-through (no runtime wrap) | **HARD-EARNED** | `tac.substrate_registry.register_substrate` does the same. Wrapping would add per-call overhead with no benefit; the contract is read at import time. |
| K=16 modes × M=8 = 128 candidate product space | **HARD-EARNED at the structural surface** | fec6's K=16 + per-pair-ensemble's M=8 multiply. The 76800 candidate-evaluations is significant but bounded; sister substrates can choose K or M independently. |
| Operator precedence of `&` BEFORE `|` | **HARD-EARNED** | Python language semantics; cannot be changed. Surfaced explicitly in the design memo §3 + docstring so callers add parens. |

## §10 Canonical-vs-unique decision per layer (Catalog #290)

| layer | DECISION | rationale |
|---|---|---|
| BoostStageContract frozen dataclass | ADOPT_CANONICAL | Mirrors SubstrateContract / CallIdLedgerSchema patterns. |
| Decorator pass-through + `_REGISTERED_*` dict | ADOPT_CANONICAL | Mirrors `tac.substrate_registry.decorator`. |
| Pipe-operator composition | FORK_BECAUSE_PRINCIPLED_MISMATCH | The substrate registry doesn't compose substrates — each is standalone. Boost stages MUST compose. |
| Pareto-front tracker | FORK_BECAUSE_PRINCIPLED_MISMATCH | No sister tracker exists (PV-6 verified); built from scratch to canonical operator's `tac.sensitivity_map` axis-pinning discipline. |
| Persistence: simple full-rewrite (not OP-10 amortized) | FORK_BECAUSE_PRINCIPLED_MISMATCH | call_id_ledger uses OP-10 because it sees high-throughput; boost stage outcomes are low-throughput. Premature optimization avoided. |
| `ResidualCascadeBuilder` vs `PerPairDecoderEnsembleSelector` vs `ModeEnsembleDispatch` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Each emits a different contract shape (cascade ordering / replace-per-pair / replace-with-product-space). Subclassing would suppress engineering. |
| Example stages (`raw_decoder` / `cascade_pose_residual_v1` / `cascade_seg_residual_v1`) | ADOPT_CANONICAL | Mirror the substrate_registry's `example_template.py`. |
| Errors: `BoostingNamespaceError` root + specific subclasses | ADOPT_CANONICAL | Mirrors `SubstrateContractError` + sister `DeterminismViolation` / `ScorerFreedomViolation`. |
| `correction_kind` enum | UNIQUE | substrate-specific to boost-stage semantics (additive / multiplicative / gated / replace / passthrough); no upstream sister. |
| `merge_policy` enum | UNIQUE | substrate-specific to parallel-merge composition. |

## §11 Observability surface (Catalog #305)

Per CLAUDE.md "Max observability — non-negotiable":

| facet | how to observe |
|---|---|
| Inspectable per layer | `pipeline.stage_contracts()` returns the tuple of contracts in order; each contract's `to_dict()` is a flat dict |
| Decomposable per signal | `BoostingPipelineResult.per_stage_outcomes` is a tuple of per-stage dicts with `stage_id` / `status` / `emitted_keys` |
| Diff-able across runs | `pipeline.to_json()` produces byte-stable representation (sort_keys=True); two runs of the same pipeline against the same seed_state are byte-identical when `deterministic=True` stages are used |
| Queryable post-hoc | `load_stage_outcomes(path)` reads the persisted ledger; rows are JSON dicts with `stage_id` / `status` / `pipeline_id` (caller-attached) / `written_at_utc` / `written_pid` / `written_host` |
| Cite-able | Each persisted outcome row carries `written_at_utc` + `written_pid` + `written_host`; combined with the upstream `call_id` (when caller attaches) the full provenance is preserved |
| Counterfactual-able | The decorator's pass-through nature means swapping a stage's contract requires editing one source file + restarting the process; the byte-mutation discipline (Catalog #139) applies to archive bytes that downstream codec stages emit |

## §12 6-hook wire-in declaration (Catalog #125)

| hook | wire-in status | rationale |
|---|---|---|
| 1. Sensitivity-map contribution | **ACTIVE — opt-in per stage** | `BoostStageContract.hook_sensitivity_contribution` field with `LEGAL_HOOK_SENSITIVITY` enum admitting `master_gradient_v1` / `scorer_conditional_entropy_map_v1` / `axis_weights_v1`. When `sensitivity_weighted=True` the pipeline auto-threads `master_gradient` into the stage function's kwargs. |
| 2. Pareto constraint | **ACTIVE — canonical** | `BoostStageContract.hook_pareto_constraint` field defaults to `"rate_distortion_v1"`. `ComposableBoostingPipeline.with_pareto_growth(tracker=...)` wires the tracker as the runtime filter. |
| 3. Bit-allocator hook | **ACTIVE — opt-in per stage** | `BoostStageContract.hook_bit_allocator_class` field with `LEGAL_HOOK_BIT_ALLOCATOR` enum. Most stages declare `not_applicable_with_rationale` and delegate bit allocation to a downstream codec stage per Catalog #272. |
| 4. Cathedral autopilot dispatch hook | **ACTIVE — pipeline-serialization-enabled** | `pipeline.to_json()` produces a stable serialized form the cathedral autopilot can rank without instantiating. `BoostStageContract.hook_autopilot_ranker` field declares ranker class. |
| 5. Continual-learning posterior update | **ACTIVE — opt-in via persistence helper** | `append_stage_outcome_locked` is the canonical writer; consumers explicitly invoke per run. Fcntl-locked per Catalog #128/#131. |
| 6. Probe-disambiguator | **ACTIVE — opt-in per stage with rationale** | `BoostStageContract.hook_probe_disambiguator` field; if None, `hook_not_applicable_rationale['hook_probe_disambiguator']` MUST carry non-empty rationale (enforced by `__post_init__`). |

## §13 Mandatory assumption statement (Catalog #292)

The shared assumption I am operating within for this design:

> "A decorator-based composable API with frozen-dataclass contracts +
> pipe-operator composition + Pareto-front-aware growth is the OPTIMAL
> engineering for a boosting namespace that future substrates will compose
> with at design time."

This assumption is HARD-EARNED (per §9 cargo-cult audit) by:
- Empirical success of the substrate_registry / call_id_ledger / continual_learning
  canonical patterns
- Spec §5.3-§5.5 + §I.6 explicit mandate
- Python language-canonical operator semantics for composition (sklearn / PyTorch)
- The 0.196-0.199 plateau is the COST of greedy stack-by-predicted-ΔS without
  the Pareto-front-aware growth filter

If the assumption is wrong, the failure mode is: the pipe-operator API is
over-engineered for early use cases that only chain 2-3 stages. Mitigation:
`ComposableBoostingPipeline.from_stage_ids([...])` provides the imperative
form for callers who prefer not to chain.

## §14 Premise verification evidence (Catalog #229)

Premise verifier landed at `.omx/tmp/tac_boosting_premise_verifier.txt`
covering 6 premises:

| premise | what we verified |
|---|---|
| PV-1 | `@register_substrate(SubstrateContract(...))` pattern at `src/tac/substrate_registry/` — frozen dataclass + decorator pass-through + duplicate-id rollback |
| PV-2 | `tac.packet_compiler.deterministic_compiler` exists at line 802; Catalog #158 STRICT enforces canonical-use |
| PV-3 | Catalog #168 `ast.Assign` + `ast.AnnAssign` walker discipline — covered by a dedicated test in `test_tac_boosting.py` |
| PV-4 | `fcntl`-locked JSONL append pattern in `tac.deploy.modal.call_id_ledger` (Catalog #128/#131) — mirrored verbatim in `tac.boosting.persistence` |
| PV-5 | PR106 format0d 2-pass additive correction grammar at `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575` — `ResidualCascadeBuilder` extends to depth N |
| PV-6 | `ParetoFrontTracker` BUILT FROM SCRATCH (no upstream sister in `tac.sensitivity_map` / `tac.optimization`) |

## §15 Worked examples (3)

### Example 1: depth-2 residual cascade with Pareto growth filter

```python
from tac.boosting import (
    ComposableBoostingPipeline, ParetoFrontTracker,
    ResidualCascadeBuilder, ResidualCascadeStageSpec, boost_stage,
)

# Build canonical contracts via the builder
builder = ResidualCascadeBuilder(
    root_stage_id="raw_decoder",
    depth=2,
    stage_specs=[
        ResidualCascadeStageSpec(stage_id="cascade_pose_v1"),
        ResidualCascadeStageSpec(stage_id="cascade_seg_v1"),
    ],
    lane_id="lane_my_substrate_20260601",
)
contracts = builder.build_contracts()

# Decorate the per-stage functions (caller-provided)
@boost_stage(contracts[0])
def cascade_pose_v1(state, *, policy): ...

@boost_stage(contracts[1])
def cascade_seg_v1(state, *, policy): ...

# Compose + run with Pareto filter
tracker = ParetoFrontTracker(axis="[contest-CUDA]")
tracker.track_anchor(rate=180_000, distortion=0.193, source="pr101_gold")

pipeline = (
    ComposableBoostingPipeline()
    | "raw_decoder"
    | "cascade_pose_v1"
    | "cascade_seg_v1"
).with_pareto_growth(reject_if_worsens_axis="rate", tracker=tracker)

result = pipeline.run(seed_state={"seed_frames": real_frames})
```

### Example 2: per-pair decoder ensemble selector

```python
from tac.boosting import (
    PerPairDecoderEnsembleSelector, PerPairDecoderEnsembleSpec, boost_stage,
)

spec = PerPairDecoderEnsembleSpec(
    stage_id="pp_decoder_M4",
    num_decoders=4,
    selector_criterion="local_variance",  # scorer-free image-domain proxy
    decoder_archive_keys=("decoder_a", "decoder_b", "decoder_c", "decoder_d"),
)
contract = PerPairDecoderEnsembleSelector(spec=spec).build_contract()

@boost_stage(contract)
def pp_decoder_M4(state, *, policy):
    # Inflate-time per-pair decoder selection; scorer-free
    ...
```

### Example 3: K=16 modes × M=8 decoders product-space dispatch

```python
from tac.boosting import ModeEnsembleDispatch, ModeEnsembleDispatchSpec, boost_stage

spec = ModeEnsembleDispatchSpec(
    stage_id="md_K16_M8",
    num_modes=16,
    num_decoders=8,
    selector_criterion="joint_mode_decoder_score_proxy",
    decoder_archive_keys=tuple(f"decoder_{i}" for i in range(8)),
)
# product_space_size = 128; per_pair_index_bits = 7
contract = ModeEnsembleDispatch(spec=spec).build_contract()

@boost_stage(contract)
def md_K16_M8(state, *, policy):
    # 600 pairs × 128 candidates each = 76800 image-domain evaluations
    # at inflate time, all scorer-free
    ...
```

## §16 Cross-references

- Spec: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
  (§5.3-§5.5 contract design + §I.6 boosting framework + §J side info)
- Sister namespace: `tac.substrate_registry` (the META layer for substrates)
- Empirical anchor: `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549-575`
- Canonical persistence pattern: `tac.deploy.modal.call_id_ledger`
- Premise verifier: `.omx/tmp/tac_boosting_premise_verifier.txt`
- Landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_tac_boosting_namespace_landed_20260517.md`
- Next namespace queued: `tac.compress_time_optimization` per §5.2 build sequence

## §17 Reactivation criteria

Re-read this memo when:
- A new substrate wants to use `@boost_stage` (consult §5 builders + §15 examples)
- Sister namespace (`tac.compress_time_optimization` / `tac.inflate_time_post_processing`
  / `tac.side_information` / `tac.search`) is being scaffolded (use the same
  decorator + dataclass + composition pattern; reference §5.3-§5.5)
- The cathedral autopilot ranker needs to rank boost-stage pipelines
  (consume via `pipeline.to_json()`; persist via `append_stage_outcome_locked`)
- A council deliberation asks "which stages compose with which substrates"
  (consult §15 worked examples)

## §18 2026-05-17 Runtime Hardening Addendum

Follow-up ledger:
`.omx/research/tac_boosting_parallel_merge_runtime_hardening_20260517_codex.md`.

The first WIP runtime implemented `&` as a build-time annotation but still ran
stages in a left-to-right loop. That would have let a "parallel" sibling read
the prior sibling's emitted state and would have resolved conflicts by
accidental order rather than by declared `merge_policy`.

Current runtime semantics:

- `|` remains sequential.
- `&` now forms an execution group: one sequential root plus following
  parallel siblings.
- Each sibling receives the same pre-group input state.
- The group output is merged once, honoring the incoming stage's
  `merge_policy`.
- `merge_policy="explicit"` fails closed on overlapping keys until a concrete
  merge callable surface exists.

Focused verification after the fix:

```text
.venv/bin/python -m pytest src/tac/tests/test_tac_boosting.py -q
87 passed in 0.33s
```


# OBSERVABILITY_SURFACE_SECTION_WAIVED:historical_design_memo_predates_catalog_305_section_header_requirement_or_is_namespace_design_not_substrate_specific_observability_per_catalog_110_113_HISTORICAL_PROVENANCE_APPEND_ONLY_discipline_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
