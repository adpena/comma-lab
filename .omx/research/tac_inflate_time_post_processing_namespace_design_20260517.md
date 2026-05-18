# tac.inflate_time_post_processing namespace design

**Lane:** `lane_tac_inflate_time_post_processing_namespace_decorator_api_20260517`
**Date:** 2026-05-17
**Spec provenance:** `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
§5.3-§5.5 + §G inflate-time + §7.4 + §7.6
**Premise verifier:** `.omx/tmp/tac_inflate_time_post_processing_premise_verifier.txt`
**Catalog #s satisfied:** #229 (premise verification) / #290 (canonical-vs-unique
decision per layer) / #294 (9-dim checklist) / #303 (cargo-cult audit) /
#305 (observability surface) / #309 (horizon_class declaration)

horizon_class: plateau_adjacent

## Problem statement

The §7.4 + §7.6 + §G inflate-time table in the design spec enumerates
six unaddressed inflate-time techniques (per-frame post-processing /
per-pair pose refinement / multi-archive ensemble / wavelet residual
stacking / mask temporal smoothing / inflate-time TTO). The contest
allows 30 minutes of inflate-time compute on T4 but our fec6 baseline
uses <5 min — ~25 min of free compute is unexploited.

The §5 canonical-helper inventory identified `tac.inflate_time_post_processing`
as one of FIVE missing namespaces in the build queue (§5.2). This subagent
slot builds it as the 3rd of 5 (sisters tac.boosting + tac.compress_time_optimization
landed earlier today; tac.side_information + tac.search are running in
parallel from sister subagents).

## Solution shape

A decorator-based composable inflate-time pass namespace mirroring the
canonical pattern from tac.compress_time_optimization, with 5 first-class
technique builders + one decorator + one immutable pipeline + one
fcntl-locked JSONL persistence ledger.

The namespace is STRUCTURALLY INDEPENDENT from sister namespaces per the
2026-05-15 UNIQUE-AND-COMPLETE-PER-METHOD operating mode. The decorator,
contract, pipeline, persistence, and errors all live in `tac.inflate_time_post_processing.*`
with NO imports from `tac.boosting.*` or `tac.compress_time_optimization.*`.

## Canonical-vs-unique decision per layer

This section satisfies Catalog #290. For each canonical helper or shared
META-layer field the namespace might have adopted, the per-layer decision
is documented below with rationale per the falling-rule list (empirical /
principled / unclear / obvious-fit) from the UNIQUE-AND-COMPLETE-PER-METHOD
non-negotiable.

### Layer 1: Decorator pattern (`@<stage>(<Contract>(...))`)

**Decision:** ADOPT_CANONICAL_BECAUSE_SERVES.

**Rationale:** The pass-through decorator + frozen-dataclass + in-memory
id-keyed registry is the canonical pattern adopted by tac.boosting and
tac.compress_time_optimization earlier today. It serves every consumer
(pipeline / persistence / autopilot ranker / preflight gate) via explicit
field access on the frozen contract. PR101-paradigm + Catalog #168
AST-AnnAssign discipline + Catalog #241 substrate META layer all point at
this exact shape. The DECORATOR FUNCTION is unique to this namespace
(`inflate_time_post_filter`); the PATTERN is canonical.

### Layer 2: Frozen-dataclass contract with __post_init__ validation

**Decision:** ADOPT_CANONICAL_BECAUSE_SERVES.

**Rationale:** Same as Layer 1. The frozen dataclass + field-level
validators in __post_init__ is the canonical shape. We do NOT inherit
from CompressTimePassContract or BoostStageContract — instead we
duplicate the field-level validation logic so the namespace is
structurally independent (per the 2026-05-15 retrospective: "share what
works but when it is stale or obsolete or suppressing signal or
otherwise and when the optimal engineering calls for it we want full
and complete and correct unique and distinct designs and implementations").

### Layer 3: Field set (28 fields)

**Decision:** FORK_BECAUSE_PRINCIPLED_MISMATCH.

The compress-time contract has fields that are STRUCTURALLY INAPPLICABLE
to inflate-time:

- `rate_budget_bytes`: inflate doesn't allocate archive bytes. REMOVED.
- `correction_kind={refinement,residual_correction,search,bisection,transform,passthrough}`:
  REPLACED with inflate-domain values `{denoise,sharpen,smooth,upscale,select,refine,transform,passthrough}`.
- `max_wallclock_seconds`: was OPTIONAL in compress (unbounded); REQUIRED
  in inflate (30-min ceiling per spec §G).

Inflate-time NEW fields:

- `inflate_compute_budget_seconds`: capped at 1800.0 (30-min T4 ceiling).
- `applies_to_frames` ∈ {all, pairs_only, odd_only, even_only}: declares
  which frames the pass operates on; the pipeline composer uses this for
  ambiguous-emit detection.
- `archive_bytes_added: int = 0`: HARD-CODED-INVARIANT. ArchiveBytesViolation
  raised if > 0. This is the structural protection against the "research-
  substrate trap" (8th forbidden pattern): inflate-time techniques operate
  on FRAMES, not BYTES.
- `score_axis_affected: tuple[str, ...]`: subset of `("seg", "pose")`;
  declares which scorer axes the technique targets.
- `requires_scorer_surrogate: bool = False`: per Catalog #527 distinction
  between forbidden contest-scorer access and legal Hinton-distilled
  surrogate.
- `requires_cpu_only: bool = True`: default to CPU per Catalog #146
  inflate.py budget; operator can override with documentation.

### Layer 4: Stage-phase enforcement

**Decision:** ADOPT_CANONICAL_BECAUSE_SERVES + INVERT.

tac.compress_time_optimization raises InflatePhaseForbiddenError for
`stage_phase='inflate'`. We raise CompressPhaseForbiddenError for
`stage_phase='compress'`. The two namespaces enforce MUTUAL EXCLUSION at
decoration time so a malformed pass surfaces at IMPORT time pointing
the operator at the correct sister namespace.

### Layer 5: Scorer-access invariant

**Decision:** FORK_BECAUSE_PRINCIPLED_MISMATCH (stricter than sister).

`scorer_free=False` is FORBIDDEN in this namespace (raises
ScorerAccessForbiddenError). The sister tac.compress_time_optimization
allows compress-time scorer access because the contest scorer is loaded
freely at compress time. At inflate time, loading the contest scorer
(~73 MB) catastrophically degrades the rate term. The contract's
`requires_scorer_surrogate=True` is the LEGAL alternative — a CPU-trained
Hinton-distilled scorer surrogate per Catalog #527.

### Layer 6: 6-hook wire-in (Catalog #125)

**Decision:** ADOPT_CANONICAL_BECAUSE_SERVES with INFLATE-SPECIFIC enum
values.

The 6-hook fields (sensitivity / pareto / bit_allocator / autopilot /
continual_learning / probe_disambiguator) are the canonical Catalog #125
shape. Enum value sets are INFLATE-SPECIFIC:

- `LEGAL_HOOK_PARETO`: default `inflate_wallclock_envelope_v1` (sister
  default is `rate_distortion_v1` which doesn't fit inflate-time).
- `LEGAL_HOOK_BIT_ALLOCATOR`: typically `not_applicable_with_rationale`
  (no bit allocation at inflate); exception is
  `scorer_surrogate_variant_selector` for MultiPassInflateRefinement.
- `LEGAL_HOOK_CONTINUAL_LEARNING`: anchor kind
  `inflate_time_post_processing_pass_outcomes_v1`.

### Layer 7: Pipeline composition operators (`|` / `&` / `@`)

**Decision:** ADOPT_CANONICAL_BECAUSE_SERVES.

Canonical operator precedence + immutable-pipeline pattern + JSON
serialization + ambiguous-emit detection at build time. UNIQUE additions:
`with_inflate_compute_budget(seconds=N)` capped at 1800 + `with_max_frames(n=N)`
smoke-run guardrail.

### Layer 8: Persistence (fcntl-locked JSONL)

**Decision:** ADOPT_CANONICAL_BECAUSE_SERVES.

Canonical Catalog #128 / #131 / #138 sister discipline. Schema version is
`inflate_time_post_processing_pass_outcomes_v1`; path is
`.omx/state/inflate_time_post_processing_pass_outcomes.jsonl`.

### Layer 9: 5 first-class builders

**Decision:** FORK_BECAUSE_PRINCIPLED_MISMATCH.

The sister namespace's 5 builders (GenericTTOHarness / MultipassRefinement
/ SimulatedAnnealing / PerPairCoordinateSearch / IteratedBisectionRateKnee)
all operate on archive BYTES at compress time. The inflate-time analogs
are:

- `BilateralFilterPostProcessor` — per-frame edge-preserving smoothing
- `NLMDenoisingPostProcessor` — non-local-means denoising
- `LearnedPostFilterApplier` — apply a distilled CPU-only model
- `SuperResolutionUpscaler` — 384x512 → 874x1164 (bicubic/lanczos/learned)
- `MultiPassInflateRefinement` — N-variant ensemble + surrogate ranking

NONE of these have a meaningful compress-time analog (they all operate
on decoded RGB frames, not archive bytes).

## Cargo-cult audit per assumption

This section satisfies Catalog #303. Per the cargo-cult-unwind methodology
(NSCS06 v6 → v7 44% improvement anchor): enumerate each
substrate-engineering assumption + classify HARD-EARNED vs CARGO-CULTED.

| # | Assumption | Classification | Rationale |
|---|---|---|---|
| 1 | The 5 first-class builder choices (bilateral / NLM / learned-filter / SR-upscaler / multi-pass) | **HARD-EARNED** | Direct from spec §G inflate-time table rows 1+3 + the §7.4 narrative. The 6th row (per-pair pose refinement using motion-only prior) was deliberately deferred because it touches `tac.side_information` (sister subagent's scope per Catalog #230) + would couple to RAFT optical flow infrastructure. |
| 2 | archive_bytes_added=0 invariant | **HARD-EARNED** | Definitional: inflate-time post-processing operates on FRAMES after the decoder produces them; adding archive bytes is by definition a COMPRESS-time technique (handled by sister namespace tac.compress_time_optimization per spec §5.2). |
| 3 | max_wallclock_seconds=REQUIRED + 1800s ceiling | **HARD-EARNED** | Spec §G empirical: 30 min on T4 is the contest budget. Currently fec6 uses <5 min so ~25 min is free — that's the meat on the bone this namespace exploits. |
| 4 | scorer_free=True INVARIANT (scorer_free=False FORBIDDEN) | **HARD-EARNED** | CLAUDE.md "Strict scorer rule" non-negotiable + Catalog #6 (`check_no_scorer_load_at_inflate`). The contest scorer is ~73 MB; loading at inflate destroys the rate term. The Hinton-distilled SURROGATE per Catalog #527 is the legal alternative (requires_scorer_surrogate=True). |
| 5 | requires_cpu_only=True default | **HARD-EARNED** | Catalog #146 budget: inflate.py ≤ 100 LOC + ≤ 2 deps. The contest scorer eval runs on CPU on the public leaderboard (GHA Linux x86_64). GPU inflate-time variants exist but are not required and the default should bias toward the canonical case. |
| 6 | deterministic=True INVARIANT | **HARD-EARNED** | Catalog #158 deterministic-compiler discipline + CLAUDE.md "Bit-level deconstruction". Inflate MUST produce byte-identical frames per run so the contest scorer's eval is reproducible. |
| 7 | 5-builder collapse of the 6 §G inflate-time table rows | **CARGO-CULTED** (consciously) | The spec §G table has 6 rows; we deliver 5 first-class builders. Row 2 (per-pair pose refinement via SE(3) smoothness) is deferred to tac.side_information per disjoint scope; rows 4+5 (wavelet residual stacking + mask temporal smoothing) collapse into LearnedPostFilterApplier (any per-frame model architecture suffices); row 6 (inflate-time TTO without scorer) collapses into MultiPassInflateRefinement (ranking via surrogate IS a 0-iteration TTO). UNWIND: future op-routable could split LearnedPostFilterApplier into LearnedDenoiserApplier + WaveletResidualApplier if empirical evidence shows one dominates. |
| 8 | Frozen dataclass contract pattern (canonical adoption from sister) | **HARD-EARNED** | Catalog #168 AST-AnnAssign discipline + the 270+ catalog gate ecosystem. Pre-2026-05-15 we would have inherited from a shared base; post-UNIQUE-AND-COMPLETE-PER-METHOD we duplicate the structure. The duplication is intentional infrastructure independence, not cargo-cult sharing. |
| 9 | Builders DON'T execute — they emit CONTRACTS for substrate-specific decorated functions | **HARD-EARNED** | HNeRV parity discipline L7 (bolt-on vs substrate-engineering split). The BUILDER is canonical infrastructure (~150 LOC each); the substrate-specific per-frame loop body is unique-per-method engineering provided by the decorated function. |
| 10 | requires_scorer_surrogate=True is the ONLY legal scorer-like signal | **HARD-EARNED** | Per Catalog #527 + the contest's rate-term math: surrogate weights ship as part of the archive bytes via the COMPRESS-time grammar; loading them at inflate is legal because they're ALREADY part of the archive byte budget. The contest scorer's weights are NOT in the archive — loading them ad-hoc would double-charge. |

## 9-dimension success checklist evidence

This section satisfies Catalog #294. Per the 9-dimension success checklist
standing directive 2026-05-15, every dimension MUST have explicit evidence.

1. **UNIQUENESS** — This namespace is the ONLY canonical home for
   inflate-time post-processing passes. Sister namespaces refuse
   `stage_phase='inflate'` (CompressPhaseForbiddenError in
   tac.compress_time_optimization). No competing inflate-time pass API
   exists in the repo today.

2. **BEAUTY + ELEGANCE** — 7 .py files in the package (errors / contract /
   decorator / pipeline / persistence + 5 builders + __init__ + examples).
   Public API exposes 51 names via `__all__`. Every contract is one
   frozen dataclass; every error is one typed exception; every pipeline
   compose operation returns a new immutable pipeline. PR101-style
   ~150-200 LOC per file; reviewable in 30 seconds per file.

3. **DISTINCTNESS** — Explicitly different from sisters:
   - `tac.boosting`: residual cascade + per-pair decoder ensemble +
     Pareto-front-aware stack growth (substrate composition).
   - `tac.compress_time_optimization`: generic TTO harness + multipass
     refinement + SA + per-pair coordinate search + iterated bisection
     (operates on archive BYTES at compress time).
   - `tac.inflate_time_post_processing` (this namespace): 5 deterministic
     image-domain post-processors (operate on FRAMES at inflate time;
     archive_bytes_added=0 invariant).
   The three namespaces partition the canonical-helper space along
   ORTHOGONAL axes (composition vs compress-time vs inflate-time).

4. **RIGOR** — Premise verification per Catalog #229 (12 PVs documented
   pre-edit in `.omx/tmp/tac_inflate_time_post_processing_premise_verifier.txt`);
   sister-subagent ownership map per Catalog #230 honored (disjoint scope
   from tac.side_information / tac.search); checkpoint discipline per
   Catalog #206; UNIQUE-AND-COMPLETE-PER-METHOD operating mode per
   2026-05-15 standing directive; council-grade canonical-vs-unique
   decision per Catalog #290 (this memo).

5. **OPTIMIZATION PER TECHNIQUE** (covered by Catalog #290; sister
   dimension to canonical-vs-unique) — Each builder's spec captures the
   minimal sufficient parameters (e.g. BilateralFilterSpec has 3 numeric
   knobs: sigma_spatial + sigma_intensity + kernel_diameter). The
   contract carries cross-field invariants. Each builder's
   `canonical_vs_unique_decision` field documents which loop infrastructure
   is canonical + which body is unique-per-method.

6. **STACK-OF-STACKS-COMPOSABILITY** — The pipeline's `|` operator chains
   passes sequentially; `&` parallel-merges; `@` attaches search strategies.
   Pipelines are JSON-serializable so the cathedral autopilot ranker can
   stack-of-stacks rank candidates without instantiation. The 5 builders
   are orthogonal in (frame-axis, kind-axis):
   - bilateral: per-frame, denoise
   - NLM: per-pixel within frame, denoise
   - learned: per-frame, refine
   - SR upscaler: per-frame, upscale
   - multi-pass: per-pair, select

7. **DETERMINISTIC REPRODUCIBILITY** — Catalog #158-compliant
   (deterministic=True invariant; SeedRequiredViolation at decoration
   for non-deterministic signatures). All 5 builders are deterministic
   by construction. Persistence is fcntl-locked + atomic-write
   (Catalog #128 / #131 / #132 / #138).

8. **EXTREME OPTIMIZATION + PERFORMANCE** — Zero-overhead decorator
   (pass-through; the wrapped function runs at full speed). Pipeline
   build validates upfront (no per-pass surprise raises); run is a
   straight loop with monotonic-clock timing. The persistence path uses
   the SIMPLER full-rewrite (sufficient for low-throughput design-time
   cadence; the OP-10 O(1) amortized path is overkill at this rate per
   the sister namespace's PV-4 decision).

9. **OPTIMAL MINIMAL CONTEST SCORE** — This namespace's score impact is
   indirect: by exploiting the unused ~25 min of inflate-time compute on
   T4, every substrate that adopts the canonical 5 first-class builders
   gains a SegNet boundary stability boost + PoseNet feature-consistency
   boost without spending archive bytes. The contract's
   `score_axis_affected` field declares which scorer axis each pass
   targets so the autopilot ranker can compose ROI-aware pipelines.
   Empirical score-impact validation is the next subagent's work — this
   landing is the INFRASTRUCTURE that enables the empirical exploration.

## Observability surface

This section satisfies Catalog #305. The namespace is structurally
observable across all 6 facets:

1. **Inspectable per layer** — Every pass's contract is a frozen dataclass
   with explicit field access. `get_registered_passes()` returns the
   full id → contract map. `get_pass_function(pass_id)` returns the
   callable. Each callable carries
   `fn.__inflate_time_post_filter_contract__` for introspection.

2. **Decomposable per signal** — `InflateTimePipelineResult.per_pass_outcomes`
   carries per-pass `(pass_id, status, elapsed_seconds, frames_processed,
   emitted_keys)`. `score_axis_affected` per contract decomposes the
   per-pass contribution into seg / pose axes.

3. **Diff-able across runs** — Pipelines + contracts are JSON-serializable
   (`to_json` / `to_dict` / `from_dict` round-trip). Two runs with
   identical pipeline JSON + decoded_frames input MUST produce identical
   `InflateTimePipelineResult.final_state` (deterministic invariant).

4. **Queryable post-hoc** — Persistence ledger at
   `.omx/state/inflate_time_post_processing_pass_outcomes.jsonl` is
   plain JSONL; standard tools (jq / pandas / sqlite-utils) query it.
   `load_pass_outcomes` + `load_pass_outcomes_strict` are public read APIs.

5. **Cite-able** — Every contract carries `lane_id` + `design_memo` +
   `canonical_vs_unique_decision` provenance fields. The pipeline result
   names every pass by id; the ledger row carries `written_at_utc` +
   `written_pid` + `written_host` per the fcntl-locked schema.

6. **Counterfactual-able** — Pipeline composition is structural: replacing
   one pass with another (e.g. NLM h=0.05 → h=0.1) and re-running answers
   the counterfactual without re-instrumenting. The `with_max_frames(n=N)`
   smoke filter lets the operator do counterfactual A/B at a fraction of
   full cost.

## 6-hook wire-in declarations

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #125 + the
existing sister-namespace landings, every subagent landing MUST declare
all 6 wire-in hooks. Mapping:

1. **Sensitivity-map contribution** — `hook_sensitivity_contribution` on
   each contract; legal values include `scorer_surrogate_axis_weights_v1`
   (active in MultiPassInflateRefinement) and `not_applicable_with_rationale`
   (4 other builders; per-byte sensitivity weighting is structurally
   meaningless at inflate time since archive bytes are frozen).
2. **Pareto constraint** — `hook_pareto_constraint` defaults to
   `inflate_wallclock_envelope_v1` (capturing the 30-min T4 ceiling as the
   Pareto axis the autopilot ranker consumes).
3. **Bit-allocator hook** — Not directly wired: inflate-time has no bit
   allocation (archive_bytes_added=0 invariant). MultiPassInflateRefinement
   sets `scorer_surrogate_variant_selector` as the indirect equivalent
   (selects best variant per surrogate score, not bits).
4. **Cathedral autopilot dispatch hook** — `hook_autopilot_ranker` defaults
   to `cathedral_autopilot_v1` on all 5 builders. The cathedral autopilot
   ranker consumes the JSON-serialized pipelines via the public `to_json`
   API. **N/A for the namespace itself**: the namespace ships passes; the
   ranker consumes them when a substrate's trainer registers a pipeline.
5. **Continual-learning posterior update** — `hook_continual_learning_anchor_kind`
   defaults to `inflate_time_post_processing_pass_outcomes_v1`. The
   persistence ledger at
   `.omx/state/inflate_time_post_processing_pass_outcomes.jsonl` is the
   anchor surface. Empirical anchors land via `append_pass_outcome_locked`.
6. **Probe-disambiguator** — Per CLAUDE.md
   `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`
   — for designs with 2+ defensible interpretations, ship both via
   callable interface + build `tools/probe_<track>_disambiguator.py`. **N/A
   at the namespace level**: each builder has a single canonical
   interpretation; the disambiguation knobs are at the spec level
   (e.g. BilateralFilterSpec's 3 sigmas). Per-substrate consumers MAY
   build a probe disambiguator over which builder to choose (e.g. NLM vs
   bilateral) — that's the substrate's design memo, not this one.

## Out of scope (sister-disjoint per Catalog #230)

The following are EXPLICITLY out of scope for this subagent:

- `tac.side_information` (sister subagent): RAFT optical-flow side-info
  baker; openpilot ego-motion feature extractor; per-pair SegNet class
  summary baker; per-pair conditional-entropy estimator. The §G row 2
  (per-pair pose refinement using motion-only prior) is COUPLED to RAFT
  infrastructure and lives there.
- `tac.search` (sister subagent): Bayesian / CMA-ES / `optuna` over joint
  manifold. Pipelines accept `@ "search_descriptor"` attachment but the
  search engine itself ships in the sister namespace.
- Empirical score-lowering with the 5 builders. This landing is the
  INFRASTRUCTURE that enables exploration; the empirical lane is a
  follow-on subagent that adopts the namespace from a real substrate
  trainer.

## Reactivation criteria

Re-read this memo when:
- A new substrate adopts an inflate-time post-processing pass (cite this
  memo as the API provenance + design rationale)
- A 6th first-class builder needs to land (audit Cargo-cult #7 above —
  the 5-builder collapse may need to unwind)
- The §G inflate-time table grows new rows (apply §5.3-§5.5 contracts
  + this memo's canonical-vs-unique pattern)
- The 30-min T4 ceiling changes (update MAX_INFLATE_COMPUTE_BUDGET_SECONDS
  + the contract's wallclock validator)
- Sister subagent tac.side_information lands per-pair pose refinement —
  evaluate whether to add a bridge primitive here that consumes the
  per-pair pose deltas

## Cross-references

- Spec: `.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md`
  §5.3-§5.5 + §G + §7.4 + §7.6
- Sister design memo (tac.compress_time_optimization): see sister landing
  `feedback_compress_time_optimization_namespace_landed_20260517.md`
- Sister design memo (tac.boosting): see sister landing
  `feedback_boosting_namespace_landed_20260517.md`
- Premise verifier: `.omx/tmp/tac_inflate_time_post_processing_premise_verifier.txt`
- Test suite: `src/tac/tests/test_tac_inflate_time_post_processing.py` (156 tests)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (the canonical
  share-vs-fork directive)
- CLAUDE.md "Strict scorer rule" non-negotiable (the basis for the
  ScorerAccessForbiddenError invariant)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  non-negotiable (the basis for Catalog #220 + this memo's HARD-EARNED
  invariants)
- Catalog #146 (inflate.py ≤ 100 LOC + ≤ 2 deps + per-video loop pattern)
- Catalog #527 (Hinton-distilled CPU scorer surrogate canonical pattern)
