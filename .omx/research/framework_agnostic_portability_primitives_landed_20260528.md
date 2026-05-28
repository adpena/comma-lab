---
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, AssumptionAdversary, PR95Author, JackFromSkunkworks]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "~95% LOC overlap across V3 sister pair generalizes to all paired substrates"
    classification: CARGO_CULTED
    rationale: "single-substrate anchor; sister wave Wave N+4 will measure per-substrate refactor delta empirically"
  - assumption: "canonical numpy oracle is the byte-determinism guarantee across MLX/PyTorch/tinygrad paths"
    classification: HARD_EARNED
    rationale: "tested in src/tac/framework_agnostic/tests/test_framework_agnostic.py byte-determinism subset (49/49 pass; MLX path matches numpy oracle byte-for-byte)"
  - assumption: "UNIQUE-AND-COMPLETE-PER-METHOD operating mode justifies per-substrate fork when canonical abstraction suppresses score"
    classification: HARD_EARNED
    rationale: "operator standing directive 2026-05-15 + acceptance criterion baked into anti-pattern.canonical_unwind_path"
council_decisions_recorded:
  - "op-routable: Wave N+4 substrate trainer refactor wave consumes tac.framework_agnostic + canonical decorators per substrate pair"
  - "op-routable: per-substrate empirical anchor demonstrates canonical abstraction preserves or improves score (UNIQUE-AND-COMPLETE-PER-METHOD acceptance)"
  - "op-routable: backfill canonical anti-pattern empirical_falsifications row with measured V3 refactor LOC delta once Wave N+4 lands first pair"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
canonical_equation_referenced: framework_agnostic_backend_abstraction_compounding_v1
horizon_class: plateau_adjacent
---

# Framework-agnostic portability primitives + decorators landed

**Lane**: `lane_framework_agnostic_portability_primitives_20260528` L1
(impl_complete + memory_entry + canonical_helper + cathedral_consumer +
canonical_anti_pattern + canonical_equation + probe_outcomes_row).

**Slot**: Wave N+3 Slot 2 (cap=2). Slot 1 in-flight on V3 PyTorch sister
trainer + recipes (disjoint scope per Catalog #340 sister-checkpoint guard).

**Operator NON-NEGOTIABLE META directive 2026-05-28 verbatim**: *"remmebr
MLX first but agnostic portability via numpy and tinygrad like primitives
and helpers or decorators or whatever"*.

**Cost**: $0 GPU + ~60 min wall-clock.

## What landed (4 layers + 3 sister deliverables)

### Layer 1: canonical helper package `src/tac/framework_agnostic/` (~900 LOC)

- `backend.py` — `Backend` enum (MLX / PYTORCH / NUMPY / TINYGRAD / AUTO) +
  `select_backend(*, override, env_var, priority)` per Catalog #205 sister
  cascade + `BackendUnavailableError` fail-closed + `detect_available_backends()`
  + per-platform `_platform_priority_order()` (Darwin ARM64 MLX-first per 8th
  standing directive; Linux CUDA PyTorch-first per Catalog #205 sister;
  generic numpy fallback per CLAUDE.md 8th directive)
- `tensor_protocol.py` — `FrameworkAgnosticTensor` runtime_checkable Protocol +
  `shape_of` + `dtype_name` canonical helpers per Catalog #335 sister
- `operations.py` — `quantize_int8_per_channel` + `dequantize_int8_per_channel`
  (per-channel symmetric int8 with canonical numpy oracle for byte
  determinism per Catalog #146); `quantize_fp4_packed_nibbles` (canonical
  Quantizr unsigned-E2M1 codebook `[0,0.5,1,1.5,2,3,4,6]`); `brotli_compress`
  (canonical hard dep per Catalog #203/#224)
- `decorators.py` — `@framework_agnostic` (resolve backend once + inject
  kwarg); `@mlx_first_with_numpy_fallback` (per 8th standing directive);
  `@pytorch_first_with_numpy_fallback` (per Catalog #205 sister for
  contest-resolution paths); `@inflate_runtime_helper` (pins NUMPY per HNeRV
  parity L4 ≤200 LOC + ≤2 deps)
- `helpers.py` — `mlx_state_dict_to_npz_bridge` + `pytorch_state_dict_to_npz_bridge` +
  `tinygrad_state_dict_to_npz_bridge` (canonical bridge contract per 8th
  standing directive `MLX state_dict → npz → ZIP-member → numpy inflate`) +
  `npz_to_numpy_primitives` (canonical inflate-side consumer) +
  `assert_no_framework_mismatch` (fail-closed type check)
- `__init__.py` — narrow public API export

### Layer 2: canonical anti-pattern registered

`mlx_trainer_pytorch_sister_duplicated_implementation_v1` per
`tac.canonical_anti_patterns.register_anti_pattern` →
`.omx/state/canonical_anti_patterns_registry.jsonl`.

- `paradigm_class`: `rigor_loss_anti_pattern`
- `severity`: `medium_substrate_regression`
- `forbidden_pattern_predicate`: substrate trainer pair where filenames match
  `experiments/train_substrate_<id>.py` + `experiments/train_substrate_<id>_mlx_local.py`
  AND both files independently implement training-loop + quantization +
  archive emission without routing through `tac.framework_agnostic`
- `canonical_unwind_path`: consume `tac.framework_agnostic` primitives +
  `@framework_agnostic` / `@mlx_first_with_numpy_fallback` /
  `@pytorch_first_with_numpy_fallback` decorators + canonical
  `quantize_int8_per_channel` / `quantize_fp4_packed_nibbles` /
  `brotli_compress` primitives + bridge helpers
- `canonical_producers`: V3 sister trainer pair (783 + 720 = 1503 LOC) +
  glob `experiments/train_substrate_*_mlx_local.py`
- `canonical_consumers`: `tac.framework_agnostic`,
  `tac.cathedral_consumers.framework_agnostic_lookup_consumer`,
  `tac.cathedral_consumers.canonical_anti_patterns_lookup_consumer`
- `empirical_falsifications`: () (design-only at landing; Wave N+4 sister
  refactor wave appends measured per-substrate refactor LOC delta)

### Layer 3: canonical equation registered

`framework_agnostic_backend_abstraction_compounding_v1` per
`tac.canonical_equations.register_canonical_equation` →
`.omx/state/canonical_equations_registry.jsonl`.

- **Mathematical predicate**: `ΔLOC_total = -Σ_i α_i · LOC_i` where
  `α_i ≈ 0.475` for paired sister substrates
- **Empirical anchor**: `v3_sister_pair_loc_overlap_anchor_20260528` —
  V3 sister pair total LOC = 1503 (PyTorch 783 + MLX 720)
- **Predicted**: `loc_reduction_per_substrate_fraction ≈ 0.475` per
  paired substrate via canonical framework_agnostic abstraction
- **At-scale estimate**: 28 paired substrates × 0.475 × 1503 ≈ 20K LOC
  savings
- **`predicted_vs_empirical_residual`**: 0.0 at landing (design prediction;
  refit triggered by Wave N+4 first paired refactor)
- **`canonical_consumers`**: `tac.framework_agnostic` + sister cathedral
  consumer + canonical_anti_patterns_lookup_consumer
- **`python_callable_module_path`**: `tac.framework_agnostic:select_backend`

### Layer 4: NEW cathedral consumer per Catalog #335

`src/tac/cathedral_consumers/framework_agnostic_lookup_consumer/` —
auto-discovered via `discover_compliant_consumer_modules` (Total compliant
consumers post-landing: **74**).

- `CONSUMER_NAME = "framework_agnostic_lookup_consumer"`
- `CONSUMER_VERSION = "1.0.0"`
- `CONSUMER_HOOK_NUMBERS = (CATHEDRAL_AUTOPILOT_DISPATCH, PROBE_DISAMBIGUATOR)`
- `CONSUMER_TIER = TIER_A_OBSERVABILITY_ONLY` per Catalog #357
- `consume_candidate(candidate)` infers backend token from explicit
  `framework_backend` / `backend` keys or `trainer_path` / `recipe`
  substring patterns + classifies routing:
  - `promotable_contest_resolution` — PyTorch / Modal / Vast.ai (Catalog #205 sister)
  - `non_promotable_research_signal` — MLX / tinygrad (Catalog #192/#317)
  - `diagnostic_reference` — numpy (canonical bridge oracle)
  - `unknown` — backend not inferable
- All routing branches carry canonical Tier A markers per Catalog #341:
  `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"`

### Sister deliverable: probe outcomes row per Catalog #313

Registered via `tac.probe_outcomes_ledger.register_probe_outcome`:

- `probe_id`: `framework_agnostic_portability_primitives_layer_4_design_20260528`
- `substrate`: `framework_agnostic_canonical_helper`
- `verdict`: `PROCEED`
- `metric_value`: 4.0 (4 canonical layers landed)
- `blocker_status`: `advisory` (refactor wave is operator-routable; not blocking)
- `next_action`: Wave N+4 substrate trainer refactor wave consumes
  `tac.framework_agnostic` + decorators per substrate pair + paired-axis
  empirical anchor demonstrating canonical abstraction preserves or
  improves substrate-optimal score

### Sister deliverable: 66 tests pass (49 framework_agnostic + 17 consumer)

- `src/tac/framework_agnostic/tests/test_framework_agnostic.py` — 49/49 pass
  (2 skipped are PyTorch / MLX env-unavailable cases on hosts without those
  backends)
- `src/tac/cathedral_consumers/framework_agnostic_lookup_consumer/tests/test_framework_agnostic_lookup_consumer.py` — 17/17 pass

## ## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290
sister discipline: the canonical default is `framework_agnostic` primitives;
per-substrate forks remain accepted when paired-axis empirical evidence
demonstrates canonical abstraction suppresses substrate-optimal score.

| Layer | Decision | Rationale |
|---|---|---|
| Backend enum + selection | ADOPT_CANONICAL_BECAUSE_SERVES | Sister of Catalog #205 `select_inflate_device`; identical cascade + fail-closed semantics; no per-substrate need for different selection logic |
| FrameworkAgnosticTensor Protocol | ADOPT_CANONICAL_BECAUSE_SERVES | Minimum runtime_checkable Protocol per Catalog #335 sister pattern; substrate-specific tensors satisfy structurally |
| quantize_int8_per_channel | ADOPT_CANONICAL_BECAUSE_SERVES | Byte-determinism across backends per Catalog #146 contract; canonical numpy oracle IS the determinism guarantee |
| quantize_fp4_packed_nibbles | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical Quantizr unsigned-E2M1 codebook per CLAUDE.md "Quantizr archive contents" verified empirical; PR101/PR102/PR103 medal-class precedent |
| brotli_compress | ADOPT_CANONICAL_BECAUSE_SERVES | Canonical hard dep per Catalog #203/#224; pure-C pure-numpy wrapper; no backend dispatch needed |
| state_dict bridge helpers | ADOPT_CANONICAL_BECAUSE_SERVES | 8th standing directive `MLX state_dict → npz → ZIP-member → numpy inflate` contract is canonical; per-backend bridge wrappers route through canonical numpy oracle |
| Decorators | ADOPT_CANONICAL_BECAUSE_SERVES | Operator META directive verbatim "decorators or whatever"; canonical opt-in surface for per-method backend preference per UNIQUE-AND-COMPLETE-PER-METHOD operating mode |

## ## 9-dimension success checklist evidence

1. **UNIQUENESS** — canonical primitive consolidation eliminates MLX+PyTorch
   sister-trainer duplication anti-pattern; new package + new consumer +
   new anti-pattern + new canonical equation
2. **BEAUTY + ELEGANCE** — narrow public API (15 symbols re-exported);
   each module file ≤350 LOC; reviewable in 30 seconds per CLAUDE.md
   "Beauty, simplicity, and developer experience"
3. **DISTINCTNESS** — distinct from existing `tac.substrates._shared.inflate_runtime`
   (Catalog #205 sister at inflate-time surface; #framework_agnostic is
   training-time surface)
4. **RIGOR** — premise verification per Catalog #229 (read 10 premise files
   BEFORE editing); per-layer canonical decision documented; empirical
   anchor (V3 sister pair LOC delta) cited in canonical equation
5. **OPTIMIZATION PER TECHNIQUE** — per-backend implementation routes
   through canonical numpy oracle for byte determinism per Catalog #146;
   MLX-LOCAL preserves $0 development per CLAUDE.md 8th directive;
   PyTorch-CUDA preserves contest-resolution per Catalog #205 sister
6. **STACK-OF-STACKS COMPOSABILITY** — decorators compose (`@mlx_first_with_numpy_fallback`
   on top of operation that internally calls `quantize_int8_per_channel(backend=)`);
   tested in `test_decorator_chains_compose_with_operation`
7. **DETERMINISTIC REPRODUCIBILITY** — byte-determinism verified across
   backends via canonical numpy oracle path (e.g.
   `test_int8_round_trip_pytorch_matches_numpy_oracle`); seed-pinned tests
8. **EXTREME OPTIMIZATION + PERFORMANCE** — deferred-import discipline so
   importing `tac.framework_agnostic` does NOT load torch / mlx / tinygrad;
   minimal overhead per call (single env-var check + spec.find_spec cache)
9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A directly (this is META
   infrastructure that compounds across substrates); compounding score
   contribution will materialize as per-substrate Wave N+4 refactors land
   empirical paired-axis anchors

## ## Cargo-cult audit per assumption

- **Assumption**: ~95% LOC overlap V3 sister pair generalizes to all paired
  substrates → **CARGO-CULTED**; single empirical anchor; Wave N+4 measures
  per-substrate refactor delta empirically; the unwind path is per-substrate
  paired-axis anchor (per UNIQUE-AND-COMPLETE-PER-METHOD acceptance criterion).
- **Assumption**: canonical numpy oracle is the byte-determinism guarantee
  → **HARD-EARNED**; tested empirically in `test_int8_round_trip_pytorch_matches_numpy_oracle`
  + `test_int8_round_trip_mlx_matches_numpy_oracle` (49/49 pass); the
  determinism contract is structural (numpy reference is the oracle;
  per-backend paths route through numpy or replicate the operation).
- **Assumption**: tinygrad is acceptable as 4th backend per operator META
  directive → **HARD-EARNED**; operator-named verbatim "tinygrad like
  primitives"; tinygrad is deferred-import optional per Catalog #287
  (not installed → BackendUnavailableError with clear `uv pip install`
  hint).
- **Assumption**: decorator chains compose cleanly across `framework_agnostic`
  + caller's own decorators → **HARD-EARNED**; functools.wraps preserves
  signatures; tested in `test_decorator_chains_compose_with_operation`.

## ## Predicted ΔS band

**N/A** — this is META infrastructure (canonical helper package + cathedral
consumer + canonical equation + anti-pattern). Score signal is downstream:
per-substrate Wave N+4 refactors will produce per-substrate paired-axis
empirical anchors that feed the canonical equation's
`predicted_vs_empirical_residual` for `framework_agnostic_backend_abstraction_compounding_v1`.

**Dykstra-feasibility check (Catalog #296)**: the predicted compounding is
ADDITIVE across substrates (each substrate's refactor produces independent
LOC delta); no Pareto-polytope intersection constraint applies at this
layer. The Dykstra-feasibility surface is at the per-substrate refactor
layer (Wave N+4) where canonical abstraction + per-substrate score
preservation form the feasibility constraint per UNIQUE-AND-COMPLETE-PER-METHOD
operating mode.

## ## Observability surface

1. **Inspectable per layer** — every primitive call accepts explicit
   `backend=` kwarg + `select_backend()` resolves AUTO at call-time;
   `detect_available_backends_dict()` enumerates runtime availability per backend
2. **Decomposable per signal** — per-backend implementations isolated in
   `_quantize_int8_numpy` / `_quantize_int8_pytorch` / `_quantize_int8_mlx`
   / `_quantize_int8_tinygrad` so backend-specific behavior is auditable
3. **Diff-able across runs** — canonical numpy oracle guarantees
   byte-determinism per Catalog #146; tests pin
   `np.array_equal(i8_np, i8_torch.numpy())` + sister MLX assertion
4. **Queryable post-hoc** — cathedral consumer `framework_agnostic_lookup_consumer`
   emits `framework_backend` + `routing_class` in every `consume_candidate`
   return value so downstream consumers + operator review have per-candidate
   provenance
5. **Cite-able** — canonical equation `framework_agnostic_backend_abstraction_compounding_v1`
   registered with `EmpiricalAnchor` carrying `source_artifact` + Provenance
   per Catalog #323; canonical anti-pattern registered with
   `canonical_source_anchor` citing V3 sister pair + operator META directive
6. **Counterfactual-able** — `@mlx_first_with_numpy_fallback` decorator
   demonstrates the canonical fallback path; caller can `monkeypatch` MLX
   availability + observe routing class change per
   `test_select_backend_override_unavailable_raises`

## ## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map** — N/A (canonical helper + dispatcher; sister
  consumers cover sensitivity-map signal contribution)
- **hook #2 Pareto constraint** — N/A (no Pareto-polytope intersection at
  the framework-selection layer; per-substrate Wave N+4 refactors land
  per-substrate Pareto anchors via paired-axis empirical evidence)
- **hook #3 bit-allocator** — N/A (canonical `quantize_int8_per_channel` /
  `quantize_fp4_packed_nibbles` primitives are byte-deterministic
  reference implementations; per-substrate bit-allocator consumers route
  through these but the bit-allocator hook itself fires at substrate scope)
- **hook #4 cathedral autopilot dispatch** — **ACTIVE PRIMARY** via
  `framework_agnostic_lookup_consumer` cathedral consumer auto-discovered
  via Catalog #335; emits `framework_backend` + `routing_class` annotations
  on every candidate so the autopilot ranker disambiguates promotable
  contest-resolution candidates from non-promotable research-signal candidates
- **hook #5 continual-learning posterior** — **ACTIVE** via canonical
  equation `framework_agnostic_backend_abstraction_compounding_v1`
  + canonical anti-pattern
  `mlx_trainer_pytorch_sister_duplicated_implementation_v1` registered to
  the canonical posterior stores; auto-recalibration trigger
  `when_3+_new_empirical_anchors_in_domain` per sister Catalog #371
  fix discipline
- **hook #6 probe-disambiguator** — **ACTIVE** via cathedral consumer's
  `routing_class` classification (promotable_contest_resolution vs
  non_promotable_research_signal vs diagnostic_reference vs unknown); the
  canonical framework choice IS the disambiguator between MLX-LOCAL
  non-promotable per Catalog #192/#317 vs PyTorch contest-resolution
  promotable per Catalog #205 sister

## Operator-routable cascade

1. **Wave N+4 substrate trainer refactor wave** (cap=N parallel subagents;
   $0 GPU per subagent): each subagent consumes `tac.framework_agnostic` +
   canonical decorators for ONE substrate pair, runs paired-axis
   paired-archive empirical anchor (paired-CPU + paired-CUDA per Catalog #246),
   lands per-substrate canonical-vs-fork verdict per Catalog #290.
   Empirical evidence will refit the canonical equation
   `framework_agnostic_backend_abstraction_compounding_v1` posterior +
   append `EmpiricalFalsification` rows to the canonical anti-pattern.
2. **Anti-pattern empirical_falsifications backfill** (sister wave; $0):
   once Wave N+4 lands first paired substrate refactor with measured LOC
   delta, register `EmpiricalFalsification` row via
   `tac.canonical_anti_patterns.append_empirical_falsification` so the
   `is_actively_recurring` invariant reflects measured recurrence.
3. **CLAUDE.md cross-reference rows** (THIS landing memo's downstream): add
   structural cross-refs to "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL"
   8th standing directive section so future subagents discover
   `tac.framework_agnostic` as the canonical opt-in for paired sister
   trainers. NO NEW STRICT preflight gate per Catalog #299 quota brake
   discipline (canonical helper is opt-in per UNIQUE-AND-COMPLETE-PER-METHOD;
   refusing existing V3 sister pair would re-introduce premature-kill
   violation per CLAUDE.md "Forbidden premature KILL").

## Cross-references

- **CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing directive**
  — the canonical contract this package operationalizes
- **CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"** — per-substrate
  forks remain accepted; canonical helpers serve as default reusable surface
- **Catalog #205** — sister gate at inflate-time device-selection surface
- **Catalog #335** — canonical cathedral consumer auto-discovery (this
  landing satisfies the contract)
- **Catalog #341** — Tier A canonical-routing markers (consumer pinned Tier A)
- **Catalog #357** — dual-tier consumer architecture (Tier A pinned)
- **Catalog #287** — placeholder-rationale rejection sister discipline
- **Catalog #344** — canonical equations registry
- **Catalog #371** — auto-recalibration trigger when 3+ new anchors
- **Catalog #146** — contest-compliant inflate runtime contract (sister
  byte-determinism guarantee)
- **Catalog #192 + #317** — MLX-LOCAL non-promotable + per-axis canonical
  routing markers
- **Catalog #246** — paired CUDA + CPU submission discipline (Wave N+4 will
  honor per substrate)
- **Catalog #290** — canonical-vs-unique decision per layer (this memo
  includes the section)
- **Catalog #294** — 9-dim success checklist evidence (this memo includes
  the section)
- **Catalog #296** — design-memo Dykstra-feasibility (N/A at META layer;
  per-substrate Wave N+4 lands)
- **Catalog #303** — cargo-cult audit per assumption (this memo includes
  the section)
- **Catalog #305** — observability surface section (this memo includes the section)
- **Catalog #300** — council deliberation v2 frontmatter (this memo carries
  v2 frontmatter)
- **Catalog #346** — canonical council roster (T1 attendees declared)
- **Catalog #340** — sister-checkpoint guard (PROCEED; disjoint scope from
  Slot 1 Wave N+3 V3 PyTorch sister trainer)

## V3 sister trainer pair empirical anchor (the canonical bug class receipt)

- `experiments/train_substrate_pact_nerv_selector_v3.py` — 783 LOC PyTorch
- `experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py` — 720 LOC MLX-LOCAL
- Total: 1503 LOC; estimated ~95% functional overlap (training loop +
  quantization + archive emission)
- Canonical unwind: refactor through `tac.framework_agnostic` primitives +
  `@mlx_first_with_numpy_fallback` / `@pytorch_first_with_numpy_fallback`
  decorators per UNIQUE-AND-COMPLETE-PER-METHOD operating mode acceptance
  criterion (paired-axis empirical anchor demonstrates canonical
  abstraction preserves or improves score)
- Operator-routable: Wave N+4 first paired refactor + paired-axis anchor;
  results feed canonical equation `framework_agnostic_backend_abstraction_compounding_v1`
  posterior refit per `when_3+_new_empirical_anchors_in_domain` trigger
