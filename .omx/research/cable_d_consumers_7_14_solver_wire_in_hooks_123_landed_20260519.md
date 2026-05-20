---
landing_kind: subagent
subagent_id: claude_slot_hh_consumers_solver_wire_in_hooks_123_20260519
lane_id: lane_cable_d_consumers_7_14_solver_wire_in_hooks_123_20260519
landed_at_utc: 2026-05-20T02:05:00Z
canonical_helper: tac.cathedral_solver_wire_in.consumers_7_14_contributions
operator_routable_summary: 9 HARD-EARNED (consumer × hook) ACTIVE solver-surface contributions per Catalog #303 cargo-cult audit; closes producer → sidecar → ranker (FF) → solver (HH) loop per Catalog #125 hooks #1+#2+#3 for the 6 Cable D consumers (7+8+9+10+12+13); observability-only per Catalog #287/#323/#341
horizon_class: apparatus_maintenance
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
---

# Cable D consumers 7-14 → SOLVER-SURFACE wire-in hooks #1+#2+#3 landed (Slot HH)

Per Slot FF cathedral autopilot cascade wire-in landing memo
(`.omx/research/cable_d_consumers_7_14_autopilot_cascade_wire_in_landed_20260519.md`,
commit `d7c28c737`) highest-EV op-routable:

> *"Wire the 6 Cable D consumer sidecars BACK into canonical sensitivity_map +
> pareto + bit_allocator solver surfaces per the producer headers' 'sister
> subagent owns wiring' declarations. The CONSUMER side is now landed at the
> ranker (Slot FF); the SOLVER side (`tac.optimization.pareto` +
> `tac.sensitivity_map`) closes the canonical 6-hook producer → sidecar →
> ranker → solver loop per Catalog #125. Est. ~50-100 LOC + ~15-20 tests per
> consumer."*

Per Catalog #125 6-hook wire-in non-negotiable: hook #4 (cathedral autopilot
dispatch) was ACTIVE for these 6 consumers post-FF (commit `d7c28c737`); hooks
#1 (sensitivity-map) + #2 (Pareto constraint) + #3 (bit-allocator) were OPEN.
This landing closes hooks #1+#2+#3.

## Phase 1: identified 9 HARD-EARNED cells via Catalog #303 cargo-cult audit

The FF prompt asks "wire 3 hooks × 6 consumers (target: 18)". A naive
"all × all" reading produces 18 cells. **This is CARGO-CULTED per Catalog #303.**

The HARD-EARNED inventory per the producer headers' wire-in declarations is
**9 ACTIVE cells**:

| # | Consumer                             | #1 Sensitivity | #2 Pareto | #3 BitAlloc | Producer-header source           |
|---|--------------------------------------|----------------|-----------|-------------|----------------------------------|
| 7 | `per_pair_pareto_envelope`           |       N/A      |  ACTIVE   |     N/A     | "feeds `tac.optimization.pareto`" |
| 8 | `per_pair_lagrangian_lambda_bisection` |   ACTIVE     |  ACTIVE   |     N/A     | "per-pair λ_R feeds `tac.sensitivity_map.axis_level_reweight`" + "feeds `tac.optimization.pareto`" |
| 9 | `per_pair_lora_supervision_signal`   |       N/A      |    N/A    |   ACTIVE    | "Feeds `tac.optimization.bit_allocator`" |
| 10| `per_pair_coding_budget_allocation`  |       N/A      |    N/A    |   ACTIVE    | "feeds `tac.optimization.bit_allocator`" |
| 12| `per_pair_kkt_residuals`             |     ACTIVE     |  ACTIVE   |     N/A     | "feeds `tac.optimization.pareto`" + canonical disambiguator (hook #6) |
| 13| `per_pair_volterra_cross_terms`      |     ACTIVE     |  ACTIVE   |     N/A     | "feed `tac.sensitivity_map.*` for second-order sensitivity analysis + `tac.optimization.pareto`" |

**Total: 9 ACTIVE + 9 N/A = 18 cells.** The N/A cells are EXPLICITLY
classified (not silent omissions). Per Catalog #303 + CLAUDE.md "Apples-to-apples
evidence discipline": forcing N/A cells would create **phantom contributions
whose existence the producer never declared** — that IS the canonical
orphan-signal-WITH-FAKE-CLAIM bug class.

The Catalog #303 audit verdict is recorded in the canonical helper's docstring
+ enforced structurally via `SolverHookContribution.__post_init__` invariant
that REFUSES construction of (consumer × hook) cells not in
`CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY`.

## Phase 2: canonical solver-wire-in package

**Decision per Catalog #290 canonical-vs-unique per layer:**

| Layer | Decision |
|---|---|
| Wire-in package location | **UNIQUE** — NEW canonical sub-package `src/tac/cathedral_solver_wire_in/` separate from `src/tac/sensitivity_map/` + `src/tac/bit_allocator.py` + `src/tac/contest_oracle/pareto_frontier.py`. Rationale per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: the existing solver-surface modules are stable canonical surfaces with their own dedicated test corpora; mutating them to add Cable-D-specific contribution accessors would conflate the solver primitive with the consumer plumbing. The new package is the canonical entry point sister subagents can extend WITHOUT touching the solver primitives. |
| Contribution dataclass | **UNIQUE** — NEW `SolverHookContribution` frozen dataclass with 6 canonical fields + 4 canonical non-promotable invariants (`score_claim_valid=False`, `promotion_eligible=False`, `axis_tag="[predicted]"`, `predicted_delta_adjustment=0.0`) enforced via `__post_init__`. Reuses pattern from `tac.cathedral.verdict_ledger` + `tac.cathedral.consumer_contract`. |
| Sidecar discovery | **ADOPT_CANONICAL** — reuse FF helpers `_latest_cable_d_consumer_sidecar_for_archive` + `_cable_d_consumer_sidecar_carries_structural_signal` via `importlib.util` lazy load. Per Catalog #110/#113 APPEND-ONLY: we DO NOT mutate the FF helpers; we read them via canonical importlib pattern (`importlib.util.spec_from_file_location` + cached module). |
| Provenance contract | **ADOPT_CANONICAL** — every contribution carries canonical `[predicted]` axis tag per Catalog #287, `score_claim_valid=False` + `promotion_eligible=False` per Catalog #323, `predicted_delta_adjustment=0.0` per Catalog #341 (the FF ranker cascade already applied the 1.01× multiplicative reward at the ranker surface; SOLVER-side adjustment would be a second-order double-count). |
| Hook taxonomy | **UNIQUE** — `CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY` frozen mapping; explicit per-consumer × per-hook eligibility. Sister of `HookNumber` enum from `tac.cathedral.consumer_contract` but as a `frozenset[str]` registry (mirrors the per-consumer `CONSUMER_HOOK_NUMBERS` tuples in each consumer's `__init__.py`). |
| Cell-refusal semantics | **UNIQUE** — N/A cells refused at construction time via `__post_init__` invariant + via the dispatcher functions (`sensitivity_map_contribution_for_consumer` etc) that raise `ValueError` with explicit Catalog #303 citation. |
| Test corpus location | **ADOPT_CANONICAL** — `src/tac/tests/test_consumers_7_14_solver_wire_in.py` mirrors sister tests `test_cathedral_consumers_7_14_cascade.py` (FF) + `test_cable_d_wire_in_master_gradient_consumers.py` (Cable D batch). |

### 4 surfaces

1. **`CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY`** (frozen Mapping) — the canonical
   HARD-EARNED registry. 6 consumers × frozenset[str] of legal hooks.

2. **`SolverHookContribution`** (frozen dataclass) — typed observability-only
   contribution with 11 fields + 4 canonical invariants enforced via
   `__post_init__` (per Catalog #287/#323/#341).

3. **Per-hook dispatcher accessors** — `sensitivity_map_contribution_for_consumer(consumer_id, archive_sha256)` + `pareto_constraint_contribution_for_consumer(...)` + `bit_allocator_contribution_for_consumer(...)`. Each refuses N/A cells with `ValueError` citing Catalog #303.

4. **Collection helper** — `collect_all_solver_contributions_for_archive(archive_sha256)` returns the tuple of 9 ACTIVE contributions for the given archive.

5. **Canonical-disambiguator helpers** — `consumer_owns_hook(consumer_id, hook_name)` returns True iff (consumer × hook) is HARD-EARNED ACTIVE; `is_solver_contribution_promotable(contribution)` always returns False per Catalog #341 invariant.

## Phase 3: tests

**File**: `src/tac/tests/test_consumers_7_14_solver_wire_in.py` (51 tests, 0.25s wall-clock, ALL PASS).

| Test group | Count | Coverage |
|---|---|---|
| Constants + canonical contract pinned | 5 | observability axis / hook pair count == 9 / 6 consumers / per-consumer hook set / 6 sidecar entries |
| `consumer_owns_hook` canonical disambiguator | 4 | active cells / N/A cells return False / unknown consumer / unknown hook |
| Hook #1 sensitivity per-consumer happy paths | 4 | absent + present per #8 / present per #12 / present per #13 |
| Hook #2 Pareto per-consumer happy paths | 4 | present per #7 / present per #8 / present per #12 / present per #13 |
| Hook #3 bit-allocator per-consumer happy paths | 3 | absent + present per #9 / present per #10 |
| Catalog #303 N/A cell refusal (parametrized) | 9 | each of the 9 N/A cells raises ValueError |
| Catalog #287/#323/#341 canonical non-promotable invariants | 6 | score_claim forbidden / promotion_eligible forbidden / axis must be predicted / predicted_delta_adjustment must be 0 / is_promotable always False / phantom-cell construction refused |
| Cross-hook consistency | 2 | consumer #8 consistent across sensitivity + pareto / consumer #13 consistent |
| `collect_all_solver_contributions_for_archive` | 4 | returns 9 / all observability-only / one sidecar staged / empty sha rejected |
| Custody-leak guards (FF helper integration) | 5 | score_claim=True rejected / promotion_eligible=True rejected / cross-archive sha rejected / wrong schema rejected / trivial signal rejected |
| Sister-regression smoke | 2 | cathedral consumer count >= 34 unchanged / FF cascade helpers callable |
| Live-repo regression guards | 3 | canonical sidecar registry pinned to 6 / no phantom solver hook in pairs / HARD-EARNED cells match producer headers |
| **Total** | **51** | **PASS** |

**Sister regression sweep (183 sister tests across 7 files, 1.32s wall-clock, ALL PASS):**

| File | Tests | Status |
|---|---|---|
| `test_cathedral_consumers_7_14_cascade.py` (FF) | 35 | PASS |
| `test_cable_d_wire_in_master_gradient_consumers.py` (AA Cable D batch) | 33 | PASS |
| `test_check_335_cathedral_consumer_directory_contract.py` | 18 | PASS |
| `test_cathedral_consumer_contract.py` | 30 | PASS |
| `test_cathedral_autopilot_auto_discovery.py` | 13 | PASS |
| `test_check_336_337_cathedral_main_discovery_invoker.py` | 27 | PASS |
| `test_check_341_cathedral_consumer_mps_prescreen_routing.py` | 27 | PASS |

**Aggregate**: 51 NEW + 183 SISTER = **234 tests PASS** across the canonical
solver-wire-in surface.

## Phase 4: live cathedral autopilot smoke

```
Live cathedral consumer count: 34
```

Consumer count UNCHANGED (was 34 per Slot FF baseline). The Slot HH work does
NOT add new consumers; it adds a SOLVER-SIDE canonical helper package that
sister subagents (cathedral autopilot ranker / future Pareto solver / future
bit-allocator coordinator) can consume.

## 6-hook wire-in declaration (per Catalog #125)

This landing closes hooks #1 + #2 + #3 for the 6 Cable D consumers:

1. **SENSITIVITY MAP** (hook #1): **ACTIVE** for consumers #8 + #12 + #13 (3 of 6).
   Each contribution exposes `n_pairs` + `n_bytes` + canonical sidecar
   `sidecar_path` for cite-chain. Downstream `tac.sensitivity_map.*`
   consumers can call `sensitivity_map_contribution_for_consumer(consumer_id,
   archive_sha)` to get the typed contribution.

2. **PARETO CONSTRAINT** (hook #2): **ACTIVE** for consumers #7 + #8 + #12 + #13
   (4 of 6). Each contribution exposes the structural-signal markers for
   downstream Pareto-frontier integration. Per `tac.boosting.pareto_front.ParetoFrontTracker`
   sister discipline: contributions are tagged `[predicted]` axis so they
   cannot accidentally enter `[contest-CUDA]` or `[contest-CPU]` Pareto
   frontiers.

3. **BIT ALLOCATOR** (hook #3): **ACTIVE** for consumers #9 + #10 (2 of 6).
   Each contribution exposes per-pair signals for downstream
   `tac.bit_allocator.allocate_bits` integration.

4. **CATHEDRAL AUTOPILOT DISPATCH** (hook #4): **already ACTIVE post-FF**
   (commit `d7c28c737`). The Slot FF cascade applies the 1.01× per-sidecar
   multiplicative reward at the ranker surface. This SOLVER-side wire-in
   composes ON TOP of hook #4 (the FF ranker already reflects sidecar
   presence in candidate ordering; this layer makes the canonical solver
   surfaces aware of the same sidecars for downstream integration).

5. **CONTINUAL-LEARNING POSTERIOR** (hook #5): **N/A AT SOLVER LAYER** — the
   sidecars themselves ARE the canonical posterior writes (per `tac.master_gradient_consumers.consumer_output_path`
   sister discipline). The SOLVER-side dispatchers are read-only; the FF
   cascade does not mutate posterior state either. Sister consumer
   `update_from_anchor` hooks (which are NO-OPs by design per the per-consumer
   `__init__.py` files) remain the canonical posterior-update entry point.

6. **PROBE-DISAMBIGUATOR** (hook #6): **ACTIVE for consumer #12** per its
   producer header declaration. The `per_pair_kkt_residuals` contribution
   IS the canonical per-pair stationarity disambiguator (HIGH residual =
   joint codec failing to balance distortion vs rate at that pair). The
   SOLVER-side `pareto_constraint_contribution_for_consumer('per_pair_kkt_residuals', ...)`
   surface is the canonical entry point downstream consumers query for this
   disambiguator.

## Cargo-cult audit per assumption

| Assumption | Classification |
|---|---|
| "All 6 consumers × all 3 hooks = 18 cells must be wired" | **CARGO-CULTED** — the "all × all" reflex creates phantom contributions whose existence the producer never declared. Per Catalog #303 + CLAUDE.md "Apples-to-apples evidence discipline" the HARD-EARNED inventory is 9 cells per producer-header declarations. The 9 N/A cells are explicitly classified (not silent omissions) and refused at construction time via `SolverHookContribution.__post_init__` invariant. |
| "More solver-surface contributions = better optimization" | **HARD-EARNED-CONSERVATIVE** — each contribution is observability-only ([predicted] axis, promotion_eligible=False, predicted_delta_adjustment=0.0); they expose canonical structural signals for downstream solver integration WITHOUT mutating scores or promotion. The 9 ACTIVE cells extinct the orphan-signal-at-cathedral-autopilot bug class at THREE NEW surfaces (hooks #1+#2+#3) beyond the FF ranker surface (hook #4); the 9 N/A cells DEFEND against phantom-contribution leakage per Catalog #341. |
| "Solver wire-in needs new mutations to `tac.sensitivity_map.*` + `tac.optimization.pareto` + `tac.bit_allocator`" | **CARGO-CULTED** — per Catalog #290 canonical-vs-unique per layer, mutating those existing canonical primitives to add Cable-D-specific accessors would conflate solver primitive with consumer plumbing. The NEW canonical sub-package `tac.cathedral_solver_wire_in` is the canonical entry point sister subagents extend WITHOUT touching the solver primitives. Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline applies to the existing solver-primitive modules; the NEW package is the canonical extension surface. |
| "Sidecar present + custody-clean is sufficient for solver contribution" | **HARD-EARNED** — the FF helper's 6-condition `_cable_d_consumer_sidecar_carries_structural_signal` gate (schema + archive_sha + score_claim=False + promotion_eligible=False + non-trivial signal + JSON parses) IS the canonical custody gate. The SOLVER-side wire-in delegates to it; no additional gates are added (per CLAUDE.md "Subagent coherence-by-default" — the FF gate is the canonical sister; adding parallel validation would be META-cargo-culted). |
| "Hook #5 continual-learning posterior should be ACTIVE at solver layer" | **HARD-EARNED-N/A** — the sidecars themselves ARE the canonical posterior writes (per `tac.master_gradient_consumers.consumer_output_path`). The SOLVER-side dispatchers are stateless read-only functions; the FF cascade does not mutate posterior either. Forcing hook #5 at solver layer would create a duplicate write path — exactly the bug class Catalog #131 fcntl-locked bare-write discipline extincts. |

## Observability surface (per Catalog #305)

The 6-facet observability declaration for the canonical solver-wire-in package:

- **Inspectable per layer**: `consumer_owns_hook(consumer_id, hook_name)` exposes the per-cell HARD-EARNED-vs-N/A classification; `CABLE_D_CONSUMERS_7_14_HOOK_REGISTRY` is a frozen Mapping queryable directly.
- **Decomposable per signal**: each `SolverHookContribution` carries explicit `n_pairs` + `n_bytes` + `sidecar_path` + `sidecar_schema` fields; downstream consumers can decompose contributions per (consumer × hook) cell.
- **Diff-able across runs**: stateless functions (no module-level mutable state). Same `(consumer_id, hook_name, archive_sha256, sidecar set)` always returns the same `SolverHookContribution` payload.
- **Queryable post-hoc**: `collect_all_solver_contributions_for_archive(archive_sha256)` returns the canonical tuple of 9 contributions for the given archive; sister subagents can iterate this surface.
- **Cite-able**: every contribution carries `sidecar_path` (canonical sidecar file path) + `sidecar_schema` (canonical schema tag) + `provenance_canonical_helper` (canonical helper module name) for full cite-chain.
- **Counterfactual-able**: removing any sidecar file from `.omx/state/master_gradient_consumers/` zeros that consumer × hook contribution (n_pairs=0 + n_bytes=0 + sidecar_present=False); operator can probe per-cell counterfactual impact by staging/removing individual sidecars.

## 9-dimension success checklist evidence

1. **UNIQUENESS**: NEW canonical sub-package `tac.cathedral_solver_wire_in` is unique surface; no sister package wires Cable D consumers 7-14 into solver-surface contracts.
2. **BEAUTY + ELEGANCE**: 425 LOC across 2 files (`__init__.py` + `consumers_7_14_contributions.py`); reviewable in 30 seconds; canonical Catalog #303 cargo-cult audit table embedded in module docstring.
3. **DISTINCTNESS**: distinct from FF cascade (which is RANKER-side multiplicative reward) AND distinct from sister `tac.cathedral_consumers/*` packages (which are CATHEDRAL-AUTOPILOT-CONSUMER side); the solver-wire-in is a NEW orthogonal axis closing hooks #1+#2+#3.
4. **RIGOR**: premise verification (Catalog #229) — read FF landing memo + 6 consumer `__init__.py` files + canonical solver-surface modules + FF cascade implementation before designing. Empirical anchor: 51/51 tests pass + 183/183 sister tests pass + cathedral consumer count unchanged at 34.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 canonical-vs-unique decision per layer (see table above); 6 explicit per-layer decisions documented.
6. **STACK-OF-STACKS COMPOSABILITY**: composes ON TOP of FF ranker surface; sister subagents extending other consumer batches can add new contribution dispatchers to `tac.cathedral_solver_wire_in` without touching this landing.
7. **DETERMINISTIC REPRODUCIBILITY**: stateless functions; same `(consumer_id, hook_name, archive_sha)` + same sidecar set = same `SolverHookContribution`. Test corpus uses `uuid.uuid4()` fresh-sha pattern for isolation; cleanup via `_staged_canonical_sidecar` context manager.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 0.25s wall-clock for 51 tests; `importlib.util` cache memoizes FF helper module load (`sys.modules` cache).
9. **OPTIMAL MINIMAL CONTEST SCORE**: this is an APPARATUS_MAINTENANCE landing per `horizon_class` frontmatter. The score-lowering effect lives at downstream sister subagents that consume `collect_all_solver_contributions_for_archive` to integrate the 6 Cable D consumer signals into solver-surface optimization loops.

## Cross-references

- Slot FF cathedral autopilot cascade wire-in: `feedback_cable_d_consumers_7_14_autopilot_cascade_wire_in_landed_20260519.md` + `cable_d_consumers_7_14_autopilot_cascade_wire_in_landed_20260519.md` (the upstream landing whose highest-EV op-routable this work addresses)
- Slot AA Cable D wire-in batch: `cable_d_wire_in_batch_landed_20260519.md` (commit `3af31f709`; the Cable D consumers 7-14 build wave)
- Slot Z cathedral autopilot activation: `cathedral_autopilot_activation_945_landed_20260519.md` (the unblocking landing per Catalog #335 paradigm shift)
- Cable D batch landing: `feedback_cable_d_master_gradient_extension_batch_landed_20260519T055121Z.md` (commit `6a1e94a63`; the producer-side landing of the 8 Cable D consumers 7-14)
- Sister catalog gates:
  - Catalog #125 (subagent landing 6-hook wire-in non-negotiable; this landing closes hooks #1+#2+#3)
  - Catalog #127 (per-call-site custody routing; SolverHookContribution honors via score_claim_valid=False + promotion_eligible=False invariants)
  - Catalog #131 (fcntl-locked bare-write discipline; the canonical sidecar root)
  - Catalog #138 (strict-load fail-closed sister discipline; `(OSError, json.JSONDecodeError)` graceful degradation)
  - Catalog #185 (META-meta-meta CLAUDE.md catalog drift detection; sister-callable regression test)
  - Catalog #229 (premise verification before edit; PV pre-flight read all required source files)
  - Catalog #230 (sister-subagent ownership map; CC + DD + EE + FF + GG disjoint at landing)
  - Catalog #287 (canonical Provenance umbrella; every contribution tagged `[predicted]` axis)
  - Catalog #290 (canonical-vs-unique decision per layer; 6 explicit decisions in this memo)
  - Catalog #294 (9-dimension success checklist evidence; section above)
  - Catalog #303 (cargo-cult audit per assumption; section above + structural enforcement via `__post_init__`)
  - Catalog #305 (observability surface declaration; section above)
  - Catalog #318 (master-gradient raw-byte-authority guard; no raw byte tensors exposed)
  - Catalog #321/#322/#323 (phantom-score-from-research-sidecar guard family; tests verify score_claim=True sidecars rejected)
  - Catalog #335 (cathedral consumer canonical contract; consumer count unchanged at 34)
  - Catalog #336/#337 (cathedral autopilot main invocation + master-gradient rerank; this landing does NOT touch invocation surface)
  - Catalog #340 (sister-checkpoint guard; PROCEED per ownership map at landing)
  - Catalog #341 (cathedral consumer routing canonical markers; SolverHookContribution honors all 3 canonical markers)

## Sister coordination

| Sister | Scope | Coordination |
|---|---|---|
| DD `adab84c8aba6dbc5f` B6 council symposiums | `.omx/research/council_t3_*_20260519.md` + `.omx/state/council_deliberation_posterior.jsonl` | DISJOINT — my scope is `src/tac/cathedral_solver_wire_in/` + `src/tac/tests/test_consumers_7_14_solver_wire_in.py` + this memo only |
| EE `af7545016a7255569` master_gradient_xray VIZ | `tools/master_gradient_xray.py` + sister test files | DISJOINT (different tool) |
| GG `a866a992b00edc7f2` B1 E.7 remediate | `.omx/operator_authorize_recipes/` + dispatch state files | DISJOINT |
| FF `cable_d_consumers_7_14_autopilot_cascade` (already landed; commit `d7c28c737`) | `tools/cathedral_autopilot_autonomous_loop.py` | DISJOINT — this landing imports FF helpers via `importlib.util` (read-only); no mutation |

Catalog #340 sister-checkpoint guard: PROCEED (no file overlap with any active sister).

## Highest-EV op-routable surfaced

**Wire hook #5 (continual-learning posterior) + hook #6 (probe-disambiguator)
for the 6 Cable D consumers via dedicated query helpers:**

- Hook #5: a canonical `tac.cathedral_solver_wire_in.query_posterior_for_consumer(consumer_id, archive_sha)`
  that reads the canonical sidecar JSON + emits a typed posterior anchor for
  downstream `tac.continual_learning.posterior_update_locked` consumption.
  Per Catalog #131 fcntl-locked discipline + Catalog #138 strict-load
  discipline. Estimated scope: ~150 LOC + ~25 tests = sister-subagent slot.

- Hook #6: extend the `consumer_12_kkt_residuals` per-pair disambiguator
  surface into a canonical `tac.cathedral_solver_wire_in.disambiguate_per_pair_stationarity(archive_sha)`
  that returns per-pair stationarity-residual ranks (top-N pairs with
  highest KKT residual = highest leverage for next dispatch). Per CLAUDE.md
  "Meta-Lagrangian/Pareto solver" + Catalog #313 probe-outcomes ledger.
  Estimated scope: ~200 LOC + ~30 tests = sister-subagent slot.

**Sister benefit**: with the SOLVER side now landed for hooks #1+#2+#3, the
remaining unclosed surfaces are hooks #5+#6. Together with hook #4 (FF), this
would close the FULL 6-hook loop for all 6 Cable D consumers per Catalog #125
non-negotiable.

## Lane status

- Lane `lane_cable_d_consumers_7_14_solver_wire_in_hooks_123_20260519`
  registered at L0 (pre-registration via memory entry).
- Gates landed in this commit batch:
  - `impl_complete` ✓ (3 dispatcher functions + 1 collection helper + 2 canonical-disambiguator helpers + 1 typed dataclass + 51 dedicated tests + memory entry)
  - `memory_entry` ✓ (this memo)
- Gates pending sister-subagent or operator follow-on:
  - `real_archive_empirical` — N/A (canonical helper is SOLVER-side read-only; no archive bytes generated)
  - `contest_cuda` — N/A (canonical helper does not produce score claims)
  - `strict_preflight` — no NEW strict preflight gate needed; sister Catalog #335 + #341 + #318 + #319 + #321/#322/#323 + #287 + #323 cover the consumer contract + routing markers + raw-byte guard + Lagrangian PRIMARY + phantom-score family + canonical Provenance umbrella. The SolverHookContribution invariants enforce custody-clean + non-promotable + observability-only at construction time.
  - `three_clean_review` — adversarial review cycle (next subagent slot)
  - `deploy_runbook` — N/A for editor-only solver-wire-in work

Expected lane level after this commit: **L1** (impl_complete + memory_entry).

— Slot HH Cable D consumers 7-14 → SOLVER-SURFACE wire-in subagent 2026-05-20
(claude_slot_hh_consumers_solver_wire_in_hooks_123_20260519)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:cable-D-solver-wire-in-hooks-123-landing-memo-trigger-tokens-describe-meta-Lagrangian-existing-equations-cited -->
