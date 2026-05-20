---
landing_kind: subagent
subagent_id: claude_slot_ll_cable_d_hooks_5_6_closure_20260519
lane_id: lane_cable_d_consumers_7_14_hooks_5_6_closure_20260519
landed_at_utc: 2026-05-20T02:22:00Z
canonical_helper: tac.cathedral_solver_wire_in (hook5_continual_learning + hook6_probe_disambiguator)
operator_routable_summary: 2 HARD-EARNED (consumer × hook) ACTIVE cells per Catalog #303 cargo-cult audit + producer-header CONSUMER_HOOK_NUMBERS declarations; closes FULL Catalog #125 6-hook loop for the 6 Cable D consumers (7+8+9+10+12+13) via the AA → FF → HH → LL cascade; observability-only per Catalog #287/#323/#341
horizon_class: apparatus_maintenance
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
---

# Cable D consumers 7-14 → hooks #5 + #6 FULL 6-hook closure landed (Slot LL)

Per Slot HH solver wire-in landing memo
(`.omx/research/cable_d_consumers_7_14_solver_wire_in_hooks_123_landed_20260519.md`,
commit `cb3a53860`) highest-EV op-routable:

> *"Wire hooks #5 (continual-learning posterior) + #6 (probe-disambiguator)
> for the 6 Cable D consumers via dedicated query helpers
> (`query_posterior_for_consumer` + `disambiguate_per_pair_stationarity`).
> Together with hook #4 (FF) + #1+#2+#3 (HH) this closes the FULL 6-hook
> loop per Catalog #125. Est. ~350 LOC + ~55 tests = sister-subagent slot."*

Per Catalog #125 6-hook wire-in non-negotiable: hooks #1+#2+#3 (HH) + hook #4
(FF) were ACTIVE pre-this-landing; hooks #5+#6 were OPEN. This landing closes
hooks #5+#6.

## Phase 1: identified 2 HARD-EARNED cells via Catalog #303 cargo-cult audit

The HH precedent established that the operator prompt's "wire 3 hooks × 6
consumers (target: 18)" framing is CARGO-CULTED per Catalog #303; HH closed
9 HARD-EARNED ACTIVE cells (NOT 18) per the producer-header declarations.

This landing applies the SAME discipline for hooks #5+#6. The operator prompt
asks "wire 2 hooks × 6 consumers (target: 12)"; the HARD-EARNED inventory
per producer-header `CONSUMER_HOOK_NUMBERS` declarations is **2 ACTIVE cells**,
NOT 12:

| # | Consumer                              | `CONSUMER_HOOK_NUMBERS` | Hook #5 | Hook #6 | Source                       |
|---|---------------------------------------|-------------------------|---------|---------|------------------------------|
| 7 | `per_pair_pareto_envelope`            | (PARETO, AUTOPILOT)     | N/A     | N/A     | NO-OP per producer header    |
| 8 | `per_pair_lagrangian_lambda_bisection`| (SENSITIVITY, AUTOPILOT)| N/A     | N/A     | NO-OP per producer header    |
| 9 | `per_pair_lora_supervision_signal`    | (AUTOPILOT, **CL_POSTERIOR**) | **ACTIVE** | N/A | declares CONTINUAL_LEARNING_POSTERIOR |
| 10| `per_pair_coding_budget_allocation`   | (BIT_ALLOCATOR, AUTOPILOT) | N/A   | N/A     | NO-OP per producer header    |
| 12| `per_pair_kkt_residuals`              | (PARETO, AUTOPILOT, **PROBE_DISAMBIG**) | N/A | **ACTIVE** | declares PROBE_DISAMBIGUATOR |
| 13| `per_pair_volterra_cross_terms`       | (SENSITIVITY, PARETO, AUTOPILOT) | N/A | N/A | NO-OP per producer header    |

**Total: 2 ACTIVE + 10 N/A = 12 cells.**

The 10 N/A cells are EXPLICITLY classified per the producer-header
`CONSUMER_HOOK_NUMBERS` tuples (not silent omissions). Per Catalog #303 +
CLAUDE.md "Apples-to-apples evidence discipline": forcing N/A cells would
create **phantom contributions whose existence the producer never declared**.

For hook #5 specifically: the 5 N/A consumers each carry the verbatim
producer-header rationale: *"Per-pair <X> sidecar JSON is canonically
persisted via `tac.master_gradient_consumers.consumer_output_path` at
`.omx/state/master_gradient_consumers/`. NO-OP by design."* — i.e. the
sidecar IS the canonical posterior write surface; a SECOND write would be
a Catalog #131 duplicate-write bug class.

For hook #6 specifically: the 5 N/A consumers each have non-disambiguating
signal types per their producer headers — Pareto envelope is an unambiguous
frontier; λ_R bisection is unambiguous when converged; LoRA targets are
derived (not disambiguating); coding-budget is a primary decision (not a
disambiguator); Volterra cross-terms are a one-shot characterization matrix
(not a runtime ambiguity-resolver).

The Catalog #303 audit verdict is recorded in both module docstrings +
enforced structurally via `Hook5PosteriorAnchor.__post_init__` +
`Hook6DisambiguatorVerdict.__post_init__` invariants that REFUSE construction
of (consumer × hook) cells not in the HARD-EARNED ACTIVE sets.

## Phase 2: canonical hook #5 + hook #6 modules per HH pattern

**Decision per Catalog #290 canonical-vs-unique per layer:**

| Layer | Decision |
|---|---|
| Module location | **ADOPT_CANONICAL_BECAUSE_SERVES** — extend the HH `src/tac/cathedral_solver_wire_in/` package with NEW `hook5_continual_learning.py` + `hook6_probe_disambiguator.py` modules per HH's own Catalog #290 pattern. Rationale per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: HH's `consumers_7_14_contributions.py` is the canonical hook-1+2+3 surface; mutating it to add hook-5+6 surfaces would conflate the orthogonal hooks. NEW modules in the SAME package keep the cite-chain tight + the test corpus colocated. |
| Anchor + Verdict dataclass | **UNIQUE-PER-HOOK** — `Hook5PosteriorAnchor` (10 canonical fields + 3 invariants per Catalog #287/#323/#341) for hook #5; `Hook6DisambiguatorVerdict` (14 canonical fields + 5 invariants including Catalog #313 verdict-taxonomy alignment) for hook #6. Each enforces N/A-cell refusal at `__post_init__` time. |
| Sidecar discovery | **ADOPT_CANONICAL** — both modules reuse HH's `_build_contribution_payload(consumer_id, archive_sha256)` from `consumers_7_14_contributions` (which itself delegates to the FF helpers `_latest_cable_d_consumer_sidecar_for_archive` + `_cable_d_consumer_sidecar_carries_structural_signal`). This is the canonical 6-condition sidecar gate (schema + custody + non-trivial signal) — no second gate added. |
| Cite-chain SHA-256 | **UNIQUE-PER-LL** — both LL modules add canonical SHA-256 of the sidecar JSON (`_sha256_of_sidecar`) for cite-chain provenance, NOT present in HH's contribution. Rationale: continual-learning posterior + probe-disambiguator anchors are MORE persistent than ranker contributions; the cite-chain SHA is the canonical mechanism for sister consumers to verify they read the SAME sidecar that produced this anchor. |
| Provenance contract | **ADOPT_CANONICAL** — every anchor + verdict carries canonical `[predicted]` axis tag per Catalog #287, `score_claim_valid=False` + `promotion_eligible=False` per Catalog #323, NO `predicted_delta_adjustment` field (posterior + disambiguator surfaces don't adjust deltas; they emit typed observability payloads). |
| Hook #6 verdict taxonomy | **ADOPT_CANONICAL** — `Hook6DisambiguatorVerdict.verdict` field defaults to `"PARTIAL"` and is validated against `tac.probe_outcomes_ledger.VALID_VERDICTS` (`{INDEPENDENT, KILL, DEFER, PROMOTE, PROCEED, PARTIAL, OPERATOR_REVIEW_REQUIRED}`). Per CLAUDE.md "Forbidden premature KILL without research exhaustion": per-pair rank is INFORMATIONAL, not blocking → default PARTIAL. |
| Top-N rank extraction | **UNIQUE-PER-LL** — hook #6 reads canonical KKT sidecar payload keys (`per_pair_residual_magnitudes` / `residual_magnitudes` / `per_pair_kkt_residual` / `per_pair_residuals`) and extracts top-N pairs by descending magnitude. NOT in HH (HH only emits structural counts). |
| Cell-refusal semantics | **ADOPT_CANONICAL-FROM-HH** — both modules raise `ValueError` citing Catalog #303 + the producer-header CONSUMER_HOOK_NUMBERS declaration that justifies the N/A. Error message also cites Catalog #131 (hook #5) or "Anti-arbitrariness primitive" (hook #6) per the specific bug class each N/A enforcement prevents. |
| Test corpus location | **ADOPT_CANONICAL-FROM-HH** — `src/tac/tests/test_cable_d_hooks_5_6_closure.py` mirrors `test_consumers_7_14_solver_wire_in.py` shape (canonical sidecar fixture helper + uuid-fresh sha pattern + parametrized N/A refusal + cross-hook consistency + sister regression + live-repo regression guards). |

### 6 NEW surfaces (hook #5)

1. **`CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE`** (frozenset) — the HARD-EARNED set. 1 consumer.
2. **`CABLE_D_CONSUMERS_7_14_HOOK_5_PAIRS`** (tuple) — 1 (consumer, "continual_learning_posterior") pair.
3. **`Hook5PosteriorAnchor`** (frozen dataclass) — typed observability-only anchor with 10 fields + 3 invariants.
4. **`query_posterior_for_consumer(consumer_id, archive_sha256)`** — canonical accessor; refuses N/A consumers with `ValueError` + Catalog #303 + #131 citation.
5. **`collect_all_hook_5_anchors_for_archive(archive_sha256)`** — returns the 1-tuple of HARD-EARNED anchors.
6. **`consumer_owns_hook_5(consumer_id)`** + **`is_hook_5_anchor_promotable(anchor)`** — canonical disambiguator helpers.

### 9 NEW surfaces (hook #6)

1. **`CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE`** (frozenset) — the HARD-EARNED set. 1 consumer.
2. **`CABLE_D_CONSUMERS_7_14_HOOK_6_PAIRS`** (tuple) — 1 (consumer, "probe_disambiguator") pair.
3. **`DEFAULT_DISAMBIGUATOR_TOP_N`** = 16 — canonical default for top-N pair rank.
4. **`HOOK_6_DEFAULT_VERDICT`** = `"PARTIAL"` — canonical default per Catalog #313 informational-rank semantics.
5. **`Hook6DisambiguatorVerdict`** (frozen dataclass) — typed observability-only verdict with 14 fields + 5 invariants (including Catalog #313 verdict-taxonomy + parallel-array invariant).
6. **`disambiguate_per_pair_stationarity(archive_sha256, *, top_n=16, consumer_id="per_pair_kkt_residuals")`** — canonical disambiguator; reads per-pair KKT residuals from sidecar JSON + emits top-N rank.
7. **`collect_all_hook_6_verdicts_for_archive(archive_sha256, *, top_n=16)`** — returns the 1-tuple of HARD-EARNED verdicts.
8. **`consumer_owns_hook_6(consumer_id)`** + **`is_hook_6_verdict_promotable(verdict)`** — canonical disambiguator helpers.
9. **Internal `_extract_top_n_pair_rank(payload, top_n)`** — payload-key flexibility (4 canonical alias keys); per-pair magnitude extraction with signed-residual `abs()` handling.

## Phase 3: tests

**File**: `src/tac/tests/test_cable_d_hooks_5_6_closure.py` (86 tests, 0.31s wall-clock, ALL PASS).

| Test group | Count | Coverage |
|---|---|---|
| Hook #5 HARD-EARNED matrix pinned | 6 | ACTIVE set is `{per_pair_lora_supervision_signal}`; pairs count == 1; per-consumer parametrized N/A check |
| Hook #5 happy paths (consumer #9 LoRA) | 3 | sidecar absent / sidecar present with payload / rationale cites producer header |
| Hook #5 Catalog #303 N/A refusal | 7 | 5 parametrized N/A consumers raise + direct anchor construction refused + empty sha rejected |
| Hook #5 Catalog #287/#323/#341 invariants | 5 | score_claim forbidden / promotion_eligible forbidden / axis must be predicted / is_promotable always False / provenance pinned |
| Hook #5 collection helper | 4 | returns 1 / all observability-only / one sidecar staged / empty sha rejected |
| Hook #6 HARD-EARNED matrix pinned | 7 | ACTIVE set is `{per_pair_kkt_residuals}`; pairs count == 1; per-consumer parametrized N/A check + unknown consumer |
| Hook #6 happy paths (consumer #12 KKT) | 7 | sidecar absent / sidecar with payload + top-N rank / alternate payload key / signed residuals abs / no payload rank / top_n exceeds n_pairs / default top_n |
| Hook #6 Catalog #303 N/A refusal | 7 | 5 parametrized N/A consumers raise + direct verdict construction refused + empty sha + negative top_n rejected |
| Hook #6 Catalog #287/#323/#341 + Catalog #313 invariants | 9 | score_claim / promotion_eligible / axis_tag / verdict taxonomy / parallel-array / is_promotable / default PARTIAL / provenance pinned / metric_name pinned |
| Hook #6 collection helper | 4 | returns 1 / all observability-only / one sidecar staged / empty sha rejected |
| **Cross-hook FULL 6-hook closure consistency** | 4 | consumer #9 consistent in hooks #3+#5 / consumer #12 consistent in hooks #1+#2+#6 / aggregate ACTIVE count = 11 / every consumer owns ≥1 solver hook |
| Sister regression (HH unchanged) | 6 | HH 9-cell registry unchanged / 6 consumers unchanged / axis tag preserved / canonical sidecar registry unchanged / consumer_owns_hook still works / SolverHookContribution still constructible |
| Live-repo regression guards | 6 | cathedral consumer count >= 34 / hook #5 ACTIVE pinned / hook #6 ACTIVE pinned / no phantom solver hook (cross-checks ACTIVE consumers DO declare the canonical HookNumber) / N/A consumers DO NOT declare CL_POSTERIOR / N/A consumers DO NOT declare PROBE_DISAMBIGUATOR |
| Catalog #313 sister probe-outcomes ledger integration | 2 | verdict in VALID_VERDICTS / metric_name + verdict shape compatible with register_probe_outcome |
| **Total** | **86** | **PASS** |

**Sister regression sweep (234 sister tests across 8 files, 1.76s wall-clock, ALL PASS):**

| File | Tests | Status |
|---|---|---|
| `test_consumers_7_14_solver_wire_in.py` (HH) | 51 | PASS |
| `test_cathedral_consumers_7_14_cascade.py` (FF) | 35 | PASS |
| `test_cable_d_wire_in_master_gradient_consumers.py` (AA Cable D batch) | 33 | PASS |
| `test_check_335_cathedral_consumer_directory_contract.py` | 18 | PASS |
| `test_cathedral_consumer_contract.py` | 30 | PASS |
| `test_cathedral_autopilot_auto_discovery.py` | 13 | PASS |
| `test_check_336_337_cathedral_main_discovery_invoker.py` | 27 | PASS |
| `test_check_341_cathedral_consumer_mps_prescreen_routing.py` | 27 | PASS |

**Aggregate**: 86 NEW + 234 SISTER = **320 tests PASS** across the canonical
solver-wire-in surface for the 6 Cable D consumers.

## Phase 4: live cathedral autopilot smoke

```
Live cathedral consumer count: 34
```

Consumer count **UNCHANGED** (was 34 per HH baseline). The LL work does NOT
add new consumers; it adds NEW canonical solver-side accessors (hook #5
posterior anchor + hook #6 disambiguator verdict) that sister subagents
(cathedral autopilot ranker / continual-learning posterior writer / probe-
outcomes ledger writer) can consume.

## 6-hook wire-in declaration (per Catalog #125) — FULL CLOSURE

This landing closes hooks #5 + #6 for the 6 Cable D consumers, completing
the FULL Catalog #125 6-hook loop:

1. **SENSITIVITY MAP** (hook #1): **ACTIVE** post-HH for consumers #8 + #12 + #13 (3 of 6).
2. **PARETO CONSTRAINT** (hook #2): **ACTIVE** post-HH for consumers #7 + #8 + #12 + #13 (4 of 6).
3. **BIT ALLOCATOR** (hook #3): **ACTIVE** post-HH for consumers #9 + #10 (2 of 6).
4. **CATHEDRAL AUTOPILOT DISPATCH** (hook #4): **ACTIVE** post-FF (commit `d7c28c737`). 1.01× per-sidecar multiplicative reward at ranker surface.
5. **CONTINUAL-LEARNING POSTERIOR** (hook #5): **ACTIVE** post-LL for consumer #9 (`per_pair_lora_supervision_signal`) — the only consumer declaring `HookNumber.CONTINUAL_LEARNING_POSTERIOR` in `CONSUMER_HOOK_NUMBERS`. Per-pair LoRA targets participate in continual-learning at training time. Canonical accessor `query_posterior_for_consumer(consumer_id, archive_sha256)`.
6. **PROBE DISAMBIGUATOR** (hook #6): **ACTIVE** post-LL for consumer #12 (`per_pair_kkt_residuals`) — the only consumer declaring `HookNumber.PROBE_DISAMBIGUATOR` in `CONSUMER_HOOK_NUMBERS`. Per-pair KKT residual `||dD/dθ + λ_R · dR/dθ||_2` is the canonical per-pair stationarity certificate. Canonical accessor `disambiguate_per_pair_stationarity(archive_sha256, top_n=16)`.

**Per-consumer FULL 6-hook closure summary**:

| # | Consumer | #1 | #2 | #3 | #4 | #5 | #6 | Total ACTIVE |
|---|---|---|---|---|---|---|---|---|
| 7 | per_pair_pareto_envelope | N/A | ✓ | N/A | ✓ | N/A | N/A | 2 of 6 |
| 8 | per_pair_lagrangian_lambda_bisection | ✓ | ✓ | N/A | ✓ | N/A | N/A | 3 of 6 |
| 9 | per_pair_lora_supervision_signal | N/A | N/A | ✓ | ✓ | ✓ | N/A | 3 of 6 |
| 10 | per_pair_coding_budget_allocation | N/A | N/A | ✓ | ✓ | N/A | N/A | 2 of 6 |
| 12 | per_pair_kkt_residuals | ✓ | ✓ | N/A | ✓ | N/A | ✓ | 4 of 6 |
| 13 | per_pair_volterra_cross_terms | ✓ | ✓ | N/A | ✓ | N/A | N/A | 3 of 6 |

**Aggregate ACTIVE cells across the 6 consumers × 6 hooks**:
- HH hooks #1+#2+#3: 9 (sensitivity 3 + pareto 4 + bit_allocator 2)
- FF hook #4: 6 (every consumer's CATHEDRAL_AUTOPILOT_DISPATCH)
- LL hook #5: 1 (LoRA)
- LL hook #6: 1 (KKT)
- **Total: 17 HARD-EARNED ACTIVE cells across 36 possible (6 consumers × 6 hooks)**

The remaining 19 cells are HARD-EARNED N/A per producer-header
`CONSUMER_HOOK_NUMBERS` declarations — NOT silent omissions, NOT cargo-culted
"all × all" reflex. Every classification traces to a specific producer-header
declaration.

## Cargo-cult audit per assumption

| Assumption | Classification |
|---|---|
| "All 6 consumers × hook #5 = 6 ACTIVE cells must be wired" | **CARGO-CULTED** — 5 of 6 consumers' producer headers explicitly say "NO-OP by design" for hook #5 (the sidecar IS the canonical posterior write surface; a second write would be a Catalog #131 duplicate-write bug class). Only consumer #9 declares `HookNumber.CONTINUAL_LEARNING_POSTERIOR` in `CONSUMER_HOOK_NUMBERS`. |
| "All 6 consumers × hook #6 = 6 ACTIVE cells must be wired" | **CARGO-CULTED** — 5 of 6 consumers have non-disambiguating signal types (Pareto envelope unambiguous; λ_R unambiguous; LoRA derived; coding-budget primary decision; Volterra one-shot). Only consumer #12 declares `HookNumber.PROBE_DISAMBIGUATOR`. |
| "More hook anchors = better Cathedral observability" | **HARD-EARNED-CONSERVATIVE** — every anchor + verdict is observability-only ([predicted] axis, `promotion_eligible=False`, NO `predicted_delta_adjustment`); they expose canonical structural signals for downstream solver integration WITHOUT mutating scores or promotion. The 2 ACTIVE cells extinct the orphan-signal-at-cathedral-autopilot bug class at the FINAL 2 surfaces beyond HH (hooks #1+#2+#3) + FF (hook #4); the 10 N/A cells DEFEND against phantom-anchor leakage per Catalog #341. |
| "Hook #5 should write directly to `.omx/state/continual_learning_posterior.json`" | **CARGO-CULTED** — would create a duplicate write path competing with the canonical `tac.master_gradient_consumers.consumer_output_path` sidecar JSON writer. Per Catalog #131 fcntl-locked bare-write discipline: the sidecar IS the canonical posterior write; THIS module emits typed READ-ONLY anchors that downstream `tac.continual_learning.posterior_update_locked` consumers (autopilot ranker / Rashomon ensemble) can query. |
| "Hook #6 verdict should default to PROCEED (allowing dispatch)" | **CARGO-CULTED** — per-pair stationarity rank is INFORMATIONAL, not blocking-or-permitting. Per Catalog #313 verdict taxonomy + CLAUDE.md "Forbidden premature KILL without research exhaustion": the default is `PARTIAL` (informational rank); full adjudication (`INDEPENDENT` / `KILL` / `DEFER` / `PROMOTE`) requires sister probe + council deliberation. |
| "Hook #6 should iterate top_n=600 (all pairs) by default" | **HARD-EARNED-CONSERVATIVE** — default `top_n=16` reflects the operator-tunable dispatch attention budget; downstream dispatchers can typically only refocus byte budget on a few-dozen pairs at most. Larger `top_n` is callable via the explicit kwarg. |
| "Hook #5/#6 anchors must be persisted by THIS module" | **CARGO-CULTED** — per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY: anchors are deterministic projections of the canonical sidecar JSON; they are NOT persisted independently by this module. The canonical persistence layer is the canonical sidecar JSON (already written by the producer) + `tac.continual_learning.posterior_update_locked` (hook #5) + `tac.probe_outcomes_ledger.register_probe_outcome` (hook #6) — downstream consumers route there. |

## Observability surface (per Catalog #305)

The 6-facet observability declaration for the LL hook #5 + #6 modules:

- **Inspectable per layer**: `consumer_owns_hook_5(consumer_id)` + `consumer_owns_hook_6(consumer_id)` expose the per-cell HARD-EARNED-vs-N/A classification; `CABLE_D_CONSUMERS_7_14_HOOK_5_ACTIVE` + `CABLE_D_CONSUMERS_7_14_HOOK_6_ACTIVE` are frozen sets queryable directly.
- **Decomposable per signal**: hook #5 anchor exposes `n_pairs` + `n_bytes` + `sidecar_path` + `sidecar_sha256` + `sidecar_schema`; hook #6 verdict exposes `n_pairs` + `top_pair_indices` + `top_pair_residual_magnitudes` (parallel arrays per Catalog #305 decomposability) + `metric_name` + `verdict`.
- **Diff-able across runs**: stateless functions (no module-level mutable state). Same `(consumer_id, archive_sha256, top_n, sidecar set)` always returns the same anchor/verdict payload.
- **Queryable post-hoc**: `query_posterior_for_consumer(consumer_id, archive_sha256)` (single) + `collect_all_hook_5_anchors_for_archive(archive_sha256)` (collection) + `disambiguate_per_pair_stationarity(archive_sha256, top_n=N)` (single) + `collect_all_hook_6_verdicts_for_archive(archive_sha256, top_n=N)` (collection).
- **Cite-able**: every anchor + verdict carries `sidecar_path` (canonical sidecar file path) + `sidecar_sha256` (canonical cite-chain SHA-256 NEW in LL) + `sidecar_schema` + `provenance_canonical_helper` (canonical helper module name).
- **Counterfactual-able**: removing any sidecar file from `.omx/state/master_gradient_consumers/` zeros that consumer × hook anchor/verdict (sidecar_present=False + n_pairs=0 + sidecar_sha256=None + top_pair_indices=()); operator can probe per-cell counterfactual impact by staging/removing individual sidecars.

## 9-dimension success checklist evidence

1. **UNIQUENESS**: NEW canonical hook #5 + hook #6 modules in `tac.cathedral_solver_wire_in` package; no sister modules emit posterior anchors or per-pair stationarity verdicts for Cable D consumers.
2. **BEAUTY + ELEGANCE**: 263 LOC (hook5) + 376 LOC (hook6) + 47 LOC (__init__.py delta) = 686 LOC across 3 files; reviewable in 30 seconds; canonical Catalog #303 cargo-cult audit table embedded in both module docstrings.
3. **DISTINCTNESS**: distinct from HH (HH = hooks #1+#2+#3 ranker-side observability with `predicted_delta_adjustment=0.0`); LL hook #5 = continual-learning posterior anchor; LL hook #6 = probe-disambiguator with per-pair stationarity rank + Catalog #313 verdict taxonomy alignment.
4. **RIGOR**: premise verification (Catalog #229) — read HH landing memo + 6 consumer `__init__.py` files + canonical `tac.continual_learning` + `tac.probe_outcomes_ledger` APIs + `HookNumber` enum + FF cascade implementation before designing. Empirical anchor: 86/86 tests pass + 234/234 sister tests pass + cathedral consumer count unchanged at 34.
5. **OPTIMIZATION PER TECHNIQUE**: per Catalog #290 canonical-vs-unique decision per layer (see Phase 2 table above); 9 explicit per-layer decisions documented.
6. **STACK-OF-STACKS COMPOSABILITY**: composes ON TOP of HH solver-wire-in + FF ranker cascade; sister subagents extending other consumer batches can add hook #5/#6 dispatchers to `tac.cathedral_solver_wire_in` without touching this landing.
7. **DETERMINISTIC REPRODUCIBILITY**: stateless functions; same `(consumer_id, archive_sha256, top_n)` + same sidecar set = same anchor/verdict. Test corpus uses `uuid.uuid4()` fresh-sha pattern for isolation; cleanup via `_staged_canonical_sidecar` context manager.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 0.31s wall-clock for 86 tests; sidecar discovery delegates to HH cached FF helpers (`importlib.util.cache` via `sys.modules`); top-N rank extraction is O(n_pairs · log(n_pairs)) via canonical `sorted()`.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this is an APPARATUS_MAINTENANCE landing per `horizon_class` frontmatter. The score-lowering effect lives at downstream sister subagents that consume the canonical hook #5 posterior anchors + hook #6 stationarity verdicts to refocus dispatch attention on highest-leverage per-pair targets.

## Cross-references

- Slot HH solver wire-in hooks #1+#2+#3 landing: `cable_d_consumers_7_14_solver_wire_in_hooks_123_landed_20260519.md` (commit `cb3a53860`; the upstream landing whose highest-EV op-routable this work addresses)
- Slot FF cathedral autopilot cascade wire-in: `cable_d_consumers_7_14_autopilot_cascade_wire_in_landed_20260519.md` (commit `d7c28c737`; hook #4 cascade)
- Slot AA Cable D wire-in batch: `cable_d_wire_in_batch_landed_20260519.md` (commit `3af31f709`; the Cable D consumers 7-14 build wave)
- Slot Z cathedral autopilot activation: `cathedral_autopilot_activation_945_landed_20260519.md` (commit `da3281c8b`; the unblocking landing per Catalog #335 paradigm shift)
- Cable D batch landing: `feedback_cable_d_master_gradient_extension_batch_landed_20260519T055121Z.md` (commit `6a1e94a63`; the producer-side landing of the 8 Cable D consumers 7-14)
- Sister catalog gates:
  - Catalog #125 (subagent landing 6-hook wire-in non-negotiable; this landing closes the FINAL 2 hooks #5+#6 — FULL CLOSURE complete for the 6 Cable D consumers)
  - Catalog #127 (per-call-site custody routing; anchors + verdicts honor via score_claim_valid=False + promotion_eligible=False invariants)
  - Catalog #131 (fcntl-locked bare-write discipline; rationale for hook #5 NO-OP at 5 N/A consumers — the sidecar IS the canonical write surface)
  - Catalog #138 (strict-load fail-closed sister discipline; `(OSError, json.JSONDecodeError)` graceful degradation in `_sha256_of_sidecar` + `_read_sidecar_payload`)
  - Catalog #185 (META-meta-meta CLAUDE.md catalog drift detection; sister-callable regression test compatible)
  - Catalog #229 (premise verification before edit; PV pre-flight read all required source files)
  - Catalog #230 (sister-subagent ownership map; GG + JJ + KK disjoint at landing per ownership map; no edit overlap)
  - Catalog #245 (Modal call_id ledger canonical 4-layer pattern; LL mirrors structure for anchor/verdict typed surfaces)
  - Catalog #287 (canonical Provenance umbrella; every anchor + verdict tagged `[predicted]` axis)
  - Catalog #290 (canonical-vs-unique decision per layer; 9 explicit decisions in this memo)
  - Catalog #294 (9-dimension success checklist evidence; section above)
  - Catalog #303 (cargo-cult audit per assumption; section above + structural enforcement via `__post_init__`)
  - Catalog #305 (observability surface declaration; section above)
  - Catalog #313 (probe-outcomes ledger canonical 4-layer; hook #6 verdict taxonomy + metric_name align with `register_probe_outcome` schema)
  - Catalog #318 (master-gradient raw-byte-authority guard; no raw byte tensors exposed; only typed structural counts + per-pair indices/magnitudes)
  - Catalog #321/#322/#323 (phantom-score-from-research-sidecar guard family; tests verify score_claim=True sidecars rejected)
  - Catalog #335 (cathedral consumer canonical contract; consumer count unchanged at 34)
  - Catalog #336/#337 (cathedral autopilot main invocation + master-gradient rerank; this landing does NOT touch invocation surface)
  - Catalog #340 (sister-checkpoint guard; PROCEED per ownership map at landing — no overlap with GG/JJ/KK)
  - Catalog #341 (cathedral consumer routing canonical markers; anchors + verdicts honor all 3 canonical markers)

## Sister coordination

| Sister | Scope | Coordination |
|---|---|---|
| GG `a866a992b00edc7f2` B1 E.7 remediate | `.omx/operator_authorize_recipes/` + dispatch state files | DISJOINT — my scope is `src/tac/cathedral_solver_wire_in/hook5_*.py` + `hook6_*.py` + `__init__.py` extension + `src/tac/tests/test_cable_d_hooks_5_6_closure.py` + this memo only |
| JJ `a62d0f688e98af6bb` Catalog #341 Path A+B | `mps_diagnostic_consumer` + `mps_gap_experiment_consumer` (NOT the 6 Cable D consumers) + `tools/master_gradient_xray.py` + NEW test file | DISJOINT (different consumers; different test file) |
| KK `a05eff4548932cdad` C6 IBPS Tier-C re-measurement | `tools/mdl_scorer_conditional_ablation.py` (CALL only) + `substrate_c6_e4_mdl_ibps` recipe + NEW measurement artifact | DISJOINT (different tool; different recipe) |
| HH (already landed; commit `cb3a53860`) | `src/tac/cathedral_solver_wire_in/consumers_7_14_contributions.py` | DISJOINT — LL EXTENDS package with NEW hook5/hook6 modules; HH module unchanged per Catalog #110/#113 APPEND-ONLY |
| FF (already landed; commit `d7c28c737`) | `tools/cathedral_autopilot_autonomous_loop.py` | DISJOINT — LL imports FF helpers via `_build_contribution_payload` (read-only); no mutation |

Catalog #340 sister-checkpoint guard: PROCEED (no file overlap with any active sister; HH + FF + Cable D batch already landed pre-LL).

## Highest-EV op-routable surfaced

**No further hooks remain to wire for the 6 Cable D consumers.** This landing
closes the AA → FF → HH → LL Cable D cascade. The 6 consumers are now FULLY
canonicalized across all 6 Catalog #125 hooks per the HARD-EARNED matrix:

- **17 HARD-EARNED ACTIVE cells** across 6 consumers × 6 hooks
- **19 HARD-EARNED N/A cells** per producer-header `CONSUMER_HOOK_NUMBERS` declarations (refused at construction time per Catalog #303)
- **Zero phantom cells** per CLAUDE.md "Apples-to-apples evidence discipline"

**Possible downstream EV** (NOT this landing's scope; deferred to operator decision):

1. **Sister cathedral consumer batches** (`per_pair_difficulty_atlas` consumer #11 + `per_pair_optimal_treatment_plan_via_lagrangian_dual` consumer #14): apply the same AA → FF → HH → LL cascade pattern to those 2 sister consumers. Per Catalog #303 producer-header inventory required first; the FF landing notes consumer #11 is already wired into ITEM_7 closure (line 1107) and consumer #14 is already wired into Catalog #319 CASCADE 1 PRIMARY (line 1393), so the AA → FF leg is already DONE; only the HH solver-side + LL posterior + disambiguator legs may need extension. Estimated scope: ~400 LOC + ~70 tests = sister-subagent slot.
2. **Downstream consumer of hook #5 posterior anchors**: a canonical `tac.continual_learning.posterior_update_locked_from_hook_5_anchor(anchor: Hook5PosteriorAnchor)` adapter that converts the typed anchor → `ContestResult` (with `evidence_tag="[predicted]"` per Catalog #287 + `score_value=0.0` placeholder + `metadata` carrying the canonical sidecar cite-chain). Enables the autopilot ranker + Rashomon ensemble to query hook #5 anchors as posterior signals without manual reconstruction. Estimated scope: ~100 LOC + ~20 tests = small sister-subagent slot.
3. **Downstream consumer of hook #6 verdicts**: a canonical `tac.probe_outcomes_ledger.register_probe_outcome_from_hook_6_verdict(verdict: Hook6DisambiguatorVerdict)` adapter that converts the typed verdict → canonical `register_probe_outcome(...)` call (with `probe_kind="cable_d_per_pair_kkt_stationarity_rank"` + `verdict=verdict.verdict` + `metric_name=verdict.metric_name` + `metric_value=verdict.top_pair_residual_magnitudes[0] if verdict.top_pair_residual_magnitudes else 0.0` + canonical evidence_path/notes). Enables canonical fcntl-locked persistence of hook #6 verdicts in the canonical Catalog #313 ledger. Estimated scope: ~120 LOC + ~25 tests = small sister-subagent slot.

These downstream EV op-routables are OPERATOR-CHOOSEABLE; this landing
declares the Cable D consumers 7-14 6-hook closure COMPLETE.

## Lane status

- Lane `lane_cable_d_consumers_7_14_hooks_5_6_closure_20260519` registered at L0 (pre-registration via memory entry).
- Gates landed in this commit batch:
  - `impl_complete` ✓ (2 query helpers + 2 collection helpers + 2 canonical-disambiguator helpers + 2 typed dataclasses + 86 dedicated tests + memory entry)
  - `memory_entry` ✓ (this memo)
- Gates pending sister-subagent or operator follow-on:
  - `real_archive_empirical` — N/A (canonical helpers are READ-ONLY observability accessors; no archive bytes generated)
  - `contest_cuda` — N/A (canonical helpers do not produce score claims)
  - `strict_preflight` — no NEW strict preflight gate needed; sister Catalog #335 + #341 + #318 + #319 + #321/#322/#323 + #287 + #323 cover the consumer contract + routing markers + raw-byte guard + Lagrangian PRIMARY + phantom-score family + canonical Provenance umbrella. The Hook5PosteriorAnchor + Hook6DisambiguatorVerdict invariants enforce custody-clean + non-promotable + observability-only at construction time per Catalog #303 #287/#323/#341.
  - `three_clean_review` — adversarial review cycle (next subagent slot)
  - `deploy_runbook` — N/A for editor-only solver-wire-in work

Expected lane level after this commit: **L1** (impl_complete + memory_entry).

— Slot LL Cable D consumers 7-14 → hooks #5 + #6 FULL closure subagent 2026-05-20
(claude_slot_ll_cable_d_hooks_5_6_closure_20260519)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:cable-D-hooks-5-6-closure-landing-memo-trigger-tokens-describe-hook-wire-in-pattern-not-new-equation -->
