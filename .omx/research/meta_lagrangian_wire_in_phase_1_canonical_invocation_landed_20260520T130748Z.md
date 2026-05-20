# META-LAGRANGIAN-WIRE-1 Phase 1: canonical invocation point landed 2026-05-20

**Lane:** `lane_meta_lagrangian_wire_1_phase_1_canonical_invocation_20260520`
**Catalog #:** 355 (STRICT-from-byte-one, live count: 0)
**Slot:** SLOT META-LAGRANGIAN-WIRE-1 (operator-routed 2026-05-20)
**Status:** Phase 1 IMPL_COMPLETE + STRICT_PREFLIGHT + DESIGN_MEMO (this file) + LANDING_MEMO

## Scope: Phase 1 only

This memo + landing land **only Phase 1** of the META-LAGRANGIAN-WIRE-1 work:

- Canonical invocation helper `invoke_meta_lagrangian_on_candidates` in `tools/cathedral_autopilot_autonomous_loop.py`
- Per-iteration call from `main()` in BOTH the `--report-only` and `run_continuous_loop` paths
- Basic bounded-proxy adjuster in `[0.95, 1.05]`
- STRICT preflight gate Catalog #355 enforcing the invocation callsite presence
- 31 dedicated tests (helper unit + STRICT gate + orchestrator wire-in)
- CLAUDE.md catalog row + sister landing memo

Phase 2-N (sister subagents over the 1-3 week window) extend the wire-in along the roadmap below.

## Why this is needed

**WIRE-IN-RIGOR-AUDIT empirical anchor (2026-05-20)** — `.omx/research/wire_in_rigor_audit_meta_class_extinction_synthesis_20260520T124439Z.md`:

> `src/tac/findings_lagrangian/` shipped 9 files / 13.6K LOC: 4-term scalar
> Lagrangian + closed-form Gaussian posterior + Lindley-1956 + Foster-2019
> active-inference action selector + Catalog #277 wavelet-multi-scale
> partition + Rudin interpretability + 3-round T3 grand-council
> ratification — and ZERO production callers imported it.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS":

> The meta-Lagrangian, Pareto, field-equation, and cross-paradigm selector
> stack is a living solver, not a one-off planning report. Any work on
> score lowering, stacking, entropy coding, hidden gems, public PR
> deconstruction, categorical labels, foveation, pose, sensitivity, or
> paradigm wiring must either improve this solver or explicitly record why
> the new signal is not yet actionable.

A canonical helper that no production caller imports is the SAME orphan-signal META-class as Catalog #336 (cathedral consumer discovery) and Catalog #337 (master-gradient rerank) at sister surfaces. The Assumption-Adversary R11 verdict — *"convention-over-implementation (importable canonical helper) is necessary but NOT sufficient"* — applies here verbatim.

Phase 1 lands the **invoker callsite** that turns the meta-Lagrangian from a dormant library into a per-iteration runtime contributor.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable.

| Layer | Decision | Rationale |
|---|---|---|
| Posterior model | **CANONICAL** | Use `tac.findings_lagrangian.posterior_update_from_anchors` — the T3 council ratified closed-form Gaussian per Q1 binding decision; building a Phase-1-specific posterior would duplicate the canonical surface this gate is wiring in. |
| 4-term Lagrangian | **CANONICAL** | Use `compute_findings_lagrangian` — the 4 terms (data_fit + occam_complexity + occam_interpretability + partition_penalty − mu_explore · info_gain) ARE the canonical T3-ratified objective; Phase 1's job is to invoke it, not redesign it. |
| Initial partition | **CANONICAL** | Use `build_initial_partition` — the 4-class cascade taxonomy is the canonical Daubechies-Mallat wavelet-multi-scale prior per Q4 amendment. |
| Residual extraction | **FORK** (Phase 1 bounded proxy) | Phase 1 uses `predicted_score_delta` as a single bounded residual (clipped to `[-1, 1]`); the canonical typed-atom flow per CLAUDE.md "Meta-Lagrangian/Pareto solver" requires per-pair master-gradient residuals + per-class component residuals which are Phase 2/3 deliverables. The fork is DOCUMENTED as Phase 1 scope-limited. |
| Adjustment factor | **FORK** (Phase 1 bounded 5% band) | Phase 1's `_lagrangian_derived_adjustment_factor` returns `[0.95, 1.05]` via a tanh/sigmoid-bounded combination of scalar Lagrangian + posterior sigma per Q7 binding decision's `1/(1+sigma)` form. The canonical dual-variable computation per Lindley 1956 + Foster 2019 is Phase 2 scope. |
| Observability annotation | **CANONICAL** | Per Catalog #287/#323: `score_claim=False` + `promotable=False` + `axis_tag="[predicted]"` on every annotation row. This matches the sister `invoke_cathedral_consumers_on_candidates` per Catalog #336 + the Phase 1 helper's own MAX-OBSERVABILITY contract per Catalog #305. |
| Wire-in callsite | **CANONICAL** | Mirrors `invoke_cathedral_consumers_on_candidates` per Catalog #336 — same callsite location in `main()` (after the autopilot consumer cascade), same observability contract, same STRICT preflight gate pattern. Building a different wire-in pattern would fragment the maintainer + reviewer mental model. |

## 9-dimension success checklist evidence

Per CLAUDE.md "9-dimension success checklist evidence" non-negotiable + Catalog #294.

1. **UNIQUENESS (class-shift not within-class)** — Phase 1 IS a within-class refinement of the existing autopilot ranker (it adds a bounded annotation; no class-shift). Phase 2+ will be class-shift when the bounded proxy is replaced with actual dual-variable Pareto-feasible KKT computation per CLAUDE.md "Meta-Lagrangian/Pareto solver" verbatim *"selectable cross-paradigm"*. Phase 1 scope-classification: WITHIN-CLASS-INFRASTRUCTURE-WIRE-IN (acceptable for a wire-in landing).
2. **BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable)** — ~280 LOC helper + ~190 LOC STRICT gate + ~430 LOC tests = total ~900 LOC. Each function is single-responsibility (residual extractor / adjustment factor / invocation orchestrator). PR101-grade reviewability.
3. **DISTINCTNESS (explicitly different from sisters)** — Catalog #336 invokes consumers via Protocol contract; Catalog #337 invokes master-gradient rerank; Catalog #355 invokes the 4-term Lagrangian. Three orthogonal annotation signals composing in the same `main()` ranking loop.
4. **RIGOR (premise verification + adversarial review + assumption classification + empirical anchor)** — Premise verification: read `src/tac/findings_lagrangian/__init__.py` + `posterior.py` + `lagrangian.py` + `action_selector.py` per Catalog #229 PV BEFORE writing the helper (verified the API surface; corrected the task brief's phantom-API names `evaluate_with_admm`/`choose_solver` → actual `compute_findings_lagrangian`/`recommend_next_action_via_expected_information_gain` per Catalog #287). Empirical anchor: live STRICT gate returns 0 violations + 31/31 tests pass + sister Catalog #336/#337 tests unaffected.
5. **OPTIMIZATION PER TECHNIQUE** — Adjustment factor uses canonical `1/(1+sigma)` downweight per Q7 binding decision (not a re-invented form); bounded tanh sign factor; defense-in-depth bound check at end.
6. **STACK-OF-STACKS-COMPOSABILITY** — The Phase 1 annotation surfaces alongside Catalog #336 cathedral consumer annotations + Catalog #337 master-gradient annotations in the same output payload. All three are observability-only and DO NOT mutate `predicted_score_delta` on the candidate row, so they compose freely without collision.
7. **DETERMINISTIC REPRODUCIBILITY** — Posterior uses `tac.findings_lagrangian.posterior_update_from_anchors` (closed-form conjugate Bayesian update; deterministic). Residual extractor is pure-function on `predicted_score_delta`. Adjustment factor is pure-function on `(scalar, sigma)`. Per CLAUDE.md "Beauty, simplicity, and developer experience".
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Helper does at most `top_n` posterior updates per iteration; each is closed-form O(N_residuals). Phase 1 N_residuals = 1 per candidate (the bounded proxy). Wall-clock contribution to the ranking loop is negligible (~1ms per 10 candidates at the Phase 1 scale).
9. **OPTIMAL MINIMAL CONTEST SCORE** — Phase 1 IS infrastructure-wire-in; it does NOT claim a score contribution (Catalog #287/#323 fail-closed via `score_claim=False`). Phase 2+ will surface dual-variable-derived Pareto-feasible ranking that DOES contribute to ranking accuracy (which downstream becomes a score contribution via better dispatch decisions).

## Cargo-cult audit per assumption

Per CLAUDE.md "Substrate design memo MUST have cargo-cult audit section" non-negotiable + Catalog #303.

| # | Assumption (Phase 1 scope) | HARD-EARNED-vs-CARGO-CULTED | Rationale + unwind path |
|---|---|---|---|
| 1 | "`predicted_score_delta` is a sufficient residual signal for Phase 1." | CARGO-CULTED | Phase 1 uses the predicted delta as a proxy because real residuals require per-pair master-gradient + per-class component anchors that don't exist at the ranking loop's call site. UNWIND PATH (Phase 2): flow per-pair master-gradient residuals + per-class component residuals into the residual extractor. |
| 2 | "Family name is a sufficient `equation_id` for Phase 1." | CARGO-CULTED | The canonical equations registry per Catalog #344 keys equations by `<topic>_v<N>` (e.g. `mps_drift_architecture_class_dependent_v1`). Phase 1 uses candidate.family as a stand-in. UNWIND PATH (Phase 2): map candidate → canonical equation_id via family-token registry lookup; fall back to a `candidate_specific_v1` equation entry. |
| 3 | "Bounded 5% adjustment band is correct for Phase 1." | HARD-EARNED | The bounded band is justified by the cargo-culted-residual-signal: a small bounded adjustment cannot do meaningful harm (or meaningful good) until the residual signal is replaced. UNWIND PATH (Phase 2): widen band proportionally to confidence in dual-variable computation. |
| 4 | "Posterior σ_obs=1.0 is a sensible default." | HARD-EARNED | This is the canonical default in `tac.findings_lagrangian.posterior_update_from_anchors`; Phase 1 uses the canonical default. |
| 5 | "Per-candidate exception trapping is correct (do not crash the autopilot loop)." | HARD-EARNED | Sister `invoke_cathedral_consumers_on_candidates` uses the same per-consumer-per-candidate exception trapping per CLAUDE.md "race-mode rigor inversion" + the autopilot's HALT-and-ASK contract. |
| 6 | "Observability-only contract (no candidate mutation) is correct for Phase 1." | HARD-EARNED | Sister Catalog #336/#337 use the same contract per Catalog #287/#323. Mutating `predicted_score_delta` from a Phase 1 bounded-proxy adjuster would falsely promote the bounded annotation to a ranking signal. |
| 7 | "Initial partition is correct for Phase 1." | CARGO-CULTED | All candidates share the canonical 4-class cascade partition; Phase 2 will route distinct candidates through distinct partition refinements per the MDL-with-wavelet-prior splitting rule per Catalog #277. |

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305. The 6 facets:

1. **Inspectable per layer** — Each Phase 1 annotation row exposes `lagrangian_scalar` + `posterior_sigma` + `adjustment_factor` + full 4-term `decompose` dict. Sister `invoke_cathedral_consumers_on_candidates` cannot inspect the underlying Lagrangian terms; THIS helper does.
2. **Decomposable per signal** — `decompose` carries `data_fit` / `occam_complexity_weighted` / `occam_interpretability_weighted` / `partition_penalty_weighted` / `info_gain_reward_weighted` / `scalar` per the canonical `FindingsLagrangianResult.decompose()` per Catalog #305.
3. **Diff-able across runs** — Posterior update is deterministic (closed-form conjugate Bayesian); given the same inputs, the same Phase 1 annotation rows land. Reviewers can diff two autopilot runs to surface adjustment-factor drift caused by candidate-pool change.
4. **Queryable post-hoc** — The output payload includes `meta_lagrangian_invocations` (top-level key in `report_payload` + `output_payload`); the cathedral verdict ledger persistence at `tac.cathedral.verdict_ledger.append_consumer_invocation_batch` covers the sister consumers but NOT the Phase 1 helper (Phase 2 op-routable: extend the verdict ledger to include the Lagrangian invocation batch).
5. **Cite-able** — Every annotation row is keyed to a `candidate_id` + `family` + `equation_id_used` so cite-chain is traceable.
6. **Counterfactual-able** — `_lagrangian_derived_adjustment_factor(scalar, sigma)` is a pure function; reviewers can probe "what would the adjustment be if scalar=X and sigma=Y?" without re-running the autopilot.

## Phase 2-N roadmap

The 1-3 week T3 Decision 5 estimate covers the full Phase 1-N landing. Phase 1 (THIS subagent) lands the canonical invocation point + per-iteration call + STRICT gate + tests + design memo + landing memo (~3h wall-clock). Phase 2+ is queued for sister subagents:

### Phase 2: actual dual-variable computation per candidate (NOT mock)

**EV estimate:** medium-high (the bounded 5% Phase 1 band becomes a real Pareto-feasible KKT ranking signal once dual variables are computed against the actual rate/seg/pose/archive-size feasible sets).

**Scope:**
- Replace `_lagrangian_derived_adjustment_factor` placeholder with actual dual-variable formulation per Lindley 1956 + Foster 2019.
- Integrate `tac.findings_lagrangian.recommend_next_action_via_expected_information_gain` as the per-candidate action-selector pass.
- Widen the adjustment-factor bound proportionally to confidence in the dual-variable computation.
- Add per-candidate cite-chain to the canonical_equations registry per Catalog #344.

**Dependencies:** Phase 1 wire-in (DONE).

### Phase 3: typed atom flow into the solver

**EV estimate:** medium-high (the canonical typed-atom rows per CLAUDE.md "Meta-Lagrangian/Pareto solver" verbatim *"candidate id, family, pareto scope, charged bytes, predicted SegNet/PoseNet/rate deltas, uncertainty, evidence grade, archive/runtime custody, interaction assumptions, conflicts, Volterra or higher-order terms, KKT/ADMM residuals, expected information gain, blockers, and next proof"* become solver-consumable).

**Scope:**
- Define a canonical `MetaLagrangianTypedAtom` dataclass mirroring the CLAUDE.md schema.
- Map `CandidateRow` → `MetaLagrangianTypedAtom` (the canonical adapter pass).
- Feed typed atoms into a partition refinement step per Daubechies MDL split per Catalog #277.
- Persist typed atoms to a fcntl-locked JSONL store per Catalog #131/#245 (sister `meta_lagrangian_typed_atoms_ledger.jsonl`).

**Dependencies:** Phase 2 (dual variables).

### Phase 4: per-element learned-optimal destination

**EV estimate:** high (the META engineering vision — per-element learned-optimal routing replaces hand-coded heuristics with solver-derived dispatch decisions).

**Scope:**
- Per-byte / per-channel / per-pair routing decisions derived from the dual variables.
- Integration with the bit-allocator surface per Catalog #353 sister.
- Integration with the master-gradient consumer cascade per Catalog #354.

**Dependencies:** Phase 3 (typed atoms).

### Phase 5+: solver-derived dispatch + continual-learning + adversarial review

**Scope:**
- Solver-derived dispatch recommendations (autopilot consumes the Phase 4 per-element routing).
- Continual-learning posterior wire-in: every empirical anchor that lands per `tac.canonical_equations.update_equation_with_empirical_anchor` triggers a Lagrangian recalibration.
- Adversarial review of the Phase 1-4 wire-in: T3 grand council symposium with assumption-adversary verdict per Catalog #292.

## 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125.

| # | Hook | Phase 1 status | Phase 2-N path |
|---|---|---|---|
| 1 | Sensitivity-map contribution | **ACTIVE** | `posterior_sigma_per_term` surfaced in every annotation row per Catalog #305 observability; downstream `tac.sensitivity_map.*` consumers can consume the sigma signal for per-axis weighting. |
| 2 | Pareto constraint | **N/A at Phase 1** | Phase 2 lands the dual-variable surface that maps directly to Pareto KKT residuals; the `MetaLagrangianTypedAtom` Phase 3 schema carries the KKT slack as a first-class field. |
| 3 | Bit-allocator hook | **N/A at Phase 1** | Phase 4 lands the per-element learned-optimal destination which IS the bit-allocator extension at the meta-Lagrangian surface. |
| 4 | Cathedral autopilot dispatch hook | **ACTIVE PRIMARY** | The Phase 1 wire-in callsite IS the structural protection. Catalog #355 STRICT preflight gate enforces presence. |
| 5 | Continual-learning posterior update | **ACTIVE** | The Phase 1 helper uses `posterior_update_from_anchors` per the canonical conjugate Bayesian update. Phase 2 will register results via `tac.canonical_equations.update_equation_with_empirical_anchor`. |
| 6 | Probe-disambiguator | **N/A at Phase 1** | Phase 2 lands the info-gain action-selector branch per Lindley 1956 + Foster 2019; that branch IS the probe-disambiguator (the canonical helper recommends a next action with maximum expected KL info gain per dollar). |

## Mission alignment

Per CLAUDE.md "Mission alignment" Consequence 5 — `council_predicted_mission_contribution`: **`apparatus_maintenance`** (Phase 1 IS infrastructure wire-in; the mission contribution is enabling Phase 2+ to land the actual solver-derived ranking + dispatch signal).

## Files

| Path | LOC | Purpose |
|---|---|---|
| `tools/cathedral_autopilot_autonomous_loop.py` (edits) | +~280 + 3 wire-in sites | Phase 1 helper + per-iteration calls + output payload extension |
| `src/tac/preflight.py` (edits) | +~190 | Catalog #355 STRICT gate + preflight_all wire-in |
| `src/tac/tests/test_meta_lagrangian_cathedral_wire_in.py` (new) | 432 | 31 tests covering helper + STRICT gate |
| `CLAUDE.md` (edits) | +1 catalog row | Catalog #355 entry per Catalog #176 META-meta |
| `.omx/research/meta_lagrangian_wire_in_phase_1_canonical_invocation_landed_20260520T130748Z.md` (this file) | ~250 | Phase 1 design memo + Phase 2-N roadmap |
| `~/.claude/projects/.../feedback_slot_meta_lagrangian_wire_1_phase_1_canonical_invocation_landed_20260520.md` (new) | TBD | Phase 1 landing memo |

## Cross-references

- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" (the canonical mandate this Phase 1 lands the wire-in for)
- CLAUDE.md "Subagent coherence-by-default" (the 6-hook wire-in non-negotiable + the orchestration-via-rules principle)
- CLAUDE.md "Forbidden score claims" (the observability-only contract this Phase 1 helper honors)
- Catalog #336 + #337 (sister invoker callsite gates; the META-pattern this gate mirrors)
- Catalog #335 (canonical cathedral consumer contract; future Phase 5+ could register the Phase 1 helper as a canonical consumer for auto-discovery)
- Catalog #287 + #323 (canonical Provenance umbrella; every annotation row carries the canonical markers)
- Catalog #305 (observability surface; the Phase 1 helper surfaces `lagrangian_scalar` + `decompose` for operator audit)
- Catalog #344 (canonical equations registry; Phase 2 will register results via the canonical posterior anchor surface)
- T3 grand-strategy review `.omx/research/council_t3_grand_strategy_review_20260520T120000Z.md` Decision 5
- WIRE-IN-RIGOR-AUDIT `.omx/research/wire_in_rigor_audit_meta_class_extinction_synthesis_20260520T124439Z.md`
- Operator routing `.omx/research/operator_routable_decisions_20260520T120607Z.md`

## Acceptance gate

This Phase 1 landing is accepted because:

1. Helper smoke-test PASSES end-to-end (the synthetic CandidateRow returns canonical-keys + bounded adjustment + observability-only contract).
2. STRICT preflight gate Catalog #355 PASSES against the live repo (live count: 0).
3. 31/31 tests PASS.
4. Sister Catalog #336/#337 tests UNAFFECTED (27/27 still pass).
5. Canonical helper `tools/cathedral_autopilot_autonomous_loop.py --help` parses cleanly (no import regression).
6. CLAUDE.md catalog row #355 lands + Catalog #176 sister-gate confirms the strict callsite has the CLAUDE.md row.
7. The 7 pre-existing Catalog #185 drift entries are on OTHER catalog rows (#170/#171/#173/#181/#182/#300/#346) caused by sister-landing untracked recipes (substrate_e_nerv / substrate_ego_nerv / substrate_nervdc); NOT caused by this Phase 1 landing per Catalog #230 ownership map.
