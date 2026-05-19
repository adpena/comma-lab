# Cargo-cult unwind audit: `unified_action_consumer` (Wave 2C)

- **Date**: 2026-05-19
- **Subagent**: WAVE-2C-CARGO-CULT-UNWIND-AUDIT (`wave_2c_cargo_cult_audit_20260519`)
- **Target**: `src/tac/cathedral_consumers/unified_action_consumer/__init__.py` (61 LOC; commit `179dc2501`)
- **Canonical reference**: `src/tac/cathedral_consumers/_example_consumer/__init__.py` (64 LOC)
- **Upstream namespace audited**: `tac.unified_action` (Action / DualVariables / OptimizerAnalyticalBoundaries / SolverChoice / SurfaceKind / TrackKind / `choose_solver` / `evaluate_with_admm` / `evaluate_with_magic_codec`)

## Source-text differential vs `_example_consumer`

`unified_action_consumer` differs from the template by ~10 lines: docstring + `CONSUMER_NAME` + rationale enumeration + `CONSUMER_HOOK_NUMBERS` declares `(CATHEDRAL_AUTOPILOT_DISPATCH, PARETO_CONSTRAINT)` (the only Wave 2C consumer that touches Pareto hook #2 — sensible given `tac.unified_action` IS the canonical Pareto/Lagrangian surface). Body of `update_from_anchor` and `consume_candidate` is byte-identical template.

## Canonical-vs-unique decision per layer

Per Catalog #290 falling-rule list:

| Layer | Canonical decision | Verdict | Evidence |
|---|---|---|---|
| L1: `CONSUMER_NAME` literal | adopt | **ADOPT_CANONICAL_BECAUSE_SERVES** | Auto-discovery registration. |
| L2: `CONSUMER_HOOK_NUMBERS = (CATHEDRAL_AUTOPILOT_DISPATCH, PARETO_CONSTRAINT)` | adopt dual-hook | **HARD-EARNED-FIRST-PRINCIPLES** | `tac.unified_action` IS the canonical Pareto/Lagrangian solver surface per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable. Hook #2 declaration is correct. Hook #5 correctly omitted (unified action evaluation is deterministic given inputs). |
| L3: `update_from_anchor(anchor)` NO-OP | adopt template | **HARD-EARNED-FIRST-PRINCIPLES** | Per docstring: *"Unified action evaluation is deterministic given input dual variables + track-kind"*. No posterior to update; matches hook-declaration honesty. |
| L4: `consume_candidate(candidate)` discards via `_ = candidate` | adopt template | **CARGO-CULTED-PATH-OF-LEAST-RESISTANCE** + **STRUCTURALLY-MISALIGNED-WITH-HOOK-#2-DECLARATION** | This is the SHARPEST cargo-cult in the Wave 2C set. Declaring hook #2 (Pareto constraint) while discarding the candidate is a structural lie. The Protocol contract says hook #2 fires when the consumer contributes a Pareto-constraint signal; this consumer contributes nothing. See cargo-cult #1 below. |
| L5: `axis_tag = "[predicted]"` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED. Unified action evaluations are predictions until paired-axis empirical. |
| L6: `promotable = False` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED. |
| L7: `confidence = 0.0` | adopt canonical | **CARGO-CULTED-INHERITED-DEFAULT** | Template literal. Per cargo-cult #1, evaluating with `choose_solver` would yield meaningful confidence. |
| L8: rationale enumerating action surfaces | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | Accurate; operator-readable. |

## Cargo-cult audit per assumption

Per Catalog #303. Three inherited assumptions:

### Assumption #1: "The unified action consumer should be observability-only — actual evaluation is the caller's responsibility"

- **Classification**: CARGO-CULTED-PATH-OF-LEAST-RESISTANCE (with EMPIRICAL-EVIDENCE-CONFIRMING-CARGO-CULT)
- **Evidence**: Per the consumer's OWN docstring: *"per-candidate unified-action evaluation requires explicit Action construction (the canonical caller per the meta-Lagrangian wire-in plan)"*. The consumer EXPLICITLY DEFERS to a "canonical caller" that may not exist yet. This is the canonical orphan-signal pattern at the META layer per the operator's NON-NEGOTIABLE 2026-05-19 directive *"What if we change the paradigm by making cathedral autopilot ingest by default"*. The Catalog #335 paradigm shift was meant to STRUCTURALLY EXTINCT this orphan-signal class; declaring hook #2 + deferring evaluation to a non-existent caller re-instantiates the orphan pattern at a sister surface.
- **Why apparatus suppressed**: implementing actual `choose_solver` + `evaluate_with_admm` requires (a) the candidate to carry sufficient payload (substrate_id + dual variables + track_kind) to construct an `Action`, (b) bounded evaluation cost (the solver shouldn't run a 100-iteration ADMM per candidate per autopilot tick), (c) integration with `tac.master_gradient_consumers` cascade so the Lagrangian dual is the canonical answer per the codex 2026-05-17 architectural correction. This is substrate-engineering work per HNeRV parity L7 that the template-clone discharge skipped.
- **Unwind hypothesis**: implement bounded unified-action consumption: (a) read `candidate.get("substrate_id")` + `candidate.get("predicted_score_delta")` + `candidate.get("estimated_dispatch_cost_usd")`; (b) construct a bounded `DualVariables` from candidate metadata (default to canonical values when missing); (c) call `tac.unified_action.choose_solver(...)` with a `max_iterations=10` ADMM bound; (d) surface the chosen solver + estimated Pareto position in `rationale` + non-zero `confidence` proportional to solver convergence. Per the codex correction: this consumer becomes the CASCADE 1 PRIMARY for the v2 reweight cascade.
- **Unwind cost**: ~$0 in compute (CPU-only bounded ADMM); ~50-80 LOC + 5-8 tests. **HIGH RISK**: must be paired with sister `cathedral_autopilot_autonomous_loop.adjust_predicted_delta_for_*` cascade so the unified-action result actually mutates ranking (per Catalog #287/#323 observability-only is the SAFE default; promoting to ranking mutation requires per-Catalog #319 sister gate review).
- **Predicted signal contribution**: per the v2 cascade in Catalog #319 / #322 / #823 the master-gradient-derived optimal plan is the canonical answer. A unified-action consumer that surfaces the planner's verdict could shave **-0.005 to -0.010 ΔS direct** on candidates where the planner predicts a Pareto improvement currently invisible to the rate-only ranker.
- **Reactivation criterion**: 3 alternative reducers per Catalog #308 = (a) `evaluate_with_admm` with axis=`contest_cpu` / (b) `evaluate_with_magic_codec` for packet-grammar-aware candidates / (c) `choose_solver` ONLY (no evaluation; surface solver-choice metadata).

### Assumption #2: "Declaring hook #2 (Pareto constraint) while no-op-ing the implementation is acceptable"

- **Classification**: CARGO-CULTED-INHERITED-DEFAULT (sharper variant of atom_consumer #2)
- **Evidence**: The Pareto-constraint hook is documented in Catalog #125 as: *"Pareto constraint added to `tac.pareto_*` (or explicitly reasoned why non-binding)"*. unified_action_consumer claims the hook but adds no Pareto constraint and doesn't explicitly reason why. The hook declaration is a documentation lie until cargo-cult #1 lands.
- **Why apparatus suppressed**: same template-clone defense. Catalog #335 STRICT validation is structural-only.
- **Unwind hypothesis (option A)**: drop `PARETO_CONSTRAINT` from `CONSUMER_HOOK_NUMBERS` until cargo-cult #1 lands. HONEST.
- **Unwind hypothesis (option B)**: pair with cargo-cult #1 — once `choose_solver` + bounded ADMM evaluation lands, hook #2 declaration is justified.
- **Unwind cost**: option A ~$0, 2 LOC. Option B = cargo-cult #1 cost.
- **Predicted signal contribution**: option A removes a lie; option B unlocks the Pareto signal.
- **Reactivation criterion**: option A trivially reversible.

### Assumption #3: "The 'canonical caller per the meta-Lagrangian wire-in plan' will be implemented elsewhere"

- **Classification**: CARGO-CULTED-INHERITED-DEFAULT (this is the META-orphan-signal pattern again)
- **Evidence**: The consumer's docstring defers to a "canonical caller per the meta-Lagrangian wire-in plan". Grepping the codebase: `grep -rn "meta_lagrangian_wire_in_plan\|meta_lagrangian.*caller" tools/ src/tac/` returns no concrete caller wire-in. The "canonical caller" is aspirational, not implemented.
- **Why apparatus suppressed**: deferring to a future canonical caller is the EXACT pattern Catalog #335 was meant to extinct. The paradigm shift was meant to extinct deferral — every namespace gets an auto-discovered consumer; the consumer IS the canonical caller per the paradigm shift.
- **Unwind hypothesis**: same as cargo-cult #1. The consumer SHOULD be the canonical caller.
- **Unwind cost**: tied to cargo-cult #1.

## Observability surface

Per Catalog #305:

1. **Inspectable per layer**: rationale + dual-hook declaration. ✓
2. **Decomposable per signal**: today NO (monolithic rationale). Post-unwind: per-solver-choice + per-Pareto-position breakdown.
3. **Diff-able across runs**: today NO (identical template). Post-unwind: solver choice + Pareto position diff with candidate metadata.
4. **Queryable post-hoc**: partial — `consumer_invocations` persisted.
5. **Cite-able**: today NO. Post-unwind: cite `Action` / `SolverChoice` / `TrackKind` enum values.
6. **Counterfactual-able**: today NO. Post-unwind: changing candidate's `predicted_score_delta` would change ADMM dual variables and hence the solver verdict.

**Conclusion**: 1-of-6 today. Cargo-cult #1 unwind would lift to 5-of-6.

## Unwind priority queue

| Rank | Unwind | Cargo-cult # | Cost | Predicted ΔS | EV ratio |
|---|---|---|---|---|---|
| 1 | Bounded unified-action evaluation per candidate (50-80 LOC) + ranking-cascade integration | #1 + #2 + #3 (composite) | ~$0 CPU, 50-80 LOC, 5-8 tests, **HIGH RISK** requires sister cascade review per Catalog #319 | -0.005 to -0.010 ΔS direct | HIGH |
| 2 | (Option A): drop hook #2 + remove "canonical caller" deferral language until cargo-cult #1 lands | #2 + #3 | ~$0, 3 LOC | -0.0005 ΔS/month indirect (operator-trust) | LOW |

**Top-1 unwind**: cargo-cult #1+#2+#3 composite. HIGHEST DIRECT-ΔS in the Wave 2C set but HIGHEST IMPLEMENTATION RISK. Requires:
1. Premise verification per Catalog #229 against `tac.unified_action` API surface.
2. Adversarial council review per Catalog #292 / #300 (T2 sextet) — this changes the autopilot ranker semantics.
3. Sister wire-in to the v2 reweight cascade per Catalog #319 / #322.
4. Empirical verification on at least one anchor that the bounded ADMM verdict matches the master-gradient-derived planner.

## Cross-references

- Sister of CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" — `tac.unified_action` IS this solver's canonical Action surface.
- Sister of Catalog #319 (Wyner-Ziv deliverability autopilot reweight v2 cascade) + Catalog #322 (phantom provenance composition_alpha) + Catalog #823 (SUPER_ADDITIVE) — together they define the canonical 3-cascade dispatch reweight; unified_action_consumer is the CASCADE 1 (PRIMARY) candidate per the codex 2026-05-17 architectural correction.
- Anchor memo: `feedback_unified_lagrangian_action_principle_GR_style_20260509.md` (the original 6-hook wire-in non-negotiable that this consumer's hook #2 declaration cites).
- Anchor memo: `feedback_q2_q3_batched_catalog_319_gate_plus_autopilot_reweight_v2_landed_20260517.md` (the canonical v2 cascade pattern).

## Verdict

`unified_action_consumer` has the SHARPEST cargo-cult in the Wave 2C set: hook #2 declaration + "canonical caller deferral" docstring + zero implementation. This consumer is the highest-EV unwind candidate by predicted-ΔS (potentially direct ranking-mutation contribution) but also the highest-risk (requires sister cascade + council review). Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308: DEFER + REQUEST-REINVESTIGATION-OF-ALTERNATIVES. Per CLAUDE.md "Design decisions — non-negotiable": cargo-cult #1 unwind is a council-grade tradeoff requiring T2 sextet sign-off before landing.

**OPERATOR-ROUTABLE PRIORITY 1**: this consumer's cargo-cult #1 is the canonical example of the META-orphan-signal pattern Catalog #335 was meant to extinct. Worth elevating to its own grand council symposium per Catalog #325 per-substrate optimal form discipline.
