# Cathedral Autopilot Smarter Design Blueprint — 2026-05-20T13:03:25Z

> **Deliverable A (master memo) of SLOT CATHEDRAL-SMARTER-DESIGN-MEMO**
> **Lane**: `lane_cathedral_autopilot_smarter_design_blueprint_20260520`
> **Cite-chain**: `wire_in_rigor_audit_meta_class_extinction_synthesis_20260520T124439Z` + `council_t3_grand_strategy_review_20260520T120000Z` + `strategy_staircase_synthesis_20260520T120000Z` + `comprehensive_plan_short_mid_long_term_20260520T120000Z` + `strategy_dependency_wirein_graph_20260520T120000Z` + `gap_bug_warning_inventory_20260520T120000Z`
> **Cross-ref**: `docs/meta_engineering_vision.md` (commit `7b6be8a44`) + `docs/ai_assisted_inverse_steganalysis_persona_council.md` + `feedback_mps_engineering_corrections_landed_20260520.md`
> **Sister deliverables**: `cathedral_autopilot_smarter_dependency_graph_20260520T130325Z.md` + `cathedral_autopilot_smarter_cost_envelope_20260520T130325Z.md` + landing memo
> **Mission-alignment** (per CLAUDE.md "Mission alignment" Consequence 5): `apparatus_maintenance` (this is a design blueprint; no direct score contribution; enables future frontier_breaking work)
> **Discipline**: Catalog #229 PV / #287 placeholder-rejection / #292 per-deliberation assumption surfacing / #294 9-dim checklist / #296 predicted-band Dykstra / #303 cargo-cult audit / #305 observability surface / #309 horizon-class declaration / #323 canonical Provenance / #343 frontier-pointer literal discipline

---

## Executive summary (operator-facing; ~500 words; HONEST)

**The operator question**: "How do we make the cathedral autopilot smarter — more feedback-loop / more mathematically grounded / more problem-space-grounded / more domain-grounded — beyond what META-LAGRANGIAN-WIRE-1 (in flight; addresses Dimension 1 Phase 1) covers?"

**The HONEST answer**:

The cathedral autopilot today is **~35% empirically grounded, ~65% observability-by-design** (WIRE-IN-RIGOR landing, 2026-05-20). The 35% is REAL: 10 in-main-line `adjust_predicted_delta_for_*` adjusters + 5 canonical-state ledgers (continual-learning posterior 118 anchors / probe-outcomes 59 outcomes / modal call-id 393 rows / canonical Provenance / canonical frontier pointer). The 65% is 44 cathedral consumers that all return `predicted_delta_adjustment=0.0` per Catalog #341 design intent. Per the WIRE-IN-RIGOR Assumption-Adversary verdict: "auto-discovery + observability annotations sufficient" is CARGO-CULTED — discovery + invocation surface extincted; score-mutation surface still 35% wired.

"Making it smarter" decomposes across **6 dimensions** (5 operator-named + 1 surfaced by WIRE-IN-RIGOR):

1. **Mathematical grounding** — replace hand-derived adjuster formulas with canonical-solver dual variables (META-LAGRANGIAN-WIRE-1 owns Phase 1; this memo blueprints Phases 2-N)
2. **Feedback-loop frequency** — tighten per-iteration posterior + canonical-equation refresh
3. **Problem-space grounding** — per-axis decomposition (d_seg / d_pose / archive_bytes) at consumer level
4. **Domain grounding** — comma2k19 + ego-motion + per-frame difficulty priors
5. **Continual-learning closure** — every empirical anchor → Bayesian update → immediate ranker re-rank same iteration
6. **Consumer dual-tier architecture** — resolve Catalog #341's observability-only design (recommend Tier B contributing consumers OR accept design intent)

**Top-3 mission-frontier-breaking recommendations** (highest EV):

1. **Phase 2-N of Dimension 1** (META-LAGRANGIAN-WIRE-1 succession). Replace each of 10 in-main-line `adjust_predicted_delta_for_*` adjusters with solver-derived dual variables from `tac.findings_lagrangian` (TRACK A) + `tac.findings_lagrangian_pp` (TRACK B). 4-8 sessions; $0 GPU.
2. **Per-axis consumer payload extension** (Dimension 3). Cathedral consumers emit `(d_seg, d_pose, rate)` triple instead of scalar ΔS; solver composes per-axis predictions via canonical equation `S = 100·d_seg + √(10·d_pose) + 25·rate/N`. 3-5 sessions; $0 GPU.
3. **Per-iteration continual-learning closure** (Dimension 5). Wire `posterior_update_locked` + `update_equation_with_empirical_anchor` invocation into cathedral autopilot main() loop so each empirical anchor refreshes ranker WITHIN same iteration. 2-3 sessions; $0 GPU.

**HONEST verdict on timeline**:

- "Actually smart" empirically demonstrated in 1-3 months **CONDITIONAL on**: (a) META-LAGRANGIAN-WIRE-1 Phase 2 lands + consumes ≥3 existing adjusters; (b) ≥1 paired-CUDA-CPU master-gradient anchor lands on contest-compliant hardware; (c) ≥1 Lagrangian OptimalPerPairTreatmentPlan sidecar lands so Cascade 1 fires empirically.
- Without those 3 prerequisites, the apparatus is structurally smarter (auto-discovery + canonical contracts + Bayesian posteriors) but the SCORE PATH remains 35% wired. Per Carmack's dissent in T3 symposium: "the apparatus has been producing infrastructure that protects the frontier; it has not been producing frontier."
- Per CLAUDE.md "Race-mode rigor inversion" Rule 2: leaderboard has NOT moved since 2026-05-15 (CPU) / 2026-05-16 (CUDA); rigor STAYS at pre-race level; we have time to mature the apparatus. The PR #110 lifecycle is the operator-attention gating event.

**Total cost envelope** for the full 6-dimension blueprint (excluding paid GPU dispatch from Step 4 of staircase): **$0 GPU + ~30-50 subagent sessions over 8-16 weeks**. The dimensions compose — Phase 2 of Dimension 1 + per-axis Dimension 3 + per-iteration Dimension 5 together close the score-mutation gap; Dimensions 2/4/6 are second-order accelerators.

---

## 1. Empirical baseline (HONEST snapshot)

### 1.1 What's already FULLY_WIRED (the 35%)

Per WIRE-IN-RIGOR per-component dossier (`.omx/research/wire_in_rigor_audit_per_component_dossier_20260520T124439Z.md`):

| Surface | LOC | Role | Empirical anchors |
|---|---|---|---|
| 10 in-main-line `adjust_predicted_delta_for_*` adjusters | `tools/cathedral_autopilot_autonomous_loop.py:1026-1148` | Score-mutating ranker cascade | Empirically grounded via Z1 revision / MDL density / class-shift / composition alpha v2 / Venn classification v2 / per-pair sister 817 sidecars / per-pair difficulty atlas / Cable D / predicted dispatch risk / realistic stacking correction |
| `tac.continual_learning.posterior_update_locked` | `src/tac/continual_learning.py` | Bayesian posterior anchor store | 118 anchors / 31 refused / last updated 2026-05-19T15:49Z |
| `tac.probe_outcomes_ledger` (Catalog #313) | `src/tac/probe_outcomes_ledger.py` | Probe-disambiguator verdict store | 59 outcomes (30 DEFER / 20 PROCEED / 6 PARTIAL / 3 INDEPENDENT) |
| `tac.deploy.modal.call_id_ledger` (Catalog #245) | `src/tac/deploy/modal/call_id_ledger.py` | Modal dispatch lifecycle ledger | 393 rows (148 dispatched / 163 failed / 81 harvested / 1 stale) |
| `tac.provenance` (Catalog #323) | `src/tac/provenance/` | Canonical Provenance umbrella | 1923 CLEAN / 2 WARN / 202 VIOLATION (down from 543 baseline) |
| `tac.canonical_frontier_pointer` (Catalog #343) | `src/tac/canonical_frontier_pointer.py` | Frontier-anchor canonical pointer | Last refreshed 2026-05-20T11:38Z; CPU 0.192051 / CUDA 0.205330 |

### 1.2 What's PARTIALLY_WIRED or FACADE (the 65%)

| Surface | Verdict | Why |
|---|---|---|
| 44 production cathedral consumers | FACADE (in aggregate; observability-only by design per Catalog #341) | 44/44 return `predicted_delta_adjustment=0.0` per canonical contract |
| `tac.unified_action` (meta-Lagrangian solver) | SCAFFOLD_ONLY | ZERO production callsites of `evaluate_with_admm` / `choose_solver` / `Action.S_total` |
| `tac.findings_lagrangian` + `tac.findings_lagrangian_pp` | SCAFFOLD_ONLY at consumer surface | Phase 1-a tests landed 2026-05-19; META-LAGRANGIAN-WIRE-1 sister is now wiring Phase 1 invocation |
| `tac.boosting` namespace | ORPHAN_SIGNAL | Rich API; only 1 production consumer outside namespace |
| `tac.compress_time_optimization` namespace | SCAFFOLD_ONLY | Consumed only by own sub-modules + tests |
| Master-gradient anchors | PARTIALLY_WIRED | 10/10 anchors `[macOS-CPU advisory]`; ZERO on contest-compliant hardware |
| Substrate composition matrix Cascade 1 | PARTIALLY_WIRED | 0 Lagrangian `OptimalPerPairTreatmentPlan` sidecars exist; Cascade falls through to 1.0× passthrough |
| WZ deliverability proof | PARTIALLY_WIRED | 1 sidecar exists (`probe_f174192aeadf_*.json`) verdict NOT_DELIVERABLE |
| `tac.canonical_equations` registry | PARTIALLY_WIRED | 11 equations registered; consumers cite them via STUB cathedral consumers (0.0 adjustment) |

### 1.3 Two non-existent surfaces flagged in graph but missing in code

Per cross-check with `strategy_dependency_wirein_graph_20260520T120000Z.md`:

- **`findings_lagrangian_consumer`** cathedral consumer package referenced in Graph 2 but does NOT exist at `src/tac/cathedral_consumers/findings_lagrangian_consumer/`. Gap surfaced; sister subagent META-LAGRANGIAN-WIRE-1 likely owns this.
- **`tac.bit_allocator`** top-level canonical helper referenced in Graph 2 Hook 3 as "proposed" — does NOT exist; per Decision 10 (canonical-helper-sister-extension over new-tool) recommended path is `tac.master_gradient_consumers.bit_allocator_from_per_byte_sensitivity` sister method.

### 1.4 Frontier-state anchor (Catalog #343 pointer discipline)

Per `tac.canonical_frontier_pointer.load_canonical_frontier_pointer_strict()` at 2026-05-20T11:38:53Z:

| Axis | Score | Archive sha (first 12) | Lane |
|---|---|---|---|
| `[contest-CPU]` | (canonical pointer) | `6bae0201fb08` | `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` |
| `[contest-CUDA T4]` | (canonical pointer) | `9cb989cef519` | `pr106_format0d_latent_score_table_20260516_contest_cuda` |

Frontier CPU hasn't moved since 2026-05-15; CUDA hasn't moved since 2026-05-16. The plateau IS the empirical anchor for "the apparatus has been producing infrastructure that protects the frontier; it has not been producing frontier" (Carmack T3 dissent).

---

## 2. Per-dimension blueprint

### Dimension 1 — Mathematical grounding (META-LAGRANGIAN-WIRE-1 owns Phase 1; this memo blueprints Phases 2-N)

#### Current state

10 in-main-line `adjust_predicted_delta_for_*` adjusters at `tools/cathedral_autopilot_autonomous_loop.py:1026-1148` mutate `predicted_score_delta` via hand-derived formulas:

- `adjust_predicted_delta_for_mdl_density` (Z1 within-class trap penalty)
- `adjust_predicted_delta_for_mdl_tier_c_density` (post-training Tier C re-measurement)
- `adjust_predicted_delta_for_class_shift` (Z1 class-shift literature reward)
- `adjust_predicted_delta_for_composition_alpha_v2` (3-cascade: Lagrangian-plan / deliverability / passthrough)
- `adjust_predicted_delta_for_predicted_dispatch_risk` (cost-band risk floor)
- `adjust_predicted_delta_for_venn_classification_v2` (3-cascade: Lagrangian-plan / deliverability / passthrough)
- `adjust_predicted_delta_for_per_pair_sister_817_sidecars`
- `adjust_predicted_delta_for_per_pair_difficulty_atlas`
- `adjust_predicted_delta_for_cable_d_consumers_7_14_sidecars`
- `adjust_predicted_delta_for_realistic_stacking_correction`

Each adjuster encodes hand-derived weights / thresholds / penalty multipliers. Per WIRE-IN-RIGOR finding: "these DO mutate `predicted_score_delta` and ARE the ~35%."

**Empirically grounded but cargo-culted at the META level**: hand-derived weights are HARD-EARNED at the individual-finding level (every weight has a citing memo) but CARGO-CULTED at the system level (no canonical solver derives them jointly; each adjuster operates independently).

#### Smarter target state

Replace each hand-derived adjuster with solver-derived dual variable from `tac.findings_lagrangian.compute_findings_lagrangian(...)`. The unified `S_total(theta, archive_bytes, hardware) = sum_i alpha_i * L_i + sum_j beta_j * C_j + gamma * R(theta)` action treats each existing adjuster as a per-track Lagrangian term L_i; dual variables α_i are derived via KKT / ADMM, not hand-set.

Per CLAUDE.md anti-fragmentation primitive: "When the unified action lands, every track plugs in by adding a term to S_total — no new orchestration layer. The coherence is structural."

#### Mechanism (Phases 1-N)

**Phase 1 (META-LAGRANGIAN-WIRE-1 owns; in flight)**: Wire `evaluate_with_admm` / `choose_solver` invocation point into cathedral autopilot main() so the solver IS called per candidate. Output: a canonical `LagrangianDualResult` object alongside the existing `predicted_score_delta`. NO existing adjuster removed; coexists.

**Phase 2 (blueprint here)**: Per-adjuster ablation. For each of the 10 in-main-line adjusters:
1. Add a `tac.findings_lagrangian` Lagrangian term that REPLICATES the adjuster's behavior (closed-form Gaussian posterior per TRACK A; hierarchical NumPyro per TRACK B).
2. Run paired comparison: original adjuster delta vs solver-derived delta on the last 30 days of continual-learning posterior anchors.
3. If solver-derived matches within ±2σ of paired residual: REPLACE the in-main-line adjuster with `solver.term[adjuster_name]` lookup.
4. If solver-derived diverges materially: SURFACE the divergence as canonical equation (`tac.canonical_equations.register_canonical_equation(...)`) + DEFER until ≥3 empirical anchors validate one side.

**Phase 3 (blueprint here)**: Cross-track composition. Use `tac.findings_lagrangian.unified.UnifiedPrediction.ensemble_prediction_from_tracks(...)` to compose TRACK A (Gaussian) + TRACK B (NumPyro hierarchical) into single posterior with per-track uncertainty. Cathedral autopilot consumes ensembled prediction.

**Phase 4 (blueprint here)**: Dykstra-feasibility integration. Add Pareto constraints C_j to `S_total` per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #296 sister discipline. Constraints: (a) rate ≤ Dykstra ceiling 450,545 bytes for sub-0.30 feasibility per 2026-04-29 anchor; (b) seg ≤ SegNet architectural ceiling ε ≈ 6.7e-4; (c) pose ≤ PoseNet architectural ceiling; (d) archive ≤ contest packet constraint.

**Phase 5 (blueprint here)**: Sensitivity-map regularization R(θ). Integrate `tac.sensitivity_map.axis_weights.default_axis_weights` + `wyner_ziv_reweight.axis_level_reweight` into the regularization term. Master-gradient anchors (when authoritative on contest hardware) drive the regularization weights.

**Phase 6+ (blueprint here)**: Deprecate in-main-line adjusters once solver-derived terms subsume them. Per Decision 6 (consolidation over addition): the unified solver SUBSUMES the per-track adjusters, not parallels them.

#### Cost / wall-clock

- Phase 1: 1-2 sessions (META-LAGRANGIAN-WIRE-1 in flight)
- Phase 2: 5-10 sessions (1 per adjuster ablation; paired-comparison validation)
- Phase 3: 1-2 sessions (ensemble integration)
- Phase 4: 2-3 sessions (Dykstra feasibility + Pareto constraints)
- Phase 5: 1-2 sessions (sensitivity-map regularization)
- Phase 6+: 1-2 sessions (deprecation + cleanup)

**Total**: 11-21 sessions over 6-12 weeks. **$0 GPU** (design + paired-comparison runs on already-landed posterior anchors).

#### Dependency chain

- Phase 1 must land before Phase 2 (need invocation point)
- Phase 2 must complete ≥3 adjuster ablations before Phase 3 (need ensemble inputs)
- Phase 4 depends on `tac.findings_lagrangian.partition.build_initial_partition()` + Dykstra feasibility helper landing as canonical
- Phase 5 depends on Dimension 4 (domain priors) + master-gradient authoritative anchors

#### 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — Phase 5 integrates `tac.sensitivity_map` into the regularization term
2. **Pareto constraint**: ACTIVE — Phase 4 adds Dykstra alternating projections per Catalog #296
3. **Bit-allocator hook**: ACTIVE — solver-derived per-byte allocation per CLAUDE.md "Meta-Lagrangian/Pareto solver"
4. **Cathedral autopilot dispatch hook**: ACTIVE PRIMARY — this dimension IS the cathedral autopilot's mathematical heart
5. **Continual-learning posterior update**: ACTIVE — posterior anchors drive `posterior_update_from_anchors` per phase
6. **Probe-disambiguator**: ACTIVE — solver-derived divergence from hand-derived adjuster IS the canonical disambiguator (per Phase 2 step 4)

#### Reactivation criteria

If Phase 2 paired-comparison shows solver-derived diverges from hand-derived adjuster by > 2σ on ≥3 of 10 adjusters AND divergence does not converge after 30 days of anchor accumulation, DEFER to T3 architecture review on whether the solver formulation is correct vs the hand-derived adjusters encode signal the solver doesn't see.

#### Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "Solver-derived dual variables will match hand-derived adjusters" | CARGO-CULTED until Phase 2 empirically validates | Each hand-derived adjuster encodes operator + sister-subagent expertise; the solver may capture this OR may miss signal |
| "Phase 2 paired-comparison on continual-learning posterior anchors is sufficient validation" | HARD-EARNED | 118 anchors / 5+ months of data; statistical power adequate |
| "Unified solver subsuming 10 adjusters extincts the apparatus_maintenance overhead" | CARGO-CULTED until empirically demonstrated | This is the structural claim; needs 30-day post-deprecation observation window |
| "TRACK A (Gaussian) + TRACK B (NumPyro) ensemble preserves per-track uncertainty" | HARD-EARNED | Slot 20 T3 council 3-round verdict; canonical `ensemble_prediction_from_tracks` already tested |

#### Observability surface

- Inspectable per layer: each Phase 2 ablation emits `paired_comparison_<adjuster>_<utc>.json` with original delta + solver delta + residual + Bayesian update
- Decomposable per signal: `LagrangianDualResult` carries per-term contributions
- Diff-able across runs: phase-by-phase ablation results queryable from continual-learning posterior
- Queryable post-hoc: `tac.findings_lagrangian.posterior.posterior_sample(...)` exposes per-equation posterior
- Cite-able: each phase emits canonical Provenance per Catalog #323
- Counterfactual-able: paired-comparison IS the canonical counterfactual (with vs without solver)

#### Horizon-class declaration

`horizon_class: frontier_protecting` (apparatus_maintenance enabling future frontier_breaking) — per T3 Decision 5 long-term mission contribution.

### Dimension 2 — Feedback-loop frequency + closing

#### Current state

Continual-learning posterior + canonical equations registry exist (FULLY_WIRED for read; write at landing-time only):

- `tac.continual_learning.posterior_update_locked` — called ONLY at empirical-anchor landing time (i.e., when a Modal dispatch outcome lands or a paired-CUDA-CPU result lands)
- `tac.canonical_equations.update_equation_with_empirical_anchor` — called ONLY at canonical-equation refresh time (operator-triggered)
- Cathedral autopilot main() reads posterior at `--load-continual-posterior` flag invocation; does NOT auto-refresh per iteration

**Gap**: feedback loop is OPEN at the cathedral autopilot iteration boundary. Per-iteration empirical anchors (when they arrive) do NOT trigger immediate ranker re-rank within the same iteration. The next ranker iteration consumes the new anchor only when explicitly reloaded.

#### Smarter target state

Tight per-iteration feedback loop:

1. Empirical anchor arrives mid-iteration (Modal harvest fires, paired-CUDA-CPU result lands, probe-disambiguator verdict registered)
2. `posterior_update_locked` invoked immediately
3. `update_equation_with_empirical_anchor` invoked immediately for any equation citing the anchor's canonical consumers
4. Cathedral autopilot ranker re-ranks remaining candidates in same iteration with new posterior
5. Sister consumers `update_from_anchor` called on the new anchor

#### Mechanism

**Step 2.1**: Wire `tac.continual_learning.posterior_update_locked` as auto-invoked subscriber to `tac.deploy.modal.call_id_ledger.update_call_id_outcome` events. Canonical pattern: when an outcome row lands (status=harvested / failed), the auto-subscriber synthesizes a `ContestResult` row and calls `posterior_update_locked` IF the outcome carries `score_axis ∈ {[contest-CPU], [contest-CUDA]}` AND `archive_sha256` present.

**Step 2.2**: Wire `tac.canonical_equations.auto_recalibrate_from_continual_learning_posterior` as a per-iteration consumer in cathedral autopilot main() body. Already exists as a callable; missing the per-iteration invocation site.

**Step 2.3**: Extend `invoke_cathedral_consumers_on_candidates` to invoke each consumer's `update_from_anchor(anchor)` method BEFORE consuming candidates, so consumers see fresh posterior. Currently only `consume_candidate(candidate)` is auto-invoked per Catalog #336.

**Step 2.4**: Add `--max-iteration-anchor-lookback-seconds N` flag to cathedral autopilot CLI so operator can tune how recent the anchor must be to trigger mid-iteration re-rank.

#### Cost / wall-clock

- Step 2.1: 2 sessions (canonical subscriber pattern; testing with synthetic anchors)
- Step 2.2: 1 session (invocation site + tests)
- Step 2.3: 1-2 sessions (auto-discovery extension + tests)
- Step 2.4: 1 session (CLI flag + tests)

**Total**: 5-7 sessions over 2-3 weeks. **$0 GPU**.

#### Dependency chain

- Independent of Dimensions 1, 3, 4, 6
- Synergizes with Dimension 5 (continual-learning closure IS the auto-subscribed surface)

#### 6-hook wire-in declaration

1. Sensitivity-map: N/A (this is feedback-loop wiring; no sensitivity signal contribution)
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE — per-iteration re-rank IS the dispatch surface change
5. Continual-learning posterior: ACTIVE PRIMARY — auto-subscriber IS the canonical posterior closure
6. Probe-disambiguator: N/A

#### Reactivation criteria

If auto-subscriber causes performance regression (cathedral autopilot iteration time > 30s per the standing harness budget per Catalog #299 sister discipline), implement lookback-window filter (`--max-iteration-anchor-lookback-seconds`) or queue async re-rank.

#### Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "Per-iteration anchor closure produces strictly better ranker decisions" | HARD-EARNED for FULLY_WIRED dimensions; CARGO-CULTED for 65% STUB consumers | Posterior changes only affect score-mutating adjusters; STUB consumers ignore posterior |
| "Mid-iteration re-rank does not violate per-iteration ordering invariants" | HARD-EARNED | Cathedral autopilot ranker is stateless per candidate; re-rank is idempotent |
| "Auto-subscriber pattern doesn't break existing Catalog #245 / #313 ledger semantics" | HARD-EARNED via subscriber-pattern canonical contract | Subscriber reads ledger events; does not mutate ledger |

#### Observability surface

- Inspectable per layer: every `posterior_update_locked` invocation logs to `.omx/state/commit-serializer.log` sister log (proposed: `.omx/state/posterior_subscriber_log.jsonl`)
- Decomposable per signal: each subscriber emits per-anchor decision rationale
- Diff-able across runs: subscriber log is append-only; diff via timestamp window
- Queryable post-hoc: subscriber log + posterior posterior_sample
- Cite-able: each subscriber event carries Provenance + cite-chain to source anchor
- Counterfactual-able: `--no-auto-posterior-subscriber` CLI flag for paired-comparison

#### Horizon-class declaration

`horizon_class: frontier_protecting` (apparatus closure; enables future frontier-breaking discoveries to propagate immediately)

### Dimension 3 — Problem-space grounding (per-axis prediction decomposition)

#### Current state

Contest scorer is `S = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes/N` per canonical equation registry. ALL existing ranker surfaces emit SCALAR `predicted_score_delta` — the 3-axis decomposition is collapsed at the consumer boundary.

Per Carmack T3 dissent + Dimension 1 Phase 4 motivation: the contest scorer's mechanics are NOT honored at the consumer level. Per-axis improvements (e.g., "this candidate is +0.001 worse on seg but -0.005 better on pose") cannot be expressed; the scalar ΔS hides per-axis tradeoffs that the Pareto frontier would surface.

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" (UPDATED 2026-05-04): the 77× SegNet > PoseNet rule applies at OLD 1.x operating point; at PR106 frontier the FLIP is empirical: marginal pose 2.71× SegNet. The current ranker does NOT see this — every consumer collapses to scalar ΔS.

Per `score_lagrangian_consumer` (`src/tac/cathedral_consumers/score_lagrangian_consumer/__init__.py`): canonical equation `S = sqrt(10 * d_pose) + 100 * d_seg + 25 * archive_bytes/N` IS cited; per-axis Lagrangian multipliers (λ_seg / λ_pose / λ_rate) are the conceptual primitives.

#### Smarter target state

Cathedral consumers emit `(d_seg, d_pose, archive_bytes)` triple instead of scalar ΔS:

```python
return {
    "predicted_axis_decomposition": {
        "d_seg_delta": -0.0001,      # SegNet distortion delta (lower better)
        "d_pose_delta": +0.000001,   # PoseNet distortion delta
        "archive_bytes_delta": -200,  # archive size delta (negative = smaller)
    },
    "predicted_score_delta": -0.00012,  # composed via canonical equation
    "axis_tag": "[predicted]",
    "promotable": False,
}
```

The cathedral autopilot ranker composes per-axis predictions into total ΔS via canonical equation; per-axis Lagrangian multipliers from Dimension 1 Phase 4 weight the composition.

#### Mechanism

**Step 3.1**: Extend `CathedralConsumerContract` (Catalog #335) to OPTIONALLY include `predicted_axis_decomposition` field. Backward-compatible (existing scalar consumers continue to work; new field is opt-in).

**Step 3.2**: Add canonical helper `tac.score_composition.compose_score_from_axes(d_seg, d_pose, archive_bytes, n_pixels)` that returns scalar score per the contest equation. Pin canonical equation `contest_score_formula_v1` in `tac.canonical_equations` registry.

**Step 3.3**: Extend cathedral autopilot ranker to read `predicted_axis_decomposition` when present; compose via `compose_score_from_axes` weighted by per-axis Lagrangian multipliers from Dimension 1 Phase 4 Dykstra dual variables.

**Step 3.4**: Convert 3-5 high-EV consumers (per Catalog #354 exploit consumers) to emit per-axis decomposition:
- `per_segnet_class_chroma_consumer` (per-class seg contribution; natural per-axis emission)
- `top_k_byte_sensitivity_consumer` (rate-axis dominant)
- `per_pair_gradient_clustering_consumer` (pose-axis dominant for ego-motion pairs)
- `substrate_fit_diagnostic_consumer` (seg + pose; not rate)
- `information_theoretic_floor_consumer` (Cramer-Rao bound per axis)

**Step 3.5**: Add canonical preflight Catalog #NEW (recommend sister extension to #341 via Catalog #287 v2 precedent) that REQUIRES consumers emitting `predicted_axis_decomposition` to ALSO emit scalar `predicted_score_delta` for backward compatibility.

#### Cost / wall-clock

- Step 3.1: 1 session (Protocol extension + tests)
- Step 3.2: 1 session (canonical helper + equation registration)
- Step 3.3: 1-2 sessions (ranker composition + tests)
- Step 3.4: 3-5 sessions (convert 3-5 consumers; per-consumer paired comparison)
- Step 3.5: 1 session (canonical preflight extension)

**Total**: 7-10 sessions over 3-4 weeks. **$0 GPU**.

#### Dependency chain

- Synergistic with Dimension 1 Phase 4 (Dykstra dual variables weight composition)
- Independent of Dimensions 2, 4, 5, 6 (composition can land before / after these)

#### 6-hook wire-in declaration

1. Sensitivity-map: ACTIVE — per-axis decomposition IS the canonical sensitivity surface at the consumer boundary
2. Pareto constraint: ACTIVE — per-axis enables Dykstra alternating projections to operate on actual Pareto polytope (seg / pose / rate / archive)
3. Bit-allocator: ACTIVE — per-axis archive_bytes_delta IS the bit-allocator's primary signal
4. Cathedral autopilot dispatch: ACTIVE — composition happens at dispatch decision
5. Continual-learning posterior: ACTIVE — per-axis posterior anchors enable per-axis Bayesian updates
6. Probe-disambiguator: ACTIVE — per-axis residuals disambiguate which axis a prediction was wrong on (vs scalar residual which hides axis attribution)

#### Reactivation criteria

If converting top-3 high-EV consumers to per-axis emission produces no measurable ranking change (ranker still produces same dispatch order across 30-day window), revisit whether per-axis Lagrangian multipliers are well-calibrated.

#### Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "Per-axis decomposition is more informative than scalar ΔS" | HARD-EARNED | CLAUDE.md PR97 anti-pattern empirical anchor: PR97 lost 0.042 by trading pose for seg unwittingly; per-axis would have surfaced the tradeoff |
| "Cathedral consumers can compute per-axis predictions without empirical anchors" | CARGO-CULTED for STUB consumers; HARD-EARNED for canonical-equation-citing consumers | Per-axis from first-principles (per `information_theoretic_floor_consumer`) is HARD-EARNED; per-axis from heuristic is CARGO-CULTED |
| "Backward compatibility (require both scalar + per-axis) preserves existing ranker invariants" | HARD-EARNED | Scalar IS composed from per-axis; preservation by construction |

#### Observability surface

- Inspectable per layer: per-consumer per-axis predictions in output JSON
- Decomposable per signal: per-axis residuals queryable post-hoc
- Diff-able across runs: per-axis posterior diffable
- Queryable post-hoc: per-axis canonical equations + posteriors
- Cite-able: per-axis Provenance per Catalog #323
- Counterfactual-able: per-axis ablation (e.g., "what if we zeroed the pose-axis contribution?")

#### Horizon-class declaration

`horizon_class: frontier_pursuit` (per-axis enables the marginal-pose-2.71×-SegNet operating-point shift to be exploitable; this IS the canonical mechanism for moving the plateau)

### Dimension 4 — Domain grounding (dashcam / ego-motion / comma2k19 priors)

#### Current state

Several domain-specific packages exist but are NOT wired into cathedral autopilot:

- `tac.ego_flow` — ego-motion canonical helper
- `tac.foveation_field` + `tac.hyperbolic_foveation` — foveation priors
- `tac.lapose_foveation_atoms` + `tac.lapose_foveation_payload_candidate` + `tac.lapose_foveation_runtime_skeleton` — LAPose substrate-engineering surfaces
- `tac.codec_pipeline_raft_pose` — RAFT-based pose pipeline
- `tac.categorical_substrate` + `tac.categorical_openpilot_mask_prior_contract` — openpilot mask priors
- `tac.substrates.pretrained_driving_prior` (DP1) — comma2k19 codebook distillation per Catalog #209 + #210
- `tac.substrates.pretrained_driving_prior.local_chunk_cache` (Catalog #213) — comma2k19 chunk cache

Per Catalog #213: comma2k19 chunk fetching is canonical-helper-routed. Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L1: substrate must train against `upstream/videos/0.mkv` (the contest video). Per CLAUDE.md "Forbidden contest_video_leakage" Catalog #209: distillation must use comma2k19 OOD data, not the contest video.

**Gap**: NO canonical `tac.domain_priors` namespace exists. The per-frame difficulty atlas is per-pair (from master-gradient) NOT per-frame. The ego-motion concentration is encoded in TT5L V2 (V2 telescopic foveation) but not surfaced as a per-frame prior. Per-class statistical priors live in `tac.categorical_substrate` but not auto-discovered by cathedral autopilot.

#### Smarter target state

Canonical `tac.domain_priors` namespace exposing:

- `per_frame_difficulty_atlas(video_path)` → per-frame difficulty score (motion magnitude / scene complexity / scorer-class entropy)
- `ego_motion_concentration_atlas(video_path)` → per-frame ego-motion magnitude (canonical Atick-Redlich / Rao-Ballard predictive coding alignment)
- `per_class_statistical_priors(video_path)` → per-class pixel count + per-class chroma variance + per-class motion magnitude (canonical openpilot mask prior contract)
- `per_pair_difficulty_atlas(video_path)` → per-pair (frame0, frame1) joint difficulty (existing master-gradient-derived; canonical wrapper)

Cathedral consumer wire-in: `domain_prior_consumer` package per Catalog #335 contract.

#### Mechanism

**Step 4.1**: Create canonical `tac.domain_priors` namespace by sister-extending existing packages (per Decision 10 canonical-helper-sister-extension over new-tool):
- `tac.domain_priors.per_frame_difficulty_atlas` wraps `tac.master_gradient_consumers.load_per_pair_gradient_from_anchor` + aggregates per-frame
- `tac.domain_priors.ego_motion_concentration_atlas` wraps `tac.ego_flow` + canonical comma2k19 ego-motion prior
- `tac.domain_priors.per_class_statistical_priors` wraps `tac.categorical_substrate` + `tac.categorical_openpilot_mask_prior_contract`
- `tac.domain_priors.per_pair_difficulty_atlas` wraps existing per-pair surface

**Step 4.2**: Land canonical equations in `tac.canonical_equations` registry:
- `per_frame_difficulty_atlas_v1` (validates against current frontier 0.192051 per-frame breakdown)
- `ego_motion_concentration_dashcam_v1` (validates against comma2k19 distribution + contest video)
- `per_class_pixel_count_distribution_v1` (validates against openpilot mask prior)

**Step 4.3**: Wire `domain_prior_consumer` cathedral consumer package. Per Catalog #335 contract; per Catalog #341 emits `[predicted]` axis tag + non-promotable markers. Consumer cites the 3 canonical equations + provides per-candidate per-axis predictions per Dimension 3.

**Step 4.4**: Validate domain priors against the 7 asymptotic-pursuit candidates from T3 Decision 4. For each candidate, check whether the substrate's design memo cites domain priors (e.g., Z6 FiLM ego-motion conditioning explicitly uses `tac.ego_flow` per design memo). Surface missing citations.

#### Cost / wall-clock

- Step 4.1: 3-4 sessions (4 canonical helpers + tests)
- Step 4.2: 2-3 sessions (3 canonical equations + initial empirical anchors)
- Step 4.3: 1-2 sessions (cathedral consumer + tests)
- Step 4.4: 1 session (audit)

**Total**: 7-10 sessions over 3-4 weeks. **$0 GPU** (priors derived from already-landed master-gradient anchors + comma2k19 cache + contest video).

#### Dependency chain

- Independent of Dimensions 1, 2, 3, 5, 6
- Synergistic with Dimension 3 (per-axis priors compose into per-axis predictions)

#### 6-hook wire-in declaration

1. Sensitivity-map: ACTIVE — domain priors ARE the canonical per-frame / per-pair / per-class sensitivity surface
2. Pareto constraint: N/A (priors don't constrain feasibility directly; they inform weights)
3. Bit-allocator: ACTIVE — per-frame difficulty drives per-frame bit allocation; per-class statistical priors drive per-class chroma allocation
4. Cathedral autopilot dispatch: ACTIVE — `domain_prior_consumer` integrates priors into ranking
5. Continual-learning posterior: ACTIVE — per-frame / per-pair / per-class anchors update domain canonical equations
6. Probe-disambiguator: ACTIVE — domain priors disambiguate between candidates that score similarly on average but differ on hard-frame / easy-frame distribution

#### Reactivation criteria

If domain priors do not surface materially different ranking decisions across the 7 asymptotic-pursuit candidates (i.e., the priors are uninformative for the candidate distribution), revisit whether the priors are well-calibrated against the contest video's specific statistics (vs comma2k19's broader distribution).

#### Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "comma2k19 distribution is informative for contest video predictions" | HARD-EARNED per Catalog #209 / #210 DP1 validation | DP1 codebook distillation already empirically validated against OOD-then-applied-to-contest pattern |
| "Per-frame difficulty atlas can be derived from per-pair master-gradient by aggregation" | CARGO-CULTED-PENDING-EMPIRICAL | Per-pair → per-frame aggregation requires choice of aggregation operator (mean / max / Volterra); per-pair has dual-frame coupling that per-frame loses |
| "Ego-motion concentration aligns with predictive-coding hierarchical Bayesian structure" | HARD-EARNED per Atick-Redlich + Rao-Ballard canonical | Z6 / Z7 / Z8 / TT5L V2 design memos all bind ego-motion to the predictive-coding lineage |
| "Per-class statistical priors compose with per-class chroma allocation per Catalog #354 exploit #5" | HARD-EARNED | exploit #5 IS the canonical mechanism; existing consumer wired |

#### Observability surface

- Inspectable per layer: per-frame / per-pair / per-class atlas surfaces queryable per video
- Decomposable per signal: per-axis (motion / scene complexity / scorer entropy) decomposition
- Diff-able across runs: contest video vs comma2k19 distribution comparison
- Queryable post-hoc: canonical helpers expose atlas APIs
- Cite-able: each atlas tagged with comma2k19 chunk sha + contest video sha + canonical equation citation
- Counterfactual-able: "what if we used uniform priors?" via `--no-domain-priors` cathedral autopilot CLI flag

#### Horizon-class declaration

`horizon_class: frontier_pursuit` (domain priors enable substrate-routing that respects per-frame / per-pair / per-class structure — canonical mechanism for breaking out of the plateau via informed routing)

### Dimension 5 — Continual-learning posterior closing

#### Current state

Per Dimension 2 above, the continual-learning posterior is FULLY_WIRED for READ but only updated at landing-time. Beyond auto-subscriber wiring (Dimension 2), the deeper gap is **closed-loop discipline** at the consumer side:

- Cathedral consumers do NOT call `posterior_update_locked` themselves (Catalog #341 forbids non-canonical writes); they emit predicted payloads only
- Per-falsification: when an empirical anchor falsifies a consumer's prediction, the consumer does NOT propose alternative substrate candidates OR boost sister substrates
- Per-ratification: when an empirical anchor RATIFIES a consumer's prediction, the consumer does NOT increase confidence weight on related candidates

Per WIRE-IN-RIGOR finding: "the cathedral consumers framework's heart is the 10 in-main-line adjusters + the 5 ledgers, not the 44 cathedral consumers" — the closure gap IS that consumers don't close on their own predictions.

#### Smarter target state

Closed-loop discipline at every consumer:

1. Per-anchor: every empirical anchor → Bayesian update → consumer's `update_from_anchor` extension that ALSO propagates to sister candidates
2. Per-falsification: consumer recommends DEFER + sister-candidate boost (per CLAUDE.md "Forbidden premature KILL" reactivation-criteria pinning)
3. Per-ratification: consumer recommends BOOST + scope-extension to similar candidates

The mechanism IS the canonical `tac.continual_learning.posterior_update_locked` + `tac.canonical_equations.update_equation_with_empirical_anchor` + sister-candidate-graph traversal.

#### Mechanism

**Step 5.1**: Extend `CathedralConsumerContract` (Catalog #335) with optional `propose_sister_candidates(anchor)` method that returns sister substrates to boost (on ratification) or defer (on falsification).

**Step 5.2**: Land canonical sister-candidate-graph: `tac.substrate_kinship_graph` namespace that records (substrate_a, substrate_b, kinship_score) edges. Kinship sources: shared canonical equations, shared lane_class, shared distinguishing-feature tokens, shared horizon-class. Used by `propose_sister_candidates` to traverse.

**Step 5.3**: Extend `invoke_cathedral_consumers_on_candidates` to invoke `propose_sister_candidates(anchor)` when anchor lands; propagate boost/defer to sister candidate scoring.

**Step 5.4**: Per CLAUDE.md "Forbidden premature KILL without research exhaustion" — every defer recommendation MUST include reactivation_criteria + register via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313. This closes Loop 1 (probe-outcome ledger) automatically.

**Step 5.5**: Per Catalog #344 — every ratified prediction MUST refresh canonical equation via `update_equation_with_empirical_anchor`. This closes Loop 2 (canonical equations) automatically.

**Step 5.6**: Add `--closed-loop-discipline-strict` CLI flag to cathedral autopilot that REFUSES iterations where Loop 1 OR Loop 2 closure is missing for any anchor in the lookback window.

#### Cost / wall-clock

- Step 5.1: 2 sessions (Protocol extension + tests)
- Step 5.2: 3-4 sessions (canonical sister-kinship-graph + initial edges from existing lane registry data)
- Step 5.3: 1-2 sessions (auto-discovery extension + tests)
- Step 5.4: 1 session (Catalog #313 wire-in)
- Step 5.5: 1 session (Catalog #344 wire-in)
- Step 5.6: 1 session (CLI flag + tests)

**Total**: 9-11 sessions over 4-5 weeks. **$0 GPU**.

#### Dependency chain

- Builds on Dimension 2 (auto-subscriber wiring)
- Sister-kinship graph independent of other dimensions
- Synergistic with Dimension 3 (per-axis predictions enable per-axis sister-kinship)

#### 6-hook wire-in declaration

1. Sensitivity-map: ACTIVE — sister-kinship graph IS sensitivity at the substrate-level
2. Pareto constraint: N/A (closure operates on posterior, not on Pareto polytope directly)
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE — closure DISPATCH propagation is the canonical surface
5. Continual-learning posterior: ACTIVE PRIMARY — this dimension IS posterior closure
6. Probe-disambiguator: ACTIVE — per-falsification disambiguator triggers ledger update

#### Reactivation criteria

If sister-candidate boost / defer produces measurable ranker drift but downstream paid dispatches do not validate the boosts/defers (i.e., kinship graph is uninformative), revisit kinship-source taxonomy.

#### Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "Sister-candidate kinship is well-defined via shared canonical equations + lane_class + distinguishing-feature tokens" | CARGO-CULTED-PENDING-EMPIRICAL | Kinship taxonomy is a design choice; needs paired-comparison against alternative taxonomies |
| "Per-falsification → sister-defer is canonical per CLAUDE.md 'Forbidden premature KILL'" | HARD-EARNED | Existing non-negotiable + Catalog #313 sister discipline |
| "Per-ratification → sister-boost is canonical" | HARD-EARNED via posterior Bayesian update structure | Standard Bayesian inference: confirming evidence shifts posterior toward related hypotheses |
| "Closed-loop discipline doesn't introduce feedback loops that cause oscillation" | CARGO-CULTED-PENDING-EMPIRICAL | Bayesian posteriors are monotonic in evidence accumulation; should not oscillate; needs empirical validation |

#### Observability surface

- Inspectable per layer: every closure event logged with cite-chain
- Decomposable per signal: per-loop (Loop 1 ledger / Loop 2 equation) closure rate
- Diff-able across runs: closure-rate trend over time
- Queryable post-hoc: closed-loop log + posterior + canonical equations
- Cite-able: each closure event carries Provenance + Catalog #313 / #344 anchor
- Counterfactual-able: `--no-closed-loop-discipline` CLI flag for paired comparison

#### Horizon-class declaration

`horizon_class: frontier_protecting → frontier_breaking transition` (closure IS the mechanism that makes individual frontier-breaking discoveries propagate to sister candidates)

### Dimension 6 — Cathedral consumer dual-tier architecture (operator may not have explicitly named)

#### Current state

Per Catalog #341 canonical contract: ALL cathedral consumers return `predicted_delta_adjustment=0.0`. This is design intent (observability-only, non-promotable, no score mutation per Catalog #335 + #341). Per WIRE-IN-RIGOR: 44/44 production consumers compliant; the framework's HEART is the 10 in-main-line adjusters, not the 44 consumers.

The design tension surfaced:
- Catalog #341 is STRUCTURALLY CORRECT for observability-only annotations
- 44 consumers fire → operator mental model is "44 things are influencing dispatch ranking" — TRUE for invocation, FALSE for score mutation
- Per Dimension 3 + 4 + 5 plans above: consumers WILL need to contribute to ranking (per-axis predictions, domain priors, sister-kinship)

#### Smarter target state

Resolve the tension via DUAL-TIER ARCHITECTURE:

**Tier A (current Catalog #341 contract)**: Observability-only consumers. Return `predicted_delta_adjustment=0.0` + `axis_tag="[predicted]"` + `promotable=False`. Backward-compatible; existing 44 consumers stay in Tier A.

**Tier B (new contract; recommend Catalog #NEW sister extension to #341)**: Contributing consumers. Return `predicted_delta_adjustment` ∈ ℝ + per-axis decomposition + `axis_tag="[predicted]"` + `promotable=False`. CRUCIAL: per-row non-promotability tag preserved even when contributing (per CLAUDE.md "Forbidden score claims" non-negotiable).

The distinction: Tier B contributes to RANKING but NOT to PROMOTION. Promotion requires empirical-anchor on contest-compliant hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. Ranking influence is design-intentful.

#### Mechanism

**Step 6.1**: Define canonical `TIER_A_CONSUMER` / `TIER_B_CONSUMER` enum in `tac.cathedral.consumer_contract`. Default to TIER_A for backward compatibility.

**Step 6.2**: Add `consumer_tier: ConsumerTier` field to `CathedralConsumerContract`. Tier A consumers MUST return `predicted_delta_adjustment=0.0`; Tier B consumers MAY return non-zero.

**Step 6.3**: Extend Catalog #341 STRICT preflight gate (per Catalog #287 v2 sister extension precedent) to enforce per-tier contract: Tier A refused if non-zero; Tier B refused if missing per-axis decomposition (depends on Dimension 3 Step 3.1).

**Step 6.4**: Update README at `src/tac/cathedral_consumers/README.md` to document the dual-tier design + when to use each tier. Per T3 Decision 7 (EV/$ rank 7): close the operator-mental-model gap explicitly.

**Step 6.5**: For each of the 3-5 high-EV consumers identified in Dimension 3 Step 3.4, propose Tier B promotion via per-consumer paired-comparison validation. ANY consumer promoting to Tier B MUST satisfy Dimension 3 per-axis emission AND Dimension 5 closed-loop discipline.

#### Cost / wall-clock

- Step 6.1: 1 session (enum + tests)
- Step 6.2: 1 session (Protocol field + tests)
- Step 6.3: 1 session (Catalog #341 sister extension)
- Step 6.4: 1 session (README update + cross-references)
- Step 6.5: 3-5 sessions (per-consumer Tier B promotion per Dimension 3 high-EV consumers)

**Total**: 7-9 sessions over 3-4 weeks. **$0 GPU**.

#### Dependency chain

- Depends on Dimension 3 (per-axis decomposition is Tier B prerequisite)
- Independent of Dimensions 1, 2, 4, 5 (can land before or after)

#### 6-hook wire-in declaration

1. Sensitivity-map: N/A (architectural decision; no sensitivity signal)
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE — Tier B consumers contribute to ranking
5. Continual-learning posterior: N/A (Dimension 5 owns this)
6. Probe-disambiguator: ACTIVE — Tier B contract distinguishes ranking-influence from promotion-influence

#### Reactivation criteria

If Tier B promotion of 3-5 high-EV consumers does NOT measurably change ranking (i.e., scalar Tier A IS sufficient), revisit whether Tier B contract adds value vs adds complexity.

#### Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| "Dual-tier preserves Catalog #341 design intent at the per-row level" | HARD-EARNED | Per-row non-promotability tag IS the structural protection; tier classifies emission, not promotion |
| "Tier B contributing consumers won't violate Catalog #127 custody routing" | HARD-EARNED | Tier B emits `[predicted]` axis tag + `promotable=False` per the same canonical Provenance discipline |
| "Operator mental model gap is the actual problem (not the design intent)" | HARD-EARNED per WIRE-IN-RIGOR Assumption-Adversary verdict | "the framework's HEART is the 10 adjusters" — operator knowing this WAS the structural finding |

#### Observability surface

- Inspectable per layer: each consumer's `consumer_tier` field queryable
- Decomposable per signal: per-tier adjustment-rate statistics
- Diff-able across runs: paired comparison Tier A vs Tier B ranking
- Queryable post-hoc: per-tier auto-discovery roster
- Cite-able: each consumer's tier classification carries Provenance
- Counterfactual-able: `--tier-a-only` CLI flag

#### Horizon-class declaration

`horizon_class: apparatus_maintenance` (architectural clarification; enables future score-mutating consumers without violating canonical contract)

---

## 3. Recommended phase ordering + critical path

### Critical-path sequencing (per dependency graph in sister deliverable)

```
CRITICAL PATH (frontier-mutating; 12-18 weeks):
    Dimension 1 Phase 1 (META-LAGRANGIAN-WIRE-1; in flight)
        ↓
    Dimension 1 Phase 2 (per-adjuster ablation)
        ↓
    Dimension 3 Step 3.1-3.3 (per-axis Protocol + ranker composition)
        ↓
    Dimension 1 Phase 4 (Dykstra feasibility + Pareto constraints; uses per-axis from Dim 3)
        ↓
    Dimension 1 Phase 3 (TRACK A + TRACK B ensemble; uses Phase 2 ablation outputs)
        ↓
    Dimension 1 Phase 5 (sensitivity-map regularization; uses Dim 4 domain priors when available)
        ↓
    Dimension 1 Phase 6+ (deprecate adjusters; structural extinction of hand-derived weights)

PARALLEL PATHS (frontier-protecting; 8-12 weeks; can run alongside critical path):
    Dimension 2 (per-iteration feedback loop)
        Independent; landing-order flexibility

    Dimension 4 (domain priors)
        Independent; lands before Dim 1 Phase 5 to feed regularization

    Dimension 5 (closed-loop discipline)
        Depends on Dim 2; independent of Dim 1/3/4/6

    Dimension 6 (dual-tier architecture)
        Depends on Dim 3 Step 3.1; independent of Dim 1/2/4/5
```

### Recommended landing order

1. **Weeks 1-2 (in flight)**: Dimension 1 Phase 1 (META-LAGRANGIAN-WIRE-1)
2. **Weeks 2-3**: Dimension 2 (feedback-loop frequency) — independent; quick win
3. **Weeks 3-6**: Dimension 3 (per-axis decomposition) — unlocks Dimensions 1 Phase 4 + 6
4. **Weeks 4-7**: Dimension 4 (domain priors) — parallel; unlocks Dimension 1 Phase 5
5. **Weeks 6-10**: Dimension 1 Phase 2 (per-adjuster ablation) — needs Phase 1 + paired comparison time
6. **Weeks 8-12**: Dimension 6 (dual-tier architecture) — depends on Dimension 3 landing
7. **Weeks 10-14**: Dimension 5 (closed-loop discipline) — depends on Dimension 2 + 3 + 6
8. **Weeks 12-18**: Dimension 1 Phases 3 + 4 + 5 + 6+ — depend on Phase 2 + Dim 3 + Dim 4

**Total wall-clock**: 12-18 weeks for full critical path + parallel paths.

---

## 4. Per-substrate-symposium prerequisite checks per Catalog #325

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable: any paid Modal/Lightning/Vast.ai dispatch >$0.30 on a substrate requires a 6-step symposium contract.

**Per-dimension paid-dispatch requirements**:

| Dimension | Requires paid dispatch? | Symposium needed? |
|---|---|---|
| 1 (mathematical grounding) | NO (paired-comparison on already-landed posterior anchors) | NO |
| 2 (feedback-loop frequency) | NO | NO |
| 3 (per-axis decomposition) | NO (composition of existing predictions) | NO |
| 4 (domain priors) | NO (derived from already-landed master-gradient + comma2k19 cache) | NO |
| 5 (closed-loop discipline) | NO (Bayesian + ledger updates) | NO |
| 6 (dual-tier architecture) | NO (architectural; no GPU work) | NO |

**ALL 6 DIMENSIONS ARE $0 GPU** by design. The blueprint operates entirely on the existing posterior + ledger + canonical-equations + master-gradient infrastructure. Per CLAUDE.md "Mission alignment" Consequence 4: "frontier-breaking moves DOMINATE rigor budget when the operator declares a frontier-breaking direction" — this blueprint is frontier_protecting (apparatus_maintenance) and does not require race-mode operator-frontier-override.

**Paid dispatch BUDGETS** for the 7 asymptotic-pursuit candidates from T3 Decision 4 are SEPARATE from this blueprint. Per T3 Decision 4: operator routes max 2 per 7-day window; cost $5-50 per dispatch. The blueprint accelerates the smarter ranking of those dispatches but does not generate them.

---

## 5. Mission-alignment check per CLAUDE.md "Mission alignment"

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5: every T2+ verdict MUST classify `council_predicted_mission_contribution`. This blueprint:

| Dimension | Mission contribution | Rationale |
|---|---|---|
| 1 (mathematical grounding) | `frontier_protecting → frontier_breaking transition` | Structural; subsumes hand-derived; enables future frontier-breaking discoveries to compose canonically |
| 2 (feedback-loop frequency) | `apparatus_maintenance` | Procedural acceleration; no direct score |
| 3 (per-axis decomposition) | `frontier_breaking enabler` | Per-axis enables the pose-axis-marginal-2.71×-SegNet operating-point shift to be exploitable |
| 4 (domain priors) | `frontier_breaking enabler` | Domain-aware routing IS the mechanism for plateau exit per HORIZON-CLASS asymptotic_pursuit |
| 5 (closed-loop discipline) | `frontier_protecting → frontier_breaking transition` | Closure propagates individual discoveries to sister candidates |
| 6 (dual-tier architecture) | `apparatus_maintenance` | Architectural clarification; enables future Tier B contributions |

**Net mission-contribution distribution**: ~50% `apparatus_maintenance + rigor_overhead`; ~50% `frontier_breaking enabler + transition`. Per Catalog #300 sister discipline: 50% is at the 60% threshold; recommend NEXT 30-day window's work bias toward Dimensions 3 + 4 (frontier-breaking enablers) over Dimensions 2 + 6 (apparatus-maintenance).

---

## 6. Operator-routable decisions surfaced

### Decision A: Confirm META-LAGRANGIAN-WIRE-1 succession plan

This blueprint assumes META-LAGRANGIAN-WIRE-1 (in flight) lands Dimension 1 Phase 1 (canonical solver invocation point). If META-LAGRANGIAN-WIRE-1 deviates (e.g., lands invocation but does NOT lay groundwork for Phase 2 ablation), this blueprint's Dimension 1 phasing needs revision. **Operator-routable**: read META-LAGRANGIAN-WIRE-1 landing memo when it lands; verify Phase 2 hook is present.

### Decision B: Approve Dimension 3 + 4 parallelization

Dimensions 3 + 4 are the frontier-breaking enablers per Mission Alignment. They are independent of each other and of Dimensions 1 / 2 / 5 / 6. Recommend operator approve parallel dispatch of 1 subagent for Dimension 3 + 1 subagent for Dimension 4 in the next 2-4 weeks. **Operator-routable**: spawn 2 design-only subagents per Catalog #325 N/A (no paid dispatch) + Catalog #229 PV discipline.

### Decision C: Defer Dimension 5 + 6 until Dimension 3 lands

Dimensions 5 (closed-loop) + 6 (dual-tier) depend on Dimension 3. Recommend DEFER until Dimension 3 Step 3.1 (Protocol extension) lands. **Operator-routable**: queue Dimension 5 + 6 subagents in TaskList; activate when Dimension 3 Step 3.1 commit lands.

### Decision D: Catalog quota management

Per Catalog #299 quota brake: catalog # at 354 / 400 ceiling. Each new gate proposed in this blueprint MUST satisfy Catalog #299 sister discipline (retire / file-level waiver / replace / sister-extension). Recommended sister-extension paths:
- Dimension 3 Step 3.5 → sister-extend Catalog #341 via Catalog #287 v2 precedent
- Dimension 6 Step 6.3 → sister-extend Catalog #341 (same)

**Operator-routable**: per Decision 6 of T3 symposium, prefer sister-extension over net-new gate.

### Decision E: HORIZON-CLASS budget allocation

Per Catalog #309 HORIZON-CLASS evaluation axis discipline + the operator standing directive on >=20% asymptotic-pursuit budget allocation. This blueprint's Dimensions 3 + 4 are asymptotic_pursuit enablers; Dimensions 2 + 6 are apparatus_maintenance. Allocation:
- 60% subagent capacity → frontier-breaking dimensions (1 Phase 2-N, 3, 4, 5)
- 40% subagent capacity → apparatus-maintenance (1 Phase 1, 2, 6)

**Operator-routable**: confirm allocation matches operator's standing horizon-class budget intent.

### Decision F: Cathedral consumer `findings_lagrangian_consumer` gap

Per Section 1.3 above: `findings_lagrangian_consumer` referenced in strategy graph but does NOT exist at `src/tac/cathedral_consumers/findings_lagrangian_consumer/`. Likely owned by META-LAGRANGIAN-WIRE-1. **Operator-routable**: verify in META-LAGRANGIAN-WIRE-1 landing memo; if not landed, escalate as Dimension 1 Phase 1 prerequisite.

### Decision G: PR #110 lifecycle NOT impacted

Per T3 Decision 1 + Staircase STEP 1: PR #110 lifecycle is HIGH-priority + NO active outreach. This blueprint operates entirely on internal apparatus (no PR-surface impact, no Claude attribution risk, no maintainer distraction). **Operator-routable**: confirm blueprint subagent dispatches do not touch any public-PR surface.

---

## 7. Concrete fix-list ranked by EV (top 15)

| Rank | Fix | Dimension | EV | Cost (sessions) | Surface |
|---|---|---|---|---|---|
| 1 | Dimension 1 Phase 2 (per-adjuster ablation) | 1 | HIGH | 5-10 | meta-Lagrangian |
| 2 | Dimension 3 Step 3.1-3.3 (per-axis Protocol + ranker composition) | 3 | HIGH | 3-4 | consumer contract |
| 3 | Dimension 2 Step 2.1 (auto-subscriber to call-id ledger) | 2 | HIGH | 2 | feedback loop |
| 4 | Dimension 4 Step 4.3 (`domain_prior_consumer` cathedral package) | 4 | HIGH | 1-2 | domain priors |
| 5 | Dimension 1 Phase 4 (Dykstra feasibility + Pareto constraints) | 1 | HIGH | 2-3 | meta-Lagrangian |
| 6 | Dimension 5 Step 5.2 (sister-kinship graph) | 5 | MEDIUM | 3-4 | substrate kinship |
| 7 | Dimension 4 Step 4.1 (`tac.domain_priors` namespace) | 4 | MEDIUM | 3-4 | canonical helpers |
| 8 | Dimension 3 Step 3.4 (convert 3-5 high-EV consumers per-axis) | 3 | MEDIUM | 3-5 | consumer migration |
| 9 | Dimension 6 Step 6.1-6.3 (dual-tier architecture + Protocol + gate) | 6 | MEDIUM | 3 | consumer contract |
| 10 | Dimension 4 Step 4.2 (3 canonical equations for domain priors) | 4 | MEDIUM | 2-3 | canonical equations |
| 11 | Dimension 1 Phase 3 (TRACK A + TRACK B ensemble) | 1 | MEDIUM | 1-2 | meta-Lagrangian |
| 12 | Dimension 5 Step 5.3 (sister-candidate auto-discovery extension) | 5 | MEDIUM | 1-2 | cathedral autopilot |
| 13 | Dimension 2 Step 2.3 (consumer pre-anchor `update_from_anchor`) | 2 | MEDIUM | 1-2 | feedback loop |
| 14 | Dimension 5 Step 5.4-5.5 (Catalog #313 + #344 wire-ins) | 5 | LOW | 2 | closed-loop |
| 15 | Dimension 1 Phase 5 (sensitivity-map regularization) | 1 | LOW | 1-2 | meta-Lagrangian |

---

## 8. 9-dimension success checklist evidence per Catalog #294

| Dim | Verdict | Evidence |
|---|---|---|
| UNIQUENESS | PASS | This blueprint is the META-strategy "how do we make cathedral autopilot smarter" memo; distinct from per-substrate / per-feature / per-bug-class symposiums |
| BEAUTY + ELEGANCE | PASS | 6 dimensions × consistent template (current state / smarter target / mechanism / cost / dependency / 6-hook / reactivation / cargo-cult / observability / horizon-class); operator-readable in 20-30 minutes |
| DISTINCTNESS | PASS | Distinct from T3 grand strategy review (which was 12 decisions; this is 6 design dimensions with phasing) |
| RIGOR | PASS | Per-dimension cargo-cult audit + Catalog #229 PV (15+ pre-flight surfaces); empirical baseline from WIRE-IN-RIGOR landing |
| OPTIMIZATION-PER-TECHNIQUE | PASS | Each dimension articulates its own canonical mechanism (Lagrangian / Bayesian / per-axis / domain / closure / tier) — no canonical-helper-borrow-by-default |
| STACK-OF-STACKS-COMPOSABILITY | PASS | Critical-path sequencing demonstrates composability; dimensions compose with explicit dependency chain |
| DETERMINISTIC-REPRODUCIBILITY | PASS | All dimensions $0 GPU; design + paired-comparison on existing posterior anchors; deterministic by construction |
| EXTREME-OPTIMIZATION-PERFORMANCE | PASS | Per-dimension cost envelope explicit; 8-16 week total; ranking by EV/$ |
| OPTIMAL-MINIMAL-CONTEST-SCORE | PARTIAL | This blueprint does NOT directly produce frontier movement; it ENABLES Dimensions 3 + 4 + 1 Phase 4 to unlock frontier-breaking via smarter ranking of the 7 asymptotic-pursuit candidates from T3 Decision 4 |

---

## 9. Observability surface per Catalog #305

| Facet | Where to inspect |
|---|---|
| Inspectable per layer | This memo + dependency graph + cost envelope + landing memo |
| Decomposable per signal | 6 dimensions × phase-by-phase decomposition |
| Diff-able across runs | Cite-chain to WIRE-IN-RIGOR landing + T3 symposium + sister deliverables |
| Queryable post-hoc | This memo + `tac.canonical_equations.query_equations()` for any equations this blueprint proposes registering |
| Cite-able | `cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md` |
| Counterfactual-able | "what if we don't pursue any of the 6 dimensions?" → cathedral autopilot remains 35% wired indefinitely; per Carmack T3 dissent the apparatus continues "producing infrastructure that protects the frontier" without producing frontier |

---

## 10. Cargo-cult audit per assumption (META-level)

| Assumption | Classification | Rationale |
|---|---|---|
| "Cathedral autopilot being 35% wired is the bottleneck for frontier movement" | HARD-EARNED-PARTIALLY-CARGO-CULTED | HARD-EARNED via WIRE-IN-RIGOR empirical findings; CARGO-CULTED because the actual bottleneck is per-substrate OPTIMAL FORM iteration (Catalog #315) + paid dispatch on contest-compliant hardware, not the apparatus itself |
| "6 dimensions of smartness compose; landing any one improves apparatus measurably" | HARD-EARNED for Dim 1 + 3 (per-axis is a known-good direction); CARGO-CULTED-PENDING-EMPIRICAL for Dim 2 + 4 + 5 + 6 | Per-axis decomposition has explicit empirical precedent (PR97 anti-pattern); other dimensions need empirical validation |
| "Phased landing (12-18 weeks) is reasonable for the 6-dimension blueprint" | HARD-EARNED-PARTIALLY-CARGO-CULTED | HARD-EARNED per T3 Decision 5 long-term timeline (4-12 weeks for unified solver maturation alone); CARGO-CULTED because parallel-dispatch could compress timeline if operator allocates 3-4 subagents simultaneously |
| "Operator-attention budget is the binding constraint" | HARD-EARNED per Daubechies Decision 12 + Carmack T3 dissent | Per CLAUDE.md "Council hierarchy: 4-tier protocol" cadence budgets + the OVER_CADENCE alerts at deliberation time |
| "$0 GPU cost is correct (no paid dispatch within the blueprint)" | HARD-EARNED | All dimensions operate on already-landed posterior anchors + ledgers + canonical equations + master-gradient outputs; no new GPU work required |

---

## 11. Predicted ΔS band per Catalog #296 + #324 (META-strategy-level)

This blueprint does NOT predict a substrate ΔS band. It predicts a CATHEDRAL-AUTOPILOT-CAPACITY-CHANGE outcome:

- Post-Dim 1 Phase 1 + Dim 3 Step 3.3: cathedral autopilot ranker mathematically grounded via canonical solver + per-axis composition; predicted ~10-30% improvement in dispatch-recommendation quality on the 7 asymptotic candidates from T3 Decision 4
- Post-Dim 4 Step 4.3: per-frame difficulty + ego-motion + per-class priors integrated; predicted additional ~5-15% dispatch-quality improvement on asymptotic candidates that depend on domain priors (Z6 / Z7 / Z8 / TT5L V2 / DreamerV3 RSSM)
- Post-Dim 1 Phase 4: Dykstra feasibility + per-axis Pareto constraints; predicted ~5-10% reduction in dispatch-recommendation false-positives (refuse infeasible candidates earlier)

**Net empirical-grounding shift**: ~35% → ~65-75% empirically grounded post-blueprint (subject to per-dimension empirical validation).

**Per Catalog #324**: NO predicted_band hardcoded; the predictions above are apparatus-capacity predictions, not substrate ΔS predictions. Validation is via paired-comparison of ranker decisions over 30-day window post-landing.

---

## 12. Canonical-vs-unique decision per layer (Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Canonical adoption | Unique implementation | Rationale |
|---|---|---|---|
| Bayesian posterior | `tac.continual_learning.posterior_update_locked` (canonical adopt) | N/A | Catalog #131 + #138 + #245 canonical 4-layer pattern |
| Canonical equations | `tac.canonical_equations.register_canonical_equation` (canonical adopt) | N/A | Catalog #344 canonical |
| Cathedral consumer Protocol | Tier A canonical adopt; Tier B sister extension (Dim 6 Step 6.1-6.3) | Per-consumer-tier classification | Per Catalog #287 v2 scope-extension precedent |
| Lagrangian formulation | TRACK A (Gaussian) + TRACK B (NumPyro) ensemble (canonical adopt) | N/A | Slot 20 T3 council 3-round verdict |
| Per-axis decomposition | NEW canonical helper `tac.score_composition.compose_score_from_axes` (Dim 3 Step 3.2) | Per-axis Lagrangian multipliers from Dim 1 Phase 4 | Score equation IS canonical per registry; multipliers are unique per substrate |
| Domain priors | NEW canonical namespace `tac.domain_priors` sister-extending `tac.ego_flow` + `tac.foveation_field` + `tac.categorical_substrate` (Dim 4 Step 4.1) | Per-substrate prior selection | Per Decision 10 canonical-helper-sister-extension |
| Sister-kinship graph | NEW canonical `tac.substrate_kinship_graph` (Dim 5 Step 5.2) | Per-kinship-edge classification | Canonical structure; per-edge weights unique |
| Auto-subscriber pattern | NEW canonical subscriber to `tac.deploy.modal.call_id_ledger` (Dim 2 Step 2.1) | N/A | Standard canonical subscriber pattern; sister to Catalog #128 / #131 |

---

## 13. Cross-reference cite-chain

This blueprint cite-chains to (per Catalog #292 maximum-signal-preservation rule):

- `wire_in_rigor_audit_meta_class_extinction_synthesis_20260520T124439Z.md` (empirical baseline)
- `wire_in_rigor_audit_per_component_dossier_20260520T124439Z.md` (per-component verdicts)
- `council_t3_grand_strategy_review_20260520T120000Z.md` (T3 12 decisions + Decision 5 long-term solver mandate)
- `strategy_staircase_synthesis_20260520T120000Z.md` (8-step falling-rule staircase)
- `comprehensive_plan_short_mid_long_term_20260520T120000Z.md` (S-1 / M-1 / L-1 / L-2 priorities)
- `strategy_dependency_wirein_graph_20260520T120000Z.md` (6-hook coverage graph)
- `gap_bug_warning_inventory_20260520T120000Z.md` (catalog # at 354 / 400; T3+T4 OVER_CADENCE)
- `docs/meta_engineering_vision.md` (commit `7b6be8a44`; META at canonical equations + per-element learned-optimal)
- `docs/ai_assisted_inverse_steganalysis_persona_council.md` (persona-council methodology)
- `feedback_mps_engineering_corrections_landed_20260520.md` (canonical equation #2 wire-in pattern)
- Future cite-chain: META-LAGRANGIAN-WIRE-1 landing memo (when it lands; this blueprint inherits Phase 2-N hooks)
- Future cite-chain: BUILD-1 NeRV trio + DREAMER-V3-FREE-PROBES landings (when they land; sister apparatus surfaces)

---

## 14. Discipline self-audit per Catalog #229 PV

Per Catalog #229 premise verification before edit: this blueprint produced AFTER reading 15+ pre-flight surfaces:

1. CLAUDE.md non-negotiables (Meta-Lagrangian / Max observability / Subagent coherence / Mission alignment / UNIQUE-AND-COMPLETE / HNeRV parity / Catalog #315)
2. WIRE-IN-RIGOR landing memo + per-component dossier + META-class extinction synthesis
3. T3 grand strategy review + staircase + comprehensive plan + dependency graph + gap inventory
4. META engineering vision (commit `7b6be8a44`)
5. AI-assisted inverse-steganalysis + persona-council methodology
6. MPS engineering corrections landing (BUILD-3 canonical-equation-#2 wire-in pattern)
7. `tools/cathedral_autopilot_autonomous_loop.py::main()` body (10 adjusters located file:line)
8. `src/tac/cathedral_consumers/` (46 packages confirmed; 44 production + 2 reference)
9. `src/tac/canonical_equations/` package (11 equations + builtins + registry)
10. `src/tac/findings_lagrangian/` + `src/tac/findings_lagrangian_pp/` packages (Phase 1-a tests landed)
11. `tac.continual_learning` + `tac.council_continual_learning` (118 posterior anchors)
12. Canonical frontier pointer state (last refreshed 2026-05-20T11:38Z)
13. Sister subagent BUILD-1 NeRV trio checkpoint state (in flight at step 4)
14. Existing probe-disambiguators (Hook 6 examples)
15. Empirical cathedral consumer contract (`per_pair_difficulty_atlas_consumer` Catalog #341 compliant; STUB-by-design)

Empirical assertions verified file:line OR via canonical helper invocation. No phantom-API citations per Catalog #287.

---

## 15. Conclusion

The cathedral autopilot today is structurally smart (auto-discovery + canonical contracts + Bayesian posteriors + canonical Provenance) and partially empirically grounded (35% via 10 adjusters + 5 ledgers). The 6-dimension blueprint above moves it toward fully empirically grounded (65-75%) over 12-18 weeks at $0 GPU.

The frontier-breaking impact depends NOT on the apparatus per se but on Dimensions 3 + 4 enabling the 7 asymptotic-pursuit candidates from T3 Decision 4 to be ranked more accurately, plus Dimension 1 Phase 2-N replacing hand-derived weights with solver-derived dual variables.

Per Carmack T3 dissent: "the apparatus has been producing infrastructure that protects the frontier; it has not been producing frontier." This blueprint enables the apparatus to participate in producing frontier — but per CLAUDE.md "Mission alignment" Consequence 4, the ultimate frontier movement comes from paid dispatch on the 7 candidates after OPTIMAL FORM iteration (Catalog #315), not from the apparatus alone.

Operator-routable next: confirm Decisions A-G above; route Dimensions 3 + 4 subagents in next 2-4 weeks; preserve $0 GPU cost envelope; honor Catalog #299 quota brake + Catalog #325 per-substrate symposium discipline + Catalog #343 frontier-pointer literal discipline throughout.
