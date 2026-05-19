---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hassabis, Welling-Teh-memorial, Boyd, MacKay]
council_quorum_met: false
council_verdict: DRAFT_PENDING_CONVOCATION
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded:
  - "DRAFT: per-substrate symposium memo prepared per Catalog #325 6-step contract; awaits operator convocation OR inner-quintet ratification"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: stack_of_stacks
deferred_substrate_retrospective_due_utc: "2026-06-19T04:36:02Z"
predicted_mission_contribution: rigor_overhead
finding_action_class: research
finding_followup_dispatch_envelope_usd: 1.80
finding_canonical_path: per_substrate_symposium_draft
---

# SGLD convergence-diagnostic per-substrate symposium DRAFT (E.8)

**Status**: DRAFT — NOT CONVENED. Awaits operator approval to convene the full T2 symposium OR ratification by inner-quintet pact (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary).

**Substrate**: `stack_of_stacks` (single-arm A1 SGLD polish canary; NOT the full multi-arm score-lowering dispatch)
**Variant**: SGLD convergence-diagnostic sweep at t_init-CAP across {0.5, 1.0, 2.0, 5.0, 10.0, 17.4}
**Dispatch envelope**: $1.80 Modal T4 (6 t_init-values * $0.30) per `substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`
**Predecessor council**: T2 Finding 2 PROCEED_WITH_REVISIONS op-routable #1 (`.omx/research/council_t2_finding_2_sgld_t_final_ceiling_cap_20260518.md`)
**Catalog #325 6-step contract compliance**: declared below

---

## Canonical 6-step contract per Catalog #325

### Step 1: Cargo-cult audit per Catalog #303

#### Cargo-cult assumption #1: "SGLD analytical formula t_final=17.4 reflects actual stack_of_stacks posterior mixing-time requirement"

**Source**: Wave 2A `8b987215a` row #8.
**Hard-earned-vs-cargo-culted**: **CARGO-CULTED** per T2 Finding 2 Assumption-Adversary verdict.
**Rationale**: Analytical formula uses canonical posterior parameters (temperature schedule + chain length to MCMC convergence within tight CI). Actual stack_of_stacks substrate posterior may have HEAVIER tails (slower convergence -> t_final TOO LOW) OR LIGHTER tails (faster convergence -> t_final correctly capped). Without empirical anchor, the 17.4x claim is unverified.
**Unwind-test plan**: This convergence-diagnostic dispatch IS the unwind test. Sweep t_init-CAP; identify plateau; compare to analytical formula.

#### Cargo-cult assumption #2: "Cost-band cap=1.0 reflects actual cost-band budget for stack_of_stacks SGLD polish"

**Source**: Wave 2A operator-set cost-band budget.
**Hard-earned-vs-cargo-culted**: **HARD-EARNED** per T2 Finding 2 Assumption-Adversary verdict.
**Rationale**: Cost-band budgets ARE operator-set + reflect actual paid-GPU constraints. The cap IS hard-earned.
**Unwind-test plan**: N/A; this is the budget axis not the formula axis.

#### Cargo-cult assumption #3: "Council 'cap=1.0' means trainer flag default `1e-4` should be `1.0`"

**Source**: Predecessor blocker audit (`.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`) Blocker B2.
**Hard-earned-vs-cargo-culted**: **CARGO-CULTED-FALSIFIED-BY-PV**.
**Rationale**: Catalog #229 premise verification confirms `--langevin-t-final` trainer default 1e-4 is the FINAL temperature of the cosine schedule (Welling-Teh canonical: temperature DECAYS from t_init to t_final during polish epochs). Council's "cap=1.0" refers to the SCHEDULE INIT temperature upper bound (`--langevin-t-init` default 0.3 in current trainer; the cost-band cap is 1.0). The "ambiguity" was a category error in the predecessor audit.
**Unwind-test plan**: This DRAFT memo's recipe documents the corrected interpretation. Operator-routable ratification required.

#### Cargo-cult assumption #4: "$0.50-1.50 envelope is sufficient for full convergence diagnostic"

**Source**: T2 Finding 2 op-routable #1.
**Hard-earned-vs-cargo-culted**: **CARGO-CULTED-PENDING-EMPIRICAL**.
**Rationale**: Council recommended t_final ∈ {0.5, 1.0, 2.0} = 3 dispatches = $0.90. This variant recipe extends to 6 t_init-values = $1.80 (above the band) for full curve coverage including the analytical formula bound t_final=17.4 (which TIMEOUTS at the 1.5h budget by design). The extended sweep is a richer empirical signal; operator-routable as separate decision.
**Unwind-test plan**: Operator chooses (a) 3-value council-minimum sweep at $0.90 OR (b) 6-value extended sweep at $1.80.

---

### Step 2: 9-dimension success checklist evidence per Catalog #294

## 9-dimension success checklist evidence

1. **UNIQUENESS**: SGLD convergence-diagnostic on stack_of_stacks substrate. The diagnostic ITSELF is canonical (Welling-Teh SGLD convergence test); the UNIQUE aspect is application to the substrate-stack_of_stacks regime where posterior structure is unknown.
2. **BEAUTY + ELEGANCE**: 6 t_init-CAP values, 100ep each, identical config except t_init. Per-t_init dispatch logs SGLD loss every 0.1*t_init; plateau identification is direct curve-fit on the 6 trajectories.
3. **DISTINCTNESS**: Distinct from production stack_of_stacks multi-arm dispatch (which is dispatch_enabled=false until full L1 blockers clear). This is the SGLD-axis Pareto diagnostic, not a score-lowering dispatch.
4. **RIGOR**: Catalog #229 premise verification corrected predecessor's category error (cap=1.0 is COST-BAND budget cap, not LANGEVIN-T-FINAL trainer-flag default). Catalog #324 `predicted_band_validation_status: pending_post_training` declared. Catalog #325 6-step contract honored.
5. **OPTIMIZATION PER TECHNIQUE**: stack_of_stacks substrate (covered by Catalog #290 canonical-vs-unique decision in substrate's own design memo); this dispatch inherits.
6. **STACK-OF-STACKS-COMPOSABILITY**: Plateau t_init identified here feeds Wave 2A row #8 retirement / formula-correction decision. Plateau IS the empirical Pareto vertex.
7. **DETERMINISTIC REPRODUCIBILITY**: Seed-pinned per trainer; identical config across 6 dispatches except `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP` env var.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: T4 100ep at 6 t_init-values fits within $1.80 total budget; t_final=17.4 timeouts intentionally at 1.5h budget bound (plateau extracted from log).
9. **OPTIMAL MINIMAL CONTEST SCORE**: This is a DIAGNOSTIC not a score-lowering dispatch. Score remains near A1 baseline 0.193. Primary output is CONVERGENCE PLATEAU which informs Wave 3+ multi-arm stack score-lowering wave.

---

### Step 3: Observability surface declaration per Catalog #305

## Observability surface

1. **Inspectable per layer**: SGLD trajectory inspectable via `tac.optimization.langevin_optimizer.LangevinOptimizer` step + loss + gradient-norm logs.
2. **Decomposable per signal**: SGLD loss decomposes into (substrate-Lagrangian, temperature, noise-term, drift-term); per-t_init dispatch logs all 4.
3. **Diff-able across runs**: Identical seed + config across 6 t_init dispatches; per-t_init diff = t_init only.
4. **Queryable post-hoc**: All 6 dispatches emit canonical Modal call_id ledger rows per Catalog #245; harvested-artifact JSON includes per-step SGLD loss trajectory.
5. **Cite-able**: Anchor = (substrate=stack_of_stacks, t_init=<T>, polish_epochs=100, commit=<git_HEAD>, call_id=<modal_call_id>); 6 anchors emit via `tac.master_gradient_anchors.append_anchor` per Catalog #245 4-layer pattern.
6. **Counterfactual-able**: Plateau identification on the 6-trajectory curve answers the canonical disambiguator question: "is t_final=17.4 needed OR is t=1.0 sufficient?"

---

### Step 4: Sextet pact deliberation (DRAFT — pending convocation)

#### Shannon LEAD (operating-within: posterior-entropy-rate via Langevin diffusion)
[DRAFT POSITION]: SGLD convergence to stationary distribution requires t_final >> mixing time. Mixing time IS the canonical Langevin diagnostic; per-t_init log of (loss, gradient-norm-decay, autocorrelation-time) directly measures mixing-time-empirical. If t=1.0 plateau is reached, IID-canonical-formula t=17.4 is over-estimating; if t=17.4 plateau is reached, formula is correct.

#### Dykstra CO-LEAD (operating-within: convergence-diagnostic IS the canonical alternating-projections feasibility)
[DRAFT POSITION]: SGLD convergence is mathematically a fixed-point iteration; empirical convergence-diagnostic IS the canonical feasibility check. The cap-vs-formula tension is resolved by empirical plateau identification, not by formula adoption. Recommend PROCEED on the 6-value extended sweep for full curve coverage.

#### Yousfi (operating-within: SGLD samples as posterior-distortion proxy)
[DRAFT POSITION]: Multi-arm stack score-lowering depends on SGLD samples being from the stationary distribution. Under-converged samples produce DOWNSTREAM artifacts based on biased samples - invisible bug class. Convergence diagnostic IS MANDATORY before downstream consumption.

#### Fridrich (operating-within: empirical-vs-analytical posterior characterization)
[DRAFT POSITION]: Wave 2A formula is correct under canonical posterior assumptions; substrate-posterior may exhibit DIFFERENT tail structure. Empirical sweep distinguishes "formula correct + cap wrong" from "formula wrong (over-estimates) + cap correct" from "formula wrong (under-estimates) + cap correct" cases.

#### Contrarian (operating-within: budget cap may be MASKING under-converged samples)
[DRAFT POSITION]: If actual posterior needs t_final=17.4 and we cap at 1.0, samples are NOT from stationary distribution. Wave-3 dispatches downstream of SGLD samples then produce DOWNSTREAM artifacts based on biased samples. The 6-value extended sweep is the canonical disambiguator; reject council's 3-value minimum as insufficient for the contrarian case (t_final=17.4 actually needed).

#### Assumption-Adversary (operating-within: HARD-EARNED-vs-CARGO-CULTED classification)
[DRAFT POSITION]:
- Cap=1.0 is HARD-EARNED (cost-band budget; operator-set)
- Formula=17.4 is HARD-EARNED-VS-CANONICAL-ASSUMPTIONS (correct given canonical parameters)
- "17.4x MISMATCH IMPLIES CAP IS WRONG" is CARGO-CULTED (the cap may be correct given actual posterior mixing time)
- Trainer flag default `1e-4` IS final temperature of cosine schedule NOT init cap: HARD-EARNED-FROM-PV (Catalog #229)

#### Hassabis (operating-within: budget-quality-tradeoff strategic-research)
[DRAFT POSITION]: SGLD t_final 17.4x current cap is a BUDGET statement, not a QUALITY statement. If t=1.0 reaches loss plateau on the actual substrate posterior, running 17x longer wastes $$ without quality improvement. The analytical formula's raw value is correct given posterior temperature; the cap is correct given budget. The right answer is empirical convergence-diagnostic, not formula vs cap. PROCEED on 6-value extended sweep.

#### Welling-Teh memorial (operating-within: SGLD canonical-author)
[DRAFT POSITION]: SGLD plateau-identification is the canonical convergence diagnostic. Cosine schedule from t_init=0.3 (current trainer default) to t_final=1e-4 over 100 polish epochs is the canonical Langevin sampler. The diagnostic should sweep `--langevin-t-init` upper bound (0.5, 1.0, 2.0, 5.0, 10.0, 17.4) at fixed `--langevin-t-final` = 1e-4 for apples-to-apples polish-completion comparison.

#### Boyd (operating-within: convex-feasibility via alternating projections)
[DRAFT POSITION]: SGLD convergence on convex feasibility region: t_init upper bound IS the temperature axis; plateau identification on (loss, gradient-norm) IS the Pareto-frontier of (convergence-quality, compute-budget). 6-value sweep maps this Pareto cleanly.

#### MacKay (operating-within: MDL + Bayesian posterior characterization)
[DRAFT POSITION]: Bayesian posterior characterization via SGLD samples requires stationarity; convergence diagnostic IS the canonical Bayesian rigor check. Per-step SGLD loss trajectory is the queryable posterior signal; plateau identification is the canonical MDL-axis convergence check.

---

### Step 5: Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL"

**Reactivation paths (priority order)**:

1. **OPERATOR FRONTIER OVERRIDE** (Catalog #300 Mission Alignment): Operator-verbatim quote in `council_override_rationale` frontmatter authorizes immediate convergence-diagnostic dispatch bypassing 14-day symposium requirement AND ratifying the blocking probe outcome `sgld_t_final_convergence_diagnostic_pending_20260518`. Cost: $1.80 (6-value) or $0.90 (3-value council-minimum). Predicted convergence-plateau identification.

2. **INNER-QUINTET RATIFICATION** (~30min): 5-of-6 inner-quintet pact ratify this DRAFT via comment in memo body + emit canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` + emit fresh probe outcome ratification via `tac.probe_outcomes_ledger.register_probe_outcome`. Cost: $0 deliberation time.

3. **FULL T2 SYMPOSIUM CONVOCATION** (~2h): All 10 listed attendees deliberate per Catalog #292 explicit-assumption-statement discipline. Output: PROCEED / PROCEED_WITH_REVISIONS / DEFER / REFUSE + posterior anchor. Cost: $0 deliberation time.

4. **DEFER UNTIL CAP-AMBIGUITY RESOLUTION** (~1h operator-attention): Operator explicitly resolves the ambiguity per the operator-routable in synthesis memo: "Council's 'cap=1.0' means SCHEDULE INIT temperature upper bound (current correct interpretation) OR something else?" The DRAFT memo's interpretation is documented but unratified.

---

### Step 6: Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status**: `pending_post_training` (declared in recipe).
**Reactivation criterion**: Post-training Tier-C density re-measurement IS the convergence diagnostic (the dispatch's primary output). Per-t_init dispatch records SGLD loss every 0.1*t_init; plateau identification is direct curve-fit. Score-band [0.190, 0.210] is bounded near A1 baseline since this is a custody-canary SGLD polish, NOT score-lowering.

---

## Sister coordination per Catalog #230 ownership map

- Sister 1 (`phase_b_mps_gap_experiment_infrastructure_build_20260518`): owns NEW `src/tac/mps_gap_experiment/` namespace; disjoint scope.
- Sister 3 (`phantom_api_backfill_wave_1_20260518`): owns ~20 EXISTING `.omx/research/*.md` memos; disjoint from this DRAFT memo.
- This DRAFT memo: only touches NEW path (`.omx/research/council_t2_sgld_convergence_symposium_DRAFT_20260519T043602Z.md`) + new variant recipe + memory entry.

---

## Catalog #229 premise verification log

- PV-0: Canonical helpers verified (tac.scorer / tac.deploy.modal.call_id_ledger / tac.probe_outcomes_ledger / tac.council_continual_learning / tac.optimization.langevin_optimizer) — all importable.
- PV-1: **PREMISE CLARIFICATION**: Predecessor audit blocker B2 framed "trainer --langevin-t-final default 1e-4 differs from council memo references (cap=1.0)". Catalog #229 PV confirms:
    - `--langevin-t-final 1e-4` is the FINAL temperature of cosine schedule (`experiments/train_substrate_stack_of_stacks.py:280`)
    - `--langevin-t-init 0.3` is the INIT temperature of cosine schedule (`experiments/train_substrate_stack_of_stacks.py:278`)
    - Council's "cap=1.0" refers to the COST-BAND budget cap (operator-set, derived from $5-15 Modal T4 envelope) NOT the SPECIFIC trainer-flag default
    - The ambiguity was a category error in the predecessor audit's framing
- PV-2: stack_of_stacks recipe `substrate_stack_of_stacks_modal_a100_dispatch.yaml` confirmed `dispatch_enabled: false` for FULL multi-arm score-lowering; separate diagnostic variant is correct approach.
- PV-3: T2 Finding 2 council memo verified at `.omx/research/council_t2_finding_2_sgld_t_final_ceiling_cap_20260518.md`.
- PV-4: Catalog #313 blocking probe outcome `sgld_t_final_convergence_diagnostic_pending_20260518` verified in `.omx/state/probe_outcomes.jsonl`.
- PV-5: Sister subagents in flight understood via `.omx/state/subagent_progress.jsonl`.

---

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-t_init plateau-trajectory anchors feed `tac.sensitivity_map.*` for SGLD-axis sensitivity ranking
2. **Pareto constraint**: ACTIVE — empirical plateau IS the Pareto-feasibility constraint for SGLD-budget-vs-quality
3. **Bit-allocator hook**: N/A — diagnostic dispatch; no per-tensor importance change
4. **Cathedral autopilot dispatch hook**: ACTIVE — plateau verdict feeds autopilot's stack_of_stacks-readiness decision
5. **Continual-learning posterior update**: ACTIVE — symposium ratification anchor + per-t_init dispatch anchor + probe outcome ratification all via canonical helpers
6. **Probe-disambiguator**: ACTIVE — this symposium IS the cap-vs-formula disambiguator per T2 Finding 2 op-routable #1

---

## Cross-references

- T2 Finding 2 council memo: `.omx/research/council_t2_finding_2_sgld_t_final_ceiling_cap_20260518.md`
- Variant recipe: `.omx/operator_authorize_recipes/substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml`
- Production recipe (sister, NOT this variant): `.omx/operator_authorize_recipes/substrate_stack_of_stacks_modal_a100_dispatch.yaml`
- SGLD trainer: `experiments/train_substrate_stack_of_stacks.py` (--langevin-t-init line 278; --langevin-t-final line 280)
- LangevinOptimizer: `src/tac/optimization/langevin_optimizer.py`
- Predecessor blocker audit: `.omx/research/3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Wave 2A row #8 analytical formula t_final=17.4: `.omx/state/8b987215a`
- Blocking probe outcome: `sgld_t_final_convergence_diagnostic_pending_20260518` in `.omx/state/probe_outcomes.jsonl`
- CLAUDE.md non-negotiables: Catalog #313 / #324 / #325 / #270 / #167 / #294 / #303 / #305 / #296 / #229 / #220 / #272 / #292 / #300 / #240


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
