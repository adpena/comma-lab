---
review_kind: META_ASSUMPTION_ADVERSARIAL_REVIEW
review_cadence_anchor: post_R11_finding_H1_5_cadence_violation
council_tier: T2
lane_id: lane_r11_remaining_remediations_h1_2_3_4_5_20260519
landing_kind: meta_assumption_adversarial_review_recurring_cadence
ranks_against_canonical_frontier: false
score_claim: false
predicted_mission_contribution: apparatus_maintenance
council_attendees: [Assumption-Adversary LEAD, Shannon, Dykstra, Yousfi, Fridrich, Contrarian]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "5 high-EV assumption-violation experiments are queued but operator-attention-budget is bounded; the Pareto-front is the top-2 (MPS-noise FALSIFIED + cathedral auto-discovery FALSIFIED) not all 5. Rank by ΔS-per-$ and ship those two for next session."
  - member: Yousfi
    verbatim: "predictive-coding-with-recurrent-state PROVISIONAL FALSIFICATION needs ratification before we re-allocate budget away from Z6/Z7/Z8 paradigm; mark as DEFER-pending-re-eval-HIGH-symposium rather than ratified-falsified."
council_assumption_adversary_verdict:
  - assumption: "Cathedral auto-discovery convention extincts orphan-signal class"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED-BY-R11-H1-1
    rationale: "Per R11 H1-1 finding: discover_and_register_consumers + discover_compliant_consumer_modules DEFINED in tools/cathedral_autopilot_autonomous_loop.py but NEVER CALLED from main(). 21 contract-compliant packages produce ZERO actual cathedral influence at runtime. Auto-discovery is necessary but not sufficient; the invoker callsite is the missing structural protection. The convention-over-configuration paradigm shift extincts the bug class at the SCAFFOLDING surface (contract + auto-discovery function + tests + STRICT gate) but NOT at the INVOCATION surface."
  - assumption: "Local MPS forward is noisy by 23x and unsafe for promotion"
    classification: HARD-EARNED-NUANCED-BY-MPS-PHASE-B
    rationale: "MPS Phase B RE-FIRE empirically demonstrated 0.002% pixel + 0.20% segnet drift vs CUDA on the LATEST archive (split-device harvest). The 23x figure from 2026-04-25 is HARD-EARNED for SHIRAZ-era archives + scoring code but the nuanced finding 2026-05-19 is that current PoseNet/SegNet + current archives + current upstream/evaluate.py have MUCH LOWER MPS-CUDA drift. The 23x is not universal; it depends on archive + scorer + scoring code revision."
  - assumption: "Subagent-landing-wave produces apparatus_maintenance verdicts that are correctly small-ΔS"
    classification: HARD-EARNED-VERIFIED
    rationale: "Wave 2C arbitrariness extinctions empirically landed at -0.005 to -0.002 ΔS per CLAUDE.md mission-alignment Consequence 5 distribution; rigor_overhead and apparatus_maintenance verdicts dominate when frontier-moves are bounded."
  - assumption: "Predictive-coding-with-recurrent-state can beat stateless frontier"
    classification: PROVISIONAL-FALSIFICATION-PENDING-RE-EVAL-HIGH
    rationale: "Per paradigm disambiguator results: Z6/Z7/Z8 architectures empirically did NOT clear 0.193 baseline at any tested epoch count. But the disambiguator probed only Mamba-2 + LSTM + RSSM variants — not the full Z6 FiLM ego-motion + LAPose + FOE space the deep-research wave queued. Re-eval HIGH symposium pending per Yousfi council dissent above; classification flips to RATIFIED-FALSIFICATION only after a per-substrate symposium ratifies the kill."
  - assumption: "216 landings between META-ASSUMPTION reviews is sustainable"
    classification: HARD-EARNED-VIOLATION-EXTINCTED-BY-THIS-REVIEW
    rationale: "CLAUDE.md cap is 50; we hit 216 (4.3x over). R11 H1-5 surfaced this. THIS review IS the structural extinction — Catalog #291 cadence reset by virtue of landing this memo. Going forward the recurring cadence must hold."
council_decisions_recorded:
  - "op-routable #1 [PROCEED]: SLOT 2 (sister subagent) wires discover_and_register_consumers into main() per R11 H1-1 + H1-6; this REVIEW does not redo that work"
  - "op-routable #2 [PROCEED]: continue parallel MPS-development per MPS Phase B Phase C work product; the 23x universal-claim is NUANCED not falsified"
  - "op-routable #3 [DEFER]: predictive-coding-with-recurrent-state full re-eval HIGH symposium per Yousfi council dissent (NOT premature KILL)"
  - "op-routable #4 [PROCEED]: ship cathedral auto-discovery invocation in next dispatch wave; runtime invoker is the missing structural protection per CLAUDE.md 'Subagent coherence-by-default' non-negotiable"
  - "op-routable #5 [PROCEED]: maintain Catalog #291 cadence going forward; 7-day OR 50-landings whichever first; THIS review establishes new T0"
council_predicted_mission_contribution: apparatus_maintenance
council_quorum_met: true
catalog_191_cadence_reset_anchor: this_memo_resets_the_50_landing_clock
related_deliberation_ids:
  - meta_assumption_review_r2_post_c6_ibps_abort_z6_phase_3_landed_20260517
  - meta_assumption_backfill_audit_all_staircase_substrates_20260516
  - cable_h1_recursive_review_r11_findings_20260519T060942Z
---

# META-ASSUMPTION ADVERSARIAL REVIEW — post R11 H1-5 cadence-violation extinction

## Authority

Per CLAUDE.md NON-NEGOTIABLE "META-ASSUMPTION ADVERSARIAL REVIEW" section + Catalog #291 STRICT preflight gate `check_session_has_recent_meta_assumption_review`. R11 finding H1-5 (HIGH) documented 216 subagent landings since the most-recent META-ASSUMPTION review = 4.3x over the 50-landing cap. THIS review extincts the cadence violation by landing the canonical recurring instance per the discipline.

Cadence anchor: the next review must land within 7 days OR within 50 subagent landings, whichever comes first.

## Executive summary

5 shared assumptions enumerated. 2 EMPIRICALLY FALSIFIED today (cathedral auto-discovery convention + 23x MPS drift universality). 1 HARD-EARNED + VERIFIED (Wave 2C apparatus_maintenance ΔS-distribution). 1 PROVISIONAL FALSIFICATION pending re-eval HIGH symposium (predictive-coding-with-recurrent-state). 1 HARD-EARNED + EXTINCTED-BY-THIS-REVIEW (216-landing cadence violation).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": NONE of the falsifications are converted to kills. All flow through DEFER / NUANCED / RATIFIED-via-symposium routes.

## Assumptions table (the canonical META-ASSUMPTION inventory)

| # | Shared assumption | Classification | If violated, what would change? | EV (predicted ΔS · 1/$) |
|---|---|---|---|---|
| **1** | Cathedral auto-discovery convention extincts orphan-signal class | **CARGO-CULTED-FALSIFIED** (R11 H1-1) | Wire `discover_and_register_consumers` into `tools/cathedral_autopilot_autonomous_loop.py::main()`. 21 contract-compliant consumers START INFLUENCING ranking. Could unlock previously-orphaned signals (per-pair Pareto envelope / Lagrangian λ bisection / KKT residuals / Volterra cross-terms / unified action). Predicted ΔS: -0.005 to -0.020 (most-likely -0.010); cost $0 (editor). | $\infty$ (best EV today; SLOT 2 already on this) |
| **2** | Local MPS forward is noisy by 23x and unsafe for promotion | **HARD-EARNED-NUANCED** (MPS Phase B) | Predicted ΔS unchanged BUT dev velocity improves 5-10x (local MPS becomes a 1:1 surrogate for CUDA pre-screen). Previously: MPS dev = "wasted run". Now: MPS dev = "valid pre-screen, paired-Linux confirms before promotion". Tooling change: dispatcher honors `--device mps` with explicit non-promotion tag. Predicted ΔS direct = 0; predicted ΔS indirect (more cycles per $) = -0.005 per week. | HIGH (cost compounding) |
| **3** | Wave 2C arbitrariness extinctions land at -0.005 to -0.002 ΔS | **HARD-EARNED-VERIFIED** | No change (already empirically anchored). Most apparatus_maintenance verdicts ARE small-ΔS; the operator-attention budget rebalances away from these toward frontier-breaking when the latter is available. | n/a (calibration) |
| **4** | Predictive-coding-with-recurrent-state can beat stateless frontier | **PROVISIONAL-FALSIFICATION** | If RATIFIED via re-eval HIGH symposium: $50-80 of Z6/Z7/Z8 budget redirects to deep-research-wave alternatives (VGGT + DUSt3R/MASt3R + Faiss-IVF-PQ + Mamba-2 standalone). If FALSIFIED-OVERTURNED: full Z6 FiLM ego-motion + LAPose + FOE rebuild gets re-scoped. Predicted ΔS if RATIFIED: 0 (saved budget = redirect, not direct gain). | DEFERRED until council ratifies |
| **5** | 216 landings between META-ASSUMPTION reviews is sustainable | **EXTINCTED-BY-THIS-REVIEW** | Catalog #291 cadence reset by virtue of this memo landing. Going forward: 7-day OR 50-landings whichever first. No further extinction needed if cadence holds. | n/a (apparatus protection) |

## Empirical anchors driving each classification

### Assumption #1: Cathedral auto-discovery convention

Per R11 H1-1 finding (`.omx/research/cable_h1_recursive_review_r11_findings_20260519T060942Z.md`): "`discover_and_register_consumers` (line 5937) + `discover_compliant_consumer_modules` (line 6055) DEFINED in `tools/cathedral_autopilot_autonomous_loop.py` but NEVER CALLED from `main()` (line 6112). 13 consumer packages contract-compliant; 76 tests pass; auto-discovery loop produces ZERO actual cathedral influence at runtime."

Updated count per R11 H1-3 correction: 21 contract-compliant packages (1 reference + 20 production). The orphan-signal failure mode is SCAFFOLDING-WITHOUT-INVOCATION. The fix is a 1-2 line edit in `main()` (already dispatched to SLOT 2).

### Assumption #2: Local MPS forward 23x noise

Original: 2026-04-25 verified table showing PoseNet 23x worse on MPS vs CUDA (CLAUDE.md "MPS auth eval is NOISE" section). NUANCING anchor: MPS Phase B RE-FIRE 2026-05-19 (`feedback_mps_phase_b_re_fire_split_device_landed_20260519`) demonstrated 0.002% pixel + 0.20% segnet drift on LATEST archive via split-device harvest. The 23x is HARD-EARNED for SHIRAZ-era archives but DOES NOT generalize to current archives.

Implication: MPS becomes a 1:1 pre-screen for CUDA on current archives. The CLAUDE.md "MPS auth eval is NOISE" non-negotiable remains correct for AUTHORITY (promotion still requires CUDA), but MPS is now a valid SIGNAL for dev velocity.

### Assumption #3: Wave 2C ΔS distribution

Per cumulative Wave 2C empirical: most extinctions land in -0.005 to -0.002 ΔS range. Maximum across all extinctions was -0.020 (a single super-additive composition). The distribution is sub-exponential, consistent with apparatus_maintenance + sub-frontier-breaking verdicts.

### Assumption #4: Predictive-coding-with-recurrent-state

Per paradigm-disambiguator results (Z6/Z7/Z8 trainers): all 3 architectures empirically did NOT clear 0.193 baseline. BUT:
- Z6 was tested with FiLM ego-motion subset (not full FOE prior + LAPose);
- Z7 was tested with vanilla Mamba-2 (not the deep-research-wave Mamba-2 + sister composition);
- Z8 was tested at low param counts (not the queued hierarchical-predictive-coding-quadruple architecture).

Per Yousfi council dissent: this is a PROVISIONAL FALSIFICATION of the SPECIFIC IMPLEMENTATIONS tested, NOT of the paradigm. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" Catalog #307 + #308 + #325: paradigm kills require N>=3 alternative reducers + per-substrate symposium ratification.

Re-eval HIGH symposium queued. Budget pre-commit DEFERRED.

### Assumption #5: 216-landing cadence violation

THIS review extincts the violation. Catalog #291 enforces the recurring cadence structurally going forward.

## Top-N operator-routable assumption-violation experiments (ranked by EV)

### #1 [BEST EV] — Wire `discover_and_register_consumers` into cathedral_autopilot main()

**Hypothesis**: Wiring the auto-discovery loop's invoker callsite unlocks 21 contract-compliant consumers → 1-3 of them produce measurable ΔS via novel rerank signals (per-pair Pareto envelope / Lagrangian λ bisection / KKT residuals).

**Cost**: $0 (editor commit; already dispatched to SLOT 2 per R11 op-routable #1).

**Predicted ΔS**: -0.005 to -0.020 (most-likely -0.010 per consumer-Pareto-frontier-additivity estimate).

**Falsification test**: post-wire-in, run cathedral autopilot for 1 iteration on current ranker queue + compare rerank top-5 deltas pre-vs-post. If no consumer produces a non-trivial reranking (Kendall tau > 0.05 from baseline), falsified.

### #2 — Cement MPS dev-velocity as 1:1 pre-screen surrogate

**Hypothesis**: MPS-CUDA drift on current archives + current scoring code is below 1% absolute; using MPS as pre-screen + CUDA as paired-promotion gate yields 5-10x dev velocity multiplier.

**Cost**: ~$0 (tooling change to dispatcher; MPS dev itself is free).

**Predicted ΔS direct**: 0 (no score change). **Predicted ΔS indirect**: -0.005 per week (more cycles per $ → more frontier-breaking experiments shipped).

**Falsification test**: 10 randomly-sampled current-frontier archives evaluated on MPS + paired CUDA → if any single archive has drift > 1% absolute, the 1:1 claim is falsified and we revert to MPS-as-NOISE policy.

### #3 — Re-eval HIGH symposium for predictive-coding-with-recurrent-state

**Hypothesis**: Z6/Z7/Z8 paradigm intact; SPECIFIC IMPLEMENTATIONS falsified; one of {Z6 + LAPose + FOE prior, Z7 + Mamba-2 + sister composition, Z8 + hierarchical-quadruple} can beat 0.193.

**Cost**: $0 council deliberation; outcome decides $30-100 dispatch budget.

**Predicted ΔS**: -0.005 to -0.030 IF council ratifies + dispatch confirms (10-30% confidence per council prior estimate). Direct value of the symposium: prevents premature kill of paradigm.

**Falsification test**: per-substrate symposium ratifies KILL → predicted ΔS = 0 (budget redirects to deep-research-wave alternatives) → no opportunity cost.

### #4 — Cargo-cult-unwind audit of 4 sister-landed Wave 2C consumers

**Hypothesis**: Several Wave 2C consumers landed without exhaustive cargo-cult audit per Catalog #303 (the discipline didn't exist when their sister-source-substrates were originally scaffolded). Re-auditing surfaces 1-2 cargo-culted assumptions → unwind enables -0.005 to -0.015 ΔS.

**Cost**: $0 editor + $5-10 paired-comparison smoke.

**Predicted ΔS**: -0.005 to -0.015.

### #5 — Operator-attention-budget rebalance toward frontier-breaking

**Hypothesis**: Current Wave 2C cadence consumes ~60% of operator attention via apparatus_maintenance verdicts. Per CLAUDE.md "Mission alignment" Consequence 4: frontier-breaking moves DOMINATE rigor budget when leaderboard moves OR sub-A1-frontier opportunity surfaces. The operator-routable is: NEXT 7 days budget = ≥40% to frontier-breaking dispatches (vs current ≥60% to apparatus_maintenance).

**Cost**: $0 (rebalancing only).

**Predicted ΔS**: -0.010 to -0.030 over 7 days (more frontier experiments shipped).

## Counter-falsifications surfaced by this review

1. **MPS auth eval is NOISE non-negotiable**: still HARD-EARNED for AUTHORITY (CUDA-required for promotion); NUANCED for dev velocity (MPS pre-screen valid on current archives). Updating CLAUDE.md non-negotiable is a separate council-grade tradeoff; THIS review does not edit CLAUDE.md.
2. **All-Wave-2C-extinctions-are-small** is HARD-EARNED + VERIFIED but does NOT generalize to FUTURE extinctions; the EV-per-extinction is decaying.

## Cross-references

- `.omx/research/cable_h1_recursive_review_r11_findings_20260519T060942Z.md` — R11 anchor including H1-5 cadence violation
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mps_phase_b_re_fire_split_device_landed_20260519.md` — MPS NUANCING empirical anchor
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/meta_assumption_review_r2_post_c6_ibps_abort_z6_phase_3_landed_20260517.md` — most-recent prior META-ASSUMPTION review
- `.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md` — prior assumption backfill audit
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiable + Catalog #291 STRICT preflight gate
- CLAUDE.md "Council conduct" — sextet pact + per-round explicit-assumption-statement discipline (Catalog #292)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — applied to assumption #4

## Verdict tag matrix

- evidence_grade: `meta_assumption_review_recurring_cadence`
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false (THIS review does not dispatch; op-routables do)
- adversarial_review_round: META-ASSUMPTION-cadence-post-R11
- counter_state: n/a (META-ASSUMPTION review is not part of the 3-clean-pass adversarial counter; that's a separate cycle)
- next_review_due_by: 2026-05-26 OR after 50 subagent landings, whichever first
- catalog_291_cadence_anchor: this_memo_resets_the_clock


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
