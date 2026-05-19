# Integrated battle plan — priority queue + DAG + cables + staircases + gates

**Date:** 2026-05-19 (UTC)
**Authority:** Operator 2026-05-19 "don't we have a big backlog ... integrated into a battle plan and priority queue and parallel cable and staircase and graph and properly gated"
**Scope:** comprehensive ENUMERATION of ALL backlog items + DAG dependencies + parallel cable bundling + staircase sequencing + cost/symposium/Catalog gates per item
**Goal:** turn the backlog from a stack of open items into an executable DAG with explicit gates that codex + 5 subagent slots + main-context Claude can drain continuously

## Canonical-vs-unique decision per layer

| Layer | Decision |
|---|---|
| Plan structure | ADOPT_CANONICAL (cables / staircases / DAG / gates per operator's framing) |
| Slot calibration | UNIQUE — promote to 5-slot saturation per operator's pushback today (was 4-cap; now 5) |
| Codex coordination | ADOPT_CANONICAL (routing-directive pickup pattern) |
| Operator-frontier-override | ADOPT_CANONICAL (Catalog #300 verbatim quote already captured 2026-05-19) |

## 9-dimension success checklist evidence

- UNIQUENESS: integrates ~50+ pending items across 9 cables not previously bundled
- BEAUTY+ELEGANCE: 9-cable structure + per-cable DAG fits on one page
- DISTINCTNESS: each cable has different scope / theme / gate-class
- RIGOR: every item carries cost + dependency + gate + sister-coordination ownership
- OPTIMIZATION-PER-TECHNIQUE: parallel cables = max throughput; staircases = correct sequencing
- STACK-OF-STACKS-COMPOSABILITY: cables explicitly compose (e.g. B+D feeds C verdicts feed B-next)
- DETERMINISTIC-REPRODUCIBILITY: every item has commit-hash + memo-path provenance
- EXTREME-OPTIMIZATION-PERFORMANCE: explicit per-cable wall-clock + cost
- OPTIMAL-MINIMAL-CONTEST-SCORE: highest-EV items (MPS / Z7-Mamba / Cargo-cult resurrection / DP1+PR101) on the critical path

## Observability surface

- Per-cable status updates live in `.omx/state/lane_registry.json`
- Per-item commit-hash provenance maintained
- Codex's autonomous output visible via `git log --author="codex" --oneline -20`
- Slot state visible via this Claude session's task list + `.omx/state/subagent_progress.jsonl`

## Cargo-cult audit per assumption

| Assumption | Classification |
|---|---|
| 5-slot saturation correct calibration | HARD-EARNED (operator pushback today; under-saturated state criticized) |
| Codex + 5 slots can run truly parallel without collision | HARD-EARNED (Catalog #157 commit-swap protection caught 1 race today; resolved automatically) |
| Battle plan in single memo enables operator triage | HARD-EARNED (prior 6-cluster triage worked; 9-cable extension follows same shape) |
| Cables fully disjoint at file scope | HARD-EARNED via Catalog #230 ownership map per cable below |

## Horizon class

`frontier_breaking` — the plan's critical path (Cables A+B+C+H) targets score-lowering at minimum wall-clock; Cables D+E+F+G+I are frontier-protecting + hardening parallel-disjoint.

---

## The 9 cables

### Cable A — MPS / Backend axis ($0.50 + verdict-branch follow-ons)

Score-lowering AXIS: unlocks free local compute IF MPS gap verdict positive.

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| A1 | MPS Phase B fire+harvest | IN FLIGHT slot 1 (#938) | $0.50 | none (operator-frontier-override ratified) |
| A2 | MLX exploration scaffold | GATED on A1=VIABLE_ADVISORY_ONLY or NOT_VIABLE | $0 editor | A1 verdict |
| A3 | VideoToolbox-decode + CUDA-train scaffold | GATED on A1=NOT_VIABLE | $0 editor | A1 verdict |
| A4 | ANE exploration via CoreML | parallel-capable | $0 editor | none; small-scope follow-on |

**Critical path**: A1 verdict gates A2/A3/A4 routing.

### Cable B — Score-lowering paid GPU ($3.30-$95 cascade)

Sequential phases gated on prior phase verdict.

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| B1 | E.7+E.8 combined dispatch (VQ K-sweep + SGLD convergence) | READY | $3.30-4.20 | operator-frontier-override ratified (e7_e8 memo 2026-05-19) |
| B2 | Z7-Mamba `mamba_ssm` Modal pre-flight smoke | GATED on codex E.1 recipe edit landing | $0.10 | codex Cluster E.1 |
| B3 | Z7-Mamba grad-clip + LR-warmup 9-config sweep | GATED on B2 success | $5-15 | B2 verdict |
| B4 | S4 anchor (fallback path) | GATED on B3 verdict (ANY converges → skip B4; ALL diverge → fire B4) | $5-10 | B3 verdict |
| B5 | DreamerV3-RSSM pivot | GATED on B3+B4 outcomes | $15-30 | B3/B4 verdicts |
| B6 | Cargo-cult resurrection TOP-3 (TT5L V2 / DP1+PR101 / one more) | GATED per Catalog #325 per-substrate symposium | $10-30 | per-substrate symposium ratification |

**Critical path**: B1 → B2 → B3 → {B4 OR B5} → reseed by verdict.
**Total envelope**: $38.40-$89.30 (within $100 budget; $10-62 buffer).

### Cable C — Substrate symposium queue (mostly $0 editor)

$0 editor work; each item produces operator-routable DRAFT per Catalog #325.

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| C1 | Per-substrate symposium #3 Z7 LSTM predictive coding | task #851 pending | $0 editor | none |
| C2 | #4 NSCS06 v8 Path B reformulation | task #852 pending | $0 editor | none |
| C3 | #5 TT5L foveation + LAPose | task #853 pending | $0 editor | none |
| C4 | #6 Z8 hierarchical predictive coding | task #854 pending | $0 editor | none |
| C5 | #7 DP1 deep-dive | task #855 pending | $0 editor | none |
| C6 | 5 RE-EVAL-HIGH DRAFTs from META-bug audit (#856-#862) | already-queued per audit verdict | $0 editor | none |

**Bundling**: one subagent handles 5 symposium DRAFTs sequentially (C1-C5) + cross-references C6's already-queued items.

### Cable D — Wire-in + integration completeness ($0 editor)

Sister of wiring+integration audit; targets identified producer→consumer gaps.

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| D1 | Wiring+integration audit | IN FLIGHT slot 3 (#940) | $0 | none |
| D2 | Master-gradient extension across analytical surfaces | task #887/#890 pending | $0 editor | none |
| D3 | Consumers 7-15 builder wave (per `tac.master_gradient_consumers` catalog) | task #799 pending | $0 editor | none |
| D4 | VIZ `tools/master_gradient_xray.py` + 5 plot types | task #797 pending | $0 editor | none |
| D5 | SUPER_ADDITIVE topology integration (lane_g_v3+siren) | tasks #823/#825-827 pending | $0 editor | Catalog #321-compliant verification |
| D6 | 12 alternative math frameworks for deterministic-optimizer | task #891 in-flight subagent acb41f8d | already in-flight | none |

**Bundling**: one subagent handles D2+D3+D4 batched (all master-gradient-family wire-ins).

### Cable E — HF Jobs wave (mostly $0 prep + paid GPU per HF Jobs cap)

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| E1 | HF dataset adpena/comma-video-segnet-image-level-600pairs build | task #876 pending | $0 prep | Hugging Face Pro account |
| E2 | HF Jobs implementation wave (5 insights: dataset + SegNet/PoseNet surrogates + DINOv3 + SAM2) | task #878 pending | $5-20 paid HF | E1 complete |
| E3 | L2 Hinton-distilled SegNet surrogate via HF Jobs T4 | task #875 in-flight | already in-flight | E1 complete |

**Bundling**: E2 batches multiple HF Jobs dispatches under single subagent.

### Cable F — Production hardening + OSS ($0 editor)

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| F1 | OSS v0.2.0-rc1 release tag | NOW UNBLOCKED (LICENSE drafted today commit `16d2323db`) | $0 editor | tasks #526/#553 |
| F2 | 4 stale memory entries triage (per-category map surfaced) | discovered today | $0 editor | none |
| F3 | Catalog #287 strict-flip after codex Wave 4 | GATED on codex Cluster C completion | $0 editor | codex Cluster C |
| F4 | META event-driven retroactive sweep gate landing | codex Cluster B in progress | $0 editor | codex Cluster B |
| F5 | PV-0 discipline canonical pre-flight | task #843 pending | $0 editor | none |

**Bundling**: one subagent handles F1+F2 + monitors F3/F4 codex output for ratification.

### Cable G — Codex parallel (autonomous; no slot competition)

| # | Item | Status |
|---|---|---|
| G1 | Cluster B META gate landing | codex in progress (autonomous loop) |
| G2 | Cluster C Phantom-API Waves 2-4 | codex in progress |
| G3 | Cluster E.1 Z7-Mamba recipe stale-blocker cleanup | codex in progress |
| G4 | Cluster F sigma=15 + 600-pair test | codex in progress |

No Claude slot consumption. Codex picks up new directives as I land them.

### Cable H — Adversarial review cycle ($0 editor)

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| H1 | RECURSIVE-REVIEW-R11 (post FIX-WAVE-10 / FIX-WAVE-11) | task #655 pending | $0 editor | none |
| H2 | RECURSIVE-REVIEW-R13-DEEPEST (foundational-level) | task #658 pending | $0 editor | H1 clean pass |
| H3 | 3-clean-pass cycle on full Wyner-Ziv stack | task #803 pending | $0 editor | Wyner-Ziv consumers stable |

**Bundling**: H1 + (if clean) H2 in one subagent sequenced.

### Cable I — Bug-class + META gate cleanup ($0 editor)

| # | Item | Status | Cost | Gates |
|---|---|---|---|---|
| I1 | Catalog # clarification batch | task #911 pending | $0 editor | none |
| I2 | CODEX RE-RUN 2026-05-21+ external review of Wyner-Ziv stack | task #809 pending | $0 codex | Wyner-Ziv stack stable |
| I3 | Wire-in #3 bit_allocator per-pair sensitivity | task #800 pending | $0 editor | D-cable wire-in completeness |
| I4 | Wire-in #4 cathedral autopilot per-byte sensitivity | task #801 pending | $0 editor | D-cable wire-in completeness |
| I5 | Wire-in #5 per-pair difficulty atlas → CL posterior | task #802 pending | $0 editor | D-cable wire-in completeness |

**Bundling**: I3+I4+I5 wire-in batch in one subagent (after D-cable completes).

---

## DAG dependencies (top-level)

```
                          ┌─→ A2 / A3 / A4 verdict-branch
A1 (MPS Phase B) ─────────┤
                          └─→ Cathedral autopilot ranker backend routing

G3 (codex E.1) ──→ B2 (mamba_ssm pre-flight) ──→ B3 (Z7-Mamba sweep) ──→ {B4, B5}

(E.7 symposium DRAFT, E.8 symposium DRAFT) ──→ B1 (combined dispatch)

C1-C5 (substrate symposiums) ──→ B6 (per-substrate paid dispatch when symposium ratified)

D1 (wiring audit) ──→ (remediation queue) ──→ D2+D3+D4+D5+D6 (parallel wire-in batch)
                                            ──→ I3+I4+I5 (per-byte sensitivity consumers)

E1 (HF dataset) ──→ E2 (HF Jobs wave) ──→ E3 (Hinton surrogate complete)

G2 (codex C) → 0 phantom-API violations → F3 strict-flip ──→ Catalog #287 fully strict

H1 (R11 clean) ──→ H2 (R13 DEEPEST) ──→ Catalog # ledger cleanup

F1 OSS release tag (NOW UNBLOCKED)
```

## Priority queue (top-down EV ranking)

1. **A1** MPS Phase B (in flight) — unlocks free local compute axis
2. **B1** E.7+E.8 dispatch — empirical answers to grand-council DEFER reactivation
3. **G1-G4** codex autonomous batch — drains 4 hardening clusters in parallel
4. **D1** wiring+integration audit (in flight) — preventive signal-loss maintenance
5. **C1-C5** substrate symposium DRAFTs — unblocks per-substrate per-Catalog-#325 future dispatches
6. **D2+D3+D4** master-gradient family wire-in — unblocks I3+I4+I5 + cathedral autopilot ranker improvements
7. **B2** mamba_ssm pre-flight — gates B3 dispatch chain
8. **F1** OSS v0.2.0-rc1 release tag — now unblocked
9. **H1** R11 adversarial review — discipline cadence
10. **C6** 5 RE-EVAL-HIGH from META-bug audit — symposium DRAFTs for resurrection candidates
11. **F2** stale memory entries triage — bookkeeping
12. **E1** HF dataset build — gates E2/E3
13. **B3-B6** Z7-Mamba cascade + cargo-cult resurrection — gated on prior verdicts
14. **I1-I5** META cleanup batch — gated on prior cables
15. **A2-A4** MPS verdict-branch follow-ons — gated on A1 verdict

## Slot allocation strategy

**5-slot saturation** (operator pushback today corrected from 4-cap):

| Slot | Current | Next-up upon completion |
|---|---|---|
| 1 | A1 MPS Phase B fire+harvest (#938) | A2/A3/A4 verdict-branch follow-on |
| 2 | B1 E.7+E.8 combined dispatch | B2 mamba_ssm pre-flight (after codex E.1 lands) |
| 3 | D1 wiring+integration audit (#940) | D2+D3+D4 master-gradient wire-in batch |
| 4 | C1-C5 substrate symposium DRAFT batch | C6 RE-EVAL-HIGH DRAFTs OR H1 R11 adversarial review |
| 5 | E1 HF dataset build OR F1+F2 OSS+stale-memo bundle | E2 HF Jobs wave OR I3+I4+I5 wire-in batch |

**External**: codex working autonomously on Cluster B+C+E.1+F (no slot competition).

## Per-cable expected wall-clock

| Cable | Wall-clock estimate | Cost |
|---|---|---|
| A | A1 ~22 min; A2/A3/A4 each ~4h editor (if fired) | $0.50 + $0 |
| B | B1 ~30 min parallel; B2 ~30 min; B3 ~2-3h Modal; B4/B5 ~2-3h each | $38.40-89.30 |
| C | C1-C5 batch ~5h editor in one subagent | $0 |
| D | D1 in-flight ~7h; D2+D3+D4 batch ~5h editor | $0 |
| E | E1 ~2h prep; E2 ~6h paid HF; E3 in-flight | $5-20 |
| F | F1+F2 ~2h editor; F3/F4 codex-gated | $0 |
| G | codex autonomous ~4-6h total | $0 codex |
| H | H1 ~5h editor; H2 conditional ~5h | $0 |
| I | I3+I4+I5 batch ~5h editor (gated on D) | $0 |

**Total session-wave wall-clock**: ~12-18h of parallel work compresses into ~4-6h wall-clock via 5-slot saturation + codex parallel.

**Total session-wave cost**: $38.40-$109.30 depending on B-cascade depth + E2 HF Jobs spend. Within $100-150 envelope.

## Gates summary

**Operator gates** (need explicit greenlight):
- Per-substrate symposium ratification (full T2 convocation vs inner-quintet pact DRAFT vs operator-frontier-override) — operator-frontier-override 2026-05-19 covers ALL cables today
- B6 cargo-cult resurrection DISPATCH (per-substrate symposium gated per Catalog #325)
- E2 HF Jobs wave dispatch ($5-20 paid)
- F1 OSS v0.2.0-rc1 tag PUSH (operator must approve release)

**Automated gates** (fire structurally):
- Catalog #229 PV (Claude-side; mandatory pre-edit)
- Catalog #243 local pre-deploy harness (auto via operator-authorize.py)
- Catalog #270 dispatch optimization protocol (auto)
- Catalog #271 codex pre-dispatch review (auto if cost >$1)
- Catalog #313 probe outcomes consultation (auto)
- Catalog #324 predicted_band_validation_status (recipe schema check)
- Catalog #325 per-substrate symposium 14-day window check (auto)
- Catalog #199 paired-env discipline for operator-authorize bypass

**Sister-coordination gates** (Catalog #230 ownership map):
- Per-subagent disjoint scope verified at dispatch time
- `.omx/state/subagent_progress.jsonl` checked at subagent start
- Catalog #157 commit-swap protection auto-fires on collision (caught 1 race today; resolved automatically)

## Continual learning + production OSS hardening parallel track

Honors operator's standing directive: hardening runs IN PARALLEL with score-lowering; no track blocks the other.

| Hardening item | Cable | Status |
|---|---|---|
| Phantom-API Wave 2-4 → strict-flip | G2 + F3 | codex + main-context monitoring |
| LICENSE / THIRD_PARTY_NOTICES (OSS unblock) | F1 | LANDED today commit `16d2323db` |
| Memory rotation per-category map | F2 (sister) | LANDED today commit `16d2323db` |
| Wiring+integration audit | D1 | IN FLIGHT slot 3 |
| META event-driven retroactive sweep | G1 + F4 | codex in progress |
| Atom canonical META adoption (wider) | C+D cables | gradual |
| 5 RE-EVAL-HIGH from META-bug audit | C6 | per-symposium pipeline |
| 3-axis self-audit elevation (time/event/gate) | H1+H2 | follow-on cadence |

## Cross-references

- Operator-frontier-override: `.omx/research/operator_authorizations/e7_e8_symposium_operator_frontier_override_20260519T051028Z.md`
- Codex routing directive: `.omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md`
- Prior battle plans (this memo supersedes their open-item lists):
  - `extinction_dispatch_queue_complete_52_row_coverage_20260518.md`
  - `meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z.md`
  - `z7_mamba_2_multi_week_path_forward_20260518.md`
  - `grand_council_findings_deliberation_wave_aggregate_dispatch_plan_20260518.md`
- Persistent codex /goal: `codex_persistent_goal_v2_5_2_compressed_with_inbox_20260518.md`

## 6-hook wire-in declaration (per Catalog #125)

1. Sensitivity-map: N/A (battle plan is meta-routing)
2. Pareto constraint: ACTIVE — items priority-ranked by EV / cost / wall-clock Pareto
3. Bit-allocator: N/A
4. Cathedral autopilot dispatch: ACTIVE — this memo IS the canonical dispatch plan
5. Continual-learning posterior: ACTIVE — each cable item's outcome reseeds posterior
6. Probe-disambiguator: ACTIVE — verdict-branch follow-ons (A2/A3/A4 / B4/B5) ARE probe-disambiguator outcomes

## Predicted mission contribution

`frontier_breaking` — the critical path (A+B+C+H cables) directly targets score-lowering at minimum wall-clock; the parallel hardening track (D+E+F+G+I) is `frontier_protecting`.

— Main-Claude 2026-05-19 (battle plan per operator pushback against under-saturation)


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
