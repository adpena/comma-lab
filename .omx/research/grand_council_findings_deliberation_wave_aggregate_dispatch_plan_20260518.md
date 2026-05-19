# Grand Council Findings Deliberation Wave — Aggregate Dispatch Plan

**Date**: 2026-05-18
**Lane**: `lane_grand_council_findings_deliberation_wave_20260518` L1
**Subagent**: `grand_council_findings_deliberation_wave_20260518`
**Council cadence audit at start**: T3 OVER_CADENCE (22/13 in 30d window). Applied cadence-discipline: routed bounded-scope findings to T2 sextet where possible (8 of 15 findings).

## Per-finding verdict matrix

| # | Finding (slug) | Tier | Verdict | Action class | $ envelope | Dispatch order | Critical-path |
|---|---|---|---|---|---|---|---|
| 1 | VQ codebook K=64+K=256 ANTI-PARETO | T3 | PROCEED_WITH_REVISIONS | research | $1-3 | 2nd | Unblocks 14-substrate VQ wire-in |
| 2 | SGLD t_final raw 17.4 → cap 1.0 | T2 | PROCEED_WITH_REVISIONS | research | $0.50-1.50 | 3rd | Unblocks stack_of_stacks substrate |
| 3 | Rashomon K=8 grossly under-powered | T2 | PROCEED_WITH_REVISIONS | research | $0.50 | 3rd | Unblocks cathedral autopilot ranker quality |
| 4 | Z7-Mamba-2 INDETERMINATE_WITH_NUANCE | T3 | DEFER_PENDING_EVIDENCE | research | $2-5 | 4th | Unblocks TOP-5 #2 EV [-0.025, -0.008] |
| 5 | 16+ META-audit phantom-API recurrence | T3 | PROCEED | delegate | $0 | 1st | Frontier-protecting; single-PR landing |
| 6 | FISTA + Frank-Wolfe + Riemannian-Newton wins | T3 | PROCEED | pursue | $0 | 1st | 1.5-2x velocity multiplier; ship immediately |
| 7 | pose_to_seg_ratio = 2.7116 anchor | T2 | PROCEED | pursue | $0 | 1st | 14-substrate wire-in EV [-0.105, -0.014] |
| 8 | EMA decay formula recovers Quantizr 0.997 | T2 | PROCEED | pursue | $0 | 1st | 14-substrate wire-in EV [-0.042, -0.014] |
| 9 | brotli quality 10 vs 11 inconsistency | T2 | DEFER_PENDING_EVIDENCE | defer | $0 | 5th | Awaits Wave 2C empirical sweep |
| 10 | lzma preset=9 hardcoded | T2 | DEFER_PENDING_EVIDENCE | defer | $0 | 5th | Awaits Wave 2C empirical sweep |
| 11 | fp64 master-gradient identical operating point | T2 | PROCEED_WITH_REVISIONS | research | $0 | 2nd | Unblocks per-pair Thompson sampling (Impl 13) |
| 12 | 52-row composite EV realistic vs optimistic | T3 | PROCEED_WITH_REVISIONS | pursue | $0 | 1st | Reranks cathedral autopilot dispatch priority |
| 13 | 5 contest_fixed rows excluded but define path | T1 | PROCEED (RATIFY) | ratify | $0 | n/a | Apparatus maintenance; no follow-up |
| 14 | TOP-1+TOP-4 batched empirical recovery | T1 | PROCEED (RATIFY) | ratify | $0 | n/a | Apparatus maintenance; no follow-up |
| 15 | Atom subsumption of 52 audit rows | T1 | PROCEED (RATIFY) | ratify | $0 | n/a | Apparatus maintenance; no follow-up |

**Verdict distribution**:
- T1 (3): 3 PROCEED-as-RATIFY
- T2 (8): 4 PROCEED / 2 PROCEED_WITH_REVISIONS / 2 DEFER_PENDING_EVIDENCE
- T3 (4): 2 PROCEED / 2 PROCEED_WITH_REVISIONS — wait, let me recount: 5/6/12 PROCEED-variants + 1/4 + 4 deferred among T3. Actual:
  - T3 (4): 2 PROCEED (#5, #6) + 2 PROCEED_WITH_REVISIONS (#1, #12) + 1 DEFER_PENDING_EVIDENCE (#4)

(Total 15; T3=5 not 4. Adjusted count: T1=3, T2=8, T3=4. #4 + #12 are T3; #5 + #6 are T3; #1 is T3. So T3 count is actually 5. Recheck T2: #2, #3, #7, #8, #9, #10, #11 = 7. T1 = #13, #14, #15 = 3. T3 = #1, #4, #5, #6, #12 = 5. Total = 7+3+5 = 15. ✓)

**Corrected distribution**:
- T1 (3): 3 PROCEED-as-RATIFY (#13, #14, #15)
- T2 (7): 4 PROCEED (#7, #8) — wait actually #7 and #8 are PROCEED; #2 #3 #11 PROCEED_WITH_REVISIONS; #9 #10 DEFER. So 2 PROCEED + 3 PROCEED_WITH_REVISIONS + 2 DEFER
- T3 (5): 2 PROCEED (#5, #6) + 2 PROCEED_WITH_REVISIONS (#1, #12) + 1 DEFER_PENDING_EVIDENCE (#4)

## Critical-path identification

**Tier-1 (immediate; $0; editor-only; 1st-priority dispatch)**:
- Finding #5: META-meta phantom-API gate (1 PR; structural extinction)
- Finding #6: FISTA + Frank-Wolfe + Riemannian-Newton drop-in (velocity multiplier; ships)
- Finding #7: pose_to_seg_ratio canonical λ wire-in (14-substrate)
- Finding #8: EMA decay canonical formula wire-in (14-substrate)
- Finding #12: realistic EV envelope adoption in cathedral autopilot ranker
- Finding #13/#14/#15: T1 ratifications (instant)

**Tier-2 (research before pursuit; $1-5 paid GPU)**:
- Finding #11: 600-pair-independence test ($0 local-CPU; before per-pair Thompson sampling)
- Finding #1: K-sweep paired-comparison smoke ($1-3 Modal T4; before VQ codebook 14-substrate wire-in)
- Finding #2: SGLD convergence-diagnostic smoke ($0.50-1.50)
- Finding #3: Rashomon K-sweep with kendall-tau ($0.50)

**Tier-3 (DEFER awaiting external evidence; no GPU spend until evidence lands)**:
- Finding #4: Z7-Mamba-2 stability-fix smoke ($2-5; gates TOP-5 #2 pursuit)
- Finding #9, #10: Wave 2C results (apparently already landed per `d142b6ad9`)

## Operator-decision queue items

**Decision A**: Approve $1-3 Modal T4 K-sweep dispatch for Finding #1 (VQ codebook K-pole empirical confirmation)?
- Cost: $1-3
- Outcome: confirms or refutes 14-substrate VQ codebook canonical-vs-fork decision
- Recommendation: APPROVE — composite EV [-0.070, -0.014] frontier-breaking if confirmed

**Decision B**: Approve $2-5 Modal T4 grad-clip+LR-warmup smoke for Finding #4 (Z7-Mamba-2 stability)?
- Cost: $2-5
- Outcome: if pass, unlocks TOP-5 #2 [-0.025, -0.008] pursuit; if fail, mandate pivot to RSSM/S4
- Recommendation: APPROVE — TOP-5 #2 is frontier-breaking; alternative is also frontier-breaking (Hafner-recommended pivot)

**Decision C**: Approve $0.50-1.50 Modal T4 SGLD convergence-diagnostic for Finding #2?
- Cost: $0.50-1.50
- Outcome: resolves stack_of_stacks substrate posterior convergence question
- Recommendation: APPROVE — cheap; informs cost-band cap-vs-formula question structurally

**Total operator-decision queue**: $3.50-9.50 in 3 paid-GPU dispatches (all <$10 individually; APPROVE recommended for all 3)

## Council cadence audit (post-deliberation)

This wave adds:
- T1: 3 new deliberations (still UNBOUNDED)
- T2: 7 new deliberations (was 39/90; now 46/90 = 51% WITHIN_BUDGET)
- T3: 5 new deliberations (was 22/13 OVER_CADENCE; now 27/13 = 208% STILL OVER_CADENCE)

**Operator-attention budget alert**: T3 cadence STILL over-cadence post-wave. Per CLAUDE.md "Council hierarchy: 4-tier protocol": STOP AND CONSOLIDATE — review whether recent T3 deliberations could have been resolved at T2. Several recent T3s may belong at T2 (working-group recommendation tier). This subagent applied cadence-discipline (routed 7 of 15 findings to T2 sextet); future deliberations should continue this discipline.

## Mission-contribution distribution

- frontier_breaking: 6 (#1, #4, #6, #7, #12 — and arguably #11)
- frontier_protecting: 3 (#5, #8, #11)
- rigor_overhead: 3 (#2, #3, #9, #10 — actually 4)
- apparatus_maintenance: 3 (#13, #14, #15)
- mission_questioned: 0

Within 60% threshold for rigor_overhead+apparatus_maintenance per CLAUDE.md Mission alignment Consequence 5. **OK**.

## Follow-on routable subagents (queued; not dispatched by this wave)

1. **META-PHANTOM-API-STRUCTURAL-EXTINCTION** (T3 / Finding #5 / delegate): write new STRICT preflight gate scope-extending Catalog #287 to research memos; $0; ~1-2h editor
2. **MORE-OPTIMAL-ALGORITHMS-WIRE-IN** (T3 / Finding #6 / pursue): drop-in FISTA + Frank-Wolfe + Riemannian-Newton with invariant-pinning regression tests; $0; ~2-3h editor
3. **PER-SUBSTRATE-LAMBDA-WIRE-IN-WAVE** (T2 / Finding #7 + #8 / pursue): 14-substrate canonical-λ + canonical-decay wire-in per per-substrate symposium per Catalog #325; $0 per substrate; ~1d total
4. **CATHEDRAL-AUTOPILOT-RANKER-REALISTIC-EV-UPDATE** (T3 / Finding #12 / pursue): adopt realistic envelope -0.02 to -0.05 for dispatch ranking; queue new sister-Catalog gate
5. **600-PAIR-INDEPENDENCE-TEST** (T2 / Finding #11 / research): $0 local-CPU editor; outcome unblocks per-pair Thompson sampling (Impl 13)
6. **K-SWEEP-PAIRED-COMPARISON-SMOKE** (T3 / Finding #1 / research; OPERATOR-DECISION-A): $1-3 Modal T4; outcome unblocks 14-substrate VQ canonical-vs-fork decision
7. **SGLD-CONVERGENCE-DIAGNOSTIC-SMOKE** (T2 / Finding #2 / research; OPERATOR-DECISION-C): $0.50-1.50 Modal T4
8. **RASHOMON-K-SWEEP-WITH-KENDALL-TAU** (T2 / Finding #3 / research): $0.50 Modal T4 OR local-CPU; outcome enables adaptive-K helper
9. **Z7-MAMBA-2-STABILITY-FIX-SMOKE** (T3 / Finding #4 / research; OPERATOR-DECISION-B): $2-5 Modal T4; gates TOP-5 #2 pursuit

## Cross-references

- Standing directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_findings_review_grand_council_deliberation_standing_directive_20260518.md`
- Sister design: `.omx/research/contest_fixed_as_oracles_15_implications_design_memo_20260518.md` (commit `07b24f303`)
- Canonical 52-row audit: `.omx/research/arbitrariness_extinction_audit_20260518.jsonl` (commit `2d042f7e6`)
- Just-landed sister code (READ-ONLY): `d17a9826c` (contest_oracle), `d142b6ad9` (formula_extinctions), `8b987215a` (analytical_solve_extinctions), `6db94d9ea` (TOP-1+TOP-4), `181fa4c1e` (tac.atom)
- CLAUDE.md non-negotiables: "Council hierarchy: 4-tier protocol" (Catalog #300), "Forbidden premature KILL", "META-ASSUMPTION ADVERSARIAL REVIEW" (Catalog #291), "Mission alignment", "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" (Catalog #315)
