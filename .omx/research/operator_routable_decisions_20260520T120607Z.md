# Operator-Routable Decisions — Task Triage — 2026-05-20T12:06:07Z

> Companion to `.omx/research/task_triage_inventory_20260520T120607Z.md`.
> Lists 4 OPERATOR-ROUTABLE tasks from the canonical-ledger active set.
> **No subagent action; operator decides each.**

## Routing rationale (per task brief Rule 5)

All 4 tasks below qualify because:
- Operator decision blocks all paths (paid-GPU authorization for tasks 1-3; HF Jobs billing for task 4)
- Cross-cutting concerns with T3 strategic synthesis Decision 3+4 (per-substrate OPTIMAL FORM iteration + asymptotic-pursuit ranking)
- Per CLAUDE.md "Executing actions with care": hard-to-reverse spend that should NOT be auto-fired by subagent

## Decision summary table

| # | Task | Block | Recommendation | Cost | Wall-clock |
|---|------|-------|---|------|---|
| 1 | C6.1 lane_17_imp LTH ($10-15 Vast.ai 4090) | T3 Decision 3+6 require OPTIMAL FORM iteration before paid spend | **DEFER 30 days** + add to Catalog #325 per-substrate symposium queue | $0 deferred | 30d window |
| 2 | C6.3 PR106 #05+#06 REFORMULATED ($10 Modal A10G) | T3 Decision 3+6 cadence | **DEFER 30 days** + per-substrate symposium queue | $0 deferred | 30d window |
| 3 | C6.5 mae_v + saug ($10-35 Vast.ai 4090) | T3 Decision 3+6 cadence | **DEFER 30 days** + per-substrate symposium queue | $0 deferred | 30d window |
| 4 | HF Jobs Catalog #523 L2 Hinton-distilled SegNet surrogate | HF Jobs 402 Payment Required | **OPERATOR DECISION**: route to Modal/Vast.ai equivalent OR reload HF Jobs prepaid | $20-40 either path | After billing decision |

---

## Decision 1 — C6.1 lane_17_imp LTH reactivation

**What's blocking**: Per T3 strategic synthesis (`council_t3_grand_strategy_review_20260520T120000Z.md`) Decisions 3 + 6:
- Decision 3: every PROCEED_WITH_REVISIONS verdict MUST iterate to PROCEED-unconditional BEFORE paid dispatch fires (Catalog #315)
- Decision 6: net new spend capped per consolidation discipline (Catalog #299)

C6.1 lane_17_imp LTH (Lottery Ticket Hypothesis reactivation) currently has no per-substrate symposium PROCEED-unconditional anchor.

**What decision is needed**: do we (a) defer 30d + add to Catalog #325 per-substrate symposium queue, OR (b) operator-frontier-override per Catalog #300 §"Mission alignment" Consequence 1 if operator declares LTH a frontier-breaking priority?

**Recommended option (a) DEFER + symposium queue**: per T3 strategic synthesis dissent (Yousfi + Carmack + Hotz + PR95Author + Contrarian): apparatus is in maintainer-engagement-await mode for PR #110; paid-GPU windows are NOT the bottleneck right now; OPTIMAL FORM iteration is the demand-side constraint.

**Alternative option (b) operator-frontier-override**: requires verbatim operator quote per Catalog #300 + recipe declares `operator_override_rationale` + `operator_override_memo`.

**Cost / consequence per option**:
- (a) $0 deferred; protects T3 cadence budget; preserves operator-attention for higher-EV asymptotic candidates
- (b) $10-15 immediate spend; bypasses cadence discipline; risk per dissent that paid-GPU windows are not the bottleneck

**Operator routing instruction**: respond YES/NO to "DEFER C6.1 lane_17_imp LTH 30d + queue per-substrate symposium per T3 Decision 3+6"

---

## Decision 2 — C6.3 PR106 #05+#06 REFORMULATED paired smoke

**What's blocking**: same as Decision 1; PR106 family is HNeRV-class (CUDA frontier 0.205330 per Ballé position in T3 symposium); reformulation candidate but no per-substrate OPTIMAL FORM symposium PROCEED-unconditional anchor.

**What decision is needed**: same as Decision 1 — DEFER + symposium OR operator-frontier-override.

**Recommended option (a) DEFER + symposium queue**: PR106 family is class-saturated per Catalog #219 MDL density gate (98%+ density on existing PR106 archives); within-family bolt-on space is saturated per Selfcomp/Ballé position; ASYMPTOTIC-pursuit candidates (DreamerV3 RSSM B2 + Z7-Mamba-2 per T3 Decision 4) are higher-EV.

**Alternative option (b) operator-frontier-override**: same Catalog #300 override mechanism.

**Cost / consequence per option**:
- (a) $0 deferred; preserves T3 cadence + asymptotic-pursuit budget
- (b) $10 Modal A10G; risk it lands at PR106 plateau (no class-shift)

**Operator routing instruction**: respond YES/NO to "DEFER C6.3 PR106 #05+#06 30d + queue per-substrate symposium"

---

## Decision 3 — C6.5 mae_v + saug operational-fix

**What's blocking**: same as Decisions 1+2.

**What decision is needed**: same triplet — DEFER OR override.

**Recommended option (a) DEFER + symposium**: mae_v + saug is operational-fix scope (not class-shift); EV/$ ratio inferior to asymptotic-pursuit ranking in T3 Decision 4.

**Cost / consequence per option**:
- (a) $0 deferred
- (b) $10-35 spend; HIGH variance in outcome ($35 ceiling reflects training-time uncertainty)

**Operator routing instruction**: respond YES/NO to "DEFER C6.5 mae_v + saug 30d + queue per-substrate symposium"

---

## Decision 4 — HF Jobs Catalog #523 L2 Hinton-distilled SegNet surrogate

**What's blocking**: HF Jobs prepaid credit balance hit `402 Payment Required` before job_id assignment. Dispatcher is engineered + pre-dispatch-clean per the BUILD_1 blocker memo; ONLY blocker is account-side billing.

**What decision is needed**: 3 options:
- (a) Operator reloads HF Jobs prepaid balance ($20-40 estimated for Phase 1) → continue HF Jobs dispatch
- (b) Re-route to Modal/Vast.ai equivalent ($20-40 estimated; same trainer code; would require small dispatcher refactor)
- (c) DEFER (wait until next paid-GPU window naturally arrives)

**Recommended option (b) re-route to Modal A10G** because:
- Modal billing is healthy (today's `fc-01KS21XSVGM2KJ5ET0ET3YCCFN` dispatched + harvested successfully)
- Modal call_id ledger (Catalog #245) + silent-no-spawn fix (Catalog #339) + canonical helper auto-discovery (Catalog #335) all wire-in for Modal; HF Jobs surface is sister-but-distinct
- Catalog #523 L2 Hinton-distilled SegNet surrogate IS HIGH-EV per T3 Decision 4 (distillation is canonical KL discipline per CLAUDE.md; the surrogate unlocks gradient-through-SegNet at differentiable training time)

**Alternative option (a) reload HF Jobs**: same $20-40 cost; preserves the existing HF Jobs dispatcher code path; small operator burden (one billing top-up).

**Alternative option (c) DEFER**: free; but trades latency for cost-protection; may compound with T3 cadence discipline if deferred indefinitely.

**Cost / consequence per option**:
- (a) $20-40 immediate HF Jobs top-up; ~1 day dispatcher latency
- (b) $20-40 immediate Modal spend; +2-4h dispatcher refactor
- (c) $0 immediate; indefinite latency

**Operator routing instruction**: choose (a) / (b) / (c) — recommended (b).

---

## Cross-references

- T3 strategic synthesis Decisions 3+4+6 + dissent (Yousfi/Carmack/Hotz/PR95Author/Contrarian) on cadence + OPTIMAL FORM + frontier-pursuit-prioritization
- Catalog #325 per-substrate OPTIMAL FORM symposium 14-day window discipline
- Catalog #300 operator-frontier-override mechanism (Mission alignment Consequence 1)
- Catalog #245 Modal call_id ledger + Catalog #339 silent-no-spawn fix verification
- Catalog #523 (referenced in BUILD_1 task title; not yet a registered preflight gate per Catalog #299 cardinality)
- CLAUDE.md "Apples-to-apples evidence discipline" (paired CUDA+CPU custody required for any score claim)

## Discipline attestation

- Catalog #229 PV: read all blocker text + T3 symposium Decisions + Catalog #313 probe-outcomes
- Catalog #287: no phantom-API citations (every Catalog # cited exists; Catalog #523 cited in source task title only, NOT as my claim)
- Catalog #323: no score claims requiring Provenance
- Per CLAUDE.md "Executing actions with care": no subagent action proposed; operator decides each
- Per CLAUDE.md "Mission alignment" Consequence 4: frontier-breaking moves DOMINATE rigor budget — recommended DEFERs respect cadence WHILE preserving operator-frontier-override escape hatch

## Total spend if operator approves all recommendations

| Decision | Recommended | Cost |
|----------|-------------|------|
| 1 | DEFER | $0 |
| 2 | DEFER | $0 |
| 3 | DEFER | $0 |
| 4 | Re-route to Modal | $20-40 |
| **TOTAL** | | **$20-40** |

Compare to total-if-approve-all-paid-as-originally-requested: $50-100 + $20-40 HF = $70-140. **Recommended path saves $50-100 + preserves T3 cadence + protects operator-attention budget for higher-EV asymptotic candidates per T3 Decision 4.**
