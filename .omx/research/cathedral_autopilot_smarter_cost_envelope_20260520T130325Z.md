# Cathedral Autopilot Smarter Cost Envelope — 2026-05-20T13:03:25Z

> **Deliverable C (cost envelope) of SLOT CATHEDRAL-SMARTER-DESIGN-MEMO**
> **Lane**: `lane_cathedral_autopilot_smarter_design_blueprint_20260520`
> **Cite-chain**: master memo `cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md` + dependency graph sister

---

## Headline numbers

- **Total $ GPU cost**: $0 across all 6 dimensions (entire blueprint operates on already-landed posterior anchors + ledgers + canonical equations + master-gradient outputs)
- **Total subagent sessions**: 35-51 over 12-18 weeks
- **Total wall-clock**: 12-18 weeks (with 1-2 parallel subagents); 6-9 weeks (with 3-4 parallel subagents per operator approval)
- **Critical path wall-clock**: 12-14 weeks (Dim 1 Phase 1 → Phase 2 → Phase 3 → Phase 4 + parallels)
- **Quick wins (≤2 weeks each)**: Dim 2 (feedback loop) + Dim 6 (dual-tier architecture)
- **High-EV/$ paths**: Dim 1 Phase 2 + Dim 3 + Dim 4

---

## Per-dimension cost table

| Dimension | Phase/Step | Sessions | Wall-clock | $ GPU | EV per Mission Alignment | Dependency chain |
|---|---|---|---|---|---|---|
| **Dim 1 — Mathematical grounding** | | | | | | |
| | Phase 1 (META-LAGRANGIAN-WIRE-1; in flight) | 1-2 | 1-2 weeks | $0 | HIGH (apparatus_maintenance) | NONE (in flight) |
| | Phase 2 (per-adjuster ablation; 10 adjusters) | 5-10 | 4-7 weeks | $0 | HIGH (frontier_protecting → frontier_breaking transition) | Phase 1 |
| | Phase 3 (TRACK A + TRACK B ensemble) | 1-2 | 1-2 weeks | $0 | MEDIUM (frontier_protecting) | Phase 2 (≥3 ablations) |
| | Phase 4 (Dykstra + Pareto constraints) | 2-3 | 2-3 weeks | $0 | HIGH (frontier_breaking enabler) | Phase 3 + Dim 3 Step 3.3 |
| | Phase 5 (sensitivity regularization R(θ)) | 1-2 | 1-2 weeks | $0 | LOW (apparatus_maintenance) | Dim 4 Step 4.3 |
| | Phase 6+ (deprecate hand-derived adjusters) | 1-2 | 1-2 weeks | $0 | HIGH (frontier_breaking via structural extinction) | Phase 5 + 30-day observation |
| **Dim 1 subtotal** | | **11-21** | **10-18 weeks** | **$0** | | |
| **Dim 2 — Feedback-loop frequency** | | | | | | |
| | Step 2.1 (auto-subscriber to call_id_ledger) | 2 | 2 weeks | $0 | MEDIUM (apparatus_maintenance) | NONE |
| | Step 2.2 (per-iteration auto_recalibrate) | 1 | 1 week | $0 | MEDIUM | Step 2.1 |
| | Step 2.3 (consumer update_from_anchor pre-consumption) | 1-2 | 1-2 weeks | $0 | MEDIUM | Step 2.2 |
| | Step 2.4 (--max-iteration-anchor-lookback CLI flag) | 1 | 1 week | $0 | LOW | Step 2.3 |
| **Dim 2 subtotal** | | **5-7** | **2-3 weeks** | **$0** | | |
| **Dim 3 — Per-axis decomposition** | | | | | | |
| | Step 3.1 (Protocol extension predicted_axis_decomposition) | 1 | 1 week | $0 | HIGH (frontier_breaking enabler) | NONE |
| | Step 3.2 (tac.score_composition canonical helper) | 1 | 1 week | $0 | HIGH | Step 3.1 |
| | Step 3.3 (ranker per-axis composition) | 1-2 | 1-2 weeks | $0 | HIGH | Step 3.2 |
| | Step 3.4 (convert 3-5 high-EV consumers) | 3-5 | 3-5 weeks | $0 | HIGH | Step 3.3 |
| | Step 3.5 (Catalog #341 sister-extension gate) | 1 | 1 week | $0 | MEDIUM | Step 3.4 |
| **Dim 3 subtotal** | | **7-10** | **3-4 weeks** | **$0** | | |
| **Dim 4 — Domain priors** | | | | | | |
| | Step 4.1 (tac.domain_priors namespace; 4 wrappers) | 3-4 | 3-4 weeks | $0 | HIGH (frontier_breaking enabler) | NONE |
| | Step 4.2 (3 canonical equations) | 2-3 | 2-3 weeks | $0 | MEDIUM | Step 4.1 |
| | Step 4.3 (domain_prior_consumer cathedral package) | 1-2 | 1-2 weeks | $0 | HIGH | Step 4.2 |
| | Step 4.4 (substrate citation audit; 7 asymptotic candidates) | 1 | 1 week | $0 | MEDIUM | Step 4.3 |
| **Dim 4 subtotal** | | **7-10** | **3-4 weeks** | **$0** | | |
| **Dim 5 — Closed-loop discipline** | | | | | | |
| | Step 5.1 (propose_sister_candidates Protocol extension) | 2 | 2 weeks | $0 | MEDIUM | Dim 2 Step 2.3 |
| | Step 5.2 (tac.substrate_kinship_graph) | 3-4 | 3-4 weeks | $0 | MEDIUM | Step 5.1 |
| | Step 5.3 (auto-discovery extension) | 1-2 | 1-2 weeks | $0 | MEDIUM | Step 5.2 |
| | Step 5.4-5.5 (Catalog #313 + #344 wire-ins) | 2 | 1-2 weeks | $0 | LOW | Step 5.3 |
| | Step 5.6 (--closed-loop-strict CLI flag) | 1 | 1 week | $0 | LOW | Step 5.4-5.5 |
| **Dim 5 subtotal** | | **9-11** | **4-5 weeks** | **$0** | | |
| **Dim 6 — Dual-tier architecture** | | | | | | |
| | Step 6.1 (ConsumerTier enum) | 1 | 1 week | $0 | LOW (apparatus_maintenance) | NONE |
| | Step 6.2 (consumer_tier Protocol field) | 1 | 1 week | $0 | LOW | Step 6.1 |
| | Step 6.3 (Catalog #341 tier-aware extension) | 1 | 1 week | $0 | LOW | Step 6.2 |
| | Step 6.4 (README dual-tier docs) | 1 | 1 week | $0 | MEDIUM (operator-mental-model gap) | Step 6.3 |
| | Step 6.5 (3-5 consumer Tier B promotions) | 3-5 | 3-5 weeks | $0 | MEDIUM | Dim 3 Step 3.4 |
| **Dim 6 subtotal** | | **7-9** | **3-4 weeks** | **$0** | | |
| **TOTAL** | | **46-68** | **12-18 weeks (serial); 6-9 weeks (4-parallel)** | **$0** | | |

---

## Critical-path cost (assuming 1 subagent/week serial)

```
Week 1-2:  META-LAGRANGIAN-WIRE-1 Phase 1 ($0; 2 sessions)
Week 2-3:  Dim 2 Step 2.1 ($0; 2 sessions) — parallel quick win
Week 3-4:  Dim 3 Step 3.1-3.2 ($0; 2 sessions)
Week 4-6:  Dim 4 Step 4.1 ($0; 4 sessions)
Week 6-8:  Dim 3 Step 3.3-3.4 ($0; 4 sessions)
Week 7-12: Dim 1 Phase 2 ($0; 8 sessions)
Week 10-12: Dim 4 Step 4.2-4.3 ($0; 4 sessions)
Week 11-13: Dim 6 Step 6.1-6.4 ($0; 4 sessions)
Week 12-14: Dim 1 Phase 4 ($0; 3 sessions)
Week 13-15: Dim 5 Step 5.1-5.3 ($0; 5 sessions)
Week 14-16: Dim 1 Phase 3 + 5 ($0; 3 sessions)
Week 15-17: Dim 5 Step 5.4-5.6 ($0; 3 sessions)
Week 16-18: Dim 6 Step 6.5 ($0; 4 sessions)
Week 17-18: Dim 1 Phase 6+ ($0; 2 sessions)

TOTAL: ~50 sessions over 18 weeks serial; $0 GPU
```

## Compressed-path cost (assuming 3 parallel subagents)

```
Week 1-2:  WIRE-1 Phase 1 || Dim 2 Step 2.1 || Dim 3 Step 3.1
Week 3-4:  Dim 1 Phase 2 (start) || Dim 3 Step 3.2-3.3 || Dim 4 Step 4.1
Week 5-6:  Dim 1 Phase 2 (cont) || Dim 3 Step 3.4 || Dim 4 Step 4.2
Week 7-8:  Dim 1 Phase 2 (end) || Dim 4 Step 4.3 || Dim 6 Step 6.1-6.2
Week 9-10: Dim 1 Phase 3+4 || Dim 5 Step 5.1-5.2 || Dim 6 Step 6.3-6.4
Week 11-12: Dim 1 Phase 5+6 || Dim 5 Step 5.3-5.6 || Dim 6 Step 6.5

TOTAL: ~50 sessions over 12 weeks (3-parallel); $0 GPU
```

## Per-cost-class roll-up

| Cost class | Total sessions | Total weeks | $ GPU |
|---|---|---|---|
| **Quick wins (≤2 weeks each)** | 10-12 | 2-4 | $0 |
| **Medium efforts (3-5 weeks)** | 25-35 | 4-8 (parallel) | $0 |
| **Long efforts (6-12 weeks)** | 10-20 | 6-12 | $0 |
| **TOTAL** | **46-68** | **12-18 serial; 6-9 parallel** | **$0** |

## EV/$ ranking (top 10 per Dimension × Phase/Step)

| Rank | Item | EV | Sessions | Wall-clock | Mission contribution |
|---|---|---|---|---|---|
| 1 | Dim 3 Step 3.1-3.3 (per-axis Protocol + ranker composition) | HIGH | 3-4 | 3-4 weeks | frontier_breaking enabler |
| 2 | Dim 1 Phase 2 (per-adjuster ablation) | HIGH | 5-10 | 4-7 weeks | frontier_protecting → frontier_breaking |
| 3 | Dim 4 Step 4.3 (domain_prior_consumer wire-in) | HIGH | 1-2 | 1-2 weeks | frontier_breaking enabler |
| 4 | Dim 2 Step 2.1 (auto-subscriber) | MEDIUM | 2 | 2 weeks | apparatus_maintenance + closure quick win |
| 5 | Dim 1 Phase 4 (Dykstra + Pareto) | HIGH | 2-3 | 2-3 weeks | frontier_breaking enabler |
| 6 | Dim 4 Step 4.1 (tac.domain_priors namespace) | MEDIUM | 3-4 | 3-4 weeks | foundation for Dim 4 wins |
| 7 | Dim 5 Step 5.2 (substrate_kinship_graph) | MEDIUM | 3-4 | 3-4 weeks | foundation for Dim 5 closure |
| 8 | Dim 6 Step 6.4 (README dual-tier docs) | MEDIUM | 1 | 1 week | operator-mental-model gap closure |
| 9 | Dim 3 Step 3.4 (convert 3-5 consumers) | MEDIUM | 3-5 | 3-5 weeks | unlocks Dim 6 Tier B |
| 10 | Dim 1 Phase 6+ (deprecate adjusters) | HIGH | 1-2 | 1-2 weeks (after 30d observation) | structural extinction |

## Compounded paid-dispatch budget impact (NOT part of blueprint)

The blueprint is $0 GPU. However, the blueprint ACCELERATES smarter ranking of paid dispatches per T3 Decision 4 (7 asymptotic-pursuit candidates; $5-50 per dispatch; max 2 per 7-day window). Once Dim 1 Phase 4 + Dim 3 + Dim 4 land, the EV of each $5-50 paid dispatch increases because the ranker recommends candidates better-aligned to per-axis + domain-prior signals.

**Estimated indirect impact on paid-dispatch budget**:
- Pre-blueprint ranking: 7 candidates × $5-50 × 2-3 dispatches/candidate (need iteration) = **$70-1050 over 8-12 weeks**
- Post-blueprint ranking (smarter selection + per-axis OPTIMAL FORM detection): same 7 candidates × $5-50 × 1-2 dispatches/candidate = **$35-700 over 8-12 weeks**

**Net savings**: $35-350 over 8-12 weeks; plus higher EV per dispatch due to per-axis ranking.

## Per-substrate-symposium prerequisite check per Catalog #325

| Dimension | Per-substrate symposium required? | Reason |
|---|---|---|
| Dim 1 (mathematical grounding) | NO | All work on existing posterior + canonical equations + ledgers; no new substrate |
| Dim 2 (feedback loop) | NO | Apparatus wiring; no substrate |
| Dim 3 (per-axis) | NO | Protocol + ranker composition; no substrate |
| Dim 4 (domain priors) | NO | Canonical helpers + auto-discovery; no substrate |
| Dim 5 (closed-loop) | NO | Apparatus + sister-kinship; no substrate |
| Dim 6 (dual-tier) | NO | Architectural; no substrate |

**ALL 6 DIMENSIONS BYPASS Catalog #325** by construction (no substrate dispatches; no paid GPU). The blueprint is pure apparatus work.

---

## Cost envelope summary

- **In-blueprint cost**: $0 GPU + 46-68 subagent sessions over 12-18 weeks (serial) or 6-9 weeks (3-4 parallel)
- **Indirect cost savings**: $35-350 over 8-12 weeks on paid-dispatch budget (post-blueprint ranking smarter)
- **HONEST verdict**: $0-100 GPU envelope for the full 3-month window (well under typical compute budget per CLAUDE.md "GPU budget and compute resources" guidance)
- **Risk**: Phase 2 paired-comparison validation (10-week wall-clock contribution) is the longest single-dimension item; if Phase 2 ablation reveals solver-derived ≠ hand-derived divergence requiring T3 architecture review, total wall-clock extends by 2-4 weeks
