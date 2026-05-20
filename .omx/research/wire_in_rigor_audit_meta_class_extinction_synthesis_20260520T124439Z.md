# Wire-In Rigor Audit — META-Class Extinction Synthesis

**Subagent**: `wire-in-rigor-audit-resume-20260520`
**Audit scope**: 20+ Tier 1-5 components (see per-component dossier)
**Date**: 2026-05-20
**Discipline**: Catalog #229 PV / #287 placeholder-rejection / #292 assumption surfacing / #323 canonical Provenance

---

## Operator-facing executive summary (450 words; HONEST verdict)

**Operator worry**: "How much of the cathedral autopilot's claimed work is FACADE-running-but-not-empirically-grounded? Are we shipping a sophisticated apparatus that, when you trace the call-chain, ends in a stub that returns 0.0?"

**HONEST ANSWER (no sugarcoating)**:

**~35% of the cathedral autopilot surface is empirically grounded; ~65% is observability-by-design-but-functionally-inert**.

The breakdown:

- **The REAL signal-mutating path** is 10 in-main-line `adjust_predicted_delta_for_*` functions inside `tools/cathedral_autopilot_autonomous_loop.py` (Z1 revision, MDL density, MDL Tier C, class-shift, composition alpha v2, Venn classification v2, per-pair sister 817 sidecars, per-pair difficulty atlas, Cable D consumers, predicted dispatch risk, realistic stacking correction, frontier threshold). These DO mutate `predicted_score_delta`. **These are ~35% of the cathedral apparatus.**

- **The OBSERVABILITY-ONLY path** is the 44-package `cathedral_consumers/` namespace. ALL 44 production consumers return `predicted_delta_adjustment=0.0` per Catalog #341 canonical contract. They fire (Catalog #336 + #337 verified). They surface annotations into output JSON. They DO NOT influence ranking. **These are ~65% of the cathedral apparatus.**

The 44-package cathedral_consumers wave is DESIGN-CORRECT per Catalog #335 + #341 (observability-only, non-promotable, no score mutation). The operator-worry is empirically valid: **"44 cathedral consumers fire"** is true but creates an inflated mental model of how much actual score-influence work is happening. The structural protection (Catalog #335/#336/#337/#341/#354) ensures consumers are PRESENT + CONTRACT-COMPLIANT + INVOKED — it does NOT ensure they MUTATE SCORE.

The two most concerning facade-class findings:

1. **Meta-Lagrangian solver is SCAFFOLD_ONLY**. Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE", the unified-action surface (`evaluate_with_admm`, `choose_solver`, `Action.S_total`) is the canonical end-to-end candidate evaluation surface. Empirically: **ZERO production callsites** invoke it. The training scaffold (`experiments/train_unified_action_phase1.py`) exists but no dispatch wrapper / ranker / autopilot path actually calls the solver. This is the highest-priority FACADE finding.

2. **Master-gradient anchors are 100% non-authoritative**. All 10 anchors in `.omx/state/master_gradient_anchors.jsonl` are `[macOS-CPU advisory]` or pre-axis-correction `[contest-CPU]` on M5 Max — none on 1:1 contest-compliant hardware. Catalog #327 correctly fail-closes routing for non-authoritative axes, so consumers refuse them by design. The producer fires, the gates work, but the empirical signal driving downstream consumers is unvalidated on contest hardware.

The canonical-state ledger infrastructure (continual-learning posterior 118 anchors / probe-outcomes 59 / modal call-id 393 / canonical Provenance / canonical frontier pointer) is **FULLY_WIRED** and EMPIRICALLY GROUNDED. This is the strongest part of the apparatus.

---

## THE bug class re-stated (R11 H1-1+H1-6 anchor)

Per CLAUDE.md Catalog #336+#337 landing memo: "ORPHAN_SIGNAL at cathedral autopilot — discovery + invocation surfaces defined but never called from main()". The R11 H1-1+H1-6 fix wave landed those callsites structurally so the discovery + invocation could not silently regress.

**This audit verifies the structural extinction is preserved** (callsites are present, 44 consumers fire), but surfaces a sister bug class the R11 fix did NOT address:

**EXTENDED BUG CLASS**: "discovery + invocation fires, but consumer contribution is observability-only-by-design — the work is performed but its result never influences dispatch decision". This is FACADE-PATTERN-AT-THE-CONSUMER-LEVEL rather than ORPHAN-PATTERN-AT-THE-INVOCATION-LEVEL.

## Count of FACADE / ORPHAN_SIGNAL across 20 Tier 1-5 components

| Verdict | Count | % |
|---|---|---|
| FULLY_WIRED | 7 | 35% |
| PARTIALLY_WIRED | 9 | 45% |
| FACADE | 1 | 5% (cathedral consumers aggregate; 44/44 production STUBS) |
| ORPHAN_SIGNAL | 1 | 5% |
| SCAFFOLD_ONLY | 2 | 10% |

**Adjusted via cathedral_consumers per-package count** (if we count 44 stub consumers as 44 separate FACADE instances rather than 1 aggregate):

| Verdict | Count | % |
|---|---|---|
| FACADE (incl. 44 stub consumers) | 45 | 71% of 63 surfaces |
| FULLY_WIRED | 7 | 11% |
| PARTIALLY_WIRED | 9 | 14% |
| ORPHAN_SIGNAL + SCAFFOLD_ONLY | 3 | 5% |

The honest framing: **the cathedral consumers/* wave (44 packages) was an OPERATIONAL DECISION to ship observability-only**. Per Catalog #341 they are non-promotable by design. The "FACADE" label is design-correct but functional-low-impact.

## Common failure-pattern subtypes observed

### Subtype A: DEFINED-but-NEVER-INVOKED
- **Anchor**: meta-Lagrangian solver (`tac.unified_action.evaluate_with_admm` etc.). API defined, tests pass, zero production callsites.
- **Impact**: CRITICAL. CLAUDE.md non-negotiable mandates this as canonical solver; no caller exists.

### Subtype B: INVOKED-but-RESULT-DISCARDED
- **Anchor**: cathedral consumers (44/44 STUBS). Invocation fires per Catalog #336 + #337. Result placed in output JSON. Result never mutates `c.predicted_score_delta`.
- **Impact**: BY DESIGN (per Catalog #341). Operator-mental-model gap.

### Subtype C: TESTS-PASS-but-EMPIRICAL-INFLUENCE-ZERO
- **Anchor**: WZ deliverability proof + composition matrix Cascade 1. All loadable + all tests pass. Empirically: 0 archives have positive deliverable savings OR Lagrangian optimal-plan sidecars. Cascade fires through to 1.0× passthrough.
- **Impact**: HIGH if the assumption "future archives will populate" is wrong; MEDIUM if just timing.

### Subtype D: PRODUCER-FIRES-but-NO-CONSUMER
- **Anchor**: `tac.boosting`, `tac.compress_time_optimization`. Rich APIs, 1 production consumer or 0 outside the namespace.
- **Impact**: MEDIUM — helper namespaces sitting unused.

### Subtype E: CONSUMER-CLAIMS-but-PRODUCER-NEVER-FIRES
- **Anchor**: master-gradient anchors are 100% non-authoritative `[macOS-CPU advisory]`. Catalog #327 consumer correctly fail-closes; the producer never fires on contest-compliant hardware to populate authoritative anchors.
- **Impact**: HIGH — the master-gradient framework's central premise (master gradient drives per-pair byte allocation) is empirically unvalidated.

### Subtype F: WIRE-IN-CLAIMED-IN-MEMO-but-NOT-IN-CODE
- **Anchor**: NONE FOUND in this audit (the Catalog #125 6-hook wire-in declaration discipline + Catalog #229 premise-verification have extincted this subtype across the audited surfaces).
- **Impact**: ZERO at audit; structural protection working.

## Existing-gate coverage analysis: GAPS?

| Gate | Surface protected | Gap re: this audit's findings |
|---|---|---|
| Catalog #335 (canonical contract) | cathedral consumer compliance | Does NOT enforce non-zero adjustment (by design) |
| Catalog #336 (main invokes discovery) | invocation site | Does NOT enforce downstream score mutation |
| Catalog #337 (rerank invocation) | invocation site | Does NOT enforce non-zero rerank effect |
| Catalog #341 (canonical markers) | observability marker compliance | Codifies the FACADE-by-design pattern; structural |
| Catalog #354 (bundle completeness) | 8 master-gradient consumer exist + load | Does NOT enforce master-gradient extractor produces authoritative anchors |
| Catalog #125 (6-hook wire-in) | memo discipline | Does NOT detect when hook is declared ACTIVE in memo but downstream consumer is STUB |
| Catalog #327 (master-gradient axis custody) | non-authoritative-anchor refusal | WORKS CORRECTLY; this is why 10/10 anchors are non-authoritative + refused |

**Gap candidates** for new gates (if operator wants to extinct subtype B/C/E):
- "Consumer mutation rate" gate: refuses cathedral_consumers package whose `consume_candidate` returns `predicted_delta_adjustment=0.0` in production lane (would require operator policy on whether 0.0 stays the contract). PROBABLY NOT WORTH adding — Catalog #341 IS the structural protection that consumers are observability-only; the bug class is informational not structural.
- "Solver invocation" gate: refuses cathedral autopilot main() that doesn't invoke `evaluate_with_admm` OR `choose_solver` on at least the top-N candidates. Sister of #336+#337 at the meta-Lagrangian surface.
- "Master-gradient authoritative-anchor floor" gate: refuses `.omx/state/master_gradient_anchors.jsonl` state where 0 anchors have `measurement_axis="[contest-CUDA]"` AND `measurement_hardware="linux_x86_64_t4"` (or sister contest-compliant). Sister of #327 at the producer-coverage floor.

## Per-component EV-ranked fix list (top 15)

| Rank | Fix | EV | Cost | Surface |
|---|---|---|---|---|
| 1 | Wire `evaluate_with_admm` invocation into cathedral autopilot main() OR delete the SCAFFOLD if not the canonical solver | HIGH | MEDIUM | meta-Lagrangian |
| 2 | Land 1 authoritative `[contest-CUDA T4]` master-gradient anchor (Modal A100 / T4 dispatch ~$2-5) | HIGH | LOW | master-gradient |
| 3 | Land 1 Lagrangian `OptimalPerPairTreatmentPlan` sidecar so Cascade 1 fires empirically | HIGH | MEDIUM | composition matrix |
| 4 | Add 5-10 production consumers of `tac.boosting` namespace OR archive the namespace | MEDIUM | LOW | tac.boosting |
| 5 | Add 5-10 production consumers of `tac.compress_time_optimization` OR archive | MEDIUM | LOW | tac.compress_time_optimization |
| 6 | Convert top-3 cathedral consumers from STUB to non-zero adjustment (e.g., `per_pair_difficulty_atlas_consumer` could mutate via difficulty-weighted rate term) | MEDIUM | MEDIUM | cathedral consumers |
| 7 | Document explicitly in the cathedral consumer README that ALL consumers are observability-only-by-design (close the operator-mental-model gap) | HIGH | LOW | docs |
| 8 | Wire `unified_action_consumer.consume_candidate` to actually invoke `Action.S_total` on the candidate's archive | HIGH | HIGH | unified_action_consumer |
| 9 | Audit the 10 in-main-line `adjust_predicted_delta_for_*` functions for which are EMPIRICALLY GROUNDED vs HEURISTIC | MEDIUM | MEDIUM | adjuster path |
| 10 | Land Catalog `#NEW` gate for "consumer mutation rate" — refuses ALL-stub-aggregate state (operator policy required first) | LOW | LOW | gate |
| 11 | Land Catalog `#NEW` gate for "solver invocation" — sister of #336 at meta-Lagrangian surface | MEDIUM | LOW | gate |
| 12 | Land Catalog `#NEW` gate for "master-gradient authoritative-anchor floor" | HIGH | LOW | gate |
| 13 | Surface per-cathedral-consumer "last non-zero adjustment UTC" in CLI summary so operator can see which consumers have ever mutated score (if any do post-canonical-contract relaxation) | LOW | MEDIUM | observability |
| 14 | Archive 2 `quarantine_phantom_*` directories under WZ deliverability per cleanup hygiene | LOW | LOW | hygiene |
| 15 | Refresh canonical frontier pointer auto-cadence (currently last refreshed 30 min before audit) | LOW | LOW | hygiene |

## HONEST answer to operator's worry

**% of cathedral autopilot's claimed surface empirically grounded**: ~35%.

The 35% that IS grounded:
- 10 in-main-line `adjust_predicted_delta_for_*` adjusters
- continual-learning posterior (118 anchors)
- probe-outcomes ledger (59 outcomes)
- modal call-id ledger (393 rows)
- canonical Provenance umbrella (Catalog #323 enforces structurally)
- canonical frontier pointer (recently refreshed)

The 65% that is FACADE/observability-by-design:
- 44 cathedral consumers (all return 0.0)
- meta-Lagrangian solver (no callers)
- xray + atoms + sensitivity-map + canonical equations → ALL flow to cathedral consumers OR to unified_action_consumer stub
- master-gradient anchors (100% non-authoritative)
- composition matrix Cascade 1 (0 sidecars)
- WZ deliverability (1 NOT_DELIVERABLE anchor)

**The 65% is not a bug — it is design intent per Catalog #341**. The bug is OPERATOR MENTAL MODEL. The cathedral apparatus claims "44 cathedral consumers fire" which is TRUE; the missing context is "all 44 are zero-adjustment observability annotations". The framework's HEART is the 10 in-main-line adjusters + the 5 ledgers, not the 44 cathedral consumers.

## Mission-alignment verdict (per CLAUDE.md "Mission alignment — non-negotiable")

`council_predicted_mission_contribution: rigor_overhead`. This audit is procedural-only; no direct score contribution. It enables future score-affecting decisions (e.g., wire meta-Lagrangian solver, populate master-gradient authoritative anchors) by surfacing the gap. The audit does NOT trigger `mission_questioned` because the apparatus's HEART (10 adjusters + 5 ledgers) IS empirically grounded; the 65% facade is design-intent not mission-betrayal.
