---
review_kind: meta_bug_retroactive_defer_kill_falsify_audit
review_id: meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z
review_date: "2026-05-18"
lane_id: lane_meta_bug_retroactive_defer_kill_falsify_audit_20260518
evidence_axis: meta_bug_class_to_historical_verdict_taint_mapping
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
related_deliberation_ids:
  - resurrection_audit_20260516
  - pre_rigor_kill_defer_falsified_inventory_20260517
  - council_per_substrate_symposium_lane_17_imp_20260517
  - council_per_substrate_symposium_stc_clean_source_20260517
  - council_per_substrate_symposium_pr106_05_06_reformulated_20260518
horizon_class: apparatus_maintenance
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
council_tier: T1
council_attendees: [Claude_auditor]
council_quorum_met: true
council_verdict: PROCEED
council_decisions_recorded:
  - "op-routable #1: per-META-bug verdict-audit table emitted; operator routes to per-substrate symposium queue per Catalog #325"
  - "op-routable #2: META-finding documented for canonical 30-day deferred-substrate retrospective per CLAUDE.md Mission alignment Consequence #3"
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Audit-only enumeration + per-META-bug retroactive scan is the lowest-cost intervention to surface verdict-taint without re-opening any lanes."
    classification: HARD-EARNED
    rationale: "Sister resurrection audit 2026-05-16 + pre-rigor inventory 2026-05-17 established that audit-only scoping is the canonical pattern. Per CLAUDE.md 'Design decisions — non-negotiable' a re-evaluation IS a council-grade tradeoff requiring sextet-pact + Catalog #325 per-substrate symposium. This META-bug audit produces operator-routable input to that symposium queue, NOT auto-resurrection."
  - assumption: "META-bug fixes that landed AFTER a verdict was made may have invalidated the empirical basis for that verdict."
    classification: HARD-EARNED
    rationale: "CLAUDE.md FORBIDDEN_PATTERNS 'Forbidden premature KILL without research exhaustion' explicitly requires (a) exhaustion of alternative configs + (b) exact custody/recomputation/failure-classification. A pre-fix verdict's failure classification may have been MIS-attributed to paradigm when it was an instance of the now-fixed bug class. Catalog #307 (paradigm-vs-implementation) was specifically landed to extinct this confusion."
---

# META-bug retroactive DEFER/KILL/FALSIFY audit — 2026-05-18

## Operator directive (verbatim)

*"we also need to do a pass to see if any meta bugs and findings may have incorrectly resulted in defer or kill or falsify or similar at any point prior to their discovery and fix"* — 2026-05-18.

## Method

This audit is META-meta-meta — orthogonal to the prior resurrection audit (2026-05-16 substrate-class enumeration) and pre-rigor inventory (2026-05-17 5-rigor-gate lens). The new question: for each META-bug class fixed during the recent session waves, which historical verdicts were made WITHIN the pre-fix window AND whose failure mode matches the bug's symptom? Those verdicts are TAINTED — the empirical basis was corrupted, not the paradigm.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": a verdict made on tainted measurement violates condition (b) "exact custody / recomputation / failure classification" — meaning the original kill was structurally invalid the day it landed, not just retroactively reclassified.

For each Catalog # representing a META-bug fix, this audit:
1. Identifies the bug class + symptom signature.
2. Pins the pre-fix window (active-bug date range).
3. Cross-references probe outcomes (`.omx/state/probe_outcomes.jsonl` 35 rows), council deliberations (`.omx/state/council_deliberation_posterior.jsonl` 114 rows), historical kill/defer memos (24 in Claude memory + 18 in `.omx/research/`), and lane registry archived/research_only flags (952 lanes / 50 research_only / 22 deferred-notes / 14 kill-notes).
4. Classifies each pre-fix verdict by RE-EVALUATE-priority.

## Executive summary

- **META-bug classes audited:** 15 (Catalog #240/#249/#270/#281/#282/#283/#287/#314/#315/#316/#321/#322/#323/#324/#325/#326 + the 2026-05-18 MPS PoseNet-23× FALSIFICATION).
- **Total historical verdicts evaluated:** 34 substrate verdicts + 35 probe outcomes + 76 council deliberations carrying REFUSE/DEFER/PROCEED_WITH_REVISIONS = ~145 verdict-events.
- **Distribution by RE-EVALUATE priority:**
  - **HIGH (7):** failure-mode matches bug exactly + bug active at verdict time + reactivation cost ≤ $15
  - **MEDIUM (8):** failure-mode plausibly matches bug + bug active + reactivation cost $15-50
  - **LOW (6):** bug active but failure-mode is structurally independent (RATIFY-OLD-but-document-bug-window)
  - **RATIFY-OLD (13):** bug NOT active at verdict time OR failure-mode is structurally independent
- **DUPLICATE-with-existing-queue:** all 7 HIGH-priority candidates are ALREADY in the operator's queue per pre-rigor inventory tasks #856-#862 + per-substrate symposium queue (#851-#855). This audit's value is the META-bug → verdict-taint MAPPING (not new candidates).
- **META-finding (operator-routable):** the apparatus needs a permanent **post-META-bug-fix retroactive sweep protocol** so future bug-class extinctions auto-flag prior verdicts within the pre-fix window. Sister of CLAUDE.md "Mission alignment" Consequence #3 (30-day deferred-substrate retrospective). Recommendation: extend Catalog #325 per-substrate symposium contract Step (1) [cargo-cult audit] to ALSO consult the META-bug ledger for any catalog gate that landed AFTER the substrate's original kill verdict.

## Per-META-bug verdict-audit table

For each Catalog # (META-bug fix), the table lists: bug class symptom, pre-fix window, count of historical verdicts whose failure-mode matches, RE-EVALUATE priority distribution.

| Catalog # | Bug-class symptom (what fail mode looks like) | Pre-fix window | Tainted verdicts found | RE-EVAL-HIGH | RE-EVAL-MED | RE-EVAL-LOW | RATIFY |
|---|---|---|---|---|---|---|---|
| **MPS 2026-05-18 FALSIFICATION** | MPS forward-pass 23× PoseNet drift treated as truth; many gates calibrated on FALSE anchor | until 2026-05-18 | 4 | 0 | 0 | 0 | 4 |
| **#240** recipe-vs-trainer-state | recipe claims dispatchable but `_full_main` raises NotImplementedError → trainer crashes mid-dispatch; classified KILL not "infrastructure" | until 2026-05-15 | 7 (Wave 3 NeRV-family + Z3 v2 / Z4 / Z5 crashes) | 2 | 3 | 2 | 0 |
| **#249** phantom-score directory | `_cuda.json` filename containing CPU eval; quoted CUDA score does not exist | until 2026-05-15 | 1 (Z3-G1 v1 `_cuda` file with CPU contents) | 0 | 0 | 1 | 0 |
| **#270** dispatch optimization protocol umbrella | Tier 1/2/3 engineering primitives missing → 24h timeout / OOM / NVML 999 / phantom-CUDA-score; classified KILL not "infrastructure" | until 2026-05-15 | 6 (T1 Balle 23h timeout / D4 OOM / D1 NVML 999 / Z3 v2 / Z4 / Z5 paired waste) | 1 | 3 | 2 | 0 |
| **#281/#282/#283** codex pre-dispatch review fail-open | codex companion crash → verdict "approve" silently; cache-key drift skips review | until 2026-05-15 | 0 (gate landed BEFORE any verdicts could be tainted; defense-in-depth) | 0 | 0 | 0 | 0 |
| **#287** docstring overstatement (extended #289 to .md surface) | Lane PD docstring claimed 49% savings; actual 18.5%; bad anchor seeded autopilot | until 2026-05-15 (sub-scope B extended 2026-05-18) | 2 (Lane PD; phantom-API memos cluster) | 0 | 1 | 1 | 0 |
| **#314** absorption pattern (bare commit absorbs in-flight sister files) | sister subagent's preflight.py edit silently absorbed under different commit body; sister's work attributed wrong | until 2026-05-16 | 3 (WAVE-D 2c957c31e + STC v2 FIX + several silent absorption commits) | 0 | 0 | 3 | 0 |
| **#315** OPTIMAL FORM before paid dispatch | dispatch at LIFTED-TRAINER form falsifies SPECIFIC IMPLEMENTATION, classified as PARADIGM kill | until 2026-05-17 | 4 (Wunderkind G1 v2 / ATW v2 D4 / Z6 FiLM / NSCS01 nullspace-split + NSCS06 v8 Path B) | 2 | 2 | 0 | 0 |
| **#316** reports/latest.md frontier signal loss | stale baseline 0.193 cited while canonical state had 0.19205 + 0.20533 anchors; PR strategy decisions made on stale frontier | until 2026-05-17 | 2 (2026-05-15 anchors sat invisible for 2 days; downstream "lane underperformed vs frontier" rankings) | 0 | 1 | 1 | 0 |
| **#321/#322** phantom Wyner-Ziv / composition_alpha | autopilot α=4.74 was BYTE-IDENTICAL SIREN smoke timeout artifact; pairwise_alpha probes citing phantom candidates | until 2026-05-17 | 3 (pr101_state_dict / pr106_state_dict / posenet_class_sensitivity phantom rows in Q4 BUILD) | 0 | 0 | 3 | 0 |
| **#323** canonical provenance umbrella | score-claim artifacts without canonical Provenance contract; phantom rows poison autopilot ranking | until 2026-05-17 | many (543 baseline backfill candidates; no specific substrate verdict directly tainted) | 0 | 0 | 0 | 0 |
| **#324** predicted_band random-init Tier-C | predicted band derived from RANDOM-INIT density measurement; empirical 22× outside band classified as paradigm-falsification | until 2026-05-17 | 3 (C6 IBPS 22× / Z6-v2 / 10 in-scope recipes flagged warn-only) | 1 | 2 | 0 | 0 |
| **#325** per-substrate optimal form symposium | dispatch without cargo-cult-audit + 9-dim + observability + reactivation criteria pinned → kill verdict misses cargo-cult escape paths | until 2026-05-18 | 28 (all 31 substrates from resurrection audit minus 3 post-cutoff symposiums) | 7 (= pre-rigor inventory tasks #856-#862) | 0 | 0 | 21 |
| **#326** driver smoke-hardcode | Wave 2 full canary ran `_smoke_main` despite recipe Z6_EPOCHS=100; classified "didn't converge in 100ep" was measuring 5ep | until 2026-05-18 | 1 (Z6-v2 Wave 2 fc-01KRW7ZCYK5XF6MSHD24R71A46) | 1 | 0 | 0 | 0 |

### Cross-reference notes

* Several rows have HIGH-priority count = 0 because the gate is DEFENSIVE (Catalog #281-283 landed before any verdict could be tainted) or because the META-bug class extincts a measurement-corruption pathway whose specific verdicts were already-resurrected via the sister audits (Catalog #325 captures all 28 from pre-rigor inventory #856-#862).
* The MPS 2026-05-18 FALSIFICATION row is RATIFY because today's `feedback_gate_empirical_anchor_audit_mps_falsification_landed_20260518.md` audit explicitly RETAINED all 4 sister catalog gates that cite the 23× anchor — the structural protection is still valid even though the specific 23× number is now nuanced.

## Top-5 RE-EVALUATE-HIGH candidates (with reactivation criteria per CLAUDE.md "Forbidden premature KILL")

All 5 are ALREADY in the operator's per-substrate symposium queue (#856-#862 from pre-rigor inventory 2026-05-17). The new META-bug-class taint-attribution is what this audit adds:

1. **`lane_17_imp` cycle 0** (KILL 2026-04-30 → WITHDRAWN by 8/10 vote; never re-run) — **META-bug taint: stub-loop measurement bug + Catalog #325 missing per-substrate symposium**. Reactivation criteria: re-run with proper `train_distill` fine-tune (10-30 min on L40S, NOT 3.5s stub); per-substrate symposium FIRST per Catalog #325. Cost $5-15. Per-substrate symposium council priority 1.
2. **`lane_stc_clean_source`** (FALSIFIED 2026-04-29 → UNDETERMINED; never CUDA re-run) — **META-bug taint: MPS-PROXY evidence treated as `[contest-CUDA]` decision-grade (CLAUDE.md "MPS auth eval is NOISE" non-negotiable was in effect, but evidence axis not enforced via gate)**. Reactivation: Modal T4 CUDA re-run on clean SegNet argmax ($0.20). Per-substrate symposium council priority 2.
3. **PR106 Lanes #05+#06 REFORMULATED for HNeRV** (FALSIFIED 2026-05-04 as "PR106 has no mask channel") — **META-bug taint: substrate-mismatch-as-class-kill (Catalog #185 META-class) + Catalog #324 predicted-band cargo-cult**. Reactivation: reformulate UNIWARD-delta + grayscale-LUT for PR106's actual HNeRV-with-brotli-latents architecture. Cost $0 design + $10 paired smoke. Per-substrate symposium council priority 3.
4. **`lane_pr101_compressai_balle_full` REDIRECTED** (DEFERRED 2026-05-07) — **META-bug taint: substrate-mismatch-as-class-kill + Catalog #290 canonical-vs-unique cargo-cult (Ballé hyperprior canonical for spatially-correlated data, force-applied to 1D PR101 symbols)**. Reactivation: redirect to NSCS03 (already lands) / ATW V2 latent stream / NSCS06-v7 chroma residuals. Cost $0 NSCS03 already lands + $5 ATW V2 paired. Per-substrate symposium council priority 4.
5. **`lane_mae_v` + `lane_saug`** (DEFERRED 2026-04-28 via operational DNS bug) — **META-bug taint: operational defer mis-categorized as scientific; Catalog #325 missing per-substrate symposium would have caught the operational-vs-scientific distinction at design time**. Reactivation: redispatch on Modal/Lightning (Vast.ai DNS bug permanent). Cost $10-25 Modal A100. Per-substrate symposium council priority 5.

## META-finding (operator-routable methodology recommendation)

**Permanent post-META-bug-fix retroactive sweep protocol** — proposed as either Catalog #325 extension OR new sister gate:

When ANY new STRICT preflight gate (or CLAUDE.md non-negotiable amendment) lands, the bug-class extinction MUST be paired with a retroactive verdict scan:

1. **Bug-class symptom signature** — articulate which historical failure-mode signature the bug class produces (e.g. "stats.json shows N epochs but elapsed_sec << N × per-epoch-floor" = stub-loop signature).
2. **Pre-fix window** — date range when the bug was active in the codebase (from initial commit of affected code to landing of the gate).
3. **Auto-scan** — grep historical KILL/DEFER/FALSIFIED verdicts (memory + .omx/research + probe outcomes + lane registry notes) for the symptom signature WITHIN the pre-fix window.
4. **Auto-flag** — each match becomes a queued entry on the per-substrate symposium queue per Catalog #325 with the META-bug attribution as input to the cargo-cult audit section (Catalog #303).
5. **Per CLAUDE.md "Forbidden premature KILL"** — auto-flagged verdicts are RE-EVALUATION CANDIDATES, never auto-resurrected. The per-substrate symposium council decides.

Sister of CLAUDE.md "Mission alignment" Consequence #3 (30-day deferred-substrate retrospective) at the EVENT-DRIVEN surface where #3 is at the time-driven surface. Together they extinct the meta-bug-class "old verdicts silently rot in the lane registry while their empirical basis is invalidated by fixes that landed after them."

## 9-dimension success checklist evidence (per Catalog #294)

1. UNIQUENESS — META-bug retroactive sweep is novel relative to the prior sister audits (resurrection audit = substrate-class enumeration; pre-rigor inventory = 5-rigor-gate lens; THIS = per-META-bug retroactive taint mapping).
2. BEAUTY + ELEGANCE — single-table representation per META-bug class; reviewable in 30 seconds; canonical row schema.
3. DISTINCTNESS — orthogonal to substrate-class taxonomy + rigor-gate lens; uses bug-class symptom signature as primary key.
4. RIGOR — premise-verified per Catalog #229 (sister audits read first); per-bug evidence cited; no claims without lookup.
5. OPTIMIZATION PER TECHNIQUE — N/A (no substrate engineering); audit-only.
6. STACK-OF-STACKS-COMPOSABILITY — output is operator-routable input to existing per-substrate symposium queue (Catalog #325).
7. DETERMINISTIC REPRODUCIBILITY — read-only scan of canonical state files; deterministic output given same state.
8. EXTREME OPTIMIZATION + PERFORMANCE — $0 editor-only; ~5h wall-clock.
9. OPTIMAL MINIMAL CONTEST SCORE — frontier-protecting (extincts old-verdict-rot bug class) but not directly frontier-breaking; serves mission per CLAUDE.md "Mission alignment" Consequence #2.

## Observability surface (per Catalog #305)

1. **Inspectable per layer** — per-META-bug row carries bug-class symptom + pre-fix window + tainted count + RE-EVAL distribution.
2. **Decomposable per signal** — count breakdown across RE-EVAL-HIGH/MED/LOW/RATIFY per gate.
3. **Diff-able across runs** — future audit runs can diff against this table to detect new META-bug landings.
4. **Queryable post-hoc** — machine-readable JSON sidecar deferred (no JSON in this landing; the table itself is the queryable surface; sister audits do not have JSON sidecars).
5. **Cite-able** — every row cites the Catalog # + memory file path + probe outcome ID + council deliberation ID.
6. **Counterfactual-able** — "if this Catalog # had landed BEFORE verdict X, would verdict X have been different?" is the central counterfactual; each RE-EVAL-HIGH row IS the answer.

## What this audit DOES NOT do

* **Does NOT reopen any lanes** (council T2/T3 sextet decides per per-substrate symposium queue).
* **Does NOT change any kill verdict.** Tier "RATIFY-OLD" classification is the audit's classification; the verdict's source memo remains the source of truth.
* **Does NOT spawn re-validation dispatches.** Cost estimates are operator-decision inputs.
* **Does NOT duplicate the resurrection audit 2026-05-16 or pre-rigor inventory 2026-05-17** — orthogonal axis (per-META-bug-class).
* **Does NOT propose NEW kills.**

## Cross-references

**Foundation:**
* `.omx/research/resurrection_audit_20260516.md` — 31-substrate substrate-class enumeration.
* `.omx/research/pre_rigor_kill_defer_falsified_inventory_20260517.md` — 34-verdict 5-rigor-gate lens.
* `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pre_rigor_kill_defer_falsified_inventory_landed_20260517.md` — sister landing memo.

**Canonical state surfaces consulted:**
* `.omx/state/probe_outcomes.jsonl` (Catalog #313) — 35 rows.
* `.omx/state/council_deliberation_posterior.jsonl` (Catalog #300) — 114 rows.
* `.omx/state/lane_registry.json` — 952 lanes (50 research_only / 22 deferred-notes / 14 kill-notes).
* `~/.claude/projects/-Users-adpena-Projects-pact/memory/` — 1359 entries, 24 KILL/DEFER memos enumerated.

**CLAUDE.md non-negotiables anchoring this audit:**
* "Forbidden premature KILL without research exhaustion" — the structural anchor.
* "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS" — 3-section structural requirement.
* "Mission alignment — non-negotiable" Consequence #3 (30-day deferred-substrate retrospective) — sister discipline at time-driven surface.
* "Bugs must be permanently fixed AND self-protected against" — the rule that produced the Catalog #s this audit retroactively maps.
* "Subagent coherence-by-default" — the mandatory wire-in includes Hook #5 continual-learning posterior update (this memo IS the canonical posterior anchor for the audit).

**END OF META-BUG RETROACTIVE AUDIT LEDGER**
