---
catalog_number: 378
gate_function: check_main_thread_spawn_mandate_invokes_catalog_376_verify_head_state
landing_utc: 2026-05-28T23:00:00Z
lane: lane_main_thread_spawn_pv_gap_extinction_20260528
sweep_window_start: 2026-05-15T00:00:00Z
sweep_window_end: 2026-05-28T23:00:00Z
related_anti_pattern: main_thread_subagent_spawn_without_catalog_376_verify_head_state_before_spawn_pv_v1
related_canonical_equation: main_thread_spawn_pv_gap_pre_catalog_376_extension_v1
sister_gates: [376, 340, 314, 302, 230, 287, 176, 185, 299, 344, 348]
council_attendees: []
council_tier: T1
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
---

# Catalog #348 Retroactive Sweep for Catalog #378

Per CLAUDE.md "Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP" non-negotiable + Wave N+25 OP7 op-routable.

## Field 1: bug-class symptom signature

The bug class is **PARENT-MAIN-THREAD agent-spawn-decision without canonical 4-layer PV check via `tac.discipline_anti_pattern_guards.verify_head_state_before_main_thread_spawn`**. Symptom signature:

1. Parent main-thread agent invokes Agent-tool spawn (canonical `Agent.spawn(prompt=...)` / `spawn_subagent(...)` / `subprocess.run(["claude", "-p", ...])` patterns) for a declared scope (file globs / lane id / declared concept).
2. The spawned subagent discovers the work is already landed at HEAD / on-disk / in canonical equations registry / in a sister-subagent's in-flight checkpoint.
3. The subagent emits a STAND_DOWN audit memo per Catalog #229 PV + #307 paradigm-vs-implementation classification.
4. The STAND_DOWN was DETECTED post-spawn at one of: Catalog #340 STAGING-time sister-checkpoint guard / Catalog #314 POST-COMMIT absorption detect / Catalog #302 edit-time-collision detector.
5. The PARENT-MAIN-THREAD had NO structural enforcement that the canonical 4-layer PV helper was invoked BEFORE the spawn, so the audit-trail evidence of which surfaces were checked is absent.

Per Wave N+25 audit memo: today (2026-05-28) alone produced 4 STAND_DOWN incidents matching the symptom signature, ALL detected at STAGING via Catalog #340 rather than at SPAWN via the canonical PV check. The PARENT-side spawn-decision PV gap is empirical.

## Field 2: pre-fix window

The pre-fix window is the apparatus's entire history BEFORE Catalog #378 lands. Concretely:

* Catalog #376 (`check_subagent_spawn_includes_head_state_pv_evidence`) landed 2026-05-28 covers the SUBAGENT-side surface (the subagent's OWN first-checkpoint row must carry PV tokens). This was sister #1 of the canonical 2-landing pattern.
* Catalog #378 (THIS gate) is sister #2 at the PARENT-MAIN-THREAD surface.
* Pre-fix window for this gate: 2026-05-15T00:00:00Z through 2026-05-28T23:00:00Z (the 14-day window since Catalog #340 + #314 + #302 sister gates established the multi-subagent collision class).

## Field 3: historical KILL/DEFER/FALSIFY search results

Searched `.omx/state/probe_outcomes.jsonl` for blocking outcomes (KILL / DEFER / FALSIFICATION) + searched `.omx/research/*stand_down*.md` for STAND_DOWN audit memos. Findings:

### Historical STAND_DOWN memos (8 total in pre-fix window)

| Date | Memo file | Type | Detection surface | Catalog #378 retroactive verdict |
|---|---|---|---|---|
| 2026-05-21 | `vq_vae_indices_blob_procedural_variant_extension_STOOD_DOWN_sister_convergence_20260521.md` | Sister-convergence STAND_DOWN | Catalog #340 STAGING | **PRE-EMPTABLE** if Catalog #378 had fired |
| 2026-05-26 | `z6_l2_3000ep_mlx_local_respawn_stand_down_landed_20260526T172232Z.md` | Predecessor crash + respawn | Catalog #314 POST-COMMIT | **PRE-EMPTABLE** if Catalog #378 had fired |
| 2026-05-26 | `t3_grand_council_symposium_review_falsifications_negatives_similar_STAND_DOWN_20260526T234500Z.md` | Sister-convergence STAND_DOWN | Catalog #340 STAGING | **PRE-EMPTABLE** if Catalog #378 had fired |
| 2026-05-28 | `cascade_a_fec10_second_order_markov_stand_down_audit_20260528.md` | Existing canonical equation | Catalog #344 + sister checkpoint | **PRE-EMPTABLE** if Catalog #378 had fired (Layer 3 + 4) |
| 2026-05-28 | `operator_override_review_paper_STAND_DOWN_per_sister_convergence_20260528.md` | Sister 6.6 min earlier identical scope | Catalog #340 STAGING | **PRE-EMPTABLE** if Catalog #378 had fired (Layer 4) |
| 2026-05-28 | `pr111_paired_cuda_ratification_refire_stand_down_disjoint_yield_to_sister_20260528.md` | Sister yield to sister | Catalog #340 STAGING | **PRE-EMPTABLE** if Catalog #378 had fired (Layer 4) |
| 2026-05-28 | `wyner_ziv_pipeline_stage_codec_resume_stand_down_per_catalog_340_variant_1_20260528.md` | Variant 1 STAND_DOWN | Catalog #340 STAGING | **PRE-EMPTABLE** if Catalog #378 had fired (Layer 4) |
| 2026-05-28 | `z4_stand_down_sister_coherence_audit_20260528T220707Z.md` | Sister z4_atick_redlich_substrate_scaffold owning identical scope | Catalog #340 STAGING | **PRE-EMPTABLE** if Catalog #378 had fired (Layer 4) |

### Historical KILL / FALSIFY verdicts in probe_outcomes.jsonl

Searched `.omx/state/probe_outcomes.jsonl` for blocking verdicts (KILL / DEFER). No verdicts in the pre-fix window match the spawn-decision-PV class. The bug class is operational (subagent-spawn-discipline) rather than substrate-class (which probe_outcomes typically tracks).

## Field 4: per-finding RE-EVAL-priority assignment

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation classification: the 8 historical STAND_DOWN incidents above are **IMPLEMENTATION-LEVEL** (the spawn-decision PV apparatus was not yet wired), NOT paradigm-level. The historical incidents are PRESERVED per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; no re-evaluation needed because they correctly characterize their own historical context.

### RE-EVAL-priority assignment for each historical finding

| Finding | RE-EVAL priority | Action |
|---|---|---|
| `vq_vae_indices_blob_procedural_variant_extension_STOOD_DOWN_sister_convergence_20260521.md` | **LOW** (historical-context-correct) | NO action; STAND_DOWN was correct given pre-#378 apparatus state |
| `z6_l2_3000ep_mlx_local_respawn_stand_down_landed_20260526T172232Z.md` | **LOW** (historical-context-correct) | NO action |
| `t3_grand_council_symposium_review_falsifications_negatives_similar_STAND_DOWN_20260526T234500Z.md` | **LOW** (historical-context-correct) | NO action |
| `cascade_a_fec10_second_order_markov_stand_down_audit_20260528.md` | **LOW** (historical-context-correct; existing canonical equation already documented per Catalog #344) | NO action |
| `operator_override_review_paper_STAND_DOWN_per_sister_convergence_20260528.md` | **LOW** (historical-context-correct) | NO action |
| `pr111_paired_cuda_ratification_refire_stand_down_disjoint_yield_to_sister_20260528.md` | **LOW** (historical-context-correct) | NO action |
| `wyner_ziv_pipeline_stage_codec_resume_stand_down_per_catalog_340_variant_1_20260528.md` | **LOW** (historical-context-correct) | NO action |
| `z4_stand_down_sister_coherence_audit_20260528T220707Z.md` | **LOW** (historical-context-correct) | NO action |

All 8 historical STAND_DOWNs are correctly classified as IMPLEMENTATION-LEVEL pre-#378 events per Catalog #307. The structural protection landed with Catalog #378 means future spawns that fire the canonical 4-layer PV helper BEFORE Agent.spawn() will PRE-EMPT the STAND_DOWN at the source rather than detect it post-hoc.

### Forward-looking RE-EVAL priority

* **HIGH** priority: ensure all NEW main-thread spawn-decision sites route through `verify_head_state_before_main_thread_spawn` per Catalog #378's structural enforcement.
* **MEDIUM** priority: monitor for new STAND_DOWN incidents in the 14-day post-#378 window; if any occur AT STAGING/POST-COMMIT surfaces (Catalog #340/#314), the spawn-decision PV was bypassed and the gate's strict-flip timeline should be moved forward.
* **LOW** priority: backfill historical pre-#378 spawn sites with the canonical helper invocation if/when they are touched for unrelated reasons (Catalog #229 PV opportunistic discipline).

## Sister-gate cross-reference

Catalog #378 sister-extincts the multi-subagent edit/commit/spawn collision class at the 10th surface:

* **Spawn-time PARENT-side**: Catalog #378 (THIS gate)
* **Spawn-time SUBAGENT-side**: Catalog #376
* **Edit-time-checkpoint**: Catalog #302
* **Edit-time-bulk-op**: Catalog #230
* **Commit-time-pre-pre-lock**: Catalog #157
* **Commit-time-staged**: Catalog #216
* **Commit-time-lock-arbitration**: Catalog #117 + #174
* **Post-resolution-residual-marker**: Catalog #248
* **Post-commit-absorption-detect**: Catalog #314
* **Staging-surface-prevent**: Catalog #340

## Mission contribution

Per Catalog #300: `apparatus_maintenance`. Closes the PARENT-MAIN-THREAD spawn-decision PV gap structurally; the canonical helper + STRICT gate sister-extinct the multi-subagent collision class at the 10th surface; unblocks future main-thread spawn-decision sites from silently producing STAND_DOWN by enforcing the canonical 4-layer PV evidence chain at the structural source-level surface.
