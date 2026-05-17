---
title: "Wave 3 K=13 LEVEL-1 substitution recommendation per lattice-coherence audit"
date: 2026-05-16
author: COHERENCE-AUDIT-LATTICE subagent
lane: lane_coherence_audit_lattice_coordinate_assignment_20260516
horizon_class: apparatus_maintenance
council_tier: T1
council_predicted_mission_contribution: frontier_protecting
related: coherence_audit_lattice_coordinate_assignment_20260516.md
---

## TL;DR

Current K=13 LEVEL-1 (per `.omx/research/k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516.md`) is ALREADY coherent with the operator binding constraint 2026-05-16 *"Remember we need outside nerv-family too"*. 12-of-13 active measurements are outside-NeRV per the canonical taxonomy (the 1 N/A slot is the NSCS06 v9 design memo disambiguator, now structurally deferred per the NSCS06 family DEFER memo `5f44f0e97`).

The PROMPT'S NeRV-family concern referred to an EARLIER hypothetical plan that does NOT match the current canonical K=13 schedule. Per CLAUDE.md "Apples-to-apples evidence discipline": the prompt's "7 of 8 frontier-pursuit as NeRV-family" claim should be re-checked against the canonical K=13 schedule memo §4 enumeration; the current frontier-pursuit bucket has **0 NeRV-family** in active slots.

## RECOMMENDED substitution: Set B (operationally cleanest)

Replace 2 deferred / structurally-blocked slots with 2 lifted-pending-council outside-NeRV Rule #2 substrates:

| Slot | Current | Substitute | Rationale |
|---|---|---|---|
| Plateau-adjacent #4 | PR106 r2 baseline re-anchor (~$3-5; baseline only) | **NSCS01 Phase 2 paid smoke** (~$10-25) | NSCS01 _full_main lifted 2026-05-15 with PR95-paradigm fully bound. Canonical Rule #2 substrate. Outside-NeRV. Predicted band [0.180-0.188] FRONTIER-PURSUIT (RECLASSIFY from plateau to frontier on substitution). |
| Disambiguator #1 | NSCS06 v9 design memo ($0; structurally deferred per NSCS06 family DEFER) | **NSCS03 Phase 2 paid smoke** (~$15-25) | NSCS03 Ballé end-to-end joint codec lifted 2026-05-15 with 76 passing tests. Outside-NeRV. Different architectural class from NSCS01 (balle_2018_end_to_end_joint_codec vs pr95_paradigm_nullspace_split). Predicted band [0.180-0.190] FRONTIER-PURSUIT. |

## Net effect of Substitution Set B

| Metric | Before | After |
|---|---|---|
| Plateau-adjacent | 4 | 3 |
| Frontier-pursuit | 5 | 7 |
| Asymptotic-pursuit | 3 | 3 |
| Disambiguator | 1 | 0 |
| Total K | 13 | 13 |
| NeRV-family count | 0 | 0 |
| Outside-NeRV count | 11 (deferring NSCS06 v8 Path B which is DEFERRED) | 12 (+1) |
| Cost envelope | $59-120 | $74-150 (within $50-150 budget; +$15-30) |
| ρ = sparsity/K (Donoho-Tanner) | 0.417 (EXACT regime) | 0.417 (preserved; sparsity unchanged) |

## Why preserve K=13 (don't grow to K=15)

Per Donoho-Tanner 2009 phase-transition analysis in the K=13 schedule memo §1: K=12 first-satisfies-Donoho-Tanner + HORIZON-CLASS allocation; K=13 is the canonical midpoint. Growing to K=15 buys 7% uncertainty reduction for ~$20-30 additional spend — diminishing returns per §5.3 of that memo. Substitution Set B preserves K=13 + costs only $15-30 within budget.

## Substitution Set A (conservative; 1 substitution only)

Replace ONLY PR106 r2 baseline re-anchor with NSCS01 Phase 2.

- Net K=13 frontier-pursuit: 5 → 6
- Net outside-NeRV: 11 → 12
- Cost delta: ~+$5-20 within budget

## Substitution Set C (aggressive; rebuild frontier-pursuit)

Replace BOTH NSCS06 v8 Path B harvest (DEFERRED) + NSCS06 v9 design (structurally deferred) with NSCS01 + NSCS03:

- Net K=13 frontier-pursuit: 5 (1 was DEFERRED) → 6 (all 6 active outside-NeRV)
- Disambiguator bucket: 1 → 0
- Net outside-NeRV: 11 → 12 (same as Set B)
- Cost delta: ~+$25-50 within budget

## Operator decision queue

1. **CHOOSE substitution set** A / B / C (recommended: B per balanced trade-off)
2. **APPROVE Phase 2 council preview for NSCS01 + NSCS03** (their recipes currently carry `research_only: true + dispatch_enabled: false` per Catalog #240; Phase 2 council green-up unlocks paid smoke)
3. **ROUTE substitution through `tools/check_lattice_coordinate.py --list-outside-nerv` BEFORE confirming** to verify the final outside-NeRV count satisfies the operator binding constraint

## Cross-references

- `.omx/research/coherence_audit_lattice_coordinate_assignment_20260516.md` (the parent audit memo — full deliverable)
- `.omx/research/k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516.md` (the canonical K=13 schedule this memo proposes to substitute on)
- `feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md` (the canonical roadmap framework)
- `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md` (NSCS01 lift evidence)
- `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md` (NSCS03 lift evidence)
- `.omx/research/nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516.md` (NSCS06 family DEFER — explains why v8 harvest + v9 design are deferred)
