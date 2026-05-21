<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:cross_session_reference_to_canonical_frontier_pointer_anchors_per_canonical_frontier_pointer_json_2026-05-15_through_2026-05-21_canonical -->
<!-- FORMALIZATION_PENDING:staircase_v3_synthesis_memo_per_t4_grand_council_symposium_no_new_canonical_equation_registration_required_per_catalog_344_meta_synthesis_scope -->
---
schema: subagent_landing_memo_v1
topic: staircase_v3_post_t4_grand_council_symposium_canonical_cascade_prioritization_20260521
created_at_utc: 2026-05-21T16:15:46Z
author: claude:overnight-ff-t4-grand-council-symposium-staircase-v3-20260521
lane_id: lane_overnight_ff_t4_grand_council_symposium_full_stack_synthesis_theoretical_floor_cascade_20260521
mission_contribution: frontier_breaking_enabler
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paid_dispatch_attempted: false
evidence_grade: "[predicted]"
predicted_band_validation_status: pending_post_training
current_head_before_landing: 99d06f9675acbd4f4d166b40a9b37d3e91f362a2
parent_synthesis_memo: .omx/research/t4_grand_council_symposium_full_stack_synthesis_theoretical_floor_cascade_landed_20260521.md
canonical_frontier_anchor:
  contest_cpu_score: "see .omx/state/canonical_frontier_pointer.json per Catalog #343"
  contest_cuda_score: "see .omx/state/canonical_frontier_pointer.json per Catalog #343"
  refreshed_at_utc: "2026-05-21T08:40:18Z per canonical_frontier_pointer.json snapshot"
event_type: adjudicated
memory_path: .omx/research/staircase_v3_post_t4_symposium_20260521.md
notes: "Sister memo to parent T4 symposium. Expands Axis 13 staircase via 3-tier prioritization + per-position (predicted_score, paid_GPU_cost, wall_clock_estimate) tuples + reactivation criteria. No mutation of CLAUDE.md or prior memos."
---

# STAIRCASE v3 — POST-T4 CANONICAL CASCADE PRIORITIZATION (2026-05-21)

## Parent context

This memo is the sister to `.omx/research/t4_grand_council_symposium_full_stack_synthesis_theoretical_floor_cascade_landed_20260521.md` (T4 grand council symposium). It expands Axis 13 (UPDATED STAIRCASE) per the operator's verbatim directive 2026-05-21.

**Canonical frontier anchor** (per `tac.canonical_frontier_pointer.json` via Catalog #343):
- Contest-CPU: `0.192051` [contest-CPU] (sha `6bae0201fb08...`; lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`; measured 2026-05-15)
- Contest-CUDA: `0.205330` [contest-CUDA] (sha `9cb989cef519...`; lane `pr106_format0d_latent_score_table_20260516_contest_cuda`; measured 2026-05-16)

## Staircase v3 — 3-tier prioritization

### TIER 1: HIGHEST-EV (terminal harvests + ready Phase 2 BUILDs; within-1-week wall-clock)

#### Position 1A: DP1 paired-smoke terminal harvest (IN-FLIGHT decision-point)

| Field | Value |
|---|---|
| **Substrate** | `pretrained_driving_prior` (DP1) |
| **Status** | Modal call_ids IN-FLIGHT: `fc-01KS4KJGDXVXZ9NYRD4HKZ9CET` (baseline) + `fc-01KS4KKYQ09DEEW6BCDRGPBE93` (procedural) |
| **Decision-point** | re-poll deadline ~2026-05-21T13:00Z via harvest cron `d2fb4d7f`; 24h hard deadline ~2026-05-22T06:29Z (Modal cache TTL) |
| **Predicted ΔS** | -0.001 to -0.004 [predicted] per canonical equation #26 IN-DOMAIN `dp1_codebook_bytes` |
| **Paid GPU cost** | $0.04 already fired (RATIFY-2 reduced-budget 50ep / 1.0h) |
| **Wall clock to terminal** | ≤24h from dispatch (deadline ~2026-05-22T06:29Z) |
| **Confidence** | HARD-EARNED-PENDING-TERMINAL |
| **Decision tree** | If rc=0: register first paid contest-axis canonical equation #26 IN-DOMAIN anchor. If rc=124: further reduction (DPP_EPOCHS 50→25 OR timeout 1.0h→0.75h) per RATIFY-2 cascade. |
| **Reactivation criteria** | Catalog #324 post-training Tier-C validation discipline |
| **Operator-routable** | YES — monitor harvest cron + retry if rc=124 |

#### Position 1B: NSCS06 v8 chroma_lut Phase 2 BUILD execution

| Field | Value |
|---|---|
| **Substrate** | `nscs06_v8_chroma_lut` |
| **Status** | L1 SCAFFOLD landed (commit `853d108e2`) + 4 binding revisions applied (commit `20b6b59b3`) + Phase 2 council `29f92af8d` PROCEED_WITH_REVISIONS (5 binding revisions outstanding) |
| **Predicted ΔS (rate-axis only, byte-precise)** | -0.002706 per canonical equation #26 closed form `-25 × 4064 / 37545489` (4064-byte exact match validated via `test_inflate_v1_and_v2_byte_stable_with_matching_lut`) |
| **Paid GPU cost** | ~$0.50 estimated for Phase 2 BUILD execution + paired-axis empirical anchor |
| **Wall clock** | ~24h for Phase 2 BUILD + paired-axis dispatch |
| **Confidence** | HARD-EARNED-DESIGN-PENDING-BUILD (byte-precise predictive evidence; awaiting paid contest-axis anchor) |
| **Reactivation criteria** | Per-substrate symposium per Catalog #325 + post-training Tier-C validation per Catalog #324 |
| **Operator-routable** | YES — Phase 2 BUILD operator-routable per OVERNIGHT-A T2 PROCEED_WITH_REVISIONS council |

### TIER 2: HIGH-EV (within-2-week wall-clock; predicted but not yet empirically validated)

#### Position 2A: 5-substrate procedural variant matrix completion

| Field | Value |
|---|---|
| **Substrates** | NSCS06 v8 + DP1 + VQ-VAE + grayscale_lut + ATW V2 (REMOVAL) |
| **Status** | Design synthesis landed per `5_substrate_procedural_variant_matrix_design_20260521.md` (commit `6b73d2d50`) |
| **Predicted ΔS (aggregate)** | -0.013 [predicted] per canonical equation #26 `_AGGREGATE_PREDICTED_DELTA_S` constant; corroborated by aggregate-12 OVERNIGHT-G recode queue range [-0.0161, -0.0488] |
| **Paid GPU cost** | ~$2-5 estimated for cross-substrate paired-axis composition anchors |
| **Wall clock** | ~3-7 days for 5-substrate composition empirical validation |
| **Confidence** | PREDICTED (aggregate per #26 closed form; no contest-axis empirical anchor yet) |
| **Reactivation criteria** | First paid contest-axis empirical anchor (DP1 IN-FLIGHT) unblocks the aggregate-5-substrate predictive lineage |
| **Operator-routable** | YES — cascade execution after Position 1A terminal harvest |

#### Position 2B: STC residual sidecar over A1 substrate (path 3a)

| Field | Value |
|---|---|
| **Substrate** | `stc_residual_sidecar_over_a1_substrate_path_3a` (NEW substrate per OVERNIGHT-J Path A pivot) |
| **Status** | Design landed per `stc_residual_sidecar_over_a1_path_a_pivot_design_local_cpu_mvp_landed_20260521.md`; recipe NOT YET AUTHORED |
| **Predicted ΔS** | -0.003 to -0.008 [predicted] per OVERNIGHT-J |
| **Paid GPU cost** | $5.20 paired-Modal estimated |
| **Wall clock** | ~2-5 days for recipe authoring + paired Modal + empirical validation |
| **Confidence** | PREDICTED (cross-substrate orthogonality predictor per canonical equation #23 + sister steganalysis paradigm Class F) |
| **Reactivation criteria** | NEW recipe authored + canonical scorer-preprocess routing per Catalog #164 + canonical auth_eval helper per Catalog #226 |
| **Operator-routable** | YES — supersedes legacy STC v2 5x silent-no-spawn dispatch surface |

### TIER 3: MID-EV (within-4-week wall-clock; cross-paradigm class shifts)

#### Position 3A: Z6/Z7/Z8 predictive-coding world-model paradigm + canonical equation #312 quadruple

| Field | Value |
|---|---|
| **Substrates** | Z6 + Z7 (LSTM + Mamba2) + Z8 (hierarchical predictive coding canonical quadruple: Rao-Ballard hierarchy + Mallat wavelet + Hafner DreamerV3 + Wyner-Ziv side-information) |
| **Status** | L0+ per `council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md` + `council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` + `council_symposium_z7_as_mamba_2_full_landing_20260518.md`; Z6 falsification IMPLEMENTATION-LEVEL pending Phase 2 BUILD |
| **Predicted ΔS** | -0.020 to -0.050 [predicted] per canonical equation #312 quadruple |
| **Paid GPU cost** | $15-30 triple-substrate composition paired anchors |
| **Wall clock** | ~1-2 weeks for Phase 2 BUILDs + paired empirical anchors + triple-substrate composition_alpha measurement |
| **Confidence** | UNTESTED-MATHEMATICALLY-GROUNDED (Hafner DreamerV3 + Rao-Ballard 1999 + Wyner-Ziv 1976 + Mallat 1988 canonical references) |
| **Reactivation criteria** | Per-substrate symposium per Catalog #325 + canonical equation #312 quadruple satisfaction |
| **Operator-routable** | YES — class-shift candidate after Tier 1-2 cascade |

#### Position 3B: Cooperative-receiver paradigm (Z4 + TT5L + Wyner-Ziv side-information)

| Field | Value |
|---|---|
| **Substrates** | Z4 cooperative_receiver_loss + TT5L transformer tokens + Wyner-Ziv canonical substrate |
| **Status** | L0+ pending Phase 2 BUILD; Atick-Redlich 1990 + Tishby-Zaslavsky 2015 + Wyner-Ziv 1976 canonical theoretical framework |
| **Predicted ΔS** | -0.020 to -0.040 [predicted] per Atick-Redlich cooperative-receiver theoretical bound + Tishby IB sufficient statistic |
| **Paid GPU cost** | $10-20 triple-substrate composition paired anchors |
| **Wall clock** | ~1-2 weeks for Phase 2 BUILDs + cooperative-receiver theoretical bound empirical validation |
| **Confidence** | UNTESTED-MATHEMATICALLY-GROUNDED (Atick-Redlich + Tishby + Wyner canonical references) |
| **Reactivation criteria** | Per-substrate symposium per Catalog #325 + canonical equation #311 ego-motion conditioning verification (Z6 sister discipline) |
| **Operator-routable** | YES — cross-paradigm class shift after Tier 3A |

### TIER 4: LOW-EV-OBSERVABILITY (DEFER-PENDING-RESEARCH; ROI uncertain)

#### Position 4A: HFV1 sensitivity-weighted sidecar recoder

| Field | Value |
|---|---|
| **Substrate** | `hfv1_pr101_sensitivity_weighted_recoded` (per `pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md`) |
| **Status** | DEFER per OVERNIGHT-K paired anchors +60-75% off frontier; rate-hurdle audit shows 60% sidecar recoder reduction needed to TIE fec6 baseline |
| **Predicted ΔS** | -0.005 to -0.020 [predicted] IF reactivation criteria satisfied |
| **Paid GPU cost** | $0.50-2.00 paired-Modal (after recoder + sensitivity-weighted refinement) |
| **Wall clock** | ~3-7 days for recoder + sensitivity-weighted refinement + PR101 merged runtime root |
| **Confidence** | DEFERRED-PENDING-RESEARCH per CLAUDE.md "Forbidden premature KILL" |
| **Reactivation criteria** | HFV1 sidecar recoder (60% byte reduction) + sensitivity-weighted refinement per master-gradient exploits #2 + #3 + PR101 GOLD frontier base merged runtime root per OVERNIGHT-K op-routable #3 |
| **Operator-routable** | OPTIONAL — only after Tier 1-3 cascade if budget remains |

#### Position 4B: HFV2 cross-substrate generalization template

| Field | Value |
|---|---|
| **Substrate** | LFV1 + other dense sidecar substrates per HFV2 magic-byte dispatch canonical pattern |
| **Status** | DEFER per OVERNIGHT-P op-routable #1; canonical template for cross-substrate generalization |
| **Predicted ΔS** | bounded by HNeRV-class substrate-class mismatch per canonical equation #25; bolt-on territory |
| **Paid GPU cost** | uncertain; depends on sister substrate fit |
| **Wall clock** | ~1-2 weeks per sister substrate generalization |
| **Confidence** | DEFERRED-PENDING-RESEARCH |
| **Reactivation criteria** | LFV1 or sister dense-sidecar substrate identified + per-substrate symposium per Catalog #325 |
| **Operator-routable** | OPTIONAL — observability-only signal value documented per OVERNIGHT-Q §2 Assumption-Adversary verdict |

### TIER 5: DEFER-PERMANENTLY (permanent operational failure mode OR superseded)

#### Position 5A: STC v2 silent-no-spawn dispatch surface (5x recurrence pattern)

| Field | Value |
|---|---|
| **Substrate** | `stc_v2_disambiguator` (legacy) |
| **Status** | PERMANENT DEFER per 5x recurrence pattern (OVERNIGHT-J Path A pivot to STC residual sidecar over A1 substrate supersedes) |
| **Reactivation criteria** | NONE — Path A supersedes (Tier 2B Position 2B) |
| **Operator-routable** | NO — superseded |

#### Position 5B: ATW V2 cdf_table_blob REPLACEMENT paradigm (reclassified to REMOVAL)

| Field | Value |
|---|---|
| **Substrate** | `atw_v2_cdf_table_blob` |
| **Status** | RECLASSIFIED to REMOVAL paradigm per RATIFY-4 EXCLUDED context #6 (commit `eb7338455`) |
| **Reactivation criteria** | NEW substrate runtime where reconstruct_from_wz_residual() actually consumes the CDF table (current runtime does NOT) |
| **Operator-routable** | OPTIONAL — depends on future runtime BUILD |

## Cascade execution cadence (per Carmack MVP-first + Race-mode rigor inversion)

**PRE-leader-shift cadence (current state; no active contest race)**:
- Carmack MVP-first 5-step phasing dominates
- Sequential cascade Tier 1 → 2 → 3
- Operator-routable decisions at each tier completion

**POST-leader-shift cadence (if leaderboard moves)**:
- Race-mode rigor inversion + parallel-dispatch
- Tier 1 + Tier 2 + Tier 3 fan-out simultaneously
- Tier 4 DEFER-PENDING-RESEARCH paths get evaluated for cross-substrate generalization
- Cap=2/day at T2 + ≤3/week at T3 council deliberation cadence

## Cost summary

| Tier | Cumulative paid GPU cost | Cumulative wall clock | Cumulative predicted ΔS |
|------|-------------------------|----------------------|-------------------------|
| 1 (Position 1A + 1B) | $0.54 | ≤48h | -0.001 to -0.007 |
| +2 (Position 2A + 2B) | $2-10 | ~5-12 days | -0.014 to -0.061 |
| +3 (Position 3A + 3B) | $10-30 | ~12-26 days | -0.054 to -0.151 |
| +4 (Position 4A + 4B) | $0.50-2 + uncertain | +3-14 days | -0.005 to -0.020 |

**Cumulative best-case cascade** (Tier 1+2+3 success across all 6 positions): ~$30-40 paid GPU + ~26 days wall clock + cumulative predicted ΔS up to ~-0.15 [contest-CPU] from current frontier 0.192051 → ~0.04 (approaching theoretical floor per §15 T4 main memo).

## Reactivation criteria + retrospective due

- 30-day score-impact retrospective due 2026-06-20 per Catalog #300 Mission alignment Consequence 3
- Annual gate audit per Catalog #300 Mission alignment Consequence 2

## Cross-references

Parent T4 synthesis memo: `.omx/research/t4_grand_council_symposium_full_stack_synthesis_theoretical_floor_cascade_landed_20260521.md`
Sister cascade graph memo: `.omx/research/cascade_graph_post_t4_symposium_20260521.md`
