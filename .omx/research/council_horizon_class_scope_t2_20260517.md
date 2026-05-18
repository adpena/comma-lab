---
schema: council_deliberation_v2
deliberation_id: council_horizon_class_scope_t2_20260517
topic: "HORIZON-CLASS scope decision: Wyner-Ziv work — FRONTIER_PURSUIT [0.147, 0.167] vs ASYMPTOTIC_PURSUIT [0.050, 0.120]"
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The May-4 race postmortem is the canonical anchor for this decision and the council is at risk of forgetting it inside 13 days. PR105 kitchen_sink lost 1776 LOC across 21 files to rem2's 241 LOC silver in the same 4h08 race window. ASYMPTOTIC_PURSUIT [0.050, 0.120] from a fec6 0.19205 baseline is a delta of 0.07-0.14 score points. Q1-Q5 deliver [0.147, 0.167] = 0.025-0.045 delta for ~$0.70 + 10.25h. Q6+ ASYMPTOTIC stacking targeting 0.07-0.14 delta requires a multiplicative ~3-5× LOC + wall-clock + GPU spend AND assumes near-orthogonal composition that has NEVER been empirically demonstrated across more than 2 stacked pre-entropy hoists. The pre-entropy substrate pivot prober (a98c94e1) identified pr101_state_dict + pr106_state_dict at ~0.47 each AND posenet_class_sensitivity at theoretical 11.6 BUT the theoretical 11.6 is the gross compressible bytes NOT the realized contest-CUDA delta — same cargo-cult class the symposium just extincted with the deliverability_proof gate. I REFUSE to recommend an unconditional ASYMPTOTIC commitment. The MINIMUM-VIABLE-PATH is FRONTIER first, EMPIRICAL ANCHOR on Q4, THEN re-evaluate Q6+ stacking with measured composition_alpha — not predicted."
  - member: Yousfi
    verbatim: "ASYMPTOTIC scale (>3 stacked hoists) requires me to certify each tier independently against PR #35 + Catalog #146 (contest_one_video_replay). Tier 1 (deterministic) composes additively without certification overhead per-stack because each is FRAME_0-derived. Tier 2 (baked constants) requires a per-baker SHA-pinned attestation: Comma2k19 palette + ImageNet luma + dashcam class-prior = 3 bakers needs 3 separate Catalog #210 codebook provenance landings. Tier 3 (scorer-features baked) requires PER-BAKER frozen-weight attestation by the operator — that is operator-blocking, not subagent-blocking. A 5-baker stack (Tier 1 + 2×Tier 2 + 2×Tier 3) means 2 separate operator-review cycles BEFORE the autopilot ranker can apply any reward. The compliance overhead of ASYMPTOTIC is not zero. I am NOT vetoing ASYMPTOTIC; I am noting the per-stack contest-compliance cost is structurally O(#stacks) at the operator-attention surface."
  - member: Assumption-Adversary
    verbatim: "The deliberation is operating within FIVE SHARED ASSUMPTIONS the council should surface explicitly. (1) The autopilot ranker's 4-tier deliberable taxonomy from the T3 symposium is GENERIC across substrates. Classification: CARGO-CULTED in the absence of >=2 empirically verified hoists. The taxonomy was DESIGNED on fec6 fec6-class evidence; whether it generalizes to pr101_state_dict + posenet_class_sensitivity is unverified. (2) Composition_alpha ≈ 1.0 (near-orthogonal Wyner-Ziv contributions across stacked pre-entropy substrates). Classification: CARGO-CULTED — Catalog #227 substrate composition matrix actively guards against composition_alpha overestimation; the pre-entropy prober has NOT measured pairwise interaction, only marginal compressibility. (3) The L5 codex review's rate-only band [−0.0019, −0.0032] generalizes from poses.bin (~4800 bytes) to state_dict (~925KB) AND posenet_class_sensitivity (20MB). Classification: CARGO-CULTED — the rate-only bound is derived from a per-section payload assumption; 20MB of raw fp32 cannot be staircased into the contest archive without splitting into multiple sections each requiring its own L5-style verification. (4) Q1-Q5 are sufficient to extinct the autopilot fake-reward bug class. Classification: HARD-EARNED — the symposium verdict is structurally bound to Q1-Q5; extending to Q6+ does not strengthen the bug-class extinction. (5) FRONTIER vs ASYMPTOTIC is a binary choice. Classification: CARGO-CULTED — the third option HYBRID (FRONTIER first with empirical anchor, gated transition to ASYMPTOTIC contingent on Q4 measured composition_alpha) is structurally distinct and is what the pre-entropy prober's evidence supports. The verdict structure should be HYBRID, not FRONTIER-only or ASYMPTOTIC-only."
council_assumption_adversary_verdict:
  - assumption: "4-tier deliverability taxonomy generalizes from fec6 to pr101_state_dict + pr106_state_dict + posenet_class_sensitivity"
    classification: CARGO-CULTED
    rationale: "The T3 symposium designed the 4-tier classification using fec6's poses.bin (~4800 bytes) as the canonical empirical anchor. pr101_state_dict is ~925KB; posenet_class_sensitivity is 20MB. Both are pre-entropy (lzma compresses 0.143-0.228) but the deliverability classification (Tier 1/2/3/4) depends on the BAKER substrate, not just the bit-level compressibility. A 925KB state_dict hoist could be Tier 2 (baked constants from compress-time codebook) OR Tier 3 (baked from scorer features) OR Tier 4 (if it requires raw scorer weights at inflate). Per Catalog #209 the contest video leakage gate applies if the codebook is trained on contest video; per Catalog #213 Comma2k19 IS the canonical OOD prior. Until the per-substrate deliverability_proof is empirically verified for each pre-entropy substrate, the 4-tier classification is a HYPOTHESIS, not a verified taxonomy."
  - assumption: "Pre-entropy substrate Wyner-Ziv hoists compose additively (composition_alpha ≈ 1.0)"
    classification: CARGO-CULTED
    rationale: "Catalog #227 (substrate composition matrix) explicitly guards against composition_alpha overestimation; the canonical lattice penalizes 0.3 < α < 0.7 SUB-ADDITIVE compositions by halving predicted ΔS and floors α ≤ 0.3 SATURATING at -0.005. The pre-entropy prober (a98c94e1) measured MARGINAL compressibility per substrate; it did NOT measure pairwise interaction. Stacking pr101_state_dict (0.477) + pr106_state_dict (0.470) + posenet_class_sensitivity (11.608) under composition_alpha = 1.0 yields predicted ΔS = 12.555; under composition_alpha = 0.5 SUB-ADDITIVE = 6.27; under composition_alpha = 0.2 SATURATING = floor -0.005. Without empirical pairwise interaction measurement, the ASYMPTOTIC band [0.050, 0.120] = baseline 0.19205 - delta 0.07-0.14 implicitly assumes α ≥ 0.7 ADDITIVE. This is a CARGO-CULTED extrapolation per the HARD-EARNED-vs-CARGO-CULTED addendum."
  - assumption: "Operator-attention budget supports ≥3 additional T2 + ≥1 T3 deliberation for Q6+ stacking"
    classification: UNCLEAR
    rationale: "Per CLAUDE.md 'Council hierarchy: 4-tier protocol' the T2 cadence ceiling is ≤3/day, ≤90/30d; T3 is ≤3/week, ≤13/30d. Q6+ ASYMPTOTIC stacking would generate (per Yousfi's per-stack compliance cost analysis) 3-5 additional T2 deliberations (one per baker design) + 1-2 T3 escalations (each Tier 3 frozen-weight attestation crosses CLAUDE.md non-negotiable scope). At T2 the cadence is within budget if spread across 3-5 days. At T3 the cadence is within budget but consumes ~15-30% of the 30-day envelope for ONE substrate class. The operator-attention surface is fungible; whether ASYMPTOTIC-Wyner-Ziv-stacking deserves 15-30% of T3 budget vs other paradigm-level work (e.g. NSCS06 cargo-cult unwinds, time_traveler L5 staircase, F-asymptote substrates per Catalog #309) is OPERATOR-decidable, not council-decidable. Classification UNCLEAR until operator routes."
  - assumption: "Race-mode rigor inversion applies symmetrically to FRONTIER and ASYMPTOTIC"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Race-mode rigor inversion + parallel-dispatch first — NON-NEGOTIABLE, HIGHEST EMPHASIS' the rule is: if the leaderboard moves, the prior INVERTS toward smallest credible bolt-on submitted within ~60 minutes. FRONTIER (Q1-Q5) is structurally compatible — Q4 is ~6h end-to-end including paired CUDA verification; in a race window we could pre-stage Q1+Q2+Q3 and trigger Q4+Q5 as the ~60-min credible bolt-on. ASYMPTOTIC (Q6+) is NOT race-compatible — the multiplicative ~3-5× cost makes the bolt-on window 18-30h. If we commit to ASYMPTOTIC and a race fires, we either ship a partial-stack archive (depriving the operator of full-stack score) or ship nothing (May-4 race anti-pattern). FRONTIER preserves race-mode optionality; ASYMPTOTIC consumes it."
  - assumption: "Mission alignment per CLAUDE.md 'Mission alignment — non-negotiable' is BEST SERVED by maximum-score-delta path"
    classification: CARGO-CULTED
    rationale: "Operator-binding standing directive 2026-05-16: 'all in service of innovation and rigor and extreme optimization and performance and score lowering'. This is NOT 'all in service of maximum theoretical score lowering' — it is 'all in service of innovation' AND 'rigor' AND 'extreme optimization' AND 'performance' AND 'score lowering'. The 5-way conjunction favors the path that maximally serves ALL FIVE, not the path that maximizes one (score lowering) at the cost of the others (rigor erosion from premature ASYMPTOTIC commitment without empirical composition_alpha measurement). FRONTIER-with-anchor preserves all 5; ASYMPTOTIC-without-anchor sacrifices RIGOR for SCORE-LOWERING-THEORETICAL."
council_decisions_recorded:
  - "OP-1 [PROCEED]: HORIZON-CLASS scope: HYBRID. Stage 1 (Q1-Q5 FRONTIER_PURSUIT band [0.147, 0.167]) is APPROVED unconditionally per the T3 symposium verdict; this is the minimum-viable-path and preserves race-mode optionality per CLAUDE.md race-mode rigor inversion non-negotiable. Stage 2 (Q6+ ASYMPTOTIC_PURSUIT band [0.050, 0.120]) is DEFERRED-PENDING-EMPIRICAL-ANCHOR per CLAUDE.md 'Forbidden premature KILL without research exhaustion' inverse — we do NOT pre-commit Q6+ without Q4 empirical composition_alpha measurement. The HYBRID structure resolves the Assumption-Adversary's 5th surfaced CARGO-CULTED assumption (binary FRONTIER vs ASYMPTOTIC)."
  - "OP-2 [PROCEED]: Stage-2 reactivation criteria explicit. Q6+ ASYMPTOTIC dispatching is contingent on ALL of: (a) Q4 first empirical anchor lands AND CPU + CUDA paired delta is within ±10% of the L5 codex predicted band [−0.0019, −0.0032]; (b) pre-entropy substrate pivot prober's pairwise composition_alpha is empirically measured (NEW probe: lzma+brotli on (state_dict_a ⊕ state_dict_b) vs sum of marginal lzma; ratio = composition_alpha proxy); (c) operator explicitly approves the T2 + T3 deliberation budget allocation per the UNCLEAR cadence assumption; (d) the per-baker Tier-2 + Tier-3 contest-compliance attestation paths are landed per Yousfi's per-stack compliance overhead."
  - "OP-3 [PROCEED]: Pre-entropy composition_alpha measurement probe — NEW op-routable Q6.preprobe. Extend `tools/pre_entropy_substrate_pivot_prober.py` with a `--pairwise-composition-alpha` mode that for every pair (substrate_i, substrate_j) in the PRE_ENTROPY canon: concatenates the raw byte arrays, runs lzma/brotli/zstd on concat vs sum-of-marginals, computes composition_alpha_ij = (1 - concat_compressed / sum_marginal_compressed). Reuse the canonical fcntl-locked write pattern. Persists to `.omx/state/wyner_ziv_deliverability/pairwise_composition_alpha_<utc>.json`. ~80 LOC delta + 8 tests. ZERO GPU spend; ~2h editor."
  - "OP-4 [PROCEED]: HORIZON-CLASS field on the FEC6 lane registry entry stays `frontier_pursuit` per the symposium's existing horizon_class declaration. Any future Q6+ ASYMPTOTIC lane registration MUST claim a NEW lane_id with `horizon_class: asymptotic_pursuit` AND the reactivation criteria from OP-2 documented in the lane registry notes per Catalog #298 retirement-discipline pattern. This preserves the canonical FEC6 lane semantics."
  - "OP-5 [DEFER_PENDING_EVIDENCE]: Q6+ implementation queue. The council DEFERS specification of Q6+ ASYMPTOTIC sequence (Q6 = pr101_state_dict Tier-2 hoist; Q7 = pr106_state_dict Tier-2; Q8 = posenet_class_sensitivity Tier-3 with operator review; Q9 = stacking integration; Q10 = pairwise verification) pending Q4 anchor landing. Per CLAUDE.md 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' Catalog #315: pre-specifying Q6+ without Q4 evidence would risk dispatch-at-lifted-trainer-form (we don't yet know which substrate composition_alpha actually delivers). The Q4 anchor IS the empirical pre-requisite for Q6+ specification."
  - "OP-6 [PROCEED]: Mission-alignment compliance. council_predicted_mission_contribution=frontier_breaking. The HYBRID structure opens a class-shift path (Q4 first empirically-validated Wyner-Ziv hoist) predicted to lower score from 0.19205 → [0.147, 0.167] in Stage 1, with structured optionality for Stage 2 extending to [0.050, 0.120] contingent on empirical evidence. This satisfies all 5 conjuncts of the operator standing directive (innovation + rigor + extreme optimization + performance + score lowering) by maximizing INNOVATION & RIGOR while preserving SCORE-LOWERING potential."
  - "OP-7 [PROCEED]: No operator-frontier-override invoked. Operator delegated this decision to council per CLAUDE.md 'Council hierarchy: 4-tier protocol' standard operating mode. The HYBRID verdict reserves operator-attention budget for the Q4 empirical anchor verdict (which is the natural Stage-1 → Stage-2 escalation point) per OP-2 reactivation criteria."
  - "OP-8 [PROCEED]: 30-day retrospective scheduling. Per CLAUDE.md 'Mission alignment — non-negotiable' operational consequence 3, this DEFERRED verdict (Stage 2 ASYMPTOTIC deferred-pending-Q4-anchor) gets a 30-day score-impact retrospective scheduled for 2026-06-16. The retrospective evaluates: did Q4 land? did the composition_alpha probe land? did the operator approve Stage 2 dispatch? did Stage 2 deliver the predicted [0.050, 0.120] band? Auto-tracked via `deferred_substrate_retrospective_due_utc` field on this anchor."
related_deliberation_ids:
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
  - feedback_pre_entropy_substrate_pivot_prober_landed_20260517
  - feedback_permanent_fix_frontier_signal_loss_landed_20260517
  - l5_wyner_ziv_rate_only_bound_adversarial_review_20260517_codex
  - falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: "lane_horizon_class_scope_t2_council_20260517_stage_2_asymptotic_pending_q4_anchor"
deferred_substrate_retrospective_due_utc: "2026-06-16T00:00:00Z"
---

# T2 Council Deliberation — HORIZON-CLASS scope for Wyner-Ziv work (2026-05-17)

**Lane:** `lane_horizon_class_scope_t2_council_20260517`
**Tier:** T2 (touches in-flight engineering tradeoffs — Stage 1 vs Stage 2 score-lowering path commitment; binding decision authority per CLAUDE.md "Council hierarchy: 4-tier protocol").
**Quorum:** 6-of-6 sextet pact. All members attended. Quorum met.
**Verdict:** PROCEED_WITH_REVISIONS — HYBRID HORIZON-CLASS scope: FRONTIER_PURSUIT [0.147, 0.167] (Stage 1, Q1-Q5) APPROVED unconditionally; ASYMPTOTIC_PURSUIT [0.050, 0.120] (Stage 2, Q6+) DEFERRED-PENDING-EMPIRICAL-ANCHOR.

## Per-member operating-within assumption statements

Per CLAUDE.md "Council conduct" Fix-7 amendment + Catalog #292: every member states explicit operating-within assumption at the top of their position.

**Shannon LEAD operating-within assumption**: "I am operating within the assumption that Wyner-Ziv 1976 + Shannon R(D|Y) provide the canonical achievable-rate bound for side-info hoisting on per-substrate raw byte sections, AND that the L5 codex review's rate-only bound [−0.0019, −0.0032] is the empirically-confirmed bound for the ~4800-byte poses.bin section. The information-theoretic ceiling per substrate class is bounded by I(X;Y) for the per-substrate byte distribution."

**Dykstra CO-LEAD operating-within assumption**: "I am operating within the assumption that Wyner-Ziv hoist stacking is governed by Pareto-feasibility intersection of (per-substrate rate axis ⊕ archive-byte budget ⊕ inflate.py LOC budget ⊕ scorer-rule compliance), and that composition feasibility requires alternating-projections verification per CLAUDE.md 'Council conduct' canonical role. I do NOT assume additive composition without empirical pairwise interaction measurement."

**Yousfi operating-within assumption**: "I am operating within the assumption that PR #35 + Catalog #146 (contest_one_video_replay) + the 4-tier deliverability taxonomy from the T3 symposium are binding contest-compliance constraints, and that each per-baker Tier-2 / Tier-3 baker requires a separate compliance attestation cycle (Catalog #210 codebook provenance + frozen-weight attestation for Tier 3)."

**Fridrich operating-within assumption**: "I am operating within the assumption that inverse-steganalysis discipline (square-root law + detector-informed embedding + UNIWARD-style spreading) applies to Wyner-Ziv stack compositions in the same form it applies to single hoists — each baker contributes to the SegNet / PoseNet detector blind spots multiplicatively if and only if the bakers are TRULY orthogonal in the scorer's feature space. Stacking orthogonality is empirical, not theoretical."

**Contrarian operating-within assumption**: "I am operating within the assumption that the May-4 race postmortem is the canonical anchor for race-mode strategic decisions (PR105 kitchen_sink 1776 LOC lost to rem2 241 LOC silver in 4h08), AND that absent the operator's explicit race-mode-override the apparatus prefers MINIMUM-VIABLE-PATH with empirical-anchor-before-stacking over MAXIMUM-THEORETICAL-PATH with predicted-anchor-only."

**Assumption-Adversary operating-within assumption**: "I am operating within the assumption that my mandate (per the META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable + CLAUDE.md 'Council conduct' Assumption-Adversary sextet seat) is to surface SHARED ASSUMPTIONS the rest of the council is operating within IMPLICITLY, classify each as HARD-EARNED vs CARGO-CULTED per the hard-earned-vs-cargo-culted addendum, AND propose at least ONE shared-assumption-violation hypothesis with veto power if the consensus doesn't engage. The 5 assumptions surfaced above are my per-round mandate fulfillment."

## Deliberation sub-questions and verdicts

### Sub-question 1: EV per dollar

**Q1-Q5 alone**: predicted ΔS = [0.025, 0.045] @ $0.70 total spend = $0.016-$0.028 per ΔS point per dollar.
**Q6+ ASYMPTOTIC**: predicted ΔS = [0.07, 0.14] @ estimated $10-50 = $0.07-$0.71 per ΔS point per dollar (10× WORSE under multiplicative cost assumption; this is the OPTIMISTIC case — pessimistic with composition_alpha 0.5 SUB-ADDITIVE halves the numerator).

**Verdict**: Q1-Q5 has 10× BETTER EV per dollar in best-case ASYMPTOTIC scenario; 50× BETTER in pessimistic SUB-ADDITIVE scenario. The marginal EV of ASYMPTOTIC is STRUCTURALLY WORSE than Q1-Q5 UNLESS composition_alpha is empirically verified ≥0.9 ADDITIVE.

### Sub-question 2: Composition_alpha realism

**Per Catalog #227 (substrate composition matrix)**: the canonical lattice penalizes 0.3 < α < 0.7 SUB-ADDITIVE by halving predicted ΔS and floors α ≤ 0.3 SATURATING at -0.005.

**Pre-entropy prober marginal evidence**: pr101_state_dict (0.477) + pr106_state_dict (0.470) + posenet_class_sensitivity (11.608) compress as marginals. **Pairwise interaction is UNMEASURED**.

**Assumption-Adversary verdict**: composition_alpha ≈ 1.0 is CARGO-CULTED. The HYBRID verdict's Stage-2 reactivation criterion (OP-2.b) requires empirical pairwise composition_alpha measurement via the OP-3 pre-entropy prober extension.

### Sub-question 3: HNeRV parity L4 budget

**Per HNeRV parity discipline L4**: inflate.py ≤ 100 LOC (default budget; explicit waiver for ≤ 200 with rationale).

**Q1-Q5 Q4 packet**: per the symposium's optimal-design Component 1, Tier 2 Comma2k19 palette ≤ 200 LOC inflate-runtime via baked constants. Within waiver.

**Q6+ stacking**: 3 stacked bakers × ~80 LOC each = ~240 LOC; 5 stacked = ~400 LOC. EXCEEDS waiver. Requires either (a) additional per-baker waiver landings (operator-attention cost) OR (b) the pipeline-stage WZ codec primitive (#814) compressing baked constants into shared lookup tables (NEW design work; pre-Q4 NOT council-approved).

**Carmack-arithmetic-revisited**: ~25-50 KB compressed-into-constants is the realistic inflate.py-deliverable side-info ceiling per Carmack's verbatim in the T3 symposium. ASYMPTOTIC's per-substrate posenet_class_sensitivity (20 MB raw) → 11.6 deliverable savings claim is CARGO-CULTED at the inflate.py LOC budget surface.

### Sub-question 4: Operator-attention budget

**Per CLAUDE.md "Council hierarchy: 4-tier protocol"**: T2 ≤3/day, ≤90/30d. T3 ≤3/week, ≤13/30d.

**Q1-Q5**: 0 additional T2 / T3 deliberations (the T3 symposium is the only deliberation; Q1-Q5 are subagent dispatches).

**Q6+ ASYMPTOTIC**: per Yousfi's per-stack compliance cost analysis, 3-5 additional T2 deliberations (one per baker design) + 1-2 T3 escalations (each Tier 3 frozen-weight attestation crosses CLAUDE.md non-negotiable scope).

**Verdict**: within budget for Q6+ if spread across 3-5 days at T2, consuming ~15-30% of 30-day T3 envelope. UNCLEAR per Assumption-Adversary (whether ASYMPTOTIC deserves 15-30% of T3 vs other paradigm-level work is OPERATOR-decidable).

### Sub-question 5: Race-mode applicability

**Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first — NON-NEGOTIABLE, HIGHEST EMPHASIS"**: if leaderboard moves, the prior INVERTS toward smallest credible bolt-on submitted within ~60 minutes.

**Q1-Q5 / FRONTIER**: race-compatible. Q4 is ~6h end-to-end including paired CUDA. With Q1+Q2+Q3 pre-staged, Q4+Q5 can fire in a ~60-min credible bolt-on window.

**Q6+ / ASYMPTOTIC**: NOT race-compatible. Multiplicative ~3-5× cost makes the bolt-on window 18-30h. If we commit to ASYMPTOTIC and a race fires, we either ship partial-stack (suboptimal score) or ship nothing (May-4 race anti-pattern — PR105 kitchen_sink lost to rem2 silver).

**Verdict**: FRONTIER preserves race-mode optionality; ASYMPTOTIC consumes it. **This is decisive in the HYBRID structure**: Stage 1 (Q1-Q5) is race-compatible and unconditionally approved; Stage 2 (Q6+) is non-race-compatible and deferred per OP-2 reactivation criteria.

## Verdict in canonical taxonomy

Per CLAUDE.md "Council hierarchy: 4-tier protocol" canonical VALID_VERDICTS = {PROCEED, PROCEED_WITH_REVISIONS, DEFER_PENDING_EVIDENCE, REFUSE, ESCALATE_TO_OPERATOR, ESCALATE_TO_HIGHER_TIER}.

**Mapping**: the operator-requested verdict vocabulary (PROCEED_FRONTIER / PROCEED_ASYMPTOTIC / PROCEED_FRONTIER_WITH_OPTIONAL_ASYMPTOTIC / DEFER_PENDING_EVIDENCE / REFUSE / ESCALATE_TO_HIGHER_TIER) maps to the canonical taxonomy as: HYBRID = PROCEED_WITH_REVISIONS (canonical name) with 8 op-routables encoding the 2-stage structure. Stage 1 = PROCEED (Q1-Q5 unconditional). Stage 2 = DEFER_PENDING_EVIDENCE (Q6+ contingent on OP-2 reactivation criteria).

The PROCEED_WITH_REVISIONS canonical verdict + OP-1 explicit HYBRID-structure declaration encodes the full operator-requested PROCEED_FRONTIER_WITH_OPTIONAL_ASYMPTOTIC semantics within the canonical schema.

## Top-3 dissent verbatims (per maximum signal preservation rule)

Per CLAUDE.md "Council hierarchy: 4-tier protocol" maximum-signal preservation rule + Catalog #300 v2 contract: dissent preserved verbatim above in the frontmatter (`council_dissent` field). Summary:

1. **Contrarian**: REFUSE unconditional ASYMPTOTIC commitment; MINIMUM-VIABLE-PATH is FRONTIER + empirical anchor + re-evaluate. Resolved by HYBRID verdict.
2. **Yousfi**: per-stack contest-compliance cost is structurally O(#stacks) at operator-attention surface; not vetoing ASYMPTOTIC but noting cost. Resolved by OP-2.d explicit contest-compliance attestation precondition.
3. **Assumption-Adversary**: 5 shared assumptions surfaced; 4 CARGO-CULTED + 1 UNCLEAR + 1 HARD-EARNED. The 5th (binary FRONTIER vs ASYMPTOTIC) IS the verdict-structure cargo-cult that the HYBRID structure resolves.

## Operator-routable implementation queue IF ASYMPTOTIC chosen (Stage 2 contingent)

Per OP-5 the Q6+ queue is DEFERRED until Q4 anchor lands AND OP-2 reactivation criteria are met. Pre-specifying here for operator visibility per CLAUDE.md "Required durable state":

| # | Lane (proposed) | LOC | Tests | Wall-clock | GPU $ | Dependencies |
|---|---|---|---|---|---|---|
| Q6.preprobe | `lane_pre_entropy_pairwise_composition_alpha_probe_20260517` | ~80 delta | 8 | ~2h | $0 | OP-3 (NEW preprobe; sister of pre-entropy pivot prober) |
| Q6 | `lane_pr101_state_dict_tier_2_wyner_ziv_hoist_20260517` | ~280 | 22 | ~6h | ~$0.70 paired | Q4 anchor + Q6.preprobe |
| Q7 | `lane_pr106_state_dict_tier_2_wyner_ziv_hoist_20260517` | ~280 | 22 | ~6h | ~$0.70 paired | Q6 (pairwise composition validation) |
| Q8 | `lane_posenet_class_sensitivity_tier_3_wyner_ziv_hoist_20260517` | ~400 | 28 | ~8h | ~$0.70 paired | Q7 + operator review (Tier 3 frozen-weight attestation) |
| Q9 | `lane_q6_q8_three_substrate_stacking_integration_20260517` | ~150 | 14 | ~6h | ~$0.70 paired (CUDA full-stack) | Q6+Q7+Q8 individually verified |
| Q10 | `lane_q9_stacked_archive_pairwise_verification_via_l5_protocol_20260517` | ~80 | 6 | ~4h | ~$0.40 paired (verification only) | Q9 |
| Q11 | `lane_lane_registry_horizon_class_asymptotic_lane_registration_20260517` | ~30 | 4 | ~30min | $0 | Q10 |

**Total (Q6+)**: ~1300 LOC + 104 tests + ~32h wall-clock + ~$3.20 GPU spend. **Returns**: ASYMPTOTIC_PURSUIT band [0.050, 0.120] WITH empirically-verified composition_alpha + per-baker contest-compliance attestation.

**Stage 1 (Q1-Q5) per T3 symposium**: ~520 LOC + 59 tests + ~10.25h + ~$0.70.
**Stage 2 (Q6-Q11)**: ~1300 LOC + 104 tests + ~32h + ~$3.20.
**HYBRID total (Stage 1 + IF Stage 2 unlocked)**: ~1820 LOC + 163 tests + ~42.25h + ~$3.90.

**3-5× multiplicative cost from Stage 1 → Stage 2 confirmed**; per Sub-question 1 EV-per-dollar analysis Stage 1 has 10× BETTER EV per dollar in best case.

## HORIZON-CLASS recommended scope

**HYBRID** — FRONTIER_PURSUIT [0.147, 0.167] unconditional (Stage 1, Q1-Q5); ASYMPTOTIC_PURSUIT [0.050, 0.120] DEFERRED-PENDING-EMPIRICAL-ANCHOR (Stage 2, Q6-Q11; reactivation criteria per OP-2).

This is structurally distinct from FRONTIER-only (commits to Stage 1; closes Stage 2) and ASYMPTOTIC-only (pre-commits Stage 1 + Stage 2). It is also structurally distinct from REFUSE (kills both stages). It is the canonical PROCEED_WITH_REVISIONS verdict in operator-requested vocabulary.

## Op-routables for Q6+ stacking (IF council proceeds beyond Q5)

Per OP-2 reactivation criteria, Q6+ proceeds when ALL of:
1. **Q4 first empirical anchor lands** with CPU + CUDA paired delta within ±10% of L5 codex predicted band [−0.0019, −0.0032].
2. **Pairwise composition_alpha empirically measured** via OP-3 pre-entropy prober extension (`lane_pre_entropy_pairwise_composition_alpha_probe_20260517`); minimum α ≥ 0.7 ADDITIVE across at least 2-of-3 candidate pairs for Q6+ to be authorized.
3. **Operator explicitly approves** the T2 + T3 deliberation budget allocation per the UNCLEAR cadence assumption.
4. **Per-baker Tier-2 + Tier-3 contest-compliance attestation paths landed** per Yousfi's per-stack compliance overhead concern (Catalog #210 codebook provenance + Catalog #271 codex pre-dispatch review per Tier-3 substrate).

Failure of ANY criterion → Stage 2 stays DEFERRED. Operator may invoke `override_invoked=true` to bypass criteria per CLAUDE.md "Mission alignment — non-negotiable" operational consequence 1, but this council has not invoked it.

## 30-day retrospective schedule

Per CLAUDE.md "Mission alignment — non-negotiable" operational consequence 3 + CouncilDeliberationRecord `deferred_substrate_retrospective_due_utc` field: 2026-06-16T00:00:00Z. The retrospective evaluates: did Q4 land? did the composition_alpha probe land? did the operator approve Stage 2 dispatch? did Stage 2 deliver predicted [0.050, 0.120] band?

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Verdict schema | ADOPT canonical (PROCEED_WITH_REVISIONS) | The HYBRID semantics fit within canonical VALID_VERDICTS via 8 op-routables; no need to fork the schema. |
| OP-2 reactivation criteria | UNIQUE (4-clause precondition) | Stage 2 reactivation is specific to this verdict; not a generalizable pattern. |
| OP-3 pre-entropy pairwise prober | ADOPT canonical (extends `tools/pre_entropy_substrate_pivot_prober.py`) | The pre-entropy prober is the canonical surface for substrate compression analysis; extension is OBVIOUS-FIT per Catalog #290 falling-rule cascade. |
| Q6+ implementation queue (pre-specification) | UNIQUE (this verdict's enumeration) | Q6-Q11 sequencing is specific to the pre-entropy prober's top-3 substrate identification; not a generalizable pattern. |
| Continual-learning anchor | ADOPT canonical (`tac.council_continual_learning.append_council_anchor`) | Catalog #300 v2 contract; no fork. |
| Deferred-retrospective scheduling | ADOPT canonical (CouncilDeliberationRecord paired fields) | Sister discipline per Catalog #300 mission-alignment extension. |
| HORIZON-CLASS field semantics | ADOPT canonical (Catalog #309) | Per CLAUDE.md HORIZON-CLASS standing directive; the HYBRID verdict declares Stage 1 as `frontier_pursuit` and pre-specifies Stage 2 as `asymptotic_pursuit` per the canonical enum. |

## Cargo-cult audit per assumption

Per Catalog #303: surfaced 5 assumptions in `council_assumption_adversary_verdict`. Classifications:

| Assumption | Classification | Unwind path |
|---|---|---|
| 4-tier deliverability taxonomy generalizes from fec6 to pr101 / pr106 / posenet | CARGO-CULTED | OP-2.a + OP-3 — empirical anchor + pairwise composition_alpha probe before Stage 2 |
| Pre-entropy substrate hoists compose additively | CARGO-CULTED | OP-3 — empirical pairwise composition_alpha measurement (Catalog #227 sister discipline) |
| Operator-attention budget supports Q6+ | UNCLEAR | OP-2.c — operator explicit approval per the UNCLEAR cadence assumption |
| Race-mode rigor inversion applies symmetrically | HARD-EARNED | DECISIVE for HYBRID verdict: FRONTIER preserves race-mode optionality, ASYMPTOTIC consumes it |
| FRONTIER vs ASYMPTOTIC is binary | CARGO-CULTED | OP-1 — HYBRID verdict structure |
| Mission alignment best served by max-score-delta path | CARGO-CULTED | The 5-way conjunction (innovation + rigor + extreme optimization + performance + score lowering) favors HYBRID over pure-score-lowering ASYMPTOTIC |

## Predicted ΔS band (with Dykstra-feasibility framing per Catalog #296)

**Stage 1 (FRONTIER_PURSUIT, Q1-Q5)**:
- Dykstra-feasibility check: intersection of (per-substrate rate-axis savings ⊕ archive-byte budget ≤200KB ⊕ inflate.py LOC ≤200 ⊕ scorer-rule + Yousfi PR #35) → FEASIBLE per T3 symposium Component 3 empirical verification protocol.
- Predicted ΔS contest-CPU: **[−0.025, −0.045]** per T3 symposium "predicted ΔS band" section.
- First-principles citation: Wyner-Ziv 1976 + Shannon R(D|Y) + Atick-Redlich 1990 cooperative-receiver.

**Stage 2 (ASYMPTOTIC_PURSUIT, Q6-Q11, contingent)**:
- Dykstra-feasibility check: intersection of (3-5 stacked substrate hoists ⊕ archive-byte budget ⊕ inflate.py LOC ≤200 WAIVER-REQUIRED ⊕ scorer-rule + per-stack Yousfi PR #35 + per-stack Catalog #210 + per-stack Catalog #213 + per-Tier-3 operator review) → UNCERTAIN-FEASIBLE pending empirical composition_alpha measurement per OP-3.
- Predicted ΔS contest-CPU under composition_alpha = 0.9 ADDITIVE: **[−0.07, −0.14]**.
- Predicted ΔS contest-CPU under composition_alpha = 0.5 SUB-ADDITIVE (Catalog #227 halving): **[−0.035, −0.07]**.
- Predicted ΔS contest-CPU under composition_alpha = 0.2 SATURATING: **floor at −0.005**.
- First-principles citation: same as Stage 1 + Catalog #227 substrate composition matrix.

**Stacked HYBRID best-case (Stage 1 + verified Stage 2 with α ≥ 0.9)**: ΔS = [−0.095, −0.185]; FEC6 0.19205 → [0.007, 0.097]. Crosses INTO ASYMPTOTIC band cleanly.

**Stacked HYBRID worst-case (Stage 1 only; Stage 2 falsified by α < 0.5)**: ΔS = [−0.025, −0.045]; FEC6 0.19205 → [0.147, 0.167]. Cleanly lands in FRONTIER band.

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable":

1. **Inspectable per layer** — Stage 1 / Stage 2 boundary is queryable via the deferred_substrate_id field + OP-1 verdict body. Per-substrate composition_alpha (when measured) lives at `.omx/state/wyner_ziv_deliverability/pairwise_composition_alpha_<utc>.json`.
2. **Decomposable per signal** — Stage 1 + Stage 2 deltas separable per OP-1; per-substrate per-tier deltas separable per the Q6-Q11 lane registry entries.
3. **Diff-able across runs** — every dispatch (Q4 / Q6 / Q7 / Q8 / Q9) writes to Modal call_id ledger per Catalog #245; pairwise composition_alpha is APPEND-ONLY JSONL per Catalog #131.
4. **Queryable post-hoc** — `tac.council_continual_learning.query_anchors_by_topic("horizon_class_scope")` returns this anchor + future amendments.
5. **Cite-able** — every Stage 1 / Stage 2 dispatch carries `related_deliberation_ids` pointing back to this anchor.
6. **Counterfactual-able** — operator can override OP-2 reactivation criteria via `override_invoked=true`; the override is recorded in the anchor body + retrospective.

## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | HYBRID verdict structure is unique to this deliberation; resolves the Assumption-Adversary's 5th CARGO-CULTED assumption (binary FRONTIER vs ASYMPTOTIC). |
| 2. BEAUTY + ELEGANCE | 8 op-routables; 6 assumption classifications; 6 dissent verbatim; reviewable in 30 seconds. |
| 3. DISTINCTNESS | Distinct from T3 symposium PROCEED_WITH_REVISIONS (which committed to Q1-Q5 unconditional without Stage-2 specification). |
| 4. RIGOR | 5 per-member operating-within assumption statements per Catalog #292; 6 Assumption-Adversary classifications per Catalog #300 v2; 5 sub-questions with explicit verdicts; 3 dissent verbatims. |
| 5. OPTIMIZATION PER TECHNIQUE | Per-stage cost/EV breakdown; per-stage race-mode compatibility analysis; per-stage Dykstra-feasibility framing. |
| 6. STACK-OF-STACKS-COMPOSABILITY | OP-3 pre-entropy pairwise prober EXPLICITLY measures composition_alpha for stacking; Stage 2 reactivation is GATED on α ≥ 0.7 ADDITIVE. |
| 7. DETERMINISTIC REPRODUCIBILITY | All Stage 2 op-routables Q6-Q11 carry lane_ids per Catalog #126 pre-registration discipline; reactivation criteria are explicit; retrospective scheduled per Catalog #300. |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | EV-per-dollar analysis: Stage 1 = $0.016-0.028/ΔS-point/$; Stage 2 best-case = $0.07-0.71/ΔS-point/$ (10× worse minimum). |
| 9. OPTIMAL MINIMAL CONTEST SCORE | HYBRID best-case [0.007, 0.097] crosses ASYMPTOTIC band cleanly IF α ≥ 0.9 ADDITIVE; HYBRID worst-case [0.147, 0.167] cleanly lands FRONTIER. |

## horizon_class: frontier_pursuit (Stage 1) → asymptotic_pursuit (Stage 2, contingent)

Per CLAUDE.md HORIZON-CLASS standing directive Catalog #309: Stage 1 = `frontier_pursuit` [0.147, 0.167]; Stage 2 (contingent per OP-2) = `asymptotic_pursuit` [0.050, 0.120]. The HYBRID structure preserves the canonical enum semantics.

## Mission alignment per CLAUDE.md "Mission alignment — non-negotiable"

`council_predicted_mission_contribution: frontier_breaking` — the HYBRID structure opens a class-shift path (Q4 first empirically-validated Wyner-Ziv hoist) predicted to lower score from 0.19205 → [0.147, 0.167] in Stage 1, with structured optionality for Stage 2 extending to [0.050, 0.120] contingent on empirical evidence per OP-2. This satisfies all 5 conjuncts of the operator standing directive 2026-05-16 (innovation + rigor + extreme optimization + performance + score lowering) by maximizing INNOVATION & RIGOR while preserving SCORE-LOWERING potential.

`council_override_invoked: false` — operator delegated this decision to council per CLAUDE.md "Council hierarchy: 4-tier protocol" standard operating mode.

## Cross-references

- CLAUDE.md "Council hierarchy: 4-tier protocol" — this is T2.
- CLAUDE.md "Mission alignment — non-negotiable" — operational consequences 1, 3, 5.
- CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons 2 + 4 + 7
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" Catalog #315
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md HORIZON-CLASS standing directive Catalog #309
- Catalog #227 (substrate composition matrix; composition_alpha penalty)
- Catalog #292 (per-deliberation explicit assumption surfacing)
- Catalog #295 (submission inflate.py works with empty PYTHONPATH)
- Catalog #298 (substrate retirement discipline 30-day staleness window)
- Catalog #300 (council deliberation v2 frontmatter + mission-alignment extension)
- Catalog #309 (HORIZON-CLASS declaration)
- Catalog #313 (predecessor-adjudicated outcome blocks dispatch)
- Catalog #316 (reports/latest.md not stale vs canonical frontier)
- Catalog #318 (planned in symposium; Venn reweight requires deliverability proof — Q2 of T3 symposium queue)
- `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` — T3 parent symposium verdict
- `.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md` — Q1-Q5 sequencing
- `.omx/research/comprehensive_state_tracker_20260517.md` — in-flight state
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pre_entropy_substrate_pivot_prober_landed_20260517.md` — empirical pivot evidence
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_permanent_fix_frontier_signal_loss_landed_20260517.md` — Catalog #316 frontier signal loss fix
- `.omx/research/l5_wyner_ziv_rate_only_bound_adversarial_review_20260517_codex.md` — L5 codex review rate-only band
- `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` — HORIZON-CLASS Pattern G (F-asymptote class-shift discipline)

## End of T2 deliberation memo
