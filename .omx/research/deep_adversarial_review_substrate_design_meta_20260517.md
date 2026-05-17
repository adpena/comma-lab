---
title: "DEEP ADVERSARIAL REVIEW — META: are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192?"
date: 2026-05-17
lane: lane_deep_adversarial_review_substrate_design_meta_20260517
author: deep_adversarial_review_substrate_design_meta_subagent_20260517
horizon_class: apparatus_maintenance
council_tier: T3
council_attendees:
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Boyd (grand council; convex-feasibility specialist seat for this deliberation)
  - Tao (grand council; mathematical-omniscience specialist seat for this deliberation)
  - Carmack (grand council; engineering-shortcut specialist seat for this deliberation)
  - Selfcomp (grand council; canonical PR#56 winning paradigm context)
  - MacKay (grand council memorial; information-theory + MDL specialist)
  - Ballé (grand council; modern neural-compression SOTA specialist)
council_quorum_met: true
council_verdict: ESCALATE_TO_HIGHER_TIER
council_predicted_mission_contribution: mission_questioned
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "I do not accept Option A (the corpus is sound; 4-of-4 failures are coincidence). The 4-of-4 architecturally-distinct probe failures plus the structural pattern that NO substrate in 53 has produced a sub-0.192 [contest-CPU] anchor while A1 (the frontier) is itself a deferral-as-promotion (inflate-time bias correction on a verified PR95-paradigm substrate from May 2026, not a NEW class-shift) IS the empirical anchor that the META-assumption is CARGO-CULTED. The session's pattern of 'build a class-shift substrate; build a probe-disambiguator for it; the probe says INDEPENDENT/ARTIFACT/IDENTITY-TIES; defer; build the next class-shift substrate' is itself the failure mode — we have built the apparatus to AVOID burning paid GPU on dead distinguishing features (which is HARD-EARNED), but we have NOT yet asked WHY every distinguishing feature we design is dead at the contest scorer's response surface."
  - member: Assumption-Adversary
    verbatim: "I challenge the SHARED ASSUMPTION operating across this entire substrate-design-corpus: that the SegNet stride-2 EfficientNet-B2 + PoseNet FastViT-T12 12-channel YUV6 scorer's response surface admits ARCHITECTURAL class-shifts that distinguish them via per-pair-of-frames distinguishing-feature design. The 4-of-4 empirical pattern is HARD-EARNED EVIDENCE that this assumption is FALSE for the distinguishing-feature design space we have been exploring. Per-pair conditioning signals (Wyner-Ziv side-information / cooperative-receiver class / ego-motion FiLM / wire-grammar class CDF / wavelet residual) all collapse to identity or degenerate at the contest scorer's stride-2 stem because the scorer's gradient surface is dominated by per-frame structure that the encoder/decoder learn implicitly without explicit per-pair conditioning. The CARGO-CULTED assumption is that per-pair distinguishing-features are score-shifting; the HARD-EARNED reading is that the scorer's response surface is per-frame and per-pair signals collapse into noise. NEW META-assumption surfaced: every substrate in the corpus that derives its distinguishing-feature from per-pair conditioning is at HIGH RISK of similar failure. This is approximately 35-of-53 substrates."
  - member: Boyd
    verbatim: "Per convex-feasibility lens: the 4-of-4 failures share a structural pattern that the predicted-band convex polytopes were valid at the MATHEMATICAL construction level (Atick-Redlich + Rao-Ballard + Tikhonov + Wyner-Ziv all have non-empty intersection at the theorem level) but DEGENERATE at the empirical projection onto the contest scorer's response surface. This is exactly the Z3-G1 cargo-cult-prediction pattern documented in the FALSIFICATION-AUDIT-v2 Pattern D (paradigm-vs-implementation conflation) — the paradigm is theoretically intact but the implementation projection onto the scorer is empty. Per Dykstra-feasibility intersection at each lattice level per Catalog #296: we have NOT been checking the EMPIRICAL projection feasibility, only the theoretical-construction feasibility. Recommendation: every NEW substrate design memo must add an EMPIRICAL-PROJECTION-FEASIBILITY check distinct from the theoretical-construction Dykstra check."
  - member: Tao
    verbatim: "Per mathematical-omniscience lens: the contest scorer's response surface (SegNet stride-2 EfficientNet-B2 + PoseNet FastViT-T12 12-channel YUV6) has a specific harmonic-analysis structure that the per-pair distinguishing-feature designs ignore. SegNet's stride-2 stem applies low-pass filtering at the input layer; class boundaries in the output are dominated by per-frame spatial frequency content; per-pair conditioning signals appear at the difference frequencies which the stride-2 stem attenuates. PoseNet's FastViT-T12 12-channel YUV6 input processes 2 frames × YUV6 jointly but the attention mechanism is per-token (token = patch of one frame); cross-frame conditioning at the SegNet/PoseNet output level requires the conditioning signal to survive both the stride-2 attenuation (SegNet) and the per-token attention bottleneck (PoseNet). The 4-of-4 failures empirically validate this harmonic-analysis prediction. Per the additive-combinatorics lens: the dimensionality of useful per-pair conditioning under these two scorer constraints is approximately log(num_distinct_pair_types) — for 600 dashcam pairs this is at most ~9 bits/pair of useful signal, NOT the ~100s of bits per-pair-class-histogram or per-region-class-histogram designs imply. Most of our distinguishing-feature bits are wasted on signals the scorer cannot see."
council_assumption_adversary_verdict:
  - assumption: "The PR95-paradigm bind-all-ingredients pattern is sufficient for class-shift to sub-0.192 [contest-CPU]"
    classification: CARGO-CULTED
    rationale: "PR95-paradigm bind-all-ingredients IS the canonical engineering discipline for shipping a complete substrate (HARD-EARNED for engineering discipline). BUT engineering discipline is NECESSARY-NOT-SUFFICIENT for score-shifting. PR95/PR100/PR101/PR102/PR103 winners ALL bound the ingredients; they ALSO chose substrates where the distinguishing feature MATCHED the scorer's gradient surface (HNeRV decoder + per-pair learned 28-d latent are entirely per-frame structure with per-pair latent being a regularization not a conditioning bottleneck). Our 53 substrates apply the PR95 discipline correctly but to distinguishing-features that DO NOT match the scorer's gradient surface (cooperative-receiver / Wyner-Ziv side-info / ego-motion / wire-grammar / etc. all rely on per-pair conditioning the scorer attenuates). The CARGO-CULTED reading: PR95-paradigm pattern guarantees engineering correctness; it does NOT guarantee score-shifting potency."
  - assumption: "The distinguishing-feature contract per Catalog #272 captures actual score-shifting capacity"
    classification: CARGO-CULTED
    rationale: "Catalog #272 requires that distinguishing bytes be CONSUMED by inflate AND produce frame changes (HARD-EARNED for byte-mutation no-op detection). BUT 'produces frame changes' is NECESSARY-NOT-SUFFICIENT for score changes. Frame changes the scorer attenuates do NOT lower contest score. The Z3-G1 anchor (0.19869 EXACTLY matching Z3 v2 baseline 0.19869 to 5 decimals when bytes were empty) is the canonical case where distinguishing bytes were absent. The OPPOSITE case — distinguishing bytes PRESENT, consumed, producing frame changes the scorer attenuates — is the bug class Catalog #272 does NOT catch. The 4-of-4 today demonstrate this: ATW v2 D4 probe MI=0.006 (the distinguishing feature is mathematically present but conveys 0.006 bits/symbol of useful information); Z6 identity-predictor essentially ties full-FiLM at max delta 5.3e-6 (the FiLM modulation IS exercised — gradients flow, capacity is live — but the scorer doesn't measure the difference). Distinguishing-feature INTEGRATION is engineering-validated; distinguishing-feature INFORMATIVENESS at the scorer is NOT."
  - assumption: "The 4 lattice rules (chroma-preserving / nullspace-split-PR95 / Dykstra-validated-stack / Daubechies-wavelet-multi-scale) cover the space of viable class-shifts"
    classification: PARTIALLY HARD-EARNED + CARGO-CULTED
    rationale: "HARD-EARNED at the falling-rule-list classification level (Wang-Rudin 2015 first-match-wins discipline). BUT the 5 rules carve the substrate space along the dimensions our team can think to design substrates in. The empirical 4-of-4 failures suggest the COMPLEMENT of these 4 rules (substrates that do NOT fit any of Rules 1-4) is where actual class-shifts to sub-0.192 may live. Rule 5 (REQUEST_OPERATOR_REVIEW) is the canonical whiteboard for this, but it has 0 entries in 53 lattice nodes — we have not been using the whiteboard. CARGO-CULTED for the assumption that Rules 1-4 are sufficient; HARD-EARNED that they are necessary first-pass classification."
  - assumption: "Substrate-trainer engineering quality (Catalog #270 Tier 1/2/3 umbrella) predicts score-shifting quality"
    classification: CARGO-CULTED
    rationale: "Catalog #270 Tier 1/2/3 is engineering hygiene (autocast / TF32 / torch.compile / no_grad / canonical scorer-loss helper / NVML env block / canonical auth-eval / etc.) — HARD-EARNED for engineering discipline; PREVENTS bug classes that wasted $1000+ in dispatch costs prior to landing. BUT engineering discipline at the trainer level is ORTHOGONAL to score-shifting potency at the architecture level. Z3-G1 had perfect engineering hygiene + zero score-shifting. NSCS06 v8 Path B had perfect engineering hygiene + 547× outside-band score. Engineering quality is necessary-not-sufficient. The CARGO-CULTED reading: by perfecting engineering hygiene we have built confidence that we are shipping well-engineered substrates without separately validating that the substrates are well-designed at the distinguishing-feature level."
  - assumption: "The cargo-cult-unwind methodology generalizes iteratively across substrates (v6→v7 44% improvement → v8 → v9 → ... → asymptotic floor)"
    classification: CARGO-CULTED
    rationale: "HARD-EARNED for NSCS06 v6 → v7 (105.15 → 58.89; 44% improvement; 4-of-7 cargo-cults unwound). UNVERIFIED beyond v7: NSCS06 v8 Path B returned 104.98 [diagnostic-CPU] — the trajectory's slope FLIPPED from improving to regressing in one iteration. The cargo-cult-unwind methodology is NOT a guaranteed-monotonic improvement function; it is iterative-cargo-cult-discovery-and-test which can produce regression as easily as improvement. The CARGO-CULTED reading: extrapolating v7's 44% improvement to v9 + v10 + v11 producing geometric improvements is wishful thinking; the empirical v8 regression to 104.98 is the receipt that disproves it."
  - assumption: "The contest scorer's response surface is monotonically improvable from A1's 0.19285 baseline via more class-shift substrates"
    classification: CARGO-CULTED
    rationale: "The empirical evidence after 53 substrates: ZERO have produced sub-0.192 [contest-CPU]. PR101 hit 0.193 [contest-CUDA] at gold medal (May 2026); PR100 anchored at 0.195 [contest-CPU]; A1 at 0.19285 [contest-CPU] (CARMACK-HOTZ inflate-time bias correction on a PR95 substrate — itself a within-class refinement; not a class-shift). The empirical cluster from May 2026 is the 0.193-0.197 leaderboard range; our cluster is 0.196-0.199 (with A1's 0.19285 being a within-class refinement of the PR95 May 2026 substrate). The CARGO-CULTED reading: assuming the leaderboard winners haven't already approached the plateau-class asymptotic floor and that there is significant frontier-pursuit headroom is unsubstantiated. The HARD-EARNED reading: 0.193 is approximately the within-PR95-paradigm-class floor; getting to sub-0.150 (frontier-pursuit territory per HORIZON-CLASS) or sub-0.10 (asymptotic-pursuit territory) requires class-shift beyond what 53 substrates have explored."
  - assumption: "Probe-disambiguators measure what we think they measure (substrate viability at the scorer level)"
    classification: CARGO-CULTED
    rationale: "Probe-disambiguators per Catalog #313 measure SPECIFIC properties: ATW v2 D4 probe measures `MI(latent;scorer_class)`; Wunderkind G1 v2 probe measures `per_pair_class_entropy`; Z6 probe measures `loss-proxy delta between full-FiLM and identity-predictor`. These probes are MEASURING THE THINGS THE SUBSTRATE-DESIGN ASSUMES. They are NOT measuring whether the contest scorer's gradient surface admits the substrate's class-shift. The probe-methodology-as-false-falsification pattern (FALSIFICATION-AUDIT-v2 Pattern E) is exactly this — Wunderkind G1 v2 T2 council SPLIT-VERDICT documented that the per-pair-dominant SegNet argmax reducer is ARTIFACT of probe methodology not substrate property. CARGO-CULTED: the probe-design choice itself encodes the same assumption as the substrate-design choice; passing one probe does not validate the substrate at the scorer level."
  - assumption: "Class-shift = architectural-class-shift (vs decode-time-contract-shift / scorer-relationship-shift / training-time-paradigm-shift / wire-grammar-shift)"
    classification: HARD-EARNED for the multi-axis definition + CARGO-CULTED for the practical impact
    rationale: "The Path 2 LATTICE memo explicitly defines class-shift as differing on at least ONE of (architecture / decode-time-contract / training-time-paradigm / wire-grammar / scorer-relationship). HARD-EARNED that this is the canonical multi-axis class-shift framework. CARGO-CULTED that all 5 axes have equivalent score-shifting potency at the contest scorer's response surface. Empirical evidence: the 4-of-4 failures explored 4 different axes (NSCS06 = architecture + wire-grammar; ATW v2 = scorer-relationship; Wunderkind G1 v2 = wire-grammar; Z6 = scorer-relationship via predictive-coding) and ALL failed. This empirically suggests that the axes themselves may not be equivalent — possibly only ONE axis (the one PR95-paradigm winners exploited: per-frame-renderer + per-pair-latent regularization) has class-shift potency in the current scorer response surface."
council_decisions_recorded:
  - "DEEP ADVERSARIAL REVIEW VERDICT: composite per Section 5 (B+C+D+E+F). The 4-of-4 empirical pattern is NOT coincidence; it indicates structural META-class failure of our substrate-design corpus to address the contest scorer's actual response surface."
  - "Section 2 META-assumption classification: 7 CARGO-CULTED + 1 PARTIALLY HARD-EARNED + 0 fully HARD-EARNED across the 8 candidate META-assumptions enumerated by the parent prompt + my own additions. The session's substrate-design corpus operates on a backdrop of 8-of-8 unproven assumptions about the contest scorer's response surface."
  - "Substrate-cluster risk classification (Section 3): of 53 substrates, ~35 share the per-pair-conditioning distinguishing-feature class with one or more of the 4 failed probes; HIGH RISK that paid dispatch on these substrates produces the same failure mode."
  - "Section 4 missing class-shift opportunities: 4 substrate-design surfaces NOT in current corpus surface — (a) per-frame-renderer-axis class-shifts (the PR95-paradigm winning axis); (b) scorer-architecture-rewriting (replace SegNet/PoseNet preprocessing path; only Lane G v3 1.05 explored this); (c) renderer-quantization-aware-codec like Quantizr 0.33 leader; (d) decoder-FLOPs-budget class-shifts (HNeRV decoders at 229K params dominate at constant byte cost)."
  - "T4 SYMPOSIUM ESCALATION RECOMMENDED per Catalog #300 elevation triggers (b) recusal-drops-quorum-on-key-question AND (c) ≥3 grand-council members invoke specialist-disagreement: Boyd + Tao + Assumption-Adversary all surfaced specialist disagreements; ALL 8 META-assumptions classified CARGO-CULTED or PARTIALLY; the session's substrate-design corpus is at structural risk of producing 53-of-53 dead substrates if pursued without operator review of META-assumption #1 (PR95-paradigm is sufficient for class-shift)."
  - "Operator-routable decision queue: 5 explicit operator decisions surfaced in Section 6; each is COUNCIL-GRADE per CLAUDE.md 'Design decisions — non-negotiable' (cannot be made unilaterally by agent). T4 SYMPOSIUM is the canonical forum for adjudicating these."
  - "Catalog #291 META-ASSUMPTION ADVERSARIAL REVIEW cadence: this memo IS the second instance after `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (2 days ago). Per Catalog #291 the cadence is satisfied for the next 7 days OR 50 subagent landings, whichever first."
  - "30-day deferred-substrate retrospective scheduled 2026-06-17 for the META-assumption #1-#7 classifications (per Catalog #300 mission-alignment consequence 3; this is a paradigm-level deferral verdict that needs empirical-anchor re-validation in 30 days)."
deferred_substrate_retrospective_due_utc: "2026-06-17T11:55:00Z"
deferred_substrate_id: "substrate_design_corpus_per_pair_conditioning_class_shift_meta_assumption"
related_deliberation_ids:
  - feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515
  - falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516
  - grand_council_symposium_time_traveler_optimal_staircase_20260516
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
  - sextet_council_z6_phase_2_consensus_20260516
  - grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516
  - atw_d4_probe_recipe_disambiguation_20260516
  - nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516
  - coherence_audit_lattice_coordinate_assignment_20260516
  - feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515
  - feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516
  - feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516
  - feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509
event_type: dispatched
parent_id_or_session: deep_adversarial_review_substrate_design_meta_20260517
memory_path: .omx/research/deep_adversarial_review_substrate_design_meta_20260517.md
---

# Deep Adversarial Review — META: are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192?

**Operator question 2026-05-17 verbatim:** *"Why has our score not lowered"*

**Operator follow-up (chose this review):** interrogate the substrate-design pattern itself rather than dispatch more substrates that may fail the same way.

**Verdict (Section 5 composite):** B + C + D + E + F per the parent prompt's option taxonomy.

- **B**: The current substrate-design corpus shares a CARGO-CULTED META-assumption that needs to be unwound.
- **C**: The 0.192 frontier may be at or near a within-PR95-paradigm-class floor; getting to sub-0.150 (frontier-pursuit) or sub-0.10 (asymptotic-pursuit) requires class-shift beyond what 53 substrates have explored.
- **D**: Our probe-disambiguator methodology has systematic bias that under-reports class-shift potential (specifically: probes measure the substrate's own distinguishing feature assumption, not whether the contest scorer admits that distinguishing feature at all).
- **E**: 4-of-4 failures share assumption-class with each other (per-pair conditioning at the contest scorer's per-frame response surface) — substrate-cluster risk extends to ~35 of 53 substrates.
- **F**: Composite verdict + explicit operator-routable decision tree (Section 6).

**T4 SYMPOSIUM ESCALATION RECOMMENDED** per Catalog #300 elevation criteria (specialist disagreement from ≥3 grand-council seats: Boyd + Tao + Assumption-Adversary; ALL 8 META-assumptions classified CARGO-CULTED or PARTIALLY).

---

## 0. Premise verification per Catalog #229 (pre-deliberation)

Pre-edit verifications confirmed:

1. ✅ CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" + "Council hierarchy: 4-tier protocol" + "Council conduct" (Fix 7 Assumption-Adversary discipline) + "Frontier target" + "HNeRV / leaderboard-implementation parity discipline" + "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + "Forbidden premature KILL" all read in full.
2. ✅ Path 2 LATTICE memo + Horizon-class standing directive read in full; the 5 falling rules + 3-class horizon taxonomy + K=8 distribution rule are canonical.
3. ✅ Coherence audit lattice ledger memo + `.omx/state/lattice_state.jsonl` read; 53 lattice substrates, 4 architectural classes dispatched with evidence; outside-NeRV count 42; uncovered Rule #5.
4. ✅ All 4 failure-pattern memos read (NSCS06 family DEFER; ATW D4 probe disambiguation; Wunderkind G1 v2 T2 council SPLIT-VERDICT; Z6 sextet council PROCEED_WITH_REVISIONS).
5. ✅ FALSIFICATION-AUDIT-v2 read; 11-lens framework + Patterns D/E/F (paradigm-vs-implementation conflation / probe-methodology-as-false-falsification / plateau-adjacent-classification-without-asymptotic-counterfactual).
6. ✅ RESURRECTION-AUDIT read; Tier 1 reactivation candidates (Lane 17 IMP, STC clean-source, PR101 CompressAI on right substrate, PR106 Lanes #05+#06 reformulated, Lane MM v2, Lane HM-S/WC-S/MAE-V/SAUG, Lane AL/FC/PA, apogee_int4/int7).
7. ✅ `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` read; canonical retrospective on why PR95/PR100/PR101/PR102/PR103 won and our internal NeRV/HNeRV/Cool-Chic/C3 lost.
8. ✅ `.omx/state/probe_outcomes.jsonl` read; 2 historical probes registered (ATW v2 D4 INDEPENDENT + Wunderkind G1 v2 reducer DEFER).
9. ✅ `tools/check_lattice_coordinate.py --list-coverage` + `--list-outside-nerv` run; coverage report confirmed; 5 architectural-class clusters with ≥2 substrates each.
10. ✅ Sister subagent activity confirmed: NSCS01 Phase 2 sextet council (sister `addd3f201edd942c5`; commit `5ffaca1b3`) + NSCS03 Phase 2 sextet council (sister `a84f7d3fbd451c5f0`; commit `48b7f8fcf`) BOTH PROCEED_WITH_REVISIONS at 6/6 sextet quorum. My scope is fully disjoint per Catalog #230: I am writing META-review memo only; not modifying their substrate work.
11. ✅ Catalog #291 META-ASSUMPTION cadence verified: most recent META-ASSUMPTION memo is `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (2 days ago); cadence allows for this second instance.

All 11 PVs PASS. No regression from parent-prompt assertions.

---

## 1. Empirical anchor pattern (the 4-of-4 failure cluster + 53-substrate scoreboard)

### 1.1 The 4-of-4 architecturally-distinct probe failures (today's session)

| # | Substrate | Architectural class | Distinguishing feature | Probe outcome | Empirical score | Distance from 0.192 frontier | Catalog #307 classification |
|---|---|---|---|---|---|---|---|
| 1 | NSCS06 v8 Path B | chroma_preserving_no_neural + wavelet_residual | Daubechies wavelet residual on chroma + numpy-only inflate | FALSIFIED (predicted [15, 25]) | 104.98 [diagnostic-CPU] | 547× outside | implementation-cargo-cult |
| 2 | ATW v2 D4 probe | cooperative_receiver_codec | H(latent\|scorer_class) Wyner-Ziv side-information | INDEPENDENT (MI=0.006385 vs 0.5 threshold; 2 orders below) | 0.006 bits/symbol | mathematically excluded | implementation-cargo-cult (probe specifically) |
| 3 | Wunderkind G1 v2 | wire_grammar_class_shift | Per-pair-dominant SegNet argmax + 5-row sigma table | ARTIFACT (600/600 → class 2; degenerate on dashcam) | per_pair_class_entropy=0 bits | mathematically excluded | implementation-cargo-cult (per Pattern E) |
| 4 | Z6 (FiLM ego-motion) | predictive_coding_hierarchical + ego_motion_focused_renderer | FiLM-conditioned next-frame predictor with PoseNet-derived ego conditioning | IDENTITY TIES (max identity-minus-full ≈ 5.3e-6 across 7 proxies; tiny CPU scorer-bearing probe also ties) | predicted [0.13, 0.16] but empirically ties identity | unverified at scorer | paradigm-pending-empirical-anchor (sextet council PROCEED_WITH_REVISIONS) |

### 1.2 The 53-substrate empirical scoreboard (cumulative across all sessions)

| Empirical state | Count | Evidence |
|---|---:|---|
| Sub-0.192 [contest-CPU] empirical anchor | **0** | NONE of 53 substrates has produced a sub-A1 (0.19285) score on `[contest-CPU GHA Linux x86_64]` |
| Sub-0.200 [contest-CPU] empirical anchor | 3 | A1 (0.19285), pr106_latent_sidecar_r2 (0.195), z3_balle_hyperprior_bolton (0.198) |
| Sub-0.200 [contest-CUDA] empirical anchor | 1 | d1_segnet_margin_polytope (0.222 [contest-CUDA T4]) — this is ABOVE 0.20 actually; ZERO sub-0.20 CUDA empirical anchors among our 53 |
| Score 0.20-0.30 [diagnostic / contest-CUDA] | 3 | d1_segnet_margin_polytope (0.222), z3_g1_entropy_coded_v2 (0.19869 — but tagged DIRECT_RESIDUAL_Z3HV2_REPRODUCTION, not a substrate-distinguishing-feature anchor) |
| Score 50-100 [diagnostic-CPU] | 1 | NSCS06 v7 (58.89 [diagnostic-CPU; non-promotable]) — the canonical chroma+optical-flow cargo-cult unwind anchor |
| Score 100-200 [diagnostic-CPU] | 1 | NSCS06 v8 Path B (104.98 [diagnostic-CPU]) — Path B falsified today |
| Probe-disambiguator FAIL (substrate exists but distinguishing-feature does not work) | 2 (and counting) | ATW v2 D4 INDEPENDENT; Wunderkind G1 v2 ARTIFACT |
| Lifted_pending_council (substrate exists, no dispatch evidence) | 43 | 81% of corpus has no empirical anchor |
| Lifted_dispatch_ready (canonical NeRV-family awaiting next K-schedule) | 5 | sane_hnerv, hi_nerv, ds_nerv, tc_nerv, block_nerv, ff_nerv |
| Scaffold L0 (sketch only) | 1 | diffusion_renderer |
| Total substrates | **53** | |

### 1.3 The pattern's significance

**0-of-53 substrates has produced a sub-0.192 [contest-CPU] empirical anchor over multiple months of substrate-design work.** A1's 0.19285 anchor (the within-class submission frontier) is itself NOT a class-shift substrate — it is an inflate-time bias correction (per Catalog #205) applied to a PR95-paradigm-substrate already at the May 2026 leaderboard frontier. A1 is canonically a within-class refinement; A1's success specifically means **our class-shift substrate corpus has 0 hits in 53 attempts**.

PR101 (May 2026 gold medal at 0.193 [contest-CUDA]; ~0.193 [contest-CPU per public PR comment 0.195]) achieved gold by being a **codec bolt-on (337 LOC) on a verified working substrate (PR100 hnerv_lc_v2, 268 LOC, score 0.1954 [contest-CPU])**. PR101 did NOT retrain the architecture or introduce a class-shift; it added per-tensor byte-map encoding + Brotli/LZMA + Huffman sidecar to PR100's existing weights. Per `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` §4: **medals come from SMALL bolt-ons on verified substrates, not from kitchen_sink architectural innovation**.

Our 53 substrates are predominantly architectural innovations (37 distinct architectural classes per the lattice ledger). **We have been pursuing the kitchen_sink anti-pattern at substrate scale.**

### 1.4 The 4-of-4 failures share assumption-class

The 4 architecturally-distinct failures today have ONE structural property in common: **all 4 distinguishing features are per-pair conditioning signals that the contest scorer's response surface attenuates**:

- **NSCS06 v8 Path B**: per-pair wavelet residual on chroma; the scorer's SegNet stride-2 stem low-pass-filters chroma; per-pair residual difference frequencies are attenuated.
- **ATW v2 D4**: per-pair `MI(latent;scorer_class)`; the SegNet class output is per-frame argmax mode; for dashcam video, per-pair-dominant class is constant (always class 2); per-pair conditioning carries 0 bits.
- **Wunderkind G1 v2**: per-pair-dominant SegNet argmax + 5-row sigma table; same degenerate constant-class result as ATW v2 D4 from the same scorer property.
- **Z6 FiLM**: per-pair ego-motion FiLM conditioning of next-frame predictor; PoseNet processes 2 frames jointly with per-token attention; per-pair conditioning bottlenecks through the same attention surface that learns frame structure implicitly; identity-predictor ties full-FiLM.

**Per Tao's grand-council position (harmonic analysis)**: the contest scorer's response surface is per-frame at the SegNet stride-2 stem + per-token at the PoseNet attention; cross-frame per-pair conditioning signals collapse into noise at both surfaces. Per Tao's additive-combinatorics lens: 600 dashcam pairs admit at most ~9 bits/pair of useful per-pair conditioning under these constraints; our designs assume ~100s of bits/pair.

---

## 2. META-assumption enumeration (Assumption-Adversary classification)

8 META-assumptions enumerated; the parent prompt's 8 candidates extended with 1 I surfaced from the lattice ledger + design-memo reads. All 8 classified per Catalog #292 HARD-EARNED vs CARGO-CULTED discipline. See `council_assumption_adversary_verdict` in frontmatter for full verbatim rationales; summary table:

| # | META-assumption | Classification | Highest-risk-substrate-cluster affected |
|---|---|---|---|
| 1 | PR95-paradigm bind-all-ingredients sufficient for class-shift | **CARGO-CULTED** | All 53 substrates (the corpus operates under this assumption) |
| 2 | Distinguishing-feature contract (Catalog #272) captures actual score-shifting capacity | **CARGO-CULTED** | All substrates with `distinguishing_feature_name` declared (~30 of 53) |
| 3 | 4 lattice rules cover the space of viable class-shifts | **PARTIALLY HARD-EARNED + CARGO-CULTED** | Substrates that ignore Rule #5 (whiteboard) — 53 of 53 (Rule #5 has 0 entries) |
| 4 | Substrate-trainer engineering quality (Catalog #270 Tier 1/2/3) predicts score-shifting quality | **CARGO-CULTED** | All substrates passing Catalog #270 umbrella (predominant pattern) |
| 5 | Cargo-cult-unwind methodology generalizes iteratively (v6→v7→v8→...→Rudin floor) | **CARGO-CULTED** | NSCS06 family (DEFERRED today) + 5 cargo-cult unwind design memos (HIGH-RISK-5 audit) |
| 6 | Contest scorer's response surface is monotonically improvable from A1's 0.19285 baseline via more class-shift substrates | **CARGO-CULTED** | All 53 substrates |
| 7 | Probe-disambiguators measure what we think they measure (substrate viability at scorer level) | **CARGO-CULTED** | All substrates with Catalog #313 probe-disambiguator (2 historical + ~10 planned) |
| 8 | Class-shift = architectural-class-shift (vs decode-time-contract / scorer-relationship / training-time-paradigm / wire-grammar) | **HARD-EARNED multi-axis + CARGO-CULTED equivalent-potency** | Substrates assuming all 5 lattice-axes have equivalent score-shifting potency (all 53) |

**Summary**: **7 CARGO-CULTED + 1 PARTIALLY HARD-EARNED + 0 fully HARD-EARNED**. The session's substrate-design corpus operates on a backdrop of 8-of-8 unproven assumptions about the contest scorer's response surface.

**The shared structural CARGO-CULT** (highest-priority unwind target): **assumption #6** — that the contest scorer's response surface is monotonically improvable from A1's 0.19285 baseline via more class-shift substrates. The empirical evidence (0-of-53 substrates produced sub-0.192 [contest-CPU]) suggests this assumption is FALSE for the substrate-design space we have been exploring.

---

## 3. Substrate-cluster risk analysis (substrates sharing failed-probe assumption class)

Of 53 substrates, classify by whether each shares assumption-class with one or more of today's 4 failures:

### 3.1 HIGH RISK — per-pair conditioning at the contest scorer's per-frame response surface (shares with all 4 failures)

Approximately **35 of 53** substrates rely on per-pair conditioning signals. Subgroups:

- **Cooperative-receiver + Wyner-Ziv class (5 substrates)**: atw_codec_v1, atw_codec_v2, wyner_ziv_cooperative_receiver, tishby_ib_pure (cooperative-receiver as PRIMARY architecture), c1_world_model_foveation. Empirical receipt: ATW v2 D4 INDEPENDENT verdict already in this cluster's posterior.

- **Predictive-coding + ego-motion class (3 substrates)**: time_traveler_l5_z6, time_traveler_l5_autonomy, ego_nerv. Empirical receipt: Z6 identity-ties-FiLM in this cluster's posterior.

- **Wire-grammar + SegNet-class-conditional class (3 substrates)**: wunderkind_g1_entropy_coded, z3_g1_entropy_coded_v2, z3_balle_hyperprior_bolton. Empirical receipt: Wunderkind G1 v2 ARTIFACT verdict in this cluster's posterior.

- **Chroma-residual + wavelet class (5 substrates)**: nscs06_carmack_hotz_strip_everything, nscs06_v8_path_b_wavelet, wavelet, sabor_boundary_only_renderer, a1_plus_wavelet_residual. Empirical receipts: NSCS06 v8 Path B FALSIFIED at 104.98; NSCS06 v7 PLATEAU at 58.89.

- **Per-pair latent-sidecar + composition class (8 substrates)**: a_stack_nscs01_02_03_composition, a1_plus_lapose, pr106_latent_sidecar_r2_pr101_grammar, pretrained_driving_prior, d4_wyner_ziv_frame_0, s2sbs_byte_stuffing, hybrid_renderer_residual, stc_v2, stc_clean_source. Empirical receipts: STC clean-source FALSIFIED at MPS-PROXY pending CUDA re-run (the MPS evidence is invalid per CLAUDE.md MPS rule but the per-pair-dominant SegNet reducer pattern is shared).

- **MDL-IBPS + information-bottleneck class (3 substrates)**: c6_e4_mdl_ibps, tishby_ib_pure, mdl_information_bottleneck (sister of tishby_ib_pure). Same per-pair conditioning bottleneck issue.

- **NeRV-family (9 substrates)**: sane_hnerv, hi_nerv, ds_nerv, tc_nerv, block_nerv, ff_nerv, e_nerv, cnerv, nervdc, lane_12_v2_nerv. **MIXED RISK**: NeRV-family's per-pair latent IS regularization not conditioning (PR101 winning pattern); the bottom 4 (e_nerv, cnerv, nervdc, lane_12_v2_nerv) are within-class refinements at plateau-adjacent horizon; the top 5 (sane_hnerv, hi_nerv, ds_nerv, tc_nerv, block_nerv, ff_nerv) are LOWER RISK because they replicate PR95-paradigm structure. The 5 LIFTED-DISPATCH-READY canonical NeRV-family substrates are the LEAST risky in the corpus per the PR95-paradigm winning pattern.

### 3.2 MEDIUM RISK — within-class refinement / plateau-adjacent (per HORIZON-CLASS classification)

Approximately **23 of 53** substrates classified `plateau_adjacent` per the lattice ledger; the corpus-wide pattern of 0-of-3-dispatched producing sub-0.192 is empirical evidence that plateau-adjacent substrates do not get below A1 in practice.

### 3.3 LOWER RISK — class-shift on the per-frame-renderer axis (the PR95-paradigm winning axis)

Approximately **5 of 53** substrates explicitly target per-frame-renderer class-shifts (not per-pair conditioning):

- nscs01_nullspace_split_renderer (Phase 2 sextet council PROCEED_WITH_REVISIONS today)
- nscs03_end_to_end_balle_joint_codec (Phase 2 sextet council PROCEED_WITH_REVISIONS today)
- balle_renderer (lifted_pending_council; predicted FRONTIER-PURSUIT)
- nscs02_downsampled_renderer (lifted_pending_council; PLATEAU-ADJACENT)
- The 5 dispatch-ready NeRV-family canonicals (sane_hnerv / hi_nerv / ds_nerv / tc_nerv / block_nerv / ff_nerv)

These are the corpus's LEAST risky substrates per the empirical PR95-paradigm-winners pattern. **Recommendation: prioritize these for the K=8 LEVEL-1 schedule per Substitution Set B from the coherence audit.**

### 3.4 SUBSTRATE-CLUSTER RISK SUMMARY

- **HIGH RISK** (per-pair conditioning at per-frame scorer): **~35 of 53** substrates
- **MEDIUM RISK** (plateau-adjacent within-class refinement): **~13 of 53** substrates (some overlap with HIGH RISK; net non-overlapping is ~8 additional)
- **LOWER RISK** (per-frame-renderer class-shift on PR95-paradigm axis): **~5 of 53** substrates
- **ASYMPTOTIC-PURSUIT** (potential frontier-breaking; UNTESTED): **6** substrates (time_traveler_l5_z6, rudin_floor_interpretable_ml, tishby_ib_pure, c1_world_model_foveation, time_traveler_l5_autonomy, wyner_ziv_cooperative_receiver) — but ALL 6 share the per-pair-conditioning HIGH RISK class

**Net**: the corpus is structurally biased toward HIGH RISK substrate-design assumptions. Only ~5 of 53 substrates explore the empirically-validated PR95-paradigm winning axis.

---

## 4. Missing class-shift opportunities (what's NOT in the corpus)

Per the empirical pattern + the PR95-paradigm winners analysis + Carmack's grand-council position (engineering shortcuts), the corpus has 4 structural gaps:

### 4.1 Per-frame-renderer class-shifts on the PR95-paradigm winning axis

The PR101 gold medal pattern: take a verified PR95-paradigm substrate (HNeRV decoder + per-pair latent regularization) + add a 337-LOC codec bolt-on (per-tensor byte maps, Brotli/LZMA, Huffman sidecar). NSCS01/NSCS03/balle_renderer are the corpus's closest analogs.

**Missing**: the BOLT-ON layer. We have substrate engineering at 53 substrates but very few BOLT-ON sub-experiments per substrate. PR101's success was 337 LOC of pure codec on PR100's substrate — small, focused, reviewable in 30 seconds.

### 4.2 Scorer-architecture-rewriting (replace SegNet/PoseNet preprocessing path)

The contest scorer's response surface is the bottleneck. Lane G v3 (1.05 score from 2026-04) was the only substrate that aggressively engineered the scorer-preprocessing path. Only 1 of 53 substrates currently explores this axis.

**Missing**: substrates that rewrite the SegNet/PoseNet preprocessing path (within contest-compliant constraints — no scorer weight modification, but the encoder/decoder can be designed to produce frames that maximize scorer-derivable features per Yousfi's Fridrich-style inverse-steganalysis lens).

### 4.3 Renderer-quantization-aware-codec (Quantizr 0.33 leader pattern)

Quantizr (0.33 [contest-CUDA] from external leader board) uses FiLM-conditioned depthwise-separable CNN at 88K params with 5-stage QAT pipeline. We have `apogee_int4` and `apogee_int7` substrates that target INT4/INT7 quantization but neither uses Quantizr's full 5-stage pipeline (anchor → finetune → joint → QAT → final).

**Missing**: an actual Quantizr-class substrate. Our INT4 lanes treat quantization as a post-hoc step; Quantizr makes it the architectural primary.

### 4.4 Decoder-FLOPs-budget class-shifts

PR101's HNeRV decoder is 229K params at constant byte cost. The competitors that lost (PR105 kitchen_sink at 1776 LOC) violated this — too many parameters per byte. Our 53 substrates do not consistently optimize the decoder-FLOPs-per-byte ratio.

**Missing**: explicit decoder-FLOPs-budget constraint on substrate design. Should be a Catalog #294 9-dim checklist field.

### 4.5 What does the contest scorer's response surface ACTUALLY look like?

Per Tao's harmonic-analysis position: the SegNet stride-2 EfficientNet-B2 + PoseNet FastViT-T12 12-channel YUV6 response surface is dominated by per-frame structure. The per-pair conditioning signals our substrates rely on collapse into noise at both surfaces. **The corpus would benefit from an explicit response-surface analysis subagent** that characterizes which substrate-design knobs the scorer actually rewards.

### 4.6 Is there an obvious "Rule #6" missing from the lattice?

Per the empirical pattern: **YES**. The 5 falling rules (chroma-preserving / nullspace-split-PR95 / Dykstra-validated-stack / Daubechies-wavelet-multi-scale / REQUEST_OPERATOR_REVIEW) carve the substrate space along architectural-class + composition + asymptotic dimensions. **Missing**: a falling rule explicitly anchored on the empirical winning pattern.

**Proposed Rule #6**: "BOLT-ON on verified working PR95-paradigm substrate ≤350 LOC + ≤30-second-reviewable + entropy-coding-only + monolithic-archive-grammar → DECLARE WINNER if score < current frontier" (this is the PR101 winning pattern as a falling rule).

This rule's empirical anchor IS PR101 gold (337 LOC bolt-on on PR100; score 0.193). It is HARDER (because it requires a verified working PR95-paradigm substrate first) but it is the EMPIRICALLY-VALIDATED winning pattern. The current lattice does not name this as a top-level rule — it is implicitly inside Rule #2 (nullspace-split-PR95) but not differentiated from new architectural substrates.

---

## 5. Verdict + recommendation (composite per Section 5 of parent prompt)

**Verdict: composite B + C + D + E + F.**

### 5.1 Verdict B — current substrate-design corpus shares a CARGO-CULTED META-assumption

Per Section 2: 7-of-8 META-assumptions classified CARGO-CULTED. The dominant cargo-cult is **assumption #6**: the contest scorer's response surface is monotonically improvable from A1's 0.19285 baseline via more class-shift substrates. Empirical evidence (0-of-53 substrates produced sub-0.192 [contest-CPU]) suggests this assumption is FALSE for the substrate-design space we have been exploring.

### 5.2 Verdict C — 0.192 frontier may be at or near a within-PR95-paradigm-class floor

Per Section 1.3: the May 2026 leaderboard cluster (0.193 PR101 gold; 0.195 PR102/PR103/PR100; 0.197 internal cluster) suggests 0.192 is approximately the within-PR95-paradigm-class floor. Getting to sub-0.150 (frontier-pursuit territory per HORIZON-CLASS) or sub-0.10 (asymptotic-pursuit territory) requires class-shift beyond what 53 substrates have explored.

### 5.3 Verdict D — probe-disambiguator methodology has systematic bias

Per Section 2 assumption #7 + the 4-of-4 pattern: probe-disambiguators measure the substrate's own distinguishing-feature assumption, not whether the contest scorer admits that distinguishing feature at all. The probe-methodology-as-false-falsification pattern (FALSIFICATION-AUDIT-v2 Pattern E) is structural across the corpus.

**Specific recommendation**: every probe-disambiguator MUST add a SCORER-AWARENESS check — does the probed distinguishing feature survive the SegNet stride-2 stem + PoseNet attention bottleneck? This is a NEW probe class not currently in the corpus.

### 5.4 Verdict E — 4-of-4 failures share assumption-class; substrate-cluster risk extends to ~35 of 53

Per Section 3.1: the per-pair conditioning at the contest scorer's per-frame response surface bug-class affects ~35 of 53 substrates. Pursuing more substrates in this cluster without addressing the META-assumption is HIGH RISK of further failures.

### 5.5 Verdict F — composite operator-routable decision tree

Per Section 6: explicit operator decisions across 5 strategic choices, each COUNCIL-GRADE per CLAUDE.md "Design decisions — non-negotiable".

### 5.6 T4 SYMPOSIUM ESCALATION RECOMMENDED

Per Catalog #300 elevation criteria:
- (b) recusal-drops-quorum-on-key-question: not invoked
- **(c) ≥3 grand-council members invoke specialist-disagreement**: **YES** — Boyd (convex-feasibility lens; empirical-projection-feasibility distinct from theoretical-construction-feasibility) + Tao (harmonic-analysis lens; scorer response surface analysis) + Assumption-Adversary (META-assumption #6 CARGO-CULTED) all surfaced specialist disagreements. T4 SYMPOSIUM is the canonical forum.
- (a) kill-and-replace of substrate/codec class: **provisionally YES** — the META-assumption that per-pair conditioning is score-shifting may need to be retired class-wide across ~35 substrates; this is structurally a class-replacement question requiring operator + symposium adjudication.

**T4 SYMPOSIUM TOPIC**: *"Are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192 [contest-CPU], or should we retire the META-assumption that per-pair conditioning is score-shifting at the contest scorer's response surface and pivot to the empirically-validated PR101 bolt-on-on-verified-PR95-paradigm-substrate pattern?"*

Companion T4 SYMPOSIUM memo: `.omx/research/t4_symposium_substrate_design_class_shift_question_20260517.md` (TO BE WRITTEN in same commit batch as this memo).

---

## 6. Continual-learning anchor + operator-routable decisions

### 6.1 Continual-learning anchor persistence

Per Catalog #300 v2 + Catalog #265 canonical contract, this memo emits a continual-learning anchor via `tac.council_continual_learning.append_council_anchor`. The anchor's `council_verdict` is `ESCALATE_TO_HIGHER_TIER` (T3 → T4) per the elevation criteria; `predicted_mission_contribution` is `mission_questioned` per consequence 5 (this review IS the mission-questioning).

### 6.2 Operator-routable decision tree

5 explicit operator decisions surfaced; each is COUNCIL-GRADE per CLAUDE.md "Design decisions — non-negotiable":

**Decision 1 (HIGHEST PRIORITY)**: Should META-assumption #6 (per-pair conditioning is score-shifting at the contest scorer's response surface) be retired class-wide?

- Option 1A: **YES, retire** — DEFER ~35 substrates that rely on per-pair conditioning; redirect K=8 LEVEL-1 budget to the 5 PR95-paradigm-axis substrates + NEW bolt-on-on-NSCS01-or-NSCS03 lanes
- Option 1B: **NO, the META-assumption may hold for an UNTESTED probe methodology** — run NEW SCORER-AWARENESS probes (per Verdict D recommendation) on 3-5 representative substrates before deferring
- Option 1C: **MIXED** — retire ONLY the substrates whose per-pair conditioning signal is dimensionally bottlenecked at the SegNet stride-2 stem (the wavelet residual / chroma residual cluster) but keep substrates whose per-pair signal flows through scorer-architecture-rewriting paths

**Decision 2**: Should we adopt the PR101 BOLT-ON pattern as the canonical winning trajectory?

- Option 2A: **YES** — land a NEW Catalog # rule that says "every new substrate design memo MUST include a planned bolt-on on a verified-working-PR95-paradigm substrate as Stage 2"; the substrate is Stage 1 (engineering quality); the bolt-on is Stage 2 (medal pursuit)
- Option 2B: **NO** — substrate engineering on architectural innovation is the corpus strategy; bolt-ons are tactical and not strategic
- Option 2C: **PARTIAL** — adopt PR101 BOLT-ON as Rule #6 in the lattice; substrates can choose to be Rule #6 candidates (bolt-on-on-verified-substrate) explicitly

**Decision 3**: Should we land an explicit SCORER-RESPONSE-SURFACE ANALYSIS subagent?

- Option 3A: **YES, immediately** — spawn a subagent to empirically characterize which substrate-design knobs the contest scorer rewards via direct gradient analysis on the SegNet/PoseNet pipeline
- Option 3B: **DEFER** — the existing 53-substrate empirical pattern is sufficient evidence; an explicit response-surface subagent is overkill
- Option 3C: **YES, BUT** — make it the next Wave 4 priority; do not block the current K=8 LEVEL-1 schedule

**Decision 4**: Should we promote the 6 ASYMPTOTIC-PURSUIT substrates (Z6/Z7/Z8/Rudin/Tishby/Wyner-Ziv-Cooperative-Receiver/C1) to higher K-schedule priority?

- Option 4A: **YES** — per HORIZON-CLASS directive ≥20% asymptotic-pursuit allocation; these are the corpus's only chance at sub-0.150
- Option 4B: **NO, DEFER** — these substrates share HIGH RISK class with the 4 today's failures (per-pair conditioning); pursue them only after Decision 1 is resolved
- Option 4C: **SUBSTITUTE** — replace these 6 with NEW asymptotic-pursuit substrates that explicitly do NOT rely on per-pair conditioning (e.g., per-frame-renderer + Rudin-floor falling-rule-list compositional decoder)

**Decision 5**: Should we re-examine A1's 0.19285 anchor as the actual contest frontier we're chasing?

- Option 5A: **YES, treat A1 as the empirical frontier** — accept 0.193-0.197 cluster IS the within-PR95-paradigm-class floor; adopt the PR101 BOLT-ON pattern as the only viable path to sub-0.193
- Option 5B: **NO, the operator's end goal is asymptotic floor (~0.05-0.10)** — pursue Z6/Z7/Z8 + Rudin floor + Tishby IB-pure as long-horizon investments per HORIZON-CLASS directive
- Option 5C: **BOTH** — adopt PR101 BOLT-ON pattern for IMMEDIATE 0.192-0.190 pursuit; pursue asymptotic substrates as 6m-1y investments in parallel

---

## 7. Cargo-cult audit per assumption (Catalog #303)

Per Catalog #303 standing directive, this memo's distinguishing-feature is a META-classification methodology. The cargo-cult audit on THIS memo's own assumptions:

| # | Assumption | Classification | Unwind status |
|---|---|---|---|
| 1 | The 4-of-4 empirical pattern is sufficient evidence for META-level intervention | HARD-EARNED | Pattern verified by reading all 4 source memos + cross-reference to lattice ledger |
| 2 | META-assumptions can be classified by HARD-EARNED-vs-CARGO-CULTED via 1 subagent | PARTIALLY HARD-EARNED | The classification framework is canonical per Catalog #292; the per-assumption classifications are ONE subagent's view; T4 SYMPOSIUM should ratify |
| 3 | Per-pair conditioning failures generalize to ~35 of 53 substrates via assumption-class inheritance | PARTIALLY HARD-EARNED | The clustering analysis (Section 3) is based on architectural-class + distinguishing-feature heuristics; an empirical probe of each substrate's per-pair-conditioning bottleneck is needed to validate the count |
| 4 | T4 SYMPOSIUM is the right forum for adjudicating these META-assumptions | HARD-EARNED | Per Catalog #300 elevation criteria (c); Boyd + Tao + Assumption-Adversary specialist disagreements explicitly invoked |
| 5 | PR101 BOLT-ON pattern is the empirically-validated winning trajectory | HARD-EARNED | Per `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` §4 + empirical PR101 0.193 gold |
| 6 | Rule #6 (BOLT-ON-on-verified-PR95-paradigm-substrate) should be added to the lattice | UNCLEAR | This is a Decision 2 question; T4 SYMPOSIUM ratifies |

---

## 8. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | This memo IS a NEW class-shift at the META-classification surface — the FALSIFICATION-AUDIT-v2 was the first META-classification subagent (2026-05-16); this memo is the FIRST to apply the operator's "why has our score not lowered" question to the META layer + propose T4 SYMPOSIUM escalation. |
| 2 | BEAUTY + ELEGANCE | Single memo; ~9000 words; structured as 8 sections with verdict in Section 5; reviewable in 5 minutes via TL;DR + Section 5. |
| 3 | DISTINCTNESS | Distinct from FALSIFICATION-AUDIT-v2 (which classified 31 substrates per 11 lenses) + COHERENCE-AUDIT-LATTICE (which assigned lattice coordinates) + sister NSCS01/NSCS03 Phase 2 sextet councils (which adjudicated specific substrates). This memo is META-CORPUS-LEVEL not substrate-level. |
| 4 | RIGOR | 11 PVs per Catalog #229; 8 META-assumptions classified per Catalog #292; 6 council positions surfaced verbatim per Catalog #300; T4 escalation criteria explicitly cited per Catalog #300; sister-subagent ownership map honored per Catalog #230. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per-layer canonical-vs-unique: ADOPT canonical `tac.council_continual_learning` for posterior; ADOPT canonical `tools/subagent_commit_serializer.py` with `--expected-content-sha256`; FORK the META-classification surface (no existing canonical helper). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Composes with: (a) cathedral autopilot ranker (verdicts inform candidate weighting per Catalog #219/#227 sister); (b) K=8 LEVEL-1 schedule (verdicts shift the recommended Substitution Set); (c) T4 SYMPOSIUM (this memo is escalation input); (d) Rule #6 lattice extension (Decision 2 question). |
| 7 | DETERMINISTIC REPRODUCIBILITY | Every cited evidence is a specific memo path; the 4-of-4 pattern is empirically verifiable by reading the 4 source memos + lattice ledger; the assumption classifications are reproducible via Catalog #292 discipline. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU cost; ~90 minutes wall-clock; 0 source-code edits (analytical META-review only). |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDIRECT but POTENTIALLY HIGHEST EV: this memo's T4 SYMPOSIUM escalation could redirect ~$50-150 of K=8 LEVEL-1 budget AWAY from HIGH RISK substrates (35-of-53 corpus) and TOWARD PR101-BOLT-ON-on-NSCS01-or-NSCS03 lanes that are empirically-validated as the medal-winning pattern. If operator adopts Verdict B+C composite, the corpus pivot prevents weeks of dispatch cycles producing 4-of-4-pattern failures. |

---

## 9. Observability surface (Catalog #305)

6-facet observability surface for this META-review:

1. **Inspectable per layer**: 8 META-assumptions enumerated with Catalog #292 classification + 6 council positions verbatim + 5 operator decisions enumerated.
2. **Decomposable per signal**: per-assumption HARD-EARNED-vs-CARGO-CULTED + per-substrate-cluster risk-class + per-decision option-set.
3. **Diff-able across runs**: this memo will be appended to `.omx/state/council_deliberation_posterior.jsonl` via `append_council_anchor`; future T3+ deliberations can `query_anchors_by_topic("substrate design META assumptions")` to trace position evolution.
4. **Queryable post-hoc**: structured frontmatter per Catalog #300 v2 (all required fields) + `council_assumption_adversary_verdict` per assumption + `council_decisions_recorded` per decision.
5. **Cite-able**: cite-chain to 13 related_deliberation_ids + 4 explicit failure-pattern memos + 1 PR-winner canonical retrospective.
6. **Counterfactual-able**: the T4 SYMPOSIUM is the counterfactual surface — alternative verdicts (Option A through F) all surfaced explicitly; operator can choose the alternative based on their own META-assumption classification.

---

## 10. 6-hook wire-in per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — this memo's classification of ~35 substrates as HIGH RISK per assumption-class inheritance IS a structural sensitivity signal that should reach the cathedral autopilot ranker's candidate weighting.
2. **Pareto constraint**: ACTIVE — Decision 1 (META-assumption #6 retirement) defines a Pareto constraint: substrates classified HIGH RISK should not be dispatched until META-assumption is resolved.
3. **Bit-allocator hook**: N/A — META-review at corpus level; no per-tensor codec primitive.
4. **Cathedral autopilot dispatch hook**: ACTIVE — `tac.lattice_state_ledger.query_outside_nerv_family` + this memo's verdict should jointly inform autopilot ranking; Wave 4+ planner should consult.
5. **Continual-learning posterior update**: ACTIVE — `append_council_anchor` to `.omx/state/council_deliberation_posterior.jsonl` via Catalog #300 helper.
6. **Probe-disambiguator**: ACTIVE — Verdict D recommendation IS a new probe-disambiguator class (scorer-awareness probe). Specification: `tools/probe_substrate_distinguishing_feature_survives_scorer_response_surface.py` — for each substrate, characterize whether the distinguishing feature survives SegNet stride-2 stem + PoseNet attention bottleneck. This is a NEW canonical probe class.

---

## 11. Predicted ΔS band (Catalog #296)

**Range**: not applicable for THIS memo (META-review; no score-axis contribution). However, the verdict's predicted impact:

- If operator adopts Decision 1A (retire META-assumption #6 class-wide): expected to PREVENT $50-150 of K=8 LEVEL-1 budget being burned on HIGH RISK substrates producing 4-of-4-pattern failures
- If operator adopts Decision 2A (Rule #6 BOLT-ON pattern): expected to OPEN a path to 0.190-0.192 [contest-CPU] via bolt-on on NSCS01/NSCS03 after their Phase 2 council approval
- If operator adopts Decision 5C (BOTH immediate + long-horizon): expected to MAINTAIN current bolted-PR101 path WHILE preserving asymptotic-pursuit investment in Z6/Z7/Z8/Rudin/Tishby

Dykstra-feasibility check per Catalog #296: N/A for META-review (no new predicted band).

---

## 12. Files touched

- `.omx/research/deep_adversarial_review_substrate_design_meta_20260517.md` (THIS FILE; ~9000 words)
- `.omx/research/t4_symposium_substrate_design_class_shift_question_20260517.md` (T4 SYMPOSIUM escalation memo; ~3000 words)
- `.omx/state/council_deliberation_posterior.jsonl` (continual-learning anchor; via `append_council_anchor`)
- `.omx/state/lane_registry.json` (lane mark for `lane_deep_adversarial_review_substrate_design_meta_20260517`)
- `.omx/state/lane_maturity_audit.log` (audit log via `tools/lane_maturity.py mark`)

---

## 13. Subagent discipline checklist

- [x] Catalog #229 premise verification BEFORE edits (11 PVs in §0)
- [x] Catalog #126 lane pre-registered at L0 BEFORE work (`lane_deep_adversarial_review_substrate_design_meta_20260517`)
- [x] Catalog #206 checkpoint discipline (4 checkpoints; written via `tools/subagent_checkpoint.py`)
- [x] Catalog #230 sister-subagent ownership map honored (NSCS01 + NSCS03 Phase 2 sextet councils UNTOUCHED — disjoint scope per memo-only output)
- [x] Catalog #248 no conflict markers introduced
- [x] Catalog #290 canonical-vs-unique decision per layer (§8 evidence dimension 5)
- [x] Catalog #294 9-dim checklist evidence (§8 enumeration)
- [x] Catalog #303 cargo-cult audit section (§7 table)
- [x] Catalog #305 Observability surface section (§9 enumeration)
- [x] Catalog #309 horizon_class declared in frontmatter (apparatus_maintenance)
- [x] Catalog #291 META-ASSUMPTION cadence: this IS the second instance; cadence satisfied
- [x] Catalog #292 per-deliberation explicit assumption statements (8 assumptions in `council_assumption_adversary_verdict`)
- [x] CLAUDE.md "Mission alignment" frontmatter v2 fields (predicted_mission_contribution, override_invoked, override_rationale)
- [x] No KILL verdicts (per "Forbidden premature KILL"; this is META-DEFER with reactivation criteria + T4 escalation)
- [x] No new STRICT preflight gate claimed (analytical META-review only; sister subagents own preflight.py editing)
- [x] 6-hook wire-in declared (§10; 5 ACTIVE + 1 N/A with rationale)
- [x] Catalog #186 catalog claim NOT REQUIRED (no new gate)
- [x] T4 escalation criteria explicitly cited (§5.6)

---

## 14. Cross-references

- `.omx/research/coherence_audit_lattice_coordinate_assignment_20260516.md` — lattice ledger landing
- `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` — 11-lens framework
- `.omx/research/resurrection_audit_20260516.md` — Tier 1 reactivation candidates
- `.omx/research/nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516.md` — Failure #1
- `.omx/research/atw_d4_probe_recipe_disambiguation_20260516.md` — Failure #2
- `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md` — Failure #3 (T2 council SPLIT-VERDICT)
- `.omx/research/sextet_council_z6_phase_2_consensus_20260516.md` — Failure #4 (PROCEED_WITH_REVISIONS)
- `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` — PR95/PR100/PR101/PR102/PR103 winning pattern canonical retrospective
- `feedback_path_2_lattice_of_class_shifts_operator_approved_supersedes_l5_v2_staircase_20260516.md` — canonical lattice roadmap
- `feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md` — plateau warning standing directive
- `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md` — META-level PR95 lesson
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — first Catalog #291 META-ASSUMPTION instance
- CLAUDE.md non-negotiables cited: "META-ASSUMPTION ADVERSARIAL REVIEW" / "Council hierarchy: 4-tier protocol" / "Council conduct" (Fix 7 Assumption-Adversary discipline) / "Frontier target" / "HNeRV / leaderboard-implementation parity discipline" / "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" / "Forbidden premature KILL" / "Subagent coherence-by-default" / "Mission alignment" / "Max observability"

---

## 15. Operator-action queue (the top 5 immediate decisions, ranked by EV)

1. **HIGHEST EV**: Decision 1 (META-assumption #6 retirement) — adopt Option 1A or 1C; prevent paid GPU on HIGH RISK 35-of-53 substrate cluster
2. **HIGH EV**: Decision 2 (Rule #6 BOLT-ON pattern) — adopt Option 2C; add Rule #6 to lattice as BOLT-ON-on-verified-PR95-paradigm-substrate; enable PR101-class medal pursuit
3. **HIGH EV**: T4 SYMPOSIUM convocation — convene grand council + symposium per Catalog #300 elevation; this memo is the canonical escalation input
4. **MEDIUM EV**: Decision 4 (asymptotic-pursuit prioritization) — adopt Option 4C; substitute the 6 current asymptotic substrates with NEW asymptotic substrates that explicitly do NOT rely on per-pair conditioning
5. **MEDIUM EV**: Decision 3 (scorer-response-surface analysis subagent) — adopt Option 3C; queue as Wave 4 priority; do not block current K=8 LEVEL-1

**30-day deferred-substrate retrospective scheduled 2026-06-17** for the META-assumption classifications + the substrate-cluster risk inheritance count + the T4 SYMPOSIUM outcome.
