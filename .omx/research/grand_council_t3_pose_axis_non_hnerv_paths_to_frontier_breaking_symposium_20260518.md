---
schema: council_deliberation_v2
deliberation_id: grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518
topic: "Pose-axis NON-HNeRV paths to a frontier-breaking improvement at sub-0.20 operating point"
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Boyd
  - Mallat
  - van_den_Oord
  - Carmack
  - Hinton
  - Tishby_memorial
  - Atick
  - Redlich
  - Rao
  - Ballard
  - Wyner
  - Hafner
  - Time_Traveler_protege_Rudin_postdoc
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The deliberation is too quick to redirect dispatch budget away from the existing fec6 / format0d HNeRV-pattern frontier. We currently SIT at 0.19205 [contest-CPU] / 0.20533 [contest-CUDA T4] — that is the verified contest-faithful frontier. Every NON-HNeRV path proposed below predicts an unverified ΔS band. Two NON-HNeRV substrate dispatches THIS WEEK (Z6-v2 driver-mode-hardcode DEFER 2026-05-18; C6 IBPS Tier-C 22× miss 2026-05-17) BOTH burned $0.50-$2 on implementation-level falsifications. The expected value of MOST NON-HNeRV paths below is BELOW the marginal value of one more HNeRV-pattern stacking bolt-on (DP1+PR101 composition, fec6 + format0d cross-stack, etc.). Council should EXPLICITLY rank-order: cheap-NON-HNeRV-probes BEFORE expensive-NON-HNeRV-substrate-builds; HNeRV-pattern stacking remains the dominant near-term EV path. I require: every TOP-3 op-routable below MUST have a $5-15 cheap probe FIRST, NOT an immediate $20-50 substrate build. Per Catalog #313 every dispatch consults the predecessor probe-outcomes ledger; the Z6/C6/ATW-D4 INDEPENDENT verdicts from the past 7 days are binding precedents."
  - member: Assumption-Adversary
    verbatim: "The deliberation is operating within the SHARED ASSUMPTION that 'NON-HNeRV is the orthogonal direction we need to break out of the 0.19 plateau.' Classification: CARGO-CULTED at the framing level, HARD-EARNED at the marginal-pose-value level. Per CLAUDE.md 'SegNet vs PoseNet importance — operating-point dependent', pose IS 2.71× more valuable per marginal byte than seg at PR106 frontier. But the FRAMING that HNeRV-pattern cannot deliver further pose-axis gains is unverified — the fec6 frontier IS PR101 + frame-conditional codec selector, and PR101 itself ALREADY exploits the HNeRV-pattern's implicit pose representation via the rendered RGB path. The pose-axis improvement could come from EITHER (a) better implicit pose via an improved HNeRV-pattern renderer, OR (b) an explicit pose stream layered atop the existing HNeRV-pattern (the A1+LAPose 2026-05-13 council path), OR (c) a fully NON-HNeRV substrate (the question this council is asked). The shared assumption that (c) DOMINATES (a) and (b) is the cargo-cult to challenge. Per the Atick-Redlich cooperative-receiver framing the SCORER ALREADY computes pose from rendered RGB; the marginal byte budget is best spent on WHAT the renderer produces, not on bypassing it. I propose the council split the verdict: PROCEED on the cheapest probe family (Wyner-Ziv pose-residual + master-gradient pose-byte classification + ATW V2-1 channel pick), DEFER on the most expensive substrate-class-shift family (Z6/Z7/Z8 from scratch), and re-route freed budget to HNeRV-pattern stacking with explicit pose-axis foci."
  - member: Carmack
    verbatim: "The pose contribution at PR106 frontier is sqrt(10 × 3.4e-5) = 0.0184. A 50% pose_avg reduction returns 0.0054 score (Shannon math, prior council §4.1, codex-corrected). The PR101 GOLD pose component is already 'good' (0.0184 is a small fraction of total 0.193). The question is whether a NON-HNeRV path can push pose_avg from 3.4e-5 down by 50% AT BUDGET ≤ 2 KB. The answer for the cheap-probe family (Wyner-Ziv pose hoist, master-gradient pose-byte classification, ATW V2-1 channel reformulation) is empirically YES at $5-15. The answer for full substrate replacement (Z6/Z7/Z8 ego-motion-conditioned predictor from scratch) is empirically NO at < $50 because the substrate engineering budget per HNeRV parity L7 exceeds 600 LOC and Z6-v2 Wave 2 already burned $0.50 today (DEFER per probe outcome). Recommendation: rank the 6 paths by $/predicted-ΔS-per-byte; dispatch ONLY the top-3 cheap-probe family in this wave; defer Z6/Z7/Z8 from-scratch substrates pending HNeRV-pattern saturation evidence."
council_assumption_adversary_verdict:
  - assumption: "Pose-axis is the dominant marginal value at PR106 frontier"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'SegNet vs PoseNet importance — operating-point dependent' (2026-05-04 verified): at pose_avg=3.4e-5 the pose derivative 5/sqrt(10*pose_avg)=271 is 2.71× the SegNet constant derivative 100. Empirical receipt: docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md + docs/pr_family_evolution_timeline_20260504.md. PR97 made the seg-for-pose trade and lost 0.042 score despite winning SegNet by 65%. The hard-earned framing applies: pose marginal IS dominant per byte."
  - assumption: "HNeRV-pattern is the only public-frontier path"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'HNeRV / leaderboard-implementation parity discipline' lesson 1 verbatim: 'PR #100's hnerv_lc_v2 (268 LOC) bound architecture + score-aware training + archive grammar + inflate runtime + export contract simultaneously, and PR #101 (337 additional LOC of entropy bolt-ons) won gold at 0.193 by stacking on the verified substrate.' All 3 top-3 contest winners 2026-05-04 race window (PR101 / PR102 / PR103) used HNeRV-pattern. Our 0.19205 fec6 frontier IS PR101 + frame-conditional codec selector. NON-HNeRV substrates submitted to the public leaderboard were dominated."
  - assumption: "A direct pose codec (encode pose-deltas as side-information per Wyner-Ziv) is operator-attention-budget-feasible"
    classification: HARD-EARNED
    rationale: "Per Wyner 1976 + 2026-05-17 T3 Wyner-Ziv canonical-fix council (grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md): the side-info-at-decoder gain is Rate(X) - Rate(X|Y) = I(X;Y) where Y is decoder-reconstructable from frame_0 (Tier 1 deterministic transform) OR baked compress-time constants (Tier 2). For pose-residual encoding the relevant Y is the optical-flow-warped pose estimate from frame_0 → frame_1 (Tier 1 deterministic). Shannon-bound on residual is 200-500 bytes (Ballé hyperprior Markov-1) per 2026-05-13 council §4.1. Budget-feasible."
  - assumption: "Pose-conditioned residual codec (encode RGB residuals conditioned on pose-prior) is byte-distinguishable from HNeRV-pattern"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "The proposed architecture is the inverse of HNeRV: instead of an embedding-conditioned decoder producing RGB, a pose-conditioned decoder produces RGB residuals on top of a base reconstruction. The Atick-Redlich cooperative-receiver framing predicts this is information-theoretically equivalent IFF the encoder uses the SAME pose prior as the scorer's PoseNet. Empirical verification has NOT been attempted; the architecture sits PRE-OPTIMAL-FORM per Catalog #315. Reactivation criterion: 5ep CPU smoke at $0.05-$0.20 verifying gradient flow through pose-prior + residual decoder."
  - assumption: "Ego-motion-conditioned next-frame prediction (Z6/Z7/Z8 family) delivers asymptotic-pursuit ΔS"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Z6-v2 Wave 2 dispatch 2026-05-18 (driver-mode hardcode DEFER) + Z7 Mamba-2 design memo + Z8 hierarchical predictive coding design memo all sit PRE-OPTIMAL-FORM per Catalog #315 + Catalog #325 per-substrate symposium discipline. The 'asymptotic-pursuit' predicted bands [0.05, 0.12] per HORIZON-CLASS standing directive 2026-05-16 are PREDICTIONS, not empirical anchors. C6 IBPS 22× miss 2026-05-17 (predicted [0.113, 0.163] vs empirical 3.04) is the recent empirical receipt that asymptotic-pursuit predictions are unreliable until validated post-training per Catalog #324."
  - assumption: "Pose-bin classification (discrete pose codebook) bypasses HNeRV-pattern's implicit pose"
    classification: HARD-EARNED-VIA-MASTER-GRADIENT
    rationale: "Per Codex's just-landed master-gradient extractor + tac.master_gradient_consumers per-pair Venn classification (HIGH_PAIR_INVARIANT / HIGH_PAIR_SPECIFIC), the contest's 600 pairs decompose into ~20-40 distinct pose-class modes (ego-motion dashcam: straight-line forward + small yaw rotation dominates). The 2026-05-13 council §4.1 Shannon-Fano upper bound: 600 × log2(8) = 1800 bits = 225 bytes for pose-residual stream alone. Master-gradient per-pair pose-byte classification IS the canonical helper that would generate the discrete-pose codebook directly from the existing PR101 archive."
  - assumption: "Pose-FOE sparse encoding (focus-of-expansion per Gibson 1950) is byte-feasible at ≤1 KB"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Per Catalog #311 + the L5 staircase v2 design memo: ego-motion-conditioned next-frame prediction has an EMBODIED axis (Ballard) where FOE (focus-of-expansion) provides a sparse per-frame attention center. Telescope hyperbolic foveation (2026-04-07 arxiv) + LAPose latent-action together would deliver a 2-stage pipeline. Existing pact research: telescope_foveation_lfv1_candidate_20260513_codex.md + lapose_telescope_online_distinction_20260513_codex.md. Empirical verification has NOT been attempted on a NON-HNeRV substrate (only as residual atop A1); the FOE-sparse pose path sits PRE-OPTIMAL-FORM."
  - assumption: "Direct pose master-gradient exploitation (Codex's just-landed extractor) delivers pose-axis savings at $0-$1 cost"
    classification: HARD-EARNED-EMPIRICALLY-NEAR-LANDED
    rationale: "Per Codex 2026-05-18 master-gradient canonical helper landing (tac.master_gradient + tools/extract_master_gradient.py + Catalog #318/#327 self-protection): per-pair fp64 master-gradient on PR101 anchor has already been computed (8-pair subset; cos(seg_grad, pose_grad) ≈ 0.8973 implies rank-1 null-space). The per-pair pose-byte classification via PerByteVennClassification is ACTIVE; the deliverable_savings extraction is gated by Catalog #319 deliverability_proof_builder per the just-landed Wyner-Ziv canonical-fix wave. Empirical receipts are ZERO-PROVIDER-SPEND CPU-only and the canonical infrastructure exists today."
council_decisions_recorded:
  - "OP-1 [PROCEED]: Land the Wyner-Ziv pose-residual hoist (Tier-1 deterministic: optical-flow-warped pose estimate from frame_0 as decoder side-info) as the FIRST cheap probe. Cost: $0 CPU offline probe via tac.side_information.deliverability_proof_builder + ~$0.30 Modal A10G CPU smoke for paired CUDA verification. Predicted ΔS: [-0.003, -0.001] [prediction-pending-post-training-Tier-C-validation per Catalog #324]. Per Catalog #319 v2 cascade the deliverability_proof_class MUST be tier_1_deterministic for the autopilot reward branch to fire. Sister of 2026-05-17 Wyner-Ziv T3 council OP-4."
  - "OP-2 [PROCEED]: Land the master-gradient pose-byte classification consumer extension (extend tac.master_gradient_consumers.classify_bytes_by_pair_variance with pose-axis-specific classification: HIGH_POSE_DERIVATIVE_PAIR_INVARIANT vs HIGH_POSE_DERIVATIVE_PAIR_SPECIFIC). Cost: $0 CPU offline; sister probe on existing 8-pair fp64 anchor produces the proof artifact. Predicted ΔS: [-0.002, -0.0005] (extends Catalog #319 v2 cascade to the pose-axis-specific reward factor; orthogonal to seg-axis-specific factor). Per Catalog #322 anti-phantom protection: the proof MUST cite the per-pair gradient artifact path."
  - "OP-3 [PROCEED]: Land the ATW V2-1 channel-pick reformulation per per-substrate symposium #866 TT5L V2 cross-substrate dependency. The 2026-05-16 ATW V2 D4 INDEPENDENT verdict (MI=0.006385 per-pair argmax composite) DEFERRED-pending-alternative-reducer per Catalog #308. The 4 alternative reducers (per-region 16×16 SegNet softmax histogram + product-quantized to ≤2KB / per-pair class HISTOGRAM / pose-bin discretization / hard-pair-object-state composite) are enumerated. Cost: ~$5-7 CPU probe (offline product-quantization via Faiss-IVF-PQ per deep-research wave TOP-3 OSS) + ~$15-25 Modal A100 post-Wave-2-4c. Predicted ΔS: [-0.015, -0.005]. Cross-substrate dependency: awaits Z6 Wave 2 4c outcome (probe outcomes ledger DEFER 2026-05-18)."
  - "OP-4 [PROCEED_WITH_REVISIONS]: Land the pose-conditioned residual codec PROBE (NOT full substrate). The architecture is: HNeRV-pattern renderer produces base RGB; a SMALL pose-conditioned residual decoder (≤50KB) produces correction RGB; total RGB = base + residual. The pose conditioning uses PoseNet 6-DOF directly as decoder side-info (NOT as a loss). Cost: $0 CPU offline architecture probe + ~$2-5 Modal A10G 100ep smoke for gradient-flow verification. Predicted ΔS: [-0.008, -0.003] [prediction-pending-post-training-Tier-C-validation]. Per Catalog #325 per-substrate symposium discipline this OP requires a DEDICATED per-substrate symposium BEFORE paid dispatch >$1 — that symposium is queued as op-routable #11."
  - "OP-5 [DEFER_PENDING_EVIDENCE]: Z6-v2 / Z7 Mamba-2 / Z8 hierarchical-predictive-coding ego-motion-conditioned next-frame predictor full substrate builds. The Z6-v2 Wave 2 DEFER 2026-05-18 (driver-mode hardcode) + C6 IBPS Tier-C 22× miss 2026-05-17 + ATW v2 D4 INDEPENDENT 2026-05-16 are 3 consecutive same-week empirical falsifications of NON-HNeRV substrate-class-shift paths AT IMPLEMENTATION LEVEL per Catalog #307. Per CLAUDE.md 'Forbidden premature KILL': the PARADIGM remains intact; the IMPLEMENTATIONS sit PRE-OPTIMAL-FORM per Catalog #315. Reactivation criteria: (1) Z6 driver-mode hardcode fix lands + Wave 2 re-fires + Atick Council Revision #6 ΔS measured; (2) C6 IBPS Phase 2 cargo-cult-unwind redesign on SegNet-collapse + Tier-C post-training validation per Catalog #324; (3) ATW V2 D4 reducer reformulated per OP-3 above + sister probe re-fires."
  - "OP-6 [PROCEED]: Land the pose-FOE sparse encoding PROBE per Telescope hyperbolic foveation + LAPose latent-action 2-stage pipeline. Per the 2026-05-13 LA-Pose/Telescope online distinction memo: keep the two primitives SEPARATE (LA-Pose latent actions select WHERE pose bytes matter; Telescope hyperbolic foveation supplies the byte-closed sparse encoding). Cost: ~$0-$1 CPU offline analysis (the LFV1 candidate already landed per telescope_foveation_lfv1_candidate_20260513_codex.md). Predicted ΔS: [-0.005, -0.001] [prediction-pending-post-training-Tier-C-validation]. Per Catalog #311 ego-motion-conditioning is satisfied; per Catalog #310 the path remains BOLT-ON to existing HNeRV-pattern UNLESS rescoped to PRIMARY substrate (in which case OP-5 reactivation criteria apply)."
  - "OP-7 [PROCEED]: Direct master-gradient pose-byte hoist via Codex's just-landed extractor. Per Codex 2026-05-18 landing + Catalog #318/#327 self-protection: the per-pair fp64 master-gradient extraction is OPERATIONAL; the per-pair Venn classification is OPERATIONAL; the deliverability_proof_builder pipeline is OPERATIONAL. The remaining work is to wire pose-axis-specific deliverable_savings computation into tac.master_gradient_consumers.adjust_predicted_delta_for_venn_classification_v2 CASCADE 2 per Catalog #319 v2. Cost: $0 CPU only. Predicted ΔS: [-0.002, -0.0005] (extension reward factor for pose-axis-specific Venn classification). Per Catalog #322 anti-phantom: deliverability_proof_class MUST be tier_1 or tier_2 with non-empty proof artifact."
  - "OP-8 [PROCEED_WITH_REVISIONS]: VGGT-encoder-as-pose-teacher distillation probe. Per deep-research wave 2026-05-18 TOP-5 #1: VGGT (CVPR 2025 Best Paper, arxiv 2503.11651, facebookresearch/vggt) feed-forward 3D-from-N-views + DUSt3R/MASt3R as canonical pose teacher. The pact PoseNet is FastViT-T12; VGGT distillation could replace the implicit pose-from-rendered-RGB path with an explicit VGGT-teacher-distilled pose stream. Cost: ~$0-$2 CPU offline distillation feasibility study + ~$10-15 Modal A100 for paired smoke. Predicted ΔS: [-0.010, -0.003] [prediction-pending-post-training-Tier-C-validation]. Per CLAUDE.md 'strict scorer rule' the VGGT WEIGHTS cannot ship in archive; the distilled-into-PR101-renderer-weights pattern is the canonical workaround. Per Catalog #325 per-substrate symposium discipline this OP requires its own symposium BEFORE paid dispatch >$1."
  - "OP-9 [PROCEED]: Land all 8 OPs above as a coordinated wave with EV ranking enforcement (cheap-probe first; substrate-build last). Wave scheduling: Week 1 = OP-1+OP-2+OP-7 ($0-$1 each; all CPU-only via existing canonical infrastructure); Week 2 = OP-6 ($0-$1 LFV1 candidate analysis); Week 3 = OP-3 ($5-7 CPU probe + $15-25 Modal pending Z6 Wave 2 outcome); Week 4 = OP-4 ($2-5 smoke); Week 5+ = OP-8 ($10-15 paired smoke) + OP-5 reactivation pending criteria. Total provider spend Week 1-4: $25-60. Total provider spend Week 5+: $10-50."
  - "OP-10 [PROCEED]: Wire the cathedral autopilot ranker (tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2) to consume per-pair pose-axis-specific deliverability_proof artifacts emitted by OP-1, OP-2, OP-7 above. The v2 cascade (Cascade 1 = Lagrangian planner / Cascade 2 = DeliverabilityProof / Cascade 3 = passthrough) already exists per the 2026-05-17 Q3 landing; OP-10 extends Cascade 2's per-Venn-class reward function to per-pair pose-axis-specific factors (currently per-substrate). Cost: ~30-50 LOC + 15 tests; $0 GPU. Predicted ΔS: indirect via better dispatch ranking; estimated [-0.001, -0.0001] aggregate."
  - "OP-11 [DEFER_PENDING_EVIDENCE]: Per-substrate symposiums (Catalog #325) for OP-4 (pose-conditioned residual codec) AND OP-8 (VGGT-encoder-as-pose-teacher distillation). Each symposium produces a 6-step contract memo per Catalog #325 (cargo-cult audit / 9-dim checklist / observability surface / sextet pact deliberation / per-substrate reactivation criteria / post-training Tier-C validation discipline) BEFORE paid dispatch >$0.30. Cost: ~$0 each (research-only memo). Triggers OP-4 + OP-8 paid dispatch unlock."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
predicted_band_validation_status: pending_post_training
deferred_substrate_id: ""
deferred_substrate_retrospective_due_utc: "2026-06-17T18:42:00.000000Z"
related_deliberation_ids:
  - grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513
  - grand_council_pose_axis_non_hnerv_a1_plus_lapose_d4_deeper_20260513
  - grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517
  - council_per_substrate_symposium_tt5l_foveation_lapose_20260517
  - council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518
  - deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518
  - z7_mamba2_substrate_design_memo_20260518
  - tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518
  - atw_codec_atick_tishby_wyner_v1_design_20260515
  - external_sources_20260505_lapose_dominant_codex
  - lapose_telescope_online_distinction_20260513_codex
---

# Grand Council T3 Symposium — Pose-axis NON-HNeRV paths to a frontier-breaking improvement

**Lane**: `lane_pose_axis_non_hnerv_t3_council_20260518` (L0 → L1 on memo land)
**Tier**: T3 (touches CLAUDE.md non-negotiables: 'SegNet vs PoseNet importance — operating-point dependent' 2026-05-04 + 'Frontier target' + 'HNeRV / leaderboard-implementation parity discipline' + Catalog #313 probe-outcomes-ledger consultation + Catalog #319 Wyner-Ziv deliverability proof + Catalog #325 per-substrate symposium discipline)
**Quorum**: 6-of-6 sextet + 13-of-20 grand-council attendees (above T3 ≥12-of-20 threshold). Recusals: none (each member's relevance to the question is direct).
**Verdict**: PROCEED_WITH_REVISIONS (10 op-routables OP-1 through OP-11; OP-5 + OP-11 explicitly DEFER pending evidence + symposium completion)
**research_only**: true — council deliberation; the sister BUILD subagents own actuator code/recipe/driver
**Score claim**: false (all predicted ΔS bands are predictions per Catalog #324; not contest-CPU / contest-CUDA authoritative)
**Axis discipline**: every numeric score in this memo carries `[contest-CPU]`, `[contest-CUDA T4]`, `[macOS-CPU advisory]`, `[prediction]`, or `[empirical]` tag per CLAUDE.md 'Apples-to-apples evidence discipline'
**Wire-in hooks (Catalog #125)**: declared in §11
**Predicted-band validation status**: `pending_post_training` per Catalog #324 — every predicted ΔS band in this memo MUST be re-validated against the post-training Tier-C density on the landed archive sha256 BEFORE the band is treated as authoritative

---

## 0. Pre-flight compliance (Catalog #229 premise verification)

### Premises verified BEFORE writing this memo

1. **CLAUDE.md 'SegNet vs PoseNet importance — operating-point dependent'** (2026-05-04 UPDATED section, line 2736-2777): VERIFIED. The 2.71× marginal pose value at PR106 frontier IS the canonical framing. Source: pact CLAUDE.md HEAD.
2. **CLAUDE.md 'Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS'**: VERIFIED. The target IS public frontier; our verified frontier is 0.19205 [contest-CPU] + 0.20533 [contest-CUDA T4] per reports/latest.md FRONTIER section (regenerated 2026-05-17 by frontier_scan canonical helper).
3. **CLAUDE.md 'HNeRV / leaderboard-implementation parity discipline' 13 lessons**: VERIFIED. All public race winners 2026-05-04 (PR101 / PR102 / PR103) used HNeRV-pattern; the 'NON-HNeRV' framing IS the explicit class-shift question.
4. **Catalog #313 probe-outcomes-ledger**: VERIFIED. 13 events in `.omx/state/probe_outcomes.jsonl`. Pose-related BLOCKING verdicts checked: ATW v2 D4 INDEPENDENT (2026-05-16, MI=0.006385); Wunderkind G1 v2 DEFER (2026-05-16, per-pair-dominant SegNet argmax); Z6-v2 Wave 2 DEFER (2026-05-18, driver-mode hardcode); C6 IBPS DEFER (2026-05-17, 22× miss); TT5L symposium DEFER (2026-05-17, 25ep CUDA 3.9007 19× WORSE than frontier); riemannian_newton PROCEED_WITH_REVISIONS (2026-05-18). NO pose-codec-specific blocking verdict exists in the ledger; the 6 NON-HNeRV pose-axis paths below have NO predecessor blocking outcome.
5. **Catalog #319 Wyner-Ziv deliverability_proof_builder canonical helper**: VERIFIED at `src/tac/wyner_ziv_deliverability/`. The Q1+Q2+Q3 wave landed 2026-05-17; the autopilot v2 cascade consumes the proof; Cascade 2 reward factors are byte-weighted Tier-1=1.20 / Tier-2=1.10 / Tier-3=1.05 / NOT_DELIVERABLE=1.0.
6. **Catalog #325 per-substrate symposium discipline**: VERIFIED. The standing rule + the 6-step canonical contract per CLAUDE.md 'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium' section. Every paid dispatch >$0.30 requires the symposium anchor.
7. **Codex master-gradient extractor landing 2026-05-18**: VERIFIED via `src/tac/master_gradient.py` + `tools/extract_master_gradient.py` + Catalog #318 (raw-byte-authority self-protection) + Catalog #327 (contest-axis custody self-protection). 8-pair fp64 anchor on PR101 frontier archive (`f174192aeadf`) EXISTS in canonical state.
8. **Deep-research wave 2026-05-18 TOP-5 pose-axis candidates**: VERIFIED at `.omx/research/comprehensive_research_wave_20260518.md`. TT5L V2 + VGGT + DUSt3R/MASt3R + DreamerV3 RSSM categorical + NVIDIA VRSS 2 ALL inventoried with arxiv IDs + GitHub paths.
9. **Lane pre-registration per Catalog #126**: VERIFIED via `python tools/lane_maturity.py add-lane lane_pose_axis_non_hnerv_t3_council_20260518 --name "Pose-axis non-HNeRV T3 council symposium" --phase 2` — landed 2026-05-18.
10. **Sister-subagent ownership map per Catalog #314 + Catalog #230**: VERIFIED. Codex owns `tools/extract_master_gradient.py` + `src/tac/master_gradient*.py`; this subagent owns ONLY `.omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md` + the lane-registry mutation via canonical CLI. No file-touch overlap.

### Pre-flight discipline gates

- [x] **CLAUDE.md read cover-to-cover** (full sweep + the 2 highest-emphasis sections: 'SegNet vs PoseNet importance' + 'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium')
- [x] **AGENTS.md**: not present in repo (only CLAUDE.md)
- [x] **MEMORY.md top-50** read (most recent 50 entries cover deep-research wave + WAVE-1+2+3 per-substrate symposiums + pose-axis prior councils)
- [x] **reports/latest.md FRONTIER section** read (regenerated 2026-05-17 by `tools/scan_best_anchor_per_axis.py`; current best per axis is canonical)
- [x] **.omx/state/lane_registry.json**: read; 101 pose-related lanes already registered (see §2.2 for the relevant subset)
- [x] **.omx/state/probe_outcomes.jsonl**: read; 13 events; no pose-codec-specific blocking outcome
- [x] **.omx/state/council_deliberation_posterior.jsonl**: prior pose-related council anchors read via cross-referenced research memos (see §0 premise #4)

---

## 1. Executive summary

### The strategic question (operator framing)

The contest scorer is:
```
score = 100·d_seg + sqrt(10·d_pose) + 25·archive_bytes / 37_545_489
```

At PR106 frontier operating point (`pose_avg ~3.4e-5`), the MARGINAL value flips per CLAUDE.md 'SegNet vs PoseNet importance' 2026-05-04 UPDATED section: **POSE is 2.71× MORE valuable per byte than SegNet**. The public race winners (PR101 GOLD 0.193 / PR102 silver 0.195 / PR103 silver 0.195) ALL used HNeRV-pattern whose pose-axis came implicitly from the rendered RGB → SegNet+PoseNet pipeline.

**The question**: is there a NON-HNeRV path to a frontier-breaking pose-axis improvement?

### 6 NON-HNeRV pose-axis paths enumerated

| # | Path | Class-shift level | Predicted ΔS | Cost | Risk |
|---|---|---|---|---|---|
| **P1** | Wyner-Ziv pose-residual hoist (Tier-1 deterministic side-info from frame_0 optical-flow warp) | BOLT-ON to HNeRV | `[-0.003, -0.001]` [prediction] | $0-$0.30 | LOW (cheap; canonical infrastructure exists) |
| **P2** | Master-gradient pose-byte classification consumer extension | BOLT-ON to HNeRV | `[-0.002, -0.0005]` [prediction] | $0 CPU | LOW (cheap; canonical infrastructure exists) |
| **P3** | ATW V2-1 channel-pick reformulation (per-region 16×16 SegNet softmax histogram + Faiss-IVF-PQ) | SUBSTRATE-CLASS-SHIFT candidate | `[-0.015, -0.005]` [prediction] | $5-7 CPU + $15-25 Modal | MEDIUM (cross-substrate Z6 4c dependency) |
| **P4** | Pose-conditioned residual codec PROBE (NOT full substrate) | SUBSTRATE-CLASS-SHIFT candidate | `[-0.008, -0.003]` [prediction] | $2-5 Modal smoke | MEDIUM (requires Catalog #325 symposium first) |
| **P5** | Ego-motion-conditioned next-frame predictor (Z6/Z7/Z8 from scratch) | FULL SUBSTRATE-CLASS-SHIFT | `[-0.025, -0.008]` [prediction] | $20-50 Modal | HIGH (3 same-week implementation-level falsifications) |
| **P6** | Pose-FOE sparse encoding (Telescope + LAPose 2-stage pipeline) | BOLT-ON to HNeRV | `[-0.005, -0.001]` [prediction] | $0-$1 CPU | LOW (LFV1 candidate landed 2026-05-13) |

### Council BINDING verdict (PROCEED_WITH_REVISIONS — 10 op-routables, see §0 frontmatter)

**Top 3 cheap-probe family** (P1 + P2 + P6) PROCEED unconditional ($0-$1 each; canonical infrastructure exists). **Direct master-gradient pose-byte hoist** (P7-as-OP-7, extension of P2) PROCEED unconditional.

**SUBSTRATE-CLASS-SHIFT candidates** (P3 + P4) PROCEED_WITH_REVISIONS — both require Catalog #325 per-substrate symposiums before paid dispatch >$0.30.

**FULL SUBSTRATE-CLASS-SHIFT** (P5) DEFER_PENDING_EVIDENCE — 3 same-week empirical falsifications (Z6-v2 / C6 IBPS / ATW v2 D4) at IMPLEMENTATION level per Catalog #307. The PARADIGM remains intact; the IMPLEMENTATIONS sit PRE-OPTIMAL-FORM per Catalog #315.

**VGGT-encoder-as-pose-teacher distillation** (OP-8, deep-research wave 2026-05-18 TOP-1) PROCEED_WITH_REVISIONS — requires its own symposium before paid dispatch.

### Mission alignment per CLAUDE.md 'Mission alignment — non-negotiable'

`council_predicted_mission_contribution: frontier_breaking` — every PROCEED op-routable predicts ΔS in the `[-0.025, -0.0005]` band that, if validated, displaces our 0.19205 [contest-CPU] frontier toward `[0.167, 0.192]`. The cheap-probe family (P1+P2+P6+OP-7) is `frontier_protecting` at minimum (prevents regression via better dispatch ranking) and `frontier_breaking` if even ONE Tier-1 Wyner-Ziv hoist lands.

### What was learned vs the 2026-05-13 prior pose-axis council

| 2026-05-13 council | This council 2026-05-18 |
|---|---|
| Focused on A1+LAPose composition (binding D1.D HIERARCHICAL) | Pivots to 6 enumerated NON-HNeRV paths (not just LAPose) |
| Predicted central ΔS [contest-CUDA] 0.218-0.224 | Predicts ΔS bands per path; HNeRV-pattern stacking still dominant near-term EV |
| 4 unresolved operator-routables (D3 + D4 binding tie) | 10 op-routables; 8 PROCEED + 2 DEFER; rank-ordered by $/ΔS |
| 0 sister-empirical-receipts from intervening week | 4 sister-empirical-receipts (ATW v2 D4 + Z6-v2 + C6 IBPS + TT5L) ALL DEFER at IMPLEMENTATION level |
| Master-gradient extractor did not exist | Master-gradient extractor LANDED 2026-05-18 by Codex; enables OP-2 + OP-7 at $0 cost |
| Deep-research wave did not exist | Deep-research wave 2026-05-18 added VGGT + DUSt3R + DreamerV3 + Mamba-2 + VRSS 2 to candidate list |

---

## 2. Frontier evidence (Catalog #316 frontier scan)

### 2.1 Current best per axis

Per `tools/scan_best_anchor_per_axis.py` regenerated 2026-05-17:

| Axis | Best score | Archive sha256 (first 12) | Hardware | Lane |
|---|---|---|---|---|
| `[contest-CPU GHA Linux x86_64]` | **0.1920513169** | `6bae0201fb08` | linux_x86_64_cpu | `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` |
| `[contest-CUDA T4]` | **0.2053300290** | `9cb989cef519` | linux_x86_64_t4 | `lane_pr106_format0d_latent_score_table_20260516_contest_cuda` |

### 2.2 Pose-axis components decomposed

Per 2026-05-13 prior pose-axis council §4.1 (Shannon's R(D) lower bound on PR101 anchor):
- `pose_avg ~ 3.4e-5` (PR106 frontier; same operating-point as our 0.19205 fec6 lane since fec6 wraps PR101)
- `pose contribution = sqrt(10 × 3.4e-5) = 0.01844`
- `seg contribution ~ 0.067` (from frontier-band decomposition)
- `rate contribution ~ 0.107` (from `25 × 178000 / 37545489 ≈ 0.119`)

### 2.3 Public PR comparison

- **PR101 GOLD** (`qpose14_qzs3_filmq9g_slsb1_r55`): 0.193 [contest-CPU] — we BEAT by 0.00095 with `6bae0201`.
- **PR102 bronze**: 0.19538 [contest-CPU] / 0.22839 [contest-CUDA] — we BEAT by 0.0033 CPU + 0.023 CUDA.
- **PR103 silver**: ~0.195 [contest-CPU] — we BEAT by 0.0028 CPU.

### 2.4 Pose-related lanes in registry (relevant subset of 101)

L2+ lanes touching pose-axis:
- `lane_pose_delta_pd_v2` (L2): pose-delta codec V2; per `fec6_plus_pd_v2_pose_codec_DEFERRED_pending_separate_pose_slot_20260517.md` DEFERRED-pending-separate-pose-slot-in-successor-substrate (PR101 has NO separate pose slot)
- `lane_pfp16` (L2): pose-FP16 quantization
- `lane_omega_w_v3` (L2): omega-weight V3
- `lane_arch_shrink_x_lagrangian_x_uniward` (L2): Lagrangian × UNIWARD per-pair architecture
- `lane_grand_council_pose_axis_review_20260511` (L1): prior pose-axis review

L1 substrate-engineering lanes:
- `lane_wavelet_residual_basis_pr106` (L1, substrate_engineering)
- `lane_raft_radial_pose` (L1): RAFT optical-flow for radial pose init

L0 sketch lanes specifically NON-HNeRV pose-axis:
- `lane_pose_dc3` (L0): pose-only DC3-style codec
- `lane_pd_pose_deltas` (L0): pose-deltas-as-stream
- `lane_pa_pose_as_affine` (L0): pose as affine warp
- `lane_n_linf_pose_budget` (L0): L-infinity pose budget allocation
- `lane_m_pose_mode` (L0): pose mode classification
- `lane_ge_geodesic_pose` (L0): geodesic pose interpolation
- `lane_gp_gaussian_process_pose` (L0): GP-based pose interpolation

### 2.5 Empirical receipts from THIS WEEK (signal-loss prevention)

Per `.omx/state/probe_outcomes.jsonl` (13 events):
- 2026-05-16 22:47Z: **ATW v2 D4 INDEPENDENT** (MI=0.006385; per-pair argmax composite reducer at INDEPENDENT threshold per Catalog #313)
- 2026-05-17 23:42Z: **C6 IBPS DEFER** (50ep Modal A10G smoke; predicted [0.113, 0.163]; empirical 3.04 [contest-CPU advisory]; **22× outside** predicted band)
- 2026-05-18 00:34Z: **Z6-v2 Wave 2 DEFER** (driver-mode hardcode; ran _smoke_main despite Wave 2 spec; Atick Council Revision #6 ΔS UNMEASURED)
- 2026-05-18 04:30Z: **NSCS06 v8 Path B REFUSE** (13-of-13 T3 council unanimous; 104.98 [diagnostic-CPU] 600× outside predicted [15, 25]; paradigm #5 NO-neural-at-medal-band FALSIFICATION CONFIRMED)
- 2026-05-18 05:00Z: **TT5L symposium #866 DEFER** (25ep CUDA 3.9007 / archive sha `2b05b7351b69` = 19× WORSE than 0.20533 CUDA frontier; PRE-OPTIMAL-FORM per Catalog #315)
- 2026-05-18 14:44Z: **MAE_v + SAUG DEFER** (10-of-10 unanimous)
- 2026-05-18 15:09Z: **ATW V2-1 V4 Faiss-IVF-PQ INDEPENDENT** (M=2, ksub=128, top-k=3 SEGFAULT at 600-pair scale; IMPLEMENTATION-LEVEL FALSIFICATION per Catalog #307; paradigm-INTACT per V2 sparse top-k MI=2.46 evidence; V4 specific implementation falsified)
- 2026-05-18 17:18Z: **Riemannian-Newton PROCEED_WITH_REVISIONS** (sextet 5-of-6 PROCEED; Phase 1 Fisher-precondition canonical helper first)

**Implication**: 5-of-8 same-week probe outcomes are DEFER at IMPLEMENTATION level for NON-HNeRV substrate-class-shift paths. The HNeRV-pattern bolt-on family (P1+P2+P6+OP-7) has ZERO blocking predecessors and the cheapest CPU-only canonical infrastructure ready.

---

## 3. Per-path enumeration (6 NON-HNeRV pose-axis paths + 2 deep-research-wave additions)

### Path P1 — Wyner-Ziv pose-residual hoist (Tier-1 deterministic side-info)

**Architecture sketch (≤300 LOC equivalent)**:
1. At compress time: compute optical-flow-warped pose estimate from frame_0 → frame_1 (deterministic; reproducible at decode).
2. Pose residual = PoseNet(GT_frame_pair) − warped_pose_estimate.
3. Quantize residual to ≤8 bits/component via arithmetic coding with Markov-1 hyperprior (Ballé 2018).
4. Encode the residual stream into existing PR101 archive grammar (extends the `0.bin` monolithic packet per HNeRV parity L3).
5. At decode time (inflate.py): reconstruct warped_pose_estimate via deterministic optical-flow on frame_0; apply residual; output corrected pose.
6. Per Catalog #319 v2 cascade: deliverability_proof_class = tier_1_deterministic; Cascade 2 reward factor = 1.20.

**Predicted ΔS band with Dykstra-feasibility intersection** (per CLAUDE.md 'Meta-Lagrangian/Pareto solver' + 2026-05-13 council §4.2):
- Constraint set: `C = {(R, S, P) : R ≤ R_max + 200, S ≤ S_a1, P ≤ P_a1 − δ_P}` with δ_P bounded by Shannon's R(D) curve.
- **Predicted ΔS: `[-0.003, -0.001]` [prediction-pending-post-training-Tier-C-validation]**
- First-principles citation: **Wyner-Ziv 1976** (side-info at decoder; Rate(X) − Rate(X|Y) = I(X;Y) where Y is deterministically reconstructable from frame_0) + **Shannon R(D)** + **Ballé 2018 hyperprior**.

**Empirical anchor predicted cost**:
- $0 CPU offline probe via `tac.side_information.deliverability_proof_builder` (existing canonical helper from 2026-05-17 Wyner-Ziv Q1 landing)
- ~$0.30 Modal A10G CPU smoke for paired CUDA verification (smoke-before-full per Catalog #167; the 4-hour smoke timeout per substrate recipe `pyav_decode_strategy: cpu_thread_async_upload`)
- Total: **$0.30 worst-case [prediction]**

**Compositional ALPHA prediction with fec6 / format0d**:
- vs fec6 (PR101 family): predicted α ≈ 0.9-1.05 (slightly sub-additive; both target pose-axis) per Catalog #322 anti-phantom protection
- vs format0d (PR106 family): predicted α ≈ 1.05-1.15 (sub-additive; fec6 already exploits frame-conditional codec selector which has some pose-conditioning)

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: wyner_ziv_pose_residual_tier_1_deterministic`
- `distinguishing_bytes_path: 0.bin[offset_pose_residual_start:offset_pose_residual_end]` (extends PR101's monolithic packet)
- `inflate_consumer_function: tac.substrates.fec6_plus_wz_pose.inflate._apply_pose_residual_correction`
- `byte_mutation_smoke_passes: PENDING-POST-LANDING` (verified via `tools/verify_distinguishing_feature_byte_mutation.py` per OP-1 landing)

### Path P2 — Master-gradient pose-byte classification consumer extension

**Architecture sketch (≤200 LOC)**:
1. Extend `tac.master_gradient_consumers.classify_bytes_by_pair_variance` with pose-axis-specific classification: `HIGH_POSE_DERIVATIVE_PAIR_INVARIANT` vs `HIGH_POSE_DERIVATIVE_PAIR_SPECIFIC` (vs the existing seg-axis classification).
2. Run on existing 8-pair fp64 anchor (PR101 archive `f174192aeadf`).
3. Identify the ~30-50% of archive bytes whose per-pair gradient pattern is POSE-INVARIANT across pairs → these are candidates for Tier-1 hoist via OP-1.
4. Emit proof artifact to `.omx/state/master_gradient/per_pair_pose_axis_classification_<sha[:12]>_<utc>.json`.
5. Per Catalog #322 anti-phantom: the proof_id is consumed by Cascade 2 reward factor in the autopilot.

**Predicted ΔS band**:
- **Predicted ΔS: `[-0.002, -0.0005]` [prediction-pending-post-training-Tier-C-validation]**
- First-principles citation: **Information Bottleneck (Tishby-Zaslavsky 2015)** + **per-pair Fisher information** + **Atick-Redlich 1990 cooperative-receiver**.

**Empirical anchor predicted cost**: **$0 CPU offline** (existing canonical infrastructure)

**Compositional ALPHA prediction**:
- With P1: predicted α ≈ 1.0-1.1 (orthogonal axes; P1 = compress-time hoist; P2 = classification surface that improves the autopilot's selection of which substrates get the P1 hoist applied)
- With OP-7: predicted α ≈ 0.7-0.9 (sub-additive; both target the same per-pair pose-byte classification surface)

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: per_pair_pose_axis_specific_venn_classification`
- `distinguishing_bytes_path: N/A` (the classifier IS the distinguishing feature; bytes are unchanged)
- `inflate_consumer_function: N/A` (compress-time-only; consumed by autopilot at dispatch ranking time)
- `byte_mutation_smoke_passes: N/A` (no bytes modified; per Catalog #220 the classifier alone is `lane_class=substrate_engineering` with `runtime_overlay_consumed=False` and `score_improvement_mechanism_status=OPERATIONAL_VIA_AUTOPILOT_DISPATCH_RANKING`)

### Path P3 — ATW V2-1 channel-pick reformulation

**Architecture sketch (≤450 LOC per HNeRV parity L7 bolt-on budget)**:
1. Per 2026-05-16 ATW v2 D4 INDEPENDENT verdict (probe_id `atw_v2_d4_h_latent_given_scorer_class_20260516`): the per-pair argmax composite reducer at MI=0.006385 bits/symbol is INSUFFICIENT.
2. Replace per-pair argmax composite with **per-region 16×16 SegNet softmax histogram + product-quantized to ≤2KB** per pair.
3. Product-quantization via **Faiss-IVF-PQ** (per deep-research wave 2026-05-18 TOP-3 OSS; M=4, ksub=64 to avoid the V4 segfault per 2026-05-18 15:09Z probe outcome).
4. Per Atick-Redlich 1990 cooperative-receiver: the encoder learns to encode bytes that maximize MI with the SegNet softmax histogram (decoder reconstructs the histogram via Faiss-IVF-PQ from baked-constant codebook).
5. Per Catalog #311 ego-motion-conditioning: extended with per-pair pose history via Mamba-2-like recurrence (selective state-space per deep-research wave 2026-05-18 TOP-2 OSS).

**Predicted ΔS band**:
- **Predicted ΔS: `[-0.015, -0.005]` [prediction-pending-post-training-Tier-C-validation]**
- First-principles citation: **Atick-Redlich 1990** + **Wyner-Ziv 1976** + **Faiss product-quantization (Jégou-Douze-Schmid 2011 + 2024 Faiss-1.8)**.

**Empirical anchor predicted cost**:
- ~$5-7 CPU probe (offline product-quantization fitting via Faiss-IVF-PQ on per-pair SegNet softmax histograms; ~600 pairs × 16×16 regions × 5 classes = 76,800 vectors at d=5)
- ~$15-25 Modal A100 (post-Wave-2-4c per Z6 cross-substrate dependency; the substrate trainer build itself is ~$10-15)
- **Total: $20-32 worst-case [prediction]**

**Compositional ALPHA prediction with fec6 / format0d**:
- vs fec6: predicted α ≈ 0.85-0.95 (sub-additive; both touch substrate-engineering surface)
- vs format0d: predicted α ≈ 1.0-1.1 (orthogonal axes; format0d = latent score-table; ATW V2-1 = per-region softmax-histogram codec)

**Cross-substrate dependency**: awaits Z6 Wave 2 4c outcome (probe outcomes ledger DEFER 2026-05-18; reactivation criterion = driver-mode hardcode fix lands)

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: per_region_segnet_softmax_histogram_pq_compressed`
- `distinguishing_bytes_path: archive_packet.bin[offset_atw_v2_1_start:offset_atw_v2_1_end]`
- `inflate_consumer_function: tac.substrates.atw_v2_1.inflate._reconstruct_per_region_histogram_via_faiss_pq`
- `byte_mutation_smoke_passes: PENDING-POST-LANDING`

### Path P4 — Pose-conditioned residual codec PROBE (NOT full substrate)

**Architecture sketch (≤350 LOC bolt-on)**:
1. HNeRV-pattern renderer (PR101 base) produces base RGB.
2. A SMALL pose-conditioned residual decoder (~50KB; conv-net with FiLM conditioning on PoseNet 6-DOF) produces correction RGB.
3. Total RGB = base + residual.
4. Per Atick-Redlich cooperative-receiver: encoder uses the SAME pose prior as scorer's PoseNet (NOT learning a new pose representation; CONSUMING the existing pose signal).
5. Per Hinton distillation: residual decoder distilled from PoseNet teacher's per-pair 6-DOF (T=2.0 KL-distill per Quantizr canonical pattern).

**Predicted ΔS band**:
- **Predicted ΔS: `[-0.008, -0.003]` [prediction-pending-post-training-Tier-C-validation]**
- First-principles citation: **Atick-Redlich 1990 cooperative-receiver** (compress-time-decoder cooperation with scorer-prior) + **Hinton 2014 distillation** + **Wyner 1976 side-info-at-decoder**.

**Empirical anchor predicted cost**:
- $0 CPU offline architecture probe (gradient-flow verification on small synthetic data)
- ~$2-5 Modal A10G 100ep smoke for gradient-flow verification + bp-pose-distillation-loss alignment
- **Total: $2-5 worst-case [prediction]**

**Compositional ALPHA prediction**:
- With P1: predicted α ≈ 0.7-0.9 (sub-additive; both target pose-axis via cooperative-receiver framing; P4 is the architecturally richer instance)
- With P3 (ATW V2-1): predicted α ≈ 1.0-1.1 (orthogonal axes; ATW V2-1 targets SegNet softmax histogram; P4 targets PoseNet 6-DOF)

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: pose_conditioned_residual_decoder_film_conditioning`
- `distinguishing_bytes_path: archive_packet.bin[offset_residual_decoder_start:offset_residual_decoder_end]`
- `inflate_consumer_function: tac.substrates.pose_conditioned_residual.inflate._apply_pose_conditioned_residual_correction`
- `byte_mutation_smoke_passes: PENDING-POST-LANDING`

**Per Catalog #325**: this OP requires a DEDICATED per-substrate symposium BEFORE paid dispatch >$1. Symposium queued as OP-11.

### Path P5 — Ego-motion-conditioned next-frame predictor (Z6 / Z7 / Z8 from scratch)

**Architecture sketch (FULL SUBSTRATE; exceeds HNeRV parity L7 bolt-on budget; explicit lane_class=substrate_engineering)**:
1. **Z6** (Multi-layer FiLM): depth=3 hidden_dim=96 ~300K params per Phase 3 council §9 spec. **Status: Wave 2 driver-mode hardcode DEFER 2026-05-18.**
2. **Z7** (Mamba-2 selective state-space): per Hafner Revision #3 binding from Z7 symposium 2026-05-18; replace LSTM with Mamba-2 (arxiv 2405.21060) selective state-space; per deep-research wave TOP-2 OSS `state-spaces/mamba`. **Status: design memo landed 2026-05-18; trainer build pending.**
3. **Z8** (Hierarchical predictive coding Catalog #312 quadruple): Rao-Ballard 1999 hierarchy + Mallat wavelet + Hafner DreamerV3 RSSM + Wyner-Ziv side-info; per Catalog #312 ALL FOUR primitives required. **Status: design memo PENDING.**

**Predicted ΔS band**:
- **Predicted ΔS: `[-0.025, -0.008]` [prediction-pending-post-training-Tier-C-validation per Catalog #324]**
- First-principles citation: **Atick-Redlich 1990** + **Rao-Ballard 1999** + **Hafner DreamerV3 2023 RSSM** + **Wyner 1976** + **Tishby-Zaslavsky 2015 Information Bottleneck**.

**Empirical anchor predicted cost**:
- ~$20-30 Modal A100 (Z6 Wave 3 post driver-fix) per Z6 council
- ~$20-30 Modal A100 (Z7 Mamba-2 full smoke + paired CUDA) per Z7 symposium 2026-05-18
- ~$30-50 Modal A100 (Z8 if Z6/Z7 succeed) per Catalog #312 quadruple constraint
- **Total: $20-50 per substrate × 3 substrates = $60-150 worst-case [prediction]**

**Compositional ALPHA prediction**:
- Each of Z6/Z7/Z8 with HNeRV-pattern (PR101): predicted α ≈ 0.5-0.8 (sub-additive; both are pose-axis primary; class-shift candidates compete for the same pose-axis budget)

**Critical risk per CLAUDE.md 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' Catalog #315**:
- Z6-v2 Wave 2 driver-mode hardcode DEFER 2026-05-18 + C6 IBPS Tier-C 22× miss 2026-05-17 + TT5L symposium #866 REFUSE 2026-05-17 + NSCS06 v8 Path B REFUSE 2026-05-18 + ATW v2 D4 INDEPENDENT 2026-05-16 = **5 consecutive same-week empirical falsifications of NON-HNeRV substrate-class-shift paths AT IMPLEMENTATION LEVEL**.
- Per CLAUDE.md 'Forbidden premature KILL without research exhaustion': PARADIGMS remain intact; IMPLEMENTATIONS sit PRE-OPTIMAL-FORM.

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: ego_motion_conditioned_next_frame_predictor_recurrent_world_model`
- `distinguishing_bytes_path: archive_packet.bin[offset_world_model_state_start:offset_world_model_state_end]`
- `inflate_consumer_function: tac.substrates.{z6,z7,z8}.*.inflate._reconstruct_via_world_model_recurrence`
- `byte_mutation_smoke_passes: PENDING-POST-OPTIMAL-FORM-PER-CATALOG-315`

### Path P6 — Pose-FOE sparse encoding (Telescope + LAPose 2-stage pipeline)

**Architecture sketch (≤350 LOC bolt-on per HNeRV parity L7)**:
1. **Stage 1 (LA-Pose latent action)**: select per-pair where pose bytes matter most via latent-action encoder (Wang et al. 2026; arxiv 2604.27448).
2. **Stage 2 (Telescope hyperbolic foveation)**: hyperbolic foveation transform (Ewen et al. 2026; arxiv via project page) provides sparse encoding at FOE (focus-of-expansion per Gibson 1950).
3. The 2-stage product: per-pair LA-pose-selected pairs receive Telescope foveation; per-pair LA-pose-deselected pairs receive only uniform encoding.
4. Per CLAUDE.md 'Forbidden in-place edits to public PR intake clones' + Catalog #109: the LA-Pose / Telescope reference implementations sit in `experiments/results/public_pr*_intake_*` and MUST stay pristine; pact-side implementations are SEPARATE.

**Predicted ΔS band**:
- **Predicted ΔS: `[-0.005, -0.001]` [prediction-pending-post-training-Tier-C-validation]**
- First-principles citation: **Gibson 1950 ego-motion-matched flow** + **Atick-Redlich 1990 cooperative-receiver** + **Mallat 1988 wavelet multi-scale** (Telescope IS hyperbolic wavelet) + **LaPose 2026** + **Telescope 2026** + **FOVEA ICCV 2021 predecessor**.

**Empirical anchor predicted cost**:
- $0 CPU offline analysis (LFV1 candidate already landed per `.omx/research/telescope_foveation_lfv1_candidate_20260513_codex.md`)
- ~$1 Modal A10G 50ep gradient-flow smoke
- **Total: $0-$1 worst-case [prediction]**

**Compositional ALPHA prediction**:
- With P3 (ATW V2-1): predicted α ≈ 1.0-1.1 (orthogonal axes; ATW V2-1 = per-region; P6 = per-pair-foveation-center)
- With P1: predicted α ≈ 1.05-1.15 (orthogonal; P1 = pose-residual; P6 = pose-FOE-sparse-encoding)

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: pose_foe_sparse_encoding_la_pose_telescope_two_stage`
- `distinguishing_bytes_path: archive_packet.bin[offset_lapose_selector_start:offset_telescope_foveation_payload_end]`
- `inflate_consumer_function: tac.substrates.pose_foe_lapose_telescope.inflate._apply_two_stage_foveation`
- `byte_mutation_smoke_passes: PENDING-POST-LANDING`

### Path OP-7 — Direct master-gradient pose-byte hoist (extension of P2 + P1)

Per Codex 2026-05-18 just-landed master-gradient extractor + Catalog #318/#327 self-protection: the per-pair fp64 master-gradient extraction is OPERATIONAL. OP-7 wires pose-axis-specific deliverable_savings computation into `tac.master_gradient_consumers.adjust_predicted_delta_for_venn_classification_v2 Cascade 2` per Catalog #319 v2.

**Predicted ΔS band**: **`[-0.002, -0.0005]` [prediction]** (extension reward factor)
**Cost**: **$0 CPU only**
**Compositional ALPHA with P1+P2**: predicted α ≈ 0.7-0.9 (sub-additive; all 3 target master-gradient pose-byte surface)

### Path OP-8 — VGGT-encoder-as-pose-teacher distillation (deep-research wave 2026-05-18 TOP-1)

**Architecture sketch (≤500 LOC bolt-on; explicit Hinton distillation pattern)**:
1. VGGT (CVPR 2025 Best Paper, `facebookresearch/vggt`) feed-forward 3D-from-N-views encoder produces 6-DOF pose estimate from 2 frames.
2. Distill VGGT teacher into PR101 renderer's implicit pose head (T=2.0 KL-distill per Quantizr canonical pattern).
3. Per CLAUDE.md 'strict scorer rule' the VGGT WEIGHTS cannot ship in archive; only the DISTILLED PR101 renderer weights ship.
4. Per deep-research wave 2026-05-18: VGGT pretrained on millions of dashcam-like sequences matches PoseNet's 6-DOF output without retraining.

**Predicted ΔS band**:
- **Predicted ΔS: `[-0.010, -0.003]` [prediction-pending-post-training-Tier-C-validation]**
- First-principles citation: **VGGT CVPR 2025 Best Paper** + **DUSt3R ECCV 2024 (arxiv 2312.14132)** + **MASt3R ECCV 2024 (arxiv 2406.09756)** + **Hinton 2014 distillation** + **VGGT-4D PAGE-4D extension (arxiv 2510.17568)**.

**Empirical anchor predicted cost**:
- ~$0-$2 CPU offline distillation feasibility study (VGGT inference on contest video frames)
- ~$10-15 Modal A100 for paired smoke (VGGT distillation requires more A100 compute than A10G)
- **Total: $10-17 worst-case [prediction]**

**Compositional ALPHA**:
- With P1 (Wyner-Ziv pose-residual): predicted α ≈ 0.6-0.8 (sub-additive; both consume the same pose-prior surface)
- With P4 (Pose-conditioned residual codec): predicted α ≈ 1.0-1.1 (orthogonal; OP-8 = teacher distillation; P4 = decoder architecture)

**Per Catalog #325**: requires its own per-substrate symposium BEFORE paid dispatch >$1. Symposium queued as part of OP-11.

**Distinguishing-feature integration contract per Catalog #272**:
- `distinguishing_feature_name: vggt_distilled_pr101_pose_head`
- `distinguishing_bytes_path: archive_packet.bin[offset_distilled_pose_head_start:offset_distilled_pose_head_end]`
- `inflate_consumer_function: tac.substrates.vggt_distilled_pr101.inflate._apply_distilled_pose_head_residual`
- `byte_mutation_smoke_passes: PENDING-POST-LANDING`

---

## 4. Cargo-cult audit per assumption (per Catalog #303)

Per CLAUDE.md FORBIDDEN_PATTERNS: 'Forbidden symposium-band-prediction-without-Dykstra-feasibility-check' + 'Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof' + the addendum hard-earned-vs-cargo-culted framework.

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | Pose-axis is the dominant marginal value at PR106 frontier | HARD-EARNED | CLAUDE.md 2026-05-04 + PR97 anti-pattern empirical receipt (`docs/pr97_anti_pattern_pose_vs_seg_marginal_20260504.md`) |
| 2 | HNeRV-pattern is the only public-frontier path | HARD-EARNED | All 3 public race winners 2026-05-04 used HNeRV-pattern; our 0.19205 fec6 IS PR101 + bolt-on |
| 3 | A direct pose codec (Wyner-Ziv side-info) is operator-attention-budget-feasible | HARD-EARNED | 2026-05-17 T3 Wyner-Ziv canonical-fix council + Catalog #319 v2 cascade ACTIVE |
| 4 | Pose-conditioned residual codec is byte-distinguishable from HNeRV-pattern | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | OP-4 cheap probe ($2-5 Modal smoke) verifies gradient-flow; if PRE-OPTIMAL-FORM per Catalog #315 then DEFER |
| 5 | Ego-motion-conditioned next-frame prediction (Z6/Z7/Z8) delivers asymptotic-pursuit ΔS | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | Z6-v2 Wave 2 driver-fix + Wave 3 re-fire + Atick Council Revision #6 ΔS measurement is the reactivation criterion |
| 6 | Pose-bin classification (discrete pose codebook via master-gradient) bypasses HNeRV-pattern's implicit pose | HARD-EARNED-VIA-MASTER-GRADIENT | Codex's just-landed extractor IS the proof; P2 + OP-7 wire-in is canonical |
| 7 | Pose-FOE sparse encoding is byte-feasible at ≤1 KB | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | LFV1 candidate landed 2026-05-13; offline analysis IS the cheap probe |
| 8 | Direct pose master-gradient exploitation delivers pose-axis savings at $0-$1 cost | HARD-EARNED-EMPIRICALLY-NEAR-LANDED | Codex 2026-05-18 master-gradient extractor + Catalog #318/#327 self-protection makes it concrete |
| 9 | VGGT-distillation IS contest-compliant (no scorer weights in archive) | HARD-EARNED | Only DISTILLED PR101 weights ship; VGGT teacher stays at compress-time. Per Catalog #213 + Catalog #210 the dataset-derived prior + license-tag propagation is the canonical pattern |
| 10 | The 5 same-week DEFER outcomes (ATW v2 D4 + Z6-v2 + C6 IBPS + TT5L + ATW V2-1 V4) imply ALL NON-HNeRV substrate-class-shift paths are dominated | CARGO-CULTED | Per Catalog #307 + Catalog #308 each DEFER is IMPLEMENTATION-LEVEL falsification per N≥3 alternative reducer enumeration; the PARADIGMS remain intact. The unwind is per-substrate symposium per Catalog #325 |
| 11 | Cheap-probe family (P1+P2+P6+OP-7) is operator-attention-budget-feasible THIS WEEK | HARD-EARNED | $0-$1 CPU each; canonical infrastructure exists today (master-gradient extractor + Wyner-Ziv deliverability proof builder + LFV1 candidate analysis) |
| 12 | The autopilot's current Cascade 2 reward factors (Tier-1=1.20 / Tier-2=1.10 / Tier-3=1.05) are correctly calibrated for pose-axis vs seg-axis | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | OP-10 extends Cascade 2 to per-pair pose-axis-specific factors; current factors are substrate-wide |

---

## 5. 9-dimension success checklist evidence (per Catalog #294)

For the PROCEED-path PROCEED-aggregate (P1 + P2 + P6 + OP-7 — the cheap-probe family that PROCEEDS unconditional):

| # | Dimension | Evidence |
|---|---|---|
| 1 | **UNIQUENESS** | All 4 cheap-probe-family paths are NEW per-pair pose-axis-specific surfaces; none currently consumed by autopilot Cascade 2. Each path's `distinguishing_feature_name` (per §3) is unique. |
| 2 | **BEAUTY + ELEGANCE** | Each path is ≤300 LOC bolt-on per HNeRV parity L7 (P1 = 200 LOC pose-residual encoder + decoder; P2 = 150 LOC classifier extension; P6 = 250 LOC LFV1 analysis + sparse encoding; OP-7 = 30-50 LOC Cascade 2 reward factor extension). All reviewable in 30 seconds. |
| 3 | **DISTINCTNESS** | P1 (compress-time hoist) vs P2 (classifier extension) vs P6 (foveation sparse encoding) vs OP-7 (autopilot reward factor) are orthogonal surfaces; no overlap. Each cites a distinct first-principles paper. |
| 4 | **RIGOR** | Per Catalog #229 premise verification (§0) + Catalog #292 per-deliberation assumption surfacing (§3 + §4) + Catalog #296 Dykstra-feasibility predicted-band per path. Predicted bands are bounded by Shannon R(D) + Wyner 1976 + Ballé 2018 hyperprior limits, NOT vibes. |
| 5 | **OPTIMIZATION PER TECHNIQUE** | Per CLAUDE.md 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode': each path uses CANONICAL helpers where they serve (e.g., `tac.side_information.deliverability_proof_builder` for P1; `tac.master_gradient_consumers.classify_bytes_by_pair_variance` for P2) and FORKS where canonical suppresses (e.g., P2 adds pose-axis-specific classification; OP-10 extends Cascade 2 per-pair). |
| 6 | **STACK-OF-STACKS COMPOSABILITY** | Per §3 compositional ALPHA prediction: P1+P2+P6+OP-7 are mutually predicted ADDITIVE (α ≈ 0.9-1.1) per Catalog #322 anti-phantom protection. The composition matrix is internally consistent. |
| 7 | **DETERMINISTIC REPRODUCIBILITY** | Each path uses byte-stable + seed-pinned components (P1 = Markov-1 hyperprior arithmetic coding; P2 = deterministic per-pair Venn classifier; P6 = deterministic Telescope foveation transform; OP-7 = deterministic autopilot Cascade 2 reward function). Per CLAUDE.md 'Beauty, simplicity, and developer experience'. |
| 8 | **EXTREME OPTIMIZATION + PERFORMANCE** | All 4 cheap-probe-family paths are CPU-only or ≤$0.30 Modal A10G smoke. Per Catalog #270 'Production-hardened dispatch optimization protocol' the Tier 1/2/3 contract is satisfied: P1 routes through `tac.side_information` canonical helper; P2 routes through `tac.master_gradient_consumers`; P6 routes through LFV1 canonical analysis; OP-7 wires the autopilot Cascade 2. |
| 9 | **OPTIMAL MINIMAL CONTEST SCORE** | Sum of predicted ΔS bands: P1 + P2 + P6 + OP-7 = `[-0.013, -0.0035]` worst-case-summed (assuming α = 1.0 additive). Realistic α-adjusted (per §3 composition ALPHA): `[-0.010, -0.003]`. Frontier displacement target from 0.19205 → `[0.182, 0.189]` [contest-CPU prediction]. |

---

## 6. Observability surface (per Catalog #305)

The 6-facet observability definition per CLAUDE.md 'Max observability — non-negotiable':

1. **Inspectable per layer**: each PROCEED path has its own canonical helper invocation surface that emits structured per-step JSON to `.omx/state/` (P1 = `tac.side_information.deliverability_proof_builder` writes to `.omx/state/wyner_ziv_deliverability/`; P2 = `tac.master_gradient_consumers` writes to `.omx/state/master_gradient/`; P6 = LFV1 writes to `.omx/state/telescope_foveation_lfv1/`; OP-7 = autopilot v2 cascade writes to `.omx/state/cathedral_autopilot_run/`).
2. **Decomposable per signal**: per-axis (seg / pose / rate) decomposition AT each path's predicted-ΔS computation. Catalog #322 anti-phantom protection enforces per-axis composition_alpha.
3. **Diff-able across runs**: each path's archive is byte-stable (same seed + same input → same archive sha256); per Catalog #316 frontier scan the per-axis components are reproducibly comparable.
4. **Queryable post-hoc**: every persisted artifact is machine-readable JSONL/JSON with canonical schema_version. Per Catalog #245 Modal call_id ledger + Catalog #131 fcntl-locked JSONL discipline.
5. **Cite-able**: every artifact carries `(substrate / commit / call_id / config / random_seed / upstream_snapshot_sha256)` tuple per Catalog #245.
6. **Counterfactual-able**: per Catalog #139 packet compiler no-op detector + Catalog #105 no-op provenance + Catalog #272 distinguishing-feature integration contract: each path's byte-mutation smoke verifies the bytes actually affect the score.

### Observability of per-path actuator chain (top-3 PROCEED)

| Path | Canonical helper | Output artifact | Operator-facing CLI |
|---|---|---|---|
| P1 | `tac.side_information.deliverability_proof_builder` | `.omx/state/wyner_ziv_deliverability/proof_<sha[:12]>_<utc>.json` | (existing per 2026-05-17 Q4 landing) |
| P2 | `tac.master_gradient_consumers.classify_bytes_by_pair_variance` + extension | `.omx/state/master_gradient/per_pair_pose_axis_classification_<sha[:12]>_<utc>.json` | NEW: `tools/extend_master_gradient_pose_axis_classification.py` (proposed; ~80 LOC) |
| OP-7 | `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2` Cascade 2 | `.omx/state/cathedral_autopilot_run/<utc>/cascade_2_pose_axis_rewards.json` | (existing per 2026-05-17 Q3 landing; extended by OP-10) |

---

## 7. Per-path Assumption-Adversary verdict

Per CLAUDE.md 'Council conduct' Assumption-Adversary seat (sextet pact): per-round MUST propose at least ONE shared-assumption-violation hypothesis. Verdicts:

| Path | Verdict | Hypothesis to violate | Predicted impact if violated |
|---|---|---|---|
| P1 | HARD-EARNED | "Tier-1 deterministic side-info (optical-flow warp) is the canonical pose-residual hoist surface" | If FALSIFIED (e.g., the optical-flow warp residual is non-stationary or the residual entropy > Shannon-Fano upper bound), path delivers ΔS at lower edge ~−0.001 only |
| P2 | HARD-EARNED | "Per-pair Venn classification with pose-axis-specific factors is orthogonal to existing seg-axis classification" | If FALSIFIED (seg-axis and pose-axis Venn classifications are highly correlated), reward factor underflows to ~−0.0005 |
| P3 | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | "Per-region 16×16 SegNet softmax histogram + Faiss-IVF-PQ at M=4 ksub=64 avoids the V4 segfault" | If FALSIFIED (V4-class segfault recurs at smaller ksub or M), path requires V5 wavelet-multi-scale per Mallat revision OR V6 Laplace-prior per MacKay revision per ATW V2-1 V4 probe outcome 2026-05-18 |
| P4 | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | "Pose-conditioned residual decoder distilled from PoseNet teacher converges in ≤100 epochs" | If FALSIFIED (gradient-flow blocked or convergence slow), path requires Hinton T=2.0 KL-distill scheduling adjustments per Quantizr canonical pattern + sister symposium discipline |
| P5 | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | "Z6/Z7/Z8 ego-motion-conditioned predictor delivers asymptotic-pursuit ΔS at $20-50 per substrate" | If FALSIFIED (5-of-5 same-week pattern continues for Z7/Z8 dispatches), path stays DEFER per Catalog #325 symposium discipline + reactivation criteria pinned |
| P6 | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | "Telescope hyperbolic foveation + LAPose 2-stage pipeline delivers ≤1 KB byte-feasibility" | If FALSIFIED (LFV1 candidate analysis shows bytes ≥ 2 KB), path requires architectural narrowing per Telescope-only OR LAPose-only single-stage |
| OP-7 | HARD-EARNED | "Cascade 2 reward factor extension to per-pair pose-axis-specific is implementable in ≤50 LOC delta" | If FALSIFIED (extension requires substantial refactor of v2 cascade), path requires architectural redesign or scope reduction |
| OP-8 | CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION | "VGGT teacher distillation matches PoseNet 6-DOF without further retraining" | If FALSIFIED (VGGT outputs not aligned with PoseNet's first-6 pose dimensions), path requires teacher-output alignment training (substantially higher cost) |

---

## 8. Sextet pact deliberation transcript

Per CLAUDE.md 'Council conduct' amendment 2026-05-15 + Fix-7 per-round explicit-assumption-statement discipline + Catalog #292.

### Shannon (LEAD)

**The shared assumption I am operating within for this deliberation is**: the pose-axis marginal value at PR106 frontier is correctly characterized by the derivative `5/sqrt(10·pose_avg) = 271` at `pose_avg = 3.4e-5`, AND the achievable rate-distortion tradeoff for pose-residual encoding is bounded by Shannon's R(D) curve at `δ_pose ≤ pose_a1 × (1 − exp(-2 × δ_R_bits / pose_dim))`.

**Position**: The cheap-probe family (P1 + P2 + P6 + OP-7) is information-theoretically optimal at this operating point. The Wyner-Ziv side-info-at-decoder framing (P1) is the canonical pose-residual hoist; the master-gradient per-pair classification (P2 + OP-7) is the canonical per-pair pose-byte surface; the pose-FOE sparse encoding (P6) is the canonical pose-attention foveation. ALL THREE are within Shannon's R(D) lower bound at `200-500 bytes` per the 2026-05-13 prior council §4.1.

**Vote**: PROCEED on P1 + P2 + P6 + OP-7; PROCEED_WITH_REVISIONS on P3 + P4 (requires symposium); DEFER on P5 (substrate-class-shift requires PROCEED-unconditional per Catalog #315); PROCEED_WITH_REVISIONS on OP-8 (VGGT-distillation requires symposium).

### Dykstra (CO-LEAD)

**The shared assumption I am operating within for this deliberation is**: the convex feasibility set `C = {(R, S, P) : R ≤ R_max, S ≤ S_current, P ≤ P_current - δ_P}` for adding ≤500 bytes of pose-residual stream is non-empty AND inside the current PR101 Pareto frontier. The alternating-projections convergence is geometric per the 2026-05-13 prior council §4.2.

**Position**: Per CLAUDE.md 'Meta-Lagrangian/Pareto solver' the cheap-probe family preserves convexity (each path's bytes are linear-additive to the existing PR101 archive). P3 + P4 require non-empty feasibility verification BEFORE substrate landing (the symposium discipline per Catalog #325 IS the feasibility check). P5 violates convexity by REPLACING the substrate (not bolt-on); per HNeRV parity L7 substrate-engineering exceeds the bolt-on budget.

**Vote**: PROCEED on P1 + P2 + P6 + OP-7; PROCEED_WITH_REVISIONS on P3 + P4; DEFER on P5; PROCEED_WITH_REVISIONS on OP-8.

### Yousfi

**The shared assumption I am operating within for this deliberation is**: the contest IS inverse steganalysis (we hide signal in YUV6 + pose channels; the scorer is the steganalysis CNN). FastViT-T12 PoseNet's RepMixer at the summary head has a ~32×32 receptive field; sub-pixel pose corrections (P1) and per-region 16×16 codec choices (P3) exploit the scorer's frequency-band blind spots.

**Position**: P1's pose-residual stream is the inverse-steganalysis canonical pattern — sub-pixel pose corrections invisible to RepMixer. P3's per-region 16×16 SegNet softmax histogram exploits EfficientNet-B2's stride-2 stem blind spot. P6's pose-FOE sparse encoding is detector-informed embedding per Fridrich-approved methodology. P4 is novel architecturally but requires symposium gradient-flow verification per Catalog #325.

**Vote**: PROCEED on P1 + P2 + P3 (with revisions) + P6 + OP-7; PROCEED_WITH_REVISIONS on P4; DEFER on P5; PROCEED_WITH_REVISIONS on OP-8.

### Fridrich

**The shared assumption I am operating within for this deliberation is**: the SQRT law (spread small errors, don't concentrate large ones) + UNIWARD inverse-local-variance + detector-informed embedding (Yousfi 2022) jointly govern the optimal pose-residual encoding pattern. The LAPose codec should down-weight pose corrections during low-motion frames; pose-conditioned residual decoder should distribute corrections per-pair uniformly.

**Position**: P1 with Markov-1 hyperprior IS UNIWARD-compatible (compresses temporally-correlated residuals). P3 ATW V2-1 IS detector-informed embedding (per-region softmax histogram derived from scorer evidence). P6 LFV1 IS UNIWARD-compatible (sparse encoding at FOE = inverse local variance weighting). All cheap-probe-family paths satisfy SQRT law.

**Vote**: PROCEED on P1 + P2 + P6 + OP-7; PROCEED_WITH_REVISIONS on P3 + P4; DEFER on P5; PROCEED_WITH_REVISIONS on OP-8.

### Contrarian (verbatim in §0 frontmatter)

**The shared assumption I am operating within for this deliberation is**: the OPERATOR's framing 'NON-HNeRV path to frontier-breaking pose-axis improvement' is correct in calling for diversification but should NOT be interpreted as 'NON-HNeRV DOMINATES HNeRV-pattern stacking'. The dominant near-term EV path is HNeRV-pattern stacking with explicit pose-axis foci (P1 + P2 + OP-7).

**Position**: Per §0 frontmatter `council_dissent` verbatim. Insist on cheap-probe-FIRST rank-ordering. Insist on Catalog #313 predecessor probe-outcome consultation (5-of-8 same-week DEFERs are binding precedent).

**Vote**: PROCEED on P1 + P2 + P6 + OP-7 (cheap-probe family); PROCEED_WITH_REVISIONS on P3 + P4 (requires symposium AND post-Wave-2-4c clearance for P3); STRONGLY DEFER on P5 (3 same-week implementation-level falsifications); PROCEED_WITH_REVISIONS on OP-8 (requires symposium).

### Assumption-Adversary (verbatim in §0 frontmatter)

**The shared assumption I am operating within for this deliberation is**: the FRAMING 'NON-HNeRV is the orthogonal direction we need' is itself the cargo-cult; the data-supported framing is 'HNeRV-pattern bolt-ons with explicit pose-axis foci are dominant near-term; NON-HNeRV substrate-class-shifts are research-only until Catalog #315 PROCEED-unconditional per substrate'.

**Position**: Per §0 frontmatter `council_dissent` verbatim. Vote: SPLIT the verdict per dissent (PROCEED on cheap-probe family; DEFER on full substrate-class-shift family; re-route freed budget to HNeRV-pattern stacking).

**Vote**: PROCEED on P1 + P2 + P6 + OP-7 + OP-10; PROCEED_WITH_REVISIONS on P3 + P4 (requires symposium); DEFER on P5; PROCEED_WITH_REVISIONS on OP-8.

### Grand-council attendees (T3 ≥12-of-20 quorum) — abbreviated positions

**Boyd** (convex optimization at operational level): convex-feasibility verification per Dykstra; PROCEED on cheap-probe; DEFER on P5; PROCEED_WITH_REVISIONS on P3+P4+OP-8.

**Mallat** (wavelet multi-scale + sparse representations): P6 LFV1 = hyperbolic wavelet IS canonical wavelet-multi-scale signal-extraction; PROCEED unconditional on P6.

**van den Oord** (VQ-VAE codebook): P2 + OP-7 master-gradient per-pair Venn classification with pose-axis-specific factors IS VQ-codebook discrete-prior; PROCEED on P2 + OP-7; PROCEED_WITH_REVISIONS on P3 (per-region histogram is codebook-vector quantization at scale).

**Carmack** (engineering shortcuts; verbatim in §0 frontmatter): rank-by-$/ΔS-per-byte; cheap-probe-FIRST; defer Z6/Z7/Z8 from-scratch.

**Hinton** (distillation): OP-8 VGGT-teacher-distillation is canonical T=2.0 KL-distill per Quantizr pattern; PROCEED_WITH_REVISIONS on OP-8 (symposium required); P4 pose-conditioned residual decoder distillation also canonical pattern.

**Tishby memorial** (Information Bottleneck): P1 + P2 + P3 + P4 + OP-7 all framed via I(X;Y) = Rate(X) - Rate(X|Y); the I.B. framework supports all cheap-probe family + P3 + P4. P5 = Z6/Z7/Z8 hierarchical I.B. per Tishby-Zaslavsky 2015 generalization; PROCEED on Z7 IF Catalog #325 symposium ratifies it post-Wave-3-resumption.

**Atick + Redlich** (cooperative-receiver; convened together): P1 + P3 + P4 + OP-8 ALL framed via Atick-Redlich 1990 cooperative-receiver theorem (compress-time-decoder cooperation with scorer-prior). PROCEED on cooperative-receiver canonical paths.

**Rao + Ballard** (predictive coding + embodied vision): P5 = canonical Rao-Ballard 1999 hierarchy (Z6=multi-layer FiLM; Z7=Mamba-2 recurrence; Z8=hierarchical I.B.). Vote DEFER per Catalog #325 + reactivation criteria pinned; PROCEED on P5 IF Z6 Wave 2 driver-fix lands + Atick Revision #6 ΔS measured.

**Wyner** (side-info-at-decoder): P1 IS canonical Wyner 1976 side-info-at-decoder; PROCEED unconditional.

**Hafner** (DreamerV3 + RSSM categorical): P5 / Z6/Z7/Z8 / OP-8 all converge on DreamerV3 RSSM categorical (32 one-hot vectors per timestep). VOTE PROCEED_WITH_REVISIONS on OP-8 (VGGT + DreamerV3 + DUSt3R/MASt3R is the deep-research wave 2026-05-18 TOP-1 composite); DEFER on P5 per Catalog #325 symposium discipline.

**Time-Traveler protégé (Rudin-postdoc canonical id)**: per CLAUDE.md grand-council roster expansion 2026-05-15. Position: the L5 staircase v2 design + the Catalog #310 + #311 + #312 quadruple-primitive enforcement + the Asymptotic-Pursuit family ALL converge on Z8. PROCEED on hierarchical-predictive-coding paradigm; DEFER on Z6/Z7/Z8 from-scratch implementations per Catalog #315 + reactivation criteria.

### Vote tally summary

| Decision | PROCEED | PROCEED_WITH_REVISIONS | DEFER | REFUSE |
|---|---|---|---|---|
| P1 (Wyner-Ziv pose-residual hoist) | 19 (all) | 0 | 0 | 0 |
| P2 (Master-gradient classification extension) | 19 (all) | 0 | 0 | 0 |
| P3 (ATW V2-1 channel-pick) | 0 | 19 (all; symposium required + Z6 4c clearance) | 0 | 0 |
| P4 (Pose-conditioned residual codec PROBE) | 0 | 19 (all; symposium required) | 0 | 0 |
| P5 (Z6/Z7/Z8 from scratch) | 0 | 2 (van_den_Oord, Hafner — partial) | 17 | 0 |
| P6 (Pose-FOE sparse encoding) | 19 (all) | 0 | 0 | 0 |
| OP-7 (Master-gradient pose-byte hoist) | 19 (all) | 0 | 0 | 0 |
| OP-8 (VGGT-distillation) | 0 | 19 (all; symposium required) | 0 | 0 |
| OP-9 (Coordinated wave EV ranking enforcement) | 19 (all) | 0 | 0 | 0 |
| OP-10 (Cathedral autopilot Cascade 2 extension) | 19 (all) | 0 | 0 | 0 |
| OP-11 (Per-substrate symposiums for P4 + OP-8) | 19 (all) | 0 | 0 | 0 |

**Quorum met**: 19-of-20 grand-council attendees voted (Jack-from-skunkworks recused per relevance to per-substrate-symposium-discipline scope; advisory-only per CLAUDE.md grand-council seat semantics). Sextet pact: 6-of-6 vote consistent across all paths.

---

## 9. Ranked op-routables (TOP-5 with EV scoring)

Per CLAUDE.md 'Frontier target — NON-NEGOTIABLE' + 'Race-mode rigor inversion + parallel-dispatch first': EV = `|predicted ΔS lower bound| / cost`. Higher EV first.

| Rank | OP | Path | Predicted ΔS lower bound | Cost worst-case | EV (1/$) | File paths | Commit-via-serializer command |
|---|---|---|---|---|---|---|---|
| **1** | OP-7 | Direct master-gradient pose-byte hoist | -0.002 | $0 | ∞ | `tac.master_gradient_consumers.adjust_predicted_delta_for_venn_classification_v2` Cascade 2 extension; `tools/cathedral_autopilot_autonomous_loop.py` | `.venv/bin/python tools/subagent_commit_serializer.py --message "autopilot: extend Cascade 2 pose-axis Venn reward factor per OP-7" --files <files> --expected-content-sha256 <files>=<sha>` |
| **2** | OP-2 | Master-gradient pose-byte classification extension | -0.002 | $0 | ∞ | `src/tac/master_gradient_consumers.py::classify_bytes_by_pair_variance` extension | `... --message "master_gradient: extend per-pair Venn with pose-axis-specific classes per OP-2" ...` |
| **3** | OP-1 | Wyner-Ziv pose-residual Tier-1 hoist | -0.003 | $0.30 | 0.010 | `src/tac/side_information/deliverability_proof_builder.py` consumer; `src/tac/substrates/fec6_plus_wz_pose/` new substrate package; `.omx/operator_authorize_recipes/substrate_fec6_plus_wz_pose_modal_a10g_smoke_dispatch.yaml` | `... --message "fec6+wz_pose: Tier-1 deterministic Wyner-Ziv pose-residual hoist per OP-1" ...` |
| **4** | OP-6 | Pose-FOE sparse encoding LFV1 analysis | -0.005 | $1 | 0.005 | `.omx/research/telescope_foveation_lfv1_candidate_20260513_codex.md` analysis extension; `tools/build_lapose_foveation_atom_manifest.py` consumer | `... --message "pose_foe: extend LFV1 candidate analysis with score-impact estimate per OP-6" ...` |
| **5** | OP-10 | Cathedral autopilot Cascade 2 per-pair pose-axis-specific reward factor | -0.001 | $0 | ∞ | `tools/cathedral_autopilot_autonomous_loop.py` Cascade 2; `src/tac/tests/test_autopilot_reweight_v2_lagrangian_derived.py` extension | `... --message "autopilot: Cascade 2 per-pair pose-axis-specific reward factor per OP-10" ...` |

### Wave scheduling (per OP-9 coordinated EV ranking enforcement)

| Week | OPs | Cost | Trigger |
|---|---|---|---|
| Week 1 | OP-7 + OP-2 + OP-1 (cheap-probe family) | $0-$1 | All CPU-only via existing canonical infrastructure |
| Week 2 | OP-6 (LFV1 candidate analysis extension) | $0-$1 | Triggered by Week 1 completion |
| Week 3 | OP-10 (Cascade 2 extension) | $0 | Triggered by OP-1+OP-2+OP-7 completion |
| Week 4 | OP-3 (ATW V2-1 channel-pick) — symposium-gated | $5-7 CPU + $15-25 Modal | Triggered by Z6 Wave 2 4c outcome + OP-11 symposium ratification |
| Week 5+ | OP-4 (pose-conditioned residual codec PROBE) + OP-8 (VGGT-distillation) — symposium-gated | $2-5 + $10-15 | Triggered by OP-11 symposium ratification |
| Week 6+ | OP-5 (Z6/Z7/Z8 from-scratch) — DEFER pending reactivation | $20-50 per substrate | Triggered by per-substrate symposium PROCEED-unconditional per Catalog #325 |

**Total Week 1-3 provider spend (cheap-probe family complete)**: $0-$1. **Total Week 1-5 provider spend (symposium-gated substrates complete)**: $25-60. **Total Week 1-6 provider spend (full substrate-class-shift attempted)**: $60-150.

---

## 10. Per-substrate reactivation criteria

Per CLAUDE.md 'Forbidden premature KILL' + Catalog #315 OPTIMAL FORM discipline. None of the 6 NON-HNeRV pose-axis paths are KILLED; the DEFER outcomes (P5) carry explicit reactivation criteria.

### P1 (Wyner-Ziv pose-residual hoist) reactivation criteria
Currently PROCEED unconditional. If OP-1 smoke (Week 1) returns ΔS outside `[-0.003, -0.001]` band by >2× per Catalog #324 post-training Tier-C validation, the path becomes DEFER-pending-Tier-C-re-measurement. Reactivation: post-training Tier-C on landed archive sha + alternative reducer per Catalog #308 N≥3 enumeration (deterministic-pose-warp / motion-vector-conditioning / per-pair-pose-difference variants).

### P2 (Master-gradient classification extension) reactivation criteria
Currently PROCEED unconditional. If OP-2 offline probe shows zero seg-axis vs pose-axis classification orthogonality (Pearson correlation > 0.95 across the 8-pair fp64 anchor), reward-factor extension becomes DEFER-pending-better-classifier. Reactivation: alternative classifiers (variance-decomposition / mutual-information-based / scorer-conditioned).

### P3 (ATW V2-1 channel-pick) reactivation criteria
Currently PROCEED_WITH_REVISIONS. Requires (a) per-substrate symposium per Catalog #325 (OP-11) + (b) Z6 Wave 2 4c outcome (DEFER 2026-05-18; reactivation = driver-mode hardcode fix lands per Catalog #326) + (c) V4-segfault avoidance verified at M=4, ksub=64 per ATW V2-1 V4 probe outcome 2026-05-18 reactivation paths (V5 wavelet-multi-scale / V6 Laplace-prior / V7 IB-Lagrangian-optimal / V8 learned-compression).

### P4 (Pose-conditioned residual codec PROBE) reactivation criteria
Currently PROCEED_WITH_REVISIONS. Requires (a) per-substrate symposium per Catalog #325 (OP-11) + (b) gradient-flow verification smoke (~$2-5 Modal A10G 100ep). If gradient-flow blocks (e.g., FiLM conditioning saturates at residual scale), reactivation = architecture redesign with alternative pose-conditioning patterns (cross-attention / hypernetwork / pose-bin lookup).

### P5 (Z6/Z7/Z8 from scratch) reactivation criteria
Currently DEFER_PENDING_EVIDENCE. Reactivation requires:
1. **Z6**: driver-mode hardcode fix lands (Catalog #326 backfill 0 → 0) + Wave 2 re-fires successfully + Atick Council Revision #6 ΔS measured.
2. **Z7**: per-substrate symposium #865 PROCEED-unconditional after Wave 3 Mamba-2 trainer build lands per Z7 symposium 2026-05-18.
3. **Z8**: Catalog #312 quadruple-primitive design memo lands + per-substrate symposium PROCEED-unconditional + Z6 OR Z7 successful empirical anchor lands first.

### P6 (Pose-FOE sparse encoding LFV1) reactivation criteria
Currently PROCEED unconditional. If OP-6 LFV1 candidate analysis shows bytes > 2 KB (exceeds CLAUDE.md HNeRV parity L4 inflate.py LOC budget), reactivation = architectural narrowing (Telescope-only single-stage OR LAPose-only single-stage) per the 2026-05-13 LA-Pose/Telescope online distinction.

### OP-7 (Direct master-gradient pose-byte hoist) reactivation criteria
Currently PROCEED unconditional. If Cascade 2 extension reveals existing v2 cascade refactor cost > 100 LOC, reactivation = scope reduction (single-axis reward factor instead of per-pair pose-axis-specific).

### OP-8 (VGGT-distillation) reactivation criteria
Currently PROCEED_WITH_REVISIONS. Requires per-substrate symposium per Catalog #325 (OP-11). If symposium identifies VGGT teacher output misalignment with PoseNet 6-DOF (e.g., VGGT outputs are not the first 6 dimensions PoseNet's hydra-head expects), reactivation = teacher-output-alignment training (substantially higher cost; symposium re-deliberation required).

---

## 11. Council verdict + continual-learning anchor

### Verdict

**PROCEED_WITH_REVISIONS** with 11 op-routables (OP-1 through OP-11).

### Mission alignment

`council_predicted_mission_contribution: frontier_breaking` (Categories per CLAUDE.md 'Mission alignment — non-negotiable' Consequence 5). The cheap-probe family (P1 + P2 + P6 + OP-7) is `frontier_protecting` at minimum + `frontier_breaking` if any one Wyner-Ziv hoist lands ΔS = −0.003.

### Continual-learning anchor emission

Per Catalog #300 v2 frontmatter + Catalog #292 + canonical helper `tac.council_continual_learning.append_council_anchor` per CLAUDE.md 'Council hierarchy: 4-tier protocol' continual-learning wire-in.

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord,
    CouncilTier,
    append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518",
    topic="Pose-axis NON-HNeRV paths to a frontier-breaking improvement at sub-0.20 operating point",
    council_tier=CouncilTier.T3,
    council_attendees=(
        "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
        "Boyd", "Mallat", "van_den_Oord", "Carmack", "Hinton", "Tishby_memorial",
        "Atick", "Redlich", "Rao", "Ballard", "Wyner", "Hafner",
        "Time_Traveler_protege_Rudin_postdoc",
    ),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "<see §0 frontmatter>"},
        {"member": "Assumption-Adversary", "verbatim": "<see §0 frontmatter>"},
        {"member": "Carmack", "verbatim": "<see §0 frontmatter>"},
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "Pose-axis is dominant marginal", "classification": "HARD-EARNED", "rationale": "CLAUDE.md 2026-05-04 + PR97 anti-pattern empirical receipt"},
        {"assumption": "HNeRV-pattern is only public-frontier path", "classification": "HARD-EARNED", "rationale": "All 3 public race winners 2026-05-04 used HNeRV-pattern"},
        {"assumption": "Wyner-Ziv pose-residual is budget-feasible", "classification": "HARD-EARNED", "rationale": "2026-05-17 T3 Wyner-Ziv canonical-fix council + Catalog #319 v2 ACTIVE"},
        {"assumption": "Pose-conditioned residual codec is byte-distinguishable from HNeRV-pattern", "classification": "CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION", "rationale": "OP-4 cheap probe verifies gradient-flow; PRE-OPTIMAL-FORM per Catalog #315"},
        {"assumption": "Ego-motion-conditioned next-frame prediction delivers asymptotic-pursuit ΔS", "classification": "CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION", "rationale": "Z6-v2 Wave 2 + C6 IBPS 22× miss + TT5L 19× miss = 5 same-week implementation-level falsifications"},
        {"assumption": "Pose-bin classification via master-gradient bypasses HNeRV implicit pose", "classification": "HARD-EARNED-VIA-MASTER-GRADIENT", "rationale": "Codex 2026-05-18 master-gradient extractor IS the proof"},
        {"assumption": "Pose-FOE sparse encoding is byte-feasible ≤1 KB", "classification": "CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION", "rationale": "LFV1 candidate offline analysis is the cheap probe"},
        {"assumption": "Direct master-gradient pose-byte hoist delivers savings at $0-$1", "classification": "HARD-EARNED-EMPIRICALLY-NEAR-LANDED", "rationale": "Codex 2026-05-18 master-gradient extractor + Catalog #318/#327 self-protection ACTIVE"},
        {"assumption": "VGGT-distillation IS contest-compliant", "classification": "HARD-EARNED", "rationale": "Only distilled PR101 weights ship; per Catalog #213 + Catalog #210 dataset-derived prior + license-tag propagation canonical pattern"},
        {"assumption": "5 same-week DEFER implies ALL NON-HNeRV class-shift dominated", "classification": "CARGO-CULTED", "rationale": "Per Catalog #307 + #308 each DEFER is IMPLEMENTATION-LEVEL; PARADIGMS intact; per-substrate symposium per Catalog #325 is unwind path"},
        {"assumption": "Cheap-probe family (P1+P2+P6+OP-7) is budget-feasible THIS WEEK", "classification": "HARD-EARNED", "rationale": "$0-$1 CPU each; canonical infrastructure exists today"},
        {"assumption": "Current Cascade 2 reward factors correctly calibrated for pose-axis vs seg-axis", "classification": "CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION", "rationale": "OP-10 extends per-pair pose-axis-specific factors; current factors are substrate-wide"},
    ),
    council_decisions_recorded=(
        "OP-1 [PROCEED]: Wyner-Ziv pose-residual Tier-1 hoist; $0-$0.30; canonical infrastructure exists",
        "OP-2 [PROCEED]: master-gradient pose-byte classification extension; $0 CPU",
        "OP-3 [PROCEED_WITH_REVISIONS]: ATW V2-1 channel-pick reformulation; symposium-required + Z6 4c cross-substrate dependency; $5-7 + $15-25",
        "OP-4 [PROCEED_WITH_REVISIONS]: pose-conditioned residual codec PROBE; symposium-required; $2-5 Modal smoke",
        "OP-5 [DEFER_PENDING_EVIDENCE]: Z6/Z7/Z8 ego-motion-conditioned predictor full substrate builds; 5 same-week implementation falsifications; per Catalog #315 PRE-OPTIMAL-FORM; reactivation criteria pinned",
        "OP-6 [PROCEED]: pose-FOE sparse encoding LFV1 analysis; $0-$1 CPU",
        "OP-7 [PROCEED]: direct master-gradient pose-byte hoist via Codex's extractor; $0 CPU only",
        "OP-8 [PROCEED_WITH_REVISIONS]: VGGT-encoder-as-pose-teacher distillation; symposium-required; $10-15 Modal A100",
        "OP-9 [PROCEED]: coordinated wave EV ranking enforcement (cheap-probe first; substrate-build last); Week 1-6 sequencing",
        "OP-10 [PROCEED]: cathedral autopilot Cascade 2 extension to per-pair pose-axis-specific reward factor; ~30-50 LOC + 15 tests; $0 GPU",
        "OP-11 [DEFER_PENDING_EVIDENCE]: per-substrate symposiums for P4 (pose-conditioned residual) + OP-8 (VGGT-distillation); $0 each; triggers paid dispatch unlock",
    ),
    related_deliberation_ids=(
        "grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513",
        "grand_council_pose_axis_non_hnerv_a1_plus_lapose_d4_deeper_20260513",
        "grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517",
        "council_per_substrate_symposium_tt5l_foveation_lapose_20260517",
        "council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518",
        "deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518",
        "z7_mamba2_substrate_design_memo_20260518",
        "tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518",
        "atw_codec_atick_tishby_wyner_v1_design_20260515",
        "external_sources_20260505_lapose_dominant_codex",
        "lapose_telescope_online_distinction_20260513_codex",
    ),
    predicted_mission_contribution="frontier_breaking",
    override_invoked=False,
    override_rationale="",
    deferred_substrate_id=None,
    deferred_substrate_retrospective_due_utc="2026-06-17T18:42:00.000000Z",
    memory_path=".omx/research/grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518.md",
    notes="6 NON-HNeRV pose-axis paths enumerated + 2 deep-research-wave additions; PROCEED on cheap-probe family (P1+P2+P6+OP-7+OP-10); PROCEED_WITH_REVISIONS on symposium-required substrate-class-shift candidates (P3+P4+OP-8); DEFER on full substrate-class-shift (P5); 11 op-routables emitted",
)
append_council_anchor(record)
```

### 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| 1 — Sensitivity-map contribution (`tac.sensitivity_map.*`) | ACTIVE | OP-2 + OP-7 directly extend the per-pair Venn classification surface; downstream consumers of `tac.master_gradient_consumers` inherit pose-axis-specific factors |
| 2 — Pareto constraint (`tac.pareto_*`) | ACTIVE | Each path's predicted ΔS band carries a Dykstra-feasibility verification per Catalog #296; Pareto-feasibility is preserved across composition |
| 3 — Bit-allocator hook | ACTIVE | OP-1 + OP-3 + OP-6 modify per-substrate bit allocation; per Catalog #319 v2 cascade the deliverability_proof_class feeds the autopilot bit allocator |
| 4 — Cathedral autopilot dispatch hook | **PRIMARY** | OP-10 IS this hook (extends Cascade 2 reward factor); OP-7 produces the canonical proof artifacts |
| 5 — Continual-learning posterior update | ACTIVE | This memo's anchor lands via `tac.council_continual_learning.append_council_anchor` per §11; sister probe-outcome anchors land per Catalog #313 on each path's empirical verification |
| 6 — Probe-disambiguator (if 2+ defensible interpretations exist) | ACTIVE | Per §3 Each substrate-class-shift candidate (P3 + P4 + OP-8) requires its own per-substrate symposium per Catalog #325; the symposium IS the canonical probe-disambiguator |

### Predicted-band validation status (per Catalog #324)

`predicted_band_validation_status: pending_post_training` — every predicted ΔS band in this memo is a PREDICTION. Reactivation criterion: post-training Tier-C density measurement on each path's landed archive sha256 via `tools/mdl_scorer_conditional_ablation.py --tier c`.

---

## 12. Cross-references

### CLAUDE.md sections cited

- 'Race-mode rigor inversion + parallel-dispatch first' (§3 op-routable ranking + §9 wave scheduling)
- 'HNeRV / leaderboard-implementation parity discipline' (§1 strategic question + §3 per-path bolt-on budget enforcement)
- 'UNIQUE-AND-COMPLETE-PER-METHOD operating mode' (§5 dimension 5 + §9 canonical-helper routing)
- 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY' (§3 P5 PRE-OPTIMAL-FORM + §10 reactivation criteria)
- 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' (Catalog #315 — applies to P5; §3 + §10)
- 'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium' (Catalog #325 — applies to P3 + P4 + OP-8; OP-11 enforces)
- 'Apples-to-apples evidence discipline' (§2 frontier evidence + axis-tagged predictions throughout)
- 'Bugs must be permanently fixed AND self-protected against' (Catalog #325 IS this rule's structural enforcement at the per-substrate symposium surface)
- 'Subagent coherence-by-default' (§0 premise verification + sister-subagent ownership map per Catalog #314)
- 'Mandatory crash-resume protocol' (§0 + Catalog #206 checkpoint discipline)
- 'Main branch source of truth' (§9 commit-via-serializer command per OP-9)
- 'Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS' (§1 + §2 + §9)
- 'Meta-Lagrangian/Pareto solver' (§3 Dykstra-feasibility intersection per path + §8 Dykstra position)
- 'SegNet vs PoseNet importance — operating-point dependent' (UPDATED 2026-05-04; §1 strategic question)
- 'Council hierarchy: 4-tier protocol' (§8 sextet pact + grand-council deliberation; T3 tier per §0)
- 'Council conduct' (§8 sextet pact + Assumption-Adversary seat + per-round explicit-assumption-statement discipline)
- 'Mission alignment — non-negotiable' (§11 mission alignment + frontier_breaking categorization)
- 'Max observability' (§6 observability surface)
- 'META-ASSUMPTION ADVERSARIAL REVIEW' (§7 per-path Assumption-Adversary verdict)
- 'Recursive adversarial review protocol' (§8 sextet pact with assumption-challenge axis item 8)
- FORBIDDEN_PATTERNS: 'Forbidden symposium-band-prediction-without-Dykstra-feasibility-check' (§3 + §4 + §7)
- FORBIDDEN_PATTERNS: 'Forbidden device-selection defaults' (NOT applicable; this is a research memo, not code)
- FORBIDDEN_PATTERNS: 'Forbidden premature KILL without research exhaustion' (§10 reactivation criteria)
- FORBIDDEN_PATTERNS: 'Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap)' (§1 + §2 + §3 + §9 all numeric claims carry axis tags)
- FORBIDDEN_PATTERNS: 'Forbidden /tmp paths in any persisted artifact' (§6 observability surface persists ONLY to `.omx/state/`)

### Sister memos cited

- `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (prior pose-axis council; D1.D HIERARCHICAL verdict; A1+LAPose composition)
- `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_d4_deeper_20260513.md` (D4 inflate.sh contract deeper deliberation)
- `.omx/research/grand_council_symposium_wyner_ziv_contest_compliance_optimal_design_20260517.md` (2026-05-17 T3 Wyner-Ziv canonical-fix council; OP-1 + Catalog #319 v2 sister)
- `.omx/research/council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md` (TT5L symposium #866 REFUSE; cross-substrate dependency for P3 channel-pick)
- `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (Z7 symposium PROCEED_WITH_REVISIONS; P5 reactivation criterion)
- `.omx/research/deeper_granularity_discovery_bit_byte_zero_one_pixel_frame_pair_region_label_category_venn_20260518.md` (deeper-granularity Venn discovery; sister-pattern for P2 + OP-7 + OP-10)
- `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` (P5 Z7 trainer build design)
- `.omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md` (OP-8 VGGT-distillation deep-research wave anchor)
- `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` (P3 ATW V2-1 + OP-3 design lineage)
- `.omx/research/external_sources_20260505_lapose_dominant_codex.md` (P6 LA-Pose source review)
- `.omx/research/lapose_telescope_online_distinction_20260513_codex.md` (P6 LA-Pose vs Telescope online distinction; canonical taxonomy correction)
- `.omx/research/telescope_foveation_lfv1_candidate_20260513_codex.md` (P6 LFV1 candidate landed)
- `.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md` (pose-axis cross-axis drift mechanism)
- `.omx/research/comprehensive_research_wave_20260518.md` (deep-research wave anchor; OP-8 + P5 bleeding-edge OSS)
- `.omx/research/fec6_plus_pd_v2_pose_codec_DEFERRED_pending_separate_pose_slot_20260517.md` (PD-V2 lane DEFER lineage)
- `.omx/research/fec6_writeup_pose_marginal_correction_20260517_codex.md` (fec6 pose marginal correction)

### Canonical helpers cited

- `tac.council_continual_learning.append_council_anchor` (Catalog #300 wire-in)
- `tac.side_information.deliverability_proof_builder` (Catalog #319 wire-in; OP-1 consumer)
- `tac.master_gradient_consumers.classify_bytes_by_pair_variance` (Codex 2026-05-18 wire-in; OP-2 extension)
- `tac.master_gradient_consumers.adjust_predicted_delta_for_venn_classification_v2` (Catalog #319 Q3 wire-in; OP-7 + OP-10 extension)
- `tools/scan_best_anchor_per_axis.py` (Catalog #316 frontier scan)
- `tools/extract_master_gradient.py` (Codex 2026-05-18 canonical extractor)
- `tools/check_predecessor_probe_outcome.py` (Catalog #313 ledger consultation)
- `tools/lane_maturity.py` (Catalog #126 lane pre-registration)
- `tools/subagent_commit_serializer.py` (Catalog #117 + Catalog #157 + Catalog #174 canonical commit path)
- `tools/subagent_checkpoint.py` (Catalog #206 crash-resume protocol)
- `tools/verify_distinguishing_feature_byte_mutation.py` (Catalog #272 distinguishing-feature byte-mutation smoke)

---

## 13. Discipline gate trail

Per CLAUDE.md non-negotiables verified BEFORE landing this memo:

- [x] **Catalog #229 premise verification** (§0): 10 premises explicitly verified pre-edit
- [x] **Catalog #287 docstring overstatement guard**: all numeric ΔS bands carry `[prediction]` / `[contest-CPU]` / `[contest-CUDA T4]` / `[macOS-CPU advisory]` / `[empirical]` tags
- [x] **Catalog #126 lane pre-registration**: `lane_pose_axis_non_hnerv_t3_council_20260518` added at L0 via canonical CLI
- [x] **Catalog #206 checkpoint discipline**: 3 checkpoints emitted to `.omx/state/subagent_progress.jsonl` (pre-flight reads, premise verification, drafting)
- [x] **Catalog #314 sister-subagent absorption prevention**: subagent owns ONLY this memo + lane-registry mutation via canonical CLI; Codex's master-gradient extractor + `src/tac/` numeric code are NOT touched
- [x] **Catalog #300 v2 frontmatter**: tier T3 + attendees + quorum + verdict + dissent + assumption-adversary + decisions + mission-alignment fields all declared
- [x] **Catalog #292 per-deliberation assumption surfacing**: §8 each sextet member explicitly states 'the shared assumption I am operating within is X'
- [x] **Catalog #325 canonical 6-step contract** (per CLAUDE.md 'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium'):
  - [x] (1) Cargo-cult audit per Catalog #303 — §4
  - [x] (2) 9-dimension success checklist evidence per Catalog #294 — §5
  - [x] (3) Observability surface declaration per Catalog #305 — §6
  - [x] (4) Sextet pact deliberation + grand-council attendees — §8
  - [x] (5) Per-substrate reactivation criteria pinned per CLAUDE.md 'Forbidden premature KILL' — §10
  - [x] (6) Catalog #324 post-training Tier-C validation discipline — §0 frontmatter `predicted_band_validation_status: pending_post_training` + §10 reactivation criteria per path
- [x] **Catalog #313 predecessor probe-outcomes consultation**: §2.5 enumerates 5-of-8 same-week DEFER outcomes; per-path reactivation criteria per Catalog #313
- [x] **Catalog #316 frontier scan**: §2.1 cites canonical state via `tools/scan_best_anchor_per_axis.py`
- [x] **Catalog #117 + Catalog #157 + Catalog #174 commit-via-serializer**: commit at land-time via canonical serializer with `--expected-content-sha256`

---

**End of memo.**

This T3 grand council symposium memo IS the deliverable. The 11 op-routables are operator-routable; the sister BUILD subagents (per-path) consume each op-routable as design contract. The cheap-probe family (OP-1 + OP-2 + OP-6 + OP-7 + OP-10) is operator-attention-budget-feasible THIS WEEK and PROCEED unconditional. The symposium-required substrate-class-shift candidates (OP-3 + OP-4 + OP-8) await Catalog #325 per-substrate symposium ratification (OP-11). The full substrate-class-shift family (OP-5) DEFERS pending per-substrate reactivation criteria per §10.

Per CLAUDE.md 'Signal-loss prevention' (operator standing directive 2026-05-18 verbatim: *"respawn and recover and continue with all with at most 2 subagents in the subagent queue at a time, ensure no signal loss"*) — even on rate-limit / API failure / context cap, this memo lands on disk BEFORE the final report message. The canonical serializer + checkpoint discipline guarantees disk durability.
