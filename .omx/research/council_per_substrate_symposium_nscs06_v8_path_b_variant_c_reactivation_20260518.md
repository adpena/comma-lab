---
review_kind: per_substrate_symposium
review_id: per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518
review_date: "2026-05-18"
lane_id: lane_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518
substrate_id: nscs06_v8_path_b_variant_c
substrate_alias: nscs06_v9_variant_c
parent_substrate_id: nscs06_carmack_hotz_strip_everything
deferred_substrate_id: nscs06_v8_path_b_wavelet_residual
horizon_class: plateau_adjacent
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Carmack
  - Hotz
  - Mallat
  - Daubechies
  - Wyner_memorial
  - Quantizr
  - Time_Traveler_protege
  - Schmidhuber
  - Boyd
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
predicted_band_validation_status: pending_post_training
predicted_band:
  variant_c1_v7_anchor_plus_db4_depth1_residual_only: [45, 55]   # Variant C-1: SURGICAL ADDITION of DB4 depth-1 on RESIDUAL only; v7 chroma anchor preserved
  variant_c2_v7_anchor_plus_per_subband_class_cdf_residual_only: [50, 58]   # Variant C-2: SURGICAL ADDITION of per-subband per-class CDF on RESIDUAL only
  variant_c3_v7_anchor_plus_uniward_conditioned_db4_residual: [40, 50]   # Variant C-3: UNIWARD-conditioned wavelet residual; Yousfi+Fridrich recommendation
  variant_c4_methodology_extension_K_coverage_validation_only: null   # Variant C-4: K-coverage measurement methodology validation; design memo only
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
dispatch_enabled: false
operator_directive: "TOP 3 cargo-cult-failed reactivation — NSCS06 v8 Path B 600x miss; Variant C = preserve v7's chroma-preserving (the wave that produced +44%) + apply v8 cargo-cult-unwinds SELECTIVELY (NOT all-at-once)"
related_deliberation_ids:
  - council_per_substrate_symposium_nscs06_v8_path_b_20260517
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
  - nscs06_path_a_chroma_optical_flow_redesign_20260516
  - nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516
  - nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516
  - feedback_deep_research_wave_landed_20260518
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201)"
  contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519)"
predecessor_probe_outcomes:
  - probe_id: nscs06_v6_dispatch
    verdict: EMPIRICAL_FALSIFICATION
    final_score: 105.15
    predicted_band: [0.10, 0.20]
    miss_factor: 553
    classification: implementation-level-falsification-of-7-cargo-culted-assumptions
  - probe_id: nscs06_v7_path_a_dispatch
    verdict: SUCCESS_44_PERCENT_IMPROVEMENT
    final_score: 58.89
    predicted_band: [40, 65]
    miss_factor: 0
    classification: paradigm-intact-iterative-rescue-via-cargo-cult-unwind
  - probe_id: nscs06_v8_path_b_dispatch
    verdict: EMPIRICAL_FALSIFICATION
    final_score: 104.98
    predicted_band: [15, 25]
    miss_factor: 600
    classification: implementation-level-cargo-cult-composition-failure
council_dissent:
  - member: Contrarian
    verbatim: "I rise to challenge the FRAMING that Variant C = 'preserve v7's chroma-preserving + apply v8 cargo-cult-unwinds selectively' is a NEW reactivation path distinct from sister symposium #864's Path B-LITE / Path B-CHROMA-AXIS already deliberated 2026-05-17. The sister symposium's Path 1 (Path B-LITE — DB4 depth-1 chroma+grayscale-only with v7 anchor preserved) IS the canonical Variant C-1; Path 2 (Path B-CHROMA-AXIS — augment v7 anchor with full-resolution wavelet Cb+Cr) IS the canonical Variant C-2. The operator's directive 2026-05-18 essentially asks 'execute sister symposium #864's Path 1+Path 2 reactivation criteria'. My CRITICAL CONCERN: the operator family-DEFER 2026-05-16 stands (NSCS06 family DEFERRED-pending-breakthrough); the sister symposium #864 REFUSE 13-of-13 unanimous T3 ratified this DEFER; this v2 Variant C symposium MUST NOT silently reverse the family-DEFER without explicit operator-frontier-override per CLAUDE.md Mission Alignment Consequence 1. RECOMMENDED: this symposium memo PRESERVES the family-DEFER + documents Variant C-1/C-2/C-3 as CANONICAL reactivation paths PENDING operator family-DEFER reversal + budget reauthorization. The operator's 2026-05-18 directive triggered this symposium but does NOT itself constitute family-DEFER reversal. My VETO is on any verdict that authorizes dispatch of Variant C-* without explicit operator family-DEFER reversal + paired-env operator-frontier-override per Catalog #199."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 + CLAUDE.md per-round explicit-assumption-statement discipline. The SHARED ASSUMPTION operating across the parent prompt and this v2 symposium: *'The v6→v7→v8 dual-direction empirical anchor (v6=105.15 / v7=58.89 / v8=104.98) reveals that cargo-cult-unwind methodology does NOT compose monotonically across architectural changes, but Variant C (surgical-addition-only) recovers monotonicity.'* I classify this CARGO-CULTED-PENDING-EMPIRICAL. HARD-EARNED basis: (a) the v6→v7 empirical anchor (+44% improvement via 4-of-7 cargo-cult-unwinds) IS HARD-EARNED-EMPIRICALLY-VERIFIED per sister symposium #864 6-of-6 sextet ratification; (b) the v7→v8 empirical regression (-78% via architectural-replacement + same 4-of-7 unwinds + 2 additional unwinds) IS HARD-EARNED-EMPIRICALLY-VERIFIED via 2 bit-identical dispatches; (c) the META cargo-cult #8 (architectural-transition-preserves-previously-unwound-cargo-cults) IS DISCOVERED but not yet generalized — applies SPECIFICALLY to v8 wavelet-codec-class-shift, NOT to ALL architectural changes. CARGO-CULTED basis: (a) the prediction that Variant C SURGICAL-ADDITION recovers monotonicity has NO empirical anchor; sister symposium #864's Path B-LITE / Path B-CHROMA-AXIS are predicted bands [50, 55] / [45, 52] — INSIDE v7's 58.89 baseline but NOT below medal-band threshold; (b) the K-coverage measurement methodology extension (sister symposium #864 op-routable #3) has NEVER been empirically tested; (c) the assumption that PRESERVING v7's chroma-anchor architecture + SELECTIVELY applying v8 unwinds will produce monotonic improvement DEPENDS on which unwinds are 'selectively' applied; the 7-cargo-cult-space has 2^7 = 128 selective combinations, only 4 of which have ANY empirical evidence (v6 baseline / v7 +44% / v8 -78% / sister #864 reformulation prediction). My assumption-violation hypothesis: *'IF Variant C-1 SURGICAL-ADDITION (DB4 depth-1 chroma+grayscale-only with v7 anchor preserved) lands final_score ≥ v7's 58.89 (no improvement), the META cargo-cult #8 is paradigm-intact (architectural-substitution-degree IS the dominant factor; surgical-addition with v7 anchor is structurally bounded by v7's plateau), and the operator should pivot to non-NSCS06 class-shift architectures or accept v7's 58.89 as the NSCS06-paradigm-ceiling.'* Required action per Catalog #308: enumerate ≥3 alternative paths INCLUDING the methodology-only validation path. VETO on PROCEED-unconditional dispatch authorization pending operator family-DEFER reversal."
  - member: Carmack
    verbatim: "I am summoned per grand council as canonical author of strip-everything paradigm + sister symposium #864 reactivation-path-1 author. My voting position from sister symposium #864 stands: REFUSE Path B in current form; Path B-LITE (preserve v7 anchor + DB4 depth-1 on residual) IS the canonical engineering response when a SECOND iteration regresses below the FIRST iteration's empirical anchor. The operator's 2026-05-18 directive ('Variant C path: keep v7 chroma-preserving + apply v8 cargo-cult-unwinds selectively') IS structurally consistent with my recommendation. Engineering discipline per Doom/Quake methodology: when feature A works and feature B regresses A while attempting C, the canonical fix is feature A + surgical addition of C (NOT feature B). Variant C-1 IS this canonical fix: v7's chroma anchor IS feature A; DB4 depth-1 on residual IS surgical addition of C; v8's full wavelet codec was feature B (architectural replacement that regressed A). The empirical prediction for Variant C-1 per sister symposium #864 my own analysis: [50, 55] band — improvement from v7's 58.89 of 3.89-8.89 score points via the DB4 depth-1 rate-axis savings; NO regression of v7's anchor architecture. NEW (this v2 symposium): the operator's 2026-05-18 directive enables a JOINT design discipline: cargo-cult-unwind methodology composition K-coverage measurement (sister #864 op-routable #3) APPLIED to Variant C-1 design memo, NOT just to v8 retrospective. The K-coverage check BEFORE dispatch IS the canonical pre-dispatch gate that would have prevented the v8 regression. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-1 design memo MUST include K-coverage measurement of which cargo-cult-unwinds are SELECTIVELY applied + which are PRESERVED + which are NEW; the K-coverage matrix IS the structural protection against META cargo-cult #8."
  - member: Hotz
    verbatim: "Concur with Carmack. The Doom-engineering discipline says: when v7 produced +44% with 4-of-7 unwinds + chroma anchor architecture, the next iteration MUST add 1-of-3-remaining-unwinds at a time (cargo-cult #3 spatial-independent CDF OR cargo-cult #5 NO-neural-at-medal-band OR cargo-cult #7 PR#56 grayscale-LUT generalizes from masks to frames) — NEVER unwind 2 + replace architecture in same iteration. v8's mistake was unwinding #3 + REPLACING the chroma anchor architecture simultaneously. Variant C-1 (DB4 depth-1 on residual ONLY) IS cargo-cult #3 unwind APPLIED SURGICALLY — the simplest thing that could possibly work. Predicted band [50, 55] per sister #864. My LOC budget recommendation: Variant C-1 implementation ≤200 LOC delta from v7 (v7 was ~600 LOC; Variant C-1 adds ~150-200 LOC for DB4 depth-1 codec + integration with v7 chroma anchor + archive grammar v8-LITE schema). PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-1 LOC budget MUST be ≤200 LOC delta from v7; if implementation exceeds budget, the surgical-addition discipline IS being violated (architecture-replacement-by-stealth). Operator family-DEFER reversal STILL required per Contrarian + Assumption-Adversary."
  - member: Shannon
    verbatim: "Information-theory grounding lens applied to Variant C: v7's per-class chroma anchor architecture preserves seg-distinguishing chroma cues at the SegNet's perceptual scale (canonical Yousfi steganalysis attack surface). v7's empirical seg=25.27 demonstrates the chroma-anchor architecture preserves I(X;Y_seg) above the SegNet's segmentation threshold. v8's per-subband per-class CDF substitution REGRESSED seg to 75.24 because the wavelet-coded chroma path lost per-class differentiation (per-subband CDFs converged on dominant-class statistics per Yousfi+Fridrich verbatim in sister symposium #864). Variant C-1 applies DB4 depth-1 ONLY to the residual stream — the chroma anchor's main signal pathway IS PRESERVED; DB4 depth-1 captures rate-axis savings (per Mallat 1989 5-10x narrowed to ~2-3x at depth-1) ON THE RESIDUAL ALONE. Predicted I(X;Y_seg) preservation: ~95-98% of v7's level (chroma anchor preserved; residual contributes minor seg signal). Predicted I(X;Y_pose) preservation: ~95-98% of v7's level (6-DOF affine preserved). Predicted rate-axis savings: 2.67 v7 → 1.50-1.75 v8-Variant-C-1 = 0.92-1.17 score reduction. Predicted final_score: 58.89 - (0.92-1.17) × (seg-rate-marginal-ratio at v7 operating point ≈ 1.0) ≈ 57.72-57.97. The Path B-LITE predicted [50, 55] band per sister symposium #864 is OPTIMISTIC; my conservative prediction shifts the band to [55, 58]. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-1 predicted band SHOULD be [55, 58] (conservative) OR [50, 55] (sister symposium #864 prediction); empirical anchor required."
  - member: Dykstra
    verbatim: "Dykstra-feasibility analysis applied to Variant C: the feasible region per the contest scoring formula is the intersection of (a) rate ≤ R; (b) seg ≤ S; (c) pose ≤ P; (d) archive ≤ B. v7's operating point (rate=2.67, seg=25.27, pose=30.94, archive=4014KB) WAS feasible at the v7 polytope. v8's operating point (rate=0.010, seg=75.24, pose=29.74, archive=15.5KB) WAS infeasible at v7's polytope (seg blowup). Variant C-1's predicted operating point (rate=1.50-1.75, seg=25-27, pose=30-31, archive=2000-2500KB) IS feasible at v7's polytope (seg preserved; rate reduced; archive reduced). Per Dykstra alternating projections methodology: Variant C-1 IS a feasible Pareto-frontier-extension of v7. The improvement is MODEST (3-5 score points) but DIRECTIONALLY CORRECT. The K-coverage methodology extension (sister #864 op-routable #3) IS the canonical pre-dispatch gate that empirically validates whether Variant C-1's predicted operating point is REACHABLE — measuring the dispersion of the design space's cargo-cult-unwind compositions around the predicted point. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-1 design memo MUST include Dykstra-feasibility analysis showing the predicted operating point is INSIDE v7's polytope + K-coverage measurement showing the parameter space's dispersion around the point."
  - member: Boyd
    verbatim: "Convex-optimization lens applied to Variant C: the cargo-cult-unwind composition space is a 7-cargo-cult lattice (each unwind either applied or not); v7 occupies the {1,2,4,6}-unwind sub-lattice point; v8 occupies the {1,2,3,4,6} + architecture-replacement sub-lattice point. Variant C-1 targets the {1,2,3,4,6}-unwind sub-lattice point WITHOUT architecture replacement (preserves v7's chroma anchor + 6-DOF affine + per-class CDF infrastructure; adds cargo-cult #3 unwind via DB4 depth-1 on residual). The convex hull of empirically-anchored points is currently {v6, v7, v8}: v6 at score 105.15, v7 at 58.89, v8 at 104.98. Variant C-1 IS structurally INSIDE the convex hull (predicted [50, 55] OR [55, 58]); ADMM/proximal-gradient methods predict the unwind-composition mathematical dual converges to the Pareto-frontier point that minimizes (score - chroma-anchor-preservation-penalty). The K-coverage methodology extension IS the empirical measurement that allows the convex-optimization framework to operate at the cargo-cult-unwind lattice surface. PROCEED_WITH_REVISIONS verdict + binding revision: the convex-optimization framework MUST be applied to the cargo-cult-unwind composition design space; the framework predicts Variant C-1 is the canonical Pareto-frontier extension from v7."
  - member: Mallat
    verbatim: "Per the Mallat wavelet hierarchical-planning lens (Catalog #277). DB4 depth-1 on residual ONLY is the canonical Mallat-multi-scale discipline applied surgically — the coarse-scale information (v7's chroma anchor) GATES the fine-scale residual (DB4 depth-1 wavelet decomposition). The depth-1 vs depth-2 choice: depth-1 has 4 subbands (LL, LH, HL, HH) at half-resolution; depth-2 has 7 subbands (LL_2, LH_2, HL_2, HH_2 at quarter-resolution + LH_1, HL_1, HH_1 at half-resolution). v8 used depth-2 (full multi-scale); Variant C-1 uses depth-1 (single-scale residual). The depth-1 choice IS the SURGICAL ADDITION discipline — adds one layer of decorrelation without architectural complexity. Predicted rate savings at depth-1 per Mallat 1989: 2-3x decorrelation (vs v8's depth-2 at 5-10x reach but compositionally REGRESSED). Variant C-1 captures 60-70% of v8's rate savings at <10% of v8's architectural complexity. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-1 DB4 depth-1 implementation MUST follow Mallat 1989 canonical-correct quadrature mirror filter convention (verified canonical-correct in sister symposium #864)."
  - member: Daubechies
    verbatim: "Compressive-coverage estimator lens (Catalog #276 sister + sister symposium #864 verbatim). The K-coverage measurement methodology extension proposed in sister symposium #864 op-routable #3 IS the canonical Daubechies-DeVore-Fornasier-Gunturk 2010 compressive-sensing methodology applied to the cargo-cult-unwind design space. The principle: from K=8 representative cargo-cult-unwind compositions (e.g., v6 + v7 + v8 + 5 sister-substrate compositions), reconstruct the full cargo-cult-unwind landscape with bounded uncertainty O(sqrt(N/K)) where N=128 (2^7 unwind subsets). For Variant C-1 specifically: the K=4 empirically-anchored compositions ({v6 baseline / v7 +44% / v8 -78% / Variant C-1 predicted}) need additional K=4 sister substrate compositions to satisfy the K-coverage bound. Sister substrate compositions identified: Path B-CHROMA-AXIS (chroma-axis-augmentation), Variant C-2 (per-subband per-class CDF on residual only), Variant C-3 (UNIWARD-conditioned wavelet residual), Variant C-4 (K-coverage methodology validation only — design memo with no empirical anchor). PROCEED_WITH_REVISIONS verdict + binding revision: this v2 symposium MUST enumerate the FULL K=8 sister substrate composition matrix as Variant C-1 + C-2 + C-3 + C-4 reactivation paths."
  - member: Wyner_memorial
    verbatim: "Memorial seat conveying Wyner-Ziv 1976 + sister symposium #864 verbatim. The Wyner-Ziv per-pixel temporal residual coding cargo-cult-empirical-falsification (sister symposium #864 cargo-cult #9) applies to v8's Wyner-Ziv-replaces-6-DOF-affine architecture, NOT to Wyner-Ziv per-se. Variant C-3 (UNIWARD-conditioned wavelet residual) re-applies Wyner-Ziv at a DIFFERENT structural surface: the wavelet residual stream (NOT the pose-warp stream). The per-pixel temporal redundancy at the wavelet residual surface IS DIFFERENT from the per-pixel temporal redundancy at the raw pixel stream — wavelet residuals capture the post-spatial-decorrelation temporal differences, which DO have higher per-pixel redundancy than raw pixel differences in dashcam ego-motion. Predicted Wyner-Ziv gain at the wavelet residual surface: 20-30% rate reduction (vs v8's 4% at the raw pixel surface). Variant C-3 IS the canonical structural-fit redirection of Wyner-Ziv to substrates with HIGHER per-pixel temporal redundancy. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-3 implementation MUST apply Wyner-Ziv to wavelet residual stream (NOT raw pixel stream)."
  - member: Quantizr
    verbatim: "Adversarial reverse-engineering lens applied to Variant C: my v8 symposium verdict 'NSCS06 family at ANY of v1-v8 displaces the AV1 monochrome mask + FP4 weight encoding primitives I empirically validated at 0.33' stands. Variant C-1 STILL displaces these primitives. The strip-everything paradigm IS structurally incompatible with my leaderboard winning recipe at the architectural level. HOWEVER: Variant C-1's predicted [50, 55] band is empirical-rigor-MORE-VALUABLE than v6's 105.15 baseline as a CARGO-CULT-COMPOSITION-METHODOLOGY anchor. Variant C-1's K-coverage measurement IS the methodology-paradigm-extension that future substrates (NOT NSCS06 family) can inherit. RECOMMENDED: dispatch Variant C-1 ONLY IF the operator's budget reauthorization explicitly funds METHODOLOGY VALIDATION (NOT score-lowering). If the budget is for score-lowering, the operator should pivot to non-NSCS06 substrate classes (HiNeRV / sane_hnerv / DP1 stacking / ATW V2-1 V4 probe / C6 IBPS RSSM categorical). PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-1 dispatch authorization MUST be conditioned on METHODOLOGY-VALIDATION budget (NOT score-lowering budget); the K-coverage measurement extension IS the canonical paradigm-fix that justifies the dispatch."
  - member: Time_Traveler_protege
    verbatim: "K-coverage methodology extension lens (sister symposium #864 op-routable #3): the v6→v7→v8 dual-direction empirical anchor IS K=3 sample points in the cargo-cult-unwind composition space. The compressive-coverage methodology requires K=8 sample points for bounded uncertainty O(sqrt(N/K)) at N=128 unwind subsets. Variant C-1/C-2/C-3 add 3 sample points to K=6; Variant C-4 (design-memo-only methodology validation) adds the 7th sample point as a STRUCTURAL anchor (NOT empirical). To reach K=8, ONE sister substrate composition must be added — recommended: sister Z3 v2 architecture-transition empirical anchor (already in lane registry at L1 with empirical 0.198 anchor) AS the K=8 sample point. With K=8 in place, the cargo-cult-unwind composition space IS reconstructable with bounded uncertainty + the methodology-paradigm-extension IS empirically validated. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-4 design memo MUST explicitly cite the K=8 sister substrate composition matrix INCLUDING Z3 v2 architecture-transition empirical anchor."
  - member: Schmidhuber
    verbatim: "Compression-as-intelligence lens: the cargo-cult-unwind methodology IS structurally analogous to the curriculum-learning + meta-learning frameworks (e.g., MAML, learn-to-learn). The K-coverage measurement methodology extension IS the meta-learning sample efficiency analysis applied to substrate-design space. PROCEED_WITH_REVISIONS verdict + binding revision: Variant C-4 methodology validation memo SHOULD cite the meta-learning + sample-efficiency literature (Finn-Abbeel-Levine 2017 MAML; Snell-Swersky-Zemel 2017 prototypical networks; Vinyals-Blundell 2016 matching networks) as canonical-prior frameworks for the K-coverage methodology extension."
council_assumption_adversary_verdict:
  - assumption: "NSCS06 v8 Path B Variant C is a NEW reactivation path distinct from sister symposium #864's Path B-LITE / Path B-CHROMA-AXIS"
    classification: CARGO-CULTED-PARTIAL
    rationale: "Per Contrarian dissent: Variant C-1 = sister symposium #864's Path B-LITE; Variant C-2 ≈ sister symposium #864's Path B-CHROMA-AXIS. The operator's 2026-05-18 directive essentially asks 'execute sister #864's Path 1+2 reactivation paths PLUS K-coverage methodology validation'. This v2 symposium MUST honor the sister symposium's prior unanimous REFUSE verdict; the operator's directive does NOT itself reverse the family-DEFER. PARTIAL: Variant C-3 (UNIWARD-conditioned wavelet residual) IS a NEW path (sister #864 did not enumerate); Variant C-4 (K-coverage methodology validation memo) IS a NEW path."
  - assumption: "v8 RE-INTRODUCED previously-unwound cargo-cults from v7"
    classification: HARD-EARNED-EMPIRICAL
    rationale: "Per sister symposium #864 §1 cargo-cult #1 + #2 v7→v8 REGRESSION analysis: cargo-cult #1 (closed-form scorer-argmax bit allocator) REGRESSED at v8 because per-subband CDFs cannot SUBSTITUTE for per-class chroma anchor; cargo-cult #2 (Y=R=G=B replication) REGRESSED at v8 at a different abstraction layer (per-subband-AC-replication). Empirical receipts: v7 seg=25.27 vs v8 seg=75.24 (3x worse); v7 pose=30.94 vs v8 pose=29.74 (4% better — minimal). The v7→v8 architectural-replacement REGRESSED 2 of v7's 4 unwinds while attempting to add 2 new ones."
  - assumption: "Surgical-addition discipline (Variant C-1) recovers monotonicity of cargo-cult-unwind methodology"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Per Assumption-Adversary verbatim + sister symposium #864 unanimous REFUSE: the surgical-addition discipline IS structurally consistent with Carmack + Hotz engineering methodology (preserve what works + surgical addition of one feature at a time); it has NEVER been empirically tested for NSCS06 family. The predicted band [50, 55] / [55, 58] per Carmack / Shannon is INSIDE v7's 58.89 baseline + ABOVE medal-band threshold. The empirical test IS the cheapest cargo-cult-unwind methodology validation experiment but does NOT itself reach medal-band frontier."
  - assumption: "PARADIGM #5 (NO-neural-at-medal-band) is still HARD-EARNED per sister symposium #864 cargo-cult #5 classification"
    classification: HARD-EARNED-EMPIRICAL
    rationale: "Per sister symposium #864 cargo-cult #5 update: 'NO-neural-at-medal-band is achievable' was CARGO-CULTED at v6 + WAIVED at v7/v8 + 'PARADIGM-LEVEL FALSIFICATION CONFIRMED' per v8 empirical anchor (547x worse than 0.192 frontier). The operator's question 'Is the NO-neural-at-medal-band claim still HARD-EARNED per Catalog #303 OR is it cargo-cult assumption #5 from the META framework?' has empirical answer: paradigm-level falsified per v6 + v7 + v8 baseline. NO ANALYTICAL-ONLY substrate has cleared 0.5 contest-CPU; the closest pure-analytical substrate is v7 at 58.89 (307x worse than frontier). Per Catalog #307 paradigm-vs-implementation: the NO-neural-at-medal-band PARADIGM IS empirically falsified at the medal-band scope but PARADIGM-INTACT at the >0.5 scope (where v7's 58.89 may represent ARCHITECTURAL-CEILING for the paradigm)."
  - assumption: "Cross-substrate cargo-cult-unwind monotonicity has a structural limit revealed by v6→v7→v8 dual-direction empirical anchor"
    classification: HARD-EARNED-EMPIRICAL
    rationale: "Per sister symposium #864 cargo-cult #8 META discovery + sister symposium #864 cargo-cult #10 META discovery: (a) cargo-cult-unwind methodology does NOT compose monotonically across architectural changes (v6→v7 +44% / v7→v8 -78% dual-direction); (b) substrate-class-shift IMPLEMENTATION (wholesale architectural redesign) is NOT canonical research-path-forward when previously-unwound cargo-cults are load-bearing for non-targeted axes. The structural limit: cargo-cult-unwind monotonicity HOLDS for SURGICAL ADDITION + BREAKS for ARCHITECTURAL REPLACEMENT. The K-coverage methodology extension (sister #864 op-routable #3 + Daubechies + Time-Traveler-protege verbatim) IS the canonical pre-dispatch gate that prevents cargo-cult-composition-failure."
  - assumption: "Alternative path: rescope to per-class chroma anchor V2 (the v7 mechanism) without further unwinds + declare 'the medal-band-class-shift may require some neural component after all'"
    classification: HARD-EARNED-EMPIRICAL
    rationale: "Per Quantizr + sister symposium #864 Path 3 Path C (hybrid analytical+neural residual decoder per v6 symposium Yousfi+Selfcomp+Ballé stacked prediction ΔS band [0, 15] medal-band-class potential). The empirical evidence at v6/v7/v8 baseline supports the rescope: v7's 58.89 IS the pure-analytical NSCS06-paradigm ceiling; further within-paradigm optimization (Variant C-1/C-2/C-3) is structurally bounded by [40, 60] band. Reaching medal-band requires class-shift to hybrid-paradigm (sister symposium #864 Path 3 Path C) OR pivot to non-NSCS06 substrate class. The operator's 2026-05-18 directive ('Variant C path') addresses the within-paradigm question; the hybrid-paradigm question remains DEFERRED-pending-operator-budget-reauthorization per sister symposium #864."
council_decisions_recorded:
  - "op-routable #1: PRIMARY Variant C-1 design memo with K-coverage measurement section per sister symposium #864 op-routable #3; predicted band [50, 55] (sister) OR [55, 58] (Shannon conservative); $5-15 Modal T4 smoke if operator family-DEFER reversed"
  - "op-routable #2: SECONDARY Variant C-2 (per-subband per-class CDF on residual only) — modest improvement over Variant C-1 at marginal complexity increase"
  - "op-routable #3: TERTIARY Variant C-3 (UNIWARD-conditioned wavelet residual with Wyner-Ziv applied to wavelet residual stream) per Yousfi+Fridrich+Wyner_memorial recommendation"
  - "op-routable #4: METHODOLOGY-PARADIGM-EXTENSION Variant C-4 (K-coverage measurement methodology validation memo only; $0 GPU); structural prerequisite for any Variant C-1/C-2/C-3 dispatch authorization"
  - "op-routable #5: PRESERVE operator family-DEFER 2026-05-16 + sister symposium #864 unanimous REFUSE verdict; Variant C-* dispatch authorization REQUIRES explicit operator family-DEFER reversal per Contrarian + Assumption-Adversary"
  - "op-routable #6: register canonical probe outcomes ledger entry per Catalog #313 for v6 EMPIRICAL_FALSIFICATION + v7 SUCCESS_44_PERCENT + v8 EMPIRICAL_FALSIFICATION + Variant C-1 PREDICTED_PROCEED_PENDING_OPERATOR_FAMILY_DEFER_REVERSAL"
  - "op-routable #7: NEW Catalog #303 sister gate proposal — check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section (sister to Catalog #303 cargo-cult audit; closes META cargo-cult #8 structurally)"
  - "op-routable #8: 30-day retrospective per CLAUDE.md Mission Alignment Consequence 3 — re-audit 2026-06-17"
---

# Per-substrate symposium: NSCS06 v8 Path B Variant C reactivation (cargo-cult composition limit + surgical-addition discipline)

**Substrate**: `nscs06_v8_path_b_variant_c` (Variant C = preserve v7 chroma anchor + apply v8 cargo-cult-unwinds SELECTIVELY)
**Empirical anchor**: NSCS06 v6 = 105.15 (553× outside-band) / v7 Path A = 58.89 (within [40, 65] band; +44% improvement) / v8 Path B = 104.98 (600× outside-band; -78% regression). META cargo-cult #8 discovered: architectural-transition-preserves-previously-unwound-cargo-cults is CARGO-CULTED-EMPIRICALLY-FALSIFIED.
**Frontier baseline**: 0.19205 [contest-CPU] (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; archive sha `6bae0201`).
**Verdict**: PROCEED_WITH_REVISIONS — Variant C-1 (preserve v7 anchor + DB4 depth-1 on residual ONLY) IS the canonical surgical-addition reactivation. **PRESERVES operator family-DEFER 2026-05-16 + sister symposium #864 unanimous REFUSE**; Variant C-* dispatch authorization REQUIRES explicit operator family-DEFER reversal per Catalog #199 paired-env operator-frontier-override.
**Council tier**: T3 (substrate-class promotion decision with paradigm-vs-implementation classification).

## 1. Why this v2 symposium

Per operator directive 2026-05-18: *"review the implementations for those that failed due to cargo cult assumptions and convene grand council symposiums to determine and design and implement and carry out their reactivation criteria"* + specifically: *"Variant C path: keep v7's chroma-preserving (the wave that produced +44%) + apply v8's cargo-cult-unwinds selectively (NOT all-at-once)"*.

**Sister symposium #864** (`council_per_substrate_symposium_nscs06_v8_path_b_20260517.md`) deliberated 2026-05-17 with **UNANIMOUS REFUSE 13-of-13 T3 vote** + 3 reactivation paths preserved:

1. **Path 1 (Path B-LITE)**: DB4 depth-1 chroma+grayscale-only with v7 anchor preserved (predicted [50, 55])
2. **Path 2 (Path B-CHROMA-AXIS)**: augment v7 anchor with full-resolution wavelet Cb+Cr (predicted [45, 52])
3. **Path 3 (Path C)**: hybrid analytical+neural residual decoder (predicted [0, 15]; DEFERRED-pending-operator-budget-reauthorization)

**THIS v2 symposium** EXTENDS sister #864 with:

1. **Variant C-1** = sister #864 Path B-LITE (CONFIRMED equivalence per Contrarian dissent)
2. **Variant C-2** ≈ sister #864 Path B-CHROMA-AXIS (per-subband per-class CDF on residual only; sister-equivalent surgical addition)
3. **Variant C-3** (NEW): UNIWARD-conditioned wavelet residual with Wyner-Ziv applied to wavelet residual stream (per Yousfi+Fridrich+Wyner_memorial)
4. **Variant C-4** (NEW): K-coverage measurement methodology validation memo only; $0 GPU; structural prerequisite for Variant C-1/C-2/C-3 dispatch

This symposium ANSWERS the operator's 4 specific questions:

1. **Re-audit which v8 cargo-cults were unwound vs which were re-introduced**: cargo-cult #3 + #6 UNWOUND; cargo-cult #1 + #2 RE-INTRODUCED (v8→v7 regression per sister #864 cargo-cult #8 META discovery)
2. **Variant C path: keep v7 chroma-preserving + apply v8 unwinds selectively**: Variant C-1/C-2/C-3 IS the canonical surgical-addition discipline (Carmack + Hotz engineering methodology); predicted band [40, 58]
3. **Is the NO-neural-at-medal-band claim still HARD-EARNED OR cargo-cult #5?**: HARD-EARNED at paradigm-level falsification scope per sister #864 cargo-cult #5 update (547× worse than 0.192 frontier); medal-band REQUIRES class-shift to hybrid-paradigm OR pivot to non-NSCS06 class
4. **Cross-substrate cargo-cult-unwind monotonicity structural limit**: HARD-EARNED-EMPIRICAL per sister #864 cargo-cult #8 + #10 META discoveries; cargo-cult-unwind monotonicity HOLDS for SURGICAL ADDITION + BREAKS for ARCHITECTURAL REPLACEMENT; K-coverage methodology extension IS the canonical pre-dispatch gate

## 2. Cargo-cult audit per assumption (Catalog #303)

Sister symposium #864 §1 enumerated 7 cargo-culted assumptions + 3 NEW META cargo-cults (#8, #9, #10). This v2 symposium UPDATES the audit per the operator's Variant C directive:

| # | Cargo-cult | v6 status | v7 Path A status | v8 Path B status | Variant C-1 INTENT | Variant C-1 PREDICTED status |
|---|------------|-----------|------------------|------------------|---------------------|------------------------------|
| 1 | Closed-form scorer-argmax bit allocator suffices | UNCONFIRMED | UNWOUND (per-class CDF) | REGRESSED (per-subband CDFs cannot recover SegNet color cues) | PRESERVED (v7's per-class CDF retained) | HARD-EARNED if v7 anchor preserved |
| 2 | Y=R=G=B replication | CARGO-CULTED | UNWOUND (per-class RGB anchor + luma scaling) | REGRESSED (per-subband-AC-replication at different abstraction layer) | PRESERVED (v7's chroma anchor architecture retained) | HARD-EARNED if v7 anchor preserved |
| 3 | Spatial-independent CDF entropy is optimal | CARGO-CULTED | WAIVED-FUTURE-PATH-B | UNWOUND (DB4 depth-2 decorrelates 5-10x per Mallat 1989) | SURGICALLY UNWOUND (DB4 depth-1 on residual ONLY; 2-3x decorrelation) | HARD-EARNED-EMPIRICAL if surgical-addition holds |
| 4 | 2-of-6 pose-warp suffices | CARGO-CULTED | UNWOUND (6-DOF affine) | UNWOUND-PRESERVED + Wyner-Ziv adds temporal residual (PARTIAL REGRESSION) | PRESERVED (v7's 6-DOF affine retained) | HARD-EARNED if v7 pose-warp preserved |
| 5 | NO-neural-at-medal-band is achievable | CARGO-CULTED | WAIVED (Path A is bounded [40, 65]) | PARADIGM-LEVEL FALSIFICATION CONFIRMED (547× worse than 0.192) | WAIVED-EXPLICIT (Variant C scope is plateau-adjacent [40, 58] NOT medal-band) | HARD-EARNED-EMPIRICAL |
| 6 | symposium-#4-band-prediction without distortion model | CARGO-CULTED | REPLACED (MacKay+Fridrich+Tao bounds) | UNWOUND-AT-PROTOCOL-LEVEL (Dykstra-feasibility check) | UNWOUND-VARIANT-C-1-LEVEL (per Boyd + Dykstra) | HARD-EARNED-EMPIRICAL |
| 7 | PR#56 grayscale-LUT generalizes from masks to frames | CARGO-CULTED | WAIVED (preserve for masks only) | WAIVED-PRESERVED | WAIVED-PRESERVED | CARGO-CULTED-EMPIRICALLY-VERIFIED |

**Variant C-1 cargo-cult-composition K-coverage matrix**: 4 v7-anchor-preserved unwinds (#1, #2, #4, #6) + 1 NEW surgical unwind (#3 DB4 depth-1) + 2 waived (#5, #7) = 5-unwind composition at the {1,2,3,4,6}-unwind-subset sub-lattice point. Sister symposium #864 cargo-cult #8 + #10 META discoveries predict this composition IS monotonic (surgical addition preserves prior unwinds) AND lands within predicted band [50, 58] per Shannon + Carmack convergence.

## 3. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | v6 baseline | v7 Path A | v8 Path B | Variant C-1 PREDICTED |
|---|-----------|-------------|-----------|-----------|-----------------------|
| 1 | UNIQUENESS (class-shift not within-class) | NO (NSCS family refinement) | NO (refinement of v6) | NO (refinement of v6/v7) | NO (refinement of v7; intentional surgical-addition) |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | YES (~88 LOC) | NO (~200 LOC; substrate_engineering) | NO (~1312 LOC + 44.8K trainer; substrate_engineering) | YES (≤200 LOC delta from v7 ≈ ~400 LOC total) |
| 3 | DISTINCTNESS | YES | YES | YES | YES (DB4 depth-1 on residual is distinct from v7's per-class CDF chroma anchor) |
| 4 | RIGOR | partial | PARTIAL-PASS (58.89 inside [40, 65]) | PARTIAL-FAIL (104.98 outside [15, 25]; composability check FAIL) | EXPECTED PARTIAL-PASS (K-coverage methodology validation IS the canonical rigor extension) |
| 5 | OPTIMIZATION PER TECHNIQUE | partial | YES (per-class chroma anchor substrate-optimal) | PARTIAL (per-subband per-class Laplacian-prior AC optimal; chroma REGRESSED) | YES (DB4 depth-1 on residual IS surgical-addition substrate-optimal; chroma anchor PRESERVED) |
| 6 | STACK-OF-STACKS-COMPOSABILITY | NO | NO (in principle composable; not tested) | EMPIRICALLY FAILED (cargo-cult #8 surfaced) | EXPECTED YES (surgical-addition discipline PRESERVES composability) |
| 7 | DETERMINISTIC REPRODUCIBILITY | YES | YES (44 tests pass) | YES (43+ tests pass; bit-identical archive) | EXPECTED YES (DB4 depth-1 deterministic) |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | YES (~30s inflate) | YES (~5 min inflate on T4) | YES (3.4s inflate; archive 258x smaller) | EXPECTED YES (~5-10 min inflate on T4; archive ~2-2.5MB) |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | 105.15 FAIL | 58.89 PARTIAL (within band; NOT medal-band) | 104.98 FAIL (worse than v6) | EXPECTED [50, 58] PARTIAL (improvement from v7 of 3-9 score points; NOT medal-band) |

**Overall**: 7 of 9 dimensions PASS or PARTIAL-PASS at Variant C-1; 0 of 9 FAIL. The mode of failure for medal-band convergence is **PARADIGM-LEVEL (per cargo-cult #5)** — the NSCS06-pure-analytical paradigm is empirically ceiling-bounded at [40, 60] per the v7 baseline + Variant C-1 prediction. Medal-band convergence REQUIRES class-shift to hybrid-paradigm (Path 3 Path C per sister #864; DEFERRED) or pivot to non-NSCS06 substrate class.

## 4. Observability surface (Catalog #305)

1. **Inspectable per layer**: Variant C-1 substrate emits per-pair DB4 depth-1 subbands (LL, LH, HL, HH) + v7's per-class chroma anchor + 6-DOF affine pose + WLV2-LITE archive grammar bytes. All inspectable via `src/tac/substrates/nscs06_v8_path_b_variant_c/inflate.py::inflate_one_video` debug mode + `wavelet_codec.py::decode_subband_arith_depth_1`. CANONICAL via sister #864 inflate path.
2. **Decomposable per signal**: per-task distortion (seg / pose / rate) at Variant C-1 measurable via `contest_auth_eval.py`; predicted decomposition seg ≈ 25-27 (v7-preserved), pose ≈ 30-31 (v7-preserved), rate ≈ 1.5-1.75 (v7's 2.67 minus 0.92-1.17 surgical-addition rate savings).
3. **Diff-able across runs**: archive sha256 + per-layer state_dict sha256; categorical posterior alphabet usage stats per layer.
4. **Queryable post-hoc**: canonical posterior at `.omx/state/council_deliberation_posterior.jsonl` per Catalog #325; probe outcomes at `.omx/state/probe_outcomes.jsonl` per Catalog #313; sister symposium #864 anchor preserved.
5. **Cite-able**: run-tuple (substrate=nscs06_v8_path_b_variant_c_1, commit, config=DB4_depth=1_surgical_addition, archive_sha, call_id) per Catalog #245.
6. **Counterfactual-able**: byte-mutation gate per Catalog #139 — verify surgical-addition residual stream consumption; sister to sister #864's K-coverage methodology validation per op-routable #3.

**Overall observability score: 6 / 6 — STRONG.** Variant C-1 inherits sister #864's substrate-level observability + ADDS the K-coverage measurement methodology validation as the missing counterfactual-able facet.

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL" + Catalog #308 ≥3 alternative-probe-methodologies)

### Reactivation Variant C-1 (PRIORITY 1): DB4 depth-1 chroma+grayscale-only with v7 anchor preserved (= sister #864 Path B-LITE)

- **Description**: Surgical addition of DB4 depth-1 wavelet decorrelation to v7's residual stream ONLY. PRESERVE v7's per-class chroma anchor architecture, per-class CDF infrastructure, 6-DOF affine pose-warp. Add cargo-cult #3 unwind via wavelet decomposition on residual only (NOT replacing chroma anchor). Maintains 4-of-7 v7 unwinds + adds 1-of-3 remaining unwinds = 5-of-7 unwind composition.
- **Predicted ΔS band**: `[50, 55]` (sister #864 prediction) OR `[55, 58]` (Shannon conservative). Mechanism: v7's 58.89 minus 0.92-3.89 score reduction from DB4 depth-1 rate-axis savings (Mallat 1989 narrowed range at depth-1).
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324. Reactivation criterion: post-training Tier-C density measurement on landed Variant C-1 archive.
- **Predicted cost**: $5-15 Modal T4 50-100ep smoke (operator family-DEFER reversal required) + $0.20 CUDA re-eval = $5.20-15.20 total
- **Structural verdict**: PRIMARY CANONICAL REACTIVATION — per Carmack + Hotz surgical-addition discipline + Shannon conservative band prediction + Boyd convex-optimization framework + Mallat wavelet-multi-scale verdict. ANSWERS operator question 2 (Variant C path).
- **Implementation complexity**: ≤200 LOC delta from v7 (v7 ≈ 400 LOC; Variant C-1 ≈ 600 LOC total). Per Hotz LOC budget enforcement: architecture-replacement-by-stealth IS being violated if implementation exceeds budget.
- **Composability**: ADDITIVE with v7 anchor architecture; PRESERVES UNIWARD-aligned chroma error distribution; PRESERVES per-class chroma signal that SegNet uses.
- **Prerequisites**: (a) **operator family-DEFER 2026-05-16 reversal** per Contrarian + Assumption-Adversary VETO conditions; (b) Variant C-1 design memo with K-coverage measurement section per op-routable #4 + sister #864 op-routable #3; (c) Catalog #325 6-step contract satisfied (THIS symposium IS step 4-5).

### Reactivation Variant C-2 (PRIORITY 2): Per-subband per-class CDF on residual only (≈ sister #864 Path B-CHROMA-AXIS)

- **Description**: Surgical addition of per-subband per-class CDF compression to v7's residual stream ONLY. PRESERVE v7's per-class chroma anchor + 6-DOF affine pose-warp. Add cargo-cult #3 unwind via per-subband per-class CDF on residual + cargo-cult #1 dual-unwind via residual-specific bit allocator. 5-of-7 unwind composition (different selective subset from Variant C-1).
- **Predicted ΔS band**: `[50, 58]` contest-CPU. Mechanism: per-subband per-class CDFs on residual preserve per-class differentiation at residual scale; slightly less rate savings than Variant C-1's DB4 (2-3x decorrelation vs 1.5-2x).
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $5-15 Modal T4 50-100ep smoke (operator family-DEFER reversal required)
- **Structural verdict**: SECONDARY EMPIRICAL — tests the per-subband CDF approach at the surgical-addition discipline (NOT v8's architectural-replacement). If Variant C-1 lands within predicted band AND Variant C-2 lands within predicted band, the surgical-addition discipline IS empirically validated across 2 different unwind compositions.
- **Implementation complexity**: ~250 LOC delta from v7 (per-subband per-class CDF infrastructure + residual-specific bit allocator).
- **Composability**: ORTHOGONAL with Variant C-1 (different unwind composition); COMPOSABLE with v7 architecture.
- **Prerequisites**: Same as Variant C-1.

### Reactivation Variant C-3 (PRIORITY 3, NEW): UNIWARD-conditioned wavelet residual with Wyner-Ziv applied to wavelet residual stream

- **Description**: Per Yousfi + Fridrich + Wyner_memorial recommendation: apply UNIWARD-canonical per-pixel cost-weighted compression to v7's chroma residual stream (errors in textured regions undetectable per Fridrich 2012 UNIWARD canonical) + apply Wyner-Ziv per-pixel temporal residual coding to the wavelet residual stream (NOT the raw pixel stream). Sister #864 cargo-cult #9 empirical-falsification (Wyner-Ziv at raw pixel surface) REDIRECTED to wavelet residual surface where per-pixel temporal redundancy IS HIGHER.
- **Predicted ΔS band**: `[40, 50]` contest-CPU. Mechanism: UNIWARD-conditioned chroma error distribution + Wyner-Ziv wavelet residual gain (20-30% rate reduction per Wyner_memorial; vs v8's 4% at raw pixel surface).
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $7-20 Modal T4 100ep smoke (operator family-DEFER reversal required)
- **Structural verdict**: TERTIARY EMPIRICAL — NEW reactivation path per Yousfi+Fridrich+Wyner_memorial; addresses cargo-cult #9 structural-fit redirection.
- **Implementation complexity**: ~350 LOC delta from v7 (UNIWARD weight computation + Wyner-Ziv wavelet residual codec + integration).
- **Composability**: ORTHOGONAL with Variant C-1/C-2; COMPOSABLE with v7 architecture.
- **Prerequisites**: Same as Variant C-1.

### Reactivation Variant C-4 (PRIORITY 4, METHODOLOGY-PARADIGM-EXTENSION): K-coverage measurement methodology validation memo only

- **Description**: Per Daubechies + Time-Traveler-protege + sister #864 op-routable #3: design memo + canonical helper for K-coverage measurement methodology extension. K=8 sample point reconstruction of cargo-cult-unwind landscape via compressive-sensing methodology (Daubechies-DeVore-Fornasier-Gunturk 2010). Sample points: {v6 baseline / v7 +44% / v8 -78% / Variant C-1 predicted / Variant C-2 predicted / Variant C-3 predicted / methodology-only structural anchor / Z3 v2 architecture-transition sister substrate anchor}. The K-coverage matrix IS the canonical pre-dispatch gate that empirically validates whether ANY future substrate's predicted operating point is REACHABLE.
- **Predicted ΔS band**: NULL (methodology validation; no score-axis claim)
- **Predicted_band_validation_status**: `pending_post_training` (for downstream substrate dispatches consuming the methodology)
- **Predicted cost**: $0 GPU (design memo + canonical helper only); ~6h editor work
- **Structural verdict**: METHODOLOGY-PARADIGM-EXTENSION — sister to NEW Catalog #303 sister gate (op-routable #7). The K-coverage measurement methodology IS the structural protection against META cargo-cult #8 (architectural-transition-preserves-previously-unwound-cargo-cults).
- **Implementation complexity**: ~400 LOC canonical helper at `tac.compressive_coverage_estimator.cargo_cult_unwind_K_coverage_matrix` + ~200 LOC tests + ~50 LOC integration with sister Catalog #303 audit.
- **Composability**: STRUCTURAL PREREQUISITE for Variant C-1/C-2/C-3 dispatch authorization.
- **Prerequisites**: NONE (canonical helper is design-memo + implementation-only; $0 GPU).

### Reactivation priority ordering

1. **Variant C-4 (METHODOLOGY-PARADIGM-EXTENSION)**: $0 GPU; STRUCTURAL PREREQUISITE for all other Variant C paths. Fire FIRST.
2. **Variant C-1 (HIGHEST EV per Variant C-* dispatch)**: $5-15 cost; surgical-addition canonical; predicted [50, 58] band.
3. **Variant C-2 (SECOND EV)**: $5-15 cost; per-subband CDF surgical-addition; predicted [50, 58] band.
4. **Variant C-3 (THIRD EV, NEW)**: $7-20 cost; UNIWARD + Wyner-Ziv-wavelet-residual; predicted [40, 50] band.
5. **Sister #864 Path 3 Path C (DEFERRED)**: $30 cost; hybrid analytical+neural; predicted [0, 15] medal-band-class; DEFERRED-pending-operator-budget-reauthorization per sister #864 + Quantizr.

**Recommendation**: Variant C-4 IMMEDIATELY ($0 GPU; methodology validation is the structural protection against future cargo-cult-composition failures). HOLD Variant C-1/C-2/C-3 funding decision until:
1. Operator family-DEFER 2026-05-16 reversal (explicit operator override per Catalog #199 paired-env)
2. Variant C-4 methodology validation memo lands + K-coverage matrix populated with K=8 sample points

If both conditions land, fire Variant C-1 first ($5-15; tests surgical-addition discipline at canonical sub-lattice point). Conditional Variant C-2/C-3 follow-on based on Variant C-1 empirical outcome.

**ALTERNATIVE PATH per Assumption-Adversary + Quantizr**: if the operator's score-lowering budget is limited, pivot to non-NSCS06 substrate class (HiNeRV / sane_hnerv / DP1 stacking / ATW V2-1 V4 probe / C6 IBPS RSSM categorical posterior per Symposium #2 of this wave). The NSCS06 family ceiling at [40, 60] is plateau-adjacent — Variant C-* improvement of 3-9 score points does NOT reach medal-band frontier.

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status for v8 Path B (104.98 empirical)**: `phantom_random_init_REGRESSED_TO_v6_EMPIRICAL_ANCHOR_104p98` per sister symposium #864 §6.

**Predicted_band_validation_status for Variant C-1**: `pending_post_training` per Catalog #324. Reactivation criterion: post-training Tier-C density measurement on landed Variant C-1 archive via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <sha>`. If empirical ΔS lands within `[50, 58]`, ratify the surgical-addition discipline + advance to L2. If outside band, surface as Catalog #324 violation and re-symposium.

**Predicted_band_validation_status for Variant C-2/C-3**: same pattern; each variant's predicted band gates its post-training Tier-C validation.

**Predicted_band_validation_status for Variant C-4 (methodology validation)**: N/A (no score-axis claim); methodology validation succeeds when K-coverage matrix is populated with K=8 sample points + bounded uncertainty O(sqrt(N/K)) is empirically demonstrated.

## 7. Continual-learning anchor (Catalog #325 dispatch eligibility gate (d))

After this memo lands, the canonical posterior anchor IS registered to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per the canonical 4-layer pattern (Catalog #245 exemplar). The anchor schema includes:

- PROCEED_WITH_REVISIONS verdict (binding on operator family-DEFER reversal + Variant C-4 methodology validation prerequisite)
- 15-seat attendee list (6 inner sextet + 9 grand-council: Carmack, Hotz, Mallat, Daubechies, Wyner_memorial, Quantizr, Time_Traveler_protege, Schmidhuber, Boyd)
- 6-assumption Assumption-Adversary verdict
- 8 op-routables
- mission-alignment=frontier_protecting (Variant C-* improves from v7 plateau but does NOT reach medal-band)
- override_invoked=false
- horizon_class=plateau_adjacent
- canonical_frontier_anchor per Catalog #316
- deferred_substrate_id=nscs06_v8_path_b_wavelet_residual (preserves sister symposium #864 deferral) + new deferred_substrate_id=nscs06_v8_path_b_variant_c_1 (THIS symposium) + deferred_substrate_retrospective_due_utc=2026-06-17T00:00:00Z

Downstream consumers per Catalog #325:

- **Catalog #325 STRICT preflight** sees PROCEED_WITH_REVISIONS verdict + preserves sister #864 REFUSE verdict; Variant C-* dispatch authorization REQUIRES explicit operator family-DEFER reversal + Variant C-4 methodology validation prerequisite landing.
- **Cathedral autopilot ranker** consumes via `tac.council_continual_learning.query_anchors_by_topic('nscs06_v8_path_b_variant_c')`; weights Variant C-4 PRIMARY methodology validation; Variant C-1 SECONDARY surgical-addition.
- **Probe-outcomes ledger (Catalog #313)** receives sister-registered probe outcomes for v6 + v7 + v8 + Variant C-1 predicted per op-routable #6.

## 8. Cross-references

- **Canonical NSCS06 v6→v7→v8→Variant C lineage**:
  - `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` (v6 falsification + 6-path enumeration)
  - `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md` (v7 Path A design + 44% improvement empirical anchor)
  - `.omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md` (v8 Path B design memo; commit 963549469)
  - `.omx/research/council_per_substrate_symposium_nscs06_v8_path_b_20260517.md` (sister #864 unanimous REFUSE T3 13-of-13)
  - `.omx/research/nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516.md` (operator family-DEFER directive)
- **Sister symposia (per-substrate Catalog #325 cohort)**:
  - `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (parent ATW V2 reactivation; cross-pollination via Z6 Wave 2 4c)
  - `.omx/research/council_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518.md` (Symposium #1 of this wave)
  - `.omx/research/council_per_substrate_symposium_c6_ibps_post_empirical_reactivation_v2_20260518.md` (Symposium #2 of this wave)
- **Sister K-coverage methodology proposal**:
  - sister #864 op-routable #3: NEW Catalog #303 sister gate `check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section`
- **Catalog gates fired by this v2 symposium**: #229 (premise verification) + #245 (Modal call_id ledger) + #265 (symposium impls canonical-contract) + #291 (per-session META-ASSUMPTION cadence) + #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist) + #296 (Dykstra-feasibility predicted-band) + #300 (council v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface) + #307 (paradigm-vs-implementation classification — v8 IMPLEMENTATION FALSIFIED; PARADIGM-partially-salvageable via Variant C; PARADIGM #5 NO-neural-at-medal-band FALSIFIED at medal-band scope) + #308 (alternative-probe-methodologies enumeration — Variant C-1/C-2/C-3/C-4 + Path C hybrid deferred) + #313 (probe-outcomes ledger registration) + #315 (substrate at optimal form before paid dispatch) + #316 (canonical frontier anchor) + #324 (post-training Tier-C validation discipline) + #325 (per-substrate symposium discipline — THIS v2 symposium EXTENDS sister #864 with Variant C reactivation paths for 14 days).
- **Catalog gates PROPOSED by this v2 symposium**: NEW Catalog #303 sister gate `check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section` (op-routable #7); structural protection against META cargo-cult #8.
- **Canonical implementations cited**: Mallat 1989 + Daubechies 1992 Ch. 6 + Wyner-Ziv 1976 + Antonini-Barlaud-Mathieu-Daubechies 1992 + Witten-Neal-Cleary 1987 + MacKay 2003 Ch. 2 + Daubechies-DeVore-Fornasier-Gunturk 2010 + Finn-Abbeel-Levine 2017 (MAML; Variant C-4 methodology validation prior framework) + Catalog #277 wavelet multi-scale ranker + Catalog #296 Dykstra-feasibility check + sister symposium #864's 22-voice grand council baseline.

## 9. Operator op-routables (for parent agent + main Claude)

1. **PRIMARY (highest EV; $0 GPU)**: Variant C-4 K-coverage measurement methodology validation memo + canonical helper `tac.compressive_coverage_estimator.cargo_cult_unwind_K_coverage_matrix`. STRUCTURAL PREREQUISITE for Variant C-1/C-2/C-3 dispatch authorization. ~6h editor work. Lane `lane_variant_c_4_k_coverage_methodology_validation_20260518` (suggested L0).

2. **CONDITIONAL ON OPERATOR FAMILY-DEFER REVERSAL**: Variant C-1 design memo + dispatch (sister symposium #864 Path B-LITE = canonical equivalent). $5-15 Modal T4 50-100ep smoke. Predicted ΔS band `[50, 58]` per Carmack + Shannon convergence.

3. **CONDITIONAL ON VARIANT C-1 EMPIRICAL VALIDATION**: Variant C-2 per-subband CDF surgical-addition. $5-15 Modal T4 smoke. Tests surgical-addition discipline at different unwind composition sub-lattice point.

4. **CONDITIONAL ON VARIANT C-1 + VARIANT C-2 EMPIRICAL VALIDATION**: Variant C-3 UNIWARD + Wyner-Ziv wavelet residual. $7-20 Modal T4 smoke. Tests cargo-cult #9 structural-fit redirection.

5. **PRESERVE operator family-DEFER 2026-05-16** + sister symposium #864 unanimous REFUSE per Contrarian + Assumption-Adversary VETO conditions. Variant C-* dispatch authorization REQUIRES explicit operator family-DEFER reversal via Catalog #199 paired-env operator-frontier-override.

6. **PROBE OUTCOMES LEDGER REGISTRATION** per Catalog #313:
   - NSCS06 v6: register as `verdict=EMPIRICAL_FALSIFICATION, status=blocking, methodology=v6_baseline_grayscale_lut_replication`
   - NSCS06 v7: register as `verdict=SUCCESS_44_PERCENT_IMPROVEMENT, status=advisory, methodology=v7_path_a_chroma_anchor_plus_6_dof_affine, expires_at_utc=2026-06-17T00:00:00Z`
   - NSCS06 v8: register as `verdict=EMPIRICAL_FALSIFICATION, status=blocking, methodology=v8_path_b_db4_depth2_per_subband_per_class_cdf_plus_wyner_ziv_temporal_residual, alternative_probe_methodologies=[variant_c_1_db4_depth_1_residual_only_v7_anchor_preserved, variant_c_2_per_subband_cdf_residual_only_v7_anchor_preserved, variant_c_3_uniward_wavelet_residual_wyner_ziv_redirected, variant_c_4_k_coverage_methodology_validation_memo_only]`
   - Variant C-1: register as `verdict=PREDICTED_PROCEED_PENDING_OPERATOR_FAMILY_DEFER_REVERSAL_AND_VARIANT_C_4_METHODOLOGY_VALIDATION, status=advisory, predicted_band=[50, 58]`

7. **NEW Catalog #303 sister gate proposal**: `check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section`. Refuses substrate design memos at `.omx/research/*_design_<YYYYMMDD>.md` (dated >= 2026-05-18) that claim cargo-cult-unwind methodology composition WITHOUT explicit K-coverage measurement section + Daubechies-DeVore-Fornasier-Gunturk 2010 citation. Closes META cargo-cult #8 structurally. Same-line waiver `# CARGO_CULT_K_COVERAGE_SECTION_OK:<rationale>`.

8. **30-DAY RETROSPECTIVE** per CLAUDE.md "Mission alignment" Consequence 3 (re-audit 2026-06-17): re-audit whether Variant C-1/C-2/C-3 outcomes (if dispatched) validated the surgical-addition discipline + whether the K-coverage methodology extension was empirically validated.

## Symposium verdict summary

- **Tier**: T3
- **Verdict**: PROCEED_WITH_REVISIONS (Variant C-4 PRIMARY methodology validation; Variant C-1/C-2/C-3 CONDITIONAL on operator family-DEFER reversal)
- **Top-priority reactivation path**: Variant C-4 K-coverage measurement methodology validation memo + canonical helper ($0 GPU; ~6h editor work; STRUCTURAL PREREQUISITE)
- **Predicted cost**: $0 immediate (Variant C-4 design memo) → if operator family-DEFER reversed: $5-15 (Variant C-1 smoke) → if Variant C-1 WIN: $10-30 follow-on (C-2 + C-3) = $0-45 total worst-case
- **Structural recommendation to operator**: Authorize Variant C-4 IMMEDIATELY ($0 GPU; methodology validation IS the canonical pre-dispatch gate that prevents future cargo-cult-composition failures across ALL substrates). HOLD Variant C-1/C-2/C-3 funding decision until (1) operator family-DEFER 2026-05-16 reversal + (2) Variant C-4 methodology validation memo lands. If both conditions land, fire Variant C-1 first ($5-15; tests surgical-addition discipline at canonical sub-lattice point). The empirical anchor at v6+v7+v8+Variant C-1 (if dispatched) closes the K=4 cargo-cult-unwind composition space; with Variant C-2 + C-3 + methodology-only + sister Z3 v2 sample points, K=8 reaches the canonical Daubechies-DeVore-Fornasier-Gunturk 2010 compressive-coverage bound.

**ALTERNATIVE PATH**: per Assumption-Adversary + Quantizr: if the operator's score-lowering budget prioritizes medal-band convergence, pivot to non-NSCS06 substrate class (HiNeRV / sane_hnerv / DP1 stacking / ATW V2-1 V4 probe per Symposium #1 / C6 IBPS RSSM categorical posterior per Symposium #2). The NSCS06 family ceiling at [40, 60] is plateau-adjacent — Variant C-* improvement of 3-9 score points does NOT reach medal-band frontier. The methodology validation (Variant C-4) IS the canonical RESCUE: future substrates inherit the K-coverage measurement discipline + structurally prevent cargo-cult-composition failures.

**Cross-pollination with sister wave**: Symposium #1 (V1 dense Faiss-IVF-PQ) recommends V4 hand-rolled probe ($0.15 free) — same K-coverage methodology applies to ATW V2-1 codebook design space. Symposium #2 (C6 IBPS RSSM categorical) recommends Path B2 ($5-15 smoke) — same K-coverage methodology applies to IB-family architecture composition. Variant C-4 methodology validation lands the canonical helper that BOTH sister symposiums consume.
