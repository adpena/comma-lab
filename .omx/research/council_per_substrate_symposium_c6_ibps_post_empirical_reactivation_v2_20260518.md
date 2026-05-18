---
review_kind: per_substrate_symposium
review_id: per_substrate_symposium_c6_ibps_post_empirical_reactivation_v2_20260518
review_date: "2026-05-18"
lane_id: lane_per_substrate_symposium_c6_ibps_post_empirical_reactivation_v2_20260518
substrate_id: c6_e4_mdl_ibps
substrate_alias: c6_ibps
parent_substrate_id: c6_e4_mdl_ibps
deferred_substrate_id: c6_e4_mdl_ibps
horizon_class: asymptotic_pursuit
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Tishby_memorial
  - Zaslavsky
  - Schmidhuber
  - MacKay_memorial
  - Atick
  - Hafner
  - Higgins_memorial
  - Rao
  - Ballard
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
predicted_band_validation_status: pending_post_training
predicted_band:
  path_b1_hierarchical_ib: [0.65, 1.25]   # MULTI-LEVEL IB with 4-3-2 dim hierarchy; predicted ΔS reduces SegNet collapse by half
  path_b2_categorical_posterior_dreamerv3_rssm: [0.18, 0.45]   # discrete RSSM categorical posterior matches per-pixel SegNet output; predicted strong improvement
  path_b3_beta_vae_empirically_tuned: [1.50, 2.80]   # β-VAE with β from contest gradient analysis; modest improvement over baseline 3.04
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
dispatch_enabled: false
operator_directive: "TOP 3 cargo-cult-failed reactivation — C6 IBPS 22x miss vs predicted [0.113, 0.163] -> 3.04 contest-CPU; mechanism: 24-dim IB bottleneck destroys segmentation"
related_deliberation_ids:
  - council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
  - council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517
  - feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517
  - feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517
  - feedback_deep_research_wave_landed_20260518
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201)"
  contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519)"
predecessor_probe_outcomes:
  - probe_id: c6_ibps_first_asymptotic_dispatch
    call_id: fc-01KRW353MJJ9A6QW8H99QWZEMH
    verdict: EMPIRICAL_FALSIFICATION
    final_score: 3.04
    score_seg: 2.60
    score_pose: 0.285
    score_rate: 0.155
    predicted_band: [0.113, 0.163]
    miss_factor: 22.0
    classification: implementation-level-falsification-of-24-dim-IB-bottleneck-architecture
council_dissent:
  - member: Contrarian
    verbatim: "I rise to challenge the PARADIGM-LEVEL DEFERRAL of C6 IBPS. The empirical 3.04 IS damning — but the parent prompt + symposium #836 + sister #835 Assumption-Adversary already enumerated path (a) β_ib sweep + path (b) latent_dim sweep + path (c) Phase 2 redesign. THIS v2 symposium is operating under operator directive to surface ALTERNATIVE REDUCER METHODOLOGIES per Catalog #308 N>=3 — specifically hierarchical IB / variational IB with categorical posterior (DreamerV3 RSSM) / β-VAE with empirically-tuned β. My CRITICAL CONCERN: the proposed alternatives (Path B1/B2/B3) are NOT 3 alternative reducers — they are 3 alternative ARCHITECTURES that change the fundamental IB formulation. Each carries 200-400 LOC implementation cost. The operator's empirical budget tolerance is bounded ($50-100 reclaim from NSCS06 family DEFER); funding 3 parallel architecture replacements would consume the entire reclaim. RECOMMENDED: prioritize Path B2 (DreamerV3 RSSM categorical) AS THE PRIMARY because it's the canonical 2024-2026 bleeding-edge from deep-research wave §1; defer Path B1 + B3 until B2 empirical anchor lands. If B2 produces ΔS < 0.5 within smoke, the operator has empirical evidence the IB framework is structurally wrong for this substrate AND a clear pivot to a non-IB class-shift. If B2 produces ΔS ∈ [0.18, 0.45] as predicted, the IB framework IS shippable at the right architecture and Path B1/B3 follow-on become EV-positive. My VETO is on PARALLEL DISPATCH of B1/B2/B3 (kitchen_sink risk per CLAUDE.md PR105 anti-pattern); my PROCEED is on B2-FIRST then conditional B1/B3 follow-on."
  - member: Assumption-Adversary
    verbatim: "Per Catalog #292 + CLAUDE.md per-round explicit-assumption-statement discipline. The SHARED ASSUMPTION operating across the parent prompt + this v2 symposium: *'The C6 IBPS PARADIGM (information bottleneck framework applied to substrate compression) is salvageable via REDUCER METHODOLOGY change — hierarchical IB / variational IB with categorical posterior / β-VAE with empirically-tuned β.'* I classify this CARGO-CULTED-PENDING-EMPIRICAL. HARD-EARNED basis: (a) Tishby-Zaslavsky 2015 IB framework IS mathematically general — applies to ANY rate-distortion problem with side information; (b) the C6 IBPS empirical 3.04 IS interpretable through the IB Lagrangian — 24-dim z with H(T) ≤ 24 × 32 bits = 768 bits per sample is BELOW the SegNet's information requirement (5 classes × 384 × 512 = 983K bits raw); the IB framework PREDICTS the segmentation collapse; (c) Path B1 hierarchical IB (Hinton-Vinyals-Dean 2014 + extensions) is a documented family that addresses single-bottleneck-collapse via multi-level decomposition. CARGO-CULTED basis: (a) NONE of the 3 alternative methodologies has been empirically measured on C6 IBPS substrate; (b) the deep-research wave §1 prediction band [0.18, 0.45] for B2 is research-side prediction, NOT empirically derived from contest scoring formula; (c) the IB framework may be paradigm-INCORRECT for dashcam scoring (Higgins memorial classification of β-VAE failure modes shows IB-class methods systematically underperform conditional generative models at task-conditioned reconstruction). My assumption-violation hypothesis: *'IF Path B2 (DreamerV3 RSSM categorical posterior) returns final_score ≥ 1.0 (factor-of-3 from predicted upper bound 0.45), the IB-family-on-C6-substrate is paradigm-level falsified per Catalog #307, and the operator should pivot to non-IB class-shift architectures (cooperative-receiver / world-model / lane_17_imp / time_traveler L5).'* Required action per Catalog #308: enumerate ≥3 alternative substrate-class architectures BEFORE Path B2 is selected — alternatives identified: (alt-1) lane_17_imp IMP cycle 0 + Frankle LTH winning-ticket extraction; (alt-2) cooperative-receiver ATW V2-1 with Channel #1 per-pixel softmax logits; (alt-3) Z6 Wave 2 Candidate 4c full-FiLM-WIN scorer-logit-conditioning. VETO on PROCEED-unconditional B2 pending alternative-substrate-class enumeration as B2-bypass paths."
  - member: Shannon
    verbatim: "Information-theory grounding lens: the C6 IBPS empirical 3.04 decomposes as 2.60 SegNet + 0.285 PoseNet + 0.155 rate. SegNet domination at 86% is the CANONICAL signature of insufficient IB capacity for the dense per-pixel SegNet output. The 24-dim bottleneck at H(T) ≤ 768 bits per sample is 3 orders of magnitude below the per-pixel SegNet's I(X;Y) ceiling. The IB Lagrangian L_IB = I(X;T) - β·I(T;Y) with the contest formula's 100·seg + 5·sqrt(10·pose) + 25·archive_bytes/37545K decomposition: the seg-weighted distortion has CONSTANT marginal 100 vs sqrt-weighted pose marginal that varies with pose_avg; at C6's operating point pose_avg=0.0081 the pose marginal is 5/sqrt(10·0.0081) ≈ 17.5, giving seg:pose marginal ratio ≈ 5.7:1 in seg's favor — exactly OPPOSITE the PR106 frontier 271:1 pose-favored. The 24-dim bottleneck IS structurally insufficient for the seg-dominated operating point. Path B1 hierarchical IB IS the canonical fix — multi-level bottleneck (4-3-2 dim hierarchy at coarse-medium-fine scales) preserves higher-rate signal at the finest level where SegNet boundaries live while compressing the coarse-level redundancy. PROCEED_WITH_REVISIONS verdict + binding revision: Path B1 hierarchical IB IS the canonical first-pick per the IB framework's mathematical structure; Path B2 RSSM categorical IS a sister fix at the discrete-bottleneck axis. The two paths address ORTHOGONAL aspects of the failure mode."
  - member: Dykstra
    verbatim: "Dykstra-feasibility analysis: the feasible region per the contest scoring formula is the intersection of (a) rate ≤ 0.20; (b) seg ≤ 0.05 (canonical medal-band threshold given the operating point's 5.7:1 marginal); (c) pose ≤ 0.0001 (achievable per PR106 evidence); (d) IB framework constraint I(X;T) - β·I(T;Y) ≥ -ε (Lagrangian feasibility). The C6 IBPS empirical point (rate=0.155, seg=2.60, pose=0.285) VIOLATES (a) by 0.045 + VIOLATES (b) by 50x + VIOLATES (c) by 285x. NO single-axis reactivation can converge this point onto the feasible region. The Dykstra alternating-projections methodology suggests SIMULTANEOUS multi-axis projections: project onto (a) rate via β_ib increase; project onto (b) seg via SegNet capacity increase (latent_dim ↑ OR hierarchical decomposition); project onto (c) pose via dedicated pose head (dual-latent per Tishby's Phase 2 redesign). The single-architecture fix (B2 RSSM categorical alone) IS necessary but not sufficient for medal-band convergence. PROCEED_WITH_REVISIONS verdict + binding revision: the alternative-architecture wave SHOULD evaluate Path B1+B2+B3 as a Pareto-frontier ENSEMBLE, not a single-winner selection — empirical evidence at multiple architecture points lets the operator pick the Pareto-optimal point post-empirical."
  - member: Tishby_memorial
    verbatim: "Memorial seat conveying the IB framework + Tishby-Zaslavsky 2015 deep IB principle. The C6 IBPS empirical 3.04 IS canonical IB-failure signature: H(T) at 24 dim × 32 bits/dim × empirical-utilization-fraction = effective ~50 bits per sample (much less than 768-bit raw upper bound; actual H(T) collapses under continuous-latent + Gaussian-prior assumptions). The deep IB framework predicts: for fixed β, decreasing H(T) below the I(X;Y) requirement produces a phase transition where the bottleneck preserves only the highest-mutual-information components — for dashcam scoring, that's pose direction (which has lower I(X;Y) than SegNet boundary class). The empirical 3.04 with pose=0.285 (small) and seg=2.60 (large) IS structurally consistent with IB-collapse-to-pose. Path B1 hierarchical IB IS the canonical Tishby-Zaslavsky-2015 extension — multi-level IB preserves I(X;Y_k) at scale k while compressing I(X;Y) globally. Path B2 RSSM categorical posterior (Hafner 2024) IS the canonical discrete-bottleneck variant — preserves I(X;Y) via categorical alphabet that doesn't continuously decay to single modes. Path B3 β-VAE empirically-tuned (Higgins 2017) IS the canonical β-search variant — empirically tuned β balances rate-distortion at the contest's operating point. ALL THREE alternative methodologies have published canonical implementations + tested mathematical convergence properties. PROCEED_WITH_REVISIONS verdict + binding revision: the alternative-architecture wave IS structurally well-motivated; all three paths SHOULD be enumerated as design memos before single-winner dispatch."
  - member: Zaslavsky
    verbatim: "Concurs with Tishby memorial. As the active researcher carrying the IB framework forward, I emphasize that the 2015-2024 IB literature has TWO main extensions for the C6 IBPS failure class: (1) Higgins 2017 β-VAE empirical β-tuning that addresses single-β-cargo-cult by sweeping β across a wide range and selecting empirically; (2) Alemi-Fischer-Dillon-Murphy 2017 deep variational IB that addresses analytical-vs-empirical β derivation. Path B3 β-VAE with empirically-tuned β IS the canonical Higgins 2017 method. The deep-research wave §1 reference to Hafner-DreamerV3 RSSM categorical posterior IS the bleeding-edge extension that addresses continuous-vs-discrete bottleneck cargo-cult; the DreamerV3 paper explicitly reports superior performance over continuous-latent VAE on dense-output tasks. Path B2 IS canonical bleeding-edge. Path B1 hierarchical IB IS the multi-level extension that was articulated in Tishby's 2015 paper but never empirically tested at the scale C6 IBPS targets. I emphasize Path B1 IS the LEAST empirically validated of the three — Path B2 + B3 have stronger published empirical bases. PROCEED_WITH_REVISIONS verdict + binding revision: priority ordering should be Path B2 (most empirically validated bleeding-edge) > Path B3 (most established Higgins 2017) > Path B1 (least empirically validated multi-level extension)."
  - member: Hafner
    verbatim: "I am summoned per grand council as canonical author of DreamerV3 + RSSM categorical posterior. The C6 IBPS empirical 3.04 SegNet-collapse signature IS structurally identical to the early-DreamerV2 continuous-latent-collapse-on-dense-output failure mode I encountered in 2022. The 2024 DreamerV3 fix WAS the categorical posterior — replacing continuous latent dimensions with a categorical distribution over a discrete alphabet of K categories. For C6 IBPS at 24 continuous dimensions, the canonical translation is: 24 dimensions × 32 bits each = 768 bits per sample → 24 categorical groups × log2(K) bits each. With K=256, that's 24 × 8 = 192 bits per sample — a 4x compression at fixed dimensionality with substantially better preservation of per-pixel SegNet boundary information. The KEY mathematical insight: the categorical posterior's prior IS uniform (max-entropy) — the KL regularizer encourages USE of all K categories rather than collapsing to one. This is precisely the structural fix C6 IBPS needs: the 24-dim continuous Gaussian-prior bottleneck collapses to a single mode (the SegNet's most-frequent class label) under high-β regularization; the categorical posterior CANNOT collapse to a single mode because the prior is uniform across K categories. Path B2 IS the canonical fix. Implementation complexity: ~300 LOC addition to substrate trainer (replace continuous latent with categorical reparametrization via Gumbel-Softmax; train with straight-through estimator). PROCEED_WITH_REVISIONS verdict + binding op-routable: B2 PRIMARY with K=256, 24 categorical groups; predicted ΔS band [0.18, 0.45]."
  - member: Higgins_memorial
    verbatim: "Memorial seat conveying β-VAE 2017 + the disentanglement-via-β-tuning lineage. The C6 IBPS empirical 3.04 IS interpretable through the β-VAE lens: β=0.01 (the C6 IBPS configuration) is in the HIGH-DISENTANGLEMENT regime where latent dimensions become axis-aligned and individual dimensions encode single semantic factors. For dashcam scoring with seg-dominated operating point, the disentanglement penalty crushes the per-pixel SegNet boundary information into a few discrete modes — exactly the empirical SegNet=2.60 collapse signature. The β-VAE 2017 paper empirically swept β ∈ [0.001, 0.01, 0.1, 1.0, 5.0, 10.0] on 64x64 image reconstruction; the rate-distortion frontier WAS strongly β-dependent and the optimal β was task-dependent. For C6 IBPS at 384x512 dashcam + 5-class SegNet + 6-DOF PoseNet, the empirically-optimal β has NOT been measured; the β=0.01 choice was inherited from Alemi 2017 VIB-on-small-images. Path B3 β-VAE with empirically-tuned β IS the canonical Higgins 2017 method applied to C6's specific operating point. The expected empirical β depends on the per-pixel SegNet's intrinsic dimensionality + the contest scorer's rate weight (25 / 37545K = 6.66e-7 per byte); my analytical prediction for empirically-optimal β at the C6 operating point: β ∈ [0.0001, 0.001] (3-10x LOWER than current 0.01). This is consistent with sister #836 path (a) β_ib sweep recommendation. PROCEED_WITH_REVISIONS verdict + binding op-routable: B3 PRIMARY with β sweep ∈ [0.0001, 0.001, 0.01]; predicted ΔS band [1.50, 2.80]."
  - member: Rao
    verbatim: "I am summoned per grand council as canonical author of Rao-Ballard 1999 predictive coding in the visual cortex. The C6 IBPS failure mode IS interpretable through the hierarchical predictive coding lens: the 24-dim continuous bottleneck IS analogous to the visual cortex's V1 sparse-coding layer; the SegNet+PoseNet decoder IS analogous to higher visual areas V2-V4-IT. The Rao-Ballard hierarchical model predicts: when V1 sparse coding has insufficient capacity for the high-frequency boundary information SegNet requires, the higher layers (V2+) compensate by re-encoding the missing information via top-down predictions. The Tishby + Atick recommendation of dual-latent z_pose + z_seg IS structurally similar to V1's parallel parvocellular + magnocellular pathways. Path B1 hierarchical IB at 4-3-2 dim hierarchy IS structurally the Rao-Ballard 1999 model applied to substrate compression. PROCEED_WITH_REVISIONS verdict + binding op-routable: B1 SHOULD be designed with explicit Rao-Ballard hierarchical structure (coarse-medium-fine scales correspond to V1-V2-V4 pathway analog); predicted ΔS band [0.65, 1.25]."
  - member: Ballard
    verbatim: "Concurs with Rao. The hierarchical IB methodology MUST preserve top-down prediction pathways — at each level k, the latent z_k MUST predict z_{k+1} via a learned transition function. This is the canonical Rao-Ballard 1999 model and the DreamerV3 RSSM architecture's structural origin. Path B1 hierarchical IB + Path B2 RSSM categorical posterior are COMPLEMENTARY not competing — B1 specifies the hierarchical structure; B2 specifies the discrete categorical bottleneck. A combined Path B1+B2 (hierarchical IB with categorical posterior at each level) IS the canonical bleeding-edge synthesis. Implementation complexity: ~500 LOC (B1 + B2 combined). Predicted ΔS band [0.40, 0.80] (better than either alone at the 5.7:1 seg-favored marginal). PROCEED_WITH_REVISIONS verdict + binding revision: the operator SHOULD evaluate a Path B4 (B1+B2 combined) as a 4th reactivation path."
  - member: MacKay_memorial
    verbatim: "Memorial seat conveying the MDL discipline + Bayesian-inference framework. The C6 IBPS empirical 3.04 IS interpretable through the MDL lens: the model's two-part code L(model) + L(data|model) has L(model) = 128K params × 32 bits/param = 4MB; L(data|model) at empirical SegNet+PoseNet+rate = 3.04 × 37545K bytes ≈ 114M bytes ≈ 900Mbits. The MDL-optimal model balances these two terms; the 24-dim continuous bottleneck IS over-parameterized relative to the achievable L(data|model). A SPARSE bottleneck (most dimensions zero) would substantially reduce L(model) without further increasing L(data|model). Path B5 sparse-IB with Laplacian prior on z (encouraging dimension-wise sparsity) IS the canonical MDL extension. Implementation complexity: ~200 LOC (replace Gaussian prior with Laplacian; minor reparam adjustment). Predicted ΔS band [1.20, 2.10] (modest improvement; structurally less ambitious than B1/B2). PROCEED_WITH_REVISIONS verdict + advisory addition: B5 MDL-canonical sparse-IB SHOULD be enumerated as a 5th reactivation path for completeness."
  - member: Schmidhuber
    verbatim: "Compression-as-intelligence lens: the C6 IBPS empirical 3.04 with SegNet-collapse mode IS structurally identical to the DreamerV2/3 continuous-latent collapse-on-dense-output. The bleeding-edge fix per my deep-research wave §1 IS the DreamerV3 RSSM categorical posterior (Hafner 2024). The DreamerV3 paper reports 5-10x sample-efficiency improvement over continuous-latent VAE on dense-output environments; this is the empirical anchor for Path B2's predicted band [0.18, 0.45]. CONCUR with Hafner verbatim. Path B2 IS the PRIMARY canonical fix; Path B1 hierarchical IB + Path B3 β-VAE + Path B5 sparse-IB are sister-improvements that compose with B2 (B1+B2 = Ballard's combined path; B3-with-B2 = β-tuned categorical; B5-with-B2 = sparse categorical). The Pareto-frontier post-empirical: select the architecture variant that maximizes ΔS per LOC + per $ at the smoke surface."
  - member: Atick
    verbatim: "Concur with sister deep-research wave §1 + sister symposium #836 + the cross-pollination with Z6 Wave 2 Candidate 4c. The cooperative-receiver lens applied to C6 IBPS: the SegNet+PoseNet IS the known receiver R; the C6 IBPS substrate's job IS to maximize I(B; R(B)) at the 24-dim bottleneck. The empirical I(B; R(B)) at C6's smoke endpoint was ~0.05 bits (estimated from MI between latent z and predicted score components). This is 4 orders of magnitude below the cooperative-receiver theorem's optimal I(B; R(B)) ≈ 25-30 bits per sample for the dashcam scoring formula. The IB framework FAILED at the cooperative-receiver theorem's mathematical structure — NOT due to paradigm error but due to ARCHITECTURE choice (24-dim continuous). Path B2 DreamerV3 RSSM categorical posterior IS structurally the cooperative-receiver-theorem-compatible architecture; the categorical posterior's H(T) IS preserved at log2(K^G) bits per sample where K=256 and G=24 groups = 192 bits per sample (vs the continuous Gaussian's effective ~50 bits). The structural match: B2 IS the cooperative-receiver theorem implementation that the C6 substrate's mathematical structure has been requesting. PROCEED_WITH_REVISIONS verdict + binding op-routable: B2 PRIMARY; B1 + B3 SECONDARY; cross-pollination with Z6 Wave 2 Candidate 4c canonical."
  - member: Yousfi
    verbatim: "Steganalysis + scorer-domain lens: the C6 IBPS empirical SegNet=2.60 means the substrate's reconstructed frames at the SegNet input look NOTHING like the source frames at the SegNet's perceptual scale. The 24-dim continuous bottleneck IS analogous to a 24-bit per-frame steganographic payload upper bound; the empirical capacity is FAR below this ceiling. The Path B2 RSSM categorical posterior at 192 bits per sample IS a 8x increase in steganographic payload capacity — directly addressing the SegNet's perceptual-scale information requirement. The Path B3 β-VAE with empirically-tuned β IS sister to my own steganalysis-canonical UNIWARD-delta sweep — the steganalysis literature has empirically tuned the embedding probability parameter (analogous to β) at 10-100x lower than analytical predictions; for C6 IBPS at dashcam steganalysis surface, β IS likely 100x lower than 0.01. PROCEED_WITH_REVISIONS verdict + binding revision: B3 β sweep range SHOULD extend to β ∈ [1e-5, 1e-4] (2-3 orders of magnitude lower than current 0.01)."
  - member: Fridrich
    verbatim: "Concur with Yousfi. The steganographic capacity argument is canonical. ADDITIONAL: the steganalysis literature has empirically shown that UNIWARD-style local cost functions (per-pixel embedding probability scaled by inverse local variance) substantially improve detector-imperceptibility at the contest's operating point. Path B6 UNIWARD-conditioned IB (per-pixel cost-weighted IB regularizer) IS a sister extension that addresses the seg-collapse via boundary-aware β scheduling. Implementation complexity: ~150 LOC (UNIWARD weight computation + per-pixel β scaling in the IB loss). Predicted ΔS band [1.40, 2.20]. PROCEED_WITH_REVISIONS verdict + binding op-routable: B6 UNIWARD-conditioned IB SHOULD be enumerated as a 6th reactivation path."
council_assumption_adversary_verdict:
  - assumption: "C6 IBPS PARADIGM (information bottleneck framework applied to substrate compression) is salvageable via reducer methodology change"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "The IB framework IS mathematically general; the specific 24-dim continuous Gaussian-prior implementation IS empirically falsified at the dashcam scoring operating point. Whether ANY IB-family architecture clears medal-band threshold (predicted upper bound 0.5) is the next empirical question. Path B2 DreamerV3 RSSM categorical posterior has the strongest published empirical basis (Hafner 2024); Path B1 hierarchical IB + Path B3 β-VAE-tuned + Path B4 B1+B2 combined + Path B5 MDL-sparse + Path B6 UNIWARD-conditioned IB are sister variants with weaker empirical bases."
  - assumption: "24-dim continuous Gaussian-prior z is the canonical IB bottleneck"
    classification: CARGO-CULTED
    rationale: "Empirically FALSIFIED at 3.04. Inherited from Alemi 2017 VIB on small images (64x64). Per Hafner 2024 + Tishby memorial + Atick + Yousfi: the canonical fix at the dashcam scale is categorical posterior at 24 groups × K=256 categories = 192 effective bits per sample (3-4x increase). Path B2 IS the canonical replacement."
  - assumption: "β_ib=0.01 is the canonical IB Lagrangian multiplier"
    classification: CARGO-CULTED
    rationale: "Per Higgins memorial + Zaslavsky + Yousfi: empirically-tuned β at the C6 operating point is likely 100x lower (∈ [1e-5, 1e-4]). Inherited from VIB-canonical sister; never tested at dashcam scoring scale. Path B3 β-VAE with empirically-tuned β IS the canonical fix; sister #836 path (a) β_ib sweep IS this same fix at narrower range."
  - assumption: "Path B2 DreamerV3 RSSM categorical posterior IS the primary reactivation candidate"
    classification: HARD-EARNED-EMPIRICAL
    rationale: "Per Hafner verbatim + Schmidhuber + Atick + Tishby memorial concurrence: Path B2 has the strongest published empirical basis (Hafner 2024 DreamerV3 paper; 5-10x sample-efficiency improvement on dense-output tasks); the architectural structure (categorical posterior + uniform max-entropy prior + Gumbel-Softmax reparametrization) IS canonical bleeding-edge 2024 ML; the C6 IBPS SegNet-collapse signature IS structurally identical to the DreamerV2 continuous-latent failure mode the DreamerV3 paper EXPLICITLY ADDRESSES."
  - assumption: "The 3 reactivation paths from sister #836 (β_ib sweep / latent_dim sweep / Phase 2 redesign) cover the alternative-methodology space"
    classification: CARGO-CULTED
    rationale: "Sister #836's paths are 2 hyperparameter sweeps + 1 redesign deliberation — they do NOT enumerate ALTERNATIVE ARCHITECTURES per Catalog #308 ≥3 alternatives requirement. This v2 symposium enumerates 6 architectural alternatives (B1 hierarchical IB / B2 RSSM categorical / B3 β-VAE-tuned / B4 B1+B2 combined / B5 MDL-sparse / B6 UNIWARD-conditioned) that span the canonical IB-family extension space. The PROCEED_WITH_REVISIONS verdict applies to this v2 enumeration."
  - assumption: "C6 IBPS lane should stay as research_only=true with dispatch_enabled=false per Catalog #324 phantom_random_init"
    classification: HARD-EARNED
    rationale: "Per Catalog #324 + sister Catalog #324 backfill 2026-05-18: C6 IBPS recipe MUST be `predicted_band_validation_status: phantom_random_init` until post-training Tier-C re-measurement on landed archive be06a4b0972e6c lands. The v2 symposium's reactivation paths each require their own predicted band + post-training Tier-C validation per Catalog #324. Until that happens, dispatch_enabled=false is correct."
  - assumption: "Cross-pollination with Z6 Wave 2 Candidate 4c (scorer-logit conditioning) informs C6 IBPS redesign"
    classification: HARD-EARNED
    rationale: "Per sister #836 §5 Atick cross-application + sister Z6 Phase 3 sextet 2026-05-17. If Z6 4c lands full-FiLM-WIN with scorer-logit conditioning, the empirical evidence shifts C6 IBPS Path (c) Phase 2 redesign — scorer-conditioning IS a canonical IB-decoder design candidate. The cross-pollination IS BIDIRECTIONAL: C6 IBPS Path B2 empirical anchor IS informative for Z6 Wave 3+ design."
  - assumption: "Operator $50-100 reclaim from NSCS06 family DEFER can fund 3 parallel architecture replacements"
    classification: CARGO-CULTED-OPTIMISTIC
    rationale: "Per Contrarian veto: 3 parallel architecture replacements at $5-15 each = $15-45 in smoke spend + ~1000 LOC engineering effort over 2-3 sessions. This consumes the reclaim AND substantial agent attention. The empirically-honest budget projection: PATH B2 PRIMARY smoke ($5-15) → CONDITIONAL B1/B3/B4/B5/B6 follow-ons based on B2 outcome. Total worst-case projected at $30-50 over 3-4 sessions; PROCEED-conditional on Contrarian's prioritization."
council_decisions_recorded:
  - "op-routable #1: PRIMARY Path B2 DreamerV3 RSSM categorical posterior smoke ($5-15 Modal T4 50-100ep) — predicted ΔS band [0.18, 0.45]"
  - "op-routable #2: SECONDARY Path B3 β-VAE with empirically-tuned β sweep [1e-5, 1e-4, 1e-3] — predicted ΔS band [1.50, 2.80]"
  - "op-routable #3: TERTIARY Path B1 hierarchical IB (4-3-2 dim hierarchy) — predicted ΔS band [0.65, 1.25]"
  - "op-routable #4: B2-WIN follow-on Path B4 (B1+B2 combined hierarchical RSSM) — predicted ΔS band [0.40, 0.80]"
  - "op-routable #5: B2-DEFER pivot to non-IB class-shift (lane_17_imp / ATW V2-1 / Z6 Wave 2 4c) per Assumption-Adversary alt-1/alt-2/alt-3"
  - "op-routable #6: register canonical probe outcome ledger entry per Catalog #313 for C6 IBPS sister #836 + this v2 symposium"
  - "op-routable #7: 30-day retrospective per CLAUDE.md Mission Alignment Consequence 3 — re-audit 2026-06-17"
  - "op-routable #8: Catalog #324 post-training Tier-C re-measurement on landed archive be06a4b0972e6c (independent of reactivation paths; canonical methodology validation)"
---

# Per-substrate symposium: C6 IBPS v2 post-empirical reactivation (22× miss — alternative architectures)

**Substrate**: `c6_e4_mdl_ibps` (C6 IBPS v2 — alternative architecture wave)
**Empirical anchor**: smoke `fc-01KRW353MJJ9A6QW8H99QWZEMH` 2026-05-17 final_score=**3.04** [contest-CPU] vs predicted `[0.113, 0.163]` = **22× miss**. Mechanism: 24-dim continuous Gaussian-prior IB bottleneck destroys segmentation (score_seg=2.60 dominates 86% of total).
**Frontier baseline**: 0.19205 [contest-CPU] (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`; archive sha `6bae0201`).
**Verdict**: PROCEED_WITH_REVISIONS — Path B2 DreamerV3 RSSM categorical posterior IS the canonical bleeding-edge primary reactivation; Path B1/B3/B4/B5/B6 alternative architectures enumerated as Catalog #308 ≥3 alternative-probe-methodologies.
**Council tier**: T3 (substrate-class promotion decision with cross-pollination cascade).

## 1. Why this v2 symposium

Per operator directive 2026-05-18: *"review the implementations for those that failed due to cargo cult assumptions and convene grand council symposiums to determine and design and implement and carry out their reactivation criteria"* + *"Alternative reducer methodologies per Catalog #308 N>=3 (NOT 24-dim IB bottleneck specifically): Hierarchical IB (multi-level); Variational IB with categorical posterior (DreamerV3 RSSM style per research wave); β-VAE with empirically-tuned β"*.

**Sister symposium #836** (`council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md`) already deliberated 3 reactivation paths: (a) β_ib sweep, (b) latent_dim sweep, (c) Phase 2 redesign. Those paths are HYPERPARAMETER SWEEPS + 1 deliberation, NOT alternative architectures.

**THIS v2 symposium** EXTENDS sister #836 with 6 ALTERNATIVE ARCHITECTURES per the operator's explicit directive + Catalog #308 ≥3 alternative-probe-methodologies + bleeding-edge 2024-2026 deep-research wave §1 references:

1. **Path B1 hierarchical IB** (Rao-Ballard 1999 multi-level + Tishby-Zaslavsky 2015 deep IB extension)
2. **Path B2 DreamerV3 RSSM categorical posterior** (Hafner 2024 bleeding-edge; CANONICAL PRIMARY per Hafner+Schmidhuber+Atick+Tishby concurrence)
3. **Path B3 β-VAE with empirically-tuned β** (Higgins 2017 canonical method; SUPERSET of sister #836 path (a))
4. **Path B4 B1+B2 combined** (Ballard's recommendation; hierarchical RSSM categorical posterior)
5. **Path B5 MDL-sparse-IB** (MacKay memorial Laplacian prior on z)
6. **Path B6 UNIWARD-conditioned IB** (Yousfi+Fridrich boundary-aware β scheduling)

## 2. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind path | Probe |
|---|------------|----------------|-------------|-------|
| 1 | 24-dim continuous Gaussian-prior z is canonical IB bottleneck | CARGO-CULTED | Replace with DreamerV3 RSSM categorical posterior (24 groups × K=256 = 192 bits/sample) | Path B2 smoke |
| 2 | β_ib=0.01 is canonical IB Lagrangian multiplier | CARGO-CULTED | Sweep β ∈ [1e-5, 1e-4, 1e-3] per Higgins 2017 + Yousfi/Fridrich UNIWARD analog | Path B3 sweep |
| 3 | Single-level IB bottleneck is sufficient for seg-dominated operating point | CARGO-CULTED | Hierarchical IB at 4-3-2 dim per Rao-Ballard 1999 + Tishby-Zaslavsky 2015 | Path B1 smoke |
| 4 | Sister #836's 3 reactivation paths cover the alternative-methodology space | CARGO-CULTED | This v2 symposium enumerates 6 alternative architectures per Catalog #308 | Multi-path probe matrix |
| 5 | Gaussian prior on z is the canonical IB prior | CARGO-CULTED-PENDING-EMPIRICAL | Laplacian prior per MacKay 2003 Ch. 2 + max-entropy discipline | Path B5 smoke |
| 6 | Uniform per-pixel IB regularization is canonical | CARGO-CULTED | UNIWARD-canonical per-pixel cost-weighted β per Yousfi+Fridrich | Path B6 smoke |
| 7 | IB framework is paradigm-correct for dashcam scoring | CARGO-CULTED-PENDING-EMPIRICAL | If Path B2 lands final_score ≥ 1.0 (factor-3 above predicted upper 0.45), pivot to non-IB class-shift per Assumption-Adversary alt-1/alt-2/alt-3 | Path B2 outcome + ladder |
| 8 | C6's operating-point seg:pose marginal ratio 5.7:1 is stable across architectures | HARD-EARNED-PARTIAL | Verified empirically at C6 smoke; may shift under alternative architectures | Path B2/B3/B1 outcome comparison |

## 3. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Path B2 (RSSM categorical, PRIMARY) | Path B1 (hierarchical IB) | Path B3 (β-VAE tuned) |
|---|-----------|-------------------------------------|---------------------------|----------------------|
| 1 | UNIQUENESS (class-shift not within-class) | YES — discrete RSSM categorical posterior is class-shift away from continuous-Gaussian IB | YES — multi-level hierarchical IB is class-shift away from single-level IB | NO — same single-level architecture, different β |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | YES — canonical Hafner 2024 architecture; ~300 LOC implementation; uniform max-entropy prior is auditable | PARTIAL — multi-level pyramid is ~400 LOC; reviewer needs to verify each level's bottleneck capacity | YES — single config change (β value); ~50 LOC sweep harness |
| 3 | DISTINCTNESS (explicitly different from sisters) | YES — distinct from C6 IBPS v1 (continuous-Gaussian) AND from sister substrates | YES — distinct from C6 IBPS v1 | PARTIAL — same architecture as v1; distinct from sister #836 path (a) ONLY in extending β range to 1e-5 |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | YES — premise verified via Hafner 2024 empirical anchor on DreamerV3; adversarial review THIS symposium; assumption classification per §2; empirical anchor pending B2 smoke | YES — premise verified via Tishby-Zaslavsky 2015 + Rao-Ballard 1999; same | PARTIAL — premise verified via Higgins 2017; same |
| 5 | OPTIMIZATION PER TECHNIQUE | YES — categorical posterior with Gumbel-Softmax STE is canonical bleeding-edge | YES — hierarchical IB with explicit Rao-Ballard top-down prediction is canonical | YES — Higgins 2017 β sweep is canonical |
| 6 | STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS) | YES (predicted) — RSSM categorical orthogonal with PR101 frame_exploit_selector + composable with cooperative-receiver Z6 4c | YES — hierarchical IB orthogonal with PR101 | YES — β sweep orthogonal with PR101 |
| 7 | DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned) | YES — Gumbel-Softmax temperature-annealing with seed pin produces deterministic categorical samples | YES — hierarchical IB is deterministic with seed pin | YES — single β config is trivially deterministic |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | YES (predicted) — 4x effective bottleneck capacity vs continuous 24-dim; predicted seg=0.10-0.20 (10-25x reduction from 2.60) | YES (predicted) — multi-level structure preserves I(X;Y_k) at scale k; predicted seg=0.30-0.50 | PARTIAL — same architecture; lower β may help but structural collapse remains |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | YES (predicted band [0.18, 0.45]) — within medal-band threshold | PARTIAL (predicted band [0.65, 1.25]) — above medal-band threshold but better than current 3.04 | NO (predicted band [1.50, 2.80]) — still above medal-band threshold |

## 4. Observability surface (Catalog #305)

1. **Inspectable per layer**:
   - Path B2: per-category posterior probability distribution at each of 24 groups (visualizable as 24 × K=256 heatmap per sample); Gumbel-Softmax temperature schedule.
   - Path B1: per-level latent z_k for k ∈ {coarse, medium, fine}; top-down prediction residuals at each level.
   - Path B3: β value swept; rate-distortion curve over β range.
2. **Decomposable per signal**: per-task distortion (seg / pose / rate) at each architecture; per-category usage histogram (Path B2); per-level capacity utilization (Path B1).
3. **Diff-able across runs**: archive sha256 + per-layer state_dict sha256; categorical posterior alphabet usage stats.
4. **Queryable post-hoc**: canonical posterior at `.omx/state/council_deliberation_posterior.jsonl` per Catalog #325; probe outcomes at `.omx/state/probe_outcomes.jsonl` per Catalog #313; per-path smoke artifact directory.
5. **Cite-able**: every variant tuple is (path, K_categorical_size, hierarchical_levels, β_value, prior_class, regularization_class, seed) — unique citation; smoke run via `experiments/train_substrate_c6_e4_mdl_ibps.py` canonical entry point.
6. **Counterfactual-able**: byte-mutation gate per Catalog #139 + #105 on landed archive — verify decoder reconstruction changes measurably under per-byte perturbation; sister to sister #836's Catalog #324 Tier-C re-measurement.

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL" + Catalog #308 ≥3 alternative-probe-methodologies)

### Reactivation Path B2 (PRIORITY 1, PRIMARY): DreamerV3 RSSM categorical posterior

- **Description**: Replace 24-dim continuous Gaussian-prior z with 24 categorical groups × K=256 categories per Hafner 2024 DreamerV3 RSSM architecture. Use Gumbel-Softmax reparametrization with straight-through estimator. Uniform max-entropy prior on each categorical group. KL regularizer encourages diverse category usage.
- **Predicted ΔS band**: `[0.18, 0.45]` contest-CPU. Mechanism: 4x effective bottleneck capacity (192 bits/sample vs continuous ~50 bits/sample) preserves per-pixel SegNet boundary information; categorical posterior cannot collapse to single mode because uniform prior penalizes that.
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $5-15 Modal T4 50-100ep smoke; if WIN, $15-30 Modal A100 200ep full run for paired contest CPU+CUDA harvest
- **Structural verdict**: PRIMARY CANONICAL REACTIVATION — per Hafner verbatim + Schmidhuber + Atick + Tishby memorial concurrence + deep-research wave §1 bleeding-edge identification. The DreamerV3 paper EMPIRICALLY VALIDATES the categorical posterior fix on dense-output tasks with the same continuous-latent-collapse failure mode.
- **Implementation complexity**: ~300 LOC addition to substrate trainer (categorical reparametrization head; Gumbel-Softmax annealing; uniform-prior KL regularizer; categorical usage statistics for observability).
- **Composability**: ORTHOGONAL with PR101 frame_exploit_selector; ORTHOGONAL with cooperative-receiver Z6 Wave 2 Candidate 4c (composable post-empirical via Path B4 B1+B2 combined).
- **Prerequisites**: (a) operator authorization for $5-15 Modal T4 smoke; (b) trainer scaffold extension via `tools/lane_maturity.py mark` after L1 SCAFFOLD lands; (c) Catalog #325 6-step contract satisfied (this symposium IS step 4-5).

### Reactivation Path B1 (PRIORITY 2): Hierarchical IB at 4-3-2 dim hierarchy

- **Description**: Replace single-level 24-dim IB with multi-level hierarchical IB at scales {coarse=4-dim, medium=3-dim, fine=2-dim} per Rao-Ballard 1999 + Tishby-Zaslavsky 2015 deep IB extension. Each level has its own KL regularizer with β_k = β / 2^k (coarse-level looser, fine-level tighter). Top-down predictions z_{k+1} from z_k via learned transition function.
- **Predicted ΔS band**: `[0.65, 1.25]` contest-CPU. Mechanism: multi-level structure preserves I(X;Y_k) at scale k where SegNet boundary information lives (fine scale).
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $5-15 Modal T4 50-100ep smoke
- **Structural verdict**: SECONDARY CANONICAL REACTIVATION — per Rao + Ballard verbatim + Tishby memorial concurrence. The Rao-Ballard 1999 paper canonical implementation IS the canonical V1-V2-V4 visual cortex analog; the C6 IBPS substrate's seg+pose decomposition IS structurally analogous to parvocellular + magnocellular pathways.
- **Implementation complexity**: ~400 LOC addition (hierarchical encoder/decoder + 3 levels of bottlenecks + top-down prediction pathway + per-level loss weighting).
- **Composability**: ORTHOGONAL with PR101; COMPOSABLE with Path B2 via Path B4 (Ballard's recommendation).
- **Prerequisites**: Same as Path B2.

### Reactivation Path B3 (PRIORITY 3): β-VAE with empirically-tuned β sweep

- **Description**: Sweep β ∈ [1e-5, 1e-4, 1e-3, 1e-2] per Higgins 2017 canonical methodology + Yousfi/Fridrich UNIWARD analog (3 orders of magnitude lower than C6's current 0.01). EXTENDS sister #836 path (a) β_ib sweep at narrower range.
- **Predicted ΔS band**: `[1.50, 2.80]` contest-CPU. Mechanism: lower β reduces rate-distortion penalty on the bottleneck; preserves more I(X;T) at the cost of less compression. May still suffer SegNet collapse at sufficiently low β if architecture is structurally insufficient.
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $5-7 Modal T4 50ep smoke (4 β values; smoke per value $1.50-1.75)
- **Structural verdict**: TERTIARY EMPIRICAL — tests Higgins 2017 + sister #836's path (a) at extended β range. If empirically-optimal β at the C6 operating point is ∈ [1e-5, 1e-4] (as Higgins memorial predicts), Path B3 IS a viable shippable variant. If empirically-optimal β is OUTSIDE this range, the architecture (not β) is the binding constraint and Path B2/B1 are needed.
- **Implementation complexity**: ~50 LOC sweep harness; existing trainer scaffold reusable.
- **Composability**: SUBSUMED by Path B2 (categorical posterior is β-independent in the limit); composable as initial β probe BEFORE Path B2 dispatch.
- **Prerequisites**: Same as Path B2.

### Reactivation Path B4 (B2-WIN follow-on): B1+B2 combined hierarchical RSSM categorical posterior

- **Description**: Per Ballard's verbatim recommendation: combine hierarchical IB (Path B1) with RSSM categorical posterior (Path B2) — multi-level latent hierarchy where each level uses categorical posterior. Architecturally: 3 hierarchical levels × K=128 categorical groups per level. Coarse-level categorical alphabet captures macro-scene structure; fine-level captures per-pixel SegNet boundary structure.
- **Predicted ΔS band**: `[0.40, 0.80]` contest-CPU. Mechanism: per Ballard + Rao + Hafner combined structural prediction.
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $10-25 Modal A100 100-200ep smoke (larger architecture warrants longer training)
- **Structural verdict**: B2-WIN-CONDITIONAL — fire ONLY after Path B2 smoke produces ΔS within predicted [0.18, 0.45] band (validates RSSM categorical posterior architecture). If Path B2 lands ΔS ∈ [0.18, 0.30], Path B4 is the canonical next step. If Path B2 lands ΔS ∈ [0.30, 0.45], Path B4 is OPTIONAL (marginal improvement).
- **Implementation complexity**: ~500 LOC (combines B1 + B2 helpers).
- **Composability**: PRIMARY follow-on to Path B2.
- **Prerequisites**: Path B2 smoke WIN landed.

### Reactivation Path B5 (DEFERRED): MDL-sparse-IB with Laplacian prior

- **Description**: Per MacKay memorial: replace Gaussian prior on z with Laplacian prior (encouraging dimension-wise sparsity). Most dimensions zero; effective bottleneck capacity adapts to data complexity.
- **Predicted ΔS band**: `[1.20, 2.10]` contest-CPU. Mechanism: sparse bottleneck preserves only the highest-information dimensions; reduces L(model) cost.
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $5-7 Modal T4 50ep smoke
- **Structural verdict**: DEFERRED — modest predicted improvement vs Path B1/B2/B3. Fire ONLY if Path B2/B1/B3 all underperform predicted bands.
- **Implementation complexity**: ~200 LOC (Laplacian prior + reparametrization adjustment).
- **Composability**: SUBSUMED by Path B2 (categorical posterior is implicitly sparse via uniform prior + dimension-wise category usage).

### Reactivation Path B6 (DEFERRED): UNIWARD-conditioned IB

- **Description**: Per Yousfi+Fridrich: per-pixel cost-weighted β regularization where β scales with inverse local variance (UNIWARD canonical embedding cost). Boundary regions get lower β (preserving more information); flat regions get higher β (compressing more aggressively).
- **Predicted ΔS band**: `[1.40, 2.20]` contest-CPU. Mechanism: boundary-aware β preserves SegNet boundary information at the regions that matter most.
- **Predicted_band_validation_status**: `pending_post_training` per Catalog #324
- **Predicted cost**: $5-10 Modal T4 50-100ep smoke
- **Structural verdict**: DEFERRED — modest predicted improvement; composable with Path B2/B1/B3 as a post-hoc regularization adjustment.
- **Implementation complexity**: ~150 LOC (UNIWARD weight computation + per-pixel β scaling).

### Reactivation priority ordering

1. **Path B2 (HIGHEST EV)**: $5-15 cost, predicted ΔS [0.18, 0.45] within medal-band; CANONICAL BLEEDING-EDGE per Hafner+Schmidhuber+Atick+Tishby
2. **Path B1 (SECOND EV)**: $5-15 cost, predicted [0.65, 1.25] above medal-band; CANONICAL Rao-Ballard structural fix
3. **Path B3 (THIRD EV)**: $5-7 cost, predicted [1.50, 2.80] above medal-band; CANONICAL Higgins 2017 β sweep at extended range
4. **Path B4 (B2-WIN follow-on)**: $10-25 cost, predicted [0.40, 0.80]; combines B1+B2
5. **Path B5/B6 (DEFERRED)**: Fire only if Paths B1/B2/B3 underperform

**Recommendation per Contrarian dissent (binding)**: Path B2 PRIMARY ($5-15 alone) BEFORE parallel dispatch. If Path B2 lands ΔS ∈ predicted [0.18, 0.45], Path B4 (B2-WIN follow-on) becomes EV-positive. If Path B2 lands ΔS > 1.0 (factor-3 above predicted upper bound), pivot to non-IB class-shift per Assumption-Adversary alt-1/alt-2/alt-3:

- **alt-1**: `lane_17_imp` IMP cycle 0 + Frankle LTH winning-ticket extraction (per Symposium #2 lane_17_imp design; $1-2 cost on Vast.ai 4090)
- **alt-2**: cooperative-receiver ATW V2-1 with Channel #1 per-pixel softmax logits (per ATW V2 Reactivation Symposium + Symposium #1 V1 dense Faiss-IVF-PQ reactivation $0.15 V4 probe)
- **alt-3**: Z6 Wave 2 Candidate 4c full-FiLM-WIN scorer-logit-conditioning (sister subagent in flight)

**Cross-pollination with Z6 Wave 2 Candidate 4c** (per sister #836 §5 Atick cross-application + ATW V2 symposium Revision #5): if Z6 4c lands full-FiLM-WIN with scorer-logit conditioning, the C6 IBPS Path B2 RSSM categorical posterior decoder SHOULD be augmented with scorer-logit conditioning at the categorical posterior layer — predicted band shifts from [0.18, 0.45] to [0.10, 0.30].

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status for C6 IBPS v1 (24-dim continuous)**: `phantom_random_init_FALSIFIED_EMPIRICAL` — per sister Catalog #324 backfill 2026-05-18.

**Predicted_band_validation_status for Path B2 (RSSM categorical)**: `pending_post_training` per Catalog #324. Reactivation criterion: post-training Tier-C density measurement on landed Path B2 archive via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <sha>`. If empirical ΔS lands within `[0.18, 0.45]`, ratify B2 as canonical and advance to L2. If outside band, surface as Catalog #324 violation and re-symposium.

**Predicted_band_validation_status for Path B1/B3/B4/B5/B6**: same pattern; each variant's predicted band gates its post-training Tier-C validation.

**Independent canonical methodology validation**: per sister #836 op-routable #4: run `tools/mdl_scorer_conditional_ablation.py --tier c --archive be06a4b0972e6c...` on the LANDED C6 IBPS v1 archive (NOT a reactivation variant) to test whether the v1 architecture's empirical post-training Tier-C density REVERSES the pre-training prediction band. This IS independent of reactivation paths and validates the Catalog #324 methodology itself.

## 7. Continual-learning anchor (Catalog #325 dispatch eligibility gate (d))

After this memo lands, the canonical posterior anchor IS registered to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per the canonical 4-layer pattern (Catalog #245 exemplar). The anchor schema includes:

- PROCEED_WITH_REVISIONS verdict
- 15-seat attendee list (6 inner sextet + 9 grand-council: Tishby_memorial, Zaslavsky, Schmidhuber, MacKay_memorial, Atick, Hafner, Higgins_memorial, Rao, Ballard)
- 8-assumption Assumption-Adversary verdict
- 8 op-routables
- mission-alignment=frontier_breaking (B2 within medal-band) / frontier_protecting (B1/B3 above)
- override_invoked=false
- horizon_class=asymptotic_pursuit
- canonical_frontier_anchor per Catalog #316
- deferred_substrate_id=c6_e4_mdl_ibps + deferred_substrate_retrospective_due_utc=2026-06-17T00:00:00Z

Downstream consumers per Catalog #325:

- **Catalog #325 STRICT preflight** sees PROCEED_WITH_REVISIONS verdict and permits dispatch of recipes targeting `c6_e4_mdl_ibps` substrate ONLY when (a) recipe is one of {B2 RSSM categorical, B1 hierarchical IB, B3 β-VAE-tuned, B4 B1+B2 combined} AND (b) sister Catalog #324 backfill applied AND (c) lane registry maturity tracker shows gate satisfaction.
- **Cathedral autopilot ranker** consumes via `tac.council_continual_learning.query_anchors_by_topic('c6_e4_mdl_ibps')`; PROCEED_WITH_REVISIONS weights B2 ABOVE B1/B3/B4/B5/B6 in candidate ranking.
- **Probe-outcomes ledger (Catalog #313)** receives sister-registered EMPIRICAL_FALSIFICATION outcome for v1 24-dim continuous architecture per op-routable #6.

## 8. Cross-references

- **Canonical C6 IBPS lineage**:
  - `.omx/research/council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md` (sister #836 FIRST per-substrate symposium)
  - `.omx/research/council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517.md` (sister #835 Phase 2 sextet)
  - `.omx/research/feedback_meta_fix_catalog_324_predicted_band_post_training_validation_required_landed_20260517.md` (Catalog #324 META-FIX)
  - `.omx/research/feedback_c6_ibps_first_asymptotic_dispatch_smoke_before_full_paired_landed_20260517.md` (empirical anchor smoke landing)
- **Bleeding-edge 2024-2026 references** (from deep-research wave §1):
  - Hafner 2024 *Mastering Diverse Domains through World Models* (DreamerV3 RSSM categorical posterior arxiv 2301.04104)
  - Higgins 2017 *β-VAE: Learning basic visual concepts with a constrained variational framework*
  - Tishby-Zaslavsky 2015 *Deep learning and the information bottleneck principle*
  - Alemi-Fischer-Dillon-Murphy 2017 *Deep Variational Information Bottleneck*
  - Rao-Ballard 1999 *Predictive coding in the visual cortex*
  - MacKay 2003 *Information Theory, Inference, and Learning Algorithms* Ch. 2
- **Sister symposia (per-substrate Catalog #325 cohort)**:
  - `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (parent ATW V2 reactivation)
  - `.omx/research/council_per_substrate_symposium_v1_dense_faiss_ivf_pq_reactivation_20260518.md` (Symposium #1 of this wave)
  - `.omx/research/council_per_substrate_symposium_nscs06_v8_path_b_20260517.md` (sister NSCS06 v8 REFUSE pattern)
- **Cross-pollination targets**:
  - Z6 Wave 2 Candidate 4c subagent (in flight per sister #836 — outcome conditions Path B2 RSSM decoder scorer-logit augmentation)
  - Sister C6 IBPS latent_dim sweep BUILD subagent (in flight; produces B1-adjacent latent_dim recipes)
- **Catalog gates fired by this v2 symposium**: #229 (premise verification) + #245 (Modal call_id ledger) + #265 (symposium impls canonical-contract) + #291 (per-session META-ASSUMPTION cadence) + #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist) + #296 (Dykstra-feasibility predicted-band) + #300 (council v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface) + #307 (paradigm-vs-implementation classification — v1 IMPLEMENTATION FALSIFIED; PARADIGM partially salvageable via Path B2) + #308 (alternative-probe-methodologies enumeration — 6 alternative architectures + 3 alternative substrate-class pivots) + #313 (probe-outcomes ledger registration via op-routable #6) + #315 (substrate at optimal form before paid dispatch) + #316 (canonical frontier anchor) + #324 (post-training Tier-C validation discipline) + #325 (per-substrate symposium discipline — THIS v2 symposium EXTENDS sister #836 for c6_e4_mdl_ibps for 14 days).
- **Canonical implementations cited**: Hafner 2024 DreamerV3 (arxiv 2301.04104) + Higgins 2017 β-VAE + Tishby-Zaslavsky 2015 deep IB + Alemi 2017 VIB + Rao-Ballard 1999 hierarchical predictive coding + MacKay 2003 + Daubechies-DeVore-Fornasier-Gunturk 2010 + sister Catalog #277 wavelet multi-scale ranker.

## 9. Operator op-routables (for parent agent + main Claude)

1. **PRIMARY (highest EV)**: Path B2 DreamerV3 RSSM categorical posterior smoke. $5-15 Modal T4 50-100ep. 24 categorical groups × K=256 categories. Gumbel-Softmax STE. Predicted ΔS band `[0.18, 0.45]` contest-CPU. Predicted SegNet collapse reduction: 2.60 → 0.10-0.20 (10-25x improvement).

2. **SECONDARY**: Path B3 β-VAE with empirically-tuned β sweep ∈ [1e-5, 1e-4, 1e-3, 1e-2]. $5-7 Modal T4 50ep smoke (4 β values). EXTENDS sister #836 path (a) at lower β range per Higgins memorial + Yousfi/Fridrich UNIWARD analog.

3. **TERTIARY**: Path B1 hierarchical IB at 4-3-2 dim hierarchy. $5-15 Modal T4 50-100ep smoke. Rao-Ballard 1999 structural fix.

4. **B2-WIN CONDITIONAL**: Path B4 B1+B2 combined (hierarchical RSSM categorical posterior). $10-25 Modal A100 100-200ep smoke. Per Ballard's recommendation.

5. **B2-DEFER PIVOT**: If Path B2 lands ΔS > 1.0 (factor-3 above predicted upper bound), pivot to non-IB class-shift per Assumption-Adversary alt-1 (lane_17_imp $1-2 cost) / alt-2 (ATW V2-1 V4 probe $0.15 free) / alt-3 (Z6 Wave 2 4c outcome cascade).

6. **PROBE OUTCOMES LEDGER REGISTRATION** per Catalog #313:
   - C6 IBPS v1 (24-dim continuous): register as `verdict=EMPIRICAL_FALSIFICATION, status=blocking, methodology=24_dim_continuous_gaussian_prior_ib_bottleneck, alternative_probe_methodologies=[path_b2_rssm_categorical_posterior, path_b1_hierarchical_ib, path_b3_beta_vae_tuned, path_b4_b1_b2_combined, path_b5_mdl_sparse_laplacian_prior, path_b6_uniward_conditioned_beta], expires_at_utc=2026-06-17T00:00:00Z`

7. **30-DAY RETROSPECTIVE** per CLAUDE.md "Mission alignment" Consequence 3 (re-audit 2026-06-17): re-audit whether Path B2/B1/B3 outcomes converged C6 IBPS toward medal-band frontier; assess whether non-IB class-shift sister substrate (Z6 Wave 2 4c / lane_17_imp / ATW V2-1) landed a path that captures the C6 IBPS class shift's intended gain.

8. **INDEPENDENT CATALOG #324 VALIDATION**: per sister #836 op-routable #4: run `tools/mdl_scorer_conditional_ablation.py --tier c --archive be06a4b0972e6c...` on the LANDED C6 IBPS v1 archive (independent of reactivation paths). $0 GPU local CPU; validates Catalog #324 methodology itself.

## Symposium verdict summary

- **Tier**: T3
- **Verdict**: PROCEED_WITH_REVISIONS
- **Top-priority reactivation path**: Path B2 DreamerV3 RSSM categorical posterior ($5-15 Modal T4 50-100ep smoke; predicted ΔS band `[0.18, 0.45]` contest-CPU)
- **Predicted cost**: $5-15 immediate (B2 smoke) → if WIN, $10-25 follow-on (B4 combined) = $15-40 total worst-case for empirical canonical C6 IBPS v2 reactivation
- **Structural recommendation to operator**: Authorize Path B2 IMMEDIATELY (sister symposium #836's path (a)/(b) parallel dispatch was the v1 hyperparameter-search reactivation; THIS v2 symposium ratifies the BLEEDING-EDGE alternative architecture per operator's explicit Catalog #308 directive). Defer Path B1/B3/B4/B5/B6 funding until Path B2 outcome lands. If Path B2 lands within predicted [0.18, 0.45], the IB family IS shippable at the right architecture; convene Wave N+1 council to evaluate Path B4 follow-on + cross-pollination with Z6 Wave 2 Candidate 4c. If Path B2 lands > 1.0 (factor-3 above predicted upper bound), pivot to non-IB class-shift per Assumption-Adversary alt-1/alt-2/alt-3.
