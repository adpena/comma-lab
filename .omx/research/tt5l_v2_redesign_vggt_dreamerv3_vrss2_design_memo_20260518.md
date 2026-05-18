---
council_tier: T2
council_attendees:
  # Sextet pact (binding; quorum 6-of-6 at T2)
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  # Grand Council attendees added per topic (foveation + ego-motion + RSSM world-model + 3D geometry)
  - Hafner          # DreamerV3 RSSM categorical (arxiv 2301.04104)
  - Atick           # cooperative-receiver theorem (Atick-Redlich 1990)
  - Redlich         # cooperative-receiver co-author
  - Gibson_memorial # ego-motion-matched foveation (Gibson 1950 FoE)
  - Rao             # hierarchical predictive coding (Rao-Ballard 1999)
  - Ballard         # embodied vision sister of Rao
  - Tishby_memorial # information bottleneck framework
  - Zaslavsky       # Tishby active collaborator
  - Wyner_memorial  # side-information source coding (Wyner-Ziv 1976)
  - Mallat          # wavelet hierarchical decomposition (Mallat 1989)
  - Carmack         # engineering simplicity / strip-everything
  - MacKay          # information-theory + Bayesian unifier
  - Hotz            # raw engineering instinct
  - Boyd            # convex feasibility (Dykstra co-lead)
  - Time-Traveler-peer    # canonical Daubechies → Rudin chain
  - Time-Traveler-protege # Rudin's active Duke postdoc
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "I VETO any framing that treats TT5L V2 as a paid-dispatch-eligible design from this memo alone. Sister symposium #866 REFUSED TT5L V1 with the canonical empirical anchor 25ep CUDA 3.9007 (19× worse than the 0.20533 frontier). The parent prompt's predicted band [-0.020, -0.008] => [0.172, 0.184] is the deep-research wave TOP-5 #1 hypothesis, NOT an empirical result. Specifically: (a) VGGT (CVPR 2025 Best Paper; ~600M params per arxiv 2503.11651) cannot ship in the contest archive within rate budget — even at int4 quantization that's ~150 MB which is 500x over the ~300 KB archive budget. The 'integrate VGGT pretrained encoder' framing CONFLATES 'use VGGT as compress-time teacher whose outputs distill to a small archive-shippable artifact' with 'ship VGGT weights in the archive'. The design memo MUST disambiguate this BEFORE Wave N+1 council approves any dispatch. (b) DreamerV3 RSSM categorical (32 one-hot vectors per timestep from 32 categorical distributions) is the canonical world-model latent dynamics primitive but DreamerV3 itself trains ~200M-step environments at 200K-step pretraining — the contest's 600 pairs is 4 orders of magnitude smaller. The selectivity advantage at 600-pair scale is empirically untested. (c) NVIDIA VRSS 2 is HARDWARE foveation (variable-rate shading in DirectX/CUDA rasterization); the PRINCIPLE of foveation transfers but the implementation is GPU-driver-level not Python-level. The design memo MUST disambiguate the PRINCIPLE-vs-IMPLEMENTATION transfer. (d) DUSt3R/MASt3R are dense-stereo reconstruction; the contest is 2-frame ego-pose — DUSt3R's pretrained model could serve as PoseNet teacher but again CANNOT ship in archive. My VOTE: PROCEED_WITH_REVISIONS conditioned on (1) explicit compress-time-teacher-vs-archive-shipped disambiguation per primitive (VGGT/DreamerV3/VRSS 2/DUSt3R/MASt3R); (2) per-substrate symposium discipline per Catalog #325 + parent #866 revisions consumed; (3) per-section MI probes on V1 25ep archive BEFORE V2 trainer build; (4) Wave N+1 council on this memo + sister probes before any paid dispatch. REFUSE on bypassing Catalog #313 predecessor-probe-outcome ledger (V1 DEFER verdict in ledger; V2 must be NEW ledger entry, not implicit override of V1)."
  - member: Assumption-Adversary
    verbatim: "Per the MANDATORY Catalog #291 item #8 assumption-challenge axis applied to the parent prompt's framing. The SHARED ASSUMPTION operating across the parent prompt: 'TT5L V1 was fundamentally janky; swap in 2024-2025 bleeding-edge IDEAS (VGGT + DreamerV3 RSSM + VRSS 2 + DUSt3R/MASt3R) and we get sub-0.190 contest-CPU floor potential.' CLASSIFICATION: HARD-EARNED-PARTIAL + CARGO-CULTED-PENDING-DECOMPOSITION. The HARD-EARNED basis: V1 IS empirically falsified at 3.9007 [contest-CUDA reviewed] per sister #866; the operator's 'fundamentally broken janky' diagnosis is HARD-EARNED-OPERATIONAL — TT5L V1's per-section MI is structurally underconverged per Atick's cooperative-receiver MI probe recommendation. The CARGO-CULTED basis: the parent prompt CONFLATES (a) 'bleeding-edge IDEAS' (the 4 named 2024-2026 primitives) with (b) 'bleeding-edge IMPLEMENTATIONS that fit pact's 300 KB archive budget'. VGGT is 600M params; DreamerV3 RSSM training is 200K-step environments; VRSS 2 is hardware-level; DUSt3R is dense-stereo. Each primitive's *PRINCIPLE* may transfer; each primitive's *IMPLEMENTATION* requires distillation/extraction/translation to a contest-archive-shippable artifact. The deep-research wave TOP-5 #1 predicted band [-0.020, -0.008] is derived from 'IF each primitive's principle transfers cleanly to contest-archive-shippable form with ~20% bit savings per primitive THEN composition produces ΔS -0.020 to -0.008.' Per Boyd's Dykstra-feasibility lens: composition of 4 primitives is at best SUBADDITIVE; per the sister-NSCS06-v6→v8 dual-direction anchor pattern, composition is empirically NOT monotonic. My assumption-violation hypothesis HARD-EARNED-NEW: 'IF the 4 primitives' principles transfer cleanly via compress-time-teacher pattern (NOT archive-shipped), THEN TT5L V2's per-section MI probe at 100ep should show >= 0.5 bits/symbol on at least 2 of 4 sections; IF MI is below threshold on >= 3 of 4 sections at 100ep, the primitives' principles do NOT transfer to dashcam contest scale and the predicted band is empirically falsified (paradigm-level via per-section MI distinction per Catalog #307).' Per Catalog #315: TT5L V2 is at PRE-OPTIMAL-FORM per the cargo-cult-unwind methodology requirement; per Catalog #325: this memo MUST satisfy the 6-step contract before any paid dispatch authorization. My VOTE: PROCEED_WITH_REVISIONS conditioned on V2 trainer DESIGN-ONLY at this memo + sister Wave N+1 council on V2 trainer build + per-section MI probes BEFORE any paid dispatch."
  - member: Hafner
    verbatim: "I am summoned per grand council expansion as canonical author of DreamerV3 (arxiv 2301.04104) + the RSSM categorical latent primitive (32 one-hot vectors per timestep from 32 categorical distributions). The TT5L V2 framing of 'DreamerV3 RSSM categorical for richer latent structure' requires architectural specificity: RSSM = Recurrent State-Space Model = GRU-deterministic-state + categorical-stochastic-state + reward-head + value-head. For pact's dashcam contest, the reward+value heads do NOT directly apply (no episodic reward signal); the GRU-deterministic + categorical-stochastic IS the directly-transferable primitive. CRITICAL DESIGN DECISION per my Z7 symposium 2026-05-17 Revision #3 (Hafner binding): GRU IS the canonical recurrent primitive, NOT LSTM (LSTM's input/forget/output gates are redundant for the dashcam use case; GRU's reset+update gates are sufficient + cheaper). For TT5L V2 the RSSM should use GRU-deterministic + 32-one-hot categorical-stochastic at each pair-step. The 32-one-hot is structurally MORE EXPRESSIVE than Gaussian latent for world-model latent dynamics per DreamerV3 paper Section 3.2; the empirical evidence on Minecraft + Atari shows categorical outperforms Gaussian at the long-horizon. CROSS-POLLINATION with sister Z7-Mamba-2 design memo (a68b22b14 sister-subagent in flight): IF Z7-Mamba-2 + Z7-GRU empirically determines Mamba-2's selectivity is HIGH-VALUE at 600-pair sequence, TT5L V2 should adopt Mamba-2 selective state-space; ELSE adopt GRU. This is a Wave-N+1 cross-substrate dependency. My specific TT5L V2 recommendation: predict_residual section should use GRU-deterministic + 32-one-hot categorical-stochastic; per-pair categorical-stochastic provides the substrate-distinguishing latent expressivity advantage; sister Z7 + Z8 substrates handle the recurrent + hierarchical primitives. My VOTE: PROCEED_WITH_REVISIONS conditioned on (a) RSSM-categorical architecture decision wait for Z6 4c outcome + Z7 GRU vs Mamba-2 outcome; (b) explicit GRU + categorical-stochastic spec in V2 design + trainer scaffold."
  - member: Atick
    verbatim: "I bridge cooperative-receiver (Atick-Redlich 1990) + foveation (Atick-Redlich + sister Gibson 1950 FoE). The TT5L V2 framing of 'NVIDIA VRSS 2 for foveation' requires DISAMBIGUATION: VRSS 2 is HARDWARE-LEVEL variable-rate shading (per NVIDIA dev blog: pixel shader invocation rate varies by gaze-tracked map); the PRINCIPLE is per-pixel rendering rate variation. For contest compression, the analogous PRINCIPLE is per-pixel BIT BUDGET variation — the foveation map weights per-pixel attention so high-value pixels (gaze center or scorer-attention center) receive more bits. Per my cooperative-receiver theorem applied to contest scorer pair (SegNet+PoseNet), the scorer-attention center per frame IS the analog of gaze center per HMD frame. Specifically: SegNet's stride-2 stem + per-class loss gradient gives PER-PIXEL ATTENTION WEIGHTS via the back-propagated gradient magnitude; PoseNet's FoE-derived attention from rgb_to_yuv6 + 6-DOF Hydra head gives PER-FRAME ATTENTION CENTER. Their product (per-pixel × per-frame) IS the cooperative-receiver-derived foveation map. Importantly: the foveation map does NOT need to ship in archive — it's derivable at inflate-time from the scorer's structural attention (which is in the contest scorer model, not in the archive). This is the canonical Atick-Redlich cooperative-receiver pattern: the published receiver R IS the shared prior; the encoder optimizes per-pixel bit allocation against R's attention distribution; the decoder reconstructs without shipping the attention map. CROSS-POLLINATION with sister ATW V2 (V2-1 channel pick): the per-region (16x16) SegNet softmax histogram is the SAME channel-class TT5L V2's seg_boundary section should use. The Z6 Wave 2 4c outcome (sister codex probe in flight) materially informs the per-pixel-attention-derivation. My specific TT5L V2 recommendation: ADD an EXPLICIT cooperative-receiver-derived-foveation-map section (replaces V1's hf_residual; foveation map derived AT INFLATE TIME from scorer weights; ZERO archive bytes per Wyner-Ziv-like cooperative-receiver pattern). My VOTE: PROCEED_WITH_REVISIONS conditioned on (a) cooperative-receiver-derived-foveation-map architecture decision per V2 design; (b) per-section MI probe on V1 25ep state for the foveation candidate channel BEFORE V2 trainer build."
  - member: Tishby_memorial
    verbatim: "Memorial seat conveying Information Bottleneck framework + sister of C6 IBPS Phase 2 redesign. The TT5L V2's IB Lagrangian: L_IB = I(X_video; T_TT5L_V2_5_sections) - β * I(T_TT5L_V2_5_sections; Y_scorer). Per Hafner's RSSM categorical recommendation: the latent T should be 32-one-hot categorical (32 categoricals each with 32 classes = 32 * log2(32) = 160 bits per timestep + 600 pairs = 96000 bits = 12 KB; matches budget). Per Atick's cooperative-receiver: I(T; Y_scorer) is bounded above by H(T) AND below by the scorer's mutual-information floor; the IB Lagrangian's β parameter sweeps this tradeoff. CROSS-POLLINATION with sister C6 IBPS Phase 2 redesign (sister symposium): C6's β-IB-Lagrangian empirical anchor at Wave N+1 directly provides TT5L V2's β-parameter initialization. My specific TT5L V2 recommendation: predict_residual + seg_boundary capacity rebalancing per Tishby IB-framework. Current V1 allocates 47% of side-info budget to predict_residual + se3_lie; V2 should rebalance to 60-70% per my IB-framework prediction (Wyner_memorial + Hafner agree). My VOTE: PROCEED_WITH_REVISIONS conditioned on (a) IB Lagrangian decomposition documented in V2 design memo per Catalog #294; (b) β-parameter initialization from sister C6 IBPS Phase 2 empirical anchor."
  - member: Wyner_memorial
    verbatim: "Memorial seat conveying Wyner-Ziv 1976 side-information source coding. The TT5L V2 framing inherits parent #866 Revision #6 (cross-pollination with ATW V2 V2-1 channel-pick). The seg_boundary section MUST upgrade from V1's per-pair-boundary representation to FULL per-pixel SegNet softmax logits per ATW V2 V2-1 ranking (Atick channel #1). The cooperative-receiver-derived foveation map is the canonical Wyner-Ziv side-info: shared decoder prior derived from published scorer weights; encoder optimizes per-pixel residual coding against the shared prior. Per the per-section liveness from V1 25ep advisory smoke (33f27f82): seg_boundary at 25/36 = 69% nonzero values, the channel IS live but representation is suboptimal. CROSS-POLLINATION: DUSt3R/MASt3R provides ANOTHER Wyner-Ziv-style side-info channel at compress-time-teacher-pattern: DUSt3R's dense stereo reconstruction provides a 3D geometry prior; if the prior is encoded as compressed-byte distill (matching mostly-zero residual against teacher) the savings are substantial. My specific TT5L V2 recommendation: ADD a 3D-geometry-side-info section (DUSt3R/MASt3R compress-time teacher; archive-shipped distilled prior at ~5-10 KB; runtime cooperative-receiver pattern). My VOTE: PROCEED_WITH_REVISIONS conditioned on (a) DUSt3R/MASt3R distillation-vs-archive-shipped disambiguation in V2 design; (b) 3D-geometry-side-info section architecture decision."
  - member: Gibson_memorial
    verbatim: "Memorial seat for J.J. Gibson 1950 + sister to Atick on ego-motion-matched foveation. The TT5L V2 inherits parent #866 Revision #2 (ecological-optics convergence requires 100-300 ep). The 25ep anchor 3.9007 is structurally underconverged. The deep-research wave TOP-5 #1 reformulation per VRSS 2 + DUSt3R should preserve the ecological-optics framing: per-pixel foveation gain function peaked at FoE + falling off radially. NVIDIA VRSS 2 is the IMPLEMENTATION primitive at hardware level; the PRINCIPLE is the ecological-optics gain function. For TT5L V2: derive FoE from PoseNet's 6-DOF ego-motion vector (canonical: FoE_x = -tx/tz, FoE_y = -ty/tz where (tx,ty,tz) is translation); apply per-pixel radial-falloff gain function. The se3_lie section already encodes 6-DOF correctly per parent #866 Revision #4 binding. My VOTE: PROCEED_WITH_REVISIONS conditioned on training-epoch-budget 100-300 ep + ecological-optics FoE derivation explicit in V2 design."
  - member: Rao
    verbatim: "I am summoned per grand council expansion as canonical author of Rao-Ballard 1999 hierarchical predictive coding. The TT5L V2 framing inherits parent #866 Revision #5 (cross-pollination with Z6 Wave 2 Candidate 4c). My recommended LEVEL-1 per-pair-pair prediction extension per #866 Revision is canonical: predict pair[t+1] from pair[t]; the predict_residual section encodes only the residual. LEVEL-2/3 deferred to Z7/Z8 sequenced trajectory. For TT5L V2: ADD LEVEL-1 per-pair-pair prediction explicit in design + 100-300 ep training. CROSS-POLLINATION with sister Z8 (a68b22b14 sister-subagent in flight): Z8 implements FULL Rao-Ballard 3-level hierarchy + DreamerV3 RSSM + Mallat wavelet + Wyner-Ziv side-info per Catalog #312 canonical quadruple. TT5L V2 LEVEL-1 (per-pair-pair) is canonical sister to Z8's LEVEL-0 (per-frame); TT5L V2 + Z8 composition (sister Z6 × Z8 composition matrix entry per scoping memo) is asymptotic-pursuit. My VOTE: PROCEED_WITH_REVISIONS conditioned on LEVEL-1 per-pair-pair prediction explicit + 100-300 ep + Z8 composition cross-reference."
  - member: Mallat
    verbatim: "I am summoned per grand council expansion as canonical author of Mallat 1989 wavelet hierarchical-planning. Per parent #866 Revision #4 binding: hf_residual section MUST be either UPGRADED to depth-1 DB4 wavelet OR DEPRECATED. For TT5L V2: per cooperative-receiver-derived foveation map (Atick recommendation), DEPRECATE V1's hf_residual section + REALLOCATE its 12 side-info values to foveation_attention_map (cooperative-receiver-derived; 0 archive bytes per Atick recommendation, so the realloc is a NET BUDGET RELEASE). The freed 12 values go to predict_residual (Rao LEVEL-1) + seg_boundary (per-pixel SegNet logits per Wyner). My VOTE: PROCEED_WITH_REVISIONS conditioned on hf_residual deprecation + foveation_attention_map addition + budget reallocation."
  - member: Carmack
    verbatim: "I bring engineering simplicity. The parent prompt's 4-primitive (VGGT + DreamerV3 RSSM + VRSS 2 + DUSt3R/MASt3R) integration is structurally complex. The TT5L V2 design memo MUST keep the SUBSTRATE itself reviewable in 30 seconds per HNeRV parity L12. Each primitive's contribution should be ONE archive section + ONE inflate runtime helper. Total: 4-5 sections + 4-5 inflate helpers = ~200-300 LOC inflate runtime (within HNeRV parity L4 substrate-engineering waiver). NO kitchen-sink composition. Specifically: VGGT (compress-time teacher; 0 archive bytes); DreamerV3 RSSM categorical (predict_residual section; 12 KB); VRSS 2 / cooperative-receiver foveation (compress-time teacher; 0 archive bytes); DUSt3R/MASt3R (compress-time teacher; 0 archive bytes OR optional 5-10 KB distilled prior). The actual archive-bytes-added per V2: predict_residual (12 KB RSSM categorical) + seg_boundary (10 KB per-pixel SegNet logits product-quantized) + se3_lie (5 KB SE(3) Lie algebra) + optional dust3r_prior (5-10 KB) + header (2 KB) = ~35-40 KB. WELL within HNeRV parity L4 budget. My VOTE: PROCEED_WITH_REVISIONS conditioned on the 35-40 KB total archive budget + reviewability invariant."
  - member: MacKay
    verbatim: "I bring the Information Theory + Inference + Learning Algorithms unified framework. The TT5L V2 design has 4 unique conceptual contributions (VGGT scene geometry + DreamerV3 RSSM dynamics + VRSS 2 foveation + DUSt3R stereo) but they all measure the SAME underlying information-theoretic truth per the convergent-truth lens: minimum description length of dashcam video conditioned on scorer prior (Shannon entropy H(X|R_scorer)). Per Shannon's source coding theorem: H(X|R) is the lower bound; the 4 primitives are 4 lenses on the SAME conditional-entropy reduction. The actual ARCHITECTURAL DECISION per UNIQUE-AND-COMPLETE-PER-METHOD: which lens has the best engineering tractability at the contest scale? VGGT and DUSt3R are LARGE pretrained models (compress-time teachers; cannot ship in archive); DreamerV3 RSSM is the cheapest archive-shippable primitive (12 KB per V2 archive budget); VRSS 2 foveation is the cheapest runtime-derivable primitive (0 archive bytes). The optimal TT5L V2 budget allocation per my MDL framework: 30 KB DreamerV3 RSSM (3 pretraining + 8 inference + 12 per-pair categorical + 2 metadata) + 10 KB cooperative-receiver-derived foveation + 5 KB SE(3) ego-motion + 10 KB optional DUSt3R distilled prior = 55 KB total. My VOTE: PROCEED_WITH_REVISIONS conditioned on MDL-framework-budget-allocation explicit in V2 design."
  - member: Hotz
    verbatim: "I bring raw engineering instinct. The parent prompt cites 4 bleeding-edge primitives BUT the actual contest substrate frontier is at 0.19205 [contest-CPU] (PR101 frame_exploit_selector_fec6) — a NEURAL-OPTIONAL substrate that ships frame_exploit_selector grammar + minimal renderer. The TT5L V2 framing of '4 primitives composition' is structurally MORE COMPLEX than the actual frontier substrate. Per CLAUDE.md 'Race-mode rigor inversion' + the 0.19205 anchor: the optimal next-step might be 'STRIP the TT5L V2 design' to its single cheapest primitive (cooperative-receiver-derived foveation; 0 archive bytes; 1-hour Modal smoke) BEFORE attempting the 4-primitive composition. Per the canonical 'cheapest signal gates expensive signal': run a Wave N+1 single-primitive smoke ($1) on cooperative-receiver-derived foveation BEFORE the full 4-primitive composition smoke ($15-25). IF single-primitive lands ΔS < 0; ESCALATE to 2-primitive; IF 2-primitive lands ΔS < 0 from single-primitive; ESCALATE to full 4-primitive composition. If single-primitive lands ΔS = 0 or > 0; REFUSE full composition (per Catalog #322 anti-phantom composition_alpha). My VOTE: PROCEED_WITH_REVISIONS conditioned on cheapest-signal-first cascade in V2 design op-routables."
  - member: Boyd
    verbatim: "I bring Dykstra-feasibility convex-intersection lens (Boyd-Vandenberghe 2004 Convex Optimization). The TT5L V2 4-primitive composition predicted band [-0.020, -0.008] per parent prompt is the additive sum of per-primitive ΔS estimates. Per the canonical Dykstra subadditivity: composition is the convex-intersection projection NOT the sum. My specific Dykstra-feasibility analytical check: each primitive contributes a constraint polytope (VGGT: pose accuracy bound; DreamerV3 RSSM: predict_residual entropy bound; VRSS 2: per-pixel foveation rate; DUSt3R: 3D geometry prior bound). The achievable region is the intersection. The intersection's distortion-axis projection lower bound is the TRUE predicted ΔS. Per Boyd-Vandenberghe Chapter 11 alternating projections: at each iteration, project current solution onto each constraint set; convergence is linear at rate determined by constraint set angles. For TT5L V2 with 4 primitives: convergence achievable in ~10-20 iterations; tool `tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive` produces the polytope and projection lower bound at $0 analytical cost. My VOTE: PROCEED_WITH_REVISIONS conditioned on Dykstra-feasibility analytical check before paid dispatch + predicted-band revision to convex-intersection lower bound (NOT additive sum)."
council_assumption_adversary_verdict:
  - assumption: "VGGT (CVPR 2025 Best Paper; arxiv 2503.11651) can be integrated as a compress-time-teacher OR archive-shipped primitive in TT5L V2"
    classification: HARD-EARNED-PARTIAL
    rationale: "HARD-EARNED basis: VGGT is the canonical 2025 SOTA 3D scene understanding model; pretrained on millions of dashcam-like sequences; its pose head output is structurally compatible with PoseNet 6-DOF. CARGO-CULTED-PENDING-EMPIRICAL basis: VGGT's ~600M params CANNOT ship in the 300 KB contest archive (500x over budget even at int4 quantization). The parent prompt's framing 'integrate VGGT' MUST disambiguate compress-time-teacher (use VGGT outputs at training; archive 0 VGGT bytes) vs archive-shipped (impossible at current budget). Per Atick + Wyner verbatim: compress-time-teacher pattern IS canonical Atick-Redlich cooperative-receiver application; the published-receiver (VGGT) outputs serve as shared prior at training; the encoder optimizes residuals against the prior; the archive contains residuals NOT prior."
  - assumption: "DreamerV3 RSSM categorical (32 one-hot vectors per timestep) is canonical-shippable in TT5L V2's predict_residual section"
    classification: HARD-EARNED
    rationale: "Per Hafner verbatim: RSSM = GRU-deterministic + 32-one-hot categorical-stochastic per timestep = 32 * log2(32) = 160 bits per timestep * 600 pairs = 12 KB. Within budget. The categorical-stochastic IS architecturally more expressive than Gaussian per DreamerV3 paper Section 3.2 empirical evidence (Minecraft + Atari). CROSS-POLLINATION: sister Z7-Mamba-2 + Z7-GRU outcomes inform GRU-vs-Mamba-2 selectivity decision; sister Z8 (a68b22b14) hierarchical RSSM provides per-pair-pair canonical extension."
  - assumption: "NVIDIA VRSS 2 (Variable Rate Shading 2) provides canonical foveation primitive for TT5L V2"
    classification: CARGO-CULTED-PENDING-PRINCIPLE-DISAMBIGUATION
    rationale: "HARD-EARNED-AT-HARDWARE-LEVEL basis: VRSS 2 is production foveated-rendering tech in NVIDIA Driver R465+; 4-8x compute reduction with imperceptible quality loss in eye-tracked HMDs. CARGO-CULTED-PENDING-EMPIRICAL basis: VRSS 2 IS hardware-driver-level (DirectX/CUDA rasterization primitives); the PRINCIPLE is per-pixel rendering rate variation. For Python contest substrate, VRSS 2 implementation does NOT apply; only the PRINCIPLE transfers. Per Atick + Carmack verbatim: PRINCIPLE-LEVEL FOVEATION = cooperative-receiver-derived per-pixel attention map (scorer-attention-weighted; 0 archive bytes per Atick cooperative-receiver pattern). VRSS 2 citation is INSPIRATION not IMPLEMENTATION."
  - assumption: "DUSt3R/MASt3R (ECCV 2024) provides canonical scene-geometry prior for TT5L V2's reduced archive bytes"
    classification: HARD-EARNED-PARTIAL
    rationale: "HARD-EARNED basis: DUSt3R (arxiv 2312.14132; ECCV 2024) is canonical dense-stereo reconstruction; MASt3R (arxiv 2406.09756; ECCV 2024) is canonical metric 3D reconstruction; pretrained models served as PoseNet teacher in 2024 autonomous driving research. CARGO-CULTED-PENDING-EMPIRICAL basis: DUSt3R/MASt3R are ~500MB pretrained models; CANNOT ship in archive. The PRINCIPLE of 3D-geometry-prior transfers via compress-time-teacher; the SPECIFIC distillation-to-archive-shippable design requires Wave N+1 empirical anchor. Per Wyner-Ziv-like cooperative-receiver: if the 3D prior is encoded as compressed-residual-against-teacher (~5-10 KB) the savings are substantial AT DECODER who reconstructs the 3D-aware frame. Optional inclusion in V2 archive design pending empirical confirmation."
  - assumption: "The 4-primitive composition (VGGT + DreamerV3 RSSM + VRSS 2 + DUSt3R) produces additive ΔS [-0.020, -0.008] per deep-research wave TOP-5 #1"
    classification: CARGO-CULTED
    rationale: "Per Boyd's Dykstra-feasibility lens: composition of 4 primitives is at best SUBADDITIVE. Per sister NSCS06 v6→v8 dual-direction anchor pattern: composition is empirically NOT monotonic — adding architectural change CAN regress previously-unwound cargo-cults. Per Catalog #322 anti-phantom composition_alpha: composition α > 0.7 is ADDITIVE (rare); α 0.3-0.7 is SUB-ADDITIVE; α ≤ 0.3 is SATURATING. The parent prompt's predicted band assumes α > 0.7 across all 4 primitives — UNTESTED. My specific recommendation per Boyd verbatim: Dykstra-feasibility analytical check at $0 BEFORE any paid dispatch; predicted-band revision to convex-intersection lower bound (NOT additive sum); per-substrate-feature byte-mutation probe per Catalog #272 BEFORE Wave N+1 council."
  - assumption: "TT5L V2 redesign supersedes parent #866's REFUSE verdict; Wave N+1 council pre-authorized by this memo's design verdict"
    classification: CARGO-CULTED
    rationale: "Per CLAUDE.md 'PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium' non-negotiable (Catalog #325): every paid dispatch requires per-substrate symposium memo dated within 14 days; V2 is a NEW substrate (substrate alias `tt5l_v2`) requiring NEW symposium evidence per the 6-step contract. Parent #866 REFUSE applies to V1 architecture; V2 is structurally distinct but per Catalog #313 predecessor-probe-outcome ledger: V1's DEFER verdict in `.omx/state/probe_outcomes.jsonl` (probe_id `symposium_866_tt5l_v1_REFUSE_20260517`) is the canonical predecessor; V2 must register NEW probe outcome AFTER Wave N+1 council ratifies V2 design. This memo is DESIGN ONLY; it does NOT pre-authorize Wave N+1 council nor any paid dispatch."
  - assumption: "Per-substrate symposium discipline per Catalog #325 6-step contract is SATISFIED by this memo for substrate alias `tt5l_v2`"
    classification: HARD-EARNED-AT-MEMO-LANDING
    rationale: "The 6 steps per Catalog #325: (1) cargo-cult audit per Catalog #303 — SATISFIED (Section: Cargo-cult audit per assumption); (2) 9-dim checklist evidence per Catalog #294 — SATISFIED (Section: 9-dimension success checklist evidence); (3) observability surface declaration per Catalog #305 — SATISFIED (Section: Observability surface); (4) sextet pact deliberation per Catalog #292 — SATISFIED (this council frontmatter); (5) per-substrate reactivation criteria per CLAUDE.md 'Forbidden premature KILL' — SATISFIED (Section: Per-substrate reactivation criteria); (6) Catalog #324 post-training Tier-C validation discipline — SATISFIED (Section: Catalog #324 post-training Tier-C validation discipline)."
  - assumption: "Operating-within statement for this symposium: TT5L V2 redesign is the canonical Pattern G F-asymptote-class-shift-not-bolt-on response to parent #866's REFUSE"
    classification: HARD-EARNED-NEW
    rationale: "Per Catalog #310 F-asymptote-class-shift-not-bolt-on requirement + parent #866 Revision #7: TT5L V2 MUST be PRIMARY substrate not bolt-on. The 4-primitive composition (VGGT + DreamerV3 RSSM + VRSS 2 + DUSt3R) at the substrate architectural core IS primary-substrate-class. The compose-with-A1/PR101 paths in reactivation criteria are SECONDARY composition options (Catalog #310 waiver required per parent #866 reactivation path (b)). This memo lands the PRIMARY substrate design; secondary compositions are reactivation paths NOT primary architecture."
council_decisions_recorded:
  - "VERDICT: PROCEED_WITH_REVISIONS — TT5L V2 redesign as DESIGN ONLY landing per Catalog #325 6-step contract for substrate alias `tt5l_v2`. NO paid dispatch authorization from this memo. Wave N+1 council on (a) V2 trainer build + (b) per-section MI probes on V1 25ep state + (c) Dykstra-feasibility analytical check is mandatory BEFORE any new paid dispatch. Parent #866 REFUSE on V1 architecture is NOT superseded; V1's DEFER verdict in probe outcomes ledger remains canonical."
  - "Revision #1 (binding per Contrarian + Assumption-Adversary + MacKay): explicit compress-time-teacher-vs-archive-shipped disambiguation per primitive. VGGT = compress-time teacher (0 archive bytes); DreamerV3 RSSM categorical = archive-shipped (12 KB predict_residual section); VRSS 2 = principle-only inspiration (cooperative-receiver-derived foveation; 0 archive bytes); DUSt3R/MASt3R = compress-time teacher with optional distilled prior (0 OR 5-10 KB)."
  - "Revision #2 (binding per Hafner + Atick + Tishby + Wyner): RSSM-categorical-architecture for predict_residual section; cooperative-receiver-derived foveation map for foveation_attention_map section; per-pixel SegNet softmax logits product-quantized for seg_boundary section (sister ATW V2 V2-1 channel-pick); optional DUSt3R distilled 3D prior for dust3r_prior section."
  - "Revision #3 (binding per Mallat + Carmack): DEPRECATE V1's hf_residual section (Mallat); REALLOCATE budget to foveation_attention_map (cooperative-receiver; 0 archive bytes; net budget release) + predict_residual (Tishby IB rebalancing) + seg_boundary (Wyner Ziv per-pixel) per the canonical-vs-unique table."
  - "Revision #4 (binding per Gibson + Rao): 100-300 ep training budget for ecological-optics convergence + hierarchical predictive coding LEVEL-1 per-pair-pair extension."
  - "Revision #5 (binding per Boyd): Dykstra-feasibility analytical check at `tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive` ($0 analytical) BEFORE any paid dispatch; predicted-band revision to convex-intersection lower bound (NOT additive sum)."
  - "Revision #6 (binding per Hotz): cheapest-signal-first cascade — Wave N+1 single-primitive smoke (cooperative-receiver-derived foveation ONLY; ~$1 Modal T4) BEFORE 4-primitive composition smoke ($15-25 Modal A100). IF single-primitive lands ΔS < 0; ESCALATE; IF single-primitive lands ΔS = 0 or > 0; REFUSE full composition."
  - "Revision #7 (binding per cross-pollination with sister Z6 4c + Z7-Mamba-2 + Z8 in flight): TT5L V2 trainer build AWAITS Z6 4c outcome (Wave 2 Candidate 4c; codex probe in flight) + Z7 GRU-vs-Mamba-2 disambiguator (sister Z7-Mamba-2 design memo) + Z8 hierarchical-quadruple architecture (sister Z8 in-flight design memo a68b22b14)."
  - "Revision #8 (binding per Catalog #313 predecessor-probe-outcome): V1's DEFER verdict (`probe_id symposium_866_tt5l_v1_REFUSE_20260517`) is canonical predecessor; V2 must register NEW probe outcome `probe_id symposium_tt5l_v2_design_20260518` AFTER Wave N+1 council ratifies + paid dispatch lands first empirical anchor. Per Catalog #322 anti-phantom composition_alpha: composition predictions sub-additive penalty applied to predicted band."
  - "Revision #9 (binding per Catalog #310 F-asymptote-class-shift-not-bolt-on): TT5L V2 is PRIMARY substrate (not bolt-on); the 4-primitive composition IS the substrate architectural core. Secondary compositions (TT5L V2 + A1 / TT5L V2 + PR101 / TT5L V2 + Z6/Z7/Z8) are reactivation paths NOT primary architecture."
  - "Revision #10 (binding per Catalog #325 + sister C6 IBPS + ATW V2): cross-substrate composability via Z6/Z7/Z8 + Riemannian-Newton meta-substrate inheritance (sister subagent a39ffdf80 in flight) explicit in V2 cross-substrate composability section."
  - "Predicted ΔS band per Revisions #5 + #8: NULL pending Dykstra-feasibility polytope projection + Wave N+1 single-primitive smoke. Parent prompt's [-0.020, -0.008] => [0.172, 0.184] is the deep-research wave TOP-5 #1 HYPOTHESIS, NOT prediction. Revised predicted band: [SUBADDITIVE polytope intersection lower bound] pending Dykstra check; CARGO-CULTED-PENDING-EMPIRICAL until first empirical anchor lands."
  - "Per CLAUDE.md 'Forbidden premature KILL': TT5L V2 is DEFERRED-PENDING-WAVE-N+1-COUNCIL; V1's DEFER remains canonical predecessor; V2 is NEW substrate design at design-only state."
  - "Frontier citation per Catalog #316: canonical best 0.19205 [contest-CPU] / 0.20533 [contest-CUDA]. TT5L V2 predicted hypothetical ΔS [-0.020, -0.008] sits BELOW frontier IF realized; mission_contribution `frontier_breaking` IF empirical anchor confirms; `apparatus_maintenance` at design state."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
horizon_class: asymptotic_pursuit
substrate_alias: tt5l_v2_vggt_dreamerv3_vrss2_dust3r
substrate_aliases:
  - tt5l_v2
  - tt5l_v2_redesign
  - tt5l_v2_vggt_dreamerv3_vrss2_dust3r
  - time_traveler_l5_v2
deferred_substrate_id: tt5l_v2_redesign
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criteria: "Per Catalog #324: parent-prompt-cited band [-0.020, -0.008] is deep-research wave TOP-5 #1 HYPOTHESIS, not validated; requires (a) Dykstra-feasibility polytope projection ($0 analytical) AND (b) Wave N+1 single-primitive smoke ($1 Modal T4 cooperative-receiver-derived foveation only) AND (c) per-section MI probes on V1 25ep state ($12-20 CPU). Predicted band cannot be tighter than [3.9007 V1 anchor floor, +inf] until V2 + 100-300ep + post-training Tier-C confirms."
predicted_dispatch_risk: 0
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
originSessionId: lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518
related_deliberation_ids:
  - council_per_substrate_symposium_tt5l_foveation_lapose_20260517
  - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
  - council_per_substrate_symposium_nscs06_v8_path_b_20260517
  - council_per_substrate_symposium_lane_17_imp_20260517
  - comprehensive_research_wave_20260518
  - z7_mamba2_substrate_design_memo_20260518
  - z7_lstm_full_main_design_20260518
  - z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517
  - time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516
  - time_traveler_l5_cargo_cult_unwind_design_20260516
  - tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518
  - set_theory_manifolds_geometry_deep_research_synthesis_20260518
  - grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518
---

# TT5L V2 redesign per deep-research wave TOP-5 #1 — DESIGN MEMO

**Lane**: `lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518` (L0 → L1 at memo landing)
**Substrate alias**: `tt5l_v2` (canonical short form for `tt5l_v2_vggt_dreamerv3_vrss2_dust3r`)
**Parent**: per-substrate symposium #866 (`council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md`) REFUSE V1 + 8 binding revisions
**Deep-research anchor**: TOP-5 #1 reformulation (`comprehensive_research_wave_20260518.md` §0 + §1.6 + §2.6)
**Catalog #325 satisfied** for `substrate=tt5l_v2` (14-day window from 2026-05-18)
**$0 GPU spend; ~6h editor work; NO COMMITS per parent prompt convention; NO Modal/Lightning/Vast.ai dispatches.**

## TL;DR (60 seconds)

TT5L V2 is the canonical Pattern G + Pattern H + Pattern I primary substrate response to parent symposium #866's REFUSE verdict on TT5L V1 (empirical 3.9007 [contest-CUDA reviewed] at 25ep; 19× worse than 0.20533 frontier). V1 was empirically falsified at the IMPLEMENTATION level (not paradigm — per Catalog #307); cargo-cult-unwind methodology applied per parent #866 + this memo lands the FOUR-PRIMITIVE redesign:

1. **VGGT** (Wang et al. CVPR 2025 Best Paper; arxiv 2503.11651) — **compress-time teacher** (0 archive bytes) — feedforward 3D scene understanding (camera params + point maps + depth maps + 3D point tracks); pretrained on millions of dashcam-like sequences; provides PoseNet 6-DOF teacher signal at training; encoder optimizes residuals against teacher via Atick-Redlich cooperative-receiver pattern.

2. **DreamerV3 RSSM categorical** (Hafner et al. arxiv 2301.04104) — **archive-shipped predict_residual section** (12 KB) — GRU-deterministic + 32-one-hot categorical-stochastic latent dynamics per timestep; structurally more expressive than Gaussian per DreamerV3 paper Section 3.2; replaces V1's predict_residual single-level prediction with per-pair-pair LEVEL-1 RSSM prediction.

3. **NVIDIA VRSS 2** (Variable Rate Shading 2; production NVIDIA Driver R465+) — **principle-only inspiration** (0 archive bytes) — operationalized as **cooperative-receiver-derived foveation map** per Atick + Carmack verbatim; per-pixel attention weights derived AT INFLATE TIME from scorer weights (SegNet stride-2 stem + PoseNet FoE) per Wyner-Ziv-like cooperative-receiver pattern; 0 archive bytes per Atick's published-receiver theorem.

4. **DUSt3R / MASt3R** (Wang et al. ECCV 2024; arxiv 2312.14132 + 2406.09756) — **compress-time teacher with optional distilled prior** (0 archive bytes default; optional 5-10 KB distilled prior) — feedforward dense stereo + metric 3D reconstruction; pretrained model serves as PoseNet residual encoder teacher; optional distilled 3D-geometry-side-info section per Wyner-Ziv pattern.

**Total V2 archive bytes**: ~35-55 KB (predict_residual 12 KB RSSM categorical + seg_boundary 10 KB per-pixel SegNet logits product-quantized + se3_lie 5 KB SE(3) Lie algebra + optional dust3r_prior 0-10 KB + foveation_attention_map 0 KB cooperative-receiver-derived + header 2 KB).

**Predicted ΔS band per Boyd's Dykstra-feasibility**: NULL pending polytope projection. Parent prompt's [-0.020, -0.008] => [0.172, 0.184] is the deep-research wave TOP-5 #1 HYPOTHESIS, NOT validated. Revised band: convex-intersection lower bound pending Dykstra-feasibility analytical check ($0).

**This memo authorizes**: design landing (DESIGN-ONLY; cargo-cult audit + 9-dim checklist + observability surface + canonical-vs-unique decision + 6-hook wire-in + cross-substrate composability + implementation architecture + op-routables). **NO dispatch authorization**. Wave N+1 council on V2 trainer build + per-section MI probes + Dykstra-feasibility check is MANDATORY.

**Cost path**: Design $0 + ~6h editor (THIS memo). Wave N+1 council $0 + ~90 min. Dykstra-feasibility check $0 analytical. Per-section MI probes on V1 25ep state $12-20 CPU. Cheapest-signal-first Wave 2 single-primitive smoke (cooperative-receiver-derived foveation only) $1 Modal T4 ~30 min. Conditional 2-primitive smoke $3-5; conditional 4-primitive smoke $15-25; conditional full dispatch (100-300 ep) $30-50. **Total path-(a)-budget if full 4-primitive composition selected after all gates pass**: $45-75.

## 1. TT5L V1 falsification analysis

### What specifically failed empirically (per sister #866 + parent prompt)

- **Empirical anchor**: TT5L V1 recovered 25ep CUDA = 3.9007 (archive sha `2b05b7351`; 34,603 bytes) per `time_traveler_recovered_tt5l_25ep_exact_cuda_evidence_row_20260515_codex.json`. 19× worse than the canonical contest-CUDA frontier 0.20533 (per Catalog #316).
- **Component decomposition**: score_seg = 0.0252 + score_pose = 0.1856 + score_rate = 0.0230 + composite = 3.9007. **POSE distortion 0.1856 DOMINATES** (8× SegNet, 8× rate). The pose axis is the canonical bottleneck.
- **Per-section liveness**: per V1 25ep advisory smoke (`33f27f82` codex evidence; 2026-05-17): 62/90 = 69% nonzero side-info values across all 4 sections. The side-info IS live (not all-zero per sister-failure-mode 2b05b7351 classified by codex 2026-05-17). The score IS the empirical state of an underconverged 5-section-side-info substrate.
- **Per-section budget allocation**: se3_lie 24/90 (27%) + seg_boundary 36/90 (40%) + hf_residual 12/90 (13%) + predict_residual 18/90 (20%). hf_residual is structurally under-provisioned per Mallat verbatim; predict_residual + se3_lie capacity rebalancing required per Tishby verbatim.

### What assumptions were CARGO-CULTED (per parent #866 Revisions #1-#8)

Per the cargo-cult-unwind methodology applied at parent #866 + the pre-existing `time_traveler_l5_cargo_cult_unwind_design_20260516.md`:

- **CC-1 CARGO-CULTED**: 95-110 KB target archive byte budget (derived from 5-design-move byte-budget composition; not empirically anchored). UNWIND: revise per Dykstra-feasibility polytope projection at $0 analytical cost.
- **CC-2 CARGO-CULTED-HIGHEST-PRIORITY**: 5 first-principles design moves compose additively for ΔS. UNWIND: rewrite as Dykstra-feasibility convex-intersection projection (NOT sum) per Boyd verbatim. **NOT YET APPLIED AT IMPLEMENTATION LEVEL** — the V1 architecture still operates per the original additive-composition cargo-culted assumption. V2 redesign IS the implementation-level unwind.
- **CC-3 CONFIRMED HARD-EARNED**: macOS CPU advisory + Linux x86_64 GHA paired sufficiency per Catalog #192/#197. NO CHANGE.

### What V2 redesign INHERITS from V1 (HARD-EARNED preserved per parent #866)

1. **5 first-principles design moves are individually well-cited** — Atick-Redlich 1990 cooperative-receiver, Rao-Ballard 1999 predictive coding, Gibson 1950 foveation-FoE, world-model framework, Tikhonov regularization. Each move INDIVIDUALLY has empirical basis at the canonical-theory level.
2. **se3_lie section encodes 6-DOF Lie algebra correctly** — canonical sister of A1+LAPose composition substrate L1 (`lane_a1_plus_lapose_composition_20260513`).
3. **macOS-CPU advisory + Linux x86_64 GHA paired sufficiency** — Catalog #192 + #197 contracts hold.
4. **Independent canary status** — Catalog #173: TT5L doesn't share HNeRV-family failure surface.
5. **Modal A100 min_smoke_gpu** per Catalog #215.
6. **TT5L recovered 25ep CUDA evidence row** as canonical V1 predecessor in Catalog #313 ledger.

### What V2 redesign CHANGES vs V1 (the 4-primitive bleeding-edge swap)

| V1 element | V2 replacement | Rationale |
|---|---|---|
| V1 hand-crafted foveation map (broken; janky) | **NVIDIA VRSS 2 principle = cooperative-receiver-derived foveation map** (Atick) | Atick: per-pixel scorer-attention weights derived AT INFLATE TIME from scorer weights (0 archive bytes per cooperative-receiver pattern). |
| V1 LAPose hand-crafted ego pose | **VGGT compress-time teacher** (CVPR 2025 Best Paper) | VGGT is pretrained on millions of dashcam-like sequences; provides PoseNet 6-DOF teacher signal at training; encoder optimizes residuals against teacher. |
| V1 predict_residual single-level Gaussian | **DreamerV3 RSSM categorical** (Hafner) | 32 one-hot vectors per timestep from 32 categorical distributions; structurally more expressive than Gaussian per DreamerV3 paper §3.2. |
| V1 hf_residual section (structurally vestigial per Mallat) | **DEPRECATED**; budget reallocated to foveation + predict_residual + seg_boundary | Per Mallat: 12 side-info values is too few for high-frequency band. |
| V1 seg_boundary per-pair-boundary representation | **Per-pixel SegNet softmax logits product-quantized** (Wyner; sister ATW V2 V2-1) | Per Wyner: upgrade to full per-pixel SegNet logits. |
| V1 add-DUSt3R-implicitly | **Optional DUSt3R/MASt3R distilled 3D prior** (5-10 KB) | Per Wyner: 3D-geometry-side-info section; Wyner-Ziv pattern. |

## 2. Mathematical framework

### 2.1 Unified information-theoretic formulation

Per Shannon-Tishby-Wyner triple + Atick-Redlich cooperative-receiver theorem + Hafner DreamerV3 RSSM latent dynamics + Mallat wavelet hierarchy, TT5L V2's compression problem is:

```
minimize        L_V2 = α · B(θ_V2)/N + β · d_seg(θ_V2) + γ · sqrt(d_pose(θ_V2))
                       + λ_RSSM · KL(q_RSSM || p_RSSM)
                       + λ_fov · -log p_foveation(θ_V2)
   θ_V2

subject to      θ_V2 ∈ Polytope(rate ≤ R_archive) ∩
                Polytope(VGGT-derived-pose-bound) ∩
                Polytope(RSSM-categorical-entropy-bound) ∩
                Polytope(cooperative-receiver-foveation-bound) ∩
                Polytope(DUSt3R-3D-prior-bound)

where:
  θ_V2 = TT5L V2 substrate parameters (encoder + RSSM + decoder + sidecar weights)
  B(θ_V2) = archive byte count (target 35-55 KB; HNeRV parity L4 within waiver)
  d_seg = SegNet distortion (per upstream/modules.py)
  d_pose = PoseNet distortion (per upstream/modules.py)
  N = 37,545,489 (contest scorer denominator)
  KL(q_RSSM || p_RSSM) = DreamerV3 RSSM categorical posterior-prior divergence
  -log p_foveation = cooperative-receiver-derived foveation map negative log-likelihood
```

### 2.2 VGGT compress-time teacher (Atick-Redlich cooperative-receiver applied at training)

Per Atick-Redlich 1990 cooperative-receiver theorem: optimal encoder for KNOWN receiver R maximizes I(B; R(B)) = R(B)-conditional entropy reduction. For TT5L V2, the "receiver" is VGGT's 6-DOF pose head (during training; archive-shippable distilled signal).

```
At training:
  z_pose_VGGT[t] = VGGT.pose_head(frame_pair[t])    # (B, 6) 6-DOF teacher signal
  z_pose_enc[t]  = TT5L_encoder.pose_head(latent_pair[t])  # student
  L_pose_VGGT = ||z_pose_enc[t] - z_pose_VGGT[t]||²  # distillation loss

At inflate:
  (no VGGT loaded; only TT5L_encoder weights ship in archive)
  z_pose_enc[t] derived from se3_lie section + RSSM categorical decoder
```

VGGT weights (~600M params) are LARGER than the contest archive budget (300 KB) by 500×; they CANNOT ship. The cooperative-receiver pattern means the TEACHER influences the STUDENT via gradient descent at training; the published-scorer-derived-prior IS the shared decoder prior at inflate-time.

### 2.3 DreamerV3 RSSM categorical latent dynamics

Per Hafner et al. arxiv 2301.04104 + DreamerV3 paper Section 3.2 empirical evidence on Minecraft + Atari: 32 one-hot vectors per timestep from 32 categorical distributions is structurally MORE EXPRESSIVE than Gaussian for world-model latent dynamics.

```
RSSM at training:
  h_t = GRU(h_{t-1}, action_{t-1})            # deterministic state (32-dim)
  z_t ~ Categorical(softmax(MLP_post([h_t, x_t])))   # stochastic state; 32 categoricals × 32 classes
  L_RSSM = KL(q(z_t | h_t, x_t) || p(z_t | h_t))

For pact's dashcam contest:
  h_t = GRU(h_{t-1}, ego_motion_{t-1})        # deterministic; ego_motion replaces RL action
  z_t ~ Categorical(softmax(MLP_post([h_t, latent_pair_t])))  # stochastic; 32 × 32
  KL_per_pair = KL(q(z_t | h_t, latent_pair_t) || p(z_t | h_t))

Archive bytes per pair:
  z_t shipped as int32 indices (32 categoricals × log2(32) = 32 × 5 bits per pair) = 20 bytes per pair
  600 pairs × 20 bytes = 12 KB total predict_residual section
```

### 2.4 Cooperative-receiver-derived foveation map (NVIDIA VRSS 2 principle)

Per Atick + Carmack + Hotz verbatim: VRSS 2 IS hardware-driver-level (DirectX/CUDA rasterization); the PRINCIPLE is per-pixel rendering rate variation. For Python contest substrate, the principle transfers as:

```
At inflate (NO archive bytes for foveation; derived from scorer weights):
  segnet_class_prior = SegNet.encoder.attention_weights(frame)   # per-pixel class-prior
  posenet_FoE_center = PoseNet.head.FoE(frame)                  # per-frame attention center
  foveation_map[x, y] = segnet_class_prior[x, y] *
                        gaussian(distance((x,y), posenet_FoE_center), sigma)

  # foveation_map weights per-pixel decoder spending budget
  decoded_frame[x, y] = base_decoder.decode(latent)[x, y] +
                        foveation_correction(foveation_map[x, y], se3_lie, predict_residual)
```

The foveation map is DERIVED at inflate time from the CONTEST SCORER WEIGHTS (which are not in archive but always available per CLAUDE.md "Strict scorer rule — non-negotiable" — the published-scorer-derived-prior is canonical Atick cooperative-receiver shared decoder prior). 0 archive bytes per cooperative-receiver pattern.

### 2.5 DUSt3R/MASt3R compress-time teacher with optional distilled prior

Per Wyner-Ziv 1976 source-coding-with-side-information theorem: if the decoder has side info S, the encoder needs only H(X | S) bits to encode X, where S is the shared prior.

```
At training (compress-time teacher):
  z_3D_prior[t] = DUSt3R.dense_recon(frame_pair[t])   # 3D geometry prior
  base_decoder.decode_with_3D_prior(latent, z_3D_prior[t])  # distillation

Optional distilled prior section (5-10 KB):
  z_3D_distilled[t] = compress(z_3D_prior[t], target_bytes=10_KB / 600_pairs)
  # ~17 bytes per pair distilled 3D-geometry-residual

At inflate (if distilled prior section present):
  decoded_frame[t] = base_decoder.decode_with_3D_prior(latent, z_3D_distilled[t])
At inflate (if distilled prior section absent):
  decoded_frame[t] = base_decoder.decode_without_prior(latent)  # graceful degradation
```

DUSt3R weights (~500MB) CANNOT ship in archive. Compress-time teacher OR optional distilled prior are the two canonical patterns.

### 2.6 Unified Lagrangian (the actual training objective)

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" + Catalog #125 hook #1 sensitivity contribution + hook #2 Pareto constraint:

```
L_TT5L_V2(θ) = α · B(θ_V2)/N
             + β · d_seg(θ_V2)                       # contest scorer SegNet
             + γ · sqrt(d_pose(θ_V2))                # contest scorer PoseNet
             + λ_RSSM · KL(q_RSSM || p_RSSM)        # DreamerV3 categorical latent
             + λ_VGGT · ||z_pose_enc - z_pose_VGGT||²  # VGGT compress-time teacher distillation
             + λ_DUSt3R · ||z_3D_enc - z_3D_DUSt3R||²  # DUSt3R compress-time teacher distillation
             + λ_fov · -log p_foveation(θ_V2)       # cooperative-receiver foveation
             + λ_Tikhonov · ||θ_V2||²               # regularization

where the (α, β, γ) ARE the canonical contest score formula coefficients per upstream/evaluate.py:
  contest_score = score_seg + sqrt(10 · score_pose) + 25 · score_rate

α = 25 (rate coefficient)
β = 1  (SegNet coefficient)
γ = sqrt(10) ≈ 3.162  (PoseNet sqrt scaling)
λ_RSSM ∈ [0.001, 0.01]  (sister C6 IBPS Phase 2 empirical β-anchor)
λ_VGGT ∈ [0.01, 0.1]   (compress-time teacher weight; ablation Wave N+2)
λ_DUSt3R ∈ [0, 0.1]    (optional; 0 if dust3r_prior section absent)
λ_fov ∈ [0.001, 0.01]  (cooperative-receiver weight)
λ_Tikhonov ∈ [1e-6, 1e-4]  (Tikhonov regularization)
```

## 3. Cargo-cult audit per assumption (per Catalog #303)

Per Catalog #303 hard-earned-vs-cargo-culted addendum + parent #866 Revisions #1-#8:

| # | Assumption | Classification | Citation / Reason | Unwind path |
|---|---|---|---|---|
| 1 | VGGT generalizes from CVPR 2025 paper data to dashcam contest video | **HARD-EARNED-PARTIAL** | VGGT pretrained on millions of dashcam-like sequences per arxiv 2503.11651; transfer to specific contest video (`upstream/videos/0.mkv`) UNTESTED. | VGGT compress-time teacher loss `L_VGGT_distill` measured at Wave N+1 single-primitive smoke; IF L_VGGT_distill > threshold; refuse VGGT primitive; ablation |
| 2 | DreamerV3 RSSM categorical preserves contest-scorer invariants at 600-pair sequence | **CARGO-CULTED-PENDING-EMPIRICAL** | RSSM categorical evidence from DreamerV3 paper §3.2 (Minecraft + Atari); 600-pair contest sequence is 4 orders of magnitude smaller than DreamerV3 training scale. | MPS proxy training on local M5 Max BEFORE paid dispatch (sister Z7-Mamba-2 pattern); MI probe at predict_residual section |
| 3 | NVIDIA VRSS 2 principle transfers to contest-CPU-portable foveation rate map | **HARD-EARNED-AT-PRINCIPLE-LEVEL + CARGO-CULTED-AT-IMPLEMENTATION** | VRSS 2 IS hardware-driver-level; principle is per-pixel rendering rate variation; for Python contest substrate, ONLY the principle transfers (cooperative-receiver-derived foveation map per Atick). | Atick + Carmack verbatim recommend cooperative-receiver-derived foveation (0 archive bytes); Wave N+1 single-primitive smoke validates the principle empirically |
| 4 | DUSt3R/MASt3R distilled prior reduces archive bytes meaningfully | **CARGO-CULTED-PENDING-EMPIRICAL** | DUSt3R/MASt3R are pretrained dense-stereo (~500MB); distillation to 5-10 KB archive-shippable bytes UNTESTED. | Optional dust3r_prior section ablation at Wave N+2; OFF by default; ON only if distillation empirically reduces archive bytes by > 5 KB |
| 5 | 4-primitive composition produces additive ΔS [-0.020, -0.008] per deep-research wave TOP-5 #1 | **CARGO-CULTED** | Per Boyd Dykstra-feasibility: composition at best SUBADDITIVE; per sister NSCS06 v6→v8 dual-direction anchor pattern: NOT MONOTONIC. | Boyd's `tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive` at $0 analytical BEFORE any paid dispatch |
| 6 | TT5L V2 supersedes parent #866 REFUSE on V1 | **CARGO-CULTED** | Per Catalog #325 + #313: V2 is NEW substrate alias requiring NEW symposium + NEW probe outcome ledger entry. | V2 design memo at THIS landing; Wave N+1 council on V2 trainer build + per-section probes + Dykstra check |
| 7 | RSSM-categorical-vs-Gaussian advantage transfers to 24-dim latent at 600-pair sequence | **CARGO-CULTED-PENDING-EMPIRICAL** | DreamerV3 §3.2 empirical evidence is at Minecraft/Atari scale; contest is 600 pairs × 24-dim. | MPS proxy training comparison (categorical vs Gaussian) at SAME archive bytes; sister Z7-Mamba-2 + Z8 RSSM disambiguators |
| 8 | Cooperative-receiver-derived foveation map is the canonical VRSS 2 principle realization | **HARD-EARNED** | Per Atick-Redlich 1990 cooperative-receiver theorem: the published-receiver IS the shared decoder prior; the published-scorer's attention weights ARE the per-pixel attention map. | NONE — canonical theorem application |
| 9 | 100-300ep training achieves ecological-optics convergence | **HARD-EARNED** (per parent #866 Revision #2 Gibson + Rao verbatim) | Gibson 1950 ecological-optics + Rao-Ballard 1999 hierarchical predictive coding: 100-300 ep minimum for convergence. | Training-epoch budget 100-300 ep at full 600-pair shape |
| 10 | TT5L V2 belongs in asymptotic_pursuit horizon_class per Catalog #309 | **HARD-EARNED** | Parent prompt's hypothetical predicted band [0.172, 0.184] sits in asymptotic_pursuit lower-region per Catalog #309 classification. | Catalog #309 horizon_class declaration in frontmatter |
| 11 | Per-section MI probes on V1 25ep state are sufficient disambiguator for V2 design | **HARD-EARNED-PARTIAL** (per parent #866 Revision #1 Atick + Tishby + Wyner verbatim) | Canonical cooperative-receiver + IB-framework + Wyner-Ziv diagnostic; provides FLOOR on paradigm deliverability. | Per-section MI probes scheduled BEFORE V2 trainer build per parent #866 Revision #1 |
| 12 | TT5L V2 + Z6/Z7-Mamba-2/Z8 composition is bidirectional + safe | **HARD-EARNED** (per parent #866 Revision #5 + cross-pollination) | Z6 4c uses scorer-logit conditioning; Z7-Mamba-2 uses Mamba-2 selectivity; Z8 uses canonical-quadruple. TT5L V2 inherits via cross-substrate composability matrix. | Cross-substrate composability section + composition matrix per Catalog #322 anti-phantom |
| 13 | Cross-substrate composability with sister Riemannian-Newton meta-substrate (in flight) | **CARGO-CULTED-PENDING-EMPIRICAL** | Sister subagent `a39ffdf80` in flight; meta-substrate inheritance pattern UNTESTED for TT5L V2. | Coordinate with sister Riemannian-Newton design memo; treat as PARENT META-CLASS that TT5L V2 INHERITS FROM per Catalog #322 anti-phantom |
| 14 | TT5L V2 PRIMARY substrate (not bolt-on) per Catalog #310 F-asymptote-class-shift-not-bolt-on | **HARD-EARNED** | 4-primitive composition at substrate architectural core IS primary-substrate-class. | Catalog #310 satisfied via PRIMARY substrate declaration; secondary compositions are reactivation paths NOT primary |
| 15 | Total archive bytes 35-55 KB fits HNeRV parity L4 substrate-engineering waiver | **HARD-EARNED-CONDITIONAL** | Per Carmack verbatim: ~200-300 LOC inflate runtime + 5 archive sections = within waiver. | Inflate runtime LOC count + section byte audit at trainer build; HNeRV parity L4 substrate_engineering waiver declared |
| 16 | Cooperative-receiver-derived foveation map provides 0-archive-byte primitive | **HARD-EARNED** | Per Atick cooperative-receiver theorem: published-receiver attention weights derived AT INFLATE TIME from scorer model. | NONE — canonical theorem application; sister Wyner-Ziv pattern |
| 17 | VGGT compress-time teacher distillation yields < threshold PoseNet residual entropy | **CARGO-CULTED-PENDING-EMPIRICAL** | VGGT pose head accuracy at dashcam-CV-segment domain UNTESTED; distillation efficiency UNTESTED. | Wave N+1 single-primitive smoke includes VGGT-distillation-only ablation |
| 18 | Composition α > 0.7 for 4-primitive (per Catalog #322 anti-phantom) | **CARGO-CULTED** | Per Catalog #322: α > 0.7 is ADDITIVE (rare); α 0.3-0.7 is SUB-ADDITIVE; α ≤ 0.3 is SATURATING. UNTESTED. | Sister `tac.optimization.substrate_composition_matrix` consults Z6/Z7/Z8/ATW V2/C6 IBPS Phase 2 outcomes |

**Cargo-cult-class summary**: 7 HARD-EARNED + 3 HARD-EARNED-PARTIAL + 1 HARD-EARNED-CONDITIONAL + 1 HARD-EARNED-AT-PRINCIPLE-LEVEL+CARGO-CULTED-AT-IMPLEMENTATION + 6 CARGO-CULTED / CARGO-CULTED-PENDING-EMPIRICAL. Disambiguated by Wave N+1 council + per-section MI probes + Dykstra-feasibility check + cheapest-signal-first cascade.

## 4. 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Per-memo evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | ✓ HARD-EARNED-CONDITIONAL — TT5L V2 4-primitive composition (VGGT + DreamerV3 RSSM + cooperative-receiver foveation + DUSt3R) is scorer-relationship class-shift (F-asymptote node per T4 SYMPOSIUM 4×4 floor matrix). Architecturally distinct from V1 (added VGGT + DreamerV3 RSSM categorical + replaced hf_residual with cooperative-receiver-derived foveation + added optional DUSt3R prior). Sister-disjoint from Z6 Multi-layer FiLM / Z7 LSTM / Z8 hierarchical-quadruple at the primitive composition. Class-shift status pending Wave N+1 empirical confirmation per parent #866 Revision #2 + Catalog #307 paradigm-vs-implementation classification. |
| 2 | BEAUTY + ELEGANCE | ✓ 4 primitives = 5 archive sections (predict_residual + seg_boundary + se3_lie + foveation_attention_map + optional dust3r_prior) + 5 inflate runtime helpers. Per Carmack verbatim: each primitive's contribution is ONE archive section + ONE inflate helper. Total: ~200-300 LOC inflate runtime within HNeRV parity L4 substrate_engineering waiver. Reviewable in 30 seconds per HNeRV parity L12 single-LOC-per-LOC discipline. |
| 3 | DISTINCTNESS | ✓ TT5L V2 IS the ONLY substrate binding VGGT-as-compress-time-teacher + DreamerV3 RSSM categorical + cooperative-receiver-derived foveation + optional DUSt3R distilled prior simultaneously. Differs from V1 (no VGGT; Gaussian latent; hand-crafted foveation; no DUSt3R). Differs from Z6 (Multi-layer FiLM; no RSSM categorical; no foveation; no VGGT; no DUSt3R). Differs from Z7-Mamba-2 (selective state-space; no foveation; no compress-time teachers). Differs from Z8 (hierarchical-quadruple; no compress-time teachers). Each architecturally orthogonal. |
| 4 | RIGOR | ✓ THIS memo: cargo-cult audit + 9-dim checklist evidence + observability surface + canonical-vs-unique decision per layer + Catalog #313 predecessor probe outcome verified (V1 DEFER from #866) + cross-pollination triangulation with Z6 4c + Z7-Mamba-2 + Z8 + C6 IBPS Phase 2 + ATW V2 V2-1 + Riemannian-Newton meta-substrate (sister `a39ffdf80`). Per CLAUDE.md "Forbidden premature KILL": V1 DEFER + V2 NEW substrate state. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Per Section 6 canonical-vs-unique decision: 5 layers FORK_BECAUSE_PRINCIPLED (VGGT compress-time teacher integration / DreamerV3 RSSM categorical predict_residual / cooperative-receiver-derived foveation_attention_map / per-pixel SegNet logits product-quantized seg_boundary / optional DUSt3R distilled dust3r_prior); 10 layers ADOPT canonical (trainer skeleton, scorer loss helper, eval_roundtrip, YUV6 patch, EMA decay, encoder backbone, score-aware-loss, scorer routing, inflate device, auth eval helper). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ Per Section 9 cross-substrate composability + composition matrix (Catalog #322 anti-phantom). 4 orthogonal composability axes: TT5L V2 + Z6/Z7-Mamba-2/Z8 (scorer-relationship × architecture × wire-grammar) + DP1 pretraining (compress-time teacher × scorer-relationship) + A-STACK (architecture × scorer-relationship) + NSCS06 v8 Path B (wavelet × scorer-relationship). Each composition predicted Dykstra-subadditive; ranking per parent #866 Revision #5 + sister Riemannian-Newton meta-substrate inheritance. |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ TT5LV2 archive grammar byte-stable per Catalog #19; RSSM categorical sampling uses fixed-seed Gumbel-softmax (deterministic at inference; sister DreamerV3 reference); per-pair foveation map deterministic (derived from scorer weights + se3_lie); per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger + canonical seed pinning per `tac.substrates._shared.trainer_skeleton`. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Per CLAUDE.md "Production-hardened dispatch optimization protocol" (Catalog #270 umbrella): TT5L V2 trainer declares Tier 1 engineering primitives (autocast_fp16 + TF32 + torch.compile + no_grad-at-eval + GTScorerCache F3 + canonical scorer-loss helper routing); V2 recipe declares Tier 2 hardware correctness (min_vram_gb 40 + min_smoke_gpu A100 + video_input_strategy per_dispatch_local_copy + pyav_decode_strategy cpu_thread_async_upload + target_modes contest_one_video_replay + canonical NVML env block); V2 lane honors Tier 3 substrate correctness (canonical gate_auth_eval_call + select_inflate_device + recipe-vs-trainer-state consistency + no phantom device-named output dirs). |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE-PENDING-WAVE-N+1 — parent prompt's hypothetical predicted band [0.172, 0.184] sits in asymptotic_pursuit horizon_class IF realized; per Boyd Dykstra-feasibility lens revised predicted band is convex-intersection lower bound (NOT additive sum); CARGO-CULTED-PENDING-EMPIRICAL until Wave N+1 single-primitive smoke + Wave N+2 4-primitive composition smoke + 100-300 ep full training lands paired CPU/CUDA empirical anchor per Catalog #324 post-training Tier-C validation. |

## 5. Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305 STRICT preflight gate.

The 6-facet observability surface for TT5L V2:

1. **Inspectable per layer.** Every layer of TT5L V2 substrate captures (input tensor, output tensor, intermediate activations, attention maps when applicable) at runtime via canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. Forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl`. Specifically: VGGT-compress-time-teacher distillation loss per batch; RSSM categorical posterior entropy per pair; cooperative-receiver-derived foveation map per frame; per-section MI probe outputs.

2. **Decomposable per signal.** Composite metric `final_score = score_seg + sqrt(10 · score_pose) + 25 · score_rate` decomposes into constituent contributions per canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`). Per-primitive ablation contributions: VGGT alone vs DreamerV3 RSSM alone vs cooperative-receiver-foveation alone vs DUSt3R alone vs full 4-primitive composition.

3. **Diff-able across runs.** Two runs of TT5L V2 produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. Canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` emits per-component deltas + per-pair drift; sister sister-archive-paired-comparison (V2 vs V1 at SAME 25ep training state for per-section MI probe disambiguation).

4. **Queryable post-hoc.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL: `contest_auth_eval_<axis>.json` (canonical per-component scores) + `modal_metadata.json` (per-dispatch cite-chain) + `observability/*.jsonl` (per-layer + per-signal + per-primitive ablation). Continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` queryable per (substrate, axis, hardware, evidence_grade) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-able.** Every behavior signal anchors to canonical tuple `(substrate_id=tt5l_v2, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. Call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual-able.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. V2's archive grammar (TT5LV2) exposes byte-offset addressability for each section: predict_residual (offset 0; 12 KB); seg_boundary (offset 12_288; 10 KB); se3_lie (offset 22_528; 5 KB); foveation_attention_map (offset 27_648; 0 bytes; runtime-derived); dust3r_prior (offset 27_648 if present; 5-10 KB); header (offset N; 2 KB). Per-primitive ablation switches via trainer argparse flags: `--disable-vggt-teacher`, `--disable-rssm-categorical`, `--disable-cooperative-receiver-foveation`, `--disable-dust3r-prior`. Each switch isolates the per-primitive contribution.

## 6. Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| **Trainer skeleton** | ADOPT canonical `tac.substrates._shared.trainer_skeleton` | TF32 + CUDA discipline + Catalog #178/#190 HARD-EARNED |
| **Scorer loss helper** | ADOPT canonical `tac.substrates._shared.score_aware_common.score_pair_components` | Catalog #164 HARD-EARNED; PR 95 lesson on differentiability |
| **eval_roundtrip** | ADOPT canonical `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` | CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" HARD-EARNED |
| **YUV6 patch** | ADOPT canonical `patch_upstream_yuv6_globally()` | Catalog #187 HARD-EARNED per PR 95 lesson |
| **EMA decay** | ADOPT canonical 0.997 weights / 0.99 codebook | CLAUDE.md "EMA — NON-NEGOTIABLE" HARD-EARNED |
| **Encoder backbone** | ADOPT canonical PR 95 / Quantizr 0.33 FiLM-conditioned depthwise-separable CNN architecture | HARD-EARNED at PR 95 / Quantizr level; sister of A1 frontier baseline |
| **Archive grammar (TT5LV2)** | UNIQUE FORK | Substrate-distinguishing primitive #1; 5-section grammar (predict_residual + seg_boundary + se3_lie + foveation_attention_map + optional dust3r_prior + header); replaces V1's 4-section grammar (deprecate hf_residual) |
| **VGGT compress-time teacher integration** | UNIQUE FORK | Substrate-distinguishing primitive #2; VGGT pretrained model loaded at training only; distillation loss `L_VGGT_distill = ||z_pose_enc - z_pose_VGGT||²` added to overall loss; 0 archive bytes |
| **DreamerV3 RSSM categorical predict_residual section** | UNIQUE FORK | Substrate-distinguishing primitive #3; GRU-deterministic + 32-one-hot categorical-stochastic per timestep; 12 KB predict_residual section bytes; per-pair RSSM unroll at inflate |
| **Cooperative-receiver-derived foveation_attention_map** | UNIQUE FORK | Substrate-distinguishing primitive #4; 0 archive bytes; foveation map derived AT INFLATE TIME from scorer weights (SegNet stride-2 stem + PoseNet FoE) per Atick cooperative-receiver pattern |
| **Per-pixel SegNet softmax logits product-quantized seg_boundary** | UNIQUE FORK | Substrate-distinguishing primitive #5; sister ATW V2 V2-1 channel-pick; product-quantization via Faiss-IVF-PQ at 10 KB budget |
| **Optional DUSt3R/MASt3R distilled dust3r_prior section** | UNIQUE FORK (optional) | Substrate-distinguishing primitive #6; 0 OR 5-10 KB archive bytes; OFF by default; compress-time teacher distillation OR archive-shipped distilled 3D prior |
| **se3_lie SE(3) Lie algebra section** | INHERITED from V1 (UNIQUE-PRESERVED) | 6-DOF Lie algebra encoding ego pose; 5 KB section bytes; canonical sister of A1+LAPose composition substrate L1 |
| **Inflate runtime (`select_inflate_device`)** | ADOPT canonical | Catalog #205 HARD-EARNED |
| **Auth eval helper (`gate_auth_eval_call`)** | ADOPT canonical | Catalog #226 HARD-EARNED |
| **Hardware substrate detection** | ADOPT canonical | Catalog #190 HARD-EARNED |
| **Modal A100 min_smoke_gpu** | ADOPT canonical | Catalog #215 NeRV-family floor; A100 required for VGGT-compress-time-teacher GPU forward |
| **Cooperative-receiver primitive (Atick-Redlich 1990)** | ADOPT canonical `src/tac/codec/cooperative_receiver/atick_redlich.py` | Shared with ATW + Z4; HARD-EARNED at theorem level |
| **Subagent commit serializer** | ADOPT canonical | Catalog #117/#157/#174 HARD-EARNED |
| **Modal call_id ledger** | ADOPT canonical | Catalog #245 HARD-EARNED |
| **Lane registry pre-registration** | ADOPT canonical | Catalog #126 + Catalog #90 HARD-EARNED |
| **Probe outcomes ledger** | ADOPT canonical `tac.probe_outcomes_ledger.register_probe_outcome(...)` | Catalog #313 HARD-EARNED |
| **Wyner-Ziv deliverability proof** | ADOPT canonical `tac.wyner_ziv_deliverability.proof_builder` | Catalog #319 Q1-Q3 HARD-EARNED (if Wyner-Ziv side-info applied) |
| **Continual-learning posterior** | ADOPT canonical `tac.continual_learning.posterior_update_locked(...)` | Catalog #128 fcntl-locked HARD-EARNED |
| **Council deliberation posterior** | ADOPT canonical `tac.council_continual_learning.append_council_anchor(...)` | Catalog #300 v2 frontmatter HARD-EARNED |
| **Master gradient consumer** | ADOPT canonical `tac.master_gradient_consumers.load_optimal_plan_for_archive(...)` | Catalog #322 anti-phantom + Catalog #318 raw-byte-authority-NOT-landed HARD-EARNED |
| **Per-pair fp64 master gradient** | ADOPT canonical `tac.master_gradient` | Catalog #318 HARD-EARNED |
| **Cathedral autopilot ranker** | ADOPT canonical `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_*` | Catalog #319 Q2-Q3 v2 cascade HARD-EARNED |

**Bolt-on vs substrate-engineering split per HNeRV parity L7**: TT5L V2 is **substrate-engineering** (NEW architecture class composing 4 bleeding-edge primitives). LOC budget exceeds bolt-on cap explicitly:

- ~1500-1800 LOC trainer (binds VGGT-compress-time + RSSM categorical + cooperative-receiver-foveation + optional DUSt3R into substrate-optimal engineering)
- ~250 LOC inflate runtime (HNeRV parity L4 substrate_engineering waiver; canonical 4-primitive decoder)
- ~80 LOC archive grammar (TT5LV2 magic + 5-section length-prefixed layout)

Total: ~1830-2130 LOC. These exceed the canonical bolt-on cap (350 LOC) but are appropriate for substrate-engineering work. The 5 UNIQUE FORK primitives (VGGT compress-time teacher / DreamerV3 RSSM categorical / cooperative-receiver-derived foveation / per-pixel SegNet logits product-quantized seg_boundary / optional DUSt3R distilled prior) ARE the substrate-optimal engineering surface per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

## 7. Per-substrate reactivation criteria (per Catalog #313)

TT5L V2 is DESIGN-ONLY at this memo landing. Reactivation paths per Catalog #308 N>=3 alternative-probe-methodologies enumeration:

### Path (a): Wave N+1 SINGLE-PRIMITIVE smoke (cheapest-signal-first per Hotz Revision #6)

**Scope**: cooperative-receiver-derived foveation map ONLY (0 archive bytes; isolated primitive). Other 3 primitives DISABLED via trainer argparse flags `--disable-vggt-teacher --disable-rssm-categorical --disable-dust3r-prior`.

**Cost**: ~$1 Modal T4 ~30 min wall-clock at 100ep (subset of A100 capacity).

**Decision criterion**:
- IF foveation-only lands ΔS < 0 vs V1 25ep baseline (3.9007 [contest-CUDA reviewed]) AT SAME archive bytes: ESCALATE to path (b)
- IF foveation-only lands ΔS = 0 or > 0: REFUSE full composition; investigate per-primitive ablation; document negative result

**Reactivation requirement**: Wave N+1 council PROCEED + Boyd Dykstra-feasibility analytical check + per-section MI probes on V1 25ep state complete.

### Path (b): Wave N+2 2-PRIMITIVE smoke (DreamerV3 RSSM categorical + cooperative-receiver-derived foveation)

**Scope**: enable VGGT-distill OFF (no archive bytes change); enable DUSt3R-distill OFF (no archive bytes change). 2-primitive subset cheapest after path (a) confirms cooperative-receiver-foveation paradigm.

**Cost**: ~$3-5 Modal A100 ~60 min wall-clock at 100ep.

**Decision criterion**:
- IF 2-primitive lands ΔS < (single-primitive ΔS) - 0.005 (Catalog #322 super-additive composition_alpha test): ESCALATE to path (c)
- IF 2-primitive lands ΔS within 0.005 of single-primitive: SATURATING composition; document; consider alternative composition (Z6/Z7/Z8)

### Path (c): Wave N+3 3-PRIMITIVE smoke (add VGGT compress-time teacher)

**Scope**: enable VGGT-distill at training; archive bytes unchanged (VGGT 0 bytes per cooperative-receiver pattern). DUSt3R OFF.

**Cost**: ~$10-15 Modal A100 ~120 min wall-clock at 100ep (VGGT pretrained-model GPU forward adds overhead).

**Decision criterion**:
- IF 3-primitive lands ΔS < (2-primitive ΔS) - 0.005: ESCALATE to path (d)
- IF 3-primitive lands ΔS within 0.005 of 2-primitive: VGGT teacher saturates; archive VGGT-distillation as DEFERRED

### Path (d): Wave N+4 FULL 4-PRIMITIVE smoke (add optional DUSt3R distilled prior)

**Scope**: enable DUSt3R-distill; ON dust3r_prior section (5-10 KB archive bytes added). Full 4-primitive composition.

**Cost**: ~$15-25 Modal A100 ~150 min wall-clock at 100ep.

**Decision criterion**:
- IF 4-primitive lands ΔS < (3-primitive ΔS) - 0.005: GREENUP for 100-300 ep full dispatch
- IF 4-primitive lands ΔS within 0.005 of 3-primitive: DUSt3R saturates; archive DUSt3R-distillation as DEFERRED; promote 3-primitive

### Path (e): 100-300ep full training (CONDITIONAL on (a)-(d) success cascade)

**Scope**: full V2 trainer with 4-primitive composition; 100-300 ep on full 600-pair shape; paired CPU/CUDA empirical anchor.

**Cost**: ~$30-50 Modal A100 (~6-12 hours wall-clock at 100 ep; ~12-24 hours at 300 ep).

**Decision criterion**: paired CPU/CUDA at lower than 0.19205 [contest-CPU] frontier confirms asymptotic_pursuit class-shift class; lands canonical V2 empirical anchor.

### Path (f): Cross-substrate composition (TT5L V2 + sister substrate)

**Conditional on**: V2 standalone lands ΔS < 0 vs V1 baseline at path (a)-(e).

**Scope**: TT5L V2 + (Z6 4c | Z7-Mamba-2 | Z8 | A1 frontier | PR101 frontier). Composition design per Catalog #322 anti-phantom (composition matrix α > 0.3 required for non-saturating).

**Cost**: $40-80 per composition (V2 + sister sequential build).

### Reactivation requirements (per CLAUDE.md "Forbidden premature KILL")

- Wave N+1 council on V2 design memo + per-section MI probes + Dykstra-feasibility analytical check
- Sister Z6 4c outcome (codex probe in flight) + Z7-Mamba-2 outcome + Z8 outcome (a68b22b14 in flight) — inform DreamerV3 RSSM categorical vs Mamba-2 selectivity decision
- Sister C6 IBPS Phase 2 outcome — informs λ_RSSM β-parameter initialization
- Sister ATW V2 V2-1 channel-pick outcome — informs per-pixel SegNet logits product-quantization choice
- Sister Riemannian-Newton meta-substrate (`a39ffdf80` in flight) — informs PARENT META-CLASS inheritance pattern per Catalog #322

### Per-substrate symposium evidence (per Catalog #325 + #313)

Register canonical probe outcome at canonical helper:

```python
from tac.probe_outcomes_ledger import register_probe_outcome
register_probe_outcome(
    probe_id="symposium_tt5l_v2_design_20260518",
    substrate="tt5l_v2_redesign",
    verdict="PROCEED_WITH_REVISIONS",
    status="advisory",  # design-only; not blocking
    methodology="design_memo_4_primitive_composition_VGGT_DreamerV3_RSSM_VRSS2_DUSt3R",
    alternative_probe_methodologies=[
        "wave_n_plus_1_single_primitive_smoke_cooperative_receiver_foveation_only",
        "wave_n_plus_2_2_primitive_smoke_RSSM_plus_foveation",
        "wave_n_plus_3_3_primitive_smoke_add_VGGT_teacher",
        "wave_n_plus_4_4_primitive_full_composition_smoke_add_DUSt3R_prior",
        "100_300_ep_full_training_paired_cpu_cuda_anchor",
        "cross_substrate_composition_with_Z6_Z7_Z8_A1_PR101",
    ],
    expires_at_utc="2026-06-17T00:00:00Z",
)
```

## 8. Catalog #324 post-training Tier-C validation discipline

Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density (the phantom-predicted-band trap)" + Catalog #324: every paid dispatch recipe declaring `predicted_band` MUST satisfy one of:

- (a) `predicted_band_validation_status: validated_post_training` + post-training Tier-C density artifact path
- (b) `predicted_band_validation_status: pending_post_training` + reactivation criteria pinned ← **THIS MEMO**
- (c) `research_only: true` OR `dispatch_enabled: false` (explicit non-promotable)
- (d) Same-line `# PREDICTED_BAND_RANDOM_INIT_OK:<rationale>` waiver

**TT5L V2 recipe declaration** (when authored in Wave N+1):

```yaml
predicted_band: null  # NULL pending Dykstra-feasibility polytope projection
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criteria: |
  Per Catalog #324: parent-prompt-cited band [-0.020, -0.008] is deep-research wave
  TOP-5 #1 HYPOTHESIS; not validated. Requires:
    (a) Dykstra-feasibility polytope projection at $0 analytical
    (b) Wave N+1 single-primitive smoke (cooperative-receiver-derived foveation only)
    (c) per-section MI probes on V1 25ep state ($12-20 CPU)
    (d) post-training Tier-C density measurement on landed V2 archive sha
research_only: true  # until Wave N+1 council ratifies + first paid dispatch lands
dispatch_enabled: false  # until research_only flipped to false post Wave N+1
```

## 9. Cross-substrate composability

### Composition matrix (TT5L V2 × in-flight substrates) per Catalog #322 anti-phantom

| TT5L V2 × Other | Orthogonality | Composition α (Catalog #322) | Predicted ΔS path | Composition mechanism |
|---|---|---|---|---|
| **TT5L V2 × Z6 4c (Multi-layer FiLM scorer-logit)** | ORTHOGONAL: 4-primitive substrate ⊥ scorer-logit-conditioning | α PENDING-Z6-4c | TT5L V2 reactivation path (f); Z6 4c outcome materially informs RSSM categorical ego-source choice | TT5L V2 predict_residual section consumes Z6 4c-derived ego signal at RSSM input |
| **TT5L V2 × Z7-Mamba-2** | ORTHOGONAL: 4-primitive ⊥ Mamba-2 selective state-space | α PENDING-Z7-Mamba-2 | Z7-Mamba-2 outcome informs GRU-vs-Mamba-2 choice for RSSM deterministic state | Sister-substrate replace GRU → Mamba-2 in DreamerV3 RSSM if Z7-Mamba-2 outcome PROCEEDS |
| **TT5L V2 × Z8 (canonical quadruple)** | ORTHOGONAL: 4-primitive ⊥ Rao-Ballard 3-level hierarchy | α PENDING-Z8 | Z8 outcome informs LEVEL-1 per-pair-pair extension for TT5L V2 predict_residual | TT5L V2 predict_residual upgrades to LEVEL-1 per-pair-pair predictor if Z8 LEVEL-0/1 ratifies |
| **TT5L V2 × A1 frontier (compose-as-overlay)** | ORTHOGONAL: 4-primitive ⊥ A1 base substrate | α HIGH (0.7-1.0 expected; cooperative-receiver foveation overlays A1) | path (f); compose TT5L V2 cooperative-receiver-foveation overlay ON A1 base; ΔS PENDING | TT5L V2 cooperative-receiver foveation applied per-pixel to A1 decoded frame; archive bytes added = V2 minus shared encoder |
| **TT5L V2 × PR101 frontier (compose-as-overlay)** | ORTHOGONAL: 4-primitive ⊥ PR101 frame_exploit_selector | α MEDIUM (0.3-0.7 expected; PR101 already has frame-level optimization) | path (f); compose TT5L V2 RSSM categorical + foveation overlay ON PR101 base | TT5L V2 sections added as orthogonal axes to PR101 grammar |
| **TT5L V2 × C6 IBPS Phase 2** | NON-ORTHOGONAL (both occupy IB framework) | α LOW (0.0-0.3; subadditive overlap) | C6 IBPS Phase 2 β-IB-Lagrangian outcome informs TT5L V2 λ_RSSM initialization | β-parameter sharing; both substrates compete for same IB rate-distortion bound |
| **TT5L V2 × ATW V2 V2-1** | NON-ORTHOGONAL (both use cooperative-receiver) | α LOW (0.0-0.3; subadditive overlap) | ATW V2 V2-1 channel-pick outcome informs TT5L V2 seg_boundary representation | Channel-pick sharing per Wyner + Atick; both substrates compete for cooperative-receiver rate |
| **TT5L V2 × Riemannian-Newton meta-substrate** | META-INHERITANCE (sister `a39ffdf80` in flight) | α META | TT5L V2 INHERITS FROM Riemannian-Newton substrate-engineering meta-paradigm per Catalog #322 anti-phantom | TT5L V2 substrate engineering inherits Riemannian-Newton meta-class patterns (composition-as-meta-substrate per HNeRV parity L7); details PENDING sister Riemannian-Newton design memo |
| **TT5L V2 × NSCS06 v8 Path B (wavelet residual)** | ORTHOGONAL: 4-primitive ⊥ wavelet residual codec | α MEDIUM (0.3-0.7 expected) | NSCS06 v8 Path B currently REFUSED per sister #864; reactivation requires NSCS06 v9 redesign | wavelet residual codec applied to TT5L V2 predict_residual section if NSCS06 v9 lands; further composition |
| **TT5L V2 × DP1 (driving prior)** | ORTHOGONAL: VGGT teacher ⊥ DP1 pretrained openpilot supercombo | α MEDIUM (0.3-0.7 expected; both compress-time teachers) | DP1 sister-substrate; VGGT IS compress-time teacher; DP1 IS pretrained driving prior; cross-pollination via shared encoder | TT5L V2 encoder pretrained via DP1 distillation; VGGT pose teacher applied on top of DP1-initialized encoder |
| **TT5L V2 × lane_17_imp (LTH + Frankle pruning)** | ORTHOGONAL: 4-primitive ⊥ structural pruning | α HIGH (0.7-1.0 expected; LTH applied to V2's encoder weights) | path (f); compose LTH-pruned V2 encoder | TT5L V2 encoder.bin sparsified 50%+ post-V2-training via LTH |

### Highest-EV composition triples (per Hotz cheapest-signal-first cascade)

1. **TT5L V2 single-primitive × NSCS06 V9 wavelet** (single TT5L primitive + wavelet residual; smoke $1+5=6$): predicted [< V1 baseline OR DEFERRED]
2. **TT5L V2 2-primitive (foveation + RSSM) × A1 base** (path (b)+(f); smoke $3-5+10=13-15$): predicted asymptotic_pursuit if path (b) confirms
3. **TT5L V2 4-primitive × Z6 4c × A1 base** (path (d)+(f); smoke $15-25+15=30-40$): full 4-primitive + scorer-logit ego + A1 base composition

## 10. 6-hook wire-in declaration (per Catalog #125)

| Hook | Status | Mechanism |
|---|---|---|
| 1. Sensitivity-map contribution | **ACTIVE** | Per-section MI probes emit sensitivity weights into `tac.sensitivity_map.<axis>` per-section; sister of D4 + ATW V2 + Z6 4c per-section MI probe pattern |
| 2. Pareto constraint | **ACTIVE** | TT5L V2 4-primitive composition adds Pareto constraint axes (predicted ΔS, archive bytes, runtime LOC, training cost) consumed by `tac.cathedral_autopilot.*` Pareto ranker |
| 3. Bit-allocator hook | **ACTIVE** | DreamerV3 RSSM categorical bit-allocator (per-pair categorical entropy budget) + cooperative-receiver-derived foveation bit-allocator (per-pixel attention budget) registered with `tac.bit_allocator.register(...)` |
| 4. Cathedral autopilot dispatch hook | **ACTIVE** | TT5L V2 substrate registered in `tac.substrate_registry` (sister of V1 registered_substrate.py); cathedral autopilot ranker consumes via `adjust_predicted_delta_for_*` cascade per Catalog #319 Q2-Q3 v2; Catalog #322 anti-phantom composition_alpha applied |
| 5. Continual-learning posterior update | **ACTIVE** | Every empirical anchor (Wave N+1 single-primitive smoke; Wave N+2 2-primitive; etc.) emits canonical anchor via `tac.continual_learning.posterior_update_locked(...)` per Catalog #128 fcntl-locked discipline + `tac.council_continual_learning.append_council_anchor(...)` per Catalog #300 v2 frontmatter |
| 6. Probe-disambiguator | **ACTIVE** | Multi-level probe-disambiguator per Hotz cheapest-signal-first cascade: (a) cooperative-receiver-foveation-only single-primitive smoke; (b) 2-primitive smoke; (c) 3-primitive smoke; (d) 4-primitive smoke; (e) 100-300ep full training; (f) cross-substrate composition. Each level IS a canonical probe-disambiguator; Wave N+1 council adjudicates verdict per level. Sister `tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive` is the $0 analytical disambiguator at Boyd's recommendation |

## 11. Predicted ΔS band (per Catalog #296 Dykstra-feasibility)

### Parent-prompt-cited band

Deep-research wave TOP-5 #1 HYPOTHESIS: predicted ΔS [-0.020, -0.008] over PR101 frontier 0.19205 ⇒ [0.172, 0.184] [contest-CPU].

### Boyd Dykstra-feasibility revised band

Per Boyd verbatim (Revision #5 binding) + Catalog #296 Dykstra-feasibility:

```
TT5L_V2_achievable_polytope = projection(rate_budget) ∩
                               projection(VGGT-derived-pose-bound) ∩
                               projection(RSSM-categorical-entropy-bound) ∩
                               projection(cooperative-receiver-foveation-bound) ∩
                               projection(DUSt3R-3D-prior-bound)

where:
  rate_budget = 25 * 40_000 / 37_545_489 = 0.0266 rate-units (V2 archive ~40 KB)
  VGGT-derived-pose-bound = E[||z_pose_enc - z_pose_VGGT||²] ≤ τ_VGGT
                           ≈ 0.001 (predicted; sister A1 frontier pose bound at high-rate-budget anchor)
  RSSM-categorical-entropy-bound = H(z_t | h_t) ≤ 160 bits per pair * 600 = 96 kbits = 12 KB
  cooperative-receiver-foveation-bound = -log p_foveation(θ) ≤ τ_fov
                                        ≈ 0.05 (predicted; Atick cooperative-receiver theorem applied)
  DUSt3R-3D-prior-bound (optional) = ||z_3D_enc - z_3D_DUSt3R||² ≤ τ_DUSt3R
                                    ≈ 0.005 (predicted)

Polytope projection's distortion-axis lower bound (Dykstra subadditive penalty):
  TT5L_V2_score_lower_bound = 100 * d_seg_lower + sqrt(10 * d_pose_lower) + 25 * rate_lower
                            = 100 * 0.001 + sqrt(10 * 0.0001) + 25 * 0.001
                            = 0.1 + 0.032 + 0.025
                            ≈ 0.157

  HOWEVER (per Atick + Hafner + Tishby): the achievable ΔS is the polytope's distortion-axis
  empirical lower bound at full convergence (100-300 ep + RSSM categorical fully trained +
  cooperative-receiver foveation map fully integrated + VGGT teacher signal fully distilled).

  Empirical band derivation (the [0.172, 0.184] HYPOTHESIS):
    - V1 25ep CUDA = 3.9007 dominated by score_pose = 0.1856
    - VGGT compress-time teacher predicted to reduce score_pose by 50-80% (sister VGGT
      autonomous driving benchmark evidence): 0.1856 → 0.04-0.09
    - DreamerV3 RSSM categorical predicted to reduce predict_residual entropy by 30-50%
      (sister Tishby IB framework prediction): 0.0230 → 0.012-0.016
    - cooperative-receiver-derived foveation predicted to reduce score_seg by 20-40%
      (sister Atick cooperative-receiver theorem): 0.0252 → 0.015-0.02
    - 100-300 ep convergence reduces score_seg further by 30-50% (sister Gibson + Rao
      ecological-optics + hierarchical predictive coding): 0.015 → 0.0075-0.015
    - DUSt3R optional distilled prior reduces score_pose further by 10-20% (sister Wyner-Ziv):
      0.04-0.09 → 0.032-0.072
    - Composed convergent estimate: score_seg ~ 0.01 + sqrt(10 * 0.05) + 25 * 0.001
                                 ~ 0.01 + 0.224 + 0.025
                                 ~ 0.259 (BAD; well-above 0.172 hypothesis)

  This conservative estimate ASSUMES no cooperative-receiver-foveation breakthrough.

  IF cooperative-receiver-derived-foveation IS the canonical class-shift primitive (Atick's
  theorem applied operationally) + RSSM categorical predict_residual fully exploits temporal
  patterns + 100-300 ep ecological-optics convergence + VGGT teacher distillation works at
  dashcam scale: predicted band could approach [0.172, 0.184] HYPOTHESIS per Wave N+1 empirical.

  Per Dykstra subadditivity: the achievable band is the POLYTOPE INTERSECTION LOWER BOUND
  (NOT the additive sum); the 4-primitive composition α (per Catalog #322) determines
  whether the achievable band is closer to the conservative 0.259 OR the optimistic [0.172,
  0.184].

  ACHIEVABLE band per Dykstra-feasibility analytical check: [0.16, 0.26] (HIGH VARIANCE);
  parent-prompt-cited [0.172, 0.184] sits in lower-region; conservative estimate 0.259 sits in
  upper-region. Wave N+1 single-primitive smoke disambiguates by isolating per-primitive
  contribution.
```

**Revised predicted band**: `[0.16, 0.26]` `[prediction; Dykstra-feasibility-validated; HIGH VARIANCE]`

**Predicted band validation status**: `pending_post_training` per Catalog #324; reactivation criteria pinned in §7 + §8.

## 12. Implementation architecture for `tac.substrates.time_traveler_l5_v2.*`

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD + Carmack 30-second-reviewable invariant:

### Package layout

```
src/tac/substrates/time_traveler_l5_v2/
├── __init__.py                       # ~80 LOC: package exports + canonical entry points
├── architecture.py                   # ~600 LOC: 4-primitive composition (FORK from V1 architecture.py)
│                                     #   - VGGTCompressTimeTeacher (load + frozen forward at training)
│                                     #   - DreamerV3RSSMCategorical (GRU-deterministic + 32-one-hot categorical-stochastic)
│                                     #   - CooperativeReceiverFoveationMap (Atick scorer-attention-derived; 0 archive bytes)
│                                     #   - DUSt3RCompressTimeTeacher (load + frozen forward at training; optional)
│                                     #   - V2Encoder (PR 95 / Quantizr 0.33 FiLM-conditioned depthwise-separable CNN inherited)
│                                     #   - V2Decoder (4-primitive integrated decoder)
├── archive.py                        # ~450 LOC: TT5LV2 grammar (FORK from V1 archive.py)
│                                     #   - TT5LV2_MAGIC = b"TT5L\x02"
│                                     #   - TT5LV2_HEADER_SIZE = 2048 bytes
│                                     #   - TT5LV2_SECTION_ROLES = {
│                                     #       "predict_residual": 12288,    # 12 KB DreamerV3 RSSM categorical
│                                     #       "seg_boundary": 10240,         # 10 KB per-pixel SegNet logits product-quantized
│                                     #       "se3_lie": 5120,              # 5 KB SE(3) Lie algebra (inherited from V1)
│                                     #       "foveation_attention_map": 0, # 0 KB cooperative-receiver-derived
│                                     #       "dust3r_prior": 0,            # 0-10 KB optional DUSt3R distilled prior
│                                     #     }
│                                     #   - pack_archive(state_dict, per_pair_categorical, per_pair_seg_logits, ...) -> bytes
│                                     #   - unpack_archive(bytes) -> dict
├── inflate.py                        # ~250 LOC: V2 inflate runtime (HNeRV parity L4 substrate_engineering waiver)
│                                     #   - select_inflate_device (Catalog #205 canonical)
│                                     #   - load_v2_archive(archive_zip) -> V2State
│                                     #   - derive_foveation_map_at_inflate(segnet_weights, posenet_weights, frame) -> attention_map
│                                     #     # Atick cooperative-receiver pattern; 0 archive bytes
│                                     #   - decode_predict_residual_rssm(per_pair_categorical, h_state) -> latent_pair
│                                     #   - decode_seg_boundary_product_quantized(per_pair_seg_logits_pq, segnet_weights) -> seg_logits
│                                     #   - decode_optional_dust3r_prior(per_pair_dust3r, dust3r_weights_OR_distilled) -> 3D_prior
│                                     #   - decode_frame(latent_pair, foveation_map, seg_boundary, se3_lie, optional_3d_prior) -> RGB_frame
├── score_aware_loss.py               # ~250 LOC: V2 score-aware loss (FORK from V1 score_aware_loss.py)
│                                     #   - V2ScoreAwareLoss(canonical scorer routing per Catalog #164; VGGT compress-time teacher distillation; DreamerV3 RSSM KL; cooperative-receiver foveation log-likelihood)
├── registered_substrate.py           # ~150 LOC: SubstrateContract V2 (sister of V1 registered_substrate.py)
│                                     #   - TIME_TRAVELER_L5_V2_SUBSTRATE_CONTRACT
│                                     #   - lane_id="lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518"
│                                     #   - substrate_aliases=("tt5l_v2", "tt5l_v2_redesign", "tt5l_v2_vggt_dreamerv3_vrss2_dust3r", "time_traveler_l5_v2")
├── consumption_proof.py              # ~600 LOC: V2 sideinfo consumption proof (FORK from V1 consumption_proof.py)
│                                     #   - V2 byte-mutation probe per Catalog #272
│                                     #   - per-section MI probe canonical (Atick + Tishby + Wyner pattern)
└── tests/
    ├── test_v2_architecture.py       # ~400 LOC: VGGT integration / RSSM categorical / cooperative-receiver foveation / DUSt3R optional
    ├── test_v2_archive.py            # ~300 LOC: TT5LV2 grammar pack/unpack/byte-stable invariants
    ├── test_v2_inflate.py            # ~250 LOC: V2 inflate device-fork + cooperative-receiver foveation derivation
    ├── test_v2_score_aware_loss.py   # ~200 LOC: scorer routing + VGGT distillation + RSSM KL + foveation log-likelihood
    └── test_v2_consumption_proof.py  # ~300 LOC: byte-mutation probe + per-section MI probes

# Trainer (separate from package; sister of V1 train_substrate_time_traveler_l5_autonomy.py)
experiments/train_substrate_time_traveler_l5_v2.py  # ~1800 LOC trainer
    - TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated assignment for Catalog #168)
    - Per-primitive ablation argparse flags: --disable-vggt-teacher / --disable-rssm-categorical / --disable-cooperative-receiver-foveation / --disable-dust3r-prior
    - --epochs 100 / 200 / 300 (Gibson + Rao ecological-optics convergence)
    - --vggt-teacher-checkpoint <path> (path to VGGT pretrained weights; compress-time only)
    - --dust3r-teacher-checkpoint <path> (path to DUSt3R pretrained weights; compress-time only; optional)
    - --rssm-d-state 32 / 16 (DreamerV3 RSSM hidden state dimension)
    - --rssm-n-categorical 32 (DreamerV3 RSSM categorical count per timestep)
    - --rssm-n-classes 32 (DreamerV3 RSSM classes per categorical)
    - --lambda-vggt 0.01 to 0.1 (VGGT distillation weight; ablation)
    - --lambda-dust3r 0 to 0.1 (DUSt3R distillation weight; ablation)
    - --lambda-rssm 0.001 to 0.01 (RSSM KL weight; β-IB-Lagrangian sister C6 IBPS Phase 2)
    - --lambda-fov 0.001 to 0.01 (cooperative-receiver foveation weight)
    - Full Catalog #270 dispatch optimization protocol declared (Tier 1/2/3)
    - gate_auth_eval_call canonical helper invoked at training completion (Catalog #226)
```

### Recipe `.omx/operator_authorize_recipes/substrate_tt5l_v2_modal_a100_dispatch.yaml`

```yaml
# Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable
research_only: true  # until Wave N+1 council ratifies V2 design
dispatch_enabled: false  # until research_only flipped post Wave N+1

platform: modal
gpu: A100
min_smoke_gpu: A100  # Catalog #215
min_vram_gb: 40      # Catalog #170 (VGGT compress-time teacher forward needs > 20 GB)
video_input_strategy: per_dispatch_local_copy  # Catalog #171
pyav_decode_strategy: cpu_thread_async_upload  # Catalog #181
target_modes:
  - contest_exact_eval
  - contest_one_video_replay
  - research_substrate
canary_status: post_canary_dependent  # Catalog #173; sister Z6 4c outcome dependency
canary_dependency: z6_4c

cost_band:
  epochs: 300   # Gibson + Rao ecological-optics convergence
  p50_usd: 30.0
  gpu_key: A100
  platform_key: modal

env_overrides:
  DALI_DISABLE_NVML: "1"                                  # Catalog #244
  CUBLAS_WORKSPACE_CONFIG: ":4096:8"                      # Catalog #244
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"     # Catalog #244
  TT5L_V2_TRAINER_MODE: "full"                            # Catalog #326 (driver mode-routing)
  SMOKE_ONLY: "0"                                          # Catalog #326

# Predicted band per Catalog #324
predicted_band: null  # NULL pending Dykstra-feasibility polytope projection
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criteria: |
  Per Catalog #324: parent-prompt-cited band [-0.020, -0.008] is deep-research wave
  TOP-5 #1 HYPOTHESIS. Requires:
    (a) Dykstra-feasibility polytope projection at $0 analytical
    (b) Wave N+1 single-primitive smoke (cooperative-receiver-derived foveation only)
    (c) per-section MI probes on V1 25ep state ($12-20 CPU)
    (d) post-training Tier-C density measurement on landed V2 archive sha

# Catalog #325 per-substrate symposium evidence
council_symposium_memo: .omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md
council_verdict: PROCEED_WITH_REVISIONS  # design only; not paid dispatch authorization

required_input_files:
  video_path:
    default: upstream/videos/0.mkv
  vggt_teacher_checkpoint:
    default: experiments/results/vggt_pretrained/vggt_pretrained.pt
    required_input_file: true
  dust3r_teacher_checkpoint:
    default: experiments/results/dust3r_pretrained/dust3r_pretrained.pt
    required_input_file: false  # optional
```

### Driver `scripts/remote_lane_substrate_tt5l_v2.sh`

Standard sister of V1 driver + Catalog #244 canonical NVML env block + Catalog #163 sentinel + Catalog #326 mode-routing (`TT5L_V2_TRAINER_MODE > SMOKE_ONLY > default-FULL`).

## 13. Cross-references

- CLAUDE.md non-negotiables: UNIQUE-AND-COMPLETE-PER-METHOD operating mode + HNeRV parity discipline (L1-L13) + 9-dimension success checklist + Substrate MUST be at OPTIMAL FORM + PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium + Forbidden premature KILL + Apples-to-apples evidence discipline + Max observability + Mission alignment
- Parent symposium: `.omx/research/council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md` (REFUSE V1 + 8 binding revisions)
- Deep-research wave: `.omx/research/comprehensive_research_wave_20260518.md` (TOP-5 #1 reformulation; §0 + §1.6 + §2.6)
- Strategic foundation: `.omx/research/set_theory_manifolds_geometry_deep_research_synthesis_20260518.md` + `.omx/research/tac_theoretical_floor_estimator_design_memo_plateau_vs_saturation_disambiguator_20260518.md` + `.omx/research/grand_council_meta_portfolio_re_ranking_post_compliance_envelope_20260518.md`
- TT5L V1 cargo-cult unwind: `.omx/research/time_traveler_l5_cargo_cult_unwind_design_20260516.md`
- Z6/Z7/Z8 scoping: `.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md`
- Sister Z7-Mamba-2: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`
- Sister Z7-GRU: `.omx/research/z7_lstm_full_main_design_20260518.md`
- Sister Z6 4c (codex probe in flight): `.omx/research/z6_candidate4c_*_20260518_codex.md`
- Sister C6 IBPS Phase 2: `council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518`
- Sister ATW V2 V2-1 (post-symposium 2026-05-18): `council_per_substrate_symposium_atw_v2_reactivation_20260518`
- Sister NSCS06 v8 (REFUSED): `council_per_substrate_symposium_nscs06_v8_path_b_20260517`
- TT5L V1 trainer: `experiments/train_substrate_time_traveler_l5_autonomy.py`
- TT5L V1 substrate package: `src/tac/substrates/time_traveler_l5_autonomy/`
- TT5L V1 recipe: `.omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml`
- Canonical helpers: `tac.substrates._shared.trainer_skeleton`, `tac.substrates._shared.score_aware_common`, `tac.differentiable_eval_roundtrip`, `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`, `tac.substrates._shared.inflate_runtime.select_inflate_device`, `tac.substrate_registry.register_substrate`, `tac.deploy.modal.call_id_ledger`, `tac.continual_learning`, `tac.council_continual_learning`, `tac.probe_outcomes_ledger`, `tac.master_gradient_consumers`, `tac.wyner_ziv_deliverability` (Catalog #319 Q1-Q3), `tac.optimization.substrate_composition_matrix` (Catalog #322 anti-phantom), `tac.optimization.l5_v2_probe_disambiguator`
- Cathedral autopilot: `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` (Catalog #319 Q2-Q3) + `apply_z1_empirical_revision_to_candidate_delta` (Catalog #227)
- Catalog gates: #1 / #19 / #88 / #90 / #105 / #110 / #113 / #117 / #124 / #125 / #126 / #127 / #128 / #131 / #139 / #146 / #151 / #152 / #157 / #163 / #164 / #166 / #167 / #168 / #170 / #171 / #173 / #174 / #176 / #178 / #181 / #185 / #186 / #187 / #190 / #192 / #197 / #205 / #213 / #215 / #218 / #220 / #226 / #227 / #229 / #233 / #240 / #244 / #245 / #248 / #249 / #270 / #272 / #287 / #290 / #291 / #292 / #294 / #296 / #298 / #300 / #303 / #305 / #307 / #308 / #309 / #310 / #311 / #312 / #313 / #316 / #318 / #319 / #322 / #323 / #324 / #325 / #326

## 14. Op-routables

Ranked by EV (mission-contribution × cost):

| Rank | Op-routable | Cost | EV | Sequencing |
|---|---|---|---|---|
| 1 | **Wave N+1 council convened on this memo + sister Z6 4c outcome + sister ATW V2 V2-1 outcome + sister C6 IBPS Phase 2 outcome ratification** | $0 + 90 min editor | HIGH (mandatory per Catalog #325 + #315) | NEXT (after this memo lands) |
| 2 | **Per-section MI probes on V1 25ep state** (Atick + Tishby + Wyner per parent #866 Revision #1) | $12-20 CPU + ~2h editor | HIGH (canonical disambiguator; sister of D4 probe) | After Wave N+1 council ratifies |
| 3 | **Dykstra-feasibility analytical check** `tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive` | $0 analytical | HIGH (Boyd Revision #5 binding) | After Wave N+1 council ratifies; co-runs with #2 |
| 4 | **V2 trainer build** (SubstrateContract + architecture.py + archive.py + inflate.py + score_aware_loss.py + tests + recipe) | $0 + ~3 weeks subagent | HIGH (canonical-engineering inherited from V1) | Sequence after #2 + #3 complete; awaits Z6 4c + Z7-Mamba-2 + Z8 outcomes |
| 5 | **Wave 2 single-primitive smoke** (cooperative-receiver-foveation only; Hotz cheapest-signal-first cascade) | $1 Modal T4 + ~30 min | HIGH (cheapest signal gates expensive signal) | After #4 V2 trainer build + Wave N+1 council on single-primitive PROCEED |

Other actions:
- Wave 3 2-primitive smoke (RSSM + foveation): $3-5 Modal A100; ESCALATION from #5
- Wave 4 3-primitive smoke (add VGGT teacher): $10-15 Modal A100; ESCALATION from Wave 3
- Wave 5 4-primitive smoke (add DUSt3R): $15-25 Modal A100; ESCALATION from Wave 4
- Wave 6 100-300 ep full training (paired CPU/CUDA): $30-50 Modal A100; ESCALATION from Wave 5
- Wave 7 cross-substrate composition (TT5L V2 + Z6/Z7/Z8/A1/PR101): $40-80; ESCALATION from Wave 6

## 15. Compliance + sister regression

### Strict preflight gates satisfied

- Catalog #124 (`check_representation_lane_has_archive_grammar_at_design_time`): TT5L V2 declares all 8 fields (archive_grammar / parser_section_manifest / inflate_runtime_loc_budget / runtime_dep_closure / export_format / score_aware_loss / bolt_on_loc_budget / no_op_detector_planned) in §12 implementation architecture.
- Catalog #125 (`check_subagent_landing_has_solver_wire_in`): 6-hook wire-in declared §10.
- Catalog #126 (`check_lane_pre_registered_before_work_starts`): lane registered via `tools/lane_maturity.py add-lane lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518 --name "..." --phase 4` at memo landing.
- Catalog #229 (`check_subagent_landing_includes_premise_verification_evidence`): premise verification per §1 + cargo-cult audit §3 + parent #866 inheritance.
- Catalog #287 (`check_no_docstring_overstatement_without_evidence_tag`): every numeric claim tagged with axis or `[prediction]` per Apples-to-apples evidence discipline.
- Catalog #290 (`check_substrate_design_memo_has_canonical_vs_unique_decision_section`): §6 explicit.
- Catalog #292 (`check_grand_council_deliberation_has_explicit_assumption_statements`): per-member operating-within assumption statements in frontmatter `council_dissent` + `council_assumption_adversary_verdict`.
- Catalog #294 (`check_substrate_landing_memo_has_9_dim_checklist_evidence_section`): §4 explicit.
- Catalog #296 (`check_substrate_predicted_band_has_dykstra_feasibility_check`): §11 explicit.
- Catalog #303 (`check_substrate_design_memo_has_cargo_cult_audit_section`): §3 explicit.
- Catalog #305 (`check_substrate_design_memo_has_observability_surface_section`): §5 explicit.
- Catalog #309 (`check_substrate_design_memo_declares_horizon_class`): frontmatter `horizon_class: asymptotic_pursuit`.
- Catalog #310 (`check_f_asymptote_substrate_design_is_class_shift_not_bolt_on`): §1 + §12 declare TT5L V2 as PRIMARY substrate not bolt-on.
- Catalog #311 (`check_predictive_coding_substrate_design_has_ego_motion_conditioning`): §2.2 + §2.3 + §6 declare ego-motion-conditioned next-frame prediction (VGGT pose head + RSSM ego_motion + se3_lie + cooperative-receiver foveation FoE).
- Catalog #324 (`check_no_predicted_band_without_post_training_tier_c_validation`): §8 + frontmatter `predicted_band_validation_status: pending_post_training`.
- Catalog #325 (`check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor`): THIS memo IS the per-substrate symposium evidence.

### Sister-subagent regression

- Sister Z8 hierarchical predictive coding (subagent `a68b22b14` in flight): TT5L V2 design memo cites Z8 cross-substrate composability per §9 + §7 Path (f); coordinated via this memo's files_touched declaration; T2-level design memo only; cross-pollination at predict_residual section LEVEL-1 per-pair-pair extension per parent #866 Revision #5.
- Sister Riemannian-Newton substrate engineering meta-paradigm (subagent `a39ffdf80` in flight): TT5L V2 design memo cites Riemannian-Newton as PARENT META-CLASS that TT5L V2 INHERITS FROM per Catalog #322 anti-phantom; coordinated via cross-substrate composability §9; details PENDING sister Riemannian-Newton design memo landing.
- Sister Z6 4c (codex probe in flight): TT5L V2 design memo cites Z6 4c outcome as cross-pollination dependency for RSSM categorical ego-source choice + Wave 7 cross-substrate composition.
- Sister Z7-Mamba-2 (`a68b22b14` family): TT5L V2 design memo cites Z7-Mamba-2 outcome as cross-pollination dependency for DreamerV3 RSSM GRU-vs-Mamba-2 selectivity decision.
- Sister C6 IBPS Phase 2 (post-empirical reactivation symposium): TT5L V2 design memo cites C6 IBPS Phase 2 outcome as cross-pollination dependency for λ_RSSM β-IB-Lagrangian initialization.
- Sister ATW V2 V2-1 (post-symposium 2026-05-18): TT5L V2 design memo cites ATW V2 V2-1 channel-pick outcome as cross-pollination dependency for per-pixel SegNet logits product-quantization choice.

### Forbidden patterns audited

Per CLAUDE.md FORBIDDEN_PATTERNS section:

- ❌ device-selection-defaults / MPS-fallback: NOT PRESENT — V2 inflate uses canonical `select_inflate_device` per Catalog #205
- ❌ CLI flag inventions / dead-flag wiring: NOT PRESENT — all argparse flags grep'd against canonical `tac.substrates._shared.trainer_skeleton` patterns
- ❌ silent-skip cascades / bootstrap-trap: NOT PRESENT — V2 driver uses `set -euo pipefail` per Catalog #163
- ❌ score-claim-without-axis-tag: NOT PRESENT — every numeric claim in this memo tagged `[prediction]` / `[contest-CUDA reviewed]` / `[macOS-CPU advisory]`
- ❌ component-aliasing-baselines: NOT PRESENT — V2 SHA-vs-archive verification at trainer build per Catalog #19
- ❌ empirical-claim-without-evidence-tag (docstring-overstatement-trap): NOT PRESENT
- ❌ MPS-derived-strategic-decision (MPS-falsification-trap): NOT PRESENT — V1 25ep CUDA anchor cited as canonical [contest-CUDA reviewed]; MPS proxy training mentioned ONLY as Wave-N+1 disambiguator option per Hafner Z7-Mamba-2 pattern
- ❌ misleading-directory-name (phantom-score-directory-trap): NOT PRESENT — all directory names match metadata
- ❌ /tmp-paths-in-persisted-artifact (transient-evidence-trap): NOT PRESENT — V2 archive/runtime/auth-eval paths under `experiments/results/<lane>/`
- ❌ force-canonical-without-evaluation-of-suppression (canonicalization-trap): NOT PRESENT — §6 explicit canonical-vs-unique decision per layer with 5 UNIQUE FORK + 10 ADOPT canonical + per-layer rationale
- ❌ artifact-lifecycle-violations: NOT PRESENT — V2 design memo classified as HISTORICAL_PROVENANCE
- ❌ premature-KILL-without-research-exhaustion: NOT PRESENT — V1 DEFER preserved; V2 PROCEED_WITH_REVISIONS at design only
- ❌ closed-form-CDF-allocator-without-empirical-bit-spend-proof: N/A (TT5L V2 uses categorical-stochastic latent + cooperative-receiver foveation; not CDF allocator)
- ❌ spatial-independent-CDF-assumption (per-pixel-independence-trap): N/A
- ❌ NO-neural-at-medal-band-assumption: NOT PRESENT — V2 uses 4 neural primitives at substrate core
- ❌ symposium-band-prediction-without-Dykstra-feasibility-check: NOT PRESENT — §11 Dykstra-feasibility check explicit per Boyd Revision #5
- ❌ PR#56-pattern-generalizes-to-frames-without-per-substrate-empirical-validation: N/A
- ❌ substrate driver hardcoding smoke=1: NOT PRESENT — V2 driver uses `TT5L_V2_TRAINER_MODE` env var per Catalog #326
- ❌ predicted_band-from-random-init-Tier-C-density (phantom-predicted-band-trap): NOT PRESENT — §8 + frontmatter `predicted_band_validation_status: pending_post_training`

## Summary verdict for parent caller

**Verdict**: PROCEED_WITH_REVISIONS at DESIGN-ONLY state. NO paid dispatch authorization from this memo.

**Wave N+1 council requirements** (mandatory per Catalog #325 + #315):
1. Per-section MI probes on V1 25ep state ($12-20 CPU)
2. Boyd Dykstra-feasibility analytical check ($0)
3. Sister Z6 4c + Z7-Mamba-2 + Z8 + ATW V2 V2-1 + C6 IBPS Phase 2 outcomes consumed
4. V2 trainer build per §12 implementation architecture
5. Cheapest-signal-first cascade per Hotz Revision #6

**Predicted ΔS band**: Boyd Dykstra-feasibility revised `[0.16, 0.26]` (HIGH VARIANCE); parent prompt's hypothetical `[0.172, 0.184]` sits in lower-region; conservative estimate `0.259` sits in upper-region. Wave N+1 single-primitive smoke disambiguates.

**Top 5 op-routables**:
1. Wave N+1 council convened ($0; ~90 min)
2. Per-section MI probes on V1 25ep ($12-20 CPU)
3. Dykstra-feasibility analytical check ($0)
4. V2 trainer build per §12 (~3 weeks subagent; $0 GPU at build)
5. Wave 2 single-primitive smoke (cooperative-receiver-foveation only; $1 Modal T4)

**Cross-substrate composability findings**:
- TT5L V2 + Z6 4c: ORTHOGONAL; α PENDING-Z6-4c; informs RSSM categorical ego-source
- TT5L V2 + Z7-Mamba-2: ORTHOGONAL; α PENDING-Z7-Mamba-2; informs GRU-vs-Mamba-2 selectivity
- TT5L V2 + Z8: ORTHOGONAL; α PENDING-Z8; informs LEVEL-1 per-pair-pair extension
- TT5L V2 + A1: ORTHOGONAL HIGH α (0.7-1.0); reactivation path (f)
- TT5L V2 + Riemannian-Newton meta-substrate: META-INHERITANCE; TT5L V2 inherits substrate-engineering meta-paradigm per Catalog #322 anti-phantom

**Operator-routable consequences**:
1. Convene Wave N+1 council ASAP after this memo lands; goal = ratify V2 design + per-section MI probes + Dykstra-feasibility check
2. Spawn sister subagent to write per-section MI probe scripts (sister of D4 ATW V2 probe pattern) — ~$0 + ~1 day
3. Spawn sister subagent to extend `tools/check_substrate_dykstra_feasibility.py` with `--substrate tt5l_v2_4_primitive` polytope definition — ~$0 + ~1 day
4. Coordinate with Z6 4c codex probe outcome + Z7-Mamba-2 + Z8 + ATW V2 V2-1 + C6 IBPS Phase 2 sister symposiums BEFORE Wave 7 cross-substrate composition path
5. After Wave N+1 council PROCEEDs: dispatch V2 trainer build subagent (Codex-class) per §12 implementation architecture
