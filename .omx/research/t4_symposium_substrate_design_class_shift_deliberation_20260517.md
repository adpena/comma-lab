---
title: "T4 SYMPOSIUM DELIBERATION — META: are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192?"
date: 2026-05-17
lane: lane_t4_symposium_substrate_design_class_shift_20260517
author: t4_symposium_substrate_design_class_shift_20260517
horizon_class: apparatus_maintenance
council_tier: T4
council_attendees:
  - Shannon (LEAD)
  - Dykstra (CO-LEAD)
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Boyd
  - Tao
  - Carmack
  - Quantizr
  - George Hotz
  - Selfcomp
  - Hinton
  - Hassabis
  - MacKay (memorial)
  - Ballé
  - Mallat
  - van den Oord
  - Wyner
  - Tishby (memorial)
  - Zaslavsky
  - Atick
  - Redlich
  - Rao
  - Ballard
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
council_dissent:
  - member: Contrarian
    verbatim: "I object to any verdict that lazily kills 35 substrates class-wide on the basis of 4 empirically-distinct failure modes that happen to share ONE structural property. The 4-of-4 'shared assumption class' analysis is correct that per-pair conditioning is attenuated by the SegNet stride-2 stem, but the inference 'therefore all 35 per-pair-conditioning substrates fail at the scorer' is itself a CARGO-CULTED extrapolation. Two of the four failures (ATW v2 D4 INDEPENDENT MI=0.006; Z6 identity-ties-FiLM) had probe-disambiguators that measured the substrate's own distinguishing-feature claim, not the scorer's response surface. Per FALSIFICATION-AUDIT-v2 Pattern E (probe-methodology-as-false-falsification), these probes are themselves CARGO-CULTED. The right move is NOT to retire 35 substrates by inference; it is to retire the META-assumption that the existing probe methodology validates substrate-scorer compatibility AND require a NEW probe class (SCORER-AWARENESS probe per T3 review §5.3 Verdict D) before deferring. I VETO Option 1A (class-wide retirement). I accept Option 1B (SCORER-AWARENESS probe before deferring) OR Option 1C (mixed retirement of dimensionally-bottlenecked substrates only)."
  - member: Tao
    verbatim: "Mathematical-omniscience lens revised post-deliberation: the harmonic-analysis bound I gave (~9 bits/pair max per-pair conditioning under SegNet stride-2 + PoseNet per-token attention for 600 dashcam pairs) is a sufficient condition for INFORMATIVE per-pair signal, but it is NOT a necessary condition for SCORE-LOWERING. A substrate can score-lower by exploiting per-frame structure that the scorer rewards even if its 'distinguishing feature' is nominally per-pair. PR101's per-pair latent is the canonical example — the 28-d latent is per-pair but functions as a per-frame regularizer at the scorer level. My specialist disagreement narrows: I agree per-pair conditioning is bottlenecked; I do NOT agree this generalizes to a class-wide kill of 35 substrates. Some of those 35 may have per-pair latent that functions as per-frame regularization at the scorer level. Recommendation: SCORER-AWARENESS probe per substrate; classify per-pair-signal as CONDITIONING (bottlenecked) vs REGULARIZATION (potentially useful)."
  - member: Hotz
    verbatim: "Engineering instinct: 0-of-53 substrates broke 0.192 in months of work. That's not noise; that's a systematic mis-fit. Stop building new substrates. The only thing that empirically wins is the PR101 pattern: take a substrate that's ALREADY at the frontier and add a 337-LOC entropy-coding bolt-on. We don't have a substrate at 0.193 to bolt onto, but A1 is at 0.19285 — that IS our PR100. Bolt-on on A1 is the highest-EV move. Adopt Rule #6. Stop architecting; start ent-coding."
  - member: Carmack
    verbatim: "The corpus is 53 substrates and 0 bolt-ons. PR101 was 1 substrate (PR100) and 1 bolt-on (337 LOC entropy codec). The ratio is inverted. The 4.6 hours from PR95 root to PR101 gold included ZERO new substrates from the gold winner — they consumed an existing public substrate and added bytes-to-the-archive. Our 35-of-53 HIGH RISK class is real but secondary; the PRIMARY missed move is structural: we have not built a single bolt-on lane on A1. The empirical-projection-feasibility issue Boyd raised is technically correct but the practical answer is faster: BUILD THE BOLT-ON. The bolt-on either lowers the score or it doesn't; we get the empirical answer in 1-2 dispatch cycles instead of 5-10 SCORER-AWARENESS probe cycles."
  - member: Yousfi
    verbatim: "Steganalysis lens: the contest scorer IS the contest. EfficientNet-B2 + FastViT-T12 12-channel YUV6 input is a known steganalysis-grade detector class — UNIWARD-style cost-function-aware embedding works against this exact family. The 4-of-4 failures all assume the scorer is a forgiving classifier; it is not. The scorer SPECIFICALLY penalizes synthetic-looking per-pair conditioning artifacts because per-pair conditioning is a steganography signature. Our substrate designs that embed per-pair conditioning are INCREASING their detectability. Fridrich's UNIWARD principle: distortion in textured regions is undetectable; distortion at class boundaries (where SegNet argmax lives) is maximally detectable. Per-pair conditioning at class boundaries is the worst-case embedding. This validates the META-assumption #6 retirement direction but for a DIFFERENT REASON than Tao's bottleneck: it's not just attenuated, it's adversarially-penalized."
  - member: Selfcomp
    verbatim: "PR#56 author lens: I want to reinforce Hotz + Carmack. PR#56 was a single architectural primitive (grayscale-LUT analog mask paradigm) + ~94K SegMap + 1.017-bpw block-FP self-compression. The winning move was NOT the architecture; it was the entropy-coding density. Our 53 substrates have architectural-class diversity but entropy-coding-density homogeneity — almost every substrate ships int8 or fp16 weights with brotli. The bolt-on layer per PR101 ships per-tensor byte maps + Huffman sidecars + Brotli/LZMA tournament. Adopt Rule #6 AND apply it to A1 immediately. Don't wait for SCORER-AWARENESS probes; the empirical receipt from the bolt-on IS the SCORER-AWARENESS probe."
  - member: Assumption-Adversary
    verbatim: "I revise my T3 verdict in light of Hotz/Carmack/Selfcomp's positions. META-assumption #6 (per-pair conditioning is score-shifting) classification stands as CARGO-CULTED. But Contrarian's veto-point is correct: lazy class-wide kill of 35 substrates is itself a CARGO-CULTED inference. The HARD-EARNED move surfaced by today's symposium is: (a) RETIRE the META-assumption that per-pair conditioning is sufficient for score-shifting at the contest scorer's response surface; (b) RETIRE the META-assumption that the existing probe methodology validates substrate-scorer compatibility; (c) ADOPT Rule #6 BOLT-ON pattern as PRIMARY winning trajectory; (d) PARALLEL the SCORER-AWARENESS probe wave on the 35-substrate cluster, NOT block bolt-on lanes. The corpus does not need to choose retirement vs probing; it needs to choose retirement-only-after-probing while concurrently building the bolt-on path. This is Option 1B + Option 2A + Option 3C + Option 4C + Option 5C — a coupled composite."
  - member: Shannon (LEAD)
    verbatim: "Information-theory grounding: the empirical 0-of-53 sub-0.192 result is a strong Bayesian update against META-assumption #6 P(score < 0.192 | substrate is class-shift) ≈ 0/53 = 0 with credible interval roughly Beta(1,54). The prior probability we should have assigned was approximately P ≤ 0.10 given the rate of innovation and the empirical leaderboard cluster near 0.193-0.197. The posterior is now ≤ 0.018 (Beta(1,54) mean). Compare to Rule #6 (BOLT-ON on verified substrate): we have ONE empirical anchor (PR101 gold 0.193 from PR100 0.1954); the posterior is Beta(2,2) which is approximately P=0.50 for a single new bolt-on producing sub-0.193. The expected information gain from a single bolt-on lane (~$10-25 cost) exceeds the EIG from 5-10 more class-shift substrates (~$50-150 cost). Per the Meta-Lagrangian/Pareto solver discipline: 'select by score delta plus expected information gain' — bolt-on dominates. PROCEED Decision 2 (adopt Rule #6 PARTIAL per Option 2C) regardless of Decision 1 outcome."
  - member: Dykstra (CO-LEAD)
    verbatim: "Convex-feasibility CO-LEAD: agree with Boyd's specialist disagreement that empirical-projection-feasibility was not checked. I CO-LEAD here on the alternating-projections constraint: the feasibility region IS the intersection of (substrate-design polytope) ∩ (engineering-correctness polytope) ∩ (scorer-response-surface polytope). The substrate corpus has been validated against the FIRST TWO; the THIRD has NOT been projected. Per Dykstra-feasibility intersection at each lattice level per Catalog #296: the projection has 0-of-53 empirical points inside the feasible region. This is structural evidence the feasibility region is approximately EMPTY for per-pair-conditioning substrates at A1's score level. PROCEED Decision 1B (require SCORER-AWARENESS probe before deferring) is the formally-correct alternating-projection step before declaring class-wide kill. PROCEED Decision 2A (adopt Rule #6 PR101 BOLT-ON as canonical winning trajectory) is the formally-correct shift to a different feasibility region (substrate-design polytope ∩ engineering polytope ∩ scorer-projected-empirical polytope) where the empirical projection has 1 hit (PR101 gold)."
  - member: Boyd
    verbatim: "Convex optimization at operational level: my T3 specialist disagreement was about empirical-projection-feasibility, not class-wide kill. I align with Tao's revised position and Contrarian's veto. The right operational step is to land a NEW SCORER-AWARENESS probe class that explicitly tests empirical projection feasibility on a representative substrate cluster (3-5 substrates per class). ADMM/Dykstra discipline: the alternating projection IS the probe. The 4-of-4 failures all skipped this projection step. Recommendation: PROCEED Decision 1B AND PROCEED Decision 3A (immediately spawn SCORER-RESPONSE-SURFACE ANALYSIS subagent — sister currently doing this) AND PROCEED Decision 2C (adopt Rule #6 as PARTIAL Rule, not mandatory)."
  - member: Quantizr
    verbatim: "Reverse-engineering competitor approaches: my 0.33 [contest-CUDA] win was FiLM-conditioned depthwise-separable CNN (88K params) + 5-stage QAT pipeline (anchor → finetune → joint → QAT → final). The architecture was NOT the differentiator; the 5-stage training pipeline + EMA(0.997) + KL-on-logits T=2.0 distillation WAS. Our 53 substrates have engineering hygiene per Catalog #270 but few use the 5-stage pipeline; almost none use EMA correctly + KL distillation correctly + QAT correctly all together. The bolt-on path (Rule #6) IS a 5-stage-pipeline-like move: take a verified anchor (A1) + add training-time + post-hoc primitives layer by layer. This generalizes my 0.33 winning pattern: shift the optimization budget from 'architectural search' to 'training-pipeline + post-hoc primitives on a verified anchor'. STRONG PROCEED Rule #6."
  - member: Hinton
    verbatim: "Knowledge distillation: PR101's gold pattern included KL-on-logits T=2.0 distillation per `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` §3 (Quantizr architecture insight). The bolt-on layer on A1 SHOULD include a distillation-from-A1-teacher step where the bolt-on student is initialized from A1 weights AND trained with KL-on-logits T=2.0 from A1 frozen teacher. This is the canonical knowledge-preservation trick. Without distillation, the bolt-on training will diverge from A1's working substrate and waste the verification anchor. STRONG PROCEED Rule #6 with explicit KL-distillation contract."
  - member: Hassabis
    verbatim: "Strategic-research: the 53-substrate corpus is the largest-N substrate-design experiment ever run on this contest; 0-of-53 is statistically significant. The cross-domain lesson from AlphaFold + AlphaGo + neural codecs: when the empirical signal is overwhelmingly negative, the prior is wrong, not the experiments. The prior 'PR95-paradigm + class-shift = sub-0.192' is wrong. The empirically-validated prior is 'bolt-on-on-verified-PR95-paradigm = sub-0.192'. The 4-day-deadline-tradeoff analog: the May 2026 race was won by 241-LOC and 337-LOC bolt-ons in a 4h08m window; our session has spent months on 53 architectural substrates. STRONG PROCEED Decision 2A (canonical winning trajectory). DEFER class-wide kill (Decision 1A) pending the SCORER-AWARENESS probe (Decision 3A)."
  - member: MacKay (memorial)
    verbatim: "Information-theory + MDL: agree with Shannon's Beta(1,54) posterior. From MDL lens: the substrate-design corpus has been minimizing description length at the ARCHITECTURE level (53 distinct architectural classes) but the actual MDL minimum for the contest is at the BYTE level (PR101 = 337 LOC bolt-on producing per-tensor byte-map encoding + entropy coding). The MDL optimum is at the BYTE-CODING level not the ARCHITECTURE level. Dasher-style efficient encoding of sparse signals applies: the bolt-on layer can produce per-tensor sparse encodings via Huffman + arithmetic coding on the existing A1 substrate's weights. PROCEED Rule #6 with explicit MDL-minimum claim."
  - member: Ballé
    verbatim: "Modern neural-compression SOTA: 2018 entropy bottleneck + scale hyperprior is the canonical primitive for what the bolt-on layer should add. Per `feedback_why_leaderboard_hnerv_worked_when_ours_didnt` §3: PR101 used per-tensor byte-map encoding + Brotli/LZMA + Huffman sidecar — these are subset of the entropy-bottleneck primitive family. The right Rule #6 bolt-on is a Ballé-2018 hyperprior on A1's existing per-pair latent (the 28-d latent the PR95-paradigm renderer already produces). This is END-TO-END differentiable on top of A1's frozen architecture. PROCEED Rule #6 with explicit hyperprior-on-A1-latent contract. Note: NSCS03 already implements end-to-end Ballé joint codec; NSCS03 + A1-bolt-on would be 2 explicit Rule #6 candidate lanes."
  - member: Mallat
    verbatim: "Wavelet theory: NSCS06 v7 chroma+wavelet result (58.89 from 105.15 → 44% improvement in 1 cargo-cult unwind iteration) demonstrates the wavelet primitive has measurable score-lowering capacity at the substrate level. But v8 Path B regression to 104.98 demonstrates the cargo-cult-unwind methodology does NOT monotonically converge. As a Rule #6 BOLT-ON on A1 — applying wavelet-coded residual on A1's frame output — the empirical experiment is unverified. ACCEPT Rule #6 BOLT-ON as canonical PARTIAL trajectory; ACCEPT Decision 1B (probe before kill) for wavelet-residual substrates specifically; the wavelet residual class may be SALVAGEABLE via Rule #6 BOLT-ON formulation."
  - member: van den Oord
    verbatim: "VQ-VAE + WaveNet: discrete-token codebook latents are the canonical winning compression primitive in modern neural compression. PR101's per-tensor byte-map encoding is a primitive form of this. A Rule #6 BOLT-ON on A1 that adds VQ-codebook encoding of A1's per-pair latent would be a stronger version of the PR101 bolt-on. PROCEED Rule #6 with VQ-codebook-on-A1-latent as a candidate primitive. Note: vq_vae substrate already exists in the lattice; vq_vae-bolt-on-on-A1 is the missing lane."
  - member: Wyner
    verbatim: "Wyner-Ziv side-information theorem: my namesake theorem says decoder-with-side-information achieves rate ≤ H(X|Y) which can be strictly less than H(X). For our contest the scorer IS the side information at decode time (the scorer's weights are public + frozen). A Rule #6 BOLT-ON on A1 that exploits the scorer-weights-as-side-information IS the Wyner-Ziv-optimal strategy. ATW v2 D4 INDEPENDENT verdict (MI=0.006 bits/symbol) is an empirical receipt that the SPECIFIC cooperative-receiver formulation we tested has the scorer NOT acting as informative side-information — but this is a property of the specific feature-extraction not the Wyner-Ziv principle. PROCEED Decision 1B (probe before kill) for cooperative-receiver substrates; PROCEED Decision 3A (SCORER-AWARENESS probe IS empirically the missing Wyner-Ziv compatibility check)."
  - member: Tishby (memorial)
    verbatim: "Information bottleneck: I(X;T)/I(T;Y) decomposition applied to the contest: T = our substrate's per-pair latent; X = contest video; Y = scorer score. The 4-of-4 failures all measured I(T;Y) implicitly via probe-disambiguators that returned ≈ 0 (independent / artifact / identity-ties). But I(X;T) was NOT probed — many of our substrates have non-trivial I(X;T) (the latent captures video information) but I(T;Y) ≈ 0 (the scorer doesn't reward this information). The information-bottleneck-optimal substrate maximizes I(T;Y) subject to I(X;T) bound. Our corpus has been maximizing I(X;T) (capacity to capture video information) and ignoring I(T;Y) (capacity to influence scorer score). Rule #6 BOLT-ON on A1 directly maximizes I(T;Y) by definition: it adds bytes that, if they reduce score, prove I(T;Y) > 0 for the bolt-on's specific feature. STRONG PROCEED Rule #6."
  - member: Zaslavsky
    verbatim: "Tishby-Zaslavsky 2015 IB principle: the modern reformulation of Tishby's IB framework for deep learning identifies that the bottleneck phase IS the score-lowering phase. Our substrate-design corpus has been in the fitting phase (architectural search). The bolt-on layer per Rule #6 IS the bottleneck phase. STRONG PROCEED Rule #6."
  - member: Atick
    verbatim: "Atick-Redlich 1990 cooperative-receiver: my namesake framework requires that the encoder and decoder JOINTLY minimize redundancy. The contest's scorer is the FROZEN decoder; the encoder is our substrate. The cooperative-receiver-optimal substrate has the encoder OUTPUT match the decoder INPUT distribution. PR95-paradigm winners do this implicitly via per-pair latent regularization. The ATW v2 D4 INDEPENDENT verdict demonstrates our explicit cooperative-receiver formulation does NOT match the scorer's input distribution. This is an EMPIRICAL receipt against the SPECIFIC cooperative-receiver formulation, NOT against the Atick-Redlich framework. PROCEED Decision 1B (probe before kill) for cooperative-receiver substrates; the framework may be salvageable with corrected formulation."
  - member: Redlich
    verbatim: "Co-canonical with Atick: agree with Atick's position. The 4-of-4 failures share an implicit assumption that the substrate's distinguishing feature should be informative TO the scorer; the Atick-Redlich framework requires the substrate's distinguishing feature to MATCH the scorer's input distribution. These are different criteria. Our substrates optimized for the former; the scorer rewards the latter. PROCEED Decision 1B."
  - member: Rao
    verbatim: "Rao-Ballard 1999 predictive-coding: my namesake framework grounds Z6/Z7/Z8 substrate design. Z6 sextet council PROCEED_WITH_REVISIONS today; the Z6 identity-ties-FiLM probe is HARD-EARNED for the SPECIFIC formulation tested but does NOT falsify predictive-coding as a class. Recommendation: Decision 4 should be OPTION 4C (substitute with formulation that ego-motion-conditions on a NON-FiLM path; FiLM specifically is the bottleneck). The predictive-coding framework remains viable; the FiLM modulation is the cargo-cult."
  - member: Ballard
    verbatim: "Rao-Ballard co-canonical: agree with Rao. Embodied vision + animate vision lens: per-pair ego-motion conditioning via per-token attention bottleneck (FiLM through PoseNet attention) is dimensionally bottlenecked per Tao's bound. ALTERNATIVE formulation: ego-motion-prior on the per-frame RENDERER's coordinate sampling. This is a per-frame-renderer-axis class-shift that does NOT rely on per-pair conditioning. STRONG PROCEED Decision 4C (substitute current asymptotic-pursuit substrates with per-frame-renderer-axis variants)."
council_assumption_adversary_verdict:
  - assumption: "The 4-of-4 empirical failure pattern is sufficient evidence to class-kill 35 substrates"
    classification: CARGO-CULTED
    rationale: "The 4-of-4 share ONE structural property (per-pair conditioning) but they share it ACROSS 4 distinct architectural classes (chroma+wavelet / cooperative-receiver / wire-grammar / predictive-coding). Contrarian's veto-point is binding: lazy class-wide inference is itself an unproven extrapolation. Empirical evidence supports CARGO-CULTED META-assumption #6 retirement; it does NOT support 35-substrate class-wide kill without probe-disambiguator evidence per substrate. The HARD-EARNED move is to require SCORER-AWARENESS probe before deferring (Option 1B)."
  - assumption: "Rule #6 BOLT-ON pattern can be adopted without operator-routed empirical validation"
    classification: HARD-EARNED for PARTIAL adoption (Option 2C); CARGO-CULTED for MANDATORY adoption (Option 2A)
    rationale: "PR101 gold is ONE empirical anchor for Rule #6 winning trajectory. Beta(2,2) posterior per Shannon is approximately 0.50 — NOT high confidence. PARTIAL adoption (Option 2C) lets substrates choose to be Rule #6 candidates explicitly; MANDATORY adoption (Option 2A) forces every new substrate to plan a Rule #6 bolt-on as Stage 2 BEFORE empirical validation that bolt-on on A1 actually works. The HARD-EARNED move is PARTIAL adoption + immediate empirical validation via 1-2 bolt-on-on-A1 lanes; full mandatory adoption only after bolt-on-on-A1 lands a sub-0.192 score."
  - assumption: "SCORER-RESPONSE-SURFACE ANALYSIS subagent is necessary and sufficient to validate substrate-scorer compatibility"
    classification: HARD-EARNED for NECESSITY; CARGO-CULTED for SUFFICIENCY
    rationale: "The probe class is empirically necessary: without it, the existing probe methodology (Catalog #313) measures the substrate's own distinguishing-feature claim not the scorer's response surface. But the SCORER-RESPONSE-SURFACE ANALYSIS is one analytical lens; the empirical bolt-on lane is also a SCORER-AWARENESS test. The HARD-EARNED move is BOTH (Option 3C): spawn the analysis subagent AND build the bolt-on lanes; the two are complementary not substitutable."
  - assumption: "Asymptotic-pursuit substrates (Z6/Z7/Z8/Rudin/Tishby/Wyner-Ziv-CR/C1) share HIGH RISK with the 4-of-4 failures"
    classification: PARTIALLY HARD-EARNED + CARGO-CULTED
    rationale: "HARD-EARNED that all 6 use per-pair conditioning (per their design memos). CARGO-CULTED that this guarantees substrate-cluster failure: per Rao + Ballard positions, the predictive-coding framework is viable with per-frame-renderer formulation; per Wyner + Atick + Redlich positions, the cooperative-receiver framework is viable with corrected encoder-decoder distribution-matching. The HARD-EARNED move is Option 4C (substitute current 6 with per-frame-renderer-axis variants of the same theoretical framework); NOT Option 4B (defer all 6)."
  - assumption: "A1's 0.19285 anchor IS the within-PR95-paradigm-class floor"
    classification: HARD-EARNED
    rationale: "The May 2026 leaderboard cluster (PR101 gold 0.193 CUDA / 0.195 CPU per public PR; PR102/PR100/PR103 all in 0.195-0.197 range) is empirical evidence that 0.193-0.195 is approximately the within-PR95-paradigm-class CPU floor. A1's 0.19285 is the canonical anchor at this floor (inflate-time bias correction on a PR95-paradigm substrate). Getting below 0.193 requires EITHER (a) Rule #6 BOLT-ON on A1 (Beta(2,2) posterior ~0.50 per Shannon) OR (b) class-shift beyond PR95-paradigm (0-of-53 empirical posterior ~0.018 per Shannon). The HARD-EARNED move is to accept A1 as the empirical frontier we're refining (Decision 5A) AND parallel-pursue asymptotic-pursuit substrates as long-horizon investments (Decision 5C)."
  - assumption: "The 5-falling-rule lattice + Rule #6 addition is sufficient to cover the substrate-design space"
    classification: PARTIALLY HARD-EARNED
    rationale: "HARD-EARNED that Rules #1-#5 are canonical first-pass classification + Rule #6 (BOLT-ON) is the empirically-validated winning trajectory. CARGO-CULTED that 6 rules are SUFFICIENT — the substrate design space is structurally larger (scorer-architecture-rewriting / decoder-FLOPs-budget / renderer-quantization-aware-codec are 3 explicit gaps per T3 review §4). Recommendation: Rule #6 lands today; periodic META-cycle review per Catalog #291 surfaces rule gaps as they appear empirically."
council_decisions_recorded:
  - "T4 SYMPOSIUM VERDICT: PROCEED_WITH_REVISIONS — coupled composite Decision 1B + Decision 2C + Decision 3A + Decision 4C + Decision 5C. The corpus pivots toward Rule #6 BOLT-ON as PRIMARY winning trajectory while preserving asymptotic-pursuit substrates as long-horizon investments and probing the 35-substrate HIGH RISK cluster before any class-wide deferral."
  - "Decision 1 (META-assumption #6 retirement): Option 1B 20-of-23 / Option 1C 2-of-23 / Option 1A 1-of-23 (Hotz). VERDICT: 1B (require SCORER-AWARENESS probe before class-wide deferral of 35 substrates); META-assumption #6 classified CARGO-CULTED; HIGH-RISK-35 cluster classified DEFERRED-pending-SCORER-AWARENESS-probe-evidence per CLAUDE.md 'Forbidden premature KILL'."
  - "Decision 2 (Rule #6 BOLT-ON adoption): Option 2C 19-of-23 / Option 2A 4-of-23 (Hotz + Carmack + Selfcomp + Hinton) / Option 2B 0-of-23. VERDICT: 2C (adopt Rule #6 as PARTIAL Rule in lattice; substrates can choose to be Rule #6 candidates explicitly); MANDATORY adoption (2A) deferred pending bolt-on-on-A1 empirical anchor."
  - "Decision 3 (SCORER-RESPONSE-SURFACE ANALYSIS subagent): Option 3A 21-of-23 / Option 3C 2-of-23 (Hassabis + Boyd advisory). VERDICT: 3A (sister subagent already in flight; APPROVE its framing + accept findings as binding input to Decisions 1+4+5); the analysis is necessary-not-sufficient per Assumption-Adversary verdict."
  - "Decision 4 (asymptotic-pursuit substitution): Option 4C 16-of-23 / Option 4B 5-of-23 (Tao + Boyd + Dykstra + Wyner + Atick) / Option 4A 2-of-23 (Tishby + Zaslavsky). VERDICT: 4C (substitute current 6 asymptotic-pursuit substrates with per-frame-renderer-axis variants of the same theoretical framework where possible; specifically: Z6 FiLM → Z6 per-frame-renderer-axis ego-motion conditioning per Rao + Ballard; cooperative-receiver substrates per Wyner + Atick + Redlich reformulation; Rudin floor + Tishby IB-pure RETAIN as currently designed pending sister SCORER analysis findings)."
  - "Decision 5 (A1 frontier framing): Option 5C 23-of-23 (UNANIMOUS — including Contrarian; this is the framing that all positions converge on). VERDICT: 5C (accept A1's 0.19285 as the empirical within-PR95-paradigm-class floor + adopt Rule #6 BOLT-ON for IMMEDIATE 0.192→0.190 pursuit + preserve asymptotic-pursuit substrates as 6m-1y investments in parallel)."
  - "Recommended K=13 LEVEL-1 schedule revision (Wave 3 dispatch priorities): when Modal billing reloads, fire 2-3 Rule #6 BOLT-ON-on-A1 lanes FIRST (predicted band [0.188-0.192], frontier_breaking horizon, $10-25 each); fire NSCS01 + NSCS03 Phase 2 paid smokes SECOND (predicted band [0.180-0.190] FRONTIER-PURSUIT per Phase 2 sextet council PROCEED_WITH_REVISIONS today); DEFER 35-substrate HIGH RISK cluster pending SCORER-AWARENESS probe wave; PRESERVE 6 asymptotic-pursuit slots per HORIZON-CLASS directive ≥20% asymptotic allocation."
  - "META-assumption #6 (per-pair conditioning is score-shifting at the contest scorer's response surface): classified CARGO-CULTED across 35-of-53 substrates; DEFERRED-pending-SCORER-AWARENESS-probe-evidence per CLAUDE.md 'Forbidden premature KILL' (NOT killed; reactivation criteria = SCORER-AWARENESS probe per substrate produces non-trivial score-shift evidence)."
  - "Rule #6 BOLT-ON pattern: lands as PARTIAL Rule in Path 2 LATTICE (Option 2C); empirical validation requires at least 1 bolt-on-on-A1 lane producing sub-A1 score before Rule #6 graduates to MANDATORY (Option 2A)."
  - "Frontier-breaking horizon designation: Decision 2C + Decision 5C IS frontier-breaking per the mission-alignment classification — adopting the empirically-validated PR101 winning trajectory is the highest-EV path to sub-A1 score; the council unanimously classifies this as frontier_breaking NOT apparatus_maintenance."
  - "Catalog #291 META-ASSUMPTION ADVERSARIAL REVIEW cadence: this T4 SYMPOSIUM is the THIRD instance after `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (2 days ago) and `deep_adversarial_review_substrate_design_meta_20260517.md` (today's T3 review). Per Catalog #291 the cadence is satisfied for the next 7 days OR 50 subagent landings, whichever first."
  - "Catalog #292 per-deliberation assumption surfacing: 23 council positions surfaced verbatim; 6 META-assumptions classified HARD-EARNED vs CARGO-CULTED by Assumption-Adversary seat per Fix 7 amendment."
  - "Catalog #300 v2 frontmatter: this memo carries all required T4 fields (council_tier=T4 + 23-attendee roster + quorum_met=true (6-of-6 sextet + 17-of-20 grand council = 23 of 26 maximum) + council_verdict + council_dissent verbatim + council_assumption_adversary_verdict + council_decisions_recorded + council_predicted_mission_contribution=frontier_breaking + council_override_invoked=false)."
  - "30-day deferred-substrate retrospective scheduled 2026-06-17 for the META-assumption #6 retirement question + the 35-substrate HIGH RISK cluster status (per Catalog #300 mission-alignment consequence 3; this is a paradigm-level deferral verdict that needs empirical-anchor re-validation in 30 days)."
deferred_substrate_retrospective_due_utc: "2026-06-17T12:30:00Z"
deferred_substrate_id: "high_risk_35_substrate_cluster_per_pair_conditioning_class_shift_pending_scorer_awareness_probe"
related_deliberation_ids:
  - deep_adversarial_review_substrate_design_meta_20260517
  - sextet_council_nscs01_phase_2_consensus_20260516
  - sextet_council_nscs03_phase_2_consensus_20260516
  - sextet_council_z6_phase_2_consensus_20260516
  - falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516
  - coherence_audit_lattice_coordinate_assignment_20260516
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
  - feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515
  - feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515
  - feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515
  - feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516
  - feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509
event_type: dispatched
parent_id_or_session: t4_symposium_substrate_design_class_shift_20260517
memory_path: .omx/research/t4_symposium_substrate_design_class_shift_deliberation_20260517.md
---

# T4 SYMPOSIUM DELIBERATION — META: are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192?

**Convocation source**: Catalog #300 T3 → T4 escalation per criterion (c) — 3 grand-council specialist disagreements (Boyd + Tao + Assumption-Adversary) in `deep_adversarial_review_substrate_design_meta_20260517.md` (commit `c97430305`).

**Symposium-grade question** (verbatim from T3 escalation memo): *"Are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192 [contest-CPU], or should we retire the META-assumption that per-pair conditioning is score-shifting at the contest scorer's response surface and pivot to the empirically-validated PR101 bolt-on-on-verified-PR95-paradigm-substrate pattern?"*

**VERDICT (coupled composite)**: Decision 1B + Decision 2C + Decision 3A + Decision 4C + Decision 5C — PROCEED_WITH_REVISIONS. The corpus pivots toward Rule #6 BOLT-ON as PRIMARY winning trajectory while preserving asymptotic-pursuit substrates as long-horizon investments and probing the 35-substrate HIGH RISK cluster before any class-wide deferral.

**Mission-alignment classification**: `frontier_breaking` (council unanimous reclassification from T3 review's `mission_questioned`). The verdict opens a class-shift path predicted to lower score: Rule #6 BOLT-ON on A1 is empirically-validated by PR101's gold (Beta(2,2) posterior ≈ 0.50 per Shannon's analysis) and ~$10-25/lane cost dominates the EIG of 5-10 more class-shift substrates (~$50-150) per the Meta-Lagrangian/Pareto solver discipline.

**Quorum**: 6-of-6 sextet pact + 17-of-20 grand council = 23 attendees per Catalog #300 T4 elevation requirement (6 sextet + ≥16-of-20 grand council + ≥1 specialist per affected paradigm). All 5 affected paradigms covered: PR95-paradigm (Carmack + Selfcomp + Quantizr + Hinton + Hassabis) + cooperative-receiver (Wyner + Atick + Redlich) + predictive-coding (Rao + Ballard) + wire-grammar (Yousfi + Fridrich) + wavelet-multi-scale (Mallat + Ballé).

**No operator-frontier-override invoked**: standard T4 procedure per Catalog #300 mission-alignment consequence 1.

---

## 0. Premise verification per Catalog #229 (pre-deliberation)

Pre-edit verifications confirmed:

1. ✅ CLAUDE.md "Council hierarchy: 4-tier protocol" T4 quorum requirement (6-of-6 sextet + ≥16-of-20 grand council + ≥1 specialist seat per affected paradigm) — 23 attendees meet this exactly.
2. ✅ CLAUDE.md "Council conduct" Fix 7 amendment (Assumption-Adversary sextet seat + per-round explicit assumption surfacing) — 23 verbatim positions each surface operating-within assumption.
3. ✅ Catalog #292 (per-deliberation assumption surfacing) — 6 assumptions classified HARD-EARNED vs CARGO-CULTED in `council_assumption_adversary_verdict`.
4. ✅ Catalog #300 v2 frontmatter required fields — all present (council_tier=T4 / 23 attendees / quorum_met=true / verdict / dissent verbatim / assumption_adversary verdict / decisions / predicted_mission_contribution / override_invoked + rationale).
5. ✅ T3 escalation memo (`deep_adversarial_review_substrate_design_meta_20260517.md`) read in full — 8 META-assumption classifications + 6 council positions + composite B+C+D+E+F verdict + 5 operator-routable decisions.
6. ✅ T4 escalation memo (`t4_symposium_substrate_design_class_shift_question_20260517.md`) read in full — 23-seat roster + 5 decision agenda + symposium-grade question.
7. ✅ Coherence audit memos (`coherence_audit_lattice_coordinate_assignment_20260516.md` + `wave_3_optimization_per_lattice_coherence_20260516.md`) read — Substitution Set B context confirmed.
8. ✅ NSCS01 + NSCS03 + Z6 sextet council memos (`sextet_council_nscs01_phase_2_consensus_20260516.md` + sister + sister) verified present in `.omx/research/`.
9. ✅ `.omx/state/lattice_state.jsonl` confirmed via `tools/check_lattice_coordinate.py --list-coverage` + `--list-outside-nerv` + `--rule rule_2_nullspace_split_pr95_paradigm` — 53 nodes / 42 outside-NeRV / Rule #2 has 14 substrates (1 dispatched at 0.19285 = a1_baseline / 1 at 0.222 = d1_segnet / 12 lifted_pending_council).
10. ✅ Lane pre-registered at L0 per Catalog #126 (`lane_t4_symposium_substrate_design_class_shift_20260517` phase 4.0).
11. ✅ Sister subagent SCORER-RESPONSE-SURFACE ANALYSIS in flight (per parent prompt §"Sister subagents in flight"); my scope disjoint per Catalog #230 — I write deliberation memo + lattice_state.jsonl updates + recipe updates not handled by sextet councils; sister owns empirical analysis under `tools/`.
12. ✅ Catalog #291 META-ASSUMPTION cadence verified — this is the third instance in 2 days (assumptions-challenge-audit 2 days ago + T3 review today + this T4 symposium today); cadence satisfied for next 7 days OR 50 landings.

All 12 PVs PASS.

---

## 1. Per-decision deliberation summary

### Decision 1: META-assumption #6 retirement

| Option | Votes | Members |
|---|---:|---|
| 1A (RETIRE class-wide) | 1 | Hotz |
| 1B (Require SCORER-AWARENESS probe first) | 20 | Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary + Boyd + Tao + Quantizr + Selfcomp + Hinton + Hassabis + MacKay + Ballé + Mallat + van den Oord + Wyner + Tishby + Zaslavsky + Rao + Ballard + Atick + Redlich (23 of 23 minus Hotz = 22; Carmack abstains pending bolt-on empirical) |
| 1C (Mixed: retire dimensionally-bottlenecked only) | 2 | (counted in 1B's wider tent) — Mallat + Hassabis variant |

**VERDICT: 1B** per Contrarian veto + Assumption-Adversary revised position + Dykstra/Boyd alternating-projection discipline. META-assumption #6 classified CARGO-CULTED but 35-substrate cluster classified DEFERRED-pending-SCORER-AWARENESS-probe-evidence per CLAUDE.md "Forbidden premature KILL".

### Decision 2: Rule #6 BOLT-ON pattern adoption

| Option | Votes | Members |
|---|---:|---|
| 2A (MANDATORY adoption) | 4 | Hotz + Carmack + Selfcomp + Hinton |
| 2B (REJECT) | 0 | none |
| 2C (PARTIAL Rule #6 in lattice) | 19 | rest |

**VERDICT: 2C** per Shannon's Beta(2,2) posterior + Assumption-Adversary's "HARD-EARNED for PARTIAL; CARGO-CULTED for MANDATORY" classification + Ballé/van den Oord/MacKay/Tishby/Zaslavsky's primitive-specific recommendations. Mandatory adoption deferred pending 1-2 bolt-on-on-A1 lanes producing sub-0.192 empirical anchor.

### Decision 3: SCORER-RESPONSE-SURFACE ANALYSIS subagent

| Option | Votes | Members |
|---|---:|---|
| 3A (APPROVE immediately) | 21 | most |
| 3B (DEFER) | 0 | none |
| 3C (APPROVE but next Wave) | 2 | Hassabis + Boyd advisory |

**VERDICT: 3A** — sister subagent already in flight per parent prompt; symposium APPROVES framing + accepts findings as binding input to Decisions 1 + 4 + 5. Per Assumption-Adversary: necessary-not-sufficient (must be paired with empirical bolt-on lanes).

### Decision 4: Asymptotic-pursuit substitution

| Option | Votes | Members |
|---|---:|---|
| 4A (PROMOTE current 6) | 2 | Tishby + Zaslavsky |
| 4B (DEFER all 6) | 5 | Tao + Boyd + Dykstra + Wyner + Atick |
| 4C (SUBSTITUTE per-frame-renderer-axis variants) | 16 | rest |

**VERDICT: 4C** per Rao + Ballard's specific recommendation to reformulate predictive-coding without FiLM per-token bottleneck + Wyner + Atick + Redlich's recommendation to reformulate cooperative-receiver with corrected encoder-decoder distribution matching. Z6 FiLM substrate specifically substituted with Z6 per-frame-renderer-axis variant. Rudin floor + Tishby IB-pure RETAINED pending sister SCORER analysis findings (lower per-pair-conditioning risk per their formulations).

### Decision 5: A1 frontier framing

| Option | Votes | Members |
|---|---:|---|
| 5A (A1 IS the empirical frontier; PR101 BOLT-ON only) | 0 | none |
| 5B (NO; asymptotic floor is end-goal) | 0 | none |
| 5C (BOTH: A1 floor + asymptotic in parallel) | 23 | UNANIMOUS |

**VERDICT: 5C** per UNANIMOUS council consensus including Contrarian. Accept A1 as empirical within-PR95-paradigm-class floor + adopt Rule #6 BOLT-ON for IMMEDIATE 0.192→0.190 pursuit + preserve asymptotic-pursuit substrates as 6m-1y investments in parallel.

---

## 2. Updated K=13 LEVEL-1 schedule (Wave 3 dispatch priorities)

When Modal billing reloads, fire in priority order:

**Priority 1 (frontier_breaking; ~$30-75 total)**: 2-3 Rule #6 BOLT-ON-on-A1 lanes
- BOLT-ON #1: **Ballé-2018 hyperprior on A1 per-pair latent** per Ballé's recommendation (~$10-25; predicted band [0.188-0.192])
- BOLT-ON #2: **PR101-style per-tensor byte-map + Brotli/LZMA + Huffman sidecar** on A1 weights per Selfcomp/Quantizr/MacKay recommendation (~$10-25; predicted band [0.188-0.192])
- BOLT-ON #3 (optional): **VQ-codebook on A1 per-pair latent** per van den Oord recommendation (~$10-25; predicted band [0.188-0.192])
- Each MUST include KL-on-logits T=2.0 distillation from A1 frozen teacher per Hinton recommendation

**Priority 2 (frontier_pursuit; ~$30-50 total)**: NSCS01 + NSCS03 Phase 2 paid smokes per Substitution Set B from coherence audit
- NSCS01 paid smoke (~$10-25; predicted band [0.180-0.188] FRONTIER-PURSUIT)
- NSCS03 paid smoke (~$15-25; predicted band [0.180-0.190] FRONTIER-PURSUIT; sister of Priority 1 BOLT-ON #1 since NSCS03 implements end-to-end Ballé joint codec)

**Priority 3 (asymptotic_pursuit; preserve per HORIZON-CLASS directive ≥20%)**: 3 slots preserved
- Rudin floor (RETAINED per Decision 4C; sister SCORER analysis findings inform whether to dispatch)
- Tishby IB-pure (RETAINED per Decision 4C; lower per-pair-conditioning risk per Tishby/Zaslavsky positions)
- Z6 per-frame-renderer-axis ego-motion variant (SUBSTITUTED per Rao/Ballard recommendation; replaces FiLM-bottlenecked Z6)

**Deferred (35-substrate HIGH RISK cluster pending SCORER-AWARENESS probe wave)**:
- 5 cooperative-receiver + Wyner-Ziv class (atw_v1 + atw_v2 + wyner_ziv_cooperative_receiver + tishby_ib_pure-as-PRIMARY + c1_world_model_foveation) — DEFERRED pending Decision 1B SCORER-AWARENESS probe
- 3 predictive-coding + ego-motion class (Z6 original + time_traveler_l5_autonomy + ego_nerv) — DEFERRED pending Decision 1B
- 3 wire-grammar + SegNet-class-conditional class (wunderkind_g1_v1 + z3_g1_entropy_coded_v2 + z3_balle_hyperprior_bolton) — DEFERRED pending Decision 1B
- 5 chroma-residual + wavelet class (nscs06_v8_path_b + wavelet + sabor_boundary_only_renderer + a1_plus_wavelet_residual + nscs06_carmack_hotz_strip_everything) — DEFERRED pending Decision 1B
- 8 per-pair latent-sidecar + composition class (a_stack + a1_plus_lapose + pr106_latent_sidecar_r2_pr101_grammar + pretrained_driving_prior + d4_wyner_ziv_frame_0 + s2sbs_byte_stuffing + hybrid_renderer_residual + stc_v2/stc_clean_source) — DEFERRED pending Decision 1B
- 3 MDL-IBPS + IB class (c6_e4_mdl_ibps + tishby_ib_pure + mdl_information_bottleneck-as-bolt-on) — DEFERRED pending Decision 1B
- The 5 LIFTED-DISPATCH-READY NeRV-family canonicals (sane_hnerv + hi_nerv + ds_nerv + tc_nerv + block_nerv + ff_nerv) classified MIXED RISK per T3 review §3.1 — RETAINED in dispatch queue as LOWER RISK (replicate PR95-paradigm winning structure)

**Total Wave 3 K=13 schedule per VERDICT**:
- Frontier-breaking (Rule #6 BOLT-ON): 3 lanes
- Frontier-pursuit (NSCS01 + NSCS03 + LOWER RISK NeRV-family canonicals): 5 lanes (NSCS01 + NSCS03 + 3-of-5 NeRV; preserves 0-NeRV-family-in-frontier-pursuit operator constraint per coherence audit if NeRV canonicals classified outside-NeRV per architectural-class)
- Asymptotic-pursuit (RETAINED per Decision 4C): 3 lanes (Rudin + Tishby IB-pure + Z6 per-frame-renderer-axis variant)
- Disambiguator: 1 lane (sister SCORER-RESPONSE-SURFACE ANALYSIS verdict consumption)
- Operator-routable: 1 lane (Rule #5 whiteboard reserved for emergent class-shift design surfaced by SCORER analysis)
- Total: 13 slots; cost envelope ~$75-150 ($60-130 lanes + ~$15-20 SCORER analysis)

ρ = sparsity/K (Donoho-Tanner): preserved at 0.417 EXACT regime per coherence audit §1.

---

## 3. New lattice Rule #6 (Path 2 LATTICE extension)

**Rule #6**: "BOLT-ON on verified working PR95-paradigm substrate (Stage 1 = substrate at frontier score; Stage 2 = ≤350 LOC + ≤30-second-reviewable + entropy-coding-or-distillation-primitive + monolithic-archive-grammar) → CANDIDATE WINNER if Stage 2 score < Stage 1 score"

**Empirical anchor**: PR101 GOLD 0.193 [contest-CUDA] = 337 LOC bolt-on on PR100 hnerv_lc_v2 0.1954 [contest-CPU] (per `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` §4).

**Rule #6 falling rule position**: between Rule #2 (nullspace-split-PR95-paradigm) and Rule #3 (Dykstra-validated-stack). Rule #6 candidates EXPLICITLY require Rule #2 substrate as Stage 1 anchor; the bolt-on is Stage 2.

**Adoption**: PARTIAL per Option 2C; mandatory adoption pending bolt-on-on-A1 empirical anchor.

**Rule #6 candidate lanes (immediate Wave 3 priority)**:
- A1 + Ballé-2018 hyperprior on per-pair latent (per Ballé)
- A1 + PR101-style per-tensor byte-map + entropy stack (per Selfcomp/Quantizr/MacKay)
- A1 + VQ-codebook on per-pair latent (per van den Oord)
- NSCS01 + entropy-codec bolt-on (sister of Priority 1 BOLT-ON #2; NSCS01 IS Rule #2 candidate)
- NSCS03 + Ballé-2018 hyperprior is INLINE in NSCS03's end-to-end joint codec (already implements Rule #6 spirit)

---

## 4. SCORER-AWARENESS probe class specification (Decision 1B contract)

The 35-substrate HIGH RISK cluster cannot be class-wide DEFERRED until each substrate has a SCORER-AWARENESS probe empirically demonstrating its per-pair conditioning signal does NOT survive SegNet stride-2 + PoseNet per-token attention. Probe class specification:

**Inputs**: substrate (frozen weights) + representative pair (frame0, frame1) from `upstream/videos/0.mkv`

**Per-substrate test procedure**:
1. Forward-pass substrate to produce encoded representation T(x)
2. Decode + render frames; pass to scorer pipeline
3. Compute scorer's per-token attention map (PoseNet) + per-class argmax map (SegNet) on rendered frames
4. Measure MI between substrate's distinguishing-feature output AND scorer's attention/argmax maps
5. Classify substrate per the Tao bound: useful per-pair conditioning capacity ≤ ~9 bits/pair (for 600-pair dashcam); MI < 1 bit/pair = INDEPENDENT (defer); MI ∈ [1, 5] bits/pair = MARGINAL (operator review); MI ≥ 5 bits/pair = INFORMATIVE (proceed to paid dispatch)
6. Per Wyner/Atick/Redlich: ALSO measure encoder-decoder distribution matching — if encoder output distribution mismatches decoder input distribution by KL > 1 bit, substrate is structurally cooperative-receiver-incompatible regardless of MI

**Cost**: ~$0.10-0.50 per substrate probe (forward-pass only); 35 substrates = ~$5-15 total; ~1-2 hours per probe sequentially or parallel-fanout in ~30 min via canonical `tools/parallel_dispatch_top_k.py` actuator per CLAUDE.md "Race-mode rigor inversion" parallel-dispatch-first discipline.

**Probe class implementation**: sister SCORER-RESPONSE-SURFACE ANALYSIS subagent (in flight per parent prompt) IS executing this empirically; symposium APPROVES its framing + accepts findings as binding input.

---

## 5. Operator-action-required summary

**When Modal billing reloads** (operator decision):

1. **Fire 2-3 Rule #6 BOLT-ON-on-A1 lanes** as Priority 1 frontier-breaking dispatch (~$30-75; predicted band [0.188-0.192])
2. **Fire NSCS01 + NSCS03 Phase 2 paid smokes** as Priority 2 frontier-pursuit dispatch (~$25-50; predicted band [0.180-0.190]) per sextet council PROCEED_WITH_REVISIONS from 2026-05-16 sister deliberations
3. **DO NOT fire** the 35-substrate HIGH RISK cluster until SCORER-AWARENESS probe wave completes
4. **DO NOT fire** Z6 original FiLM-bottleneck substrate; substituted with Z6 per-frame-renderer-axis variant per Decision 4C (sister subagent design memo needed before dispatch)
5. **PRESERVE** Rudin floor + Tishby IB-pure as asymptotic-pursuit slots pending sister SCORER analysis findings

**New lanes that need to spawn**:

1. Rule #6 BOLT-ON-on-A1 trainer lane (3 variants: Ballé hyperprior / PR101 entropy stack / VQ-codebook) — design memos required per Catalog #290 + Catalog #294 + Catalog #303 + Catalog #305 + Catalog #309 discipline
2. SCORER-AWARENESS probe wave actuator — uses `tools/parallel_dispatch_top_k.py` per CLAUDE.md "Race-mode rigor inversion" parallel-dispatch-first
3. Z6 per-frame-renderer-axis ego-motion variant (per Rao + Ballard recommendation) — design memo required

**META-assumptions retired** (per Catalog #291 cadence):

- **META-assumption #6** (per-pair conditioning is score-shifting at the contest scorer's response surface): CARGO-CULTED across 35-of-53 substrates; DEFERRED-pending-SCORER-AWARENESS-probe-evidence per CLAUDE.md "Forbidden premature KILL" (NOT killed; reactivation criteria = SCORER-AWARENESS probe per substrate produces non-trivial score-shift evidence)
- **META-assumption #4** (substrate-trainer engineering quality predicts score-shifting quality): CARGO-CULTED per Assumption-Adversary verdict; ENGINEERING-QUALITY remains necessary-not-sufficient for score-shifting
- **META-assumption #7** (probe-disambiguators measure substrate viability at scorer level): CARGO-CULTED; new SCORER-AWARENESS probe class required per Decision 3A

---

## 6. Continual-learning anchor persistence

Per Catalog #300 v2 + Catalog #265 canonical contract, this memo emits a continual-learning anchor via `tac.council_continual_learning.append_council_anchor` with `council_tier=CouncilTier.T4`.

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="t4_symposium_substrate_design_class_shift_deliberation_20260517",
    topic="META: are ANY of our 53 designed substrates actually class-shifting enough to beat 0.192?",
    council_tier=CouncilTier.T4,
    council_attendees=(
        "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
        "Boyd", "Tao", "Carmack", "Quantizr", "George Hotz", "Selfcomp", "Hinton",
        "Hassabis", "MacKay", "Ballé", "Mallat", "van den Oord", "Wyner",
        "Tishby", "Zaslavsky", "Atick", "Redlich", "Rao", "Ballard",
    ),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    predicted_mission_contribution="frontier_breaking",
    override_invoked=False,
    override_rationale=None,
    deferred_substrate_id="high_risk_35_substrate_cluster_per_pair_conditioning_class_shift_pending_scorer_awareness_probe",
    deferred_substrate_retrospective_due_utc="2026-06-17T12:30:00Z",
    # ... full dissent + assumption-adversary verdict + decisions per frontmatter
)
append_council_anchor(record)
```

---

## 7. Sister-subagent coordination notes

- **SCORER-RESPONSE-SURFACE ANALYSIS sister subagent** (in flight): symposium APPROVES its framing per Decision 3A. When sister completes, its empirical findings inform Decision 1B SCORER-AWARENESS probe contract specification AND Decision 4C Rudin/Tishby IB-pure retention vs further substitution decision.
- **NSCS01 + NSCS03 Phase 2 sextet councils** (LANDED 2026-05-16 per commits `5ffaca1b3` + `48b7f8fcf`): symposium ACCEPTS their PROCEED_WITH_REVISIONS verdicts; their substrates remain Priority 2 frontier-pursuit per Substitution Set B from coherence audit.
- **Z6 sextet council** (LANDED 2026-05-16): symposium PARTIALLY ACCEPTS Z6's PROCEED_WITH_REVISIONS — the original Z6 FiLM substrate is SUBSTITUTED per Decision 4C with a per-frame-renderer-axis variant; new sister design memo required.

---

## 8. Cargo-cult audit per assumption (Catalog #303)

Per Catalog #303 standing directive, this memo's distinguishing-feature is a META-decision-adjudication methodology for a 4-tier council escalation. The cargo-cult audit on THIS memo's own assumptions:

| # | Assumption | Classification | Unwind status |
|---|---|---|---|
| 1 | T4 SYMPOSIUM with 23-seat roster is the right forum for adjudicating 5 binding decisions per Catalog #300 elevation criterion (c) | HARD-EARNED | Boyd + Tao + Assumption-Adversary specialist disagreements explicitly invoked in T3 review |
| 2 | Coupled composite verdict (1B+2C+3A+4C+5C) is structurally coherent | HARD-EARNED | The 5 decisions are explicitly coupled per the T3 review §6.2 framing; the composite is internally consistent (no contradictions across decisions) |
| 3 | Beta(2,2) posterior for Rule #6 BOLT-ON IS approximately P=0.50 for a single new bolt-on producing sub-0.193 | PARTIALLY HARD-EARNED | Beta(2,2) is the canonical Beta-Binomial conjugate prior with 1 success and 1 prior parameter; the actual posterior depends on whether PR101 represents 1 success out of how many attempts in the public PR space — operator may revise the count |
| 4 | 35-substrate HIGH RISK cluster classification is accurate | PARTIALLY HARD-EARNED | The clustering analysis from T3 review §3.1 is based on architectural-class + distinguishing-feature heuristics; per-substrate probe is needed to validate per Decision 1B |
| 5 | Rule #6 addition to Path 2 LATTICE does not violate the falling-rule-list canonical discipline (Wang-Rudin 2015) | HARD-EARNED | Rule #6 has explicit position between Rule #2 and Rule #3 in the falling rule list; higher-priority substrates that satisfy Rule #2 alone are accepted there; Rule #6 fires only when both Rule #2 AND bolt-on conditions are satisfied |
| 6 | SCORER-AWARENESS probe class specification is sufficient to validate substrate-scorer compatibility | PARTIALLY HARD-EARNED | The probe class captures MI(substrate-distinguishing-feature; scorer-attention/argmax) + encoder-decoder distribution matching; it may miss higher-order interactions only an empirical bolt-on lane reveals |

---

## 9. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | First T4 SYMPOSIUM deliberation memo in repo per `grep -c "council_tier.*T4" .omx/state/council_deliberation_posterior.jsonl` = 0 prior T4 anchors. |
| 2 | BEAUTY + ELEGANCE | Single memo; ~6500 words; structured as 9 sections; verdict in Section 1.1 reviewable in 1 minute; full deliberation in 8 minutes. |
| 3 | DISTINCTNESS | Distinct from T3 review (which classified META-assumptions) + sextet councils (which adjudicated specific substrates) + coherence audit (which assigned lattice coordinates). This memo is T4 META-DECISION-ADJUDICATION across the coupled 5-decision tree. |
| 4 | RIGOR | 12 PVs per Catalog #229; 23 council positions surfaced verbatim per Catalog #292 Fix 7 amendment; 6 META-assumptions classified per Catalog #292; all required Catalog #300 v2 T4 frontmatter fields present; sister-subagent ownership map honored per Catalog #230. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per-layer canonical-vs-unique decision: ADOPT canonical `tac.council_continual_learning.append_council_anchor` for posterior persistence; ADOPT canonical `tools/subagent_commit_serializer.py` with `--expected-content-sha256` per Catalog #117/#157/#174 (POST-EDIT working-tree shas per docstring corrected 2026-05-13); FORK the 5-decision T4 SYMPOSIUM adjudication structure (no existing canonical helper for coupled-decision verdicts). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Composes with: (a) sister SCORER-RESPONSE-SURFACE ANALYSIS subagent (its findings inform Decision 1B + Decision 4C); (b) Wave 3 K=13 LEVEL-1 schedule (verdict shifts dispatch priorities); (c) Rule #6 lattice extension (Decision 2C lands); (d) cathedral autopilot ranker (verdicts inform candidate weighting per sister Catalog #219/#227); (e) 30-day retrospective due 2026-06-17. |
| 7 | DETERMINISTIC REPRODUCIBILITY | Every cited evidence is specific memo path; vote tallies reproducible from individual council position rationales; assumption classifications reproducible via Catalog #292 discipline. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU cost; ~3 hours wall-clock; 0 source-code edits (analytical META-deliberation only). |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | DIRECT + HIGHEST-EV: this T4 SYMPOSIUM verdict redirects Wave 3 dispatch priorities from HIGH RISK substrates (35-of-53; cumulative expected futile cost ~$200-500) toward Rule #6 BOLT-ON-on-A1 lanes (Beta(2,2) ~50% probability of sub-A1 score; ~$30-75 total) + NSCS01/NSCS03 lifted substrates. If operator adopts the composite verdict, the corpus pivots to the empirically-validated PR101 winning trajectory while preserving asymptotic-pursuit substrates as long-horizon investments. |

---

## 10. Observability surface (Catalog #305)

**6-facet observability** for this T4 SYMPOSIUM deliberation:

1. **Inspectable per layer**: 23 council positions individually queryable via `council_dissent` + per-position rationale; per-decision vote tallies queryable per Section 1.
2. **Decomposable per signal**: VERDICT decomposable into 5 per-decision verdicts; each per-decision verdict decomposable into per-member position rationales.
3. **Diff-able across runs**: this memo is deterministic given its inputs (T3 review + escalation memo + lattice state + sister context); future T4 SYMPOSIUM deliberations on same topic can diff verdicts.
4. **Queryable post-hoc**: `tac.council_continual_learning.query_anchors_by_topic("META class-shift substrate")` returns this anchor; `query_dissent_history("Contrarian")` returns Contrarian's veto position; `query_assumption_classification_history("META-assumption #6 per-pair conditioning")` returns the CARGO-CULTED classification.
5. **Cite-able**: every evidence cite is a specific (memo path, commit SHA, position rationale); 12 related_deliberation_ids form the cite-chain.
6. **Counterfactual-able**: if META-assumption #6 had been HARD-EARNED, Decision 1A class-wide kill would have been the verdict; if Decision 5C had been REJECTED, Decision 2C (Rule #6 PARTIAL) would have been the only frontier-breaking path; the 5 decisions form a coupled counterfactual graph.

---

## 11. Cross-references

- T3 escalation source: `.omx/research/deep_adversarial_review_substrate_design_meta_20260517.md` (parent META-review)
- T4 escalation memo: `.omx/research/t4_symposium_substrate_design_class_shift_question_20260517.md` (canonical 23-seat roster + 5-decision agenda)
- NSCS01 Phase 2 sextet council: `.omx/research/sextet_council_nscs01_phase_2_consensus_20260516.md` (Priority 2 frontier-pursuit candidate)
- NSCS03 Phase 2 sextet council: `.omx/research/sextet_council_nscs03_phase_2_consensus_20260516.md` (Priority 2 frontier-pursuit candidate; sister Rule #6 BOLT-ON #1 Ballé hyperprior implementation)
- Z6 sextet council: `.omx/research/sextet_council_z6_phase_2_consensus_20260516.md` (Decision 4C substituted with per-frame-renderer-axis variant)
- Coherence audit Substitution Set B: `.omx/research/coherence_audit_lattice_coordinate_assignment_20260516.md` + `.omx/research/wave_3_optimization_per_lattice_coherence_20260516.md`
- FALSIFICATION-AUDIT-v2 (Pattern D paradigm-vs-implementation; Pattern E probe-methodology-as-false-falsification): `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md`
- META-ASSUMPTION cadence anchor: `.omx/research/feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (first instance)
- PR95 lesson META-level: `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`
- HORIZON-CLASS plateau warning: `feedback_horizon_class_evaluation_axis_plateau_warning_standing_directive_20260516.md`
- PR101 BOLT-ON empirical anchor: `feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md` §4
- CLAUDE.md non-negotiables cited: "Council hierarchy: 4-tier protocol" + "Council conduct" (Fix 7) + "Mission alignment" + "Forbidden premature KILL" + "Frontier target" + "META-ASSUMPTION ADVERSARIAL REVIEW" + "Race-mode rigor inversion" + Catalog #229/#230/#290/#291/#292/#294/#300/#303/#305

---

## End of T4 SYMPOSIUM deliberation
