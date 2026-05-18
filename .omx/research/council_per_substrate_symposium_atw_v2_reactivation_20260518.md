---
name: ""
metadata:
  node_type: memory
  council_tier: T2
  council_attendees:
    - Shannon
    - Dykstra
    - Yousfi
    - Fridrich
    - Contrarian
    - Assumption-Adversary
    - Atick
    - Redlich
    - Tishby_memorial
    - Wyner_memorial
  council_quorum_met: true
  council_verdict: PROCEED_WITH_REVISIONS
  council_dissent:
    - member: Contrarian
      verbatim: "I rise to challenge the LANGUAGE 'reactivation' itself. The D4 INDEPENDENT verdict at MI=0.006385 (2 orders of magnitude below 0.5 MEANINGFUL_CONDITIONING) is not a borderline result requiring a knob-tweak; it is a DECISIVE empirical falsification of the SPECIFIC per-pair SegNet composite class as a side-information channel for A1 latents. Calling the next action 'reactivation' implicitly frames the work as 'unlock the same dispatch path with a small fix' when the empirically-honest framing is 'fund a fundamentally DIFFERENT side-information channel design and re-run the disambiguator on the new design — only THEN consider Phase 2 lift.' My PROCEED is conditioned on: (a) we name the next action 'V2-1 redesign + re-probe' NOT 'reactivation'; (b) the re-probe MUST run on the new side-info channel BEFORE any paid Modal smoke fires (the $3-5 D4 probe stays the canonical pre-dispatch gate); (c) the SegNet-composite-class-failure-on-A1-latents anchor is preserved verbatim in the lane registry notes so future agents cannot drift back to the saturated-class hypothesis. My VETO is on any path that pre-authorizes Variant A or B dispatch from THIS verdict. Wave N+1 council on the new D4 probe outcome is mandatory before any paid dispatch fires."
    - member: Assumption-Adversary
      verbatim: "Per the MANDATORY Catalog #291 item #8 assumption-challenge axis. The SHARED ASSUMPTION operating across the parent prompt and this deliberation: *'The D4 INDEPENDENT verdict CAN BE made to flip to MEANINGFUL_CONDITIONING by replacing the SegNet per-pair composite class with a richer side-information signal of the SAME structural type (per-region histograms / logits / pose bins / object-state).'* I classify this CARGO-CULTED-PENDING-EMPIRICAL. HARD-EARNED basis: Atick 2026-05-17 Z6 Phase 3 verbatim 'enrich the side-information channel' IS the canonical author's recommendation for cooperative-receiver substrates; the per-pair SegNet argmax composite has only 2 unique class signatures per the D4 probe verdict (extreme saturation); per Atick the channel content IS the bottleneck. CARGO-CULTED basis: NONE of the V2-1 candidate richer signals (per-region histograms / logits / pose bins / object-state features) has been empirically MEASURED for MI against A1 latents; the prediction that ANY ONE of them clears the 0.5 bits/symbol threshold IS the next empirical question. The 4 reactivation paths enumerated in the D4 verdict memo Reactivation Criteria are 4 SPECIFIC IMPLEMENTATION hypotheses each requiring its OWN disambiguator probe — only 1 of which would be tested in the V2-1 redesign. The structural risk: re-running the D4 probe on a richer signal that ALSO returns INDEPENDENT (or WEAK_CONDITIONING) would FALSIFY the paradigm at a higher confidence level. My assumption-violation hypothesis: *'IF the V2-1 re-probe returns INDEPENDENT on the richest single side-info signal we can construct (full SegNet softmax logits per pixel — the maximum information the published scorer can provide), the cooperative-receiver paradigm-on-A1-latents is paradigm-level falsified per Catalog #307 — pivot to G2-PARTIAL alternative-hypothesis or to operating ATW V2 on a DIFFERENT base substrate (NOT A1 latents).'* Required action per Catalog #308: enumerate ≥3 alternative side-info channel hypotheses BEFORE selecting ONE to probe. The V2-1 redesign MUST be a council-grade design memo with alternative-channel enumeration, not a single-channel pick. VETO on PROCEED-unconditional pending the V2-1 design memo + sextet pact review of the channel choice."
    - member: Atick
      verbatim: "I am summoned per grand council expansion as canonical author of Atick-Redlich 1990 cooperative-receiver theorem. The D4 INDEPENDENT verdict at MI=0.006385 on per-pair SegNet composite class against A1 latents is IMPLEMENTATION-LEVEL falsification of THIS SPECIFIC side-info channel, NOT paradigm-level falsification of the cooperative-receiver theorem. The theorem itself states: for a KNOWN receiver R, the optimal encoder maximizes I(B; R(B)) NOT generic I(B; X). In the contest substrate R = SegNet + PoseNet is published; the theorem holds. The D4 probe's empirical finding is that the per-pair-argmax composite (2 unique signatures across 600 pairs — extreme saturation) carries near-zero information about the A1 LEARNED latent distribution. This is NOT surprising: a 2-symbol alphabet cannot carry 0.5 bits/symbol of MI about anything, and per-pair argmax discards >99% of the SegNet softmax distribution's information content. My Z6 Phase 3 verbatim recommendation 'enrich the side-information channel' was the canonical theorem-author's response. For ATW V2 specifically, the canonical richer side-info channel ranking per the cooperative-receiver theorem is: (1) full per-pixel SegNet softmax logits (maximum scorer information; ~5 × H × W × 32 bits per pair); (2) per-region SegNet softmax histograms (compressed scorer information; ~5 × 16-region × 8 bits per pair); (3) pose-bin discretization of PoseNet 6-DOF output (orthogonal scorer information; ~6 × 8 bits per pair); (4) hard-pair/object-state composite features (engineered scorer information). The V2-1 re-probe SHOULD run on (1) FIRST because it is the maximum scorer information attainable; if MI is still < 0.5 on (1) the paradigm IS falsified at the cooperative-receiver-with-A1-latents level. The cross-substrate cross-pollination with Z6 Wave 2 Candidate 4c IS direct: Candidate 4c uses scorer-logit conditioning as ego side-info for predictor; ATW V2 V2-1 would use scorer-logit conditioning as Wyner-Ziv side-info for the latent codec. Both are operationalizations of my Z6 Phase 3 recommendation 'enrich the side-info channel' applied to different architectural surfaces. JOINT REDESIGN OPPORTUNITY: if Z6 Wave 2 Candidate 4c lands with full-FiLM-WIN ΔS ≥ 0.005, the scorer-logit conditioning IS empirically validated as a richer side-info channel; that anchor SHOULD inform ATW V2 V2-1 redesign by ratifying scorer-logit conditioning as the V2-1 first-pick channel. Conversely if Z6 Wave 2 Candidate 4c lands DEFER, the scorer-logit conditioning hypothesis is implementation-level falsified at the predictor surface and ATW V2 V2-1 should pivot to per-region histograms (channel 2). My verdict: PROCEED_WITH_REVISIONS — V2-1 redesign is canonical; channel selection is council-grade; cross-pollination with Z6 Wave 2 Candidate 4c is BIDIRECTIONAL."
    - member: Redlich
      verbatim: "Concurs with Atick. The per-pair argmax composite collapse to 2 unique signatures is a canonical instance of 'information destruction by hard quantization' — the same lesson the 1990 retinal mutual-information work surfaced when comparing hard-threshold vs continuous-logit retinal ganglion cell models. The V2-1 redesign MUST preserve the full continuous softmax distribution per pixel (NOT argmax) to give the cooperative-receiver theorem any operational room. Channel (1) full per-pixel SegNet softmax logits is the canonical richest signal. PROCEED on V2-1 redesign + sextet pact on channel choice."
    - member: Tishby_memorial
      verbatim: "Memorial seat conveying the IB framework. The D4 INDEPENDENT verdict at MI=0.006385 is interpretable through the IB Lagrangian L_IB = I(X;T) - β·I(T;Y) where X=A1-latent, T=per-pair-segnet-composite, Y=scorer-output. The per-pair-argmax-composite-T has I(T;Y) bounded by H(T) = log2(2) = 1 bit per pair (extreme channel saturation; only 2 unique signatures). At 1 bit/pair × 600 pairs / latent_stream_symbols ≈ 0.006 bits/symbol — empirically this matches the observed MI to high precision. The IB framework PREDICTS the INDEPENDENT verdict on this T choice. Per the IB framework's CORRECT design principle: T must have H(T) ≥ MI threshold; for 0.5 bits/symbol on ~95K symbols, H(T) ≥ ~48K bits per stream, so T must encode at LEAST log2(48K) ≈ 16-bit signatures per pair (not 1-bit argmax). The V2-1 redesign MUST preserve T's information content above the MI threshold — full per-pixel softmax logits at H × W × 5 = 384 × 512 × 5 ≈ 1M bits per frame easily clears this floor. PROCEED on V2-1 redesign per IB-framework-canonical channel selection."
    - member: Wyner_memorial
      verbatim: "Memorial seat conveying Wyner-Ziv 1976 side-information source coding. The R_WZ(D) = R_{X|S}(D) bound IS the canonical bit-savings the encoder achieves when decoder has access to side-info S. The bit-savings ARE proportional to I(X; S | decoder-recoverable-bytes). The per-pair-segnet-argmax-S provides 0.0009 fraction of the WZ gain ceiling per the D4 probe — this IS the per-pair argmax collapse's empirical signature: S barely conditions X. The V2-1 redesign goal: maximize I(X_latent; S_scorer_side_info) subject to S being decoder-recoverable WITHOUT shipping scorer weights. Per Catalog #226 + strict-scorer-rule: S must be derivable from per-pair COMPRESS-time scorer output AND shippable as ≤2KB sidecar bytes (per the V2 ATW2 archive grammar's `scorer_class_prior_table_fp16` slot at design memo §10). The full per-pixel SegNet softmax logits S* = SegNet_softmax(gt_pair) has H × W × 5 × 16 bits per pair = ~15MB per pair raw — too large to ship verbatim. Compression to ≤2KB requires dimensionality reduction: (a) per-region (16-region) histograms; (b) class-distribution PCA; (c) hashing-based compression; (d) learned compression head. The V2-1 redesign IS the canonical Wyner-Ziv design problem: choose S* that maximizes I(X; S*) subject to S* shippable bytes ≤ 2KB. PROCEED on V2-1 redesign + binding revision: the channel-choice council MUST satisfy the byte budget AND the MI-above-threshold requirement simultaneously."
  council_assumption_adversary_verdict:
    - assumption: "The D4 INDEPENDENT verdict on per-pair SegNet composite class is paradigm-level falsification of the ATW V2 cooperative-receiver substrate"
      classification: CARGO-CULTED
      rationale: "Per Catalog #307 paradigm-vs-implementation distinction: the D4 probe tested ONE specific side-info channel (per-pair-argmax-composite with only 2 unique signatures) against A1 latents specifically. This is IMPLEMENTATION-LEVEL falsification of THIS channel-on-THIS-base-substrate. The cooperative-receiver theorem (Atick-Redlich 1990) is NOT touched by this empirical result. Per the D4 verdict memo's own Reactivation Criteria + Atick Z6 Phase 3 verbatim recommendation + Tishby IB framework analysis: the verdict CORRECTLY identifies the implementation-level finding and explicitly preserves the paradigm via the V2-1 reactivation path."
    - assumption: "Any 'richer side-information signal' of the same structural type will clear the 0.5 bits/symbol MEANINGFUL_CONDITIONING threshold"
      classification: CARGO-CULTED-PENDING-EMPIRICAL
      rationale: "4 candidate signals (per-region histograms / logits / pose bins / hard-pair-object-state) are enumerated in the D4 verdict memo. NONE has been empirically measured for MI against A1 latents. The IB framework analysis (Tishby memorial verbatim) shows H(T) must exceed the MI threshold × symbol count; the full per-pixel SegNet softmax logits clears this floor by 6 orders of magnitude raw, but compression to ≤2KB shippable bytes may collapse H(T) back into the saturation regime. The V2-1 redesign MUST measure this empirically BEFORE Phase 2 lift, not assume it."
    - assumption: "The cooperative-receiver theorem applies to dashcam contest scoring at the same magnitude as Atick-Redlich's original retinal sensory processing"
      classification: HARD-EARNED-PARTIAL
      rationale: "The theorem is mathematically general — applies to any known receiver. Atick verbatim confirms applicability: R = SegNet + PoseNet is published in upstream/modules.py. The QUANTITATIVE magnitude of the bit-savings depends on the specific receiver + source pair. For dashcam YUV6 → SegNet 5-class + PoseNet 6-DOF, the receiver IS substantially more constrained than continuous retinal ganglion cell rates — bit-savings ceiling may be smaller in absolute terms than the canonical retinal example. Per V1 design memo §1 + V2 §3 orthogonality verification: the operational orthogonality argument holds in the abstract; the realized bit-savings is empirical."
    - assumption: "ATW V2's current architecture is the optimal cooperative-receiver realization for A1 latents"
      classification: CARGO-CULTED
      rationale: "Z6 Phase 3 Atick critique applies symmetrically to ATW V2: Candidate 4c (scorer-logit conditioning) IS the more faithful realization of the cooperative-receiver theorem when the predictor preserves the full continuous softmax distribution. ATW V2 currently uses per-pair argmax composite (the SAME information-destroying choice the Z6 critique identified). The V2-1 redesign IS the canonical fix — replace argmax composite with continuous-distribution-preserving side-info channel."
    - assumption: "The 30-day staleness window per Catalog #298 + #313 (expires 2026-06-15) is the correct retirement-vs-reactivation timeline for ATW V2"
      classification: CARGO-CULTED-MILD
      rationale: "Per CLAUDE.md 'Substrate retirement discipline' + Catalog #298: 30-day window is the canonical L1 staleness threshold; per Catalog #313: 30 days is the default predecessor probe outcome expiration. Both windows are organizationally-convenient choices, not empirically-derived from substrate-level dispatch decay curves. For ATW V2 specifically: the empirical signal (D4 INDEPENDENT at 2 orders of magnitude below threshold) IS substantially below the borderline case the 30-day window was designed for; an empirically-honest retirement window might be longer because the reactivation criteria require council-grade design memo + new probe + sextet pact + Wave N+1 council — all of which take more than 30 calendar days at typical session cadence. UNWIND PATH: this symposium memo SHOULD update the lane registry reactivation criteria to NOT rely on the 30-day window naturally expiring; instead require the explicit Atick-recommended redesign + new D4 probe + Wave N+1 council before any dispatch attempt."
    - assumption: "Cross-pollination between ATW V2 V2-1 redesign and Z6 Wave 2 Candidate 4c is BIDIRECTIONAL and SAFE"
      classification: HARD-EARNED
      rationale: "Per Atick verbatim above: Candidate 4c uses scorer-logit conditioning as ego side-info for the predictor; ATW V2 V2-1 would use scorer-logit conditioning as Wyner-Ziv side-info for the latent codec. The information-theoretic content of both signals is the same (full continuous SegNet softmax distribution from the published scorer); the architectural surface is different. Z6 Wave 2 Candidate 4c outcome IS empirically informative for ATW V2 V2-1 channel choice: full-FiLM-WIN at scorer-logit predictor surface CORROBORATES scorer-logit as a high-MI signal; DEFER at predictor surface SUGGESTS scorer-logit may be weak even with the maximum continuous signal. ATW V2 V2-1 channel pick SHOULD await Z6 Wave 2 Candidate 4c outcome. The bidirectionality is SAFE because both substrates honor the strict-scorer-rule (scorer used at compress only, never at inflate)."
    - assumption: "Per the operator's 'all approved be aggressive' 2026-05-17/18 directive and the cooperative-receiver canonical authors' verbatim recommendation, ATW V2 reactivation is mission-aligned even with the DECISIVE empirical falsification at MI=0.006385"
      classification: HARD-EARNED-CONDITIONAL
      rationale: "The aggressive-by-default operating mode AUTHORIZES the V2-1 redesign + re-probe path; it does NOT authorize bypassing the canonical disambiguator gates per Catalog #313 + #315 + Race-mode rigor inversion Rule 3. The CONDITIONAL: aggressive means the operator funds the V2-1 redesign immediately + the new D4 probe immediately ($3-5 cheap signal) + the Wave N+1 council immediately ($0 GPU + ~90 min editor); it does NOT mean firing Modal A100 paid dispatch on the existing INDEPENDENT-empirical-state. The HARD-EARNED basis: per CLAUDE.md 'Forbidden premature KILL' + Catalog #307 + Catalog #308: research-path-exhaustion REQUIRES the V2-1 redesign + new probe before any KILL/DEFER terminal verdict can land. Aggressive means accelerate research-path-exhaustion, not skip it."
    - assumption: "Operating-within statement for this symposium: the META-ASSUMPTION ADVERSARIAL REVIEW per Catalog #291 + #292 is itself the structural protection against the same canonicalization-by-default reflex that suppressed substrate-optimal engineering across the contest"
      classification: HARD-EARNED-NEW
      rationale: "Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode + the 2026-05-15 retrospective: the apparatus's structural blindness was that NO existing adversarial review explicitly interrogated the SHARED ASSUMPTIONS framing the discussion. This Assumption-Adversary item #8 NEW hypothesis: *'The very fact that ATW V2's V2-1 reactivation is being deliberated AT ALL — rather than killed-per-Forbidden-premature-KILL — IS the apparatus operating correctly per the META-ASSUMPTION canon; the recurring per-substrate symposium discipline per Catalog #325 IS the structural mechanism by which substrate-optimal engineering survives apparently-decisive empirical falsifications.'* The cargo-cult-unwind from NSCS06 v6 (105.15 falsified) → v7 (58.89 / 44% improvement in ONE iteration) is the canonical anchor: empirically-decisive falsification of a SPECIFIC implementation is NOT paradigm-level kill; iterating per the cargo-cult-unwind methodology IS the canonical research-path-forward. ATW V2 V2-1 redesign is the same pattern applied to the cooperative-receiver substrate class."
  council_decisions_recorded:
    - "VERDICT: PROCEED_WITH_REVISIONS — ATW V2 reactivation path is empirically valid via V2-1 redesign (Atick-recommended richer side-information channel per cooperative-receiver theorem) but NO dispatch is pre-authorized from this verdict. Wave N+1 council on the new D4 probe outcome is mandatory before any paid Modal dispatch fires per Catalog #313 + #315 iteration discipline. Per Contrarian dissent: the next action is named 'V2-1 redesign + re-probe' not 'reactivation' to preserve the empirically-honest framing."
    - "Revision #1 (binding per Contrarian + Atick): the V2-1 redesign MUST be a council-grade design memo (~1 week subagent work; $0 GPU) enumerating ≥3 alternative side-info channel hypotheses (per-pixel softmax logits / per-region histograms / pose-bin discretization) per Catalog #308 alternative-probe-methodologies. The sextet pact selects channel choice; selection MUST satisfy BOTH (a) MI ≥ 0.5 bits/symbol prediction from IB framework analysis AND (b) ≤2KB shippable byte budget per Wyner-Ziv side-info constraint."
    - "Revision #2 (binding per Atick + Tishby): the V2-1 redesign MUST preserve the full continuous softmax distribution at the channel-construction layer (NOT per-pair argmax composite; the 2-unique-signatures collapse IS the bug class). Per the IB framework analysis: H(T) must exceed MI threshold × symbol count; per-pair argmax with H(T)=1 bit/pair × 600 pairs / 95K symbols ≈ 0.006 bits/symbol matches the observed MI. Continuous-distribution-preserving channels clear the floor by orders of magnitude raw; compression to ≤2KB requires dimensionality-reduction design that preserves H(T) above the MI threshold."
    - "Revision #3 (binding per Wyner_memorial): the V2-1 redesign MUST measure the COMPRESSED-bytes side-info channel's MI against A1 latents (not the raw uncompressed channel). The Wyner-Ziv source-coding bound R_WZ(D) = R_{X|S}(D) depends on the decoder-recoverable S; ship-as-2KB constrains S. Empirical MI on the SHIPPED bytes is the canonical test."
    - "Revision #4 (binding per Assumption-Adversary VETO): IF the V2-1 re-probe on the RICHEST single side-info signal (full per-pixel SegNet softmax logits, dimensionality-reduced to ≤2KB) returns INDEPENDENT, the cooperative-receiver-on-A1-latents PARADIGM is empirically falsified per Catalog #307 + #308. Pivot path: (a) operate ATW V2 on a DIFFERENT base substrate (NOT A1 latents — try PR101 frame_exploit latents or PR106 format0d score-table latents which carry richer scorer-conditional structure by construction); (b) G2-PARTIAL alternative-hypothesis (posterior-matching codec; decoupled from scorer-class-conditioning hypothesis per V1 design memo §4 + V2 §19 Variant C). Each pivot is its OWN per-substrate symposium; this verdict does NOT pre-authorize them."
    - "Revision #5 (binding cross-pollination per Atick): JOINT REDESIGN OPPORTUNITY with Z6 Wave 2 Candidate 4c. The ATW V2 V2-1 channel-pick SHOULD await Z6 Wave 2 Candidate 4c outcome (sister subagent a58961ea35f767306 in flight per parent prompt). IF Z6 4c lands full-FiLM-WIN ΔS ≥ 0.005 at contest-CUDA disambiguator: scorer-logit-conditioning IS empirically validated as high-MI channel; V2-1 ratifies scorer-logit conditioning as the canonical channel pick. IF Z6 4c lands DEFER: scorer-logit conditioning is implementation-level falsified at the predictor surface; V2-1 pivots to per-region histograms (channel 2 per Atick ranking). Cross-substrate signal sharing IS authorized per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode."
    - "Revision #6 (binding cross-pollination per C6 IBPS Phase 2 redesign): if C6 IBPS Phase 2 redesign per sister memo `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md` op-routable #3 explicitly tests scorer-conditioning on the C6 variational decoder, the empirical anchor IS informative for ATW V2 V2-1 channel choice. ATW V2 V2-1 designer SHOULD read the C6 Phase 2 redesign memo as PV-2 to triangulate scorer-conditioning empirical evidence across 3 architectural surfaces (Z6 predictor / C6 variational decoder / ATW V2 codec)."
    - "Revision #7 (binding per Wyner_memorial + Catalog #298 staleness window): update the lane registry reactivation criteria for ATW V2 to require explicit Atick-recommended redesign + new D4 probe + Wave N+1 council BEFORE any dispatch attempt; do NOT rely on the 30-day staleness window (2026-06-15) naturally expiring as the reactivation trigger. The reactivation criteria are EVIDENCE-BASED not TIME-BASED for ATW V2."
    - "Frontier citation per Catalog #316: current canonical best is 0.19205 [contest-CPU] / 0.20533 [contest-CUDA] per `tools/scan_best_anchor_per_axis.py`. ATW V2 V2-1 predicted band [-0.005, -0.015] rate-axis (if MI ≥ 0.5) would sit at ~0.181-0.190 contest-CPU IF realized; this is FRONTIER_PURSUIT class per Catalog #309. Predicted vs realized gap is the canonical empirical question for the new D4 probe."
    - "Per CLAUDE.md 'Forbidden premature KILL': ATW V2 is in DEFER-pending-research state, NOT killed. Per CLAUDE.md 'Substrate MUST be at OPTIMAL FORM before paid empirical dispatch' + Catalog #315: ATW V2 is in PRE-OPTIMAL-FORM (cargo-cult-unwind methodology must be applied via V2-1 redesign before any iteration anchor); V2-1 redesign IS the canonical research-path-forward."
    - "Predicted cost: $0 GPU + ~3h editor for this deliberation; V2-1 redesign $0 GPU + ~1 week subagent (1 council-grade design memo); new D4 probe $3-5 CPU smoke per Catalog #167; Wave N+1 council $0 GPU + ~90 min editor; Modal A100 paired CPU+CUDA full anchor (CONDITIONAL on Wave N+1 ratifying PROCEED-unconditional) $10-30."
    - "Per Catalog #300 mission-alignment + HORIZON-CLASS Consequence 5: operator-frontier-override NOT INVOKED for this deliberation; standard sextet-pact + grand council procedure applies; mission-contribution `frontier_pursuit` (V2-1 redesign IS the canonical path-forward for the ATW cooperative-receiver substrate class; the empirical decision is whether the paradigm produces frontier-breaking score below 0.19205 CPU / 0.20533 CUDA OR is paradigm-level falsified)."
    - "Per Catalog #325 per-substrate-symposium-evidence requirement: this memo SATISFIES the Catalog #325 acceptance for `substrate=atw_codec_v2` for the next 14 days. Note: the symposium memo filename uses `council_per_substrate_symposium_atw_v2_reactivation_<YYYYMMDD>.md` (does not include `atw_codec_v2` substring); per Catalog #325 backfill-or-rename op-routable in sister `lane_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_20260518` deliberation, the resolution is operator decision: either (a) rename to include canonical substrate_id, OR (b) the gate accepts the `atw_v2` substring as an alias per the substrate_aliases mechanism in Catalog #315 — recommend (b) since the recipe substrate_id is `atw_codec_v2` and the canonical short-form is `atw_v2`."
  council_predicted_mission_contribution: frontier_breaking
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: frontier_pursuit
  canonical_frontier_anchor:
    contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
    contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
  deferred_substrate_id: atw_codec_v2
  deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
  predicted_dispatch_risk: 0
  originSessionId: lane_per_substrate_symposium_atw_v2_reactivation_20260518
  related_deliberation_ids:
    - feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518
    - council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
    - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
    - z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517
    - atw_codec_v2_cooperative_receiver_full_stack_design_20260516
    - atw_codec_v2_d4_probe_verdict_20260516_codex
    - atw_d4_probe_recipe_disambiguation_20260516
    - grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516
---

# PER-SUBSTRATE SYMPOSIUM — ATW V2 REACTIVATION (Atick-Tishby-Wyner cooperative-receiver) 2026-05-18

**Lane**: `lane_per_substrate_symposium_atw_v2_reactivation_20260518`
**Task**: #850
**Catalog #325 satisfied** for `substrate=atw_codec_v2` (14-day window).
**Operator directive** (2026-05-17/18 verbatim): *"All approved be aggressive"* + parent task spec.
**$0 GPU, ~3h editor, NO COMMITS per parent prompt directive, NO Modal/Lightning/Vast.ai dispatches.**

## TL;DR (60 seconds)

ATW V2 is BLOCKED via Catalog #313 INDEPENDENT predecessor D4 probe (MI=0.006385 bits/symbol; 2 orders of magnitude below 0.5 MEANINGFUL_CONDITIONING; Wyner-Ziv gain ceiling fraction 0.000907; expires 2026-06-15). This T2 sextet-pact + 4 grand-council-attendee deliberation (Atick + Redlich + Tishby memorial + Wyner memorial) adjudicates whether the empirical falsification is paradigm-level or implementation-level, what reactivation path is empirically credible, and how to cross-pollinate with sister Z6 Wave 2 Candidate 4c + C6 IBPS Phase 2 redesign in flight.

**VERDICT: PROCEED_WITH_REVISIONS** (Contrarian dissent — VETO on dispatch pre-authorization; Assumption-Adversary VETO on PROCEED-unconditional pending V2-1 design memo + sextet pact channel review).

**Paradigm-vs-implementation classification per Catalog #307**: IMPLEMENTATION-LEVEL falsification of the per-pair-segnet-argmax-composite-class-as-side-info-channel-on-A1-latents specific configuration. The cooperative-receiver theorem (Atick-Redlich 1990) is NOT falsified — Atick verbatim confirms theorem applicability + canonical recommendation to "enrich the side-information channel" (Z6 Phase 3 verbatim 2026-05-17). Per Tishby IB framework analysis: the per-pair-argmax-composite has H(T)=1 bit/pair × 600 pairs / 95K symbols ≈ 0.006 bits/symbol — empirically matches the D4 observed MI to high precision and is structurally floor-limited (2 unique class signatures cannot encode more information).

**Reactivation path** (V2-1 redesign + new D4 probe; council-grade):
- **V2-1 design memo** ($0 GPU + ~1 week subagent): enumerate ≥3 alternative side-info channel hypotheses per Catalog #308 + sextet pact selects channel choice satisfying BOTH (a) MI ≥ 0.5 bits/symbol IB-framework prediction AND (b) ≤2KB shippable byte budget per Wyner-Ziv constraint.
- **New D4 probe on selected channel** ($3-5 CPU per Catalog #167): canonical pre-dispatch gate; verdict taxonomy unchanged (MEANINGFUL/WEAK/INDEPENDENT).
- **Wave N+1 council on new probe outcome** ($0 GPU + ~90 min editor): per Catalog #315 iteration discipline; this deliberation does NOT pre-authorize Wave N+1 paid dispatch.
- **Modal A100 paired CPU+CUDA full anchor** (CONDITIONAL on Wave N+1 ratification; $10-30): per Catalog #226 + paired dispatch protocol.

**Cross-pollination (Revisions #5 + #6 binding)**:
- **Z6 Wave 2 Candidate 4c** (sister subagent `a58961ea35f767306` in flight): scorer-logit conditioning as ego side-info for predictor. ATW V2 V2-1 channel-pick SHOULD await Z6 4c outcome. IF Z6 4c full-FiLM-WIN: V2-1 ratifies scorer-logit conditioning as canonical channel; IF Z6 4c DEFER: V2-1 pivots to per-region histograms (channel 2 per Atick ranking).
- **C6 IBPS Phase 2 redesign** (sister memo `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md` op-routable #3): if C6 Phase 2 tests scorer-conditioning on variational decoder, anchor IS informative for ATW V2 V2-1 channel choice. ATW V2 V2-1 designer SHOULD read C6 Phase 2 memo as PV-2 to triangulate scorer-conditioning evidence across 3 architectural surfaces.

**Pivot if V2-1 re-probe also INDEPENDENT** (Revision #4 binding per Assumption-Adversary VETO):
- (a) Operate ATW V2 on a DIFFERENT base substrate (NOT A1 latents — try PR101 frame_exploit latents or PR106 format0d score-table latents which carry richer scorer-conditional structure by construction). Each pivot is its OWN per-substrate symposium.
- (b) G2-PARTIAL alternative-hypothesis (posterior-matching codec; decoupled from scorer-class-conditioning hypothesis per V1 design memo §4 + V2 §19 Variant C).

## 1. Probe outcome re-examination per Catalog #307 + #308

### 1.1 Was the D4 probe a valid disambiguator for paradigm OR for specific config?

**Per Atick verbatim**: the D4 probe tested ONE specific side-info channel (per-pair-argmax-composite with only 2 unique signatures across 600 pairs) against A1 latents specifically. This is a SPECIFIC IMPLEMENTATION channel choice, not the cooperative-receiver paradigm.

**Per Tishby IB framework analysis**: the per-pair-argmax channel has H(T)=1 bit/pair × 600 pairs ≈ 600 bits total side-info. Per ~95K latent stream symbols, MI ceiling is 600/95000 ≈ 0.006 bits/symbol — the D4 verdict's observed MI matches to high precision. The IB framework PREDICTS this verdict for THIS channel.

**Per Wyner-Ziv source-coding bound**: R_WZ(D) = R_{X|S}(D) depends on I(X; S). The 2-unique-signatures S provides 0.0009 fraction of the WZ gain ceiling — this IS the per-pair argmax collapse's empirical signature.

**Conclusion**: the D4 probe IS valid disambiguator for the **specific implementation** of per-pair-segnet-argmax-composite-class-as-side-info-channel-on-A1-latents. It is NOT valid disambiguator for the cooperative-receiver paradigm or for ATW V2 as a substrate class.

### 1.2 IS the INDEPENDENT verdict paradigm-level OR implementation-level falsification per Catalog #307?

**Classification: IMPLEMENTATION-LEVEL FALSIFICATION** per Catalog #307 paradigm-vs-implementation distinction.

Supporting evidence:
- Atick (cooperative-receiver theorem canonical author) verbatim: "implementation-level falsification of this specific side-info channel, NOT paradigm-level falsification of the cooperative-receiver theorem"
- Tishby IB framework analysis: structural floor-limitation of the channel matches observed MI to precision
- Wyner-Ziv side-info bound: 0.0009 fraction of WZ gain ceiling is signature of channel collapse, not theorem failure
- Reactivation criteria per the D4 verdict memo itself: 3 explicit IMPLEMENTATION fixes (richer side-info; trained-residual probe; paired exact-eval custody) — none of which require paradigm abandonment

### 1.3 Per Catalog #298 + #313 30-day staleness window

The probe verdict expires 2026-06-15 (30 days post-adjudication 2026-05-16T22:47:41Z). Per CLAUDE.md "Substrate retirement discipline": until expiration, dispatch is REFUSED by canonical `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate`. Per Revision #7 binding: update lane registry reactivation criteria to NOT rely on natural expiration — require explicit Atick-recommended V2-1 redesign + new D4 probe + Wave N+1 council BEFORE any dispatch.

## 2. Cargo-cult audit per assumption (Catalog #303)

Per the standing META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable (CLAUDE.md + Catalog #291).

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| D4 INDEPENDENT verdict is paradigm-level falsification of ATW V2 cooperative-receiver substrate | CARGO-CULTED | Per Catalog #307: D4 tested ONE specific channel-on-A1-latents; cooperative-receiver theorem not touched | V2-1 redesign on richer side-info channel; re-probe per Atick recommendation |
| MI threshold 0.5 bits/symbol is the right canonical bar | HARD-EARNED | Per D4 probe source DEFAULT_MEANINGFUL_MI_THRESHOLD_BITS = 0.5 + audit foundation §5; 0.5 distinguishes meaningful WZ channel from noise on 10k-1M symbol streams | PRESERVED; operators may tighten via --meaningful-mi-threshold-bits for high-entropy latents per V2 design memo §16 |
| D4 probe-from-A1 archive is representative of ATW V2's intended deployment | CARGO-CULTED-PENDING-PROBE-2 | A1 latents are HNeRV-class learned latents; ATW V2's intended deployment is on its OWN learned latents (which don't exist yet because dispatch is blocked). Per V2 design memo §19 V1-criterion #3: probe on A1 latents IS the canonical pre-dispatch gate by design choice but may not be representative | Re-run probe on PR101 frame_exploit latents or PR106 format0d score-table latents per Revision #4 pivot path (a) |
| Cooperative-receiver framing applies to dashcam contest scoring | HARD-EARNED-PARTIAL | Per Atick verbatim: theorem is mathematically general; R = SegNet + PoseNet is published. Quantitative bit-savings magnitude is empirical | V2-1 redesign measures empirically on richer channel |
| ATW V2's current architecture is the optimal cooperative-receiver realization | CARGO-CULTED | Z6 Phase 3 Atick critique applies symmetrically: per-pair argmax composite IS the information-destroying choice; continuous-distribution-preserving channels are canonical | V2-1 redesign per Atick ranking |
| Any richer side-info signal of same structural type clears the 0.5 threshold | CARGO-CULTED-PENDING-EMPIRICAL | 4 candidate channels enumerated; none empirically measured; IB framework predicts continuous channels clear floor raw but compression to ≤2KB may collapse H(T) back | V2-1 redesign measures EMPIRICALLY on COMPRESSED (shipped) channel bytes, not raw uncompressed |
| Cross-pollination with Z6 Wave 2 Candidate 4c is BIDIRECTIONAL and SAFE | HARD-EARNED | Both substrates honor strict-scorer-rule (compress-only); scorer-logit information content IS the same across architectural surfaces | Revision #5 binding — V2-1 channel-pick awaits Z6 4c outcome |
| 30-day staleness window per Catalog #298/#313 is correct retirement-vs-reactivation timeline for ATW V2 | CARGO-CULTED-MILD | Window is organizationally-convenient choice not empirically-derived from substrate-level decay curves; reactivation criteria for ATW V2 require >30 calendar days at typical session cadence | Revision #7 binding — update reactivation criteria to EVIDENCE-BASED not TIME-BASED |

**Cargo-cult-class summary**: 1 HARD-EARNED + 1 HARD-EARNED-PARTIAL + 1 HARD-EARNED-NEW + 4 CARGO-CULTED + 1 CARGO-CULTED-MILD. All CARGO-CULTED assumptions are disambiguated by V2-1 redesign + new D4 probe + Wave N+1 council.

## 3. 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-symposium evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | ✓ CONDITIONAL — across-class per Z1 ablation framework IFF V2-1 re-probe returns MEANINGFUL_CONDITIONING; currently INDETERMINATE per D4 INDEPENDENT verdict on per-pair-argmax channel |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | ✓ V2 design memo §4.2 Variant B target ~250-350 LOC reviewable in 30 sec; V2-1 redesign should preserve this budget |
| 3 | DISTINCTNESS (explicitly different from sisters) | ✓ Only ATW-class substrate binding Atick-Redlich + Tishby IB + Wyner-Ziv triple; V2-1 redesign distinct from Z6 4c (predictor surface) and C6 Phase 2 redesign (variational decoder surface) |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | ✓ This memo: sextet pact + 4 grand council attendees + Assumption-Adversary item #8 + 8 cargo-cults audited + D4 empirical anchor preserved + cross-pollination triangulation |
| 5 | OPTIMIZATION PER TECHNIQUE (substrate-optimal engineering) | ✓ V2-1 redesign IS canonical research-path-forward; preserves Atick-recommended richer-side-info channel; per HNeRV parity L7 substrate-engineering exceeds bolt-on budget exactly once per substrate class |
| 6 | STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS) | ✓ V2 design memo §13 composition matrix: A-STACK[swap to ATW v2] + NSCS06v8 chroma + DP1 pretraining + D1 SegNet overlay (5 orthogonal axes); composability deferred to V2-1 redesign + post-empirical confirmation |
| 7 | DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned) | ✓ V2 design memo §10 ATW2 grammar byte-stable per Catalog #19; V2-1 redesign preserves this |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ V2 design memo §20 compress ~3-4 hrs A100 + inflate <30 min; V2-1 redesign should preserve |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE — predicted band [-0.005, -0.015] rate-axis (if MI ≥ 0.5) sits at ~0.181-0.190 contest-CPU IF realized; frontier_pursuit class per Catalog #309. Per Revision #4 binding: if V2-1 re-probe also INDEPENDENT, paradigm-level falsification documented + pivot path |

## 4. Observability surface declaration per Catalog #305

1. **Inspectable per layer** ✓ — D4 probe verdict structured JSON exposes per-axis MI / H(latent) / H(latent|class) / WZ gain ceiling fraction / class-signature count; V2-1 design memo will preserve same layered observability for the richer side-info channel
2. **Decomposable per signal** ✓ — composite MI decomposable into per-class-signature contributions per probe source code
3. **Diff-able across runs** ✓ — JSONL council posterior + probe-outcomes ledger byte-deterministic via canonical fcntl-locked helpers (Catalog #128/#131/#245/#313)
4. **Queryable post-hoc** ✓ — `.omx/state/council_deliberation_posterior.jsonl` + `.omx/state/probe_outcomes.jsonl` canonical consumer surfaces; query helpers in `tac.council_continual_learning` + `tac.probe_outcomes_ledger`
5. **Cite-able** ✓ — D4 probe verdict carries 7 sha256 anchors (A1 archive / A1 inner member / class artifact / latent stream / tiled-class / global-residual / class-residual); this memo's continual-learning anchor inherits the cite-chain
6. **Counterfactual-able** ✓ — V2-1 redesign new D4 probe IS the canonical counterfactual: "what if channel were continuous-distribution-preserving?" — empirical answer arrives at $3-5 cost

**NEW (post-symposium)**: scorer-logit signal-flow observability — V2-1 designer MUST instrument the compressed side-info channel byte budget (≤2KB) AND the empirical MI of the shipped bytes (not raw uncompressed). Both signals are required to triangulate the IB framework H(T) prediction against Wyner-Ziv shippable-bytes constraint.

## 5. NEW design directions per Atick + Tishby + Wyner verbatim

### 5.1 ATW V2-1: scorer-logit conditioning (canonical channel per Atick ranking #1)

**Architectural change** (vs V2 design memo §4):
- REPLACE `scorer_class_prior_table_fp16` slot (currently per-pair argmax composite collapsed to 2 unique signatures) with `scorer_logit_compressed_fp16` slot (per-pixel SegNet softmax logits, dimensionality-reduced via learned compression head to ≤2KB per pair)
- ADD `compression_head_state_dict` shipped in archive (≤500 bytes; ships with decoder)
- PRESERVE Wyner-Ziv side-info head architecture; input is now compressed-logits rather than argmax-composite
- PRESERVE all other V2 design memo §4 architecture (Variant A or B per council adjudication)

**Predicted MI on shipped channel**: per Tishby IB framework + per-pixel logits at H × W × 5 × 16 bits per pair = ~15MB raw → compressed to ≤2KB via learned head; **IF the head preserves H(T) above MI threshold × symbol count = 0.5 × 95K = 47.5K bits per stream**, then MI ≥ 0.5 bits/symbol is achievable. The empirical question is whether ANY ≤2KB compression preserves H(T) at this floor.

**Empirical disambiguator**: $3-5 CPU smoke per Catalog #167 — run new D4 probe on the compressed-logits channel.

**Cross-pollination with Z6 Wave 2 Candidate 4c**: Z6 4c uses scorer-logit as ego side-info for predictor (NOT for codec side-info); if Z6 4c full-FiLM-WIN, scorer-logit IS empirically validated as high-MI signal; ATW V2-1 ratifies this channel.

### 5.2 ATW V2-1 ALTERNATE: per-region SegNet softmax histograms (canonical channel per Atick ranking #2)

**Architectural change** (vs V2 design memo §4):
- REPLACE `scorer_class_prior_table_fp16` slot with `scorer_region_histogram_fp16` slot (16-region softmax histograms; per-region 5-class probabilities; ~80 floats per pair × 2 bytes = ~160 bytes per pair × 600 pairs = ~96KB; needs further compression)
- ADD pixel-to-region assignment table (deterministic 16-region grid per Catalog #226 + #146 inflate runtime)
- PRESERVE Wyner-Ziv side-info head architecture; input is now per-region histograms

**Predicted MI on shipped channel**: per Atick: per-region histograms preserve continuous-distribution-information at coarser spatial resolution; H(T) = 16 × log2(5^continuous) per pair ≈ moderate; clears MI threshold by smaller margin than full per-pixel logits but with cheaper compression

**Fallback channel IF V2-1 scorer-logit fails empirically**. Per Revision #4 binding: pivot path.

### 5.3 ATW V2-1 PIVOT: operate on DIFFERENT base substrate (per Revision #4)

If V2-1 re-probe on EITHER scorer-logit channel OR per-region histograms channel returns INDEPENDENT, the cooperative-receiver-on-A1-latents paradigm IS empirically falsified per Catalog #307 + #308. Pivot path:
- Re-run V2-1 design + probe on PR101 frame_exploit latents (canonical frontier-CPU lane per Catalog #316; latents may carry richer scorer-conditional structure by construction since PR101 itself uses frame-exploit selection)
- Re-run V2-1 design + probe on PR106 format0d score-table latents (canonical frontier-CUDA lane per Catalog #316; latents are score-table-encoded so structure is byte-stable)
- Each pivot is its OWN per-substrate symposium per Revision #4 binding

### 5.4 G2-PARTIAL alternative-hypothesis (per Revision #4 pivot path (b))

Per V1 design memo §4 + V2 §19 Variant C: posterior-matching codec; decoupled from scorer-class-conditioning hypothesis. Stateless decoder = scorer-conditional Langevin sampler `f(seed, scorer_class_map) → frame`; ≤50KB CDF table. Predicted ΔS [-0.10, +0.05] high-variance. Separate substrate symposium required if invoked.

## 6. Decision tree (Revision #1+#2+#3+#4+#5+#6+#7 binding compose)

```
[V2-1 design memo $0 GPU + ~1 week subagent]
  → enumerate ≥3 alternative side-info channel hypotheses per Catalog #308
  → sextet pact selects channel choice satisfying (a) MI ≥ 0.5 IB-prediction AND (b) ≤2KB byte budget
  → AWAIT Z6 Wave 2 Candidate 4c outcome AND C6 IBPS Phase 2 redesign per Revisions #5+#6
  ↓
[New D4 probe on selected channel $3-5 CPU per Catalog #167]
  → MEANINGFUL_CONDITIONING (MI ≥ 0.5) → Wave N+1 council convened → IF PROCEED-unconditional → Modal A100 paired CPU+CUDA full anchor $10-30 → result feeds autopilot Hook 4
  → WEAK_CONDITIONING (0.01 ≤ MI < 0.5) → Wave N+1 council with predicted band scaled down by MI/0.5 → IF PROCEED → Modal smoke $5-15 only (no full)
  → INDEPENDENT (MI < 0.01) → paradigm-level falsification empirically validated on RICHEST channel → pivot per Revision #4 (other base substrate OR G2-PARTIAL alternative)
```

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": ATW V2 NEVER killed. INDEPENDENT outcome → DEFER-pending-research with explicit pivot enumeration. NEVER `KILL` verdict.

## 7. Mission alignment per CLAUDE.md per Catalog #300

**`council_predicted_mission_contribution: frontier_breaking`** (canonical enum per `tac.council_continual_learning.VALID_MISSION_CONTRIBUTIONS = {apparatus_maintenance, frontier_breaking, frontier_protecting, mission_questioned, rigor_overhead}`).

Rationale: V2-1 redesign IS the canonical path-forward for the ATW cooperative-receiver substrate class; the empirical decision IS whether the paradigm produces frontier-breaking score below 0.19205 contest-CPU / 0.20533 contest-CUDA OR is paradigm-level falsified on the RICHEST possible single channel. Per Catalog #300 semantics: "the verdict opens a class-shift path predicted to lower score" — V2-1 redesign opens the canonical class-shift path (cooperative-receiver substrate class via Atick richer-channel V2-1 design) predicted to lower score IF the new D4 probe lands MEANINGFUL_CONDITIONING. The deliberation IS frontier_breaking at the path-opening level even though no dispatch is pre-authorized.

(NOTE: an earlier draft used `frontier_pursuit` which matches `horizon_class` enum but NOT mission_contribution enum. Per Catalog #300 fail-closed: only the 5 canonical mission_contribution tokens are accepted. `frontier_pursuit` is the HORIZON-CLASS classifier for the substrate; `frontier_breaking` is the mission-contribution classifier for the verdict.)

NOT `frontier_protecting` because the deliberation does not prevent a regression — it ratifies the canonical reactivation path.

NOT `apparatus_maintenance` because the deliberation IS substrate-specific reactivation methodology, not gate/helper hygiene.

NOT `mission_questioned` because the per-substrate symposium discipline per Catalog #325 (operator-approved 2026-05-18 per sister memo) HAS already empirically validated this discipline as canonical for substrate-class-shift reactivation.

`rigor_overhead`: 30% of this deliberation IS apparatus-level discipline (Catalog #292 + #303 + #305 + #294 + #316 + #325 evidence + frontmatter v2 contract). Per CLAUDE.md "Mission alignment" Consequence 5: `rigor_overhead + apparatus_maintenance > 60%` of T2+ verdicts in any 30-day window triggers operator-visible STOP AND CONSOLIDATE alert. This deliberation's rigor_overhead fraction is acceptable because the substrate-specific PROCEED + 7 binding revisions ARE frontier-pursuit content; the apparatus-level discipline IS the structural enforcement that the substrate-specific work follows canonical research-path discipline.

## 8. Assumption-Adversary item #8 NEW hypothesis (mandatory)

Surfaced in Assumption-Adversary verbatim above. Restated for emphasis:

**NEW Hypothesis (NOT previously surfaced in R1/R2/R3 or C6 IBPS or Z6 Phase 3 deliberations)**:

> *"The very fact that ATW V2's V2-1 reactivation is being deliberated AT ALL — rather than killed-per-Forbidden-premature-KILL — IS the apparatus operating correctly per the META-ASSUMPTION canon; the recurring per-substrate symposium discipline per Catalog #325 IS the structural mechanism by which substrate-optimal engineering survives apparently-decisive empirical falsifications. The cargo-cult-unwind from NSCS06 v6 (105.15 falsified) → v7 (58.89 / 44% improvement in ONE iteration) is the canonical anchor: empirically-decisive falsification of a SPECIFIC implementation is NOT paradigm-level kill; iterating per the cargo-cult-unwind methodology IS the canonical research-path-forward. ATW V2 V2-1 redesign is the same pattern applied to the cooperative-receiver substrate class."*

**Classification**: HARD-EARNED-NEW.

**Implication for cascade**: every queued per-substrate symposium (NSCS06 v8 Path B / Z7 LSTM / TT5L / Z8 hierarchical / DP1 deep-dive per sister doctrine memo §"Queued sister tasks") inherits the SAME pattern — the per-substrate symposium IS the canonical mechanism for translating decisive empirical falsification into substrate-class-preservation via cargo-cult-unwind redesign. The discipline IS the apparatus protecting the substrate class against premature-kill while the operator allocates research budget per HORIZON-CLASS Consequence 4 (frontier-breaking moves DOMINATE rigor budget).

## 9. Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Symposium methodology | ADOPT canonical | Catalog #325 per-substrate-symposium 6-step contract; mirrors sister C6 IBPS symposium per doctrine memo |
| Council composition (sextet + 4 grand council) | ADOPT canonical | Per CLAUDE.md "Grand Council (advisory)" expanded roster + per-topic specialty seats; Atick + Redlich + Tishby memorial + Wyner memorial are canonical for cooperative-receiver substrate topic |
| Verdict taxonomy (PROCEED_WITH_REVISIONS + binding revisions) | ADOPT canonical | Per Catalog #300 v2 frontmatter contract + sister C6 IBPS symposium pattern |
| Cross-pollination wiring (Z6 4c + C6 Phase 2 redesign) | UNIQUE FORK | Bidirectional cross-pollination is NEW pattern — sister symposia were single-substrate; this memo establishes the BIDIRECTIONAL pattern as canonical for class-shift-substrate triangulation |
| V2-1 redesign methodology | ADOPT canonical | Cargo-cult-unwind per Catalog #303 + #294 + #305; channel-enumeration per Catalog #308 alternative-probe-methodologies |
| Pivot enumeration (Revision #4: other base substrate OR G2-PARTIAL) | ADOPT canonical | Per CLAUDE.md "Forbidden premature KILL" + Catalog #308: enumerate ≥3 alternatives before terminal verdict |
| 30-day staleness window override (Revision #7) | UNIQUE FORK | Per CLAUDE.md "Substrate retirement discipline" + Catalog #298 default is TIME-BASED; for ATW V2 specifically the reactivation criteria require >30 days at typical cadence — EVIDENCE-BASED reactivation criteria forked from canonical default; documented in lane registry notes per Catalog #311 |

## 10. Predicted ΔS band per Catalog #296 + Dykstra-feasibility check

**Predicted band**: `NULL pending V2-1 re-probe verdict on RICHEST channel + Dykstra-feasibility check` [prediction; deferred]

Conditional revisions per V2 design memo §18 (preserved + sharpened):
- **IF V2-1 re-probe on scorer-logit channel returns MEANINGFUL_CONDITIONING (MI ≥ 0.5)**: predicted band `[-0.005, -0.015]` rate-axis on A1 baseline (per V2 design memo §18 first-principles-bound via Tishby IB lower bound + Wyner-Ziv side-info savings × A1 rate-axis 0.20) → contest-CPU band [0.181, 0.190] (current canonical CPU best 0.19205 per Catalog #316; would beat by 0.002-0.011)
- **IF V2-1 re-probe returns WEAK_CONDITIONING (0.01 ≤ MI < 0.5)**: revise to `[-0.002, -0.005]` [prediction; downscaled by MI/0.5 ratio] → contest-CPU band [0.187, 0.190]
- **IF V2-1 re-probe returns INDEPENDENT (MI < 0.01)**: paradigm-level falsified per Catalog #307; pivot per Revision #4

**Dykstra-feasibility check**: per V2 design memo §18 + Boyd convex-feasibility lens — the alternating-projections intersection of {rate-feasible: archive_bytes ≤ 300KB} ∩ {seg-feasible: SegNet distortion ≤ A1 baseline} ∩ {pose-feasible: PoseNet distortion ≤ A1 baseline} ∩ {composability-feasible: orthogonal to NSCS06 v8 chroma + DP1 pretraining + D1 SegNet overlay} is NON-EMPTY per the abstract orthogonality verification at V2 §3. The empirical realization is conditional on the V2-1 re-probe outcome.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: predicted band is grounded in Tishby IB lower bound + Wyner-Ziv side-info savings (NOT extrapolated from sister substrate empirical anchors).

## 11. Continual-learning posterior anchor per Catalog #300

This memo's frontmatter satisfies v2 contract. Posterior anchor appended via `tac.council_continual_learning.append_council_anchor` in a follow-up CLI invocation (see "Op-routables" below).

Schema fields populated:
- `deliberation_id`: `atw_v2_reactivation_symposium_20260518`
- `topic`: ATW V2 reactivation per Atick richer-side-info channel V2-1 redesign
- `council_tier`: T2
- `council_attendees`: (10-member tuple per frontmatter)
- `council_quorum_met`: true (6/6 sextet attended + 4/4 grand council attendees)
- `council_verdict`: PROCEED_WITH_REVISIONS
- `council_dissent`: (4-member tuple verbatim per frontmatter)
- `council_assumption_adversary_verdict`: (8-assumption tuple per frontmatter)
- `council_decisions_recorded`: (12-row tuple per frontmatter)
- `predicted_mission_contribution`: frontier_pursuit
- `override_invoked`: false
- `override_rationale`: ""
- `deferred_substrate_id`: atw_codec_v2
- `deferred_substrate_retrospective_due_utc`: 2026-06-17T00:00:00Z
- `related_deliberation_ids`: (8-deliberation tuple per frontmatter)

## 12. Op-routables (ranked)

1. **Operator authorizes V2-1 design memo subagent** — $0 GPU + ~1 week editor; enumerate ≥3 alternative side-info channel hypotheses per Catalog #308 + sextet pact selects channel choice satisfying BOTH MI prediction AND byte budget. Output: `.omx/research/atw_codec_v2_1_redesign_richer_side_info_channel_20260YYY.md` (subagent assigns date).

2. **Operator runs continual-learning posterior append** for this symposium memo:
   ```
   .venv/bin/python -c "from tac.council_continual_learning import CouncilDeliberationRecord, CouncilTier, append_council_anchor; ..."
   ```
   (canonical helper invocation; the subagent that authorized this memo should run this AFTER memo lands; if not, operator runs manually per Catalog #128/#131 fcntl-locked discipline).

3. **Operator queues V2-1 re-probe subagent task** — depends on Z6 Wave 2 Candidate 4c outcome + C6 IBPS Phase 2 redesign outcome per Revisions #5 + #6. Conditional spawn: after V2-1 design memo lands AND Z6 4c outcome AND C6 Phase 2 redesign outcome are known, queue a sister subagent for the new D4 probe execution.

4. **Lane registry reactivation criteria update** — per Revision #7: lane `lane_atw_codec_v2_substrate_build_20260516` notes update to EVIDENCE-BASED criteria (V2-1 redesign + new D4 probe + Wave N+1 council BEFORE any dispatch attempt; do not rely on 30-day natural expiration). Operator runs:
   ```
   .venv/bin/python tools/lane_maturity.py mark lane_atw_codec_v2_substrate_build_20260516 \
       --gate three_clean_review \
       --evidence "T2 symposium 2026-05-18 PROCEED_WITH_REVISIONS; reactivation criteria EVIDENCE-BASED per memo Revision #7"
   ```

5. **Bidirectional cross-pollination wiring** — Z6 Wave 2 Candidate 4c subagent (`a58961ea35f767306`) outcome IS empirically informative; C6 IBPS Phase 2 redesign subagent (TBD) outcome IS empirically informative. ATW V2-1 designer MUST read both outcomes as PV-1 + PV-2 per Catalog #229 premise-verification-before-edit pattern.

6. **Catalog #325 naming-canonicalization** — per `council_decisions_recorded` decision #12: operator chooses (a) rename symposium memo to include `atw_codec_v2` substring OR (b) the gate accepts `atw_v2` as substrate_alias. Recommend (b) per Catalog #315 substrate_aliases mechanism.

7. **30-day retrospective due 2026-06-17** per Catalog #300 Consequence 3 — audit ATW V2 V2-1 redesign + new D4 probe outcome at that time; was the reactivation path executed? Did the new D4 probe land MEANINGFUL_CONDITIONING? Did Wave N+1 ratify PROCEED-unconditional? Did Modal dispatch fire? Result feeds the mission_contribution distribution audit per Consequence 5.

## 13. Premise verification per Catalog #229

PV-0 sister coordination: 4 ATW V2 sister-relevant subagents checked at session start. No live ATW V2 sister; #849 driver-fix is cross-substrate audit (may touch ATW V2 driver but disjoint from this memo's scope); #856 just completed.

PV-1 D4 probe verdict INDEPENDENT MI=0.006385 verified verbatim from `.omx/state/probe_outcomes.jsonl` + `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md`.

PV-2 ATW V2 trainer `_full_main` IS implemented per sister `atw_d4_probe_recipe_disambiguation_20260516.md` premise verification (28/28 tests pass); NO NotImplementedError. Recipe `dispatch_enabled: false + research_only: true` honored.

PV-3 Atick Z6 Phase 3 verbatim recommendation "enrich the side-information channel" verified verbatim from `council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517.md`.

PV-4 Z6 4-candidate Path B memo Table 5 row for Candidate 4b explicitly cites "DEFERRED-PENDING-REDESIGN (empirically blocked by ATW v2 D4 predecessor probe); requires richer-side-info redesign per Atick recommendation BEFORE this sub-option becomes empirically credible" — verified verbatim from `z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517.md`.

PV-5 Tishby IB framework analysis (H(T) = 1 bit/pair × 600 pairs / 95K symbols ≈ 0.006 bits/symbol matches observed MI) verified via arithmetic.

PV-6 Wyner-Ziv source-coding bound (R_WZ(D) = R_{X|S}(D)) + canonical shippable-bytes constraint (≤2KB per V2 design memo §10 archive grammar) verified via V1 + V2 design memos.

PV-7 Catalog #313 30-day staleness window expiration date 2026-06-15 verified via `.venv/bin/python tools/check_predecessor_probe_outcome.py --substrate atw_codec_v2 --json` output.

PV-8 Frontier citation per Catalog #316 — current canonical best is 0.19205 [contest-CPU] / 0.20533 [contest-CUDA] verified via MEMORY.md top entry + sister memo cross-references.

## 14. Checkpoint trace

1 checkpoint registered to `.omx/state/subagent_progress.jsonl` per Catalog #206:
- Step 1: PV-0 sister coord + reads of doctrine + C6 IBPS + Z6 Phase 3 + D4 verdict + design memo + recipe; canonical pre-flight per CLAUDE.md "Subagent coherence-by-default" satisfied
- Step complete (this memo lands): symposium deliberation + frontmatter v2 + verdict + 7 binding revisions + 8-assumption cargo-cult audit + 9-dim checklist + observability + cross-pollination + decision tree + op-routables

## 15. Cross-references

- Sister doctrine landing memo (Catalog #325 establishment): `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md`
- C6 IBPS FIRST per-substrate symposium (sister): `.omx/research/council_c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518.md`
- Z6 Phase 3 sextet council (Atick Candidate 4c verbatim): `.omx/research/council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517.md`
- Z6 4-candidate Path B redesign memo (Table 5 Candidate 4b cross-reference): `.omx/research/z6_v2_cargo_cult_unwind_4_candidate_redesign_path_b_20260517.md`
- ATW V2 design memo (V1-inherited 3-knob + V2-distinguishing G1+B3+G2-PARTIAL): `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`
- ATW V2 D4 probe verdict (INDEPENDENT MI=0.006385 anchor): `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md` + `.omx/state/h_latent_given_scorer_class_atw_codec_v2.json`
- ATW V2 D4 probe recipe disambiguation (sister): `.omx/research/atw_d4_probe_recipe_disambiguation_20260516.md`
- Grand council T3 batched Phase 2 lift (ATW v2 PROCEED-TO-D4-PROBE-FIRST-THEN-LIFT verdict): `.omx/research/grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md`
- Catalog #313 canonical helper + ledger: `tac.probe_outcomes_ledger` + `.omx/state/probe_outcomes.jsonl`
- Catalog #325 STRICT preflight gate + 6-step contract: `src/tac/preflight.py::check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor`
- Catalog #316 frontier-anchor scan: `tools/scan_best_anchor_per_axis.py` + `tac.frontier_scan`
- Catalog #245 Modal call-id ledger: `tac.deploy.modal.call_id_ledger` + `.omx/state/modal_call_id_ledger.jsonl`
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable section
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" non-negotiable
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" Rule 3 (cheap signal gates expensive signal)
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
- CLAUDE.md "Council conduct" Fix-7 amendment (per-round explicit-assumption-statement discipline)
- CLAUDE.md "Council hierarchy: 4-tier protocol" + Mission alignment non-negotiable subsection
- CLAUDE.md "Max observability" non-negotiable
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305.

This memo produces structured observability for downstream consumers (sister of §4 above):

| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| Symposium verdict + frontmatter v2 | `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (this file) | per Catalog #300 v2 frontmatter | Operator + downstream V2-1 redesign subagent + Wave N+1 council |
| Continual-learning posterior anchor | `.omx/state/council_deliberation_posterior.jsonl` (appended via `tac.council_continual_learning.append_council_anchor`) | per Catalog #128/#131/#300 | Cathedral autopilot Hook 4 + Rashomon ensemble + Assumption-Adversary classification-stability monitor |
| Lane registry mark + notes | `.omx/state/lane_registry.json` + `.omx/state/lane_maturity_audit.log` | per `tools/lane_maturity.py` schema | Lane registry validator + Catalog #325 gate + Catalog #298 retirement audit |
| Catalog #313 probe outcome status quo (UNCHANGED) | `.omx/state/probe_outcomes.jsonl` (no new row this memo) | per probe-outcomes-ledger schema | Catalog #313 gate + autopilot dispatch refusal |
| Subagent checkpoint trail | `.omx/state/subagent_progress.jsonl` per Catalog #206 | per checkpoint schema | Crash-resume successor (if any) + parent audit |

### Observability invariants

- **No score_claim**: this memo records a SYMPOSIUM DELIBERATION + V2-1 REACTIVATION-PATH RECOMMENDATION, not a score result. `score_claim=false` implicit; no axis label applies because no score was measured.
- **No phantom directories** per Catalog #249: memo path is canonical `.omx/research/` per CLAUDE.md.
- **Cite-chain preserved**: all 8 related deliberations cited in frontmatter `related_deliberation_ids`.
- **Counterfactual-able**: a future agent can verify the cross-pollination prediction by reading Z6 Wave 2 Candidate 4c outcome + C6 IBPS Phase 2 redesign outcome AND running the new D4 probe per V2-1 redesign channel choice; the empirical disambiguator answers EACH cargo-cult-classified assumption.
- **Frontier citation per Catalog #316**: pinned in frontmatter `canonical_frontier_anchor`.

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

NOT APPLICABLE — Catalog #311 enforces ego-motion-conditioned next-frame prediction for cooperative-receiver framing in PREDICTIVE-CODING substrates (Z6/Z7/Z8 architecture class). ATW V2 is a CODEC substrate (Wyner-Ziv side-info conditioning for the LATENT codec, not next-frame prediction). The cooperative-receiver theorem applies at the encoder-decoder level, not at the temporal-prediction level. Per Catalog #311's own scope: ego-motion conditioning is for substrates "applying Atick-Redlich 1990 cooperative-receiver framing to dashcam video" via next-frame prediction; ATW V2 applies the same theorem to latent compression. Different architectural surface; different empirical disambiguator.

Per V2 design memo §3 orthogonality verification: ATW V2 binds Atick-Redlich + Tishby IB + Wyner-Ziv at the CODEC layer (compress-side encoder + decoder + side-info), NOT the predictor layer. Catalog #311 ego-motion conditioning requirement is structurally not applicable. NO waiver needed — the gate's scope correctly excludes ATW V2 by architecture-class.

## Bottom-line summary

- **Symposium VERDICT**: PROCEED_WITH_REVISIONS — 7 binding revisions (Contrarian + Assumption-Adversary + Atick + Wyner_memorial)
- **Paradigm classification**: IMPLEMENTATION-LEVEL falsification per Catalog #307 (cooperative-receiver theorem INTACT; per-pair-segnet-argmax-composite channel-on-A1-latents is the specific implementation falsified)
- **Reactivation path**: V2-1 redesign (~1 week subagent + $0 GPU) → new D4 probe ($3-5 CPU) → Wave N+1 council ($0 GPU) → CONDITIONAL Modal A100 paired dispatch ($10-30) — total envelope $13-35 inclusive of probe
- **Cross-pollination wired**: BIDIRECTIONAL with Z6 Wave 2 Candidate 4c (scorer-logit conditioning at predictor surface) + C6 IBPS Phase 2 redesign (scorer-conditioning at variational decoder surface)
- **Pivot path enumerated**: IF V2-1 re-probe ALSO INDEPENDENT on RICHEST channel → operate on different base substrate (PR101/PR106 latents) OR G2-PARTIAL alternative — each pivot is its own symposium
- **Catalog #325 satisfied** for `substrate=atw_codec_v2` for next 14 days
- **NO COMMITS per parent prompt** ✓
- **NO Modal/Lightning/Vast.ai dispatches** ✓
- **Total cost**: $0 GPU, ~3h editor
- **Lane**: `lane_per_substrate_symposium_atw_v2_reactivation_20260518` L1 (impl_complete + three_clean_review + memory_entry after operator marks gates)

## 2026-05-18 Codex byte-closed V2-1 side-info probe addendum

Codex landed and ran the V2-1 byte-budget side-info probe requested by this
symposium's next-action chain:

- Tool: `tools/probe_atw_v2_1_byte_closed_side_info_channel.py`
- Tests: `src/tac/tests/test_probe_atw_v2_1_byte_closed_side_info_channel.py`
- JSON: `.omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.json`
- Markdown: `.omx/research/atw_v2_1_byte_closed_side_info_probe_20260518_codex.md`
- Packet output dir:
  `experiments/results/atw_v2_1_sideinfo_probe_20260518T062431Z`
- Command: `.venv/bin/python tools/probe_atw_v2_1_byte_closed_side_info_channel.py`
- Axis: `[diagnostic-CPU; ATW V2-1 byte-closed side-info MI probe]`
- Score authority: `score_claim=false`, `promotion_eligible=false`,
  `ready_for_paid_dispatch=false`
- Dispatch/spend: none

Result:

| Channel | Packet bytes | Budget ok | MI bits/symbol | Threshold | Verdict |
|---|---:|---|---:|---:|---|
| `per_pixel_histogram` | 204 | true | 0.022656927447 | 0.5 | WEAK_CONDITIONING |
| `per_region_histogram` | 323 | true | 0.047381530305 | 1.0 | WEAK_CONDITIONING |
| `per_pair_class_2_fraction` | 127 | true | 0.009692520351 | 0.2 | INDEPENDENT |
| `per_frame_argmax` | 117 | true | 0.000000000000 | 0.2 | INDEPENDENT |

Interpretation:

- The <=2KB Wyner-Ziv side-info byte budget is **not** the blocker for the
  current richer reducer artifacts. Dictionary-coded ATW21SI packets are tiny:
  best channel `per_region_histogram` is 323 bytes, with rate score cost
  `0.000215072442`.
- The conditioning signal is still too weak to authorize ATW Phase 2:
  best observed MI is `0.047381530305` bits/symbol, far below the
  `1.0` bit meaningful threshold for the per-region reducer and below the
  global `MEANINGFUL_CONDITIONING` standard needed before paid dispatch.
- The next gate changes from "build byte-closed side-info probe" to
  `design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe`.
  That means either preserve a richer scorer-logit signal before the histogram
  reducer collapses it, or rerun the probe on trained ATW residuals rather than
  A1 HNeRV latents. No Modal A100 spend is justified from this evidence.
