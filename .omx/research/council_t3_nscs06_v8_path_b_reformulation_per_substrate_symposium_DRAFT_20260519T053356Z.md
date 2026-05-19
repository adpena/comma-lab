---
schema: council_deliberation_v2
deliberation_id: council_t3_nscs06_v8_path_b_reformulation_per_substrate_symposium_DRAFT_20260519
topic: "NSCS06 v8 Path B reformulation per-substrate symposium DRAFT — wavelet-residual hierarchical compression on v7 chroma-anchored substrate; supersedes 2026-05-17 + 2026-05-18 prior symposiums with T3 DRAFT for operator-routable ratification"
review_kind: per_substrate_optimal_form_symposium_T3_grand_council_DRAFT
review_date: "2026-05-19"
lane_id: lane_cable_c_substrate_symposium_draft_batch_20260519
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Carmack, Hotz, Mallat, Daubechies, Wyner_memorial, Quantizr, Schmidhuber, Boyd, MacKay_memorial]
council_quorum_met: false
council_verdict: DRAFT_PENDING_OPERATOR_CONVOCATION
council_dissent:
  - member: Contrarian
    verbatim: "DRAFT only — supersedes prior NSCS06 v8 Path B symposiums (2026-05-17 REFUSE + 2026-05-18 PROCEED_WITH_REVISIONS Variant C reactivation). NSCS06 v6 EMPIRICAL FALSIFICATION (105.15 at archive sha 03ef568bc918; predicted band [0.10, 0.20]; 553x miss) IS the canonical anchor. Per CLAUDE.md FORBIDDEN PATTERN 'Forbidden NO-neural-at-medal-band-assumption (the strip-everything-medal-class trap)': the NO-neural medal-band assumption was PARTIALLY DISPROVED by NSCS06 v7 (105.15 → 58.89 in ONE iteration via per-class chroma anchors functioning as quasi-neural priors). The reformulation question is whether SURGICAL addition of wavelet-residual or per-subband CDF on RESIDUAL ONLY (preserving v7 chroma-anchor) can break the medal-band threshold."
  - member: Hotz
    verbatim: "NSCS06 lineage is the canonical NO-neural-at-medal-band test bed. 6 prior failures (v1-v6) + v7 +44% rescue + v8 -78% regression (cargo-cult composition fail) IS the empirical staircase. The reformulation MUST preserve v7's chroma-preserving primitive AND apply v8 cargo-cult-unwinds SELECTIVELY (NOT all-at-once). Per CLAUDE.md 'NEW META-CARGO-CULT #8 architectural-transition-preserves-previously-unwound-cargo-cults': dual-direction empirical anchor v6→v7 +44% / v7→v8 -78% IS hard-earned. Variant C-1 (v7 anchor + DB4 depth-1 residual only) is the canonical minimal-perturbation reformulation."
council_assumption_adversary_verdict:
  - assumption: "v7's per-class chroma anchor IS the substrate-class-shift carrier (NOT v6's grayscale strip)"
    classification: HARD-EARNED
    rationale: "Empirical receipts: v6 (grayscale strip; chroma destroyed) = 105.15; v7 (chroma anchored per-class) = 58.89 (44% improvement in ONE iteration). The chroma anchor IS the substrate-class-shift carrier. v8 Path A (further chroma refinements) preserves it; v8 Path B (wavelet residual added) MUST preserve it via surgical-addition-to-residual-only architecture."
  - assumption: "Wavelet-residual hierarchical compression (DB4 depth=1) preserves v7's chroma anchor when applied to residual only"
    classification: HARD-EARNED-PARTIAL
    rationale: "Per Mallat 2026-05-18 verbatim (Variant C-1 reactivation): DB4 wavelet decomposition on RESIDUAL (frame - v7_anchor_reconstruction) preserves anchor via byte-level decomposition. The residual is the high-frequency signal NOT captured by v7's chroma anchor; DB4 wavelet compression on residual is THEORETICALLY orthogonal to anchor. Empirical disambiguator: Variant C-1 smoke MUST verify residual-only compression preserves frame-level reconstruction to within v7 baseline."
  - assumption: "Predicted band [45, 55] (Variant C-1) / [50, 58] (Variant C-2) / [40, 50] (Variant C-3) is calibrated"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Predicted bands derived from v7 = 58.89 anchor + theoretical DB4 / per-subband-CDF / UNIWARD-conditioned savings extrapolations. Per Catalog #324: predicted_band_validation_status MUST be pending_post_training; per Catalog #316: comparison vs canonical frontier 0.19205 [contest-CPU] / 0.20533 [contest-CUDA] required AT smoke-time. Even optimistic Variant C-3 [40, 50] is FAR above contest-CPU frontier 0.19 — NSCS06 family is currently NO-PROMOTION territory."
  - assumption: "NSCS06 family deserves continued reformulation investment despite contest-CPU frontier 0.19205 being 200x BELOW v7 baseline"
    classification: CARGO-CULTED-PENDING-OPERATOR-DECISION
    rationale: "Operator-routable: NSCS06 NO-neural-at-medal-band hypothesis IS empirically FALSIFIED at the medal-class level (medal-class ≤ 0.20). Even theoretical breakthrough Variant C reformulation [40-58] is 200x ABOVE medal class. Per CLAUDE.md 'Forbidden premature KILL': NSCS06 family should be REFRAMED as 'methodology validation substrate' (cargo-cult-unwind discipline test bed) per Variant C-4 design memo, NOT score-lowering substrate. Operator-decision: continue NSCS06 reformulation OR retire to research_only=true."
  - assumption: "Variant C-1 (DB4 depth=1 residual-only) is the cheapest disambiguator path"
    classification: HARD-EARNED
    rationale: "Per Mallat + Daubechies 2026-05-18 verbatim: DB4 wavelet decomposition is the lowest-LOC + fewest-parameter addition. ~$5-10 Modal T4 smoke envelope. Variant C-2 + C-3 are higher-cost; Variant C-4 is design-only no GPU."
council_decisions_recorded:
  - "DRAFT enumerates 6-step Catalog #325 contract for NSCS06 v8 Path B reformulation"
  - "Variant C-1 (DB4 depth=1 residual-only) recommended as Wave 1 smoke disambiguator"
  - "Per CLAUDE.md 'Forbidden premature KILL': NSCS06 family REFRAMED to research_only=true unless operator continues funding"
  - "Operator-routable: ratification mechanism choice + decide if NSCS06 family deserves continued investment OR retire"
  - "Per Catalog #316: even theoretical breakthrough [40] is 200x ABOVE contest-CPU frontier 0.192 — NO promotion-eligibility under any predicted band"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: nscs06_v8_path_b_wavelet_residual
substrate_aliases:
  - nscs06_v8_path_b
  - nscs06_v9_variant_c
  - nscs06_carmack_hotz_strip_everything
deferred_substrate_retrospective_due_utc: "2026-06-18T05:33:56Z"
horizon_class: plateau_adjacent
predicted_band:
  variant_c1_v7_anchor_plus_db4_depth1_residual_only: [45, 55]
  variant_c2_v7_anchor_plus_per_subband_class_cdf_residual_only: [50, 58]
  variant_c3_v7_anchor_plus_uniward_conditioned_db4_residual: [40, 50]
  variant_c4_methodology_extension_K_coverage_validation_only: null
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
dispatch_enabled: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.1920513169 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.2053300290 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
predecessor_probe_outcomes:
  - probe_id: nscs06_v6_dispatch
    verdict: EMPIRICAL_FALSIFICATION
    final_score: 105.15
    predicted_band: [0.10, 0.20]
  - probe_id: nscs06_v7_rescue
    verdict: PARTIAL_RESCUE
    final_score: 58.89
    improvement_vs_v6: "44%"
  - probe_id: nscs06_v8_path_b_dispatch
    verdict: EMPIRICAL_FALSIFICATION_REGRESSION
    final_score: 104.98
    improvement_vs_v7: "-78%"
related_deliberation_ids:
  - council_per_substrate_symposium_nscs06_v8_path_b_20260517
  - council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518
  - nscs06_path_a_chroma_optical_flow_redesign_20260516
  - nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516
  - nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516
  - grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516
---

# DRAFT: T3 grand council symposium — NSCS06 v8 Path B reformulation

**Status**: DRAFT — operator-convocation pending. NOT a binding council verdict.
**Lane**: `lane_cable_c_substrate_symposium_draft_batch_20260519` L1
**Per Catalog #325**: this DRAFT satisfies the 6-step contract structurally; full convocation activates symposium evidence per Catalog #325 14-day window.
**Supersession**: this DRAFT supersedes 2026-05-17 REFUSE memo + 2026-05-18 PROCEED_WITH_REVISIONS Variant C reactivation memo by re-elevating to T3 DRAFT format with **CRITICAL operator-decision** layered: NSCS06 family's NO-neural-at-medal-band hypothesis is empirically falsified at medal-class — should the family continue receiving investment OR retire to research_only=true?

## Symposium attendees (proposed)

**Sextet pact**:
- **Shannon LEAD** — information-theoretic capacity of v7-chroma + wavelet-residual hierarchical decomposition
- **Dykstra CO-LEAD** — convex-feasibility of multi-constraint composition (chroma + wavelet + UNIWARD)
- **Yousfi** — inverse-steganalysis UNIWARD-conditioned wavelet residual (Variant C-3)
- **Fridrich** — STC parity-check codes for wavelet subband bit-allocation
- **Contrarian** — VETO power on lazy NSCS06 family continuation
- **Assumption-Adversary** — challenges medal-band hypothesis itself

**Grand council added per topic**:
- **Carmack** — strip-everything engineering shortcut authority
- **Hotz** — don't-chase-failed-paradigm engineering instinct
- **Mallat** — wavelet decomposition canonical authority (DB4 / multi-scale)
- **Daubechies** — Daubechies wavelet family + compressive-sensing canonical authority
- **Wyner_memorial** — Wyner-Ziv source-coding canonical framing for residual sidechannel
- **Quantizr** — adversarial reverse-engineering vs PR101/PR106 frontier
- **Schmidhuber** — compression-as-intelligence + MDL framework
- **Boyd** — convex-feasibility of bit-allocation across wavelet subbands
- **MacKay_memorial** — MDL + Bayesian inference for per-subband prior selection

## Step 1 — Cargo-cult audit per Catalog #303

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-nscs06-1 | "v6's grayscale-strip is the substrate-class-shift carrier" | HARD-EARNED-FALSIFIED | v7 +44% empirical anchor REFUTES — chroma anchor IS the carrier, NOT grayscale strip. v6 was DESTROYING the carrier. |
| CC-nscs06-2 | "v8 all-at-once cargo-cult-unwinds compose monotonically" | HARD-EARNED-FALSIFIED | v7→v8 -78% empirical anchor REFUTES — NEW META-CARGO-CULT #8 architectural-transition-preserves-previously-unwound-cargo-cults. v8 regressed unwinds #1+#2 while unwinding #3+#6. |
| CC-nscs06-3 | "DB4 wavelet on RESIDUAL preserves v7 chroma anchor" | HARD-EARNED-PARTIAL | Mallat 2026-05-18 verbatim: theoretically orthogonal. Empirical disambiguator: Variant C-1 smoke MUST verify residual-only compression preserves frame-level reconstruction. |
| CC-nscs06-4 | "NO-neural medal-band achievable at NSCS06 substrate class" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | v6 105.15 / v7 58.89 / v8 104.98 — ALL >300x above contest-CPU frontier 0.19205. Even theoretical breakthrough Variant C-3 [40] is 200x ABOVE. NSCS06 family does NOT reach medal-class under any predicted-band variant. |
| CC-nscs06-5 | "NSCS06 family deserves continued reformulation investment" | OPERATOR-DECISION | Per CLAUDE.md 'Forbidden premature KILL': NSCS06 deserves reframing as 'methodology validation substrate' (cargo-cult-unwind discipline test bed) per Variant C-4 design memo. Operator decides: continue funding OR retire to research_only=true. |
| CC-nscs06-6 | "Per-subband per-class CDF (Variant C-2) is sister-substrate-engineering of v7 chroma anchor" | HARD-EARNED | Per MacKay_memorial: Bayesian per-class prior selection at per-subband granularity is canonical MDL approach. Higher-LOC than Variant C-1 but theoretically tighter. |
| CC-nscs06-7 | "UNIWARD-conditioned DB4 residual (Variant C-3) is dominant at low-byte-budget" | HARD-EARNED-PARTIAL | Per Yousfi + Fridrich 2026-05-18 verbatim: UNIWARD weighting puts bits in HIGH-variance regions where errors are undetectable; DB4 residual is high-frequency by construction. Theoretical optimization but empirically untested at NSCS06 scale. |

## Step 2 — 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-symposium-DRAFT evidence |
|---|---|---|
| 1 | UNIQUENESS | ✓ NSCS06 = the ONLY NO-neural substrate family in the codebase. Variant C-1/C-2/C-3 are architecturally distinct within NSCS06 lineage. Plateau_adjacent class per Catalog #309. |
| 2 | BEAUTY + ELEGANCE | ✓ Variant C-1 (DB4 depth=1 residual-only) = minimal-LOC addition to v7 trainer; <200 LOC per HNeRV parity L4 inflate budget. v7 anchor preserved verbatim. |
| 3 | DISTINCTNESS | ✓ Variant C-1 (DB4) vs C-2 (per-subband per-class CDF) vs C-3 (UNIWARD-conditioned DB4) — 3 architecturally orthogonal residual-compression primitives within v7 anchor preservation. |
| 4 | RIGOR | ✓ THIS DRAFT + 2026-05-17 REFUSE + 2026-05-18 PROCEED_WITH_REVISIONS Variant C reactivation = 3 council memos; 6 prior v1-v6 dispatch anchors; v7 +44% empirical rescue; v8 -78% empirical regression; Catalog #303 cargo-cult audit; Catalog #229 PV; Catalog #313 probe-ledger consultation; Catalog #316 frontier preservation. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Variant C-1 IS the minimal-perturbation reformulation per Carmack + Hotz convergent. Variant C-2/C-3 are sister extensions. Variant C-4 is methodology validation only. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ NSCS06 v8 Path B Variant C lineage is sister to D1 SegNet overlay + DP1 pretraining + Z6/Z7 ego-conditioning at the COMPRESS-TIME ANCHOR level. Composable orthogonal to neural substrates. |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ DB4 wavelet decomposition is canonical deterministic. v7 chroma anchor byte-stable per Catalog #5. Variant C-1/C-2/C-3 inherit determinism. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Variant C-1 smoke envelope $5-10 Modal T4 (~30 min wall-clock for NO-neural substrate). Wave 2 full Variant C-1 $10-15. Variant C-2 ~$15. Variant C-3 ~$20. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | NEGATIVE — predicted bands [40, 58] are 200x ABOVE contest-CPU frontier 0.19205. NO promotion-eligibility under any Variant C predicted band. NSCS06 family is NOT score-lowering substrate. |

## Step 3 — Observability surface declaration per Catalog #305

**Per-NSCS06-v8-Path-B-Variant-C observability**:
1. **Inspectable per layer**: per-pair v7 chroma anchor reconstruction (preserved bytes) + per-pair DB4 residual decomposition (4 subbands LL/LH/HL/HH) + per-subband bit-allocation + per-subband CDF (Variant C-2) + per-pair UNIWARD weighting (Variant C-3)
2. **Decomposable per signal**: v7 baseline contribution vs residual contribution to final score + per-subband compression ratio + per-class anchor preservation invariant
3. **Diff-able across runs**: paired comparison v7 vs Variant C-1 at SAME pair_indices; Variant C-1 vs C-2 vs C-3 at SAME pair_indices
4. **Queryable post-hoc**: per-config Modal call_id ledger row per Catalog #245 + per-config probe-outcome ledger row per Catalog #313 + per-config build_manifest.json per Catalog #220
5. **Cite-able**: cite v6 EMPIRICAL_FALSIFICATION + v7 PARTIAL_RESCUE + v8 EMPIRICAL_FALSIFICATION_REGRESSION + Variant C-1/C-2/C-3 paired comparison
6. **Counterfactual-able**: "what if DB4 → DB2 vs DB6 vs DB8?" + "what if per-subband CDF granularity = per-frame vs per-class vs per-region?" + "what if UNIWARD weighting source = scorer-derived vs hand-tuned?"

## Step 4 — Sextet pact deliberation (DRAFT positions)

### Shannon LEAD position (DRAFT)

*"Operating-within assumption: v7 chroma anchor IS the substrate-class-shift carrier per v7 +44% empirical anchor. The information-theoretic question is whether DB4 wavelet residual decomposition recovers high-frequency information NOT captured by chroma anchor. Wavelet theory (Mallat 1989) says YES — DB4 is asymptotically optimal for piecewise-smooth signals; dashcam frames are largely piecewise-smooth. PROCEED on Variant C-1 DRAFT design. STRONG RECOMMENDATION: NSCS06 family should be reframed as methodology validation NOT score-lowering substrate — predicted bands are 200x ABOVE medal class."*

### Dykstra CO-LEAD position (DRAFT)

*"Operating-within assumption: multi-constraint composition (chroma anchor preservation + wavelet decomposition + UNIWARD weighting). Convex feasibility requires INTERSECTION of feasible regions. Variant C-1 (just chroma + wavelet) has 2 constraints; Variant C-3 (chroma + wavelet + UNIWARD) has 3. Dykstra alternating-projection feasibility check REQUIRED: Variant C-1 is convex-feasible by construction (orthogonal axes); Variant C-3 needs explicit check that UNIWARD weighting doesn't violate chroma anchor preservation. APPROVE Variant C-1; CONDITIONAL APPROVE Variant C-3 pending Dykstra feasibility-check probe."*

### Yousfi position (DRAFT)

*"UNIWARD-conditioned DB4 residual (Variant C-3) is canonical inverse-steganalysis approach: weight loss by inverse local variance puts bits in HIGH-variance regions where scorer errors are undetectable. STRONG RECOMMENDATION Variant C-3 as Wave 2 path after Variant C-1 baseline empirical anchor lands."*

### Fridrich position (DRAFT)

*"STC parity-check codes for wavelet subband bit-allocation IS canonical per Filler 2026-05-17 lineage. PROCEED on Variant C DRAFT design. STRONG RECOMMENDATION: explicit STC-codes for per-subband payload allocation as Wave 3+ refinement."*

### Contrarian position (DRAFT)

*"Operating-within assumption: NSCS06 family is empirically FALSIFIED at medal-class. v6 / v7 / v8 ALL >300x above contest-CPU frontier 0.19205. Even theoretical breakthrough Variant C-3 [40] is 200x ABOVE. Per CLAUDE.md 'Forbidden premature KILL': NSCS06 DEFER per Catalog #298 (NOT KILL). VETO any DRAFT path that pre-authorizes Wave 2 paid dispatch as 'score-lowering substrate'. Variant C IS methodology validation; operator decides if methodology validation deserves continued GPU budget."*

### Assumption-Adversary position (DRAFT) [Catalog #291 + #292]

*"Operating-within assumption (META): NSCS06 family deserves continued investment despite empirical failure at medal-class. The SHARED ASSUMPTION is that NO-neural medal-band is achievable at SOME NSCS06 variant. Per CLAUDE.md FORBIDDEN PATTERN 'Forbidden NO-neural-at-medal-band-assumption': this hypothesis is CARGO-CULTED-EMPIRICALLY-FALSIFIED. The HARD-EARNED reframing is NSCS06 as 'methodology validation substrate' (cargo-cult-unwind discipline test bed). Per CLAUDE.md 'Forbidden premature KILL': NSCS06 deserves DEFER not KILL; reactivation criterion = NEW evidence that NO-neural medal-band is achievable at a DIFFERENT substrate class. THIS DRAFT MUST recommend research_only=true reframing OR operator-frontier-override per Catalog #300."* — VETO if not engaged.

### Carmack position (DRAFT)

*"v7's chroma anchor IS the strip-everything insight applied correctly. v8 cargo-cult-stack regression IS proof that strip-everything must be applied SURGICALLY not BULK. Variant C-1 IS the canonical strip-everything reformulation per HNeRV parity L7 substrate-engineering budget. PROCEED on Variant C-1 design."*

### Hotz position (DRAFT)

*"Operating-within assumption: don't-chase-failed-paradigm. NSCS06 family has 6 empirical failures + 1 partial rescue + 1 catastrophic regression. The methodology validation framing IS the canonical answer. STRONG RECOMMENDATION: reframe to research_only=true; redirect $20-50/month NSCS06 budget to higher-EV paradigms (Z6 Wave 2 / DP1 stacking / time_traveler L5 / C6 IBPS Phase 2)."*

### Mallat position (DRAFT)

*"DB4 wavelet decomposition on residual is canonical asymptotic-optimal for piecewise-smooth signals. PROCEED on Variant C-1 DRAFT design. STRONG RECOMMENDATION: depth=1 sufficient for empirical disambiguator; depth>=2 only justified if Variant C-1 empirical anchor lands within predicted band."*

### Daubechies position (DRAFT)

*"DB4 (4-tap) is the canonical Daubechies wavelet for compactly-supported orthonormal decomposition. Sister DB6 / DB8 have wider support → better smoothness but more boundary artifacts on small frames. PROCEED on Variant C-1 DB4 choice."*

### Wyner_memorial position (DRAFT)

*"Memorial seat conveying Wyner-Ziv 1976 source-coding-with-side-info canonical framing. Wavelet residual IS encoder-decoder shared-info channel: encoder computes residual against v7 anchor; decoder regenerates v7 anchor + receives wavelet-compressed residual. The Wyner-Ziv source-coding theorem applies — residual compression is BOUNDED BELOW by H(residual | v7_anchor). Empirical question: is H(residual | v7_anchor) tractable at DB4 depth=1? Variant C-1 IS the empirical probe."*

### Quantizr position (DRAFT)

*"Adversarial reverse-engineering: PR106 format0d (canonical CUDA frontier 0.20533) IS at medal class. NSCS06 v7 (58.89) is 287x ABOVE. NSCS06 family does NOT reach medal class under any predicted-band variant. The reformulation is METHODOLOGY VALIDATION not score-lowering. STRONG RECOMMENDATION: align with Hotz reframing to research_only=true UNLESS operator explicitly continues funding."*

### Schmidhuber position (DRAFT)

*"Compression-as-intelligence: NSCS06 family is empirical exploration of NO-neural medal-band hypothesis. The hypothesis is FALSIFIED at medal-class. The METHODOLOGY (cargo-cult-unwind discipline; per-class anchor preservation; surgical residual addition) IS VALUABLE for future neural substrates. PROCEED on Variant C-1/C-4 DRAFT design; methodology validation IS the canonical contribution."*

### Boyd position (DRAFT)

*"Convex-feasibility of per-subband bit-allocation: wavelet subbands (LL/LH/HL/HH at depth=1) are mutually orthogonal; bit-allocation across subbands is convex problem solvable by Lagrangian or ADMM. Variant C-1 IS minimal feasible bit-allocation; Variant C-2 (per-class CDF) adds Bayesian structure (canonical MacKay framing). PROCEED on Variant C-1 / C-2 DRAFT."*

### MacKay_memorial position (DRAFT)

*"Memorial seat: MDL framework. Per-subband per-class CDF (Variant C-2) is canonical Bayesian prior selection. The CDF approximates per-class wavelet coefficient distribution; arithmetic coding against this CDF approaches per-class entropy floor. PROCEED on Variant C-2 design IF Variant C-1 empirical anchor lands within predicted band."*

## Step 5 — Per-substrate reactivation criteria pinned per Catalog #298 + #308

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

| Stage | If verdict | Reactivation path |
|---|---|---|
| Variant C-1 smoke | smoke score < 55 (within predicted band) | Wave 2 full Variant C-1 + Variant C-2 paired sweep |
| Variant C-1 smoke | smoke score >= 55 (worse than v7 baseline 58.89) OR crashes | DEFER Variant C-1 per Catalog #298; reactivation = explicit chroma-anchor-preservation invariant violation diagnostic |
| Variant C-1 full | full score does NOT improve vs v7 baseline 58.89 | DEFER NSCS06 family per Catalog #298 → reactivation = NEW evidence that NO-neural medal-band is achievable at DIFFERENT substrate class |
| All Variant C variants | All fail to improve vs v7 baseline | NSCS06 family research_only=true per Catalog #298 → reactivation = NEW operator directive OR NEW paradigm emerges |

## Step 6 — Catalog #324 post-training Tier-C validation discipline

Recipe declares `predicted_band_validation_status: pending_post_training`. Reactivation criterion: post-training Tier-C density measurement on Variant C-1 archive after Wave 2 full dispatch via `tools/mdl_scorer_conditional_ablation.py --tier c`. Predicted bands [40, 58] are research priors; promotion-eligibility under ANY band is structurally UNAVAILABLE per Catalog #316 frontier comparison (200x above 0.19205).

## Operator-routable decisions

**Decision 1**: Continue NSCS06 reformulation investment OR retire to research_only=true?
- **CONTINUE** ($20-50/month methodology validation; Variant C-1 smoke $5-10 + full $10-15 = $15-25 Wave 1 envelope)
- **RETIRE** to research_only=true; redirect budget to Z6 Wave 2 / DP1 stacking / time_traveler L5 / C6 IBPS Phase 2

**Decision 2** (if CONTINUE): convocation mechanism choice
- Full T3 convocation ($0 editor; ~3h council deliberation)
- Inner-quintet pact ratification ($0 editor; ~1h)
- Operator-frontier-override per Catalog #300 Consequence 1

## Cross-substrate dependencies

- **Sister cargo-cult-unwind methodology** validated on Z6 v6→v7 +44% rescue lineage (cross-substrate methodology contribution)
- **Sister gate Catalog #303** (per-substrate cargo-cult audit): NSCS06 IS canonical test bed
- **Sister gate Catalog #298** (substrate retirement discipline): NSCS06 deserves DEFER not KILL

## Predicted cost per Variant per Wave

- Variant C-1 smoke: $5-10 (Modal T4 ~30 min)
- Variant C-1 full: $10-15 (Modal T4 ~2 hours)
- Variant C-2 smoke + full: ~$15
- Variant C-3 smoke + full: ~$20
- Variant C-4 design only: $0

## Continual-learning posterior anchor

Per Catalog #300 + `tac.council_continual_learning.append_council_anchor`: this DRAFT must emit v2 posterior anchor at convocation. `deferred_substrate_id` = `nscs06_v8_path_b_wavelet_residual`; `predicted_mission_contribution` = `rigor_overhead` (methodology validation NOT frontier-breaking); retrospective due 2026-06-18T05:33:56Z.
