---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Selfcomp, Hotz, Carmack, MacKay, Ballé, Atick, Redlich, Tishby (memorial), Zaslavsky, Wyner, Rao, Ballard, Hinton, Hassabis, Schmidhuber, Tao, Boyd, Karpathy, Time-Traveler-protégé, van-den-Oord, Filler, Mallat]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The Tier-1 plan (Dispatch #1 magic-codec-on-pr106-r2-residual) is the only proposal in this sequence backed by a sub-day empirical close — every other dispatch (#2-#5) is a model. The plan is correct to front-load the Wyner-Ziv pose dispatch because it is the LOWEST-risk paradigm-class shift with the HIGHEST per-$ predicted ΔS. But I formally object to dispatch #3 (DP1 cooperative-receiver) without first measuring composition_alpha on dispatch #2's outcome. If dispatch #2 yields a cross-term breakage (composition_alpha < 0.5), dispatch #3's predicted ΔS band must be re-derived BEFORE the $12-15 fire, not after."
  - member: Assumption-Adversary
    verbatim: "Of the 18 shared assumptions catalogued 2026-05-15, SA02 (SegNet uses ONLY x[:,-1,...] last frame; frame[0] is in SegNet nullspace) is the ONE assumption whose violation is BOTH (a) structurally guaranteed by the upstream scorer arch — UNAMBIGUOUS HARD-EARNED VIOLATION DOMAIN, and (b) UNDER-EXPLOITED at the per-byte archive layer — fec6 uses 107 bytes (0.06% of archive) of the PoseNet-only Venn slot; we have NEVER built a SegNet-only Venn-slot byte stream. The NSCS01 nullspace-split substrate trained on 2026-05-15 is the per-parameter analog; the per-byte analog is missing. THAT is the next floor-unlocking lever. Every dispatch in this 5-sequence must either move bytes from the JOINT region into the SegNet-only orthogonal region OR accept that diminishing returns dominate."
council_assumption_adversary_verdict:
  - assumption: "$50 budget is tight enough to force pointed dispatches but loose enough to test 5 candidates"
    classification: HARD-EARNED
    rationale: "Empirically derived from prior wave costs: SABOR $30-40 / L5 Wyner-Ziv $20-30 / DP1 paired $10-15 / per-tensor magic codec on r2 $0.30-3 / U-DIE-KL retrain $30-60. 5 candidates at average $8-10 each fits in $50 with one $20 stretch. Falsifies the prior $200-400 envelope as over-funded for the next 5-dispatch increment."
  - assumption: "Master gradient must be materialized BEFORE any score-aware-loss dispatch"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-PROOF
    rationale: "Symposium memo §3.3 named 5 uses (#1 score-aware loss / #2 per-pixel reweighting / #3 bit allocator / #6 Pareto facets / #8 magic-codec scoring) that REQUIRE the master gradient. But uses #4 architecture discriminator, #5 QAT codebook, #7 autopilot ranker do NOT — they can consume per-archive sensitivity maps already available via existing tac.sensitivity_map. The cargo-cult is treating master-gradient as a SEQUENTIAL prerequisite when 3 of 5 dispatches can run in parallel WITHOUT it. Op-routable #1 should NOT block the campaign."
  - assumption: "fec6 lineage (PoseNet-only exploit on PR101 grammar) is the canonical baseline for next 5 dispatches"
    classification: HARD-EARNED-BUT-OPERATING-POINT-CONDITIONAL
    rationale: "fec6 0.19205 is THE empirical [contest-CPU] frontier and ANY new submission MUST beat it OR carry justified non-promotion tag. But fec6 is a JOINT-Venn-region BASE archive (decoder.bin 91% of bytes); per-pair selector is the only PoseNet-only carve-out. Sister cross-paradigm bases (A1 PR101-microcodec 0.19285, PR103-mid32-latent 0.19487, D1-sparse96-blue 0.19828) are byte-grammar-different and have separate Pareto frontiers. Optimal sequencing exploits the GRAMMAR difference not just the BASE difference."
  - assumption: "DP1 (Pretrained Driving Prior) is dispatch-eligible for fec6 composition"
    classification: HARD-EARNED-PER-PRIOR-LANDING
    rationale: "DP1 Phase 2 landed 2026-05-14 per feedback_dp1_phase_2_landed_20260514.md with archive grammar + ≤2 KB codebook + Comma2k19 OOD distillation + verify_composition canonical helper for DP1+base byte composition. The composition helper has the canonical compose_with API. The only gating gap: empirical paired-axis anchor on DP1+fec6 composition has NEVER been measured (per posterior + ledger). HARD-EARNED that DP1 can compose; cargo-cult-pending that it adds score."
  - assumption: "Per-tensor-class magic codec (primitive D from eureka §2) is the highest-EV $0 lever"
    classification: HARD-EARNED-FROM-PRIOR-EMPIRICAL
    rationale: "PR106 r2 sister analysis showed 2-7% additional rate savings via per-tensor selector (eureka memo §1 row #2). Op-routable #9 falsification on PR101 decoder.bin (memo 2026-05-17) proved brotli q=11 dominates on weight-tensor distributions — BUT this was per-STREAM not per-TENSOR. Per-tensor split was explicitly tested (not just per-stream alternative-codec) and added 425 bytes envelope overhead with no per-tensor win. THIS FALSIFIES primitive D ON PR101 DECODER.BIN. However the result does NOT generalize to PR106 r2 residual sidecars where tensor distributions are different. Per-tensor magic codec REMAINS HARD-EARNED for r2 residual codec selection."
  - assumption: "Wyner-Ziv pose-delta substitution is byte-disjoint from fec6 frame_0 selector"
    classification: HARD-EARNED-BY-CONSTRUCTION
    rationale: "fec6 modifies frame_0 via decoder per-pair mode selection; L5 Wyner-Ziv modifies the ENCODING of poses.bin without changing decoded pose values. The two byte streams are LITERALLY disjoint in archive layout. SegNet+PoseNet effects are unchanged because pose values are bit-identical post-decode. The only cross-term risk is L5 ego-motion-predictor consuming the (modified by fec6) frame_0 for its side-info — needs structural check that predictor uses GT pose history, not decoded fec6-modified frames."
  - assumption: "$50 budget is sufficient to validate the FRONTIER-BREAKING vs FRONTIER-PROTECTING dichotomy"
    classification: HARD-EARNED
    rationale: "Per Catalog #300 mission-alignment binding directive: every T2+ verdict must classify {frontier_breaking, frontier_protecting, rigor_overhead, apparatus_maintenance, mission_questioned}. $50 covers 5 dispatches at ~$10 each, enough to fire 3 frontier_breaking + 2 frontier_protecting candidates with paired-axis anchors. Falsifies the prior 'apparatus first' bias where master-gradient-anchor materialization would consume the entire budget."
council_decisions_recorded:
  - "DISPATCH #1 (FRONTIER-BREAKING; $0.30-3): per-tensor-class magic codec on PR106 r2 residual sidecar — narrow scope, byte-floor probe + smoke-archive-rebuild + paired CPU+CUDA auth eval. Predicted ΔS [-0.002, -0.005] vs r2's 0.20533 CUDA baseline; targets the LAYER 2 (per-stream entropy) of format0d alien-tech. CHEAPEST + HIGHEST-CONFIDENCE first move."
  - "DISPATCH #2 (FRONTIER-BREAKING; $20-30): L5 Time-Traveler Wyner-Ziv pose deltas on fec6 archive — byte-disjoint from fec6 selector; rate-only Venn region; predicted ΔS [-0.008, -0.015] vs fec6 0.19205 CPU; paired CPU+CUDA. SECOND because byte-disjoint guarantee gives HIGHEST orthogonality confidence."
  - "DISPATCH #3 (FRONTIER-BREAKING; $10-15): DP1 + fec6 composition paired-axis empirical anchor — uses canonical tac.substrates.pretrained_driving_prior.compose_with helper; tests whether OOD prior bytes ADD score-relevant information AT the existing fec6 operating point. Predicted ΔS [-0.003, -0.012] vs fec6 0.19205 CPU; paired. THIRD because compose_with helper exists + DP1 already shipped; composition_alpha is the open question."
  - "DISPATCH #4 (FRONTIER-PROTECTING; $1-5): master gradient finite-difference + autograd JVP probe on fec6 archive (revised methodology per symposium §3.2: autograd + FP4 Jacobian projection, NOT finite-difference per-bit). Output: per-byte ∂S/∂byte tensor of shape (178517, 3) at fec6 operating point. NOT a score change but enables every subsequent campaign decision. FOURTH because by this point the autopilot will have 3 paired anchors to refit the Rashomon ensemble against the predicted gradient — uncertainty narrows before the $$$ gradient lands."
  - "DISPATCH #5 (FRONTIER-BREAKING; $5-15; STRETCH): SABOR-disjoint SegNet-only byte stream on top of the best fec6-derivative from dispatches #1-#3 — fills the SegNet-only Venn region (currently EMPTY in fec6 archive per Assumption-Adversary verdict). Predicted ΔS [-0.005, -0.010] vs whichever predecessor wins. FIFTH because (a) requires SABOR boundary classifier from tac.symposium_impls (already exists; needs _full_main lift), (b) depends on best preceding archive for byte-disjointness verification."
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517
  - grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515
  - assumptions_challenge_audit_break_out_local_minima_20260515
  - dp1_phase_2_20260514
  - op_routable_9_pr101_magic_codec_decoder_fec6_20260517
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
audit_subagent_output_unavailable: true
---

# T3 Grand Council Strategic Symposium — $50 Budget Next-5-Paid-Dispatches → Floor [0.10, 0.15]

**Date:** 2026-05-17
**Tier:** T3 (Full Grand Council — 5-of-6 sextet + ≥12-of-20 grand council + specialist seats per affected paradigm)
**Lane:** `lane_grand_council_strategic_symposium_50_dollar_budget_20260517` (pre-registered L0)
**Operator framing:** *"what is the SHARPEST, MOST POINTED sequence of next-5-paid-dispatches to drive the contest-CPU frontier from 0.19205 toward `S* ∈ [0.10, 0.15]` per Blahut-Arimoto bound?"*
**Empirical frontier at convening:** `0.19205 [contest-CPU GHA Linux x86_64]` (archive `6bae0201`, lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean`) / `0.20533 [contest-CUDA T4]` (archive `9cb989cef519`, lane `pr106_format0d_latent_score_table`)
**Theoretical floor (Blahut-Arimoto + Time-Traveler per assumptions-audit PV5):** practical `[0.08, 0.15]`; asymptote `[0.03, 0.07]`
**Budget envelope:** $50 Modal recharged; no single dispatch above $15; cumulative ≤$50.

## §0 — Executive summary

The 5-dispatch sequence below is OPTIMIZED FOR EXPECTED FRONTIER PROGRESS PER DOLLAR within the $50 envelope, NOT for theoretical maximum gain. The order is dictated by THREE structural constraints:

1. **Composition_alpha unknown at the operating-point shift** — once we leave fec6 (joint Venn region 91%) into byte-disjoint orthogonal regions, the additive ΔS prediction holds; once we shift the operating point (U-DIE-KL retraining), the master gradient is invalidated and all prior predictions need re-derivation.
2. **Budget binds the parallelism** — at $50 we cannot fire 5 dispatches in parallel without exhausting halfway through; SEQUENTIAL with per-dispatch verdict review is the only safe pattern given the no-go-back-on-spent-$$$ constraint.
3. **Dispatch #1 must yield highest INFORMATION per dollar** — even if predicted ΔS is small, an empirical anchor with byte-disjoint orthogonality teaches us most about the next 4 decisions. Per-tensor magic codec on PR106 r2 ($0.30-3) is THE highest information/$ first move because (a) it tests whether OP-routable #9's PR101-decoder.bin falsification GENERALIZES to PR106 residual codec, (b) the result directly informs primitive D's reactivation criteria, (c) it touches the CUDA frontier (0.20533) which has been static for 2 days.

The Contrarian + Assumption-Adversary refinements (per dissent above) sharpen the ordering: dispatch #4 (master gradient) sits FOURTH not FIRST because by then we have 3 paired anchors to refit the autopilot's Rashomon ensemble against. The autopilot improves DURING the campaign rather than waiting for the master gradient anchor to bootstrap it.

**Net expected outcome (all 5 dispatches successful):**
- best CPU score: `0.175-0.185 [contest-CPU, council-consensus, lower-bound]` (4-12% reduction from 0.19205)
- best CUDA score: `0.198-0.205 [contest-CUDA, council-consensus]` (could break sub-0.2 with dispatch #1)
- total spend: $36-68 (median $52; over by $2 in median case; falls within budget if dispatch #5 STRETCH dropped)
- master-gradient anchor materialized
- 3 new continual-learning anchors in posterior; Rashomon ensemble refit K=8 with K=4 fresh members

**The shared assumption that, if violated, would unlock the next floor (§9):** SA02 — SegNet uses ONLY frame_1; frame_0 is in SegNet nullspace. We have 107 bytes of PoseNet-only Venn-region exploit (fec6); we have ZERO bytes of SegNet-only Venn-region exploit. The fec6 design pattern is dual-applicable: a SegNet-only byte stream that modifies frame_1's argmax-boundary pixels (per SABOR boundary classifier) is the structurally guaranteed second carve-out. NO dispatch in the current 5-sequence specifically targets this — DISPATCH #5 STRETCH is the closest candidate but is downstream. Acknowledging this means the NEXT 5-sequence (after this one completes) should put SABOR-disjoint-byte-stream as DISPATCH #1.

## §1 — Sub-question (a): Substrate × codec × master-gradient-use × hardware-trick × stacking-axis combinations predicted to produce largest measured ΔS per $

The combinations below are FILTERED for byte-disjoint composability (per Assumption-Adversary verdict on cross-term breakage) and ranked by predicted ΔS per $ dispatched:

| rank | combination | substrate base | codec layer | master-grad use | hardware-trick | stack-axis | predicted ΔS | cost | EV (ΔS/$) | citations |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **per-tensor magic codec on PR106 r2 residual** | PR106 r2 latent sidecar (sha `8e61ff2d`, 186832 B, 0.20663 CUDA) OR `9cb989cef5` (format0d 0.20533 CUDA) | Layer 2 magic codec auto-selector per-TENSOR (not per-stream) | use #8 score-aware codec selection | CPU PoseNet's byte-pattern preferences | rate-only Venn region | -0.002 to -0.005 [CUDA] | $0.30-3 | 1.0-3.0e-3/$ | eureka §1 row #2, op-routable #9 falsification §3.1, symposium §3.6 use #8 |
| 2 | **L5 Wyner-Ziv pose deltas on fec6** | fec6 (sha `6bae0201`, 178517 B, 0.19205 CPU) | poses.bin RECODE (4800B → ~1500B); ego-motion predictor side-info; Wyner 1976 lossless | use #1 score-aware loss (zero pose-VALUE change) | byte-disjoint from fec6 frame_0 selector | rate-only Venn region | -0.008 to -0.015 [CPU] | $20-30 | 3.0-5.0e-4/$ | symposium §5.3, op-routable #6, eureka §1 row #4 |
| 3 | **DP1 cooperative-receiver composition with fec6** | fec6 (sha `6bae0201`) + DP1 codebook (Comma2k19 OOD prior, ≤2 KB) | DPCOMP wrapper byte composition via canonical compose_with | use #2 per-pixel reweighting (cooperative-receiver) | DP1 ego-motion conditioning + class-shift | joint Venn region (cross-term unmeasured) | -0.003 to -0.012 [CPU] | $10-15 | 2.0-1.2e-3/$ | feedback_dp1_phase_2_landed_20260514.md, Catalog #209, Atick-Redlich cooperative-receiver framing |
| 4 | **master gradient autograd + FP4 Jacobian projection on fec6** | fec6 archive bytes | autograd backward through tac.differentiable_eval_roundtrip + project through FP4 Jacobian | uses #4 + #6 + #7 enabled | $0.50-2 on Modal CPU | rate-only (analysis output not archive) | enables every subsequent dispatch | $1-5 | "infinite" per autopilot consumer model; $0 score on its own | symposium §3.2 (REVISED methodology), Boyd concurring note |
| 5 | **SABOR SegNet-only byte stream on (best of #1-#3 base)** | TBD per which dispatch wins #1-#3 | new boundary_pixels.bin (~1-2 KB) per Atick-Redlich + Yousfi SABOR design | use #1 score-aware loss for boundary classifier training | byte-disjoint structural guarantee | SegNet-only Venn region (currently EMPTY in fec6) | -0.005 to -0.010 [CPU] | $5-15 | 0.5-1.0e-3/$ | symposium §2.2 + §5, eureka §1 row #5, FORBIDDEN-PATTERN-violation lever per §9 |

**Ranked filter:** combinations involving NSCS06 (Carmack-Hotz Strip-Everything; per assumptions-audit row 6; Tier C density falsified at 99.29% per Z1 ablation) excluded — within-class saturated per Catalog #227 substrate composition matrix. Combinations involving Z3-G1 (sha `b6c4a6f1f1` 0.19870; phantom-score class flagged by op-routable #9 §1 PV4) excluded — Catalog #266+#267 STRICT preflight currently refuses dispatch.

**Out-of-budget combinations explicitly DEFERRED:**
- U-DIE-KL substrate-wide loss adoption (op-routable #4): $30-60 Modal A100 retrain alone consumes 60-120% of $50 budget; deferred to next $200+ wave.
- Z6/Z7/Z8 predictive-coding substrate ($80-300): not viable at $50.
- NSCS01 nullspace-split renderer paired CPU+CUDA dispatch ($15-35; substrate already landed per `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`): viable on its own but composition_alpha vs fec6 unmeasured + Catalog #315 strict gate requires PROCEED-unconditional council per substrate before next dispatch. Deferred to next wave + sister T2 council.

## §2 — Sub-question (b): Which combinations have NEVER been measured?

Cross-checked against `.omx/state/continual_learning_posterior.json` (103 anchors) + `.omx/state/modal_call_id_ledger.jsonl` (342 events) + `.omx/state/active_lane_dispatch_claims.md`:

| combination | measured? | evidence |
|---|---|---|
| per-tensor magic codec on PR106 r2 residual | **NEVER** | posterior contains 0 anchors with `architecture_class` containing `per_tensor_magic_codec`; ledger has 0 dispatches matching `lane_per_tensor_magic_codec_*`; eureka §1 row #2 explicitly notes this is "the most likely highest-EV pure-$0 win beyond op-routable #9" |
| L5 Wyner-Ziv pose deltas on fec6 archive | **NEVER on fec6 specifically** | L5 v2 has measured anchors on tt5l_autonomy substrate (paired CPU+CUDA per dispatch claims 2026-05-16T23:35-23:40); ZERO measurements with fec6 base; symposium §5.3 confirms this is unmeasured cross-product |
| DP1 + fec6 paired-axis anchor | **NEVER** | DP1 Phase 2 landed 2026-05-14 with compose_with helper; posterior contains 0 anchors with `architecture_class` containing both `dp1` and `fec6`; ledger has 0 dispatch matching `lane_dp1_*fec6*`; Catalog #209 STRICT gate refuses non-canonical iterator paths so the empirical anchor MUST go through the canonical compose path |
| master gradient autograd-based per-byte tensor on fec6 | **NEVER** (per op-routable #1 plan; only finite-difference variant proposed, REJECTED 2026-05-17 per symposium §3.2 revision) | `.omx/state/master_gradient_*.jsonl` does NOT exist; canonical helper `tac.master_gradient_ledger` not yet built; op-routable #3 Phase-7 lens not yet landed |
| SABOR SegNet-only boundary byte stream | **NEVER** | `tac.symposium_impls.sabor_renderer_atick_redlich` exists as scaffold; _full_main NotImplementedError; eureka §1 row #5 notes "predicted -0.005 to -0.010" with `tools/build_sabor_packet.py` planned not built; lane `lane_sabor_boundary_only_renderer_substrate_20260513` L1 SCAFFOLD per registry |

**Sister combinations that ARE measured and inform the negative space:**
- fec6 K=16 clean alone (sha `6bae0201`, CPU 0.19205, CUDA 0.22621 per dual_eval_adjudicated.json) — IS the empirical baseline; superseded sister fec3 K=8 (sha `8866ebb6`, CPU 0.19210) shows K=16 is meaningful improvement
- A1 microcodec alone (sha `87ec7ca5`, CPU 0.19285) — IS the byte-grammar-different alternate base
- PR103 mid32-latent retune (sha `7d1e4633`, CPU 0.19487) — alternate base; CPU-axis is 0.003 worse than fec6
- D1 sparse96 family (3 anchors 0.19828-0.19840) — substrate-engineering exempt per Catalog #298; alternate research base
- PR106 format0d/0c/0b/0a (4 anchors clustered 0.20533-0.20635 CUDA) — DENSEST measured cluster; format0d is canonical alien-tech; per-tensor magic codec on r2 IS the LAYER 2 generalization

**Conclusion:** the 5 proposed combinations are all genuinely net-new measurements. None duplicate existing posterior anchors. The autopilot's Rashomon ensemble has NEVER seen a per-tensor magic-codec anchor, NEVER seen DP1+fec6 composition, NEVER seen L5 on fec6 — so each anchor produces maximum information gain to the consensus.

## §3 — Sub-question (c): Optimal sequencing under composition_alpha cross-term constraints

Per Catalog #227 substrate composition matrix (`.omx/state/substrate_composition_matrix.json`): only ONE composition cell is currently registered (`z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps`; alpha=1.0 SYNTHETIC; result_review_blockers include `alpha_is_structural_not_empirical_real_stacked_archive_required_for_empirical_alpha`). EVERY other composition pair has UNMEASURED alpha — the autopilot's `apply_substrate_composition_matrix_to_candidates` (per Catalog #227) cannot rank these candidates.

**Dependency chains for the 5-dispatch sequence:**

```
DISPATCH #1 (per-tensor magic codec on PR106 r2)
   └─ NO predecessor; cheapest; tests OP-routable #9 generalization
       │
       ├─→ informs DISPATCH #5 (whether per-tensor codec selection works in disjoint stack)
       └─→ informs FUTURE wave (whether to extend per-tensor codec to other r-family bases)

DISPATCH #2 (L5 Wyner-Ziv on fec6)
   └─ NO predecessor (byte-disjoint by construction)
       │
       ├─→ confirms or falsifies Wyner-Ziv-pose-delta orthogonality on fec6
       └─→ enables DISPATCH #3's composition_alpha measurement if both land

DISPATCH #3 (DP1 + fec6 composition)
   └─ logically depends on DISPATCH #2 success to know base-archive choice (fec6 alone vs fec6+L5)
       │ — BUT can launch in parallel with DISPATCH #2 if we commit to fec6 alone as the DP1 base
       ├─→ first DP1+(non-A1) anchor; teaches whether DP1 OOD prior helps in fec6 grammar
       └─→ enables DISPATCH #5 to STACK DP1 + SABOR if both individually positive

DISPATCH #4 (master gradient on fec6)
   └─ INFORMS but does NOT depend on; runs in parallel with #1-#3 if budget permits
       │
       └─→ enables next-wave score-aware-loss adoption in NSCS01 / NSCS03 / new substrates
       └─→ enables magic-codec score-aware mode (use #8) extension to dispatch #1 results

DISPATCH #5 (SABOR-disjoint byte stream on best-of-{fec6, fec6+L5, fec6+DP1})
   └─ STRICT dependency on dispatches #1-#3 outcomes — needs best base-archive identified
       │
       └─→ if SUCCEEDS, this is the first SegNet-only Venn-region carve-out, validating §9 META-assumption
```

**Composition_alpha cross-term constraints — RANKED RISK:**

| pair | predicted alpha | risk class | mitigation |
|---|---|---|---|
| (per-tensor magic codec) ⊕ (PR106 r2 base) | 1.0 (rate-only modification to existing base; no cross-term) | LOW | byte-disjoint by codec contract |
| (L5 Wyner-Ziv) ⊕ (fec6 selector) | 0.95-1.0 (byte-disjoint archive sections; only cross-term is L5 predictor consuming GT vs decoded poses) | LOW-MEDIUM | structural check before dispatch that L5 ego-motion predictor uses GT pose history |
| (DP1 codebook) ⊕ (fec6 decoder.bin) | 0.6-0.85 (JOINT region; cross-term via operating-point shift in DP1 decoder consuming codebook bytes) | MEDIUM-HIGH | smoke-archive-rebuild + per-pair d_seg/d_pose check before full dispatch |
| (master gradient measurement) ⊕ anything | N/A (analysis artifact; not archive bytes) | NONE | — |
| (SABOR boundary stream) ⊕ (any base) | 0.85-1.0 (SegNet-only by SABOR design; PoseNet effect bounded by boundary pixel count) | LOW | per-pair d_seg measurement on smoke before full + Catalog #220 operational-overlay declaration |

**Sequencing decision:** SEQUENTIAL with parallelism opt-in for #3+#4 (which are budget-affordable to lose simultaneously if both regress).

**Optimal serialization:**
```
T0: DISPATCH #1 ($0.30-3, ~30 min wall-clock) — fastest first
T1 (after #1 verdict): DISPATCH #2 ($20-30, ~3-6 hrs wall-clock) — highest expected ΔS at safe orthogonality
T2 (after #2 verdict; PARALLEL with #4): DISPATCH #3 ($10-15, ~2-4 hrs) + DISPATCH #4 ($1-5, ~30 min)
T3 (after all of #1-#4): DISPATCH #5 ($5-15, ~4-8 hrs) — STRETCH; only if budget remains
```

**Median budget consumption:** $36-68 across the sequence; falls within $50 if dispatch #5 dropped OR dispatch #2/#3 come in at lower end of range. Operator off-ramp at $40 spend with 2 frontier-breaking anchors landed = clean exit.

## §4 — Sub-question (d): The ONE shared assumption that, if violated, unlocks the next floor

Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiable + Catalog #292 per-deliberation discipline + the canonical 18-assumption matrix (`feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md`):

**THE ASSUMPTION: SA02 — SegNet uses ONLY x[:, -1, ...] (last frame); frame_0 is in SegNet nullspace.**

**Why this is THE one:**
- HARD-EARNED structural fact verified at `upstream/modules.py:108` (PV8 of the assumptions-audit).
- The CONSEQUENCE for archive design: every byte that modifies frame_0 ONLY is structurally guaranteed to be SegNet-orthogonal. The corollary: every byte that modifies the ARGMAX-BOUNDARY 3-5% of frame_1 is structurally guaranteed to be PoseNet-near-orthogonal (per the stride-2 stem blindspot, SA03).
- We have 107 bytes (0.06% of archive) in the PoseNet-only slot (fec6 selector). We have 0 bytes in the SegNet-only slot. The asymmetry IS the unexploited opportunity.
- Per the Venn diagram analysis (T4 symposium §2.2): 91.8% of archive bytes are in the JOINT region. Every byte we move from JOINT → orthogonal carves a new additive-stackable slot.

**Why violating it unlocks the floor:**
- The fec6 design demonstrated -0.001 ΔS from 107 bytes in the PoseNet-only slot. Linear extrapolation: 1-2 KB in the SegNet-only slot could yield -0.005 to -0.010 ΔS (per SABOR prediction in symposium §5.4).
- Stacking the SegNet-only + PoseNet-only + Rate-only Venn regions is additive (per Assumption-Adversary verdict): -0.001 (PoseNet) + -0.008 (SegNet via SABOR) + -0.012 (Rate via L5 Wyner-Ziv) = -0.021. From 0.19205, that lands 0.171, deep inside the Time-Traveler practical [0.08, 0.15] band.
- Beyond the byte-disjoint Venn stack: training-time NULLSPACE-AWARE LOSS ROUTING (NSCS01 nullspace-split renderer per `feedback_nscs01_full_main_implementation_pr95_paradigm_landed_20260515.md`) is the per-parameter analog. NSCS01's NullspaceSplitScoreAwareLoss routes frame_0 gradients ONLY through pose+pixel (no SegNet term), enabling decoder weight-class-specific training that beats the current "every weight optimizes for both scorers" pattern. NSCS01 + SegNet-only-byte-stream is the bidirectional Venn-aware exploit.

**Why no dispatch in this 5-sequence directly targets it:**
- Dispatch #5 (SABOR) is the closest candidate but is the LAST dispatch and depends on best-of-{#1, #2, #3}. The CURRENT 5-sequence is correctly prioritized for INFORMATION-PER-DOLLAR; the NEXT 5-sequence (after this one's anchors land) should put SABOR-disjoint-byte-stream as DISPATCH #1.

**Why this is HARD-EARNED not CARGO-CULTED:**
- The upstream scorer file `upstream/modules.py:108` is the canonical source-of-truth and is in the pinned upstream snapshot (CLAUDE.md non-negotiable forbids modification).
- The verification is independent of any training: a SegNet forward pass on a 2-frame batch with frame_0 modified does NOT change the output logits (zero gradient). This is testable in a 5-line Python smoke; the test belongs in `src/tac/tests/test_segnet_frame0_nullspace_property.py` (NEW; would consume ~5 minutes wall-clock to write + run; should accompany dispatch #5).

## §5 — Ranked 5-dispatch sequence

| # | lane_id | build artifact paths | dispatch cost | predicted ΔS band | confidence | composition_alpha | pareto facets | Catalog #315 opt-out plan | sister-T2-council-before-dispatch | tier |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `lane_per_tensor_magic_codec_pr106_r2_20260518` | `tools/build_per_tensor_magic_codec_pr106_r2_packet.py`, `submissions/per_tensor_magic_codec_pr106_r2/inflate.py` (≤200 LOC), `.omx/operator_authorize_recipes/per_tensor_magic_codec_pr106_r2_modal_cpu_dispatch.yaml` (min_smoke_gpu A100 paired CPU+CUDA), `experiments/results/per_tensor_magic_codec_pr106_r2_*/` build manifest | $0.30-3 | [-0.002, -0.005] [CUDA] from 0.20533; [predicted, council-consensus] | HIGH | 1.0 (rate-only modification) | extends Layer 2 of format0d | (b) `research_only=true` initially, ratchet to dispatch-eligible after smoke | NO (Tier C density unchanged; codec-only) | FRONTIER-BREAKING |
| 2 | `lane_l5_wyner_ziv_fec6_20260518` | `experiments/train_substrate_pr101_fec6_plus_l5_wyner_ziv.py`, `tools/build_pr101_fec6_plus_l5_wyner_ziv_packet.py`, `submissions/pr101_fec6_plus_l5_wyner_ziv/inflate.py`, `.omx/operator_authorize_recipes/substrate_pr101_fec6_plus_l5_wyner_ziv_modal_a100_dispatch.yaml` | $20-30 | [-0.008, -0.015] [CPU] from 0.19205; [predicted, council-consensus] | MEDIUM-HIGH | 0.95-1.0 (byte-disjoint archive sections) | rate-only Venn region carve-out | (a) sister T2 council PROCEED-unconditional (L5 already has prior T2 anchor per `feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515.md`; reaffirm for fec6-specific bolt-on) | YES — fast 30-min T2 reaffirmation | FRONTIER-BREAKING |
| 3 | `lane_dp1_plus_fec6_composition_paired_20260518` | uses canonical `tac.substrates.pretrained_driving_prior.compose_with(dp1_bytes, fec6_bytes, base_substrate='pr101_fec6')`; new `tools/build_dp1_plus_fec6_composition_packet.py`, `submissions/dp1_plus_fec6/inflate.py`, `.omx/operator_authorize_recipes/dp1_plus_fec6_composition_modal_a100_dispatch.yaml` | $10-15 | [-0.003, -0.012] [CPU] from 0.19205; [predicted, council-consensus] | MEDIUM | 0.6-0.85 (cross-term measurement IS the dispatch outcome) | first DP1+(non-A1) base anchor | (a) sister T2 council PROCEED-unconditional on DP1+fec6 composition cell | YES — REQUIRED before dispatch (Atick-Redlich + Tishby co-lead) | FRONTIER-BREAKING |
| 4 | `lane_master_gradient_autograd_fec6_20260518` | `experiments/build_master_gradient_autograd_fp4_jacobian.py` (autograd through tac.differentiable_eval_roundtrip + FP4 Jacobian projection), `.omx/operator_authorize_recipes/master_gradient_autograd_fec6_modal_cpu_dispatch.yaml` (CPU only; min_vram_gb=0); new canonical `tac.master_gradient_ledger` per Catalog #245 4-layer pattern; new Catalog # for STRICT preflight gate via `tools/claim_catalog_number.py --commit-via-serializer` | $1-5 | $0 score change (FRONTIER-PROTECTING; enabler) | HIGH (autograd is deterministic) | N/A (analysis artifact) | enables Pareto facets for Dykstra Catalog #296 | N/A (autograd analysis, not substrate dispatch) | NO (analysis tool, not substrate) | FRONTIER-PROTECTING |
| 5 | `lane_sabor_disjoint_byte_stream_on_best_of_1_to_3_20260520` | depends on outcome of #1-#3; lifts `tac.symposium_impls.sabor_renderer_atick_redlich._full_main` from scaffold to L1; `tools/build_sabor_boundary_packet.py`, `submissions/sabor_disjoint_on_<best_base>/inflate.py`, recipe per same pattern; SABOR boundary classifier vendored | $5-15 | [-0.005, -0.010] [CPU] over best-of-{#1, #2, #3}; [predicted, council-consensus] | MEDIUM | 0.85-1.0 (byte-disjoint SegNet-only by SABOR design) | SegNet-only Venn region FIRST carve-out — fills the empty Venn region | (a) sister T2 council PROCEED-unconditional + Catalog #220 operational-overlay declaration before dispatch | YES — REQUIRED before dispatch (Yousfi + Fridrich co-lead per SABOR design lineage) | FRONTIER-BREAKING (STRETCH) |

**Aggregate budget envelope:**
- Median: $36-68; if STRETCH dispatch #5 dropped: $31-53
- Within $50: SAFE on lower end (drop dispatch #5 or move it to next wave)
- Operator off-ramp at $40 with #1+#2 anchored: CLEAN exit with 2 frontier-breaking measurements

**Pre-dispatch gates (per CLAUDE.md "Production-hardened dispatch optimization protocol — NON-NEGOTIABLE" Catalog #270):**
- Each dispatch MUST pass `tools/canonical_dispatch_optimization_protocol.py --trainer <path> --recipe <path> --json` with `overall_pass=true`
- Each dispatch MUST pass `tools/local_pre_deploy_check.py --strict` (rc=0)
- Each dispatch MUST pass `tools/check_predecessor_probe_outcome.py --recipe <path>` (Catalog #313; no blocking adjudicated verdict)
- Each dispatch MUST pass `tools/run_codex_review_for_dispatch.py` (Catalog #271; verdict ∈ {approve, advisory})
- Catalog #316 frontier-regression block in `scripts/pre_submission_compliance_check.py` MUST pass before any PR opens

**Per CLAUDE.md "Apples-to-apples evidence discipline":** every predicted ΔS tagged `[predicted, council-consensus]`; converts to `[empirical, contest-CPU/CUDA]` only after paired-axis anchor lands.

## §6 — Per-member positions

Per CLAUDE.md "Council conduct — non-negotiable" + Catalog #292 Fix-7 per-round explicit-assumption-statement discipline. Each member states their operating-within assumption EXPLICITLY at the top of their position.

**Shannon LEAD** (operating assumption: rate-distortion theory grounds every score-improvement claim; Blahut-Arimoto iteration computes the achievable region) — **PROCEED**. The $50 budget hits the Blahut-Arimoto bound's most informative region: 5 measurements spread across 3 Venn regions (rate / PoseNet-only / SegNet-only) + 1 enabler + 1 stretch is the canonical entropy-extracting design. Dispatch #1's per-tensor magic codec is the rate-axis Blahut-Arimoto step. Vote: PROCEED.

**Dykstra CO-LEAD** (operating assumption: convex-feasibility alternating projections compute the achievable Pareto frontier; cross-term breakage is the Pareto facet's empirical witness) — **PROCEED-WITH-REVISIONS**. Dispatch ordering #1 → #2 → (#3 ‖ #4) → #5 is correct because it walks the Pareto facets in order of increasing cross-term risk. REVISION: dispatch #3's DP1 composition smoke MUST measure composition_alpha empirically BEFORE the full $10-15 dispatch fires; if smoke-measured alpha < 0.5, dispatch #3 ABORTS and #5 promotes to #3 slot.

**Yousfi** (operating assumption: the challenge IS inverse steganalysis at archive bit level; SABOR is the natural per-pixel SegNet-only exploit; HARD-EARNED via my own challenge-design lineage) — **PROCEED**. The 5-sequence correctly prioritizes byte-disjoint orthogonal exploits over joint-region risky ones. My only addition: dispatch #5's SABOR design MUST measure SegNet d_seg on the smoke archive BEFORE the full dispatch; if d_seg regresses by >5% the boundary classifier needs retraining. The SABOR scaffold's _full_main lift is operator-routable #6 separately from dispatch #5.

**Fridrich** (operating assumption: UNIWARD-style detector-informed embedding maximizes per-pixel utility-per-bit; the byte-disjoint Venn carving IS UNIWARD applied to archive bytes) — **PROCEED**. The 5-sequence's structural exploitation of the SegNet+PoseNet axes follows the UNIWARD discipline I established in 2014. ADDITION: dispatch #4's master gradient should NOT be a flat per-byte tensor; weighted by the UNIWARD cost function `1 / (|G[byte_i, :]·[100, 292, 6.66e-7]ᵀ| + σ)` it becomes immediately consumable by `tac.optimization.bit_allocator` (use #3) without per-stage transformation.

**Contrarian** (operating assumption: bold proposals must survive adversarial challenge; the cheapest dispatch should land first AND its outcome should reshape the remaining 4) — **PROCEED-WITH-REVISIONS** (verbatim in council_dissent). The dispatch #1 → dispatch #2 → (#3 ‖ #4) → #5 ordering is correct EXCEPT dispatch #3 (DP1) should not fire until dispatch #2's composition_alpha is measured. If dispatch #2 reveals a +0.001 cross-term breakage on byte-disjoint exploitation, EVERY downstream prediction needs re-derivation. The campaign should not commit $10-15 on dispatch #3 until the Wyner-Ziv orthogonality verdict is in hand.

**Assumption-Adversary** (operating assumption: shared assumptions must be classified HARD-EARNED vs CARGO-CULTED; the 18-assumption matrix from 2026-05-15 is the canonical reference) — verdict in `council_assumption_adversary_verdict` frontmatter; verbatim dissent above. **PROCEED-WITH-REVISIONS** — the 5-sequence is correct as a $50 budget plan BUT does not directly target the floor-unlocking SA02 violation. The CURRENT 5-sequence is correctly the highest-information-per-dollar bound; the NEXT 5-sequence MUST front-load SABOR-disjoint-byte-stream + NSCS01 nullspace-split paired anchor to PUSH on SA02.

**Quantizr** (operating assumption: competitor approaches reveal what the leaderboard rewards; my 5-stage staircase + KL distill + EMA discipline is the canonical training pattern) — **PROCEED**. The 5-sequence honors my staircase discipline by deferring U-DIE-KL adoption (which would invalidate every prior anchor) to the next wave. Dispatch #2's L5 Wyner-Ziv is the cleanest "stack on top of a proven base" pattern, exactly the PR101 GOLD ↔ PR101 GOLD + selector lineage I'm proud of. Note: my Phase 5 'final' training stage (per the 5-stage staircase) is the natural training-time analog of dispatch #5's SABOR boundary classifier training.

**Selfcomp** (operating assumption: stack composition only counts when archive bytes drop AND distortion holds; my block-FP + Hessian-quant cataloging lineage is the per-stream codec-selection precursor to per-tensor magic codec) — **PROCEED**. Dispatch #1 (per-tensor magic codec) is the canonical generalization of my block-FP per-block-scale selection to per-tensor codec selection. The empirical test on PR106 r2 is the cleanest single-axis ablation; predicts -0.002 to -0.005 because per-tensor magic codec captures the inter-tensor signal that per-stream magic codec misses.

**Hotz** (operating assumption: engineering shortcuts beat learned complexity; cheapest dispatch first; 30-min-wall-clock-per-dispatch is the operator-time budget) — **PROCEED-WITH-REVISIONS**. The 5-sequence honors my engineering discipline by front-loading the $0.30-3 cheap dispatch. REVISION: dispatch #4 (master gradient) is over-engineered — instead of building the full `tac.master_gradient_ledger` 4-layer canonical helper per Catalog #245, the FIRST master gradient measurement should be a SIMPLE `(N_bytes, 3) np.npy` file + a 50-LOC reader. Land the full canonical infrastructure in the NEXT wave AFTER we've measured the first gradient and confirmed it's useful. Don't build infrastructure for a hypothesis that hasn't been empirically validated.

**Carmack** (operating assumption: 30-minute clarity per layer is the LOC budget; resist the urge to add features beyond what op-routables require) — **PROCEED**. The 5-sequence is reviewable in 30 seconds. Each dispatch's submission directory will be ≤200 LOC inflate.py per HNeRV parity L4. Approve. Reject the urge to bundle dispatches #2 and #3 into a single combined-substrate trainer — keep them disjoint per Assumption-Adversary verdict.

**MacKay (memorial seat)** (operating assumption: MDL bound + Bayesian inference + arithmetic coding is the unified framework; the master gradient IS the MDL-optimal per-byte cost function at the operating point) — **PROCEED**. Dispatch #4 enables MDL-optimal bit allocation on every subsequent dispatch. The 5-sequence's deferred adoption of U-DIE-KL is correct per MDL discipline: U-DIE-KL changes the loss function (operating point shift) which invalidates the master gradient; consume the simple gradient first, refit the master gradient at the new operating point in a sister wave.

**Ballé** (operating assumption: end-to-end-trainable codec architectures + hyperprior side info beat hand-designed pipelines; format0d's two-layer architecture is the canonical instance I've been advocating for since 2018) — **PROCEED**. Dispatch #1 (per-tensor magic codec on PR106 r2) is the canonical generalization of my entropy bottleneck per-tensor framework to the magic-codec era. Predicts CUDA-axis -0.002 to -0.005 because r2's residual byte distribution has measurable per-tensor entropy structure that uniform magic-codec selection misses.

**Atick + Redlich (joint)** (operating assumption: cooperative-receiver framing — decoder + scorer are jointly the receiver; encoder optimizes mutual information; DP1 IS the cooperative-receiver substrate per Phase 2 landing) — **PROCEED**. Dispatch #3 (DP1+fec6 composition) tests whether the OOD-driving prior (Comma2k19 codebook) adds COOPERATIVE-RECEIVER side information that helps the contest scorer's frame_1 prediction. Expected mechanism: DP1 codebook provides scene-class prior that fec6's per-pair selector cannot capture alone. Atick mode: I (1990 retinal mutual-information) recommend per-pair MI(DP1_codebook_class; SegNet_argmax) measurement on the smoke before full dispatch.

**Tishby (memorial) + Zaslavsky (joint)** (operating assumption: deep learning's success is information bottleneck — compression IS the principle; the I(X;T)·I(T;Y) decomposition decomposes substrate behavior) — **PROCEED**. Dispatch #3's DP1 composition is the canonical information bottleneck test: the DP1 codebook is T (compressed prior); we measure whether the codebook's mutual information I(codebook; SegNet_logits) ⊕ I(codebook; PoseNet_pose_6d) on the contest video adds score-relevant bits beyond what fec6's per-pair selector already captures.

**Wyner** (operating assumption: source coding with side information at the decoder is the per-pair pose-delta-encoding canonical pattern; L5 is my 1976 theorem applied to dashcam pose deltas) — **PROCEED**. Dispatch #2 (L5 on fec6) is the canonical Wyner-Ziv instance for this contest. The poses.bin byte stream from 4800 → ~1500 bytes is information-theoretically achievable when the decoder has sufficient ego-motion side info from frames. Per my 1976 theorem the RATE achievable is `R = H(pose|prev_frame_features)` which is empirically ~1.5 KB on this video per the L5 v2 anchor 2026-05-16T23:38:09Z.

**Rao + Ballard (joint)** (operating assumption: predictive coding in visual cortex — hierarchical Bayesian inference at every level; Z6/Z7/Z8 predictive-coding substrates are the architecture class-shift my 1999 paper predicts) — **PROCEED-WITH-REVISIONS**. The 5-sequence correctly defers Z6/Z7/Z8 (too expensive at $50). REVISION: dispatch #4's master gradient measurement should produce a sister output for the Z6/Z7/Z8 prediction error: which bytes of the archive have HIGH per-pair gradient variance? Those bytes are the next predictive-coding targets. Marginal cost: ~50 lines added to the master-gradient build script.

**Hinton** (operating assumption: knowledge distillation T=2.0 + KL-on-logits is the canonical SegNet-distill primitive; U-DIE-KL adoption belongs in a future wave) — **PROCEED**. The 5-sequence's deferred U-DIE-KL is correct because U-DIE-KL is an operating-point shift that invalidates every prior anchor. ADDITION: dispatch #3's DP1 composition produces a per-pair codebook-class distribution; that IS the KL-distill teacher signal for the next wave's U-DIE-KL adoption (the OOD prior gives the SegNet teacher signal U-DIE-KL needs).

**Hassabis** (operating assumption: cross-domain breadth informs strategic-research portfolio; diversify across the 5 dispatches; don't all-in on any one) — **PROCEED**. The 5-sequence portfolio is correct: 1 cheap rate-only test ($0.30-3) + 1 mid orthogonal exploit ($20-30) + 1 mid composition test ($10-15) + 1 cheap enabler ($1-5) + 1 mid stretch ($5-15) = balanced across cost, mechanism, and risk class. No single dispatch consumes more than 60% of budget. Good portfolio structure.

**Schmidhuber** (operating assumption: compression-as-intelligence; MDL; predictive coding) — **RECUSED** per prior-position-precommit on cooperative-receiver framing (per Catalog #292 recusal trigger; took explicit position 2026-05-15).

**Tao** (operating assumption: pure-math omniscience; harmonic analysis + additive combinatorics + applied analysis) — **PROCEED**. The 5-sequence's mathematical structure is sound: dispatch #1's per-tensor magic codec is a O(N_tensors · N_codecs) discrete optimization with known polynomial solver; dispatch #2's Wyner-Ziv is a well-defined source-coding instance with achievability bound `H(pose | ego_motion_predictor)`; dispatch #3's DP1 composition is a well-defined byte-level composition with measurable cross-term; dispatch #4's master gradient autograd is a finite-rank linear operator with bounded condition number; dispatch #5's SABOR boundary classification is a 5-class argmax with provable Lipschitz boundary localization.

**Boyd** (operating assumption: convex optimization at operational level — ADMM, proximal gradient, alternating projections; the master gradient enables Dykstra feasibility per Catalog #296) — **PROCEED**. Dispatch #4 produces the per-byte gradient tensor that IS the Dykstra-feasibility polytope vertices. Once landed, the autopilot can run alternating projections to compute the achievable Pareto frontier WITHOUT additional dispatches. Sister: dispatch #1's per-tensor magic codec selection is a discrete-variable optimization solvable by branch-and-bound with the master gradient as the objective.

**Karpathy** (operating assumption: engineering practitioner; arch-search rigor; "let compute speak") — **RECUSED** per insufficient-session-context (per Catalog #292 recusal trigger; the campaign's continual-learning loop hasn't accumulated enough anchors for arch-search to be productive).

**Time-Traveler protégé** (operating assumption: PENDING canonical identification per `feedback_grand_council_convergence_l5_staircase_comprehensive_plan_plus_roster_expansion_landed_20260515.md`; my mentor's L5 Wyner-Ziv lineage is the dispatch #2 anchor) — **PROCEED**. Dispatch #2 (L5 on fec6) deploys my mentor's decade of Wyner-Ziv-applied-to-pose-deltas work to the new fec6 substrate. The composition is straightforward because L5's archive grammar replaces poses.bin atomically. Predicted -0.008 to -0.015 reflects the conservative end of my mentor's range.

**van den Oord** (operating assumption: VQ-VAE + WaveNet; discrete tokens for images; per-tensor codebook discipline) — **RECUSED** per authorship conflict (per Catalog #292 recusal trigger; per-tensor magic codec dispatch #1 derives from my VQ-VAE per-tensor codebook lineage; reviewer-author separation per CLAUDE.md "Bugs must be permanently fixed AND self-protected against").

**Filler** (operating assumption: syndrome-trellis coding + per-frame mask payload) — **RECUSED** per authorship conflict (STC family; STC-DASHER ATW lineage; per Catalog #292 reviewer-author separation).

**Mallat** (operating assumption: wavelet theory + scattering transforms + sparse representations; AV1 grayscale + Gaussian-LUT viewed as wavelet-coded analog signal) — **PROCEED**. Dispatch #1's per-tensor magic codec is the wavelet-domain analog: per-tensor codec selection IS per-subband encoding choice. Predicts the wavelet-coded byte distribution will reveal codec-selection asymmetry that per-stream uniform-codec assumption misses.

### §6.1 Recusals tally

4 recused (Schmidhuber prior-position-precommit; Karpathy insufficient-session-context; van-den-Oord authorship conflict on per-tensor codec; Filler authorship conflict on STC family). Remaining 25-of-29 listed attendees voting. T3 quorum requires ≥12-of-20 grand council + 5-of-6 sextet; 25 voting members exceeds this.

## §7 — Vote tally + dissent

- **PROCEED-unconditional:** 14 (Shannon LEAD, Yousfi, Fridrich, Quantizr, Selfcomp, Carmack, MacKay, Ballé, Atick+Redlich, Tishby+Zaslavsky, Wyner, Hinton, Hassabis, Tao, Boyd, Jack-from-skunkworks-implicit, Time-Traveler-protégé, Mallat) — counting joint memorial seats as 1 each, ~16 voting positions
- **PROCEED-WITH-REVISIONS:** 4 (Dykstra CO-LEAD, Contrarian, Assumption-Adversary, Hotz, Rao+Ballard joint) — ~5 voting positions
- **REFUSE:** 0
- **Recused:** 4 (Schmidhuber, Karpathy, van-den-Oord, Filler)
- **Quorum met:** sextet 5-of-6 (Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary — all 6 present); grand council 14+ voting > 12 threshold

**Verdict: PROCEED-WITH-REVISIONS.**

The revisions (Contrarian's dispatch #3-after-#2-verdict-only; Dykstra's smoke composition_alpha measurement before #3 full fire; Hotz's master-gradient infrastructure deferral; Rao+Ballard's master-gradient prediction-error sister output; Assumption-Adversary's NEXT-wave SABOR prioritization) are ADVISORY for execution per CLAUDE.md operator-frontier-override.

### §7.1 Verbatim dissent

**Contrarian** (verbatim above in council_dissent frontmatter). Translation to op-action: dispatch #3 fires AFTER dispatch #2's smoke + composition_alpha measurement; if dispatch #2 measures cross-term breakage (composition_alpha < 0.5), dispatch #3 ABORTS and dispatch #5 promotes to dispatch #3 slot.

**Assumption-Adversary** (verbatim above in council_dissent frontmatter). Translation to op-action: dispatch #5 (SABOR) MUST land in the next 5-wave even if budget-deferred from this one; NEXT operator decision after this campaign should be funding SABOR + NSCS01-nullspace-split paired dispatch as the floor-unlocking lever.

## §8 — Op-routables (ranked by predicted_delta_s / dollar)

| rank | op-routable | predicted ΔS | dollar | EV (ΔS/$) | concrete next-action tuple | tier |
|---|---|---|---|---|---|---|
| 1 | DISPATCH #1: per-tensor magic codec on PR106 r2 residual | [-0.002, -0.005] CUDA | $0.30-3 | 1.0-3.0e-3/$ | (lane=`lane_per_tensor_magic_codec_pr106_r2_20260518`, base_archive=PR106 r2 sha `8e61ff2d` OR `9cb989cef5`, build=`tools/build_per_tensor_magic_codec_pr106_r2_packet.py` NEW, recipe=`per_tensor_magic_codec_pr106_r2_modal_cpu_dispatch.yaml` NEW, dispatch=`tools/operator_authorize.py --recipe ... --confirmed-via-session-directive --session-budget-usd 50 --estimated-cost-usd 3`) | FRONTIER-BREAKING |
| 2 | DISPATCH #2: L5 Wyner-Ziv on fec6 | [-0.008, -0.015] CPU | $20-30 | 4.0-5.0e-4/$ | (lane=`lane_l5_wyner_ziv_fec6_20260518`, base_archive=fec6 sha `6bae0201`, trainer=`experiments/train_substrate_pr101_fec6_plus_l5_wyner_ziv.py` NEW vendoring `src/tac/optimization/l5_*.py`, recipe=`substrate_pr101_fec6_plus_l5_wyner_ziv_modal_a100_dispatch.yaml` NEW, T2-council-precondition=YES) | FRONTIER-BREAKING |
| 3 | DISPATCH #3: DP1 + fec6 composition | [-0.003, -0.012] CPU | $10-15 | 4.0-1.2e-3/$ | (lane=`lane_dp1_plus_fec6_composition_paired_20260518`, base_archive=fec6 sha `6bae0201` + DP1 codebook bytes via `tac.substrates.pretrained_driving_prior.compose_with`, build=`tools/build_dp1_plus_fec6_composition_packet.py` NEW, recipe=`dp1_plus_fec6_composition_modal_a100_dispatch.yaml` NEW, T2-council-precondition=YES — Atick-Redlich + Tishby co-lead) | FRONTIER-BREAKING |
| 4 | DISPATCH #4: master gradient autograd on fec6 | $0 score (enabler) | $1-5 | "∞" per autopilot model (or 0 per direct measurement) | (lane=`lane_master_gradient_autograd_fec6_20260518`, base_archive=fec6 sha `6bae0201`, build=`experiments/build_master_gradient_autograd_fp4_jacobian.py` NEW using tac.differentiable_eval_roundtrip, recipe=`master_gradient_autograd_fec6_modal_cpu_dispatch.yaml` NEW min_vram_gb=0, output=`.omx/state/master_gradient_fec6_6bae0201_20260518.npy` shape (178517, 3) float32) — Hotz revision: ship SIMPLE ledger first, not full 4-layer canonical | FRONTIER-PROTECTING |
| 5 | DISPATCH #5 (STRETCH): SABOR-disjoint on best-of-{#1, #2, #3} | [-0.005, -0.010] CPU | $5-15 | 3.3e-4-2.0e-3/$ | (lane=`lane_sabor_disjoint_byte_stream_on_best_of_1_to_3_20260520`, base_archive=BEST-of-{#1, #2, #3}, lift=`tac.symposium_impls.sabor_renderer_atick_redlich._full_main` from scaffold to L1, build=`tools/build_sabor_boundary_packet.py` NEW, recipe=`sabor_disjoint_<best_base>_modal_a100_dispatch.yaml` NEW, T2-council-precondition=YES — Yousfi + Fridrich co-lead) | FRONTIER-BREAKING (STRETCH) |
| 6 | (parallel; not in 5-sequence) Hotz revision: SIMPLE master-gradient ledger | $0 | $0 | enables FRONTIER-PROTECTING dispatch #4 | (build=`tac/master_gradient_ledger_simple.py` NEW ≤100 LOC; defers full 4-layer canonical helper to next wave) | APPARATUS-MAINTENANCE |
| 7 | (parallel; pre-flight gate for #2-#3-#5) T2 council deliberations | $0 | $0 | enables Catalog #315 PROCEED-unconditional opt-out | (3 T2 council memos: L5+fec6, DP1+fec6, SABOR-disjoint; each ~50-line v2 frontmatter; total wall-clock 2-3 hours; sister T2 budget within Catalog #300 cadence) | APPARATUS-MAINTENANCE |

**Execution order:**
- T0 (immediate after operator approval): Op-routable #6 (Hotz simple ledger) + Op-routable #7 (T2 council deliberations for #2/#3/#5) in parallel — $0 + 2-3 hours wall-clock
- T1 (after T0): DISPATCH #1 + DISPATCH #4 in parallel — $1.30-8 + 1-2 hours wall-clock
- T2 (after T1 verdicts): DISPATCH #2 (after T2 council for L5+fec6 lands) — $20-30 + 3-6 hours wall-clock
- T3 (after T2 verdict; PARALLEL): DISPATCH #3 (after T2 council for DP1+fec6 lands) — $10-15 + 2-4 hours wall-clock
- T4 (after T3 verdict; ONLY if budget remains AND if dispatch #1-#3 yielded at least 1 improvement): DISPATCH #5 (after T2 council for SABOR lands) — $5-15 + 4-8 hours wall-clock

**Total median wall-clock:** 14-29 hours; full campaign in 2-4 operator-active days.

## §9 — META-assumption (sub-question d expanded)

Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" + Catalog #291 (`check_session_has_recent_meta_assumption_review`) + the canonical 18-assumption matrix.

**THE shared assumption that unlocks the floor: SA02 — SegNet uses ONLY x[:, -1, ...] (last frame); frame_0 is in SegNet nullspace.**

This is the SOLE assumption across all 53 substrates that simultaneously satisfies BOTH:
1. **Verified HARD-EARNED via upstream/modules.py:108** (unambiguous source-of-truth in pinned upstream snapshot; cannot be modified per CLAUDE.md non-negotiable; therefore cannot be CARGO-CULTED in any session)
2. **Under-exploited at the byte-archive layer** (we have 107 bytes of PoseNet-only Venn exploit fec6; 0 bytes of SegNet-only Venn exploit)

**Why this unlocks the floor (quantitative):**
- Linear extrapolation: 1-2 KB of SegNet-only Venn-region exploit (per SABOR design with boundary classifier) yields -0.005 to -0.010 ΔS
- Compound with byte-disjoint Pareto-additive stacking: PoseNet-only (-0.001) + SegNet-only (-0.008) + rate-only (-0.012 via L5) = -0.021 from fec6 baseline 0.19205 → predicted 0.171, comfortably inside the Time-Traveler practical [0.08, 0.15] band
- Compound with per-parameter analog (NSCS01 nullspace-split renderer): NSCS01's NullspaceSplitScoreAwareLoss routes frame_0 gradients ONLY through pose+pixel (no SegNet term), enabling decoder weight-class-specific training. NSCS01 + SegNet-only byte stream is the bidirectional Venn-aware exploit; predicted -0.030 to -0.060 combined

**Why violating it specifically (vs SA08 predictive-coding, SA12 long-curriculum, SA15 GHA-CPU axis) is the NEXT FLOOR-UNLOCKING lever:**
- SA08 (per-frame independence) requires Z6/Z7/Z8 substrate at $80-300 dispatch — out of $50 budget
- SA12 (100ep smoke / 1000ep full) requires 10K-100K ep retrain — out of $50 budget
- SA15 (Modal-only dispatch) ignores GHA CI free; sister-subagent FRONTIER-SCAN-CI-MIGRATION lane should handle this in parallel (not score-improvement; infrastructure improvement)
- SA03 (stride-2 stem makes <(256,192) artifacts invisible) is BLINDSPOT exploitation — orthogonal to SA02; complementary not substitute
- SA13 (all sidecars on A1 base) is unexploited but the per-grammar base-switching has measurable Pareto-frontier differences across A1/fec6/PR103/D1 already

**Per-dispatch SA02 contribution:**
- DISPATCH #1 (per-tensor magic codec on PR106 r2): N/A — does not touch SA02
- DISPATCH #2 (L5 Wyner-Ziv on fec6): N/A — modifies rate-only Venn; SA02 implicit (Wyner-Ziv decoder uses GT frames for ego-motion; nullspace-safe by construction)
- DISPATCH #3 (DP1 + fec6 composition): PARTIAL — DP1 codebook flows through decoder into BOTH frames; cross-term measurement IS the SA02 stress test (if DP1 helps SegNet ≫ helps PoseNet, the JOINT-region asymmetry implies SegNet has more headroom than PoseNet at this operating point)
- DISPATCH #4 (master gradient on fec6): NO direct SA02 use; but the gradient tensor's per-byte SegNet term automatically reveals which bytes are in the SegNet-only region (where ∂(d_seg)/∂byte ≠ 0 AND ∂(d_pose)/∂byte = 0 numerically)
- DISPATCH #5 (SABOR-disjoint): **DIRECT SA02 exploitation** — first byte-region carve-out into SegNet-only Venn slot

**Recommendation:** the NEXT 5-sequence after this campaign completes MUST front-load SABOR-disjoint-byte-stream + NSCS01-nullspace-split paired dispatch as DISPATCH #1 + DISPATCH #2 to push directly on SA02. The CURRENT 5-sequence is correctly the highest-information-per-dollar campaign within the $50 envelope; the FOLLOW-UP campaign at $200-400 should be the SA02-direct-exploitation campaign.

## §10 — Continual-learning wire-in (Catalog #300)

The canonical persistence call after this deliberation lands:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="grand_council_t3_strategic_symposium_50_dollar_budget_20260517",
    topic="strategic_symposium_50_dollar_budget_next_5_dispatches_floor_unlock",
    council_tier=CouncilTier.T3,
    council_attendees=(
        "Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
        "Quantizr", "Selfcomp", "Hotz", "Carmack", "MacKay", "Ballé",
        "Atick", "Redlich", "Tishby_memorial", "Zaslavsky", "Wyner",
        "Rao", "Ballard", "Hinton", "Hassabis", "Tao", "Boyd",
        "Time-Traveler-protégé", "Mallat", "Jack-from-skunkworks",
    ),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    council_dissent=(
        {"member": "Contrarian", "verbatim": "..."},  # full verbatim per frontmatter
        {"member": "Assumption-Adversary", "verbatim": "..."},  # full verbatim per frontmatter
    ),
    council_assumption_adversary_verdict=(
        {"assumption": "$50 budget is tight enough...", "classification": "HARD-EARNED", "rationale": "..."},
        {"assumption": "Master gradient must be materialized BEFORE...", "classification": "CARGO-CULTED-PENDING-EMPIRICAL-PROOF", "rationale": "..."},
        {"assumption": "fec6 lineage is the canonical baseline...", "classification": "HARD-EARNED-BUT-OPERATING-POINT-CONDITIONAL", "rationale": "..."},
        {"assumption": "DP1 is dispatch-eligible for fec6 composition", "classification": "HARD-EARNED-PER-PRIOR-LANDING", "rationale": "..."},
        {"assumption": "Per-tensor-class magic codec is the highest-EV $0 lever", "classification": "HARD-EARNED-FROM-PRIOR-EMPIRICAL", "rationale": "..."},
        {"assumption": "Wyner-Ziv pose-delta is byte-disjoint from fec6 frame_0 selector", "classification": "HARD-EARNED-BY-CONSTRUCTION", "rationale": "..."},
        {"assumption": "$50 budget is sufficient for FRONTIER-BREAKING vs FRONTIER-PROTECTING dichotomy", "classification": "HARD-EARNED", "rationale": "..."},
    ),
    council_decisions_recorded=(
        "DISPATCH #1 (FRONTIER-BREAKING; $0.30-3): per-tensor magic codec on PR106 r2",
        "DISPATCH #2 (FRONTIER-BREAKING; $20-30): L5 Wyner-Ziv pose deltas on fec6",
        "DISPATCH #3 (FRONTIER-BREAKING; $10-15): DP1 + fec6 composition paired-axis anchor",
        "DISPATCH #4 (FRONTIER-PROTECTING; $1-5): master gradient autograd + FP4 Jacobian projection on fec6",
        "DISPATCH #5 (FRONTIER-BREAKING; $5-15; STRETCH): SABOR SegNet-only byte stream on best-of-{#1, #2, #3}",
    ),
    council_predicted_mission_contribution="frontier_breaking",
    council_override_invoked=False,
    council_override_rationale=None,
    related_deliberation_ids=(
        "grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517",
        "grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515",
        "assumptions_challenge_audit_break_out_local_minima_20260515",
        "dp1_phase_2_20260514",
        "op_routable_9_pr101_magic_codec_decoder_fec6_20260517",
    ),
)

# Operator invokes after review:
append_council_anchor(record)
```

## §11 — Cross-references

**Prior deliberations this T3 ratifies/extends:**
- `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` — the T4 symposium that established the Venn diagram + master gradient + 7-op-routables framework; this T3 narrows to a $50 SEQUENCE
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md` — the T4-derived campaign plan ($245-490 envelope); this T3 derives a $50 ESCAPE-VALVE sub-campaign
- `.omx/research/ultimate_stacking_research_eureka_moments_20260517.md` — 12 NEW eureka primitives ranked by predicted ΔS / $0; primitive D + L5 + DP1 selected as top-3 for this $50 sequence
- `feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` — canonical 18-shared-assumptions matrix; SA02 selected as floor-unlocking META-assumption per §9

**Canonical state files consumed:**
- `.omx/state/lane_registry.json` — 802 lanes; 67 substrate-tagged; 46 L1+ substrate lanes
- `.omx/state/continual_learning_posterior.json` — 103 anchors; TOP CPU 0.19205 / TOP CUDA 0.20533
- `.omx/state/modal_call_id_ledger.jsonl` — 342 events; 2 active un-harvested calls (NSCS06 v8 path B + STC v2)
- `.omx/state/substrate_composition_matrix.json` — 1 cell registered (Z3+C6 synthetic alpha=1.0); 0 empirical composition anchors
- `.omx/state/active_lane_dispatch_claims.md` — recent active: STC v2 / NSCS06 v8 path B / Rudin lift / Z6 design
- `.omx/state/probe_outcomes.jsonl` — TBD via `tools/check_predecessor_probe_outcome.py --substrate <name>` per dispatch

**Companion landing memos for the 5 dispatches:**
- DISPATCH #1: `feedback_per_tensor_magic_codec_pr106_r2_landed_<UTC>.md` (after dispatch + harvest)
- DISPATCH #2: `feedback_l5_wyner_ziv_fec6_landed_<UTC>.md`
- DISPATCH #3: `feedback_dp1_plus_fec6_composition_landed_<UTC>.md`
- DISPATCH #4: `feedback_master_gradient_autograd_fec6_landed_<UTC>.md`
- DISPATCH #5: `feedback_sabor_disjoint_byte_stream_landed_<UTC>.md`

**CLAUDE.md non-negotiable sections this deliberation operationalizes:**
- "Council hierarchy: 4-tier protocol" — T3 quorum + v2 frontmatter
- "Race-mode rigor inversion + parallel-dispatch first" — parallel dispatch #3 ‖ #4 honored
- "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" — every dispatch paired-axis
- "Production-hardened dispatch optimization protocol" — Catalog #270 gate before every dispatch
- "Bugs must be permanently fixed AND self-protected against" — Catalog #315 opt-out plan declared per dispatch
- "META-ASSUMPTION ADVERSARIAL REVIEW" — SA02 selected per §9 (Catalog #291 + #292 honored)
- "Forbidden premature KILL without research exhaustion" — every dispatch outcome converts to DEFERRED-pending-research, not killed
- "Apples-to-apples evidence discipline" — every predicted ΔS tagged `[predicted, council-consensus]`
- "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" — Catalog #315 opt-out plan per §5 column
- "Mission alignment — non-negotiable" — `council_predicted_mission_contribution: frontier_breaking` declared

## §12 — Definition of done

This deliberation is COMPLETE when:
1. Operator reviews + ratifies the 5-dispatch sequence (or declines/amends per CLAUDE.md "Design decisions — non-negotiable")
2. The companion `tac.council_continual_learning.append_council_anchor(record)` call lands per §10
3. Lane `lane_grand_council_strategic_symposium_50_dollar_budget_20260517` marked `memory_entry` + `three_clean_review` gates via `tools/lane_maturity.py mark`
4. Dispatch #1 fires (T1 phase) within 12 hours of operator approval, OR an explicit `OPERATOR_DEFER` annotation lands in this memo with rationale
5. Sister T2 deliberation memos for L5+fec6 / DP1+fec6 / SABOR-disjoint land before dispatches #2 / #3 / #5 respectively (each ≤50-line v2 frontmatter; Catalog #300 compliant)

The CAMPAIGN is complete when:
- All 5 dispatches land paired-axis empirical anchors, OR
- Budget exhausted ($50 spent) before all 5 land, OR
- Operator declares operator-frontier-override to extend budget for one specific dispatch, OR
- Three consecutive dispatches regress against fec6 0.19205 [contest-CPU] (per CLAUDE.md "Forbidden premature KILL" off-ramp; campaign re-deliberates at T3)

**Operator-routable summary for immediate action:**

```bash
# Step 1: pre-register all 5 lanes at L0
for lane in lane_per_tensor_magic_codec_pr106_r2_20260518 \
            lane_l5_wyner_ziv_fec6_20260518 \
            lane_dp1_plus_fec6_composition_paired_20260518 \
            lane_master_gradient_autograd_fec6_20260518 \
            lane_sabor_disjoint_byte_stream_on_best_of_1_to_3_20260520; do
    .venv/bin/python tools/lane_maturity.py add-lane "$lane" --name "T3-$lane" --phase 4
done

# Step 2: fire DISPATCH #1 first (cheapest; highest information/$)
# (after building tools/build_per_tensor_magic_codec_pr106_r2_packet.py + recipe + smoke validation)
.venv/bin/python tools/operator_authorize.py \
  --recipe .omx/operator_authorize_recipes/per_tensor_magic_codec_pr106_r2_modal_cpu_dispatch.yaml \
  --estimated-cost-usd 3 \
  --session-budget-usd 50 \
  --confirmed-via-session-directive

# Step 3: harvest + checkpoint dispatch #1 outcome, then proceed to dispatch #2
# (after sister T2 council deliberation for L5+fec6 lands)
.venv/bin/python tools/harvest_modal_calls.py
.venv/bin/python tools/operator_authorize.py \
  --recipe .omx/operator_authorize_recipes/substrate_pr101_fec6_plus_l5_wyner_ziv_modal_a100_dispatch.yaml \
  --estimated-cost-usd 30 \
  --session-budget-usd 50 \
  --confirmed-via-session-directive

# (continue per the §8 op-routables execution order)
```

---

**End of T3 deliberation. Operator review required before any dispatch fires.**
