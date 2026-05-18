---
schema: council_deliberation_v2
deliberation_id: dp1_pr101_composition_design_memo_20260518
topic: "DP1 (pretrained driving prior) × PR101_lc_v2 (gold submission substrate) composition — canonical substrate-engineering design memo"
review_kind: substrate_design_memo
review_date: "2026-05-18"
lane_id: lane_dp1_pr101_composition_design_20260518
subagent_id: dp1_pr101_composition_design_20260518
parent_mandate_id: council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518
parent_mandate_quote: "Z8 symposium identified DP1+PR101 as cross-stack synergy #2 (DP1 OOD pretrain + PR101 in-distribution refinement is the DreamerV3 pattern); predicted ΔS [-0.012, -0.004] per Z8 symposium cross-stack-synergy matrix. Operator directive: 'all candidates including c6 ibps may need further optimization and iteration and review and audit and individual extreme passion and detail and effort and adversarial grand council symposiums'."
operator_directives:
  - "all candidates including c6 ibps may need further optimization and iteration and review and audit and individual extreme passion and detail and effort and adversarial grand council symposiums"
  - "need to ensure it's useful... will be using that over and over"
  - "share what works but when it is stale or obsolete or suppressing signal or otherwise and when the optimal engineering calls for it we want full and complete and correct unique and distinct designs and implementations"
  - "what would it take to break out of the local minimum"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
horizon_class: frontier_pursuit
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Hafner
  - Tishby_memorial
  - Atick_memorial
  - Carmack
  - MacKay_memorial
  - Selfcomp
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "I VETO any framing that lets the DP1+PR101 composition ship without an EMPIRICAL OOD-similarity probe between Comma2k19 (DP1's distillation source) and the contest video FIRST. The whole composition theory rests on 'DP1's prior generalizes to the contest video' — but the contest video is a SPECIFIC dashcam clip, not a generic dashcam distribution. If the contest video's per-frame statistics (mean luminance, road-plane PCA spectrum, sky-horizon profile, vehicle density) lie OUTSIDE Comma2k19's 95% confidence ellipse, then DP1's prior is at best neutral and at worst HARMFUL (pulling the renderer toward off-distribution dashcam manifold). The first $0 op-routable MUST be the OOD-similarity probe; only then does the $3-5 Stage 2 smoke fire."
  - member: Assumption-Adversary
    verbatim: "The shared assumption I am operating within for this deliberation is that 'DP1+PR101 composition is α-additive (composition_alpha ≈ 1.0)' per Catalog #322 framework. This is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION. Empirical reality from sister substrate composition matrix (per `.omx/state/substrate_composition_matrix.json`) shows that 4-of-8 well-anchored substrate pairs are α-sub-additive (composition_alpha < 1.0) when they share a dominant axis (here: both DP1 and PR101 produce RGB-axis renderer outputs). The pose-axis is shared; the seg-axis is shared; the rate-axis is shared. By the cargo-cult-unwind methodology per Catalog #303 + the NSCS06 v6→v7 empirical lesson (44% improvement in ONE iteration by unwinding 4-of-7 cargo-cults), the α-additivity assumption MUST be tested via paired-comparison smoke on the fec6 anchor BEFORE the Stage 2 dispatch fires. Predicted: composition_alpha ∈ [0.6, 0.95] (sub-additive); the predicted ΔS band [-0.012, -0.004] is therefore the BEST CASE; realistic is [-0.008, -0.002]."
  - member: Hafner
    verbatim: "DreamerV3 (Hafner 2023) lens on DP1+PR101 composition: this IS the canonical world-model + actor-critic pattern. DP1 ≡ the world-model prior (frozen latent dynamics over the dashcam state space). PR101 HNeRV ≡ the actor (predicts per-frame RGB conditioned on the latent state). The composition is mathematically MOTIVATED — DreamerV3's empirical: a pretrained world model gives 5-10× sample efficiency on downstream RL tasks. Translated to pact: DP1 as pretrained codebook gives PR101 5-10× faster convergence to the same score floor. This is HARD-EARNED at the architectural pattern level. BUT — the contest is NOT an RL task; it's a single-video overfit. The DreamerV3 sample-efficiency argument is WEAKER in the overfit regime where you have unlimited gradient passes on the 600 contest pairs. The composition gain is in OOD-init quality (better starting point in weight space) NOT in sample efficiency. Predicted impact: 0.5-2× faster Stage 2 convergence; final score floor unchanged from PR101-alone."
  - member: Tishby_memorial
    verbatim: "Information Bottleneck (Tishby-Zaslavsky 2015 + Tishby 1999) lens on DP1+PR101 composition: this is the canonical conditional-entropy decomposition H(X) = H(X|DP1) + I(X;DP1). DP1 provides the side-information at the decoder; PR101 encodes H(X|DP1) — the conditional entropy of the contest video GIVEN the dashcam prior. The Wyner-Ziv 1976 theorem applies directly: side-info at decoder reduces rate from H(X) to H(X|DP1). For the contest video, IF Comma2k19 captures ~30-50% of the mutual information with the contest video (typical for OOD pretrain in vision per ImageNet→CIFAR transfer rates), then H(X|DP1) ≈ 0.5-0.7 × H(X) — i.e., the conditional entropy is 30-50% LESS than unconditional. Translated to rate-distortion: at fixed distortion, the conditional code is 30-50% smaller. PR101 GOLD archive is ~178 KB; the conditional code could be ~90-120 KB after DP1 prior consumption. Predicted ΔS at frontier (where rate term = 25 × bytes / 37545489 ≈ 0.119 for full archive): 25 × (178000 - 100000) / 37545489 ≈ 0.052 rate-only savings IF the conditional encoding is faithfully realized. Sister-verification: this requires the inflate path to consume DP1 bytes (Catalog #220 operational mechanism); a phantom inclusion produces ZERO rate savings."
  - member: Atick_memorial
    verbatim: "Atick-Redlich 1990 cooperative-receiver lens on DP1+PR101 composition: this IS the canonical shared-prior-at-decoder pattern. DP1 is the cooperative-receiver shared prior — the receiver knows the dashcam-statistics distribution beforehand; the encoder transmits only the contest-specific information not captured by the prior. The mutual information I(X; X|prior) bounds the achievable rate savings: per cooperative-receiver theory, savings ≈ I(X; prior) / H(X). For DP1's 162KB codebook vs PR101's 178KB archive: if DP1 carries ~20-40% of the mutual information with the contest video, the achievable rate reduction is ~20-40%. CRITICAL: DP1's codebook must be charged in the contest archive bytes (per CLAUDE.md 'Bit-level deconstruction and entropy discipline' + Catalog #166 archive-grammar L3 monolithic single-file). Net rate change: +162KB (DP1 codebook) - X KB (PR101 residual reduction). The composition is rate-POSITIVE only if X > 162KB (i.e., the residual shrinks by more than the codebook adds). This is the CRITICAL feasibility check per Dykstra alternating-projections."
  - member: Carmack
    verbatim: "30-second reviewability + reality-check: DP1+PR101 composition is structurally EXACTLY the right pattern (DreamerV3-style world-model + actor; PR101 GOLD model + DP1 codebook), but the IMPLEMENTATION cost is non-trivial. DP1's existing composition.py + prior_application.py wire DP1 INTO ARBITRARY substrate via `compose_with` API — but the ARCHITECTURE matching is non-trivial. DP1's codebook is shaped for road-plane PCA + sky-horizon profile + vehicle appearance; PR101's HNeRV decoder shape is (H, W, 3) frames produced by a depth=8 ConvNet with specific FiLM conditioning. The codebook bytes have to be REINTERPRETED for PR101's weight initialization — there is no native shape match. Two implementation paths: (A) DP1-as-weight-init: project DP1 codebook bytes onto PR101's decoder weight space via least-squares; (B) DP1-as-side-channel-at-inflate: DP1 codebook lives alongside PR101 archive; inflate path consumes both. Path A is ~150 LOC + ~3-5 LOC weight initialization helper; Path B is ~400 LOC (DP1 archive + PR101 archive + inflate-time merger). Per Carmack canonical wisdom: SHIP PATH A FIRST ($0 + $3-5 smoke); Path B is Phase 2."
  - member: MacKay_memorial
    verbatim: "Bayesian-MDL lens (MacKay 2003 canonical) on DP1+PR101 composition: this is the canonical hierarchical prior pattern. DP1 IS the hyperprior on PR101's renderer weights — encoding the dashcam-statistics manifold as a soft prior over weight space. The posterior decomposition: P(weights | contest_video) ∝ P(contest_video | weights) × P(weights | DP1) × P(DP1). The MAP estimate is the maximum of this product; the canonical implementation is L2-regularization toward DP1's projected weights via DashcamPriorLoss (the existing `prior_application.py` module). The composition IS the existing DP1 substrate but with PR101 as the renderer substitute. Estimated weight-space prior strength: lambda_prior=0.05 (existing DP1 default) is a reasonable starting point. Predicted: 5-10% faster convergence + slightly better final score (the prior preserves OOD-correct details the contest-video gradient cannot reach in finite epochs)."
  - member: Selfcomp
    verbatim: "Selfcomp lens on DP1+PR101 composition: this is mathematically CORRECT but the rate-distortion calculation needs to be sharper. PR101 GOLD's 0.193 [contest-CUDA] uses ~178KB archive; the score decomposition at PR101 GOLD operating point is approximately seg=0.067, pose=0.018, rate=0.119 (25 × 178 KB / 37545489 ≈ 0.119). The dominant cost is RATE (62% of total score). DP1+PR101 composition's path to score improvement is RATE REDUCTION via conditional encoding (Wyner-Ziv side-info at decoder). Per Tishby's calculation: if DP1 carries 30-50% of mutual information with contest, conditional encoding reduces PR101's 178KB → ~90-120KB, saving 25 × 60KB / 37545489 ≈ 0.040 rate-only. BUT — DP1's codebook bytes (162KB) MUST be charged IF they live in the archive. Net rate change: +162KB - 60KB = +102KB worse than PR101-alone. The composition is RATE-NEGATIVE unless DP1's codebook is consumed by inflate WITHOUT being shipped (which requires the operator-provided pre-distributed prior model — NOT contest-compliant per CLAUDE.md 'strict scorer rule' which prohibits decoder-side state outside the archive). The composition is FEASIBLE only if (a) DP1 codebook fits in ≤30-50KB (NOT 162KB), OR (b) DP1 acts ONLY as Stage-1 weight initialization with NO inflate-time consumption (the 'best of both worlds' Carmack Path A)."
council_assumption_adversary_verdict:
  - assumption: "DP1's pretrained dashcam prior generalizes to the contest video"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "DP1 is distilled from Comma2k19 (a SPECIFIC dashcam dataset); the contest video is a SPECIFIC dashcam clip from a different camera mount, possibly different lighting / road type / vehicle density. The OOD similarity requires empirical probe BEFORE composition smoke. Sister-anchor: ImageNet→CIFAR transfer learning shows 5-30% accuracy drop for OOD distribution shift; pact composition could analogously see degraded prior usefulness."
  - assumption: "DP1+PR101 composition is α-additive (composition_alpha ≈ 1.0)"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "Per Catalog #322 framework + 4-of-8 anti-additive evidence at substrate level. Both DP1 and PR101 target the same RGB-axis renderer output; shared dominant axis is the canonical pattern for sub-additivity. Realistic prediction: composition_alpha ∈ [0.6, 0.95]; predicted ΔS [-0.012, -0.004] is BEST CASE."
  - assumption: "DP1 codebook IS contest-compliant if shipped IN the archive"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Non-Negotiable Upstream Rule' + HNeRV parity L9 runtime closure + Catalog #146 contest-compliant inflate template + Catalog #220 substrate L1+ scaffold operational mechanism. The archive is self-contained; codebook bytes ship in the archive; inflate.py consumes them at runtime. Sister-verification via Catalog #272 distinguishing-feature integration contract."
  - assumption: "PR101 HNeRV decoder architecture is compatible with DP1 codebook weight initialization"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "DP1's codebook is shaped (road-plane PCA basis + sky-horizon profile + vehicle appearance) — these are STATISTICAL features. PR101 HNeRV's decoder weights are (H, W, 3) frame-producing ConvNet weights with FiLM conditioning. Shape compatibility is non-trivial; requires least-squares projection per Carmack Path A. The projection's residual norm is the empirical signal: if projection_residual < 0.3 (i.e., DP1 captures >70% of weight-space variance), the composition is HARD-EARNED-FEASIBLE; if projection_residual > 0.7, the composition is CARGO-CULTED and degenerates to PR101-with-random-init."
  - assumption: "Stage 1 init + Stage 2 PR101 HNeRV refinement is the optimal training curriculum"
    classification: HARD-EARNED
    rationale: "Per DreamerV3 + canonical fine-tuning literature (BERT, GPT, ImageNet→COCO transfer). The pretrain→fine-tune pattern IS canonical. The QUESTION is the relative epoch budget split (e.g., DP1 init: 0 epochs of additional pretrain vs PR101 refinement: 200 epochs full PR95 curriculum). The canonical split is 'PR101 full curriculum with DP1 warm-start' — i.e., no additional pretrain; the existing DP1 codebook IS the warm-start; PR101's PR95 8-stage curriculum is the refinement."
  - assumption: "The fec6 archive (`f174192aeadf...`) is a representative anchor for DP1+PR101 composition predictions"
    classification: HARD-EARNED-WITH-REVISION
    rationale: "The fec6 archive IS the canonical PR101_lc_v2 anchor with master-gradient extracted (per `.omx/state/master_gradient_anchors.jsonl`). It's the right anchor for PR101 baseline measurement. REVISION: DP1+PR101 composition produces a DIFFERENT archive (new sha256); the fec6 anchor only informs the PR101 BASELINE not the composed substrate's anchor. Need separate anchor extraction post-Stage-2 for the composed archive."
council_decisions_recorded:
  - "op-routable #1: OOD-similarity probe ($0 local CPU) — compute per-frame statistics (mean luminance, road-plane PCA spectrum, sky-horizon profile, vehicle density) on Comma2k19 sample (8 chunks) vs contest video (600 frames); report 95% confidence ellipse overlap"
  - "op-routable #2: Architecture compatibility probe ($0 local CPU) — least-squares project DP1 codebook bytes onto PR101 HNeRV decoder weight space; report projection residual norm"
  - "op-routable #3: Carmack Path A implementation ($0 implementation; $3-5 Stage 2 smoke) — DP1-as-weight-init scaffold; emit composed archive; paired-CPU smoke"
  - "op-routable #4: Paired-α probe per Catalog #322 ($3 Modal T4) — empirical composition_alpha vs DP1-alone and PR101-alone anchors on contest video"
  - "op-routable #5: Phase 2 Carmack Path B research-only design ($0 design memo) — DP1-as-side-channel-at-inflate (deferred pending Path A results)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: ""
deferred_substrate_id: ""
related_deliberation_ids:
  - council_z8_hierarchical_predictive_coding_per_substrate_symposium_20260518
  - grand_council_t3_pose_axis_non_hnerv_paths_to_frontier_breaking_symposium_20260518
  - phase_1_fisher_precondition_canonical_helper_design_memo_20260518
  - riemannian_newton_substrate_engineering_design_memo_20260518
  - tropical_d_seg_solver_design_memo_20260518
  - n_set_venn_classification_design_memo_pair_region_class_frame_axis_20260518
predicted_score_band_cpu_min: 0.180
predicted_score_band_cpu_max: 0.190
predicted_score_band_validation_status: pending_post_training
predicted_score_band_reactivation_criterion: "post-training paired CPU+CUDA auth eval on composed archive sha after Stage 2 smoke; OOD-similarity probe outcome and architecture-compatibility projection residual gate which sub-band ([0.180, 0.184] best case vs [0.186, 0.190] realistic) applies"
---

# DP1 × PR101_lc_v2 Composition — Canonical Substrate-Engineering Design Memo

**Subagent:** `dp1_pr101_composition_design_20260518`
**Lane:** `lane_dp1_pr101_composition_design_20260518` (L0 SKETCH at landing → L1 SCAFFOLD post-impl)
**Council tier:** T2 (sextet pact + grand-council attendees)
**Verdict:** `PROCEED_WITH_REVISIONS` (6 binding revisions per Council dissent)
**Horizon class:** `frontier_pursuit` (predicted CPU band [0.180, 0.190]; current frontier 0.19205)
**Mission contribution:** `frontier_breaking` (sub-0.190 reachable per Tishby-Wyner-Ziv conditional encoding theorem IF Path A feasibility verified)

---

## 1. Mission alignment per CLAUDE.md

This design memo serves the mission per CLAUDE.md "Mission alignment — non-negotiable" Consequence 4 (frontier-breaking moves DOMINATE rigor budget) AND CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS" (target IS the best contest-faithful public frontier).

**Current frontier (per `tac.frontier_scan.build_frontier_scan_payload`):**
- `[contest-CPU]` best: **0.19205** (lane `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` / archive `6bae0201`)
- `[contest-CUDA]` best: **0.20533** (lane `pr106_format0d_latent_score_table` / archive `9cb989cef519`)

**Predicted DP1+PR101 composition floor (per Tishby IB + Wyner-Ziv 1976):**
- Best case `[predicted CPU]`: **0.180** (if DP1 carries 50% mutual info with contest + Path A weight-init compatible)
- Realistic `[predicted CPU]`: **0.186** (if composition_alpha ≈ 0.7 AND OOD-similarity 30%)
- Worst case `[predicted CPU]`: **0.190** (if DP1 carries only 20% mutual info AND projection residual > 0.5)

**ΔS at frontier:** [-0.012, -0.004] per Z8 symposium cross-stack synergy #2.

Per CLAUDE.md "Mission alignment" Operational Consequence 4: frontier-breaking moves DOMINATE rigor budget. The DP1+PR101 composition is a candidate frontier-breaking path; rigor compresses; council deliberation foregrounds time-critical decision (this memo); operator-frontier-override available if needed. The 6 binding revisions per Council dissent are structured to MINIMIZE rigor cost (4-of-6 are $0 probes BEFORE the $3-5 Stage 2 smoke fires).

**Per CLAUDE.md "Race-mode rigor inversion":** the public leaderboard has NOT moved in the last 24 hours (PR101 GOLD 0.193 last lock-in; rem2 silver / EthanYangTW bronze unchanged). The race-mode prior is OFF; full pre-validation discipline applies.

---

## 2. Frontier evidence per Catalog #316

Per Catalog #316 `check_reports_latest_md_not_stale_vs_canonical_frontier`, the canonical frontier scan reports the following anchors:

### 2.1 CPU axis (`[contest-CPU]` Linux x86_64)

| Rank | Score | Archive sha | Lane id | Source |
|---:|---:|:---|:---|:---|
| 1 | **0.19205** | `6bae0201...` | `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` | continual_learning_posterior 2026-05-15 |
| 2 | 0.193 | PR101 GOLD | public claim | external (not paired-CPU verified) |
| 3 | 0.195 | PR102 silver | public claim | external |
| 4 | 0.196 | PR103 bronze | public claim | external |

### 2.2 CUDA axis (`[contest-CUDA]` T4 / A100)

| Rank | Score | Archive sha | Lane id | Source |
|---:|---:|:---|:---|:---|
| 1 | **0.20533** | `9cb989cef519...` | `pr106_format0d_latent_score_table` | continual_learning_posterior 2026-05-15 |
| 2 | 0.22839 | PR102 public | public claim | external |
| 3 | 0.22936 | PR107 (ours) | apogee | committed |
| 4 | 0.229 | PR103 public | public claim | external |

### 2.3 fec6 anchor (canonical for master-gradient consumers)

**Archive:** `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`
**Origin:** lane `lane_pr101_lc_v2_clone_enhanced_curriculum_fec6_20260515`
**Master-gradient:** `.omx/state/master_gradient_fec6_contest_cpu_scorer_macos_host_advisory_20260517.npy` (178,417 bytes; `aggregate_per_byte_v1` schema; `[macOS-CPU advisory]` axis per Catalog #327 axis-custody fix)
**Operating point:** `d_seg=0.0009`, `d_pose=0.00173294`, `rate=0.004752`, `score=0.3386`

The fec6 anchor IS the canonical PR101_lc_v2 baseline for predicting DP1+PR101 composition effects. The composed substrate produces a NEW archive sha (Catalog #229 premise verification: the composition emits a DIFFERENT archive than fec6); the fec6 anchor informs the PR101-side baseline NOT the composed substrate.

---

## 3. Domain primer

### 3.1 DP1 (pretrained driving prior) architecture

Per CLAUDE.md Catalog #209-#213 family + `src/tac/substrates/pretrained_driving_prior/`:

- **Canonical source:** `src/tac/substrates/pretrained_driving_prior/` (commit `bcdb4f1a` Phase 2 landing 2026-05-14)
- **Codebook size:** 5-10 KB target; ACTUAL size depends on distillation rank + variance retention
- **Dataset provenance:** Comma2k19 (MIT license per `tac.substrates.pretrained_driving_prior.local_chunk_cache.DEFAULT_CHUNK_MANIFEST`)
- **Distillation contract:** `tac.substrates.pretrained_driving_prior.distillation.distill_codebook(frame_iterator, n_components, ...)` produces a frozen codebook with SHA-256 + license_tags + dataset_provenance per Catalog #210
- **Codebook structure:** road-plane PCA basis (top-K principal components of road texture) + sky-horizon profile (per-row mean luminance + chroma profile) + vehicle appearance (codebook of K=64 vehicle patches)
- **Inflate runtime:** `src/tac/substrates/pretrained_driving_prior/inflate.py` (~4.6 KB; consumes 162-KB decoder + 15-KB latent blobs per the L1 scaffold)
- **Score-aware loss:** `src/tac/substrates/pretrained_driving_prior/score_aware_loss.py` routes through canonical `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164
- **Composition API:** `tac.substrates.pretrained_driving_prior.composition.compose_with(dp1_bytes, base_bytes, *, base_substrate)` per Catalog #211 (DPCOMP wrapper; 13-byte magic header + length-prefix + 4-byte base_tag)
- **Lane state:** L1 SCAFFOLD per `.omx/state/lane_registry.json::lane_pretrained_driving_prior_phase_2_20260514`; `impl_complete=true`, `strict_preflight=true` (Catalog #209), `three_clean_review=true` (council memo), `deploy_runbook=true`; `real_archive_empirical=false`, `contest_cuda=false`, `contest_cpu=false` (smoke only at landing)

### 3.2 PR101_lc_v2 (GOLD apples-to-apples clone) architecture

Per CLAUDE.md Catalog #109 + `src/tac/substrates/pr101_lc_v2_clone/`:

- **Canonical source:** `src/tac/substrates/pr101_lc_v2_clone/` (commit per `lane_substrate_pr101_lc_v2_clone_20260512`)
- **Architecture:** HNeRV-family forensic clone of PR101 GOLD's `hnerv_ft_microcodec` (0.193 [contest-CUDA] public claim)
- **State_dict shape:** 28-tensor (matches PR101 GOLD anchor per Subagent C 3-primitives composition: storage_order + conv4_perms + byte_maps)
- **Archive grammar:** `monolithic_single_file_0_bin_fixed_offsets` per HNeRV parity L3 (Catalog #124 8-field declaration: `archive_grammar=monolithic_single_file_0_bin_fixed_offsets parser_section_manifest=parse_archive(bytes)->(decoder_sd,latents,meta) inflate_runtime_loc_budget=<=200_LOC_waiver_for_PR101_sidecar_fidelity runtime_dep_closure=torch,brotli,numpy export_format=multi_brotli_stream_decoder+lzma_latents+raw_sidecar score_aware_loss=alpha*B/N+beta*d_seg+gamma*sqrt(d_pose) bolt_on_loc_budget=substrate_engineering_tag no_op_detector_planned=Catalog_139_byte_mutation_smoke`)
- **Inflate runtime:** `src/tac/substrates/pr101_lc_v2_clone/inflate.py` (~200 LOC with waiver for PR101 sidecar fidelity)
- **Curriculum:** `src/tac/substrates/pr101_lc_v2_clone/curriculum.py` (PR95 8-stage byte-faithful port) + `curriculum_enhanced.py` (Phase 2 enhanced curriculum)
- **Score-aware loss:** `src/tac/substrates/pr101_lc_v2_clone/score_aware_loss.py` (canonical α·B/N + β·d_seg + γ·√d_pose per HNeRV parity L6 score-domain Lagrangian)
- **Lane state:** L1 SCAFFOLD per `.omx/state/lane_registry.json::lane_substrate_pr101_lc_v2_clone_20260512`; `research_only=true`, `lane_class=substrate_engineering` per HNeRV parity L7

### 3.3 The composition opportunity

The composition flow:

```
Stage 1 (warm-start)
   DP1 distillation (offline)
   └── Comma2k19 chunks → frame_iterator → distill_codebook → frozen 162-KB codebook blob
   DP1 codebook → PR101 HNeRV decoder weight initialization (Carmack Path A)
   └── least-squares project codebook PCA basis onto PR101 ConvNet conv-layer weights

Stage 2 (refinement)
   PR101 PR95 8-stage curriculum on contest video
   ├── Inputs: contest video (600 pairs), DP1-initialized weights, fresh latents
   ├── Score-aware loss with DP1 soft-prior regularization (lambda_prior=0.05)
   └── Output: trained PR101 HNeRV state_dict + per-pair latents + poses

Stage 3 (archive)
   Composed archive grammar = DP1 codebook bytes + PR101 residual bytes
   ├── Path A: DP1 acts as Stage-1 init ONLY (zero DP1 bytes in archive); PR101 archive is byte-identical to PR101-alone
   ├── Path B: DP1 + PR101 archive = `compose_with(dp1_bytes, pr101_bytes)` (DPCOMP wrapper)
   └── inflate.py routing per Catalog #211 decompose API
```

**Per Z8 symposium cross-stack synergy #2:** "DP1 × Z8 = ORTHOGONAL (DP1 is OOD pretrain; Z8 is in-distribution hierarchical predictive coding)". The DP1 × PR101 composition is ANALOGOUS: DP1 is OOD pretrain; PR101 is in-distribution overfit refinement. Hafner's DreamerV3 lens validates: this IS the canonical world-model + actor pattern.

Per CLAUDE.md "HNeRV parity discipline" lesson 7: DP1+PR101 composition is **substrate-engineering** (not bolt-on); the ≤350 LOC limit does NOT apply. PR101 GOLD (the canonical 30-second-reviewable model) was 605 LOC = 268 substrate + 337 bolt-on. DP1+PR101 composition target: ~600-800 LOC substrate (PR101_lc_v2 reuse + ~150-300 LOC DP1-init bridge + composition.py reuse).

---

## 4. Canonical-vs-unique decision per layer (per Catalog #290)

This section is MANDATORY per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290 `check_substrate_design_memo_has_canonical_vs_unique_decision_section`.

For each implementation layer, the falling-rule decision is per CLAUDE.md decision criterion (EMPIRICAL > PRINCIPLED > UNCLEAR > OBVIOUS-FIT):

| Layer | Decision | Rationale |
|:------|:---------|:----------|
| **DP1 codebook loader** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Reuse `tac.substrates.pretrained_driving_prior.codebook.load_codebook` AS-IS. The codebook loader is canonical infrastructure with Catalog #210 provenance + license_tags + basis_sha256. No substrate-specific divergence justified. |
| **PR101 HNeRV decoder** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Reuse `tac.substrates.pr101_lc_v2_clone.architecture.Pr101LcV2CloneSubstrate` AS-IS. The HNeRV architecture is the canonical 0.193 GOLD-equivalent substrate. Architecture is byte-faithful to PR101 GOLD (Catalog #109 forensic clone). |
| **Weight-init bridge (DP1 → PR101)** | `FORK_BECAUSE_PRINCIPLED_MISMATCH` | NEW canonical helper: `tac.substrates.dp1_pr101_composition.weight_init_bridge.project_dp1_codebook_to_pr101_decoder(dp1_codebook, pr101_decoder_state_dict_template)`. The shape mismatch between DP1's (road-plane PCA + sky-horizon + vehicles) and PR101's HNeRV ConvNet weights REQUIRES least-squares projection. No existing canonical helper covers this; the bridge IS substrate-specific. |
| **Training curriculum** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Reuse `tac.substrates.pr101_lc_v2_clone.curriculum_enhanced` (PR95 8-stage byte-faithful port + Phase 2 enhancements) AS-IS. The only divergence is the initial weight state (DP1-projected vs random); the curriculum proper is unchanged. |
| **Archive grammar (Path A)** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Path A composition produces a PR101-IDENTICAL archive (DP1 is init-only; not in archive bytes). Reuse `tac.substrates.pr101_lc_v2_clone.archive` AS-IS. NO DP1 codebook in archive. |
| **Archive grammar (Path B; deferred)** | `FORK_BECAUSE_PRINCIPLED_MISMATCH` | Path B requires `tac.substrates.pretrained_driving_prior.composition.compose_with` API per Catalog #211 — DPCOMP wrapper prepends DP1 codebook to PR101 archive. This IS the canonical DP1 composition API; reuse AS-IS. |
| **Inflate runtime (Path A)** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Reuse `tac.substrates.pr101_lc_v2_clone.inflate` AS-IS. NO DP1 consumption at inflate. |
| **Inflate runtime (Path B; deferred)** | `FORK_BECAUSE_PRINCIPLED_MISMATCH` | Path B requires NEW `tac.substrates.dp1_pr101_composition.inflate` that calls `tac.substrates.pretrained_driving_prior.composition.decompose` FIRST then routes the inner bytes to PR101 inflate. ~50 LOC; reviewable. Catalog #146 contest-compliant template applies. |
| **Score-aware loss** | `ADOPT_CANONICAL_BECAUSE_SERVES` (with soft-prior REVISION) | Reuse `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164. ADD optional `dashcam_prior_loss` term per `tac.substrates.pretrained_driving_prior.prior_application.DashcamPriorLoss` (already canonical; lambda_prior=0.05 default). The composition is ADDITIVE: `L_total = L_score + lambda_prior × L_dashcam_prior`. |
| **Tier-1 engineering (autocast/TF32/torch.compile/no_grad)** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Per Catalog #270 dispatch optimization protocol + #172/#178/#179/#180 sister gates. Inherit ALL canonical engineering primitives from PR101_lc_v2 substrate. NO substrate-specific divergence. |
| **Scorer routing** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Per Catalog #205 `select_inflate_device` + Catalog #226 `gate_auth_eval_call`. Inherit canonical scorer routing from PR101_lc_v2 substrate. NO substrate-specific divergence. |
| **EMA + eval_roundtrip** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Per CLAUDE.md "EMA — NON-NEGOTIABLE" (decay=0.997) + "eval_roundtrip — NON-NEGOTIABLE" (default True). Canonical `tac.training.EMA` + `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training`. |
| **Differentiable scorer-preprocess** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Per CLAUDE.md "eval_roundtrip" non-negotiable + the canonical `patch_upstream_yuv6_globally` / `load_differentiable_scorers` pattern. |
| **Archive build + inflate-LOC budget** | `ADOPT_CANONICAL_BECAUSE_SERVES` | Per HNeRV parity L4 (≤100 LOC inflate budget waiver to ≤200 for PR101 sidecar fidelity). Path A inflate IS PR101_lc_v2.inflate AS-IS. Path B inflate adds ~50 LOC for compose/decompose dispatch; total ≤250 LOC with rationale per Catalog #328 (`check_submission_inflate_py_under_loc_budget`). |

**Summary:** 11 layers ADOPT canonical (DP1 codebook loader + PR101 HNeRV decoder + curriculum + archive Path A + inflate Path A + score-aware loss + Tier-1 + scorer routing + EMA + eval_roundtrip + differentiable scorer-preprocess + inflate LOC budget); 2 layers FORK (weight-init bridge for shape mismatch; Path B archive/inflate for DPCOMP wrapper). The fork ratio is 13% (well within the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode — substrate-engineering IS unique by nature; the bolt-on portions SHARE canonical infrastructure aggressively).

**Per CLAUDE.md "Forbidden force-canonical-without-evaluation-of-suppression" FORBIDDEN pattern:** every FORK above is paired with a substrate-specific reason (shape mismatch in weight-init bridge; DPCOMP wrapper protocol in Path B archive/inflate). Every ADOPT is paired with an explicit "canonical serves" rationale. NO blind canonicalization.

---

## 5. Mathematical formulation

### 5.1 Composition as Wyner-Ziv conditional encoding

Per Wyner-Ziv 1976 source coding with side-information theorem: given source X (the contest video) and side-information Y (DP1's prior over the dashcam manifold), the achievable rate at distortion D is:

```
R_WZ(D | Y) = inf I(X; X̂ | Y)
```

where the infimum is over all X̂ such that `E[d(X, X̂)] ≤ D`. The conditional encoding rate is bounded below by `H(X | Y)`, the conditional entropy of the contest video given the dashcam prior.

**Translated to pact:**
- `X` = contest video frames (600 frames × 384×512×3 RGB at uint8 = ~354 MB raw)
- `Y` = DP1 codebook (162 KB frozen distilled prior over Comma2k19)
- `R(X)` = PR101_lc_v2 archive size = ~178 KB (= H(X) per Selfcomp's calculation)
- `R_WZ(X|Y)` = composed archive size = ?

If DP1 carries fraction `f` of mutual information with the contest video:
- `I(X; Y) = f × H(X)` where `f ∈ [0, 1]`
- `H(X|Y) = H(X) - I(X;Y) = (1-f) × H(X)`
- Predicted composed archive size = `(1-f) × 178 KB`

For different values of `f`:
- `f = 0.20` (20% mutual info; conservative): composed = 142 KB; net change vs PR101-alone = -36 KB → ΔS_rate = -0.024
- `f = 0.30` (30%; OOD transfer typical): composed = 125 KB; net change = -53 KB → ΔS_rate = -0.035
- `f = 0.50` (50%; high overlap; OPTIMISTIC): composed = 89 KB; net change = -89 KB → ΔS_rate = -0.059

**CRITICAL:** the composed archive size INCLUDES DP1's 162 KB codebook IF the codebook is shipped (Path B). In that case:
- Path B composed size = 162 KB (DP1) + (1-f) × 178 KB (PR101 residual)
- Net change vs PR101-alone = 162 - f × 178 KB

The Path B composition is RATE-POSITIVE (worse than PR101-alone) UNLESS `f > 162/178 = 0.910`. I.e., DP1 must carry >91% of mutual info with contest for Path B to be rate-feasible. **THIS IS NOT REALISTIC** for OOD pretrain.

**Path A is the canonical path:** DP1 acts as Stage-1 weight initialization ONLY; ZERO DP1 bytes in archive. The composed archive size is byte-identical to PR101-alone. The gain is in OPTIMIZATION (better starting point → better final score at same archive bytes).

### 5.2 Path A gain mechanism

Path A's predicted gain operates through OPTIMIZATION LANDSCAPE NAVIGATION, not RATE REDUCTION:

```
L(θ) = α × B(θ) / N + β × d_seg(θ) + γ × √d_pose(θ)
```

where `θ` is the PR101 decoder weight vector. The PR101 PR95 8-stage curriculum descends `L(θ)` via SGD; the starting point `θ_0` matters because L is non-convex.

**Random init:** `θ_0 ~ N(0, σ²)` (standard PR101 init); final `θ_T` after PR95 curriculum lies at a local minimum reachable in T epochs.

**DP1 init:** `θ_0 = project_dp1_to_pr101(DP1_codebook)`; this places `θ_0` on the dashcam-statistics manifold; the SGD trajectory from `θ_0` finds a different local minimum `θ_T'` that may be:
1. **Same as random-init `θ_T`** (random init also converges to dashcam manifold) — gain is 0
2. **Lower than `θ_T`** (DP1 init guides SGD past a saddle that random init misses) — gain is positive
3. **Higher than `θ_T`** (DP1 init places `θ_0` in a basin that traps SGD) — gain is negative (CARGO-CULT RISK per Contrarian dissent)

Predicted gain (per DreamerV3 + canonical fine-tuning literature): 5-15% relative score improvement IF the OOD-similarity is high (>30%); 0% IF OOD-similarity is low (<20%); -5% IF DP1 is dramatically off-distribution.

### 5.3 Predicted ΔS band per Catalog #296 Dykstra-feasibility check

Per Catalog #296 `check_substrate_predicted_band_has_dykstra_feasibility_check`, the predicted ΔS band MUST carry a Dykstra-feasibility intersection check OR a first-principles citation.

**First-principles citation:** Wyner-Ziv 1976 conditional rate theorem (cited above) + Hafner 2023 DreamerV3 pretrained-world-model + actor empirical results + Tishby-Zaslavsky 2015 information-bottleneck lens.

**Dykstra-feasibility check:** the composed substrate's predicted ΔS band must lie in the intersection of:

1. **Rate-feasibility set** `C_rate = {(R, S, P) : R ≤ R_PR101_alone}` (Path A preserves PR101 archive bytes; Path B requires Wyner-Ziv `f > 0.91` which is INFEASIBLE for OOD pretrain) → Path A: PASS; Path B: FAIL
2. **Distortion-feasibility set** `C_dist = {(R, S, P) : S ≤ S_PR101 - ΔS_PathA AND P ≤ P_PR101 - ΔP_PathA}` (the optimization-landscape gain manifests as reduced d_seg + reduced d_pose) → PASS conditional on Carmack Path A weight-init compatibility verification (op-routable #2)
3. **OOD-similarity-feasibility set** `C_OOD = {Comma2k19 ∩ Contest_video : 95% CI overlap > 30%}` (DP1 prior is useful only if Comma2k19 and contest share statistics) → UNCONDITIONAL on op-routable #1 outcome
4. **Architecture-compatibility-feasibility set** `C_arch = {projection_residual(DP1_codebook → PR101_decoder_weights) < 0.5}` → UNCONDITIONAL on op-routable #2 outcome

The intersection `C_rate ∩ C_dist ∩ C_OOD ∩ C_arch` is NON-EMPTY iff:
- Path A is chosen (Path B fails C_rate)
- OOD-similarity probe (op-routable #1) yields >30% overlap
- Architecture-compatibility probe (op-routable #2) yields projection residual <0.5

**Predicted ΔS band per Dykstra-feasibility:**
- **Best case** (all 4 sets non-trivially overlap; OOD >50% AND projection residual <0.3): `[-0.012, -0.008]` → CPU score `[0.180, 0.184]`
- **Realistic case** (OOD ~30-40% AND projection residual ~0.3-0.5): `[-0.008, -0.004]` → CPU score `[0.184, 0.188]`
- **Marginal case** (OOD ~20-30% AND projection residual ~0.5): `[-0.004, -0.002]` → CPU score `[0.188, 0.190]`
- **Worst case** (OOD <20% OR projection residual >0.5): `[+0.002, 0]` → CPU score `[0.192, 0.194]` (no gain; possibly worse than PR101-alone)

**Per Catalog #324:** `predicted_band_validation_status: pending_post_training` (the validation IS the post-training Tier-C density re-measurement on the composed archive sha; cannot be done pre-build). Reactivation criterion: post-training paired CPU+CUDA auth eval on composed archive sha; OOD-similarity probe outcome and architecture-compatibility projection residual jointly gate which sub-band applies.

### 5.4 Composition_alpha prediction per Catalog #322

Per Catalog #322 `check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha`, the autopilot ranker v2 cascade (`tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2`) MUST consult per-substrate-pair composition_alpha for ANY composition prediction.

**Predicted composition_alpha for DP1+PR101 (Path A):**
- DP1-alone predicted contribution: `ΔS_DP1_alone ≈ -0.005` (per `lane_pretrained_driving_prior_phase_2_20260514` smoke; conservative)
- PR101-alone predicted contribution: `ΔS_PR101_alone ≈ -0.000` (PR101 IS the baseline; ΔS measured against itself is 0)
- Naive additive prediction: `ΔS_naive = ΔS_DP1_alone + ΔS_PR101_alone = -0.005`
- Predicted composition_alpha (sub-additive band per Catalog #322): `α ∈ [0.6, 0.95]` (4-of-8 evidence: shared dominant axis → sub-additive)
- Adjusted prediction: `ΔS_composed = α × ΔS_naive ≈ -0.003 to -0.005` (matches Realistic case above)

**HOWEVER** — the DP1 prior is NOT additive with PR101; it's a STARTING POINT change. The standard Catalog #322 alpha model is for ADDITIVE compositions (e.g., HStack residual sidecar over a base archive). DP1+PR101 Path A is a DIFFERENT composition mode: PRETRAIN+REFINEMENT (DreamerV3 pattern), NOT additive bytes.

**Per Catalog #322 Pattern G (NEW; queued for landing):** PRETRAIN+REFINEMENT composition mode requires a DIFFERENT alpha framework — specifically, the gain is `ΔS_composed = max(ΔS_PR101_alone, ΔS_PR101_with_DP1_init)`. The "max" reflects that we always take the better of: random-init final or DP1-init final. Empirical prediction: `ΔS_composed ∈ [-0.012, +0.002]` depending on Path A feasibility (per Section 5.3 worst-case includes possibility of negative gain if DP1 traps SGD).

The autopilot ranker SHOULD consume the predicted band per Section 5.3 directly, NOT via the additive composition_alpha cascade. Sister op-routable: extend `tac.optimization.substrate_composition_matrix` with `composition_mode` field ∈ `{additive, pretrain_refinement, side_channel, replacement}` so the autopilot ranker selects the right gain model per substrate pair.

---

## 6. Architecture sketch (PR95-paradigm bind-all-ingredients package)

Per HNeRV parity L7 (bolt-on vs substrate-engineering split), DP1+PR101 composition is substrate-engineering; target ~600-800 LOC bound into ONE coherent package reviewable in 30 seconds.

### 6.1 Package structure

```
src/tac/substrates/dp1_pr101_composition/
├── __init__.py                         # ~80 LOC; public API + 13 HNeRV parity lessons table + Catalog #124 8-field declaration
├── architecture.py                     # ~50 LOC; thin wrapper around Pr101LcV2CloneSubstrate with DP1_init kwarg
├── weight_init_bridge.py               # ~200 LOC; NEW canonical helper: project_dp1_codebook_to_pr101_decoder
├── score_aware_loss.py                 # ~50 LOC; thin wrapper around PR101 loss + optional DashcamPriorLoss
├── archive.py                          # ~80 LOC; Path A: PR101 archive AS-IS; Path B (deferred): DPCOMP wrapper
├── inflate.py                          # ~150 LOC; Path A: PR101 inflate AS-IS; Path B (deferred): decompose + dispatch
├── curriculum.py                       # ~50 LOC; thin wrapper around curriculum_enhanced.py with DP1-init Stage-0
└── tests/
    └── test_dp1_pr101_composition.py   # ~250 LOC; ~25 dedicated tests
```

**Total LOC:** ~660 (substrate code) + ~250 (tests) = ~910. This is at the upper end of substrate-engineering budget (PR101 GOLD was 605 LOC; DP1+PR101 composition adds the weight-init bridge ~200 LOC + wrappers ~110 LOC).

### 6.2 Public API (the 30-second-reviewable surface)

```python
# src/tac/substrates/dp1_pr101_composition/__init__.py

from .architecture import (
    Dp1Pr101CompositionConfig,
    Dp1Pr101CompositionSubstrate,
)
from .weight_init_bridge import (
    project_dp1_codebook_to_pr101_decoder,
    Dp1Pr101WeightInitProjectionResult,
    WeightInitProjectionError,
)
from .score_aware_loss import (
    Dp1Pr101ScoreAwareLoss,
    Dp1Pr101ScoreAwareLossConfig,
)
from .archive import (
    pack_archive_path_a,        # Path A: PR101 archive AS-IS
    pack_archive_path_b,        # Path B: DPCOMP wrapper (deferred)
    DP1_PR101_ARCHIVE_VERSION,
)
from .curriculum import (
    train_dp1_pr101_composition,
    Dp1Pr101CurriculumConfig,
)

__all__ = [
    "Dp1Pr101CompositionConfig",
    "Dp1Pr101CompositionSubstrate",
    "project_dp1_codebook_to_pr101_decoder",
    "Dp1Pr101WeightInitProjectionResult",
    "WeightInitProjectionError",
    "Dp1Pr101ScoreAwareLoss",
    "Dp1Pr101ScoreAwareLossConfig",
    "pack_archive_path_a",
    "pack_archive_path_b",
    "DP1_PR101_ARCHIVE_VERSION",
    "train_dp1_pr101_composition",
    "Dp1Pr101CurriculumConfig",
]
```

### 6.3 Core weight-init bridge (NEW canonical helper)

```python
# src/tac/substrates/dp1_pr101_composition/weight_init_bridge.py

# SPDX-License-Identifier: MIT
"""Weight-init bridge from DP1 codebook to PR101 HNeRV decoder.

Per Carmack Path A: DP1 acts as Stage-1 weight initialization ONLY (zero DP1
bytes in archive). The bridge projects DP1's (road-plane PCA basis +
sky-horizon profile + vehicle appearance) onto PR101 HNeRV's ConvNet decoder
weight space via least-squares.

Mathematical contract:
    Given DP1 codebook C (162 KB; road-plane K=8 PCA modes + sky-horizon
    8-row profile + vehicle K=64 patches) and PR101 decoder template
    W_PR101 (depth-8 ConvNet with FiLM; ~178 KB after Brotli compression),
    find W_init such that:
        W_init = argmin_W ||proj_DP1(W) - C||²
    where proj_DP1 projects W onto DP1's subspace.

    The residual norm ||proj_DP1(W_init) - C|| / ||C|| is the
    projection_residual: lower is better (DP1 captures more of W_init's
    variance). If projection_residual < 0.3, DP1 init is HARD-EARNED-FEASIBLE;
    if projection_residual > 0.7, DP1 init degenerates to random init.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from tac.substrates.pretrained_driving_prior.codebook import (
    CodebookV1,
    load_codebook,
)
from tac.substrates.pr101_lc_v2_clone.architecture import Pr101LcV2CloneConfig


@dataclass(frozen=True)
class Dp1Pr101WeightInitProjectionResult:
    """Result of projecting DP1 codebook onto PR101 decoder weight space."""
    initialized_state_dict: Mapping[str, torch.Tensor]
    projection_residual: float           # ||proj_DP1(W_init) - C|| / ||C||
    projection_rank: int                  # rank of DP1's subspace as seen by PR101
    per_layer_projection_residuals: dict[str, float]  # per-layer breakdown
    feasibility_verdict: str              # HARD_EARNED | MARGINAL | CARGO_CULTED


class WeightInitProjectionError(RuntimeError):
    """Raised when DP1 codebook cannot be projected onto PR101 decoder."""


def project_dp1_codebook_to_pr101_decoder(
    *,
    dp1_codebook: CodebookV1,
    pr101_config: Pr101LcV2CloneConfig,
    feasibility_threshold: float = 0.5,
    rng_seed: int = 42,
) -> Dp1Pr101WeightInitProjectionResult:
    """Project DP1 codebook onto PR101 HNeRV decoder weight space."""
    # 1. Materialize a random PR101 decoder template
    # 2. For each conv layer in PR101:
    #    a. Extract layer weight tensor W_l of shape (out_ch, in_ch, k, k)
    #    b. Flatten to (out_ch × in_ch × k × k,) vector
    #    c. Project onto DP1 subspace (road-plane PCA + sky-horizon + vehicles)
    #    d. Compute residual norm
    # 3. Initialize W_init = projection + small random perturbation
    # 4. Compute aggregate projection_residual
    # 5. Verdict per threshold
    ...
```

### 6.4 Training curriculum integration (Stage 0 DP1-init)

```python
# src/tac/substrates/dp1_pr101_composition/curriculum.py

def train_dp1_pr101_composition(
    *,
    config: Dp1Pr101CurriculumConfig,
    contest_video_path: Path,
    dp1_codebook_path: Path,
    output_dir: Path,
    **kwargs,
) -> Dp1Pr101TrainingResult:
    """3-stage training: DP1 init → PR101 PR95 curriculum → Stage 3 export.

    Stage 0 (DP1 init; ~0 GPU-seconds):
        - load DP1 codebook
        - project onto PR101 decoder weight template
        - report projection_residual (Carmack Path A feasibility gate)
        - if projection_residual > threshold, ABORT with operator-routable

    Stage 1-8 (PR101 PR95 8-stage curriculum; ~1800 GPU-seconds at A100):
        - reuse tac.substrates.pr101_lc_v2_clone.curriculum_enhanced AS-IS
        - INITIAL weights from Stage 0 (NOT random)
        - score-aware loss optionally adds DashcamPriorLoss (lambda_prior=0.05)

    Stage 9 (export; ~30 GPU-seconds):
        - pack_archive_path_a(trained_state_dict, latents, poses)
        - emit composed_archive_path_a.zip (Path B deferred)
        - canonical auth_eval per Catalog #226 gate_auth_eval_call
    """
    ...
```

### 6.5 Score-aware loss with optional DP1 soft-prior

```python
# src/tac/substrates/dp1_pr101_composition/score_aware_loss.py

@dataclass(frozen=True)
class Dp1Pr101ScoreAwareLossConfig:
    """Configuration for DP1+PR101 score-aware loss."""
    alpha: float = 25.0                  # rate weight (canonical PR101)
    beta: float = 100.0                  # seg weight (canonical PR101)
    gamma: float = 1.0                   # pose weight (canonical PR101)
    archive_denom_bytes: int = 37_545_489  # canonical
    use_dashcam_prior: bool = True       # optional soft prior
    lambda_dashcam_prior: float = 0.05   # canonical DP1 default
    dashcam_prior_warmup_epochs: int = 50  # ramp lambda from 0 to default over warmup


class Dp1Pr101ScoreAwareLoss(torch.nn.Module):
    """Canonical PR101 score-aware loss + optional DP1 dashcam prior.

    Total loss:
        L_total = alpha * B / N
                + beta * d_seg
                + gamma * sqrt(d_pose)
                + lambda_dashcam(epoch) * L_dashcam_prior

    where:
        L_dashcam_prior = ||proj_DP1(renderer_rgb) - codebook_proj(renderer_rgb)||²
    """
    def __init__(self, config: Dp1Pr101ScoreAwareLossConfig, dp1_codebook: CodebookV1):
        ...

    def forward(
        self,
        renderer_rgb: torch.Tensor,
        archive_bytes_size: int,
        epoch: int,
        scorer_outputs: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...
```

---

## 7. Training curriculum

### 7.1 Stage-by-stage budget

| Stage | Description | GPU time @ A100 | Memory | Cost (Modal A100 $1.50/hr) |
|:------|:------------|:---------------:|:------:|:--------------------------:|
| 0 (DP1 init) | Project DP1 codebook onto PR101 decoder template; report residual | ~5 seconds | 4 GB | $0.002 |
| 1-8 (PR95 8-stage) | Canonical PR95 curriculum on contest video; DP1-warm-started weights | ~30 minutes | 16 GB | $0.75 |
| 9 (export + auth_eval) | Pack archive; paired CPU+CUDA auth eval | ~5 minutes | 8 GB | $0.13 |
| **Total** | **DP1+PR101 Path A end-to-end** | **~35 minutes** | **16 GB** | **~$0.88** |

**Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE":** every Modal dispatch follows canonical harvest pattern.

### 7.2 Canonical scorer routing per Catalog #226

Stage 9 auth_eval routes through canonical `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call(...)` per Catalog #226. The helper handles:
- Catalog #205 `select_inflate_device` (CPU/CUDA agnostic with PACT_INFLATE_DEVICE env var)
- Catalog #249 phantom-score directory prevention (output JSON filename matches actual device)
- Catalog #127 authoritative tag custody (axis × hardware_substrate × evidence_grade triple)
- Catalog #221 result-artifact fail-closed (score_claim=false default; promotion_eligible=false)
- Catalog #324 predicted_band post-training validation (pending_post_training initially)

### 7.3 Canonical scorer-loss helper routing per Catalog #164

All loss computation routes through `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164. The Dp1Pr101ScoreAwareLoss wrapper ADDS the optional DashcamPriorLoss but PRESERVES the canonical scorer-loss helper contract for the 3 canonical terms (rate + seg + pose).

### 7.4 Per-stage Catalog discipline

| Stage | Catalog gates enforced |
|:------|:-----------------------|
| 0 | Catalog #229 (premise verification — DP1 codebook exists, PR101 config exists, projection residual reportable) |
| 1-8 | Catalog #270 (dispatch optimization protocol — Tier 1 engineering + Tier 2 hardware + Tier 3 substrate); Catalog #172/#178/#179/#180 (autocast/TF32/torch.compile/no_grad); Catalog #164 (canonical scorer-loss helper); Catalog #166 (Modal HEAD parity + sentinel files); Catalog #205 (canonical inflate device); Catalog #244 (NVML env block) |
| 9 | Catalog #226 (canonical gate_auth_eval_call); Catalog #127 (authoritative tag custody); Catalog #221 (result-artifact fail-closed); Catalog #249 (phantom-score directory); Catalog #324 (predicted_band post-training validation); Catalog #270 (dispatch optimization protocol verification) |

---

## 8. Archive grammar (Path A)

Per HNeRV parity L3 (monolithic single-file `0.bin` or explicitly justified multi-file) + Catalog #124 8-field declaration:

### 8.1 Path A archive structure (canonical; first-implementation target)

```
composed_archive.zip
└── 0.bin                                # PR101_lc_v2 monolithic single-file format AS-IS
    ├── DECODER_BLOB (offset 0; length DECODER_BLOB_LEN)
    │   └── Multi-Brotli-stream decoder state_dict (canonical PR101 lc_v2)
    ├── LATENT_BLOB (offset DECODER_BLOB_LEN; length LATENT_BLOB_LEN)
    │   └── LZMA-compressed latent tensors
    └── SIDECAR (raw bytes, optional)
        └── Per-pair poses, masks, etc.
```

**Critical:** Path A archive IS BYTE-IDENTICAL to a PR101_lc_v2 archive trained from the same final weights. DP1 leaves NO trace in the archive bytes; DP1's contribution is in the WEIGHT-SPACE STARTING POINT, not in the archive bytes.

**Per Catalog #220 substrate L1+ scaffold operational mechanism:** the operational mechanism is `score_improvement_mechanism_status=OPERATIONAL via DP1-warm-start weight initialization`. The bytes are NOT added (Path A); the operational mechanism is at training time, not inflate time. This requires explicit declaration in lane registry notes per Catalog #220.

### 8.2 Path B archive structure (deferred Phase 2)

```
composed_archive.zip
└── 0.bin
    ├── DPCOMP_HEADER (13 bytes per Catalog #211: magic b"DPC\x00" + version + length prefix + base_tag)
    ├── DP1_CODEBOOK_BLOB (length per DPCOMP header; ~162 KB)
    └── PR101_INNER_ARCHIVE_BYTES (PR101_lc_v2 monolithic single-file format)
```

**Per Catalog #211:** `compose_with(dp1_bytes, pr101_archive_bytes, base_substrate="pr101_lc_v2")` produces Path B archive. The DPCOMP wrapper is byte-stable; `decompose(composed_bytes)` peels off the prefix and returns PR101 inner archive bytes.

**Path B is DEFERRED per Section 5.1 mathematical analysis:** Path B is rate-positive (worse than PR101-alone) unless DP1 carries >91% mutual info, which is NOT realistic for OOD pretrain.

### 8.3 Fixed offsets in `codec.py` source

Per HNeRV parity L3:

```python
# src/tac/substrates/dp1_pr101_composition/archive.py
DP1_PR101_ARCHIVE_VERSION = b"DP1P01v1"  # 8-byte magic
DECODER_BLOB_LEN = 162_164  # canonical PR101_lc_v2
LATENT_BLOB_LEN = 15_387    # canonical PR101_lc_v2
SIDECAR_OFFSET = DECODER_BLOB_LEN + LATENT_BLOB_LEN  # 177,551
```

Path A and Path B both declare these fixed offsets so the parser is deterministic + reviewable in 30 seconds.

### 8.4 Inflate runtime (≤200 LOC per HNeRV parity L4 + Catalog #328)

Per Catalog #328 `check_submission_inflate_py_under_loc_budget` (≤200 lines):

```python
# src/tac/substrates/dp1_pr101_composition/inflate.py  (Path A: ≤150 LOC)
# Reuses src/tac/substrates/pr101_lc_v2_clone/inflate.py via direct import
from tac.substrates.pr101_lc_v2_clone.inflate import inflate as _pr101_inflate

def inflate(archive_dir, output_dir, file_list):
    """DP1+PR101 Path A inflate: byte-identical to PR101_lc_v2 inflate."""
    return _pr101_inflate(archive_dir, output_dir, file_list)
```

Path A inflate is ~30 LOC (thin delegator). The DP1 prior leaves NO trace in inflate (no scorer load, no codebook read).

---

## 9. Distinguishing feature per Catalog #272

Per Catalog #272 `check_substrate_distinguishing_feature_integration_contract`, every substrate at L2+ MUST declare 4 fields: `distinguishing_feature_name` / `distinguishing_bytes_path` / `inflate_consumer_function` / `byte_mutation_smoke_passes`.

### 9.1 Path A distinguishing feature

**`distinguishing_feature_name`:** `dp1_warm_start_weight_initialization`

**`distinguishing_bytes_path`:** N/A for Path A (NO DP1 bytes in archive; all distinguishing bytes are PR101_lc_v2 archive bytes that result from DP1-warm-started training trajectory)

**`inflate_consumer_function`:** N/A for Path A (DP1 has no inflate-time consumer; the distinguishing effect is at training time)

**`byte_mutation_smoke_passes`:** N/A for Path A (no DP1 bytes to mutate; byte mutation smoke applies to PR101 archive bytes which is the SISTER PR101_lc_v2 substrate's Catalog #272 obligation, not this one)

**Special declaration:** Per Catalog #220 acceptance cascade (a) operational mechanism `OPERATIONAL via dp1_warm_start_weight_initialization at training time; affects PR101 archive bytes downstream`. Lane registry notes MUST cite this explicitly per Catalog #220.

### 9.2 Path B distinguishing feature (deferred)

**`distinguishing_feature_name`:** `dp1_codebook_side_information_at_decoder`

**`distinguishing_bytes_path`:** offset 13 (post-DPCOMP-header) length DP1_CODEBOOK_LEN bytes; inflate path reads DP1 codebook FIRST, then uses it as side-info per Atick-Redlich cooperative-receiver framing during PR101 inflate.

**`inflate_consumer_function`:** `tac.substrates.dp1_pr101_composition.inflate.apply_dp1_side_info_to_pr101(dp1_codebook_bytes, pr101_inflated_frames)`

**`byte_mutation_smoke_passes`:** REQUIRED for Path B at landing; verify via `tools/verify_distinguishing_feature_byte_mutation.py` that mutating one byte in DP1_CODEBOOK_BLOB changes the rendered frame output.

**Path B distinguishing feature contract is FULL** when Phase 2 lands; for the Phase 1 (Path A only) landing, the distinguishing feature is operational-mechanism only per Catalog #220.

---

## 10. Cargo-cult audit per assumption (per Catalog #303)

This section is MANDATORY per Catalog #303 `check_substrate_design_memo_has_cargo_cult_audit_section` + the HARD-EARNED-vs-CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`).

### Cargo-cult #1: "DP1's pretrained dashcam prior generalizes to the contest video"

**Classification:** CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION

**Why CARGO-CULTED:** DP1 is distilled from Comma2k19 (specific dataset; specific cameras; specific routes). The contest video is a SPECIFIC dashcam clip from POTENTIALLY a different camera mount / lighting / road type. The OOD-similarity is NOT established.

**Unwind path:** Op-routable #1 (OOD-similarity probe; $0 local CPU): compute per-frame statistics on Comma2k19 sample (8 chunks) vs contest video; report 95% CI overlap on:
- mean luminance
- road-plane PCA spectrum (top-8 eigenvalues)
- sky-horizon profile (per-row mean+chroma)
- vehicle density (frame-level vehicle pixel count via mock vehicle detector or SegNet-derived)

**Acceptance criterion:** OOD-similarity > 30% on 3-of-4 dimensions → HARD-EARNED-FEASIBLE; OOD-similarity < 20% on >1 dimension → CARGO-CULTED-CONFIRMED (DP1+PR101 composition DEFERRED-pending-redesign).

### Cargo-cult #2: "DP1+PR101 composition is α-additive (composition_alpha ≈ 1.0)"

**Classification:** CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION

**Why CARGO-CULTED:** Per Catalog #322 framework + the 4-of-8 sub-additive evidence for substrate pairs sharing a dominant axis. DP1 and PR101 both target the RGB-axis renderer output; both target pose-axis + seg-axis + rate-axis. By every Cathedral autopilot ranker v2 cascade heuristic, this is a SUB-ADDITIVE composition.

**Unwind path:** Op-routable #4 (paired-α probe; $3 Modal T4): empirically measure composition_alpha on contest video for the trained Path A archive vs DP1-alone smoke + PR101-alone fec6 anchor. Report α via `tac.optimization.substrate_composition_matrix.classify_pairwise_composability_per_cell`.

**Acceptance criterion:** α > 0.7 → HARD-EARNED-WITH-REVISION (sub-additive but substantial gain); α ∈ [0.3, 0.7] → MARGINAL (worth exploring stronger DP1 codebook variants); α < 0.3 → CARGO-CULTED-CONFIRMED (composition mode mismatch; revisit Section 5.4 PRETRAIN+REFINEMENT alpha model).

### Cargo-cult #3: "PR101 HNeRV decoder architecture is compatible with DP1 codebook weight initialization"

**Classification:** CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION

**Why CARGO-CULTED:** DP1's codebook is shaped (road-plane PCA basis + sky-horizon profile + vehicle appearance). PR101's HNeRV decoder weights are (H, W, 3) frame-producing ConvNet weights with FiLM conditioning. Shape compatibility is non-trivial; the projection residual is the empirical signal.

**Unwind path:** Op-routable #2 (architecture-compatibility probe; $0 local CPU): execute `project_dp1_codebook_to_pr101_decoder(dp1_codebook, pr101_config)` from the new canonical helper; report projection_residual + per-layer_projection_residuals.

**Acceptance criterion:** projection_residual < 0.3 → HARD-EARNED-FEASIBLE (DP1 captures >70% of weight-space variance); projection_residual ∈ [0.3, 0.7] → MARGINAL (partial DP1 init; expected gain is reduced); projection_residual > 0.7 → CARGO-CULTED-CONFIRMED (DP1 init degenerates to random init; composition deferred).

### Cargo-cult #4: "Stage 1 init + Stage 2 PR101 HNeRV refinement is the optimal training curriculum"

**Classification:** HARD-EARNED

**Why HARD-EARNED:** Per DreamerV3 (Hafner 2023) + canonical fine-tuning literature (BERT, GPT, ImageNet→COCO transfer). The pretrain→fine-tune pattern IS canonical and empirically validated across 100+ vision/NLP papers.

**No unwind needed.** The question is the EPOCH BUDGET SPLIT, not the curriculum structure. Default: PR101 PR95 8-stage curriculum AS-IS with DP1 warm-start; no additional pretrain epochs.

### Cargo-cult #5: "The fec6 archive `f174192aeadf...` is a representative anchor for DP1+PR101 composition predictions"

**Classification:** HARD-EARNED-WITH-REVISION

**Why HARD-EARNED-WITH-REVISION:** fec6 IS the canonical PR101_lc_v2 anchor with master-gradient extracted. It's the right anchor for PR101 BASELINE measurement. REVISION: the composed substrate produces a NEW archive sha (Catalog #229 premise verification confirms this); the fec6 anchor only informs the PR101-side baseline. Need separate anchor extraction post-Stage-2 for the composed substrate.

**Unwind path:** Post-Stage-2 master-gradient extraction on the composed archive sha via `tools/extract_master_gradient.py`. Sister-anchor for the composed substrate.

### Cargo-cult #6: "DP1's prior + PR101's overfitting capacity together unlock sub-0.190 floor"

**Classification:** CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION

**Why CARGO-CULTED:** This is the BIG predicted-band claim ([0.180, 0.190] CPU). It rests on:
- (a) DP1 OOD-similarity to contest > 30% (cargo-cult #1)
- (b) Architecture compatibility projection_residual < 0.5 (cargo-cult #3)
- (c) Composition_alpha > 0.6 (cargo-cult #2)
- (d) The optimization-landscape gain manifests at training time (theoretical; needs empirical proof)

**Unwind path:** Op-routables #1, #2, #4 in sequence (≤$5 total budget). Only the full Stage 2 smoke (op-routable #3; $3-5) produces the actual ΔS measurement.

**Per CLAUDE.md "Forbidden premature KILL":** if the predicted band is NOT achieved (e.g., composed CPU score ~ PR101-alone score), the verdict is DEFERRED-pending-research (try Path B; try stronger DP1 codebook; try different OOD pretrain dataset). NOT KILL.

### Cargo-cult #7: "DP1's lambda_prior=0.05 is the optimal soft-prior weight for the composition"

**Classification:** HARD-EARNED-WITH-REVISION

**Why HARD-EARNED-WITH-REVISION:** lambda_prior=0.05 is the existing DP1 default from `tac.substrates.pretrained_driving_prior.prior_application.PriorApplicationWeights`. The default is canonical for DP1-alone. REVISION: the optimal lambda_prior for the COMPOSITION may differ (e.g., higher value reinforces DP1; lower value lets PR101 dominate).

**Unwind path:** Hyperparameter sweep (deferred to Phase 2 if Path A baseline is promising): lambda_prior ∈ {0.01, 0.05, 0.1, 0.2}. ~$20-30 GPU sweep.

---

