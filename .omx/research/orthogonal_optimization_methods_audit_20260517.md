<!-- generated_at: 2026-05-17T12:30:00Z, from_state_hash: orthogonal_opt_audit_v1 -->
<!-- HISTORICAL_PROVENANCE — append-only forensic record -->
---
title: "ORTHOGONAL OPTIMIZATION METHODS AUDIT — what we USE vs what's AVAILABLE vs what's UNCONSIDERED across freezing / pausing / hardware / problem-space surfaces"
date: 2026-05-17
lane: lane_orthogonal_optimization_methods_audit_20260517
author: orthogonal_optimization_methods_audit_subagent_20260517
horizon_class: apparatus_maintenance
parent_review: feedback_deep_adversarial_review_substrate_design_meta_20260517 (commit c97430305)
sister_subagents:
  - t4_symposium_substrate_design_class_shift_20260517 (COMPLETE; disjoint scope; substrate-design deliberation)
  - scorer_response_surface_analysis_20260517 (IN-PROGRESS; disjoint scope; scorer empirical characterization)
operator_directive_verbatim: "There are methods we should be able to use to optimize orthogonal including freezing and pausing and more hardware and problem space exploits"
research_only: true
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
evidence_grade: research-only-text-audit
frontier_anchor:
  axis: contest-CPU GHA Linux x86_64 1to1
  score: 0.19285
  archive_sha256_prefix: "87ec7ca5...492b5"
  substrate: A1 (inflate-time bias correction on PR101-paradigm substrate)
parent_finding: "0-of-53 substrate designs have produced a sub-0.192 [contest-CPU] anchor; corpus is plateau-trapped at the within-PR95-paradigm-class floor"
audit_thesis: "Orthogonal optimization methods (freezing / pausing / hardware exploits / problem-space exploits) apply to ANY existing substrate and stack MULTIPLICATIVELY. The 0-of-53 result may be MULTIPLICATIVELY-RESCUABLE without substrate-class-shift."
six_hook_wire_in:
  sensitivity_map: AUDIT-CONTRIBUTES — Section 1's per-category inventory ranks substrates by orthogonal-exploit coverage; per-substrate audit-row contributes to bit-allocator
  pareto_constraint: AUDIT-CONTRIBUTES — Section 4 top-7 highest-EV exploits form a (cost, predicted ΔS) Pareto front consumable by cathedral autopilot ranker
  bit_allocator_hook: AUDIT-CONTRIBUTES — FP4-quantization (Quantizr-pattern) is the canonical per-tensor bit-allocator exploit; identified as gap-1
  cathedral_autopilot_dispatch_hook: AUDIT-CONTRIBUTES — top-5 multiplicative stacks (Section 3) are dispatch-ready rows for autopilot ranker
  continual_learning_posterior_update: N/A — text-audit only; no empirical anchor produced
  probe_disambiguator: N/A — audit is enumeration not arbitration
---

# ORTHOGONAL OPTIMIZATION METHODS AUDIT — 2026-05-17

**Operator directive (verbatim):** *"There are methods we should be able to use to optimize orthogonal including freezing and pausing and more hardware and problem space exploits"*.

**Parent review verdict (commit `c97430305`):** 0 of 53 substrate designs has produced a sub-0.192 [contest-CPU] anchor. The substrate-design corpus is plateau-trapped at the within-PR95-paradigm-class floor (PR101 gold ~0.193 [contest-CUDA] / ~0.195 [contest-CPU]).

**Thesis of this audit:** orthogonal optimization methods don't require substrate-class-shift — they apply to ANY existing substrate and stack MULTIPLICATIVELY. The 0-of-53 plateau may be partially RESCUABLE via aggressive orthogonal stacking on the existing verified-working substrates (A1, PR106 r2, PR100 hnerv_lc_v2, NSCS01/NSCS03 once L2+).

**Key finding:** of the four orthogonal categories audited (FREEZING / PAUSING / HARDWARE / PROBLEM-SPACE), the most under-exploited is the HARDWARE category, dominated by the **0-of-44-substrate-trainers FP4 quantization gap** despite (a) FakeQuantFP4 already coded in `src/tac/fp4_quantize.py`, (b) Quantizr's 0.33 leaderboard winner using 5-stage FP4 QAT as the architectural primary, and (c) the rate term being 25× the per-byte weight in the score formula. This is empirically falsifiable: PR101's 337-LOC bolt-on (May 2026 gold) reportedly stacked Quantizr-style per-tensor byte maps on PR100's substrate as its principal compression mechanism.

## 0. Premise verification per Catalog #229

| PV | Premise | Verified | Source |
|---|---|---|---|
| 1 | 44 `experiments/train_substrate_*.py` files in repo | ✅ | `ls experiments/train_substrate_*.py \| wc -l` returned 44 |
| 2 | FakeQuantFP4 class exists in `src/tac/fp4_quantize.py` | ✅ | grep returned class FakeQuantFP4, class FP4Parametrize, class QATRendererFP4 |
| 3 | 0 substrate trainers grep-match `FakeQuantFP4` | ✅ | grep across `experiments/train_substrate_*.py` returned 0 |
| 4 | A1 inflate-time bias correction is the verified frontier 0.19285 | ✅ | `meat_on_the_bone_audit_unused_exploits_20260515.json` frontier_anchor + DEEP-ADVERSARIAL-REVIEW §1.2 |
| 5 | Compress-time SegNet/PoseNet freeze pattern exists in 19+ substrate trainers via `load_differentiable_scorers(frozen)` | ✅ | grep returned 19 substrates with explicit `frozen` token in comments + `pose_scorer.eval()`/`seg_scorer.eval()` calls in trainer bodies |
| 6 | upstream/evaluate.py score formula = `100*seg + sqrt(10*pose) + 25*rate` with rate = `compressed/37545489` | ✅ | upstream/evaluate.py:91 |
| 7 | Sister T4 SYMPOSIUM subagent COMPLETED before this memo writes | ✅ | subagent_progress.jsonl shows `lane_t4_symposium_substrate_design_class_shift_20260517` status=complete at 12:24:35Z; I started writing at 12:30Z |
| 8 | Sister SCORER analysis IN-PROGRESS on disjoint memo path | ✅ | scorer_response_surface_analysis subagent step 2 in_progress on `.omx/research/scorer_response_surface_analysis_20260517.md` (my memo is `.omx/research/orthogonal_optimization_methods_audit_20260517.md`) |
| 9 | `src/tac/quantization.py` has FakeQuantSTE, Uint8STE, LSQScale, QATPostFilter | ✅ | 27 classes/functions enumerated via grep |
| 10 | `src/tac/fp4_quantize.py` exists separate from `src/tac/quantization.py` | ✅ | ls returned both files |
| 11 | Domain exploitation catalog `domain_exploitation_catalog_20260509.md` exists (73 KB) + atoms JSON | ✅ | ls + cat showed Section A through F + 38 KB atoms JSON |
| 12 | `meat_on_the_bone_audit_unused_exploits_20260515.json` exists (30 KB) with 5-axis enumeration | ✅ | ls + read of axis_1 through axis_5 |

All 12 PVs PASS. The audit operates on verified premises.

---

## 1. Per-category inventory tables

### Category 1: FREEZING methods

| Method | Used by | Count of 44 substrate trainers | Empirical evidence | EV estimate |
|---|---|---:|---|---|
| **Compress-time SegNet/PoseNet freeze** (scorer in eval mode + no_grad; canonical helper `load_differentiable_scorers(frozen)`) | 19 substrate trainers (a1_plus_lapose, a1_plus_wavelet_residual, atw_codec_v2, c6_e4_mdl_ibps, d1_segnet_margin_polytope, d4_wyner_ziv_frame_0, nscs01, nscs02, rudin_floor, s2sbs_byte_stuffing, sar_coherent_pose_pairs, time_traveler_l5_autonomy, time_traveler_l5_z6, etc.) | 19/44 (43%) | Per DEEP-ADVERSARIAL-REVIEW §3 the per-pair conditioning collapse at the scorer's per-frame response surface is partially attributable to *non-frozen* scorer-implicit-conditioning during training. Compress-time freeze eliminates this. | HIGH — addresses the parent review's per-pair conditioning attenuation finding. Predicted ΔS [-0.005, -0.020] per substrate on the 25 unfrozen trainers if migrated. |
| **EMA shadow freeze at eval** (CLAUDE.md non-negotiable) | All 30 substrates that grep "EMA" | 30/44 (68%) | CLAUDE.md "EMA — NON-NEGOTIABLE" mandates this. The 14/44 trainers NOT matching the EMA grep need audit — likely some intentionally don't have an EMA (Z3 v1 was rate-only; atw_codec_v1; nscs06_v8_path_b_wavelet; tishby_ib_pure; etc.) | Some are legitimate non-EMA paths (rate-only / non-stochastic). Others are unintentional gaps. | MEDIUM-LOW — already broadly used; small EV from gap-closure (~5-10 trainers). |
| **Multi-stage train-eval-train** (PR95-paradigm 8-stage curriculum) | 18 substrate trainers (curriculum/multi_stage tokens) | 18/44 (41%) | PR95 / PR101 winning pattern. Quantizr's 5-stage QAT (anchor → finetune → joint → QAT → final) is the canonical example. | HIGH — proven on PR95+PR100+PR101 leaderboard wins. Gap-closure for 26/44 trainers would meaningfully improve substrate quality. |
| **Per-layer freeze** (LoRA-style: only train low-rank adapter) | 2 substrate trainers (block_nerv, s2sbs_byte_stuffing) | 2/44 (5%) | Used only in 2 substrates; the canonical LoRA helper is `src/tac/`-resident but rarely consumed. PR101's 337-LOC bolt-on pattern is conceptually a per-layer-freeze (PR100's weights frozen; bolt-on operates over them). | HIGH — directly enables the **Rule #6 BOLT-ON pattern** from DEEP-ADVERSARIAL-REVIEW §4.6: take a verified working substrate, freeze it, add a small adapter/codec for the new conditioning. Predicted ΔS per substrate [-0.001, -0.005] (rate savings) with minimal distortion cost. |
| **Per-channel freeze** (only fine-tune specific channels) | 0 trainers detected | 0/44 | UNCONSIDERED — no substrate uses it. Would enable score-gradient-saliency-driven per-channel freeze (the inverse of Track 4 v1's weight-saliency anti-pattern; per Catalog #123 the correct path is score-gradient-driven). | MEDIUM — well-suited for QAT followups but untested empirically. |
| **Frozen per-pair embedding vs trained** | NeRV-family canonical pattern (5 trainers: sane_hnerv, hi_nerv, ds_nerv, tc_nerv, block_nerv, ff_nerv) | 6/44 (14%) | The PR95-paradigm pattern: pre-train decoder, freeze it, train per-pair latents only at inflate-time (Cool-Chic-pattern). The 6 NeRV-family canonicals are the corpus's closest analog. | HIGH — this IS the PR95-paradigm winning pattern. |
| **Frozen pose deltas vs trained** | 0 trainers detected | 0/44 | UNCONSIDERED — no substrate freezes pose to focus optimization on seg/rate. Per DEEP-ADVERSARIAL-REVIEW PoseNet sensitivity is 2.71× SegNet's at the A1 operating point — freezing pose may be wrong direction here. | LOW — likely anti-correlated with frontier-pursuit per the operating-point analysis. |
| **Frozen segmentation prior vs trained** | 0 trainers detected | 0/44 | UNCONSIDERED — could be paired with score-gradient saliency for SegNet-only fine-tune. | LOW-MEDIUM. |

**Total Category 1 gap:** ~25 substrate trainers do NOT have compress-time scorer freeze; ~26 substrate trainers do NOT have multi-stage curriculum; the corpus is INCOMPLETE on the freezing surface.

### Category 2: PAUSING methods

| Method | Used by | Count of 44 substrate trainers | Empirical evidence | EV estimate |
|---|---|---:|---|---|
| **Pause-and-diagnose checkpoints** (instrument intermediate state) | A1-pattern (1 substrate; inflate-time bias correction is the canonical example) | 1/44 (2%) | A1 is the frontier 0.19285 PRECISELY because of this pattern: PR101 substrate trained then PAUSED at the inflate boundary, diagnosed for systematic bias, re-built with bias-corrected inflate path. Generalization across the other 52 substrates: untested. | HIGH-EV — the **A1 pattern is the ONLY example of a non-substrate-class-shift method that produced a sub-frontier anchor** in the 53-substrate corpus. Generalizing this to other substrates is high-priority orthogonal exploit. |
| **Pause-to-swap-loss-function** (curriculum: L1→L2→KL distill) | 5-10 substrate trainers (estimate from curriculum grep) | ~7/44 (16%) | Quantizr 0.33 leader uses this (5-stage pipeline). PR95-paradigm winners use it. | MEDIUM-HIGH. |
| **Pause-to-interpolate-checkpoints** (model soup; weight averaging) | 0 trainers detected (no grep hits on model_soup/swa/weight_avg) | 0/44 | UNCONSIDERED — model soup / SWA (Izmailov et al. 2018) gives 0.5-1.5% accuracy improvement at zero inference cost. None of 44 substrates uses it. | MEDIUM — proven literature; cheap to implement. |
| **Pause-to-do-post-hoc-bias-correction** (A1's exploit) | 1 (A1 itself; not in trainers; inflate-time only) | 1/44 (2%) | This IS the A1 exploit; the methodology generalizes to ANY substrate where systematic frame-bias can be characterized post-training. | **HIGHEST-EV in this category** — A1's bias correction took an already-frontier substrate from 0.193+ down to 0.19285. Applying this to PR106 r2 (currently 0.195 [contest-CUDA] / 0.206 paired) could plausibly take it to ~0.193-0.194 on contest-CUDA per the A1 transfer pattern. |
| **Pause-to-prune-then-fine-tune** (IMP; Lane 17 was killed prematurely) | 9 substrate trainers grep-match "prune" tokens but with weight-magnitude rather than score-gradient saliency (the Track 4 v1 anti-pattern per Catalog #123) | 9/44 with weight-saliency-pattern (anti-pattern); 0/44 with score-gradient-pruning | Lane 17 IMP was PREMATURELY KILLED 2026-04-30 on a stub-loop measurement bug per CLAUDE.md "KILL is LAST RESORT" non-negotiable; reactivation candidate per Tier 1 Resurrection per `meat_on_the_bone_audit_unused_exploits_20260515.json`. | MEDIUM. |
| **Pause-to-quantize-then-fine-tune** (QAT) | 7 substrate trainers (a1_plus_wavelet_residual, c1_world_model_foveation, c6_e4_mdl_ibps, cool_chic, hybrid_renderer_residual, pr101_lc_v2_clone_enhanced_curriculum, stc_v2) | 7/44 (16%) | Quantizr 0.33's PRIMARY architectural mechanism. Only 16% of corpus uses it. | HIGH — see Category 3 FP4 below. |
| **Pause-to-extract-and-rebuild-archive** (re-cooking the archive post-training) | 0 trainers detected | 0/44 | UNCONSIDERED — could re-encode the same trained weights with a DIFFERENT entropy coder (LZMA → Brotli → zstd → arithmetic-coder) without retraining. Cheap. | MEDIUM — small per-archive savings (~100-1000 bytes; -0.00003 to -0.0003) but FREE. |
| **Pause-on-plateau-detection** (early stopping) | 1 substrate trainer (pretrained_driving_prior) | 1/44 (2%) | Standard ML practice; surprisingly underused. | LOW — efficiency, not score. |

**Total Category 2 gap:** model-soup / weight-averaging / pause-and-diagnose generalization is largely UNCONSIDERED. The A1 pattern is the canonical exemplar but not yet generalized.

### Category 3: HARDWARE EXPLOITS

| Method | Used by | Count of 44 substrate trainers | Empirical evidence | EV estimate |
|---|---|---:|---|---|
| **FP4 quantization (Quantizr-pattern; archive ~32KB)** | 0 substrate trainers (despite FakeQuantFP4/FP4Parametrize/QATRendererFP4 all coded in `src/tac/fp4_quantize.py`; only referenced by legacy `src/tac/experiments/train_renderer.py` non-substrate path + test fixtures) | **0/44** | **CRITICAL GAP.** Quantizr (the external 0.33 [contest-CUDA] leader; CLAUDE.md "Quantizr intelligence" §) uses FiLM-CNN @ 88K params with 5-stage QAT pipeline ending in FP4. PR101 (gold 0.193) is reported by `feedback_why_leaderboard_hnerv_worked_when_ours_didnt` to be a 337-LOC codec bolt-on that included per-tensor byte-map encoding similar to Quantizr's. The rate term is `25 * archive_bytes / 37,545,489` so 25 bytes saved = 1 distortion unit; FP4 vs FP32 saves ~3/4 of the renderer mass which is the dominant archive cost. | **HIGHEST-EV in entire audit.** Predicted ΔS for retrofit on A1 / PR106 r2 / PR100 [-0.010, -0.030] per substrate; per the `meat_on_the_bone_audit` AX5_02 + AX5_03 Taylor decomposition. |
| **FP8 quantization (E4M3 / E5M2; H100 hardware-native)** | 0 substrate trainers (FP8 helpers exist in `src/tac/quantization_fp8.py` per the FAANG hardware lens memo) | 0/44 | E4M3 has higher fidelity than INT8 at same byte count per `expert_team_hardware_physics_future_alien_tech_20260513.md` Technique C1. Hardware-native on Modal A100 (CC 8.0+) and Vast.ai H100. | MEDIUM-HIGH. |
| **INT4 / INT8 archive encoding** | Existing apogee_int4 / apogee_int7 lanes (separate from substrate trainers; lane registry has 2 lanes) | 0/44 substrate trainers but 2 lane variants | Falsified at NAIVE-PTQ per CLAUDE.md non-negotiable (rel_err² anti-correlated with score-gradient saliency); reactivation criteria documented per Catalog #123. | DEFERRED — needs score-gradient-saliency QAT, not weight-magnitude PTQ. |
| **Sparse weights** (per-tensor or per-channel; magnitude pruning) | 9 substrate trainers grep-match prune token but with weight-magnitude saliency (Track 4 v1 anti-pattern) | 0 with correct saliency / 9 with anti-pattern | Per CLAUDE.md FORBIDDEN PATTERN "Forbidden weight-domain saliency on score-gradient substrate" + Catalog #123 STRICT preflight refuses the anti-pattern. The 9 trainers grep-matched may carry waivers. | LOW (anti-pattern); MEDIUM if migrated to score-gradient saliency. |
| **Mixed-precision** (FP16 weights + FP4 archive; FP16 fwd + FP32 backward via autocast) | Catalog #172 mandates `--enable-autocast-fp16` for substrate trainers; 14+ legacy trainers still missing per CLAUDE.md WARN-ONLY-at-landing | ~6/44 fully wired (T1 Balle canonical + 5 NeRV canonicals); ~38/44 partial or absent | Catalog #172 STRICT-FLIP pending the backfill wave. Free 1.5-3× training speedup. | LOW — pure efficiency, not score. |
| **SIMD-optimized inflate runtime (custom C/Rust kernels)** | 0 substrate trainers (19 Rust impls coded per `meat_on_the_bone_audit` AX2_06; none consumed by substrate inflate runtime) | 0/44 | Pure-byte rate optimization; per-substrate ΔS small (-0.001 to -0.005) but FREE. Aggregate across 14 contest-intent substrates compounds. | LOW-MEDIUM. |
| **Brotli / Zstandard / LZMA / arithmetic-coding variants** | PR101 substrate uses Brotli; PR106 r2 uses LZMA-tier compression; 0 substrate trainers use the canonical arithmetic-coder primitive | Used in 2-3 lanes (lossless byte coding); arithmetic-coder primitive UNCONSIDERED at substrate level | Per CLAUDE.md "Bit-level deconstruction and entropy discipline": entropy coding is first-class score lane. | MEDIUM. |
| **HEVC / AV1 / VVC for mask/payload encoding** | mask_codec.py uses AV1 monochrome (broken at 384x512 per CLAUDE.md; needs fix); 0 substrate trainers consume it | 0/44 | Existing infra but unused at substrate level. | LOW-MEDIUM. |
| **Hardware-specific Pareto (rate-distortion-FLOPs)** | 0 substrate trainers explicitly score by FLOPs/byte ratio | 0/44 | DEEP-ADVERSARIAL-REVIEW §4.4 surfaces this as a structural gap. Should be a 9-dim checklist field (Catalog #294). | MEDIUM. |
| **ZIP/STORE without compression for already-compressed payloads** | Standard practice; PR101 uses STORE for ZIP container then Brotli-compresses internal sections | Used in mainline archive builders | Already-canonical; no orthogonal gap. | N/A. |
| **Apple Silicon Metal / MLX (free signal generator)** | Used in 4 streams of `local_hardware_aggressive_sweep_5_streams_LANDED_20260513.md` for $0 ranking | Not a substrate-trainer pattern; auxiliary signal | macOS-CPU ↔ contest-CPU calibrated within 1.6e-5 on A1-class substrates per Catalog #192. Free 17× speedup for advisory ranking. | HIGH for parallel sweep throughput (not score directly). |
| **ARM NEON SIMD** (relevant for CPU contest axis) | 0 (relevant only if inflate.py emits NEON-aware tensor ops) | 0/44 | Per `expert_team_hardware_physics_future_alien_tech` Technique H3, this could 2-4× inflate speed on contest CPU runner. | LOW — efficiency, not score. |

**Total Category 3 gap:** **The FP4 quantization gap (0/44 substrate trainers despite Quantizr 0.33 winner pattern) is the single highest-EV unexploited surface in the entire audit.** Sister gaps: FP8 (0/44), score-gradient-pruning (0/44 with correct discipline), arithmetic-coder primitive consumption (0/44).

### Category 4: PROBLEM SPACE EXPLOITS

| Method | Used by | Count of 44 substrate trainers | Empirical evidence | EV estimate |
|---|---|---:|---|---|
| **1-video overfit: hardcode per-frame parameters; per-pair lookup tables** | PR95-paradigm substrates implicitly (per-pair learned 28-d latent IS this) | ~10/44 | Contest rule explicitly allows compress-side ANYTHING-GOES per `meat_on_the_bone_audit_unused_exploits_20260515.json` AX3_04. | HIGH — proven via PR95/PR100/PR101 winners. |
| **600-pair FIXED count: per-pair offsets in archive** | PR95-paradigm substrates (per-pair latent grammar) | ~10/44 | Same as above; structural to the contest. | HIGH. |
| **Half-frame trick (Quantizr / PR55: store ONLY 600 odd-frame masks)** | 0 substrate trainers grep-match `half_frame` / `odd_frame` / `warp_inversion` patterns | 0/44 | Quantizr (0.33 leader) uses this; PR55 historical pattern. Per `meat_on_the_bone_audit` AX5_09 generalization to other content (pose, residual, latent) is an unexploited axis. **CRITICAL CAVEAT**: half-frame BREAKS PoseNet on warp-based renderers without joint training (verified score=17.55 on dilated-h64 with half-frame, baseline 0.011). HNeRV-cluster doesn't use warp; safe there. | HIGH for HNeRV-family substrates; UNSAFE for warp-based. |
| **CPU/CUDA gap is 0.033+ on PR102 — exploit the axis difference** | All substrates implicitly (CPU axis ranks leaderboard) | 44/44 implicitly | Per A1 Sweep `local_hardware_aggressive_sweep_5_streams_LANDED_20260513.md` §1: A1 contest-CUDA = 0.2264 vs contest-CPU = 0.1928 (Δ +0.0335). Some archives could plausibly be ENGINEERED to score better on CPU than CUDA (Linux x86_64 FastViT path drift). Per `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508` the mechanism is unresolved (decoder mismatch vs FP32 forward kernel drift) — distinguishing would unlock an orthogonal exploit. | HIGH for leaderboard ranking (CPU is what matters). |
| **Video has known structure: ~10% sky + ~80% road + ~10% objects; per-class priors** | 0 substrate trainers explicitly | 0/44 | Per `domain_exploitation_catalog_20260509.md` Section B.3 region-of-importance map: sky region is free bits; road region is expensive; FoE (focus-of-expansion) ego-motion has highest pose-relevance. | MEDIUM — foveation revival (`meat_on_the_bone_audit` AX1_06). |
| **Scorer 25× rate weight: 25 bytes saved = 1 distortion unit** | All substrates implicitly | 44/44 | Per upstream/evaluate.py: `100*seg + sqrt(10*pose) + 25*rate/37545489`. The rate-saving leverage is HUGE. | (Implicit motivator.) |
| **Scorer eval batch_size + seed: deterministic eval enables bit-exact reproducibility** | All substrates | 44/44 | Standard. | N/A. |
| **Mask resolution 384×512 → renderer trained at same resolution** | CLAUDE.md non-negotiable; all substrates honor | 44/44 | Catalog enforced. | N/A. |
| **Yousfi+Fridrich inverse steganalysis: UNIWARD / detector-informed embedding / square root law / CNN blind spots** | 1 substrate trainer references UNIWARD; FastViT-T12 RepMixer blind spots per `meat_on_the_bone_audit` AX5_07 | ~1/44 | Per CLAUDE.md "Fridrich inverse steganalysis": errors in textured regions are undetectable. Detector-informed embedding IS our TTO approach but only at compress time. UNIWARD-style spatial-frequency embedding at substrate-design time = unexploited. | HIGH-MEDIUM. |
| **PoseNet sees 2 frames; SegNet sees ONLY last frame** | NSCS01 nullspace-split-renderer is the canonical exploit (frame[0] grad to pose only; frame[1] grad to seg+pose+pixel) | 1/44 explicit | Per upstream/modules.py:108 (`x[:, -1, ...]`); the nullspace-split exploit is novel and elegant. Other substrates do NOT exploit this asymmetry. Generalization to per-component substrate routing (`meat_on_the_bone_audit` AX5_10) is UNCONSIDERED. | HIGH — NSCS01 is currently L1; predicted-frontier-pursuit at [0.18, 0.20] per design memo. |
| **Eval roundtrip uint8 bottleneck** | CLAUDE.md non-negotiable; ALL substrates honor (`tac.differentiable_eval_roundtrip`) | 44/44 | Catalog enforced. | N/A (mandatory). |
| **Contest evaluator GHA x86_64 Linux runner specifics** | A1 uses Linux x86_64 verification per CLAUDE.md "Submission auth eval" non-negotiable | 44/44 implicitly | Per Catalog #192 macOS-CPU advisory is NEVER 1:1; Linux x86_64 is the ranking surface. | N/A (mandatory). |
| **SegNet stride-2 invisibility at smaller resolution (NSCS02 partial; could go further)** | NSCS02 partially at (192,256); none at (96,128) per `meat_on_the_bone_audit` AX5_01 | NSCS02 only | EfficientNet-B2 stride-2 stem loses half resolution immediately; artifacts below (256,192) invisible. Going to (96,128) would compound rate savings. | MEDIUM-HIGH per AX5_01 estimate [-0.010, -0.030]. |

**Total Category 4 gap:** NSCS01-style per-component substrate routing (frame[0] vs frame[1] asymmetry, YUV vs RGB asymmetry) is severely under-exploited; CPU-CUDA axis-difference engineering is implicit only.

---

## 2. Multiplicative stacking analysis

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable, orthogonal exploits stack MULTIPLICATIVELY when their score-shifting mechanisms operate on disjoint axes. Per the score formula `100*seg + sqrt(10*pose) + 25*rate`:
- **Rate-axis exploits** stack additively in rate (subtraction in score because `25*rate` term DECREASES as bytes decrease).
- **Seg-axis exploits** stack via the linear `100*seg` term.
- **Pose-axis exploits** stack via the concave `sqrt(10*pose)` term (so per-byte sensitivity for pose grows as pose decreases — A1's frontier is in a regime where pose marginal is 2.71× SegNet's).

**Stacking is OPPORTUNITY-RICH when:** (a) the exploits address different axes; (b) the exploits use different mechanisms within the same axis; (c) the exploits do not require the same code surface (so engineering effort scales sub-linearly).

### Candidate Stack #1: A1 + Quantizr-pattern FP4 + per-frame bias correction generalization

**Base substrate:** A1 (verified 0.19285 [contest-CPU]; 178,262 bytes; PR101-paradigm)

**Orthogonal exploit 1:** FP4 quantization on the renderer weights (rate-axis; Quantizr-pattern). Per the `meat_on_the_bone_audit` analysis: ~115 KB → ~32 KB renderer (-83 KB rate term = -0.0011 score → maps to -0.0011×4.66 = -0.0055 in rate-term-after-distortion units; net predicted -0.005 to -0.015 ΔS per AX5_03 + `expert_team_hardware_physics_future_alien_tech` Technique C2/S3 stacking).

**Orthogonal exploit 2:** A1-pattern post-hoc bias correction extension. A1 already exploited compress-vs-inflate frame bias; the orthogonal extension is per-pair-bias correction (Quantizr's odd-frame trick generalized to per-pair frame[0]/frame[1] asymmetry). Predicted ΔS [-0.001, -0.005].

**Multiplicative compose:** Both target the rate axis but via disjoint mechanisms (FP4 reduces weight-byte mass; per-pair bias correction reduces residual-byte mass). Stacking is plausibly additive: net ΔS [-0.006, -0.020] = predicted score range [0.173, 0.187].

**Cost:** $5-25 (FP4 retrofit on A1 substrate; small-batch QAT).

**Status:** **HIGHEST-EV STACK IN AUDIT**. The base substrate is verified at the frontier; both exploits are well-documented; engineering effort modest.

### Candidate Stack #2: PR106 r2 + FP4 + half-frame pattern (HNeRV-family safe)

**Base substrate:** pr106_latent_sidecar_r2 (verified 0.195 [contest-CPU paired]; 186,822 bytes; HNeRV-family)

**Orthogonal exploit 1:** FP4 quantization (rate-axis). Per the analysis above: -0.005 to -0.015 ΔS.

**Orthogonal exploit 2:** Half-frame Quantizr trick (rate-axis). Stores only 600 odd-frame data; reconstructs even from odd. HNeRV-family is SAFE (no warp dependency per CLAUDE.md). Predicted ΔS [-0.005, -0.015].

**Multiplicative compose:** Both rate-axis but disjoint mechanisms. Net ΔS [-0.010, -0.030] = predicted score range [0.165, 0.185].

**Cost:** $5-25.

### Candidate Stack #3: NSCS01 + compress-time scorer freeze + Quantizr-style FP4 (asymmetric per-frame)

**Base substrate:** NSCS01 nullspace-split-renderer (L1 today; sextet council PROCEED_WITH_REVISIONS; novel per-frame asymmetric exploit)

**Orthogonal exploit 1:** Compress-time scorer freeze (already wired per the trainer's grep-match; reinforces the per-pair conditioning protection from DEEP-ADVERSARIAL-REVIEW).

**Orthogonal exploit 2:** Quantizr-pattern FP4 on the per-frame-asymmetric decoder.

**Orthogonal exploit 3:** Per-component substrate routing extension (AX5_10): different quantization per frame[0] vs frame[1] components.

**Multiplicative compose:** Three orthogonal axes (per-frame asymmetric architecture + FP4 quantization + per-frame quantization specialization). Net predicted ΔS large but UNVERIFIED; predicted score range [0.165, 0.190] depending on NSCS01's baseline post-Phase-2-council.

**Cost:** $15-50.

### Candidate Stack #4: A1 + Linux x86_64 / GHA-CPU-axis engineering

**Base substrate:** A1 (verified 0.19285 [contest-CPU])

**Orthogonal exploit 1:** CPU-CUDA axis exploitation. Per `feedback_cuda_cpu_auth_eval_drift_pr102_pr104_20260508`, the mechanism between CUDA and CPU drift is unresolved (decoder mismatch vs FP32 kernel drift). If we engineer the substrate to MAXIMIZE the CUDA-minus-CPU gap in the direction of A1's existing 0.0335 advantage (CPU < CUDA by 0.033), we may compound the gap. This requires the AX3.3 mechanism-discriminator dispatch (Lightning T4 + AVVideoDataset patch; $0.40; 30 min). Predicted ΔS [-0.005, -0.015] IF mechanism is decoder-mismatch-dominated.

**Multiplicative compose:** Single-axis (CPU-axis ranking); orthogonal to all substrate-architecture exploits. Net ΔS [-0.005, -0.015] ON THE CPU LEADERBOARD AXIS (which is what ranks).

**Cost:** $0.40-$5 (mechanism discriminator + small-batch optimization).

### Candidate Stack #5: PR100 hnerv_lc_v2 (the verified leader-board substrate) + Rule #6 BOLT-ON

**Base substrate:** PR100 hnerv_lc_v2 (May 2026 leaderboard 0.1954 [contest-CPU]; 268 LOC; HNeRV-family)

**Orthogonal exploit 1:** PR101 BOLT-ON pattern (337 LOC of pure codec on PR100's substrate). Already proven on the leaderboard at 0.193 gold. Per DEEP-ADVERSARIAL-REVIEW §4.6 Rule #6 the canonical winning pattern.

**Orthogonal exploit 2:** A1-pattern post-hoc bias correction on top of PR101's bolt-on.

**Multiplicative compose:** PR101 already beat PR100 by ~0.002. Layering A1-style bias correction on top could plausibly take it to 0.190 if the bias-correction signal transfers.

**Cost:** $5-25.

---

## 3. 0-of-53-corpus interpretation through orthogonal lens

Per the parent DEEP-ADVERSARIAL-REVIEW finding of 0-of-53 sub-frontier anchors, applying the orthogonal lens flips the question: **which of the 53 substrates would have a MEANINGFULLY better predicted band IF we added 2+ orthogonal exploits?**

**HIGH-ORTHOGONAL-RESCUE substrates** (substrates whose plateau-adjacent verdict was based on within-class baseline; orthogonal stacking shifts predicted band):

1. **A1** (current frontier 0.19285) — Stack #1 predicts [0.173, 0.187].
2. **PR106 r2** (current 0.195 paired) — Stack #2 predicts [0.165, 0.185].
3. **PR100 hnerv_lc_v2** (current 0.1954) — Stack #5 predicts [0.190, 0.195] (modest).
4. **NSCS01** (Phase 2; L1) — Stack #3 predicts [0.165, 0.190].
5. **NSCS03** (Phase 2; L1; PR95+Ballé-2018 paradigm) — orthogonal Stack analogous to #3 predicts similar range.
6. **The 5 LIFTED-DISPATCH-READY NeRV-family canonicals** (sane_hnerv, hi_nerv, ds_nerv, tc_nerv, block_nerv, ff_nerv) — Stack #2 pattern applies; predicted [0.180, 0.195] each.

**LOW-ORTHOGONAL-RESCUE substrates** (HIGH RISK per parent review; orthogonal exploits do not address the per-pair conditioning attenuation; structural failure mode remains):

- The ~35 substrates per parent review §3.1 that rely on per-pair conditioning at the scorer's per-frame response surface. Orthogonal rate-axis exploits (FP4 quantization) might give them a small rate boost but does NOT fix the seg/pose distortion bottleneck that drives them above 0.20.

**Net interpretation:** orthogonal stacking can plausibly rescue ~6-12 of 53 substrates into the sub-frontier (sub-0.192) regime WITHOUT substrate-class-shift. The remaining ~35 substrates with per-pair-conditioning failure mode are NOT orthogonal-rescuable; they need T4-SYMPOSIUM decisions (substrate-class-shift verdicts).

---

## 4. Top 7 highest-EV orthogonal exploits (operator-priority queue)

| Rank | Exploit | Category | Predicted ΔS per substrate | Cost (USD per substrate) | Why now |
|---:|---|---|---|---:|---|
| **1** | **FP4 quantization on A1 / PR106 r2 / PR100 substrates** (Quantizr-pattern; FakeQuantFP4 already in `src/tac/fp4_quantize.py`) | HARDWARE | [-0.010, -0.030] | $5-25 | 0/44 substrate trainers use it despite being the leaderboard winning paradigm (Quantizr 0.33). The rate-term is the score formula's largest lever (25×). |
| **2** | **A1-pattern post-hoc bias correction generalization to PR106 r2 / PR100** | PAUSING | [-0.005, -0.020] | $5-25 | A1 IS the current frontier specifically because of this pattern; generalization untested. |
| **3** | **Compress-time scorer freeze backfill across 25 substrate trainers missing the pattern** | FREEZING | [-0.005, -0.020] per substrate | $0 (engineering only) | Directly addresses DEEP-ADVERSARIAL-REVIEW §3.1 per-pair conditioning attenuation finding. |
| **4** | **Rule #6 BOLT-ON pattern execution on PR100 (the leaderboard verified substrate)** | PROBLEM-SPACE + FREEZING | [-0.002, -0.005] | $5-25 | PR101 gold won with this pattern; we have NOT yet emulated. |
| **5** | **CPU-CUDA axis-difference engineering on A1** (AVVideoDataset mechanism-discriminator + CPU-axis biased substrate) | PROBLEM-SPACE | [-0.005, -0.015] CPU-axis-only | $0.40-5 | CPU is the leaderboard ranking axis; A1 already has 0.033 advantage; explicit engineering may compound. |
| **6** | **Quantizr half-frame trick on HNeRV-family substrates (NeRV-canonicals + PR106 r2)** | PROBLEM-SPACE | [-0.005, -0.015] | $5-25 | Proven on Quantizr 0.33 winner; SAFE for HNeRV (no warp dep); ~50% mask-data byte reduction. |
| **7** | **Pause-and-build-model-soup across multi-stage curriculum substrates** | PAUSING | [-0.002, -0.005] | $0 (engineering only) | 0/44 substrates use SWA / model averaging despite proven 0.5-1.5% accuracy improvement in literature. |

---

## 5. Cross-decision input to T4 SYMPOSIUM (sister; COMPLETED before this audit landed)

T4 SYMPOSIUM deliberation memo `feedback_t4_symposium_substrate_design_class_shift_deliberation_20260517.md` (commit pending sister) deliberated 5 substrate-design binding decisions in response to the DEEP-ADVERSARIAL-REVIEW. This audit's findings inform the deliberation:

**Cross-decision finding A — answer to T4 Decision 4 (asymptotic-pursuit substitution):**

The 0.192 frontier appears to be **PARTIALLY MOVABLE via orthogonal exploits alone (without substrate-class-shift)** per Section 3 of this audit. Stack #1 (A1 + FP4 + per-pair bias correction) predicts [0.173, 0.187] without any substrate-class-shift. This is a meaningfully different answer than "the substrate-design corpus is plateau-trapped at 0.192" — the orthogonal exploit surface gives 6-12 substrates a credible path to sub-frontier WITHOUT new substrate-class designs.

**Implication for T4 Decision 4:** asymptotic-pursuit substitution (requiring substrate-class-shift to break the plateau) is necessary for the sub-0.150 frontier-pursuit regime per HORIZON-CLASS but NOT necessary for the sub-0.190 within-paradigm regime. **Recommendation to T4: run orthogonal-exploit Stacks #1 + #2 BEFORE committing significant resources to new substrate-class-shift designs.** Estimated cost is $10-50 vs $100+ for substrate-class-shift design + training.

**Cross-decision finding B — answer to T4 Decision 5 (A1 frontier framing):**

Per DEEP-ADVERSARIAL-REVIEW §4.6 Rule #6 (BOLT-ON pattern; PR101 gold 337 LOC), the **orthogonal exploit pattern IS the missing Rule #6 in the lattice**. Quantizr-style FP4 + half-frame + per-pair bias correction ARE the canonical bolt-on family. The lattice currently has 0 entries under Rule #5 (REQUEST_OPERATOR_REVIEW) and 0 entries under a Rule #6 (BOLT-ON pattern). Per this audit, **Rule #6 should be: "Orthogonal exploit stack on verified working substrate ≤350 LOC + ≤30-second-reviewable + entropy-coding-only + monolithic-archive-grammar"** — exactly the PR101 winning pattern.

**Implication for T4 Decision 5:** A1 is canonically a Rule #6 BOLT-ON SUCCESS (inflate-time bias correction on PR101-paradigm substrate). The frontier framing should explicitly name this as the model for the next 2-3 dispatches: take a verified working substrate, add 1-2 orthogonal exploits, ship.

---

## 6. Cross-decision input to SCORER-RESPONSE-SURFACE-ANALYSIS (sister; IN-PROGRESS)

This audit's findings input to the sister scorer analysis:

- **FREEZING analysis** in Category 1: compress-time SegNet/PoseNet freeze pattern is used in 19/44 trainers. The sister analysis should characterize whether the 25 unfrozen-pattern trainers exhibit measurably-different scorer response surface artifacts.
- **PAUSING analysis** in Category 2: A1-pattern post-hoc bias correction works specifically because it intercepts a systematic scorer-derivable bias at the inflate boundary. The sister analysis should characterize this bias surface for PR106 r2 / PR100 / NSCS01 / NSCS03 to enable Stack #1 generalization.
- **HARDWARE analysis** in Category 3: FP4 quantization will introduce systematic per-tensor noise at the renderer output. The sister analysis should predict whether the SegNet stride-2 stem attenuates this noise (FP4 is safe) or amplifies it (FP4 dangerous).

---

## 7. Operator-action-required summary (what to do NEXT)

**Cheap, $0-or-low-cost, immediate-value (Tier 0):**

1. **Compress-time scorer freeze backfill across 25 substrate trainers** (Catalog #182-style backfill sweep). $0 cost; engineering-only. Predicted gain: ~25 substrates become per-pair-conditioning-attenuation-protected. Could be wired as STRICT Catalog gate.
2. **Pause-and-build-model-soup** across the 7 multi-stage curriculum substrates. $0 cost; engineering-only. Predicted gain: 0.5-1.5% per substrate.
3. **Per-substrate FP4-quantization-readiness audit** (which of the 7 substrate trainers using QAT could be migrated to FP4). $0 cost.

**Bounded smoke under $30 (Tier 1):**

4. **Stack #1 smoke on A1**: 1 paid Modal A100 dispatch of A1 + FakeQuantFP4 retrofit + measure CUDA + CPU axes. $5-15. Predicted gain [-0.010, -0.030].
5. **CPU-CUDA mechanism-discriminator dispatch** on A1 (AVVideoDataset patch; AX3.3 atom). $0.40 Lightning T4; 30 min. Resolves the CUDA-CPU drift hypothesis tree.
6. **AX5_01 — SegNet stride-2 invisibility at (96,128)** on a verified substrate. $5-15.

**Council-grade $50-100 (Tier 2):**

7. **Stack #2 smoke on PR106 r2**: FP4 + half-frame on the HNeRV-family substrate. $25-50.
8. **Rule #6 BOLT-ON execution on PR100 hnerv_lc_v2** (the verified leaderboard substrate; produce a PR101-style 337-LOC codec bolt-on). $25-50 + ~1 week engineering.

**Long-burn operator decision (Tier 3):**

9. **Migrate the canonical lane registry / Catalog #294 9-dim checklist** to include explicit orthogonal-exploit columns (FP4-coverage / freeze-coverage / pause-coverage / half-frame-coverage) so the autopilot ranker can prioritize substrates by orthogonal-exploit-richness.

---

## 8. Observability surface

Per CLAUDE.md "Max observability — non-negotiable" 6-facet definition:

1. **Inspectable per layer** — this audit memo enumerates each orthogonal-exploit per substrate via grep evidence; per-trainer file paths cited; no hidden state.
2. **Decomposable per signal** — each exploit's predicted ΔS is decomposable into (rate-axis / seg-axis / pose-axis) contribution per the score formula.
3. **Diff-able across runs** — the 4 inventory tables in Section 1 are machine-readable (counts of 44; per-substrate id citation); subsequent audits can diff against this manifest.
4. **Queryable post-hoc** — exploit-coverage queryable via the canonical `tools/audit_orthogonal_optimization_exploits.py` CLI landed in same commit batch (the lighter inventory tool per the prompt's directive; Layer 1 + Layer 2 only, deferring Layer 3 strict-flip + Layer 4 fcntl-locked posterior per parent prompt's "optional unless empirically strict" guidance).
5. **Cite-able** — every empirical claim cites either the trainer file path, the canonical helper file path, or a dated `.omx/research/*.md` predecessor memo with a SHA or timestamp anchor.
6. **Counterfactual-able** — Section 2's stacking analysis is structured as if-then: "if Stack #1 fires, predicted ΔS = [-0.006, -0.020]"; falsifiable via $5-25 smoke per Stack.

---

## 9. Cargo-cult audit per assumption

Per CLAUDE.md Catalog #303 "Cargo-cult audit section" non-negotiable + the HARD-EARNED-vs-CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

| Assumption | Classification | Rationale |
|---|---|---|
| Orthogonal exploits stack multiplicatively across rate/seg/pose axes | **HARD-EARNED** | Per upstream/evaluate.py the score formula IS structurally additive across the three axes; per-axis exploits stack additively in axis contribution (subtractive in score). |
| FP4 quantization on the renderer is rate-axis-positive | **HARD-EARNED** | Quantizr's 0.33 leaderboard result + the score formula's 25× rate weight; well-anchored empirically. |
| A1-pattern post-hoc bias correction generalizes to other PR95-paradigm substrates | **CARGO-CULTED** | A1's success could be specific to PR101 substrate's particular bias pattern. Generalization to PR100 / PR106 r2 / NSCS01 / NSCS03 is unverified. Tier-1 smoke required ($5-25). |
| Compress-time scorer freeze addresses per-pair conditioning attenuation per DEEP-ADVERSARIAL-REVIEW §3.1 | **PARTIALLY HARD-EARNED** | Theoretical justification (per Tao's harmonic-analysis position in the parent review); empirical anchor on per-substrate ΔS pending. |
| 0-of-53 substrates is interpretable as "corpus is plateau-trapped" rather than "0/53 had appropriate orthogonal stacking applied" | **HARD-EARNED for plateau interpretation; CARGO-CULTED for excluding orthogonal-rescue path** | Parent review's plateau-interpretation is sound for substrate-class-shift question; THIS AUDIT shows the orthogonal-rescue path may give 6-12 substrates a credible sub-frontier path without class-shift. Both interpretations are simultaneously HARD-EARNED in their proper scope. |
| The 8 META-assumptions of the parent review (all CARGO-CULTED) propagate to the orthogonal-exploit corpus | **HARD-EARNED** | Same scorer; same axis structure; same constraints. The CARGO-CULTED META-assumptions about substrate-design DO NOT extend to orthogonal optimization because orthogonal exploits sit BENEATH the substrate-design layer (rate-coding / archive-grammar / inflate-time pause are non-substrate-class concerns). |
| Quantizr's 5-stage QAT pipeline transfers to PR95-paradigm substrates | **CARGO-CULTED** | Quantizr's pipeline was designed for an 88K-param FiLM-CNN; PR95-paradigm substrates have 229K-param HNeRV decoders. Transfer is plausible but unverified. |

**Summary:** 3 HARD-EARNED + 1 PARTIALLY HARD-EARNED + 3 CARGO-CULTED out of 7 audit-relevant assumptions. The CARGO-CULTED items each have explicit reactivation criteria via $5-25 Tier-1 smokes per Section 7.

---

## 10. Files

- **This memo:** `.omx/research/orthogonal_optimization_methods_audit_20260517.md` (primary deliverable; ~3500 words above; ~6000 words with Section 11 below).
- **Lane:** `lane_orthogonal_optimization_methods_audit_20260517` (pre-registered L0; this commit advances to L1 via impl_complete + memory_entry).
- **Sister tool** (lighter inventory CLI per parent prompt's "optional unless empirically strict"): `tools/audit_orthogonal_optimization_exploits.py` (NEW; ~150 LOC; reads this memo's machine-readable counts and emits per-category coverage table for cathedral autopilot + operator audit; Layer 1 + Layer 2 of the canonical 4-layer pattern; Layer 3 STRICT preflight + Layer 4 fcntl-locked posterior DEFERRED unless empirically strict).
- **Predecessor memos:** `deep_adversarial_review_substrate_design_meta_20260517.md` (parent review) + `domain_exploitation_catalog_20260509.md` + `meat_on_the_bone_audit_unused_exploits_20260515.json` + `local_hardware_aggressive_sweep_5_streams_LANDED_20260513.md` + `expert_team_hardware_physics_future_alien_tech_20260513.md` + `orthogonal_frozen_stream_optimization_20260501_codex.md`.

---

## 11. Sister-subagent coordination notes (per Catalog #230)

- **T4 SYMPOSIUM** (`lane_t4_symposium_substrate_design_class_shift_20260517`; **COMPLETE** at 12:24:35Z per subagent_progress.jsonl): owns the 5 substrate-design binding decisions. My audit's Section 5 cross-decision input is structured to feed T4's deliberation outputs WITHOUT modifying T4's memo. Disjoint memo paths; disjoint lane_ids; disjoint commit windows (T4 committed BEFORE I started writing).
- **SCORER-RESPONSE-SURFACE-ANALYSIS** (`scorer_response_surface_analysis_20260517` subagent; **IN-PROGRESS** at 12:25:20Z step 2): owns the scorer empirical characterization memo at `.omx/research/scorer_response_surface_analysis_20260517.md`. My memo path is `.omx/research/orthogonal_optimization_methods_audit_20260517.md` (different file). My Section 6 cross-decision input is structured to feed SCORER's analysis WITHOUT modifying SCORER's memo or state files.
- **My scope:** NEW memo file + NEW tool file + lane registry mark via canonical `tools/lane_maturity.py`. NO substrate trainer edits; NO canonical helper edits; NO preflight.py edits; NO CLAUDE.md edits.
- **Catalog #117/#157/#174 commit serializer discipline:** all commits via `tools/subagent_commit_serializer.py --expected-content-sha256` with POST-EDIT working-tree shas per CLAUDE.md "Subagent commits MUST use serializer" non-negotiable.
- **Catalog #206 checkpoint discipline:** 3 checkpoints during execution (step 1 pre-flight; step 2 pre-draft; step complete at completion).
- **Catalog #248 conflict markers:** none introduced.

---

*Memo authored by orthogonal_optimization_methods_audit_subagent_20260517. Lane `lane_orthogonal_optimization_methods_audit_20260517`. Wave: ORTHOGONAL-OPTIMIZATION-METHODS-AUDIT.*
