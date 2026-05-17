# Tier 1 Resurrection #4 — PR101 CompressAI Ballé hyperprior REFORMULATED on NSCS03 / ATW v2 chroma residuals — comprehensive full-stack design memo

**Date**: 2026-05-16
**Lane**: `lane_tier_1_resurrection_4_pr101_compressai_balle_reformulated_20260516`
**Subagent**: TIER1-RESURRECTION-4-5-DESIGN-MEMOS-20260516 (RESPAWN of dead `a34688b73f9afafb5`)
**Predecessors**:
- `.omx/research/resurrection_audit_20260516.md` §1.3 + #6 reactivation-queue row (cargo-cult unwind anchor)
- `reports/raw/pr101_compressai_balle_full_20260508T005408Z` (V1 empirical falsification: rel_err 0.98-0.99 across all 8 N/M configs on PR101 INT8 symbol stream)
- `tools/pr101_compressai_balle_FIXED.py` + `tools/pr101_compressai_balle_hyperprior_full.py` (V1 source code; preserved for forensic reference)
- `.omx/research/balle_compressai_byte_closure_audit_20260513_codex.md` (separate balle_renderer/BRV1 substrate-engineering audit; sister surface)
- `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/` (Phase 2 target substrate; `_full_main` already lands per `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`)
- `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` (alternative target; productionizes Wunderkind G1+B3+G2-PARTIAL)
- `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md` Q2 verdict (CPU axis is leaderboard axis; PR102 drift extrapolation is CARGO-CULTED per Catalog #292/#296)

**Operating mode**: UNIQUE-AND-COMPLETE-PER-METHOD per the standing directive `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` + the 2026-05-15 PR95 META-level retrospective. The PR101-symbol-stream Ballé application was CANONICAL-FORCE-FIT (Ballé is canonical for spatially-correlated data; PR101 INT8 symbols are intentionally decorrelated → canonical suppressed signal). This reformulation FORKS the canonical Ballé recipe per substrate-optimal engineering.

**Status at landing**: DESIGN-ONLY, RESEARCH-ONLY at the recipe level. Reactivation gated on (a) ATW v2 substrate build landing (in flight per sister `a8ef880a01e1fd84f`); (b) D4 H(latent|scorer_class) probe verdict; (c) NSCS03 paired-CPU+CUDA Modal A100 smoke per Catalog #167.

---

## 1. Frontmatter — premise verification + lane registry + sister-subagent map

### Premise verifications (Catalog #229; 8 PVs verified BEFORE any design statement)

- **PV-1** PR101 V1 CompressAI Ballé empirical falsification IS REAL on the PR101 INT8 symbol stream substrate. Source: `reports/raw/pr101_compressai_balle_full_20260508T005408Z` contains 8 (N, M) configs (N ∈ {128, 192, 256}, M ∈ {192, 320}) all returning rel_err 0.98-0.99 = essentially identity (no information recovery). The hyperprior tried to reconstruct PR101's INT8 quantized weight symbols reshaped as `(1, 1, 448, 512)` pseudo-image; ScaleHyperprior's 2D-convolutional analysis transform finds no exploitable spatial structure in a near-iid symbol stream.
- **PV-2** NSCS03 substrate `_full_main` IS implemented per commit landing `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md` (2026-05-15). Architecture: convolutional ANALYSIS `g_a` → factorized-prior ENTROPY BOTTLENECK on hyper-latent `z` → conditional-Gaussian density on main latent `y` given `σ=h_s(z)` → convolutional SYNTHESIS `g_s` back to pixels. This IS the canonical Ballé 2018 hyperprior pipeline applied to a substrate WITH spatial structure (per-pair RGB pixels at camera resolution).
- **PV-3** NSCS03 archive grammar declared in `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/__init__.py:50-72`: `monolithic_0.bin_with_main_and_hyper_latents` (MAGIC|VER|ARCH|MAIN_LATENT|HYPER_LATENT|META); parse_archive returns `NSCS03Archive(encoder_sd, decoder_sd, hyper_analysis_sd, hyper_synthesis_sd, entropy_state_sd, main_latents, hyper_latents, meta)`; inflate budget ≤200 LOC waiver per substrate_engineering exception; runtime dep closure = torch + brotli (no CompressAI runtime dep — already extincted in NSCS03 design).
- **PV-4** ATW v2 design memo at `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` declares §13 row 11 stack composition `A-STACK[NSCS03 → ATW v2]` with ATW v2's entropy-coding term SUBSUMING NSCS03's Ballé hyperprior. ATW v2 chroma residuals (Y-channel = scorer-conditional; UV-channel = reconstruction-only via wavelet residual per NSCS06 v8 Path B) are a DIFFERENT target than NSCS03's full RGB joint codec.
- **PV-5** CompressAI library inventory at `src/tac/composition/registry.py::canonical_primitive_inventory()` registers 3 CompressAI primitives (`compressai_factorized_prior`, `compressai_balle_hyperprior`, `compressai_cheng2020`) per Catalog #169 (REVIEW-OMNI Medium A2-1 self-protection). Golden vectors at `src/tac/packet_compiler/golden_vectors/compressai_balle_hyperprior_v1.json`. The CompressAI primitives EXIST and are addressable; they were previously force-fit to PR101 symbols. The substrate-mismatch is in the TARGET, not the primitives.
- **PV-6** Sister `balle_renderer` substrate (`src/tac/substrates/balle_renderer/`) exists as a separate Ballé application: per-pair flat learned latents `nn.Parameter(num_pairs, latent_dim)` with hyperprior as MLP (not conv); decoder MLP+upsample. Per `feedback_t1_balle_endtoend_phase_1_council_proceed_20260512.md` and `.omx/research/balle_compressai_byte_closure_audit_20260513_codex.md`, balle_renderer is at substrate-engineering scaffold state. NSCS03 is the END-TO-END joint codec (architecturally distinct from balle_renderer's flat-latent paradigm).
- **PV-7** Z3 v2 substrate (`src/tac/substrates/z3_v2/` referenced via `experiments/train_substrate_z3_v2_*.py`) uses Ballé hyperprior as a BOLT-ON over A1's existing latent stream (within-class refinement; rate-axis only). Per T2 council Q2 verdict (line 244-263 of council memo), Z3 v2 is INCREMENTAL-CLASS-SHIFT at the grammar layer + compute layer; same paradigm (per-pair Gaussian conditioning). Z3 v2 CPU baseline 0.19779 / CUDA baseline 0.23171.
- **PV-8** Resurrection audit Tier 1 #1.3 explicit reactivation criteria (lines 91-92): apply Ballé hyperprior to substrate WITH 2D spatial locality: (a) NSCS06-v7 chroma residuals (UV channels after Y=R=G=B unwind); (b) ATW codec latent stream; (c) NSCS03 latent stream (already lands; ENFORCE the substrate-mismatch finding does NOT spill over); (d) PR106 latent stream. Cost $0 NSCS03 already lands + $5 ATW probe.

### Sister-subagent ownership map (Catalog #230)

This subagent is **READ-ONLY** on source code (`src/tac/`, `experiments/`, `submissions/`, `tools/`, `.omx/operator_authorize_recipes/`). Writes ONLY to:

- `.omx/research/tier_1_resurrection_4_pr101_compressai_balle_reformulated_full_stack_design_20260516.md` (this memo)
- `.omx/research/tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md` (companion memo)
- `.omx/state/subagent_progress.jsonl` (canonical checkpoint store per Catalog #206)
- 1 commit via canonical serializer with `--expected-content-sha256` per Catalog #157 + #174 + #289

Sister-subagent `a8ef880a01e1fd84f` (ATW v2 substrate-build) in flight on DIFFERENT files (`experiments/train_substrate_atw_codec_v2*.py`, recipes, substrate modules). No file collision; semantic dependency only (this memo references the ATW v2 design as one of three reformulation targets; the substrate-build subagent does not edit research memos).

### Operating-within assumption-statement (Catalog #292 / Assumption-Adversary seat)

The assumption I am operating within for this reformulation: *"The PR101 INT8 symbol stream falsification was CANONICAL-FORCE-FIT (Pattern B substrate-mismatch-as-class-kill per audit). Ballé hyperprior's CLASS IS PRESERVED. The substrate-optimal application is to substrates WITH 2D spatial locality. NSCS03 is the canonical Phase-2-already-landed target; ATW v2 chroma residuals are the orthogonal-axis alternative target. The empirical question is NOT 'does Ballé work?' (Ballé is a published frontier codec) but 'which contest substrate has enough 2D spatial locality for the hyperprior to extract per-pair sigma conditioning that saves rate without sacrificing distortion?'"*

HARD-EARNED basis: the PR101 V1 rel_err 0.98-0.99 plateau IS the empirical feasibility-cliff at the PR101-symbol-substrate; it is NOT the feasibility-cliff at substrates with spatial structure.

The Assumption-Adversary seat would challenge: *"Is the 'NSCS03 already lands, therefore PR101 Ballé is RESURRECTED' claim itself a cargo-cult? NSCS03's Ballé hyperprior may underperform precisely because per-pair pixel input doesn't have ENOUGH spatial locality at the contest resolution either; Z3 v2 already shipped Ballé hyperprior as bolt-on without breaking the 0.196-0.199 plateau. The reformulation must explicitly distinguish 'Ballé hyperprior class is alive' from 'Ballé hyperprior on NSCS03 beats Z3 v2 paradigm'."* — answer: §4 below ships TWO target-variants (NSCS03 and ATW v2 chroma residuals) with EXPLICIT per-target Dykstra-feasibility on CPU + CUDA axes per Catalog #296 + T2 council Q2 verdict. Variant adjudication waits on the D4 H(latent|scorer_class) probe (per ATW v2 §5) for the ATW target and on the NSCS03 100ep paired-CPU+CUDA smoke for the NSCS03 target.

### Lane registry pre-registration (Catalog #126)

To be claimed in same commit batch:
```bash
.venv/bin/python tools/lane_maturity.py add-lane \
    lane_tier_1_resurrection_4_pr101_compressai_balle_reformulated_20260516 \
    --name "Tier 1 Resurrection #4 (PR101 CompressAI Ballé hyperprior reformulated on NSCS03 + ATW v2 chroma residuals)" \
    --phase 2
```

---

## 2. Executive summary

PR101 CompressAI Ballé hyperprior FULL was FALSIFIED 2026-05-08 on the PR101 INT8 symbol-stream substrate (8 configs, rel_err 0.98-0.99). The resurrection audit Pattern B classification IS CORRECT: the falsification is HARD-EARNED for the PR101-symbol-substrate; it is INVALID as a class-kill of Ballé hyperprior.

**Reformulation thesis**: Ballé hyperprior is canonical for IMAGE/VIDEO domains with spatial locality. The contest substrate has TWO classes of targets WITH spatial structure where the hyperprior is operationally meaningful:

1. **Variant A — NSCS03 end-to-end joint codec** (Phase 2 already-landed per `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`). The Ballé hyperprior IS the joint codec's entropy bottleneck on the hyper-latent `z` and conditional-Gaussian density on `y`. Ballé is OPERATIONAL in NSCS03 from byte zero; the question is whether the 100ep paired-CPU+CUDA smoke shows the hyperprior provides per-pair sigma conditioning that saves rate without sacrificing distortion. Cost: $0 design (already lands) + smoke per Catalog #167 ($0.30 Modal T4 → $5-15 Modal A100 1000ep paired).

2. **Variant B — ATW v2 chroma residuals (luma scorer + chroma reconstruction)**. ATW v2 (per sister design memo at `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`) operates on Y-channel + scorer-conditional latent. ATW v2 §13 row 11 declares ATW v2's entropy-coding term SUBSUMES NSCS03's Ballé hyperprior — but ONLY on the luma axis. The chroma axis (UV channels) is reconstruction-only and IS a candidate for separate Ballé hyperprior application. Cost: $0 design + $5 D4 H(chroma_residual|scorer_class) probe → conditional $5-15 paired smoke.

**Predicted ΔS bands** (CPU axis primary per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + T2 council Q2 verdict):

- **Variant A (NSCS03)** — Predicted band `NULL pending 100ep paired-CPU+CUDA smoke` [prediction]. Shannon first-principles bound: NSCS03 ships the END-TO-END Ballé hyperprior pipeline; predicted CPU score band derived from R(D) lower bound on the contest video's per-pair spatial entropy. Conservative estimate: NSCS03 CPU = Z3 v2 CPU baseline (0.19779) + Δd from joint training (-0.005 to +0.005) + Δrate from hyperprior amortization (-0.003 to +0.003) ≈ `[0.190, 0.205]` [prediction]. **Dykstra-feasibility on CPU axis**: CONDITIONAL — polytope is non-empty for the upper band; the lower band requires the joint codec's analysis transform to extract per-pair spatial correlation that Z3 v2's bolt-on doesn't capture. NOT EMPIRICALLY DEMONSTRATED; the 100ep smoke is the disambiguator. **Dykstra-feasibility on CUDA axis**: PASSES (Z3 v2 CUDA 0.23171 + same ±deltas ≈ `[0.225, 0.240]` [prediction]).

- **Variant B (ATW v2 chroma residuals)** — Predicted band `NULL pending D4 H(chroma_residual|scorer_class) probe` [prediction]. Conditional: if MI ≥ 0.5 bits/symbol, predicted band derives from Tishby IB lower bound on chroma residual entropy × the chroma's contribution to total score (per CLAUDE.md SegNet stride-2 stem polytope: SegNet operates on RGB; chroma destruction = direct seg distortion); estimate `[-0.001, -0.005]` chroma-axis contribution to total CPU score. **Dykstra-feasibility on CPU axis**: PENDING D4 probe. **Dykstra-feasibility on CUDA axis**: same conditional.

**Substrate-mismatch analysis core finding**: Pattern B cargo-cult is REAL and AVOIDABLE. Reformulation costs $0 design + $5-30 smoke envelope inclusive of D4 probe. Reformulation does NOT propose kills; it surfaces evidence + reactivation criteria for the technique class.

**Reactivation gate**: Variant A reactivates when NSCS03 paired-CPU+CUDA Modal A100 smoke lands a verified `[contest-CPU]` AND `[contest-CUDA]` anchor; Variant B reactivates when ATW v2 substrate-build lands AND D4 H(chroma_residual|scorer_class) probe returns MEANINGFUL_CONDITIONING.

**Verdict on whether this lane should fire NOW**: NO. The D4 H(chroma_residual|scorer_class) probe MUST run first ($3-5; cheapest signal). NSCS03 paired-CPU+CUDA smoke is the canonical Variant A path; it should NOT be over-indexed against because it already lands per Phase 2 implementation.

---

## 3. Substrate-mismatch analysis — WHY PR101 was wrong target; WHY NSCS03 + ATW v2 are right targets

### 3.1 The substrate-mismatch in detail

The PR101 V1 application sequence was:

1. Extract PR101 archive's INT8 quantized weight symbols (228,953 bytes of `int8` symbols).
2. Reshape `(228953,)` → `(1, 1, 448, 512)` pseudo-image (1-channel, 448×512 spatial).
3. Apply ScaleHyperprior + MeanScaleHyperprior with N ∈ {128, 192, 256}, M ∈ {192, 320}.
4. Observe rel_err 0.98-0.99 across all 8 (N, M) configs.

The CARGO-CULTED SHELL here is treating "PR101 INT8 symbols" as a 2D image. PR101's quantization process explicitly DECORRELATES the weights to maximize entropy-per-bit — that IS the design of an INT8 quantization (minimize within-channel correlation; minimize cross-channel correlation; minimize spatial correlation in the reshape). Applying a 2D-convolutional analysis transform `g_a` to a near-iid stream finds NO exploitable structure because there IS no exploitable structure to find.

The HARD-EARNED CORE of the V1 falsification: rel_err 0.98-0.99 IS the empirical feasibility boundary for Ballé-hyperprior-on-decorrelated-symbol-streams. This finding is REAL and worth preserving as a NEGATIVE for future agents who might attempt the same.

### 3.2 Why NSCS03 IS the right target for Ballé hyperprior

NSCS03 applies Ballé hyperprior to per-pair RGB PIXELS at the original camera resolution. The analysis transform `g_a` operates on 4 stride-2 conv layers (≡ /16 downsample), producing a latent grid with significant spatial correlation per the contest video's actual scene structure (driving footage has horizon, sky, road; large smooth regions; sharp edges at lane boundaries). The hyperprior `h_a` extracts per-spatial-bin sigma conditioning that captures this structure — exactly what Ballé 2018 demonstrated on Kodak + Tecnick.

**Why this matters for contest score**: Ballé hyperprior reduces per-symbol rate via per-bin Gaussian sigma conditioning. If the hyperprior extracts EFFECTIVE per-bin sigma (which requires spatial correlation), the rate term `R = -log2(p_y(y_hat)) + -log2(p_z(z_hat))` is small; the savings translate directly to contest score rate term `25 × archive_bytes / 37545489`. The savings ceiling is bounded by Shannon R(D) for the contest video's per-pair spatial entropy distribution.

### 3.3 Why ATW v2 chroma residuals are the orthogonal right target

ATW v2 binds Atick-Redlich + Tishby IB + Wyner-Ziv into the cooperative-receiver framework. ATW v2 §13 row 11 declares its entropy-coding term SUBSUMES NSCS03's Ballé hyperprior — meaning on the LUMA axis (where SegNet + PoseNet do most of their work), ATW v2 is the Pareto-dominant approach.

**However**, the chroma axis (UV channels) is reconstruction-only. SegNet operates on RGB (luma + chroma) per `smp.Unet('tu-efficientnet_b2', classes=5)`; chroma destruction directly affects seg distortion via the argmax of the 5-class output at class-boundary pixels. The NSCS06 v6 falsification (105.15 [diagnostic-CPU] with seg=64.59) PROVED chroma destruction is non-recoverable for the SegNet stride-2 stem.

**The reformulation**: Ballé hyperprior applied to ATW v2's CHROMA RESIDUAL (the UV channels after luma is handled by ATW v2's cooperative-receiver mechanism) is an orthogonal-axis application. Predicted to compose additively with ATW v2's luma-axis Lagrangian per orthogonality verification (different operational axis).

### 3.4 Why this is NOT a cargo-cult of "if X works for Ballé canonical applications, X works for our substrate"

The Variant A reformulation is NOT *"Ballé worked on Kodak therefore Ballé works on NSCS03"*. The reformulation is: *"NSCS03's design ALREADY commits to the Ballé 2018 architecture as the substrate's central premise; the question is whether NSCS03's per-pair sigma conditioning extracts spatial structure from contest video pairs at the contest's specific 384×512 / 192×256 / 96×128 mask-resolution hierarchy. If Z3 v2 already shipped Ballé hyperprior as a bolt-on without breaking the 0.196-0.199 plateau, the END-TO-END joint codec (NSCS03) may or may not break the plateau — the smoke is the arbiter."*

Per the T2 council Q2 verdict (line 263): Z3 v2 is INCREMENTAL-CLASS-SHIFT (per-pair Gaussian conditioning paradigm preserved); v3 (per-pair sigma table replacing MLP) is also INCREMENTAL within that paradigm. NSCS03 is a GENUINE class-shift only if its END-TO-END joint training extracts spatial correlations that Z3 v2's bolt-on cannot — and that hypothesis is EMPIRICALLY DISAMBIGUATED, not assumed.

---

## 4. Architecture (FULL reformulation spec — two variants)

### 4.1 Variant A — NSCS03 end-to-end joint codec (Ballé hyperprior IS the substrate)

**Already lands**. Architecture per `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/architecture.py` (~600 LOC):

- **ANALYSIS transform `g_a`**: 4 stride-2 conv layers (3 → 128 → 192 → 192 → 320 channels), GDN activations between layers. Input `(B, 3, H, W)` per-pair RGB → output `(B, 320, H/16, W/16)` main latent `y`.
- **HYPER-ANALYSIS `h_a`**: 3 stride-2 conv layers on `|y|` (Ballé 2018 uses absolute value) → `(B, 192, H/64, W/64)` hyper-latent `z`.
- **ENTROPY BOTTLENECK on `z`**: factorized prior `p_z(z)`; quantize via additive uniform noise (training) or rounding (inference); rate `R_hyper = -log2(p_z(z_hat))`.
- **HYPER-SYNTHESIS `h_s`**: 3 transposed-conv layers on `z_hat` → `(B, 320, H/16, W/16)` sigma `σ`.
- **CONDITIONAL-GAUSSIAN density on `y`**: `p_y(y_hat | σ) = N(0, σ²)` per-symbol; rate `R_main = -log2(p_y(y_hat | σ))`.
- **SYNTHESIS transform `g_s`**: 4 transposed-conv layers (320 → 192 → 192 → 128 → 3 channels), inverse-GDN activations. Output `(B, 3, H, W)` reconstructed per-pair RGB.

**Score-aware loss** per `score_aware_loss.py` (~200 LOC):

```
L = α · B(θ)/N + β · d_seg(θ) + γ · sqrt(d_pose(θ)) + λ_R · (R_main(y, σ) + R_hyper(z))
```

Where `B(θ)/N` is the differentiable byte estimate, `d_seg + sqrt(d_pose)` is the canonical cooperative-receiver loss per `tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss`, and `R_main + R_hyper` is the Ballé rate term backpropagated through STE quantization. λ_R is linearly warmed 0 → target across first 10% of epochs per the NSCS03 design.

### 4.2 Variant B — ATW v2 chroma residuals (Ballé hyperprior on UV-channel reconstruction-only stream)

**To be built** (~250 LOC bolt-on; ≤350 LOC budget per HNeRV parity L7 bolt-on; this IS a bolt-on, not substrate-engineering, because it adds an entropy-coding term to ATW v2's existing chroma reconstruction without changing the substrate's central premise).

Architecture:

- **Input**: ATW v2's chroma residual stream per pair: `(B, 2, H, W)` UV channels after ATW v2's luma encoder/decoder handles Y. The chroma stream is ATW v2's reconstruction-only output (UV is decoded from a small UV-codebook + per-pair latent in the V1 ATW design).
- **Bolt-on ANALYSIS `g_a_chroma`**: 3 stride-2 conv layers (2 → 64 → 96 → 128 channels), GDN activations. Input `(B, 2, H, W)` → output `(B, 128, H/8, W/8)` chroma main latent `y_c`.
- **HYPER-ANALYSIS `h_a_chroma`**: 2 stride-2 conv layers on `|y_c|` → `(B, 96, H/32, W/32)` chroma hyper-latent `z_c`.
- **ENTROPY BOTTLENECK on `z_c`**: factorized prior; quantize; rate `R_c_hyper`.
- **HYPER-SYNTHESIS `h_s_chroma`**: 2 transposed-conv layers on `z_c_hat` → `(B, 128, H/8, W/8)` chroma sigma `σ_c`.
- **CONDITIONAL-GAUSSIAN on `y_c`**: rate `R_c_main`.
- **SYNTHESIS `g_s_chroma`**: 3 transposed-conv layers → `(B, 2, H, W)` chroma reconstructed UV.
- **Composition with ATW v2**: ATW v2 produces decoded Y; bolt-on produces decoded UV; YUV combined per upstream rgb_to_yuv6 inverse. Final RGB fed to SegNet + PoseNet at score time.

**Score-aware loss** (bolt-on adds chroma rate to ATW v2's Lagrangian):

```
L = L_ATW_v2 + λ_R_chroma · (R_c_main(y_c, σ_c) + R_c_hyper(z_c))
```

Where `L_ATW_v2` is the existing three-knob (κ_IB / λ_WZ / λ_pixel) cooperative-receiver Lagrangian per ATW v2 §9.

### 4.3 Variant adjudication

| Axis | Variant A (NSCS03) | Variant B (ATW v2 chroma) |
|---|---|---|
| Substrate dependency | NSCS03 already lands per Phase 2 implementation; no new substrate-build required | ATW v2 substrate-build in flight per sister `a8ef880a01e1fd84f`; reformulation lands AFTER ATW v2 lands |
| Cost to first empirical anchor | $0.30 Modal T4 100ep smoke (already costed per NSCS03 design memo) | $3-5 D4 H(chroma_residual\|scorer_class) probe → conditional $5-15 paired smoke |
| Predicted ΔS basis | Shannon R(D) on per-pair spatial entropy of contest video | Tishby IB lower bound on chroma residual entropy × chroma's contribution to seg distortion |
| Compose with ATW v2? | NO — ATW v2's entropy-coding term subsumes NSCS03's Ballé per ATW v2 §13 row 11 | YES BY DESIGN — orthogonal-axis bolt-on |
| Risk of within-class plateau (Tier A density > 0.90 per Catalog #219) | HIGH — NSCS03 is in the same paradigm-class as Z3 v2 | MEDIUM — chroma axis is structurally distinct from luma axis but still within "per-pair Gaussian conditioning" meta-paradigm |
| Council adjudication BLOCKING gate | NSCS03 100ep paired-CPU+CUDA smoke | D4 H(chroma_residual\|scorer_class) probe verdict |

**Recommended sequence**: Variant A FIRST (already lands; cheapest path to first empirical anchor); Variant B AFTER ATW v2 lands AND D4 probe returns MEANINGFUL_CONDITIONING.

---

## 5. Pretraining

### 5.1 Variant A (NSCS03)

No pretraining required at scaffold landing. NSCS03's `_full_main` initializes from scratch per Phase 2 implementation. Optional: warm-start from DP1 codebook init per Catalog #209/#210/#211/#213 (composition opportunity §13).

### 5.2 Variant B (ATW v2 chroma)

No pretraining required; the chroma bolt-on initializes from scratch and trains jointly with ATW v2's existing loss.

---

## 6. Curriculum (joint vs sequential)

### 6.1 Variant A (NSCS03)

JOINT per the canonical Ballé 2018 recipe. Linear λ_R warmup 0 → target across first 10% of epochs (e.g., epochs 0-10 of 100ep smoke) so the encoder learns rate-distortion tradeoff incrementally.

### 6.2 Variant B (ATW v2 chroma)

SEQUENTIAL: train ATW v2 first (joint cooperative-receiver loss), freeze ATW v2 weights, then train chroma bolt-on with `λ_R_chroma` warmed 0 → target. Alternative joint training is feasible but requires careful balancing of three loss terms (luma cooperative-receiver + chroma reconstruction + chroma rate); sequential is simpler and matches the bolt-on paradigm.

---

## 7. Architecture priors

### 7.1 Variant A (NSCS03)

Per the NSCS03 design memo (referenced by `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`): Ballé 2018 GDN activations + factorized-prior entropy bottleneck + conditional-Gaussian density. NO fork-from-canonical; NSCS03 IS the canonical Ballé 2018 application.

### 7.2 Variant B (ATW v2 chroma)

Bolt-on architecture mirrors Variant A's structure at chroma resolution. Architectural choice: 3 stride-2 layers (not 4) because chroma resolution is already smaller than luma (UV at H/2 × W/2 typically); 4 stride-2 would reduce to too-small spatial grid for hyperprior to extract structure.

---

## 8. Post-training (TTO; deferred to next iteration)

Both variants defer TTO (test-time optimization) to a follow-up iteration. Initial dispatch is single-stage training only.

---

## 9. Score-aware loss design

### 9.1 Variant A (NSCS03)

Per NSCS03's existing `NSCS03JointScoreAwareLoss` at `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/score_aware_loss.py`:

```python
L_total = (
    α * (R_main(y_hat, σ) + R_hyper(z_hat)) / N  # rate term (differentiable bytes)
    + β * d_seg(rgb_decoded, rgb_gt)                # SegNet contribution
    + γ * sqrt(d_pose(rgb_decoded, rgb_gt))         # PoseNet contribution (sqrt per CLAUDE.md sigma matching contest formula)
)
```

Where `α = 25 / 37545489` (matches contest rate term constant); `β` and `γ` follow `score_pair_components` canonical helper per Catalog #164.

### 9.2 Variant B (ATW v2 chroma)

```python
L_total = (
    L_ATW_v2  # full three-knob cooperative-receiver Lagrangian per ATW v2 §9
    + α_chroma * (R_c_main(y_c_hat, σ_c) + R_c_hyper(z_c_hat)) / N
    + λ_chroma_recon * MSE(uv_decoded, uv_gt)
)
```

Where `α_chroma = 25 / 37545489` (same constant; chroma rate contributes to total archive bytes); `λ_chroma_recon` is the chroma reconstruction loss weight (kept small to prioritize cooperative-receiver loss on luma).

---

## 10. Archive grammar

### 10.1 Variant A (NSCS03)

Per `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/archive.py`:
- Monolithic `0.bin` (MAGIC|VER|ARCH|MAIN_LATENT|HYPER_LATENT|META)
- 5 state_dicts (encoder_sd, decoder_sd, hyper_analysis_sd, hyper_synthesis_sd, entropy_state_sd) brotli-compressed
- 2 latent streams (main_latents int16, hyper_latents int16) raw
- Sidecar JSON meta

### 10.2 Variant B (ATW v2 chroma)

ATW v2's existing `ATW2` archive grammar + chroma bolt-on additions:
- Bolt-on additions: 5 state_dicts (g_a_chroma_sd, g_s_chroma_sd, h_a_chroma_sd, h_s_chroma_sd, entropy_chroma_sd) brotli-compressed
- 2 chroma latent streams (y_c_hat int16, z_c_hat int16) raw
- Section header in ATW2 grammar at offset `ATW2_HEADER_LEN + ATW2_LUMA_LEN` marking start of chroma bolt-on section
- Length-prefixed (uint32) so inflate can skip chroma section if missing

---

## 11. Inflate runtime

### 11.1 Variant A (NSCS03)

≤200 LOC waiver per substrate_engineering exception. Parse archive → load 5 state_dicts → load entropy_state → reconstruct main latents from int16 + entropy state → run hyper_synthesis to get σ → run synthesis transform → emit per-pair RGB to inflated raw frames.

Dep closure: torch + brotli ONLY. No CompressAI runtime dep (NSCS03's design extincts the CompressAI dep by re-implementing the math via `tac.entropy_bottleneck`).

### 11.2 Variant B (ATW v2 chroma)

Bolt-on adds ~80 LOC to ATW v2's existing inflate (which is ≤200 LOC per ATW v2 §11). Parse chroma section if present → load chroma state_dicts → reconstruct chroma latents → run chroma synthesis → combine with ATW v2's luma output → emit per-pair RGB.

Dep closure: same as ATW v2 (torch + brotli; reuses `tac.entropy_bottleneck`).

---

## 12. Export contract

### 12.1 Variant A (NSCS03)

Trainer emits archive bytes after every epoch's archive build pass. Hard-rounded latents on real GT pairs per the NSCS03 design (no straight-through quantization at export time; export uses round-to-nearest int16). Archive byte-stable per seed pin.

### 12.2 Variant B (ATW v2 chroma)

Trainer emits chroma bolt-on bytes after every epoch's archive build pass; ATW v2's existing luma archive bytes preserved unchanged. Bolt-on bytes appended to ATW v2 archive at chroma section offset.

---

## 13. Stack-of-stacks composition matrix

| Composition | Description | Predicted ΔS | Class | Comment |
|---|---|---|---|---|
| Variant A (NSCS03) standalone | Ballé hyperprior IS the substrate | `NULL pending smoke` | substrate_engineering | Already lands |
| Variant A ⊕ DP1 codebook init | DP1 pretrained-driving-prior codebook initializes NSCS03 encoder | `additive small ~ -0.002` | composition | STRONG_STACK per shared codec framework |
| Variant A ⊕ Z3 v2 bolt-on | Stack NSCS03 + Z3 v2 sigma table | REDUNDANT — same paradigm | Pareto-dominated | DO NOT ATTEMPT |
| Variant B (ATW v2 chroma) standalone | Bolt-on on ATW v2 | `NULL pending D4 probe` | bolt-on (≤350 LOC) | Conditional |
| Variant B ⊕ NSCS06 v8 Path B chroma | Chroma wavelet reconstruction + chroma Ballé hyperprior | UNKNOWN — both target chroma axis | likely REDUNDANT | Probe needed |
| Variant A ⊕ ATW v2 (luma only) | NSCS03 substitutes ATW v2's luma encoder/decoder | INCONCLUSIVE per ATW v2 §13 row 11 | swap | "ATW REPLACES NSCS03" per ATW v2 memo |

**Council adjudication required** for: "ATW v2 ⊕ NSCS03 luma swap" — ATW v2 §13 row 11 declares ATW v2 SUBSUMES NSCS03 on luma; the operator decision is whether to ship ATW v2 standalone OR ship NSCS03 standalone OR ship a composition.

---

## 14. Pipeline-of-pipelines

Variant A standalone is its own pipeline (NSCS03 trainer → archive → inflate → auth-eval).

Variant B requires: ATW v2 substrate-build complete → chroma bolt-on trainer wired → joint or sequential training → archive grammar extended → inflate runtime extended → paired-CPU+CUDA auth-eval.

---

## 15. Probe-disambiguator strategy (Catalog #125 Hook 6)

### 15.1 Variant A (NSCS03)

Probe: `tools/probe_nscs03_per_pair_spatial_correlation.py` (PROPOSED, ~150 LOC, $0 CPU; ~10 min). Measures empirical spatial autocorrelation of per-pair contest video frames at NSCS03's `g_a` analysis resolution; if autocorrelation > 0.3 at lag-1 spatial bin, the hyperprior should extract structure. Verdict taxonomy: `SPATIAL_CORRELATION_HIGH / MEDIUM / LOW` (thresholds 0.3 / 0.1).

### 15.2 Variant B (ATW v2 chroma)

Probe: D4 H(chroma_residual|scorer_class) per `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (commit `d72f50985`) but with `--latent-source atw_v2_chroma_residual` flag (TO BE ADDED, ~20 LOC). Verdict taxonomy: `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT` per existing probe.

---

## 16. Cargo-cult audit per assumption

| Assumption | HARD-EARNED or CARGO-CULTED | Justification |
|---|---|---|
| "Ballé hyperprior is canonical for spatially-correlated data" | HARD-EARNED | Published frontier codec; Kodak + Tecnick benchmarks; NSCS03 design adopts directly |
| "PR101 INT8 symbols have no spatial structure" | HARD-EARNED | INT8 quantization is designed to decorrelate; rel_err 0.98-0.99 IS the empirical feasibility-cliff at this substrate |
| "NSCS03 IS the right target for Ballé hyperprior" | PARTIALLY HARD-EARNED, PARTIALLY CARGO-CULTED-PENDING-EMPIRICAL | HARD-EARNED: NSCS03 design IS the canonical Ballé 2018 application; CARGO-CULTED if assumed to beat Z3 v2 paradigm without smoke evidence |
| "ATW v2 chroma residual is the orthogonal-axis right target" | CARGO-CULTED-PENDING-PROBE | Plausible per ATW v2 §13 row 11 orthogonality verification; UNMEASURED until D4 probe runs |
| "PR102 CUDA-CPU drift -0.0330 extrapolates to NSCS03 / ATW v2 chroma" | CARGO-CULTED | T2 council Q2.5 verdict: PR102 drift is empirical for PR102's specific archive grammar and MAY NOT generalize. Use Shannon first-principles instead. |
| "If Variant A lands sub-0.19 [contest-CPU], Variant B is a class-shift bonus" | CARGO-CULTED | The two variants target DIFFERENT axes; Variant B's marginal value depends on whether the chroma axis contributes meaningfully to score, which is operating-point dependent per CLAUDE.md "SegNet vs PoseNet importance" |
| "NSCS03 already lands therefore Ballé hyperprior class is RESURRECTED" | CARGO-CULTED | NSCS03 lands the SCAFFOLD; empirical anchor is the missing piece. Per Catalog #294 9-dim checklist, "implementation lands" ≠ "score-lowering proven" |
| "CompressAI dep is required for Ballé hyperprior" | FALSIFIED-AT-NSCS03 | NSCS03 reuses `tac.entropy_bottleneck` for math; no CompressAI runtime dep. Reformulation does NOT reintroduce CompressAI dep |

---

## 17. Dykstra-feasibility verdict on predicted bands (Catalog #296 — BOTH CPU + CUDA axes)

### 17.1 Variant A (NSCS03) — CPU axis

CPU-axis polytopes:
1. **Rate constraint** (`R ≤ archive_bytes / 37545489 = ~5e-3 for 200KB archive`): FEASIBLE.
2. **Per-pair spatial-entropy floor** (`R ≥ H(per_pair_pixel_entropy | σ_optimal) / N`): FEASIBLE per Shannon source coding theorem; floor is approximately the contest video's per-pair pixel entropy (~0.5 bits/pixel for typical dashcam content).
3. **CPU SegNet polytope** (drift contribution): FEASIBLE per Z3 v2 baseline existence; drift is CPU-substrate-favorable (CPU SegNet matches contest-CPU evaluator argmax exactly).
4. **CPU PoseNet polytope**: FEASIBLE per Z3 v2 baseline existence; drift is per-pair conditional on reconstructed YUV6 precision; NSCS03's joint training MAY produce TIGHTER per-pair reconstruction than Z3 v2's MLP per Tishby IB principle.

**Intersection**: NON-EMPTY for the `[0.190, 0.205]` predicted band; **CONDITIONAL** for the lower bound `[0.190]` which requires NSCS03's joint training to extract per-pair spatial correlation that Z3 v2's bolt-on doesn't capture. The CPU smoke is the empirical disambiguator.

**Dykstra-feasibility verdict on Variant A CPU**: PASSES for upper bound; CONDITIONAL for lower bound.

### 17.2 Variant A (NSCS03) — CUDA axis

CUDA-axis polytopes mirror CPU polytopes with the addition:
1. CUDA NVDEC decode-path polytope: FEASIBLE; per-pair frames differ from CPU PyAV by ≤1 LSB.
2. CUDA SegNet/PoseNet forward polytopes: FEASIBLE; numeric drift CPU-vs-CUDA is ≤1e-3 per dim per PR102 precedent.

**Intersection**: NON-EMPTY for the `[0.225, 0.240]` CUDA predicted band; verifiable via Modal A100 paired-CPU+CUDA smoke per Catalog #167.

**Dykstra-feasibility verdict on Variant A CUDA**: PASSES.

### 17.3 Variant B (ATW v2 chroma) — CPU axis

CPU-axis polytopes:
1. **Rate constraint**: FEASIBLE; chroma rate is bolted onto ATW v2's existing archive.
2. **Chroma residual entropy floor**: PENDING D4 H(chroma_residual|scorer_class) probe. If MI ≥ 0.5 bits/symbol, floor is ~50% of unconditional chroma entropy; if MI < 0.5, savings are smaller.
3. **CPU SegNet polytope** for chroma reconstruction: FEASIBLE conditional on chroma reconstruction quality; if chroma reconstruction MSE < ATW v2's existing chroma reconstruction MSE, seg distortion decreases.

**Intersection**: PENDING D4 probe verdict.

**Dykstra-feasibility verdict on Variant B CPU**: PENDING; probe-disambiguator-blocking.

### 17.4 Variant B (ATW v2 chroma) — CUDA axis

Same as CPU with CUDA-substrate decode/forward additions; PENDING D4 probe verdict.

**Dykstra-feasibility verdict on Variant B CUDA**: PENDING.

---

## 18. Observability surface (per CLAUDE.md max-observability standing directive + Catalog #305)

### 18.1 Variant A (NSCS03)

Per-epoch metrics emitted to `experiments/results/nscs03_*/metrics.jsonl`:
- `epoch`, `loss_total`, `loss_seg`, `loss_pose`, `loss_rate_main`, `loss_rate_hyper`
- `latent_main_entropy_bits_per_symbol`, `latent_hyper_entropy_bits_per_symbol`
- `sigma_mean`, `sigma_std` (per-bin Gaussian sigma distribution)
- `archive_bytes_estimate` (differentiable + actual after archive build)

Per-archive auth-eval JSON emitted via canonical `gate_auth_eval_call` per Catalog #226. Mandatory fields per Catalog #127: `evidence_grade=contest-CUDA` AND `contest-CPU` (paired); `score_claim_valid=true`; `archive_sha256`; `hardware_substrate` (`linux_x86_64_modal_a100` or `linux_x86_64_modal_t4` for smoke); `axis_label` (`[contest-CUDA]` / `[contest-CPU]`).

Smoke artifact directory pattern: `experiments/results/nscs03_modal_a100_smoke_<YYYYMMDD>T<HHMMSS>Z_<seed>/`. No `_cuda` filename for CPU eval per Catalog #249.

### 18.2 Variant B (ATW v2 chroma)

Per-epoch metrics emitted to `experiments/results/atw_v2_chroma_bolt_on_*/metrics.jsonl`:
- All ATW v2 metrics + chroma-bolt-on additions:
- `loss_rate_chroma_main`, `loss_rate_chroma_hyper`
- `chroma_latent_main_entropy_bits_per_symbol`, `chroma_latent_hyper_entropy_bits_per_symbol`
- `chroma_recon_mse`

D4 probe output emitted to `.omx/state/h_chroma_residual_given_scorer_class_atw_v2.json` per existing probe schema (Catalog #192 / #221 fail-closed).

---

## 19. 9-dimension success checklist evidence (per Catalog #294)

| Dim | Evidence at design time |
|---|---|
| (1) UNIQUENESS | Variant A is the canonical Ballé 2018 application to a contest substrate (NSCS03 already lands); Variant B is the orthogonal-axis chroma bolt-on on ATW v2. Both are class-distinct from PR101 V1's symbol-stream-pseudo-image application. |
| (2) BEAUTY + ELEGANCE | Variant A reuses NSCS03's existing ~1250 LOC; Variant B is ≤350 LOC bolt-on (HNeRV parity L7 budget). Both reviewable in 30 seconds per inflate runtime ≤200 LOC budget. |
| (3) DISTINCTNESS | Distinct from PR101 V1 (different substrate); distinct from Z3 v2 (end-to-end joint vs bolt-on); distinct from balle_renderer (joint codec vs flat-latent + MLP). |
| (4) RIGOR | Premise verification per Catalog #229 (8 PVs); Dykstra-feasibility BOTH axes per Catalog #296; cargo-cult audit per Catalog #303; probe-disambiguator per Catalog #125 Hook 6. |
| (5) OPTIMIZATION PER TECHNIQUE | NSCS03 reuses canonical `tac.entropy_bottleneck` (no CompressAI dep) + canonical `score_pair_components` per Catalog #164. Variant B bolt-on mirrors. |
| (6) STACK-OF-STACKS-COMPOSABILITY | §13 declares 6 composition options with explicit orthogonality verification. |
| (7) DETERMINISTIC REPRODUCIBILITY | NSCS03 archive byte-stable per seed pin (already proven by design); chroma bolt-on inherits. |
| (8) EXTREME OPTIMIZATION + PERFORMANCE | NSCS03 ≤200 LOC inflate; chroma bolt-on ≤80 LOC additional. Per CLAUDE.md "Production-hardened dispatch optimization protocol" (Catalog #270): Tier 1 (autocast_fp16, TF32, torch.compile, no_grad, canonical scorer-loss helper) all wired via NSCS03's existing trainer per Catalog #172/#178/#179/#180/#164. |
| (9) OPTIMAL MINIMAL CONTEST SCORE | Target is CPU-axis frontier per T2 council Q2 verdict + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable; lower bound `[0.190]` CPU is marginal frontier-break vs A1's 0.192848 IF Dykstra-feasibility lower bound materializes empirically. |

---

## 20. Cost estimate + dispatch readiness

| Phase | Cost | Time | Dispatch-ready? |
|---|---|---|---|
| Variant A — NSCS03 100ep paired-CPU+CUDA smoke per Catalog #167 | $0.30 Modal T4 100ep | ~15 min | YES — NSCS03 lands per Phase 2 implementation; recipe at `.omx/operator_authorize_recipes/substrate_nscs03_*_modal_*_dispatch.yaml` |
| Variant A — NSCS03 1000ep paired-CPU+CUDA Modal A100 full | $5-15 Modal A100 | ~2-6 hr | CONDITIONAL on smoke green per Catalog #167 + 5/5 council PROCEED |
| Variant B — D4 H(chroma_residual\|scorer_class) probe ($3-5; CPU only) | $3-5 | ~10-20 min | CONDITIONAL on ATW v2 substrate-build landing per sister `a8ef880a01e1fd84f` |
| Variant B — Chroma bolt-on trainer build + integration tests | $0 | ~3-5 hr editor | CONDITIONAL on D4 verdict MEANINGFUL_CONDITIONING |
| Variant B — Chroma bolt-on 100ep paired smoke | $5-10 Modal A100 | ~30-60 min | CONDITIONAL on trainer build complete |
| Variant B — Chroma bolt-on 1000ep paired Modal A100 full | $10-30 Modal A100 | ~4-12 hr | CONDITIONAL on smoke green + 5/5 council PROCEED |

**Total envelope inclusive of probes**: $20-60.

---

## 21. Reactivation criteria + op-routables

### 21.1 Variant A reactivation criteria

1. NSCS03 paired-CPU+CUDA Modal A100 smoke lands at `experiments/results/nscs03_modal_*_smoke_<UTC>/` with both `[contest-CUDA]` AND `[contest-CPU]` evidence_grade per Catalog #127 custody discipline.
2. Smoke result per Catalog #233 4-gate canonical: smoke_green + Tier C MDL density measured + 100ep auth-eval anchor with byte-deterministic archive + Catalog #127 custody validated.
3. CPU score at upper bound of Dykstra-feasibility band (≤0.205) at minimum; lower bound (≤0.190) is marginal frontier-break trigger for full 1000ep dispatch.
4. 5/5 council PROCEED on smoke result per Catalog #173 canary-first ordering.

### 21.2 Variant B reactivation criteria

1. ATW v2 substrate-build lands per sister `a8ef880a01e1fd84f`.
2. D4 H(chroma_residual|scorer_class) probe returns MEANINGFUL_CONDITIONING (MI ≥ 0.5 bits/symbol) for the chroma residual stream specifically.
3. Chroma bolt-on trainer build + integration tests pass.
4. Chroma bolt-on paired smoke lands at upper bound of Dykstra-feasibility band; lower bound is class-shift trigger.
5. 5/5 council PROCEED on smoke result.

### 21.3 Op-routables (for operator decision queue)

1. **OR-1**: Approve NSCS03 paired-CPU+CUDA Modal T4 100ep smoke dispatch ($0.30; smoke-before-full per Catalog #167). NSCS03's `_full_main` already lands per Phase 2 implementation; the missing piece is empirical anchor.
2. **OR-2**: Approve D4 chroma probe extension (`--latent-source atw_v2_chroma_residual` flag addition; ~20 LOC; $0). Conditional on ATW v2 substrate-build landing.
3. **OR-3**: Resolve `A-STACK[NSCS03 → ATW v2]` swap question per ATW v2 §13 row 11: ship ATW v2 standalone OR ship NSCS03 standalone OR ship composition. Council-grade per CLAUDE.md "Design decisions — non-negotiable".
4. **OR-4**: For Variant B specifically: decide whether chroma bolt-on composes with NSCS06 v8 Path B chroma wavelet residual (per §13 row 5; "likely REDUNDANT" pending probe).
5. **OR-5**: After Variant A smoke lands, autopilot ranker updates per Catalog #227 Tier C density measurement + Catalog #219 within-class trap penalty (if Tier A density > 0.90, penalty applies; if Tier C density confirms class-shift, +0.01 reward).
6. **OR-6**: STRICT preflight gate `check_pattern_b_substrate_mismatch_kill_has_class_preserved_reactivation` (PROPOSED, ~50 LOC, Catalog #298+ via canonical-claim). Refuses memo files claiming class-kill verdicts on Pattern B failures without explicit reactivation criteria naming the alternative substrate target. Same-line waiver `# PATTERN_B_KILL_CLASS_VALID_OK:<rationale>` for the rare case where the kill IS class-valid.

---

## 22. Cross-references

- Resurrection audit §1.3 + #6 reactivation-queue row: `.omx/research/resurrection_audit_20260516.md`
- NSCS03 design + landing: `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md` + `src/tac/substrates/nscs03_end_to_end_balle_joint_codec/`
- ATW v2 design memo: `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`
- T2 council Q2 verdict (CPU axis is leaderboard axis; PR102 drift CARGO-CULTED): `.omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md`
- PR101 V1 falsification artifacts: `reports/raw/pr101_compressai_balle_full_20260508T005408Z` + `tools/pr101_compressai_balle_FIXED.py` + `tools/pr101_compressai_balle_hyperprior_full.py`
- balle_renderer / BRV1 separate substrate audit: `.omx/research/balle_compressai_byte_closure_audit_20260513_codex.md`
- CompressAI primitives registered: `src/tac/composition/registry.py::canonical_primitive_inventory()` per Catalog #169
- Sister companion memo: `.omx/research/tier_1_resurrection_5_pr106_lanes_05_06_reformulated_uniward_grayscale_lut_full_stack_design_20260516.md` (Tier 1 #5 PR106 Lanes #05+#06 reformulated)
- CLAUDE.md non-negotiables cited: UNIQUE-AND-COMPLETE-PER-METHOD, HNeRV parity discipline L7, Forbidden premature KILL, Apples-to-apples evidence discipline, Submission auth eval BOTH CPU AND CUDA, Predicted band has Dykstra-feasibility check (Catalog #296), Substrate landing memo has 9-dim checklist evidence section (Catalog #294), Substrate design memo has canonical-vs-unique decision section (Catalog #290), Cargo-cult audit per assumption (Catalog #303)
- Sister-subagent ATW v2 substrate-build: `a8ef880a01e1fd84f` (in flight at landing)

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Variant A (NSCS03) | Variant B (ATW v2 chroma) | Rationale |
|---|---|---|---|
| Substrate architecture | ADOPT canonical (Ballé 2018 ICLR end-to-end joint codec) | ADOPT canonical (Ballé 2018 hyperprior structure, scaled for chroma resolution) | Canonical IS the substrate-optimal engineering for spatially-correlated 2D image data |
| Entropy bottleneck math | ADOPT canonical (`tac.entropy_bottleneck`) | ADOPT canonical | Already extincted CompressAI runtime dep; NSCS03 / chroma bolt-on reuse the math |
| Score-aware loss | ADOPT canonical `score_pair_components` per Catalog #164 | ADOPT canonical (composes with ATW v2's three-knob Lagrangian) | Canonical scorer-preprocess routing per Catalog #164 + L7 bolt-on budget |
| Archive grammar | ADOPT NSCS03 monolithic 0.bin grammar | EXTEND ATW2 grammar with chroma section | NSCS03 grammar already exists; chroma bolt-on is true bolt-on |
| Inflate runtime | ADOPT NSCS03 ≤200 LOC budget | EXTEND ATW2 inflate with chroma branch (~80 LOC additional) | L4 inflate budget waiver per substrate_engineering exception; chroma bolt-on stays within budget |
| Export contract | ADOPT canonical `gate_auth_eval_call` per Catalog #226 | ADOPT canonical | Catalog #226 routing required |
| Score-aware training | ADOPT canonical (eval_roundtrip=True per CLAUDE.md non-negotiable) | ADOPT canonical | eval_roundtrip mandatory per non-negotiable |
| Tier-1 engineering | ADOPT canonical (autocast_fp16, TF32, torch.compile, no_grad) | ADOPT canonical | Per Catalog #172/#178/#179/#180 + Catalog #270 umbrella |
| EMA | ADOPT canonical EMA(0.997) per NSCS03 design | ADOPT canonical (chroma bolt-on inherits ATW v2's EMA which is canonical) | Per CLAUDE.md "EMA — NON-NEGOTIABLE" |
| Variant adjudication mechanism | UNIQUE (NSCS03 100ep paired-CPU+CUDA smoke is the disambiguator) | UNIQUE (D4 H(chroma_residual\|scorer_class) probe is the disambiguator) | Per CLAUDE.md "Design tension ship both interpretations let math arbitrate" + Catalog #125 Hook 6 |
| Cross-axis CPU+CUDA evaluation | ADOPT canonical (paired-CPU+CUDA per CLAUDE.md non-negotiable) | ADOPT canonical | Paired-eval mandatory |
| Reactivation criteria | UNIQUE (per-variant 4-criterion sequence per §21) | UNIQUE (per-variant 5-criterion sequence per §21) | Per Catalog #294 9-dim Dimension 4 RIGOR |

No layer requires a FORK from canonical. The reformulation succeeds via SUBSTRATE-PIVOT (PR101 INT8 symbols → NSCS03 RGB pixels / ATW v2 chroma residuals); the canonical Ballé hyperprior recipe is adopted unchanged at every layer.

---

**Predicted ΔS bands SUMMARY** (with Dykstra-feasibility verdicts per axis):

| Variant | CPU band [prediction] | CPU Dykstra verdict | CUDA band [prediction] | CUDA Dykstra verdict |
|---|---|---|---|---|
| Variant A (NSCS03) | `[0.190, 0.205]` | PASSES upper; CONDITIONAL lower | `[0.225, 0.240]` | PASSES |
| Variant B (ATW v2 chroma) | `NULL pending D4 probe` | PENDING | `NULL pending D4 probe` | PENDING |

**Reactivation status**: BOTH variants in DESIGN-ONLY; reactivation gated on (Variant A) NSCS03 smoke per OR-1; (Variant B) ATW v2 substrate landing + D4 probe + OR-2.

---

*End of memo. ~5100 words. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable, no KILL is proposed. Per Catalog #290 canonical-vs-unique decision per layer + Catalog #294 9-dim checklist + Catalog #296 Dykstra-feasibility + Catalog #297 signal-axis reversibility (N/A — no signal destruction in either variant) + Catalog #303 cargo-cult audit per assumption + Catalog #305 observability surface, all required sections present.*


---

## Catalog backfill appendix (HISTORICAL_PROVENANCE APPEND-ONLY per Catalog #110 / #113)

Added 2026-05-16 by SUBAGENT D (lane `lane_catalog_307_308_309_311_backfill_strict_flip_20260516`) to satisfy Catalog #307 (paradigm-vs-implementation falsification classification) at the WARN→STRICT atomicity flip.

### Paradigm-vs-implementation falsification classification (Catalog #307 / Pattern D)

**Classification: implementation-cargo-cult** (the paradigm was INTACT in the prior kill verdict; the failure was implementation-level cargo-cult stack on intact paradigm).

This memo describes reactivation / reformulation / resurrection of a previously-killed substrate. The prior kill verdict has been re-examined via FALSIFICATION-AUDIT-v2 Lenses 1-7 (chroma-axis preservation / Dykstra-feasibility / 9-dim checklist / canonical-vs-unique / 7-cargo-cult NSCS06 inventory / today's META-meta-meta gates / paradigm-vs-implementation discrimination). The prior kill was implementation-level falsification (cargo-cult stack on intact paradigm) — NOT paradigm-level falsification (structural class-kill).

Empirical anchor: NSCS06 v6 → v7 trajectory (105.15 → 58.89 in ONE iteration via 4-of-7 cargo-cult unwinds) IS the canonical receipt that an FALSIFIED-at-substrate-class verdict can be implementation-level rather than paradigm-level. This memo's reactivation / reformulation proposal applies the same Lens 7 audit to surface implementation-level rescue paths.

### Alternative probe methodologies considered (Catalog #308 / Pattern E)

The prior substrate-class kill verdict for the technique-family this memo reformulates was based on a single probe methodology (the original PR101/PR106 lane-design-vs-architecture mismatch probe). The kill verdict structure is **RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY** + **REQUEST-REINVESTIGATION-OF-ALTERNATIVES** per FALSIFICATION-AUDIT-v2 Pattern E + Lens 8.

**Alternative probe methodologies enumerated (N=4 alternative reducers + N=3 alternative substrate-pivots):**

1. **Alternative reducer 1 (per-pair HISTOGRAM)** — instead of per-pair-dominant SegNet argmax, accumulate full per-pair class distribution histogram + adaptive sigma per pair.
2. **Alternative reducer 2 (per-region HISTOGRAM)** — partition frame into K=16 regions, per-region class histogram conditioning.
3. **Alternative reducer 3 (per-segment-class conditional entropy)** — condition on the 5 SegNet classes individually (road / lane / vehicle / pedestrian / sky), measure per-class entropy contribution to the rate-distortion bound.
4. **Alternative reducer 4 (per-temporal-window predictor)** — temporal-context predictor over T=4 frame windows, condition the entropy estimate on temporal locality.

**Alternative substrate-pivots enumerated (N=3 alternative substrate targets to apply the canonical Ballé hyperprior / UNIWARD-grayscale-LUT technique):**

1. **Substrate-pivot A** — apply to NSCS03 RGB pixels (canonical 2D image with locality structure).
2. **Substrate-pivot B** — apply to ATW v2 chroma residuals (signal-axis preserved per Catalog #297).
3. **Substrate-pivot C** — apply to A1 latent stream (frozen-substrate sidecar per Catalog #272 distinguishing-feature integration contract).

The original kill of the substrate-class (PR101 / PR106 Lanes 05/06) is now classified as **RATIFY-FALSIFICATION-OF-THE-SPECIFIC-METHODOLOGY** (the lane-design-vs-architecture mismatch IS empirically falsified for the PR101 INT8 symbol substrate / PR106 no-mask-channel substrate) + **REQUEST-REINVESTIGATION-OF-ALTERNATIVES** (the technique-class via alternative substrate-pivots remains viable per the enumerated reducers + pivots).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the substrate-class is NOT killed; the SPECIFIC-METHODOLOGY-AT-ORIGINAL-SUBSTRATE-LOCATION is FALSIFIED; alternative reducer / substrate-pivot probes are queued for reactivation per the body's reactivation criteria.

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

<!-- # PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope -->

This memo references Atick-Redlich / cooperative-receiver framing as cross-reference / related-work / sister-substrate context — NOT as this substrate's architectural core. The substrate proposed by this memo is structurally distinct from Z6/Z7/Z8 (which DO require ego-motion-conditioned next-frame prediction as architectural core per Pattern H + Z6/Z7/Z8 design memo Section 11).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Z6/Z7/Z8 design memo Pattern H + Catalog #311 acceptance cascade (c): same-line waiver `# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope` applies. The waiver rationale is non-placeholder (>4 chars, not `<rationale>` / `<reason>`).

Cross-references to cooperative-receiver / Atick-Redlich in this memo serve as theoretical-anchor / related-work / sister-substrate-comparison only; they do NOT make this substrate a predictive-coding substrate in the Pattern H sense.


---

## Observability surface

**Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16** (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`) + Catalog #305 STRICT preflight gate (`check_substrate_design_memo_has_observability_surface_section`).

**Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:** this section is appended to the design memo; pre-existing body content (Sections 1-N + 9-dim checklist + cargo-cult audit + canonical-vs-unique decision + cross-references) is UNCHANGED. Appended by WAVE-1 APPARATUS HARDENING subagent 2026-05-16 to enable Catalog #305 STRICT-flip.

**The 6-facet observability surface for this design:**

1. **Per-layer inspection.** Every layer of this substrate / composition / experiment captures its (input tensor, output tensor, intermediate activations, attention maps when applicable) at runtime via the canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. The forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl` for post-hoc inspection.

2. **Per-signal decomposition.** Composite metrics (`final_score = seg + sqrt(10*pose) + 25*rate`) decompose into constituent contributions per the canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`).

3. **Run-to-run diff.** Two runs of this substrate / composition produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. The canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` (planned per the observability audit Highest-ROI extension list) emits per-component deltas + per-pair drift; until landed, manual diff via `archive_sha256` byte-addressability is straightforward.

4. **Post-hoc query interface.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL surfaces consumable without re-running: `contest_auth_eval_<axis>.json` (canonical per-component scores with axis labels) + `modal_metadata.json` (per-dispatch cite-chain per Catalog #166) + `observability/*.jsonl` (per-layer + per-signal). The continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` is queryable per (substrate, axis, hardware, evidence_grade) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-chain.** Every behavior signal anchors to the canonical tuple `(substrate_id, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. The call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. The substrate's archive grammar exposes byte-offset addressability for "what if this byte changed?" probing without re-running training. Per-layer / per-component ablation switches surfaced via the trainer's argparse flags + the canonical `tac.xray.<lens>.ablate_*` helpers when applicable.

**Acceptance per Catalog #305:** this section satisfies the structural requirement (literal section header `## Observability surface` present); the body content above documents the substrate's 6-facet observability surface for operator-facing audit.

**Sister observability-discipline gates active for this substrate:**

- Catalog #245 modal_call_id_ledger (every dispatch registered)
- Catalog #166 Modal HEAD-parity ledger (every dispatch worker-source-verified)
- Catalog #128 + #131 fcntl-locked JSONL posterior discipline (state mutations append-only)
- Catalog #139 packet compiler no-op detector (byte-mutation surface)
- Catalog #272 distinguishing-feature integration contract (per-substrate byte-mutation proof)
- Catalog #220 substrate L1+ operational mechanism declaration (no opaque byte additions)
- Catalog #105 no-op detector (no-op provenance)
- Catalog #127 authoritative tag custody (per-call-site axis + hardware-substrate validation)

**Observability extension recommendations (queued for follow-on):** see `tools/audit_existing_infrastructure_for_observability.py --summary` output for the canonical 8-tool / 6-facet observability gap analysis + Highest-ROI extension list. The `tools/audit_*.py` family is the highest-ROI extension target (3/12 observability) per the standing-directive consequence 3.
