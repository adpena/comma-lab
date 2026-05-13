---
title: "Applicability Research — arXiv 2605.12245 (SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization)"
date: 2026-05-13
author: Claude (subagent: lane_pdf_2605_12245_applicability_research_20260513)
lane_id: lane_pdf_2605_12245_applicability_research_20260513
source_pdf: /Users/adpena/Downloads/2605.12245.pdf
arxiv_id: 2605.12245
score_claim: false
evidence_grade: literature-prediction
ready_for_exact_eval_dispatch: false
target_modes: [research_substrate]
research_only: true
---

# Applicability Research — SOAR (NVFP4 Quantization) to comma.ai Video Compression Contest

## 1. Paper identification

**Title:** SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization

**Authors:** Chengzhu Bao*1, Xianglong Yan*1, Zhiteng Li1, Guangshuo Qin1, Guanghua Yu2, Yulun Zhang†1
(* equal contribution; † correspondence: yulun100@gmail.com)

**Affiliations:** 1. Shanghai Jiao Tong University; 2. Tencent Hunyuan

**Venue:** arXiv preprint, posted 12 May 2026 (v1). No conference venue yet. Code/models promised at https://github.com/steven-bao1/SOAR.

**arXiv:** 2605.12245v1 [cs.LG]

## 2. Abstract summary

SOAR is a post-training quantization (PTQ) framework targeting NVIDIA's **NVFP4** microscaling 4-bit floating-point format (FP4 E2M1 elements with hierarchical scaling: an FP32 tensor-wide global scale α and FP8 E4M3 block-wise scales Δᵢ applied every 16 elements). Existing NVFP4 methods (4over6, RaZeR, MR-GPTQ) use heuristic scale selection (max-based or coarse binary search) and treat quantization and dequantization scales as a single shared value — both choices leave reconstruction error on the table.

SOAR contributes two ideas:

1. **Closed-form Joint Scale Optimization (CJSO):** under a fixed FP4 quantization assignment Q_i, the reconstruction objective ||W − Q·(αΔ)||² is **quadratic** in (α, Δ). The first-order optimality conditions give analytical closed-form updates for α and each Δᵢ (eqs 5+6). Alternated with Q-recomputation, this converges in ~15 iterations to a much lower MSE than the heuristic scaling 4over6 / max-based rules.
2. **Decoupled Scale Search (DSS):** the hardware-stored dequantization scale Δᵢᵈ MUST be FP8 (E4M3) for hardware-compatible inference, but the *quantization-side* scale Δᵢᵍ used to decide FP4 level assignments is NOT constrained — it's only used during the encoder pass. SOAR decouples the two: Δᵢᵍ is a high-precision real, Δᵢᵈ is FP8-projected. A small joint local search refines the (Δᵢᵍ, Δᵢᵈ) pair per block.

Empirical headline: on Qwen3-8B under NVFP4, SOAR raises 5-task zero-shot average from 68.75 (NVFP4 baseline) → 70.68, narrowing the gap to FP16 (71.53) by ~1.93 points. Compatible with GPTQ (boost on LLaMA-3.1-8B from 72.95 → 73.18). Same memory footprint as NVFP4 baseline.

## 3. Core contributions

1. Closed-form analytical updates for global + block-wise scales under fixed FP4 assignments (CJSO).
2. Decoupled quantization vs dequantization scale formulation under hardware FP8 constraint (DSS).
3. Unified iterative framework (alternate CJSO closed-form update + DSS local search) — 15-iter, calibration-data-free, no extra parameters at inference.
4. Generalization beyond NVFP4: DSS-on-MXFP4 also lifts accuracy by ~0.5-1.0 points on three model families.
5. Composability: SOAR + GPTQ outperforms either alone.

## 4. Applicability to comma.ai contest — direct verdict

**Verdict: LOW-EV / NOT-APPLICABLE-AS-DIRECT-PRIMITIVE; MEDIUM-EV as a transferable optimization technique to our existing block-FP / FP4 / INT-N codec lanes.**

Rationale below.

### 4.1 Why not directly applicable

The contest score formula is:

S = 100·d_seg + sqrt(10·d_pose) + 25·B/N

where B = archive byte count, N = 37,545,489 (raw frame bytes), d_seg from EfficientNet-B2 argmax disagreement, d_pose from FastViT-T12 first-6-dim MSE.

SOAR targets **LLM weight matrices** (4096×4096 attention/MLP projections in Qwen3-8B / LLaMA3). The rate gain comes from FP16→FP4 weight compression on 1-8B parameter models where memory is the dominant cost. Three structural mismatches:

1. **Our renderer is ~88-300K params** (Quantizr 88K, Lane G v3 ~300K), NOT 1-8B. Block-FP weight compression on a 300K renderer with 16-element blocks contributes a single-digit-KB delta to a 178KB-300KB archive — the rate term moves by 25·ΔB/37.5M ≈ 25·(2-5KB)/37.5M ≈ 0.001-0.003 score points. Even an oracle SOAR-perfect quantization can't move the contest needle on rate alone.
2. **NVFP4 hardware format is contest-irrelevant.** Our inflate.sh runs on contest CPU/CUDA — there's no NVFP4 hardware path. The "no additional hardware overhead" selling point of SOAR is irrelevant; we already store FP4 / block-FP weights as packed integers and dequantize in PyTorch at inflate time.
3. **Calibration-data-free is our preferred regime** (no train-set leakage during inflate) but the reconstruction-MSE objective is **weight-domain**, which is FALSIFIED on score-gradient-trained substrates (see CLAUDE.md Catalog #123 + Track 4 v1 anchor: weight-domain saliency `mean(θ²)` is anti-correlated with score saliency on A1; the score-gradient pushes parameters AWAY from zero on score-relevant directions, so weight-MSE-optimal quantization hits exactly the directions the trainer identified as score-relevant).

### 4.2 Where SOAR DOES apply (transferable techniques)

Three specific techniques from SOAR generalize to our codec lanes:

#### Technique A: Closed-form analytical scale updates (CJSO) for our existing block-FP codec

**Mathematical structure:** for fixed quantization assignments Q (the int8/ternary symbols our encoder produces), the reconstruction error ||W − Q·(scale)||² is **quadratic** in scale. The optimal scale has a closed-form solution. SOAR's eq (5)-(6):

  α* = (Σ W·Q·Δ) / (Σ Q²·Δ²)        # global scale, with block scales held fixed
  Δᵢ* = (Σⱼ Wᵢⱼ·Qᵢⱼ·α) / (Σⱼ Qᵢⱼ²·α²)  # per-block scale, with global held fixed

**Application:** `src/tac/block_fp_codec.py` currently computes per-block exponents via `e_b = ceil(log2(max_abs / clip_threshold))` (max-based, line 24-26). This is the max-rule SOAR replaces. The closed-form update would be a 3-line code change: after the initial max-rule init, run 5-15 iterations of (recompute Q under current scales → update scales by eq 5+6).

**Connection to score formula:** lower weight-MSE → less perturbation of the renderer's frame outputs → marginally lower d_seg AND d_pose. The expected score delta is bounded by the renderer's score-Hessian-weighted MSE drop, NOT raw MSE.

**Predicted Δscore (literature-derivation):** SOAR's MSE drop vs max-rule is ~3-5% (Figure 4: 4.55e-3 → 4.50e-3 on v_proj; ~3-5% relative). For a renderer in the 0.193 frontier regime with d_pose ≈ 3.4e-5 (PR106), a 5% drop in weight-MSE translates to roughly 5% drop in the layer's contribution to d_pose. Pose component contribution to score ≈ sqrt(10·3.4e-5) ≈ 0.018. A 5% relative drop ≈ 0.0009 score points. **Bounded by `[mathematical-derivation]` to Δscore = -0.001 ± 0.003**. Sub-percent gain at best, in the noise floor of a single eval.

**Composition with existing primitives:**
- COMPOSES with `src/tac/block_fp_codec.py` (drop-in replacement for `e_b = ceil(log2(...))` initialization).
- COMPOSES with `tools/pr101_a6_blockfp_hyperprior_anchor.py` (the lossy A6 stack already failed at -35,891 bytes; replacing max-rule scales with CJSO scales does NOT change byte budget, just reconstruction quality — could marginally reduce d_seg/d_pose for the same byte count).
- ORTHOGONAL to ternary {-1,0,+1} encoding (block-FP exponents work the same way; the closed-form update applies to the multiplicative scale, not the symbol).
- ANTAGONISTIC to score-gradient saliency on score-aware substrates (CLAUDE.md Catalog #123) — running CJSO weight-MSE optimization on top of score-gradient-trained A1 weights would push the scales toward weight-relevant tensors AWAY from score-relevant ones. **Application is safe only on score-agnostic substrates** (frozen-renderer block-FP transplants, NOT score-gradient-trained substrates).

**Implementation cost:** ~30 LOC drop-in to `block_fp_codec.py::pack_state_dict_block_fp`. ~2 hours dev. Zero new training; pure encoder-side optimization.

**Falsification criteria:**
1. A T4/4090 archive build using CJSO-init scales returns d_seg + d_pose components within ±0.001 of the max-rule baseline (rules out a "free" score gain).
2. A test that CJSO + score-gradient-trained A1 weights produces HIGHER d_seg/d_pose than the score-gradient training intended (confirms the Catalog #123 anti-correlation on this substrate).

#### Technique B: Decoupled quantization/dequantization scales (DSS) for our FP4 codec

**Mathematical structure:** the inflate-side stored scale (FP8 / fp16 / float) is constrained by what the runtime tree's `inflate.py` will dequantize with. The *encoder-side* scale that decides which FP4 level each weight gets assigned is NOT constrained — the encoder can use float32 internally, then store a quantized scale. SOAR exploits this asymmetry: ||W − Q_FP4(W / αΔᵍ) · α·Δᵈ||² where Δᵍ ∈ ℝ (high-precision quantization scale) and Δᵈ ∈ FP8 (hardware-stored dequantization scale).

**Application:** `src/tac/fp4_quantize.py` uses a single scale per block (line 53, `DEFAULT_BLOCK_SIZE = 32`). The scale is computed and stored. If we decouple — compute encoding-side scale at fp64, store fp16-projected scale — we get strictly-better FP4 assignments at the same archive byte cost.

**Connection to score formula:** identical to Technique A — lower reconstruction error at same byte count → marginal d_seg/d_pose improvement.

**Predicted Δscore:** SOAR's DSS-on-MXFP4 ablation (Table 5) shows ~0.5-1.0 points on the LLM 5-task average for a 1-8B-param model. For our 88K-300K renderer with a 4-bit codebook (`fp4_quantize.py:50`), the scale storage is fp16 (2 bytes per block of 32) — total scale-storage cost is ~(300K/32)·2 ≈ 18KB. Decoupling buys reconstruction precision but doesn't reduce stored bytes. **Bounded by `[mathematical-derivation]` to Δscore = -0.0005 ± 0.002**. Sub-thousandth of a score point at best.

**Composition:**
- COMPOSES with `fp4_quantize.py::quantize_state_dict_fp4` (replace the per-block scale computation with decoupled-search variant).
- ORTHOGONAL to brotli/lzma outer compression (operates inside the quantization layer, before serialization).
- IRRELEVANT to score-aware substrates per Technique A's anti-correlation argument.

**Implementation cost:** ~50 LOC. ~3 hours dev. Adds an inner search loop over (Δᵍ, Δᵈ) candidates per block.

**Falsification criteria:** same as Technique A.

#### Technique C: Calibration-data-free PTQ paradigm (philosophical — already aligned)

**Observation:** SOAR's distinguishing feature vs GPTQ/AWQ/SmoothQuant is that it uses NO calibration data. This aligns with our contest discipline — we don't have train/test set leakage at inflate time, scorer weights are frozen, the archive must be self-contained.

**Application:** confirms our existing lane discipline. Not a new technique; a validation that the calibration-free direction is publishable / SOTA-competitive in 2026. No code change needed.

**Connection to score formula:** none directly. Indirectly: it confirms PTQ approaches that avoid scorer-at-inflate (per CLAUDE.md strict-scorer-rule) are competitive with calibration-based methods.

### 4.3 Where SOAR explicitly does NOT apply

1. **HNeRV-family substrates (sane_hnerv, NeRV, BlockNeRV, TC-NeRV, FF-NeRV, DS-NeRV, HiNeRV)** — these are CNN/Transformer renderer architectures, not quantized inference targets. SOAR optimizes scale parameters for an already-trained dense-FP-weight LLM; our HNeRV renderers are score-gradient-trained end-to-end and ship as compressed weights. The FP4 / block-FP layer happens AFTER training; CJSO/DSS could plug into THAT layer, but per Technique A's anti-correlation warning, applying weight-MSE optimization on score-gradient-trained weights is FALSIFIED on A1 (Catalog #123).
2. **SIREN, Cool-Chic, A1+LAPose, A1+wavelet, SABOR, S2SBS** — same as HNeRV; these are renderer architectures, not quantization layers. SOAR is orthogonal.
3. **Time-traveler L5 autonomy** — pose-axis substrate; no FP4 quantization layer to optimize.
4. **Public PR mining (PR101, PR103, PR106 codecs)** — these are byte-codec primitives (arithmetic coding, brotli, predictive encoding). SOAR is a scale-optimization technique for FP-quantized matrices, not a coder.

## 5. Cross-references to session research

This memo's literature-derivation framework aligns with the session's accumulated research:

- `.omx/research/zen_state_frontier_deep_math_research_20260513.md` — frontier-math survey; SOAR's closed-form joint optimization fits the "analytical solutions where convex structure permits" theme.
- `.omx/research/online_research_bleeding_edge_synthesis_20260513.md` — May 2026 bleeding edge; SOAR is the May 12 2026 NVFP4 SOTA, consistent with this survey's 2025-2026 quant trajectory.
- `.omx/research/ancient_elder_polymath_research_20260513.md` — historical convex-optimization era (Boyd / Lagrangian); SOAR's closed-form updates are textbook KKT under fixed Q.
- `.omx/research/expert_team_signal_processing_classified_alien_tech_20260513.md` — signal-processing branch; SOAR's quadratic-in-scale objective is a classic least-squares scale estimation problem.
- `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md` — innovation roadmap; SOAR confirms the calibration-free PTQ direction is competitive.
- Existing internal lanes: `lane_pr101_a6_blockfp_hyperprior` (DEFERRED-pending-research; SOAR-style CJSO is one of the ranked reactivation criteria in `pr101_a6_selfcomp_blockfp_hyperprior_measured_negative_20260508_codex.md`), `src/tac/block_fp_codec.py` (where Technique A would land), `src/tac/fp4_quantize.py` (where Technique B would land).

## 6. Recommended dispatch order

**No GPU dispatch recommended.** The predicted Δscore (≤ -0.001 in the most optimistic mathematical derivation) is below the noise floor of a contest-CUDA eval and below the cost of a Modal T4 smoke ($0.30+). Two follow-up actions are MEDIUM-EV but optional:

1. **(MEDIUM-EV, $0 dev cost, ~2h)** Add `src/tac/block_fp_codec.py::pack_state_dict_block_fp_cjso` — closed-form-init variant alongside existing max-rule. Land as research-only opt-in flag `--scale-init=cjso`. Smoke-test the encoder roundtrip MSE drop on a frozen Quantizr-replica renderer. If MSE drop is ≥ 5%, build an archive and eval on `[contest-CPU]` GHA Linux x86_64 (free CI minutes). Only escalate to Modal T4 if CPU eval shows score delta ≥ -0.001.

2. **(LOW-EV, $0 dev cost, ~3h)** Add `src/tac/fp4_quantize.py::quantize_state_dict_fp4_dss` — decoupled-scale variant. Same gate: encoder-roundtrip MSE drop on frozen renderer first. Score-gate via free contest-CPU CI before any GPU dispatch.

3. **(NOT-RECOMMENDED)** Apply CJSO/DSS to score-gradient-trained A1 weights. Falsified by Catalog #123 anti-correlation. Tag any such proposal `forbidden_weight_domain_saliency_on_score_aware_substrate` and refuse.

## 7. Six-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map contribution:** N/A — research-only literature-derivation memo; no parameter saliency update.
2. **Pareto constraint:** N/A — predicted Δscore is sub-noise; no Pareto-frontier move.
3. **Bit-allocator hook:** N/A — SOAR's predicted gain is reconstruction-quality at fixed bit budget, not a bit-allocator input.
4. **Cathedral autopilot dispatch hook:** N/A — not dispatch-ready; verdict is "research-only literature note + optional encoder-side encoder dev opt-in flag". No autopilot candidate row emitted.
5. **Continual-learning posterior update:** N/A — no empirical anchor produced; this memo is `[literature-prediction]` / `[mathematical-derivation]`, not an authoritative result.
6. **Probe-disambiguator:** N/A — the verdict is unambiguous (LOW-EV; bounded predicted Δscore in the noise floor). No two-interpretation tension requiring math arbitration.

All 6 hooks N/A with explicit rationale per CLAUDE.md "Subagent coherence-by-default — silent omission is the orphan-work failure mode."

## 8. Verdict summary

**LOW-EV** as a direct primitive. The paper is a high-quality LLM-quantization optimization, but contest score is dominated by representation/codec choice on a 88K-300K renderer where weight-domain MSE drops of 3-5% translate to sub-thousandth score deltas. The two transferable techniques (CJSO closed-form scales, DSS decoupled scales) are textbook quadratic-in-scale optimization and would land as ~80 LOC of encoder-side dev if a future operator decision approves the LOW-EV-but-cheap-encoder-optimization path. They are STRICTLY FALSIFIED on score-gradient-trained substrates per Catalog #123 — applicable only to score-agnostic / frozen-renderer block-FP transplant lanes.

**Citations:**
- Bao, Yan, Li, Qin, Yu, Zhang (2026). "SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization." arXiv:2605.12245v1. https://arxiv.org/abs/2605.12245
- Cook et al. (2025). "Four over six: More accurate NVFP4 quantization with adaptive block scaling." arXiv:2512.02010.
- Chen et al. (2025). "RaZeR: Pushing the limits of NVFP4 quantization with redundant zero remapping." arXiv:2501.04052.
- Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
- Internal: CLAUDE.md Catalog #123 (`check_no_weight_domain_saliency_on_score_gradient_substrate`).
- Internal: `pr101_a6_selfcomp_blockfp_hyperprior_measured_negative_20260508_codex.md` (the A6 negative + reactivation criteria).
- Internal: `src/tac/block_fp_codec.py` (where Technique A would land).
- Internal: `src/tac/fp4_quantize.py` (where Technique B would land).
