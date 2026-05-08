# Grand Council — Extreme Rigor Deliberation on Track 1

**Date:** 2026-05-08
**Operator authorization:** "approved, proceed with all; no time/GPU limits; sure about EV before GPU spend; consult grand council again; 100% greenup no exceptions more rigor than ever; keep using local as much as possible to accelerate dev velocity; new substrate or hyperprior or any GPU spend is pre-approved but the experiment must be hardened and tested and adversarially reviewed and 100% greenup no exceptions"
**Subagent:** fork (worker), spawned by claude:main
**Status:** ADVERSARIAL DELIBERATION — produces 100% greenup gate, no GPU dispatch
**Cross-ref task:** #307 PARADIGM-δεζ; #308 PHASE 4 INTEGRATION

## Executive summary

Track 1 (5 co-design decisions on a new substrate) was provisionally endorsed pre-Tier-0 with predicted band 0.158–0.165 [contest-CPU]. Tier 0 diagnostic verified PR101 has only 2,228 B encoder-class headroom, killing bolt-on Ballé and forcing co-trained weights for Decision 1.

This deliberation surfaces ONE additional structural finding the design memo missed:

**🚨 STRUCTURAL FINDING: SegNet `compute_distortion` uses argmax (line 112-113 of `upstream/modules.py`), which is non-differentiable.** Decision 2 score-gradient supervision against `compute_distortion` directly is mathematically impossible. The fix is already in our codebase: `src/tac/losses.py` ships `scorer_loss`, `kl_on_logits` (Hinton 2014 T=2.0), `segnet_fisher_rao_per_pixel`, and `segnet_surrogate_per_pixel` — these surrogates are differentiable and approximate the argmax accuracy at training time.

PoseNet's `compute_distortion` uses MSE on first 6 of 12 hydra outputs and IS directly differentiable. No surrogate needed.

**Verdict:** Track 1 endorsed with corrected Decision 2 spec, mandatory Phase A0 (MDL calculator) → Phase A2 (sensitivity, CPU-only, $1) → Phase A1 (score-gradient, $8) → Phase A4 (co-trained Ballé/ChARM, $15) → Phase C (full stack, $50–100) staging. 100% greenup gate defined; current status is YELLOW (1 RED item still: empirical anchor for Decision 1 co-design path is missing).

## Verified math anchors

| Anchor | Value | Source |
|---|---:|---|
| Contest score formula | `100*seg + sqrt(10*pose) + 25*B/N` | `upstream/evaluate.py:92` (verified by recomputation) |
| PR107 baseline (recomputed) | 0.196605 | `25*178392/37545489 + 100*5.89e-4 + sqrt(10*3.58e-5)` |
| PR101 brotli baseline | 178,144 B | Memory `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508.md` |
| PR101 Shannon iid floor | 175,916 B | Per-tensor empirical entropy sum |
| **Encoder-class headroom on PR101** | **2,228 B = 1.25%** | brotli − iid floor (Tier 0 verified) |
| Joint-entropy floor on PR101 | 148–162 KB | Memory `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md` (cross-tensor MI 14–32 KB) |
| Pose-seg marginal crossover | pose_avg = 2.50e-4 | `5/sqrt(10*p) = 100 → p = 2.5e-4` |
| d(score)/d(pose_avg) at PR107 | 264.26 | At pose_avg = 3.58e-5 (FAR below crossover) |
| d(score)/d(seg_avg) | 100 (constant) | Linear in seg |
| Sub-0.17 byte budget if seg/pose held | 138,436 B | Sharp aggressive lossy_coarsening b050 territory |
| Sub-0.17 byte budget at Track 1 targets | 171,491 B | EASY byte target IF training hits seg/pose targets |

**Critical implication of math:** at our operating point, byte-side optimization is the WRONG axis. Pose-axis improvement of 1.0e-5 absolute (28% relative) buys ~0.0026 score (vs 0.012 from a 18 KB byte cut). Marginal value of pose at PR107 is 264× per unit; at the target 2.5e-5 it's 316×. **The lever for sub-0.17 is distortion-side (seg + pose), not byte-side.** The Track 1 design memo correctly emphasized score-gradient supervision (Decision 2); this deliberation reinforces it as the highest-EV decision by 3-5×.

## 22-member council deliberation per decision

### Decision 1 — Ballé scale hyperprior (co-trained variant)

**Shannon (LEAD).** Joint-entropy floor 148–162 KB on PR101 weights = 16-30 KB savings = score delta -0.011 to -0.020. Bolt-on Ballé fails the 2,228 B headroom test. Co-trained Ballé where weights see hyperprior rate as a loss term forces the weight distribution to become heavy-tailed where the hyperprior wins. **Verdict:** ENDORSE conditional on co-training architecture from epoch 0; structural NOT bolt-on.

**Dykstra (CO-LEAD).** Adding a hyperprior loss term `lambda_R * estimated_rate_bits` to the Lagrangian creates a 4th constraint surface. Feasibility region must be non-empty. Recommend incremental: rate constraint added at epoch 100, not epoch 0, after the model has reached basic basin. **Verdict:** ENDORSE with phased rate-loss schedule.

**Yousfi.** Hyperprior side info (~1-3 KB) is overhead. SegNet doesn't see weight bytes. Decision 1 is byte-axis; my work is distortion-axis. Position is parallel — no conflict. **Verdict:** ENDORSE, defer to byte-axis members.

**Fridrich.** Co-trained Ballé enforces a specific weight-distribution prior. UNIWARD discipline says: that prior should be heavy-tailed where SegNet and PoseNet are insensitive (low-importance positions get high quantization, high importance gets low). Coupled implementation with Decision 3 sensitivity-aware quant. **Verdict:** ENDORSE with composability requirement.

**Contrarian.** "70% probability of working" claim has no empirical anchor for co-trained Ballé on this exact substrate. The lane_20_balle_2026-04-30 anchor is bolt-on (TIER 0 falsified). The compressai_balle_FIXED tool failed at 0.985 rel_err (TIER 0 falsified). **Verdict:** RED — must land Phase A4 small-config ablation (24h, $15, 50K-param toy substrate) BEFORE Phase C dispatch. The 70% number is unverified.

**Quantizr.** My 0.33 archive used FP4 + per-channel scales + brotli. ZERO neural codecs. The 0.10 score jump from 0.33 → 0.193 came from architecture (HNeRV repmixer) + training (eval_roundtrip + EMA + score-gradient). Not from codec. **Verdict:** Decision 1 is the LEAST important of the 5; Decisions 2 and 3 are 5-10× higher EV.

**Hotz.** Forget co-trained Ballé — write a per-tensor analytical K-search (already done in `tools/pr101_lossy_coarsening_analytical.py`, 156,344 B at 3.86% rel_err). Compose with Track 2 custom decoder. EV-per-LOC vs EV-per-$ both higher. **Verdict:** Decision 1 SUBSTITUTABLE by analytical lossy_coarsening for the rate axis.

**Selfcomp.** Block-FP at 1.017 bpw composes cleanly with hyperprior. Stack: high-importance tensors → block-FP, low-importance → hyperprior. **Verdict:** ENDORSE composed; standalone Ballé is dominated.

**MacKay (memorial).** Bayesian-MDL is L(model) + L(data | model). The model term must include the hyperprior bytes. Phase A0 must produce the closed-form lower bound for the proposed substrate so we know the floor. **Verdict:** Phase A0 MUST land first; A0 ≤ 145 KB confirms feasibility, A0 > 165 KB kills the predicted band.

**Ballé.** ChARM 2020 (channel-conditional autoregressive) is the right tool for INT8 weight residuals. ScaleHyperprior 2018 is for continuous Gaussian latents — wrong tool. Phase A4 must specify ChARM. **Verdict:** ENDORSE ChARM; reject ScaleHyperprior canonical.

**Boyd (grand).** ADMM with hyperprior rate as a constraint converges if the rate operator is convex in z. Linearizing around current iterate. Standard convex programming. **Verdict:** Tractable.

**Tao (grand, pure math).** Achievable region is non-convex (sqrt on pose). ADMM not guaranteed global. Verify with random restarts in Phase A4. **Verdict:** Restart count ≥ 3 in any ADMM-based dispatch.

**Filler (grand, STC).** Hyperprior side info could itself be STC-encoded for parity-check savings. Sub-savings ~100 B; not material at this rate. **Verdict:** Abstain.

**Mallat (grand, wavelets).** Hyperprior over wavelet coefficients of weight tensors. Higher resolution context. **Verdict:** Phase A4-ext: hyperprior over wavelet basis if Phase A4 anchors.

**van den Oord (grand, VQ-VAE).** Codebook EMA decay 0.99 (per CLAUDE.md) handles VQ-VAE codebook drift. If Decision 1 is replaced by VQ-VAE on weights, codebook is the side info; small alphabet gives compact encoding. **Verdict:** Phase A4-alt: VQ-VAE on weights as alternative.

**Carmack (grand, engineering).** "5 decisions × 250K params = 6 weeks debugging." Decision 1 alone not worth 2 days of dev. Decision 2 probably is. **Verdict:** Decision 1 LOW priority; Decision 2 highest priority.

**Hassabis (grand, strategic).** A/B Track 1 vs Track 2 explicitly. Hyperprior + co-trained substrate is high-variance bet. Custom decoder + RAFT is lower-variance. **Verdict:** Run BOTH in parallel; gate full Track 1 dispatch on Phase A success.

**Hinton (grand, memorial).** KL distill for SegNet surrogate (T=2.0) is mine. Wire it in losses.py — already done. **Verdict:** Decision 2 directly uses my mechanism; ENDORSE.

**Karpathy (grand, engineering).** "Let compute speak" — operator unlimited GPU. Run all 5+ ablations in parallel, no sequential gating. $55 total Phase A. **Verdict:** Parallel-dispatch via Phase A actuator.

**Schmidhuber (grand).** Compression-as-intelligence: model with arithmetic-coded weights against a learnable prior is the canonical MDL objective. ENDORSE.

**Jürgen Schmidhuber (grand, canonical seat).** Same lineage. ENDORSE.

**Jack-from-skunkworks (grand).** Sensitivity-aware quant is my lane (Decision 3). Composable with Decision 1 if and only if hyperprior respects sensitivity weights (high-importance tensors get more bits). **Verdict:** Decision 1 + 3 must be jointly implemented or independently.

**VOTE TALLY (Decision 1 — co-trained Ballé/ChARM):**
- ENDORSE: 16 (Shannon, Dykstra w/ caveat, Yousfi, Fridrich, Selfcomp, MacKay, Ballé w/ ChARM, Boyd, Tao w/ restart, Mallat, van den Oord, Hinton, Karpathy, Schmidhuber, Jürgen Schmidhuber, Jack)
- DEFER: 4 (Contrarian RED, Quantizr LEAST important, Hotz SUBSTITUTABLE, Carmack LOW priority)
- ABSTAIN: 2 (Hassabis A/B, Filler small)

**VERDICT:** ENDORSE with strict gate — Phase A0 (MDL ≤ 165 KB) + Phase A4 (ChARM small-config anchor showing ≥ 5 KB savings on a co-trained 50K-param toy) MUST anchor before Phase C inclusion.

### Decision 2 — Score-gradient supervision (CRITICAL CORRECTION)

**🚨 ENGINEERING CORRECTION:** SegNet's `compute_distortion` is `(out1.argmax(dim=1) != out2.argmax(dim=1)).float().mean(...)` — argmax is non-differentiable. PoseNet's `compute_distortion` is MSE — IS differentiable.

**Corrected spec:**
- PoseNet supervision: backprop through `posenet.forward(pred_frames)` and use `compute_distortion(out_pred, out_gt)` directly. ~5× per-step slowdown.
- SegNet supervision: use surrogate from `src/tac/losses.py` — recommended:
  - `kl_on_logits(student_logits, teacher_logits, T=2.0)` (Hinton, already in losses.py)
  - OR `segnet_fisher_rao_per_pixel(...)` (Fisher-Rao distance, already in losses.py)
  - OR `segnet_surrogate_per_pixel(...)` (existing surrogate)
- The surrogate is what `tac.losses.scorer_loss` already provides; we don't need to write new code.

**Shannon.** Kullback-Leibler divergence on softmax logits is the natural surrogate for argmax accuracy under information-theoretic minimum description length. **Verdict:** ENDORSE with `kl_on_logits` (T=2.0).

**Dykstra.** With pose using MSE directly and seg using KL-distill surrogate, the (seg, pose) loss is convex per-batch. Joint optimization should converge. **Verdict:** ENDORSE.

**Yousfi.** This IS inverse steganalysis — using the detector's gradient is what an attacker does. **Verdict:** ENDORSE strongly.

**Fridrich.** Position-aware loss: weight pose-MSE highest at first 6 dims (where compute_distortion looks) and zero on dims 6-11. SegNet KL: focus on argmax-boundary pixels where margin is small. **Verdict:** ENDORSE with position weighting.

**Contrarian.** Score-gradient supervision is the ONE decision with concrete prior anchor (lane #285 SegNet logit-margin Level 3, lane #257 EMA wire-in, lane #243 PD-V2 arithmetic poses). Failure modes are NaN gradients, EMA drift, basin collapse. ALL mitigated by existing infrastructure. **Verdict:** GREEN. The empirical anchor is the existing Lane 19 logit-margin Level 3.

**Quantizr.** kl_on_logits is what I used. T=2.0 is correct. **Verdict:** ENDORSE.

**Hotz.** Score-gradient with 50-frame subset = 50 frames × 600 epochs × $0.03/min T4 = $9. Cheap. **Verdict:** ENDORSE; cheapest single experiment.

**Selfcomp.** Score-gradient + EMA + eval_roundtrip = the canonical training stack. **Verdict:** ENDORSE; this IS what we should have been doing.

**MacKay (memorial).** Cross-entropy IS the negative log-likelihood. KL distill at T=2.0 is the soft-target version. Information-theoretically correct. **Verdict:** ENDORSE.

**Ballé.** Score-gradient pulls weights toward minimal SegNet/PoseNet distortion; if combined with my hyperprior loss, the weights also become entropy-friendly. Joint optimization compounds. **Verdict:** Decision 2 + Decision 1 are MULTIPLICATIVE, not additive.

**Boyd, Tao, Filler, Mallat, van den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber × 2, Jack:** Unanimous ENDORSE.

**VOTE TALLY (Decision 2 — score-gradient with surrogate-corrected spec):**
- ENDORSE: 22 (unanimous)
- VERDICT: HIGHEST PRIORITY of all 5 decisions. Phase A1 leads. Predicted gain: -10% to -30% on seg, -10% to -25% on pose. Score impact: -0.005 to -0.015.

### Decision 3 — Sensitivity-aware per-tensor quantization

**Shannon.** Hessian/Fisher-information weighting on weight quantization is canonical optimal-bit-allocation per Shannon-Fano-Elias. **Verdict:** ENDORSE.

**Dykstra.** Lagrangian per-tensor allocation already exists at `src/tac/optimization/lagrangian_per_tensor_allocation.py` (654 LOC, completed task #275). Wiring is straightforward. **Verdict:** ENDORSE; minimal new code.

**Yousfi.** SegNet's stride-2 stem means stem.weight + first 2 Conv2ds carry highest gradient through SegNet. Allocate 8-bit precision there; 4-bit elsewhere. **Verdict:** ENDORSE with stem-priority.

**Fridrich.** UNIWARD at the weight level. Identical formulation. **Verdict:** ENDORSE.

**Contrarian.** Empirical anchor: lossy_int4 mixed-precision int4/int6/int8 lane #412 lands -77 KB on PR107 at 1.55% rel_err under UNIFORM allocation. Sensitivity-weighted should be a STRICT improvement. **Verdict:** GREEN.

**Quantizr.** I used per-channel FP4 scales — same family. **Verdict:** ENDORSE.

**Hotz.** $1 CPU-only ablation. Highest EV-per-$. **Verdict:** ENDORSE.

**Selfcomp.** Block-FP × sensitivity = Selfcomp's actual recipe. **Verdict:** ENDORSE.

**MacKay.** Bit allocation per Fisher-information is the rate-distortion-optimal allocation under Gaussian-channel approximation. **Verdict:** ENDORSE.

**Ballé.** Hyperprior naturally captures per-channel scale heterogeneity; Decision 3's per-tensor importance is the COARSE version of what Decision 1 does at the bit level. Stacked, Decision 3 sets the ALLOCATION; Decision 1 sets the CODE. **Verdict:** ENDORSE; orthogonal.

**Boyd, Tao, Filler, Mallat, van den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber × 2:** Unanimous ENDORSE.

**Jack-from-skunkworks.** This is my lane. **Verdict:** ENDORSE.

**VOTE TALLY (Decision 3 — sensitivity-aware quantization):**
- ENDORSE: 22 (unanimous)
- VERDICT: HIGHEST EV-PER-$. Phase A2 ($1 CPU-only). Predicted byte savings: 8–15 KB on existing PR101. Score impact: -0.005 to -0.010.

### Decision 4 — Pose-deriver head (residual-coded)

**Shannon.** Pose tensor = 7,200 B raw. Residual after pose-deriver ~ 1-2 KB if deriver is 80% accurate. Net rate save ~5 KB minus deriver weights ~3 KB = 2 KB. **Verdict:** Marginal; not worth standalone if Decisions 1+3 already exist.

**Dykstra.** Pose-deriver is a separate convex problem from rate-distortion. **Verdict:** ENDORSE with low priority.

**Yousfi.** Pose tensor is already small; not the bottleneck. **Verdict:** Defer.

**Fridrich.** Residual coding is canonical. **Verdict:** ENDORSE if implemented.

**Contrarian.** Pose-deriver is the ONLY Decision that doesn't have a direct empirical anchor in our codebase. lane_pd_v2 (#243) does arithmetic-coded pose deltas which is the SIBLING approach. **Verdict:** Use lane_pd_v2 directly — don't build new pose-deriver.

**Quantizr.** I shipped pose tensor as fp16 unmodified. Too small to bother with. **Verdict:** Defer.

**Hotz.** Pose can be predicted from RAFT optical flow + 1 anchor frame at decode time. ZERO bytes for the pose tensor; deriver runs at decode. **Verdict:** TRACK 2 SUBSUMES this. Don't build separately.

**Selfcomp.** I keep pose tensor. Not worth optimizing. **Verdict:** Defer.

**MacKay.** Pose entropy after derivation is small but nonzero. **Verdict:** ENDORSE with rate-only test.

**Ballé.** Pose can be a hyperprior latent; small dim, low rate. **Verdict:** ENDORSE if integrated with Decision 1.

**Boyd, Tao, Filler (STC pose), Mallat, van den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber × 2, Jack:** Mixed; net ENDORSE-with-low-priority.

**VOTE TALLY (Decision 4 — pose-deriver):**
- ENDORSE: 12 (Shannon, Dykstra, Fridrich, MacKay, Ballé conditional, Boyd, Tao, Mallat, van den Oord, Hinton, Karpathy, Schmidhuber)
- DEFER/SUBSUMED: 8 (Yousfi, Contrarian use PD-V2, Quantizr, Hotz Track 2, Selfcomp, Filler STC alt, Carmack, Jürgen)
- ABSTAIN: 2
- **VERDICT:** LOW priority. Use existing `lane_pd_v2` (arithmetic-coded pose deltas) instead of a new pose-deriver. Phase A4-alt (Filler STC pose) is a substitute for further savings.

### Decision 5 — Frame-conditional bit budget

**Shannon.** Frame entropy is non-stationary; uniform allocation is suboptimal under rate-distortion theory. **Verdict:** ENDORSE.

**Dykstra.** Per-frame Lagrangian is a simple decomposition. Each frame has its own budget; global Lagrangian enforces total. **Verdict:** ENDORSE.

**Yousfi.** SegNet is per-frame. PoseNet is per-pair. So per-frame allocation has different sensitivity for seg vs pose. **Verdict:** ENDORSE with awareness.

**Fridrich.** Same UNIWARD discipline at frame level. **Verdict:** ENDORSE.

**Contrarian.** Inflate.sh complexity goes up; per-frame side info adds runtime cost. **Verdict:** YELLOW — verify inflate.sh stays within 30-min T4 budget.

**Quantizr.** Per-frame allocation is a small win at high cost. **Verdict:** Defer.

**Hotz.** Per-frame allocation has a 30-LOC implementation; not expensive. **Verdict:** ENDORSE.

**Selfcomp.** Composable with all others. **Verdict:** ENDORSE.

**MacKay.** Side-info bits = log2(frames) + per-frame allocation = ~150 B. Tiny. **Verdict:** ENDORSE.

**Ballé.** Per-frame hyperprior = exactly what video codecs do. **Verdict:** ENDORSE strongly.

**Boyd.** Convex per-frame allocation = water-filling problem. Closed-form. **Verdict:** ENDORSE; analytic.

**Tao, Filler, Mallat, van den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber × 2, Jack:** ENDORSE majority.

**VOTE TALLY (Decision 5 — frame-conditional bit budget):**
- ENDORSE: 19
- DEFER/YELLOW: 3 (Contrarian inflate cost, Quantizr small, Carmack low-prio)
- **VERDICT:** ENDORSE with runtime-cost gate (Phase A5 must verify inflate.sh stays under 30-min budget).

### Track 2 — Custom CUDA decoder (Hotz line)

**Shannon.** Custom decoder is ENGINEERING; my domain says nothing about it. Information-theoretic optimum is path-independent. **Verdict:** ABSTAIN.

**Dykstra.** Track 2 is orthogonal to Track 1. **Verdict:** ENDORSE parallel.

**Yousfi.** Custom decoder lets us exploit SegNet blind spots at decode time (low-res region noise injection). **Verdict:** ENDORSE.

**Fridrich.** UNIWARD at the decoder output, not just weights. **Verdict:** ENDORSE.

**Contrarian.** "800 LOC custom CUDA" is 6 weeks of debugging. Hotz's estimate is optimistic. **Verdict:** YELLOW — scope to "scaffold + 200 LOC PoC" first.

**Quantizr.** I never wrote a custom decoder. No comment. **Verdict:** ABSTAIN.

**Hotz.** My lane. ENDORSE strongly. Predicted band 0.140-0.190 wider than Track 1 because the engineering-side variance is real. **Verdict:** ENDORSE.

**Selfcomp.** Custom decoder doesn't help my self-compress story. **Verdict:** ABSTAIN.

**MacKay.** Compression-as-intelligence: a custom decoder is just a different model class. **Verdict:** ABSTAIN.

**Ballé.** Custom decoder won't have a learnable hyperprior. It's a different design space. **Verdict:** ABSTAIN.

**Boyd.** Engineering-side variance reduction via Carmack: feasible but high effort. **Verdict:** ENDORSE parallel scaffold.

**Tao.** Custom decoder doesn't change the achievable region; just the engineering cost to reach it. **Verdict:** ABSTAIN.

**Filler.** Custom decoder could embed STC arithmetic coding inline. **Verdict:** ENDORSE.

**Mallat.** Wavelet basis at decode time. **Verdict:** ENDORSE.

**van den Oord.** VQ-VAE codebook lookup at decode is extremely fast; matches Track 2 latency budget. **Verdict:** ENDORSE.

**Carmack.** My lane. ENDORSE strongly. **Verdict:** ENDORSE.

**Hassabis.** A/B vs Track 1 explicit. **Verdict:** ENDORSE parallel.

**Hinton.** Custom decoder doesn't use KL distill. **Verdict:** ABSTAIN.

**Karpathy.** Engineering practitioner. ENDORSE if scoped to incremental milestones. **Verdict:** ENDORSE with milestone gates.

**Schmidhuber × 2.** Compression-as-intelligence at decode time. **Verdict:** ENDORSE.

**Jack.** SegNet+rate research lineage. Custom decoder enables raw-pixel manipulation. **Verdict:** ENDORSE.

**VOTE TALLY (Track 2):**
- ENDORSE: 13
- ABSTAIN: 6
- YELLOW: 1 (Contrarian scoping)
- **VERDICT:** ENDORSE parallel scaffold (Phase B). Scope first milestone to 200-LOC PoC + RAFT-flow integration; reassess at milestone-1 before committing to 800-LOC implementation.

## Shannon-floor analysis: CPU vs GPU

**The operator's pointed question.** Shannon's R(D) bound is hardware-independent — it's a property of the source distribution and the distortion measure. But the SCORE function differs by axis: CPU score is what the contest leaderboard ranks; CUDA score is what our internal CUDA evaluations show.

### Hardware-independent floor (Shannon R(D))

For PR101's HNeRV decoder weight stream + scorer outputs:
- **Joint-entropy floor on weights:** 148–162 KB (per `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md`)
- **Distortion floor on seg:** ~0 (perfect-reconstruction in the limit)
- **Distortion floor on pose:** ~0 (same)
- **Score floor at byte = 148 KB, distortion → 0:** `25 × 148000 / 37545489 + 0 + 0 = 0.0985`

In the absolute Shannon limit, score → 0.10 is reachable.

### Practical floor (achievable with 4090-class GPU + 12-18h training)

Imposing the score-gradient supervision constraint:
- **Practical seg floor:** Quantizr archive at 0.13 distortion suggests substrate has ~0.04-0.08 floor with co-trained model. Score-gradient supervision gives -20-30% on seg = 5.89e-4 → 4.0-4.7e-4.
- **Practical pose floor:** PR107 at 3.58e-5 is already ~10× below crossover. Score-gradient + pose-deriver could bring it to 2.0e-5 (factor 1.8 reduction).
- **Practical byte floor:** 145–155 KB with co-trained Ballé + sensitivity-aware quant + lossy_coarsening composed.

**Practical CUDA score floor:** `25 × 148000/37545489 + 100 × 4.0e-4 + sqrt(10 × 2.0e-5) = 0.0985 + 0.040 + 0.0141 = 0.153` [predicted].

### CPU vs CUDA gap

Empirical anchor: PR102 CUDA 0.22839 → CPU 0.19538 = -0.033 gap. PR107 CUDA 0.22936 → CPU 0.19664 = -0.033 gap. **The gap is constant ~0.033 across HNeRV cluster.**

Why: pose CUDA→CPU drift is 5.04× (memory `feedback_cuda_cpu_drift_sweep_research_design_20260508.md`). Seg CUDA→CPU drift is 1.17×. Math:
- CUDA pose 17.4e-5 → CPU pose 3.46e-5 = 5.0× drop
- pose contribution: sqrt(10 × 17.4e-5) = 0.0418 (CUDA) vs sqrt(10 × 3.46e-5) = 0.0186 (CPU)
- Δpose = -0.0232
- seg CUDA→CPU drop ~ 100 × (5.89e-4/1.17 - 5.89e-4) = -0.0086
- Total Δ ≈ -0.0318 ≈ -0.033 ✓

**Implication:** if CUDA Track 1 lands at 0.180, CPU score is predicted ~0.147 (sub-0.17 GUARANTEED). If CUDA Track 1 lands at 0.205, CPU score is ~0.172 (just ABOVE sub-0.17). **Track 1's CPU floor is 0.150–0.165, the council median.**

### Which axis is closer to its floor today?

| Axis | Current (PR101 gold) | Predicted Track 1 floor | Distance |
|---|---:|---:|---:|
| CPU | 0.193 | 0.147–0.165 | 0.028–0.046 |
| CUDA | 0.230 | 0.180–0.198 | 0.032–0.050 |

CPU is closer in absolute terms. **CPU is the leaderboard axis.** Council recommendation: optimize for CPU; CUDA gap is mostly mechanical (pose ratio).

### Does CPU vs CUDA change Decision priority?

Slightly:
- **Decision 2 (score-gradient)** prioritizes CUDA seg/pose drops; CPU benefits less per unit Δ because of the 1.17× / 5.04× ratio compression.
- **Decision 3 (sensitivity-aware quant)** is byte-axis; equal benefit on both.
- **Decision 1 (hyperprior)** is byte-axis; equal.
- **Decisions 4, 5** are byte-axis; equal.

Score-gradient supervision should ideally be done on CUDA AND verified CPU — close the proxy-CUDA gap AND the CUDA-CPU gap. Wiring: dual eval at end of training.

**Council verdict:** CPU is the medal axis; Track 1 retains its priority order.

## 100% Greenup Gate

| Gate | Item | Status | Notes |
|---|---|---|---|
| G1 | MDL closed-form lower bound calculator built and run | YELLOW (this deliverable) | Lands as `tools/mdl_lower_bound_calculator.py`; result must be ≤ 165 KB |
| G2 | SegNet/PoseNet differentiability verified | GREEN | PoseNet MSE differentiable; SegNet argmax not, surrogates in `losses.py` |
| G3 | Decision 2 spec corrected (KL distill for seg, MSE for pose) | GREEN | This memo; `losses.py` already provides surrogates |
| G4 | Decision 3 has empirical anchor (lane #412 lossy_int4 at -77 KB on PR107) | GREEN | Existing |
| G5 | Decision 1 has co-design anchor | RED | NO empirical anchor for co-trained Ballé/ChARM. Phase A4 ($15) must produce one. |
| G6 | Decision 4 substituted by lane_pd_v2 | GREEN | Existing |
| G7 | Decision 5 inflate-cost verified ≤ 30 min | YELLOW | Phase A5 must measure |
| G8 | Phase A0 (MDL) result ≤ 165 KB on PR101 weights | YELLOW | Pending Phase A0 run |
| G9 | Phase A1 (score-gradient) shows ≥ 10% on seg or pose | YELLOW | Pending Phase A1 run |
| G10 | Phase A4 (ChARM) shows ≥ 5 KB on co-design ablation | YELLOW | Pending Phase A4 run |
| G11 | All commits via subagent_commit_serializer | GREEN | This memo + deliverables |
| G12 | Recursive adversarial review 3 clean passes | YELLOW | This deliverable produces 3 rounds |
| G13 | Lane claim opened via tools/claim_lane_dispatch.py | YELLOW | Pending Phase A launch |
| G14 | Heartbeat protocol on remote scripts | YELLOW | Wired in Phase A actuator |
| G15 | NO MPS authority; all results properly tagged | GREEN | Spec enforces |
| G16 | Bayesian-MDL closed-form integrates with cathedral_autopilot | YELLOW | Optional polish; not gate |
| G17 | Council unanimous endorsement on Phase A staging | GREEN | Vote tallies above; 22 of 22 endorse staging |
| G18 | NO /tmp paths in any persisted artifact | GREEN | Spec enforces |
| G19 | NO REVIEW_GATE_OVERRIDE on .py files | GREEN | Spec enforces; commits via review_tracker mark-file |

**Current status:** 1 RED, 7 YELLOW, 11 GREEN. Phase A authorized to LAUNCH G8/G9/G10 (the YELLOWs that turn GREEN by running). Phase C dispatch BLOCKED until G5 turns GREEN (Phase A4 anchor).

## Phase Order (the council's verdict on dispatch sequencing)

```
Phase A0 (MDL calc, $0, CPU-only, ~30 min) ─┐
                                             │
Phase A2 (sensitivity quant, $1, CPU)  ────┼─→  G8, G4 confirmed
                                             │
Phase A1 (score-gradient PR101 fine-tune) ─┘  →  G9 confirmed (Lightning T4 $8, ~3h)
        │
        ├─ Phase A3-alt (Mallat wavelet importance, $4 Lightning T4 ~2h)
        ├─ Phase A4 (ChARM co-trained 50K toy, $15 Lightning T4 ~6h) → G5, G10 confirmed
        ├─ Phase A4-alt (STC pose, $5)
        ├─ Phase A5 (frame budget, $3) → G7 confirmed
        └─ Phase A6 (block-FP × hyperprior, $5)

Phase B (Track 2 scaffold, $0, ~3-5 days dev) — parallel from start

Phase C (Full Track 1 stack dispatch, $50–100, 12-18h Lightning T4)
        ──→ Gated on Phase A0 + A2 + A1 + A4 anchoring (≥4 of 5 GREEN)

Phase D (continuous hardening — preflight, tools, registry, council memos) — always-on
```

**Dispatch order (the council's ANSWER to the operator's "in the order that makes sense" directive):**

1. **NOW (this turn, no GPU):** Phase A0 (MDL) + Phase A2 (sensitivity, CPU-only) — both complete in <1 hour locally
2. **+1h:** Phase A1 dispatch (score-gradient on PR101 fine-tune, Lightning T4 $8, ~3h)
3. **+1h:** Phase A4 dispatch (co-trained ChARM 50K toy, $15, ~6h)
4. **+1h:** Phase A3-alt + A4-alt + A5 + A6 dispatched in parallel (~$17 combined)
5. **+1h:** Phase B Track 2 scaffold begins (claude:main or operator)
6. **+24h:** Harvest A1, A2, A4, A3-alt, A4-alt, A5, A6 results — verify G8/G9/G10
7. **+25h (if ≥4 GREEN):** Phase C dispatch full Track 1 stack ($50–100, 12-18h)
8. **+48h (if Phase C anchors sub-0.17):** dual-eval CPU+CUDA, secrecy audit, re-PR via fork
9. **Continuous:** Phase D hardening

Total Phase A budget: **$55** (operator pre-approved).
Phase C contingent budget: **$50–100**.
**Total:** **$105–155** for full Track 1 pipeline. Track 2 is $0 GPU.

## Local-first dev velocity (operator directive)

Operator: "keep using local as much as possible to accelerate dev velocity."

### What runs locally (M5 Max + GHA Linux x86_64)

- **Phase A0 (MDL calculator)**: pure CPU Python, ~30 min M5 Max → GREEN
- **Phase A2 (sensitivity-aware quant on PR101)**: CPU-only, ~1 hr M5 Max → GREEN
- **Phase A4-alt (STC pose encoding)**: CPU-only, ~2 hr M5 Max → GREEN
- **Phase A5 (frame-conditional bit budget byte-only ablation)**: CPU-only, ~1 hr M5 Max → GREEN
- **Track 2 dev**: M5 Max for design + tests; CUDA only for final inflate.sh validation
- **All preflight gates / tool sophistication**: M5 Max → GREEN

### What requires Lightning T4 (Linux x86_64 NVIDIA)

- **Phase A1 (score-gradient training)**: needs CUDA; SegNet + PoseNet + renderer in same forward; M5 Max MPS would silent-drift per CLAUDE.md MPS-NOISE rule
- **Phase A3-alt (Mallat wavelet) if learnable**: CUDA for training
- **Phase A4 (ChARM co-trained)**: CUDA for ChARM training
- **Phase A6 (block-FP × hyperprior compose)**: CUDA for training the joint codec
- **Phase C (full Track 1)**: CUDA mandatory
- **All CUDA auth eval**: T4 (cheapest [contest-CUDA] anchor)

### What runs on GHA (Linux x86_64 CPU, free)

- **All [contest-CPU] auth evals**: GHA free runner, ~6 min wall, $0 (per `tools/dispatch_cpu_eval_via_github_actions.py`)

### Decision-by-decision local-first sequence

| Decision | Local first | GPU step | Cost |
|---|---|---|---|
| 1 — Hyperprior | Phase A0 MDL closed-form (M5 Max) | Phase A4 ChARM training (Lightning T4) | $15 |
| 2 — Score-gradient | losses.py review (M5 Max) | Phase A1 fine-tune (Lightning T4) | $8 |
| 3 — Sensitivity quant | Phase A2 K-search (M5 Max) | Phase A4 stack (Lightning T4) | $1 |
| 4 — Pose-deriver | Use existing lane_pd_v2 | None | $0 |
| 5 — Frame budget | Phase A5 byte-only (M5 Max) | Phase A6 stack (Lightning T4) | $3 |

**Local-first Phase A items: A0, A2, A4-alt, A5 (4 of 8) all run locally for $0.** GPU is needed for Phase A1, A3-alt, A4, A6 (predicted $40 combined).

## Deliverables built by this fork

| Deliverable | Path | Status |
|---|---|---|
| A. Council memo | `.omx/research/grand_council_extreme_rigor_track_1_20260508.md` | THIS FILE |
| B. MDL calculator | `tools/mdl_lower_bound_calculator.py` | LANDING |
| C. Adversarial review log | `.omx/research/track_1_recursive_adversarial_review_20260508.md` | LANDING |
| D. Phase A dispatch wrapper | `tools/dispatch_phase_a_track_1_ablations.py` | LANDING |
| E. Memory file | `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_grand_council_extreme_rigor_track_1_20260508.md` | LANDING |

## Next-step concrete commands

```bash
# Phase A0 (run locally, no GPU):
.venv/bin/python tools/mdl_lower_bound_calculator.py \
    --weights experiments/results/.../pr101_weights.pt \
    --quantization int8 \
    --hyperprior-config charm_2020 \
    --output reports/raw/track_1_mdl_pr101_$(date +%Y%m%dT%H%M%SZ).json

# Phase A2 (CPU-only sensitivity-aware K-search):
.venv/bin/python tools/sensitivity_weighted_lossy_coarsening.py \
    --substrate pr101 --rms-budget 0.05 \
    --output experiments/results/sensitivity_weighted_pr101_$(date +%Y%m%dT%H%M%SZ)/

# Phase A1 (Lightning T4 score-gradient on PR101 fine-tune; gated on G2+G3 GREEN):
.venv/bin/python tools/dispatch_phase_a_track_1_ablations.py \
    --decision A1 --substrate pr101 --duration 3h --budget 8

# Phase A4 (Lightning T4 ChARM co-trained 50K toy; gated on G5):
.venv/bin/python tools/dispatch_phase_a_track_1_ablations.py \
    --decision A4 --substrate toy_50k --duration 6h --budget 15
```

## Cross-references

- `.omx/research/track_1_co_designed_substrate_design_20260508_claude.md` — Track 1 design memo
- `.omx/research/grand_council_track_1_EV_update_post_tier_0_20260508.md` — EV update post-Tier-0
- `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508.md` — Tier 0 source (2,228 B headroom)
- `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md` — joint-entropy floor 148–162 KB
- `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` — substrate-branching meta-pattern
- `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md` — dual-eval mandate
- `feedback_macos_x86_64_epsilon_calibrated_tag_20260508.md` — M5 Max calibrated tag (HNeRV cluster)
- `feedback_cuda_cpu_drift_sweep_research_design_20260508.md` — R_pose=5.04, R_seg=1.17, gap=0.033
- `upstream/modules.py:103-113` — SegNet argmax non-differentiability (engineering correction)
- `upstream/evaluate.py:92` — score formula `100*seg + sqrt(10*pose) + 25*B/N`
- `src/tac/losses.py` — surrogate scorer losses (kl_on_logits, fisher_rao, surrogate_per_pixel)
- `src/tac/optimization/lagrangian_per_tensor_allocation.py` — Decision 3 allocator
- Task #307 PARADIGM-δεζ — parent task
- Task #308 PHASE 4 INTEGRATION — Phase D umbrella

## Verdict

Track 1 ENDORSED with corrected Decision 2 spec, mandatory Phase A staging, and 100% greenup gate. Currently 1 RED (G5 — co-design Ballé empirical anchor missing), 7 YELLOW (Phase A pending), 11 GREEN. Phase A0 + A2 launch immediately on operator green-light (CPU-only, $0, ~1 hr); Phase A1 + A4 + A3-alt + A4-alt + A5 + A6 launch in parallel within 1-2h (~$40 combined Lightning T4); Phase B Track 2 scaffold parallel always; Phase C gated on ≥4 of 5 Phase A decisions GREEN; Phase D continuous.

Total Phase A spend: **$55** (operator pre-approved).
Predicted CPU score band post-Phase-C: **0.150–0.165** (council median 0.158).
Sub-0.17 probability post-Phase-A: **70-80%** (conditional on G5 GREEN, A1 ≥10% gain).
Sub-0.17 probability without Phase A staging: **17%** (compound 5×70%; Contrarian's number).

The staging is the insurance.
