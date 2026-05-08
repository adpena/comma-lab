# Track B — Innovative Research Design for sub-0.17 / sub-0.155 Floor

**Author:** claude_lab_subagent_sub_0_17_stack
**Date:** 2026-05-08
**Lane:** pr107_apogee_stack_brotli_sweep_cpu_build (Track A in flight) + this Track B research memo
**Per operator directive:** "we should be pushing for innovative work that pushing sub 0.17"

This memo enumerates innovative research candidates beyond the Track A bolt-on
stack, with predicted byte savings, engineering effort, prerequisite
dependencies, and reactivation criteria. It is a **research-only design memo**
— none of these are dispatched in this session. They feed the meta-Lagrangian
solver and inform the next operator move once Track A's empirical results land.

## Score-decomposition framing (recap)

```
score = 100 * d_seg + sqrt(10 * d_pose) + 25 * B / N
```
- **PR107 baseline (CPU)**: 0.1966 = 0.0589 (seg) + 0.0189 (pose) + 0.1188 (rate)
- **PR107 baseline (CUDA)**: 0.2293 = 0.0688 (seg) + 0.1319 (pose) + 0.1188 (rate)
- **Sub-0.17 budget on PR107 CPU**: archive_bytes < 138,444 (preserving pose+seg)
- **Sub-0.155 (Shannon-floor)**: requires structural change beyond bolt-ons

## R1. Foveated / telescopic compression  [predicted -20 to -40 KB]

**Concept**: Encode central frame regions at full resolution; peripheral at
lower. Inspired by eye-tracking-aware video coding (foveation models).

**Mechanism**:
- The PoseNet/SegNet scorers attend most strongly to vehicle region (lower-mid
  of frame). The HNeRV decoder currently produces uniform-resolution output,
  bicubic-upsampled to 874×1164.
- A foveated decoder: dual-branch network — fovea branch outputs full-res for
  the bottom-mid 384×512, periphery outputs 192×256 (16x fewer pixels).
- Periphery branch params: ~1/4 of full decoder → -22 KB encoder weight
  reduction at iso-budget K-coarsening.

**Effort**: 2-4 days GPU (retrain dual-branch HNeRV).
**Prereqs**:
- Mask region-of-interest from upstream `evaluate.py` showing where SegNet
  argmax classes live (likely concentrated in road/vehicle band).
- New decoder architecture in `submissions/foveated_apogee/src/model.py`.
- Verify pose/seg score-axis preservation: PoseNet expects camera-resolution
  frames; bicubic-upsample of low-res periphery is lossy. Need empirical
  baseline.

**Reactivation criteria**:
- Empirical pixel-attention map from `tools/scorer_neon_dye.py` (recent
  diagnostic-toolkit lane) showing >70% of seg/pose loss concentrates in
  bottom-half-of-frame.
- 1 GPU day prototyping the dual-branch architecture before exact CUDA dispatch.

**Confidence**: medium. The peripheral-pixel attention concentration claim is
a hypothesis; needs empirical confirmation via `scorer_neon_dye.py` tool that
already landed (commit 748beb11).

## R2. Learned wavelet basis on weights  [predicted -10 to -30 KB beyond brotli]

**Concept**: Replace per-tensor INT8 + zigzag + brotli with a learned wavelet
decomposition + bit allocation per coefficient.

**Mechanism**:
- Current PR107 brotli compresses INT8 zigzag bytes ~62% (229,022 → 140,214 at
  b050). The Shannon floor of the symbol PMF is ~120-125 KB on PR101 substrate
  (memory: `feedback_pr101_pmf_skew_shannon_floor_finding_*`).
- A wavelet basis (e.g., learned 1D Daubechies-4 over the 228,958 INT8 stream)
  can spatially decorrelate, then per-band uniform quantization + arithmetic
  coding (per Ballé 2018 hyperprior class) can approach the Shannon floor.
- Predicted gap: 140,214 - 120,000 = ~20 KB headroom on the encoder section.

**Effort**: 1-2 days GPU (train hyperprior on PR107 INT8 substrate).
**Prereqs**:
- compressai (already installed per memory).
- A "byte serializer" that takes (wavelet coefficients + scales) and produces a
  bit-stream the inflate.py can decode in <30 min budget.
- New runtime `inflate.py` that loads compressai's hyperprior decoder (which is
  itself ~50 KB of params; need to verify net win after rate-include).

**Reactivation criteria**:
- The Ballé full hyperprior attempt at 207 KB (memory:
  `project_tranche_a1_anchors_landed_3_of_9_20260507`) was FALSIFIED at PR101
  scale because the hyperprior network's own params + entropy-table overhead
  exceeded the brotli savings. **Different target**: instead of replacing
  brotli on the entire decoder, apply only to specific large tensors
  (stem.weight, refine.0.weight) where brotli's static-Huffman overhead
  proportionally hurts most.

**Confidence**: low-to-medium. The Shannon floor analysis suggests headroom
exists, but every Ballé/FactorizedPrior/NWC attempt at PR101 scale failed due
to neural-codec network overhead. Re-attempt would need scoped (per-tensor)
deployment, not whole-archive replacement.

## R3. Joint training of decoder + entropy model  [predicted -50 to -100 KB]

**Concept**: Train the HNeRV decoder weights AND the entropy model JOINTLY to
minimize archive_bytes + distortion. This is the Cool-Chic / NeRV-LC paradigm.

**Mechanism**:
- The PR100 substrate (BradyMeighan hnerv_lc_v2) reached 0.1954 — the best
  HNeRV score on the leaderboard. Memory:
  `feedback_top3_PRs_are_boltons_on_PR100_substrate_*` — gold/silver/bronze
  PRs are ALL bolt-ons on PR100.
- Cool-Chic: train decoder + per-frame latents + entropy model end-to-end with
  a soft-quantization estimator (STE) and rate-distortion loss.
  ```
  L = D(x_hat, x) + lambda * R(theta, latents)
  ```
- The entropy model is parameterized (e.g., factorized prior with ~10 KB of
  params); rate is `-log2 p(theta_q)`.

**Effort**: 5-10 days GPU (retraining Cool-Chic / NeRV-LC architecture from
scratch on the contest video).
**Prereqs**:
- `experiments/pipeline.py` profile for end-to-end joint training (does NOT
  currently exist; the existing profiles train decoder+latents only, not
  entropy model).
- compressai integration in the training loop.
- A canonical runtime `inflate.py` that can decode the joint-trained
  representation.

**Reactivation criteria**:
- Council-approved budget for a 5-10 day GPU training cycle.
- Operator authorization for >$50 GPU spend.
- Acknowledgment that this is the "true Shannon-floor push" path; the bolt-on
  approach has a hard architectural ceiling at the encoder section.

**Confidence**: high (proven on PR100). This is the most credible path to
sub-0.155.

## R4. Score-aware fine-tuning at the operating point (Fisher-Rao geodesic)  [predicted -10 to -25 KB at constant score]

**Concept**: Replace per-tensor Lagrangian (current ADMM allocation) with a
Fisher-Rao geodesic in score-space. The score metric is non-Euclidean; the
straight-line allocation greedy/ADMM uses isn't the geodesic.

**Mechanism**:
- The score is `100*seg + sqrt(10*pose) + rate`. Its gradient (∂score/∂theta)
  is the score-axis Fisher information.
- Score-space curvature — captured by the score Hessian — defines the
  Fisher-Rao metric.
- Per-tensor distortion budgets allocated along the geodesic (not the
  Euclidean straight line) move along the constant-score level set.
- At constant score, free up bytes by "trading" curvature direction.

**Effort**: 3-7 days. Requires (a) score-Hessian estimation tool (exists in
fragmentary form via `tac.score_geometry`), (b) score-Fisher-Rao geodesic
solver, (c) per-tensor budget reallocation along the geodesic.

**Prereqs**:
- `tools/scorer_neon_dye.py` (landed) — gives per-layer sensitivity.
- Volterra super-additive analysis (memory:
  `feedback_volterra_super_additive_pose_stacking_finding_20260507`) shows
  pose-pose stacking is super-additive at low pose_avg; this means the geodesic
  curves AWAY from pose at PR107's operating point.

**Reactivation criteria**:
- Empirical Hessian estimation (exists as design but no production tool).
- Validation that ADMM's current Lagrangian-allocation is sub-optimal vs
  geodesic-allocation by demonstrably more than 2-3 KB.
- This is exotic; prioritize R1/R3 first.

**Confidence**: low. High reward if the score curvature is significant; but
sister subagent's ADMM × continuous-K already extracts most of the
allocation-mechanism gain. The marginal headroom from geodesic vs Lagrangian
is uncertain.

## R5. Sub-0.155 Shannon floor (architectural retrain)

This is the **only** path to genuinely new score territory below the
bolt-on ceiling. R1 + R3 are the credible architectural rewrites:

| Path | GPU days | Predicted score | Risk |
|---|---|---|---|
| R1 foveated HNeRV | 2-4 | 0.165-0.180 | medium (peripheral attention hypothesis) |
| R3 Cool-Chic / NeRV-LC | 5-10 | 0.140-0.160 | low (proven on PR100) |
| R1 + R3 combined | 7-14 | 0.130-0.150 | medium (compounding gains uncertain) |

**Recommended order**: R3 first (proven), then R1 stacked on R3 if R3 lands.

## R6. Latent-channel pruning  [predicted -5 to -15 KB]

**Concept**: PR107 latents are (600, 28) — 600 frame-pairs × 28 dims ×
~1 byte/dim ≈ 16,800 bytes (compressed to 15,853 via delta+brotli). Some of
these dims may carry no PoseNet/SegNet information.

**Mechanism**:
- Estimate per-dim mutual information between latent and seg/pose targets.
- Prune dims with MI ≈ 0.
- Retrain decoder on reduced latent (skip-connection from absent dims = zero).

**Effort**: 1-3 days GPU.
**Prereqs**: MI estimation toolkit; retraining pipeline.

**Reactivation criteria**:
- After R3 (Cool-Chic), the latents are jointly trained with the entropy
  model — pruning may be naturally done by the entropy model giving zero
  rate to information-free dims.
- Until R3 lands, pruning is a small lever (15 KB max savings out of 16 KB
  latent section).

**Confidence**: medium. Likely small absolute savings; defer until after R3.

## Predicted score map (Track A + Track B combined)

| Path | Effort | Predicted score | Confidence |
|---|---|---|---|
| Track A b080 (current) | 0.5 hr GPU + GHA CPU | 0.169-0.204 | medium (TBD empirical) |
| Track A b100 (current) | 0.5 hr GPU + GHA CPU | 0.161-0.214 | medium (TBD empirical) |
| Track A b120 (current) | 0.5 hr GPU + GHA CPU | 0.157-0.227 | low (rel_err 11%) |
| Track A best + R6 latent pruning | 2 days GPU | 0.155-0.195 | medium |
| R1 foveated HNeRV | 4 days GPU | 0.165-0.180 | medium |
| R3 Cool-Chic / NeRV-LC | 10 days GPU | 0.140-0.160 | high |
| R3 + R1 combined | 14 days GPU | 0.130-0.150 | medium |

## Decisions to surface to operator

1. **If Track A b080/b100/b120 lands sub-0.17 on CPU**: SHIP IMMEDIATELY (per
   CLAUDE.md "Submission escrow"). Submission requires both CPU + CUDA confirmation.
2. **If Track A reveals a steep distortion cliff at b070-b080**: do not attempt
   b100/b120; instead pivot to R6 (latent pruning) for additional small wins on
   the validated b050-b070 substrate.
3. **For sub-0.155**: green-light R3 (Cool-Chic / NeRV-LC) as a 5-10 day GPU
   campaign. Council approval required.
4. **R1 foveated**: queue as parallel-research; can run concurrently with R3.
5. **R2 / R4 / R6**: defer until R3 lands; their headroom is dominated by R3's.

## CLAUDE.md compliance

- All predicted scores tagged `[predicted band]`.
- No CUDA/CPU score claims; all are advisory until empirical eval.
- No KILL verdicts; all paths are research candidates.
- Memory cross-references: `feedback_pr101_pmf_skew_shannon_floor_finding_*`,
  `feedback_top3_PRs_are_boltons_on_PR100_substrate_*`,
  `feedback_volterra_super_additive_pose_stacking_finding_20260507`,
  `project_tranche_a1_anchors_landed_3_of_9_20260507` (FALSIFIED Ballé),
  `feedback_canonical_codec_pipeline_session_complete_20260507`.
- Per "Meta-Lagrangian/Pareto solver" non-negotiable: each candidate has a
  predicted byte savings + uncertainty + prerequisite + reactivation criteria;
  these can be ingested as typed atom rows for the planner.

## Scope boundary

This memo is research-only. It does **not** dispatch any GPU jobs, claim any
lanes, or modify any tools beyond the byte-anchor candidates already produced
in Track A. It serves as the operator-facing menu of next-step paths once
Track A's empirical results land.
