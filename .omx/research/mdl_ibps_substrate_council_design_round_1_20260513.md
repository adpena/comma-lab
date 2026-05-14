---
title: E4 MDL-IBPS substrate — council-grade architecture design Round 1
date: 2026-05-13
lane_id: lane_mdl_ibps_substrate_20260513
status: research-only design memo (no archive bytes; no GPU dispatch)
score_claim: false
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true
evidence_axes:
  - first-principles-bound
  - mathematical-derivation
  - council-position
hnerv_parity_audit: design-time only (8 fields enumerated below; archive grammar declared BEFORE training script)
---

# E4 MDL-IBPS — Council-grade architecture design Round 1

## TL;DR

Per CLAUDE.md "Design decisions — non-negotiable" + "Council conduct — non-negotiable", the inner-ten council has deliberated the **E4 MDL-Optimal Procedural-plus-Patches with Information-Bottleneck Pose Sidecar** substrate proposed by `feedback_zen_state_frontier_deep_math_research_landed_20260513.md` (Domain 4 × Domain 9 × Council F O5 × LA-Pose).

**Verdict**: **ARCHITECTURE-CONTESTED** — 6/10 council members vote PROCEED-TO-L0-SCAFFOLD with explicit reservations; 3/10 vote DEFER-PENDING-EXISTENCE-PROOF; 1/10 (Contrarian) issues SUPER-VETO-ON-AS-PROPOSED with a constructive re-scope.

The architecture as proposed bundles two structurally distinct ideas (procedural baseline; IB pose sidecar). Council recommends operator routing on the question of whether to ship them as **one composite substrate** (zen E4 as written) or as **two sequenced lanes** (E4-A procedural-only at L0; E4-B IB sidecar at L0; later compose).

Predicted ΔS band (design-time): **[0.115, 0.165]** from PR101 0.193 minus zen prediction {-0.030, -0.080}. **Confidence: LOW** until a 1-frame existence-proof of the procedural baseline matching ≥ 50% of SegNet's argmax in ≤ 10 KB lands. Contrarian's pessimistic counter-band is **[0.180, 0.205]** if the procedural prior fails to enter E(V_GT).

L0 scaffold deferred this commit-batch pending operator decision on **decomposition** (composite vs sequenced); sister-subagent infrastructure for `tools/probe_mdl_ibps_disambiguator.py` is documented per CLAUDE.md "Anti-arbitrariness primitive".

---

## 1. Architecture proposal restated

Per `zen_state_frontier_deep_math_research_20260513.md` §10 E4:

1. **Procedural renderer** (5-10 KB Python): ego-motion-driven viewport + road-plane texture + parametric obstacles. Output: a baseline RGB stream `R_proc(t)` deterministic from a tiny parameter set `θ_proc`.
2. **IB pose sidecar** (Tishby 1999): minimum-rate compressed pose representation `T = q(X)` such that `I(T; Y_scorer) ≥ I_target` while `I(X; T)` is minimized. Y = SegNet/PoseNet outputs.
3. **Residual patch stream** `R_patch(t) = R_GT(t) - R_proc(t)` encoded via a small HNeRV-like decoder applied selectively to high-residual regions.

Composite contest archive:
```
0.bin = procedural_params_bytes (~5-10 KB)
      ‖ ib_sidecar_bytes        (~5-15 KB)
      ‖ residual_decoder_bytes  (~30-60 KB)
      ‖ residual_latents_bytes  (~10-25 KB)
      ‖ meta_json_bytes         (~0.5 KB)
                                  ----------
total target                       ~50-115 KB  vs PR101 178 KB
```

Predicted ΔS = -0.030 to -0.080 per zen memo §10 E4.

---

## 2. Council positions (Round 1)

### Shannon LEAD (rate-distortion grounding)

The proposal is the natural extension of the cooperative-receiver-conditional-entropy framework (sister memo `feedback_expert_team_signal_processing_alien_tech_landed_20260513.md` §N3). The procedural baseline encodes `H(V_GT | scorer-equivalence-class-membership)` rather than `H(V_GT)` — that is the correct rate-distortion floor for a contest with KNOWN scorer.

The IB sidecar is a SEPARATE rate-distortion claim: it bounds the pose-residual rate by `R_IB(D_pose) = min I(X; T) s.t. I(T; Y) ≥ I_max - D_pose`. For the PR106 r2 operating point (pose_avg = 3.4e-5, marginal sensitivity 271×), even a 5 KB pose sidecar that reduces residual pose distortion by 20% buys -0.005 to -0.012 score independent of the procedural rate gain.

**The two ideas multiply in EV but ADD in risk.** I endorse PROCEED-TO-L0 on the condition that the design memo enumerate the rate-distortion bound for EACH component separately and sum them under explicit additivity assumptions. The composite ΔS prediction is a Pareto-cone WALK, not a sum.

**ΔS prediction**: -0.020 to -0.060 (NARROWER than zen because I'm subtracting the procedural-baseline existence risk).
**Confidence**: MEDIUM.
**Vote**: PROCEED-TO-L0-SCAFFOLD with rate-distortion-bound section in the design memo.

---

### Dykstra CO-LEAD (feasibility region)

The achievable region is a 4-dimensional convex set: `{(R_proc + R_ib + R_patch, d_seg, d_pose, T_train) : R_total ≤ B, d_seg ≤ S_seg, d_pose ≤ S_pose, T_train ≤ T_max}`. The intersection-of-feasible-sets approach (Dykstra alternating projections) computes the achievable Pareto frontier point-by-point.

The CRUCIAL question for E4 is whether `(R_proc + R_ib + R_patch, d_seg, d_pose)` is convexly dominated by `(R_PR101, d_seg_PR101, d_pose_PR101)`. Convex domination would mean E4 is strictly Pareto-superior at SOME budget; absence of domination means E4 is only superior in a BAND of byte budgets.

**My prediction**: E4 dominates PR101 in the **80-130 KB band** (where procedural overhead pays off and HNeRV's flat-prior latent is wasteful). Below 80 KB, the procedural baseline alone is too lossy. Above 130 KB, HNeRV-only with more latents catches up.

This is exactly the Pareto-frontier-walk Shannon described. Operationally: **the operator should NOT compare E4-at-100 KB to PR101-at-178 KB and conclude superiority**. Compare at the same byte budget OR run both at multiple points and overlay the convex hull.

**ΔS prediction**: -0.025 to -0.055 in the 80-120 KB band; -0.005 to -0.015 outside it.
**Confidence**: MEDIUM-HIGH (the Pareto-walk argument is robust; the band locations are predictions).
**Vote**: PROCEED-TO-L0-SCAFFOLD with explicit Pareto-cone evaluation at multiple byte budgets, NOT a single point.

---

### Yousfi (contest-compliance + archive grammar)

The archive grammar is admissible per CLAUDE.md HNeRV parity discipline lessons 2-4: monolithic `0.bin` with declared offsets `[procedural_params, ib_sidecar, residual_decoder, residual_latents, meta_json]`. Inflate runtime would call:

```
inflate(archive_bytes, file_list):
    proc, ib, dec_sd, latents, meta = parse_archive(archive_bytes)
    for each video in file_list:
        R_proc = procedural_render(proc)
        R_patch = hnerv_decode(dec_sd, latents)   # BILINEAR-skip allowed
        R = R_proc + R_patch
        write_video(R, output_dir/<video>.mkv)
```

Inflate.py LOC budget: ≤ 100 LOC per L4. The procedural renderer adds ~30 LOC; HNeRV decode is ~70 LOC. **Within budget if the procedural code is genuinely tiny.** If procedural needs >50 LOC, we are over budget and the substrate-engineering exception (L7) applies.

**Critical compliance question**: does the inflate-time procedural renderer load PoseNet or SegNet? **Per the proposal — NO**, the procedural model is parametric (ego-motion + road texture + obstacles) and runs without scorer access. The IB sidecar at INFLATE time is just T → R-patch-conditional bytes; it does NOT load the scorer.

**Adversarial concern**: the IB ENCODE-time path (training) requires the scorer to compute I(T; Y). This is a TRAINING-ONLY scorer load — same pattern as PR101 score-aware loss — and therefore compliant.

**ΔS prediction**: I do not predict ΔS; I confirm compliance.
**Vote**: PROCEED-TO-L0-SCAFFOLD with archive grammar declared in the design memo (8 Catalog #124 fields below).

---

### Fridrich (steganalysis-adversarial robustness)

The procedural baseline is a STRUCTURED prior. The contest scorer (SegNet EfficientNet-B2 + PoseNet FastViT-T12) is trained on Comma2k19 dashcam video. The procedural model encodes the COMMON STRUCTURE of dashcam video (forward camera, road plane, ego-motion). This is the same insight as my UNIWARD steganography: errors in textured / structured regions are undetectable.

**The procedural baseline EXPLOITS the structured-error blindspot.** Pixels reconstructed via the procedural prior are STRUCTURED in exactly the way SegNet's training data was structured. Therefore SegNet's argmax on procedural-baseline pixels should agree with its argmax on GT pixels at high rate.

**Risk**: the procedural model has finite expressive power. If the GT video contains an OUT-OF-DISTRIBUTION event (sudden obstacle, weather change, unusual lighting), the procedural baseline produces a high-residual frame and the residual patch must absorb it. This is architecturally fine; it just shifts bytes from procedural to patch.

**Operational note**: the residual patches should be ENTROPY-CODED with a Yousfi-style detector-informed embedding cost (Allerton 2022). Per-pixel cost = inverse local SegNet/PoseNet sensitivity. Combined with my UNIWARD-style locality cost.

**ΔS prediction**: -0.010 to -0.040 from procedural-baseline alone via SegNet structural agreement; -0.005 to -0.015 from detector-informed residual entropy coding. Total -0.015 to -0.055 — **CONSISTENT WITH ZEN UPPER END**.
**Confidence**: MEDIUM-HIGH on the Yousfi-UNIWARD analog argument; LOW on the detector-informed coding term until empirical.
**Vote**: PROCEED-TO-L0-SCAFFOLD.

---

### Contrarian (REASONS the predicted -0.030 to -0.080 won't materialize)

I am invoking SUPER-VETO-AS-PROPOSED. The composite substrate as described is a **kitchen-sink anti-pattern at the substrate level** per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first":

1. **The procedural baseline is PR105's ghost.** PR105 was 1776 LOC across 21 files and LOST to rem2's 241 LOC silver. The procedural model + IB sidecar + residual patch + meta JSON is structurally similar — three independent pieces, each with its own training signal, each with its own potential to fail in a 30-second council review.

2. **The procedural-baseline existence proof is missing.** Before any L1 build, someone needs to demonstrate that hand-crafting a 5-10 KB ego-motion model on ONE GT video frame achieves ≥ 50% SegNet argmax match. Without this, the Solomonoff-MDL argument is purely theoretical. The zen memo §10 E4 falsification criterion (procedural-only beats HNeRV by <40 KB at same score, MPPA hypothesis weakened) requires a FULL training run; a 1-frame existence-proof is cheaper and falsifies faster.

3. **The IB-objective is non-trivial to optimize in the training inner loop.** Tishby's IB requires `I(X; T)` (joint entropy of input X and bottleneck T) and `I(T; Y)` (joint entropy of T and scorer outputs). Both terms require either a variational bound (Alemi 2016, beta-VAE-style) or Monte Carlo estimation. Adding a non-convex inner-loop term to the training objective on top of SegNet+PoseNet+rate is asking for optimization instability.

4. **The PR106 r2 falsification cluster.** Per `adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md`, three PR106 sidecar variants (c3, wavelet, cool_chic) failed CPU axis at +0.022 vs CUDA. The sidecar pattern itself is empirically fragile; adding a procedural baseline + IB optimization to a sidecar architecture compounds the fragility.

5. **The -0.030 to -0.080 ΔS band is THE LARGEST IN THE MEMO**. Per CLAUDE.md "Council conduct" — *"Unanimous votes should be scrutinized. If all five members agree instantly, someone isn't thinking hard enough."* The largest claimed ΔS deserves the MOST scrutiny, not the least.

**CONSTRUCTIVE RE-SCOPE**: split E4 into TWO sequenced lanes, each with its own existence-proof gate:

- **E4-A: procedural-baseline-only** at L0 SCAFFOLD. Existence proof = 1-frame hand-craft + SegNet argmax measurement on macOS-CPU advisory. Total cost $0. Falsification gate: < 40% SegNet argmax match at 10 KB → DEFER-pending-architecture-revision.
- **E4-B: IB pose sidecar** at L0 SCAFFOLD INDEPENDENTLY. Existence proof = synthetic-pose-data IB optimization measuring I(X;T) vs I(T;Y) curve (Alemi-2016 reproduction on toy data). Total cost $0. Falsification gate: cannot achieve I(T;Y)/I(X;T) > 0.5 → DEFER-pending-IB-objective-redesign.

Composition (E4 as written) only after BOTH existence proofs land.

**Pessimistic ΔS prediction (composite as proposed)**: -0.005 to +0.012 (i.e. it could MAKE THE SCORE WORSE) if the procedural baseline fails to enter E(V_GT) and the residual patches end up replicating PR101 latent overhead PLUS procedural overhead.
**Pessimistic floor**: 0.205 (worse than PR101 0.193).
**Vote**: SUPER-VETO-AS-PROPOSED (composite substrate). VOTE-PROCEED-TO-L0 on the SEQUENCED RE-SCOPE (E4-A first, E4-B second).

---

### Quantizr (vs PR101 substrate; replacement or sidecar?)

PR101 (the 0.193 [contest-CUDA] frontier we're chasing) is a single monolithic HNeRV decoder + 600 latents. Total 178 KB. The latent-rate is ~25 KB; the decoder-rate is ~150 KB. The decoder is overwhelmingly dominant.

**E4's claim**: replace ~80 KB of decoder bytes with ~10 KB of procedural code. That's an 8x compression on the decoder budget IF the procedural model actually replaces what the decoder was doing.

**My empirical priorr**: HNeRV decoders learn a SCENE PRIOR (road geometry, sky, vehicles) implicitly via gradient descent. A procedural model encodes this prior EXPLICITLY. The question is whether explicit encoding captures the same information density as gradient-descent-learned encoding.

**My read**: a well-designed procedural model captures **80-90% of the scene structure but 30-50% of the texture detail**. The texture detail is what the residual patches cover. So the byte budget should be procedural ~10 KB + residual ~50-80 KB ≈ 60-90 KB total — a 50% compression vs PR101 178 KB.

ΔS from rate alone: 80 KB × 25/N ≈ -0.054 (where N = 37,545,489 video bytes).

**HOWEVER** — the residual patches must also reach the SAME (d_seg, d_pose) PR101 reaches. If the procedural baseline introduces ≥ 5% extra distortion that the residual must absorb, the residual budget grows. If it grows to >80 KB the rate gain shrinks.

**My prediction**: composite is REPLACEMENT (not sidecar to PR101). It will not stack with PR101; it competes with PR101.

**ΔS prediction**: -0.020 to -0.045 net (rate gain -0.054 minus distortion penalty 0.005-0.025). Centroid: -0.032.
**Confidence**: MEDIUM.
**Vote**: PROCEED-TO-L0-SCAFFOLD on the SEQUENCED RE-SCOPE per Contrarian.

---

### Hotz (engineering rigor; IB tractability in training inner loop)

The IB objective is theoretically beautiful and engineering-fragile. Alemi 2016 (DeepIB) gave us VIB — a variational bound that's tractable in PyTorch:

```
L_VIB = E_x[E_t~q(t|x)[-log p(y|t)] + β · KL(q(t|x) || p(t))]
```

This is just a beta-VAE with the prediction head. **Engineering effort**: ~50 LOC for the VIB encoder/decoder; 1 day to wire into the training loop. Tractable.

**The harder engineering question is the procedural renderer.** A 5-10 KB ego-motion model that renders road + sky + obstacles in a way that achieves 50%+ SegNet argmax match is NOT a 1-day build. Realistic dev cost: 5-7 days for a research-quality procedural renderer that can be trained end-to-end with the residual.

**Critical engineering choice**: is the procedural model DIFFERENTIABLE? If yes, we can backprop SegNet+PoseNet through it and jointly train procedural-params + IB sidecar + residual patches under one objective. If no, the procedural params are HAND-TUNED and the residual absorbs all the slack.

I recommend: **build the procedural model as a differentiable nn.Module from byte one**. Even if it's hand-initialized, gradient flow lets us co-train.

**Engineering-ready time**: 5-7 days for a council-clean L1 implementation. **Budget**: $0 (CPU prototype is sufficient; GPU only for end-to-end training).

**ΔS prediction**: I defer to Shannon/Quantizr/Fridrich. My prediction is conditional on the engineering landing cleanly: **if it lands, -0.025 to -0.055; if procedural is hand-tuned (non-differentiable), -0.005 to -0.020**.
**Confidence**: MEDIUM on engineering-ready; HIGH on the differentiable-vs-hand-tuned axis.
**Vote**: PROCEED-TO-L0-SCAFFOLD with EXPLICIT differentiable-procedural requirement in the design.

---

### Selfcomp (archive byte-anatomy; encoding into bytes)

The procedural prior + IB bottleneck encodes naturally into the **block-FP weight self-compression** paradigm I championed in PR #56. Procedural params are typed: pose spline coefficients (float32 → block-FP4 with 1.017 bpw), texture LUT (uint8 → arithmetic-coded), obstacle parameters (mixed types). The IB sidecar T is by construction LOW-DIMENSIONAL — naturally amenable to block-FP4 or even ternary quantization (Council Z-1 EUREKA E2 if we go aggressive).

**Byte anatomy estimate**:
```
procedural_params:
  ego_motion_spline (8 control pts × 6 DoF × 4 bytes)            =   192 B
  road_texture_LUT (16 × 16 × 3 RGB × 1 byte)                    =   768 B
  obstacle_priors (10 obstacles × 8 params × 4 bytes)            =   320 B
  block-FP overhead (codebook + scales)                          =   200 B
                                                                   ------
                                                                   1,480 B  (~1.5 KB)

ib_sidecar:
  600 pairs × T_dim=8 × 1 byte (8-bit quant)                     = 4,800 B
  IB decoder weights (200 params × FP4)                          =   100 B
  arithmetic coder side-info                                     =   200 B
                                                                   ------
                                                                   5,100 B  (~5 KB)

residual_decoder:
  HNeRV-lite decoder (60K params × FP4)                          = 30,000 B
  brotli compression (30% overhead)                              =  9,000 B
                                                                   ------
                                                                  39,000 B  (~39 KB)

residual_latents:
  600 pairs × 12 floats × 2 bytes (int16)                        = 14,400 B
  arithmetic-coded                                               = 12,000 B
                                                                   ------
                                                                  12,000 B  (~12 KB)

meta_json + offsets                                              =    400 B
                                                                   ------
TOTAL                                                              57,980 B  (~58 KB)
```

**Predicted rate axis**: 25 × 58000 / 37545489 = **0.0386**.

PR101 rate axis: 25 × 178000 / 37545489 = **0.119**.

**ΔS from rate alone**: -0.080. **CONSISTENT WITH ZEN UPPER END**.

**This assumes (d_seg, d_pose) are unchanged from PR101**. If the procedural baseline introduces extra distortion δ_seg ≈ 0.005 and δ_pose ≈ 5e-5, ΔS_distortion ≈ +0.010 + 0.0035 ≈ +0.013. Net: -0.080 + 0.013 = **-0.067**.

**ΔS prediction**: -0.040 to -0.070. **STRONGLY CONSISTENT WITH ZEN UPPER END**.
**Confidence**: MEDIUM-HIGH on the byte budget; MEDIUM on the distortion estimate.
**Vote**: PROCEED-TO-L0-SCAFFOLD. The byte arithmetic checks.

---

### MacKay (memorial seat — information-theory consistency)

The MDL accounting closes IF AND ONLY IF the procedural-baseline residual entropy is correctly bounded.

`L(model) = K(procedural_params) + K(ib_sidecar) + K(residual | procedural, ib_sidecar)`

The third term is the SUBTLE one. Standard MDL accounting assumes the residual is encoded against a NULL prior. But here, the procedural baseline IS the prior, so the residual encoding rate is `H(R_GT - R_proc | R_proc, IB_sidecar)`.

**Key consistency check**: is `H(R_GT - R_proc) < H(R_GT)`? This is the BEDROCK assumption. If the procedural baseline ADDS entropy (e.g. by introducing texture artifacts that the residual must remove), the MDL accounting INVERTS and we end up encoding MORE bytes than HNeRV-only.

The Wallace-Boulton 1968 MML decomposition gives us a clean test: compute the empirical residual entropy on a few frames; if `H_emp(residual) < H_emp(GT)` by at least 30%, the MDL gain is real.

**Variational tightening**: the IB sidecar's role per Tishby 1999 is to bound `H(R | T)` from below. If we use a hierarchical IB (sequence of bottlenecks at increasing capacity), we get a tighter bound — this is Hinton-van Camp 1993 minimum-description-length networks applied to the contest scorer.

**My recommendation**: the design memo MUST include an empirical entropy-test gate. Before any L1 build, run the procedural baseline on 3-5 GT frames and measure `H(R_GT - R_proc)` vs `H(R_GT)`. If the ratio < 0.7, REVISE-procedural-design before continuing.

**ΔS prediction**: cannot estimate without the entropy-test data. Conditional: if `H(R_GT - R_proc) / H(R_GT) ≤ 0.4`, ΔS ≈ -0.05 to -0.07. If ratio ≈ 0.7, ΔS ≈ -0.02 to -0.04. If ratio > 0.85, ΔS ≈ 0 to -0.005 (procedural baseline didn't help).
**Confidence**: HIGH on the entropy-test methodology; UNKNOWN on the empirical ratio.
**Vote**: PROCEED-TO-L0-SCAFFOLD with entropy-test gate as the FIRST acceptance criterion.

---

### Ballé (relate to learned hyperprior; non-parametric extension)

E4 IS a non-parametric extension of my 2018 entropy bottleneck + scale hyperprior, with the procedural baseline as the **structural prior** and the IB sidecar as the **learned hyperprior**.

In Ballé 2018:
```
Bits = -log2 p_y(y | hyperprior_z)
hyperprior_z encoded with side-info bits
```

In E4:
```
Bits = -log2 p_residual(residual | procedural_baseline, ib_sidecar)
procedural_baseline encoded with structural-prior bits
ib_sidecar encoded with side-info bits
```

The mapping is exact. The procedural baseline plays the role of `hyperprior_z` but with HAND-DESIGNED structure. The IB sidecar adds a SECOND layer of conditioning: `p_residual(residual | procedural, ib)`.

**Ballé 2018 empirical insight**: the hyperprior gains 10-30% rate at same distortion vs factorized prior. E4's procedural-PLUS-IB compounds two hyperprior layers; expected gain is 25-50% over factorized — consistent with Selfcomp's 58 KB vs 178 KB ratio (67% reduction).

**My CRITICAL design suggestion**: don't make the procedural baseline FIXED. Make it LEARNABLE per-video (within the byte budget). This is the difference between Ballé 2018 (learned hyperprior) and Ballé 2016 (factorized prior). Per-video learnable procedural params = 1-2 KB extra side-info but unlocks the full hyperprior gain.

**ΔS prediction**: -0.030 to -0.060 with LEARNABLE procedural. -0.010 to -0.025 with FIXED procedural.
**Confidence**: HIGH on the analogical argument; MEDIUM on the empirical gain magnitude.
**Vote**: PROCEED-TO-L0-SCAFFOLD with LEARNABLE-procedural requirement (not hand-tuned).

---

## 3. Vote tally

| Member        | Vote                                       | ΔS centroid | Confidence |
|---------------|--------------------------------------------|------------:|-----------:|
| Shannon LEAD  | PROCEED-TO-L0 (with R(D) bound section)    | -0.040      | MEDIUM     |
| Dykstra       | PROCEED-TO-L0 (with multi-budget Pareto)   | -0.040      | MEDIUM-HIGH|
| Yousfi        | PROCEED-TO-L0 (compliance verified)        | n/a         | n/a        |
| Fridrich      | PROCEED-TO-L0 (UNIWARD analog)             | -0.035      | MEDIUM-HIGH|
| Contrarian    | **SUPER-VETO-AS-PROPOSED** + sequenced re-scope | +0.005 (pessimistic) | LOW |
| Quantizr      | PROCEED-TO-L0 (sequenced re-scope)         | -0.032      | MEDIUM     |
| Hotz          | PROCEED-TO-L0 (differentiable required)    | -0.040      | MEDIUM     |
| Selfcomp      | PROCEED-TO-L0 (byte arithmetic checks)     | -0.055      | MEDIUM-HIGH|
| MacKay        | PROCEED-TO-L0 (entropy-test gate FIRST)    | conditional | HIGH on method |
| Ballé         | PROCEED-TO-L0 (LEARNABLE-procedural)       | -0.045      | HIGH       |

**Tally**: 9 PROCEED-TO-L0 / 1 SUPER-VETO-AS-PROPOSED. Per CLAUDE.md "Council conduct — non-negotiable", a SUPER-VETO is binding for the architecture-as-proposed unless the operator over-rides. The Contrarian's constructive re-scope (sequenced E4-A then E4-B) is COMPATIBLE with all 9 PROCEED votes — NONE objected to splitting.

**ARCHITECTURE-CONTESTED** for the COMPOSITE proposal.
**ARCHITECTURE-CONSENSUS** for the SEQUENCED RE-SCOPE (E4-A then E4-B).

---

## 4. Architecture verdict

**ARCHITECTURE-CONTESTED** on whether to ship E4 as a composite substrate OR as two sequenced lanes (E4-A procedural-only, E4-B IB sidecar).

**Operator-routable decisions surfaced**:

### Decision MDL-IBPS-1: Composite vs Sequenced?
- **Option A (composite, as zen E4 written)**: ship procedural + IB + residual as ONE substrate. Higher EV if it works. Risk: existence-proof for procedural is missing; PR105 kitchen-sink anti-pattern.
- **Option B (sequenced, Contrarian's re-scope)**: ship E4-A procedural-only at L0 SCAFFOLD with 1-frame entropy-test gate; ship E4-B IB sidecar at L0 SCAFFOLD independently with synthetic-data IB-objective-tractability gate; compose only after BOTH gates pass. Lower per-step risk; EV recovered if BOTH pass.
- **Council recommendation**: Option B (9/10 council members compatible; Contrarian's veto resolved).

### Decision MDL-IBPS-2: Differentiable procedural vs hand-tuned procedural?
- **Option A (differentiable, Hotz)**: procedural model is a `nn.Module` with learnable params; co-train with residual via gradient descent through SegNet/PoseNet. Engineering cost +1-2 days. EV +0.005 to +0.015 ΔS over hand-tuned.
- **Option B (hand-tuned, simpler)**: procedural model is hand-initialized constants; residual absorbs all slack. Engineering cost -1-2 days. EV -0.005 to -0.015 ΔS vs differentiable.
- **Council recommendation**: Option A (differentiable). Hotz/Ballé/Quantizr explicit; others compatible.

### Decision MDL-IBPS-3: Learnable per-video procedural vs single shared procedural?
- **Option A (learnable per-video, Ballé)**: 1-2 KB extra side-info per video for per-video procedural params. Unlocks full hyperprior gain.
- **Option B (single shared)**: one set of procedural params shared across the contest video set. Lower side-info cost.
- **Council recommendation**: Option A on the SINGLE contest video (one-video-replay target mode per CLAUDE.md "Contest vs production target modes"); Option B for production-generalized target mode. Default for THIS lane: Option A on contest mode.

### Decision MDL-IBPS-4: Entropy-test gate FIRST?
- **MacKay strongly recommends** an empirical entropy test BEFORE any training run: compute `H(R_GT - R_proc) / H(R_GT)` on 3-5 GT frames. Gate threshold: ratio ≤ 0.7 or DEFER-pending-procedural-revision.
- **Council recommendation**: STRONG YES. This gate is $0 cost on macOS-CPU and falsifies the substrate within 1 day if the procedural design is wrong.

### Decision MDL-IBPS-5: probe-disambiguator pattern?
- Per CLAUDE.md "Anti-arbitrariness primitive: probe-disambiguator pattern" — when 2+ defensible interpretations exist, ship BOTH modes. E4-A (procedural-only) vs E4-B (IB-only) vs E4 composite are 3 defensible interpretations of the zen prediction.
- **Council recommendation**: build `tools/probe_mdl_ibps_disambiguator.py` after E4-A and E4-B both reach L1. Compose only after the probe verdict.

---

## 5. Predicted ΔS band (design-time)

**Composite (zen as written)**: ΔS ∈ [-0.080, -0.030] per zen memo. Mapped to score: `S ∈ [0.115, 0.165]` from PR101 0.193.

**Sequenced E4-A only (procedural baseline-only)**: ΔS ∈ [-0.045, -0.015] per Quantizr's analysis. `S ∈ [0.150, 0.180]`.

**Sequenced E4-B only (IB sidecar on PR101 substrate)**: ΔS ∈ [-0.012, -0.005] per Shannon's per-component analysis. `S ∈ [0.183, 0.190]` (much smaller; IB sidecar alone is a marginal improvement).

**Sequenced E4-A + E4-B composed (Pareto-walk)**: ΔS ∈ [-0.060, -0.020] (Dykstra alternating-projections estimate; Amdahl-bounded composition of E4-A + E4-B).

**Contrarian pessimistic counter-band**: ΔS ∈ [+0.012, -0.005] (procedural baseline fails to enter E(V_GT); residual replicates PR101 latent overhead PLUS procedural overhead).

**Confidence aggregate (research-only)**: LOW until existence-proof of procedural baseline lands.

---

## 6. Catalog #124 archive grammar (8 fields, design-time declaration)

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" + STRICT preflight Catalog #124:

| Field | Value |
|---|---|
| `archive_grammar` | monolithic single-file 0.bin with declared fixed offsets `[procedural_params, ib_sidecar, residual_decoder, residual_latents, meta_json]` |
| `parser_section_manifest` | `parse_archive(bytes) -> (procedural_params, ib_sidecar, residual_decoder_sd, residual_latents, meta_json)` |
| `inflate_runtime_loc_budget` | ≤ 100 LOC (target ~75 LOC: 30 procedural + 45 HNeRV-lite decode) |
| `runtime_dep_closure` | torch, brotli (same as PR101) |
| `export_format` | block-FP4 procedural params + 8-bit quantized IB sidecar + brotli-compressed residual decoder state_dict + int16 residual latents + utf8-json meta |
| `score_aware_loss` | `L = alpha * B(theta)/N + beta * d_seg(theta) + gamma * sqrt(d_pose(theta)) + lambda_ib * (I(X;T) - beta_ib * I(T;Y))` |
| `bolt_on_loc_budget` | `lane_class=substrate_engineering` (~600 LOC total: 80 procedural + 100 IB + 200 HNeRV-lite + 100 archive + 75 inflate + tests) |
| `no_op_detector_planned` | Catalog #139 `_build_no_op_proof` + executable byte-mutation smoke (mutate 1 byte in residual_decoder section, verify output frame changes); pose-residual smoke (mutate 1 byte in IB sidecar, verify pose changes); procedural smoke (mutate 1 byte in procedural_params, verify ego-motion changes) |

---

## 7. L0 scaffold status

**DEFERRED-pending-operator-decision** on Decision MDL-IBPS-1 (composite vs sequenced).

If operator selects **Option B (sequenced)**, the next subagent should:
1. Land `src/tac/substrates/mdl_ibps_procedural/` (E4-A) at L0 SCAFFOLD with the entropy-test gate as the first test.
2. Land `src/tac/substrates/mdl_ibps_sidecar/` (E4-B) at L0 SCAFFOLD independently with the VIB-objective-tractability gate as the first test.
3. Register both lanes via `tools/lane_maturity.py add-lane`.
4. NO training script lands until both gates pass.

If operator selects **Option A (composite)**, the next subagent should:
1. Land `src/tac/substrates/mdl_ibps/` at L0 SCAFFOLD as one substrate package.
2. Include the entropy-test gate AND the VIB-objective-tractability gate as composite acceptance criteria.
3. Register single lane.
4. NO training script lands until BOTH composite gates pass.

The current lane registry entry `lane_mdl_ibps_substrate_20260513` is a placeholder at L0 SKETCH; the operator's decision determines whether to split it or keep it unified.

---

## 8. Wire-in hooks (CLAUDE.md Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable:

1. **Sensitivity-map contribution**: N/A — research-only design memo. The implementation lane (E4-A or E4-B or composite E4) will register `tac.sensitivity_map.*` contributions when the substrate produces empirical per-pixel score-relevance maps.
2. **Pareto constraint**: PARTIAL — this memo declares the achievable region structure (Dykstra). The implementation lane will register concrete Pareto constraints to `tac.pareto_*` when training anchors land.
3. **Bit-allocator hook**: N/A — research-only. Implementation lane will register block-FP4 procedural quantization + IB sidecar 8-bit quantization with the bit-allocator when it ships.
4. **Cathedral autopilot dispatch hook**: N/A — research-only. Implementation lane will register dispatch hook when an archive grammar is byte-closed.
5. **Continual-learning posterior update**: N/A — no empirical anchors yet. Implementation lane will trigger posterior updates per Catalog #127/#128 when contest-CUDA / contest-CPU anchors land.
6. **Probe-disambiguator**: **HOOK ACTIVE** — multiple defensible interpretations enumerated (composite E4 vs sequenced E4-A+E4-B vs E4-B-only-on-PR101). `tools/probe_mdl_ibps_disambiguator.py` is the council-mandated builder per CLAUDE.md "Anti-arbitrariness primitive: probe-disambiguator pattern".

---

## 9. Falsification criteria

Per CLAUDE.md "KILL is the LAST RESORT" — the lane is DEFERRED-pending-research, not killed, on any negative result. Reactivation criteria below.

### Falsification gate F1 (MacKay entropy test, $0 cost)
Compute `H(R_GT - R_proc) / H(R_GT)` on 3-5 GT frames after hand-crafted procedural baseline.
- ratio ≤ 0.4: STRONG positive (proceed to L1 with confidence)
- 0.4 < ratio ≤ 0.7: WEAK positive (proceed to L1 with caveat)
- 0.7 < ratio ≤ 0.85: WEAK negative (DEFER-pending-procedural-revision)
- ratio > 0.85: STRONG negative (DEFER-pending-procedural-architecture-revision; reactivate when a different procedural design is proposed)

### Falsification gate F2 (Contrarian existence proof, $0 cost)
Hand-craft a 5-10 KB procedural baseline; measure SegNet argmax match on 1 GT frame.
- ≥ 60% match: PROCEED
- 40-60% match: PROCEED with CAVEAT
- 20-40% match: DEFER-pending-procedural-revision
- < 20% match: STRONG negative (DEFER until alternative procedural prior class proposed)

### Falsification gate F3 (Hotz IB-objective tractability, $0 cost)
Reproduce Alemi-2016 VIB on toy data (MNIST or synthetic Gaussian); measure I(X;T)/I(T;Y) curve.
- I(T;Y)/I(X;T) > 0.6 reachable: PROCEED
- I(T;Y)/I(X;T) ∈ [0.4, 0.6]: PROCEED with reduced ΔS expectation
- I(T;Y)/I(X;T) < 0.4: DEFER-pending-IB-objective-redesign

### Falsification gate F4 (Quantizr smoke training, $1-3 GPU cost)
Run a 200-epoch smoke on Modal T4 with E4-A only (procedural baseline + residual, no IB sidecar). Measure auth eval `[contest-CUDA T4]`.
- Score ≤ 0.180: STRONG positive (E4 hypothesis confirmed)
- 0.180 < Score ≤ 0.200: WEAK positive (proceed to E4-B)
- 0.200 < Score ≤ 0.220: WEAK negative (DEFER-pending-architecture-revision)
- Score > 0.220: STRONG negative (DEFER until alternative procedural class proposed)

### Falsification gate F5 (composite full training, $5-15 GPU cost)
Only after F1-F4 pass. Full composite E4 training. Measure auth eval `[contest-CUDA]` AND `[contest-CPU GHA Linux x86_64]` per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
- Score ≤ 0.165 (zen optimistic): STRONG positive
- 0.165 < Score ≤ 0.180: PROCEED
- 0.180 < Score < 0.193: WEAK positive (still beats PR101 frontier)
- Score ≥ 0.193: WEAK negative (no improvement over frontier; DEFER-pending-architecture-revision)
- Score ≥ 0.205 (Contrarian pessimistic): STRONG negative (DEFER)

---

## 10. Sister memo cross-refs

- `feedback_zen_state_frontier_deep_math_research_landed_20260513.md` (E4 source)
- `feedback_ancient_elder_polymath_landed_20260513.md` (Shannon-1959 + Solomonoff/MDL framework)
- `feedback_expert_team_fields_medalist_math_biology_alien_tech_landed_20260513.md` (Atick-Redlich efficient coding; B-1)
- `feedback_expert_team_signal_processing_alien_tech_landed_20260513.md` (Wyner-Ziv N3; cooperative-receiver framework)
- `adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md` (E4 floor v3 designation; LARGEST single bet)
- `council_lane_mdl_bayesian_design_20260430.md` (prior Bayesian-MDL council pass; predates this design)

---

## 11. Citations (mathematical anchors)

1. Tishby, Pereira, Bialek 1999. "The information bottleneck method." Allerton.
2. Alemi, Fischer, Dillon, Murphy 2016. "Deep variational information bottleneck." arXiv:1612.00410.
3. Solomonoff 1964. "A formal theory of inductive inference." Information and Control 7.
4. Rissanen 1978. "Modeling by shortest data description." Automatica 14.
5. Ballé, Minnen, Singh, Hwang, Johnston 2018. "Variational image compression with a scale hyperprior." arXiv:1802.01436.
6. Hinton & van Camp 1993. "Keeping neural networks simple by minimizing the description length of the weights." COLT.
7. Wallace & Boulton 1968. "An information measure for classification." Computer J 11.
8. Yousfi 2022. "Detector-informed embedding for steganography." Allerton.
9. Bertschinger & Natschläger 2004. "Real-time computation at the edge of chaos in recurrent neural networks." Neural Comp 16.

---

## 12. Status

**Design-time** declaration only. NO archive bytes. NO training run. NO score claim.

**Lane**: `lane_mdl_ibps_substrate_20260513` registered at L0 SKETCH (placeholder).
**Operator decisions surfaced**: 5 (MDL-IBPS-1 through MDL-IBPS-5).
**Predicted ΔS band**: `[0.115, 0.165]` composite per zen; `[0.150, 0.180]` E4-A only; `[0.183, 0.190]` E4-B only on PR101; Contrarian pessimistic `[0.180, 0.205]`.
**Confidence**: LOW until existence-proofs land.

END.
