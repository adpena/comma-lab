---
name: Codec stacking + composition + unlimited-compress canonical orders + score-arithmetic priority
description: 2026-04-29 PM. Two grand council codex sessions delivered clean verdicts on bit-budget allocation + stacking composition under unlimited compress time. Final verdict: "lowest theoretical floor is not one magic codec. It is a slot-aware stack: choose one mask representation, one renderer representation, one pose representation, optimize them with joint ADMM, then arithmetic-code every discrete residual that survives overhead." Includes canonical composition orders + bad-order traps + per-stack expected score gains.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Score-arithmetic priority (bit-budget codex)

Concrete priority ranking (highest EV first):

1. **Mask-layer optimization** — STC redesign, learned mask priors, hyperprior, NeRV/Cool-Chic. 60KB saved on masks = 0.04 score (45× the shipped Lane PD impact of 0.00089).
2. **Architecture exploration** that removes mask payload dependence
3. **Joint-ADMM** once frontiers measured (1-2 weeks, +0.015-0.050 score)
4. **Renderer water-fill + arithmetic (Ω-W-V2)** — 1-2 days, +0.005 score (only ~0.000 unless renderer is fully Selfcomp-exported)
5. **Lane PD-V2** as low-risk filler — 1-2 days, +0.0007-0.0011 ceiling
6. **Codebooks/side-info** polish only when it enables larger stream savings

Quantizr 250KB target → rate term 0.166 → only path consistent with Phase 4 floor 0.12-0.20. Pose savings cap at ~5KB total (-0.003 score even at infinite compression). Mask layer has 50× more headroom.

## Per-stream waterline (Boyd/Dykstra)

KKT condition at optimum: `dScore_s / dByte_s = common waterline` across all active streams. Right now NOT equilibrated:
- Pose byte: dScore/dByte ≈ 0.00067 (small; saturates at ~5KB total)
- Mask byte: dScore/dByte ≈ 0.00067 BUT 50× more headroom (60KB+ savings achievable)

Joint ADMM solves this projection cleanly under convex relaxation; the real problem is discrete and nonconvex, so robust implementation is empirical frontier sampling + convex-hull relaxation + exact candidate evaluation.

## Canonical composition order (stacking codex)

```
scorer-aware analysis
  → representation choice (NeRV/Cool-Chic/STC/wavelet for masks; block-FP/IMP for weights; Kalman/poly/delta for poses)
  → prediction / transform (Kalman residual, polynomial+residual, DCT/DWT, optical flow)
  → water-fill / quantize / block-FP / VQ
  → hyperprior fit (only if amortizable: large streams)
  → arithmetic coding  ← ALWAYS TERMINAL
  → archive packing
```

**Pairwise precedence rules**:
- Water-fill MUST follow block-FP eligibility + curvature estimation, BUT precede qint arithmetic
- Predictive coding (Kalman, polynomial, optical flow) precedes VQ, STC, wavelet, arithmetic — creates residuals
- VQ precedes hyperprior/arithmetic — creates symbols
- Hyperprior precedes arithmetic — defines probability model
- NeRV/Cool-Chic, STC, wavelet are mask-representation alternatives FIRST; if hybridized, choose one base + code residual with the other
- ADMM wraps all: sets budgets, invokes each proximal codec, measures byte/distortion deltas, iterates

## Concrete stack orders by lane family

| Stack | Order | Use case |
|---|---|---|
| Selfcomp weights | block-FP → water-fill → hyperprior → arithmetic | Renderer payload optimization |
| Predictive VQ residuals | predict → VQ → hyperprior → arithmetic | Pose Kalman+VQ |
| Wavelet residual masks | predict → wavelet → hyperprior → arithmetic | Mask residual coding |
| Predictive STC mask classes | predict → STC → arithmetic | Lane STC redesign properly |
| NeRV weights | NeRV → block-FP → water-fill → hyperprior → arithmetic | Cool-Chic-derived weights |
| Full stack outer loop | ADMM(representation, prediction, quantization, entropy) | Joint optimization |

## BAD orders (no-op traps + warnings)

- **arithmetic before anything**: no-op trap (entropy coding does nothing without symbols)
- **block-FP after arithmetic**: destroys the coded stream (cannot unblock entropy)
- **water-fill before block-FP levels**: gives unrealizable real-valued bits (allocator must project to discrete ladder)
- **hyperprior on raw float tensors**: usually header bloat (need quantization first)
- **STC after AV1**: wrong unless coding a decoded RESIDUAL (AV1 has already made class stream inaccessible)
- **scorer-side inflate tricks**: violates strict-scorer-rule; inflate-time decoder may not load SegNet/PoseNet

## Failure modes / warnings

- **Per-channel allocation saturation**: if dim wants 12 bits but ceiling is 8, extra Lagrangian mass wasted. Project back to discrete ladder (e.g. qint_max=31).
- **Arithmetic overhead trap**: 200B entropy model on 3-5KB pose stream erases gain. Static histograms first; learned entropy only for large streams.
- **Hyperprior is both bit-saver AND bit-cost**: charge side-info inside archive.zip. No "free prior" accounting.
- **ADMM divergence**: use adaptive penalty, restarts, exact byte projection after every codec call.
- **Strict-scorer-rule non-negotiable**: SegNet/PoseNet may guide compress-time search; inflate-time decoder may not load them.

## Expected score gains by lane (basis points, 100bp = 0.01 score)

| Lane | Score gain | Implementation cost |
|---|---|---|
| Lane PD-V2 (arithmetic-coded pose deltas) | 7-11 bp (deterministic) | 1-2 days |
| Lane Ω-W-V2 (water-fill + arithmetic on Selfcomp/block-FP renderer) | 200-450 bp if eligible, 70-90 bp pose-only | 1-2 days |
| Lane Joint-ADMM | 150-500 bp | 1-2 weeks |
| Wavelet mask coding | 80-300 bp (mostly overlapping with STC/NeRV) | 1-2 weeks |
| NeRV/Cool-Chic mask codec | replaces AV1; no specific number | 2-3 weeks |
| STC redesign (predictive boundary coding) | depends on smoke; potentially 200-400 bp | 1-2 weeks |

## Hard kill / promotion criteria (Carmack)

- Any candidate stack that cannot beat its simpler static-arithmetic baseline by 50 bp AFTER overhead → DIES
- Lane PD-V2 hard overhead gate: `if encoded + header >= current pose_delta_v1, keep current PD` — don't ship a regression
- ADMM candidate that converges to useless discrete point → restart with adaptive penalty

## Dispatch order (Phase 1.5, next 2 weeks)

1. **Lane PD-V2** (low-risk filler, exercises entropy path): 1-2 days, deterministic +7-11 bp
2. **Lane Ω-W-V2** (only on Selfcomp-exported renderers, not ASYM): 1-2 days, +200-450 bp if eligible
3. **Joint test**: PD + Ω-W + qint arithmetic on a compatible renderer. Predicted archive gain: 200-450 bp if renderer stream is eligible, ~70-90 bp pose-only

Phase 2 enhanced:
- Make Boyd's ADMM the COORDINATOR, not another lane
- Inputs: per-stream R(D), measured entropy, scorer sensitivities, strict inflate-time budget
- Hyperprior implementation only where stream size amortizes side-info (renderer qints, mask residuals, maybe wavelet coefficients). Tiny pose streams use static tables.

Phase 3 enhanced:
- Multi-pass compress optimization treating every archive byte as assigned to stream with measured marginal score value
- 12h compress-time sweeps over: boundary_fraction, qint ladder, codebook size, hyperprior payload, ADMM penalty
- Karpathy mandate: every pass logs exact archive bytes, entropy estimates, scorer deltas, failed independence assumptions

## Final council verdict

"The lowest theoretical floor is not one magic codec. It is a slot-aware stack: choose one mask representation, one renderer representation, one pose representation, optimize them with joint ADMM, then arithmetic-code every discrete residual that survives overhead."

## Cross-refs

- /tmp/codex_runs/bit_budget_allocation_research.log (per-stream R(D), 14K lines, clean verdict)
- /tmp/codex_runs/stacking_composition_unlimited_compress.log (stacking matrix, 5K lines, clean verdict)
- project_6month_strategic_plan_20260429.md (Phase 1-4 high-level plan)
- project_phases_2_3_4_design_implementation_math_provenance_20260429.md (15 lane scoping)
- src/tac/{pose_delta_codec, water_filling_codec, arithmetic_qint_codec, block_fp_codec, learnable_class_targets, stc_boundary_codec, contrib/coolchic_renderer}.py
- experiments/build_lane_stc_av1_residual_smoke.py (the CPU smoke that surfaced AV1+residual structural issues)
