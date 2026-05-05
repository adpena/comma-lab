---
name: Complete lane taxonomy + stacking strategy for sub-Quantizr — what each lane teaches and how to compose them
description: 2026-04-27 strategic synthesis of all 18+ lanes dispatched today. Catalogs what each lane TEACHES (the empirical question it answers) and how lanes can be COMPOSED for the lowest possible contest-CUDA score. Score frontier 1.15; theoretical stack floor ~0.20-0.40 if 6+ orthogonal wedges land. Quantizr leader 0.33.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
# Lane Taxonomy: What Each Lane Teaches

## TIER 1 — Validated landed lanes (truth in hand)

| Lane | Score | Teaches |
|------|-------|---------|
| **Lane A** | **1.15** [contest-CUDA] | Pose TTO warm-started from baseline poses → PoseNet floor 0.005 reachable in 500 steps × 1.3h. SegNet floor 0.0046. Rate 0.0185 = 60% of total score. **Distortion is at architectural floor; rate is the entire wedge for further wins.** |
| Lane B-alt | 1.146 | Brotli on whole archive = -0.0037. Marginal. |
| Lane F V1 | 2.73 | Bug + arch — invalidated. |
| Lane F V2 | 1.79 | Bug-fixed FP4 still has **20× PoseNet penalty** (0.005 → 0.101). **FP4 uniform quantization is architecturally hostile to dilated-h64 ASYM PoseNet path.** Don't try uniform FP4 anywhere. |
| Lane M+N V1 | 2.35 | Mis-engineered (1-DOF save bug, baseline-padding-with-zeros). Sensitivity ≠ control: PoseNet's rank-1 sensitivity ≠ renderer's pose-input subspace. |
| Lane D V1 | killed @62% | Joint half-frame retrofit on full-frame-trained renderer failed to converge (proxy plateau). Possibly LR-starvation bug. |
| Lane G v1/v2 | killed | KL bug 14000× — fixed at f17de3bb. Untested post-fix. |

## TIER 2 — Code landed, awaiting Vast.ai dispatch

### A. Quantization lanes (rate attack on renderer.bin)

| Lane | Hypothesis | Predicted band | Teaches |
|------|-----------|---------------|---------|
| **Lane S** | Per-channel SC + Lagrangian protects PoseNet-sensitive layers (head, FiLM, fuse_conv) at FP32 while bulk reaches 2.5 bits avg | [0.85, 1.20] | Whether structured per-channel quantization avoids Lane F's 20× PoseNet penalty |
| **Lane W** | Lane S + hard-pair-weighted training steers SC bit-allocation to preserve channels critical for the worst pairs | [0.85, 1.10] | Whether per-pair sensitivity signal is real and exploitable |
| **Lane Ω V1 (water-fill)** | Per-WEIGHT bit-depth via Hessian + water-fill | [0.70, 1.05] | Whether per-weight resolution beats per-channel |
| **Lane Ω V2 (Lagrangian)** | Same as V1 but bit-depth fully learnable via Lagrangian dual ascent (no arbitrary water-fill) | [0.70, 1.05] | Mathematically optimal RD allocation vs heuristic |
| **Lane F-V3** | INT8 warmup (50ep) + lower lr (2.5e-6) + cosine anneal | [1.30, 1.80] | Closes "is FP4 fundamentally workable on ASYM-PoseNet?" definitively |

### B. Architecture lanes (rate attack via smaller arch)

| Lane | Hypothesis | Predicted band | Teaches |
|------|-----------|---------------|---------|
| **Lane I** | Cool-Chic CCh1 renderer (smaller arch, ~30-80KB binary) | [0.95, 1.30] | Whether Cool-Chic capacity matches dilated-h64 quality |
| **Lane K** | DSConv 88K from scratch | [0.85, 1.10] | Whether smaller arch trains to Lane A quality |
| **Lane V** | Quantizr replica: 88K + half-frame from epoch 0 + KL distill T=2.0 | [0.50, 1.10] | The closest path to 0.33 — biggest swing |
| **Lane GH** | Ghost modules (half params via cheap depthwise extension) | [1.05, 1.30] | Whether ghost modules substitute for DSConv |
| **Lane SZ** | szabolcs no-masks paradigm (Gaussian LUT + per-frame affine + shared latent + 1.017 bits/weight) | [0.30, 0.50] | Whether the szabolcs architectural paradigm is viable for replication |

### C. Pose lanes (rate attack on optimized_poses.pt + distortion improvement)

| Lane | Hypothesis | Predicted band | Teaches |
|------|-----------|---------------|---------|
| **Lane G v3** | KL distill auxiliary loss (weight=0.002 post-fix) on Lane A pose TTO | [1.10, 1.18] | Does Quantizr's KL distill recipe transfer to pose TTO? |
| **Lane M-V2** | Radial-zoom 1-DOF properly engineered (frozen baseline dims 1-5) | [1.10, 1.30] | Does rank-1 PoseNet hypothesis hold when properly padded? |
| **Lane LR** | LoRA pose rank-1 (606 params vs 3600) | [1.10, 1.18] | -0.005 rate from poses; quality preserved? |
| **Lane LM** | Lane-mark displacement → zero-cost pose dim 0 (computed at inflate time from masks) | [1.05, 1.15] | Physical pose prior; saves 15KB poses entirely |
| **Lane OS** | openpilot supercombo seeding for pose TTO | [1.05, 1.15] | Does compress-time supercombo init beat baseline init? |

### D. Mask lanes (rate attack on masks.mkv)

| Lane | Hypothesis | Predicted band | Teaches |
|------|-----------|---------------|---------|
| **Lane SI** | Saliency-inverted mask CRF (high CRF where scorer doesn't look) | [1.10, 1.18] | Does scorer blind-spot exploitation save mask bytes? |
| **Lane H** | Higher CRF on masks.mkv | tested previously | Mask CRF sweep |

### E. Training/loss lanes (improve training signal)

| Lane | Hypothesis | Predicted band | Teaches |
|------|-----------|---------------|---------|
| **Lane PS** | Per-class SegNet weighting (boost lane + boundary classes) | [1.05, 1.20] | Does class-conditional loss help? |
| **Lane D-V2** | LR fix (warmer floor + extended P2) | wide | Was Lane D plateau an LR bug? |

# Stacking Strategy — Composition Rules

## Orthogonal axes (independent rate wedges that compose)

1. **Renderer.bin** (rate ~0.46 of score):
   - Quantization (Lane S OR W OR Ω) on whatever architecture wins (Lane K/V/I/GH)
2. **Masks.mkv** (rate ~0.27 of score):
   - Cool-Chic mask codec (if Lane I-B builds it) OR Saliency-inverted CRF (Lane SI) OR higher CRF baseline
3. **Optimized_poses.pt** (rate ~0.005 of score):
   - LoRA (Lane LR) OR zero-cost lane-mark (Lane LM) OR both
4. **Pose distortion** (~0.22 of score):
   - More TTO + KL aux (Lane G v3) + supercombo seeding (Lane OS) on top of Lane A
5. **SegNet distortion** (~0.46 of score):
   - Per-class weighting (Lane PS) + KL distill on logits (Quantizr trick) integrated into training

## Mutually exclusive axes (pick one)

- **Architecture** (only one renderer in archive): dilated-h64 (Lane A baseline) OR DSConv-88K (Lane K) OR Cool-Chic (Lane I) OR Ghost-h64 (Lane GH) OR Quantizr-replica (Lane V) OR szabolcs (Lane SZ)
- **Quantization scheme** (only one applied to chosen arch): per-tensor (FP4 dead) OR per-channel SC (Lane S) OR hard-pair-weighted SC (Lane W) OR per-weight Hessian (Lane Ω)

## Council-projected optimal stacks

### CONSERVATIVE STACK (high confidence to land sub-1.0):
- Arch: dilated-h64 (Lane A baseline, validated)
- Quant: Lane S (per-channel SC)
- Pose: Lane G v3 (KL aux) + Lane LR (LoRA rank-1)
- Mask: Lane SI (saliency-inverted CRF)
- Result projection: 0.85-0.95

### AGGRESSIVE STACK (medium confidence):
- Arch: Lane V (Quantizr replica 88K + half-frame)
- Quant: Lane W (hard-pair SC, applied during V's training)
- Pose: Lane LM (zero-cost) + Lane OS (supercombo seed)
- Mask: half-frame (built into Lane V) + Lane SI on the half-frame masks
- Result projection: 0.40-0.70

### MOONSHOT STACK (low confidence, paper-worthy if it works):
- Arch: Lane SZ (szabolcs no-masks Gaussian LUT)
- Quant: 1.017 bits/weight block-FP (built into szabolcs)
- Pose: Lane LM zero-cost + Lane OS supercombo
- No masks in archive at all
- Result projection: 0.20-0.50

# Per-Lane Learning Priority

After each Vast.ai dispatch, record into `.omx/research/findings.md`:
- The HYPOTHESIS being tested
- The PREDICTED band before launch
- The ACTUAL [contest-CUDA] result
- The DELTA (if predicted [0.85, 1.10] and actual is 0.92, prediction was correct; if actual is 1.50, hypothesis is invalidated)
- The COMPOSABILITY: does this lane stack with others, or does it conflict?

# Composition Conflict Matrix

Lanes that CANNOT compose (architectural incompatibility):
- Lane S/W/Ω require AsymmetricPairGenerator architecture (dilated-h64, Lane K, Lane V, Lane GH all compatible). NOT compatible with Lane I (CCh1) or Lane SZ (different format).
- Lane W requires Lane A's pair_weights.pt → only useful if dilated-h64 or near-clone arch is the renderer
- Lane LM/OS work with ANY renderer (pose seeding is upstream of renderer choice)
- Lane SI works on masks.mkv → not applicable if Lane SZ (no masks)
- Lane LR works on optimized_poses.pt → not applicable if Lane LM zero-cost (no poses.pt)

# Open Research Questions (to be answered by Vast.ai dispatch results)

Q1. Is the heavy-tail PoseNet sensitivity signal stable across runs? (Answered by Lane W's per-pair profile across multiple seeds.)
Q2. Does per-channel SC's bit-allocation actually preserve the FiLM-adjacent layers? (Answered by Lane S's auth eval breakdown.)
Q3. Does the rank-1 PoseNet hypothesis survive when properly engineered? (Answered by Lane M-V2.)
Q4. Is FP4 architecturally dead, or does a different recipe (INT8 warmup) save it? (Answered by Lane F-V3.)
Q5. Can a smaller arch (88K) match Lane A's distortion floor? (Answered by Lane K.)
Q6. Does the szabolcs paradigm replicate in our pipeline? (Answered by Lane SZ.)

# Memory Recall Anchors

- All previous council deliberations: `project_council_findings_20260414`, `project_master_council_verdict`, `project_three_breakthroughs`
- All previous architectural insights: `project_quantizr_full_intel_20260421`, `project_szabolcs_full_re_20260426`
- All previous proxy-auth gap learnings: `feedback_proxy_auth_math_useless`, `feedback_proxy_auth_gap_835x`
- All previous hard-pair / overfitting lessons: `feedback_overfit_is_the_goal`, `feedback_posenet_tracking`, `feedback_curriculum_must_use_full_score`
- All previous quantization research: `project_research_survey_20260420`, `project_5stage_quantization_advantage`
- All previous architecture knowledge: `project_hardware_geometry_chroma_full`, `project_arbitrary_vs_learnable_taxonomy`

# Status (2026-04-27 EOD)

- 11 lanes have committed/staged code (Lane S, I done; Lane F-V3+M-V2 done; Lane W, K, V, Ω, OS, LM, SI, D-V2+G-v3, PS, LR, GH, SZ in flight)
- Lane A (1.15) is the only validated frontier
- Vast.ai dispatch wave coming after subagents land
- Budget: ~$300 secured (no $25 cap as of user clarification)
- Days to deadline: ~5-6
