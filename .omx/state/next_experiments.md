# Next Experiments -- 2026-04-15 (DX Hardening Update)

## Scoreboard
- **Current best [contest-compliant]**: auth=0.87 (renderer baseline, no TTO)
- **Current best [unlimited-compute]**: auth=0.43 (TTO v5b, gradient fix)
- **Best proxy (600 pairs)**: 0.275 (v6 hinge+phase2+embedding, RTX 4090)
- **Quantizr threat**: 0.33 (PR#55, FiLM+DSConv+eval-resize)
- **Target**: sub-0.25 [contest-compliant] to beat Quantizr

## Re-Validation Results (correct checkpoint cff8dca4)

### Step Curve xent (baseline, 30 pairs)
| Steps | PoseNet  | SegNet  | Score |
|-------|----------|---------|-------|
| 0     | 0.0374   | 0.00197 | 0.809 |
| 100   | 0.0093   | 0.00169 | 0.473 |
| 200   | 0.0013   | 0.00155 | 0.267 |
| 500   | 0.0004   | 0.00126 | 0.192 |

### Step Curve hinge (BREAKTHROUGH, 30 pairs)
| Steps | PoseNet  | SegNet  | Score |
|-------|----------|---------|-------|
| 0     | 0.0375   | 0.00197 | 0.810 |
| 100   | 0.0076   | 0.00131 | 0.407 |
| 200   | 0.0008   | 0.00102 | 0.190 |
| 500   | 0.0007   | 0.00064 | 0.145 |

### v6 TTO Full Run (hinge+phase2+embedding, 600 pairs)
- Baseline: 0.634 -> TTO: 0.275 (57% reduction)
- PoseNet: 0.00249 (86% improvement)
- SegNet: 0.00118 (45% improvement)
- 150 P1 + 200 P2 steps, 29.4 min on RTX 4090

### Key Findings
- Hinge beats xent at every step count from 50+ (24-29% better)
- SegNet with hinge: 0.000639 vs 0.001259 at 500 steps (49% better!)
- Early TTO (10-25 steps) HURTS PoseNet -- SegNet-PoseNet tug-of-war
- Phase transition at ~100 steps holds with correct checkpoint

## Completed Experiments
- [x] TTO step curve xent with correct checkpoint (re-validated)
- [x] TTO step curve hinge (BREAKTHROUGH - 24-49% better than xent)
- [x] v6 TTO full run: proxy 0.275 on 600 pairs
- [x] check_vastai.py DX bugs found and fixed (6 bugs)
- [x] Pair difficulty map v2 (600 pairs, correct checkpoint)
- [x] Vast.ai instance lifecycle: create, deploy, run, download, destroy
- [x] DX hardening: build_deploy_bundle.sh, experiment registry v7, state files updated

## Queue (Priority Order)

### P0: Auth Eval of v6 TTO [BLOCKING]
- **What**: Run authoritative evaluation on v6 TTO frames
- **Why**: proxy=0.275 needs auth confirmation. v5b was proxy ~0.60 -> auth 0.41
- **How**: Use tto_frames.pt (708MB, downloaded locally) through inflate pipeline
- **Expected**: auth ~0.18-0.28 (if hinge improves proxy-auth correlation)
- **Cost**: Free (local M5 Max)

### P1: 500-Step Hinge TTO Full Run [HIGH]
- **What**: Run hinge TTO with 500 P1 steps on all 600 pairs
- **Why**: Step curve shows continued SegNet improvement at 500 steps with hinge
- **Config**: tto_v7_hinge_roundtrip (registry entry added this session)
- **Cost**: ~$0.25 on RTX 4090 (~1 hr)
- **Expected**: proxy ~0.140-0.160

### P2: FiLM Architecture Smoke Test [HIGH — Quantizr validated]
- **What**: Add FiLM pose conditioning to inflate_renderer.py AsymmetricPairGenerator
- **Why**: Quantizr PR#55 = 0.33 with FiLM. Directly addresses DP-SIMS PoseNet failure.
- **Hypothesis**: FiLM on GT pose vectors enables temporal coherence without recurrence.
- **Design**: FiLM after each warp feature block, conditioned on (dx, dy, dz, droll, dpitch, dyaw)
- **Test**: 30 pairs, 100 steps hinge, MPS local
- **Success**: PoseNet < 0.005 at 100 steps
- **Council**: Needs tripartite pact sign-off before implementation

### P3: Adaptive TTO Budget [MEDIUM]
- **What**: Allocate TTO steps based on per-pair difficulty map
- **Why**: 120 hard pairs need 500+ steps, 300 easy pairs need ~100
- **Expected**: Same proxy quality at 60% compute cost
- **Dependency**: Difficulty map v2 exists (pair_difficulty_v2/), ready to use

### P4: DSConv Rate Reduction [MEDIUM]
- **What**: Replace standard convs with depthwise separable convs in renderer
- **Why**: Quantizr PR#55 uses DSConv. Reduces archive size (rate term).
- **Expected**: 3-5x parameter reduction, ~0.02 rate improvement = ~0.5 score points
- **Council**: Design decision — needs tripartite pact vote

### P5: Higher Hinge Margin Sweep [LOW]
- **What**: Test hinge_margin 1.0, 2.0 on step curve
- **Why**: margin=0.5 might not push SegNet hard enough
- **Cost**: ~$0.10 on RTX 4090

### P6: TTO Distillation [SPECULATIVE]
- Use v6 TTO frames as training targets for distilled renderer
- Gated behind auth eval of v6 confirming proxy gains hold

## Decision Gates
- 2026-04-16: v6 auth eval -> proxy-auth correlation with hinge
- 2026-04-17: 500-step hinge auth -> find saturation, confirm auth gap
- 2026-04-18: FiLM smoke test -> MPS, 30 pairs, 100 steps hinge
- 2026-04-21: Lock architecture (FiLM vs warp, binding council vote)
- 2026-05-03: DEADLINE

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- PSD architecture (failed auth eval, stayed with dilated)
- Any TTO without rgb_to_yuv6 patch
- Cosine LR for TTO (empirically worse than constant LR)
- xent loss for TTO (hinge is strictly better from 50+ steps)
- BT.601 color matrix change

## Reconsidering (FiLM-revived, see docs/archive/killed_techniques/README.md)
- DP-SIMS independent generation + FiLM (pending smoke test)
- Constrained generation from noise + FiLM + hinge (gated behind Lane 2)
- SIREN/NeRV memorization (lane open, Cool-chic validates paradigm)
