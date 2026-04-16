# Next Experiments -- 2026-04-16

## Current Best: proxy=0.275 (TTO v6, hinge+phase2+embedding)
## Prev Auth Best: auth=0.41 (TTO v5b, embedding loss)
## Renderer Baseline: auth=0.87
## Target: sub-0.20

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
- [x] TTO step curve hinge (BREAKTHROUGH - 24% better than xent)
- [x] v6 TTO full run: proxy 0.275 on 600 pairs
- [x] check_vastai.py DX bugs found and fixed (6 bugs)
- [x] Pair difficulty map v2 (600 pairs, correct checkpoint)
- [x] Vast.ai instance lifecycle: create, deploy, run, download, destroy

## Queue (Priority Order)

### P0: Auth Eval of v6 TTO (NEXT)
- **What**: Run authoritative evaluation on v6 TTO frames
- **Why**: proxy=0.275 needs auth confirmation. v5b was proxy ~0.60 -> auth 0.41
- **How**: Use tto_frames.pt (708MB, downloaded locally) through inflate pipeline
- **Expected**: auth ~0.20-0.30 (if proxy-auth correlation holds)

### P1: More TTO Steps with Hinge
- **What**: Run hinge TTO with 300-500 P1 steps instead of 150
- **Why**: Step curve shows continued improvement up to 500 steps
- **Config**: --tto-steps 300-500 --segnet-loss-mode hinge
- **Cost**: ~$0.20 on RTX 4090

### P2: Adaptive TTO Budget
- **What**: Allocate TTO steps based on per-pair difficulty
- **Why**: 120 hard pairs need 500+ steps, 300 easy pairs need ~100
- **Expected**: Same quality at 60% compute cost

### P3: Higher Hinge Margin
- **What**: Test hinge_margin 1.0, 2.0 on step curve
- **Why**: margin=0.5 might not push SegNet hard enough

### P4: SegNet-Only Phase 2 with More Steps
- **What**: Extend P2 to 500-1000 steps
- **Why**: SegNet is still improving at step 500 with hinge

### P5: TTO Distillation
- Use v6 TTO frames as training targets for distilled renderer

## Decision Gates
- 2026-04-16: v6 auth eval -> proxy-auth correlation with hinge
- 2026-04-17: More steps experiment -> find hinge saturation point
- 2026-04-18: Adaptive TTO budget -> compute savings
- 2026-04-21: Lock final approach
- 2026-05-03: DEADLINE

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- PSD architecture
- Any TTO without rgb_to_yuv6 patch
- Cosine LR for TTO (empirically worse than constant LR)
- xent loss for TTO (hinge is strictly better from 50+ steps)
