# Next Experiments -- 2026-04-15

## Current Best: auth=0.74 (TTO v3, embedding loss -- DEAD)
## Renderer Baseline: auth=0.87
## Target: sub-0.50

## Active Runs (both on Modal T4)

### TTO v4 (RUNNING)
- Config: v1 loss (output MSE) + seg_odd_only + antialias_weight=0.2
- Expected improvement: v1 showed 6.3% proxy gain, seg_odd_only frees even-frame gradient
- Pre-registered criteria:
  - Success: proxy < 0.55 (significant over v3's 0.62)
  - Kill: proxy > 0.60 (no improvement over baseline)
  - Promote: auth < 0.65

### Joint Pair Generator v1 (RUNNING)
- Y-shaped U-Net, 576K params, 5000 epochs
- Backup path if TTO doesn't reach sub-0.50
- Pre-registered criteria:
  - Success: proxy < 0.60 (Quantizr-competitive)
  - Kill: proxy > 0.80 after 2000 epochs

## Queue (after active runs complete)

### Priority 1: TTO Tuning
1. **TTO v5**: Higher pose_weight + lower antialias for PoseNet-focused optimization
   - If v4 shows PoseNet improvement but not enough
   - Config: pose_weight=20-50, seg_weight=50, antialias=0.1
2. **TTO v6**: Longer patience + more steps
   - If v4 early-stops too soon
   - Config: patience=300, steps=1000

### Priority 2: Ensemble
3. **Best-of-N ensemble**: Combine TTO v4 + joint pair per-pair selection
   - Guaranteed >= best individual approach
   - Run after both active experiments complete

### Priority 3: Architecture
4. **Joint pair + TTO**: Use joint pair output as TTO init instead of renderer
   - Could give better starting point than asymmetric warp renderer
5. **Lower bit-depth export**: Test 3-bit or 2-bit renderer.bin for rate savings
   - Risk: quality degradation may negate rate savings

## Decision Gates
- 2026-04-15: TTO v4 auth eval -> decide TTO tuning direction
- 2026-04-17: Joint pair + TTO v4 ensemble -> decide final approach
- 2026-04-21: Lock final submission approach
- 2026-05-03: DEADLINE

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- SegNet loss weight > 100
- PSD architecture
- Embedding loss for TTO (v3: auth=0.74, PoseNet got WORSE)
