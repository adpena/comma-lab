# Next Experiments -- 2026-04-15

## Current Best: auth=0.43 (TTO v5a, gradient fix)
## Renderer Baseline: auth=0.87
## Target: sub-0.20

## Paradigm Shift: SegNet Dominance (77:1 Leverage Ratio)

Step curve experiment COMPLETE. PoseNet saturates at 100 steps. SegNet is
the binding constraint -- 98.7% of remaining score.

## Completed Experiments
- [x] TTO step curve (Vast.ai 4090): phase transition at 100 steps, SegNet breakthrough at 500
- [x] Per-pair difficulty map script (experiments/pair_difficulty_map.py)

## In Progress
- [ ] Distillation TTO targets (Vast.ai Instance B, 12/60 batches)

## Queue (Priority Order -- SegNet First)

### Priority 1: Adaptive TTO Budget Allocation
1. **Per-pair difficulty map**: Run pair_difficulty_map.py with renderer checkpoint
   - Identifies easy vs hard pairs for budget allocation
   - Easy pairs: 100 steps (PoseNet saturation)
   - Hard pairs: 500+ steps (SegNet optimization)
   - Estimated savings: 3-4x total TTO compute

### Priority 2: SegNet Architectural Improvements
2. **SegNet-focused TTO loss**: Increase seg_weight beyond 100
   - Step curve shows SegNet only moves at 500 steps -- maybe it needs MORE weight
   - Test seg_weight=500, 1000 with pose_weight=1 (already saturated)
3. **SegNet feature loss**: MSE on SegNet intermediate features instead of argmax
   - Current: hard disagreement (argmax != argmax) is non-differentiable
   - Proposed: soft disagreement on logits provides gradient for smooth optimization
4. **Mask-conditioned residual learning**: Train a small network to predict
   SegNet-optimal residuals given the current renderer output + mask

### Priority 3: TTO Distillation
5. **Distillation targets**: Use 500-step TTO outputs as training targets
   - Train renderer to directly output what TTO would produce
   - Eliminates TTO at inflate time entirely
   - In progress on Vast.ai (12/60 batches)

### Priority 4: Ensemble & Compression
6. **Best-of-N ensemble**: Per-pair selection across TTO variants
7. **Archive LZMA compression**: Free rate improvement

## Decision Gates
- 2026-04-16: Per-pair difficulty map results -> allocate adaptive budget
- 2026-04-17: SegNet-focused TTO experiment results
- 2026-04-19: Distillation targets complete -> train distilled renderer
- 2026-04-21: Lock final approach
- 2026-05-03: DEADLINE

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- PSD architecture
- Any TTO without rgb_to_yuv6 patch
- Uniform step allocation (step curve proves diminishing returns after 100)
