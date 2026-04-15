# Next Experiments -- 2026-04-15

## Current Best: auth=0.43 (TTO v5a, gradient fix)
## Renderer Baseline: auth=0.87
## Target: sub-0.20

## Paradigm Shift: SegNet Dominance (77:1 Leverage Ratio)

Step curve experiment COMPLETE. PoseNet saturates at 100 steps. SegNet is
the binding constraint -- 98.7% of remaining score.

## Completed Experiments
- [x] TTO step curve (Vast.ai 4090): phase transition at 100 steps, SegNet breakthrough at 500
- [x] TTO step curve cosine LR: empirically worse, constant LR wins
- [x] Hinge loss implementation: ready for experiment (P0)
- [x] Two-phase TTO implementation: ready for experiment (P1)
- [x] simulate_resize default fix: now True by default
- [x] check_vastai.py: canonical Vast.ai interaction script
- [x] download_modal_tto_frames.py: data permanence script

## URGENT: Data Permanence
- [ ] Download v5a tto_frames.pt from Modal (auth 0.43, 500-step)
- [ ] Download v5b tto_frames.pt from Modal (auth 0.41, 500-step)
- [ ] Run: `python scripts/download_modal_tto_frames.py`

## Queue (Priority Order -- SegNet First)

### P0: Hinge Loss Step Curve (NEXT EXPERIMENT)
- **What**: Run tto_step_curve with --segnet-loss-mode hinge
- **Why**: Hinge loss ignores easy SegNet pixels, focuses gradient on hard ones
- **Config**: `tto_step_curve_hinge` experiment in registry
- **Expected**: Faster SegNet convergence, lower SegNet score at same step count
- **Success criteria**: SegNet < 0.30 at 500 steps (vs 0.3435 with MSE)
- **Kill criteria**: SegNet worse than MSE at all step counts
- **Cost**: ~$0.12 on RTX 4090

### P1: Two-Phase TTO Validation
- **What**: Run TTO with two-phase schedule (100 PoseNet + 400 SegNet-only)
- **Why**: Don't waste steps on PoseNet after it saturates at 100
- **Expected**: Better SegNet than uniform 500 steps
- **Cost**: ~$0.12 on RTX 4090

### P2: Per-Pair Difficulty Map
1. **Per-pair difficulty map**: Run pair_difficulty_map.py with renderer checkpoint
   - Identifies easy vs hard pairs for budget allocation
   - Easy pairs: 100 steps (PoseNet saturation)
   - Hard pairs: 500+ steps (SegNet optimization)
   - Estimated savings: 3-4x total TTO compute

### P3: SegNet Architectural Improvements
2. **SegNet-focused TTO loss**: Increase seg_weight beyond 100
   - Step curve shows SegNet only moves at 500 steps -- maybe it needs MORE weight
   - Test seg_weight=500, 1000 with pose_weight=1 (already saturated)
3. **SegNet feature loss**: MSE on SegNet intermediate features instead of argmax
   - Current: hard disagreement (argmax != argmax) is non-differentiable
   - Proposed: soft disagreement on logits provides gradient for smooth optimization

### P4: Latent-Conditioned Renderer
4. **Pair-specific latent**: Condition renderer on a per-pair latent vector
   - Amortizes TTO: learn a mapping from pair difficulty to optimal residual
   - Train on 500-step TTO targets as ground truth
   - Eliminates TTO at inflate time entirely

### P5: TTO Distillation
5. **Distillation targets**: Use 500-step TTO outputs as training targets
   - Train renderer to directly output what TTO would produce
   - Previous attempt incomplete (12/60 batches before instance destroyed)
   - Requires downloading v5a tto_frames.pt from Modal first

### P6: Ensemble & Compression
6. **Best-of-N ensemble**: Per-pair selection across TTO variants
7. **Archive LZMA compression**: Free rate improvement

## Decision Gates
- 2026-04-16: Download Modal TTO frames -> data permanence achieved
- 2026-04-16: Hinge loss step curve -> validate or kill hinge approach
- 2026-04-17: Two-phase TTO validation
- 2026-04-18: Per-pair difficulty map results -> allocate adaptive budget
- 2026-04-20: Distillation targets -> train distilled renderer
- 2026-04-21: Lock final approach
- 2026-05-03: DEADLINE

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- PSD architecture
- Any TTO without rgb_to_yuv6 patch
- Uniform step allocation (step curve proves diminishing returns after 100)
- Cosine LR for TTO (empirically worse than constant LR)
