# Next Experiments -- 2026-04-15

## Current Best: auth=0.41 (TTO v5b, embedding loss)
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
- [x] Pair difficulty map: COMPLETED (600 pairs, 41.1s on MPS)
- [x] tto_v6_hinge_phase2 registered in Vast.ai experiment registry

## Pair Difficulty Map Results
- 600 pairs analyzed with simulate_resize=True
- PoseNet MSE: mean=158.98, range=[85.75, 199.55]
- SegNet disagree: mean=0.505, range=[0.490, 0.519]
- 120 hard pairs (top 20%), 300 easy pairs (bottom 50%)
- Hardest pair: #523 (score=95.43), easiest: #514 (score=79.74)

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

### P0.5: v6 Full Run (ALL Improvements Combined)
- **What**: Run tto_v6_hinge_phase2 on Vast.ai 4090
- **Why**: Combines ALL discoveries: embedding + hinge + two-phase + simulate_resize
- **Config**: `tto_v6_hinge_phase2` in Vast.ai registry
- **Expected**: auth < 0.30 (if hinge + phase2 work as theorized)
- **Cost**: ~$0.25 on RTX 4090

### P1: Two-Phase TTO Validation
- **What**: Run TTO with two-phase schedule (100 PoseNet + 400 SegNet-only)
- **Why**: Don't waste steps on PoseNet after it saturates at 100
- **Expected**: Better SegNet than uniform 500 steps
- **Cost**: ~$0.12 on RTX 4090

### P2: Adaptive TTO Budget (Uses Pair Difficulty Map)
- **What**: Allocate TTO steps based on per-pair difficulty
- **Why**: 120 hard pairs need 500+ steps, 300 easy pairs need ~100
- **Expected**: 3-4x total TTO compute savings with same quality
- **Requires**: pair_difficulty_map.json (COMPLETED)

### P3: SegNet Architectural Improvements
- **SegNet-focused TTO loss**: Increase seg_weight beyond 100
- **SegNet feature loss**: MSE on SegNet intermediate features

### P4: Latent-Conditioned Renderer
- Pair-specific latent vectors: amortize TTO into renderer

### P5: TTO Distillation
- Use 500-step TTO outputs as training targets
- Requires downloading v5a/v5b tto_frames.pt from Modal

### P6: Ensemble & Compression
- Best-of-N ensemble: per-pair selection across TTO variants
- Archive LZMA compression: free rate improvement

## Decision Gates
- 2026-04-16: Download Modal TTO frames -> data permanence achieved
- 2026-04-16: Hinge loss step curve -> validate or kill hinge approach
- 2026-04-17: v6 full run -> combined improvements auth eval
- 2026-04-18: Two-phase TTO validation
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
