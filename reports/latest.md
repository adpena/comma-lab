# Latest Report -- 2026-04-15

## Session 35 Summary: DX Hardening + SegNet Paradigm Shift

### Current Best Scores
| Track | Auth Score | Details |
|-------|-----------|---------|
| TTO v5b (embedding) | **0.41** | 500-step, embedding loss, seg_odd_only |
| TTO v5a (output MSE) | **0.43** | 500-step, first valid TTO with PoseNet gradients |
| Renderer baseline | **0.87** | asym_v5_lagrangian_fixed, ep12600 |

### Paradigm Shift: SegNet Dominates at 77:1 Leverage Ratio

Step curve experiment (Vast.ai RTX 4090, 30 pairs) revealed:
- PoseNet saturates at 100 TTO steps (165.27 -> 0.042, 3970x reduction)
- SegNet contributes 98.7% of remaining score after PoseNet convergence
- 500-step breakthrough: SegNet moves from 0.5036 to 0.3435 (32% reduction)
- All future effort must target SegNet

### Three Breakthroughs Implemented (UNTESTED)

1. **Hinge loss for SegNet** (P0): Logit-margin hinge loss focuses gradient on
   boundary pixels at risk of argmax flip. Ignores already-correct pixels.
   Expected 2-5x faster SegNet convergence.

2. **Two-phase TTO** (P1): Phase 1 (100 steps) = joint PoseNet+SegNet optimization.
   Phase 2 (200+ steps) = SegNet-only on odd frames. Freezes even frames and
   PoseNet after Phase 1, preventing PoseNet regression during SegNet polish.

3. **Latent codes per pair**: Pair-specific learnable vectors for amortized TTO.
   Not yet integrated into deployment.

### Session Commits: 40+
- Hinge loss implementation in tac library
- Two-phase TTO with Phase 2 SegNet-only mode
- simulate_resize default changed to True
- check_vastai.py canonical DX script
- download_modal_tto_frames.py data permanence
- PROVENANCE.md experiment provenance documentation
- Pair difficulty map script (first run this session)
- Vast.ai tto_v6_hinge_phase2 experiment registered

### Council Decisions (Binding)
- Hinge loss approved unanimously (15-0)
- Two-phase TTO approved unanimously (15-0)
- Cosine LR killed (empirically worse than constant)
- SegNet is the binding constraint, all effort must target it

### What's Ready to Deploy
- `tto_v6_hinge_phase2` experiment in Vast.ai registry: combines ALL discoveries
  (embedding loss, hinge loss, two-phase, constant LR, simulate_resize, seg_odd_only)
- `tto_step_curve_hinge` experiment: validates hinge loss improvement curve
- Cost estimate: ~$0.12-0.25 per experiment on RTX 4090

### Vast.ai Budget
- Spent: $0.27 of $24.00 hard cap
- Remaining: $23.73
- All instances destroyed

### Critical Data on Modal Volume
- `asym_v5_lagrangian_fixed/tto_v5a_output_mse/tto_frames.pt` (auth 0.43)
- `asym_v5_lagrangian_fixed/tto_v5b_embedding/tto_frames.pt` (auth 0.41)
- MUST download before Modal access expires

### 18-Day Plan (Deadline: May 3, 2026)
1. Download Modal TTO frames (data permanence)
2. Hinge loss step curve validation
3. Two-phase TTO validation
4. Per-pair difficulty map -> adaptive budget allocation
5. Distillation targets from 500-step TTO
6. Lock final approach by April 21
7. Final submission packaging

---

## Renderer Baseline (reference)
- Track: `robust_current`
- Variant: `asym_v5_lagrangian_fixed`
- Platform: `modal_t4`
- Auth score: **0.87** (seg=0.21, pose=0.56, rate=0.10)
- Checkpoint: `renderer_best.pt` at ep12600
