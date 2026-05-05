---
name: Modal Renderer+TTO Experiment Design
description: NEXT SESSION PRIORITY #1. Load v5 renderer, generate frames, TTO refine, auth eval. Design spec ready.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Experiment: Renderer + TTO on Modal

**This is the pivotal experiment. If PoseNet drops from 0.031 to <0.005, score goes from 0.87 to ~0.54.**

### Pipeline
1. Load v5 renderer checkpoint from Modal volume: `/results/asym_v5_lagrangian_fixed/renderer_best.pt`
2. Load upstream scorers (SegNet + PoseNet) + GT video
3. Extract masks via SegNet (same as auth_eval)
4. Generate 1200 frames via renderer (same as auth_eval)
5. Extract GT pose targets: for each pair (2k, 2k+1), run PoseNet on GT → (600, 6)
6. Run `coupled_trajectory_optimize(init_frames=renderer_frames, expected_pose=gt_targets, ...)`
   - num_steps=500, lr=0.005 (lower for fine-tuning)
   - seg_weight=100, pose_weight=10, compress_weight=0.5
   - init_frames = renderer output (N, 384, 512, 3)
7. Upsample TTO frames to camera resolution (874x1164)
8. Write frames.raw, run upstream evaluate.py
9. Report auth score

### Implementation Options
A. **Extend auth_eval function** in modal_asymmetric_warp_deploy.py — add TTO step between frame generation and scoring
B. **New Modal function** specifically for TTO — cleaner but more code
C. **Standalone script** that runs on any CUDA machine (Modal, bat00, 3090) — most portable

Option C is best: write `experiments/renderer_tto.py` that works on any machine with the renderer checkpoint + upstream scorer.

### Memory Considerations (T4 16GB VRAM)
- Renderer: ~1MB VRAM
- Scorers: SegNet ~10MB, PoseNet ~5MB
- 1200 frames at 384x512x3 with requires_grad: 1200 * 384 * 512 * 3 * 4 = 2.8GB
- Autograd graph for SegNet+PoseNet forward on 1200 frames: potentially 10-20GB
- **MUST batch the TTO**: optimize chunks of ~100-200 frames at a time, not all 1200
- The coupled optimizer currently operates on ALL frames simultaneously — needs batching

### Critical Design Decision
The current coupled_trajectory_optimize treats ALL N frames as a single optimization variable.
For 1200 frames this is 2.8GB of requires_grad tensors + autograd graph = OOM on T4.
Need to either:
1. **Batch by pairs**: optimize 50 pairs (100 frames) at a time, 12 batches
2. **Sliding window**: optimize frames in overlapping windows (pairs share frames)
3. **Accept OOM and use A10G/3090**: 24GB VRAM handles 1200 frames

For v1: use option 1 (batch by pairs). Each batch of 50 pairs is independent for PoseNet
(non-overlapping pairs). SegNet is per-frame anyway. This naturally parallelizes.

### How to apply
Next session: implement experiments/renderer_tto.py, smoke test on MPS with 20 frames,
then launch on Modal with full 1200 frames.
