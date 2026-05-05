---
name: Missing techniques from deep memory audit — 9 researched but unwired
description: Per-class SegNet weights (code exists, not in profiles), scorer-space gen (constrained_gen.py, never scored), LoRA TTO, YUV null space, SWA, Ghost modules, knowledge distillation, multi-pass Fridrich, constrained gen without renderer.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Techniques with code that exists but ISN'T wired:

1. **Per-class SegNet weighting**: compute_segnet_constraint_loss already accepts
   per_class_weights parameter. Lane markings (1.2% of pixels) need 15x weight.
   compute_segnet_class_weights() exists. NOT in WILDE or SHIRAZ profiles.
   FIX: add per_class_weights=[1.0, 15.0, 1.5, 3.0, 7.0] to the training loop.

2. **Scorer-space generation** (GPU Eureka #1): constrained_gen.py implements
   generating frames directly in scorer-input space then inverting to RGB.
   Artifacts from inversion land in preprocessing null-space. NEVER SCORED.
   Projected 0.135. This is the theoretical endgame.

3. **Constrained gen without renderer** (GPU Eureka #4): Start from noise, apply
   scorer constraints via gradient descent. Frames emerge from constraints.
   Archive = masks(239B) + pose targets(7KB) + noise seed(64B) = 8KB.
   constrained_gen.py + variational_gen.py + scorer_manifold.py exist.

## Techniques researched but NOT implemented:

4. **LoRA TTO**: Store base_weights + small rank-4 delta per video.
   Massive rate savings. Never tried.

5. **YUV null space exploitation**: PoseNet's rgb_to_yuv6 creates a 6-channel
   representation with 4:2:0 chroma subsampling. Sub-2x2 chroma patterns are
   invisible to PoseNet. We can modify chroma channels freely.

6. **SWA (Stochastic Weight Averaging)**: Average weights across last N checkpoints.
   Free quality improvement, no extra training cost. torch.optim.swa_utils.

7. **Multi-pass Fridrich fine-tuning**: After QAT, do a dedicated pass with ONLY
   Fridrich losses (texture+linf+markov) at very low LR. Pushes errors further
   into scorer null space without disturbing the learned quality.

8. **Knowledge distillation**: Train a large model (181K params), distill into a
   smaller model (54K params) that fits in a smaller archive. Rate savings.

9. **Ghost modules** (GhostNet Han et al. 2020): Generate feature maps from cheap
   linear operations on existing features. Further param reduction beyond DSConv.
