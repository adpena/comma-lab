---
name: Adaptive Self-Improving Hyperparameters (Design Goal)
description: Derive hyperparameters as functions of current training state so they auto-tune — no more arbitrary values
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Idea

Instead of static hyperparameters, derive them as functions of the current operating point:

1. **segnet_weight(pose, seg)** = d(score)/d(seg) / d(score)/d(pose)
   = 100 / (5/sqrt(10*pose))
   = 20 * sqrt(10*pose)
   At pose=0.01: segnet_weight = 6.3
   At pose=0.001: segnet_weight = 2.0
   Auto-rebalances as PoseNet improves — naturally reduces SegNet pressure when PoseNet needs help.

2. **boundary_weight(seg, boundary_frac)** = should equalize gradient contribution
   = (1 - boundary_frac) / boundary_frac * (seg_target / current_seg)
   Increases as seg approaches target, concentrating more on boundaries.

3. **temperature(KL_div)** = anneal based on actual KL divergence, not epoch count
   When KL is high (distributions far apart): keep T high for smooth gradients
   When KL is low (distributions nearly matched): drop T for argmax pressure

## Implementation

At each eval epoch, compute the current (pose, seg) from eval_scorer_loss.
Then update the training hyperparameters for the next eval_every epochs.
The JSONL telemetry log provides the data needed.

This turns the training loop into a self-improving system where the loss function
adapts to its own progress. No manual tuning needed.

**Why:** Every "council recommendation" for a specific number (bw=150, bw=5, sw=100, sw=30)
has been wrong because the optimal value depends on the current operating point, which
changes during training. Adaptive weights solve this structurally.

**Key insight from the regression:** bw=150 was optimal at the START of training
(when SegNet needed heavy emphasis) but catastrophic at ep 194 (when PoseNet had
regressed and needed protection). An adaptive weight would have noticed pose regressing
and automatically shifted weight back.
