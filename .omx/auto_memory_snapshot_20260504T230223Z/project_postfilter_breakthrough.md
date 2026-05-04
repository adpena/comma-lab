---
name: Learned Post-Filter Breakthrough
description: Tiny CNN trained against scorer gradients reduces PoseNet by 71% on proxy. The most innovative technique in the lab.
type: project
---

## The breakthrough (2026-04-07)

Trained a tiny CNN (3,203 params, 7.5KB int8) directly against the scorer's loss function via backprop through PoseNet+SegNet. Applied as a post-filter after bicubic upscale in inflate.py.

**Architecture:**
- 3 conv layers: 3 → 16 → 16 → 3 channels, 3×3 kernels
- ReLU activations, residual connection
- Total: 3,203 parameters, 7.5KB quantized to int8

**Training:**
- Loss: actual scorer formula `100*segnet + sqrt(10*posenet)`
- Differentiable SegNet surrogate (softmax overlap)
- Differentiable PoseNet via patched yuv6 conversion
- Adam + cosine annealing, 100 epochs
- Trained on the single test video (overfit is fine — only 1 video)

**Proxy results (without rate term):**
| Metric | Baseline | With post-filter | Delta |
|--------|----------|------------------|-------|
| Score | 2.4653 | 1.6430 | **-0.8223** |
| PoseNet | 0.367512 | 0.106774 | **-71%** |
| SegNet | 0.005483 | 0.006097 | +11% |

**Why this is the breakthrough:**
- Every previous technique either improved one metric while hurting another, OR was zero-sum
- The post-filter is *supervised by the actual scorer* — it learns what the scorer actually wants
- Cost: only 7.5KB of model weights shipped in archive (~0.0002 rate penalty)
- Net expected improvement: 0.05-0.20 on real scorer

**Files:**
- Training: `experiments/train_postfilter.py`
- Weights: `experiments/postfilter_weights/postfilter_int8.pt` (7.5KB)
- Inflate: `submissions/robust_current/inflate_postfilter.py`
- Config: `PYTHON_INFLATE=postfilter` in config.env

**Status as of 2026-04-07:** First full scorer run in progress. If it confirms even half the proxy improvement, we beat 1.95 and become the leader.

**No one else on the leaderboard has deployed learned post-filters.** This is our differentiation vector.
