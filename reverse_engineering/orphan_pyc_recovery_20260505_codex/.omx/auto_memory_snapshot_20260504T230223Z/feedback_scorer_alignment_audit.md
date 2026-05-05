---
name: Scorer Alignment Audit (openpilot policy lens)
description: Loss functions and augmentations must align with what PoseNet and SegNet actually measure in openpilot's downstream policy model — not generic perceptual quality.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
CONTEST SCORERS ARE NOT GENERIC. They're openpilot policy components:
- **SegNet (EfficientNet-B2 U-Net, 5 classes)** = perception. Output → openpilot's path planner. CLASS BOUNDARIES (where road meets curb, lane line edges) are what the policy uses to decide where the car can drive. Errors INSIDE a class region are invisible; errors AT boundaries shift the path.
- **PoseNet (FastViT-T12 with YUV6 input, 12-dim → first 6 used)** = ego-motion estimation. Output → openpilot's lateral/longitudinal control. TEMPORAL DERIVATIVES across consecutive frames are what it reads. Persistent low-amplitude noise is invisible; consistent frame-to-frame drift is read as motion.

WHAT WE'VE WIRED (R-fridrich-wire 2026-04-25):
- ✅ UNIWARD texture loss (hide errors in textured regions)
- ✅ L∞ penalty (spread errors, square root law)
- ✅ Markov gradient continuity
- ✅ KL distillation T=2.0 on SegNet logits (Quantizr's recipe)
- ✅ Mask augmentation (CRF=63 noise during training)

GAPS WHERE WIRING IS SUBOPTIMAL FOR OUR SCORERS:

1. **Loss space mismatch.** All Fridrich losses operate in RGB. PoseNet input is YUV6 (4 luma + 2 chroma). Errors that vanish in RGB may be visible in YUV6. Should compute reconstruction in YUV6 space when targeting PoseNet (use `frame_utils.rgb_to_yuv6` from upstream).

2. **No frame-pair temporal-consistency loss.** PoseNet measures motion BETWEEN frame_t and frame_t+1. We compute losses per-frame independently. The right loss is: L(motion(rendered_pair)) - L(motion(gt_pair)) where motion is the warp-residual between consecutive frames. Without this, we train for spatial fidelity and PoseNet measures temporal fidelity — orthogonal axes.

3. **Boundary-weighted L∞ instead of uniform L∞.** L∞ at SegNet class boundaries costs 100x more than L∞ in uniform regions (only argmax flips matter). Should weight L∞ by SegNet boundary mask.

4. **UNIWARD wavelet cost is generic CNN steganography.** It assumes a CNN trained on natural images with standard receptive fields. PoseNet has FastViT (transformer-style attention) — its blind spots may be at different frequency bands than UNIWARD predicts. Need a Yousfi/Fridrich-class study against the ACTUAL PoseNet to recompute the cost map.

5. **Mask augmentation uses fixed CRF=63 only.** PoseNet sees mask noise from CRF=50, 56, 60, 63 depending on what archive ships. Should mix multiple CRF variants per training so the renderer is robust across the rate-distortion frontier.

WHAT THIS MEANS FOR FUTURE WORK:
- A "renderer trained with Fridrich losses" is BETTER than uniform MSE but still suboptimal for our scorers
- The right losses are SCORER-DERIVED — gradient through the actual SegNet/PoseNet computes the optimal direction
- That's what `scorer_loss` and `kl_distill_scorer_loss` already do; the auxiliary Fridrich losses are PRIORS that may HELP or HURT depending on whether they align with the scorer
- Empirical test required: train with vs without each Fridrich loss, measure auth score delta. Don't assume.
