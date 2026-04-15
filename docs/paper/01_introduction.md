# 1. Introduction

The comma.ai video compression challenge asks participants to compress a 60-second driving video (1200 frames, 1164x874, HEVC) such that two frozen neural networks --- a semantic segmentation model (SegNet) and an ego-motion estimator (PoseNet) --- produce outputs as close as possible to those computed on the original. The scoring formula is

$$S = 100 \cdot \bar{d}_{\text{seg}} + \sqrt{10 \cdot \bar{d}_{\text{pose}}} + 25 \cdot r$$

where $\bar{d}_{\text{seg}}$ is mean cross-entropy disagreement per frame, $\bar{d}_{\text{pose}}$ is mean L2 distance between 6-DOF ego-motion estimates on consecutive pairs, and $r = |\text{archive}| / |\text{uncompressed}|$ is the compression ratio. Lower is better. The three terms have very different scales and sensitivities: SegNet contributes linearly at 100x, PoseNet through a square root (diminishing returns), and rate linearly at 25x. These weights reflect comma.ai's actual product priorities --- SegNet has no backup sensor on the vehicle, while PoseNet's ego-motion estimates are redundant with IMU and wheel odometry.

The challenge permits any approach: standard codecs, neural compression, generative models. Compressed artifacts (including any neural network weights used at decompression time) must fit inside a single archive whose size counts toward the rate term. External libraries are permitted, but large learned artifacts must be included. Scorer networks may be used during compression but not loaded from outside the archive during decompression.

## Our approach

We developed an asymmetric warp renderer --- a 287K-parameter conditional generative model that produces frame pairs from compact semantic masks. The renderer is trained end-to-end against the frozen scorers using Lagrangian annealing with hard constraints on both SegNet and PoseNet distortion, following a constrained optimization formulation inspired by Fridrich's steganalysis framework [Fridrich 2009]. At test time, we apply coupled trajectory optimization (TTO), a form of test-time adaptation that directly optimizes the generated frames through the scorer networks.

The final score is **0.43**, down from 1.97 at the start of the project. The trajectory:

| Stage | Score | Approach |
|-------|-------|----------|
| Baseline (H.265 re-encode) | 1.97 | CRF 28, no postfilter |
| CPU postfilter (CNN) | 1.33 | Dilated h=64, trained against scorers |
| Renderer (asymmetric warp) | 0.87 | Lagrangian annealing, FP4 quantized |
| + TTO (blind PoseNet) | 0.70 | 500 steps, SegNet spillover only |
| + TTO (gradient fix) | 0.43 | Differentiable BT.601 YUV conversion |

The single largest improvement --- 0.70 to 0.43, a 38.6% reduction --- came from discovering and fixing a gradient obstruction bug. The upstream scorer's RGB-to-YUV conversion was decorated with `@torch.no_grad`, silently zeroing all PoseNet gradients during test-time optimization. Every TTO experiment before the fix was optimizing PoseNet blindly. The optimizer moved pixels that happened to reduce PoseNet loss through SegNet gradient spillover, but it could not see PoseNet's actual loss surface.

This bug was invisible to standard testing. PoseNet loss changed during optimization (because SegNet gradients moved pixels that incidentally affected PoseNet output), so nothing appeared wrong. It was caught only through adversarial review --- tracing the gradient computation graph by hand when the council demanded an explanation for why 50 steps of gradient descent made PoseNet worse, not better.

The discovery illustrates a broader point about optimization through frozen networks: the gradient flow must be validated end-to-end before trusting any result. A loss that changes is not proof that gradients are flowing correctly.

## Contributions

1. An asymmetric warp architecture for scorer-aware video generation that achieves state-of-the-art on the comma.ai challenge (0.43 vs. the next best compliant submission at 0.60).

2. A test-time optimization procedure with coupled trajectory loss, where PoseNet gradients flow through consecutive frame pairs jointly.

3. The identification and resolution of a gradient obstruction bug in the upstream scorer pipeline, with a generalizable 1ms runtime validation check.

4. An analysis of the scoring formula's geometry --- the sqrt concavity on PoseNet, SegNet's odd-frame blindness, and their implications for optimization strategy.

5. A documented account of the research process, including 25+ negative results, as a case study in human-AI collaborative engineering.
