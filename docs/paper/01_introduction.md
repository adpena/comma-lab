# 1. Introduction

The comma.ai video compression challenge asks participants to compress a 60-second driving video (1200 frames, 1164x874, HEVC) such that two frozen neural networks --- a semantic segmentation model (SegNet) and an ego-motion estimator (PoseNet) --- produce outputs as close as possible to those computed on the original. The scoring formula is

$$S = 100 \cdot \bar{d}_{\text{seg}} + \sqrt{10 \cdot \bar{d}_{\text{pose}}} + 25 \cdot r$$

where $\bar{d}_{\text{seg}}$ is mean cross-entropy disagreement per frame, $\bar{d}_{\text{pose}}$ is mean L2 distance between 6-DOF ego-motion estimates on consecutive pairs, and $r = |\text{archive}| / |\text{uncompressed}|$ is the compression ratio. Lower is better. The three terms have very different scales and sensitivities: SegNet contributes linearly at 100x, PoseNet through a square root (diminishing returns), and rate linearly at 25x. These weights reflect comma.ai's actual product priorities --- SegNet has no backup sensor on the vehicle, while PoseNet's ego-motion estimates are redundant with IMU and wheel odometry.

The challenge is, at its core, an instance of **inverse steganalysis**. The contest organizer, Yassine Yousfi, was a PhD student of Jessica Fridrich at the Binghamton DDE Lab --- the world's leading steganalysis group. SegNet and PoseNet are not arbitrary neural networks; they are *detectors*, designed to identify whether an image has been modified. The EfficientNet backbone in SegNet shares architectural lineage with Yousfi's steganalysis classifiers [Yousfi et al. 2020, Yousfi and Fridrich 2022]. Our task is the inverse: produce modified images that these detectors cannot distinguish from originals. This is detector-informed embedding [Yousfi, Dworetzky, and Fridrich 2022], applied to video compression rather than message hiding.

This framing is not metaphorical. Fridrich's constrained optimization framework --- minimize the embedding payload subject to a detectability constraint --- maps directly to the scoring formula: minimize rate subject to scorer distortion constraints. The techniques developed over two decades of steganalysis research (UNIWARD's texture-weighted distortion [Holub and Fridrich 2014], the square root law for error spreading [Ker, Filler, and Fridrich 2008], HUGO's Markov chain statistics [Filler, Judas, and Fridrich 2010]) become concrete loss terms in our training pipeline. The contest IS steganalysis, and the scorers ARE detectors.

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

1. An asymmetric warp architecture for scorer-aware video generation that achieves state-of-the-art on the comma.ai challenge (0.43 vs. the next best compliant submission at 0.60), built on CLADE per-class normalization, dilated residual blocks, and Fridrich inverse steganalysis loss terms.

2. The discovery that PoseNet's output Jacobian has effective rank 1.008 --- a single degree of freedom (radial zoom from the focus of expansion) captures 99.8% of pose variance. This reduces the motion model from 50K parameters to 600 learned scalars (1.2 KB), validated at 10.6x PoseNet improvement.

3. A test-time optimization procedure with coupled trajectory loss, where PoseNet gradients flow through consecutive frame pairs jointly.

4. The identification and resolution of a gradient obstruction bug in the upstream scorer pipeline, with a generalizable 1ms runtime validation check. This bug was present across 15 files in 3 distinct patterns, and its discovery prompted a systematic audit whose lessons are documented as measurement methodology.

5. An analysis of the scoring formula's geometry --- the sqrt concavity on PoseNet, SegNet's odd-frame blindness, and their implications for optimization strategy.

6. A documented account of the research process, including 25+ negative results and a chain of measurement bugs that invalidated weeks of scores, as a case study in both the promise and the pitfalls of human-AI collaborative engineering.
